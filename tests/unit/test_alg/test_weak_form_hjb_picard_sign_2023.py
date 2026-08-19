"""Issue #2023: the weak-form HJB Picard branch applied the Hamiltonian with the wrong sign.

`WeakFormHJBSolver.solve_hjb_system` has two branches. The Newton one assembles a residual; the
Picard one -- **the default**, `use_newton=False` -- moves `H` to the right-hand side, and moving it
flips its sign. It read `rhs += self._M @ H_values` from #1131 (675e0049, 2026-05-29) until this
change, so `HJBFEMSolver` and `MeshlessGalerkinHJBSolver` solved `-u_t - H - D Lap(u) = 0` on the
path a caller gets by default.

WHY IT SURVIVED
---------------
Nothing ran both branches on the same problem and compared. Of the thirteen test files naming these
solvers only three mention `use_newton` at all, and none against an oracle. Two branches of one
method silently solving different equations is invisible to any test that exercises one branch.

THE FIXTURE
-----------
`H = c` constant: a quadratic control cost whose coupling returns a constant, with `m == 1`. With
`u_T = 0` and no-flux walls the solution stays spatially constant, so `p == 0` (killing the kinetic
half, leaving `H == c` exactly) and `Lap u == 0`. The equation collapses to `u'(t) = c`, giving

    u(t) = c (t - T),   u(0) = -c T

with NO discretization error in the measured quantity -- the exact solution is in the FE space. That
is what makes a sign inversion readable as `-1x` rather than as a convergence question.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import Mesh1D, TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

_T, _NT, _SIGMA, _NE = 0.2, 20, 0.3, 40
_CONSTANTS = (1.0, 2.0, -3.0)


def _hamiltonian(c: float):
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m, _c=c: np.full_like(np.asarray(m, dtype=float), _c),
        coupling_dm=lambda m: np.zeros_like(np.asarray(m, dtype=float)),
    )


def _components(c: float):
    return MFGComponents(
        m_initial=lambda x: np.ones_like(np.asarray(x, dtype=float)),
        u_terminal=lambda x: np.asarray(x, dtype=float) * 0.0,
        hamiltonian=_hamiltonian(c),
    )


def _mesh_problem(c: float):
    mesh = Mesh1D(bounds=(0.0, 1.0), num_elements=_NE)
    mesh.generate_mesh()
    mesh.boundary_conditions = no_flux_bc(dimension=1)
    return MFGProblem(geometry=mesh, T=_T, Nt=_NT, sigma=_SIGMA, coupling_coefficient=0.0, components=_components(c))


def _grid_problem(c: float):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[_NE + 1], boundary_conditions=no_flux_bc(dimension=1))
    return MFGProblem(geometry=grid, T=_T, Nt=_NT, sigma=_SIGMA, coupling_coefficient=0.0, components=_components(c))


@pytest.mark.parametrize("c", _CONSTANTS)
def test_the_picard_branch_agrees_with_the_analytic_constant_hamiltonian(c):
    """u(0) = -c*T, and the wrong sign returns +c*T. Exactly -1x, so the assertion can be tight."""
    pytest.importorskip("skfem", reason="scikit-fem required for the weak-form FEM solvers")
    from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver

    solver = HJBFEMSolver(_mesh_problem(c), order=1)
    n = len(solver._disc.dof_coordinates)
    u = solver.solve_hjb_system(
        M_density=np.ones((_NT + 1, n)),
        U_terminal=np.zeros(n),
        U_coupling_prev=np.zeros((_NT + 1, n)),
        use_newton=False,
    )
    u0 = np.asarray(u)[0]

    # The fixture only means what it claims if the solution really is spatially constant: p == 0 is
    # what makes H == c rather than c + |grad u|^2/2.
    assert float(np.ptp(u0)) < 1e-12, f"u(0) is not spatially constant (spread {float(np.ptp(u0)):.3e})"
    assert float(np.mean(u0)) == pytest.approx(-c * _T, abs=1e-10), (
        f"picard u(0) = {float(np.mean(u0)):+.6f}, analytic {-c * _T:+.6f}. A value of {c * _T:+.6f} "
        f"is the #2023 sign inversion: `rhs += self._M @ H_values` where the equation "
        f"-u_t + H - D*Lap(u) = 0 requires `-=`."
    )


@pytest.mark.parametrize("c", _CONSTANTS)
def test_the_two_branches_of_one_method_solve_the_same_equation(c):
    """Newton vs Picard on identical input. This is the comparison that was never made.

    Asserting each against the analytic value separately would also catch #2023, but this pins the
    stronger property: whatever the convention is, one method must not have two of them.
    """
    pytest.importorskip("skfem", reason="scikit-fem required for the weak-form FEM solvers")
    from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver

    solver = HJBFEMSolver(_mesh_problem(c), order=1)
    n = len(solver._disc.dof_coordinates)
    kw = {
        "M_density": np.ones((_NT + 1, n)),
        "U_terminal": np.zeros(n),
        "U_coupling_prev": np.zeros((_NT + 1, n)),
    }
    picard = np.asarray(solver.solve_hjb_system(**kw, use_newton=False))[0]
    newton = np.asarray(solver.solve_hjb_system(**kw, use_newton=True))[0]
    gap = float(np.abs(picard - newton).max())
    assert gap < 1e-10, (
        f"the Picard and Newton branches disagree by {gap:.3e} on a problem whose exact solution "
        f"lies in the FE space. They are two discretizations of one equation and must agree here."
    )


@pytest.mark.parametrize("c", _CONSTANTS)
def test_the_weak_form_family_agrees_with_the_reference_fdm_solver(c):
    """Cross-implementation, because a shared convention error inside one family would pass the
    two tests above. `HJBFDMSolver` is the suite's reference HJB discretization and does not share
    this assembly code."""
    pytest.importorskip("skfem", reason="scikit-fem required for the weak-form FEM solvers")
    from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver
    from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
    from mfgarchon.alg.numerical.meshless_galerkin.hjb_solver import MeshlessGalerkinHJBSolver

    n = _NE + 1
    fdm = np.asarray(
        HJBFDMSolver(_grid_problem(c)).solve_hjb_system(
            M_density=np.ones((_NT + 1, n)), U_terminal=np.zeros(n), U_coupling_prev=np.zeros((_NT + 1, n))
        )
    )[0]

    fem_solver = HJBFEMSolver(_mesh_problem(c), order=1)
    n_fem = len(fem_solver._disc.dof_coordinates)
    fem = np.asarray(
        fem_solver.solve_hjb_system(
            M_density=np.ones((_NT + 1, n_fem)),
            U_terminal=np.zeros(n_fem),
            U_coupling_prev=np.zeros((_NT + 1, n_fem)),
            use_newton=False,
        )
    )[0]

    meshless_solver = MeshlessGalerkinHJBSolver(
        _grid_problem(c), np.linspace(0.0, 1.0, n).reshape(-1, 1), delta=2.6 / np.sqrt(n)
    )
    meshless = np.asarray(meshless_solver.solve_hjb_system(np.ones((_NT + 1, n)), np.zeros(n), np.zeros((_NT + 1, n))))[
        0
    ]

    # All three are spatially constant here, so comparing means is comparing the fields.
    #
    # 1e-6 is set from the reference's own accuracy, not tuned until green: FDM lands 1.48e-07 off
    # the analytic value at c = 1 (its no-flux treatment is not exact even on a spatially constant
    # field), while the weak-form solvers are exact to ~3e-16. The defect this discriminates is a
    # factor of -1, i.e. 0.4 in absolute terms at c = 1 -- six orders of magnitude above the bound.
    for label, field in (("weak-form FEM", fem), ("meshless Galerkin", meshless)):
        assert float(np.mean(field)) == pytest.approx(float(np.mean(fdm)), abs=1e-6), (
            f"{label} u(0) = {float(np.mean(field)):+.9f} against reference FDM "
            f"{float(np.mean(fdm)):+.9f} (analytic {-c * _T:+.9f})"
        )

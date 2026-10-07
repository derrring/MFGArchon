"""The meshless Galerkin pair declares the BCs it honours, so the gate refuses the rest (#2512 S4).

Both solvers declared nothing, so `_validate_bc_support` returned early and every BC constructed:
measured at `7eb53144`, the HJB solved `neumann_bc(value=0.7)` identically to 0.0, the FP solved
`neumann_bc(value=1.0)` identically to no-flux, and PERIODIC / ROBIN raised only once the solve reached
a condensation hook.

What each honours: the natural condition is du/dn = 0 on the HJB and zero flux on the FP (no-flux,
homogeneous Neumann, reflecting); Nitsche imposes a Dirichlet value on the HJB, and an absorbing wall
(m = 0) on the FP -- the exit reading of a shared Dirichlet (#2512, convention row 5).
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver
from mfgarchon.alg.numerical.meshless_galerkin.hjb_solver import MeshlessGalerkinHJBSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    dirichlet_bc,
    neumann_bc,
    no_flux_bc,
    periodic_bc,
)

NX = 11
XS = np.linspace(0.0, 1.0, NX)[:, None]


def _problem(bc) -> MFGProblem:
    comps = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        m_initial=lambda x: np.ones_like(np.asarray(x, dtype=float)),
        u_terminal=lambda x: np.cos(np.pi * np.asarray(x, dtype=float)),
    )
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[NX], boundary_conditions=bc)
    return MFGProblem(geometry=grid, T=0.5, Nt=4, volatility=0.3, components=comps)


def _robin():
    return BoundaryConditions(dimension=1, segments=[BCSegment(name="r", bc_type=BCType.ROBIN, alpha=1.0, beta=1.0)])


def _reflecting():
    return BoundaryConditions(dimension=1, segments=[BCSegment(name="r", bc_type=BCType.REFLECTING)])


@pytest.mark.parametrize("cls", [MeshlessGalerkinHJBSolver, MeshlessGalerkinFPSolver])
@pytest.mark.parametrize(
    "bc_factory",
    [
        lambda: neumann_bc(dimension=1, value=0.7),
        lambda: periodic_bc(dimension=1),
        _robin,
    ],
    ids=["neumann_nonzero", "periodic", "robin"],
)
def test_an_unhonoured_bc_is_refused_at_construction(cls, bc_factory):
    with pytest.raises(NotImplementedError, match="MeshlessGalerkin"):
        cls(_problem(bc_factory()), XS, delta=0.35)


def test_the_fp_takes_a_shared_dirichlet_value_as_an_exit():
    """The pair reads only the shared BC, where DIRICHLET(g) is an exit: the HJB's u = g, an absorbing
    wall here (#2512, convention row 5). So a nonzero value constructs."""
    MeshlessGalerkinFPSolver(_problem(dirichlet_bc(dimension=1, value=0.5)), XS, delta=0.35)


@pytest.mark.parametrize("cls", [MeshlessGalerkinHJBSolver, MeshlessGalerkinFPSolver])
@pytest.mark.parametrize(
    "bc_factory",
    [
        lambda: no_flux_bc(dimension=1),
        lambda: neumann_bc(dimension=1, value=0.0),
        _reflecting,
        lambda: dirichlet_bc(dimension=1, value=0.0),
    ],
    ids=["no_flux", "neumann_zero", "reflecting", "dirichlet_zero"],
)
def test_an_honoured_bc_constructs(cls, bc_factory):
    cls(_problem(bc_factory()), XS, delta=0.35)


def test_the_hjb_still_imposes_a_nonzero_dirichlet_value():
    """Nitsche carries the value on the HJB, so it is declared without a value refusal.

    Nitsche imposes it weakly: 0.697 / 0.712 on this fixture (11 points, delta 0.35). The band separates
    "the value arrives" from "the value is dropped", which would leave the wall near 0 or near the
    terminal data, +-1.
    """
    solver = MeshlessGalerkinHJBSolver(_problem(dirichlet_bc(dimension=1, value=0.7)), XS, delta=0.35)
    n = solver.n_dof
    U = np.asarray(
        solver.solve_hjb_system(
            M_density=np.ones((5, n)), U_terminal=np.cos(np.pi * XS[:, 0]), U_coupling_prev=np.zeros((5, n))
        )
    )
    np.testing.assert_allclose(U[0, [0, -1]], 0.7, atol=5e-2)


@pytest.mark.parametrize("cls", [MeshlessGalerkinHJBSolver, MeshlessGalerkinFPSolver])
def test_a_dirichlet_default_bc_is_refused_not_solved_as_the_natural_condition(cls):
    """Nitsche reads only segments: a default-DIRICHLET right wall kept all its mass on the FP. The
    Nitsche terms are built at construction, so that is where it is refused."""
    bc = BoundaryConditions(
        dimension=1,
        segments=[BCSegment(name="l", bc_type=BCType.NO_FLUX, boundary="x_min")],
        default_bc=BCType.DIRICHLET,
        default_value=0.0,
    )
    with pytest.raises(NotImplementedError, match="default_bc"):
        cls(_problem(bc), XS, delta=0.35)


@pytest.mark.parametrize("cls", [MeshlessGalerkinHJBSolver, MeshlessGalerkinFPSolver])
def test_a_region_named_segment_leaves_its_other_faces_to_the_default(cls):
    """A region-named segment governs only its own face; the default still governs the rest."""
    bc = BoundaryConditions(
        dimension=1,
        segments=[BCSegment(name="r", bc_type=BCType.NO_FLUX, region_name="x_max")],
        default_bc=BCType.DIRICHLET,
        default_value=0.0,
    )
    with pytest.raises(NotImplementedError, match="default_bc"):
        cls(_problem(bc), XS, delta=0.35)


@pytest.mark.parametrize("cls", [MeshlessGalerkinHJBSolver, MeshlessGalerkinFPSolver])
def test_face_aliases_cover_their_faces(cls):
    """ "left" and "right" are x_min and x_max (#1939), so no face is left to the default."""
    bc = BoundaryConditions(
        dimension=1,
        segments=[
            BCSegment(name="l", bc_type=BCType.NO_FLUX, boundary="left"),
            BCSegment(name="r", bc_type=BCType.NO_FLUX, boundary="right"),
        ],
        default_bc=BCType.DIRICHLET,
        default_value=0.0,
    )
    cls(_problem(bc), XS, delta=0.35)


def test_a_coupled_exit_cost_runs_through_the_factory():
    """One shared dirichlet_bc(0.5) on the meshless pair built by create_paired_solvers: the exit cost
    reaches the HJB (u = g at both walls, to Nitsche's weak accuracy) and the FP absorbs (mass falls,
    non-increasing). A value refusal on the FP -- which this pair once had -- made this configuration
    impossible, since the pair reads one BC and the FP takes no BC of its own.

    M is not asserted to vanish at the wall: Nitsche imposes the absorbing wall weakly, M = 0.125 at T
    on 21 points.
    """
    from mfgarchon import Conditions, Model
    from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
    from mfgarchon.factory.scheme_factory import create_paired_solvers
    from mfgarchon.types import NumericalScheme

    g, nx = 0.5, 21
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=dirichlet_bc(dimension=1, value=g)
    )
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m),
            volatility=0.4,
        ),
        domain=grid,
        conditions=Conditions(u_terminal=lambda x: g, m_initial=lambda x: 1.0, T=0.5),
        Nt=10,
    )
    hjb, fp = create_paired_solvers(
        problem,
        NumericalScheme.MESHLESS_GALERKIN,
        hjb_config={"collocation_points": np.linspace(0.0, 1.0, nx)[:, None], "delta": 0.18},
        fp_config={},
    )
    result = FixedPointIterator(problem, hjb_solver=hjb, fp_solver=fp).solve(max_iterations=3, tolerance=1e-3)
    U, M = np.asarray(result.U), np.asarray(result.M)
    np.testing.assert_allclose(U[:, [0, -1]], g, atol=1e-3)
    mass = np.array([float(grid.integrate(M[k])) for k in range(M.shape[0])])
    assert np.all(np.diff(mass) <= 1e-12), f"mass rose: {mass}"
    assert mass[-1] < 0.5 * mass[0], f"the exit did not absorb: {mass[0]:.3f} -> {mass[-1]:.3f}"

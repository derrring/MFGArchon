"""Every FP solver names its initial density `M_initial`, as the abstract base does (#2377).

The base declared `m_initial_condition`, and so did FPGFDMSolver; WeakFormFPSolver said `m_initial`,
inherited by FPFEMSolver and MeshlessGalerkinFPSolver; the other six said `M_initial`, the name the
v0.17.0 rename chose (#1356 removed the old names at v0.20) and FPNetworkSolver still aliases.
Measured at `31de2179` over the ten concrete library solvers: `m_initial_condition=` bound on two
and raised TypeError on eight, and `M_initial=` raised on four. Positional calls worked everywhere,
which is why it survived.

`test_every_fp_solver_names_its_first_parameter_M_initial` pins the name on every concrete library
subclass of `BaseFPSolver`, inherited or not, and requires all nine to be reached; renaming any one of
them back fails it. The weak form's old spelling stays accepted for the deprecation window (AGENTS.md,
clause 4), and the policy's clause 2 equivalence tests below hold it to the new one. FP-GFDM's
`m_initial_condition` row went with the solver, withdrawn in #2583. Its refusing stub's `solve_fp_system`
keeps the new first parameter, `M_initial`, and takes no `m_initial_condition`.
"""

from __future__ import annotations

import inspect
import warnings

import pytest

import numpy as np

# Imported so that every BaseFPSolver subclass exists before the walk below looks for it. Neither the
# FEM nor the meshless-Galerkin module needs scikit-fem at import time.
import mfgarchon.alg.numerical.fem.fp_fem_solver
import mfgarchon.alg.numerical.fp_solvers
import mfgarchon.alg.numerical.meshless_galerkin.fp_solver
import mfgarchon.alg.numerical.network_solvers
import mfgarchon.alg.numerical.weak_form_fp_solver  # noqa: F401
from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.core.model import Conditions, Model
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

_N = 11

# The walk below must reach at least these, or it is not looking at the population it claims to.
_KNOWN = {
    "FPFDMSolver",
    "FPFVMSolver",
    "FPGFDMSolver",
    "FPParticleSolver",
    "FPSLSolver",
    "FPNetworkSolver",
    "WeakFormFPSolver",
    "FPFEMSolver",
    "MeshlessGalerkinFPSolver",
}


def _concrete_subclasses(cls):
    for sub in cls.__subclasses__():
        if not inspect.isabstract(sub):
            yield sub
        yield from _concrete_subclasses(sub)


def test_every_fp_solver_names_its_first_parameter_M_initial():
    base_first = list(inspect.signature(BaseFPSolver.solve_fp_system).parameters)[1]
    assert base_first == "M_initial", f"the abstract base names it {base_first!r}"

    # Library solvers only: a test double elsewhere in the suite subclasses the base without being one.
    found = {sub.__name__: sub for sub in _concrete_subclasses(BaseFPSolver) if sub.__module__.startswith("mfgarchon.")}
    missing = _KNOWN - set(found)
    assert not missing, f"the subclass walk did not reach {sorted(missing)}"

    wrong = {
        name: list(inspect.signature(cls.solve_fp_system).parameters)[1]
        for name, cls in found.items()
        if list(inspect.signature(cls.solve_fp_system).parameters)[1] != "M_initial"
    }
    assert not wrong, f"first parameter of solve_fp_system is not M_initial on {wrong}"


def _problem():
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[_N], boundary_conditions=no_flux_bc(dimension=1))
    bump = lambda x: np.exp(-10 * (np.asarray(x, dtype=float) - 0.5) ** 2)  # noqa: E731
    mass = grid.integrate(bump(grid.get_spatial_grid()[:, 0]))
    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: np.asarray(m) * 0.0,
                coupling_dm=lambda m: np.asarray(m) * 0.0,
            )
        ),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: bump(x) / mass, u_terminal=lambda x: 0.0, T=0.2),
        Nt=5,
    )


def _weak_form():
    pytest.importorskip("skfem", reason="scikit-fem required for the weak-form FEM solver")
    from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver
    from mfgarchon.geometry import Mesh1D

    mesh = Mesh1D(bounds=(0.0, 1.0), num_elements=_N - 1)
    mesh.generate_mesh()
    mesh.boundary_conditions = no_flux_bc(dimension=1)
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: np.asarray(m) * 0.0,
                coupling_dm=lambda m: np.asarray(m) * 0.0,
            )
        ),
        domain=mesh,
        # (N-1)/N integrates to 1 on the mesh's uniform-cell measure (N nodes, cells of 1/(N-1)).
        conditions=Conditions(
            m_initial=lambda x: np.full_like(np.asarray(x, dtype=float), (_N - 1) / _N), u_terminal=lambda x: 0.0, T=0.2
        ),
        Nt=5,
    )
    solver = FPFEMSolver(problem, order=1)
    x = solver._disc.dof_coordinates[:, 0]
    return solver, 1.0 + 0.5 * np.sin(np.pi * x), {"potential_field": None}


@pytest.mark.parametrize(("build", "old"), [(_weak_form, "m_initial")])
def test_the_deprecated_keyword_warns_and_solves_the_same_problem(build, old):
    solver, m0, rest = build()
    new = np.asarray(solver.solve_fp_system(M_initial=m0.copy(), **rest))
    with pytest.warns(DeprecationWarning, match=old):
        via_old = np.asarray(solver.solve_fp_system(**{old: m0.copy()}, **rest))
    np.testing.assert_array_equal(via_old, new)


@pytest.mark.parametrize(("build", "old"), [(_weak_form, "m_initial")])
def test_both_spellings_at_once_are_refused(build, old):
    solver, m0, rest = build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError, match="M_initial"):
            solver.solve_fp_system(M_initial=m0.copy(), **{old: m0.copy()}, **rest)


@pytest.mark.parametrize("build", [_weak_form])
def test_no_initial_density_is_refused_by_name(build):
    solver, _, rest = build()
    with pytest.raises(ValueError, match="M_initial is required"):
        solver.solve_fp_system(**rest)

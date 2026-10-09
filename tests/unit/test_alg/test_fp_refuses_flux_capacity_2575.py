"""No FP solver honours a capacity-limited exit, so every one refuses `flux_capacity` (#2575).

`BCSegment.flux_capacity` was accepted on every route and ignored on every route. On #2531's fixture,
FP-FDM's outflow over the horizon was 0.081667 with and without a cap of 1e-6, and the particle solver's
was 0.071469 both ways, at `5e4b64e4`. The particle helper that reads the cap,
`_apply_boundary_conditions_with_flux_limits`, has no caller. The refusal sits in
`BaseFPSolver._validate_bc_support`, which every FP solver reaches on the BC it solves with.

Retirement: the capacity-limited exit is scheduled in this phase (user ruling, 2026-10-09). When it lands,
each refusal below becomes that capability's oracle, and this file goes.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

REFUSAL = r"carry a flux_capacity, a capacity-limited exit, which no FP solver implements yet"
CAP = 1e-6


def _exit(dimension: int, cap: float | None) -> BoundaryConditions:
    segments = [
        BCSegment(name="wall", bc_type=BCType.NO_FLUX, boundary="x_min"),
        BCSegment(name="exit", bc_type=BCType.DIRICHLET, boundary="x_max", value=0.0, flux_capacity=cap),
    ]
    if dimension == 2:
        segments += [
            BCSegment(name="bottom", bc_type=BCType.NO_FLUX, boundary="y_min"),
            BCSegment(name="top", bc_type=BCType.NO_FLUX, boundary="y_max"),
        ]
    return BoundaryConditions(dimension=dimension, segments=segments)


def _model() -> Model:
    return Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.4)


def _grid_problem(bc: BoundaryConditions) -> MFGProblem:
    """#2531's fixture: 21 points on [0, 1], T = 0.5, Nt = 10, with ``bc`` as the problem's shared BC."""
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=bc)
    x = np.linspace(0.0, 1.0, 21)
    bump = lambda z: np.exp(-20 * (np.asarray(z, dtype=float) - 0.6) ** 2)  # noqa: E731
    scale = 1.0 / np.trapezoid(bump(x), x)
    return MFGProblem(
        model=_model(),
        domain=grid,
        conditions=Conditions(
            m_initial=lambda z: scale * bump(z),
            u_terminal=lambda z: 0.0 * np.asarray(z, dtype=float),
            T=0.5,
        ),
        Nt=10,
    )


def _mesh_problem(bc: BoundaryConditions) -> MFGProblem:
    """A P1 mesh of the unit square with ``bc`` as the problem's shared BC, as #2525 builds it."""
    import skfem

    from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
    from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

    geometry = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
    geometry.mesh_data = skfem_to_meshdata(skfem.MeshTri.init_sqsymmetric().refined(2))
    geometry.boundary_conditions = bc
    return MFGProblem(
        model=_model(),
        domain=geometry,
        conditions=Conditions(m_initial=lambda p: 1.0 + 0.0 * p[0], u_terminal=lambda p: 0.0, T=0.5),
        Nt=10,
    )


def _fem(cap):
    from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver

    return FPFEMSolver(_mesh_problem(_exit(2, cap)), order=1)


ROUTES = {
    "fdm-shared": lambda cap: FPFDMSolver(_grid_problem(_exit(1, cap))),
    "fdm-explicit": lambda cap: FPFDMSolver(_grid_problem(_exit(1, None)), boundary_conditions=_exit(1, cap)),
    "particle-shared": lambda cap: FPParticleSolver(_grid_problem(_exit(1, cap)), num_particles=200),
    "particle-explicit": lambda cap: FPParticleSolver(
        _grid_problem(_exit(1, None)), num_particles=200, boundary_conditions=_exit(1, cap)
    ),
    "fem-shared": _fem,  # FP-FEM takes no explicit BC (#2532)
}


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_a_capped_exit_is_refused_and_the_uncapped_one_is_not(route):
    build = ROUTES[route]
    build(None)  # the control: the same exit without a cap constructs
    with pytest.raises(NotImplementedError, match=REFUSAL):
        build(CAP)


def test_the_default_solve_refuses_a_capped_exit():
    """`problem.solve()` with the default scheme builds FP-FDM, so the silent answer was on the default path."""
    with pytest.raises(NotImplementedError, match=REFUSAL):
        _grid_problem(_exit(1, CAP)).solve(max_iterations=1)

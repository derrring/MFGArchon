"""#2490 B1d: the particle solver's choice between its fast path and the per-point path counts ``default_bc`` and
reads each condition's particle action, rather than the segments alone."""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver
from mfgarchon.alg.numerical.fp_solvers.fp_particle_bc import needs_segment_aware_bc
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc


def _problem(dimension: int, n: int) -> MFGProblem:
    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.05
        ),
        domain=TensorProductGrid(
            bounds=[(0.0, 1.0)] * dimension, Nx_points=[n] * dimension, boundary_conditions=no_flux_bc(dimension)
        ),
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.2),
        Nt=5,
    )


def _reflecting_x_min(dimension: int, default: BCType) -> BoundaryConditions:
    return BoundaryConditions(
        segments=[BCSegment(name="wall", bc_type=BCType.REFLECTING, value=0.0, boundary="x_min")],
        dimension=dimension,
        default_bc=default,
    )


@pytest.mark.parametrize(
    "bc",
    [
        _reflecting_x_min(1, BCType.DIRICHLET),
        BoundaryConditions(segments=[], dimension=1, default_bc=BCType.DIRICHLET),
    ],
    ids=["reflecting x_min, DIRICHLET default", "no segment, DIRICHLET default"],
)
def test_walls_that_default_bc_makes_absorbing_absorb(bc):
    """``default_bc=DIRICHLET`` makes x_max absorb. The routing read the segments alone, so one reflecting segment, or
    none, sent the solve down the reflecting fast path, and x_max reflected every particle. 200 particles start in
    [0.8, 0.95] and drift right at speed 3 for t = 0.2, so their mean position ends past x = 1."""
    solver = FPParticleSolver(_problem(1, 41), num_particles=200, seed=0, boundary_conditions=bc)
    start = np.random.default_rng(0).uniform(0.8, 0.95, (200, 1))
    solver.solve_fp_system(
        initial_particles=start,
        drift_field=lambda t, x, m: np.full(np.shape(x), 3.0),
        drift_needs_density=False,
        show_progress=False,
    )
    final = np.asarray(solver.M_particles_trajectory[-1], dtype=float).reshape(-1)
    assert np.count_nonzero(np.isfinite(final)) < 100, f"{np.count_nonzero(np.isfinite(final))} of 200 remain"


def test_one_condition_on_every_wall_takes_the_fast_path():
    """Four NO_FLUX faces beside a NO_FLUX default carry one condition everywhere, so the fast path serves. Read by
    the segments alone, four segments were "mixed" and sent the solve down the per-point path."""
    bc = BoundaryConditions(
        segments=[
            BCSegment(name=face, bc_type=BCType.NO_FLUX, value=0.0, boundary=face)
            for face in ("x_min", "x_max", "y_min", "y_max")
        ],
        dimension=2,
        default_bc=BCType.NO_FLUX,
    )
    assert needs_segment_aware_bc(bc) is False


@pytest.mark.parametrize("dimension", [1, 2])
def test_two_reflecting_conditions_keep_the_fast_path(dimension):
    """A REFLECTING x_min beside a NO_FLUX default is two conditions with one particle action, so the fast path
    serves it. Routed per point by condition, the solve raised on the grid-drift path with ``preserve_indices=True``,
    which only the fast path implements (#1119; #2739 review 2). U = 3 * sum(x) drives every particle toward x_min at
    speed 3, so x_min and, in 2-D, the default's y_min both reflect particles. Reflected, all 200 end pressed against
    those walls, below 0.6 on every axis; a wrap would carry the ones crossing x_min to near 1 (0.995 under
    periodic_bc)."""
    n = 41 if dimension == 1 else 11
    shape = (n,) * dimension
    grid = np.meshgrid(*[np.linspace(0.0, 1.0, n)] * dimension, indexing="ij")
    potential = np.broadcast_to(3.0 * sum(grid), (6, *shape)).copy()
    solver = FPParticleSolver(
        _problem(dimension, n),
        num_particles=200,
        seed=0,
        boundary_conditions=_reflecting_x_min(dimension, BCType.NO_FLUX),
        preserve_indices=True,
    )
    solver.solve_fp_system(np.ones(shape), potential, show_progress=False)
    final = np.asarray(solver.M_particles_trajectory[-1], dtype=float).reshape(200, dimension)
    assert np.all(np.isfinite(final)), f"{np.count_nonzero(np.all(np.isfinite(final), axis=1))} of 200 remain"
    assert final.min() >= 0.0, f"a particle ended at {final.min()}, outside the box"
    assert final.max() < 0.75, f"a particle ended at {final.max()}: not held against the reflecting walls"

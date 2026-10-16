"""#2490 B1d: what a particle does at the boundary has one owner, `particle_action_at`, and it reads the owner's corner
rule and ``default_bc``."""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc
from mfgarchon.geometry.boundary.applicator_particle import ParticleApplicator


@pytest.mark.parametrize(
    ("x_max", "y_max", "fate"),
    [
        ((BCType.REFLECTING, 0), (BCType.DIRICHLET, 0), "absorbed"),
        ((BCType.DIRICHLET, 0), (BCType.REFLECTING, 1), "reflected"),
    ],
    ids=["equal priority: the exit, essential before natural", "the wall's higher priority"],
)
def test_a_particle_beyond_a_corner_takes_the_corners_segment(x_max, y_max, fate):
    """A particle beyond the corner (1, 1) reads the corner, where ``corner_strategy="priority"`` orders the two faces'
    segments by priority, then DIRICHLET before a natural condition (`BoundaryConditions._corner_key`). The applicator
    named the first face it found the particle on, x before y, and asked that face alone, so a reflecting x_max
    swallowed an absorbing y_max (#1561 item 1), and an absorbing x_max a reflecting y_max of higher priority."""
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="x_max", bc_type=x_max[0], value=0.0, boundary="x_max", priority=x_max[1]),
            BCSegment(name="y_max", bc_type=y_max[0], value=0.0, boundary="y_max", priority=y_max[1]),
        ],
        dimension=2,
        default_bc=BCType.NO_FLUX,
        corner_strategy="priority",
    )
    remaining, absorbed, _ = ParticleApplicator().apply(np.array([[1.05, 1.05]]), bc, [(0.0, 1.0), (0.0, 1.0)])
    if fate == "absorbed":
        assert absorbed.tolist() == [True], remaining
        return
    assert absorbed.tolist() == [False]
    np.testing.assert_allclose(remaining, [[0.95, 0.95]], rtol=0, atol=1e-12)


def test_walls_that_default_bc_makes_absorbing_absorb():
    """A reflecting x_min and ``default_bc=DIRICHLET``: x_max absorbs. The solver's routing read the segments alone, so
    one reflecting segment sent the solve down the reflecting fast path, and x_max reflected every particle. It reads
    `BoundaryConditions.uniform_segment`, which counts ``default_bc``. 200 particles start in [0.8, 0.95] and drift
    right at speed 3 for t = 0.2, so their mean position ends past x = 1."""
    bc = BoundaryConditions(
        segments=[BCSegment(name="wall", bc_type=BCType.REFLECTING, value=0.0, boundary="x_min")],
        dimension=1,
        default_bc=BCType.DIRICHLET,
    )
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.05
        ),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.2),
        Nt=5,
    )
    solver = FPParticleSolver(problem, num_particles=200, seed=0, boundary_conditions=bc)
    start = np.random.default_rng(0).uniform(0.8, 0.95, (200, 1))
    solver.solve_fp_system(
        initial_particles=start,
        drift_field=lambda t, x, m: np.full(np.shape(x), 3.0),
        drift_needs_density=False,
        show_progress=False,
    )
    final = np.asarray(solver.M_particles_trajectory[-1], dtype=float).reshape(-1)
    assert np.count_nonzero(np.isfinite(final)) < 100, f"{np.count_nonzero(np.isfinite(final))} of 200 remain"

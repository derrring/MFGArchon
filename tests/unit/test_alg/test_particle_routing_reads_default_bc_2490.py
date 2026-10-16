"""#2490 B1d: the particle solver's choice between its reflecting fast path and the per-point path reads
`BoundaryConditions.uniform_segment`, which counts ``default_bc``, rather than the segments alone."""

from __future__ import annotations

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver
from mfgarchon.alg.numerical.fp_solvers.fp_particle_bc import needs_segment_aware_bc
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc


def test_walls_that_default_bc_makes_absorbing_absorb():
    """A reflecting x_min and ``default_bc=DIRICHLET``: x_max absorbs. The routing read the segments alone, so one
    reflecting segment sent the solve down the reflecting fast path, and x_max reflected every particle. 200 particles
    start in [0.8, 0.95] and drift right at speed 3 for t = 0.2, so their mean position ends past x = 1."""
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

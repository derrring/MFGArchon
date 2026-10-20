"""#2780: a particle past a wall, read on the face it crossed, is read at its nearest point of the box by every
placement field, so a ``region`` or ``sdf_region`` exit holding only the face's own coordinate holds it as the same part
of the face written with ``boundary=`` does."""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc


def _remaining(placement: dict) -> int:
    """Of 200 particles started in [0.8, 0.95] x [0.43, 0.47] and drifting right at speed 3 for T = 0.2, how many the
    exit placed by ``placement`` on x_max leaves in the domain, the rest of the boundary NO_FLUX."""
    exit_segment = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, **placement)
    bc = BoundaryConditions(segments=[exit_segment], dimension=2, default_bc=BCType.NO_FLUX)
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=no_flux_bc(dimension=2)
    )
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.05
        ),
        domain=grid,
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.2),
        Nt=5,
    )
    solver = FPParticleSolver(problem, num_particles=200, seed=0, boundary_conditions=bc)
    solver.solve_fp_system(
        initial_particles=np.random.default_rng(0).uniform([0.8, 0.43], [0.95, 0.47], (200, 2)),
        drift_field=lambda t, x, m: np.broadcast_to(np.array([3.0, 0.0]), np.shape(x)),
        drift_needs_density=False,
        show_progress=False,
    )
    final = np.asarray(solver.M_particles_trajectory[-1], dtype=float).reshape(-1, 2)
    return int(np.count_nonzero(np.all(np.isfinite(final), axis=1)))


@pytest.mark.parametrize(
    ("placement", "twin"),
    [
        ({"region": {"x": (1.0, 1.0), "y": (0.42, 0.48)}}, (0.42, 0.48)),
        ({"region": {"x": (1.0, 1.0), "y": (0.3, 0.7)}}, (0.3, 0.7)),
        ({"sdf_region": lambda p: float(np.hypot(p[0] - 1.0, p[1] - 0.45) - 0.05)}, (0.40, 0.50)),
    ],
    ids=["a region between the nodes", "a region holding nodes", "an sdf_region disk"],
)
def test_an_exit_on_part_of_a_face_absorbs_as_its_boundary_placed_twin(placement, twin):
    """Each exit holds the part of x_max that ``boundary="x_max"`` with ``region={"y": twin}`` holds, the twin reading
    the stepped point's y alone. The point owner read a ``region`` or ``sdf_region`` at the stepped point, x > 1, where
    the exit holds nothing, so every particle crossed it under ``default_bc``: 200, 200 and 35 remained against the
    twin's 12, 0 and 0. The 12 cross x = 1 outside the narrow strip and are reflected under both placements."""
    expected = _remaining({"boundary": "x_max", "region": {"y": twin}})
    assert expected < 200, "no particle reached the exit, so the comparison is vacuous"
    assert _remaining(placement) == expected

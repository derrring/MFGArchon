"""#2780: a particle past a wall, read on the face it crossed, is read at its nearest point of the box by every
placement region, so a ``region`` or ``sdf_region`` exit on part of a face holds it as the same part of the face
written with ``boundary=`` does. The outward normal a ``normal_direction`` reads under ``domain_sdf`` is read at the
particle itself."""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc


def _remaining(placement: dict, start=((0.8, 0.43), (0.95, 0.47)), drift=(3.0, 0.0), domain_sdf=None) -> int:
    """Of 200 particles started uniformly in the box ``start`` and drifting at ``drift`` for T = 0.2, how many the exit
    placed by ``placement`` leaves in the domain, the rest of the boundary NO_FLUX."""
    exit_segment = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, **placement)
    bc = BoundaryConditions(segments=[exit_segment], dimension=2, default_bc=BCType.NO_FLUX, domain_sdf=domain_sdf)
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
        initial_particles=np.random.default_rng(0).uniform(start[0], start[1], (200, 2)),
        drift_field=lambda t, x, m: np.broadcast_to(np.array(drift), np.shape(x)),
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
    the stepped point's y alone. The point owner read a ``region`` or ``sdf_region`` at the stepped point, x > 1: the
    region exits held no stepped point and the disk only those that stepped into it, so the others crossed under
    ``default_bc``: 200, 200 and 35 remained against the twin's 12, 0 and 0. The 12 are read past the wall at a y
    outside the narrow strip and are reflected under both placements."""
    expected = _remaining({"boundary": "x_max", "region": {"y": twin}})
    assert expected < 200, "no particle reached the exit, so the comparison is vacuous"
    assert _remaining(placement) == expected


def _unit_box_sdf(p):
    q = np.abs(np.asarray(p, dtype=float).ravel() - 0.5) - 0.5
    return float(np.linalg.norm(np.maximum(q, 0.0)) + min(q.max(), 0.0))


def test_a_normal_direction_exit_reads_the_sdf_normal_at_the_particle():
    """A BC carrying the unit box's SDF reads a ``normal_direction`` exit by the SDF's gradient. The particle applicator
    gives such a BC the step's box, so a particle past the corner (1, 1) read at its nearest point of the box would be
    read at the corner, a kink, where the gradient is the diagonal: an exit whose ``normal_tolerance`` exceeds 1/sqrt(2)
    would hold no such particle. #2781's review 1 measured it: of these 200 particles, driven toward the corner, 85
    stayed under an exit with normal (1, 0) and tolerance 0.85, where the ``boundary="x_max"`` twin left none. Read
    at the particle, as before #2780, the exit leaves none either."""
    corner = {"start": ((0.85, 0.9), (0.95, 0.98)), "drift": (3.0, 1.5), "domain_sdf": _unit_box_sdf}
    expected = _remaining({"boundary": "x_max"}, **corner)
    assert expected < 200, "no particle reached the exit, so the comparison is vacuous"
    normal_exit = {"normal_direction": np.array([1.0, 0.0]), "normal_tolerance": 0.85}
    assert _remaining(normal_exit, **corner) == expected

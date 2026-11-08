"""#2910: how many distinct nodes each wrapping axis spans on a Cartesian grid has one owner, `periodic_axis_spans`, and
its readers report what it says.

The oracles of the approved plan (#2910, issuecomment-6724735013), stage 1:
- cross-reader identity: each reader gives the span the rule gives. The rule, written here independently of the code:
  an axis that wraps spans n - 1 distinct nodes on a grid whose last node repeats node 0 (endpoint-inclusive), and
  does not wrap otherwise;
- twin nodes equal: on an endpoint-inclusive grid every FP solver returns node 0 and its repeat equal, from a value
  field whose repeat differs from node 0 (#2909 review 3, SF5).
A cell red at stage 1's head is ``xfail(strict=True)`` naming the stage that moves its reader. When it turns green its
strict XPASS fails the run and prints the reason, which says what to do.
"""

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling.fixed_point_utils import compute_fp_velocity_field
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver, FPFVMSolver, FPParticleSolver, FPSLSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BoundaryConditions, periodic_bc
from mfgarchon.geometry.boundary.conditions import periodic_axis_span
from mfgarchon.geometry.boundary.types import BCType
from mfgarchon.geometry.grids.tensor_grid import periodic_axis_spans

N = 9
STAGE_2 = pytest.mark.xfail(
    strict=True,
    reason="remove this xfail: stage 2 of #2910/#2911 moved this reader to the layout owner, so it now agrees",
)
P, W = BCType.PERIODIC, BCType.NO_FLUX


def _faces(kinds):
    segments = []
    for axis, name in enumerate("xy"):
        for side, kind in zip(("min", "max"), kinds[axis], strict=True):
            segments.append(BCSegment(name=f"{name}{side}", bc_type=kind, boundary=f"{name}_{side}"))
    return BoundaryConditions(segments=segments, dimension=2)


#: spelling -> (the boundary condition, the span the rule gives on each axis of an endpoint-inclusive 9 x 9 grid)
SPELLINGS = {
    "uniform": (lambda: periodic_bc(dimension=2), (N - 1, N - 1)),
    "per face": (lambda: _faces([(P, P), (P, P)]), (N - 1, N - 1)),
    "default_bc": (lambda: BoundaryConditions(segments=[], default_bc=P, dimension=2), (N - 1, N - 1)),
    "cylinder, x wraps": (lambda: _faces([(P, P), (W, W)]), (N - 1, None)),
    "cylinder, y wraps": (lambda: _faces([(W, W), (P, P)]), (None, N - 1)),
}


def _grid(bc):
    return TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 2.0)], Nx_points=[N, N], boundary_conditions=bc)


@pytest.mark.parametrize("spelling", list(SPELLINGS))
def test_the_layout_owner_gives_the_span_the_rule_gives(spelling):
    make, expected = SPELLINGS[spelling]
    bc = make()
    assert periodic_axis_spans(_grid(bc), bc) == expected


@pytest.mark.parametrize(
    "spelling",
    [pytest.param(s, marks=STAGE_2) if "cylinder" in s else s for s in SPELLINGS],
)
def test_the_bc_field_reader_gives_the_same_span(spelling):
    """`periodic_axis_span` reads the BC's convention field and wraps every axis or none; stage 2 moves it."""
    make, expected = SPELLINGS[spelling]
    grid = _grid(make())
    bc = grid.get_boundary_conditions()
    assert tuple(periodic_axis_span(bc, N, 2, grid) for _ in range(2)) == expected


def test_a_grid_whose_axes_differ_in_layout_has_no_one_convention():
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[11, 10],
        spacing_type="custom",
        custom_coordinates=[np.linspace(0, 1, 11), np.linspace(0, 1, 10, endpoint=False)],
        boundary_conditions=_faces([(W, W), (W, W)]),
    )
    with pytest.raises(ValueError, match=r"axes differ in where the last node sits.*periodic_axis_spans"):
        _ = grid.periodic_convention


class _Recording(SeparableHamiltonian):
    """The library quadratic H, recording where the velocity builder asks for its control."""

    def __init__(self):
        super().__init__(QuadraticControlCost(control_cost=1.0))
        self.points = []

    def evaluate_optimal_control(self, state):
        self.points.append(np.asarray(state.x, dtype=float).ravel())
        return super().evaluate_optimal_control(state)


def _problem_1d(n=21):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=periodic_bc(dimension=1))
    return MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(QuadraticControlCost(control_cost=1.0)), volatility=0.3),
        domain=grid,
        conditions=Conditions(
            m_initial=lambda x: 1.0 + 0.5 * np.cos(2 * np.pi * np.asarray(x[0])), u_terminal=lambda x: 0.0, T=0.3
        ),
        Nt=6,
    )


def test_the_velocity_builder_asks_at_the_faces_and_reads_node_0_across_the_wrap():
    """The faces sit midway between neighbours, inside the domain; the last one, across the wrap, reads node 0
    (#2909 review 3, SF10). The value field's repeat differs from node 0, so a builder reading the repeat shows."""
    problem, n = _problem_1d(), 21
    x = np.linspace(0.0, 1.0, n)
    U = np.stack([0.3 * np.sin(2 * np.pi * x) + 0.2 * x] * 2)
    recording = _Recording()
    velocity = compute_fp_velocity_field(problem, U, np.ones_like(U), recording)
    np.testing.assert_allclose(np.concatenate(recording.points[:1]), 0.5 * (x[:-1] + x[1:]), atol=1e-14)
    h = x[1] - x[0]
    np.testing.assert_allclose(velocity[0, -1], -(U[0, 0] - U[0, -2]) / h, atol=1e-12)


@pytest.mark.parametrize(
    "solver",
    ["FP-FDM", "FP-FVM", "FP-SL", pytest.param("FP-Particle", marks=STAGE_2)],
)
def test_every_fp_solver_returns_twin_nodes_equal(solver):
    problem, n = _problem_1d(), 21
    x = np.linspace(0.0, 1.0, n)
    m0 = 1.0 + 0.5 * np.cos(2 * np.pi * x)
    U = np.stack([0.3 * np.sin(2 * np.pi * x) + 0.2 * x] * (problem.Nt + 1))
    make = {
        "FP-FDM": lambda: FPFDMSolver(problem),
        "FP-FVM": lambda: FPFVMSolver(problem),
        "FP-SL": lambda: FPSLSolver(problem),
        "FP-Particle": lambda: FPParticleSolver(problem, num_particles=4000, seed=0),
    }[solver]
    M = np.asarray(make().solve_fp_system(m0, potential_field=U, show_progress=False))
    np.testing.assert_allclose(M[:, 0], M[:, -1], atol=1e-12 * np.abs(M).max())

"""#2490 (B1d's binding): a BC handed to an FP solver is bound to its problem's geometry, as the shared BC is by its
grid, so a reader that asks the owner without a geometry decides on the grid.

Measured at main b1ffd98a over five FP solvers and four exits (#2490 comment 6247352258 and the gate after it):
binding changes one cell, FP-particle with a ``region_name`` exit, from a refusal to the shared route's answer, and
leaves every other cell unchanged. #2753's review 1 found one more answer outside those cells, a ``region_name`` that is
also a face label on a grid that marks it, now read by the mark as the shared route reads it. It also refuses a BC of
another dimension than the geometry, which six FP solvers accepted at main.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc

UNIT_SQUARE = np.array([[0.0, 1.0], [0.0, 1.0]])


def _grid_marking_a_gate():
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=no_flux_bc(dimension=2)
    )
    grid.mark_region("gate", predicate=lambda x: (x[:, 0] > 0.95) & (x[:, 1] > 0.35) & (x[:, 1] < 0.65))
    return grid


def _problem(grid):
    return MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.1),
        domain=grid,
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.2),
        Nt=5,
    )


def _gate_exit(dimension=2):
    return BoundaryConditions(
        segments=[BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, region_name="gate", priority=1)],
        dimension=dimension,
        default_bc=BCType.NO_FLUX,
        domain_bounds=UNIT_SQUARE.copy() if dimension == 2 else None,
    )


def test_an_explicit_region_name_exit_is_read_on_the_grid_that_marks_it():
    """200 particles drift right from (0.9, 0.5) into the gate, the x_max nodes y = 0.4 ... 0.6, and all are absorbed,
    as with the same BC set on the grid. At main the first boundary particle raised "This reader has no geometry"."""
    solver = FPParticleSolver(
        _problem(_grid_marking_a_gate()), num_particles=200, seed=0, boundary_conditions=_gate_exit()
    )
    start = np.column_stack([np.full(200, 0.9), np.full(200, 0.5)])
    solver.solve_fp_system(
        initial_particles=start,
        drift_field=lambda t, x, m: np.broadcast_to([3.0, 0.0], np.shape(x)),
        drift_needs_density=False,
        show_progress=False,
    )
    final = np.asarray(solver.M_particles_trajectory[-1], dtype=float).reshape(-1, 2)
    assert np.count_nonzero(np.all(np.isfinite(final), axis=1)) == 0


def test_the_solver_binds_a_copy_and_leaves_the_callers_bc_unbound():
    """The solver's BC is bound to the problem's grid; the caller's object is not that BC, and after a solve it is still
    unbound, has no box, and keeps its segments in its own order. At main the solver held the caller's object, and the
    solve wrote the box into it (#2738's shape, on the explicit route); and a copy that shared the caller's segment list
    re-sorted it by priority (#2753 review 1)."""
    grid = _grid_marking_a_gate()
    bc = BoundaryConditions(
        segments=[BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max")],
        dimension=2,
        default_bc=BCType.NO_FLUX,
    )
    bc.segments.append(BCSegment(name="side", bc_type=BCType.NO_FLUX, boundary="y_max", priority=5))
    solver = FPParticleSolver(_problem(grid), num_particles=50, seed=0, boundary_conditions=bc)
    solver.solve_fp_system(
        initial_particles=np.column_stack([np.full(50, 0.9), np.full(50, 0.5)]),
        drift_field=lambda t, x, m: np.broadcast_to([3.0, 0.0], np.shape(x)),
        drift_needs_density=False,
        show_progress=False,
    )
    assert solver.boundary_conditions is not bc
    assert solver.boundary_conditions._geometry_ref() is grid
    assert bc._geometry_ref is None
    assert bc.domain_bounds is None
    assert [segment.name for segment in bc.segments] == ["exit", "side"]


def test_a_bc_of_another_dimension_is_refused_naming_the_dimension_to_build_it_in():
    """A 1-D BC handed to a solver on a 2-D grid ran at main, read as if it had two axes. It is refused, and the remedy,
    the BC built with the geometry's dimension, constructs."""
    grid = _grid_marking_a_gate()
    with pytest.raises(ValueError, match=r"BC dimension mismatch.*Build the BC with dimension=2"):
        FPParticleSolver(_problem(grid), num_particles=10, seed=0, boundary_conditions=_gate_exit(dimension=1))
    FPParticleSolver(_problem(grid), num_particles=10, seed=0, boundary_conditions=_gate_exit(dimension=2))

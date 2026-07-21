"""The nD SL fold must refuse a mixed per-axis BC wherever it reads one (Issue #1560).

`get_bc_type_string` returns the FIRST segment's type by contract. `_trace_characteristic_backward`
passes that single string to `apply_boundary_conditions_nd`, which loops `for d in range(dimension)`
and applies the same operation to every axis -- so no-flux walls on one axis plus periodic on
another silently drops one transform, and reordering the segments changes the physics.

A construction-time guard existed. It is not sufficient on its own: the solver re-reads
`get_boundary_conditions()` at solve time, so a BC that is unset when the solver is built, or
replaced afterwards, reaches the fold unchecked. These tests pin the check at the point of use.

Per-axis handling is the actual fix and remains open on #1560.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    no_flux_bc,
    periodic_bc,
)


def _components():
    return MFGComponents(
        m_initial=lambda x: float(np.exp(-np.sum((np.atleast_1d(x) - 0.5) ** 2))),
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
    )


def _grid_and_problem(bc):
    grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=bc)
    return grid, MFGProblem(geometry=grid, Nt=5, T=1.0, components=_components())


def _mixed_bc():
    """no-flux on one axis (reflect) beside periodic on another (wrap)."""
    return BoundaryConditions(
        dimension=2,
        default_bc=BCType.NO_FLUX,
        segments=[
            BCSegment(name="wall_x", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="periodic_y", bc_type=BCType.PERIODIC, boundary="y_min"),
        ],
    )


def test_mixed_bc_is_refused_at_construction():
    """The early guard: cheapest place to tell the caller."""
    _, problem = _grid_and_problem(_mixed_bc())

    with pytest.raises(NotImplementedError, match="different geometric operations"):
        HJBSemiLagrangianSolver(problem)


def test_mixed_bc_is_refused_when_it_arrives_after_construction():
    """The check must sit at the point of use, not only at construction.

    The solver re-reads `get_boundary_conditions()` every solve, so a construction-time check
    alone is bypassed. Before this was pinned, the swapped-in BC reached the fold and
    `segments[0]`'s operation was applied to both axes with no error.
    """
    grid, problem = _grid_and_problem(no_flux_bc(dimension=2))
    solver = HJBSemiLagrangianSolver(problem)

    grid._boundary_conditions = _mixed_bc()

    with pytest.raises(NotImplementedError, match="different geometric operations"):
        solver.solve_hjb_system(np.ones((6, 11, 11)), np.zeros((11, 11)), np.zeros((6, 11, 11)))


@pytest.mark.parametrize("bc_factory", [lambda: no_flux_bc(dimension=2), lambda: periodic_bc(dimension=2)])
def test_a_uniform_bc_still_solves(bc_factory):
    """The refusal must be about disagreement, not about having segments at all."""
    _, problem = _grid_and_problem(bc_factory())
    solver = HJBSemiLagrangianSolver(problem)

    result = solver.solve_hjb_system(np.ones((6, 11, 11)), np.zeros((11, 11)), np.zeros((6, 11, 11)))

    assert np.all(np.isfinite(result))

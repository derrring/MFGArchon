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


def _grid_and_problem(bc, dim=2):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * dim, Nx_points=[11] * dim, boundary_conditions=bc)
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


def test_every_bc_type_read_goes_through_the_checked_accessor():
    """Pin all seven call sites at once, as a backstop to the behavioural matrix below.

    An independent review of #1696 showed that reverting any *single* site to the raw
    `get_bc_type_string` left the whole suite green: only two of the seven sites were exercised,
    and those two are mutually redundant. This asserts the invariant on the source, so a single
    reverted site fails loudly regardless of which dispatch path reaches it.

    It is a backstop, not the primary evidence -- `test_refusal_holds_across_the_dispatch_matrix`
    covers the behaviour. Three known evasions, all requiring a deliberate edit rather than a
    revert: an early `return get_bc_type_string(bc)` inside the helper's own line range; a net
    *addition* written in attribute form (`bc_utils.get_bc_type_string(...)`), which `ast.Name`
    does not match; and adding a legitimate eighth site, which fails on the count.
    """
    import ast
    import inspect

    from mfgarchon.alg.numerical.hjb_solvers import hjb_semi_lagrangian

    tree = ast.parse(inspect.getsource(hjb_semi_lagrangian))
    helper = "_checked_bc_type_string"

    raw_calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "get_bc_type_string"
        # the helper is the one legitimate caller
        and not any(
            isinstance(fn, ast.FunctionDef)
            and fn.name == helper
            and fn.lineno <= node.lineno <= (fn.end_lineno or fn.lineno)
            for fn in ast.walk(tree)
        )
    ]

    assert not raw_calls, (
        f"get_bc_type_string called directly at line(s) {raw_calls}; route through {helper} "
        "so a mixed per-axis BC is refused rather than silently applied to every axis (#1560)"
    )

    checked = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == helper
    )
    assert checked == 7, (
        f"expected 7 checked reads, found {checked}. Sites as of #1696: "
        "_solve_timestep_semi_lagrangian, _canonical_cs_step (x2), _apply_boundary_to_point, "
        "_trace_characteristic_backward (x2), _get_diffusion_bc_type. Update both if you add one."
    )


def test_refusal_is_not_retyped_by_the_pointwise_handler():
    """`_advect_pointwise` wraps each node in `except Exception -> RuntimeError`.

    Without a passthrough the refusal reaches the caller as RuntimeError, so the declared contract
    holds on some paths and not others. Found by review of #1696.
    """
    grid, problem = _grid_and_problem(no_flux_bc(dimension=1), dim=1)
    solver = HJBSemiLagrangianSolver(problem, characteristic_solver="rk4")

    grid._boundary_conditions = BoundaryConditions(
        dimension=1,
        default_bc=BCType.NO_FLUX,
        segments=[
            BCSegment(name="wall", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="wrap", bc_type=BCType.PERIODIC, boundary="x_max"),
        ],
    )

    with pytest.raises(NotImplementedError, match="different geometric operations"):
        solver.solve_hjb_system(np.ones((6, 11)), np.zeros(11), np.zeros((6, 11)))


def _mixed_bc_nd(dim):
    """no-flux beside periodic -- distinct geometric operations, on any dimension."""
    other = "y_min" if dim > 1 else "x_max"
    return BoundaryConditions(
        dimension=dim,
        default_bc=BCType.NO_FLUX,
        segments=[
            BCSegment(name="wall", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="wrap", bc_type=BCType.PERIODIC, boundary=other),
        ],
    )


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("characteristic_solver", ["explicit_euler", "rk2", "rk4"])
@pytest.mark.parametrize("diffusion_method", ["adi", "none", "explicit", "stochastic", "canonical_cs"])
def test_refusal_holds_across_the_dispatch_matrix(dim, characteristic_solver, diffusion_method):
    """The refusal must hold on every dispatch path, with its declared type.

    The seven routed sites sit behind different combinations of dimension, characteristic solver
    and diffusion method; the two behavioural tests above reach only two of them. Review of #1696
    established that this matrix discriminates: at the parent commit exactly three cells regressed
    (dim=1, rk4, diffusion in {adi, none, explicit}) -- the `_advect_pointwise` path whose
    `except Exception` retyped the refusal to RuntimeError.

    The PR body had claimed behavioural coverage here needed fixtures that did not exist yet. It
    did not; this is that coverage.
    """
    grid, problem = _grid_and_problem(no_flux_bc(dimension=dim), dim=dim)
    solver = HJBSemiLagrangianSolver(
        problem, characteristic_solver=characteristic_solver, diffusion_method=diffusion_method
    )

    grid._boundary_conditions = _mixed_bc_nd(dim)

    shape = (11,) * dim
    with pytest.raises(NotImplementedError, match="different geometric operations"):
        solver.solve_hjb_system(np.ones((6, *shape)), np.zeros(shape), np.zeros((6, *shape)))

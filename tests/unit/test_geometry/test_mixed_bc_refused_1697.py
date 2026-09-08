"""One owner refuses a per-axis-blind BC collapse, for HJB-SL and FP-SL alike (#1560, #1697).

``get_bc_type_string`` returns the FIRST segment's type: ``BoundaryConditions.type`` deliberately
raises ``ValueError`` for a mixed BC, ``bc_utils`` swallows that raise, and execution falls through
to ``segments[0].bc_type``. The one signal that the BC is mixed is discarded, and the fold then
applies that single operation to every axis.

Two levers produce the collapse, and a guard that handles only the first is insufficient:

1. **Segment order.** Reordering the list flips the surviving type.
2. **``default_bc``.** ``get_bc_type_string`` never reads it, so a partially-covering segment list
   plus a differing default collapses identically **with no permutation available**. A guard that
   unions only over ``segments`` lets this straight through.

Per-axis handling remains open on #1560 (HJB) and #1697 (FP).
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    no_flux_bc,
    periodic_bc,
)
from mfgarchon.geometry.boundary.bc_utils import (
    checked_bc_type_string,
    geometric_operations,
    get_bc_type_string,
    refuse_mixed_per_axis,
)

CONSUMER = {"consumer": "TestSolver", "alternative": "Use one BC type across axes."}

_RETIRE_1700B = """`get_bc_type_string` no longer raises on a segment-free BC.

That is #1700 part B landing, which this assertion exists to notice: it calls that configuration
legitimate, so the ValueError is a defect and its removal is progress. Delete THIS assertion and the
one after it -- you are standing at the first of the two -- and the paragraph of the test docstring
that introduces them. Keep everything above: the guard/lookup split is a responsibility argument
(#2284) and never depended on the ValueError existing. Do NOT restore the raise."""

_LOOKUP_NO_LONGER_REACHED = """`checked_bc_type_string` did not raise on a segment-free BC, but
`get_bc_type_string` still does -- so #1700 has NOT landed and the composite has stopped routing
through the lookup.

The split (#2284) was meant to leave `checked_bc_type_string` as guard-then-lookup. Restore the
lookup, or if the composite is deliberately gone, re-point this test at whatever now owns
collapse-to-a-value rather than deleting it."""


def _seg(name, bc_type, boundary):
    return BCSegment(name=name, bc_type=bc_type, boundary=boundary)


def _two_segments(first_periodic=False):
    order = [BCType.PERIODIC, BCType.NO_FLUX] if first_periodic else [BCType.NO_FLUX, BCType.PERIODIC]
    return BoundaryConditions(
        dimension=2,
        default_bc=order[0],
        segments=[_seg("a", order[0], "x_min"), _seg("b", order[1], "y_min")],
    )


def test_the_raise_that_marks_a_bc_mixed_is_swallowed_by_the_accessor():
    """Pins the root cause, so a future 'fix' that only patches a solver is visibly partial."""
    mixed = _two_segments()

    with pytest.raises(ValueError, match="only valid for uniform BCs"):
        _ = mixed.type

    # ... yet the accessor returns a plain answer, having discarded the one signal it had.
    assert get_bc_type_string(mixed) == "no_flux"


def test_segment_order_changes_what_the_accessor_returns():
    """Lever 1. This is the form the issue was originally filed against."""
    assert get_bc_type_string(_two_segments(first_periodic=False)) == "no_flux"
    assert get_bc_type_string(_two_segments(first_periodic=True)) == "periodic"


def test_default_bc_is_a_second_lever_with_no_permutation_available():
    """Lever 2, and the reason a segments-only guard is insufficient.

    Neither BC below is a reordering of the other -- each has exactly one segment -- yet they
    collapse to different operations. A guard unioning only over ``segments`` sees a single element
    in both cases and permits them.
    """
    a = BoundaryConditions(dimension=2, default_bc=BCType.PERIODIC, segments=[_seg("w", BCType.NO_FLUX, "x_min")])
    b = BoundaryConditions(dimension=2, default_bc=BCType.NO_FLUX, segments=[_seg("p", BCType.PERIODIC, "y_min")])

    assert get_bc_type_string(a) == "no_flux"
    assert get_bc_type_string(b) == "periodic"

    # What a segments-only guard would see: one element each, so no disagreement.
    for bc in (a, b):
        ops_from_segments = {str(getattr(s.bc_type, "value", s.bc_type)) for s in bc.segments}
        assert len(ops_from_segments) == 1, "a segments-only guard cannot see this disagreement"

    # What the shipped guard sees, because it also unions default_bc:
    assert geometric_operations(a) == {"reflect", "periodic"}
    assert geometric_operations(b) == {"reflect", "periodic"}


@pytest.mark.parametrize(
    "bc_factory",
    [
        pytest.param(lambda: _two_segments(False), id="two-segments"),
        pytest.param(lambda: _two_segments(True), id="two-segments-reordered"),
        pytest.param(
            lambda: BoundaryConditions(
                dimension=2, default_bc=BCType.PERIODIC, segments=[_seg("w", BCType.NO_FLUX, "x_min")]
            ),
            id="one-segment-plus-differing-default",
        ),
    ],
)
def test_mixed_bc_is_refused(bc_factory):
    with pytest.raises(NotImplementedError, match="different geometric operations"):
        checked_bc_type_string(bc_factory(), **CONSUMER)


@pytest.mark.parametrize(
    "bc_factory",
    [
        pytest.param(lambda: no_flux_bc(dimension=2), id="uniform-no-flux"),
        pytest.param(lambda: periodic_bc(dimension=2), id="uniform-periodic"),
        pytest.param(
            lambda: BoundaryConditions(
                dimension=2, default_bc=BCType.NO_FLUX, segments=[_seg("w", BCType.NO_FLUX, "x_min")]
            ),
            id="segments-agree-with-default",
        ),
        pytest.param(
            lambda: BoundaryConditions(
                dimension=2,
                default_bc=BCType.NO_FLUX,
                segments=[_seg("w", BCType.NO_FLUX, "x_min"), _seg("n", BCType.NEUMANN, "y_min")],
            ),
            id="different-bctypes-same-operation",
        ),
        pytest.param(lambda: None, id="none"),
    ],
)
def test_a_bc_without_disagreement_is_accepted(bc_factory):
    """The refusal must key on disagreement, not on having segments, and not on BCType identity.

    ``NEUMANN`` and ``NO_FLUX`` are distinct BCTypes that map to the same geometric operation. The
    guard must let that through; refusing it would be a false positive on a legitimate wall.
    """
    checked_bc_type_string(bc_factory(), **CONSUMER)


def test_reflect_and_periodic_coincide_without_a_boundary_crossing_drift():
    """Why a naive regression fixture cannot detect this defect.

    If no characteristic foot leaves the domain, 'reflect' and 'periodic' are the same map on the
    reachable set, so the collapse is invisible. Any behavioural test for #1697 needs destinations
    that actually cross a wall. Asserted here so the constraint is not rediscovered by hand.
    """
    lo, hi = 0.0, 1.0
    inside = np.array([0.2, 0.5, 0.8])
    crossing = np.array([-0.15, 1.10])

    def reflect(x):
        return hi - np.abs(np.mod(x - lo, 2 * (hi - lo)) - (hi - lo))

    def periodic(x):
        return lo + np.mod(x - lo, hi - lo)

    np.testing.assert_allclose(reflect(inside), periodic(inside), atol=1e-15)
    assert np.max(np.abs(reflect(crossing) - periodic(crossing))) > 0.1


def _fp_problem(bc, dim=2, n=9, nt=4):
    from mfgarchon import MFGProblem
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.geometry import TensorProductGrid

    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * dim, Nx_points=[n] * dim, boundary_conditions=bc)
    components = MFGComponents(
        m_initial=lambda x: float(np.exp(-np.sum((np.atleast_1d(x) - 0.5) ** 2))),
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
    )
    return grid, MFGProblem(geometry=grid, Nt=nt, T=0.5, components=components)


def test_fp_sl_solver_refuses_a_mixed_bc_at_solve_time():
    """The wiring, not the helper.

    Asserting on ``checked_bc_type_string`` alone does not pin that FPSLSolver *calls* it -- those
    assertions stay green with the solver reverted to the raw accessor. This constructs the solver
    and solves. Verified by mutation: routing ``_get_bc_operation_type`` back to
    ``get_bc_type_string`` fails this test and only this one.

    The BC is swapped in on the geometry after construction on purpose: FPSLSolver caches only an
    explicitly-passed BC and otherwise resolves the geometry live at each point of use, so a
    construction-time check alone would be bypassed here exactly as on the HJB side (#1560).
    """
    from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver

    grid, problem = _fp_problem(no_flux_bc(dimension=2))
    solver = FPSLSolver(problem)

    # ONLY the geometry's BC is replaced. Writing solver.boundary_conditions as well would pass
    # even if the solver never re-read anything -- review of #1702 found exactly that: the test
    # asserted a re-read mechanism the FP solver did not have, and passed by poking the cache.
    grid._boundary_conditions = _two_segments()

    m0 = np.ones((9, 9)) / 81.0
    u = np.zeros((5, 9, 9))
    with pytest.raises(NotImplementedError, match="different geometric operations"):
        solver.solve_fp_system(M_initial=m0, potential_field=u)


def test_fp_sl_solver_still_solves_a_uniform_bc():
    """The refusal must not cost the ordinary case."""
    from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver

    _, problem = _fp_problem(no_flux_bc(dimension=2))
    result = FPSLSolver(problem).solve_fp_system(M_initial=np.ones((9, 9)) / 81.0, potential_field=np.zeros((5, 9, 9)))

    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)


def test_a_duck_typed_bc_is_checked_not_waved_through():
    """Reach is by duck typing, and that choice is pinned in both directions.

    An `isinstance` gate would be a fail-silent branch in front of a fail-loud body: an adapter or
    wrapper that is not literally a BoundaryConditions would return an empty set, read as "nothing
    disagrees". Review of #1702 measured that a gate was present and that removing it changed no
    test -- unpinned in either direction, so the next refactor could flip it invisibly.
    """
    from types import SimpleNamespace

    duck = SimpleNamespace(
        segments=[_seg("w", BCType.NO_FLUX, "x_min"), _seg("p", BCType.PERIODIC, "y_min")],
        default_bc=BCType.NO_FLUX,
    )
    assert geometric_operations(duck) == {"reflect", "periodic"}

    with pytest.raises(NotImplementedError, match="different geometric operations"):
        checked_bc_type_string(duck, **CONSUMER)


def test_an_object_carrying_neither_field_is_not_a_segmented_bc():
    """A legacy BC has no per-axis information; it must not raise, and must not be refused."""
    from types import SimpleNamespace

    assert geometric_operations(SimpleNamespace(type="periodic")) == set()
    assert geometric_operations(None) == set()


def test_the_guard_and_the_lookup_are_separable_2284(still_refused):
    """`refuse_mixed_per_axis` is the predicate; `checked_bc_type_string` is it plus the lookup.

    They were one function until #2284, and the difference is not cosmetic: a segment-free BC asks
    for exactly one geometric operation, so the guard passes it, while `get_bc_type_string` raises
    `ValueError` on it. A caller that wants only the refusal must not inherit that.

    Mutation, measured for #2284: appending `get_bc_type_string(boundary_conditions)` to
    `refuse_mixed_per_axis` -- the split semantically undone -- kills this test and only this one.
    Measured in #2288 over the 17 files matching
    `grep -rlE 'semi_lagrangian|bc_utils|checked_bc_type_string|geometric_operations' tests/`:
    1 failed, 361 passed, 7 xfailed. Anchored to the PR rather than to a branch sha because this
    repository squash-merges, so a branch commit is not an ancestor of `main` and a reader greping
    history for it finds nothing.

    **The last two assertions pin an open defect, deliberately, and retire with it.** That
    `ValueError` is #1700 part B, which calls an empty segment list with a uniform default "a
    legitimate configuration" -- so it is a bug, not a contract, and the first two assertions are
    what this test is really for. They are two rather than one so that the absence has a cause: the
    lookup is checked directly, then the composite, and each carries the message true of its own
    trigger.
    """
    segment_free = BoundaryConditions(dimension=2, segments=[], default_bc=BCType.NO_FLUX)

    assert geometric_operations(segment_free) == {"reflect"}
    assert refuse_mixed_per_axis(segment_free, **CONSUMER) is None

    # The CAUSE, observed where it lives. #1700B is about `get_bc_type_string`, so that is what the
    # retirement condition has to watch. Asserting only the composite cannot separate "#1700 landed"
    # from "the composite stopped calling the lookup" -- measured for #2288: replacing
    # `checked_bc_type_string`'s body with `return None`, leaving `get_bc_type_string` untouched and
    # still raising, produced a byte-identical retirement message declaring #1700 had landed.
    with still_refused("only valid for uniform BCs", _RETIRE_1700B, ValueError):
        get_bc_type_string(segment_free)

    # ...and that the composite still routes through it, which is the other cause and its own message.
    with still_refused("only valid for uniform BCs", _LOOKUP_NO_LONGER_REACHED, ValueError):
        checked_bc_type_string(segment_free, **CONSUMER)


def test_hjb_sl_refuses_the_rename_signature_at_construction_2284():
    """The wiring, not the helper -- and the defect the consolidation actually closed.

    `HJBSemiLagrangianSolver.__init__` carried its own copy of the collapse predicate until #2284.
    The copy read `getattr(bc, "default_bc", None)`, so on the #1691 rename signature -- `segments`
    present, `default_bc` renamed away -- it treated the absence as "no default" and constructed,
    while `geometric_operations` refuses to guess and raises. The per-axis disagreement carried by
    the renamed field was invisible to construction.

    Mutation, measured for #2284: restoring the inline block in `__init__` (its own `_sl_ops` set
    over `segments` plus `getattr(bc, "default_bc", None)`) kills this test and only this test.
    Asserting on `refuse_mixed_per_axis` alone would not -- those assertions stay green with the
    constructor reverted, which is why this one builds the solver.
    """
    from mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian import HJBSemiLagrangianSolver
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid

    class RenamedBC:
        """Every segment agrees, so only the renamed default carries the disagreement."""

        def __init__(self):
            self.segments = [
                _seg("wx", BCType.NO_FLUX, "x_min"),
                _seg("ex", BCType.NO_FLUX, "x_max"),
                _seg("wy", BCType.NO_FLUX, "y_min"),
                _seg("ey", BCType.NO_FLUX, "y_max"),
            ]
            self.default = BCType.PERIODIC  # not `default_bc`
            self.dimension = 2

        def get_bc_type_string(self):
            return "no_flux"

    # The premise: the segments alone do NOT disagree, so a segments-only union sees nothing.
    duck = RenamedBC()
    assert len({seg.bc_type for seg in duck.segments}) == 1

    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[6, 6], boundary_conditions=no_flux_bc(dimension=2)
    )
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0))
    problem = MFGProblem(
        geometry=grid,
        T=0.2,
        Nt=2,
        sigma=0.1,
        components=MFGComponents(hamiltonian=H, u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0),
    )

    class _WithRenamedBC(HJBSemiLagrangianSolver):
        def get_boundary_conditions(self):
            return duck

    with pytest.raises(AttributeError, match="has 'segments' but no 'default_bc'"):
        _WithRenamedBC(problem=problem)

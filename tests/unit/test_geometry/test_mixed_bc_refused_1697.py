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
)

CONSUMER = {"consumer": "TestSolver", "alternative": "Use one BC type across axes."}


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


def test_a_renamed_attribute_fails_loudly_rather_than_disabling_the_guard():
    """The guard's failure mode must not be silence.

    Reading ``segments``/``default_bc`` through ``getattr(..., None)`` would turn a rename into an
    empty operation set, which reads as 'nothing disagrees' and makes every caller's guard a no-op
    -- the Issue #1691 shape. Direct attribute access is deliberate.
    """

    bc = BoundaryConditions(
        dimension=2,
        default_bc=BCType.PERIODIC,
        segments=[_seg("w", BCType.NO_FLUX, "x_min")],
    )
    assert geometric_operations(bc) == {"reflect", "periodic"}, "precondition: this BC disagrees"

    del bc.segments  # stands in for the field having been renamed

    with pytest.raises(AttributeError, match="segments"):
        geometric_operations(bc)


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


def test_half_a_renamed_pair_raises_rather_than_under_reporting():
    """One field present and the other missing is the signature of a rename.

    Degrading to an empty set here would report "nothing disagrees" for a BC that does, which is
    strictly worse than crashing -- the guard would be silently disabled for every caller.
    """
    from types import SimpleNamespace

    with pytest.raises(AttributeError, match="default_bc"):
        geometric_operations(SimpleNamespace(segments=[_seg("w", BCType.NO_FLUX, "x_min")]))

    with pytest.raises(AttributeError, match="segments"):
        geometric_operations(SimpleNamespace(default_bc=BCType.PERIODIC))

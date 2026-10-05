"""The SL pair reads a BC face by face, because the one-type accessor collapses a mixed BC (#1560, #1697).

``get_bc_type_string`` returns the FIRST segment's type: ``BoundaryConditions.type`` deliberately
raises ``ValueError`` for a mixed BC, ``bc_utils`` swallows that raise, and execution falls through
to ``segments[0].bc_type``. The one signal that the BC is mixed is discarded, and the fold then
applies that single operation to every axis.

Two levers produce the collapse, and a guard that handles only the first is insufficient:

1. **Segment order.** Reordering the list flips the surviving type.
2. **``default_bc``.** ``get_bc_type_string`` never reads it, so a partially-covering segment list
   plus a differing default collapses identically **with no permutation available**. A guard that
   unions only over ``segments`` lets this straight through.

HJB-SL and FP-SL therefore do not read that accessor for a segmented BC. `bc_utils.per_axis_operations`
reads each face, gives each axis its own operation, and refuses only an axis whose two faces disagree,
so neither lever reaches them. The channel the collapse used to break is pinned by
`tests/unit/test_alg/test_sl_channel_separates_by_axis_1560_1697.py`.
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
    geometric_operations,
    get_bc_type_string,
    per_axis_operations,
)

CONSUMER = {"consumer": "TestSolver"}

_OBSERVED_1700B = "`get_bc_type_string` no longer raises on a segment-free BC."

_CAUSES_1700B = {
    "#1700 part B landed": (
        "which this assertion exists to notice -- that issue calls the configuration legitimate, so "
        "the ValueError is a defect and its removal is progress. Do NOT restore the raise. Delete, in "
        "this order: (1) this assertion and its comment block; (2) the docstring paragraph beginning "
        '"The last assertion pins an open defect"; (3) in the docstring\'s SECOND paragraph, the '
        'clause ", the very lookup that raises `ValueError` on it" and rewrite what remains. '
        "Everything else stays: the per-axis owner reads faces and never depended on the ValueError "
        "existing"
    ),
    # NOT "renamed": renaming `get_bc_type_string` aborts collection, so the pin could never print
    # that cause -- and naming a cause the pin cannot observe is what this fixture exists to prevent,
    # caught in the #2290 review inside the diff that introduced the fixture (#2288).
    #
    # What aborts is this module's own `from mfgarchon.geometry.boundary.bc_utils import (...)`, not
    # the boundary package's re-export: `boundary/__init__.py` resolves it through a lazy
    # `__getattr__` that is never consulted. Traced by applying the rename and reading the traceback,
    # which is the only way any version of this comment has been right: the one frame is that import
    # statement, and conftest loads.
    #
    # Until #1756 the abort came earlier and took the whole session with it: conftest's import of
    # `mfgarchon` reached the name eagerly through `fp_semi_lagrangian.py`, which #1756 deleted.
    #
    # Two earlier versions of this comment named chains with hops that do not exist -- one of them
    # written to correct the other, and taken from a review rather than re-run. Both got the
    # conclusion right and the mechanism wrong, in the comment explaining why unobserved causal
    # claims are refused. Corrected in #2290 from the traceback above.
    "the lookup was re-routed internally, still importable under this name": (
        "the refusal is owed by whatever now resolves a segment-free BC; re-point this assertion"
    ),
}


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

    # What `geometric_operations` sees, because it also unions default_bc:
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
def test_an_axis_whose_faces_disagree_is_refused(bc_factory):
    """Each BC here leaves one axis periodic on one face only -- the default covers the other face, and
    a fold has one rule per axis. It is refused rather than resolved by whichever face is read first."""
    with pytest.raises(NotImplementedError, match="the two faces of an axis must agree"):
        per_axis_operations(bc_factory(), 2, **CONSUMER)


@pytest.mark.parametrize(
    ("bc_factory", "expected"),
    [
        pytest.param(lambda: no_flux_bc(dimension=2), ("reflect", "reflect"), id="uniform-no-flux"),
        pytest.param(lambda: periodic_bc(dimension=2), ("periodic", "periodic"), id="uniform-periodic"),
        pytest.param(
            lambda: BoundaryConditions(
                dimension=2, default_bc=BCType.NO_FLUX, segments=[_seg("w", BCType.NO_FLUX, "x_min")]
            ),
            ("reflect", "reflect"),
            id="segments-agree-with-default",
        ),
        pytest.param(
            lambda: BoundaryConditions(
                dimension=2,
                default_bc=BCType.NO_FLUX,
                segments=[_seg("w", BCType.NO_FLUX, "x_min"), _seg("n", BCType.NEUMANN, "y_min")],
            ),
            ("reflect", "reflect"),
            id="different-bctypes-same-operation",
        ),
        pytest.param(lambda: _x_walls_y_periodic(walls_first=True), ("reflect", "periodic"), id="channel"),
        pytest.param(lambda: _x_walls_y_periodic(walls_first=False), ("reflect", "periodic"), id="channel-reordered"),
        pytest.param(
            lambda: BoundaryConditions(
                dimension=2,
                default_bc=BCType.PERIODIC,
                segments=[_seg(f"x{s}", BCType.NO_FLUX, f"x_{s}") for s in ("min", "max")],
            ),
            ("reflect", "periodic"),
            id="walls-over-a-periodic-default",
        ),
        pytest.param(lambda: None, ("clamp", "clamp"), id="none"),
    ],
)
def test_each_axis_gets_the_operation_its_faces_ask_for(bc_factory, expected):
    """The refusal keys on an axis's two faces disagreeing -- not on segments, not on BCType identity, and
    not on the BC asking for more than one operation.

    ``NEUMANN`` and ``NO_FLUX`` are distinct BCTypes that map to the same geometric operation; refusing
    them would be a false positive on a legitimate wall. The channel is answered the same in both segment
    orders, which is lever 1 gone, and the walls laid over a periodic default are read face by face, which
    is lever 2 gone. ``None`` carries no faces and keeps the one operation the accessor gives it.
    """
    assert per_axis_operations(bc_factory(), 2, **CONSUMER) == expected


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


def _x_walls_y_periodic(walls_first=True):
    """Every face named, each axis whole: one geometric operation per axis, two across the boundary."""
    walls = [_seg(f"x{s}", BCType.NO_FLUX, f"x_{s}") for s in ("min", "max")]
    seam = [_seg(f"y{s}", BCType.PERIODIC, f"y_{s}") for s in ("min", "max")]
    return BoundaryConditions(
        dimension=2, default_bc=BCType.NO_FLUX, segments=walls + seam if walls_first else seam + walls
    )


def test_fp_sl_solver_refuses_a_half_periodic_axis_at_solve_time():
    """The wiring, not the helper.

    Asserting on ``per_axis_operations`` alone does not pin that FPSLSolver *calls* it at solve time --
    those assertions stay green with the solver's own check removed. This constructs the solver and
    solves.

    The BC is swapped in on the geometry after construction on purpose: FPSLSolver caches only an
    explicitly-passed BC and otherwise resolves the geometry live at each point of use, so a
    construction-time check alone would be bypassed here exactly as on the HJB side (#1560).

    The message is matched because since #2467 the velocity reads U's BC per face and refuses a
    half-periodic axis on its own: with the solver's check removed this BC still raises, only with that
    other message.
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
    with pytest.raises(NotImplementedError, match="FPSLSolver: axis 1 asks for"):
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

    # It has no face reader, so it cannot be read per axis -- and must fail rather than collapse.
    with pytest.raises(AttributeError):
        per_axis_operations(duck, 2, **CONSUMER)


def test_an_object_carrying_neither_field_is_not_a_segmented_bc():
    """A legacy BC has no per-axis information; it must not raise, and must not be refused."""
    from types import SimpleNamespace

    assert geometric_operations(SimpleNamespace(type="periodic")) == set()
    assert geometric_operations(None) == set()


def test_the_per_axis_owner_does_not_read_the_lookup_2284(still_refused):
    """A segment-free BC reaches the SL pair as its default's operation on every axis.

    It asks for one operation on every face, and `per_axis_operations` reads it face by face, so it
    answers without `get_bc_type_string`, the very lookup that raises `ValueError` on it. Until
    #2284 the solvers' guard and that lookup were one function, and a caller that wanted only the
    refusal inherited the raise; reading faces keeps the two apart for good.

    **The last assertion pins an open defect, deliberately, and retires with it.** That `ValueError`
    is #1700 part B, which calls an empty segment list with a uniform default "a legitimate
    configuration" -- so it is a bug, not a contract, and the first two assertions are what this test
    is really for. It is kept so that the second has a cause on record: the per-axis owner passes on a
    BC the lookup still refuses.
    """
    segment_free = BoundaryConditions(dimension=2, segments=[], default_bc=BCType.NO_FLUX)

    assert geometric_operations(segment_free) == {"reflect"}
    assert per_axis_operations(segment_free, 2, **CONSUMER) == ("reflect", "reflect")

    # The CAUSE, observed where it lives. #1700B is about `get_bc_type_string`, so that is what the
    # retirement condition has to watch.
    with still_refused(
        "only valid for uniform BCs",
        observed=_OBSERVED_1700B,
        causes=_CAUSES_1700B,
        exc_type=ValueError,
    ):
        get_bc_type_string(segment_free)


def test_hjb_sl_refuses_the_rename_signature_at_construction_2284(monkeypatch):
    """The wiring, not the helper -- and the defect the consolidation actually closed.

    `HJBSemiLagrangianSolver.__init__` carried its own copy of the collapse predicate until #2284.
    The copy read `getattr(bc, "default_bc", None)`, so on the #1691 rename signature -- `segments`
    present, `default_bc` renamed away -- it treated the absence as "no default" and constructed,
    while `geometric_operations` refuses to guess and raises. The per-axis disagreement carried by
    the renamed field was invisible to construction.

    Mutation, measured for #2284: restoring the inline block in `__init__` (its own `_sl_ops` set
    over `segments` plus `getattr(bc, "default_bc", None)`) kills this test and only this test.
    Asserting on `geometric_operations` alone would not -- those assertions stay green with the
    constructor reverted, which is why this one builds the solver. The constructor now reaches it
    through its per-axis read, `per_axis_operations`.

    The #1936 refusal of a Neumann value is lifted here: it runs first, and on this malformed BC its
    predicate raises the same AttributeError, which satisfied this test with the constructor's own BC
    read removed (measured in #2461's review).
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
        volatility=0.1,
        components=MFGComponents(hamiltonian=H, u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0),
    )

    monkeypatch.setattr(HJBSemiLagrangianSolver, "honors_inhomogeneous_neumann", True)

    class _WithRenamedBC(HJBSemiLagrangianSolver):
        def get_boundary_conditions(self):
            return duck

    with pytest.raises(AttributeError, match="has 'segments' but no 'default_bc'"):
        _WithRenamedBC(problem=problem)

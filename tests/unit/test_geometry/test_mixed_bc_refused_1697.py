"""The SL pair reads a BC face by face, because the one-type accessor collapses a mixed BC (#1560, #1697).

Until #2512 row B1, ``get_bc_type_string`` returned the FIRST segment's type: ``BoundaryConditions.type``
deliberately raises ``ValueError`` for a mixed BC, ``bc_utils`` swallowed that raise, and execution fell
through to ``segments[0].bc_type``. The one signal that the BC is mixed was discarded, and the fold then
applied that single operation to every axis. The accessor now passes the raise on.

Two levers produced the collapse, and a guard that handles only the first is insufficient:

1. **Segment order.** Reordering the list flipped the surviving type.
2. **``default_bc``.** ``get_bc_type_string`` never read it, so a partially-covering segment list
   plus a differing default collapsed identically **with no permutation available**. A guard that
   unions only over ``segments`` lets this straight through.

HJB-SL and FP-SL therefore do not read that accessor for a segmented BC. `bc_utils.per_axis_operations`
reads each face, gives each axis its own operation, and refuses only an axis whose two faces disagree,
so neither lever reaches them. The channel the collapse used to break is pinned by
`tests/unit/test_alg/test_sl_channel_separates_by_axis_1560_1697.py`.
"""

from __future__ import annotations

from functools import partial

import pytest

import numpy as np

from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    bc_utils,
    no_flux_bc,
    periodic_bc,
)
from mfgarchon.geometry.boundary.bc_utils import (
    geometric_operations,
    get_bc_type_string,
    per_axis_operations,
)

CONSUMER = {"consumer": "TestSolver"}


def _seg(name, bc_type, boundary):
    return BCSegment(name=name, bc_type=bc_type, boundary=boundary)


def _two_segments(first_periodic=False):
    order = [BCType.PERIODIC, BCType.NO_FLUX] if first_periodic else [BCType.NO_FLUX, BCType.PERIODIC]
    return BoundaryConditions(
        dimension=2,
        default_bc=order[0],
        segments=[_seg("a", order[0], "x_min"), _seg("b", order[1], "y_min")],
    )


def test_the_accessor_passes_on_the_raise_that_marks_a_bc_mixed():
    """The root cause, closed where it lived (#2512 row B1): the raise that marks a BC mixed is the answer.

    Until then the accessor swallowed it and returned the first segment's type, which is why this pin
    exists: a 'fix' that only patched a solver would have left it green.
    """
    mixed = _two_segments()

    with pytest.raises(ValueError, match="only valid for uniform BCs"):
        _ = mixed.type
    with pytest.raises(ValueError, match="only valid for uniform BCs"):
        get_bc_type_string(mixed)


def test_segment_order_no_longer_reaches_the_accessor():
    """Lever 1, the form the issue was originally filed against: either order is refused, not answered."""
    for first_periodic in (False, True):
        with pytest.raises(ValueError, match="only valid for uniform BCs"):
            get_bc_type_string(_two_segments(first_periodic=first_periodic))


def test_default_bc_is_a_second_lever_with_no_permutation_available():
    """Lever 2, and the reason a segments-only guard is insufficient.

    Neither BC below is a reordering of the other -- each has exactly one segment -- yet they
    collapsed to different operations until #2512 row B1. A guard unioning only over ``segments`` sees a single element
    in both cases and permits them.
    """
    a = BoundaryConditions(dimension=2, default_bc=BCType.PERIODIC, segments=[_seg("w", BCType.NO_FLUX, "x_min")])
    b = BoundaryConditions(dimension=2, default_bc=BCType.NO_FLUX, segments=[_seg("p", BCType.PERIODIC, "y_min")])

    # Until #2512 row B1 the accessor answered "no_flux" and "periodic" here; neither is the BC's type.
    for bc in (a, b):
        with pytest.raises(ValueError, match="only valid for uniform BCs"):
            get_bc_type_string(bc)

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
    ("bc_factory", "refusal"),
    [
        pytest.param(
            lambda: BoundaryConditions(
                dimension=2,
                default_bc=BCType.NO_FLUX,
                domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
                segments=[
                    BCSegment(name=f"y{s}", bc_type=BCType.PERIODIC, boundary=f"y_{s}", region={"x": (0.0, 0.5)})
                    for s in ("min", "max")
                ],
            ),
            "covers part of the",
            id="seam-on-half-of-each-face",
        ),
        pytest.param(
            lambda: BoundaryConditions(
                dimension=2,
                default_bc=BCType.PERIODIC,
                domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
                segments=[
                    BCSegment(name=f"x{s}", bc_type=BCType.NO_FLUX, boundary=f"x_{s}", region={"y": (0.25, 0.75)})
                    for s in ("min", "max")
                ],
            ),
            "covers part of the",
            id="partial-barrier-on-a-seam",
        ),
        pytest.param(
            lambda: BoundaryConditions(
                dimension=2,
                default_bc=BCType.PERIODIC,
                segments=[_seg(f"x{s}", BCType.NO_FLUX, f"X_{s.upper()}") for s in ("min", "max")],
            ),
            "names no face of a box domain",
            id="walls-misspelt-over-a-periodic-default",
        ),
    ],
)
def test_a_segment_the_face_reader_cannot_place_is_refused_in_a_mix(bc_factory, refusal):
    """The per-axis read takes each face's operation from the face owner (#2512 row B1), which refuses what it
    cannot represent. A ``region``-restricted segment covering part of a face with a different operation is
    refused: a periodic seam on half of each face would otherwise be solved as the full channel, and a
    barrier on part of a seam as a full wall (#2490). A segment whose ``boundary`` names no face of the
    domain is refused rather than dropped, so walls spelt ``"X_MIN"`` over a periodic default are not solved
    as a fully periodic domain. (``boundary="all"`` is every face, so a seam on "all" under x walls is the
    channel; that case is in the next test.)"""
    with pytest.raises(ValueError, match=refusal):
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
        pytest.param(
            lambda: BoundaryConditions(
                dimension=2,
                default_bc=BCType.NO_FLUX,
                domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
                segments=[BCSegment(name="n", bc_type=BCType.NEUMANN, boundary="x_min", region={"y": (0.0, 0.5)})],
            ),
            ("reflect", "reflect"),
            id="part-of-a-face-with-one-operation",
        ),
        pytest.param(
            lambda: BoundaryConditions(
                dimension=2,
                default_bc=BCType.NO_FLUX,
                segments=[
                    BCSegment(name="seam", bc_type=BCType.PERIODIC, boundary="all", priority=0),
                    *(
                        BCSegment(name=f"x{s}", bc_type=BCType.NO_FLUX, boundary=f"x_{s}", priority=1)
                        for s in ("min", "max")
                    ),
                ],
            ),
            ("reflect", "periodic"),
            id="seam-everywhere-walls-over-it",
        ),
        pytest.param(lambda: None, ("clamp", "clamp"), id="none"),
    ],
)
def test_each_axis_gets_the_operation_its_faces_ask_for(bc_factory, expected):
    """An axis is refused when its two faces disagree, and in a mix of operations when a segment's face
    cannot be told or covers part of one -- never for having segments, for BCType identity, or for asking
    for a different operation on each axis.

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


def _sl_problem(bc, dim=2, n=9, nt=4):
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


@pytest.mark.parametrize("solver_name", ["HJBSemiLagrangianSolver", "FPSLSolver"])
@pytest.mark.parametrize("when", ["construction", "solve"])
def test_a_half_periodic_axis_is_refused_where_each_solver_reads_it(solver_name, when):
    """The wiring, not the helper: each solver reads its BC per axis when it is built and again when it
    solves.

    Asserting on ``per_axis_operations`` alone does not pin that a solver calls it -- those assertions
    stay green with every call site removed. This builds each solver, and solves.

    For ``solve`` the BC is swapped in on the geometry after construction on purpose: both solvers
    resolve the geometry live at each point of use unless a BC was passed explicitly, so a
    construction-time read alone would be bypassed (#1560).

    The message is matched because since #2467 the velocity reads U's BC per face and refuses a
    half-periodic axis on its own: with a solver's solve-time read removed, the solve still raises, with
    that other message.
    """
    from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
    from mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian import HJBSemiLagrangianSolver

    solver_cls = {"HJBSemiLagrangianSolver": HJBSemiLagrangianSolver, "FPSLSolver": FPSLSolver}[solver_name]
    refusal = f"{solver_name}: axis 1 asks for"

    if when == "construction":
        _, problem = _sl_problem(_two_segments())
        with pytest.raises(NotImplementedError, match=refusal):
            solver_cls(problem)
        return

    grid, problem = _sl_problem(no_flux_bc(dimension=2))
    solver = solver_cls(problem)

    # ONLY the geometry's BC is replaced. Writing solver.boundary_conditions as well would pass
    # even if the solver never re-read anything -- review of #1702 found exactly that: the test
    # asserted a re-read mechanism the FP solver did not have, and passed by poking the cache.
    grid._boundary_conditions = _two_segments()

    if solver_cls is FPSLSolver:
        solve = partial(solver.solve_fp_system, M_initial=np.ones((9, 9)) / 81.0, potential_field=np.zeros((5, 9, 9)))
    else:
        solve = partial(solver.solve_hjb_system, np.ones((5, 9, 9)), np.zeros((9, 9)), np.zeros((5, 9, 9)))
    with pytest.raises(NotImplementedError, match=refusal):
        solve()


def test_fp_sl_solver_still_solves_a_uniform_bc():
    """The refusal must not cost the ordinary case."""
    from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver

    _, problem = _sl_problem(no_flux_bc(dimension=2))
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

    # It lacks the surface the per-axis read uses (`is_uniform`, `get_bc_type_at_boundary`), so it cannot
    # be read per axis -- and must fail rather than collapse.
    with pytest.raises(AttributeError):
        per_axis_operations(duck, 2, **CONSUMER)


def test_an_object_carrying_neither_field_is_not_a_segmented_bc():
    """A legacy BC has no per-axis information; it must not raise, and must not be refused."""
    from types import SimpleNamespace

    assert geometric_operations(SimpleNamespace(type="periodic")) == set()
    assert geometric_operations(None) == set()


def test_the_per_axis_owner_does_not_read_the_lookup_2284(monkeypatch):
    """A segment-free BC reaches the SL pair as its default's operation on every axis.

    It asks for one operation on every face, and `per_axis_operations` reads it face by face, so it
    answers without `get_bc_type_string`. Until #2284 the solvers' guard and that lookup were one
    function, and a caller that wanted only the refusal inherited the lookup's answer; reading faces
    keeps the two apart for good. The lookup is made to raise here, so a per-axis read that went back
    through it would fail this test.

    The lookup itself answers a segment-free BC with its default's type since #2512 row B1: every face
    imposes the default, so the face owner calls it uniform. That is #1700 part B, which calls the
    configuration legitimate.
    """
    segment_free = BoundaryConditions(dimension=2, segments=[], default_bc=BCType.NO_FLUX)
    assert get_bc_type_string(segment_free) == "no_flux"

    def raising(_bc):
        raise AssertionError("per_axis_operations read get_bc_type_string")

    monkeypatch.setattr(bc_utils, "get_bc_type_string", raising)
    assert geometric_operations(segment_free) == {"reflect"}
    assert per_axis_operations(segment_free, 2, **CONSUMER) == ("reflect", "reflect")


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

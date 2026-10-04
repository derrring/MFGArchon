"""Every (applicator, BCType) cell must either apply the condition or refuse it. Never be silent.

The cell count is **derived from the enums**, not listed. Adding a `BCType` member makes this file
fail until someone decides what each applicator does with it — which is the property a hand-written
list of known cases cannot have, and the reason the discrimination ratchet's mutations (declared in
`scripts/test_discrimination.py`'s `MUTATIONS`) reach only some of `BCType`'s 8 members. How many is
deliberately NOT stated, because no predicate for it is written down anywhere and the answer moves
with the one you pick: `BCType.X` matched literally over `repr(MUTATIONS)` returns 0, the lowercase
spellings the mutations actually use return 5 (`neumann`, `no_flux`, `periodic`, `robin`, and
`reflect` for REFLECTING), and a reviewer counting by hand returned 4. `absorbing` appears there too
and is not a `BCType` member. A figure with three answers and no owner is what #2349 is about; the 8
has an owner -- it is derived from the enum by this file's own parametrisation. #1948

`GraphApplicator` is indexed by its own alphabet and over TWO axes, (`GraphBCType` × `field_type`),
because THREE of its five arms branch on the field type inside themselves — DIRICHLET pins only the
value field, ABSORBING does opposite things to the two halves, SOURCE injects only into the density.
NEUMANN and CUSTOM do not, so four of the ten cells are two labels over one code path; that is the
honest cost of indexing by the specification rather than by the branch, and it is recorded here
rather than left for a reader to discover. Both axes are derived, the second from `apply`'s own
`Literal`, so the growth property above holds for a new field type as well as a new member.

What a declaration asserts: that a branch exists. Whether that branch is *correct* is a separate
axis, per cell, tracked at #1946 — `FDMApplicator` handles both `EXTRAPOLATION_*` in the sense this
file measures while writing unset memory on the uniform path.
"""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

import pytest

import numpy as np
import numpy.typing as npt

from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
from mfgarchon.geometry.boundary.applicator_fdm import FDMApplicator
from mfgarchon.geometry.boundary.applicator_graph import GraphApplicator, GraphBCType, NodeBC
from mfgarchon.geometry.boundary.applicator_implicit import ImplicitApplicator
from mfgarchon.geometry.boundary.applicator_interpolation import InterpolationApplicator
from mfgarchon.geometry.boundary.applicator_meshfree import MeshfreeApplicator
from mfgarchon.geometry.boundary.applicator_particle import ParticleApplicator
from mfgarchon.geometry.implicit.hypersphere import Hypersphere

if TYPE_CHECKING:
    from collections.abc import Callable

_CENTRE = np.array([0.5, 0.5])
_RADIUS = 0.4


def _cloud_with_interior() -> np.ndarray:
    """Three rings plus the centre.

    A boundary-only ring is degenerate for this machinery: every boundary point's nearest interior
    neighbour is itself, the normal interpolation collapses to a no-op, and `NO_FLUX` reads as
    silent when it is not. The first version of this table was built on such a ring and misreported
    one cell of twenty-four. `test_the_cloud_has_interior` pins the property.
    """
    points = [list(_CENTRE)]
    for radius in (0.13, 0.26, _RADIUS):
        for angle in np.linspace(0.0, 2.0 * np.pi, 17)[:-1]:
            points.append([_CENTRE[0] + radius * np.cos(angle), _CENTRE[1] + radius * np.sin(angle)])
    return np.array(points)


_CLOUD = _cloud_with_interior()
#: A radial ramp, so the normal derivative is non-zero and a correct zero-flux condition must move
#: the boundary. A constant field would make every flux condition indistinguishable from a no-op.
_RADIAL = 1.0 + 4.0 * np.linalg.norm(_CLOUD - _CENTRE, axis=1)
_LINE = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
#: Particle positions on [0, 1] with one outside each end, so a wall that does anything at all must
#: move or remove them. All-interior particles would make every condition indistinguishable from a
#: no-op, the same degeneracy `_cloud_with_interior` documents for the meshfree rows.
_PARTICLES = np.array([[-0.2], [0.5], [1.3]])
#: Node values on a 5-node path graph. Non-constant for the reason `_RADIAL` is.
_NODES = np.array([1.0, 2.0, 4.0, 7.0, 11.0])


def _uniform_bc(bc_type: BCType, dimension: int) -> BoundaryConditions:
    extra = {"alpha": 1.0, "beta": 1.0} if bc_type is BCType.ROBIN else {}
    return BoundaryConditions(
        segments=[BCSegment(name="s", bc_type=bc_type, value=2.5, **extra)],
        dimension=dimension,
        default_bc=bc_type,
    )


def _sphere() -> Hypersphere:
    return Hypersphere(center=_CENTRE, radius=_RADIUS)


#: (name, class, callable taking a BCType, the input field it is given).
_APPLICATORS = [
    (
        "FDMApplicator",
        FDMApplicator,
        lambda t: FDMApplicator(dimension=1).apply(_LINE.copy(), _uniform_bc(t, 1), grid_spacing=0.25),
        _LINE,
    ),
    (
        "MeshfreeApplicator",
        MeshfreeApplicator,
        lambda t: MeshfreeApplicator(_sphere()).apply(_RADIAL.copy(), _uniform_bc(t, 2), _CLOUD),
        _RADIAL,
    ),
    (
        "ImplicitApplicator",
        ImplicitApplicator,
        lambda t: ImplicitApplicator(_sphere(), boundary_tolerance=1e-8).apply(
            _RADIAL.copy(), _uniform_bc(t, 2), _CLOUD, spacing=0.13
        ),
        _RADIAL,
    ),
    (
        # No `apply` at all -- `enforce_values` is this applicator's only imposition entry point,
        # and it is the one `HJBSemiLagrangianSolver` calls on its `interp_bc_applicator`. Driving
        # it through `apply` would measure an absent method rather than the class.
        "InterpolationApplicator",
        InterpolationApplicator,
        lambda t: InterpolationApplicator(dimension=1).enforce_values(
            _LINE.copy(), _uniform_bc(t, 1), spacing=np.array([0.25])
        ),
        _LINE,
    ),
    (
        # The field here is particle POSITIONS, not samples of a field, and `apply` returns
        # (remaining, absorbed_mask, exit_positions). Element 0 is the surviving ensemble, so an
        # absorbing condition legitimately returns FEWER rows, which the assertion below tolerates
        # by comparing shapes before values. Silence still means the same thing here: the ensemble
        # came back untouched.
        "ParticleApplicator",
        ParticleApplicator,
        lambda t: ParticleApplicator().apply(_PARTICLES.copy(), _uniform_bc(t, 1), [(0.0, 1.0)])[0],
        _PARTICLES,
    ),
]

#: Cells that come back SILENT -- neither applying the condition nor refusing it. Marked
#: `xfail(strict=True)` VIA THE PARAMETRISATION (see `_param`), so fixing one turns this table red
#: for "unexpectedly passing" and forces the reader back here. It was an imperative `pytest.xfail()`
#: until 2026-09-09, which aborts before the assertion and so could never report XPASS -- the
#: promise this comment used to make was false for as long as it was written that way.
#:
#: Emptied by #1948 step 2. Its two entries were `InterpolationApplicator` x `EXTRAPOLATION_*`, recorded
#: as silent because "the order is already fixed" -- which was false: `extrapolation_order` is the
#: zero-Neumann extrapolation's order, and the two members simply fell through the dispatch with the
#: boundary value left as interpolated. The applicator now declares its types and refuses these two
#: (maintainer ruling 2026-10-04). Kept as an empty set so the next silent cell has somewhere honest
#: to go.
_KNOWN_SILENT: set[tuple[str, BCType]] = set()

#: Applicators that never adopted `_SUPPORTED_BC_TYPES`. `None`, or the attribute being absent,
#: means "not migrated" -- which
#: `test_a_declaration_that_is_absent_disables_the_gate_rather_than_failing_closed` establishes as a
#: deliberate convention (#1456) rather than a violation. The declaration half of the contract
#: cannot be asserted for them; the SILENCE half still is.
#:
#: The exemption ASSERTS ITS OWN PREMISE rather than promising a retirement in prose. A name-keyed
#: `if` around the assertion is skipped forever once an applicator migrates -- measured on the first
#: version of this change: giving `ParticleApplicator` a `_SUPPORTED_BC_TYPES` left the table at
#: 43 passed / 10 xfailed, tripping nothing. That is the same promise-without-a-mechanism defect
#: this file fixes for `_KNOWN_SILENT`, and it does not get to reappear two definitions later.
_UNDECLARED_APPLICATORS: set[str] = {
    "ParticleApplicator",
}

_CELLS = [(name, cls, call, field, bc_type) for name, cls, call, field in _APPLICATORS for bc_type in BCType]


def _param(name, cls, call, field, bc_type):
    """A known-silent cell is marked `xfail(strict=True)`, NOT skipped at runtime.

    The distinction is the whole retirement mechanism and it is not cosmetic. Measured against this
    repository's own `pytest.ini` (`xfail_strict = true`, #1663):

        @pytest.mark.xfail  on a test that passes  ->  FAILED [XPASS(strict)]
        pytest.xfail(...)   on a test that passes  ->  xfailed, silently

    `pytest.xfail()` is imperative: it aborts the test at the call, so the assertion below never
    runs and the cell can never report XPASS. An exemption written that way is permanent and
    silent. This file previously used it while its own comment promised that "fixing one turns this
    table red for unexpectedly passing" -- the promise and the mechanism disagreed, which is the
    shape `test_the_refusing_branch_still_refuses_if_the_gate_is_bypassed` warns about in its own
    docstring: a pin that quietly stops pinning.
    """
    marks = (
        [pytest.mark.xfail(strict=True, reason=f"{name} is silent on {bc_type.name}; #1948 step 2")]
        if (name, bc_type) in _KNOWN_SILENT
        else []
    )
    return pytest.param(name, cls, call, field, bc_type, marks=marks)


_PARAMS = [_param(*cell) for cell in _CELLS]


def _apply_and_classify(call, bc_type):
    """Return ("refused", message) or ("applied", array).

    Split out so the assertions live outside the `except` blocks; asserting inside one hides the
    original traceback when the assertion itself fails.
    """
    try:
        return "applied", np.asarray(call(bc_type))
    except NotImplementedError as exc:
        return "refused", str(exc)
    except (ValueError, TypeError) as exc:
        return "refused", f"{type(exc).__name__}: {exc}"


def test_the_cloud_has_interior():
    """Precondition for every meshfree row. Without it those rows measure the fixture."""
    applicator = ImplicitApplicator(_sphere(), boundary_tolerance=1e-8)
    on_boundary = applicator._detect_boundary_points(_CLOUD)

    assert on_boundary.any(), "no boundary points, so every meshfree row would be vacuous"
    assert (~on_boundary).any(), "no interior points: the normal interpolation degenerates to a no-op"

    from scipy.spatial import cKDTree

    nearest, _ = cKDTree(_CLOUD[~on_boundary]).query(_CLOUD[on_boundary])
    assert nearest.min() > 1e-6, "a boundary point's nearest interior neighbour is itself"


def test_the_product_is_derived_from_the_enum_and_not_listed():
    """If someone adds a `BCType` member, this count moves and the parametrised test below gains
    rows that nobody has decided about. That automatic growth is the whole mechanism.

    The graph product is derived the same way on both of its axes, so the same sentence holds for a
    new `GraphBCType` member and for a new `field_type` -- but it is pinned by
    `test_the_graph_field_type_axis_matches_what_this_table_decided` and NOT by a line here. The
    obvious `len(_GRAPH_CELLS) == len(list(GraphBCType)) * len(_GRAPH_FIELD_TYPES)` is true by
    construction of the comprehension that builds `_GRAPH_CELLS`, so no production change can redden
    it; measured, it passes while the product silently halves."""
    assert len(_CELLS) == len(_APPLICATORS) * len(list(BCType))


@pytest.mark.parametrize(
    ("name", "cls", "call", "field", "bc_type"),
    _PARAMS,
    ids=[f"{name}-{t.name}" for name, _, _, _, t in _CELLS],
)
def test_each_cell_either_applies_or_refuses(name, cls, call, field, bc_type):
    """The three outcomes an unhandled type produced across this family were: a silent no-op, a bare
    `pass`, and `else: 0.0` — plus one applicator that raised. Only the raise is right, and this
    asserts it for every cell.

    A silent no-op is indistinguishable at the call site from a correctly-applied condition that
    happened to change nothing, which is why it cannot be allowed even where it is harmless.
    """
    # `getattr` with a default, not attribute access: the convention has THREE states, not two.
    # `FDMApplicator` declares 8, `MeshfreeApplicator` 2, `ImplicitApplicator` 7;
    # `GraphApplicator` declares `None` ("not migrated", #1456), `InterpolationApplicator` 6 since #1948; and
    # `ParticleApplicator` has no such attribute at all. Bare `cls._SUPPORTED_BC_TYPES` raises
    # `AttributeError` on the third state, which reads as a test bug rather than as the missing
    # declaration it is.
    declared = bc_type in (getattr(cls, "_SUPPORTED_BC_TYPES", None) or frozenset())

    outcome, result = _apply_and_classify(call, bc_type)

    if outcome == "refused":
        assert not declared, f"{name} declares {bc_type.name} yet refused it: {result}"
        return

    if name in _UNDECLARED_APPLICATORS:
        assert getattr(cls, "_SUPPORTED_BC_TYPES", None) is None, (
            f"{name} has migrated to _SUPPORTED_BC_TYPES; remove it from _UNDECLARED_APPLICATORS and decide its cells"
        )
    else:
        assert declared, f"{name} does not declare {bc_type.name} but applied it without refusing"

    baseline = np.asarray(field)
    if result.shape == baseline.shape:
        assert not np.allclose(result, baseline), (
            f"{name} declares {bc_type.name} and returned the field unchanged. A caller cannot tell "
            f"that from a condition that was applied and happened to change nothing; declare it "
            f"unsupported, or make the branch do something."
        )


#: `GraphApplicator` gets its OWN product, against its OWN alphabet, over TWO axes. #1948 is
#: explicit about the first: "GraphApplicator is measured against GraphBCType, not BCType -- the
#: family distinction is real and the test must encode it rather than flatten it", and its opening
#: section puts the pairing beyond dispute: "GraphApplicator carrying its own GraphBCType (five
#: members, disjoint from BCType) is likewise correct ... None of that is in question."
#:
#: An earlier version of this file indexed it by `BCType` instead and recorded the eight resulting
#: cells as known gaps. That flattened exactly the distinction the issue asks the test to encode,
#: and it needed nine lines of comment to explain why eight `xfail(strict=True)` rows were not bugs
#: -- which is the tell that the encoding was wrong, not that the applicator was.
#:
#: The SECOND axis is `field_type`. THREE of the five arms of `GraphApplicator.apply` branch on it
#: inside themselves -- DIRICHLET pins only the value field (#1471), ABSORBING does opposite things
#: to the two halves (#1478), SOURCE injects only into the density -- so the dispatch surface is the
#: product and a single-`field_type` table sees one column of it. NEUMANN (a bare `pass`) and CUSTOM
#: (which branches on `callable(bc.value)`) do NOT read `field_type`, so their four cells are two
#: labels over one code path each. Ten cells cover seven distinct paths, and saying so here is
#: cheaper than letting the count read as coverage it does not have.
#:
#: Measured 2026-09-10 against the version indexed by `GraphBCType` alone, which called `apply` at
#: its default `field_type="value"`, three of its five rows did not say what their label claimed --
#: and the three do NOT share one cause, which is why the fix has two halves:
#:
#:   SOURCE     recorded silent    WRONG about the member: it applies on "density"   -- the axis
#:   DIRICHLET  recorded applying  INCOMPLETE: the verdict was true, the density      -- the axis
#:                                 half was silent, unmarked and out of view
#:   CUSTOM     recorded silent    WRONG about the member: it applies given the       -- the fixture
#:                                 callable its API documents (see `_graph_value`)
#:
#: The axis alone leaves six cells silent, not four; CUSTOM's arm never reads `field_type`, so only
#: the per-member fixture moves it. No reader of the old table could have told, in any of the three
#: cases: it was green either way.
#:
#: The cross-alphabet question is real and is not this table's: `BCType` and `GraphBCType` share the
#: `.value` strings {"dirichlet", "neumann"} while being different members, so a `BCType` handed to
#: `NodeBC` matches no branch and returns the field untouched. That belongs with the alphabet work
#: at #2292, not here.
def _graph_field_types() -> tuple[str, ...]:
    """The second axis, read from `apply`'s own `Literal` rather than listed here.

    Same property the first axis gets from `GraphBCType`: adding a field type makes this table grow
    rows nobody has decided about. A hand-written copy would be the incident log the module
    docstring rejects, and there is a live reason to distrust one -- #1940 was a `field_type` string
    that matched no branch and returned the field untouched.

    `localns` is required because `applicator_graph` keeps `NDArray` under `if TYPE_CHECKING` while
    `from __future__ import annotations` defers the whole signature, so `get_type_hints` cannot
    resolve the `field` parameter without being handed that name.
    """
    hints = typing.get_type_hints(GraphApplicator.apply, localns={"NDArray": npt.NDArray})
    return typing.get_args(hints["field_type"])


_GRAPH_FIELD_TYPES = _graph_field_types()


def _custom_node_bc(field: npt.NDArray[np.floating], node: int, t: float) -> float:
    """The callable `set_custom_bc` documents and constructs: `(field, node, t) -> float`."""
    del t
    return 0.5 * float(field[node])


def _graph_value(bc_type: GraphBCType) -> float | Callable[..., float]:
    """The `value` each member's dispatch arm can actually use: a callable for CUSTOM, a scalar else.

    "Its own API" would be too strong for one member: `set_absorbing_nodes` takes no `value` at all
    and hard-codes `0.0`, so the 2.5 below is not what an ABSORBING region is built with. Both
    verdicts are unchanged by it (the arm writes `value` on "value" and `0.0` on "density", and 2.5
    and 0.0 both differ from every node in `_NODES`), so this is stated rather than fixed.

    Not a convenience. `NodeBC.value` is annotated `float | Callable[[int, float], float]` and
    `NodeBC.get_value` calls it that way, while the CUSTOM branch of `apply` calls it as
    `(field, node, t)` -- the signature `set_custom_bc` documents and then constructs a `NodeBC`
    with. One field carries two incompatible callable contracts, selected by `bc_type`, which is
    #1948's "no shared contract" one level below the applicators.

    So the CUSTOM cell reports whichever of three outcomes the fixture chooses, from an applicator
    that never changes. Measured 2026-09-10: a scalar falls through the branch's `callable(bc.value)`
    guard and reads SILENT, the two-argument `(node, t)` form raises `TypeError` and reads REFUSED,
    and the documented three-argument form applies. Only the third is a fact about the class.
    """
    if bc_type is GraphBCType.CUSTOM:
        return _custom_node_bc
    # NEUMANN is zero flux on a graph, and a non-zero value is refused when the `NodeBC` is built
    # (`test_a_non_zero_graph_neumann_flux_is_refused`), so its cells are measured at the one value
    # the type admits.
    return 0.0 if bc_type is GraphBCType.NEUMANN else 2.5


_GRAPH_CELLS = [(t, ft) for t in GraphBCType for ft in _GRAPH_FIELD_TYPES]

#: Graph cells that leave the field unchanged ON PURPOSE are DECLARED on the class
#: (`GraphApplicator._DECLARED_IDENTITY`, with a reason each), not listed here. Until #1948 step 2 they
#: were four `xfail(strict=True)` entries whose only announcement was a source comment, so a caller
#: could not tell a deliberate no-op from an unhandled cell. The maintainer ruled (2026-10-04) that
#: these four are identity by design and must say so. The table holds the declaration to the
#: behaviour in both directions: a declared cell that changes the field fails, and so does an
#: undeclared cell that does not.
_GRAPH_DECLARED_IDENTITY = GraphApplicator._DECLARED_IDENTITY


#: The field types this table has DECIDED about. The axis itself stays derived, so the product
#: grows on its own; this is the direction deriving cannot give you. A derived population can shrink
#: as quietly as it grows, and shrinking reads as a clean run: measured 2026-09-10, narrowing the
#: annotation to `Literal["value"]` halves the graph product from ten cells to five and the file
#: reports `47 passed, 4 xfailed` -- no failure, and the two orphaned "density" entries then in
#: `_GRAPH_KNOWN_SILENT` (since #1948 step 2, `GraphApplicator._DECLARED_IDENTITY`) simply stop
#: matching anything.
_GRAPH_FIELD_TYPES_DECIDED = frozenset({"value", "density"})


def test_the_graph_field_type_axis_matches_what_this_table_decided():
    """Derived population, pinned content: growth is automatic, shrinkage is loud.

    Three failure modes, and the first two are why a bare `assert _GRAPH_FIELD_TYPES` is not enough:

        annotation -> `str`                 `get_args` returns () -- zero cells, nothing red
        annotation -> `Literal["value"]`    five cells instead of ten, nothing red
        annotation gains a member           five new cells nobody has decided about

    The equality catches all three and makes each of them someone's decision. What it does NOT catch
    is the opposite drift: `apply` states its vocabulary twice, once as the `Literal` this axis reads
    and once as the tuple in its own guard, and a member added to the GUARD alone leaves the axis and
    this pin unmoved. That direction is unclosed, and is stated rather than implied.
    """
    assert set(_GRAPH_FIELD_TYPES) == _GRAPH_FIELD_TYPES_DECIDED, (
        f"`apply`'s Literal is now {sorted(_GRAPH_FIELD_TYPES)} while this table has decided about "
        f"{sorted(_GRAPH_FIELD_TYPES_DECIDED)}. Growth adds cells nobody has ruled on; shrinkage "
        f"drops cells silently. Decide what the applicator does for each, then move this pin."
    )

    for field_type in _GRAPH_FIELD_TYPES:
        GraphApplicator(num_nodes=_NODES.size).apply(_NODES.copy(), field_type=field_type)

    with pytest.raises(ValueError, match="field_type must be"):
        GraphApplicator(num_nodes=_NODES.size).apply(_NODES.copy(), field_type="VALUE")


def test_no_graph_cell_refuses_by_accident():
    """A refusal must be deliberate. `_apply_and_classify` reports `ValueError` and `TypeError` as
    "refused" too, so without this a cell that starts raising for a BUG reason scores as compliant.

    Not hypothetical: `_graph_value` records the two-argument callable raising `TypeError` from
    inside the CUSTOM arm, which reads exactly like a branch that had been fixed to refuse.

    This is a sweep and not a line inside the parametrised cell above, and the difference is
    load-bearing. Until #1948 step 2, four of the ten cells carried `xfail(strict=True)`, which absorbs a
    failing assertion as an expected failure -- measured 2026-09-10: with the NEUMANN arm raising
    `TypeError`, the same assertion written inside the cell left the file at `50 passed, 6 xfailed`.
    Those four were the cells step 2 changed. No cell carries a mark now; the sweep stays the one
    place every cell's refusal is checked.

    `NotImplementedError` is the refusal the rest of this family raises and what #1948 asks for
    ("an unhandled type must raise with the type named"). `_apply_and_classify` returns its message
    bare and prefixes the type name only for the other two, so the absence of that prefix is the
    discriminator.
    """
    accidental = []
    for bc_type, field_type in _GRAPH_CELLS:
        applicator = GraphApplicator(num_nodes=_NODES.size).add_node_bc(
            NodeBC(nodes=[0, _NODES.size - 1], bc_type=bc_type, value=_graph_value(bc_type), name="s")
        )
        outcome, result = _apply_and_classify(
            lambda _t, _a=applicator, _f=field_type: _a.apply(_NODES.copy(), field_type=_f),
            bc_type,
        )
        if outcome == "refused" and result.startswith(("ValueError:", "TypeError:")):
            accidental.append(f"{bc_type.name}/{field_type}: {result}")

    assert not accidental, (
        "GraphApplicator refused these cells with an exception that reads as a bug rather than as a "
        "declared refusal; this table would otherwise score them compliant:\n  " + "\n  ".join(accidental)
    )


@pytest.mark.parametrize(
    ("bc_type", "field_type"),
    [pytest.param(t, ft) for (t, ft) in _GRAPH_CELLS],
    ids=[f"{t.name}-{ft}" for (t, ft) in _GRAPH_CELLS],
)
def test_each_graph_cell_either_applies_or_refuses(bc_type, field_type):
    """Same contract as the table above, over the alphabet and the field types this class speaks.

    Both axes are derived -- `GraphBCType` and `apply`'s own `Literal` -- so a new member of either
    raises the uncovered-cell count on its own. That is what neither list being hand-maintained
    buys, and it is the property #1948 asks for.
    """
    field = _NODES.copy()
    applicator = GraphApplicator(num_nodes=_NODES.size).add_node_bc(
        NodeBC(nodes=[0, _NODES.size - 1], bc_type=bc_type, value=_graph_value(bc_type), name="s")
    )
    outcome, result = _apply_and_classify(
        lambda _t: applicator.apply(field.copy(), field_type=field_type),
        bc_type,
    )

    if outcome == "refused":
        # Whether the refusal is a DELIBERATE one is asserted by
        # `test_no_graph_cell_refuses_by_accident`, which sweeps every cell in one place. It was split
        # out while four of these cells carried `xfail(strict=True)`, which would have absorbed a
        # check placed here (#1948 step 2 removed the marks).
        return

    if (bc_type, field_type) in _GRAPH_DECLARED_IDENTITY:
        np.testing.assert_array_equal(
            result,
            field,
            err_msg=f"GraphApplicator declares {bc_type.name}/{field_type} an identity and changed the field",
        )
        return

    assert not np.allclose(result, field), (
        f"GraphApplicator returned the field unchanged for GraphBCType.{bc_type.name} on "
        f"field_type={field_type!r}. A caller cannot tell that from a condition that was applied "
        f"and happened to change nothing; declare it in _DECLARED_IDENTITY with its reason, or refuse it."
    )


@pytest.mark.parametrize("value", [2.5, lambda node, t: 0.0], ids=["non_zero", "callable"])
def test_a_non_zero_graph_neumann_flux_is_refused(value):
    """Graph NEUMANN is zero flux; `apply` leaves both fields unchanged for it. A flux it would drop is
    refused when the `NodeBC` is built (maintainer ruling 2026-10-04). A callable cannot be checked for
    zero, and its value is never read, so it is refused too. Control: the zero flux builds."""
    NodeBC(nodes=[0], bc_type=GraphBCType.NEUMANN, value=0.0, name="wall")
    with pytest.raises(NotImplementedError, match="zero flux"):
        NodeBC(nodes=[0], bc_type=GraphBCType.NEUMANN, value=value, name="wall")


def test_a_declaration_that_is_absent_disables_the_gate_rather_than_failing_closed():
    """`_SUPPORTED_BC_TYPES = None` means "not migrated", matching
    `BaseSolver._validate_bc_support`'s convention for un-migrated solvers (#1456).

    Failing closed instead would make every applicator that has not declared refuse everything,
    which is a worse default than the silence it replaces.
    """

    class _Undeclared(FDMApplicator):
        _SUPPORTED_BC_TYPES = None

    assert _Undeclared(dimension=1).supported_bc_types is None
    _Undeclared(dimension=1).apply(_LINE.copy(), _uniform_bc(BCType.PERIODIC, 1), grid_spacing=0.25)


def test_the_gate_reads_default_bc_and_not_only_the_segments():
    """`_validate_bc_support` adds `bc.default_bc` to the requested set.

    Every cell above puts the same type in both the segment and `default_bc`, so dropping the
    `default_bc` line changes nothing there -- measured, that mutation passes the whole table. Here
    the segment is a type the applicator declares and the default is one it does not, so only a gate
    that reads both can refuse it.
    """
    bc = BoundaryConditions(
        segments=[BCSegment(name="s", bc_type=BCType.DIRICHLET, value=1.0)],
        dimension=2,
        default_bc=BCType.PERIODIC,  # ImplicitApplicator deliberately does not declare this
    )
    assert BCType.DIRICHLET in ImplicitApplicator._SUPPORTED_BC_TYPES
    assert BCType.PERIODIC not in ImplicitApplicator._SUPPORTED_BC_TYPES

    with pytest.raises(NotImplementedError, match="does not support"):
        ImplicitApplicator(_sphere(), boundary_tolerance=1e-8).apply(_RADIAL.copy(), bc, _CLOUD, spacing=0.13)


def test_the_refusing_branch_still_refuses_if_the_gate_is_bypassed():
    """Defence in depth, pinned explicitly because the gate makes it unreachable.

    `MeshfreeApplicator`'s NO_FLUX branch used to be a bare `pass` whose own comment said the
    boundary values "should match nearby interior" and then did nothing. It now raises. With the
    declaration withdrawn the gate refuses NO_FLUX first, so reverting the branch to `pass` passes
    the whole table -- measured. A correct fix made an existing target unreachable, which is how a
    pin quietly stops pinning.

    This calls with a subclass that declares NO_FLUX while inheriting the parent's `apply`, which is
    exactly the shape a future subclass would have.
    """

    class _DeclaresNoFlux(MeshfreeApplicator):
        _SUPPORTED_BC_TYPES = frozenset({BCType.DIRICHLET, BCType.ROBIN, BCType.NO_FLUX})

    with pytest.raises(NotImplementedError, match="derivative operators"):
        _DeclaresNoFlux(_sphere()).apply(_RADIAL.copy(), _uniform_bc(BCType.NO_FLUX, 2), _CLOUD)

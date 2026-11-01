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

import enum
import math
import numbers
import typing
from decimal import Decimal
from fractions import Fraction
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
    # `FDMApplicator` declares 8, `MeshfreeApplicator` 2, `ImplicitApplicator` 5;
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
#: (which applies its rule to either field) do NOT read `field_type`, so their four cells are two
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
#:                                 callable its API documents (see `_graph_condition`)
#:
#: The axis alone leaves six cells silent, not four; CUSTOM's arm never reads `field_type`, so only
#: the per-member fixture moves it. No reader of the old table could have told, in any of the three
#: cases: it was green either way.
#:
#: The cross-alphabet question is real and is not this table's: `BCType` and `GraphBCType` share the
#: `.value` strings {"dirichlet", "neumann"} while being different members, so a `BCType` handed to
#: `NodeBC` matched no branch and returned the field untouched. #2292 refuses it where it is handed
#: over (the acceptance gate below), and row N1's guard, `test_graph_bc_alphabet_has_one_owner_2512.py`,
#: holds each reader to one alphabet.
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


def _custom_node_bc(node: int, field: npt.NDArray[np.floating]) -> float:
    """A CUSTOM rule, a callable of `(t, node, field)` that declares no time, so the cells apply it with none (#2296)."""
    return 0.5 * float(field[node])


def _graph_condition(bc_type: GraphBCType) -> dict[str, float | Callable[..., float]]:
    """What each member's dispatch arm reads: CUSTOM a rule, of its own field, and the others a scalar value.

    "Its own API" would be too strong for one member: `set_absorbing_nodes` takes no `value` at all
    and hard-codes `0.0`, so the 2.5 below is not what an ABSORBING region is built with. Both
    verdicts are unchanged by it (the arm writes `value` on "value" and `0.0` on "density", and 2.5
    and 0.0 both differ from every node in `_NODES`), so this is stated rather than fixed.

    Measured 2026-09-10, one field carried both contracts: the CUSTOM cell read SILENT for a scalar, REFUSED for a
    `(node, t)` callable and applied for `(field, node, t)`, from an applicator that never changed. A CUSTOM condition
    now takes its rule as `rule=` and refuses a non-zero value (#2296).
    """
    if bc_type is GraphBCType.CUSTOM:
        return {"rule": _custom_node_bc}
    # NEUMANN is zero flux on a graph, and a non-zero value is refused when the `NodeBC` is built
    # (`test_a_non_zero_graph_neumann_flux_is_refused`), so its cells are measured at the one value
    # the type admits.
    return {"value": 0.0 if bc_type is GraphBCType.NEUMANN else 2.5}


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

    Not hypothetical: `_graph_condition` records a two-argument callable raising `TypeError` from
    inside the CUSTOM arm, which read exactly like a branch that had been fixed to refuse.

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
            NodeBC(nodes=[0, _NODES.size - 1], bc_type=bc_type, name="s", **_graph_condition(bc_type))
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
        NodeBC(nodes=[0, _NODES.size - 1], bc_type=bc_type, name="s", **_graph_condition(bc_type))
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


@pytest.mark.parametrize("value", [2.5, lambda t, node: 0.0], ids=["non_zero", "callable"])
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
    """The base gate, `BaseBCApplicator._validate_bc_support`, adds `bc.default_bc` to the requested set.

    Every cell above puts the same type in both the segment and `default_bc`, so dropping the
    `default_bc` line changes nothing there -- measured, that mutation passes the whole table. Here
    the segment is a type the applicator declares and the default is one it does not, so only a gate
    that reads both can refuse it. The meshfree and implicit applicators gate the one condition they impose
    instead (`the_one_condition`, #2717), which `test_implicit_applicator` pins; FDM keeps the base gate.
    """

    class _DirichletOnly(FDMApplicator):
        _SUPPORTED_BC_TYPES = frozenset({BCType.DIRICHLET})

    bc = BoundaryConditions(
        segments=[BCSegment(name="s", bc_type=BCType.DIRICHLET, value=1.0)],
        dimension=1,
        default_bc=BCType.PERIODIC,
    )
    with pytest.raises(NotImplementedError, match="does not support"):
        _DirichletOnly(dimension=1).apply(_LINE.copy(), bc, grid_spacing=0.25)


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


@pytest.mark.parametrize("bc_type", [BCType.EXTRAPOLATION_LINEAR, BCType.EXTRAPOLATION_QUADRATIC])
def test_the_implicit_applicator_refuses_an_extrapolation_by_gate_and_by_branch(bc_type):
    """`ImplicitApplicator` has no branch for either EXTRAPOLATION type. Both were declared and fell to
    `else: result = 0.0`, a homogeneous Dirichlet wall (#2717, review 2). Neither is declared now, so the gate refuses
    it; past the gate the `else` raises, which a subclass that declares them reaches. The table above cannot see
    this: it passes "declared and applied" as readily as "undeclared and refused" (#2717, review 3)."""

    class _DeclaresExtrapolation(ImplicitApplicator):
        _SUPPORTED_BC_TYPES = ImplicitApplicator._SUPPORTED_BC_TYPES | {bc_type}

    bc = _uniform_bc(bc_type, 2)
    with pytest.raises(NotImplementedError, match="does not support"):
        ImplicitApplicator(_sphere(), boundary_tolerance=1e-8).apply(_RADIAL.copy(), bc, _CLOUD, spacing=0.13)
    with pytest.raises(NotImplementedError, match="has no branch"):
        _DeclaresExtrapolation(_sphere(), boundary_tolerance=1e-8).apply(_RADIAL.copy(), bc, _CLOUD, spacing=0.13)


# ---------------------------------------------------------------------------------------------------------------------
# The acceptance gate (#2292, #2512 row N1). The table above measures `apply` on inputs a NodeBC already holds; this
# one measures the places a graph BC input is accepted (the population of #2292's class line, each acceptor by input
# class), so that an input is either taken as declared or refused by name where it is handed over -- never stored and
# dropped later. A hand-listed table: an acceptor added later is not in it until a row is.
# ---------------------------------------------------------------------------------------------------------------------


class _ForeignEnum(enum.Enum):
    """An enum that is no BC type, whose value a BC type also spells."""

    PERIODIC = "periodic"


def _maze_leaves() -> list[int]:
    """The 3 x 3 maze's dead ends, from its own adjacency, without the applicator."""
    return sorted(int(i) for i in np.where((_maze().get_adjacency_matrix() != 0).sum(axis=1) == 1)[0])


def _maze_box_field(value: float) -> np.ndarray:
    """The 3 x 3 maze's value field with ``value`` on every node its own positions put on the bounding box."""
    positions = np.asarray(_maze().get_node_positions(), dtype=float)
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    on_box = np.any(np.isclose(positions, lo) | np.isclose(positions, hi), axis=1)
    return np.where(on_box, value, 0.0)


def _maze(boundary_conditions=None):
    """A 3 x 3 maze, a graph geometry with no `network_data`."""
    from mfgarchon.geometry.graph.maze_generator import MazeGeometry

    return MazeGeometry(3, 3, seed=0, boundary_conditions=boundary_conditions)


def _gate_cases():
    """Each accept case returns True when the input was taken as declared; each refuse case names its error."""
    from mfgarchon.geometry.boundary.applicator_graph import (
        EdgeBC,
        GraphBCConfig,
        create_graph_applicator,
        create_maze_applicator,
    )
    from mfgarchon.geometry.boundary.bc_utils import bc_type_to_geometric_operation

    def applicator():
        return GraphApplicator(num_nodes=5)

    accepts, refuses = [], []
    # the type a condition is built with
    for t in GraphBCType:
        accepts.append(
            (f"NodeBC[{t.name}]", lambda t=t: NodeBC(nodes=[0], bc_type=t, **_graph_condition(t)).bc_type is t)
        )
    for t in BCType:
        refuses.append((f"NodeBC[BCType.{t.name}]", lambda t=t: NodeBC(nodes=[0], bc_type=t), TypeError, "GraphBCType"))
        accepts.append((f"BCSegment[{t.name}]", lambda t=t: BCSegment(name="s", bc_type=t, value=0.0).bc_type is t))
    for t in GraphBCType:
        refuses.append(
            (f"BCSegment[GraphBCType.{t.name}]", lambda t=t: BCSegment(name="s", bc_type=t), TypeError, "BCType")
        )
    refuses.append(("NodeBC[str]", lambda: NodeBC(nodes=[0], bc_type="dirichlet"), TypeError, "GraphBCType"))
    # the nodes a condition names, through every route that takes them: outside the graph, then not node indices
    routes = {
        "add_node_bc": lambda n: applicator().add_node_bc(NodeBC(nodes=n)),
        "from_config": lambda n: GraphApplicator.from_config(GraphBCConfig(node_bcs=[NodeBC(nodes=n)]), 5),
        "set_dirichlet_nodes": lambda n: applicator().set_dirichlet_nodes(n),
        "set_absorbing_nodes": lambda n: applicator().set_absorbing_nodes(n),
        "set_source_nodes": lambda n: applicator().set_source_nodes(n),
        "set_custom_bc": lambda n: applicator().set_custom_bc(n, lambda t, node, field: 0.0),
        "callable nodes": lambda n: applicator().add_node_bc(NodeBC(nodes=lambda num, n=n: n)),
        "apply_direct": lambda n: applicator().apply_direct(np.ones(5), n, 1.0),
        "create_graph_applicator": lambda n: create_graph_applicator(5, boundary_nodes=n),
    }
    for nodes in ([5], [-1], [7, 0]):
        for route, call in routes.items():
            refuses.append((f"{route}{nodes}", lambda c=call, n=nodes: c(n), ValueError, "not nodes of this graph"))
    for nodes in ([2.7], [2.0], ["3"], [True], [np.True_]):
        for route, call in routes.items():
            refuses.append((f"{route}{nodes!r}", lambda c=call, n=nodes: c(n), TypeError, "not node indices"))
    # a bare node, 0 among them, which a truth test took as no nodes (found by A1's fuzz)
    for nodes in (3, 0):
        for route, call in routes.items():
            refuses.append(
                (f"{route}[{nodes}]", lambda c=call, n=nodes: c(n), TypeError, "a list, tuple or range of nodes")
            )
    refuses.append(("NodeBC[nodes=5]", lambda: NodeBC(nodes=5), TypeError, "a list, tuple or range of nodes"))
    refuses.append(("NodeBC[nodes='12']", lambda: NodeBC(nodes="12"), TypeError, "a list, tuple or range of nodes"))
    for factory, call in {
        "create_maze_applicator[exit_nodes=0]": lambda: create_maze_applicator(_maze(), exit_nodes=0),
        "create_maze_applicator[entry_nodes=0]": lambda: create_maze_applicator(_maze(), entry_nodes=0),
    }.items():
        refuses.append((factory, call, TypeError, "a list, tuple or range of nodes"))
    for threshold in ("1", True, float("nan"), float("inf")):
        refuses.append(
            (
                f"from_network_geometry[degree_threshold={threshold!r}]",
                lambda th=threshold: GraphApplicator.from_network_geometry(_maze(), "degree", degree_threshold=th),
                TypeError,
                "degree_threshold",
            )
        )
    refuses.append(
        (
            "from_network_geometry[custom, None]",
            lambda: GraphApplicator.from_network_geometry(_maze(), "custom", boundary_nodes=None),
            TypeError,
            "boundary_nodes is None",
        )
    )
    # a factory's value is read whether or not there are nodes to impose it on
    refuses.append(
        (
            "create_graph_applicator[source_value=a str, no source nodes]",
            lambda: create_graph_applicator(5, boundary_nodes=[1], source_value="2.5"),
            TypeError,
            "source_value is a str",
        )
    )
    refuses.append(
        (
            "from_network_geometry[boundary_value=a str]",
            lambda: GraphApplicator.from_network_geometry(_maze(), "leaf", boundary_value="2.5"),
            TypeError,
            "boundary_value is a str",
        )
    )
    # node 0 as a one-element array, which a truth test dropped, and an entry generator read once
    accepts.append(
        (
            "create_graph_applicator[np.array([0])]",
            lambda: np.array_equal(
                create_graph_applicator(5, boundary_nodes=np.array([0]), boundary_value=2.5).apply(np.zeros(5)),
                [2.5, 0, 0, 0, 0],
            ),
        )
    )
    accepts.append(
        (
            "from_network_geometry[custom, np.array([0])]",
            lambda: np.array_equal(
                GraphApplicator.from_network_geometry(
                    _maze(), "custom", boundary_nodes=np.array([0]), boundary_value=2.5
                ).apply(np.zeros(9)),
                [2.5, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
        )
    )
    refuses.append(
        (
            "create_maze_applicator[entry generator]",
            lambda: create_maze_applicator(_maze(), entry_nodes=(n for n in [4])),
            TypeError,
            "a list, tuple or range of nodes",
        )
    )
    accepts.append(
        (
            "add_node_bc[0, 4]",
            lambda: np.array_equal(
                applicator().add_node_bc(NodeBC(nodes=[0, 4], value=2.5)).apply(np.zeros(5)), [2.5, 0, 0, 0, 2.5]
            ),
        )
    )
    accepts.append(
        (
            "add_node_bc[np.int64 nodes]",
            lambda: np.array_equal(
                applicator().add_node_bc(NodeBC(nodes=np.array([1]), value=2.5)).apply(np.zeros(5)), [0, 2.5, 0, 0, 0]
            ),
        )
    )
    # two conditions at one node: refused when both write one field, taken when their fields differ
    refuses.append(
        (
            "add_node_bc[DIRICHLET, then ABSORBING at node 0]",
            lambda: (
                applicator()
                .add_node_bc(NodeBC(nodes=[0], name="a"))
                .add_node_bc(NodeBC(nodes=[0], bc_type=GraphBCType.ABSORBING, name="b"))
            ),
            ValueError,
            "both write the value field",
        )
    )
    refuses.append(
        (
            "create_maze_applicator[an exit named as an entry]",
            lambda: create_maze_applicator(_maze(), exit_nodes=[4], entry_nodes=[4]),
            ValueError,
            "both write the density field",
        )
    )

    def dirichlet_and_source_at_node_0():
        both = applicator().add_node_bc(NodeBC(nodes=[0], value=2.5, name="d"))
        both.add_node_bc(NodeBC(nodes=[0], bc_type=GraphBCType.SOURCE, value=0.5, name="s"))
        return np.array_equal(both.apply(np.zeros(5)), [2.5, 0, 0, 0, 0]) and np.array_equal(
            both.apply(np.zeros(5), field_type="density"), [0.5, 0, 0, 0, 0]
        )

    def a_dead_end_named_as_an_entry():
        maze = _maze()
        leaf = int(np.where((maze.get_adjacency_matrix() != 0).sum(axis=1) == 1)[0][0])
        density = create_maze_applicator(maze, entry_nodes=[leaf]).apply(np.zeros(9), field_type="density")
        return bool(density[leaf] == 1.0)

    accepts.append(("add_node_bc[DIRICHLET and SOURCE at node 0]", dirichlet_and_source_at_node_0))
    accepts.append(("create_maze_applicator[a dead end named as an entry]", a_dead_end_named_as_an_entry))
    # a continuum default the same check as a segment's type reads (#2292's continuum direction)
    for t in GraphBCType:
        refuses.append(
            (
                f"BoundaryConditions[default_bc=GraphBCType.{t.name}]",
                lambda t=t: BoundaryConditions(dimension=1, segments=[], default_bc=t),
                TypeError,
                "default_bc must be a BCType",
            )
        )
    accepts.append(
        (
            "BoundaryConditions[default_bc=BCType.NO_FLUX]",
            lambda: (
                BoundaryConditions(dimension=1, segments=[], default_bc=BCType.NO_FLUX).default_bc is BCType.NO_FLUX
            ),
        )
    )
    # found by review 2: a node twice with a value each, a set paired with a value list, a 0-d array, a node a lookup
    # reads, and a threshold rounded on its way in
    refuses.append(
        (
            "apply_direct[a node twice, a value each]",
            lambda: applicator().apply_direct(np.zeros(5), [0, 0], [1.0, 2.0]),
            ValueError,
            "more than once",
        )
    )
    refuses.append(
        (
            "apply_direct[a set with a value list]",
            lambda: applicator().apply_direct(np.zeros(5), {3, 1}, [10.0, 20.0]),
            TypeError,
            "is a set",
        )
    )
    accepts.append(
        (
            "apply_direct[a node twice, one value]",
            lambda: np.array_equal(applicator().apply_direct(np.zeros(5), [0, 0], 2.5), [2.5, 0, 0, 0, 0]),
        )
    )
    refuses.append(
        (
            "apply_direct[a 0-d array value]",
            lambda: applicator().apply_direct(np.zeros(5), [1], np.array(2.5)),
            TypeError,
            "values is a 0-D",
        )
    )
    for route, call in routes.items():
        refuses.append(
            (f"{route}[np.array(0)]", lambda c=call: c(np.array(0)), TypeError, "a list, tuple or range of nodes")
        )
    # found by review 3: a value list with entries of different shapes, nodes in an order nobody declared, and a
    # finite threshold beyond float range
    refuses.append(
        (
            "apply_direct[a value list of a number and a column]",
            lambda: applicator().apply_direct(np.zeros((2, 5)), [1, 3], [7.0, np.array([8.0, 9.0])]),
            TypeError,
            "values is a list",
        )
    )
    for label, nodes in (("a generator", lambda: (n for n in [3, 1])), ("a dict's keys", lambda: {3: 0, 1: 0}.keys())):
        refuses.append(
            (
                f"apply_direct[{label} with a value list]",
                lambda nodes=nodes: applicator().apply_direct(np.zeros(5), nodes(), [10.0, 20.0]),
                TypeError,
                "a list, tuple or range of nodes",
            )
        )
    refuses.append(
        (
            "apply_direct[a set with one value]",
            lambda: applicator().apply_direct(np.zeros(5), {3, 1}, 2.5),
            TypeError,
            "a list, tuple or range of nodes",
        )
    )
    for threshold, expected in ((10**400, list(range(9))), (-(10**400), []), (Fraction(10**400, 3), list(range(9)))):
        accepts.append(
            (
                f"from_network_geometry[degree_threshold={'-' if threshold < 0 else ''}10**400-sized]",
                lambda th=threshold, e=expected: (
                    sorted(
                        GraphApplicator.from_network_geometry(
                            _maze(), "degree", degree_threshold=th
                        ).get_boundary_nodes()
                    )
                    == e
                ),
            )
        )
    refuses.append(
        (
            "from_network_geometry[degree_threshold=Decimal]",
            lambda: GraphApplicator.from_network_geometry(_maze(), "degree", degree_threshold=Decimal("2.5")),
            TypeError,
            "degree_threshold",
        )
    )

    def lookup(node):
        return applicator().set_dirichlet_nodes([1, 2], value=7.0, name="d").get_bc_value("d", node, 0.0)

    for node, error, match in ((True, TypeError, "not node indices"), (2.0, TypeError, "not node indices")):
        refuses.append((f"get_bc_value[{node!r}]", lambda n=node: lookup(n), error, match))
    refuses.append(("get_bc_value[99]", lambda: lookup(99), ValueError, "not nodes of this graph"))
    accepts.append(("get_bc_value[1]", lambda: lookup(1) == 7.0))
    accepts.append(
        (
            "from_network_geometry[degree_threshold just below 2]",
            lambda: (
                sorted(
                    GraphApplicator.from_network_geometry(
                        _maze(), "degree", degree_threshold=Fraction(2**60 - 1, 2**59)
                    ).get_boundary_nodes()
                )
                == _maze_leaves()
            ),
        )
    )
    # an empty collection is no condition, as it was
    accepts.append(
        (
            "create_graph_applicator[[]]",
            lambda: create_graph_applicator(5, boundary_nodes=[]).get_bc_by_name("boundary") is None,
        )
    )
    accepts.append(
        (
            "create_maze_applicator[exit_nodes=[]]",
            lambda: create_maze_applicator(_maze(), exit_nodes=[]).get_bc_by_name("exit") is None,
        )
    )
    # the object a graph geometry or a configuration is handed
    for wrong in ([NodeBC(nodes=[0])], NodeBC(nodes=[0]), BoundaryConditions(dimension=1, segments=[]), "dirichlet"):
        name = type(wrong).__name__
        refuses.append((f"geometry[{name}]", lambda w=wrong: _maze(w), TypeError, "GraphBCConfig"))
        refuses.append(
            (
                f"set_boundary_conditions[{name}]",
                lambda w=wrong: _maze().set_boundary_conditions(w),
                TypeError,
                "GraphBCConfig",
            )
        )
        refuses.append(
            (f"from_config[{name}]", lambda w=wrong: GraphApplicator.from_config(w, 5), TypeError, "GraphBCConfig")
        )
    refuses.append(("GraphBCConfig[node_bcs entry]", lambda: GraphBCConfig(node_bcs=["x"]), TypeError, "not a NodeBC"))
    refuses.append(
        (
            "GraphBCConfig[a bare NodeBC]",
            lambda: GraphBCConfig(node_bcs=NodeBC(nodes=[0])),
            TypeError,
            "a list or tuple of NodeBC",
        )
    )
    refuses.append(("add_node_bc[str]", lambda: applicator().add_node_bc("x"), TypeError, "not a NodeBC"))
    config = GraphBCConfig(node_bcs=[NodeBC(nodes=[0])])
    accepts.append(("geometry[GraphBCConfig]", lambda: _maze(config).get_boundary_conditions() is config))
    accepts.append(("geometry[None]", lambda: _maze().get_boundary_conditions() is None))
    # an iterator is kept, not consumed by the check and stored empty; and a config keeps tuples
    refuses.append(
        (
            "GraphBCConfig[generator]",
            lambda: GraphBCConfig(node_bcs=(b for b in [NodeBC(nodes=[0])])),
            TypeError,
            "a list or tuple of NodeBC",
        )
    )
    accepts.append(
        (
            "stored as tuples",
            lambda: (
                isinstance(GraphBCConfig(node_bcs=[NodeBC(nodes=[0, 1])]).node_bcs, tuple)
                and NodeBC(nodes=[0, 1]).nodes == (0, 1)
            ),
        )
    )
    # edge conditions, until an edge operator is ruled (#2493)
    refuses.append(("EdgeBC", lambda: EdgeBC(edges=[(0, 1)]), NotImplementedError, "#2493"))
    refuses.append(
        ("GraphBCConfig[edge_bcs]", lambda: GraphBCConfig(edge_bcs=[object()]), NotImplementedError, "#2493")
    )
    refuses.append(("add_edge_bc", lambda: applicator().add_edge_bc(None), NotImplementedError, "#2493"))
    refuses.append(
        (
            "apply_edge_direct",
            lambda: applicator().apply_edge_direct(np.zeros(3), [(0, 1)], 1.0),
            NotImplementedError,
            "#2493",
        )
    )
    # a condition, once accepted, does not change
    refuses.append(
        ("NodeBC mutated", lambda: setattr(NodeBC(nodes=[0]), "value", 2.5), AttributeError, "cannot assign")
    )
    # a value is one form on every route that takes one, a factory's boundary_value among them; float() would read a
    # string, a bool or a Decimal, and overflows on a number too large for the field
    for label, value in (
        ("a str", "2.5"),
        ("a bool", True),
        ("a Decimal", Decimal("2.5")),
        ("None", None),
        ("a 0-d array", np.array(2.5)),
        ("a list", [2.5]),
        ("10**400", 10**400),
    ):
        refuses.append((f"NodeBC[value={label}]", lambda v=value: NodeBC(nodes=[0], value=v), TypeError, "value is"))
    refuses.append(
        (
            "create_graph_applicator[boundary_value=a str]",
            lambda: create_graph_applicator(5, boundary_nodes=[1], boundary_value="2.5"),
            TypeError,
            "value is a str",
        )
    )
    for label, values, match in (
        ("10**400", 10**400, "values is an int, a real number no float holds"),
        ("a list holding 10**400", [10**400, 1.0], "values is a list"),
    ):
        refuses.append(
            (
                f"apply_direct[values={label}]",
                lambda v=values: applicator().apply_direct(np.zeros(5), [1, 3], v),
                TypeError,
                match,
            )
        )
    # an array of nodes is an integer array; a values array is 1-D and real; a factory takes a callable for its nodes
    for label, nodes in (("a float array", np.array([1.0, 3.0])), ("a bool array", np.array([False, True]))):
        refuses.append(
            (
                f"apply_direct[nodes as {label}]",
                lambda v=nodes: applicator().apply_direct(np.zeros(5), v, 2.5),
                TypeError,
                "a list, tuple or range of nodes",
            )
        )
    for label, values in (("a 2-D array", np.array([[7.0, 8.0]])), ("a bool array", np.array([True, False]))):
        refuses.append(
            (
                f"apply_direct[values as {label}]",
                lambda v=values: applicator().apply_direct(np.zeros((2, 5)), [1, 3], v),
                TypeError,
                "values is a",
            )
        )
    # review 4: a masked array, whose masked entries read as None and were written as NaN; a finite real number whose
    # float() saturates to inf rather than raising; set_custom_bc's rule is its bc_func
    refuses.append(
        (
            "apply_direct[values as a masked array]",
            lambda: applicator().apply_direct(np.ones((2, 5)), [1, 3], np.ma.array([7.0, 8.0], mask=[False, True])),
            TypeError,
            "values is a 1-D float64 MaskedArray",
        )
    )
    refuses.append(
        (
            "apply_direct[nodes as a masked array]",
            lambda: applicator().apply_direct(np.ones(5), np.ma.array([1, 3], mask=[False, True]), 2.5),
            TypeError,
            "boundary_nodes is a 1-D int64 MaskedArray",
        )
    )

    class Saturating:
        """A real number beyond float range whose float() is inf, as sympy's and gmpy2's are."""

        def __float__(self):
            return math.inf

    numbers.Real.register(Saturating)

    class NoFloat:
        """A real number that defines no conversion to a float."""

    numbers.Real.register(NoFloat)
    refuses.append(
        (
            "NodeBC[value=a real number float() cannot convert]",
            lambda: NodeBC(nodes=[0], value=NoFloat()),
            TypeError,
            "a real number no float holds",
        )
    )

    class NaNFloat:
        """A real number whose float() is NaN, which it is not."""

        def __float__(self):
            return math.nan

        def __eq__(self, other):
            return other is self

        __hash__ = object.__hash__

    numbers.Real.register(NaNFloat)
    refuses.append(
        (
            "NodeBC[value=a real number whose float() is NaN]",
            lambda: NodeBC(nodes=[0], value=NaNFloat()),
            TypeError,
            "a real number no float holds",
        )
    )
    # a long double array, refused on every host: where it is wider than a float (x86-64) an entry beyond float range
    # was written as inf (review 5, M1)
    refuses.append(
        (
            "apply_direct[values as a long double array]",
            lambda: applicator().apply_direct(np.zeros(5), [1, 3], np.array([1.0, 2.0], dtype=np.longdouble)),
            TypeError,
            "values is a 1-D longdouble array",
        )
    )

    class Sub(np.ndarray):
        """A numpy array subclass, whose tolist() the readers cannot rely on."""

    refuses.append(
        (
            "apply_direct[nodes as an ndarray subclass]",
            lambda: applicator().apply_direct(np.ones(5), np.array([1, 3]).view(Sub), 2.5),
            TypeError,
            "boundary_nodes is a 1-D int64 Sub",
        )
    )
    refuses.append(
        (
            "NodeBC[value=a real number whose float() saturates]",
            lambda: NodeBC(nodes=[0], value=Saturating()),
            TypeError,
            "a real number no float holds",
        )
    )
    accepts.append(
        (
            "NodeBC[value=inf]",
            lambda: np.array_equal(
                GraphApplicator(5).add_node_bc(NodeBC(nodes=[1], value=math.inf)).apply(np.zeros(5)),
                [0, math.inf, 0, 0, 0],
            ),
        )
    )
    refuses.append(
        (
            "set_custom_bc[bc_func=a str]",
            lambda: applicator().set_custom_bc([0], "x"),
            TypeError,
            "set_custom_bc: bc_func is a str",
        )
    )
    accepts.append(
        (
            "create_graph_applicator[boundary_nodes=a callable]",
            lambda: np.array_equal(
                create_graph_applicator(5, boundary_nodes=lambda n: [n - 1], boundary_value=2.5).apply(np.zeros(5)),
                [0, 0, 0, 0, 2.5],
            ),
        )
    )
    accepts.append(
        (
            "NodeBC[value=Fraction(5, 2)]",
            lambda: np.array_equal(
                GraphApplicator(5).add_node_bc(NodeBC(nodes=[1], value=Fraction(5, 2))).apply(np.zeros(5)),
                [0, 2.5, 0, 0, 0],
            ),
        )
    )
    refuses.append(
        ("GraphBCConfig mutated", lambda: setattr(GraphBCConfig(), "node_bcs", ()), AttributeError, "cannot assign")
    )

    # a lookup that cannot pick one condition, and an input the reader does not read
    def two_named_alike():
        return applicator().set_dirichlet_nodes([0]).set_dirichlet_nodes([4]).get_bc_by_name("dirichlet")

    refuses.append(("get_bc_by_name[ambiguous]", two_named_alike, ValueError, "2 conditions are named"))
    refuses.append(
        (
            "apply_direct[short values]",
            lambda: applicator().apply_direct(np.ones(5), [0, 1], [9.0]),
            ValueError,
            "1 value",
        )
    )
    refuses.append(
        (
            "from_network_geometry[typo]",
            lambda: GraphApplicator.from_network_geometry(_maze(), boundary_values=7.0),
            TypeError,
            "boundary_values",
        )
    )
    accepts.append(
        (
            "from_network_geometry[maze, spatial]",
            lambda: np.array_equal(
                GraphApplicator.from_network_geometry(_maze(), "spatial", boundary_value=3.0).apply(np.zeros(9)),
                _maze_box_field(3.0),
            ),
        )
    )
    accepts.append(
        (
            "create_maze_applicator",
            lambda: sorted(create_maze_applicator(_maze()).get_boundary_nodes()) == _maze_leaves(),
        )
    )
    # the continuum direction: every string a BCType spells is read or refused by name, and so is any other string
    expected = {"neumann": "reflect", "no_flux": "reflect", "robin": "reflect", "reflecting": "reflect"}
    expected |= {"periodic": "periodic", "dirichlet": "clamp"}
    for t in BCType:
        if t is BCType.FLUX or t.name.startswith("EXTRAPOLATION"):
            refuses.append(
                (
                    f"geometric_operation[{t.value}]",
                    lambda t=t: bc_type_to_geometric_operation(t.value),
                    NotImplementedError,
                    "no geometric operation",
                )
            )
        else:
            accepts.append(
                (
                    f"geometric_operation[{t.value}]",
                    lambda t=t: bc_type_to_geometric_operation(t.value) == expected[t.value],
                )
            )
            accepts.append(
                (
                    f"geometric_operation[BCType.{t.name}]",
                    lambda t=t: bc_type_to_geometric_operation(t) == expected[t.value],
                )
            )
    for s in ("source", "custom", "nonsense"):
        refuses.append(
            (
                f"geometric_operation[{s}]",
                lambda s=s: bc_type_to_geometric_operation(s),
                ValueError,
                "is no BC type this reads",
            )
        )
    refuses.append(
        ("geometric_operation[object]", lambda: bc_type_to_geometric_operation(object()), TypeError, "not a BC type")
    )
    # a member is read by its value only as a BCType: the graph alphabet and a foreign enum are refused (review 2, M1)
    for t in GraphBCType:
        refuses.append(
            (
                f"geometric_operation[GraphBCType.{t.name}]",
                lambda t=t: bc_type_to_geometric_operation(t),
                TypeError,
                "must be a BCType",
            )
        )
    refuses.append(
        (
            "geometric_operation[a foreign enum]",
            lambda: bc_type_to_geometric_operation(_ForeignEnum.PERIODIC),
            TypeError,
            "must be a BCType",
        )
    )
    return accepts, refuses


_GATE_ACCEPTS, _GATE_REFUSES = _gate_cases()


@pytest.mark.parametrize(("case", "call"), _GATE_ACCEPTS, ids=[c[0] for c in _GATE_ACCEPTS])
def test_the_gate_takes_each_input_it_can_impose(case, call):
    assert call() is True, f"{case}: accepted, but not as declared"


@pytest.mark.parametrize(("case", "call", "error", "match"), _GATE_REFUSES, ids=[c[0] for c in _GATE_REFUSES])
def test_the_gate_refuses_each_input_it_cannot_impose_by_name(case, call, error, match):
    with pytest.raises(error, match=match):
        call()


def _one_refusal_per_acceptor():
    """Each route that takes a graph BC input, one input outside its form, and the ACCEPTED_FORMS entry it reads."""
    from mfgarchon.geometry.boundary.applicator_graph import (
        GraphBCConfig,
        create_graph_applicator,
        create_maze_applicator,
    )

    def lookup(node):
        return GraphApplicator(5).set_dirichlet_nodes([1], value=7.0, name="d").get_bc_value("d", node, 0.0)

    rows = {
        "NodeBC.nodes": (lambda: NodeBC(nodes={1}), "condition_nodes"),
        "NodeBC.nodes, a callable's return": (
            lambda: GraphApplicator(5).add_node_bc(NodeBC(nodes=lambda n: {1})),
            "nodes",
        ),
        "set_dirichlet_nodes.nodes": (lambda: GraphApplicator(5).set_dirichlet_nodes(iter([1])), "condition_nodes"),
        "set_source_nodes.nodes": (lambda: GraphApplicator(5).set_source_nodes(frozenset({1})), "condition_nodes"),
        "set_absorbing_nodes.nodes": (lambda: GraphApplicator(5).set_absorbing_nodes(None), "condition_nodes"),
        "set_custom_bc.nodes": (
            lambda: GraphApplicator(5).set_custom_bc({1}, lambda t, node, field: 0.0),
            "condition_nodes",
        ),
        "apply_direct.boundary_nodes": (lambda: GraphApplicator(5).apply_direct(np.zeros(5), {1}, 1.0), "nodes"),
        "create_graph_applicator.absorbing_nodes": (
            lambda: create_graph_applicator(5, absorbing_nodes=(n for n in [1])),
            "factory_nodes",
        ),
        "create_maze_applicator.entry_nodes": (
            lambda: create_maze_applicator(_maze(), entry_nodes={0}),
            "factory_nodes",
        ),
        "from_network_geometry.boundary_nodes": (
            lambda: GraphApplicator.from_network_geometry(_maze(), "custom", boundary_nodes={0}),
            "condition_nodes",
        ),
        "a node's entry": (lambda: GraphApplicator(5).set_dirichlet_nodes([1.0]), "node"),
        "a node outside the graph": (lambda: GraphApplicator(5).apply_direct(np.zeros(5), [7], 1.0), "node"),
        "get_bc_value.node": (lambda: lookup(2.0), "node"),
        "NodeBC.value": (lambda: NodeBC(nodes=[0], value="2.5"), "value"),
        "set_dirichlet_nodes.value": (lambda: GraphApplicator(5).set_dirichlet_nodes([0], value=True), "value"),
        "set_custom_bc.bc_func": (lambda: GraphApplicator(5).set_custom_bc([0], "x"), "rule"),
        "create_graph_applicator.boundary_value": (
            lambda: create_graph_applicator(5, boundary_value=Decimal(1)),
            "value",
        ),
        "from_network_geometry.boundary_value": (
            lambda: GraphApplicator.from_network_geometry(_maze(), "leaf", boundary_value=None),
            "value",
        ),
        "apply_direct.values, a count": (lambda: GraphApplicator(5).apply_direct(np.zeros(5), [1, 2], [1.0]), "values"),
        "apply_direct.values, a column": (
            lambda: GraphApplicator(5).apply_direct(np.zeros((2, 5)), [1], [np.array([1.0, 2.0])]),
            "values",
        ),
        "GraphBCConfig.node_bcs": (lambda: GraphBCConfig(node_bcs=iter([])), "conditions"),
        "GraphBCConfig.node_bcs, an entry": (lambda: GraphBCConfig(node_bcs=["x"]), "conditions"),
        "GraphBCConfig.edge_bcs": (lambda: GraphBCConfig(edge_bcs={1}), "conditions"),
        "from_network_geometry.degree_threshold": (
            lambda: GraphApplicator.from_network_geometry(_maze(), "degree", degree_threshold=float("inf")),
            "threshold",
        ),
    }
    return [pytest.param(call, key, id=route) for route, (call, key) in rows.items()]


@pytest.mark.parametrize(("call", "key"), _one_refusal_per_acceptor())
def test_each_route_refuses_in_the_words_of_the_form_it_reads(call, key):
    """Every acceptor reads its input through one table, `ACCEPTED_FORMS`, and refuses in that entry's words: a
    refusal written beside the table would let the forms a route takes drift from the ones it names (#2292)."""
    from mfgarchon.geometry.boundary.applicator_graph import ACCEPTED_FORMS

    with pytest.raises((TypeError, ValueError)) as refused:
        call()
    assert ACCEPTED_FORMS[key] in str(refused.value)
    # the nodes entry is quoted inside the two that also take a callable or None: a route quotes its own, not a wider one
    wider = [words for other, words in ACCEPTED_FORMS.items() if other != key and ACCEPTED_FORMS[key] in words]
    assert not any(words in str(refused.value) for words in wider)


def test_the_maze_applicator_reads_the_mazes_own_exits():
    """`create_maze_applicator` on the geometry its docstring names: it found no node count, and `_node_degrees`
    read only `network_data`, so a maze's exits were never found (#2292). The exits are the leaves of the maze's
    own adjacency, counted here without the applicator."""
    from mfgarchon.geometry.boundary.applicator_graph import create_maze_applicator

    maze = _maze()
    leaves = sorted(int(i) for i in np.where((maze.get_adjacency_matrix() != 0).sum(axis=1) == 1)[0])
    assert leaves, "the fixture maze has no dead end to test against"
    assert sorted(create_maze_applicator(maze).get_bc_by_name("exit").get_nodes(maze.num_spatial_points)) == leaves

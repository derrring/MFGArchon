"""Every (applicator, BCType) cell must either apply the condition or refuse it. Never be silent.

The cell count is **derived from the enums**, not listed. Adding a `BCType` member makes this file
fail until someone decides what each applicator does with it — which is the property a hand-written
list of known cases cannot have, and the reason the discrimination ratchet's 24 mutations name only
2 of `BCType`'s 8 members. #1948

What a declaration asserts: that a branch exists. Whether that branch is *correct* is a separate
axis, per cell, tracked at #1946 — `FDMApplicator` handles both `EXTRAPOLATION_*` in the sense this
file measures while writing unset memory on the uniform path.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
from mfgarchon.geometry.boundary.applicator_fdm import FDMApplicator
from mfgarchon.geometry.boundary.applicator_graph import GraphApplicator, GraphBCType, NodeBC
from mfgarchon.geometry.boundary.applicator_implicit import ImplicitApplicator
from mfgarchon.geometry.boundary.applicator_interpolation import InterpolationApplicator
from mfgarchon.geometry.boundary.applicator_meshfree import MeshfreeApplicator
from mfgarchon.geometry.boundary.applicator_particle import ParticleApplicator
from mfgarchon.geometry.implicit.hypersphere import Hypersphere

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
        lambda t: FDMApplicator(dimension=1).apply(_LINE.copy(), _uniform_bc(t, 1)),
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
#: The one entry here is not a DECLARED cell: `InterpolationApplicator` has
#: `_SUPPORTED_BC_TYPES = None`, so there is no declaration to withdraw and the earlier version of
#: this comment -- which argued from `MeshfreeApplicator` being reachable from
#: `base_solver.apply_bc` -- was defending cells the set does not contain.
_KNOWN_SILENT: set[tuple[str, BCType]] = {
    # `InterpolationApplicator` extrapolates by ORDER (`extrapolation_order` on the constructor), so
    # the two EXTRAPOLATION_* members select nothing: the order is already fixed and the branch has
    # nothing left to do. Silent rather than refusing.
    ("InterpolationApplicator", BCType.EXTRAPOLATION_LINEAR),
    ("InterpolationApplicator", BCType.EXTRAPOLATION_QUADRATIC),
}

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
    "InterpolationApplicator",
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
    rows that nobody has decided about. That automatic growth is the whole mechanism."""
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
    # `InterpolationApplicator` and `GraphApplicator` declare `None` ("not migrated", #1456); and
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


#: `GraphApplicator` gets its OWN product, against its OWN alphabet. #1948 is explicit that this is
#: the encoding it wants: "GraphApplicator is measured against GraphBCType, not BCType -- the family
#: distinction is real and the test must encode it rather than flatten it", and its opening section
#: puts the pairing beyond dispute: "GraphApplicator carrying its own GraphBCType (five members,
#: disjoint from BCType) is likewise correct ... None of that is in question."
#:
#: An earlier version of this file indexed it by `BCType` instead and recorded the eight resulting
#: cells as known gaps. That flattened exactly the distinction the issue asks the test to encode,
#: and it needed nine lines of comment to explain why eight `xfail(strict=True)` rows were not bugs
#: -- which is the tell that the encoding was wrong, not that the applicator was.
#:
#: The cross-alphabet question is real and is not this table's: `BCType` and `GraphBCType` share the
#: `.value` strings {"dirichlet", "neumann"} while being different members, so a `BCType` handed to
#: `NodeBC` matches no branch and returns the field untouched. That belongs with the alphabet work,
#: not here.
_GRAPH_CELLS = [(t,) for t in GraphBCType]

#: Graph cells that come back silent, with the same `xfail(strict=True)` mechanism and the same
#: retirement as `_KNOWN_SILENT`. Measured 2026-09-09: DIRICHLET and ABSORBING apply; these three do
#: not.
#:
#: NEUMANN is the interesting one and the reason this table forbids silence even where it looks
#: harmless. The enum documents it as "Zero flux (no change)", so a no-op may well be the CORRECT
#: answer on a graph -- and that is precisely the case the module docstring names: a caller cannot
#: distinguish it from an unhandled type. Deciding whether it should apply-and-say-so or refuse is
#: #1948 step 2, not a thing to settle by leaving the branch quiet.
_GRAPH_KNOWN_SILENT: set[GraphBCType] = {
    GraphBCType.NEUMANN,
    GraphBCType.SOURCE,
    GraphBCType.CUSTOM,
}


@pytest.mark.parametrize(
    "bc_type",
    [
        pytest.param(
            t,
            marks=(
                [pytest.mark.xfail(strict=True, reason=f"GraphApplicator is silent on {t.name}; #1948 step 2")]
                if t in _GRAPH_KNOWN_SILENT
                else []
            ),
        )
        for (t,) in _GRAPH_CELLS
    ],
    ids=[t.name for (t,) in _GRAPH_CELLS],
)
def test_each_graph_cell_either_applies_or_refuses(bc_type):
    """Same contract as the table above, over the alphabet this applicator actually speaks.

    Derived from `GraphBCType`, so a new member raises the uncovered-cell count on its own -- the
    property the main table has and the reason neither list is hand-maintained.
    """
    field = _NODES.copy()
    applicator = GraphApplicator(num_nodes=_NODES.size).add_node_bc(
        NodeBC(nodes=[0, _NODES.size - 1], bc_type=bc_type, value=2.5, name="s")
    )
    outcome, result = _apply_and_classify(lambda _t: applicator.apply(field.copy()), bc_type)

    if outcome == "refused":
        return

    assert not np.allclose(result, field), (
        f"GraphApplicator returned the field unchanged for GraphBCType.{bc_type.name}. A caller "
        f"cannot tell that from a condition that was applied and happened to change nothing."
    )


def test_a_declaration_that_is_absent_disables_the_gate_rather_than_failing_closed():
    """`_SUPPORTED_BC_TYPES = None` means "not migrated", matching
    `BaseSolver._validate_bc_support`'s convention for un-migrated solvers (#1456).

    Failing closed instead would make every applicator that has not declared refuse everything,
    which is a worse default than the silence it replaces.
    """

    class _Undeclared(FDMApplicator):
        _SUPPORTED_BC_TYPES = None

    assert _Undeclared(dimension=1).supported_bc_types is None
    _Undeclared(dimension=1).apply(_LINE.copy(), _uniform_bc(BCType.PERIODIC, 1))


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

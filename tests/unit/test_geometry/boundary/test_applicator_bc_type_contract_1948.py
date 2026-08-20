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
from mfgarchon.geometry.boundary.applicator_implicit import ImplicitApplicator
from mfgarchon.geometry.boundary.applicator_meshfree import MeshfreeApplicator
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
]

#: Cells that are declared and come back SILENT. Each is a known gap with an issue, marked
#: `strict` so that fixing one turns this table red for "unexpectedly passing" -- the reader is then
#: forced back here to move the cell out of this set rather than leaving a stale exemption.
#: Refusing them instead is not free: `MeshfreeApplicator` is reachable from `base_solver.apply_bc`,
#: so withdrawing the declaration would turn a silent no-op into a raise on a live path. That is a
#: step-2 decision (#1948), not a side effect of writing the table.
_KNOWN_SILENT: set[tuple[str, BCType]] = set()

_CELLS = [(name, cls, call, field, bc_type) for name, cls, call, field in _APPLICATORS for bc_type in BCType]


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
    _CELLS,
    ids=[f"{name}-{t.name}" for name, _, _, _, t in _CELLS],
)
def test_each_cell_either_applies_or_refuses(name, cls, call, field, bc_type):
    """The three outcomes an unhandled type produced across this family were: a silent no-op, a bare
    `pass`, and `else: 0.0` — plus one applicator that raised. Only the raise is right, and this
    asserts it for every cell.

    A silent no-op is indistinguishable at the call site from a correctly-applied condition that
    happened to change nothing, which is why it cannot be allowed even where it is harmless.
    """
    if (name, bc_type) in _KNOWN_SILENT:
        pytest.xfail(f"{name} declares {bc_type.name} and its branch is a no-op; #1948 step 2")

    declared = bc_type in (cls._SUPPORTED_BC_TYPES or frozenset())

    outcome, result = _apply_and_classify(call, bc_type)

    if outcome == "refused":
        assert not declared, f"{name} declares {bc_type.name} yet refused it: {result}"
        return

    assert declared, f"{name} does not declare {bc_type.name} but applied it without refusing"

    baseline = np.asarray(field)
    if result.shape == baseline.shape:
        assert not np.allclose(result, baseline), (
            f"{name} declares {bc_type.name} and returned the field unchanged. A caller cannot tell "
            f"that from a condition that was applied and happened to change nothing; declare it "
            f"unsupported, or make the branch do something."
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

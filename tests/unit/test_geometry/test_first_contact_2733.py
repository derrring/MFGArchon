"""#2733 PR 1: the exact classification of a segment against an implicit domain, as its geometry owner.

`ImplicitDomain.segment_intervals(a, b)` gives the parts of each segment a -> b inside the domain, with the leaf and
face that bound each end: a slab test on a box, a quadratic on a ball, and each CSG node combining its children's
intervals, a complement entering as the closed set phi >= 0. `first_contact(a, b)` reads the first end. Main's crossing
search samples phi at 32 points per leg and bisects (`_leg_crossing`), so a feature thinner than a sample step can be
passed over, wherever no sample lands in it, as the slab case below is at d23a5c87 (#2733), and the face it names
reads one leaf only (#2732's B1). No solver reads the new owner yet; the contact loop is PR 2."""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.implicit import (
    ComplementDomain,
    DifferenceDomain,
    Hyperrectangle,
    Hypersphere,
    ImplicitDomain,
    IntersectionDomain,
    UnionDomain,
)


def _box(*bounds):
    return Hyperrectangle(np.array(bounds, dtype=float))


_SQUARE = _box((0, 1), (0, 1))
_SLAB = _box((0.5, 0.501), (0, 1))
_OUTER, _HOLE = Hypersphere([0.5, 0.5], 0.3), Hypersphere([0.5, 0.5], 0.1)
_WALL_1D, _HOLE_1D = _box((0.2, 0.8)), _box((0.4, 0.6))
_ROOM_1, _ROOM_2 = _box((0, 0.5), (0, 1)), _box((0.3, 1), (0, 1))
_WEST, _EAST = _box((0, 0.5), (0, 1)), _box((0.5, 1), (0, 1))
_ARM_X, _ARM_Y, _QUADRANT = _box((0, 1), (0, 0.5)), _box((0, 0.5), (0, 1)), _box((0.5, 1), (0.5, 1))
_BLOCK = _box((0.4, 0.6), (0.4, 0.6))
_DISC = Hypersphere([0.0, 0.0], 1.0)

# (case, domain, a, b, intervals, contact t, contact point, outward normal, the leaf, its face)
_CASES = [
    (
        "a slab of width 1e-3, against a step of 0.102, which d23a5c87's scan passes over",
        DifferenceDomain(_SQUARE, _SLAB),
        [0.45, 0.3],
        [0.55, 0.32],
        [(0.0, 0.5), (0.51, 1.0)],
        0.5,
        [0.5, 0.31],
        [1.0, 0.0],
        _SLAB,
        0,
    ),
    (
        "B1: across a ring into its hole, the re-entry bounded by the hole",
        DifferenceDomain(_SQUARE, DifferenceDomain(_OUTER, _HOLE)),
        [0.9, 0.5],
        [0.5005, 0.53],
        "ring",
        "ring",
        "ring",
        "ring",
        _OUTER,
        0,
    ),
    (
        "B1 in 1-D: [0, 1] minus ([0.2, 0.8] minus [0.4, 0.6]), 0.1 -> 0.5",
        DifferenceDomain(_box((0, 1)), DifferenceDomain(_WALL_1D, _HOLE_1D)),
        [0.1],
        [0.5],
        [(0.0, 0.25), (0.75, 1.0)],
        0.25,
        [0.2],
        [1.0],
        _WALL_1D,
        0,
    ),
    (
        "B2: two rooms sharing a wall, leaving through the floor",
        UnionDomain([_ROOM_1, _ROOM_2]),
        [0.3, 0.01],
        [0.6, -0.01],
        [(0.0, 0.5)],
        0.5,
        [0.45, 0.0],
        [0.0, -1.0],
        None,
        2,
    ),
    (
        "B2, the rooms in the other order",
        UnionDomain([_ROOM_2, _ROOM_1]),
        [0.3, 0.01],
        [0.6, -0.01],
        [(0.0, 0.5)],
        0.5,
        [0.45, 0.0],
        [0.0, -1.0],
        None,
        2,
    ),
    (
        "two rooms whose walls touch: a step across the wall makes no contact",
        UnionDomain([_WEST, _EAST]),
        [0.3, 0.5],
        [0.7, 0.5],
        [(0.0, 1.0)],
        np.nan,
        [np.nan, np.nan],
        [np.nan, np.nan],
        None,
        -1,
    ),
    (
        "M1: an L as the union of its arms, a step cutting the concave corner",
        UnionDomain([_ARM_X, _ARM_Y]),
        [0.4, 0.4],
        [0.6, 0.55],
        [(0.0, 2 / 3)],
        2 / 3,
        [0.6 - 0.2 / 3, 0.5],
        [0.0, 1.0],
        _ARM_X,
        3,
    ),
    (
        "M1: the same L as the square minus a quadrant",
        DifferenceDomain(_SQUARE, _QUADRANT),
        [0.4, 0.4],
        [0.6, 0.55],
        [(0.0, 2 / 3)],
        2 / 3,
        [0.6 - 0.2 / 3, 0.5],
        [0.0, 1.0],
        _QUADRANT,
        2,
    ),
    (
        "a box corner, left through either face that meets there",
        _SQUARE,
        [0.9, 0.9],
        [1.1, 1.1],
        [(0.0, 0.5)],
        0.5,
        [1.0, 1.0],
        "by face",
        _SQUARE,
        frozenset({1, 3}),
    ),
    (
        "the re-trace from that corner after the x reflection leaves through the y face at once",
        _SQUARE,
        [1.0, 1.0],
        [0.9, 1.1],
        [(0.0, 0.0)],
        0.0,
        [1.0, 1.0],
        [0.0, 1.0],
        _SQUARE,
        3,
    ),
    (
        "a start outside the box: the contact is the entry",
        _SQUARE,
        [-1.0, 0.5],
        [0.5, 0.5],
        [(2 / 3, 1.0)],
        2 / 3,
        [0.0, 0.5],
        [-1.0, 0.0],
        _SQUARE,
        0,
    ),
    (
        "a start outside the box whose step ends on its face: the entry at t = 1",
        _SQUARE,
        [-1.0, 0.5],
        [0.0, 0.5],
        [(1.0, 1.0)],
        1.0,
        [0.0, 0.5],
        [-1.0, 0.0],
        _SQUARE,
        0,
    ),
    (
        "the unit disc touched by a tangent at t = 0.5, its discriminant exactly 0: an entry",
        _DISC,
        [-0.75, 1.0],
        [0.75, 1.0],
        [(0.5, 0.5)],
        0.5,
        [0.0, 1.0],
        [0.0, 1.0],
        _DISC,
        0,
    ),
    (
        "the outside of the unit disc along that tangent: no contact",
        ComplementDomain(_DISC),
        [-0.75, 1.0],
        [0.75, 1.0],
        [(0.0, 1.0)],
        np.nan,
        [np.nan, np.nan],
        [np.nan, np.nan],
        None,
        -1,
    ),
    (
        "the outside of a box, touched only at its corner at t = 0.5: no contact",
        ComplementDomain(_box((0.25, 0.75), (0.25, 0.75))),
        [0.5, 1.0],
        [1.0, 0.5],
        [(0.0, 1.0)],
        np.nan,
        [np.nan, np.nan],
        [np.nan, np.nan],
        None,
        -1,
    ),
    (
        "a chord of the unit disc from a point on it, (1, 0) -> (-2, 0): the far root",
        _DISC,
        [1.0, 0.0],
        [-2.0, 0.0],
        [(0.0, 2 / 3)],
        2 / 3,
        [-1.0, 0.0],
        [-1.0, 0.0],
        _DISC,
        0,
    ),
    (
        "a chord of the unit disc from outside",
        _DISC,
        [-2.0, 0.5],
        [2.0, 0.5],
        [((2 - 0.75**0.5) / 4, (2 + 0.75**0.5) / 4)],
        (2 - 0.75**0.5) / 4,
        [-(0.75**0.5), 0.5],
        [-(0.75**0.5), 0.5],
        _DISC,
        0,
    ),
    (
        "the outside of the unit disc, left into the disc",
        ComplementDomain(_DISC),
        [-2.0, 0.0],
        [0.0, 0.0],
        [(0.0, 0.5)],
        0.5,
        [-1.0, 0.0],
        [1.0, 0.0],
        _DISC,
        0,
    ),
    (
        "a step along an obstacle's face stays in the domain, as `contains` says there",
        DifferenceDomain(_SQUARE, _BLOCK),
        [0.1, 0.4],
        [0.9, 0.4],
        [(0.0, 1.0)],
        np.nan,
        [np.nan, np.nan],
        [np.nan, np.nan],
        None,
        -1,
    ),
]


def _ring_closed_form():
    """B1's intervals and contact: the path a + t (b - a) runs toward the centre and ends in the hole, so it leaves
    the domain at the outer circle's smaller root, radius 0.3, and enters it again at the hole circle's smaller root,
    radius 0.1; the domain's outward normal on the outer circle points into the ring, toward the centre."""
    a, b, c = np.array([0.9, 0.5]), np.array([0.5005, 0.53]), np.array([0.5, 0.5])
    d, f = b - a, a - c
    roots = [np.roots([d @ d, 2 * f @ d, f @ f - r**2]) for r in (0.3, 0.1)]
    t_out, t_back = float(np.min(roots[0])), float(np.min(roots[1]))
    point = a + t_out * d
    return [(0.0, t_out), (t_back, 1.0)], t_out, point, -(point - c) / np.linalg.norm(point - c)


@pytest.mark.parametrize(
    ("case", "domain", "a", "b", "intervals", "t", "point", "normal", "leaf", "face"),
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_the_closed_forms(case, domain, a, b, intervals, t, point, normal, leaf, face):
    """Each case against its closed form: the intervals, and the first contact's parameter, point, outward normal, leaf
    and face. Where two leaves' ends coincide, as both rooms' floors do in B2, the leaf is not asserted: the end
    carries one of them, and the contact's geometry is the same in either order. At the box corner either face is
    the closed form, and the normal is that face's; which one is taken is the box's tie rule, not geometry."""
    if intervals == "ring":
        intervals, t, point, normal = _ring_closed_form()
    if isinstance(normal, str):
        taken = int(domain.first_contact(np.array(a, dtype=float), np.array(b, dtype=float)).leaf_face[0])
        normal = np.eye(len(a))[taken // 2] * (1.0 if taken % 2 else -1.0)
    found = domain.segment_intervals(np.array(a, dtype=float), np.array(b, dtype=float))
    contact = domain.first_contact(np.array(a, dtype=float), np.array(b, dtype=float))
    real = ~np.isnan(found.t[0, :, 0])
    np.testing.assert_allclose(found.t[0][real], np.array(intervals), rtol=0, atol=1e-12)
    # An end that is the segment's own end, strictly inside the domain, is bounded by no surface.
    if domain.signed_distance(np.array(a, dtype=float)) < 0:
        assert found.leaf[0, 0, 0] == -1
    if domain.signed_distance(np.array(b, dtype=float)) < 0:
        assert found.leaf[0][real][-1, 1] == -1
    np.testing.assert_allclose(contact.t, [t], rtol=0, atol=1e-12)
    np.testing.assert_allclose(contact.point, [point], rtol=0, atol=1e-12)
    np.testing.assert_allclose(contact.normal, [normal], rtol=0, atol=1e-12)
    assert contact.leaf_face[0] in face if isinstance(face, frozenset) else contact.leaf_face[0] == face
    if leaf is not None:
        assert contact.leaves[contact.leaf[0]] is leaf
    assert contact.inside_at_start[0] == bool(domain.contains(np.array(a, dtype=float)))


def _segments(rng, n, dim, low, high, *, on=None):
    a = rng.uniform(low, high, (n, dim))
    if on is not None:
        a[: n // 2] = on(a[: n // 2])
    return a, a + rng.normal(0.0, 0.3, (n, dim))


def _snap_to_box_faces(points):
    """Every other start put exactly on a face of [0, 1]^d by its first coordinate, and every fourth also by its last:
    a corner in 2-D, an edge in 3-D."""
    out = points.copy()
    out[::2, 0] = np.round(out[::2, 0])
    out[::4, -1] = np.round(out[::4, -1])
    return out


def _walk(domain, flip=False):
    """The domain's leaves depth first, each with whether it enters complemented: read off the tree here, not from
    the owner's ``_leaves``."""
    if isinstance(domain, ComplementDomain):
        yield from _walk(domain.domain, not flip)
    elif isinstance(domain, (UnionDomain, IntersectionDomain)):
        for child in domain.domains:
            yield from _walk(child, flip)
    else:
        yield domain, flip


def _on_leaf_segments(domain, rng, dim, n):
    """Segments the uniform draw never makes, built on the domain's own box leaves: a start on a leaf's face, one axis
    still, a step along a face, a zero-length step, and a start on a leaf's corner."""
    boxes = [leaf for leaf, _ in _walk(domain) if isinstance(leaf, Hyperrectangle)]
    a, d = rng.uniform(0.0, 1.0, (n, dim)), rng.normal(0.0, 0.3, (n, dim))
    for i in range(n):
        box, axis = boxes[rng.integers(len(boxes))], rng.integers(dim)
        kind = i % 5
        if kind in (0, 2):
            a[i, axis] = box.bounds[axis, rng.integers(2)]
        if kind in (1, 2):
            d[i, axis] = 0.0
        if kind == 3:
            d[i] = 0.0
        if kind == 4:
            a[i] = box.bounds[np.arange(dim), rng.integers(2, size=dim)]
    return a, a + d


_ORACLE_DOMAINS = [
    (
        "1-D, [0, 1] minus ([0.2, 0.8] minus [0.4, 0.6])",
        DifferenceDomain(_box((0, 1)), DifferenceDomain(_WALL_1D, _HOLE_1D)),
        1,
    ),
    (
        "2-D, the square minus a slab, a ring and a disc",
        DifferenceDomain(_SQUARE, UnionDomain([_SLAB, DifferenceDomain(_OUTER, _HOLE), Hypersphere([0.15, 0.8], 0.1)])),
        2,
    ),
    ("2-D, an L of two arms", UnionDomain([_ARM_X, _ARM_Y]), 2),
    (
        "3-D, the cube minus a ball and a bar",
        DifferenceDomain(
            _box((0, 1), (0, 1), (0, 1)),
            UnionDomain([Hypersphere([0.5, 0.5, 0.5], 0.25), _box((0.1, 0.2), (0, 1), (0.7, 0.8))]),
        ),
        3,
    ),
]


@pytest.mark.parametrize(("case", "domain", "dim"), _ORACLE_DOMAINS, ids=[c[0] for c in _ORACLE_DOMAINS])
def test_the_intervals_are_where_contains_holds(case, domain, dim):
    """Dense sampling against the intervals, and every contact against the surface it names.

    - 300 uniform segments, 75 of whose starts are snapped onto a face of [0, 1]^d, 38 of those onto two faces in 2-D
      and 3-D (on the L some of those lie outside it), and 100 built on the domain's own box leaves: starts on a face
      or a corner, a still axis, a step along a face, a zero-length step. At 2001 points along each, a point lies in an
      interval exactly where `contains` holds, away from 1e-9 of an interval end, where the ends' rounding decides;
      whether the start is inside agrees with `contains` with no band (#2732's A2); and the first contact is the first
      interval's end.
    - At each contact, against the tree read here rather than through the owner: the leaf named is that depth-first
      leaf, the contact point lies on the face named, the normal is that face's outward normal, negated for a leaf
      that enters complemented, and it faces along the step for an exit and against it for an entry.
    - Each of the first 40 segments, classified alone, gives the same contact as in the batch."""
    rng = np.random.default_rng(2733)
    a, b = _segments(rng, 300, dim, 0.0, 1.0, on=_snap_to_box_faces)
    a2, b2 = _on_leaf_segments(domain, rng, dim, 100)
    a, b = np.concatenate([a, a2]), np.concatenate([b, b2])
    found = domain.segment_intervals(a, b)
    contact = domain.first_contact(a, b)
    s = np.linspace(0.0, 1.0, 2001)
    for i in range(len(a)):
        lo, hi = found.t[i, :, 0], found.t[i, :, 1]
        real = ~np.isnan(lo)
        held = np.any((s[:, None] >= lo[real]) & (s[:, None] <= hi[real]), axis=1)
        near = np.any(np.abs(s[:, None] - found.t[i][real].ravel()) < 1e-9, axis=1)
        inside = np.asarray(domain.contains(a[i] + s[:, None] * (b[i] - a[i])))
        assert np.array_equal(held[~near], inside[~near]), f"segment {i}: {a[i]} -> {b[i]}"
    assert np.array_equal(contact.inside_at_start, np.asarray(domain.contains(a)))
    first = np.where(found.t[:, 0, 0] == 0.0, found.t[:, 0, 1], found.t[:, 0, 0])
    expected = np.where((found.t[:, 0, 0] == 0.0) & (first == 1.0), np.nan, first)
    np.testing.assert_array_equal(contact.t, expected)

    walk = list(_walk(domain))
    assert [leaf for leaf, _ in walk] == list(contact.leaves)
    for i in np.flatnonzero(~np.isnan(contact.t)):
        leaf, flip = walk[contact.leaf[i]]
        p, face = contact.point[i], int(contact.leaf_face[i])
        if isinstance(leaf, Hyperrectangle):
            axis, side = divmod(face, 2)
            assert abs(p[axis] - leaf.bounds[axis, side]) < 1e-9, f"segment {i}: {p} is off face {face} of {leaf}"
            others = np.arange(dim) != axis
            assert np.all((leaf.bounds[others, 0] - 1e-9 <= p[others]) & (p[others] <= leaf.bounds[others, 1] + 1e-9))
            own = np.eye(dim)[axis] * (1.0 if side else -1.0)
        else:
            assert abs(np.linalg.norm(p - leaf.center) - leaf.radius) < 1e-9, f"segment {i}: {p} is off {leaf}"
            own = (p - leaf.center) / np.linalg.norm(p - leaf.center)
        np.testing.assert_allclose(contact.normal[i], -own if flip else own, rtol=0, atol=1e-9)
        along = float((b[i] - a[i]) @ contact.normal[i])
        assert along >= -1e-12 if contact.inside_at_start[i] else along <= 1e-12, f"segment {i}: d.n = {along}"
    for i in range(40):
        alone = domain.first_contact(a[i], b[i])
        for field in ("t", "leaf", "leaf_face", "inside_at_start", "normal", "point"):
            np.testing.assert_array_equal(getattr(alone, field), getattr(contact, field)[i : i + 1])


@pytest.mark.parametrize(
    "domain",
    [
        _SQUARE,
        _DISC,
        ComplementDomain(_DISC),
        UnionDomain([_ROOM_1, _ROOM_2]),
        IntersectionDomain([_SQUARE, _DISC]),
        DifferenceDomain(_SQUARE, _SLAB),
    ],
    ids=["a box", "a ball", "a complement", "a union", "an intersection", "a difference"],
)
def test_an_empty_batch_is_classified_as_empty(domain):
    """N = 0 segments give empty results on every node, the combining ones included: their sweep reshaped each part's
    ends with an inferred width, which N = 0 leaves undefined. One segment gives one row."""
    none = np.zeros((0, 2))
    found = domain.segment_intervals(none, none)
    contact = domain.first_contact(none, none)
    assert found.t.shape == (0, 1, 2)
    assert contact.t.shape == (0,)
    assert contact.normal.shape == (0, 2)
    one = domain.first_contact(np.array([[0.5, 0.5]]), np.array([[0.5, 2.0]]))
    assert one.t.shape == (1,)
    assert not np.isnan(one.t[0])


def test_a_start_is_admitted_by_the_leaves_own_forms_with_no_band():
    """#2732's A2: whether a start is inside is decided by the intervals at t = 0, which agree with `contains`
    exactly. On points of the unit circle, where |x|^2 - 1 and |x| - 1 disagree in sign about once in 25, and on a box
    at 2e10, where one ulp of a coordinate is 3.8e-6, larger than the band of about 1e-6 that refused both its
    faces."""
    rng = np.random.default_rng(0)
    theta = rng.uniform(0.0, 2.0 * np.pi, 4000)
    on_circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    for domain in (_DISC, ComplementDomain(_DISC)):
        contact = domain.first_contact(on_circle, 0.5 * on_circle)
        assert np.array_equal(contact.inside_at_start, np.asarray(domain.contains(on_circle)))
    far = _box((2e10, 2e10 + 1.0), (0.0, 1.0))
    starts = np.array([[2e10, 0.5], [2e10 + 1.0, 0.5], [np.nextafter(2e10, 0.0), 0.5], [2e10, 1.0]])
    contact = far.first_contact(starts, starts + np.array([[0.5, 0.0], [-0.5, 0.0], [0.5, 0.0], [0.5, 0.5]]))
    assert contact.inside_at_start.tolist() == [True, True, False, True]
    assert np.array_equal(contact.inside_at_start, np.asarray(far.contains(starts)))


class _PhiOnly(ImplicitDomain):
    """A disc given by |x|^2 - r^2, which has the right sign and is not a distance."""

    dimension = 2

    def signed_distance(self, x):
        x = np.asarray(x, dtype=float)
        return np.sum(np.atleast_2d(x) ** 2, axis=1) - 0.25

    def get_bounding_box(self):
        return np.array([[-0.5, 0.5], [-0.5, 0.5]])


_NOT_SEGMENTS = [
    (
        "a phi-only domain",
        _PhiOnly(),
        [[0.1, 0.1]],
        [[0.9, 0.2]],
        NotImplementedError,
        r"_PhiOnly has no exact segment classification.*not sphere-traced.*#2733 D1",
    ),
    (
        "a CSG domain over one",
        DifferenceDomain(_box((-1, 1), (-1, 1)), _PhiOnly()),
        [[0.1, 0.1]],
        [[0.9, 0.2]],
        NotImplementedError,
        r"_PhiOnly has no exact segment classification",
    ),
    (
        "a periodic box",
        Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]]), periodic_dims=(0,)),
        [[0.1, 0.1]],
        [[0.9, 0.2]],
        NotImplementedError,
        r"periodic in axes \(0,\).*#2704",
    ),
    ("a NaN end", _SQUARE, [[0.1, np.nan]], [[0.9, 0.2]], ValueError, r"segment ends must be finite"),
    (
        "ends of two shapes",
        _SQUARE,
        [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
        [[0.9, 0.2]],
        ValueError,
        r"segment ends must both have shape",
    ),
    ("another dimension", _SQUARE, [[0.1, 0.1, 0.1]], [[0.9, 0.2, 0.2]], ValueError, r"with d = 2 for this domain"),
    ("an extent that overflows", _SQUARE, [[-1e308, 0.5]], [[1e308, 0.5]], ValueError, r"extent b - a overflows"),
]


@pytest.mark.parametrize(
    ("row", "domain", "a", "b", "error", "match"), _NOT_SEGMENTS, ids=[r[0] for r in _NOT_SEGMENTS]
)
def test_a_domain_or_segment_it_cannot_classify_is_refused_by_name(row, domain, a, b, error, match):
    """A domain given only by its phi is refused, not sphere-traced: phi is never checked to be a distance, and this
    one is not, so a search that stepped by it could pass over a feature with nothing raised (#2733 D1). A periodic
    box is refused too, since a segment across a periodic axis wraps (#2704). So are segments that are not segments of
    the domain's space: a NaN end, ends of two shapes or of another dimension, and ends whose difference overflows,
    whose direction would be infinite."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    with pytest.raises(error, match=match):
        domain.segment_intervals(a, b)
    with pytest.raises(error, match=match):
        domain.first_contact(a, b)

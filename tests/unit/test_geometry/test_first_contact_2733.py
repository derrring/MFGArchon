"""#2733 PR 1: the exact classification of a segment against an implicit domain, as its geometry owner.

`ImplicitDomain.segment_intervals(a, b)` gives the parts of each segment a -> b inside the domain, with the leaf and
face that bound each end: a slab test on a box, a quadratic on a ball, and each CSG node combining its children's
intervals, a complement entering as the closed set phi >= 0. `first_contact(a, b)` reads the first end. Main's crossing
search samples phi at 32 points per leg and bisects (`_leg_crossing`), so a feature thinner than a sample step is
passed over (#2710), and the face it names reads one leaf only (#2732's B1). No solver reads the new owner yet; the
contact loop is PR 2."""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.implicit import (
    ComplementDomain,
    DifferenceDomain,
    Hyperrectangle,
    Hypersphere,
    ImplicitDomain,
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
        "a slab of width 1e-3, against a step of 0.102 (#2710)",
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
        "a box corner, left through its x face",
        _SQUARE,
        [0.9, 0.9],
        [1.1, 1.1],
        [(0.0, 0.5)],
        0.5,
        [1.0, 1.0],
        [1.0, 0.0],
        _SQUARE,
        1,
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
    carries one of them, and the contact's geometry is the same in either order."""
    if intervals == "ring":
        intervals, t, point, normal = _ring_closed_form()
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
    assert contact.leaf_face[0] == face
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
    """Dense sampling against the intervals: at 2001 points along each of 300 segments, a point lies in an interval
    exactly where `contains` holds, away from 1e-9 of an interval end, where the ends' rounding decides. 75 of the 300
    starts sit exactly on a face of the outer box, 38 of them, in 2-D and 3-D, on two faces at once, and whether a
    start is inside agrees with `contains` there too, with no band (#2732's A2). The first contact is the first
    interval's end."""
    rng = np.random.default_rng(2733)
    a, b = _segments(rng, 300, dim, 0.0, 1.0, on=_snap_to_box_faces)
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


@pytest.mark.parametrize(
    ("domain", "match"),
    [
        (_PhiOnly(), r"_PhiOnly has no exact segment classification.*not sphere-traced.*#2733 D1"),
        (DifferenceDomain(_box((-1, 1), (-1, 1)), _PhiOnly()), r"_PhiOnly has no exact segment classification"),
        (Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]]), periodic_dims=(0,)), r"periodic in axes \(0,\).*#2704"),
    ],
    ids=["a phi-only domain", "a CSG domain over one", "a periodic box"],
)
def test_a_domain_with_no_exact_form_is_refused_by_name(domain, match):
    """A domain given only by its phi is refused, not sphere-traced: phi is never checked to be a distance, and this
    one is not, so a search that stepped by it could pass over a feature with nothing raised (#2733 D1). A periodic
    box is refused too, since a segment across a periodic axis wraps (#2704)."""
    a, b = np.array([[0.1, 0.1]]), np.array([[0.9, 0.2]])
    with pytest.raises(NotImplementedError, match=match):
        domain.segment_intervals(a, b)
    with pytest.raises(NotImplementedError, match=match):
        domain.first_contact(a, b)

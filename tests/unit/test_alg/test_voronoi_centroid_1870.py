"""`CellGeometry.centroid` against the closed form, because nothing else looks at it.

The field is populated for every clipped Voronoi cell and read by nothing in this package -- its
consumer is `mfg-research`, whose own pin sits behind a `skipif` on the field's existence and so
self-skips exactly when it would matter. Measured before this file existed: halving every centroid
(`/ (3.0 * twice_area)` -> `/ (6.0 * twice_area)`) left the full local gate at
`5960 passed ... GATE GREEN`, byte-identical.

The oracle is external in the sense the repository's close-out policy means: the area centroid of a
polygon has a closed form, computed here independently of the implementation rather than captured
from it. So "there is no oracle for this yet" would have been false.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.meshless_galerkin.voronoi_cells import _polygon_centroid

# (name, CCW vertices, centroid by closed form)
_SHAPES = [
    ("unit square", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], (0.5, 0.5)),
    ("right triangle", [(0.0, 0.0), (3.0, 0.0), (0.0, 3.0)], (1.0, 1.0)),
    ("rectangle off the origin", [(2.0, 5.0), (6.0, 5.0), (6.0, 8.0), (2.0, 8.0)], (4.0, 6.5)),
    (
        "regular hexagon centred at (7, -3)",
        [(7.0 + np.cos(k * np.pi / 3), -3.0 + np.sin(k * np.pi / 3)) for k in range(6)],
        (7.0, -3.0),
    ),
    # Non-convex: an L of three unit squares. Composite centroid = sum(A_i c_i) / sum(A_i)
    # = ((0.5+1.5+0.5)/3, (0.5+0.5+1.5)/3) = (5/6, 5/6)... computed below from the pieces.
    (
        "non-convex L",
        [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)],
        ((1.0 * 2.0 * 1.0 + 0.5 * 1.0 * 1.0) / 3.0, (1.0 * 2.0 * 1.0 + 0.5 * 1.0 * 1.0) / 3.0),
    ),
]


@pytest.mark.parametrize(("name", "vertices", "expected"), _SHAPES, ids=[s[0] for s in _SHAPES])
def test_the_centroid_matches_the_closed_form(name, vertices, expected):
    got = _polygon_centroid(np.asarray(vertices, dtype=float))
    assert np.allclose(got, expected, rtol=0, atol=1e-13), f"{name}: {got} != {expected}"


def test_the_centroid_is_translation_equivariant():
    """A centroid must move exactly with its polygon. Catches a scale error the shapes above
    would also catch, and additionally any term that ignores the offset."""
    poly = np.asarray(_SHAPES[4][1], dtype=float)
    shift = np.array([1e3, -7.5])
    assert np.allclose(_polygon_centroid(poly + shift) - _polygon_centroid(poly), shift, rtol=0, atol=1e-9)


def test_the_centroid_lies_inside_the_bounding_box():
    """The 7x7 tiling's cells are known rectangles, so the whole path has a closed form.

    The bounding-box containment below is kept as the cheap catch-all, but it is a one-sided bound.
    The Voronoi cell of a grid node is the rectangle cut by the perpendicular bisectors, clipped to
    the domain -- edges at the midpoints between neighbouring nodes -- so its centroid and area are
    known without evaluating any of this package's code. That is what makes this an oracle rather
    than an agreement check: comparing `cell.centroid` against `_polygon_centroid(cell.polygon)`
    would move both sides together under the halving mutation this file was written for, and pass.

    Measured at HEAD: max centroid deviation 2.8e-15, max area deviation 8.7e-17. atol=1e-12 leaves
    ~360x margin on the centroids. Under the halving mutation the deviation is 0.25.
    """
    from mfgarchon.alg.numerical.meshless_galerkin.voronoi_cells import clipped_voronoi_cells

    xs = np.linspace(0.0, 1.0, 7)
    nodes = np.array([[x, y] for x in xs for y in xs])
    cells = clipped_voronoi_cells(nodes, bounds=[(0.0, 1.0), (0.0, 1.0)])
    assert len(cells) == len(nodes)

    edges = np.concatenate([[0.0], (xs[:-1] + xs[1:]) / 2.0, [1.0]])
    centres = (edges[:-1] + edges[1:]) / 2.0
    widths = edges[1:] - edges[:-1]
    expected_centroids = np.array([[centres[i], centres[j]] for i in range(len(xs)) for j in range(len(xs))])
    expected_areas = np.array([widths[i] * widths[j] for i in range(len(xs)) for j in range(len(xs))])

    np.testing.assert_allclose([c.centroid for c in cells], expected_centroids, rtol=0, atol=1e-12)
    np.testing.assert_allclose([c.area for c in cells], expected_areas, rtol=0, atol=1e-14)

    for cell in cells:
        lo, hi = cell.polygon.min(axis=0), cell.polygon.max(axis=0)
        where = f"centroid {cell.centroid} outside its own polygon's bounding box [{lo}, {hi}]"
        assert np.all(cell.centroid >= lo - 1e-12), where
        assert np.all(cell.centroid <= hi + 1e-12), where


def test_the_area_weighted_centroid_of_a_symmetric_tiling_is_the_domain_centre():
    """One property over all cells at once, independent of any single polygon's closed form."""
    from mfgarchon.alg.numerical.meshless_galerkin.voronoi_cells import clipped_voronoi_cells

    xs = np.linspace(0.0, 1.0, 7)
    nodes = np.array([[x, y] for x in xs for y in xs])
    cells = clipped_voronoi_cells(nodes, bounds=[(0.0, 1.0), (0.0, 1.0)])
    areas = np.array([c.area for c in cells])
    centroids = np.array([c.centroid for c in cells])
    assert np.allclose((areas[:, None] * centroids).sum(axis=0) / areas.sum(), [0.5, 0.5], atol=1e-12)


def test_the_public_path_computes_the_area_centroid_and_not_the_vertex_mean():
    """The closed-form tests call the helper directly, so they cannot see the call site.

    Measured before this test existed: replacing `centroid = _polygon_centroid(poly)` in
    `clipped_voronoi_cells` with `poly.mean(axis=0)` left all nine other assertions green. The two
    that touch the public path could not separate them -- a vertex mean is always inside its own
    bounding box, and on a SYMMETRIC tiling its area-weighted average lands on the domain centre by
    symmetry. So this uses an irregular cloud, where the two differ, and compares against the
    closed form cell by cell.
    """
    from mfgarchon.alg.numerical.meshless_galerkin.voronoi_cells import clipped_voronoi_cells

    rng = np.random.default_rng(20260811)
    nodes = rng.uniform(0.05, 0.95, size=(60, 2))
    cells = clipped_voronoi_cells(nodes, bounds=[(0.0, 1.0), (0.0, 1.0)])

    worst = 0.0
    for cell in cells:
        expected = _polygon_centroid(cell.polygon)
        assert np.allclose(cell.centroid, expected, rtol=0, atol=1e-13)
        worst = max(worst, float(np.abs(expected - cell.polygon.mean(axis=0)).max()))

    assert worst > 1e-3, (
        f"the vertex mean and the area centroid differ by only {worst:.2e} on this cloud, so the "
        "assertions above would pass for either and this test has stopped discriminating"
    )


def test_the_winding_order_does_not_change_the_centroid():
    """A clockwise polygon must give the same centroid as its counter-clockwise reversal.

    Both the numerator and `twice_area` change sign together, so the quotient is winding-invariant.
    Taking `abs` of the denominator alone -- a natural-looking edit -- negates the result for
    clockwise input, and every shape above is counter-clockwise, so nothing here noticed. Measured:
    that mutation left all nine assertions green.
    """
    for _, vertices, expected in _SHAPES:
        ccw = np.asarray(vertices, dtype=float)
        assert np.allclose(_polygon_centroid(ccw[::-1]), expected, rtol=0, atol=1e-13)


def test_a_degenerate_polygon_raises_rather_than_dividing_by_a_vanishing_area():
    """The guard the PR advertises, tested directly because the caller may never reach it.

    The caller refuses at `area <= 1e-14` and this threshold is `abs(2A) <= 2e-14`, the same bound
    once, except that the caller reverses the polygon when the area comes out negative and reversal
    changes the shoelace sum's summation order. Searched for a polygon that passes one and trips the
    other: 320k random polygons over extents 1e-8..1e-6, offsets 0..1e7 and 3..9 vertices, plus 45
    deliberate slivers of length up to 1e4 and thickness down to 1e-17 -- none, and the largest
    reversal asymmetry seen was 4.0e-28 against a 2e-14 threshold. A review reported three such
    polygons out of 5106; that has not been reproduced here and the disagreement is recorded rather
    than resolved.

    Either way this test is what makes the branch live: unreachable, it would otherwise be defensive
    code for an impossible state; reachable, it is a real precondition with no other coverage."""
    collinear = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(ValueError, match="degenerate"):
        _polygon_centroid(collinear)

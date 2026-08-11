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
    """A weaker property than the closed form, and it is here for the shapes the closed form does
    not cover: every clipped Voronoi cell produced by the public path."""
    from mfgarchon.alg.numerical.meshless_galerkin.voronoi_cells import clipped_voronoi_cells

    xs = np.linspace(0.0, 1.0, 7)
    nodes = np.array([[x, y] for x in xs for y in xs])
    cells = clipped_voronoi_cells(nodes, bounds=[(0.0, 1.0), (0.0, 1.0)])
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


def test_a_degenerate_polygon_raises_rather_than_dividing_by_a_vanishing_area():
    """The guard the PR advertises. It is unreachable through `clipped_voronoi_cells`, whose own
    area check (`<= 1e-14`) fires first on the same polygon, so it is tested directly -- otherwise
    it is an untested branch guarding a state its only caller has already refused."""
    collinear = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(ValueError, match="degenerate"):
        _polygon_centroid(collinear)

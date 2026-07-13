"""Sweep-plan S2 / RFC #1574: `project_to_boundary` returns the NEAREST point on the axis-aligned box
boundary, single-sourced across the base `Geometry` default and `TensorProductGrid`.

Two prior divergent, both-wrong impls: the base default snapped one coordinate without clipping the
others (returned points OFF the boundary for exterior queries near a corner -- finding #22); the
TensorProductGrid override snapped by the ORIGINAL point's unbounded-plane distance (non-nearest,
e.g. [1.2, 0.1] -> [1.0, 0.0] instead of [1.0, 0.1] -- finding #17). Both now route through
`nearest_point_on_box_boundary`.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.base import nearest_point_on_box_boundary
from mfgarchon.geometry.boundary import no_flux_bc

_LO = np.array([0.0, 0.0])
_HI = np.array([1.0, 1.0])


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ([1.2, 0.1], [1.0, 0.1]),  # #17 exterior near a face: was [1.0, 0.0] (non-nearest)
        ([1.5, 1.5], [1.0, 1.0]),  # #22 exterior past a corner: was off-boundary
        ([-0.3, 0.4], [0.0, 0.4]),  # exterior other side
        ([0.1, 0.5], [0.0, 0.5]),  # interior: nearest face is x_min
        ([0.5, 0.9], [0.5, 1.0]),  # interior: nearest face is y_max
        ([0.0, 0.5], [0.0, 0.5]),  # already on boundary: unchanged
    ],
)
def test_nearest_point_on_box_boundary(point, expected):
    got = nearest_point_on_box_boundary(np.array(point), _LO, _HI)[0]
    assert np.allclose(got, expected), f"{point} -> {got.tolist()}, expected {expected}"


def test_result_is_always_on_the_boundary():
    """The off-boundary defect (#22): every projected point must have at least one coordinate on a
    face. Discriminating -- the old base impl left exterior-corner points off the boundary."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.5, 1.5, size=(200, 2))  # mix of interior and exterior
    proj = nearest_point_on_box_boundary(pts, _LO, _HI)
    on_face = np.isclose(proj, _LO) | np.isclose(proj, _HI)
    assert np.all(on_face.any(axis=1)), "some projected points are off the boundary"
    # for EXTERIOR points the nearest boundary point is the clip-to-box point (distance = box distance)
    exterior = np.any((pts < _LO) | (pts > _HI), axis=1)
    assert np.allclose(proj[exterior], np.clip(pts, _LO, _HI)[exterior]), "exterior projection not nearest"


def test_tensor_grid_project_routes_through_single_source():
    """TensorProductGrid.project_to_boundary (closest branch) == the shared helper (single-source).
    Discriminating: the old override's original-point-distance loop diverged for exterior points."""
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=no_flux_bc(dimension=2)
    )
    pts = np.array([[1.2, 0.1], [1.5, 1.5], [0.1, 0.5], [0.5, 0.9], [-0.3, 0.4]])
    from_grid = grid.project_to_boundary(pts)
    from_helper = nearest_point_on_box_boundary(pts, _LO, _HI)
    assert np.allclose(from_grid, from_helper)
    # the specific #17 regression value
    assert np.allclose(grid.project_to_boundary(np.array([1.2, 0.1])), [1.0, 0.1])


def test_tensor_grid_named_boundary_still_projects_to_that_face():
    """The boundary_name feature (project to a specific named face) is preserved by the refactor."""
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=no_flux_bc(dimension=2)
    )
    got = grid.project_to_boundary(np.array([0.3, 0.7]), boundary_name="x_max")
    assert np.isclose(got[0], 1.0)  # x snapped to the max face
    assert np.isclose(got[1], 0.7)  # y unchanged

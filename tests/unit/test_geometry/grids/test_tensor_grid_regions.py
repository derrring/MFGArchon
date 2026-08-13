"""
Unit tests for TensorProductGrid region marking functionality.

Tests the SupportsRegionMarking protocol implementation (Issue #590 Phase 1.3).

Created: 2026-01-18
Part of: Geometry & BC Architecture Implementation (Issue #589)
"""

from __future__ import annotations

import pytest

import numpy as np
from numpy.testing import assert_array_equal

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.protocols import SupportsRegionMarking


class TestRegionMarkingProtocolCompliance:
    """Test that TensorProductGrid implements SupportsRegionMarking."""

    def test_implements_protocol(self):
        """Verify TensorProductGrid implements SupportsRegionMarking."""
        grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx=[10, 10], boundary_conditions=no_flux_bc(dimension=2))
        assert isinstance(grid, SupportsRegionMarking)


class TestMarkRegionBasicFunctionality:
    """Test mark_region() method."""

    @pytest.fixture
    def grid_2d(self):
        """Create 2D test grid."""
        return TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx=[10, 10], boundary_conditions=no_flux_bc(dimension=2))

    def test_mark_region_with_predicate(self, grid_2d):
        """Test marking region with predicate function."""
        # Mark left half of domain
        grid_2d.mark_region("left_half", predicate=lambda x: x[:, 0] < 0.5)

        # Verify region exists
        assert "left_half" in grid_2d.get_region_names()

        # Verify mask shape
        mask = grid_2d.get_region_mask("left_half")
        assert mask.shape == (grid_2d.total_points(),)
        assert mask.dtype == bool

        # Verify correctness: should mark ~half the points
        # Grid is 11x11 = 121 points, x < 0.5 marks x-indices [0,1,2,3,4] = 5*11 = 55 points
        assert mask.sum() == 5 * 11

    def test_mark_region_with_direct_mask(self, grid_2d):
        """Test marking region with direct boolean mask."""
        total_pts = grid_2d.total_points()
        custom_mask = np.zeros(total_pts, dtype=bool)
        custom_mask[:50] = True  # Mark first 50 points

        grid_2d.mark_region("custom", mask=custom_mask)

        retrieved_mask = grid_2d.get_region_mask("custom")
        assert_array_equal(retrieved_mask, custom_mask)

    def test_mark_region_with_boundary(self, grid_2d):
        """Test marking region with boundary name."""
        # Mark left boundary (x_min)
        grid_2d.mark_region("left_wall", boundary="x_min")

        mask = grid_2d.get_region_mask("left_wall")

        # Left boundary: x-index = 0, all y-indices
        # Should have 11 points (one for each y value)
        assert mask.sum() == 11

        # Verify points are actually on left boundary
        points = grid_2d.flatten()
        left_boundary_points = points[mask]
        assert np.allclose(left_boundary_points[:, 0], 0.0)

    def test_mark_region_duplicate_name_raises(self, grid_2d):
        """Test that marking duplicate region name raises error."""
        grid_2d.mark_region("test", predicate=lambda x: x[:, 0] < 0.5)

        with pytest.raises(ValueError, match="already exists"):
            grid_2d.mark_region("test", predicate=lambda x: x[:, 1] < 0.5)

    def test_mark_region_no_specification_raises(self, grid_2d):
        """Test that omitting all specification methods raises error."""
        with pytest.raises(ValueError, match="Must specify one of"):
            grid_2d.mark_region("test")

    def test_mark_region_multiple_specifications_raises(self, grid_2d):
        """Test that providing multiple specifications raises error."""
        mask = np.ones(grid_2d.total_points(), dtype=bool)

        with pytest.raises(ValueError, match="Cannot specify multiple"):
            grid_2d.mark_region(
                "test",
                predicate=lambda x: x[:, 0] < 0.5,
                mask=mask,
            )

    def test_mark_region_wrong_mask_shape_raises(self, grid_2d):
        """Test that wrong mask shape raises error."""
        wrong_mask = np.ones(50, dtype=bool)  # Grid has 121 points

        with pytest.raises(ValueError, match="Mask must have shape"):
            grid_2d.mark_region("test", mask=wrong_mask)


class TestBoundaryRegionMarking:
    """Test marking boundary regions."""

    @pytest.fixture
    def grid_2d(self):
        """Create 2D test grid."""
        return TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx=[10, 10], boundary_conditions=no_flux_bc(dimension=2))

    @pytest.mark.parametrize(
        ("boundary_name", "expected_count", "expected_coord_idx", "expected_value"),
        [
            ("x_min", 11, 0, 0.0),  # Left: 11 points, x=0
            ("x_max", 11, 0, 1.0),  # Right: 11 points, x=1
            ("y_min", 11, 1, 0.0),  # Bottom: 11 points, y=0
            ("y_max", 11, 1, 1.0),  # Top: 11 points, y=1
        ],
    )
    def test_standard_boundary_names(
        self,
        grid_2d,
        boundary_name,
        expected_count,
        expected_coord_idx,
        expected_value,
    ):
        """Test all standard boundary names for 2D grid."""
        grid_2d.mark_region("boundary", boundary=boundary_name)

        mask = grid_2d.get_region_mask("boundary")
        assert mask.sum() == expected_count

        # Verify points are on correct boundary
        points = grid_2d.flatten()
        boundary_points = points[mask]
        assert np.allclose(boundary_points[:, expected_coord_idx], expected_value)

    def test_invalid_boundary_name_raises(self, grid_2d):
        """Test that invalid boundary name raises error."""
        with pytest.raises(ValueError, match=r"Invalid boundary name|out of range"):
            grid_2d.mark_region("test", boundary="z_min")  # 2D grid has no z

    def test_3d_boundary_naming(self):
        """Test boundary naming for 3D grid."""
        grid_3d = TensorProductGrid(bounds=[(0, 1)] * 3, Nx=[5, 5, 5], boundary_conditions=no_flux_bc(dimension=3))

        # Should support x, y, z boundaries
        grid_3d.mark_region("left", boundary="x_min")
        grid_3d.mark_region("front", boundary="y_min")
        grid_3d.mark_region("bottom", boundary="z_min")

        # Each boundary should have 6*6 = 36 points
        assert grid_3d.get_region_mask("left").sum() == 6 * 6
        assert grid_3d.get_region_mask("front").sum() == 6 * 6
        assert grid_3d.get_region_mask("bottom").sum() == 6 * 6

    def test_high_dimensional_boundary_naming(self):
        """Test generic boundary naming for high-dimensional grids."""
        grid_4d = TensorProductGrid(bounds=[(0, 1)] * 4, Nx=[3, 3, 3, 3], boundary_conditions=no_flux_bc(dimension=4))

        # Use generic format: dim0_min, dim1_max, etc.
        grid_4d.mark_region("dim0_left", boundary="dim0_min")
        grid_4d.mark_region("dim3_right", boundary="dim3_max")

        # Each boundary should have 4^3 = 64 points
        assert grid_4d.get_region_mask("dim0_left").sum() == 4**3
        assert grid_4d.get_region_mask("dim3_right").sum() == 4**3


class TestRegionRetrieval:
    """Test get_region_mask() and get_region_names()."""

    @pytest.fixture
    def grid_with_regions(self):
        """Create grid with several marked regions."""
        grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx=[20, 20], boundary_conditions=no_flux_bc(dimension=2))

        grid.mark_region("inlet", predicate=lambda x: x[:, 0] < 0.1)
        grid.mark_region("outlet", predicate=lambda x: x[:, 0] > 0.9)
        grid.mark_region("walls", boundary="y_min")

        return grid

    def test_get_region_mask_retrieves_correct_mask(self, grid_with_regions):
        """Test that get_region_mask returns the correct mask."""
        inlet_mask = grid_with_regions.get_region_mask("inlet")

        # Elementwise equality, not containment: `np.all(points[mask][:, 0] < 0.1)` is vacuously
        # true on an all-False mask, so an implementation that marks nothing passes it.  Comparing
        # against the predicate applied to flatten() also pins the ORDER -- a C-vs-Fortran swap in
        # the tensor grid fails this while any count-only check survives.
        points = grid_with_regions.flatten()
        np.testing.assert_array_equal(inlet_mask, points[:, 0] < 0.1)

        # Analytic count guards against both sides being empty together: bounds [0,1] with Nx=20
        # gives 21 points per axis at spacing 0.05, so x < 0.1 selects x in {0.00, 0.05} --
        # 2 columns x 21 rows = 42 of the 441 points.  Measured 42.
        assert inlet_mask.sum() == 42

    def test_get_region_mask_nonexistent_raises(self, grid_with_regions):
        """Test that getting nonexistent region raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            grid_with_regions.get_region_mask("nonexistent")

    def test_get_region_names_returns_all_names(self, grid_with_regions):
        """Test that get_region_names returns all registered regions."""
        names = grid_with_regions.get_region_names()

        assert set(names) == {"inlet", "outlet", "walls"}
        # Order should match registration order
        assert names == ["inlet", "outlet", "walls"]

    def test_get_region_names_empty_initially(self):
        """Test that new grid has no regions."""
        grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx=[10, 10], boundary_conditions=no_flux_bc(dimension=2))
        assert grid.get_region_names() == []


class TestRegionOperations:
    """Test intersect_regions() and union_regions()."""

    @pytest.fixture
    def grid_with_overlapping_regions(self):
        """Create grid with overlapping regions."""
        grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx=[20, 20], boundary_conditions=no_flux_bc(dimension=2))

        # Left half
        grid.mark_region("left", predicate=lambda x: x[:, 0] < 0.5)
        # Bottom half
        grid.mark_region("bottom", predicate=lambda x: x[:, 1] < 0.5)
        # Center square
        grid.mark_region(
            "center",
            predicate=lambda x: np.all((x >= [0.4, 0.4]) & (x <= [0.6, 0.6]), axis=1),
        )

        return grid

    def test_intersect_regions_two_regions(self, grid_with_overlapping_regions):
        """Test intersection of two regions."""
        # Left AND bottom = bottom-left quadrant
        intersection = grid_with_overlapping_regions.intersect_regions("left", "bottom")

        points = grid_with_overlapping_regions.flatten()
        intersect_points = points[intersection]

        # All points should satisfy both conditions
        assert np.all(intersect_points[:, 0] < 0.5)
        assert np.all(intersect_points[:, 1] < 0.5)

        # Should be approximately 1/4 of total points
        # 21x21 grid, bottom-left quadrant: x<0.5 AND y<0.5
        # x-indices [0..10] (11 values), y-indices [0..10] (11 values)
        # Actually: x<0.5 gives 10 points (0.0, 0.05, ..., 0.45), y<0.5 gives 10 points
        # So intersection is 10*10 = 100 points
        expected = 10 * 10  # Both x and y go from 0 to <0.5 in steps of 0.05
        assert intersection.sum() == expected

    def test_intersect_regions_multiple(self, grid_with_overlapping_regions):
        """Test intersection of more than two regions."""
        # Left AND bottom AND center
        intersection = grid_with_overlapping_regions.intersect_regions("left", "bottom", "center")

        # Should be non-empty (center overlaps with left/bottom)
        # Center is [0.4, 0.6] × [0.4, 0.6]
        # Left is x < 0.5, Bottom is y < 0.5
        # Intersection: [0.4, 0.5) × [0.4, 0.5)
        assert intersection.sum() > 0  # Some overlap exists

    def test_union_regions_two_regions(self, grid_with_overlapping_regions):
        """Test union of two regions."""
        # Left OR bottom
        union = grid_with_overlapping_regions.union_regions("left", "bottom")

        # Should include all points where x<0.5 OR y<0.5
        # Grid is 21x21 = 441 points
        # Left half (x<0.5): 10x21 = 210 points
        # Bottom half (y<0.5): 21x10 = 210 points
        # Overlap (bottom-left, x<0.5 AND y<0.5): 10x10 = 100 points
        # Union: 210 + 210 - 100 = 320 points
        assert union.sum() == 320

    def test_union_regions_multiple(self, grid_with_overlapping_regions):
        """Test union of multiple regions."""
        union = grid_with_overlapping_regions.union_regions("left", "bottom", "center")

        # Should include all points in any of the three regions
        # Center is [0.4, 0.6] × [0.4, 0.6], which extends slightly beyond left/bottom
        # Left is x<0.5, Bottom is y<0.5, so center sticks out in both dimensions
        # Union should be larger than just left OR bottom
        left_or_bottom = grid_with_overlapping_regions.union_regions("left", "bottom")
        # Center adds points with x≥0.5 OR y≥0.5 (but still in [0.4, 0.6]²)
        assert union.sum() >= left_or_bottom.sum()

    def test_intersect_empty_raises(self, grid_with_overlapping_regions):
        """Test that intersect with no regions raises error."""
        with pytest.raises(ValueError, match="Must provide at least one"):
            grid_with_overlapping_regions.intersect_regions()

    def test_union_empty_raises(self, grid_with_overlapping_regions):
        """Test that union with no regions raises error."""
        with pytest.raises(ValueError, match="Must provide at least one"):
            grid_with_overlapping_regions.union_regions()

    def test_intersect_nonexistent_region_raises(self, grid_with_overlapping_regions):
        """Test that intersecting nonexistent region raises error."""
        with pytest.raises(KeyError):
            grid_with_overlapping_regions.intersect_regions("left", "nonexistent")

    def test_union_nonexistent_region_raises(self, grid_with_overlapping_regions):
        """Test that union with nonexistent region raises error."""
        with pytest.raises(KeyError):
            grid_with_overlapping_regions.union_regions("bottom", "nonexistent")


class TestRegionMarkingUseCases:
    """Test realistic use cases for region marking."""

    def test_mixed_boundary_conditions_setup(self):
        """Test setting up regions for mixed boundary conditions."""
        # 2D domain with inlet, outlet, and walls
        grid = TensorProductGrid(bounds=[(0, 10), (0, 5)], Nx=[100, 50], boundary_conditions=no_flux_bc(dimension=2))

        # Mark boundary regions
        grid.mark_region("inlet", boundary="x_min")  # Left
        grid.mark_region("outlet", boundary="x_max")  # Right
        grid.mark_region("wall_bottom", boundary="y_min")
        grid.mark_region("wall_top", boundary="y_max")

        # Combine top/bottom into "walls"
        walls_mask = grid.union_regions("wall_bottom", "wall_top")

        # Verify: inlet, outlet, walls should partition the boundary
        inlet = grid.get_region_mask("inlet")
        outlet = grid.get_region_mask("outlet")

        # Should have distinct regions
        assert np.sum(inlet) == 51  # 51 points on left
        assert np.sum(outlet) == 51  # 51 points on right
        assert np.sum(walls_mask) == 2 * 101  # Top + bottom

    def test_obstacle_region_marking(self):
        """Test marking obstacle/safe zone regions."""
        grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx=[50, 50], boundary_conditions=no_flux_bc(dimension=2))

        # Mark circular obstacle at center
        center = np.array([0.5, 0.5])
        radius = 0.2

        def is_in_obstacle(points):
            distances = np.linalg.norm(points - center, axis=1)
            return distances < radius

        grid.mark_region("obstacle", predicate=is_in_obstacle)

        obstacle_mask = grid.get_region_mask("obstacle")

        # Elementwise equality, not containment (an all-False mask satisfies containment
        # vacuously).  This is the file's only NON-SEPARABLE predicate -- every other one is an
        # axis-aligned half-space -- so it is the only place an implementation evaluating the
        # predicate componentwise instead of on whole points can show up, and only once the
        # comparison is elementwise.
        points = grid.flatten()
        np.testing.assert_array_equal(obstacle_mask, np.linalg.norm(points - center, axis=1) < radius)

        # Vacuity guard with an analytic cross-check: 51x51 = 2601 points at spacing 0.02, and a
        # strict inequality on a lattice marks 309 of them -- 11.88% against the exact area
        # fraction pi*0.2^2 = 12.57%, the expected shortfall for a strict test on grid points.
        assert obstacle_mask.sum() == 309

    def test_1d_domain_region_marking(self):
        """Test region marking for 1D problems."""
        grid = TensorProductGrid(bounds=[(0, 1)], Nx=[100], boundary_conditions=no_flux_bc(dimension=1))

        # Mark regions by intervals
        grid.mark_region("left_third", predicate=lambda x: x[:, 0] < 1.0 / 3.0)
        grid.mark_region("middle_third", predicate=lambda x: (x[:, 0] >= 1.0 / 3.0) & (x[:, 0] < 2.0 / 3.0))
        grid.mark_region("right_third", predicate=lambda x: x[:, 0] >= 2.0 / 3.0)

        # Verify: three regions partition the domain
        left = grid.get_region_mask("left_third")
        middle = grid.get_region_mask("middle_third")
        right = grid.get_region_mask("right_third")

        # Should be disjoint
        assert np.sum(left & middle) == 0
        assert np.sum(middle & right) == 0
        assert np.sum(left & right) == 0

        # Should cover entire domain (101 points total)
        assert left.sum() + middle.sum() + right.sum() == 101


if __name__ == "__main__":
    """Run smoke tests."""
    import sys

    print("Running region marking tests...")
    pytest.main([__file__, "-v", "--tb=short", *sys.argv[1:]])

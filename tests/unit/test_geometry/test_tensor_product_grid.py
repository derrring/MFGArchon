"""
Unit tests for TensorProductGrid class.

Tests comprehensive functionality of tensor product grids including:
- Grid initialization and validation
- Coordinate generation (uniform and custom)
- Meshgrid creation and flattening
- Index conversion (multi <-> flat)
- Grid refinement and coarsening
- Volume element computation
- Spacing queries
"""

import pytest

import numpy as np

from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def bc_1d():
    """Default BC for 1D tests (Issue #674: explicit BC required)."""
    return no_flux_bc(dimension=1)


@pytest.fixture
def bc_2d():
    """Default BC for 2D tests (Issue #674: explicit BC required)."""
    return no_flux_bc(dimension=2)


@pytest.fixture
def bc_3d():
    """Default BC for 3D tests (Issue #674: explicit BC required)."""
    return no_flux_bc(dimension=3)


# ============================================================================
# Test Initialization and Validation
# ============================================================================


class TestInitialization:
    """Test grid initialization and parameter validation."""

    def test_1d_uniform_grid(self, bc_1d) -> None:
        """Test 1D uniform grid initialization."""
        grid = TensorProductGrid(bounds=[(0.0, 10.0)], Nx_points=[101], boundary_conditions=bc_1d)

        assert grid.dimension == 1
        assert grid.bounds == [(0.0, 10.0)]
        assert grid.Nx_points == [101]
        assert grid.is_uniform is True
        assert len(grid.coordinates) == 1
        assert len(grid.coordinates[0]) == 101

    def test_2d_uniform_grid(self, bc_2d) -> None:
        """Test 2D uniform grid initialization."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 2.0)], Nx_points=[11, 21], boundary_conditions=bc_2d)

        assert grid.dimension == 2
        assert grid.bounds == [(0.0, 1.0), (0.0, 2.0)]
        assert grid.Nx_points == [11, 21]
        assert grid.is_uniform is True
        assert len(grid.coordinates) == 2

    def test_3d_uniform_grid(self, bc_3d) -> None:
        """Test 3D uniform grid initialization."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            Nx_points=[11, 11, 11],
            boundary_conditions=bc_3d,
        )

        assert grid.dimension == 3
        assert grid.total_points() == 11 * 11 * 11

    def test_bc_required_raises(self) -> None:
        """Test that missing boundary_conditions raises ValueError (Issue #674)."""
        with pytest.raises(ValueError, match="boundary_conditions must be explicitly specified"):
            TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[10])

    def test_invalid_dimension_raises(self, bc_1d) -> None:
        """Test that invalid/empty bounds raises ValueError."""
        # Empty bounds should raise (Issue #676: dimension inferred from bounds)
        with pytest.raises(ValueError, match="bounds cannot be empty"):
            TensorProductGrid(bounds=[], Nx_points=[], boundary_conditions=bc_1d)

        # Explicit dimension with empty bounds still raises empty bounds error
        with pytest.raises(ValueError, match="bounds cannot be empty"):
            TensorProductGrid(bounds=[], Nx_points=[], dimension=1, boundary_conditions=bc_1d)

    def test_mismatched_bounds_length_raises(self, bc_1d) -> None:
        """Test that explicit dimension mismatching bounds raises ValueError."""
        # Issue #676: dimension inferred from bounds, explicit dimension must match
        with pytest.raises(ValueError, match="dimension=2 doesn't match len\\(bounds\\)=1"):
            TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[10, 10], dimension=2, boundary_conditions=bc_1d)

    def test_mismatched_num_points_length_raises(self, bc_2d) -> None:
        """Test that mismatched Nx_points length raises ValueError."""
        # Issue #676: dimension inferred from len(bounds)=2, Nx_points must match
        with pytest.raises(ValueError, match="Nx/Nx_points must have length 2"):
            TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10], boundary_conditions=bc_2d)

    def test_custom_coordinates_grid(self, bc_2d) -> None:
        """Test grid with custom (non-uniform) coordinates."""
        custom_x = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
        custom_y = np.array([0.0, 0.5, 1.0])

        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[5, 3],
            spacing_type="custom",
            custom_coordinates=[custom_x, custom_y],
            boundary_conditions=bc_2d,
        )

        assert grid.dimension == 2
        assert grid.is_uniform is False
        assert len(grid.coordinates[0]) == 5
        assert len(grid.coordinates[1]) == 3
        assert np.allclose(grid.coordinates[0], custom_x)
        assert np.allclose(grid.coordinates[1], custom_y)

    def test_custom_requires_coordinates(self, bc_2d) -> None:
        """Test that custom spacing type requires custom_coordinates."""
        with pytest.raises(ValueError, match="custom_coordinates required"):
            TensorProductGrid(
                bounds=[(0.0, 1.0), (0.0, 1.0)],
                Nx_points=[10, 10],
                spacing_type="custom",
                boundary_conditions=bc_2d,
            )

    def test_unknown_spacing_type_raises(self, bc_2d) -> None:
        """Test that unknown spacing type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown spacing_type"):
            TensorProductGrid(
                bounds=[(0.0, 1.0), (0.0, 1.0)],
                Nx_points=[10, 10],
                spacing_type="invalid",
                boundary_conditions=bc_2d,
            )


# ============================================================================
# Test Grid Properties and Methods
# ============================================================================


class TestGridProperties:
    """Test grid properties and basic methods."""

    def test_total_points_1d(self, bc_1d) -> None:
        """Test total_points for 1D grid."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[101], boundary_conditions=bc_1d)

        assert grid.total_points() == 101

    def test_total_points_2d(self, bc_2d) -> None:
        """Test total_points for 2D grid."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 21], boundary_conditions=bc_2d)

        assert grid.total_points() == 11 * 21

    def test_total_points_3d(self, bc_3d) -> None:
        """Test total_points for 3D grid."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            Nx_points=[10, 10, 10],
            boundary_conditions=bc_3d,
        )

        assert grid.total_points() == 10 * 10 * 10

    def test_uniform_spacing_1d(self, bc_1d) -> None:
        """Test uniform spacing computation in 1D."""
        grid = TensorProductGrid(bounds=[(0.0, 10.0)], Nx_points=[11], boundary_conditions=bc_1d)

        assert grid.is_uniform is True
        assert grid.spacing[0] == pytest.approx(1.0)

    def test_uniform_spacing_2d(self, bc_2d) -> None:
        """Test uniform spacing computation in 2D."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 2.0)], Nx_points=[11, 21], boundary_conditions=bc_2d)

        assert grid.spacing[0] == pytest.approx(0.1)
        assert grid.spacing[1] == pytest.approx(0.1)

    def test_repr_string(self, bc_2d) -> None:
        """Test string representation."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 10], boundary_conditions=bc_2d)

        repr_str = repr(grid)
        assert "TensorProductGrid" in repr_str
        assert "dimension=2" in repr_str
        assert "total_points=100" in repr_str


# ============================================================================
# Test Meshgrid and Flattening
# ============================================================================


class TestMeshgridAndFlatten:
    """Test meshgrid creation and grid point flattening."""

    def test_meshgrid_1d(self, bc_1d) -> None:
        """Test meshgrid for 1D grid."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=bc_1d)

        (X,) = grid.meshgrid()

        assert X.shape == (11,)
        assert X[0] == pytest.approx(0.0)
        assert X[-1] == pytest.approx(1.0)

    def test_meshgrid_2d_ij_indexing(self, bc_2d) -> None:
        """Test meshgrid for 2D grid with 'ij' indexing."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 2.0)], Nx_points=[3, 4], boundary_conditions=bc_2d)

        X, Y = grid.meshgrid(indexing="ij")

        assert X.shape == (3, 4)
        assert Y.shape == (3, 4)
        assert X[0, 0] == pytest.approx(0.0)
        assert X[2, 0] == pytest.approx(1.0)
        assert Y[0, 0] == pytest.approx(0.0)
        assert Y[0, 3] == pytest.approx(2.0)

    def test_meshgrid_2d_xy_indexing(self, bc_2d) -> None:
        """Test meshgrid for 2D grid with 'xy' indexing."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 2.0)], Nx_points=[3, 4], boundary_conditions=bc_2d)

        X, Y = grid.meshgrid(indexing="xy")

        assert X.shape == (4, 3)
        assert Y.shape == (4, 3)
        # Both outputs share that shape, so swapping them passes the shape check alone.
        # Under 'xy' the first axis carries y and the second carries x.
        np.testing.assert_allclose(X[0, :], [0.0, 0.5, 1.0], atol=1e-12)
        np.testing.assert_allclose(Y[:, 0], [0.0, 2 / 3, 4 / 3, 2.0], atol=1e-12)

    def test_flatten_1d(self, bc_1d) -> None:
        """Test flattening for 1D grid."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=bc_1d)

        points = grid.flatten()

        assert points.shape == (11, 1)
        assert points[0, 0] == pytest.approx(0.0)
        assert points[-1, 0] == pytest.approx(1.0)

    def test_flatten_2d(self, bc_2d) -> None:
        """Test flattening for 2D grid."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 2.0)], Nx_points=[3, 4], boundary_conditions=bc_2d)

        points = grid.flatten()

        assert points.shape == (12, 2)  # 3 * 4 points
        # Check corners
        assert np.allclose(points[0], [0.0, 0.0])
        assert np.allclose(points[-1], [1.0, 2.0])

    def test_flatten_3d(self, bc_3d) -> None:
        """Test flattening for 3D grid."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            Nx_points=[2, 2, 2],
            boundary_conditions=bc_3d,
        )

        points = grid.flatten()

        assert points.shape == (8, 3)  # 2 * 2 * 2 points
        # An xy/ij mix-up across three axes leaves the shape untouched, so pin the
        # order: 'ij' indexing, C-order ravel, last axis varying fastest.
        expected = np.array([[i, j, k] for i in (0.0, 1.0) for j in (0.0, 1.0) for k in (0.0, 1.0)], dtype=float)
        np.testing.assert_allclose(points, expected, atol=1e-15)


# ============================================================================
# Test Index Conversion
# ============================================================================


class TestIndexConversion:
    """Test conversion between multi-index and flat index."""

    def test_get_index_2d(self, bc_2d) -> None:
        """Test multi-index to flat index conversion in 2D."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 10], boundary_conditions=bc_2d)

        # Test corner points
        assert grid.get_index((0, 0)) == 0
        assert grid.get_index((9, 9)) == 99
        assert grid.get_index((5, 3)) == 5 * 10 + 3

    def test_get_index_3d(self, bc_3d) -> None:
        """Test multi-index to flat index conversion in 3D."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            Nx_points=[10, 10, 10],
            boundary_conditions=bc_3d,
        )

        assert grid.get_index((0, 0, 0)) == 0
        assert grid.get_index((9, 9, 9)) == 999
        assert grid.get_index((5, 3, 7)) == 5 * 100 + 3 * 10 + 7

    def test_get_multi_index_2d(self, bc_2d) -> None:
        """Test flat index to multi-index conversion in 2D."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 10], boundary_conditions=bc_2d)

        assert grid.get_multi_index(0) == (0, 0)
        assert grid.get_multi_index(99) == (9, 9)
        assert grid.get_multi_index(53) == (5, 3)

    def test_get_multi_index_3d(self, bc_3d) -> None:
        """Test flat index to multi-index conversion in 3D."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            Nx_points=[10, 10, 10],
            boundary_conditions=bc_3d,
        )

        assert grid.get_multi_index(0) == (0, 0, 0)
        assert grid.get_multi_index(999) == (9, 9, 9)
        assert grid.get_multi_index(537) == (5, 3, 7)

    def test_index_conversion_roundtrip(self, bc_2d) -> None:
        """Test that index conversions are inverses."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 10], boundary_conditions=bc_2d)

        # Test several points
        for i in range(10):
            for j in range(10):
                multi = (i, j)
                flat = grid.get_index(multi)
                recovered = grid.get_multi_index(flat)
                assert recovered == multi

    def test_get_index_wrong_dimension_raises(self, bc_2d) -> None:
        """Test that wrong multi_index dimension raises ValueError."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 10], boundary_conditions=bc_2d)

        with pytest.raises(ValueError, match="multi_index must have length 2"):
            grid.get_index((0, 0, 0))

    def test_get_multi_index_out_of_range_raises(self, bc_2d) -> None:
        """Test that out-of-range flat index raises ValueError."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 10], boundary_conditions=bc_2d)

        with pytest.raises(ValueError, match="out of range"):
            grid.get_multi_index(100)

        with pytest.raises(ValueError, match="out of range"):
            grid.get_multi_index(-1)


# ============================================================================
# Test Grid Refinement and Coarsening
# ============================================================================


class TestRefinementCoarsening:
    """Test grid refinement and coarsening operations."""

    def test_refine_uniform_factor(self, bc_2d) -> None:
        """Test refining grid with uniform factor."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=bc_2d)

        fine_grid = grid.refine(2)

        assert fine_grid.Nx_points == [21, 21]
        assert fine_grid.bounds == grid.bounds
        assert fine_grid.dimension == grid.dimension

    def test_refine_per_dimension_factors(self, bc_2d) -> None:
        """Test refining grid with per-dimension factors."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=bc_2d)

        fine_grid = grid.refine([2, 3])

        assert fine_grid.Nx_points == [21, 31]

    def test_refine_preserves_bounds(self, bc_2d) -> None:
        """Test that refinement preserves domain bounds."""
        grid = TensorProductGrid(bounds=[(0.0, 10.0), (0.0, 5.0)], Nx_points=[11, 11], boundary_conditions=bc_2d)

        fine_grid = grid.refine(2)

        assert fine_grid.bounds == grid.bounds
        assert fine_grid.coordinates[0][0] == pytest.approx(0.0)
        assert fine_grid.coordinates[0][-1] == pytest.approx(10.0)
        assert fine_grid.coordinates[1][0] == pytest.approx(0.0)
        assert fine_grid.coordinates[1][-1] == pytest.approx(5.0)

    def test_coarsen_uniform_factor(self, bc_2d) -> None:
        """Test coarsening grid with uniform factor."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[21, 21], boundary_conditions=bc_2d)

        coarse_grid = grid.coarsen(2)

        assert coarse_grid.Nx_points == [11, 11]
        assert coarse_grid.bounds == grid.bounds

    def test_coarsen_per_dimension_factors(self, bc_2d) -> None:
        """Test coarsening grid with per-dimension factors."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[21, 31], boundary_conditions=bc_2d)

        coarse_grid = grid.coarsen([2, 3])

        assert coarse_grid.Nx_points == [11, 11]

    def test_refine_coarsen_consistency(self, bc_2d) -> None:
        """Test that refine and coarsen are (approximately) inverse operations."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=bc_2d)

        fine_grid = grid.refine(2)
        recovered = fine_grid.coarsen(2)

        assert recovered.Nx_points == grid.Nx_points


# ============================================================================
# Test Spacing Queries
# ============================================================================


class TestSpacingQueries:
    """Test grid spacing queries."""

    def test_get_spacing_uniform_1d(self, bc_1d) -> None:
        """Test get_spacing for uniform 1D grid."""
        grid = TensorProductGrid(bounds=[(0.0, 10.0)], Nx_points=[11], boundary_conditions=bc_1d)

        spacing = grid.get_spacing(0)

        assert spacing == pytest.approx(1.0)

    def test_get_spacing_uniform_2d(self, bc_2d) -> None:
        """Test get_spacing for uniform 2D grid."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 2.0)], Nx_points=[11, 21], boundary_conditions=bc_2d)

        spacing_x = grid.get_spacing(0)
        spacing_y = grid.get_spacing(1)

        assert spacing_x == pytest.approx(0.1)
        assert spacing_y == pytest.approx(0.1)

    def test_get_spacing_custom(self, bc_1d) -> None:
        """Test get_spacing for non-uniform grid."""
        custom_x = np.array([0.0, 0.1, 0.3, 0.6, 1.0])

        grid = TensorProductGrid(
            bounds=[(0.0, 1.0)],
            Nx_points=[5],
            spacing_type="custom",
            custom_coordinates=[custom_x],
            boundary_conditions=bc_1d,
        )

        spacings = grid.get_spacing(0)

        assert isinstance(spacings, np.ndarray)
        assert len(spacings) == 4  # n-1 spacings for n points
        assert spacings[0] == pytest.approx(0.1)
        assert spacings[1] == pytest.approx(0.2)
        assert spacings[2] == pytest.approx(0.3)
        assert spacings[3] == pytest.approx(0.4)

    def test_get_spacing_invalid_dimension_raises(self, bc_2d) -> None:
        """Test that invalid dimension index raises ValueError."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 10], boundary_conditions=bc_2d)

        with pytest.raises(ValueError, match="dimension_idx 2 >= dimension 2"):
            grid.get_spacing(2)


# ============================================================================
# Test Volume Element Computation
# ============================================================================


class TestVolumeElement:
    """Test volume element computation."""

    def test_volume_element_uniform_1d(self, bc_1d) -> None:
        """Test volume element for uniform 1D grid."""
        grid = TensorProductGrid(bounds=[(0.0, 10.0)], Nx_points=[11], boundary_conditions=bc_1d)

        vol = grid.volume_element()

        assert vol == pytest.approx(1.0)

    def test_volume_element_uniform_2d(self, bc_2d) -> None:
        """Test volume element for uniform 2D grid."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 2.0)], Nx_points=[11, 21], boundary_conditions=bc_2d)

        vol = grid.volume_element()

        assert vol == pytest.approx(0.1 * 0.1)

    def test_volume_element_uniform_3d(self, bc_3d) -> None:
        """Test volume element for uniform 3D grid."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            Nx_points=[11, 11, 11],
            boundary_conditions=bc_3d,
        )

        vol = grid.volume_element()

        assert vol == pytest.approx(0.1 * 0.1 * 0.1)

    def test_volume_element_custom_requires_index(self, bc_1d) -> None:
        """Test that custom grid requires multi_index for volume element."""
        custom_x = np.array([0.0, 0.1, 0.3, 0.6, 1.0])

        grid = TensorProductGrid(
            bounds=[(0.0, 1.0)],
            Nx_points=[5],
            spacing_type="custom",
            custom_coordinates=[custom_x],
            boundary_conditions=bc_1d,
        )

        with pytest.raises(ValueError, match="multi_index required"):
            grid.volume_element()

    def test_volume_element_custom_with_index(self, bc_1d) -> None:
        """Test volume element for custom grid with multi_index."""
        custom_x = np.array([0.0, 0.1, 0.3, 0.6, 1.0])

        grid = TensorProductGrid(
            bounds=[(0.0, 1.0)],
            Nx_points=[5],
            spacing_type="custom",
            custom_coordinates=[custom_x],
            boundary_conditions=bc_1d,
        )

        # Test at first point
        vol_0 = grid.volume_element(multi_index=(0,))
        assert vol_0 == pytest.approx(0.1)

        # Test at middle point
        vol_2 = grid.volume_element(multi_index=(2,))
        assert vol_2 == pytest.approx(0.25)  # avg of 0.2 and 0.3

        # Test at last point
        vol_4 = grid.volume_element(multi_index=(4,))
        assert vol_4 == pytest.approx(0.4)


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_point_grid(self, bc_1d) -> None:
        """Test grid with single point."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[1], boundary_conditions=bc_1d)

        assert grid.total_points() == 1
        assert grid.spacing[0] == 0.0  # Single point has zero spacing

    def test_two_point_grid(self, bc_1d) -> None:
        """Test grid with two points."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[2], boundary_conditions=bc_1d)

        assert grid.total_points() == 2
        assert grid.spacing[0] == pytest.approx(1.0)

    def test_asymmetric_2d_grid(self, bc_2d) -> None:
        """Test 2D grid with different resolutions in each dimension."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 101], boundary_conditions=bc_2d)

        assert grid.total_points() == 11 * 101
        assert grid.spacing[0] == pytest.approx(0.1)
        assert grid.spacing[1] == pytest.approx(0.01)

    def test_negative_bounds(self, bc_2d) -> None:
        """Test grid with negative bounds."""
        grid = TensorProductGrid(bounds=[(-1.0, 1.0), (-2.0, 2.0)], Nx_points=[11, 21], boundary_conditions=bc_2d)

        assert grid.bounds == [(-1.0, 1.0), (-2.0, 2.0)]
        X, Y = grid.meshgrid()
        assert X[0, 0] == pytest.approx(-1.0)
        assert Y[0, 0] == pytest.approx(-2.0)

    def test_large_grid_total_points(self, bc_3d) -> None:
        """Test total_points calculation for large grid."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            Nx_points=[100, 100, 100],
            boundary_conditions=bc_3d,
        )

        assert grid.total_points() == 1000000

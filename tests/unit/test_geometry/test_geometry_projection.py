"""
Unit tests for GeometryProjector (Issue #257).

Tests projection operations between different geometry discretizations,
including particle↔grid and grid↔grid projections.
"""

import pytest

import numpy as np

from mfgarchon.geometry import GeometryProjector, TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


class TestGeometryProjectorBasics:
    """Test basic GeometryProjector initialization and detection."""

    def test_same_geometry_no_projection(self):
        """Test that same geometry returns identity projection."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[11, 11],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        projector = GeometryProjector(hjb_geometry=grid, fp_geometry=grid)

        # Test HJB → FP (should be identity)
        U = np.random.rand(11, 11)
        U_projected = projector.project_hjb_to_fp(U)
        np.testing.assert_array_equal(U, U_projected)

        # Test FP → HJB (should be identity)
        M = np.random.rand(11, 11)
        M_projected = projector.project_fp_to_hjb(M)
        np.testing.assert_array_equal(M, M_projected)

    def test_factory_create(self):
        """Test factory method."""
        grid1 = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[11, 11],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        grid2 = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[21, 21],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        projector = GeometryProjector.create(hjb_geometry=grid1, fp_geometry=grid2)

        assert projector is not None
        assert projector.hjb_geometry is grid1
        assert projector.fp_geometry is grid2


class TestGrid1DProjections:
    """Test projections for 1D grids."""

    def test_grid_to_grid_1d_interpolation(self):
        """Test 1D grid → grid interpolation."""
        # Coarse grid (HJB)
        coarse_grid = TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1)
        )

        # Fine grid (FP)
        fine_grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))

        projector = GeometryProjector(hjb_geometry=coarse_grid, fp_geometry=fine_grid)

        # Create test function on coarse grid: u(x) = x (linear function for exact interpolation)
        # Note: For 1D grids, get_spatial_grid() returns (N, 1), but values should be 1D array
        coarse_points = coarse_grid.get_spatial_grid().ravel()  # Shape (11,)
        U_coarse = coarse_points

        # Project to fine grid
        U_fine = projector.project_hjb_to_fp(U_coarse)

        # Verify interpolation accuracy
        fine_points = fine_grid.get_spatial_grid().ravel()  # Shape (21,)
        U_expected = fine_points

        # Linear interpolation should be exact for linear functions
        np.testing.assert_allclose(U_fine, U_expected, rtol=1e-10, atol=1e-12)

    def test_grid_to_grid_1d_conservation(self):
        """Test that grid→grid projection preserves integral."""
        # Fine grid (FP solver) - 21 points
        fine_grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))

        # Coarse grid (HJB solver) - 11 points
        coarse_grid = TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1)
        )

        # Setup projector: FP (fine) → HJB (coarse)
        projector = GeometryProjector(hjb_geometry=coarse_grid, fp_geometry=fine_grid)

        # Create density on fine grid (FP)
        # Note: For 1D grids, use ravel() to get 1D arrays
        points_fine = fine_grid.get_spatial_grid().ravel()  # Shape (21,)
        M_fine = np.exp(-((points_fine - 0.5) ** 2) / 0.1)  # Gaussian
        M_fine /= np.trapezoid(M_fine, points_fine)  # Normalize

        # Project from FP (fine) to HJB (coarse)
        M_coarse = projector.project_fp_to_hjb(M_fine)

        # Check integral preservation (approximately)
        points_coarse = coarse_grid.get_spatial_grid().ravel()  # Shape (11,)
        integral_fine = np.trapezoid(M_fine, points_fine)
        integral_coarse = np.trapezoid(M_coarse, points_coarse)

        np.testing.assert_allclose(integral_coarse, integral_fine, rtol=0.1)


class TestGrid2DProjections:
    """Test projections for 2D grids."""

    def test_grid_to_grid_2d_smooth_function(self):
        """Test 2D grid → grid interpolation accuracy."""
        grid1 = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[11, 11],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        grid2 = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[21, 21],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        projector = GeometryProjector(hjb_geometry=grid1, fp_geometry=grid2)

        # Create smooth test function: u(x,y) = x^2 + y^2
        points1 = grid1.get_spatial_grid()  # (N, 2)
        U_flat = points1[:, 0] ** 2 + points1[:, 1] ** 2
        U1 = U_flat.reshape(11, 11)

        # Project to grid2
        U2 = projector.project_hjb_to_fp(U1)

        # Verify on grid2
        points2 = grid2.get_spatial_grid()
        U_expected_flat = points2[:, 0] ** 2 + points2[:, 1] ** 2
        U_expected = U_expected_flat.reshape(21, 21)

        # Allow interpolation error
        np.testing.assert_allclose(U2, U_expected, rtol=1e-2, atol=1e-2)


class TestGrid3DProjections:
    """Test projections for 3D grids."""

    def test_grid_to_grid_3d_shape(self):
        """Test 3D grid → grid projection shape."""
        grid1 = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            Nx_points=[4, 4, 4],
            boundary_conditions=no_flux_bc(dimension=3),
        )
        grid2 = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            Nx_points=[7, 7, 7],
            boundary_conditions=no_flux_bc(dimension=3),
        )

        projector = GeometryProjector(hjb_geometry=grid1, fp_geometry=grid2)

        # Create test field
        U1 = np.random.rand(4, 4, 4)

        # Project
        U2 = projector.project_hjb_to_fp(U1)

        # Check shape
        assert U2.shape == (7, 7, 7)


class TestParticleGridProjections:
    """Test particle ↔ grid projections (simulated with different grids for now)."""

    def test_fine_to_coarse_density_projection(self):
        """Test projecting from fine grid to coarse grid (simulates particle → grid)."""
        # Fine grid simulates particle locations
        fine_grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[51, 51],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        # Coarse grid for HJB
        coarse_grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[11, 11],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        projector = GeometryProjector(hjb_geometry=coarse_grid, fp_geometry=fine_grid)

        # Create density on fine grid (simulates particle density)
        points_fine = fine_grid.get_spatial_grid()
        x, y = points_fine[:, 0], points_fine[:, 1]
        M_fine_flat = np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.05)
        M_fine = M_fine_flat.reshape(51, 51)

        # Normalize
        M_fine /= np.sum(M_fine)

        # Project to coarse grid
        M_coarse = projector.project_fp_to_hjb(M_fine)

        # Check shape
        assert M_coarse.shape == (11, 11)

        # Check that density is positive and has reasonable magnitude
        # Note: Simple interpolation doesn't preserve mass - proper restriction operators needed
        assert np.all(M_coarse >= 0.0)  # Non-negative density
        assert np.sum(M_coarse) > 0.0  # Non-zero mass
        assert M_coarse.max() > M_coarse.mean()  # Has a peak

    def test_projection_preserves_peak_location(self):
        """Test that density peak location is preserved during projection."""
        fine_grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[31, 31],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        coarse_grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[11, 11],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        projector = GeometryProjector(hjb_geometry=coarse_grid, fp_geometry=fine_grid)

        # Gaussian centered at (0.7, 0.3)
        center = np.array([0.7, 0.3])
        points_fine = fine_grid.get_spatial_grid()
        dist_sq = np.sum((points_fine - center) ** 2, axis=1)
        M_fine_flat = np.exp(-dist_sq / 0.01)
        M_fine = M_fine_flat.reshape(31, 31)

        # Project to coarse
        M_coarse = projector.project_fp_to_hjb(M_fine)

        # Find peak location in coarse grid
        peak_coarse = np.unravel_index(np.argmax(M_coarse), M_coarse.shape)

        # Get actual coordinates
        points_coarse = coarse_grid.get_spatial_grid().reshape(11, 11, 2)
        coord_coarse = points_coarse[peak_coarse]

        # Peak should be near (0.7, 0.3)
        assert np.linalg.norm(coord_coarse - center) < 0.15  # Within 1.5 grid cells


class TestProjectionMethods:
    """Test different projection methods."""

    def test_auto_detection_grid_to_grid(self):
        """Test auto-detection chooses appropriate method for grid→grid."""
        grid1 = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[6, 6], boundary_conditions=no_flux_bc(dimension=2)
        )
        grid2 = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[11, 11],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        projector = GeometryProjector(hjb_geometry=grid1, fp_geometry=grid2, projection_method="auto")

        # Should auto-detect grid→grid methods
        assert projector.hjb_to_fp_method == "grid_interpolation"
        assert projector.fp_to_hjb_method == "grid_restriction"

    def test_manual_method_override(self):
        """Test manual projection method override."""
        grid1 = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[6, 6], boundary_conditions=no_flux_bc(dimension=2)
        )
        grid2 = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[11, 11],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        # Force interpolation method
        projector = GeometryProjector(hjb_geometry=grid1, fp_geometry=grid2, projection_method="interpolation")

        assert projector.hjb_to_fp_method == "interpolation"
        assert projector.fp_to_hjb_method == "interpolation"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_values(self):
        """Test projection with zero/empty values."""
        grid1 = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[6, 6], boundary_conditions=no_flux_bc(dimension=2)
        )
        grid2 = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[11, 11],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        projector = GeometryProjector(hjb_geometry=grid1, fp_geometry=grid2)

        # Zero field
        U_zero = np.zeros((6, 6))
        U_projected = projector.project_hjb_to_fp(U_zero)

        np.testing.assert_array_equal(U_projected, np.zeros((11, 11)))

    def test_constant_field(self):
        """Test projection of constant field."""
        grid1 = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[6, 6], boundary_conditions=no_flux_bc(dimension=2)
        )
        grid2 = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[11, 11],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        projector = GeometryProjector(hjb_geometry=grid1, fp_geometry=grid2)

        # Constant field
        U_const = np.ones((6, 6)) * 5.0
        U_projected = projector.project_hjb_to_fp(U_const)

        # Should remain constant
        np.testing.assert_allclose(U_projected, 5.0 * np.ones((11, 11)), rtol=1e-10)


class TestProjectionRegistry:
    """Test ProjectionRegistry functionality (Issue #257 Phase 2)."""

    def setup_method(self):
        """Clear registry before each test."""
        from mfgarchon.geometry import ProjectionRegistry

        ProjectionRegistry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        from mfgarchon.geometry import ProjectionRegistry

        ProjectionRegistry.clear()

    def test_registry_registration(self):
        """Test basic registration of custom projector."""
        from mfgarchon.geometry import ProjectionRegistry

        # Register a custom projector
        @ProjectionRegistry.register(TensorProductGrid, TensorProductGrid, "hjb_to_fp")
        def custom_projector(source, target, values, **kwargs):
            return values * 2.0  # Double all values

        # Check it's registered
        registered = ProjectionRegistry.list_registered()
        assert (TensorProductGrid, TensorProductGrid, "hjb_to_fp") in registered

    def test_registry_integration_with_projector(self):
        """Test that GeometryProjector uses registered projectors."""
        from mfgarchon.geometry import GeometryProjector, ProjectionRegistry

        # Register custom projector that doubles values
        @ProjectionRegistry.register(TensorProductGrid, TensorProductGrid, "hjb_to_fp")
        def double_values(source, target, values, **kwargs):
            # Note: Must return proper shape for target geometry
            # In this test, source and target have same resolution
            return values * 2.0

        # Create projector - should auto-detect and use registry
        grid1 = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[6, 6], boundary_conditions=no_flux_bc(dimension=2)
        )
        grid2 = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[6, 6], boundary_conditions=no_flux_bc(dimension=2)
        )
        projector = GeometryProjector(hjb_geometry=grid1, fp_geometry=grid2, projection_method="auto")

        # Check that registry method is selected
        assert projector.hjb_to_fp_method == "registry"

        # Test projection uses custom function
        U = np.ones((6, 6))
        U_projected = projector.project_hjb_to_fp(U)
        np.testing.assert_allclose(U_projected, 2.0 * np.ones((6, 6)))

    def test_registry_fallback_to_builtin(self):
        """Test that projector falls back to built-in methods when no registry match."""
        from mfgarchon.geometry import GeometryProjector

        # Don't register anything - registry is empty

        # Create projector with different resolutions
        grid1 = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[6, 6], boundary_conditions=no_flux_bc(dimension=2)
        )
        grid2 = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[11, 11],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        projector = GeometryProjector(hjb_geometry=grid1, fp_geometry=grid2, projection_method="auto")

        # Should fall back to built-in grid_interpolation
        assert projector.hjb_to_fp_method == "grid_interpolation"
        assert projector.fp_to_hjb_method == "grid_restriction"

    def test_registry_category_match(self):
        """Test registry lookup with category (base class) match."""
        from mfgarchon.geometry import ProjectionRegistry
        from mfgarchon.geometry.base import CartesianGrid

        # Register projector for CartesianGrid base class
        @ProjectionRegistry.register(CartesianGrid, CartesianGrid, "hjb_to_fp")
        def cartesian_projector(source, target, values, **kwargs):
            return values * 3.0

        # Test with TensorProductGrid (which is a CartesianGrid)
        grid1 = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[6, 6], boundary_conditions=no_flux_bc(dimension=2)
        )
        grid2 = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[6, 6], boundary_conditions=no_flux_bc(dimension=2)
        )

        # Should find category match
        func = ProjectionRegistry.get_projector(grid1, grid2, "hjb_to_fp")
        assert func is not None
        assert func is cartesian_projector

    def test_registry_direction_specificity(self):
        """Test that registry distinguishes between hjb_to_fp and fp_to_hjb."""
        from mfgarchon.geometry import ProjectionRegistry

        # Register different functions for each direction
        @ProjectionRegistry.register(TensorProductGrid, TensorProductGrid, "hjb_to_fp")
        def hjb_to_fp_func(source, target, values, **kwargs):
            return values * 2.0

        @ProjectionRegistry.register(TensorProductGrid, TensorProductGrid, "fp_to_hjb")
        def fp_to_hjb_func(source, target, values, **kwargs):
            return values * 3.0

        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[6, 6], boundary_conditions=no_flux_bc(dimension=2)
        )

        # Check correct functions are returned for each direction
        func_hjb = ProjectionRegistry.get_projector(grid, grid, "hjb_to_fp")
        func_fp = ProjectionRegistry.get_projector(grid, grid, "fp_to_hjb")

        assert func_hjb is hjb_to_fp_func
        assert func_fp is fp_to_hjb_func

    def test_registry_clear(self):
        """Test clearing the registry."""
        from mfgarchon.geometry import ProjectionRegistry

        # Register something
        @ProjectionRegistry.register(TensorProductGrid, TensorProductGrid, "hjb_to_fp")
        def temp_func(source, target, values, **kwargs):
            return values

        assert len(ProjectionRegistry.list_registered()) > 0

        # Clear
        ProjectionRegistry.clear()

        # Should be empty
        assert len(ProjectionRegistry.list_registered()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

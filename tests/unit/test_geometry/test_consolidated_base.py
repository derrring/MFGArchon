"""
Tests for unified Geometry ABC and consolidated geometry base classes.

Part of: Issue #245 Phase 2 - Geometry Architecture Consolidation
Updated: Added AdaptiveGeometry protocol tests (Issue #459)
"""

import pytest

import numpy as np

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.base import CartesianGrid, Geometry
from mfgarchon.geometry.boundary import BCType, no_flux_bc
from mfgarchon.geometry.protocol import AdaptiveGeometry, is_adaptive


class TestGeometryProtocolCompliance:
    """Test that geometries satisfy Geometry ABC."""

    def test_tensorproductgrid_is_geometry(self):
        """Verify TensorProductGrid satisfies Geometry ABC."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[10, 10],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        assert isinstance(grid, Geometry)
        assert isinstance(grid, CartesianGrid)


class TestSolverOperations:
    """Test solver operation methods."""

    def test_laplacian_on_quadratic_function_2d(self):
        """
        Test finite difference Laplacian on u(x,y) = x² + y².

        Analytical Laplacian: Δu = 2 + 2 = 4 everywhere
        """
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[21, 21],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        laplacian = grid.get_laplacian_operator()

        # Create test function: u(x,y) = x² + y²
        x = np.linspace(0, 1, 21)
        y = np.linspace(0, 1, 21)
        X, Y = np.meshgrid(x, y, indexing="ij")
        u = X**2 + Y**2

        # Compute full Laplacian field
        lap_field = laplacian(u)

        # Test at several interior points (avoid boundaries)
        test_points = [(5, 5), (10, 10), (15, 15), (8, 12)]
        for idx in test_points:
            lap_value = lap_field[idx]
            # FD Laplacian should be close to 4.0
            assert np.isclose(lap_value, 4.0, rtol=0.05), f"Laplacian at {idx}: {lap_value}, expected 4.0"

    def test_laplacian_on_linear_function_1d(self):
        """
        Test Laplacian on u(x) = 2*x.

        Analytical Laplacian: Δu = 0 everywhere
        """
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))

        laplacian = grid.get_laplacian_operator()

        # Create test function: u(x) = 2*x
        x = np.linspace(0, 1, 21)
        u = 2 * x

        # Compute full Laplacian field
        lap_field = laplacian(u)

        # Test at interior points
        for idx in [(5,), (10,), (15,)]:
            lap_value = lap_field[idx]
            # Laplacian of linear function should be 0
            assert np.isclose(lap_value, 0.0, atol=1e-10), f"Laplacian at {idx}: {lap_value}, expected 0.0"

    def test_gradient_operator_exists(self):
        """Test that gradient operator returns tuple of callable operators."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[10, 10],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        gradient_ops = grid.get_gradient_operator()
        # Returns tuple of PartialDerivOperator (one per dimension)
        assert isinstance(gradient_ops, tuple)
        assert len(gradient_ops) == 2
        assert all(callable(op) for op in gradient_ops)

    def test_gradient_on_linear_function_2d(self):
        """
        Test finite difference gradient on u(x,y) = 2*x + 3*y.

        Analytical gradient: ∇u = [2, 3]
        """
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[21, 21],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        grad_ops = grid.get_gradient_operator()  # Returns tuple of partial derivative operators

        # Create test function: u(x,y) = 2*x + 3*y
        x = np.linspace(0, 1, 21)
        y = np.linspace(0, 1, 21)
        X, Y = np.meshgrid(x, y, indexing="ij")
        u = 2 * X + 3 * Y

        # Compute full gradient fields
        du_dx_field = grad_ops[0](u)
        du_dy_field = grad_ops[1](u)

        # Test at several interior points
        test_points = [(5, 5), (10, 10), (15, 15), (8, 12)]
        for idx in test_points:
            grad = np.array([du_dx_field[idx], du_dy_field[idx]])
            assert grad.shape == (2,)
            # Gradient should be [2, 3]
            assert np.allclose(grad, [2.0, 3.0], rtol=0.05), f"Gradient at {idx}: {grad}, expected [2, 3]"

    def test_gradient_on_quadratic_function_1d(self):
        """
        Test gradient on u(x) = x².

        Analytical gradient: du/dx = 2*x
        """
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))

        grad_ops = grid.get_gradient_operator()  # Returns tuple with single operator in 1D

        # Create test function: u(x) = x²
        x = np.linspace(0, 1, 21)
        u = x**2

        # Compute full gradient field
        du_dx_field = grad_ops[0](u)

        # Test at x=0.5 (index 10)
        grad_value = du_dx_field[10]
        # At x=0.5, du/dx = 2*0.5 = 1.0
        assert np.isclose(grad_value, 1.0, rtol=0.05), f"Gradient at x=0.5: {grad_value}, expected 1.0"

    def test_interpolator_on_linear_function(self):
        """
        Test linear interpolation on u(x,y) = 2*x + 3*y.

        Interpolated values should match analytical values at arbitrary points.
        """
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[11, 11],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        interpolate = grid.get_interpolator()

        # Create test function: u(x,y) = 2*x + 3*y
        x = np.linspace(0, 1, 11)
        y = np.linspace(0, 1, 11)
        X, Y = np.meshgrid(x, y, indexing="ij")
        u = 2 * X + 3 * Y

        # Test interpolation at several arbitrary points
        test_points = [
            (0.5, 0.5, 2 * 0.5 + 3 * 0.5),  # (x, y, expected)
            (0.3, 0.7, 2 * 0.3 + 3 * 0.7),
            (0.25, 0.25, 2 * 0.25 + 3 * 0.25),
        ]

        for x_val, y_val, expected in test_points:
            interp_value = interpolate(u, np.array([x_val, y_val]))
            assert np.isclose(interp_value, expected, rtol=1e-10), (
                f"Interpolation at ({x_val}, {y_val}): {interp_value}, expected {expected}"
            )

    def test_boundary_handler_exists(self):
        """Test that boundary handler is returned."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[10, 10],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        bc_handler = grid.get_boundary_handler()
        # Resolution order is "stored BC first, else build from bc_type" -- and bc_type
        # defaults to "periodic". If that precedence ever inverts, a no-flux grid hands
        # solvers a periodic handler, which `is not None` cannot see.
        assert bc_handler is grid.get_boundary_conditions()
        assert bc_handler.default_bc is BCType.NO_FLUX


class TestCartesianGridUtilities:
    """Test CartesianGrid-specific utility methods."""

    def test_get_grid_spacing(self):
        """Test grid spacing calculation."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 2.0)],
            Nx_points=[11, 21],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        dx = grid.get_grid_spacing()
        assert len(dx) == 2
        # dx1 = (1-0) / (11-1) = 0.1
        assert np.isclose(dx[0], 0.1, rtol=1e-10)
        # dx2 = (2-0) / (21-1) = 0.1
        assert np.isclose(dx[1], 0.1, rtol=1e-10)

    def test_get_grid_shape(self):
        """Test grid shape retrieval."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[10, 20],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        shape = grid.get_grid_shape()
        assert shape == (10, 20)

    def test_get_bounds(self):
        """Test bounding box retrieval."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (-1.0, 1.0), (0.0, 2.0)],
            Nx_points=[10, 10, 10],
            boundary_conditions=no_flux_bc(dimension=3),
        )

        min_coords, max_coords = grid.get_bounds()
        assert np.allclose(min_coords, [0.0, -1.0, 0.0])
        assert np.allclose(max_coords, [1.0, 1.0, 2.0])


class TestDataInterface:
    """Test data interface methods from Geometry ABC."""

    def test_dimension_property(self):
        """Test dimension property."""
        grid_1d = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[10], boundary_conditions=no_flux_bc(dimension=1))
        grid_2d = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[10, 10],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        grid_3d = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            Nx_points=[5, 5, 5],
            boundary_conditions=no_flux_bc(dimension=3),
        )

        assert grid_1d.dimension == 1
        assert grid_2d.dimension == 2
        assert grid_3d.dimension == 3

    def test_num_spatial_points(self):
        """Test total spatial points calculation."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[10, 20],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        assert grid.num_spatial_points == 200

    def test_get_spatial_grid(self):
        """Test spatial grid retrieval."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[5, 5], boundary_conditions=no_flux_bc(dimension=2)
        )

        points = grid.get_spatial_grid()
        assert points.shape == (25, 2)  # 5*5 points, 2D
        # Pin the flattening convention the shape cannot see: 'ij' indexing with a
        # C-order ravel, so the last axis varies fastest. A Fortran-order flip would
        # give points[1] == [0.25, 0.0] with the shape unchanged.
        np.testing.assert_allclose(points[0], [0.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(points[1], [0.0, 0.25], atol=1e-15)
        np.testing.assert_allclose(points[-1], [1.0, 1.0], atol=1e-15)

    def test_get_problem_config(self):
        """Test problem configuration dictionary."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 2.0)],
            Nx_points=[10, 20],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        config = grid.get_problem_config()
        assert config["num_spatial_points"] == 200
        assert config["spatial_shape"] == (10, 20)
        assert config["spatial_bounds"] == ((0.0, 1.0), (0.0, 2.0))
        assert config["spatial_discretization"] == (10, 20)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_laplacian_wrong_field_shape(self):
        """Test that Laplacian raises error for wrong field shape."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[10, 10],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        laplacian = grid.get_laplacian_operator()

        # Wrong shape field
        u_wrong = np.random.rand(5, 5)  # Expected (10, 10)

        with pytest.raises(ValueError):
            laplacian(u_wrong)

    def test_gradient_wrong_field_shape(self):
        """Test that gradient raises error for wrong field shape."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[10, 10],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        grad_ops = grid.get_gradient_operator()

        # Wrong shape field
        u_wrong = np.random.rand(5, 5)  # Expected (10, 10)

        with pytest.raises(ValueError):
            grad_ops[0](u_wrong)

    def test_interpolator_wrong_point_dimension(self):
        """Test that interpolator raises error for wrong point dimension."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[10, 10],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        interpolate = grid.get_interpolator()
        u = np.random.rand(10, 10)

        with pytest.raises(ValueError, match="Point must have length 2"):
            interpolate(u, np.array([0.5]))  # 1D point for 2D grid


class TestAdaptiveGeometryProtocol:
    """Test AdaptiveGeometry protocol for AMR support (Issue #459)."""

    def test_is_adaptive_helper_function(self):
        """Test is_adaptive() helper function."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))

        # is_adaptive should return False for non-adaptive geometries
        assert is_adaptive(grid) is False
        assert is_adaptive("not a geometry") is False
        assert is_adaptive(None) is False

    def test_mock_adaptive_geometry_compliance(self):
        """Test that a mock adaptive geometry satisfies the protocol."""

        class MockAdaptiveGrid:
            """Mock adaptive geometry for protocol testing."""

            def refine(self, criteria):
                return 4  # Refined 4 cells

            def coarsen(self, criteria):
                return 2  # Coarsened 2 cells

            def adapt(self, solution_data):
                return {"refined": 4, "coarsened": 2, "total_cells": 102}

            @property
            def max_refinement_level(self):
                return 3

            @property
            def num_leaf_cells(self):
                return 100

        mock = MockAdaptiveGrid()

        # Mock should satisfy the protocol
        assert isinstance(mock, AdaptiveGeometry)
        assert is_adaptive(mock)

        # Test method calls
        assert mock.refine({}) == 4
        assert mock.coarsen({}) == 2
        result = mock.adapt({"u": None})
        assert result["refined"] == 4
        assert mock.max_refinement_level == 3
        assert mock.num_leaf_cells == 100

    def test_incomplete_adaptive_geometry_rejected(self):
        """Test that incomplete implementations are rejected."""

        class IncompleteAdaptive:
            """Missing required methods."""

            def refine(self, criteria):
                return 0

            # Missing: coarsen, adapt, max_refinement_level, num_leaf_cells

        incomplete = IncompleteAdaptive()

        # Should NOT satisfy the protocol
        assert not isinstance(incomplete, AdaptiveGeometry)
        assert not is_adaptive(incomplete)

    def test_adaptive_geometry_protocol_is_orthogonal(self):
        """Test that AdaptiveGeometry is orthogonal to base Geometry."""

        class AdaptiveCartesianMock(CartesianGrid):
            """Mock that implements both CartesianGrid and AdaptiveGeometry."""

            def __init__(self):
                # Minimal CartesianGrid initialization
                self._dimension = 2
                self._bounds = [(0.0, 1.0), (0.0, 1.0)]
                self._Nx_points = [10, 10]

            @property
            def dimension(self):
                return self._dimension

            @property
            def geometry_type(self):
                from mfgarchon.geometry.protocol import GeometryType

                return GeometryType.CARTESIAN_GRID

            @property
            def num_spatial_points(self):
                return 100

            def get_spatial_grid(self):
                return np.zeros((100, 2))

            def get_bounds(self):
                return (np.array([0.0, 0.0]), np.array([1.0, 1.0]))

            def get_problem_config(self):
                return {}

            def is_on_boundary(self, points, tolerance=1e-10):
                return np.zeros(len(points), dtype=bool)

            def get_boundary_normal(self, points):
                return np.zeros_like(points)

            def project_to_boundary(self, points):
                return points

            def project_to_interior(self, points):
                return points

            def get_boundary_regions(self):
                return {"all": {}}

            def get_grid_spacing(self):
                return (0.1, 0.1)

            def get_grid_shape(self):
                return (10, 10)

            def get_laplacian_operator(self):
                return lambda u, idx: 0.0

            def get_gradient_operator(self):
                return lambda u, idx: np.zeros(2)

            def get_interpolator(self):
                return lambda u, point: 0.0

            def get_boundary_handler(self, bc_type="dirichlet"):
                return None

            # AdaptiveGeometry methods
            def refine(self, criteria):
                return 4

            def coarsen(self, criteria):
                return 0

            def adapt(self, solution_data):
                return {"refined": 4, "coarsened": 0, "total_cells": 104}

            @property
            def max_refinement_level(self):
                return 2

            @property
            def num_leaf_cells(self):
                return 100

        mock = AdaptiveCartesianMock()

        # Should satisfy both protocols
        assert isinstance(mock, Geometry)
        assert isinstance(mock, CartesianGrid)
        assert isinstance(mock, AdaptiveGeometry)
        assert is_adaptive(mock)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

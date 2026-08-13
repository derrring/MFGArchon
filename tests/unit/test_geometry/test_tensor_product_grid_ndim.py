"""
Unit tests for n-dimensional (d>3) TensorProductGrid support.

Tests the dimension-agnostic geometry infrastructure added in #187.
"""

import warnings

import pytest

import numpy as np

from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid


class TestNDimensionalGrids:
    """Test arbitrary-dimensional tensor product grids."""

    def test_4d_grid_basic(self):
        """Test basic 4D grid creation."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            Nx_points=[10, 10, 10, 10],
            boundary_conditions=no_flux_bc(dimension=4),
        )

        assert grid.dimension == 4
        assert len(grid.coordinates) == 4
        assert all(len(coords) == 10 for coords in grid.coordinates)

        # Check total points
        total_points = 10**4
        flat_points = grid.flatten()
        assert flat_points.shape == (total_points, 4)

    def test_5d_grid_small(self):
        """Test 5D grid with small resolution."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0)] * 5,
            Nx_points=[5] * 5,  # 5^5 = 3,125 points
            boundary_conditions=no_flux_bc(dimension=5),
        )

        assert grid.dimension == 5
        total_points = 5**5
        flat_points = grid.flatten()
        assert flat_points.shape == (total_points, 5)

        # Check bounds
        assert np.all(flat_points >= 0.0)
        assert np.all(flat_points <= 1.0)

    def test_6d_grid_sparse(self):
        """Test 6D grid with very sparse resolution."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0)] * 6,
            Nx_points=[3] * 6,  # 3^6 = 729 points
            boundary_conditions=no_flux_bc(dimension=6),
        )

        assert grid.dimension == 6
        total_points = 3**6
        flat_points = grid.flatten()
        assert flat_points.shape == (total_points, 6)

    def test_10d_grid_minimal(self):
        """Test 10D grid with minimal resolution."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0)] * 10,
            Nx_points=[2] * 10,  # 2^10 = 1,024 points
            boundary_conditions=no_flux_bc(dimension=10),
        )

        assert grid.dimension == 10
        flat_points = grid.flatten()
        assert flat_points.shape == (1024, 10)

    def test_performance_warning(self):
        """Test that performance warning is issued for d>3."""
        with pytest.warns(UserWarning, match="O\\(N\\^d\\)"):
            TensorProductGrid(bounds=[(0.0, 1.0)] * 4, Nx_points=[10] * 4, boundary_conditions=no_flux_bc(dimension=4))

    def test_non_uniform_resolution_4d(self):
        """Test 4D grid with different resolution per dimension."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0)] * 4,
            Nx_points=[5, 10, 15, 20],  # Total: 15,000 points
            boundary_conditions=no_flux_bc(dimension=4),
        )

        assert grid.dimension == 4
        assert len(grid.coordinates[0]) == 5
        assert len(grid.coordinates[1]) == 10
        assert len(grid.coordinates[2]) == 15
        assert len(grid.coordinates[3]) == 20

        flat_points = grid.flatten()
        assert flat_points.shape == (5 * 10 * 15 * 20, 4)

    def test_meshgrid_4d(self):
        """Test meshgrid generation for 4D."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0)] * 4, Nx_points=[3, 4, 5, 6], boundary_conditions=no_flux_bc(dimension=4)
        )

        mesh = grid.meshgrid()
        assert len(mesh) == 4
        assert mesh[0].shape == (3, 4, 5, 6)
        assert mesh[1].shape == (3, 4, 5, 6)
        assert mesh[2].shape == (3, 4, 5, 6)
        assert mesh[3].shape == (3, 4, 5, 6)

    def test_bounds_correctness_5d(self):
        """Each of the five disjoint bound intervals is spanned exactly, not merely respected.

        Containment alone is satisfied by a degenerate grid -- one that collapsed every axis
        onto its lower bound is inside every interval. Asserting the endpoints and the count of
        distinct values per axis rules that out, while the intervals being pairwise disjoint
        still catches a per-axis mix-up in flatten().
        """
        bounds = [(0.0, 1.0), (-1.0, 1.0), (2.0, 3.0), (-0.5, 0.5), (10.0, 20.0)]
        grid = TensorProductGrid(bounds=bounds, Nx_points=[4] * 5, boundary_conditions=no_flux_bc(dimension=5))

        flat_points = grid.flatten()

        for d in range(5):
            assert np.all(flat_points[:, d] >= bounds[d][0])
            assert np.all(flat_points[:, d] <= bounds[d][1])
            # Endpoints reached exactly (measured exact on both ends of all five axes).
            assert flat_points[:, d].min() == pytest.approx(bounds[d][0])
            assert flat_points[:, d].max() == pytest.approx(bounds[d][1])
            # Nx_points=4 distinct coordinates on this axis, so the axis did not collapse.
            assert len(np.unique(flat_points[:, d])) == 4

    def test_spacing_4d(self):
        """Test uniform spacing in 4D."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0)] * 4, Nx_points=[11] * 4, boundary_conditions=no_flux_bc(dimension=4)
        )

        assert grid.is_uniform
        expected_spacing = 1.0 / 10.0  # (1.0 - 0.0) / (11 - 1)
        assert all(np.isclose(s, expected_spacing) for s in grid.spacing)


class TestHighDimensionalEdgeCases:
    """Test edge cases for high-dimensional grids."""

    def test_negative_dimension_raises(self):
        """Test that negative dimension raises error."""
        with pytest.raises(ValueError, match=r"dimension|doesn't match"):
            TensorProductGrid(
                dimension=-1, bounds=[(0.0, 1.0)], Nx_points=[10], boundary_conditions=no_flux_bc(dimension=1)
            )

    def test_zero_dimension_raises(self):
        """Test that zero dimension raises error."""
        with pytest.raises(ValueError, match=r"bounds|dimension"):
            TensorProductGrid(bounds=[], Nx_points=[], boundary_conditions=no_flux_bc(dimension=1))

    def test_bounds_dimension_mismatch(self):
        """Test that bounds/dimension mismatch raises error."""
        with pytest.raises(ValueError, match=r"must have length|mismatch"):
            TensorProductGrid(bounds=[(0.0, 1.0)] * 3, Nx_points=[10] * 4, boundary_conditions=no_flux_bc(dimension=4))

    def test_num_points_dimension_mismatch(self):
        """Test that num_points/dimension mismatch raises error."""
        with pytest.raises(ValueError, match="must have length 4"):
            TensorProductGrid(bounds=[(0.0, 1.0)] * 4, Nx_points=[10] * 3, boundary_conditions=no_flux_bc(dimension=4))


class TestBackwardCompatibility:
    """Ensure 1D/2D/3D grids still work as before."""

    def test_1d_grid_still_works(self):
        """Test that 1D grids work without warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[100], boundary_conditions=no_flux_bc(dimension=1))

        assert grid.dimension == 1
        assert grid.flatten().shape[0] == 100

    def test_2d_grid_still_works(self):
        """Test that 2D grids work without warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            grid = TensorProductGrid(
                bounds=[(0.0, 1.0), (0.0, 1.0)],
                Nx_points=[50, 50],
                boundary_conditions=no_flux_bc(dimension=2),
            )

        assert grid.dimension == 2
        assert grid.flatten().shape[0] == 2500

    def test_3d_grid_still_works(self):
        """Test that 3D grids work without warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            grid = TensorProductGrid(
                bounds=[(0.0, 1.0)] * 3,
                Nx_points=[20, 20, 20],
                boundary_conditions=no_flux_bc(dimension=3),
            )

        assert grid.dimension == 3
        assert grid.flatten().shape[0] == 8000

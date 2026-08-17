"""
Tests for particle interpolation utilities.

Tests cover:
- Grid-to-particle interpolation (1D, 2D, 3D)
- Particle-to-grid interpolation (1D, 2D, 3D)
- Different interpolation methods
- KDE bandwidth estimation
- Edge cases and error handling
"""

import pytest

import numpy as np

from mfgarchon.utils.numerical import (
    estimate_kde_bandwidth,
    interpolate_grid_to_particles,
    interpolate_particles_to_grid,
)


class TestGridToParticles1D:
    """Test 1D grid-to-particle interpolation."""

    def test_linear_interpolation(self):
        """Test linear interpolation on 1D grid."""
        # Create smooth function on grid
        x_grid = np.linspace(0, 1, 51)
        u_grid = np.sin(2 * np.pi * x_grid)

        # Interpolate to particles
        particles = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        u_particles = interpolate_grid_to_particles(u_grid, grid_bounds=(0, 1), particle_positions=particles)

        # Check values (sin is smooth, should be accurate)
        expected = np.sin(2 * np.pi * particles)
        np.testing.assert_allclose(u_particles, expected, atol=1e-2)

    def test_gaussian_function(self):
        """Test interpolation of Gaussian function."""
        x_grid = np.linspace(0, 1, 101)
        u_grid = np.exp(-50 * (x_grid - 0.5) ** 2)

        # Particles near center
        particles = np.array([0.4, 0.45, 0.5, 0.55, 0.6])
        u_particles = interpolate_grid_to_particles(u_grid, grid_bounds=(0, 1), particle_positions=particles)

        # Check values are positive and peak at center
        assert np.all(u_particles > 0)
        assert u_particles[2] == np.max(u_particles)  # Peak at 0.5


class TestGridToParticles2D:
    """Test 2D grid-to-particle interpolation."""

    def test_2d_gaussian(self):
        """Test 2D Gaussian interpolation."""
        x = y = np.linspace(0, 1, 51)
        X, Y = np.meshgrid(x, y, indexing="ij")
        u_grid = np.exp(-10 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))

        # Particles
        particles = np.array([[0.5, 0.5], [0.3, 0.3], [0.7, 0.7]])
        u_particles = interpolate_grid_to_particles(u_grid, grid_bounds=((0, 1), (0, 1)), particle_positions=particles)

        # Center should be maximum
        assert u_particles[0] > u_particles[1]
        assert u_particles[0] > u_particles[2]

    def test_2d_random_particles(self):
        """Test with random particle distribution."""
        x = y = np.linspace(0, 1, 31)
        X, Y = np.meshgrid(x, y, indexing="ij")
        u_grid = X + Y  # Simple linear function

        # Random particles
        np.random.seed(42)
        particles = np.random.uniform(0, 1, (50, 2))
        u_particles = interpolate_grid_to_particles(u_grid, grid_bounds=((0, 1), (0, 1)), particle_positions=particles)

        # Check linearity preserved
        expected = particles[:, 0] + particles[:, 1]
        np.testing.assert_allclose(u_particles, expected, atol=0.1)


class TestParticlesToGrid1D:
    """Test 1D particle-to-grid interpolation."""

    def test_rbf_method(self):
        """Test RBF interpolation from particles to grid."""
        # Create particles with known function
        np.random.seed(42)
        particles = np.sort(np.random.uniform(0, 1, 30))
        values = np.sin(2 * np.pi * particles)

        # Interpolate to grid
        u_grid = interpolate_particles_to_grid(values, particles, grid_shape=(51,), grid_bounds=(0, 1), method="rbf")

        # Check smoothness (no NaN or Inf)
        assert np.all(np.isfinite(u_grid))
        assert len(u_grid) == 51

        grid_x = np.linspace(0, 1, 51)

        # Polynomial reproduction: an RBF with linear polynomial augmentation reproduces an
        # affine function exactly, whatever the particle positions.  Measured 1.3e-15, so 1e-10
        # is ~1e5 margin.  This also pins the particle -> grid coordinate mapping, since the
        # right values at the wrong abscissae would not match 3x + 1.
        u_affine = interpolate_particles_to_grid(
            3.0 * particles + 1.0, particles, grid_shape=(51,), grid_bounds=(0, 1), method="rbf"
        )
        np.testing.assert_allclose(u_affine, 3.0 * grid_x + 1.0, atol=1e-10)

        # Accuracy on the smooth function the fixture actually builds, restricted to the convex
        # hull of the particles -- outside it the RBF extrapolates and no oracle governs it.
        # Measured max error 4.2e-3 inside the hull (median 5.6e-4), so 2e-2 is ~5x margin.
        inside_hull = (grid_x >= particles.min()) & (grid_x <= particles.max())
        np.testing.assert_allclose(u_grid[inside_hull], np.sin(2 * np.pi * grid_x)[inside_hull], atol=2e-2)

    def test_nearest_method(self):
        """Test nearest neighbor interpolation."""
        particles = np.array([0.2, 0.5, 0.8])
        values = np.array([1.0, 2.0, 3.0])

        u_grid = interpolate_particles_to_grid(
            values, particles, grid_shape=(11,), grid_bounds=(0, 1), method="nearest"
        )

        # Grid points near 0.5 should have value ~2.0
        assert np.abs(u_grid[5] - 2.0) < 0.1


class TestParticlesToGrid2D:
    """Test 2D particle-to-grid interpolation."""

    def test_2d_rbf(self):
        """Test 2D RBF interpolation."""
        # Create particles
        np.random.seed(42)
        particles = np.random.uniform(0, 1, (100, 2))
        values = np.exp(-10 * np.sum((particles - 0.5) ** 2, axis=1))

        # Interpolate to grid
        u_grid = interpolate_particles_to_grid(
            values, particles, grid_shape=(31, 31), grid_bounds=((0, 1), (0, 1)), method="rbf"
        )

        # Check shape and finite values
        assert u_grid.shape == (31, 31)
        assert np.all(np.isfinite(u_grid))

        # Maximum should be near center
        max_idx = np.unravel_index(np.argmax(u_grid), u_grid.shape)
        assert 10 < max_idx[0] < 20  # Near center in x
        assert 10 < max_idx[1] < 20  # Near center in y


class TestKDEBandwidth:
    """Test KDE bandwidth estimation."""

    def test_scott_rule_1d(self):
        """Test Scott's rule in 1D."""
        particles = np.random.randn(100, 1)
        bw = estimate_kde_bandwidth(particles, method="scott")

        # Should be reasonable (not too small or large)
        assert 0.1 < bw < 1.0

    def test_silverman_rule_2d(self):
        """Test Silverman's rule in 2D."""
        particles = np.random.randn(200, 2)
        bw = estimate_kde_bandwidth(particles, method="silverman")

        # Should be positive and reasonable
        assert bw > 0
        assert bw < 1.0

    def test_invalid_method(self):
        """Test error handling for invalid method."""
        particles = np.random.randn(50, 2)
        with pytest.raises(ValueError, match="Unknown method"):
            estimate_kde_bandwidth(particles, method="invalid")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_particle(self):
        """Test with single particle (using nearest neighbor since RBF needs >= 2 points)."""
        particle = np.array([0.5])
        value = np.array([1.0])

        # Use nearest neighbor method for single particle
        u_grid = interpolate_particles_to_grid(value, particle, grid_shape=(11,), grid_bounds=(0, 1), method="nearest")

        assert u_grid.shape == (11,)
        assert np.all(np.isfinite(u_grid))
        # All grid points should get the single particle's value
        assert np.all(u_grid == 1.0)

    def test_out_of_bounds_particles(self):
        """Test particles outside grid bounds."""
        x_grid = np.linspace(0, 1, 51)
        u_grid = np.sin(2 * np.pi * x_grid)

        # Particles outside bounds (should return fill_value=0.0)
        particles = np.array([-0.5, 1.5])
        u_particles = interpolate_grid_to_particles(u_grid, grid_bounds=(0, 1), particle_positions=particles)

        # Should be filled with zeros
        assert np.allclose(u_particles, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

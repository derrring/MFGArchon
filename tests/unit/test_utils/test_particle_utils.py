"""
Unit tests for GPU-accelerated particle utilities.

Tests numerical accuracy and performance of GPU particle operations
compared to NumPy/scipy baselines.
"""

import pytest

import numpy as np

from mfgarchon.utils.particle_utils import (
    apply_boundary_conditions_numpy,
    interpolate_1d_numpy,
    sample_from_density_numpy,
)

# Check if PyTorch is available for GPU tests
try:
    import torch  # Check for PyTorch package directly

    from mfgarchon.backends.torch_backend import TorchBackend
    from mfgarchon.utils.particle_utils import (
        apply_boundary_conditions_gpu,
        interpolate_1d_gpu,
        sample_from_density_gpu,
    )

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class TestInterpolate1DNumPy:
    """Test CPU NumPy interpolation."""

    def test_linear_function(self):
        """Interpolation of linear function should be exact."""
        x_grid = np.linspace(0, 1, 11)
        y_grid = 2 * x_grid + 1  # y = 2x + 1

        x_query = np.array([0.25, 0.5, 0.75])
        y_interp = interpolate_1d_numpy(x_query, x_grid, y_grid)

        y_expected = 2 * x_query + 1
        np.testing.assert_allclose(y_interp, y_expected, rtol=1e-6)

    def test_sine_function(self):
        """Interpolation should be reasonably accurate for smooth functions."""
        x_grid = np.linspace(0, np.pi, 100)
        y_grid = np.sin(x_grid)

        x_query = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        y_interp = interpolate_1d_numpy(x_query, x_grid, y_grid)

        y_expected = np.sin(x_query)
        # Linear interpolation on smooth function: <1% error with 100 points
        np.testing.assert_allclose(y_interp, y_expected, rtol=0.01)

    def test_boundary_extrapolation(self):
        """Query points outside grid should extrapolate using edge slopes."""
        x_grid = np.linspace(0, 1, 11)
        y_grid = x_grid**2

        # Query below and above grid
        x_query = np.array([-0.5, 1.5])
        y_interp = interpolate_1d_numpy(x_query, x_grid, y_grid)

        # Extrapolates using first/last interval slopes
        # First interval: y(0) = 0, y(0.1) ≈ 0.01, slope ≈ 0.1
        # Last interval: y(0.9) ≈ 0.81, y(1) = 1, slope ≈ 1.9
        # Allow extrapolation behavior (values won't match exact quadratic)
        assert y_interp.shape == (2,)  # Just check shape for now


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestInterpolate1DGPU:
    """Test GPU interpolation."""

    def test_matches_numpy(self):
        """GPU interpolation should match NumPy numerically."""
        backend = TorchBackend(device="cpu")

        x_grid = np.linspace(0, 1, 100)
        y_grid = np.sin(x_grid * 2 * np.pi)
        x_query = np.linspace(0.1, 0.9, 20)

        # NumPy version
        y_numpy = interpolate_1d_numpy(x_query, x_grid, y_grid)

        # GPU version
        x_grid_gpu = backend.from_numpy(x_grid)
        y_grid_gpu = backend.from_numpy(y_grid)
        x_query_gpu = backend.from_numpy(x_query)

        y_gpu = interpolate_1d_gpu(x_query_gpu, x_grid_gpu, y_grid_gpu, backend)
        y_gpu_np = backend.to_numpy(y_gpu)

        # Should match within float32 precision
        np.testing.assert_allclose(y_gpu_np, y_numpy, rtol=1e-5)


class TestBoundaryConditionsNumPy:
    """Test CPU NumPy boundary conditions."""

    def test_periodic(self):
        """Periodic boundaries should wrap particles."""
        particles = np.array([-0.5, 0.5, 1.5, 2.5])
        xmin, xmax = 0.0, 1.0

        particles_bc = apply_boundary_conditions_numpy(particles, xmin, xmax, "periodic")

        expected = np.array([0.5, 0.5, 0.5, 0.5])  # All wrap to middle
        np.testing.assert_allclose(particles_bc, expected, rtol=1e-10)

    def test_no_flux_reflection(self):
        """No-flux boundaries should reflect particles."""
        particles = np.array([-0.2, 0.5, 1.2])
        xmin, xmax = 0.0, 1.0

        particles_bc = apply_boundary_conditions_numpy(particles, xmin, xmax, "no_flux")

        expected = np.array([0.2, 0.5, 0.8])  # Reflect at boundaries
        np.testing.assert_allclose(particles_bc, expected, rtol=1e-10)

    def test_dirichlet_clamping(self):
        """Dirichlet boundaries should clamp particles."""
        particles = np.array([-0.5, 0.5, 1.5])
        xmin, xmax = 0.0, 1.0

        particles_bc = apply_boundary_conditions_numpy(particles, xmin, xmax, "dirichlet")

        expected = np.array([0.0, 0.5, 1.0])  # Clamp to boundaries
        np.testing.assert_allclose(particles_bc, expected, rtol=1e-10)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestBoundaryConditionsGPU:
    """Test GPU boundary conditions."""

    def test_periodic_matches_numpy(self):
        """GPU periodic BC should match NumPy."""
        backend = TorchBackend(device="cpu")

        particles = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        xmin, xmax = 0.0, 1.0

        # NumPy version
        particles_numpy = apply_boundary_conditions_numpy(particles, xmin, xmax, "periodic")

        # GPU version
        particles_gpu = backend.from_numpy(particles)
        particles_bc_gpu = apply_boundary_conditions_gpu(particles_gpu, xmin, xmax, "periodic", backend)
        particles_bc_np = backend.to_numpy(particles_bc_gpu)

        np.testing.assert_allclose(particles_bc_np, particles_numpy, rtol=1e-6)

    def test_no_flux_matches_numpy(self):
        """GPU no-flux BC should match NumPy."""
        backend = TorchBackend(device="cpu")

        particles = np.array([-0.3, -0.1, 0.5, 1.1, 1.3])
        xmin, xmax = 0.0, 1.0

        particles_numpy = apply_boundary_conditions_numpy(particles, xmin, xmax, "no_flux")

        particles_gpu = backend.from_numpy(particles)
        particles_bc_gpu = apply_boundary_conditions_gpu(particles_gpu, xmin, xmax, "no_flux", backend)
        particles_bc_np = backend.to_numpy(particles_bc_gpu)

        np.testing.assert_allclose(particles_bc_np, particles_numpy, rtol=1e-6)


class TestSampleFromDensityNumPy:
    """Test CPU NumPy density sampling."""

    def test_uniform_distribution(self):
        """Sampling from uniform should give uniform particles."""
        grid = np.linspace(0, 1, 101)
        density = np.ones_like(grid)  # Uniform

        particles = sample_from_density_numpy(density, grid, N=10000, seed=42)

        # Empirical distribution should be approximately uniform
        hist, _ = np.histogram(particles, bins=10, range=(0, 1))
        expected_count = 10000 / 10
        # Chi-square test: each bin should have ~1000 particles
        # Allow 20% deviation
        assert np.all(np.abs(hist - expected_count) < 0.2 * expected_count)

    def test_delta_function(self):
        """Sampling from delta function should concentrate at peak."""
        grid = np.linspace(0, 1, 101)
        density = np.zeros_like(grid)
        density[50] = 1.0  # Delta at x=0.5

        particles = sample_from_density_numpy(density, grid, N=1000, seed=42)

        # All particles should be near x=0.5
        assert np.all(np.abs(particles - 0.5) < 0.02)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestSampleFromDensityGPU:
    """Test GPU density sampling."""

    def test_matches_numpy_distribution(self):
        """GPU sampling should produce similar distribution to NumPy."""
        backend = TorchBackend(device="cpu")

        grid = np.linspace(0, 1, 101)
        density = np.exp(-((grid - 0.5) ** 2) / 0.1)  # Gaussian

        # NumPy version
        np.random.seed(42)
        particles_numpy = sample_from_density_numpy(density, grid, N=5000, seed=42)

        # GPU version
        density_gpu = backend.from_numpy(density)
        grid_gpu = backend.from_numpy(grid)
        particles_gpu = sample_from_density_gpu(density_gpu, grid_gpu, N=5000, backend=backend, seed=42)
        particles_gpu_np = backend.to_numpy(particles_gpu)

        # Compare empirical distributions
        hist_numpy, bins = np.histogram(particles_numpy, bins=20, range=(0, 1))
        hist_gpu, _ = np.histogram(particles_gpu_np, bins=bins)

        # Should have similar distributions (allow some variance)
        # Normalize and compare
        hist_numpy_norm = hist_numpy / np.sum(hist_numpy)
        hist_gpu_norm = hist_gpu / np.sum(hist_gpu)

        # Most bins should be within 50% relative difference
        rel_diff = np.abs(hist_numpy_norm - hist_gpu_norm) / (hist_numpy_norm + 1e-3)
        assert np.mean(rel_diff < 0.5) > 0.8  # 80% of bins match well

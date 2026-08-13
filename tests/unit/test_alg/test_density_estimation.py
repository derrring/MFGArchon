"""
Unit tests for GPU-accelerated density estimation.

Tests numerical accuracy and performance of gaussian_kde_gpu
compared to scipy.stats.gaussian_kde baseline.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.density_estimation import (
    adaptive_bandwidth_selection,
    gaussian_kde_gpu,
    gaussian_kde_numpy,
)

pytestmark = pytest.mark.optional_torch

# Check if PyTorch is available for GPU tests
try:
    import torch  # Check for PyTorch package directly

    from mfgarchon.backends.torch_backend import TorchBackend

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Check if scipy is available for baseline tests
try:
    import scipy.stats  # noqa: F401

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class TestAdaptiveBandwidthSelection:
    """Test automatic bandwidth selection methods."""

    def test_scott_rule(self):
        """Scott's rule should give reasonable bandwidth for standard normal."""
        np.random.seed(42)
        particles = np.random.randn(1000)

        bandwidth = adaptive_bandwidth_selection(particles, method="scott")

        # For N=1000, Scott's rule: h = σ N^(-1/5) ≈ 1.0 × 1000^(-0.2) ≈ 0.25
        assert 0.2 < bandwidth < 0.3

    def test_silverman_rule(self):
        """Silverman's rule should give similar bandwidth to Scott."""
        np.random.seed(42)
        particles = np.random.randn(1000)

        bandwidth = adaptive_bandwidth_selection(particles, method="silverman")

        # Should be close to Scott's rule for normal distribution
        assert 0.15 < bandwidth < 0.3


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not available")
class TestGaussianKDENumPy:
    """Test CPU fallback KDE using scipy."""

    def test_basic_kde(self):
        """Basic KDE should produce valid density."""
        np.random.seed(42)
        particles = np.random.randn(100)
        grid = np.linspace(-3, 3, 50)

        density = gaussian_kde_numpy(particles, grid, bandwidth=0.3)

        # Density should be non-negative
        assert np.all(density >= 0)

        # Density should integrate to approximately 1
        dx = grid[1] - grid[0]
        mass = np.sum(density) * dx
        assert 0.95 < mass < 1.05


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestGaussianKDEGPU:
    """Test GPU-accelerated KDE."""

    def test_gpu_kde_basic(self):
        """GPU KDE should produce valid density."""
        backend = TorchBackend(device="cpu")  # Use CPU for CI compatibility
        np.random.seed(42)
        particles = np.random.randn(100)
        grid = np.linspace(-3, 3, 50)

        density = gaussian_kde_gpu(particles, grid, bandwidth=0.3, backend=backend)

        # Density should be non-negative
        assert np.all(density >= 0)

        # Density should integrate to approximately 1
        dx = grid[1] - grid[0]
        mass = np.sum(density) * dx
        assert 0.95 < mass < 1.05

    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not available for comparison")
    def test_gpu_matches_scipy(self):
        """GPU KDE should match scipy.stats.gaussian_kde numerically."""
        backend = TorchBackend(device="cpu")
        np.random.seed(42)
        particles = np.random.randn(100)
        grid = np.linspace(-3, 3, 50)
        bandwidth = 0.3

        # GPU version
        density_gpu = gaussian_kde_gpu(particles, grid, bandwidth, backend)

        # CPU scipy version
        density_scipy = gaussian_kde_numpy(particles, grid, bandwidth)

        # Should match within float32 precision
        relative_error = np.abs(density_gpu - density_scipy) / (density_scipy + 1e-10)
        assert np.max(relative_error) < 0.01  # 1% relative error tolerance

    def test_large_particle_count(self):
        """GPU KDE should handle large particle ensembles."""
        backend = TorchBackend(device="cpu")
        np.random.seed(42)
        particles = np.random.randn(10000)  # 10k particles
        grid = np.linspace(-3, 3, 100)

        density = gaussian_kde_gpu(particles, grid, bandwidth=0.25, backend=backend)

        # Basic validity checks
        assert np.all(density >= 0)
        assert np.all(np.isfinite(density))

        # Mass conservation
        dx = grid[1] - grid[0]
        mass = np.sum(density) * dx
        assert 0.95 < mass < 1.05


@pytest.mark.skipif(not (TORCH_AVAILABLE and SCIPY_AVAILABLE), reason="Requires both PyTorch and scipy")
class TestPerformanceComparison:
    """Benchmark GPU vs CPU KDE performance."""

    def test_speedup_estimation(self):
        """Measure relative performance of GPU vs CPU KDE."""
        import time

        backend = TorchBackend(device="cpu")  # Use CPU for fair comparison
        np.random.seed(42)

        # Test with moderate particle count
        N = 5000
        particles = np.random.randn(N)
        grid = np.linspace(-3, 3, 100)
        bandwidth = 0.25

        # Warmup
        _ = gaussian_kde_gpu(particles, grid, bandwidth, backend)
        _ = gaussian_kde_numpy(particles, grid, bandwidth)

        # Time GPU version
        start = time.time()
        for _ in range(10):
            _ = gaussian_kde_gpu(particles, grid, bandwidth, backend)
        time_gpu = (time.time() - start) / 10

        # Time CPU version
        start = time.time()
        for _ in range(10):
            _ = gaussian_kde_numpy(particles, grid, bandwidth)
        time_cpu = (time.time() - start) / 10

        # On MPS device, GPU should be faster
        # On CPU device (CI), GPU might be similar or slower
        # Just verify both complete successfully
        assert time_gpu > 0
        assert time_cpu > 0

        print(f"\nKDE Performance (N={N}):")
        print(f"  CPU (scipy): {time_cpu * 1000:.2f} ms")
        print(f"  GPU (torch): {time_gpu * 1000:.2f} ms")
        if time_cpu > time_gpu:
            print(f"  Speedup: {time_cpu / time_gpu:.2f}x")


class TestWendlandC2Kernel:
    """Issue #1028: Wendland C^2 kernel option in 1D KDE.

    Validates the kernel form, normalization, and the compact-support
    property that motivates the option (eliminates Gaussian-tail spillover
    handled today by the measure-restriction operator).
    """

    def test_invalid_kernel_rejected(self):
        """Unknown kernel string raises ValueError."""
        from mfgarchon.alg.numerical.density_estimation import gaussian_kde_gpu
        from mfgarchon.backends.numpy_backend import NumPyBackend

        with pytest.raises(ValueError, match="Unknown kernel"):
            gaussian_kde_gpu(
                np.zeros(10),
                np.linspace(-1, 1, 5),
                0.1,
                NumPyBackend(),
                kernel="bogus",
            )

    def test_wendland_normalization_constant(self):
        """Z_1(delta) = 2*delta/3 by direct integration of phi(u) = (1-u)^4(4u+1)."""
        from mfgarchon.alg.numerical.density_estimation import (
            _wendland_c2_normalization_1d,
        )

        for delta in (0.1, 0.5, 1.0, 2.5):
            assert np.isclose(_wendland_c2_normalization_1d(delta), 2 * delta / 3)

    def test_wendland_unit_mass_single_particle(self):
        """Single particle at origin -> integrated density approaches 1.

        With delta = 0.5 and a fine grid covering the support, the
        Riemann sum approximates the analytic integral of K_h to machine
        precision (modulo trapezoidal error).
        """
        from mfgarchon.alg.numerical.density_estimation import gaussian_kde_gpu
        from mfgarchon.backends.numpy_backend import NumPyBackend

        backend = NumPyBackend()
        delta = 0.5
        grid = np.linspace(-3, 3, 1001)  # very fine
        dx = grid[1] - grid[0]
        density = gaussian_kde_gpu(np.array([0.0]), grid, delta, backend, kernel="wendland_c2")
        mass = np.sum(density) * dx
        assert 0.99 < mass < 1.01, f"Wendland mass deviates: {mass}"

    def test_wendland_compact_support(self):
        """Density is exactly zero at points beyond delta from every particle."""
        from mfgarchon.alg.numerical.density_estimation import gaussian_kde_gpu
        from mfgarchon.backends.numpy_backend import NumPyBackend

        backend = NumPyBackend()
        # Particles concentrated near origin
        particles = np.array([0.0, 0.05, -0.05])
        delta = 0.3
        grid = np.linspace(-2.0, 2.0, 401)
        density = gaussian_kde_gpu(particles, grid, delta, backend, kernel="wendland_c2")

        # Points farther than delta from any particle: density must be 0
        far_mask = np.array([np.min(np.abs(g - particles)) > delta for g in grid])
        assert np.all(density[far_mask] == 0.0), (
            "Wendland kernel must give zero density outside support of any particle"
        )

    def test_wendland_total_mass_preserves(self):
        """Wendland KDE preserves total mass to within trapezoidal-rule accuracy."""
        from mfgarchon.alg.numerical.density_estimation import gaussian_kde_gpu
        from mfgarchon.backends.numpy_backend import NumPyBackend

        rng = np.random.default_rng(0)
        N = 5000
        particles = rng.normal(0.0, 1.0, size=N)
        grid = np.linspace(-5, 5, 1001)
        dx = grid[1] - grid[0]
        density = gaussian_kde_gpu(particles, grid, 0.5, NumPyBackend(), kernel="wendland_c2")
        mass = np.sum(density) * dx
        assert 0.97 < mass < 1.03, f"Wendland total mass: {mass}"

    def test_gaussian_default_unchanged(self):
        """Calling without kernel= keeps the Gaussian default behavior."""
        from mfgarchon.alg.numerical.density_estimation import gaussian_kde_gpu
        from mfgarchon.backends.numpy_backend import NumPyBackend

        rng = np.random.default_rng(0)
        particles = rng.normal(0.0, 1.0, size=2000)
        grid = np.linspace(-3, 3, 200)
        density_default = gaussian_kde_gpu(particles, grid, 0.1, NumPyBackend())
        density_explicit = gaussian_kde_gpu(particles, grid, 0.1, NumPyBackend(), kernel="gaussian")
        np.testing.assert_array_equal(density_default, density_explicit)

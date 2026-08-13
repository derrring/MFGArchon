#!/usr/bin/env python3
"""Test PyTorch KDE implementation against scipy baseline."""

import pytest

import numpy as np

pytestmark = pytest.mark.optional_torch

# Check if PyTorch is available - import HAS_TORCH flag from torch_utils
try:
    from mfgarchon.utils.acceleration.torch_utils import HAS_TORCH
    from mfgarchon.utils.acceleration.torch_utils import GaussianKDE as TorchKDE

    TORCH_AVAILABLE = HAS_TORCH  # Use the actual torch availability flag
except ImportError:
    TORCH_AVAILABLE = False
    TorchKDE = None
    HAS_TORCH = False

# Test scipy KDE for comparison
try:
    from scipy.stats import gaussian_kde

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    gaussian_kde = None
    print("⚠️  Scipy not available, skipping comparison tests")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available (optional dependency)")
def test_basic_kde():
    """Test basic KDE functionality."""
    print("=" * 80)
    print("BASIC KDE TEST")
    print("=" * 80)

    # Create sample data
    np.random.seed(42)
    particles = np.random.normal(loc=0.5, scale=0.1, size=100)

    # Evaluation points
    x_eval = np.linspace(0, 1, 50)

    # PyTorch KDE
    torch_kde = TorchKDE(particles, bw_method="scott")
    torch_density = torch_kde(x_eval)

    print(f"\nParticles: {len(particles)} points")
    print(f"Evaluation: {len(x_eval)} points")
    print(f"PyTorch KDE bandwidth: {torch_kde.bandwidth:.6f}")
    print(f"PyTorch density range: [{torch_density.min():.6f}, {torch_density.max():.6f}]")
    print(f"PyTorch density integral (trapz): {np.trapezoid(torch_density, x_eval):.6f}")

    # Compare with scipy if available
    if HAS_SCIPY:
        scipy_kde = gaussian_kde(particles, bw_method="scott")
        scipy_density = scipy_kde(x_eval)

        print(f"\nScipy KDE bandwidth: {scipy_kde.factor * scipy_kde.covariance[0, 0] ** 0.5:.6f}")
        print(f"Scipy density range: [{scipy_density.min():.6f}, {scipy_density.max():.6f}]")
        print(f"Scipy density integral (trapz): {np.trapezoid(scipy_density, x_eval):.6f}")

        # Compute relative error
        rel_error = np.abs(torch_density - scipy_density) / (scipy_density + 1e-10)
        mean_rel_error = np.mean(rel_error)
        max_rel_error = np.max(rel_error)

        print("\n✅ Comparison:")
        print(f"  Mean relative error: {mean_rel_error:.6f}")
        print(f"  Max relative error: {max_rel_error:.6f}")

        # Validation
        if mean_rel_error < 0.01:  # < 1% average error
            print("  ✅ PyTorch KDE matches scipy KDE (< 1% average error)")
        else:
            print(f"  ⚠️  Warning: Mean error {mean_rel_error:.2%} exceeds 1%")

    else:
        print("\n✅ PyTorch KDE computed successfully (scipy not available for comparison)")

    # Verify KDE produces finite, non-negative densities
    assert np.all(np.isfinite(torch_density)), "KDE density contains non-finite values"
    assert np.all(torch_density >= 0), "KDE density contains negative values"
    if HAS_SCIPY:
        assert mean_rel_error < 0.05, f"Mean relative error vs scipy too large: {mean_rel_error}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available (optional dependency)")
def test_bandwidth_methods():
    """Test different bandwidth selection methods."""
    print("\n" + "=" * 80)
    print("BANDWIDTH METHODS TEST")
    print("=" * 80)

    np.random.seed(42)
    particles = np.random.normal(loc=0.5, scale=0.15, size=50)
    x_eval = np.linspace(0, 1, 30)

    for method in ["scott", "silverman", 0.1]:
        method_name = method if isinstance(method, str) else f"fixed={method}"
        kde = TorchKDE(particles, bw_method=method)
        density = kde(x_eval)

        print(f"\n{method_name}:")
        print(f"  Bandwidth: {kde.bandwidth:.6f}")
        print(f"  Peak density: {density.max():.6f}")
        print(f"  Integral: {np.trapezoid(density, x_eval):.6f}")

        assert kde.bandwidth > 0, f"Bandwidth should be positive, got {kde.bandwidth}"
        assert np.all(np.isfinite(density)), f"Density not finite for method {method_name}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available (optional dependency)")
def test_edge_cases():
    """Test edge cases and robustness."""
    print("\n" + "=" * 80)
    print("EDGE CASES TEST")
    print("=" * 80)

    # Single particle
    print("\n1. Single particle:")
    kde_single = TorchKDE(np.array([0.5]), bw_method=0.1)
    x_eval = np.linspace(0, 1, 50)
    density_single = kde_single(x_eval)
    print(f"   Density at particle location (x=0.5): {density_single[25]:.6f}")
    print(f"   Integral: {np.trapezoid(density_single, x_eval):.6f}")

    # Two particles
    print("\n2. Two particles:")
    kde_two = TorchKDE(np.array([0.3, 0.7]), bw_method=0.1)
    density_two = kde_two(x_eval)
    print(f"   Density has two modes: {len(find_peaks(density_two))} peaks detected")
    print(f"   Integral: {np.trapezoid(density_two, x_eval):.6f}")

    # Many particles (performance test)
    print("\n3. Large dataset (performance):")
    import time

    particles_large = np.random.normal(loc=0.5, scale=0.1, size=10000)
    x_eval_large = np.linspace(0, 1, 500)

    start = time.time()
    kde_large = TorchKDE(particles_large, bw_method="scott")
    density_large = kde_large(x_eval_large)
    elapsed = time.time() - start

    print(f"   {len(particles_large)} particles, {len(x_eval_large)} evaluation points")
    print(f"   Time: {elapsed:.4f} seconds")
    print(f"   Integral: {np.trapezoid(density_large, x_eval_large):.6f}")

    # Verify all edge case results are finite
    assert np.all(np.isfinite(density_single)), "Single particle KDE not finite"
    assert np.all(np.isfinite(density_two)), "Two particle KDE not finite"
    assert np.all(np.isfinite(density_large)), "Large dataset KDE not finite"

    # Each of these edge cases has a closed form, which the finiteness checks above cannot see.
    # A scalar bw_method is the ABSOLUTE bandwidth here (measured: kde.bandwidth == 0.1 for both
    # the 1- and 2-particle cases), so a single particle at 0.5 IS the unit Gaussian kernel and
    # two particles are an equal-weight mixture of two of them. Measured max relative deviation
    # 1.29e-06 for the single (float32 accumulation in torch), so rtol=1e-5 keeps a 7.8x margin;
    # 1e-6 would be under the floor. The mixture form subsumes the discarded find_peaks count --
    # a density matching it to 4.2e-07 has exactly the two modes the print claims.
    def kernel(x, mu):
        return np.exp(-((x - mu) ** 2) / (2 * 0.1**2)) / (0.1 * np.sqrt(2 * np.pi))

    np.testing.assert_allclose(density_single, kernel(x_eval, 0.5), rtol=1e-5)
    np.testing.assert_allclose(density_two, 0.5 * (kernel(x_eval, 0.3) + kernel(x_eval, 0.7)), rtol=1e-5)

    # A KDE is a probability density: it must integrate to 1. N(0.5, 0.1) puts 5 sigma inside
    # [0, 1], so the truncation loss is ~6e-07 and the measured integral is 0.99999..1.00000 over
    # 20 draws; abs=0.01 is a 1000x margin and still catches a missing 1/N or 1/h normalisation.
    assert np.trapezoid(density_large, x_eval_large) == pytest.approx(1.0, abs=0.01)


def find_peaks(signal, threshold=0.01):
    """Simple peak detection."""
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1] and signal[i] > threshold:
            peaks.append(i)
    return peaks


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available (optional dependency)")
def test_device_compatibility():
    """Test device compatibility (CPU, CUDA, MPS)."""
    print("\n" + "=" * 80)
    print("DEVICE COMPATIBILITY TEST")
    print("=" * 80)

    from mfgarchon.utils.acceleration.torch_utils import HAS_CUDA, HAS_MPS

    particles = np.random.normal(loc=0.5, scale=0.1, size=100)
    x_eval = np.linspace(0, 1, 50)

    # CPU
    print("\n1. CPU:")
    kde_cpu = TorchKDE(particles, bw_method="scott", device="cpu")
    density_cpu = kde_cpu(x_eval)
    print(f"   ✅ CPU KDE works, peak density: {density_cpu.max():.6f}")
    assert np.all(np.isfinite(density_cpu)), "CPU KDE density not finite"

    # CUDA
    if HAS_CUDA:
        print("\n2. CUDA:")
        kde_cuda = TorchKDE(particles, bw_method="scott", device="cuda")
        density_cuda = kde_cuda(x_eval)
        print(f"   ✅ CUDA KDE works, peak density: {density_cuda.max():.6f}")

        # Verify consistency: the same KDE on two backends must agree (see the MPS branch below
        # for the measured threshold; CUDA is not available on this machine to measure against).
        rel_error = np.abs(density_cpu - density_cuda) / (density_cpu + 1e-10)
        print(f"   CPU vs CUDA max error: {np.max(rel_error):.2e}")
        assert np.max(rel_error) < 1e-4, f"CPU vs CUDA KDE disagree: {np.max(rel_error):.2e}"
    else:
        print("\n2. CUDA: ❌ Not available")

    # MPS
    if HAS_MPS:
        print("\n3. MPS (Apple Silicon):")
        kde_mps = TorchKDE(particles, bw_method="scott", device="mps")
        density_mps = kde_mps(x_eval)
        print(f"   ✅ MPS KDE works, peak density: {density_mps.max():.6f}")

        # Verify consistency. This is the whole subject of the test and it was computed and
        # thrown away: a backend returning systematically wrong densities passed. The pointwise
        # ratio is dominated by the distribution tails, where density_cpu falls to ~2e-14 and the
        # 1e-10 floor takes over, so the statistic is noisy by construction -- measured over 150
        # unseeded draws it tops out at 5.85e-06 (median 2.45e-07), and 1e-4 keeps a 17x margin
        # while still catching any real disagreement, which is orders larger.
        rel_error = np.abs(density_cpu - density_mps) / (density_cpu + 1e-10)
        print(f"   CPU vs MPS max error: {np.max(rel_error):.2e}")
        assert np.max(rel_error) < 1e-4, f"CPU vs MPS KDE disagree: {np.max(rel_error):.2e}"
    else:
        print("\n3. MPS: ❌ Not available")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("PYTORCH KDE IMPLEMENTATION TEST")
    print("=" * 80)

    test_basic_kde()
    test_bandwidth_methods()
    test_edge_cases()
    test_device_compatibility()

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("""
✅ PyTorch KDE implemented successfully
✅ Matches scipy.stats.gaussian_kde behavior
✅ Supports scott/silverman/fixed bandwidth
✅ Handles edge cases correctly
✅ Device-agnostic (CPU/CUDA/MPS)

Next Steps:
1. Integrate into fp_particle.py solver
2. Add backend-aware KDE selection
3. Performance benchmark vs scipy
""")


if __name__ == "__main__":
    main()

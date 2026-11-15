"""
Benchmark: GPU-Accelerated KDE Performance for Particle Solvers

This benchmark measures the actual speedup achieved by GPU-accelerated
Gaussian KDE compared to CPU scipy baseline for particle-based MFG solvers.

Tests multiple particle counts (N) to demonstrate scaling behavior.

Results are saved to: examples/outputs/benchmarks/
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfgarchon.utils.numerical.particle.density_estimation import (
    gaussian_kde_gpu,
    gaussian_kde_numpy,
)

# Backend availability checks
TORCH_AVAILABLE = False
SCIPY_AVAILABLE = False

try:
    from mfgarchon.backends.torch_backend import TorchBackend

    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available. GPU benchmarks will be skipped.")

try:
    import scipy.stats  # noqa: F401

    SCIPY_AVAILABLE = True
except ImportError:
    print("Error: scipy is required for CPU baseline. Install with: pip install scipy")
    exit(1)


def benchmark_kde_single(particles, grid, bandwidth, backend=None, warmup=True, n_runs=10):
    """
    Benchmark KDE performance for single configuration.

    Parameters
    ----------
    particles : np.ndarray
        Particle positions, shape (N,)
    grid : np.ndarray
        Grid points, shape (Nx,)
    bandwidth : float
        Bandwidth factor
    backend : BaseBackend, optional
        If provided, use GPU. If None, use CPU scipy.
    warmup : bool
        Whether to run warmup iterations
    n_runs : int
        Number of timing runs

    Returns
    -------
    dict
        Timing results and statistics
    """
    N = len(particles)
    Nx = len(grid)

    if backend is not None:
        # GPU version
        def kde_func():
            return gaussian_kde_gpu(particles, grid, bandwidth, backend)

        device = backend.device if hasattr(backend, "device") else "unknown"
    else:
        # CPU scipy version
        def kde_func():
            return gaussian_kde_numpy(particles, grid, bandwidth)

        device = "cpu"

    # Warmup runs (important for GPU)
    if warmup:
        for _ in range(3):
            _ = kde_func()

    # Timing runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        density = kde_func()
        end = time.perf_counter()
        times.append(end - start)

    # Statistics
    times_ms = np.array(times) * 1000
    results = {
        "N": N,
        "Nx": Nx,
        "device": device,
        "mean_ms": np.mean(times_ms),
        "std_ms": np.std(times_ms),
        "min_ms": np.min(times_ms),
        "max_ms": np.max(times_ms),
        "median_ms": np.median(times_ms),
        "density_shape": density.shape,
        "density_mass": np.sum(density) * (grid[1] - grid[0]),
    }

    return results


def run_comprehensive_benchmark():
    """
    Run comprehensive KDE benchmark across multiple particle counts.

    Tests CPU (scipy) vs GPU (PyTorch MPS/CUDA) performance.
    """
    print("=" * 70)
    print("GPU-Accelerated KDE Benchmark for Particle Solvers")
    print("=" * 70)
    print()

    # Configuration
    particle_counts = [1000, 5000, 10000, 50000, 100000]
    grid_size = 100
    bandwidth = 0.3
    n_runs = 10

    # Setup output directory
    output_dir = Path(__file__).parent.parent / "outputs" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize backend for GPU tests
    if TORCH_AVAILABLE:
        backend_mps = TorchBackend(device="mps")
        print(f"GPU Backend: {backend_mps.device}")
        print()
    else:
        backend_mps = None
        print("GPU Backend: Not available")
        print()

    # Results storage
    results_cpu = []
    results_gpu = []

    # Run benchmarks for each particle count
    for N in particle_counts:
        print(f"Testing N={N:,} particles...")
        print("-" * 70)

        # Generate test data
        np.random.seed(42)
        particles = np.random.randn(N)
        grid = np.linspace(-3, 3, grid_size)

        # CPU baseline (scipy)
        print("  CPU (scipy): ", end="", flush=True)
        cpu_result = benchmark_kde_single(particles, grid, bandwidth, backend=None, warmup=True, n_runs=n_runs)
        results_cpu.append(cpu_result)
        print(f"{cpu_result['mean_ms']:.2f} ± {cpu_result['std_ms']:.2f} ms")

        # GPU (PyTorch MPS)
        if backend_mps is not None:
            print("  GPU (MPS):   ", end="", flush=True)
            gpu_result = benchmark_kde_single(
                particles, grid, bandwidth, backend=backend_mps, warmup=True, n_runs=n_runs
            )
            results_gpu.append(gpu_result)
            print(f"{gpu_result['mean_ms']:.2f} ± {gpu_result['std_ms']:.2f} ms")

            # Speedup calculation
            speedup = cpu_result["mean_ms"] / gpu_result["mean_ms"]
            print(f"  Speedup:     {speedup:.2f}x")

            # Verify numerical accuracy
            np.random.seed(42)  # Reset for identical data
            particles_test = np.random.randn(100)  # Small test
            grid_test = np.linspace(-3, 3, 50)
            density_cpu = gaussian_kde_numpy(particles_test, grid_test, bandwidth)
            density_gpu = gaussian_kde_gpu(particles_test, grid_test, bandwidth, backend_mps)
            rel_error = np.max(np.abs(density_gpu - density_cpu) / (density_cpu + 1e-10))
            print(f"  Accuracy:    {rel_error * 100:.3f}% max relative error")
        else:
            print("  GPU (MPS):   Skipped (PyTorch not available)")

        print()

    # Summary statistics
    print("=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print()

    if results_gpu:
        print("Performance Scaling:")
        print(f"{'N':>10} | {'CPU (ms)':>12} | {'GPU (ms)':>12} | {'Speedup':>10}")
        print("-" * 70)
        for cpu_res, gpu_res in zip(results_cpu, results_gpu, strict=False):
            N = cpu_res["N"]
            cpu_time = cpu_res["mean_ms"]
            gpu_time = gpu_res["mean_ms"]
            speedup = cpu_time / gpu_time
            print(f"{N:>10,} | {cpu_time:>12.2f} | {gpu_time:>12.2f} | {speedup:>10.2f}x")

        # Create visualization
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Absolute timing
        N_vals = [r["N"] for r in results_cpu]
        cpu_times = [r["mean_ms"] for r in results_cpu]
        gpu_times = [r["mean_ms"] for r in results_gpu]

        ax1.loglog(N_vals, cpu_times, "o-", label="CPU (scipy)", linewidth=2, markersize=8)
        ax1.loglog(N_vals, gpu_times, "s-", label="GPU (MPS)", linewidth=2, markersize=8)
        ax1.set_xlabel("Number of Particles (N)", fontsize=12)
        ax1.set_ylabel("Time (ms)", fontsize=12)
        ax1.set_title("KDE Performance: CPU vs GPU", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Speedup scaling
        speedups = [cpu_times[i] / gpu_times[i] for i in range(len(N_vals))]
        ax2.semilogx(N_vals, speedups, "d-", color="green", linewidth=2, markersize=8)
        ax2.axhline(y=10, color="red", linestyle="--", alpha=0.5, label="10x speedup target")
        ax2.set_xlabel("Number of Particles (N)", fontsize=12)
        ax2.set_ylabel("Speedup (×)", fontsize=12)
        ax2.set_title("GPU Speedup vs Particle Count", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "kde_gpu_benchmark.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print()
        print(f"Plot saved to: {plot_path}")

        # Save numerical results
        results_path = output_dir / "kde_gpu_benchmark_results.txt"
        with open(results_path, "w") as f:
            f.write("GPU-Accelerated KDE Benchmark Results\n")
            f.write("=" * 70 + "\n\n")
            f.write("Configuration:\n")
            f.write(f"  Grid size: {grid_size}\n")
            f.write(f"  Bandwidth: {bandwidth}\n")
            f.write(f"  Runs per test: {n_runs}\n\n")
            f.write("Performance Scaling:\n")
            f.write(f"{'N':>10} | {'CPU (ms)':>12} | {'GPU (ms)':>12} | {'Speedup':>10}\n")
            f.write("-" * 70 + "\n")
            for cpu_res, gpu_res in zip(results_cpu, results_gpu, strict=False):
                N = cpu_res["N"]
                cpu_time = cpu_res["mean_ms"]
                gpu_time = gpu_res["mean_ms"]
                speedup = cpu_time / gpu_time
                f.write(f"{N:>10,} | {cpu_time:>12.2f} | {gpu_time:>12.2f} | {speedup:>10.2f}x\n")

            f.write(f"\n\nMaximum speedup: {max(speedups):.2f}x (N={N_vals[speedups.index(max(speedups))]:,})\n")
            f.write(f"Average speedup: {np.mean(speedups):.2f}x\n")

        print(f"Results saved to: {results_path}")

        # Analysis
        max_speedup = max(speedups)
        max_speedup_N = N_vals[speedups.index(max_speedup)]
        avg_speedup = np.mean(speedups)

        print()
        print("=" * 70)
        print("Analysis")
        print("=" * 70)
        print(f"Maximum speedup: {max_speedup:.2f}x (achieved at N={max_speedup_N:,})")
        print(f"Average speedup: {avg_speedup:.2f}x")
        print()

        if max_speedup >= 10:
            print("✅ Phase 1 target achieved: >10x speedup for large N")
        else:
            print(f"⚠️  Phase 1 target: {max_speedup:.2f}x / 10x speedup")

        # Projected overall solver speedup
        kde_fraction = 0.70  # KDE is 70% of particle solver time
        overall_speedup = 1 / ((1 - kde_fraction) + kde_fraction / avg_speedup)
        print(f"\nProjected overall particle solver speedup: {overall_speedup:.2f}x")
        print(f"  (Assumes KDE is {kde_fraction * 100:.0f}% of total time)")

    else:
        print("GPU benchmarks skipped (PyTorch not available)")
        print("Only CPU baseline results:")
        for res in results_cpu:
            print(f"  N={res['N']:>6,}: {res['mean_ms']:>8.2f} ms")

    print()
    print("=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_comprehensive_benchmark()

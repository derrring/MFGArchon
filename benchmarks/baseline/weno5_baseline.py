#!/usr/bin/env python3
"""
WENO5 Baseline Performance Measurement

Establishes performance baseline for WENO5 smoothness indicators
to determine if optimization is necessary.

Decision criteria:
- If baseline > 100 μs → Optimization worth it
- If baseline < 10 μs → Not worth optimizing
- If baseline 10-100 μs → Borderline
"""

import timeit
from pathlib import Path

import numpy as np
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def benchmark_smoothness_indicators():
    """Benchmark WENO5 smoothness indicators."""
    from mfgarchon import MFGProblem
    from mfgarchon.alg.numerical.hjb_solvers.hjb_weno import HJBWenoSolver

    # Create solver
    problem = MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[100 + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        Nt=50,
        T=1.0,
    )
    solver = HJBWenoSolver(problem, weno_variant="weno5")

    # Test data (typical 5-point stencil)
    u = np.random.randn(5)

    # Warmup (important for consistent measurements)
    for _ in range(100):
        solver._compute_smoothness_indicators(u)

    # Benchmark with timeit (more accurate than manual timing)
    times = timeit.repeat(lambda: solver._compute_smoothness_indicators(u), number=10000, repeat=10)

    # Statistics
    median_time_us = np.median(times) / 10000 * 1e6
    min_time_us = min(times) / 10000 * 1e6
    max_time_us = max(times) / 10000 * 1e6
    std_time_us = np.std(times) / 10000 * 1e6

    # Print results
    print("=" * 60)
    print("WENO5 SMOOTHNESS INDICATORS - BASELINE PERFORMANCE")
    print("=" * 60)
    print("Test configuration:")
    print(f"  Stencil size: {u.shape}")
    print("  Iterations:   10,000")
    print("  Repeats:      10")
    print()
    print("Performance:")
    print(f"  Median time: {median_time_us:.2f} μs per call")
    print(f"  Min time:    {min_time_us:.2f} μs")
    print(f"  Max time:    {max_time_us:.2f} μs")
    print(f"  Std dev:     {std_time_us:.2f} μs")
    print("=" * 60)

    # Decision guidance
    print()
    print("OPTIMIZATION RECOMMENDATION:")
    if median_time_us > 100:
        print(f"  🔴 DEFINITELY OPTIMIZE (baseline {median_time_us:.0f} μs > 100 μs threshold)")
        print("     → Expected to benefit significantly from acceleration")
    elif median_time_us < 10:
        print(f"  🟢 NOT WORTH OPTIMIZING (baseline {median_time_us:.0f} μs < 10 μs threshold)")
        print("     → Already fast enough, focus elsewhere")
    else:
        print(f"  🟡 BORDERLINE (baseline {median_time_us:.0f} μs in 10-100 μs range)")
        print("     → May benefit from optimization, proceed with caution")

    # Extrapolate to full solve
    Nx, Nt = 100, 50
    iterations = 10  # typical fixed-point iterations
    calls_per_solve = Nx * Nt * iterations  # rough estimate
    total_time_s = calls_per_solve * median_time_us / 1e6

    print()
    print(f"EXTRAPOLATION TO FULL SOLVE (Nx={Nx}, Nt={Nt}, {iterations} iterations):")
    print(f"  Estimated calls:     {calls_per_solve:,}")
    print(f"  Est. smoothness time: {total_time_s:.2f} s")
    print(f"  (This is {total_time_s / 60:.1f} minutes if > 60s)")

    if total_time_s > 10:
        print("  ⚠️ WARNING: Smoothness computation takes significant time")
        print("     → Strong candidate for optimization")

    print("=" * 60)

    # Save results
    output_file = Path(__file__).parent / "baseline_results.txt"
    with open(output_file, "w") as f:
        f.write("WENO5 Baseline Performance\n")
        f.write("Date: 2025-10-09\n")
        f.write(f"Median: {median_time_us:.2f} μs\n")
        f.write(f"Min: {min_time_us:.2f} μs\n")
        f.write(f"Max: {max_time_us:.2f} μs\n")
        f.write(f"Std: {std_time_us:.2f} μs\n")

    print(f"\n✅ Results saved to: {output_file}")

    return median_time_us


if __name__ == "__main__":
    baseline_time = benchmark_smoothness_indicators()

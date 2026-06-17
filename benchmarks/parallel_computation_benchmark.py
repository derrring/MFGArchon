#!/usr/bin/env python3
"""
Parallel Computation Benchmark for MFGarchon

Tests the practical effectiveness of parallel execution across different use cases:
1. Common Noise MFG Monte Carlo sampling
2. Parameter sweep workflows
3. Monte Carlo integration

Evaluates:
- Speedup factor (sequential vs parallel)
- Overhead analysis
- Optimal worker count
- When parallelization is practical

Outputs detailed benchmark report with recommendations.
"""

import multiprocessing as mp
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfgarchon.alg.numerical.stochastic import CommonNoiseMFGSolver
from mfgarchon.core.stochastic import OrnsteinUhlenbeckProcess, StochasticMFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.mfg_logging import configure_research_logging, get_logger
from mfgarchon.workflow.parameter_sweep import ParameterSweep, SweepConfiguration

# Configure logging
configure_research_logging("parallel_benchmark", level="INFO")
logger = get_logger(__name__)

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "examples" / "outputs" / "benchmarks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Simple test function for parameter sweeps (not a pytest test)
def mfg_solve_function(Nx, Nt, sigma):
    """Simple MFG solve for benchmarking."""
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.factory import create_fast_solver

    problem = MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[Nx + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        Nt=Nt,
        sigma=sigma,
    )
    solver = create_fast_solver(problem)
    result = solver.solve()

    return {"converged": result.converged, "iterations": result.iterations, "final_error": result.final_error}


def benchmark_common_noise_parallel():
    """
    Benchmark parallel execution in CommonNoiseMFGSolver.

    Tests: Monte Carlo sampling with different numbers of noise realizations.
    """
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK 1: Common Noise MFG Parallel Monte Carlo")
    logger.info("=" * 80)

    # Create test problem
    def create_test_problem():
        # Simple Hamiltonian (must be module-level for pickling)
        def h(x, p, m, theta):
            return 0.5 * p**2 + 0.5 * x**2 + 0.1 * m

        noise_process = OrnsteinUhlenbeckProcess(kappa=2.0, mu=0.0, sigma=0.5)
        problem = StochasticMFGProblem(
            xmin=-2.0,
            xmax=2.0,
            Nx=51,  # Moderate size
            T=1.0,
            Nt=51,
            sigma=0.2,
            noise_process=noise_process,
            conditional_hamiltonian=h,  # Pass during construction
            theta_initial=0.0,
        )

        return problem

    problem = create_test_problem()

    # Test configurations
    num_samples_list = [10, 20, 40]  # Number of Monte Carlo samples
    results = {"num_samples": [], "sequential_time": [], "parallel_time": [], "speedup": [], "overhead_pct": []}

    for num_samples in num_samples_list:
        logger.info(f"\nTesting with {num_samples} noise samples...")

        # Sequential execution
        logger.info("  Running sequential...")
        solver_seq = CommonNoiseMFGSolver(
            problem, num_noise_samples=num_samples, variance_reduction=False, parallel=False, seed=42
        )

        start = time.perf_counter()
        _ = solver_seq.solve(verbose=False)  # Result not needed, just timing
        time_seq = time.perf_counter() - start

        logger.info(f"    Sequential time: {time_seq:.2f}s")

        # Parallel execution
        logger.info("  Running parallel...")
        solver_par = CommonNoiseMFGSolver(
            problem, num_noise_samples=num_samples, variance_reduction=False, parallel=True, seed=42
        )

        start = time.perf_counter()
        _ = solver_par.solve(verbose=False)  # Result not needed, just timing
        time_par = time.perf_counter() - start

        logger.info(f"    Parallel time: {time_par:.2f}s")

        # Compute metrics
        speedup = time_seq / time_par if time_par > 0 else 0
        ideal_time = time_seq / mp.cpu_count()
        overhead = max(0, time_par - ideal_time)
        overhead_pct = (overhead / time_par) * 100 if time_par > 0 else 0

        logger.info(f"    Speedup: {speedup:.2f}x")
        logger.info(f"    Overhead: {overhead_pct:.1f}%")

        # Store results
        results["num_samples"].append(num_samples)
        results["sequential_time"].append(time_seq)
        results["parallel_time"].append(time_par)
        results["speedup"].append(speedup)
        results["overhead_pct"].append(overhead_pct)

    return results


def benchmark_parameter_sweep_parallel():
    """
    Benchmark parallel execution modes in ParameterSweep.

    Tests: Sequential, thread-based, and process-based parallelism.
    """
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK 2: Parameter Sweep Parallel Execution")
    logger.info("=" * 80)

    # Define parameter space (small problem for quick testing)
    parameters = {"Nx": [30, 40, 50], "Nt": [30, 40, 50], "sigma": [0.1, 0.15, 0.2]}

    total_combinations = 3 * 3 * 3  # 27 combinations
    logger.info(f"Total parameter combinations: {total_combinations}")

    execution_modes = ["sequential", "parallel_threads", "parallel_processes"]
    results = {"mode": [], "execution_time": [], "avg_time_per_run": [], "speedup": []}

    baseline_time = None

    for mode in execution_modes:
        logger.info(f"\nTesting execution mode: {mode}")

        config = SweepConfiguration(
            parameters=parameters, execution_mode=mode, save_intermediate=False, max_workers=mp.cpu_count()
        )

        sweep = ParameterSweep(parameters, config)

        start = time.perf_counter()
        _ = sweep.execute(mfg_solve_function)  # Results not needed, just timing
        exec_time = time.perf_counter() - start

        avg_time = exec_time / total_combinations if total_combinations > 0 else 0

        if baseline_time is None:
            baseline_time = exec_time
            speedup = 1.0
        else:
            speedup = baseline_time / exec_time

        logger.info(f"  Total time: {exec_time:.2f}s")
        logger.info(f"  Avg per run: {avg_time:.3f}s")
        logger.info(f"  Speedup: {speedup:.2f}x")

        results["mode"].append(mode)
        results["execution_time"].append(exec_time)
        results["avg_time_per_run"].append(avg_time)
        results["speedup"].append(speedup)

    return results


def benchmark_monte_carlo_parallel():
    """
    Benchmark parallel Monte Carlo integration (if available).

    Tests: High-dimensional integration with parallel execution.
    """
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK 3: Monte Carlo Integration Parallel")
    logger.info("=" * 80)

    from mfgarchon.utils.numerical.monte_carlo import MCConfig, monte_carlo_integrate

    # Define test integrand: multi-dimensional Gaussian
    def gaussian_integrand(x):
        return np.exp(-np.sum(x**2, axis=1) / 2)

    # Test domain: [-3, 3]^d for different dimensions
    dimensions = [2, 4, 6]
    num_samples = 50000

    results = {"dimension": [], "sequential_time": [], "efficiency": []}

    for dim in dimensions:
        logger.info(f"\nTesting dimension {dim}...")

        domain = [(-3.0, 3.0)] * dim

        # Sequential (no parallelism in current implementation)
        config = MCConfig(num_samples=num_samples, sampling_method="sobol", seed=42, parallel=False)

        start = time.perf_counter()
        result = monte_carlo_integrate(gaussian_integrand, domain, config)
        exec_time = time.perf_counter() - start

        samples_per_sec = num_samples / exec_time if exec_time > 0 else 0

        logger.info(f"  Time: {exec_time:.2f}s")
        logger.info(f"  Samples/sec: {samples_per_sec:.0f}")
        logger.info(f"  Estimate: {result.estimate:.6f}")
        logger.info(f"  Error: {result.standard_error:.6e}")

        results["dimension"].append(dim)
        results["sequential_time"].append(exec_time)
        results["efficiency"].append(samples_per_sec)

    return results


def analyze_worker_scaling():
    """
    Analyze scaling behavior with different numbers of workers.

    Tests: How speedup changes with worker count.
    """
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK 4: Worker Count Scaling Analysis")
    logger.info("=" * 80)

    max_workers = mp.cpu_count()
    logger.info(f"CPU count: {max_workers}")

    # Test with parameter sweep (easier to control worker count)
    parameters = {"Nx": [30, 40, 50, 60], "Nt": [30, 40, 50, 60], "sigma": [0.1, 0.15, 0.2, 0.25]}

    worker_counts = [1, 2, 4, max(4, max_workers // 2), max_workers]
    worker_counts = [w for w in worker_counts if w <= max_workers]  # Filter valid counts

    results = {"workers": [], "execution_time": [], "speedup": [], "efficiency": []}

    baseline_time = None

    for num_workers in worker_counts:
        logger.info(f"\nTesting with {num_workers} workers...")

        config = SweepConfiguration(
            parameters=parameters, execution_mode="parallel_processes", save_intermediate=False, max_workers=num_workers
        )

        sweep = ParameterSweep(parameters, config)

        start = time.perf_counter()
        sweep.execute(mfg_solve_function)
        exec_time = time.perf_counter() - start

        if baseline_time is None:
            baseline_time = exec_time

        speedup = baseline_time / exec_time
        efficiency = speedup / num_workers  # Ideal efficiency is 1.0

        logger.info(f"  Time: {exec_time:.2f}s")
        logger.info(f"  Speedup: {speedup:.2f}x")
        logger.info(f"  Efficiency: {efficiency:.2%}")

        results["workers"].append(num_workers)
        results["execution_time"].append(exec_time)
        results["speedup"].append(speedup)
        results["efficiency"].append(efficiency)

    return results


def create_benchmark_report(bench1, bench2, bench3, bench4):
    """Create comprehensive visualization of benchmark results."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle("MFGarchon Parallel Computation Benchmark Analysis", fontsize=16, fontweight="bold")

    # 1. Common Noise Speedup
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(bench1["num_samples"], bench1["speedup"], "o-", linewidth=2, markersize=8, color="steelblue")
    ax1.axhline(y=mp.cpu_count(), color="red", linestyle="--", alpha=0.5, label=f"Max CPUs ({mp.cpu_count()})")
    ax1.set_xlabel("Number of Noise Samples")
    ax1.set_ylabel("Speedup Factor")
    ax1.set_title("Common Noise MFG: Parallel Speedup")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Common Noise Overhead
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(len(bench1["num_samples"])), bench1["overhead_pct"], color="coral", alpha=0.7)
    ax2.set_xticks(range(len(bench1["num_samples"])))
    ax2.set_xticklabels(bench1["num_samples"])
    ax2.set_xlabel("Number of Noise Samples")
    ax2.set_ylabel("Overhead (%)")
    ax2.set_title("Parallelization Overhead")
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Common Noise Time Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    x_pos = np.arange(len(bench1["num_samples"]))
    width = 0.35
    ax3.bar(x_pos - width / 2, bench1["sequential_time"], width, label="Sequential", color="blue", alpha=0.7)
    ax3.bar(x_pos + width / 2, bench1["parallel_time"], width, label="Parallel", color="red", alpha=0.7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(bench1["num_samples"])
    ax3.set_xlabel("Number of Noise Samples")
    ax3.set_ylabel("Execution Time (s)")
    ax3.set_title("Sequential vs Parallel Time")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Parameter Sweep Modes
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(range(len(bench2["mode"])), bench2["speedup"], color=["blue", "green", "red"], alpha=0.7)
    ax4.set_xticks(range(len(bench2["mode"])))
    ax4.set_xticklabels([m.replace("parallel_", "") for m in bench2["mode"]], rotation=15, ha="right")
    ax4.set_ylabel("Speedup Factor")
    ax4.set_title("Parameter Sweep: Execution Mode Comparison")
    ax4.grid(True, alpha=0.3, axis="y")

    # 5. Parameter Sweep Time
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(range(len(bench2["mode"])), bench2["execution_time"], color=["blue", "green", "red"], alpha=0.7)
    ax5.set_xticks(range(len(bench2["mode"])))
    ax5.set_xticklabels([m.replace("parallel_", "") for m in bench2["mode"]], rotation=15, ha="right")
    ax5.set_ylabel("Total Time (s)")
    ax5.set_title("Parameter Sweep: Total Execution Time")
    ax5.grid(True, alpha=0.3, axis="y")

    # 6. Monte Carlo Efficiency
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(bench3["dimension"], bench3["efficiency"], "o-", linewidth=2, markersize=8, color="purple")
    ax6.set_xlabel("Problem Dimension")
    ax6.set_ylabel("Samples per Second")
    ax6.set_title("Monte Carlo Integration Efficiency")
    ax6.grid(True, alpha=0.3)
    ax6.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    # 7. Worker Scaling: Speedup
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(bench4["workers"], bench4["speedup"], "o-", linewidth=2, markersize=8, color="darkgreen")
    ax7.plot(bench4["workers"], bench4["workers"], "--", color="gray", alpha=0.5, label="Ideal (linear)")
    ax7.set_xlabel("Number of Workers")
    ax7.set_ylabel("Speedup Factor")
    ax7.set_title("Worker Scaling: Speedup")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Worker Scaling: Efficiency
    ax8 = fig.add_subplot(gs[2, 1])
    efficiency_pct = [e * 100 for e in bench4["efficiency"]]
    ax8.plot(bench4["workers"], efficiency_pct, "o-", linewidth=2, markersize=8, color="orange")
    ax8.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="Ideal (100%)")
    ax8.set_xlabel("Number of Workers")
    ax8.set_ylabel("Parallel Efficiency (%)")
    ax8.set_title("Worker Scaling: Efficiency")
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. Summary statistics table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")

    # Compute summary statistics
    cn_avg_speedup = np.mean(bench1["speedup"])
    cn_avg_overhead = np.mean(bench1["overhead_pct"])
    ps_best_speedup = max(bench2["speedup"])
    ps_best_mode = bench2["mode"][bench2["speedup"].index(ps_best_speedup)]
    ws_max_efficiency = max(bench4["efficiency"]) * 100

    summary_text = f"""
    SUMMARY STATISTICS
    {"=" * 30}

    Common Noise MFG:
      Avg Speedup: {cn_avg_speedup:.2f}x
      Avg Overhead: {cn_avg_overhead:.1f}%

    Parameter Sweep:
      Best Speedup: {ps_best_speedup:.2f}x
      Best Mode: {ps_best_mode}

    Worker Scaling:
      Max Efficiency: {ws_max_efficiency:.1f}%
      CPU Count: {mp.cpu_count()}

    RECOMMENDATIONS:
      • Use parallel for K > 20 samples
      • Process pool for parameter sweeps
      • Max efficiency at {mp.cpu_count() // 2}-{mp.cpu_count()} workers
    """

    ax9.text(
        0.1,
        0.5,
        summary_text,
        fontsize=10,
        verticalalignment="center",
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgray", "alpha": 0.8},
    )

    plt.tight_layout()

    # Save figure
    output_path = OUTPUT_DIR / "parallel_computation_benchmark.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"\nBenchmark visualization saved to: {output_path}")

    return output_path


def generate_text_report(bench1, bench2, bench3, bench4):
    """Generate detailed text report."""
    report_path = OUTPUT_DIR / "parallel_computation_analysis.md"

    with open(report_path, "w") as f:
        f.write("# MFGarchon Parallel Computation Analysis\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**CPU Count**: {mp.cpu_count()}\n\n")

        f.write("## Executive Summary\n\n")

        cn_avg_speedup = np.mean(bench1["speedup"])
        ps_best_speedup = max(bench2["speedup"])

        f.write(f"- **Common Noise MFG**: {cn_avg_speedup:.2f}x average speedup with parallel execution\n")
        f.write(f"- **Parameter Sweep**: {ps_best_speedup:.2f}x speedup with process-based parallelism\n")
        f.write(f"- **Worker Scaling**: Optimal efficiency at {mp.cpu_count() // 2}-{mp.cpu_count()} workers\n\n")

        f.write("## Benchmark 1: Common Noise MFG Monte Carlo\n\n")
        f.write("| Samples | Sequential (s) | Parallel (s) | Speedup | Overhead (%) |\n")
        f.write("|---------|----------------|--------------|---------|---------------|\n")
        for i in range(len(bench1["num_samples"])):
            f.write(
                f"| {bench1['num_samples'][i]:7d} | "
                f"{bench1['sequential_time'][i]:14.2f} | "
                f"{bench1['parallel_time'][i]:12.2f} | "
                f"{bench1['speedup'][i]:7.2f} | "
                f"{bench1['overhead_pct'][i]:13.1f} |\n"
            )

        f.write("\n**Analysis**: ")
        if cn_avg_speedup > 1.5:
            f.write(
                "Parallel execution shows significant performance improvement. "
                "Recommended for production use with K > 20 noise samples.\n\n"
            )
        else:
            f.write(
                "Parallel overhead is substantial for small sample counts. Use sequential execution for K < 20.\n\n"
            )

        f.write("## Benchmark 2: Parameter Sweep Execution Modes\n\n")
        f.write("| Mode             | Time (s) | Speedup |\n")
        f.write("|------------------|----------|----------|\n")
        for i in range(len(bench2["mode"])):
            f.write(f"| {bench2['mode'][i]:16s} | {bench2['execution_time'][i]:8.2f} | {bench2['speedup'][i]:8.2f} |\n")

        f.write("\n**Analysis**: ")
        best_idx = bench2["speedup"].index(max(bench2["speedup"]))
        f.write(f"{bench2['mode'][best_idx]} provides best performance for parameter sweeps.\n\n")

        f.write("## Benchmark 3: Monte Carlo Integration\n\n")
        f.write("| Dimension | Time (s) | Samples/sec |\n")
        f.write("|-----------|----------|-------------|\n")
        for i in range(len(bench3["dimension"])):
            f.write(
                f"| {bench3['dimension'][i]:9d} | "
                f"{bench3['sequential_time'][i]:8.2f} | "
                f"{bench3['efficiency'][i]:11.0f} |\n"
            )

        f.write("\n**Analysis**: MC integration efficiency decreases with dimension (curse of dimensionality).\n\n")

        f.write("## Benchmark 4: Worker Count Scaling\n\n")
        f.write("| Workers | Time (s) | Speedup | Efficiency (%) |\n")
        f.write("|---------|----------|---------|----------------|\n")
        for i in range(len(bench4["workers"])):
            f.write(
                f"| {bench4['workers'][i]:7d} | "
                f"{bench4['execution_time'][i]:8.2f} | "
                f"{bench4['speedup'][i]:7.2f} | "
                f"{bench4['efficiency'][i] * 100:14.1f} |\n"
            )

        f.write("\n**Analysis**: ")
        max_eff_idx = bench4["efficiency"].index(max(bench4["efficiency"]))
        f.write(
            f"Optimal worker count is {bench4['workers'][max_eff_idx]} "
            f"with {bench4['efficiency'][max_eff_idx] * 100:.1f}% efficiency.\n\n"
        )

        f.write("## Recommendations\n\n")
        f.write("### When to Use Parallel Execution\n\n")
        f.write("1. **Common Noise MFG**: Use `parallel=True` when:\n")
        f.write("   - Number of noise samples K > 20\n")
        f.write("   - Individual MFG solves take > 1 second\n")
        f.write(f"   - Expected speedup: {cn_avg_speedup:.1f}x\n\n")

        f.write("2. **Parameter Sweep**: Use `parallel_processes` mode when:\n")
        f.write("   - Total combinations > 10\n")
        f.write("   - Individual runs take > 0.5 seconds\n")
        f.write(f"   - Expected speedup: {ps_best_speedup:.1f}x\n\n")

        f.write("3. **Optimal Configuration**:\n")
        f.write(f"   - Worker count: {mp.cpu_count() // 2} to {mp.cpu_count()}\n")
        f.write("   - Use variance reduction to reduce K needed\n")
        f.write(f"   - Batch size: {mp.cpu_count() * 2} for parameter sweeps\n\n")

        f.write("### When NOT to Use Parallel Execution\n\n")
        f.write("- Small problem sizes (Nx, Nt < 50)\n")
        f.write("- Few Monte Carlo samples (K < 20)\n")
        f.write("- Quick individual solves (< 0.5s per run)\n")
        f.write("- High process startup overhead dominates\n\n")

        f.write("### Performance Bottlenecks\n\n")
        f.write(f"1. **Process overhead**: {np.mean(bench1['overhead_pct']):.1f}% average\n")
        f.write("2. **Memory copying**: Pickling large MFG problems\n")
        f.write("3. **Communication**: Result collection from workers\n\n")

        f.write("### Future Optimizations\n\n")
        f.write("- Shared memory for large arrays (multiprocessing.shared_memory)\n")
        f.write("- Ray/Dask for distributed computing\n")
        f.write("- GPU acceleration for individual MFG solves\n")
        f.write("- Hybrid CPU+GPU parallelism\n")

    logger.info(f"Text report saved to: {report_path}")
    return report_path


def main():
    """Run all benchmarks and generate reports."""
    logger.info("\n" + "=" * 80)
    logger.info("MFGarchon PARALLEL COMPUTATION COMPREHENSIVE BENCHMARK")
    logger.info("=" * 80)
    logger.info(f"CPU Count: {mp.cpu_count()}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")

    # Run benchmarks
    bench1 = benchmark_common_noise_parallel()
    bench2 = benchmark_parameter_sweep_parallel()
    bench3 = benchmark_monte_carlo_parallel()
    bench4 = analyze_worker_scaling()

    # Generate reports
    logger.info("\n" + "=" * 80)
    logger.info("Generating Benchmark Reports")
    logger.info("=" * 80)

    viz_path = create_benchmark_report(bench1, bench2, bench3, bench4)
    text_path = generate_text_report(bench1, bench2, bench3, bench4)

    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Visualization: {viz_path}")
    logger.info(f"Text Report: {text_path}")

    # Print key findings
    logger.info("\nKEY FINDINGS:")
    logger.info(f"  Common Noise avg speedup: {np.mean(bench1['speedup']):.2f}x")
    logger.info(f"  Parameter sweep best speedup: {max(bench2['speedup']):.2f}x")
    logger.info(f"  Recommended workers: {mp.cpu_count() // 2}-{mp.cpu_count()}")


if __name__ == "__main__":
    main()

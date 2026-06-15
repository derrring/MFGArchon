#!/usr/bin/env python3
"""
WENO Family Comparison Demo

This example demonstrates the new unified WENO family solver, comparing different
WENO variants (WENO5, WENO-Z, WENO-M, WENO-JS) on a challenging MFG problem with
discontinuities and steep gradients.

The demo showcases:
1. Easy switching between WENO variants
2. Performance comparison across different methods
3. Resolution quality analysis
4. Academic-quality benchmarking setup

Usage:
    python examples/advanced/weno_family_comparison_demo.py

Requirements:
    - numpy, matplotlib
    - mfgarchon with strategic typing excellence
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.hjb_solvers import HJBWenoSolver
from mfgarchon.utils.mfg_logging import configure_research_logging, get_logger

# Configure logging for research session
configure_research_logging("weno_family_comparison", level="INFO")
logger = get_logger(__name__)


def create_challenging_mfg_problem() -> MFGProblem:
    """
    Create MFG problem with features that challenge WENO schemes:
    - Sharp initial conditions
    - Low diffusion (sharp gradients)
    - High congestion (nonlinear effects)
    """
    problem = MFGProblem(
        xmin=0.0,
        xmax=1.0,
        Nx=128,  # Fine grid for high resolution
        T=1.0,
        Nt=200,
        sigma=0.05,  # Low diffusion -> sharp features
        coupling_coefficient=2.0,  # High congestion -> nonlinear effects
    )

    # Create sharp initial condition with discontinuous derivative
    x = problem.geometry.get_spatial_grid()

    # Sharp Gaussian with discontinuous derivative at boundaries
    initial_u = np.exp(-50 * (x - 0.3) ** 2) + 0.5 * np.exp(-100 * (x - 0.7) ** 2)

    # Add some noise to make it more challenging
    initial_u += 0.1 * np.sin(20 * np.pi * x)

    # Override initial condition
    problem._custom_initial_u = initial_u

    logger.info("Created challenging MFG problem with sharp features")
    Nx_intervals = problem.geometry.get_grid_shape()[0] - 1
    logger.info(f"Grid: {Nx_intervals} intervals, Time steps: {problem.Nt}")
    logger.info(f"Diffusion σ = {problem.diffusion}, Congestion = {problem.coupling_coefficient}")

    return problem


def run_weno_variant_comparison() -> dict[str, dict]:
    """
    Run comprehensive comparison of all WENO variants.

    Returns:
        Results dictionary with performance metrics for each variant
    """
    problem = create_challenging_mfg_problem()

    # WENO variants to compare
    weno_variants = ["weno5", "weno-z", "weno-m", "weno-js"]

    results = {}
    solutions = {}

    logger.info("Starting WENO family comparison...")

    for variant in weno_variants:
        logger.info(f"Running WENO variant: {variant.upper()}")

        # Create solver for this variant
        solver = HJBWenoSolver(
            problem=problem, weno_variant=variant, cfl_number=0.3, weno_epsilon=1e-6, time_integration="tvd_rk3"
        )

        # Get variant information
        variant_info = solver.get_variant_info()
        logger.info(f"  Description: {variant_info['description']}")
        logger.info(f"  Best for: {variant_info['best_for']}")

        # Solve with timing
        start_time = time.time()

        try:
            # Simple solve for comparison
            # Note: Using a basic iteration for comparison
            nx = problem.geometry.num_spatial_points - 1  # number of intervals
            u_current = getattr(problem, "_custom_initial_u", np.ones(nx))
            m_current = np.ones(nx) / nx  # Normalized initial density

            dt = problem.T / problem.Nt

            # Run several time steps for comparison
            n_steps = min(50, problem.Nt)  # Limit for demo purposes

            for step in range(n_steps):
                u_current = solver.solve_hjb_step(u_current, m_current, dt)

                # Simple density update (not full MFG coupling for this demo)
                if step % 10 == 0:
                    logger.info(f"  Step {step}/{n_steps}, max |u| = {np.max(np.abs(u_current)):.4f}")

            execution_time = time.time() - start_time

            # Store results
            results[variant] = {
                "execution_time": execution_time,
                "final_solution": u_current.copy(),
                "max_value": np.max(np.abs(u_current)),
                "total_variation": np.sum(np.abs(np.diff(u_current))),
                "variant_info": variant_info,
                "success": True,
            }

            solutions[variant] = u_current

            logger.info(f"  ✅ {variant.upper()} completed in {execution_time:.3f}s")
            logger.info(f"  Max |u| = {results[variant]['max_value']:.4f}")
            logger.info(f"  Total variation = {results[variant]['total_variation']:.4f}")

        except Exception as e:
            logger.error(f"  ❌ {variant.upper()} failed: {e}")
            results[variant] = {"success": False, "error": str(e), "variant_info": variant_info}

    return results, solutions


def analyze_and_plot_results(results: dict[str, dict], solutions: dict[str, np.ndarray]) -> None:
    """
    Create comprehensive analysis and visualization of WENO comparison results.
    """
    logger.info("Analyzing WENO family comparison results...")

    # Filter successful results
    successful_variants = {k: v for k, v in results.items() if v.get("success", False)}

    if not successful_variants:
        logger.error("No successful WENO variants to analyze!")
        return

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("WENO Family Comparison: Advanced MFG Solver Benchmarking", fontsize=16, fontweight="bold")

    # Plot 1: Solution profiles
    ax = axes[0, 0]
    x = np.linspace(0, 1, len(next(iter(solutions.values()))))

    colors = ["blue", "red", "green", "orange"]
    line_styles = ["-", "--", "-.", ":"]

    for i, (variant, solution) in enumerate(solutions.items()):
        ax.plot(
            x,
            solution,
            color=colors[i % len(colors)],
            linestyle=line_styles[i % len(line_styles)],
            linewidth=2,
            label=f"{variant.upper()}",
            alpha=0.8,
        )

    ax.set_xlabel("Spatial coordinate x")
    ax.set_ylabel("Value function u(x)")
    ax.set_title("Final Solutions Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Performance metrics
    ax = axes[0, 1]
    variants = list(successful_variants.keys())
    execution_times = [successful_variants[v]["execution_time"] for v in variants]

    bars = ax.bar(
        range(len(variants)), execution_times, color=["blue", "red", "green", "orange"][: len(variants)], alpha=0.7
    )
    ax.set_xlabel("WENO Variant")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("Computational Performance")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([v.upper() for v in variants])
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, time_val in zip(bars, execution_times, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, height + height * 0.01, f"{time_val:.3f}s", ha="center", va="bottom"
        )

    # Plot 3: Total variation (measure of oscillations)
    ax = axes[1, 0]
    total_variations = [successful_variants[v]["total_variation"] for v in variants]

    bars = ax.bar(
        range(len(variants)), total_variations, color=["blue", "red", "green", "orange"][: len(variants)], alpha=0.7
    )
    ax.set_xlabel("WENO Variant")
    ax.set_ylabel("Total Variation")
    ax.set_title("Solution Quality: Total Variation\n(Lower = Less Oscillatory)")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([v.upper() for v in variants])
    ax.grid(True, alpha=0.3)

    # Add value labels
    for bar, tv_val in zip(bars, total_variations, strict=False):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + height * 0.01, f"{tv_val:.2f}", ha="center", va="bottom")

    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis("off")

    # Create summary table
    table_data = []
    headers = ["Variant", "Time (s)", "Max |u|", "Total Var.", "Best For"]

    for variant in variants:
        data = successful_variants[variant]
        table_data.append(
            [
                variant.upper(),
                f"{data['execution_time']:.3f}",
                f"{data['max_value']:.3f}",
                f"{data['total_variation']:.2f}",
                data["variant_info"]["best_for"][:25] + "..."
                if len(data["variant_info"]["best_for"]) > 25
                else data["variant_info"]["best_for"],
            ]
        )

    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc="center", loc="center", cellColours=None)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.0)
    ax.set_title("WENO Variants Performance Summary", pad=20)

    plt.tight_layout()
    plt.savefig("weno_family_comparison.png", dpi=300, bbox_inches="tight")
    logger.info("Saved comparison plot: weno_family_comparison.png")

    # Print detailed analysis
    logger.info("\n" + "=" * 60)
    logger.info("WENO FAMILY COMPARISON ANALYSIS")
    logger.info("=" * 60)

    for variant, data in successful_variants.items():
        logger.info(f"\n{variant.upper()} Results:")
        logger.info(f"  Execution time: {data['execution_time']:.3f} seconds")
        logger.info("  Solution quality:")
        logger.info(f"    Max |u|: {data['max_value']:.4f}")
        logger.info(f"    Total variation: {data['total_variation']:.4f}")
        logger.info(f"  Description: {data['variant_info']['description']}")
        logger.info(f"  Best for: {data['variant_info']['best_for']}")

    # Performance ranking
    logger.info(f"\n{'Performance Ranking:'}")
    sorted_by_speed = sorted(successful_variants.items(), key=lambda x: x[1]["execution_time"])
    logger.info("  Speed (fastest to slowest):")
    for i, (variant, data) in enumerate(sorted_by_speed):
        logger.info(f"    {i + 1}. {variant.upper()} ({data['execution_time']:.3f}s)")

    sorted_by_quality = sorted(successful_variants.items(), key=lambda x: x[1]["total_variation"])
    logger.info("  Quality - least oscillatory (best to worst):")
    for i, (variant, data) in enumerate(sorted_by_quality):
        logger.info(f"    {i + 1}. {variant.upper()} (TV = {data['total_variation']:.3f})")


def main():
    """
    Main function to run WENO family comparison demonstration.
    """
    logger.info("🚀 WENO Family Comparison Demo Starting")
    logger.info("Testing advanced MFG solver variants with strategic typing excellence")

    try:
        # Run comprehensive comparison
        results, solutions = run_weno_variant_comparison()

        # Analyze and visualize results
        analyze_and_plot_results(results, solutions)

        logger.info("🎊 WENO Family Comparison Demo Completed Successfully!")
        logger.info("This demonstrates the power of the unified WENO family solver")
        logger.info("Users can easily switch between variants for optimal performance")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()

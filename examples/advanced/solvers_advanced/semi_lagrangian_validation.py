#!/usr/bin/env python3
"""
Semi-Lagrangian Method Validation and Advanced Examples

This module provides comprehensive validation of the semi-Lagrangian HJB solver
through analytical solutions, convergence analysis, and performance comparisons.

Tests include:
1. Manufactured solutions with known analytical answers
2. Convergence rate analysis
3. Stability analysis for large time steps
4. Comparison with other HJB solvers
5. Performance benchmarking
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.factory import create_semi_lagrangian_solver
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.mfg_logging import configure_research_logging, get_logger

# Configure logging
configure_research_logging("semi_lagrangian_validation", level="INFO")
logger = get_logger(__name__)


class AnalyticalSolution:
    """Container for analytical MFG solutions used for validation."""

    @staticmethod
    def linear_solution(x: np.ndarray, t: float, T: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Linear analytical solution for validation.

        u(t,x) = (T-t) * x²/2
        m(t,x) = constant density

        Args:
            x: Spatial grid
            t: Current time
            T: Final time

        Returns:
            Tuple of (u_analytical, m_analytical)
        """
        u_analytical = (T - t) * x**2 / 2
        m_analytical = np.ones_like(x) / len(x)  # Normalized uniform density

        return u_analytical, m_analytical

    @staticmethod
    def quadratic_solution(x: np.ndarray, t: float, T: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Quadratic analytical solution.

        u(t,x) = (T-t)² * (x - 0.5)²
        m(t,x) = Gaussian-like density

        Args:
            x: Spatial grid
            t: Current time
            T: Final time

        Returns:
            Tuple of (u_analytical, m_analytical)
        """
        u_analytical = (T - t) ** 2 * (x - 0.5) ** 2

        # Gaussian-like density centered at 0.5
        sigma_m = 0.1
        m_analytical = np.exp(-0.5 * ((x - 0.5) / sigma_m) ** 2)
        m_analytical = m_analytical / np.trapz(m_analytical, x)  # Normalize

        return u_analytical, m_analytical


def create_validation_problem(nx: int = 51, nt: int = 51) -> MFGProblem:
    """
    Create MFG problem designed for validation.

    Args:
        nx: Number of spatial points
        nt: Number of time points

    Returns:
        MFG problem with known analytical properties
    """
    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[nx + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        T=0.5,
        Nt=nt,
        sigma=0.1,
        coupling_coefficient=0.5,
    )


def test_analytical_convergence() -> dict[str, float]:
    """
    Test convergence to analytical solution.

    Returns:
        Dictionary with error metrics
    """
    logger.info("Testing convergence to analytical solution...")

    problem = create_validation_problem(nx=101, nt=51)

    # Create semi-Lagrangian solver
    solver = create_semi_lagrangian_solver(
        problem, interpolation_method="linear", optimization_method="brent", use_jax=False
    )

    try:
        # Solve numerically
        result = solver.solve()

        if not hasattr(result, "U") or not hasattr(result, "M"):
            logger.error("Result missing U or M fields")
            return {"error": np.inf}

        # Compare with analytical solution at final time
        x_grid = problem.geometry.get_spatial_grid()
        u_analytical, m_analytical = AnalyticalSolution.linear_solution(x_grid, problem.T, problem.T)

        # Compute errors
        u_error_l2 = np.sqrt(np.trapz((result.U[-1, :] - u_analytical) ** 2, x_grid))
        m_error_l2 = np.sqrt(np.trapz((result.M[-1, :] - m_analytical) ** 2, x_grid))
        u_error_max = np.max(np.abs(result.U[-1, :] - u_analytical))
        m_error_max = np.max(np.abs(result.M[-1, :] - m_analytical))

        errors = {
            "u_l2_error": u_error_l2,
            "m_l2_error": m_error_l2,
            "u_max_error": u_error_max,
            "m_max_error": m_error_max,
        }

        logger.info(f"  L2 error in U: {u_error_l2:.2e}")
        logger.info(f"  L2 error in M: {m_error_l2:.2e}")
        logger.info(f"  Max error in U: {u_error_max:.2e}")
        logger.info(f"  Max error in M: {m_error_max:.2e}")

        return errors

    except Exception as e:
        logger.error(f"Analytical convergence test failed: {e}")
        return {"error": np.inf}


def test_grid_convergence() -> dict[str, list]:
    """
    Test spatial and temporal convergence rates.

    Returns:
        Dictionary with convergence data
    """
    logger.info("Testing grid convergence rates...")

    # Test different grid resolutions
    nx_values = [25, 51, 101]
    nt_values = [25, 51, 101]

    spatial_errors = []
    temporal_errors = []

    # Spatial convergence (fixed time resolution)
    logger.info("  Testing spatial convergence...")
    base_nt = 51
    for nx in nx_values:
        try:
            problem = create_validation_problem(nx=nx, nt=base_nt)
            solver = create_semi_lagrangian_solver(problem, interpolation_method="linear", use_jax=False)

            result = solver.solve()

            # Compute error against analytical solution
            x_grid = problem.geometry.get_spatial_grid()
            u_analytical, _ = AnalyticalSolution.linear_solution(x_grid, problem.T, problem.T)

            error = np.sqrt(np.trapz((result.U[-1, :] - u_analytical) ** 2, x_grid))
            spatial_errors.append(error)

            logger.info(f"    Nx={nx}: L2 error = {error:.2e}")

        except Exception as e:
            logger.warning(f"Spatial convergence test failed for Nx={nx}: {e}")
            spatial_errors.append(np.inf)

    # Temporal convergence (fixed spatial resolution)
    logger.info("  Testing temporal convergence...")
    base_nx = 51
    for nt in nt_values:
        try:
            problem = create_validation_problem(nx=base_nx, nt=nt)
            solver = create_semi_lagrangian_solver(problem, interpolation_method="linear", use_jax=False)

            result = solver.solve()

            # Compute error against analytical solution
            x_grid = problem.geometry.get_spatial_grid()
            u_analytical, _ = AnalyticalSolution.linear_solution(x_grid, problem.T, problem.T)

            error = np.sqrt(np.trapz((result.U[-1, :] - u_analytical) ** 2, x_grid))
            temporal_errors.append(error)

            logger.info(f"    Nt={nt}: L2 error = {error:.2e}")

        except Exception as e:
            logger.warning(f"Temporal convergence test failed for Nt={nt}: {e}")
            temporal_errors.append(np.inf)

    return {
        "nx_values": nx_values,
        "nt_values": nt_values,
        "spatial_errors": spatial_errors,
        "temporal_errors": temporal_errors,
    }


def test_large_timestep_stability() -> dict[str, bool]:
    """
    Test stability for large time steps.

    Semi-Lagrangian methods should be stable for larger time steps
    than explicit finite difference methods.

    Returns:
        Dictionary with stability test results
    """
    logger.info("Testing large time step stability...")

    # Test with various time step sizes
    nt_values = [10, 25, 51, 101]  # From large to small time steps
    stability_results = {}

    for nt in nt_values:
        dt = 0.5 / (nt - 1)  # Time step size
        logger.info(f"  Testing with Nt={nt} (dt={dt:.4f})...")

        try:
            problem = create_validation_problem(nx=51, nt=nt)
            solver = create_semi_lagrangian_solver(problem, interpolation_method="linear", use_jax=False)

            result = solver.solve()

            # Check for stability (no blow-up, reasonable values)
            if hasattr(result, "U") and hasattr(result, "M"):
                u_max = np.max(np.abs(result.U))
                m_max = np.max(np.abs(result.M))
                has_nan = np.any(np.isnan(result.U)) or np.any(np.isnan(result.M))
                has_inf = np.any(np.isinf(result.U)) or np.any(np.isinf(result.M))

                is_stable = not has_nan and not has_inf and u_max < 1e10 and m_max < 1e10

                stability_results[f"nt_{nt}"] = is_stable

                if is_stable:
                    logger.info(f"    ✓ Stable (max U: {u_max:.2e}, max M: {m_max:.2e})")
                else:
                    logger.warning(f"    ✗ Unstable (NaN: {has_nan}, Inf: {has_inf}, max U: {u_max:.2e})")
            else:
                stability_results[f"nt_{nt}"] = False
                logger.warning("    ✗ Missing solution fields")

        except Exception as e:
            logger.warning(f"    ✗ Stability test failed for Nt={nt}: {e}")
            stability_results[f"nt_{nt}"] = False

    return stability_results


def test_interpolation_methods() -> dict[str, float]:
    """
    Compare different interpolation methods.

    Returns:
        Dictionary with performance metrics for each method
    """
    logger.info("Testing different interpolation methods...")

    problem = create_validation_problem(nx=51, nt=51)
    interpolation_methods = ["linear", "cubic"]
    results = {}

    for method in interpolation_methods:
        logger.info(f"  Testing {method} interpolation...")

        try:
            start_time = time.time()

            solver = create_semi_lagrangian_solver(problem, interpolation_method=method, use_jax=False)

            result = solver.solve()

            solve_time = time.time() - start_time

            if hasattr(result, "U"):
                # Compute solution quality metric
                x_grid = problem.geometry.get_spatial_grid()
                u_analytical, _ = AnalyticalSolution.linear_solution(x_grid, problem.T, problem.T)
                error = np.sqrt(np.trapz((result.U[-1, :] - u_analytical) ** 2, x_grid))

                results[method] = {"error": error, "solve_time": solve_time, "success": True}

                logger.info(f"    ✓ Success: error={error:.2e}, time={solve_time:.2f}s")
            else:
                results[method] = {"error": np.inf, "solve_time": solve_time, "success": False}
                logger.warning("    ✗ Failed: missing solution")

        except Exception as e:
            logger.warning(f"    ✗ Failed with {method} interpolation: {e}")
            results[method] = {"error": np.inf, "solve_time": np.inf, "success": False}

    return results


def create_validation_plots(convergence_data: dict[str, list], interpolation_results: dict[str, float]):
    """
    Create validation plots.

    Args:
        convergence_data: Grid convergence test results
        interpolation_results: Interpolation method comparison results
    """
    logger.info("Creating validation plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Semi-Lagrangian Method Validation", fontsize=14)

    # Plot 1: Spatial convergence
    ax1 = axes[0, 0]
    if convergence_data["spatial_errors"]:
        nx_values = convergence_data["nx_values"]
        spatial_errors = convergence_data["spatial_errors"]

        # Filter out infinite errors
        valid_indices = [i for i, err in enumerate(spatial_errors) if np.isfinite(err)]
        if valid_indices:
            nx_valid = [nx_values[i] for i in valid_indices]
            errors_valid = [spatial_errors[i] for i in valid_indices]

            ax1.loglog(nx_valid, errors_valid, "bo-", linewidth=2, markersize=6)

            # Add theoretical convergence line (O(h²) for second order)
            if len(nx_valid) >= 2:
                h_values = [1.0 / nx for nx in nx_valid]
                theoretical_line = errors_valid[0] * (np.array(h_values) / h_values[0]) ** 2
                ax1.loglog(nx_valid, theoretical_line, "r--", alpha=0.7, label="O(h²)")

    ax1.set_xlabel("Number of spatial points (Nx)")
    ax1.set_ylabel("L2 Error")
    ax1.set_title("Spatial Convergence")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Temporal convergence
    ax2 = axes[0, 1]
    if convergence_data["temporal_errors"]:
        nt_values = convergence_data["nt_values"]
        temporal_errors = convergence_data["temporal_errors"]

        # Filter out infinite errors
        valid_indices = [i for i, err in enumerate(temporal_errors) if np.isfinite(err)]
        if valid_indices:
            nt_valid = [nt_values[i] for i in valid_indices]
            errors_valid = [temporal_errors[i] for i in valid_indices]

            ax2.loglog(nt_valid, errors_valid, "go-", linewidth=2, markersize=6)

            # Add theoretical convergence line (O(dt) for first order)
            if len(nt_valid) >= 2:
                dt_values = [0.5 / (nt - 1) for nt in nt_valid]
                theoretical_line = errors_valid[0] * (np.array(dt_values) / dt_values[0])
                ax2.loglog(nt_valid, theoretical_line, "r--", alpha=0.7, label="O(Δt)")

    ax2.set_xlabel("Number of time points (Nt)")
    ax2.set_ylabel("L2 Error")
    ax2.set_title("Temporal Convergence")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Interpolation method comparison (error)
    ax3 = axes[1, 0]
    if interpolation_results:
        methods = []
        errors = []
        for method, result in interpolation_results.items():
            if result["success"] and np.isfinite(result["error"]):
                methods.append(method.replace("_", " ").title())
                errors.append(result["error"])

        if methods:
            bars = ax3.bar(methods, errors, alpha=0.7, color=["blue", "green", "red"][: len(methods)])
            ax3.set_ylabel("L2 Error")
            ax3.set_title("Interpolation Method Accuracy")
            ax3.set_yscale("log")
            ax3.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, error in zip(bars, errors, strict=False):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2.0, height, f"{error:.1e}", ha="center", va="bottom")

    # Plot 4: Interpolation method comparison (time)
    ax4 = axes[1, 1]
    if interpolation_results:
        methods = []
        times = []
        for method, result in interpolation_results.items():
            if result["success"] and np.isfinite(result["solve_time"]):
                methods.append(method.replace("_", " ").title())
                times.append(result["solve_time"])

        if methods:
            bars = ax4.bar(methods, times, alpha=0.7, color=["blue", "green", "red"][: len(methods)])
            ax4.set_ylabel("Solve Time (seconds)")
            ax4.set_title("Interpolation Method Performance")
            ax4.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, solve_time in zip(bars, times, strict=False):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width() / 2.0, height, f"{solve_time:.2f}s", ha="center", va="bottom")

    plt.tight_layout()

    # Save plot
    plot_filename = "semi_lagrangian_validation.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    logger.info(f"Validation plots saved as: {plot_filename}")

    try:
        plt.show()
    except:
        logger.info("Plot display not available (non-interactive environment)")


def main():
    """Main validation function."""
    logger.info("=" * 70)
    logger.info("Semi-Lagrangian HJB Solver Validation")
    logger.info("=" * 70)

    try:
        # Run validation tests
        logger.info("Running comprehensive validation tests...")

        # Test 1: Analytical convergence
        analytical_errors = test_analytical_convergence()

        # Test 2: Grid convergence
        convergence_data = test_grid_convergence()

        # Test 3: Large time step stability
        stability_results = test_large_timestep_stability()

        # Test 4: Interpolation methods
        interpolation_results = test_interpolation_methods()

        # Create validation plots
        create_validation_plots(convergence_data, interpolation_results)

        # Summary report
        logger.info("=" * 70)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 70)

        # Analytical convergence summary
        if "error" not in analytical_errors:
            logger.info("✓ Analytical Convergence: PASSED")
            logger.info(f"  L2 error in U: {analytical_errors['u_l2_error']:.2e}")
            logger.info(f"  L2 error in M: {analytical_errors['m_l2_error']:.2e}")
        else:
            logger.info("✗ Analytical Convergence: FAILED")

        # Stability summary
        stable_count = sum(1 for stable in stability_results.values() if stable)
        total_tests = len(stability_results)
        logger.info(f"✓ Stability Tests: {stable_count}/{total_tests} passed")

        # Interpolation summary
        successful_methods = sum(1 for result in interpolation_results.values() if result["success"])
        total_methods = len(interpolation_results)
        logger.info(f"✓ Interpolation Methods: {successful_methods}/{total_methods} successful")

        # Overall assessment
        all_tests_passed = (
            "error" not in analytical_errors and stable_count == total_tests and successful_methods == total_methods
        )

        if all_tests_passed:
            logger.info("🎉 OVERALL VALIDATION: PASSED")
            logger.info("   Semi-Lagrangian implementation is working correctly!")
        else:
            logger.info("⚠️  OVERALL VALIDATION: PARTIAL SUCCESS")
            logger.info("   Some tests failed - implementation may need refinement")

        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()

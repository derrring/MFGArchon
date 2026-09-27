"""
Comprehensive Benchmarking Suite for High-Dimensional MFG Problems

This module provides systematic performance evaluation tools for comparing
different solvers, geometries, and optimization strategies across problem dimensions.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil

import matplotlib.pyplot as plt
import numpy as np

from mfgarchon.utils.mfg_logging.logger import get_logger
from mfgarchon.utils.performance.optimization import PerformanceMonitor

if TYPE_CHECKING:
    from mfgarchon import MFGProblem

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Container for individual benchmark results."""

    test_name: str
    dimension: int
    grid_size: tuple[int, ...]
    total_vertices: int
    solver_method: str
    converged: bool
    iterations: int
    solve_time: float
    memory_peak_mb: float
    memory_final_mb: float
    final_residual: float
    convergence_rate: float | None = None
    additional_metrics: dict[str, Any] | None = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class BenchmarkSuite:
    """Configuration for benchmark test suite."""

    name: str
    test_problems: list[dict[str, Any]]
    solver_methods: list[str]
    grid_sizes: list[tuple[int, ...]]
    repetitions: int = 3
    timeout_seconds: float = 600.0  # 10 minutes max per test
    save_results: bool = True
    output_directory: str = "benchmark_results"


class HighDimMFGBenchmark:
    """Main benchmarking engine for high-dimensional MFG problems."""

    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize benchmark system.

        Args:
            output_dir: Directory for saving benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.monitor = PerformanceMonitor()
        self.results: list[BenchmarkResult] = []

    def run_convergence_benchmark(
        self, grid_sizes: list[tuple[int, ...]], solver_methods: list[str] | None = None
    ) -> list[BenchmarkResult]:
        """
        Run convergence analysis across different grid sizes.

        Args:
            grid_sizes: List of grid dimensions to test
            solver_methods: Solver methods to compare

        Returns:
            List of benchmark results
        """
        if solver_methods is None:
            solver_methods = ["damped_fixed_point"]
            # Note: "particle_collocation" has been removed from core package

        results = []

        for grid_size in grid_sizes:
            dimension = len(grid_size)
            total_vertices = np.prod(grid_size)

            logger.info(f"Testing {dimension}D grid {grid_size} ({total_vertices} vertices)")

            # Create test problem
            if dimension == 2:
                from typing import cast

                problem = self._create_2d_test_problem(cast("tuple[int, int]", grid_size))
            elif dimension == 3:
                from typing import cast

                problem = self._create_3d_test_problem(cast("tuple[int, int, int]", grid_size))
            else:
                logger.warning(f"Unsupported dimension {dimension}, skipping")
                continue

            for method in solver_methods:
                try:
                    result = self._run_single_benchmark(problem, method, f"convergence_{dimension}d", grid_size)
                    results.append(result)
                    self.results.append(result)

                except Exception as e:
                    logger.error(f"Benchmark failed for {method} on {grid_size}: {e}")

        return results

    def run_solver_comparison(
        self, grid_size: tuple[int, ...], solver_configs: list[dict[str, Any]]
    ) -> list[BenchmarkResult]:
        """
        Compare different solver configurations on the same problem.

        Args:
            grid_size: Fixed grid size for comparison
            solver_configs: List of solver configuration dictionaries

        Returns:
            List of benchmark results
        """
        dimension = len(grid_size)
        results = []

        # Create test problem once
        if dimension == 2:
            from typing import cast

            problem = self._create_2d_test_problem(cast("tuple[int, int]", grid_size))
        elif dimension == 3:
            from typing import cast

            problem = self._create_3d_test_problem(cast("tuple[int, int, int]", grid_size))
        else:
            raise ValueError(f"Unsupported dimension {dimension}")

        logger.info(f"Comparing solvers on {dimension}D grid {grid_size}")

        for i, config in enumerate(solver_configs):
            method = config.get("method", f"custom_{i}")
            try:
                result = self._run_single_benchmark(
                    problem, method, f"solver_comparison_{dimension}d", grid_size, config
                )
                results.append(result)
                self.results.append(result)

            except Exception as e:
                logger.error(f"Solver comparison failed for {method}: {e}")

        return results

    def run_scaling_analysis(self, base_grid: tuple[int, ...], scaling_factors: list[float]) -> list[BenchmarkResult]:
        """
        Analyze computational scaling with problem size.

        Args:
            base_grid: Base grid dimensions
            scaling_factors: Factors to scale the base grid

        Returns:
            List of benchmark results showing scaling behavior
        """
        results = []
        dimension = len(base_grid)

        for factor in scaling_factors:
            scaled_grid = tuple(int(dim * factor) for dim in base_grid)
            total_vertices = np.prod(scaled_grid)

            logger.info(f"Scaling test: factor {factor:.2f}, grid {scaled_grid} ({total_vertices} vertices)")

            if dimension == 2:
                from typing import cast

                problem = self._create_2d_test_problem(cast("tuple[int, int]", scaled_grid))
            elif dimension == 3:
                from typing import cast

                problem = self._create_3d_test_problem(cast("tuple[int, int, int]", scaled_grid))
            else:
                continue

            try:
                result = self._run_single_benchmark(problem, "damped_fixed_point", f"scaling_{dimension}d", scaled_grid)
                result.additional_metrics = result.additional_metrics or {}
                result.additional_metrics["scaling_factor"] = factor
                results.append(result)
                self.results.append(result)

            except Exception as e:
                logger.error(f"Scaling test failed for factor {factor}: {e}")

        return results

    def run_memory_profiling(self, grid_sizes: list[tuple[int, ...]]) -> list[BenchmarkResult]:
        """
        Profile memory usage across different problem sizes.

        Args:
            grid_sizes: Grid sizes to test for memory usage

        Returns:
            List of results with detailed memory profiling
        """
        results = []

        for grid_size in grid_sizes:
            dimension = len(grid_size)
            total_vertices = np.prod(grid_size)

            logger.info(f"Memory profiling {dimension}D grid {grid_size} ({total_vertices} vertices)")

            # Monitor memory during problem creation
            with self.monitor.monitor_operation(f"create_problem_{dimension}d"):
                if dimension == 2:
                    from typing import cast

                    problem = self._create_2d_test_problem(cast("tuple[int, int]", grid_size))
                elif dimension == 3:
                    from typing import cast

                    problem = self._create_3d_test_problem(cast("tuple[int, int, int]", grid_size))
                else:
                    continue

            creation_metrics = self.monitor.metrics_history[-1]

            # Monitor memory during solving
            try:
                result = self._run_single_benchmark(
                    problem, "damped_fixed_point", f"memory_profile_{dimension}d", grid_size
                )

                # Add creation metrics to results
                result.additional_metrics = result.additional_metrics or {}
                result.additional_metrics.update(
                    {
                        "creation_time": creation_metrics.duration,
                        "creation_memory_mb": creation_metrics.memory_used,
                        "creation_peak_mb": creation_metrics.memory_peak_delta,
                    }
                )

                results.append(result)
                self.results.append(result)

            except Exception as e:
                logger.error(f"Memory profiling failed for {grid_size}: {e}")

        return results

    def _create_2d_test_problem(self, grid_size: tuple[int, int]):
        """Create standardized 2D test problem."""
        from mfgarchon import MFGComponents, MFGProblem

        class SimpleBenchmarkProblem(MFGProblem):
            def __init__(self, spatial_bounds, grid_resolution, T, Nt):
                super().__init__(
                    spatial_bounds=spatial_bounds,
                    spatial_discretization=list(grid_resolution),
                    T=T,
                    Nt=Nt,
                    volatility=0.1,
                )
                self.grid_resolution = grid_resolution

            def setup_components(self):
                # Modern derivs format: gradient keys are tuples like (1,) for 1D, (1,0) for x in 2D
                def simple_hamiltonian(x_idx, x_position, m_at_x, derivs, t_idx, current_time, problem, **kwargs):
                    try:
                        # For 1D: gradient is derivs.get((1,), 0.0)
                        p = derivs.get((1,), 0.0) if isinstance(derivs, dict) else 0.0
                        return 0.5 * p**2 + 0.1 * m_at_x * p**2
                    except (KeyError, TypeError, ValueError, IndexError, FloatingPointError):
                        return 0.0

                def hamiltonian_dm(x_idx, x_position, m_at_x, derivs, t_idx, current_time, problem, **kwargs):
                    try:
                        p = derivs.get((1,), 0.0) if isinstance(derivs, dict) else 0.0
                        return 0.1 * p**2
                    except (KeyError, TypeError, ValueError, IndexError, FloatingPointError):
                        return 0.0

                def initial_density_grid(x_position):
                    try:
                        x_idx = int(x_position)
                        if x_idx >= self.num_spatial_points:
                            return 1e-10
                        coords = self.mesh_data.vertices[x_idx]
                        center = np.array([0.5] * self.dimension)
                        distance = np.linalg.norm(coords - center)
                        return max(np.exp(-(distance**2) / (2 * 0.2**2)), 1e-10)
                    except (ValueError, IndexError, AttributeError):
                        return 1e-10

                def terminal_cost_grid(x_position):
                    try:
                        x_idx = int(x_position)
                        if x_idx >= self.num_spatial_points:
                            return 0.0
                        coords = self.mesh_data.vertices[x_idx]
                        target = np.array([0.8] * self.dimension)
                        distance = np.linalg.norm(coords - target)
                        return 0.5 * distance**2
                    except (ValueError, IndexError, AttributeError):
                        return 0.0

                def running_cost_grid(x_idx, x_position, m_at_x, t_idx, current_time, problem, **kwargs):
                    return 0.01

                return MFGComponents(
                    hamiltonian_func=simple_hamiltonian,
                    hamiltonian_dm_func=hamiltonian_dm,
                    m_initial=initial_density_grid,
                    u_terminal=terminal_cost_grid,
                )

            def hamiltonian(self, x_idx, x_position, m_at_x, derivs, t_idx, current_time, problem, **kwargs):
                try:
                    p = derivs.get((1,), 0.0) if isinstance(derivs, dict) else 0.0
                    return 0.5 * p**2 + 0.1 * m_at_x * p**2
                except (KeyError, TypeError, ValueError, IndexError, FloatingPointError):
                    return 0.0

            def initial_density(self, x_position):
                try:
                    x_idx = int(x_position)
                    if x_idx >= self.num_spatial_points:
                        return 1e-10
                    coords = self.mesh_data.vertices[x_idx]
                    center = np.array([0.5] * self.dimension)
                    distance = np.linalg.norm(coords - center)
                    return max(np.exp(-(distance**2) / (2 * 0.2**2)), 1e-10)
                except (ValueError, IndexError, AttributeError):
                    return 1e-10

            def terminal_cost(self, x_position):
                try:
                    x_idx = int(x_position)
                    if x_idx >= self.num_spatial_points:
                        return 0.0
                    coords = self.mesh_data.vertices[x_idx]
                    target = np.array([0.8] * self.dimension)
                    distance = np.linalg.norm(coords - target)
                    return 0.5 * distance**2
                except (ValueError, IndexError, AttributeError):
                    return 0.0

            def running_cost(self, x_idx, x_position, m_at_x, t_idx, current_time, problem, **kwargs):
                return 0.01

        return SimpleBenchmarkProblem(spatial_bounds=[(0.0, 1.0), (0.0, 1.0)], grid_resolution=grid_size, T=1.0, Nt=21)

    def _create_3d_test_problem(self, grid_size: tuple[int, int, int]):
        """Create standardized 3D test problem."""
        from mfgarchon import MFGComponents, MFGProblem

        class SimpleBenchmarkProblem(MFGProblem):
            def __init__(self, spatial_bounds, grid_resolution, T, Nt):
                super().__init__(
                    spatial_bounds=spatial_bounds,
                    spatial_discretization=list(grid_resolution),
                    T=T,
                    Nt=Nt,
                    volatility=0.1,
                )
                self.grid_resolution = grid_resolution

            def setup_components(self):
                # Modern derivs format: gradient keys are tuples like (1,) for 1D, (1,0) for x in 2D
                def simple_hamiltonian(x_idx, x_position, m_at_x, derivs, t_idx, current_time, problem, **kwargs):
                    try:
                        # For 1D: gradient is derivs.get((1,), 0.0)
                        p = derivs.get((1,), 0.0) if isinstance(derivs, dict) else 0.0
                        return 0.5 * p**2 + 0.1 * m_at_x * p**2
                    except (KeyError, TypeError, ValueError, IndexError, FloatingPointError):
                        return 0.0

                def hamiltonian_dm(x_idx, x_position, m_at_x, derivs, t_idx, current_time, problem, **kwargs):
                    try:
                        p = derivs.get((1,), 0.0) if isinstance(derivs, dict) else 0.0
                        return 0.1 * p**2
                    except (KeyError, TypeError, ValueError, IndexError, FloatingPointError):
                        return 0.0

                def initial_density_grid(x_position):
                    try:
                        x_idx = int(x_position)
                        if x_idx >= self.num_spatial_points:
                            return 1e-10
                        coords = self.mesh_data.vertices[x_idx]
                        center = np.array([0.5] * self.dimension)
                        distance = np.linalg.norm(coords - center)
                        return max(np.exp(-(distance**2) / (2 * 0.2**2)), 1e-10)
                    except (ValueError, IndexError, AttributeError):
                        return 1e-10

                def terminal_cost_grid(x_position):
                    try:
                        x_idx = int(x_position)
                        if x_idx >= self.num_spatial_points:
                            return 0.0
                        coords = self.mesh_data.vertices[x_idx]
                        target = np.array([0.8] * self.dimension)
                        distance = np.linalg.norm(coords - target)
                        return 0.5 * distance**2
                    except (ValueError, IndexError, AttributeError):
                        return 0.0

                def running_cost_grid(x_idx, x_position, m_at_x, t_idx, current_time, problem, **kwargs):
                    return 0.01

                return MFGComponents(
                    hamiltonian_func=simple_hamiltonian,
                    hamiltonian_dm_func=hamiltonian_dm,
                    m_initial=initial_density_grid,
                    u_terminal=terminal_cost_grid,
                )

            def hamiltonian(self, x_idx, x_position, m_at_x, derivs, t_idx, current_time, problem, **kwargs):
                try:
                    p = derivs.get((1,), 0.0) if isinstance(derivs, dict) else 0.0
                    return 0.5 * p**2 + 0.1 * m_at_x * p**2
                except (KeyError, TypeError, ValueError, IndexError, FloatingPointError):
                    return 0.0

            def initial_density(self, x_position):
                try:
                    x_idx = int(x_position)
                    if x_idx >= self.num_spatial_points:
                        return 1e-10
                    coords = self.mesh_data.vertices[x_idx]
                    center = np.array([0.5] * self.dimension)
                    distance = np.linalg.norm(coords - center)
                    return max(np.exp(-(distance**2) / (2 * 0.2**2)), 1e-10)
                except (ValueError, IndexError, AttributeError):
                    return 1e-10

            def terminal_cost(self, x_position):
                try:
                    x_idx = int(x_position)
                    if x_idx >= self.num_spatial_points:
                        return 0.0
                    coords = self.mesh_data.vertices[x_idx]
                    target = np.array([0.8] * self.dimension)
                    distance = np.linalg.norm(coords - target)
                    return 0.5 * distance**2
                except (ValueError, IndexError, AttributeError):
                    return 0.0

            def running_cost(self, x_idx, x_position, m_at_x, t_idx, current_time, problem, **kwargs):
                return 0.01

        return SimpleBenchmarkProblem(
            spatial_bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], grid_resolution=grid_size, T=1.0, Nt=21
        )

    def _run_single_benchmark(
        self,
        problem: MFGProblem,
        method: str,
        test_name: str,
        grid_size: tuple[int, ...],
        config: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Run a single benchmark test with comprehensive monitoring."""
        dimension = len(grid_size)
        total_vertices = np.prod(grid_size)

        # Default solver parameters
        default_params = {"damping_factor": 0.5, "max_iterations": 50, "tolerance": 1e-4}

        if config:
            default_params.update(config)

        # Memory monitoring
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024**2)  # MB

        start_time = time.time()
        converged = False
        iterations = 0
        final_residual = float("inf")

        try:
            if method == "damped_fixed_point":
                result = problem.solve_with_damped_fixed_point(**default_params)  # type: ignore[arg-type]
            else:
                raise ValueError(
                    f"Unknown method: {method}. Note: 'particle_collocation' has been removed from core package."
                )

            # Extract convergence information
            converged = result.get("converged", False)
            iterations = result.get("iterations", 0)
            final_residual = result.get("final_residual", float("inf"))

        except Exception as e:
            logger.error(f"Solver failed: {e}")
            result = {}

        end_time = time.time()
        solve_time = end_time - start_time

        # Final memory measurement
        memory_after = process.memory_info().rss / (1024**2)  # MB
        memory_peak = memory_after  # Simplified - could use more sophisticated monitoring

        # Calculate convergence rate if multiple iterations
        convergence_rate = None
        if iterations > 1 and final_residual < 1.0:
            convergence_rate = -np.log(final_residual) / iterations

        return BenchmarkResult(
            test_name=test_name,
            dimension=dimension,
            grid_size=grid_size,
            total_vertices=int(total_vertices),
            solver_method=method,
            converged=converged,
            iterations=iterations,
            solve_time=solve_time,
            memory_peak_mb=memory_peak,
            memory_final_mb=memory_after,
            final_residual=final_residual,
            convergence_rate=convergence_rate,
            additional_metrics={
                "memory_used_mb": memory_after - memory_before,
                "vertices_per_second": total_vertices / solve_time if solve_time > 0 else 0,
                "solver_config": default_params,
            },
        )

    def generate_benchmark_report(self, save_plots: bool = True) -> dict[str, Any]:
        """
        Generate comprehensive benchmark report with analysis and visualizations.

        Args:
            save_plots: Whether to save plot files

        Returns:
            Dictionary containing benchmark analysis
        """
        if not self.results:
            logger.warning("No benchmark results available")
            return {}

        # Organize results by test type and dimension
        results_by_test: dict[str, list[BenchmarkResult]] = {}
        for result in self.results:
            test_key = f"{result.test_name}_{result.dimension}d"
            if test_key not in results_by_test:
                results_by_test[test_key] = []
            results_by_test[test_key].append(result)

        # Generate analysis
        analysis = {
            "summary": self._generate_summary_statistics(),
            "convergence_analysis": self._analyze_convergence_behavior(),
            "performance_analysis": self._analyze_performance_scaling(),
            "memory_analysis": self._analyze_memory_usage(),
            "solver_comparison": self._compare_solver_methods(),
        }

        # Generate plots if requested
        if save_plots:
            self._generate_benchmark_plots()

        # Save full results
        self._save_benchmark_data(analysis)

        return analysis

    def _generate_summary_statistics(self) -> dict[str, Any]:
        """Generate summary statistics from all benchmark results."""
        if not self.results:
            return {}

        successful_results = [r for r in self.results if r.converged]
        total_tests = len(self.results)
        successful_tests = len(successful_results)

        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "average_solve_time": np.mean([r.solve_time for r in successful_results]),
            "average_iterations": np.mean([r.iterations for r in successful_results]),
            "average_memory_mb": np.mean([r.memory_peak_mb for r in successful_results]),
            "dimensions_tested": list({r.dimension for r in self.results}),
            "max_vertices_tested": max(r.total_vertices for r in self.results),
            "solver_methods": list({r.solver_method for r in self.results}),
        }

    def _analyze_convergence_behavior(self) -> dict[str, Any]:
        """Analyze convergence behavior across different problem sizes."""
        convergence_results = [r for r in self.results if "convergence" in r.test_name]

        if not convergence_results:
            return {}

        # Group by dimension
        by_dimension: dict[int, list[BenchmarkResult]] = {}
        for result in convergence_results:
            dim = result.dimension
            if dim not in by_dimension:
                by_dimension[dim] = []
            by_dimension[dim].append(result)

        analysis = {}
        for dim, results in by_dimension.items():
            successful = [r for r in results if r.converged]
            if successful:
                analysis[f"{dim}d"] = {
                    "convergence_rate": len(successful) / len(results),
                    "avg_iterations": np.mean([r.iterations for r in successful]),
                    "avg_solve_time": np.mean([r.solve_time for r in successful]),
                    "scaling_exponent": self._estimate_scaling_exponent(successful),
                }

        return analysis

    def _analyze_performance_scaling(self) -> dict[str, Any]:
        """Analyze performance scaling with problem size."""
        scaling_results = [r for r in self.results if "scaling" in r.test_name]

        if not scaling_results:
            return {}

        # Sort by total vertices
        scaling_results.sort(key=lambda r: r.total_vertices)

        vertices = [r.total_vertices for r in scaling_results]
        times = [r.solve_time for r in scaling_results]

        # Fit power law: time ~ vertices^alpha
        if len(vertices) > 2:
            log_vertices = np.log(vertices)
            log_times = np.log(times)
            alpha = np.polyfit(log_vertices, log_times, 1)[0]
        else:
            alpha = None

        return {
            "scaling_exponent": alpha,
            "min_vertices": min(vertices),
            "max_vertices": max(vertices),
            "min_time": min(times),
            "max_time": max(times),
            "efficiency_trend": "improving" if alpha and alpha < 2 else "degrading",
        }

    def _analyze_memory_usage(self) -> dict[str, Any]:
        """Analyze memory usage patterns."""
        memory_results = list(self.results)

        if not memory_results:
            return {}

        vertices = [r.total_vertices for r in memory_results]
        memory = [r.memory_peak_mb for r in memory_results]

        # Memory scaling analysis
        if len(vertices) > 2:
            log_vertices = np.log(vertices)
            log_memory = np.log(memory)
            memory_scaling = np.polyfit(log_vertices, log_memory, 1)[0]
        else:
            memory_scaling = None

        return {
            "memory_scaling_exponent": memory_scaling,
            "avg_memory_per_vertex_kb": np.mean([m / v * 1024 for m, v in zip(memory, vertices, strict=False)]),
            "max_memory_mb": max(memory),
            "memory_efficiency": "good" if memory_scaling and memory_scaling < 2 else "needs_improvement",
        }

    def _compare_solver_methods(self) -> dict[str, Any]:
        """Compare performance of different solver methods."""
        by_method: dict[str, list[BenchmarkResult]] = {}
        for result in self.results:
            method = result.solver_method
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(result)

        comparison = {}
        for method, results in by_method.items():
            successful = [r for r in results if r.converged]
            if successful:
                comparison[method] = {
                    "success_rate": len(successful) / len(results),
                    "avg_solve_time": np.mean([r.solve_time for r in successful]),
                    "avg_iterations": np.mean([r.iterations for r in successful]),
                    "avg_memory_mb": np.mean([r.memory_peak_mb for r in successful]),
                    "total_tests": len(results),
                }

        return comparison

    def _estimate_scaling_exponent(self, results: list[BenchmarkResult]) -> float | None:
        """Estimate computational scaling exponent."""
        if len(results) < 3:
            return None

        # Sort by problem size
        results.sort(key=lambda r: r.total_vertices)
        vertices = [r.total_vertices for r in results]
        times = [r.solve_time for r in results]

        # Fit power law
        log_vertices = np.log(vertices)
        log_times = np.log(times)
        return np.polyfit(log_vertices, log_times, 1)[0]

    def _generate_benchmark_plots(self):
        """Generate visualization plots for benchmark results."""
        # Scaling plot
        _fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Solve time vs problem size
        for dim in [2, 3]:
            results = [r for r in self.results if r.dimension == dim and r.converged]
            if results:
                vertices = [r.total_vertices for r in results]
                times = [r.solve_time for r in results]
                ax1.loglog(vertices, times, "o-", label=f"{dim}D")

        ax1.set_xlabel("Total Vertices")
        ax1.set_ylabel("Solve Time (s)")
        ax1.set_title("Computational Scaling")
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Memory usage vs problem size
        for dim in [2, 3]:
            results = [r for r in self.results if r.dimension == dim and r.converged]
            if results:
                vertices = [r.total_vertices for r in results]
                memory = [r.memory_peak_mb for r in results]
                ax2.loglog(vertices, memory, "s-", label=f"{dim}D")

        ax2.set_xlabel("Total Vertices")
        ax2.set_ylabel("Peak Memory (MB)")
        ax2.set_title("Memory Scaling")
        ax2.legend()
        ax2.grid(True)

        # Plot 3: Convergence iterations
        for method in {r.solver_method for r in self.results}:
            results = [r for r in self.results if r.solver_method == method and r.converged]
            if results:
                vertices = [r.total_vertices for r in results]
                iterations = [r.iterations for r in results]
                ax3.semilogx(vertices, iterations, "v-", label=method)

        ax3.set_xlabel("Total Vertices")
        ax3.set_ylabel("Iterations to Convergence")
        ax3.set_title("Convergence Behavior")
        ax3.legend()
        ax3.grid(True)

        # Plot 4: Solver method comparison
        methods = list({r.solver_method for r in self.results})
        avg_times = []
        for method in methods:
            results = [r for r in self.results if r.solver_method == method and r.converged]
            if results:
                avg_times.append(np.mean([r.solve_time for r in results]))
            else:
                avg_times.append(0)

        ax4.bar(methods, avg_times)
        ax4.set_ylabel("Average Solve Time (s)")
        ax4.set_title("Solver Method Comparison")
        ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / "benchmark_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Benchmark plots saved to {self.output_dir}")

    def _save_benchmark_data(self, analysis: dict[str, Any]):
        """Save benchmark results and analysis to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw results
        results_data = [asdict(result) for result in self.results]
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        # Save analysis
        analysis_file = self.output_dir / f"benchmark_analysis_{timestamp}.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2, default=str)

        logger.info(f"Benchmark data saved to {self.output_dir}")


# Predefined benchmark suites
def create_quick_benchmark_suite() -> BenchmarkSuite:
    """Create a quick benchmark suite for development testing."""
    return BenchmarkSuite(
        name="quick_test",
        test_problems=[{"type": "convergence"}, {"type": "solver_comparison"}],
        solver_methods=["damped_fixed_point"],
        grid_sizes=[(8, 8), (16, 16), (8, 8, 8)],
        repetitions=1,
        timeout_seconds=120.0,
    )


def create_comprehensive_benchmark_suite() -> BenchmarkSuite:
    """Create a comprehensive benchmark suite for full evaluation."""
    return BenchmarkSuite(
        name="comprehensive",
        test_problems=[
            {"type": "convergence"},
            {"type": "solver_comparison"},
            {"type": "scaling"},
            {"type": "memory_profiling"},
        ],
        solver_methods=["damped_fixed_point"],
        # Note: "particle_collocation" has been removed from core package
        grid_sizes=[
            (16, 16),
            (32, 32),
            (64, 64),  # 2D tests
            (8, 8, 8),
            (16, 16, 16),
            (32, 32, 32),  # 3D tests
        ],
        repetitions=3,
        timeout_seconds=600.0,
    )


def run_standard_benchmarks(output_dir: str = "benchmark_results") -> dict[str, Any]:
    """
    Run standard benchmark suite and return comprehensive analysis.

    Args:
        output_dir: Directory for saving results

    Returns:
        Dictionary containing complete benchmark analysis
    """
    benchmark = HighDimMFGBenchmark(output_dir)

    logger.info("Starting comprehensive high-dimensional MFG benchmarks")

    # Run convergence analysis
    convergence_grids = [(16, 16), (32, 32), (8, 8, 8), (16, 16, 16)]
    benchmark.run_convergence_benchmark(convergence_grids)

    # Run scaling analysis
    base_2d = (16, 16)
    base_3d = (8, 8, 8)
    scaling_factors = [1.0, 1.5, 2.0]

    benchmark.run_scaling_analysis(base_2d, scaling_factors)
    benchmark.run_scaling_analysis(base_3d, scaling_factors)

    # Run memory profiling
    memory_grids = [(16, 16), (32, 32), (8, 8, 8), (16, 16, 16)]
    benchmark.run_memory_profiling(memory_grids)

    # Generate comprehensive report
    analysis = benchmark.generate_benchmark_report(save_plots=True)

    logger.info("Benchmark suite completed successfully")
    return analysis

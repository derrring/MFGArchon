"""
Performance Monitoring System for MFGarchon Package

This module provides comprehensive performance monitoring capabilities including
execution time tracking, memory usage analysis, and regression detection for
the MFGarchon solver framework.
"""

from __future__ import annotations

import functools
import json
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import psutil

import numpy as np

from mfgarchon.utils.mfg_logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for comprehensive performance metrics."""

    method_name: str
    execution_time: float
    peak_memory_mb: float
    cpu_percent: float
    problem_size: dict[str, int] = field(default_factory=dict)  # Nx, Nt, etc.
    convergence_info: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    git_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerformanceMetrics:
        """Create instance from dictionary."""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection."""

    method_name: str
    mean_execution_time: float
    std_execution_time: float
    mean_memory_mb: float
    std_memory_mb: float
    sample_count: int
    last_updated: datetime = field(default_factory=datetime.now)

    def is_regression(
        self,
        metrics: PerformanceMetrics,
        time_threshold: float = 1.5,
        memory_threshold: float = 1.3,
    ) -> tuple[bool, str]:
        """
        Check if metrics indicate performance regression.

        Args:
            metrics: Performance metrics to check
            time_threshold: Factor above baseline mean to consider regression
            memory_threshold: Factor above baseline mean to consider regression

        Returns:
            Tuple of (is_regression, description)
        """
        time_regression = metrics.execution_time > self.mean_execution_time * time_threshold
        memory_regression = metrics.peak_memory_mb > self.mean_memory_mb * memory_threshold

        issues = []
        if time_regression:
            slowdown = metrics.execution_time / self.mean_execution_time
            issues.append(f"Execution time regression: {slowdown:.2f}x slower than baseline")

        if memory_regression:
            increase = metrics.peak_memory_mb / self.mean_memory_mb
            issues.append(f"Memory usage regression: {increase:.2f}x higher than baseline")

        return (time_regression or memory_regression, "; ".join(issues))


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""

    def __init__(self, storage_path: Path | None = None):
        """
        Initialize performance monitor.

        Args:
            storage_path: Path to store performance data (default: ./performance_data)
        """
        # No mkdir here. `global_performance_monitor = PerformanceMonitor()` runs at module
        # level, so this constructor created `performance_data/` in the CALLER's working
        # directory as a side effect of `import mfgarchon`. `_load_stored_data` below only
        # reads, and guards on `.exists()`; `_save_data` is the one path that needs the
        # directory, and it creates it. #1674, same shape as #1917.
        self.storage_path = storage_path or Path("performance_data")

        self.metrics_history: dict[str, list[PerformanceMetrics]] = {}
        self.baselines: dict[str, PerformanceBaseline] = {}

        # Load existing data
        self._load_stored_data()

    def _load_stored_data(self) -> None:
        """Load previously stored performance data."""
        try:
            # Load baselines
            baseline_file = self.storage_path / "baselines.json"
            if baseline_file.exists():
                with open(baseline_file) as f:
                    baseline_data = json.load(f)
                for name, data in baseline_data.items():
                    data["last_updated"] = datetime.fromisoformat(data["last_updated"])
                    self.baselines[name] = PerformanceBaseline(**data)

            # Load metrics history (last 100 entries per method)
            metrics_file = self.storage_path / "metrics_history.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    history_data = json.load(f)
                for name, metrics_list in history_data.items():
                    self.metrics_history[name] = [PerformanceMetrics.from_dict(m) for m in metrics_list[-100:]]
        except Exception as e:
            warnings.warn(f"Failed to load performance data: {e}", UserWarning)

    def _save_data(self) -> None:
        """Save performance data to storage."""
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            # Save baselines
            baseline_file = self.storage_path / "baselines.json"
            baseline_data = {}
            for name, baseline in self.baselines.items():
                data = asdict(baseline)
                data["last_updated"] = baseline.last_updated.isoformat()
                baseline_data[name] = data

            with open(baseline_file, "w") as f:
                json.dump(baseline_data, f, indent=2)

            # Save metrics history (last 100 entries per method)
            metrics_file = self.storage_path / "metrics_history.json"
            history_data = {}
            for name, metrics_list in self.metrics_history.items():
                history_data[name] = [m.to_dict() for m in metrics_list[-100:]]

            with open(metrics_file, "w") as f:
                json.dump(history_data, f, indent=2)

        except Exception as e:
            warnings.warn(f"Failed to save performance data: {e}", UserWarning)

    def performance_tracked(
        self,
        method_name: str | None = None,
        track_convergence: bool = True,
        update_baseline: bool = True,
    ):
        """
        Decorator to track performance of methods.

        Args:
            method_name: Override method name for tracking
            track_convergence: Whether to extract convergence information
            update_baseline: Whether to update performance baseline
        """

        def decorator(func: Callable) -> Callable:
            name = method_name or f"{func.__module__}.{func.__qualname__}"

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Extract problem size if available
                problem_size = {}
                if args and hasattr(args[0], "problem"):
                    problem = args[0].problem
                    if hasattr(problem, "geometry"):
                        grid_shape = problem.geometry.get_grid_shape()
                        problem_size["Nx"] = grid_shape[0] - 1  # intervals
                    if hasattr(problem, "Nt"):
                        problem_size["Nt"] = problem.Nt

                # Start monitoring
                start_time = time.time()
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024  # MB

                try:
                    result = func(*args, **kwargs)

                    # Calculate metrics
                    execution_time = time.time() - start_time
                    end_memory = process.memory_info().rss / 1024 / 1024
                    peak_memory = max(start_memory, end_memory)
                    cpu_percent = process.cpu_percent()

                    # Extract convergence information
                    convergence_info = {}
                    if track_convergence and hasattr(result, "convergence_info"):
                        convergence_info = result.convergence_info
                    elif track_convergence and args and hasattr(args[0], "iterations_run"):
                        # For iterative solvers
                        solver = args[0]
                        convergence_info = {
                            "iterations": getattr(solver, "iterations_run", 0),
                            "converged": getattr(solver, "convergence_achieved", False),
                        }

                    # Create metrics object
                    metrics = PerformanceMetrics(
                        method_name=name,
                        execution_time=execution_time,
                        peak_memory_mb=peak_memory,
                        cpu_percent=cpu_percent,
                        problem_size=problem_size,
                        convergence_info=convergence_info,
                    )

                    # Store metrics
                    if name not in self.metrics_history:
                        self.metrics_history[name] = []
                    self.metrics_history[name].append(metrics)

                    # Check for regression
                    self._check_regression(metrics)

                    # Update baseline if requested
                    if update_baseline:
                        self._update_baseline(name)

                    # Save data periodically
                    if len(self.metrics_history[name]) % 10 == 0:
                        self._save_data()

                    return result

                except Exception as e:
                    # Issue #547: Use logger instead of print for consistency
                    # Still track failed execution time before re-raising
                    execution_time = time.time() - start_time
                    logger.warning(
                        "Performance tracking for '%s' failed after %.2fs: %s",
                        name,
                        execution_time,
                        e,
                    )
                    raise

            return wrapper

        return decorator

    def _check_regression(self, metrics: PerformanceMetrics) -> None:
        """Check for performance regression against baseline."""
        if metrics.method_name in self.baselines:
            baseline = self.baselines[metrics.method_name]
            is_regression, description = baseline.is_regression(metrics)

            if is_regression:
                warning_msg = (
                    f"WARNING:  Performance regression detected in {metrics.method_name}:\n"
                    f"   {description}\n"
                    f"   Current: {metrics.execution_time:.2f}s, {metrics.peak_memory_mb:.1f}MB\n"
                    f"   Baseline: {baseline.mean_execution_time:.2f}s, {baseline.mean_memory_mb:.1f}MB"
                )
                warnings.warn(warning_msg, UserWarning)

    def _update_baseline(self, method_name: str, min_samples: int = 5) -> None:
        """Update performance baseline for a method."""
        if method_name not in self.metrics_history:
            return

        recent_metrics = self.metrics_history[method_name][-20:]  # Last 20 runs

        if len(recent_metrics) < min_samples:
            return

        # Calculate statistics
        times = [m.execution_time for m in recent_metrics]
        memories = [m.peak_memory_mb for m in recent_metrics]

        baseline = PerformanceBaseline(
            method_name=method_name,
            mean_execution_time=float(np.mean(times)),
            std_execution_time=float(np.std(times)),
            mean_memory_mb=float(np.mean(memories)),
            std_memory_mb=float(np.std(memories)),
            sample_count=len(recent_metrics),
        )

        self.baselines[method_name] = baseline

    def get_performance_report(self, method_name: str | None = None) -> str:
        """
        Generate comprehensive performance report.

        Args:
            method_name: Specific method to report on (None for all methods)

        Returns:
            Formatted performance report
        """
        if method_name and method_name not in self.metrics_history:
            return f"No performance data found for {method_name}"

        methods = [method_name] if method_name else list(self.metrics_history.keys())

        report = f"""
Performance Monitoring Report
============================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Storage: {self.storage_path}

"""

        for method in methods:
            if method not in self.metrics_history:
                continue

            metrics_list = self.metrics_history[method]
            recent_metrics = metrics_list[-10:]  # Last 10 runs

            if not recent_metrics:
                continue

            # Calculate statistics
            times = [m.execution_time for m in recent_metrics]
            memories = [m.peak_memory_mb for m in recent_metrics]

            report += f"\n{method}:\n"
            report += "=" * len(method) + "\n"
            report += f"  Total runs: {len(metrics_list)}\n"
            report += f"  Recent performance (last {len(recent_metrics)} runs):\n"
            report += f"    Execution time: {np.mean(times):.2f}s ± {np.std(times):.2f}s\n"
            report += f"    Memory usage: {np.mean(memories):.1f}MB ± {np.std(memories):.1f}MB\n"

            # Baseline comparison
            if method in self.baselines:
                baseline = self.baselines[method]
                current_time = np.mean(times)
                current_memory = np.mean(memories)

                time_change = (current_time / baseline.mean_execution_time - 1) * 100
                memory_change = (current_memory / baseline.mean_memory_mb - 1) * 100

                report += "  Baseline comparison:\n"
                report += f"    Time change: {time_change:+.1f}%\n"
                report += f"    Memory change: {memory_change:+.1f}%\n"

            # Problem sizes
            if recent_metrics[0].problem_size:
                sizes = recent_metrics[0].problem_size
                report += f"  Problem size: {sizes}\n"

            # Convergence info
            convergence_data = [m.convergence_info for m in recent_metrics if m.convergence_info]
            if convergence_data:
                iterations = [c.get("iterations", 0) for c in convergence_data]
                if iterations:
                    report += f"  Convergence: {np.mean(iterations):.1f} iterations (avg)\n"

        return report

    def export_performance_data(self, output_file: Path) -> None:
        """Export all performance data to JSON file."""
        export_data: dict[str, Any] = {
            "baselines": {name: asdict(baseline) for name, baseline in self.baselines.items()},
            "metrics_history": {name: [m.to_dict() for m in metrics] for name, metrics in self.metrics_history.items()},
            "export_timestamp": datetime.now().isoformat(),
        }

        # Convert datetime objects to ISO format for baselines
        baselines_dict = export_data["baselines"]
        if isinstance(baselines_dict, dict):
            for baseline_data in baselines_dict.values():
                if "last_updated" in baseline_data:
                    if isinstance(baseline_data["last_updated"], datetime):
                        baseline_data["last_updated"] = baseline_data["last_updated"].isoformat()

        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)

    def clear_data(self, method_name: str | None = None) -> None:
        """Clear performance data for specified method or all methods."""
        if method_name:
            self.metrics_history.pop(method_name, None)
            self.baselines.pop(method_name, None)
        else:
            self.metrics_history.clear()
            self.baselines.clear()

        self._save_data()


# Global performance monitor instance
global_performance_monitor = PerformanceMonitor()


# Convenience functions and decorators
def performance_tracked(method_name: str | None = None, **kwargs):
    """Convenience decorator using global monitor."""
    return global_performance_monitor.performance_tracked(method_name, **kwargs)


def get_performance_report(method_name: str | None = None) -> str:
    """Get performance report using global monitor."""
    return global_performance_monitor.get_performance_report(method_name)


def benchmark_solver(
    solver_class, problem, config_variations: list[dict[str, Any]], repetitions: int = 3
) -> dict[str, Any]:
    """
    Benchmark solver performance across different configurations.

    Args:
        solver_class: Solver class to benchmark
        problem: Problem instance
        config_variations: List of configuration dictionaries to test
        repetitions: Number of repetitions per configuration

    Returns:
        Benchmarking results dictionary
    """
    # Extract problem size
    Nx_intervals = 0
    if hasattr(problem, "geometry"):
        grid_shape = problem.geometry.get_grid_shape()
        Nx_intervals = grid_shape[0] - 1

    results: dict[str, Any] = {
        "solver_class": solver_class.__name__,
        "problem_size": {
            "Nx": Nx_intervals,
            "Nt": getattr(problem, "Nt", 0),
        },
        "configurations": [],
    }

    for i, config in enumerate(config_variations):
        config_results: dict[str, Any] = {
            "config_id": i,
            "config": config,
            "runs": [],
            "statistics": {},
        }

        # Run multiple times
        times = []
        memories = []

        for rep in range(repetitions):
            start_time = time.time()
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024

            try:
                solver = solver_class(problem, **config)
                solver.solve()

                execution_time = time.time() - start_time
                end_memory = process.memory_info().rss / 1024 / 1024
                peak_memory = max(start_memory, end_memory)

                run_data = {
                    "repetition": rep,
                    "execution_time": execution_time,
                    "peak_memory_mb": peak_memory,
                    "success": True,
                }

                # Extract convergence info if available
                if hasattr(solver, "iterations_run"):
                    run_data["iterations"] = solver.iterations_run
                if hasattr(solver, "convergence_achieved"):
                    run_data["converged"] = solver.convergence_achieved

                config_results["runs"].append(run_data)
                times.append(execution_time)
                memories.append(peak_memory)

            except Exception as e:
                config_results["runs"].append(
                    {
                        "repetition": rep,
                        "execution_time": time.time() - start_time,
                        "success": False,
                        "error": str(e),
                    }
                )

        # Calculate statistics
        if times:
            config_results["statistics"] = {
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "mean_memory": np.mean(memories),
                "std_memory": np.std(memories),
                "success_rate": len(times) / repetitions,
            }

        results["configurations"].append(config_results)

    return results

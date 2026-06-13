"""
Memory Management Utilities for MFGarchon Package

This module provides tools for monitoring and managing memory usage during
computations, particularly important for large-scale MFG problems that can
consume significant memory.
"""

from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import TYPE_CHECKING, Any

import psutil

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class MemoryStats:
    """Container for memory usage statistics."""

    current_memory_gb: float
    peak_memory_gb: float
    memory_limit_gb: float
    timestamp: datetime = field(default_factory=datetime.now)
    warnings: list[str] = field(default_factory=list)


class MemoryMonitor:
    """Monitor and manage memory usage during computations."""

    def __init__(self, max_memory_gb: float = 8.0, warning_threshold: float = 0.8):
        """
        Initialize memory monitor.

        Args:
            max_memory_gb: Maximum allowed memory usage in GB
            warning_threshold: Fraction of max_memory_gb at which to issue warnings
        """
        self.max_memory_gb = max_memory_gb
        self.warning_threshold = warning_threshold
        self.peak_memory = 0.0
        self.memory_warnings: list[str] = []
        self.process = psutil.Process()

    def get_current_memory_gb(self) -> float:
        """Get current memory usage in GB."""
        memory_info = self.process.memory_info()
        return memory_info.rss / (1024**3)

    def check_memory_usage(self) -> MemoryStats:
        """
        Check current memory usage and return comprehensive statistics.

        Returns:
            MemoryStats object with current usage and warnings
        """
        current_memory = self.get_current_memory_gb()

        # Update peak memory
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory

        # Check for warnings
        warning_threshold_gb = self.max_memory_gb * self.warning_threshold
        current_warnings = []

        if current_memory > warning_threshold_gb:
            warning_msg = (
                f"Memory usage ({current_memory:.2f} GB) exceeds warning threshold ({warning_threshold_gb:.2f} GB)"
            )
            current_warnings.append(warning_msg)
            self.memory_warnings.append(warning_msg)

        if current_memory > self.max_memory_gb:
            critical_msg = f"Memory usage ({current_memory:.2f} GB) exceeds limit ({self.max_memory_gb} GB)"
            current_warnings.append(critical_msg)
            self.memory_warnings.append(critical_msg)

        return MemoryStats(
            current_memory_gb=current_memory,
            peak_memory_gb=self.peak_memory,
            memory_limit_gb=self.max_memory_gb,
            warnings=current_warnings,
        )

    def cleanup_arrays(self, *arrays) -> int:
        """
        Explicitly clean up large arrays and force garbage collection.

        Args:
            *arrays: Variable number of array objects to clean up

        Returns:
            Number of arrays cleaned up
        """
        cleaned_count = 0

        for arr in arrays:
            if hasattr(arr, "shape") and hasattr(arr, "size"):
                # Only clean up large arrays (>1M elements)
                if arr.size > 1_000_000:
                    del arr
                    cleaned_count += 1
            elif hasattr(arr, "__len__"):
                # Handle lists of arrays
                if len(arr) > 1000:
                    del arr
                    cleaned_count += 1

        # Force garbage collection
        gc.collect()

        return cleaned_count

    def get_array_memory_usage(self, array: np.ndarray) -> float:
        """
        Calculate memory usage of a numpy array in GB.

        Args:
            array: Numpy array to analyze

        Returns:
            Memory usage in GB
        """
        if not isinstance(array, np.ndarray):
            return 0.0

        return array.nbytes / (1024**3)

    def suggest_memory_optimization(self) -> list[str]:
        """
        Suggest memory optimization strategies based on current usage.

        Returns:
            List of optimization suggestions
        """
        suggestions = []
        current_memory = self.get_current_memory_gb()

        if current_memory > self.max_memory_gb * 0.7:
            suggestions.extend(
                [
                    "Consider reducing problem size (Nx, Nt)",
                    "Use dtype=np.float32 instead of float64 for large arrays",
                    "Enable warm start to reuse previous solutions",
                    "Process data in smaller chunks",
                ]
            )

        if current_memory > self.max_memory_gb * 0.9:
            suggestions.extend(
                [
                    "CRITICAL: Immediate memory cleanup required",
                    "Consider using out-of-core computation methods",
                    "Increase available system memory",
                ]
            )

        return suggestions


def memory_monitored(
    max_memory_gb: float = 8.0,
    cleanup_on_exit: bool = True,
    raise_on_exceed: bool = False,
):
    """
    Decorator to monitor memory usage during method execution.

    Args:
        max_memory_gb: Maximum allowed memory usage
        cleanup_on_exit: Whether to force garbage collection on exit
        raise_on_exceed: Whether to raise exception if memory limit exceeded

    Example:
        @memory_monitored(max_memory_gb=4.0, raise_on_exceed=True)
        def solve_large_problem(self):
            # Method implementation
            pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            monitor = MemoryMonitor(max_memory_gb)

            # Store monitor on instance for access during execution
            self._memory_monitor = monitor

            # Initial memory check
            monitor.check_memory_usage()

            try:
                result = func(self, *args, **kwargs)

                # Final memory check
                final_stats = monitor.check_memory_usage()

                # Report warnings
                if final_stats.warnings:
                    warnings.warn(
                        f"Memory warnings during {func.__name__}:\n"
                        + "\n".join(f"  - {warning}" for warning in final_stats.warnings),
                        UserWarning,
                    )

                # Raise exception if configured
                if raise_on_exceed and final_stats.current_memory_gb > max_memory_gb:
                    raise MemoryError(
                        f"Memory usage ({final_stats.current_memory_gb:.2f} GB) "
                        f"exceeded limit ({max_memory_gb} GB) in {func.__name__}"
                    )

                return result

            finally:
                if cleanup_on_exit:
                    # Cleanup temporary variables. The decorated method may belong
                    # to any class, so `_temp_arrays` is an optional attribute on a
                    # foreign instance (the decorator owns no __init__ to declare it).
                    # Use the getattr-with-default pattern per CLAUDE.md rather than
                    # hasattr duck-typing or a bare attribute access that would raise
                    # for instances that never define it.
                    temp_arrays = getattr(self, "_temp_arrays", None)
                    if temp_arrays is not None:
                        monitor.cleanup_arrays(*temp_arrays)

                    # Force garbage collection
                    gc.collect()

        return wrapper

    return decorator


class MemoryProfiler:
    """Profile memory usage across multiple solver runs."""

    def __init__(self) -> None:
        self.profiles: dict[str, list[MemoryStats]] = {}

    def start_profiling(self, profile_name: str, max_memory_gb: float = 8.0) -> MemoryMonitor:
        """Start profiling a new computation."""
        monitor = MemoryMonitor(max_memory_gb)
        if profile_name not in self.profiles:
            self.profiles[profile_name] = []
        return monitor

    def add_profile_point(self, profile_name: str, stats: MemoryStats):
        """Add a memory measurement to the profile."""
        if profile_name in self.profiles:
            self.profiles[profile_name].append(stats)

    def get_profile_summary(self, profile_name: str) -> dict[str, Any]:
        """Get summary statistics for a profile."""
        if profile_name not in self.profiles or not self.profiles[profile_name]:
            return {}

        stats_list = self.profiles[profile_name]
        memory_values = [s.current_memory_gb for s in stats_list]

        return {
            "profile_name": profile_name,
            "num_measurements": len(stats_list),
            "mean_memory_gb": np.mean(memory_values),
            "max_memory_gb": np.max(memory_values),
            "min_memory_gb": np.min(memory_values),
            "std_memory_gb": np.std(memory_values),
            "total_warnings": sum(len(s.warnings) for s in stats_list),
            "duration": stats_list[-1].timestamp - stats_list[0].timestamp,
        }


# Utility functions for common memory management tasks


def estimate_problem_memory_requirements(
    nx: int, nt: int, num_solvers: int = 2, dtype_size: int = 8
) -> dict[str, float]:
    """
    Estimate memory requirements for an MFG problem.

    Args:
        nx: Number of spatial grid points
        nt: Number of temporal grid points
        num_solvers: Number of concurrent solvers (HJB + FP)
        dtype_size: Size of floating point type in bytes (8 for float64)

    Returns:
        Dictionary with memory estimates in GB
    """
    # Base arrays: U, M solutions
    solution_memory = 2 * (nt + 1) * (nx + 1) * dtype_size

    # Temporary arrays during computation
    temp_memory = solution_memory * 0.5  # Conservative estimate

    # Solver overhead
    solver_overhead = solution_memory * 0.2

    # Total per solver
    per_solver_memory = solution_memory + temp_memory + solver_overhead
    total_memory = per_solver_memory * num_solvers

    return {
        "solution_arrays_gb": solution_memory / (1024**3),
        "temporary_arrays_gb": temp_memory / (1024**3),
        "solver_overhead_gb": solver_overhead / (1024**3),
        "total_estimated_gb": total_memory / (1024**3),
        "recommended_system_gb": total_memory * 1.5 / (1024**3),  # 50% safety margin
    }


def check_system_memory_availability() -> dict[str, float]:
    """
    Check system memory availability.

    Returns:
        Dictionary with system memory information in GB
    """
    memory = psutil.virtual_memory()

    return {
        "total_gb": memory.total / (1024**3),
        "available_gb": memory.available / (1024**3),
        "used_gb": memory.used / (1024**3),
        "free_gb": memory.free / (1024**3),
        "percent_used": memory.percent,
    }


def memory_usage_report(monitor: MemoryMonitor) -> str:
    """
    Generate a formatted memory usage report.

    Args:
        monitor: MemoryMonitor instance

    Returns:
        Formatted report string
    """
    stats = monitor.check_memory_usage()
    system_info = check_system_memory_availability()

    report = f"""
Memory Usage Report
==================
Current Usage: {stats.current_memory_gb:.2f} GB
Peak Usage:    {stats.peak_memory_gb:.2f} GB
Memory Limit:  {stats.memory_limit_gb:.2f} GB
Utilization:   {(stats.current_memory_gb / stats.memory_limit_gb) * 100:.1f}%

System Memory:
  Total:     {system_info["total_gb"]:.2f} GB
  Available: {system_info["available_gb"]:.2f} GB
  Used:      {system_info["percent_used"]:.1f}%

Warnings: {len(stats.warnings)}
"""

    if stats.warnings:
        report += "\nWarnings:\n"
        for i, warning in enumerate(stats.warnings, 1):
            report += f"  {i}. {warning}\n"

    if stats.current_memory_gb > stats.memory_limit_gb * 0.8:
        suggestions = monitor.suggest_memory_optimization()
        report += "\nOptimization Suggestions:\n"
        for i, suggestion in enumerate(suggestions, 1):
            report += f"  {i}. {suggestion}\n"

    return report


# Global memory profiler instance
global_memory_profiler = MemoryProfiler()

#!/usr/bin/env python3
"""
Unit tests for mfgarchon/utils/memory_management.py

Tests comprehensive memory monitoring and management utilities including:
- MemoryStats dataclass
- MemoryMonitor class (monitoring, cleanup, suggestions)
- memory_monitored decorator
- MemoryProfiler class
- Utility functions (estimation, system checks, reporting)

Coverage target: mfgarchon/utils/memory_management.py (379 lines, likely 0% -> 70%+)
"""

import time
import warnings
from datetime import datetime

import pytest

import numpy as np

from mfgarchon.utils.memory_management import (
    MemoryMonitor,
    MemoryProfiler,
    MemoryStats,
    check_system_memory_availability,
    estimate_problem_memory_requirements,
    global_memory_profiler,
    memory_monitored,
    memory_usage_report,
)

# =============================================================================
# Test MemoryStats Dataclass
# =============================================================================


@pytest.mark.unit
def test_memory_stats_creation():
    """Test MemoryStats dataclass creation."""
    stats = MemoryStats(
        current_memory_gb=2.5,
        peak_memory_gb=3.0,
        memory_limit_gb=8.0,
        warnings=["Warning 1", "Warning 2"],
    )

    assert stats.current_memory_gb == 2.5
    assert stats.peak_memory_gb == 3.0
    assert stats.memory_limit_gb == 8.0
    assert len(stats.warnings) == 2
    assert isinstance(stats.timestamp, datetime)


@pytest.mark.unit
def test_memory_stats_defaults():
    """Test MemoryStats default values."""
    stats = MemoryStats(current_memory_gb=1.0, peak_memory_gb=1.5, memory_limit_gb=4.0)

    assert stats.warnings == []
    assert isinstance(stats.timestamp, datetime)


# =============================================================================
# Test MemoryMonitor Class
# =============================================================================


@pytest.mark.unit
def test_memory_monitor_initialization():
    """Test MemoryMonitor initialization."""
    monitor = MemoryMonitor(max_memory_gb=4.0, warning_threshold=0.7)

    assert monitor.max_memory_gb == 4.0
    assert monitor.warning_threshold == 0.7
    assert monitor.peak_memory == 0.0
    assert monitor.memory_warnings == []
    assert monitor.process is not None


@pytest.mark.unit
def test_memory_monitor_defaults():
    """Test MemoryMonitor default values."""
    monitor = MemoryMonitor()

    assert monitor.max_memory_gb == 8.0
    assert monitor.warning_threshold == 0.8


@pytest.mark.unit
def test_get_current_memory_gb():
    """Test getting current memory usage."""
    monitor = MemoryMonitor()
    memory_gb = monitor.get_current_memory_gb()

    assert isinstance(memory_gb, float)
    assert memory_gb > 0.0  # Process uses some memory


@pytest.mark.unit
def test_check_memory_usage_normal():
    """Test memory check with normal usage."""
    monitor = MemoryMonitor(max_memory_gb=100.0)  # High limit to avoid warnings
    stats = monitor.check_memory_usage()

    assert isinstance(stats, MemoryStats)
    assert stats.current_memory_gb > 0.0
    assert stats.peak_memory_gb >= stats.current_memory_gb
    assert stats.memory_limit_gb == 100.0
    assert len(stats.warnings) == 0  # No warnings with high limit


@pytest.mark.unit
def test_check_memory_usage_warning():
    """Test memory check triggers warning."""
    monitor = MemoryMonitor(max_memory_gb=0.01, warning_threshold=0.0001)  # Very low limits

    stats = monitor.check_memory_usage()

    # Should have warnings due to low limits
    assert len(stats.warnings) > 0
    assert "exceeds" in stats.warnings[0].lower()


@pytest.mark.unit
def test_peak_memory_tracking():
    """Test peak memory tracking across multiple checks."""
    monitor = MemoryMonitor(max_memory_gb=100.0)

    stats1 = monitor.check_memory_usage()
    first_peak = stats1.peak_memory_gb

    # Peak should update or stay same
    stats2 = monitor.check_memory_usage()
    assert stats2.peak_memory_gb >= first_peak


@pytest.mark.unit
def test_cleanup_arrays_large():
    """Test cleanup of large arrays."""
    monitor = MemoryMonitor()

    # Create large arrays (>1M elements)
    large_array1 = np.ones((1000, 1100))  # 1.1M elements
    large_array2 = np.ones((2000, 600))  # 1.2M elements

    cleaned = monitor.cleanup_arrays(large_array1, large_array2)

    # Should clean up 2 large arrays
    assert cleaned == 2


@pytest.mark.unit
def test_cleanup_arrays_small():
    """Test cleanup skips small arrays."""
    monitor = MemoryMonitor()

    # Create small arrays (<1M elements)
    small_array = np.ones((100, 100))  # 10k elements

    cleaned = monitor.cleanup_arrays(small_array)

    # Should not clean up small arrays
    assert cleaned == 0


@pytest.mark.unit
def test_cleanup_arrays_mixed():
    """Test cleanup with mixed array sizes."""
    monitor = MemoryMonitor()

    small = np.ones((100, 100))
    large = np.ones((1500, 700))  # 1.05M elements

    cleaned = monitor.cleanup_arrays(small, large)

    assert cleaned == 1


@pytest.mark.unit
def test_get_array_memory_usage():
    """Test calculating array memory usage."""
    monitor = MemoryMonitor()

    # Create array with known size
    array = np.ones((1000, 1000), dtype=np.float64)  # 8MB

    memory_gb = monitor.get_array_memory_usage(array)

    # Should be approximately 0.0075 GB (8MB)
    assert 0.007 < memory_gb < 0.009


@pytest.mark.unit
def test_get_array_memory_usage_non_array():
    """Test memory usage returns 0 for non-arrays."""
    monitor = MemoryMonitor()

    memory_gb = monitor.get_array_memory_usage("not an array")

    assert memory_gb == 0.0


@pytest.mark.unit
def test_suggest_memory_optimization_low_usage():
    """Test suggestions with low memory usage."""
    monitor = MemoryMonitor(max_memory_gb=100.0)  # High limit

    suggestions = monitor.suggest_memory_optimization()

    # Should have no suggestions with plenty of memory
    assert len(suggestions) == 0


@pytest.mark.unit
def test_suggest_memory_optimization_high_usage():
    """Test suggestions with high memory usage."""
    monitor = MemoryMonitor(max_memory_gb=0.1)  # Very low limit

    suggestions = monitor.suggest_memory_optimization()

    # Should have suggestions due to high usage
    assert len(suggestions) > 0
    assert any("reduce" in s.lower() or "smaller" in s.lower() for s in suggestions)


# =============================================================================
# Test memory_monitored Decorator
# =============================================================================


@pytest.mark.unit
def test_memory_monitored_decorator_warnings():
    """Test memory_monitored decorator issues warnings."""

    class MockSolver:
        @memory_monitored(max_memory_gb=0.01)  # Very low limit to trigger warnings
        def compute(self):
            return "done"

    solver = MockSolver()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        solver.compute()

        # Should issue memory warnings
        assert len(w) > 0
        assert "memory" in str(w[0].message).lower()


@pytest.mark.unit
def test_memory_monitored_decorator_raise_on_exceed():
    """Test memory_monitored raises on memory exceed."""

    class MockSolver:
        @memory_monitored(max_memory_gb=0.001, raise_on_exceed=True)
        def compute(self):
            return "result"

    solver = MockSolver()

    with pytest.raises(MemoryError) as exc_info:
        solver.compute()

    assert "exceeded limit" in str(exc_info.value)


@pytest.mark.unit
def test_memory_monitored_decorator_cleanup():
    """Test memory_monitored cleanup on exit."""

    class MockSolver:
        def __init__(self):
            self._temp_arrays = []

        @memory_monitored(max_memory_gb=10.0, cleanup_on_exit=True)
        def compute(self):
            self._temp_arrays.append(np.ones((1500, 700)))
            return "done"

    solver = MockSolver()
    result = solver.compute()

    assert result == "done"


# =============================================================================
# Test MemoryProfiler Class
# =============================================================================


@pytest.mark.unit
def test_memory_profiler_initialization():
    """Test MemoryProfiler initialization."""
    profiler = MemoryProfiler()

    assert profiler.profiles == {}


@pytest.mark.unit
def test_memory_profiler_start_profiling():
    """Test starting a new profile."""
    profiler = MemoryProfiler()

    monitor = profiler.start_profiling("test_profile", max_memory_gb=5.0)

    assert isinstance(monitor, MemoryMonitor)
    assert "test_profile" in profiler.profiles
    assert profiler.profiles["test_profile"] == []


@pytest.mark.unit
def test_memory_profiler_add_profile_point():
    """Test adding measurements to profile."""
    profiler = MemoryProfiler()
    profiler.start_profiling("test")

    stats = MemoryStats(current_memory_gb=1.0, peak_memory_gb=1.2, memory_limit_gb=4.0)
    profiler.add_profile_point("test", stats)

    assert len(profiler.profiles["test"]) == 1
    assert profiler.profiles["test"][0] == stats


@pytest.mark.unit
def test_memory_profiler_get_summary():
    """Test getting profile summary statistics."""
    profiler = MemoryProfiler()
    profiler.start_profiling("test")

    # Add multiple measurements
    for i in range(5):
        stats = MemoryStats(current_memory_gb=1.0 + i * 0.1, peak_memory_gb=2.0, memory_limit_gb=4.0)
        profiler.add_profile_point("test", stats)
        time.sleep(0.01)  # Small delay for timestamp

    summary = profiler.get_profile_summary("test")

    assert summary["profile_name"] == "test"
    assert summary["num_measurements"] == 5
    assert "mean_memory_gb" in summary
    assert "max_memory_gb" in summary
    assert "std_memory_gb" in summary


@pytest.mark.unit
def test_memory_profiler_get_summary_empty():
    """Test getting summary for empty profile."""
    profiler = MemoryProfiler()

    summary = profiler.get_profile_summary("nonexistent")

    assert summary == {}


@pytest.mark.unit
def test_global_memory_profiler():
    """Test global memory profiler instance."""
    assert isinstance(global_memory_profiler, MemoryProfiler)


# =============================================================================
# Test Utility Functions
# =============================================================================


@pytest.mark.unit
def test_estimate_problem_memory_requirements_basic():
    """Test memory estimation for MFG problem."""
    estimates = estimate_problem_memory_requirements(nx=100, nt=200, num_solvers=2, dtype_size=8)

    assert "solution_arrays_gb" in estimates
    assert "temporary_arrays_gb" in estimates
    assert "solver_overhead_gb" in estimates
    assert "total_estimated_gb" in estimates
    assert "recommended_system_gb" in estimates

    # All values should be positive
    for value in estimates.values():
        assert value > 0.0


@pytest.mark.unit
def test_estimate_problem_memory_requirements_scaling():
    """Test memory estimation scales with problem size."""
    small = estimate_problem_memory_requirements(nx=50, nt=100)
    large = estimate_problem_memory_requirements(nx=200, nt=400)

    # Larger problem should require more memory
    assert large["total_estimated_gb"] > small["total_estimated_gb"]


@pytest.mark.unit
def test_estimate_problem_memory_float32_vs_float64():
    """Test memory estimation for different dtypes."""
    float64 = estimate_problem_memory_requirements(nx=100, nt=100, dtype_size=8)
    float32 = estimate_problem_memory_requirements(nx=100, nt=100, dtype_size=4)

    # Float64 should use approximately 2x memory
    assert float64["total_estimated_gb"] > float32["total_estimated_gb"]
    assert abs(float64["total_estimated_gb"] / float32["total_estimated_gb"] - 2.0) < 0.5


@pytest.mark.unit
def test_check_system_memory_availability():
    """Test checking system memory."""
    memory_info = check_system_memory_availability()

    assert "total_gb" in memory_info
    assert "available_gb" in memory_info
    assert "used_gb" in memory_info
    assert "free_gb" in memory_info
    assert "percent_used" in memory_info

    # Basic sanity checks
    assert memory_info["total_gb"] > 0.0
    assert memory_info["available_gb"] >= 0.0
    assert 0.0 <= memory_info["percent_used"] <= 100.0


@pytest.mark.unit
def test_memory_usage_report():
    """Test generating memory usage report."""
    monitor = MemoryMonitor(max_memory_gb=10.0)
    monitor.check_memory_usage()  # Update stats

    report = memory_usage_report(monitor)

    assert "Memory Usage Report" in report
    assert "Current Usage" in report
    assert "Peak Usage" in report
    assert "System Memory" in report
    assert "GB" in report


@pytest.mark.unit
def test_memory_usage_report_with_warnings():
    """Test memory report includes warnings."""
    monitor = MemoryMonitor(max_memory_gb=0.01, warning_threshold=0.0001)
    monitor.check_memory_usage()  # Trigger warnings

    report = memory_usage_report(monitor)

    assert "Warnings:" in report or "warnings" in report.lower()


@pytest.mark.unit
def test_memory_usage_report_with_suggestions():
    """Test memory report includes optimization suggestions."""
    monitor = MemoryMonitor(max_memory_gb=0.1)  # Low limit to trigger suggestions
    monitor.check_memory_usage()

    report = memory_usage_report(monitor)

    # Check for suggestions section if usage is high
    if monitor.get_current_memory_gb() > 0.08:
        assert "Optimization Suggestions" in report or "suggestion" in report.lower()


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
def test_monitor_multiple_operations():
    """Test monitoring multiple operations."""
    monitor = MemoryMonitor(max_memory_gb=10.0)

    # Perform multiple checks
    stats_list = []
    for _ in range(3):
        stats = monitor.check_memory_usage()
        stats_list.append(stats)
        time.sleep(0.01)

    # Peak should be non-decreasing
    peaks = [s.peak_memory_gb for s in stats_list]
    for i in range(1, len(peaks)):
        assert peaks[i] >= peaks[i - 1]


@pytest.mark.unit
def test_end_to_end_monitoring_workflow():
    """Test complete monitoring workflow."""
    # Initialize profiler
    profiler = MemoryProfiler()

    # Start profiling
    monitor = profiler.start_profiling("workflow_test", max_memory_gb=10.0)

    # Simulate computations
    for _i in range(3):
        # Create some arrays
        _ = np.ones((100, 100))

        # Check memory
        stats = monitor.check_memory_usage()
        profiler.add_profile_point("workflow_test", stats)

        time.sleep(0.01)

    # Get summary
    summary = profiler.get_profile_summary("workflow_test")

    assert summary["num_measurements"] == 3
    assert summary["mean_memory_gb"] > 0.0


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.unit
def test_memory_monitor_very_high_limit():
    """Test monitor with very high memory limit."""
    monitor = MemoryMonitor(max_memory_gb=1000.0)
    stats = monitor.check_memory_usage()

    assert len(stats.warnings) == 0


@pytest.mark.unit
def test_memory_monitor_zero_warning_threshold():
    """Test monitor with zero warning threshold."""
    monitor = MemoryMonitor(max_memory_gb=10.0, warning_threshold=0.0)
    stats = monitor.check_memory_usage()

    # Should always warn with 0 threshold
    assert len(stats.warnings) > 0


@pytest.mark.unit
def test_cleanup_arrays_with_lists():
    """Test cleanup handles list of arrays."""
    monitor = MemoryMonitor()

    # Create large list
    large_list = list(range(2000))

    cleaned = monitor.cleanup_arrays(large_list)

    assert cleaned == 1


@pytest.mark.unit
def test_memory_profiler_multiple_profiles():
    """Test profiler handles multiple profiles."""
    profiler = MemoryProfiler()

    profiler.start_profiling("profile1")
    profiler.start_profiling("profile2")

    stats = MemoryStats(current_memory_gb=1.0, peak_memory_gb=1.5, memory_limit_gb=4.0)

    profiler.add_profile_point("profile1", stats)
    profiler.add_profile_point("profile2", stats)

    assert len(profiler.profiles) == 2
    assert len(profiler.profiles["profile1"]) == 1
    assert len(profiler.profiles["profile2"]) == 1

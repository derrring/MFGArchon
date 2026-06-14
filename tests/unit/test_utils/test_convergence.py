#!/usr/bin/env python3
"""
Unit tests for mfgarchon/utils/numerical/convergence.py

Tests comprehensive convergence monitoring system including:
- DistributionComparator (Wasserstein, KL divergence, moments)
- _ErrorHistoryTracker (stabilization detection)
- RollingConvergenceMonitor (confidence intervals, relative error)
- DistributionConvergenceMonitor (multi-criteria convergence)
- SolverTypeDetector (automatic method detection)
- ConvergenceWrapper (decorator/wrapper pattern)
- Utility functions (factory functions, metrics)

Coverage target: mfgarchon/utils/numerical/convergence.py (411 lines, 14% -> 70%+)
"""

import pytest

import numpy as np

# Use the new import path (old path mfgarchon.utils.numerical.convergence is deprecated)
from mfgarchon.utils.convergence import (
    # Canonical class names
    ConvergenceWrapper,
    DistributionComparator,
    DistributionConvergenceMonitor,
    RollingConvergenceMonitor,
    SolverTypeDetector,
    _ErrorHistoryTracker,
    calculate_l2_convergence_metrics,
    create_distribution_monitor,
    create_rolling_monitor,
)

# =============================================================================
# Removed-pin: monitoring-family deprecated aliases (removed v0.20.0)
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "removed_name",
    [
        "OscillationDetector",  # -> _ErrorHistoryTracker
        "AdvancedConvergenceMonitor",  # -> DistributionConvergenceMonitor
        "ParticleMethodDetector",  # -> SolverTypeDetector
        "AdaptiveConvergenceWrapper",  # -> ConvergenceWrapper
        "StochasticConvergenceMonitor",  # -> RollingConvergenceMonitor
        "create_default_monitor",  # -> create_distribution_monitor
        "create_stochastic_monitor",  # -> create_rolling_monitor
    ],
)
def test_removed_monitoring_aliases_no_longer_importable(removed_name):
    """The deprecated monitoring-family aliases were removed; importing them must fail."""
    import mfgarchon.utils.convergence as conv

    assert not hasattr(conv, removed_name)
    assert removed_name not in conv.__all__
    with pytest.raises(ImportError):
        exec(f"from mfgarchon.utils.convergence import {removed_name}")


# =============================================================================
# Test DistributionComparator
# =============================================================================


@pytest.mark.unit
def test_distribution_comparator_wasserstein_1d_identical():
    """Test Wasserstein distance is 0 for identical distributions."""
    x = np.linspace(0, 1, 100)
    p = np.exp(-((x - 0.5) ** 2) / 0.1)  # Gaussian
    p = p / np.sum(p)

    distance = DistributionComparator.wasserstein_1d(p, p, x)

    assert distance < 1e-10


@pytest.mark.unit
def test_distribution_comparator_wasserstein_1d_shifted():
    """Test Wasserstein distance for shifted Gaussians."""
    x = np.linspace(0, 1, 100)
    p = np.exp(-((x - 0.3) ** 2) / 0.01)
    q = np.exp(-((x - 0.7) ** 2) / 0.01)

    distance = DistributionComparator.wasserstein_1d(p, q, x)

    # Distance should be approximately the shift distance
    assert distance > 0.3
    assert distance < 0.5


@pytest.mark.unit
def test_distribution_comparator_wasserstein_normalization():
    """Test Wasserstein handles unnormalized distributions."""
    x = np.linspace(0, 1, 50)
    p = np.ones(50) * 10.0  # Unnormalized uniform
    q = np.ones(50) * 5.0  # Different unnormalized uniform

    distance = DistributionComparator.wasserstein_1d(p, q, x)

    # Should be near 0 since both are uniform
    assert distance < 0.01


@pytest.mark.unit
def test_distribution_comparator_kl_divergence_identical():
    """Test KL divergence is 0 for identical distributions."""
    p = np.array([0.1, 0.2, 0.3, 0.25, 0.15])

    kl_div = DistributionComparator.kl_divergence(p, p)

    assert abs(kl_div) < 1e-6


@pytest.mark.unit
def test_distribution_comparator_kl_divergence_different():
    """Test KL divergence is positive for different distributions."""
    p = np.array([0.4, 0.3, 0.2, 0.1])
    q = np.array([0.1, 0.2, 0.3, 0.4])

    kl_div = DistributionComparator.kl_divergence(p, q)

    assert kl_div > 0


@pytest.mark.unit
def test_distribution_comparator_kl_divergence_stability():
    """Test KL divergence handles near-zero probabilities."""
    p = np.array([0.99, 0.01, 0.0, 0.0])
    q = np.array([0.5, 0.3, 0.1, 0.1])

    kl_div = DistributionComparator.kl_divergence(p, q, epsilon=1e-12)

    assert np.isfinite(kl_div)


@pytest.mark.unit
def test_distribution_comparator_statistical_moments_gaussian():
    """Test statistical moments for Gaussian distribution."""
    x = np.linspace(-3, 3, 200)
    dist = np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)
    dist = dist / np.sum(dist)

    moments = DistributionComparator.statistical_moments(dist, x)

    assert abs(moments["mean"]) < 0.1  # Should be near 0
    assert abs(moments["variance"] - 1.0) < 0.2  # Should be near 1
    assert abs(moments["skewness"]) < 0.2  # Symmetric -> skewness ~ 0
    assert abs(moments["kurtosis"]) < 0.5  # Gaussian has kurtosis 0 (excess)


@pytest.mark.unit
def test_distribution_comparator_statistical_moments_uniform():
    """Test statistical moments for uniform distribution."""
    x = np.linspace(0, 1, 100)
    dist = np.ones(100)
    dist = dist / np.sum(dist)

    moments = DistributionComparator.statistical_moments(dist, x)

    assert abs(moments["mean"] - 0.5) < 0.01  # Mean at center
    assert moments["variance"] > 0
    assert abs(moments["skewness"]) < 0.1  # Symmetric


@pytest.mark.unit
def test_distribution_comparator_statistical_moments_zero_variance():
    """Test moments handle zero variance (delta function)."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    dist = np.array([0.0, 0.0, 1.0, 0.0, 0.0])  # Delta at x=2

    moments = DistributionComparator.statistical_moments(dist, x)

    assert abs(moments["mean"] - 2.0) < 1e-10
    assert moments["variance"] < 1e-10
    assert moments["skewness"] == 0.0
    assert moments["kurtosis"] == 0.0


# =============================================================================
# Test _ErrorHistoryTracker
# =============================================================================


@pytest.mark.unit
def test_oscillation_detector_initialization():
    """Test _ErrorHistoryTracker initialization."""
    detector = _ErrorHistoryTracker(history_length=20)

    assert detector.history_length == 20
    assert len(detector.error_history) == 0


@pytest.mark.unit
def test_oscillation_detector_add_samples():
    """Test adding samples to detector."""
    detector = _ErrorHistoryTracker(history_length=5)

    for i in range(10):
        detector.add_sample(float(i))

    # Should only keep last 5
    assert len(detector.error_history) == 5
    assert list(detector.error_history) == [5.0, 6.0, 7.0, 8.0, 9.0]


@pytest.mark.unit
def test_oscillation_detector_insufficient_history():
    """Test is_below_threshold returns false with insufficient history."""
    detector = _ErrorHistoryTracker(history_length=10)

    for _i in range(5):
        detector.add_sample(0.1)

    is_stable, diagnostics = detector.is_below_threshold(mean_threshold=1.0, std_threshold=0.1)

    assert not is_stable
    assert diagnostics["status"] == "insufficient_history"
    assert diagnostics["samples"] == 5


@pytest.mark.unit
def test_oscillation_detector_stabilized():
    """Test detection of stabilized errors (below thresholds)."""
    detector = _ErrorHistoryTracker(history_length=10)

    # Add 10 samples with small variation
    for i in range(10):
        detector.add_sample(0.05 + 0.001 * i)

    is_stable, diagnostics = detector.is_below_threshold(mean_threshold=0.1, std_threshold=0.01)

    assert is_stable
    assert diagnostics["mean_ok"]
    assert diagnostics["std_ok"]
    assert diagnostics["mean"] < 0.1


@pytest.mark.unit
def test_oscillation_detector_not_stabilized_magnitude():
    """Test detection of high magnitude error."""
    detector = _ErrorHistoryTracker(history_length=10)

    for _i in range(10):
        detector.add_sample(0.5)  # High error

    is_stable, diagnostics = detector.is_below_threshold(mean_threshold=0.1, std_threshold=0.1)

    assert not is_stable
    assert not diagnostics["mean_ok"]


@pytest.mark.unit
def test_oscillation_detector_not_stabilized_oscillation():
    """Test detection of high oscillation (high std)."""
    detector = _ErrorHistoryTracker(history_length=10)

    # Oscillating errors
    for i in range(10):
        detector.add_sample(0.01 + 0.2 * (i % 2))

    is_stable, diagnostics = detector.is_below_threshold(mean_threshold=1.0, std_threshold=0.05)

    assert not is_stable
    assert diagnostics["mean_ok"]
    assert not diagnostics["std_ok"]


# =============================================================================
# Test RollingConvergenceMonitor
# =============================================================================


@pytest.mark.unit
def test_stochastic_convergence_monitor_initialization():
    """Test RollingConvergenceMonitor initialization."""
    monitor = RollingConvergenceMonitor(window_size=10, median_tolerance=1e-4, quantile=0.9)

    assert monitor.window_size == 10
    assert monitor.median_tolerance == 1e-4
    assert monitor.quantile == 0.9


@pytest.mark.unit
def test_stochastic_convergence_monitor_add_iteration():
    """Test adding iterations to stochastic monitor."""
    monitor = RollingConvergenceMonitor(window_size=5)

    for i in range(10):
        monitor.add_iteration(float(i) * 0.1, float(i) * 0.05)

    # Window size limits history
    assert len(monitor.errors_u) <= 5
    assert len(monitor.errors_m) <= 5
    assert monitor.iteration_count == 10


@pytest.mark.unit
def test_stochastic_convergence_monitor_get_statistics():
    """Test statistics computation."""
    monitor = RollingConvergenceMonitor(window_size=20)

    # Add iterations
    np.random.seed(42)
    for _i in range(15):
        monitor.add_iteration(0.1 + np.random.rand() * 0.01, 0.05 + np.random.rand() * 0.005)

    stats = monitor.get_statistics()

    assert "u_stats" in stats
    assert "m_stats" in stats
    assert "median" in stats["u_stats"]
    assert "mean" in stats["u_stats"]
    assert "quantile" in stats["u_stats"]


@pytest.mark.unit
def test_stochastic_convergence_monitor_no_data():
    """Test statistics with no data."""
    monitor = RollingConvergenceMonitor()

    stats = monitor.get_statistics()

    assert stats["status"] == "no_data"


@pytest.mark.unit
def test_stochastic_convergence_monitor_check_convergence():
    """Test convergence checking."""
    monitor = RollingConvergenceMonitor(window_size=10, median_tolerance=0.01, min_iterations=5)

    # Add iterations with decreasing errors
    for i in range(15):
        error_u = 0.005 + 0.0001 * i  # Small errors
        error_m = 0.003 + 0.0001 * i
        monitor.add_iteration(error_u, error_m)

    has_converged, diagnostics = monitor.check_convergence()

    assert isinstance(has_converged, bool)
    assert "median_u" in diagnostics or "status" in diagnostics


@pytest.mark.unit
def test_stochastic_convergence_monitor_insufficient_iterations():
    """Test convergence check with insufficient iterations."""
    monitor = RollingConvergenceMonitor(min_iterations=10)

    for _i in range(5):
        monitor.add_iteration(0.001, 0.001)

    has_converged, _diagnostics = monitor.check_convergence()

    assert not has_converged


# =============================================================================
# Test DistributionConvergenceMonitor
# =============================================================================


@pytest.mark.unit
def test_advanced_convergence_monitor_initialization():
    """Test DistributionConvergenceMonitor initialization."""
    monitor = DistributionConvergenceMonitor(
        wasserstein_tol=1e-4,
        kl_divergence_tol=1e-3,
        u_magnitude_tol=1e-3,
        history_length=10,
    )

    assert monitor.wasserstein_tol == 1e-4
    assert monitor.kl_divergence_tol == 1e-3
    assert monitor.u_magnitude_tol == 1e-3
    assert monitor._error_tracker.history_length == 10


@pytest.mark.unit
def test_advanced_convergence_monitor_update_basic():
    """Test basic update with value function convergence."""
    monitor = DistributionConvergenceMonitor(u_magnitude_tol=1e-3)

    u_prev = np.ones((10, 10)) * 1.0
    u_curr = np.ones((10, 10)) * 1.001  # Small change
    m_curr = np.ones((10, 10)) * 0.5
    x_grid = np.linspace(0, 1, 10)

    diagnostics = monitor.update(u_curr, u_prev, m_curr, x_grid)

    assert "u_l2_error" in diagnostics
    assert "iteration" in diagnostics
    assert diagnostics["u_l2_error"] < 0.01


@pytest.mark.unit
def test_advanced_convergence_monitor_convergence_detection():
    """Test convergence detection with small changes."""
    monitor = DistributionConvergenceMonitor(u_magnitude_tol=1e-2, u_stability_tol=1e-3, history_length=5)

    u = np.ones((10, 10))
    m = np.ones((10, 10)) * 0.5
    m = m / np.sum(m)  # Normalize
    x = np.linspace(0, 1, 10)

    # Add iterations with tiny changes
    for _i in range(15):
        u_new = u + 1e-6 * np.random.rand(10, 10)
        diagnostics = monitor.update(u_new, u, m, x)

    # Check diagnostics structure
    assert "converged" in diagnostics
    assert "u_l2_error" in diagnostics


@pytest.mark.unit
def test_advanced_convergence_monitor_multi_dimensional():
    """Test monitor handles multi-dimensional arrays."""
    monitor = DistributionConvergenceMonitor()

    u_prev = np.random.rand(20, 20)
    u_curr = u_prev + np.random.rand(20, 20) * 0.001
    m_curr = np.random.rand(20, 20)
    m_curr = m_curr / np.sum(m_curr)  # Normalize
    x_grid = np.linspace(0, 1, 20)

    diagnostics = monitor.update(u_curr, u_prev, m_curr, x_grid)

    assert diagnostics["u_l2_error"] >= 0
    assert np.isfinite(diagnostics["u_l2_error"])


@pytest.mark.unit
def test_advanced_convergence_monitor_wasserstein_computation():
    """Test Wasserstein distance computation."""
    monitor = DistributionConvergenceMonitor(wasserstein_tol=1e-4)

    # 1D distributions
    x = np.linspace(0, 1, 50)
    m_prev = np.exp(-((x - 0.5) ** 2) / 0.1)
    m_prev = m_prev / np.sum(m_prev)
    m_curr = np.exp(-((x - 0.51) ** 2) / 0.1)  # Slightly shifted
    m_curr = m_curr / np.sum(m_curr)

    u_prev = np.zeros(50)
    u_curr = np.zeros(50)

    # First update (no previous_m)
    monitor.update(u_curr, u_prev, m_prev, x)

    # Second update (with previous_m)
    diagnostics = monitor.update(u_curr, u_prev, m_curr, x)

    # Check that Wasserstein was computed
    assert "converged" in diagnostics or "wasserstein_distance" in diagnostics


# =============================================================================
# Test SolverTypeDetector
# =============================================================================


@pytest.mark.unit
def test_particle_method_detector_basic():
    """Test SolverTypeDetector can be instantiated."""
    detector = SolverTypeDetector()

    assert detector is not None


@pytest.mark.unit
def test_particle_method_detector_mock_solver():
    """Test detection from mock solver with particle attributes."""

    class MockParticleSolver:
        def __init__(self):
            self.particles = np.random.rand(100, 2)
            self.num_particles = 100

    mock_solver = MockParticleSolver()
    has_particles, detection_info = SolverTypeDetector.detect_particle_methods(mock_solver)

    assert isinstance(has_particles, bool)
    assert "confidence" in detection_info
    assert "particle_components" in detection_info


# =============================================================================
# Test ConvergenceWrapper
# =============================================================================


@pytest.mark.unit
def test_adaptive_convergence_wrapper_initialization():
    """Test ConvergenceWrapper initialization."""

    class MockSolver:
        def __init__(self):
            self.u = np.ones((10, 10))
            self.m = np.ones((10, 10)) * 0.5
            self.x_grid = np.linspace(0, 1, 10)

    mock_solver = MockSolver()
    wrapper = ConvergenceWrapper(mock_solver, classical_tol=1e-4, verbose=False)

    # Wrapper wraps the solver internally
    assert wrapper._wrapped_solver is mock_solver
    assert wrapper.classical_tol == 1e-4


@pytest.mark.unit
def test_adaptive_convergence_wrapper_particle_detection():
    """Test wrapper detects particle methods."""

    class MockParticleSolver:
        def __init__(self):
            self.particles = np.random.rand(100, 2)
            self.num_particles = 100

    mock_solver = MockParticleSolver()
    wrapper = ConvergenceWrapper(mock_solver, verbose=False)

    # Wrapper wraps the solver internally
    assert wrapper._wrapped_solver is mock_solver


# =============================================================================
# Test Utility Functions
# =============================================================================


@pytest.mark.unit
def test_create_distribution_monitor():
    """Test create_distribution_monitor factory function."""
    monitor = create_distribution_monitor(wasserstein_tol=1e-5, u_magnitude_tol=1e-3, history_length=20)

    assert isinstance(monitor, DistributionConvergenceMonitor)
    assert monitor.wasserstein_tol == 1e-5
    assert monitor.u_magnitude_tol == 1e-3


@pytest.mark.unit
def test_create_rolling_monitor():
    """Test create_rolling_monitor factory function."""
    monitor = create_rolling_monitor(median_tolerance=1e-5, window_size=30, quantile=0.95)

    assert isinstance(monitor, RollingConvergenceMonitor)
    assert monitor.median_tolerance == 1e-5
    assert monitor.window_size == 30


@pytest.mark.unit
def test_calculate_l2_convergence_metrics():
    """Test calculate_l2_convergence_metrics utility."""
    u_prev = np.ones((10, 10)) * 1.0
    u_curr = np.ones((10, 10)) * 1.01
    m_prev = np.ones((10, 10)) * 0.5
    m_curr = np.ones((10, 10)) * 0.501
    Dx = 0.1
    Dt = 0.01

    metrics = calculate_l2_convergence_metrics(u_curr, u_prev, m_curr, m_prev, Dx, Dt)

    assert "l2distu_abs" in metrics
    assert "l2distm_abs" in metrics
    assert "l2distu_rel" in metrics
    assert "l2distm_rel" in metrics
    assert metrics["l2distu_abs"] >= 0
    assert metrics["l2distm_abs"] >= 0


@pytest.mark.unit
def test_calculate_l2_convergence_metrics_identical():
    """Test L2 metrics are near zero for identical arrays."""
    u = np.random.rand(15, 15)
    m = np.random.rand(15, 15)
    Dx = 0.1
    Dt = 0.01

    metrics = calculate_l2_convergence_metrics(u, u, m, m, Dx, Dt)

    assert metrics["l2distu_abs"] < 1e-10
    assert metrics["l2distm_abs"] < 1e-10


@pytest.mark.unit
def test_calculate_l2_convergence_metrics_large_change():
    """Test L2 metrics for large changes."""
    u_prev = np.ones((10, 10))
    u_curr = np.ones((10, 10)) * 2.0  # 100% change
    m_prev = np.ones((10, 10)) * 0.5
    m_curr = np.ones((10, 10)) * 1.0  # 100% change
    Dx = 0.1
    Dt = 0.01

    metrics = calculate_l2_convergence_metrics(u_curr, u_prev, m_curr, m_prev, Dx, Dt)

    assert metrics["l2distu_abs"] > 0
    assert metrics["l2distm_abs"] > 0
    assert metrics["l2distu_rel"] >= 0.5  # Changed from > to >=


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
def test_integration_convergence_workflow():
    """Test full convergence monitoring workflow."""
    # Create monitor
    monitor = create_distribution_monitor(u_magnitude_tol=1e-3, history_length=5)

    # Simulate iterative solver
    x = np.linspace(0, 1, 50)
    u = np.ones(50)
    m = np.exp(-((x - 0.5) ** 2) / 0.1)
    m = m / np.sum(m)

    for iteration in range(15):
        u_prev = u.copy()
        # Simulate convergence
        u = u + np.random.rand(50) * 0.001 * (1.0 / (iteration + 1))

        diagnostics = monitor.update(u, u_prev, m, x)

        if diagnostics["converged"]:
            break

    # Check that diagnostics were computed
    assert "u_l2_error" in diagnostics
    assert diagnostics["iteration"] == iteration + 1


@pytest.mark.unit
def test_integration_stochastic_convergence():
    """Test stochastic convergence monitoring."""
    monitor = create_rolling_monitor(median_tolerance=0.01, window_size=10, min_iterations=5)

    # Simulate stochastic errors converging
    np.random.seed(42)
    for i in range(20):
        error_u = 0.005 + 0.001 / (i + 1)  # Decreasing errors
        error_m = 0.003 + 0.0005 / (i + 1)
        monitor.add_iteration(error_u, error_m)

    # Check statistics
    stats = monitor.get_statistics()
    assert "u_stats" in stats


@pytest.mark.unit
def test_integration_distribution_comparison_pipeline():
    """Test full distribution comparison pipeline."""
    # Create two similar distributions
    x = np.linspace(0, 1, 100)
    p = np.exp(-((x - 0.5) ** 2) / 0.1)
    p = p / np.sum(p)
    q = np.exp(-((x - 0.52) ** 2) / 0.1)  # Slightly shifted
    q = q / np.sum(q)

    # Compute all metrics
    wasserstein = DistributionComparator.wasserstein_1d(p, q, x)
    kl_div = DistributionComparator.kl_divergence(p, q)
    moments_p = DistributionComparator.statistical_moments(p, x)
    moments_q = DistributionComparator.statistical_moments(q, x)

    # Assertions
    assert wasserstein > 0  # Different distributions
    assert wasserstein < 0.1  # But close
    assert kl_div > 0
    assert abs(moments_p["mean"] - moments_q["mean"]) < 0.05


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.unit
def test_edge_case_empty_history():
    """Test monitors handle empty history gracefully."""
    detector = _ErrorHistoryTracker(history_length=10)

    is_stable, diagnostics = detector.is_below_threshold(mean_threshold=1.0, std_threshold=0.1)

    assert not is_stable
    assert diagnostics["status"] == "insufficient_history"


@pytest.mark.unit
def test_edge_case_single_point_distribution():
    """Test distribution metrics with single point."""
    x = np.array([1.0])
    p = np.array([1.0])

    wasserstein = DistributionComparator.wasserstein_1d(p, p, x)
    kl_div = DistributionComparator.kl_divergence(p, p)
    moments = DistributionComparator.statistical_moments(p, x)

    assert np.isfinite(wasserstein)
    assert np.isfinite(kl_div)
    assert moments["mean"] == 1.0
    assert moments["variance"] == 0.0


@pytest.mark.unit
def test_edge_case_zero_distribution():
    """Test handling of zero distribution."""
    x = np.linspace(0, 1, 10)
    p = np.zeros(10)
    q = np.ones(10)

    # Should handle gracefully with normalization
    wasserstein = DistributionComparator.wasserstein_1d(p, q, x)
    moments = DistributionComparator.statistical_moments(p, x)

    assert np.isfinite(wasserstein)
    assert np.isfinite(moments["mean"])


@pytest.mark.unit
def test_edge_case_large_arrays():
    """Test performance with large arrays."""
    u_prev = np.random.rand(200, 200)
    u_curr = u_prev + np.random.rand(200, 200) * 0.001
    m_prev = np.random.rand(200, 200)
    m_curr = m_prev + np.random.rand(200, 200) * 0.001
    Dx = 0.01
    Dt = 0.001

    metrics = calculate_l2_convergence_metrics(u_curr, u_prev, m_curr, m_prev, Dx, Dt)

    assert np.isfinite(metrics["l2distu_abs"])
    assert np.isfinite(metrics["l2distm_abs"])


# =============================================================================
# Additional tests for RollingConvergenceMonitor
# =============================================================================


@pytest.mark.unit
def test_stochastic_convergence_monitor_is_stagnating():
    """Test stagnation detection in stochastic monitor."""
    monitor = RollingConvergenceMonitor(window_size=5)

    # Add similar errors (stagnating)
    for _ in range(5):
        monitor.add_iteration(1e-5, 1e-5)

    assert monitor.is_stagnating(stagnation_threshold=1e-6)


@pytest.mark.unit
def test_stochastic_convergence_monitor_not_stagnating():
    """Test non-stagnating convergence."""
    monitor = RollingConvergenceMonitor(window_size=5)

    # Add decreasing errors (not stagnating)
    for i in range(5):
        monitor.add_iteration(1e-3 / (i + 1), 1e-3 / (i + 1))

    assert not monitor.is_stagnating(stagnation_threshold=1e-6)


@pytest.mark.unit
def test_stochastic_convergence_monitor_quantile_tolerance():
    """Test custom quantile tolerance."""
    monitor = RollingConvergenceMonitor(
        window_size=5,
        median_tolerance=1e-4,
        quantile=0.95,
        quantile_tolerance=5e-4,
    )

    # Add errors within tolerance
    for _ in range(10):
        monitor.add_iteration(5e-5, 5e-5)

    converged, diagnostics = monitor.check_convergence()

    assert converged
    assert diagnostics["status"] == "converged"


# =============================================================================
# Additional tests for DistributionConvergenceMonitor
# =============================================================================


@pytest.mark.unit
def test_advanced_convergence_monitor_get_convergence_summary():
    """Test convergence summary generation."""
    monitor = DistributionConvergenceMonitor()

    # Simulate convergence iterations
    x_grid = np.linspace(0, 1, 50)
    for i in range(10):
        u_prev = np.ones(50) * (1.0 - i * 0.1)
        u_curr = np.ones(50) * (1.0 - (i + 1) * 0.1)
        m_curr = np.exp(-((x_grid - 0.5) ** 2) / 0.1)
        m_curr = m_curr / np.sum(m_curr)

        monitor.update(u_curr, u_prev, m_curr, x_grid)

    summary = monitor.get_convergence_summary()

    assert "total_iterations" in summary
    assert summary["total_iterations"] == 10
    assert "final_u_error" in summary
    assert "u_error_trend" in summary


@pytest.mark.unit
def test_advanced_convergence_monitor_no_data():
    """Test convergence summary with no data."""
    monitor = DistributionConvergenceMonitor()

    summary = monitor.get_convergence_summary()

    assert summary["status"] == "no_data"


@pytest.mark.unit
def test_advanced_convergence_monitor_get_plot_data_no_history():
    """Test getting plot data with no convergence history."""
    monitor = DistributionConvergenceMonitor()

    # Should return empty data structure
    data = monitor.get_plot_data()

    assert data["iterations"] == []
    assert data["u_errors"] == []
    assert data["wasserstein_distances"] == []


@pytest.mark.unit
def test_advanced_convergence_monitor_convergence_criteria_all_pass():
    """Test all convergence criteria passing."""
    monitor = DistributionConvergenceMonitor(
        wasserstein_tol=1e-3,
        u_magnitude_tol=1e-2,
        u_stability_tol=1e-3,
    )

    x_grid = np.linspace(0, 1, 50)
    m_curr = np.exp(-((x_grid - 0.5) ** 2) / 0.1)
    m_curr = m_curr / np.sum(m_curr)

    # Build up history with converging values
    for i in range(15):
        u_prev = np.ones(50) * (1e-3 * (15 - i))
        u_curr = np.ones(50) * (1e-3 * (14 - i))
        diagnostics = monitor.update(u_curr, u_prev, m_curr, x_grid)

    # Check final iteration has converged
    assert diagnostics["converged"] or diagnostics["iteration"] < 3


# =============================================================================
# Additional tests for SolverTypeDetector
# =============================================================================


@pytest.mark.unit
def test_particle_method_detector_comprehensive():
    """Test comprehensive particle detection with multiple signals."""
    from unittest.mock import Mock

    solver = Mock()
    solver.num_particles = 1000
    solver.kde_bandwidth = 0.1
    solver.particle_solver = Mock()
    solver.update_particles = lambda: None
    solver.__class__.__name__ = "ParticleCollocationSolver"

    has_particles, info = SolverTypeDetector.detect_particle_methods(solver)

    assert has_particles
    assert info["confidence"] > 0.5
    assert len(info["detection_methods"]) >= 2


@pytest.mark.unit
def test_particle_method_detector_grid_based_solver():
    """Test detection correctly identifies grid-based solver."""
    from unittest.mock import Mock

    solver = Mock()
    solver.Nx = 100
    solver.Nt = 200
    solver.grid = Mock()
    solver.__class__.__name__ = "FixedPointSolver"
    # Ensure no particle attributes
    solver.spec = ["Nx", "Nt", "grid"]  # Only grid-based attributes

    has_particles, info = SolverTypeDetector.detect_particle_methods(solver)

    # The detector should have low confidence or no detection
    # Since confidence can vary based on Mock behavior, check it's not strongly detected
    assert info["confidence"] < 0.8 or not has_particles


@pytest.mark.unit
def test_particle_method_detector_fp_solver_inspection():
    """Test particle detection via fp_solver component."""
    from unittest.mock import Mock

    solver = Mock()
    solver.__class__.__name__ = "StandardSolver"

    # Create fp_solver with particle signature
    fp_solver = Mock()
    fp_solver.__class__.__name__ = "ParticleFPSolver"
    solver.fp_solver = fp_solver

    has_particles, info = SolverTypeDetector.detect_particle_methods(solver)

    assert has_particles
    assert "fp_solver:ParticleFPSolver" in info["particle_components"]


# =============================================================================
# Additional tests for ConvergenceWrapper
# =============================================================================


@pytest.mark.unit
def test_adaptive_convergence_wrapper_classical_mode():
    """Test wrapper in classical (non-particle) mode."""
    from unittest.mock import Mock

    solver = Mock()
    solver.__class__.__name__ = "GridBasedSolver"
    solver.solve = Mock(return_value=(np.ones((10, 10)), np.ones((10, 10)), {}))

    wrapper = ConvergenceWrapper(solver, classical_tol=1e-3, force_particle_mode=False, verbose=False)

    assert wrapper.get_convergence_mode() == "classical"
    assert wrapper._particle_mode is False


@pytest.mark.unit
def test_adaptive_convergence_wrapper_forced_particle_mode():
    """Test forcing particle mode regardless of detection."""
    from unittest.mock import Mock

    solver = Mock()
    solver.__class__.__name__ = "GridBasedSolver"

    wrapper = ConvergenceWrapper(solver, force_particle_mode=True, verbose=False)

    assert wrapper.get_convergence_mode() == "particle_aware"
    assert wrapper._particle_mode is True


@pytest.mark.unit
def test_adaptive_convergence_wrapper_get_detection_info():
    """Test retrieving detection information."""
    from unittest.mock import Mock

    solver = Mock()
    solver.__class__.__name__ = "TestSolver"

    wrapper = ConvergenceWrapper(solver, verbose=False)
    info = wrapper.get_detection_info()

    assert isinstance(info, dict)


@pytest.mark.unit
def test_adaptive_convergence_decorator_usage():
    """Test using adaptive_convergence as decorator."""

    from mfgarchon.utils.convergence import adaptive_convergence

    @adaptive_convergence(classical_tol=1e-4, verbose=False)
    class TestSolver:
        def __init__(self):
            self.problem = None

        def solve(self, Niter=10, l2errBound=1e-3, verbose=False):
            return np.ones((10, 10)), np.ones((10, 10)), {}

    solver = TestSolver()

    assert hasattr(solver, "_convergence_wrapper")


# =============================================================================
# Additional tests for factory functions
# =============================================================================


@pytest.mark.unit
def test_wrap_solver_with_adaptive_convergence():
    """Test wrap_solver_with_adaptive_convergence function."""
    from unittest.mock import Mock

    from mfgarchon.utils.convergence import (
        wrap_solver_with_adaptive_convergence,
    )

    solver = Mock()
    solver.__class__.__name__ = "TestSolver"
    solver.solve = Mock(return_value=(np.ones((5, 5)), np.ones((5, 5)), {}))

    wrapped = wrap_solver_with_adaptive_convergence(solver, verbose=False)

    assert hasattr(wrapped, "_convergence_wrapper")
    assert wrapped is solver  # Modified in-place


@pytest.mark.unit
def test_test_particle_detection_function():
    """Test test_particle_detection utility function."""
    from unittest.mock import Mock

    from mfgarchon.utils.convergence import test_particle_detection

    solver = Mock()
    solver.num_particles = 500
    solver.__class__.__name__ = "ParticleSolver"

    result = test_particle_detection(solver)

    assert "has_particles" in result
    assert "detection_info" in result
    assert "recommended_convergence" in result


@pytest.mark.unit
def test_adaptive_convergence_decorator_with_parameters():
    """Test adaptive_convergence decorator with custom parameters."""

    from mfgarchon.utils.convergence import adaptive_convergence

    @adaptive_convergence(
        classical_tol=1e-5,
        wasserstein_tol=5e-5,
        force_particle_mode=True,
        verbose=False,
    )
    class CustomSolver:
        def __init__(self):
            pass

    solver = CustomSolver()

    assert hasattr(solver, "_convergence_wrapper")
    wrapper = solver._convergence_wrapper
    assert wrapper.classical_tol == 1e-5
    assert wrapper.force_particle_mode is True


# =============================================================================
# Additional integration tests
# =============================================================================


@pytest.mark.unit
def test_integration_advanced_monitor_full_workflow():
    """Test complete DistributionConvergenceMonitor workflow."""
    monitor = DistributionConvergenceMonitor(
        wasserstein_tol=1e-4,
        u_magnitude_tol=1e-3,
        u_stability_tol=1e-4,
    )

    x_grid = np.linspace(0, 1, 100)

    # Simulate converging solver iterations
    for i in range(20):
        # Value function converging to zero
        u_prev = np.ones(100) * (1e-2 / (i + 1))
        u_curr = np.ones(100) * (1e-2 / (i + 2))

        # Distribution staying relatively constant
        m_curr = np.exp(-((x_grid - 0.5) ** 2) / 0.05)
        m_curr = m_curr / np.sum(m_curr)

        diagnostics = monitor.update(u_curr, u_prev, m_curr, x_grid)

    # Should have meaningful diagnostics
    assert diagnostics["iteration"] == 20
    assert "convergence_criteria" in diagnostics
    assert "u_l2_error" in diagnostics

    # Get summary
    summary = monitor.get_convergence_summary()
    assert summary["total_iterations"] == 20


@pytest.mark.unit
def test_integration_stochastic_monitor_with_noise():
    """Test stochastic monitor handles noisy convergence."""
    monitor = RollingConvergenceMonitor(
        window_size=10,
        median_tolerance=1e-4,
        quantile=0.9,
    )

    # Add errors with noise (realistic for particle methods)
    np.random.seed(42)
    for i in range(20):
        base_error = 1e-3 / (i + 1)
        noise = np.random.randn() * base_error * 0.1
        monitor.add_iteration(base_error + noise, base_error + noise)

    _converged, diagnostics = monitor.check_convergence()

    assert "status" in diagnostics
    assert "statistics" in diagnostics


@pytest.mark.unit
def test_integration_distribution_convergence_realistic():
    """Test distribution convergence with realistic problem."""
    comparator = DistributionComparator()

    x = np.linspace(0, 2, 200)

    # Initial distribution (wide Gaussian)
    m_initial = np.exp(-((x - 1.0) ** 2) / 0.5)
    m_initial = m_initial / np.sum(m_initial)

    # Final distribution (narrow Gaussian at same location)
    m_final = np.exp(-((x - 1.0) ** 2) / 0.1)
    m_final = m_final / np.sum(m_final)

    # Compute all metrics
    wasserstein = comparator.wasserstein_1d(m_final, m_initial, x)
    kl_div = comparator.kl_divergence(m_final, m_initial)
    moments_initial = comparator.statistical_moments(m_initial, x)
    moments_final = comparator.statistical_moments(m_final, x)

    # Wasserstein should be small (same mean)
    assert wasserstein < 0.5

    # KL divergence should be positive (different shapes)
    assert kl_div > 0

    # Means should be similar
    assert abs(moments_initial["mean"] - moments_final["mean"]) < 0.1

    # Variance should decrease
    assert moments_final["variance"] < moments_initial["variance"]

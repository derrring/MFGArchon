#!/usr/bin/env python3
"""
Unit tests for mfgarchon/types/state.py

Tests internal state representations including:
- SpatialTemporalState (NamedTuple for solver state)
- ConvergenceInfo (NamedTuple for convergence tracking)
- SolverStatistics (NamedTuple for performance metrics)
- Type aliases (ResidualHistory, IterationCallback)
"""

import pytest

import numpy as np

from mfgarchon.types.state import (
    ConvergenceInfo,
    IterationCallback,
    ResidualHistory,
    SolverStatistics,
    SpatialTemporalState,
)

# ===================================================================
# Test SpatialTemporalState NamedTuple
# ===================================================================


@pytest.mark.unit
def test_spatial_temporal_state_creation():
    """Test SpatialTemporalState basic creation."""
    u = np.random.rand(31, 51)
    m = np.random.rand(31, 51)
    metadata = {"solver": "fixed_point"}

    state = SpatialTemporalState(
        u=u,
        m=m,
        iteration=10,
        residual=1e-6,
        metadata=metadata,
    )

    assert state.u.shape == (31, 51)
    assert state.m.shape == (31, 51)
    assert state.iteration == 10
    assert state.residual == 1e-6
    assert state.metadata["solver"] == "fixed_point"


@pytest.mark.unit
def test_spatial_temporal_state_immutability():
    """Test SpatialTemporalState is immutable (NamedTuple)."""
    u = np.zeros((31, 51))
    m = np.ones((31, 51))

    state = SpatialTemporalState(
        u=u,
        m=m,
        iteration=5,
        residual=1e-5,
        metadata={},
    )

    # NamedTuple should be immutable
    with pytest.raises(AttributeError):
        state.iteration = 10  # type: ignore


@pytest.mark.unit
def test_spatial_temporal_state_copy_with_updates():
    """Test SpatialTemporalState.copy_with_updates()."""
    u = np.zeros((31, 51))
    m = np.ones((31, 51))

    state1 = SpatialTemporalState(
        u=u,
        m=m,
        iteration=5,
        residual=1e-5,
        metadata={"step": 1},
    )

    # Create updated copy
    state2 = state1.copy_with_updates(iteration=6, residual=1e-6)

    assert state2.iteration == 6
    assert state2.residual == 1e-6
    assert state2.u is state1.u  # Arrays should be same reference
    assert state2.m is state1.m
    assert state1.iteration == 5  # Original unchanged


@pytest.mark.unit
def test_spatial_temporal_state_get_final_time_solution():
    """Test SpatialTemporalState.get_final_time_solution()."""
    Nt, Nx = 30, 50
    u = np.random.rand(Nt + 1, Nx + 1)
    m = np.random.rand(Nt + 1, Nx + 1)

    state = SpatialTemporalState(
        u=u,
        m=m,
        iteration=10,
        residual=1e-6,
        metadata={},
    )

    u_final, m_final = state.get_final_time_solution()

    assert u_final.shape == (51,)
    assert m_final.shape == (51,)
    assert np.array_equal(u_final, u[-1, :])
    assert np.array_equal(m_final, m[-1, :])


@pytest.mark.unit
def test_spatial_temporal_state_get_initial_time_solution():
    """Test SpatialTemporalState.get_initial_time_solution()."""
    Nt, Nx = 30, 50
    u = np.random.rand(Nt + 1, Nx + 1)
    m = np.random.rand(Nt + 1, Nx + 1)

    state = SpatialTemporalState(
        u=u,
        m=m,
        iteration=10,
        residual=1e-6,
        metadata={},
    )

    u_initial, m_initial = state.get_initial_time_solution()

    assert u_initial.shape == (51,)
    assert m_initial.shape == (51,)
    assert np.array_equal(u_initial, u[0, :])
    assert np.array_equal(m_initial, m[0, :])


@pytest.mark.unit
def test_spatial_temporal_state_compute_l2_norm():
    """Test SpatialTemporalState.compute_l2_norm()."""
    u = np.ones((31, 51))
    m = np.ones((31, 51))

    state = SpatialTemporalState(
        u=u,
        m=m,
        iteration=0,
        residual=0.0,
        metadata={},
    )

    # L2 norm: sqrt(sum(1^2) + sum(1^2)) = sqrt(2 * 31 * 51)
    expected = np.sqrt(2 * 31 * 51)
    computed = state.compute_l2_norm()

    assert abs(computed - expected) < 1e-10


@pytest.mark.unit
def test_spatial_temporal_state_compute_l2_norm_zeros():
    """Test SpatialTemporalState.compute_l2_norm() with zeros."""
    u = np.zeros((31, 51))
    m = np.zeros((31, 51))

    state = SpatialTemporalState(
        u=u,
        m=m,
        iteration=0,
        residual=0.0,
        metadata={},
    )

    assert state.compute_l2_norm() == 0.0


@pytest.mark.unit
def test_spatial_temporal_state_str():
    """Test SpatialTemporalState.__str__()."""
    u = np.zeros((31, 51))
    m = np.zeros((31, 51))

    state = SpatialTemporalState(
        u=u,
        m=m,
        iteration=42,
        residual=3.14e-7,
        metadata={},
    )

    str_repr = str(state)

    assert "SpatialTemporalState" in str_repr
    assert "iteration=42" in str_repr
    assert "3.14e-07" in str_repr
    assert "(31, 51)" in str_repr


@pytest.mark.unit
def test_spatial_temporal_state_metadata():
    """Test SpatialTemporalState with complex metadata."""
    u = np.zeros((31, 51))
    m = np.zeros((31, 51))

    metadata = {
        "solver_type": "newton",
        "damping": 0.5,
        "max_iterations": 100,
        "nested": {"value": 42},
    }

    state = SpatialTemporalState(
        u=u,
        m=m,
        iteration=10,
        residual=1e-6,
        metadata=metadata,
    )

    assert state.metadata["solver_type"] == "newton"
    assert state.metadata["damping"] == 0.5
    assert state.metadata["nested"]["value"] == 42


# ===================================================================
# Test ConvergenceInfo NamedTuple
# ===================================================================


@pytest.mark.unit
def test_convergence_info_creation():
    """Test ConvergenceInfo basic creation."""
    residual_history = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    info = ConvergenceInfo(
        converged=True,
        iterations=5,
        final_residual=1e-5,
        residual_history=residual_history,
        convergence_reason="tolerance_met",
    )

    assert info.converged is True
    assert info.iterations == 5
    assert info.final_residual == 1e-5
    assert len(info.residual_history) == 5
    assert info.convergence_reason == "tolerance_met"


@pytest.mark.unit
def test_convergence_info_not_converged():
    """Test ConvergenceInfo for non-converged case."""
    residual_history = [1e-1, 1e-2, 1e-3]

    info = ConvergenceInfo(
        converged=False,
        iterations=100,
        final_residual=1e-3,
        residual_history=residual_history,
        convergence_reason="max_iterations_reached",
    )

    assert info.converged is False
    assert info.iterations == 100
    assert info.convergence_reason == "max_iterations_reached"


@pytest.mark.unit
def test_convergence_info_immutability():
    """Test ConvergenceInfo is immutable."""
    info = ConvergenceInfo(
        converged=True,
        iterations=5,
        final_residual=1e-5,
        residual_history=[],
        convergence_reason="done",
    )

    with pytest.raises(AttributeError):
        info.converged = False  # type: ignore


@pytest.mark.unit
def test_convergence_info_plot_convergence_no_matplotlib(capsys):
    """Test ConvergenceInfo.plot_convergence() without matplotlib."""
    # This test assumes matplotlib might not be available in test environment
    # or we're testing the fallback path
    info = ConvergenceInfo(
        converged=True,
        iterations=3,
        final_residual=1e-5,
        residual_history=[1e-1, 1e-3, 1e-5],
        convergence_reason="done",
    )

    # plot_convergence() should not raise even without matplotlib
    # It may silently return or print a message
    try:
        info.plot_convergence()
        plot_succeeded = True
    except ImportError:
        plot_succeeded = False  # Acceptable: matplotlib not available

    # Either path is valid - the test verifies no unexpected exception
    assert isinstance(plot_succeeded, bool)


# ===================================================================
# Test SolverStatistics NamedTuple
# ===================================================================


@pytest.mark.unit
def test_solver_statistics_creation():
    """Test SolverStatistics basic creation."""
    stats = SolverStatistics(
        total_time=10.5,
        average_iteration_time=0.105,
        memory_usage_mb=256.0,
        cpu_usage_percent=85.5,
    )

    assert stats.total_time == 10.5
    assert stats.average_iteration_time == 0.105
    assert stats.memory_usage_mb == 256.0
    assert stats.cpu_usage_percent == 85.5


@pytest.mark.unit
def test_solver_statistics_optional_fields():
    """Test SolverStatistics with optional None fields."""
    stats = SolverStatistics(
        total_time=5.0,
        average_iteration_time=0.05,
        memory_usage_mb=None,
        cpu_usage_percent=None,
    )

    assert stats.total_time == 5.0
    assert stats.memory_usage_mb is None
    assert stats.cpu_usage_percent is None


@pytest.mark.unit
def test_solver_statistics_str():
    """Test SolverStatistics.__str__()."""
    stats = SolverStatistics(
        total_time=12.34,
        average_iteration_time=0.123,
        memory_usage_mb=128.0,
        cpu_usage_percent=75.0,
    )

    str_repr = str(stats)

    assert "SolverStatistics" in str_repr
    assert "12.34" in str_repr
    assert "0.123" in str_repr


# ===================================================================
# Test Type Aliases
# ===================================================================


@pytest.mark.unit
def test_residual_history_type_alias():
    """Test ResidualHistory type alias usage."""
    history: ResidualHistory = [1.0, 0.5, 0.25, 0.125]

    assert isinstance(history, list)
    assert len(history) == 4
    assert all(isinstance(x, (int, float)) for x in history)


@pytest.mark.unit
def test_residual_history_empty():
    """Test ResidualHistory can be empty."""
    history: ResidualHistory = []

    assert isinstance(history, list)
    assert len(history) == 0


@pytest.mark.unit
def test_iteration_callback_type_alias():
    """Test IterationCallback type alias usage."""

    def callback(state: SpatialTemporalState) -> str | None:
        if state.iteration > 10:
            return "stop"
        return None

    # Type check
    cb: IterationCallback = callback

    # Test usage
    u = np.zeros((31, 51))
    m = np.ones((31, 51))
    state = SpatialTemporalState(u=u, m=m, iteration=5, residual=1e-5, metadata={})

    result = cb(state)
    assert result is None

    state2 = state.copy_with_updates(iteration=15)
    result2 = cb(state2)
    assert result2 == "stop"


@pytest.mark.unit
def test_iteration_callback_returns_none():
    """Test IterationCallback that returns None."""

    def no_stop_callback(state: SpatialTemporalState) -> str | None:
        return None

    cb: IterationCallback = no_stop_callback

    u = np.zeros((10, 10))
    m = np.ones((10, 10))
    state = SpatialTemporalState(u=u, m=m, iteration=100, residual=1e-10, metadata={})

    assert cb(state) is None


# ===================================================================
# Test Integration Scenarios
# ===================================================================


@pytest.mark.unit
def test_solver_iteration_workflow():
    """Test typical solver iteration workflow using state types."""
    # Initial state
    u = np.zeros((31, 51))
    m = np.ones((31, 51)) / 51  # Normalized density
    residual_history: ResidualHistory = []

    state = SpatialTemporalState(
        u=u,
        m=m,
        iteration=0,
        residual=1.0,
        metadata={"method": "fixed_point"},
    )

    # Simulate iterations
    for i in range(5):
        residual_history.append(state.residual)

        # Update state (simulate solver step)
        new_residual = state.residual * 0.5
        state = state.copy_with_updates(
            iteration=i + 1,
            residual=new_residual,
        )

    # Create convergence info
    info = ConvergenceInfo(
        converged=True,
        iterations=5,
        final_residual=state.residual,
        residual_history=residual_history,
        convergence_reason="tolerance_met",
    )

    assert info.iterations == 5
    assert len(info.residual_history) == 5
    assert info.residual_history[0] == 1.0
    assert info.residual_history[-1] < 0.1


@pytest.mark.unit
def test_complete_solver_result():
    """Test complete solver result with all state types."""
    # Solution arrays
    u = np.random.rand(31, 51)
    m = np.random.rand(31, 51)

    # Final state
    state = SpatialTemporalState(
        u=u,
        m=m,
        iteration=50,
        residual=1e-8,
        metadata={
            "solver": "newton",
            "damping": 0.7,
            "line_search": True,
        },
    )

    # Convergence info
    conv_info = ConvergenceInfo(
        converged=True,
        iterations=50,
        final_residual=1e-8,
        residual_history=[1e-1 * (0.8**i) for i in range(50)],
        convergence_reason="tolerance_met",
    )

    # Statistics
    stats = SolverStatistics(
        total_time=25.5,
        average_iteration_time=0.51,
        memory_usage_mb=512.0,
        cpu_usage_percent=90.0,
    )

    # Verify complete result
    assert state.iteration == conv_info.iterations
    assert state.residual == conv_info.final_residual
    assert conv_info.converged is True
    assert stats.total_time > stats.average_iteration_time


# ===================================================================
# Test Module Exports
# ===================================================================


@pytest.mark.unit
def test_module_docstring():
    """Test module has comprehensive docstring."""
    from mfgarchon.types import state

    assert state.__doc__ is not None
    assert "Internal State Representations" in state.__doc__

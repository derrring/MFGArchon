#!/usr/bin/env python3
"""
Unit tests for mfgarchon/hooks/base.py

Tests SolverHooks base class including:
- Hook method signatures and default implementations
- Hook method call patterns
- Return value handling (None, "stop", "restart", bool)
- Custom hook implementations
- Integration with solver workflow
"""

import pytest

import numpy as np

from mfgarchon.hooks.base import SolverHooks
from mfgarchon.types.state import SpatialTemporalState

# ===================================================================
# Test SolverHooks Class Structure
# ===================================================================


@pytest.mark.unit
def test_solver_hooks_instantiation():
    """Test SolverHooks can be instantiated."""
    hooks = SolverHooks()
    assert isinstance(hooks, SolverHooks)


@pytest.mark.unit
def test_solver_hooks_methods_are_callable():
    """Test all hook methods are callable."""
    hooks = SolverHooks()
    assert callable(hooks.on_solve_start)
    assert callable(hooks.on_iteration_start)
    assert callable(hooks.on_iteration_end)
    assert callable(hooks.on_convergence_check)
    assert callable(hooks.on_solve_end)


# ===================================================================
# Test Default Hook Implementations
# ===================================================================


@pytest.mark.unit
def test_on_solve_start_default():
    """Test on_solve_start default implementation does nothing."""
    hooks = SolverHooks()
    state = SpatialTemporalState(
        u=np.zeros((31, 51)),
        m=np.ones((31, 51)),
        iteration=0,
        residual=1e-3,
        metadata={},
    )

    # Should not raise and should return None
    result = hooks.on_solve_start(state)
    assert result is None


@pytest.mark.unit
def test_on_iteration_start_default():
    """Test on_iteration_start default implementation does nothing."""
    hooks = SolverHooks()
    state = SpatialTemporalState(
        u=np.zeros((31, 51)),
        m=np.ones((31, 51)),
        iteration=1,
        residual=1e-3,
        metadata={},
    )

    result = hooks.on_iteration_start(state)
    assert result is None


@pytest.mark.unit
def test_on_iteration_end_default():
    """Test on_iteration_end default implementation returns None."""
    hooks = SolverHooks()
    state = SpatialTemporalState(
        u=np.zeros((31, 51)),
        m=np.ones((31, 51)),
        iteration=1,
        residual=1e-3,
        metadata={},
    )

    result = hooks.on_iteration_end(state)
    assert result is None


@pytest.mark.unit
def test_on_convergence_check_default():
    """Test on_convergence_check default implementation returns None."""
    hooks = SolverHooks()
    state = SpatialTemporalState(
        u=np.zeros((31, 51)),
        m=np.ones((31, 51)),
        iteration=1,
        residual=1e-6,
        metadata={},
    )

    result = hooks.on_convergence_check(state)
    assert result is None


@pytest.mark.unit
def test_on_solve_end_default():
    """Test on_solve_end default implementation returns result unchanged."""
    hooks = SolverHooks()

    # Create a mock result object
    class MockResult:
        def __init__(self):
            self.u = np.zeros((31, 51))
            self.m = np.ones((31, 51))
            self.converged = True
            self.iterations = 10

    result = MockResult()
    returned_result = hooks.on_solve_end(result)

    assert returned_result is result  # Should return same object


# ===================================================================
# Test Custom Hook Implementations
# ===================================================================


@pytest.mark.unit
def test_custom_on_solve_start():
    """Test custom on_solve_start implementation."""

    class CustomHooks(SolverHooks):
        def __init__(self):
            self.solve_started = False

        def on_solve_start(self, initial_state):
            self.solve_started = True

    hooks = CustomHooks()
    state = SpatialTemporalState(
        u=np.zeros((31, 51)),
        m=np.ones((31, 51)),
        iteration=0,
        residual=0.0,
        metadata={},
    )

    assert hooks.solve_started is False
    hooks.on_solve_start(state)
    assert hooks.solve_started is True


@pytest.mark.unit
def test_custom_on_iteration_end_with_logging():
    """Test custom on_iteration_end with logging."""

    class LoggingHooks(SolverHooks):
        def __init__(self):
            self.iteration_log = []

        def on_iteration_end(self, state):
            self.iteration_log.append((state.iteration, state.residual))
            return None

    hooks = LoggingHooks()
    state1 = SpatialTemporalState(
        u=np.zeros((31, 51)),
        m=np.ones((31, 51)),
        iteration=1,
        residual=1e-3,
        metadata={},
    )
    state2 = SpatialTemporalState(
        u=np.zeros((31, 51)),
        m=np.ones((31, 51)),
        iteration=2,
        residual=1e-4,
        metadata={},
    )

    hooks.on_iteration_end(state1)
    hooks.on_iteration_end(state2)

    assert len(hooks.iteration_log) == 2
    assert hooks.iteration_log[0] == (1, 1e-3)
    assert hooks.iteration_log[1] == (2, 1e-4)


@pytest.mark.unit
def test_custom_on_iteration_end_with_stop_control():
    """Test custom on_iteration_end returning stop control."""

    class StopAfterNHooks(SolverHooks):
        def __init__(self, max_iterations):
            self.max_iterations = max_iterations

        def on_iteration_end(self, state):
            if state.iteration >= self.max_iterations:
                return "stop"
            return None

    hooks = StopAfterNHooks(max_iterations=5)

    state4 = SpatialTemporalState(
        u=np.zeros((31, 51)),
        m=np.ones((31, 51)),
        iteration=4,
        residual=1e-3,
        metadata={},
    )
    state5 = SpatialTemporalState(
        u=np.zeros((31, 51)),
        m=np.ones((31, 51)),
        iteration=5,
        residual=1e-3,
        metadata={},
    )

    assert hooks.on_iteration_end(state4) is None
    assert hooks.on_iteration_end(state5) == "stop"


@pytest.mark.unit
def test_custom_on_convergence_check_force_converge():
    """Test custom on_convergence_check forcing convergence."""

    class ForceConvergeHooks(SolverHooks):
        def on_convergence_check(self, state):
            # Force convergence when residual < 1e-4
            if state.residual < 1e-4:
                return True
            return None

    hooks = ForceConvergeHooks()

    state_not_converged = SpatialTemporalState(
        u=np.zeros((31, 51)),
        m=np.ones((31, 51)),
        iteration=5,
        residual=1e-3,
        metadata={},
    )
    state_converged = SpatialTemporalState(
        u=np.zeros((31, 51)),
        m=np.ones((31, 51)),
        iteration=6,
        residual=1e-5,
        metadata={},
    )

    assert hooks.on_convergence_check(state_not_converged) is None
    assert hooks.on_convergence_check(state_converged) is True


@pytest.mark.unit
def test_custom_on_solve_end_adds_metadata():
    """Test custom on_solve_end adding metadata."""

    class MetadataHooks(SolverHooks):
        def on_solve_end(self, result):
            # Add custom metadata to result
            result.custom_metadata = {"processing": "complete"}
            return result

    class MockResult:
        def __init__(self):
            self.u = np.zeros((31, 51))
            self.converged = True

    hooks = MetadataHooks()
    result = MockResult()

    assert not hasattr(result, "custom_metadata")
    returned_result = hooks.on_solve_end(result)
    assert hasattr(returned_result, "custom_metadata")
    assert returned_result.custom_metadata == {"processing": "complete"}


# ===================================================================
# Test Hook Method Return Values
# ===================================================================


@pytest.mark.unit
def test_on_iteration_end_return_values():
    """Test on_iteration_end different return values."""

    class ReturnTestHooks(SolverHooks):
        def __init__(self, return_value):
            self.return_value = return_value

        def on_iteration_end(self, state):
            return self.return_value

    state = SpatialTemporalState(
        u=np.zeros((31, 51)),
        m=np.ones((31, 51)),
        iteration=1,
        residual=1e-3,
        metadata={},
    )

    # Test None return
    hooks_none = ReturnTestHooks(None)
    assert hooks_none.on_iteration_end(state) is None

    # Test "stop" return
    hooks_stop = ReturnTestHooks("stop")
    assert hooks_stop.on_iteration_end(state) == "stop"

    # Test "restart" return
    hooks_restart = ReturnTestHooks("restart")
    assert hooks_restart.on_iteration_end(state) == "restart"


@pytest.mark.unit
def test_on_convergence_check_return_values():
    """Test on_convergence_check different return values."""

    class ConvergenceTestHooks(SolverHooks):
        def __init__(self, return_value):
            self.return_value = return_value

        def on_convergence_check(self, state):
            return self.return_value

    state = SpatialTemporalState(
        u=np.zeros((31, 51)),
        m=np.ones((31, 51)),
        iteration=1,
        residual=1e-3,
        metadata={},
    )

    # Test None return
    hooks_none = ConvergenceTestHooks(None)
    assert hooks_none.on_convergence_check(state) is None

    # Test True return (force convergence)
    hooks_true = ConvergenceTestHooks(True)
    assert hooks_true.on_convergence_check(state) is True

    # Test False return (force non-convergence)
    hooks_false = ConvergenceTestHooks(False)
    assert hooks_false.on_convergence_check(state) is False


# ===================================================================
# Test Hook Inheritance and Composition
# ===================================================================


@pytest.mark.unit
def test_multiple_hooks_inheritance():
    """Test inheriting from SolverHooks and overriding multiple methods."""

    class MultiHooks(SolverHooks):
        def __init__(self):
            self.events = []

        def on_solve_start(self, initial_state):
            self.events.append("solve_start")

        def on_iteration_start(self, state):
            self.events.append(f"iter_start_{state.iteration}")

        def on_iteration_end(self, state):
            self.events.append(f"iter_end_{state.iteration}")
            return None

        def on_solve_end(self, result):
            self.events.append("solve_end")
            return result

    hooks = MultiHooks()

    # Simulate solver workflow
    initial_state = SpatialTemporalState(
        u=np.zeros((31, 51)),
        m=np.ones((31, 51)),
        iteration=0,
        residual=0.0,
        metadata={},
    )
    hooks.on_solve_start(initial_state)

    for i in range(1, 3):
        state = SpatialTemporalState(
            u=np.zeros((31, 51)),
            m=np.ones((31, 51)),
            iteration=i,
            residual=1e-3,
            metadata={},
        )
        hooks.on_iteration_start(state)
        hooks.on_iteration_end(state)

    class MockResult:
        pass

    hooks.on_solve_end(MockResult())

    expected_events = [
        "solve_start",
        "iter_start_1",
        "iter_end_1",
        "iter_start_2",
        "iter_end_2",
        "solve_end",
    ]
    assert hooks.events == expected_events


@pytest.mark.unit
def test_partial_override():
    """Test overriding only some hook methods."""

    class PartialHooks(SolverHooks):
        def __init__(self):
            self.iteration_count = 0

        def on_iteration_end(self, state):
            self.iteration_count += 1
            return None

    hooks = PartialHooks()

    # Other hooks should still work with defaults
    state = SpatialTemporalState(
        u=np.zeros((31, 51)),
        m=np.ones((31, 51)),
        iteration=1,
        residual=1e-3,
        metadata={},
    )

    hooks.on_solve_start(state)  # Uses default (no-op)
    hooks.on_iteration_start(state)  # Uses default (no-op)
    hooks.on_iteration_end(state)  # Uses custom

    assert hooks.iteration_count == 1


# ===================================================================
# Test Module Exports
# ===================================================================


@pytest.mark.unit
def test_module_docstring():
    """Test module has docstring."""
    from mfgarchon.hooks import base

    assert base.__doc__ is not None
    assert "Base Hooks System" in base.__doc__


# ===================================================================
# Test Usage Patterns
# ===================================================================


@pytest.mark.unit
def test_progress_monitoring_pattern():
    """Test typical progress monitoring usage pattern."""

    class ProgressHooks(SolverHooks):
        def __init__(self):
            self.residuals = []

        def on_iteration_end(self, state):
            self.residuals.append(state.residual)
            return None

    hooks = ProgressHooks()

    # Simulate convergence
    residuals = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    for i, res in enumerate(residuals, start=1):
        state = SpatialTemporalState(
            u=np.zeros((31, 51)),
            m=np.ones((31, 51)),
            iteration=i,
            residual=res,
            metadata={},
        )
        hooks.on_iteration_end(state)

    assert len(hooks.residuals) == 5
    assert hooks.residuals == residuals


@pytest.mark.unit
def test_early_stopping_pattern():
    """Test typical early stopping usage pattern."""

    class EarlyStopHooks(SolverHooks):
        def __init__(self, patience=3):
            self.patience = patience
            self.best_residual = float("inf")
            self.no_improvement_count = 0

        def on_iteration_end(self, state):
            if state.residual < self.best_residual:
                self.best_residual = state.residual
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            if self.no_improvement_count >= self.patience:
                return "stop"
            return None

    hooks = EarlyStopHooks(patience=2)

    # Simulate stagnation - patience=2 means stop after 2 no-improvement iterations
    # Iteration 1: residual=1e-3, improvement, no_improvement_count=0
    # Iteration 2: residual=1e-4, improvement, no_improvement_count=0
    # Iteration 3: residual=1e-4, no improvement, no_improvement_count=1
    # Iteration 4: residual=1e-4, no improvement, no_improvement_count=2, returns "stop"
    residuals = [1e-3, 1e-4, 1e-4, 1e-4]
    results = []
    for i, res in enumerate(residuals, start=1):
        state = SpatialTemporalState(
            u=np.zeros((31, 51)),
            m=np.ones((31, 51)),
            iteration=i,
            residual=res,
            metadata={},
        )
        result = hooks.on_iteration_end(state)
        results.append(result)

    # First two iterations should not stop
    assert results[0] is None  # Iteration 1: improvement
    assert results[1] is None  # Iteration 2: improvement
    assert results[2] is None  # Iteration 3: no_improvement_count=1
    assert results[3] == "stop"  # Iteration 4: no_improvement_count=2, stop

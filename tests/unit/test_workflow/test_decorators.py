"""
Unit tests for Workflow Decorators.

Tests decorator functions for workflow steps, experiments, parameter studies,
and utility decorators for timing, caching, retry logic, and validation.
"""

import tempfile
import time

import pytest

from mfgarchon.workflow.decorators import (
    cached,
    experiment,
    log_execution,
    parameter_study,
    retry,
    timed,
    validate_inputs,
    workflow_step,
)
from mfgarchon.workflow.workflow_manager import Workflow

# ============================================================================
# Test: @workflow_step Decorator
# ============================================================================


@pytest.mark.unit
def test_workflow_step_basic():
    """Test basic workflow_step decorator."""
    wf = Workflow(name="test")

    @workflow_step(workflow=wf, name="compute")
    def my_function(**kwargs):
        return {"result": 42}

    step_id = my_function()

    assert step_id in wf.steps
    assert wf.steps[step_id].name == "compute"


@pytest.mark.unit
@pytest.mark.fast
def test_workflow_step_without_workflow():
    """Test workflow_step decorator without workflow executes directly."""

    @workflow_step(name="compute")
    def my_function(x, **kwargs):
        return x * 2

    result = my_function(5)

    assert result == 10


@pytest.mark.unit
def test_workflow_step_with_dependencies():
    """Test workflow_step with dependencies."""
    wf = Workflow(name="test")

    @workflow_step(workflow=wf, name="step1")
    def step1(**kwargs):
        return {"value": 10}

    @workflow_step(workflow=wf, name="step2", dependencies=["step1"])
    def step2(**kwargs):
        return {"value": 20}

    step1()  # Register step1
    step2_id = step2()  # Register step2

    # Dependencies are stored by step name, not ID
    assert "step1" in wf.steps[step2_id].dependencies


# ============================================================================
# Test: @experiment Decorator
# ============================================================================


@pytest.mark.unit
def test_experiment_decorator_basic():
    """Test basic experiment decorator."""
    with tempfile.TemporaryDirectory() as tmpdir:

        @experiment(name="test_exp", workspace_path=tmpdir)
        def my_experiment(experiment=None, **kwargs):
            return {"result": 100}

        result = my_experiment()

        assert result["result"] == 100


@pytest.mark.unit
def test_experiment_decorator_with_tags():
    """Test experiment decorator with tags."""
    with tempfile.TemporaryDirectory() as tmpdir:

        @experiment(
            name="tagged_exp",
            tags=["ml", "research"],
            workspace_path=tmpdir,
        )
        def my_experiment(experiment=None, **kwargs):
            assert experiment is not None
            return 42

        result = my_experiment()

        assert result == 42


@pytest.mark.unit
def test_experiment_decorator_auto_save():
    """Test experiment decorator with auto_save."""
    with tempfile.TemporaryDirectory() as tmpdir:

        @experiment(name="save_test", auto_save=True, workspace_path=tmpdir)
        def my_experiment(experiment=None, **kwargs):
            experiment.add_result("value", 123)
            return 123

        result = my_experiment()

        assert result == 123


@pytest.mark.unit
def test_experiment_decorator_handles_exception():
    """Test experiment decorator handles exceptions."""
    with tempfile.TemporaryDirectory() as tmpdir:

        @experiment(name="fail_test", workspace_path=tmpdir)
        def failing_experiment(experiment=None, **kwargs):
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_experiment()


# ============================================================================
# Test: @parameter_study Decorator
# ============================================================================


@pytest.mark.unit
def test_parameter_study_basic():
    """Test basic parameter_study decorator."""
    params = {"x": [1, 2, 3]}

    @parameter_study(parameters=params)
    def compute(x, **kwargs):
        return x * 2

    result = compute()

    assert "results" in result
    assert len(result["results"]) == 3


@pytest.mark.unit
def test_parameter_study_multiple_parameters():
    """Test parameter_study with multiple parameters."""
    params = {"a": [1, 2], "b": [10, 20]}

    @parameter_study(parameters=params)
    def compute(a, b, **kwargs):
        return a + b

    result = compute()

    assert len(result["results"]) == 4  # 2 × 2 = 4


@pytest.mark.unit
@pytest.mark.fast
def test_parameter_study_returns_analysis():
    """Test parameter_study returns analysis."""
    params = {"x": [5, 10]}

    @parameter_study(parameters=params)
    def compute(x, **kwargs):
        return {"value": x}

    result = compute()

    assert "analysis" in result
    assert "dataframe" in result


# ============================================================================
# Test: @timed Decorator
# ============================================================================


@pytest.mark.unit
def test_timed_decorator():
    """Test timed decorator measures execution time."""

    @timed
    def slow_function():
        time.sleep(0.1)
        return 42

    result = slow_function()

    assert result == 42


@pytest.mark.unit
@pytest.mark.fast
def test_timed_decorator_fast_function():
    """Test timed decorator with fast function."""

    @timed
    def fast_function():
        return 100

    result = fast_function()

    assert result == 100


# ============================================================================
# Test: @cached Decorator
# ============================================================================


@pytest.mark.unit
def test_cached_decorator_basic():
    """Test basic cached decorator."""
    call_count = {"count": 0}

    with tempfile.TemporaryDirectory() as tmpdir:

        @cached(cache_dir=tmpdir)
        def expensive_computation(x):
            call_count["count"] += 1
            return x * 2

        # First call
        result1 = expensive_computation(5)
        assert result1 == 10
        assert call_count["count"] == 1

        # Second call with same argument - should use cache
        result2 = expensive_computation(5)
        assert result2 == 10
        assert call_count["count"] == 1  # Not incremented


@pytest.mark.unit
def test_cached_decorator_different_args():
    """Test cached decorator with different arguments."""
    call_count = {"count": 0}

    with tempfile.TemporaryDirectory() as tmpdir:

        @cached(cache_dir=tmpdir)
        def compute(x):
            call_count["count"] += 1
            return x * 3

        result1 = compute(1)
        result2 = compute(2)

        assert result1 == 3
        assert result2 == 6
        assert call_count["count"] == 2  # Both computed


@pytest.mark.unit
def test_cached_decorator_with_ttl():
    """Test cached decorator with TTL."""

    @cached(ttl_seconds=1)
    def compute(x):
        return x * 2

    result1 = compute(5)
    time.sleep(1.1)
    result2 = compute(5)

    # Should recompute after TTL expires
    assert result1 == 10
    assert result2 == 10


# ============================================================================
# Test: @retry Decorator
# ============================================================================


@pytest.mark.unit
def test_retry_decorator_success_first_try():
    """Test retry decorator when function succeeds on first try."""

    @retry(max_attempts=3)
    def reliable_function():
        return 42

    result = reliable_function()

    assert result == 42


@pytest.mark.unit
def test_retry_decorator_eventual_success():
    """Test retry decorator retries until success."""
    attempt_count = {"count": 0}

    @retry(max_attempts=3, delay_seconds=0.1)
    def unreliable_function():
        attempt_count["count"] += 1
        if attempt_count["count"] < 3:
            raise ValueError("Temporary error")
        return "success"

    result = unreliable_function()

    assert result == "success"
    assert attempt_count["count"] == 3


@pytest.mark.unit
def test_retry_decorator_max_attempts_reached():
    """Test retry decorator fails after max attempts."""

    @retry(max_attempts=2, delay_seconds=0.1)
    def always_fails():
        raise RuntimeError("Persistent error")

    with pytest.raises(RuntimeError, match="Persistent error"):
        always_fails()


# ============================================================================
# Test: @validate_inputs Decorator
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_validate_inputs_basic():
    """Test basic validate_inputs decorator."""
    rules = {"x": lambda x: x > 0}

    @validate_inputs(rules)
    def positive_only(x):
        return x * 2

    result = positive_only(5)

    assert result == 10


@pytest.mark.unit
@pytest.mark.fast
def test_validate_inputs_validation_fails():
    """Test validate_inputs raises on invalid input."""
    rules = {"x": lambda x: x > 0}

    @validate_inputs(rules)
    def positive_only(x):
        return x * 2

    with pytest.raises(ValueError, match="Validation failed for parameter 'x'"):
        positive_only(-5)


@pytest.mark.unit
@pytest.mark.fast
def test_validate_inputs_multiple_parameters():
    """Test validate_inputs with multiple parameters."""
    rules = {
        "x": lambda x: x > 0,
        "y": lambda y: y < 100,
    }

    @validate_inputs(rules)
    def bounded_function(x, y):
        return x + y

    result = bounded_function(5, 10)

    assert result == 15


# ============================================================================
# Test: @log_execution Decorator
# ============================================================================


@pytest.mark.unit
def test_log_execution_basic():
    """Test basic log_execution decorator."""

    @log_execution()
    def my_function(x):
        return x * 2

    result = my_function(10)

    assert result == 20


@pytest.mark.unit
def test_log_execution_with_inputs():
    """Test log_execution with input logging."""

    @log_execution(log_inputs=True)
    def debug_function(x):
        return x * 2

    result = debug_function(21)

    assert result == 42


@pytest.mark.unit
def test_log_execution_handles_exception():
    """Test log_execution handles exceptions."""

    @log_execution()
    def failing_function():
        raise ValueError("Test error")

    with pytest.raises(ValueError, match="Test error"):
        failing_function()


# ============================================================================
# Test: Decorator Combinations
# ============================================================================


@pytest.mark.unit
def test_multiple_decorators():
    """Test combining multiple decorators."""

    @timed
    @cached()
    def expensive_function(x):
        time.sleep(0.1)
        return x * 2

    result1 = expensive_function(5)
    result2 = expensive_function(5)  # Should use cache

    assert result1 == 10
    assert result2 == 10


@pytest.mark.unit
def test_validate_and_retry():
    """Test combining validate_inputs and retry."""
    rules = {"x": lambda x: x > 0}

    @retry(max_attempts=2, delay_seconds=0.1)
    @validate_inputs(rules)
    def validated_function(x):
        return x * 2

    result = validated_function(5)

    assert result == 10


# ============================================================================
# Test: Edge Cases
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_workflow_step_uses_function_name_by_default():
    """Test workflow_step uses function name when name not provided."""
    wf = Workflow(name="test")

    @workflow_step(workflow=wf)
    def my_named_function(**kwargs):
        return 42

    step_id = my_named_function()

    assert "my_named_function" in wf.steps[step_id].name


@pytest.mark.unit
def test_cached_with_unhashable_args():
    """Test cached decorator handles different argument types."""

    @cached()
    def compute(x, y):
        return x + y

    result1 = compute(1, 2)
    result2 = compute(1, 2)

    assert result1 == 3
    assert result2 == 3


@pytest.mark.unit
def test_retry_with_specific_exception():
    """Test retry can handle specific exception types."""
    attempt_count = {"count": 0}

    @retry(max_attempts=3, delay_seconds=0.1)
    def function_with_specific_error():
        attempt_count["count"] += 1
        if attempt_count["count"] < 2:
            raise ValueError("Retry this")
        return "success"

    result = function_with_specific_error()

    assert result == "success"
    assert attempt_count["count"] == 2

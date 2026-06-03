"""
Unit tests for Parameter Sweep.

Tests parameter space exploration including combination generation,
execution modes, result collection, and error handling.
"""

import pickle
import tempfile
from pathlib import Path

import pytest

from mfgarchon.workflow.parameter_sweep import ParameterSweep, SweepConfiguration

# ============================================================================
# Test: SweepConfiguration
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_sweep_config_default_values():
    """Test SweepConfiguration default values."""
    params = {"x": [1, 2]}
    config = SweepConfiguration(parameters=params)

    assert config.parameters == params
    assert config.execution_mode == "sequential"
    assert config.max_workers is not None
    assert config.save_intermediate is True
    assert config.retry_failed is True
    assert config.max_retries == 3


@pytest.mark.unit
@pytest.mark.fast
def test_sweep_config_custom_values():
    """Test SweepConfiguration with custom values."""
    params = {"x": [1, 2]}
    config = SweepConfiguration(
        parameters=params,
        execution_mode="parallel_threads",
        max_workers=4,
        batch_size=10,
        save_intermediate=False,
        retry_failed=False,
        max_retries=5,
    )

    assert config.execution_mode == "parallel_threads"
    assert config.max_workers == 4
    assert config.batch_size == 10
    assert config.save_intermediate is False
    assert config.retry_failed is False
    assert config.max_retries == 5


@pytest.mark.unit
@pytest.mark.fast
def test_sweep_config_creates_output_dir():
    """Test SweepConfiguration creates output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        params = {"x": [1, 2]}
        config = SweepConfiguration(parameters=params, output_dir=Path(tmpdir) / "sweeps")

        assert config.output_dir.exists()
        assert config.output_dir.is_dir()


# ============================================================================
# Test: ParameterSweep Initialization
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_parameter_sweep_creation():
    """Test basic parameter sweep creation."""
    params = {"x": [1, 2, 3]}
    sweep = ParameterSweep(params)

    assert sweep.parameters == params
    assert sweep.total_combinations == 3
    assert len(sweep.parameter_combinations) == 3


@pytest.mark.unit
@pytest.mark.fast
def test_parameter_sweep_with_config():
    """Test parameter sweep with custom configuration."""
    params = {"x": [1, 2]}
    config = SweepConfiguration(parameters=params, execution_mode="parallel_threads")

    sweep = ParameterSweep(params, config=config)

    assert sweep.config == config
    assert sweep.config.execution_mode == "parallel_threads"


@pytest.mark.unit
@pytest.mark.fast
def test_parameter_sweep_initializes_empty_results():
    """Test parameter sweep initializes with empty results."""
    params = {"x": [1, 2]}
    sweep = ParameterSweep(params)

    assert len(sweep.results) == 0
    assert len(sweep.failed_runs) == 0
    assert sweep.completed_runs == 0


# ============================================================================
# Test: Parameter Combination Generation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_generate_combinations_single_parameter():
    """Test combination generation with single parameter."""
    params = {"x": [1, 2, 3]}
    sweep = ParameterSweep(params)

    combinations = sweep.parameter_combinations

    assert len(combinations) == 3
    assert {"x": 1} in combinations
    assert {"x": 2} in combinations
    assert {"x": 3} in combinations


@pytest.mark.unit
@pytest.mark.fast
def test_generate_combinations_multiple_parameters():
    """Test combination generation with multiple parameters."""
    params = {"x": [1, 2], "y": [10, 20]}
    sweep = ParameterSweep(params)

    combinations = sweep.parameter_combinations

    assert len(combinations) == 4  # 2 × 2 = 4
    assert {"x": 1, "y": 10} in combinations
    assert {"x": 1, "y": 20} in combinations
    assert {"x": 2, "y": 10} in combinations
    assert {"x": 2, "y": 20} in combinations


@pytest.mark.unit
@pytest.mark.fast
def test_generate_combinations_three_parameters():
    """Test combination generation with three parameters."""
    params = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}
    sweep = ParameterSweep(params)

    combinations = sweep.parameter_combinations

    assert len(combinations) == 8  # 2 × 2 × 2 = 8


@pytest.mark.unit
@pytest.mark.fast
def test_generate_combinations_single_value():
    """Test combination generation with single-value parameter."""
    params = {"x": [1], "y": [2, 3]}
    sweep = ParameterSweep(params)

    combinations = sweep.parameter_combinations

    assert len(combinations) == 2  # 1 × 2 = 2


@pytest.mark.unit
@pytest.mark.fast
def test_generate_combinations_scalar_converted_to_list():
    """Test scalar parameter values converted to lists."""
    params = {"x": 5}  # Scalar, not list
    sweep = ParameterSweep(params)

    combinations = sweep.parameter_combinations

    assert len(combinations) == 1
    assert combinations[0] == {"x": 5}


@pytest.mark.unit
@pytest.mark.fast
def test_total_combinations_correct():
    """Test total_combinations attribute is correct."""
    params = {"a": [1, 2, 3], "b": [4, 5]}
    sweep = ParameterSweep(params)

    assert sweep.total_combinations == 6  # 3 × 2 = 6


# ============================================================================
# Test: Sequential Execution
# ============================================================================


@pytest.mark.unit
def test_execute_sequential_single_parameter():
    """Test sequential execution with single parameter."""
    params = {"x": [1, 2, 3]}
    sweep = ParameterSweep(params)

    def compute(x):
        return {"result": x * 2}

    results = sweep.execute(compute)

    assert len(results) == 3
    assert results[0]["result"] == 2
    assert results[1]["result"] == 4
    assert results[2]["result"] == 6


@pytest.mark.unit
def test_execute_sequential_multiple_parameters():
    """Test sequential execution with multiple parameters."""
    params = {"x": [1, 2], "y": [10, 20]}
    sweep = ParameterSweep(params)

    def compute(x, y):
        return {"sum": x + y}

    results = sweep.execute(compute)

    assert len(results) == 4


@pytest.mark.unit
def test_execute_sequential_all_combinations():
    """Test sequential execution runs all combinations."""
    params = {"a": [1, 2], "b": [3, 4]}
    sweep = ParameterSweep(params)

    executed_params = []

    def compute(a, b):
        executed_params.append((a, b))
        return {"value": a * b}

    sweep.execute(compute)

    assert len(executed_params) == 4
    assert (1, 3) in executed_params
    assert (1, 4) in executed_params
    assert (2, 3) in executed_params
    assert (2, 4) in executed_params


@pytest.mark.unit
def test_execute_tracks_completed_runs():
    """Test execution tracks completed runs."""
    params = {"x": [1, 2, 3]}
    sweep = ParameterSweep(params)

    def compute(x):
        return x

    sweep.execute(compute)

    assert sweep.completed_runs == 3


@pytest.mark.unit
def test_execute_sets_execution_times():
    """Test execution sets start and end times."""
    params = {"x": [1, 2]}
    sweep = ParameterSweep(params)

    def compute(x):
        return x

    sweep.execute(compute)

    assert sweep.start_time is not None
    assert sweep.end_time is not None
    assert sweep.end_time >= sweep.start_time


# ============================================================================
# Test: Result Collection
# ============================================================================


@pytest.mark.unit
def test_results_include_parameters():
    """Test results include parameter values."""
    params = {"x": [1, 2]}
    sweep = ParameterSweep(params)

    def compute(x):
        return {"result": x * 2}

    results = sweep.execute(compute)

    assert "parameters" in results[0]
    assert results[0]["parameters"]["x"] in [1, 2]


@pytest.mark.unit
def test_results_include_outputs():
    """Test results include function outputs."""
    params = {"x": [1, 2]}
    sweep = ParameterSweep(params)

    def compute(x):
        return {"value": x * 3}

    results = sweep.execute(compute)

    # Result dict is flattened into top level, not nested under 'outputs'
    assert "value" in results[0]
    assert results[0]["value"] == 3  # x=1, so value=1*3=3


@pytest.mark.unit
def test_results_include_execution_info():
    """Test results include execution metadata."""
    params = {"x": [1]}
    sweep = ParameterSweep(params)

    def compute(x):
        return x

    results = sweep.execute(compute)

    assert "run_id" in results[0]
    assert "execution_time" in results[0]


# ============================================================================
# Test: Error Handling
# ============================================================================


@pytest.mark.unit
def test_execute_handles_function_failure():
    """Test execution handles individual function failures."""
    params = {"x": [1, 2, 3]}
    sweep = ParameterSweep(params)

    def compute(x):
        if x == 2:
            raise ValueError("Intentional failure")
        return {"result": x}

    results = sweep.execute(compute)

    # Should still complete other runs
    assert len(results) + len(sweep.failed_runs) == 3


@pytest.mark.unit
def test_execute_records_failed_runs():
    """Test execution records failed runs."""
    params = {"x": [1, 2]}
    sweep = ParameterSweep(params)

    def compute(x):
        if x == 2:
            raise RuntimeError("Error")
        return x

    sweep.execute(compute)

    assert len(sweep.failed_runs) >= 1


@pytest.mark.unit
def test_execute_continues_after_failure():
    """Test execution continues after single failure."""
    params = {"x": [1, 2, 3]}
    sweep = ParameterSweep(params)

    def compute(x):
        if x == 2:
            raise ValueError("Fail")
        return {"value": x}

    results = sweep.execute(compute)

    # Should have results from x=1 and x=3
    assert len(results) == 2


# ============================================================================
# Test: Edge Cases
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_empty_parameter_space():
    """Test parameter sweep with empty parameter dict."""
    params = {}
    sweep = ParameterSweep(params)

    assert sweep.total_combinations == 1  # One empty combination
    assert sweep.parameter_combinations == [{}]


@pytest.mark.unit
def test_single_combination():
    """Test parameter sweep with single combination."""
    params = {"x": [1]}
    sweep = ParameterSweep(params)

    def compute(x):
        return {"result": x * 2}

    results = sweep.execute(compute)

    assert len(results) == 1
    # Result dict is flattened into top level
    assert results[0]["result"] == 2  # Flattened 'result' key from function return


@pytest.mark.unit
def test_large_parameter_space():
    """Test parameter sweep with many combinations."""
    params = {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "c": [9, 10]}
    sweep = ParameterSweep(params)

    assert sweep.total_combinations == 32  # 4 × 4 × 2 = 32


@pytest.mark.unit
def test_function_with_no_return():
    """Test execution handles functions with no return value."""
    params = {"x": [1, 2]}
    sweep = ParameterSweep(params)

    def compute(x):
        pass  # No return

    results = sweep.execute(compute)

    assert len(results) == 2


@pytest.mark.unit
def test_function_returning_none():
    """Test execution handles functions returning None."""
    params = {"x": [1, 2]}
    sweep = ParameterSweep(params)

    def compute(x):
        return None

    results = sweep.execute(compute)

    assert len(results) == 2


# ============================================================================
# Test: Parameter Types
# ============================================================================


@pytest.mark.unit
def test_integer_parameters():
    """Test parameter sweep with integer parameters."""
    params = {"count": [1, 2, 3]}
    sweep = ParameterSweep(params)

    def compute(count):
        return {"value": count}

    results = sweep.execute(compute)

    assert len(results) == 3


@pytest.mark.unit
def test_float_parameters():
    """Test parameter sweep with float parameters."""
    params = {"sigma": [0.1, 0.2, 0.3]}
    sweep = ParameterSweep(params)

    def compute(sigma):
        return {"value": sigma * 2}

    results = sweep.execute(compute)

    assert len(results) == 3


@pytest.mark.unit
def test_string_parameters():
    """Test parameter sweep with string parameters."""
    params = {"method": ["newton", "picard"]}
    sweep = ParameterSweep(params)

    def compute(method):
        return {"method_used": method}

    results = sweep.execute(compute)

    assert len(results) == 2


@pytest.mark.unit
def test_mixed_parameter_types():
    """Test parameter sweep with mixed parameter types."""
    params = {"n": [10, 20], "alpha": [0.1, 0.2], "method": ["A", "B"]}
    sweep = ParameterSweep(params)

    assert sweep.total_combinations == 8  # 2 × 2 × 2 = 8


# ============================================================================
# Test: Execution Modes
# ============================================================================


@pytest.mark.unit
def test_execution_mode_sequential():
    """Test explicit sequential execution mode."""
    params = {"x": [1, 2, 3]}
    config = SweepConfiguration(parameters=params, execution_mode="sequential")
    sweep = ParameterSweep(params, config=config)

    def compute(x):
        return {"result": x}

    results = sweep.execute(compute)

    assert len(results) == 3


@pytest.mark.unit
def test_invalid_execution_mode_raises():
    """Test invalid execution mode raises ValueError."""
    params = {"x": [1, 2]}
    config = SweepConfiguration(parameters=params, execution_mode="invalid_mode")
    sweep = ParameterSweep(params, config=config)

    def compute(x):
        return x

    with pytest.raises(ValueError, match="Unknown execution mode"):
        sweep.execute(compute)


# ============================================================================
# Test: Results Storage
# ============================================================================


@pytest.mark.unit
def test_results_stored_in_sweep_object():
    """Test results are stored in sweep object."""
    params = {"x": [1, 2]}
    sweep = ParameterSweep(params)

    def compute(x):
        return {"value": x}

    results = sweep.execute(compute)

    assert sweep.results == results
    assert len(sweep.results) == 2


@pytest.mark.unit
def test_results_persist_after_execution():
    """Test results persist in sweep object after execution."""
    params = {"x": [1, 2, 3]}
    sweep = ParameterSweep(params)

    def compute(x):
        return x

    sweep.execute(compute)

    assert len(sweep.results) == 3
    # Check we can access results multiple times
    assert len(sweep.results) == 3


# ============================================================================
# Test: Issue #1080 — picklability for ProcessPoolExecutor under macOS spawn
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_parameter_sweep_is_picklable_issue_1080():
    """ParameterSweep must pickle (for ProcessPoolExecutor under macOS/Windows 'spawn')
    despite its logger holding a non-picklable threading.Lock. __getstate__ drops the
    logger; __setstate__ re-initializes it in the worker."""
    sweep = ParameterSweep({"x": [1, 2], "y": [3, 4]})
    restored = pickle.loads(pickle.dumps(sweep))
    assert restored.total_combinations == sweep.total_combinations
    assert restored.parameters == sweep.parameters
    assert restored.logger is not None
    restored.logger.info("worker logger re-initialized OK")  # must not raise


@pytest.mark.unit
@pytest.mark.fast
def test_parallel_processes_fails_loud_on_unpicklable_function_issue_1080():
    """A non-picklable sweep function (lambda/closure) must raise an actionable ValueError at
    the call site, not a cryptic deep-pickle TypeError inside a worker (macOS/Windows spawn)."""
    sweep = ParameterSweep({"x": [1, 2]})
    with pytest.raises(ValueError, match="not picklable"):
        sweep._execute_parallel_processes(lambda x: x)

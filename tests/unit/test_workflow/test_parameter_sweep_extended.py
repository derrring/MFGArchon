"""
Extended unit tests for Parameter Sweep - Additional coverage.

Tests parallel execution, result analysis, DataFrame export, file I/O,
and factory functions to increase coverage from 70% to 85%.
"""

import json
import pickle
import tempfile
import time
from pathlib import Path

import pytest

pl = pytest.importorskip("polars")

import numpy as np  # noqa: E402

from mfgarchon.workflow.parameter_sweep import (  # noqa: E402
    ParameterSweep,
    SweepConfiguration,
    create_adaptive_sweep,
    create_grid_sweep,
    create_random_sweep,
)

# ============================================================================
# Test: Parallel Execution - Thread Pool
# ============================================================================


@pytest.mark.unit
def test_execute_parallel_threads_basic():
    """Test parallel execution using thread pool."""
    params = {"x": [1, 2, 3, 4]}
    config = SweepConfiguration(
        parameters=params, execution_mode="parallel_threads", max_workers=2, save_intermediate=False
    )
    sweep = ParameterSweep(params, config=config)

    def compute(x):
        return {"result": x * 2}

    results = sweep.execute(compute)

    assert len(results) == 4
    result_values = [r["result"] for r in results]
    assert set(result_values) == {2, 4, 6, 8}


@pytest.mark.unit
def test_execute_parallel_threads_correctness():
    """Test parallel threads produce same results as sequential."""
    params = {"a": [1, 2, 3], "b": [4, 5]}

    # Sequential execution
    config_seq = SweepConfiguration(parameters=params, execution_mode="sequential", save_intermediate=False)
    sweep_seq = ParameterSweep(params, config=config_seq)

    def compute(a, b):
        return {"sum": a + b, "product": a * b}

    results_seq = sweep_seq.execute(compute)

    # Parallel execution
    config_par = SweepConfiguration(
        parameters=params, execution_mode="parallel_threads", max_workers=2, save_intermediate=False
    )
    sweep_par = ParameterSweep(params, config=config_par)
    results_par = sweep_par.execute(compute)

    # Sort results by run_id for comparison
    results_seq_sorted = sorted(results_seq, key=lambda x: x["run_id"])
    results_par_sorted = sorted(results_par, key=lambda x: x["run_id"])

    assert len(results_seq_sorted) == len(results_par_sorted)

    for res_seq, res_par in zip(results_seq_sorted, results_par_sorted, strict=True):
        assert res_seq["parameters"] == res_par["parameters"]
        assert res_seq["sum"] == res_par["sum"]
        assert res_seq["product"] == res_par["product"]


@pytest.mark.unit
def test_execute_parallel_threads_handles_errors():
    """Test parallel threads handle function errors gracefully."""
    params = {"x": [1, 2, 3, 4]}
    config = SweepConfiguration(
        parameters=params, execution_mode="parallel_threads", max_workers=2, save_intermediate=False
    )
    sweep = ParameterSweep(params, config=config)

    def compute(x):
        if x == 2:
            raise ValueError("Intentional error")
        return {"result": x}

    results = sweep.execute(compute)

    # Should have 3 successful and 1 failed
    assert len(results) + len(sweep.failed_runs) == 4
    assert len(sweep.failed_runs) >= 1


@pytest.mark.unit
def test_execute_parallel_threads_with_timeout():
    """Test parallel threads with timeout configuration."""
    params = {"x": [1, 2]}
    config = SweepConfiguration(
        parameters=params,
        execution_mode="parallel_threads",
        max_workers=2,
        timeout_per_run=10.0,
        save_intermediate=False,
    )
    sweep = ParameterSweep(params, config=config)

    def compute(x):
        return {"value": x}

    results = sweep.execute(compute)

    assert len(results) == 2


# ============================================================================
# Test: Parallel Execution - Process Pool
# ============================================================================


# Define module-level function for pickling
def _test_compute_process(x):
    """Module-level function for process pool tests."""
    return {"result": x * 3}


@pytest.mark.unit
def test_execute_parallel_processes_basic():
    """Test parallel execution using process pool."""
    params = {"x": [1, 2, 3]}
    config = SweepConfiguration(
        parameters=params, execution_mode="parallel_processes", max_workers=2, save_intermediate=False
    )
    sweep = ParameterSweep(params, config=config)

    results = sweep.execute(_test_compute_process)

    assert len(results) == 3
    result_values = [r["result"] for r in results]
    assert set(result_values) == {3, 6, 9}


@pytest.mark.unit
def test_execute_parallel_processes_correctness():
    """Test parallel processes produce same results as sequential."""
    params = {"x": [1, 2, 3, 4]}

    # Sequential execution
    config_seq = SweepConfiguration(parameters=params, execution_mode="sequential", save_intermediate=False)
    sweep_seq = ParameterSweep(params, config=config_seq)
    results_seq = sweep_seq.execute(_test_compute_process)

    # Parallel execution
    config_par = SweepConfiguration(
        parameters=params, execution_mode="parallel_processes", max_workers=2, save_intermediate=False
    )
    sweep_par = ParameterSweep(params, config=config_par)
    results_par = sweep_par.execute(_test_compute_process)

    # Sort and compare
    results_seq_sorted = sorted(results_seq, key=lambda x: x["run_id"])
    results_par_sorted = sorted(results_par, key=lambda x: x["run_id"])

    for res_seq, res_par in zip(results_seq_sorted, results_par_sorted, strict=True):
        assert res_seq["result"] == res_par["result"]


def _test_compute_error(x):
    """Module-level function that raises error for x=2."""
    if x == 2:
        raise RuntimeError("Process error")
    return {"value": x}


@pytest.mark.unit
def test_execute_parallel_processes_handles_errors():
    """Test parallel processes handle function errors gracefully."""
    params = {"x": [1, 2, 3]}
    config = SweepConfiguration(
        parameters=params,
        execution_mode="parallel_processes",
        max_workers=2,
        retry_failed=False,
        save_intermediate=False,
    )
    sweep = ParameterSweep(params, config=config)

    results = sweep.execute(_test_compute_error)

    # Should have 2 successful and 1 failed
    assert len(results) + len(sweep.failed_runs) == 3


@pytest.mark.unit
def test_parallel_execution_worker_count():
    """Test worker count configuration for parallel execution."""
    params = {"x": [1, 2, 3, 4, 5]}
    config = SweepConfiguration(
        parameters=params, execution_mode="parallel_threads", max_workers=3, save_intermediate=False
    )
    sweep = ParameterSweep(params, config=config)

    def compute(x):
        return {"result": x}

    results = sweep.execute(compute)

    assert len(results) == 5
    assert sweep.config.max_workers == 3


# ============================================================================
# Test: Result Analysis
# ============================================================================


@pytest.mark.unit
def test_analyze_results_summary():
    """Test result analysis produces correct summary statistics."""
    params = {"x": [1, 2, 3]}
    sweep = ParameterSweep(params)

    def compute(x):
        time.sleep(0.01)  # Small delay for timing
        return {"value": x * 2}

    sweep.execute(compute)

    analysis = sweep.analyze_results()

    assert "summary" in analysis
    assert analysis["summary"]["total_runs"] == 3
    assert analysis["summary"]["successful_runs"] == 3
    assert analysis["summary"]["failed_runs"] == 0
    assert analysis["summary"]["success_rate"] == 1.0
    assert analysis["summary"]["total_time"] > 0
    assert analysis["summary"]["average_time_per_run"] > 0


@pytest.mark.unit
def test_analyze_results_no_results():
    """Test result analysis with no results."""
    params = {"x": [1, 2]}
    sweep = ParameterSweep(params)

    # Don't execute, so no results
    analysis = sweep.analyze_results()

    assert "error" in analysis
    assert analysis["error"] == "No successful results to analyze"


@pytest.mark.unit
def test_analyze_results_parameter_effects():
    """Test parameter effects analysis."""
    params = {"method": ["A", "B"], "size": [10, 20]}
    sweep = ParameterSweep(params)

    def compute(method, size):
        # Different output based on parameters
        value = 100 if method == "A" else 200
        value += size
        return {"output": value}

    sweep.execute(compute)

    analysis = sweep.analyze_results()

    assert "parameter_analysis" in analysis
    # Parameter analysis should identify effects of method and size


@pytest.mark.unit
def test_analyze_results_performance_metrics():
    """Test performance metrics in analysis."""
    params = {"x": [1, 2, 3]}
    sweep = ParameterSweep(params)

    def compute(x):
        return {"value": x, "converged": x > 1, "iterations": x * 10}

    sweep.execute(compute)

    analysis = sweep.analyze_results()

    assert "performance_analysis" in analysis
    perf = analysis["performance_analysis"]

    assert "execution_time" in perf
    assert "convergence_rate" in perf
    assert "iterations" in perf


@pytest.mark.unit
def test_analyze_results_with_failures():
    """Test result analysis with some failed runs."""
    params = {"x": [1, 2, 3, 4]}
    sweep = ParameterSweep(params)

    def compute(x):
        if x == 2 or x == 3:
            raise ValueError("Fail")
        return {"result": x}

    sweep.execute(compute)

    analysis = sweep.analyze_results()

    assert analysis["summary"]["successful_runs"] == 2
    assert analysis["summary"]["failed_runs"] == 2
    assert analysis["summary"]["success_rate"] == 0.5


# ============================================================================
# Test: DataFrame Export
# ============================================================================


@pytest.mark.unit
def test_to_dataframe_basic():
    """Test conversion to DataFrame."""
    params = {"x": [1, 2, 3]}
    sweep = ParameterSweep(params)

    def compute(x):
        return {"my_output": x * 2}

    sweep.execute(compute)

    df = sweep.to_dataframe()

    assert df is not None
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 3
    assert "x" in df.columns  # Parameter column
    assert "my_output" in df.columns  # Result column (flattened from returned dict)
    assert "run_id" in df.columns
    assert "execution_time" in df.columns


@pytest.mark.unit
def test_to_dataframe_no_results():
    """Test to_dataframe with no results returns None."""
    params = {"x": [1, 2]}
    sweep = ParameterSweep(params)

    # No execution
    df = sweep.to_dataframe()

    assert df is None


@pytest.mark.unit
def test_to_dataframe_multiple_parameters():
    """Test DataFrame contains all parameter columns."""
    params = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}
    sweep = ParameterSweep(params)

    def compute(a, b, c):
        return {"sum": a + b + c}

    sweep.execute(compute)

    df = sweep.to_dataframe()

    assert "a" in df.columns
    assert "b" in df.columns
    assert "c" in df.columns
    assert "sum" in df.columns


@pytest.mark.unit
def test_to_dataframe_flattens_results():
    """Test DataFrame flattens nested result dictionaries."""
    params = {"x": [1, 2]}
    sweep = ParameterSweep(params)

    def compute(x):
        return {"metric_a": x * 2, "metric_b": x * 3, "metadata": x * 4}

    sweep.execute(compute)

    df = sweep.to_dataframe()

    # All result keys should be flattened to columns
    assert "metric_a" in df.columns
    assert "metric_b" in df.columns
    assert "metadata" in df.columns


@pytest.mark.unit
def test_save_to_csv_basic():
    """Test saving results to CSV file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        params = {"x": [1, 2, 3]}
        config = SweepConfiguration(parameters=params, output_dir=Path(tmpdir))
        sweep = ParameterSweep(params, config=config)

        def compute(x):
            return {"result": x * 2}

        sweep.execute(compute)

        csv_path = sweep.save_to_csv("test_results.csv")

        assert Path(csv_path).exists()
        assert Path(csv_path).suffix == ".csv"


@pytest.mark.unit
def test_save_to_csv_content():
    """Test CSV file contains correct data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        params = {"x": [1, 2]}
        config = SweepConfiguration(parameters=params, output_dir=Path(tmpdir))
        sweep = ParameterSweep(params, config=config)

        def compute(x):
            return {"value": x * 10}

        sweep.execute(compute)

        csv_path = sweep.save_to_csv("output.csv")

        # Read CSV and verify content
        df_loaded = pl.read_csv(csv_path)

        assert len(df_loaded) == 2
        assert "x" in df_loaded.columns
        assert "value" in df_loaded.columns
        assert set(df_loaded["value"]) == {10, 20}


@pytest.mark.unit
def test_save_to_csv_no_results_raises():
    """Test save_to_csv raises error with no results."""
    params = {"x": [1, 2]}
    sweep = ParameterSweep(params)

    # No execution
    with pytest.raises(ValueError, match="No results to save"):
        sweep.save_to_csv()


# ============================================================================
# Test: File I/O and Persistence
# ============================================================================


@pytest.mark.unit
def test_save_results_creates_files():
    """Test _save_results creates pickle and JSON files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        params = {"x": [1, 2]}
        config = SweepConfiguration(parameters=params, output_dir=Path(tmpdir), save_intermediate=True)
        sweep = ParameterSweep(params, config=config)

        def compute(x):
            return {"result": x}

        sweep.execute(compute)

        # Check files exist
        pkl_files = list(Path(tmpdir).glob("sweep_results_*.pkl"))
        json_files = list(Path(tmpdir).glob("sweep_summary_*.json"))

        assert len(pkl_files) >= 1
        assert len(json_files) >= 1


@pytest.mark.unit
def test_save_results_pickle_content():
    """Test pickle file contains complete sweep data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        params = {"x": [1, 2, 3]}
        config = SweepConfiguration(parameters=params, output_dir=Path(tmpdir), save_intermediate=True)
        sweep = ParameterSweep(params, config=config)

        def compute(x):
            return {"value": x * 5}

        sweep.execute(compute)

        # Load pickle file
        pkl_files = list(Path(tmpdir).glob("sweep_results_*.pkl"))
        assert len(pkl_files) > 0

        with open(pkl_files[0], "rb") as f:
            data = pickle.load(f)

        assert "parameters" in data
        assert "results" in data
        assert "metadata" in data
        assert len(data["results"]) == 3


@pytest.mark.unit
def test_save_results_json_summary():
    """Test JSON file contains sweep summary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        params = {"x": [1, 2]}
        config = SweepConfiguration(parameters=params, output_dir=Path(tmpdir), save_intermediate=True)
        sweep = ParameterSweep(params, config=config)

        def compute(x):
            return {"result": x}

        sweep.execute(compute)

        # Load JSON file
        json_files = list(Path(tmpdir).glob("sweep_summary_*.json"))
        assert len(json_files) > 0

        with open(json_files[0]) as f:
            summary = json.load(f)

        assert "parameters" in summary
        assert "total_combinations" in summary
        assert "successful_runs" in summary
        assert summary["total_combinations"] == 2
        assert summary["successful_runs"] == 2


@pytest.mark.unit
def test_intermediate_results_saved():
    """Test intermediate results are saved during execution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use 15 combinations to trigger intermediate save (every 10 runs)
        params = {"x": list(range(15))}
        config = SweepConfiguration(parameters=params, output_dir=Path(tmpdir), save_intermediate=True)
        sweep = ParameterSweep(params, config=config)

        def compute(x):
            return {"value": x}

        sweep.execute(compute)

        # Check intermediate files exist
        intermediate_files = list(Path(tmpdir).glob("intermediate_*.json"))

        # Should have at least one intermediate save (at 10 runs)
        assert len(intermediate_files) >= 1


@pytest.mark.unit
def test_save_intermediate_disabled():
    """Test save_intermediate=False prevents intermediate saves."""
    with tempfile.TemporaryDirectory() as tmpdir:
        params = {"x": list(range(15))}
        config = SweepConfiguration(parameters=params, output_dir=Path(tmpdir), save_intermediate=False)
        sweep = ParameterSweep(params, config=config)

        def compute(x):
            return {"value": x}

        sweep.execute(compute)

        # Should have no intermediate files
        intermediate_files = list(Path(tmpdir).glob("intermediate_*.json"))
        assert len(intermediate_files) == 0

        # But final results should not be saved either (since save_intermediate controls final save in execute)
        # Actually checking source code, _save_results() is called if save_intermediate is True
        # So with save_intermediate=False, no final results files either
        pkl_files = list(Path(tmpdir).glob("sweep_results_*.pkl"))
        json_files = list(Path(tmpdir).glob("sweep_summary_*.json"))

        # These should be empty because save_intermediate=False prevents _save_results() call
        assert len(pkl_files) == 0
        assert len(json_files) == 0


# ============================================================================
# Test: Factory Functions
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_create_grid_sweep():
    """Test create_grid_sweep factory function."""
    params = {"a": [1, 2], "b": [3, 4]}

    sweep = create_grid_sweep(params)

    assert isinstance(sweep, ParameterSweep)
    assert sweep.parameters == params
    assert sweep.total_combinations == 4


@pytest.mark.unit
@pytest.mark.fast
def test_create_random_sweep_basic():
    """Test create_random_sweep factory function.

    Issue #1284 / 2026-06-11 survey: the previous implementation generated
    n_samples values PER parameter then took the Cartesian product, yielding
    n_samples**k combinations for k parameters.  The correct behaviour is
    exactly n_samples independent joint draws (one value per parameter per draw).
    """
    params = {"x": (0.0, 1.0), "y": (5.0, 10.0)}
    n_samples = 20

    sweep = create_random_sweep(params, n_samples=n_samples)

    assert isinstance(sweep, ParameterSweep)
    # Total combinations == n_samples (paired draws, not Cartesian product).
    assert sweep.total_combinations == n_samples


@pytest.mark.unit
@pytest.mark.fast
def test_create_random_sweep_integer_parameters():
    """Test create_random_sweep with integer parameters."""
    params = {"count": (1, 10)}  # Integer range
    n_samples = 15

    sweep = create_random_sweep(params, n_samples=n_samples)

    # Execute to check integer values
    def compute(count):
        assert isinstance(count, int)
        return {"value": count}

    results = sweep.execute(compute)

    assert len(results) == n_samples
    # All count values should be integers between 1 and 10
    counts = [r["parameters"]["count"] for r in results]
    assert all(isinstance(c, int) for c in counts)
    assert all(1 <= c <= 10 for c in counts)


@pytest.mark.unit
@pytest.mark.fast
def test_create_random_sweep_float_parameters():
    """Test create_random_sweep with float parameters."""
    params = {"sigma": (0.1, 0.5)}  # Float range
    n_samples = 10

    sweep = create_random_sweep(params, n_samples=n_samples)

    def compute(sigma):
        assert isinstance(sigma, (float, np.floating))
        return {"value": sigma}

    results = sweep.execute(compute)

    assert len(results) == n_samples
    # All sigma values should be floats between 0.1 and 0.5
    sigmas = [r["parameters"]["sigma"] for r in results]
    assert all(0.1 <= s <= 0.5 for s in sigmas)


@pytest.mark.unit
@pytest.mark.fast
def test_create_adaptive_sweep_fallback():
    """Test create_adaptive_sweep returns grid sweep as fallback."""
    params = {"x": [1, 2, 3]}

    def objective(x):
        return x**2

    sweep = create_adaptive_sweep(params, objective, n_initial=5, n_iterations=10)

    # Should return a ParameterSweep (fallback to grid sweep)
    assert isinstance(sweep, ParameterSweep)
    assert sweep.parameters == params


# ============================================================================
# Test: Retry Logic
# ============================================================================


@pytest.mark.unit
def test_retry_failed_runs():
    """Test retry logic for failed runs."""
    params = {"x": [1, 2, 3]}
    config = SweepConfiguration(parameters=params, retry_failed=True, max_retries=3, save_intermediate=False)
    sweep = ParameterSweep(params, config=config)

    # This function will fail twice for x=2, then succeed
    executed_count = {"total": 0}

    def compute(x):
        executed_count["total"] += 1
        if x == 2 and executed_count["total"] < 5:  # Fail first few times for x=2
            raise RuntimeError("Intentional failure")
        return {"value": x}

    results = sweep.execute(compute)

    # Should eventually succeed with retries
    assert len(results) + len(sweep.failed_runs) == 3


@pytest.mark.unit
def test_retry_disabled():
    """Test retry_failed=False prevents retries."""
    params = {"x": [1, 2, 3]}
    config = SweepConfiguration(parameters=params, retry_failed=False, save_intermediate=False)
    sweep = ParameterSweep(params, config=config)

    attempt_count = {"count": 0}

    def compute(x):
        attempt_count["count"] += 1
        if x == 2:
            raise ValueError("Fail")
        return {"value": x}

    sweep.execute(compute)

    # Should have exactly 3 attempts (no retries for x=2)
    assert attempt_count["count"] == 3


@pytest.mark.unit
def test_max_retries_limit():
    """Test max_retries limit is respected."""
    params = {"x": [1]}
    config = SweepConfiguration(parameters=params, retry_failed=True, max_retries=2, save_intermediate=False)
    sweep = ParameterSweep(params, config=config)

    attempt_count = {"count": 0}

    def compute(x):
        attempt_count["count"] += 1
        # Always fail
        raise RuntimeError(f"Attempt {attempt_count['count']}")

    sweep.execute(compute)

    # Should try: initial + 2 retries = 3 total attempts
    assert attempt_count["count"] == 3
    assert len(sweep.failed_runs) == 1

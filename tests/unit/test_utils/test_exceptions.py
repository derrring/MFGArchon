#!/usr/bin/env python3
"""
Unit tests for mfgarchon/utils/exceptions.py

Tests comprehensive exception handling system including:
- MFGSolverError (base exception)
- ConvergenceError (convergence failures)
- ConfigurationError (invalid parameters)
- SolutionNotAvailableError (missing solutions)
- DimensionMismatchError (array dimension issues)
- NumericalInstabilityError (numerical problems)
- Validation utilities

Coverage target: mfgarchon/utils/exceptions.py (158 lines, 16% -> 70%+)
"""

import pytest

import numpy as np

from mfgarchon.utils.exceptions import (
    ConfigurationError,
    ConvergenceError,
    DimensionMismatchError,
    MFGSolverError,
    NumericalInstabilityError,
    SolutionNotAvailableError,
    check_numerical_stability,
    validate_array_dimensions,
    validate_parameter_value,
    validate_solver_state,
)

# =============================================================================
# Test MFGSolverError (Base Exception)
# =============================================================================


@pytest.mark.unit
def test_mfg_solver_error_basic():
    """Test basic MFGSolverError creation."""
    error = MFGSolverError("Test error message", solver_name="TestSolver")

    assert "TestSolver" in str(error)
    assert "Test error message" in str(error)
    assert error.solver_name == "TestSolver"


@pytest.mark.unit
def test_mfg_solver_error_with_suggestion():
    """Test MFGSolverError with suggested action."""
    error = MFGSolverError("Error occurred", solver_name="MySolver", suggested_action="Try increasing max_iterations")

    error_str = str(error)
    assert "MySolver" in error_str
    assert "Suggestion" in error_str
    assert "increasing max_iterations" in error_str


@pytest.mark.unit
def test_mfg_solver_error_with_diagnostics():
    """Test MFGSolverError with diagnostic data."""
    error = MFGSolverError(
        "Diagnostic test",
        solver_name="DiagSolver",
        diagnostic_data={"iterations": 100, "error": 0.01, "status": "failed"},
    )

    error_str = str(error)
    assert "Diagnostic Information" in error_str
    assert "iterations: 100" in error_str
    assert "error: 0.01" in error_str


@pytest.mark.unit
def test_mfg_solver_error_with_error_code():
    """Test MFGSolverError with error code."""
    error = MFGSolverError("Code test", error_code="ERR001")

    assert "Error Code: ERR001" in str(error)


# =============================================================================
# Test ConvergenceError
# =============================================================================


@pytest.mark.unit
def test_convergence_error_basic():
    """Test ConvergenceError with basic parameters."""
    error = ConvergenceError(
        iterations_used=100, max_iterations=200, final_error=0.01, tolerance=1e-6, solver_name="HJBSolver"
    )

    error_str = str(error)
    assert "HJBSolver" in error_str
    assert "100 iterations" in error_str
    assert "CONVERGENCE_FAILURE" in error_str


@pytest.mark.unit
def test_convergence_error_with_history():
    """Test ConvergenceError with convergence history."""
    history = [1.0, 0.5, 0.25, 0.15, 0.12, 0.11, 0.105, 0.102, 0.101, 0.1]
    error = ConvergenceError(
        iterations_used=10,
        max_iterations=20,
        final_error=0.1,
        tolerance=1e-3,
        convergence_history=history,
    )

    error_str = str(error)
    assert "convergence_trend" in error_str.lower() or "iterations_used: 10" in error_str


@pytest.mark.unit
def test_convergence_error_diagnostic_data():
    """Test ConvergenceError includes diagnostic data."""
    error = ConvergenceError(
        iterations_used=50, max_iterations=100, final_error=1e-3, tolerance=1e-6, solver_name="TestSolver"
    )

    assert error.diagnostic_data is not None
    assert "iterations_used" in error.diagnostic_data
    assert "final_error" in error.diagnostic_data
    assert "required_tolerance" in error.diagnostic_data


# =============================================================================
# Test ConfigurationError
# =============================================================================


@pytest.mark.unit
def test_configuration_error_basic():
    """Test ConfigurationError with parameter details."""
    error = ConfigurationError(
        parameter_name="max_iterations",
        provided_value=-10,
        expected_type=int,
        valid_range=(1, 10000),
        solver_name="MySolver",
    )

    error_str = str(error)
    assert "max_iterations" in error_str
    assert "-10" in error_str
    assert "int" in error_str


@pytest.mark.unit
def test_configuration_error_auto_suggestion():
    """Test ConfigurationError generates suggestions."""
    error = ConfigurationError(
        parameter_name="tolerance", provided_value=10.0, expected_type=float, valid_range=(1e-10, 1e-3)
    )

    error_str = str(error)
    assert "INVALID_CONFIGURATION" in error_str


# =============================================================================
# Test SolutionNotAvailableError
# =============================================================================


@pytest.mark.unit
def test_solution_not_available_error():
    """Test SolutionNotAvailableError creation."""
    error = SolutionNotAvailableError(operation_attempted="plot_solution", solver_name="HJBSolver")

    error_str = str(error)
    assert "plot_solution" in error_str
    assert "HJBSolver" in error_str
    assert "SOLUTION_NOT_AVAILABLE" in error_str


@pytest.mark.unit
def test_solution_not_available_with_reason():
    """Test SolutionNotAvailableError with solver state."""
    error = SolutionNotAvailableError(
        operation_attempted="get_value_function", solver_name="MySolver", solver_state="initialized"
    )

    error_str = str(error)
    assert "get_value_function" in error_str
    assert "not been run" in error_str


# =============================================================================
# Test DimensionMismatchError
# =============================================================================


@pytest.mark.unit
def test_dimension_mismatch_error_basic():
    """Test DimensionMismatchError with shape info."""
    error = DimensionMismatchError(
        array_name="value_function",
        provided_shape=(10, 20),
        expected_shape=(10, 10),
        solver_name="TestSolver",
    )

    error_str = str(error)
    assert "value_function" in error_str
    assert "(10, 20)" in error_str
    assert "(10, 10)" in error_str
    assert "DIMENSION_MISMATCH" in error_str


@pytest.mark.unit
def test_dimension_mismatch_error_suggestions():
    """Test DimensionMismatchError includes helpful suggestions."""
    error = DimensionMismatchError(
        array_name="density", provided_shape=(100,), expected_shape=(10, 10), solver_name="FPSolver"
    )

    error_str = str(error)
    assert "Suggestion" in error_str or "density" in error_str


# =============================================================================
# Test NumericalInstabilityError
# =============================================================================


@pytest.mark.unit
def test_numerical_instability_error_basic():
    """Test NumericalInstabilityError creation."""
    error = NumericalInstabilityError(instability_type="NaN detected", solver_name="MySolver", iteration_number=50)

    error_str = str(error)
    assert "NaN detected" in error_str
    assert "50" in error_str
    assert "NUMERICAL_INSTABILITY" in error_str


@pytest.mark.unit
def test_numerical_instability_error_with_values():
    """Test NumericalInstabilityError with problematic values."""
    error = NumericalInstabilityError(
        instability_type="Overflow",
        solver_name="HJBSolver",
        iteration_number=10,
        problematic_values={"max_value": 1e308, "location": "grid[5,5]"},
    )

    error_str = str(error)
    assert "Overflow" in error_str
    assert "1e" in error_str or "max_value" in error_str


# =============================================================================
# Test Validation Utilities
# =============================================================================


@pytest.mark.unit
def test_validate_solver_state_with_solution():
    """Test validate_solver_state when solution exists."""

    class MockSolver:
        def __init__(self):
            self.u = np.ones((10, 10))
            self.m = np.ones((10, 10))
            self._solution_computed = True

    solver = MockSolver()

    # Should not raise
    result = validate_solver_state(solver, "test_operation")
    assert result is None


@pytest.mark.unit
def test_validate_solver_state_without_solution():
    """Test validate_solver_state raises when solution missing."""

    class MockSolver:
        def __init__(self):
            self.u = None
            self.m = None

    solver = MockSolver()

    with pytest.raises(SolutionNotAvailableError) as exc_info:
        validate_solver_state(solver, "get_solution")

    assert "get_solution" in str(exc_info.value)


@pytest.mark.unit
def test_validate_array_dimensions_correct():
    """Test validate_array_dimensions with correct dimensions."""
    arr = np.ones((10, 20))

    # Should not raise
    result = validate_array_dimensions(arr, expected_shape=(10, 20), array_name="test_array")
    assert result is None


@pytest.mark.unit
def test_validate_array_dimensions_incorrect():
    """Test validate_array_dimensions raises on mismatch."""
    arr = np.ones((10, 20))

    with pytest.raises(DimensionMismatchError) as exc_info:
        validate_array_dimensions(arr, expected_shape=(10, 10), array_name="my_array")

    error_str = str(exc_info.value)
    assert "my_array" in error_str
    assert "(10, 20)" in error_str


@pytest.mark.unit
def test_validate_parameter_value_in_range():
    """Test validate_parameter_value with valid value."""
    # Should not raise
    result = validate_parameter_value(value=1e-6, parameter_name="tolerance", valid_range=(1e-10, 1.0))
    assert result is None


@pytest.mark.unit
def test_validate_parameter_value_out_of_range():
    """Test validate_parameter_value raises on invalid value."""
    with pytest.raises(ConfigurationError) as exc_info:
        validate_parameter_value(value=-10, parameter_name="max_iterations", valid_range=(1, 1000))

    assert "max_iterations" in str(exc_info.value)


@pytest.mark.unit
def test_validate_parameter_value_wrong_type():
    """Test validate_parameter_value validates type."""
    with pytest.raises(ConfigurationError) as exc_info:
        validate_parameter_value(value="invalid", parameter_name="num_points", expected_type=int)

    assert "num_points" in str(exc_info.value)


@pytest.mark.unit
def test_check_numerical_stability_clean_array():
    """Test check_numerical_stability with clean data."""
    arr = np.ones((10, 10))

    # Should not raise
    result = check_numerical_stability(arr, "test_array")
    assert result is None


@pytest.mark.unit
def test_check_numerical_stability_nan():
    """Test check_numerical_stability detects NaN."""
    arr = np.ones((10, 10))
    arr[5, 5] = np.nan

    with pytest.raises(NumericalInstabilityError) as exc_info:
        check_numerical_stability(arr, "value_function")

    error_str = str(exc_info.value)
    assert "NaN" in error_str or "nan" in error_str.lower()


@pytest.mark.unit
def test_check_numerical_stability_inf():
    """Test check_numerical_stability detects infinity."""
    arr = np.ones((10, 10))
    arr[3, 3] = np.inf

    with pytest.raises(NumericalInstabilityError) as exc_info:
        check_numerical_stability(arr, "density")

    error_str = str(exc_info.value)
    assert "inf" in error_str.lower() or "infinite" in error_str.lower()


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
def test_exception_hierarchy():
    """Test exception inheritance hierarchy."""
    # All custom exceptions should inherit from MFGSolverError
    assert issubclass(ConvergenceError, MFGSolverError)
    assert issubclass(ConfigurationError, MFGSolverError)
    assert issubclass(SolutionNotAvailableError, MFGSolverError)
    assert issubclass(DimensionMismatchError, MFGSolverError)
    assert issubclass(NumericalInstabilityError, MFGSolverError)


@pytest.mark.unit
def test_exception_catching():
    """Test exceptions can be caught by base class."""
    with pytest.raises(MFGSolverError):
        raise ConvergenceError(iterations_used=10, max_iterations=20, final_error=0.1, tolerance=1e-6)

    with pytest.raises(MFGSolverError):
        raise ConfigurationError(parameter_name="test", provided_value=None, expected_type=str)


@pytest.mark.unit
def test_error_message_formatting():
    """Test consistent error message formatting across all exceptions."""
    errors = [
        ConvergenceError(10, 20, 0.1, 1e-6, solver_name="TestSolver"),
        ConfigurationError(parameter_name="param", provided_value="bad", expected_type=str, solver_name="TestSolver"),
        SolutionNotAvailableError(operation_attempted="op", solver_name="TestSolver"),
        DimensionMismatchError(array_name="arr", provided_shape=(10,), expected_shape=(20,), solver_name="TestSolver"),
        NumericalInstabilityError(instability_type="type", solver_name="TestSolver"),
    ]

    for error in errors:
        error_str = str(error)
        # All should include solver name in brackets
        assert "[TestSolver]" in error_str
        # All should have some diagnostic info or message
        assert len(error_str) > 20


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.unit
def test_mfg_solver_error_no_solver_name():
    """Test MFGSolverError with no solver name."""
    error = MFGSolverError("Test message")

    error_str = str(error)
    assert "Unknown Solver" in error_str or "Test message" in error_str


@pytest.mark.unit
def test_validate_array_dimensions_none_array():
    """Test validate_array_dimensions with None array."""

    class MockSolver:
        pass

    # The behaviour is deterministic, so decide it rather than accepting any of three types:
    # exceptions.py:371 evaluates ``array.shape`` with no None guard, so None reaches the
    # attribute lookup and the interpreter raises. Accepting a set of types let the diagnostic
    # regress unnoticed.
    #
    # This pins the CURRENT contract, and that contract is a gap: a boundary validator should
    # reject None with its own DimensionMismatchError, not lean on an AttributeError from the
    # interpreter. If the guard is added, this becomes pytest.raises(DimensionMismatchError).
    with pytest.raises(AttributeError, match=r"NoneType.*shape"):
        validate_array_dimensions(None, expected_shape=(10, 10), array_name="test", solver_name="Test")


@pytest.mark.unit
def test_validate_parameter_value_none_constraints():
    """Test validate_parameter_value with no constraints."""
    # Should not raise - no validation applied
    result = validate_parameter_value(value=100, parameter_name="param")
    assert result is None


@pytest.mark.unit
def test_check_numerical_stability_empty_array():
    """Test check_numerical_stability with empty array."""
    arr = np.array([])

    # The old form assigned a bool literal on both branches and asserted isinstance(..., bool),
    # which is true whether the call returns or raises either listed type. Traced instead:
    # exceptions.py:406-422 runs np.any over the empty array for all three guards (isnan, isinf,
    # |x| > 1e10), each False by vacuity, so control reaches the bare ``return``. An empty array
    # is not an instability; pin that, rather than accepting either outcome.
    assert check_numerical_stability(arr, "empty_array") is None

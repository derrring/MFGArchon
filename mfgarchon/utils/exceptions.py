"""
Enhanced exception classes for MFGarchon with helpful error messages and user guidance.

This module provides specialized exception classes that give users clear,
actionable error messages with suggested solutions.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class MFGSolverError(Exception):
    """
    Base exception for MFG solver errors with helpful context and suggestions.

    This exception class provides structured error information including:
    - Clear error description
    - Solver context information
    - Suggested actions for resolution
    - Optional diagnostic data
    """

    def __init__(
        self,
        message: str,
        solver_name: str | None = None,
        suggested_action: str | None = None,
        error_code: str | None = None,
        diagnostic_data: dict[str, Any] | None = None,
    ):
        self.solver_name = solver_name or "Unknown Solver"
        self.suggested_action = suggested_action
        self.error_code = error_code
        self.diagnostic_data = diagnostic_data or {}

        # Format comprehensive error message
        full_message = f"[{self.solver_name}] {message}"

        if self.suggested_action:
            full_message += f"\nSuggestion: {self.suggested_action}"

        if self.error_code:
            full_message += f"\nError Code: {self.error_code}"

        if self.diagnostic_data:
            full_message += "\nDiagnostic Information:"
            for key, value in self.diagnostic_data.items():
                full_message += f"\n   • {key}: {value}"

        super().__init__(full_message)


class ConvergenceError(MFGSolverError):
    """Exception raised when solver fails to converge."""

    def __init__(
        self,
        iterations_used: int,
        max_iterations: int,
        final_error: float,
        tolerance: float,
        solver_name: str | None = None,
        convergence_history: list[float] | None = None,
    ):
        diagnostic_data = {
            "iterations_used": iterations_used,
            "max_iterations": max_iterations,
            "final_error": f"{final_error:.2e}",
            "required_tolerance": f"{tolerance:.2e}",
            "error_ratio": f"{final_error / tolerance:.1f}x too large",
        }

        if convergence_history:
            diagnostic_data["convergence_trend"] = _analyze_convergence_trend(convergence_history)

        # Generate helpful suggestions based on the convergence pattern
        suggested_action = _generate_convergence_suggestions(
            final_error, tolerance, iterations_used, max_iterations, convergence_history
        )

        message = f"Failed to converge after {iterations_used} iterations"

        super().__init__(
            message=message,
            solver_name=solver_name,
            suggested_action=suggested_action,
            error_code="CONVERGENCE_FAILURE",
            diagnostic_data=diagnostic_data,
        )


class ConfigurationError(MFGSolverError):
    """Exception raised when solver configuration is invalid."""

    def __init__(
        self,
        parameter_name: str,
        provided_value: Any,
        expected_type: type | None = None,
        valid_range: tuple | None = None,
        solver_name: str | None = None,
    ):
        diagnostic_data = {
            "parameter": parameter_name,
            "provided_value": str(provided_value),
            "provided_type": type(provided_value).__name__,
        }

        if expected_type:
            diagnostic_data["expected_type"] = expected_type.__name__

        if valid_range:
            diagnostic_data["valid_range"] = f"[{valid_range[0]}, {valid_range[1]}]"

        # Generate specific suggestions based on the configuration error
        suggested_action = _generate_configuration_suggestions(
            parameter_name, provided_value, expected_type, valid_range
        )

        message = f"Invalid configuration for parameter '{parameter_name}'"

        super().__init__(
            message=message,
            solver_name=solver_name,
            suggested_action=suggested_action,
            error_code="INVALID_CONFIGURATION",
            diagnostic_data=diagnostic_data,
        )


class SolutionNotAvailableError(MFGSolverError):
    """Exception raised when trying to access solution before solving."""

    def __init__(
        self,
        operation_attempted: str,
        solver_name: str | None = None,
        solver_state: str | None = None,
    ):
        diagnostic_data = {
            "attempted_operation": operation_attempted,
            "solver_state": solver_state or "not_solved",
        }

        suggested_action = f"Call solve() method first before attempting '{operation_attempted}'"

        message = f"Cannot perform '{operation_attempted}' - solver has not been run"

        super().__init__(
            message=message,
            solver_name=solver_name,
            suggested_action=suggested_action,
            error_code="SOLUTION_NOT_AVAILABLE",
            diagnostic_data=diagnostic_data,
        )


class DimensionMismatchError(MFGSolverError):
    """Exception raised when array dimensions don't match expected values."""

    def __init__(
        self,
        array_name: str,
        provided_shape: tuple,
        expected_shape: tuple,
        solver_name: str | None = None,
        context: str | None = None,
    ):
        diagnostic_data = {
            "array_name": array_name,
            "provided_shape": str(provided_shape),
            "expected_shape": str(expected_shape),
            "dimension_mismatch": _describe_dimension_mismatch(provided_shape, expected_shape),
        }

        if context:
            diagnostic_data["context"] = context

        suggested_action = _generate_dimension_suggestions(array_name, provided_shape, expected_shape)

        message = f"Dimension mismatch for {array_name}"

        super().__init__(
            message=message,
            solver_name=solver_name,
            suggested_action=suggested_action,
            error_code="DIMENSION_MISMATCH",
            diagnostic_data=diagnostic_data,
        )


class NumericalInstabilityError(MFGSolverError):
    """Exception raised when numerical instability is detected."""

    def __init__(
        self,
        instability_type: str,
        iteration_number: int | None = None,
        problematic_values: dict[str, Any] | None = None,
        solver_name: str | None = None,
    ):
        diagnostic_data = {"instability_type": instability_type}

        if iteration_number is not None:
            diagnostic_data["iteration"] = str(iteration_number)

        if problematic_values:
            diagnostic_data.update(problematic_values)

        suggested_action = _generate_stability_suggestions(instability_type, problematic_values)

        message = f"Numerical instability detected: {instability_type}"

        super().__init__(
            message=message,
            solver_name=solver_name,
            suggested_action=suggested_action,
            error_code="NUMERICAL_INSTABILITY",
            diagnostic_data=diagnostic_data,
        )


# Helper functions for generating specific suggestions


def _analyze_convergence_trend(history: list[float]) -> str:
    """Analyze convergence history to determine trend."""
    if len(history) < 3:
        return "insufficient_data"

    recent = history[-3:]

    # Check if converging
    if recent[-1] < recent[-2] < recent[-3]:
        return "converging_slowly"
    elif recent[-1] > recent[-2] * 1.1:
        return "diverging"
    elif max(recent) / min(recent) < 1.1:
        return "stagnating"
    else:
        return "oscillating"


def _generate_convergence_suggestions(
    final_error: float,
    tolerance: float,
    iterations_used: int,
    max_iterations: int,
    history: list[float] | None = None,
) -> str:
    """Generate specific suggestions for convergence issues."""

    error_ratio = final_error / tolerance

    if error_ratio < 2:
        return "Increase max_iterations slightly - very close to convergence"
    elif error_ratio < 10:
        return "Try: 1) Increase max_iterations, 2) Relax tolerance slightly, or 3) Improve initial guess"
    else:
        base_suggestion = (
            "Large error suggests: 1) Check problem parameters, 2) Reduce time step size, 3) Use better initialization"
        )

        if history and len(history) > 3:
            trend = _analyze_convergence_trend(history)
            if trend == "diverging":
                return base_suggestion + ", 4) Reduce damping parameter"
            elif trend == "oscillating":
                return base_suggestion + ", 4) Increase damping parameter"

        return base_suggestion


def _generate_configuration_suggestions(
    parameter_name: str,
    provided_value: Any,
    expected_type: type | None,
    valid_range: tuple | None,
) -> str:
    """Generate specific suggestions for configuration errors."""

    suggestions = []

    if expected_type and not isinstance(provided_value, expected_type):
        suggestions.append(f"Convert {parameter_name} to {expected_type.__name__}")

    if valid_range and isinstance(provided_value, int | float):
        if provided_value < valid_range[0]:
            suggestions.append(f"Increase {parameter_name} to at least {valid_range[0]}")
        elif provided_value > valid_range[1]:
            suggestions.append(f"Decrease {parameter_name} to at most {valid_range[1]}")

    # Common parameter-specific suggestions
    if "tolerance" in parameter_name.lower() and isinstance(provided_value, int | float):
        if provided_value <= 0:
            suggestions.append("Tolerance must be positive")
        elif provided_value > 1e-2:
            suggestions.append("Consider smaller tolerance for better accuracy")

    if "iteration" in parameter_name.lower() and isinstance(provided_value, int | float):
        if provided_value <= 0:
            suggestions.append("Number of iterations must be positive")
        elif provided_value < 5:
            suggestions.append("Consider at least 5 iterations for meaningful convergence")

    return " | ".join(suggestions) if suggestions else f"Check {parameter_name} value and try again"


def _describe_dimension_mismatch(provided_shape: tuple, expected_shape: tuple) -> str:
    """Describe the specific nature of dimension mismatch."""

    if len(provided_shape) != len(expected_shape):
        return f"Wrong number of dimensions: got {len(provided_shape)}, expected {len(expected_shape)}"

    mismatches = []
    for i, (provided, expected) in enumerate(zip(provided_shape, expected_shape, strict=False)):
        if provided != expected:
            mismatches.append(f"axis {i}: got {provided}, expected {expected}")

    return " | ".join(mismatches)


def _generate_dimension_suggestions(array_name: str, provided_shape: tuple, expected_shape: tuple) -> str:
    """Generate specific suggestions for dimension errors."""

    if "warm_start" in array_name.lower():
        return f"Ensure warm start data matches problem dimensions: reshape to {expected_shape}"

    if len(provided_shape) < len(expected_shape):
        return f"Add missing dimensions to {array_name}: reshape or expand to {expected_shape}"
    elif len(provided_shape) > len(expected_shape):
        return f"Remove extra dimensions from {array_name}: reshape to {expected_shape}"
    else:
        return f"Reshape {array_name} to match problem grid: {expected_shape}"


def _generate_stability_suggestions(instability_type: str, problematic_values: dict[str, Any] | None) -> str:
    """Generate suggestions for numerical stability issues."""

    if "nan" in instability_type.lower():
        return "Check for: 1) Division by zero, 2) Invalid initial conditions, 3) Too large time steps"
    elif "inf" in instability_type.lower():
        return "Reduce: 1) Time step size, 2) Parameter values, 3) Initial condition magnitudes"
    elif "oscillation" in instability_type.lower():
        return "Try: 1) Increase damping parameter, 2) Reduce time step, 3) Better initial guess"
    elif "divergence" in instability_type.lower():
        return "Consider: 1) Smaller time steps, 2) More conservative parameters, 3) Improved regularization"
    else:
        return "Check numerical parameters and consider using more stable solver settings"


# Convenience functions for common error scenarios


def validate_solver_state(solver, operation_name: str):
    """Validate that solver has been run before accessing results."""
    if not getattr(solver, "_solution_computed", False):
        raise SolutionNotAvailableError(
            operation_attempted=operation_name,
            solver_name=getattr(solver, "__class__", type(solver)).__name__,
        )


def validate_array_dimensions(
    array: np.ndarray, expected_shape: tuple, array_name: str, solver_name: str | None = None
):
    """Validate that array has expected dimensions."""
    if array.shape != expected_shape:
        raise DimensionMismatchError(
            array_name=array_name,
            provided_shape=array.shape,
            expected_shape=expected_shape,
            solver_name=solver_name,
        )


def validate_parameter_value(
    value: Any,
    parameter_name: str,
    expected_type: type | None = None,
    valid_range: tuple | None = None,
    solver_name: str | None = None,
):
    """Validate parameter value and type."""
    if expected_type and not isinstance(value, expected_type):
        raise ConfigurationError(
            parameter_name=parameter_name,
            provided_value=value,
            expected_type=expected_type,
            solver_name=solver_name,
        )

    if valid_range and isinstance(value, int | float):
        if not (valid_range[0] <= value <= valid_range[1]):
            raise ConfigurationError(
                parameter_name=parameter_name,
                provided_value=value,
                valid_range=valid_range,
                solver_name=solver_name,
            )


def check_numerical_stability(
    array: np.ndarray, array_name: str, iteration: int | None = None, solver_name: str | None = None
):
    """Check array for numerical stability issues."""
    problematic_values = {}

    if np.any(np.isnan(array)):
        problematic_values["nan_count"] = np.sum(np.isnan(array))
        instability_type = "NaN values detected"
    elif np.any(np.isinf(array)):
        problematic_values["inf_count"] = np.sum(np.isinf(array))
        instability_type = "Infinite values detected"
    elif np.any(np.abs(array) > 1e10):
        problematic_values["max_magnitude"] = f"{np.max(np.abs(array)):.2e}"
        instability_type = "Extremely large values detected"
    else:
        return  # All good

    problematic_values["array_name"] = array_name

    raise NumericalInstabilityError(
        instability_type=instability_type,
        iteration_number=iteration,
        problematic_values=problematic_values,
        solver_name=solver_name,
    )


# General library exceptions (Issue #546)
# These are for non-solver errors (I/O, configuration, dependencies)


class LibraryError(Exception):
    """Base exception for general library errors (not solver-specific).

    Use this for errors in:
    - File I/O operations
    - Configuration loading
    - Dependency management
    - General utilities

    For solver-specific errors, use MFGSolverError hierarchy instead.
    """


class FileOperationError(LibraryError):
    """Raised when file I/O operation fails.

    Examples:
        - Cannot read/write checkpoint file
        - Invalid file format
        - Permission denied
    """


class InvalidConfigurationError(LibraryError):
    """Raised when configuration file or parameters are invalid.

    This is for general configuration errors. For solver parameter
    validation, use ConfigurationError (from MFGSolverError hierarchy).

    Examples:
        - Missing required configuration file
        - Invalid YAML/JSON syntax
        - Missing required configuration keys
    """


class DependencyMissingError(LibraryError):
    """Raised when required dependency is unavailable.

    Examples:
        - JAX not installed when required
        - Incompatible library version detected
        - Missing optional dependency without fallback
    """

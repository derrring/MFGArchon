#!/usr/bin/env python3
"""
Common Validation Utilities for MFG Solvers

This module provides centralized validation functions to eliminate
code duplication across different solver implementations.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SolutionValidationError(Exception):
    """Exception raised when solution validation fails."""


def validate_solution_array(
    solution: NDArray[np.floating],
    name: str,
    allow_nan: bool = False,
    allow_inf: bool = False,
    min_value: float | None = None,
    max_value: float | None = None,
) -> NDArray[np.floating]:
    """
    Validate a solution array for common numerical issues.

    Args:
        solution: Array to validate
        name: Name of the solution (for error messages)
        allow_nan: Whether NaN values are acceptable
        allow_inf: Whether infinite values are acceptable
        min_value: Minimum acceptable value (optional)
        max_value: Maximum acceptable value (optional)

    Returns:
        The validated solution array

    Raises:
        SolutionValidationError: If validation fails
    """
    if not isinstance(solution, np.ndarray):
        raise SolutionValidationError(f"{name} must be a numpy array, got {type(solution)}")

    if solution.size == 0:
        raise SolutionValidationError(f"{name} array is empty")

    # Check for NaN values
    if not allow_nan and np.any(np.isnan(solution)):
        nan_count = np.sum(np.isnan(solution))
        raise SolutionValidationError(f"{name} contains {nan_count} NaN values")

    # Check for infinite values
    if not allow_inf and np.any(np.isinf(solution)):
        inf_count = np.sum(np.isinf(solution))
        raise SolutionValidationError(f"{name} contains {inf_count} infinite values")

    # Check value bounds
    finite_solution = solution[np.isfinite(solution)]
    if finite_solution.size > 0:
        if min_value is not None and np.any(finite_solution < min_value):
            min_val = np.min(finite_solution)
            raise SolutionValidationError(f"{name} contains values below minimum {min_value}: {min_val}")

        if max_value is not None and np.any(finite_solution > max_value):
            max_val = np.max(finite_solution)
            raise SolutionValidationError(f"{name} contains values above maximum {max_value}: {max_val}")

    return solution


def validate_mfg_solution(U: NDArray[np.floating], M: NDArray[np.floating], strict: bool = True) -> dict[str, Any]:
    """
    Validate a complete MFG solution (value function and distribution).

    Args:
        U: Value function array
        M: Distribution array
        strict: Whether to raise exceptions or just return warnings

    Returns:
        Dictionary with validation results and diagnostics

    Raises:
        SolutionValidationError: If validation fails and strict=True
    """
    validation_results: dict[str, Any] = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "diagnostics": {},
    }

    try:
        # Validate value function
        validate_solution_array(U, "Value function U")

        # Check for reasonable value function properties
        if U.ndim >= 2:
            # Check terminal condition (should be reasonable)
            U_terminal = U[-1, :]
            if np.all(U_terminal == 0):
                validation_results["warnings"].append("Terminal value function is all zeros")

        validation_results["diagnostics"]["U_stats"] = {
            "shape": U.shape,
            "min": np.min(U[np.isfinite(U)]) if np.any(np.isfinite(U)) else np.nan,
            "max": np.max(U[np.isfinite(U)]) if np.any(np.isfinite(U)) else np.nan,
            "mean": np.mean(U[np.isfinite(U)]) if np.any(np.isfinite(U)) else np.nan,
        }

    except SolutionValidationError as e:
        validation_results["valid"] = False
        validation_results["errors"].append(f"Value function validation failed: {e}")
        if strict:
            raise

    try:
        # Validate distribution
        validate_solution_array(M, "Distribution M", min_value=0.0)

        # Check mass conservation
        if M.ndim >= 2:
            # Compute total mass at each time step
            # #2145: the grid's measure, not `sum(m)*dx`. The end nodes hold half a cell each.
            # This function has no geometry, so it keeps the [0,1] assumption it already made --
            # harmless here because a uniform rescale of the weights cancels in the ratio below,
            # while the CHOICE of functional does not: one of the two is conserved at a no-flux
            # wall and the other is not, which is the whole of #2145.
            from mfgarchon.utils.numerical.quadrature import quadrature_weights_1d

            if M.shape[1] > 1:
                w = quadrature_weights_1d(np.linspace(0.0, 1.0, M.shape[1]))
                total_masses = M @ w
            else:
                total_masses = np.sum(M, axis=1)

            mass_variation = np.std(total_masses) / np.mean(total_masses) * 100
            validation_results["diagnostics"]["mass_conservation_error"] = mass_variation

            if mass_variation > 1.0:  # 1% threshold
                validation_results["warnings"].append(f"Mass conservation error: {mass_variation:.2f}%")

        validation_results["diagnostics"]["M_stats"] = {
            "shape": M.shape,
            "min": np.min(M[np.isfinite(M)]) if np.any(np.isfinite(M)) else np.nan,
            "max": np.max(M[np.isfinite(M)]) if np.any(np.isfinite(M)) else np.nan,
            "total_mass": (np.sum(M[np.isfinite(M)]) if np.any(np.isfinite(M)) else np.nan),
        }

    except SolutionValidationError as e:
        validation_results["valid"] = False
        validation_results["errors"].append(f"Distribution validation failed: {e}")
        if strict:
            raise

    # Shape consistency check
    if U.shape != M.shape:
        error_msg = f"Shape mismatch: U{U.shape} vs M{M.shape}"
        validation_results["valid"] = False
        validation_results["errors"].append(error_msg)
        if strict:
            raise SolutionValidationError(error_msg)

    return validation_results


def validate_convergence_parameters(max_iterations: int, tolerance: float, parameter_name: str = "convergence") -> None:
    """
    Validate common convergence parameters.

    Args:
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        parameter_name: Name for error messages

    Raises:
        ValueError: If parameters are invalid
    """
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError(f"{parameter_name} max_iterations must be a positive integer, got {max_iterations}")

    if not isinstance(tolerance, int | float) or tolerance <= 0:
        raise ValueError(f"{parameter_name} tolerance must be a positive number, got {tolerance}")

    if tolerance >= 1.0:
        warnings.warn(f"{parameter_name} tolerance {tolerance} is unusually large (>=1.0)")


def safe_solution_return(U: NDArray[np.floating], M: NDArray[np.floating], info: dict | None = None) -> tuple:
    """
    Safely return MFG solution with validation.

    Args:
        U: Value function
        M: Distribution
        info: Additional information dictionary

    Returns:
        Tuple of (U, M, info) with validation performed
    """
    try:
        validation_results = validate_mfg_solution(U, M, strict=False)

        if info is None:
            info = {}

        info["validation"] = validation_results

        if not validation_results["valid"]:
            info["solution_status"] = "invalid"
            warnings.warn(f"Solution validation failed: {validation_results['errors']}")
        else:
            info["solution_status"] = "valid"
            if validation_results["warnings"]:
                warnings.warn(f"Solution warnings: {validation_results['warnings']}")

        return U, M, info

    except Exception as e:
        warnings.warn(f"Solution validation error: {e}")
        if info is None:
            info = {}
        info["validation_error"] = str(e)
        info["solution_status"] = "validation_failed"
        return U, M, info


# Backward compatibility aliases
def validate_solution(solution: NDArray[np.floating], name: str) -> bool:
    """
    Legacy validation function for backward compatibility.

    Args:
        solution: Solution array to validate
        name: Name for error messages

    Returns:
        True if valid

    Raises:
        SolutionValidationError: If validation fails
    """
    validate_solution_array(solution, name)
    return True

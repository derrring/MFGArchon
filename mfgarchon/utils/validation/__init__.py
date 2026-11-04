"""
Validation utilities for MFGarchon.

This module provides comprehensive validation for MFG problem inputs,
custom functions, arrays, and runtime solver outputs.

Modules:
    protocol: Core types (ValidationResult, ValidationError, ValidationIssue)
    components: IC/BC validation (m_initial, u_terminal)
    functions: Custom function validation (Hamiltonian, drift, running_cost)
    arrays: Array validation (dtype, shape, dimension)
    runtime: Runtime checks (NaN/Inf, bounds)

Usage:
    from mfgarchon.utils.validation import (
        validate_components,
        validate_custom_functions,
        check_finite,
    )

    # Validate components at problem construction
    result = validate_components(components, geometry)
    if not result.is_valid:
        raise ValidationError(result)

Note: For convergence/divergence monitoring during solving, use:
    from mfgarchon.utils.convergence import DistributionConvergenceMonitor

Issue #689: Validation module infrastructure
Parent: #685 (Comprehensive Input Validation Initiative)
"""

# Array validation (Issue #687)
from mfgarchon.utils.validation.arrays import (
    validate_array_dtype,
    validate_array_shape,
    validate_field_dimension,
    validate_field_shape,
    validate_finite,
    validate_non_negative,
)

# Components validation (Issue #679, #681-#684)
from mfgarchon.utils.validation.components import (
    detect_callable_signature,
    validate_boundary_conditions,
    validate_components,
    validate_m_initial,
    validate_mass_normalization,
    validate_u_final,  # Deprecated: use validate_u_terminal
    validate_u_terminal,
)

# Custom function validation (Issue #686)
from mfgarchon.utils.validation.functions import (
    validate_custom_functions,
    validate_hamiltonian,
    validate_hamiltonian_consistency,
    validate_hamiltonian_derivative,
)
from mfgarchon.utils.validation.protocol import (
    ValidationError,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    Validator,
)

# Runtime validation (Issue #688)
# Note: For convergence/divergence monitors, use mfgarchon.utils.convergence
from mfgarchon.utils.validation.runtime import (
    check_bounds,
    check_finite,
    validate_solver_output,
)

__all__ = [
    # Protocol
    "ValidationResult",
    "ValidationError",
    "ValidationIssue",
    "ValidationSeverity",
    "Validator",
    # Components
    "validate_components",
    "validate_m_initial",
    "validate_u_terminal",
    "validate_u_final",  # Deprecated alias
    "validate_mass_normalization",
    "validate_boundary_conditions",
    "detect_callable_signature",
    # Functions
    "validate_custom_functions",
    "validate_hamiltonian",
    "validate_hamiltonian_derivative",
    "validate_hamiltonian_consistency",
    # Arrays
    "validate_array_dtype",
    "validate_array_shape",
    "validate_field_shape",
    "validate_field_dimension",
    "validate_finite",
    "validate_non_negative",
    # Runtime
    "check_finite",
    "check_bounds",
    "validate_solver_output",
]

"""
Finite Difference Stencils for MFGarchon.

This module provides low-level stencil implementations with fixed coefficients.
Stencils are the building blocks for differential operators.

Conceptual Distinction:
    - **Stencils** (this module): Fixed coefficient formulas
        e.g., central diff = [-1, 0, 1] / (2h)
    - **Reconstruction** (operators/reconstruction/): Adaptive strategies
        e.g., WENO combines multiple stencils based on smoothness

Available Stencils:
    First-order derivatives:
        - gradient_central: 2nd-order, symmetric
        - gradient_forward: 1st-order, positive bias
        - gradient_backward: 1st-order, negative bias
        - gradient_upwind: the Rouy-Tourin upwind momentum (`upwind_momentum`'s preset)
        - upwind_momentum / upwind_momentum_derivatives: the HJB numerical Hamiltonian's momentum (#2313)

    Second-order derivatives:
        - laplacian_stencil_1d: Standard 3-point stencil
        - laplacian_stencil_nd: Sum of 1D stencils

    Utilities:
        - fix_boundaries_one_sided: Correct boundary values
        - get_gradient_stencil_coefficients: For matrix assembly
        - get_laplacian_stencil_coefficients: For matrix assembly

Usage:
    >>> from mfgarchon.operators.stencils import gradient_central, gradient_upwind
    >>> du_dx = gradient_central(u, axis=0, h=0.1)

Note:
    For boundary-aware operators with LinearOperator interface,
    use mfgarchon.operators.differential instead.

Created: 2026-01-24 (Operator module reorganization)
"""

from mfgarchon.operators.stencils.finite_difference import (
    DEFAULT_NUMERICAL_HAMILTONIAN,
    NumericalHamiltonian,
    fix_boundaries_one_sided,
    get_gradient_stencil_coefficients,
    get_laplacian_stencil_coefficients,
    gradient_backward,
    gradient_central,
    gradient_forward,
    gradient_upwind,
    laplacian_stencil_1d,
    laplacian_stencil_nd,
    upwind_momentum,
    upwind_momentum_derivatives,
)

__all__ = [
    # First-order derivatives
    "gradient_central",
    "gradient_forward",
    "gradient_backward",
    "gradient_upwind",
    "upwind_momentum",
    "upwind_momentum_derivatives",
    "NumericalHamiltonian",
    "DEFAULT_NUMERICAL_HAMILTONIAN",
    # Boundary handling
    "fix_boundaries_one_sided",
    # Second-order derivatives
    "laplacian_stencil_1d",
    "laplacian_stencil_nd",
    # Coefficients for matrix assembly
    "get_gradient_stencil_coefficients",
    "get_laplacian_stencil_coefficients",
]

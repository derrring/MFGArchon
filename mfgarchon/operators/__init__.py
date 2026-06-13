"""
Operator Framework for MFGarchon.

This module provides mathematical operators for PDE solving:

Organization:
    differential/     - Differential operators (gradient, laplacian, divergence, advection)
    integro_diff/     - Integro-differential (non-local PDE) operators: Levy jumps, graphon
    interpolation/    - Interpolation and projection operators
    stencils/         - Low-level finite difference stencils
    reconstruction/   - High-order reconstruction strategies (WENO, ENO)

Conceptual Hierarchy:
    Stencils (fixed coefficients) -> Reconstruction (adaptive) -> Differential Operators

LinearOperator Classes:
    All differential operators implement scipy.sparse.linalg.LinearOperator interface:
    - Matrix-vector product: L @ u_flat
    - Callable interface: L(u) (preserves field shape)
    - Operator algebra: L1 + L2, alpha * L, L1 @ L2
    - Iterative solvers: gmres(L, b), cg(L, b)

Usage:
    >>> from mfgarchon.operators import LaplacianOperator, PartialDerivOperator
    >>> L = LaplacianOperator(spacings=[0.1, 0.1], field_shape=(50, 50), bc=bc)
    >>> Lu = L(u)
    >>>
    >>> # Partial derivative
    >>> d_dx = PartialDerivOperator(direction=0, spacings=[0.1, 0.1], field_shape=(50, 50))
    >>> du_dx = d_dx(u)

Created: 2026-01-24 (Operator module separation from geometry)
"""

# Differential operators
from mfgarchon.operators.differential import (
    AdvectionOperator,
    DivergenceOperator,
    GradientComponentOperator,  # Deprecated alias for PartialDerivOperator
    InterfaceJumpOperator,
    LaplacianOperator,
    PartialDerivOperator,
)

# Interpolation operators
from mfgarchon.operators.interpolation import (
    GeometryProjector,
    InterpolationOperator,
    ProjectionRegistry,
)

# Reconstruction strategies
from mfgarchon.operators.reconstruction import (
    compute_weno5_derivative_1d,
    compute_weno5_godunov_upwind_1d,
)

# Low-level stencils
from mfgarchon.operators.stencils import (
    gradient_backward,
    gradient_central,
    gradient_forward,
    gradient_upwind,
    laplacian_stencil_1d,
    laplacian_stencil_nd,
)

__all__ = [
    # Differential operators
    "LaplacianOperator",
    "PartialDerivOperator",
    "GradientComponentOperator",  # Deprecated alias
    "DivergenceOperator",
    "AdvectionOperator",
    "InterfaceJumpOperator",
    # Interpolation
    "InterpolationOperator",
    "GeometryProjector",
    "ProjectionRegistry",
    # Reconstruction
    "compute_weno5_derivative_1d",
    "compute_weno5_godunov_upwind_1d",
    # Stencils (low-level)
    "gradient_central",
    "gradient_forward",
    "gradient_backward",
    "gradient_upwind",
    "laplacian_stencil_1d",
    "laplacian_stencil_nd",
]

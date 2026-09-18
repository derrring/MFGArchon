"""
Differential Operators for MFGarchon.

This module provides differential operators as scipy LinearOperator classes:
    - LaplacianOperator: Laplacian (second derivative)
    - DiffusionOperator: Unified diffusion operator (scalar or tensor)
    - PartialDerivOperator: Partial derivative (d/dxi)
    - DirectDerivOperator: Directional derivative (v . grad)
    - NormalDerivOperator: Normal derivative (d/dn = n . grad)
    - DivergenceOperator: Divergence of vector fields
    - AdvectionOperator: Advection/transport operator
    - InterfaceJumpOperator: Jump conditions across interfaces

All operators support:
    - scipy.sparse.linalg.LinearOperator interface
    - Callable interface: L(u) preserves field shape
    - Boundary condition handling

Usage:
    >>> from mfgarchon.operators.differential import LaplacianOperator, DiffusionOperator
    >>> L = LaplacianOperator(spacings=[0.1, 0.1], field_shape=(50, 50), bc=bc)
    >>> Lu = L(u)  # Callable interface
    >>> Lu_flat = L @ u.ravel()  # scipy interface
    >>>
    >>> # Unified diffusion (scalar or tensor coefficient)
    >>> D = DiffusionOperator(coefficient=sigma, spacings=[dx, dy],
    ...                       field_shape=(Nx, Ny), bc=bc)
    >>> Du = D(u)
    >>>
    >>> # Partial derivatives
    >>> from mfgarchon.operators.differential import PartialDerivOperator
    >>> d_dx = PartialDerivOperator(direction=0, spacings=[dx, dy], field_shape=(Nx, Ny))
    >>> du_dx = d_dx(u)
    >>>
    >>> # Normal derivative (Issue #658 Phase 2)
    >>> from mfgarchon.operators.differential import NormalDerivOperator
    >>> D_n = NormalDerivOperator.from_axis(axis=0, sign=-1, spacings=[dx], field_shape=(N,))
    >>> du_dn = D_n(u)  # Outward normal derivative
"""

from mfgarchon.operators.differential.advection import AdvectionOperator
from mfgarchon.operators.differential.diffusion import DiffusionOperator
from mfgarchon.operators.differential.directional import (
    DirectDerivOperator,
    NormalDerivOperator,
)
from mfgarchon.operators.differential.divergence import DivergenceOperator
from mfgarchon.operators.differential.function_gradient import (
    function_gradient,
    outward_normal_from_sdf,
)
from mfgarchon.operators.differential.gradient import (
    GradientOperator,
    PartialDerivOperator,
)
from mfgarchon.operators.differential.interface_jump import InterfaceJumpOperator
from mfgarchon.operators.differential.laplacian import LaplacianOperator

__all__ = [
    # Second-order operators
    "LaplacianOperator",
    "DiffusionOperator",
    # First-order operators (grid-based)
    "PartialDerivOperator",
    "GradientOperator",  # Full gradient ∇u (Issue #658 Phase 3)
    "DirectDerivOperator",
    "NormalDerivOperator",
    "DivergenceOperator",
    "AdvectionOperator",
    # Pointwise function gradient (Issue #662)
    "function_gradient",
    "outward_normal_from_sdf",
    # Interface operators
    "InterfaceJumpOperator",
]

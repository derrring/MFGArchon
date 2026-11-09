"""
Curvature Computation for Level Set Methods.

Mean curvature of the level set interface is computed via:
    κ = ∇·(∇φ / |∇φ|) = ∇·n

where n = ∇φ/|∇φ| is the unit normal field.

For geometric interpretation:
- κ > 0: Convex interface (bulging outward)
- κ < 0: Concave interface (bulging inward)
- κ = 1/R: Sphere of radius R

Applications:
- Curvature-dependent velocity: V = V₀ + ε·κ (surface tension effects)
- Mean curvature flow: ∂φ/∂t = κ|∇φ|
- Stefan problems with surface tension

Mathematical Background:
    For a curve in 2D or surface in 3D defined by φ = 0:
        κ = ∇·(∇φ/|∇φ|)
          = (φₓₓφᵧ² - 2φₓφᵧφₓᵧ + φᵧᵧφₓ²) / (φₓ² + φᵧ²)^{3/2}  (2D)

    But the divergence formula is dimension-agnostic and numerically stable.

References:
- Osher & Fedkiw (2003): Level Set Methods, Chapter 3
- Sethian (1999): Level Set Methods and Fast Marching Methods, Section 2.4

Created: 2026-01-18 (Issue #592 Milestone 3.1.3)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid

# Module logger
logger = get_logger(__name__)


def compute_curvature(
    phi: NDArray[np.float64],
    geometry: TensorProductGrid,
    epsilon: float = 1e-10,
) -> NDArray[np.float64]:
    """
    Compute mean curvature: κ = ∇·(∇φ/|∇φ|) = ∇·n.

    Uses geometry's gradient and divergence operators for dimension-agnostic
    computation. Works for 1D, 2D, 3D, and higher dimensions.

    Args:
        phi: Level set function, shape (Nx, Ny, ...) matching geometry
        geometry: Grid providing gradient and divergence operators
        epsilon: Regularization to avoid division by zero (default: 1e-10)
            Added to |∇φ| to prevent singularities at critical points

    Returns:
        Curvature field κ, same shape as phi

    Raises:
        AttributeError: If geometry doesn't support required operators

    Example:
        >>> # 2D circle: φ = ||x - c|| - R
        >>> phi_circle = np.linalg.norm(X - center, axis=0) - radius
        >>> kappa = compute_curvature(phi_circle, grid)
        >>> # On interface (φ ≈ 0): κ ≈ 1/R

    Note:
        Curvature is most accurate near the interface (|φ| ≈ 0) for signed
        distance functions. For best results, reinitialize φ before computing
        curvature to ensure |∇φ| = 1.
    """
    # Get operators from geometry (Issue #595 infrastructure)
    grad_ops = geometry.get_gradient_operator()
    div_op = geometry.get_divergence_operator()

    # Compute gradient components: ∇φ = (φₓ, φᵧ, ...)
    grad_phi = np.array([grad_op(phi) for grad_op in grad_ops])

    # Compute gradient magnitude: |∇φ|
    grad_mag = np.linalg.norm(grad_phi, axis=0) + epsilon

    # Compute unit normal field: n = ∇φ / |∇φ|
    # Shape: (dimension, Nx, Ny, ...)
    normal_field = grad_phi / grad_mag

    # Compute divergence of normal: κ = ∇·n
    curvature = div_op(normal_field)

    return curvature

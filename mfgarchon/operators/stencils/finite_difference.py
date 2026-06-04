"""
Finite Difference Stencils for MFGarchon.

This module provides low-level finite difference stencil implementations.
Stencils are the building blocks for differential operators - they define
the coefficient weights for approximating derivatives.

Conceptual Hierarchy:
    Stencils (this module)
        ↓ (fixed coefficients)
    Reconstruction (operators/reconstruction/)
        ↓ (adaptive weighting)
    Differential Operators (operators/differential/)
        ↓ (LinearOperator interface)
    Solvers (alg/numerical/)

Stencil Types:
    - CENTRAL: 2nd-order accurate, symmetric, no directional bias
    - FORWARD: 1st-order accurate, uses u[i+1] - u[i]
    - BACKWARD: 1st-order accurate, uses u[i] - u[i-1]
    - UPWIND: Godunov selection based on flow direction (stable for advection)
    - ONE_SIDED: Boundary handling with forward/backward at edges

Mathematical Background:
    Central:   ∂u/∂x ≈ (u[i+1] - u[i-1]) / (2h)     Error: O(h²)
    Forward:   ∂u/∂x ≈ (u[i+1] - u[i]) / h          Error: O(h)
    Backward:  ∂u/∂x ≈ (u[i] - u[i-1]) / h          Error: O(h)

Usage:
    >>> from mfgarchon.operators.stencils import gradient_central, gradient_upwind
    >>> du_dx = gradient_central(u, axis=0, h=0.1)

Note:
    These functions operate on arrays directly using np.roll for periodic wrapping.
    For boundary-aware operators, use the LinearOperator classes in operators/differential/.

Created: 2026-01-24 (Operator module reorganization)
Extracted from: mfgarchon/utils/numerical/tensor_calculus.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.geometry.boundary import BoundaryConditions


def _roll(xp: type, u: NDArray, shift: int, axis: int) -> NDArray:
    """Backend-aware periodic shift.

    ``numpy``/``cupy`` use the ``axis=`` keyword; ``torch.roll`` uses ``dims=``.
    Detecting the module by name keeps the numpy/cupy path byte-identical
    while routing torch tensors to the correct keyword (Issue #1194).
    """
    if xp.__name__ == "torch":
        return xp.roll(u, shift, dims=axis)
    return xp.roll(u, shift, axis=axis)


# =============================================================================
# First-Order Derivative Stencils
# =============================================================================


def gradient_central(u: NDArray, axis: int, h: float, xp: type = np) -> NDArray:
    """
    Central difference approximation for first derivative.

    Formula: ∂u/∂x ≈ (u[i+1] - u[i-1]) / (2h)

    Properties:
        - 2nd-order accurate: O(h²)
        - Symmetric stencil: [-1, 0, 1] / (2h)
        - No directional bias (good for elliptic problems)
        - May oscillate near discontinuities

    Args:
        u: Input array
        axis: Axis along which to differentiate
        h: Grid spacing
        xp: Array module (numpy or cupy for GPU)

    Returns:
        Approximation of ∂u/∂x with same shape as u

    Note:
        Uses periodic wrapping via np.roll. For non-periodic boundaries,
        use fix_boundaries_one_sided() or the LinearOperator classes.
    """
    return (_roll(xp, u, -1, axis) - _roll(xp, u, 1, axis)) / (2 * h)


def gradient_forward(u: NDArray, axis: int, h: float, xp: type = np) -> NDArray:
    """
    Forward difference approximation for first derivative.

    Formula: ∂u/∂x ≈ (u[i+1] - u[i]) / h

    Properties:
        - 1st-order accurate: O(h)
        - Stencil: [-1, 1] / h
        - Biased in positive direction
        - Stable for advection with negative velocity (v < 0)

    Args:
        u: Input array
        axis: Axis along which to differentiate
        h: Grid spacing
        xp: Array module (numpy or cupy for GPU)

    Returns:
        Approximation of ∂u/∂x with same shape as u
    """
    return (_roll(xp, u, -1, axis) - u) / h


def gradient_backward(u: NDArray, axis: int, h: float, xp: type = np) -> NDArray:
    """
    Backward difference approximation for first derivative.

    Formula: ∂u/∂x ≈ (u[i] - u[i-1]) / h

    Properties:
        - 1st-order accurate: O(h)
        - Stencil: [1, -1] / h (shifted)
        - Biased in negative direction
        - Stable for advection with positive velocity (v > 0)

    Args:
        u: Input array
        axis: Axis along which to differentiate
        h: Grid spacing
        xp: Array module (numpy or cupy for GPU)

    Returns:
        Approximation of ∂u/∂x with same shape as u
    """
    return (u - _roll(xp, u, 1, axis)) / h


def gradient_upwind(u: NDArray, axis: int, h: float, xp: type = np) -> NDArray:
    """
    Godunov upwind scheme for first derivative.

    Selects forward or backward difference based on local flow direction
    (sign of central gradient). This provides numerical stability for
    advection-dominated problems.

    Selection rule:
        If ∂u/∂x >= 0: use backward difference (information flows right)
        If ∂u/∂x < 0:  use forward difference (information flows left)

    Properties:
        - 1st-order accurate: O(h)
        - Automatically selects stable direction
        - Introduces numerical diffusion (stabilizing)
        - Essential for hyperbolic PDEs (HJB, transport)

    Args:
        u: Input array
        axis: Axis along which to differentiate
        h: Grid spacing
        xp: Array module (numpy or cupy for GPU)

    Returns:
        Approximation of ∂u/∂x with same shape as u

    Note:
        This is the Godunov flux for scalar conservation laws.
        For WENO/ENO reconstruction, see operators/reconstruction/.
    """
    grad_forward = gradient_forward(u, axis, h, xp)
    grad_backward = gradient_backward(u, axis, h, xp)
    grad_central = (grad_forward + grad_backward) / 2.0
    return xp.where(grad_central >= 0, grad_backward, grad_forward)


# =============================================================================
# Boundary Handling
# =============================================================================


def fix_boundaries_one_sided(grad: NDArray, u: NDArray, axis: int, h: float, xp: type = np) -> NDArray:
    """
    Replace boundary values with one-sided differences.

    Central differences wrap around at boundaries (via np.roll), which is
    incorrect for non-periodic BCs. This function fixes the boundary values
    using appropriate one-sided stencils.

    Boundary corrections:
        Left (i=0):  use forward difference
        Right (i=-1): use backward difference

    Args:
        grad: Gradient array computed with central differences
        u: Original input array
        axis: Axis along which gradient was computed
        h: Grid spacing
        xp: Array module (numpy or cupy for GPU)

    Returns:
        Gradient array with corrected boundary values (modified in-place)

    Example:
        >>> grad = gradient_central(u, axis=0, h=dx)
        >>> grad = fix_boundaries_one_sided(grad, u, axis=0, h=dx)
    """
    ndim = u.ndim

    # Left boundary: forward difference
    left_slice = [slice(None)] * ndim
    left_slice[axis] = 0
    next_slice = [slice(None)] * ndim
    next_slice[axis] = 1
    grad[tuple(left_slice)] = (u[tuple(next_slice)] - u[tuple(left_slice)]) / h

    # Right boundary: backward difference
    right_slice = [slice(None)] * ndim
    right_slice[axis] = -1
    prev_slice = [slice(None)] * ndim
    prev_slice[axis] = -2
    grad[tuple(right_slice)] = (u[tuple(right_slice)] - u[tuple(prev_slice)]) / h

    return grad


# =============================================================================
# Second-Order Derivative Stencils
# =============================================================================


def laplacian_stencil_1d(u: NDArray, h: float, xp: type = np) -> NDArray:
    """
    Standard 3-point Laplacian stencil in 1D.

    Formula: ∂²u/∂x² ≈ (u[i+1] - 2u[i] + u[i-1]) / h²

    Properties:
        - 2nd-order accurate: O(h²)
        - Stencil: [1, -2, 1] / h²
        - Symmetric, negative semi-definite

    Args:
        u: Input 1D array
        h: Grid spacing
        xp: Array module

    Returns:
        Approximation of ∂²u/∂x²
    """
    return (xp.roll(u, -1) - 2 * u + xp.roll(u, 1)) / (h * h)


def laplacian_stencil_nd(u: NDArray, spacings: list[float] | tuple[float, ...], xp: type = np) -> NDArray:
    """
    Standard Laplacian stencil in n dimensions.

    Formula: Δu = Σ_d ∂²u/∂x_d²

    Uses the 3-point stencil in each dimension and sums.

    Args:
        u: Input n-dimensional array
        spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
        xp: Array module

    Returns:
        Approximation of Δu = ∇²u
    """
    result = xp.zeros_like(u)
    for axis, h in enumerate(spacings):
        result += (_roll(xp, u, -1, axis) - 2 * u + _roll(xp, u, 1, axis)) / (h * h)
    return result


def weighted_laplacian_stencil_nd(
    u: NDArray,
    spacings: list[float] | tuple[float, ...],
    axis_weights: NDArray,
    xp: type = np,
) -> NDArray:
    """
    Weighted Laplacian stencil in n dimensions.

    Formula: Σ_d w_d * ∂²u/∂x_d²

    For diagonal diffusion tensor Σ = diag(σ₀², σ₁², ...),
    computes the anisotropic diffusion: Σ_d σ_d² ∂²u/∂x_d².

    Args:
        u: Input n-dimensional array
        spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
        axis_weights: Per-axis weights, shape (ndim,)
        xp: Array module

    Returns:
        Weighted Laplacian approximation
    """
    result = xp.zeros_like(u)
    for axis, (h, w) in enumerate(zip(spacings, axis_weights, strict=True)):
        result += w * (_roll(xp, u, -1, axis) - 2 * u + _roll(xp, u, 1, axis)) / (h * h)
    return result


# =============================================================================
# Stencil Coefficients (for matrix assembly)
# =============================================================================


def get_gradient_stencil_coefficients(scheme: str, h: float) -> tuple[list[int], list[float]]:
    """
    Get stencil offsets and coefficients for gradient approximation.

    Useful for building sparse matrices where you need explicit coefficients.

    Args:
        scheme: One of "central", "forward", "backward"
        h: Grid spacing

    Returns:
        Tuple of (offsets, coefficients) where:
            offsets: List of index offsets from center [e.g., [-1, 1]]
            coefficients: List of weights [e.g., [-0.5/h, 0.5/h]]

    Example:
        >>> offsets, coeffs = get_gradient_stencil_coefficients("central", h=0.1)
        >>> # offsets = [-1, 1], coeffs = [-5.0, 5.0]
    """
    if scheme == "central":
        return [-1, 1], [-1.0 / (2 * h), 1.0 / (2 * h)]
    elif scheme == "forward":
        return [0, 1], [-1.0 / h, 1.0 / h]
    elif scheme == "backward":
        return [-1, 0], [-1.0 / h, 1.0 / h]
    else:
        raise ValueError(f"Unknown scheme: {scheme}. Use 'central', 'forward', 'backward'")


def get_laplacian_stencil_coefficients(h: float) -> tuple[list[int], list[float]]:
    """
    Get stencil offsets and coefficients for 1D Laplacian.

    Args:
        h: Grid spacing

    Returns:
        Tuple of (offsets, coefficients)
        offsets = [-1, 0, 1], coeffs = [1/h², -2/h², 1/h²]
    """
    h2 = h * h
    return [-1, 0, 1], [1.0 / h2, -2.0 / h2, 1.0 / h2]


def gradient_nd(
    u: NDArray,
    spacings: list[float] | tuple[float, ...],
    xp: type = np,
) -> list[NDArray]:
    """
    Compute gradient in all dimensions using central differences.

    This is a simple wrapper that applies gradient_central to each dimension.
    No boundary condition handling - uses periodic wrapping via np.roll.

    Replaces tensor_calculus.gradient_simple for particle methods where
    BC is handled separately.

    Args:
        u: Input n-dimensional array
        spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
        xp: Array module (numpy or cupy for GPU)

    Returns:
        List of gradient arrays, one per dimension: [∂u/∂x₀, ∂u/∂x₁, ...]

    Example:
        >>> u = np.sin(np.linspace(0, 2*np.pi, 100))
        >>> grad = gradient_nd(u, spacings=[0.1])
        >>> # grad[0] contains ∂u/∂x

    Note:
        Issue #625: Added to replace tensor_calculus.gradient_simple
    """
    gradients = []
    for axis, h in enumerate(spacings):
        if h > 1e-14:
            gradients.append(gradient_central(u, axis=axis, h=h, xp=xp))
        else:
            gradients.append(xp.zeros_like(u))
    return gradients


def laplacian_with_bc(
    u: NDArray,
    spacings: list[float] | tuple[float, ...],
    bc: BoundaryConditions | None = None,
    time: float = 0.0,
) -> NDArray:
    """
    Compute Laplacian with boundary condition handling.

    This function combines ghost cell padding from geometry/boundary
    with the laplacian_stencil_nd computation.

    Replaces tensor_calculus.laplacian for cases needing BC-aware Laplacian.

    Args:
        u: Input n-dimensional array
        spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
        bc: Boundary conditions. If None, uses periodic BC.
        time: Current time for time-dependent BCs

    Returns:
        Laplacian of u with same shape as input

    Example:
        >>> from mfgarchon.geometry.boundary import neumann_bc
        >>> u = np.sin(np.linspace(0, np.pi, 50))
        >>> lap_u = laplacian_with_bc(u, spacings=[0.1], bc=neumann_bc(1))

    Note:
        Issue #625: Added to replace tensor_calculus.laplacian
    """
    from mfgarchon.geometry.boundary import pad_array_with_ghosts

    # Apply ghost cells if BC provided
    if bc is not None:
        u_work = pad_array_with_ghosts(u, bc, ghost_depth=1, time=time)
    else:
        u_work = u

    # Compute laplacian
    lap = laplacian_stencil_nd(u_work, spacings)

    # Extract interior if ghost cells were added
    if bc is not None:
        slices = [slice(1, -1)] * u.ndim
        lap = lap[tuple(slices)]

    return lap


def weighted_laplacian_with_bc(
    u: NDArray,
    spacings: list[float] | tuple[float, ...],
    axis_weights: NDArray,
    bc: BoundaryConditions | None = None,
    time: float = 0.0,
) -> NDArray:
    """
    Compute weighted Laplacian with boundary condition handling.

    For diagonal diffusion tensor Sigma = diag(sigma_0^2, sigma_1^2, ...),
    computes: sum_d sigma_d^2 * d^2u/dx_d^2

    This is the anisotropic analogue of laplacian_with_bc().

    Args:
        u: Input n-dimensional array
        spacings: Grid spacing per dimension [h_0, h_1, ..., h_{d-1}]
        axis_weights: Per-axis weights (e.g., diagonal of diffusion tensor), shape (ndim,)
        bc: Boundary conditions. If None, uses periodic BC.
        time: Current time for time-dependent BCs

    Returns:
        Weighted Laplacian of u with same shape as input
    """
    from mfgarchon.geometry.boundary import pad_array_with_ghosts

    # Apply ghost cells if BC provided
    if bc is not None:
        u_work = pad_array_with_ghosts(u, bc, ghost_depth=1, time=time)
    else:
        u_work = u

    # Compute weighted laplacian
    lap = weighted_laplacian_stencil_nd(u_work, spacings, axis_weights)

    # Extract interior if ghost cells were added
    if bc is not None:
        slices = [slice(1, -1)] * u.ndim
        lap = lap[tuple(slices)]

    return lap


__all__ = [
    # First-order derivatives
    "gradient_central",
    "gradient_forward",
    "gradient_backward",
    "gradient_upwind",
    "gradient_nd",
    # Boundary handling
    "fix_boundaries_one_sided",
    # Second-order derivatives
    "laplacian_stencil_1d",
    "laplacian_stencil_nd",
    "laplacian_with_bc",
    "weighted_laplacian_stencil_nd",
    "weighted_laplacian_with_bc",
    # Coefficients for matrix assembly
    "get_gradient_stencil_coefficients",
    "get_laplacian_stencil_coefficients",
]

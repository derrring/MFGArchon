"""
WENO5 gradient computation for operator framework.

Provides 5th-order accurate one-sided derivatives using WENO5 reconstruction.
This module is integrated into the GradientOperator framework, making WENO5
available to all solvers (HJB, FP, level set) through the standard operator interface.

Mathematical Background:
    WENO5 (Weighted Essentially Non-Oscillatory 5th-order) scheme:
    - Uses 5-point stencil for 5th-order spatial accuracy
    - Three candidate 3-point stencils weighted by smoothness
    - Automatic upwind bias selection for shock-capturing
    - Preserves monotonicity near discontinuities

References:
    - Jiang & Shu (1996): Efficient Implementation of Weighted ENO Schemes
    - Osher & Fedkiw (2003): Level Set Methods, Chapter 6

Created: 2026-01-18 (Issue #606 - WENO5 Operator Refactoring)
Refactored from: mfgarchon/alg/numerical/weno/weno5_gradients.py (Issue #605)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def compute_weno5_derivative_1d(
    u: NDArray[np.float64],
    spacing: float,
    bias: str = "left",
    epsilon: float = 1e-6,
) -> NDArray[np.float64]:
    """
    Compute derivative using WENO5 reconstruction in 1D.

    Parameters
    ----------
    u : NDArray
        Field to differentiate (1D array).
    spacing : float
        Grid spacing (dx).
    bias : str, default="left"
        Upwind bias direction:
        - "left": Left-biased (for positive velocity)
        - "right": Right-biased (for negative velocity)
    epsilon : float, default=1e-6
        Smoothness indicator regularization (prevents division by zero).

    Returns
    -------
    du : NDArray
        Derivative of u, same shape as input.

    Notes
    -----
    **WENO5 Scheme** (Jiang & Shu 1996):
    - Interior points (i=2 to N-3): Full 5th-order WENO5
    - Boundary points (i<2 or i>N-3): Fallback to 1st-order upwind

    **Algorithm**:
    1. Compute smoothness indicators β_k for each of 3 candidate stencils
    2. Compute nonlinear weights ω_k based on smoothness
    3. Weighted combination of 3rd-order stencil derivatives

    **Bias Selection**:
    - "left": Favors left-biased stencil (d₀=0.1, d₁=0.6, d₂=0.3)
    - "right": Favors right-biased stencil (d₀=0.3, d₁=0.6, d₂=0.1)

    References
    ----------
    Jiang & Shu (1996): Efficient Implementation of Weighted ENO Schemes

    Examples
    --------
    >>> # Compute left-biased derivative
    >>> x = np.linspace(0, 1, 100)
    >>> u = np.sin(2 * np.pi * x)
    >>> du_dx = compute_weno5_derivative_1d(u, spacing=x[1]-x[0], bias="left")
    >>>
    >>> # For upwind selection in level set evolution
    >>> velocity = np.ones_like(x)
    >>> bias = "left" if velocity[0] > 0 else "right"
    >>> du_upwind = compute_weno5_derivative_1d(u, spacing=dx, bias=bias)
    """
    N = len(u)
    du = np.zeros_like(u)

    # Set ideal weights based on bias
    if bias == "left":
        d0, d1, d2 = 0.1, 0.6, 0.3
    elif bias == "right":
        d0, d1, d2 = 0.3, 0.6, 0.1
    else:
        raise ValueError(f"Invalid bias '{bias}', must be 'left' or 'right'")

    # Interior points: WENO5 reconstruction (i = 2, ..., N-3)
    for i in range(2, N - 2):
        du[i] = _weno5_derivative_at_point(u, i, spacing, d0, d1, d2, epsilon)

    # Boundary points: fallback to lower-order upwind
    # Left boundary (i = 0, 1)
    du[0] = (u[1] - u[0]) / spacing  # 1st-order forward
    du[1] = (u[1] - u[0]) / spacing  # 1st-order backward

    # Right boundary (i = N-2, N-1)
    du[N - 2] = (u[N - 1] - u[N - 2]) / spacing
    du[N - 1] = (u[N - 1] - u[N - 2]) / spacing

    return du


def _weno5_derivative_at_point(
    u: NDArray[np.float64],
    i: int,
    dx: float,
    d0: float,
    d1: float,
    d2: float,
    epsilon: float,
) -> float:
    """
    Compute WENO5 derivative at single interior point.

    Parameters
    ----------
    u : NDArray
        Field values.
    i : int
        Point index (must satisfy 2 <= i <= N-3).
    dx : float
        Grid spacing.
    d0, d1, d2 : float
        Ideal weights for three stencils.
    epsilon : float
        Smoothness indicator regularization.

    Returns
    -------
    du_i : float
        WENO5 derivative at point i.

    Notes
    -----
    Uses stencil: u[i-2], u[i-1], u[i], u[i+1], u[i+2]

    Three candidate derivative approximations (2nd order each):
    - Stencil 0: left-biased (i-2, i-1, i)
    - Stencil 1: centered (i-1, i, i+1)
    - Stencil 2: right-biased (i, i+1, i+2)
    """
    # Three candidate derivative approximations (2nd order each)
    # Stencil 0: left-biased (i-2, i-1, i)
    q0 = (3 * u[i] - 4 * u[i - 1] + u[i - 2]) / (2 * dx)

    # Stencil 1: centered (i-1, i, i+1)
    q1 = (u[i + 1] - u[i - 1]) / (2 * dx)

    # Stencil 2: right-biased (i, i+1, i+2)
    q2 = (-u[i + 2] + 4 * u[i + 1] - 3 * u[i]) / (2 * dx)

    # Smoothness indicators (measure solution variation)
    # Normalized by dx^2 for scale invariance
    beta0 = (13 / 12) * ((u[i - 2] - 2 * u[i - 1] + u[i]) / dx) ** 2 + (1 / 4) * (
        (u[i - 2] - 4 * u[i - 1] + 3 * u[i]) / dx
    ) ** 2

    beta1 = (13 / 12) * ((u[i - 1] - 2 * u[i] + u[i + 1]) / dx) ** 2 + (1 / 4) * ((u[i - 1] - u[i + 1]) / dx) ** 2

    beta2 = (13 / 12) * ((u[i] - 2 * u[i + 1] + u[i + 2]) / dx) ** 2 + (1 / 4) * (
        (3 * u[i] - 4 * u[i + 1] + u[i + 2]) / dx
    ) ** 2

    # Nonlinear weights (favor smooth stencils)
    alpha0 = d0 / (epsilon + beta0) ** 2
    alpha1 = d1 / (epsilon + beta1) ** 2
    alpha2 = d2 / (epsilon + beta2) ** 2

    alpha_sum = alpha0 + alpha1 + alpha2

    omega0 = alpha0 / alpha_sum
    omega1 = alpha1 / alpha_sum
    omega2 = alpha2 / alpha_sum

    # WENO5 derivative: weighted combination
    return omega0 * q0 + omega1 * q1 + omega2 * q2


def compute_weno5_godunov_upwind_1d(
    u: NDArray[np.float64],
    velocity: NDArray[np.float64],
    spacing: float,
    epsilon: float = 1e-6,
) -> NDArray[np.float64]:
    """
    Compute Godunov upwind derivative using WENO5 for 1D level set evolution.

    Parameters
    ----------
    u : NDArray
        Field to differentiate (1D array).
    velocity : NDArray
        Velocity field (same shape as u).
    spacing : float
        Grid spacing (dx).
    epsilon : float, default=1e-6
        Smoothness indicator regularization.

    Returns
    -------
    du_upwind : NDArray
        Upwind derivative: du^+ if V>0, du^- if V<0.

    Notes
    -----
    **Godunov Upwind Selection**:
        du = du^+ if V > 0 else du^-

    This is the standard upwind flux selection for level set evolution:
        ∂φ/∂t + V|∇φ| = 0

    Examples
    --------
    >>> # Level set evolution with variable velocity
    >>> phi = x - 0.5
    >>> velocity = np.sin(2 * np.pi * x)
    >>> dphi_upwind = compute_weno5_godunov_upwind_1d(phi, velocity, dx)
    """
    # Compute both one-sided derivatives
    du_left = compute_weno5_derivative_1d(u, spacing, bias="left", epsilon=epsilon)
    du_right = compute_weno5_derivative_1d(u, spacing, bias="right", epsilon=epsilon)

    # Godunov upwind selection
    du_upwind = np.where(velocity > 0, du_left, du_right)

    return du_upwind


def compute_weno5_derivative_nd(
    u: NDArray[np.float64],
    spacings: Sequence[float],
    axis: int,
    bias: str = "left",
    epsilon: float = 1e-6,
) -> NDArray[np.float64]:
    """
    Compute WENO5 derivative along a single axis of an nD array.

    For 1D input, delegates directly to compute_weno5_derivative_1d.
    For nD input, swaps the target axis to position 0, applies 1D WENO5
    to each transverse slice, then swaps back.

    Parameters
    ----------
    u : NDArray
        Field to differentiate (1D, 2D, or 3D array).
    spacings : Sequence[float]
        Grid spacing per dimension, e.g. (dx, dy) for 2D.
    axis : int
        Axis along which to differentiate (0=x, 1=y, ...).
    bias : str, default="left"
        Upwind bias direction ("left" or "right").
    epsilon : float, default=1e-6
        Smoothness indicator regularization.

    Returns
    -------
    du : NDArray
        Derivative along the specified axis, same shape as input.

    Examples
    --------
    >>> # 2D derivative along x-axis
    >>> u = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    >>> du_dx = compute_weno5_derivative_nd(u, spacings=(dx, dy), axis=0, bias="left")
    """
    ndim = u.ndim

    if len(spacings) != ndim:
        raise ValueError(f"len(spacings)={len(spacings)} must match array ndim={ndim}")

    if axis < 0 or axis >= ndim:
        raise ValueError(f"axis={axis} out of range for {ndim}D array")

    # 1D: delegate directly
    if ndim == 1:
        return compute_weno5_derivative_1d(u, spacings[0], bias=bias, epsilon=epsilon)

    dx = spacings[axis]

    # Move target axis to position 0 for uniform iteration
    u_swapped = np.moveaxis(u, axis, 0)
    result_swapped = np.zeros_like(u_swapped)

    # Iterate over all transverse slices (product of remaining axes)
    # u_swapped has shape (N_axis, *remaining_shape)
    remaining_shape = u_swapped.shape[1:]
    for idx in np.ndindex(*remaining_shape):
        # Extract 1D slice along the target axis
        slc = (slice(None), *idx)
        result_swapped[slc] = compute_weno5_derivative_1d(u_swapped[slc], dx, bias=bias, epsilon=epsilon)

    # Move axis back to original position
    return np.moveaxis(result_swapped, 0, axis)

"""
Shared BC Value Enforcement Utilities (Issue #636).

This module provides dimension-agnostic utilities for enforcing boundary
condition values on solution arrays. These utilities are used by multiple
applicators (FDM, Interpolation, etc.) to avoid code duplication.

The key distinction from ghost cell methods:
- **Ghost cells** (FDMApplicator.apply): Pad array for stencil operations
- **Value enforcement** (this module): Set boundary values to satisfy BC

Supported BC Types:
- Neumann (du/dn = g): Extrapolation-based enforcement
- Dirichlet (u = g): Direct value assignment
- Periodic: Identify the two coincident endpoints (endpoint-inclusive grid; see
  enforce_periodic_value_nd for why this is not a copy from the opposite interior)

Usage:
    >>> from mfgarchon.geometry.boundary.enforcement import (
    ...     enforce_neumann_value_nd,
    ...     enforce_dirichlet_value_nd,
    ... )
    >>>
    >>> # Enforce Neumann BC with 2nd-order extrapolation
    >>> enforce_neumann_value_nd(field, axis=0, side="min", order=2)
    >>>
    >>> # Enforce Dirichlet BC
    >>> enforce_dirichlet_value_nd(field, axis=0, side="min", value=0.0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def enforce_neumann_value_nd(
    field: NDArray[np.floating],
    axis: int,
    side: Literal["min", "max"],
    grad_value: float = 0.0,
    spacing: float = 1.0,
    order: int = 2,
) -> None:
    """
    Enforce Neumann BC value along specified axis (in-place).

    Sets boundary value to satisfy the gradient constraint du/dn = g.

    Args:
        field: Solution array (modified in-place)
        axis: Dimension index (0, 1, 2, ...)
        side: "min" or "max" boundary
        grad_value: Neumann BC gradient value (default: 0.0 for zero-flux)
        spacing: Grid spacing h (needed for non-zero gradient)
        order: Extrapolation order for zero-flux case
            - order=1: u[0] = u[1] (O(h) accurate, simple copy)
            - order=2: u[0] = (4*u[1] - u[2])/3 (O(h²) accurate)

    Formulas:
        For grad_value = 0 (zero-flux):
            - order=1: u[boundary] = u[neighbor]
            - order=2: u[boundary] = (4*u[neighbor] - u[next])/3

        For grad_value != 0:
            - min side: u[0] = u[1] - g*h
            - max side: u[-1] = u[-2] + g*h

    Example:
        >>> field = np.array([0.9, 1.0, 1.1, 1.2, 1.3])
        >>> enforce_neumann_value_nd(field, axis=0, side="min", order=2)
        >>> # field[0] is now (4*1.0 - 1.1)/3 = 0.9667
    """
    ndim = field.ndim
    n_points = field.shape[axis]

    # Build slicers
    boundary_slicer = [slice(None)] * ndim
    neighbor_slicer = [slice(None)] * ndim
    next_slicer = [slice(None)] * ndim

    if side == "min":
        boundary_slicer[axis] = 0
        neighbor_slicer[axis] = 1
        next_slicer[axis] = 2
    else:  # "max"
        boundary_slicer[axis] = -1
        neighbor_slicer[axis] = -2
        next_slicer[axis] = -3

    boundary_slicer = tuple(boundary_slicer)
    neighbor_slicer = tuple(neighbor_slicer)
    next_slicer = tuple(next_slicer)

    # Check if zero-flux (use extrapolation) or non-zero gradient
    if np.isclose(grad_value, 0.0):
        # Zero-flux: use extrapolation
        if order >= 2 and n_points >= 3:
            # 2nd-order: u[0] = (4*u[1] - u[2])/3
            field[boundary_slicer] = (4.0 * field[neighbor_slicer] - field[next_slicer]) / 3.0
        else:
            # 1st-order: u[0] = u[1]
            field[boundary_slicer] = field[neighbor_slicer]
    else:
        # Non-zero gradient: u[0] = u[1] - g*h (min) or u[-1] = u[-2] + g*h (max)
        if side == "min":
            field[boundary_slicer] = field[neighbor_slicer] - grad_value * spacing
        else:
            field[boundary_slicer] = field[neighbor_slicer] + grad_value * spacing


def enforce_dirichlet_value_nd(
    field: NDArray[np.floating],
    axis: int,
    side: Literal["min", "max"],
    value: float,
) -> None:
    """
    Enforce Dirichlet BC value along specified axis (in-place).

    Sets boundary value directly: u(boundary) = g.

    Args:
        field: Solution array (modified in-place)
        axis: Dimension index (0, 1, 2, ...)
        side: "min" or "max" boundary
        value: Dirichlet boundary value

    Example:
        >>> field = np.array([0.9, 1.0, 1.1, 1.2, 1.3])
        >>> enforce_dirichlet_value_nd(field, axis=0, side="min", value=0.0)
        >>> # field[0] is now 0.0
    """
    ndim = field.ndim
    slicer = [slice(None)] * ndim

    if side == "min":
        slicer[axis] = 0
    else:  # "max"
        slicer[axis] = -1

    field[tuple(slicer)] = value


def enforce_periodic_value_nd(
    field: NDArray[np.floating],
    axis: int,
) -> None:
    """
    Enforce periodic BC along ``axis`` (in-place) on an endpoint-inclusive grid.

    ``TensorProductGrid`` builds ``np.linspace(x_min, x_max, N)``, so ``field[0]`` and
    ``field[-1]`` are BOTH real nodes AND the same physical point: the period is ``L``, not
    ``L + dx``. They are one degree of freedom that the scheme has computed twice, so enforcement
    is the projection that makes them agree -- their mean, which privileges neither end.

    Do NOT restore the halo form ``field[0] = field[-2]; field[-1] = field[1]``. That is correct
    only for an array carrying one ghost cell per side, where index 0 and -1 are duplicates of the
    opposite interior. Applied to a halo-free grid it moves both endpoints one cell in opposite
    directions and destroys the periodicity it claims to enforce: on ``sin(2 pi x)`` with N=21 it
    took a seam of 2.4e-16 to 6.2e-01 (Issue #1820). No caller in this package passes a haloed
    array; if one ever does, it needs its own function rather than a reinterpretation of this one.

    Args:
        field: Solution array (modified in-place)
        axis: Dimension index (0, 1, 2, ...)

    Example:
        >>> field = np.array([0.0, 1.0, 2.0, 3.0, 0.5])
        >>> enforce_periodic_value_nd(field, axis=0)
        >>> # field[0] == field[-1] == 0.25
    """
    ndim = field.ndim

    first = [slice(None)] * ndim
    last = [slice(None)] * ndim
    first[axis] = 0
    last[axis] = -1

    identified = 0.5 * (field[tuple(first)] + field[tuple(last)])
    field[tuple(first)] = identified
    field[tuple(last)] = identified


def enforce_robin_value_nd(
    field: NDArray[np.floating],
    axis: int,
    side: Literal["min", "max"],
    alpha: float,
    beta: float,
    rhs_value: float,
    spacing: float = 1.0,
) -> None:
    """
    Enforce Robin BC value along specified axis (in-place).

    Robin BC: alpha*u + beta*du/dn = g

    For the special case beta=0 (pure Dirichlet): u = g/alpha
    For the special case alpha=0 (pure Neumann): du/dn = g/beta

    General case uses extrapolation-based enforcement.

    Args:
        field: Solution array (modified in-place)
        axis: Dimension index (0, 1, 2, ...)
        side: "min" or "max" boundary
        alpha: Coefficient on u
        beta: Coefficient on du/dn
        rhs_value: Right-hand side value g
        spacing: Grid spacing h

    Example:
        >>> # Robin BC: u + 0.5*du/dn = 1.0
        >>> enforce_robin_value_nd(field, axis=0, side="min", alpha=1.0, beta=0.5, rhs_value=1.0)
    """
    # Handle special cases
    if np.isclose(beta, 0.0):
        # Pure Dirichlet: alpha*u = g => u = g/alpha
        if not np.isclose(alpha, 0.0):
            enforce_dirichlet_value_nd(field, axis, side, rhs_value / alpha)
        return

    if np.isclose(alpha, 0.0):
        # Pure Neumann: beta*du/dn = g => du/dn = g/beta
        enforce_neumann_value_nd(field, axis, side, rhs_value / beta, spacing, order=1)
        return

    # General Robin case: use 2nd-order extrapolation with Robin constraint
    # For simplicity, fall back to Neumann-like extrapolation
    # (More accurate Robin enforcement would require solving a system)
    ndim = field.ndim
    n_points = field.shape[axis]

    boundary_slicer = [slice(None)] * ndim
    neighbor_slicer = [slice(None)] * ndim
    next_slicer = [slice(None)] * ndim

    if side == "min":
        boundary_slicer[axis] = 0
        neighbor_slicer[axis] = 1
        next_slicer[axis] = 2
    else:
        boundary_slicer[axis] = -1
        neighbor_slicer[axis] = -2
        next_slicer[axis] = -3

    boundary_slicer = tuple(boundary_slicer)
    neighbor_slicer = tuple(neighbor_slicer)
    next_slicer = tuple(next_slicer)

    if n_points >= 3:
        # 2nd-order extrapolation
        field[boundary_slicer] = (4.0 * field[neighbor_slicer] - field[next_slicer]) / 3.0
    else:
        field[boundary_slicer] = field[neighbor_slicer]


__all__ = [
    "enforce_neumann_value_nd",
    "enforce_dirichlet_value_nd",
    "enforce_periodic_value_nd",
    "enforce_robin_value_nd",
]

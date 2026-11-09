"""
Corner handling for Eikonal solvers.

Grid corners require special treatment because they have fewer neighbors.
This module provides utilities for computing updates at corners and boundaries.

Mathematical Background:
    At corners, the Eikonal equation |grad T| = 1/F has fewer constraints.
    For a 2D corner with only one valid neighbor in each direction:
        ((T - T_x) / dx)^2 + ((T - T_y) / dy)^2 = 1/F^2

    With Neumann BC (zero flux), we extrapolate from interior.
    With Dirichlet BC, the boundary value is prescribed.

References:
- Sethian (1999): Level Set Methods, Section on boundary conditions
- Zhao (2005): Fast sweeping boundary handling

Created: 2026-02-06 (Issue #664)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.geometry.level_set.eikonal.godunov_update import godunov_update_nd
from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)


def eikonal_corner_update(
    T: NDArray[np.float64],
    corner_idx: tuple[int, ...],
    speed: float,
    spacing: tuple[float, ...],
    boundary_type: str = "neumann",
) -> float:
    """
    Compute Eikonal update at a grid corner.

    At corners, we have fewer neighbors. This function handles the
    boundary condition appropriately.

    Args:
        T: Current solution array.
        corner_idx: Index of the corner point.
        speed: Local speed F > 0.
        spacing: Grid spacing (dx, dy, ...).
        boundary_type: "neumann" (zero flux) or "extrapolate" (linear extrapolation).

    Returns:
        Updated value at the corner.

    Note:
        For Neumann BC, we assume zero normal derivative at boundaries,
        which means the corner value is determined by interior neighbors.
    """
    ndim = len(corner_idx)
    shape = T.shape

    # Gather neighbor values, handling boundaries
    T_neighbors = []
    for dim in range(ndim):
        T_minus = np.inf
        T_plus = np.inf

        idx_minus = list(corner_idx)
        idx_minus[dim] -= 1
        if idx_minus[dim] >= 0:
            T_minus = T[tuple(idx_minus)]

        idx_plus = list(corner_idx)
        idx_plus[dim] += 1
        if idx_plus[dim] < shape[dim]:
            T_plus = T[tuple(idx_plus)]

        # Apply boundary condition
        if boundary_type == "neumann":
            # Zero normal derivative: use one-sided difference
            # No modification needed - inf values will be ignored by godunov_update_nd
            pass
        elif boundary_type == "extrapolate":
            # Linear extrapolation from interior
            if np.isinf(T_minus) and not np.isinf(T_plus):
                # Left boundary: extrapolate from right
                idx_plus2 = list(corner_idx)
                idx_plus2[dim] += 2
                if idx_plus2[dim] < shape[dim]:
                    T_plus2 = T[tuple(idx_plus2)]
                    T_minus = 2 * T_plus - T_plus2
            elif np.isinf(T_plus) and not np.isinf(T_minus):
                # Right boundary: extrapolate from left
                idx_minus2 = list(corner_idx)
                idx_minus2[dim] -= 2
                if idx_minus2[dim] >= 0:
                    T_minus2 = T[tuple(idx_minus2)]
                    T_plus = 2 * T_minus - T_minus2

        T_neighbors.append((T_minus, T_plus))

    return godunov_update_nd(T_neighbors, list(spacing), speed)


def identify_corner_points(shape: tuple[int, ...]) -> list[tuple[int, ...]]:
    """
    Identify all corner points of a grid.

    A corner point has at least one coordinate at 0 or N-1 in each dimension.

    Args:
        shape: Grid shape (Nx, Ny, ...).

    Returns:
        List of corner indices.
    """
    ndim = len(shape)
    corners = []

    # Generate all combinations of boundary coordinates
    from itertools import product

    for boundary_combo in product(*[[(0,), (shape[d] - 1,), range(1, shape[d] - 1)] for d in range(ndim)]):
        # A corner has boundary coordinates in ALL dimensions
        # Check if this is a true corner (boundary in all dims)
        is_corner = all(isinstance(boundary_combo[d], tuple) for d in range(ndim))
        if is_corner:
            corners.append(tuple(bc[0] if isinstance(bc, tuple) else bc for bc in boundary_combo))

    return corners


def identify_boundary_points(
    shape: tuple[int, ...],
    include_corners: bool = True,
) -> list[tuple[int, ...]]:
    """
    Identify all boundary points (points with at least one coordinate at 0 or N-1).

    Args:
        shape: Grid shape.
        include_corners: If False, exclude corner points.

    Returns:
        List of boundary point indices.
    """
    ndim = len(shape)
    boundary_points = []

    # Iterate through all points
    from itertools import product

    for idx in product(*[range(s) for s in shape]):
        # Check if any coordinate is at boundary
        is_boundary = any(idx[d] == 0 or idx[d] == shape[d] - 1 for d in range(ndim))

        if is_boundary:
            if include_corners:
                boundary_points.append(idx)
            else:
                # Exclude corners (boundary in ALL dimensions)
                is_corner = all(idx[d] == 0 or idx[d] == shape[d] - 1 for d in range(ndim))
                if not is_corner:
                    boundary_points.append(idx)

    return boundary_points

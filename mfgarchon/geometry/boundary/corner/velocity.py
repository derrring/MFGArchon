"""
Velocity-based boundary reflection (Specular Reflection) for particle methods.

This module provides velocity reflection at domain boundaries, essential for:
- Billiard dynamics (deterministic particle systems)
- Hard-sphere collisions
- Reflective boundary conditions with velocity state

The key formula is specular reflection:
    v_new = v - 2(v·n)n

where n is the outward normal at the boundary.

Corner Handling:
    At corners, the normal is ambiguous. Use corner_strategy parameter:
    - "average": Diagonal reflection (recommended)
    - "priority": Reflect off first face only
    - "mollify": Smooth transition (for SDF-like domains)

Anti-Zeno Safeguard:
    Optional damping prevents infinite corner bounces (Zeno trap):
    v_new = damping * (v - 2(v·n)n)

Reference:
    See Issue #521 for corner handling architecture.
    See docs/development/CORNER_HANDLING_IMPLEMENTATION_STATUS.md

Created: 2026-01-25 (Issue #521)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def reflect_velocity(
    position: NDArray[np.floating],
    velocity: NDArray[np.floating],
    bounds: list[tuple[float, float]] | NDArray[np.floating],
    corner_strategy: Literal["average", "priority", "mollify"] = "average",
    damping: float = 1.0,
    tolerance: float = 1e-10,
) -> NDArray[np.floating]:
    """
    Reflect velocity at boundary using specular reflection.

    Implements: v_new = damping * (v - 2(v·n)n)

    At corners (where multiple boundaries meet), the corner_strategy
    determines how the normal is computed.

    Args:
        position: Current position at or near boundary, shape (d,) or (N, d)
        velocity: Incoming velocity vector, shape (d,) or (N, d)
        bounds: Domain bounds as [(xmin, xmax), ...] or (d, 2) array
        corner_strategy: How to handle corners:
            - "average": Use averaged normal (diagonal reflection)
            - "priority": Use first face's normal (dimension 0 priority)
            - "mollify": Use mollified normal (radial from corner vertex)
        damping: Energy damping factor in [0, 1]. Default 1.0 (elastic).
            Use < 1.0 to prevent Zeno trap at corners.
        tolerance: Distance tolerance for boundary detection

    Returns:
        Reflected velocity, same shape as input

    Examples:
        >>> # 2D reflection at x_max boundary
        >>> position = np.array([1.0, 0.5])  # at x=1 boundary
        >>> velocity = np.array([1.0, 0.5])  # moving toward boundary
        >>> bounds = [(0, 1), (0, 1)]
        >>> v_new = reflect_velocity(position, velocity, bounds)
        >>> # v_new = [-1.0, 0.5] (x-component reversed)

        >>> # Corner reflection with damping
        >>> position = np.array([1.0, 1.0])  # at corner
        >>> velocity = np.array([1.0, 1.0])  # moving into corner
        >>> v_new = reflect_velocity(position, velocity, bounds, damping=0.9)

    Note:
        For position-only boundary handling (SDE particles), use
        reflect_positions() instead - it doesn't require velocity state.
    """
    position = np.atleast_1d(position)
    velocity = np.atleast_1d(velocity)
    bounds_arr = np.asarray(bounds)

    if bounds_arr.ndim == 1:
        bounds_arr = bounds_arr.reshape(1, 2)

    # Handle batch vs single
    single_input = position.ndim == 1
    if single_input:
        position = position.reshape(1, -1)
        velocity = velocity.reshape(1, -1)

    n_particles, _ndim = position.shape
    result = velocity.copy()

    domain_min = bounds_arr[:, 0]
    domain_max = bounds_arr[:, 1]

    for i in range(n_particles):
        # Get normal at this position
        normal = _get_boundary_normal(
            position[i],
            domain_min,
            domain_max,
            corner_strategy,
            tolerance,
        )

        if np.linalg.norm(normal) < 1e-12:
            # Not on boundary, no reflection
            continue

        # Specular reflection: v_new = v - 2(v·n)n
        v_dot_n = np.dot(velocity[i], normal)

        # Only reflect if moving into boundary (v·n > 0 for outward normal)
        if v_dot_n > 0:
            result[i] = damping * (velocity[i] - 2 * v_dot_n * normal)

    if single_input:
        return result[0]
    return result


def reflect_velocity_with_normal(
    velocity: NDArray[np.floating],
    normal: NDArray[np.floating],
    damping: float = 1.0,
) -> NDArray[np.floating]:
    """
    Reflect velocity given a pre-computed normal (low-level API).

    Use this when you already have the boundary normal from geometry queries.

    Args:
        velocity: Incoming velocity vector, shape (d,)
        normal: Unit outward normal at boundary, shape (d,)
        damping: Energy damping factor in [0, 1]

    Returns:
        Reflected velocity, shape (d,)

    Examples:
        >>> velocity = np.array([1.0, 1.0])
        >>> normal = np.array([1.0, 0.0])  # x_max boundary
        >>> v_new = reflect_velocity_with_normal(velocity, normal)
        >>> # v_new = [-1.0, 1.0]
    """
    v_dot_n = np.dot(velocity, normal)

    # Only reflect if moving into boundary
    if v_dot_n > 0:
        return damping * (velocity - 2 * v_dot_n * normal)
    return velocity.copy()


def _get_boundary_normal(
    point: NDArray[np.floating],
    domain_min: NDArray[np.floating],
    domain_max: NDArray[np.floating],
    corner_strategy: str,
    tolerance: float,
) -> NDArray[np.floating]:
    """
    Compute outward normal at boundary point.

    Internal function - for public API, use DomainGeometry.get_boundary_normal().
    """
    ndim = len(point)
    normal = np.zeros(ndim)

    # Detect which boundaries the point is on
    near_min = np.abs(point - domain_min) < tolerance
    near_max = np.abs(point - domain_max) < tolerance

    n_boundaries = np.sum(near_min | near_max)

    if n_boundaries == 0:
        return normal  # Not on boundary

    if n_boundaries == 1 or corner_strategy == "priority":
        # Single boundary or priority mode: first boundary found
        for dim in range(ndim):
            if near_min[dim]:
                normal[dim] = -1.0
                break
            elif near_max[dim]:
                normal[dim] = 1.0
                break
    elif corner_strategy == "average":
        # Corner: sum all face normals
        for dim in range(ndim):
            if near_min[dim]:
                normal[dim] -= 1.0
            if near_max[dim]:
                normal[dim] += 1.0
        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal /= norm
    elif corner_strategy == "mollify":
        # Treat corner as rounded: normal points from corner vertex
        corner_vertex = np.where(near_min, domain_min, domain_max)
        direction = point - corner_vertex
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            normal = direction / norm
        else:
            # At exact corner, fall back to average
            for dim in range(ndim):
                if near_min[dim]:
                    normal[dim] -= 1.0
                if near_max[dim]:
                    normal[dim] += 1.0
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                normal /= norm

    return normal


__all__ = [
    "reflect_velocity",
    "reflect_velocity_with_normal",
]

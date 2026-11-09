"""
Godunov upwind update formulas for the Eikonal equation.

The Eikonal equation |grad T| = 1/F is discretized using upwind finite differences.
At each grid point, we solve a local quadratic equation using neighbor values,
selecting the upwind direction (information flows from smaller to larger T).

Mathematical Background:
    For 1D: |dT/dx| = 1/F  =>  T = T_neighbor + dx/F

    For 2D: sqrt((dT/dx)^2 + (dT/dy)^2) = 1/F

    Using upwind differences (T_x = (T - T_xmin)/dx if T_xmin < T_xmax):
        sqrt(((T - a)/dx)^2 + ((T - b)/dy)^2) = 1/F

    where a = min(T_left, T_right), b = min(T_bottom, T_top).

    This yields a quadratic equation in T with solution via the quadratic formula.

Godunov Scheme:
    The Godunov flux selects the upwind direction automatically:
    - If both neighbors contribute: solve full quadratic
    - If one dimension dominates: reduce to 1D update
    - Always ensures causality (information flows from known to unknown)

References:
- Sethian (1996): Fast marching methods, SIAM Review
- Rouy & Tourin (1992): A viscosity solutions approach to shape-from-shading
- Zhao (2005): Fast sweeping method, Math. Comp.

Created: 2026-02-06 (Issue #664)
"""

from __future__ import annotations

import numpy as np

from mfgarchon.utils.mfg_logging import get_logger

logger = get_logger(__name__)


def godunov_update_1d(
    T_neighbors: tuple[float, float],
    dx: float,
    speed: float = 1.0,
) -> float:
    """
    Compute Godunov update for 1D Eikonal equation.

    Solves: |dT/dx| = 1/F

    Args:
        T_neighbors: (T_left, T_right) neighbor values. Use np.inf for boundaries.
        dx: Grid spacing.
        speed: Local speed F > 0. Solution satisfies |grad T| = 1/F.

    Returns:
        Updated value T satisfying the Eikonal equation.

    Example:
        >>> T_new = godunov_update_1d((0.1, 0.3), dx=0.01, speed=1.0)
        >>> # T_new = min(T_left, T_right) + dx/F = 0.1 + 0.01 = 0.11
    """
    # Select upwind neighbor (smaller T)
    T_min = min(T_neighbors)

    if np.isinf(T_min):
        # No valid neighbors
        return np.inf

    # 1D solution: T = T_min + dx/F
    return T_min + dx / speed


def godunov_update_2d(
    T_x_neighbors: tuple[float, float],
    T_y_neighbors: tuple[float, float],
    dx: float,
    dy: float,
    speed: float = 1.0,
) -> float:
    """
    Compute Godunov update for 2D Eikonal equation.

    Solves: sqrt((dT/dx)^2 + (dT/dy)^2) = 1/F

    The upwind scheme selects the minimum neighbor in each direction:
        a = min(T_left, T_right)
        b = min(T_bottom, T_top)

    Then solves:
        sqrt(((T - a)/dx)^2 + ((T - b)/dy)^2) = 1/F

    Args:
        T_x_neighbors: (T_left, T_right) neighbors in x-direction.
        T_y_neighbors: (T_bottom, T_top) neighbors in y-direction.
        dx: Grid spacing in x.
        dy: Grid spacing in y.
        speed: Local speed F > 0.

    Returns:
        Updated value T satisfying the 2D Eikonal equation.

    Note:
        If the 2D solution is invalid (discriminant < 0), falls back to
        1D updates, which is correct when one direction dominates.
    """
    # Upwind selection: take minimum in each direction
    a = min(T_x_neighbors)  # x-direction upwind
    b = min(T_y_neighbors)  # y-direction upwind

    # Handle boundary cases (no valid neighbor)
    a_valid = not np.isinf(a)
    b_valid = not np.isinf(b)

    if not a_valid and not b_valid:
        return np.inf

    if not a_valid:
        # Only y-direction contributes
        return b + dy / speed

    if not b_valid:
        # Only x-direction contributes
        return a + dx / speed

    # Both directions valid: solve full 2D quadratic
    # |grad T|^2 = 1/F^2
    # ((T-a)/dx)^2 + ((T-b)/dy)^2 = 1/F^2
    #
    # Let: alpha = 1/dx^2, beta = 1/dy^2, rhs = 1/F^2
    # alpha*(T-a)^2 + beta*(T-b)^2 = rhs
    # (alpha + beta)*T^2 - 2*(alpha*a + beta*b)*T + (alpha*a^2 + beta*b^2 - rhs) = 0

    alpha = 1.0 / (dx * dx)
    beta = 1.0 / (dy * dy)
    rhs = 1.0 / (speed * speed)

    # Quadratic coefficients: A*T^2 + B*T + C = 0
    A = alpha + beta
    B = -2.0 * (alpha * a + beta * b)
    C = alpha * a * a + beta * b * b - rhs

    discriminant = B * B - 4.0 * A * C

    if discriminant < 0:
        # 2D update not valid: one direction dominates
        # Fall back to 1D updates and take minimum
        T_x = a + dx / speed
        T_y = b + dy / speed
        return min(T_x, T_y)

    # Take larger root (we want T > max(a, b))
    T_2d = (-B + np.sqrt(discriminant)) / (2.0 * A)

    # Verify causality: T must be >= both upwind values
    # If not, fall back to 1D
    if T_2d < a or T_2d < b:
        T_x = a + dx / speed
        T_y = b + dy / speed
        return min(T_x, T_y)

    return T_2d


def godunov_update_nd(
    T_neighbors: list[tuple[float, float]],
    spacing: list[float] | tuple[float, ...],
    speed: float = 1.0,
) -> float:
    """
    Compute Godunov update for n-dimensional Eikonal equation.

    Solves: |grad T| = 1/F in arbitrary dimensions.

    The algorithm:
    1. Select upwind neighbor in each dimension (minimum T)
    2. Sort dimensions by upwind value (smallest first)
    3. Progressively add dimensions until solution is valid

    This ensures causality: information flows from known (small T) to unknown.

    Args:
        T_neighbors: List of (T_minus, T_plus) neighbor pairs for each dimension.
                    Use np.inf for invalid/boundary neighbors.
        spacing: Grid spacing in each dimension [dx, dy, dz, ...].
        speed: Local speed F > 0.

    Returns:
        Updated value T satisfying the nD Eikonal equation.

    Example:
        >>> # 3D case
        >>> T_neighbors = [(0.1, 0.3), (0.2, 0.4), (0.15, 0.35)]
        >>> spacing = [0.01, 0.01, 0.01]
        >>> T_new = godunov_update_nd(T_neighbors, spacing, speed=1.0)
    """
    ndim = len(T_neighbors)

    if ndim == 1:
        return godunov_update_1d(T_neighbors[0], spacing[0], speed)

    if ndim == 2:
        return godunov_update_2d(T_neighbors[0], T_neighbors[1], spacing[0], spacing[1], speed)

    # General n-dimensional case
    # Select upwind values and sort by magnitude
    upwind_data = []  # List of (T_upwind, 1/dx^2) tuples
    for dim in range(ndim):
        T_upwind = min(T_neighbors[dim])
        if not np.isinf(T_upwind):
            h = spacing[dim]
            upwind_data.append((T_upwind, 1.0 / (h * h), h))

    if not upwind_data:
        return np.inf

    # Sort by upwind value (smallest first for causality)
    upwind_data.sort(key=lambda x: x[0])

    # Progressively add dimensions
    rhs = 1.0 / (speed * speed)

    for k in range(1, len(upwind_data) + 1):
        # Try k-dimensional update with smallest k upwind values
        T_solution = _solve_nd_quadratic(upwind_data[:k], rhs)

        if T_solution is not None:
            # Verify causality: T must be >= all included upwind values
            T_max_upwind = upwind_data[k - 1][0]
            if T_solution >= T_max_upwind:
                # Check if adding more dimensions would give smaller T
                # (continue to potentially find better solution)
                if k < len(upwind_data):
                    # Check if next dimension could contribute
                    T_next = upwind_data[k][0]
                    if T_solution > T_next:
                        continue  # Try including more dimensions
                return T_solution

    # Fallback: 1D update with smallest upwind value
    T_min, _, h_min = upwind_data[0]
    return T_min + h_min / speed


def _solve_nd_quadratic(
    upwind_data: list[tuple[float, float, float]],
    rhs: float,
) -> float | None:
    """
    Solve the n-dimensional quadratic for Godunov update.

    Solves: sum_i ((T - a_i) / h_i)^2 = rhs
           sum_i alpha_i * (T - a_i)^2 = rhs  where alpha_i = 1/h_i^2

    Expanding:
        (sum alpha_i) * T^2 - 2*(sum alpha_i*a_i)*T + (sum alpha_i*a_i^2) = rhs

    Args:
        upwind_data: List of (T_upwind, 1/h^2, h) tuples for active dimensions.
        rhs: Right-hand side = 1/F^2.

    Returns:
        Solution T, or None if no valid solution (discriminant < 0).
    """
    # Compute quadratic coefficients
    A = sum(alpha for _, alpha, _ in upwind_data)
    B = -2.0 * sum(alpha * a for a, alpha, _ in upwind_data)
    C = sum(alpha * a * a for a, alpha, _ in upwind_data) - rhs

    discriminant = B * B - 4.0 * A * C

    if discriminant < 0:
        return None

    # Take larger root (T > all upwind values)
    return (-B + np.sqrt(discriminant)) / (2.0 * A)

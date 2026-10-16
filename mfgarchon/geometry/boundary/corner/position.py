"""
Position-based boundary reflection (Issue #521).

This module provides position transformations for NON-PERIODIC boundaries:
- reflect_positions: Fold reflection (NO_FLUX/REFLECTING BC)
- absorb_positions: Clamp to domain (DIRICHLET BC for particles)

For PERIODIC boundaries, see: geometry/boundary/periodic.py

Corner Handling:
    At corners (codim >= 2), all dimensions are processed simultaneously,
    producing diagonal reflection. This is equivalent to 'average' corner
    strategy for position-based reflection.

    Example: A particle at (-0.1, -0.1) in domain [0,1]x[0,1] is reflected
    to (0.1, 0.1) - diagonal reflection at the corner.

Reference:
    See Issue #521 for corner handling architecture and design decisions.

Created: 2025-01-25 (Issue #521)
Refactored: 2026-01-29 (Issue #711 - moved periodic to boundary/periodic.py)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.utils.deprecation import deprecated

if TYPE_CHECKING:
    from numpy.typing import NDArray


def reflect_positions(
    positions: NDArray[np.floating],
    bounds: list[tuple[float, float]] | NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Reflect positions back into domain using fold reflection (n-D).

    At corners, all dimensions are processed simultaneously, producing
    diagonal reflection. This is equivalent to 'average' corner strategy
    for position-based reflection (Issue #521).

    The fold reflection algorithm handles particles that travel multiple
    domain widths in a single step, correctly "folding" them back into
    the domain like light reflecting between mirrors.

    Args:
        positions: Particle positions, shape (N, d) or (d,) for single point
        bounds: Domain bounds as [(xmin, xmax), (ymin, ymax), ...] or (d, 2) array

    Returns:
        Reflected positions, same shape as input

    Examples:
        >>> # 2D corner reflection
        >>> positions = np.array([[-0.1, -0.1], [0.5, 0.5], [1.2, 0.8]])
        >>> bounds = [(0, 1), (0, 1)]
        >>> reflected = reflect_positions(positions, bounds)
        >>> # (-0.1, -0.1) -> (0.1, 0.1) diagonal reflection at corner
        >>> # (1.2, 0.8) -> (0.8, 0.8) reflection at x boundary

    Note:
        This function is the canonical implementation for position-based
        boundary reflection. All particle BC handlers should use this.
        See Issue #521 for corner handling architecture.
    """
    # A (d,) point comes back as (d,), and a (1, d) array as (1, d): the shape the input had, not the row count (#2691).
    was_1d = np.ndim(positions) == 1
    positions = np.atleast_2d(positions)
    result = positions.copy()

    bounds = np.asarray(bounds)
    if bounds.ndim == 1:
        # Single dimension: bounds = [xmin, xmax]
        bounds = bounds.reshape(1, 2)

    ndim = result.shape[1]
    if bounds.shape[0] != ndim:
        raise ValueError(f"Bounds dimension {bounds.shape[0]} != positions dimension {ndim}")

    # Apply fold reflection per dimension (simultaneous, not sequential)
    for d in range(ndim):
        xmin, xmax = bounds[d, 0], bounds[d, 1]
        Lx = xmax - xmin

        if Lx > 1e-14:
            shifted = result[:, d] - xmin
            period = 2 * Lx
            pos_in_period = shifted % period
            in_second_half = pos_in_period > Lx
            pos_in_period[in_second_half] = period - pos_in_period[in_second_half]
            result[:, d] = xmin + pos_in_period

    if was_1d:
        return result[0]
    return result


def absorb_positions(
    positions: NDArray[np.floating],
    bounds: list[tuple[float, float]] | NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Clamp positions to domain boundaries (absorbing/Dirichlet BC, n-D).

    Args:
        positions: Particle positions, shape (N, d) or (d,) for single point
        bounds: Domain bounds as [(xmin, xmax), (ymin, ymax), ...] or (d, 2) array

    Returns:
        Clamped positions, same shape as input
    """
    # A (d,) point comes back as (d,), and a (1, d) array as (1, d): the shape the input had, not the row count (#2691).
    was_1d = np.ndim(positions) == 1
    positions = np.atleast_2d(positions)
    result = positions.copy()

    bounds = np.asarray(bounds)
    if bounds.ndim == 1:
        bounds = bounds.reshape(1, 2)

    ndim = result.shape[1]
    for d in range(ndim):
        result[:, d] = np.clip(result[:, d], bounds[d, 0], bounds[d, 1])

    if was_1d:
        return result[0]
    return result


# Backward compatibility: import from new location
# TODO: Remove in v1.0.0 after deprecation period
@deprecated(
    since="v0.17.0",
    replacement="Use mfgarchon.geometry.boundary.periodic.wrap_positions instead.",
)
def wrap_positions(
    positions: NDArray[np.floating],
    bounds: list[tuple[float, float]] | NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    DEPRECATED: Use mfgarchon.geometry.boundary.periodic.wrap_positions instead.

    This function is kept for backward compatibility and will be removed in v1.0.0.
    """
    from mfgarchon.geometry.boundary.periodic import wrap_positions as _wrap

    return _wrap(positions, bounds)


__all__ = [
    "reflect_positions",
    "absorb_positions",
    "wrap_positions",  # Deprecated, for backward compat
]


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    """Quick smoke test for boundary reflection utilities."""
    print("Testing boundary reflection utilities (corner/position.py)...")

    bounds_2d = [(0, 1), (0, 1)]

    # Test 2D reflect at corner
    print("\n1. reflect_positions:")
    positions_2d = np.array([[-0.1, -0.1], [0.5, 0.5], [1.2, 0.8]])
    reflected_2d = reflect_positions(positions_2d, bounds_2d)
    expected_2d = np.array([[0.1, 0.1], [0.5, 0.5], [0.8, 0.8]])
    assert np.allclose(reflected_2d, expected_2d), f"2D reflect failed: {reflected_2d}"
    print("   2D reflect (corner): passed")

    # Test 3D corner reflection
    positions_3d = np.array([[-0.2, -0.2, -0.2]])
    bounds_3d = [(0, 1), (0, 1), (0, 1)]
    reflected_3d = reflect_positions(positions_3d, bounds_3d)
    expected_3d = np.array([[0.2, 0.2, 0.2]])  # Diagonal reflection
    assert np.allclose(reflected_3d, expected_3d), f"3D corner reflect failed: {reflected_3d}"
    print("   3D corner reflect: passed")

    # Test single point input
    single_point = np.array([1.5, 0.5])
    single_reflected = reflect_positions(single_point, bounds_2d)
    assert single_reflected.shape == (2,), f"Single point shape wrong: {single_reflected.shape}"
    assert np.allclose(single_reflected, [0.5, 0.5]), f"Single point value wrong: {single_reflected}"
    print("   Single point handling: passed")

    # Test large displacement (multiple domain widths)
    large_disp = np.array([[-5.0, 7.0]])  # Way outside [0,2]x[0,2]
    bounds_large = [(0, 2), (0, 2)]
    large_result = reflect_positions(large_disp, bounds_large)
    assert np.allclose(large_result, [[1.0, 1.0]]), f"Large displacement failed: {large_result}"
    print("   Large displacement: passed")

    # Test 2D absorb
    print("\n2. absorb_positions:")
    positions_absorb = np.array([[-0.5, 1.5], [0.5, 0.5]])
    absorbed = absorb_positions(positions_absorb, bounds_2d)
    expected_absorb = np.array([[0.0, 1.0], [0.5, 0.5]])
    assert np.allclose(absorbed, expected_absorb), f"2D absorb failed: {absorbed}"
    print("   2D absorb: passed")

    print("\nAll corner/position.py tests passed!")

"""
Region predicate factories for geometry marking.

Provides convenience functions that return predicates compatible with
``geometry.mark_region(name, predicate=...)``. Each factory returns a callable
``(NDArray) -> NDArray[np.bool_]`` where input has shape ``(N, d)``.

Usage:
    from mfgarchon.geometry.predicates import box_region, sphere_region

    grid.mark_region("inlet", predicate=box_region([0, 0], [0.1, 1]))
    grid.mark_region("obstacle", predicate=sphere_region([0.5, 0.5], 0.1))

Created: 2026-03-28 (BC Roadmap Phase 1.3 - Region Registry)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.types.callable_protocols import read_region

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def box_region(
    lower: NDArray | list[float],
    upper: NDArray | list[float],
) -> Callable[[NDArray], NDArray[np.bool_]]:
    """
    Create predicate for an axis-aligned box region.

    Selects points x where lower_i <= x_i <= upper_i for all dimensions i.

    Args:
        lower: Lower corner coordinates, shape (d,)
        upper: Upper corner coordinates, shape (d,)

    Returns:
        Predicate: (N, d) -> (N,) boolean mask

    Example:
        >>> pred = box_region([0, 0], [0.1, 1.0])
        >>> grid.mark_region("inlet", predicate=pred)
    """
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)

    def predicate(points: NDArray) -> NDArray[np.bool_]:
        return np.all((points >= lo) & (points <= hi), axis=-1)

    return predicate


def sphere_region(
    center: NDArray | list[float],
    radius: float,
) -> Callable[[NDArray], NDArray[np.bool_]]:
    """
    Create predicate for a spherical (ball) region.

    Selects points x where ||x - center|| <= radius.

    Args:
        center: Center coordinates, shape (d,)
        radius: Radius of the sphere

    Returns:
        Predicate: (N, d) -> (N,) boolean mask

    Example:
        >>> pred = sphere_region([0.5, 0.5], 0.1)
        >>> grid.mark_region("obstacle", predicate=pred)
    """
    c = np.asarray(center, dtype=np.float64)
    r = float(radius)

    def predicate(points: NDArray) -> NDArray[np.bool_]:
        return np.linalg.norm(points - c, axis=-1) <= r

    return predicate


def sdf_region(
    sdf: Callable[..., NDArray[np.floating]],
) -> Callable[[NDArray], NDArray[np.bool_]]:
    """
    Create predicate from a signed distance function.

    Selects points x where sdf(x) <= 0 (interior + boundary of SDF domain).

    Args:
        sdf: Signed distance function reading ``x``, a coordinate tuple.
             Negative = interior, zero = boundary, positive = exterior.

    Returns:
        Predicate: (N, d) -> (N,) boolean mask

    Example:
        >>> # Ellipse SDF
        >>> sdf_fn = lambda x: (x[0]/2)**2 + (x[1]/1)**2 - 1
        >>> grid.mark_region("ellipse", predicate=sdf_region(sdf_fn))
    """

    def predicate(points: NDArray) -> NDArray[np.bool_]:
        return read_region(sdf, points, role="sdf") <= 0

    return predicate


def halfspace_region(
    normal: NDArray | list[float],
    offset: float = 0.0,
) -> Callable[[NDArray], NDArray[np.bool_]]:
    """
    Create predicate for a half-space region.

    Selects points x where n . x <= offset (i.e., on the "negative" side
    of the hyperplane n . x = offset).

    Args:
        normal: Outward normal vector of the hyperplane, shape (d,)
        offset: Signed offset from origin (default 0)

    Returns:
        Predicate: (N, d) -> (N,) boolean mask

    Example:
        >>> # Region where x + y <= 1 (below the diagonal)
        >>> pred = halfspace_region([1, 1], offset=1.0)
        >>> grid.mark_region("lower_triangle", predicate=pred)
    """
    n = np.asarray(normal, dtype=np.float64)
    d = float(offset)

    def predicate(points: NDArray) -> NDArray[np.bool_]:
        return points @ n <= d

    return predicate

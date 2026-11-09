"""
Visibility queries for meshfree methods in domains with obstacles.

Provides line-of-sight checking between points using signed distance functions (SDF).
Primary use case: filtering GFDM stencil neighbors to prevent "cross-wall" stencils
in narrow channels between obstacles.

The core idea is simple: sample the SDF along the line segment between two points.
If any sample falls inside an obstacle (SDF < 0), the line of sight is blocked.

SDF Convention (consistent with mfgarchon.geometry.implicit):
    phi(x) < 0  =>  x inside obstacle (blocked)
    phi(x) = 0  =>  x on obstacle boundary
    phi(x) > 0  =>  x outside obstacle (clear)

Note: The obstacle_sdf here should be the SDF of the *obstacle region* (e.g., a
UnionDomain of pillars), NOT the full computational domain. For a DifferenceDomain
(rectangle minus pillars), pass the SDF of the pillars union.

Example:
    >>> from mfgarchon.geometry.implicit import Hypersphere, UnionDomain
    >>> pillars = UnionDomain([
    ...     Hypersphere(center=[6.0, 3.5], radius=1.3),
    ...     Hypersphere(center=[6.0, 6.5], radius=1.3),
    ... ])
    >>> from mfgarchon.geometry.visibility import filter_visible_neighbors
    >>> mask = filter_visible_neighbors(
    ...     center=np.array([5.0, 5.0]),
    ...     candidates=candidate_points,
    ...     obstacle_sdf=pillars.signed_distance,
    ... )
    >>> visible_candidates = candidate_points[mask]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.types.callable_protocols import read_region

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def segments_clear(
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    obstacle_sdf: Callable[..., NDArray[np.float64]],
    n_samples: int = 10,
    margin: float = 0.0,
) -> NDArray[np.bool_]:
    """Check whether line segments between point pairs avoid obstacles.

    For each pair (p1[i], p2[i]), samples n_samples points along the segment
    and checks that the obstacle SDF is > margin at all samples.

    Args:
        p1: Start points, shape (N, d) or (d,) for a single point.
        p2: End points, shape (N, d) or (M, d). If p1 is (d,) and p2 is (M, d),
            broadcasts p1 to all segments (1-to-many check).
        obstacle_sdf: Callable reading ``x``, a coordinate tuple, to SDF values.
            Negative values = inside obstacle.
        n_samples: Number of interior samples along each segment (excluding
            endpoints, which are assumed to be valid collocation points).
            More samples = fewer false positives but slower. Default 10 is
            sufficient for convex obstacles (spheres, rectangles).
        margin: Safety margin. A segment is blocked if any sample has
            SDF < margin. Use margin > 0 for conservative filtering
            (keep stencils away from obstacle surfaces).

    Returns:
        Boolean array of shape (N,) or (M,). True = segment is clear (no
        obstacle intersection), False = blocked.

    Complexity:
        O(N * n_samples * d) for SDF evaluation. Fully vectorized.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)

    # Handle broadcasting: single p1 with multiple p2
    if p1.ndim == 1:
        p1 = p1[np.newaxis, :]  # (1, d)
    if p2.ndim == 1:
        p2 = p2[np.newaxis, :]

    n_segments = max(p1.shape[0], p2.shape[0])

    # Sample interior points along each segment: t in (0, 1) exclusive
    # Excluding endpoints avoids false positives from points ON obstacle boundary
    t = np.linspace(0.0, 1.0, n_samples + 2)[1:-1]  # (n_samples,)

    # Build sample points: shape (n_segments, n_samples, d)
    # p1: (N, 1, d), p2: (M, 1, d), t: (1, n_samples, 1)
    t_col = t[np.newaxis, :, np.newaxis]  # (1, n_samples, 1)
    samples = p1[:, np.newaxis, :] * (1.0 - t_col) + p2[:, np.newaxis, :] * t_col

    # Flatten for batch SDF evaluation: (n_segments * n_samples, d)
    d = samples.shape[-1]
    flat_samples = samples.reshape(-1, d)

    # Evaluate SDF at all sample points
    sdf_values = read_region(obstacle_sdf, flat_samples, role="obstacle_sdf")  # (n_segments * n_samples,)
    sdf_values = np.asarray(sdf_values).reshape(n_segments, n_samples)

    # Segment is clear if ALL samples are outside obstacle (SDF > margin)
    return np.all(sdf_values > margin, axis=1)


def filter_visible_neighbors(
    center: NDArray[np.float64],
    candidates: NDArray[np.float64],
    obstacle_sdf: Callable[..., NDArray[np.float64]],
    n_samples: int = 10,
    margin: float = 0.0,
) -> NDArray[np.bool_]:
    """Filter candidate neighbors by line-of-sight visibility from a center point.

    Returns a boolean mask indicating which candidates are visible (not occluded
    by obstacles) from the center point.

    This is the primary API for GFDM stencil filtering. Typical usage:

        neighbors = tree.query_ball_point(center, delta)
        mask = filter_visible_neighbors(center, points[neighbors], obstacle_sdf)
        visible_neighbors = neighbors[mask]

    Args:
        center: Reference point, shape (d,).
        candidates: Candidate neighbor points, shape (M, d).
        obstacle_sdf: SDF of the obstacle region. See module docstring for convention.
        n_samples: Interior samples per segment. Default 10.
        margin: Safety margin for obstacle proximity. Default 0.0.

    Returns:
        Boolean mask of shape (M,). True = visible (line of sight clear).
    """
    if candidates.ndim == 1:
        candidates = candidates[np.newaxis, :]

    if len(candidates) == 0:
        return np.array([], dtype=bool)

    return segments_clear(center, candidates, obstacle_sdf, n_samples, margin)

"""
Signed Distance Function (SDF) Utilities for MFG Computations.

This module provides convenient utility functions for computing signed distance
functions (SDFs) to common geometric primitives. These are commonly needed in:
- Obstacle avoidance problems
- Constrained MFG domains
- Boundary condition specification
- Visualization and level set methods

The utilities wrap the full `mfgarchon.geometry.implicit` infrastructure with
simpler function-based APIs for quick prototyping.

Key Functions:
- sdf_sphere: Distance to sphere/ball
- sdf_box: Distance to axis-aligned box
- sdf_union: Minimum of multiple SDFs (union)
- sdf_intersection: Maximum of multiple SDFs (intersection)
- sdf_complement: Negation of SDF (complement)
- sdf_difference: Difference of two SDFs

Use Cases:
- Quick obstacle specification in research code
- Initialization of level set methods
- Boundary proximity calculations
- Visualization of implicit boundaries

Examples:
    >>> import numpy as np
    >>> from mfgarchon.utils.numerical import sdf_sphere, sdf_box
    >>>
    >>> # Distance to unit sphere centered at origin
    >>> points = np.array([[0, 0], [1, 0], [2, 0]])
    >>> dist = sdf_sphere(points, center=[0, 0], radius=1.0)
    >>> # [-1.0, 0.0, 1.0] (inside, on boundary, outside)
    >>>
    >>> # Distance to unit box
    >>> dist_box = sdf_box(points, bounds=[[-1, 1], [-1, 1]])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.types.callable_protocols import read_region

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Import existing SDF infrastructure
from mfgarchon.geometry.implicit import (
    Hyperrectangle,
    Hypersphere,
)


def sdf_sphere(
    points: NDArray[np.floating],
    center: NDArray[np.floating] | list[float] | tuple[float, ...],
    radius: float,
) -> NDArray[np.floating]:
    """
    Compute signed distance to a sphere/ball.

    Convention: Negative inside, zero on boundary, positive outside.

    Parameters
    ----------
    points : ndarray
        Points to evaluate, shape (d,) for single point or (N, d) for N points
    center : array_like
        Center of sphere, length d
    radius : float
        Radius of sphere

    Returns
    -------
    ndarray
        Signed distances, shape () for single point or (N,) for multiple points
        - dist < 0: Inside sphere
        - dist = 0: On sphere boundary
        - dist > 0: Outside sphere

    Examples
    --------
    1D (interval):
    >>> points = np.array([0.0, 0.5, 1.0, 1.5])
    >>> dist = sdf_sphere(points, center=[0.5], radius=0.5)
    >>> # [-0.5, 0.0, 0.0, 0.5]

    2D (circle):
    >>> points = np.array([[0, 0], [1, 0], [2, 0]])
    >>> dist = sdf_sphere(points, center=[0, 0], radius=1.0)
    >>> # [-1.0, 0.0, 1.0]

    3D (ball):
    >>> points = np.random.uniform(-2, 2, (100, 3))
    >>> dist = sdf_sphere(points, center=[0, 0, 0], radius=1.0)
    """
    center = np.asarray(center, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)

    # Handle 1D arrays: reshape (N,) -> (N, 1)
    if points.ndim == 1:
        if len(center) == 1:
            # 1D case: both should be (N, 1)
            points = points.reshape(-1, 1)
        # else: single point in d-D, keep as (d,) for Hypersphere

    # Use existing Hypersphere infrastructure
    sphere = Hypersphere(center=center, radius=radius)
    return sphere.signed_distance(points)


def sdf_box(
    points: NDArray[np.floating],
    bounds: NDArray[np.floating] | list[tuple[float, float]],
) -> NDArray[np.floating]:
    """
    Compute signed distance to an axis-aligned box/hyperrectangle.

    Convention: Negative inside, zero on boundary, positive outside.

    Parameters
    ----------
    points : ndarray
        Points to evaluate, shape (d,) or (N, d)
    bounds : array_like
        Box bounds, shape (d, 2) where bounds[i] = [min_i, max_i]
        Or list of tuples: [(xmin, xmax), (ymin, ymax), ...]

    Returns
    -------
    ndarray
        Signed distances, shape () or (N,)

    Examples
    --------
    1D interval:
    >>> points = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    >>> dist = sdf_box(points, bounds=[[0, 1]])
    >>> # [1.0, 0.0, -0.5, 0.0, 1.0]

    2D rectangle:
    >>> points = np.array([[0.5, 0.5], [0, 0], [2, 2]])
    >>> dist = sdf_box(points, bounds=[[0, 1], [0, 1]])
    >>> # [-0.5, 0.0, ~1.4]

    3D box:
    >>> points = np.array([[0.5, 0.5, 0.5], [-1, 0, 0]])
    >>> dist = sdf_box(points, bounds=[[0, 1], [0, 1], [0, 1]])
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)

    # Handle 1D arrays: reshape (N,) -> (N, 1)
    if points.ndim == 1 and len(bounds) == 1:
        points = points.reshape(-1, 1)

    # Use existing Hyperrectangle infrastructure
    box = Hyperrectangle(bounds=bounds)
    return box.signed_distance(points)


def sdf_union(
    *sdfs: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Compute union of multiple SDFs (minimum operation).

    The union of domains corresponds to taking the minimum of their SDFs.

    Parameters
    ----------
    *sdfs : ndarray
        Multiple SDF arrays to combine, all same shape

    Returns
    -------
    ndarray
        Union SDF = min(sdf1, sdf2, ...), same shape as inputs

    Examples
    --------
    Union of two circles:
    >>> points = np.linspace(-2, 2, 100).reshape(-1, 1)
    >>> sdf1 = sdf_sphere(points, center=[-0.5], radius=0.5)
    >>> sdf2 = sdf_sphere(points, center=[0.5], radius=0.5)
    >>> union = sdf_union(sdf1, sdf2)  # Capsule-like shape

    Union of box and sphere (rounded box):
    >>> points = np.random.uniform(-2, 2, (1000, 2))
    >>> box_dist = sdf_box(points, bounds=[[-1, 1], [-1, 1]])
    >>> sphere_dist = sdf_sphere(points, center=[0, 0], radius=1.5)
    >>> union = sdf_union(box_dist, sphere_dist)
    """
    if len(sdfs) == 0:
        raise ValueError("At least one SDF required")
    if len(sdfs) == 1:
        return sdfs[0]

    # Union = minimum (least restrictive)
    return np.minimum.reduce(sdfs)


def sdf_intersection(
    *sdfs: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Compute intersection of multiple SDFs (maximum operation).

    The intersection of domains corresponds to taking the maximum of their SDFs.

    Parameters
    ----------
    *sdfs : ndarray
        Multiple SDF arrays to combine, all same shape

    Returns
    -------
    ndarray
        Intersection SDF = max(sdf1, sdf2, ...), same shape as inputs

    Examples
    --------
    Intersection of two spheres:
    >>> points = np.linspace(-2, 2, 100).reshape(-1, 1)
    >>> sdf1 = sdf_sphere(points, center=[-0.3], radius=0.7)
    >>> sdf2 = sdf_sphere(points, center=[0.3], radius=0.7)
    >>> intersection = sdf_intersection(sdf1, sdf2)  # Lens shape

    Box with circular constraint:
    >>> points = np.random.uniform(-2, 2, (1000, 2))
    >>> box_dist = sdf_box(points, bounds=[[-1, 1], [-1, 1]])
    >>> sphere_dist = sdf_sphere(points, center=[0, 0], radius=0.8)
    >>> intersection = sdf_intersection(box_dist, sphere_dist)
    """
    if len(sdfs) == 0:
        raise ValueError("At least one SDF required")
    if len(sdfs) == 1:
        return sdfs[0]

    # Intersection = maximum (most restrictive)
    return np.maximum.reduce(sdfs)


def sdf_complement(
    sdf: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Compute complement of an SDF (negation).

    The complement reverses inside/outside.

    Parameters
    ----------
    sdf : ndarray
        SDF to complement

    Returns
    -------
    ndarray
        Complement SDF = -sdf, same shape as input

    Examples
    --------
    Exterior of sphere:
    >>> points = np.array([[0, 0], [1, 0], [2, 0]])
    >>> sphere_dist = sdf_sphere(points, center=[0, 0], radius=1.0)
    >>> # [-1.0, 0.0, 1.0]
    >>> exterior = sdf_complement(sphere_dist)
    >>> # [1.0, 0.0, -1.0] (now exterior is "inside")
    """
    return -sdf


def sdf_difference(
    sdf_a: NDArray[np.floating],
    sdf_b: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Compute difference of two SDFs (A \\ B).

    Represents domain A with domain B removed (A minus B).

    Parameters
    ----------
    sdf_a : ndarray
        SDF of domain A
    sdf_b : ndarray
        SDF of domain B (to subtract from A)

    Returns
    -------
    ndarray
        Difference SDF = max(sdf_a, -sdf_b), same shape as inputs

    Examples
    --------
    Box with circular hole:
    >>> points = np.random.uniform(-2, 2, (1000, 2))
    >>> box = sdf_box(points, bounds=[[-1, 1], [-1, 1]])
    >>> hole = sdf_sphere(points, center=[0, 0], radius=0.5)
    >>> domain = sdf_difference(box, hole)
    >>> # Points inside box but outside hole have negative SDF

    Annulus (ring):
    >>> outer = sdf_sphere(points, center=[0, 0], radius=1.0)
    >>> inner = sdf_sphere(points, center=[0, 0], radius=0.5)
    >>> ring = sdf_difference(outer, inner)
    """
    # A \ B = A ∩ complement(B) = max(sdf_a, -sdf_b)
    return sdf_intersection(sdf_a, sdf_complement(sdf_b))


def sdf_smooth_union(
    sdf_a: NDArray[np.floating],
    sdf_b: NDArray[np.floating],
    smoothing: float = 0.1,
) -> NDArray[np.floating]:
    """
    Compute smooth union of two SDFs.

    Creates a smooth blend between two shapes instead of a sharp seam.
    Useful for visualization and gradient-based methods.

    Parameters
    ----------
    sdf_a, sdf_b : ndarray
        SDFs to blend
    smoothing : float
        Smoothing radius (k parameter). Larger = smoother blend.
        Common range: 0.01 to 0.5

    Returns
    -------
    ndarray
        Smooth union SDF, same shape as inputs

    Notes
    -----
    Uses polynomial smooth minimum (Quilez, 2008):
    https://iquilezles.org/articles/smin/

    Examples
    --------
    Smooth blend of two circles:
    >>> points = np.linspace(-2, 2, 100).reshape(-1, 1)
    >>> sdf1 = sdf_sphere(points, center=[-0.5], radius=0.5)
    >>> sdf2 = sdf_sphere(points, center=[0.5], radius=0.5)
    >>> smooth = sdf_smooth_union(sdf1, sdf2, smoothing=0.2)
    """
    h = np.clip(0.5 + 0.5 * (sdf_b - sdf_a) / smoothing, 0.0, 1.0)
    return sdf_b * h + sdf_a * (1.0 - h) - smoothing * h * (1.0 - h)


def sdf_smooth_intersection(
    sdf_a: NDArray[np.floating],
    sdf_b: NDArray[np.floating],
    smoothing: float = 0.1,
) -> NDArray[np.floating]:
    """
    Compute smooth intersection of two SDFs.

    Creates a smooth blend at the intersection instead of a sharp corner.

    Parameters
    ----------
    sdf_a, sdf_b : ndarray
        SDFs to blend
    smoothing : float
        Smoothing radius. Larger = smoother blend.

    Returns
    -------
    ndarray
        Smooth intersection SDF, same shape as inputs

    Examples
    --------
    Smooth intersection of two spheres:
    >>> points = np.linspace(-2, 2, 100).reshape(-1, 1)
    >>> sdf1 = sdf_sphere(points, center=[-0.3], radius=0.7)
    >>> sdf2 = sdf_sphere(points, center=[0.3], radius=0.7)
    >>> smooth = sdf_smooth_intersection(sdf1, sdf2, smoothing=0.1)
    """
    h = np.clip(0.5 - 0.5 * (sdf_b - sdf_a) / smoothing, 0.0, 1.0)
    return sdf_b * h + sdf_a * (1.0 - h) + smoothing * h * (1.0 - h)


def sdf_gradient(
    points: NDArray[np.floating],
    sdf_func: callable,
    epsilon: float = 1e-5,
) -> NDArray[np.floating]:
    """
    Compute gradient of SDF using finite differences.

    The gradient ∇φ points in the direction of steepest increase and
    has magnitude |∇φ| = 1 for exact SDFs.

    Parameters
    ----------
    points : ndarray
        Points to evaluate gradient, shape (d,) or (N, d)
    sdf_func : callable
        SDF function reading ``x``, a coordinate tuple
    epsilon : float
        Finite difference step size

    Returns
    -------
    ndarray
        SDF gradients, shape (d,) or (N, d)

    Examples
    --------
    Gradient of sphere SDF (points outward from center):
    >>> points = np.array([[0.5, 0], [0, 0.5]])
    >>> def sdf(x):
    ...     return sdf_sphere(np.stack(x, axis=-1), center=[0, 0], radius=1.0)
    >>> grad = sdf_gradient(points, sdf)
    >>> # grad ≈ [[0.5, 0], [0, 0.5]] (normalized radial direction)

    Use for obstacle avoidance:
    >>> # Gradient points away from obstacle
    >>> obstacle_grad = sdf_gradient(agent_position, obstacle_sdf)
    >>> avoidance_force = -obstacle_grad  # Push away from obstacle
    """
    points = np.asarray(points, dtype=np.float64)
    single_point = points.ndim == 1

    if single_point:
        points = points.reshape(1, -1)

    N, d = points.shape
    grad = np.zeros((N, d), dtype=np.float64)

    def read(rows: NDArray) -> NDArray:
        if single_point:
            return read_region(sdf_func, rows[0], role="sdf_func", one_point=True)
        return read_region(sdf_func, rows, role="sdf_func")

    for i in range(d):
        # Finite difference along axis i
        points_plus = points.copy()
        points_minus = points.copy()
        points_plus[:, i] += epsilon
        points_minus[:, i] -= epsilon

        grad[:, i] = (read(points_plus) - read(points_minus)) / (2 * epsilon)

    if single_point:
        grad = grad.ravel()

    return grad


__all__ = [
    "sdf_box",
    "sdf_complement",
    "sdf_difference",
    "sdf_gradient",
    "sdf_intersection",
    "sdf_smooth_intersection",
    "sdf_smooth_union",
    "sdf_sphere",
    "sdf_union",
]

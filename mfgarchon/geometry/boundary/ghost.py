"""
Ghost point generation for meshfree boundary conditions.

This module provides utilities for creating ghost points via reflection,
which is essential for enforcing Neumann/no-flux boundary conditions
structurally in meshfree methods (GFDM, RBF, KDE).

Architecture:
    - periodic.py: Ghost points for PERIODIC BC (topology-based, shift by period)
    - ghost.py: Ghost points for NEUMANN/NO-FLUX BC (geometry-based, reflection)

The key difference:
    - Periodic: x_ghost = x + L (shift by period length)
    - Reflection: x_ghost = x - 2*(x-x_boundary)·n * n (mirror across tangent plane)

Functions:
    create_reflection_ghost_points: Create ghost points by reflection
    compute_normal_from_sdf: Compute outward normal using SDF gradient
    compute_normal_from_bounds: Compute outward normal for hyperrectangle

Reference:
    - Issue #576: Ghost node architecture
    - Issue #531: GFDM boundary stencil degeneracy fix
    - docs/archive/ghost_nodes_investigation_2025-12/

Created: 2026-01-30
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.geometry.boundary.tolerances import ONWALL_TOL

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def compute_normal_from_bounds(
    point: NDArray[np.floating],
    bounds: list[tuple[float, float]] | NDArray[np.floating],
    tol: float = ONWALL_TOL,
) -> NDArray[np.floating]:
    """
    Compute outward normal vector for a point on hyperrectangle boundary.

    For points on faces, the normal is axis-aligned.
    For corner points (on multiple faces), returns normalized average.

    Args:
        point: Boundary point coordinates, shape (d,)
        bounds: Domain bounds [(xmin, xmax), ...] or (d, 2) array
        tol: Tolerance for boundary detection

    Returns:
        Unit outward normal vector, shape (d,)

    Example:
        >>> bounds = [(0, 1), (0, 1)]
        >>> normal = compute_normal_from_bounds(np.array([0.0, 0.5]), bounds)
        >>> # Returns [-1, 0] (left boundary, normal points outward/left)
    """
    point = np.asarray(point)
    bounds = np.asarray(bounds)
    d = len(point)

    normal = np.zeros(d)

    for dim in range(d):
        low, high = bounds[dim, 0], bounds[dim, 1]
        if abs(point[dim] - low) < tol:
            normal[dim] -= 1.0  # At lower boundary, outward is negative
        if abs(point[dim] - high) < tol:
            normal[dim] += 1.0  # At upper boundary, outward is positive

    # Normalize (handles corners)
    norm = np.linalg.norm(normal)
    if norm > 1e-12:
        normal /= norm

    return normal


def compute_normal_from_sdf(
    point: NDArray[np.floating],
    sdf_func: Callable[[NDArray], NDArray],
    eps: float = 1e-6,
) -> NDArray[np.floating]:
    """
    Compute outward normal vector using SDF gradient.

    For implicit domains, the SDF gradient points outward (toward positive SDF).
    This works for any domain shape, including non-convex.

    Args:
        point: Boundary point coordinates, shape (d,)
        sdf_func: Signed distance function, takes (N, d) returns (N,)
        eps: Finite difference step size

    Returns:
        Unit outward normal vector, shape (d,)

    Example:
        >>> from mfgarchon.geometry.implicit import Hyperrectangle
        >>> domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        >>> normal = compute_normal_from_sdf(np.array([0.0, 0.5]), domain.signed_distance)
        >>> # Returns approximately [-1, 0]
    """
    point = np.asarray(point)
    d = len(point)

    # Compute gradient via central differences
    grad = np.zeros(d)
    for dim in range(d):
        shift = np.zeros(d)
        shift[dim] = eps

        point_plus = (point + shift).reshape(1, -1)
        point_minus = (point - shift).reshape(1, -1)

        sdf_plus = sdf_func(point_plus)[0]
        sdf_minus = sdf_func(point_minus)[0]

        grad[dim] = (sdf_plus - sdf_minus) / (2 * eps)

    # Normalize
    norm = np.linalg.norm(grad)
    if norm > 1e-12:
        grad /= norm

    return grad


def reflect_point_across_plane(
    point: NDArray[np.floating],
    plane_point: NDArray[np.floating],
    plane_normal: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Reflect a point across a plane defined by a point and normal.

    The reflection formula:
        x_reflected = x - 2 * ((x - p) · n) * n

    where p is a point on the plane and n is the unit normal.

    Args:
        point: Point to reflect, shape (d,)
        plane_point: A point on the reflection plane, shape (d,)
        plane_normal: Unit normal to the plane, shape (d,)

    Returns:
        Reflected point, shape (d,)
    """
    offset = point - plane_point
    projection = np.dot(offset, plane_normal)
    return point - 2 * projection * plane_normal


def create_reflection_ghost_points(
    boundary_point: NDArray[np.floating],
    interior_points: NDArray[np.floating],
    normal: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Create ghost points by reflecting interior points across the tangent plane.

    For a boundary point with outward normal n, each interior neighbor x_j
    is reflected to create a ghost point:
        x_ghost = x_j - 2 * ((x_j - x_boundary) · n) * n

    This ensures centroid balance for the boundary point's stencil,
    which is essential for stable GFDM/RBF approximations.

    Args:
        boundary_point: The boundary point (center of stencil), shape (d,)
        interior_points: Interior neighbor points, shape (n_interior, d)
        normal: Unit outward normal at boundary point, shape (d,)

    Returns:
        Ghost points (reflected), shape (n_interior, d)

    Example:
        >>> # Boundary point at left edge
        >>> boundary_pt = np.array([0.0, 0.5])
        >>> interior_pts = np.array([[0.1, 0.5], [0.1, 0.6]])
        >>> normal = np.array([-1.0, 0.0])  # Outward normal
        >>> ghosts = create_reflection_ghost_points(boundary_pt, interior_pts, normal)
        >>> # ghosts ≈ [[-0.1, 0.5], [-0.1, 0.6]]
    """
    boundary_point = np.asarray(boundary_point)
    interior_points = np.atleast_2d(interior_points)
    normal = np.asarray(normal)

    if len(interior_points) == 0:
        return np.zeros((0, len(boundary_point)))

    # Compute offsets from boundary point
    offsets = interior_points - boundary_point

    # Project offsets onto normal direction
    projections = offsets @ normal  # Shape: (n_interior,)

    # Reflect: x_ghost = x_interior - 2 * projection * normal
    ghost_offsets = offsets - 2 * projections[:, np.newaxis] * normal[np.newaxis, :]
    ghost_points = boundary_point + ghost_offsets

    return ghost_points


def create_ghost_stencil(
    boundary_point: NDArray[np.floating],
    neighbor_points: NDArray[np.floating],
    normal: NDArray[np.floating],
    return_interior_mask: bool = False,
) -> (
    tuple[NDArray[np.floating], NDArray[np.floating]]
    | tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.bool_]]
):
    """
    Create augmented stencil with ghost points for a boundary point.

    Given a boundary point and its neighbors (some interior, some on boundary),
    this function:
    1. Identifies interior neighbors (those on the interior side of the tangent plane)
    2. Creates ghost points by reflecting interior neighbors
    3. Returns the augmented stencil (original neighbors + ghost points)

    This is the main entry point for ghost stencil construction in GFDM.

    Args:
        boundary_point: The boundary point (stencil center), shape (d,)
        neighbor_points: All neighbor points, shape (n_neighbors, d)
        normal: Unit outward normal at boundary point, shape (d,)
        return_interior_mask: If True, also return mask identifying interior neighbors

    Returns:
        If return_interior_mask=False:
            (ghost_points, augmented_stencil)
        If return_interior_mask=True:
            (ghost_points, augmented_stencil, interior_mask)

        Where:
            ghost_points: Reflected ghost points, shape (n_ghost, d)
            augmented_stencil: Original + ghost points, shape (n_neighbors + n_ghost, d)
            interior_mask: Boolean mask for interior neighbors, shape (n_neighbors,)

    Example:
        >>> boundary_pt = np.array([0.0, 0.5])
        >>> neighbors = np.array([[0.1, 0.4], [0.1, 0.6], [0.0, 0.4]])  # 2 interior, 1 boundary
        >>> normal = np.array([-1.0, 0.0])
        >>> ghosts, augmented = create_ghost_stencil(boundary_pt, neighbors, normal)
        >>> print(f"Added {len(ghosts)} ghost points")
    """
    boundary_point = np.asarray(boundary_point)
    neighbor_points = np.atleast_2d(neighbor_points)
    normal = np.asarray(normal)

    # Identify interior neighbors (on the interior side of tangent plane)
    # Interior means: (x - boundary_point) · normal < 0
    offsets = neighbor_points - boundary_point
    normal_components = offsets @ normal
    interior_mask = normal_components < -ONWALL_TOL

    # Get interior neighbors
    interior_points = neighbor_points[interior_mask]

    # Create ghost points
    ghost_points = create_reflection_ghost_points(boundary_point, interior_points, normal)

    # Augmented stencil
    if len(ghost_points) > 0:
        augmented_stencil = np.vstack([neighbor_points, ghost_points])
    else:
        augmented_stencil = neighbor_points.copy()

    if return_interior_mask:
        return ghost_points, augmented_stencil, interior_mask
    return ghost_points, augmented_stencil


def create_ghost_points_for_kde(
    query_points: NDArray[np.floating],
    sample_points: NDArray[np.floating],
    bounds: list[tuple[float, float]] | NDArray[np.floating],
    boundary_width: float | None = None,
) -> NDArray[np.floating]:
    """
    Create ghost points for KDE boundary correction.

    For kernel density estimation near boundaries, ghost points are used
    to prevent density underestimation at the boundary. This function
    reflects sample points that are within `boundary_width` of the boundary.

    Args:
        query_points: Points where density will be estimated, shape (n_query, d)
        sample_points: Sample points (particles), shape (n_samples, d)
        bounds: Domain bounds [(xmin, xmax), ...] or (d, 2) array
        boundary_width: Width of boundary region for reflection.
            If None, uses 10% of smallest domain dimension.

    Returns:
        Augmented sample points including ghosts, shape (n_augmented, d)

    Note:
        This is a simplified version for hyperrectangle domains.
        For implicit domains with curved boundaries, use the SDF-based approach.

    See Also:
        mfgarchon.utils.numerical.particle.kde_boundary.reflection_kde
    """
    sample_points = np.atleast_2d(sample_points)
    bounds = np.asarray(bounds)
    _n_samples, d = sample_points.shape

    if boundary_width is None:
        domain_sizes = bounds[:, 1] - bounds[:, 0]
        boundary_width = 0.1 * np.min(domain_sizes)

    ghost_list = [sample_points]  # Start with original points

    # For each dimension, reflect points near boundaries
    for dim in range(d):
        low, high = bounds[dim, 0], bounds[dim, 1]

        # Points near lower boundary
        near_low = sample_points[:, dim] < low + boundary_width
        if np.any(near_low):
            reflected = sample_points[near_low].copy()
            reflected[:, dim] = 2 * low - reflected[:, dim]
            ghost_list.append(reflected)

        # Points near upper boundary
        near_high = sample_points[:, dim] > high - boundary_width
        if np.any(near_high):
            reflected = sample_points[near_high].copy()
            reflected[:, dim] = 2 * high - reflected[:, dim]
            ghost_list.append(reflected)

    return np.vstack(ghost_list)


__all__ = [
    "compute_normal_from_bounds",
    "compute_normal_from_sdf",
    "reflect_point_across_plane",
    "create_reflection_ghost_points",
    "create_ghost_stencil",
    "create_ghost_points_for_kde",
]


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    """Quick smoke test for ghost point utilities."""
    print("Testing ghost point utilities...")

    # Test 1: compute_normal_from_bounds
    print("\n1. compute_normal_from_bounds:")
    bounds_2d = np.array([[0, 1], [0, 1]])

    # Left boundary
    n_left = compute_normal_from_bounds(np.array([0.0, 0.5]), bounds_2d)
    assert np.allclose(n_left, [-1, 0]), f"Expected [-1, 0], got {n_left}"
    print(f"   Left boundary: n = {n_left}")

    # Corner (bottom-left)
    n_corner = compute_normal_from_bounds(np.array([0.0, 0.0]), bounds_2d)
    expected_corner = np.array([-1, -1]) / np.sqrt(2)
    assert np.allclose(n_corner, expected_corner), f"Expected {expected_corner}, got {n_corner}"
    print(f"   Corner (0,0): n = {n_corner}")

    # Test 2: reflect_point_across_plane
    print("\n2. reflect_point_across_plane:")
    p = np.array([0.1, 0.5])
    plane_pt = np.array([0.0, 0.5])
    plane_n = np.array([-1.0, 0.0])
    reflected = reflect_point_across_plane(p, plane_pt, plane_n)
    expected = np.array([-0.1, 0.5])
    assert np.allclose(reflected, expected), f"Expected {expected}, got {reflected}"
    print(f"   Point {p} -> {reflected}")

    # Test 3: create_reflection_ghost_points
    print("\n3. create_reflection_ghost_points:")
    boundary_pt = np.array([0.0, 0.5])
    interior_pts = np.array([[0.1, 0.4], [0.1, 0.6], [0.2, 0.5]])
    normal = np.array([-1.0, 0.0])
    ghosts = create_reflection_ghost_points(boundary_pt, interior_pts, normal)
    expected_ghosts = np.array([[-0.1, 0.4], [-0.1, 0.6], [-0.2, 0.5]])
    assert np.allclose(ghosts, expected_ghosts), f"Expected {expected_ghosts}, got {ghosts}"
    print(f"   Created {len(ghosts)} ghost points")
    for _i, (orig, ghost) in enumerate(zip(interior_pts, ghosts, strict=False)):
        print(f"     {orig} -> {ghost}")

    # Test 4: create_ghost_stencil
    print("\n4. create_ghost_stencil:")
    neighbors = np.array(
        [
            [0.1, 0.4],  # Interior
            [0.1, 0.6],  # Interior
            [0.0, 0.4],  # On boundary (not interior)
            [0.0, 0.6],  # On boundary (not interior)
        ]
    )
    ghosts, augmented, interior_mask = create_ghost_stencil(boundary_pt, neighbors, normal, return_interior_mask=True)
    print(f"   Original neighbors: {len(neighbors)}")
    print(f"   Interior neighbors: {np.sum(interior_mask)}")
    print(f"   Ghost points added: {len(ghosts)}")
    print(f"   Augmented stencil size: {len(augmented)}")

    # Test 5: create_ghost_points_for_kde
    print("\n5. create_ghost_points_for_kde:")
    samples = np.random.rand(100, 2)
    augmented_samples = create_ghost_points_for_kde(
        query_points=samples,  # Not used in current impl
        sample_points=samples,
        bounds=bounds_2d,
        boundary_width=0.1,
    )
    print(f"   Original samples: {len(samples)}")
    print(f"   Augmented samples: {len(augmented_samples)}")
    print(f"   Ghost points added: {len(augmented_samples) - len(samples)}")

    print("\n" + "=" * 60)
    print("All ghost point utility tests passed!")
    print("=" * 60)

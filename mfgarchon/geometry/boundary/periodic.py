"""
Periodic boundary condition utilities (Issue #711).

This module provides utilities for PERIODIC boundary conditions, parallel to
enforcement.py which handles DIRICHLET/NEUMANN/ROBIN.

Architecture:
    BCType.DIRICHLET/NEUMANN/ROBIN → enforcement.py (value-based)
    BCType.PERIODIC                → periodic.py (topology-based)

The key difference is that periodic BCs work at the TOPOLOGY level
(index wrapping, ghost point augmentation) rather than computing
ghost cell VALUES like other BC types.

Functions:
    wrap_positions: Wrap particle positions to fundamental domain
    wrap_displacement: Wrap a displacement vector to the minimum image
    create_periodic_ghost_points: Augment point cloud for KD-tree search

Reference:
    - Issue #711: Periodic BC support for GFDM
    - geometry/protocols/topology.py: SupportsPeriodic protocol

Created: 2026-01-29 (Issue #711 refactor)
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def wrap_positions(
    positions: NDArray[np.floating],
    bounds: list[tuple[float, float]] | NDArray[np.floating],
    periodic_dims: tuple[int, ...] | None = None,
) -> NDArray[np.floating]:
    """
    Wrap positions around domain boundaries (periodic BC, n-D).

    For periodic dimension i with period L_i = xmax_i - xmin_i:
        x_wrapped = xmin + (x - xmin) mod L

    Args:
        positions: Particle positions, shape (N, d) or (d,) for single point
        bounds: Domain bounds as [(xmin, xmax), (ymin, ymax), ...] or (d, 2) array
        periodic_dims: Dimensions to wrap. If None, all dimensions are wrapped.

    Returns:
        Wrapped positions in [xmin, xmax), same shape as input

    Examples:
        >>> positions = np.array([[1.5, 0.5], [-0.3, 0.5]])
        >>> bounds = [(0, 1), (0, 1)]
        >>> wrapped = wrap_positions(positions, bounds)
        >>> # (1.5, 0.5) -> (0.5, 0.5), (-0.3, 0.5) -> (0.7, 0.5)

        >>> # Cylinder: periodic in x only
        >>> wrapped = wrap_positions(positions, bounds, periodic_dims=(0,))

    Note:
        This is the canonical utility for particle position wrapping.
        Used by MeshfreeApplicator._apply_periodic_bc() and others.
    """
    positions = np.atleast_2d(positions)
    was_1d = positions.shape[0] == 1 and len(positions.shape) == 2
    result = positions.copy()

    bounds = np.asarray(bounds)
    if bounds.ndim == 1:
        bounds = bounds.reshape(1, 2)

    ndim = result.shape[1]
    if bounds.shape[0] != ndim:
        raise ValueError(f"Bounds dimension {bounds.shape[0]} != positions dimension {ndim}")

    # Default: all dimensions periodic
    if periodic_dims is None:
        periodic_dims = tuple(range(ndim))

    for d in periodic_dims:
        xmin, xmax = bounds[d, 0], bounds[d, 1]
        Lx = xmax - xmin
        if Lx > 1e-14:
            result[:, d] = xmin + ((result[:, d] - xmin) % Lx)

    if was_1d:
        return result[0]
    return result


def wrap_displacement(
    delta: NDArray[np.floating],
    periods: dict[int, float],
) -> NDArray[np.floating]:
    """
    Wrap a displacement vector to the minimum image, for periodic dimensions only.

    For periodic dimension i with period L_i:
        delta_wrapped[i] = delta[i] - L_i * round(delta[i] / L_i)

    which maps the displacement into [-L_i/2, L_i/2] -- the shortest way round the torus.

    Takes `periods` rather than a geometry, because that dict is exactly what the
    :class:`~mfgarchon.geometry.protocols.topology.SupportsPeriodic` protocol promises
    (``get_periods()``). Consumers can therefore work with ANY geometry satisfying the protocol.
    The previous arrangement -- an identical loop living as ``Hyperrectangle.wrap_displacement`` --
    tied the one caller to the one class that happened to implement it, and `TensorProductGrid`
    satisfies the protocol without it, so a periodic grid passed the `isinstance` guard and then
    raised `AttributeError` (Issue #1841).

    Args:
        delta: Displacement vectors, shape (N, d) or (d,) for a single vector
        periods: Dimension index -> period length, for periodic dimensions only. An empty
            dict means nothing is periodic and the input is returned unchanged.

    Returns:
        Wrapped displacement, same shape as the input.

    Example:
        >>> import numpy as np
        >>> wrap_displacement(np.array([0.9]), {0: 1.0})   # shorter to go backwards
        array([-0.1])
    """
    if delta.ndim > 2:
        # `wrapped[:, dim_idx]` below indexes axis 1, which is the coordinate axis only for
        # rank 1 and 2. Silently wrapping one slab of a rank-3 stack returns a correctly
        # SHAPED array of wrong distances -- the failure mode this function was made the
        # single owner to prevent -- so refuse instead of guessing (Issue #1853).
        raise ValueError(
            f"wrap_displacement expects shape (N, d) or (d,), got ndim={delta.ndim} "
            f"with shape {delta.shape}. Reshape to (N, d) and restore the leading axes afterwards."
        )

    if not periods:
        return delta

    single_vector = delta.ndim == 1
    if single_vector:
        delta = delta.reshape(1, -1)

    # float, not delta.dtype: an integer displacement array would truncate the wrapped value on
    # write-back, so two points half a non-integral period apart report distance 0 (Issue #1853).
    wrapped = np.asarray(delta, dtype=float)
    wrapped = wrapped.copy() if wrapped is delta else wrapped
    for dim_idx, length in periods.items():
        wrapped[:, dim_idx] = wrapped[:, dim_idx] - length * np.round(wrapped[:, dim_idx] / length)

    return wrapped[0] if single_vector else wrapped


def periodic_distance(
    points1: NDArray[np.floating],
    points2: NDArray[np.floating],
    periods: dict[int, float],
) -> NDArray[np.floating]:
    """
    Euclidean distance with periodic axes reduced to the minimum image.

    Single owner of the *distance* form, as :func:`wrap_displacement` is of the displacement
    form. Both ``Hyperrectangle`` and ``TensorProductGrid`` delegate here; keeping a copy of
    this body in each class is what let one rule become three in the first place (Issue #1853).

    Args:
        points1: Shape (N, d) or (d,).
        points2: Same shape as points1, or broadcastable to it.
        periods: Dimension index -> period length, for periodic dimensions only. Empty means
            nothing is periodic and the plain Euclidean distance is returned.

    Returns:
        Distances of shape (N,), or a scalar when ``points1`` is a single (d,) vector.
    """
    diff = np.asarray(points1, dtype=float) - np.asarray(points2, dtype=float)

    # Keyed on points1, not on diff: a (d,) query against a (1, d) reference broadcasts to a
    # rank-2 diff, and the documented contract is that a single query point returns a scalar.
    single_point = np.ndim(points1) == 1
    if diff.ndim == 1:
        diff = diff.reshape(1, -1)

    distances = np.linalg.norm(wrap_displacement(diff, periods), axis=1)
    return distances[0] if single_point and distances.shape == (1,) else distances


def create_periodic_ghost_points(
    points: NDArray[np.floating],
    bounds: list[tuple[float, float]] | NDArray[np.floating],
    periodic_dims: tuple[int, ...] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.int64]]:
    """
    Create augmented point cloud with ghost copies for periodic neighbor search.

    For meshfree methods (GFDM, RBF) on periodic domains, KD-tree neighbor
    search requires ghost points near domain boundaries to find periodic
    neighbors correctly.

    For d-dimensional domain with |P| periodic dimensions, creates 3^|P|
    copies of the point cloud shifted by +/- L in each periodic direction.

    Args:
        points: Original points, shape (n_points, dimension)
        bounds: Domain bounds as [(xmin, xmax), ...] or (d, 2) array
        periodic_dims: Dimensions with periodic topology. If None, all dims periodic.

    Returns:
        Tuple of (augmented_points, original_indices):
            - augmented_points: Shape (n_augmented, dimension)
            - original_indices: Maps augmented index -> original point index

    Examples:
        >>> # 2D torus - creates 9 copies (3^2)
        >>> points = np.random.rand(100, 2)
        >>> bounds = [(0, 1), (0, 1)]
        >>> aug_pts, orig_idx = create_periodic_ghost_points(points, bounds)
        >>> # aug_pts.shape == (900, 2), orig_idx.shape == (900,)

        >>> # Cylinder - periodic in x only (3 copies)
        >>> aug_pts, orig_idx = create_periodic_ghost_points(
        ...     points, bounds, periodic_dims=(0,)
        ... )
        >>> # aug_pts.shape == (300, 2)

    Note:
        Issue #711: Canonical utility for periodic meshfree methods.
        Used by TaylorOperator for GFDM on periodic domains.
    """
    points = np.atleast_2d(points)
    n_points, ndim = points.shape

    bounds = np.asarray(bounds)
    if bounds.ndim == 1:
        bounds = bounds.reshape(1, 2)

    # Default: all dimensions periodic
    if periodic_dims is None:
        periodic_dims = tuple(range(ndim))

    if not periodic_dims:
        # No periodicity
        return points, np.arange(n_points, dtype=np.int64)

    # Generate shift combinations for periodic dimensions
    shift_options = []
    for d in range(ndim):
        if d in periodic_dims:
            shift_options.append([-1, 0, 1])
        else:
            shift_options.append([0])

    shifts = list(itertools.product(*shift_options))

    # Period lengths
    period_lengths = np.array([bounds[d, 1] - bounds[d, 0] for d in range(ndim)])

    augmented_list = []
    index_list = []

    for shift in shifts:
        shift_vec = np.array(shift) * period_lengths
        shifted_points = points + shift_vec
        augmented_list.append(shifted_points)
        index_list.append(np.arange(n_points, dtype=np.int64))

    return np.vstack(augmented_list), np.concatenate(index_list)


def create_reflection_ghost_points(
    points: NDArray[np.floating],
    bounds: list[tuple[float, float]] | NDArray[np.floating],
    margin: float,
    reflect_dims: tuple[int, ...] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.int64]]:
    """
    Create augmented point cloud with reflected ghost copies for boundary KDE.

    For KDE at domain boundaries, the kernel extends into "empty" space outside
    the domain, causing ~50% density underestimation. Reflecting particles about
    boundaries fixes this bias for no-flux/Neumann BC (Issue #709).

    Unlike periodic ghost points which create 3^|P| copies, reflection only
    creates copies for particles within `margin` of each boundary face.

    Args:
        points: Original points, shape (n_points, dimension)
        bounds: Domain bounds as [(xmin, xmax), ...] or (d, 2) array
        margin: Distance from boundary to include particles for reflection.
            Typically 3*bandwidth for KDE.
        reflect_dims: Dimensions with reflecting boundaries. If None, all dims.

    Returns:
        Tuple of (augmented_points, original_indices):
            - augmented_points: Shape (n_augmented, dimension)
            - original_indices: Maps augmented index -> original point index

    Examples:
        >>> # 2D domain with reflecting boundaries
        >>> points = np.random.rand(100, 2)
        >>> bounds = [(0, 1), (0, 1)]
        >>> aug_pts, orig_idx = create_reflection_ghost_points(
        ...     points, bounds, margin=0.1
        ... )
        >>> # Only particles near boundaries are reflected

    Note:
        Issue #709: Boundary KDE correction for particle methods.
        For periodic BC, use create_periodic_ghost_points() instead.
    """
    points = np.atleast_2d(points)
    n_points, ndim = points.shape

    bounds = np.asarray(bounds)
    if bounds.ndim == 1:
        bounds = bounds.reshape(1, 2)

    # Default: all dimensions have reflecting boundaries
    if reflect_dims is None:
        reflect_dims = tuple(range(ndim))

    if not reflect_dims:
        # No reflection needed
        return points, np.arange(n_points, dtype=np.int64)

    augmented_list = [points]
    index_list = [np.arange(n_points, dtype=np.int64)]

    for d in reflect_dims:
        xmin, xmax = bounds[d, 0], bounds[d, 1]

        # Find particles near left boundary (within margin of xmin)
        near_left_mask = points[:, d] < xmin + margin
        if np.any(near_left_mask):
            pts_near_left = points[near_left_mask].copy()
            # Reflect about xmin: x_reflected = 2*xmin - x
            pts_near_left[:, d] = 2 * xmin - pts_near_left[:, d]
            augmented_list.append(pts_near_left)
            index_list.append(np.where(near_left_mask)[0].astype(np.int64))

        # Find particles near right boundary (within margin of xmax)
        near_right_mask = points[:, d] > xmax - margin
        if np.any(near_right_mask):
            pts_near_right = points[near_right_mask].copy()
            # Reflect about xmax: x_reflected = 2*xmax - x
            pts_near_right[:, d] = 2 * xmax - pts_near_right[:, d]
            augmented_list.append(pts_near_right)
            index_list.append(np.where(near_right_mask)[0].astype(np.int64))

    return np.vstack(augmented_list), np.concatenate(index_list)


__all__ = [
    "wrap_positions",
    "wrap_displacement",
    "create_periodic_ghost_points",
    "create_reflection_ghost_points",
]


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    """Quick smoke test for periodic boundary utilities."""
    print("Testing periodic boundary utilities...")

    bounds_2d = [(0, 1), (0, 1)]

    # Test wrap_positions
    print("\n1. wrap_positions:")
    positions_wrap = np.array([[1.5, 0.5], [-0.3, 0.5]])
    wrapped = wrap_positions(positions_wrap, bounds_2d)
    expected_wrap = np.array([[0.5, 0.5], [0.7, 0.5]])
    assert np.allclose(wrapped, expected_wrap), f"wrap failed: {wrapped}"
    print("   Full wrap: passed")

    # Test partial periodic (cylinder)
    wrapped_partial = wrap_positions(positions_wrap, bounds_2d, periodic_dims=(0,))
    expected_partial = np.array([[0.5, 0.5], [0.7, 0.5]])  # y unchanged
    assert np.allclose(wrapped_partial, expected_partial)
    print("   Partial wrap (cylinder): passed")

    # Test single point
    single = np.array([1.5, 0.5])
    single_wrapped = wrap_positions(single, bounds_2d)
    assert single_wrapped.shape == (2,)
    assert np.allclose(single_wrapped, [0.5, 0.5])
    print("   Single point: passed")

    # Test create_periodic_ghost_points
    print("\n2. create_periodic_ghost_points:")
    points = np.random.rand(10, 2)

    # Full torus (3^2 = 9 copies)
    aug, idx = create_periodic_ghost_points(points, bounds_2d)
    assert aug.shape == (90, 2), f"Expected (90, 2), got {aug.shape}"
    assert idx.shape == (90,)
    print(f"   2D torus: {len(points)} -> {len(aug)} points (9x)")

    # Cylinder (3^1 = 3 copies)
    aug_cyl, idx_cyl = create_periodic_ghost_points(points, bounds_2d, periodic_dims=(0,))
    assert aug_cyl.shape == (30, 2), f"Expected (30, 2), got {aug_cyl.shape}"
    print(f"   Cylinder: {len(points)} -> {len(aug_cyl)} points (3x)")

    # Non-periodic (1 copy)
    aug_none, idx_none = create_periodic_ghost_points(points, bounds_2d, periodic_dims=())
    assert aug_none.shape == (10, 2)
    print(f"   Non-periodic: {len(points)} -> {len(aug_none)} points (1x)")

    # Test create_reflection_ghost_points (Issue #709)
    print("\n3. create_reflection_ghost_points:")

    # Create particles, some near boundaries
    np.random.seed(42)
    particles = np.random.rand(100, 2)

    # Reflection with margin=0.1
    margin = 0.1
    aug_reflect, idx_reflect = create_reflection_ghost_points(particles, bounds_2d, margin=margin)
    n_ghosts = len(aug_reflect) - len(particles)
    print(f"   {len(particles)} particles -> {len(aug_reflect)} with reflection (margin={margin})")
    print(f"   Added {n_ghosts} reflection ghosts")

    # Verify reflected points are outside domain
    outside_left_x = aug_reflect[len(particles) :, 0] < 0
    outside_right_x = aug_reflect[len(particles) :, 0] > 1
    outside_left_y = aug_reflect[len(particles) :, 1] < 0
    outside_right_y = aug_reflect[len(particles) :, 1] > 1
    # Reflected points should be outside domain in at least one dimension
    # (some may be outside in x, some in y)
    print(f"   Ghosts outside domain: {np.sum(outside_left_x | outside_right_x | outside_left_y | outside_right_y)}")

    # Test with larger margin
    margin_large = 0.3
    aug_large, idx_large = create_reflection_ghost_points(particles, bounds_2d, margin=margin_large)
    print(f"   With margin={margin_large}: {len(aug_large)} total")
    assert len(aug_large) > len(aug_reflect), "Larger margin should include more ghosts"

    # Test partial reflection (only x dimension)
    aug_x_only, idx_x_only = create_reflection_ghost_points(particles, bounds_2d, margin=0.1, reflect_dims=(0,))
    print(f"   Reflect x only: {len(aug_x_only)} total")

    print("   Reflection ghost points: passed")

    print("\nAll periodic/reflection utilities tests passed!")

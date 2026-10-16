"""
Dimension-agnostic boundary condition handling for particle-based FP solvers (Issue #635).

This module provides unified functions that work for any dimension:
- Topology detection from boundary conditions
- Boundary enforcement (periodic wrap, reflecting bounce)
- A reflected step's wall hits, the path the obstacle placement reads (`reflection_waypoints`)
- Segment-aware BC checking

All functions accept both 1D and nD inputs with consistent APIs.

Usage:
    from mfgarchon.alg.numerical.fp_solvers.fp_particle_bc import (
        get_topology_per_dimension,
        apply_boundary_conditions,
        needs_segment_aware_bc,
    )
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mfgarchon.geometry.boundary.conditions import BoundaryConditions, periodic_faces
from mfgarchon.geometry.boundary.types import BCType

# =============================================================================
# Topology Detection (unified)
# =============================================================================


def get_topology_per_dimension(
    boundary_conditions: BoundaryConditions | str | None,
    dimension: int,
    geometry: Any = None,
) -> list[str]:
    """
    Get grid topology for each dimension from boundary conditions (dimension-agnostic).

    This determines the INDEXING STRATEGY for particles, not the physical BC:
    - "periodic": Space wraps around (particles use modular arithmetic)
    - "bounded": Space has walls (particles reflect at boundaries)

    Note: This is about topology (how space connects), not physics (what values
    are prescribed). For particles, all non-periodic boundaries are treated as
    reflecting walls, regardless of whether the underlying BC is Dirichlet,
    Neumann, Robin, or no-flux.

    Parameters
    ----------
    boundary_conditions : BoundaryConditions or None
        Boundary condition specification. A torus's is ``periodic_bc``, which the problem's BC owner
        answers for a geometry periodic on every axis (#2530, row B9).
    dimension : int
        Number of spatial dimensions
    geometry : the solver's grid, on which the face owner decides a region or sdf_region placement

    Returns
    -------
    topologies : list[str]
        Topology per dimension: ["periodic", "bounded", ...]

    Examples
    --------
    >>> from mfgarchon.geometry.boundary import periodic_bc, neumann_bc
    >>> get_topology_per_dimension(periodic_bc(1), 1)  # ["periodic"]
    >>> get_topology_per_dimension(neumann_bc(2), 2)   # ["bounded", "bounded"]
    """
    # Default to bounded (reflecting walls) for all dimensions
    topologies = ["bounded"] * dimension

    if boundary_conditions is None:
        return topologies

    bc = boundary_conditions
    if not isinstance(bc, BoundaryConditions):
        raise TypeError(
            "get_topology_per_dimension takes a BoundaryConditions or None, not "
            f"{type(bc).__name__} ({bc!r}): read as bounded walls, it would solve a boundary nobody declared "
            "(#2634, review 2)."
        )

    # One condition everywhere: read with no placement, so no geometry is needed.
    uniform = bc.uniform_segment
    if uniform is not None:
        return ["periodic" if uniform.bc_type == BCType.PERIODIC else "bounded"] * dimension

    # Read through the face owner with periodicity as the equivalence (`periodic_faces`, #2512 row B1): the topology
    # asks only whether an axis wraps, so a segment covering part of a face is accepted where it agrees with the rest of
    # the face on that, and refused where it does not. Periodic topology needs both ends of an axis periodic.
    periodic = periodic_faces(bc, dimension, geometry)
    return ["periodic" if periodic[2 * d] and periodic[2 * d + 1] else "bounded" for d in range(dimension)]


def needs_segment_aware_bc(boundary_conditions: BoundaryConditions | None) -> bool:
    """
    Check if boundary conditions require segment-aware handling.

    Returns True if:
    - BC has multiple segments with different types
    - BC has absorbing (DIRICHLET) segments that should remove particles

    Returns False for uniform periodic/reflecting BC where fast path can be used.

    Parameters
    ----------
    boundary_conditions : BoundaryConditions or None
        Boundary condition specification.

    Returns
    -------
    bool
        True if segment-aware handling is needed

    Examples
    --------
    >>> needs_segment_aware_bc(periodic_bc(1))  # False (uniform)
    >>> needs_segment_aware_bc(mixed_bc_with_exit)  # True (has DIRICHLET)
    """
    if boundary_conditions is None:
        return False

    bc = boundary_conditions
    if isinstance(bc, str):
        raise ValueError(f"needs_segment_aware_bc: a BC is a BoundaryConditions, not the string {bc!r}")

    # Check if uniform BC (same type everywhere) - Issue #543: use getattr instead of hasattr
    is_uniform = getattr(bc, "is_uniform", False)
    if is_uniform:
        # Uniform BC - use fast path unless it's DIRICHLET (absorbing). One unplaced segment is what
        # `is_uniform` means, and `uniform_segment` is that segment.
        segment = bc.uniform_segment
        return segment is not None and segment.bc_type == BCType.DIRICHLET

    # Mixed BC with segments - need segment-aware handling
    segments = getattr(bc, "segments", [])
    if len(segments) > 1:
        return True

    # Check for DIRICHLET segments (absorbing BC)
    return any(segment.bc_type == BCType.DIRICHLET for segment in segments)


# =============================================================================
# Boundary Enforcement (unified)
# =============================================================================


def apply_boundary_conditions(
    particles: np.ndarray,
    bounds: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    topology: str | list[str] = "bounded",
) -> np.ndarray:
    """
    Apply boundary handling per dimension based on topology (dimension-agnostic).

    This function delegates to the canonical implementations in
    mfgarchon.utils.numerical.particle.boundary (Issue #521).

    At corners, all dimensions are processed simultaneously (not sequentially),
    producing diagonal reflection. This is equivalent to 'average' corner
    strategy for position-based reflection.

    Parameters
    ----------
    particles : np.ndarray
        Particle positions.
        - 1D: shape (num_particles,) or (num_particles, 1)
        - nD: shape (num_particles, dimension)
    bounds : list of tuple or tuple of tuple
        Bounds per dimension [(xmin, xmax), (ymin, ymax), ...]
    topology : str or list[str]
        Grid topology: "periodic" (wrap) or "bounded" (reflect).
        Can be a single string (same for all dims) or per-dimension list.

    Returns
    -------
    particles : np.ndarray
        Updated particle positions (may be modified in place for efficiency)

    Examples
    --------
    >>> particles_1d = np.array([1.5, -0.2, 0.5])  # Some outside [0, 1]
    >>> apply_boundary_conditions(particles_1d, [(0, 1)], "bounded")
    >>> # particles_1d now reflected into [0, 1]

    Note
    ----
    See Issue #521 for corner handling architecture. Position-based reflection
    uses fold reflection which implicitly produces diagonal corner reflection.
    """
    from mfgarchon.geometry.boundary.corner import reflect_positions, wrap_positions

    particles = np.asarray(particles)
    bounds_list = list(bounds)

    # Handle 1D case: convert to 2D for uniform processing
    is_1d_flat = particles.ndim == 1
    if is_1d_flat:
        particles = particles[:, np.newaxis]

    dimension = particles.shape[1] if particles.ndim > 1 else 1

    # Handle per-dimension topologies
    if isinstance(topology, str):
        topologies = [topology] * dimension
    else:
        topologies = list(topology)

    # Check if all dimensions have same topology (common case)
    all_periodic = all(t == "periodic" for t in topologies)
    all_bounded = all(t != "periodic" for t in topologies)

    if all_periodic:
        # Fast path: all periodic
        result = wrap_positions(particles, bounds_list)
    elif all_bounded:
        # Fast path: all bounded (reflecting)
        result = reflect_positions(particles, bounds_list)
    else:
        # Mixed topology: process dimension by dimension
        result = particles.copy()
        for d in range(dimension):
            xmin, xmax = bounds_list[d]
            Lx = xmax - xmin

            if Lx < 1e-14:
                continue  # Skip degenerate dimension

            dim_topology = topologies[d] if d < len(topologies) else "bounded"

            if dim_topology == "periodic":
                # Periodic: wrap around
                result[:, d] = xmin + (result[:, d] - xmin) % Lx
            else:
                # Bounded: fold reflection
                shifted = result[:, d] - xmin
                period = 2 * Lx
                pos_in_period = shifted % period
                in_second_half = pos_in_period > Lx
                pos_in_period[in_second_half] = period - pos_in_period[in_second_half]
                result[:, d] = xmin + pos_in_period

    # Convert back to 1D if input was 1D
    if is_1d_flat:
        return result.ravel()

    return result


# =============================================================================
# Coordinate/Grid Utilities (unified)
# =============================================================================


def create_coordinate_arrays(
    bounds: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    grid_shape: tuple[int, ...],
) -> list[np.ndarray]:
    """
    Create coordinate arrays from bounds and grid shape (dimension-agnostic).

    Parameters
    ----------
    bounds : list or tuple of (min, max) tuples
        Domain bounds per dimension
    grid_shape : tuple[int, ...]
        Number of grid points per dimension

    Returns
    -------
    coordinates : list[np.ndarray]
        1D coordinate arrays for each dimension

    Examples
    --------
    >>> coords = create_coordinate_arrays([(0, 1), (0, 2)], (11, 21))
    >>> coords[0]  # array([0.0, 0.1, ..., 1.0])
    >>> coords[1]  # array([0.0, 0.1, ..., 2.0])
    """
    coordinates = []
    for d, (bmin, bmax) in enumerate(bounds):
        n_points = grid_shape[d]
        coords = np.linspace(bmin, bmax, n_points)
        coordinates.append(coords)
    return coordinates


def compute_spacings(
    bounds: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    grid_shape: tuple[int, ...],
) -> list[float]:
    """
    Compute grid spacings from bounds and shape (dimension-agnostic).

    Parameters
    ----------
    bounds : list or tuple of (min, max) tuples
        Domain bounds per dimension
    grid_shape : tuple[int, ...]
        Number of grid points per dimension

    Returns
    -------
    spacings : list[float]
        Grid spacing per dimension [dx, dy, ...]

    Examples
    --------
    >>> compute_spacings([(0, 1), (0, 2)], (11, 21))
    [0.1, 0.1]
    """
    spacings = []
    for d, (bmin, bmax) in enumerate(bounds):
        n_points = grid_shape[d]
        if n_points > 1:
            dx = (bmax - bmin) / (n_points - 1)
        else:
            dx = bmax - bmin  # Single point
        spacings.append(dx)
    return spacings


# =============================================================================
# Smoke Tests
# =============================================================================

if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing unified fp_particle_bc functions...")

    # Test 1: Topology detection
    # We can't easily test with real BoundaryConditions without imports
    # So test the None case and basic logic
    topo_none = get_topology_per_dimension(None, 2)
    assert topo_none == ["bounded", "bounded"], f"Expected bounded, got {topo_none}"
    print("  get_topology_per_dimension (None BC): OK")

    # Test 2: needs_segment_aware_bc
    result = needs_segment_aware_bc(None)
    assert result is False, "None BC should not need segment-aware"
    print("  needs_segment_aware_bc (None BC): OK")

    # Test 3: Boundary conditions - bounded (reflecting)
    # Test 1D
    particles_1d = np.array([-0.1, 0.5, 1.1, 1.5, 2.5])
    result_1d = apply_boundary_conditions(particles_1d, [(0.0, 1.0)], "bounded")
    assert result_1d.shape == (5,), f"Expected (5,), got {result_1d.shape}"
    assert np.all(result_1d >= 0), f"Negative values: {result_1d}"
    assert np.all(result_1d <= 1), f"Values > 1: {result_1d}"
    print("  apply_boundary_conditions (1D bounded): OK")

    # Test 2D
    particles_2d = np.array(
        [
            [-0.1, 0.5],  # x out of bounds (left)
            [0.5, -0.2],  # y out of bounds (bottom)
            [1.1, 0.5],  # x out of bounds (right)
            [0.5, 1.3],  # y out of bounds (top)
            [0.5, 0.5],  # inside
        ]
    )
    result_2d = apply_boundary_conditions(particles_2d, [(0.0, 1.0), (0.0, 1.0)], "bounded")
    assert result_2d.shape == (5, 2), f"Expected (5, 2), got {result_2d.shape}"
    assert np.all(result_2d >= 0), f"Negative values: {result_2d}"
    assert np.all(result_2d <= 1), f"Values > 1: {result_2d}"
    print("  apply_boundary_conditions (2D bounded): OK")

    # Test 4: Boundary conditions - periodic
    particles_periodic = np.array([-0.1, 0.5, 1.1, 1.5])
    result_periodic = apply_boundary_conditions(particles_periodic, [(0.0, 1.0)], "periodic")
    assert np.all(result_periodic >= 0), f"Negative values: {result_periodic}"
    assert np.all(result_periodic <= 1), f"Values > 1: {result_periodic}"
    # -0.1 should wrap to 0.9, 1.1 should wrap to 0.1
    assert np.abs(result_periodic[0] - 0.9) < 1e-10, f"Expected 0.9, got {result_periodic[0]}"
    assert np.abs(result_periodic[2] - 0.1) < 1e-10, f"Expected 0.1, got {result_periodic[2]}"
    print("  apply_boundary_conditions (1D periodic): OK")

    # Test 5: Mixed topology (x periodic, y bounded)
    particles_mixed = np.array(
        [
            [-0.1, -0.1],  # Both out
            [1.1, 1.1],  # Both out
        ]
    )
    result_mixed = apply_boundary_conditions(particles_mixed, [(0.0, 1.0), (0.0, 1.0)], ["periodic", "bounded"])
    # x should wrap, y should reflect
    assert np.abs(result_mixed[0, 0] - 0.9) < 1e-10, f"x periodic: expected 0.9, got {result_mixed[0, 0]}"
    assert np.abs(result_mixed[0, 1] - 0.1) < 1e-10, f"y bounded: expected 0.1, got {result_mixed[0, 1]}"
    print("  apply_boundary_conditions (mixed periodic/bounded): OK")

    # Test 6: Coordinate array creation
    coords = create_coordinate_arrays([(0.0, 1.0), (0.0, 2.0)], (11, 21))
    assert len(coords) == 2
    assert len(coords[0]) == 11
    assert len(coords[1]) == 21
    assert np.abs(coords[0][-1] - 1.0) < 1e-10
    assert np.abs(coords[1][-1] - 2.0) < 1e-10
    print("  create_coordinate_arrays: OK")

    # Test 7: Spacing computation
    spacings = compute_spacings([(0.0, 1.0), (0.0, 2.0)], (11, 21))
    assert len(spacings) == 2
    assert np.abs(spacings[0] - 0.1) < 1e-10
    assert np.abs(spacings[1] - 0.1) < 1e-10
    print("  compute_spacings: OK")

    print("\nAll smoke tests passed!")


def reflection_waypoints(previous: np.ndarray, raw: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray | None:
    """The wall hits of each step that left the box, folded back in: the vertices of its reflected path between
    ``previous`` and its reflected end (#2706, review 1).

    The obstacle placement names the face a step met. For a reflected step that is a face on the broken path
    ``previous`` -> wall hit -> ... -> reflected end, not on the chord from ``previous`` to the reflected end, which
    can cross a face the particle never approached. The hits are where the unfolded straight step crosses a wall
    plane, folded with `reflect_positions`, the fold the boundary condition itself applies.

    Returns shape (N, k, dimension), rows of NaN where a step has fewer than ``k`` hits, or None when no step left
    the box.
    """
    from mfgarchon.geometry.boundary.corner import reflect_positions

    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    length = hi - lo
    prev = np.asarray(previous, dtype=float).reshape(-1, len(bounds))
    end = np.asarray(raw, dtype=float).reshape(prev.shape)
    left = np.flatnonzero(np.any((end < lo) | (end > hi), axis=1))
    if left.size == 0:
        return None
    hits = []
    for i in left:
        a, b = prev[i], end[i]
        ts = []
        for k in range(prev.shape[1]):
            if b[k] == a[k]:
                continue
            first = np.ceil((min(a[k], b[k]) - lo[k]) / length[k])
            last = np.floor((max(a[k], b[k]) - lo[k]) / length[k])
            for j in np.arange(first, last + 1):
                t = (lo[k] + j * length[k] - a[k]) / (b[k] - a[k])
                if 0.0 < t < 1.0:
                    ts.append(t)
        hits.append(sorted(ts))
    via = np.full((len(prev), max(len(t) for t in hits), prev.shape[1]), np.nan)
    for i, ts in zip(left, hits, strict=True):
        if ts:
            unfolded = prev[i][None, :] + np.asarray(ts)[:, None] * (end[i] - prev[i])[None, :]
            via[i, : len(ts)] = reflect_positions(unfolded, bounds)
    return via

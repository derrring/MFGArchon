"""
Point cloud geometry for particle-based MFG solvers.

This module provides a lightweight geometry wrapper for particle arrays,
enabling particle-particle projection in hybrid MFG solvers (e.g., GFDM + Particle FP).

Two patterns are supported:
1. Fixed collocation + moving particles: Requires projection between different point sets
2. Co-moving particles: Identity projection when same particle set is used

Created: 2025-11-12
Part of: Issue #269 - Particle-Particle Projection Support
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.geometry.boundary.tolerances import ONWALL_TOL
from mfgarchon.geometry.protocol import GeometryProtocol, GeometryType

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PointCloudGeometry:
    """
    Lightweight geometry wrapper for particle arrays.

    This class wraps particle position arrays to enable them to be used with
    the GeometryProjector system for particle-particle projections.

    Attributes:
        positions: Particle positions (N, dimension)
        dimension: Spatial dimension
        num_particles: Number of particles
        metadata: Optional metadata (e.g., particle IDs, weights)

    Examples:
        >>> # Fixed collocation points for GFDM
        >>> collocation = np.random.uniform(0, 1, (500, 2))
        >>> hjb_geometry = PointCloudGeometry(collocation)
        >>>
        >>> # Evolving particles for FP
        >>> particles = np.random.uniform(0, 1, (1000, 2))
        >>> fp_geometry = PointCloudGeometry(particles)
        >>>
        >>> # Use with GeometryProjector
        >>> from mfgarchon.geometry import GeometryProjector
        >>> projector = GeometryProjector(hjb_geometry, fp_geometry)
        >>> U_on_particles = projector.project_hjb_to_fp(U_on_collocation)
    """

    def __init__(
        self,
        positions: NDArray[np.floating],
        metadata: dict | None = None,
    ):
        """
        Initialize point cloud geometry.

        Args:
            positions: Particle positions (N, dimension) or (N,) for 1D
            metadata: Optional metadata dict (particle IDs, weights, etc.)

        Raises:
            ValueError: If positions have invalid shape
        """
        self.positions = np.asarray(positions, dtype=float)

        # Handle 1D case
        if self.positions.ndim == 1:
            self.positions = self.positions.reshape(-1, 1)

        if self.positions.ndim != 2:
            raise ValueError(f"Positions must be 2D array (N, dimension), got shape {self.positions.shape}")

        self.metadata = metadata or {}

    @property
    def dimension(self) -> int:
        """Spatial dimension of the point cloud."""
        return self.positions.shape[1]

    @property
    def num_particles(self) -> int:
        """Number of particles in the cloud."""
        return self.positions.shape[0]

    @property
    def num_spatial_points(self) -> int:
        """Total number of discrete spatial points (alias for num_particles)."""
        return self.num_particles

    @property
    def geometry_type(self) -> GeometryType:
        """Type of geometry (CUSTOM for point clouds)."""
        return GeometryType.CUSTOM

    def get_spatial_grid(self) -> NDArray[np.floating]:
        """
        Get spatial grid representation (returns particle positions).

        Returns:
            Particle positions (N, dimension)
        """
        return self.positions

    def get_problem_config(self) -> dict:
        """
        Return configuration dict for MFGProblem initialization.

        Returns:
            Dictionary with keys:
                - num_spatial_points: Number of particles
                - spatial_shape: (num_particles,)
                - spatial_bounds: Bounding box of particles
                - spatial_discretization: None (not applicable)
                - legacy_1d_attrs: None
        """
        min_coords = np.min(self.positions, axis=0)
        max_coords = np.max(self.positions, axis=0)
        bounds = tuple((float(mn), float(mx)) for mn, mx in zip(min_coords, max_coords, strict=True))

        return {
            "num_spatial_points": self.num_particles,
            "spatial_shape": (self.num_particles,),
            "spatial_bounds": bounds,
            "spatial_discretization": None,
            "legacy_1d_attrs": None,
        }

    @property
    def bounds(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Bounding box of the point cloud: (min_coords, max_coords)."""
        return np.min(self.positions, axis=0), np.max(self.positions, axis=0)

    def get_bounds(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Get bounding box of the point cloud."""
        return self.bounds

    # =========================================================================
    # Boundary Methods (required by GeometryProtocol)
    # =========================================================================

    def is_on_boundary(
        self,
        points: NDArray[np.floating],
        tolerance: float = ONWALL_TOL,
    ) -> NDArray[np.bool_]:
        """Check if points are on the bounding box boundary."""
        points = np.atleast_2d(points)
        min_coords, max_coords = self.bounds
        on_boundary = np.zeros(len(points), dtype=bool)

        for i, p in enumerate(points):
            for d in range(self.dimension):
                if abs(p[d] - min_coords[d]) < tolerance or abs(p[d] - max_coords[d]) < tolerance:
                    on_boundary[i] = True
                    break

        return on_boundary

    def get_boundary_normal(
        self,
        points: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Get outward normal at boundary points (axis-aligned for bounding box)."""
        points = np.atleast_2d(points)
        normals = np.zeros_like(points)
        min_coords, max_coords = self.bounds
        tolerance = ONWALL_TOL

        for i, p in enumerate(points):
            for d in range(self.dimension):
                if abs(p[d] - min_coords[d]) < tolerance:
                    normals[i, d] = -1.0
                    break
                elif abs(p[d] - max_coords[d]) < tolerance:
                    normals[i, d] = 1.0
                    break

        return normals

    def project_to_boundary(
        self,
        points: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Project points to nearest bounding box boundary."""
        points = np.atleast_2d(points)
        min_coords, max_coords = self.bounds
        projected = points.copy()

        for i, p in enumerate(points):
            min_dist = float("inf")
            best_proj = p.copy()

            for d in range(self.dimension):
                dist_min = abs(p[d] - min_coords[d])
                if dist_min < min_dist:
                    min_dist = dist_min
                    best_proj = p.copy()
                    best_proj[d] = min_coords[d]

                dist_max = abs(p[d] - max_coords[d])
                if dist_max < min_dist:
                    min_dist = dist_max
                    best_proj = p.copy()
                    best_proj[d] = max_coords[d]

            projected[i] = best_proj

        return projected

    def project_to_interior(
        self,
        points: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Project points to interior (clip to bounding box)."""
        points = np.atleast_2d(points)
        min_coords, max_coords = self.bounds
        return np.clip(points, min_coords, max_coords)

    def get_boundary_regions(self) -> dict[str, dict]:
        """Get boundary regions (axis-aligned for bounding box)."""
        min_coords, max_coords = self.bounds
        regions = {}
        axis_names = ["x", "y", "z", "w", "v"]

        for d in range(self.dimension):
            axis = axis_names[d] if d < len(axis_names) else f"dim{d}"
            regions[f"{axis}_min"] = {"axis": d, "side": "min", "value": float(min_coords[d])}
            regions[f"{axis}_max"] = {"axis": d, "side": "max", "value": float(max_coords[d])}

        return regions

    def get_grid_shape(self) -> tuple[int]:
        """
        Get discretization shape for point cloud.

        Returns:
            (N,) where N is the number of points in the cloud.

        Notes:
            Point clouds are unstructured, so shape is just (num_points,).
        """
        return (self.num_particles,)

    def get_boundary_conditions(self):
        """
        Get boundary conditions for point cloud.

        Returns:
            None - point clouds don't have inherent boundary conditions.
            BCs should be specified by the problem or solver.
        """
        return None

    # Boundary Helper Methods (Added for Issue #545 - Unified BC Workflow)
    def get_boundary_indices(
        self,
        points: NDArray[np.floating],
        tolerance: float = ONWALL_TOL,
    ) -> NDArray[np.intp]:
        """Get indices of boundary points (default implementation)."""
        on_boundary = self.is_on_boundary(points, tolerance)
        return np.where(on_boundary)[0]

    def get_boundary_info(
        self,
        points: NDArray[np.floating],
        tolerance: float = ONWALL_TOL,
    ) -> tuple[NDArray[np.intp], NDArray[np.floating]]:
        """Get boundary indices and normals (default implementation)."""
        boundary_indices = self.get_boundary_indices(points, tolerance)
        if len(boundary_indices) == 0:
            return boundary_indices, np.array([], dtype=np.float64).reshape(0, self.dimension)
        boundary_points = points[boundary_indices]
        normals = self.get_boundary_normal(boundary_points)
        return boundary_indices, normals

    def get_collocation_points(self) -> NDArray[np.floating]:
        """
        Get collocation points (the point cloud itself).

        Returns:
            Array of shape (N, d) containing particle positions.
        """
        return self.positions

    def is_same_pointset(self, other: PointCloudGeometry, tol: float = 1e-10) -> bool:
        """
        Check if two point clouds represent the same particle set.

        This enables identity projection optimization when HJB and FP use the
        same particles (co-moving particle pattern).

        Args:
            other: Another PointCloudGeometry
            tol: Tolerance for position comparison

        Returns:
            True if point clouds have same positions (up to tolerance)

        Examples:
            >>> # Co-moving particles (identity projection)
            >>> particles = np.random.uniform(0, 1, (1000, 2))
            >>> geom1 = PointCloudGeometry(particles)
            >>> geom2 = PointCloudGeometry(particles)  # Same array
            >>> geom1.is_same_pointset(geom2)
            True
            >>>
            >>> # Different particles (need projection)
            >>> collocation = np.random.uniform(0, 1, (500, 2))
            >>> geom3 = PointCloudGeometry(collocation)
            >>> geom1.is_same_pointset(geom3)
            False
        """
        if not isinstance(other, PointCloudGeometry):
            return False

        if self.num_particles != other.num_particles:
            return False

        if self.dimension != other.dimension:
            return False

        # Check if arrays are identical (same object)
        if self.positions is other.positions:
            return True

        # Check if positions are numerically equal
        return np.allclose(self.positions, other.positions, atol=tol, rtol=0.0)

    def __repr__(self) -> str:
        """String representation."""
        return f"PointCloudGeometry(num_particles={self.num_particles}, dimension={self.dimension})"


# Register with GeometryProtocol (runtime checkable)
assert isinstance(PointCloudGeometry(np.random.rand(10, 2)), GeometryProtocol), (
    "PointCloudGeometry must implement GeometryProtocol"
)


__all__ = ["PointCloudGeometry"]

"""
Geometry-aware collocation point generation for meshfree methods.

This module provides unified sampling for interior and boundary points,
with support for all geometry types (grids, meshes, implicit, CSG).

Architecture:
    Layer 2 in the collocation sampling hierarchy:
    - Uses Layer 1 (utils/numerical/particle/sampling.py) for base algorithms
    - Provides geometry-specific strategies for different domain types
    - Returns CollocationPointSet with points, boundary info, and normals

Key Classes:
    - CollocationSampler: Main interface, dispatches to geometry-specific samplers
    - CollocationPointSet: Container for points, boundary mask, normals, region IDs
    - BaseCollocationStrategy: ABC for geometry-specific implementations

Example:
    >>> from mfgarchon.geometry.collocation import CollocationSampler
    >>> from mfgarchon.geometry import Hyperrectangle
    >>>
    >>> domain = Hyperrectangle(bounds=[[0, 1], [0, 1]])
    >>> sampler = CollocationSampler(domain)
    >>> coll = sampler.generate_collocation(n_interior=400, n_boundary=100)
    >>>
    >>> # Use with GFDM solver
    >>> solver = HJBGFDMSolver(
    ...     problem,
    ...     collocation_points=coll.points,
    ...     boundary_indices=coll.boundary_indices,
    ... )

See Also:
    - Issue #482: Boundary Point Generation Module for Meshfree Methods
    - docs/development/COLLOCATION_POINT_GENERATION_REPORT.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

# Issue #543: Runtime import for isinstance() checks
from mfgarchon.geometry.protocol import GeometryProtocol
from mfgarchon.geometry.protocols import SupportsPeriodic
from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

from mfgarchon.utils.numerical.particle.sampling import (
    MCConfig,
    PoissonDiskSampler,
    QuasiMCSampler,
    UniformMCSampler,
)

logger = get_logger(__name__)


@dataclass
class CollocationPointSet:
    """
    Container for collocation point data.

    This dataclass holds all information needed for meshfree methods:
    - Combined interior and boundary points
    - Boolean mask identifying boundary points
    - Outward unit normals at boundary points
    - Region IDs for multi-component boundaries (e.g., CSG domains)

    Attributes:
        points: All collocation points, shape (N, d)
        is_boundary: Boolean mask, True for boundary points, shape (N,)
        normals: Outward unit normals, zeros for interior, shape (N, d)
        region_ids: Boundary region identifier, -1 for interior, shape (N,)
        fill_distance: Fill distance h (max empty circle radius), computed during optimization
        separation_distance: Separation distance q = min_dist/2 (Wendland convention)
        mesh_ratio: Mesh ratio h/q, optimal ~1.155 (hexagonal) or ~1.414 (square)

    Properties:
        boundary_indices: Array of indices where is_boundary is True
        interior_indices: Array of indices where is_boundary is False
        n_interior: Number of interior points
        n_boundary: Number of boundary points
    """

    points: NDArray  # (N, d)
    is_boundary: NDArray  # (N,) bool
    normals: NDArray  # (N, d)
    region_ids: NDArray = field(default_factory=lambda: np.array([]))  # (N,) int
    fill_distance: float | None = None  # h, computed during optimization
    separation_distance: float | None = None  # q = min_dist/2
    mesh_ratio: float | None = None  # h/q

    def __post_init__(self):
        """Validate and set defaults."""
        if len(self.region_ids) == 0:
            # Default: all interior = -1, all boundary = 0
            self.region_ids = np.where(self.is_boundary, 0, -1)

    @property
    def boundary_indices(self) -> NDArray:
        """Indices of boundary points (for backward compatibility with GFDM)."""
        return np.where(self.is_boundary)[0]

    @property
    def interior_indices(self) -> NDArray:
        """Indices of interior points."""
        return np.where(~self.is_boundary)[0]

    @property
    def n_interior(self) -> int:
        """Number of interior points."""
        return int(np.sum(~self.is_boundary))

    @property
    def n_boundary(self) -> int:
        """Number of boundary points."""
        return int(np.sum(self.is_boundary))

    @property
    def dimension(self) -> int:
        """Spatial dimension."""
        return self.points.shape[1]

    def get_boundary_points(self) -> NDArray:
        """Return only boundary points."""
        return self.points[self.is_boundary]

    def get_interior_points(self) -> NDArray:
        """Return only interior points."""
        return self.points[~self.is_boundary]

    def get_boundary_normals(self) -> NDArray:
        """Return normals at boundary points only."""
        return self.normals[self.is_boundary]

    def get_region_boundary_indices(self, region_id: int) -> NDArray:
        """Get boundary indices for a specific region."""
        return np.where((self.is_boundary) & (self.region_ids == region_id))[0]


class BaseCollocationStrategy(ABC):
    """Abstract base class for geometry-specific collocation strategies."""

    def __init__(self, geometry: GeometryProtocol):
        self.geometry = geometry
        self.dimension = self._get_dimension()

    @abstractmethod
    def _get_dimension(self) -> int:
        """Get spatial dimension from geometry."""

    @abstractmethod
    def sample_interior(
        self,
        n_points: int,
        method: Literal["lloyd", "poisson_disk", "sobol", "uniform"] = "lloyd",
        seed: int | None = None,
    ) -> NDArray:
        """
        Sample interior collocation points.

        Args:
            n_points: Number of interior points to sample
            method: Sampling method:
                - "lloyd" (recommended): CVT/Lloyd relaxation for quasi-uniform
                - "poisson_disk": Blue noise (poor for non-convex domains)
                - "sobol": Quasi-random sequence
                - "uniform": Pure random
            seed: Random seed for reproducibility

        Returns:
            Interior points, shape (n_points, d)
        """

    @abstractmethod
    def sample_boundary(
        self,
        n_points: int,
        method: Literal["uniform", "poisson_disk"] = "uniform",
        seed: int | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Sample boundary collocation points.

        Args:
            n_points: Number of boundary points to sample
            method: Sampling method
            seed: Random seed for reproducibility

        Returns:
            Tuple of (points, normals, region_ids):
            - points: Boundary points, shape (n_points, d)
            - normals: Outward unit normals, shape (n_points, d)
            - region_ids: Region identifiers, shape (n_points,)
        """


class CartesianGridCollocation(BaseCollocationStrategy):
    """
    Collocation strategy for Cartesian grids (TensorProductGrid).

    For Cartesian grids, points are extracted from the grid structure:
    - Interior: Grid nodes not on any boundary face
    - Boundary: Grid nodes on faces, with explicit corner handling
    """

    def _get_dimension(self) -> int:
        """Get dimension from grid."""
        # Issue #543: Use GeometryProtocol instead of hasattr()
        if not isinstance(self.geometry, GeometryProtocol):
            raise TypeError(
                f"Geometry must implement GeometryProtocol, got {type(self.geometry).__name__}. "
                f"All geometries must provide dimension property."
            )
        return self.geometry.dimension

    def _get_bounds(self) -> list[tuple[float, float]]:
        """Get domain bounds."""
        # Issue #543: Use GeometryProtocol.get_bounds() instead of hasattr()
        if not isinstance(self.geometry, GeometryProtocol):
            raise TypeError(f"Geometry must implement GeometryProtocol, got {type(self.geometry).__name__}")

        bounds_tuple = self.geometry.get_bounds()
        if bounds_tuple is None:
            raise ValueError("Geometry does not have bounded domain (get_bounds() returned None)")

        # Convert from (min_coords, max_coords) to list of (min, max) tuples
        min_coords, max_coords = bounds_tuple
        return [(float(min_coords[i]), float(max_coords[i])) for i in range(len(min_coords))]

    def sample_interior(
        self,
        n_points: int,
        method: Literal["lloyd", "poisson_disk", "sobol", "uniform"] = "lloyd",
        seed: int | None = None,
        refine_steps: int = 20,
    ) -> NDArray:
        """Sample interior points using specified method.

        Args:
            n_points: Number of points to generate
            method: Sampling method:
                - "lloyd": CVT/Lloyd relaxation (best quasi-uniform)
                - "poisson_disk": Blue noise with minimum spacing
                - "sobol": Quasi-random low-discrepancy
                - "uniform": Pure random
            seed: Random seed
            refine_steps: Lloyd iterations (only for method="lloyd")
        """
        bounds = self._get_bounds()
        config = MCConfig(seed=seed)

        if method == "lloyd":
            # Initialize with Sobol, then apply Lloyd relaxation
            sampler = QuasiMCSampler(bounds, config, "sobol")
            points = sampler.sample(n_points)
            if refine_steps > 0 and len(points) > 1:
                points = self._refine_lloyd_cartesian(points, bounds, refine_steps)
            return points
        elif method == "poisson_disk":
            sampler = PoissonDiskSampler(bounds, config)
        elif method == "sobol":
            sampler = QuasiMCSampler(bounds, config, "sobol")
        else:
            sampler = UniformMCSampler(bounds, config)

        return sampler.sample(n_points)

    def _refine_lloyd_cartesian(
        self,
        points: NDArray,
        bounds: list[tuple[float, float]],
        steps: int = 20,
    ) -> NDArray:
        """Lloyd/CVT relaxation for Cartesian domains (no SDF needed)."""
        from scipy.spatial import cKDTree

        N, d = points.shape
        current = points.copy()

        bbox_volume = np.prod([b[1] - b[0] for b in bounds])
        target_spacing = (bbox_volume / N) ** (1.0 / d)
        interaction_radius = 2.5 * target_spacing

        for step in range(steps):
            forces = np.zeros_like(current)
            tree = cKDTree(current)
            pairs = tree.query_pairs(r=interaction_radius)

            for i, j in pairs:
                diff = current[i] - current[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = np.random.randn(d) * 0.01 * target_spacing
                    dist = np.linalg.norm(diff)
                force_mag = 1.0 / (dist**2 + 0.01 * target_spacing**2)
                force_vec = (diff / dist) * force_mag
                forces[i] += force_vec
                forces[j] -= force_vec

            # Boundary repulsion for Cartesian box
            for dim in range(d):
                lo, hi = bounds[dim]
                margin = target_spacing * 1.5
                # Near lower bound
                near_lo = current[:, dim] < lo + margin
                for i in np.where(near_lo)[0]:
                    dist_to_wall = max(current[i, dim] - lo, 1e-10)
                    forces[i, dim] += 0.5 / (dist_to_wall**2)
                # Near upper bound
                near_hi = current[:, dim] > hi - margin
                for i in np.where(near_hi)[0]:
                    dist_to_wall = max(hi - current[i, dim], 1e-10)
                    forces[i, dim] -= 0.5 / (dist_to_wall**2)

            # Adaptive time step with annealing
            max_force = np.max(np.linalg.norm(forces, axis=1))
            anneal = 1.0 - 0.5 * (step / steps)
            dt = min(0.15 * anneal, 0.4 * target_spacing / (max_force + 1e-10))

            current = current + forces * dt

            # Clamp to bounds
            for dim in range(d):
                current[:, dim] = np.clip(current[:, dim], bounds[dim][0] + 1e-6, bounds[dim][1] - 1e-6)

        return current

    def sample_boundary(
        self,
        n_points: int,
        method: Literal["uniform", "poisson_disk"] = "uniform",
        seed: int | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Sample boundary points on faces of Cartesian domain.

        Distributes points across all 2*d faces, with explicit corner inclusion.
        """
        bounds = self._get_bounds()
        d = self.dimension
        rng = np.random.RandomState(seed)

        # Allocate points per face (2*d faces total)
        n_faces = 2 * d
        remainder = n_points % n_faces

        all_points = []
        all_normals = []
        all_region_ids = []

        # Add corners first (always include)
        corners = self._generate_corners(bounds)
        n_corners = len(corners)
        corner_normals = self._compute_corner_normals(corners, bounds)

        all_points.append(corners)
        all_normals.append(corner_normals)
        all_region_ids.append(np.full(n_corners, -2))  # -2 for corners

        # Remaining points to distribute
        n_remaining = n_points - n_corners

        if n_remaining > 0:
            points_per_face = n_remaining // n_faces

            region_id = 0
            for dim in range(d):
                for side in [0, 1]:  # min, max
                    n_face = points_per_face + (1 if region_id < remainder else 0)
                    if n_face <= 0:
                        region_id += 1
                        continue

                    face_pts = self._sample_face(dim, side, n_face, bounds, rng)
                    normal = np.zeros(d)
                    normal[dim] = 1.0 if side == 1 else -1.0

                    all_points.append(face_pts)
                    all_normals.append(np.tile(normal, (len(face_pts), 1)))
                    all_region_ids.append(np.full(len(face_pts), region_id))

                    region_id += 1

        points = np.vstack(all_points)
        normals = np.vstack(all_normals)
        region_ids = np.concatenate(all_region_ids)

        return points, normals, region_ids

    def _generate_corners(self, bounds: list[tuple[float, float]]) -> NDArray:
        """Generate all corner vertices of the domain."""
        from itertools import product

        d = len(bounds)
        corners = []
        for indices in product([0, 1], repeat=d):
            corner = [bounds[dim][idx] for dim, idx in enumerate(indices)]
            corners.append(corner)
        return np.array(corners)

    def _compute_corner_normals(self, corners: NDArray, bounds: list[tuple[float, float]]) -> NDArray:
        """Compute averaged normals at corners."""
        d = len(bounds)
        normals = np.zeros_like(corners)

        for i, corner in enumerate(corners):
            normal = np.zeros(d)
            for dim in range(d):
                if np.isclose(corner[dim], bounds[dim][0]):
                    normal[dim] -= 1.0
                elif np.isclose(corner[dim], bounds[dim][1]):
                    normal[dim] += 1.0
            # Normalize
            norm = np.linalg.norm(normal)
            if norm > 0:
                normals[i] = normal / norm

        return normals

    def _sample_face(
        self,
        dim: int,
        side: int,
        n_points: int,
        bounds: list[tuple[float, float]],
        rng: np.random.RandomState,
    ) -> NDArray:
        """Sample points evenly spaced on a face (excluding corners).

        Uses linspace for even spacing, which produces much better mesh quality
        for GFDM than random uniform sampling.
        """
        d = len(bounds)
        points = np.zeros((n_points, d))

        for i in range(d):
            if i == dim:
                # Fixed coordinate on this face
                points[:, i] = bounds[dim][side]
            else:
                # Evenly spaced (excluding corners which are added separately)
                # Use linspace with n_points+2 to exclude endpoints
                lo, hi = bounds[i]
                points[:, i] = np.linspace(lo, hi, n_points + 2)[1:-1]

        return points


class ImplicitDomainCollocation(BaseCollocationStrategy):
    """
    Collocation strategy for implicit domains defined by signed distance functions.

    Uses Newton projection to find boundary points on the zero level set.

    Sampling Methods:
        - "lloyd" (recommended): CVT/Lloyd relaxation via particle repulsion.
          Best for non-convex domains. Achieves quasi-uniform coverage.
        - "sobol": Quasi-random low-discrepancy sequence + rejection sampling.
          Good for convex domains, may have gaps in non-convex regions.
        - "poisson_disk": Blue noise with minimum spacing guarantee.
          WARNING: Poor coverage in non-convex geometries (thin corridors,
          regions separated by obstacles). Use "lloyd" instead.
        - "uniform": Pure random sampling. Not recommended.

    For non-convex domains (CSG with obstacles, L-shaped regions, etc.),
    always use method="lloyd" which applies particle repulsion to fill
    all regions uniformly regardless of geometry complexity.
    """

    def _get_dimension(self) -> int:
        """Get dimension from implicit domain."""
        # Issue #543: Use GeometryProtocol instead of hasattr()
        if not isinstance(self.geometry, GeometryProtocol):
            raise TypeError(f"Geometry must implement GeometryProtocol, got {type(self.geometry).__name__}")
        return self.geometry.dimension

    def _get_bounds(self) -> list[tuple[float, float]]:
        """Get bounding box of implicit domain."""
        # Issue #543: Use GeometryProtocol.get_bounds() instead of hasattr()
        if not isinstance(self.geometry, GeometryProtocol):
            raise TypeError(f"Geometry must implement GeometryProtocol, got {type(self.geometry).__name__}")

        bounds_tuple = self.geometry.get_bounds()
        if bounds_tuple is None:
            raise ValueError("Implicit domain must have bounded domain for collocation")

        # Convert from (min_coords, max_coords) to list of (min, max) tuples
        min_coords, max_coords = bounds_tuple
        return [(float(min_coords[i]), float(max_coords[i])) for i in range(len(min_coords))]

    def sample_interior(
        self,
        n_points: int,
        method: Literal["lloyd", "poisson_disk", "sobol", "uniform"] = "lloyd",
        seed: int | None = None,
        refine_steps: int = 20,
    ) -> NDArray:
        """
        Sample interior points for implicit domain.

        Args:
            n_points: Number of points to generate
            method: Sampling method:
                - "lloyd" (default, recommended): CVT/Lloyd relaxation via particle
                  repulsion. Best quasi-uniform coverage for any geometry.
                - "poisson_disk": Blue noise. WARNING: poor for non-convex domains.
                - "sobol": Quasi-random sequence + rejection.
                - "uniform": Pure random (not recommended).
            seed: Random seed for reproducibility
            refine_steps: Number of Lloyd/particle repulsion iterations.
                For method="lloyd", this is the main algorithm (default 20).
                For other methods, this refines the initial distribution.
                Set to 0 to disable refinement (not recommended for non-convex).

        Returns:
            Interior points, shape (n_points, d)

        Note:
            For non-convex domains (obstacles, thin corridors, L-shapes),
            use method="lloyd" which fills all regions uniformly via
            particle repulsion regardless of geometry complexity.
        """
        bounds = self._get_bounds()
        rng = np.random.RandomState(seed)

        if method == "lloyd":
            # Lloyd/CVT: Initialize with rejection sampling, then heavy refinement
            points = self._sample_initial_rejection(n_points, bounds, rng)
            # Lloyd uses more iterations for better convergence
            lloyd_steps = max(refine_steps, 30)
            if len(points) > 1:
                points = self._refine_lloyd_cvt(points, bounds, lloyd_steps)
        else:
            # Other methods: initial sampling + optional refinement
            if method == "poisson_disk":
                logger.warning(
                    "Poisson disk may have poor coverage in non-convex domains. "
                    "Consider method='lloyd' for better uniformity."
                )
            points = self._sample_initial_rejection(n_points, bounds, rng)

            # Refine via particle repulsion
            if refine_steps > 0 and len(points) > 1:
                points = self._refine_particle_repulsion(points, bounds, refine_steps)

        logger.debug(f"ImplicitCollocation: {len(points)} points, method={method}")
        return points

    def _sample_initial_rejection(
        self,
        n_points: int,
        bounds: list[tuple[float, float]],
        rng: np.random.RandomState,
    ) -> NDArray:
        """Generate initial points via simple rejection sampling."""
        accepted = []
        max_attempts = n_points * 20

        for _ in range(max_attempts):
            if len(accepted) >= n_points:
                break
            candidate = np.array([rng.uniform(b[0], b[1]) for b in bounds])
            if self.geometry.signed_distance(candidate.reshape(1, -1))[0] < 0:
                accepted.append(candidate)

        return np.array(accepted[:n_points])

    def _refine_particle_repulsion(
        self,
        points: NDArray,
        bounds: list[tuple[float, float]],
        steps: int = 20,
        dt: float = 0.1,
    ) -> NDArray:
        """
        Refine point distribution using particle repulsion (energy minimization).

        Minimizes the energy functional:
            E = sum_{i!=j} 1/||x_i - x_j||^p + sum_i 1/d(x_i, boundary)^q

        Points naturally "flow" into non-convex regions, solving the thin-wall
        problem that plagues Poisson disk sampling.

        Reference: Lloyd's relaxation / Centroidal Voronoi Tessellation
        """
        from scipy.spatial import cKDTree

        N, d = points.shape
        current = points.copy()

        # Estimate interaction radius from point density
        bbox_volume = np.prod([b[1] - b[0] for b in bounds])
        target_spacing = (bbox_volume / N) ** (1.0 / d)
        interaction_radius = 3.0 * target_spacing

        for _step in range(steps):
            forces = np.zeros_like(current)

            # 1. Inter-particle repulsion (via KDTree for efficiency)
            tree = cKDTree(current)
            pairs = tree.query_pairs(r=interaction_radius)

            for i, j in pairs:
                diff = current[i] - current[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue
                # Repulsion: 1/r^2 kernel (adjustable)
                force_mag = 1.0 / (dist**2 + 1e-6)
                force_vec = (diff / dist) * force_mag
                forces[i] += force_vec
                forces[j] -= force_vec

            # 2. Boundary repulsion (push away from walls)
            sdf = self.geometry.signed_distance(current)
            grad_sdf = self._compute_sdf_gradient(current)

            # Repel from boundary: force proportional to 1/d^2
            # Only apply when close to boundary (within target_spacing)
            close_to_boundary = np.abs(sdf) < target_spacing
            for i in np.where(close_to_boundary)[0]:
                boundary_dist = np.abs(sdf[i])
                if boundary_dist < 1e-10:
                    boundary_dist = 1e-10
                # Push inward (opposite to gradient which points outward)
                force_mag = 0.5 / (boundary_dist**2)
                forces[i] -= grad_sdf[i] * force_mag * np.sign(sdf[i])

            # 3. Adaptive time step (reduce if forces are large)
            max_force = np.max(np.linalg.norm(forces, axis=1))
            effective_dt = min(dt, 0.5 * target_spacing / (max_force + 1e-10))

            # 4. Update positions
            current = current + forces * effective_dt

            # 5. Project back to domain interior if pushed outside
            sdf_new = self.geometry.signed_distance(current)
            outside = sdf_new >= 0

            if np.any(outside):
                # Newton projection back to boundary, then slight inward push
                grad = self._compute_sdf_gradient(current[outside])
                grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
                grad_norm = np.maximum(grad_norm, 1e-10)
                # Project to boundary then push slightly inside
                current[outside] -= (sdf_new[outside, np.newaxis] + 0.01 * target_spacing) * grad / grad_norm

            # 6. Clamp to bounds
            for i in range(d):
                current[:, i] = np.clip(current[:, i], bounds[i][0] + 1e-6, bounds[i][1] - 1e-6)

        return current

    def _refine_lloyd_cvt(
        self,
        points: NDArray,
        bounds: list[tuple[float, float]],
        steps: int = 30,
        boundary_strength: float = 0.8,
    ) -> NDArray:
        """
        Lloyd/CVT relaxation for quasi-uniform point distribution.

        This is the recommended method for non-convex domains. Uses particle
        repulsion with adaptive parameters optimized for CVT convergence.

        Algorithm:
            1. Inter-particle repulsion (inverse-square kernel)
            2. Boundary repulsion (keeps points inside domain)
            3. Adaptive time stepping for stability
            4. Project outside points back to domain

        The result approximates a Centroidal Voronoi Tessellation (CVT) where
        each point is at the centroid of its Voronoi cell, achieving optimal
        quasi-uniform coverage.

        Args:
            points: Initial point positions, shape (N, d)
            bounds: Bounding box [(min, max), ...] per dimension
            steps: Number of Lloyd iterations (30 recommended for convergence)
            boundary_strength: Strength of boundary repulsion (higher = more
                separation from boundary, helps mesh ratio). Default 0.8.

        Returns:
            Refined points with quasi-uniform spacing, shape (N, d)

        Reference:
            Lloyd, S. (1982). "Least squares quantization in PCM."
            Du, Faber, Gunzburger (1999). "Centroidal Voronoi Tessellations."
        """
        from scipy.spatial import cKDTree

        N, d = points.shape
        current = points.copy()

        # Compute target spacing from point density
        bbox_volume = np.prod([b[1] - b[0] for b in bounds])
        target_spacing = (bbox_volume / N) ** (1.0 / d)

        # Interaction radius: affects how far particles "see" each other
        # Larger = smoother but slower; smaller = faster but may have gaps
        interaction_radius = 2.5 * target_spacing

        # Adaptive parameters for stable convergence
        dt_base = 0.15  # Base time step

        for step in range(steps):
            forces = np.zeros_like(current)

            # 1. Inter-particle repulsion via KDTree
            tree = cKDTree(current)
            pairs = tree.query_pairs(r=interaction_radius)

            for i, j in pairs:
                diff = current[i] - current[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Identical points: apply random perturbation
                    diff = np.random.randn(d) * 0.01 * target_spacing
                    dist = np.linalg.norm(diff)

                # Inverse-square repulsion (softer than 1/r^3)
                force_mag = 1.0 / (dist**2 + 0.01 * target_spacing**2)
                force_vec = (diff / dist) * force_mag
                forces[i] += force_vec
                forces[j] -= force_vec

            # 2. Boundary repulsion using SDF
            sdf = self.geometry.signed_distance(current)
            grad_sdf = self._compute_sdf_gradient(current)

            # Apply boundary repulsion when close to boundary
            boundary_zone = target_spacing * 1.5
            close_mask = np.abs(sdf) < boundary_zone

            for i in np.where(close_mask)[0]:
                boundary_dist = max(np.abs(sdf[i]), 1e-10)
                # Repulsion strength increases near boundary
                force_mag = boundary_strength / (boundary_dist**2)
                # Push inward (opposite to SDF gradient)
                forces[i] -= grad_sdf[i] * force_mag * np.sign(sdf[i])

            # 3. Adaptive time step (decreases as we converge)
            max_force = np.max(np.linalg.norm(forces, axis=1))
            # Anneal dt: larger early, smaller late for fine-tuning
            anneal_factor = 1.0 - 0.5 * (step / steps)
            dt = min(dt_base * anneal_factor, 0.4 * target_spacing / (max_force + 1e-10))

            # 4. Update positions
            current = current + forces * dt

            # 5. Project outside points back to domain interior
            sdf_new = self.geometry.signed_distance(current)
            outside = sdf_new >= 0

            if np.any(outside):
                grad = self._compute_sdf_gradient(current[outside])
                grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
                grad_norm = np.maximum(grad_norm, 1e-10)
                # Project to boundary, then push slightly inside
                push_dist = 0.02 * target_spacing
                current[outside] -= (sdf_new[outside, np.newaxis] + push_dist) * grad / grad_norm

            # 6. Clamp to bounding box
            for dim in range(d):
                margin = 1e-6
                current[:, dim] = np.clip(current[:, dim], bounds[dim][0] + margin, bounds[dim][1] - margin)

        return current

    def sample_boundary(
        self,
        n_points: int,
        method: Literal["uniform", "poisson_disk"] = "uniform",
        seed: int | None = None,
        refine_steps: int = 30,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Sample boundary points with geodesic repulsion refinement.

        Algorithm:
        1. Generate interior seed points
        2. Project each to boundary via Newton iteration on SDF
        3. Refine: Apply tangential repulsion for uniform spacing along boundary
        4. Compute normals from SDF gradient

        The refinement step moves points along the boundary (tangent direction)
        to achieve uniform spacing, solving clustering from projection.

        Args:
            n_points: Number of boundary points
            method: Initial seed generation method
            seed: Random seed
            refine_steps: Boundary repulsion iterations (0 to disable)
        """
        # Generate seed points inside domain
        seeds = self.sample_interior(n_points, method="sobol", seed=seed, refine_steps=0)

        # Project to boundary
        boundary_points = self._project_to_boundary(seeds)

        # Refine via tangential repulsion (geodesic relaxation)
        if refine_steps > 0 and len(boundary_points) > 1:
            boundary_points = self._refine_boundary_repulsion(boundary_points, refine_steps)

        # Compute normals from SDF gradient
        normals = self._compute_normals(boundary_points)

        # All same region for single implicit domain
        region_ids = np.zeros(len(boundary_points), dtype=int)

        return boundary_points, normals, region_ids

    def _refine_boundary_repulsion(
        self,
        points: NDArray,
        steps: int = 30,
        dt: float = 0.05,
    ) -> NDArray:
        """
        Refine boundary point distribution using tangential repulsion.

        Points are constrained to the boundary manifold (SDF = 0).
        Forces are projected to the tangent space before update.
        After each step, points are re-projected to ensure SDF = 0.

        This achieves uniform spacing along the boundary curve/surface.
        """
        from scipy.spatial import cKDTree

        N, _ = points.shape
        current = points.copy()

        # Estimate target spacing from arc length / perimeter approximation
        # Use mean distance between consecutive sorted points as estimate
        target_spacing = self._estimate_boundary_spacing(current, N)
        interaction_radius = 4.0 * target_spacing

        for _step in range(steps):
            forces = np.zeros_like(current)

            # 1. Inter-particle repulsion (KDTree)
            tree = cKDTree(current)
            pairs = tree.query_pairs(r=interaction_radius)

            for i, j in pairs:
                diff = current[i] - current[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue
                # Repulsion: 1/r^2 kernel
                force_mag = 1.0 / (dist**2 + 1e-6)
                force_vec = (diff / dist) * force_mag
                forces[i] += force_vec
                forces[j] -= force_vec

            # 2. Project forces to tangent space (remove normal component)
            normals = self._compute_normals(current)
            for i in range(N):
                normal_component = np.dot(forces[i], normals[i]) * normals[i]
                forces[i] = forces[i] - normal_component  # Tangential only

            # 3. Adaptive time step
            max_force = np.max(np.linalg.norm(forces, axis=1))
            effective_dt = min(dt, 0.3 * target_spacing / (max_force + 1e-10))

            # 4. Update positions (tangential movement)
            current = current + forces * effective_dt

            # 5. Re-project to boundary (maintain SDF = 0)
            current = self._project_to_boundary(current, max_iter=5, tol=1e-7)

        return current

    def _estimate_boundary_spacing(self, points: NDArray, n_points: int) -> float:
        """Estimate target spacing for boundary points."""
        bounds = self._get_bounds()
        d = len(bounds)

        if d == 2:
            # Rough perimeter estimate: sum of bounding box edges
            perimeter = 2 * sum(b[1] - b[0] for b in bounds)
            return perimeter / n_points
        else:
            # Higher dimensions: use bounding box surface area estimate
            bbox_dims = np.array([b[1] - b[0] for b in bounds])
            surface_area = 2 * sum(np.prod(np.delete(bbox_dims, i)) for i in range(d))
            return (surface_area / n_points) ** (1.0 / (d - 1))

    def _project_to_boundary(
        self,
        points: NDArray,
        max_iter: int = 20,
        tol: float = 1e-8,
    ) -> NDArray:
        """Project points to boundary using Newton iteration."""
        projected = points.copy()

        for _ in range(max_iter):
            # Evaluate SDF and gradient
            sdf = self.geometry.signed_distance(projected)
            grad = self._compute_sdf_gradient(projected)

            # Newton step: x_new = x - φ(x) * ∇φ(x) / |∇φ(x)|²
            grad_norm_sq = np.sum(grad**2, axis=1, keepdims=True)
            grad_norm_sq = np.maximum(grad_norm_sq, 1e-12)  # Avoid division by zero

            step = (sdf[:, np.newaxis] * grad) / grad_norm_sq
            projected = projected - step

            # Check convergence
            if np.max(np.abs(sdf)) < tol:
                break

        return projected

    def _compute_sdf_gradient(self, points: NDArray, eps: float = 1e-6) -> NDArray:
        """
        Compute SDF gradient, preferring analytical if available.

        Note:
            Issue #662: Numerical fallback delegated to canonical implementation
            in operators/differential/function_gradient.py
        """
        # Issue #543: Use try/except for optional method instead of hasattr()
        # Try analytical gradient first
        try:
            return self.geometry.signed_distance_gradient(points)
        except AttributeError:
            pass

        # Finite difference fallback (Issue #662: use canonical implementation)
        from mfgarchon.operators.differential.function_gradient import function_gradient

        return function_gradient(self.geometry.signed_distance, points, eps=eps)

    def _compute_normals(self, points: NDArray) -> NDArray:
        """
        Compute outward unit normals at boundary points.

        Note:
            Issue #662: Uses canonical implementation from operators/.
        """
        # Issue #662: Use canonical implementation
        from mfgarchon.operators.differential.function_gradient import (
            outward_normal_from_sdf,
        )

        # Try analytical gradient first, then numerical
        try:
            grad = self.geometry.signed_distance_gradient(points)
            norms = np.linalg.norm(grad, axis=1, keepdims=True)
            return grad / np.maximum(norms, 1e-12)
        except AttributeError:
            return outward_normal_from_sdf(self.geometry.signed_distance, points)


def _optimize_mesh_ratio_interior(
    interior: NDArray,
    boundary: NDArray,
    geometry: GeometryProtocol,
    target_ratio: float | Literal["optimal"] = "optimal",
    max_iterations: int = 30,
    n_test_points: int = 5000,
    seed: int | None = None,
) -> NDArray:
    """
    Optimize interior point distribution to achieve target mesh ratio h/q.

    Uses Adaptive Repulsion algorithm with:
    - 1/r^2 soft-cutoff repulsion between all points (equal for interior and boundary)
    - Adaptive learning rate (increases when improving, decreases when not)
    - Best-tracking (returns best seen configuration, not last)
    - Optional wall force as safety net (minimal, only near domain boundary)

    This preserves N_col exactly while improving mesh quality.
    Typical results: h/q ~ 1.8-2.0 (close to theoretical optimal 1.414 for square grid).

    Args:
        interior: Interior points (N_int, d) - will be optimized
        boundary: Boundary points (N_bdy, d) - FIXED, used for repulsion
        geometry: Geometry for bounds and SDF
        target_ratio: Target mesh ratio h/q. Can be:
            - float: Stop when h/q <= target (e.g., 5.0)
            - 'optimal': Auto-converge until plateau or h/q < 2.0
        max_iterations: Maximum optimization iterations (default 30 for 'optimal')
        n_test_points: Test points for fill distance estimation
        seed: Random seed

    Returns:
        Tuple of (optimized_interior, fill_distance, separation_distance, mesh_ratio)
        - optimized_interior: Optimized interior points (same N_int)
        - fill_distance: Fill distance h for the best configuration
        - separation_distance: Separation distance q for the best configuration
        - mesh_ratio: Mesh ratio h/q for the best configuration

    Note:
        Mesh ratio uses Wendland convention: q = min_distance / 2.
        This guarantees h/q >= 1 for any point distribution.
        Theoretical optimal: h/q ~ 1.155 (hexagonal), ~ 1.414 (square grid).
    """
    from scipy.spatial import cKDTree

    if len(interior) == 0:
        return interior, None, None, None

    N_int, d = interior.shape
    current_interior = interior.copy()

    # Get bounds
    bounds_tuple = geometry.get_bounds()
    if bounds_tuple is None:
        return current_interior
    min_coords, max_coords = bounds_tuple
    bounds = [(float(min_coords[i]), float(max_coords[i])) for i in range(d)]

    # Total points for spacing calculation
    N_total = N_int + len(boundary)
    bbox_volume = np.prod([b[1] - b[0] for b in bounds])
    target_spacing = (bbox_volume / N_total) ** (1.0 / d)

    rng = np.random.RandomState(seed)
    auto_mode = target_ratio == "optimal"

    # Track best result (interior points + mesh distances)
    best_ratio = float("inf")
    best_interior = current_interior.copy()
    best_h: float | None = None
    best_q: float | None = None
    no_improvement_count = 0

    # Adaptive learning rate
    lr = 0.2

    # Repulsion cutoff
    cutoff = 3.0 * target_spacing

    # Wall margin (safety net, only activates very close to boundary)
    wall_margin = 0.15 * target_spacing

    for iteration in range(max_iterations):
        # Combine for mesh quality computation
        if len(boundary) > 0:
            all_points = np.vstack([current_interior, boundary])
        else:
            all_points = current_interior

        tree = cKDTree(all_points)

        # Separation distance (q) = half of min distance (Wendland convention)
        nn_dist, _ = tree.query(all_points, k=2)
        q = np.min(nn_dist[:, 1]) / 2.0

        # Fill distance (h) via Monte Carlo
        test_points = np.zeros((n_test_points, d))
        for i, (lo, hi) in enumerate(bounds):
            test_points[:, i] = rng.uniform(lo, hi, n_test_points)

        try:
            sdf = geometry.signed_distance(test_points)
            test_points = test_points[sdf < 0]
        except AttributeError:
            pass

        if len(test_points) > 0:
            dist_to_nearest, _ = tree.query(test_points, k=1)
            h = np.max(dist_to_nearest)
        else:
            h = target_spacing * 2

        mesh_ratio = h / q if q > 1e-10 else float("inf")

        # Track best result and adapt learning rate
        if mesh_ratio < best_ratio:
            best_ratio = mesh_ratio
            best_interior = current_interior.copy()
            best_h = h
            best_q = q
            no_improvement_count = 0
            lr = min(lr * 1.1, 0.35)  # Increase lr when improving
        else:
            no_improvement_count += 1
            lr = max(lr * 0.85, 0.05)  # Decrease lr when not improving

        # Log progress
        log_fn = logger.info if auto_mode else logger.debug
        best_marker = " *" if mesh_ratio == best_ratio else ""
        log_fn(f"Mesh ratio optimization iter {iteration}: h={h:.4f}, q={q:.4f}, h/q={mesh_ratio:.2f}{best_marker}")

        # Check convergence
        if auto_mode:
            if best_ratio < 2.0:
                logger.info(
                    f"Mesh ratio optimization: near-optimal h/q={best_ratio:.2f} < 2.0 "
                    f"achieved after {iteration} iterations"
                )
                break
            if no_improvement_count >= 5:
                logger.info(
                    f"Mesh ratio optimization: converged at h/q={best_ratio:.2f} "
                    f"(no improvement for 5 iterations) after {iteration} iterations"
                )
                break
        else:
            if mesh_ratio <= target_ratio:
                logger.info(
                    f"Mesh ratio optimization: achieved h/q={mesh_ratio:.2f} <= {target_ratio} "
                    f"after {iteration} iterations"
                )
                break

        # Compute repulsion forces using Adaptive Repulsion algorithm
        forces = np.zeros_like(current_interior)

        for i in range(N_int):
            # Point-point repulsion (equal strength for interior and boundary)
            neighbors = tree.query_ball_point(current_interior[i], r=cutoff)
            for j in neighbors:
                if j == i:
                    continue
                diff = current_interior[i] - all_points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = rng.randn(d) * 0.01 * target_spacing
                    dist = np.linalg.norm(diff)

                if dist < cutoff:
                    # 1/r^2 repulsion with soft cutoff
                    strength = (target_spacing / dist) ** 2 * (1 - dist / cutoff)
                    forces[i] += strength * diff / dist

            # Minimal wall force (safety net)
            for dim in range(d):
                d_lo = current_interior[i, dim] - bounds[dim][0]
                if 0 < d_lo < wall_margin:
                    wall_strength = ((wall_margin - d_lo) / wall_margin) ** 3
                    forces[i, dim] += wall_strength * target_spacing

                d_hi = bounds[dim][1] - current_interior[i, dim]
                if 0 < d_hi < wall_margin:
                    wall_strength = ((wall_margin - d_hi) / wall_margin) ** 3
                    forces[i, dim] -= wall_strength * target_spacing

        # Normalize and apply forces
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 1e-10:
            forces = forces / max_force * target_spacing

        current_interior = current_interior + lr * forces

        # Hard clip to stay inside domain (safety)
        for dim in range(d):
            eps = 1e-6
            current_interior[:, dim] = np.clip(current_interior[:, dim], bounds[dim][0] + eps, bounds[dim][1] - eps)

    else:
        # Loop exhausted
        if auto_mode:
            logger.info(f"Mesh ratio optimization: max iterations reached, best h/q={best_ratio:.2f}")
        else:
            import warnings

            warnings.warn(
                f"Mesh ratio target h/q={target_ratio} not achieved after {max_iterations} "
                f"iterations. Final h/q={mesh_ratio:.2f}. Consider increasing max_iterations "
                f"or using target_mesh_ratio='optimal' for auto-convergence.",
                UserWarning,
                stacklevel=4,
            )

    return best_interior, best_h, best_q, best_ratio


def _lloyd_relaxation_interior_only(
    interior: NDArray,
    boundary: NDArray,
    bounds: list[tuple[float, float]],
    geometry: GeometryProtocol,
    steps: int = 20,
    boundary_strength: float = 1.5,
    target_spacing: float | None = None,
) -> NDArray:
    """
    Lloyd relaxation for interior points only, with boundary as fixed repulsors.

    Args:
        interior: Interior points to relax (N_int, d)
        boundary: Fixed boundary points (N_bdy, d) - used for repulsion only
        bounds: Domain bounds
        geometry: Geometry for SDF
        steps: Relaxation iterations
        boundary_strength: Boundary repulsion strength
        target_spacing: Target spacing (computed if None)

    Returns:
        Relaxed interior points (same shape)
    """
    from scipy.spatial import cKDTree

    if len(interior) == 0:
        return interior

    N_int, d = interior.shape
    current = interior.copy()

    N_total = N_int + len(boundary)
    if target_spacing is None:
        bbox_volume = np.prod([b[1] - b[0] for b in bounds])
        target_spacing = (bbox_volume / N_total) ** (1.0 / d)

    interaction_radius = 2.5 * target_spacing
    dt_base = 0.15

    for step in range(steps):
        forces = np.zeros_like(current)

        # Build tree with all points (interior + boundary)
        if len(boundary) > 0:
            all_points = np.vstack([current, boundary])
        else:
            all_points = current

        tree = cKDTree(all_points)

        # Inter-particle repulsion (interior-interior and interior-boundary)
        for i in range(N_int):
            # Find neighbors of interior point i
            neighbors = tree.query_ball_point(current[i], r=interaction_radius)

            for j in neighbors:
                if j == i:
                    continue

                # Get the other point (could be interior or boundary)
                other_pt = all_points[j]
                diff = current[i] - other_pt
                dist = np.linalg.norm(diff)

                if dist < 1e-10:
                    diff = np.random.randn(d) * 0.01 * target_spacing
                    dist = np.linalg.norm(diff)

                # Stronger repulsion from boundary points
                if j >= N_int:  # This is a boundary point
                    force_mag = boundary_strength / (dist**2 + 0.01 * target_spacing**2)
                else:
                    force_mag = 1.0 / (dist**2 + 0.01 * target_spacing**2)

                force_vec = (diff / dist) * force_mag
                forces[i] += force_vec

        # Domain boundary repulsion using SDF
        try:
            sdf = geometry.signed_distance(current)
            eps = 1e-6
            grad_sdf = np.zeros_like(current)
            for dim in range(d):
                shift = np.zeros(d)
                shift[dim] = eps
                sdf_plus = geometry.signed_distance(current + shift)
                sdf_minus = geometry.signed_distance(current - shift)
                grad_sdf[:, dim] = (sdf_plus - sdf_minus) / (2 * eps)

            boundary_zone = target_spacing * 2.0
            close_mask = np.abs(sdf) < boundary_zone

            for i in np.where(close_mask)[0]:
                boundary_dist = max(np.abs(sdf[i]), 1e-10)
                force_mag = boundary_strength / (boundary_dist**2)
                forces[i] -= grad_sdf[i] * force_mag * np.sign(sdf[i])
        except AttributeError:
            # Box boundary fallback
            for dim in range(d):
                lo, hi = bounds[dim]
                margin = target_spacing * 2.0
                near_lo = current[:, dim] < lo + margin
                near_hi = current[:, dim] > hi - margin
                for i in np.where(near_lo)[0]:
                    dist_to_wall = max(current[i, dim] - lo, 1e-10)
                    forces[i, dim] += boundary_strength / (dist_to_wall**2)
                for i in np.where(near_hi)[0]:
                    dist_to_wall = max(hi - current[i, dim], 1e-10)
                    forces[i, dim] -= boundary_strength / (dist_to_wall**2)

        # Adaptive time step
        max_force = np.max(np.linalg.norm(forces, axis=1))
        anneal = 1.0 - 0.5 * (step / steps)
        dt = min(dt_base * anneal, 0.4 * target_spacing / (max_force + 1e-10))

        current = current + forces * dt

        # Project outside points back
        try:
            sdf_new = geometry.signed_distance(current)
            outside = sdf_new >= 0
            if np.any(outside):
                grad = np.zeros((np.sum(outside), d))
                for dim in range(d):
                    shift = np.zeros(d)
                    shift[dim] = 1e-6
                    sdf_plus = geometry.signed_distance(current[outside] + shift)
                    sdf_minus = geometry.signed_distance(current[outside] - shift)
                    grad[:, dim] = (sdf_plus - sdf_minus) / (2e-6)
                grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
                grad_norm = np.maximum(grad_norm, 1e-10)
                push_dist = 0.02 * target_spacing
                current[outside] -= (sdf_new[outside, np.newaxis] + push_dist) * grad / grad_norm
        except AttributeError:
            pass

        # Clamp to bounds
        for dim in range(d):
            margin = 1e-6
            current[:, dim] = np.clip(current[:, dim], bounds[dim][0] + margin, bounds[dim][1] - margin)

    return current


def _optimize_mesh_ratio(
    points: NDArray,
    geometry: GeometryProtocol,
    target_ratio: float = 5.0,
    max_iterations: int = 5,
    n_test_points: int = 5000,
    seed: int | None = None,
) -> NDArray:
    """
    Optimize point distribution to achieve target mesh ratio h/q WITHOUT changing N.

    This applies additional Lloyd relaxation with increasing boundary repulsion
    strength until the mesh ratio is acceptable.

    Algorithm:
    1. Compute current mesh ratio h/q
    2. If h/q > target_ratio, apply Lloyd relaxation with stronger boundary repulsion
    3. Repeat until ratio is achieved or max_iterations reached

    The key insight is that poor mesh ratio often comes from interior points
    clustering near boundary points. Increasing boundary repulsion during
    relaxation pushes interior points away, improving both h and q.

    Args:
        points: Collocation points (N, d) - N is preserved
        geometry: Geometry for bounds and SDF
        target_ratio: Target mesh ratio h/q (default 5.0)
        max_iterations: Maximum optimization iterations
        n_test_points: Test points for fill distance estimation
        seed: Random seed

    Returns:
        Optimized points with same shape as input (N unchanged)
    """
    from scipy.spatial import cKDTree

    N, d = points.shape
    current = points.copy()

    # Get bounds
    bounds_tuple = geometry.get_bounds()
    if bounds_tuple is None:
        return current
    min_coords, max_coords = bounds_tuple
    bounds = [(float(min_coords[i]), float(max_coords[i])) for i in range(d)]

    bbox_volume = np.prod([b[1] - b[0] for b in bounds])
    target_spacing = (bbox_volume / N) ** (1.0 / d)

    rng = np.random.RandomState(seed)

    for iteration in range(max_iterations):
        # Compute current mesh distances
        tree = cKDTree(current)

        # Separation distance (q) = half of min distance (Wendland convention)
        nn_dist, _ = tree.query(current, k=2)
        q = np.min(nn_dist[:, 1]) / 2.0

        # Fill distance (h) via Monte Carlo
        test_points = np.zeros((n_test_points, d))
        for i, (lo, hi) in enumerate(bounds):
            test_points[:, i] = rng.uniform(lo, hi, n_test_points)

        # Filter to interior if SDF available
        try:
            sdf = geometry.signed_distance(test_points)
            test_points = test_points[sdf < 0]
        except AttributeError:
            pass

        if len(test_points) > 0:
            dist_to_nearest, _ = tree.query(test_points, k=1)
            h = np.max(dist_to_nearest)
        else:
            h = target_spacing * 2  # Fallback

        mesh_ratio = h / q if q > 1e-10 else float("inf")

        logger.debug(f"Mesh ratio optimization iter {iteration}: h={h:.4f}, q={q:.4f}, h/q={mesh_ratio:.2f}")

        if mesh_ratio <= target_ratio:
            logger.debug(f"Target mesh ratio {target_ratio} achieved after {iteration} iterations")
            break

        # Apply Lloyd relaxation with increasing boundary strength
        # Higher strength pushes interior points away from boundary
        boundary_strength = 1.5 + 0.5 * iteration  # Increase each iteration

        # One relaxation pass
        current = _lloyd_relaxation_pass(
            current,
            bounds,
            geometry,
            steps=20,
            boundary_strength=boundary_strength,
            target_spacing=target_spacing,
        )

    return current


def _lloyd_relaxation_pass(
    points: NDArray,
    bounds: list[tuple[float, float]],
    geometry: GeometryProtocol,
    steps: int = 20,
    boundary_strength: float = 1.5,
    target_spacing: float | None = None,
) -> NDArray:
    """
    Single pass of Lloyd relaxation with configurable boundary strength.

    Args:
        points: Points to relax (N, d)
        bounds: Domain bounds
        geometry: Geometry for SDF
        steps: Relaxation iterations
        boundary_strength: Boundary repulsion strength (higher = more separation)
        target_spacing: Target point spacing (computed if None)

    Returns:
        Relaxed points (same shape)
    """
    from scipy.spatial import cKDTree

    N, d = points.shape
    current = points.copy()

    if target_spacing is None:
        bbox_volume = np.prod([b[1] - b[0] for b in bounds])
        target_spacing = (bbox_volume / N) ** (1.0 / d)

    interaction_radius = 2.5 * target_spacing
    dt_base = 0.15

    for step in range(steps):
        forces = np.zeros_like(current)

        # Inter-particle repulsion
        tree = cKDTree(current)
        pairs = tree.query_pairs(r=interaction_radius)

        for i, j in pairs:
            diff = current[i] - current[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                diff = np.random.randn(d) * 0.01 * target_spacing
                dist = np.linalg.norm(diff)

            force_mag = 1.0 / (dist**2 + 0.01 * target_spacing**2)
            force_vec = (diff / dist) * force_mag
            forces[i] += force_vec
            forces[j] -= force_vec

        # Boundary repulsion using SDF
        try:
            sdf = geometry.signed_distance(current)
            # Compute SDF gradient numerically
            eps = 1e-6
            grad_sdf = np.zeros_like(current)
            for dim in range(d):
                shift = np.zeros(d)
                shift[dim] = eps
                sdf_plus = geometry.signed_distance(current + shift)
                sdf_minus = geometry.signed_distance(current - shift)
                grad_sdf[:, dim] = (sdf_plus - sdf_minus) / (2 * eps)

            # Apply boundary repulsion when close
            boundary_zone = target_spacing * 2.0
            close_mask = np.abs(sdf) < boundary_zone

            for i in np.where(close_mask)[0]:
                boundary_dist = max(np.abs(sdf[i]), 1e-10)
                force_mag = boundary_strength / (boundary_dist**2)
                forces[i] -= grad_sdf[i] * force_mag * np.sign(sdf[i])
        except AttributeError:
            # No SDF, use box boundary repulsion
            for dim in range(d):
                lo, hi = bounds[dim]
                margin = target_spacing * 2.0
                near_lo = current[:, dim] < lo + margin
                near_hi = current[:, dim] > hi - margin
                for i in np.where(near_lo)[0]:
                    dist_to_wall = max(current[i, dim] - lo, 1e-10)
                    forces[i, dim] += boundary_strength / (dist_to_wall**2)
                for i in np.where(near_hi)[0]:
                    dist_to_wall = max(hi - current[i, dim], 1e-10)
                    forces[i, dim] -= boundary_strength / (dist_to_wall**2)

        # Adaptive time step with annealing
        max_force = np.max(np.linalg.norm(forces, axis=1))
        anneal = 1.0 - 0.5 * (step / steps)
        dt = min(dt_base * anneal, 0.4 * target_spacing / (max_force + 1e-10))

        current = current + forces * dt

        # Project outside points back
        try:
            sdf_new = geometry.signed_distance(current)
            outside = sdf_new >= 0
            if np.any(outside):
                # Compute gradient for projection
                grad = np.zeros((np.sum(outside), d))
                for dim in range(d):
                    shift = np.zeros(d)
                    shift[dim] = 1e-6
                    sdf_plus = geometry.signed_distance(current[outside] + shift)
                    sdf_minus = geometry.signed_distance(current[outside] - shift)
                    grad[:, dim] = (sdf_plus - sdf_minus) / (2e-6)
                grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
                grad_norm = np.maximum(grad_norm, 1e-10)
                push_dist = 0.02 * target_spacing
                current[outside] -= (sdf_new[outside, np.newaxis] + push_dist) * grad / grad_norm
        except AttributeError:
            pass

        # Clamp to bounds
        for dim in range(d):
            margin = 1e-6
            current[:, dim] = np.clip(current[:, dim], bounds[dim][0] + margin, bounds[dim][1] - margin)

    return current


def _find_coverage_gaps(
    collocation_points: NDArray,
    geometry: GeometryProtocol,
    h_max: float,
    n_test_points: int = 10000,
    seed: int | None = None,
) -> NDArray:
    """
    Find points in the domain that are farther than h_max from any collocation point.

    These are "coverage gaps" that need additional collocation points.

    Args:
        collocation_points: Current collocation points (N, d)
        geometry: Geometry for bounds and SDF
        h_max: Maximum allowed fill distance
        n_test_points: Number of test points to sample
        seed: Random seed

    Returns:
        Array of gap locations (points farther than h_max from nearest colloc), shape (M, d)
    """
    from scipy.spatial import cKDTree

    d = collocation_points.shape[1]
    rng = np.random.RandomState(seed)

    # Get domain bounds
    bounds_tuple = geometry.get_bounds()
    if bounds_tuple is None:
        raise ValueError("Geometry must have bounded domain for fill distance computation")
    min_coords, max_coords = bounds_tuple

    # Sample test points uniformly in bounding box
    test_points = np.zeros((n_test_points, d))
    for i in range(d):
        test_points[:, i] = rng.uniform(min_coords[i], max_coords[i], n_test_points)

    # Filter to interior (if geometry has SDF)
    try:
        sdf = geometry.signed_distance(test_points)
        interior_mask = sdf < 0
        test_points = test_points[interior_mask]
    except AttributeError:
        pass  # No SDF, use all test points

    if len(test_points) == 0:
        return np.array([]).reshape(0, d)

    # Find distances to nearest collocation point
    tree = cKDTree(collocation_points)
    distances, _ = tree.query(test_points, k=1)

    # Return test points that are farther than h_max (these are gaps)
    gap_mask = distances > h_max
    return test_points[gap_mask]


def _enforce_max_fill_distance(
    interior: NDArray,
    boundary: NDArray,
    geometry: GeometryProtocol,
    h_max: float,
    min_separation: float | None = None,
    max_additions: int = 100,
    max_iterations: int = 10,
    n_test_points: int = 10000,
    seed: int | None = None,
) -> NDArray:
    """
    Ensure fill distance does not exceed h_max by adding interior points in sparse regions.

    Algorithm (greedy adaptive refinement):
    1. Combine current interior and boundary points
    2. Find test points farther than h_max from any collocation point (gaps)
    3. Greedily add new interior points at gap locations:
       - Select the point farthest from existing points
       - Add it if it respects min_separation
       - Repeat until no more gaps or max_additions reached
    4. Iterate until fill distance <= h_max or max_iterations reached

    Args:
        interior: Current interior points (N_int, d)
        boundary: Boundary points (N_bdy, d) - not modified
        geometry: Geometry object for SDF queries
        h_max: Maximum allowed fill distance
        min_separation: Minimum distance between points (default: h_max / 3)
        max_additions: Maximum number of points to add per iteration
        max_iterations: Maximum refinement iterations
        n_test_points: Number of test points for gap detection
        seed: Random seed

    Returns:
        Updated interior points array with additional points added
    """
    from scipy.spatial import cKDTree

    if len(interior) == 0 and len(boundary) == 0:
        logger.warning("No collocation points provided for h_max enforcement")
        return interior

    # Default min_separation to avoid clustering
    if min_separation is None:
        min_separation = h_max / 3.0

    current_interior = interior.copy()
    total_added = 0

    for iteration in range(max_iterations):
        # Combine all points
        if len(current_interior) > 0 and len(boundary) > 0:
            all_points = np.vstack([current_interior, boundary])
        elif len(current_interior) > 0:
            all_points = current_interior
        else:
            all_points = boundary

        # Find coverage gaps
        gap_points = _find_coverage_gaps(
            all_points,
            geometry,
            h_max,
            n_test_points=n_test_points,
            seed=seed + iteration if seed else None,
        )

        if len(gap_points) == 0:
            logger.debug(f"h_max enforcement: converged after {iteration} iterations, added {total_added} points")
            break

        # Greedy selection: add points from gaps, respecting min_separation
        tree = cKDTree(all_points)
        added_this_iter = 0

        # Sort gaps by distance to nearest existing point (largest first)
        distances_to_nearest, _ = tree.query(gap_points, k=1)
        sorted_indices = np.argsort(-distances_to_nearest)  # Descending

        new_points = []
        for idx in sorted_indices:
            if added_this_iter >= max_additions:
                break

            candidate = gap_points[idx]

            # Check if candidate respects min_separation from all existing points
            dist_to_existing = tree.query(candidate.reshape(1, -1), k=1)[0][0]
            if dist_to_existing < min_separation:
                continue

            # Check if candidate respects min_separation from already-added new points
            if len(new_points) > 0:
                new_tree = cKDTree(np.array(new_points))
                dist_to_new = new_tree.query(candidate.reshape(1, -1), k=1)[0][0]
                if dist_to_new < min_separation:
                    continue

            # Verify point is inside domain
            try:
                sdf = geometry.signed_distance(candidate.reshape(1, -1))[0]
                if sdf >= -min_separation / 2:  # Too close to or outside boundary
                    continue
            except AttributeError:
                pass

            new_points.append(candidate)
            added_this_iter += 1

        if added_this_iter == 0:
            logger.debug(
                f"h_max enforcement: stopped at iteration {iteration}, no valid points to add (min_sep constraint)"
            )
            break

        # Add new points to interior
        current_interior = np.vstack([current_interior, np.array(new_points)])
        total_added += added_this_iter

        logger.debug(
            f"h_max enforcement iteration {iteration + 1}: "
            f"added {added_this_iter} points, total = {len(current_interior)}"
        )

    if total_added > 0:
        logger.info(f"h_max enforcement: added {total_added} interior points to achieve h <= {h_max:.4f}")

    return current_interior


def _enforce_separation(
    interior: NDArray,
    boundary: NDArray,
    normals_bdy: NDArray,
    region_ids_bdy: NDArray,
    min_separation: float,
    boundary_margin: float,
    geometry: GeometryProtocol,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Enforce minimum separation between all collocation points.

    This ensures GFDM stability by preventing point clustering that leads
    to ill-conditioned linear systems.

    Strategy:
    1. Keep all boundary points (they define the domain)
    2. Remove interior points that are too close to boundary points
    3. Remove interior points that are too close to each other
    4. Optionally push interior points away from domain boundary

    Args:
        interior: Interior points (N_int, d)
        boundary: Boundary points (N_bdy, d)
        normals_bdy: Normals at boundary points (N_bdy, d)
        region_ids_bdy: Region IDs at boundary points (N_bdy,)
        min_separation: Minimum distance between any two points
        boundary_margin: Minimum distance from interior to domain boundary
        geometry: Geometry object for SDF queries

    Returns:
        Filtered (interior, boundary, normals, region_ids) tuple
    """
    from scipy.spatial import cKDTree

    if len(interior) == 0 or len(boundary) == 0:
        return interior, boundary, normals_bdy, region_ids_bdy

    # Step 1: Remove interior points too close to boundary points
    boundary_tree = cKDTree(boundary)
    distances_to_boundary, _ = boundary_tree.query(interior, k=1)
    interior_keep_mask = distances_to_boundary >= min_separation

    interior_filtered = interior[interior_keep_mask]

    if len(interior_filtered) == 0:
        logger.warning(
            f"All interior points removed due to min_separation={min_separation}. "
            f"Consider reducing min_separation or increasing domain size."
        )
        return interior_filtered, boundary, normals_bdy, region_ids_bdy

    # Step 2: Remove interior points too close to each other
    # Use greedy algorithm: keep points in order, remove conflicts
    interior_tree = cKDTree(interior_filtered)
    pairs = interior_tree.query_pairs(r=min_separation)

    # Build conflict graph and greedily remove
    to_remove = set()
    for i, j in pairs:
        if i not in to_remove and j not in to_remove:
            # Remove the one with higher index (arbitrary but consistent)
            to_remove.add(j)

    keep_indices = [i for i in range(len(interior_filtered)) if i not in to_remove]
    interior_filtered = interior_filtered[keep_indices]

    # Step 3: Push interior points away from domain boundary (if geometry supports SDF)
    if boundary_margin > 0:
        try:
            sdf = geometry.signed_distance(interior_filtered)
            # Points with |sdf| < boundary_margin are too close to boundary
            too_close_mask = np.abs(sdf) < boundary_margin
            if np.any(too_close_mask):
                # Remove points too close to domain boundary
                interior_filtered = interior_filtered[~too_close_mask]
                logger.debug(
                    f"Removed {np.sum(too_close_mask)} interior points within {boundary_margin} of domain boundary"
                )
        except AttributeError:
            # Geometry doesn't support SDF, skip this step
            pass

    n_removed = len(interior) - len(interior_filtered)
    if n_removed > 0:
        logger.info(
            f"Separation enforcement: removed {n_removed}/{len(interior)} interior points "
            f"(min_sep={min_separation:.4f}, margin={boundary_margin:.4f})"
        )

    return interior_filtered, boundary, normals_bdy, region_ids_bdy


class CollocationSampler:
    """
    Unified collocation sampler for any geometry type.

    Automatically selects the appropriate strategy based on geometry type:
    - TensorProductGrid → CartesianGridCollocation
    - ImplicitDomain/Hypersphere/Hyperrectangle → ImplicitDomainCollocation
    - CSG domains → CSGDomainCollocation (delegates to components)

    Example:
        >>> from mfgarchon.geometry import Hyperrectangle
        >>> domain = Hyperrectangle(bounds=[[0, 1], [0, 1]])
        >>> sampler = CollocationSampler(domain)
        >>> coll = sampler.generate_collocation(n_interior=400, n_boundary=100)
        >>> print(f"Generated {coll.n_interior} interior, {coll.n_boundary} boundary")
    """

    def __init__(self, geometry: GeometryProtocol):
        """
        Initialize collocation sampler.

        Args:
            geometry: Geometry object supporting the GeometryProtocol
        """
        self.geometry = geometry
        self._strategy = self._select_strategy()

    def _select_strategy(self) -> BaseCollocationStrategy:
        """Select appropriate strategy based on geometry type."""
        # Issue #543: Use try/except and concrete types instead of hasattr()
        geom = self.geometry

        # Check for TensorProductGrid by name (concrete type check)
        if type(geom).__name__ == "TensorProductGrid":
            return CartesianGridCollocation(geom)

        # Check for Hyperrectangle - use CartesianGridCollocation for axis-aligned box
        # This produces evenly-spaced boundary points via linspace, which is critical
        # for GFDM mesh ratio optimization (Issue #711)
        if type(geom).__name__ == "Hyperrectangle":
            return CartesianGridCollocation(geom)

        # Check for grid-like geometry (has get_grid_points method)
        try:
            _ = geom.get_grid_points
            return CartesianGridCollocation(geom)
        except AttributeError:
            pass

        # Check for implicit domain (has signed_distance method)
        try:
            _ = geom.signed_distance
            # Check for CSG composite (has domains or base_domain)
            try:
                _ = geom.domains
                # TODO: Implement CSGDomainCollocation
                return ImplicitDomainCollocation(geom)
            except AttributeError:
                try:
                    _ = geom.base_domain
                    # TODO: Implement CSGDomainCollocation
                    return ImplicitDomainCollocation(geom)
                except AttributeError:
                    pass
            return ImplicitDomainCollocation(geom)
        except AttributeError:
            pass

        # Check for mesh (has vertices)
        try:
            _ = geom.vertices
            # TODO: Implement MeshCollocation
            raise NotImplementedError("MeshCollocation not yet implemented")
        except AttributeError:
            pass

        # Fallback: use GeometryProtocol bounds
        if isinstance(geom, GeometryProtocol):
            bounds = geom.get_bounds()
            if bounds is not None:
                return CartesianGridCollocation(geom)

        raise ValueError(
            f"Unsupported geometry type: {type(geom).__name__}. "
            f"Geometry must implement GeometryProtocol with valid bounds, "
            f"or provide signed_distance, get_grid_points, or vertices methods."
        )

    def sample_interior(
        self,
        n_points: int,
        method: Literal["poisson_disk", "sobol", "uniform"] = "poisson_disk",
        seed: int | None = None,
        refine_steps: int = 20,
    ) -> NDArray:
        """
        Sample interior collocation points.

        Args:
            n_points: Number of interior points
            method: Sampling method ("poisson_disk", "sobol", "uniform")
            seed: Random seed
            refine_steps: Particle repulsion iterations (0 to disable)

        Returns:
            Interior points, shape (n_points, d)
        """
        # Check if strategy supports refine_steps
        import inspect

        sig = inspect.signature(self._strategy.sample_interior)
        if "refine_steps" in sig.parameters:
            return self._strategy.sample_interior(n_points, method, seed, refine_steps)
        return self._strategy.sample_interior(n_points, method, seed)

    def sample_boundary(
        self,
        n_points: int,
        method: Literal["uniform", "poisson_disk"] = "uniform",
        seed: int | None = None,
        refine_steps: int = 30,
    ) -> tuple[NDArray, NDArray]:
        """
        Sample boundary collocation points.

        Args:
            n_points: Number of boundary points
            method: Sampling method
            seed: Random seed
            refine_steps: Boundary repulsion iterations (0 to disable)

        Returns:
            Tuple of (points, normals)
        """
        # Check if strategy supports refine_steps
        import inspect

        sig = inspect.signature(self._strategy.sample_boundary)
        if "refine_steps" in sig.parameters:
            points, normals, _ = self._strategy.sample_boundary(n_points, method, seed, refine_steps)
        else:
            points, normals, _ = self._strategy.sample_boundary(n_points, method, seed)
        return points, normals

    def generate_collocation(
        self,
        n_interior: int,
        n_boundary: int,
        interior_method: Literal["poisson_disk", "sobol", "uniform"] = "poisson_disk",
        boundary_method: Literal["uniform", "poisson_disk"] = "uniform",
        seed: int | None = None,
        min_separation: float | None = None,
        boundary_margin: float | None = None,
        max_fill_distance: float | None = None,
        target_mesh_ratio: float | Literal["optimal"] | None = None,
    ) -> CollocationPointSet:
        """
        Generate complete collocation point set.

        Args:
            n_interior: Number of interior points
            n_boundary: Number of boundary points. For fully periodic domains
                (torus), this is automatically set to 0 with a warning.
            interior_method: Sampling method for interior
            boundary_method: Sampling method for boundary
            seed: Random seed
            min_separation: Minimum distance between any two points (h_min).
                If None, no enforcement. If specified, interior points that
                are too close to boundary points will be removed or relocated.
                NOTE: This may change N_col.
            boundary_margin: Minimum distance from interior points to domain
                boundary. If None, defaults to min_separation. This prevents
                interior points from clustering near boundary points.
            max_fill_distance: Maximum allowed fill distance (h_max).
                If specified, additional interior points will be added in
                sparse regions to ensure no point in the domain is farther
                than h_max from a collocation point.
                NOTE: This may change N_col.
            target_mesh_ratio: Target mesh ratio h/q. Can be:
                - float (e.g., 5.0): Stop when h/q <= target
                - 'optimal': Auto-converge until plateau or h/q < 2.0
                Applies iterative Lloyd relaxation WITHOUT changing N.
                This is the RECOMMENDED approach for EOC studies where N must
                be controlled exactly.

        Returns:
            CollocationPointSet with combined points and metadata

        Note:
            When min_separation or max_fill_distance is specified, the actual
            number of points may differ from requested (N changes).

            When target_mesh_ratio is specified, N is preserved but the point
            distribution is optimized to achieve the target h/q ratio.

            If both min_separation and target_mesh_ratio are specified,
            min_separation is applied first (may change N), then mesh ratio
            optimization is applied (preserves the new N).

            For fully periodic domains (torus topology), n_boundary is forced
            to 0 since there are no true boundaries (Issue #720).
        """
        # Issue #720: Detect fully periodic domains and force n_boundary=0
        if isinstance(self.geometry, SupportsPeriodic):
            periodic_dims = self.geometry.periodic_dimensions
            dimension = self.geometry.dimension
            if len(periodic_dims) == dimension and n_boundary > 0:
                import warnings

                warnings.warn(
                    f"Fully periodic domain (torus) detected with {len(periodic_dims)} periodic dimensions. "
                    f"Forcing n_boundary=0 (was {n_boundary}). Torus has no boundaries.",
                    UserWarning,
                    stacklevel=2,
                )
                n_boundary = 0

        # Sample interior
        interior = self.sample_interior(n_interior, interior_method, seed)

        d = interior.shape[1]

        # Sample boundary (skip if n_boundary=0, e.g., for fully periodic domains)
        if n_boundary > 0:
            boundary, normals_bdy, region_ids_bdy = self._strategy.sample_boundary(n_boundary, boundary_method, seed)
        else:
            # No boundary points for periodic domains
            boundary = np.zeros((0, d))
            normals_bdy = np.zeros((0, d))
            region_ids_bdy = np.array([], dtype=int)

        # Enforce minimum separation if specified
        if min_separation is not None and min_separation > 0:
            margin = boundary_margin if boundary_margin is not None else min_separation
            interior, boundary, normals_bdy, region_ids_bdy = _enforce_separation(
                interior,
                boundary,
                normals_bdy,
                region_ids_bdy,
                min_separation=min_separation,
                boundary_margin=margin,
                geometry=self.geometry,
            )
            logger.debug(
                f"After separation enforcement: {len(interior)} interior, "
                f"{len(boundary)} boundary (min_sep={min_separation:.4f})"
            )

        # Enforce maximum fill distance if specified (add points in sparse regions)
        # NOTE: This changes N_col
        if max_fill_distance is not None and max_fill_distance > 0:
            # Use min_separation for the new points, default to h_max/3 if not specified
            sep_for_new = min_separation if min_separation is not None else max_fill_distance / 3.0
            interior = _enforce_max_fill_distance(
                interior,
                boundary,
                self.geometry,
                h_max=max_fill_distance,
                min_separation=sep_for_new,
                seed=seed,
            )
            logger.debug(f"After h_max enforcement: {len(interior)} interior (h_max={max_fill_distance:.4f})")

        # Optimize mesh ratio if specified (does NOT change N_col)
        # This is the RECOMMENDED approach for EOC studies
        should_optimize = (
            target_mesh_ratio is not None
            and len(interior) > 0
            and (target_mesh_ratio == "optimal" or target_mesh_ratio > 0)
        )
        fill_distance: float | None = None
        separation_distance: float | None = None
        mesh_ratio_val: float | None = None

        if should_optimize:
            # Only optimize interior points; boundary points stay fixed
            interior, fill_distance, separation_distance, mesh_ratio_val = _optimize_mesh_ratio_interior(
                interior,
                boundary,
                self.geometry,
                target_ratio=target_mesh_ratio,
                seed=seed,
            )
            target_desc = target_mesh_ratio if target_mesh_ratio == "optimal" else f"{target_mesh_ratio:.1f}"
            logger.debug(
                f"After mesh ratio optimization: {len(interior)} interior, "
                f"{len(boundary)} boundary (target h/q={target_desc})"
            )

        # Combine
        n_int = len(interior)
        n_bdy = len(boundary)

        points = np.vstack([interior, boundary])
        is_boundary = np.concatenate([np.zeros(n_int, dtype=bool), np.ones(n_bdy, dtype=bool)])
        normals = np.vstack([np.zeros((n_int, d)), normals_bdy])
        region_ids = np.concatenate([np.full(n_int, -1), region_ids_bdy])

        return CollocationPointSet(
            points=points,
            is_boundary=is_boundary,
            normals=normals,
            region_ids=region_ids,
            fill_distance=fill_distance,
            separation_distance=separation_distance,
            mesh_ratio=mesh_ratio_val,
        )


# Convenience function
def generate_collocation(
    geometry: GeometryProtocol,
    n_interior: int,
    n_boundary: int,
    min_separation: float | None = None,
    boundary_margin: float | None = None,
    max_fill_distance: float | None = None,
    target_mesh_ratio: float | Literal["optimal"] | None = None,
    **kwargs,
) -> CollocationPointSet:
    """
    Convenience function to generate collocation points.

    There are two approaches to mesh quality control:

    **Approach 1: Fixed N, optimize distribution (RECOMMENDED for EOC)**
        Use `target_mesh_ratio` to optimize point distribution while keeping
        N = n_interior + n_boundary exactly. Best for convergence studies.

    **Approach 2: Constraint enforcement, N may vary**
        Use `min_separation` and/or `max_fill_distance` to enforce hard
        constraints. N may increase or decrease to satisfy constraints.

    Args:
        geometry: Geometry object
        n_interior: Number of interior points
        n_boundary: Number of boundary points
        min_separation: (Approach 2) Minimum distance between points (h_min).
            May DECREASE N by removing clustered points.
        boundary_margin: Minimum distance from interior to boundary.
            Defaults to min_separation if not specified.
        max_fill_distance: (Approach 2) Maximum fill distance (h_max).
            May INCREASE N by adding points in sparse regions.
        target_mesh_ratio: (Approach 1) Target mesh ratio h/q. Can be:
            - float (e.g., 5.0): Stop when h/q <= target
            - 'optimal': Auto-converge until plateau or h/q < 2.0
            Optimizes distribution WITHOUT changing N. Recommended for EOC.
        **kwargs: Additional arguments (interior_method, seed, etc.)

    Returns:
        CollocationPointSet

    Example:
        >>> from mfgarchon.geometry import Hyperrectangle
        >>> domain = Hyperrectangle(bounds=[[0, 1], [0, 1]])
        >>>
        >>> # Approach 1a: Auto-optimize mesh quality (recommended)
        >>> coll = generate_collocation(domain, 100, 40, target_mesh_ratio='optimal')
        >>> assert len(coll.points) == 140  # N preserved
        >>>
        >>> # Approach 1b: Explicit target
        >>> coll = generate_collocation(domain, 100, 40, target_mesh_ratio=5.0)
        >>>
        >>> # Approach 2: Hard constraints (N may vary)
        >>> coll = generate_collocation(domain, 100, 40, min_separation=0.05)
        >>> # N might be < 140 if points removed
        >>>
        >>> # Periodic domain (torus): n_boundary forced to 0 (Issue #720)
        >>> torus = Hyperrectangle(bounds=[[0, 1], [0, 1]], periodic_dims=(0, 1))
        >>> coll = generate_collocation(torus, 100, 40)  # Warning: n_boundary -> 0
        >>> assert coll.n_boundary == 0  # Torus has no boundaries
    """
    sampler = CollocationSampler(geometry)
    return sampler.generate_collocation(
        n_interior,
        n_boundary,
        min_separation=min_separation,
        boundary_margin=boundary_margin,
        max_fill_distance=max_fill_distance,
        target_mesh_ratio=target_mesh_ratio,
        **kwargs,
    )

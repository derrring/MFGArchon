"""
Region and boundary trait protocols for geometry capabilities.

This module defines Protocol classes for boundary-related geometry capabilities:
- Boundary normal computation
- Boundary projection
- Signed distance function (SDF) evaluation
- Region marking and query

These protocols enable geometry-agnostic BC application and region-based
problem setup (mixed BCs, obstacle definitions, etc.).

Example:
    # Mark region for mixed BC
    if isinstance(geometry, SupportsRegionMarking):
        geometry.mark_region("inlet", lambda x: x[0] < 0.1)

    # Apply BC to marked region
    bc_segment = BCSegment(
        name="inlet_bc",
        bc_type=BCType.DIRICHLET,
        value=1.0,
        boundary="inlet",  # References marked region
    )

Created: 2026-01-17 (Issue #590 - Phase 1.1)
Part of: Geometry & BC Architecture Implementation (Issue #589)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from numpy.typing import NDArray


@runtime_checkable
class SupportsBoundaryNormal(Protocol):
    """
    Geometry can compute outward normal vectors at boundary points.

    Universal Outward Normal Convention (Issue #661):
        - n points FROM domain interior TO exterior
        - ∂u/∂n > 0 means u increases in the outward direction
        - For reflecting BC: velocity component along n reverses sign
        - For SDF φ (with φ < 0 inside, φ > 0 outside): n = ∇φ / |∇φ|

    Applications:
        - Neumann BC (∂u/∂n = g): Need normal direction for derivative
        - Robin BC (αu + β∂u/∂n = g): Need normal for mixed condition
        - Flux computation: J·n where n is outward normal
        - Reflecting BC (particles): Reflect velocity across normal

    Discretization-Specific Implementations:
        - TensorProductGrid: Axis-aligned normals ([±1,0,0], [0,±1,0], etc.)
        - ImplicitDomain: SDF gradient gives normal
        - UnstructuredMesh: Element face normals from mesh topology
        - GraphGeometry: Not applicable (discrete graph)
    """

    def get_outward_normal(
        self,
        points: NDArray,
        boundary_name: str | None = None,
    ) -> NDArray:
        """
        Compute outward unit normal vectors at boundary points.

        Args:
            points: Points at which to evaluate normal, shape (num_points, dimension)
                    or (dimension,) for single point
            boundary_name: Optional boundary region name (e.g., "x_min", "x_max").
                           If None, assumes points are on boundary and computes normal.

        Returns:
            Outward unit normals, shape (num_points, dimension) or (dimension,) for single point.
            Each row is a unit vector: |n| = 1

        Raises:
            ValueError: If points not on boundary (when boundary_name=None)
            ValueError: If boundary_name not found in geometry

        Example:
            >>> grid = TensorProductGrid(bounds=[(0,1), (0,1)], ...)
            >>> # Get normal at left boundary (x=0)
            >>> normal = grid.get_outward_normal(points=np.array([0.0, 0.5]), boundary_name="x_min")
            >>> assert np.allclose(normal, [-1.0, 0.0])  # Points left (outward)

        Note:
            - For rectangular domains: normals are axis-aligned
            - For curved boundaries: normals computed from SDF gradient
            - For mesh geometries: normals from element face orientation
        """
        ...


@runtime_checkable
class SupportsBoundaryProjection(Protocol):
    """
    Geometry can project points onto boundary.

    Boundary projection is needed for:
    - Particle methods: When particle crosses boundary, project back
    - Semi-Lagrangian: Project characteristic foot points inside domain
    - Constraint satisfaction: Enforce points remain in domain
    - Closest point on boundary queries

    Mathematical Definition:
        For point x ∉ Ω, find x_b ∈ ∂Ω minimizing |x - x_b|

    Discretization-Specific Implementations:
        - TensorProductGrid: Clamp to domain bounds
        - ImplicitDomain: SDF-guided projection (gradient descent on φ)
        - UnstructuredMesh: Project to nearest face
    """

    def project_to_boundary(
        self,
        points: NDArray,
        boundary_name: str | None = None,
    ) -> NDArray:
        """
        Project points onto domain boundary.

        Args:
            points: Points to project, shape (num_points, dimension) or (dimension,) for single
            boundary_name: Optional specific boundary to project onto (e.g., "x_min").
                           If None, project to closest boundary.

        Returns:
            Projected points on boundary, same shape as input

        Example:
            >>> grid = TensorProductGrid(bounds=[(0,1), (0,1)], ...)
            >>> # Particle outside domain
            >>> x_particle = np.array([1.2, 0.5])  # Outside x_max
            >>> x_boundary = grid.project_to_boundary(x_particle)
            >>> assert np.allclose(x_boundary, [1.0, 0.5])  # Projected to x=1 boundary

        Note:
            - For rectangular domains: Clamp each coordinate independently
            - For general domains: Use SDF gradient descent or closest point algorithm
        """
        ...

    def project_to_interior(
        self,
        points: NDArray,
        tolerance: float = 1e-10,
    ) -> NDArray:
        """
        Project points from outside domain into interior.

        This is the inverse of project_to_boundary: points outside domain
        are moved just inside the boundary.

        Args:
            points: Points to project, shape (num_points, dimension) or (dimension,)
            tolerance: Distance to move inside boundary (for numerical stability)

        Returns:
            Projected points in interior, same shape as input

        Example:
            >>> # Particle crossed boundary, move back inside
            >>> x_outside = np.array([1.05, 0.5])  # Just outside x_max=1.0
            >>> x_inside = grid.project_to_interior(x_outside, tolerance=1e-3)
            >>> assert x_inside[0] <= 1.0  # Back inside domain

        Note:
            - For rectangular domains: Clamp to [xmin + tol, xmax - tol]
            - For SDF domains: Move inward by tolerance along normal
        """
        ...


@runtime_checkable
class SupportsBoundaryDistance(Protocol):
    """
    Geometry can compute signed distance to boundary (SDF).

    Signed Distance Function (SDF) φ(x) satisfies:
        - φ(x) < 0: x is inside domain Ω
        - φ(x) = 0: x is on boundary ∂Ω
        - φ(x) > 0: x is outside domain
        - |∇φ(x)| = 1: SDF property (distance to boundary)

    SDF is essential for:
    - ImplicitDomain geometries (primary representation)
    - Level Set method (interface tracking)
    - Boundary proximity queries
    - Adaptive mesh refinement near boundaries

    Discretization-Specific Implementations:
        - ImplicitDomain: SDF is native representation
        - TensorProductGrid: Computed from domain bounds
        - UnstructuredMesh: Computed from mesh faces (expensive)
    """

    def get_signed_distance(
        self,
        points: NDArray,
    ) -> NDArray:
        """
        Compute signed distance to boundary for given points.

        Args:
            points: Query points, shape (num_points, dimension) or (dimension,) for single point

        Returns:
            Signed distances, shape (num_points,) or scalar for single point
                - Negative: Inside domain
                - Zero: On boundary
                - Positive: Outside domain

        Example:
            >>> # Circle of radius R at origin
            >>> domain = ImplicitDomain(sdf=lambda x: np.linalg.norm(x) - R)
            >>> points = np.array([[0, 0], [R, 0], [2*R, 0]])  # Center, boundary, outside
            >>> phi = domain.get_signed_distance(points)
            >>> assert phi[0] < 0  # Inside
            >>> assert np.isclose(phi[1], 0)  # On boundary
            >>> assert phi[2] > 0  # Outside

        Note:
            - For rectangular domains: φ(x) = max(max(xmin - x), max(x - xmax))
            - For general domains: SDF must be provided or computed
            - If geometry doesn't store exact SDF, may return approximate distance
        """
        ...


@runtime_checkable
class SupportsRegionMarking(Protocol):
    """
    Geometry can mark and query named spatial regions.

    Region marking enables:
    - Mixed boundary conditions (different BC on different regions)
    - Obstacle/constraint definitions (e.g., "safe_zone" for obstacle problems)
    - Source/sink regions (e.g., "inlet", "outlet")
    - Material/domain partitioning

    Example:
        >>> geometry.mark_region("inlet", predicate=lambda x: x[0] < 0.1)
        >>> geometry.mark_region("exit", predicate=lambda x: x[1] > 0.9)
        >>> inlet_mask = geometry.get_region_mask("inlet")
        >>> both_mask = geometry.intersect_regions("inlet", "exit")

    Created: 2026-01-17 (Issue #590 - Phase 1.3)
    """

    def mark_region(
        self,
        name: str,
        predicate: Callable[..., NDArray[np.bool_]] | None = None,
        mask: NDArray[np.bool_] | None = None,
        boundary: str | None = None,
    ) -> None:
        """
        Mark a named spatial region for later reference.

        Args:
            name: Unique name for this region (e.g., "inlet", "obstacle", "safe_zone")
            predicate: Function of ``x``, a coordinate tuple, to a bool per point
                       True where point is in region
            mask: Boolean mask directly specifying region (num_grid_points,)
            boundary: Standard boundary name (e.g., "x_min", "x_max") for rectangular domains

        Raises:
            ValueError: If name already exists
            ValueError: If neither predicate, mask, nor boundary provided

        Example:
            >>> # Box region
            >>> geometry.mark_region(
            ...     "inlet",
            ...     predicate=lambda x: np.all((x >= [0, 0]) & (x <= [0.1, 1.0]), axis=-1)
            ... )
            >>>
            >>> # Circular obstacle
            >>> geometry.mark_region(
            ...     "obstacle",
            ...     predicate=lambda x: np.linalg.norm(x - center) < radius
            ... )
            >>>
            >>> # Boundary region
            >>> geometry.mark_region("left_wall", boundary="x_min")

        Note:
            - Predicates evaluated at geometry's grid points or discretization points
            - For meshfree geometries, predicate evaluated at collocation points
            - Regions can be referenced in BCSegment via boundary parameter
        """
        ...

    def get_region_mask(
        self,
        name: str,
    ) -> NDArray[np.bool_]:
        """
        Get boolean mask for named region.

        Args:
            name: Region name (from mark_region call)

        Returns:
            Boolean mask of shape (num_grid_points,)
            True at grid points in region

        Raises:
            KeyError: If region name not found

        Example:
            >>> inlet_mask = geometry.get_region_mask("inlet")
            >>> u[inlet_mask] = 1.0  # Set value in region
        """
        ...

    def intersect_regions(
        self,
        *names: str,
    ) -> NDArray[np.bool_]:
        """
        Get intersection of multiple regions (boolean AND).

        Args:
            *names: Region names to intersect

        Returns:
            Boolean mask: True where all regions overlap

        Example:
            >>> # Points that are both in "inlet" and "high_priority"
            >>> mask = geometry.intersect_regions("inlet", "high_priority")
        """
        ...

    def union_regions(
        self,
        *names: str,
    ) -> NDArray[np.bool_]:
        """
        Get union of multiple regions (boolean OR).

        Args:
            *names: Region names to union

        Returns:
            Boolean mask: True where any region is True

        Example:
            >>> # All exit points
            >>> mask = geometry.union_regions("exit_top", "exit_bottom", "exit_sides")
        """
        ...

    def get_region_names(self) -> list[str]:
        """
        Get list of all registered region names.

        Returns:
            List of region names in registration order

        Example:
            >>> names = geometry.get_region_names()
            >>> print("Available regions:", names)
            Available regions: ['inlet', 'exit', 'obstacle', 'walls']
        """
        ...

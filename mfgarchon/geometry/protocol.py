#!/usr/bin/env python3
"""
Unified geometry protocol for MFG problems.

This module defines the protocols that all geometry objects must satisfy:
- GeometryProtocol: Core geometry interface (dimension, points, grid)
- BoundaryAwareProtocol: Boundary information interface (detection, normals, projection)

The separation allows BC applicators to work with any geometry that provides
boundary information, while keeping geometry definition and BC application separate.

Created: 2025-11-05
Updated: 2025-11-27 - Added BoundaryAwareProtocol for unified BC handling
Part of: Unified Geometry Parameter Design
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class GeometryType(Enum):
    """
    Enumeration of supported geometry types.

    Attributes:
        CARTESIAN_GRID: Rectangular tensor product grid
        NETWORK: Graph/network topology
        MAZE: Grid-based maze with obstacles
        UNSTRUCTURED_MESH: Unstructured mesh geometry (any dimension)
        IMPLICIT: Implicit geometry (level set or SDF)
        CUSTOM: User-defined custom geometry
    """

    CARTESIAN_GRID = "cartesian_grid"
    NETWORK = "network"
    MAZE = "maze"
    UNSTRUCTURED_MESH = "unstructured_mesh"
    IMPLICIT = "implicit"
    CUSTOM = "custom"


@runtime_checkable
class GeometryProtocol(Protocol):
    """
    Protocol that all geometry objects must satisfy.

    This defines the complete interface required for a geometry to be used
    with MFGProblem solvers, including boundary information.

    Every domain has a boundary (even "unbounded" is a boundary type).
    Boundary methods are MANDATORY - they enable unified BC handling.

    Core geometry methods:
        - dimension: int - Spatial dimension
        - geometry_type: GeometryType - Type of geometry
        - num_spatial_points: int - Total number of discrete spatial points
        - get_spatial_grid() - Returns grid/mesh representation
        - get_bounds() - Returns bounding box

    Boundary methods (mandatory - every domain has boundary):
        - is_on_boundary() - Check if points are on boundary
        - get_boundary_normal() - Get outward normal at boundary points
        - project_to_boundary() - Project points onto boundary
        - project_to_interior() - Project outside points back into domain
        - get_boundary_regions() - Get named boundary regions for mixed BCs
    """

    # =========================================================================
    # Core Geometry Methods
    # =========================================================================

    @property
    def dimension(self) -> int:
        """Spatial dimension of the geometry."""
        ...

    @property
    def geometry_type(self) -> GeometryType:
        """Type of geometry."""
        ...

    @property
    def num_spatial_points(self) -> int:
        """Total number of discrete spatial points."""
        ...

    def get_spatial_grid(self) -> np.ndarray | list[np.ndarray]:
        """
        Get spatial grid representation.

        Returns:
            For Cartesian grids: meshgrid arrays
            For networks: adjacency matrix or node list
            For implicit: sampled points satisfying constraint
        """
        ...

    def get_bounds(self) -> tuple[NDArray[np.floating], NDArray[np.floating]] | None:
        """
        Get bounding box of the domain.

        Returns:
            (min_coords, max_coords) tuple of arrays, or None if unbounded
            - min_coords: Array of shape (d,) with minimum coordinates
            - max_coords: Array of shape (d,) with maximum coordinates
        """
        ...

    def get_problem_config(self) -> dict:
        """
        Return configuration dict for MFGProblem initialization.

        This polymorphic method allows each geometry type to specify how it
        should configure MFGProblem, avoiding hasattr checks and duck typing.

        Returns:
            Dictionary with keys:
                - num_spatial_points: int - Total number of spatial points
                - spatial_shape: tuple - Shape of spatial arrays
                - spatial_bounds: tuple of tuples or None - Bounds [(min, max), ...]
                - legacy_1d_attrs: dict or None - Legacy 1D attributes (xmin, xmax, etc.)

        Added in v0.10.1 for polymorphic geometry handling.
        """
        ...

    # =========================================================================
    # Discretization and Boundary Conditions Methods
    # =========================================================================

    def get_grid_shape(self) -> tuple[int, ...]:
        """
        Get grid discretization shape.

        Returns:
            Tuple of grid sizes along each dimension: (N₁, N₂, ..., Nₐ)
            where d is the spatial dimension.

            - Structured grids: (Nx, Ny, Nz, ...) for arbitrary dimension d
            - Unstructured/meshfree: (N,) where N is total number of points

        Notes:
            This method replaces legacy problem.Nx, problem.Ny attributes.
            All solvers should use geometry.get_grid_shape() instead.

        Examples:
            >>> # 1D grid
            >>> grid_1d.get_grid_shape()
            (100,)
            >>> # 3D grid
            >>> grid_3d.get_grid_shape()
            (50, 50, 50)
        """
        ...

    def get_boundary_conditions(self):
        """
        Get boundary conditions for this geometry.

        Returns:
            BoundaryConditions object defining BC for this domain.
            May return None if BCs are not yet specified (deferred to solve time).

        Notes:
            This is the standard method all solvers should use to access BCs.
            Replaces inconsistent patterns like get_boundary_handler() and
            direct attribute access.
        """
        ...

    def get_collocation_points(self) -> NDArray[np.floating]:
        """
        Get collocation points for discretization.

        Returns:
            Array of shape (N, d) where:
            - N: number of collocation points
            - d: spatial dimension

        Notes:
            - For structured grids: returns flattened grid points
            - For meshfree methods: returns scattered collocation points
            - For GFDM/particle methods: primary discretization points

        Examples:
            >>> geometry.get_collocation_points()
            array([[0.0, 0.0],
                   [0.5, 0.0],
                   [1.0, 0.0],
                   ...])
        """
        ...

    # =========================================================================
    # Boundary Methods (mandatory - every domain has boundary)
    # =========================================================================

    def is_on_boundary(
        self,
        points: NDArray[np.floating],
        tolerance: float = 1e-10,
    ) -> NDArray[np.bool_]:
        """
        Check if points are on the domain boundary.

        Args:
            points: Array of shape (n, d) - points to check
            tolerance: Distance tolerance for boundary detection

        Returns:
            Boolean array of shape (n,) - True if point is on boundary

        Notes:
            - For bounded domains: points at domain edges
            - For unbounded domains: returns all False
            - For periodic domains: returns all False (no real boundary)
        """
        ...

    def get_boundary_normal(
        self,
        points: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Get outward unit normal vectors at boundary points.

        Universal Outward Normal Convention (Issue #661):
            - n points FROM domain interior TO exterior
            - ∂u/∂n > 0 means u increases in the outward direction
            - For reflecting BC: particle velocity component along n reverses sign
            - For SDF φ (with φ < 0 inside, φ > 0 outside): n = ∇φ / |∇φ|

        Args:
            points: Array of shape (n, d) - boundary points

        Returns:
            Array of shape (n, d) - unit outward normal at each point
            Each row satisfies |n| = 1 (unit vector)

        Notes:
            - For non-boundary points, behavior is undefined
            - For implicit domains: gradient of SDF → n = ∇φ / |∇φ|
            - For rectangular grids: axis-aligned normals (e.g., [-1,0] at x_min)
            - For unstructured mesh: element face normals from mesh topology

        Example:
            >>> # Rectangle [0,1] × [0,1]
            >>> grid = TensorProductGrid(bounds=[(0,1), (0,1)], ...)
            >>> # Left boundary (x=0): outward normal points left
            >>> n = grid.get_boundary_normal(np.array([[0.0, 0.5]]))
            >>> assert np.allclose(n, [[-1.0, 0.0]])

        See Also:
            - get_outward_normal(): Alias with optional boundary_name parameter
            - SupportsBoundaryNormal protocol in geometry/protocols/regions.py
        """
        ...

    def project_to_boundary(
        self,
        points: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Project points onto the domain boundary.

        Args:
            points: Array of shape (n, d) - points to project

        Returns:
            Array of shape (n, d) - projected points on boundary
        """
        ...

    def project_to_interior(
        self,
        points: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Project points outside the domain back into the interior.

        For points already inside, returns them unchanged.
        For points outside, projects to nearest boundary point.

        Args:
            points: Array of shape (n, d) - points to project

        Returns:
            Array of shape (n, d) - points guaranteed to be inside domain
        """
        ...

    def get_boundary_regions(self) -> dict[str, dict]:
        """
        Get information about distinct boundary regions.

        Returns:
            Dictionary mapping region names to region info:
            {
                "x_min": {"axis": 0, "side": "min", "value": 0.0},
                "x_max": {"axis": 0, "side": "max", "value": 1.0},
                "all": {},  # For simple domains with uniform BC
            }

        For simple domains, returns {"all": {}}.
        For complex domains, returns distinct regions for mixed BCs.
        """
        ...

    # =========================================================================
    # Boundary Helper Methods (Added for Issue #545 - Unified BC Workflow)
    # =========================================================================

    def get_boundary_indices(
        self,
        points: NDArray[np.floating],
        tolerance: float = 1e-10,
    ) -> NDArray[np.intp]:
        """
        Get indices of points that lie on the domain boundary.

        This is a convenience method that wraps `is_on_boundary()` and returns
        indices instead of a boolean array. Simplifies solver BC detection workflow.

        Args:
            points: Array of shape (n, d) - points to check
            tolerance: Distance tolerance for boundary detection

        Returns:
            Array of indices (1D integer array) of boundary points

        Example:
            >>> geometry = TensorGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx=50, Ny=50)
            >>> collocation_points = geometry.get_collocation_points()
            >>> boundary_indices = geometry.get_boundary_indices(collocation_points)
            >>> boundary_points = collocation_points[boundary_indices]

        Added:
            v0.16.17 for Issue #545 (Unified BC workflow)
        """
        on_boundary = self.is_on_boundary(points, tolerance)
        return np.where(on_boundary)[0]

    def get_boundary_info(
        self,
        points: NDArray[np.floating],
        tolerance: float = 1e-10,
    ) -> tuple[NDArray[np.intp], NDArray[np.floating]]:
        """
        Get boundary indices and outward normals in one call.

        Convenience method combining `get_boundary_indices()` and
        `get_boundary_normal()` for common solver workflow.

        Args:
            points: Array of shape (n, d) - points to check
            tolerance: Distance tolerance for boundary detection

        Returns:
            Tuple of (boundary_indices, normals):
            - boundary_indices: Array of shape (m,) - indices of boundary points
            - normals: Array of shape (m, d) - outward unit normals at boundary points

        Example:
            >>> geometry = TensorGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx=50, Ny=50)
            >>> collocation_points = geometry.get_collocation_points()
            >>> boundary_indices, normals = geometry.get_boundary_info(collocation_points)
            >>>
            >>> # Apply boundary reflection (particle solver)
            >>> for idx, normal in zip(boundary_indices, normals):
            ...     velocities[idx] = reflect_velocity(velocities[idx], normal)

        Added:
            v0.16.17 for Issue #545 (Unified BC workflow)
        """
        boundary_indices = self.get_boundary_indices(points, tolerance)

        if len(boundary_indices) == 0:
            # No boundary points - return empty arrays
            return boundary_indices, np.array([], dtype=np.float64).reshape(0, self.dimension)

        # Get normals for boundary points
        boundary_points = points[boundary_indices]
        normals = self.get_boundary_normal(boundary_points)

        return boundary_indices, normals


# =============================================================================
# Backward Compatibility Alias
# =============================================================================

# BoundaryAwareProtocol is now merged into GeometryProtocol
# Every domain has a boundary - boundary methods are mandatory
BoundaryAwareProtocol = GeometryProtocol  # Deprecated alias


# =============================================================================
# Boundary Type Enumeration
# =============================================================================


# =============================================================================
# Adaptive Geometry Protocol
# =============================================================================


@runtime_checkable
class AdaptiveGeometry(Protocol):
    """
    Protocol for geometries supporting runtime mesh adaptation (AMR).

    This protocol defines the interface for adaptive mesh refinement.
    Future implementations will wrap external libraries such as:
    - pyAMReX: Block-structured AMR, GPU support
    - Clawpack/AMRClaw: Hyperbolic PDEs
    - pyAMG: Mesh adaptation for complex geometries

    Status: NOT IMPLEMENTED - This is a placeholder for future development.

    Use Cases:
        - Adaptive mesh refinement for error-driven refinement
        - Multi-resolution simulations
        - Local refinement near singularities or boundaries

    Examples:
        >>> from mfgarchon.geometry.protocol import is_adaptive
        >>> if is_adaptive(geometry):
        ...     geometry.adapt(solution)
    """

    def refine(self, criteria: object) -> int:
        """
        Refine cells/elements meeting refinement criteria.

        Args:
            criteria: Refinement criteria (error threshold, region, etc.)
                     Specific type depends on implementation.

        Returns:
            Number of cells/elements refined.

        Notes:
            - For CartesianGrid: subdivides cells (interval bisection, quadtree split)
            - For UnstructuredMesh: refines elements (red-green, bisection)
        """
        ...

    def coarsen(self, criteria: object) -> int:
        """
        Coarsen cells/elements meeting coarsening criteria.

        Args:
            criteria: Coarsening criteria (error threshold, region, etc.)
                     Specific type depends on implementation.

        Returns:
            Number of cells/elements coarsened.

        Notes:
            - May not be supported by all adaptive geometries
            - Returns 0 if coarsening not implemented
        """
        ...

    def adapt(self, solution_data: dict[str, object]) -> dict[str, int]:
        """
        Perform full adaptation cycle (refine + coarsen).

        Args:
            solution_data: Dictionary with solution arrays for error estimation.
                          Typically contains 'u' (value function), 'm' (density).

        Returns:
            Dictionary with adaptation statistics:
            {
                'refined': int,   # Number of cells refined
                'coarsened': int, # Number of cells coarsened
                'total_cells': int,  # Final cell count
            }

        Notes:
            This is the main entry point for adaptive solvers.
        """
        ...

    @property
    def max_refinement_level(self) -> int:
        """
        Maximum refinement level in the current mesh.

        Returns:
            Maximum level (0 = coarsest, higher = finer).
        """
        ...

    @property
    def num_leaf_cells(self) -> int:
        """
        Number of active (leaf) cells/elements.

        Returns:
            Count of cells at the finest local resolution (not subdivided).

        Notes:
            - For tree-based: leaf nodes only
            - For element-based: active elements only
        """
        ...


class BoundaryType(Enum):
    """Type of boundary geometry."""

    POINT = "point"  # 0D boundary (1D domain endpoints)
    EDGE = "edge"  # 1D boundary (2D domain edges)
    FACE = "face"  # 2D boundary (3D domain faces)
    IMPLICIT = "implicit"  # SDF-defined boundary (any dimension)
    NODE = "node"  # Graph node boundary


def is_adaptive(geometry: object) -> bool:
    """
    Check if a geometry supports adaptive mesh refinement.

    Args:
        geometry: Object to check

    Returns:
        True if object implements AdaptiveGeometry protocol, False otherwise

    Examples:
        >>> from mfgarchon.geometry import TensorProductGrid, is_adaptive
        >>> grid = TensorProductGrid(bounds=[(0, 1)], Nx_points=[11])
        >>> is_adaptive(grid)  # Regular grids are not adaptive
        False
    """
    return isinstance(geometry, AdaptiveGeometry)


def is_boundary_aware(geometry: object) -> bool:
    """
    Check if an object implements GeometryProtocol (which includes boundary methods).

    Since boundary methods are now mandatory in GeometryProtocol, this is
    equivalent to is_geometry_compatible().

    Args:
        geometry: Object to check

    Returns:
        True if object implements GeometryProtocol, False otherwise

    Examples:
        >>> from mfgarchon.geometry import TensorProductGrid, Hyperrectangle
        >>> grid = TensorProductGrid(bounds=[(0, 1)], Nx_points=[11])
        >>> is_boundary_aware(grid)
        True
        >>> is_boundary_aware(Hyperrectangle([[0, 1], [0, 1]]))
        True
    """
    return isinstance(geometry, GeometryProtocol)


def validate_boundary_aware(geometry: object) -> None:
    """
    Validate that an object implements GeometryProtocol (including boundary methods).

    Since boundary methods are now mandatory, this validates the full protocol.

    Args:
        geometry: Object to validate

    Raises:
        TypeError: If geometry does not implement the protocol
    """
    if not is_boundary_aware(geometry):
        raise TypeError(
            f"Object of type {type(geometry).__name__} does not implement "
            f"GeometryProtocol. Required methods include: dimension, get_bounds, "
            f"is_on_boundary, get_boundary_normal, project_to_boundary, "
            f"project_to_interior, get_boundary_regions"
        )


def detect_geometry_type(geometry: object) -> GeometryType:
    """
    Detect the type of a geometry object.

    Uses runtime type checking to determine what kind of geometry
    object has been provided. Checks for explicit geometry_type attribute
    first, then falls back to class name inspection.

    Args:
        geometry: Geometry object to classify

    Returns:
        GeometryType enum value

    Raises:
        ValueError: If geometry type cannot be determined

    Examples:
        >>> from mfgarchon.geometry import TensorProductGrid
        >>> grid = TensorProductGrid(bounds=[(0, 1)], Nx_points=[11])
        >>> detect_geometry_type(grid)
        <GeometryType.CARTESIAN_GRID: 'cartesian_grid'>

        >>> from mfgarchon.geometry import NetworkGeometry
        >>> network = NetworkGeometry(topology="scale_free", n_nodes=100)
        >>> detect_geometry_type(network)
        <GeometryType.NETWORK: 'network'>
    """
    # Issue #543: Use try/except instead of hasattr() for optional attribute
    # Check for explicit geometry_type attribute
    try:
        geom_type = geometry.geometry_type
        if isinstance(geom_type, GeometryType):
            return geom_type
        elif isinstance(geom_type, str):
            return GeometryType(geom_type)
    except AttributeError:
        pass

    # Fall back to class name inspection
    class_name = type(geometry).__name__.lower()

    if "network" in class_name or "graph" in class_name:
        return GeometryType.NETWORK
    elif "maze" in class_name:
        return GeometryType.MAZE
    elif any(
        x in class_name
        for x in [
            "domain2d",
            "domain_2d",
            "domain3d",
            "domain_3d",
            "mesh2d",
            "mesh_2d",
            "mesh3d",
            "mesh_3d",
            "unstructured",
        ]
    ):
        return GeometryType.UNSTRUCTURED_MESH
    elif "implicit" in class_name or "levelset" in class_name or "sdf" in class_name:
        return GeometryType.IMPLICIT
    elif any(x in class_name for x in ["domain1d", "domain_1d", "cartesian", "grid"]):
        return GeometryType.CARTESIAN_GRID
    else:
        # Default to CUSTOM for unknown types
        return GeometryType.CUSTOM


def is_geometry_compatible(geometry: object) -> bool:
    """
    Check if an object satisfies the GeometryProtocol.

    Args:
        geometry: Object to check

    Returns:
        True if object implements GeometryProtocol, False otherwise

    Examples:
        >>> from mfgarchon.geometry import TensorProductGrid
        >>> grid = TensorProductGrid(bounds=[(0, 1)], Nx_points=[11])
        >>> is_geometry_compatible(grid)
        True

        >>> is_geometry_compatible("not a geometry")
        False
    """
    return isinstance(geometry, GeometryProtocol)


def validate_geometry(geometry: object) -> None:
    """
    Validate that an object satisfies GeometryProtocol.

    Args:
        geometry: Object to validate

    Raises:
        TypeError: If geometry does not satisfy protocol
        ValueError: If geometry has invalid properties

    Examples:
        >>> from mfgarchon.geometry import TensorProductGrid
        >>> grid = TensorProductGrid(bounds=[(0, 1)], Nx_points=[11])
        >>> validate_geometry(grid)  # No error

        >>> validate_geometry("invalid")
        Traceback (most recent call last):
            ...
        TypeError: Object does not satisfy GeometryProtocol
    """
    if not is_geometry_compatible(geometry):
        raise TypeError(
            f"Object of type {type(geometry).__name__} does not satisfy GeometryProtocol. "
            f"Required attributes: dimension, geometry_type, num_spatial_points, get_spatial_grid()"
        )

    # Validate properties
    if not isinstance(geometry.dimension, int) or geometry.dimension < 1:
        raise ValueError(f"geometry.dimension must be a positive integer, got {geometry.dimension}")

    if not isinstance(geometry.num_spatial_points, int) or geometry.num_spatial_points < 1:
        raise ValueError(f"geometry.num_spatial_points must be a positive integer, got {geometry.num_spatial_points}")

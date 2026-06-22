"""
Base geometry classes for MFGarchon mesh generation pipeline.

This module defines the abstract base classes and core data structures
for the Gmsh → Meshio → PyVista mesh generation system.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MeshVisualizationMode(Enum):
    """
    Mesh visualization display modes for PyVista rendering.

    Modes control edge display and quality metric coloring:
        SURFACE: Show only surface (no edges, no quality coloring)
        WITH_EDGES: Show surface with edges (default)
        QUALITY: Show quality-colored surface (no edges)
        QUALITY_WITH_EDGES: Show quality-colored surface with edges

    Examples:
        >>> mesh.visualize_mesh(mode=MeshVisualizationMode.WITH_EDGES)
        >>> mesh.visualize_mesh(mode="quality")  # String shorthand
    """

    SURFACE = auto()  # show_edges=False, show_quality=False
    WITH_EDGES = auto()  # show_edges=True, show_quality=False (default)
    QUALITY = auto()  # show_edges=False, show_quality=True
    QUALITY_WITH_EDGES = auto()  # show_edges=True, show_quality=True


@dataclass
class MeshData:
    """
    Universal mesh data container compatible with Gmsh → Meshio → PyVista pipeline.

    This class serves as the central data structure for mesh information,
    providing seamless conversion between different mesh formats.

    Supports arbitrary dimensions, though mesh generation tools (Gmsh) are
    limited to d≤3. For d>3, use point clouds or implicit domains.

    Attributes:
        vertices: (N_vertices, dim) coordinates [any dimension]
        elements: Element connectivity (varies by element type)
        element_type: Type of elements ("triangle", "tetrahedron", "point", etc.)
        boundary_tags: Boundary region identifiers
        element_tags: Element region identifiers
        boundary_faces: Boundary face connectivity (2D: edges, 3D: faces, d>3: experimental)
        dimension: Spatial dimension (any positive integer, though d>3 is experimental)
        quality_metrics: Mesh quality analysis data
        element_volumes: Element measures (areas, volumes, hypervolumes)
        metadata: Additional mesh information and parameters
    """

    vertices: NDArray[np.floating]  # (N_vertices, dim) coordinates
    elements: NDArray[np.integer]  # Element connectivity
    element_type: str  # "triangle", "tetrahedron", etc.
    boundary_tags: NDArray[np.integer]  # Boundary region IDs
    element_tags: NDArray[np.integer]  # Element region IDs
    boundary_faces: NDArray[np.integer]  # Boundary connectivity
    dimension: int  # 2 or 3
    quality_metrics: dict[str, Any] | None = None
    element_volumes: NDArray[np.floating] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        """Initialize derived quantities and validate mesh data."""
        if self.quality_metrics is None:
            self.quality_metrics = {}
        if self.metadata is None:
            self.metadata = {}

        # Validate dimensions
        if self.dimension < 1:
            raise ValueError(f"Dimension must be positive, got {self.dimension}")

        # Warn about experimental status for high dimensions
        if self.dimension > 3:
            import warnings

            warnings.warn(
                f"MeshData with dimension={self.dimension} is experimental. "
                f"Mesh generation tools (Gmsh) do not support d>3. "
                f"Consider using point clouds or implicit domains.",
                category=UserWarning,
                stacklevel=2,
            )

        if self.vertices.shape[1] != self.dimension:
            raise ValueError(
                f"Vertex coordinates dimension {self.vertices.shape[1]} "
                f"doesn't match specified dimension {self.dimension}"
            )

    @property
    def num_vertices(self) -> int:
        """Number of vertices in the mesh."""
        return self.vertices.shape[0]

    @property
    def num_elements(self) -> int:
        """Number of elements in the mesh."""
        return self.elements.shape[0]

    @property
    def bounds(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Bounding box of the mesh: (min_coords, max_coords)."""
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)

    def compute_element_volumes(self) -> NDArray[np.floating]:
        """Compute element areas (2D) or volumes (3D)."""
        if self.element_volumes is not None:
            return self.element_volumes

        if self.element_type == "triangle" and self.dimension == 2:
            self.element_volumes = self._compute_triangle_areas()
        elif self.element_type == "tetrahedron" and self.dimension == 3:
            self.element_volumes = self._compute_tetrahedron_volumes()
        else:
            raise NotImplementedError(
                f"Volume computation not implemented for {self.element_type} elements in {self.dimension}D"
            )

        return self.element_volumes

    def _compute_triangle_areas(self) -> NDArray[np.floating]:
        """Compute areas of triangular elements."""
        areas = np.zeros(self.num_elements)
        for i, element in enumerate(self.elements):
            v0, v1, v2 = self.vertices[element]
            # 2D triangle area = 0.5 * |x_a*y_b - y_a*x_b|. Only 2D triangles reach here
            # (see compute_element_volumes' dispatch), so compute the scalar cross explicitly:
            # numpy 2.x removed np.cross for 2D inputs.
            a = v1 - v0
            b = v2 - v0
            areas[i] = 0.5 * abs(float(a[0] * b[1] - a[1] * b[0]))
        return areas

    def _compute_tetrahedron_volumes(self) -> NDArray[np.floating]:
        """Compute volumes of tetrahedral elements."""
        volumes = np.zeros(self.num_elements)
        for i, element in enumerate(self.elements):
            v0, v1, v2, v3 = self.vertices[element]
            # Volume = (1/6) * |det(v1-v0, v2-v0, v3-v0)|
            matrix = np.array([v1 - v0, v2 - v0, v3 - v0]).T
            volumes[i] = abs(np.linalg.det(matrix)) / 6.0
        return volumes

    def to_meshio(self):
        """Convert to meshio format for I/O operations.

        Note: boundary_tags are not included in the meshio export as they
        represent boundary elements (edges/faces) rather than vertices or cells.
        Only element_tags (region identifiers) are exported as cell_data.
        """
        try:
            import meshio
        except ImportError as err:
            raise ImportError("meshio is required for mesh I/O operations") from err

        cells = [(self.element_type, self.elements)]

        # meshio cell_data format: {"data_name": [array_for_block1, array_for_block2, ...]}
        cell_data = {}
        if len(self.element_tags) > 0:
            cell_data["region"] = [self.element_tags]

        # Note: boundary_tags are not included because they don't map to
        # meshio's data model (not vertex data, not bulk element data)

        return meshio.Mesh(
            points=self.vertices,
            cells=cells,
            cell_data=cell_data if cell_data else None,
        )

    def to_pyvista(self):
        """Convert to PyVista format for visualization."""
        try:
            import pyvista as pv
        except ImportError as err:
            raise ImportError("pyvista is required for mesh visualization") from err

        if self.element_type == "triangle":
            # Create PyVista PolyData for 2D triangular mesh
            faces = np.hstack(
                [
                    np.full((self.num_elements, 1), 3),  # 3 vertices per triangle
                    self.elements,
                ]
            )
            mesh = pv.PolyData(self.vertices, faces)

        elif self.element_type == "tetrahedron":
            # Create PyVista UnstructuredGrid for 3D tetrahedral mesh
            cells = np.hstack(
                [
                    np.full((self.num_elements, 1), 4),  # 4 vertices per tetrahedron
                    self.elements,
                ]
            )
            cell_types = np.full(self.num_elements, pv.CellType.TETRA)
            mesh = pv.UnstructuredGrid(cells, cell_types, self.vertices)

        else:
            raise NotImplementedError(f"PyVista conversion not implemented for {self.element_type} elements")

        # Add data arrays
        if len(self.boundary_tags) > 0:
            mesh.point_data["boundary_tags"] = self.boundary_tags
        if len(self.element_tags) > 0:
            mesh.cell_data["element_tags"] = self.element_tags

        return mesh


# Note: This file now contains only mesh data structures.
# The unified geometry hierarchy is in base.py (Geometry ABC and subclasses).
# BaseGeometry was removed as part of Issue #245 consolidation (completed in Issue #320).

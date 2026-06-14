"""
1D unstructured mesh for FEM/FVM methods in MFGarchon.

This module implements 1D line element meshes for finite element methods,
supporting variable spacing, multiple regions, and adaptive refinement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.geometry.base import UnstructuredMesh
from mfgarchon.geometry.meshes.mesh_data import MeshData

if TYPE_CHECKING:
    from collections.abc import Callable


class Mesh1D(UnstructuredMesh):
    """1D unstructured mesh with line elements for FEM/FVM methods."""

    def __init__(
        self,
        bounds: tuple[float, float] = (0.0, 1.0),
        num_elements: int = 10,
        mesh_spacing: str = "uniform",
        refinement_function: Callable[[float], float] | None = None,
        regions: list[dict] | None = None,
    ):
        """
        Initialize 1D mesh.

        Args:
            bounds: (xmin, xmax) domain boundaries
            num_elements: Number of line elements
            mesh_spacing: "uniform", "geometric", or "adaptive"
            refinement_function: Optional function h(x) controlling local mesh size
            regions: List of region specifications [{"start": x1, "end": x2, "tag": 1}, ...]
        """
        super().__init__(dimension=1)

        self.xmin, self.xmax = bounds
        self.length = self.xmax - self.xmin
        self.num_elements = num_elements
        self.mesh_spacing = mesh_spacing
        self.refinement_function = refinement_function
        self.regions = regions or []

        if self.xmax <= self.xmin:
            raise ValueError("xmax must be greater than xmin")

    def create_gmsh_geometry(self):
        """
        Create Gmsh geometry (not used for simple 1D meshes).

        Note: Mesh1D uses direct mesh generation instead of Gmsh.
        This method is provided for BaseGeometry compatibility.
        """
        raise NotImplementedError("Mesh1D uses direct generation, not Gmsh")

    def generate_mesh(self) -> MeshData:
        """
        Generate 1D line element mesh.

        Returns:
            MeshData with vertices and line elements
        """
        if self.mesh_spacing == "uniform":
            vertices = self._generate_uniform_mesh()
        elif self.mesh_spacing == "geometric":
            vertices = self._generate_geometric_mesh()
        elif self.mesh_spacing == "adaptive":
            vertices = self._generate_adaptive_mesh()
        else:
            raise ValueError(f"Unknown mesh spacing: {self.mesh_spacing}")

        # Create line elements (connectivity)
        num_vertices = len(vertices)
        elements = np.array([[i, i + 1] for i in range(num_vertices - 1)])

        # Assign element tags based on regions
        element_tags = self._assign_element_tags(vertices, elements)

        # Boundary tags (0 = left, 1 = right)
        boundary_tags = np.array([0, 1])  # Left and right boundaries

        # Create MeshData
        mesh_data = MeshData(
            vertices=vertices.reshape(-1, 1),  # Shape (N, 1) for 1D
            elements=elements,
            element_type="line",
            boundary_tags=boundary_tags,
            element_tags=element_tags,
            boundary_faces=np.array([[0], [num_vertices - 1]]),  # Boundary nodes
            dimension=1,
        )

        # Compute element lengths
        mesh_data.element_volumes = self._compute_element_lengths(vertices, elements)

        self.mesh_data = mesh_data
        return mesh_data

    def _generate_uniform_mesh(self) -> np.ndarray:
        """Generate uniformly spaced vertices."""
        return np.linspace(self.xmin, self.xmax, self.num_elements + 1)

    def _generate_geometric_mesh(self, ratio: float = 1.1) -> np.ndarray:
        """Generate geometrically spaced vertices (refined near boundaries)."""
        # Geometric progression from center to boundaries
        n = self.num_elements + 1
        center = (self.xmin + self.xmax) / 2.0
        half_length = self.length / 2.0

        # Create geometric spacing from 0 to 1
        s = np.array([ratio**i for i in range(n // 2 + 1)])
        s = s / s[-1]  # Normalize to [0, 1]

        # Mirror and map to domain
        vertices_left = center - half_length * s[::-1]
        vertices_right = center + half_length * s[1:]

        return np.concatenate([vertices_left, vertices_right])

    def _generate_adaptive_mesh(self) -> np.ndarray:
        """Generate adaptively refined mesh using refinement function."""
        if self.refinement_function is None:
            # Default: refine near boundaries
            def default_refinement(x: float) -> float:
                """Refinement function: smaller h near boundaries."""
                dist_to_boundary = min(abs(x - self.xmin), abs(x - self.xmax))
                return 0.1 + 0.9 * (dist_to_boundary / (self.length / 2.0))

            self.refinement_function = default_refinement

        # Start with uniform mesh
        vertices = [self.xmin]
        x = self.xmin

        while x < self.xmax:
            # Determine local mesh size
            h = self.refinement_function(x) * (self.length / self.num_elements)
            x_new = min(x + h, self.xmax)
            vertices.append(x_new)
            x = x_new

        return np.array(vertices)

    def _assign_element_tags(self, vertices: np.ndarray, elements: np.ndarray) -> np.ndarray:
        """Assign element tags based on regions."""
        element_tags = np.ones(len(elements), dtype=int)

        if not self.regions:
            return element_tags

        # Compute element centers
        element_centers = (vertices[elements[:, 0]] + vertices[elements[:, 1]]) / 2.0

        # Assign tags based on which region each element center belongs to
        for region in self.regions:
            start = region["start"]
            end = region["end"]
            tag = region["tag"]

            mask = (element_centers >= start) & (element_centers <= end)
            element_tags[mask] = tag

        return element_tags

    def _compute_element_lengths(self, vertices: np.ndarray, elements: np.ndarray) -> np.ndarray:
        """Compute lengths of line elements."""
        return np.abs(vertices[elements[:, 1]] - vertices[elements[:, 0]])

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get 1D domain bounding box."""
        return np.array([self.xmin]), np.array([self.xmax])

    def set_mesh_parameters(self, num_elements: int | None = None, spacing: str | None = None):
        """Set mesh generation parameters."""
        if num_elements is not None:
            self.num_elements = num_elements
        if spacing is not None:
            self.mesh_spacing = spacing

    def export_mesh(self, file_format: str = "", filename: str = "") -> None:
        """Export mesh in specified format.

        Args:
            file_format: Meshio file format string (e.g. "vtk", "gmsh")
            filename: Output file path
        """
        if self.mesh_data is None:
            self.generate_mesh()

        if self.mesh_data is None:
            raise RuntimeError("Failed to generate mesh data")

        try:
            import meshio
        except ImportError:
            raise ImportError("meshio required for mesh export") from None

        # Convert to meshio format
        cells = [("line", self.mesh_data.elements)]

        mesh = meshio.Mesh(
            points=self.mesh_data.vertices,
            cells=cells,
            cell_data={"physical": [self.mesh_data.element_tags]},
        )

        # Write mesh
        if file_format:
            mesh.write(filename, file_format=file_format)
        else:
            mesh.write(filename)

    def __str__(self) -> str:
        """String representation."""
        return f"Mesh1D([{self.xmin}, {self.xmax}], {self.num_elements} elements, {self.mesh_spacing})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"Mesh1D(bounds=({self.xmin}, {self.xmax}), num_elements={self.num_elements}, spacing={self.mesh_spacing})"
        )

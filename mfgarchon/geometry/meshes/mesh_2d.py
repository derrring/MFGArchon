"""
2D unstructured mesh for FEM/FVM methods in MFGarchon.

This module implements 2D triangular meshes using the Gmsh → Meshio → PyVista pipeline,
supporting rectangles, circles, complex shapes, holes, and CAD import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.geometry.meshes.mesh_base import _MeshGeneratorBase
from mfgarchon.geometry.meshes.mesh_data import MeshData

if TYPE_CHECKING:
    from collections.abc import Callable


class Mesh2D(_MeshGeneratorBase):
    """2D unstructured triangular mesh for FEM/FVM methods using Gmsh pipeline."""

    def __init__(
        self,
        domain_type: str = "rectangle",
        bounds: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        holes: list[dict] | None = None,
        mesh_size: float = 0.1,
        **kwargs,
    ):
        """
        Initialize 2D domain.

        Args:
            domain_type: rectangle, "circle", "polygon", "custom", "cad_import"
            bounds: (xmin, xmax, ymin, ymax) for rectangular domains
            holes: List of hole specifications [{"type": circle, "center": (x,y), "radius": r}, ...]
            mesh_size: Target mesh element size
            **kwargs: Additional domain-specific parameters
        """
        super().__init__(
            dimension=2,
            domain_type=domain_type,
            bounds_tuple=bounds,
            holes=holes,
            mesh_size=mesh_size,
            **kwargs,
        )

    def _setup_domain_parameters(self):
        """Setup parameters based on domain type."""
        if self.domain_type == "rectangle":
            self.xmin, self.xmax, self.ymin, self.ymax = self._bounds_tuple

        elif self.domain_type == "circle":
            self.center = self.kwargs.get("center", (0.5, 0.5))
            self.radius = self.kwargs.get("radius", 0.4)

        elif self.domain_type == "polygon":
            self.vertices = self.kwargs.get("vertices", [])
            if len(self.vertices) < 3:
                raise ValueError("Polygon must have at least 3 vertices")

        elif self.domain_type == "custom":
            self.geometry_func = self.kwargs.get("geometry_func")
            if self.geometry_func is None:
                raise ValueError("Custom domain requires 'geometry_func' parameter")

        elif self.domain_type == "cad_import":
            self.cad_file = self.kwargs.get("cad_file")
            if self.cad_file is None:
                raise ValueError("CAD import requires 'cad_file' parameter")

    def _get_domain_type_dispatch(self) -> dict[str, Callable[[], None]]:
        """Map domain_type to Gmsh geometry creator method."""
        return {
            "rectangle": self._create_rectangle_gmsh,
            "circle": self._create_circle_gmsh,
            "polygon": self._create_polygon_gmsh,
            "custom": self._create_custom_gmsh,
            "cad_import": self._import_cad_gmsh,
        }

    def _get_model_name(self) -> str:
        return "domain_2d"

    def _create_rectangle_gmsh(self):
        """Create rectangular domain in Gmsh."""
        import gmsh

        # Create corner points
        p1 = gmsh.model.geo.addPoint(self.xmin, self.ymin, 0.0)
        p2 = gmsh.model.geo.addPoint(self.xmax, self.ymin, 0.0)
        p3 = gmsh.model.geo.addPoint(self.xmax, self.ymax, 0.0)
        p4 = gmsh.model.geo.addPoint(self.xmin, self.ymax, 0.0)

        # Create edges
        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)

        # Create curve loop and surface
        curve_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        self.main_surface = gmsh.model.geo.addPlaneSurface([curve_loop])

        # Add physical groups for boundary conditions
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [l1], 1)  # Bottom boundary
        gmsh.model.addPhysicalGroup(1, [l2], 2)  # Right boundary
        gmsh.model.addPhysicalGroup(1, [l3], 3)  # Top boundary
        gmsh.model.addPhysicalGroup(1, [l4], 4)  # Left boundary
        gmsh.model.addPhysicalGroup(2, [self.main_surface], 1)  # Domain

    def _create_circle_gmsh(self):
        """Create circular domain in Gmsh."""
        import gmsh

        center_x, center_y = self.center

        # Create center point
        center_point = gmsh.model.geo.addPoint(center_x, center_y, 0.0)

        # Create points on circumference
        p1 = gmsh.model.geo.addPoint(center_x + self.radius, center_y, 0.0)
        p2 = gmsh.model.geo.addPoint(center_x, center_y + self.radius, 0.0)
        p3 = gmsh.model.geo.addPoint(center_x - self.radius, center_y, 0.0)
        p4 = gmsh.model.geo.addPoint(center_x, center_y - self.radius, 0.0)

        # Create circular arcs
        c1 = gmsh.model.geo.addCircleArc(p1, center_point, p2)
        c2 = gmsh.model.geo.addCircleArc(p2, center_point, p3)
        c3 = gmsh.model.geo.addCircleArc(p3, center_point, p4)
        c4 = gmsh.model.geo.addCircleArc(p4, center_point, p1)

        # Create curve loop and surface
        curve_loop = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])
        self.main_surface = gmsh.model.geo.addPlaneSurface([curve_loop])

        # Add physical groups
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [c1, c2, c3, c4], 1)  # Circular boundary
        gmsh.model.addPhysicalGroup(2, [self.main_surface], 1)  # Domain

    def _create_polygon_gmsh(self):
        """Create polygonal domain in Gmsh."""
        import gmsh

        # Create points
        points = []
        for x, y in self.vertices:
            points.append(gmsh.model.geo.addPoint(x, y, 0.0))

        # Create lines
        lines = []
        for i in range(len(points)):
            next_i = (i + 1) % len(points)
            lines.append(gmsh.model.geo.addLine(points[i], points[next_i]))

        # Create curve loop and surface
        curve_loop = gmsh.model.geo.addCurveLoop(lines)
        self.main_surface = gmsh.model.geo.addPlaneSurface([curve_loop])

        # Add physical groups
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, lines, 1)  # Boundary
        gmsh.model.addPhysicalGroup(2, [self.main_surface], 1)  # Domain

    def _create_custom_gmsh(self):
        """Create custom domain using user-provided function."""
        try:
            import gmsh
        except ImportError:
            raise ImportError("gmsh is required for custom geometry creation") from None

        if self.geometry_func is None:
            raise RuntimeError("geometry_func is None")

        # User-provided geometry function should create Gmsh geometry
        # and set self.main_surface
        self.main_surface = self.geometry_func(gmsh)

    def _import_cad_gmsh(self):
        """Import CAD geometry into Gmsh."""
        import gmsh

        # Import CAD file
        gmsh.model.occ.importShapes(self.cad_file)
        gmsh.model.occ.synchronize()

        # Get surfaces (assuming single surface for now)
        surfaces = gmsh.model.getEntities(2)
        if len(surfaces) == 0:
            raise ValueError("No surfaces found in CAD file")
        self.main_surface = surfaces[0][1]

    def _add_holes_gmsh(self):
        """Add holes to the domain using OCC kernel for proper boolean operations."""
        if not self.holes:
            return

        import gmsh

        # Convert existing geo entities to OCC for boolean operations
        # This is necessary because geo kernel doesn't support cut() operation
        try:
            gmsh.model.geo.synchronize()

            # Remove the geo-based main surface
            # We'll recreate everything in OCC kernel
            gmsh.model.geo.remove([(2, self.main_surface)])
            gmsh.model.geo.synchronize()

            # Recreate main domain in OCC kernel
            if self.domain_type == "rectangle":
                xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax
                main_rect = gmsh.model.occ.addRectangle(xmin, ymin, 0.0, xmax - xmin, ymax - ymin)
                main_entities = [(2, main_rect)]

            elif self.domain_type == "circle":
                center_x, center_y = self.center
                main_disk = gmsh.model.occ.addDisk(center_x, center_y, 0.0, self.radius, self.radius)
                main_entities = [(2, main_disk)]

            else:
                # For other types, holes are not supported yet
                import warnings

                warnings.warn(
                    f"Holes not yet supported for domain_type='{self.domain_type}'. "
                    "Only 'rectangle' and 'circle' domains support holes currently.",
                    stacklevel=2,
                )
                return

            # Create hole entities in OCC kernel
            hole_entities = []
            for hole in self.holes:
                if hole["type"] == "circle":
                    center = hole["center"]
                    radius = hole["radius"]
                    hole_disk = gmsh.model.occ.addDisk(center[0], center[1], 0.0, radius, radius)
                    hole_entities.append((2, hole_disk))

                elif hole["type"] == "rectangle":
                    bounds = hole["bounds"]  # (xmin, xmax, ymin, ymax)
                    xmin, xmax, ymin, ymax = bounds
                    hole_rect = gmsh.model.occ.addRectangle(xmin, ymin, 0.0, xmax - xmin, ymax - ymin)
                    hole_entities.append((2, hole_rect))

            # Perform boolean cut: main - holes
            if hole_entities:
                import contextlib

                gmsh.model.occ.synchronize()
                result, _result_map = gmsh.model.occ.cut(main_entities, hole_entities, removeTool=True)
                gmsh.model.occ.synchronize()

                # Update main surface to the result
                if result:
                    self.main_surface = result[0][1]

                    # Add physical groups for the result
                    # Get boundary curves
                    boundary_curves = gmsh.model.getBoundary(result, oriented=False)
                    curve_tags = [abs(c[1]) for c in boundary_curves]
                    if curve_tags:
                        with contextlib.suppress(Exception):
                            # Physical group might already exist, that's okay
                            gmsh.model.addPhysicalGroup(1, curve_tags, 1)  # Boundary

                    with contextlib.suppress(Exception):
                        # Physical group might already exist, that's okay
                        gmsh.model.addPhysicalGroup(2, [self.main_surface], 1)  # Domain

        except Exception as e:
            # If hole creation fails at boolean operation stage, emit warning
            import warnings

            warnings.warn(f"Hole creation failed: {e}. Continuing with domain without holes.", stacklevel=2)

    def generate_mesh(self) -> MeshData:
        """Generate 2D triangular mesh using Gmsh → Meshio pipeline.

        If ``self.mesh_data`` is already populated (e.g. injected from a pre-built mesh via
        ``mfgarchon.alg.numerical.fem.mesh_adapter.skfem_to_meshdata``), it is returned as-is —
        a gmsh-free path for callers that supply their own mesh, and idempotent memoization
        otherwise.

        Note: gmsh generation here emits **triangular (simplex)** elements only. The FEM solve
        path also supports **quadrilateral** meshes (Issue #470), but quad generation is not
        implemented — inject a pre-built quad mesh instead, e.g. ``skfem.MeshQuad.init_tensor``
        → ``skfem_to_meshdata`` (gmsh-free)."""
        if self.mesh_data is not None:
            return self.mesh_data
        try:
            import gmsh
            import meshio  # noqa: F401
        except ImportError:
            raise ImportError("gmsh and meshio are required for mesh generation") from None

        # Create geometry
        self.create_gmsh_geometry()

        # Generate mesh
        gmsh.model.mesh.generate(2)  # 2D mesh

        # Extract mesh data
        _node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        element_types, _element_tags, element_connectivity = gmsh.model.mesh.getElements()

        # Process vertices
        vertices = node_coords.reshape(-1, 3)[:, :2]  # Remove z-coordinate for 2D

        # Process elements (triangles)
        triangles = None
        for i, elem_type in enumerate(element_types):
            if elem_type == 2:  # Triangle element type in Gmsh
                connectivity = element_connectivity[i]
                triangles = connectivity.reshape(-1, 3) - 1  # Convert to 0-based indexing
                break

        if triangles is None:
            raise RuntimeError("No triangular elements found in generated mesh")

        # Get boundary information
        boundary_tags = np.zeros(len(vertices), dtype=int)  # Placeholder
        element_tags_array = np.ones(len(triangles), dtype=int)  # All elements in same region

        # Create boundary faces (edges for 2D)
        boundary_faces = self._extract_boundary_edges(triangles)

        gmsh.finalize()

        # Create MeshData object
        self.mesh_data = MeshData(
            vertices=vertices,
            elements=triangles,
            element_type="triangle",
            boundary_tags=boundary_tags,
            element_tags=element_tags_array,
            boundary_faces=boundary_faces,
            dimension=2,
            metadata={
                "domain_type": self.domain_type,
                "mesh_size": self.mesh_size,
                "bounds": self._bounds_tuple,
            },
        )

        return self.mesh_data

    def _extract_boundary_edges(self, triangles: np.ndarray) -> np.ndarray:
        """Extract boundary edges from triangular mesh."""
        # Create edge dictionary to find boundary edges
        edge_count: dict[tuple[int, int], int] = {}

        # Count each edge
        for triangle in triangles:
            edges = [
                tuple(sorted([triangle[0], triangle[1]])),
                tuple(sorted([triangle[1], triangle[2]])),
                tuple(sorted([triangle[2], triangle[0]])),
            ]

            for edge in edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1

        # Boundary edges appear only once
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

        return np.array(boundary_edges)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get 2D domain bounding box."""
        if self.domain_type == "rectangle":
            min_coords = np.array([self.xmin, self.ymin])
            max_coords = np.array([self.xmax, self.ymax])
            return min_coords, max_coords

        elif self.domain_type == "circle":
            center_x, center_y = self.center
            min_coords = np.array([center_x - self.radius, center_y - self.radius])
            max_coords = np.array([center_x + self.radius, center_y + self.radius])
            return min_coords, max_coords

        elif self.domain_type == "polygon":
            vertices_array = np.array(self.vertices)
            min_coords = np.min(vertices_array, axis=0)
            max_coords = np.max(vertices_array, axis=0)
            return min_coords, max_coords

        else:
            # For custom domains, generate mesh first to get bounds
            if self.mesh_data is None:
                self.generate_mesh()
            if self.mesh_data is None:
                raise RuntimeError("Failed to generate mesh data")
            return self.mesh_data.bounds

    def export_mesh(self, file_format: str, filename: str) -> None:
        """Export mesh in specified format using meshio."""
        if self.mesh_data is None:
            self.generate_mesh()

        if self.mesh_data is None:
            raise RuntimeError("Failed to generate mesh data")

        meshio_mesh = self.mesh_data.to_meshio()
        meshio_mesh.write(filename, file_format=file_format)

    def refine_mesh(
        self,
        refinement_factor: float = 0.5,
        adaptive: bool = False,
        indicator_function: Callable | None = None,
    ) -> MeshData:
        """Refine mesh uniformly or adaptively."""
        if adaptive and indicator_function is None:
            raise ValueError("Adaptive refinement requires indicator_function")

        if adaptive:
            # Adaptive refinement based on indicator function
            if indicator_function is None:
                raise RuntimeError("indicator_function is None")
            return self._refine_adaptive(indicator_function)
        else:
            # Uniform refinement
            self.mesh_size *= refinement_factor
            return self.generate_mesh()

    def _refine_adaptive(self, indicator_function: Callable) -> MeshData:
        """Adaptive mesh refinement using indicator function."""
        # This is a placeholder for adaptive refinement
        # Full implementation would involve error estimation and local refinement
        raise NotImplementedError("Adaptive refinement not yet implemented")

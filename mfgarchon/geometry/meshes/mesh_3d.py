"""
3D unstructured mesh for FEM/FVM methods in MFGarchon.

This module implements 3D tetrahedral meshes using the Gmsh → Meshio → PyVista pipeline,
supporting boxes, spheres, cylinders, complex shapes, and CAD import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.geometry.meshes.mesh_base import _MeshGeneratorBase
from mfgarchon.geometry.meshes.mesh_data import MeshData
from mfgarchon.utils.deprecation import deprecated_parameter
from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


class Mesh3D(_MeshGeneratorBase):
    """3D unstructured tetrahedral mesh for FEM/FVM methods using Gmsh pipeline."""

    def __init__(
        self,
        domain_type: str = "box",
        bounds: tuple[float, float, float, float, float, float] = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        holes: list[dict] | None = None,
        mesh_size: float = 0.1,
        **kwargs,
    ):
        """
        Initialize 3D domain.

        Args:
            domain_type: box, "sphere", "cylinder", "polyhedron", "custom", "cad_import"
            bounds: (xmin, xmax, ymin, ymax, zmin, zmax) for box domains
            holes: List of hole specifications [{"type": sphere, "center": (x,y,z), "radius": r}, ...]
            mesh_size: Target mesh element size
            **kwargs: Additional domain-specific parameters
        """
        super().__init__(
            dimension=3,
            domain_type=domain_type,
            bounds_tuple=bounds,
            holes=holes,
            mesh_size=mesh_size,
            **kwargs,
        )

    def _setup_domain_parameters(self):
        """Setup parameters based on domain type."""
        if self.domain_type == "box":
            self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = self._bounds_tuple

        elif self.domain_type == "sphere":
            self.center = self.kwargs.get("center", (0.5, 0.5, 0.5))
            self.radius = self.kwargs.get("radius", 0.4)

        elif self.domain_type == "cylinder":
            self.center = self.kwargs.get("center", (0.5, 0.5))  # (x, y) center
            self.radius = self.kwargs.get("radius", 0.4)
            self.height = self.kwargs.get("height", 1.0)
            self.axis = self.kwargs.get("axis", "z")  # "x", "y", or "z"

        elif self.domain_type == "polyhedron":
            self.vertices = self.kwargs.get("vertices", [])
            self.faces = self.kwargs.get("faces", [])
            if len(self.vertices) < 4:
                raise ValueError("Polyhedron must have at least 4 vertices")
            if len(self.faces) < 4:
                raise ValueError("Polyhedron must have at least 4 faces")

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
            "box": self._create_box_gmsh,
            "sphere": self._create_sphere_gmsh,
            "cylinder": self._create_cylinder_gmsh,
            "polyhedron": self._create_polyhedron_gmsh,
            "custom": self._create_custom_gmsh,
            "cad_import": self._import_cad_gmsh,
        }

    def _get_model_name(self) -> str:
        return "domain_3d"

    def _post_gmsh_options(self) -> None:
        """Set 3D-specific Gmsh options."""
        import gmsh

        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay for 3D

    def _create_box_gmsh(self):
        """Create box geometry in Gmsh."""
        import gmsh

        # Create box
        box_tag = gmsh.model.occ.addBox(
            self.xmin, self.ymin, self.zmin, self.xmax - self.xmin, self.ymax - self.ymin, self.zmax - self.zmin
        )

        gmsh.model.occ.synchronize()

        # Tag physical volumes and surfaces
        gmsh.model.addPhysicalGroup(3, [box_tag], 1)  # Volume
        gmsh.model.setPhysicalName(3, 1, "domain")

        # Tag boundary faces by geometric location
        boundary_tags = gmsh.model.getBoundary([(3, box_tag)], oriented=False)
        surface_tags = [abs(tag[1]) for tag in boundary_tags]

        # Identify surfaces by their mass center coordinates
        # 1=xmin (left), 2=xmax (right), 3=ymin (front), 4=ymax (back), 5=zmin (bottom), 6=zmax (top)
        surface_map = {}

        # Compute tolerance based on domain size (relative tolerance)
        domain_size = max(self.xmax - self.xmin, self.ymax - self.ymin, self.zmax - self.zmin)
        tol = domain_size * 1e-6  # Relative tolerance

        for surf_tag in surface_tags:
            # Get surface center of mass
            mass = gmsh.model.occ.getCenterOfMass(2, surf_tag)
            x, y, z = mass[0], mass[1], mass[2]

            # Identify surface by which coordinate is at boundary
            if abs(x - self.xmin) < tol:
                surface_map[1] = surf_tag  # xmin (left)
            elif abs(x - self.xmax) < tol:
                surface_map[2] = surf_tag  # xmax (right)
            elif abs(y - self.ymin) < tol:
                surface_map[3] = surf_tag  # ymin (front)
            elif abs(y - self.ymax) < tol:
                surface_map[4] = surf_tag  # ymax (back)
            elif abs(z - self.zmin) < tol:
                surface_map[5] = surf_tag  # zmin (bottom)
            elif abs(z - self.zmax) < tol:
                surface_map[6] = surf_tag  # zmax (top)

        # Assign physical groups with correct IDs
        boundary_names = ["left", "right", "front", "back", "bottom", "top"]
        for boundary_id in range(1, 7):
            if boundary_id in surface_map:
                try:
                    gmsh.model.addPhysicalGroup(2, [surface_map[boundary_id]], boundary_id)
                    gmsh.model.setPhysicalName(2, boundary_id, boundary_names[boundary_id - 1])
                except Exception as e:
                    # Surface might not exist or physical group might already be defined
                    logger.warning(
                        f"Failed to tag boundary surface '{boundary_names[boundary_id - 1]}' (id={boundary_id}): {e}"
                    )

        # If not all surfaces were mapped, it might be due to tolerance issues
        # In this case, just assign all surfaces to a single physical group
        if len(surface_map) < 6:
            # Fallback: assign all surfaces to physical group 1
            try:
                gmsh.model.addPhysicalGroup(2, surface_tags, 1)
                gmsh.model.setPhysicalName(2, 1, "boundary")
            except Exception as e:
                logger.warning(f"Failed to assign fallback boundary group: {e}")

    def _create_sphere_gmsh(self):
        """Create sphere geometry in Gmsh."""
        import gmsh

        # Create sphere
        sphere_tag = gmsh.model.occ.addSphere(self.center[0], self.center[1], self.center[2], self.radius)

        gmsh.model.occ.synchronize()

        # Tag physical volume
        gmsh.model.addPhysicalGroup(3, [sphere_tag], 1)
        gmsh.model.setPhysicalName(3, 1, "domain")

        # Tag boundary surface
        boundary_tags = gmsh.model.getBoundary([(3, sphere_tag)], oriented=False)
        surface_tags = [abs(tag[1]) for tag in boundary_tags]
        gmsh.model.addPhysicalGroup(2, surface_tags, 1)
        gmsh.model.setPhysicalName(2, 1, "sphere_surface")

    def _create_cylinder_gmsh(self):
        """Create cylinder geometry in Gmsh."""
        import gmsh

        # Create cylinder based on axis
        if self.axis == "z":
            cylinder_tag = gmsh.model.occ.addCylinder(
                self.center[0],
                self.center[1],
                0.0,  # center at z=0
                0.0,
                0.0,
                self.height,  # height along z
                self.radius,
            )
        elif self.axis == "y":
            cylinder_tag = gmsh.model.occ.addCylinder(
                self.center[0],
                0.0,
                self.center[1],  # center at y=0
                0.0,
                self.height,
                0.0,  # height along y
                self.radius,
            )
        elif self.axis == "x":
            cylinder_tag = gmsh.model.occ.addCylinder(
                0.0,
                self.center[0],
                self.center[1],  # center at x=0
                self.height,
                0.0,
                0.0,  # height along x
                self.radius,
            )
        else:
            raise ValueError(f"Invalid cylinder axis: {self.axis}")

        gmsh.model.occ.synchronize()

        # Tag physical volume
        gmsh.model.addPhysicalGroup(3, [cylinder_tag], 1)
        gmsh.model.setPhysicalName(3, 1, "domain")

        # Tag boundary surfaces (curved surface + 2 circular faces)
        boundary_tags = gmsh.model.getBoundary([(3, cylinder_tag)], oriented=False)
        surface_tags = [abs(tag[1]) for tag in boundary_tags]

        for i, surf_tag in enumerate(surface_tags):
            gmsh.model.addPhysicalGroup(2, [surf_tag], i + 1)
            gmsh.model.setPhysicalName(2, i + 1, f"cylinder_surface_{i + 1}")

    def _create_polyhedron_gmsh(self):
        """Create polyhedron geometry in Gmsh."""
        import gmsh

        # Add vertices
        point_tags = []
        for vertex in self.vertices:
            tag = gmsh.model.occ.addPoint(vertex[0], vertex[1], vertex[2])
            point_tags.append(tag)

        # Create faces (assuming triangular faces)
        surface_tags = []
        for face in self.faces:
            if len(face) == 3:  # Triangular face
                # Create lines
                line_tags = []
                for i in range(3):
                    line_tag = gmsh.model.occ.addLine(point_tags[face[i]], point_tags[face[(i + 1) % 3]])
                    line_tags.append(line_tag)

                # Create curve loop and surface
                curve_loop = gmsh.model.occ.addCurveLoop(line_tags)
                surface_tag = gmsh.model.occ.addPlaneSurface([curve_loop])
                surface_tags.append(surface_tag)

        # Create volume from surfaces
        surface_loop = gmsh.model.occ.addSurfaceLoop(surface_tags)
        volume_tag = gmsh.model.occ.addVolume([surface_loop])

        gmsh.model.occ.synchronize()

        # Tag physical groups
        gmsh.model.addPhysicalGroup(3, [volume_tag], 1)
        gmsh.model.setPhysicalName(3, 1, "domain")

        for i, surf_tag in enumerate(surface_tags):
            gmsh.model.addPhysicalGroup(2, [surf_tag], i + 1)
            gmsh.model.setPhysicalName(2, i + 1, f"face_{i + 1}")

    def _create_custom_gmsh(self):
        """Create custom geometry using user-provided function."""
        # Call user-provided geometry function
        # Expected to return Gmsh model with tagged physical groups
        self.geometry_func(self)

    def _import_cad_gmsh(self):
        """Import CAD geometry from file."""
        import gmsh

        gmsh.model.occ.importShapes(self.cad_file)
        gmsh.model.occ.synchronize()

        # Auto-assign physical groups
        volumes = gmsh.model.getEntities(3)
        if volumes:
            volume_tags = [tag[1] for tag in volumes]
            gmsh.model.addPhysicalGroup(3, volume_tags, 1)
            gmsh.model.setPhysicalName(3, 1, "imported_domain")

        surfaces = gmsh.model.getEntities(2)
        if surfaces:
            surface_tags = [tag[1] for tag in surfaces]
            for i, surf_tag in enumerate(surface_tags):
                gmsh.model.addPhysicalGroup(2, [surf_tag], i + 1)
                gmsh.model.setPhysicalName(2, i + 1, f"imported_surface_{i + 1}")

    def _add_holes_gmsh(self):
        """Add holes to the 3D domain."""
        import gmsh

        if not self.holes:
            return

        hole_tags = []
        for hole in self.holes:
            if hole["type"] == "sphere":
                center = hole["center"]
                radius = hole["radius"]
                hole_tag = gmsh.model.occ.addSphere(center[0], center[1], center[2], radius)
                hole_tags.append(hole_tag)

            elif hole["type"] == "box":
                bounds = hole["bounds"]  # (xmin, xmax, ymin, ymax, zmin, zmax)
                hole_tag = gmsh.model.occ.addBox(
                    bounds[0],
                    bounds[2],
                    bounds[4],  # x, y, z min
                    bounds[1] - bounds[0],  # width
                    bounds[3] - bounds[2],  # height
                    bounds[5] - bounds[4],  # depth
                )
                hole_tags.append(hole_tag)

            elif hole["type"] == "cylinder":
                center = hole["center"]  # (x, y) center
                radius = hole["radius"]
                height = hole.get("height", 1.0)
                axis = hole.get("axis", "z")

                if axis == "z":
                    hole_tag = gmsh.model.occ.addCylinder(center[0], center[1], 0.0, 0.0, 0.0, height, radius)
                elif axis == "y":
                    hole_tag = gmsh.model.occ.addCylinder(center[0], 0.0, center[1], 0.0, height, 0.0, radius)
                elif axis == "x":
                    hole_tag = gmsh.model.occ.addCylinder(0.0, center[0], center[1], height, 0.0, 0.0, radius)
                hole_tags.append(hole_tag)

        if hole_tags:
            # Get main domain volumes
            domain_volumes = gmsh.model.getEntities(3)
            main_volume_tags = [tag[1] for tag in domain_volumes]

            # Boolean subtraction: domain - holes
            for hole_tag in hole_tags:
                result = gmsh.model.occ.cut([(3, main_volume_tags[0])], [(3, hole_tag)])
                if result[0]:
                    main_volume_tags = [tag[1] for tag in result[0]]

            gmsh.model.occ.synchronize()

    def generate_mesh(self) -> MeshData:
        """Generate 3D tetrahedral mesh.

        Returns a pre-populated ``self.mesh_data`` as-is if present (gmsh-free injected-mesh
        path / idempotent memoization).

        Note: gmsh generation here emits **tetrahedral (simplex)** elements only. Non-simplex
        meshes (e.g. hexahedral) for the FEM solve path (Issue #470) are not generated here —
        inject a pre-built mesh instead, e.g. ``skfem.MeshHex.init_tensor`` → ``skfem_to_meshdata``."""
        if self.mesh_data is not None:
            return self.mesh_data
        try:
            import gmsh
            import meshio  # noqa: F401
        except ImportError:
            raise ImportError("gmsh and meshio are required for mesh generation") from None

        # Create geometry
        self.create_gmsh_geometry()

        # Generate 3D mesh
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")

        # Extract mesh data
        _node_tags, coord, _ = gmsh.model.mesh.getNodes()
        vertices = coord.reshape(-1, 3)

        # Get tetrahedral elements
        element_types, element_tags, element_connectivity = gmsh.model.mesh.getElements(3)

        if len(element_types) == 0:
            raise RuntimeError("No 3D elements generated")

        # Process tetrahedral elements (type 4 in Gmsh)
        tetrahedra = None
        for elem_type, _tags, connectivity in zip(element_types, element_tags, element_connectivity, strict=False):
            if elem_type == 4:  # Tetrahedron
                tetrahedra = connectivity.reshape(-1, 4) - 1  # Convert to 0-based indexing
                break

        if tetrahedra is None:
            raise RuntimeError("No tetrahedral elements found")

        # Get boundary faces (triangles on domain boundary)
        boundary_faces = []
        boundary_tags = []

        # Get all physical surfaces and their elements
        physical_surfaces = gmsh.model.getPhysicalGroups(2)
        surface_element_map = {}  # Maps element tag to physical group

        for dim_tag in physical_surfaces:
            dim, phys_tag = dim_tag
            # Get entities in this physical group
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag)
            for entity in entities:
                # Get mesh elements on this entity
                elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim, entity)
                for elem_type, tags in zip(elem_types, elem_tags, strict=False):
                    if elem_type == 2:  # Triangle
                        for tag in tags:
                            surface_element_map[int(tag)] = phys_tag

        # Now get boundary faces
        for dim in [2]:  # Surface elements
            elem_types, elem_tags, elem_connectivity = gmsh.model.mesh.getElements(dim)
            for elem_type, tags, connectivity in zip(elem_types, elem_tags, elem_connectivity, strict=False):
                if elem_type == 2:  # Triangle
                    faces = connectivity.reshape(-1, 3) - 1
                    boundary_faces.extend(faces)

                    # Get physical group tags for boundary identification
                    for tag in tags:
                        phys_tag = surface_element_map.get(int(tag), 0)
                        boundary_tags.append(phys_tag)

        # Get element physical tags (for region identification)
        # Create mapping from volume element tags to physical groups
        physical_volumes = gmsh.model.getPhysicalGroups(3)
        volume_element_map = {}  # Maps element tag to physical group

        for dim_tag in physical_volumes:
            dim, phys_tag = dim_tag
            # Get entities in this physical group
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag)
            for entity in entities:
                # Get mesh elements on this entity
                elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim, entity)
                for elem_type, tags in zip(elem_types, elem_tags, strict=False):
                    if elem_type == 4:  # Tetrahedron
                        for tag in tags:
                            volume_element_map[int(tag)] = phys_tag

        # Use the mapping to assign physical tags
        element_physical_tags = np.ones(len(tetrahedra), dtype=int)
        for i, elem_tag in enumerate(element_tags[0]):
            phys_tag = volume_element_map.get(int(elem_tag), 1)
            element_physical_tags[i] = phys_tag

        gmsh.finalize()

        # Create MeshData
        mesh_data = MeshData(
            vertices=vertices,
            elements=tetrahedra,
            element_type="tetrahedron",
            boundary_tags=np.array(boundary_tags) if boundary_tags else np.array([]),
            element_tags=element_physical_tags,
            boundary_faces=np.array(boundary_faces) if boundary_faces else np.array([]).reshape(0, 3),
            dimension=3,
        )

        # Compute quality metrics
        mesh_data.quality_metrics = self.compute_mesh_quality_3d(mesh_data)
        mesh_data.element_volumes = self._compute_tetrahedron_volumes(mesh_data.vertices, mesh_data.elements)

        self.mesh_data = mesh_data
        return mesh_data

    def compute_mesh_quality_3d(self, mesh_data: MeshData) -> dict[str, float]:
        """Compute 3D mesh quality metrics."""
        volumes = mesh_data.element_volumes
        if volumes is None:
            volumes = self._compute_tetrahedron_volumes(mesh_data.vertices, mesh_data.elements)

        # Radius ratio quality measure for tetrahedra
        quality_ratios: list[float] = []
        for elem in mesh_data.elements:
            coords = mesh_data.vertices[elem]

            # Compute circumradius and inradius
            circumradius = self._compute_circumradius(coords)
            inradius = self._compute_inradius(coords)

            if circumradius > 0:
                quality_ratios.append(inradius / circumradius)
            else:
                quality_ratios.append(0.0)

        quality_ratios_array = np.array(quality_ratios)

        return {
            "min_volume": float(np.min(volumes)),
            "max_volume": float(np.max(volumes)),
            "mean_volume": float(np.mean(volumes)),
            "volume_ratio": float(np.max(volumes) / np.min(volumes)) if np.min(volumes) > 0 else np.inf,
            "min_quality": float(np.min(quality_ratios_array)),
            "mean_quality": float(np.mean(quality_ratios_array)),
            "num_poor_elements": int(np.sum(quality_ratios_array < 0.1)),
            "quality_histogram": np.histogram(quality_ratios_array, bins=10)[0].tolist(),
        }

    def _compute_tetrahedron_volumes(self, vertices: np.ndarray, elements: np.ndarray) -> np.ndarray:
        """Compute volumes of tetrahedral elements."""
        volumes = np.zeros(len(elements))

        for i, elem in enumerate(elements):
            coords = vertices[elem]  # 4x3 array

            # Volume = |det(b-a, c-a, d-a)| / 6
            v1 = coords[1] - coords[0]
            v2 = coords[2] - coords[0]
            v3 = coords[3] - coords[0]

            # Volume using scalar triple product
            volume = abs(np.dot(v1, np.cross(v2, v3))) / 6.0
            volumes[i] = volume

        return volumes

    def _compute_circumradius(self, coords: np.ndarray) -> float:
        """Compute circumradius of tetrahedron."""
        # Simplified calculation for demonstration
        # In practice, would use more robust geometric computation
        edges: list[float] = []
        for i in range(4):
            for j in range(i + 1, 4):
                edge_length = float(np.linalg.norm(coords[i] - coords[j]))
                edges.append(edge_length)

        return max(edges) / 2.0  # Approximation

    def _compute_inradius(self, coords: np.ndarray) -> float:
        """Compute inradius of tetrahedron."""
        # Volume
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]
        v3 = coords[3] - coords[0]
        volume = abs(np.dot(v1, np.cross(v2, v3))) / 6.0

        # Surface area (sum of 4 triangular faces)
        faces = [
            [coords[0], coords[1], coords[2]],
            [coords[0], coords[1], coords[3]],
            [coords[0], coords[2], coords[3]],
            [coords[1], coords[2], coords[3]],
        ]

        surface_area = 0.0
        for face in faces:
            # Area of triangle using cross product
            edge1 = face[1] - face[0]
            edge2 = face[2] - face[0]
            area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
            surface_area += float(area)

        # Inradius = 3 * Volume / Surface_area
        if surface_area > 0:
            return 3.0 * volume / surface_area
        else:
            return 0.0

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get 3D domain bounding box."""
        if self.domain_type == "box":
            min_coords = np.array([self.xmin, self.ymin, self.zmin])
            max_coords = np.array([self.xmax, self.ymax, self.zmax])
            return min_coords, max_coords

        elif self.domain_type == "sphere":
            center_x, center_y, center_z = self.center
            min_coords = np.array([center_x - self.radius, center_y - self.radius, center_z - self.radius])
            max_coords = np.array([center_x + self.radius, center_y + self.radius, center_z + self.radius])
            return min_coords, max_coords

        elif self.domain_type == "cylinder":
            if self.axis == "z":
                center_x, center_y = self.center
                min_coords = np.array([center_x - self.radius, center_y - self.radius, 0.0])
                max_coords = np.array([center_x + self.radius, center_y + self.radius, self.height])
            elif self.axis == "y":
                center_x, center_z = self.center
                min_coords = np.array([center_x - self.radius, 0.0, center_z - self.radius])
                max_coords = np.array([center_x + self.radius, self.height, center_z + self.radius])
            else:  # axis == "x"
                center_y, center_z = self.center
                min_coords = np.array([0.0, center_y - self.radius, center_z - self.radius])
                max_coords = np.array([self.height, center_y + self.radius, center_z + self.radius])
            return min_coords, max_coords

        elif self.domain_type == "polyhedron":
            vertices_array = np.array(self.vertices)
            min_coords = np.min(vertices_array, axis=0)
            max_coords = np.max(vertices_array, axis=0)
            return min_coords, max_coords

        else:
            # For custom domains, generate mesh first to get bounds
            if self.mesh_data is None:
                self.mesh_data = self.generate_mesh()
            return self.mesh_data.bounds

    @deprecated_parameter(
        param_name="format_type",
        since="v0.17.12",
        replacement="file_format",
    )
    def export_mesh(self, file_format: str = "", filename: str = "", format_type: str | None = None) -> None:
        """Export mesh in specified format.

        Args:
            file_format: Meshio file format string (e.g. "vtk", "gmsh")
            filename: Output file path
            format_type: Deprecated, use ``file_format`` instead.
        """
        if format_type is not None:
            file_format = format_type

        if self.mesh_data is None:
            self.generate_mesh()

        if self.mesh_data is None:
            raise RuntimeError("Failed to generate mesh data")

        try:
            import meshio
        except ImportError:
            raise ImportError("meshio required for mesh export") from None

        # Convert to meshio format
        cells = [("tetra", self.mesh_data.elements)]

        mesh = meshio.Mesh(
            points=self.mesh_data.vertices, cells=cells, cell_data={"physical": [self.mesh_data.element_tags]}
        )

        # Write mesh
        if file_format:
            mesh.write(filename, file_format=file_format)
        else:
            mesh.write(filename)

    def visualize_interactive(self, show_quality: bool = False):
        """Create interactive 3D visualization using PyVista."""
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("pyvista required for interactive visualization") from None

        # Issue #543: Explicit None check instead of hasattr (initialized in __init__)
        if self.mesh_data is None:
            self.mesh_data = self.generate_mesh()

        # Create PyVista mesh
        cells = np.hstack(
            [
                np.full((len(self.mesh_data.elements), 1), 4),  # Tetrahedron cell type
                self.mesh_data.elements,
            ]
        )

        mesh = pv.UnstructuredGrid(cells, np.full(len(self.mesh_data.elements), 10), self.mesh_data.vertices)

        # Create plotter
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, show_edges=True, opacity=0.6)

        if show_quality and self.mesh_data.quality_metrics:
            # Add quality visualization
            quality_data = np.array(
                [self.mesh_data.quality_metrics.get("min_quality", 0.5)] * len(self.mesh_data.elements)
            )
            mesh.cell_data["quality"] = quality_data  # type: ignore[attr-defined]
            plotter.add_mesh(mesh, scalars="quality", cmap="coolwarm")

        plotter.show_grid()
        plotter.add_axes()
        plotter.show()

"""
Mesh adapter between MFGarchon's MeshData and scikit-fem's Mesh types.

Handles bidirectional conversion:
    MeshData -> skfem.Mesh (for assembly)
    skfem.Mesh -> MeshData (for MFGarchon pipeline)

The key difference is array layout:
    MeshData:  vertices (N, dim), elements (M, nodes_per_elem)
    skfem:     nodes (dim, N),    elements (nodes_per_elem, M)

Issue #773 Phase 1: Core integration
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mfgarchon.geometry.meshes.mesh_data import MeshData


def _import_skfem():
    """Import scikit-fem with clear error message."""
    try:
        import skfem
    except ImportError:
        raise ImportError("scikit-fem is required for FEM assembly. Install with: pip install scikit-fem") from None
    return skfem


def meshdata_to_skfem(mesh_data: MeshData) -> skfem.Mesh:
    """
    Convert MFGarchon MeshData to scikit-fem Mesh.

    Args:
        mesh_data: MFGarchon mesh with vertices, elements, and element_type.

    Returns:
        scikit-fem Mesh object (MeshTri, MeshTet, MeshQuad, or MeshLine).

    Raises:
        ValueError: If element_type is not supported.
        ImportError: If scikit-fem is not installed.
    """
    skfem = _import_skfem()

    nodes = mesh_data.vertices.T.astype(np.float64)  # (dim, N)
    elements = mesh_data.elements.T.astype(np.int64)  # (nodes_per_elem, M)

    # Map element types to skfem mesh classes
    mesh_classes = {
        "triangle": skfem.MeshTri,
        "tetrahedron": skfem.MeshTet,
        "quad": skfem.MeshQuad,
        "line": skfem.MeshLine,
    }

    mesh_cls = mesh_classes.get(mesh_data.element_type)
    if mesh_cls is None:
        raise ValueError(
            f"Unsupported element type '{mesh_data.element_type}' for scikit-fem. "
            f"Supported: {list(mesh_classes.keys())}"
        )

    mesh = mesh_cls(nodes, elements)

    # Transfer boundary tags if available (gmsh physical-group path)
    if mesh_data.boundary_faces is not None and len(mesh_data.boundary_faces) > 0:
        _apply_boundary_tags(mesh, mesh_data)

    # Issue #607: tag axis-aligned wall facets as named boundaries (x_min/x_max/...) matching the
    # BoundaryFace naming, so a BCSegment(boundary="x_min") resolves to the right facet set. This
    # is the unified segment->facet specification for axis-aligned (box) mesh domains; without it
    # mesh.boundaries is None and the FEM bc_adapter falls back to the entire boundary.
    mesh = _tag_axis_aligned_boundaries(mesh)

    return mesh


def _tag_axis_aligned_boundaries(mesh: skfem.Mesh) -> skfem.Mesh:
    """Return ``mesh`` with named axis-aligned wall boundaries (``x_min``/``x_max``/``y_min``/...).

    A facet is on wall ``<axis>_<side>`` when its coordinate on that axis is within ``BOUNDARY_TOL``
    of the mesh bounding-box bound. Names follow ``BoundaryFace.to_string()`` (axis 0->x, 1->y,
    2->z), matching ``BCSegment.boundary``. For non-box domains these tag only the facets touching
    the bounding box on each axis; SDF/region-based markers for curved domains are out of scope here.
    """
    from mfgarchon.geometry.boundary.tolerances import BOUNDARY_TOL

    dim = mesh.p.shape[0]
    mins = mesh.p.min(axis=1)
    maxs = mesh.p.max(axis=1)
    axis_names = "xyz"
    predicates: dict = {}
    for d in range(dim):
        name = axis_names[d] if d < len(axis_names) else f"axis{d}"
        predicates[f"{name}_min"] = lambda x, d=d, b=mins[d]: np.isclose(x[d], b, atol=BOUNDARY_TOL)
        predicates[f"{name}_max"] = lambda x, d=d, b=maxs[d]: np.isclose(x[d], b, atol=BOUNDARY_TOL)
    return mesh.with_boundaries(predicates)


def skfem_to_meshdata(mesh: skfem.Mesh) -> MeshData:
    """
    Convert scikit-fem Mesh to MFGarchon MeshData.

    Args:
        mesh: scikit-fem Mesh object.

    Returns:
        MeshData with vertices, elements, and element_type.
    """
    from mfgarchon.geometry.meshes.mesh_data import MeshData

    vertices = mesh.p.T.astype(np.float64)  # (N, dim)
    elements = mesh.t.T.astype(np.int64)  # (M, nodes_per_elem)

    # Map skfem mesh type to element string
    skfem = _import_skfem()
    type_map = {
        skfem.MeshTri: "triangle",
        skfem.MeshTri1: "triangle",
        skfem.MeshTet: "tetrahedron",
        skfem.MeshTet1: "tetrahedron",
        skfem.MeshQuad: "quad",
        skfem.MeshQuad1: "quad",
        skfem.MeshLine: "line",
        skfem.MeshLine1: "line",
    }

    element_type = None
    for cls, name in type_map.items():
        if isinstance(mesh, cls):
            element_type = name
            break

    if element_type is None:
        element_type = "unknown"

    dim = mesh.p.shape[0]

    # Extract boundary faces
    boundary_facets = mesh.boundary_facets()
    if len(boundary_facets) > 0:
        boundary_faces = mesh.facets[:, boundary_facets].T.astype(np.int64)
    else:
        boundary_faces = np.empty((0, dim), dtype=np.int64)

    return MeshData(
        vertices=vertices,
        elements=elements,
        element_type=element_type,
        boundary_tags=np.zeros(len(boundary_faces), dtype=np.int64),
        element_tags=np.zeros(len(elements), dtype=np.int64),
        boundary_faces=boundary_faces,
        dimension=dim,
    )


def _apply_boundary_tags(mesh: skfem.Mesh, mesh_data: MeshData) -> None:
    """Transfer boundary tags from MeshData to skfem Mesh boundaries dict.

    Maps MeshData's boundary_tags to skfem's mesh.boundaries dict,
    which maps string names to arrays of facet indices.
    """
    if mesh_data.boundary_tags is None or len(mesh_data.boundary_tags) == 0:
        return

    # Issue #1260: mesh.boundaries is a read-only property backed by mesh._boundaries, which is
    # None on a freshly constructed MeshTri/MeshLine/MeshTet (skfem 12.0.1).  Item-assignment on
    # None raises TypeError; initialize the backing field to an empty dict before use.
    # 2026-06-10 audit.
    if mesh._boundaries is None:
        mesh._boundaries = {}

    unique_tags = np.unique(mesh_data.boundary_tags)
    for tag in unique_tags:
        if tag == 0:
            continue  # Skip default/untagged
        mask = mesh_data.boundary_tags == tag
        facet_indices = np.where(mask)[0]
        mesh._boundaries[f"region_{tag}"] = facet_indices


if __name__ == "__main__":
    """Smoke test for mesh adapter."""
    import skfem

    print("Testing mesh adapter...")

    # Create a simple skfem mesh
    mesh = skfem.MeshTri.init_symmetric()
    print(f"skfem mesh: {mesh.p.shape[1]} nodes, {mesh.t.shape[1]} elements")

    # Convert to MeshData
    md = skfem_to_meshdata(mesh)
    print(f"MeshData: {md.vertices.shape[0]} vertices, {md.elements.shape[0]} elements, type={md.element_type}")

    # Round-trip
    mesh2 = meshdata_to_skfem(md)
    print(f"Round-trip: {mesh2.p.shape[1]} nodes, {mesh2.t.shape[1]} elements")

    assert mesh2.p.shape == mesh.p.shape
    assert mesh2.t.shape == mesh.t.shape
    assert np.allclose(mesh2.p, mesh.p)
    assert np.array_equal(mesh2.t, mesh.t)

    print("Round-trip passed.")

"""
Mesh adapter between MFGarchon's MeshData and scikit-fem's Mesh types.

Handles bidirectional conversion:
    MeshData -> skfem.Mesh (for assembly)
    skfem.Mesh -> MeshData (for MFGarchon pipeline)

The key difference is array layout:
    MeshData:  vertices (N, dim), elements (M, nodes_per_elem)
    skfem:     nodes (dim, N),    elements (nodes_per_elem, M)

Supported element families (Issue #470): line, triangle, quad (2D), tetrahedron,
hexahedron (3D). The family is fixed by ``MeshData.element_type`` (forward) or the
skfem mesh class (reverse). P1 and P2 Lagrange variants of each family map to the
same ``element_type`` string; the polynomial order is chosen later in
``assembly.create_basis``.

Bring-your-own-mesh (no in-process gmsh dependency)
---------------------------------------------------
Complex unstructured meshes do not need an in-process mesh generator. mfgarchon
relies on skfem's element families plus its mesh I/O, so the supported paths are:

1. Structured / tensor-product meshes, gmsh-free::

       import skfem, numpy as np
       xs = np.linspace(0.0, 1.0, n)
       mesh = skfem.MeshHex.init_tensor(xs, xs, xs)   # or MeshTet/MeshQuad/MeshTri
       mesh_data = skfem_to_meshdata(mesh)            # -> MeshData -> FEM solve

2. External mesher (gmsh, Cubit, ...) -> meshio file -> skfem (recommended for
   complex unstructured geometry)::

       # Generate `domain.msh` (or .vtk/.xdmf/...) in *any* external mesher.
       mesh = skfem.Mesh.load("domain.msh")           # skfem reads via meshio
       mesh_data = skfem_to_meshdata(mesh)            # -> MeshData -> FEM solve

   ``skfem.Mesh.load`` uses meshio (already a dependency), so any meshio-supported
   format works. This keeps gmsh an *out-of-process* tool: mfgarchon never imports
   it. meshio's tetra/hexahedron cell names map onto the families above. Prism /
   pyramid / mixed-element meshes are out of scope (see notes below and Issue #470).

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
        scikit-fem Mesh object (MeshTri, MeshTet, MeshQuad, MeshHex, or MeshLine).

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
        "hexahedron": skfem.MeshHex,  # Issue #470: 3D tensor-product family
        "line": skfem.MeshLine,
    }

    mesh_cls = mesh_classes.get(mesh_data.element_type)
    if mesh_cls is None:
        raise ValueError(
            f"Unsupported element type '{mesh_data.element_type}' for scikit-fem. "
            f"Supported: {list(mesh_classes.keys())}"
        )

    # Issue #1489 (F6): scikit-fem silently TRUNCATES a connectivity with the wrong node count (e.g. a
    # 4-node quad mislabeled 'triangle' is sliced to 3 rows -> a wrong half-domain mesh, no error).
    # Validate the connectivity width against the element family before constructing.
    expected_nodes = {"triangle": 3, "tetrahedron": 4, "quad": 4, "hexahedron": 8, "line": 2}
    n_expected = expected_nodes[mesh_data.element_type]
    if elements.shape[0] != n_expected:
        raise ValueError(
            f"element_type '{mesh_data.element_type}' expects {n_expected} nodes per element, but the "
            f"connectivity has {elements.shape[0]} (elements array shape {tuple(mesh_data.elements.shape)}). "
            f"scikit-fem would silently truncate this to a wrong mesh (#1489)."
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
    # Issue #470: hexahedron is the 3D tensor-product family. In skfem 12.0.1
    # ``MeshHex is MeshHex1`` (linear), and ``MeshHex2`` (P2-geometry) subclasses it,
    # so the ``MeshHex`` isinstance test below already catches both; ``MeshHex2`` is
    # listed for parity with the explicit P1/P2 entries of the other families.
    #
    # Prism / wedge (skfem.MeshWedge1) is intentionally NOT mapped here. Its
    # points+cells round-trip, but two issues block a clean mapping (deferred to a
    # follow-up, see Issue #470): (1) skfem stores the wedge's triangular faces as
    # degenerate 4-node quad facets (e.g. ``[0, 0, 2, 3]``), so ``boundary_faces``
    # are lossy; (2) meshio's cell name is ``"wedge"``, not ``"prism"``, so the
    # round-trip name through ``MeshData.to_meshio`` needs a maintainer decision.
    type_map = {
        skfem.MeshTri: "triangle",
        skfem.MeshTri1: "triangle",
        skfem.MeshTet: "tetrahedron",
        skfem.MeshTet1: "tetrahedron",
        skfem.MeshQuad: "quad",
        skfem.MeshQuad1: "quad",
        skfem.MeshHex: "hexahedron",
        skfem.MeshHex2: "hexahedron",
        skfem.MeshLine: "line",
        skfem.MeshLine1: "line",
    }

    element_type = None
    for cls, name in type_map.items():
        if isinstance(mesh, cls):
            element_type = name
            break

    if element_type is None:
        # Issue #1489 (F8): fail loud rather than stamping 'unknown', which produces a mislabeled
        # MeshData that only fails (confusingly) on a later re-conversion.
        raise ValueError(
            f"Unsupported scikit-fem mesh class {type(mesh).__name__!r} for MeshData conversion "
            f"(#1489). Supported: MeshTri, MeshTet, MeshQuad, MeshHex, MeshLine."
        )

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

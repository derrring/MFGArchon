"""
Pinning tests for Issue #1260: two FEM adapter defects.

(A) apply_bc_to_fem_system had no terminal else for unhandled BCTypes, so
    EXTRAPOLATION_LINEAR/QUADRATIC silently degraded to Neumann BC.
    Fix: terminal else raises NotImplementedError.

(B) _apply_boundary_tags crashed with TypeError on gmsh-tagged meshes because
    mesh.boundaries is None on a freshly constructed skfem Mesh.
    Fix: initialize mesh._boundaries to {} before item-assignment.

Both tests FAIL on the unfixed code and PASS after the fix.
"""

from __future__ import annotations

import pytest

import numpy as np

skfem = pytest.importorskip("skfem", reason="scikit-fem required")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _minimal_skfem_basis():
    """Return a tiny (A, rhs, basis) triple for bc_adapter tests."""
    from scipy import sparse

    mesh = skfem.MeshTri.init_sqsymmetric()
    elem = skfem.ElementTriP1()
    basis = skfem.Basis(mesh, elem)
    n = basis.N
    A = sparse.eye(n, format="csr")
    rhs = np.ones(n)
    return A, rhs, basis


def _extrapolation_bc(dimension: int = 2):
    """BoundaryConditions with one EXTRAPOLATION_LINEAR segment."""
    from mfgarchon.geometry.boundary.conditions import BoundaryConditions
    from mfgarchon.geometry.boundary.types import BCSegment, BCType

    return BoundaryConditions(
        segments=[
            BCSegment(
                name="left_extrap",
                bc_type=BCType.EXTRAPOLATION_LINEAR,
                value=0.0,
                boundary="x_min",
            )
        ],
        dimension=dimension,
    )


def _tagged_meshdata():
    """MeshData for a 1D line mesh with nonzero boundary_tags."""
    from mfgarchon.geometry.meshes.mesh_data import MeshData

    # A trivial 1D segment: 3 nodes, 2 line elements, 2 boundary faces
    vertices = np.array([[0.0], [0.5], [1.0]])
    elements = np.array([[0, 1], [1, 2]])
    boundary_faces = np.array([[0], [2]])
    boundary_tags = np.array([1, 2], dtype=np.int64)  # nonzero tags
    element_tags = np.zeros(2, dtype=np.int64)
    return MeshData(
        vertices=vertices,
        elements=elements,
        element_type="line",
        boundary_tags=boundary_tags,
        element_tags=element_tags,
        boundary_faces=boundary_faces,
        dimension=1,
    )


# ---------------------------------------------------------------------------
# (A) EXTRAPOLATION_LINEAR must raise NotImplementedError (Issue #1260)
# ---------------------------------------------------------------------------


def test_extrapolation_linear_raises_not_implemented():
    """apply_bc_to_fem_system must raise NotImplementedError for EXTRAPOLATION_LINEAR.

    Before the fix: the segment matched no branch in the if/elif chain and the
    function returned (A, rhs) unchanged — silent Neumann degrade.
    After the fix: the terminal else raises NotImplementedError.
    """
    from mfgarchon.alg.numerical.fem.bc_adapter import apply_bc_to_fem_system

    A, rhs, basis = _minimal_skfem_basis()
    bc = _extrapolation_bc(dimension=2)

    with pytest.raises(NotImplementedError, match="EXTRAPOLATION"):
        apply_bc_to_fem_system(A, rhs, basis, bc)


def test_extrapolation_quadratic_raises_not_implemented():
    """apply_bc_to_fem_system must raise NotImplementedError for EXTRAPOLATION_QUADRATIC."""
    from mfgarchon.alg.numerical.fem.bc_adapter import apply_bc_to_fem_system
    from mfgarchon.geometry.boundary.conditions import BoundaryConditions
    from mfgarchon.geometry.boundary.types import BCSegment, BCType

    A, rhs, basis = _minimal_skfem_basis()
    bc = BoundaryConditions(
        segments=[
            BCSegment(
                name="right_extrap",
                bc_type=BCType.EXTRAPOLATION_QUADRATIC,
                value=0.0,
                boundary="x_max",
            )
        ],
        dimension=2,
    )

    with pytest.raises(NotImplementedError, match="EXTRAPOLATION"):
        apply_bc_to_fem_system(A, rhs, basis, bc)


# ---------------------------------------------------------------------------
# (B) gmsh-tagged MeshData must not crash (Issue #1260)
# ---------------------------------------------------------------------------


def test_tagged_meshdata_to_skfem_no_crash():
    """meshdata_to_skfem must not raise TypeError for nonzero boundary_tags.

    Before the fix: _apply_boundary_tags did mesh.boundaries[...] = ... on a
    None boundaries dict, raising TypeError.
    After the fix: mesh._boundaries is initialized to {} first.
    """
    from mfgarchon.alg.numerical.fem.mesh_adapter import meshdata_to_skfem

    md = _tagged_meshdata()
    # Must not raise TypeError
    mesh = meshdata_to_skfem(md)

    # Verify the boundary tags were actually registered
    assert mesh.boundaries is not None, "boundaries dict must be populated"
    assert "region_1" in mesh.boundaries, "region_1 must be registered"
    assert "region_2" in mesh.boundaries, "region_2 must be registered"


def test_meshdata_to_skfem_fails_loud_on_wrong_node_count():
    """Issue #1489 (F6): a connectivity whose node count mismatches the element family must raise,
    not let scikit-fem silently truncate it to a wrong mesh (a 4-node quad mislabeled 'triangle'
    would be sliced to 3 rows -> a half-domain mesh with no error)."""
    import pytest

    import numpy as np

    from mfgarchon.alg.numerical.fem.mesh_adapter import meshdata_to_skfem
    from mfgarchon.geometry.meshes.mesh_data import MeshData

    md = MeshData(
        vertices=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        elements=np.array([[0, 1, 2, 3]]),  # 4 nodes -> valid quad, MISLABELED as triangle
        element_type="triangle",
        boundary_tags=np.array([], dtype=np.int64),
        element_tags=np.array([0], dtype=np.int64),
        boundary_faces=np.empty((0, 2), dtype=np.int64),
        dimension=2,
    )
    with pytest.raises(ValueError, match="expects 3 nodes"):
        meshdata_to_skfem(md)

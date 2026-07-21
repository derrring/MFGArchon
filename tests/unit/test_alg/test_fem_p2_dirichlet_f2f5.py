"""Issue #1489 (F2/F4/F5): FEM P2 Dirichlet BC handling.
- F2: the whole-boundary fallback resolves ALL DOFs (incl. P2 edge DOFs), not vertices only.
- F4: a corner DOF shared by two Dirichlet segments with conflicting values fails loud.
- F5: a callable Dirichlet value evaluates at the DOF coordinates (basis.doflocs), not vertex coords.
"""

from __future__ import annotations

import pytest

import numpy as np

skfem = pytest.importorskip("skfem", reason="scikit-fem required")


def _p2_basis_box():
    mesh = skfem.MeshTri.init_sqsymmetric().refined(2).with_boundaries(
        {
            "x_min": lambda x: np.isclose(x[0], 0.0),
            "y_min": lambda x: np.isclose(x[1], 0.0),
        }
    )
    from mfgarchon.alg.numerical.fem.assembly import create_basis

    return create_basis(mesh, order=2)  # P2


def test_p2_callable_dirichlet_does_not_crash():
    """F5: a callable Dirichlet on a P2 boundary indexes basis.doflocs (a coord per DOF), not mesh.p
    (vertices only) — the latter IndexError'd on edge-midpoint DOF indices >= n_vertices."""
    from mfgarchon.alg.numerical.fem.bc_adapter import _evaluate_segment_values, _find_segment_dofs
    from mfgarchon.geometry.boundary.conditions import BCSegment, BCType

    basis = _p2_basis_box()
    seg = BCSegment(name="inlet", bc_type=BCType.DIRICHLET, boundary="x_min", value=lambda x: 2.0 * x[1])
    dofs = _find_segment_dofs(basis.mesh, basis, seg)
    vals = _evaluate_segment_values(seg, basis, dofs)  # must not raise
    assert len(vals) == len(dofs) > 0


def test_conflicting_corner_dirichlet_fails_loud():
    """F4: two Dirichlet walls meeting at a corner with DIFFERENT values -> the shared corner DOF has
    conflicting values -> condensation must fail loud, not double-count / last-write-win."""
    from scipy import sparse

    from mfgarchon.alg.numerical.fem.assembly import assemble_stiffness
    from mfgarchon.alg.numerical.fem.bc_adapter import apply_bc_to_fem_system
    from mfgarchon.geometry.boundary.conditions import BCSegment, BCType, BoundaryConditions

    basis = _p2_basis_box()
    a = assemble_stiffness(basis) + sparse.eye(basis.N, format="csr")
    rhs = np.ones(basis.N)
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="L", bc_type=BCType.DIRICHLET, boundary="x_min", value=1.0),
            BCSegment(name="B", bc_type=BCType.DIRICHLET, boundary="y_min", value=2.0),  # corner (0,0) conflicts
        ],
        dimension=2,
    )
    with pytest.raises(ValueError, match="Conflicting"):
        apply_bc_to_fem_system(a, rhs, basis, bc)

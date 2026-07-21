"""Issue #1489 (F7): FEM basis creation fails loud on a degenerate (zero-measure) element instead of
silently assembling NaN stiffness/mass entries (only a numpy RuntimeWarning otherwise)."""

from __future__ import annotations

import pytest

skfem = pytest.importorskip("skfem", reason="scikit-fem required")


def test_create_basis_fails_loud_on_degenerate_element():
    from mfgarchon.alg.numerical.fem.assembly import create_basis

    m = skfem.MeshTri.init_sqsymmetric()
    p = m.p.copy()
    p[:, 1] = p[:, 0]  # collapse a vertex onto another -> zero-area (degenerate) triangles
    degenerate = skfem.MeshTri(p, m.t)
    with pytest.raises(ValueError, match="degenerate"):
        create_basis(degenerate, order=1)


def test_create_basis_accepts_valid_mesh():
    from mfgarchon.alg.numerical.fem.assembly import create_basis

    basis = create_basis(skfem.MeshTri.init_sqsymmetric().refined(1), order=2)  # no raise
    assert basis.N > 0

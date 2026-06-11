"""Issue #1252 (2026-06-10 audit): P2 mass-lumped nodal-gradient recovery is invalid.

Row-sum mass lumping (M_lumped = M.sum(axis=1)) assumes strictly positive lumped masses.
For P2+ Lagrange the vertex shape function integrates to ~0 over a triangle and to a negative
value over a tetrahedron, so the consistent-mass row sum at every vertex DOF is ~0 or < 0. The
old code clamped that to 1e-15 and 1/1e-15 = 1e15 turned the recovered vertex gradient into
garbage, silently feeding nonsense into H(grad u). _build_gradient_operators must now fail loud
for P2+ and keep working for P1. The fix exercises the real method via a minimal carrier holding
genuine skfem-assembled P1/P2 mass matrices.
"""

import pytest

import numpy as np
from scipy import sparse

skfem = pytest.importorskip("skfem", reason="scikit-fem required for FEM gradient-recovery test")

from mfgarchon.alg.numerical.fem.assembly import assemble_mass, create_basis  # noqa: E402
from mfgarchon.alg.numerical.weak_form_hjb_solver import WeakFormHJBSolver  # noqa: E402


class _GradStub:
    """Minimal carrier exposing the attributes _build_gradient_operators reads/writes."""

    def __init__(self, M, n):
        self._M = M
        # _R_grad is only consumed AFTER the positivity guard; identity placeholders suffice.
        self._R_grad = [sparse.eye(n, format="csr"), sparse.eye(n, format="csr")]
        self._G_grad = None
        self._M_lumped_inv = None


def _mass_matrix(order):
    mesh = skfem.MeshTri.init_sqsymmetric().refined(2)
    basis = create_basis(mesh, order=order)
    return assemble_mass(basis), basis.N


def test_p2_gradient_lumping_fails_loud():
    """P2: lumped row sums are pathological (~0/negative) -> must raise NotImplementedError."""
    M, n = _mass_matrix(order=2)
    m_lumped = np.asarray(M.sum(axis=1)).ravel()
    assert m_lumped.min() < 1e-12 * m_lumped.max(), "premise: P2 vertex DOFs should have ~0/negative lumped row sums"
    stub = _GradStub(M, n)
    with pytest.raises(NotImplementedError, match="lumping"):
        WeakFormHJBSolver._build_gradient_operators(stub)


def test_p1_gradient_lumping_ok():
    """P1: all lumped row sums strictly positive -> operators build without raising."""
    M, n = _mass_matrix(order=1)
    m_lumped = np.asarray(M.sum(axis=1)).ravel()
    assert m_lumped.min() > 1e-12 * m_lumped.max(), "P1 lumped row sums must be strictly positive"
    stub = _GradStub(M, n)
    WeakFormHJBSolver._build_gradient_operators(stub)  # must not raise
    assert stub._G_grad is not None
    assert len(stub._G_grad) == 2

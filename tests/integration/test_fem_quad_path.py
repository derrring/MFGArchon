"""End-to-end Quadrilateral FEM path — the smallest actionable slice of #470.

The mesh-generation layer (``Mesh2D``/``Mesh3D``) only emits simplex (Tri/Tet) meshes
via gmsh, and gmsh is not even an installed dependency. But the FEM *solve* path
(``mesh_adapter`` → ``create_basis`` → assembly) is element-family-agnostic, and skfem
can build a quad mesh gmsh-free via ``MeshQuad.init_tensor``. This test injects such a
mesh and drives a full Poisson solve at **both** Quad-P1 and Quad-P2, which:

- exercises the ``element_map`` Quad-P2 entry added for #470 (``(MeshQuad, 2) →
  ElementQuad2``); before it, ``order=2`` on a quad mesh raised ``ValueError``;
- confirms a quad ``MeshData`` round-trips through ``mesh_adapter`` (``element_type='quad'``);
- checks P2 is genuinely more accurate than P1 on the same mesh (not just "it runs").

EOC note: this is a *new FEM scheme* path; it does not touch the byte-identical
1D-LQ-FDM Picard or 2D-scattered-GFDM paper paths.
"""

from __future__ import annotations

import pytest

import numpy as np

skfem = pytest.importorskip("skfem", reason="scikit-fem required for FEM tests")

from mfgarchon.alg.numerical.fem.assembly import (  # noqa: E402
    apply_dirichlet_bc,
    assemble_mass,
    assemble_stiffness,
    create_basis,
)
from mfgarchon.alg.numerical.fem.mesh_adapter import meshdata_to_skfem, skfem_to_meshdata  # noqa: E402


def _quad_mesh(n: int) -> skfem.Mesh:
    """gmsh-free quad mesh on the unit square: an (n x n) node tensor grid."""
    xs = np.linspace(0.0, 1.0, n)
    return skfem.MeshQuad.init_tensor(xs, xs)


def _solve_poisson(mesh: skfem.Mesh, order: int):
    r"""Solve $-\Delta u = f$ with $u = u_{\text{exact}}$ on $\partial\Omega$ for the
    manufactured solution $u(x,y) = \sin(\pi x)\sin(\pi y)$ (so $f = 2\pi^2 u$, $u=0$
    on the unit-square boundary). Returns (max nodal error, n_dofs)."""
    basis = create_basis(mesh, order=order)
    coords = basis.doflocs  # (2, N)
    x, y = coords[0], coords[1]
    u_exact = np.sin(np.pi * x) * np.sin(np.pi * y)
    f = 2.0 * np.pi**2 * u_exact

    K = assemble_stiffness(basis)
    M = assemble_mass(basis)
    rhs = M @ f  # consistent FEM load vector ∫ f φ_i

    boundary_dofs = np.unique(basis.get_dofs().flatten())
    K_int, f_int = apply_dirichlet_bc(K, rhs, boundary_dofs, values=u_exact[boundary_dofs])

    from scipy.sparse.linalg import spsolve

    interior = np.setdiff1d(np.arange(basis.N), boundary_dofs)
    u = u_exact.copy()
    u[interior] = spsolve(K_int.tocsr(), f_int)

    return float(np.max(np.abs(u - u_exact))), basis.N


def test_quad_meshdata_roundtrip():
    """A quad mesh round-trips through the mesh adapter as ``element_type='quad'``."""
    mesh = _quad_mesh(5)
    md = skfem_to_meshdata(mesh)
    assert md.element_type == "quad"
    assert md.elements.shape[1] == 4  # 4 nodes per quad

    mesh2 = meshdata_to_skfem(md)
    assert "Quad" in type(mesh2).__name__
    assert mesh2.p.shape == mesh.p.shape
    assert mesh2.t.shape == mesh.t.shape


@pytest.mark.parametrize("order", [1, 2])
def test_quad_poisson_solves(order):
    """Quad P1 and P2 both assemble and solve Poisson end-to-end (#470 element_map).

    Before the fix, ``order=2`` on a quad mesh raised ``ValueError`` (no
    ``(MeshQuad, 2)`` entry in ``element_map``).
    """
    err, ndof = _solve_poisson(_quad_mesh(9), order=order)
    assert ndof > 0
    assert err < 0.05, f"Quad P{order} Poisson nodal error {err:.3e} too large"


def test_quad_p2_more_accurate_than_p1():
    """On the same mesh, Quad-P2 should be markedly more accurate than Quad-P1.

    Distinguishes a genuine P2 solve from a silent fallback to P1.
    """
    mesh = _quad_mesh(9)
    err_p1, n_p1 = _solve_poisson(mesh, order=1)
    err_p2, n_p2 = _solve_poisson(mesh, order=2)
    assert n_p2 > n_p1  # P2 adds edge/center dofs
    assert err_p2 < 0.25 * err_p1, f"P2 error {err_p2:.3e} not better than P1 {err_p1:.3e}"

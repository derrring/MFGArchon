"""Hexahedral mesh family support — Issue #470 (skfem-native, gmsh-free).

skfem 12.0.1 already ships the 3D tensor-product family (``MeshHex`` + ``ElementHex1``/
``ElementHex2``) and structured generation (``MeshHex.init_tensor``). The mfgarchon gap
was only that the ``MeshData`` <-> skfem adapter mapped triangle/tetrahedron/quad but not
hexahedron, and that ``assembly.create_basis`` had no hex entry. This test pins both:

1. a hex mesh round-trips through ``mesh_adapter`` (points + cells preserved, and the
   ``element_type`` string is ``"hexahedron"``);
2. the full FEM solve path (``create_basis`` -> assembly -> solve) runs on a hex mesh and
   produces a finite solution, with P2 genuinely more accurate than P1 on the same mesh.

No gmsh dependency is involved: the mesh is built gmsh-free via ``MeshHex.init_tensor``,
matching the "bring your own mesh" path documented in ``mesh_adapter``.

Prism / wedge is intentionally out of scope here (see ``mesh_adapter`` reverse-map note).
"""

from __future__ import annotations

import pytest

import numpy as np

skfem = pytest.importorskip("skfem", reason="scikit-fem required for FEM tests")

from mfgarchon.alg.numerical.fem.assembly import (  # noqa: E402
    assemble_mass,
    assemble_stiffness,
    create_basis,
)
from mfgarchon.alg.numerical.fem.mesh_adapter import (  # noqa: E402
    meshdata_to_skfem,
    skfem_to_meshdata,
)


def _hex_mesh(n: int) -> skfem.Mesh:
    """gmsh-free hex mesh on the unit cube: an (n x n x n) node tensor grid."""
    xs = np.linspace(0.0, 1.0, n)
    return skfem.MeshHex.init_tensor(xs, xs, xs)


# ---------------------------------------------------------------------------
# (1) Adapter round-trip: points + cells preserved, element_type correct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", ["init_tensor", "refined"])
def test_hex_roundtrips_through_adapter(builder: str):
    """A hex mesh -> MeshData -> hex mesh preserves points and cells exactly."""
    if builder == "init_tensor":
        mesh = _hex_mesh(4)
    else:
        mesh = skfem.MeshHex().refined(2)

    md = skfem_to_meshdata(mesh)
    assert md.element_type == "hexahedron"
    assert md.dimension == 3
    assert md.vertices.shape == (mesh.p.shape[1], 3)
    assert md.elements.shape == (mesh.t.shape[1], 8)

    mesh_rt = meshdata_to_skfem(md)
    assert isinstance(mesh_rt, skfem.MeshHex)
    assert np.allclose(mesh_rt.p, mesh.p), "vertices must round-trip"
    assert np.array_equal(mesh_rt.t, mesh.t), "cells must round-trip"


def test_hex_meshdata_reports_hexahedron_for_p2_geometry():
    """A P2-geometry hex (MeshHex2) maps to the same 'hexahedron' string."""
    # MeshHex2 subclasses MeshHex1; the reverse map must still resolve it.
    mesh_p2 = skfem.MeshHex2.init_tensor(np.linspace(0, 1, 3), np.linspace(0, 1, 3), np.linspace(0, 1, 3))
    md = skfem_to_meshdata(mesh_p2)
    assert md.element_type == "hexahedron"


def test_hex_axis_aligned_boundaries_are_tagged():
    """The box-wall tagging path applies to hex meshes (x_min/.../z_max)."""
    mesh = meshdata_to_skfem(skfem_to_meshdata(_hex_mesh(3)))
    assert mesh.boundaries is not None
    for name in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        assert name in mesh.boundaries, f"{name} wall must be tagged on a hex box"


# ---------------------------------------------------------------------------
# (2) FEM solve path on a hex mesh: runs, finite, P2 beats P1
# ---------------------------------------------------------------------------


def _solve_poisson(mesh: skfem.Mesh, order: int) -> tuple[float, int]:
    r"""Solve $-\Delta u = f$ with Dirichlet data on $\partial\Omega$ for the
    manufactured solution $u(x,y,z) = \sin(\pi x)\sin(\pi y)\sin(\pi z)$
    (so $f = 3\pi^2 u$, $u = 0$ on the unit-cube boundary).

    Returns (max nodal error, n_dofs).
    """
    from scipy.sparse.linalg import spsolve

    basis = create_basis(mesh, order=order)
    x, y, z = basis.doflocs
    u_exact = np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
    f = 3.0 * np.pi**2 * u_exact

    K = assemble_stiffness(basis)
    M = assemble_mass(basis)
    rhs = M @ f  # consistent FEM load vector

    boundary_dofs = np.unique(basis.get_dofs().flatten())
    interior = np.setdiff1d(np.arange(basis.N), boundary_dofs)

    u = np.zeros(basis.N)
    u[boundary_dofs] = u_exact[boundary_dofs]  # homogeneous here, but explicit
    rhs_int = (rhs - K @ u)[interior]
    u[interior] = spsolve(K[interior][:, interior], rhs_int)

    return float(np.max(np.abs(u - u_exact))), int(basis.N)


def test_hex_fem_poisson_solve_runs_and_is_finite():
    """The FEM solve path runs end-to-end on a hex mesh with a finite result."""
    mesh = _hex_mesh(6)
    err_p1, n_p1 = _solve_poisson(mesh, order=1)
    assert np.isfinite(err_p1)
    assert n_p1 == 6**3
    # P1 on a 5x5x5-element cube should be at least roughly accurate.
    assert err_p1 < 0.1, f"hex P1 Poisson error unexpectedly large: {err_p1}"


def test_hex_fem_p2_more_accurate_than_p1():
    """Q2 must be genuinely more accurate than Q1 on the same hex mesh."""
    mesh = _hex_mesh(6)
    err_p1, _ = _solve_poisson(mesh, order=1)
    err_p2, n_p2 = _solve_poisson(mesh, order=2)
    assert np.isfinite(err_p2)
    assert n_p2 > 6**3, "P2 must introduce extra DOFs beyond the P1 vertices"
    assert err_p2 < err_p1, f"P2 ({err_p2}) should beat P1 ({err_p1})"

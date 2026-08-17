"""
Integration test for FEM-based MFG solving (#773).

Tests complete MFG solve on unstructured triangular mesh using
HJBFEMSolver + FPFEMSolver via the factory.
"""

from __future__ import annotations

import pytest

import numpy as np

skfem = pytest.importorskip("skfem", reason="scikit-fem required for FEM tests")

from mfgarchon.alg.numerical.fem import meshdata_to_skfem  # noqa: E402
from mfgarchon.alg.numerical.fem.assembly import assemble_mass, assemble_stiffness, create_basis  # noqa: E402
from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata  # noqa: E402


@pytest.mark.integration
class TestMeshAdapter:
    """Test MeshData <-> skfem.Mesh conversion."""

    def test_roundtrip_triangle(self):
        mesh = skfem.MeshTri.init_symmetric()
        md = skfem_to_meshdata(mesh)
        mesh2 = meshdata_to_skfem(md)

        assert mesh2.p.shape == mesh.p.shape
        assert mesh2.t.shape == mesh.t.shape
        np.testing.assert_allclose(mesh2.p, mesh.p)

    def test_roundtrip_refined(self):
        mesh = skfem.MeshTri.init_sqsymmetric().refined(2)
        md = skfem_to_meshdata(mesh)
        mesh2 = meshdata_to_skfem(md)

        assert mesh2.p.shape == mesh.p.shape
        assert mesh2.t.shape == mesh.t.shape
        np.testing.assert_allclose(mesh2.p, mesh.p)
        np.testing.assert_array_equal(mesh2.t, mesh.t, err_msg="Refined connectivity renumbered by round-trip")


@pytest.mark.integration
class TestFEMAssembly:
    """Test FEM assembly produces valid matrices."""

    def test_stiffness_symmetric(self):
        mesh = skfem.MeshTri.init_sqsymmetric().refined(1)
        basis = create_basis(mesh, order=1)
        K = assemble_stiffness(basis)

        assert K.shape[0] == K.shape[1]
        np.testing.assert_allclose((K - K.T).data, 0.0, atol=1e-12, err_msg="Stiffness not symmetric")

    def test_mass_positive_definite(self):
        mesh = skfem.MeshTri.init_sqsymmetric().refined(1)
        basis = create_basis(mesh, order=1)
        M = assemble_mass(basis)

        # Mass matrix should have positive diagonal
        assert np.all(M.diagonal() > 0)

        # Positive diagonal is necessary, not sufficient: assert the property the test is named for.
        # Measured lambda_min = 7.13e-3 against lambda_max = 5.82e-2, i.e. 14 orders above the
        # eigensolver noise floor eps*lambda_max ~ 1.3e-17.
        ev = np.linalg.eigvalsh(M.toarray())
        assert ev.min() > 0, f"mass matrix not positive definite: lambda_min={ev.min():.3e}"

    def test_mass_integrates_to_area(self):
        mesh = skfem.MeshTri.init_sqsymmetric().refined(1)
        basis = create_basis(mesh, order=1)
        M = assemble_mass(basis)

        # sum(M @ ones) = integral(1) = area of domain = 1.0 for unit square
        total = M @ np.ones(basis.N)
        np.testing.assert_allclose(total.sum(), 1.0, atol=1e-12)


@pytest.mark.integration
class TestFEMHeatEquation:
    """Test FEM solvers on a pure diffusion (heat equation) problem."""

    def test_backward_heat_converges(self):
        """Backward heat equation should smooth and remain finite."""
        mesh = skfem.MeshTri.init_sqsymmetric().refined(2)
        basis = create_basis(mesh, order=1)
        K = assemble_stiffness(basis)
        M = assemble_mass(basis)

        N = basis.N
        dt = 0.01
        D = 0.1

        A = M / dt + D * K
        boundary = mesh.boundary_nodes()
        interior = np.setdiff1d(np.arange(N), boundary)

        # Terminal: bump at center
        x, y = mesh.p
        u = np.exp(-20 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))

        for _ in range(5):
            rhs = (M / dt) @ u
            u_new = np.zeros(N)
            from scipy.sparse.linalg import spsolve

            u_new[interior] = spsolve(A[np.ix_(interior, interior)], rhs[interior])
            u = u_new

        assert np.all(np.isfinite(u))
        assert u.max() < 1.0  # Should have diffused

    def test_forward_heat_conserves_mass(self):
        """Forward heat equation should conserve mass."""
        mesh = skfem.MeshTri.init_sqsymmetric().refined(2)
        basis = create_basis(mesh, order=1)
        K = assemble_stiffness(basis)
        M = assemble_mass(basis)

        dt = 0.01
        D = 0.1

        A = M / dt + D * K

        x, y = mesh.p
        m = np.exp(-20 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))

        initial_mass = (M @ m).sum()

        from scipy.sparse.linalg import spsolve

        for _ in range(10):
            rhs = (M / dt) @ m
            m = spsolve(A, rhs)
            m = np.maximum(m, 0.0)

        final_mass = (M @ m).sum()
        rel_error = abs(final_mass - initial_mass) / initial_mass

        assert rel_error < 0.01, f"Mass conservation error: {rel_error:.2%}"


@pytest.mark.integration
class TestFEMFactoryIntegration:
    """Test that FEM solvers are wired into the scheme factory."""

    def test_fem_recommended_for_unstructured(self):
        """Auto-selection should recommend FEM for unstructured meshes."""
        from mfgarchon.factory.scheme_factory import get_recommended_scheme
        from mfgarchon.geometry.protocol import GeometryType

        # Create a mock problem with unstructured geometry
        class MockGeometry:
            geometry_type = GeometryType.UNSTRUCTURED_MESH

        class MockProblem:
            geometry = MockGeometry()

        scheme = get_recommended_scheme(MockProblem())

        from mfgarchon.types.schemes import NumericalScheme

        assert scheme == NumericalScheme.FEM_P1


if __name__ == "__main__":
    print("Testing FEM MFG integration...")

    # Mesh adapter
    mesh = skfem.MeshTri.init_sqsymmetric().refined(2)
    md = skfem_to_meshdata(mesh)
    mesh2 = meshdata_to_skfem(md)
    assert mesh2.p.shape == mesh.p.shape
    print(f"Mesh adapter: {mesh.p.shape[1]} nodes, round-trip OK")

    # Assembly
    basis = create_basis(mesh, order=1)
    K = assemble_stiffness(basis)
    M = assemble_mass(basis)
    print(f"Assembly: K {K.shape}, M {M.shape}")

    # Heat equation
    N = basis.N
    dt = 0.01
    D = 0.1
    x, y = mesh.p
    m = np.exp(-20 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
    initial_mass = (M @ m).sum()

    A = M / dt + D * K
    from scipy.sparse.linalg import spsolve

    for _ in range(10):
        m = spsolve(A, (M / dt) @ m)
        m = np.maximum(m, 0.0)

    final_mass = (M @ m).sum()
    print(f"Mass conservation: {abs(final_mass - initial_mass) / initial_mass:.2e}")

    print("All smoke tests passed.")

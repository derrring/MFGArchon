"""
Unit tests for sparse matrix operations.

Tests sparse linear algebra utilities for 1D, 2D, and 3D grids.
"""

import pytest

import numpy as np
import scipy.sparse as sp

from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid
from mfgarchon.utils.sparse_operations import SparseMatrixBuilder, SparseSolver, estimate_sparsity, sparse_matmul


class TestSparseMatrixBuilder:
    """Test sparse matrix construction from finite difference stencils."""

    def test_laplacian_1d(self):
        """Test 1D Laplacian construction."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
        builder = SparseMatrixBuilder(grid, matrix_format="csr")

        L = builder.build_laplacian(boundary_conditions="dirichlet")

        # Check shape and sparsity
        assert L.shape == (11, 11)
        assert L.nnz <= 3 * 11  # Tridiagonal structure

        # Check interior stencil
        dx = 0.1
        expected_center = 2.0 / dx**2
        expected_off = -1.0 / dx**2

        # Point i=5 (interior)
        row = L.getrow(5).toarray().flatten()
        assert np.isclose(row[4], expected_off)
        assert np.isclose(row[5], expected_center)
        assert np.isclose(row[6], expected_off)

        # Boundary points should have identity
        assert L[0, 0] == 1.0
        assert L[10, 10] == 1.0

    def test_laplacian_2d(self):
        """Test 2D Laplacian with 5-point stencil."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[11, 11],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        builder = SparseMatrixBuilder(grid, matrix_format="csr")

        L = builder.build_laplacian(boundary_conditions="dirichlet")

        # Check shape
        N = 11 * 11
        assert L.shape == (N, N)

        # Check sparsity (5-point stencil)
        max_nnz = 5 * N  # At most 5 nonzeros per row
        assert L.nnz <= max_nnz

        # Check interior point stencil
        dx = dy = 0.1
        expected_center = -2.0 * (1 / dx**2 + 1 / dy**2)

        # Point (5,5) -> flat index
        idx = grid.get_index((5, 5))
        row = L.getrow(idx).toarray().flatten()

        # Should have 5 nonzeros
        nonzero_count = np.count_nonzero(row)
        assert nonzero_count == 5

        # Check center value
        assert np.isclose(row[idx], expected_center)

    def test_laplacian_3d(self):
        """Test 3D Laplacian with 7-point stencil."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            Nx_points=[6, 6, 6],
            boundary_conditions=no_flux_bc(dimension=3),
        )
        builder = SparseMatrixBuilder(grid, matrix_format="csr")

        L = builder.build_laplacian(boundary_conditions="dirichlet")

        # Check shape
        N = 6 * 6 * 6
        assert L.shape == (N, N)

        # Check sparsity (7-point stencil)
        max_nnz = 7 * N
        assert L.nnz <= max_nnz

        # Check interior point
        idx = grid.get_index((3, 3, 3))
        row = L.getrow(idx).toarray().flatten()

        # Interior point should have 7 nonzeros
        nonzero_count = np.count_nonzero(row)
        assert nonzero_count == 7

    def test_gradient_1d(self):
        """Test 1D gradient matrix."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
        builder = SparseMatrixBuilder(grid, matrix_format="csr")

        # Central difference
        G = builder.build_gradient(direction=0, order=2)

        assert G.shape == (11, 11)

        # Test on linear function u(x) = 2x
        x = grid.coordinates[0]
        u = 2 * x
        du_dx = G @ u

        # Should be approximately 2 everywhere (except boundaries)
        assert np.allclose(du_dx[1:-1], 2.0, atol=1e-10)

    def test_gradient_2d(self):
        """Test 2D gradient matrices."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[11, 11],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        builder = SparseMatrixBuilder(grid, matrix_format="csr")

        # Gradient in x-direction
        Gx = builder.build_gradient(direction=0, order=2)

        # Test on u(x,y) = x^2
        X, _Y = grid.meshgrid()
        u = (X**2).flatten()
        du_dx = Gx @ u

        # Should be approximately 2x (interior points)
        du_dx_2d = du_dx.reshape(11, 11)
        # Check central region
        expected = 2 * X[3:8, 3:8]
        actual = du_dx_2d[3:8, 3:8]
        assert np.allclose(actual, expected, atol=1e-1)

    def test_format_conversion(self):
        """Test building matrices in different sparse formats."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))

        for fmt in ["csr", "csc", "lil"]:
            builder = SparseMatrixBuilder(grid, matrix_format=fmt)
            L = builder.build_laplacian()

            # Check format
            if fmt == "csr":
                assert sp.isspmatrix_csr(L)
            elif fmt == "csc":
                assert sp.isspmatrix_csc(L)
            elif fmt == "lil":
                assert sp.isspmatrix_lil(L)


class TestSparseSolver:
    """Test sparse linear system solvers."""

    def test_direct_solver_1d(self):
        """Test direct solver on 1D Poisson equation."""
        # Solve: -u'' = f with u(0) = u(1) = 0
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        builder = SparseMatrixBuilder(grid, matrix_format="csr")

        L = builder.build_laplacian(boundary_conditions="dirichlet")

        # Right-hand side: f(x) = sin(πx)
        x = grid.coordinates[0]
        f = np.sin(np.pi * x)

        # Exact solution: u(x) = sin(πx) / π²
        u_exact = np.sin(np.pi * x) / (np.pi**2)

        # Solve
        solver = SparseSolver(method="direct", backend="scipy")
        u_numerical = solver.solve(L, f)

        # Compare (with relaxed tolerance for boundary effects)
        assert np.allclose(u_numerical, u_exact, atol=1e-3)

    def test_iterative_solver_cg(self):
        """Test CG iterative solver."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[101], boundary_conditions=no_flux_bc(dimension=1))
        builder = SparseMatrixBuilder(grid, matrix_format="csr")

        # Symmetric positive definite system
        L = builder.build_laplacian(boundary_conditions="dirichlet")

        x = grid.coordinates[0]
        b = np.sin(np.pi * x)

        solver = SparseSolver(method="cg", backend="scipy", tol=1e-8, max_iter=1000)
        u = solver.solve(L, b)

        # Check residual
        residual = np.linalg.norm(L @ u - b)
        assert residual < 1e-6

    def test_iterative_solver_gmres(self):
        """Test GMRES iterative solver."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[21, 21],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        builder = SparseMatrixBuilder(grid, matrix_format="csr")

        L = builder.build_laplacian(boundary_conditions="dirichlet")

        # Random right-hand side
        np.random.seed(42)
        b = np.random.randn(grid.total_points())

        solver = SparseSolver(method="gmres", backend="scipy", tol=1e-8, max_iter=1000, preconditioner="ilu")
        u = solver.solve(L, b)

        # Check residual
        residual = np.linalg.norm(L @ u - b)
        assert residual < 1e-6

    def test_solver_convergence_callback(self):
        """Test solver callback for monitoring convergence."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        builder = SparseMatrixBuilder(grid, matrix_format="csr")

        L = builder.build_laplacian(boundary_conditions="dirichlet")
        b = np.ones(grid.total_points())

        residuals = []

        def callback(pr_norm):
            # GMRES callback receives preconditioned residual norm, not solution vector
            residuals.append(pr_norm)

        solver = SparseSolver(method="gmres", backend="scipy", tol=1e-8)
        solver.solve(L, b, callback=callback)

        # Check that residuals were recorded
        # (May converge in 1 iteration for simple problems)
        assert len(residuals) > 0
        if len(residuals) > 1:
            assert residuals[-1] <= residuals[0]


class TestSparseUtilities:
    """Test sparse matrix utility functions."""

    def test_sparse_matmul(self):
        """Test sparse matrix multiplication."""
        # Create simple sparse matrices
        A = sp.csr_matrix([[1, 2, 0], [0, 3, 4], [5, 0, 6]])
        B = sp.csr_matrix([[1, 0], [0, 1], [1, 1]])

        # Sparse-sparse multiplication
        C = sparse_matmul(A, B, output_format="csr")
        assert sp.isspmatrix_csr(C)
        assert C.shape == (3, 2)

        # Sparse-dense multiplication
        B_dense = np.array([[1, 0], [0, 1], [1, 1]])
        C_dense = sparse_matmul(A, B_dense)
        assert isinstance(C_dense, np.ndarray)
        assert C_dense.shape == (3, 2)

    def test_estimate_sparsity(self):
        """Test sparsity analysis."""
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[50, 50],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        builder = SparseMatrixBuilder(grid, matrix_format="csr")

        L = builder.build_laplacian()

        stats = estimate_sparsity(L)

        # Check metrics
        assert "nnz" in stats
        assert "density" in stats
        assert "memory_mb" in stats
        assert "bandwidth" in stats

        # Verify values
        assert stats["nnz"] > 0
        assert 0 < stats["density"] < 1
        assert stats["memory_mb"] > 0

        # For 2D Laplacian, density should be very low
        assert stats["density"] < 0.002  # Less than 0.2%


class TestIntegrationWithGrid:
    """Integration tests with tensor product grids."""

    def test_poisson_2d(self):
        """Solve 2D Poisson equation: -Δu = f."""
        # Domain: [0,1] × [0,1]
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[31, 31],
            boundary_conditions=no_flux_bc(dimension=2),
        )

        builder = SparseMatrixBuilder(grid, matrix_format="csr")
        L = builder.build_laplacian(boundary_conditions="dirichlet")

        # Simple test: constant source term in interior
        # -Δu = 1 with u(boundary) = 0
        f = np.ones(grid.Nx_points)
        # Set boundary to zero (Dirichlet BC)
        f[0, :] = 0
        f[-1, :] = 0
        f[:, 0] = 0
        f[:, -1] = 0
        f_flat = f.flatten()

        # Solve
        solver = SparseSolver(method="direct", backend="scipy")
        u_flat = solver.solve(-L, f_flat)  # Note: -L for proper Poisson equation
        u_numerical = u_flat.reshape(grid.Nx_points)

        # Check properties:
        # 1. Boundary conditions
        assert np.allclose(u_numerical[0, :], 0)
        assert np.allclose(u_numerical[-1, :], 0)
        assert np.allclose(u_numerical[:, 0], 0)
        assert np.allclose(u_numerical[:, -1], 0)

        # 2. Solution should be positive in interior
        assert np.all(u_numerical[1:-1, 1:-1] > 0)

        # 3. Maximum at center (by symmetry)
        center = (15, 15)
        assert u_numerical[center] == np.max(u_numerical)

    def test_grid_refinement_convergence(self):
        """Test convergence with grid refinement."""
        errors = []

        for N in [11, 21, 41]:
            grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[N], boundary_conditions=no_flux_bc(dimension=1))
            builder = SparseMatrixBuilder(grid, matrix_format="csr")

            L = builder.build_laplacian(boundary_conditions="dirichlet")

            x = grid.coordinates[0]
            f = np.sin(np.pi * x)
            u_exact = np.sin(np.pi * x) / (np.pi**2)

            solver = SparseSolver(method="direct")
            u_numerical = solver.solve(L, f)

            error = np.max(np.abs(u_numerical - u_exact))
            errors.append(error)

        # Errors should decrease with refinement
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]

        # Check convergence rate (should be O(h²))
        ratio = errors[0] / errors[1]
        assert 3 < ratio < 5  # Roughly 4 for second-order method


@pytest.mark.skipif(
    True,  # Always skip unless GPU available
    reason="CuPy GPU tests require CUDA installation",
)
class TestGPUSolver:
    """Test GPU-accelerated solvers (requires CuPy)."""

    def test_cupy_solver(self):
        """Test CuPy GPU solver."""
        try:
            import cupy as cp  # noqa: F401
        except ImportError:
            pytest.skip("CuPy not available")

        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[51, 51],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        builder = SparseMatrixBuilder(grid, matrix_format="csr")

        L = builder.build_laplacian()
        b = np.random.randn(grid.total_points())

        solver = SparseSolver(method="cg", backend="cupy", tol=1e-8)
        u = solver.solve(L, b)

        # Check residual
        residual = np.linalg.norm(L @ u - b)
        assert residual < 1e-6

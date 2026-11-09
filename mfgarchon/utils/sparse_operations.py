"""
Sparse matrix operations for efficient large-scale MFG problems.

This module provides optimized sparse linear algebra operations for 2D/3D MFG problems,
supporting both CPU (SciPy) and optional GPU acceleration (CuPy/JAX).

Mathematical Background:
    Large-scale MFG problems lead to sparse linear systems:
        A u = b    (HJB equation discretization)

    For 2D problems with Nx × Ny grid points:
        - System size: N = Nx · Ny (e.g., 100×100 = 10,000)
        - Dense storage: O(N²) ≈ 100 MB for float64
        - Sparse storage: O(5N) ≈ 400 KB (5-point stencil)

    For 3D problems with Nx × Ny × Nz:
        - System size: N = Nx · Ny · Nz (e.g., 50×50×50 = 125,000)
        - Dense storage: O(N²) ≈ 125 GB (infeasible!)
        - Sparse storage: O(7N) ≈ 7 MB (7-point stencil)

Sparse Matrix Formats:
    - **CSR (Compressed Sparse Row)**: Optimal for matrix-vector products
    - **CSC (Compressed Sparse Column)**: Optimal for column operations
    - **LIL (List of Lists)**: Efficient for incremental construction
    - **DOK (Dictionary of Keys)**: Efficient for random access during construction

References:
    - Saad (2003): Iterative Methods for Sparse Linear Systems
    - Davis (2006): Direct Methods for Sparse Linear Systems
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid


class SparseMatrixBuilder:
    """
    Builder for constructing sparse matrices from finite difference stencils.

    Provides efficient methods for building sparse linear systems arising from
    discretization of differential operators on structured grids.

    Attributes:
        grid: Tensor product grid defining the spatial discretization
        format: Target sparse matrix format (csr, csc, lil)
        dtype: Data type for matrix entries (default: float64)

    Example:
        >>> from mfgarchon.geometry import TensorProductGrid
        >>> grid = TensorProductGrid(2, [(0,1), (0,1)], [101, 101])
        >>> builder = SparseMatrixBuilder(grid, format="csr")
        >>> # Build Laplacian matrix
        >>> laplacian = builder.build_laplacian(coefficients=1.0)
        >>> # laplacian is (10201, 10201) sparse matrix with ~50K nonzeros
    """

    def __init__(
        self,
        grid: TensorProductGrid,
        matrix_format: Literal["csr", "csc", "lil", "dok"] = "csr",
        dtype: type = np.float64,
    ):
        """
        Initialize sparse matrix builder.

        Args:
            grid: Tensor product grid for spatial discretization
            matrix_format: Target sparse format (csr/csc/lil/dok)
            dtype: Data type for matrix entries

        Raises:
            ValueError: If unsupported sparse format requested
        """
        if matrix_format not in ["csr", "csc", "lil", "dok"]:
            raise ValueError(f"Unsupported sparse format: {matrix_format}")

        self.grid = grid
        self.format = matrix_format
        self.dtype = dtype
        self.N = grid.total_points()

    def build_laplacian(
        self,
        coefficients: float | NDArray | None = None,
        boundary_conditions: str = "dirichlet",
    ) -> sp.spmatrix:
        """
        Build sparse Laplacian matrix for diffusion operator.

        Discretizes: -∇·(D∇u) where D is diffusion coefficient

        For uniform grid with spacing h:
            1D: (-u[i-1] + 2u[i] - u[i+1]) / h²
            2D: 5-point stencil with weights [-1, -1, 4, -1, -1] / h²
            3D: 7-point stencil with weights [-1, -1, -1, 6, -1, -1, -1] / h²

        Args:
            coefficients: Diffusion coefficient D(x) (scalar, array, or None=1.0)
            boundary_conditions: Boundary treatment (dirichlet/neumann)

        Returns:
            Sparse matrix L of shape (N, N) representing -Δ operator

        Raises:
            NotImplementedError: If grid dimension > 3
        """
        if self.grid.dimension == 1:
            return self._build_laplacian_1d(coefficients, boundary_conditions)
        elif self.grid.dimension == 2:
            return self._build_laplacian_2d(coefficients, boundary_conditions)
        elif self.grid.dimension == 3:
            return self._build_laplacian_3d(coefficients, boundary_conditions)
        else:
            # Use generalized nD implementation for d >= 4
            return self._build_laplacian_nd(coefficients, boundary_conditions)

    def _build_laplacian_1d(
        self,
        coefficients: float | NDArray | None,
        boundary_conditions: str,
    ) -> sp.spmatrix:
        """Build 1D Laplacian with 3-point stencil."""
        nx = self.grid.Nx_points[0]
        dx = self.grid.spacing[0]

        # Create tridiagonal matrix
        diagonals = [
            -np.ones(nx - 1) / dx**2,  # Lower diagonal
            2 * np.ones(nx) / dx**2,  # Main diagonal
            -np.ones(nx - 1) / dx**2,  # Upper diagonal
        ]
        offsets = [-1, 0, 1]

        L = sp.diags(diagonals, offsets, shape=(nx, nx), format="lil", dtype=self.dtype)

        # Apply boundary conditions
        if boundary_conditions == "dirichlet":
            # Zero boundary: u[0] = u[nx-1] = 0
            L[0, :] = 0
            L[0, 0] = 1
            L[-1, :] = 0
            L[-1, -1] = 1
        elif boundary_conditions == "neumann":
            # Zero-flux: ∂u/∂x = 0 at boundaries
            L[0, 1] = -2 / dx**2
            L[-1, -2] = -2 / dx**2

        return L.asformat(self.format)

    def _build_laplacian_2d(
        self,
        coefficients: float | NDArray | None,
        boundary_conditions: str,
    ) -> sp.spmatrix:
        """Build 2D Laplacian with 5-point stencil."""
        nx, ny = self.grid.Nx_points
        dx, dy = self.grid.spacing

        # Build using LIL format for efficient construction
        L = sp.lil_matrix((self.N, self.N), dtype=self.dtype)

        # 5-point stencil coefficients
        coeff_x = 1.0 / dx**2
        coeff_y = 1.0 / dy**2
        coeff_center = -2.0 * (coeff_x + coeff_y)

        # Fill matrix using grid indexing
        for i in range(nx):
            for j in range(ny):
                idx = self.grid.get_index((i, j))

                # Interior points: 5-point stencil
                if i > 0 and i < nx - 1 and j > 0 and j < ny - 1:
                    L[idx, idx] = coeff_center
                    L[idx, self.grid.get_index((i - 1, j))] = coeff_x  # Left
                    L[idx, self.grid.get_index((i + 1, j))] = coeff_x  # Right
                    L[idx, self.grid.get_index((i, j - 1))] = coeff_y  # Down
                    L[idx, self.grid.get_index((i, j + 1))] = coeff_y  # Up

                # Boundary points
                elif boundary_conditions == "dirichlet":
                    L[idx, idx] = 1.0
                elif boundary_conditions == "neumann":
                    # Simplified Neumann: zero-flux approximation
                    L[idx, idx] = 1.0

        return L.asformat(self.format)

    def _build_laplacian_3d(
        self,
        coefficients: float | NDArray | None,
        boundary_conditions: str,
    ) -> sp.spmatrix:
        """Build 3D Laplacian with 7-point stencil."""
        nx, ny, nz = self.grid.Nx_points
        dx, dy, dz = self.grid.spacing

        L = sp.lil_matrix((self.N, self.N), dtype=self.dtype)

        # 7-point stencil coefficients
        coeff_x = 1.0 / dx**2
        coeff_y = 1.0 / dy**2
        coeff_z = 1.0 / dz**2
        coeff_center = -2.0 * (coeff_x + coeff_y + coeff_z)

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    idx = self.grid.get_index((i, j, k))

                    # Interior points
                    if i > 0 and i < nx - 1 and j > 0 and j < ny - 1 and k > 0 and k < nz - 1:
                        L[idx, idx] = coeff_center
                        L[idx, self.grid.get_index((i - 1, j, k))] = coeff_x
                        L[idx, self.grid.get_index((i + 1, j, k))] = coeff_x
                        L[idx, self.grid.get_index((i, j - 1, k))] = coeff_y
                        L[idx, self.grid.get_index((i, j + 1, k))] = coeff_y
                        L[idx, self.grid.get_index((i, j, k - 1))] = coeff_z
                        L[idx, self.grid.get_index((i, j, k + 1))] = coeff_z

                    # Boundary treatment
                    elif boundary_conditions == "dirichlet" or boundary_conditions == "neumann":
                        L[idx, idx] = 1.0

        return L.asformat(self.format)

    def _build_laplacian_nd(
        self,
        coefficients: float | NDArray | None,
        boundary_conditions: str,
    ) -> sp.spmatrix:
        """Build nD Laplacian with (2n+1)-point stencil for arbitrary dimension.

        Generalizes 1D/2D/3D Laplacian to n dimensions using tensor product structure.
        Uses (2n+1)-point stencil: center point + 2 neighbors per dimension.

        Mathematical formulation:
            Laplacian(u) = sum_d (d^2u/dx_d^2)
            Discrete: L[i] = -2*sum_d(1/dx_d^2) * u[i] + sum_d(1/dx_d^2)*(u[i-e_d] + u[i+e_d])

        Args:
            coefficients: Diffusion coefficient D(x) (currently uniform only)
            boundary_conditions: "dirichlet" or "neumann"

        Returns:
            Sparse matrix L of shape (N, N) representing -Laplacian operator
        """
        ndim = self.grid.dimension
        num_points = self.grid.Nx_points
        spacing = self.grid.spacing

        L = sp.lil_matrix((self.N, self.N), dtype=self.dtype)

        # Stencil coefficients: 1/dx_d^2 for each dimension
        coeffs = [1.0 / (spacing[d] ** 2) for d in range(ndim)]
        coeff_center = -2.0 * sum(coeffs)

        # Iterate over all grid points using itertools.product
        for multi_idx in itertools.product(*[range(n) for n in num_points]):
            idx = self.grid.get_index(multi_idx)

            # Check if interior point (all indices > 0 and < n_d - 1)
            is_interior = all(0 < multi_idx[d] < num_points[d] - 1 for d in range(ndim))

            if is_interior:
                # Center coefficient
                L[idx, idx] = coeff_center

                # Off-diagonal entries: neighbors in each dimension
                for d in range(ndim):
                    # Neighbor in -d direction
                    neighbor_minus = list(multi_idx)
                    neighbor_minus[d] -= 1
                    L[idx, self.grid.get_index(tuple(neighbor_minus))] = coeffs[d]

                    # Neighbor in +d direction
                    neighbor_plus = list(multi_idx)
                    neighbor_plus[d] += 1
                    L[idx, self.grid.get_index(tuple(neighbor_plus))] = coeffs[d]

            # Boundary treatment
            elif boundary_conditions in ("dirichlet", "neumann"):
                L[idx, idx] = 1.0

        return L.asformat(self.format)

    def build_gradient(self, direction: int = 0, order: int = 2) -> sp.spmatrix:
        """
        Build sparse gradient matrix for spatial derivative.

        Args:
            direction: Spatial direction (0=x, 1=y, 2=z)
            order: Finite difference order (1=forward, 2=central)

        Returns:
            Sparse matrix G representing ∂/∂xᵢ operator

        Raises:
            ValueError: If direction >= grid dimension
        """
        if direction >= self.grid.dimension:
            raise ValueError(f"Direction {direction} invalid for {self.grid.dimension}D grid")

        if self.grid.dimension == 1:
            return self._build_gradient_1d(order)
        elif self.grid.dimension == 2:
            return self._build_gradient_2d(direction, order)
        elif self.grid.dimension == 3:
            return self._build_gradient_3d(direction, order)
        else:
            # Use generalized nD implementation for d >= 4
            return self._build_gradient_nd(direction, order)

    def _build_gradient_1d(self, order: int) -> sp.spmatrix:
        """Build 1D gradient matrix."""
        nx = self.grid.Nx_points[0]
        dx = self.grid.spacing[0]

        if order == 1:
            # Forward difference: (u[i+1] - u[i]) / dx
            diagonals = [-np.ones(nx) / dx, np.ones(nx - 1) / dx]
            offsets = [0, 1]
        else:
            # Central difference: (u[i+1] - u[i-1]) / (2dx)
            diagonals = [-np.ones(nx - 1) / (2 * dx), np.ones(nx - 1) / (2 * dx)]
            offsets = [-1, 1]

        return sp.diags(diagonals, offsets, shape=(nx, nx), format=self.format, dtype=self.dtype)

    def _build_gradient_2d(self, direction: int, order: int) -> sp.spmatrix:
        """Build 2D gradient matrix along specified direction."""
        nx, ny = self.grid.Nx_points
        dx, dy = self.grid.spacing

        G = sp.lil_matrix((self.N, self.N), dtype=self.dtype)

        h = dx if direction == 0 else dy

        for i in range(nx):
            for j in range(ny):
                idx = self.grid.get_index((i, j))

                if direction == 0:  # ∂/∂x
                    if order == 2 and i > 0 and i < nx - 1:
                        G[idx, self.grid.get_index((i - 1, j))] = -1 / (2 * h)
                        G[idx, self.grid.get_index((i + 1, j))] = 1 / (2 * h)
                    elif i < nx - 1:
                        G[idx, idx] = -1 / h
                        G[idx, self.grid.get_index((i + 1, j))] = 1 / h
                else:  # ∂/∂y
                    if order == 2 and j > 0 and j < ny - 1:
                        G[idx, self.grid.get_index((i, j - 1))] = -1 / (2 * h)
                        G[idx, self.grid.get_index((i, j + 1))] = 1 / (2 * h)
                    elif j < ny - 1:
                        G[idx, idx] = -1 / h
                        G[idx, self.grid.get_index((i, j + 1))] = 1 / h

        return G.asformat(self.format)

    def _build_gradient_3d(self, direction: int, order: int) -> sp.spmatrix:
        """Build 3D gradient matrix along specified direction."""
        nx, ny, nz = self.grid.Nx_points
        dx, dy, dz = self.grid.spacing

        G = sp.lil_matrix((self.N, self.N), dtype=self.dtype)

        h = [dx, dy, dz][direction]

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    idx = self.grid.get_index((i, j, k))

                    # Central or forward difference based on order and position
                    if direction == 0:  # ∂/∂x
                        if order == 2 and i > 0 and i < nx - 1:
                            G[idx, self.grid.get_index((i - 1, j, k))] = -1 / (2 * h)
                            G[idx, self.grid.get_index((i + 1, j, k))] = 1 / (2 * h)
                        elif i < nx - 1:
                            G[idx, idx] = -1 / h
                            G[idx, self.grid.get_index((i + 1, j, k))] = 1 / h
                    elif direction == 1:  # ∂/∂y
                        if order == 2 and j > 0 and j < ny - 1:
                            G[idx, self.grid.get_index((i, j - 1, k))] = -1 / (2 * h)
                            G[idx, self.grid.get_index((i, j + 1, k))] = 1 / (2 * h)
                        elif j < ny - 1:
                            G[idx, idx] = -1 / h
                            G[idx, self.grid.get_index((i, j + 1, k))] = 1 / h
                    else:  # ∂/∂z
                        if order == 2 and k > 0 and k < nz - 1:
                            G[idx, self.grid.get_index((i, j, k - 1))] = -1 / (2 * h)
                            G[idx, self.grid.get_index((i, j, k + 1))] = 1 / (2 * h)
                        elif k < nz - 1:
                            G[idx, idx] = -1 / h
                            G[idx, self.grid.get_index((i, j, k + 1))] = 1 / h

        return G.asformat(self.format)

    def _build_gradient_nd(self, direction: int, order: int) -> sp.spmatrix:
        """Build nD gradient matrix along specified direction.

        Generalizes 1D/2D/3D gradient to n dimensions using tensor product structure.

        Mathematical formulation:
            Gradient in direction d: du/dx_d
            Central (order=2): (u[i+e_d] - u[i-e_d]) / (2*h_d)
            Forward (order=1): (u[i+e_d] - u[i]) / h_d

        Args:
            direction: Spatial direction (0, 1, 2, ...)
            order: Finite difference order (1=forward, 2=central)

        Returns:
            Sparse matrix G representing d/dx_d operator
        """
        num_points = self.grid.Nx_points
        spacing = self.grid.spacing

        G = sp.lil_matrix((self.N, self.N), dtype=self.dtype)

        h = spacing[direction]
        n_dir = num_points[direction]

        # Iterate over all grid points using itertools.product
        for multi_idx in itertools.product(*[range(n) for n in num_points]):
            idx = self.grid.get_index(multi_idx)
            i_dir = multi_idx[direction]  # Index in the gradient direction

            if order == 2:
                # Central difference: interior points only
                if 0 < i_dir < n_dir - 1:
                    # Neighbor in -direction
                    neighbor_minus = list(multi_idx)
                    neighbor_minus[direction] -= 1
                    G[idx, self.grid.get_index(tuple(neighbor_minus))] = -1 / (2 * h)

                    # Neighbor in +direction
                    neighbor_plus = list(multi_idx)
                    neighbor_plus[direction] += 1
                    G[idx, self.grid.get_index(tuple(neighbor_plus))] = 1 / (2 * h)
            else:
                # Forward difference: all but last point in direction
                if i_dir < n_dir - 1:
                    G[idx, idx] = -1 / h

                    neighbor_plus = list(multi_idx)
                    neighbor_plus[direction] += 1
                    G[idx, self.grid.get_index(tuple(neighbor_plus))] = 1 / h

        return G.asformat(self.format)


class SparseSolver:
    """
    Sparse linear system solver with multiple backend support.

    Provides unified interface to various sparse linear solvers:
    - Direct methods: SuperLU, UMFPACK, Cholesky
    - Iterative methods: CG, GMRES, BiCGSTAB
    - GPU acceleration: CuPy (optional)

    Example:
        >>> from mfgarchon.utils.sparse_operations import SparseSolver
        >>> solver = SparseSolver(method="gmres", backend="scipy")
        >>> x = solver.solve(A, b, tol=1e-8)
    """

    def __init__(
        self,
        method: Literal["direct", "cg", "gmres", "bicgstab"] = "gmres",
        backend: Literal["scipy", "cupy"] = "scipy",
        max_iter: int | None = None,
        tol: float = 1e-8,
        preconditioner: str | None = "ilu",
    ):
        """
        Initialize sparse solver.

        Args:
            method: Solver method (direct/cg/gmres/bicgstab)
            backend: Computation backend (scipy/cupy)
            max_iter: Maximum iterations for iterative methods
            tol: Convergence tolerance
            preconditioner: Preconditioner type (ilu/jacobi/None)
        """
        self.method = method
        self.backend = backend
        self.max_iter = max_iter
        self.tol = tol
        self.preconditioner = preconditioner

        # Check GPU availability if requested
        if backend == "cupy":
            try:
                import cupy as cp  # noqa: F401

                self.gpu_available = True
            except ImportError:
                import warnings

                warnings.warn("CuPy not available, falling back to SciPy backend")
                self.backend = "scipy"
                self.gpu_available = False
        else:
            self.gpu_available = False

    def solve(
        self,
        A: sp.spmatrix,
        b: NDArray,
        x0: NDArray | None = None,
        callback: Callable | None = None,
    ) -> NDArray:
        """
        Solve sparse linear system Ax = b.

        Args:
            A: Sparse coefficient matrix
            b: Right-hand side vector
            x0: Initial guess for iterative methods
            callback: Optional callback function(xk) called each iteration

        Returns:
            Solution vector x

        Raises:
            RuntimeError: If solver fails to converge
        """
        if self.backend == "cupy" and self.gpu_available:
            return self._solve_cupy(A, b, x0, callback)
        else:
            return self._solve_scipy(A, b, x0, callback)

    def _solve_scipy(
        self,
        A: sp.spmatrix,
        b: NDArray,
        x0: NDArray | None,
        callback: Callable | None,
    ) -> NDArray:
        """Solve using SciPy backend."""
        # Build preconditioner if requested
        M = None
        if self.preconditioner == "ilu" and self.method != "direct":
            try:
                M = spla.spilu(A.tocsc())
                M = spla.LinearOperator(A.shape, M.solve)
            except RuntimeError:
                import warnings

                warnings.warn("ILU preconditioner failed, proceeding without preconditioning")

        # Solve based on method
        if self.method == "direct":
            return spla.spsolve(A, b)

        elif self.method == "cg":
            x, info = spla.cg(A, b, x0=x0, rtol=self.tol, maxiter=self.max_iter, M=M, callback=callback)
            if info != 0:
                raise RuntimeError(f"CG solver failed with info={info}")
            return x

        elif self.method == "gmres":
            x, info = spla.gmres(A, b, x0=x0, rtol=self.tol, maxiter=self.max_iter, M=M, callback=callback)
            if info != 0:
                raise RuntimeError(f"GMRES solver failed with info={info}")
            return x

        elif self.method == "bicgstab":
            x, info = spla.bicgstab(A, b, x0=x0, rtol=self.tol, maxiter=self.max_iter, M=M, callback=callback)
            if info != 0:
                raise RuntimeError(f"BiCGSTAB solver failed with info={info}")
            return x

        else:
            raise ValueError(f"Unknown solver method: {self.method}")

    def _solve_cupy(
        self,
        A: sp.spmatrix,
        b: NDArray,
        x0: NDArray | None,
        callback: Callable | None,
    ) -> NDArray:
        """Solve using CuPy GPU backend."""
        import cupy as cp
        import cupyx.scipy.sparse as cpsp
        import cupyx.scipy.sparse.linalg as cpsla

        # Transfer to GPU
        A_gpu = cpsp.csr_matrix(A)
        b_gpu = cp.asarray(b)
        x0_gpu = cp.asarray(x0) if x0 is not None else None

        # Solve on GPU
        if self.method == "cg":
            x_gpu, info = cpsla.cg(A_gpu, b_gpu, x0=x0_gpu, rtol=self.tol, maxiter=self.max_iter)
        elif self.method == "gmres":
            x_gpu, info = cpsla.gmres(A_gpu, b_gpu, x0=x0_gpu, rtol=self.tol, maxiter=self.max_iter)
        else:
            raise ValueError(f"GPU backend does not support method: {self.method}")

        if info != 0:
            raise RuntimeError(f"GPU solver failed with info={info}")

        # Transfer back to CPU
        return cp.asnumpy(x_gpu)


def sparse_matmul(A: sp.spmatrix, B: sp.spmatrix | NDArray, output_format: str = "csr") -> sp.spmatrix | NDArray:
    """
    Efficient sparse matrix multiplication.

    Args:
        A: First sparse matrix
        B: Second matrix (sparse or dense)
        output_format: Output format for sparse result

    Returns:
        Product A @ B in requested format
    """
    if isinstance(B, np.ndarray):
        # Sparse-dense product (returns dense)
        return A @ B
    else:
        # Sparse-sparse product
        C = A @ B
        return C.asformat(output_format)


def estimate_sparsity(A: sp.spmatrix) -> dict[str, float]:
    """
    Analyze sparsity structure of matrix.

    Args:
        A: Sparse matrix to analyze

    Returns:
        Dictionary with sparsity metrics:
        - nnz: Number of nonzeros
        - density: Fraction of nonzero entries
        - memory_mb: Memory usage in megabytes
        - bandwidth: Matrix bandwidth
    """
    nnz = A.nnz
    total = A.shape[0] * A.shape[1]
    density = nnz / total if total > 0 else 0

    # Estimate memory (CSR format)
    memory_bytes = nnz * (A.dtype.itemsize + 4) + (A.shape[0] + 1) * 4
    memory_mb = memory_bytes / (1024**2)

    # Compute bandwidth (for CSR format)
    if sp.isspmatrix_csr(A):
        bandwidth = 0
        for i in range(A.shape[0]):
            row_indices = A.indices[A.indptr[i] : A.indptr[i + 1]]
            if len(row_indices) > 0:
                bandwidth = max(bandwidth, max(abs(row_indices - i)))
    else:
        bandwidth = None

    return {
        "nnz": nnz,
        "density": density,
        "memory_mb": memory_mb,
        "bandwidth": bandwidth,
    }

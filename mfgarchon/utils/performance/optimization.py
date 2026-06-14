"""
Performance Optimization Utilities for High-Dimensional MFG Problems

This module provides tools for optimizing performance of large-scale MFG computations,
including memory management, sparse operations, and parallel processing support.
"""

from __future__ import annotations

import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import psutil

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
from scipy.sparse import linalg as sp_linalg

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    import numba

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp  # noqa: F401

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Container for performance monitoring data."""

    operation_name: str
    start_time: float
    end_time: float
    memory_before: float
    memory_after: float
    memory_peak: float
    cpu_percent: float
    additional_metrics: dict[str, Any] = None

    @property
    def duration(self) -> float:
        """Operation duration in seconds."""
        return self.end_time - self.start_time

    @property
    def memory_used(self) -> float:
        """Memory used during operation in MB."""
        return self.memory_after - self.memory_before

    @property
    def memory_peak_delta(self) -> float:
        """Peak memory usage above initial in MB."""
        return self.memory_peak - self.memory_before


class PerformanceMonitor:
    """Performance monitoring and profiling for MFG computations."""

    def __init__(self) -> None:
        self.metrics_history: list[PerformanceMetrics] = []
        self.process = psutil.Process()

    @contextmanager
    def monitor_operation(self, operation_name: str, **additional_metrics):
        """Context manager for monitoring operation performance."""
        # Initial measurements
        start_time = time.time()
        memory_before = self._get_memory_usage()
        cpu_before = self.process.cpu_percent()

        peak_memory = memory_before

        try:
            yield self

            # Monitor peak memory during operation (simplified)
            peak_memory = max(peak_memory, self._get_memory_usage())

        finally:
            # Final measurements
            end_time = time.time()
            memory_after = self._get_memory_usage()
            cpu_after = self.process.cpu_percent()

            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_peak=peak_memory,
                cpu_percent=(cpu_before + cpu_after) / 2,
                additional_metrics=additional_metrics,
            )

            self.metrics_history.append(metrics)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def get_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        if not self.metrics_history:
            return {"message": "No operations monitored"}

        total_time = sum(m.duration for m in self.metrics_history)
        total_memory = sum(m.memory_used for m in self.metrics_history)
        peak_memory = max(m.memory_peak for m in self.metrics_history)

        return {
            "total_operations": len(self.metrics_history),
            "total_time": total_time,
            "average_time_per_operation": total_time / len(self.metrics_history),
            "total_memory_used": total_memory,
            "peak_memory": peak_memory,
            "operations": [
                {"name": m.operation_name, "duration": m.duration, "memory_used": m.memory_used}
                for m in self.metrics_history
            ],
        }

    def clear_history(self):
        """Clear performance history."""
        self.metrics_history.clear()


class SparseMatrixOptimizer:
    """Sparse matrix operations optimized for MFG problems."""

    @staticmethod
    def create_laplacian_3d(
        Nx: int | None = None,
        Ny: int | None = None,
        Nz: int | None = None,
        dx: float = 0.01,
        dy: float = 0.01,
        dz: float = 0.01,
    ) -> csr_matrix:
        """
        Create 3D discrete Laplacian operator using sparse matrices.

        Args:
            Nx: Number of grid points in x direction (preferred)
            Ny: Number of grid points in y direction (preferred)
            Nz: Number of grid points in z direction (preferred)
            dx: Grid spacing in x direction
            dy: Grid spacing in y direction
            dz: Grid spacing in z direction

        Returns:
            Sparse Laplacian matrix
        """
        # Require all grid dimensions
        if Nx is None or Ny is None or Nz is None:
            raise ValueError("Grid dimensions Nx, Ny, Nz must be provided")

        total_points = Nx * Ny * Nz

        # Pre-allocate arrays for sparse matrix construction
        row_indices = []
        col_indices = []
        data = []

        def index_3d(i: int, j: int, k: int) -> int:
            """Convert 3D indices to 1D index."""
            return i * Ny * Nz + j * Nz + k

        # Build Laplacian stencil
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    center_idx = index_3d(i, j, k)

                    # Center point coefficient
                    center_coeff = -2.0 * (1.0 / dx**2 + 1.0 / dy**2 + 1.0 / dz**2)
                    row_indices.append(center_idx)
                    col_indices.append(center_idx)
                    data.append(center_coeff)

                    # x-direction neighbors
                    if i > 0:
                        neighbor_idx = index_3d(i - 1, j, k)
                        row_indices.append(center_idx)
                        col_indices.append(neighbor_idx)
                        data.append(1.0 / dx**2)

                    if i < Nx - 1:
                        neighbor_idx = index_3d(i + 1, j, k)
                        row_indices.append(center_idx)
                        col_indices.append(neighbor_idx)
                        data.append(1.0 / dx**2)

                    # y-direction neighbors
                    if j > 0:
                        neighbor_idx = index_3d(i, j - 1, k)
                        row_indices.append(center_idx)
                        col_indices.append(neighbor_idx)
                        data.append(1.0 / dy**2)

                    if j < Ny - 1:
                        neighbor_idx = index_3d(i, j + 1, k)
                        row_indices.append(center_idx)
                        col_indices.append(neighbor_idx)
                        data.append(1.0 / dy**2)

                    # z-direction neighbors
                    if k > 0:
                        neighbor_idx = index_3d(i, j, k - 1)
                        row_indices.append(center_idx)
                        col_indices.append(neighbor_idx)
                        data.append(1.0 / dz**2)

                    if k < Nz - 1:
                        neighbor_idx = index_3d(i, j, k + 1)
                        row_indices.append(center_idx)
                        col_indices.append(neighbor_idx)
                        data.append(1.0 / dz**2)

        return csr_matrix((data, (row_indices, col_indices)), shape=(total_points, total_points))

    @staticmethod
    def create_gradient_operators_3d(
        nx: int, ny: int, nz: int, dx: float, dy: float, dz: float
    ) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
        """
        Create 3D discrete gradient operators.

        Returns:
            Tuple of (grad_x, grad_y, grad_z) sparse matrices
        """
        total_points = nx * ny * nz

        def index_3d(i: int, j: int, k: int) -> int:
            return i * ny * nz + j * nz + k

        # Gradient in x-direction
        row_x, col_x, data_x = [], [], []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    center_idx = index_3d(i, j, k)

                    if i == 0:  # Forward difference
                        row_x.extend([center_idx, center_idx])
                        col_x.extend([center_idx, index_3d(i + 1, j, k)])
                        data_x.extend([-1.0 / dx, 1.0 / dx])
                    elif i == nx - 1:  # Backward difference
                        row_x.extend([center_idx, center_idx])
                        col_x.extend([index_3d(i - 1, j, k), center_idx])
                        data_x.extend([-1.0 / dx, 1.0 / dx])
                    else:  # Central difference
                        row_x.extend([center_idx, center_idx])
                        col_x.extend([index_3d(i - 1, j, k), index_3d(i + 1, j, k)])
                        data_x.extend([-1.0 / (2 * dx), 1.0 / (2 * dx)])

        grad_x = csr_matrix((data_x, (row_x, col_x)), shape=(total_points, total_points))

        # Similar for y and z directions (simplified for brevity)
        # In practice, would implement full gradient operators
        grad_y = csr_matrix((total_points, total_points))  # Placeholder
        grad_z = csr_matrix((total_points, total_points))  # Placeholder

        return grad_x, grad_y, grad_z

    @staticmethod
    def solve_sparse_system(A: csr_matrix, b: np.ndarray, method: str = "spsolve", **kwargs) -> np.ndarray:
        """
        Solve sparse linear system Ax = b with optimized methods.

        Args:
            A: Sparse coefficient matrix
            b: Right-hand side
            method: Solver method ('spsolve', 'cg', 'bicgstab')

        Returns:
            Solution vector
        """
        if method == "spsolve":
            return sp_linalg.spsolve(A, b)
        elif method == "cg":
            x, info = sp_linalg.cg(A, b, **kwargs)
            if info != 0:
                warnings.warn(f"CG solver did not converge (info={info})")
            return x
        elif method == "bicgstab":
            x, info = sp_linalg.bicgstab(A, b, **kwargs)
            if info != 0:
                warnings.warn(f"BiCGSTAB solver did not converge (info={info})")
            return x
        else:
            raise ValueError(f"Unknown solver method: {method}")

    @staticmethod
    def optimize_matrix_structure(matrix: csr_matrix, reorder_method: str = "rcm") -> csr_matrix:
        """
        Optimize sparse matrix structure for better performance.

        Args:
            matrix: Input sparse matrix
            reorder_method: Reordering method ('rcm', 'amd', 'nested_dissection')

        Returns:
            Reordered matrix with improved structure
        """
        try:
            from scipy.sparse.csgraph import reverse_cuthill_mckee

            if reorder_method == "rcm":
                # Reverse Cuthill-McKee reordering
                perm = reverse_cuthill_mckee(matrix)
                return matrix[perm, :][:, perm]
            else:
                warnings.warn(f"Reordering method {reorder_method} not available, using original matrix")
                return matrix
        except ImportError:
            warnings.warn("SciPy graph algorithms not available for matrix reordering")
            return matrix

    @staticmethod
    def create_block_diagonal_preconditioner(matrix: csr_matrix, block_size: int) -> csr_matrix:
        """
        Create block diagonal preconditioner for iterative solvers.

        Args:
            matrix: Input matrix
            block_size: Size of diagonal blocks

        Returns:
            Block diagonal preconditioner matrix
        """
        n = matrix.shape[0]
        precond_data = []
        precond_rows = []
        precond_cols = []

        for i in range(0, n, block_size):
            end_idx = min(i + block_size, n)
            block = matrix[i:end_idx, i:end_idx].toarray()

            try:
                block_inv = np.linalg.inv(block)
                for row in range(block.shape[0]):
                    for col in range(block.shape[1]):
                        if abs(block_inv[row, col]) > 1e-14:
                            precond_rows.append(i + row)
                            precond_cols.append(i + col)
                            precond_data.append(block_inv[row, col])
            except np.linalg.LinAlgError:
                # Use diagonal if block is singular
                for j in range(block.shape[0]):
                    if abs(block[j, j]) > 1e-14:
                        precond_rows.append(i + j)
                        precond_cols.append(i + j)
                        precond_data.append(1.0 / block[j, j])

        return csr_matrix((precond_data, (precond_rows, precond_cols)), shape=(n, n))

    @staticmethod
    def adaptive_matrix_format(
        matrix: csr_matrix | csc_matrix | lil_matrix, operation_type: str
    ) -> csr_matrix | csc_matrix:
        """
        Convert matrix to optimal format for specific operations.

        Args:
            matrix: Input sparse matrix
            operation_type: Type of operation ('matvec', 'solve', 'transpose')

        Returns:
            Matrix in optimal format
        """
        if operation_type in ["matvec", "solve"]:
            # CSR is optimal for matrix-vector products and solving
            return matrix.tocsr()
        elif operation_type == "transpose":
            # CSC is optimal for transpose operations
            return matrix.tocsc()
        else:
            return matrix.tocsr()  # Default to CSR

    @staticmethod
    def estimate_solve_complexity(matrix: csr_matrix, method: str = "spsolve") -> dict[str, int | float]:
        """
        Estimate computational complexity of solving linear system.

        Args:
            matrix: Coefficient matrix
            method: Solution method

        Returns:
            Dictionary with complexity estimates
        """
        n = matrix.shape[0]
        nnz = matrix.nnz

        complexity_estimates = {
            "matrix_size": n,
            "nonzeros": nnz,
            "fill_factor": nnz / (n * n),
            "estimated_memory_mb": 0.0,
            "estimated_time_seconds": 0.0,
        }

        if method == "spsolve":
            # Direct solver complexity: O(n^1.5) for sparse matrices
            complexity_estimates["estimated_time_seconds"] = (n**1.5) * 1e-8
            complexity_estimates["estimated_memory_mb"] = nnz * 16 / (1024**2)  # Rough estimate
        elif method in ["cg", "bicgstab"]:
            # Iterative solver: depends on condition number
            # Rough estimate: O(n * nnz * iterations)
            estimated_iterations = min(50, int(np.sqrt(n)))
            complexity_estimates["estimated_time_seconds"] = n * nnz * estimated_iterations * 1e-9
            complexity_estimates["estimated_memory_mb"] = n * 8 * 5 / (1024**2)  # 5 vectors storage

        return complexity_estimates


class AdvancedSparseOperations:
    """Advanced sparse matrix operations for high-dimensional MFG problems."""

    @staticmethod
    def kronecker_sparse(A: csr_matrix, B: csr_matrix) -> csr_matrix:
        """
        Compute Kronecker product of two sparse matrices efficiently.

        Args:
            A: First sparse matrix
            B: Second sparse matrix

        Returns:
            Kronecker product A ⊗ B
        """
        from scipy.sparse import kron

        return kron(A, B, format="csr")

    @staticmethod
    def tensor_product_operator(operators: list[csr_matrix]) -> csr_matrix:
        """
        Create tensor product of multiple operators for multi-dimensional problems.

        Args:
            operators: List of 1D operators to combine

        Returns:
            Multi-dimensional operator as tensor product
        """
        if len(operators) == 1:
            return operators[0]

        result = operators[0]
        for op in operators[1:]:
            # Create identity matrices for tensor product
            I_result = csr_matrix(np.eye(result.shape[0]))
            I_op = csr_matrix(np.eye(op.shape[0]))

            # Compute tensor products
            term1 = AdvancedSparseOperations.kronecker_sparse(result, I_op)
            term2 = AdvancedSparseOperations.kronecker_sparse(I_result, op)

            result = term1 + term2

        return result

    @staticmethod
    def create_multiscale_operator(
        fine_matrix: csr_matrix, coarse_matrix: csr_matrix, restriction: csr_matrix, prolongation: csr_matrix
    ) -> dict[str, csr_matrix]:
        """
        Create multiscale operator system for multigrid methods.

        Args:
            fine_matrix: Fine-level operator
            coarse_matrix: Coarse-level operator
            restriction: Restriction operator (fine to coarse)
            prolongation: Prolongation operator (coarse to fine)

        Returns:
            Dictionary containing multiscale operators
        """
        # Galerkin coarse operator: R * A * P
        galerkin_operator = restriction @ fine_matrix @ prolongation

        # Error correction operators
        pre_smoother = fine_matrix.copy()  # Could be improved with specific smoothers
        post_smoother = fine_matrix.copy()

        return {
            "fine_operator": fine_matrix,
            "coarse_operator": coarse_matrix,
            "galerkin_operator": galerkin_operator,
            "restriction": restriction,
            "prolongation": prolongation,
            "pre_smoother": pre_smoother,
            "post_smoother": post_smoother,
        }

    @staticmethod
    def fast_matrix_powers(matrix: csr_matrix, power: int, method: str = "repeated_squaring") -> csr_matrix:
        """
        Compute matrix powers efficiently for sparse matrices.

        Args:
            matrix: Input sparse matrix
            power: Power to compute
            method: Algorithm ('repeated_squaring', 'iterative')

        Returns:
            Matrix raised to the specified power
        """
        if power == 0:
            return csr_matrix(np.eye(matrix.shape[0]))
        if power == 1:
            return matrix.copy()

        if method == "repeated_squaring":
            result = csr_matrix(np.eye(matrix.shape[0]))
            base = matrix.copy()

            while power > 0:
                if power % 2 == 1:
                    result = result @ base
                base = base @ base
                power //= 2

            return result

        elif method == "iterative":
            result = matrix.copy()
            for _ in range(power - 1):
                result = result @ matrix
            return result

        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def matrix_exponential_sparse(matrix: csr_matrix, time_step: float, method: str = "pade") -> csr_matrix:
        """
        Compute matrix exponential for sparse matrices (time integration).

        Args:
            matrix: Input sparse matrix
            time_step: Time step parameter
            method: Method for computing exponential

        Returns:
            Matrix exponential exp(time_step * matrix)
        """
        if method == "pade":
            # Pade approximation (simplified implementation)
            scaled_matrix = time_step * matrix

            # First-order Pade: (I - A/2)^-1 * (I + A/2)
            identity_mat = csr_matrix(np.eye(matrix.shape[0]))
            numerator = identity_mat + 0.5 * scaled_matrix
            denominator = identity_mat - 0.5 * scaled_matrix

            try:
                # Solve (I - A/2) * result = (I + A/2)
                result_dense = sp_linalg.spsolve(denominator, numerator.toarray())
                return csr_matrix(result_dense)
            except (np.linalg.LinAlgError, RuntimeError):
                # Sparse solve may fail due to singular matrix or memory issues
                warnings.warn("Pade approximation failed, using first-order approximation")
                return identity_mat + scaled_matrix

        elif method == "taylor":
            # Taylor series approximation: I + A + A^2/2! + A^3/3! + ...
            identity_mat = csr_matrix(np.eye(matrix.shape[0]))
            scaled_matrix = time_step * matrix
            result = identity_mat.copy()
            term = identity_mat.copy()

            for k in range(1, 10):  # Limit terms for efficiency
                term = term @ scaled_matrix / k
                result = result + term

                # Check convergence (simplified)
                if term.nnz == 0 or np.abs(term.data).max() < 1e-12:
                    break

            return result

        else:
            raise ValueError(f"Unknown method: {method}")


class ParallelSparseOperations:
    """Parallel sparse matrix operations for multi-core systems."""

    @staticmethod
    def parallel_matvec(matrix: csr_matrix, vectors: np.ndarray) -> np.ndarray:
        """
        Parallel matrix-vector multiplication for multiple vectors.

        Args:
            matrix: Sparse matrix
            vectors: Array of vectors (shape: n_vectors x matrix_size)

        Returns:
            Result of matrix-vector products
        """
        if NUMBA_AVAILABLE:
            # Use Numba for parallel operations if available
            return ParallelSparseOperations._numba_parallel_matvec(matrix, vectors)
        else:
            # Fallback to sequential
            return np.array([matrix @ v for v in vectors])

    @staticmethod
    def _numba_parallel_matvec(matrix: csr_matrix, vectors: np.ndarray) -> np.ndarray:
        """Numba-accelerated parallel matrix-vector multiplication."""
        if not NUMBA_AVAILABLE:
            raise ImportError("Numba not available for parallel operations")

        # This would contain actual Numba implementation
        # Placeholder for now
        return np.array([matrix @ v for v in vectors])

    @staticmethod
    def chunk_based_operations(matrix: csr_matrix, operation: Callable, chunk_size: int = 1000) -> Any:
        """
        Process large sparse matrices in chunks to manage memory.

        Args:
            matrix: Large sparse matrix
            operation: Function to apply to each chunk
            chunk_size: Size of chunks for processing

        Returns:
            Combined result from chunk operations
        """
        n_rows = matrix.shape[0]
        results = []

        for start_idx in range(0, n_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, n_rows)
            chunk = matrix[start_idx:end_idx, :]
            result = operation(chunk)
            results.append(result)

        # Combine results (implementation depends on operation type)
        if isinstance(results[0], np.ndarray):
            return np.concatenate(results)
        else:
            return results


class MemoryOptimizer:
    """Memory optimization utilities for large-scale problems."""

    @staticmethod
    def estimate_memory_requirements(problem_size: tuple[int, ...], data_type: str = "float64") -> dict[str, float]:
        """
        Estimate memory requirements for MFG problem.

        Args:
            problem_size: Tuple of (spatial_points, time_steps)
            data_type: Data type for arrays

        Returns:
            Dictionary with memory estimates in MB
        """
        spatial_points, time_steps = problem_size[:2]

        # Bytes per element
        bytes_per_element = {"float32": 4, "float64": 8, "complex64": 8, "complex128": 16}.get(data_type, 8)

        estimates = {
            "density_field": spatial_points * time_steps * bytes_per_element / 1024**2,
            "value_field": spatial_points * time_steps * bytes_per_element / 1024**2,
            "temporary_arrays": spatial_points * 5 * bytes_per_element / 1024**2,  # Estimate
            "sparse_matrices": spatial_points * 20 * bytes_per_element / 1024**2,  # Estimate for 3D
        }

        estimates["total_estimated"] = sum(estimates.values())
        estimates["recommended_ram"] = estimates["total_estimated"] * 3  # Safety factor

        return estimates

    @staticmethod
    def optimize_array_layout(array: np.ndarray, access_pattern: str = "C") -> np.ndarray:
        """
        Optimize array memory layout for access pattern.

        Args:
            array: Input array
            access_pattern: C for row-major, "F" for column-major

        Returns:
            Optimized array
        """
        if access_pattern == "C" and not array.flags.c_contiguous:
            return np.ascontiguousarray(array)
        elif access_pattern == "F" and not array.flags.f_contiguous:
            return np.asfortranarray(array)
        else:
            return array

    @staticmethod
    def chunk_computation(
        computation_func: Callable, data: np.ndarray, chunk_size: int = 1000, axis: int = 0
    ) -> np.ndarray:
        """
        Break large computation into chunks to manage memory.

        Args:
            computation_func: Function to apply to each chunk
            data: Input data array
            chunk_size: Size of each chunk
            axis: Axis along which to chunk

        Returns:
            Concatenated results
        """
        chunks = np.array_split(data, max(1, data.shape[axis] // chunk_size), axis=axis)

        results = []
        for chunk in chunks:
            result = computation_func(chunk)
            results.append(result)

        return np.concatenate(results, axis=axis)


class AccelerationBackend:
    """Backend selection and optimization for different compute platforms."""

    def __init__(self):
        self.available_backends = self._detect_backends()
        self.current_backend = "numpy"

    def _detect_backends(self) -> dict[str, bool]:
        """Detect available acceleration backends."""
        backends = {
            "numpy": True,  # Always available
            "numba": NUMBA_AVAILABLE,
            "jax": JAX_AVAILABLE,
        }

        # Check for GPU availability
        if JAX_AVAILABLE:
            try:
                jax.devices("gpu")
                backends["jax_gpu"] = True
            except (RuntimeError, ValueError):
                # GPU devices may not be available or device query may fail
                backends["jax_gpu"] = False

        return backends

    def set_backend(self, backend: str):
        """Set computational backend."""
        if backend not in self.available_backends:
            raise ValueError(f"Backend {backend} not available")

        if not self.available_backends[backend]:
            raise ValueError(f"Backend {backend} detected but not functional")

        self.current_backend = backend

    def get_backend_info(self) -> dict[str, Any]:
        """Get information about available backends."""
        return {
            "current_backend": self.current_backend,
            "available_backends": self.available_backends,
            "recommendations": self._get_recommendations(),
        }

    def _get_recommendations(self) -> list[str]:
        """Get performance recommendations."""
        recommendations = []

        if self.available_backends.get("jax_gpu", False):
            recommendations.append("Consider using JAX with GPU for large 3D problems")
        elif self.available_backends.get("jax", False):
            recommendations.append("JAX available for CPU acceleration")

        if self.available_backends.get("numba", False):
            recommendations.append("Numba JIT compilation available for tight loops")

        if not any(self.available_backends[k] for k in ["numba", "jax"]):
            recommendations.append("Consider installing numba or jax for acceleration")

        return recommendations

    def optimize_function(self, func: Callable, backend: str | None = None) -> Callable:
        """
        Optimize function for selected backend.

        Args:
            func: Function to optimize
            backend: Backend to use (None for current)

        Returns:
            Optimized function
        """
        backend = backend or self.current_backend

        if backend == "numba" and NUMBA_AVAILABLE:
            return numba.jit(func, nopython=True)
        elif backend.startswith("jax") and JAX_AVAILABLE:
            return jax.jit(func)
        else:
            return func  # Return original function


class ParallelizationHelper:
    """Helper class for parallel computation strategies."""

    @staticmethod
    def estimate_optimal_chunks(total_size: int, memory_limit_mb: float = 1000, element_size_bytes: int = 8) -> int:
        """
        Estimate optimal chunk size for parallel processing.

        Args:
            total_size: Total number of elements
            memory_limit_mb: Memory limit per chunk in MB
            element_size_bytes: Size of each element in bytes

        Returns:
            Recommended chunk size
        """
        max_elements_per_chunk = int(memory_limit_mb * 1024**2 / element_size_bytes)
        num_chunks = max(1, total_size // max_elements_per_chunk)

        # Try to balance chunk sizes
        optimal_chunk_size = total_size // num_chunks

        return max(1, optimal_chunk_size)

    @staticmethod
    def create_load_balanced_chunks(data_sizes: list[int], num_workers: int) -> list[list[int]]:
        """
        Create load-balanced work distribution.

        Args:
            data_sizes: Sizes of individual work items
            num_workers: Number of parallel workers

        Returns:
            List of work assignments for each worker
        """
        # Sort by size (largest first) for better load balancing
        indexed_sizes = [(i, size) for i, size in enumerate(data_sizes)]
        indexed_sizes.sort(key=lambda x: x[1], reverse=True)

        # Initialize worker loads
        workers: list[list[int]] = [[] for _ in range(num_workers)]
        worker_loads = [0] * num_workers

        # Assign work using greedy algorithm
        for idx, size in indexed_sizes:
            # Find worker with smallest current load
            min_worker = np.argmin(worker_loads)
            workers[min_worker].append(idx)
            worker_loads[min_worker] += size

        return workers


def create_performance_optimizer() -> dict[str, Any]:
    """
    Factory function to create a complete performance optimization suite.

    Returns:
        Dictionary with optimization tools
    """
    return {
        "monitor": PerformanceMonitor(),
        "sparse_optimizer": SparseMatrixOptimizer(),
        "memory_optimizer": MemoryOptimizer(),
        "acceleration_backend": AccelerationBackend(),
        "parallelization_helper": ParallelizationHelper(),
    }


# High-level optimization functions


def optimize_mfg_problem_performance(problem_params: dict[str, Any]) -> dict[str, Any]:
    """
    Analyze MFG problem and provide performance optimization recommendations.

    Args:
        problem_params: Dictionary with problem parameters
                       (spatial_points, time_steps, dimension, etc.)

    Returns:
        Dictionary with optimization recommendations
    """
    spatial_points = problem_params.get("spatial_points", 1000)
    time_steps = problem_params.get("time_steps", 100)
    dimension = problem_params.get("dimension", 2)

    optimizer_suite = create_performance_optimizer()
    memory_optimizer = optimizer_suite["memory_optimizer"]
    backend = optimizer_suite["acceleration_backend"]

    # Memory analysis
    memory_est = memory_optimizer.estimate_memory_requirements((spatial_points, time_steps))

    # Performance recommendations
    recommendations = []

    if memory_est["total_estimated"] > 1000:  # > 1GB
        recommendations.append("Consider using sparse matrices for large problems")
        recommendations.append("Implement memory chunking for time stepping")

    if dimension >= 3 and spatial_points > 10000:
        recommendations.append("Use adaptive mesh refinement for 3D problems")
        recommendations.append("Consider GPU acceleration for large 3D problems")

    if spatial_points > 50000:
        recommendations.append("Implement parallel matrix operations")
        recommendations.append("Use iterative solvers instead of direct methods")

    return {
        "memory_estimates": memory_est,
        "backend_info": backend.get_backend_info(),
        "recommendations": recommendations,
        "optimization_suite": optimizer_suite,
    }

"""
Laplacian operator for tensor product grids.

This module provides LinearOperator implementation of the discrete Laplacian
for structured grids, wrapping the tensor_calculus infrastructure.

Mathematical Background:
    Laplacian operator: Δu = ∇²u = ∑ᵢ ∂²u/∂xᵢ²

    Discretization (2nd-order central differences):
        ∂²u/∂x² ≈ (u[i+1] - 2u[i] + u[i-1]) / h²

    For multi-dimensional grids:
        Δu = Δₓu + Δᵧu + Δᵧu + ...

References:
    - LeVeque (2007): Finite Difference Methods for ODEs and PDEs
    - Strang (2007): Computational Science and Engineering

Created: 2026-01-17 (Issue #595 - Operator Refactoring)
Part of: Issue #590 Phase 1.2 - TensorProductGrid Operator Traits
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import LinearOperator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mfgarchon.geometry.boundary import BoundaryConditions


class LaplacianOperator(LinearOperator):
    """
    Discrete Laplacian operator Δu = ∇²u for tensor product grids.

    Implements scipy.sparse.linalg.LinearOperator interface for compatibility
    with iterative solvers (gmres, cg, etc.) and operator composition.

    The operator wraps mfgarchon.utils.numerical.tensor_calculus.laplacian with
    grid-specific parameters (spacings, BC) curried into the operator object.

    Attributes:
        spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
        field_shape: Shape of input field (Nx, Ny, ...) or (Nx,) for 1D
        bc: Boundary conditions (None for periodic)
        order: Discretization order (2, 4, 6, ...)
        shape: Operator shape (N, N) where N = ∏field_shape
        dtype: Data type (float64)

    Usage:
        >>> # Create operator
        >>> L = LaplacianOperator(spacings=[0.1, 0.1], field_shape=(50, 50), bc=bc)
        >>>
        >>> # Apply via matrix-vector product (scipy-compatible)
        >>> u_flat = u.ravel()
        >>> Lu_flat = L @ u_flat
        >>>
        >>> # Apply via callable (preserves field shape)
        >>> Lu = L(u)  # Same as (L @ u.ravel()).reshape(u.shape)
        >>>
        >>> # Use with iterative solvers
        >>> from scipy.sparse.linalg import gmres
        >>> u_solution, info = gmres(L, b)

    Example:
        >>> import numpy as np
        >>> from mfgarchon.geometry.boundary import neumann_bc
        >>>
        >>> # 2D Poisson problem: Δu = f with Neumann BC
        >>> bc = neumann_bc(dimension=2)
        >>> L = LaplacianOperator(spacings=[0.01, 0.01], field_shape=(100, 100), bc=bc)
        >>>
        >>> # Right-hand side
        >>> f = np.ones((100, 100))
        >>>
        >>> # Solve Δu = f
        >>> u_flat, info = gmres(L, f.ravel(), tol=1e-6)
        >>> u = u_flat.reshape(100, 100)
    """

    def __init__(
        self,
        spacings: Sequence[float],
        field_shape: tuple[int, ...] | int,
        bc: BoundaryConditions | None = None,
        order: int = 2,
        time: float = 0.0,
        mass_conservative: bool = False,
        coefficient_field: NDArray | None = None,
    ):
        """
        Initialize Laplacian operator.

        Args:
            spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
            field_shape: Shape of field arrays (Nx, Ny, ...) or Nx for 1D
            bc: Boundary conditions (None for periodic/wrap)
            order: Discretization order (currently only order=2 supported)
            time: Time for time-dependent BCs (default 0.0)
            mass_conservative: Affects ONLY ``as_scipy_sparse()`` under a no-flux/Neumann BC
                (Issue #1184). Default ``False`` keeps the 2nd-order-accurate ghost-mirror
                stencil (diag ``-2/h²``, wall neighbor ``+2/h²``; row-conservative but NOT
                column-conservative -- it leaks mass when used as the implicit FP system
                matrix, because ``1ᵀL ≠ 0``). ``True`` emits the finite-volume zero-flux
                stencil (wall diag ``-1/h²``, neighbor ``+1/h²``), which is BOTH row- and
                column-conservative (``1ᵀL = 0``), so the implicit diffusion solve conserves
                mass exactly -- at the cost of 1st-order wall accuracy. The FP mass path sets
                this; HJB/elliptic consumers (matvec, accuracy-critical) leave it ``False``.
            coefficient_field: Optional per-point diffusion field ``D(x)`` (same shape as the
                field) for a spatially-varying ``∇·(D∇·)`` (Issue #1183). Only honored by
                ``as_scipy_sparse()`` with ``mass_conservative=True``: the stencil uses the
                face-averaged ``D_{i+1/2} = ½(D_i + D_{i+1})``, so the flux telescopes and
                ``1ᵀL = 0`` holds even for varying ``D`` (a *point-value* ``D_i·Δ`` would leak).
                ``None`` (default) builds the unit Laplacian (caller multiplies by a scalar ``D``).

        Raises:
            ValueError: If order != 2 (higher orders not yet implemented)
        """
        # Handle 1D shape
        if isinstance(field_shape, int):
            field_shape = (field_shape,)
        else:
            field_shape = tuple(field_shape)

        self.spacings = list(spacings)
        self.field_shape = field_shape
        self.bc = bc
        self.order = order
        self.time = time
        self.mass_conservative = mass_conservative
        self.coefficient_field = None if coefficient_field is None else np.asarray(coefficient_field, dtype=np.float64)

        # Validate
        if len(self.spacings) != len(self.field_shape):
            raise ValueError(f"spacings length {len(self.spacings)} != field_shape dimensions {len(self.field_shape)}")

        if order != 2:
            raise ValueError(f"Only order=2 currently supported, got {order}")

        if self.coefficient_field is not None:
            if self.coefficient_field.shape != self.field_shape:
                raise ValueError(
                    f"coefficient_field shape {self.coefficient_field.shape} != field_shape {self.field_shape}"
                )
            if not mass_conservative:
                raise ValueError(
                    "coefficient_field requires mass_conservative=True (variable-coefficient diffusion "
                    "is only assembled in the conservative finite-volume branch; Issue #1183)."
                )

        # Compute operator shape
        N = int(np.prod(field_shape))
        super().__init__(shape=(N, N), dtype=np.float64)

    def _matvec(self, u_flat: NDArray) -> NDArray:
        """
        Apply Laplacian to flattened field.

        This is the core LinearOperator method required by scipy.

        Args:
            u_flat: Flattened field array, shape (N,)

        Returns:
            Laplacian of u, flattened, shape (N,)
        """
        # Issue #625: Migrated from tensor_calculus to stencils
        from mfgarchon.operators.stencils.finite_difference import laplacian_with_bc

        # Reshape to field
        u = u_flat.reshape(self.field_shape)

        # Apply Laplacian using stencil with BC handling
        Lu = laplacian_with_bc(u, self.spacings, bc=self.bc, time=self.time)

        # Return flattened
        return Lu.ravel()

    def __call__(self, u: NDArray) -> NDArray:
        """
        Apply Laplacian to field (preserves shape).

        This method allows using the operator as L(u) in addition to L @ u.ravel().
        Automatically handles reshaping.

        Args:
            u: Field array, shape field_shape or (N,)

        Returns:
            Laplacian of u, same shape as input

        Example:
            >>> L = LaplacianOperator(spacings=[0.1, 0.1], field_shape=(50, 50))
            >>> u = np.random.rand(50, 50)
            >>> Lu = L(u)  # Returns (50, 50) array
        """
        # Handle already-flattened input
        if u.ndim == 1:
            return self._matvec(u)

        # Handle field input
        if u.shape != self.field_shape:
            raise ValueError(f"Input shape {u.shape} doesn't match field_shape {self.field_shape}")

        Lu_flat = self._matvec(u.ravel())
        return Lu_flat.reshape(self.field_shape)

    def as_scipy_sparse(self) -> sparse.spmatrix:
        """
        Export Laplacian as scipy sparse matrix (Issue #597 Milestone 2).

        This method enables using the Laplacian operator in implicit time-stepping
        schemes that require matrix representations (e.g., FP solver).

        Implementation uses direct sparse assembly with correct one-sided stencils
        at Neumann boundaries to match coefficient folding behavior.

        Returns:
            Sparse CSR matrix representation of Laplacian operator

        Example:
            >>> L_op = LaplacianOperator(spacings=[0.1, 0.1], field_shape=(50, 50), bc=bc)
            >>> L_matrix = L_op.as_scipy_sparse()
            >>> # Use in implicit scheme: (I - dt*D*L) @ u = rhs
            >>> import scipy.sparse as sp
            >>> A = sp.eye(2500) - 0.01 * 0.5 * L_matrix

        Notes:
            - Returns CSR format for efficient matrix-vector products
            - BC handling via direct sparse assembly (matches coefficient folding)
            - Neumann BC: Uses one-sided stencils at boundaries
            - Periodic BC: Uses wrapped indices
            - For large grids (N > 100k), raises ValueError

        Raises:
            ValueError: If grid too large (N > 100k points)
        """

        N = int(np.prod(self.field_shape))

        # Threshold for sparse construction
        if N > 100_000:
            raise ValueError(
                f"Grid size {N} too large for current as_scipy_sparse() implementation. "
                f"Use matrix-free methods (LinearOperator interface) instead."
            )

        # Use direct sparse assembly for correct BC handling
        return self._build_sparse_laplacian_direct()

    def _build_sparse_laplacian_direct(self) -> sparse.spmatrix:
        """
        Build sparse Laplacian via vectorized assembly with correct BC handling.

        Issue #928: Replaced point-by-point Python loop with NumPy vectorized
        index computation. Assembly cost: O(ndim) NumPy operations instead of
        O(N * ndim) Python operations.

        For each dimension d, computes strides and masks for interior/boundary
        points, then assembles all (row, col, val) triples via array operations.

        Returns:
            Sparse CSR matrix with correct boundary stencils
        """
        N = int(np.prod(self.field_shape))
        ndim = len(self.field_shape)

        # Determine BC type
        bc_type = None
        if self.bc is not None:
            try:
                bc_type = self.bc.bc_type.value if hasattr(self.bc.bc_type, "value") else str(self.bc.bc_type)
            except (AttributeError, ValueError) as e:
                # Issue #1071 / fail-fast: a PROVIDED bc whose bc_type cannot be determined must
                # NOT silently degrade to the bc=None periodic default (that would solve a
                # different boundary than the caller specified). Surface it.
                raise ValueError(
                    f"LaplacianOperator: could not determine bc_type from the provided bc "
                    f"({type(self.bc).__name__}): {e}. A BoundaryConditions exposing a single bc_type "
                    f"is required; the operator will not silently treat a provided bc as periodic."
                ) from e

        # All flat indices
        all_idx = np.arange(N)
        # Multi-index array: (N, ndim)
        multi_indices = np.array(np.unravel_index(all_idx, self.field_shape)).T

        rows_list: list[np.ndarray] = []
        cols_list: list[np.ndarray] = []
        vals_list: list[np.ndarray] = []

        for d in range(ndim):
            h = self.spacings[d]
            h2 = h**2
            n_d = self.field_shape[d]
            i_d = multi_indices[:, d]  # (N,) index in dimension d

            # Compute strides: flat index offset for +-1 in dimension d
            stride = int(np.prod(self.field_shape[d + 1 :])) if d < ndim - 1 else 1

            # Boundary masks
            at_min = i_d == 0
            at_max = i_d == n_d - 1
            interior = ~at_min & ~at_max

            if bc_type in ("neumann", "no_flux") and self.mass_conservative:
                # Finite-volume zero-flux stencil (Issue #1184): omit the ghost face at the
                # wall so each cell's row telescopes against its neighbors' columns -> BOTH
                # row- and column-conservative (1ᵀL = 0), 1st-order accurate at the wall.
                # Each face carries the face-averaged diffusion D_{i+1/2} = ½(D_i + D_{i+1})
                # (Issue #1183); with no coefficient_field every face coefficient is 1 (the
                # caller multiplies the unit Laplacian by a scalar D -> byte-identical).
                d_flat = None if self.coefficient_field is None else self.coefficient_field.ravel()
                ii = all_idx[interior]
                amin = all_idx[at_min]
                amax = all_idx[at_max]
                if d_flat is None:
                    f_lo_i = np.ones(ii.size)
                    f_hi_i = np.ones(ii.size)
                    f_hi_min = np.ones(amin.size)
                    f_lo_max = np.ones(amax.size)
                else:
                    f_lo_i = 0.5 * (d_flat[ii] + d_flat[ii - stride])  # face i-1/2
                    f_hi_i = 0.5 * (d_flat[ii] + d_flat[ii + stride])  # face i+1/2
                    f_hi_min = 0.5 * (d_flat[amin] + d_flat[amin + stride])
                    f_lo_max = 0.5 * (d_flat[amax] + d_flat[amax - stride])

                # Interior cell: diag -(f_lo+f_hi)/h², neighbors +f/h²
                rows_list.append(ii)
                cols_list.append(ii)
                vals_list.append(-(f_lo_i + f_hi_i) / h2)
                rows_list.append(ii)
                cols_list.append(ii - stride)
                vals_list.append(f_lo_i / h2)
                rows_list.append(ii)
                cols_list.append(ii + stride)
                vals_list.append(f_hi_i / h2)

                # Wall cells: only the single interior-facing face (zero flux through the wall)
                rows_list.append(amin)
                cols_list.append(amin)
                vals_list.append(-f_hi_min / h2)
                rows_list.append(amin)
                cols_list.append(amin + stride)
                vals_list.append(f_hi_min / h2)
                rows_list.append(amax)
                cols_list.append(amax)
                vals_list.append(-f_lo_max / h2)
                rows_list.append(amax)
                cols_list.append(amax - stride)
                vals_list.append(f_lo_max / h2)

            elif bc_type in ("neumann", "no_flux"):
                # ALL points get diagonal -2/h² (interior and boundary)
                rows_list.append(all_idx)
                cols_list.append(all_idx)
                vals_list.append(np.full(N, -2.0 / h2))

                # Interior: left and right neighbors, each +1/h²
                rows_list.append(all_idx[interior])
                cols_list.append(all_idx[interior] - stride)
                vals_list.append(np.full(int(interior.sum()), 1.0 / h2))

                rows_list.append(all_idx[interior])
                cols_list.append(all_idx[interior] + stride)
                vals_list.append(np.full(int(interior.sum()), 1.0 / h2))

                # Neumann boundary: mirror ghost → neighbor gets +2/h² (Issue #668 fix)
                rows_list.append(all_idx[at_min])
                cols_list.append(all_idx[at_min] + stride)
                vals_list.append(np.full(int(at_min.sum()), 2.0 / h2))

                rows_list.append(all_idx[at_max])
                cols_list.append(all_idx[at_max] - stride)
                vals_list.append(np.full(int(at_max.sum()), 2.0 / h2))

            elif bc_type == "periodic" or bc_type is None:
                # ALL points: diagonal -2/h²
                rows_list.append(all_idx)
                cols_list.append(all_idx)
                vals_list.append(np.full(N, -2.0 / h2))

                # Left neighbor (wrapped for periodic)
                left_idx = multi_indices.copy()
                left_idx[:, d] = (i_d - 1 + n_d) % n_d
                left_flat = np.ravel_multi_index(left_idx.T, self.field_shape)
                rows_list.append(all_idx)
                cols_list.append(left_flat)
                vals_list.append(np.full(N, 1.0 / h2))

                # Right neighbor (wrapped for periodic)
                right_idx = multi_indices.copy()
                right_idx[:, d] = (i_d + 1) % n_d
                right_flat = np.ravel_multi_index(right_idx.T, self.field_shape)
                rows_list.append(all_idx)
                cols_list.append(right_flat)
                vals_list.append(np.full(N, 1.0 / h2))

            elif bc_type == "dirichlet":
                # ALL points: diagonal -2/h²
                rows_list.append(all_idx)
                cols_list.append(all_idx)
                vals_list.append(np.full(N, -2.0 / h2))

                # Interior: both neighbors +1/h²
                rows_list.append(all_idx[interior])
                cols_list.append(all_idx[interior] - stride)
                vals_list.append(np.full(int(interior.sum()), 1.0 / h2))

                rows_list.append(all_idx[interior])
                cols_list.append(all_idx[interior] + stride)
                vals_list.append(np.full(int(interior.sum()), 1.0 / h2))

                # Boundary: only interior-side neighbor +1/h² (ghost omitted)
                rows_list.append(all_idx[at_min])
                cols_list.append(all_idx[at_min] + stride)
                vals_list.append(np.full(int(at_min.sum()), 1.0 / h2))

                rows_list.append(all_idx[at_max])
                cols_list.append(all_idx[at_max] - stride)
                vals_list.append(np.full(int(at_max.sum()), 1.0 / h2))

            else:
                # Issue #1071 / fail-fast: an unhandled bc_type previously produced an
                # interior-only stencil — boundary rows with NO diffusion entries, i.e. a
                # silently under-constrained boundary (wrong physics, no error). Surface it.
                raise NotImplementedError(
                    f"LaplacianOperator does not implement bc_type={bc_type!r}. Supported: "
                    "'periodic' (or bc=None), 'neumann'/'no_flux', 'dirichlet'. It will not "
                    "silently emit a boundary-diffusion-free interior-only stencil for an "
                    "unhandled BC (Issue #1071, fail-fast)."
                )

        # Single concatenation + single sparse construction
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        vals = np.concatenate(vals_list)

        return sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()

    def __repr__(self) -> str:
        """String representation for debugging."""
        bc_str = f"bc={self.bc.bc_type.value}" if self.bc else "bc=periodic"
        return (
            f"LaplacianOperator(\n"
            f"  field_shape={self.field_shape},\n"
            f"  spacings={self.spacings},\n"
            f"  {bc_str},\n"
            f"  order={self.order},\n"
            f"  shape={self.shape}\n"
            f")"
        )


if __name__ == "__main__":
    """Smoke test for LaplacianOperator."""
    print("Testing LaplacianOperator...")

    # Test 1D
    print("\n[1D Laplacian]")
    L_1d = LaplacianOperator(spacings=[0.1], field_shape=100)
    print(f"  Operator shape: {L_1d.shape}")
    print(f"  Field shape: {L_1d.field_shape}")

    u_1d = np.sin(np.linspace(0, 2 * np.pi, 100))
    Lu_1d = L_1d(u_1d)
    print(f"  Input shape: {u_1d.shape}, Output shape: {Lu_1d.shape}")
    assert Lu_1d.shape == u_1d.shape

    # Also test @ syntax
    Lu_1d_matvec = L_1d @ u_1d.ravel()
    assert np.allclose(Lu_1d.ravel(), Lu_1d_matvec)
    print("  ✓ L(u) == L @ u.ravel()")

    # Test 2D
    print("\n[2D Laplacian]")
    from mfgarchon.geometry.boundary import neumann_bc

    bc = neumann_bc(dimension=2)

    # Test on quadratic function: u = x² + y², Δu = 4
    Nx, Ny = 50, 50
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    u_2d = X**2 + Y**2

    L_2d = LaplacianOperator(spacings=[dx, dy], field_shape=(Nx, Ny), bc=bc)
    print(f"  Operator shape: {L_2d.shape}")
    print(f"  Field shape: {L_2d.field_shape}")
    print(f"  Grid spacing: dx={dx:.4f}, dy={dy:.4f}")
    print(f"  BC: {bc.bc_type.value}")

    Lu_2d = L_2d(u_2d)
    print(f"  Input shape: {u_2d.shape}, Output shape: {Lu_2d.shape}")
    assert Lu_2d.shape == u_2d.shape

    # Check interior values (should be ~4.0 for continuous case)
    # Analytical: Δ(x²+y²) = 2 + 2 = 4
    interior = Lu_2d[10:-10, 10:-10]
    mean_val = np.mean(interior)
    std_val = np.std(interior)
    print(f"  Δ(x²+y²) interior: mean={mean_val:.3f}, std={std_val:.3f} (expected = 4.0)")
    # Should be very close to 4.0 with correct spacing
    assert 3.5 < mean_val < 4.5, f"Mean value {mean_val} outside expected range [3.5, 4.5]"
    print(f"  ✓ Laplacian accuracy check passed (error = {abs(mean_val - 4.0):.3e})")

    # Test isinstance check with scipy
    print("\n[scipy compatibility]")
    assert isinstance(L_2d, LinearOperator)
    print("  ✓ isinstance(L, scipy.sparse.linalg.LinearOperator)")

    # Test repr
    print("\n[String representation]")
    print(L_2d)

    print("\n✅ All LaplacianOperator tests passed!")

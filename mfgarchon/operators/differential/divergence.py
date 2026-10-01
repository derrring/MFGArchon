"""
Divergence operator for tensor product grids.

This module provides LinearOperator implementation of the discrete divergence
for structured grids, using finite difference stencils.

Mathematical Background:
    Divergence operator: div(F) = ∇·F = ∑ᵢ ∂Fᵢ/∂xᵢ

    For vector field F = (Fx, Fy, Fz):
        div(F) = ∂Fx/∂x + ∂Fy/∂y + ∂Fz/∂z

    Discretization (2nd-order central differences):
        ∂F/∂x ≈ (F[i+1] - F[i-1]) / (2h)

    Conservative form (for flux):
        ∇·F conserves total mass when integrated over domain

References:
    - LeVeque (2002): Finite Volume Methods for Hyperbolic Problems
    - Toro (2009): Riemann Solvers and Numerical Methods for Fluid Dynamics

Created: 2026-01-17 (Issue #595 Phase 2 - Operator Refactoring)
Part of: Geometry Operator LinearOperator Migration
Migrated: 2026-01-25 (Issue #625 - tensor_calculus → stencils migration)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse.linalg import LinearOperator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mfgarchon.geometry.boundary import BoundaryConditions


class DivergenceOperator(LinearOperator):
    """
    Discrete divergence operator ∇·F for tensor product grids.

    Implements scipy.sparse.linalg.LinearOperator interface for compatibility
    with iterative solvers and operator composition.

    Uses finite difference stencils with grid-specific parameters (spacings, BC)
    curried into the operator object.

    **Mathematical Context**:
        Divergence measures the "outflow" of a vector field:
            - div(F) > 0: Source (net outflow)
            - div(F) < 0: Sink (net inflow)
            - div(F) = 0: Divergence-free (incompressible flow)

    **Operator Shape**:
        Input:  Vector field F flattened to shape (dimension × N,)
        Output: Scalar field div(F) flattened to shape (N,)
        Operator shape: (N, dimension × N)

        where N = ∏field_shape (total number of grid points)

    Attributes:
        spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
        field_shape: Shape of scalar fields (Nx, Ny, ...) or (Nx,) for 1D
        bc: Boundary conditions (None for periodic)
        time: Time for time-dependent BCs
        dimension: Number of spatial dimensions
        shape: Operator shape (N, dimension × N)
        dtype: Data type (float64)

    Usage:
        >>> # Create operator
        >>> div_op = DivergenceOperator(spacings=[0.1, 0.1], field_shape=(50, 50), bc=bc)
        >>>
        >>> # Apply via matrix-vector product (scipy-compatible)
        >>> F_flat = F.ravel()  # Vector field (2, 50, 50) → (5000,)
        >>> div_F_flat = div_op @ F_flat  # Shape: (2500,)
        >>>
        >>> # Apply via callable (preserves field shape)
        >>> div_F = div_op(F)  # Input: (2, 50, 50) → Output: (50, 50)

    Example:
        >>> import numpy as np
        >>> from mfgarchon.geometry.boundary import neumann_bc
        >>>
        >>> # 2D vector field F = (x, y)
        >>> # Analytical: div(F) = ∂x/∂x + ∂y/∂y = 1 + 1 = 2
        >>> Nx, Ny = 100, 100
        >>> x = np.linspace(0, 1, Nx)
        >>> y = np.linspace(0, 1, Ny)
        >>> X, Y = np.meshgrid(x, y, indexing='ij')
        >>> F = np.stack([X, Y], axis=0)  # Shape: (2, 100, 100)
        >>>
        >>> bc = neumann_bc(dimension=2)
        >>> div_op = DivergenceOperator(spacings=[0.01, 0.01], field_shape=(Nx, Ny), bc=bc)
        >>>
        >>> div_F = div_op(F)  # Should be ≈ 2.0 everywhere
        >>> print(f"Mean divergence: {np.mean(div_F):.3f}")  # ~2.0
    """

    def __init__(
        self,
        spacings: Sequence[float],
        field_shape: tuple[int, ...] | int,
        bc: BoundaryConditions | None = None,
        time: float = 0.0,
    ):
        """
        Initialize divergence operator.

        Args:
            spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
            field_shape: Shape of scalar field arrays (Nx, Ny, ...) or Nx for 1D
            bc: Boundary conditions (None for periodic/wrap)
            time: Time for time-dependent BCs (default 0.0)

        Note:
            The vector field input has shape (dimension, Nx, Ny, ...).
            The divergence output has shape (Nx, Ny, ...).
        """
        # Handle 1D shape
        if isinstance(field_shape, int):
            field_shape = (field_shape,)
        else:
            field_shape = tuple(field_shape)

        self.spacings = list(spacings)
        self.field_shape = field_shape
        self.bc = bc
        self.time = time
        self.dimension = len(field_shape)

        # Validate
        if len(self.spacings) != self.dimension:
            raise ValueError(f"spacings length {len(self.spacings)} != field_shape dimensions {self.dimension}")

        # Compute operator shape
        # Input: vector field (dimension, Nx, Ny, ...) → flattened (dimension × N,)
        # Output: scalar field (Nx, Ny, ...) → flattened (N,)
        N = int(np.prod(field_shape))
        super().__init__(shape=(N, self.dimension * N), dtype=np.float64)

    def _matvec(self, F_flat: NDArray) -> NDArray:
        """
        Apply divergence to flattened vector field.

        This is the core LinearOperator method required by scipy.

        Args:
            F_flat: Flattened vector field, shape (dimension × N,)

        Returns:
            Divergence of F, flattened, shape (N,)

        Example:
            >>> # 2D case: F.shape = (2, 50, 50) → F_flat.shape = (5000,)
            >>> # Output: div_F_flat.shape = (2500,)

        Note:
            Issue #625: Migrated from tensor_calculus to stencils module.
        """
        from mfgarchon.operators.stencils.finite_difference import gradient_central

        # Reshape to vector field (dimension, Nx, Ny, ...)
        F = F_flat.reshape((self.dimension, *self.field_shape))

        # Apply ghost cell padding if BC provided (for non-periodic)
        if self.bc is not None:
            from mfgarchon.geometry.boundary import pad_array_with_ghosts

            # Pad each component separately
            F_work = np.stack(
                [
                    pad_array_with_ghosts(F[d], self.bc, ghost_depth=1, time=self.time, spacing=self.spacings)
                    for d in range(self.dimension)
                ],
                axis=0,
            )
        else:
            F_work = F

        # Compute divergence: sum of ∂Fᵢ/∂xᵢ for each component
        div_F = np.zeros(F_work[0].shape, dtype=F.dtype)
        for d in range(self.dimension):
            h = self.spacings[d]
            div_F += gradient_central(F_work[d], axis=d, h=h)

        # Extract interior if ghost cells were added
        if self.bc is not None:
            slices = [slice(1, -1)] * len(self.field_shape)
            div_F = div_F[tuple(slices)]

        # Return flattened
        return div_F.ravel()

    def __call__(self, F: NDArray) -> NDArray:
        """
        Apply divergence to vector field (convenience method).

        This preserves the field shape, unlike matrix-vector product which
        operates on flattened arrays.

        Args:
            F: Vector field, shape (dimension, Nx, Ny, ...)

        Returns:
            Divergence of F, shape (Nx, Ny, ...)

        Example:
            >>> div_op = DivergenceOperator(spacings=[0.1, 0.1], field_shape=(50, 50))
            >>> F = np.random.rand(2, 50, 50)
            >>> div_F = div_op(F)  # Shape: (50, 50)
        """
        # Validate input shape
        expected_shape = (self.dimension, *self.field_shape)
        if F.shape != expected_shape:
            raise ValueError(f"Vector field shape {F.shape} doesn't match expected {expected_shape}")

        # Apply via _matvec
        div_F_flat = self._matvec(F.ravel())
        return div_F_flat.reshape(self.field_shape)

    def __repr__(self) -> str:
        """String representation of operator."""
        return (
            f"DivergenceOperator(\n"
            f"  dimension={self.dimension},\n"
            f"  field_shape={self.field_shape},\n"
            f"  spacings={self.spacings},\n"
            f"  bc={'None (periodic)' if self.bc is None else type(self.bc).__name__},\n"
            f"  operator_shape={self.shape}\n"
            f")"
        )


if __name__ == "__main__":
    """Smoke test for DivergenceOperator."""
    import numpy as np

    from mfgarchon.geometry.boundary import neumann_bc

    print("Testing DivergenceOperator...")

    # Test 2D: F = (x, y) → div(F) = 2
    print("\n[Test 1: Analytical divergence]")
    Nx, Ny = 50, 50
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    F = np.stack([X, Y], axis=0)  # F = (x, y)

    bc = neumann_bc(dimension=2)
    div_op = DivergenceOperator(spacings=[dx, dy], field_shape=(Nx, Ny), bc=bc)

    print(f"  Operator: {div_op.shape}")
    print(f"  Vector field shape: {F.shape}")

    # Test callable interface
    div_F = div_op(F)
    print(f"  Divergence shape: {div_F.shape}")
    mean_div = np.mean(div_F[5:-5, 5:-5])  # Interior points
    print(f"  Mean div(x, y) = {mean_div:.3f} (expected = 2.0)")
    assert div_F.shape == (Nx, Ny)
    assert 1.8 < mean_div < 2.2, f"Expected ~2.0, got {mean_div}"
    print("  ✓ Callable interface works")

    # Test matrix-vector interface
    print("\n[Test 2: LinearOperator interface]")
    F_flat = F.ravel()
    div_F_flat = div_op @ F_flat

    print(f"  Input (flattened): {F_flat.shape}")
    print(f"  Output (flattened): {div_F_flat.shape}")
    div_F_reshaped = div_F_flat.reshape(Nx, Ny)
    error = np.max(np.abs(div_F_reshaped - div_F))
    print(f"  Consistency check: max|matvec - callable| = {error:.2e}")
    assert error < 1e-12
    print("  ✓ LinearOperator interface works")

    # Test 1D case
    print("\n[Test 3: 1D divergence]")
    Nx_1d = 100
    x_1d = np.linspace(0, 1, Nx_1d)
    dx_1d = x_1d[1] - x_1d[0]

    F_1d = x_1d**2  # F = x² → div(F) = dF/dx = 2x
    F_1d = F_1d[np.newaxis, :]  # Shape: (1, 100)

    div_op_1d = DivergenceOperator(spacings=[dx_1d], field_shape=(Nx_1d,))
    div_F_1d = div_op_1d(F_1d)

    expected_1d = 2 * x_1d
    error_1d = np.max(np.abs(div_F_1d[5:-5] - expected_1d[5:-5]))
    print(f"  Interior error: {error_1d:.2e} (expected = 2x)")
    assert error_1d < 0.05
    print("  ✓ 1D divergence works")

    print("\n✅ All DivergenceOperator tests passed!")

"""
Diffusion operator for tensor product grids.

This module provides a unified diffusion operator ∇·(Σ∇u) that handles:
- Scalar coefficients: σ → σ²Δu (isotropic diffusion)
- Tensor coefficients: Σ → ∇·(Σ∇u) (anisotropic diffusion)
- Spatially varying tensors: Σ(x) → ∇·(Σ(x)∇u)

Mathematical Background:
    Isotropic diffusion:
        D = σ²,  ∇·(D∇u) = D·Δu = σ²(∂²u/∂x² + ∂²u/∂y² + ...)

    Anisotropic diffusion with constant tensor Σ:
        ∇·(Σ∇u) = Σᵢⱼ ∂²u/∂xᵢ∂xⱼ

    Anisotropic diffusion with spatially varying tensor Σ(x):
        ∇·(Σ(x)∇u) = Σᵢⱼ ∂²u/∂xᵢ∂xⱼ + (∂Σᵢⱼ/∂xᵢ)(∂u/∂xⱼ)

    The flux-based discretization computes:
        1. Gradients at cell faces
        2. Face-averaged tensor components
        3. Flux = Σ·∇u at faces
        4. Divergence of fluxes

References:
    - LeVeque (2007): Finite Difference Methods for ODEs and PDEs
    - Strang (2007): Computational Science and Engineering

Created: 2026-01-25 (Issue #625 - tensor_calculus migration)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse.linalg import LinearOperator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mfgarchon.geometry.boundary import BoundaryConditions

# =============================================================================
# Numba JIT Support
# =============================================================================

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        """Dummy decorator when Numba not available."""

        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


USE_NUMBA = os.environ.get("MFG_USE_NUMBA", "auto")
if USE_NUMBA == "auto":
    USE_NUMBA = NUMBA_AVAILABLE
elif USE_NUMBA.lower() in ("true", "1", "yes"):
    USE_NUMBA = True
else:
    USE_NUMBA = False


class DiffusionOperator(LinearOperator):
    """
    Unified diffusion operator ∇·(D∇u) for tensor product grids.

    Path A (RFC #1596): the ``coefficient`` is the already-converted PDE **diffusion tensor D**
    (D = σ²/2 for a scalar volatility σ; D = ½ S Sᵀ for a std-dev matrix S). The operator applies
    D directly with NO internal squaring on any branch — squaring/conversion σ→D lives in the one
    owner ``diffusion_from_volatility`` (Issue #811). If you have a VOLATILITY σ, construct via
    :meth:`from_volatility`, which routes the conversion through that single source.

    Dispatch on ``coefficient`` shape (all values are D, used directly):
        - scalar D      → isotropic:  D·Δu
        - (d,) vector   → diagonal anisotropic: diag(D)·∂²  (D_i per axis)
        - (d,d) matrix  → constant anisotropic: ∇·(D∇u)  (D symmetric PSD, gated)
        - (*shape,d,d)  → spatially varying anisotropic

    (Before RFC #1596 the scalar branch squared its input (σ→σ²) while the vector/tensor branches
    used the input directly, so identical isotropic physics gave a 10× different result depending
    on whether it was written as a scalar, vector, or diagonal matrix — the #1549 shape-flip.)

    Implements scipy.sparse.linalg.LinearOperator interface for compatibility
    with iterative solvers and operator composition.

    Attributes:
        coefficient: PDE diffusion tensor D (scalar, diagonal vector, tensor, or field)
        spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
        field_shape: Shape of input field (N₀, N₁, ...)
        bc: Boundary conditions
        shape: Operator shape (N, N) where N = ∏field_shape
        dtype: Data type (float64)

    Usage:
        >>> # Isotropic diffusion — coefficient is D (= σ²/2 for volatility σ)
        >>> Dop = DiffusionOperator(coefficient=0.005, spacings=[0.1, 0.1],
        ...                         field_shape=(50, 50), bc=bc)
        >>> Du = Dop(u)  # Computes 0.005 * Δu
        >>>
        >>> # Same physics from a volatility σ=0.1 via the single-source converter
        >>> Dop = DiffusionOperator.from_volatility(0.1, spacings=[0.1, 0.1],
        ...                                         field_shape=(50, 50), bc=bc)
        >>> Du = Dop(u)  # Computes (0.1²/2) * Δu = 0.005 * Δu (identical to above)
        >>>
        >>> # Anisotropic diffusion (constant diffusion tensor D, symmetric PSD)
        >>> Dtensor = np.array([[0.02, 0.0], [0.0, 0.005]])
        >>> Dop = DiffusionOperator(coefficient=Dtensor, spacings=[0.1, 0.1],
        ...                         field_shape=(50, 50), bc=bc)
        >>> Du = Dop(u)  # Computes ∇·(D∇u)
    """

    def __init__(
        self,
        coefficient: float | NDArray,
        spacings: Sequence[float],
        field_shape: tuple[int, ...] | int,
        bc: BoundaryConditions | None = None,
        time: float = 0.0,
    ):
        """
        Initialize diffusion operator.

        Args:
            coefficient: PDE diffusion tensor D, applied directly (RFC #1596; no internal
                squaring — pass D = σ²/2, or use :meth:`from_volatility` for a volatility σ):
                - scalar D: isotropic diffusion D·Δu
                - (d,) array: diagonal anisotropic, D_i per axis
                - (d, d) array: constant diffusion tensor D (symmetric PSD, validated)
                - (*field_shape, d, d) array: spatially varying tensor D(x)
            spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
            field_shape: Shape of field arrays (N₀, N₁, ...) or N for 1D
            bc: Boundary conditions (None for periodic)
            time: Time for time-dependent BCs (default 0.0)

        Raises:
            ValueError: If coefficient shape is invalid for the field shape
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
        self._ndim = len(field_shape)

        # Validate spacings
        if len(self.spacings) != self._ndim:
            raise ValueError(f"spacings length {len(self.spacings)} != field_shape dimensions {self._ndim}")

        # Process and validate coefficient
        self.coefficient, self._coeff_type = self._process_coefficient(coefficient)

        # Compute operator shape
        N = int(np.prod(field_shape))
        super().__init__(shape=(N, N), dtype=np.float64)

    @classmethod
    def from_volatility(
        cls,
        sigma: float | NDArray,
        spacings: Sequence[float],
        field_shape: tuple[int, ...] | int,
        bc: BoundaryConditions | None = None,
        time: float = 0.0,
    ) -> DiffusionOperator:
        """Construct from SDE **volatility** σ (a standard deviation), converting σ→D through the
        single-source :func:`diffusion_from_volatility` (Issue #811, RFC #1596).

        Use this when you hold a volatility; the plain constructor takes the already-converted
        diffusion tensor D directly (Path A). Shapes: scalar σ → D = σ²/2; ``(d,)`` per-axis σ →
        D_i = σ_i²/2; ``(d, d)`` symmetric std-dev matrix S → D = ½ S Sᵀ (S gated symmetric PSD).
        """
        from mfgarchon.utils.pde_coefficients import diffusion_from_volatility, validate_symmetric_psd

        if np.isscalar(sigma):
            D: float | NDArray = diffusion_from_volatility(float(sigma))
        else:
            S = np.asarray(sigma, dtype=float)
            if S.ndim == 0:
                D = diffusion_from_volatility(S)
            elif S.ndim == 1:
                D = diffusion_from_volatility(S, kind="field")
            elif S.ndim == 2:
                validate_symmetric_psd(S, name="DiffusionOperator volatility")
                D = diffusion_from_volatility(S, kind="tensor")
            else:
                D = diffusion_from_volatility(S, kind="tensor")  # (*spatial, d, d) noise matrix
        return cls(D, spacings, field_shape, bc=bc, time=time)

    def _process_coefficient(self, coeff: float | NDArray) -> tuple[float | NDArray, str]:
        """
        Process coefficient and determine type.

        Returns:
            Tuple of (processed_coefficient, coefficient_type)
            where type is one of: "scalar", "constant_tensor", "varying_tensor"
        """
        d = self._ndim

        if np.isscalar(coeff):
            return float(coeff), "scalar"

        coeff = np.asarray(coeff)

        if coeff.ndim == 0:
            # 0-d array (scalar wrapped in array)
            return float(coeff), "scalar"

        if coeff.ndim == 1 and len(coeff) == d:
            # Diagonal diffusion tensor: [D₀, D₁, ...] → diag(D) (values are D_i, used directly).
            return np.diag(coeff), "constant_tensor"

        if coeff.ndim == 2 and coeff.shape == (d, d):
            # Constant diffusion tensor D. Gate symmetry + PSD (RFC #1596): a diffusion tensor
            # is symmetric PSD; an asymmetric input is caller confusion (e.g. a Cholesky factor).
            from mfgarchon.utils.pde_coefficients import validate_symmetric_psd

            validate_symmetric_psd(coeff, name="DiffusionOperator diffusion tensor D")
            return coeff, "constant_tensor"

        if coeff.shape == (*self.field_shape, d, d):
            # Spatially varying tensor D(x). Gate symmetry (RFC #1596), vectorized over the field:
            # the downstream kernel reads D[...,i,j] and D[...,j,i] independently, so an asymmetric
            # D_field would inject a spurious antisymmetric-advection term. Per-point PSD is left to
            # the caller (an O(N) eigendecomposition per grid point is too costly here); symmetry is
            # the load-bearing check and is one cheap array op.
            max_asym = float(np.max(np.abs(coeff - np.swapaxes(coeff, -1, -2)))) if coeff.size else 0.0
            if max_asym > 1e-10:
                raise ValueError(
                    f"DiffusionOperator spatially-varying diffusion tensor D(x) must be symmetric at "
                    f"every point (it is 1/2 S S^T for a std-dev matrix S; RFC #1596). "
                    f"Max asymmetry |D - D^T| = {max_asym:.3e}."
                )
            return coeff, "varying_tensor"

        raise ValueError(
            f"Invalid coefficient shape {coeff.shape} for {d}D field. "
            f"Expected scalar, ({d},), ({d},{d}), or {(*self.field_shape, d, d)}."
        )

    def _matvec(self, u_flat: NDArray) -> NDArray:
        """
        Apply diffusion operator to flattened field.

        This is the core LinearOperator method required by scipy.

        Args:
            u_flat: Flattened field array, shape (N,)

        Returns:
            Diffusion of u, flattened, shape (N,)
        """
        # Reshape to field
        u = u_flat.reshape(self.field_shape)

        # Apply diffusion based on coefficient type
        if self._coeff_type == "scalar":
            result = self._apply_scalar_diffusion(u)
        else:
            result = self._apply_tensor_diffusion(u)

        return result.ravel()

    def __call__(self, u: NDArray) -> NDArray:
        """
        Apply diffusion operator to field (preserves shape).

        Args:
            u: Field array, shape field_shape or (N,)

        Returns:
            Diffusion of u, same shape as input
        """
        # Handle already-flattened input
        if u.ndim == 1:
            return self._matvec(u)

        # Handle field input
        if u.shape != self.field_shape:
            raise ValueError(f"Input shape {u.shape} doesn't match field_shape {self.field_shape}")

        result_flat = self._matvec(u.ravel())
        return result_flat.reshape(self.field_shape)

    def _apply_scalar_diffusion(self, u: NDArray) -> NDArray:
        """
        Apply isotropic diffusion: D·Δu (RFC #1596: coefficient is the diffusion D, not σ;
        no internal squaring — conversion σ→D lives in ``diffusion_from_volatility``).

        Uses stencil-based Laplacian with BC handling.
        """
        from mfgarchon.operators.stencils.finite_difference import laplacian_with_bc

        D = float(self.coefficient)
        lap = laplacian_with_bc(u, self.spacings, bc=self.bc, time=self.time)
        return D * lap

    def _apply_tensor_diffusion(self, u: NDArray) -> NDArray:
        """
        Apply anisotropic diffusion: ∇·(Σ∇u).

        Issue #1228: single-sourced through ``utils.numerical.tensor_calculus.diffusion`` (the
        unified lower-level operator) instead of a private per-dimension copy. Verified
        bit-identical (max|Δ| = 0) across 1D/2D/3D, scalar / diagonal / full / spatially-varying
        tensors, and periodic / no-flux / Dirichlet BCs before the private
        ``_tensor_diffusion_{1d,2d,nd}`` were removed. ``tensor_calculus.diffusion`` handles the
        constant-(d,d) tensor directly (no broadcast needed).
        """
        from mfgarchon.utils.numerical.tensor_calculus import diffusion as tensor_calculus_diffusion

        return tensor_calculus_diffusion(u, self.coefficient, self.spacings, bc=self.bc, time=self.time)

    def __repr__(self) -> str:
        """String representation for debugging."""
        bc_str = f"bc={self.bc.bc_type.value}" if self.bc else "bc=periodic"
        coeff_str = (
            f"coefficient={self.coefficient}"
            if self._coeff_type == "scalar"
            else f"coefficient_type={self._coeff_type}"
        )
        return (
            f"DiffusionOperator(\n"
            f"  field_shape={self.field_shape},\n"
            f"  spacings={self.spacings},\n"
            f"  {coeff_str},\n"
            f"  {bc_str},\n"
            f"  shape={self.shape}\n"
            f")"
        )


# =============================================================================
# Numba JIT Kernels
# =============================================================================


@njit(cache=True)
def _compute_tensor_kernel_2d(
    m_padded: np.ndarray,
    Sigma: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """JIT-compiled kernel for 2D full tensor diffusion."""
    Ny, Nx = Sigma.shape[0], Sigma.shape[1]
    result = np.zeros((Ny, Nx))

    for i in range(Ny):
        for j in range(Nx):
            s11 = Sigma[i, j, 0, 0]
            s12 = Sigma[i, j, 0, 1]
            s21 = Sigma[i, j, 1, 0]
            s22 = Sigma[i, j, 1, 1]

            # Face-averaged tensor components
            if j < Nx - 1:
                s11_xp = 0.5 * (s11 + Sigma[i, j + 1, 0, 0])
                s12_xp = 0.5 * (s12 + Sigma[i, j + 1, 0, 1])
            else:
                s11_xp, s12_xp = s11, s12

            if j > 0:
                s11_xm = 0.5 * (s11 + Sigma[i, j - 1, 0, 0])
                s12_xm = 0.5 * (s12 + Sigma[i, j - 1, 0, 1])
            else:
                s11_xm, s12_xm = s11, s12

            if i < Ny - 1:
                s21_yp = 0.5 * (s21 + Sigma[i + 1, j, 1, 0])
                s22_yp = 0.5 * (s22 + Sigma[i + 1, j, 1, 1])
            else:
                s21_yp, s22_yp = s21, s22

            if i > 0:
                s21_ym = 0.5 * (s21 + Sigma[i - 1, j, 1, 0])
                s22_ym = 0.5 * (s22 + Sigma[i - 1, j, 1, 1])
            else:
                s21_ym, s22_ym = s21, s22

            # Padded indices
            ip, jp = i + 1, j + 1

            # Gradients at faces
            dm_dx_xp = (m_padded[ip, jp + 1] - m_padded[ip, jp]) / dx
            dm_dy_xp = (
                0.25
                * (
                    (m_padded[ip + 1, jp + 1] - m_padded[ip - 1, jp + 1])
                    + (m_padded[ip + 1, jp] - m_padded[ip - 1, jp])
                )
                / dy
            )

            dm_dx_xm = (m_padded[ip, jp] - m_padded[ip, jp - 1]) / dx
            dm_dy_xm = (
                0.25
                * (
                    (m_padded[ip + 1, jp] - m_padded[ip - 1, jp])
                    + (m_padded[ip + 1, jp - 1] - m_padded[ip - 1, jp - 1])
                )
                / dy
            )

            dm_dy_yp = (m_padded[ip + 1, jp] - m_padded[ip, jp]) / dy
            dm_dx_yp = (
                0.25
                * (
                    (m_padded[ip + 1, jp + 1] - m_padded[ip + 1, jp - 1])
                    + (m_padded[ip, jp + 1] - m_padded[ip, jp - 1])
                )
                / dx
            )

            dm_dy_ym = (m_padded[ip, jp] - m_padded[ip - 1, jp]) / dy
            dm_dx_ym = (
                0.25
                * (
                    (m_padded[ip, jp + 1] - m_padded[ip, jp - 1])
                    + (m_padded[ip - 1, jp + 1] - m_padded[ip - 1, jp - 1])
                )
                / dx
            )

            # Fluxes
            Fx_xp = s11_xp * dm_dx_xp + s12_xp * dm_dy_xp
            Fx_xm = s11_xm * dm_dx_xm + s12_xm * dm_dy_xm
            Fy_yp = s21_yp * dm_dx_yp + s22_yp * dm_dy_yp
            Fy_ym = s21_ym * dm_dx_ym + s22_ym * dm_dy_ym

            # Divergence
            result[i, j] = (Fx_xp - Fx_xm) / dx + (Fy_yp - Fy_ym) / dy

    return result


# =============================================================================
# Convenience Function
# =============================================================================


def apply_diffusion(
    u: NDArray,
    coefficient: float | NDArray,
    spacings: Sequence[float],
    bc: BoundaryConditions | None = None,
    time: float = 0.0,
) -> NDArray:
    """
    Apply diffusion operator ∇·(Σ∇u) to a field.

    This is a convenience function that creates a DiffusionOperator
    and applies it in one call. For repeated application with the
    same coefficient, prefer creating the operator once.

    Args:
        u: Input field array
        coefficient: Diffusion coefficient (scalar or tensor)
        spacings: Grid spacing per dimension
        bc: Boundary conditions (None for periodic)
        time: Time for time-dependent BCs

    Returns:
        Diffusion of u, same shape as input

    Example:
        >>> from mfgarchon.operators.differential.diffusion import apply_diffusion
        >>> result = apply_diffusion(u, sigma=0.1, spacings=[dx, dy], bc=bc)
    """
    op = DiffusionOperator(
        coefficient=coefficient,
        spacings=spacings,
        field_shape=u.shape,
        bc=bc,
        time=time,
    )
    return op(u)


# =============================================================================
# Smoke Tests
# =============================================================================

if __name__ == "__main__":
    """Smoke test for DiffusionOperator."""
    print("Testing DiffusionOperator...")

    from mfgarchon.geometry.boundary import neumann_bc, periodic_bc

    # Test 1D isotropic
    print("\n[1D Isotropic Diffusion]")
    x = np.linspace(0, 2 * np.pi, 100)
    dx = x[1] - x[0]
    u_1d = np.sin(x)
    bc_1d = periodic_bc(dimension=1)

    D_1d = DiffusionOperator(coefficient=1.0, spacings=[dx], field_shape=100, bc=bc_1d)
    print(f"  Operator: {D_1d}")
    Du_1d = D_1d(u_1d)
    print(f"  Input shape: {u_1d.shape}, Output shape: {Du_1d.shape}")

    # For u = sin(x), σ²Δu = -sin(x) when σ=1
    expected = -np.sin(x)
    error_1d = np.max(np.abs(Du_1d[5:-5] - expected[5:-5]))
    print(f"  Error (interior): {error_1d:.2e}")
    assert error_1d < 0.01, f"1D isotropic error too large: {error_1d}"
    print("  OK")

    # Test 2D isotropic
    print("\n[2D Isotropic Diffusion]")
    Nx, Ny = 50, 50
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    u_2d = X**2 + Y**2  # Δu = 4
    bc_2d = neumann_bc(dimension=2)

    D_2d = DiffusionOperator(coefficient=1.0, spacings=[dx, dy], field_shape=(Nx, Ny), bc=bc_2d)
    Du_2d = D_2d(u_2d)
    print(f"  Input shape: {u_2d.shape}, Output shape: {Du_2d.shape}")

    # For u = x² + y², σ²Δu = 4 when σ=1
    interior = Du_2d[5:-5, 5:-5]
    mean_val = np.mean(interior)
    print(f"  Δ(x²+y²) interior mean: {mean_val:.3f} (expected = 4.0)")
    assert 3.5 < mean_val < 4.5, f"2D isotropic mean {mean_val} outside range"
    print("  OK")

    # Test 2D anisotropic (constant tensor)
    print("\n[2D Anisotropic Diffusion - Constant Tensor]")
    Sigma = np.array([[0.1, 0.0], [0.0, 0.05]])
    D_aniso = DiffusionOperator(coefficient=Sigma, spacings=[dx, dy], field_shape=(Nx, Ny), bc=bc_2d)
    Du_aniso = D_aniso(u_2d)
    print(f"  Tensor Σ:\n    {Sigma}")
    print(f"  Output shape: {Du_aniso.shape}")
    assert not np.any(np.isnan(Du_aniso)), "NaN in anisotropic result"
    print("  OK")

    # Test 2D anisotropic (spatially varying)
    print("\n[2D Anisotropic Diffusion - Spatially Varying]")
    Sigma_field = np.zeros((Nx, Ny, 2, 2))
    Sigma_field[..., 0, 0] = 0.1 * (1 + X)  # σ_xx varies with x
    Sigma_field[..., 1, 1] = 0.05 * (1 + Y)  # σ_yy varies with y
    D_varying = DiffusionOperator(coefficient=Sigma_field, spacings=[dx, dy], field_shape=(Nx, Ny), bc=bc_2d)
    Du_varying = D_varying(u_2d)
    print(f"  Σ_xx range: [{Sigma_field[..., 0, 0].min():.2f}, {Sigma_field[..., 0, 0].max():.2f}]")
    print(f"  Σ_yy range: [{Sigma_field[..., 1, 1].min():.2f}, {Sigma_field[..., 1, 1].max():.2f}]")
    print(f"  Output shape: {Du_varying.shape}")
    assert not np.any(np.isnan(Du_varying)), "NaN in varying tensor result"
    print("  OK")

    # Test convenience function
    print("\n[Convenience Function]")
    result = apply_diffusion(u_2d, coefficient=1.0, spacings=[dx, dy], bc=bc_2d)
    assert np.allclose(result, Du_2d), "apply_diffusion doesn't match operator"
    print("  apply_diffusion() matches DiffusionOperator()")
    print("  OK")

    # Test scipy compatibility
    print("\n[scipy Compatibility]")
    from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator

    assert isinstance(D_2d, ScipyLinearOperator)
    print("  isinstance(D, scipy.sparse.linalg.LinearOperator)")

    # Test @ syntax
    u_flat = u_2d.ravel()
    Du_matvec = D_2d @ u_flat
    assert np.allclose(Du_2d.ravel(), Du_matvec)
    print("  D(u) == D @ u.ravel()")
    print("  OK")

    print("\nAll DiffusionOperator tests passed!")

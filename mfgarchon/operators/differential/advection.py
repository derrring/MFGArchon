"""
Advection operator for tensor product grids.

This module provides LinearOperator implementation of discrete advection
for structured grids, using finite difference stencils.

Mathematical Background:
    Advection operator models transport by a velocity field v:

    1. **Gradient form** (non-conservative): A(m) = v·∇m
       - Used in HJB equations: ∂u/∂t + H(∇u) = 0 where H(p) = max_α[α·p - L(α)]
       - Non-conservative: doesn't preserve integral ∫m dx

    2. **Divergence form** (conservative): A(m) = ∇·(vm)
       - Used in Fokker-Planck equations: ∂m/∂t + ∇·(αm) = 0
       - Conservative: preserves mass ∫m dx = const

    Relationship:
        ∇·(vm) = v·∇m + m(∇·v)  (product rule)

        If ∇·v = 0 (incompressible flow), both forms are equivalent.

    Discretization Schemes:
        - **Upwind**: 1st-order, stable, dissipative (numerical viscosity)
        - **Centered**: 2nd-order, may oscillate without stabilization

References:
    - LeVeque (2002): Finite Volume Methods for Hyperbolic Problems
    - Godunov & Ryabenkii (1987): Difference Schemes
    - Achdou & Capuzzo-Dolcetta (2010): Mean Field Games

Created: 2026-01-17 (Issue #595 Phase 2 - Operator Refactoring)
Part of: Geometry Operator LinearOperator Migration
Migrated: 2026-01-25 (Issue #625 - tensor_calculus → stencils migration)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.sparse.linalg import LinearOperator

if TYPE_CHECKING:
    from collections.abc import Sequence

    import scipy.sparse as sparse
    from numpy.typing import NDArray

    from mfgarchon.geometry.boundary import BoundaryConditions


class AdvectionOperator(LinearOperator):
    """
    Discrete advection operator for tensor product grids.

    Implements scipy.sparse.linalg.LinearOperator interface for compatibility
    with iterative solvers and operator composition.

    Uses finite difference stencils with grid-specific parameters and velocity
    field curried into the operator object.

    **Mathematical Context**:
        Advection describes transport by a velocity/drift field:
            - Gradient form: v·∇m (HJB equations)
            - Divergence form: ∇·(vm) (Fokker-Planck equations)

    **MFG Applications**:
        - **HJB equation**: ∂u/∂t + v·∇u = 0 (use form="gradient")
        - **FP equation**: ∂m/∂t + ∇·(vm) = 0 (use form="divergence")

    **Operator Shape**:
        Input:  Scalar field m flattened to shape (N,)
        Output: Advection term flattened to shape (N,)
        Operator shape: (N, N)

        where N = ∏field_shape (total number of grid points)

    Attributes:
        velocity_field: Drift/velocity field, shape (dimension, Nx, Ny, ...)
        spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
        field_shape: Shape of scalar fields (Nx, Ny, ...) or (Nx,) for 1D
        scheme: "upwind" (1st-order) or "centered" (2nd-order)
        form: "gradient" (v·∇m) or "divergence" (∇·(vm))
        bc: Boundary conditions (None for periodic)
        time: Time for time-dependent BCs
        shape: Operator shape (N, N)
        dtype: Data type (float64)

    Usage:
        >>> # Create operator with upwind scheme
        >>> v = np.ones((2, 50, 50))  # Constant velocity (1, 1)
        >>> adv_op = AdvectionOperator(
        ...     velocity_field=v,
        ...     spacings=[0.1, 0.1],
        ...     field_shape=(50, 50),
        ...     scheme="upwind",
        ...     form="divergence",
        ...     bc=bc
        ... )
        >>>
        >>> # Apply via matrix-vector product
        >>> m_flat = m.ravel()
        >>> adv_m_flat = adv_op @ m_flat
        >>>
        >>> # Apply via callable (preserves field shape)
        >>> adv_m = adv_op(m)  # Input/Output: (50, 50)

    Example:
        >>> import numpy as np
        >>> from mfgarchon.geometry.boundary import no_flux_bc
        >>>
        >>> # Test conservative advection: ∇·(vm) with constant v
        >>> Nx, Ny = 100, 100
        >>> x = np.linspace(0, 1, Nx)
        >>> y = np.linspace(0, 1, Ny)
        >>> X, Y = np.meshgrid(x, y, indexing='ij')
        >>>
        >>> # Linear density m = x
        >>> m = X
        >>> # Constant velocity v = (1, 0)
        >>> v = np.zeros((2, Nx, Ny))
        >>> v[0] = 1.0
        >>>
        >>> bc = no_flux_bc(dimension=2)
        >>> adv_op = AdvectionOperator(
        ...     velocity_field=v,
        ...     spacings=[0.01, 0.01],
        ...     field_shape=(Nx, Ny),
        ...     scheme="upwind",
        ...     form="gradient",  # v·∇m
        ...     bc=bc
        ... )
        >>>
        >>> # v·∇(x) = (1,0)·(1,0) = 1.0
        >>> adv_m = adv_op(m)
        >>> print(f"Mean advection: {np.mean(adv_m):.3f}")  # ~1.0
    """

    def __init__(
        self,
        velocity_field: NDArray,
        spacings: Sequence[float],
        field_shape: tuple[int, ...] | int,
        scheme: Literal["upwind", "centered"] = "upwind",
        form: Literal["gradient", "divergence"] = "divergence",
        bc: BoundaryConditions | None = None,
        time: float = 0.0,
        mass_conservative: bool = False,
    ):
        """
        Initialize advection operator.

        Args:
            velocity_field: Velocity/drift field, shape (dimension, Nx, Ny, ...)
            spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
            field_shape: Shape of scalar field arrays (Nx, Ny, ...) or Nx for 1D
            scheme: Discretization scheme
                - "upwind": 1st-order upwind (stable, dissipative)
                - "centered": 2nd-order centered (may oscillate)
            form: Advection form
                - "gradient": v·∇m (non-conservative, for HJB)
                - "divergence": ∇·(vm) (conservative, for FP)
            bc: Boundary conditions (None for periodic/wrap)
            time: Time for time-dependent BCs (default 0.0)
            mass_conservative: If True (only for ``form="divergence"``, ``scheme="upwind"``),
                use a finite-volume flux-difference discretisation that is discretely
                mass-conservative at no-flux walls — the advective flux through a no-flux
                boundary face is set to zero, so ``1ᵀA = 0`` (column-conservative) and a
                density piled against a wall by strong drift does not leak (Issue #1184).
                Default ``False`` keeps the node-based ``gradient_upwind`` divergence
                (byte-identical), which is conservative in the interior but leaks
                ``±(v·m)`` at no-flux walls. Mirrors ``LaplacianOperator.mass_conservative``
                (Issue #1184 diffusion half / #1202).

        Raises:
            ValueError: If scheme or form invalid
            ValueError: If velocity_field shape incompatible with field_shape
        """
        # Handle 1D shape
        if isinstance(field_shape, int):
            field_shape = (field_shape,)
        else:
            field_shape = tuple(field_shape)

        # Validate velocity_field shape
        expected_v_shape = (len(field_shape), *field_shape)
        if velocity_field.shape != expected_v_shape:
            raise ValueError(
                f"velocity_field shape {velocity_field.shape} doesn't match "
                f"expected {expected_v_shape} for field_shape={field_shape}"
            )

        # Validate scheme and form
        if scheme not in ("upwind", "centered"):
            raise ValueError(f"Unknown scheme: {scheme}. Use 'upwind' or 'centered'.")

        if form not in ("gradient", "divergence"):
            raise ValueError(f"Unknown form: {form}. Use 'gradient' or 'divergence'.")

        if mass_conservative and not (form == "divergence" and scheme == "upwind"):
            raise ValueError(
                "mass_conservative=True is only supported for form='divergence' with "
                f"scheme='upwind' (got form={form!r}, scheme={scheme!r})."
            )

        self.velocity_field = velocity_field
        self.spacings = list(spacings)
        self.field_shape = field_shape
        self.scheme = scheme
        self.form = form
        self.bc = bc
        self.time = time
        self.mass_conservative = mass_conservative
        self.dimension = len(field_shape)

        # Validate
        if len(self.spacings) != self.dimension:
            raise ValueError(f"spacings length {len(self.spacings)} != field_shape dimensions {self.dimension}")

        # Compute operator shape (N, N) - scalar field to scalar field
        N = int(np.prod(field_shape))
        super().__init__(shape=(N, N), dtype=np.float64)

    def _matvec(self, m_flat: NDArray) -> NDArray:
        """
        Apply advection to flattened scalar field.

        This is the core LinearOperator method required by scipy.

        Args:
            m_flat: Flattened scalar field, shape (N,)

        Returns:
            Advection term, flattened, shape (N,)
                - Gradient form: v·∇m
                - Divergence form: ∇·(vm)

        Note:
            Issue #625: Migrated from tensor_calculus to stencils module.
        """
        from mfgarchon.operators.stencils.finite_difference import (
            gradient_central,
            gradient_upwind,
        )

        # Select gradient function based on scheme
        grad_fn = gradient_upwind if self.scheme == "upwind" else gradient_central

        # Reshape to field
        m = m_flat.reshape(self.field_shape)

        # Issue #1184: opt-in discretely-conservative finite-volume divergence.
        if self.mass_conservative:
            return self._divergence_upwind_conservative(m).ravel()

        # Apply ghost cell padding if BC provided (for non-periodic)
        if self.bc is not None:
            from mfgarchon.geometry.boundary import pad_array_with_ghosts

            m_work = pad_array_with_ghosts(m, self.bc, ghost_depth=1, time=self.time)
            # Also pad velocity field
            v_work = np.stack(
                [
                    pad_array_with_ghosts(self.velocity_field[d], self.bc, ghost_depth=1, time=self.time)
                    for d in range(self.dimension)
                ],
                axis=0,
            )
        else:
            m_work = m
            v_work = self.velocity_field

        if self.form == "gradient":
            # Gradient form: v·∇m = ∑ vᵢ * ∂m/∂xᵢ
            adv_m = np.zeros_like(m_work)
            for d in range(self.dimension):
                h = self.spacings[d]
                dm_dxi = grad_fn(m_work, axis=d, h=h)
                adv_m += v_work[d] * dm_dxi

        elif self.form == "divergence":
            # Divergence form: ∇·(vm) = ∑ ∂(vᵢ*m)/∂xᵢ
            adv_m = np.zeros_like(m_work)
            for d in range(self.dimension):
                h = self.spacings[d]
                flux = v_work[d] * m_work  # flux component
                adv_m += grad_fn(flux, axis=d, h=h)
        else:
            raise ValueError(f"Unknown form: {self.form}")

        # Extract interior if ghost cells were added
        if self.bc is not None:
            slices = [slice(1, -1)] * len(self.field_shape)
            adv_m = adv_m[tuple(slices)]

        # Return flattened
        return adv_m.ravel()

    def _divergence_upwind_conservative(self, m: NDArray) -> NDArray:
        """Discretely-conservative finite-volume upwind divergence ``∇·(vm)`` (Issue #1184).

        Per axis, builds the upwind flux at each interior cell face
        ``F_{i+1/2} = v_{i+1/2} m_i`` if ``v_{i+1/2} >= 0`` else ``v_{i+1/2} m_{i+1}``
        (``v_{i+1/2}`` = node-average), then ``div_i = (F_{i+1/2} - F_{i-1/2}) / h``. The
        flux telescopes, so the column sum equals the net boundary-face flux:

        - **No-flux BC**: the two wall faces are set to zero flux -> ``1ᵀA = 0`` exactly
          (mass conserved even when strong drift piles density against the wall).
        - **Periodic (bc=None)**: the wrap face closes the telescope -> ``1ᵀA = 0``.

        Only no-flux / Neumann / reflecting / periodic boundaries are supported here (the
        FP use case); other BCs would need an inflow flux and should not opt in.
        """
        from mfgarchon.geometry.boundary import BCType, BoundaryConditions

        # Issue #1428: an explicit periodic BoundaryConditions object must be treated like
        # bc=None (wrap-face telescoping), not rejected — previously only bc=None counted as
        # periodic, so a periodic BoundaryConditions raised despite periodic being supported.
        periodic = self.bc is None
        if not periodic:
            ok = isinstance(self.bc, BoundaryConditions) and self.bc.is_uniform
            seg_type = self.bc.segments[0].bc_type if ok else None
            if seg_type == BCType.PERIODIC:
                periodic = True
            elif seg_type not in (BCType.NO_FLUX, BCType.NEUMANN, BCType.REFLECTING):
                raise NotImplementedError(
                    "mass_conservative divergence supports only no-flux/Neumann/reflecting "
                    f"(zero wall flux) or periodic boundaries; got {self.bc!r}."
                )

        div = np.zeros_like(m, dtype=np.float64)
        for d in range(self.dimension):
            h = self.spacings[d]
            md = np.moveaxis(m, d, -1)
            vd = np.moveaxis(self.velocity_field[d], d, -1)
            # Upwind flux on interior faces i+1/2 (i = 0 .. N-2), velocity averaged to the face.
            v_face = 0.5 * (vd[..., :-1] + vd[..., 1:])
            flux = np.where(v_face >= 0.0, v_face * md[..., :-1], v_face * md[..., 1:])
            dvar = np.empty_like(md)
            if periodic:
                v_wrap = 0.5 * (vd[..., -1] + vd[..., 0])
                flux_wrap = np.where(v_wrap >= 0.0, v_wrap * md[..., -1], v_wrap * md[..., 0])
                dvar[..., 0] = (flux[..., 0] - flux_wrap) / h
                dvar[..., 1:-1] = (flux[..., 1:] - flux[..., :-1]) / h
                dvar[..., -1] = (flux_wrap - flux[..., -1]) / h
            else:
                # No-flux walls: zero flux through both boundary faces.
                dvar[..., 0] = flux[..., 0] / h
                dvar[..., 1:-1] = (flux[..., 1:] - flux[..., :-1]) / h
                dvar[..., -1] = -flux[..., -1] / h
            div += np.moveaxis(dvar, -1, d)
        return div

    def __call__(self, m: NDArray) -> NDArray:
        """
        Apply advection to scalar field (convenience method).

        This preserves the field shape, unlike matrix-vector product which
        operates on flattened arrays.

        Args:
            m: Scalar field, shape (Nx, Ny, ...)

        Returns:
            Advection term, shape (Nx, Ny, ...)
                - Gradient form: v·∇m
                - Divergence form: ∇·(vm)

        Example:
            >>> adv_op = AdvectionOperator(v, spacings=[0.1, 0.1], field_shape=(50, 50))
            >>> m = np.random.rand(50, 50)
            >>> adv_m = adv_op(m)  # Shape: (50, 50)
        """
        # Validate input shape
        if m.shape != self.field_shape:
            raise ValueError(f"Field shape {m.shape} doesn't match expected {self.field_shape}")

        # Apply via _matvec
        adv_m_flat = self._matvec(m.ravel())
        return adv_m_flat.reshape(self.field_shape)

    def as_scipy_sparse(self, max_grid_size: int = 100_000) -> sparse.spmatrix:
        """
        Convert operator to scipy sparse matrix (CSR format).

        **⚠️ IMPORTANT - Godunov Paradox Limitation**:

        This method works by probing the operator with unit vectors (e_j).
        For operators using Godunov upwinding (scheme="upwind" in tensor_calculus),
        this can produce incorrect matrices due to state-dependent flux limiting.

        **Why this matters**: Godunov upwind selects flux direction based on
        sign(∇m), which changes between localized (unit vector) and distributed
        (actual density) fields. The extracted matrix represents the operator
        on impulses, not general smooth fields.

        **Recommendation**:
        - ✅ Use this method for periodic BC or exploratory analysis
        - ❌ Do NOT use for implicit solver Jacobians
        - ✅ For implicit solvers, use velocity-based upwind sparse construction
          (see fp_fdm_alg_*.py modules)

        **See**: Godunov paradox and defect correction analysis for full
        mathematical explanation and the Defect Correction solution strategy.

        Args:
            max_grid_size: Maximum allowed grid size (default 100,000).
                Raises ValueError if exceeded.

        Returns:
            Sparse CSR matrix representing the advection operator evaluated
            on unit vectors (may not equal operator on smooth fields for Godunov).

        Raises:
            ValueError: If grid too large (N > max_grid_size)

        Example:
            >>> # Exploratory use (understand stencil structure)
            >>> adv_op = AdvectionOperator(v, spacings=[0.1], field_shape=(100,))
            >>> A_adv = adv_op.as_scipy_sparse()
            >>> print(f"Sparsity: {A_adv.nnz / (100*100) * 100:.1f}%")

            >>> # For implicit solvers, use velocity-based construction instead:
            >>> # See mfgarchon/alg/numerical/fp_solvers/fp_fdm_alg_gradient_upwind.py

        Notes:
            - Returns CSR format for efficient matrix-vector products
            - Slower than direct assembly (O(N²) vs O(N))
            - Godunov limitation documented in Issue #597 Milestone 3
            - Suitable for analysis, not for production implicit solvers
        """
        import scipy.sparse as sparse

        N = int(np.prod(self.field_shape))

        if max_grid_size < N:
            raise ValueError(
                f"Grid size {N} exceeds max_grid_size={max_grid_size}. "
                f"For large grids, use matrix-free methods (LinearOperator interface) instead."
            )

        # Build matrix by evaluating operator on unit vectors
        # This is robust and automatically correct for all schemes/BCs
        rows = []
        cols = []
        vals = []

        for j in range(N):
            # Create j-th unit vector
            e_j = np.zeros(N, dtype=np.float64)
            e_j[j] = 1.0

            # Apply operator: A @ e_j = j-th column of A
            col_j = self._matvec(e_j)

            # Extract nonzeros (threshold for numerical stability)
            nz_mask = np.abs(col_j) > 1e-14
            nz_indices = np.where(nz_mask)[0]

            if len(nz_indices) > 0:
                rows.extend(nz_indices.tolist())
                cols.extend([j] * len(nz_indices))
                vals.extend(col_j[nz_indices].tolist())

        # Build COO matrix and convert to CSR for efficient operations
        return sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()

    def __repr__(self) -> str:
        """String representation of operator."""
        return (
            f"AdvectionOperator(\n"
            f"  dimension={self.dimension},\n"
            f"  field_shape={self.field_shape},\n"
            f"  spacings={self.spacings},\n"
            f"  scheme='{self.scheme}',\n"
            f"  form='{self.form}',\n"
            f"  bc={'None (periodic)' if self.bc is None else type(self.bc).__name__},\n"
            f"  operator_shape={self.shape}\n"
            f")"
        )


if __name__ == "__main__":
    """Smoke test for AdvectionOperator."""
    import numpy as np

    from mfgarchon.geometry.boundary import neumann_bc

    print("Testing AdvectionOperator...")

    # Test 2D gradient form: v·∇m with m = x, v = (1, 0)
    print("\n[Test 1: Gradient form v·∇m]")
    Nx, Ny = 50, 50
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    m = X  # m = x
    v = np.zeros((2, Nx, Ny))
    v[0] = 1.0  # v = (1, 0)

    bc = neumann_bc(dimension=2)
    adv_op = AdvectionOperator(
        velocity_field=v,
        spacings=[dx, dy],
        field_shape=(Nx, Ny),
        scheme="upwind",
        form="gradient",
        bc=bc,
    )

    print(f"  Operator: {adv_op.shape}")
    print(f"  Velocity field shape: {v.shape}")
    print(f"  Density shape: {m.shape}")

    # Test callable interface
    # v·∇(x) = (1,0)·(1,0) = 1.0
    adv_m = adv_op(m)
    print(f"  Advection shape: {adv_m.shape}")
    mean_adv = np.mean(adv_m[5:-5, 5:-5])  # Interior points
    print(f"  Mean v·∇(x) = {mean_adv:.3f} (expected ≈ 1.0)")
    assert adv_m.shape == (Nx, Ny)
    assert 0.8 < mean_adv < 1.2, f"Expected ~1.0, got {mean_adv}"
    print("  ✓ Gradient form works")

    # Test divergence form: ∇·(vm)
    print("\n[Test 2: Divergence form ∇·(vm)]")
    adv_op_div = AdvectionOperator(
        velocity_field=v,
        spacings=[dx, dy],
        field_shape=(Nx, Ny),
        scheme="upwind",
        form="divergence",
        bc=bc,
    )

    # ∇·(x * (1,0)) = ∂(x)/∂x = 1.0 (since ∂v/∂y = 0 and v_y = 0)
    adv_m_div = adv_op_div(m)
    mean_adv_div = np.mean(adv_m_div[5:-5, 5:-5])
    print(f"  Mean ∇·(vm) = {mean_adv_div:.3f} (expected ≈ 1.0)")
    assert 0.8 < mean_adv_div < 1.2
    print("  ✓ Divergence form works")

    # Test LinearOperator interface
    print("\n[Test 3: LinearOperator interface]")
    m_flat = m.ravel()
    adv_m_flat = adv_op @ m_flat

    print(f"  Input (flattened): {m_flat.shape}")
    print(f"  Output (flattened): {adv_m_flat.shape}")
    adv_m_reshaped = adv_m_flat.reshape(Nx, Ny)
    error = np.max(np.abs(adv_m_reshaped - adv_m))
    print(f"  Consistency check: max|matvec - callable| = {error:.2e}")
    assert error < 1e-12
    print("  ✓ LinearOperator interface works")

    # Test 1D case
    print("\n[Test 4: 1D advection]")
    Nx_1d = 100
    x_1d = np.linspace(0, 1, Nx_1d)
    dx_1d = x_1d[1] - x_1d[0]

    m_1d = x_1d**2  # m = x²
    v_1d = np.ones((1, Nx_1d))  # v = 1

    adv_op_1d = AdvectionOperator(
        velocity_field=v_1d,
        spacings=[dx_1d],
        field_shape=(Nx_1d,),
        scheme="centered",  # Test centered scheme
        form="gradient",
    )

    # v·∇(x²) = 1 · 2x = 2x
    adv_m_1d = adv_op_1d(m_1d)
    expected_1d = 2 * x_1d
    error_1d = np.max(np.abs(adv_m_1d[5:-5] - expected_1d[5:-5]))
    print(f"  Interior error: {error_1d:.2e} (expected = 2x)")
    assert error_1d < 0.05
    print("  ✓ 1D advection works")

    print("\n✅ All AdvectionOperator tests passed!")

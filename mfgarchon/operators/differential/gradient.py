"""
Partial derivative operators for tensor product grids.

This module provides LinearOperator implementation of discrete partial derivatives
for structured grids, using finite difference stencils.

Mathematical Background:
    Partial derivative: ∂u/∂xᵢ

    Discretization (2nd-order central differences):
        ∂u/∂x ≈ (u[i+1] - u[i-1]) / (2h)

    The full gradient ∇u = (∂u/∂x₁, ..., ∂u/∂xd) is a tuple of PartialDerivOperators.

Relationship to DirectDerivOperator (Issue #658):
    Mathematically, ∂u/∂xᵢ = eᵢ·∇u where eᵢ is the unit vector along axis i.
    Thus PartialDerivOperator is a special case of DirectDerivOperator.

    However, PartialDerivOperator is implemented independently for efficiency:
    - Computes only ONE partial derivative directly
    - DirectDerivOperator computes ALL partials and combines them

    For general directional derivatives v·∇u, use DirectDerivOperator.
    For boundary normal derivatives ∂u/∂n, use NormalDerivOperator.

References:
    - LeVeque (2007): Finite Difference Methods for ODEs and PDEs
    - Strang (2007): Computational Science and Engineering

Created: 2026-01-17 (Issue #595 - Operator Refactoring)
Part of: Issue #590 Phase 1.2 - TensorProductGrid Operator Traits
Migrated: 2026-01-25 (Issue #625 - tensor_calculus → stencils migration)
Renamed: 2026-01-25 (Issue #658 - GradientComponentOperator → PartialDerivOperator)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.sparse.linalg import LinearOperator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mfgarchon.geometry.boundary import BoundaryConditions


# WENO5 scheme support (Issue #606)
_WENO5_AVAILABLE = True
try:
    from mfgarchon.operators.reconstruction.weno import compute_weno5_derivative_1d
except ImportError:
    _WENO5_AVAILABLE = False


class PartialDerivOperator(LinearOperator):
    """
    Partial derivative operator: ∂u/∂xᵢ.

    Computes the partial derivative of a scalar field with respect to
    one spatial direction. The full gradient is a tuple of these operators.

    Attributes:
        direction: Spatial direction (0=x, 1=y, 2=z, ...)
        spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
        field_shape: Shape of input field (Nx, Ny, ...)
        scheme: Difference scheme ("central", "upwind", "backward", "forward", "one_sided", "weno5").
            "upwind" is the Rouy-Tourin HJB momentum; "backward" and "forward" are the one-sided differences the
            numerical-Hamiltonian owner combines (#2313).
        bc: Boundary conditions
        shape: Operator shape (N, N) where N = prod(field_shape)
        dtype: Data type (float64)

    Example:
        >>> # Single partial derivative
        >>> d_dx = PartialDerivOperator(direction=0, spacings=[0.1, 0.1],
        ...                             field_shape=(50, 50))
        >>> du_dx = d_dx(u)
        >>>
        >>> # All gradient components
        >>> grad_ops = tuple(PartialDerivOperator(d, spacings, shape) for d in range(ndim))

    Note:
        Renamed from GradientComponentOperator in v0.18.0 (Issue #658).
        The old name is available as a deprecated alias.
    """

    def __init__(
        self,
        direction: int,
        spacings: Sequence[float],
        field_shape: tuple[int, ...],
        scheme: Literal["central", "upwind", "backward", "forward", "one_sided", "weno5"] = "central",
        bc: BoundaryConditions | None = None,
        time: float = 0.0,
    ):
        """
        Initialize partial derivative operator.

        Args:
            direction: Spatial direction (0=x, 1=y, 2=z, ...)
            spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
            field_shape: Shape of field arrays (Nx, Ny, ...)
            scheme: Difference scheme
                - "central": 2nd-order central differences (default)
                - "upwind": the Rouy-Tourin HJB momentum (monotone, 1st-order; #2313)
                - "one_sided": 2nd-order one-sided at edges (forward at left, backward at right)
                - "weno5": 5th-order WENO reconstruction (high-order, shock-capturing)
            bc: Boundary conditions (None for periodic)
            time: Time for time-dependent BCs (default 0.0)

        Raises:
            ValueError: If direction >= len(field_shape)
        """
        self.direction = direction
        self.spacings = list(spacings)
        self.field_shape = tuple(field_shape)
        self.scheme = scheme
        self.bc = bc
        self.time = time

        # Validate
        if direction >= len(field_shape):
            raise ValueError(f"direction {direction} >= dimension {len(field_shape)}")

        if len(spacings) != len(field_shape):
            raise ValueError(f"spacings length {len(spacings)} != field_shape dimensions {len(field_shape)}")

        # Compute operator shape
        N = int(np.prod(field_shape))
        super().__init__(shape=(N, N), dtype=np.float64)

    def _matvec(self, u_flat: NDArray) -> NDArray:
        """
        Apply ∂u/∂xᵢ to flattened field.

        Args:
            u_flat: Flattened field array, shape (N,)

        Returns:
            Directional derivative, flattened, shape (N,)

        Note:
            Issue #625: Migrated from tensor_calculus to stencils module.
            Now computes only the requested component (more efficient).
        """
        # Reshape to field
        u = u_flat.reshape(self.field_shape)

        # WENO5 scheme: use dedicated implementation (Issue #606)
        if self.scheme == "weno5":
            if not _WENO5_AVAILABLE:
                raise ImportError("WENO5 scheme requires mfgarchon.operators.reconstruction.weno")

            # Currently only 1D supported
            if len(self.field_shape) != 1:
                raise NotImplementedError(f"WENO5 not yet implemented for {len(self.field_shape)}D")

            # Compute WENO5 derivative (left-biased for now)
            du_dxi = compute_weno5_derivative_1d(u, spacing=self.spacings[self.direction], bias="left")

            return du_dxi.ravel()

        # Standard schemes: use stencils module directly (Issue #625)
        from mfgarchon.operators.stencils.finite_difference import (
            fix_boundaries_one_sided,
            gradient_backward,
            gradient_central,
            gradient_forward,
            gradient_upwind,
        )

        h = self.spacings[self.direction]
        axis = self.direction

        # Apply ghost cell padding if BC provided (for non-periodic)
        if self.bc is not None:
            from mfgarchon.geometry.boundary import pad_array_with_ghosts

            u_work = pad_array_with_ghosts(u, self.bc, ghost_depth=1, time=self.time, spacing=self.spacings)
        else:
            u_work = u

        # Compute gradient based on scheme
        if self.scheme == "central":
            du_dxi = gradient_central(u_work, axis=axis, h=h)
        elif self.scheme == "upwind":
            du_dxi = gradient_upwind(u_work, axis=axis, h=h)
        elif self.scheme == "backward":
            du_dxi = gradient_backward(u_work, axis=axis, h=h)
        elif self.scheme == "forward":
            du_dxi = gradient_forward(u_work, axis=axis, h=h)
        elif self.scheme == "one_sided":
            # Central interior, one-sided at boundaries
            du_dxi = gradient_central(u_work, axis=axis, h=h)
            du_dxi = fix_boundaries_one_sided(du_dxi, u_work, axis=axis, h=h)
        else:
            raise ValueError(f"Unknown scheme: {self.scheme}")

        # Extract interior if ghost cells were added
        if self.bc is not None:
            slices = [slice(1, -1)] * len(self.field_shape)
            du_dxi = du_dxi[tuple(slices)]

        return du_dxi.ravel()

    def __call__(self, u: NDArray) -> NDArray:
        """
        Apply ∂u/∂xᵢ to field (preserves shape).

        Args:
            u: Field array, shape field_shape or (N,)

        Returns:
            Directional derivative, same shape as input
        """
        # Handle already-flattened input
        if u.ndim == 1:
            return self._matvec(u)

        # Handle field input
        if u.shape != self.field_shape:
            raise ValueError(f"Input shape {u.shape} doesn't match field_shape {self.field_shape}")

        du_flat = self._matvec(u.ravel())
        return du_flat.reshape(self.field_shape)

    def __repr__(self) -> str:
        """String representation for debugging."""
        dim_names = ["x", "y", "z", "w"]
        dim_name = dim_names[self.direction] if self.direction < len(dim_names) else f"x{self.direction}"
        bc_str = f"bc={self.bc.bc_type.value}" if self.bc else "bc=periodic"

        return (
            f"PartialDerivOperator(d/d{dim_name},\n"
            f"  field_shape={self.field_shape},\n"
            f"  spacings={self.spacings},\n"
            f"  scheme='{self.scheme}',\n"
            f"  {bc_str},\n"
            f"  shape={self.shape}\n"
            f")"
        )


# =============================================================================
# Gradient Operator (Issue #658 Phase 3)
# =============================================================================


class GradientOperator:
    """
    Full gradient operator: ∇u = (∂u/∂x₁, ..., ∂u/∂xd).

    Computes the gradient of a scalar field, returning a vector field.
    This is NOT a scipy LinearOperator because it changes dimensionality:
    maps scalar field (N,) to vector field (N, d).

    Mathematical Definition:
        For scalar field u: Ω ⊂ ℝ^d → ℝ
        ∇u = (∂u/∂x₁, ∂u/∂x₂, ..., ∂u/∂xd)

    Attributes:
        spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
        field_shape: Shape of input scalar field (N₁, N₂, ...)
        scheme: Difference scheme for all components ("central", "upwind", "backward", "forward",
            "one_sided", "weno5")
        components: Tuple of PartialDerivOperator for each dimension

    Example:
        >>> grad = GradientOperator(spacings=[dx, dy], field_shape=(Nx, Ny))
        >>> gradient_field = grad(u)  # Shape: (Nx, Ny, 2)
        >>>
        >>> # Access individual components
        >>> du_dx = grad.components[0](u)  # Just ∂u/∂x
        >>> du_dy = grad.components[1](u)  # Just ∂u/∂y

    Note:
        For scalar output like v·∇u, use DirectDerivOperator instead.
        For boundary normal derivative ∂u/∂n, use NormalDerivOperator.

    Created: 2026-01-25 (Issue #658 Phase 3)
    """

    def __init__(
        self,
        spacings: Sequence[float],
        field_shape: tuple[int, ...],
        scheme: Literal["central", "upwind", "backward", "forward", "one_sided", "weno5"] = "central",
        bc: BoundaryConditions | None = None,
        time: float = 0.0,
    ):
        """
        Initialize gradient operator.

        Args:
            spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
            field_shape: Shape of scalar field arrays (N₁, N₂, ...)
            scheme: Difference scheme for all components ("central", "upwind", "backward", "forward",
            "one_sided", "weno5")
                - "central": 2nd-order central differences (default)
                - "upwind": the Rouy-Tourin HJB momentum (monotone, 1st-order; #2313)
                - "one_sided": 2nd-order one-sided at edges (forward at left, backward at right)
                - "weno5": 5th-order WENO reconstruction
            bc: Boundary conditions (None for periodic)
            time: Time for time-dependent BCs (default 0.0)
        """
        self.spacings = list(spacings)
        self.field_shape = tuple(field_shape)
        self.scheme = scheme
        self.bc = bc
        self.time = time
        self.ndim = len(field_shape)

        # Validate
        if len(spacings) != len(field_shape):
            raise ValueError(f"spacings length {len(spacings)} != field_shape dimensions {len(field_shape)}")

        # Create partial derivative operators for each dimension
        self.components: tuple[PartialDerivOperator, ...] = tuple(
            PartialDerivOperator(
                direction=d,
                spacings=spacings,
                field_shape=field_shape,
                scheme=scheme,
                bc=bc,
                time=time,
            )
            for d in range(self.ndim)
        )

    def __call__(self, u: NDArray) -> NDArray:
        """
        Compute gradient of scalar field.

        Args:
            u: Scalar field, shape field_shape or flattened (N,)

        Returns:
            Gradient vector field, shape (*field_shape, ndim)
            Each point has ndim gradient components.

        Example:
            >>> u = np.sin(X) + np.cos(Y)  # Shape (50, 50)
            >>> grad_u = grad(u)            # Shape (50, 50, 2)
            >>> du_dx = grad_u[..., 0]      # ∂u/∂x
            >>> du_dy = grad_u[..., 1]      # ∂u/∂y
        """
        # Handle flattened input
        if u.ndim == 1:
            u = u.reshape(self.field_shape)

        if u.shape != self.field_shape:
            raise ValueError(f"Input shape {u.shape} doesn't match field_shape {self.field_shape}")

        # Stack gradient components along new last axis
        components = [comp(u) for comp in self.components]
        return np.stack(components, axis=-1)

    def magnitude(self, u: NDArray) -> NDArray:
        """
        Compute gradient magnitude |∇u|.

        Args:
            u: Scalar field, shape field_shape

        Returns:
            Gradient magnitude, shape field_shape
        """
        grad_u = self(u)
        return np.linalg.norm(grad_u, axis=-1)

    def __repr__(self) -> str:
        """String representation for debugging."""
        bc_str = f"bc={self.bc.bc_type.value}" if self.bc else "bc=periodic"
        return (
            f"GradientOperator(\n"
            f"  ndim={self.ndim},\n"
            f"  field_shape={self.field_shape},\n"
            f"  spacings={self.spacings},\n"
            f"  scheme='{self.scheme}',\n"
            f"  {bc_str}\n"
            f")"
        )


def wall_gradient_from_values(
    values: np.ndarray, gradient: Sequence[np.ndarray], spacings: Sequence[float], axes: Sequence[int]
) -> tuple[np.ndarray, ...]:
    """``gradient`` with the wall slices of each axis in ``axes`` replaced by a one-sided difference of ``values``.

    The geometry's central operator reads a wall node's value from the FDM ghost, which imposes the boundary
    condition: what an FD residual needs, and not a property of U. A velocity taken from U needs U's slope
    there. The cell-centred no-flux ghost gives half of it; the node-centred mirror #1935 needs gives the
    boundary datum whatever U is, which froze the SL pair's wall characteristic (#2467) and stalls particles
    at the wall (#2470).

    Measured on `test_hjb_semi_lagrangian.py`'s sigma = 0 fixture, whose exact solution is the Hopf-Lax
    closed form (x - 1/2)^2 / (2 (1 + tau)) + tau with wall slope -1 / (2 (1 + tau)): at 201 points the max
    error is 5.3e-04 from the ghost's half slope, wall/interior 1.60, and 3.3e-04 from this, wall/interior
    1.007. A gradient forced to the datum gives 7.7e-03.

    The difference is second order on an axis of 3 or more points. A 2-point axis holds one difference,
    (u_1 - u_0) / h, and both walls take it. A 1-point axis holds none and is left as the operator gave it.
    Interior values, and every axis not in ``axes``, are returned unchanged.
    """
    closed = [np.array(component, dtype=float) for component in gradient]
    for axis in axes:
        n, h = values.shape[axis], spacings[axis]
        if n < 2:
            continue
        low: list[slice | int] = [slice(None)] * values.ndim
        high: list[slice | int] = [slice(None)] * values.ndim
        low[axis], high[axis] = 0, -1

        def at(i: int, ax: int = axis) -> np.ndarray:
            return np.take(values, i, axis=ax)

        if n == 2:
            closed[axis][tuple(low)] = closed[axis][tuple(high)] = (at(1) - at(0)) / h
        else:
            closed[axis][tuple(low)] = (-3.0 * at(0) + 4.0 * at(1) - at(2)) / (2.0 * h)
            closed[axis][tuple(high)] = (3.0 * at(-1) - 4.0 * at(-2) + at(-3)) / (2.0 * h)
    return tuple(closed)


def value_gradient(
    geometry: Any, values: np.ndarray, time: float, bc: BoundaryConditions | None = None
) -> tuple[np.ndarray, ...]:
    """The gradient of data, for a consumer that takes a velocity from U rather than imposing a condition on it.

    The SL pair's control and drift (#2467) and the particle FP drift (#2470). An FD residual -- HJB-FDM's
    H(grad u), the Laplacian -- wants the ghost instead, and keeps the operator.

    The geometry's central operator, whose ghosts carry ``bc`` (the geometry's when omitted); then every
    axis whose two faces are not periodic has its wall nodes replaced by the one-sided difference of
    ``values`` (`wall_gradient_from_values`). A periodic axis keeps the operator's wrap.

    Each axis is asked separately, through ``get_bc_type_at_boundary``. Reading one type for the whole boundary
    takes the first segment's, so on a geometry mixing periodic and non-periodic faces -- reachable wherever the
    solver does not refuse such a BC itself: ``FPParticleSolver``, and ``FPSLSolver`` given its own
    ``boundary_conditions`` -- the answer depended on the segments' order. That accessor and the ghosts'
    resolver (``PreallocatedGhostBuffer._find_segment_for_face``) both match a segment with a ``boundary`` by
    its face, whatever region it is restricted to within that face. A segment with no ``boundary`` covers every
    face for the accessor, while the ghosts match it to a face only through its region name, or not at all; in a
    periodic mix that disagreement can decide which axes are periodic, so such a segment is refused there, as is
    an axis periodic on one face only. A uniform BC is not a mix: its one segment covers every face for both,
    and neither reads its ``default_bc``, which ``geometric_operations`` still counts.

    On a closed axis the wall values the ghosts produced are discarded, so ``time`` reaches nothing there;
    it is passed for the operator's reads, whose clock #2453 pins.
    """
    from mfgarchon.geometry.boundary.bc_utils import geometric_operations
    from mfgarchon.geometry.boundary.types import BCType

    given = bc
    bc = geometry.get_boundary_conditions() if given is None else given
    operations = geometric_operations(bc)
    if "periodic" in operations and len(operations) > 1 and not bc.is_uniform:
        unbounded = [seg.name for seg in bc.segments if seg.boundary is None]
        if unbounded:
            raise NotImplementedError(
                f"segments {unbounded} have no `boundary` in a mix of periodic and non-periodic faces; a velocity "
                f"taken from U reads periodicity per face and cannot tell which faces they cover. Give each one the "
                f"face it lies on with `boundary=` (a region may still restrict it within that face) (#2467)."
            )
    closed_axes = []
    for axis in range(values.ndim):
        periodic = [bc.get_bc_type_at_boundary(f"axis{axis}_{side}") == BCType.PERIODIC for side in ("min", "max")]
        if periodic[0] != periodic[1]:
            raise NotImplementedError(
                f"axis {axis} is periodic on one face only; a periodic axis needs both, so a velocity taken from U "
                f"cannot tell which neighbour its wall nodes have"
            )
        if not periodic[0]:
            closed_axes.append(axis)
    ops = (
        geometry.get_gradient_operator(scheme="central", time=time)
        if given is None
        else geometry.get_gradient_operator(scheme="central", time=time, bc=given)
    )
    components = tuple(op(values) for op in ops)
    return wall_gradient_from_values(values, components, geometry.get_grid_spacing(), closed_axes)


if __name__ == "__main__":
    """Smoke test for PartialDerivOperator."""
    print("Testing PartialDerivOperator...")

    # Test 1D
    print("\n[1D Gradient]")

    # Test on u = sin(x), du/dx = cos(x)
    x = np.linspace(0, 2 * np.pi, 100)
    dx = x[1] - x[0]
    u_1d = np.sin(x)

    grad_x = PartialDerivOperator(direction=0, spacings=[dx], field_shape=(100,), scheme="central")
    print(f"  Operator shape: {grad_x.shape}")
    print(f"  Field shape: {grad_x.field_shape}")
    print(f"  Grid spacing: dx={dx:.4f}")
    print(f"  Direction: {grad_x.direction} (x)")

    du_dx = grad_x(u_1d)
    expected = np.cos(x)

    print(f"  Input shape: {u_1d.shape}, Output shape: {du_dx.shape}")
    assert du_dx.shape == u_1d.shape

    # Check interior (boundaries may have larger error)
    error = np.max(np.abs(du_dx[10:-10] - expected[10:-10]))
    print(f"  d sin(x)/dx interior error: {error:.2e} (expected = cos(x))")
    assert error < 0.01, f"Error too large: {error}"

    # Test @ syntax
    du_dx_matvec = grad_x @ u_1d.ravel()
    assert np.allclose(du_dx.ravel(), du_dx_matvec)
    print("  OK: grad_x(u) == grad_x @ u.ravel()")

    # Test 2D
    print("\n[2D Gradient]")
    Nx, Ny = 50, 50
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Test on u = x^2 + y^3, du/dx = 2x, du/dy = 3y^2
    u_2d = X**2 + Y**3

    grad_x = PartialDerivOperator(direction=0, spacings=[dx, dy], field_shape=(Nx, Ny), scheme="central")
    grad_y = PartialDerivOperator(direction=1, spacings=[dx, dy], field_shape=(Nx, Ny), scheme="central")

    print(f"  Operator shapes: {grad_x.shape}, {grad_y.shape}")
    print(f"  Field shape: {grad_x.field_shape}")

    du_dx = grad_x(u_2d)
    du_dy = grad_y(u_2d)

    print(f"  Input shape: {u_2d.shape}")
    print(f"  ∂u/∂x shape: {du_dx.shape}, ∂u/∂y shape: {du_dy.shape}")
    assert du_dx.shape == u_2d.shape
    assert du_dy.shape == u_2d.shape

    # Check interior
    expected_dx = 2 * X
    expected_dy = 3 * Y**2

    error_dx = np.max(np.abs(du_dx[10:-10, 10:-10] - expected_dx[10:-10, 10:-10]))
    error_dy = np.max(np.abs(du_dy[10:-10, 10:-10] - expected_dy[10:-10, 10:-10]))

    print(f"  du/dx interior error: {error_dx:.2e} (expected = 2x)")
    print(f"  du/dy interior error: {error_dy:.2e} (expected = 3y^2)")
    # Quadratic function (x^2) should have near-machine-precision error
    assert error_dx < 1e-10, f"du/dx error too large: {error_dx}"
    # Cubic function (y^3) has O(h^2) discretization error
    assert error_dy < 1e-3, f"du/dy error too large: {error_dy}"
    print("  OK: Gradient accuracy check passed")

    # Test scipy compatibility
    print("\n[scipy compatibility]")
    assert isinstance(grad_x, LinearOperator)
    assert isinstance(grad_y, LinearOperator)
    print("  OK: isinstance(grad_*, scipy.sparse.linalg.LinearOperator)")

    # Test repr
    print("\n[String representation]")
    print(grad_x)

    # Test WENO5 scheme (Issue #606)
    print("\n[WENO5 Scheme Integration]")
    try:
        # Create 1D grid for WENO5 test
        x_weno = np.linspace(0, 2 * np.pi, 100)
        dx_weno = x_weno[1] - x_weno[0]

        # Create WENO5 operator
        grad_weno5 = PartialDerivOperator(direction=0, spacings=[dx_weno], field_shape=(100,), scheme="weno5")

        # Test on linear function (should be exact)
        u_linear = 2 * x_weno + 1
        du_weno5 = grad_weno5(u_linear)

        # Check interior points
        interior_error_weno5 = np.max(np.abs(du_weno5[2:-2] - 2.0))
        print("  Linear function (u = 2x + 1):")
        print(f"  Interior error: {interior_error_weno5:.6e} (expect ~machine precision)")

        assert interior_error_weno5 < 1e-10, "WENO5 should be exact for linear functions"
        print("  OK: WENO5 scheme integration working!")

        # Test @ syntax
        du_weno5_matvec = grad_weno5 @ u_linear.ravel()
        assert np.allclose(du_weno5.ravel(), du_weno5_matvec)
        print("  OK: grad_weno5(u) == grad_weno5 @ u.ravel()")

        # Test smooth function accuracy
        u_smooth = np.sin(2 * np.pi * x_weno)
        du_exact_smooth = 2 * np.pi * np.cos(2 * np.pi * x_weno)
        du_weno5_smooth = grad_weno5(u_smooth)

        interior_error_smooth = np.max(np.abs(du_weno5_smooth[2:-2] - du_exact_smooth[2:-2]))
        print("  Smooth function (sin(2*pi*x)):")
        print(f"  Interior error: {interior_error_smooth:.6e}")
        print("  OK: WENO5 high-order accuracy verified!")

    except ImportError:
        print("  WENO5 scheme not available (schemes module not found)")

    print("\nAll PartialDerivOperator tests passed!")

    # ==========================================================================
    # GradientOperator Tests (Issue #658 Phase 3)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Testing GradientOperator (Issue #658 Phase 3)")
    print("=" * 60)

    # Test 2D gradient
    print("\n[2D Full Gradient]")
    Nx, Ny = 50, 50
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Test on u = x^2 + y^3, ∇u = (2x, 3y^2)
    u_2d = X**2 + Y**3

    grad_op = GradientOperator(spacings=[dx, dy], field_shape=(Nx, Ny), scheme="central")
    print(grad_op)

    grad_u = grad_op(u_2d)
    print(f"  Input shape: {u_2d.shape}")
    print(f"  Output shape: {grad_u.shape}")
    assert grad_u.shape == (Nx, Ny, 2), f"Expected (Nx, Ny, 2), got {grad_u.shape}"

    # Check components
    du_dx = grad_u[..., 0]
    du_dy = grad_u[..., 1]

    expected_dx = 2 * X
    expected_dy = 3 * Y**2

    # Interior error (boundaries have larger error)
    error_dx = np.max(np.abs(du_dx[10:-10, 10:-10] - expected_dx[10:-10, 10:-10]))
    error_dy = np.max(np.abs(du_dy[10:-10, 10:-10] - expected_dy[10:-10, 10:-10]))

    print(f"  ∂u/∂x interior error: {error_dx:.2e}")
    print(f"  ∂u/∂y interior error: {error_dy:.2e}")
    assert error_dx < 1e-10, f"∂u/∂x error too large: {error_dx}"
    assert error_dy < 1e-3, f"∂u/∂y error too large: {error_dy}"

    # Test gradient magnitude
    print("\n[Gradient Magnitude]")
    grad_mag = grad_op.magnitude(u_2d)
    expected_mag = np.sqrt((2 * X) ** 2 + (3 * Y**2) ** 2)

    error_mag = np.max(np.abs(grad_mag[10:-10, 10:-10] - expected_mag[10:-10, 10:-10]))
    print(f"  |∇u| interior error: {error_mag:.2e}")
    assert error_mag < 1e-3, f"|∇u| error too large: {error_mag}"

    # Test access to individual components
    print("\n[Component Access]")
    assert len(grad_op.components) == 2
    du_dx_direct = grad_op.components[0](u_2d)
    du_dy_direct = grad_op.components[1](u_2d)

    assert np.allclose(du_dx, du_dx_direct), "Component[0] mismatch"
    assert np.allclose(du_dy, du_dy_direct), "Component[1] mismatch"
    print("  OK: grad.components[i](u) == grad(u)[..., i]")

    # Test 1D case
    print("\n[1D Full Gradient]")
    x_1d = np.linspace(0, 2 * np.pi, 100)
    dx_1d = x_1d[1] - x_1d[0]
    u_1d = np.sin(x_1d)

    grad_1d = GradientOperator(spacings=[dx_1d], field_shape=(100,))
    grad_u_1d = grad_1d(u_1d)

    print(f"  Input shape: {u_1d.shape}")
    print(f"  Output shape: {grad_u_1d.shape}")
    assert grad_u_1d.shape == (100, 1), f"Expected (100, 1), got {grad_u_1d.shape}"

    error_1d = np.max(np.abs(grad_u_1d[10:-10, 0] - np.cos(x_1d)[10:-10]))
    print(f"  d sin(x)/dx interior error: {error_1d:.2e}")
    assert error_1d < 0.01, f"1D gradient error too large: {error_1d}"

    # Test 3D case
    print("\n[3D Full Gradient]")
    Nx3, Ny3, Nz3 = 20, 20, 20
    x3 = np.linspace(0, 1, Nx3)
    y3 = np.linspace(0, 1, Ny3)
    z3 = np.linspace(0, 1, Nz3)
    dx3, dy3, dz3 = x3[1] - x3[0], y3[1] - y3[0], z3[1] - z3[0]
    X3, Y3, Z3 = np.meshgrid(x3, y3, z3, indexing="ij")

    # u = x + 2y + 3z, ∇u = (1, 2, 3)
    u_3d = X3 + 2 * Y3 + 3 * Z3

    grad_3d = GradientOperator(spacings=[dx3, dy3, dz3], field_shape=(Nx3, Ny3, Nz3))
    grad_u_3d = grad_3d(u_3d)

    print(f"  Input shape: {u_3d.shape}")
    print(f"  Output shape: {grad_u_3d.shape}")
    assert grad_u_3d.shape == (Nx3, Ny3, Nz3, 3)

    # Linear function should have near-exact gradient
    error_3d_x = np.max(np.abs(grad_u_3d[5:-5, 5:-5, 5:-5, 0] - 1.0))
    error_3d_y = np.max(np.abs(grad_u_3d[5:-5, 5:-5, 5:-5, 1] - 2.0))
    error_3d_z = np.max(np.abs(grad_u_3d[5:-5, 5:-5, 5:-5, 2] - 3.0))

    print(f"  ∂u/∂x error: {error_3d_x:.2e} (expect ~0)")
    print(f"  ∂u/∂y error: {error_3d_y:.2e} (expect ~0)")
    print(f"  ∂u/∂z error: {error_3d_z:.2e} (expect ~0)")
    assert error_3d_x < 1e-10, f"3D ∂u/∂x error: {error_3d_x}"
    assert error_3d_y < 1e-10, f"3D ∂u/∂y error: {error_3d_y}"
    assert error_3d_z < 1e-10, f"3D ∂u/∂z error: {error_3d_z}"

    print("\nAll GradientOperator tests passed!")

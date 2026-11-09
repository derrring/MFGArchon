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
        time: float | None = None,
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
            time: Time for time-dependent BCs (None, the default, reads a constant or g(x) value and refuses one that declares t (#2512))

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
        time: float | None = None,
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
            time: Time for time-dependent BCs (None, the default, reads a constant or g(x) value and refuses one that declares t (#2512))
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

    Each face is asked separately, of the face owner (:meth:`BoundaryConditions.segment_for_face`, #2512 row B1),
    with the geometry: reading one type for the whole boundary took the first segment's, so on a geometry mixing
    periodic and non-periodic faces -- reachable wherever the solver does not refuse such a BC itself:
    ``FPParticleSolver``, and ``FPSLSolver`` given its own ``boundary_conditions`` -- the answer depended on the
    segments' order. The owner places every segment the way the ghosts do, so no segment is refused here for being
    unplaced; a segment covering part of a face is accepted only if it agrees with the rest of the face on
    periodicity, the one thing this consumer reads. An axis periodic on one face only is refused.

    On a closed axis the wall values the ghosts produced are discarded, so ``time`` reaches nothing there;
    it is passed for the operator's reads, whose clock #2453 pins.
    """
    from mfgarchon.geometry.boundary.types import BCType, BoundaryFace

    bc = geometry.get_boundary_conditions() if bc is None else bc

    def is_periodic(segment: Any) -> bool:
        return segment.bc_type == BCType.PERIODIC

    def same_periodicity(part: Any, governing: Any) -> bool:
        return is_periodic(part) == is_periodic(governing)

    closed_axes = []
    for axis in range(values.ndim):
        periodic = [
            is_periodic(bc.segment_for_face(BoundaryFace(axis, side), geometry, equivalent=same_periodicity))
            for side in ("min", "max")
        ]
        if periodic[0] != periodic[1]:
            raise NotImplementedError(
                f"axis {axis} is periodic on one face only; a periodic axis needs both, so a velocity taken from U "
                f"cannot tell which neighbour its wall nodes have"
            )
        if not periodic[0]:
            closed_axes.append(axis)
    # The operator gets the BC this consumer reads: the wrap on a periodic axis, and on a closed one a no-flux
    # ghost whose wall values are replaced below anyway. Handing it the full BC asked its ghosts to impose every
    # condition on every face -- a Dirichlet door on part of a closed face among them, which no face-level ghost
    # can represent and which this consumer then discards.
    from mfgarchon.geometry.boundary import BCSegment, BoundaryConditions

    read = BoundaryConditions(
        dimension=values.ndim,
        segments=[
            BCSegment(
                name=f"{'periodic' if axis not in closed_axes else 'closed'} {face.to_string()}",
                bc_type=BCType.NO_FLUX if axis in closed_axes else BCType.PERIODIC,
                boundary=face.to_string(),
            )
            for axis in range(values.ndim)
            for face in (BoundaryFace(axis, "min"), BoundaryFace(axis, "max"))
        ],
        default_bc=BCType.NO_FLUX,
        periodic_convention=bc.periodic_convention,
    )
    ops = geometry.get_gradient_operator(scheme="central", time=time, bc=read)
    components = tuple(op(values) for op in ops)
    return wall_gradient_from_values(values, components, geometry.get_grid_spacing(), closed_axes)

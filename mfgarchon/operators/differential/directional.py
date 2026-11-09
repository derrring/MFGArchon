"""
Directional derivative operators: v·∇u and ∂u/∂n.

This module provides:
- DirectDerivOperator: General directional derivative v·∇u
- NormalDerivOperator: Normal derivative ∂u/∂n (specialization)

Mathematical Background:
    Directional derivative: (v·∇)u = Σᵢ vᵢ ∂u/∂xᵢ

    Special cases:
    - v = eᵢ (unit vector): (eᵢ·∇)u = ∂u/∂xᵢ  (partial derivative)
    - v = n (outward normal): (n·∇)u = ∂u/∂n  (normal derivative)

    The direction v can be:
    - Constant: same direction everywhere, shape (d,)
    - Spatially varying: different direction at each point, shape (*field_shape, d)

    For NormalDerivOperator, the outward normal can be computed from:
    - Structured grids: ±eᵢ at axis-aligned boundaries
    - SDF (signed distance function): n = ∇φ / |∇φ|
    - Explicit: User-provided normal field

    Universal Outward Normal Convention (Issue #661):
    - Outward normal points FROM domain TO exterior
    - ∂u/∂n > 0 means u increases in outward direction
    - Consistent across all geometry types

References:
    - LeVeque (2007): Finite Difference Methods for ODEs and PDEs
    - Issue #658: Operator Library Cleanup
    - Issue #661: Universal Outward Normal Convention

Created: 2026-01-25 (Issue #658 Phase 2)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.sparse.linalg import LinearOperator

from mfgarchon.operators.differential.gradient import PartialDerivOperator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mfgarchon.geometry.boundary import BoundaryConditions


class DirectDerivOperator(LinearOperator):
    """
    Directional derivative operator: v·∇u.

    Computes the derivative of a scalar field in the direction of vector v.
    The direction can be constant (same everywhere) or spatially varying.

    Attributes:
        direction: Direction vector(s), shape (d,) or (*field_shape, d)
        spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
        field_shape: Shape of input field (Nx, Ny, ...)
        scheme: Difference scheme ("central", "upwind", "one_sided")
        bc: Boundary conditions
        shape: Operator shape (N, N) where N = prod(field_shape)
        dtype: Data type (float64)

    Example:
        >>> # Constant direction (e.g., advection velocity)
        >>> v = np.array([1.0, 0.5])  # direction in 2D
        >>> D_v = DirectDerivOperator(v, spacings=[dx, dy], field_shape=(Nx, Ny))
        >>> du_dv = D_v(u)  # v·∇u
        >>>
        >>> # Spatially varying direction (e.g., streamlines)
        >>> v_field = np.stack([vx, vy], axis=-1)  # shape (Nx, Ny, 2)
        >>> D_v = DirectDerivOperator(v_field, spacings=[dx, dy], field_shape=(Nx, Ny))

    Note:
        This is a base class. For axis-aligned derivatives, use PartialDerivOperator
        which is more efficient. For boundary normal derivatives, use NormalDerivOperator.
    """

    def __init__(
        self,
        direction: NDArray,
        spacings: Sequence[float],
        field_shape: tuple[int, ...],
        scheme: Literal["central", "upwind", "one_sided"] = "central",
        bc: BoundaryConditions | None = None,
        time: float | None = None,
    ):
        """
        Initialize directional derivative operator.

        Args:
            direction: Direction vector(s)
                - Constant: shape (d,) for same direction everywhere
                - Varying: shape (*field_shape, d) for position-dependent direction
            spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
            field_shape: Shape of field arrays (Nx, Ny, ...)
            scheme: Difference scheme for gradient computation
                - "central": 2nd-order central differences (default)
                - "upwind": Godunov upwind (monotone, 1st-order)
                - "one_sided": 2nd-order one-sided at edges (forward at left, backward at right)
            bc: Boundary conditions (None for periodic)
            time: Time for time-dependent BCs (None, the default, reads a constant or g(x) value and refuses one that declares t (#2512))

        Raises:
            ValueError: If direction shape is incompatible with field_shape
        """
        self.spacings = list(spacings)
        self.field_shape = tuple(field_shape)
        self.scheme = scheme
        self.bc = bc
        self.time = time
        self.ndim = len(field_shape)

        # Validate and store direction
        direction = np.asarray(direction, dtype=np.float64)
        self._validate_direction(direction)
        self.direction = direction
        self._is_constant = direction.ndim == 1

        # Create partial derivative operators for each dimension
        self._partial_derivs = [
            PartialDerivOperator(
                direction=i,
                spacings=spacings,
                field_shape=field_shape,
                scheme=scheme,
                bc=bc,
                time=time,
            )
            for i in range(self.ndim)
        ]

        # Compute operator shape
        N = int(np.prod(field_shape))
        super().__init__(shape=(N, N), dtype=np.float64)

    def _validate_direction(self, direction: NDArray) -> None:
        """Validate direction array shape."""
        if direction.ndim == 1:
            # Constant direction: shape (d,)
            if len(direction) != self.ndim:
                raise ValueError(
                    f"Constant direction has {len(direction)} components, expected {self.ndim} for {self.ndim}D field"
                )
        elif direction.ndim == self.ndim + 1:
            # Spatially varying: shape (*field_shape, d)
            expected_shape = (*self.field_shape, self.ndim)
            if direction.shape != expected_shape:
                raise ValueError(f"Varying direction shape {direction.shape} doesn't match expected {expected_shape}")
        else:
            raise ValueError(f"Direction must have shape (d,) or (*field_shape, d), got shape {direction.shape}")

    def _matvec(self, u_flat: NDArray) -> NDArray:
        """
        Apply v·∇u to flattened field.

        Args:
            u_flat: Flattened field array, shape (N,)

        Returns:
            Directional derivative, flattened, shape (N,)
        """
        u = u_flat.reshape(self.field_shape)
        result = np.zeros_like(u)

        for i, partial_op in enumerate(self._partial_derivs):
            # Compute ∂u/∂xᵢ
            du_dxi = partial_op(u)

            # Multiply by direction component
            if self._is_constant:
                result += self.direction[i] * du_dxi
            else:
                # Spatially varying: direction[..., i] has shape field_shape
                result += self.direction[..., i] * du_dxi

        return result.ravel()

    def __call__(self, u: NDArray) -> NDArray:
        """
        Apply v·∇u to field (preserves shape).

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

    @property
    def is_constant_direction(self) -> bool:
        """Whether direction is constant (same everywhere)."""
        return self._is_constant

    def __repr__(self) -> str:
        """String representation for debugging."""
        if self._is_constant:
            dir_str = f"direction={self.direction.tolist()}"
        else:
            dir_str = f"direction=varying{self.direction.shape}"
        bc_str = f"bc={self.bc.bc_type.value}" if self.bc else "bc=periodic"

        return (
            f"DirectDerivOperator(\n"
            f"  {dir_str},\n"
            f"  field_shape={self.field_shape},\n"
            f"  spacings={self.spacings},\n"
            f"  scheme='{self.scheme}',\n"
            f"  {bc_str},\n"
            f"  shape={self.shape}\n"
            f")"
        )


class NormalDerivOperator(DirectDerivOperator):
    """
    Normal derivative operator: ∂u/∂n = n·∇u.

    Computes the derivative of a scalar field in the outward normal direction.
    This is a specialization of DirectDerivOperator where the direction is
    the outward-pointing unit normal vector.

    The outward normal can be provided directly or computed from:
    - Structured grid boundaries (axis-aligned: ±eᵢ)
    - Signed distance function (SDF): n = ∇φ / |∇φ|

    Universal Outward Normal Convention (Issue #661):
        - n points FROM domain interior TO exterior
        - ∂u/∂n > 0 means u increases outward
        - Consistent across structured grids, SDF, and meshfree

    Attributes:
        normal: Outward normal vector(s), shape (d,) or (*field_shape, d)
        normal_source: How normal was computed ("explicit", "sdf", "axis")

    Example:
        >>> # From explicit normal field
        >>> normals = compute_outward_normals(...)  # shape (Nx, Ny, 2)
        >>> D_n = NormalDerivOperator(normals, spacings=[dx, dy], field_shape=(Nx, Ny))
        >>> du_dn = D_n(u)
        >>>
        >>> # From SDF (signed distance function)
        >>> D_n = NormalDerivOperator.from_sdf(sdf, spacings=[dx, dy])
        >>> du_dn = D_n(u)
        >>>
        >>> # For axis-aligned boundary (e.g., left boundary in 1D)
        >>> D_n = NormalDerivOperator.from_axis(axis=0, sign=-1, spacings=[dx], field_shape=(N,))

    Note:
        For boundary conditions, this operator computes ∂u/∂n at ALL grid points.
        For BC application at specific boundaries, use the geometry.boundary module.
    """

    def __init__(
        self,
        normal: NDArray,
        spacings: Sequence[float],
        field_shape: tuple[int, ...],
        scheme: Literal["central", "upwind", "one_sided"] = "central",
        bc: BoundaryConditions | None = None,
        time: float | None = None,
        normal_source: str = "explicit",
    ):
        """
        Initialize normal derivative operator.

        Args:
            normal: Outward normal vector(s)
                - Constant: shape (d,) for same normal everywhere
                - Varying: shape (*field_shape, d) for position-dependent normal
                Must be unit vectors (|n| = 1)
            spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
            field_shape: Shape of field arrays (Nx, Ny, ...)
            scheme: Difference scheme for gradient computation
            bc: Boundary conditions (None for periodic)
            time: Time for time-dependent BCs (None, the default, reads a constant or g(x) value and refuses one that declares t (#2512))
            normal_source: How normal was computed ("explicit", "sdf", "axis")

        Raises:
            ValueError: If normal is not unit length (within tolerance)
        """
        # Validate unit normal
        normal = np.asarray(normal, dtype=np.float64)
        self._validate_unit_normal(normal)

        # Store metadata
        self.normal = normal
        self.normal_source = normal_source

        # Initialize base class with normal as direction
        super().__init__(
            direction=normal,
            spacings=spacings,
            field_shape=field_shape,
            scheme=scheme,
            bc=bc,
            time=time,
        )

    def _validate_unit_normal(self, normal: NDArray, tol: float = 1e-6) -> None:
        """Validate that normal is unit length."""
        if normal.ndim == 1:
            # Constant normal
            norm = np.linalg.norm(normal)
            if abs(norm - 1.0) > tol:
                raise ValueError(f"Normal must be unit vector, got |n| = {norm:.6f}")
        else:
            # Spatially varying: check all normals
            # norm along last axis
            norms = np.linalg.norm(normal, axis=-1)
            max_deviation = np.max(np.abs(norms - 1.0))
            if max_deviation > tol:
                raise ValueError(f"All normals must be unit vectors, max |n| deviation = {max_deviation:.6f}")

    @classmethod
    def from_sdf(
        cls,
        sdf: NDArray,
        spacings: Sequence[float],
        scheme: Literal["central", "upwind", "one_sided"] = "central",
        bc: BoundaryConditions | None = None,
        time: float | None = None,
    ) -> NormalDerivOperator:
        """
        Create NormalDerivOperator from signed distance function.

        The outward normal is computed as n = ∇φ / |∇φ| where φ is the SDF.
        By convention, φ > 0 outside the domain, so ∇φ points outward.

        Args:
            sdf: Signed distance function, shape field_shape
                Convention: φ < 0 inside, φ > 0 outside, φ = 0 on boundary
            spacings: Grid spacing per dimension
            scheme: Difference scheme for gradient computation
            bc: Boundary conditions
            time: Time for time-dependent BCs

        Returns:
            NormalDerivOperator with outward normals from SDF gradient

        Example:
            >>> # Create SDF for a circle
            >>> X, Y = np.meshgrid(x, y, indexing="ij")
            >>> sdf = np.sqrt((X - cx)**2 + (Y - cy)**2) - radius  # + outside
            >>> D_n = NormalDerivOperator.from_sdf(sdf, spacings=[dx, dy])
        """
        field_shape = sdf.shape
        ndim = len(field_shape)

        # Compute gradient of SDF
        grad_components = []
        for i in range(ndim):
            partial_op = PartialDerivOperator(
                direction=i,
                spacings=spacings,
                field_shape=field_shape,
                scheme=scheme,
                bc=bc,
                time=time,
            )
            grad_components.append(partial_op(sdf))

        # Stack to get gradient field: shape (*field_shape, d)
        grad_sdf = np.stack(grad_components, axis=-1)

        # Compute magnitude
        grad_mag = np.linalg.norm(grad_sdf, axis=-1, keepdims=True)

        # Normalize (avoid division by zero)
        eps = 1e-10
        normal = grad_sdf / np.maximum(grad_mag, eps)

        return cls(
            normal=normal,
            spacings=spacings,
            field_shape=field_shape,
            scheme=scheme,
            bc=bc,
            time=time,
            normal_source="sdf",
        )

    @classmethod
    def from_axis(
        cls,
        axis: int,
        sign: Literal[-1, 1],
        spacings: Sequence[float],
        field_shape: tuple[int, ...],
        scheme: Literal["central", "upwind", "one_sided"] = "central",
        bc: BoundaryConditions | None = None,
        time: float | None = None,
    ) -> NormalDerivOperator:
        """
        Create NormalDerivOperator for axis-aligned boundary.

        For structured grids with axis-aligned boundaries, the outward normal
        is simply ±eᵢ where eᵢ is the unit vector along axis i.

        Args:
            axis: Spatial axis (0=x, 1=y, 2=z)
            sign: Direction of outward normal along axis
                -1: normal points in -xᵢ direction (e.g., left boundary)
                +1: normal points in +xᵢ direction (e.g., right boundary)
            spacings: Grid spacing per dimension
            field_shape: Shape of field arrays
            scheme: Difference scheme
            bc: Boundary conditions
            time: Time for time-dependent BCs

        Returns:
            NormalDerivOperator with constant axis-aligned normal

        Example:
            >>> # Left boundary in 1D: outward normal points left (-x)
            >>> D_n_left = NormalDerivOperator.from_axis(axis=0, sign=-1, ...)
            >>>
            >>> # Right boundary in 1D: outward normal points right (+x)
            >>> D_n_right = NormalDerivOperator.from_axis(axis=0, sign=+1, ...)
            >>>
            >>> # Top boundary in 2D (y-axis): outward normal points up (+y)
            >>> D_n_top = NormalDerivOperator.from_axis(axis=1, sign=+1, ...)
        """
        ndim = len(spacings)
        if axis >= ndim:
            raise ValueError(f"axis {axis} >= ndim {ndim}")
        if sign not in (-1, 1):
            raise ValueError(f"sign must be -1 or +1, got {sign}")

        # Create unit vector along axis
        normal = np.zeros(ndim, dtype=np.float64)
        normal[axis] = float(sign)

        return cls(
            normal=normal,
            spacings=spacings,
            field_shape=field_shape,
            scheme=scheme,
            bc=bc,
            time=time,
            normal_source="axis",
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        if self._is_constant:
            normal_str = f"normal={self.normal.tolist()}"
        else:
            normal_str = f"normal=varying{self.normal.shape}"
        bc_str = f"bc={self.bc.bc_type.value}" if self.bc else "bc=periodic"

        return (
            f"NormalDerivOperator(\n"
            f"  {normal_str},\n"
            f"  source='{self.normal_source}',\n"
            f"  field_shape={self.field_shape},\n"
            f"  spacings={self.spacings},\n"
            f"  scheme='{self.scheme}',\n"
            f"  {bc_str},\n"
            f"  shape={self.shape}\n"
            f")"
        )

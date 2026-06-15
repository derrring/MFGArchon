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
        time: float = 0.0,
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
            time: Time for time-dependent BCs (default 0.0)

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
        time: float = 0.0,
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
            time: Time for time-dependent BCs (default 0.0)
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
        time: float = 0.0,
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
        time: float = 0.0,
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


if __name__ == "__main__":
    """Smoke test for DirectDerivOperator."""
    print("Testing DirectDerivOperator...")

    # Test 1: Constant direction in 1D
    print("\n[1D Constant Direction]")

    x = np.linspace(0, 2 * np.pi, 100)
    dx = x[1] - x[0]
    u_1d = np.sin(x)

    # Direction = 1.0 (should give same as PartialDerivOperator)
    D_v = DirectDerivOperator(direction=np.array([1.0]), spacings=[dx], field_shape=(100,))
    print(f"  Operator: {D_v.shape}, constant direction=[1.0]")

    du_dv = D_v(u_1d)
    expected = np.cos(x)

    error = np.max(np.abs(du_dv[10:-10] - expected[10:-10]))
    print(f"  d sin(x)/dx interior error: {error:.2e}")
    assert error < 0.01, f"Error too large: {error}"
    print("  OK: Constant direction in 1D")

    # Test 2: Constant direction in 2D
    print("\n[2D Constant Direction]")

    Nx, Ny = 50, 50
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    # u = x^2 + y^2, v = (1, 2)
    # v·∇u = 1*2x + 2*2y = 2x + 4y
    u_2d = X**2 + Y**2
    v = np.array([1.0, 2.0])

    D_v = DirectDerivOperator(direction=v, spacings=[dx, dy], field_shape=(Nx, Ny))
    print(f"  Operator: {D_v.shape}, direction={v.tolist()}")

    du_dv = D_v(u_2d)
    expected = 2 * X + 4 * Y

    error = np.max(np.abs(du_dv[5:-5, 5:-5] - expected[5:-5, 5:-5]))
    print(f"  v·∇(x² + y²) interior error: {error:.2e}")
    assert error < 1e-10, f"Error too large: {error}"
    print("  OK: Constant direction in 2D")

    # Test 3: Spatially varying direction in 2D
    print("\n[2D Spatially Varying Direction]")

    # v(x,y) = (y, -x) - rotation field
    # u = x^2 + y^2
    # v·∇u = y*2x + (-x)*2y = 2xy - 2xy = 0
    v_field = np.stack([Y, -X], axis=-1)  # shape (Nx, Ny, 2)

    D_v_varying = DirectDerivOperator(direction=v_field, spacings=[dx, dy], field_shape=(Nx, Ny))
    print(f"  Operator: {D_v_varying.shape}, varying direction")
    print(f"  Direction field shape: {v_field.shape}")

    du_dv_varying = D_v_varying(u_2d)

    # Should be approximately zero (rotation of radial function)
    error = np.max(np.abs(du_dv_varying[5:-5, 5:-5]))
    print(f"  (y,-x)·∇(x² + y²) interior max: {error:.2e} (expected ~0)")
    assert error < 1e-10, f"Error too large: {error}"
    print("  OK: Spatially varying direction")

    # Test 4: scipy compatibility
    print("\n[scipy compatibility]")
    assert isinstance(D_v, LinearOperator)
    du_matvec = D_v @ u_2d.ravel()
    assert np.allclose(D_v(u_2d).ravel(), du_matvec)
    print("  OK: D_v(u) == D_v @ u.ravel()")

    # Test repr
    print("\n[String representation]")
    print(D_v)

    print("\nAll DirectDerivOperator tests passed!")

    # =========================================================================
    # NormalDerivOperator Tests
    # =========================================================================
    print("\n" + "=" * 60)
    print("Testing NormalDerivOperator...")
    print("=" * 60)

    # Test 1: Constant normal (axis-aligned) in 1D
    print("\n[1D Axis-Aligned Normal]")

    x = np.linspace(0, 1, 100)
    dx = x[1] - x[0]
    u_1d = x**2  # du/dx = 2x

    # Left boundary: outward normal = -1 (points left)
    D_n_left = NormalDerivOperator.from_axis(axis=0, sign=-1, spacings=[dx], field_shape=(100,))
    print(f"  Left boundary: {D_n_left.normal}")
    du_dn_left = D_n_left(u_1d)
    # du/dn = n·∇u = -1 * 2x = -2x
    expected_left = -2 * x
    error_left = np.max(np.abs(du_dn_left[5:-5] - expected_left[5:-5]))
    print(f"  du/dn (n=-1) interior error: {error_left:.2e}")
    assert error_left < 1e-10

    # Right boundary: outward normal = +1 (points right)
    D_n_right = NormalDerivOperator.from_axis(axis=0, sign=+1, spacings=[dx], field_shape=(100,))
    print(f"  Right boundary: {D_n_right.normal}")
    du_dn_right = D_n_right(u_1d)
    # du/dn = n·∇u = +1 * 2x = 2x
    expected_right = 2 * x
    error_right = np.max(np.abs(du_dn_right[5:-5] - expected_right[5:-5]))
    print(f"  du/dn (n=+1) interior error: {error_right:.2e}")
    assert error_right < 1e-10
    print("  OK: Axis-aligned normals in 1D")

    # Test 2: From SDF (circle in 2D)
    print("\n[2D Normal from SDF (Circle)]")

    Nx, Ny = 50, 50
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    # SDF for circle centered at origin, radius 0.5
    # phi > 0 outside, phi < 0 inside
    radius = 0.5
    sdf_circle = np.sqrt(X**2 + Y**2) - radius

    D_n_sdf = NormalDerivOperator.from_sdf(sdf_circle, spacings=[dx, dy])
    print(f"  SDF shape: {sdf_circle.shape}")
    print(f"  Normal field shape: {D_n_sdf.normal.shape}")
    print(f"  Normal source: {D_n_sdf.normal_source}")

    # Test on radial function u = x^2 + y^2
    # Gradient: ∇u = (2x, 2y)
    # Outward normal on circle: n = (x, y) / r
    # du/dn = n·∇u = (x/r)(2x) + (y/r)(2y) = 2(x^2 + y^2)/r = 2r
    u_radial = X**2 + Y**2
    du_dn_sdf = D_n_sdf(u_radial)

    # Check on the circle (where |∇φ| = 1)
    # Near the boundary (sdf ≈ 0), du/dn should be ≈ 2*radius = 1.0
    near_boundary = np.abs(sdf_circle) < 0.1
    r_field = np.sqrt(X**2 + Y**2)
    expected_du_dn = 2 * r_field  # du/dn = 2r

    # Compare where we're near the boundary
    if np.any(near_boundary):
        error_boundary = np.mean(np.abs(du_dn_sdf[near_boundary] - expected_du_dn[near_boundary]))
        print(f"  Near-boundary mean error: {error_boundary:.2e}")
        # Allow larger error due to discrete SDF gradient
        assert error_boundary < 0.2, f"Error too large: {error_boundary}"
    print("  OK: Normal from SDF")

    # Test 3: Explicit normal in 2D
    print("\n[2D Explicit Normal Field]")

    # Create radial outward normals (like expanding circle)
    r_safe = np.maximum(r_field, 1e-10)
    explicit_normals = np.stack([X / r_safe, Y / r_safe], axis=-1)
    # Fix origin (set to arbitrary unit vector)
    origin_mask = r_field < 1e-8
    explicit_normals[origin_mask, :] = [1.0, 0.0]

    D_n_explicit = NormalDerivOperator(normal=explicit_normals, spacings=[dx, dy], field_shape=(Nx, Ny))
    print(f"  Explicit normal field shape: {explicit_normals.shape}")

    du_dn_explicit = D_n_explicit(u_radial)

    # du/dn = (x/r, y/r) · (2x, 2y) = 2(x² + y²)/r = 2r
    error_explicit = np.max(np.abs(du_dn_explicit[5:-5, 5:-5] - expected_du_dn[5:-5, 5:-5]))
    print(f"  du/dn interior error: {error_explicit:.2e}")
    assert error_explicit < 0.1
    print("  OK: Explicit normal field")

    # Test 4: Unit normal validation
    print("\n[Unit Normal Validation]")
    try:
        bad_normal = np.array([1.0, 1.0])  # Not unit length
        NormalDerivOperator(bad_normal, spacings=[dx, dy], field_shape=(Nx, Ny))
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        print(f"  Correctly rejected non-unit normal: {e}")
    print("  OK: Unit normal validation")

    # Test 5: scipy compatibility
    print("\n[scipy compatibility]")
    assert isinstance(D_n_right, LinearOperator)
    du_matvec = D_n_right @ u_1d.ravel()
    assert np.allclose(D_n_right(u_1d).ravel(), du_matvec)
    print("  OK: D_n(u) == D_n @ u.ravel()")

    # Test repr
    print("\n[String representation]")
    print(D_n_right)
    print()
    print(D_n_sdf)

    print("\nAll NormalDerivOperator tests passed!")

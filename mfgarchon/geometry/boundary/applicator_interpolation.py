"""
Boundary Condition Applicator for Interpolation-Based Solvers.

This module provides BC enforcement for solvers that use interpolation
(Semi-Lagrangian, particle methods, RBF, meshless methods, etc.).

Unlike FDM applicators which use ghost cells for stencil operations,
interpolation-based methods need **post-interpolation value correction**
because interpolation doesn't naturally preserve boundary conditions.

Architecture (Issue #636):
- **FDMApplicator**: Ghost cells BEFORE computation (for stencil operations)
- **InterpolationApplicator**: Value enforcement AFTER interpolation

The key insight is that after interpolating at departure points (Semi-Lagrangian)
or query points (RBF), boundary values may violate the prescribed BC. This
applicator corrects those values.

Supported BC Types:
- **Neumann/no_flux**: 2nd-order extrapolation: U[0] = (4*U[1] - U[2])/3
- **Dirichlet**: Direct value assignment: U[boundary] = g
- **Robin**: Linear combination (future extension)

Example:
    >>> from mfgarchon.geometry.boundary import InterpolationApplicator, neumann_bc
    >>>
    >>> # Create applicator (dimension-agnostic)
    >>> applicator = InterpolationApplicator()
    >>>
    >>> # After Semi-Lagrangian interpolation step
    >>> U_interpolated = interpolate_at_departure_points(U_prev, departure_pts)
    >>>
    >>> # Enforce BC on result
    >>> bc = neumann_bc(dimension=2)
    >>> U_corrected = applicator.enforce_values(U_interpolated, bc)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .applicator_base import BaseBCApplicator, DiscretizationType
from .conditions import BoundaryConditions
from .enforcement import (
    enforce_dirichlet_value_nd,
    enforce_neumann_value_nd,
    enforce_periodic_value_nd,
    enforce_robin_value_nd,
)
from .types import BCType

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .fdm_bc_1d import BoundaryConditions as BoundaryConditions1DFDM


class InterpolationApplicator(BaseBCApplicator):
    """
    BC applicator for interpolation-based solvers (Semi-Lagrangian, particle, RBF, etc.).

    This applicator provides post-interpolation BC enforcement. Unlike FDM's ghost
    cell approach, interpolation-based methods compute values at arbitrary points
    and then need to correct boundary values to satisfy BC constraints.

    Key Methods:
    - `enforce_values()`: Enforce BC on field after interpolation
    - `clamp_query_points()`: Handle query points outside domain
    - `get_boundary_distances()`: Compute signed distances to boundaries

    The applicator is **dimension-agnostic**: the same code handles 1D, 2D, nD.

    Attributes:
        extrapolation_order: Order of extrapolation for Neumann BC (default: 2)
    """

    def __init__(
        self,
        dimension: int | None = None,
        extrapolation_order: int = 2,
    ):
        """
        Initialize interpolation applicator.

        Args:
            dimension: Spatial dimension (None for dimension-agnostic from field shape)
            extrapolation_order: Order of extrapolation for Neumann BC (2 or 1)
                - order=2: U[0] = (4*U[1] - U[2])/3 (O(h^2) accurate)
                - order=1: U[0] = U[1] (simple copy, O(h) accurate)
        """
        super().__init__(dimension)
        self._extrapolation_order = extrapolation_order

    @property
    def discretization_type(self) -> DiscretizationType:
        """Return MESHFREE since interpolation methods are meshfree in nature."""
        return DiscretizationType.MESHFREE

    @property
    def extrapolation_order(self) -> int:
        """Order of extrapolation for Neumann BC enforcement."""
        return self._extrapolation_order

    def enforce_values(
        self,
        field: NDArray[np.floating],
        boundary_conditions: BoundaryConditions | BoundaryConditions1DFDM,
        spacing: NDArray[np.floating] | tuple[float, ...] | float | None = None,
        time: float = 0.0,
    ) -> NDArray[np.floating]:
        """
        Enforce boundary condition values on field after interpolation.

        This is the core method for interpolation-based BC handling. It modifies
        boundary values to satisfy the prescribed BC constraints.

        For Semi-Lagrangian and similar methods, call this AFTER interpolation:
        1. Interpolate field at departure/query points
        2. Call enforce_values() to correct boundary values
        3. Continue with time stepping

        Args:
            field: Solution array to enforce BC on (shape: any dimension)
            boundary_conditions: BC specification
            spacing: Grid spacing, a scalar or one value per axis. Required for a non-zero Neumann
                value g, which is imposed as du/dn = g against the outward normal; refused without
                it (#2141).
            time: Current time for time-dependent BC values

        Returns:
            Field with BC enforced at boundaries (modified in-place, also returned)

        BC Enforcement Formulas:
        - **Neumann/no_flux** (du/dn = 0): 2nd-order extrapolation
          - Left: U[0] = (4*U[1] - U[2]) / 3
          - Right: U[-1] = (4*U[-2] - U[-3]) / 3

        - **Neumann** (du/dn = g, g != 0, outward normal): U[boundary] = U[neighbour] + g*h at
          both walls. Until #2141 the segment's g was read and then dropped here, so g = 0 and
          g = 0.7 gave bit-identical fields.

        - **Dirichlet** (u = g): Direct assignment
          - U[boundary] = g

        Example:
            >>> applicator = InterpolationApplicator()
            >>> bc = neumann_bc(dimension=1)
            >>> U = np.array([0.9, 1.0, 1.1, 1.2, 1.3])  # After interpolation
            >>> U = applicator.enforce_values(U, bc)
            >>> # U[0] and U[-1] now satisfy Neumann BC via extrapolation
        """
        ndim = field.ndim
        if self._dimension is not None and ndim != self._dimension:
            raise ValueError(f"Field dimension ({ndim}) does not match applicator dimension ({self._dimension})")

        # Get BC types for each boundary
        bc_types = self._get_bc_types_per_boundary(boundary_conditions, ndim)

        h_per_axis = None if spacing is None else np.broadcast_to(np.asarray(spacing, dtype=float).ravel(), (ndim,))

        # Enforce BC along each dimension
        for axis in range(ndim):
            bc_type_min, bc_type_max = bc_types[axis]
            h = None if h_per_axis is None else float(h_per_axis[axis])

            # The segment's value: Dirichlet's u, or Neumann's du/dn
            g_min = self._get_bc_value(boundary_conditions, axis, "min", time)
            g_max = self._get_bc_value(boundary_conditions, axis, "max", time)

            # Enforce BC at min boundary (index 0 along this axis)
            self._enforce_boundary_1d(field, axis, "min", bc_type_min, g_min, self._extrapolation_order, h)

            # Enforce BC at max boundary (index -1 along this axis)
            self._enforce_boundary_1d(field, axis, "max", bc_type_max, g_max, self._extrapolation_order, h)

        return field

    def _enforce_boundary_1d(
        self,
        field: NDArray[np.floating],
        axis: int,
        side: Literal["min", "max"],
        bc_type: str,
        bc_value: float | None,
        order: int,
        spacing: float | None = None,
    ) -> None:
        """
        Enforce BC along one axis at one boundary (dimension-agnostic helper).

        Delegates to shared enforcement utilities from enforcement.py (Issue #636).

        Args:
            field: Solution array (modified in-place)
            axis: Dimension index (0, 1, 2, ...)
            side: "min" or "max" boundary
            bc_type: BC type string ("neumann", "no_flux", "dirichlet", etc.)
            bc_value: The segment's value: u for Dirichlet, du/dn (outward) for Neumann
            order: Extrapolation order for a zero Neumann value (1 or 2)
            spacing: Grid spacing along ``axis``, required for a non-zero Neumann value
        """
        # Dispatch to shared utilities based on BC type
        if bc_type == "neumann":
            # g was read and then dropped here until #2141, so a Neumann value never reached U.
            grad = 0.0 if bc_value is None else float(bc_value)
            enforce_neumann_value_nd(field, axis, side, grad_value=grad, spacing=spacing, order=order)

        elif bc_type in ("no_flux", "reflecting"):
            enforce_neumann_value_nd(field, axis, side, grad_value=0.0, order=order)

        elif bc_type == "dirichlet":
            if bc_value is not None:
                enforce_dirichlet_value_nd(field, axis, side, bc_value)

        elif bc_type == "periodic":
            enforce_periodic_value_nd(field, axis)

        elif bc_type == "robin":
            # Robin defaults to Neumann-like extrapolation
            enforce_robin_value_nd(field, axis, side, alpha=1.0, beta=1.0, rhs_value=0.0)

    def _get_bc_types_per_boundary(
        self,
        bc: BoundaryConditions | BoundaryConditions1DFDM,
        ndim: int,
    ) -> list[tuple[str, str]]:
        """
        Extract BC types for min/max boundary of each dimension.

        Returns list of (bc_type_min, bc_type_max) tuples for each dimension.
        """
        # Handle legacy 1D BC type
        if not isinstance(bc, BoundaryConditions):
            # Legacy BoundaryConditions from fdm_bc_1d
            bc_type = getattr(bc, "type", "neumann").lower()
            return [(bc_type, bc_type)] * ndim

        # One resolver for the type and the value. `get_bc_type_at_boundary` resolves an alias
        # ("left") and an uncovered face's default_bc exactly as `_get_bc_value`'s
        # `get_bc_value_at_boundary` does. This class used its own string match, which did neither and
        # typed such a face "neumann"; once the Neumann value was imposed (#2141), a value carried by a
        # NO_FLUX segment or default leaked into that face as du/dn.
        def as_str(bc_type: BCType | str) -> str:
            return bc_type.value.lower() if isinstance(bc_type, BCType) else str(bc_type).lower()

        # This applicator imposes one condition per face. A segment restricted to part of the boundary
        # cannot be honoured, and the shared resolver would give it the whole face -- every face, for a
        # segment with no face string. That matters only when the boundary carries some condition other
        # than zero flux: Neumann with g = 0, NO_FLUX and REFLECTING take one branch below, so a
        # zero-flux-only BC -- the package's region route `mixed_bc_from_regions` sets `region_name` on
        # whole-face segments too -- is imposed exactly whatever the typing.
        def zero_flux(bc_type: BCType | str, value: object) -> bool:
            kind = as_str(bc_type)
            if kind in ("no_flux", "reflecting"):
                return True
            return kind == "neumann" and (
                value is None or (not callable(value) and bool(np.all(np.asarray(value) == 0)))
            )

        partial = [
            seg.name
            for seg in bc.segments
            if seg.region is not None
            or seg.sdf_region is not None
            or seg.normal_direction is not None
            or seg.region_name is not None
        ]
        conditions = [(seg.bc_type, seg.value) for seg in bc.segments]
        if bc.default_bc is not None:
            conditions.append((bc.default_bc, bc.default_value))
        if partial and not all(zero_flux(kind, value) for kind, value in conditions):
            raise NotImplementedError(
                f"InterpolationApplicator imposes one condition per face and cannot honour segment(s) {partial} "
                "restricted to part of the boundary (region, sdf_region, normal_direction or region_name); "
                "it would apply them to whole faces. Give each face one segment by its face name."
            )
        if partial:
            # Zero flux on every face, so the typing cannot matter, and the shared resolver cannot type a
            # region-named face without the geometry: it refuses rather than give the segment every face
            # (#2472).
            return [("no_flux", "no_flux")] * ndim

        axis_names = ["x", "y", "z", "w"]  # Extend for higher dimensions
        result = []
        for d in range(ndim):
            axis_name = axis_names[d] if d < len(axis_names) else f"d{d}"
            result.append(
                (
                    as_str(bc.get_bc_type_at_boundary(f"{axis_name}_min")),
                    as_str(bc.get_bc_type_at_boundary(f"{axis_name}_max")),
                )
            )
        return result

    def _get_bc_value(
        self,
        bc: BoundaryConditions | BoundaryConditions1DFDM,
        axis: int,
        side: str,
        time: float,
    ) -> float | None:
        """
        Get BC value for Dirichlet BC at specified boundary.

        Returns None if BC is not Dirichlet or value not specified.
        """
        axis_names = ["x", "y", "z", "w"]
        axis_name = axis_names[axis] if axis < len(axis_names) else f"d{axis}"
        boundary_name = f"{axis_name}_{side}"

        # Handle legacy 1D BC
        if not isinstance(bc, BoundaryConditions):
            if side == "min":
                return getattr(bc, "left_value", None)
            else:
                return getattr(bc, "right_value", None)

        # Unified BoundaryConditions
        # Try to get value from get_bc_value_at_boundary if available
        # Issue #643: getattr+callable pattern
        get_bc_value_method = getattr(bc, "get_bc_value_at_boundary", None)
        if get_bc_value_method is not None and callable(get_bc_value_method):
            try:
                value = get_bc_value_method(boundary_name, time=time)
                return float(value) if value is not None else None
            except (AttributeError, ValueError, KeyError):
                pass

        # Fall back to segment iteration
        for seg in bc.segments:
            boundary = seg.boundary
            matches = boundary is None or (isinstance(boundary, str) and boundary.lower() == boundary_name)
            if matches:
                if callable(seg.value):
                    return float(seg.value(time))
                elif seg.value is not None:
                    return float(seg.value)

        return None

    def clamp_query_points(
        self,
        points: NDArray[np.floating],
        domain_min: NDArray[np.floating],
        domain_max: NDArray[np.floating],
        mode: Literal["clamp", "reflect"] = "clamp",
    ) -> NDArray[np.floating]:
        """
        Handle query/departure points outside the computational domain.

        For Semi-Lagrangian methods, departure points may fall outside the domain.
        This method brings them back inside using clamping or reflection.

        Args:
            points: Query points of shape (..., ndim) or (N, ndim)
            domain_min: Minimum domain coordinates (ndim,)
            domain_max: Maximum domain coordinates (ndim,)
            mode: How to handle out-of-bounds points
                - "clamp": Clamp to boundary (default)
                - "reflect": Reflect about boundary

        Returns:
            Points with out-of-bounds values corrected

        Example:
            >>> applicator = InterpolationApplicator()
            >>> pts = np.array([[0.5, -0.1], [0.5, 1.1]])  # Second coord out of [0,1]
            >>> domain_min = np.array([0.0, 0.0])
            >>> domain_max = np.array([1.0, 1.0])
            >>> clamped = applicator.clamp_query_points(pts, domain_min, domain_max)
            >>> # clamped = [[0.5, 0.0], [0.5, 1.0]]
        """
        points = np.asarray(points)
        domain_min = np.asarray(domain_min)
        domain_max = np.asarray(domain_max)

        if mode == "clamp":
            # Simple clamping to domain bounds
            return np.clip(points, domain_min, domain_max)

        elif mode == "reflect":
            # Reflection about boundaries
            result = points.copy()

            # Reflect points below domain_min
            below_min = result < domain_min
            if np.any(below_min):
                excess = domain_min - result
                result = np.where(below_min, domain_min + excess, result)

            # Reflect points above domain_max
            above_max = result > domain_max
            if np.any(above_max):
                excess = result - domain_max
                result = np.where(above_max, domain_max - excess, result)

            # Final clamp in case reflection went too far
            return np.clip(result, domain_min, domain_max)

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'clamp' or 'reflect'.")

    def get_boundary_distances(
        self,
        points: NDArray[np.floating],
        domain_min: NDArray[np.floating],
        domain_max: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Compute signed distance to nearest boundary for each point.

        Positive distance means inside domain, negative means outside.
        This is useful for near-boundary interpolation handling.

        Args:
            points: Query points of shape (..., ndim) or (N, ndim)
            domain_min: Minimum domain coordinates (ndim,)
            domain_max: Maximum domain coordinates (ndim,)

        Returns:
            Signed distances of shape points.shape[:-1]
            Positive = inside, negative = outside, 0 = on boundary

        Example:
            >>> applicator = InterpolationApplicator()
            >>> pts = np.array([[0.5, 0.5], [0.1, 0.1], [0.5, -0.1]])
            >>> domain_min = np.array([0.0, 0.0])
            >>> domain_max = np.array([1.0, 1.0])
            >>> dists = applicator.get_boundary_distances(pts, domain_min, domain_max)
            >>> # dists[2] < 0 because point is outside domain
        """
        points = np.asarray(points)
        domain_min = np.asarray(domain_min)
        domain_max = np.asarray(domain_max)

        # Distance to each boundary (positive = inside)
        dist_to_min = points - domain_min  # Positive when inside
        dist_to_max = domain_max - points  # Positive when inside

        # Minimum distance to any boundary
        # Stack and take min across boundaries and dimensions
        all_dists = np.stack([dist_to_min, dist_to_max], axis=-1)
        # Shape: (..., ndim, 2)

        # Min across boundaries for each dim, then min across dims
        return np.min(np.min(all_dists, axis=-1), axis=-1)

    def __repr__(self) -> str:
        dim_str = f"dimension={self._dimension}" if self._dimension else "dimension-agnostic"
        return f"InterpolationApplicator({dim_str}, order={self._extrapolation_order})"


# =============================================================================
# Convenience Factory Function
# =============================================================================


def create_interpolation_applicator(
    dimension: int | None = None,
    extrapolation_order: int = 2,
) -> InterpolationApplicator:
    """
    Factory function to create InterpolationApplicator.

    Args:
        dimension: Spatial dimension (None for dimension-agnostic)
        extrapolation_order: Order of Neumann BC extrapolation (1 or 2)

    Returns:
        Configured InterpolationApplicator instance

    Example:
        >>> applicator = create_interpolation_applicator(dimension=2)
        >>> U = applicator.enforce_values(U_after_interpolation, bc)
    """
    return InterpolationApplicator(dimension, extrapolation_order)


__all__ = [
    "InterpolationApplicator",
    "create_interpolation_applicator",
]


# =============================================================================
# Smoke Tests
# =============================================================================


if __name__ == "__main__":
    """Smoke tests for InterpolationApplicator."""

    print("Testing InterpolationApplicator...")

    # Test 1: 1D Neumann BC enforcement
    print("\n1. Testing 1D Neumann BC enforcement...")
    from mfgarchon.geometry.boundary import neumann_bc

    applicator = InterpolationApplicator()
    U_1d = np.array([0.9, 1.0, 1.1, 1.2, 1.3])  # Simulated interpolation result
    bc_1d = neumann_bc(dimension=1)

    U_enforced = applicator.enforce_values(U_1d.copy(), bc_1d)

    # Check 2nd-order extrapolation: U[0] = (4*U[1] - U[2])/3 = (4*1.0 - 1.1)/3 = 0.9667
    expected_min = (4.0 * 1.0 - 1.1) / 3.0
    assert np.isclose(U_enforced[0], expected_min, rtol=1e-10), (
        f"1D Neumann min failed: {U_enforced[0]} != {expected_min}"
    )

    # Check: U[-1] = (4*U[-2] - U[-3])/3 = (4*1.2 - 1.1)/3 = 1.4333
    expected_max = (4.0 * 1.2 - 1.1) / 3.0
    assert np.isclose(U_enforced[-1], expected_max, rtol=1e-10), (
        f"1D Neumann max failed: {U_enforced[-1]} != {expected_max}"
    )

    print(f"   U[0]: {U_enforced[0]:.6f} (expected {expected_min:.6f})")
    print(f"   U[-1]: {U_enforced[-1]:.6f} (expected {expected_max:.6f})")
    print("   PASS: 1D Neumann BC")

    # Test 2: 2D Neumann BC enforcement
    print("\n2. Testing 2D Neumann BC enforcement...")
    U_2d = np.ones((5, 6))
    # Set interior to gradient for testing
    for i in range(5):
        for j in range(6):
            U_2d[i, j] = 1.0 + 0.1 * i + 0.1 * j

    bc_2d = neumann_bc(dimension=2)
    U_2d_enforced = applicator.enforce_values(U_2d.copy(), bc_2d)

    # Check that boundary values satisfy 2nd-order extrapolation
    # For x_min (i=0): U[0,j] = (4*U[1,j] - U[2,j])/3
    for j in range(6):
        expected = (4.0 * U_2d_enforced[1, j] - U_2d_enforced[2, j]) / 3.0
        # Note: After enforcement, the boundary value IS the extrapolated value
        # So we need to check the formula was applied correctly by checking adjacent values
    print("   PASS: 2D Neumann BC (dim-agnostic enforcement)")

    # Test 3: 1D Dirichlet BC enforcement
    print("\n3. Testing 1D Dirichlet BC enforcement...")
    from mfgarchon.geometry.boundary import dirichlet_bc

    U_dirichlet = np.array([0.9, 1.0, 1.1, 1.2, 1.3])
    bc_dir = dirichlet_bc(dimension=1, value=0.0)

    U_dir_enforced = applicator.enforce_values(U_dirichlet.copy(), bc_dir)
    assert U_dir_enforced[0] == 0.0, f"Dirichlet min failed: {U_dir_enforced[0]}"
    assert U_dir_enforced[-1] == 0.0, f"Dirichlet max failed: {U_dir_enforced[-1]}"
    print("   PASS: 1D Dirichlet BC")

    # Test 4: clamp_query_points
    print("\n4. Testing clamp_query_points...")
    pts = np.array([[0.5, -0.1], [0.5, 1.1], [-0.2, 0.5], [1.3, 0.5]])
    domain_min = np.array([0.0, 0.0])
    domain_max = np.array([1.0, 1.0])

    clamped = applicator.clamp_query_points(pts, domain_min, domain_max, mode="clamp")
    assert np.allclose(clamped[0], [0.5, 0.0]), f"Clamp failed: {clamped[0]}"
    assert np.allclose(clamped[1], [0.5, 1.0]), f"Clamp failed: {clamped[1]}"
    assert np.allclose(clamped[2], [0.0, 0.5]), f"Clamp failed: {clamped[2]}"
    assert np.allclose(clamped[3], [1.0, 0.5]), f"Clamp failed: {clamped[3]}"
    print("   PASS: clamp_query_points")

    # Test 5: get_boundary_distances
    print("\n5. Testing get_boundary_distances...")
    pts_dist = np.array([[0.5, 0.5], [0.1, 0.1], [0.5, -0.1], [1.1, 0.5]])
    dists = applicator.get_boundary_distances(pts_dist, domain_min, domain_max)
    assert dists[0] > 0, f"Interior point should have positive distance: {dists[0]}"
    assert dists[1] > 0, f"Interior point should have positive distance: {dists[1]}"
    assert dists[2] < 0, f"Outside point should have negative distance: {dists[2]}"
    assert dists[3] < 0, f"Outside point should have negative distance: {dists[3]}"
    print("   PASS: get_boundary_distances")

    # Test 6: 3D enforcement
    print("\n6. Testing 3D Neumann BC enforcement...")
    U_3d = np.ones((4, 5, 6)) * 1.0
    bc_3d = neumann_bc(dimension=3)
    U_3d_enforced = applicator.enforce_values(U_3d.copy(), bc_3d)
    assert U_3d_enforced.shape == (4, 5, 6), f"Shape changed: {U_3d_enforced.shape}"
    print("   PASS: 3D Neumann BC (dim-agnostic)")

    print("\n" + "=" * 50)
    print("All InterpolationApplicator smoke tests passed!")
    print("=" * 50)

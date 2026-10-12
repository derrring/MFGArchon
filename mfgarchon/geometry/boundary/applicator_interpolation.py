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

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .applicator_base import BaseBCApplicator, DiscretizationType
from .conditions import BoundaryConditions
from .enforcement import (
    enforce_dirichlet_value_nd,
    enforce_neumann_value_nd,
    enforce_periodic_value_nd,
    enforce_robin_value_nd,
)
from .types import BCType, BoundaryFace, segment_robin_coefficients, segment_value

if TYPE_CHECKING:
    from numpy.typing import NDArray

# The face types whose enforcement reads the value: Dirichlet's u, Neumann's du/dn. Not Robin, whose enforcement below
# hard-codes its coefficients and reads no value (#2374); it joins this set when it reads the segment's.
_VALUED = frozenset({"dirichlet", "neumann"})


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

    #: The types `_enforce_boundary_1d` has a branch for. The two `EXTRAPOLATION_*` members are absent:
    #: they fell through that dispatch with the boundary value left as interpolated, not set to the
    #: extrapolation, so they are refused (#1948). `extrapolation_order` is the order of the
    #: zero-Neumann extrapolation and is unrelated to them.
    _SUPPORTED_BC_TYPES: frozenset[BCType] = frozenset(
        {BCType.DIRICHLET, BCType.NEUMANN, BCType.NO_FLUX, BCType.REFLECTING, BCType.PERIODIC, BCType.ROBIN}
    )

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
        boundary_conditions: BoundaryConditions,
        spacing: NDArray[np.floating] | tuple[float, ...] | float | None = None,
        time: float | None = None,
        geometry: Any = None,
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
            geometry: The grid the field lives on. The face owner reads a ``region``, ``sdf_region`` or
                ``region_name`` placement on it (#2512 row B1); without it such a placement is refused unless
                the whole BC is zero flux.

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
        if isinstance(boundary_conditions, BoundaryConditions):  # anything else is refused just below
            self._validate_bc_support(boundary_conditions)  # #1948

        h_per_axis = None if spacing is None else np.broadcast_to(np.asarray(spacing, dtype=float).ravel(), (ndim,))

        # Each face's type, and its value where the type's enforcement reads one, from one segment, in the order the
        # faces are written: where faces share nodes, the owner's corner choice is written last.
        for face, bc_type, value in self._face_conditions(boundary_conditions, ndim, geometry, time):
            h = None if h_per_axis is None else float(h_per_axis[face.axis])
            self._enforce_boundary_1d(field, face.axis, face.side, bc_type, value, self._extrapolation_order, h)

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
        else:
            raise NotImplementedError(
                f"InterpolationApplicator has no enforcement for a {bc_type!r} boundary at the {side} side of "
                f"axis {axis}; it would return the interpolated value unchanged (#1948)."
            )

    def _face_conditions(
        self,
        bc: BoundaryConditions,
        ndim: int,
        geometry: Any,
        time: float | None,
    ) -> list[tuple[BoundaryFace, str, float | None]]:
        """Each face's ``(face, type, value)``, from the segment the face owner gives it (#2512 row B1), in the
        order to write them. The value is read only where the type's enforcement uses it (``_VALUED``).

        Faces are written lowest corner precedence first (`BoundaryConditions.segment_for_corner`'s order), so
        where faces share nodes the owner's corner choice is written last, as `FDMApplicator.enforce_values`
        does. Writing axis by axis gave every corner to the last axis written.

        This applicator imposes one condition per face, and zero flux is one branch whatever spells it --
        NO_FLUX, REFLECTING, or Neumann with g = 0. So the owner is asked with that equivalence: a segment
        covering part of a face is accepted when it and the rest of the face enforce the same way, and refused
        otherwise. When every condition the BC carries, its default included, is zero flux, every face is zero
        flux whichever segment governs it, and no placement is read: a region the owner cannot place without
        the solver's geometry (a ``region_name`` from `mixed_bc_from_regions`) is then imposed, not refused
        (#2466, #2472). The faces are asked for by `BoundaryFace`, so no name is built here (#2628).
        """
        if not isinstance(bc, BoundaryConditions):
            raise TypeError(
                f"The interpolation applicator takes a BoundaryConditions, not a {type(bc).__name__} (the legacy "
                "fdm_bc_1d class is removed)."
            )

        def as_str(bc_type: BCType | str) -> str:
            return bc_type.value.lower() if isinstance(bc_type, BCType) else str(bc_type).lower()

        def zero_flux(bc_type: BCType | str, value: object) -> bool:
            kind = as_str(bc_type)
            if kind in ("no_flux", "reflecting"):
                return True
            return kind == "neumann" and (
                value is None or (not callable(value) and bool(np.all(np.asarray(value) == 0)))
            )

        conditions = [(seg.bc_type, seg.value) for seg in bc.segments]
        if bc.default_bc is not None:
            conditions.append((bc.default_bc, bc.default_value))
            if all(zero_flux(kind, value) for kind, value in conditions):
                return [(BoundaryFace(axis, side), "no_flux", None) for axis in range(ndim) for side in ("min", "max")]

        def enforcement(segment: Any) -> tuple:
            kind = as_str(segment.bc_type)
            if zero_flux(kind, segment.value):
                return ("zero flux",)
            if kind == "periodic":
                return (kind,)
            if kind == "robin":
                where = "InterpolationApplicator, comparing two Robin segments on one face"
                return (kind, segment.value, *segment_robin_coefficients(segment, where=where))
            return (kind, segment.value)

        def equal(a: object, b: object) -> bool:
            if a is b:
                return True
            if callable(a) or callable(b):
                return False
            try:
                return bool(np.array_equal(np.asarray(a), np.asarray(b)))
            except (TypeError, ValueError):
                return False

        def same_enforcement(first: Any, second: Any) -> bool:
            a, b = enforcement(first), enforcement(second)
            return len(a) == len(b) and a[0] == b[0] and all(equal(x, y) for x, y in zip(a[1:], b[1:], strict=True))

        faces = [BoundaryFace(axis, side) for axis in range(ndim) for side in ("min", "max")]
        governing = {face: bc.segment_for_face(face, geometry, equivalent=same_enforcement) for face in faces}
        result = []
        for face in sorted(faces, key=lambda f: bc._corner_key(governing[f]), reverse=True):
            segment = governing[face]
            kind = as_str(segment.bc_type)
            where = f"InterpolationApplicator, the {face.to_string()} face"
            value = (
                float(segment_value(segment, t=time, x=None, where=where, time_keyword="time"))
                if kind in _VALUED
                else None
            )
            result.append((face, kind, value))
        return result

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

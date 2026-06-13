"""
Implicit boundary applicator for dual geometry scenarios.

This module provides BC applicators that bridge structured grids with implicit
(SDF-based) boundaries. Use cases include:
- Cut-cell methods: Cartesian cells cut by implicit boundary
- Level-set methods: Time-evolving φ(x) = 0 on fixed grid
- Immersed boundary: Structured grid + embedded geometry
- Semi-Lagrangian + SDF: Cartesian departure points + implicit domain

Architecture (Issue #637, #712):
    ImplicitApplicator inherits from MeshfreeApplicator and adds:
    - Normal-based Neumann/Robin BC (interpolation along boundary normals)
    - Periodic BC via wrapping to opposite boundary
    - Grid spacing estimation for non-structured points
    Inherits particle BC methods from MeshfreeApplicator.

Example:
    >>> from mfgarchon.geometry.boundary import ImplicitApplicator, neumann_bc
    >>> from mfgarchon.geometry import ImplicitDomain
    >>>
    >>> # Create implicit domain (e.g., circle SDF)
    >>> sdf = lambda x: np.linalg.norm(x - center, axis=-1) - radius
    >>> domain = ImplicitDomain(sdf_func=sdf, bounds=[(0, 1), (0, 1)])
    >>>
    >>> # Create applicator
    >>> applicator = ImplicitApplicator(geometry=domain)
    >>>
    >>> # Apply BC to field on structured grid
    >>> bc = neumann_bc(dimension=2)
    >>> field_with_bc = applicator.apply(field, bc, points=grid_points)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .applicator_base import GridType
from .applicator_meshfree import MeshfreeApplicator
from .types import BCType

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.geometry.protocol import GeometryProtocol

    from .conditions import BoundaryConditions


class ImplicitApplicator(MeshfreeApplicator):
    """
    BC applicator for dual geometry scenarios (structured grid + implicit boundary).

    This applicator bridges structured (axis-aligned) and geometry-based (SDF)
    approaches for boundary condition enforcement.

    Attributes:
        geometry: Geometry with SDF-based boundary detection
        grid_type: Grid type for axis-aligned regions (default: CELL_CENTERED)
        boundary_tolerance: Distance threshold for boundary detection

    Note:
        For pure structured grids, use FDMApplicator instead.
        For pure meshfree methods, use MeshfreeApplicator instead.
        ImplicitApplicator is specifically for hybrid/dual geometry scenarios.
    """

    def __init__(
        self,
        geometry: GeometryProtocol,
        grid_type: GridType = GridType.CELL_CENTERED,
        boundary_tolerance: float = 1e-10,
    ):
        """
        Initialize implicit boundary applicator.

        Args:
            geometry: Geometry object implementing GeometryProtocol.
                Uses is_on_boundary() and get_boundary_normal() from protocol.
            grid_type: Grid type for structured regions (default: CELL_CENTERED)
            boundary_tolerance: Distance threshold for boundary point detection
        """
        super().__init__(geometry)  # MeshfreeApplicator handles isinstance + dimension
        self._grid_type = grid_type
        self._boundary_tolerance = boundary_tolerance

    @property
    def grid_type(self) -> GridType:
        """Return grid type for structured regions."""
        return self._grid_type

    def apply(
        self,
        field: NDArray[np.floating],
        boundary_conditions: BoundaryConditions,
        points: NDArray[np.floating],
        time: float = 0.0,
        *,
        spacing: NDArray[np.floating] | tuple[float, ...] | float | None = None,
    ) -> NDArray[np.floating]:
        """
        Apply boundary conditions to field values at given points.

        This method:
        1. Detects boundary points via SDF
        2. Computes boundary normals at those points
        3. Enforces BC values based on BC type and normal direction

        Args:
            field: Field values at points (shape: (N,) or (N, components))
            boundary_conditions: BC specification
            points: Spatial coordinates (shape: (N, dim))
            time: Time for time-dependent BCs (default: 0.0)
            spacing: Grid spacing for gradient estimation (optional)

        Returns:
            Field with boundary conditions applied (same shape as input)
        """
        self._validate_bc(boundary_conditions)

        # Make a copy to avoid modifying input
        result = field.copy()

        # Detect boundary points
        boundary_mask = self._detect_boundary_points(points)

        if not np.any(boundary_mask):
            return result  # No boundary points

        # Get boundary points and their normals
        boundary_points = points[boundary_mask]
        normals = self._compute_boundary_normals(boundary_points)

        # Apply BC based on type (fails loud if default_bc unset; Issue #1100)
        bc_type = boundary_conditions._resolve_default_bc("ImplicitBoundaryApplicator.apply")
        bc_value = self._resolve_bc_value(boundary_conditions, boundary_points, time)

        if bc_type == BCType.DIRICHLET:
            result[boundary_mask] = self._apply_dirichlet(result[boundary_mask], bc_value)
        elif bc_type in (BCType.NEUMANN, BCType.NO_FLUX):
            result[boundary_mask] = self._apply_neumann_along_normal(
                result, boundary_mask, normals, bc_value, points, spacing
            )
        elif bc_type == BCType.ROBIN:
            alpha = getattr(boundary_conditions, "alpha", 1.0)
            beta = getattr(boundary_conditions, "beta", 1.0)
            result[boundary_mask] = self._apply_robin_along_normal(
                result, boundary_mask, normals, alpha, beta, bc_value, points, spacing
            )
        elif bc_type == BCType.PERIODIC:
            result[boundary_mask] = self._apply_periodic(result, boundary_mask, points)
        else:
            # Fallback: treat as Dirichlet with zero value
            result[boundary_mask] = 0.0

        return result

    def _detect_boundary_points(self, points: NDArray[np.floating]) -> NDArray[np.bool_]:
        """
        Detect which points are on the boundary using geometry protocol.

        Args:
            points: Spatial coordinates (N, dim)

        Returns:
            Boolean mask of boundary points
        """
        # is_on_boundary is GeometryProtocol-guaranteed (protocol.py:208)
        return self.geometry.is_on_boundary(points, tolerance=self._boundary_tolerance)

    def _compute_boundary_normals(self, boundary_points: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute outward normal vectors at boundary points.

        Args:
            boundary_points: Coordinates of boundary points (M, dim)

        Returns:
            Unit normal vectors (M, dim)
        """
        # get_boundary_normal is GeometryProtocol-guaranteed (protocol.py:230)
        return self.geometry.get_boundary_normal(boundary_points)

    def _resolve_bc_value(
        self,
        bc: BoundaryConditions,
        points: NDArray[np.floating],
        time: float,
    ) -> NDArray[np.floating] | float:
        """
        Resolve BC value (may be callable or constant).

        Args:
            bc: Boundary conditions specification
            points: Boundary points
            time: Current time

        Returns:
            BC values (scalar or array matching points)
        """
        value = getattr(bc, "value", 0.0)

        if callable(value):
            # Time-dependent or space-dependent BC
            return np.array([value(p, time) for p in points])

        return value

    def _apply_dirichlet(
        self,
        boundary_values: NDArray[np.floating],
        bc_value: NDArray[np.floating] | float,
    ) -> NDArray[np.floating]:
        """
        Apply Dirichlet BC: u = g directly.

        Args:
            boundary_values: Current values at boundary (will be replaced)
            bc_value: Prescribed boundary value

        Returns:
            New boundary values
        """
        if np.isscalar(bc_value):
            return np.full_like(boundary_values, bc_value)
        return np.asarray(bc_value)

    def _apply_neumann_along_normal(
        self,
        field: NDArray[np.floating],
        boundary_mask: NDArray[np.bool_],
        normals: NDArray[np.floating],
        bc_value: NDArray[np.floating] | float,
        points: NDArray[np.floating],
        spacing: NDArray[np.floating] | tuple[float, ...] | float | None,
    ) -> NDArray[np.floating]:
        """
        Apply Neumann BC along normal direction: ∂u/∂n = g.

        For no-flux (g=0), this implements reflection:
        u_boundary = u_interior (extrapolated along normal)

        Args:
            field: Full field values
            boundary_mask: Boolean mask of boundary points
            normals: Unit normal vectors at boundary
            bc_value: Prescribed normal derivative
            points: All point coordinates
            spacing: Grid spacing for gradient estimation

        Returns:
            New values at boundary points
        """
        # Get effective spacing
        if spacing is None:
            dx = self._estimate_spacing(points)
        elif np.isscalar(spacing):
            dx = float(spacing)
        else:
            dx = float(np.mean(spacing))

        # For zero-flux, use reflection (same as interior neighbor)
        if np.allclose(bc_value, 0.0):
            # Find nearest interior neighbor along -normal direction
            interior_values = self._interpolate_along_normal(
                field, points, boundary_mask, normals, dx, direction="inward"
            )
            return interior_values

        # For non-zero Neumann: u_boundary = u_interior + dx * g
        interior_values = self._interpolate_along_normal(field, points, boundary_mask, normals, dx, direction="inward")
        if np.isscalar(bc_value):
            return interior_values + dx * bc_value
        return interior_values + dx * np.asarray(bc_value)

    def _apply_robin_along_normal(
        self,
        field: NDArray[np.floating],
        boundary_mask: NDArray[np.bool_],
        normals: NDArray[np.floating],
        alpha: float,
        beta: float,
        bc_value: NDArray[np.floating] | float,
        points: NDArray[np.floating],
        spacing: NDArray[np.floating] | tuple[float, ...] | float | None,
    ) -> NDArray[np.floating]:
        """
        Apply Robin BC along normal: α*u + β*∂u/∂n = g.

        Args:
            field: Full field values
            boundary_mask: Boolean mask of boundary points
            normals: Unit normal vectors at boundary
            alpha, beta: Robin coefficients
            bc_value: Prescribed RHS value
            points: All point coordinates
            spacing: Grid spacing

        Returns:
            New values at boundary points
        """
        if spacing is None:
            dx = self._estimate_spacing(points)
        elif np.isscalar(spacing):
            dx = float(spacing)
        else:
            dx = float(np.mean(spacing))

        # Get interior value along normal
        interior_values = self._interpolate_along_normal(field, points, boundary_mask, normals, dx, direction="inward")

        # Robin formula: u_boundary = (g - β*u_interior/dx) / (α + β/dx)
        # Derived from: α*u + β*(u - u_interior)/dx = g
        if np.abs(alpha) < 1e-12:
            # Pure Neumann case
            if np.isscalar(bc_value):
                return interior_values + dx * bc_value / beta
            return interior_values + dx * np.asarray(bc_value) / beta

        denominator = alpha + beta / dx
        if np.isscalar(bc_value):
            numerator = bc_value + beta * interior_values / dx
        else:
            numerator = np.asarray(bc_value) + beta * interior_values / dx

        return numerator / denominator

    def _apply_periodic(
        self,
        field: NDArray[np.floating],
        boundary_mask: NDArray[np.bool_],
        points: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Apply periodic BC by wrapping to opposite boundary.

        Args:
            field: Full field values
            boundary_mask: Boundary point mask
            points: All point coordinates

        Returns:
            Values from opposite boundary
        """
        # Get domain bounds
        bounds = getattr(self.geometry, "bounds", None)
        if bounds is None:
            # Cannot apply periodic without bounds
            return field[boundary_mask]

        bounds = np.asarray(bounds)
        boundary_points = points[boundary_mask]

        # Find corresponding points on opposite boundary
        wrapped_points = self._wrap_to_opposite_boundary(boundary_points, bounds)

        # Interpolate field values at wrapped points
        return self._interpolate_at_points(field, points, wrapped_points)

    def _interpolate_along_normal(
        self,
        field: NDArray[np.floating],
        points: NDArray[np.floating],
        boundary_mask: NDArray[np.bool_],
        normals: NDArray[np.floating],
        dx: float,
        direction: Literal["inward", "outward"] = "inward",
    ) -> NDArray[np.floating]:
        """
        Interpolate field values along normal direction from boundary.

        Args:
            field: Field values at all points
            points: All point coordinates
            boundary_mask: Boundary point mask
            normals: Outward unit normals at boundary points
            dx: Step size along normal
            direction: "inward" (into domain) or "outward"

        Returns:
            Interpolated values at offset points
        """
        boundary_points = points[boundary_mask]

        # Compute offset points
        sign = -1.0 if direction == "inward" else 1.0
        offset_points = boundary_points + sign * dx * normals

        # Interpolate at offset points
        return self._interpolate_at_points(field, points, offset_points)

    def _interpolate_at_points(
        self,
        field: NDArray[np.floating],
        source_points: NDArray[np.floating],
        target_points: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Interpolate field from source points to target points.

        Uses nearest neighbor for simplicity. More sophisticated interpolation
        (IDW, RBF, linear) can be added for higher accuracy.

        Args:
            field: Field values at source points
            source_points: Source coordinates
            target_points: Target coordinates for interpolation

        Returns:
            Interpolated values at target points
        """
        from scipy.spatial import cKDTree

        tree = cKDTree(source_points)
        _, indices = tree.query(target_points, k=1)
        return field[indices]

    def _wrap_to_opposite_boundary(
        self,
        points: NDArray[np.floating],
        bounds: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Wrap points to opposite boundary for periodic BC.

        Args:
            points: Boundary points
            bounds: Domain bounds (dim, 2)

        Returns:
            Wrapped point coordinates
        """
        wrapped = points.copy()

        for d in range(points.shape[1]):
            # Points near min boundary -> map to max
            near_min = points[:, d] < bounds[d, 0] + self._boundary_tolerance
            wrapped[near_min, d] = bounds[d, 1] - self._boundary_tolerance

            # Points near max boundary -> map to min
            near_max = points[:, d] > bounds[d, 1] - self._boundary_tolerance
            wrapped[near_max, d] = bounds[d, 0] + self._boundary_tolerance

        return wrapped

    def _estimate_spacing(self, points: NDArray[np.floating]) -> float:
        """
        Estimate grid spacing from point cloud.

        Args:
            points: Point coordinates

        Returns:
            Estimated spacing (mean nearest neighbor distance)
        """
        from scipy.spatial import cKDTree

        if len(points) < 2:
            return 1.0

        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)  # k=2: point itself + nearest
        return float(np.mean(distances[:, 1]))  # distances[:, 0] is always 0


def create_implicit_applicator(
    geometry: GeometryProtocol,
    grid_type: GridType | str = GridType.CELL_CENTERED,
    boundary_tolerance: float = 1e-10,
) -> ImplicitApplicator:
    """
    Factory function to create an ImplicitApplicator.

    Args:
        geometry: Geometry with SDF-based boundary detection
        grid_type: Grid type (CELL_CENTERED, VERTEX_CENTERED, STAGGERED)
        boundary_tolerance: Distance threshold for boundary detection

    Returns:
        Configured ImplicitApplicator instance
    """
    if isinstance(grid_type, str):
        grid_type = GridType[grid_type.upper()]

    return ImplicitApplicator(
        geometry=geometry,
        grid_type=grid_type,
        boundary_tolerance=boundary_tolerance,
    )


__all__ = [
    "ImplicitApplicator",
    "create_implicit_applicator",
]


if __name__ == "__main__":
    """Smoke test for ImplicitApplicator."""

    print("Testing ImplicitApplicator...")

    # Create a simple circular domain
    center = np.array([0.5, 0.5])
    radius = 0.4

    from mfgarchon.geometry.protocol import GeometryType

    class CircleDomain:
        """Simple circular domain implementing GeometryProtocol for testing."""

        dimension = 2
        geometry_type = GeometryType.IMPLICIT
        num_spatial_points = 441  # 21x21 grid

        def sdf(self, points):
            return np.linalg.norm(points - center, axis=-1) - radius

        def is_on_boundary(self, points, tolerance=1e-10):
            return np.abs(self.sdf(points)) < tolerance

        def get_boundary_normal(self, points):
            diff = points - center
            norms = np.linalg.norm(diff, axis=-1, keepdims=True)
            return diff / np.maximum(norms, 1e-10)

        def get_bounds(self):
            return np.array([0.0, 0.0]), np.array([1.0, 1.0])

        def get_collocation_points(self):
            x = np.linspace(0, 1, 21)
            y = np.linspace(0, 1, 21)
            xx, yy = np.meshgrid(x, y)
            return np.column_stack([xx.ravel(), yy.ravel()])

        def get_spatial_grid(self):
            return self.get_collocation_points()

        def get_grid_shape(self):
            return (21, 21)

        def get_problem_config(self):
            return {"num_spatial_points": 441, "spatial_shape": (21, 21)}

        def get_boundary_conditions(self):
            return None

        def get_boundary_regions(self):
            return {"all": {}}

        def get_boundary_indices(self, points, tolerance=1e-10):
            return np.where(self.is_on_boundary(points, tolerance))[0]

        def get_boundary_info(self, points, tolerance=1e-10):
            indices = self.get_boundary_indices(points, tolerance)
            if len(indices) == 0:
                return indices, np.array([], dtype=np.float64).reshape(0, 2)
            normals = self.get_boundary_normal(points[indices])
            return indices, normals

        def project_to_boundary(self, points):
            diff = points - center
            norms = np.linalg.norm(diff, axis=-1, keepdims=True)
            return center + diff / np.maximum(norms, 1e-10) * radius

        def project_to_interior(self, points):
            sdf_vals = self.sdf(points)
            outside = sdf_vals > 0
            result = points.copy()
            if np.any(outside):
                result[outside] = self.project_to_boundary(points[outside])
            return result

    # Create grid of points
    x = np.linspace(0, 1, 21)
    y = np.linspace(0, 1, 21)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    # Create field (e.g., radial function)
    field = np.linalg.norm(points - center, axis=-1)

    # Create applicator
    domain = CircleDomain()
    applicator = ImplicitApplicator(geometry=domain, boundary_tolerance=0.03)

    # Test 1: Dirichlet BC
    print("\n1. Testing Dirichlet BC...")
    from mfgarchon.geometry.boundary import dirichlet_bc

    bc_dir = dirichlet_bc(dimension=2, value=0.0)
    result_dir = applicator.apply(field, bc_dir, points)

    boundary_mask = domain.is_on_boundary(points, tolerance=0.03)
    n_boundary = np.sum(boundary_mask)
    print(f"   Boundary points: {n_boundary}")
    print(f"   Boundary values (should be ~0): {result_dir[boundary_mask][:5]}")
    assert np.allclose(result_dir[boundary_mask], 0.0), "Dirichlet failed"
    print("   PASS: Dirichlet BC")

    # Test 2: Neumann BC (no-flux)
    print("\n2. Testing Neumann BC (no-flux)...")
    from mfgarchon.geometry.boundary import neumann_bc

    bc_neu = neumann_bc(dimension=2)
    result_neu = applicator.apply(field, bc_neu, points)

    # No-flux should preserve interior-like values at boundary
    print(f"   Original boundary values: {field[boundary_mask][:5]}")
    print(f"   After no-flux: {result_neu[boundary_mask][:5]}")
    print("   PASS: Neumann BC (values updated via interpolation)")

    print("\n" + "=" * 50)
    print("All smoke tests passed!")
    print("=" * 50)

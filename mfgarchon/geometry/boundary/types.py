"""
Boundary condition types and segments (dimension-agnostic).

This module provides core type definitions for boundary conditions:
- BCType: Enum for boundary condition types
- BCSegment: Specification for a geometric boundary segment

These types are used by the BoundaryConditions class in conditions.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.geometry.boundary.tolerances import SDF_BOUNDARY_TOL
from mfgarchon.geometry.protocols import SupportsRegionMarking
from mfgarchon.utils.deprecation import deprecated

if TYPE_CHECKING:
    from collections.abc import Callable

    from .providers import BCValueProvider


@deprecated(
    since="v0.18.0",
    replacement="use mfgarchon.operators.differential.function_gradient() instead",
    reason="Issue #662: Consolidated 3 duplicate SDF gradient implementations",
)
def _compute_sdf_gradient(
    point: np.ndarray,
    sdf_func: Callable[[np.ndarray], float],
    epsilon: float = 1e-5,
) -> np.ndarray:
    """
    Compute SDF gradient using central finite differences.

    .. deprecated::
        Use ``mfgarchon.operators.differential.function_gradient`` instead.
        This function delegates to the canonical implementation in operators/.

    Args:
        point: Point at which to evaluate gradient (1D array)
        sdf_func: SDF function
        epsilon: Finite difference step size

    Returns:
        Gradient vector (same shape as point)

    Note:
        Issue #662: Consolidated into operators/differential/function_gradient.py
    """
    # Import canonical implementation (Issue #662)
    from mfgarchon.operators.differential.function_gradient import function_gradient

    # Validate input
    point = np.asarray(point, dtype=float)
    if not np.all(np.isfinite(point)):
        raise ValueError(f"SDF gradient: Point contains non-finite values: {point}")

    # Use canonical implementation with adaptive epsilon
    grad = function_gradient(sdf_func, point, eps=epsilon, adaptive_eps=True)

    # Warn if gradient is suspiciously small (preserve original behavior)
    grad_norm = np.linalg.norm(grad)
    if grad_norm < 1e-10:
        import warnings

        warnings.warn(
            f"SDF gradient has very small magnitude {grad_norm:.2e} at point {point}. "
            f"This may indicate a degenerate SDF or a point far from the boundary.",
            RuntimeWarning,
            stacklevel=2,
        )

    return grad


class BCType(Enum):
    """
    Boundary condition types (dimension-agnostic).

    Standard BC Types:
        DIRICHLET: Fixed value at boundary (u = g)
        NEUMANN: Fixed normal derivative at boundary (du/dn = g)
        ROBIN: Mixed condition (alpha*u + beta*du/dn = g)
        PERIODIC: Wrap-around boundaries (u(x_min) = u(x_max))

    Impermeable Boundary Types (NO_FLUX and REFLECTING):
        These represent the SAME physical concept - an impermeable wall where
        no mass/probability/particles can cross - but are named differently
        based on the discretization context:

        NO_FLUX: Used for field-based methods (FDM, FEM, GFDM)
            - Enforces zero probability flux: J dot n = 0
            - For pure diffusion: du/dn = 0 (zero Neumann)
            - Implementation: ghost cells via np.pad(mode="edge")

        REFLECTING: Used for particle-based methods (Lagrangian FP)
            - Enforces elastic reflection at boundary
            - Particles bounce back into domain
            - Implementation: modular fold reflection algorithm

        Mathematically equivalent for mass conservation. Choose based on solver:
        - FP density equation (field) -> NO_FLUX
        - Particle simulation -> REFLECTING
        - HJB value function -> NEUMANN (with g=0)

    Extrapolation Types (for unbounded/truncated domains):
        EXTRAPOLATION_LINEAR: Zero second derivative (d²u/dx² = 0)
            - Use for HJB value functions with linear growth
            - Ghost = 2*u_0 - u_1

        EXTRAPOLATION_QUADRATIC: Zero third derivative (d³u/dx³ = 0)
            - Use for LQG-type problems with quadratic value growth
            - Ghost = 3*u_0 - 3*u_1 + u_2
    """

    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ROBIN = "robin"
    PERIODIC = "periodic"
    REFLECTING = "reflecting"
    NO_FLUX = "no_flux"
    EXTRAPOLATION_LINEAR = "extrapolation_linear"
    EXTRAPOLATION_QUADRATIC = "extrapolation_quadratic"


class BoundaryEntity:
    """Dimension-agnostic boundary entity identifier for any codimension.

    Represents a boundary entity on a rectangular domain by specifying
    which axes are constrained and to which side:

    - codim-1 (face): one axis constrained  (e.g., x_min plane in 3D)
    - codim-2 (edge): two axes constrained  (e.g., x_min AND y_max edge)
    - codim-d (vertex): all d axes constrained (corner point)

    Use the ``BoundaryFace`` subclass for the common codim-1 case,
    which provides convenient ``.axis`` and ``.side`` scalar properties.

    Parameters
    ----------
    axes : tuple[int, ...]
        Dimension indices that define the entity.
    sides : tuple[str, ...]
        "min" or "max" for each constrained axis.

    Examples
    --------
    >>> BoundaryEntity((0,), ("min",))                    # codim-1 face
    >>> BoundaryEntity((0, 1), ("min", "max"))            # codim-2 edge
    >>> BoundaryEntity((0, 1, 2), ("min", "min", "max"))  # codim-3 vertex
    """

    __slots__ = ("axes", "sides")

    def __init__(self, axes: tuple[int, ...], sides: tuple[str, ...]):
        if len(axes) != len(sides):
            msg = f"axes and sides must have same length, got {len(axes)} and {len(sides)}"
            raise ValueError(msg)
        for s in sides:
            if s not in ("min", "max"):
                msg = f"Each side must be 'min' or 'max', got '{s}'"
                raise ValueError(msg)
        self.axes = axes
        self.sides = sides

    @property
    def codim(self) -> int:
        """Codimension of this boundary entity."""
        return len(self.axes)

    @property
    def is_face(self) -> bool:
        """True if this is a codim-1 entity (face)."""
        return self.codim == 1

    def __repr__(self) -> str:
        if self.is_face:
            return f"BoundaryFace(axis={self.axes[0]}, side='{self.sides[0]}')"
        return f"BoundaryEntity(axes={self.axes}, sides={self.sides})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BoundaryEntity):
            return self.axes == other.axes and self.sides == other.sides
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.axes, self.sides))

    def to_string(self, axis_names: dict[int, str] | None = None) -> str:
        """Convert to legacy string format.

        >>> BoundaryEntity((0,), ("min",)).to_string()           # "x_min"
        >>> BoundaryEntity((0, 1), ("min", "max")).to_string()   # "x_min_y_max"
        """
        if axis_names is None:
            axis_names = {0: "x", 1: "y", 2: "z", 3: "w"}
        parts = []
        for ax, sd in zip(self.axes, self.sides, strict=True):
            name = axis_names.get(ax, f"axis{ax}")
            parts.append(f"{name}_{sd}")
        return "_".join(parts)


class BoundaryFace(BoundaryEntity):
    """Codimension-1 boundary face (the common case).

    Convenience subclass of BoundaryEntity for faces. Provides scalar
    ``.axis`` and ``.side`` properties instead of tuple access.

    Parameters
    ----------
    axis : int
        Dimension index (0 = first spatial dim, 1 = second, etc.)
    side : str
        "min" or "max" — which end of the axis.

    Examples
    --------
    >>> BoundaryFace(0, "min")   # left boundary in 1D, x_min in 2D
    >>> BoundaryFace(1, "max")   # y_max in 2D
    >>> BoundaryFace(3, "min")   # 4th dimension lower bound
    >>> isinstance(BoundaryFace(0, "min"), BoundaryEntity)  # True
    """

    def __init__(self, axis: int, side: str):
        super().__init__((axis,), (side,))

    @property
    def axis(self) -> int:
        """The constrained axis index."""
        return self.axes[0]

    @property
    def side(self) -> str:
        """The side: 'min' or 'max'."""
        return self.sides[0]


# String aliases for backward compatibility
_BOUNDARY_STRING_TO_FACE: dict[str, BoundaryFace] = {
    # 1D
    "left": BoundaryFace(0, "min"),
    "right": BoundaryFace(0, "max"),
    "x_min": BoundaryFace(0, "min"),
    "x_max": BoundaryFace(0, "max"),
    # 2D
    "bottom": BoundaryFace(1, "min"),
    "top": BoundaryFace(1, "max"),
    "y_min": BoundaryFace(1, "min"),
    "y_max": BoundaryFace(1, "max"),
    # 3D
    "front": BoundaryFace(2, "min"),
    "back": BoundaryFace(2, "max"),
    "z_min": BoundaryFace(2, "min"),
    "z_max": BoundaryFace(2, "max"),
    # 4D
    "w_min": BoundaryFace(3, "min"),
    "w_max": BoundaryFace(3, "max"),
}


def parse_boundary_face(boundary: str | int | BoundaryFace | None) -> BoundaryFace | None:
    """Normalize any boundary identifier to BoundaryFace.

    Accepts:
    - BoundaryFace: pass through
    - str: look up in alias table, or parse "axis{N}_{side}" format
    - int: Gmsh physical group tag (returned as-is, not a BoundaryFace)
    - None: return None

    >>> parse_boundary_face("x_min")        # BoundaryFace(axis=0, side='min')
    >>> parse_boundary_face("left")         # BoundaryFace(axis=0, side='min')
    >>> parse_boundary_face("axis5_max")    # BoundaryFace(axis=5, side='max')
    >>> parse_boundary_face(BoundaryFace(0, "min"))  # pass through
    """
    if boundary is None:
        return None
    if isinstance(boundary, BoundaryFace):
        return boundary
    if isinstance(boundary, int):
        return None  # Gmsh tag, not a face identifier
    if isinstance(boundary, str):
        # Check alias table first
        if boundary in _BOUNDARY_STRING_TO_FACE:
            return _BOUNDARY_STRING_TO_FACE[boundary]
        # Try "axis{N}_{side}" format
        if boundary.startswith("axis") and "_" in boundary:
            parts = boundary.split("_", 1)
            try:
                axis = int(parts[0][4:])  # "axis5" -> 5
                side = parts[1]
                if side in ("min", "max"):
                    return BoundaryFace(axis, side)
            except (ValueError, IndexError):
                pass
        # Try "{letter}_{side}" for unknown letters
        if len(boundary) > 2 and boundary[-4:] in ("_min", "_max"):
            side = boundary[-3:]
            return BoundaryFace(0, side)  # Default to axis 0 for unknown
    return None


@dataclass
class BCSegment:
    """
    Boundary condition specification for a geometric segment (dimension-agnostic).

    Defines a boundary condition that applies to a specific region of the domain
    boundary. Works for any spatial dimension using multiple matching methods:

    **Rectangular domains:**
    - `boundary`: String identifier for boundary location (e.g., "left", "right")
    - `region`: Dict mapping axis names/indices to coordinate ranges

    **General domains (SDF-based):**
    - `sdf_region`: Callable SDF defining the segment region (negative = inside)
    - `normal_direction`: Match by outward normal direction
    - `normal_tolerance`: Cosine threshold for normal matching

    **Marked regions** (Issue #596 Phase 2.5):
    - `region_name`: References regions marked via geometry.mark_region()
    - Enables semantic naming and complex region shapes

    Attributes:
        name: Human-readable identifier for this segment
        bc_type: Type of boundary condition (Dirichlet, Neumann, etc.)
        value: BC value (constant or function of (x, t))
        boundary: Boundary location identifier (e.g., "left", "right", "x_min", "x_max")
        region: Axis-specific ranges defining segment extent
        sdf_region: SDF function defining segment region (phi < 0 means in segment)
        normal_direction: Target outward normal for matching (unit vector)
        normal_tolerance: Cosine of max angle between actual and target normal (0.7 = ~45 deg)
        region_name: Name of region marked via geometry.mark_region() (Issue #596 Phase 2.5)
        priority: Priority for overlapping segments (higher overrides lower)

    Examples:
        1D segment (rectangular):
        >>> bc = BCSegment(name="left_wall", bc_type=BCType.DIRICHLET, value=0.0, boundary="left")

        2D exit segment on rectangular domain (right wall, y in [4.25, 5.75]):
        >>> exit_bc = BCSegment(
        ...     name="exit",
        ...     bc_type=BCType.DIRICHLET,
        ...     value=0.0,
        ...     boundary="right",
        ...     region={"y": (4.25, 5.75)}
        ... )

        2D exit segment on circular domain (top, normal pointing up):
        >>> exit_bc = BCSegment(
        ...     name="exit",
        ...     bc_type=BCType.DIRICHLET,
        ...     value=0.0,
        ...     normal_direction=np.array([0.0, 1.0]),
        ...     normal_tolerance=0.7,  # cos(45 deg)
        ... )

        SDF-based region (small ball around a point):
        >>> corner_bc = BCSegment(
        ...     name="corner",
        ...     bc_type=BCType.NEUMANN,
        ...     value=0.0,
        ...     sdf_region=lambda x: np.linalg.norm(x - np.array([1, 1])) - 0.2,
        ...     priority=2,
        ... )

        Marked region (Issue #596 Phase 2.5):
        >>> # First mark region on geometry
        >>> geometry.mark_region("inlet", predicate=lambda x: x[:, 0] < 0.1)
        >>> # Then reference it in BC segment
        >>> inlet_bc = BCSegment(
        ...     name="inlet_bc",
        ...     bc_type=BCType.DIRICHLET,
        ...     value=1.0,
        ...     region_name="inlet",
        ... )
    """

    name: str
    bc_type: BCType
    # BC value: static float, callable f(x,t), or BCValueProvider (Issue #625)
    # Providers are resolved by FixedPointIterator with current iteration state
    value: float | Callable | BCValueProvider = 0.0

    # Robin BC coefficients: alpha*u + beta*du/dn = g
    alpha: float = 1.0  # Weight on u (Dirichlet term)
    beta: float = 0.0  # Weight on du/dn (Neumann term)

    # Boundary matching: str for named boundaries ("x_min", "left"),
    # int for Gmsh physical group tags (Issue #732 Tier 1)
    boundary: str | int | None = None
    region: dict[str | int, tuple[float, float]] | None = None

    # SDF-based matching (general domains)
    sdf_region: Callable[[np.ndarray], float] | None = None
    normal_direction: np.ndarray | None = None
    normal_tolerance: float = 0.7  # cos(~45 deg)

    # Marked region matching (Issue #596 Phase 2.5)
    # References regions marked via geometry.mark_region(name, ...)
    region_name: str | None = None

    priority: int = 0

    # Flux-limited absorption (for DIRICHLET exits)
    # Units: mass/time for density methods, particles/time for Lagrangian methods
    # None = unlimited (instant absorption)
    flux_capacity: float | None = None

    @property
    def face(self) -> BoundaryFace | None:
        """Dimension-agnostic boundary face identifier.

        Parses the string `boundary` field into a structured BoundaryFace.
        Returns None if boundary is not a rectangular face (e.g., SDF region).

        >>> seg = BCSegment(name="left", bc_type=BCType.NEUMANN, boundary="x_min")
        >>> seg.face  # BoundaryFace(axis=0, side='min')
        >>> seg.face.axis  # 0
        >>> seg.face.side  # 'min'
        """
        return parse_boundary_face(self.boundary)

    def __post_init__(self) -> None:
        """Validate BCSegment specification (Issue #596 Phase 2.5)."""
        # Issue #612: Fix validation - boundary+region are complementary filters, not exclusive
        # Valid combinations:
        #   - boundary alone (entire boundary wall)
        #   - region alone (any boundary with coords in range)
        #   - boundary + region (specific part of specific wall) ✅
        #   - sdf_region alone or sdf_region + normal_direction (SDF-based matching)
        #   - region_name alone (marked regions)

        has_boundary = self.boundary is not None
        has_region = self.region is not None
        has_sdf_region = self.sdf_region is not None
        has_normal = self.normal_direction is not None
        has_region_name = self.region_name is not None

        # Rectangular domain methods: boundary, region (can combine)
        rectangular_methods = has_boundary or has_region

        # SDF domain methods: sdf_region, normal_direction (can combine)
        sdf_methods = has_sdf_region or has_normal

        # Marked region method: region_name (exclusive)
        marked_region = has_region_name

        # Check for invalid combinations across method categories
        active_categories = sum([rectangular_methods, sdf_methods, marked_region])

        if active_categories > 1:
            # Conflicting method categories
            active_specs = []
            if rectangular_methods:
                specs = []
                if has_boundary:
                    specs.append("boundary")
                if has_region:
                    specs.append("region")
                active_specs.append(f"rectangular ({', '.join(specs)})")
            if sdf_methods:
                specs = []
                if has_sdf_region:
                    specs.append("sdf_region")
                if has_normal:
                    specs.append("normal_direction")
                active_specs.append(f"SDF ({', '.join(specs)})")
            if marked_region:
                active_specs.append("region_name")

            raise ValueError(
                f"BCSegment '{self.name}': Cannot mix region specification methods from different categories. "
                f"Got {len(active_specs)} categories: {', '.join(active_specs)}. "
                f"Use either: (1) rectangular domain (boundary, region), "
                f"(2) SDF domain (sdf_region, normal_direction), or (3) marked regions (region_name)."
            )

    def matches_point(
        self,
        point: np.ndarray,
        boundary_id: str | None,
        domain_bounds: np.ndarray | None,
        tolerance: float = SDF_BOUNDARY_TOL,
        axis_names: dict[int, str] | None = None,
        domain_sdf: Callable[[np.ndarray], float] | None = None,
        geometry: SupportsRegionMarking | None = None,
    ) -> bool:
        """
        Check if this BC segment applies to a given boundary point.

        Supports multiple matching modes:
        1. Boundary ID matching (rectangular domains)
        2. Coordinate range matching (rectangular domains)
        3. SDF region matching (general domains)
        4. Normal direction matching (general domains)
        5. Region name matching (Issue #596 Phase 2.5)

        Args:
            point: Spatial coordinates as 1D array of shape (dimension,)
            boundary_id: Boundary identifier (e.g., "left", "right", "x_min", "top")
                        Can be None for SDF-based domains.
            domain_bounds: Domain bounds as array of shape (dimension, 2)
                          bounds[i, 0] = min, bounds[i, 1] = max for axis i
                          Can be None for SDF-based domains.
            tolerance: Tolerance for geometric comparisons
            axis_names: Optional mapping from axis index to name (e.g., {0: "x", 1: "y"})
            domain_sdf: Optional domain SDF for normal-based matching
            geometry: Geometry object with marked regions (Issue #596 Phase 2.5).
                     Required if this segment uses region_name.

        Returns:
            True if this BC segment applies to the given point

        Raises:
            ValueError: If point is empty, domain_bounds are invalid, or geometry is
                       missing when region_name is used
        """
        # Input validation
        point = np.asarray(point, dtype=float)
        if point.size == 0:
            raise ValueError(f"BCSegment '{self.name}': Point array cannot be empty")

        if domain_bounds is not None:
            domain_bounds = np.asarray(domain_bounds, dtype=float)
            if domain_bounds.ndim != 2 or domain_bounds.shape[1] != 2:
                raise ValueError(
                    f"BCSegment '{self.name}': domain_bounds must have shape (dimension, 2), got {domain_bounds.shape}"
                )
            if not (domain_bounds[:, 0] <= domain_bounds[:, 1]).all():
                raise ValueError(
                    f"BCSegment '{self.name}': domain_bounds must have min <= max for all axes. "
                    f"Got bounds where min > max: {domain_bounds}"
                )
            if point.shape[0] != domain_bounds.shape[0]:
                raise ValueError(
                    f"BCSegment '{self.name}': Point dimension {point.shape[0]} "
                    f"does not match domain_bounds dimension {domain_bounds.shape[0]}"
                )

        # Method 1: Check boundary identifier match (rectangular domains)
        if self.boundary is not None and self.boundary != "all":
            if boundary_id is None or self.boundary != boundary_id:
                return False

        # Method 2: Check coordinate range constraints (rectangular domains)
        if self.region is not None:
            if axis_names is None:
                # Default axis names: x, y, z, w, ...
                axis_names = {0: "x", 1: "y", 2: "z", 3: "w"}

            for axis_key, (range_min, range_max) in self.region.items():
                # Convert axis key to index
                if isinstance(axis_key, str):
                    # Find index from axis name
                    axis_idx = next((idx for idx, name in axis_names.items() if name == axis_key), None)
                    if axis_idx is None:
                        raise ValueError(f"Unknown axis name: {axis_key}")
                else:
                    axis_idx = int(axis_key)

                # Check if point coordinate is within range
                if axis_idx >= len(point):
                    raise ValueError(f"Axis index {axis_idx} exceeds point dimension {len(point)}")

                coord = point[axis_idx]
                if coord < range_min - tolerance or coord > range_max + tolerance:
                    return False

        # Method 3: Check SDF region constraint (general domains)
        if self.sdf_region is not None:
            sdf_val = self.sdf_region(point)
            if sdf_val > tolerance:
                return False

        # Method 4: Check normal direction match (general domains)
        if self.normal_direction is not None and domain_sdf is not None:
            # Compute outward normal via SDF gradient (finite differences)
            normal = _compute_sdf_gradient(point, domain_sdf, epsilon=1e-5)
            normal_norm = np.linalg.norm(normal)
            if normal_norm > 1e-12:
                normal = normal / normal_norm

                # Normalize target direction
                target = np.asarray(self.normal_direction, dtype=float)
                target = target / (np.linalg.norm(target) + 1e-12)

                # Check cosine similarity
                cos_angle = np.dot(normal, target)
                if cos_angle < self.normal_tolerance:
                    return False

        # Method 5: Check region name match (Issue #596 Phase 2.5)
        if self.region_name is not None:
            # Validate geometry is provided
            if geometry is None:
                raise ValueError(
                    f"BCSegment '{self.name}' uses region_name='{self.region_name}' "
                    f"but no geometry provided to matches_point(). "
                    f"Region-based BCs require geometry parameter."
                )

            # Validate geometry supports region marking
            if not isinstance(geometry, SupportsRegionMarking):
                raise TypeError(
                    f"BCSegment '{self.name}' uses region_name but geometry "
                    f"{type(geometry).__name__} doesn't implement SupportsRegionMarking"
                )

            # Get region mask from geometry
            try:
                region_mask = geometry.get_region_mask(self.region_name)
            except (KeyError, ValueError) as e:
                raise ValueError(
                    f"BCSegment '{self.name}': Failed to get region mask for '{self.region_name}'. Error: {e}"
                ) from e

            # Convert point to grid indices
            # Assumption: point is in physical coordinates, need to map to grid indices
            # This requires the geometry to provide a method to map coordinates to indices
            # For now, we'll check if the point's grid index is in the region mask
            # This is a simplified implementation - may need refinement for general geometries

            # Try to get grid indices from geometry
            # Use getattr pattern per CLAUDE.md (no hasattr for duck typing)
            point_to_indices_method = getattr(geometry, "point_to_indices", None)
            if callable(point_to_indices_method):
                try:
                    indices = point_to_indices_method(point)
                    # Check if indices are in region (region_mask is boolean array)
                    in_region = region_mask[tuple(indices)]
                    if not in_region:
                        return False
                except (IndexError, AttributeError, TypeError):
                    # Fallback: point not mappable to grid indices
                    # This could happen for points outside domain or at boundaries
                    # Conservative approach: assume point is NOT in region
                    return False
            else:
                # Geometry doesn't support point_to_indices
                # For TensorProductGrid, we can compute indices manually
                # Use getattr pattern per CLAUDE.md
                bounds = getattr(geometry, "bounds", None)
                Nx_points = getattr(geometry, "Nx_points", None)
                if bounds is not None and Nx_points is not None:
                    try:
                        # Manual index computation for structured grids
                        indices = []
                        for dim_idx in range(len(point)):
                            x_min, x_max = bounds[dim_idx]
                            # Compute grid index (with clamping to handle boundary points)
                            grid_idx = int((point[dim_idx] - x_min) / (x_max - x_min) * (Nx_points[dim_idx] - 1))
                            grid_idx = max(0, min(grid_idx, Nx_points[dim_idx] - 1))
                            indices.append(grid_idx)

                        # Convert multi-dimensional indices to flat index
                        # Region masks are flattened 1D arrays
                        flat_idx = np.ravel_multi_index(indices, Nx_points)

                        # Check region mask
                        in_region = region_mask[flat_idx]
                        if not in_region:
                            return False
                    except (IndexError, AttributeError, TypeError, KeyError):
                        # Fallback: assume NOT in region
                        return False
                else:
                    raise NotImplementedError(
                        f"BCSegment '{self.name}': Geometry {type(geometry).__name__} doesn't support "
                        f"point-to-index mapping required for region_name matching. "
                        f"Geometry must implement either point_to_indices() method or provide bounds/Nx_points attributes."
                    )

        return True

    def get_value(
        self,
        point: np.ndarray,
        t: float = 0.0,
        state: dict[str, Any] | None = None,
    ) -> float:
        """
        Evaluate BC value at a point.

        Args:
            point: Spatial coordinates as 1D array
            t: Time
            state: Optional iteration state dict for BCValueProvider resolution
                   (Issue #625). Required if value is a provider. Standard keys:
                   'm_current', 'U_current', 'geometry', 'sigma', 'iteration'.

        Returns:
            BC value at this point and time

        Note:
            Value resolution order:
            1. BCValueProvider: resolved via provider.compute(state)
            2. Callable: tries multiple signature patterns (see below)
            3. Float: returned directly

            For callable values, tries multiple signature patterns:
            1. value(point, t) - generic array interface
            2. value(*point, t) - coordinate expansion
            3. value(point) - no time dependency
            4. Falls back to 0.0 if all fail

        Raises:
            ValueError: If value is a provider but state is not provided
        """
        # Check for BCValueProvider first (Issue #625)
        # Late import: real cycle — types -> providers -> bc_coupling -> types
        from .providers import is_provider

        if is_provider(self.value):
            if state is None:
                raise ValueError(
                    f"BCSegment '{self.name}' has a BCValueProvider value but no state "
                    f"was provided to get_value(). Providers require iteration state "
                    f"for resolution. Either pass state dict or use resolve_provider() "
                    f"in the coupling iterator."
                )
            result = self.value.compute(state)
            return float(result) if np.isscalar(result) else result

        if callable(self.value):
            # Strategy 1: Try generic array interface first (safest)
            try:
                result = self.value(point, t)
                return float(result)
            except (TypeError, ValueError):
                pass

            # Strategy 2: Try coordinate expansion (*point, t)
            # Works for value(x, t), value(x, y, t), value(x, y, z, t), etc.
            try:
                coords = (*tuple(point), t)
                result = self.value(*coords)
                return float(result)
            except (TypeError, ValueError):
                pass

            # Strategy 3: Try without time (stationary BC)
            try:
                result = self.value(point)
                return float(result)
            except (TypeError, ValueError):
                pass

            # Strategy 4: Try coordinate expansion without time
            try:
                result = self.value(*point)
                return float(result)
            except (TypeError, ValueError):
                pass

            # If all strategies fail, return zero and warn
            import warnings

            warnings.warn(
                f"BCSegment '{self.name}': Could not evaluate callable value. "
                f"Tried signatures: value(point, t), value(*point, t), value(point), value(*point). "
                f"Returning 0.0 as fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0.0

        return float(self.value)

    def __str__(self) -> str:
        """String representation."""
        parts = [f"{self.name}: {self.bc_type.value}"]

        if self.boundary is not None:
            parts.append(f"on {self.boundary}")

        if self.region is not None:
            region_strs = [f"{axis}in{rng}" for axis, rng in self.region.items()]
            parts.append(f"where {', '.join(region_strs)}")

        if self.sdf_region is not None:
            parts.append("[sdf_region]")

        if self.normal_direction is not None:
            dir_str = ",".join(f"{x:.2f}" for x in self.normal_direction)
            parts.append(f"[normal=({dir_str})]")

        if self.priority > 0:
            parts.append(f"(priority={self.priority})")

        return " ".join(parts)


def create_standard_boundary_names(dimension: int) -> dict[str, str]:
    """
    Create standard boundary name mappings for rectangular domains.

    Args:
        dimension: Spatial dimension

    Returns:
        Dictionary mapping standard names to axis-specific names
    """
    if dimension == 1:
        return {"left": "x_min", "right": "x_max"}
    elif dimension == 2:
        return {
            "left": "x_min",
            "right": "x_max",
            "bottom": "y_min",
            "top": "y_max",
        }
    elif dimension == 3:
        return {
            "left": "x_min",
            "right": "x_max",
            "bottom": "y_min",
            "top": "y_max",
            "front": "z_min",
            "back": "z_max",
        }
    else:
        # nD - use generic names (x, y, z for first 3, then dim3, dim4, ...)
        names = {}
        for i in range(dimension):
            if i < 3:
                axis = ["x", "y", "z"][i]
            else:
                axis = f"dim{i}"
            names[f"{axis}_min"] = f"{axis}_min"
            names[f"{axis}_max"] = f"{axis}_max"
        return names

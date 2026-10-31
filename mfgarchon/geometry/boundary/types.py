"""
Boundary condition types and segments (dimension-agnostic).

This module provides core type definitions for boundary conditions:
- BCType: Enum for boundary condition types
- BCSegment: Specification for a geometric boundary segment

These types are used by the BoundaryConditions class in conditions.py.
"""

from __future__ import annotations

import functools
import inspect
import numbers
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.geometry.boundary.providers import is_provider
from mfgarchon.geometry.boundary.tolerances import SDF_BOUNDARY_TOL
from mfgarchon.types.callable_protocols import BOUNDARY_VALUE_SLOTS, bind_user_callable
from mfgarchon.utils.deprecation import deprecated

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.geometry.protocols import SupportsRegionMarking
    from mfgarchon.types.callable_protocols import BoundCallable

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


class PeriodicGridConvention(Enum):
    """Which periodic grid a wrap-around operation is acting on. Issue #1822.

    ``BCType.PERIODIC`` says the domain wraps. It does NOT say where the last node sits, and the
    two answers give different neighbours for the same index:

    ``ENDPOINT_INCLUSIVE``
        ``np.linspace(x_min, x_max, N)`` -- what ``TensorProductGrid`` builds. ``x[0]`` and
        ``x[-1]`` are the same physical point counted twice, so the period is ``L``, the spacing
        is ``L / (N - 1)``, and the node one step left of ``x[0]`` is ``x[-2]``.

    ``ENDPOINT_EXCLUSIVE``
        ``np.linspace(x_min, x_max, N, endpoint=False)`` -- what the operator layer
        (``LaplacianOperator``, ``IsotropicDiffusion``) builds. ``N`` distinct nodes, spacing
        ``L / N``, and the node one step left of ``x[0]`` is ``x[-1]``.

    Both are live in this package, they differ by exactly one node, and a wrap that assumes the
    wrong one shifts every periodic stencil by a cell while still returning a finite, plausible
    field. Measured both ways on ``Nx=21``: wrapping an inclusive grid as exclusive gave
    ``HJBWENOSolver`` a periodic seam of 2.63e-01 from exactly periodic input and stalled
    ``ImplicitHeatSolver``'s seam at 1.5e-02 under refinement (#1825); wrapping an exclusive grid
    as inclusive took the operator layer's Laplacian from 1.3e-02 to 6.3e+02 of error. Neither
    raises, which is why this is carried rather than guessed at each site.

    **How it gets decided.** The choice rides on the periodic ``BoundaryConditions``, since a wrap
    happens only because that object says PERIODIC. Left unstated (``None``) it means
    ``ENDPOINT_EXCLUSIVE`` -- how this package wrapped before #1822 -- so adding it changed no
    existing caller's numbers. ``TensorProductGrid`` binds the convention it measures from its own
    coordinates onto any periodic BC attached to it, the way it binds ``dimension``, and refuses
    one that contradicts them. So solver paths are correct without declaring anything, and a
    caller with no grid (the operator layer) states it or keeps the historical layout.
    """

    ENDPOINT_INCLUSIVE = "endpoint_inclusive"
    ENDPOINT_EXCLUSIVE = "endpoint_exclusive"


def repeated_endpoint_count(convention: PeriodicGridConvention | None) -> int:
    """How many trailing nodes of a periodic axis repeat a node the array already holds.

    One under ``ENDPOINT_INCLUSIVE`` (``x[-1]`` *is* ``x[0]``), zero otherwise; unstated means the
    exclusive layout this package wrapped with before #1822.

    Every periodic wrap in the package is one of two expressions of this single number, which is
    why it is a function and not two constants: a ghost fill steps over the repeat
    (``skip``), and a modular neighbour index divides by the count of distinct cells
    (``span = N - repeated_endpoint_count(...)``). Written out separately at each site they drift,
    and the drift is silent -- both readings return a finite, plausible field.
    """
    return 1 if convention is PeriodicGridConvention.ENDPOINT_INCLUSIVE else 0


class BCType(Enum):
    """
    Boundary condition types (dimension-agnostic).

    Standard BC Types:
        DIRICHLET: Fixed value at boundary (u = g)
        NEUMANN: Fixed normal derivative at boundary (du/dn = g). On a BC the HJB and FP share, g is
            the HJB's datum. The FP reads g = 0 as zero flux (NO_FLUX) and refuses a nonzero g, which says
            nothing about the agents' mass at the wall; the two equations' BCs are then specified separately.
            An FP solver refuses a NEUMANN handed to it explicitly, which would mean dm/dn = g (#2512, row B3).
        ROBIN: Mixed condition (alpha*u + beta*du/dn = g)
        PERIODIC: Wrap-around boundaries (u(x_min) = u(x_max))
        FLUX: A prescribed total flux, J.n = q with J_i = alpha_i*m - d_j(A_ij*m), for a Fokker-Planck solver's own BC
            (#2581). q is mass per unit time per unit boundary measure, outward positive, and q = 0 is NO_FLUX. It
            is not NEUMANN: J.n = q and -D*dm/dn = q differ by (v.n - dD/dn)*m, so they part wherever
            v.n != dD/dn. FP-side only, so it is
            refused on a BC the HJB and FP share, and a step is refused rather than clipped past the gate where the
            clip would fabricate mass the flux decides (B-3, `refuse_what_a_flux_takes_negative`; any other step
            whose own clip is past the gate is the scheme's, flux included) (#2581, user rulings 2026-10-09 and
            2026-10-30).

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
    FLUX = "flux"
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
        # Numeric-axis formats. Both spellings exist in the tree and mean the same thing:
        # `BoundaryEntity.to_string` emits "axis{N}_{side}" for axes past w, and `dim{N}_{side}` is
        # emitted by `flux_diagnostics._get_boundary_names`, by `create_standard_boundary_names`
        # below for axes past z, and by the `get_boundary_regions` tables of `geometry/base.py`,
        # `graph/network_geometry.py` and `implicit/point_cloud.py` for axes past v. Two other
        # emitters were deleted when their readers began asking the face owner:
        # `_compat._get_boundary_name_nd` (#2512 row B1, #2628) and
        # `applicator_particle._get_boundary_id` (row B1, stage B1d). Accepting only the first
        # spelling meant a segment declared on "dim3_min" resolved to axis 0 -- see below.
        #
        # Two further spellings exist and are deliberately NOT accepted: those three tables emit
        # "v_min" for axis 4, and `adjoint/operators.py::_get_axis_name` emits "x_1_min" for
        # dim > 3. Both resolved to axis 0 before and to None now, and neither reaches this function
        # -- they are dict keys. `tensor_grid.py` carries a fifth, independent regex resolver that
        # `mark_region` uses instead of this one. So "one resolver" is true of the path this
        # function serves and not of the tree; the rest is #1936's ground.
        for prefix in ("axis", "dim"):
            if boundary.startswith(prefix) and "_" in boundary:
                head, _, side = boundary.partition("_")
                if side in ("min", "max"):
                    try:
                        return BoundaryFace(int(head[len(prefix) :]), side)
                    except ValueError:
                        pass
        # ~~"{letter}_{side}" for unknown letters -> BoundaryFace(0, side)~~ [REMOVED 2026-08-15]
        # An unrecognised prefix now returns None rather than guessing axis 0. That branch turned a
        # naming mismatch into a *dropped boundary condition*: "dim3_min" became axis 0, so a
        # Dirichlet exit declared on the fourth axis was applied to the first, while the axis it was
        # written for silently took the default. "inlet_min" resolved the same way. Returning None
        # lets the caller fall through to its declared default, which is a wrong answer the caller
        # asked for rather than one this function invented. #1939
    return None


def _normal_selects_face(segment: BCSegment, face: BoundaryFace) -> bool:
    """Is ``face``'s outward normal within ``segment.normal_tolerance`` (a cosine) of its ``normal_direction``?"""
    outward = np.zeros((1, face.axis + 1))
    outward[0, face.axis] = -1.0 if face.side == "min" else 1.0
    return bool(normal_holds(segment, outward, f"the {face.to_string()} face")[0])


def normal_holds(segment: BCSegment, normals: np.ndarray, where: str) -> np.ndarray:
    """For each outward unit normal (a row of ``normals``), is it within ``segment.normal_tolerance`` (a cosine) of
    the segment's ``normal_direction``? Read against a box face's normal (`_normal_selects_face`) and a mesh facet's
    (`BoundaryConditions.segments_for_facets`, #2512 row B1). `BCSegment.matches_point` on a domain given by an SDF
    reads the normal from the SDF's gradient instead, with its own comparison."""
    normals = np.atleast_2d(np.asarray(normals, dtype=float))
    target = np.asarray(segment.normal_direction, dtype=float).ravel()
    norm = float(np.linalg.norm(target))
    if norm == 0.0 or normals.shape[1] > target.size:
        raise ValueError(
            f"BC segment {segment.name!r}: normal_direction {target.tolist()} cannot select {where} (zero vector, or "
            f"fewer components than the normal's {normals.shape[1]})."
        )
    return normals @ target[: normals.shape[1]] / norm >= segment.normal_tolerance


def region_axis(segment: BCSegment, key: str | int, dimension: int, axis_names: dict[int, str] | None = None) -> int:
    """The axis a ``region`` key names, for every reader of a ``region`` (#2490 B1e-2): a name, among ``axis_names``
    (by default x, y, z, w), or an integer, ``numbers.Integral`` included (a numpy integer). It must name one of the
    domain's ``dimension`` axes. Anything else is refused: a float is not truncated to an axis, a negative index does
    not count from the last axis, and a bool is not an index."""
    if axis_names is None:
        axis_names = {0: "x", 1: "y", 2: "z", 3: "w"}
    if isinstance(key, str):
        axis = next((idx for idx, name in axis_names.items() if name == key), None)
    elif isinstance(key, (bool, np.bool_)):
        axis = None
    elif isinstance(key, numbers.Integral):
        axis = int(key)
    else:
        axis = None
    if axis is None or not 0 <= axis < dimension:
        raise ValueError(
            f"BC segment {segment.name!r}: region axis {key!r} names no axis of a {dimension}-D domain. Name an axis "
            f"({', '.join(axis_names[i] for i in sorted(axis_names) if i < dimension)}) or give its index, an integer "
            f"from 0 to {dimension - 1}."
        )
    return axis


def region_holds(
    segment: BCSegment, points: np.ndarray, tolerance: float, axis_names: dict[int, str] | None = None
) -> np.ndarray:
    """For each point (a row of ``points``), does ``segment.region`` hold it: every named axis within its range, to
    ``tolerance``? The point family's one reading of ``region``: `BCSegment.matches_point` and a mesh facet's midpoint
    (`BoundaryConditions.segments_for_facets`, #2512 row B1). Each key is read by `region_axis`, as the face readers
    read it."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    held = np.ones(points.shape[0], dtype=bool)
    for axis_key, (range_min, range_max) in segment.region.items():
        coord = points[:, region_axis(segment, axis_key, points.shape[1], axis_names)]
        held &= (coord >= range_min - tolerance) & (coord <= range_max + tolerance)
    return held


def sdf_region_holds(segment: BCSegment, points: np.ndarray, tolerance: float) -> np.ndarray:
    """For each point (a row of ``points``), is it inside ``segment.sdf_region``, to ``tolerance``? The point family's
    one reading of ``sdf_region``, as `region_holds` is of ``region``."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return np.array([segment.sdf_region(point) <= tolerance for point in points], dtype=bool)


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
        value: BC value: a constant, a callable g(t, x) named by its parameters ((t, x), (x) or (t)), or a provider
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
    # BC value: a constant, a callable g(t, x) -- time first, bound by its parameter names (#2512 row B2) -- or a
    # BCValueProvider (Issue #625), resolved by the coupling layer with the current iteration state.
    # Read it through `segment_value`, the one reading of it.
    value: float | Callable | BCValueProvider = 0.0

    # Robin BC coefficients: alpha*u + beta*du/dn = g
    # `float | BCValueProvider | NDArray` for the same reason `value` is: a provider may sit
    # here and resolve to a field. `ResolvedBC` was widened for this in #1957; this class was
    # the one left behind. The declared type is what a reader checks against, and it said
    # `float` while the resolver stored an array.
    alpha: float | Any = 1.0  # Weight on u (Dirichlet term)
    beta: float | Any = 0.0  # Weight on du/dn (Neumann term)

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
    # No FP solver honours a cap yet; every FP solver that takes a segment BC refuses one carrying it (#2575).
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
        # Bind a callable value now, so one written (x, t) or with unnamed parameters is refused when the segment is
        # built, not when a solver first reads it (#2512 row B2, user ruling).
        if callable(self.value) and not is_provider(self.value):
            self.value_binding()

    def value_binding(self) -> BoundCallable:
        """The binding of this segment's callable value to ``(t, x)``, made once per value object.

        Keyed on the value's identity, so a ``value`` reassigned after construction is bound afresh rather than
        read through the binding of the callable it replaced.
        """
        value = self.value
        if not callable(value):
            raise TypeError(
                f"BCSegment '{self.name}' has a value of type {type(value).__name__}, not a callable to bind."
            )
        cached = self.__dict__.get("_value_binding")
        if cached is None or cached.fn is not value:
            cached = bind_boundary_value(value)
            self.__dict__["_value_binding"] = cached
        return cached

    def matches_point(
        self,
        point: np.ndarray,
        boundary_id: str | None,
        domain_bounds: np.ndarray | None,
        tolerance: float = SDF_BOUNDARY_TOL,
        axis_names: dict[int, str] | None = None,
        domain_sdf: Callable[[np.ndarray], float] | None = None,
        geometry: SupportsRegionMarking | None = None,
        normal_at: np.ndarray | None = None,
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
            geometry: The grid whose marked-region mask places a region_name segment
                     (Issue #596 Phase 2.5); a region name that is a face label is read
                     without one (#2490).
            normal_at: Where the outward normal of ``domain_sdf`` is read; by default ``point``.
                     `BoundaryConditions.segment_for_point` passes the point as it was asked, and reads the
                     regions at its nearest point of the box (#2780). Moved onto a corner of a box SDF, the
                     point would read the kink's diagonal, which matches neither face above a ``normal_tolerance``
                     of 1/sqrt(2) and both at or below it.

        Returns:
            True if this BC segment applies to the given point

        Raises:
            ValueError: If point is empty, domain_bounds are invalid, or a region_name
                       segment cannot be read at the point (region_name_holds_point)
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
        #
        # Compared as FACES, not as strings. `parse_boundary_face` is the owner and it resolves
        # aliases -- "left" and "x_min" are the same face -- but this site used `!=` on the raw
        # strings, so the same `BoundaryConditions` object answered differently depending on which
        # route asked. Measured end-to-end on a 1-D FP-FDM solve: with `boundary="x_min"` the
        # absorbing wall gave `m[0] = 0.0`; with `boundary="left"`, the alias `BCSegment`'s own
        # docstring lists first, it gave 3.9e-02 -- the wall had silently become no-flux. #1939
        if self.boundary is not None and self.boundary != "all":
            if boundary_id is None:
                return False
            mine = parse_boundary_face(self.boundary)
            theirs = parse_boundary_face(boundary_id)
            if mine is None or theirs is None:
                if self.boundary != boundary_id:
                    return False
            elif mine != theirs:
                return False

        # Method 2: Check coordinate range constraints (rectangular domains)
        if self.region is not None and not region_holds(self, point[None, :], tolerance, axis_names)[0]:
            return False

        # Method 3: Check SDF region constraint (general domains)
        if self.sdf_region is not None and not sdf_region_holds(self, point[None, :], tolerance)[0]:
            return False

        # Method 4: Check normal direction match. On a box domain (no SDF) the outward normal is the
        # face's own, read from `boundary_id`; before, it was not read at all, so the segment matched on
        # every face (#2512 row B1).
        if self.normal_direction is not None and domain_sdf is None:
            face = parse_boundary_face(boundary_id)
            if face is None:
                raise ValueError(
                    f"BCSegment '{self.name}': normal_direction needs the outward normal at the point, and "
                    "there is neither a domain_sdf nor a box face to read it from."
                )
            if not _normal_selects_face(self, face):
                return False
        if self.normal_direction is not None and domain_sdf is not None:
            # Compute outward normal via SDF gradient (finite differences)
            normal = _compute_sdf_gradient(point if normal_at is None else normal_at, domain_sdf, epsilon=1e-5)
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

        # Method 5: a marked region, read as the face rule reads it (#2490 B1e-1)
        if self.region_name is not None:
            from .conditions import region_name_holds_point

            if not region_name_holds_point(self, point, boundary_id, domain_bounds, tolerance, geometry):
                return False

        return True

    def get_value(
        self,
        point: np.ndarray,
        t: float | None = None,
        state: dict[str, Any] | None = None,
    ) -> float:
        """
        Evaluate BC value at a point.

        Args:
            point: Spatial coordinates as 1D array
            t: The time to read a time-dependent value at. With none, a value that declares ``t`` is refused,
               not read at a default time (#2512 row B2); a constant, or ``g(x)``, needs none.
            state: Optional iteration state dict for BCValueProvider resolution
                   (Issue #625). Required if value is a provider. Standard keys:
                   'm_current', 'U_current', 'geometry', 'volatility', 'iteration'.

        Returns:
            BC value at this point and time

        Read through the one owner, `segment_value` (#2512 row B2): a callable is g(t, x), bound by its parameter
        names, so ``point`` is passed as ``x`` and ``t`` as ``t``; a constant follows the owner's rules; a provider
        is computed from ``state`` and refused without it.

        Raises:
            NotImplementedError: If the value is a provider and ``state`` is not given, or a value the owner refuses.
            TypeError: If a callable value, reassigned after the segment was built, cannot be bound by name.
            ValueError: If ``point`` is not one point of shape ``(d,)``.
        """
        return segment_value(
            self, t=t, x=np.asarray(point, dtype=float), where="BCSegment.get_value", state=state, time_keyword="t"
        )

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


def _declares_time(value: Any, segment: BCSegment | None = None) -> bool:
    """Whether ``value`` is a callable that declares ``t``, which an accessor called with no time cannot read."""
    if not callable(value) or is_provider(value):
        return False
    bound = segment.value_binding() if segment is not None else bind_boundary_value(value)
    return "t" in bound.passes


def bind_boundary_value(fn: Callable[..., Any]) -> BoundCallable:
    """Bind a boundary value callable to ``(t, x)`` by its parameter names, and refuse it otherwise.

    Stricter than `bind_user_callable` for a reordered role, which binds ``(t, pos)`` positionally since ``t`` is in
    place: a boundary value is called by name only (user ruling, #2512 row B2), so one with a parameter named
    otherwise and no default is refused, and so is one that cannot be called by name -- a positional-only
    parameter, or a signature that cannot be read. A parameter named otherwise with a default keeps its default.

    It is a function the user writes, a ``def`` or a ``lambda``, or a ``functools.partial`` of one (user ruling,
    2026-10-11): another callable's parameter names were not written against the slots. A SciPy interpolant takes
    ``(x, ...)`` and would read time data at the position; a class instance, a bound method, a builtin, a ufunc, a
    vectorized or jax-wrapped function are refused the same way, with the wrap that makes each one a function.
    """
    name = getattr(fn, "__qualname__", fn)
    target = fn
    while isinstance(target, functools.partial):
        target = target.func
    if not inspect.isfunction(target):
        raise TypeError(
            f"boundary value {name!r} is of type {type(fn).__name__}, not a plain def or lambda: a boundary value is a "
            "function whose parameter names declare what it reads, and a def that a decorator turned into another "
            "object (functools.cache, lru_cache, numba.njit) counts as another callable. Wrap it, such as "
            "`lambda t: f(t)`, `lambda t: s(t)` for an interpolant over time, or `lambda t: np.sin(t)` (#2512 row B2)."
        )
    try:
        declared = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        declared = None
    if declared is None or any(p.kind is inspect.Parameter.POSITIONAL_ONLY for p in declared):
        # `def f(t, /)`: renaming cannot help, wrapping does.
        what = (
            "has no signature Python can read"
            if declared is None
            else f"takes {inspect.signature(fn)}, with positional-only parameters"
        )
        raise TypeError(
            f"boundary value {name!r} {what}, so it cannot be called by name: wrap it in a function that names what "
            "it reads, such as `lambda t: f(t)` or `lambda x: f(x[0])` (#2512 row B2)."
        )
    bound = bind_user_callable(fn, BOUNDARY_VALUE_SLOTS, role="boundary value")
    if bound.binding != "keyword":
        try:
            params = ", ".join(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            params = "an unreadable signature"
        raise TypeError(
            f"boundary value {name!r} takes ({params}), whose parameters are not all "
            "named after the slots: a boundary value names them t, x, time first (#2378 phase 5, #2375 ruling 8), "
            "and is called by name."
        )
    return bound


def _with_default(fn: Callable[..., Any], *names: str) -> str:
    """The words ", with a default" when ``fn`` gives the parameter of one of ``names`` a default, else none."""
    parameters = inspect.signature(fn).parameters
    for name in names:
        if name in parameters and parameters[name].default is not inspect.Parameter.empty:
            return ", with a default"
    return ""


def segment_value(
    segment: BCSegment,
    *,
    t: float | None,
    x: NDArray | None,
    where: str,
    state: dict[str, Any] | None = None,
    time_keyword: str | None = None,
) -> Any:
    """The one reading of a boundary segment's value, for every reader (#2512 row B2): `boundary_value`, with the
    binding the segment made of a callable value when it was built.

    A FLUX's value is q itself, which a None constant would read as 0 and solve as NO_FLUX: it is refused here, the read
    every FLUX reader makes, whenever and however the value became None (#2581)."""
    if segment.bc_type == BCType.FLUX and segment.value is None:
        raise ValueError(
            f"Segment '{segment.name}' with flux BC requires a value: q, the prescribed mass flux J.n through it, "
            f"outward positive (q = 0 is NO_FLUX); read by {where} (#2581)."
        )
    return boundary_value(
        segment.value,
        kind=f"{segment.bc_type.name.replace('_', ' ').title()} segment '{segment.name}'",
        t=t,
        x=x,
        where=where,
        state=state,
        binding=segment.value_binding() if callable(segment.value) and not is_provider(segment.value) else None,
        time_keyword=time_keyword,
    )


def boundary_value(
    value: Any,
    *,
    kind: str,
    t: float | None,
    x: NDArray | None,
    where: str,
    state: dict[str, Any] | None = None,
    binding: BoundCallable | None = None,
    time_keyword: str | None = None,
) -> Any:
    """The one reading of a boundary value -- a segment's, or a ``default_value`` (#2512 row B2).

    **A callable** is ``g(t, x)``, time first (#2378 phase 5, #2375 ruling 8; user ruling for boundary values). It
    is bound by its parameter names when the segment is built (`BCSegment.value_binding`) and called by keyword with
    what it declares: ``(t, x)``, ``(x)``, ``(t)`` or ``(time)``. ``x`` is one point, a 1-D array of shape
    ``(d,)``, ``d = 1`` included. A reader with no time passes ``t=None``, and one with no point ``x=None``. A
    callable that declares what its reader lacks is refused there, at first use, and the message names ``where``,
    the reader and the face. A reader that takes a time but was given none names its parameter in ``time_keyword``,
    and the refusal says to pass it; a reader with no time to take advises a value that needs none. It returns one
    number at the point.

    **A constant, and what a callable or a provider returns**, are read by one function, `_read_constant`, in three
    layers. First the dtype: an array, masked or not, must have an integer or a floating dtype, and any other array
    is refused at the container, whatever it holds, as is a boolean or complex scalar. Then one unwrap: an array of
    one element is that element, and a return of more than one number is refused. Then the value:
    ``None`` (0 for a constant, refused for a return), a masked element and whatever `_describe_bc_value` calls zero
    are 0, a finite real is its value, and anything else, a non-finite number included, is refused.

    **A** ``BCValueProvider`` is resolved with ``state``, and its result is read as a return is: one number, since
    a value per node is a separate capability (#2600). Without ``state`` it is refused (``NotImplementedError``): the
    coupling layer resolves providers before a solve (`with_resolved_providers`).
    """
    if is_provider(value):
        if state is None:
            raise NotImplementedError(
                f"{kind} has a value of type {type(value).__name__}, a BCValueProvider, and {where} has no "
                "iteration state to resolve it with: providers are resolved by the coupling layer before a solve "
                "(with_resolved_providers)."
            )
        # A provider returns one number, read as a return is (#2512 row B2; one value per node is #2600).
        return _read_constant(
            value.compute(state),
            kind=f"{kind} (what its provider computed)",
            where=where,
            returned=True,
            many=" A value per node is a planned capability, tracked in #2600.",
        )
    if callable(value):
        bound = binding if binding is not None else bind_boundary_value(value)
        if "t" in bound.passes and t is None:
            if time_keyword is not None:
                raise NotImplementedError(
                    f"{kind}: its value declares t{_with_default(bound.fn, 't', 'time')}, but {where} was given no "
                    f"time. A caller that has one passes it as {time_keyword}=; a solver that reaches {where} without "
                    "one does not yet support a time-dependent value there (#2615; #2512 row B2)."
                )
            raise NotImplementedError(
                f"{kind}: its value declares t{_with_default(bound.fn, 't', 'time')}, but {where} has no time to pass "
                f"it; write it as {'a constant, since it has no point either' if x is None else 'g(x) if it does not vary in time'} "
                "(#2512 row B2)."
            )
        if "x" in bound.passes and x is None:
            # A default for x would read one fixed position, not the face, so it does not make x optional here.
            raise NotImplementedError(
                f"{kind}: its value declares x{_with_default(bound.fn, 'x')}, but {where} reads one value for the "
                "whole face and has no point to pass it; a value that varies along the face is not supported there, "
                f"so write it as {'a constant, since it has no time either' if t is None else 'g(t) if it does not vary along the face'} "
                "(#2512 row B2)."
            )
        if x is not None and np.ndim(x) != 1:
            raise ValueError(f"{where} passed x of shape {np.shape(x)}; a boundary value takes one point, shape (d,).")
        return _read_constant(bound(t=t, x=x), kind=f"{kind} (what its callable returned)", where=where, returned=True)
    return _read_constant(value, kind=kind, where=where)


def _dtype_refusal(value: Any) -> str | None:
    """What layer 1 refuses ``value`` as, or None: "boolean", "complex", or "dtype" for any other non-numeric array.

    An array, masked or not, must have an integer or a floating dtype; anything else is refused at the container,
    whatever it holds, so nothing is walked: a boolean or complex array, an object array (and so any value nested in
    one), a string, bytes, datetime or timedelta array (audit session ruling on review 1 of #2601). A scalar is
    refused here only if it is a boolean or a complex number; layer 3 reads every other one.
    """
    if isinstance(value, (bool, np.bool_)):
        return "boolean"
    if isinstance(value, (complex, np.complexfloating)):
        return "complex"
    if isinstance(value, np.ndarray):
        dtype = np.ma.getdata(value).dtype
        if dtype == np.bool_:
            return "boolean"
        if np.issubdtype(dtype, np.complexfloating):
            return "complex"
        # Signed, unsigned or floating, by kind: `np.issubdtype(..., np.integer)` would admit timedelta64.
        if dtype.kind not in "iuf":
            return "dtype"
    return None


_VALUE_ADVICE = "give a real number, a real array of one element, or a callable g(t, x)"


def _read_constant(
    value: Any, *, kind: str, where: str, returned: bool = False, advice: str = _VALUE_ADVICE, many: str = ""
) -> float:
    """A constant boundary value, or what a callable or a provider returned: `boundary_value`'s contract, in layers.

    1. **dtype:** an array, masked or not, must have an integer or a floating dtype, and any other array is refused
       at the container; so is a boolean or complex scalar.
    2. **one unwrap:** an array of one element, of any shape, masked or not, is that element; a masked
       ``np.matrix`` is the exception: it stays two-dimensional, so layer 3 reads it whole, as 0 if it is masked or
       zero and refused otherwise. A returned array of
       more than one element is refused here too, since a callable returns one number at a point.
    3. **the value:** ``None`` is 0 for a constant and refused for a return, which a function with no ``return``
       gives; a masked element is 0, and so is whatever `_describe_bc_value` calls zero; a finite real is its
       value; anything else, and a non-finite number, is refused.

    One function for constants and returns, so a value reads the same whichever way it arrives (#2512 row B2; the
    layers close reviews 1 and 2 of #2598, one class rather than one corner at a time).
    """
    from mfgarchon.geometry.boundary.bc_utils import _describe_bc_value

    leaf = _dtype_refusal(value)
    if leaf == "boolean":
        raise NotImplementedError(f"{kind} has a boolean value, which is not a boundary datum: {advice} (#2593).")
    if leaf == "complex":
        # Before the zero test: `_describe_bc_value` reads np.complex128(1j) as zero, since `float()` drops the
        # imaginary part, so a purely imaginary datum would become 0 in silence (#2512 row B2).
        raise NotImplementedError(f"{kind} has a complex value {value!r}; {where} takes a real boundary datum.")
    if leaf == "dtype":
        raise NotImplementedError(
            f"{kind} has a value of type {type(value).__name__} with dtype {np.ma.getdata(value).dtype}, which {where} "
            f"cannot read: {advice} (#2512 row B2)."
        )
    if returned and isinstance(value, np.ndarray) and value.size != 1:
        raise NotImplementedError(
            f"{kind}: its value returned {value.size} numbers, shape {value.shape}, at one point; {where} reads one "
            f"number per point (#2512 row B2).{many}"
        )
    if isinstance(value, np.ndarray) and value.size == 1:
        value = np.ma.asarray(value).reshape(-1)[0] if np.ma.isMaskedArray(value) else np.asarray(value).reshape(-1)[0]
    if value is None:
        if returned:
            raise NotImplementedError(
                f"{kind}: its value returned None at one point, which {where} cannot read; does the function lack a "
                "`return`? (#2512 row B2)"
            )
        return 0.0
    if value is np.ma.masked or _describe_bc_value(value) is None:
        return 0.0
    real = isinstance(value, numbers.Real) or (
        isinstance(value, np.ndarray)
        and value.ndim == 0
        and np.issubdtype(value.dtype, np.number)
        and not np.iscomplexobj(value)
    )
    if not real:
        raise NotImplementedError(
            f"{kind} has a value of type {type(value).__name__}, which {where} cannot read: {advice} (#2593)."
        )
    constant = float(value)
    if not np.isfinite(constant):
        raise NotImplementedError(
            f"{kind} has a non-finite value {constant}; {where} would impose it and return a non-finite solution "
            "(#2593)."
        )
    return constant


def segment_robin_coefficients(
    segment: BCSegment, *, where: str, state: dict[str, Any] | None = None, as_stored: bool = False
) -> tuple[Any, Any]:
    """The one reading of a segment's Robin coefficients ``(alpha, beta)``, for every reader (#2512 row B2).

    Each is a constant, read by `boundary_value`'s constant contract, or a provider resolved with ``state``. A
    callable coefficient is refused: no reader evaluates one per point or per time (#1237). ``None`` is refused
    rather than read as 0, as `BoundaryConditions` validation refuses it for a Robin segment.

    ``as_stored=True`` returns the two coefficients as the segment stores them, a provider unresolved, for a reader
    that only compares two segments and writes no row -- an equivalence: the owner's `_same_condition`, HJB-GFDM's
    `_same_bc_row`, the interpolation applicator's `_same_enforcement`. Such a comparison may run before the coupling
    layer resolves any provider: HJB-GFDM classifies its points at construction (#2512 row B1).
    """
    if as_stored:
        return tuple(getattr(segment, name) for name in ("alpha", "beta"))
    coefficients = []
    for name in ("alpha", "beta"):
        coefficient = getattr(segment, name)
        kind = f"{segment.bc_type.name.replace('_', ' ').title()} segment {segment.name!r}'s {name}"
        if coefficient is None:
            raise NotImplementedError(f"{kind} is None; {where} needs a number (#2512 row B2).")
        if callable(coefficient) and not is_provider(coefficient):
            raise NotImplementedError(
                f"{kind} is a callable; a Robin coefficient is one constant per segment, since no reader evaluates "
                f"one per point or per time, so {where} cannot read it (#1237)."
            )
        if isinstance(coefficient, np.ndarray) and coefficient.size > 1:
            # A provider may resolve a coefficient to a field along the wall (`with_resolved_providers` keeps it); no
            # reader takes one yet. A planned capability, not a malformed input.
            raise NotImplementedError(
                f"{kind} is a field of {coefficient.size} values along the wall; {where} reads one number per "
                "segment. A coefficient per node is a planned capability, tracked in #2600."
            )
        returned = is_provider(coefficient)
        if returned:
            if state is None:
                raise NotImplementedError(
                    f"{kind} is a BCValueProvider, and {where} has no iteration state to resolve it with: providers "
                    "are resolved by the coupling layer before a solve (with_resolved_providers)."
                )
            coefficient = coefficient.compute(state)
        coefficients.append(
            _read_constant(
                coefficient,
                kind=kind,
                where=where,
                returned=returned,
                advice="give a real number, or a real array of one element: a Robin coefficient is one constant per segment",
                many=" A coefficient per node is a planned capability, tracked in #2600.",
            )
        )
    return coefficients[0], coefficients[1]


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

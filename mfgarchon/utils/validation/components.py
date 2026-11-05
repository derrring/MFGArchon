"""
Validation for MFG problem components (IC/BC).

This module validates initial conditions (m_initial) and terminal conditions (u_terminal)
provided via MFGComponents, and boundary condition compatibility with geometry.

Issues:
- #679: IC/BC Validation, BC-geometry compatibility
- #681: Core IC/BC array and callable validation
- #682: Geometry-agnostic IC/BC validation
- #683: Mass normalization validation
- #684: Callable signature detection
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.validation.protocol import ValidationResult

logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.geometry.protocol import GeometryProtocol


def validate_components(
    components: MFGComponents | None,
    geometry: GeometryProtocol,
    *,
    require_m_initial: bool = True,
    require_u_terminal: bool = True,
    check_mass_normalization: bool = False,
    terminal_time: float = 0.0,
) -> ValidationResult:
    """
    Validate MFGComponents against geometry.

    Args:
        components: MFGComponents to validate
        geometry: Geometry to validate against
        require_m_initial: Whether m_initial is required
        require_u_terminal: Whether u_terminal is required
        check_mass_normalization: Whether to verify integral of m_initial = 1
        terminal_time: The horizon T, at which a time-first ``u_terminal(t, x)`` is read (#2431).
            It defaults to 0, so a caller validating a time-first ``u_terminal`` passes T.

    Returns:
        ValidationResult with any issues found

    Example:
        result = validate_components(components, geometry)
        if not result.is_valid:
            raise ValidationError(result)
    """
    result = ValidationResult()

    if components is None:
        if require_m_initial or require_u_terminal:
            result.add_error(
                "MFGComponents is required but was None",
                location="components",
                suggestion="Pass components=MFGComponents(m_initial=..., u_terminal=...)",
            )
        return result

    # Validate m_initial
    if require_m_initial:
        m_result = validate_m_initial(components.m_initial, geometry)
        result.issues.extend(m_result.issues)
        if not m_result.is_valid:
            result.is_valid = False

    # Validate u_terminal
    if require_u_terminal:
        u_result = validate_u_terminal(components.u_terminal, geometry, terminal_time=terminal_time)
        result.issues.extend(u_result.issues)
        if not u_result.is_valid:
            result.is_valid = False

    # Check mass normalization if requested
    if check_mass_normalization and components.m_initial is not None:
        mass_result = validate_mass_normalization(components.m_initial, geometry)
        result.issues.extend(mass_result.issues)
        # Mass normalization is a warning, not an error
        # (doesn't invalidate the result)

    # Validate BC-geometry compatibility (Issue #679)
    bc = getattr(components, "boundary_conditions", None)
    bc_result = validate_boundary_conditions(bc, geometry)
    result.issues.extend(bc_result.issues)
    if not bc_result.is_valid:
        result.is_valid = False

    return result


def validate_m_initial(
    m_initial: Callable | NDArray[np.floating] | None,
    geometry: GeometryProtocol,
) -> ValidationResult:
    """
    Validate initial density m_initial.

    Checks:
    - Not None (required)
    - If callable: correct signature and return type
    - If array: correct shape matching geometry
    - Values are finite (no NaN/Inf)

    Args:
        m_initial: Initial density (callable or array)
        geometry: Geometry for shape validation

    Returns:
        ValidationResult

    Issue #681: Core IC/BC validation
    """
    result = ValidationResult()

    if m_initial is None:
        result.add_error(
            "m_initial (initial density) is required but was None",
            location="m_initial",
            suggestion="Provide m_initial=lambda x: ... or m_initial=np.array(...)",
        )
        return result

    # Validate based on type
    if callable(m_initial):
        return _validate_callable_ic(m_initial, geometry, "m_initial")
    elif isinstance(m_initial, np.ndarray):
        return _validate_array_ic(m_initial, geometry, "m_initial")
    else:
        result.add_error(
            f"m_initial must be callable or ndarray, got {type(m_initial).__name__}",
            location="m_initial",
        )

    return result


def validate_u_terminal(
    u_terminal: Callable | NDArray[np.floating] | None,
    geometry: GeometryProtocol,
    *,
    terminal_time: float = 0.0,
) -> ValidationResult:
    """
    Validate terminal value function u_terminal.

    Checks:
    - Not None (required)
    - If callable: correct signature and return type
    - If array: correct shape matching geometry
    - Values are finite (no NaN/Inf)

    Args:
        u_terminal: Terminal value function (callable or array)
        geometry: Geometry for shape validation
        terminal_time: The horizon T, at which a time-first ``u_terminal(t, x)`` is read (#2431).
            It defaults to 0, so a caller validating a time-first ``u_terminal`` passes T.

    Returns:
        ValidationResult

    Issue #681: Core IC/BC validation
    """
    result = ValidationResult()

    if u_terminal is None:
        result.add_error(
            "u_terminal (terminal condition) is required but was None",
            location="u_terminal",
            suggestion="Provide u_terminal=lambda x: ... or u_terminal=np.array(...)",
        )
        return result

    # Validate based on type
    if callable(u_terminal):
        return _validate_callable_ic(u_terminal, geometry, "u_terminal", time_value=terminal_time)
    elif isinstance(u_terminal, np.ndarray):
        return _validate_array_ic(u_terminal, geometry, "u_terminal")
    else:
        result.add_error(
            f"u_terminal must be callable or ndarray, got {type(u_terminal).__name__}",
            location="u_terminal",
        )

    return result


def validate_u_final(
    u_final: Callable | NDArray[np.floating] | None,
    geometry: GeometryProtocol,
) -> ValidationResult:
    """Deprecated: use validate_u_terminal() instead.

    .. deprecated:: v0.17.6
        Use :func:`validate_u_terminal` instead. Will be removed in v1.0.0.
    """
    from mfgarchon.utils.deprecation import deprecated

    # Apply decorator dynamically to avoid import cycle at module level
    @deprecated(
        since="v0.17.6",
        replacement="use validate_u_terminal() instead",
        reason="Renamed for consistency with MFG literature terminology",
    )
    def _deprecated_validate_u_final(
        u_final: Callable | NDArray[np.floating] | None,
        geometry: GeometryProtocol,
    ) -> ValidationResult:
        return validate_u_terminal(u_final, geometry)

    return _deprecated_validate_u_final(u_final, geometry)


def validate_mass_normalization(
    m_initial: Callable | NDArray[np.floating],
    geometry: GeometryProtocol,
    tolerance: float = 0.1,
) -> ValidationResult:
    """
    Validate that initial density integrates to approximately 1.

    For probability density interpretation, we expect:
        integral of m_initial over domain = 1

    Supports geometry-specific integration:
      - Cartesian grids: rectangle rule (sum * cell_volume)
      - Implicit domains: Monte Carlo (mean * domain_volume)
      - Other geometry types: emits warning that mass check is unsupported

    Args:
        m_initial: Initial density (callable or pre-evaluated array).
        geometry: Geometry for integration.
        tolerance: Absolute tolerance for |mass - 1|.

    Returns:
        ValidationResult with warning if mass is not approximately 1.

    Issue #683: Mass normalization validation.
    """
    result = ValidationResult()

    try:
        # Get density values on the grid
        if callable(m_initial):
            values = _evaluate_callable_on_grid(m_initial, geometry)
            if values is None:
                result.add_warning(
                    "Could not evaluate callable m_initial for mass normalization",
                    location="m_initial",
                )
                return result
        else:
            values = m_initial

        # Compute integral via geometry-specific method
        total_mass = _compute_mass(values, geometry)
        if total_mass is None:
            geo_name = type(geometry).__name__
            result.add_warning(
                f"Mass normalization not supported for geometry type {geo_name}",
                location="m_initial",
                suggestion="Mass check is available for Cartesian grids and implicit domains",
            )
            return result

        result.context["computed_mass"] = total_mass

        # Check if mass is approximately 1
        if abs(total_mass - 1.0) > tolerance:
            result.add_warning(
                f"Initial density mass = {total_mass:.4f}, expected 1.0 (tolerance: {tolerance})",
                location="m_initial",
                suggestion="Normalize m_initial so that its integral equals 1",
            )

    except Exception as e:
        result.add_warning(
            f"Could not verify mass normalization: {e}",
            location="m_initial",
        )

    return result


def validate_boundary_conditions(
    boundary_conditions: object | None,
    geometry: GeometryProtocol,
) -> ValidationResult:
    """
    Validate boundary conditions compatibility with geometry.

    Checks:
    - BC dimension matches geometry dimension (error if mismatch)
    - Periodic BC is only used on Cartesian grids (warning otherwise)

    Args:
        boundary_conditions: BoundaryConditions object (or None to skip).
        geometry: Geometry to validate against.

    Returns:
        ValidationResult with any issues found.

    Issue #679: BC-geometry compatibility validation.
    """
    from mfgarchon.geometry.boundary.types import BCType
    from mfgarchon.geometry.protocol import GeometryType

    result = ValidationResult()

    if boundary_conditions is None:
        return result

    # Get BC dimension (may be None for lazy-bound BCs)
    bc_dim = getattr(boundary_conditions, "dimension", None)
    geo_dim = geometry.dimension

    # --- Check 1: BC dimension vs geometry dimension ---
    if bc_dim is not None and bc_dim != geo_dim:
        result.add_error(
            f"Boundary condition dimension ({bc_dim}) does not match geometry dimension ({geo_dim})",
            location="boundary_conditions",
            suggestion=f"Create boundary conditions with dimension={geo_dim}",
        )

    # --- Check 2: Periodic BC on non-Cartesian geometry ---
    segments = getattr(boundary_conditions, "segments", [])
    default_bc = getattr(boundary_conditions, "default_bc", None)
    geo_type = geometry.geometry_type

    has_periodic = default_bc == BCType.PERIODIC
    if not has_periodic:
        has_periodic = any(getattr(seg, "bc_type", None) == BCType.PERIODIC for seg in segments)

    if has_periodic and geo_type != GeometryType.CARTESIAN_GRID:
        geo_name = type(geometry).__name__
        result.add_warning(
            f"Periodic boundary conditions are intended for Cartesian grids, "
            f"but geometry is {geo_name} (type: {geo_type.value})",
            location="boundary_conditions",
            suggestion="Use Dirichlet, Neumann, or Robin BC for non-Cartesian geometries",
        )

    return result


# --- Internal helpers (geometry-agnostic, Issue #682) ---


def _get_validation_points(
    geometry: GeometryProtocol,
    n_samples: int = 3,
) -> np.ndarray:
    """Get sample points for callable validation from any geometry type.

    Uses ``get_collocation_points()`` (part of GeometryProtocol) which
    returns ``(N, d)`` for all geometries.  Interior points are preferred
    to avoid boundary edge cases.

    For implicit/meshfree geometries where ``get_collocation_points()``
    requires arguments (e.g., Hyperrectangle), falls back to generating
    sample points from bounds.

    Args:
        geometry: Any geometry implementing GeometryProtocol.
        n_samples: Number of sample points to return.

    Returns:
        Array of shape ``(n_samples, d)`` with sample points.

    Raises:
        ValueError: If the geometry has no collocation points and no bounds.

    Issue #682: Geometry-agnostic IC/BC validation.
    """
    try:
        points = geometry.get_collocation_points()  # (N, d) per protocol
    except TypeError as err:
        # Implicit geometries (Hyperrectangle, etc.) require arguments.
        # Fall back to generating sample points from bounds.
        bounds = geometry.get_bounds()
        if bounds is None:
            raise ValueError(
                f"Geometry {type(geometry).__name__}.get_collocation_points() requires "
                "arguments and get_bounds() returned None. Cannot generate sample points."
            ) from err
        min_coords, max_coords = bounds
        dim = len(min_coords)
        # Generate n_samples interior points (avoid exact boundaries)
        margin = 0.1
        points = np.zeros((n_samples, dim))
        for i in range(n_samples):
            # Spread samples across domain interior
            t = (i + 1) / (n_samples + 1)
            points[i] = (
                min_coords + t * (max_coords - min_coords) * (1 - 2 * margin) + margin * (max_coords - min_coords)
            )
        return points

    # Handle 1D case where points may be (N,) instead of (N, 1)
    if points.ndim == 1:
        points = points.reshape(-1, 1)

    n_total = len(points)
    if n_total == 0:
        raise ValueError(f"Geometry {type(geometry).__name__} returned 0 collocation points")

    if n_total <= n_samples:
        return points

    # Sample interior points (skip first/last to avoid boundary)
    indices = np.linspace(0, n_total - 1, n_samples + 2, dtype=int)[1:-1]
    return points[indices]


def _get_expected_ic_shape(
    geometry: GeometryProtocol,
) -> tuple[int, ...]:
    """Get expected IC/BC array shape for any geometry type.

    Uses ``get_grid_shape()`` (part of GeometryProtocol) with a fallback
    to ``(num_spatial_points,)`` if the geometry returns None.

    Returns:
        Expected shape tuple for IC/BC arrays.

    Issue #682: Geometry-agnostic IC/BC validation.
    """
    shape = geometry.get_grid_shape()
    if shape is not None:
        return shape
    # Fallback for geometries whose base class returns None
    return (geometry.num_spatial_points,)


def _validate_callable_ic(
    func: Callable,
    geometry: GeometryProtocol,
    name: str,
    *,
    time_value: float = 0.0,
) -> ValidationResult:
    """Validate a callable initial/terminal condition.

    Binds it by the names it declares and reads it at one point of the geometry, ``_get_validation_points()``.

    Issues #682, #2875: Geometry-agnostic callable validation.
    """
    from mfgarchon.types.callable_protocols import CONDITION_SLOTS, _remembered, point_tuple

    result = ValidationResult()

    # Get a sample point from any geometry type (Issue #682)
    try:
        points = _get_validation_points(geometry, n_samples=1)
        sample_point = points[0]  # (d,) ndarray
    except Exception as e:
        result.add_error(
            f"Could not get sample point from geometry: {e}",
            location=name,
        )
        return result

    try:
        bound = _remembered(func, CONDITION_SLOTS, name)
    except TypeError as e:
        result.add_error(
            str(e),
            location=name,
            suggestion=f"Name {name}'s parameters after what they read: x, and t if it reads time",
        )
        return result

    result.context["geometry_type"] = type(geometry).__name__

    # Evaluate the callable at the sample point to check output quality
    try:
        value = bound(t=time_value, x=point_tuple(sample_point))
    except Exception as e:
        result.add_error(
            f"{name} callable raised exception at sample point {sample_point}: {e}",
            location=name,
            suggestion=f"x is a coordinate tuple: read {name}'s coordinates as x[0], x[1], ...",
        )
        return result

    # Check return type
    if not isinstance(value, (int, float, np.integer, np.floating, np.ndarray)):
        result.add_error(
            f"{name} must return numeric value, got {type(value).__name__}",
            location=name,
        )
        return result

    # Check for NaN/Inf
    if np.isscalar(value):
        if not np.isfinite(value):
            result.add_error(
                f"{name} returned non-finite value at sample point",
                location=name,
            )
    elif isinstance(value, np.ndarray):
        if not np.all(np.isfinite(value)):
            result.add_error(
                f"{name} returned array with NaN or Inf values",
                location=name,
            )

    return result


def _validate_array_ic(
    arr: NDArray[np.floating],
    geometry: GeometryProtocol,
    name: str,
) -> ValidationResult:
    """Validate an array initial/terminal condition.

    Uses ``_get_expected_ic_shape()`` for geometry-agnostic shape inference.

    Issue #682: Geometry-agnostic array validation.
    """
    result = ValidationResult()

    # Get expected shape from any geometry type (Issue #682)
    try:
        expected_shape = _get_expected_ic_shape(geometry)
    except Exception as e:
        result.add_error(
            f"Could not get expected IC shape from geometry: {e}",
            location=name,
        )
        return result

    # Check shape
    if arr.shape != expected_shape:
        result.add_error(
            f"{name} has shape {arr.shape}, expected {expected_shape}",
            location=name,
            suggestion=(f"Reshape {name} to match geometry grid shape. Geometry type: {type(geometry).__name__}"),
        )
        return result

    # Check for NaN/Inf
    if not np.all(np.isfinite(arr)):
        n_nan = np.sum(np.isnan(arr))
        n_inf = np.sum(np.isinf(arr))
        result.add_error(
            f"{name} contains {n_nan} NaN and {n_inf} Inf values",
            location=name,
        )

    return result


# --- Mass computation helpers (Issue #683) ---


def _evaluate_callable_on_grid(
    func: Callable,
    geometry: GeometryProtocol,
) -> np.ndarray | None:
    """Evaluate a callable IC on all collocation points.

    Returns None if evaluation fails (callable has incompatible signature
    or crashes). This is a best-effort helper for mass normalization --
    signature issues should already be caught by ``_validate_callable_ic()``.

    Issue #683: Mass normalization validation.
    """
    from mfgarchon.types.callable_protocols import CONDITION_SLOTS, _remembered, coordinate_tuple

    try:
        points = geometry.get_collocation_points()
        if points.ndim == 1:
            points = points.reshape(-1, 1)

        bound = _remembered(func, CONDITION_SLOTS, "condition")
        values = np.asarray(bound(t=0.0, x=coordinate_tuple(points, points.shape[1])), dtype=float)
        return np.array(np.broadcast_to(values.ravel() if values.size > 1 else values, (points.shape[0],)))
    except Exception:
        logger.debug("Failed to evaluate callable on grid, returning None", exc_info=True)
        return None


def _compute_mass(
    values: np.ndarray,
    geometry: GeometryProtocol,
) -> float | None:
    """Compute total mass (integral) of density over geometry domain.

    Dispatches to geometry-specific integration methods:
      - CARTESIAN_GRID: rectangle rule (sum * cell_volume)
      - IMPLICIT: Monte Carlo (mean * domain_volume)

    Returns:
        Computed mass, or None if integration is unsupported.

    Issue #683: Mass normalization validation.
    """
    from mfgarchon.geometry.protocol import GeometryType

    geo_type = geometry.geometry_type

    if geo_type == GeometryType.CARTESIAN_GRID:
        return _compute_mass_cartesian(values, geometry)
    elif geo_type == GeometryType.IMPLICIT:
        return _compute_mass_implicit(values, geometry)
    else:
        return None


def _compute_mass_cartesian(
    values: np.ndarray,
    geometry: GeometryProtocol,
) -> float:
    """Compute mass using trapezoidal rule for Cartesian grids.

    Uses ``np.trapezoid`` along each axis for accurate integration
    (handles boundary points correctly, unlike simple rectangle rule).

    Issue #683: Mass normalization validation.
    """
    bounds = geometry.get_bounds()
    shape = geometry.get_grid_shape()

    if bounds is None or shape is None:
        return float(np.sum(values))

    min_coords = np.asarray(bounds[0], dtype=float)
    max_coords = np.asarray(bounds[1], dtype=float)
    lengths = max_coords - min_coords
    n_points = np.array(shape, dtype=float)

    # Compute spacing per dimension
    dx = np.where(n_points > 1, lengths / (n_points - 1), lengths)

    # Reshape flat array to grid shape if needed
    grid_values = values.reshape(shape) if values.shape != shape else values

    # Trapezoidal integration along each axis (numpy 2.0+)
    result = grid_values
    for axis in range(len(shape)):
        result = np.trapezoid(result, dx=float(dx[axis]), axis=0)

    return float(result)


def _compute_mass_implicit(
    values: np.ndarray,
    geometry: GeometryProtocol,
) -> float | None:
    """Compute mass using Monte Carlo for implicit domains.

    For particle-based representation:
        mass = mean(density) * domain_volume

    Uses ``geometry.compute_volume()`` if available (ImplicitDomain, Hyperrectangle,
    Hypersphere all provide this method).

    Issue #683: Mass normalization validation.
    """
    compute_volume = getattr(geometry, "compute_volume", None)
    if not callable(compute_volume):
        return None

    volume = compute_volume()
    return float(np.mean(values)) * volume

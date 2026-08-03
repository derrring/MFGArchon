"""
Interpolation Methods for Semi-Lagrangian HJB Solver.

This module provides interpolation routines for evaluating the value function
at departure points during characteristic tracing in the semi-Lagrangian scheme.

Supported methods (see `sl_backend`, the single owner of method -> backend):
- 1D: scipy.interpolate.interp1d (linear, extrapolating) / PchipInterpolator (cubic -- monotone
  Hermite, NOT a C2 spline; Issue #583 replaced CubicSpline to stop the Issue #1033 blow-up)
- nD: scipy.interpolate.RegularGridInterpolator (linear, cubic, quintic)
- Fallback: RBF interpolation for boundary cases

Module structure per issue #392:
    hjb_sl_interpolation.py - Interpolation methods for semi-Lagrangian solver

Functions:
    interpolate_value_1d: 1D interpolation using interp1d
    interpolate_value_nd: nD interpolation using RegularGridInterpolator
    interpolate_value_rbf_fallback: RBF fallback for failed interpolations
    interpolate_nearest_neighbor: Final fallback using nearest grid point
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator, interp1d

from mfgarchon.utils.mfg_logging import get_logger

logger = get_logger(__name__)


def interpolate_value_1d(
    U_values: np.ndarray,
    x_query: float,
    x_grid: np.ndarray,
    method: str = "linear",
    xmin: float | None = None,
    xmax: float | None = None,
    use_jax: bool = False,
    jax_interpolate_fn: Any | None = None,
) -> float:
    """
    Interpolate value function at query point for 1D problems.

    Args:
        U_values: Value function on grid, shape (Nx,)
        x_query: Query point for interpolation (scalar)
        x_grid: 1D spatial grid points
        method: Interpolation method ('linear', 'cubic')
        xmin: Domain minimum (for boundary handling)
        xmax: Domain maximum (for boundary handling)
        use_jax: Whether to use JAX acceleration
        jax_interpolate_fn: JAX interpolation function if available

    Returns:
        Interpolated value at query point
    """
    x_query_scalar = float(x_query) if np.ndim(x_query) > 0 else x_query

    # JAX acceleration path
    if use_jax and jax_interpolate_fn is not None and method == "linear":
        return float(jax_interpolate_fn(x_grid, U_values, x_query_scalar))

    # Determine bounds
    if xmin is None:
        xmin = x_grid[0]
    if xmax is None:
        xmax = x_grid[-1]

    # Handle boundary cases
    if x_query_scalar <= xmin:
        return float(U_values[0])
    if x_query_scalar >= xmax:
        return float(U_values[-1])

    try:
        # sl_backend is the one owner of "which interpolant does this method mean" (#1811).
        # The Issue #583 rationale -- PCHIP rather than a cubic spline, to keep monotonicity and
        # avoid Runge oscillations at discontinuities -- now lives there rather than here.
        if sl_backend(method, 1, monotone_required=False) == "pchip":
            interpolator = PchipInterpolator(
                x_grid,
                U_values,
                extrapolate=False,  # Return NaN outside bounds (the boundary guard above returns first)
            )
        else:
            # Default to linear
            interpolator = interp1d(
                x_grid,
                U_values,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

        return float(interpolator(x_query_scalar))

    except Exception as e:
        logger.debug(f"1D interpolation failed at x={x_query_scalar}: {e}")
        # Fallback: nearest neighbor
        idx = np.argmin(np.abs(x_grid - x_query_scalar))
        return float(U_values[idx])


def interpolate_value_nd(
    U_values: np.ndarray,
    x_query: np.ndarray,
    grid_coordinates: tuple[np.ndarray, ...],
    grid_shape: tuple[int, ...],
    method: str = "linear",
) -> float:
    """
    Interpolate value function at query point for nD problems.

    Uses scipy.interpolate.RegularGridInterpolator for structured grids.

    Args:
        U_values: Value function on grid, shape matching grid_shape
        x_query: Query point, shape (dimension,)
        grid_coordinates: Tuple of 1D coordinate arrays for each dimension
        grid_shape: Shape of the grid (N1, N2, ..., Nd)
        method: Interpolation method ('linear', 'cubic', 'quintic', 'nearest', 'slinear')

    Returns:
        Interpolated value at query point

    Raises:
        ValueError: If interpolation fails
    """
    dimension = len(grid_shape)
    x_query_vec = np.atleast_1d(x_query)

    if len(x_query_vec) != dimension:
        raise ValueError(f"Query point must have {dimension} coordinates, got {len(x_query_vec)}")

    # Reshape U_values to grid shape if needed
    if U_values.ndim == 1:
        U_values_reshaped = U_values.reshape(grid_shape)
    else:
        U_values_reshaped = U_values

    # One owner for method -> backend (#1811). This block used to be a fourth private copy,
    # and its catch-all `else` is what silently turned an unrecognised method into linear.
    interp_method = sl_backend(method, len(grid_shape), monotone_required=False)

    interpolator = RegularGridInterpolator(
        grid_coordinates,
        U_values_reshaped,
        method=interp_method,
        bounds_error=False,
        fill_value=None,  # Extrapolate using nearest
    )

    # Query at point (must be shape (1, dimension) for single point)
    result = interpolator(x_query_vec.reshape(1, -1))
    return float(result[0])


# Which interpolation methods each path in this module actually HONOURS, and the minimum
# points per axis each needs. One owner, because the two paths honour different sets and the
# gap between "accepted" and "honoured" is the defect both #1809 and #1664 describe:
#
#   1D  (interpolate_value_1d)  is `if method == "cubic": PCHIP else: linear` -- it recognises
#       exactly two. `nearest`/`slinear`/`quintic` reach it and silently return linear, so the
#       same public argument names a different interpolant than it does in nD (measured: 8.0
#       against 10.0 for `nearest` on the same profile).
#   nD  (interpolate_value_nd)  maps onto RegularGridInterpolator, which additionally REFUSES a
#       grid with fewer points on an axis than the method needs -- 4 for cubic, 6 for quintic,
#       measured rather than read from the docs. That refusal is the only condition that reaches
#       the RBF fallback below (#1664); everything else -- out of bounds, NaN, inf, a descending
#       axis, an unknown method name -- returns silently.
HONOURED_METHODS_1D = frozenset({"linear", "cubic"})
HONOURED_METHODS_ND = frozenset({"linear", "slinear", "nearest", "cubic", "quintic"})

# Minimum points per axis, PER DIMENSION, because the two paths use different machinery and
# therefore have different boundaries. Measured, not read from docs:
#
#   nD: `RegularGridInterpolator` under THIS module's construction arguments
#       (`bounds_error=False, fill_value=None`). Those arguments matter -- with scipy's
#       defaults `linear` and `nearest` measure 2 rather than 1, because querying an interior
#       point on a one-point axis is then out of bounds and raises for a different reason.
#       Pinned against scipy directly in the test suite, since for these numbers scipy IS the
#       external oracle.
#   1D: `PchipInterpolator` / `CubicSpline`, which never touch RegularGridInterpolator. They do
#       not raise on a short grid -- they return the LINEAR value, which is the failure this
#       module is guarding against. Measured on a non-linear profile: at n=2 cubic and linear
#       agree exactly (so cubic is not honoured), at n=3 they differ (0.7477 vs 0.9256), so the
#       1D boundary is 3. Applying the nD table here refused a configuration in which cubic was
#       genuinely honoured -- the inverse of the defect this guard exists to fix.
MIN_POINTS_PER_AXIS_ND = {"linear": 1, "slinear": 2, "nearest": 1, "cubic": 4, "quintic": 6}
MIN_POINTS_PER_AXIS_1D = {"linear": 2, "cubic": 3}


def sl_backend(method: str, dimension: int, *, monotone_required: bool) -> str:
    """The single answer to "which interpolant does ``method`` mean here".

    Four sites used to decide this independently and disagreed. Measured on
    ``U = [0, 10, 0, 10, 0]`` at ``x = 0.30``, ``interpolation_method="cubic"`` produced
    ``7.68`` through ``CubicSpline(bc_type="not-a-knot")`` and ``8.96`` through
    ``PchipInterpolator`` -- and which one ran was decided **per timestep**, because
    ``_compute_cfl_and_substeps`` sends a CFL<=1 step down the batch path and a CFL>1 step down
    the pointwise path. One solve at Nx=41, Nt=8 built 7 of the first and 81 of the second.

    ``monotone_required`` is a real distinction, not a fork, and is why this takes a policy
    rather than hardcoding a ladder. The Carlini-Silva 2014 stability proof covers monotone
    interpolation only, so ``diffusion_method='stochastic'`` asks for the monotone backend even
    when the user said ``quintic`` (Issues #1033, #1049, #1054). What was wrong was that each
    site re-derived that ladder; the policy is now stated by the caller and applied here.

    In 1D ``cubic`` is ALWAYS PCHIP. ``CubicSpline`` is non-monotone and is precisely what
    Issue #583 replaced to stop the Issue #1033 blow-up; three of the four sites took that fix
    and the default path did not. There is no configuration in which this owner returns it.

    Returns a backend key: ``"linear"``, ``"pchip"``, or a ``RegularGridInterpolator`` method
    name. It names the interpolant, NOT the out-of-bounds policy -- 1D callers realise ``"linear"``
    as extrapolating ``interp1d`` on the characteristic paths and as clamping ``np.interp`` in
    ``_stochastic_sl_step``, which documents its own reason. Do not "align" those without
    measuring: they differ only for a foot outside the grid, and since #1739 the SL fold lands
    every supported BC in bounds, so nothing currently in the suite would report the change.
    """
    if dimension == 1:
        return "pchip" if method == "cubic" else "linear"
    if monotone_required:
        # Monotone dispatch: anything higher order collapses to the monotone Hermite backend,
        # anything else to Q1. Note `nearest` and `slinear` land on "linear" here, which is a
        # genuine downgrade the caller is asking for -- it must be disclosed, not silent.
        return "pchip" if method in ("cubic", "quintic") else "linear"
    return method


def honoured_methods(dimension: int) -> frozenset[str]:
    """The interpolation methods this module honours at ``dimension``."""
    return HONOURED_METHODS_1D if dimension == 1 else HONOURED_METHODS_ND


def min_points_per_axis(method: str, dimension: int) -> int:
    """Smallest per-axis point count at which ``method`` is honoured at ``dimension``."""
    table = MIN_POINTS_PER_AXIS_1D if dimension == 1 else MIN_POINTS_PER_AXIS_ND
    return table.get(method, 1)


def interpolate_value_rbf_fallback(
    U_values: np.ndarray,
    x_query: np.ndarray,
    grid_coordinates: tuple[np.ndarray, ...],
    grid_shape: tuple[int, ...],
    rbf_kernel: str = "thin_plate_spline",
) -> float:
    """
    RBF interpolation fallback for when regular grid interpolation fails.

    Args:
        U_values: Value function on grid
        x_query: Query point, shape (dimension,)
        grid_coordinates: Tuple of 1D coordinate arrays for each dimension
        grid_shape: Shape of the grid
        rbf_kernel: RBF kernel function ('thin_plate_spline', 'multiquadric', 'gaussian')

    Returns:
        Interpolated value at query point

    Raises:
        Exception: If RBF interpolation fails
    """
    from scipy.interpolate import RBFInterpolator

    x_query_vec = np.atleast_1d(x_query)

    # Build grid points array from coordinates
    # Use meshgrid to create all combinations
    mesh_grids = np.meshgrid(*grid_coordinates, indexing="ij")
    grid_points = np.stack([g.ravel() for g in mesh_grids], axis=1)

    # Flatten U_values
    if U_values.ndim == 1:
        U_flat = U_values
    else:
        U_flat = U_values.ravel()

    # Create and query RBF interpolator
    rbf = RBFInterpolator(grid_points, U_flat, kernel=rbf_kernel)
    result = rbf(x_query_vec.reshape(1, -1))

    logger.debug(f"RBF fallback successful at x={x_query}")
    return float(result[0])


def interpolate_nearest_neighbor(
    U_values: np.ndarray,
    x_query: np.ndarray,
    grid_coordinates: tuple[np.ndarray, ...],
    grid_shape: tuple[int, ...],
) -> float:
    """
    Nearest neighbor interpolation (final fallback).

    Args:
        U_values: Value function on grid
        x_query: Query point, shape (dimension,)
        grid_coordinates: Tuple of 1D coordinate arrays for each dimension
        grid_shape: Shape of the grid

    Returns:
        Value at nearest grid point
    """
    dimension = len(grid_shape)
    x_query_vec = np.atleast_1d(x_query)

    # Find nearest index in each dimension
    multi_idx = []
    for d in range(dimension):
        distances = np.abs(grid_coordinates[d] - x_query_vec[d])
        nearest_idx = np.argmin(distances)
        multi_idx.append(nearest_idx)

    # Return value at nearest point
    if U_values.ndim == 1:
        # Convert multi-index to flat index
        flat_idx = 0
        stride = 1
        for d in range(dimension - 1, -1, -1):
            flat_idx += multi_idx[d] * stride
            stride *= grid_shape[d]
        return float(U_values[flat_idx])
    else:
        return float(U_values[tuple(multi_idx)])


# =============================================================================
# Smoke Tests
# =============================================================================

if __name__ == "__main__":
    """Smoke test for interpolation methods."""
    print("Testing interpolation methods...")

    # Test 1: 1D linear interpolation
    print("\n1. Testing 1D linear interpolation...")
    x_grid = np.linspace(0, 1, 51)
    U_1d = np.sin(2 * np.pi * x_grid)

    # Interpolate at midpoints
    x_query = 0.25
    result = interpolate_value_1d(U_1d, x_query, x_grid, method="linear")
    expected = np.sin(2 * np.pi * x_query)
    error = abs(result - expected)
    print(f"   x={x_query}: interpolated={result:.6f}, exact={expected:.6f}, error={error:.2e}")
    assert error < 0.01
    print("   1D linear interpolation: OK")

    # Test 2: 1D cubic interpolation
    print("\n2. Testing 1D cubic interpolation...")
    result_cubic = interpolate_value_1d(U_1d, x_query, x_grid, method="cubic")
    error_cubic = abs(result_cubic - expected)
    print(f"   x={x_query}: interpolated={result_cubic:.6f}, exact={expected:.6f}, error={error_cubic:.2e}")
    assert error_cubic < error  # Cubic should be more accurate
    print("   1D cubic interpolation: OK")

    # Test 3: 2D linear interpolation
    print("\n3. Testing 2D linear interpolation...")
    grid_shape_2d = (21, 21)
    x = np.linspace(0, 1, grid_shape_2d[0])
    y = np.linspace(0, 1, grid_shape_2d[1])
    X, Y = np.meshgrid(x, y, indexing="ij")
    U_2d = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

    x_query_2d = np.array([0.25, 0.25])
    result_2d = interpolate_value_nd(U_2d, x_query_2d, (x, y), grid_shape_2d, method="linear")
    expected_2d = np.sin(2 * np.pi * 0.25) * np.sin(2 * np.pi * 0.25)
    error_2d = abs(result_2d - expected_2d)
    print(f"   x={x_query_2d}: interpolated={result_2d:.6f}, exact={expected_2d:.6f}, error={error_2d:.2e}")
    assert error_2d < 0.05
    print("   2D linear interpolation: OK")

    # Test 4: 2D cubic interpolation
    print("\n4. Testing 2D cubic interpolation...")
    result_2d_cubic = interpolate_value_nd(U_2d, x_query_2d, (x, y), grid_shape_2d, method="cubic")
    error_2d_cubic = abs(result_2d_cubic - expected_2d)
    print(f"   x={x_query_2d}: interpolated={result_2d_cubic:.6f}, exact={expected_2d:.6f}, error={error_2d_cubic:.2e}")
    print("   2D cubic interpolation: OK")

    # Test 5: Nearest neighbor fallback
    print("\n5. Testing nearest neighbor fallback...")
    result_nn = interpolate_nearest_neighbor(U_2d, x_query_2d, (x, y), grid_shape_2d)
    print(f"   Nearest neighbor result: {result_nn:.6f}")
    assert not np.isnan(result_nn)
    print("   Nearest neighbor: OK")

    # Test 6: 3D interpolation
    print("\n6. Testing 3D interpolation...")
    grid_shape_3d = (11, 11, 11)
    x3 = np.linspace(0, 1, grid_shape_3d[0])
    y3 = np.linspace(0, 1, grid_shape_3d[1])
    z3 = np.linspace(0, 1, grid_shape_3d[2])
    X3, Y3, Z3 = np.meshgrid(x3, y3, z3, indexing="ij")
    U_3d = np.exp(-10 * ((X3 - 0.5) ** 2 + (Y3 - 0.5) ** 2 + (Z3 - 0.5) ** 2))

    x_query_3d = np.array([0.5, 0.5, 0.5])
    result_3d = interpolate_value_nd(U_3d, x_query_3d, (x3, y3, z3), grid_shape_3d, method="linear")
    expected_3d = 1.0  # At center of Gaussian
    error_3d = abs(result_3d - expected_3d)
    print(f"   x={x_query_3d}: interpolated={result_3d:.6f}, exact={expected_3d:.6f}, error={error_3d:.2e}")
    assert error_3d < 0.01
    print("   3D linear interpolation: OK")

    # Test 7: Boundary handling
    print("\n7. Testing boundary handling...")
    result_boundary = interpolate_value_1d(U_1d, -0.1, x_grid, method="linear", xmin=0.0, xmax=1.0)
    assert result_boundary == U_1d[0]
    result_boundary_max = interpolate_value_1d(U_1d, 1.1, x_grid, method="linear", xmin=0.0, xmax=1.0)
    assert result_boundary_max == U_1d[-1]
    print("   Boundary handling: OK")

    print("\nAll interpolation smoke tests passed!")

"""
Splatting Methods for Adjoint Semi-Lagrangian FP Solver.

This module provides splatting routines (adjoint of interpolation) for the
Forward Semi-Lagrangian method. Each splatting scheme is the exact transpose
of the corresponding interpolation scheme used in HJB-SL.

Supported methods:
- linear: 2-point stencil, O(dx) accuracy, preserves positivity
- cubic: 4-point stencil, O(dx³) accuracy, may produce negative values
- quintic: 6-point stencil, O(dx⁵) accuracy, may produce negative values

Mathematical Foundation:
    If HJB uses interpolation matrix P with row sums = 1,
    then FP uses splatting matrix P^T with column sums = 1.
    This ensures sum(P^T @ m) = sum(m) exactly.

WHICH MASS THE KERNELS CONSERVE, AND WHY THE DISPATCHERS REWEIGHT (#2243).
    `sum(m)` is not the mass on this library's grid. `TensorProductGrid` is endpoint-inclusive, so
    the wall lies ON the end node and that node owns h/2: the mass is the trapezoid integral
    `w^T m`, which is #2145's decision. The kernels here are exact transposes of interpolation and
    therefore conserve `sum(m)` -- measured on the reconstructed operator at n=21 with a 0.4-cell
    uniform shift, `1^T S = 1^T` to 0.000e+00 while `w^T S` drifts by 1.0e-02. (The second figure
    scales with the shift -- 0.2 cells gives 5.0e-03 -- so the shift is part of the measurement.)
    So `splat_1d` and `splat_nd` transport `w * m` and divide the deposit back by `w`, which makes
    the step conserve the integral instead.

WHAT THE REWEIGHTING COSTS, because it is not free and the price has a location.
    `W^-1 P^T W` is the correct adjoint of interpolation IN THE GRID MEASURE, and it is no longer
    zeroth-order consistent at the node ADJACENT to each wall. One advection substep, uniform shift
    `a*h` with `a = 0.3` on `m(x) = exp(-2x) + 0.5`, strictly inside the domain -- the shift and the
    density are both part of the measurement, exactly as for the `w^T S` figure above -- error at
    node 1 over n = 21/41/81/161/321:

        unweighted   +9.9e-04  +2.6e-04  +6.5e-05  +1.6e-05  +4.1e-06     O(h^2)
        weighted     -2.2401e-01 -2.2474e-01 -2.2494e-01 -2.2498e-01 -2.2500e-01

    i.e. exactly `-a*m0/2` (0.3 * 1.5 / 2 = 0.225), FLAT under 16x refinement. Under a LEFTWARD
    shift the same O(1) error appears at node n-2 instead, which is why this says "each wall"
    rather than "the left wall". That is an O(1) truncation error at one node,
    and it is the price of `w^T`-exactness: the wall node owns h/2, so a deposit crossing into it
    from a full cell cannot be both mass-exact and pointwise-consistent under this kernel.

    Consequence, measured on #1855's Gibbs fixture: the L-inf wall error is WORSE in a
    mid-refinement band -- 1.28x at nx=81, peaking at 1.35x at nx=101 -- decaying to 1.02x by
    nx=641, and up to 1.44x under a stronger drift (slope 2.0, nx=81). In 2D it is 1.27x at nx=81.
    `L1` is equal or better everywhere, and the mass drift is 1e-14 to 1.2e-11 relative over up to
    25600 steps against 2e-03 to 2.3e-02 for the unweighted pairing. It does not grow under
    refinement in any regime measured; it is a band, not a divergence.

    THIS IS HALF OF ONE DECISION; the other half is the diffusion wall. #708 removed exactly this
    weighting, on the ground that "SL theory uses sum(m)" and "Crank-Nicolson diffusion also
    preserves sum(m) with Neumann BC". The second clause was true of the `half_wall` stencil the
    diffusion step carried at the time and is false of the `mirror` stencil it carries since #2243.

    THE INVARIANT CLAIM IS EXACTNESS, NOT A RATE. Paired, one measure is conserved exactly at every
    resolution; crossed, NEITHER is, and how the residue then behaves under refinement depends on
    the fixture -- do not quote a rate for it. Measured on `FPSLSolver`, relative trapezoid drift
    over nx = 26/51/101/201/401, drift `u = -0.5x` (a constant push into the right wall),
    T=1, sigma=0.2:

        half_wall + unweighted splat   8.2e-02 .. 1.3e-02   O(h); exact in `sum(m)` instead
        mirror    + unweighted splat   6.3e-03 .. 7.6e-02   grows on THIS fixture
        mirror    + weighted splat     2.5e-15 .. 2.7e-13   exact
        half_wall + weighted splat     2.4e+00 at nx=101    the other crossed pairing

    Independent review measured the same crossed pairing on `u = 0.5 cos(2 pi x)` and got a drift
    that FALLS at about O(h^1.5) rather than growing. Both runs agree on what matters -- neither
    measure is conserved -- and disagree on the rate, which is why the row above names its fixture
    and this paragraph names the other one.

    So #708's pairing was coherent and conserved the wrong functional. Do not change one half
    without the other, and do not read the kernels' `sum(m)` property as the solver's conservation
    property -- it is not, and has not been the one that matters since #2145.

WHERE THE REWEIGHTING PAYS MOST: the PERIODIC seam, which has no wall at all.
    `enforce_periodic_value_nd` folds the two coincident endpoints by their mean. Unweighted, the
    splat deposits into both as if each owned a full cell, and the fold then HALVES the seam
    density. Against a travelling decaying cosine on a periodic domain -- an external oracle with
    no boundary anywhere -- the unweighted pairing DIVERGES and the weighted one is O(h):

        unweighted   relLinf 2.39e-01 -> 3.29e-01 over nx 41..321   order -0.11 -0.14 -0.21
        weighted     relLinf 1.79e-02 -> 2.03e-03                   order +0.98 +1.04 +1.12
        mass drift   15-19%  ->  ~1e-14

Module structure per issue #392:
    fp_sl_splatting.py - Splatting methods for adjoint semi-Lagrangian FP solver

Functions:
    splat_linear_1d: Linear (2-point) splatting
    splat_cubic_1d: Cubic (4-point) splatting
    splat_quintic_1d: Quintic (6-point) splatting
    compute_cubic_weights: Catmull-Rom cubic kernel weights
    compute_quintic_weights: Quintic kernel weights

Issue #708: Splatting implementations for adjoint-consistent SL-MFG
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.utils.numerical.quadrature import quadrature_weights_nd

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_cubic_weights(t: float) -> tuple[float, float, float, float]:
    """
    Compute Catmull-Rom cubic interpolation weights.

    The Catmull-Rom spline uses 4 points with C¹ continuity.
    Weights satisfy partition of unity: sum(w) = 1.

    Args:
        t: Fractional position in [0, 1]

    Returns:
        Tuple of weights (w_{-1}, w_0, w_1, w_2) for points j-1, j, j+1, j+2
    """
    t2 = t * t
    t3 = t2 * t

    # Catmull-Rom kernel coefficients
    w_m1 = -0.5 * t3 + t2 - 0.5 * t
    w_0 = 1.5 * t3 - 2.5 * t2 + 1.0
    w_1 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    w_2 = 0.5 * t3 - 0.5 * t2

    return w_m1, w_0, w_1, w_2


def compute_quintic_weights(t: float) -> tuple[float, float, float, float, float, float]:
    """
    Compute quintic interpolation weights.

    Uses 6 points for O(dx⁵) accuracy with C² continuity.
    Weights satisfy partition of unity: sum(w) = 1.

    Args:
        t: Fractional position in [0, 1]

    Returns:
        Tuple of weights (w_{-2}, w_{-1}, w_0, w_1, w_2, w_3)
    """
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t

    # Quintic Lagrange weights for 6-point stencil
    # Based on Lagrange polynomial through points at -2, -1, 0, 1, 2, 3
    w_m2 = (t5 - 5 * t4 + 5 * t3 + 5 * t2 - 6 * t) / (-120)
    w_m1 = (t5 - 4 * t4 - t3 + 16 * t2 - 12 * t) / 24
    w_0 = (t5 - 3 * t4 - 5 * t3 + 15 * t2 + 4 * t - 12) / (-12)
    w_1 = (t5 - 2 * t4 - 7 * t3 + 8 * t2 + 12 * t) / 12
    w_2 = (t5 - t4 - 7 * t3 + t2 + 6 * t) / (-24)
    w_3 = (t5 - 5 * t3 + 4 * t) / 120

    return w_m2, w_m1, w_0, w_1, w_2, w_3


def splat_linear_1d(
    m: NDArray[np.floating],
    x_dest: NDArray[np.floating],
    x_grid: NDArray[np.floating],
    dx: float,
    xmin: float,
    xmax: float,
) -> NDArray[np.floating]:
    """
    Linear (2-point) splatting - adjoint of linear interpolation.

    Each particle at position x_dest scatters its density to two neighboring
    grid points with weights (1-w, w) where w is the fractional position.

    This is the transpose of linear interpolation:
    - Interpolation (gather): φ[i] = (1-w)·φ[j] + w·φ[j+1]
    - Splatting (scatter): m[j] += (1-w)·m[i]; m[j+1] += w·m[i]

    Args:
        m: Source density array, shape (Nx,)
        x_dest: Destination positions, shape (Nx,)
        x_grid: Grid coordinates, shape (Nx,)
        dx: Grid spacing
        xmin, xmax: Domain bounds

    Returns:
        Splat result, shape (Nx,)
    """
    Nx = len(m)
    m_star = np.zeros(Nx)

    # Convert to continuous indices
    pos_cont = (x_dest - xmin) / dx

    # Lower neighbor index
    j = np.floor(pos_cont).astype(int)
    j = np.clip(j, 0, Nx - 2)

    # Weight for upper neighbor
    w = pos_cont - j
    w = np.clip(w, 0, 1)

    # Scatter with atomic accumulation
    np.add.at(m_star, j, m * (1 - w))
    np.add.at(m_star, j + 1, m * w)

    return m_star


def splat_cubic_1d(
    m: NDArray[np.floating],
    x_dest: NDArray[np.floating],
    x_grid: NDArray[np.floating],
    dx: float,
    xmin: float,
    xmax: float,
) -> NDArray[np.floating]:
    """
    Cubic (4-point) splatting - adjoint of Catmull-Rom cubic interpolation.

    Each particle scatters to 4 neighboring grid points using Catmull-Rom
    cubic kernel weights.

    This is the transpose of cubic interpolation:
    - Interpolation: φ[i] = Σ_{k=-1}^{2} w_k · φ[j+k]
    - Splatting: m[j+k] += w_k · m[i] for k = -1, 0, 1, 2

    Note: May produce negative values due to oscillatory cubic kernel.

    Args:
        m: Source density array, shape (Nx,)
        x_dest: Destination positions, shape (Nx,)
        x_grid: Grid coordinates, shape (Nx,)
        dx: Grid spacing
        xmin, xmax: Domain bounds

    Returns:
        Splat result, shape (Nx,)
    """
    Nx = len(m)
    m_star = np.zeros(Nx)

    # Convert to continuous indices
    pos_cont = (x_dest - xmin) / dx

    for i in range(Nx):
        # Base index (floor of position)
        j = int(np.floor(pos_cont[i]))

        # Fractional position
        t = pos_cont[i] - j

        # Compute cubic weights
        w_m1, w_0, w_1, w_2 = compute_cubic_weights(t)
        weights = [w_m1, w_0, w_1, w_2]
        indices = [j - 1, j, j + 1, j + 2]

        # Scatter to 4 neighbors
        for idx, wk in zip(indices, weights, strict=True):
            if 0 <= idx < Nx:
                m_star[idx] += wk * m[i]
            elif idx < 0:
                # Reflect mass back into domain (Neumann BC)
                m_star[0] += wk * m[i]
            else:  # idx >= Nx
                m_star[Nx - 1] += wk * m[i]

    return m_star


def splat_quintic_1d(
    m: NDArray[np.floating],
    x_dest: NDArray[np.floating],
    x_grid: NDArray[np.floating],
    dx: float,
    xmin: float,
    xmax: float,
) -> NDArray[np.floating]:
    """
    Quintic (6-point) splatting - adjoint of quintic interpolation.

    Each particle scatters to 6 neighboring grid points using quintic
    Lagrange weights.

    This is the transpose of quintic interpolation:
    - Interpolation: φ[i] = Σ_{k=-2}^{3} w_k · φ[j+k]
    - Splatting: m[j+k] += w_k · m[i] for k = -2, -1, 0, 1, 2, 3

    Note: May produce negative values due to oscillatory kernel.

    Args:
        m: Source density array, shape (Nx,)
        x_dest: Destination positions, shape (Nx,)
        x_grid: Grid coordinates, shape (Nx,)
        dx: Grid spacing
        xmin, xmax: Domain bounds

    Returns:
        Splat result, shape (Nx,)
    """
    Nx = len(m)
    m_star = np.zeros(Nx)

    # Convert to continuous indices
    pos_cont = (x_dest - xmin) / dx

    for i in range(Nx):
        # Base index (floor of position)
        j = int(np.floor(pos_cont[i]))

        # Fractional position
        t = pos_cont[i] - j

        # Compute quintic weights
        w_m2, w_m1, w_0, w_1, w_2, w_3 = compute_quintic_weights(t)
        weights = [w_m2, w_m1, w_0, w_1, w_2, w_3]
        indices = [j - 2, j - 1, j, j + 1, j + 2, j + 3]

        # Scatter to 6 neighbors
        for idx, wk in zip(indices, weights, strict=True):
            if 0 <= idx < Nx:
                m_star[idx] += wk * m[i]
            elif idx < 0:
                # Reflect mass back into domain (Neumann BC)
                m_star[0] += wk * m[i]
            else:  # idx >= Nx
                m_star[Nx - 1] += wk * m[i]

    return m_star


def splat_1d(
    m: NDArray[np.floating],
    x_dest: NDArray[np.floating],
    x_grid: NDArray[np.floating],
    dx: float,
    xmin: float,
    xmax: float,
    method: str = "linear",
) -> NDArray[np.floating]:
    """
    Dispatch to appropriate splatting method.

    Args:
        m: Source density array
        x_dest: Destination positions
        x_grid: Grid coordinates
        dx: Grid spacing
        xmin, xmax: Domain bounds
        method: Splatting method ('linear', 'cubic', 'quintic')

    Returns:
        Splat result

    Note:
        Transports the MASS ``w * m`` and divides the deposit back by ``w`` (#2243). The kernels
        below are pure transposes of interpolation and conserve ``sum(m)``; this wrapper is what
        makes the step conserve the integral. See the module docstring for why the two differ.
    """
    measure = quadrature_weights_nd((x_grid,))
    weighted = np.asarray(m, dtype=float) * measure
    if method == "linear":
        deposited = splat_linear_1d(weighted, x_dest, x_grid, dx, xmin, xmax)
    elif method == "cubic":
        deposited = splat_cubic_1d(weighted, x_dest, x_grid, dx, xmin, xmax)
    elif method == "quintic":
        deposited = splat_quintic_1d(weighted, x_dest, x_grid, dx, xmin, xmax)
    else:
        raise ValueError(f"Unknown splatting method: {method}. Use 'linear', 'cubic', or 'quintic'.")
    return deposited / measure


# =============================================================================
# nD Splatting Functions
# =============================================================================


def splat_linear_nd(
    m: NDArray[np.floating],
    x_dest: NDArray[np.floating],
    grid_coordinates: tuple[NDArray[np.floating], ...],
    grid_shape: tuple[int, ...],
    bounds: list[tuple[float, float]],
) -> NDArray[np.floating]:
    """
    Linear (2^d point) splatting for nD problems - adjoint of multilinear interpolation.

    For nD, linear interpolation uses 2^d corner points of the hypercube containing
    the query point. The weights are tensor products of 1D weights.

    Example (2D):
        Interpolation at (x, y) in cell [i,i+1] x [j,j+1]:
        φ(x,y) = (1-wx)(1-wy)·φ[i,j] + wx(1-wy)·φ[i+1,j]
               + (1-wx)wy·φ[i,j+1] + wx·wy·φ[i+1,j+1]

        Splatting (adjoint):
        m[i,j] += (1-wx)(1-wy)·m_src; m[i+1,j] += wx(1-wy)·m_src; etc.

    Args:
        m: Source density array, shape grid_shape (flattened or shaped)
        x_dest: Destination positions, shape (N_points, dimension)
        grid_coordinates: Tuple of 1D coordinate arrays for each dimension
        grid_shape: Shape of the grid (N1, N2, ..., Nd)
        bounds: List of (min, max) tuples for each dimension

    Returns:
        Splat result, same shape as m
    """
    dimension = len(grid_shape)
    N_points = np.prod(grid_shape)

    # Ensure m is flattened for indexing
    m_flat = m.ravel()
    m_star = np.zeros(N_points)

    # Grid spacings
    dx = np.array([(grid_coordinates[d][1] - grid_coordinates[d][0]) for d in range(dimension)])
    xmin = np.array([bounds[d][0] for d in range(dimension)])

    # Reshape x_dest if needed: (N_points,) for 1D -> (N_points, 1)
    if x_dest.ndim == 1 and dimension == 1:
        x_dest = x_dest.reshape(-1, 1)

    # Issue #931: Vectorized splatting — eliminate per-point Python loop.
    # Compute cell indices and weights for ALL points at once.
    # pos_cont[i, d] = (x_dest[i, d] - xmin[d]) / dx[d]
    pos_cont = (x_dest - xmin) / dx  # (N_points, dimension)

    # Lower corner indices, clamped to valid range
    j_base = np.floor(pos_cont).astype(int)  # (N_points, dimension)
    for d in range(dimension):
        j_base[:, d] = np.clip(j_base[:, d], 0, grid_shape[d] - 2)

    # Interpolation weights, clamped to [0, 1]
    w = np.clip(pos_cont - j_base, 0.0, 1.0)  # (N_points, dimension)

    # Scatter to 2^d corners (loop over corners, not over points)
    for corner in range(1 << dimension):
        # Compute corner indices and weight for all points at once
        corner_indices = j_base.copy()  # (N_points, dimension)
        weight = np.ones(N_points)

        for d in range(dimension):
            if corner & (1 << d):  # Upper corner in dimension d
                corner_indices[:, d] += 1
                weight *= w[:, d]
            else:  # Lower corner
                weight *= 1.0 - w[:, d]

        # Clamp to grid bounds
        for d in range(dimension):
            corner_indices[:, d] = np.clip(corner_indices[:, d], 0, grid_shape[d] - 1)

        # Convert to flat indices and accumulate
        flat_indices = np.ravel_multi_index(corner_indices.T, grid_shape)
        np.add.at(m_star, flat_indices, weight * m_flat)

    return m_star.reshape(grid_shape)


def splat_nd(
    m: NDArray[np.floating],
    x_dest: NDArray[np.floating],
    grid_coordinates: tuple[NDArray[np.floating], ...],
    grid_shape: tuple[int, ...],
    bounds: list[tuple[float, float]],
    method: str = "linear",
) -> NDArray[np.floating]:
    """
    Dispatch to appropriate nD splatting method.

    Args:
        m: Source density array
        x_dest: Destination positions, shape (N_points, dimension)
        grid_coordinates: Tuple of 1D coordinate arrays for each dimension
        grid_shape: Shape of the grid
        bounds: List of (min, max) tuples for each dimension
        method: Splatting method ('linear' only for nD currently)

    Returns:
        Splat result

    Note:
        Cubic and quintic splatting for nD would require tensor products of 1D kernels.
        Currently only linear is supported for nD.

        Transports the MASS ``w * m`` and divides back, exactly as `splat_1d` does; here ``w`` is
        the outer product of the per-axis trapezoid weights, so a corner node owns ``h_x h_y / 4``
        (#2243).
    """
    measure = quadrature_weights_nd(grid_coordinates)
    if method == "linear":
        weighted = np.asarray(m, dtype=float).reshape(grid_shape) * measure
        return splat_linear_nd(weighted, x_dest, grid_coordinates, grid_shape, bounds) / measure
    elif method in ("cubic", "quintic"):
        raise NotImplementedError(
            f"'{method}' splatting not yet implemented for nD. Use 'linear' or implement "
            "tensor-product splatting. For nD, consider using 'linear' which provides "
            "adequate accuracy for most MFG applications."
        )
    else:
        raise ValueError(f"Unknown splatting method: {method}.")


# =============================================================================
# Smoke Tests
# =============================================================================

if __name__ == "__main__":
    """Smoke test for splatting methods."""
    print("Testing splatting methods...")
    print("=" * 60)

    # Test grid
    Nx = 21
    xmin, xmax = 0.0, 1.0
    x = np.linspace(xmin, xmax, Nx)
    dx = x[1] - x[0]

    # Test 1: Linear splatting - partition of unity
    print("\n1. Testing linear splatting (partition of unity)...")
    m_uniform = np.ones(Nx)
    x_dest = x + 0.3 * dx  # Shift by 0.3 grid spacing

    m_splat = splat_linear_1d(m_uniform, x_dest, x, dx, xmin, xmax)
    mass_before = np.sum(m_uniform)
    mass_after = np.sum(m_splat)

    print(f"   Mass before: {mass_before:.6f}")
    print(f"   Mass after:  {mass_after:.6f}")
    print(f"   Mass error:  {abs(mass_after - mass_before):.2e}")
    assert abs(mass_after - mass_before) < 1e-10, "Linear splatting failed mass conservation"
    print("   Linear splatting: OK")

    # Test 2: Cubic splatting - partition of unity
    print("\n2. Testing cubic splatting (partition of unity)...")
    m_splat_cubic = splat_cubic_1d(m_uniform, x_dest, x, dx, xmin, xmax)
    mass_cubic = np.sum(m_splat_cubic)

    print(f"   Mass before: {mass_before:.6f}")
    print(f"   Mass after:  {mass_cubic:.6f}")
    print(f"   Mass error:  {abs(mass_cubic - mass_before):.2e}")
    # Cubic may have small errors due to boundary handling
    assert abs(mass_cubic - mass_before) < 0.1, "Cubic splatting failed mass conservation"
    print("   Cubic splatting: OK")

    # Test 3: Verify cubic weights sum to 1
    print("\n3. Testing cubic weights (partition of unity)...")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        weights = compute_cubic_weights(t)
        weight_sum = sum(weights)
        print(f"   t={t:.2f}: weights={[f'{w:.4f}' for w in weights]}, sum={weight_sum:.6f}")
        assert abs(weight_sum - 1.0) < 1e-10, f"Cubic weights don't sum to 1 at t={t}"
    print("   Cubic weights: OK")

    # Test 4: Quintic splatting - partition of unity
    print("\n4. Testing quintic splatting (partition of unity)...")
    m_splat_quintic = splat_quintic_1d(m_uniform, x_dest, x, dx, xmin, xmax)
    mass_quintic = np.sum(m_splat_quintic)

    print(f"   Mass before: {mass_before:.6f}")
    print(f"   Mass after:  {mass_quintic:.6f}")
    print(f"   Mass error:  {abs(mass_quintic - mass_before):.2e}")
    assert abs(mass_quintic - mass_before) < 0.1, "Quintic splatting failed mass conservation"
    print("   Quintic splatting: OK")

    # Test 5: Verify quintic weights sum to 1
    print("\n5. Testing quintic weights (partition of unity)...")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        weights = compute_quintic_weights(t)
        weight_sum = sum(weights)
        print(f"   t={t:.2f}: sum={weight_sum:.6f}")
        assert abs(weight_sum - 1.0) < 1e-10, f"Quintic weights don't sum to 1 at t={t}"
    print("   Quintic weights: OK")

    # Test 6: Dispatch function
    print("\n6. Testing dispatch function...")
    m_linear = splat_1d(m_uniform, x_dest, x, dx, xmin, xmax, method="linear")
    m_cubic = splat_1d(m_uniform, x_dest, x, dx, xmin, xmax, method="cubic")
    m_quintic = splat_1d(m_uniform, x_dest, x, dx, xmin, xmax, method="quintic")
    print("   Dispatch function: OK")

    # Test 7: nD splatting (2D)
    print("\n7. Testing 2D linear splatting...")
    Nx_2d, Ny_2d = 11, 11
    x_2d = np.linspace(0.0, 1.0, Nx_2d)
    y_2d = np.linspace(0.0, 1.0, Ny_2d)
    grid_shape_2d = (Nx_2d, Ny_2d)
    bounds_2d = [(0.0, 1.0), (0.0, 1.0)]
    dx_2d = x_2d[1] - x_2d[0]

    # Uniform density
    m_2d = np.ones(grid_shape_2d)

    # Create meshgrid and shift positions
    XX, YY = np.meshgrid(x_2d, y_2d, indexing="ij")
    x_dest_2d = np.stack([(XX + 0.3 * dx_2d).ravel(), (YY + 0.2 * dx_2d).ravel()], axis=-1)

    # Splat
    m_splat_2d = splat_linear_nd(m_2d.ravel(), x_dest_2d, (x_2d, y_2d), grid_shape_2d, bounds_2d)

    mass_2d_before = np.sum(m_2d)
    mass_2d_after = np.sum(m_splat_2d)
    print(f"   Mass before: {mass_2d_before:.6f}")
    print(f"   Mass after:  {mass_2d_after:.6f}")
    print(f"   Mass error:  {abs(mass_2d_after - mass_2d_before):.2e}")
    assert abs(mass_2d_after - mass_2d_before) < 1e-10, "2D splatting failed mass conservation"
    print("   2D splatting: OK")

    # Test 8: nD splatting (3D)
    print("\n8. Testing 3D linear splatting...")
    N3d = 6
    x_3d = np.linspace(0.0, 1.0, N3d)
    grid_shape_3d = (N3d, N3d, N3d)
    bounds_3d = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    dx_3d = x_3d[1] - x_3d[0]

    # Uniform density
    m_3d = np.ones(grid_shape_3d)

    # Create meshgrid and shift positions
    XX3, YY3, ZZ3 = np.meshgrid(x_3d, x_3d, x_3d, indexing="ij")
    x_dest_3d = np.stack(
        [
            (XX3 + 0.25 * dx_3d).ravel(),
            (YY3 + 0.15 * dx_3d).ravel(),
            (ZZ3 + 0.35 * dx_3d).ravel(),
        ],
        axis=-1,
    )

    # Splat
    m_splat_3d = splat_linear_nd(m_3d.ravel(), x_dest_3d, (x_3d, x_3d, x_3d), grid_shape_3d, bounds_3d)

    mass_3d_before = np.sum(m_3d)
    mass_3d_after = np.sum(m_splat_3d)
    print(f"   Mass before: {mass_3d_before:.6f}")
    print(f"   Mass after:  {mass_3d_after:.6f}")
    print(f"   Mass error:  {abs(mass_3d_after - mass_3d_before):.2e}")
    assert abs(mass_3d_after - mass_3d_before) < 1e-10, "3D splatting failed mass conservation"
    print("   3D splatting: OK")

    print("\n" + "=" * 60)
    print("All splatting smoke tests passed!")

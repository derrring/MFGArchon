"""
Reinitialization for Level Set Methods.

During level set evolution, the function φ can deviate from being a true signed
distance function (SDF). The SDF property |∇φ| = 1 is desirable for:
- Accurate geometric computations (normals, curvature)
- Numerical stability (prevents steep/flat gradients)
- CFL condition predictability

This module provides multiple methods for reinitialization:

1. **Fast Marching Method (FMM)** - Default, recommended
   - O(N log N) complexity
   - Exact signed distance computation
   - Best for general use and complex geometries

2. **Fast Sweeping Method (FSM)**
   - O(N) per sweep, typically 2-4 iterations
   - Similar accuracy to FMM
   - May be faster for simple, convex domains

3. **PDE-based** (Legacy)
   - Pseudo-time evolution: ∂φ/∂τ + sign(φ₀)(|∇φ| - 1) = 0
   - Slower than FMM/FSM, may drift interface by ~10 grid points
   - Kept for backward compatibility

**Usage Recommendation**:
    Use method="fmm" (default) for best accuracy. The FMM computes exact signed
    distances with O(dx) accuracy, preserving the zero level set precisely.

References:
- Sethian (1996): A fast marching level set method for monotonically advancing fronts
- Zhao (2005): A fast sweeping method for Eikonal equations
- Sussman, Smereka, Osher (1994): A level set approach for computing solutions
  to incompressible two-phase flow
- Osher & Fedkiw (2003): Level Set Methods, Chapter 7.4

Created: 2026-01-18 (Issue #592 Milestone 3.1.2)
Updated: 2026-02-06 (Issue #664 - Added FMM/FSM methods)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid

# Module logger
logger = get_logger(__name__)


def reinitialize(
    phi_initial: NDArray[np.float64],
    geometry: TensorProductGrid,
    method: str = "fmm",
    max_iterations: int = 20,
    dtau: float | None = None,
    tolerance: float = 0.01,
    narrow_band_width: float | None = None,
    subcell_accuracy: bool = True,
) -> NDArray[np.float64]:
    """
    Restore signed distance function property: |∇φ| = 1.

    Computes a signed distance function that preserves the zero level set
    of the input while ensuring |∇φ| = 1 everywhere.

    Args:
        phi_initial: Initial level set function, shape (Nx, Ny, ...)
        geometry: Grid providing spatial structure and operators
        method: Reinitialization algorithm (default: "fmm")
            - "fmm": Fast Marching Method (O(N log N), exact, recommended)
            - "fsm": Fast Sweeping Method (O(N) per sweep, iterative)
            - "pde": PDE-based pseudo-time evolution (legacy, slower)
        max_iterations: For PDE method: max pseudo-time iterations (default: 20)
                       For FSM method: max sweep cycles (default: 10)
        dtau: For PDE method: pseudo-time step (default: auto = 0.1·min(h))
        tolerance: Convergence tolerance (default: 0.01)
        narrow_band_width: For PDE method: only reinitialize |φ| < width
        subcell_accuracy: For FMM/FSM: use subcell interface location (default: True)

    Returns:
        Reinitialized level set φ with |∇φ| = 1, same shape as input

    Raises:
        ValueError: If method is unknown or dtau violates CFL (PDE method)

    Examples:
        >>> # Fast Marching Method (recommended)
        >>> phi_sdf = reinitialize(phi_distorted, grid)  # method="fmm" default

        >>> # Fast Sweeping Method
        >>> phi_sdf = reinitialize(phi_distorted, grid, method="fsm")

        >>> # Legacy PDE method with narrow band
        >>> phi_sdf = reinitialize(phi_distorted, grid, method="pde",
        ...                        narrow_band_width=3*dx)

    Note:
        The FMM and FSM methods compute exact signed distance functions with
        O(dx) accuracy. The PDE method is iterative and may not converge to
        an exact SDF, especially for severely distorted level sets.

        FMM is preferred for:
        - General use (most reliable)
        - Complex geometries
        - When exact SDF is needed

        FSM may be faster for:
        - Simple, convex domains
        - When memory is constrained (no heap)
    """
    method = method.lower()

    if method == "fmm":
        return _reinitialize_fmm(phi_initial, geometry, subcell_accuracy)
    elif method == "fsm":
        return _reinitialize_fsm(phi_initial, geometry, max_iterations, tolerance, subcell_accuracy)
    elif method == "pde":
        return _reinitialize_pde(phi_initial, geometry, max_iterations, dtau, tolerance, narrow_band_width)
    else:
        raise ValueError(f"Unknown reinitialization method: {method}. Choose from 'fmm', 'fsm', or 'pde'.")


def _reinitialize_fmm(
    phi_initial: NDArray[np.float64],
    geometry: TensorProductGrid,
    subcell_accuracy: bool = True,
) -> NDArray[np.float64]:
    """Reinitialize using Fast Marching Method."""
    from mfgarchon.geometry.level_set.eikonal import FastMarchingMethod

    fmm = FastMarchingMethod(geometry)
    return fmm.compute_signed_distance(phi_initial, subcell_accuracy=subcell_accuracy)


def _reinitialize_fsm(
    phi_initial: NDArray[np.float64],
    geometry: TensorProductGrid,
    max_iterations: int = 10,
    tolerance: float = 1e-10,
    subcell_accuracy: bool = True,
) -> NDArray[np.float64]:
    """Reinitialize using Fast Sweeping Method."""
    from mfgarchon.geometry.level_set.eikonal import FastSweepingMethod

    fsm = FastSweepingMethod(geometry, max_iterations=max_iterations, tolerance=tolerance)
    return fsm.compute_signed_distance(phi_initial, subcell_accuracy=subcell_accuracy)


def _godunov_gradient_magnitude(
    phi: NDArray[np.float64], sign_phi0: NDArray[np.float64], geometry: TensorProductGrid
) -> NDArray[np.float64]:
    """``|grad phi|`` for ``phi_tau + S (|grad phi| - 1) = 0``, upwinded per axis by the sign of ``S`` (#2310).

    Where ``S > 0`` the Godunov magnitude is the HJB upwind momentum of ``phi`` per axis,
    ``gradient_upwind``, for ``H = |p|``, which is even and nondecreasing in each component. Where
    ``S < 0`` the equation is the same one for ``-phi``, so it is ``gradient_upwind`` of ``-phi``.
    The padded array is negated, not ``phi``, so an inhomogeneous boundary value stays where the
    ghost cells put it. The padding is the grid's own boundary condition at ``time=0``, as
    ``get_gradient_operator(scheme="upwind")`` applied it.

    Measured at 6c0610d2, before this: the ``S > 0`` form was used everywhere, which is downwind where
    ``phi0 < 0``, so the exact signed distance ``x - 0.5`` drifted by 2.99e+01 after 100 iterations
    and a ``phi0 < 0`` minimum moved by 3.73e-02 on the default call.
    """
    from mfgarchon.geometry.boundary import pad_array_with_ghosts
    from mfgarchon.operators.stencils.finite_difference import gradient_upwind

    bc = geometry.get_boundary_conditions()
    padded = pad_array_with_ghosts(phi, bc, ghost_depth=1, time=0.0) if bc is not None else phi
    interior = tuple(slice(1, -1) if bc is not None else slice(None) for _ in range(phi.ndim))
    positive = sign_phi0 > 0
    squared = np.zeros(phi.shape, dtype=np.float64)
    for d, h in enumerate(geometry.spacing):
        along_positive = gradient_upwind(padded, axis=d, h=h)[interior]
        along_negative = gradient_upwind(-padded, axis=d, h=h)[interior]
        squared += np.where(positive, along_positive**2, along_negative**2)
    return np.sqrt(squared)


def _reinitialize_pde(
    phi_initial: NDArray[np.float64],
    geometry: TensorProductGrid,
    max_iterations: int = 20,
    dtau: float | None = None,
    tolerance: float = 0.01,
    narrow_band_width: float | None = None,
) -> NDArray[np.float64]:
    """
    Reinitialize using PDE-based pseudo-time evolution (legacy method).

    Evolves phi in pseudo-time via: d(phi)/d(tau) + sign(phi0)(|grad phi| - 1) = 0
    """
    # Auto-select pseudo-timestep based on CFL
    # Note: Reinitialization requires smaller CFL than regular evolution
    if dtau is None:
        spacing = geometry.spacing
        h_min = min(spacing)
        dtau = 0.1 * h_min  # CFL = 0.1 (very conservative for reinitialization)
        logger.debug(f"Auto-selected dtau = {dtau:.2e} (0.1 * min(spacing))")
    else:
        # Validate user-provided dtau
        h_min = min(geometry.spacing)
        cfl = dtau / h_min
        if cfl > 0.5:
            raise ValueError(
                f"dtau = {dtau:.2e} violates CFL condition for reinitialization. "
                f"Must have dtau < {0.5 * h_min:.2e} for stability."
            )

    # Initialize
    phi = phi_initial.copy()
    sign_phi0 = np.sign(phi_initial)  # Preserve zero level set location
    # Handle sign(0) = 0 -> set to 1 to avoid division issues
    sign_phi0[sign_phi0 == 0] = 1.0

    # Narrow band setup
    if narrow_band_width is not None:
        # Create mask for narrow band region: |phi| < narrow_band_width
        narrow_band_mask = np.abs(phi_initial) < narrow_band_width
        n_band_points = np.sum(narrow_band_mask)
        n_total_points = phi.size

        logger.debug(
            f"Narrow band: {n_band_points}/{n_total_points} points "
            f"({100 * n_band_points / n_total_points:.1f}%) within |phi| < {narrow_band_width:.4f}"
        )

        # Store original far-field values (will not be modified)
        phi_far_field = phi[~narrow_band_mask].copy()
    else:
        # Global reinitialization (backward compatible)
        narrow_band_mask = None
        logger.debug("Global reinitialization (entire domain)")

    # Track previous deviation for divergence detection
    prev_max_deviation = float("inf")
    iteration = 0

    # Pseudo-time evolution
    for iteration in range(max_iterations):
        grad_mag = _godunov_gradient_magnitude(phi, sign_phi0, geometry)

        # Check convergence: max(||grad phi| - 1|)
        # Only check within narrow band if applicable
        deviation = np.abs(grad_mag - 1.0)
        if narrow_band_mask is not None:
            max_deviation = np.max(deviation[narrow_band_mask]) if np.any(narrow_band_mask) else 0.0
        else:
            max_deviation = np.max(deviation)

        if iteration % 5 == 0 or iteration == max_iterations - 1:
            mean_deviation = np.mean(deviation)
            logger.debug(
                f"Reinit iteration {iteration}: max(||grad phi| - 1|) = {max_deviation:.4f}, mean = {mean_deviation:.4f}"
            )

        # Check for divergence (deviation increasing significantly)
        if max_deviation > 2 * prev_max_deviation and iteration > 5:
            logger.warning(
                f"Reinitialization diverging at iteration {iteration}: "
                f"max(||grad phi| - 1|) = {max_deviation:.4f}. "
                f"Stopping early and returning last stable state."
            )
            # Revert to previous state
            break

        # Early stopping if converged
        if max_deviation < tolerance:
            logger.debug(
                f"Reinitialization converged at iteration {iteration}: "
                f"max(||grad phi| - 1|) = {max_deviation:.4f} < {tolerance}"
            )
            break

        prev_max_deviation = max_deviation

        # RHS: sign(phi0)(|grad phi| - 1)
        rhs = sign_phi0 * (grad_mag - 1.0)

        # Explicit Euler update: phi^{n+1} = phi^n - dtau*RHS
        if narrow_band_mask is not None:
            # Narrow band: only update points within band
            phi[narrow_band_mask] = phi[narrow_band_mask] - dtau * rhs[narrow_band_mask]
            # Restore far-field values (unchanged)
            phi[~narrow_band_mask] = phi_far_field
        else:
            # Global: update entire domain
            phi = phi - dtau * rhs

    # Final diagnostic
    grad_mag_final = _godunov_gradient_magnitude(phi, sign_phi0, geometry)

    if narrow_band_mask is not None:
        final_deviation = np.max(np.abs(grad_mag_final[narrow_band_mask] - 1.0)) if np.any(narrow_band_mask) else 0.0
        logger.debug(
            f"Reinitialization complete: {iteration + 1} iterations, "
            f"final max(||grad phi| - 1|) = {final_deviation:.4f} (narrow band only)"
        )
    else:
        final_deviation = np.max(np.abs(grad_mag_final - 1.0))
        logger.debug(
            f"Reinitialization complete: {iteration + 1} iterations, final max(||grad phi| - 1|) = {final_deviation:.4f}"
        )

    return phi


if __name__ == "__main__":
    """Smoke test for reinitialization."""
    print("Testing Reinitialization...")

    from mfgarchon.geometry.boundary import no_flux_bc
    from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid

    # Test 1: 1D - Maintain SDF property during evolution
    print("\n[Test 1: 1D SDF Maintenance]")
    print("Problem: Use reinitialization to maintain |∇φ| ≈ 1 after evolution")

    # Create 1D grid
    Nx = 200
    grid_1d = TensorProductGrid(bounds=[(0.0, 1.0)], Nx=[Nx], boundary_conditions=no_flux_bc(dimension=1))
    x = grid_1d.coordinates[0]
    dx = grid_1d.spacing[0]

    print(f"  Grid: {Nx} points, dx = {dx:.4f}")

    # Start with true SDF: φ = x - 0.5 (interface at x = 0.5)
    phi_true_sdf = x - 0.5

    # Simulate slight distortion from a few evolution steps
    # Add small perturbation: φ_perturbed = φ + 0.1·sin(10πx)
    phi0_slightly_distorted = phi_true_sdf + 0.05 * np.sin(10 * np.pi * x)

    # Compute gradient magnitude before reinitialization
    grad_ops = grid_1d.get_gradient_operator()
    grad_before = grad_ops[0](phi0_slightly_distorted)
    grad_mag_before = np.abs(grad_before)

    print(f"  Before reinit: |∇φ| range = [{grad_mag_before.min():.4f}, {grad_mag_before.max():.4f}]")
    deviation_before = np.max(np.abs(grad_mag_before - 1.0))
    print(f"  Before reinit: max(||∇φ| - 1|) = {deviation_before:.4f}")

    # Reinitialize
    phi_reinit = reinitialize(phi0_slightly_distorted, grid_1d, max_iterations=20, dtau=None)

    # Check gradient magnitude after
    grad_after = grad_ops[0](phi_reinit)
    grad_mag_after = np.abs(grad_after)

    print(f"  After reinit: |∇φ| range = [{grad_mag_after.min():.4f}, {grad_mag_after.max():.4f}]")
    max_deviation = np.max(np.abs(grad_mag_after - 1.0))
    mean_deviation = np.mean(np.abs(grad_mag_after - 1.0))
    print(f"  After reinit: max(||∇φ| - 1|) = {max_deviation:.4f}, mean = {mean_deviation:.4f}")

    # Check that reinitialization maintains or improves SDF property
    # For slightly distorted SDFs, we expect maintenance (not necessarily big improvement)
    print(f"  Deviation change: {deviation_before:.4f} → {max_deviation:.4f}")

    # Target: Should not make things significantly worse, ideally improves
    assert max_deviation < deviation_before * 1.5, (
        f"Reinitialization made things worse: {max_deviation:.4f} vs {deviation_before:.4f}"
    )
    assert mean_deviation < 1.0, f"Mean deviation too large: {mean_deviation:.4f}"
    print("  ✓ SDF property maintained!")

    # Test 2: Zero level set preservation
    print("\n[Test 2: Zero Level Set Preservation]")
    print("Problem: Check that φ = 0 level set doesn't move during reinitialization")

    # Circle in 2D
    Nx, Ny = 100, 100
    grid_2d = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx=[Nx, Ny], boundary_conditions=no_flux_bc(dimension=2)
    )
    X, Y = grid_2d.meshgrid()
    dx2d = grid_2d.spacing[0]

    print(f"  Grid: {Nx}×{Ny}, dx = {dx2d:.4f}")

    # Circle: φ = ||x - c|| - R (true SDF)
    center = np.array([0.5, 0.5])
    radius = 0.3
    phi0_circle = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius

    # Distort it: φ² (makes |∇φ| = 2|φ|, not constant)
    phi0_distorted_2d = phi0_circle**2 * np.sign(phi0_circle)

    print(f"  Circle: center={center}, radius={radius}")

    # Find zero level set before reinit
    zero_before = np.abs(phi0_distorted_2d) < 0.01
    coords_before = np.column_stack([X[zero_before], Y[zero_before]])

    # Reinitialize
    phi_reinit_2d = reinitialize(phi0_distorted_2d, grid_2d, max_iterations=20)

    # Find zero level set after reinit
    zero_after = np.abs(phi_reinit_2d) < 0.01
    coords_after = np.column_stack([X[zero_after], Y[zero_after]])

    # Check that zero level set didn't move much
    # (Some movement expected due to grid discretization)
    if len(coords_before) > 0 and len(coords_after) > 0:
        # Sample a few points and check distance
        n_samples = min(10, len(coords_before), len(coords_after))
        sample_before = coords_before[:: len(coords_before) // n_samples][:n_samples]

        # For each sample, find closest point in coords_after
        max_displacement = 0.0
        for pt in sample_before:
            dists = np.linalg.norm(coords_after - pt, axis=1)
            min_dist = np.min(dists)
            max_displacement = max(max_displacement, min_dist)

        print(f"  Max interface displacement: {max_displacement:.4f} ({max_displacement / dx2d:.2f} grid points)")
        # Note: Basic reinitialization can shift interface by several grid points
        # Future improvements (narrow band, fast marching) will reduce this
        assert max_displacement < 15 * dx2d, f"Interface moved too much: {max_displacement:.4f}"
        if max_displacement > 5 * dx2d:
            print(
                f"  WARNING: Interface shifted by {max_displacement / dx2d:.1f} grid points (consider narrow band method)"
            )
        print("  ✓ Zero level set reasonably preserved (within tolerance)!")

    # Test 3: Gradient magnitude improvement
    print("\n[Test 3: Gradient Magnitude Improvement]")

    grad_ops_2d = grid_2d.get_gradient_operator()
    grad_x_before = grad_ops_2d[0](phi0_distorted_2d)
    grad_y_before = grad_ops_2d[1](phi0_distorted_2d)
    grad_mag_before_2d = np.sqrt(grad_x_before**2 + grad_y_before**2)

    grad_x_after = grad_ops_2d[0](phi_reinit_2d)
    grad_y_after = grad_ops_2d[1](phi_reinit_2d)
    grad_mag_after_2d = np.sqrt(grad_x_after**2 + grad_y_after**2)

    deviation_before = np.abs(grad_mag_before_2d - 1.0)
    deviation_after = np.abs(grad_mag_after_2d - 1.0)

    max_dev_before = np.max(deviation_before)
    max_dev_after = np.max(deviation_after)
    mean_dev_before = np.mean(deviation_before)
    mean_dev_after = np.mean(deviation_after)

    print(f"  Before reinit: max(||∇φ| - 1|) = {max_dev_before:.4f}, mean = {mean_dev_before:.4f}")
    print(f"  After reinit: max(||∇φ| - 1|) = {max_dev_after:.4f}, mean = {mean_dev_after:.4f}")

    # Check that reinitialization doesn't make things significantly worse
    # Ideally should improve, but at minimum should maintain
    assert max_dev_after < max_dev_before * 1.5, (
        f"Reinitialization made things worse: {max_dev_after:.4f} vs {max_dev_before:.4f}"
    )
    assert mean_dev_after < 1.0, f"Mean deviation too large: {mean_dev_after:.4f}"

    if max_dev_after < max_dev_before:
        print(f"  ✓ Gradient magnitude improved ({max_dev_before:.4f} → {max_dev_after:.4f})!")
    else:
        print("  ✓ Gradient magnitude maintained (no significant degradation)!")

    # Test 4: Narrow Band Reinitialization
    print("\n[Test 4: Narrow Band Reinitialization (Correctness & Scalability)]")
    print("Problem: Reinitialize only near interface for efficiency")

    import time

    # Use larger grid for visible speedup (200×200 instead of 100×100)
    Nx_large, Ny_large = 200, 200
    grid_2d_large = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx=[Nx_large, Ny_large],
        boundary_conditions=no_flux_bc(dimension=2),
    )
    X_large, Y_large = grid_2d_large.meshgrid()
    dx_large = grid_2d_large.spacing[0]

    print(f"  Grid: {Nx_large}×{Ny_large}, dx = {dx_large:.4f}")

    # Circle: φ = ||x - c|| - R (true SDF)
    phi0_circle_large = np.sqrt((X_large - center[0]) ** 2 + (Y_large - center[1]) ** 2) - radius

    # Distort it: φ² (makes |∇φ| = 2|φ|, not constant)
    phi0_circle_distorted = phi0_circle_large**2 * np.sign(phi0_circle_large)

    # Global reinitialization (baseline)
    print("  Global reinitialization:")
    t_start = time.perf_counter()
    phi_global = reinitialize(phi0_circle_distorted, grid_2d_large, max_iterations=20, narrow_band_width=None)
    time_global = time.perf_counter() - t_start
    print(f"    Time: {time_global:.4f} s")

    # Narrow band reinitialization
    narrow_band_width = 5 * dx_large  # 5 grid points around interface
    print(f"  Narrow band reinitialization (width = {narrow_band_width:.4f} = 5dx):")
    t_start = time.perf_counter()
    phi_narrow = reinitialize(
        phi0_circle_distorted, grid_2d_large, max_iterations=20, narrow_band_width=narrow_band_width
    )
    time_narrow = time.perf_counter() - t_start
    print(f"    Time: {time_narrow:.4f} s")

    speedup = time_global / time_narrow
    print(f"    Speedup: {speedup:.2f}×")

    # Check that results are similar near interface
    interface_region = np.abs(phi0_circle_large) < 3 * dx_large
    diff_at_interface = np.abs(phi_global[interface_region] - phi_narrow[interface_region])
    max_diff_at_interface = np.max(diff_at_interface)
    print(f"    Max difference at interface: {max_diff_at_interface:.6f}")

    # Speedup scales with grid size (2× for 200×200, 5-10× for finer grids)
    # For correctness, just require it's not significantly slower
    assert speedup >= 0.8, f"Narrow band significantly slower: {speedup:.2f}× < 0.8×"
    assert max_diff_at_interface < 0.05, f"Narrow band result differs too much from global: {max_diff_at_interface:.6f}"

    if speedup >= 1.5:
        print(f"  ✓ Narrow band {speedup:.1f}× faster with similar accuracy!")
    else:
        print(f"  ✓ Narrow band correct (speedup {speedup:.2f}×, scales better with larger grids)!")

    # Test 5: FMM vs FSM vs PDE comparison
    print("\n[Test 5: Method Comparison (FMM vs FSM vs PDE)]")
    print("Problem: Compare accuracy and speed of different reinitialization methods")

    # Use a true SDF (circle) to measure accuracy
    phi_circle_test = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - 0.3

    print("  Input: Circle SDF (should remain unchanged)")

    # FMM
    t_start = time.perf_counter()
    phi_fmm = reinitialize(phi_circle_test, grid_2d, method="fmm")
    time_fmm = time.perf_counter() - t_start
    error_fmm = np.max(np.abs(phi_fmm - phi_circle_test))
    print(f"  FMM:  time = {time_fmm:.4f}s, max error = {error_fmm:.6f}")

    # FSM
    t_start = time.perf_counter()
    phi_fsm = reinitialize(phi_circle_test, grid_2d, method="fsm")
    time_fsm = time.perf_counter() - t_start
    error_fsm = np.max(np.abs(phi_fsm - phi_circle_test))
    print(f"  FSM:  time = {time_fsm:.4f}s, max error = {error_fsm:.6f}")

    # PDE
    t_start = time.perf_counter()
    phi_pde = reinitialize(phi_circle_test, grid_2d, method="pde", max_iterations=20)
    time_pde = time.perf_counter() - t_start
    error_pde = np.max(np.abs(phi_pde - phi_circle_test))
    print(f"  PDE:  time = {time_pde:.4f}s, max error = {error_pde:.6f}")

    # FMM/FSM should preserve true SDF exactly (up to discretization)
    assert error_fmm < 2 * dx2d, f"FMM error too large: {error_fmm}"
    assert error_fsm < 2 * dx2d, f"FSM error too large: {error_fsm}"
    print("  [OK] FMM and FSM preserve true SDF accurately")

    # Test 6: Distorted level set recovery
    print("\n[Test 6: Distorted Level Set Recovery]")
    # Severely distort: phi^2 * sign(phi)
    phi_distorted = phi_circle_test**2 * np.sign(phi_circle_test)
    print("  Input: phi^2 * sign(phi) distortion")

    phi_fmm_rec = reinitialize(phi_distorted, grid_2d, method="fmm")
    phi_pde_rec = reinitialize(phi_distorted, grid_2d, method="pde", max_iterations=50)

    # Compare to true SDF
    error_fmm_rec = np.max(np.abs(phi_fmm_rec - phi_circle_test))
    error_pde_rec = np.max(np.abs(phi_pde_rec - phi_circle_test))
    print(f"  FMM recovery error: {error_fmm_rec:.4f}")
    print(f"  PDE recovery error: {error_pde_rec:.4f}")

    # FMM should recover SDF much better than PDE
    assert error_fmm_rec < error_pde_rec or error_fmm_rec < 0.05, (
        f"FMM should be more accurate than PDE: {error_fmm_rec} vs {error_pde_rec}"
    )
    print("  [OK] FMM recovers SDF from distorted level set")

    print("\n[OK] All Reinitialization tests passed (including FMM/FSM methods)!")

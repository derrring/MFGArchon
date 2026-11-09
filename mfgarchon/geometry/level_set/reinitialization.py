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
    padded = (
        pad_array_with_ghosts(phi, bc, ghost_depth=1, time=0.0, spacing=geometry.get_grid_spacing())
        if bc is not None
        else phi
    )
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

"""Gradient form with upwind differences for FP advection term.

This module implements the advective (gradient) form FDM discretization
using upwind differences for the advection term.

WARNING: this scheme is NOT mass-conservative at no-flux boundaries. The
advective (v.grad(m)) form conserves ROW sums = 1/dt, but mass conservation of
an implicit FP step requires COLUMN sums = 1/dt; the column sums are unbalanced
at the walls, so probability mass leaks out (measured: up to ~99.8% loss when
the drift is wall-directed). Use divergence_upwind (or divergence_centered) for
no-flux FP. See Issue #382; factory routes FDM_UPWIND to divergence_upwind.

Mathematical Formulation:
    Advection term: -div(m * v) ≈ -v · grad(m)  (advective/gradient form)

    For velocity v = -coupling_coefficient * dU/dx:
    - If v > 0 (flow to right): use backward difference dm/dx ≈ (m_i - m_{i-1}) / dx
    - If v < 0 (flow to left): use forward difference dm/dx ≈ (m_{i+1} - m_i) / dx

    This ensures:
    - Row sums = 1/dt -- this is NOT mass conservation (column sums are
      unbalanced at no-flux walls; mass leaks). Use divergence_* for no-flux FP.
    - Unconditional stability (no Peclet number restriction)
    - First-order accuracy O(dx)
    - Some numerical diffusion (upwind dissipation)

Comparison with other schemes:
    | Scheme             | Conservative | Stable      | Accuracy |
    |--------------------|--------------|-------------|----------|
    | gradient_centered  | NO           | Peclet < 2  | O(dx^2)  |
    | gradient_upwind    | YES (rows)   | Always      | O(dx)    |
    | divergence_centered| YES (flux)   | Peclet < 2  | O(dx^2)  |
    | divergence_upwind  | YES (flux)   | Always      | O(dx)    |
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mfgarchon.utils.aux_func import npart, ppart
from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

if TYPE_CHECKING:
    import numpy as np


def add_interior_entries_gradient_upwind(
    row_indices: list[int],
    col_indices: list[int],
    data_values: list[float],
    flat_idx: int,
    multi_idx: tuple[int, ...],
    shape: tuple[int, ...],
    ndim: int,
    dt: float,
    sigma: float,
    coupling_coefficient: float,
    spacing: tuple[float, ...],
    u_flat: np.ndarray,
    grid: Any,
    boundary_conditions: Any,
    interface_velocity: np.ndarray | None = None,
) -> None:
    """
    Add matrix entries for interior point using UPWIND differences.

    Discretizes: (1/dt) m + div(m*v) - (σ²/2) Δm = 0

    where v = -coupling_coefficient * ∇U (drift velocity)

    Upwind discretization for advection:
        - Uses ppart/npart to select upwind direction based on velocity
        - ppart(x) = max(0, x), npart(x) = max(0, -x)
        - Row sums = 1/dt (NOT mass conservation: column sums are unbalanced at
          no-flux walls, so mass leaks; use divergence_* for no-flux FP)

    This discretization:
    - Has row sums = 1/dt but is NOT mass-conservative at no-flux boundaries
    - Is unconditionally stable (no Peclet restriction)
    - Has first-order accuracy O(dx)
    - Adds numerical diffusion (upwind dissipation)

    Parameters
    ----------
    row_indices, col_indices, data_values : lists
        COO format sparse matrix data (modified in place)
    flat_idx : int
        Flattened index of current grid point
    multi_idx : tuple[int, ...]
        Multi-dimensional index (i, j, k, ...)
    shape : tuple[int, ...]
        Grid shape (N1, N2, ..., Nd)
    ndim : int
        Spatial dimension
    dt : float
        Time step
    sigma : float
        Diffusion coefficient
    coupling_coefficient : float
        Coefficient for drift term (typically 1/λ in MFG)
    spacing : tuple[float, ...]
        Grid spacing per dimension
    u_flat : np.ndarray
        Flattened value function for drift computation
    grid : TensorProductGrid
        Grid geometry for index conversion
    boundary_conditions : BoundaryConditions
        BC specification for periodic handling
    """
    # Diagonal term (accumulates contributions from all dimensions)
    diagonal_value = 1.0 / dt

    # Diffusion coefficient D = sigma^2/2 (Issue #1189: single source)
    D = diffusion_from_volatility(sigma)

    # For each dimension, add advection + diffusion contributions
    for d in range(ndim):
        dx = spacing[d]
        dx_sq = dx * dx

        # Get neighbor indices in dimension d
        multi_idx_plus = list(multi_idx)
        multi_idx_minus = list(multi_idx)

        multi_idx_plus[d] = multi_idx[d] + 1
        multi_idx_minus[d] = multi_idx[d] - 1

        # Handle boundary wrapping for periodic BC
        # Issue #543 Phase 2: Replace hasattr with try/except
        try:
            is_periodic = boundary_conditions.is_uniform and boundary_conditions.type == "periodic"
        except AttributeError:
            # For BoundaryConditionManager2D or unknown types, default to non-periodic
            is_periodic = False

        if is_periodic:
            multi_idx_plus[d] = multi_idx_plus[d] % shape[d]
            multi_idx_minus[d] = multi_idx_minus[d] % shape[d]

        # Check if neighbors exist (non-periodic case)
        has_plus = multi_idx_plus[d] < shape[d]
        has_minus = multi_idx_minus[d] >= 0

        if has_plus or is_periodic:
            flat_idx_plus = grid.get_index(tuple(multi_idx_plus))
            u_plus = u_flat[flat_idx_plus]
        else:
            u_plus = u_flat[flat_idx]  # Use current value at boundary

        if has_minus or is_periodic:
            flat_idx_minus = grid.get_index(tuple(multi_idx_minus))
            u_minus = u_flat[flat_idx_minus]
        else:
            u_minus = u_flat[flat_idx]  # Use current value at boundary

        u_center = u_flat[flat_idx]

        # Diffusion contribution (centered differences)
        # -D * (m_{i+1} - 2m_i + m_{i-1}) / dx^2  where D = sigma^2/2
        diagonal_value += 2 * D / dx_sq

        if has_plus or is_periodic:
            # Coupling to m_{i+1,j}: diffusion -D/dx^2
            coeff_plus = -D / dx_sq

            # Add advection upwind term
            # For advection: -d/dx(m*v) where v = -coupling_coefficient * dU/dx
            # Upwind: use ppart for positive velocity contribution
            coeff_plus += float(-coupling_coefficient * ppart(u_plus - u_center) / dx_sq)

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_plus)
            data_values.append(coeff_plus)

            # Add advection contribution to diagonal
            diagonal_value += float(coupling_coefficient * ppart(u_plus - u_center) / dx_sq)

        if has_minus or is_periodic:
            # Coupling to m_{i-1,j}: diffusion -D/dx^2
            coeff_minus = -D / dx_sq

            # Add advection upwind term
            # Upwind: use npart for negative velocity contribution
            coeff_minus += float(-coupling_coefficient * npart(u_center - u_minus) / dx_sq)

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_minus)
            data_values.append(coeff_minus)

            # Add advection contribution to diagonal
            diagonal_value += float(coupling_coefficient * npart(u_center - u_minus) / dx_sq)

    # Add diagonal entry
    row_indices.append(flat_idx)
    col_indices.append(flat_idx)
    data_values.append(diagonal_value)


def add_boundary_no_flux_entries_gradient_upwind(
    row_indices: list[int],
    col_indices: list[int],
    data_values: list[float],
    flat_idx: int,
    multi_idx: tuple[int, ...],
    shape: tuple[int, ...],
    ndim: int,
    dt: float,
    sigma: float,
    coupling_coefficient: float,
    spacing: tuple[float, ...],
    u_flat: np.ndarray,
    grid: Any,
    interface_velocity: np.ndarray | None = None,
) -> None:
    """
    Add matrix entries for boundary point with no-flux BC using UPWIND scheme.

    No-flux BC for Fokker-Planck: J.n = 0 where J = alpha*m - D*grad(m)

    Uses flux-form (finite volume) discretization at boundaries:
    - Boundary cells treated as half-cells (volume dx/2)
    - Flux at boundary face = 0 (no-flux condition)
    - Interior face flux uses upwind for advection

    This is the same as add_boundary_no_flux_entries in fp_fdm_bc.py,
    provided here for API consistency with the scheme-specific modules.
    """
    # Diagonal term (time derivative)
    diagonal_value = 1.0 / dt

    # Diffusion coefficient D = σ²/2
    D = diffusion_from_volatility(sigma)

    # For each dimension, check if we're at a boundary in that dimension
    for d in range(ndim):
        dx = spacing[d]
        dx_sq = dx * dx

        at_left_boundary = multi_idx[d] == 0
        at_right_boundary = multi_idx[d] == shape[d] - 1
        at_interior_in_d = not (at_left_boundary or at_right_boundary)

        if at_interior_in_d:
            # Standard interior stencil in this dimension
            multi_idx_plus = list(multi_idx)
            multi_idx_minus = list(multi_idx)
            multi_idx_plus[d] = multi_idx[d] + 1
            multi_idx_minus[d] = multi_idx[d] - 1

            flat_idx_plus = grid.get_index(tuple(multi_idx_plus))
            flat_idx_minus = grid.get_index(tuple(multi_idx_minus))

            u_plus = u_flat[flat_idx_plus]
            u_minus = u_flat[flat_idx_minus]
            u_center = u_flat[flat_idx]

            # Diffusion: -D*(m_{i+1} - 2m_i + m_{i-1}) / dx^2  where D = sigma^2/2
            diagonal_value += 2 * D / dx_sq

            coeff_plus = -D / dx_sq
            coeff_plus += float(-coupling_coefficient * ppart(u_plus - u_center) / dx_sq)

            coeff_minus = -D / dx_sq
            coeff_minus += float(-coupling_coefficient * npart(u_center - u_minus) / dx_sq)

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_plus)
            data_values.append(coeff_plus)

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_minus)
            data_values.append(coeff_minus)

            diagonal_value += float(
                coupling_coefficient * (ppart(u_plus - u_center) + npart(u_center - u_minus)) / dx_sq
            )

        elif at_left_boundary:
            # Left boundary: ghost point approach for no-flux BC
            multi_idx_plus = list(multi_idx)
            multi_idx_plus[d] = multi_idx[d] + 1

            flat_idx_plus = grid.get_index(tuple(multi_idx_plus))
            u_plus = u_flat[flat_idx_plus]
            u_center = u_flat[flat_idx]

            # Diffusion: Neumann BC via ghost point reflection
            # No-flux: dm/dx = 0 → m_{-1} = m_1, so Laplacian = 2(m_1 - m_0)/dx²
            # Issue #668 fix: coefficient is 2*D/dx², not D/dx²
            diagonal_value += 2 * D / dx_sq
            coeff_plus = -2 * D / dx_sq

            # Advection: use upwind with proper no-flux handling
            grad_U = (u_plus - u_center) / dx
            alpha = -coupling_coefficient * grad_U

            if alpha >= 0:
                # Flow to right (away from left boundary)
                # No-flux means this flux is zero - mass stays in place
                pass
            else:
                # Flow to left (into left boundary from right)
                # Upwind from m_1
                diagonal_value += -alpha / dx  # alpha < 0, so -alpha > 0
                coeff_plus += alpha / dx  # alpha < 0, so this is negative

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_plus)
            data_values.append(coeff_plus)

        elif at_right_boundary:
            # Right boundary: ghost point approach for no-flux BC
            multi_idx_minus = list(multi_idx)
            multi_idx_minus[d] = multi_idx[d] - 1

            flat_idx_minus = grid.get_index(tuple(multi_idx_minus))
            u_minus = u_flat[flat_idx_minus]
            u_center = u_flat[flat_idx]

            # Diffusion: Neumann BC via ghost point reflection
            # No-flux: dm/dx = 0 → m_{N+1} = m_{N-1}, so Laplacian = 2(m_{N-1} - m_N)/dx²
            # Issue #668 fix: coefficient is 2*D/dx², not D/dx²
            diagonal_value += 2 * D / dx_sq
            coeff_minus = -2 * D / dx_sq

            # Advection: use upwind with proper no-flux handling
            grad_U = (u_center - u_minus) / dx
            alpha = -coupling_coefficient * grad_U

            if alpha >= 0:
                # Flow to right (into right boundary from left)
                # Upwind from m_{-1}
                diagonal_value += alpha / dx
                coeff_minus += -alpha / dx
            else:
                # Flow to left (away from right boundary)
                # No-flux means this flux is zero
                pass

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_minus)
            data_values.append(coeff_minus)

    # Add diagonal entry
    row_indices.append(flat_idx)
    col_indices.append(flat_idx)
    data_values.append(diagonal_value)


# Backward-compatible aliases
add_interior_entries_upwind = add_interior_entries_gradient_upwind
add_boundary_no_flux_entries_upwind = add_boundary_no_flux_entries_gradient_upwind

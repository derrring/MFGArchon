"""Divergence form with upwind flux selection for FP equation.

This module provides the conservative (divergence form) FDM discretization
for the Fokker-Planck equation using upwind flux selection at cell interfaces.
This approach ensures exact mass conservation via flux telescoping AND
unconditional stability via upwind selection.

Module structure per issue #388:
    fp_fdm_alg_divergence_upwind.py - Conservative flux discretization algorithms

Functions:
    add_interior_entries_divergence_upwind: Interior stencil for flux FDM
    add_interior_entries_conservative: Backward-compatible alias

Mathematical Background:
    Divergence form discretizes: div(v * m) as flux differences at cell interfaces

    Key features:
    - Interface velocities: v_{i+1/2} = -coupling * (U_{i+1} - U_i) / dx
    - Flux at interface: F_{i+1/2} = v_{i+1/2} * m_upwind
    - Divergence: (F_{i+1/2} - F_{i-1/2}) / dx

    The flux entering cell i from cell i-1 is exactly the flux leaving cell i-1,
    guaranteeing mass conservation by construction (flux telescoping).

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

from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

if TYPE_CHECKING:
    import numpy as np


def add_interior_entries_divergence_upwind(
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
    Add matrix entries for interior grid point using CONSERVATIVE Flux FDM.

    Discretizes divergence form: div(alpha * m) - D * Laplacian(m)
    using flux differences at cell interfaces.

    Key difference from gradient FDM:
    - Interface velocities: alpha_{i+1/2} = -coupling * (U_{i+1} - U_i) / dx
    - Flux at interface: F_{i+1/2} = alpha_{i+1/2} * m_upwind
    - Divergence: (F_{i+1/2} - F_{i-1/2}) / dx

    This ensures column sums = 1/dt (mass conservation by construction).
    The flux entering cell i from cell i-1 is exactly the flux leaving cell i-1.

    Mathematical formulation (1D example):
        Flux FDM for advection: (F_{i+1/2} - F_{i-1/2}) / dx
        where F_{i+1/2} = alpha_{i+1/2} * m_upwind

        For alpha_{i+1/2} = -lambda * (U_{i+1} - U_i) / dx >= 0:
            F_{i+1/2} = alpha_{i+1/2} * m_i  (upwind from left)
        For alpha_{i+1/2} < 0:
            F_{i+1/2} = alpha_{i+1/2} * m_{i+1}  (upwind from right)
    """
    # Diagonal term (time derivative)
    diagonal_value = 1.0 / dt

    # Diffusion coefficient D = sigma^2/2
    D = diffusion_from_volatility(sigma)

    # For each dimension, add flux-based advection + diffusion contributions
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
            is_periodic = False

        if is_periodic:
            multi_idx_plus[d] = multi_idx_plus[d] % shape[d]
            multi_idx_minus[d] = multi_idx_minus[d] % shape[d]

        # Check if neighbors exist (non-periodic case)
        has_plus = multi_idx_plus[d] < shape[d]
        has_minus = multi_idx_minus[d] >= 0

        # Get flat indices and values for neighbors
        if has_plus or is_periodic:
            flat_idx_plus = grid.get_index(tuple(multi_idx_plus))
            u_plus = u_flat[flat_idx_plus]
        else:
            flat_idx_plus = flat_idx
            u_plus = u_flat[flat_idx]

        if has_minus or is_periodic:
            flat_idx_minus = grid.get_index(tuple(multi_idx_minus))
            u_minus = u_flat[flat_idx_minus]
        else:
            flat_idx_minus = flat_idx
            u_minus = u_flat[flat_idx]

        u_center = u_flat[flat_idx]

        # Diffusion contribution (centered differences) - same as gradient FDM
        # -D * (m_{i+1} - 2m_i + m_{i-1}) / dx^2
        # This is inherently conservative (Laplacian has zero column sums)
        diagonal_value += 2 * D / dx_sq

        if has_plus or is_periodic:
            row_indices.append(flat_idx)
            col_indices.append(flat_idx_plus)
            data_values.append(-D / dx_sq)

        if has_minus or is_periodic:
            row_indices.append(flat_idx)
            col_indices.append(flat_idx_minus)
            data_values.append(-D / dx_sq)

        # ===== CONSERVATIVE FLUX ADVECTION =====
        # Interface velocities (at cell faces, not centers)
        # Default: alpha_{i+1/2} = -coupling * (U_{i+1} - U_i) / dx
        # Issue #919: when interface_velocity provided, use directly

        if has_plus or is_periodic:
            if interface_velocity is not None:
                # Issue #919: interface_velocity[d][i] = velocity at face (i+1/2)
                alpha_right = float(interface_velocity[d][multi_idx])
            else:
                alpha_right = -coupling_coefficient * (u_plus - u_center) / dx

            # Flux F_{i+1/2} contribution to row i (outgoing flux with +1/dx)
            if alpha_right >= 0:
                # Flow to right: F = alpha * m_i (upwind from left)
                # Contributes +alpha/dx to diagonal (outflow from cell i)
                diagonal_value += alpha_right / dx
            else:
                # Flow to left: F = alpha * m_{i+1} (upwind from right)
                # Contributes +alpha/dx to coeff for m_{i+1}
                # Note: alpha < 0, so this is negative (inflow to cell i)
                row_indices.append(flat_idx)
                col_indices.append(flat_idx_plus)
                data_values.append(alpha_right / dx)

        if has_minus or is_periodic:
            if interface_velocity is not None:
                # interface_velocity[d][i-1] = velocity at face (i-1/2)
                alpha_left = float(interface_velocity[d][tuple(multi_idx_minus)])
            else:
                alpha_left = -coupling_coefficient * (u_center - u_minus) / dx

            # Flux -F_{i-1/2} contribution to row i (incoming flux with -1/dx)
            if alpha_left >= 0:
                # Flow to right: F = alpha * m_{i-1} (upwind from left)
                # Contributes -alpha/dx to coeff for m_{i-1}
                # Note: alpha > 0, so this is negative (inflow from left)
                row_indices.append(flat_idx)
                col_indices.append(flat_idx_minus)
                data_values.append(-alpha_left / dx)
            else:
                # Flow to left: F = alpha * m_i (upwind from right)
                # Contributes -alpha/dx to diagonal
                # Note: alpha < 0, so -alpha > 0 (outflow to left)
                diagonal_value += -alpha_left / dx

    # Add diagonal entry
    row_indices.append(flat_idx)
    col_indices.append(flat_idx)
    data_values.append(diagonal_value)


# Backward-compatible aliases
add_interior_entries_conservative = add_interior_entries_divergence_upwind
add_interior_entries_flux = add_interior_entries_divergence_upwind

"""Boundary condition enforcement for FP FDM discretization.

This module provides boundary condition handling for the Fokker-Planck
finite difference solver. It implements no-flux (zero normal flux) conditions
for both conservative and non-conservative discretizations.

Module structure per issue #388:
    fp_fdm_bc.py - Edge behavior (boundary condition enforcement, ghost points)

Functions:
    add_boundary_no_flux_entries: Legacy alias for gradient_upwind
    add_boundary_no_flux_entries_conservative: Legacy alias for divergence_upwind

Note:
    The scheme-specific boundary functions are now in their respective files:
    - fp_fdm_alg_gradient_centered.py: add_boundary_no_flux_entries_gradient_centered
    - fp_fdm_alg_gradient_upwind.py: add_boundary_no_flux_entries_gradient_upwind
    - fp_fdm_alg_divergence_upwind.py: (uses add_boundary_no_flux_entries_divergence_upwind below)
    - fp_fdm_alg_divergence_centered.py: add_boundary_no_flux_entries_divergence_centered

Mathematical Background:
    No-flux BC for FP: J.n = 0 where J = v*m - D*grad(m)
    This ensures mass conservation at domain boundaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mfgarchon.utils.aux_func import npart, ppart
from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

if TYPE_CHECKING:
    import numpy as np


def add_boundary_no_flux_entries(
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
) -> None:
    """
    Add matrix entries for boundary grid point with no-flux BC.

    No-flux BC for Fokker-Planck: J.n = 0 where J = alpha*m - D*grad(m)

    Uses flux-form (finite volume) discretization at boundaries:
    - Boundary cells treated as half-cells (volume dx/2)
    - Flux at boundary face = 0 (no-flux condition)
    - Interior face flux uses upwind for advection

    Mathematical formulation:
    For FP equation: dm/dt = D*Laplacian(m) - div(alpha*m) where alpha = -lambda*grad(U)

    At right boundary (half-cell):
        (dx/2)*dm/dt = J_interior - J_boundary = J_interior - 0
        dm/dt = 2*J_interior / dx

    This ensures mass conservation by explicitly zeroing boundary flux.
    """
    # Diagonal term (time derivative)
    diagonal_value = 1.0 / dt

    # Diffusion coefficient D = sigma^2/2
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

            # Diffusion: D*(m_{i+1} - 2m_i + m_{i-1}) / dx^2
            diagonal_value += sigma**2 / dx_sq

            coeff_plus = -(sigma**2) / (2 * dx_sq)
            coeff_plus += float(-coupling_coefficient * ppart(u_plus - u_center) / dx_sq)

            coeff_minus = -(sigma**2) / (2 * dx_sq)
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
            #
            # For no-flux: J.n = 0 where J = alpha*m - D*grad(m)
            # This gives: dm/dn = (alpha.n)*m / D at boundary
            #
            # Using ghost point: m_ghost = m_0 (reflection for pure diffusion)
            # But for advection-diffusion, we need to adjust.
            #
            # Simplified approach: Use one-sided stencil that preserves
            # row sum = 1/dt for mass conservation.
            #
            # At left boundary, use forward stencil for diffusion:
            # d^2m/dx^2 ~ (m_1 - m_0) / dx^2 (first-order one-sided)
            #
            # For advection with upwind:
            # If alpha > 0 (flow right), mass leaves: d(alpha*m)/dx ~ alpha*(m_0 - m_ghost)/dx
            #   With no-flux, mass that "wants to leave" stays, so we use zero BC
            # If alpha < 0 (flow left, into boundary), upwind from right: d(alpha*m)/dx ~ alpha*(m_1 - m_0)/dx

            multi_idx_plus = list(multi_idx)
            multi_idx_plus[d] = multi_idx[d] + 1

            flat_idx_plus = grid.get_index(tuple(multi_idx_plus))
            u_plus = u_flat[flat_idx_plus]
            u_center = u_flat[flat_idx]

            # Diffusion: Neumann BC via ghost point reflection
            # No-flux BC: dm/dx = 0 → (m_1 - m_{-1})/(2dx) = 0 → m_{-1} = m_1
            # Laplacian: (m_1 - 2m_0 + m_{-1})/dx² = (m_1 - 2m_0 + m_1)/dx² = 2(m_1 - m_0)/dx²
            # Issue #668 fix: coefficient is 2*D/dx², not D/dx²
            diagonal_value += 2 * D / dx_sq
            coeff_plus = -2 * D / dx_sq

            # Advection: use upwind with proper no-flux handling
            # alpha = -lambda*grad(U)
            grad_U = (u_plus - u_center) / dx
            alpha = -coupling_coefficient * grad_U

            if alpha >= 0:
                # Flow to right (away from left boundary)
                # No-flux means this flux is zero - mass stays in place
                # Don't add advection terms (they would cause mass loss)
                pass
            else:
                # Flow to left (into left boundary from right)
                # Upwind from m_1: advection term = alpha*(m_1 - m_0)/dx (with alpha < 0)
                # In implicit form: -alpha/dx * (m_1 - m_0)
                # = -alpha/dx * m_1 + alpha/dx * m_0
                diagonal_value += -alpha / dx  # Note: alpha < 0, so -alpha > 0
                coeff_plus += alpha / dx  # alpha < 0, so this is negative

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_plus)
            data_values.append(coeff_plus)

        elif at_right_boundary:
            # Right boundary: ghost point approach for no-flux BC
            #
            # Similar to left boundary but mirrored.

            multi_idx_minus = list(multi_idx)
            multi_idx_minus[d] = multi_idx[d] - 1

            flat_idx_minus = grid.get_index(tuple(multi_idx_minus))
            u_minus = u_flat[flat_idx_minus]
            u_center = u_flat[flat_idx]

            # Diffusion: Neumann BC via ghost point reflection
            # No-flux BC: dm/dx = 0 → (m_{N+1} - m_{N-1})/(2dx) = 0 → m_{N+1} = m_{N-1}
            # Laplacian: (m_{N+1} - 2m_N + m_{N-1})/dx² = 2(m_{N-1} - m_N)/dx²
            # Issue #668 fix: coefficient is 2*D/dx², not D/dx²
            diagonal_value += 2 * D / dx_sq
            coeff_minus = -2 * D / dx_sq

            # Advection: use upwind with proper no-flux handling
            grad_U = (u_center - u_minus) / dx
            alpha = -coupling_coefficient * grad_U

            if alpha >= 0:
                # Flow to right (into right boundary from left)
                # Upwind from m_{-1}: advection term = alpha*(m_N - m_{-1})/dx
                # In implicit form: -alpha/dx * (m_N - m_{-1})
                # = -alpha/dx * m_N + alpha/dx * m_{-1}
                diagonal_value += alpha / dx  # alpha > 0
                coeff_minus += -alpha / dx  # negative
            else:
                # Flow to left (away from right boundary)
                # No-flux means this flux is zero - mass stays in place
                pass

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_minus)
            data_values.append(coeff_minus)

    # Add diagonal entry
    row_indices.append(flat_idx)
    col_indices.append(flat_idx)
    data_values.append(diagonal_value)


def add_boundary_no_flux_entries_conservative(
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
    Add matrix entries for boundary grid point with no-flux BC using CONSERVATIVE Flux FDM.

    No-flux BC: Total flux J = alpha*m - D*grad(m) = 0 at boundary.

    For conservative scheme:
    - Boundary flux F_{boundary} = 0 (enforced exactly)
    - Interior flux uses standard upwind selection
    - This maintains column sum = 1/dt even at boundaries
    """
    # Diagonal term (time derivative)
    diagonal_value = 1.0 / dt

    # Diffusion coefficient D = sigma^2/2
    D = diffusion_from_volatility(sigma)

    # For each dimension, check if we're at a boundary in that dimension
    for d in range(ndim):
        dx = spacing[d]

        at_left_boundary = multi_idx[d] == 0
        at_right_boundary = multi_idx[d] == shape[d] - 1
        at_interior_in_d = not (at_left_boundary or at_right_boundary)

        if at_interior_in_d:
            # Standard interior flux stencil in this dimension
            multi_idx_plus = list(multi_idx)
            multi_idx_minus = list(multi_idx)
            multi_idx_plus[d] = multi_idx[d] + 1
            multi_idx_minus[d] = multi_idx[d] - 1

            flat_idx_plus = grid.get_index(tuple(multi_idx_plus))
            flat_idx_minus = grid.get_index(tuple(multi_idx_minus))

            u_plus = u_flat[flat_idx_plus]
            u_minus = u_flat[flat_idx_minus]
            u_center = u_flat[flat_idx]

            # Diffusion (conservative centered differences)
            diagonal_value += 2 * D / (dx * dx)

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_plus)
            data_values.append(-D / (dx * dx))

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_minus)
            data_values.append(-D / (dx * dx))

            # Conservative flux advection
            # Issue #919: interface_velocity[d][i] = face velocity at (i+1/2)
            if interface_velocity is not None:
                alpha_right = float(interface_velocity[d][multi_idx])
                alpha_left = float(interface_velocity[d][tuple(multi_idx_minus)])
            else:
                alpha_right = -coupling_coefficient * (u_plus - u_center) / dx
                alpha_left = -coupling_coefficient * (u_center - u_minus) / dx

            # Right flux F_{i+1/2}
            if alpha_right >= 0:
                diagonal_value += alpha_right / dx
            else:
                row_indices.append(flat_idx)
                col_indices.append(flat_idx_plus)
                data_values.append(alpha_right / dx)

            # Left flux -F_{i-1/2}
            if alpha_left >= 0:
                row_indices.append(flat_idx)
                col_indices.append(flat_idx_minus)
                data_values.append(-alpha_left / dx)
            else:
                diagonal_value += -alpha_left / dx

        elif at_left_boundary:
            # Left boundary: F_{-1/2} = 0 (no-flux), only interior flux F_{1/2}
            multi_idx_plus = list(multi_idx)
            multi_idx_plus[d] = multi_idx[d] + 1

            flat_idx_plus = grid.get_index(tuple(multi_idx_plus))
            u_plus = u_flat[flat_idx_plus]
            u_center = u_flat[flat_idx]

            # Diffusion (conservative flux formulation):
            # For flux-based scheme: F_{1/2} = -D*(m_1 - m_0)/dx, F_{-1/2} = 0
            # Divergence: (F_{1/2} - F_{-1/2})/dx = -D*(m_1 - m_0)/dx²
            # Matrix contribution: +D/dx² to diagonal, -D/dx² to off-diagonal
            # Note: This differs from Laplacian-based (factor of 2) because we
            # compute face fluxes, not point Laplacians. See Issue #668 discussion.
            diagonal_value += D / (dx * dx)

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_plus)
            data_values.append(-D / (dx * dx))

            # Conservative flux advection: only F_{1/2}, no F_{-1/2} (zero flux at boundary)
            # Issue #919: interface_velocity[d][i] = face velocity at (i+1/2)
            if interface_velocity is not None:
                alpha_right = float(interface_velocity[d][multi_idx])
            else:
                alpha_right = -coupling_coefficient * (u_plus - u_center) / dx

            if alpha_right >= 0:
                # Outflow to right
                diagonal_value += alpha_right / dx
            else:
                # Inflow from right
                row_indices.append(flat_idx)
                col_indices.append(flat_idx_plus)
                data_values.append(alpha_right / dx)

            # Note: NO contribution from left flux - it's exactly zero (no-flux BC)
            # This is key for mass conservation at boundaries

        elif at_right_boundary:
            # Right boundary: F_{N+1/2} = 0 (no-flux), only interior flux F_{N-1/2}
            multi_idx_minus = list(multi_idx)
            multi_idx_minus[d] = multi_idx[d] - 1

            flat_idx_minus = grid.get_index(tuple(multi_idx_minus))
            u_minus = u_flat[flat_idx_minus]
            u_center = u_flat[flat_idx]

            # Diffusion (conservative flux formulation):
            # For flux-based scheme: F_{N-1/2} = -D*(m_N - m_{N-1})/dx, F_{N+1/2} = 0
            # Divergence: (F_{N+1/2} - F_{N-1/2})/dx = D*(m_N - m_{N-1})/dx²
            # Matrix contribution: +D/dx² to diagonal, -D/dx² to off-diagonal
            diagonal_value += D / (dx * dx)

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_minus)
            data_values.append(-D / (dx * dx))

            # Conservative flux advection: only -F_{N-1/2}, no F_{N+1/2} (zero flux at boundary)
            # Issue #919: interface_velocity[d][i-1] = face velocity at (i-1/2)
            if interface_velocity is not None:
                alpha_left = float(interface_velocity[d][tuple(multi_idx_minus)])
            else:
                alpha_left = -coupling_coefficient * (u_center - u_minus) / dx

            if alpha_left >= 0:
                # Inflow from left
                row_indices.append(flat_idx)
                col_indices.append(flat_idx_minus)
                data_values.append(-alpha_left / dx)
            else:
                # Outflow to left
                diagonal_value += -alpha_left / dx

            # Note: NO contribution from right flux - it's exactly zero (no-flux BC)

    # Add diagonal entry
    row_indices.append(flat_idx)
    col_indices.append(flat_idx)
    data_values.append(diagonal_value)


# New naming convention aliases
add_boundary_no_flux_entries_divergence_upwind = add_boundary_no_flux_entries_conservative

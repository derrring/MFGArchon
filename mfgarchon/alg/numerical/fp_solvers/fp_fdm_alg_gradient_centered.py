"""Gradient form with centered differences for FP advection term.

This module implements the advective (gradient) form FDM discretization
using central differences for the advection term.

WARNING: This scheme is NOT mass-conservative and can produce oscillations
for advection-dominated flows (Peclet number > 2). Use only for:
- Demonstrating why conservative schemes are needed
- Diffusion-dominated problems where advection is weak

Mathematical Formulation:
    Advection term: -div(m * v) ≈ -v · grad(m)  (advective/gradient form)

    Central difference: dm/dx ≈ (m_{i+1} - m_{i-1}) / (2*dx)

    This is second-order accurate O(dx^2) but:
    - Does NOT satisfy discrete mass conservation (row sums != 1/dt)
    - Produces spurious oscillations when |v|*dx/(2*sigma^2) > 1

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


def add_interior_entries_gradient_centered(
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
    Add matrix entries for interior point using CENTERED differences.

    Discretizes: (1/dt) m + v·∇m - (σ²/2) Δm = 0

    where v = -coupling_coefficient * ∇U (drift velocity)

    Central difference for advection:
        v·∇m ≈ v_x * (m_{i+1,j} - m_{i-1,j})/(2Δx) + v_y * (m_{i,j+1} - m_{i,j-1})/(2Δy)

    WARNING: This discretization:
    - Is NOT mass-conservative (row sums ≠ 1/dt in general)
    - Can produce oscillations for Peclet > 2
    - Should only be used for demonstration/comparison purposes

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

    # Check for periodic BC
    # Issue #543 Phase 2: Replace hasattr with try/except
    try:
        is_periodic = boundary_conditions.is_uniform and boundary_conditions.type == "periodic"
    except AttributeError:
        is_periodic = False

    # For each dimension, add advection + diffusion contributions
    for d in range(ndim):
        dx = spacing[d]
        dx_sq = dx * dx

        # Get neighbor indices in dimension d
        multi_idx_plus = list(multi_idx)
        multi_idx_minus = list(multi_idx)

        multi_idx_plus[d] = multi_idx[d] + 1
        multi_idx_minus[d] = multi_idx[d] - 1

        # Handle periodic wrapping
        if is_periodic:
            multi_idx_plus[d] = multi_idx_plus[d] % shape[d]
            multi_idx_minus[d] = multi_idx_minus[d] % shape[d]

        # Check if neighbors exist (non-periodic case)
        has_plus = multi_idx_plus[d] < shape[d]
        has_minus = multi_idx_minus[d] >= 0

        # Get U values for velocity computation
        if has_plus or is_periodic:
            flat_idx_plus = grid.get_index(tuple(multi_idx_plus))
            u_plus = u_flat[flat_idx_plus]
        else:
            u_plus = u_flat[flat_idx]

        if has_minus or is_periodic:
            flat_idx_minus = grid.get_index(tuple(multi_idx_minus))
            u_minus = u_flat[flat_idx_minus]
        else:
            u_minus = u_flat[flat_idx]

        # Compute velocity at this point: v = -coupling_coefficient * dU/dx
        # Using central difference for dU/dx
        velocity_d = -coupling_coefficient * (u_plus - u_minus) / (2 * dx)

        # Diffusion contribution (centered differences) - same as upwind
        # -D * (m_{i+1} - 2m_i + m_{i-1}) / dx^2  where D = sigma^2/2
        diagonal_value += 2 * D / dx_sq

        if has_plus or is_periodic:
            # Diffusion: coupling to m_{i+1}: -D/dx^2
            coeff_plus = -D / dx_sq

            # Advection (CENTERED): v * (m_{i+1} - m_{i-1}) / (2dx)
            # Contribution to m_{i+1}: v / (2dx)
            coeff_plus += velocity_d / (2 * dx)

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_plus)
            data_values.append(coeff_plus)

        if has_minus or is_periodic:
            # Diffusion: coupling to m_{i-1}: -D/dx^2
            coeff_minus = -D / dx_sq

            # Advection (CENTERED): v * (m_{i+1} - m_{i-1}) / (2dx)
            # Contribution to m_{i-1}: -v / (2dx)
            coeff_minus += -velocity_d / (2 * dx)

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_minus)
            data_values.append(coeff_minus)

    # Add diagonal entry
    row_indices.append(flat_idx)
    col_indices.append(flat_idx)
    data_values.append(diagonal_value)


def add_boundary_no_flux_entries_gradient_centered(
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
    Add matrix entries for boundary point with no-flux BC using centered scheme.

    For no-flux BC: ∂m/∂n = 0 at boundary

    Uses one-sided differences at boundaries to maintain scheme order.
    """
    # Diagonal term
    diagonal_value = 1.0 / dt

    # Diffusion coefficient D = sigma^2/2 (Issue #1189: single source)
    D = diffusion_from_volatility(sigma)

    for d in range(ndim):
        dx = spacing[d]
        dx_sq = dx * dx

        # Determine boundary type in this dimension
        at_lower = multi_idx[d] == 0
        at_upper = multi_idx[d] == shape[d] - 1

        if at_lower or at_upper:
            # One-sided difference for diffusion at boundary
            # No-flux: use ghost point reflection
            if at_lower:
                # At x=0: use forward difference
                multi_idx_plus = list(multi_idx)
                multi_idx_plus[d] = 1
                flat_idx_plus = grid.get_index(tuple(multi_idx_plus))

                # Diffusion: Neumann BC via ghost point reflection
                # No-flux: dm/dx = 0 → m_{-1} = m_1, so Laplacian = 2(m_1 - m_0)/dx²
                # Coefficient is 2D/dx² = σ²/dx² (Issue #668 fix; Issue #1189: route through D)
                diagonal_value += 2 * D / dx_sq
                coeff_plus = -2 * D / dx_sq

                row_indices.append(flat_idx)
                col_indices.append(flat_idx_plus)
                data_values.append(coeff_plus)

            elif at_upper:
                # At x=L: use backward difference
                multi_idx_minus = list(multi_idx)
                multi_idx_minus[d] = shape[d] - 2
                flat_idx_minus = grid.get_index(tuple(multi_idx_minus))

                # Diffusion: Neumann BC via ghost point reflection
                # No-flux: dm/dx = 0 → m_{N+1} = m_{N-1}, so Laplacian = 2(m_{N-1} - m_N)/dx²
                # Coefficient is 2D/dx² = σ²/dx² (Issue #668 fix; Issue #1189: route through D)
                diagonal_value += 2 * D / dx_sq
                coeff_minus = -2 * D / dx_sq

                row_indices.append(flat_idx)
                col_indices.append(flat_idx_minus)
                data_values.append(coeff_minus)
        else:
            # Interior in this dimension - standard centered stencil
            multi_idx_plus = list(multi_idx)
            multi_idx_minus = list(multi_idx)
            multi_idx_plus[d] = multi_idx[d] + 1
            multi_idx_minus[d] = multi_idx[d] - 1

            flat_idx_plus = grid.get_index(tuple(multi_idx_plus))
            flat_idx_minus = grid.get_index(tuple(multi_idx_minus))

            u_plus = u_flat[flat_idx_plus]
            u_minus = u_flat[flat_idx_minus]

            velocity_d = -coupling_coefficient * (u_plus - u_minus) / (2 * dx)

            # Diffusion: -D * (m_{i+1} - 2m_i + m_{i-1}) / dx^2  where D = sigma^2/2
            diagonal_value += 2 * D / dx_sq

            coeff_plus = -D / dx_sq + velocity_d / (2 * dx)
            coeff_minus = -D / dx_sq - velocity_d / (2 * dx)

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_plus)
            data_values.append(coeff_plus)

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_minus)
            data_values.append(coeff_minus)

    # Add diagonal entry
    row_indices.append(flat_idx)
    col_indices.append(flat_idx)
    data_values.append(diagonal_value)


# Backward-compatible aliases
add_interior_entries_centered = add_interior_entries_gradient_centered
add_boundary_no_flux_entries_centered = add_boundary_no_flux_entries_gradient_centered

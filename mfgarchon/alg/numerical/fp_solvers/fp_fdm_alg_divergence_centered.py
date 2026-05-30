"""Divergence form with centered differences for FP advection term.

This module implements the conservative (divergence) form FDM discretization
using central differences for the flux at cell faces.

Mathematical Formulation:
    Advection term: -div(m * v) discretized in divergence form

    Flux at cell face (centered average):
        F_{i+1/2} = (v_{i+1} * m_{i+1} + v_i * m_i) / 2

    Divergence discretization:
        div(m * v) ≈ (F_{i+1/2} - F_{i-1/2}) / dx

    This is:
    - CONSERVATIVE: Fluxes telescope, ensuring global mass conservation
    - UNSTABLE: Oscillates for Peclet > 2 (advection-dominated flows)
    - Second-order accurate O(dx^2)

Comparison with other schemes:
    | Scheme             | Conservative | Stable      | Accuracy |
    |--------------------|--------------|-------------|----------|
    | gradient_centered  | NO           | Peclet < 2  | O(dx^2)  |
    | gradient_upwind    | YES (rows)   | Always      | O(dx)    |
    | divergence_centered| YES (flux)   | Peclet < 2  | O(dx^2)  |
    | divergence_upwind  | YES (flux)   | Always      | O(dx)    |

Use Case:
    - Demonstrating that conservation alone doesn't guarantee good results
    - Diffusion-dominated problems (low Peclet number)
    - Comparison studies of FDM schemes
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def add_interior_entries_divergence_centered(
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
    Add matrix entries for interior point using DIVERGENCE form with CENTERED fluxes.

    Discretizes: (1/dt) m + div(m*v) - (sigma^2/2) Laplacian(m) = 0

    where v = -coupling_coefficient * grad(U) (drift velocity)

    Divergence form with centered flux averaging:
        F_{i+1/2} = (v_{i+1} * m_{i+1} + v_i * m_i) / 2
        div(m*v) ≈ (F_{i+1/2} - F_{i-1/2}) / dx

    This discretization:
    - IS mass-conservative (fluxes telescope)
    - Can produce oscillations for Peclet > 2
    - Has second-order accuracy O(dx^2)

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
        Coefficient for drift term (typically 1/lambda in MFG)
    spacing : tuple[float, ...]
        Grid spacing per dimension
    u_flat : np.ndarray
        Flattened value function for drift computation
    grid : TensorProductGrid
        Grid geometry for index conversion
    boundary_conditions : BoundaryConditions
        BC specification for periodic handling
    """
    # Diagonal term (time derivative)
    diagonal_value = 1.0 / dt

    # Diffusion coefficient D = sigma^2/2
    D = sigma**2 / 2.0

    # Check for periodic BC
    # Issue #543 Phase 2: Replace hasattr with try/except
    try:
        is_periodic = boundary_conditions.is_uniform and boundary_conditions.type == "periodic"
    except AttributeError:
        is_periodic = False

    # For each dimension, add flux-based advection + diffusion contributions
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

        # Get flat indices and U values
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

        # Compute velocities at cell centers: v = -coupling_coefficient * dU/dx
        # v_i = -coupling_coefficient * (U_{i+1} - U_{i-1}) / (2*dx)  [central diff]
        # But for flux computation, we need v at each node

        # Velocity at node i (using central difference for dU/dx)
        if (has_plus or is_periodic) and (has_minus or is_periodic):
            v_center = -coupling_coefficient * (u_plus - u_minus) / (2 * dx)
        elif has_plus or is_periodic:
            v_center = -coupling_coefficient * (u_plus - u_center) / dx
        elif has_minus or is_periodic:
            v_center = -coupling_coefficient * (u_center - u_minus) / dx
        else:
            v_center = 0.0

        # Velocity at node i+1
        if has_plus or is_periodic:
            # Need U at i+2 for central difference at i+1
            multi_idx_plus2 = list(multi_idx)
            multi_idx_plus2[d] = multi_idx[d] + 2
            if is_periodic:
                multi_idx_plus2[d] = multi_idx_plus2[d] % shape[d]

            if multi_idx_plus2[d] < shape[d] or is_periodic:
                flat_idx_plus2 = grid.get_index(tuple(multi_idx_plus2))
                u_plus2 = u_flat[flat_idx_plus2]
                v_plus = -coupling_coefficient * (u_plus2 - u_center) / (2 * dx)
            else:
                # One-sided at boundary
                v_plus = -coupling_coefficient * (u_plus - u_center) / dx
        else:
            v_plus = v_center

        # Velocity at node i-1
        if has_minus or is_periodic:
            # Need U at i-2 for central difference at i-1
            multi_idx_minus2 = list(multi_idx)
            multi_idx_minus2[d] = multi_idx[d] - 2
            if is_periodic:
                multi_idx_minus2[d] = multi_idx_minus2[d] % shape[d]

            if multi_idx_minus2[d] >= 0 or is_periodic:
                flat_idx_minus2 = grid.get_index(tuple(multi_idx_minus2))
                u_minus2 = u_flat[flat_idx_minus2]
                v_minus = -coupling_coefficient * (u_center - u_minus2) / (2 * dx)
            else:
                # One-sided at boundary
                v_minus = -coupling_coefficient * (u_center - u_minus) / dx
        else:
            v_minus = v_center

        # Diffusion (centered differences) - same for all schemes
        # -D * (m_{i+1} - 2m_i + m_{i-1}) / dx^2
        diagonal_value += 2 * D / dx_sq

        if has_plus or is_periodic:
            row_indices.append(flat_idx)
            col_indices.append(flat_idx_plus)
            data_values.append(-D / dx_sq)

        if has_minus or is_periodic:
            row_indices.append(flat_idx)
            col_indices.append(flat_idx_minus)
            data_values.append(-D / dx_sq)

        # ===== CONSERVATIVE CENTERED FLUX ADVECTION =====
        # Flux at face i+1/2 (centered average):
        #   F_{i+1/2} = (v_{i+1} * m_{i+1} + v_i * m_i) / 2
        #
        # Divergence: (F_{i+1/2} - F_{i-1/2}) / dx
        #
        # Contribution to equation for m_i:
        #   (1/dx) * [(v_{i+1}*m_{i+1} + v_i*m_i)/2 - (v_i*m_i + v_{i-1}*m_{i-1})/2]
        # = (1/(2*dx)) * [v_{i+1}*m_{i+1} - v_{i-1}*m_{i-1}]
        #
        # In matrix form (implicit):
        #   coeff for m_{i+1}: v_{i+1} / (2*dx)
        #   coeff for m_{i-1}: -v_{i-1} / (2*dx)
        #   coeff for m_i (diagonal): 0 from advection (cancels out!)

        if has_plus or is_periodic:
            # Contribution from F_{i+1/2} to m_{i+1}
            coeff_plus_adv = v_plus / (2 * dx)
            row_indices.append(flat_idx)
            col_indices.append(flat_idx_plus)
            data_values.append(coeff_plus_adv)

        if has_minus or is_periodic:
            # Contribution from -F_{i-1/2} to m_{i-1}
            coeff_minus_adv = -v_minus / (2 * dx)
            row_indices.append(flat_idx)
            col_indices.append(flat_idx_minus)
            data_values.append(coeff_minus_adv)

    # Add diagonal entry
    row_indices.append(flat_idx)
    col_indices.append(flat_idx)
    data_values.append(diagonal_value)


def add_boundary_no_flux_entries_divergence_centered(
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
    Add matrix entries for boundary point with no-flux BC using divergence form + centered.

    No-flux BC: Total flux J = v*m - D*grad(m) = 0 at boundary.

    For divergence form with centered fluxes:
    - Boundary flux F_{boundary} = 0 (enforced exactly)
    - Interior flux uses centered averaging
    """
    # Diagonal term (time derivative)
    diagonal_value = 1.0 / dt

    # Diffusion coefficient D = sigma^2/2
    D = sigma**2 / 2.0

    for d in range(ndim):
        dx = spacing[d]
        dx_sq = dx * dx

        at_left_boundary = multi_idx[d] == 0
        at_right_boundary = multi_idx[d] == shape[d] - 1
        at_interior_in_d = not (at_left_boundary or at_right_boundary)

        if at_interior_in_d:
            # Standard interior flux stencil
            multi_idx_plus = list(multi_idx)
            multi_idx_minus = list(multi_idx)
            multi_idx_plus[d] = multi_idx[d] + 1
            multi_idx_minus[d] = multi_idx[d] - 1

            flat_idx_plus = grid.get_index(tuple(multi_idx_plus))
            flat_idx_minus = grid.get_index(tuple(multi_idx_minus))

            u_plus = u_flat[flat_idx_plus]
            u_minus = u_flat[flat_idx_minus]
            u_center = u_flat[flat_idx]

            # Velocities at nodes (central difference)
            v_center = -coupling_coefficient * (u_plus - u_minus) / (2 * dx)

            # For v_plus, need U at i+2
            multi_idx_plus2 = list(multi_idx)
            multi_idx_plus2[d] = multi_idx[d] + 2
            if multi_idx_plus2[d] < shape[d]:
                flat_idx_plus2 = grid.get_index(tuple(multi_idx_plus2))
                u_plus2 = u_flat[flat_idx_plus2]
                v_plus = -coupling_coefficient * (u_plus2 - u_center) / (2 * dx)
            else:
                v_plus = -coupling_coefficient * (u_plus - u_center) / dx

            # For v_minus, need U at i-2
            multi_idx_minus2 = list(multi_idx)
            multi_idx_minus2[d] = multi_idx[d] - 2
            if multi_idx_minus2[d] >= 0:
                flat_idx_minus2 = grid.get_index(tuple(multi_idx_minus2))
                u_minus2 = u_flat[flat_idx_minus2]
                v_minus = -coupling_coefficient * (u_center - u_minus2) / (2 * dx)
            else:
                v_minus = -coupling_coefficient * (u_center - u_minus) / dx

            # Diffusion (centered)
            diagonal_value += 2 * D / dx_sq

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_plus)
            data_values.append(-D / dx_sq)

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_minus)
            data_values.append(-D / dx_sq)

            # Centered flux advection
            row_indices.append(flat_idx)
            col_indices.append(flat_idx_plus)
            data_values.append(v_plus / (2 * dx))

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_minus)
            data_values.append(-v_minus / (2 * dx))

        elif at_left_boundary:
            # Left boundary: F_{-1/2} = 0 (no-flux), only interior flux F_{1/2}
            multi_idx_plus = list(multi_idx)
            multi_idx_plus[d] = 1

            flat_idx_plus = grid.get_index(tuple(multi_idx_plus))
            u_plus = u_flat[flat_idx_plus]
            u_center = u_flat[flat_idx]

            # Diffusion (conservative flux formulation):
            # For flux-based scheme: F_{1/2} = -D*(m_1 - m_0)/dx, F_{-1/2} = 0
            # Matrix contribution: +D/dx² to diagonal, -D/dx² to off-diagonal
            diagonal_value += D / dx_sq
            row_indices.append(flat_idx)
            col_indices.append(flat_idx_plus)
            data_values.append(-D / dx_sq)

            # Velocity at node 1 for the shared face F_{1/2}. It MUST use the same central
            # stencil the interior handler uses for node 1 (whose neighbors 0 and 2 exist),
            # or F_{1/2} is double-valued, the fluxes do not telescope, and mass leaks at
            # the wall (Issue #1149). One-sided only when node 2 is absent (n <= 2).
            multi_idx_plus2 = list(multi_idx)
            multi_idx_plus2[d] = 2
            if multi_idx_plus2[d] < shape[d]:
                u_plus2 = u_flat[grid.get_index(tuple(multi_idx_plus2))]
                v_plus = -coupling_coefficient * (u_plus2 - u_center) / (2 * dx)
            else:
                v_plus = -coupling_coefficient * (u_plus - u_center) / dx

            # Centered flux at right face only: F_{1/2} = (v_1*m_1 + v_0*m_0)/2
            # But F_{-1/2} = 0, so divergence = F_{1/2} / dx
            # Contribution: v_plus * m_plus / (2*dx) + v_center * m_center / (2*dx)

            # Velocity at node 0
            v_center = -coupling_coefficient * (u_plus - u_center) / dx  # one-sided

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_plus)
            data_values.append(v_plus / (2 * dx))

            diagonal_value += v_center / (2 * dx)

        elif at_right_boundary:
            # Right boundary: F_{N+1/2} = 0 (no-flux), only interior flux F_{N-1/2}
            multi_idx_minus = list(multi_idx)
            multi_idx_minus[d] = shape[d] - 2

            flat_idx_minus = grid.get_index(tuple(multi_idx_minus))
            u_minus = u_flat[flat_idx_minus]
            u_center = u_flat[flat_idx]

            # Diffusion (conservative flux formulation):
            # For flux-based scheme: F_{N-1/2} = -D*(m_N - m_{N-1})/dx, F_{N+1/2} = 0
            # Matrix contribution: +D/dx² to diagonal, -D/dx² to off-diagonal
            diagonal_value += D / dx_sq
            row_indices.append(flat_idx)
            col_indices.append(flat_idx_minus)
            data_values.append(-D / dx_sq)

            # Velocity at node N-2 for the shared face F_{N-1/2}. Use the same central
            # stencil the interior handler uses for node N-2 (whose neighbors N-3 and N-1
            # exist), or F_{N-1/2} is double-valued and mass leaks at the wall (Issue #1149).
            # One-sided only when node N-3 is absent (n <= 2).
            multi_idx_minus2 = list(multi_idx)
            multi_idx_minus2[d] = shape[d] - 3
            if multi_idx_minus2[d] >= 0:
                u_minus2 = u_flat[grid.get_index(tuple(multi_idx_minus2))]
                v_minus = -coupling_coefficient * (u_center - u_minus2) / (2 * dx)
            else:
                v_minus = -coupling_coefficient * (u_center - u_minus) / dx

            # Centered flux at left face only: -F_{N-1/2} = -(v_N*m_N + v_{N-1}*m_{N-1})/2
            # Contribution: -v_minus * m_minus / (2*dx) - v_center * m_center / (2*dx)

            # Velocity at node N
            v_center = -coupling_coefficient * (u_center - u_minus) / dx  # one-sided

            row_indices.append(flat_idx)
            col_indices.append(flat_idx_minus)
            data_values.append(-v_minus / (2 * dx))

            diagonal_value += -v_center / (2 * dx)

    # Add diagonal entry
    row_indices.append(flat_idx)
    col_indices.append(flat_idx)
    data_values.append(diagonal_value)

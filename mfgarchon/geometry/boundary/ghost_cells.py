"""
Ghost cell formula functions for FDM boundary conditions.

These functions compute ghost cell values for structured grid (FDM) solvers.
They implement the mathematical formulas for Dirichlet, Neumann, Robin,
no-flux, and extrapolation boundary conditions.

Note: This module is distinct from ghost.py, which provides ghost POINT
generation for meshfree methods (reflection-based). This module provides
ghost cell VALUE computation for structured grids.

Extracted from applicator_base.py (mechanical refactor, no logic changes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocols import GridType


@dataclass
class GhostCellConfig:
    """Configuration for ghost cell computation.

    Controls how ghost cell formulas are applied based on grid type.

    Attributes:
        grid_type: Grid centering type. Cell-centered grids have the
            boundary at cell faces (ghost = 2g - interior for Dirichlet).
            Vertex-centered grids have the boundary at grid points.
    """

    grid_type: GridType | str = GridType.CELL_CENTERED

    def __post_init__(self) -> None:
        """Convert string grid_type to enum for backward compatibility."""
        if isinstance(self.grid_type, str):
            self.grid_type = GridType.VERTEX_CENTERED if self.grid_type == "vertex_centered" else GridType.CELL_CENTERED

    @property
    def is_vertex_centered(self) -> bool:
        """Check if grid is vertex-centered."""
        return self.grid_type == GridType.VERTEX_CENTERED

    @property
    def is_cell_centered(self) -> bool:
        """Check if grid is cell-centered."""
        return self.grid_type == GridType.CELL_CENTERED


# =============================================================================
# Ghost Cell Formula Helpers (used by FDM applicators)
# =============================================================================


def ghost_cell_dirichlet(
    interior_value: float,
    boundary_value: float,
    grid_type: GridType = GridType.CELL_CENTERED,
) -> float:
    """
    Compute ghost cell value for Dirichlet BC.

    For cell-centered grids (boundary at cell face):
        u_boundary = (u_ghost + u_interior) / 2 = g
        => u_ghost = 2*g - u_interior

    For vertex-centered grids (boundary at vertex):
        u_ghost = g (direct assignment)
    """
    if grid_type == GridType.VERTEX_CENTERED:
        return boundary_value
    else:
        return 2.0 * boundary_value - interior_value


def ghost_cell_neumann(
    interior_value: float,
    flux_value: float,
    dx: float,
) -> float:
    """
    Compute ghost cell value for Neumann BC. `flux_value` is du/dn.

    For cell-centered grids:
        du/dn = (u_ghost - u_interior) / (2*dx) * sign = g
        => u_ghost = u_interior + 2*dx*g*sign

    Args:
        interior_value: Value at interior point
        flux_value: Prescribed flux (du/dn)
        dx: Grid spacing
    """
    # One formula, both centrings, both walls. The ghost-to-interior separation is `dx` either way
    # -- cell-centred puts them at -dx/2 and +dx/2, vertex-centred at -dx and 0 -- so there is no
    # geometric reason to branch, and `flux_value` is du/dn, which already carries the direction.
    #
    # ~~cell-centred: interior + 2*dx*g*sign~~ [CORRECTED 2026-08-18, #1972]. That branch stated
    # `du/dn = (u_ghost - u_interior)/(2*dx)*sign` and satisfied it to machine zero at every
    # resolution -- self-consistent with a wrong definition, which is why nothing caught it. On
    # `u = sin(2*pi*x)` the residual against the correct spacing was frozen at exactly `2*pi = k`,
    # i.e. the discrete flux was exactly 2g, not approximately. `outward_normal_sign` went with it:
    # it converted du/dn to du/dx so the formula could convert back, and the round trip is where
    # the sign errors lived.
    #
    # Verified exact on 12 combinations (2 centrings x 2 walls x slopes 3, -1.7, 0) against
    # `u = a*x`, which any first-order ghost rule reproduces exactly; O(h^2) on `u = x^2`.
    return interior_value + dx * flux_value


def ghost_cell_robin(
    interior_value: float,
    rhs_value: float,
    alpha: float,
    beta: float,
    dx: float,
    outward_normal_sign: float = 1.0,
    grid_type: GridType = GridType.CELL_CENTERED,
) -> float:
    """
    Compute ghost cell value for Robin BC: alpha*u + beta*du/dn = g.

    For cell-centered grids (ghost at -dx/2, interior at +dx/2, boundary at 0):
        u_boundary = (u_ghost + u_interior) / 2
        du/dn = (u_ghost - u_interior) / dx  (distance between cell centers is dx)

        alpha * (u_ghost + u_interior)/2 + beta * (u_ghost - u_interior)/dx = g

    Solving for u_ghost:
        u_ghost * (alpha/2 + beta/dx) = g - u_interior * (alpha/2 - beta/dx)

    IMPORTANT: For cell-centered grids, du/dn = (u_ghost - u_interior)/dx for BOTH
    boundaries because ghost is always "outside" and interior is always "inside"
    regardless of left/right. The outward_normal_sign parameter is kept for backward
    compatibility but is NOT used in the cell-centered formula.

    For vertex-centered grids, the sign convention differs.
    """
    if grid_type == GridType.VERTEX_CENTERED:
        # Vertex-centered: sign matters because derivative direction differs
        if abs(alpha) > 1e-12:
            return (rhs_value - beta * outward_normal_sign * interior_value / dx) / alpha
        else:
            return interior_value + dx * rhs_value / beta * outward_normal_sign

    # Cell-centered: ghost and interior are dx apart
    # CRITICAL: du/dn = (u_ghost - u_interior)/dx for BOTH left and right boundaries
    # The outward_normal_sign is NOT used here because the geometry is symmetric:
    # - At left boundary: ghost at -dx/2, interior at +dx/2
    # - At right boundary: interior at L-dx/2, ghost at L+dx/2
    # In both cases, (u_ghost - u_interior)/dx gives the outward normal derivative.
    coeff_ghost = alpha / 2.0 + beta / dx
    coeff_interior = alpha / 2.0 - beta / dx

    # alpha, beta and rhs_value may be FIELDS -- one value per boundary point. The impermeable
    # wall of a Fokker-Planck equation is this condition with alpha = the outward normal drift,
    # which varies along the boundary and is recomputed each Picard iterate. The arithmetic
    # above is already elementwise; only this guard needed to stop assuming a scalar.
    singular = np.abs(coeff_ghost) < 1e-12
    if np.any(singular):
        where = (
            ""
            if np.isscalar(singular) or singular.ndim == 0
            else f" at {int(np.count_nonzero(singular))} of {singular.size} boundary points"
        )
        raise ValueError(
            f"Robin BC coefficients lead to singular ghost cell formula{where}: "
            f"alpha/2 + beta/dx = 0, so the condition does not determine the ghost value."
        )

    return (rhs_value - interior_value * coeff_interior) / coeff_ghost


# =============================================================================
# High-Order Ghost Cell Extrapolation (for WENO and other high-order schemes)
# =============================================================================


def high_order_ghost_dirichlet(
    interior_values: list[float],
    boundary_value: float,
    order: int = 4,
    grid_type: GridType = GridType.CELL_CENTERED,
) -> list[float]:
    """
    Compute high-order accurate ghost cell values for Dirichlet BC.

    For WENO5 and other high-order schemes, 2nd-order ghost cells degrade
    boundary accuracy. This function provides 4th or 5th order extrapolation.

    Mathematical derivation (cell-centered, 4th order):
        Given: u_b = g (Dirichlet BC at cell face x = x_0 - dx/2)
        Want: ghost values u_{-1}, u_{-2} that preserve polynomial accuracy

        Using Lagrange interpolation through boundary point and interior points:
        - u_{-1} extrapolated from {g, u_0, u_1, u_2}
        - u_{-2} extrapolated from {g, u_{-1}, u_0, u_1, u_2}

    Args:
        interior_values: Interior point values [u_0, u_1, u_2, ...] from boundary inward
        boundary_value: Dirichlet BC value g
        order: Extrapolation order (4 or 5)
        grid_type: Grid type (cell-centered or vertex-centered)

    Returns:
        Ghost values [u_{-1}, u_{-2}] (first is adjacent to interior)

    References:
        - Fedkiw et al. (1999): "A Non-oscillatory Eulerian Approach..."
        - Shu (1998): "Essentially Non-Oscillatory and WENO Schemes..."
    """
    if order < 4:
        # Fall back to 2nd-order for low orders
        u_int = interior_values[0]
        g = boundary_value
        if grid_type == GridType.VERTEX_CENTERED:
            return [g, 2 * g - u_int]
        else:
            u_ghost_1 = 2.0 * g - u_int
            u_ghost_2 = 2.0 * g - interior_values[1] if len(interior_values) > 1 else u_ghost_1
            return [u_ghost_1, u_ghost_2]

    # High-order extrapolation (4th or 5th order)
    g = boundary_value
    u = interior_values

    if grid_type == GridType.VERTEX_CENTERED:
        # For vertex-centered, boundary is at a grid point
        # u_{-1} = g directly
        # u_{-2} extrapolated using polynomial through g, u_0, u_1, u_2
        if order >= 4 and len(u) >= 3:
            # 4th-order extrapolation for u_{-2}
            # Using Lagrange polynomial through (x=-1, g), (x=0, u0), (x=1, u1), (x=2, u2)
            # evaluated at x=-2
            u_ghost_2 = 4 * g - 6 * u[0] + 4 * u[1] - u[2]
        else:
            u_ghost_2 = 2 * g - u[0]
        return [g, u_ghost_2]

    # Cell-centered: boundary at cell face (x = x_0 - dx/2)
    # Ghost cell centers are at x = x_0 - dx, x_0 - 2*dx, etc.
    # Boundary value g is at x = x_0 - dx/2

    if order >= 5 and len(u) >= 4:
        # 5th-order Lagrange extrapolation
        # Points: (x=-0.5, g), (x=0, u0), (x=1, u1), (x=2, u2), (x=3, u3)
        # Evaluate at x=-1 and x=-2

        # Coefficients derived from Lagrange interpolation formula
        # u_{-1} at x = -1:
        u_ghost_1 = (16 / 5) * g - 3 * u[0] + (8 / 5) * u[1] - (1 / 3) * u[2] + (1 / 30) * u[3]

        # u_{-2} at x = -2:
        u_ghost_2 = (48 / 5) * g - 12 * u[0] + 8 * u[1] - (8 / 3) * u[2] + (2 / 5) * u[3]
        return [u_ghost_1, u_ghost_2]

    elif order >= 4 and len(u) >= 3:
        # 4th-order Lagrange extrapolation
        # Points: (x=-0.5, g), (x=0, u0), (x=1, u1), (x=2, u2)
        # Evaluate at x=-1 and x=-2

        # u_{-1} at x = -1 (using 4-point Lagrange)
        u_ghost_1 = (16 / 5) * g - 3 * u[0] + (8 / 5) * u[1] - (1 / 5) * u[2]

        # u_{-2} at x = -2
        u_ghost_2 = (48 / 5) * g - 12 * u[0] + 8 * u[1] - (8 / 5) * u[2]
        return [u_ghost_1, u_ghost_2]

    else:
        # Fall back to 2nd-order
        u_ghost_1 = 2.0 * g - u[0]
        u_ghost_2 = 2.0 * g - u[1] if len(u) > 1 else u_ghost_1
        return [u_ghost_1, u_ghost_2]


def high_order_ghost_neumann(
    interior_values: list[float],
    flux_value: float,
    dx: float,
    outward_normal_sign: float = 1.0,
    order: int = 4,
    grid_type: GridType = GridType.CELL_CENTERED,
) -> list[float]:
    """
    RETIRED (#1936): this never delivered the order it advertised.

    Every branch was measured against an exact polynomial on a cell-centred max
    wall (dx=1, face at x=0, interior at x=-0.5, -1.5, ...; ghost at x=+0.5),
    fed the exact face derivative. On ``u = x`` -- which any second-order ghost
    formula reproduces exactly -- it returned:

        order=4 (3rd-order one-sided stencil): -0.5909, exact +0.5   err 1.09
        order=5 (4th-order one-sided stencil): -0.4600, exact +0.5   err 0.96
        order<4 fallback (u[0] + 2*dx*g):      +1.5000, exact +0.5   err 1.00

    ``ghost_cell_neumann`` under that same harness returns +0.5 exactly. The
    fallback's ``2*dx`` is a vertex-centred step length used on a cell-centred
    layout, and the two high-order branches impose the derivative constraint at
    the ghost centre rather than at the face. The vertex-centred branch's
    ``if order >= 4`` arm and its ``else`` computed the same expression.

    Nothing called this, in the package or the tests, which is why the error
    survived to be found by reading rather than by a failure. It raises instead
    of being deleted because it shipped in v0.21.0.

    Use :func:`ghost_cell_neumann`, or state the required order on #1936 so a
    replacement can be derived against a convergence test.
    """
    raise NotImplementedError(
        "high_order_ghost_neumann is RETIRED (#1936): all three branches are "
        "wrong -- none reproduces u = x, where ghost_cell_neumann is exact. "
        "Use ghost_cell_neumann for the second-order Neumann ghost value."
    )


# =============================================================================
# Physics-Aware Ghost Cell Formulas
# =============================================================================
# IMPORTANT LESSON: The discretized BC must match the physics, not just the
# mathematical form. For advection-diffusion equations (like Fokker-Planck),
# a "no-flux" BC means J·n = 0 where J = v*rho - D*grad(rho).
#
# - Naive approach: Neumann (d rho/dn = 0) only zeroes diffusion flux
# - Correct approach: Robin BC that zeroes TOTAL flux
#
# This distinction is crucial for mass conservation in FP equations.
# =============================================================================


def ghost_cell_fp_no_flux(
    interior_value: float,
    drift_velocity: float,
    diffusion_coeff: float,
    dx: float,
    outward_normal_sign: float = 1.0,
    grid_type: GridType = GridType.CELL_CENTERED,
) -> float:
    """
    Compute ghost cell value for Fokker-Planck no-flux (zero total flux) BC.

    IMPORTANT: For advection-diffusion equations like Fokker-Planck, a "no-flux"
    BC means the TOTAL flux J = v*rho - D*grad(rho) = 0, not just d rho/dn = 0.

    This requires a Robin-type ghost cell formula that accounts for both
    advection and diffusion contributions to the flux.

    Mathematical derivation:
        Total flux: J = v*rho - D*d rho/dx
        No-flux BC: J*n = 0 at boundary

        At boundary (cell face for cell-centered):
            v_n * rho_face - D * (d rho/dn)_face = 0

        Using cell-centered discretization:
            rho_face = (rho_ghost + rho_interior) / 2
            d rho/dn = (rho_ghost - rho_interior) / dx

        Substituting and solving for rho_ghost:
            v_n * (rho_ghost + rho_interior)/2 = D * (rho_ghost - rho_interior)/dx
            rho_ghost = rho_interior * (2D + v_n*dx) / (2D - v_n*dx)

        Physical interpretation:
            - When v_n > 0 (outflow): rho_ghost > rho_interior (diffusion opposes outflow)
            - When v_n < 0 (inflow): rho_ghost < rho_interior (diffusion opposes inflow)
            - When v_n = 0: rho_ghost = rho_interior (pure Neumann)

    Args:
        interior_value: Density at interior point rho_interior
        drift_velocity: Normal component of drift velocity v*n (positive = outward)
        diffusion_coeff: Diffusion coefficient D = sigma^2/2
        dx: Grid spacing
        outward_normal_sign: +1 for max boundary (outward normal points positive),
                            -1 for min boundary (outward normal points negative)
        grid_type: Grid type (cell-centered or vertex-centered)

    Returns:
        Ghost cell value that ensures zero total flux at boundary

    Example:
        >>> # Left boundary with leftward drift (into boundary)
        >>> rho_ghost = ghost_cell_fp_no_flux(
        ...     interior_value=1.0,
        ...     drift_velocity=-0.5,  # v < 0, drift toward left boundary
        ...     diffusion_coeff=0.125,  # D = 0.5^2/2
        ...     dx=0.1,
        ...     outward_normal_sign=-1.0  # Left boundary
        ... )

    References:
        - Achdou & Lauriere (2020): Mean Field Games and Applications, Section on FP BCs
        - LeVeque (2002): Finite Volume Methods for Hyperbolic Problems
    """
    D = diffusion_coeff
    v_n = drift_velocity * outward_normal_sign  # Normal velocity (positive = outward)

    if grid_type == GridType.VERTEX_CENTERED:
        # Vertex-centered: boundary at grid point
        # rho_ghost = rho_interior * (D + v_n*dx) / (D - v_n*dx)
        numerator = D + v_n * dx
        denominator = D - v_n * dx
    else:
        # Cell-centered: boundary at cell face
        # rho_ghost = rho_interior * (2*D + v_n*dx) / (2*D - v_n*dx)
        numerator = 2.0 * D + v_n * dx
        denominator = 2.0 * D - v_n * dx

    # Handle edge case where denominator is near zero
    # This happens when diffusion is very small and drift is large
    if abs(denominator) < 1e-12:
        # Fall back to pure advection limit: reflect density
        return interior_value

    return interior_value * (numerator / denominator)


def ghost_cell_advection_diffusion_no_flux(
    interior_value: float,
    velocity_normal: float,
    diffusion_coeff: float,
    dx: float,
    grid_type: GridType = GridType.CELL_CENTERED,
) -> float:
    """
    Alias for ghost_cell_fp_no_flux with clearer parameter naming.

    This is the same as ghost_cell_fp_no_flux but with velocity_normal
    already accounting for the boundary orientation (positive = outward flow).

    Use this for general advection-diffusion equations where the no-flux BC
    means zero total flux J = v*u - D*grad(u) = 0.
    """
    # velocity_normal is already v*n (positive = outward)
    return ghost_cell_fp_no_flux(
        interior_value=interior_value,
        drift_velocity=velocity_normal,
        diffusion_coeff=diffusion_coeff,
        dx=dx,
        outward_normal_sign=1.0,  # Already accounted for in velocity_normal
        grid_type=grid_type,
    )


# =============================================================================
# Extrapolation Ghost Cell Formulas (for unbounded domains)
# =============================================================================


def ghost_cell_linear_extrapolation(
    interior_values: tuple[float, float],
) -> float:
    """
    Compute ghost cell value using linear extrapolation.

    This is equivalent to the **Zero Second Derivative Condition** (d^2 u/dx^2 = 0
    at the boundary). The function is assumed to continue linearly beyond the
    computational domain.

    Mathematical derivation:
        Let u_0 = first interior point, u_1 = second interior point
        Linear extrapolation: u_ghost = 2*u_0 - u_1

        This ensures: (u_ghost - 2*u_0 + u_1) / dx^2 = 0  (zero second derivative)

    Use cases:
        - HJB value functions on truncated unbounded domains
        - Far-field boundary conditions where solution grows linearly
        - Outflow boundaries in steady-state problems

    Args:
        interior_values: Tuple of (u_0, u_1) where u_0 is adjacent to ghost,
                        u_1 is one cell further into the interior

    Returns:
        Ghost cell value from linear extrapolation

    Example:
        >>> # At right boundary with interior values
        >>> u_ghost = ghost_cell_linear_extrapolation((u[-1], u[-2]))
        >>> # At left boundary with interior values
        >>> u_ghost = ghost_cell_linear_extrapolation((u[0], u[1]))

    Note:
        For problems with quadratic growth (e.g., LQG control), use
        ghost_cell_quadratic_extrapolation() instead.
    """
    u_0, u_1 = interior_values
    return 2.0 * u_0 - u_1


def ghost_cell_quadratic_extrapolation(
    interior_values: tuple[float, float, float],
) -> float:
    """
    Compute ghost cell value using quadratic extrapolation.

    This is equivalent to the **Zero Third Derivative Condition** (d^3 u/dx^3 = 0
    at the boundary). The function is assumed to continue quadratically beyond
    the computational domain.

    Mathematical derivation:
        Let u_0, u_1, u_2 = three interior points (u_0 adjacent to ghost)
        Quadratic extrapolation: u_ghost = 3*u_0 - 3*u_1 + u_2

        This ensures the third derivative vanishes at the boundary.

    Use cases:
        - LQG-type HJB problems with quadratic value functions
        - Problems where linear extrapolation creates artificial "kinks"
        - Higher-accuracy far-field conditions

    Args:
        interior_values: Tuple of (u_0, u_1, u_2) where u_0 is adjacent to ghost,
                        u_1 is one cell in, u_2 is two cells into interior

    Returns:
        Ghost cell value from quadratic extrapolation

    Example:
        >>> # At right boundary
        >>> u_ghost = ghost_cell_quadratic_extrapolation((u[-1], u[-2], u[-3]))
        >>> # At left boundary
        >>> u_ghost = ghost_cell_quadratic_extrapolation((u[0], u[1], u[2]))

    Note:
        Requires at least 3 interior points. For smaller domains, use
        ghost_cell_linear_extrapolation() instead.
    """
    u_0, u_1, u_2 = interior_values
    return 3.0 * u_0 - 3.0 * u_1 + u_2


__all__ = [
    # Ghost cell helpers (2nd-order)
    "ghost_cell_dirichlet",
    "ghost_cell_neumann",
    "ghost_cell_robin",
    # High-order ghost cell extrapolation (4th/5th order for WENO)
    "high_order_ghost_dirichlet",
    "high_order_ghost_neumann",
    # Physics-aware ghost cell (for advection-diffusion/FP)
    "ghost_cell_fp_no_flux",
    "ghost_cell_advection_diffusion_no_flux",
    # Extrapolation ghost cell (for unbounded domains)
    "ghost_cell_linear_extrapolation",
    "ghost_cell_quadratic_extrapolation",
]

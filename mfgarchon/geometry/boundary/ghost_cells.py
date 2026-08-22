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

    One formula, both centrings, both walls:
        du/dn = (u_ghost - u_interior) / dx = g
        => u_ghost = u_interior + dx*g

    The separation is `dx` on either centring, and du/dn already carries the wall's direction,
    so there is no sign argument. See the note below for the `2*dx` version this replaced (#1972).

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
    RETIRED (#1936): the cell-centred branches cannot reproduce a constant field.

    For a Dirichlet ghost the prescribed value is itself a value of `u` at the face, so every
    coefficient must sum to 1 or the formula cannot return `u` for `u` constant. Measured on
    `u = 1` with `boundary_value = 1`, where both ghosts must be exactly 1.0:

        CELL_CENTERED (the DEFAULT)  order=4  ->  [1.6, 4.0]
        CELL_CENTERED                order=5  ->  [1.5, 3.3333]  (needs 4 interior values;
                                                            with 3 it falls to the order=4 row)
        CELL_CENTERED                order<4  ->  [1.0, 1.0]     (the fallback is fine)
        VERTEX_CENTERED              all      ->  [1.0, 1.0]     (correct)

    So the default path is wrong from constants onward -- more elementary than the failure that
    retired `high_order_ghost_neumann` alongside it, whose order=4 branch at least reproduces a
    constant and only breaks from linear on.

    Nothing called this, in the package or the tests. Retired rather than deleted for the same
    reason as its neighbour: it shipped in v0.21.0, so the import stays valid.

    Use :func:`ghost_cell_dirichlet`, which returns 1.0 here.
    """
    raise NotImplementedError(
        "high_order_ghost_dirichlet is RETIRED (#1936): its cell-centred branches -- the "
        "default grid_type -- cannot reproduce a constant field, returning [1.6, 4.0] at "
        "order=4 and [1.5, 3.3333] at order=5 where u = 1 requires [1.0, 1.0]. "
        "Use ghost_cell_dirichlet."
    )


def high_order_ghost_neumann(
    interior_values: list[float],
    flux_value: float,
    dx: float,
    outward_normal_sign: float = 1.0,
    order: int = 4,
    grid_type: GridType = GridType.CELL_CENTERED,
) -> list[float]:
    """
    RETIRED (#1936): first-order in the ghost value at the max wall, where the rule it improves
    on is third. Cell-centred throughout; see defect 3 for the min wall and the other rates.

    Not uniformly wrong, which is why reading it did not settle the question. Measured on
    `u = x` with the exact face derivative fed in, cell-centred, `dx=1`, ghosts nearest-first:

        max wall (sign=+1, the DEFAULT)   order=4  -0.5909   order=5  -0.4600   exact +0.5
        min wall (sign=-1)                order=4  -0.5000   order=5  -0.5000   exact -0.5  (exact)
        max wall                          order<4  +1.5, +2.5   exact +0.5, +1.5
        min wall                          order<4  +2.5, +5.5   exact -0.5, -1.5

    Three separable defects, and the headline is the third:

    1. `flux_value * outward_normal_sign` is used as the INWARD derivative. At the min wall that
       coincides with the truth, which is why those two rows are exact; at the max wall it is
       negated. The whole max-wall error is that one term: order=4 is off by exactly `12*h*g/11`
       (1.0909) and order=5 by `24*h*g/25` (0.9600).

    2. Both high-order branches impose the derivative constraint at the GHOST CENTRE, not at the
       face -- `(-11*u_{-1} + 18*u_0 - 9*u_1 + 2*u_2)/(6h)` is `p'` at `u_{-1}`. A linear field has
       the same derivative everywhere, so this is invisible above and shows only where the flux
       term cannot mask it: `u = x^2` with `g = 0` gives 0.5455 (order=4) and 0.4800 (order=5).

    3. Defect 1 ALONE costs the order the name promises -- not the pair. On `u = exp(x)`,
       cell-centred, over h = 0.2 -> 0.025:

           max wall (defect 1 present)   rate 1.04, 1.02, 1.01     -> O(h)
           min wall (defect 1 absent)    rate 1.94, 1.98, 1.99     -> O(h^2), defect 2 alone
           `ghost_cell_neumann`          rate 3.00, 3.00, 3.00     -> O(h^3)

       So it is worse than the rule it was written to improve on at BOTH walls, and a derivative
       built from the max-wall ghost is O(1) wrong.

    The `order<4` fallback is a separate failure of a different kind: `u[0] + 2*dx*g` is the
    repo's own pre-#1972 formula, `interior + 2*dx*g*sign`, struck in this file on 2026-08-18 --
    a retired rule that survived in a copy nothing was checking. It is not the only one: the
    one-cell form `u_ghost = u_interior + 2*dx*g` survives at SIX sites. #1936 corrects two of
    them, here and in `NeumannCalculator`; `protocols.py:226/487/492` and the user guide still
    carry it, and #2057 tracks those. (`u_next_interior +- 2*dx*g` in `_compat.py` and
    `applicator_fdm.py` is a different, correct formula -- two cells, so 2*dx is right there.)

    Also, the vertex-centred branch's `if order >= 4` arm and its `else` computed the same
    expression, so that path advertised fourth order and delivered constant-derivative
    extrapolation.

    Nothing called this, in the package or the tests, which is why the errors survived to be found
    by reading rather than by a failure. It raises instead of being deleted because it shipped:
    added 2025-12-17 by `1a1ebec6`, present and exported in v0.20.0 and every tag since.

    Use :func:`ghost_cell_neumann`. A correct face-constrained 4-point formula is on #1936, derived
    twice independently during review; it is not implemented here because #1936's own acceptance
    criterion is that the ghost value gain an owner and the implementation count DROP, and adding
    a sixth implementation for zero consumers moves that counter the wrong way.
    """
    raise NotImplementedError(
        "high_order_ghost_neumann is RETIRED (#1936). Cell-centred ghost value, u = exp(x): it "
        "converges at O(h) at the max wall and O(h^2) at the min, where ghost_cell_neumann gives "
        "O(h^3) -- worse than the rule it was written to improve on, at both walls. It uses the "
        "flux as the inward derivative, which is why order=4 and order=5 are exact at the min "
        "wall and negated at the max; its order<4 fallback is the pre-#1972 "
        "`interior + 2*dx*g*sign` that this file already struck. "
        "Use ghost_cell_neumann for the Neumann ghost value."
    )


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

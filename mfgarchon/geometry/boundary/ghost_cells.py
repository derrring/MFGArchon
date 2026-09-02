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
    # `u = a*x`, which any consistent first-order ghost rule reproduces exactly; O(h^2) on
    # `u = x^2`. The word `consistent` is #1972's own -- it is in that commit's message and was
    # dropped on the way into this file. Without it the clause is false, and its counterexample is
    # the rule #1972 removed: 6 of the 8 non-zero-slope ghosts it produces are wrong. (#2129, which
    # attacked this line for a different and mistaken reason and is closed invalid.)
    return interior_value + dx * flux_value


def ghost_cell_robin(
    interior_value: float,
    rhs_value: float,
    alpha: float,
    beta: float,
    dx: float,
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

    There is NO sign argument, at either centring. `du/dn` already carries the wall's direction,
    and the ghost sits outside while the interior sits inside at both walls, so
    `(u_ghost - u_interior)/dx` is the outward normal derivative either way.

    This took an `outward_normal_sign` until #2063, unused cell-centred and APPLIED BACKWARDS
    vertex-centred. `RobinCalculator` was the only caller in the package that passed it, and
    passing the physically correct -1.0 at the min wall is what broke it: measured on `u = 3x`,
    dx = 0.1, alpha = 0, beta = 1, it returned +0.3000 where -0.3000 is exact. The seven callers
    that omitted it took the +1.0 default and were right by accident. Ignoring the argument
    entirely is exact on 8 of 8 -- two centrings, two walls, two slopes -- which is why it is
    gone rather than merely un-defaulted. Same removal, and the same reason, as #1972's from
    `ghost_cell_neumann`.

    """
    if grid_type == GridType.VERTEX_CENTERED:
        # ONE formula, every alpha. The wall IS the interior node here, so u_b = u_interior and
        # du/dn = (u_ghost - u_interior)/dx; substituting into alpha*u_b + beta*du/dn = g gives
        # the line below. Verified exact on 16 combinations -- 2 walls x 2 (slope, offset) x
        # {(0,1), (2,1), (2,0.5), (-1.5,2)} -- with zero failures.
        #
        # This was two branches until #2064, split on `abs(alpha) > 1e-12` purely to avoid dividing
        # by beta, and the alpha != 0 arm solved for a quantity multiplied by alpha rather than for
        # the ghost: it returned -10.5 where 3.3 is exact, at BOTH walls, independently of any sign.
        # The split is what hid the fact that no sign term belongs here at all -- du/dn already
        # carries the wall's direction, which is why #2063 could remove `outward_normal_sign`.
        # beta = 0 is COMPUTABLE, not undetermined, and an earlier revision of this change raised
        # on it. `alpha*u + 0*du/dn = g` is the Dirichlet condition `u = g/alpha`; the old two-arm
        # code returned exactly that (measured: alpha=2, beta=0, g=6 gave 3.0), and
        # `enforcement.py`'s `enforce_robin_value_nd` computes the same `rhs/alpha` rather than
        # refusing. Raising here would have converted a correct answer into an error and put this
        # owner at odds with that one.
        #
        # The threshold that raise used was dimensionally wrong as well: `|beta| < 1e-12` is not
        # scale-free, so the same physical condition scaled by 1e-13 was refused while the exact
        # answer was computable, and `alpha=1e6, beta=1e-11` passed and returned a result 182%
        # wrong. An FP wall sets `beta = -D`, so `sigma = 1e-6` would have been refused and told to
        # become an absorbing wall -- the opposite condition. (#2064)
        _beta = np.asarray(beta, dtype=float)
        _alpha = np.asarray(alpha, dtype=float)
        _degenerate = (np.abs(_beta) == 0.0) & (np.abs(_alpha) == 0.0)
        if np.any(_degenerate):
            raise ValueError(
                "ghost_cell_robin: alpha = beta = 0 does not constrain the solution, so no ghost "
                "value exists. This is the only degenerate case: beta = 0 alone is the Dirichlet "
                "condition u = g/alpha and is computed, not refused."
            )
        if np.any(np.abs(_beta) == 0.0):
            # Pure Dirichlet: the wall IS the interior node here, so the value there is g/alpha and
            # the ghost is the linear continuation through it from the interior. With no derivative
            # constraint the only defensible continuation is the constant one.
            _dirichlet = np.where(np.abs(_beta) == 0.0, rhs_value / np.where(_alpha == 0.0, 1.0, _alpha), 0.0)
            _robin = np.where(
                np.abs(_beta) == 0.0,
                0.0,
                interior_value + dx * (rhs_value - alpha * interior_value) / np.where(_beta == 0.0, 1.0, _beta),
            )
            result = np.where(np.abs(_beta) == 0.0, _dirichlet, _robin)
            return float(result) if np.ndim(result) == 0 else result
        return interior_value + dx * (rhs_value - alpha * interior_value) / beta

    # Cell-centered: ghost and interior are dx apart, and the geometry is symmetric --
    # ghost at -dx/2 / interior at +dx/2 on the left, interior at L-dx/2 / ghost at L+dx/2 on the
    # right -- so (u_ghost - u_interior)/dx is the outward normal derivative at both walls.
    coeff_ghost = alpha / 2.0 + beta / dx
    coeff_interior = alpha / 2.0 - beta / dx

    # alpha, beta and rhs_value may be FIELDS -- one value per boundary point. The impermeable
    # wall of a Fokker-Planck equation is this condition with alpha = the outward normal drift,
    # which varies along the boundary and is recomputed each Picard iterate. The arithmetic
    # above is already elementwise; only this guard needed to stop assuming a scalar.
    # #2217: the threshold must be RELATIVE, because `coeff_ghost` carries units and the pair
    # (alpha, beta) has no canonical normalisation. Scaling the pair by any non-zero constant
    # leaves the Robin condition unchanged WHEN `rhs_value` IS ZERO -- and only then, because
    # scaling `alpha` and `beta` by `s` turns `alpha*u + beta*du/dn = g` into `... = g/s`. At
    # `g = 0` that is no condition at all and the ghost is invariant; at `g != 0` it is a different
    # wall, and the ghost moves by 99% at `alpha=2, beta=0.5, dx=0.1, g=0.7` far from any
    # cancellation. The docstring of `normal_frame_coefficients` states the same qualification and
    # is the copy to follow; an unqualified version of this sentence shipped in #2221 and was
    # corrected in review. What the verdict must be invariant under is therefore the SCALING, which
    # `coeff_ghost` depends on and the physical wall does not -- against an absolute 1e-12 it was
    # not: over 144 inputs built to straddle the cancellation point the verdict flipped under
    # rescaling in 96. Concretely, and this is the sentence worth keeping -- a caller that multiplied
    # its pair by 100 to keep the coefficients in familiar units moved the refusal boundary by 100.
    #
    # The scale is `max(|alpha|/2, |beta|/dx)` -- the larger of the two terms whose cancellation
    # this predicate is about -- so it fires on GENUINE cancellation rather than on both terms
    # merely being small. Two rejected alternatives. `|alpha|/2 + |beta|/dx` is equally invariant
    # and strictly more conservative for no gain. Normalising the pair by `max(|alpha|, |beta|)` --
    # #2217's own preferred option -- fails for a reason this comment got WRONG before review: it
    # is exactly as scale-invariant as the chosen one, both being positively homogeneous of degree
    # 1, and the "4 of 144" once cited here was rounding in decimal rescalings of on-threshold
    # inputs, not scale-variance (under exact power-of-two rescalings both flip 0). Its real defect
    # is dimensional: its EFFECTIVE tolerance drifts with `dx` -- measured 2.0e-12 at `dx = 0.01`
    # and 0.1, then 1.0e-11 at `dx = 10` and 1.0e-10 at `dx = 100`, where the chosen scale holds
    # 1.0e-12 throughout.
    #
    # `<=` and not `<`, and that is the whole handling of the degenerate pair: at alpha = beta = 0
    # the scale is 0, so a strict `<` would compare `0 < 0` and ANSWER a condition that carries no
    # information at all. `<=` raises there, as the absolute threshold did.
    # NOT invariant at the top of the float range, and the qualification is owed: at
    # `alpha=2, beta=0.5, dx=0.1` -- `coeff_ghost = 6`, nowhere near cancellation -- `s = 1e308`
    # overflows `beta/dx` to `inf`, so `coeff_scale` is `inf` and `inf <= 1e-12*inf` is True and this
    # raises. Powers of two do not escape it (`s = 2**1023` also raises). Exotic, and left unrepaired
    # because a pair that overflows has no meaningful ghost either way -- but "exactly
    # scale-invariant" is false as an unqualified claim, and the message below then prints
    # `alpha/2 + beta/dx = 0` for a value that is `inf`.
    #
    # DIRECT CALLERS: this refusal set moves in BOTH directions for anyone not coming through
    # `ghost_cell_fp_no_flux`, and three modules do (`calculators.py`, `_compat.py`,
    # `applicator_fdm.py`), two of them passing a non-zero `g`. The new/old threshold ratio is
    # `S = max(|alpha|/2, |beta|/dx)`: it widens the refusal set when `S > 1` and narrows it when
    # `S < 1`. Measured at `g = 0.7`: `alpha=-200, beta=1, dx=0.01` went ok -> raise, and
    # `alpha=-0.02, beta=0.01, dx=1` went raise -> ok. Exact cancellation still refuses under both.
    coeff_scale = np.maximum(np.abs(alpha) / 2.0, np.abs(beta) / dx)
    singular = np.abs(coeff_ghost) <= 1e-12 * coeff_scale
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
    one-cell form survived in several docstrings, corrected across #1936 and #2057.

    NO COUNT IS GIVEN HERE, deliberately. Two revisions of this note carried one, and both were
    right for a spelling and wrong for the population: `u_ghost = u_interior + 2*dx*g` and
    `u_g = u_i +- 2*dx*g` are the same claim, and counting the first missed `applicator_fdm`'s own
    module header, which stated the one-cell form with a two-cell step and was wrong at both walls.
    An earlier version of this note EXCLUDED that site, asserting the occurrence there was the
    two-cell `u_next_interior +- 2*dx*g` -- a string that appears zero times in that file. A
    miscounted exclusion shielded the site the count existed to find, so the count is gone rather
    than corrected: a tally over a hand-chosen literal cannot audit the predicate that chose it.

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


def normal_frame_coefficients(
    drift_velocity: float,
    diffusion_coeff: float,
    outward_normal_sign: float,
) -> tuple[float, float]:
    """The Robin pair ``(alpha, beta)`` for ``J . n = 0``, from AXIS-frame ``v`` and ``D``.

    The Fokker-Planck no-flux condition is ``J . n = 0`` with ``J = v*rho - D*grad(rho)``. Written
    as a Robin condition ``alpha*rho + beta*drho/dn = 0`` it is ``alpha = v . n`` and
    ``beta = -D`` -- so this returns ``(drift_velocity * outward_normal_sign, -diffusion_coeff)``.

    ~~``alpha = v``, ``beta = -outward_normal_sign * D``~~ [CORRECTED before merge]. That pair is
    ``outward_normal_sign`` TIMES this one. It yields the same ghost, because scaling both
    coefficients leaves the Robin condition unchanged when the right-hand side is zero -- measured,
    2000 combinations, zero value differences and zero raise/no-raise differences. But it is not the
    normal-frame pair this function is named for, it diverges the moment a caller has a non-zero
    ``g`` (1.045455 against 0.590909 at ``g = 0.5``), and the docstring below it claimed a
    projection of ``drift_velocity`` that the returned value did not perform.

    WHY THIS HAS A NAME. ``ghost_cell_robin`` takes coefficients that are ALREADY in the normal
    frame, which is why it correctly has no sign argument -- ``du/dn`` carries the wall's direction,
    and #2063 removed a sign from it after measuring it unused at cell-centring and applied
    BACKWARDS at vertex-centring. ``ghost_cell_fp_no_flux`` takes axis-frame physical quantities, so
    something has to do the projection. That something was a second copy of Robin's algebra (#2128).
    It is one step, it belongs to neither caller, and a second caller holding ``v`` and ``D`` would
    otherwise write the copy again -- which is the failure this package has now measured twice
    (#2214, and ``applicator_implicit`` in #1936).

    ``outward_normal_sign`` is the factor converting ``drift_velocity`` to ``v . n``, NOT "which wall
    this is": a caller already holding ``v . n`` passes ``+1.0`` at BOTH walls. Reading it as a wall
    identifier is what produced #2063.

    Verified against an EXTERNAL oracle, not against the delegation: the exact zero-flux profile
    ``rho_interior * exp(v_n*dx/D)``, which neither this function nor :func:`ghost_cell_robin`
    implements, is approached at each branch's order (~3 cell, ~2 vertex) under refinement, and a
    flipped conversion sign approximates ``exp(-z)`` instead and collapses that order. Per-element
    values are pinned separately at both centrings and both wall directions.

    IF YOU ARE THE SECOND CALLER THIS EXISTS FOR, READ THIS. ~~Handing this pair straight to
    :func:`ghost_cell_robin` gives the right ghost and the WRONG refusal set … any caller routing
    this pair into robin needs the same ``2*dx`` scaling until #2217 lands.~~
    [SUPERSEDED 2026-09-02] SUPERSEDED-BY: #2217. Every sentence of that paragraph was falsified by
    the change that fixed it, and the correction commit for the round before this one CITED this
    docstring as the copy to follow while its own diff was making it false. Robin's threshold is now
    relative, ``v_n = 0, D = 5e-12, dx = 10`` answers unscaled, :func:`ghost_cell_fp_no_flux` no
    longer scales, and no caller needs to.

    WHAT STILL HOLDS, and it is the part that was never about the threshold: with a non-zero
    right-hand side, rescaling the pair is NOT free. Scaling ``alpha`` and ``beta`` by ``s`` turns
    ``alpha*u + beta*du/dn = g`` into ``... = g/s``, a different wall -- three plausible choices give
    0.944, -0.167 and 0.944 at ``g = 0.5``. The Fokker-Planck wall always passes ``g = 0``, which is
    why scaling was ever safe there; a caller with ``g != 0`` has a question this function does not
    answer.
    """
    return drift_velocity * outward_normal_sign, -diffusion_coeff


def ghost_cell_fp_no_flux(
    interior_value: float,
    drift_velocity: float,
    diffusion_coeff: float,
    dx: float,
    outward_normal_sign: float,
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

        Vertex-centered, where the wall IS the interior node, so the wall density is
        rho_interior rather than a face average (#2068):
            v_n * rho_interior = D * (rho_ghost - rho_interior)/dx
            rho_ghost = rho_interior * (D + v_n*dx) / D

        THE TWO ARE NOT THE SAME ORDER. The exact zero-flux profile is
        rho(s) = rho(wall)*exp(v_n*s/D) along the outward normal, so each form is a rational
        approximation to exp(z), z = v_n*dx/D. Measured against it (#2068):

            old vertex   (1+z)/(1-z) = 1 + 2z + ...   O(dx),    rate 1.06   [CORRECTED AWAY]
            vertex       1 + z                        O(dx^2),  rate 2.01
            cell-centred (1+z/2)/(1-z/2)              O(dx^3),  rate 3.04   <- Pade(1,1) of exp

        THE TABLE IS THE ERROR IN THE GHOST VALUE, which is one order above the order in which each
        form imposes the CONDITION: cell-centred imposes it to second order, the vertex form to
        first. Both readings are in use -- `calculators.py` labels these "(2nd order)" and
        "(1st order)", which is the condition reading -- and conflating them is why the clause below
        looks like a contradiction. It is not: a second-order vertex ghost in the CONDITION sense
        needs two interior values and this signature carries one; that is a property of the
        interface, not something to fix here.

        The struck row is kept because it is the evidence for the current one, not a description of
        live code. Its leading term is 2z where the condition requires z -- inconsistent rather than
        merely coarse -- and it carried a pole at dx = D/v_n, HALF the cell-centred limit 2D/v_n
        because the wrong geometry doubles the effective step: at D = 0.125, rho = 1, v_x = +0.5 it
        returned 499 at dx = 0.249 and -501 at dx = 0.251. The current vertex form is linear in dx
        and has no pole. #2128 moved this function's algebra into `ghost_cell_robin`; these
        measurements are about the SCHEME and outlive the implementation that carried them.

        Physical interpretation:
            - When v_n > 0 (outflow): rho_ghost > rho_interior (diffusion opposes outflow)
            - When v_n < 0 (inflow): rho_ghost < rho_interior (diffusion opposes inflow)
            - When v_n = 0: rho_ghost = rho_interior (pure Neumann)

    Args:
        interior_value: Density at interior point rho_interior
        drift_velocity: Drift velocity in the AXIS direction, v_x -- not v*n. This routine
            multiplies it by `outward_normal_sign` to obtain v*n, so passing v*n instead
            double-counts the orientation: measured at a min wall, that leaves a total flux
            residual of 1.25 rather than 0. A caller that already holds v*n passes it here with
            `outward_normal_sign=1.0`, which is what `ghost_cell_advection_diffusion_no_flux`
            does. No default: it is required precisely because the two callers disagree about
            which quantity they hold, and +1.0 silently means "max wall" (#2063)
        diffusion_coeff: Diffusion coefficient D = sigma^2/2
        dx: Grid spacing
        outward_normal_sign: the factor converting `drift_velocity` to v*n -- NOT "which wall
            this is". A caller holding the axis velocity v_x passes the wall's outward normal
            (+1 max, -1 min), which is what `ZeroFluxCalculator` does; a caller already holding
            v*n passes +1.0 at BOTH walls, which is what
            `ghost_cell_advection_diffusion_no_flux` does. Reading it as a wall identifier is what
            made the `drift_velocity` line above say v*n (#2063). Required, no default.
        grid_type: Grid type (cell-centered or vertex-centered)

    Returns:
        Ghost cell value that ensures zero total flux at boundary

    Raises:
        NotImplementedError: ``diffusion_coeff`` has more than one element, at either centring.
            ``main`` raised ``ValueError`` for a multi-element ndarray and ``TypeError`` for a list
            or tuple; this refuses everything but size 1, with one type, on both centrings (#2128).
            A field-valued ``drift_velocity`` IS supported.
        ValueError: cell-centred, at ``2D = v_n*dx``, where the ghost's own coefficient vanishes and
            the condition does not determine it. Raised by :func:`ghost_cell_robin`, which this
            delegates to. Before #2128 this returned ``interior_value`` silently.

    KNOWN, UNGUARDED, AND NOT INTRODUCED HERE: cell-centred with ``D = 0`` and ``v_n != 0`` returns a
    NEGATIVE density -- ``-1.0`` for ``interior=1, v=0.4, dx=0.1`` -- because the closed form becomes
    ``(v_n*dx)/(-v_n*dx)``. ``main`` returns the same value by the same arithmetic, so this is
    inherited rather than caused. The vertex path guards ``D ~ 0`` and returns ``interior_value``, so
    the two centrings disagree about zero diffusion. #2220 asks the physics question for both sides
    at once and reframes it: at ``D = 0`` the equation is first order, so an outflow wall admits no
    boundary condition and an inflow wall admits exactly one. A ghost-cell scheme must still supply a
    number either way -- what is absent at outflow is a number the BOUNDARY CONDITION determines, and
    the standard choice there is extrapolation. Stated here because #2128's acceptance asks for
    ``D -> 0`` to have a stated behaviour, and "it returns a negative density and nobody decided
    that" is the honest one.

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
    # MULTI-ELEMENT `D` is refused on BOTH centrings. Placed before the conversion so a list reaches
    # this message rather than dying on `-diffusion_coeff` with a bare TypeError.
    #
    # THE PREDICATE IS `size != 1`, NOT `ndim != 0`, and the difference is not pedantry. `main`'s
    # guard was `if abs(denominator) < 1e-12`, and `bool()` on a ONE-ELEMENT array is legal, so
    # `main` ACCEPTED shape-(1,) and (1,1); it raised `ValueError` for a multi-element ndarray and
    # `TypeError` for a list or tuple. `ndim != 0`
    # refuses the size-1 case too -- an undeclared capability removal inside a consolidation, which
    # is the thing this branch keeps promising not to do, and a draft of this guard shipped it. That
    # acceptance is an accident of `bool()` on a one-element array rather than a designed feature; it
    # is preserved because #2128 is a consolidation, and whether to keep it is a separate question.
    #
    # Without the guard the vertex path -- a scalar `abs()` -- raises on an array while the cell path
    # now answers: an asymmetry this delegation would have introduced by side effect, which #2128's
    # own acceptance ("works or raises identically on both centrings") forbids. Field-valued DRIFT is
    # untouched and now works on BOTH centrings, where `main` raised on the cell path.
    #
    # The exception TYPE changed here: `main` raised `ValueError`, this raises `NotImplementedError`.
    # No in-tree caller catches either on this path; an external one might.
    if np.size(diffusion_coeff) != 1:
        raise NotImplementedError(
            f"ghost_cell_fp_no_flux: multi-element diffusion_coeff (shape "
            f"{np.shape(diffusion_coeff)}) is not supported at either centring. The pre-#2128 code "
            f"raised ValueError here for a multi-element ndarray (TypeError for a list or tuple) and "
            f"accepted a size-1 array; this refuses everything but size 1 "
            f"on both centrings rather than enabling it on one only. A field-valued drift_velocity "
            f"IS supported."
        )

    alpha, beta = normal_frame_coefficients(drift_velocity, diffusion_coeff, outward_normal_sign)

    # VERTEX-CENTRED, D ~ 0: old behaviour preserved DELIBERATELY, and it is not obviously right.
    # `J . n = v_n * rho_wall = 0` with `v_n != 0` forces `rho_wall = 0`, and on a vertex grid the
    # wall IS the interior node -- so the condition constrains the INTERIOR value and says nothing
    # about the ghost. The pre-#2128 code returned `interior_value` here (a silent reflection it
    # called "the pure advection limit"). What removing this guard would give is NOT one value:
    # `ghost_cell_robin`'s `beta == 0` branch is entered only at `D == 0.0` EXACTLY, where it returns
    # the implied Dirichlet 0.0; anywhere else in `|D| < 1e-12` beta is non-zero and the ordinary
    # formula runs and blows up -- measured, 4.0e+11 at `D = 1e-13`, -4.0e+11 at `D = -1e-13`. So the
    # guard mostly prevents a blow-up and only at one point prevents a 0.0. Which value
    # belongs here is a physics decision about sigma = 0 problems, and #2128 is a consolidation, so
    # it does not get to make that decision by side effect: the old value stays until #2220 settles
    # it. The threshold is the pre-#2128 one, unchanged.
    if grid_type == GridType.VERTEX_CENTERED and abs(diffusion_coeff) < 1e-12:
        return interior_value

    # CELL-CENTRED at `2D = v_n*dx`: this RAISES, where the pre-#2128 code returned
    # `interior_value`. That change is intended -- the ghost's own coefficient vanishes there, the
    # condition does not determine it, and returning the interior value is the fabricated answer the
    # fail-fast ratchet exists to remove.
    #
    # [#2217] The pair is passed UNSCALED. Until #2217 it was scaled by `2*dx`, because
    # `ghost_cell_robin` compared a quantity with units against an absolute `1e-12` and was
    # therefore scale-blind; the scaling existed only to make robin's tested quantity equal this
    # function's own pre-#2128 predicate `|2D - v_n*dx|`. #2217 made robin's threshold relative, so
    # the scaling became provably inert -- measured, 0 verdict differences across scales
    # `1.0 / 2*dx / 4*dx / 17.3` over 720 inputs -- and removing it is that fix's own tail rather
    # than a cleanup. Removal costs rounding only: 276 of 1792 inputs differ, worst 5 ulp and
    # 9.3e-16 relative, with 0 verdict changes.
    #
    # THE REFUSAL SET MOVED, and that is the point of #2217, not a side effect. It is no longer
    # `|2D - v_n*dx| < 1e-12`. The dominant direction is refusal -> answer, and its headline case is
    # `v_n = 0`, which is homogeneous Neumann: the ghost is exactly `interior_value` and refusing it
    # was the well-posed problem #2217 reports being rejected. Counts here are GRID-DEPENDENT and
    # are not a characterisation -- a 720-input grid shows 12 such changes, a 1944-input grid shows
    # 106, spanning `v_n = 0` and `v_n = +/-1e-13` over several `D`.
    #
    # BOTH DIRECTIONS HAVE THE SAME EXACT CRITERION, and it is the same number. A pre-review
    # version of this comment asserted the answer -> refusal direction did not exist at all, and the
    # version after that gave it an exact `iff` while leaving refusal -> answer as an anecdote about
    # `v_n = 0`. They are mirrors: near cancellation the old code refuses below `D = 1/2` and the new
    # code refuses above it, at every `dx`, and which direction a sweep reports is decided by which
    # side of 1/2 its `D` values sample. Measured on one construction, varying only `D`:
    # `D in {0.6, 1, 2, 10}` gives 232 answer -> refusal and 0 the other way; `D in {0.05, 0.1, 0.25,
    # 0.4}` gives 0 and 1230. So DO NOT call either direction dominant -- that is a property of the
    # grid, exactly as the counts below are.
    # Inputs the old code answered and this one refuses occupy a band with a sharp entry condition:
    # the old effective threshold on `coeff_ghost` was `1e-12/(2*dx)` (the `2*dx` scaling), the new
    # one at cancellation is `1e-12*(D/dx)`, so the band is non-empty exactly when `D > 1/2`.
    # Measured, sweeping `v_n` through cancellation at three spacings: 0 hits at `D = 0.25` and
    # `0.4`, then 28 at `D = 0.6`, 134 at `D = 1.0`, 266 at `D = 10`. Those are not regressions --
    # `2D` and `v_n*dx` agree to ~13 significant figures there and the old answers were cancellation
    # garbage (-2.0e12 at one such point), so refusing is the better behaviour -- but it is a
    # widening of the refusal set on a function whose owner has live callers in four modules, and it
    # is recorded rather than discovered later.
    #
    # Two consequences of the removal worth stating because the old comment block asserted them:
    # the scaled pair's overflow at `dx = D = 1e200` (a silent `nan` where `main` returned 3.0) and
    # its denormal failure at `D = 1e-300, dx = 1e-20` are both gone with the multiplication that
    # caused them. The sign of the old scale was a surviving non-fault; there is now no scale whose
    # sign could be free.
    return ghost_cell_robin(interior_value, 0.0, alpha, beta, dx, grid_type)


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
    "normal_frame_coefficients",
    "ghost_cell_advection_diffusion_no_flux",
    # Extrapolation ghost cell (for unbounded domains)
    "ghost_cell_linear_extrapolation",
    "ghost_cell_quadratic_extrapolation",
]

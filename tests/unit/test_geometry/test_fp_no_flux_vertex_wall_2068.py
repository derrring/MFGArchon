"""#2068: `ghost_cell_fp_no_flux`'s vertex-centred branch used the cell-centred wall geometry.

It was consistent with `rho_face = (rho_g + rho_i)/2` AND `drho/dn = (rho_g - rho_i)/(2*dx)` -- a
wall midway between ghost and interior at separation `2*dx`. On a vertex grid the wall IS the
interior node, which is why `ghost_cell_dirichlet` returns `g` there unmodified and why
`ghost_cell_neumann` uses separation `dx` on both centrings (#1972). So the branch satisfied its own
stencil to machine zero while leaving a total flux that did not converge.

The tests below are deliberately NOT the defining equation rearranged. The first version of this
file was: it asserted `v_n*rho_i == D*(rho_g - rho_i)/dx`, the implemented formula moved across the
equals sign, identically zero at every `dx`. Three properties replace it, plus a control:

- the two centrings discretise the SAME continuous condition, so their leading correction in `dx`
  must agree. The old form's was exactly twice the cell-centred one, which is the "diffusive flux
  is exactly twice what the condition requires" the issue measured.
- the ghost stays within the O(z^2) truncation bound of the exact profile
  `rho(s) = rho(wall)*exp(v_n*s/D)`, and converges to it at rate 2 -- with the cell arm's rate 3 as
  the control that the two have not been collapsed into one formula. That profile is an oracle
  rather than a restatement: nothing in this module produces it.
- the ghost stays BOUNDED across the retired form's pole at `dx = D/v_n`, where that form gives
  +499 / -501.

Every one of them admits a form strictly more accurate than the fix -- the exact `exp(z)` and the
Pade `(1+z/2)/(1-z/2)` pass all of them. That is what this file failed to do the first time, and it
is why none is written as an equality against the shipped values.

`v = 0` is the non-discriminating input: every form here returns `rho_interior` there, so the
`drift=0` case below is a control on the harness, not evidence about the geometry.
"""

from __future__ import annotations

import itertools

import pytest

import numpy as np

from mfgarchon.geometry.boundary.ghost_cells import (
    GridType,
    ghost_cell_fp_no_flux,
    ghost_cell_neumann,
)

_D = 0.125
_RHO = 1.0


@pytest.mark.parametrize("drift", [0.5, -0.5, 1.3])
def test_the_two_centrings_agree_to_leading_order_in_dx(drift):
    """Both discretise `v_n*rho - D*drho/dn = 0`, so `(rho_g - rho_i)` must match as `dx -> 0`.

    Measured at `dx = 3.125e-3`, `v_x = +0.5`: the corrected branch gives a ratio of 0.99375 and
    the retired one 2.01266. The 0.02 tolerance separates them by a factor of 50.
    """
    dx = 3.125e-3
    cell = ghost_cell_fp_no_flux(_RHO, drift, _D, dx, 1.0, grid_type=GridType.CELL_CENTERED) - _RHO
    vertex = ghost_cell_fp_no_flux(_RHO, drift, _D, dx, 1.0, grid_type=GridType.VERTEX_CENTERED) - _RHO
    assert vertex / cell == pytest.approx(1.0, abs=0.02), (
        f"leading correction differs by {vertex / cell:.4f}x; the retired 2*dx geometry gives 2"
    )


def test_the_vertex_ghost_stays_within_the_truncation_bound_of_the_exact_profile():
    """Replaces a test that could not fail. The first version of this file asserted

        v_n*rho_i - D*(rho_g - rho_i)/dx == 0

    which is the implemented formula rearranged: substitute `rho_g = rho_i*(D + v_n*dx)/D` and it
    is identically zero in exact arithmetic, at every `dx`, including `dx = 1e6`. It measured
    floating-point associativity. Worse, it REJECTED the exact ghost `rho_i*exp(z)` -- residual
    -0.1148, -0.0535, -0.0259, -0.0127, -0.0063 over that loop, against a 1e-12 tolerance -- and
    rejected the strictly more accurate Pade form. It was a characterization test pinning `1 + z`
    while its own docstring disclaimed being one.

    The oracle is the exact zero-flux profile, which no discretisation in this module produces:
    `rho(s) = rho(wall)*exp(v_n*s/D)` along the outward normal, so the exact ghost one step out is
    `rho_i*exp(z)` at `z = v_n*dx/D`. The vertex arm is the truncated series `1 + z`, whose error
    is `z^2/2 + O(z^3)`; the bound below carries 50% headroom on that leading term.

    It admits anything at least this accurate -- the exact profile and the Pade form both pass --
    and rejects the retired `(1+z)/(1-z)`, which is the Pade of `exp(2z)` and fails at every
    resolution by one to two orders.
    """
    v_n, d, rho_i = 0.5, 0.125, 1.0
    for dx in (0.1, 0.05, 0.025, 0.0125):
        z = v_n * dx / d
        got = ghost_cell_fp_no_flux(rho_i, v_n, d, dx, 1.0, grid_type=GridType.VERTEX_CENTERED)
        error = abs(got - rho_i * np.exp(z))
        assert error <= 1.5 * z**2 / 2 * rho_i, (
            f"dx={dx}: |ghost - rho_i*exp(z)| = {error:.3e}, past the O(z^2) bound "
            f"{1.5 * z**2 / 2 * rho_i:.3e}. The retired form fails here by 1-2 orders."
        )


def test_the_ghost_stays_bounded_across_the_retired_forms_pole():
    """Why the correction is not merely more accurate: the retired form was singular.

    `(D + v_n*dx)/(D - v_n*dx)` blows up at `dx = D/v_n` -- HALF the cell-centred `2D/v_n`, because
    the wrong geometry doubles the effective step -- and past it returns a negative density, which
    is not a small error in a Fokker-Planck solve. Measured there: `+499.0` at `dx = 0.249` and
    `-501.0` at `dx = 0.251`.

    The assertion is a BOUND, not the fix's exact values, so it does not punish a better formula.
    Over the asserted window the corrected `1 + z` peaks at 2.20, the exact `exp(z)` at 3.32 and
    the Pade form at 4.00, all at the widest `dx`; only the retired one leaves the bound, by two
    orders. (An earlier revision quoted 2.0 / 2.72 / 2.99 as maxima -- those are the values AT the
    pole, not over the window.)

    NOT asserted, deliberately: that the corrected form goes negative under strong INWARD drift at
    `z < -1`. It does -- `z = -4` gives `-3.0` where the exact profile gives `0.0183` -- and an
    earlier version of this test asserted `inward < 0.0`, which turns a resolution bound into a
    contract and makes a positivity-preserving successor read as a regression. The bound is in the
    fragment instead.
    """
    v_x = 0.5
    pole = _D / v_x
    for dx in (0.8 * pole, 0.996 * pole, pole, 1.004 * pole, 1.2 * pole):
        g = ghost_cell_fp_no_flux(_RHO, v_x, _D, dx, 1.0, grid_type=GridType.VERTEX_CENTERED)
        assert np.isfinite(g), f"dx={dx}: ghost={g}"
        assert 0.0 < g <= 10.0 * _RHO, (
            f"dx={dx} (retired pole at {pole}): ghost={g}. The retired form gives +499 / -501 here."
        )


@pytest.mark.parametrize("grid_type", [GridType.CELL_CENTERED, GridType.VERTEX_CENTERED])
def test_zero_drift_reduces_to_the_neumann_ghost(grid_type):
    """The control. Every form returns `rho_interior` here, so this cannot discriminate geometry.

    It is here to catch a different failure: that the FP branch and `ghost_cell_neumann` -- which
    owns the pure-Neumann ghost for both centrings since #1972 -- stop agreeing where they must.
    """
    got = ghost_cell_fp_no_flux(1.7, 0.0, _D, 0.1, 1.0, grid_type=grid_type)
    assert got == pytest.approx(ghost_cell_neumann(1.7, 0.0, 0.1), abs=1e-15)


def test_the_vertex_ghost_converges_to_the_exact_no_flux_profile():
    """The analytic oracle, independent of any discretisation in this module.

    `v*rho - D*drho/dx = 0` integrates to `rho(x) = rho(0)*exp(v*x/D)`, so the exact ghost one step
    outside a max wall is `rho_i*exp(v_n*dx/D)`. Each form here is a rational approximation to
    `exp(z)` at `z = v_n*dx/D`:

        retired vertex   (1+z)/(1-z) = 1 + 2z + ...   O(dx)     <- leading term is 2z, not z
        this branch      1 + z                        O(dx^2)
        cell-centred     (1+z/2)/(1-z/2)              O(dx^3)   <- the Pade(1,1) of exp

    So the retired form was INCONSISTENT, not merely coarse, and this branch is consistent while
    remaining one order below the cell-centred one. The bound below is 1.8, which the corrected
    branch clears at a measured 2.01 and the retired one fails at 1.06.
    """
    v_n, d, rho_i = 0.5, 0.125, 1.0
    errors = []
    for dx in (0.05, 0.025, 0.0125, 0.00625):
        exact = rho_i * np.exp(v_n * dx / d)
        got = ghost_cell_fp_no_flux(rho_i, v_n, d, dx, 1.0, grid_type=GridType.VERTEX_CENTERED)
        errors.append(abs(got - exact))

    if max(errors) < 1e-14:
        return  # an exact ghost has no rate to measure, and must not fail a convergence test
    rates = [np.log2(a / b) for a, b in itertools.pairwise(errors)]
    assert min(rates) > 1.8, f"vertex ghost converges at {rates}, expected ~2"

    # The control: the cell-centred branch must stay a full order above, or the two have been
    # collapsed into one formula rather than each given its own geometry.
    cell_errors = [
        abs(
            ghost_cell_fp_no_flux(rho_i, v_n, d, dx, 1.0, grid_type=GridType.CELL_CENTERED)
            - rho_i * np.exp(v_n * dx / d)
        )
        for dx in (0.05, 0.025, 0.0125, 0.00625)
    ]
    if max(cell_errors) < 1e-14:
        return
    cell_rates = [np.log2(a / b) for a, b in itertools.pairwise(cell_errors)]
    assert min(cell_rates) > 2.8, f"cell-centred converges at {cell_rates}, expected ~3"

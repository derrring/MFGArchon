"""#2068: `ghost_cell_fp_no_flux`'s vertex-centred branch used the cell-centred wall geometry.

It was consistent with `rho_face = (rho_g + rho_i)/2` AND `drho/dn = (rho_g - rho_i)/(2*dx)` -- a
wall midway between ghost and interior at separation `2*dx`. On a vertex grid the wall IS the
interior node, which is why `ghost_cell_dirichlet` returns `g` there unmodified and why
`ghost_cell_neumann` uses separation `dx` on both centrings (#1972). So the branch satisfied its own
stencil to machine zero while leaving a total flux that did not converge.

The tests below are deliberately NOT the defining equation rearranged -- asserting
`v_n*rho_i == D*(rho_g - rho_i)/dx` would restate the fix. Two independent properties instead:

- the two centrings discretise the SAME continuous condition, so their leading correction in `dx`
  must agree. The old form's was exactly twice the cell-centred one, which is the "diffusive flux
  is exactly twice what the condition requires" the issue measured.
- the flux residual must converge under refinement. The old form's froze at `-v_x*rho`.

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


def test_the_vertex_flux_residual_converges_under_refinement():
    """The property the defect violated. The residual is evaluated in the VERTEX geometry.

    The wall is the node, so the wall density is `rho_interior` and the separation is `dx`. Under
    the retired form this residual froze at `-v_x*rho = -0.5` from `dx = 0.1` to `dx = 0.00625`
    while the cell-centred control below was machine zero at every one -- which is what put the
    defect in the branch rather than in the harness.
    """
    v_x = 0.5
    for dx in (0.1, 0.05, 0.025, 0.0125, 0.00625):
        g = ghost_cell_fp_no_flux(_RHO, v_x, _D, dx, 1.0, grid_type=GridType.VERTEX_CENTERED)
        residual = v_x * _RHO - _D * (g - _RHO) / dx
        assert residual == pytest.approx(0.0, abs=1e-12), f"dx={dx}: J.n = {residual:.6f}"

        # the control that puts a failure in the vertex branch and not in this loop
        gc = ghost_cell_fp_no_flux(_RHO, v_x, _D, dx, 1.0, grid_type=GridType.CELL_CENTERED)
        control = v_x * (gc + _RHO) / 2 - _D * (gc - _RHO) / dx
        assert control == pytest.approx(0.0, abs=1e-12), f"harness is wrong, not the branch: {control}"


def test_the_old_vertex_form_had_a_pole_at_half_the_cell_centred_limit():
    """Not a restatement: it is why the correction is not merely more accurate.

    `(D + v_n*dx)/(D - v_n*dx)` is singular at `dx = D/v_n` -- half the cell-centred `2D/v_n`,
    because the wrong geometry doubles the effective step -- and returns a NEGATIVE density past
    it. A negative density is not a small error in a Fokker-Planck solve. The corrected form is
    linear in `dx`, so it has no pole; it still turns negative under strong INWARD drift beyond
    `dx > D/|v_n|`, which is a resolution requirement and is asserted here so the difference is
    not mistaken for unconditional positivity.
    """
    v_x = 0.5
    for dx in (0.2, 0.249, 0.251, 0.3, 1.0):
        g = ghost_cell_fp_no_flux(_RHO, v_x, _D, dx, 1.0, grid_type=GridType.VERTEX_CENTERED)
        assert np.isfinite(g), f"outward drift, dx={dx}: ghost={g} -- the old form poles at 0.25"
        assert g > 0.0, f"outward drift, dx={dx}: ghost={g} -- a density must stay positive"

    inward = ghost_cell_fp_no_flux(_RHO, -v_x, _D, 1.0, 1.0, grid_type=GridType.VERTEX_CENTERED)
    assert inward < 0.0, "the linear form still has a positivity limit; it is a resolution bound"


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
    cell_rates = [np.log2(a / b) for a, b in itertools.pairwise(cell_errors)]
    assert min(cell_rates) > 2.8, f"cell-centred converges at {cell_rates}, expected ~3"

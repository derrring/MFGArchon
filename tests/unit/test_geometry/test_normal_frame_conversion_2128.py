#!/usr/bin/env python3
"""Issue #2128: the FP no-flux ghost is Robin's algebra plus a frame conversion, not a second copy.

WHAT THIS PINS
--------------
`ghost_cell_robin` takes coefficients ALREADY in the normal frame, which is why it correctly has no
sign argument -- `du/dn` carries the wall's direction, and #2063 removed a sign from it after
measuring it unused at cell-centring and applied BACKWARDS at vertex-centring.
`ghost_cell_fp_no_flux` takes axis-frame `v` and `D`, so `J . n = 0` needs the outward normal to
project them. That projection is `normal_frame_coefficients`, and before #2128 it existed only as a
second copy of Robin's closed form.

THE ORACLE IS EXTERNAL, NOT AN AGREEMENT
----------------------------------------
Two implementations agreeing proves nothing when one was derived from the other -- which is exactly
what this change makes true of these two. So the load-bearing test here is NOT `fp_no_flux ==
robin`; it is both against the **exact zero-flux profile**, which neither implements:

    J . n = 0  =>  v_n*rho - D*drho/dn = 0  =>  rho(s) = rho_wall * exp(v_n*s/D)

so the exact ghost one cell out is `rho_interior * exp(z)`, `z = v_n*dx/D`, and each branch is a
rational approximation to `exp(z)`: cell-centred is its Pade(1,1) and converges at order 3, the
vertex form is the truncated series `1 + z` and converges at order 2 (#2068 measured 3.04 and 2.01).
A delegation that got the conversion wrong would not merely disagree with the old code -- it would
lose the order against the analytic profile, which is checkable without either implementation.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary.ghost_cells import (
    GridType,
    ghost_cell_fp_no_flux,
    ghost_cell_robin,
    normal_frame_coefficients,
)

_CENTRINGS = [GridType.CELL_CENTERED, GridType.VERTEX_CENTERED]
#: (v, D, interior, dx). Includes a negative drift and an interior value below zero, so a sign
#: error in the conversion cannot cancel against a symmetric fixture.
_CASES = [(0.4, 0.2, 1.0, 0.1), (-0.7, 0.35, 2.5, 0.05), (1.3, 0.9, -0.4, 0.2)]
#: Stated orders against exp(z) (#2068): 3.04 cell-centred, 2.01 vertex. Asserted with margin.
_MIN_ORDER = {GridType.CELL_CENTERED: 2.6, GridType.VERTEX_CENTERED: 1.7}


def test_the_conversion_returns_the_normal_frame_pair():
    """`J . n = v_n*rho - D*drho/dn = 0` is Robin with `alpha = v.n` and `beta = -D`.

    Pinned as the NORMAL-FRAME pair, not merely as a pair that happens to give the right ghost. An
    earlier draft returned `(v, -sign*D)` -- `sign` times this one. It yields an identical ghost,
    because scaling both coefficients leaves a homogeneous Robin condition unchanged, and it passed
    every value test in this file. It is still wrong for a function with this name: it diverges the
    moment a caller has a non-zero right-hand side (1.045455 against 0.590909 at `g = 0.5`), and it
    puts the sign on the diffusion coefficient rather than on the velocity being projected.
    """
    assert normal_frame_coefficients(0.4, 0.2, -1.0) == (-0.4, -0.2)
    assert normal_frame_coefficients(0.4, 0.2, +1.0) == (0.4, -0.2)
    assert normal_frame_coefficients(-0.7, 0.35, +1.0) == (-0.7, -0.35)
    assert normal_frame_coefficients(-0.7, 0.35, -1.0) == (0.7, -0.35)


@pytest.mark.parametrize("grid_type", _CENTRINGS, ids=lambda g: g.name)
@pytest.mark.parametrize("sign", [-1.0, +1.0], ids=["min_wall", "max_wall"])
@pytest.mark.parametrize(("v", "D", "u", "dx"), _CASES)
def test_the_ghost_is_robin_evaluated_at_the_converted_pair(grid_type, sign, v, D, u, dx):
    """The delegation, stated as an identity a re-implementation would break."""
    alpha, beta = normal_frame_coefficients(v, D, sign)
    # The scale is `ghost_cell_fp_no_flux`'s own and is not decoration: `ghost_cell_robin`'s
    # singularity threshold is absolute while the quantity it tests has units (#2217), so the pair
    # is scaled by `2*dx` to make that threshold mean what this function's predicate meant before
    # #2128. Only its magnitude matters -- robin tests an absolute value -- and the scale leaves
    # the condition, and therefore the ghost, unchanged.
    scale = 2.0 * dx
    assert ghost_cell_fp_no_flux(u, v, D, dx, sign, grid_type) == ghost_cell_robin(
        u, 0.0, scale * alpha, scale * beta, dx, grid_type
    )


@pytest.mark.parametrize("grid_type", _CENTRINGS, ids=lambda g: g.name)
@pytest.mark.parametrize("sign", [-1.0, +1.0], ids=["min_wall", "max_wall"])
def test_the_ghost_converges_to_the_exact_zero_flux_profile(grid_type, sign):
    """THE external oracle: `rho_exact = rho_interior * exp(v_n*dx/D)`, which neither branch implements.

    Refining `dx` alone at fixed `v` and `D` drives `z -> 0`, so the error must fall at the branch's
    order. This is what a wrong conversion loses: flip the sign of `beta` and `z` flips with it, so
    the approximation is to `exp(-z)` and the observed order collapses.
    """
    v, D, u = 0.6, 0.25, 1.0
    v_n = v * sign
    errors, spacings = [], [0.02, 0.01, 0.005]
    for dx in spacings:
        exact = u * np.exp(v_n * dx / D)
        got = ghost_cell_fp_no_flux(u, v, D, dx, sign, grid_type)
        errors.append(abs(got - exact))

    orders = [np.log(errors[i] / errors[i + 1]) / np.log(spacings[i] / spacings[i + 1]) for i in range(len(errors) - 1)]
    observed = float(np.min(orders))
    assert observed >= _MIN_ORDER[grid_type], (
        f"{grid_type.name} at the {'max' if sign > 0 else 'min'} wall converged at order "
        f"{observed:.2f} against the analytic profile rho*exp(v_n*dx/D), below the "
        f"{_MIN_ORDER[grid_type]} this branch is required to hold (#2068 measured "
        f"{'3.04' if grid_type is GridType.CELL_CENTERED else '2.01'}). Errors {errors} at dx "
        f"{spacings}. A conversion with the wrong sign on beta approximates exp(-z) and lands here."
    )


def test_the_cell_centred_degeneracy_refuses_instead_of_reflecting():
    """BEHAVIOUR CHANGE, the only one in #2128, pinned deliberately.

    At `alpha/2 + beta/dx = 0` -- equivalently `2D = v_n*dx` -- the ghost's own coefficient vanishes,
    so the condition does not determine it. The pre-#2128 code returned `interior_value` there, a
    silent fallback it called "the pure advection limit"; `ghost_cell_robin` raises, and the
    delegation makes the refusal win. Measured on `u=1, v=4, D=0.2, dx=0.1`: `1.0` before, ValueError
    after, and that is the whole behavioural diff of this change.

    The second arm is the over-fire control -- halving the drift leaves the degeneracy and must solve.
    """
    D, dx, u, sign = 0.2, 0.1, 1.0, +1.0
    v_degenerate = 2.0 * D / dx * sign
    with pytest.raises(ValueError, match="singular ghost cell formula"):
        ghost_cell_fp_no_flux(u, v_degenerate, D, dx, sign, GridType.CELL_CENTERED)

    # Over-fire control, asserted against the closed form rather than `isfinite`, which passes on
    # any finite wrong answer -- including the `interior_value` this change stopped returning.
    v_near = 0.5 * v_degenerate
    v_n = v_near * sign
    expected = u * (2.0 * D + v_n * dx) / (2.0 * D - v_n * dx)
    assert ghost_cell_fp_no_flux(u, v_near, D, dx, sign, GridType.CELL_CENTERED) == pytest.approx(expected)

    # And the case the scaling exists for: `v_n = 0` is homogeneous Neumann, so the ghost is exactly
    # `interior_value`. Delegating an UNSCALED pair refused this on a coarse grid (#2217).
    assert ghost_cell_fp_no_flux(u, 0.0, 5e-12, 10.0, sign, GridType.CELL_CENTERED) == pytest.approx(u)


@pytest.mark.parametrize("diffusion", [0.0, 1e-13], ids=["D_zero", "D_below_threshold"])
def test_the_vertex_zero_diffusion_value_is_preserved_not_inherited(diffusion):
    """The OTHER degeneracy, and this one is pinned so the delegation does NOT change it (#2215).

    Vertex-centred with `D ~ 0` is a different condition from the cell-centred one above: there the
    ghost's coefficient vanishes, here `beta` does. `J . n = v_n*rho_wall = 0` with `v_n != 0` forces
    `rho_wall = 0`, and on a vertex grid the wall IS the interior node -- so the condition constrains
    the interior value and says nothing about the ghost. The pre-#2128 code returned
    `interior_value`; `ghost_cell_robin`'s `beta == 0` branch returns the implied Dirichlet `0.0`.
    Both put a number where nothing is determined.

    #2128 is a consolidation and does not get to pick between them by side effect, so the old value
    is preserved behind an explicit guard and the choice is #2215's. This test exists to make the
    preservation deliberate: if the guard is removed, this fails rather than the value quietly
    becoming 0.0.
    """
    u, v, dx, sign = 1.0, 0.4, 0.1, +1.0
    assert ghost_cell_fp_no_flux(u, v, diffusion, dx, sign, GridType.VERTEX_CENTERED) == u

    #: The threshold is the pre-#2128 one. Just above it the branch is live and enormous, which is
    #: the discontinuity #2215 records -- pinned here so "preserved" means the value AND its edge.
    live = ghost_cell_fp_no_flux(u, v, 1e-9, dx, sign, GridType.VERTEX_CENTERED)
    assert live > 1e6, "above the guard's threshold the vertex branch must be live, not clamped"


def test_field_valued_inputs_behave_the_same_on_both_centrings():
    """#2128's own acceptance bullet: "works or raises identically on both centrings".

    Two different surfaces, and the delegation moved them in opposite directions:

    - **Multi-element `D`** — `main` raised `ValueError` for size > 1 and, because `bool()` on a
      one-element array is legal, silently ACCEPTED shape-(1,). The delegation made the CELL path
      answer for any array while the vertex guard, a scalar `abs()`, kept raising: an asymmetry
      introduced by side effect. Refused on both for size > 1, size-1 still accepted — `main`'s
      acceptance set exactly. A draft guarded on `ndim != 0`, which also refused size-1: an
      undeclared capability removal inside a consolidation, and the reason this arm pins the size-1
      case rather than only the refusal.
    - **Field-valued drift** — `main` raised on the cell path (`truth value of an array is
      ambiguous`) and the delegation makes it work on both. That is a genuine gain, unclaimed until
      the review found it, and pinned here so it is not lost silently.
    """
    u = np.ones(3)

    for grid_type in _CENTRINGS:
        with pytest.raises(NotImplementedError, match="multi-element diffusion_coeff"):
            ghost_cell_fp_no_flux(u, 0.4, np.array([0.2, 1e-13, 0.5]), 0.1, +1.0, grid_type)

        #: size-1 is `main`'s acceptance set and must survive the guard, which a draft's
        #: `ndim != 0` predicate silently removed.
        got = ghost_cell_fp_no_flux(1.0, 0.4, np.array([0.2]), 0.1, +1.0, grid_type)
        assert np.shape(got) == (1,)
        assert np.isfinite(got).all()

    #: Asserted against the closed form PER ELEMENT, not against shape and finiteness. An earlier
    #: draft of this arm checked `isfinite` plus "three distinct values", and an array-only defect --
    #: each boundary point receiving another point's drift -- passed it AND the entire 1348-test
    #: geometry suite, because no scalar test can express a permutation. That is the same defect
    #: round 1 raised against this file's over-fire control, re-created in the arm added to pin an
    #: array capability. Only a per-element oracle sees it.
    drift = np.array([0.4, -0.2, 1.1])
    D, dx, sign = 0.2, 0.1, +1.0
    v_n = drift * sign
    expected = {
        GridType.CELL_CENTERED: u * (2.0 * D + v_n * dx) / (2.0 * D - v_n * dx),
        GridType.VERTEX_CENTERED: u * (D + v_n * dx) / D,
    }
    for grid_type in _CENTRINGS:
        got = ghost_cell_fp_no_flux(u, drift, D, dx, sign, grid_type)
        assert np.shape(got) == (3,), f"{grid_type.name}: a field-valued drift must stay elementwise"
        np.testing.assert_allclose(got, expected[grid_type], rtol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""The ghost buffer must use the grid's spacing, not fall back to dx = 1.0.

`PreallocatedGhostBuffer` derived its spacing only from `domain_bounds`, and almost no caller
passes those: `pad_array_with_ghosts(..., geometry=None)` left `_grid_spacing = None` and every
consumer read `dx = 1.0`. An inhomogeneous Neumann condition was therefore applied as `g/h`
instead of `g`. Reading the flux back off the ghost, `-(padded[1] - padded[0])/dx`, returned
exactly `g/h` for a requested `g = 2` -- and does so for any state, because that expression is the
algebraic inverse of the ghost write:

    Nx = 21 / 41 / 81 / 161   ->   40 / 80 / 160 / 320      before
                              ->   2.0 at every Nx          after

So what the assertions below certify is that the ghost ENCODES what the caller asked for, not that
any derivative the solver goes on to form is exact. It is not: the centred wall gradient the HJB
path builds converges to `g/2 - u'(wall)/2` at O(h) -- for `u = x^2 - 2x` under `neumann(2.0)`,
1.9750 / 1.9875 / 1.9938 at Nx = 21 / 41 / 81 against the true 2.0. That gap is the node-centring
half of #1904 and is not what this file measures.

Robin read the same fallback, in the denominator `alpha + beta/dx`.

Found by review of #1902 (which is blocked on #1904, the node-centring half of the same function).
`HJBFDMSolver.honors_inhomogeneous_neumann` is True and pinned, so the declaration was live and
wrong -- the existing applicator test builds its buffer WITH `domain_bounds`, validating the
instrument on a path the solver never takes, which is why this passed.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers.base_hjb import _compute_gradient_array_1d, _compute_laplacian_1d
from mfgarchon.geometry.boundary import neumann_bc, no_flux_bc, robin_bc
from mfgarchon.geometry.boundary.applicator_fdm import pad_array_with_ghosts


@pytest.mark.parametrize("nx", [21, 41, 81, 161])
def test_inhomogeneous_neumann_recovers_the_requested_flux_at_every_resolution(nx: int):
    """The law: what the caller asked for is what the ghost encodes, independent of h."""
    g = 2.0
    dx = 1.0 / (nx - 1)
    u = 0.5 * np.cos(2 * np.pi * np.linspace(0.0, 1.0, nx))
    padded = pad_array_with_ghosts(u, neumann_bc(dimension=1, value=g), ghost_depth=1, time=0.0, spacing=dx)

    # Outward normal is -x at the low wall, +x at the high wall.
    low = -(padded[1] - padded[0]) / dx
    high = (padded[-1] - padded[-2]) / dx
    assert low == pytest.approx(g, abs=1e-12), f"nx={nx}: du/dn = {low} at the low wall, requested {g}"
    assert high == pytest.approx(g, abs=1e-12), f"nx={nx}: du/dn = {high} at the high wall, requested {g}"


def test_the_fallback_would_diverge_which_is_why_the_spacing_is_threaded():
    """Control: without spacing the ghost is spacing-independent, so the flux scales as 1/h.

    Asserts the DEFECT is still reachable through the un-threaded call, so the test above is
    measuring the threading rather than something that was always true.
    """
    u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bc = neumann_bc(dimension=1, value=2.0)
    without = pad_array_with_ghosts(u, bc, ghost_depth=1, time=0.0)
    with_small = pad_array_with_ghosts(u, bc, ghost_depth=1, time=0.0, spacing=0.05)
    assert without[0] == pytest.approx(3.0), "the un-threaded fallback is no longer dx = 1.0"
    assert with_small[0] == pytest.approx(1.1), "the threaded call does not use the spacing given"
    assert without[0] != with_small[0], "spacing makes no difference; the threading is inert"


@pytest.mark.parametrize("scalar_or_sequence", [0.05, [0.05], (0.05,)])
def test_spacing_accepts_a_scalar_or_one_value_per_axis(scalar_or_sequence):
    u = np.array([1.0, 2.0, 3.0])
    bc = neumann_bc(dimension=1, value=1.0)
    padded = pad_array_with_ghosts(u, bc, ghost_depth=1, time=0.0, spacing=scalar_or_sequence)
    assert padded[0] == pytest.approx(1.05)


def test_a_wrong_length_spacing_raises_rather_than_broadcasting_silently():
    """Fail loud: a 2-entry spacing for a 1-D array is a caller bug, not something to guess at."""
    u = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="one per axis"):
        pad_array_with_ghosts(u, neumann_bc(dimension=1, value=1.0), ghost_depth=1, time=0.0, spacing=[0.1, 0.2])


def test_zero_flux_is_unaffected_because_it_multiplies_the_spacing_by_zero():
    """Scope control: no-flux must be byte-identical, or this change is wider than claimed.

    The node-centring defect in the same function (#1904) is NOT addressed here and this test
    would not see it -- it compares the two spellings against each other, not against the truth.
    """
    u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    for bc in (no_flux_bc(dimension=1), neumann_bc(dimension=1, value=0.0)):
        a = pad_array_with_ghosts(u, bc, ghost_depth=1, time=0.0)
        b = pad_array_with_ghosts(u, bc, ghost_depth=1, time=0.0, spacing=0.05)
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("nx", [21, 81])
@pytest.mark.parametrize("g", [2.0, -3.0])
def test_the_solver_gradient_path_now_carries_the_spacing(nx: int, g: float):
    """End-to-end through `_compute_gradient_array_1d`, which is what the residual calls.

    The law, not a band: the ghost differs from the `g = 0` ghost by exactly `dx*g`, and the
    centred difference divides by `2*dx`, so requesting `g` shifts the wall rows by exactly
    `-g/2` and `+g/2` -- for ANY interior field and ANY spacing. A random field is used to
    make that independence part of the assertion rather than a remark.

    ~~`1e-9 < abs(tight[0] - loose[0]) < 5.0`~~ was the original form and pinned nothing
    [CORRECTED 2026-08-13, found by independent review of #1906]: that band admits the correct
    1.0, the halved 0.5, and the measured-but-wrong 1.2447 alike.
    """
    dx = 1.0 / (nx - 1)
    u = np.random.default_rng(7).normal(size=nx)
    tight = _compute_gradient_array_1d(u, dx, bc=neumann_bc(dimension=1, value=g), upwind=False, time=0.0)
    loose = _compute_gradient_array_1d(u, dx, bc=neumann_bc(dimension=1, value=0.0), upwind=False, time=0.0)
    assert tight[0] - loose[0] == pytest.approx(-g / 2, abs=1e-12), f"nx={nx}, g={g}: low wall"
    assert tight[-1] - loose[-1] == pytest.approx(g / 2, abs=1e-12), f"nx={nx}, g={g}: high wall"


@pytest.mark.parametrize("dx", [0.5, 0.05, 0.01])
def test_robin_reads_the_same_spacing(dx: float):
    """Robin used the fallback too, in the denominator `alpha + beta/dx`.

    The law: the ghost is the exact algebraic solution of the condition the applicator writes,
    `alpha*(u_g + u_i)/2 + beta*(u_g - u_i)/dx*outward_sign = g`, FOR THE SPACING GIVEN. Assert
    that residual at both walls. It is machine-zero when the spacing is threaded and O(1/dx)
    when the buffer substitutes 1.0 -- 32 and 105 at dx = 0.05 and 0.01 on this fixture.

    ~~`coarse[0] != fine[0]`~~ was the original form and pinned nothing [CORRECTED 2026-08-13,
    found by independent review of #1906]: two spacings give different ghosts under any
    denominator whatever, and it probed only index 0.

    NOTE the sign convention this asserts is the applicator's own, `outward_sign = -1` at the
    low wall, which #1907 reports is one factor too many there. This test pins the ghost against
    the equation the code currently writes; it is deliberately NOT the oracle for that defect,
    and it will need re-pointing when #1907 lands.
    """
    u = np.random.default_rng(7).normal(size=7)
    alpha, beta, g = 1.0, 1.0, 0.5
    bc = robin_bc(dimension=1, alpha=alpha, beta=beta, value=g)
    p = pad_array_with_ghosts(u, bc, ghost_depth=1, time=0.0, spacing=dx)
    low = alpha * (p[0] + p[1]) / 2 + beta * (p[0] - p[1]) / dx * (-1.0) - g
    high = alpha * (p[-1] + p[-2]) / 2 + beta * (p[-1] - p[-2]) / dx * (+1.0) - g
    assert low == pytest.approx(0.0, abs=1e-12), f"dx={dx}: Robin low wall residual {low:.3e}"
    assert high == pytest.approx(0.0, abs=1e-12), f"dx={dx}: Robin high wall residual {high:.3e}"


@pytest.mark.parametrize("h", [0.5, 0.05, 0.01])
@pytest.mark.parametrize("g", [3.0, -1.5])
def test_the_laplacian_path_threads_its_own_spacings_parameter(h: float, g: float):
    """`laplacian_with_bc` already took `spacings` and dropped it at the padding call.

    The law: the wall row is `(u[1] - 2*u[0] + ghost)/h^2` with `ghost = u[0] + h*g`, so
    `lap[0]*h^2 - (u[1] - u[0])` is exactly `h*g` -- for any field and any h. Under the
    `dx = 1.0` fallback the ghost carries `1.0*g` instead and the identity misses by `(1-h)*g`.

    ~~`not np.allclose(coarse[0]*0.5**2, fine[0]*0.01**2)`~~ was the original form and pinned
    nothing [CORRECTED 2026-08-13, found by independent review of #1906]: it asserts only that
    the wall row is not spacing-independent, which almost any wrong formula also satisfies.
    """
    u = np.random.default_rng(7).normal(size=6)
    lap = _compute_laplacian_1d(u, h, bc=neumann_bc(dimension=1, value=g), time=0.0)
    assert lap[0] * h * h - (u[1] - u[0]) == pytest.approx(h * g, abs=1e-12), f"h={h}, g={g}: wall row"

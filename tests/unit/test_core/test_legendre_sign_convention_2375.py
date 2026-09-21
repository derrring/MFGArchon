"""The Legendre pairing carries `-p·α`, so `α* = -∂H/∂p` holds for every L. #2375 ruling 5.

WHY NOTHING IN THE LIBRARY COULD HAVE CAUGHT THIS. Every concrete control cost shipped here is
**even in α** -- measured **bit-exact**, `L(a) - L(-a) == 0.0` for `QuadraticControlCost`,
`L1ControlCost`, `BoundedControlCost` and the Moreau-Yosida regularisation, probed across the
`ℓ1` kink at 0, the interior, and deep saturation at `|a| = 120` against `max_control = 1`.
(`BoundedControlCost`'s domain is `(-max_control, max_control)` by construction, so "on a
symmetric domain" is unconditional here, not a caveat.) And
`sup{-p·α - L}` equals `sup{+p·α - L}` exactly when `L(-α) = L(α)`. So the change is byte-identical
across the whole library and a test built on any shipped cost passes before and after while pinning
nothing. The discriminating `L` has to be constructed here, and this file constructs it.

THE SEARCH CONFIGURATION IS PART OF THE ASSERTION, not scaffolding. `DualHamiltonian` locates `α*`
by grid search over `np.linspace(*alpha_bounds, n_search)`, and `n_search` **defaults to 100**. At
the defaults -- `alpha_bounds=(-10, 10)`, step 0.202 -- the even-`L` probe returns -1.9192 instead of
-2.0000, and the non-even rows shift by 2.3e-2, 1.7e-2 and 9.1e-3 (the `p=0.6` row in the THIRD
decimal, not the second). Every test below therefore states `ALPHA_BOUNDS` and `N_SEARCH`
explicitly.

One limit on that, so the docstring does not over-promise: those constants govern the
`DualHamiltonian` assertions only. `LagrangianBase.optimal_control` resolves its own search box
from `control_bounds()`, which these fixtures leave as `None`, so it falls back to
`_FALLBACK_CONTROL_BOUNDS = (-10.0, 10.0)` and is not configured by `ALPHA_BOUNDS`/`N_SEARCH` at
all. Its agreement with the Dual path is therefore evidence about the convention, not about the
grid.

    A run whose even-`L` control does not return exactly -2.0000 has measured its own grid,
    not the library.

That is what `test_the_even_lagrangian_pins_the_grid_before_anything_else_is_believed` is for, and it
is not a formality: it is the only thing separating "the convention is wrong" from "the step size is
0.2". Both produce a number near -2 and neither announces which.
"""

from __future__ import annotations

import contextlib
import sys

import pytest

import numpy as np
import scipy.optimize

from mfgarchon.core.hamiltonian import DualHamiltonian, DualLagrangian, HamiltonianBase, LagrangianBase

# THE NON-EVEN L IS IMPORTED, NOT REDEFINED. `AsymmetricL` in the sibling
# `test_lagrangian_base_alpha_sign_1642.py` is the same Lagrangian with the same 0.3, and it has been
# the suite's only non-even one since #1642. An earlier draft of this file duplicated it, justified by
# "no test module here imports from another -- 0 of 250 files". THAT COUNT WAS WRONG: the pattern is
# anchored at column 0 and every real cross-module import in this repository is INDENTED inside a
# function, so the query could not see them. `test_coupled_mfg_mms.py:119` imports a fixture class
# from a sibling test module in exactly this way, and `test_warning_census_writer.py:40` imports from
# `tests.conftest`. The duplicate's real cost was not the nine lines: nothing pinned the two copies
# to agree, so moving `AsymmetricL.OFFSET` to 0.4 would have left both files green while they tested
# different Lagrangians. Same mechanism and fallback as the precedent.
try:
    from test_lagrangian_base_alpha_sign_1642 import AsymmetricL
except ModuleNotFoundError:  # pragma: no cover - runner-dependent
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_lagrangian_base_alpha_sign_1642 import AsymmetricL


# Stated, not defaulted -- see the module docstring. step = 10/(100001-1) = 1e-4.
ALPHA_BOUNDS = (-5.0, 5.0)
N_SEARCH = 100001
ODD_COEFF = AsymmetricL.OFFSET  # one owner; see the import note below


class _EvenLagrangian(LagrangianBase):
    """`L(a) = 0.5a²`. The control: every convention agrees on it, so a disagreement is the grid."""

    def __call__(self, x, alpha, m, t=0.0):
        a = np.atleast_1d(np.asarray(alpha, dtype=float))
        return float(0.5 * np.sum(a**2))


X = np.zeros(1)
M = 1.0


def _dual(lagrangian):
    return DualHamiltonian(lagrangian, alpha_bounds=ALPHA_BOUNDS, n_search=N_SEARCH)


def test_the_even_lagrangian_pins_the_grid_before_anything_else_is_believed():
    """MANDATORY CONTROL, and it must be read before the rows below.

    The two conventions agree exactly on an even `L`, so this asserts the grid resolves the optimum
    and nothing about the convention. If it fails, every other number in this file is a measurement
    of `np.linspace` -- at the constructor defaults the same probe returns -1.9192.

    Analytic: `argmin{p·a + 0.5a²} = -p`, so -2.0 at `p = +2`.
    """
    even = _EvenLagrangian()
    p = np.array([2.0])
    assert float(np.ravel(_dual(even).optimal_control(X, M, p))[0]) == pytest.approx(-2.0, abs=5e-5)
    assert float(np.ravel(even.optimal_control(X, M, p))[0]) == pytest.approx(-2.0, abs=5e-5)


@pytest.mark.parametrize(
    ("p", "expected_alpha_star"),
    [(2.0, -2.3), (-2.0, 1.7), (0.6, -0.9)],
)
def test_a_non_even_lagrangian_gives_the_control_that_actually_minimises_cost(p, expected_alpha_star):
    """THE ARTIFACT. `α*` must minimise `p·α + L(α)` -- the cost-to-go the agent pays.

    Analytic, so no grid supplies the expected value: `d/da (p·a + 0.5a² + 0.3a) = 0` gives
    `α* = -(p + 0.3)`. Before this change the library returned `-(p - 0.3)` -- the right magnitude
    off by twice the odd coefficient, and correct for every even `L`, which is why it survived.

    Both classes are asserted because they are two owners of one convention: `DualHamiltonian`
    through `_find_optimal_alpha`, `LagrangianBase` through `conjugate_argmax`. Pinning one leaves
    the other free to fork, which is the failure #1642 records.
    """
    assert expected_alpha_star == pytest.approx(-(p + ODD_COEFF), abs=1e-12), "expectation is analytic"
    non_even = AsymmetricL()
    pv = np.array([p])
    assert float(np.ravel(_dual(non_even).optimal_control(X, M, pv))[0]) == pytest.approx(expected_alpha_star, abs=5e-4)
    assert float(np.ravel(non_even.optimal_control(X, M, pv))[0]) == pytest.approx(expected_alpha_star, abs=5e-4)


def test_the_hamiltonian_value_is_the_conjugate_at_minus_p():
    """`H = sup{-p·α - L}` is `L*(-p)`, not `L*(p)` -- the value moved, not just the argmax.

    Closed form for this `L`: `H(p) = 0.5(p + 0.3)²`. Asserting the VALUE and not only the optimiser,
    because `__call__` and `_find_optimal_alpha` carry the objective separately -- three branches
    each -- and flipping one without the other leaves an `H` whose gradient is not its own argmax.
    """
    h = _dual(AsymmetricL())
    for p in (2.0, -2.0, 0.6):
        assert float(h(X, M, np.array([p]))) == pytest.approx(0.5 * (p + ODD_COEFF) ** 2, abs=5e-4)


def test_the_inverse_transforms_gradient_carries_the_same_sign():
    """PINS `DualLagrangian._find_optimal_p`, which the round-trip test does NOT reach.

    Measured: un-negating that function's return leaves every other test in this file green, because
    the round trip reads `DualLagrangian.__call__` (a sup VALUE) and never its argmax. So `d_alpha`
    was an arm of this change that nothing exercised -- the same gap #2375's phase-1 sibling hit on
    four guard arms, found the same way, by mutating the thing just written.

    Analytic: with `H(p) = 0.5(p + 0.3)²`, the inverse `L(α) = sup_p {-p·α - H(p)}` is maximised at
    `p* = -(α + 0.3)`, and `∂L/∂α = -p* = α + 0.3` -- which is `L'(α)` for the original
    `L = 0.5α² + 0.3α`, as a recovered Legendre pair requires. Un-negated it returns `-(α + 0.3)`,
    wrong by a sign for every α, and exactly zero at α = -0.3.
    """
    h = DualHamiltonian(AsymmetricL(), alpha_bounds=(-6.0, 6.0), n_search=601)
    inverse = DualLagrangian(h, p_bounds=(-6.0, 6.0), n_search=601)
    for alpha in (1.0, -1.0, 0.5):
        expected = alpha + ODD_COEFF
        # The two candidate answers are +/-(alpha + 0.3), i.e. 2*|alpha + 0.3| >= 0.4 apart, so a
        # 0.02 grid step resolves WHICH sign without resolving the value tightly. Deliberately
        # coarse: a nested double search at 1e-4 cost 42s for no extra discrimination.
        assert abs(2 * expected) > 10 * 5e-2, "the two signs must be far apart relative to tolerance"
        got = float(np.ravel(inverse.d_alpha(X, np.array([alpha]), M))[0])
        assert got == pytest.approx(expected, abs=5e-2)


def test_the_legendre_round_trip_recovers_L_and_not_its_mirror():
    """BOTH sides of the pair carry `-p·α`, or `L → H → L` returns `L(-α)`.

    This is the reason the inverse transform is in the same change. Flipping `H` alone leaves a
    round trip that returns `L(-α)`: a well-formed Lagrangian, equal to `L` for every even `L`, and
    wrong for exactly the non-even ones the convention was changed for. `L(1) = 0.8` against
    `L(-1) = 0.2`, so the two outcomes are 0.6 apart and no grid error explains the gap.

    Coarser grids than the module constants on purpose: this asserts WHICH of two well-separated
    values is returned, and a 1e-4 step over a nested double search costs minutes for no extra
    discrimination.
    """
    lagrangian = AsymmetricL()
    h = DualHamiltonian(lagrangian, alpha_bounds=(-6.0, 6.0), n_search=1201)
    round_trip = DualLagrangian(h, p_bounds=(-6.0, 6.0), n_search=241)
    for alpha in (1.0, -1.0, 0.5):
        direct = lagrangian(X, np.array([alpha]), M)
        mirror = lagrangian(X, np.array([-alpha]), M)
        assert abs(direct - mirror) > 0.2, "the probe must separate L(a) from L(-a)"
        assert float(round_trip(X, np.array([alpha]), M)) == pytest.approx(direct, abs=2e-2)


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# nD PINS. Everything above exercises the d == 1 branch only, and review measured the consequence:
# of 23 behaviour-bearing lines this change touches, NINE survived the full gate -- all six nD
# `ImportError` fallback lines, plus three live `DualLagrangian` nD scipy lines. My own seven
# mutations were one per FUNCTION and every one of them landed on that function's 1-D line, so the
# nD half of each function went unmeasured while the matrix read 7/7.
#
# `DualHamiltonian`'s nD scipy lines turned out to be pinned INCIDENTALLY, by a 2-D Godunov
# monotonicity probe in test_hjb_fdm_upwind_hypothesis_2311.py -- a consumer, not a convention test.
# `DualLagrangian` has no such accidental consumer, which is why its lines were bare.
# ─────────────────────────────────────────────────────────────────────────────────────────────────

BOX_2D = (-8.0, 8.0)
X2 = np.zeros(2)
P_2D = [(2.0, -1.0), (0.6, 0.6), (-2.0, 3.0)]


class _AnalyticH2D(HamiltonianBase):
    """`H(p) = 0.5 Σ (p_i + 0.3)²` — the exact conjugate of `AsymmetricL`, in closed form.

    Gives the inverse transform a source whose answer is known analytically, so a `DualLagrangian`
    assertion does not have to trust a nested numerical `DualHamiltonian` underneath it.
    """

    def __call__(self, x, m, p, t=0.0):
        return float(0.5 * np.sum((np.atleast_1d(p) + ODD_COEFF) ** 2))

    def dp(self, x, m, p, t=0.0):
        return np.atleast_1d(p) + ODD_COEFF


@contextlib.contextmanager
def _no_scipy_optimize():
    """Force the `except ImportError` arm, which an installed scipy otherwise makes unreachable.

    Six of the nine unpinned lines live in these fallbacks. Setting the entry to `None` makes
    `from scipy.optimize import ...` raise `ImportError`, which is the only way to reach them
    without uninstalling scipy. Restored in `finally`; tests within an xdist worker run
    sequentially and workers are separate processes, so no other test observes the patch.
    """
    saved = sys.modules.get("scipy.optimize")
    sys.modules["scipy.optimize"] = None
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop("scipy.optimize", None)
        else:
            sys.modules["scipy.optimize"] = saved


@pytest.mark.parametrize("p", P_2D)
def test_the_nd_scipy_branch_carries_the_pairing(p):
    """H, ∂H/∂p and the drift together, nD. Analytic: `α*_i = -(p_i + 0.3)`, `H = 0.5 Σ (p_i+0.3)²`.

    All three because they come from two functions that carry the objective separately —
    `__call__` for the value, `_find_optimal_alpha` for the argmax — and a flip in one alone leaves
    an H whose gradient is not its own maximiser.
    """
    pv = np.array(p, dtype=float)
    dh = DualHamiltonian(AsymmetricL(), alpha_bounds=BOX_2D, n_search=4001)
    assert float(dh(X2, M, pv)) == pytest.approx(0.5 * np.sum((pv + ODD_COEFF) ** 2), abs=1e-4)
    np.testing.assert_allclose(np.ravel(dh.dp(X2, M, pv)), pv + ODD_COEFF, atol=1e-4)
    np.testing.assert_allclose(np.ravel(dh.optimal_control(X2, M, pv)), -(pv + ODD_COEFF), atol=1e-4)


@pytest.mark.parametrize("p", P_2D)
def test_the_nd_importerror_fallback_carries_the_pairing(p):
    """The 20-point-per-axis fallback — coarse, so the expectation is that exact grid's answer.

    Asserted at 1e-9 rather than loosely: the objective is separable, so the product-grid argmax is
    the per-axis argmax and the expected value is computable exactly. A tolerance wide enough to
    absorb the coarse grid would also absorb a sign error, which is the only defect in scope here.
    """
    pv = np.array(p, dtype=float)
    dh = DualHamiltonian(AsymmetricL(), alpha_bounds=BOX_2D, n_search=4001)
    grid = np.linspace(BOX_2D[0], BOX_2D[1], 20)
    best = np.array([grid[int(np.argmax(-pi * grid - (0.5 * grid**2 + ODD_COEFF * grid)))] for pi in pv])
    expected_h = float(-np.dot(pv, best) - (0.5 * np.sum(best**2) + ODD_COEFF * np.sum(best)))
    with _no_scipy_optimize():
        assert float(dh(X2, M, pv)) == pytest.approx(expected_h, abs=1e-9)
        np.testing.assert_allclose(np.ravel(dh.dp(X2, M, pv)), -best, atol=1e-9)


@pytest.mark.parametrize("alpha", [(1.0, -1.0), (0.5, 0.5), (-2.0, 1.5)])
def test_the_inverse_nd_scipy_branch_carries_the_pairing(alpha):
    """THE THREE LIVE UNPINNED LINES. `DualLagrangian` had no nD test at all.

    Against the closed-form source `_AnalyticH2D`, the inverse must recover `AsymmetricL`:
    `L(α) = 0.5 Σ α_i² + 0.3 Σ α_i` and `∂L/∂α_i = α_i + 0.3`.
    """
    av = np.array(alpha, dtype=float)
    dl = DualLagrangian(_AnalyticH2D(), p_bounds=BOX_2D, n_search=4001)
    expected = float(0.5 * np.sum(av**2) + ODD_COEFF * np.sum(av))
    assert float(dl(X2, av, M)) == pytest.approx(expected, abs=1e-4)
    np.testing.assert_allclose(np.ravel(dl.d_alpha(X2, av, M)), av + ODD_COEFF, atol=1e-4)


@pytest.mark.parametrize("alpha", [(1.0, -1.0), (0.5, 0.5)])
def test_the_inverse_nd_importerror_fallback_carries_the_pairing(alpha):
    """The inverse's own 20-point fallback, exact against that grid for the same reason."""
    av = np.array(alpha, dtype=float)
    dl = DualLagrangian(_AnalyticH2D(), p_bounds=BOX_2D, n_search=4001)
    grid = np.linspace(BOX_2D[0], BOX_2D[1], 20)
    best = np.array([grid[int(np.argmax(-grid * ai - 0.5 * (grid + ODD_COEFF) ** 2))] for ai in av])
    expected_l = float(-np.dot(best, av) - 0.5 * np.sum((best + ODD_COEFF) ** 2))
    with _no_scipy_optimize():
        assert float(dl(X2, av, M)) == pytest.approx(expected_l, abs=1e-9)
        np.testing.assert_allclose(np.ravel(dl.d_alpha(X2, av, M)), -best, atol=1e-9)


class _NonConvexL(LagrangianBase):
    """`L(a) = 0.5|a|² + 2Σcos(3a_i) + 0.3Σa_i` — non-even AND non-convex, and separable.

    `LagrangianBase` requires neither evenness nor convexity, and `legendre_transform`'s own
    docstring concedes non-convex input by calling the transform involutive only "up to
    convexification". Non-convexity is what gives the nD *local* scipy search more than one basin to
    settle in, which is the only way to observe where it STARTS. Separability keeps the global
    reference a per-axis 1-D scan rather than a product grid.
    """

    def __call__(self, x, alpha, m, t=0.0):
        a = np.atleast_1d(np.asarray(alpha, dtype=float))
        return float(0.5 * np.sum(a**2) + 2.0 * np.sum(np.cos(3.0 * a)) + ODD_COEFF * np.sum(a))

    def control_bounds(self):
        """Declared so ALL paths search the same box the reference grid uses.

        Without it `LagrangianBase.optimal_control` falls back to `_FALLBACK_CONTROL_BOUNDS`
        (-10, 10) while the reference is computed on (-6, 6). The two argmaxes agree to <1e-3 at
        the momenta asserted here, but they diverge above |p| ~ 5.4 — so the test would one day
        fail for a reason that is neither the convention nor the seed. Declaring it also
        early-returns `_reject_fallback_truncation`, whose guard is not what this test is about.
        """
        return (-6.0, 6.0)


@pytest.mark.parametrize("p", [(1.5, -2.5), (-1.0, -1.0), (0.6, -0.4)])
def test_the_nd_initial_guess_points_at_the_new_maximiser(p):
    """PINS `x0 = clip(-p)`, which nothing else does — an initial guess is invisible on a convex L.

    Under the old pairing the maximiser sat near `+p/λ` and `x0 = clip(+p)` started ON it. This
    change moved the maximiser to near `-p/λ` and left `x0` behind, so the nD scipy search began a
    full `2p/λ` away and settled in the wrong basin. Measured at these three momenta, `H`:

        x0 = clip(+p)   -5.6891   1.5990   1.9914
        x0 = clip(-p)    7.3516   4.3758   3.9748
        global sup       7.3497   4.3755   3.9737

    The revert is silent on every shipped cost — all four are convex as well as even, so the local
    search converges from anywhere. That is why the whole per-line mutation sweep killed 23 of 23
    convention lines and left all five `x0` lines standing: they are unobservable without a
    non-convex `L`.

    `p = (2.0, 1.0)` is DELIBERATELY EXCLUDED. There the fix reaches 6.7164 against a global 7.1231
    — better than the broken 5.5487 but still short. That residual is a pre-existing hazard this
    change does not own: the nD scipy branch does LOCAL search while its own 1-D and `ImportError`
    siblings grid GLOBALLY, so the three branches of one function already disagree on non-convex
    input. Asserting the global optimum there would pin a defect this PR did not introduce and
    cannot fix by moving a starting point.
    """
    pv = np.array(p, dtype=float)
    box = (-6.0, 6.0)
    dh = DualHamiltonian(_NonConvexL(), alpha_bounds=box, n_search=100)
    # Global reference by separability: the per-axis argmax IS the product-grid argmax.
    grid = np.linspace(box[0], box[1], 4001)
    per_axis = [np.max(-pi * grid - (0.5 * grid**2 + 2.0 * np.cos(3.0 * grid) + ODD_COEFF * grid)) for pi in pv]
    global_sup = float(np.sum(per_axis))
    got = float(dh(X2, M, pv))
    # Within 1e-3 of the global sup. The broken start misses by 2.78 to 13.04, so the margin is
    # three to four orders of magnitude -- not a threshold tuned to pass.
    assert got == pytest.approx(global_sup, abs=1e-3), (
        f"nD search missed the global sup by {global_sup - got:.4f}; x0 may point at +p"
    )

    # EVERY SEEDED SEARCH, not only the one that computes H. There are five `x0` lines, and a first
    # version of this test pinned exactly one -- `DualHamiltonian.__call__`'s -- because the others
    # are reached through `dp`, through `LagrangianBase` directly, and through the two inverse
    # functions. Measured: reverting any of those four left this file entirely green.
    per_axis_argmax = np.array(
        [grid[int(np.argmax(-pi * grid - (0.5 * grid**2 + 2.0 * np.cos(3.0 * grid) + ODD_COEFF * grid)))] for pi in pv]
    )
    # atol is ONE GRID STEP of the reference (12/4000 = 3e-3), not a tuned number: the reference is
    # a discrete argmax and scipy's is continuous, so they legitimately differ by up to half a step.
    # An earlier 1e-3 was tighter than the reference's own resolution and failed at 1.1e-3. The
    # broken start lands in a different basin entirely, ~2 away, so this still discriminates 700x.
    argmax_atol = 3e-3
    np.testing.assert_allclose(np.ravel(dh.dp(X2, M, pv)), -per_axis_argmax, atol=argmax_atol)
    np.testing.assert_allclose(np.ravel(dh.optimal_control(X2, M, pv)), per_axis_argmax, atol=argmax_atol)
    np.testing.assert_allclose(np.ravel(_NonConvexL().optimal_control(X2, M, pv)), per_axis_argmax, atol=argmax_atol)


class _NonConvexH(HamiltonianBase):
    """`H(p) = 0.5|p|² + 2Σcos(3p_i)` — non-convex and separable, a source for the inverse."""

    def __call__(self, x, m, p, t=0.0):
        q = np.atleast_1d(np.asarray(p, dtype=float))
        return float(0.5 * np.sum(q**2) + 2.0 * np.sum(np.cos(3.0 * q)))

    def dp(self, x, m, p, t=0.0):
        q = np.atleast_1d(np.asarray(p, dtype=float))
        return q - 6.0 * np.sin(3.0 * q)


@pytest.mark.parametrize("alpha", [(1.5, -2.5), (-1.0, -1.0)])
def test_the_inverse_nd_initial_guess_points_at_the_new_maximiser(alpha):
    """The two `x0` sites on the inverse side, against a NON-CONVEX source Hamiltonian.

    Same mechanism and the same blindness as the forward case: on a convex `H` the local search
    converges from anywhere, so `DualLagrangian`'s two `x0` lines are unobservable without this.
    """
    av = np.array(alpha, dtype=float)
    box = (-6.0, 6.0)
    dl = DualLagrangian(_NonConvexH(), p_bounds=box, n_search=100)
    grid = np.linspace(box[0], box[1], 4001)
    obj = [-grid * ai - (0.5 * grid**2 + 2.0 * np.cos(3.0 * grid)) for ai in av]
    global_sup = float(sum(float(np.max(o)) for o in obj))
    per_axis_argmax = np.array([grid[int(np.argmax(o))] for o in obj])
    assert float(dl(X2, av, M)) == pytest.approx(global_sup, abs=1e-3)
    np.testing.assert_allclose(np.ravel(dl.d_alpha(X2, av, M)), -per_axis_argmax, atol=3e-3)


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# MECHANISM PIN for the five `x0` sites. The tests above assert a CONSEQUENCE — "H equals the global
# sup on a non-convex L at these momenta" — and review measured what that costs: over 400 random
# momenta the predicate has a 26% FALSE RED rate (correct code misses the global sup) and a 21%
# FALSE GREEN rate (the broken `clip(+p)` still finds it). The three momenta asserted above sit in
# the ~62% that discriminate, and nothing pins that they will keep doing so — which basin L-BFGS-B
# settles in is a scipy implementation detail, not a library contract. The excluded `p = (2.0, 1.0)`
# is a disclosed member of that 26%, and needing an exclusion list is the tell.
#
# AND THE CONSEQUENCE PIN DOES NOT PIN THE IDENTITY. Measured: `clip(-p) -> clip(+p)`, `-> -p/2` and
# `-> -2p` are all killed above, but `clip(-p) -> np.zeros(d)` SURVIVES the whole file. So that pin
# catches the sign and the magnitude and not the value. The limiting factor stopped being which
# lines get mutated and became WHICH WRONG VALUE they get mutated to: `-> revert` is a narrow
# mutation operator, and an operator is a population predicate like any other.
#
# The contract itself is exact and cheap to observe: `x0 == clip(-p, *box)`, and the library imports
# `minimize` from `scipy.optimize` INSIDE each function at call time, so recording the argument is a
# two-line intercept. No tolerance, no excluded momentum, no basin dependence, 0.04s against 10.8s —
# and it reaches |p| > 6, where the clip itself binds and no consequence test goes.
#
# Both are kept on purpose. The consequence tests are the only thing saying WHY the seed matters and
# that it lands in the right basin rather than merely being negated; this is the load-bearing pin.
# ─────────────────────────────────────────────────────────────────────────────────────────────────


class _Convex2D(LagrangianBase):
    """Any L works: `x0` is chosen before L is ever evaluated. Convex on purpose — the mechanism is
    observable exactly where the consequence is not."""

    def __call__(self, x, alpha, m, t=0.0):
        a = np.atleast_1d(np.asarray(alpha, dtype=float))
        return float(0.5 * np.sum(a**2) + ODD_COEFF * np.sum(a))

    def control_bounds(self):
        return SEED_BOX


SEED_BOX = (-6.0, 6.0)
# (9.0, -7.5) straddles the box so the clip itself binds; (0.0, 0.0) is the fixed point of negation,
# where a sign error is invisible and only an identity assertion can fail.
SEED_CASES = [(2.0, -1.0), (0.6, 0.6), (-2.0, 3.0), (9.0, -7.5), (0.0, 0.0)]


@pytest.fixture
def seeds(monkeypatch):
    """Record every `x0` handed to `scipy.optimize.minimize`, then delegate to the real one."""
    captured: list[np.ndarray] = []
    real = scipy.optimize.minimize

    def spy(fun, x0, *args, **kwargs):
        captured.append(np.array(x0, dtype=float, copy=True))
        return real(fun, x0, *args, **kwargs)

    monkeypatch.setattr(scipy.optimize, "minimize", spy)
    return captured


@pytest.mark.parametrize("p", SEED_CASES)
def test_every_nd_search_is_seeded_at_minus_p(seeds, p):
    """All five seeded searches, exactly, with no tolerance. Kills the `zeros` edit the
    consequence pin lets through."""
    pv = np.array(p, dtype=float)
    expected = np.clip(-pv, *SEED_BOX)

    _Convex2D().conjugate_argmax(X2, M, pv)
    DualHamiltonian(_Convex2D(), alpha_bounds=SEED_BOX, n_search=101)(X2, M, pv)
    DualHamiltonian(_Convex2D(), alpha_bounds=SEED_BOX, n_search=101).dp(X2, M, pv)
    assert len(seeds) == 3, f"expected one seed per search, got {len(seeds)}"
    for got in seeds:
        np.testing.assert_array_equal(got, expected)

    seeds.clear()
    DualLagrangian(_AnalyticH2D(), p_bounds=SEED_BOX, n_search=101)(X2, pv, M)
    DualLagrangian(_AnalyticH2D(), p_bounds=SEED_BOX, n_search=101).d_alpha(X2, pv, M)
    assert len(seeds) == 2
    for got in seeds:
        np.testing.assert_array_equal(got, expected)


def test_the_seed_intercept_can_fail(seeds):
    """CONTROL. An intercept that never fires asserts nothing, and one that cannot fail is scenery.

    Asserts the intercept fired, that it is NOT the pre-fix `+p`, and — at a momentum where the
    clip binds — that the recorded value is the clipped one rather than raw `-p`.
    """
    pv = np.array([9.0, -7.5])
    _Convex2D().conjugate_argmax(X2, M, pv)
    assert seeds, "the intercept never fired -- it is not on the import path the library uses"
    assert not np.allclose(seeds[0], np.clip(pv, *SEED_BOX)), "seeded at +p, the pre-fix value"
    np.testing.assert_array_equal(seeds[0], np.clip(-pv, *SEED_BOX))
    assert not np.allclose(seeds[0], -pv), "the clip must bind at this momentum, or it is untested"

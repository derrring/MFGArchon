"""Pinning tests for the H<->L convention and the admissible control set.

Issue #1642, capabilities B1 (hl-convention-pin) and B3 (controlcost-effective-domain).

B1 pins the (V, f) sign convention documented on ``MFGOperatorBase``:

    L(x, alpha, m, t) = L_ctrl(alpha) - V(x, t) - f(m)

asserted in its conjugate form, ``sup_alpha { p.alpha - L } == H``.

Which tests carry that pin, precisely -- a conjugate round trip only
discriminates when the two sides have INDEPENDENT sources for V and f:

- ``TestSeparableRoundTrip.test_conjugate_of_lagrangian_recovers_hamiltonian``
  -- its 9 ``V + f != 0`` rows (3 control costs x 3 non-zero (V, f) cells). These
  carried strict xfail until Issue #1645 (B2) flipped the sign in
  ``SeparableLagrangian.__call__``; they are now the load-bearing pin on the
  Separable side. Its 3 ``V0_f0`` rows are non-discriminating by construction,
  the disputed term being identically zero there.
- ``TestCongestionRoundTrip`` (12 tests = 3 costs x 4 (V, f) cells). The L side
  is an analytic closed form written out in this module, NOT
  ``H.legendre_transform()``, so V and f are independently sourced and a sign
  error in ``CongestionHamiltonian`` breaks the identity by 2(V+f).

What these tests catch:

- Adding V and f to BOTH H and L (the pre-B2 ``SeparableLagrangian`` fork):
  breaks the identity by exactly 2(V+f), constant in p (Issue #1645).
- A (V, f) sign error in ``CongestionHamiltonian.__call__``.
- A conjugate/optimization box that ignores the admissible control set: for
  ``L1ControlCost(lambda_=0.5)`` at p=5 an unrestricted sup returns 225.0
  against a true H of 4.5 (50x), because ``lagrangian()`` omits the indicator.
- A second owner of the admissible set diverging from ``effective_domain()``.
- Losing the domain for a Moreau-Yosida-wrapped cost (regularizing H must not
  enlarge A).

What they do NOT catch: the alpha-sign question on ``LagrangianBase.optimal_control``
(Issue #1642, B5) -- every shipped L_ctrl is even in alpha, so both sign
conventions give the same conjugate value. ``test_conjugate_is_alpha_sign_blind``
pins that evenness rather than leaving it implicit.
"""

from __future__ import annotations

import pytest

import numpy as np
from scipy.optimize import minimize_scalar

from mfgarchon.core.hamiltonian import (
    BoundedControlCost,
    CongestionHamiltonian,
    ControlCostBase,
    L1ControlCost,
    QuadraticControlCost,
    SeparableLagrangian,
)

# Sweep points chosen to straddle every kink the shipped costs have, so the pin
# is not evaluated only on smooth interiors:
#   0.0  L1 dead zone (|p| <= lambda) and the kink at the origin
#   0.5  exactly on the L1 activation threshold lambda=0.5
#   1.5  Bounded still quadratic (threshold lambda*max_control = 2.0)
#   3.0  Bounded saturated, L1 active
#   8.0  deep saturation, where an unrestricted conjugate diverges worst
P_SWEEP = [0.0, 0.5, 1.5, 3.0, 8.0]

X_POINT = np.array([0.25])
M_VALUE = 2.0

# Non-trivial V and f: constant-in-x V is enough because the fork is an additive
# offset, and a constant makes the expected gap 2(V+f) exactly computable.
V_NONZERO = 0.7
F_SLOPE = 0.3


def _potential(x, t):
    return V_NONZERO


def _coupling(m):
    return F_SLOPE * m


def _conjugate(L, p, *, bounds, alpha_sign=1.0):
    """sup_alpha { alpha_sign * p * alpha - L(x, alpha, m, t) } over ``bounds``.

    ``bounds`` must be the admissible control set: the shipped ``lagrangian()``
    implementations omit the indicator of their effective domain, so an
    unrestricted sup silently overshoots (see module docstring).
    """

    def neg_objective(a):
        return -(alpha_sign * p * a - float(L(X_POINT, np.array([a]), M_VALUE, 0.0)))

    res = minimize_scalar(neg_objective, bounds=bounds, method="bounded", options={"xatol": 1e-13})
    return -res.fun


def _search_box(cost: ControlCostBase) -> tuple[float, float]:
    """Admissible set from the single owner, widened when A = R."""
    return cost.effective_domain() or (-50.0, 50.0)


CONTROL_COSTS = {
    "quadratic": lambda: QuadraticControlCost(lambda_=2.0),
    "bounded": lambda: BoundedControlCost(lambda_=1.0, max_control=2.0),
    "l1": lambda: L1ControlCost(lambda_=0.5),
}

CONGESTION_SLOPE = 3.0


def _congestion_factor(m):
    return 1.0 + CONGESTION_SLOPE * m


# Analytic control part of L for CongestionHamiltonian, written out by hand.
#
# H_kin(p) = g(p) / c(m) with g = control_cost.evaluate, so the control part of L
# is the conjugate (g/c)*(alpha) = g*(c*alpha) / c. The lambda_ values below
# restate those of CONTROL_COSTS deliberately: an independent transcription is
# the point, so that these formulas share no source with the Hamiltonian.
#
#   quadratic  g* = lambda/2 |a|^2          -> lambda*c/2 |a|^2,   A = R
#   bounded    g* = lambda/2 |a|^2 + I_A    -> lambda*c/2 |a|^2,   A/c
#   l1         g* = lambda |a| + I_A        -> lambda |a|,         A/c
CONGESTION_L_CTRL = {
    "quadratic": lambda a, c_m: 0.5 * 2.0 * c_m * a**2,
    "bounded": lambda a, c_m: 0.5 * 1.0 * c_m * a**2,
    "l1": lambda a, c_m: 0.5 * abs(a),
}


def _congestion_lagrangian(cost_name, potential, coupling):
    """Analytic L(x, alpha, m, t) = L_ctrl^{c(m)}(alpha) - V(x, t) - f(m).

    Independently sourced from ``CongestionHamiltonian`` -- see the note on
    ``TestCongestionRoundTrip`` for why that independence is what makes the
    round trip discriminating.
    """
    l_ctrl = CONGESTION_L_CTRL[cost_name]

    def L(x, alpha, m, t=0.0):
        v = potential(x, t) if potential is not None else 0.0
        f = coupling(m) if coupling is not None else 0.0
        return l_ctrl(float(np.atleast_1d(alpha)[0]), _congestion_factor(m)) - v - f

    return L


def _congestion_box(cost) -> tuple[float, float]:
    """A / c(m). Congestion SHRINKS the admissible set: dom((g/c)*) = dom(g*)/c."""
    domain = cost.effective_domain()
    if domain is None:
        return (-50.0, 50.0)
    c_m = _congestion_factor(M_VALUE)
    return (domain[0] / c_m, domain[1] / c_m)


# (V, f) grid. The separable fork is invisible when V + f == 0, so the
# (zero, zero) cell must pass today and the other three must not.
VF_GRID = {
    "V0_f0": (None, None),
    "V0_fnz": (None, _coupling),
    "Vnz_f0": (_potential, None),
    "Vnz_fnz": (_potential, _coupling),
}


def _separable_params():
    """Cartesian product. Every cell passes since Issue #1645 (B2) closed the fork."""
    params = []
    for cost_name in CONTROL_COSTS:
        for vf_name, (pot, coup) in VF_GRID.items():
            params.append(pytest.param(cost_name, pot, coup, id=f"{cost_name}-{vf_name}"))
    return params


class TestSeparableRoundTrip:
    """sup_alpha { p.alpha - L } == H for SeparableLagrangian / SeparableHamiltonian."""

    @pytest.mark.parametrize(("cost_name", "potential", "coupling"), _separable_params())
    def test_conjugate_of_lagrangian_recovers_hamiltonian(self, cost_name, potential, coupling):
        cost = CONTROL_COSTS[cost_name]()
        L = SeparableLagrangian(control_cost=cost, potential=potential, coupling=coupling)
        H = L.as_hamiltonian()
        box = _search_box(cost)

        for p in P_SWEEP:
            expected = float(H(X_POINT, M_VALUE, np.array([p])))
            got = _conjugate(L, p, bounds=box)
            assert got == pytest.approx(expected, abs=1e-6), (
                f"{cost_name} p={p}: conjugate of L gave {got}, H gave {expected}"
            )

    @pytest.mark.parametrize("cost_name", list(CONTROL_COSTS))
    def test_conjugate_is_alpha_sign_blind(self, cost_name):
        """Every shipped L_ctrl is even in alpha, so sup{+p.a - L} == sup{-p.a - L}.

        This is why the (V, f) pin above is independent of the Issue #1642 B5
        alpha-sign question. If a non-even control cost is ever added, this test
        fails and B5 stops being optional.
        """
        cost = CONTROL_COSTS[cost_name]()
        L = SeparableLagrangian(control_cost=cost)
        box = _search_box(cost)
        for p in P_SWEEP:
            plus = _conjugate(L, p, bounds=box, alpha_sign=+1.0)
            minus = _conjugate(L, p, bounds=box, alpha_sign=-1.0)
            assert plus == pytest.approx(minus, abs=1e-6), f"{cost_name} p={p}: L_ctrl is not even in alpha"


class TestCongestionRoundTrip:
    """sup_alpha { p.alpha - L } == H for the non-separable CongestionHamiltonian.

    The L side is the ANALYTIC Lagrangian built by ``_congestion_lagrangian``,
    deliberately not ``H.legendre_transform()``. That method returns a
    ``DualLagrangian``, which computes L = sup_p { p.alpha - H } from the SAME H
    object, so the round trip collapses to ``H** == H`` -- true for every convex
    H whatever the (V, f) signs are. Such a test asserts convexity, not a
    convention, and stays green under a sign flip in
    ``CongestionHamiltonian.__call__``.

    Sourcing V and f independently here is what gives the assertion teeth: a
    sign error on either term shifts H by 2V or 2f while the conjugate stays
    put. There is no shipped analytic CongestionLagrangian to import (Issue
    #1642, B6 owns that), hence the closed forms in this module.

    Tolerance is 1e-6, set by the single bounded scalar sup; the errors measured
    are ~4e-8. The fork this guards against is 2(V+f) = 2.6.
    """

    @pytest.mark.parametrize("cost_name", list(CONTROL_COSTS))
    @pytest.mark.parametrize(("vf_name", "vf"), list(VF_GRID.items()))
    def test_analytic_conjugate_recovers_hamiltonian(self, cost_name, vf_name, vf):
        potential, coupling = vf
        cost = CONTROL_COSTS[cost_name]()
        H = CongestionHamiltonian(
            control_cost=cost,
            congestion_factor=_congestion_factor,
            potential=potential,
            coupling=coupling,
        )
        L = _congestion_lagrangian(cost_name, potential, coupling)
        box = _congestion_box(cost)

        for p in P_SWEEP:
            expected = float(H(X_POINT, M_VALUE, np.array([p])))
            got = _conjugate(L, p, bounds=box)
            assert got == pytest.approx(expected, abs=1e-6), (
                f"{cost_name}/{vf_name} p={p}: conjugate of the analytic L gave {got}, H gave {expected}"
            )


class TestEffectiveDomain:
    """ControlCostBase.effective_domain() is the single owner of A (B3)."""

    def test_quadratic_is_unbounded(self):
        assert QuadraticControlCost(lambda_=2.0).effective_domain() is None

    def test_bounded_reports_max_control(self):
        assert BoundedControlCost(lambda_=1.0, max_control=2.0).effective_domain() == (-2.0, 2.0)

    def test_l1_reports_bang_bang_interval(self):
        assert L1ControlCost(lambda_=0.5).effective_domain() == (-1.0, 1.0)

    @pytest.mark.parametrize(
        ("cost_factory", "expected"),
        [
            (lambda: QuadraticControlCost(lambda_=2.0), None),
            (lambda: BoundedControlCost(lambda_=1.0, max_control=2.0), (-2.0, 2.0)),
            (lambda: L1ControlCost(lambda_=0.5), (-1.0, 1.0)),
        ],
    )
    def test_control_bounds_delegates_to_effective_domain(self, cost_factory, expected):
        """The isinstance ladder is gone; SeparableLagrangian must read the owner."""
        cost = cost_factory()
        assert SeparableLagrangian(control_cost=cost).control_bounds() == expected
        assert SeparableLagrangian(control_cost=cost).control_bounds() == cost.effective_domain()

    @pytest.mark.parametrize(
        ("cost_factory", "expected"),
        [
            (lambda: BoundedControlCost(lambda_=1.0, max_control=2.0), (-2.0, 2.0)),
            (lambda: L1ControlCost(lambda_=0.5), (-1.0, 1.0)),
            (lambda: QuadraticControlCost(lambda_=2.0), None),
        ],
    )
    def test_moreau_yosida_preserves_domain(self, cost_factory, expected):
        """dom(L + eps/2|.|^2) == dom(L). Smoothing H must not enlarge A.

        BEHAVIOR CHANGE vs the removed isinstance ladder, which returned None
        here because a wrapped cost matched neither branch -- a regularized
        bounded problem silently claimed an unbounded control set. Reachable via
        SeparableHamiltonian.regularize() -> MFGComponents deriving a
        SeparableLagrangian -> hjb_semi_lagrangian control_bounds().
        """
        wrapped = cost_factory().regularize(0.1)
        assert wrapped.effective_domain() == expected
        assert SeparableLagrangian(control_cost=wrapped).control_bounds() == expected

    def test_lagrangian_does_not_enforce_the_domain(self):
        """Forcing evidence for B3, pinned so the gap cannot be forgotten.

        ``lagrangian()`` returns a finite value for infeasible alpha. Making it
        fail loud is Issue #1644 (B4); when that lands, this test must be
        replaced by one asserting the raise -- not deleted.
        """
        cost = BoundedControlCost(lambda_=1.0, max_control=2.0)
        infeasible = np.array([5.0])
        assert cost.effective_domain() == (-2.0, 2.0)
        assert float(cost.lagrangian(infeasible)) == pytest.approx(12.5)

    def test_unrestricted_conjugate_overshoots_without_the_domain(self):
        """Why every conjugate consumer must read effective_domain().

        L1 at p=5: true H is 4.5; a sup taken over (-50, 50) instead of the
        effective domain returns ~225.
        """
        cost = L1ControlCost(lambda_=0.5)
        L = SeparableLagrangian(control_cost=cost)
        # Issue #1672/D9: evaluate reduces over the component axis, so a (1,) momentum
        # yields one scalar rather than a length-1 array.
        true_h = float(cost.evaluate(np.array([5.0])))

        on_domain = _conjugate(L, 5.0, bounds=cost.effective_domain())
        off_domain = _conjugate(L, 5.0, bounds=(-50.0, 50.0))

        assert on_domain == pytest.approx(true_h, abs=1e-6)
        assert off_domain > 40.0 * true_h

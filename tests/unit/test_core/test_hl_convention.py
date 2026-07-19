"""Pinning tests for the H<->L convention and the admissible control set.

Issue #1642, capabilities B1 (hl-convention-pin) and B3 (controlcost-effective-domain).

B1 pins the (V, f) sign convention documented on ``MFGOperatorBase``:

    L(x, alpha, m, t) = L_ctrl(alpha) - V(x, t) - f(m)

asserted in its conjugate form, ``sup_alpha { p.alpha - L } == H``, over
(Separable, Congestion) x (Quadratic, Bounded, L1) x (V=0, V!=0) x (f=0, f!=0).

What these tests catch:

- Adding V and f to BOTH H and L (the ``SeparableLagrangian`` fork): breaks the
  identity by exactly 2(V+f), constant in p. Recorded as strict xfail, Issue #1645.
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

# (V, f) grid. The separable fork is invisible when V + f == 0, so the
# (zero, zero) cell must pass today and the other three must not.
VF_GRID = {
    "V0_f0": (None, None),
    "V0_fnz": (None, _coupling),
    "Vnz_f0": (_potential, None),
    "Vnz_fnz": (_potential, _coupling),
}


def _separable_params():
    """Cartesian product, xfailing exactly the cells where V + f != 0."""
    params = []
    for cost_name in CONTROL_COSTS:
        for vf_name, (pot, coup) in VF_GRID.items():
            marks = ()
            if (pot, coup) != (None, None):
                marks = pytest.mark.xfail(
                    strict=True,
                    reason=(
                        "Issue #1645 (B2): SeparableLagrangian.__call__ returns "
                        "L_ctrl + V + f instead of L_ctrl - V - f, so it is not "
                        "self-conjugate; the gap is exactly 2(V+f). Flip the sign "
                        "in B2 and this xfail becomes an XPASS (strict=True fails "
                        "the suite) -- remove the mark then."
                    ),
                )
            params.append(pytest.param(cost_name, pot, coup, marks=marks, id=f"{cost_name}-{vf_name}"))
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
    def test_separable_fork_gap_is_exactly_two_v_plus_f(self, cost_name):
        """Quantify the Issue #1645 fork rather than only marking it xfail.

        Pins the gap at 2(V+f) and pins that it is CONSTANT in p -- a p-dependent
        gap would mean the control term is also wrong, not just the (V, f) sign.
        """
        cost = CONTROL_COSTS[cost_name]()
        L = SeparableLagrangian(control_cost=cost, potential=_potential, coupling=_coupling)
        H = L.as_hamiltonian()
        box = _search_box(cost)
        expected_gap = 2.0 * (V_NONZERO + F_SLOPE * M_VALUE)

        gaps = [float(H(X_POINT, M_VALUE, np.array([p]))) - _conjugate(L, p, bounds=box) for p in P_SWEEP]
        for p, gap in zip(P_SWEEP, gaps, strict=True):
            assert gap == pytest.approx(expected_gap, abs=1e-5), (
                f"{cost_name} p={p}: gap {gap} != 2(V+f) {expected_gap}"
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
    """H -> L -> H involution for the non-separable congestion Hamiltonian.

    There is no analytic CongestionLagrangian (Issue #1642, B6 owns that), so the
    L side is the numerical ``DualLagrangian``. The tolerance below is set by
    NESTED numerical optimization (an outer sup over alpha of an inner sup over
    p), NOT by the convention: the fork this guards against is 2(V+f) = 2.6,
    two orders of magnitude above the tolerance.
    """

    NESTED_OPT_TOL = 5e-2

    @pytest.mark.parametrize("cost_name", ["quadratic", "bounded"])
    @pytest.mark.parametrize(("vf_name", "vf"), list(VF_GRID.items()))
    def test_dual_lagrangian_round_trips(self, cost_name, vf_name, vf):
        potential, coupling = vf
        cost = CONTROL_COSTS[cost_name]()
        H = CongestionHamiltonian(
            control_cost=cost,
            congestion_factor=lambda m: 1.0 + 3.0 * m,
            potential=potential,
            coupling=coupling,
        )
        L = H.legendre_transform(p_bounds=(-200.0, 200.0), n_search=400)

        for p in [0.0, 0.5, 1.5, 3.0]:
            expected = float(H(X_POINT, M_VALUE, np.array([p])))
            got = _conjugate(L, p, bounds=(-50.0, 50.0))
            assert got == pytest.approx(expected, abs=self.NESTED_OPT_TOL), (
                f"{cost_name}/{vf_name} p={p}: round-trip gave {got}, H gave {expected}"
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
        true_h = float(cost.evaluate(np.array([5.0]))[0])

        on_domain = _conjugate(L, 5.0, bounds=cost.effective_domain())
        off_domain = _conjugate(L, 5.0, bounds=(-50.0, 50.0))

        assert on_domain == pytest.approx(true_h, abs=1e-6)
        assert off_domain > 40.0 * true_h

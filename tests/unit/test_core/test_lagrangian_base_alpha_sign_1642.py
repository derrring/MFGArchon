"""Issue #1642 capability B5: LagrangianBase's alpha* / H sign conventions.

``LagrangianBase.optimal_control`` used to return the bare conjugate maximizer
``argmax_alpha {p.alpha - L}`` -- i.e. ``+p/lambda`` under MINIMIZE, where both
``HamiltonianBase.optimal_control`` and the analytic
``SeparableLagrangian.optimal_control`` return ``-p/lambda``.
``LagrangianBase.evaluate_hamiltonian`` then fed that wrong alpha* back into
``p.alpha* - L(alpha*)``, and the two sign errors CANCELLED: the composed
``DualLagrangian.evaluate_hamiltonian`` returned the correct ``H``.

That cancellation is why this file exists. Fixing either site alone re-breaks
the composed path, so a single-site revert must fail a test. The two classes
below are split accordingly:

- ``TestOptimalControlSign``      catches a revert of ``optimal_control``
- ``TestEvaluateHamiltonianValue`` catches a revert of ``evaluate_hamiltonian``

Both use LagrangianBase subclasses that do NOT override the two methods, so the
base implementations are the code under test. ``SeparableLagrangian`` overrides
both analytically and serves as the convention reference.
"""

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import (
    BoundedControlCost,
    L1ControlCost,
    LagrangianBase,
    OptimizationSense,
    QuadraticControlCost,
    SeparableHamiltonian,
    SeparableLagrangian,
)

X = np.array([0.5])
M = 0.3
T = 0.0
# p=0 is the kink/stationary edge; +/- pairs catch a global sign flip that a
# single positive p would show as a magnitude change only.
P_VALUES = [np.array([2.0]), np.array([1.0]), np.array([-1.5]), np.array([0.0])]


class PlainQuadraticL(LagrangianBase):
    """L(alpha) = lambda/2 |alpha|^2, with NO analytic override.

    Exercises the base-class numerical path that SeparableLagrangian bypasses.
    """

    def __init__(self, lam: float, sense: OptimizationSense = OptimizationSense.MINIMIZE):
        super().__init__(sense=sense)
        self.lam = lam

    def __call__(self, x, alpha, m, t=0.0):
        return float(0.5 * self.lam * np.sum(np.atleast_1d(alpha) ** 2))


class AsymmetricL(LagrangianBase):
    """L(alpha) = 0.5 alpha^2 + 0.3 alpha -- deliberately NOT even in alpha.

    Discrimination matters here: for even L, L*(p) == L*(-p), so an even-L test
    cannot tell 'evaluate_hamiltonian is the conjugate at +p' from 'at -p'. This
    L separates them -- at p=2, L*(p)=1.445 while L*(-p)=2.645.

    Closed form: argmax_alpha {p.alpha - L} = p - 0.3, so L*(p) = 0.5 (p-0.3)^2.
    """

    OFFSET = 0.3

    def __call__(self, x, alpha, m, t=0.0):
        a = np.atleast_1d(alpha)
        return float(0.5 * np.sum(a**2) + self.OFFSET * np.sum(a))

    @classmethod
    def conjugate(cls, p: float) -> float:
        return 0.5 * (p - cls.OFFSET) ** 2

    @classmethod
    def conjugate_argmax(cls, p: float) -> float:
        return p - cls.OFFSET


class TestOptimalControlSign:
    """alpha* = -sign * dH/dp, matching HamiltonianBase and SeparableLagrangian.

    Reverting ``optimal_control`` to ``return self._conjugate_argmax(...)`` flips
    every MINIMIZE row here.
    """

    @pytest.mark.parametrize("p", P_VALUES)
    def test_minimize_alpha_star_is_negative_p_over_lambda(self, p):
        """MINIMIZE: alpha* = -p/lambda. The pre-#1642 base returned +p/lambda."""
        L = PlainQuadraticL(2.0, sense=OptimizationSense.MINIMIZE)
        np.testing.assert_allclose(L.optimal_control(X, M, p, T), [-p[0] / 2.0], atol=1e-6)

    @pytest.mark.parametrize("p", P_VALUES)
    def test_maximize_alpha_star_is_positive_p_over_lambda(self, p):
        """MAXIMIZE: alpha* = +p/lambda -- unchanged by #1642.

        Pins that the fix is sense-aware rather than an unconditional negation:
        a ``return -self._conjugate_argmax(...)`` would break this row.
        """
        L = PlainQuadraticL(2.0, sense=OptimizationSense.MAXIMIZE)
        np.testing.assert_allclose(L.optimal_control(X, M, p, T), [p[0] / 2.0], atol=1e-6)

    @pytest.mark.parametrize("sense", [OptimizationSense.MINIMIZE, OptimizationSense.MAXIMIZE])
    @pytest.mark.parametrize("p", P_VALUES)
    def test_base_agrees_with_analytic_separable_override(self, sense, p):
        """Single source of truth: the numerical base path and the analytic
        SeparableLagrangian override must produce the same alpha* for the same L.

        These are two parallel implementations of one quantity; before #1642 they
        held opposite conventions with nothing asserting agreement.
        """
        cost = QuadraticControlCost(lambda_=2.0, sense=sense)
        analytic = SeparableLagrangian(control_cost=cost, sense=sense)
        numerical = PlainQuadraticL(2.0, sense=sense)
        np.testing.assert_allclose(
            numerical.optimal_control(X, M, p, T),
            analytic.optimal_control(X, M, p, T),
            atol=1e-6,
        )

    @pytest.mark.parametrize("sense", [OptimizationSense.MINIMIZE, OptimizationSense.MAXIMIZE])
    @pytest.mark.parametrize("p", [np.array([2.0]), np.array([1.0]), np.array([-1.5])])
    def test_dual_lagrangian_alpha_star_matches_its_source_hamiltonian(self, sense, p):
        """The composed path: H -> DualLagrangian -> optimal_control must return
        the same alpha* as H.optimal_control. This is the docstring's promise.

        Doubly numerical (a grid sup inside a scalar maximization), so it carries
        ~2e-3 of grid-quantization noise -- hence atol=5e-3 rather than the 1e-6
        the single-transform rows use. The revert this must catch moves alpha* by
        2|alpha*| >= 1.0, so it is checked as an explicit sign match too; the
        magnitude assertion alone would still catch it with 200x margin.

        p=0 is excluded deliberately: alpha*=0 there, so the row cannot
        discriminate a sign revert at any tolerance.
        """
        cost = QuadraticControlCost(lambda_=2.0, sense=sense)
        H = SeparableHamiltonian(control_cost=cost, sense=sense)
        dual_L = H.legendre_transform(p_bounds=(-50.0, 50.0), n_search=8001)

        from_dual = dual_L.optimal_control(X, M, p, T)
        from_hamiltonian = H.optimal_control(X, M, p, T)
        np.testing.assert_allclose(from_dual, from_hamiltonian, atol=5e-3)
        assert np.sign(from_dual[0]) == np.sign(from_hamiltonian[0])

    def test_asymmetric_lagrangian_alpha_star_is_negated_argmax(self):
        """Asymmetric L: alpha* = -(p - 0.3), not -(p + 0.3) and not +(p - 0.3).

        Pins that the sign is applied to the maximizer rather than to p.
        """
        L = AsymmetricL(sense=OptimizationSense.MINIMIZE)
        for p_val in (2.0, -1.5, 0.0):
            np.testing.assert_allclose(
                L.optimal_control(X, M, np.array([p_val]), T),
                [-AsymmetricL.conjugate_argmax(p_val)],
                atol=1e-6,
            )

    def test_nd_branch_carries_the_same_sign(self):
        """The d>1 L-BFGS-B branch is a separate code path from the 1D scalar one."""
        L = PlainQuadraticL(2.0, sense=OptimizationSense.MINIMIZE)
        x2 = np.array([0.5, 0.5])
        p2 = np.array([2.0, -1.0])
        np.testing.assert_allclose(L.optimal_control(x2, M, p2, T), [-1.0, 0.5], atol=1e-5)

    @pytest.mark.parametrize(
        ("cost", "p", "expected"),
        [
            # |p|=0.7 > lambda=0.5 -> bang-bang alpha* = -sign(p) = -1
            (L1ControlCost(lambda_=0.5), np.array([0.7]), -1.0),
            (L1ControlCost(lambda_=0.5), np.array([-0.7]), 1.0),
            # Bounded: alpha* = -clip(p/lambda, +/-max_control) = -min(3/1, 2) = -2
            (BoundedControlCost(lambda_=1.0, max_control=2.0), np.array([3.0]), -2.0),
        ],
    )
    def test_nonsmooth_costs_agree_with_analytic_override(self, cost, p, expected):
        """Non-smooth / constrained costs: the numerical base path hits kinks and
        active bounds, where a sign error is easiest to hide behind a clipped value.
        """
        analytic = SeparableLagrangian(control_cost=cost)
        np.testing.assert_allclose(analytic.optimal_control(X, M, p, T), [expected], atol=1e-9)

        dual_L = SeparableHamiltonian(control_cost=cost).legendre_transform(p_bounds=(-50.0, 50.0), n_search=4001)
        np.testing.assert_allclose(dual_L.optimal_control(X, M, p, T), [expected], atol=1e-3)


class TestEvaluateHamiltonianValue:
    """H = sup_alpha {p.alpha - L} = L*(p), independent of OptimizationSense (#1185).

    Reverting ``evaluate_hamiltonian`` to evaluate at ``self.optimal_control(...)``
    instead of ``self._conjugate_argmax(...)`` flips every MINIMIZE row here (and
    produces a value that is neither L*(p) nor L*(-p) for asymmetric L).
    """

    @pytest.mark.parametrize("sense", [OptimizationSense.MINIMIZE, OptimizationSense.MAXIMIZE])
    @pytest.mark.parametrize("p", P_VALUES)
    def test_quadratic_hamiltonian_is_the_positive_conjugate(self, sense, p):
        """H(p) = |p|^2/(2 lambda) >= 0, both senses. A sign revert makes it <= 0."""
        L = PlainQuadraticL(2.0, sense=sense)
        expected = p[0] ** 2 / (2 * 2.0)
        np.testing.assert_allclose(L.evaluate_hamiltonian(X, M, p, T), expected, atol=1e-9)

    def test_asymmetric_hamiltonian_is_conjugate_at_plus_p(self):
        """The discriminating case: L*(p) vs L*(-p).

        At p=2 these are 1.445 and 2.645; evaluating at the sign-flipped control
        instead gives yet a third value. Only the conjugate at +p passes.
        """
        L = AsymmetricL(sense=OptimizationSense.MINIMIZE)
        for p_val in (2.0, -1.5, 1.0):
            got = L.evaluate_hamiltonian(X, M, np.array([p_val]), T)
            np.testing.assert_allclose(got, AsymmetricL.conjugate(p_val), atol=1e-9)
            # and is NOT the conjugate at -p (guards a p -> -p slip)
            if p_val != 0.0:
                assert abs(got - AsymmetricL.conjugate(-p_val)) > 1e-3

    @pytest.mark.parametrize("sense", [OptimizationSense.MINIMIZE, OptimizationSense.MAXIMIZE])
    @pytest.mark.parametrize("p", P_VALUES)
    def test_dual_lagrangian_round_trips_to_its_source_hamiltonian(self, sense, p):
        """THE composed path the #1642 map flags: H -> DualLagrangian ->
        evaluate_hamiltonian must recover H. This value was already correct
        before the fix (by cancellation) and must stay correct after it.
        """
        cost = QuadraticControlCost(lambda_=2.0, sense=sense)
        H = SeparableHamiltonian(
            control_cost=cost,
            potential=lambda x_, t_: 0.7,
            coupling=lambda m_: -(m_**2),
            sense=sense,
        )
        dual_L = H.legendre_transform(p_bounds=(-50.0, 50.0), n_search=2001)
        np.testing.assert_allclose(
            dual_L.evaluate_hamiltonian(X, M, p, T),
            H(X, M, p, T),
            atol=1e-6,
        )

    def test_nd_branch_hamiltonian_value(self):
        """d>1 branch: H([2,-1]) = (4+1)/(2*2) = 1.25."""
        L = PlainQuadraticL(2.0, sense=OptimizationSense.MINIMIZE)
        got = L.evaluate_hamiltonian(np.array([0.5, 0.5]), M, np.array([2.0, -1.0]), T)
        np.testing.assert_allclose(got, 1.25, atol=1e-8)


class TestSitesAreIndependentlyPinned:
    """The joint invariant: alpha* and H are related by alpha* = -sign * dH/dp.

    Reverting BOTH sites together restores evaluate_hamiltonian but leaves
    optimal_control wrong, so this holds only when both are fixed.
    """

    @pytest.mark.parametrize("sense", [OptimizationSense.MINIMIZE, OptimizationSense.MAXIMIZE])
    def test_envelope_relation_between_the_two_methods(self, sense):
        """H(p) == p . (-sign * alpha*) - L(-sign * alpha*).

        Recovers dH/dp from the published alpha* and checks it reproduces the
        published H. Fails if either method drifts from the shared convention.
        """
        L = PlainQuadraticL(2.0, sense=sense)
        expected_sign = 1 if sense == OptimizationSense.MINIMIZE else -1
        assert L._sign == expected_sign

        for p in P_VALUES:
            alpha_star = L.optimal_control(X, M, p, T)
            dH_dp = -L._sign * alpha_star
            reconstructed = float(np.sum(np.atleast_1d(p) * dH_dp)) - float(L(X, dH_dp, M, T))
            np.testing.assert_allclose(reconstructed, L.evaluate_hamiltonian(X, M, p, T), atol=1e-9)

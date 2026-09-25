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

    def __init__(self, lam: float):
        super().__init__()
        self.lam = lam

    def __call__(self, t, x, alpha, m):
        return float(0.5 * self.lam * np.sum(np.atleast_1d(alpha) ** 2))


class AsymmetricL(LagrangianBase):
    """L(alpha) = 0.5 alpha^2 + 0.3 alpha -- deliberately NOT even in alpha.

    Discrimination matters here: for even L, L*(p) == L*(-p), so an even-L test
    cannot tell 'evaluate_hamiltonian is the conjugate at +p' from 'at -p'. This
    L separates them -- at p=2, L*(p)=1.445 while L*(-p)=2.645.

    Closed form: argmax_alpha {p.alpha - L} = p - 0.3, so L*(p) = 0.5 (p-0.3)^2.
    """

    OFFSET = 0.3

    def __call__(self, t, x, alpha, m):
        a = np.atleast_1d(alpha)
        return float(0.5 * np.sum(a**2) + self.OFFSET * np.sum(a))

    @classmethod
    def conjugate(cls, p: float) -> float:
        return 0.5 * (p - cls.OFFSET) ** 2

    @classmethod
    def analytic_argmax(cls, p: float) -> float:
        """argmax {+p.a - L} = p - OFFSET. The OLD pairing's maximizer; kept because the
        pre-#2375 relation is still what `conjugate_argmax`'s negation is measured against."""
        return p - cls.OFFSET

    @classmethod
    def true_minimiser(cls, p: float) -> float:
        """argmin {p.a + L(a)} = -(p + OFFSET) -- the control that actually minimises cost-to-go.

        Under ruling 5 of #2375 this is what `optimal_control` returns. Note it is NOT
        `-analytic_argmax(p)`: those differ by 2*OFFSET and coincide only for even L, which is
        the whole content of the ruling.
        """
        return -(p + cls.OFFSET)


class TestOptimalControlSign:
    """alpha* = -dH/dp, matching HamiltonianBase and SeparableLagrangian.

    The ``sign`` factor this line used to carry went with `OptimizationSense` in #2373; under
    ruling 5 of #2375 the relation holds for every L rather than only for even ones.

    Reverting ``optimal_control`` to ``return self.conjugate_argmax(...)`` flips
    every MINIMIZE row here.
    """

    @pytest.mark.parametrize("p", P_VALUES)
    def test_minimize_alpha_star_is_negative_p_over_lambda(self, p):
        """MINIMIZE: alpha* = -p/lambda. The pre-#1642 base returned +p/lambda."""
        L = PlainQuadraticL(2.0)
        np.testing.assert_allclose(L.optimal_control(x=X, m=M, p=p, t=T), [-p[0] / 2.0], atol=1e-6)

    @pytest.mark.parametrize("p", P_VALUES)
    def test_base_agrees_with_analytic_separable_override(self, p):
        """Single source of truth: the numerical base path and the analytic
        SeparableLagrangian override must produce the same alpha* for the same L.

        These are two parallel implementations of one quantity; before #1642 they
        held opposite conventions with nothing asserting agreement.
        """
        cost = QuadraticControlCost(lambda_=2.0)
        analytic = SeparableLagrangian(control_cost=cost)
        numerical = PlainQuadraticL(2.0)
        np.testing.assert_allclose(
            numerical.optimal_control(x=X, m=M, p=p, t=T),
            analytic.optimal_control(x=X, m=M, p=p, t=T),
            atol=1e-6,
        )

    @pytest.mark.parametrize("p", [np.array([2.0]), np.array([1.0]), np.array([-1.5])])
    def test_dual_lagrangian_alpha_star_matches_its_source_hamiltonian(self, p):
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
        cost = QuadraticControlCost(lambda_=2.0)
        H = SeparableHamiltonian(control_cost=cost)
        dual_L = H.legendre_transform(p_bounds=(-50.0, 50.0), n_search=8001)

        from_dual = dual_L.optimal_control(x=X, m=M, p=p, t=T)
        from_hamiltonian = H.optimal_control(x=X, m=M, p=p, t=T)
        np.testing.assert_allclose(from_dual, from_hamiltonian, atol=5e-3)
        assert np.sign(from_dual[0]) == np.sign(from_hamiltonian[0])

    def test_asymmetric_lagrangian_alpha_star_is_negated_argmax(self):
        """Asymmetric L: alpha* = -(p + 0.3) -- the control that actually minimises cost-to-go.

        [INVERTED 2026-09-21 by ruling 5 of #2375.] This assertion previously read
        ``-analytic_argmax(p) = -(p - 0.3)``, and the docstring it replaced ended:

            Ruling 5 of #2375 moves the library to H = sup{-p.a - L}; the test for that
            change needs a non-even L to discriminate at all, and this is it. When that
            lands, this assertion inverts.

        It has landed and it has inverted. The prediction was exact, including that this
        class is the discriminator: the ruling's own work order said the discriminating L
        "must be constructed" because every shipped cost is even, and it was already here.
        This test is what went red when the convention flipped.

        The two candidates differ by 2*OFFSET and coincide for every even L, so nothing
        else in the suite could tell them apart. p=0.0 is retained deliberately: it is the
        one p where -(p - 0.3) and -(p + 0.3) still differ (+0.3 against -0.3), so it
        discriminates rather than passing vacuously.
        """
        L = AsymmetricL()
        for p_val in (2.0, -1.5, 0.0):
            np.testing.assert_allclose(
                L.optimal_control(x=X, m=M, p=np.array([p_val]), t=T),
                [AsymmetricL.true_minimiser(p_val)],
                atol=1e-6,
            )

    def test_nd_branch_carries_the_same_sign(self):
        """The d>1 L-BFGS-B branch is a separate code path from the 1D scalar one."""
        L = PlainQuadraticL(2.0)
        x2 = np.array([0.5, 0.5])
        p2 = np.array([2.0, -1.0])
        np.testing.assert_allclose(L.optimal_control(x=x2, m=M, p=p2, t=T), [-1.0, 0.5], atol=1e-5)

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
        np.testing.assert_allclose(analytic.optimal_control(x=X, m=M, p=p, t=T), [expected], atol=1e-9)

        dual_L = SeparableHamiltonian(control_cost=cost).legendre_transform(p_bounds=(-50.0, 50.0), n_search=4001)
        np.testing.assert_allclose(dual_L.optimal_control(x=X, m=M, p=p, t=T), [expected], atol=1e-3)


class TestEvaluateHamiltonianValue:
    """H = sup_alpha {-p.alpha - L} = L*(-p) (#1185; the sign is ruling 5 of #2375).

    Reverting ``evaluate_hamiltonian`` to evaluate at ``self.optimal_control(...)``
    instead of ``self.conjugate_argmax(...)`` flips every MINIMIZE row here (and
    produces a value that is neither L*(p) nor L*(-p) for asymmetric L).
    """

    @pytest.mark.parametrize("p", P_VALUES)
    def test_quadratic_hamiltonian_is_the_positive_conjugate(self, p):
        """H(p) = |p|^2/(2 lambda) >= 0. A sign revert makes it <= 0."""
        L = PlainQuadraticL(2.0)
        expected = p[0] ** 2 / (2 * 2.0)
        np.testing.assert_allclose(L.evaluate_hamiltonian(x=X, m=M, p=p, t=T), expected, atol=1e-9)

    def test_asymmetric_hamiltonian_is_conjugate_at_minus_p(self):
        """The discriminating case: L*(-p), not L*(p). [RENAMED AND INVERTED 2026-09-21, #2375 ruling 5.]

        Was ``test_asymmetric_hamiltonian_is_conjugate_at_plus_p`` asserting ``conjugate(p)``.
        Ruling 5 makes H = sup{-p.a - L} = L*(-p), so the assertion and the name both move; a
        name left saying ``plus_p`` would be a test whose title contradicts its body, which is
        worse than either convention.

        At p=2 the two candidates are 1.445 and 2.645 -- still 1.2 apart, so the guard below
        still guards. Evaluating at the sign-flipped control gives a third value again.
        """
        L = AsymmetricL()
        for p_val in (2.0, -1.5, 1.0):
            got = L.evaluate_hamiltonian(x=X, m=M, p=np.array([p_val]), t=T)
            np.testing.assert_allclose(got, AsymmetricL.conjugate(-p_val), atol=1e-9)
            # and is NOT the conjugate at +p (guards a p -> -p slip, now in the other direction)
            if p_val != 0.0:
                assert abs(got - AsymmetricL.conjugate(p_val)) > 1e-3

    @pytest.mark.parametrize("p", P_VALUES)
    def test_dual_lagrangian_round_trips_to_its_source_hamiltonian(self, p):
        """THE composed path the #1642 map flags: H -> DualLagrangian ->
        evaluate_hamiltonian must recover H. This value was already correct
        before the fix (by cancellation) and must stay correct after it.
        """
        cost = QuadraticControlCost(lambda_=2.0)
        H = SeparableHamiltonian(
            control_cost=cost,
            potential=lambda t, x: 0.7,
            coupling=lambda m_: -(m_**2),
        )
        dual_L = H.legendre_transform(p_bounds=(-50.0, 50.0), n_search=2001)
        np.testing.assert_allclose(
            dual_L.evaluate_hamiltonian(x=X, m=M, p=p, t=T),
            H(x=X, m=M, p=p, t=T),
            atol=1e-6,
        )

    def test_nd_branch_hamiltonian_value(self):
        """d>1 branch: H([2,-1]) = (4+1)/(2*2) = 1.25."""
        L = PlainQuadraticL(2.0)
        got = L.evaluate_hamiltonian(x=np.array([0.5, 0.5]), m=M, p=np.array([2.0, -1.0]), t=T)
        np.testing.assert_allclose(got, 1.25, atol=1e-8)


class TestSitesAreIndependentlyPinned:
    """The joint invariant: alpha* and H are related by alpha* = -sign * dH/dp.

    Reverting BOTH sites together restores evaluate_hamiltonian but leaves
    optimal_control wrong, so this holds only when both are fixed.
    """

    def test_envelope_relation_between_the_two_methods(self):
        """H(p) == p . (-alpha*) - L(-alpha*).

        Recovers dH/dp from the published alpha* and checks it reproduces the
        published H. Fails if either method drifts from the shared convention.
        The factor was `-L._sign * alpha*` until #2373 removed the direction;
        with one direction it is a plain negation, and reading a deleted
        attribute here would have been a silent `AttributeError` in a test that
        exists to catch silent drift.
        """
        L = PlainQuadraticL(2.0)

        for p in P_VALUES:
            alpha_star = L.optimal_control(x=X, m=M, p=p, t=T)
            dH_dp = -alpha_star
            reconstructed = float(np.sum(np.atleast_1d(p) * dH_dp)) - float(L(x=X, alpha=dH_dp, m=M, t=T))
            np.testing.assert_allclose(reconstructed, L.evaluate_hamiltonian(x=X, m=M, p=p, t=T), atol=1e-9)


class AnalyticUnboundedL(LagrangianBase):
    """L(alpha) = 0.5|alpha|^2 on A = R^d, supplying the closed-form maximizer.

    Exactly the shape no in-repo subclass has: an analytic override plus
    ``control_bounds() -> None``. SeparableLagrangian overrides every consumer
    and DualLagrangian overrides none, so without this class the public
    extension point has no caller and a regression on it is invisible to CI.

    Closed form: argmax_alpha {p.alpha - 0.5|alpha|^2} = p, so L*(p) = 0.5|p|^2,
    which is unbounded -- p=100 puts the maximizer far outside the (-10, 10)
    fallback box that the numerical default would otherwise silently truncate.
    """

    def __call__(self, t, x, alpha, m):
        return float(0.5 * np.sum(np.atleast_1d(alpha) ** 2))

    def conjugate_argmax(self, t, x, p, m):
        return np.atleast_1d(p).astype(float)

    def control_bounds(self):
        return None


class TestAnalyticOverrideReachesBothConsumers:
    """conjugate_argmax() is THE extension point: overriding it must reach
    optimal_control() AND evaluate_hamiltonian().

    Routing evaluate_hamiltonian() through the private maximizer while
    optimal_control() advertised itself as the override point sent
    analytic subclasses through the (-10, 10) numerical fallback: at p=100 it
    returned 949.999 for an exact 5000.0 (81% low), with no warning.
    """

    # p=25 and p=100 put the true maximizer outside (-10, 10); p=5 stays inside,
    # so it passes either way and cannot discriminate -- it is the control row.
    @pytest.mark.parametrize("p_val", [5.0, 25.0, 100.0])
    def test_evaluate_hamiltonian_uses_the_analytic_maximizer(self, p_val):
        """H(p) = 0.5 p^2 exactly, not the fallback-truncated value."""
        L = AnalyticUnboundedL()
        got = L.evaluate_hamiltonian(x=X, m=M, p=np.array([p_val]), t=T)
        np.testing.assert_allclose(got, 0.5 * p_val**2, rtol=1e-12)

    @pytest.mark.parametrize("p_val", [5.0, 25.0, 100.0])
    def test_optimal_control_uses_the_analytic_maximizer(self, p_val):
        """alpha* = -sign * argmax = -p under MINIMIZE, exact at every magnitude."""
        L = AnalyticUnboundedL()
        np.testing.assert_allclose(L.optimal_control(x=X, m=M, p=np.array([p_val]), t=T), [-p_val], rtol=1e-12)

    def test_both_consumers_read_one_source(self):
        """Byte-identical agreement, so the two cannot re-fork onto private copies."""
        L = AnalyticUnboundedL()
        for p_val in (5.0, 25.0, 100.0):
            p = np.array([p_val])
            dH_dp = L.conjugate_argmax(x=X, m=M, p=p, t=T)
            from_control = -L.optimal_control(x=X, m=M, p=p, t=T)
            assert from_control.tobytes() == dH_dp.tobytes()
            expected = float(np.sum(p * dH_dp)) - float(L(x=X, alpha=dH_dp, m=M, t=T))
            assert L.evaluate_hamiltonian(x=X, m=M, p=p, t=T) == expected


class TestFallbackBoxTruncationIsLoud:
    """A search terminating on the (-10, 10) fallback edge must raise.

    The box stands in for control_bounds() returning None; a maximizer on its
    edge is truncated, not optimal. Returning it silently is the fail-silent
    pattern that produced the 81% error above.
    """

    def test_conjugate_argmax_raises_when_the_fallback_box_binds(self):
        L = PlainQuadraticL(0.01)
        with pytest.raises(ValueError, match=r"control_bounds\(\)"):
            L.conjugate_argmax(x=X, m=M, p=np.array([100.0]), t=T)

    def test_evaluate_hamiltonian_propagates_the_raise(self):
        """The consumer must not swallow it -- this is the silent-81% path."""
        L = PlainQuadraticL(0.01)
        with pytest.raises(ValueError, match=r"fallback box"):
            L.evaluate_hamiltonian(x=X, m=M, p=np.array([100.0]), t=T)

    def test_optimal_control_propagates_the_raise(self):
        L = PlainQuadraticL(0.01)
        with pytest.raises(ValueError, match=r"fallback box"):
            L.optimal_control(x=X, m=M, p=np.array([100.0]), t=T)

    def test_nd_branch_also_raises(self):
        """The L-BFGS-B branch is a separate code path from the 1D scalar one."""
        L = PlainQuadraticL(0.01)
        with pytest.raises(ValueError, match=r"fallback box"):
            L.conjugate_argmax(x=np.array([0.5, 0.5]), m=M, p=np.array([100.0, -100.0]), t=T)

    def test_nd_partial_truncation_raises(self):
        """One component truncated, one interior -- the case that discriminates any from all.

        test_nd_branch_also_raises drives both components to the same bound, so it passes
        identically whether the guard quantifies with np.any or np.all. A partially
        truncated result is still a truncated result: the returned vector is not the argmax,
        so it must not be handed back silently. With lambda=0.01 the unconstrained maximizer
        is (10000.0, 1.0) -- component 0 is pinned to the fallback edge while component 1 sits
        interior at 1.0.
        """
        L = PlainQuadraticL(0.01)
        with pytest.raises(ValueError, match=r"fallback box"):
            L.conjugate_argmax(x=np.array([0.5, 0.5]), m=M, p=np.array([100.0, 0.01]), t=T)

    def test_proximal_raises_on_the_same_hazard(self):
        """Same fallback box, same truncation, same owner (_resolve_search_bounds)."""
        L = PlainQuadraticL(2.0)
        with pytest.raises(ValueError, match=r"control_bounds\(\)"):
            L.proximal(1.0, np.array([50.0]))

    def test_declared_bounds_at_the_edge_do_not_raise(self):
        """A real active constraint is legitimate -- only the FALLBACK box is
        a stand-in. Guarding on the value alone would break bounded controls."""

        class BoundedL(PlainQuadraticL):
            def control_bounds(self):
                return (-1.0, 1.0)

        L = BoundedL(2.0)
        argmax = L.conjugate_argmax(x=X, m=M, p=np.array([100.0]), t=T)
        np.testing.assert_allclose(argmax, [1.0], atol=1e-5)
        np.testing.assert_allclose(L.proximal(1.0, np.array([50.0])), [1.0], atol=1e-5)

    def test_declared_bounds_equal_to_the_fallback_are_still_honored(self):
        """The sharp case: A = [-10, 10] DECLARED coincides with the fallback box.

        The two are then indistinguishable by value -- only provenance separates
        them. A real control set whose optimum is on its own boundary is a valid
        active constraint and must return, so the guard has to key on
        ``control_bounds() is None``, not on the numbers. Pins
        ``_resolve_search_bounds``'s used_fallback flag and the guard's early
        return; without either, this legitimate model raises.
        """

        class DeclaredWideL(PlainQuadraticL):
            def control_bounds(self):
                return (-10.0, 10.0)

        L = DeclaredWideL(0.01)
        np.testing.assert_allclose(L.conjugate_argmax(x=X, m=M, p=np.array([100.0]), t=T), [10.0], atol=1e-4)
        np.testing.assert_allclose(L.proximal(1.0, np.array([50.0])), [10.0], atol=1e-4)

        # Same numbers, no declaration -> the box is a stand-in -> must raise.
        undeclared = PlainQuadraticL(0.01)
        with pytest.raises(ValueError, match=r"control_bounds\(\)"):
            undeclared.conjugate_argmax(x=X, m=M, p=np.array([100.0]), t=T)

    def test_interior_maximizer_is_unaffected(self):
        """The guard must not fire on the ordinary in-box case."""
        L = PlainQuadraticL(2.0)
        np.testing.assert_allclose(L.conjugate_argmax(x=X, m=M, p=np.array([2.0]), t=T), [1.0], atol=1e-6)

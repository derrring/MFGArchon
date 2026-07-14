"""Issue #1528 PR-1 -- owner byte-identity pinning test for the FP advective drift.

Single-owner refactor: the FP advective drift alpha* must flow through ONE owner primitive,
``H.optimal_control(x, m, p, t)`` on the problem's ``hamiltonian_class`` -- NOT the hand-coded
``-fp_drift_coefficient(problem) * grad(U)`` fork (the ``c = 1/lambda`` scalar the PR eliminates).

The owner DIVIDES: ``QuadraticControlCost.optimal_control`` returns ``-sign * p / lambda``
(``core/hamiltonian.py``). The legacy hand-coded path MULTIPLIES by ``fl(1/lambda)``. These two
forms are:

  * BYTE-IDENTICAL (0 ULP) for dyadic lambda (1.0, 0.5, 2.0) -- ``fl(1/lambda)`` is exact, so
    ``fl(1/lambda)*p == p/lambda``. The paper configs use ``control_cost=1.0`` => byte-identical.
  * SEPARATED by exactly <= 1 ULP for non-dyadic lambda (2.5, 0.7) -- ``fl(1/lambda)`` carries a
    rounding error, so multiply and divide disagree in the last bit (~4.4e-16 at lambda=2.5), a
    divergence ~12 orders of magnitude below any O(h^2) discretization scale.

This is the whole safety argument for routing the drift through the owner: for the paper regime the
switch is bit-for-bit invisible, and elsewhere it is sub-ULP. This test PINS the OWNER's byte-identity
contract -- ``H.optimal_control`` vs ``-p/lambda`` -- against the CURRENT owner. It does NOT exercise
the edited ``_face_velocity_from_potential`` / ``_build_advection`` call sites; their site-level
byte-identity was verified separately during review.

Discrimination (the test must FAIL for a wrong owner, not merely pass for the right one):
  * G-017 regression: substituting the legacy ``coupling_coefficient`` default (0.5) for ``1/lambda``
    -- a sense-blind scalar copy -- diverges grossly (>> ULP) wherever ``1/lambda != 0.5``. The test
    rejects it.
  * Multiply-vs-divide separation: the legacy multiply form ``-(fl(1/lambda))*p`` is proven to
    separate from the owner's ``-p/lambda`` at STRICTLY > 0 but <= 1 ULP for the non-dyadic lambda,
    so the byte-identity claim is discriminating, not vacuous.

On-disk validation: mfg-research/experiments/fp_drift_hamiltonian_routing/ (numerical confirmation
that dyadic lambda is byte-identical and non-dyadic diverges by exactly <= 1 ULP).
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import (
    OptimizationSense,
    QuadraticControlCost,
    SeparableHamiltonian,
)

# lambda values whose reciprocal is exactly representable in IEEE-754 binary64
# => fl(1/lambda) * p == p / lambda bit-for-bit (paper path uses control_cost=1.0).
DYADIC_LAMBDAS = [1.0, 0.5, 2.0]
# lambda values whose reciprocal carries a rounding error => multiply and divide
# forms separate at exactly <= 1 ULP.
NONDYADIC_LAMBDAS = [2.5, 0.7]

# Fixed pseudo-random momentum p = grad(U); seeded for reproducibility. Scaled to O(1) so the
# drift magnitude is representative, not degenerate near zero.
_P = np.random.default_rng(1528).standard_normal(64) * 3.0

# The owner primitive is a pure function of p for a SeparableHamiltonian; x, m, t are inert.
_X = np.array([0.5])
_M = 1.0
_T = 0.0


def _owner(lam: float) -> SeparableHamiltonian:
    """The single owner: problem.hamiltonian_class with a MINIMIZE quadratic control cost."""
    return SeparableHamiltonian(control_cost=QuadraticControlCost(sense=OptimizationSense.MINIMIZE, lambda_=lam))


@pytest.mark.parametrize("lam", DYADIC_LAMBDAS)
def test_owner_byte_identical_to_divide_form_dyadic(lam):
    """Dyadic lambda: the owner ``H.optimal_control`` equals ``-p/lambda`` bit-for-bit.

    This is the paper-path guarantee (control_cost=1.0 is in this set): routing the FP drift through
    the owner cannot perturb a single bit of the converged density.
    """
    H = _owner(lam)
    alpha = H.optimal_control(_X, _M, _P, _T)
    expected = -_P / lam
    assert alpha.shape == _P.shape, "owner must return alpha* with the SAME shape as p"
    assert np.array_equal(alpha, expected), (
        f"owner H.optimal_control diverged from -p/lambda at dyadic lambda={lam}; "
        "the paper-path byte-identity contract is broken"
    )


@pytest.mark.parametrize("lam", NONDYADIC_LAMBDAS)
def test_owner_within_one_ulp_of_divide_form_nondyadic(lam):
    """Non-dyadic lambda: the owner is within <= 1 ULP of ``-p/lambda``.

    The owner IS the divide form, so the realized gap is 0; the <= 1 ULP bound is the contract the
    per-site edits are checked against (a site that feeds a differently-rounded gradient could drift
    up to this bound and no further). 1 ULP ~ 4.4e-16 is ~12 orders below any O(h^2) scale.
    """
    H = _owner(lam)
    alpha = H.optimal_control(_X, _M, _P, _T)
    expected = -_P / lam
    diff = np.abs(alpha - expected)
    one_ulp = np.spacing(np.abs(expected))
    assert alpha.shape == _P.shape
    assert np.all(diff <= one_ulp), (
        f"owner exceeded 1 ULP from -p/lambda at non-dyadic lambda={lam}: "
        f"max|diff|={diff.max():.3e} > maxULP={one_ulp.max():.3e}"
    )


@pytest.mark.parametrize("lam", DYADIC_LAMBDAS)
def test_legacy_multiply_form_byte_identical_dyadic(lam):
    """The legacy hand-coded ``-(fl(1/lambda))*p`` matches the owner bit-for-bit at dyadic lambda.

    This is why the refactor is invisible on the paper path: the multiply form the sites currently
    hand-write and the owner's divide form are the SAME bits when 1/lambda is exact.
    """
    H = _owner(lam)
    alpha = H.optimal_control(_X, _M, _P, _T)
    legacy_multiply = -(1.0 / lam) * _P
    assert np.array_equal(alpha, legacy_multiply), (
        f"legacy multiply form and owner divide form must be byte-identical at dyadic lambda={lam}"
    )


@pytest.mark.parametrize("lam", NONDYADIC_LAMBDAS)
def test_legacy_multiply_form_separates_at_most_one_ulp_nondyadic(lam):
    """DISCRIMINATION: at non-dyadic lambda the legacy multiply form STRICTLY separates from the
    owner divide form, by > 0 but <= 1 ULP.

    Without a strictly-positive separation the byte-identity claim would be vacuous (the multiply and
    divide forms would be trivially equal everywhere and the test would prove nothing). This asserts
    the separation is real (last-bit disagreement) yet provably bounded by 1 ULP -- exactly the
    safety envelope the PR relies on for non-paper lambda.
    """
    H = _owner(lam)
    alpha = H.optimal_control(_X, _M, _P, _T)  # divide form, -p/lambda
    legacy_multiply = -(1.0 / lam) * _P  # multiply form, -(fl(1/lambda))*p
    diff = np.abs(alpha - legacy_multiply)
    one_ulp = np.spacing(np.abs(alpha))
    assert np.any(diff > 0.0), (
        f"expected the multiply form to disagree with the divide owner in the last bit at "
        f"non-dyadic lambda={lam}; a vacuous (zero-separation) case makes the pin non-discriminating"
    )
    assert np.all(diff <= one_ulp), (
        f"multiply-vs-divide separation exceeded 1 ULP at lambda={lam}: "
        f"max|diff|={diff.max():.3e} > maxULP={one_ulp.max():.3e}"
    )


@pytest.mark.parametrize("lam", DYADIC_LAMBDAS + NONDYADIC_LAMBDAS)
def test_g017_wrong_coefficient_is_rejected(lam):
    """DISCRIMINATION (G-017): a sense-blind ``coupling_coefficient`` copy (legacy default 0.5) used
    in place of ``1/lambda`` must be caught by this pin wherever ``1/lambda != 0.5``.

    G-017 is the regression where a second, hand-copied drift coefficient silently diverges from the
    Hamiltonian's ``1/lambda``. This asserts the owner is NOT the wrong-coefficient form -- the gap is
    gross (>> ULP), not sub-ULP -- so a reintroduced fork cannot pass unnoticed. lambda=2.0 is
    exempted because there ``1/lambda == 0.5`` and the wrong coefficient is accidentally correct.
    """
    wrong_coeff = 0.5  # legacy coupling_coefficient default
    if np.isclose(1.0 / lam, wrong_coeff):
        pytest.skip(f"at lambda={lam}, 1/lambda == {wrong_coeff}; the wrong coefficient coincides")
    H = _owner(lam)
    alpha = H.optimal_control(_X, _M, _P, _T)
    g017_form = -wrong_coeff * _P
    assert not np.array_equal(alpha, g017_form), (
        f"owner coincided with the G-017 wrong-coefficient form (0.5*p) at lambda={lam}; "
        "the pin fails to catch a divergent coupling_coefficient copy"
    )
    # And the divergence must be gross, not sub-ULP -- otherwise the discrimination is marginal.
    gross = np.max(np.abs(alpha - g017_form))
    one_ulp = np.max(np.spacing(np.abs(alpha)))
    assert gross > 1e3 * one_ulp, (
        f"G-017 divergence at lambda={lam} was only {gross:.3e} (~{gross / one_ulp:.1f} ULP); "
        "expected a gross error many orders above ULP"
    )

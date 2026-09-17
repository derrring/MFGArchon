"""`ControlCostBase` refuses a `sense` that is not an `OptimizationSense` (#2341).

`sense` is the first positional parameter, so a value meant for the control-cost weight lands there, and the sign
derivation reads anything that is not MINIMIZE as MAXIMIZE:

    self.sign = 1 if sense == OptimizationSense.MINIMIZE else -1

At `65bcf659` that made `L1ControlCost(1e-12)` a cost with `sense=1e-12`, `sign=-1` and `lambda` left at its 1.0
default. Nothing raised, `evaluate` returned finite values, and `optimal_control` returned the opposite sign, so the
caller got a plausible Hamiltonian with an inverted control. The library's own docstring taught the shape
(`SeparableHamiltonian(control_cost=QuadraticControlCost(1.0))` at `hamiltonian.py:1386`), and doctests do not run in
the gate, so nothing had ever executed it.

Oracle: the enum. There are exactly two valid senses and the sign is a function of which one, so a `sense` outside the
enum has no defined sign -- the old code did not fail to compute it, it computed the MAXIMIZE one.

Scope: validation, not a signature change. Making `sense` keyword-only would also close this, and would additionally
refuse `L1ControlCost(OptimizationSense.MAXIMIZE, 0.5)`, which is a legal call that means what it says. Of 370
ControlCost constructor calls in the tree (AST, not grep) exactly one passes a positional argument, and it is
`_MoreauYosidaControlCost(base, epsilon)`, whose signature is its own -- so the narrower fix costs nothing and the
wider one would remove something that works.
"""

from __future__ import annotations

import pytest

from mfgarchon.core.hamiltonian import L1ControlCost, OptimizationSense, QuadraticControlCost


@pytest.mark.parametrize("cost_class", [QuadraticControlCost, L1ControlCost])
def test_a_number_as_sense_is_refused_and_the_message_names_the_keyword(cost_class):
    """The failure this issue is about, and the message has to carry the fix: the caller wanted `lambda_`."""
    with pytest.raises(TypeError, match=r"sense must be an OptimizationSense.*number 1e-12.*lambda_=1e-12.*2341"):
        cost_class(1e-12)


def test_a_sense_outside_the_enum_is_refused_whatever_its_type():
    """A misspelling is the same defect as a number: it is not MINIMIZE, so the old code returned MAXIMIZE's sign."""
    with pytest.raises(TypeError, match=r"sense must be an OptimizationSense.*'minimise'.*type str.*2341"):
        L1ControlCost(sense="minimise")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("sense", "expected_sign"), [(OptimizationSense.MINIMIZE, 1), (OptimizationSense.MAXIMIZE, -1)]
)
def test_both_real_senses_still_set_their_own_sign(sense, expected_sign):
    """The control that the refusal did not just break the path it guards.

    Without this, a validation that rejected every `sense` would pass the two tests above. The signs are the
    convention itself (#1642): MINIMIZE gives ``alpha = -dH/dp``, MAXIMIZE ``+dH/dp``.
    """
    assert QuadraticControlCost(sense=sense, lambda_=2.0).sign == expected_sign
    assert L1ControlCost(sense=sense, lambda_=2.0).sign == expected_sign


def test_the_weight_reaches_lambda_when_passed_as_one():
    """The call the refusal's message recommends has to work, and to put the value where the caller meant it.

    At `65bcf659` the positional form left `lambda` at 1.0 -- so the number the caller passed was silently replaced
    by a default, not merely misrouted.
    """
    assert L1ControlCost(lambda_=1e-12).lambda_ == 1e-12
    assert QuadraticControlCost(lambda_=2.5).lambda_ == 2.5


def test_regularize_carries_the_base_sense_through():
    """`_MoreauYosidaControlCost` is the one in-tree caller that passes a positional argument, and it is unaffected:
    its signature is ``(base, epsilon)``, and the sense it forwards is the base's own enum member."""
    for sense, expected_sign in [(OptimizationSense.MINIMIZE, 1), (OptimizationSense.MAXIMIZE, -1)]:
        regularized = L1ControlCost(sense=sense, lambda_=0.5).regularize(0.1)
        assert regularized.sense is sense
        assert regularized.sign == expected_sign

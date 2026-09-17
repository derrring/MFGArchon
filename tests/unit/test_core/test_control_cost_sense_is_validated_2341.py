"""`ControlCostBase` refuses a `sense` that is not an `OptimizationSense` (#2341).

`sense` is the first positional parameter, so a value meant for the control-cost weight lands there, and the sign
derivation reads anything that is not MINIMIZE as MAXIMIZE:

    self.sign = 1 if sense == OptimizationSense.MINIMIZE else -1

At `65bcf659` that made `L1ControlCost(1e-12)` a cost with `sense=1e-12`, `sign=-1` and `lambda` left at its 1.0
default. Nothing raised, `evaluate` returned finite values, and `optimal_control` returned the opposite sign, so the
caller got a plausible Hamiltonian with an inverted control. The library's own docstring taught the shape -- `SeparableHamiltonian(control_cost=QuadraticControlCost(1.0))`, in
`MFGProblem`'s Hamiltonian example -- and doctests do not run in the gate, so nothing had ever executed it. (No line
number here on purpose: the fix's own diff moved that docstring, and the review of #2345 found both citations already
stale in the commit that wrote them.)

Oracle: the enum. There are exactly two valid senses and the sign is a function of which one, so a `sense` outside the
enum has no defined sign -- the old code did not fail to compute it, it computed the MAXIMIZE one.

Both constructors that derive the sign now route through `_sign_for_sense`, which is the highest owning
abstraction for this (AGENTS.md's rule for a cross-cutting defect): `MFGOperatorBase.__init__` carried the identical
expression with `sense` first positional and `finite_diff_eps`, another number, next -- so
`SeparableHamiltonian(cost, None, None, None, 0.5)` set the sense to 0.5. The review of #2345 found that sibling.

Scope: validation, not a signature change. Making `sense` keyword-only would also close this, and would additionally
refuse `L1ControlCost(OptimizationSense.MAXIMIZE, 0.5)`, which is a legal call that means what it says. Of 370
ControlCost constructor calls in the `*.py` tree (AST, not grep) exactly one passes a positional argument, and it is
`_MoreauYosidaControlCost(base, epsilon)`, whose signature is its own. The review of #2345 extended the census past
that population -- 18 occurrences across 657 non-`*.py` files, covering 5 notebooks, the quickstart, two CI workflows
and two JSON baselines, with 0 positional -- so the narrower fix costs nothing and the wider one would remove
something that works. What validation does NOT close is the arity hole: a third enum member would still need a sign,
which is why `_sign_for_sense` raises on one instead of defaulting.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import (
    L1ControlCost,
    OptimizationSense,
    QuadraticControlCost,
    SeparableHamiltonian,
)


@pytest.mark.parametrize("cost_class", [QuadraticControlCost, L1ControlCost])
def test_a_number_as_sense_is_refused_and_the_message_names_the_keyword(cost_class):
    """The failure this issue is about, and the message has to carry the fix: the caller wanted `lambda_`.

    The class name is in the pattern because without it the parametrization bought nothing: hardcoding the owner to
    `"ControlCostBase"` at the call site passed every test in this file, so every cost class could have named the
    wrong class in its own refusal (review of #2345, round 2).
    """
    with pytest.raises(
        TypeError,
        match=rf"{cost_class.__name__}: sense must be an OptimizationSense.*number 1e-12.*lambda_=1e-12.*2341",
    ):
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


@pytest.mark.parametrize(
    ("value", "expected_keyword"),
    [(1e-12, "lambda_"), (np.float32(0.5), "lambda_")],
)
def test_the_refusal_names_the_keyword_the_value_belonged_to(value, expected_keyword):
    """The hint has to be the keyword of the class that refused, and it must not be built as a callable suggestion.

    An earlier version suggested `{type(self).__name__}(lambda_=...)`, which is false for
    `_MoreauYosidaControlCost` -- its signature is `(base, epsilon)` and it takes no `lambda_`, so the refusal
    recommended a call that raises (review of #2345). `np.float32` is here because the earlier
    `isinstance(sense, (int, float))` predicate sent it, `np.int64`, `Decimal` and `Fraction` to the generic message.
    NOT `np.float64`, which is a `float` subclass and always took the helpful branch -- the set that moved is narrower
    than "every non-builtin number", which is what an earlier version of this docstring claimed.
    """
    with pytest.raises(TypeError, match=rf"sense must be an OptimizationSense.*number.*{expected_keyword}=.*2341"):
        L1ControlCost(value)


def test_the_operator_hierarchy_refuses_the_same_way_and_names_its_own_keyword():
    """`MFGOperatorBase` held the identical derivation, and `sense` is first positional there too (#2341).

    Its next parameter is `finite_diff_eps`, so the same slip lands the same way: at 09eaaa28
    `SeparableHamiltonian(cost, None, None, None, 0.5)` gave `sense=0.5` and `sense_sign=-1.0` with nothing raised.
    The keyword in the message is that class's own, not the cost hierarchy's.
    """
    cost = QuadraticControlCost(lambda_=1.0)
    with pytest.raises(TypeError, match=r"SeparableHamiltonian.*sense must be an OptimizationSense.*finite_diff_eps="):
        SeparableHamiltonian(cost, None, None, None, 0.5)


def test_a_bool_is_refused_by_the_generic_branch_and_not_offered_as_a_weight():
    """`True` is an `int` subclass, and the number branch is deliberately closed to it.

    If it were open, the refusal would advise `lambda_=True` -- and that call SUCCEEDS, storing `True` as a numeric
    weight, because the `lambda_ <= 0` guard accepts a bool. So the helpful message would hand the caller a second
    silent type confusion. The generic message is the right one here, and this pins the routing rather than the
    wording (review of #2345 found the exclusion unpinned).
    """
    with pytest.raises(TypeError, match=r"got True of type bool.*Valid values"):
        L1ControlCost(True)


def test_both_senses_keep_their_own_optimal_control():
    """The consequence the refusal exists to prevent, stated as the invariant it protects.

    At 09eaaa28 the broken construction returned MAXIMIZE's control bit-identically: on p = [-1, 2],
    `QuadraticControlCost(1.0)` gave [-1, 2] where the intended MINIMIZE call gives [1, -2] (#1642's convention).
    """
    p = np.array([-1.0, 2.0])
    minimise = QuadraticControlCost(lambda_=1.0).optimal_control(p)
    maximise = QuadraticControlCost(sense=OptimizationSense.MAXIMIZE, lambda_=1.0).optimal_control(p)
    np.testing.assert_allclose(minimise, [1.0, -2.0])
    np.testing.assert_allclose(maximise, -np.asarray(minimise))


def _sense_with_no_convention() -> OptimizationSense:
    """An `OptimizationSense` that is neither member, which is how the third-member branch is reachable today.

    `object.__new__` bypasses `EnumMeta.__call__`, so this satisfies `isinstance(x, OptimizationSense)` while
    `x not in list(OptimizationSense)`. Subclassing is not an alternative -- Python refuses to extend an enum that
    has members. `_name_` is set here because the branch's own message reads it; the case where it is absent is the
    second assertion below, and it was a real defect: `.name` raised AttributeError and the refusal died on its own
    edge case (review of #2345, round 2).
    """
    sense = object.__new__(OptimizationSense)
    sense._name_ = "SADDLE"
    sense._value_ = "saddle"
    return sense


def test_a_sense_with_no_sign_convention_is_refused_rather_than_given_maximize_s():
    """The arity hole, which validation alone does not close (#2341).

    At `13cc92ee` both sites returned -1 for such a value -- MAXIMIZE's sign, silently, because the derivation was
    `1 if sense == MINIMIZE else -1`. I had declared this branch unreachable and therefore unpinnable; the review of
    #2345 showed the construction above reaches it through both constructors, so it is pinned rather than disclosed.
    """
    saddle = _sense_with_no_convention()
    with pytest.raises(NotImplementedError, match=r"L1ControlCost: no sign convention is defined for SADDLE.*2341"):
        L1ControlCost(saddle)
    with pytest.raises(NotImplementedError, match=r"SeparableHamiltonian: no sign convention is defined for SADDLE"):
        SeparableHamiltonian(QuadraticControlCost(lambda_=1.0), sense=saddle)


def test_the_refusal_survives_a_sense_that_cannot_even_be_named():
    """The message interpolated `sense.name`, which does not exist on an instance built by `object.__new__`.

    So the branch raised `AttributeError` from inside its own diagnostic instead of the refusal it exists to deliver
    -- the verdict was red either way, which is exactly why it needs its own assertion rather than a count.
    """
    nameless = object.__new__(OptimizationSense)
    with pytest.raises(NotImplementedError, match=r"no sign convention is defined for.*2341"):
        L1ControlCost(nameless)

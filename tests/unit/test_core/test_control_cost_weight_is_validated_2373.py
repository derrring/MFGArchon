"""The control cost's weight is validated, because removing `sense` un-blocked a defect. #2373

REPLACES `test_control_cost_sense_is_validated_2341.py`, which was deleted with the parameter it
pinned. That file's subject was `sense` validation; with the parameter gone there is nothing left
to raise and the file could not be repaired into a passing state. Three of its guarantees were
covered nowhere else in the suite, and they are carried here.

THE INVERSION THIS FILE EXISTS FOR. `sense` used to be the FIRST positional parameter of
`ControlCostBase.__init__`, immediately ahead of the weight. That made `L1ControlCost(1e-12)` a
silent sense-flip -- the defect #2341 recorded -- and it also meant the sense validator was
incidentally guarding the slot. #2373 removed the parameter, which fixes #2341 at the root; but it
also hands the slot straight to `control_cost`, where the only check was `lam <= 0`. Measured on
the removal branch BEFORE this guard existed:

    L1ControlCost(True).lambda_  ->  True, of type bool

because `bool` subclasses `int` and `True <= 0` is False. The old file predicted exactly this in
the docstring of `test_a_bool_is_refused_by_the_generic_branch_and_not_offered_as_a_weight`: it
declined to advise `lambda_=True` because "that call SUCCEEDS, storing `True` as a numeric weight".
Under #2341 that path was unreachable, since `True` hit the sense validator first. Under #2373 it
is reachable, so the prediction became live and the guard is the compensation.

The other direction moved too, and the right way: `L1ControlCost(1e-12)` was REFUSED before, as a
malformed sense. A tiny positive weight is legitimate and is now accepted.

SCOPE IS BOTH INHERITING SLOTS, not just the weight the filename names. `sense` was
first-positional on `MFGOperatorBase` too, so `population_index` inherited a slot the same way and
its guard is pinned here rather than in a second file.

EVERY GUARD IN THIS FILE IS MUTATION-VERIFIED, and that is not decoration -- three of them were
pinned by NOTHING until the mutations were run, and all three had passed review. The matrix, each
mutation killing exactly one test and the unmutated file at 10 passed:

    population_index: numbers.Integral -> int        1 failed
    population_index: `bool` arm deleted             1 failed
    weight:           numbers.Real -> (int, float)   1 failed
    weight:           `bool` arm deleted             1 failed
    weight:           math.isfinite disabled         1 failed
    weight:           positivity check disabled      1 failed

Why reading could not substitute: every in-tree caller passes a literal, so each guard's whole
population is inputs nobody in this repository produces. A guard like that is green under its own
deletion, and the suite reports it as covered.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import L1ControlCost, QuadraticControlCost, SeparableHamiltonian


def test_a_bool_is_refused_as_a_weight():
    """THE #2373 REGRESSION GUARD. `True` is an `int` subclass, so `lam <= 0` accepts it.

    Without this, `L1ControlCost(True)` stores `True` as the weight and every downstream division
    by `lambda_` silently uses 1. The failure is a well-formed object, not an exception, which is
    why the assertion is on the refusal rather than on any computed value.
    """
    with pytest.raises(TypeError, match=r"real number.*True.*bool"):
        L1ControlCost(True)
    with pytest.raises(TypeError, match=r"real number.*True.*bool"):
        QuadraticControlCost(lambda_=True)


def test_a_non_number_is_refused_with_a_message_that_names_the_parameter():
    """Before the guard this raised from the comparison itself -- ``'<=' not supported between
    instances of 'str' and 'int'`` -- which names neither the class nor the parameter. The
    diagnostic is part of the contract here, so the match is on the wording."""
    with pytest.raises(TypeError, match=r"L1ControlCost.*real number.*str"):
        L1ControlCost("abc")


def test_a_tiny_positive_weight_survives_and_reaches_lambda():
    """Carried from the deleted file, where it was the only pin on a legitimately tiny weight.

    Every other `lambda_` assertion in the suite is order-unity (1.0, 2.0, 2.5, 3.0), and the
    negative direction is covered elsewhere by a `ValueError` on a non-positive weight. This is
    the tiny-but-valid direction, and under #2341 it was REFUSED -- `1e-12` in the first slot was
    read as a malformed sense.
    """
    assert L1ControlCost(lambda_=1e-12).lambda_ == 1e-12
    assert QuadraticControlCost(lambda_=2.5).lambda_ == 2.5
    assert L1ControlCost(1e-12).lambda_ == 1e-12  # positionally, the call #2341 could not make


def test_a_non_positive_weight_is_still_refused():
    """The pre-existing guard, pinned here because the new type check runs BEFORE it and could
    mask it -- a reordering that swallows this would otherwise be invisible."""
    with pytest.raises(ValueError, match=r"must be positive"):
        L1ControlCost(0.0)
    with pytest.raises(ValueError, match=r"must be positive"):
        QuadraticControlCost(lambda_=-1.0)


def test_the_optimal_control_convention_is_minus_dh_dp():
    """EXTERNAL ORACLE, carried from the deleted file. #1642's convention, analytic.

    `alpha* = -p/lambda` for a quadratic cost. Kept because `test_hamiltonian_classes.py` was a
    collection error while #2373 was in flight, and during that window no collectable test in the
    suite exercised `ControlCostBase.optimal_control` at all.
    """
    p = np.array([-1.0, 2.0])
    np.testing.assert_allclose(QuadraticControlCost(lambda_=1.0).optimal_control(p), [1.0, -2.0])
    np.testing.assert_allclose(QuadraticControlCost(lambda_=2.0).optimal_control(p), [0.5, -1.0])


def test_a_non_finite_weight_is_refused():
    """PINS `math.isfinite`, which nothing pinned -- the third unpinned guard in this file, all
    three found by mutating rather than reading.

    `inf` and `nan` are both `numbers.Real`, and `inf <= 0` and `nan <= 0` are both False, so a
    non-finite weight passes the type arm AND the positivity arm. Measured: `L1ControlCost(inf)`
    and `L1ControlCost(nan)` each gave `alpha* = [0, 0, 0]` -- a silent no-control solve with no
    nan anywhere downstream to notice it.

    The check must run BEFORE the sign test, so `ValueError` is asserted with a message distinct
    from the positivity one: replacing this guard with a pass-through left the file green.
    """
    for bad in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError, match=r"must be finite"):
            L1ControlCost(bad)
        with pytest.raises(ValueError, match=r"must be finite"):
            QuadraticControlCost(lambda_=bad)
    # numpy's non-finites take the same path -- they are `numbers.Real` too
    with pytest.raises(ValueError, match=r"must be finite"):
        L1ControlCost(np.float64("inf"))


def test_a_numpy_float_is_accepted_as_a_weight():
    """PINS THE WEIGHT GUARD'S ABC, which nothing pinned -- found by mutating the guard rather
    than reading it.

    `bd124952`'s commit message argues that narrowing to `(int, float)` "would have satisfied mypy
    too and silently rejected `np.float32`". Measured before this test existed: making exactly that
    narrowing left this file at **8 passed**, green. The argument was recorded in prose, and the
    edit it argues against cost nothing that reddened.

    `np.float32`, `np.int64` and `Fraction` are all `numbers.Real` and none is an `(int, float)`.
    They reach `lambda_` as 2.0, 2.0 and 0.5 today, so the narrowing would be a silent refusal of
    input the library accepts -- and `np.float32` is what a caller gets from a single-precision
    array without asking for it.
    """
    assert L1ControlCost(np.float32(2.0)).lambda_ == 2.0
    assert L1ControlCost(np.int64(2)).lambda_ == 2.0
    assert QuadraticControlCost(lambda_=Fraction(1, 2)).lambda_ == 0.5
    # NOT TYPE-TIDINESS: this assert is the ONLY pin on `float(lam)`. Measured before it existed,
    # deleting the coercion left both the suite AND the mypy ratchet green (1300, no package
    # moved), so nothing anywhere objected. Simplifying it away unpins the coercion silently.
    assert type(L1ControlCost(np.float32(2.0)).lambda_) is float, "stored unnormalised"


def test_a_numpy_integer_is_accepted_as_a_population_index():
    """PINS THE ABC. Measured: with `isinstance(population_index, int)` instead of
    `numbers.Integral`, the whole suite stays green -- 489 passed, byte-identical to unmutated --
    because every caller in the tree passes a literal `0`, `1`, `2` or `k`. So the guard's entire
    population is inputs nobody in-tree produces, and reverting it costs nothing that reddens.

    `isinstance(np.int64(1), int)` is False. Before the guard existed this slot took anything; a
    bare `int` check would newly REFUSE what the library used to accept, which is a narrowing
    dressed as a validation.
    """
    c = QuadraticControlCost(control_cost=1.0)
    for v in (np.int64(0), np.int32(2), np.uint8(1), np.int8(3)):
        h = SeparableHamiltonian(control_cost=c, population_index=v)
        assert h.population_index == int(v)
        # As with `float(lam)`: this assert is the ONLY pin on `int(population_index)`, and
        # deleting that coercion was green on both the suite and the mypy ratchet before it.
        assert type(h.population_index) is int, f"{type(v).__name__} was stored unnormalised"


def test_a_bool_is_refused_as_a_population_index():
    """PINS THE `bool` ARM, which is the one that looks redundant and is not.

    `isinstance(True, numbers.Integral)` is True -- every ABC in the numeric tower admits bools --
    so the ABC beside it does NOT refuse `True` and this arm is the only thing that does. Measured:
    deleting the arm leaves the suite at 489 passed, unchanged. It is the exact edit a reader makes
    on seeing two isinstance checks that appear to cover the same types.

    The mirror case is why both arms stay: `np.bool_(True)` is neither a `bool` nor an `Integral`,
    so the ARM ABOVE cannot see it and only the ABC refuses it. Each arm covers what the other
    misses; neither alone closes the parameter.
    """
    c = QuadraticControlCost(control_cost=1.0)
    with pytest.raises(TypeError, match=r"population_index must be an integer"):
        SeparableHamiltonian(control_cost=c, population_index=True)
    with pytest.raises(TypeError, match=r"population_index must be an integer"):
        SeparableHamiltonian(control_cost=c, population_index=np.bool_(True))


def test_a_negative_population_index_is_refused():
    """PINS THE VALUE ARM, which my own six-mutation matrix missed by enumerating the type
    lattice and not the guard's statements.

    Each guard has three arms -- type, bool, value. The weight's value arm (positivity) was
    already pinned; this one was not: `if population_index < 0` -> `if False` left the file at
    10 passed while the sibling's `lam <= 0` reddened. The asymmetry between siblings is the
    tell, and it is the same shape as the `int`-vs-`Integral` asymmetry two rounds earlier.

    It is not cosmetic: `_extract_own_density` slices `m[k*N:(k+1)*N]`, so `k = -1` gives
    `m[-N:0]` -- an empty array, not an error.
    """
    c = QuadraticControlCost(control_cost=1.0)
    with pytest.raises(ValueError, match=r"population_index must be non-negative"):
        SeparableHamiltonian(control_cost=c, population_index=-1)
    with pytest.raises(ValueError, match=r"population_index must be non-negative"):
        SeparableHamiltonian(control_cost=c, population_index=np.int64(-2))


def test_sense_is_no_longer_accepted_by_either_owner():
    """#2373's own artifact, as a test: there is no replacement keyword and no migration target,
    so the refusal is Python's own `TypeError` on an unexpected keyword rather than a deprecation."""
    with pytest.raises(TypeError, match=r"unexpected keyword argument 'sense'"):
        QuadraticControlCost(sense="minimize", control_cost=1.0)
    with pytest.raises(TypeError, match=r"unexpected keyword argument 'sense'"):
        L1ControlCost(sense="minimize", lambda_=1.0)
    # "EITHER OWNER" MEANS BOTH BASE CLASSES. The two above are `ControlCostBase` subclasses;
    # in #2341's vocabulary the second owner is `MFGOperatorBase`, and nothing in the suite
    # passed `sense=` to one. The name claimed a universal the assertions did not reach.
    with pytest.raises(TypeError, match=r"unexpected keyword argument 'sense'"):
        SeparableHamiltonian(QuadraticControlCost(lambda_=1.0), sense="minimize")

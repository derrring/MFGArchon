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
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import L1ControlCost, QuadraticControlCost


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


def test_sense_is_no_longer_accepted_by_either_owner():
    """#2373's own artifact, as a test: there is no replacement keyword and no migration target,
    so the refusal is Python's own `TypeError` on an unexpected keyword rather than a deprecation."""
    with pytest.raises(TypeError, match=r"unexpected keyword argument 'sense'"):
        QuadraticControlCost(sense="minimize", control_cost=1.0)
    with pytest.raises(TypeError, match=r"unexpected keyword argument 'sense'"):
        L1ControlCost(sense="minimize", lambda_=1.0)

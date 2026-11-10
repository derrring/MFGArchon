"""The one owner of an inner iteration that stopped without converging lives in ``utils/``, beside ``ConvergenceError``
(#2644 B2, stage 1), so ``core/`` and ``geometry/`` can reach it: ``mfgarchon.utils.exceptions.report_inner_failure``.

It decides by policy and the failure carries its own message: "raise", the default, raises a ``ConvergenceError`` with
that message; "record" appends the failure to its caller's record and warns once. B1's HJB wrapper,
``base_hjb.report_inner_solve_failure``, is one caller of it: its failure is raised by the owner, and its warning still
points where it pointed. The owner is reached through its module inside each test, so a tree without it fails each
test by name.
"""

from __future__ import annotations

import math
import traceback
import warnings

import pytest

from mfgarchon.utils import exceptions


def test_the_owner_raises_by_default_with_the_failures_own_message():
    with pytest.raises(exceptions.ConvergenceError) as raised:
        exceptions.report_inner_failure(
            object(), "AMember stopped.", solver="AMember", steps=3, residual=0.5, on_raise=" Raise hint."
        )
    error = raised.value
    assert str(error) == "AMember stopped. Raise hint."
    assert (error.solver_type, error.iterations, error.residual) == ("AMember", 3, 0.5)
    assert isinstance(error, RuntimeError)


def _member(record):
    exceptions.report_inner_failure(  # the line the warning names
        "the failure",
        "AMember stopped.",
        solver="AMember",
        steps=3,
        residual=math.nan,
        policy="record",
        record=record,
        on_record=" Record hint.",
        stacklevel=1,
    )


def test_the_owner_records_and_warns_once_where_its_stacklevel_says():
    record: list = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _member(record)
    assert record == ["the failure"]
    assert [(w.category, str(w.message)) for w in caught] == [(RuntimeWarning, "AMember stopped. Record hint.")]
    assert caught[0].filename == __file__
    assert caught[0].lineno == _member.__code__.co_firstlineno + 1


def test_the_owner_refuses_a_policy_it_does_not_know_and_a_record_never_begun():
    with pytest.raises(ValueError, match="it is 'raise' or 'record'"):
        exceptions.report_inner_failure(object(), "m", solver="AMember", steps=1, residual=1.0, policy="warn")
    with pytest.raises(RuntimeError, match="began none"):
        exceptions.report_inner_failure(object(), "m", solver="AMember", steps=1, residual=1.0, policy="record")


def _hjb_failure():
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import InnerSolveFailure

    return InnerSolveFailure(t_idx=7, residual=2.0, tolerance=1e-6, steps=4, reason="budget exhausted", measure="m")


def test_the_hjb_wrapper_hands_its_failure_to_the_owner():
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import report_inner_solve_failure

    with pytest.raises(exceptions.ConvergenceError) as raised:
        report_inner_solve_failure(_hjb_failure(), policy="raise", record=None, solver="AMember")
    frames = [(frame.name, frame.filename) for frame in traceback.extract_tb(raised.value.__traceback__)]
    assert frames[-1] == ("report_inner_failure", exceptions.__file__), frames
    assert str(raised.value) == (
        "AMember did not converge at t_idx=7: budget exhausted after 4 step(s), residual 2.000e+00 against a tolerance "
        "of 1.000e-06 (m). The iterate for this time step is not a root of the discrete HJB (#2644). To keep the "
        "iterate, set on_inner_failure='record' and read inner_solve_failures()."
    )


def _hjb_member(record, policy):
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import report_inner_solve_failure

    report_inner_solve_failure(_hjb_failure(), policy=policy, record=record, solver="AMember")


def _hjb_caller(record, policy):
    _hjb_member(record, policy)  # the line the warning names: the member's caller


@pytest.mark.parametrize("policy", ["record", "fallback"])
def test_the_hjb_wrapper_records_and_its_warning_names_the_members_caller(policy):
    record: list = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _hjb_caller(record, policy)
    assert record == [_hjb_failure()]
    assert [w.category for w in caught] == [RuntimeWarning]
    assert str(caught[0].message).endswith(
        "A coupled solve reads inner_solve_failures() and does not report convergence over it (#1878)."
    )
    assert (caught[0].filename, caught[0].lineno) == (__file__, _hjb_caller.__code__.co_firstlineno + 1)

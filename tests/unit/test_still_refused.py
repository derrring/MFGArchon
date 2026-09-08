"""The `still_refused` fixture's contract, which five defect pins depend on (#2288).

The fixture exists because a class-3 pin fires on an ABSENCE -- the guard stopped raising -- and an
absence has more causes than the fix the pin was written to demand. Three times in one session a
retirement message named a cause its own pin did not observe; each was caught by an independent
reviewer and none by the author, and the second was written inside the commit that fixed the first.
Once, replacing a helper's body with `return None` produced a byte-identical message declaring that
a different, unrelated fix had landed.

Free-form text cannot be checked, so the fixture stopped accepting it: the caller states what was
observed and enumerates the causes, and a single-cause message is refused unless something excludes
the others. This file pins that refusal and the premise's placement, because both are silent when
broken -- a pin whose guarantee has been removed looks exactly like one that still has it.

ADMISSION (#2288). Class 1, on the mutations recorded per test. This is the shape
`tests/unit/test_mfg_caplog.py` uses for the sibling fixture in the same conftest.

WHAT THIS FILE DOES NOT PIN, because nothing mechanical can: that the enumerated causes are
COMPLETE, and that an `excluded=` sentence is true. Both are free text and a caller can be wrong in
them. What the fixture buys is that the single-cause message -- the shape all three incidents took
-- becomes an act of commission rather than of omission.
"""

from __future__ import annotations

import contextlib

import pytest

_OBSERVED = "the guard did not refuse."
_TWO_CAUSES = {"the capability landed": "delete this pin", "the guard moved": "re-point this pin"}
_ONE_CAUSE = {"the capability landed": "delete this pin"}


def test_a_single_cause_message_is_refused(still_refused):
    """The load-bearing rule. Without it the fixture is the old free-form one wearing a new shape.

    Mutation, measured for #2288: deleting the `len(causes) < 2 and premise is None and excluded is
    None` branch from `tests/conftest.py` kills this test and only this one across the fixture's
    five call sites and this file.
    """
    with (
        pytest.raises(ValueError, match="single-cause retirement message is refused"),
        still_refused("x", observed=_OBSERVED, causes=_ONE_CAUSE),
    ):
        raise NotImplementedError("x")


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"causes": _TWO_CAUSES}, "two causes stand on their own"),
        ({"causes": _ONE_CAUSE, "premise": lambda: None}, "a premise excludes the others"),
        ({"causes": _ONE_CAUSE, "excluded": "the assertion above rules it out"}, "excluded does too"),
    ],
)
def test_the_three_admissible_shapes_are_accepted(still_refused, kwargs, why):
    """The other half of the refusal above: without it, a fixture refusing everything would pass."""
    with still_refused("x", observed=_OBSERVED, **kwargs):
        raise NotImplementedError("x")


def test_an_empty_observation_or_no_cause_is_refused(still_refused):
    """Both are the message rendering itself into nothing."""
    for bad in ({"observed": "", "causes": _TWO_CAUSES}, {"observed": _OBSERVED, "causes": {}}):
        with pytest.raises(ValueError), still_refused("x", **bad):
            raise NotImplementedError("x")


def _raise_it():
    raise NotImplementedError("x")


def _do_nothing():
    return None


@pytest.mark.parametrize(("body", "path"), [(_raise_it, "raising"), (_do_nothing, "non-raising")])
def test_the_premise_runs_on_the_raising_and_the_non_raising_path(still_refused, body, path):
    """Incident #1's actual mechanism, and the reason `premise` is a parameter rather than a
    convention.

    A check placed on the line AFTER the `with` does not run when the pin retires, because
    `pytest.fail` propagates out of the context manager -- so the message cites a check that did not
    execute. The fixture runs it in a `finally` inside the block instead.

    Mutation, measured for #2288: running `premise()` only inside the `except exc_type` branch --
    so it fires when the guard refused and not when it did not, which is incident #1's shape --
    kills this test's non-raising parametrization and
    `test_a_failing_premise_reports_itself_rather_than_the_retirement`. Control 35 passed across
    this file and the fixture's four call-site files; mutated, 2 failed.

    A weaker mutation kills nothing and is recorded because it was the one first claimed here:
    moving `premise()` to after the `except` still runs it on both paths, so the property survives.
    The discriminating edit is not where the call sits but which paths reach it.
    """
    ran = []
    ctx = still_refused("x", observed=_OBSERVED, causes=_ONE_CAUSE, premise=lambda: ran.append(1))
    with contextlib.suppress(BaseException), ctx:
        body()
    assert ran, f"the premise did not run on the {path} path"


def _failing_premise():
    raise AssertionError("premise failed")


def test_a_failing_premise_reports_itself_rather_than_the_retirement(still_refused):
    """The premise's AssertionError must pass through, not be swallowed into a false retirement.

    `test_gpu_particle_refuses_absorbing_bc_1910.py` depends on exactly this: its premise asserts
    the GPU strategy was selected, and a routing change must say so rather than print "the
    capability landed". A broad `exc_type` would defeat it, which is why that knob is documented as
    load-bearing on the fixture.
    """
    with (
        pytest.raises(AssertionError, match="premise failed"),
        still_refused(
            "x",
            observed=_OBSERVED,
            causes=_ONE_CAUSE,
            premise=_failing_premise,
        ),
    ):
        pass


def test_the_rendered_message_carries_every_cause_and_its_instruction(still_refused):
    """A reader acts on this text. If a cause is enumerated but not rendered, it is not there."""
    with pytest.raises(BaseException) as exc, still_refused("x", observed=_OBSERVED, causes=_TWO_CAUSES):
        pass
    message = str(exc.value)
    assert _OBSERVED in message
    for cause, instruction in _TWO_CAUSES.items():
        assert cause in message, f"cause {cause!r} was enumerated but not rendered"
        assert instruction in message, f"instruction for {cause!r} was not rendered"


def test_a_refusal_for_the_wrong_reason_is_not_a_pass(still_refused):
    """`match` still discriminates: the guard must refuse for the pinned reason, not merely refuse."""
    with (
        pytest.raises(AssertionError, match="not for the pinned reason"),
        still_refused("the pinned substring", observed=_OBSERVED, causes=_TWO_CAUSES),
    ):
        raise NotImplementedError("some other refusal entirely")

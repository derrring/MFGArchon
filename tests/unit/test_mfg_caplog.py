"""The `mfg_caplog` fixture, and the thing it is independent of (Issue #2083).

`MFGLogger._setup_logger` sets `propagate = False` on every logger it configures, so an mfgarchon
record never reaches the root logger. What `caplog` does about that is **version-dependent**, which
is why six test modules each grew their own record-collecting handler and why four tests written
with plain `caplog` were red:

- pytest 8.4.1 (`uv run --extra dev`): `catching_logs.__enter__` attaches its handler to the root
  logger only, so `caplog` sees no mfgarchon record at all, whatever the creation site. On this
  version `propagate = False` is the whole cause.
- pytest 9.1.1 (the gate interpreter): `__enter__` also attaches to every non-propagating logger
  **that already exists** when it runs. That sweep runs once per test PHASE, so the discriminator
  is not module-import vs function-local -- a logger created in a fixture is visible in the test
  body -- it is whether the logger existed before this phase's sweep. One born mid-solve did not,
  which is the `fp_gfdm.py:575` case, so whether a test passes depends on whether an earlier test
  in the same worker happened to create the logger first.

`mfg_caplog` attaches to the emitting logger itself, so it depends on neither the pytest version
nor the order tests run in. These tests pin that, and the `assert logger.propagate is False`
below pins the premise: if propagation is ever turned on, this file says so.
"""

from __future__ import annotations

import logging

import pytest

from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.mfg_logging.logger import MFGLogger

# A logger of our own, obtained the way production obtains one: MFGLogger configures it, so the
# propagate=False under test is set by the code under test and not by this file.
PROBE = "mfgarchon.tests.mfg_caplog_probe"
PROBE_PREFIX = "mfgarchon.tests."


@pytest.fixture(autouse=True)
def _forget_the_probe_loggers():
    """Leave no probe logger behind.

    `MFGLogger.get_logger` caches for the life of the process and `logging` never forgets a
    name, so without this the file is not re-runnable in one interpreter: the born-late test
    below asserts its logger does NOT exist yet, and on a second run it would find its own
    leftover. It also keeps pytest 9's per-phase sweep over `loggerDict` from paying for two
    phantom loggers in every later test of the worker.
    """
    yield
    for name in [n for n in list(MFGLogger._loggers) if n.startswith(PROBE_PREFIX)]:
        MFGLogger._loggers.pop(name, None)
    for name in [n for n in list(logging.Logger.manager.loggerDict) if n.startswith(PROBE_PREFIX)]:
        logging.Logger.manager.loggerDict.pop(name, None)


def test_it_captures_a_record_from_a_non_propagating_logger(mfg_caplog):
    logger = get_logger(PROBE)
    assert logger.propagate is False, (
        "MFGLogger no longer disables propagation -- the premise of this fixture has changed"
    )

    with mfg_caplog.at_level(logging.WARNING, logger=PROBE):
        logger.warning("the drift was reported")

    assert mfg_caplog.messages == ["the drift was reported"]


def test_it_sees_a_logger_born_after_pytest_swept_for_them(mfg_caplog, caplog):
    """The case that made #2083 look like a plain red, measured on its own shape.

    pytest attaches its capture handler to the non-propagating loggers that exist when the test
    phase begins. This logger is created inside the test, after that sweep, exactly as
    `fp_gfdm.py:575` creates its own inside a solve -- so `caplog` cannot see it on either
    pytest version, while `mfg_caplog` attaches on demand and does.
    """
    born_late = "mfgarchon.tests.mfg_caplog_born_late"
    assert born_late not in logging.Logger.manager.loggerDict, (
        "this logger must not exist yet -- the point of the test is that it is created below"
    )

    logger = get_logger(born_late)
    with mfg_caplog.at_level(logging.WARNING, logger=born_late):
        logger.warning("emitted through a logger pytest never swept")

    assert mfg_caplog.messages == ["emitted through a logger pytest never swept"]
    assert caplog.records == [], (
        "caplog saw a logger created after its handler sweep; the mechanism behind #2083 has changed"
    )


def test_the_capture_does_not_depend_on_the_logger_already_existing(mfg_caplog):
    """The other half: a logger that DOES exist beforehand is captured the same way, so the
    fixture's behaviour does not turn on creation order the way plain `caplog` does.

    The precondition is asserted rather than assumed. Stated as "the test above created it",
    it was satisfied by `get_logger` creating it here instead, which made this a duplicate of
    the first test whenever it ran alone.
    """
    logger = get_logger(PROBE)
    assert PROBE in MFGLogger._loggers, "precondition: the logger must exist before at_level"

    with mfg_caplog.at_level(logging.WARNING, logger=PROBE):
        logger.warning("same capture, pre-existing logger")

    assert mfg_caplog.messages == ["same capture, pre-existing logger"]


def test_the_logger_is_the_one_the_package_hands_out(mfg_caplog):
    """`at_level` must resolve the name through mfgarchon's `get_logger`, not `logging`'s.

    Swapping it for `logging.getLogger` leaves every other test in this file green -- the
    records still arrive, because by then something else has usually configured the logger.
    What breaks is the guarantee in the docstring: that the captured object carries the
    configuration production gives it. This is the test that fails on that swap.
    """
    fresh = "mfgarchon.utils.cli"  # a real module, so the static name check accepts it
    MFGLogger._loggers.pop(fresh, None)

    with mfg_caplog.at_level(logging.WARNING, logger=fresh):
        logging.getLogger(fresh).warning("configured by the package, not by logging")

    assert mfg_caplog.messages == ["configured by the package, not by logging"]
    # `logging.getLogger` never touches this registry, whatever the logger's prior state, so it
    # is the one assertion that separates the two calls. Note it is the ONLY one available: the
    # logger's `propagate`/handlers cannot be asserted here, because `at_level` deliberately
    # leaves an in-package logger configured (see its `finally`).
    assert fresh in MFGLogger._loggers, "at_level did not go through mfgarchon's get_logger"


def test_it_captures_at_and_above_the_level_it_was_given(mfg_caplog):
    logger = get_logger(PROBE)

    with mfg_caplog.at_level(logging.WARNING, logger=PROBE):
        logger.info("below the level")
        logger.warning("at the level")
        logger.error("above the level")

    assert mfg_caplog.messages == ["at the level", "above the level"]


def test_a_lower_level_admits_what_the_higher_one_filtered(mfg_caplog):
    """The control for the test above: `below the level` is absent because of the level, not
    because a DEBUG-range record cannot be captured at all."""
    logger = get_logger(PROBE)

    with mfg_caplog.at_level(logging.DEBUG, logger=PROBE):
        logger.debug("below the level")

    assert mfg_caplog.messages == ["below the level"]


def test_messages_are_formatted_with_their_args(mfg_caplog):
    """`.records` keeps the record, so a test can pin the formatting args (the CFL diagnostic
    does exactly that); `.messages` is the formatted form."""
    logger = get_logger(PROBE)

    with mfg_caplog.at_level(logging.WARNING, logger=PROBE):
        logger.warning("drift %.2f at t=%d", 1.5, 3)

    assert mfg_caplog.messages == ["drift 1.50 at t=3"]
    assert mfg_caplog.records[0].args == (1.5, 3)


def test_records_accumulate_across_blocks_until_cleared(mfg_caplog):
    logger = get_logger(PROBE)

    with mfg_caplog.at_level(logging.WARNING, logger=PROBE):
        logger.warning("first")
    with mfg_caplog.at_level(logging.WARNING, logger=PROBE):
        logger.warning("second")
    assert mfg_caplog.messages == ["first", "second"]

    mfg_caplog.clear()
    assert mfg_caplog.records == []


def test_the_handler_and_the_level_are_restored_on_exit(mfg_caplog):
    """A leaked handler would capture the next test's records, and a leaked level would change
    what the next test's solver prints -- both silently."""
    logger = get_logger(PROBE)
    level_before, handlers_before = logger.level, list(logger.handlers)

    with mfg_caplog.at_level(logging.DEBUG, logger=PROBE):
        logger.debug("inside the block")
    logger.warning("after the block")

    assert logger.level == level_before
    assert logger.handlers == handlers_before
    assert mfg_caplog.messages == ["inside the block"]


def test_it_restores_even_when_the_body_raises(mfg_caplog):
    logger = get_logger(PROBE)
    level_before, handlers_before = logger.level, list(logger.handlers)

    with pytest.raises(RuntimeError, match="boom"), mfg_caplog.at_level(logging.DEBUG, logger=PROBE):
        raise RuntimeError("boom")

    assert logger.level == level_before
    assert logger.handlers == handlers_before


def test_a_wrong_logger_name_captures_nothing_and_is_NOT_caught(mfg_caplog):
    """The known hole, pinned so that it is recorded rather than rediscovered.

    Two guards against this were built and both removed: each re-created the order-dependence
    the fixture exists to remove (see `at_level`'s docstring). This test states the residual
    limitation as a fact, so a reader meets it here rather than in a vacuous absence assertion.

    It also fails if a third guard is added without updating the docs -- which is the point:
    the next person to close this hole has to come through here.
    """
    typo = "mfgarchon.core.measrue"  # measure, misspelt
    real = get_logger("mfgarchon.core.measure")

    with mfg_caplog.at_level(logging.WARNING, logger=typo):
        real.warning("emitted on the RIGHT logger, captured on the wrong one")

    assert mfg_caplog.records == [], (
        "a misspelt logger name still captures nothing -- if this now raises or captures, the "
        "hole has been closed and `at_level`'s docstring and AGENTS.md must say how"
    )


def test_the_discipline_that_replaces_the_guard(mfg_caplog):
    """What an absence assertion owes instead: a presence assertion on the same name.

    A typo fails the presence half loudly, which is what makes the absence half mean something.
    This is the shape `test_fp_network_mass_gate.py` already uses.
    """
    name = "mfgarchon.core.measure"
    logger = get_logger(name)

    with mfg_caplog.at_level(logging.WARNING, logger=name):
        pass
    assert mfg_caplog.records == [], "nothing logged yet"

    with mfg_caplog.at_level(logging.WARNING, logger=name):
        logger.warning("the same name, now used")
    assert mfg_caplog.messages == ["the same name, now used"], "the presence half is what proves the name was right"


def test_it_refuses_a_nested_capture_of_the_same_logger(mfg_caplog):
    """Two collectors on one logger append the same record twice, which silently doubles a
    count assertion -- and two converted tests assert exact counts."""
    get_logger(PROBE)

    with mfg_caplog.at_level(logging.WARNING, logger=PROBE):
        with (
            pytest.raises(RuntimeError, match="already capturing"),
            mfg_caplog.at_level(logging.WARNING, logger=PROBE),
        ):
            pass
        get_logger(PROBE).warning("counted once")

    assert mfg_caplog.messages == ["counted once"]


def test_a_logger_outside_the_package_is_left_as_it_was_found(mfg_caplog):
    """`get_logger` configures a logger it has not seen -- clears handlers, sets the level,
    attaches a StreamHandler, sets `propagate = False`. For a name outside the package there is
    no argument for leaving that behind: measured on `matplotlib`, an otherwise empty block left
    it at `propagate=True -> False, handlers=0 -> 1` for the rest of the process, and a later
    `caplog` assertion on a third-party warning would go red -- its absence form vacuously.

    In-package names are deliberately NOT restored; see `at_level`'s `finally`.
    """
    outside = "zz_mfg_caplog_probe_outside_the_package"
    assert outside not in logging.Logger.manager.loggerDict, "must not exist yet"

    with mfg_caplog.at_level(logging.WARNING, logger=outside):
        pass

    after = logging.getLogger(outside)
    assert (after.propagate, after.handlers) == (True, []), (
        f"left an out-of-package logger configured: propagate={after.propagate}, {len(after.handlers)} handler(s)"
    )


@pytest.mark.parametrize("missing", ["", None])
def test_it_refuses_a_missing_logger_name(mfg_caplog, missing):
    """There is no root to fall back to, so a silent fallback would reproduce exactly the
    failure this fixture removes: a capture that collects nothing and says nothing."""
    with pytest.raises(ValueError, match="name of the logger"), mfg_caplog.at_level(logging.WARNING, logger=missing):
        pass  # pragma: no cover -- at_level raises on __enter__

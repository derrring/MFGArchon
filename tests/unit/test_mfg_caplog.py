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
    fresh = "mfgarchon.tests.mfg_caplog_configuration"
    assert fresh not in MFGLogger._loggers

    with mfg_caplog.at_level(logging.WARNING, logger=fresh):
        logging.getLogger(fresh).warning("configured by the package, not by logging")

    assert mfg_caplog.messages == ["configured by the package, not by logging"]
    assert fresh in MFGLogger._loggers, "at_level did not go through mfgarchon's get_logger"
    assert logging.getLogger(fresh).propagate is False, "the logger did not get package configuration"


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


def test_it_refuses_a_name_the_package_has_never_used(mfg_caplog):
    """The failure a required `logger=` does NOT prevent: a name that is merely wrong.

    `assert not mfg_caplog.records` is satisfied by a misspelt logger exactly as it is by a
    solve that did not warn, and nothing in the test distinguishes them. Six absence
    assertions in this repository rest on a hand-typed string, so the fixture refuses a name
    no mfgarchon code has ever asked for when the block also captured nothing.
    """
    typo = "mfgarchon.tests.mfg_caplog_probe_typo"

    with (
        pytest.raises(LookupError, match="has ever obtained a logger named"),
        mfg_caplog.at_level(logging.WARNING, logger=typo),
    ):
        get_logger(PROBE).warning("emitted on the RIGHT logger, captured on the wrong one")

    assert mfg_caplog.records == []
    assert typo not in MFGLogger._loggers, "the refused name must not be left registered"
    assert typo not in logging.Logger.manager.loggerDict


def test_a_name_the_package_does_use_is_accepted_even_when_it_stays_silent(mfg_caplog):
    """The control for the refusal above, and the reason it is safe for the six absence
    assertions: a real mfgarchon logger that emits nothing must NOT raise. Without this, the
    check above would be indistinguishable from one that forbids absence assertions entirely."""
    import mfgarchon.core.measure  # noqa: F401  -- its module-level get_logger registers the name

    real_and_silent = "mfgarchon.core.measure"
    assert real_and_silent in MFGLogger._loggers

    with mfg_caplog.at_level(logging.WARNING, logger=real_and_silent):
        pass

    assert mfg_caplog.records == []


def test_a_body_that_raises_is_not_masked_by_the_name_check(mfg_caplog):
    """The name check runs after the block, so it must not replace a real exception with its
    own. An unknown name plus zero records is exactly the state a failed body leaves behind."""
    unknown = "mfgarchon.tests.mfg_caplog_never_used"

    with (
        pytest.raises(RuntimeError, match="the body's own failure"),
        mfg_caplog.at_level(logging.WARNING, logger=unknown),
    ):
        raise RuntimeError("the body's own failure")


def test_it_refuses_a_nested_capture_of_the_same_logger(mfg_caplog):
    """Two collectors on one logger append the same record twice, which silently doubles a
    count assertion -- and two converted tests assert exact counts."""
    with mfg_caplog.at_level(logging.WARNING, logger=PROBE):
        with (
            pytest.raises(RuntimeError, match="already capturing"),
            mfg_caplog.at_level(logging.WARNING, logger=PROBE),
        ):
            pass
        get_logger(PROBE).warning("counted once")

    assert mfg_caplog.messages == ["counted once"]


@pytest.mark.parametrize("missing", ["", None])
def test_it_refuses_a_missing_logger_name(mfg_caplog, missing):
    """There is no root to fall back to, so a silent fallback would reproduce exactly the
    failure this fixture removes: a capture that collects nothing and says nothing."""
    with pytest.raises(ValueError, match="name of the logger"), mfg_caplog.at_level(logging.WARNING, logger=missing):
        pass  # pragma: no cover -- at_level raises on __enter__

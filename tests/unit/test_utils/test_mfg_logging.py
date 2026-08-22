"""
Unit tests for mfgarchon.utils.mfg_logging module.

Tests include:
- Thread safety of logger creation (Issue #620)
- Handler deduplication
- Logger caching
"""

from __future__ import annotations

import concurrent.futures
import logging

import pytest

from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.mfg_logging.logger import MFGLogger


class TestThreadSafety:
    """Test thread-safe logger creation (Issue #620)."""

    def setup_method(self):
        """Clean state before each test."""
        # Clear cached loggers for isolated tests
        # Remove test loggers from both cache and logging module
        test_loggers = [k for k in MFGLogger._loggers if k.startswith("test.")]
        for name in test_loggers:
            del MFGLogger._loggers[name]
            # Also reset handlers on the actual logger
            logger = logging.getLogger(name)
            logger.handlers.clear()

    def test_concurrent_logger_creation_no_duplicate_handlers(self):
        """Multiple threads creating same logger should not duplicate handlers."""
        handler_counts: dict[str, int] = {}
        errors: list[str] = []

        def get_logger_from_thread(thread_id: int) -> str:
            """Simulate concurrent logger access."""
            # 5 unique loggers, multiple threads per logger
            logger_name = f"test.thread_{thread_id % 5}"
            logger = get_logger(logger_name)
            handler_counts[logger_name] = len(logger.handlers)
            return logger_name

        # Run 50 concurrent requests across 10 worker threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_logger_from_thread, i) for i in range(50)]
            concurrent.futures.wait(futures)

        # Verify no duplicate handlers
        for name, count in handler_counts.items():
            if count > 1:
                errors.append(f"Logger '{name}' has {count} handlers (expected <=1)")

        assert not errors, f"Handler duplication detected: {errors}"

    def test_concurrent_logger_creation_all_cached(self):
        """All loggers should be properly cached after concurrent creation."""

        def get_logger_from_thread(thread_id: int) -> str:
            logger_name = f"test.cache_{thread_id % 5}"
            get_logger(logger_name)
            return logger_name

        # Run concurrent logger creation
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_logger_from_thread, i) for i in range(50)]
            concurrent.futures.wait(futures)

        # Verify all loggers cached
        expected_loggers = {f"test.cache_{i}" for i in range(5)}
        cached_loggers = {k for k in MFGLogger._loggers if k.startswith("test.cache_")}

        assert cached_loggers == expected_loggers, f"Cache mismatch: expected {expected_loggers}, got {cached_loggers}"


class TestLoggerCreation:
    """Test basic logger creation functionality."""

    def test_get_logger_returns_logger(self):
        """get_logger should return a logging.Logger instance."""
        logger = get_logger("test.basic")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_caches_logger(self):
        """Logger should be cached in MFGLogger._loggers."""
        logger_name = "test.cached"
        logger = get_logger(logger_name)
        assert logger_name in MFGLogger._loggers
        assert MFGLogger._loggers[logger_name] is logger

    def test_repeated_get_logger_returns_same_instance(self):
        """Calling get_logger twice should return the same logger, already configured (Issue #620)."""
        logger1 = get_logger("test.repeated")
        n = len(logger1.handlers)
        logger2 = get_logger("test.repeated")
        assert logger1 is logger2
        # Identity alone is guaranteed by stdlib getLogger; what get_logger owes is a CONFIGURED
        # logger, which a bare logging.getLogger (0 handlers, propagate=True) would not be.
        assert len(logger2.handlers) == n == 1
        assert logger2.propagate is False


class TestHandlerDeduplication:
    """Test handler deduplication for mixed usage scenarios."""

    def test_no_duplicate_handlers_on_existing_logger(self):
        """If logger already has handlers, should not add more."""
        logger_name = "test.existing_handlers"

        # Pre-create logger with handler via standard logging
        existing_logger = logging.getLogger(logger_name)
        existing_handler = logging.NullHandler()
        existing_logger.addHandler(existing_handler)
        initial_count = len(existing_logger.handlers)

        # Now get via mfg_logging
        mfg_logger = get_logger(logger_name)

        # Should not have added duplicate handlers
        # (may have added one if none existed, but not duplicates)
        assert mfg_logger is existing_logger
        # Handler count should not have increased significantly
        assert len(mfg_logger.handlers) <= initial_count + 1

        # Cleanup
        existing_logger.handlers.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestLogAnalyserSeesEveryLevel:
    """#1918: the writer's padding and the reader's regex are two owners of one line format."""

    def test_every_level_written_by_the_real_formatter_is_parsed_back(self, tmp_path):
        r"""CRITICAL was invisible to the analyser, and it is the level the analyser filters FOR.

        Both ends are the real thing: `MFGFormatter` is the writer (`logger.py`), `LogAnalyzer` is
        the reader (`analysis.py`), and nothing here restates either one's format.

        What it catches, mutation-tested rather than asserted: a reader regression, a change to the
        ` - ` separator, a dropped `datefmt`, and a reordering of the name/level fields. What it
        does NOT catch is a change to the PADDING WIDTH -- `-8s` to `-9s` still passes -- which is
        the exact class this bug came from. That is the repaired regex being tolerant of padding,
        not the test being weak, but it is the limit of what this pins.

        `%(levelname)-8s` pads to width 8. CRITICAL is exactly 8 characters, so it is the one level
        that arrives with no padding, leaving a single space where the reader's `(\w+)\s+ - ` needed
        two. Measured before the fix: 5 levels emitted, 4 parsed, CRITICAL the only miss -- and
        `get_summary_statistics` and `find_error_patterns` both select
        `level in ("ERROR", "CRITICAL")`, so the highest severity was absent from both failure
        reports. (Named, not cited by line: this test's own edits move those lines.)
        """
        import logging

        from mfgarchon.utils.mfg_logging.analysis import LogAnalyzer
        from mfgarchon.utils.mfg_logging.logger import MFGFormatter

        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        log_path = tmp_path / "probe.log"
        logger = logging.getLogger("mfgarchon.test.levels")
        previous_propagate, previous_level = logger.propagate, logger.level
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(log_path, mode="w")
        handler.setFormatter(MFGFormatter())
        logger.addHandler(handler)
        try:
            for level in levels:
                logger.log(getattr(logging, level), f"probe line for {level}")
        finally:
            handler.close()
            logger.handlers.clear()
            logger.propagate, logger.level = previous_propagate, previous_level

        analyzer = LogAnalyzer(str(log_path))
        analyzer.parse_log_file()
        parsed = {entry["level"] for entry in analyzer.entries}

        assert parsed == set(levels), f"emitted {levels}, parsed back {sorted(parsed)}"


class TestEveryWriterUsesTheOwnedFormat:
    """#2058: MFGFormatter owns the line format; a hand-rolled copy is a second owner."""

    def test_the_logging_hook_writes_a_file_LogAnalyzer_can_read(self, tmp_path):
        """`LoggingHook` built its own `logging.Formatter` and produced unreadable files.

        Two departures from `MFGFormatter`, and only ONE of them matters. The missing `datefmt`
        makes `asctime` carry milliseconds, so the reader's timestamp field never matches and the
        line is dropped before its level is examined -- necessary and sufficient. The missing
        `-8s` padding costs nothing, because #2056 widened the level group to accept the single
        space an unpadded levelname leaves; the test 47 lines above pins exactly that tolerance.

        Measured 2x2 against the current reader: `datefmt` alone gives 5/5 at every level with or
        without padding, no `datefmt` gives 0/5 either way. An earlier revision of this docstring
        said both defects were independently fatal -- true of the pre-#1918 reader, and carried
        forward without re-measuring against the one #2056 shipped.

        This goes through the writer a user actually gets -- `LoggingHook(log_file=...)` -- rather
        than through `MFGFormatter` directly, because the defect was that the hook did not use it.
        """
        import logging

        from mfgarchon.hooks.visualization import LoggingHook
        from mfgarchon.utils.mfg_logging.analysis import LogAnalyzer

        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        log_path = tmp_path / "solver.log"
        # LoggingHook uses the fixed logger name "MFGSolver" and always addHandler()s, so a test
        # that does not clean up leaks a handler into every later construction.
        existing = logging.getLogger("MFGSolver")
        saved_handlers, saved_level = list(existing.handlers), existing.level
        saved_propagate = existing.propagate
        cached_logger = MFGLogger._loggers.get("MFGSolver")
        existing.handlers.clear()
        try:
            hook = LoggingHook(log_file=str(log_path), log_level="DEBUG")
            for level in levels:
                hook.logger.log(getattr(logging, level), f"probe line for {level}")
            for handler in hook.logger.handlers:
                handler.flush()
                if isinstance(handler, logging.FileHandler):
                    handler.close()
        finally:
            logging.getLogger("MFGSolver").handlers.clear()
            existing.handlers.extend(saved_handlers)
            existing.level = saved_level
            existing.propagate = saved_propagate
            # MFGLogger caches by name and short-circuits _setup_logger on a hit, so leaving
            # "MFGSolver" in the cache hands every later get_logger a handler-less,
            # non-propagating logger: INFO vanishes and WARNING+ falls through to lastResort on
            # unformatted stderr. Restoring handlers is not enough; the cache entry has to go.
            if cached_logger is None:
                MFGLogger._loggers.pop("MFGSolver", None)
            else:
                MFGLogger._loggers["MFGSolver"] = cached_logger

        analyzer = LogAnalyzer(str(log_path))
        analyzer.parse_log_file()
        parsed = {entry["level"] for entry in analyzer.entries}

        assert parsed == set(levels), f"emitted {levels}, parsed back {sorted(parsed)}"

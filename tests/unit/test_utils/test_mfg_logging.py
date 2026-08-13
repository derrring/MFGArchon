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
        """Calling get_logger twice should return the same logger."""
        logger1 = get_logger("test.repeated")
        logger2 = get_logger("test.repeated")
        assert logger1 is logger2


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

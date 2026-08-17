"""Tests for mfgarchon.utils.progress module."""

from __future__ import annotations

import time
from io import StringIO
from unittest.mock import patch

from mfgarchon.utils.progress import (
    IterationProgress,
    RichProgressBar,
    SolverTimer,
    progress_context,
    solver_progress,
    time_solver_operation,
    timed_operation,
)


class TestSolverTimer:
    """Test SolverTimer context manager."""

    def test_timer_basic_usage(self):
        """Test basic timer usage."""
        with patch("sys.stdout", new=StringIO()):
            timer = SolverTimer("Test Operation", verbose=False)
            with timer:
                time.sleep(0.01)

            assert timer.duration is not None
            assert timer.duration >= 0.01
            assert timer.start_time is not None
            assert timer.end_time is not None

    def test_timer_verbose_output(self):
        """Test timer prints output when verbose=True."""
        with patch("sys.stdout", new=StringIO()) as mock_stdout:
            with SolverTimer("Verbose Test", verbose=True):
                time.sleep(0.01)

            output = mock_stdout.getvalue()
            assert "Starting Verbose Test" in output
            assert "SUCCESS" in output
            assert "completed" in output

    def test_timer_silent_mode(self):
        """Test timer does not print when verbose=False."""
        with patch("sys.stdout", new=StringIO()) as mock_stdout:
            with SolverTimer("Silent Test", verbose=False):
                time.sleep(0.01)

            output = mock_stdout.getvalue()
            assert output == ""

    def test_timer_format_milliseconds(self):
        """Test duration formatting for milliseconds."""
        timer = SolverTimer("Test", verbose=False)
        timer.duration = 0.0123
        formatted = timer.format_duration()
        assert "ms" in formatted
        assert "12.3" in formatted

    def test_timer_format_seconds(self):
        """Test duration formatting for seconds."""
        timer = SolverTimer("Test", verbose=False)
        timer.duration = 5.67
        formatted = timer.format_duration()
        assert "s" in formatted
        assert "5.67" in formatted

    def test_timer_format_minutes(self):
        """Test duration formatting for minutes."""
        timer = SolverTimer("Test", verbose=False)
        timer.duration = 125.3  # 2 minutes 5.3 seconds
        formatted = timer.format_duration()
        assert "m" in formatted
        assert "2m" in formatted

    def test_timer_format_hours(self):
        """Test duration formatting for hours."""
        timer = SolverTimer("Test", verbose=False)
        timer.duration = 7325  # 2 hours 2 minutes 5 seconds
        formatted = timer.format_duration()
        assert "h" in formatted
        assert "2h" in formatted

    def test_timer_format_none_duration(self):
        """Test duration formatting when duration is None."""
        timer = SolverTimer("Test", verbose=False)
        timer.duration = None
        formatted = timer.format_duration()
        assert formatted == "Unknown"

    def test_timer_with_exception(self):
        """Test timer handles exceptions properly."""
        with patch("sys.stdout", new=StringIO()) as mock_stdout:
            try:
                with SolverTimer("Exception Test", verbose=True):
                    time.sleep(0.01)
                    raise ValueError("Test exception")
            except ValueError:
                pass

            output = mock_stdout.getvalue()
            assert "ERROR" in output
            assert "failed" in output


class TestIterationProgress:
    """Test IterationProgress class."""

    def test_iteration_progress_initialization(self):
        """Test IterationProgress initialization."""
        progress = IterationProgress(
            max_iterations=100,
            description="Test Progress",
            update_frequency=10,
            disable=True,  # Disable for testing
        )

        assert progress.max_iterations == 100
        assert progress.description == "Test Progress"
        assert progress.update_frequency == 10
        assert progress.disable is True

    def test_iteration_progress_context_manager(self):
        """Test IterationProgress as context manager."""
        with patch("sys.stdout", new=StringIO()), IterationProgress(max_iterations=10, disable=True) as progress:
            assert progress is not None
            assert progress.current_iteration == 0

    def test_iteration_progress_update(self):
        """update() accumulates iterations and formats the error into the postfix.

        Previously run with ``disable=True``, where ``update`` returns at its first line
        (``if self.disable or not self.pbar: return``) -- so neither call reached any update
        logic and the assertion only re-stated the context manager's return. The disabled
        early-return is covered by ``test_iteration_progress_disabled``; this drives the
        enabled path, where both observables are real: the accumulator advances by n, and the
        error is formatted as ``f"{error:.2e}"`` into the postfix the bar stores.
        """
        with patch("sys.stdout", new=StringIO()), IterationProgress(max_iterations=10) as progress:
            progress.update(1)
            progress.update(2, error=0.01)

            assert progress.current_iteration == 3
            assert progress.pbar.postfix_data["err"] == "1.00e-02"

    def test_iteration_progress_set_description(self):
        """Test IterationProgress set_description method."""
        with patch("sys.stdout", new=StringIO()), IterationProgress(max_iterations=10, disable=True) as progress:
            progress.set_description("New Description")
            assert progress is not None
            assert progress.max_iterations == 10

    def test_iteration_progress_disabled(self):
        """Test IterationProgress when disabled."""
        with IterationProgress(max_iterations=100, disable=True) as progress:
            progress.update(10)
            assert progress.pbar is None


class TestRichProgressBar:
    """Test RichProgressBar wrapper."""

    def test_rich_progress_initialization(self):
        """Test RichProgressBar initialization."""
        pbar = RichProgressBar(total=100, desc="Test", disable=True)
        assert pbar.total == 100
        assert pbar.desc == "Test"
        assert pbar.disable is True

    def test_rich_progress_context_manager(self):
        """Test RichProgressBar as context manager."""
        with RichProgressBar(total=10, disable=True) as pbar:
            assert pbar is not None
            assert pbar.n == 0

    def test_rich_progress_update(self):
        """Test RichProgressBar update method."""
        with RichProgressBar(total=10, disable=True) as pbar:
            pbar.update(1)
            assert pbar.n == 1
            pbar.update(2)
            assert pbar.n == 3

    def test_rich_progress_set_postfix(self):
        """Test RichProgressBar set_postfix method."""
        with RichProgressBar(total=10, disable=True) as pbar:
            pbar.set_postfix(loss=0.5, accuracy=0.95)
            assert "loss" in pbar.postfix_data
            assert pbar.postfix_data["loss"] == 0.5

    def test_rich_progress_set_description(self):
        """Test RichProgressBar set_description method."""
        with RichProgressBar(total=10, disable=True) as pbar:
            pbar.set_description("New Description")
            assert pbar.desc == "New Description"

    def test_rich_progress_iteration(self):
        """Test RichProgressBar iteration."""
        items = [1, 2, 3, 4, 5]
        pbar = RichProgressBar(iterable=items, disable=True)
        collected = []
        for item in pbar:
            collected.append(item)
        assert collected == items

    def test_rich_progress_close(self):
        """close() stops a started bar, clears _started, and is idempotent.

        Previously the bar was never entered, so ``self.progress`` was None and ``_started``
        False: ``close()`` -- ``if self.progress and self._started:`` -- took no branch at all,
        and the assertion re-read the constructor argument. No other test in this file calls
        close() (the iteration tests exit through ``__exit__``), so this is the only place the
        state transition close() owns can be pinned.
        """
        with patch("sys.stdout", new=StringIO()):
            pbar = RichProgressBar(total=10)
            assert pbar._started is False

            pbar.__enter__()
            assert pbar._started is True

            pbar.close()
            assert pbar._started is False

            pbar.close()  # idempotent: must not raise on an already-closed bar
            assert pbar._started is False


class TestTimedOperation:
    """Test timed_operation decorator."""

    def test_timed_operation_basic(self):
        """Test basic timed_operation usage as decorator."""

        @timed_operation("Test Op", verbose=False)
        def test_func():
            time.sleep(0.01)
            return {"value": 42}

        with patch("sys.stdout", new=StringIO()):
            result = test_func()

        assert result["value"] == 42
        assert "execution_time" in result

    def test_timed_operation_verbose(self):
        """Test timed_operation with verbose output."""

        @timed_operation("Verbose Op", verbose=True)
        def test_func():
            time.sleep(0.01)
            return {"value": 1}

        with patch("sys.stdout", new=StringIO()) as mock_stdout:
            test_func()

        output = mock_stdout.getvalue()
        assert "Verbose Op" in output

    def test_timed_operation_default_description(self):
        """Test timed_operation with default description."""

        @timed_operation(verbose=False)
        def my_function():
            time.sleep(0.01)
            return {"done": True}

        with patch("sys.stdout", new=StringIO()):
            result = my_function()

        assert result["done"] is True
        assert "execution_time" in result


class TestProgressContext:
    """Test progress_context context manager."""

    def test_progress_context_basic(self):
        """Test basic progress_context usage."""
        items = [1, 2, 3, 4, 5]
        with patch("sys.stdout", new=StringIO()), progress_context(items, description="Test", disable=True) as pbar:
            count = 0
            for _item in pbar:
                count += 1
            assert count == 5

    def test_progress_context_with_range(self):
        """Test progress_context with range."""
        with patch("sys.stdout", new=StringIO()), progress_context(range(10), description="Test", disable=True) as pbar:
            total = sum(x for x in pbar)
            assert total == 45  # sum(0..9)


class TestSolverProgress:
    """Test solver_progress factory function."""

    def test_solver_progress_creation(self):
        """Test solver_progress creates IterationProgress."""
        progress = solver_progress(max_iterations=100, description="Test", disable=True)
        assert isinstance(progress, IterationProgress)
        assert progress.max_iterations == 100


class TestTimeSolverOperationDecorator:
    """Test time_solver_operation decorator."""

    def test_decorator_basic_usage(self):
        """Test decorator on simple function."""

        @time_solver_operation
        def simple_function():
            time.sleep(0.01)
            return {"result": 42}

        with patch("sys.stdout", new=StringIO()):
            result = simple_function()

        assert result["result"] == 42
        assert "execution_time" in result

    def test_decorator_with_args(self):
        """Test decorator on function with arguments."""

        @time_solver_operation
        def function_with_args(x, y):
            time.sleep(0.01)
            return {"sum": x + y}

        with patch("sys.stdout", new=StringIO()):
            result = function_with_args(2, 3)

        assert result["sum"] == 5
        assert "execution_time" in result

    def test_decorator_with_dict_result(self):
        """Test decorator adds timing to dict results."""

        @time_solver_operation
        def returns_dict():
            time.sleep(0.01)
            return {"value": 10}

        with patch("sys.stdout", new=StringIO()):
            result = returns_dict()

        assert "execution_time" in result
        assert isinstance(result["execution_time"], float)
        assert result["execution_time"] > 0

    def test_decorator_with_non_dict_result(self):
        """Test decorator with non-dict return value."""

        @time_solver_operation
        def returns_tuple():
            time.sleep(0.01)
            return (1, 2, 3)

        with patch("sys.stdout", new=StringIO()):
            result = returns_tuple()

        # Non-dict results are returned unchanged
        assert result == (1, 2, 3)


class TestRichProgressBarAdvanced:
    """Test advanced RichProgressBar features."""

    def test_rich_progress_format_description_with_postfix(self):
        """Test description formatting with postfix data."""
        pbar = RichProgressBar(total=10, desc="Test", disable=True)
        pbar.postfix_data = {"loss": 0.5, "acc": 0.9}
        formatted = pbar._format_description()
        assert "Test" in formatted
        assert "loss=0.5" in formatted
        assert "acc=0.9" in formatted

    def test_rich_progress_empty_iterable(self):
        """Test RichProgressBar with empty iterable."""
        pbar = RichProgressBar(iterable=None, disable=True)
        result = list(pbar)
        assert result == []


class TestIntegration:
    """Integration tests for progress module."""

    def test_progress_with_solver_timer_integration(self):
        """Test using IterationProgress with SolverTimer together."""
        with (
            patch("sys.stdout", new=StringIO()),
            SolverTimer("Solver", verbose=False) as timer,
            IterationProgress(max_iterations=5, disable=True) as progress,
        ):
            for _i in range(5):
                time.sleep(0.001)
                progress.update(1)

        assert timer.duration is not None
        assert timer.duration > 0

    def test_nested_progress_contexts(self):
        """Test nested progress contexts."""
        outer_count = 0
        inner_count = 0
        with patch("sys.stdout", new=StringIO()):
            outer_items = range(3)
            with progress_context(outer_items, description="Outer", disable=True) as outer:
                for _i in outer:
                    outer_count += 1
                    inner_items = range(2)
                    with progress_context(inner_items, description="Inner", disable=True) as inner:
                        for _j in inner:
                            inner_count += 1
                            time.sleep(0.001)
        assert outer_count == 3
        assert inner_count == 6

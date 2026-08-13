"""Tests for mfgarchon.utils.solver_decorators module."""

from __future__ import annotations

import time
from io import StringIO
from unittest.mock import patch

import pytest

from mfgarchon.utils.solver_decorators import (
    SolverMonitoringOptions,
    SolverProgressMixin,
    enhanced_solver_method,
    format_solver_summary,
    update_solver_progress,
    upgrade_solver_with_progress,
    with_progress_monitoring,
)


class DummySolver:
    """Minimal solver for testing decorators."""

    def __init__(self, max_iterations=10):
        self.max_iterations = max_iterations
        self.__class__.__name__ = "DummySolver"

    def solve(self, max_iterations=None, verbose=True):
        """Simple solve method."""
        iterations = max_iterations or self.max_iterations
        time.sleep(0.01)
        return {"iterations": iterations, "converged": True}


class TestSolverMonitoringOptions:
    """Test SolverMonitoringOptions Flag enum."""

    def test_none_value(self):
        """Test NONE flag has value 0."""
        assert SolverMonitoringOptions.NONE.value == 0

    def test_flag_combination_with_or(self):
        """Test flags can be combined with bitwise OR."""
        combined = SolverMonitoringOptions.PROGRESS | SolverMonitoringOptions.TIMING
        assert isinstance(combined, SolverMonitoringOptions)
        assert SolverMonitoringOptions.PROGRESS in combined
        assert SolverMonitoringOptions.TIMING in combined
        assert SolverMonitoringOptions.CONVERGENCE not in combined

    def test_all_flag_includes_all_options(self):
        """Test ALL flag includes all monitoring options."""
        all_options = SolverMonitoringOptions.ALL
        assert SolverMonitoringOptions.CONVERGENCE in all_options
        assert SolverMonitoringOptions.PROGRESS in all_options
        assert SolverMonitoringOptions.TIMING in all_options

    def test_membership_checking(self):
        """Test membership checking with 'in' operator."""
        options = SolverMonitoringOptions.PROGRESS | SolverMonitoringOptions.TIMING

        assert SolverMonitoringOptions.PROGRESS in options
        assert SolverMonitoringOptions.TIMING in options
        assert SolverMonitoringOptions.CONVERGENCE not in options

    def test_single_flag(self):
        """Test single flag works correctly."""
        options = SolverMonitoringOptions.PROGRESS
        assert SolverMonitoringOptions.PROGRESS in options
        assert SolverMonitoringOptions.TIMING not in options
        assert SolverMonitoringOptions.CONVERGENCE not in options

    def test_triple_combination(self):
        """Test combining three flags."""
        options = (
            SolverMonitoringOptions.CONVERGENCE | SolverMonitoringOptions.PROGRESS | SolverMonitoringOptions.TIMING
        )
        assert SolverMonitoringOptions.CONVERGENCE in options
        assert SolverMonitoringOptions.PROGRESS in options
        assert SolverMonitoringOptions.TIMING in options

    def test_enhanced_solver_with_new_enum_api(self):
        """Test enhanced_solver_method with new enum-based API."""

        @enhanced_solver_method(options=SolverMonitoringOptions.PROGRESS | SolverMonitoringOptions.TIMING)
        def solve(self, max_iterations=10, verbose=False, **kwargs):
            time.sleep(0.01)
            return {"converged": True}

        solver = DummySolver()
        with patch("sys.stdout", new=StringIO()):
            result = solve(solver)

        assert result["converged"] is True
        # Timing should be enabled
        assert "execution_time" in result

    def test_enhanced_solver_with_none_flag(self):
        """Test enhanced_solver_method with NONE flag."""

        @enhanced_solver_method(options=SolverMonitoringOptions.NONE)
        def solve(self, max_iterations=10, verbose=True, **kwargs):
            return {"converged": True}

        solver = DummySolver()
        result = solve(solver)

        assert result["converged"] is True
        assert "execution_time" not in result


class TestWithProgressMonitoring:
    """Test with_progress_monitoring decorator."""

    def test_decorator_basic_usage(self):
        """Test decorator works on simple solve method."""

        @with_progress_monitoring(show_progress=False, show_timing=False)
        def solve(self, max_iterations=10, verbose=True, **kwargs):
            return {"iterations": max_iterations}

        solver = DummySolver()
        result = solve(solver, max_iterations=5)
        assert result["iterations"] == 5

    def test_decorator_with_timing(self):
        """Test decorator adds timing information."""

        @with_progress_monitoring(show_progress=False, show_timing=True)
        def solve(self, max_iterations=10, verbose=True, **kwargs):
            time.sleep(0.01)
            return {"iterations": max_iterations}

        solver = DummySolver()
        with patch("sys.stdout", new=StringIO()):
            result = solve(solver, max_iterations=5)

        assert "execution_time" in result
        # Duration might be None due to timing of __exit__ call
        assert result["execution_time"] is None or result["execution_time"] > 0

    def test_decorator_finds_max_iterations_from_kwargs(self):
        """Alias kwarg names resolve, and the instance attribute takes precedence over them.

        ``show_progress=True`` is required for the resolved value to be observable at all: the
        tracker that carries it is only constructed when progress is enabled.

        The previous version ran this on a solver that HAS ``max_iterations``, where the alias
        lookup never happens. solver_decorators.py resolves with
        ``if param_name in kwargs: ... break; elif hasattr(self, param_name): ... break``, so on
        such a solver the first loop pass always breaks on the elif and every alias is dead.
        Measured: DummySolver(10) + Niter=20 resolves to 10, not 20. So the alias branch needs a
        solver without the attribute; the precedence the old fixture silently relied on is
        pinned separately below.
        """

        @with_progress_monitoring(show_progress=True, show_timing=False)
        def solve(self, **kwargs):
            return {"n": kwargs["_progress_tracker"].max_iterations}

        class _NoMaxIter:
            pass

        for param_name in ["max_iterations", "Niter", "max_picard_iterations"]:
            assert solve(_NoMaxIter(), **{param_name: 20})["n"] == 20, f"alias {param_name} not resolved"

        # Instance attribute wins over an alias kwarg (measured 10, not 20)...
        assert solve(DummySolver(max_iterations=10), Niter=20)["n"] == 10
        # ...but the kwarg wins when it is the FIRST name in the lookup order.
        assert solve(DummySolver(max_iterations=10), max_iterations=20)["n"] == 20

    def test_decorator_finds_max_iterations_from_instance(self):
        """The value discovered on the instance is the one the tracker is built with.

        The old version returned a literal ``{}`` and asserted ``result is not None``, which can
        only fail if the decorator swallows the return value entirely -- the discovery branch
        this test is named for ran and was then not observed. Reading it off the injected
        tracker makes it observable (measured 15).
        """

        @with_progress_monitoring(show_progress=True, show_timing=False)
        def solve(self, **kwargs):
            return {"n": kwargs["_progress_tracker"].max_iterations}

        assert solve(DummySolver(max_iterations=15))["n"] == 15

    def test_decorator_disables_progress_when_verbose_false(self):
        """verbose=False suppresses the tracker injection; verbose=True performs it.

        The old version read ``"_progress_tracker" in locals()`` inside the decorated body,
        which is a question about the body's own locals, not about what the decorator injected
        -- so nothing about verbose was asserted, as its own comment conceded. The decorator
        computes ``disable_progress = not (show_progress and verbose)`` and injects the tracker
        into kwargs only when that is False.
        """

        @with_progress_monitoring(show_progress=True, show_timing=False)
        def solve(self, max_iterations=10, verbose=True, **kwargs):
            return {"tracker": kwargs.get("_progress_tracker")}

        solver = DummySolver()

        assert solve(solver, verbose=True)["tracker"] is not None
        assert solve(solver, verbose=False)["tracker"] is None

    def test_decorator_handles_exceptions(self):
        """Test decorator properly handles exceptions."""

        @with_progress_monitoring(show_progress=False, show_timing=True)
        def solve(self, max_iterations=10, verbose=True, **kwargs):
            raise ValueError("Solver failed")

        solver = DummySolver()
        with pytest.raises(ValueError, match="Solver failed"), patch("sys.stdout", new=StringIO()):
            solve(solver)

    def test_decorator_with_metadata_result(self):
        """Test decorator adds timing to results with metadata attribute."""

        class ResultWithMetadata:
            def __init__(self):
                self.metadata = {}

        @with_progress_monitoring(show_progress=False, show_timing=True)
        def solve(self, max_iterations=10, verbose=True, **kwargs):
            time.sleep(0.01)
            return ResultWithMetadata()

        solver = DummySolver()
        with patch("sys.stdout", new=StringIO()):
            result = solve(solver)

        assert "execution_time" in result.metadata
        # Duration might be None due to timing of __exit__ call
        assert result.metadata["execution_time"] is None or result.metadata["execution_time"] > 0

    def test_decorator_update_frequency(self):
        """The configured update_frequency reaches the tracker, and the auto path differs from it.

        With ``show_progress=False`` (the old fixture) the decorator computes
        ``disable_progress = not (False and verbose) = True`` and never enters the block
        containing ``freq = update_frequency or max(1, max_iterations // 20)`` -- so the
        parameter was not merely unobserved, it was never read. Enabling progress makes it
        observable. The auto case at max_iterations=200 gives 10 rather than 5, so the two
        branches cannot be confused with each other.
        """

        @with_progress_monitoring(show_progress=True, show_timing=False, update_frequency=5)
        def solve_explicit(self, max_iterations=100, verbose=True, **kwargs):
            return {"f": kwargs["_progress_tracker"].update_frequency}

        @with_progress_monitoring(show_progress=True, show_timing=False)
        def solve_auto(self, max_iterations=100, verbose=True, **kwargs):
            return {"f": kwargs["_progress_tracker"].update_frequency}

        solver = DummySolver()

        assert solve_explicit(solver, max_iterations=100)["f"] == 5
        assert solve_explicit(solver, max_iterations=200)["f"] == 5  # explicit value ignores N
        assert solve_auto(solver, max_iterations=100)["f"] == 5  # max(1, 100 // 20)
        assert solve_auto(solver, max_iterations=200)["f"] == 10  # max(1, 200 // 20)


class TestEnhancedSolverMethod:
    """Test enhanced_solver_method decorator."""

    def test_enhanced_progress_only(self):
        """PROGRESS alone must not add timing -- the assertion that separates it from its neighbours.

        PROGRESS-alone is a distinct route: the decorator takes
        ``with_progress_monitoring(show_progress=True, show_timing=enable_timing)`` with
        enable_timing False, so no SolverTimer is constructed. ``result["converged"]`` is shared
        by every option and cannot see which route ran. Measured result keys: PROGRESS gives
        ['converged'], while PROGRESS|TIMING and ALL both give ['converged', 'execution_time'].
        Mirrors ``test_enhanced_no_features``, which already does this for NONE.
        """

        @enhanced_solver_method(options=SolverMonitoringOptions.PROGRESS)
        def solve(self, max_iterations=10, verbose=True, **kwargs):
            return {"converged": True}

        solver = DummySolver()
        with patch("sys.stdout", new=StringIO()):
            result = solve(solver)

        assert result["converged"] is True
        assert "execution_time" not in result

    def test_enhanced_timing_only(self):
        """Test enhanced decorator with only timing enabled."""

        @enhanced_solver_method(options=SolverMonitoringOptions.TIMING)
        def solve(self, max_iterations=10, verbose=True, **kwargs):
            time.sleep(0.01)
            return {"converged": True}

        solver = DummySolver()
        with patch("sys.stdout", new=StringIO()):
            result = solve(solver)

        assert "execution_time" in result
        assert result["converged"] is True

    def test_enhanced_no_features(self):
        """Test enhanced decorator with all features disabled."""

        @enhanced_solver_method(options=SolverMonitoringOptions.NONE)
        def solve(self, max_iterations=10, verbose=True, **kwargs):
            return {"converged": True}

        solver = DummySolver()
        result = solve(solver)

        assert result["converged"] is True
        assert "execution_time" not in result


class TestSolverProgressMixin:
    """Test SolverProgressMixin class."""

    def test_mixin_initialization(self):
        """Test mixin initializes with defaults."""

        class TestSolver(SolverProgressMixin):
            pass

        solver = TestSolver()
        assert solver._progress_enabled is True
        assert solver._timing_enabled is True

    def test_mixin_enable_progress(self):
        """Test enable_progress method."""

        class TestSolver(SolverProgressMixin):
            pass

        solver = TestSolver()

        solver.enable_progress(True)
        assert solver._progress_enabled is True

        solver.enable_progress(False)
        assert solver._progress_enabled is False

    def test_mixin_should_show_progress(self):
        """Test _should_show_progress method."""

        class TestSolver(SolverProgressMixin):
            pass

        solver = TestSolver()

        # Both enabled and verbose
        assert solver._should_show_progress(verbose=True) is True

        # Not verbose
        assert solver._should_show_progress(verbose=False) is False

        # Progress disabled
        solver.enable_progress(False)
        assert solver._should_show_progress(verbose=True) is False

    def test_mixin_create_progress_tracker(self):
        """Test _create_progress_tracker method."""

        class TestSolver(SolverProgressMixin):
            pass

        solver = TestSolver()

        # Progress enabled
        tracker = solver._create_progress_tracker(100, "Test Progress")
        assert tracker is not None
        assert tracker.max_iterations == 100

        # Progress disabled
        solver.enable_progress(False)
        tracker = solver._create_progress_tracker(100)
        assert tracker is None

    def test_mixin_with_inheritance(self):
        """Test mixin works with multiple inheritance."""

        class BaseSolver:
            def __init__(self):
                self.name = "Base"

        class EnhancedSolver(SolverProgressMixin, BaseSolver):
            pass

        solver = EnhancedSolver()
        assert solver.name == "Base"
        assert solver._progress_enabled is True


class TestUpgradeSolverWithProgress:
    """Test upgrade_solver_with_progress class decorator."""

    def test_upgraded_solver_name(self):
        """Test upgraded solver has correct name."""

        class MySolver:
            pass

        EnhancedSolver = upgrade_solver_with_progress(MySolver)
        assert "Enhanced" in EnhancedSolver.__name__

    def test_upgraded_solver_solve_method(self):
        """Test upgraded solver's solve method works."""

        class WorkingSolver:
            def solve(self, max_iterations=10, verbose=False, **kwargs):
                time.sleep(0.01)
                return {"converged": True}

        EnhancedSolver = upgrade_solver_with_progress(WorkingSolver)

        solver = EnhancedSolver()
        solver.enable_progress(False)  # Disable for cleaner test

        with patch("sys.stdout", new=StringIO()):
            result = solver.solve()

        assert result["converged"] is True


class RecordingTracker:
    """Tracker stub that records the (n, error, additional_info) triples forwarded to update().

    The four tests below previously used ``IterationProgress(..., disable=True)``, whose update()
    returns at its first line (``if self.disable or not self.pbar: return``) -- so nothing
    downstream of the call was reached and none of the tests had an assert statement at all.
    Recording the call makes the forwarding contract observable.
    """

    def __init__(self):
        self.calls = []

    def update(self, n, error=None, additional_info=None):
        self.calls.append((n, error, additional_info))


class TestUpdateSolverProgress:
    """Test update_solver_progress utility function."""

    def test_update_with_valid_tracker(self):
        """Forwarding contract: advance by exactly 1, raw error through, formatted error in info.

        Note what the recorded n exposes: the ``iteration`` argument is DEAD. The body calls
        ``progress_tracker.update(1, error=error, additional_info=info)`` and never passes
        ``iteration`` anywhere, so all four tests in this class supply iteration=5 with no
        effect. Pinning the advance-by-1 is what states that contract.
        """
        tracker = RecordingTracker()

        update_solver_progress(tracker, iteration=5, error=1e-5)

        assert tracker.calls == [(1, 1e-5, {"error": "1.00e-05"})]

    def test_update_with_none_tracker(self):
        """Test updating progress with None tracker (should be safe)."""
        update_solver_progress(None, iteration=5, error=1e-5)
        # Should not raise any exceptions

    def test_update_with_additional_info(self):
        """Extra metrics are collected into additional_info alongside the formatted error.

        This pins the convention that the error appears TWICE and in two forms -- as a raw float
        in ``error=`` and as a "%.2e" string inside additional_info -- which solver_decorators.py
        implements and nothing else checks.
        """
        tracker = RecordingTracker()

        update_solver_progress(tracker, iteration=5, error=1e-5, residual=0.001, step_size=0.1)

        assert tracker.calls == [(1, 1e-5, {"residual": 0.001, "step_size": 0.1, "error": "1.00e-05"})]

    def test_update_without_error(self):
        """No error given: no "error" key is injected, and error=None is forwarded as None.

        The only exercise of the ``if error is not None:`` false branch. An implementation that
        formatted None into the string ("None", "NaN", or a TypeError) is caught here and
        nowhere else.
        """
        tracker = RecordingTracker()

        update_solver_progress(tracker, iteration=5)

        assert tracker.calls == [(1, None, {})]


class TestFormatSolverSummary:
    """Test format_solver_summary utility function."""

    def test_summary_converged(self):
        """Test formatting summary for converged solver."""
        summary = format_solver_summary(
            solver_name="TestSolver", iterations=10, final_error=1e-6, execution_time=2.5, converged=True
        )

        assert "TestSolver" in summary
        assert "SUCCESS" in summary
        assert "10 iterations" in summary
        assert "1.00e-06" in summary
        assert "2.50s" in summary

    def test_summary_not_converged(self):
        """Test formatting summary for non-converged solver."""
        summary = format_solver_summary(
            solver_name="SlowSolver", iterations=100, final_error=0.1, execution_time=10.0, converged=False
        )

        assert "SlowSolver" in summary
        assert "WARNING" in summary
        assert "Max iterations reached" in summary

    def test_summary_milliseconds_timing(self):
        """Test summary formats milliseconds correctly."""
        summary = format_solver_summary(solver_name="FastSolver", iterations=5, execution_time=0.5)

        assert "500.0ms" in summary

    def test_summary_minutes_timing(self):
        """Test summary formats minutes correctly."""
        summary = format_solver_summary(solver_name="SlowSolver", iterations=100, execution_time=125.0)

        assert "2m" in summary
        assert "5.0s" in summary

    def test_summary_without_error(self):
        """Test summary without final error."""
        summary = format_solver_summary(solver_name="SimpleSolver", iterations=10, execution_time=1.0)

        assert "SimpleSolver" in summary
        assert "10 iterations" in summary
        # Should not crash without error

    def test_summary_without_timing(self):
        """Test summary without execution time."""
        summary = format_solver_summary(solver_name="QuickSolver", iterations=5, final_error=1e-8)

        assert "QuickSolver" in summary
        assert "1.00e-08" in summary
        # Should not crash without timing

    def test_summary_minimal(self):
        """Test summary with minimal information."""
        summary = format_solver_summary(solver_name="MinimalSolver", iterations=1)

        assert "MinimalSolver" in summary
        assert "1 iterations" in summary


class TestDecoratorIntegration:
    """Integration tests for decorator combinations."""

    def test_multiple_decorators(self):
        """Stacked decorators: the payload survives both layers AND the outer TIMING still fires.

        Stacking is where functools.wraps and kwargs forwarding break. ``result["converged"]``
        proves the payload came back through both wrappers, but not that the outer layer still
        functions -- which is precisely what dies silently when an inner wrapper defeats the
        outer's inspection. The outer options=TIMING with PROGRESS absent routes through
        ``time_solver_operation`` (the ``elif enable_timing:`` branch), which is what injects
        execution_time. ``test_decorator_with_mixin`` already asserts it for the unstacked
        TIMING case, so the two become comparable.
        """

        @enhanced_solver_method(options=SolverMonitoringOptions.TIMING)
        @with_progress_monitoring(show_progress=False, show_timing=False)
        def solve(self, max_iterations=10, verbose=False, **kwargs):
            time.sleep(0.01)
            return {"converged": True}

        solver = DummySolver()
        with patch("sys.stdout", new=StringIO()):
            result = solve(solver)

        assert result["converged"] is True
        assert "execution_time" in result

    def test_decorator_with_mixin(self):
        """Test decorator on method of class with mixin."""

        class MixedSolver(SolverProgressMixin):
            @enhanced_solver_method(options=SolverMonitoringOptions.TIMING)
            def solve(self, max_iterations=10, verbose=False, **kwargs):
                time.sleep(0.01)
                return {"converged": True}

        solver = MixedSolver()
        with patch("sys.stdout", new=StringIO()):
            result = solver.solve()

        assert result["converged"] is True
        assert "execution_time" in result

    def test_real_world_usage_pattern(self):
        """Test realistic usage pattern with full solver."""

        class RealisticSolver(SolverProgressMixin):
            def __init__(self):
                super().__init__()
                self.max_iterations = 50

            @enhanced_solver_method(options=SolverMonitoringOptions.ALL)
            def solve(self, max_iterations=None, tolerance=1e-6, verbose=True, **kwargs):
                iterations = max_iterations or self.max_iterations
                error = 1.0

                for i in range(iterations):
                    error = error / 2  # Simulate convergence
                    if error < tolerance:
                        return {"converged": True, "iterations": i + 1, "final_error": error}

                return {"converged": False, "iterations": iterations, "final_error": error}

        solver = RealisticSolver()
        solver.enable_progress(False)  # Disable for test

        with patch("sys.stdout", new=StringIO()):
            result = solver.solve(max_iterations=25, tolerance=1e-6)

        assert result["converged"] is True
        assert "execution_time" in result
        assert result["iterations"] <= 25  # Should converge within limit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

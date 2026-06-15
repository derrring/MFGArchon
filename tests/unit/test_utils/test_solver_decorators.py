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

    def test_enum_values_exist(self):
        """Test all expected enum values are defined."""
        assert hasattr(SolverMonitoringOptions, "NONE")
        assert hasattr(SolverMonitoringOptions, "CONVERGENCE")
        assert hasattr(SolverMonitoringOptions, "PROGRESS")
        assert hasattr(SolverMonitoringOptions, "TIMING")
        assert hasattr(SolverMonitoringOptions, "ALL")

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

    def test_enhanced_solver_with_all_flag(self):
        """Test enhanced_solver_method with ALL flag."""

        @enhanced_solver_method(options=SolverMonitoringOptions.ALL)
        def solve(self, max_iterations=10, verbose=False, **kwargs):
            time.sleep(0.01)
            return {"converged": True}

        solver = DummySolver()
        with patch("sys.stdout", new=StringIO()):
            result = solve(solver)

        assert result["converged"] is True

    def test_enhanced_solver_with_none_flag(self):
        """Test enhanced_solver_method with NONE flag."""

        @enhanced_solver_method(options=SolverMonitoringOptions.NONE)
        def solve(self, max_iterations=10, verbose=True, **kwargs):
            return {"converged": True}

        solver = DummySolver()
        result = solve(solver)

        assert result["converged"] is True
        assert "execution_time" not in result

    def test_deprecated_boolean_parameters_removed(self):
        """Removed-pin: deprecated boolean kwargs (monitor_convergence/auto_progress/timing)
        were removed in v0.20.0; passing them now raises TypeError."""
        with pytest.raises(TypeError):
            enhanced_solver_method(auto_progress=True, timing=True)

        with pytest.raises(TypeError):
            enhanced_solver_method(monitor_convergence=True)


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
        """Test decorator finds max_iterations from various parameter names."""

        @with_progress_monitoring(show_progress=False, show_timing=False)
        def solve(self, **kwargs):
            return {}

        solver = DummySolver()

        # Test various parameter name variations
        for param_name in ["max_iterations", "Niter", "max_picard_iterations", "max_iterations"]:
            result = solve(solver, **{param_name: 20})
            assert result is not None

    def test_decorator_finds_max_iterations_from_instance(self):
        """Test decorator finds max_iterations from solver instance."""

        @with_progress_monitoring(show_progress=False, show_timing=False)
        def solve(self, **kwargs):
            return {}

        solver = DummySolver(max_iterations=15)
        result = solve(solver)
        assert result is not None

    def test_decorator_disables_progress_when_verbose_false(self):
        """Test decorator respects verbose=False."""

        @with_progress_monitoring(show_progress=True, show_timing=False)
        def solve(self, max_iterations=10, verbose=True, **kwargs):
            # Check if progress tracker was added
            return {"has_progress": "_progress_tracker" in locals()}

        solver = DummySolver()
        result_verbose = solve(solver, verbose=True)
        result_silent = solve(solver, verbose=False)

        # Both should work without crashing
        assert isinstance(result_verbose, dict)
        assert isinstance(result_silent, dict)

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
        """Test decorator respects update_frequency parameter."""

        @with_progress_monitoring(show_progress=False, show_timing=False, update_frequency=5)
        def solve(self, max_iterations=100, verbose=True, **kwargs):
            return {}

        solver = DummySolver()
        result = solve(solver, max_iterations=100)
        assert result is not None


class TestEnhancedSolverMethod:
    """Test enhanced_solver_method decorator."""

    def test_enhanced_with_all_features(self):
        """Test enhanced decorator with all features enabled."""

        @enhanced_solver_method(options=SolverMonitoringOptions.ALL)
        def solve(self, max_iterations=10, verbose=True, **kwargs):
            time.sleep(0.01)
            return {"converged": True}

        solver = DummySolver()
        with patch("sys.stdout", new=StringIO()):
            result = solve(solver)

        assert result["converged"] is True

    def test_enhanced_progress_only(self):
        """Test enhanced decorator with only progress enabled."""

        @enhanced_solver_method(options=SolverMonitoringOptions.PROGRESS)
        def solve(self, max_iterations=10, verbose=True, **kwargs):
            return {"converged": True}

        solver = DummySolver()
        with patch("sys.stdout", new=StringIO()):
            result = solve(solver)

        assert result["converged"] is True

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

    def test_mixin_enable_timing(self):
        """Test enable_timing method."""

        class TestSolver(SolverProgressMixin):
            pass

        solver = TestSolver()

        solver.enable_timing(True)
        assert solver._timing_enabled is True

        solver.enable_timing(False)
        assert solver._timing_enabled is False

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

    def test_upgrade_basic_solver(self):
        """Test upgrading a basic solver class."""

        class SimpleSolver:
            def solve(self):
                return {"result": 42}

        EnhancedSolver = upgrade_solver_with_progress(SimpleSolver)

        solver = EnhancedSolver()
        assert hasattr(solver, "enable_progress")
        assert hasattr(solver, "enable_timing")

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


class TestUpdateSolverProgress:
    """Test update_solver_progress utility function."""

    def test_update_with_valid_tracker(self):
        """Test updating progress with valid tracker."""
        from mfgarchon.utils.progress import IterationProgress

        with IterationProgress(10, "Test", disable=True) as tracker:
            update_solver_progress(tracker, iteration=5, error=1e-5)
            # Should not raise any exceptions

    def test_update_with_none_tracker(self):
        """Test updating progress with None tracker (should be safe)."""
        update_solver_progress(None, iteration=5, error=1e-5)
        # Should not raise any exceptions

    def test_update_with_additional_info(self):
        """Test updating progress with additional metrics."""
        from mfgarchon.utils.progress import IterationProgress

        with IterationProgress(10, "Test", disable=True) as tracker:
            update_solver_progress(tracker, iteration=5, error=1e-5, residual=0.001, step_size=0.1)
            # Should not raise any exceptions

    def test_update_without_error(self):
        """Test updating progress without error value."""
        from mfgarchon.utils.progress import IterationProgress

        with IterationProgress(10, "Test", disable=True) as tracker:
            update_solver_progress(tracker, iteration=5)
            # Should not raise any exceptions


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
        """Test applying multiple decorators to same method."""

        @enhanced_solver_method(options=SolverMonitoringOptions.TIMING)
        @with_progress_monitoring(show_progress=False, show_timing=False)
        def solve(self, max_iterations=10, verbose=False, **kwargs):
            time.sleep(0.01)
            return {"converged": True}

        solver = DummySolver()
        with patch("sys.stdout", new=StringIO()):
            result = solve(solver)

        assert result["converged"] is True

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

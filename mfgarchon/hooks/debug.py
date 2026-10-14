"""
Debugging and Analysis Hooks for MFG Solvers

This module provides ready-to-use hooks for debugging, monitoring,
and analyzing solver behavior during execution.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import numpy as np

from mfgarchon.utils.deprecation import NEVER_READ, retired_parameters

from .base import SolverHooks

if TYPE_CHECKING:
    from mfgarchon.types import MFGResult, SpatialTemporalState


class DebugHook(SolverHooks):
    """
    General-purpose debugging hook with configurable output levels.

    Example:
        debug = DebugHook(level="verbose")
        result = solver.solve(problem, hooks=debug)
    """

    def __init__(self, level: str = "normal", output_file: str | None = None):
        """
        Initialize debug hook.

        Args:
            level: Debug level ("minimal", "normal", "verbose", "detailed")
            output_file: Optional file to write debug output to
        """
        self.level = level
        self.output_file = None
        self.file_handle: TextIO | None = None
        self.start_time: float | None = None
        self.iteration_times: list[float] = []

        if output_file:
            self.output_file = Path(output_file)
            self.file_handle = open(self.output_file, "w")  # noqa: SIM115  # File closed in on_solve_end

    def _write(self, message: str):
        """Write message to console and/or file."""
        print(message)
        if self.file_handle:
            self.file_handle.write(message + "\n")
            self.file_handle.flush()

    def on_solve_start(self, initial_state: SpatialTemporalState) -> None:
        """Log solve start information."""
        self.start_time = time.time()
        self.iteration_times = []

        self._write("=" * 60)
        self._write("MFG SOLVER DEBUG SESSION STARTED")
        self._write("=" * 60)
        self._write(f"Initial residual: {initial_state.residual:.6e}")

        if self.level in ["verbose", "detailed"]:
            self._write(f"Initial state shape: u={initial_state.u.shape}, m={initial_state.m.shape}")
            if initial_state.metadata:
                self._write(f"Metadata: {initial_state.metadata}")

    def on_iteration_start(self, state: SpatialTemporalState) -> None:
        """Log iteration start (only for detailed level)."""
        if self.level == "detailed":
            self._write(f"Starting iteration {state.iteration}")

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        """Log iteration completion."""
        current_time = time.time()
        if self.start_time:
            iteration_time = current_time - self.start_time - sum(self.iteration_times)
            self.iteration_times.append(iteration_time)

        if self.level == "minimal":
            if state.iteration % 10 == 0:
                self._write(f"Iteration {state.iteration:4d}: residual = {state.residual:.2e}")
        elif self.level == "normal":
            if state.iteration % 5 == 0:
                self._write(f"Iteration {state.iteration:4d}: residual = {state.residual:.6e}")
        elif self.level in ["verbose", "detailed"]:
            avg_time = sum(self.iteration_times) / len(self.iteration_times) if self.iteration_times else 0
            self._write(
                f"Iteration {state.iteration:4d}: residual = {state.residual:.6e}, "
                f"time = {iteration_time:.3f}s, avg = {avg_time:.3f}s"
            )

        return None

    def on_solve_end(self, result: MFGResult) -> MFGResult:
        """Log final results."""
        total_time = time.time() - self.start_time if self.start_time else 0

        self._write("=" * 60)
        self._write("SOLVER COMPLETED")
        self._write(f"Converged: {result.converged}")
        self._write(f"Total iterations: {result.iterations}")
        self._write(f"Total time: {total_time:.3f}s")

        convergence_info = getattr(result, "convergence_info", None)
        if convergence_info is not None:
            final_residual = getattr(convergence_info, "final_residual", None)
            if final_residual is not None:
                self._write(f"Final residual: {final_residual:.6e}")

        if self.level in ["verbose", "detailed"]:
            avg_iter_time = total_time / result.iterations if result.iterations > 0 else 0
            self._write(f"Average iteration time: {avg_iter_time:.3f}s")

        self._write("=" * 60)

        if self.file_handle:
            self.file_handle.close()

        return result


class PerformanceHook(SolverHooks):
    """
    Monitor solver performance metrics during execution.

    Example:
        perf = PerformanceHook(track_memory=True)
        result = solver.solve(problem, hooks=perf)
        print(f"Peak memory: {perf.peak_memory_mb:.1f} MB")
    """

    @retired_parameters(
        {"track_detailed_timing": "iteration times are recorded whatever it is set to"},
        issue="#2648",
        since="v0.22.0",
        never_read=True,
    )
    def __init__(self, track_memory: bool = False, track_detailed_timing: bool = NEVER_READ):
        self.track_memory = track_memory
        self.track_detailed_timing = track_detailed_timing

        # Performance metrics
        self.start_time: float | None = None
        self.iteration_times: list[float] = []
        self.memory_usage: list[float] = []
        self.peak_memory_mb = 0.0
        self.convergence_rate_history: list[float] = []

        # Try to import performance monitoring tools
        self.psutil_available = False
        if track_memory:
            try:
                import psutil

                self.psutil = psutil
                self.psutil_available = True
            except ImportError:
                print("Warning: psutil not available for memory tracking")

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        if self.psutil_available:
            process = self.psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        return 0.0

    def on_solve_start(self, initial_state: SpatialTemporalState) -> None:
        """Start performance monitoring."""
        self.start_time = time.time()
        self.iteration_times = []
        self.memory_usage = []
        self.convergence_rate_history = []

        if self.track_memory:
            initial_memory = self._get_memory_usage_mb()
            self.memory_usage.append(initial_memory)
            print(f"Initial memory usage: {initial_memory:.1f} MB")

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        """Record performance metrics."""
        current_time = time.time()
        if self.start_time:
            iteration_time = current_time - self.start_time - sum(self.iteration_times)
            self.iteration_times.append(iteration_time)

        # Track memory usage
        if self.track_memory:
            memory_mb = self._get_memory_usage_mb()
            self.memory_usage.append(memory_mb)
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)

        # Track convergence rate
        if len(self.convergence_rate_history) > 0 and state.residual > 0:
            prev_residual = self.convergence_rate_history[-1]
            if prev_residual > 0:
                rate = state.residual / prev_residual
                if rate < 1.0:  # Only track improvement
                    self.convergence_rate_history.append(rate)
        if state.residual > 0:
            self.convergence_rate_history.append(state.residual)

        return None

    def on_solve_end(self, result: MFGResult) -> MFGResult:
        """Report final performance statistics."""
        total_time = time.time() - self.start_time if self.start_time else 0

        print("\nPERFORMANCE SUMMARY:")
        print(f"Total solving time: {total_time:.3f}s")

        if self.iteration_times:
            avg_time = sum(self.iteration_times) / len(self.iteration_times)
            min_time = min(self.iteration_times)
            max_time = max(self.iteration_times)
            print(f"Iteration timing - Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")

        if self.track_memory and self.memory_usage:
            print(f"Memory usage - Peak: {self.peak_memory_mb:.1f} MB, Final: {self.memory_usage[-1]:.1f} MB")

        if len(self.convergence_rate_history) > 5:
            recent_rates = self.convergence_rate_history[-5:]
            avg_rate = sum(recent_rates) / len(recent_rates)
            print(f"Recent convergence rate: {avg_rate:.3e}")

        return result


class ConvergenceAnalysisHook(SolverHooks):
    """
    Analyze convergence behavior and detect convergence issues.

    Example:
        conv_analysis = ConvergenceAnalysisHook()
        result = solver.solve(problem, hooks=conv_analysis)
        if conv_analysis.oscillation_detected:
            print("Warning: Oscillatory convergence detected")
    """

    def __init__(self, analysis_window: int = 20):
        self.analysis_window = analysis_window
        self.residual_history: list[float] = []
        self.oscillation_detected = False
        self.stagnation_detected = False
        self.divergence_detected = False

    def _analyze_oscillation(self) -> bool:
        """Detect oscillatory behavior in residual."""
        if len(self.residual_history) < self.analysis_window:
            return False

        recent = self.residual_history[-self.analysis_window :]
        # Check for alternating increase/decrease pattern
        direction_changes = 0
        for i in range(1, len(recent) - 1):
            prev_direction = 1 if recent[i] > recent[i - 1] else -1
            curr_direction = 1 if recent[i + 1] > recent[i] else -1
            if prev_direction != curr_direction:
                direction_changes += 1

        # If more than 60% of steps change direction, consider it oscillatory
        return direction_changes > 0.6 * (len(recent) - 2)

    def _analyze_stagnation(self) -> bool:
        """Detect stagnation in convergence."""
        if len(self.residual_history) < self.analysis_window:
            return False

        recent = self.residual_history[-self.analysis_window :]
        if recent[0] <= 0:
            return False

        # Check relative improvement
        improvement = (recent[0] - recent[-1]) / recent[0]
        return improvement < 1e-6  # Less than 0.0001% improvement

    def _analyze_divergence(self) -> bool:
        """Detect divergence."""
        if len(self.residual_history) < 5:
            return False

        recent = self.residual_history[-5:]
        # Check if residual is consistently increasing
        return all(recent[i] > recent[i - 1] for i in range(1, len(recent)))

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        """Analyze convergence behavior."""
        self.residual_history.append(state.residual)

        # Perform analysis every few iterations
        if state.iteration % 5 == 0:
            if self._analyze_oscillation():
                if not self.oscillation_detected:
                    print(f"Warning: Oscillatory convergence detected at iteration {state.iteration}")
                    self.oscillation_detected = True

            if self._analyze_stagnation():
                if not self.stagnation_detected:
                    print(f"Warning: Convergence stagnation detected at iteration {state.iteration}")
                    self.stagnation_detected = True

            if self._analyze_divergence():
                if not self.divergence_detected:
                    print(f"Warning: Potential divergence detected at iteration {state.iteration}")
                    self.divergence_detected = True

        return None


class StateInspectionHook(SolverHooks):
    """
    Inspect and validate solution state at each iteration.

    Example:
        inspector = StateInspectionHook(check_mass_conservation=True)
        result = solver.solve(problem, hooks=inspector)
    """

    def __init__(
        self,
        check_nan: bool = True,
        check_mass_conservation: bool = True,
        check_bounds: bool = True,
        mass_tolerance: float = 1e-6,
    ):
        self.check_nan = check_nan
        self.check_mass_conservation = check_mass_conservation
        self.check_bounds = check_bounds
        self.mass_tolerance = mass_tolerance
        self.issues_found: list[str] = []

    def _check_for_nan(self, state: SpatialTemporalState) -> list[str]:
        """Check for NaN values in solution."""
        issues = []
        if np.isnan(state.u).any():
            issues.append("NaN values in u (value function)")
        if np.isnan(state.m).any():
            issues.append("NaN values in m (density function)")
        if np.isnan(state.residual):
            issues.append("NaN in residual")
        return issues

    def _check_mass_conservation(self, state: SpatialTemporalState) -> list[str]:
        """Check mass conservation."""
        issues = []
        try:
            # Assuming m is density function - check if it integrates to approximately 1
            if "x_grid" in state.metadata:
                x_grid = state.metadata["x_grid"]
                for t_idx in range(state.m.shape[0]):
                    mass = np.trapezoid(state.m[t_idx, :], x_grid)
                    mass_error = float(abs(float(mass) - 1.0))
                    if mass_error > self.mass_tolerance:
                        issues.append(f"Mass conservation violated at t={t_idx}: mass={float(mass):.6f}")
                        break  # Only report first violation to avoid spam
        except Exception as e:
            issues.append(f"Could not check mass conservation: {e}")
        return issues

    def _check_bounds(self, state: SpatialTemporalState) -> list[str]:
        """Check for reasonable bounds on solution values."""
        issues = []

        # Check for extremely large values
        if np.max(np.abs(state.u)) > 1e10:
            issues.append(f"Very large values in u: max(|u|) = {np.max(np.abs(state.u)):.2e}")

        if np.max(state.m) > 1e10:
            issues.append(f"Very large values in m: max(m) = {np.max(state.m):.2e}")

        # Check for negative density
        if np.min(state.m) < -1e-10:  # Allow small numerical errors
            issues.append(f"Negative density detected: min(m) = {np.min(state.m):.2e}")

        return issues

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        """Inspect solution state."""
        issues = []

        if self.check_nan:
            issues.extend(self._check_for_nan(state))

        if self.check_mass_conservation:
            issues.extend(self._check_mass_conservation(state))

        if self.check_bounds:
            issues.extend(self._check_bounds(state))

        if issues:
            print(f"State inspection issues at iteration {state.iteration}:")
            for issue in issues:
                print(f"  - {issue}")
            self.issues_found.extend(issues)

            # Stop if critical issues found
            if any("NaN" in issue for issue in issues):
                return "stop"

        return None

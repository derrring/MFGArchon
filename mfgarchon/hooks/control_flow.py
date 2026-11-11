"""
Advanced Control Flow for Solver Hooks

This module provides sophisticated control flow mechanisms for solver hooks,
allowing for complex adaptive behavior during solving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .base import SolverHooks

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfgarchon.types import SpatialTemporalState


@dataclass
class ControlState:
    """
    Enhanced control state for complex solver control flow.

    This provides more information than simple string control signals.
    """

    action: str  # "continue", "stop", "restart", "pause", "adjust"
    reason: str  # Human-readable reason for the action
    metadata: dict[str, Any]  # Additional control data

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AdaptiveControlHook(SolverHooks):
    """
    Hook that adapts solver behavior based on runtime conditions.

    This hook can automatically adjust solver parameters, restart
    with different settings, or stop early based on sophisticated
    heuristics.

    Example:
        adaptive = AdaptiveControlHook()
        adaptive.add_convergence_rule(
            condition=lambda state: state.residual > 1e-2 and state.iteration > 50,
            action="restart",
            adjustment=lambda config: config.with_tolerance(config.tolerance * 10)
        )
        adaptive.add_stop_rule(
            condition=lambda state: state.residual < 1e-12,
            reason="Ultra-high accuracy achieved"
        )
    """

    def __init__(self) -> None:
        self.convergence_rules: list[dict[str, Any]] = []
        self.stop_rules: list[dict[str, Any]] = []
        self.adjustment_rules: list[dict[str, Any]] = []
        self.restart_count = 0
        self.max_restarts = 3

    def add_convergence_rule(
        self,
        condition: Callable[[SpatialTemporalState], bool],
        action: str,
        reason: str = "",
        adjustment: Callable | None = None,
    ):
        """Add a rule for convergence control."""
        self.convergence_rules.append(
            {"condition": condition, "action": action, "reason": reason, "adjustment": adjustment}
        )

    def add_stop_rule(self, condition: Callable[[SpatialTemporalState], bool], reason: str = "Stop condition met"):
        """Add a rule for early stopping."""
        self.stop_rules.append({"condition": condition, "reason": reason})

    def add_adjustment_rule(
        self,
        condition: Callable[[SpatialTemporalState], bool],
        adjustment: Callable,
        reason: str = "Parameter adjustment",
    ):
        """Add a rule for parameter adjustment."""
        self.adjustment_rules.append({"condition": condition, "adjustment": adjustment, "reason": reason})

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        """Apply adaptive control rules."""

        # Check stop rules first
        for rule in self.stop_rules:
            if rule["condition"](state):
                print(f"Stopping early: {rule['reason']}")
                return "stop"

        # Check convergence override rules
        for rule in self.convergence_rules:
            if rule["condition"](state):
                action = rule["action"]
                reason = rule["reason"] or "Convergence rule triggered"

                if action == "restart" and self.restart_count < self.max_restarts:
                    self.restart_count += 1
                    print(f"Restarting ({self.restart_count}/{self.max_restarts}): {reason}")
                    return "restart"
                elif action == "stop":
                    print(f"Stopping: {reason}")
                    return "stop"

        return None

    def on_convergence_check(self, state: SpatialTemporalState) -> bool | None:
        """Apply convergence override logic."""

        # Check if any convergence rules should override default behavior
        for rule in self.convergence_rules:
            if rule["condition"](state):
                action = rule["action"]
                if action == "force_converge":
                    return True
                elif action == "force_continue":
                    return False

        return None


class PerformanceControlHook(SolverHooks):
    """
    Hook that monitors performance and adjusts solver behavior accordingly.

    This hook tracks solving performance metrics and can automatically
    switch to faster/slower methods based on progress.

    Example:
        perf_control = PerformanceControlHook()
        perf_control.set_stagnation_limit(10)  # Restart if no progress for 10 iterations
        perf_control.set_slow_progress_threshold(0.1)  # Switch to faster method if slow
    """

    def __init__(self) -> None:
        self.residual_history: list[float] = []
        self.stagnation_limit = 20
        self.slow_progress_threshold = 0.01
        self.fast_progress_threshold = 0.5
        self.last_significant_improvement = 0

    def set_stagnation_limit(self, iterations: int):
        """Set number of iterations without progress before restart."""
        self.stagnation_limit = iterations

    def set_slow_progress_threshold(self, threshold: float):
        """Set threshold for detecting slow progress."""
        self.slow_progress_threshold = threshold

    def _calculate_progress_rate(self, state: SpatialTemporalState) -> float:
        """Calculate recent progress rate."""
        if len(self.residual_history) < 5:
            return 1.0  # Assume good progress initially

        recent_residuals = self.residual_history[-5:]
        if recent_residuals[0] <= 0:
            return 0.0

        # Calculate relative improvement over last 5 iterations
        improvement = (recent_residuals[0] - recent_residuals[-1]) / recent_residuals[0]
        return max(improvement, 0.0)

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        """Monitor performance and suggest control actions."""
        self.residual_history.append(state.residual)

        # Calculate progress metrics
        progress_rate = self._calculate_progress_rate(state)

        # Detect stagnation
        if progress_rate < 1e-6:  # Essentially no progress
            stagnation_period = state.iteration - self.last_significant_improvement
            if stagnation_period > self.stagnation_limit:
                print(f"Stagnation detected: no progress for {stagnation_period} iterations")
                return "restart"
        else:
            self.last_significant_improvement = state.iteration

        # Monitor progress rate for method adjustment recommendations
        if progress_rate < self.slow_progress_threshold:
            print(f"Slow progress detected (rate: {progress_rate:.2e}), consider faster method")
        elif progress_rate > self.fast_progress_threshold:
            print(f"Fast progress detected (rate: {progress_rate:.2e}), could use more accurate method")

        return None


class WatchdogHook(SolverHooks):
    """
    Hook that monitors for problematic solver behavior and takes corrective action.

    This hook watches for common numerical issues like:
    - Residual explosion (divergence)
    - Oscillatory behavior
    - Abnormal state values (NaN, Inf)
    - Memory usage issues

    Example:
        watchdog = WatchdogHook()
        watchdog.set_divergence_threshold(1e3)
        watchdog.enable_nan_detection(True)
        watchdog.enable_oscillation_detection(True)
    """

    def __init__(self):
        self.divergence_threshold = 1e6
        self.nan_detection_enabled = True
        self.oscillation_detection_enabled = True
        self.oscillation_window = 10
        self.memory_limit_gb = None

    def set_divergence_threshold(self, threshold: float):
        """Set threshold for detecting divergence."""
        self.divergence_threshold = threshold

    def enable_nan_detection(self, enabled: bool = True):
        """Enable/disable NaN detection."""
        self.nan_detection_enabled = enabled

    def enable_oscillation_detection(self, enabled: bool = True, window: int = 10):
        """Enable/disable oscillation detection."""
        self.oscillation_detection_enabled = enabled
        self.oscillation_window = window

    def set_memory_limit(self, limit_gb: float):
        """Set memory usage limit in GB."""
        self.memory_limit_gb = limit_gb

    def _check_for_nans(self, state: SpatialTemporalState) -> bool:
        """Check if state contains NaN values."""
        import numpy as np

        return bool(np.isnan(state.u).any() or np.isnan(state.m).any() or np.isnan(state.residual))

    def _check_memory_usage(self) -> float:
        """Check current memory usage in GB."""
        import psutil

        return psutil.Process().memory_info().rss / (1024**3)  # Convert to GB

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        """Monitor for problematic behavior."""

        # Check for divergence
        if state.residual > self.divergence_threshold:
            print(f"Divergence detected: residual {state.residual:.2e} > threshold {self.divergence_threshold:.2e}")
            return "restart"

        # Check for NaN values
        if self.nan_detection_enabled and self._check_for_nans(state):
            print("NaN values detected in solution")
            return "restart"

        # Check memory usage
        if self.memory_limit_gb is not None:
            memory_usage = self._check_memory_usage()
            if memory_usage > self.memory_limit_gb:
                print(f"Memory limit exceeded: {memory_usage:.1f}GB > {self.memory_limit_gb:.1f}GB")
                return "stop"

        return None


class ConditionalStopHook(SolverHooks):
    """
    Hook that provides flexible stopping criteria based on user-defined conditions.

    Example:
        stop_hook = ConditionalStopHook()

        # Stop if high accuracy achieved
        stop_hook.add_condition(
            lambda state: state.residual < 1e-10,
            "Ultra-high accuracy achieved"
        )

        # Stop if taking too long
        stop_hook.add_condition(
            lambda state: state.iteration > 1000,
            "Maximum iterations reached"
        )
    """

    def __init__(self) -> None:
        self.conditions: list[dict[str, Any]] = []

    def add_condition(
        self, condition: Callable[[SpatialTemporalState], bool], reason: str = "Stop condition met", priority: int = 0
    ):
        """Add a stopping condition."""
        self.conditions.append({"condition": condition, "reason": reason, "priority": priority})
        # Sort by priority (higher priority first)
        self.conditions.sort(key=lambda x: x["priority"], reverse=True)

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        """Check all stopping conditions."""
        for cond in self.conditions:
            if cond["condition"](state):
                print(f"Stopping: {cond['reason']}")
                return "stop"
        return None

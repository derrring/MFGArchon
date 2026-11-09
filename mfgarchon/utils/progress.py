#!/usr/bin/env python3
"""
Progress Monitoring Utilities for MFGarchon

Provides elegant progress bars, timing, and performance monitoring for long-running
solver operations using rich.

Usage:
    # Simple tqdm-like interface
    from mfgarchon.utils.progress import tqdm, trange
    for i in tqdm(range(100), desc="Processing"):
        ...

    # Advanced rich features (panels, tables, live displays)
    from mfgarchon.utils.progress import console, Progress, Panel, Table
    with Progress() as progress:
        task = progress.add_task("Solving...", total=100)
        ...

    # Solver-specific utilities
    from mfgarchon.utils.progress import solver_progress, SolverTimer
"""

from __future__ import annotations

import contextvars
import functools
import os
import sys
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

if TYPE_CHECKING:
    from collections.abc import Iterator

T = TypeVar("T")

# Context variable for hierarchical progress routing (Issue #934).
# When a HierarchicalProgress is active, create_progress_bar() auto-routes
# to it as a transient subtask instead of creating a standalone bar.
_progress_context: contextvars.ContextVar[Any] = contextvars.ContextVar("_progress_context", default=None)


# =============================================================================
# Terminal Detection Utilities (Issue #709)
# =============================================================================

# Whitelisted terminals that support Rich progress bars but may be misdetected
_TERMINAL_WHITELIST = frozenset(
    {
        "vscode",  # VS Code integrated terminal
        "iTerm.app",  # iTerm2 on macOS
        "Apple_Terminal",  # macOS Terminal.app
        "Hyper",  # Hyper terminal
        "Alacritty",  # Alacritty terminal
    }
)

# Environments that capture output (Rich progress bars won't render properly)
# These use text-based logging by default, but can be overridden with FORCE_COLOR=1
_CAPTURED_OUTPUT_ENVS = frozenset(
    {
        "CLAUDECODE",  # Claude Code CLI
        "JUPYTER_RUNTIME",  # Jupyter notebooks
        "CI",  # CI/CD pipelines
    }
)


def is_captured_output_env() -> bool:
    """Check if running in an environment that captures output (Claude Code, CI, Jupyter)."""
    return any(os.environ.get(env) for env in _CAPTURED_OUTPUT_ENVS)


def is_interactive_terminal() -> bool:
    """
    Detect if running in an interactive terminal that supports Rich progress bars.

    Uses a multi-layered detection strategy:
    1. NO_COLOR=1: Force disable (standard convention)
    2. FORCE_COLOR=1: Force enable (explicit override)
    3. Captured output environments (Claude Code, CI): Disable by default
    4. Terminal whitelist (iTerm2, VS Code, etc.): Enable if isatty()
    5. Standard TTY detection (sys.stdout.isatty())

    Returns:
        True if Rich progress bars should be enabled, False for text-based logging.

    Environment Variables:
        NO_COLOR=1: Force disable Rich progress bars (takes precedence)
        FORCE_COLOR=1: Force enable Rich progress bars
        CLAUDECODE=1: Detected as captured output environment

    Note:
        Claude Code captures stdout for display, so Rich progress bars won't
        render properly by default. Use FORCE_COLOR=1 to override if needed.

    Example:
        >>> if is_interactive_terminal():
        ...     # Use Rich progress bars
        ... else:
        ...     # Use text-based logging
    """
    # NO_COLOR takes precedence (standard convention)
    if os.environ.get("NO_COLOR"):
        return False

    # FORCE_COLOR explicit override
    if os.environ.get("FORCE_COLOR") == "1":
        return True

    # Captured output environments default to text mode
    # (Claude Code, CI/CD, Jupyter - output is captured, not rendered)
    if is_captured_output_env():
        return False

    # Check terminal whitelist
    term_program = os.environ.get("TERM_PROGRAM", "")
    if term_program in _TERMINAL_WHITELIST:
        # Even if whitelisted, respect actual pipe/redirect
        # (e.g., iTerm2 but output redirected to file)
        return sys.stdout.isatty()

    # Standard TTY detection
    return sys.stdout.isatty()


def _format_metrics(metrics: dict[str, Any]) -> str:
    """Format metrics dict as compact display string.

    Shared by HierarchicalProgress and _TransientSubtask.
    Automatically uses scientific notation for very small or large values.

    Args:
        metrics: Dictionary of metric name -> value pairs

    Returns:
        Formatted string like "error_U=1.20e-04 | error_M=3.50e-05"
    """
    if not metrics:
        return ""
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            if value != 0 and (abs(value) < 1e-3 or abs(value) > 1e3):
                formatted = f"{value:.2e}"
            else:
                formatted = f"{value:.4f}"
        else:
            formatted = str(value)
        parts.append(f"{key}={formatted}")
    return " | ".join(parts)


def should_show_progress(explicit: bool | None = None) -> bool:
    """Centralized decision: should progress bars be shown?

    Use this instead of hardcoding show_progress=True in solver code.
    Respects terminal detection, environment variables, and explicit overrides.

    Args:
        explicit: User override.
            - None: Auto-detect via is_interactive_terminal()
            - True: Force show (interactive notebook, explicit user request)
            - False: Force hide (called from coupling loop, background job)

    Returns:
        True if Rich progress bars should be rendered.

    Example:
        >>> def solve(self, ..., show_progress: bool | None = None):
        ...     verbose = should_show_progress(show_progress)
        ...     timestep_range = create_progress_bar(range(Nt), verbose=verbose)
    """
    if explicit is not None:
        return explicit
    return is_interactive_terminal()


def get_console(force_terminal: bool | None = None) -> Console:
    """
    Get a Rich Console with smart terminal detection.

    Args:
        force_terminal: Override terminal detection.
            - None: Auto-detect using is_interactive_terminal()
            - True: Force enable terminal features
            - False: Force disable terminal features

    Returns:
        Configured Rich Console instance.

    Example:
        >>> console = get_console()
        >>> console.print("[bold green]Success![/]")
    """
    if force_terminal is None:
        force_terminal = is_interactive_terminal()
    return Console(force_terminal=force_terminal)


# Default console with smart detection
console = get_console()

__all__ = [
    # Terminal detection (Issue #709, #934)
    "is_interactive_terminal",
    "is_captured_output_env",
    "should_show_progress",
    "get_console",
    # Rich components
    "console",
    "Progress",
    "Panel",
    "Table",
    # tqdm-like interface
    "tqdm",
    "trange",
    "RichProgressBar",
    # Protocol pattern (Issue #587)
    "ProgressTracker",
    "NoOpProgressBar",
    "create_progress_bar",
    # Hierarchical progress (Issue #614)
    "HierarchicalProgress",
    # Solver utilities
    "SolverTimer",
    "IterationProgress",
    "solver_progress",
    "timed_operation",
    "time_solver_operation",
    "progress_context",
]


@runtime_checkable
class ProgressTracker(Protocol[T]):
    """
    Structural type for progress reporting (Issue #587).

    Defines the contract that all progress tracking implementations must satisfy.
    Enables compile-time type checking and eliminates runtime hasattr checks.

    All implementations must provide:
        - __iter__: Iterate over the tracked sequence
        - update_metrics: Display solver metrics (convergence errors, etc.)
        - log: Print informational messages

    Benefits:
        - Static type checking: Mypy can verify all method calls
        - No hasattr needed: Protocol guarantees method availability
        - Clear contract: Implementers know exactly what to provide
        - Polymorphism: verbose=True and verbose=False return same interface

    Example:
        progress: ProgressTracker[int] = create_progress_bar(range(100), verbose=True)
        for i in progress:
            progress.update_metrics(error=compute_error())
            if converged:
                progress.log("Converged!")
                break
    """

    def __iter__(self) -> Iterator[T]:
        """Iterate over the sequence being tracked."""
        ...

    def update_metrics(self, **kwargs: Any) -> None:
        """
        Update displayed metrics (e.g., convergence error, iteration count).

        Args:
            **kwargs: Metric key-value pairs to display

        Example:
            progress.update_metrics(error=1e-6, norm=0.5)
        """
        ...

    def log(self, message: str) -> None:
        """
        Print an informational message.

        Args:
            message: Message to display

        Example:
            progress.log("Solver converged!")
        """
        ...


class RichProgressBar:
    """
    Rich-based progress bar with tqdm-compatible interface.

    Provides familiar tqdm API while using rich's rendering.
    For advanced rich features, use Progress directly.
    """

    def __init__(
        self,
        iterable: Any = None,
        total: int | None = None,
        desc: str | None = None,
        disable: bool = False,
        **kwargs: Any,
    ):
        self.iterable = iterable
        self.total = total or (len(iterable) if iterable and hasattr(iterable, "__len__") else None)
        self.desc = desc or ""
        self.disable = disable
        self.n = 0
        self.progress: Progress | None = None
        self.task_id: int | None = None
        self.postfix_data: dict[str, Any] = {}
        self._started = False

    def __iter__(self) -> Iterator[Any]:
        if self.iterable is None:
            return iter([])

        if self.disable:
            yield from self.iterable
            return

        # Auto-start progress for iteration
        with self:
            for item in self.iterable:
                yield item
                self.update(1)

    def __enter__(self) -> RichProgressBar:
        if not self.disable:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TextColumn("({task.completed}/{task.total})"),
                TimeElapsedColumn(),
                TextColumn("ETA:"),
                TimeRemainingColumn(),
                console=console,
                transient=False,
            )
            self.progress.__enter__()
            self.task_id = self.progress.add_task(self.desc, total=self.total or 100)
            self._started = True
        return self

    def __exit__(self, *args: Any) -> None:
        if self.progress:
            self.progress.__exit__(*args)
            self._started = False

    def update(self, n: int = 1) -> None:
        """Advance progress by n steps."""
        self.n += n
        if not self.disable and self.progress and self.task_id is not None:
            desc = self._format_description()
            self.progress.update(self.task_id, advance=n, description=desc)

    def set_postfix(self, **kwargs: Any) -> None:
        """
        Set postfix values to display.

        .. deprecated:: 0.17.0
            Use :meth:`update_metrics` instead. This method is kept for
            backward compatibility but will be removed in v1.0.0.
        """
        self.update_metrics(**kwargs)

    def update_metrics(self, **kwargs: Any) -> None:
        """
        Update displayed metrics (Issue #587 Protocol API).

        Replaces set_postfix() with clearer semantics. Formats floats
        using scientific notation for solver metrics.

        Args:
            **kwargs: Metric key-value pairs to display

        Example:
            progress.update_metrics(error=1.5e-6, iteration=42)
        """
        self.postfix_data.update(kwargs)
        if not self.disable and self.progress and self.task_id is not None:
            desc = self._format_description()
            self.progress.update(self.task_id, description=desc)

    def log(self, message: str) -> None:
        """
        Print an informational message (Issue #587 Protocol API).

        Displays message above the progress bar using Rich console.
        Safe to call whether progress bar is active or disabled.

        Args:
            message: Message to display

        Example:
            progress.log("Convergence achieved!")
        """
        if not self.disable:
            console.print(f"[blue]INFO[/]: {message}")

    def set_description(self, desc: str) -> None:
        """Update the progress bar description."""
        self.desc = desc
        if not self.disable and self.progress and self.task_id is not None:
            self.progress.update(self.task_id, description=self._format_description())

    def _format_description(self) -> str:
        """Format description with postfix data."""
        if self.postfix_data:
            postfix_str = ", ".join(f"{k}={v}" for k, v in self.postfix_data.items())
            return f"{self.desc} [{postfix_str}]"
        return self.desc

    def close(self) -> None:
        """Close the progress bar."""
        if self.progress and self._started:
            self.progress.stop()
            self._started = False


class NoOpProgressBar:
    """
    Null Object Pattern for silent progress tracking (Issue #587).

    Provides the same interface as RichProgressBar but with zero behavior.
    Used when verbose=False to eliminate conditional logic in solver code.

    Key Properties:
        - Zero performance overhead: All methods are pass/yield statements
        - Type-safe: Satisfies ProgressTracker Protocol
        - Same interface: Solvers work identically with RichProgressBar or NoOpProgressBar
        - No conditionals needed: progress.update_metrics() always works

    Example:
        progress = NoOpProgressBar(range(100))
        for i in progress:
            progress.update_metrics(error=1e-5)  # Does nothing, no overhead
            progress.log("Done!")  # Does nothing
    """

    def __init__(
        self,
        iterable: Any = None,
        total: int | None = None,
        desc: str | None = None,
        disable: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize no-op progress bar.

        Args:
            iterable: Sequence to iterate over (passed through unchanged)
            total: Ignored (for API compatibility)
            desc: Ignored (for API compatibility)
            disable: Ignored (always disabled)
            **kwargs: Ignored (for API compatibility)
        """
        self.iterable = iterable

    def __iter__(self) -> Iterator[Any]:
        """Pass through iteration with zero overhead."""
        if self.iterable is None:
            return iter([])
        yield from self.iterable

    def __enter__(self) -> NoOpProgressBar:
        """No-op context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """No-op context manager exit."""

    def update(self, n: int = 1) -> None:
        """No-op update (Protocol compatibility)."""

    def update_metrics(self, **kwargs: Any) -> None:
        """No-op metrics update (Protocol API - Issue #587)."""

    def log(self, message: str) -> None:
        """No-op logging (Protocol API - Issue #587)."""

    def set_postfix(self, **kwargs: Any) -> None:
        """No-op set_postfix (backward compatibility)."""

    def set_description(self, desc: str) -> None:
        """No-op set_description (backward compatibility)."""

    def close(self) -> None:
        """No-op close (backward compatibility)."""


# =============================================================================
# Hierarchical Progress (Issue #614)
# =============================================================================


class HierarchicalProgress:
    """
    Universal hierarchical progress manager with dynamic metrics (Issue #614).

    Supports nested task structures where main tasks persist and subtasks
    are transient (disappear after completion). Designed for multi-level
    solver loops like MFG Picard iteration with inner HJB/FP solves.

    Key Features:
        - Dynamic metrics via **kwargs (not hardcoded field names)
        - Transient subtasks that auto-remove on completion
        - Null-object pattern for verbose=False (no conditionals needed)
        - Auto-formatting of scientific notation for small/large values
        - Reentrant support: Can be passed to child solvers (Optional Injection pattern)

    Design Pattern: Optional Injection with Default Fallback
        - Standalone mode: Create own Progress, own the console
        - Coupled mode: Reuse passed manager, add subtasks to existing display

    Supported Use Cases:
        - MFG Picard iteration (error_U, error_M)
        - Hyperparameter grid search (alpha, lr, loss)
        - Training loops (epoch, loss, accuracy)
        - Data processing pipelines (no metrics)

    Example (Standalone - HJB solver alone):
        with HierarchicalProgress() as progress:
            task = progress.add_task("HJB Solver", total=100)
            for t in range(100):
                # ... solve ...
                progress.update(task, error=err)

    Example (Coupled - Picard with HJB/FP):
        with HierarchicalProgress() as progress:
            picard_task = progress.add_task("MFG Picard", total=100)

            for i in range(100):
                # Pass progress to inner solver
                hjb_solver.solve(..., progress_manager=progress)
                fp_solver.solve(..., progress_manager=progress)
                progress.update(picard_task, error_U=err_u, error_M=err_m)

    Example (Child solver receiving progress_manager):
        def solve(self, ..., progress_manager: HierarchicalProgress | None = None):
            is_standalone = (progress_manager is None)
            ctx = progress_manager if not is_standalone else HierarchicalProgress(verbose=self.verbose)

            with ctx:
                # transient=True in coupled mode, False in standalone
                with ctx.transient_subtask("HJB", total=Nt, transient=not is_standalone) as task:
                    for t in range(Nt):
                        task.advance()
                        task.update_metrics(error=err)

    Output (Coupled):
        MFG Picard ━━━━━━━━━━╺━━━━━━━ 45% 0:01:23 error_U=1.20e-04 | error_M=3.50e-05
          └─ HJB ━━━━━━━━━━━━━━━━━━━━━━ 60%              (disappears after completion)
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize hierarchical progress manager.

        Args:
            verbose: If True, display progress bars; if False, all operations are no-ops
        """
        self.verbose = verbose
        self._progress: Progress | None = None
        self._entry_count: int = 0  # Reentrant counter for nested context managers
        self._task_metrics: dict[int, dict[str, Any]] = {}  # Accumulated metrics per task
        self._ctx_token: contextvars.Token[Any] | None = None

    def __enter__(self) -> HierarchicalProgress:
        """
        Start the progress display (if not already active).

        Reentrant behavior:
            - First entry (standalone): Creates and starts Progress display
            - Subsequent entries (coupled): Increments counter, reuses existing display
        """
        self._entry_count += 1

        if self.verbose and self._entry_count == 1:
            # First entry - create and start display
            self._progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TextColumn("{task.fields[metrics_str]}"),  # Dynamic metrics slot
                console=console,
                transient=False,
            )
            self._progress.start()
            # Register as active context for create_progress_bar auto-routing
            self._ctx_token = _progress_context.set(self)
        # Else: Already active (passed from parent), reuse display
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Stop the progress display (only on final exit).

        Reentrant behavior:
            - Decrements entry counter
            - Only stops display when counter reaches 0 (original entry exits)
        """
        self._entry_count -= 1

        if self._progress and self._entry_count == 0:
            # Reset context before stopping display
            if self._ctx_token is not None:
                _progress_context.reset(self._ctx_token)
                self._ctx_token = None
            self._progress.stop()
            self._progress = None

    @property
    def is_active(self) -> bool:
        """Check if progress display is currently active."""
        return self._entry_count > 0

    def add_task(self, description: str, total: int, **initial_metrics: Any) -> int | None:
        """
        Add a persistent task with optional initial metrics.

        Args:
            description: Task label (e.g., "MFG Picard", "Grid Search", "Training")
            total: Total number of iterations
            **initial_metrics: Any key-value pairs to display (e.g., error_U=0.0, lr=0.001)

        Returns:
            Task ID for updates, or None if verbose=False

        Example:
            task = progress.add_task("Picard", total=100, error_U=0.0, error_M=0.0)
        """
        if not self._progress:
            return None
        metrics_str = _format_metrics(initial_metrics)
        return self._progress.add_task(description, total=total, metrics_str=metrics_str)

    def update(self, task_id: int | None, advance: int = 1, **metrics: Any) -> None:
        """
        Update task progress and metrics.

        Args:
            task_id: Task to update (from add_task)
            advance: Steps to advance (default 1)
            **metrics: Updated metrics to display

        Example:
            progress.update(task, error_U=1.2e-4, error_M=3.5e-5)
            progress.update(task, advance=0, loss=0.023)  # Update metrics only
        """
        if not self._progress or task_id is None:
            return
        updates: dict[str, Any] = {"advance": advance}
        if metrics:
            if task_id not in self._task_metrics:
                self._task_metrics[task_id] = {}
            self._task_metrics[task_id].update(metrics)
            updates["metrics_str"] = _format_metrics(self._task_metrics[task_id])
        self._progress.update(task_id, **updates)

    def subtask(self, description: str, total: int) -> _TransientSubtask | _NullSubtask:
        """
        Create a transient subtask that disappears after completion.

        Subtasks appear indented below the main task and are automatically
        removed when the context manager exits (even on exception).

        Args:
            description: Subtask label (will be prefixed with "└─")
            total: Total number of iterations

        Returns:
            Context manager for the subtask

        Example:
            with progress.subtask("FP Solver", total=100) as fp:
                for t in range(100):
                    # ... compute ...
                    fp.advance()
            # Subtask automatically removed here
        """
        if not self._progress:
            return _NullSubtask()
        return _TransientSubtask(self._progress, f"  └─ {description}", total, transient=True)

    def transient_subtask(
        self, description: str, total: int, transient: bool = True
    ) -> _TransientSubtask | _NullSubtask:
        """
        Create a subtask with controllable transient behavior.

        Use this when you need control over whether the subtask disappears
        after completion. Useful for the "Optional Injection" pattern where:
        - Coupled mode (transient=True): Subtask disappears after completion
        - Standalone mode (transient=False): Task persists for user visibility

        Args:
            description: Subtask label (will be prefixed with "└─" if transient)
            total: Total number of iterations
            transient: If True, remove after completion; if False, persist

        Returns:
            Context manager for the subtask

        Example (inner solver with optional injection):
            def solve(self, ..., progress_manager=None):
                is_standalone = (progress_manager is None)
                ctx = progress_manager or HierarchicalProgress(verbose=True)
                with ctx:
                    with ctx.transient_subtask("HJB", total=Nt, transient=not is_standalone) as t:
                        for step in range(Nt):
                            t.advance()
        """
        if not self._progress:
            return _NullSubtask()
        # Only indent if transient (child task visual style)
        prefix = "  └─ " if transient else ""
        return _TransientSubtask(self._progress, f"{prefix}{description}", total, transient=transient)


class _TransientSubtask:
    """
    Subtask with optional auto-remove on completion.

    Used internally by HierarchicalProgress.subtask() and transient_subtask().

    Behavior controlled by `transient` parameter:
        - transient=True: Remove from display on __exit__ (for coupled mode)
        - transient=False: Keep in display after completion (for standalone mode)

    Thread Safety:
        Not thread-safe. Designed for single-threaded solver loops.
    """

    def __init__(self, progress: Progress, description: str, total: int, transient: bool = True):
        """
        Initialize subtask.

        Args:
            progress: Parent Progress instance
            description: Subtask description (already formatted with indent if needed)
            total: Total iterations
            transient: If True, remove on exit; if False, persist
        """
        self._progress = progress
        self._description = description
        self._total = total
        self._task_id: int | None = None
        self._transient = transient
        self._metrics: dict[str, Any] = {}

    def __enter__(self) -> _TransientSubtask:
        """Add subtask to progress display."""
        self._task_id = self._progress.add_task(self._description, total=self._total, metrics_str="")
        return self

    def advance(self, steps: int = 1) -> None:
        """
        Advance subtask progress.

        Args:
            steps: Number of steps to advance (default 1)
        """
        if self._task_id is not None:
            self._progress.update(self._task_id, advance=steps)

    def update_metrics(self, **kwargs: Any) -> None:
        """
        Update subtask metrics display.

        Args:
            **kwargs: Metric key-value pairs to display

        Example:
            subtask.update_metrics(error=1.2e-5, t=0.5)
        """
        if self._task_id is not None:
            self._metrics.update(kwargs)
            self._progress.update(self._task_id, metrics_str=_format_metrics(self._metrics))

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Handle subtask completion.

        Behavior:
            - transient=True: Remove from display (for nested/coupled mode)
            - transient=False: Keep in display (for standalone mode)

        Always handles cleanup safely, even if an exception occurred.
        """
        try:
            if self._task_id is not None and self._transient:
                self._progress.remove_task(self._task_id)
        except Exception:
            # Don't mask the actual error with UI errors
            pass


class _NullSubtask:
    """
    No-op subtask for verbose=False mode.

    Provides the same interface as _TransientSubtask but with zero behavior.
    Eliminates the need for conditional checks in solver code.
    """

    def __enter__(self) -> _NullSubtask:
        """No-op context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """No-op context manager exit."""

    def advance(self, steps: int = 1) -> None:
        """No-op advance (for API compatibility)."""

    def update_metrics(self, **kwargs: Any) -> None:
        """No-op metrics update (protocol compliance with _TransientSubtask)."""


def get_progress(
    progress: HierarchicalProgress | None = None,
    *,
    show_progress: bool | None = None,
) -> tuple[HierarchicalProgress, bool]:
    """Resolve a progress manager for solver use.

    Implements the Optional Injection pattern: if a progress manager is
    injected, reuse it (coupled mode); otherwise create a new one (standalone).

    Args:
        progress: Injected progress manager from coupling iterator, or None.
        show_progress: Explicit show/hide override (used when progress=None).

    Returns:
        (manager, is_owned) where is_owned=True means the caller created it
        and must use it as a context manager.

    Example:
        mgr, owned = get_progress(progress, show_progress=show_progress)
        ctx = mgr if owned else contextlib.nullcontext(mgr)
        with ctx:
            with mgr.transient_subtask("HJB", total=Nt, transient=not owned) as task:
                for n in range(Nt):
                    task.advance()
    """
    if progress is not None:
        return progress, False
    return HierarchicalProgress(verbose=should_show_progress(show_progress)), True


class _ChildProgressBar:
    """ProgressTracker that routes to a parent HierarchicalProgress subtask.

    Created by ``create_progress_bar`` when a ``HierarchicalProgress`` context
    is active (Issue #934). Implements the same interface as ``RichProgressBar``
    so solver code works unchanged.

    The subtask is transient: it appears indented under the parent task and
    auto-removes when iteration completes.
    """

    def __init__(
        self,
        parent: HierarchicalProgress,
        iterable: Any,
        desc: str = "",
        total: int | None = None,
    ):
        self._parent = parent
        self._iterable = iterable
        self._desc = desc
        self._total = total
        self._subtask: _TransientSubtask | None = None

    def __iter__(self) -> Any:
        """Iterate with automatic subtask lifecycle."""
        self._subtask = self._parent.transient_subtask(self._desc, total=self._total or 0, transient=True)
        ctx = self._subtask.__enter__()
        try:
            for item in self._iterable:
                yield item
                ctx.advance()
        finally:
            self._subtask.__exit__(None, None, None)
            self._subtask = None

    def update_metrics(self, **kwargs: Any) -> None:
        """Route metrics to parent subtask."""
        if self._subtask is not None:
            self._subtask.update_metrics(**kwargs)

    def log(self, message: str) -> None:
        """Suppress log in child mode (parent controls display)."""


def create_progress_bar(
    iterable: Any,
    *,
    verbose: bool = True,
    desc: str = "",
    total: int | None = None,
) -> ProgressTracker[Any]:
    """
    Factory function ensuring consistent ProgressTracker return type (Issue #587).

    Automatically routes to a parent ``HierarchicalProgress`` when one is active
    in the current context (Issue #934). This enables zero-parameter progress
    coordination between coupling iterators and inner solvers.

    Routing logic:
        1. Parent context active + verbose → _ChildProgressBar (transient subtask)
        2. No parent + verbose=True → RichProgressBar (standalone)
        3. verbose=False → NoOpProgressBar (silent, regardless of context)

    Args:
        iterable: Sequence to iterate over
        verbose: If True, show progress bar; if False, silent pass-through.
            When False, overrides parent context (solver can still opt out).
        desc: Progress bar description (used as subtask label in child mode)
        total: Total iterations (auto-detected if iterable has __len__)

    Returns:
        ProgressTracker instance

    Example:
        progress = create_progress_bar(range(100), verbose=True, desc="Solving")
        for i in progress:
            progress.update_metrics(error=compute_error())  # Always works
            if converged:
                progress.log("Converged!")  # Always works
                break
    """
    if not verbose:
        return NoOpProgressBar(iterable, desc=desc, total=total, disable=True)

    # Check for parent HierarchicalProgress context (Issue #934)
    parent = _progress_context.get(None)
    if parent is not None and parent.verbose:
        effective_total = total
        if effective_total is None:
            try:
                effective_total = len(iterable)
            except TypeError:
                effective_total = 0
        return _ChildProgressBar(parent, iterable, desc=desc, total=effective_total)

    return RichProgressBar(iterable, desc=desc, total=total, disable=False)


# tqdm-like interface using rich
tqdm = RichProgressBar


def trange(n: int, **kwargs: Any) -> RichProgressBar:
    """Create progress bar over range(n)."""
    return tqdm(range(n), total=n, **kwargs)


class SolverTimer:
    """
    Context manager for timing solver operations with detailed statistics.

    Example:
        with SolverTimer("HJB solve") as timer:
            result = solver.solve()
        print(f"Took {timer.duration:.2f}s")
    """

    def __init__(self, description: str = "Operation", verbose: bool = True):
        self.description = description
        self.verbose = verbose
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.duration: float | None = None

    def __enter__(self) -> SolverTimer:
        self.start_time = time.perf_counter()
        if self.verbose:
            console.print(f"[bold blue]Starting {self.description}...[/]")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_time = time.perf_counter()
        self.duration = self.end_time - (self.start_time or 0)

        if self.verbose:
            formatted = self.format_duration()
            if exc_type is None:
                console.print(f"[bold green]SUCCESS:[/] {self.description} completed in {formatted}")
            else:
                console.print(f"[bold red]ERROR:[/] {self.description} failed after {formatted}")

    def format_duration(self) -> str:
        """Format duration in human-readable form."""
        if self.duration is None:
            return "Unknown"

        if self.duration < 1:
            return f"{self.duration * 1000:.1f}ms"
        elif self.duration < 60:
            return f"{self.duration:.2f}s"
        elif self.duration < 3600:
            minutes = int(self.duration // 60)
            seconds = self.duration % 60
            return f"{minutes}m {seconds:.1f}s"
        else:
            hours = int(self.duration // 3600)
            minutes = int((self.duration % 3600) // 60)
            seconds = self.duration % 60
            return f"{hours}h {minutes}m {seconds:.1f}s"


class IterationProgress:
    """
    Advanced progress tracking for iterative solvers with convergence monitoring.

    Example:
        with IterationProgress(100, "Newton iteration") as progress:
            for i in range(100):
                error = solver.step()
                progress.update(1, error=error)
                if error < tol:
                    break
    """

    def __init__(
        self,
        max_iterations: int,
        description: str = "Solver Iterations",
        update_frequency: int = 1,
        disable: bool = False,
    ):
        self.max_iterations = max_iterations
        self.description = description
        self.update_frequency = update_frequency
        self.disable = disable

        self.pbar: RichProgressBar | None = None
        self.start_time: float | None = None
        self.current_iteration = 0

    def __enter__(self) -> IterationProgress:
        if self.disable:
            return self

        self.start_time = time.perf_counter()
        self.pbar = tqdm(
            total=self.max_iterations,
            desc=self.description,
            disable=self.disable,
        )
        self.pbar.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        if self.pbar:
            self.pbar.__exit__(*args)

    def update(
        self,
        n: int = 1,
        error: float | None = None,
        additional_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Update progress with optional convergence information.

        Args:
            n: Number of iterations to advance
            error: Current error/residual value
            additional_info: Additional metrics to display
        """
        if self.disable or not self.pbar:
            return

        self.current_iteration += n

        # Update every update_frequency iterations or on final iteration
        if self.current_iteration % self.update_frequency == 0 or self.current_iteration >= self.max_iterations:
            postfix: dict[str, Any] = {}
            if error is not None:
                postfix["err"] = f"{error:.2e}"
            if additional_info:
                postfix.update(additional_info)

            self.pbar.update(n)
            if postfix:
                self.pbar.set_postfix(**postfix)

    def set_description(self, desc: str) -> None:
        """Update the progress bar description."""
        if self.pbar:
            self.pbar.set_description(desc)


def solver_progress(
    max_iterations: int,
    description: str = "Solver Progress",
    **kwargs: Any,
) -> IterationProgress:
    """
    Create progress tracker optimized for solver iterations.

    Args:
        max_iterations: Maximum number of iterations
        description: Progress description
        **kwargs: Additional arguments for IterationProgress

    Returns:
        Configured IterationProgress instance
    """
    return IterationProgress(
        max_iterations=max_iterations,
        description=description,
        update_frequency=max(1, max_iterations // 100),  # Update every 1%
        **kwargs,
    )


def timed_operation(description: str | None = None, verbose: bool = True):
    """
    Decorator for timing function calls.

    Args:
        description: Custom description for the operation
        verbose: Whether to print timing information
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            desc = description or f"Function '{func.__name__}'"

            with SolverTimer(desc, verbose=verbose) as timer:
                result = func(*args, **kwargs)

            # Add timing info to result if it's a dictionary
            if isinstance(result, dict) and "execution_time" not in result:
                result["execution_time"] = timer.duration

            return result

        return wrapper

    return decorator


def time_solver_operation(func):
    """
    Decorator specifically for timing solver operations.

    Automatically adds execution time to solver results.
    """
    return timed_operation(description=f"Solver operation '{func.__name__}'", verbose=True)(func)


@contextmanager
def progress_context(
    iterable: Any,
    description: str = "Processing",
    disable: bool = False,
):
    """
    Context manager for easy progress tracking of iterables.

    Args:
        iterable: Iterable to track progress for
        description: Description to show
        disable: Whether to disable progress bar

    Yields:
        Progress bar wrapped iterable
    """
    try:
        total = len(iterable) if hasattr(iterable, "__len__") else None
    except (TypeError, AttributeError):
        total = None

    with tqdm(iterable, desc=description, total=total, disable=disable) as pbar:
        yield pbar

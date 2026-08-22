"""
Visualization Hooks for MFG Solvers

This module provides hooks for real-time and post-processing visualization
of solver progress and solutions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.mfg_logging.logger import MFGFormatter

from .base import SolverHooks

if TYPE_CHECKING:
    from mfgarchon.types import MFGResult, SpatialTemporalState


class PlottingHook(SolverHooks):
    """
    Real-time plotting of solver progress and solution evolution.

    Requires matplotlib for plotting functionality.

    Example:
        plotter = PlottingHook(plot_every=10, save_plots=True)
        result = solver.solve(problem, hooks=plotter)
    """

    def __init__(
        self,
        plot_every: int = 10,
        save_plots: bool = False,
        output_dir: str = "solver_plots",
        plot_convergence: bool = True,
        plot_solution: bool = True,
        interactive: bool = False,
    ):
        """
        Initialize plotting hook.

        Args:
            plot_every: Plot every N iterations
            save_plots: Whether to save plots to files
            output_dir: Directory to save plots
            plot_convergence: Whether to plot convergence history
            plot_solution: Whether to plot solution evolution
            interactive: Whether to use interactive plotting
        """
        self.plot_every = plot_every
        self.save_plots = save_plots
        self.output_dir = Path(output_dir)
        self.plot_convergence = plot_convergence
        self.plot_solution = plot_solution
        self.interactive = interactive

        self.residual_history: list[float] = []
        self.matplotlib_available = False

        # Try to import matplotlib
        try:
            import matplotlib.animation as animation
            import matplotlib.pyplot as plt

            self.plt = plt
            self.animation = animation
            self.matplotlib_available = True

            if interactive:
                plt.ion()  # Turn on interactive mode

            if save_plots:
                self.output_dir.mkdir(exist_ok=True)

        except ImportError:
            print("Warning: matplotlib not available for plotting")

    def _plot_convergence(self, iteration: int):
        """Plot convergence history."""
        if not self.matplotlib_available:
            return

        self.plt.figure(figsize=(10, 6))
        self.plt.semilogy(self.residual_history, "b-", linewidth=2)
        self.plt.xlabel("Iteration")
        self.plt.ylabel("Residual (log scale)")
        self.plt.title(f"Convergence History (Iteration {iteration})")
        self.plt.grid(True, alpha=0.3)

        if self.save_plots:
            filename = self.output_dir / f"convergence_iter_{iteration:04d}.png"
            self.plt.savefig(filename, dpi=150, bbox_inches="tight")

        if self.interactive:
            self.plt.show()
            self.plt.pause(0.01)
        else:
            self.plt.close()

    def _plot_solution(self, state: SpatialTemporalState):
        """Plot current solution state."""
        if not self.matplotlib_available:
            return

        # Create subplot layout
        fig, axes = self.plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Solution State - Iteration {state.iteration}", fontsize=16)

        # Plot value function u
        im1 = axes[0, 0].imshow(state.u, aspect="auto", cmap="viridis", origin="lower")
        axes[0, 0].set_title("Value Function u(t,x)")
        axes[0, 0].set_xlabel("Space")
        axes[0, 0].set_ylabel("Time")
        self.plt.colorbar(im1, ax=axes[0, 0])

        # Plot density function m
        im2 = axes[0, 1].imshow(state.m, aspect="auto", cmap="plasma", origin="lower")
        axes[0, 1].set_title("Density m(t,x)")
        axes[0, 1].set_xlabel("Space")
        axes[0, 1].set_ylabel("Time")
        self.plt.colorbar(im2, ax=axes[0, 1])

        # Plot final time profiles
        if hasattr(state, "metadata") and "x_grid" in state.metadata:
            x_grid = state.metadata["x_grid"]
            axes[1, 0].plot(x_grid, state.u[-1, :], "b-", label="u(T,x)", linewidth=2)
            axes[1, 0].plot(x_grid, state.m[-1, :], "r-", label="m(T,x)", linewidth=2)
            axes[1, 0].set_xlabel("x")
            axes[1, 0].set_title("Final Time Profiles")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Plot convergence history
        if self.residual_history:
            axes[1, 1].semilogy(self.residual_history, "g-", linewidth=2)
            axes[1, 1].set_xlabel("Iteration")
            axes[1, 1].set_ylabel("Residual (log scale)")
            axes[1, 1].set_title("Convergence History")
            axes[1, 1].grid(True, alpha=0.3)

        self.plt.tight_layout()

        if self.save_plots:
            filename = self.output_dir / f"solution_iter_{state.iteration:04d}.png"
            self.plt.savefig(filename, dpi=150, bbox_inches="tight")

        if self.interactive:
            self.plt.show()
            self.plt.pause(0.01)
        else:
            self.plt.close()

    def on_solve_start(self, initial_state: SpatialTemporalState) -> None:
        """Initialize plotting."""
        self.residual_history = [initial_state.residual]

        if self.plot_solution and self.matplotlib_available:
            self._plot_solution(initial_state)

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        """Update plots."""
        self.residual_history.append(state.residual)

        if state.iteration % self.plot_every == 0:
            if self.plot_convergence:
                self._plot_convergence(state.iteration)

            if self.plot_solution:
                self._plot_solution(state)

        return None

    def on_solve_end(self, result: MFGResult) -> MFGResult:
        """Create final plots."""
        if self.matplotlib_available:
            # Final convergence plot
            if self.plot_convergence:
                self._plot_convergence(result.iterations)

            # Final solution plot
            if self.plot_solution and hasattr(result, "U") and hasattr(result, "M"):
                # Create a final state for plotting
                final_state = type(
                    "State",
                    (),
                    {
                        "u": result.U,
                        "m": result.M,
                        "iteration": result.iterations,
                        "metadata": getattr(result, "metadata", {}),
                    },
                )()
                self._plot_solution(final_state)

            if self.interactive:
                self.plt.ioff()  # Turn off interactive mode

        return result


class AnimationHook(SolverHooks):
    """
    Create animations of solution evolution during solving.

    Example:
        animator = AnimationHook(animate_every=5, save_animation=True)
        result = solver.solve(problem, hooks=animator)
        animator.save_animation("solution_evolution.mp4")
    """

    def __init__(self, animate_every: int = 5, max_frames: int = 100, save_animation: bool = False):
        self.animate_every = animate_every
        self.max_frames = max_frames
        self.save_animation = save_animation

        self.frames_u: list[np.ndarray] = []
        self.frames_m: list[np.ndarray] = []
        self.frame_iterations: list[int] = []
        self.matplotlib_available = False

        # Try to import matplotlib
        try:
            import matplotlib.animation as animation
            import matplotlib.pyplot as plt

            self.plt = plt
            self.animation = animation
            self.matplotlib_available = True
        except ImportError:
            print("Warning: matplotlib not available for animation")

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        """Capture frames for animation."""
        if state.iteration % self.animate_every == 0 and len(self.frames_u) < self.max_frames:
            self.frames_u.append(state.u.copy())
            self.frames_m.append(state.m.copy())
            self.frame_iterations.append(state.iteration)

        return None

    def create_animation(self, filename: str = "solution_evolution.mp4"):
        """Create and save animation."""
        if not self.matplotlib_available or not self.frames_u:
            print("Cannot create animation: matplotlib unavailable or no frames captured")
            return

        fig, (ax1, ax2) = self.plt.subplots(1, 2, figsize=(15, 6))

        # Initial setup
        im1 = ax1.imshow(self.frames_u[0], aspect="auto", cmap="viridis", origin="lower")
        ax1.set_title("Value Function u(t,x)")
        ax1.set_xlabel("Space")
        ax1.set_ylabel("Time")
        self.plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(self.frames_m[0], aspect="auto", cmap="plasma", origin="lower")
        ax2.set_title("Density m(t,x)")
        ax2.set_xlabel("Space")
        ax2.set_ylabel("Time")
        self.plt.colorbar(im2, ax=ax2)

        def animate(frame_idx):
            iteration = self.frame_iterations[frame_idx]
            fig.suptitle(f"Solution Evolution - Iteration {iteration}", fontsize=16)

            im1.set_array(self.frames_u[frame_idx])
            im1.set_clim(vmin=self.frames_u[frame_idx].min(), vmax=self.frames_u[frame_idx].max())

            im2.set_array(self.frames_m[frame_idx])
            im2.set_clim(vmin=self.frames_m[frame_idx].min(), vmax=self.frames_m[frame_idx].max())

            return [im1, im2]

        anim = self.animation.FuncAnimation(
            fig, animate, frames=len(self.frames_u), interval=200, blit=False, repeat=True
        )

        if self.save_animation:
            print(f"Saving animation to {filename}...")
            anim.save(filename, writer="pillow", fps=5)
            print("Animation saved!")

        self.plt.show()
        return anim


class LoggingHook(SolverHooks):
    """
    Structured logging of solver progress and state information.

    Example:
        logger = LoggingHook(log_file="solver.log", log_level="INFO")
        result = solver.solve(problem, hooks=logger)
    """

    def __init__(self, log_file: str | None = None, log_level: str = "INFO", log_format: str = "detailed"):
        """
        Initialize logging hook.

        Args:
            log_file: Optional file to write logs to
            log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
            log_format: Format style ("simple", "detailed", "json")
        """
        self.log_file = log_file
        self.log_level = log_level
        self.log_format = log_format
        self.log_entries: list[dict[str, Any]] = []

        # Set up logging
        import json

        self.json = json
        self.logging_available = True

        # Set up logger using mfg_logging for consistent formatting
        self.logger = get_logger("MFGSolver")
        self.logger.setLevel(getattr(logging, log_level))

        # Set up handlers
        if log_file:
            handler = logging.FileHandler(log_file)
            # MFGFormatter, not a hand-rolled string: it is the format LogAnalyzer parses.
            # This line was `logging.Formatter("%(asctime)s - %(name)s - %(levelname)s -
            # %(message)s")`, which differs in two ways that both defeat the reader -- no
            # `datefmt`, so asctime carries milliseconds and the timestamp field never matches,
            # and no `-8s` padding, so the level field never matches either. Measured: a file
            # written that way yields 0 entries from LogAnalyzer at every level (#2058).
            formatter = MFGFormatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _log_entry(self, level: str, message: str, data: dict[str, Any] | None = None):
        """Create a log entry."""
        entry = {"level": level, "message": message, "data": data or {}}
        self.log_entries.append(entry)

        if self.logging_available:
            log_func = getattr(self.logger, level.lower(), self.logger.info)
            if self.log_format == "json":
                log_func(self.json.dumps(entry))
            else:
                log_func(message)

        # Also print to console for immediate feedback
        print(f"[{level}] {message}")

    def on_solve_start(self, initial_state: SpatialTemporalState) -> None:
        """Log solve start."""
        self._log_entry(
            "INFO",
            "Solver started",
            {
                "initial_residual": float(initial_state.residual),
                "u_shape": list(initial_state.u.shape),
                "m_shape": list(initial_state.m.shape),
            },
        )

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        """Log iteration completion."""
        if state.iteration % 10 == 0:  # Log every 10th iteration
            self._log_entry(
                "INFO",
                f"Iteration {state.iteration} completed",
                {
                    "iteration": state.iteration,
                    "residual": float(state.residual),
                    "u_min": float(np.min(state.u)),
                    "u_max": float(np.max(state.u)),
                    "m_min": float(np.min(state.m)),
                    "m_max": float(np.max(state.m)),
                },
            )

        return None

    def on_solve_end(self, result: MFGResult) -> MFGResult:
        """Log solve completion."""
        self._log_entry(
            "INFO",
            "Solver completed",
            {
                "converged": result.converged,
                "total_iterations": result.iterations,
                "final_residual": float(getattr(result, "final_residual", 0.0)),
            },
        )

        return result

    def export_log(self, filename: str):
        """Export log entries to file."""
        if self.log_format == "json":
            with open(filename, "w") as f:
                self.json.dump(self.log_entries, f, indent=2)
        else:
            with open(filename, "w") as f:
                for entry in self.log_entries:
                    f.write(f"[{entry['level']}] {entry['message']}\n")
                    if entry["data"]:
                        f.write(f"  Data: {entry['data']}\n")
                    f.write("\n")


class ProgressBarHook(SolverHooks):
    """
    Display a progress bar during solving.

    Requires tqdm for progress bar functionality.

    Example:
        progress = ProgressBarHook(max_iterations=100)
        result = solver.solve(problem, hooks=progress)
    """

    def __init__(self, max_iterations: int | None = None, update_every: int = 1):
        self.max_iterations = max_iterations
        self.update_every = update_every
        self.pbar = None
        self.tqdm_available = False

        # Import progress bar (requires rich)
        try:
            from mfgarchon.utils.progress import RichProgressBar

            self.progress_bar_class = RichProgressBar
            self.tqdm_available = True
        except ImportError:
            self.tqdm_available = False
            print("Warning: Progress bar not available (install rich)")

    def on_solve_start(self, initial_state: SpatialTemporalState) -> None:
        """Initialize progress bar."""
        if self.tqdm_available:
            desc = f"Solving MFG (residual: {initial_state.residual:.2e})"
            self.pbar = self.progress_bar_class(total=self.max_iterations, desc=desc, unit="iter", dynamic_ncols=True)

    def on_iteration_end(self, state: SpatialTemporalState) -> str | None:
        """Update progress bar."""
        if self.pbar and state.iteration % self.update_every == 0:
            self.pbar.set_description(f"Solving MFG (residual: {state.residual:.2e})")
            self.pbar.update(self.update_every)

        return None

    def on_solve_end(self, result: MFGResult) -> MFGResult:
        """Close progress bar."""
        if self.pbar:
            self.pbar.set_description(f"Completed ({'converged' if result.converged else 'not converged'})")
            self.pbar.close()

        return result

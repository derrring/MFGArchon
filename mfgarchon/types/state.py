"""
Internal State Representations

These are the types used internally by solvers and accessible
through the hooks system for advanced customization.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SpatialTemporalState(NamedTuple):
    """
    Internal state representation for iterative MFG solvers.

    This type is used in the hooks system to provide access to
    intermediate solution states during solving.

    Attributes:
        u: Value function array, shape (Nt+1, Nx+1)
        m: Density function array, shape (Nt+1, Nx+1)
        iteration: Current iteration number
        residual: Current residual/error measure
        metadata: Additional solver-specific data
    """

    u: NDArray  # Value function u(t,x)
    m: NDArray  # Density function m(t,x)
    iteration: int  # Current iteration number
    residual: float  # Current residual/error
    metadata: dict[str, Any]  # Additional solver-specific data

    def copy_with_updates(self, **kwargs) -> SpatialTemporalState:
        """Create a copy with updated fields."""
        return self._replace(**kwargs)

    def get_final_time_solution(self) -> tuple[NDArray, NDArray]:
        """Get solution at final time T."""
        return self.u[-1, :], self.m[-1, :]

    def get_initial_time_solution(self) -> tuple[NDArray, NDArray]:
        """Get solution at initial time t=0."""
        return self.u[0, :], self.m[0, :]

    def compute_l2_norm(self) -> float:
        """Compute L2 norm of current solution."""
        return float(np.sqrt(np.sum(self.u**2) + np.sum(self.m**2)))

    def __str__(self) -> str:
        return f"SpatialTemporalState(iteration={self.iteration}, residual={self.residual:.2e}, shape={self.u.shape})"


class ConvergenceInfo(NamedTuple):
    """Information about solver convergence."""

    converged: bool
    iterations: int
    final_residual: float
    residual_history: list[float]
    convergence_reason: str

    def plot_convergence(self) -> None:
        """Plot convergence history."""
        # The except covers the IMPORT only. Wrapping the body reported any ImportError raised
        # from inside matplotlib -- `plt.figure()` on a backend whose own dependency is missing
        # raises one -- as "Matplotlib not available", which is both wrong and unactionable: it
        # names the one package that IS installed. #2090.
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return

        plt.figure(figsize=(8, 6))
        plt.semilogy(self.residual_history)
        plt.xlabel("Iteration")
        plt.ylabel("Residual")
        plt.title("Convergence History")
        plt.grid(True)
        plt.show()


class SolverStatistics(NamedTuple):
    """Statistics collected during solving."""

    total_time: float
    average_iteration_time: float
    memory_usage_mb: float | None
    cpu_usage_percent: float | None

    def __str__(self) -> str:
        return f"SolverStatistics(total_time={self.total_time:.2f}s, avg_iter_time={self.average_iteration_time:.3f}s)"


# Additional type aliases for internal use
ResidualHistory = list[float]
IterationCallback = Callable[[SpatialTemporalState], str | None]

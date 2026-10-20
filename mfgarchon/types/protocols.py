"""
Core Type Protocols for MFGarchon

These protocols define the clean, simple interfaces that users interact with.
They use duck typing - any object that implements these methods will work,
regardless of inheritance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from numpy.typing import NDArray


@runtime_checkable
class MFGProblem(Protocol):
    """
    Protocol for MFG problem objects.

    Any object that implements these methods can be used as an MFG problem,
    regardless of its actual class hierarchy.
    """

    def get_domain_bounds(self) -> tuple[float, float]:
        """Get spatial domain bounds (xmin, xmax)."""
        ...

    def get_time_horizon(self) -> float:
        """Get time horizon T."""
        ...

    def evaluate_hamiltonian(self, x: float, p: float, m: float, t: float) -> float:
        """Evaluate Hamiltonian H(x, p, m, t)."""
        ...

    def get_initial_density(self) -> NDArray:
        """Get initial density m_0(x)."""
        ...

    def get_initial_value_function(self) -> NDArray:
        """Get initial value function u_0(x) (or terminal condition)."""
        ...


@runtime_checkable
class MFGSolver(Protocol):
    """
    Protocol for MFG solver objects.

    Any object that implements solve() can be used as a solver.
    """

    def solve(self, problem: MFGProblem, **kwargs) -> MFGResult:
        """
        Solve the MFG problem.

        Args:
            problem: MFG problem to solve
            **kwargs: Additional solver parameters

        Returns:
            Solution result
        """
        ...


@runtime_checkable
class MFGResult(Protocol):
    """
    Protocol for MFG solution results.

    Provides clean access to solutions and analysis methods.
    """

    @property
    def u(self) -> NDArray:
        """Value function u(t, x)."""
        ...

    @property
    def m(self) -> NDArray:
        """Density function m(t, x)."""
        ...

    @property
    def converged(self) -> bool:
        """Whether the solver converged."""
        ...

    @property
    def iterations(self) -> int:
        """Number of iterations performed."""
        ...

    def plot_solution(self, **kwargs) -> None:
        """Plot the solution (u and m)."""
        ...

    def export_data(self, filename: str) -> None:
        """Export solution data to file."""
        ...


@runtime_checkable
class SolverConfig(Protocol):
    """
    Protocol for solver configuration objects.

    Provides a clean interface for solver parameters.
    """

    @property
    def max_iterations(self) -> int:
        """Maximum number of iterations."""
        ...

    @property
    def tolerance(self) -> float:
        """Convergence tolerance."""
        ...

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a configuration parameter."""
        ...


# =============================================================================
# Subsystem Solver Protocols (Issue #543 Phase 2)
# =============================================================================


@runtime_checkable
class HJBSolverProtocol(Protocol):
    """
    Protocol for Hamilton-Jacobi-Bellman (HJB) equation solvers.

    This protocol defines the interface that HJB solvers must implement
    for use in coupling algorithms (Fictitious Play, Fixed Point Iteration).

    Issue #543 Phase 2: Eliminates hasattr checks like:
        if hasattr(self.hjb_solver, "solve_hjb_system"):

    Methods:
        solve(...): Main solve method (preferred modern API)
        solve_hjb_system(...): Legacy solve method (for backward compatibility)

    Note: Solvers should implement at least one of these methods. A solver handed to a coupling loop also carries
    ``problem``, the problem it was built for: the loop compares its geometry with the problem it solves, and refuses
    a solver without one by name (#2775). A solver whose DOFs sit at points of its own, not its geometry's, declares
    them as ``dof_points``, and the loop compares the two solvers' points.
    """

    def solve(
        self,
        m: NDArray,
        dt: float | None = None,
        **kwargs,
    ) -> NDArray:
        """
        Solve HJB equation given density m.

        Args:
            m: Density field m(t, x) at all time steps
            dt: Time step (optional, can be inferred from problem)
            **kwargs: Additional solver parameters

        Returns:
            Value function u(t, x) at all time steps

        Note: Modern API. Preferred over solve_hjb_system().
        """
        ...


@runtime_checkable
class FPSolverProtocol(Protocol):
    """
    Protocol for Fokker-Planck (FP) equation solvers.

    This protocol defines the interface that FP solvers must implement
    for use in coupling algorithms (Fictitious Play, Fixed Point Iteration).

    Issue #543 Phase 2: Eliminates hasattr checks like:
        if hasattr(self.fp_solver, "solve_fp_system"):

    Methods:
        solve(...): Main solve method (preferred modern API)
        solve_fp_system(...): Legacy solve method (for backward compatibility)

    Note: Solvers should implement at least one of these methods. A solver handed to a coupling loop also carries
    ``problem``, the problem it was built for: the loop compares its geometry with the problem it solves, and refuses
    a solver without one by name (#2775). A solver whose DOFs sit at points of its own, not its geometry's, declares
    them as ``dof_points``, and the loop compares the two solvers' points.
    """

    def solve(
        self,
        u: NDArray,
        dt: float | None = None,
        **kwargs,
    ) -> NDArray:
        """
        Solve Fokker-Planck equation given value function u.

        Args:
            u: Value function u(t, x) at all time steps
            dt: Time step (optional, can be inferred from problem)
            **kwargs: Additional solver parameters

        Returns:
            Density field m(t, x) at all time steps

        Note: Modern API. Preferred over solve_fp_system().
        """
        ...

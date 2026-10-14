"""
Policy iteration utilities for HJB-MFG problems.

This module provides helpers for applying policy iteration (Howard's algorithm)
to HJB equations in the context of Mean Field Games.

Policy iteration is particularly effective for:
- Optimal control problems with explicit control variables
- Problems where Hamiltonian is convex in control
- Cases where policy improvement can be computed cheaply

References:
- Howard, R. A. (1960). Dynamic Programming and Markov Processes
- Krylov, N. V. (2008). Controlled Diffusion Processes (Section on policy iteration)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from mfgarchon.utils.deprecation import NEVER_READ, retired_parameters

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class HJBPolicyProblem(Protocol):
    """
    Protocol for HJB problems that support policy iteration.

    A problem implementing this protocol can be solved using policy iteration.
    """

    def evaluate_policy(
        self,
        policy: NDArray,
        density: NDArray,
        U_terminal: NDArray,
        U_guess: NDArray,
    ) -> NDArray:
        """
        Evaluate policy by solving linear HJB equation.

        Given a fixed policy α(x,t), solve:
            ∂u/∂t + H(x, ∇u, α, m) = 0
            u(T, x) = u_T(x)

        Args:
            policy: Policy array, shape depends on problem dimension
            density: Density m(t,x), shape (Nt+1, ...)
            U_terminal: Terminal condition u_T(x)
            U_guess: Initial guess for u

        Returns:
            Value function u(t,x) for the given policy
        """
        ...

    def improve_policy(
        self,
        U: NDArray,
        density: NDArray,
    ) -> NDArray:
        """
        Improve policy by maximizing Hamiltonian.

        Given value function u(t,x), compute improved policy:
            α'(x,t) = argmax_a H(x, ∇u, a, m)

        Args:
            U: Current value function, shape (Nt+1, ...)
            density: Density m(t,x), shape (Nt+1, ...)

        Returns:
            Improved policy α'(x,t)
        """
        ...


def policy_iteration_hjb(
    problem: HJBPolicyProblem,
    density: NDArray,
    U_terminal: NDArray,
    policy_init: NDArray | None = None,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    verbose: bool = True,
) -> tuple[NDArray, NDArray, dict]:
    """
    Solve HJB equation using policy iteration (Howard's algorithm).

    Algorithm:
        1. Start with initial policy α₀
        2. Policy evaluation: Solve HJB with α_k fixed (linear PDE)
        3. Policy improvement: α_{k+1} = argmax_a H(x, ∇u_k, a, m)
        4. Repeat until ||α_{k+1} - α_k|| < tolerance

    Args:
        problem: Problem implementing HJBPolicyProblem protocol
        density: Density m(t,x), shape (Nt+1, ...)
        U_terminal: Terminal condition u_T(x)
        policy_init: Initial policy guess. If None, uses zero control.
        max_iterations: Maximum number of policy iterations
        tolerance: Convergence tolerance on policy change
        verbose: Print iteration progress

    Returns:
        (U, policy, info) where:
            - U: Value function u(t,x)
            - policy: Optimal policy α*(x,t)
            - info: Dictionary with convergence information
                - converged: bool
                - iterations: int
                - policy_errors: list of policy change norms
    """
    if verbose:
        print("Starting policy iteration for HJB equation")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Tolerance: {tolerance:.2e}")

    # Initialize policy
    if policy_init is None:
        # Zero control as initial guess
        policy = np.zeros_like(U_terminal)
        if len(policy.shape) > len(U_terminal.shape):
            # Handle case where policy is over time+space
            policy = np.zeros_like(density)
    else:
        policy = policy_init.copy()

    policy_errors = []
    U = None

    for iteration in range(max_iterations):
        # Policy Evaluation: Solve HJB with fixed policy
        U_guess = U if U is not None else np.zeros_like(density)
        U_new = problem.evaluate_policy(policy, density, U_terminal, U_guess)

        # Policy Improvement: Maximize Hamiltonian
        policy_new = problem.improve_policy(U_new, density)

        # Check convergence
        policy_change = np.linalg.norm(policy_new - policy)
        policy_errors.append(policy_change)

        if verbose:
            print(f"  Iteration {iteration + 1}: policy change = {policy_change:.2e}")

        # Update policy
        policy = policy_new
        U = U_new

        # Check convergence
        if policy_change < tolerance:
            if verbose:
                print(f"  Converged in {iteration + 1} iterations")
            return (
                U,
                policy,
                {
                    "converged": True,
                    "iterations": iteration + 1,
                    "policy_errors": policy_errors,
                },
            )

    if verbose:
        print(f"  Did not converge in {max_iterations} iterations")
        print(f"  Final policy change: {policy_errors[-1]:.2e}")

    return (
        U,
        policy,
        {
            "converged": False,
            "iterations": max_iterations,
            "policy_errors": policy_errors,
        },
    )


class LQPolicyIterationHelper:
    """
    Helper for applying policy iteration to Linear-Quadratic HJB problems.

    For LQ problems with Hamiltonian of the form:
        H(x, p, α, m) = 0.5 * |α|² + α·p + V(x, m)

    Where α is the control. The optimal control is:
        α*(x,t) = -p = -∇u

    This is a reference implementation showing how to structure
    policy iteration for HJB-MFG problems.
    """

    @retired_parameters(
        {"potential_func": "the policy evaluation uses the HJB solver's own problem; put V there (#2558)"},
        issue="#2648",
        since="v0.22.0",
        never_read=True,
    )
    def __init__(
        self,
        hjb_solver,
        potential_func: Callable[[float, float, float], float] | None = NEVER_READ,
    ):
        """
        Initialize LQ policy iteration helper.

        Args:
            hjb_solver: HJB solver instance (e.g., HJBFDMSolver)
            potential_func: Refused when passed (#2648): the policy evaluation uses the HJB solver's own
                problem; put V there.
        """
        self.hjb_solver = hjb_solver
        self.potential_func = potential_func

    def evaluate_policy_1d(
        self,
        policy: NDArray,
        density: NDArray,
        U_terminal: NDArray,
        U_guess: NDArray,
    ) -> NDArray:
        """
        Evaluate policy for 1D LQ problem.

        Solves: ∂u/∂t + 0.5*α² + α*∂u/∂x + V(x,m) = 0

        This is a linear PDE in u when α is fixed.

        Args:
            policy: Policy α(t,x), shape (Nt+1, Nx+1)
            density: Density m(t,x), shape (Nt+1, Nx+1)
            U_terminal: Terminal condition
            U_guess: Initial guess

        Returns:
            Value function u(t,x)
        """
        # For true policy iteration, we would solve a modified HJB
        # with α fixed. Here we demonstrate the structure.

        # In practice, this requires modifying the Hamiltonian:
        # H_fixed(x, p, m) = 0.5*α(x)² + α(x)*p + V(x, m)

        # For demonstration, use the existing solver with a note
        # that this is a simplified version
        print("Note: This is a demonstration. Full implementation requires")
        print("      solving the linearized HJB with fixed policy.")

        # Fallback: Use standard HJB solver
        U_result = self.hjb_solver.solve_hjb_system(density, U_terminal, U_guess)

        return U_result

    def improve_policy_1d(
        self,
        U: NDArray,
        density: NDArray,
    ) -> NDArray:
        """
        Improve policy for 1D LQ problem.

        For LQ Hamiltonian H = 0.5*α² + α*p + V(x,m):
            ∂H/∂α = α + p = 0  =>  α* = -p = -∂u/∂x

        Args:
            U: Current value function, shape (Nt+1, Nx+1)
            density: Density m(t,x), shape (Nt+1, Nx+1)

        Returns:
            Improved policy α*(t,x) = -∂u/∂x
        """
        Nt, Nx_plus_1 = U.shape
        Nx = Nx_plus_1 - 1
        dx = self.hjb_solver.problem.geometry.get_grid_spacing()[0]

        # Compute gradient ∂u/∂x using central differences
        policy_improved = np.zeros_like(U)

        for n in range(Nt):
            for i in range(Nx_plus_1):
                if i == 0:
                    # Forward difference at left boundary
                    dudx = (U[n, i + 1] - U[n, i]) / dx
                elif i == Nx:
                    # Backward difference at right boundary
                    dudx = (U[n, i] - U[n, i - 1]) / dx
                else:
                    # Central difference in interior
                    dudx = (U[n, i + 1] - U[n, i - 1]) / (2 * dx)

                # Optimal policy: α* = -∇u
                policy_improved[n, i] = -dudx

        return policy_improved


@retired_parameters(
    {"potential_func": "the policy evaluation uses the HJB solver's own problem; put V there (#2558)"},
    issue="#2648",
    since="v0.22.0",
    never_read=True,
)
def create_lq_policy_problem(
    hjb_solver,
    potential_func: Callable | None = NEVER_READ,
) -> HJBPolicyProblem:
    """
    Create a policy iteration problem from an HJB solver.

    This is a factory function that wraps an HJB solver to make it
    compatible with the policy iteration framework.

    Args:
        hjb_solver: HJB solver instance
        potential_func: Refused when passed (#2648): the policy evaluation uses the HJB solver's own problem;
            put V there.

    Returns:
        Object implementing HJBPolicyProblem protocol
    """
    helper = LQPolicyIterationHelper(hjb_solver, potential_func)

    class LQPolicyProblem:
        """Wrapper making HJB solver compatible with policy iteration."""

        def evaluate_policy(self, policy, density, U_terminal, U_guess):
            return helper.evaluate_policy_1d(policy, density, U_terminal, U_guess)

        def improve_policy(self, U, density):
            return helper.improve_policy_1d(U, density)

    return LQPolicyProblem()

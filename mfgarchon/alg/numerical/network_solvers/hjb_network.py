"""
HJB Solver for Networks/Graphs.

This module implements Hamilton-Jacobi-Bellman equation solvers
for Mean Field Games on network structures.

Mathematical formulation:
∂u/∂t + H_i(m, ∇_G u, t) = 0  at node i
u(T, i) = g(i)                  terminal condition

where:
- H_i: Hamiltonian at node i
- ∇_G: Discrete gradient on graph
- Network-specific boundary conditions

Key algorithms:
- Explicit time stepping on networks
- Implicit schemes with graph Laplacians
- Policy iteration for network control problems
- Value iteration on discrete state spaces
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver

if TYPE_CHECKING:
    from mfgarchon.extensions.topology import NetworkMFGProblem


class NetworkHJBSolver(BaseHJBSolver):
    """
    HJB solver for Mean Field Games on networks.

    Solves the discrete HJB equation:
    ∂u/∂t + H_i(m, ∇_G u, t) = 0

    with network-specific Hamiltonians and discrete operators.

    Required Geometry Traits (Issue #596 Phase 2.4):
        - SupportsGraphLaplacian: Discrete Laplacian L = D - A for diffusion operators
        - SupportsAdjacency: Adjacency matrix A and neighbor queries for connectivity

    Compatible Geometries:
        - NetworkGeometry (Grid, Random, ScaleFree, Custom networks)
        - MazeGeometry (2D grids with obstacles)
        - Any graph geometry implementing required traits

    Note:
        Uses trait-based graph operators for discrete differential equations on networks.
        Trait validation occurs at problem/geometry level.
    """

    def __init__(
        self,
        problem: NetworkMFGProblem,
        scheme: str | type = "RK45",
        tolerance: float = 1e-6,
    ):
        """
        Initialize network HJB solver.

        Args:
            problem: Network MFG problem instance
            scheme: Any ``scipy.integrate.solve_ivp`` method — either a name
                string ("RK45", "BDF", etc.) or an ``OdeSolver`` subclass
                for custom integrators. Default "RK45" (adaptive, O(dt^5)).
            tolerance: ODE solver tolerance (rtol for solve_ivp)
        """
        super().__init__(problem)

        self.network_problem = problem
        self.scheme = scheme
        self.tolerance = tolerance

        # Network properties
        self.num_nodes = problem.num_nodes
        self.adjacency_matrix = problem.get_adjacency_matrix()
        self.laplacian_matrix = problem.get_laplacian_matrix()

        # Time discretization
        self.dt = problem.T / problem.Nt
        self.times = np.linspace(0, problem.T, problem.Nt + 1)

        # Solver name
        scheme_name = scheme if isinstance(scheme, str) else scheme.__name__
        self.hjb_method_name = f"NetworkHJB_{scheme_name}"

        # Precompute neighbor lists for Hamiltonian evaluation
        self.gradient_ops: dict[int, list[int]] = {}
        for i in range(self.num_nodes):
            self.gradient_ops[i] = self.network_problem.get_node_neighbors(i)

    def solve_hjb_system(
        self,
        M_density: np.ndarray | None = None,
        U_terminal: np.ndarray | None = None,
        U_coupling_prev: np.ndarray | None = None,
        volatility_field: float | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Solve HJB system on network with given density evolution.

        Args:
            M_density: (Nt+1, num_nodes) density evolution from FP solver
            U_terminal: Terminal condition u(T, i)
            U_coupling_prev: Previous Picard iterate for coupling
            volatility_field: Diffusion coefficient (not yet used in network solver)

        Returns:
            (Nt+1, num_nodes) value function evolution
        """
        # Validate required parameters
        if M_density is None:
            raise ValueError("M_density is required")
        if U_terminal is None:
            raise ValueError("U_terminal is required")

        # Extract dimensions from input
        # M_density has shape (n_time_points, num_nodes) where n_time_points = problem.Nt + 1
        n_time_points = M_density.shape[0]
        return self._solve_ode(U_terminal, M_density, n_time_points)

    def _evaluate_hamiltonian_batch(self, u: np.ndarray, m: np.ndarray, t: float) -> np.ndarray:
        """Evaluate Hamiltonian at all nodes (Issue #960).

        Returns H_i for i = 0, ..., N-1 as a single array.
        Eliminates per-node Python loop from caller.
        """
        H = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            neighbors = self.gradient_ops[i]
            H[i] = self.network_problem.hamiltonian(i, neighbors, m, u, t)
        return H

    def _solve_ode(
        self,
        U_terminal: np.ndarray,
        M_density: np.ndarray,
        n_time_points: int,
    ) -> np.ndarray:
        """Solve HJB backward via scipy.integrate.solve_ivp (Issue #960).

        Reformulates the backward HJB system as an ODE:
            du/ds = H(u, m(T-s), T-s),  s in [0, T],  u(0) = U_terminal

        where s = T - t is the reversed time variable.

        Benefits over hand-coded Euler:
        - Adaptive time stepping (no CFL constraint)
        - Higher-order accuracy (RK45 = O(dt^5))
        - Stiff solvers available (BDF, Radau)
        """
        from scipy.integrate import solve_ivp

        T = self.network_problem.T

        def rhs(s, u_flat):
            # s is forward time in the reversed system: t = T - s
            t_physical = T - s
            # Interpolate density at physical time t
            t_idx = min(int(t_physical / self.dt), n_time_points - 1)
            m = M_density[t_idx, :]
            # HJB: du/dt + H = 0  =>  du/ds = H (sign flip from time reversal)
            return self._evaluate_hamiltonian_batch(u_flat, m, t_physical)

        sol = solve_ivp(
            rhs,
            [0, T],
            U_terminal,
            method=self.scheme,
            t_eval=np.linspace(0, T, n_time_points),
            rtol=self.tolerance,
            atol=self.tolerance * 0.1,
        )

        if not sol.success:
            import warnings

            warnings.warn(
                f"Network HJB ODE solver did not converge: {sol.message}",
                RuntimeWarning,
                stacklevel=2,
            )

        # sol.y shape: (num_nodes, n_time_points) — reversed time
        # Flip back to physical time: index 0 = t=0, index -1 = t=T
        U = sol.y.T[::-1]  # (n_time_points, num_nodes)

        # Ensure correct shape (solve_ivp may return fewer points if adaptive)
        if U.shape[0] != n_time_points:
            from scipy.interpolate import interp1d

            s_eval = np.linspace(0, T, n_time_points)
            interp = interp1d(sol.t, sol.y, axis=1, fill_value="extrapolate")
            U = interp(s_eval).T[::-1]

        return U


class NetworkPolicyIterationHJBSolver(NetworkHJBSolver):
    """
    Policy iteration solver for network HJB equations.

    Alternates between:
    1. Policy evaluation: Solve linear system for current policy
    2. Policy improvement: Update control policy
    """

    def __init__(
        self,
        problem: NetworkMFGProblem,
        max_policy_iterations: int = 50,
        policy_tolerance: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        """
        Initialize policy iteration HJB solver.

        Args:
            problem: Network MFG problem
            max_policy_iterations: Maximum policy iteration steps
            policy_tolerance: Policy convergence tolerance
            **kwargs: Additional arguments for base solver
        """
        super().__init__(problem, scheme="BDF", **kwargs)  # Scheme unused — policy iteration overrides solve

        self.max_policy_iterations = max_policy_iterations
        self.policy_tolerance = policy_tolerance
        self.hjb_method_name = "NetworkHJB_PolicyIteration"

        # Current policy (action for each node)
        self.current_policy: dict[int, int] = {}

    def solve_hjb_system(
        self,
        M_density: np.ndarray | None = None,
        U_terminal: np.ndarray | None = None,
        U_coupling_prev: np.ndarray | None = None,
        volatility_field: float | np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve HJB using policy iteration."""
        # Validate required parameters
        if M_density is None:
            raise ValueError("M_density is required")
        if U_terminal is None:
            raise ValueError("U_terminal is required")

        # Nt = number of time intervals
        # n_time_points = Nt + 1 (number of time knots including t=0 and t=T)
        Nt = self.network_problem.Nt
        n_time_points = Nt + 1
        U = np.zeros((n_time_points, self.num_nodes))

        # Set terminal condition at index Nt (last time point)
        U[Nt, :] = U_terminal

        # Backward time stepping with policy iteration
        # Nt steps from index (Nt-1) down to 0
        for n in range(Nt - 1, -1, -1):
            t = self.times[n]
            m_current = M_density[n, :]

            U[n, :] = self._policy_iteration_step(U[n + 1, :], m_current, t)

            # Apply boundary conditions
            U[n, :] = self.network_problem.apply_boundary_conditions(U[n, :], t)

        return U

    def _policy_iteration_step(self, u_next: np.ndarray, m: np.ndarray, t: float) -> np.ndarray:
        """Single time step using policy iteration."""
        u_current = u_next.copy()

        # Initialize policy (greedy with respect to u_next)
        self._initialize_policy(u_next, m, t)

        # Policy iteration loop
        for _policy_iter in range(self.max_policy_iterations):
            # Policy evaluation: solve linear system
            u_new = self._policy_evaluation(u_next, m, t)

            # Policy improvement
            old_policy = self.current_policy.copy()
            self._policy_improvement(u_new, m, t)

            # Check policy convergence
            if self._policies_equal(old_policy, self.current_policy):
                u_current = u_new
                break

            u_current = u_new

        return u_current

    def _initialize_policy(self, u: np.ndarray, m: np.ndarray, t: float) -> None:
        """Initialize policy via H.optimal_control (Issue #916).

        Uses NetworkHamiltonian.optimal_control() to compute transition rates,
        then extracts the dominant action (node with highest rate) as discrete policy.
        Falls back to greedy neighbor search if no hamiltonian_class.
        """
        H = self.network_problem.hamiltonian_class
        for i in range(self.num_nodes):
            neighbors = self.gradient_ops[i]
            if not neighbors:
                self.current_policy[i] = i
                continue

            if H is not None:
                # Issue #916: use H.optimal_control for transition rates
                rates = H.optimal_control(np.array([i]), m, u, t)
                rates_arr = np.atleast_1d(rates)
                # Pick neighbor with highest rate (dominant transition)
                best_action = i
                best_rate = 0.0
                for neighbor in neighbors:
                    if neighbor < len(rates_arr) and rates_arr[neighbor] > best_rate:
                        best_rate = rates_arr[neighbor]
                        best_action = neighbor
                self.current_policy[i] = best_action
            else:
                # Legacy: greedy cost minimization
                best_action = i
                best_cost = float("inf")
                for neighbor in neighbors:
                    cost = self._compute_action_cost(i, neighbor, u, m, t)
                    if cost < best_cost:
                        best_cost = cost
                        best_action = neighbor
                self.current_policy[i] = best_action

    def _policy_evaluation(self, u_next: np.ndarray, m: np.ndarray, t: float) -> np.ndarray:
        """Evaluate current policy by solving linear system.

        Issue #916: uses H.optimal_control for transition rates when available,
        falling back to edge_cost for the rate coefficient.
        """
        H = self.network_problem.hamiltonian_class

        A = sp.lil_matrix((self.num_nodes, self.num_nodes))
        b = np.zeros(self.num_nodes)

        for i in range(self.num_nodes):
            action = self.current_policy[i]
            A[i, i] = 1.0 / self.dt

            if action != i:
                if H is not None:
                    # Rate from H.optimal_control
                    rates = H.optimal_control(np.array([i]), m, u_next, t)
                    rate = float(np.atleast_1d(rates)[action])
                    A[i, i] += rate
                    A[i, action] -= rate
                else:
                    # Legacy: edge cost
                    edge_cost = self.network_problem.edge_cost(i, action, t)
                    A[i, action] -= 1.0 / self.dt
                    b[i] = edge_cost

            # Potential and coupling
            potential = self.network_problem.node_potential(i, t)
            coupling = self.network_problem.density_coupling(i, m, t)
            b[i] += potential + coupling + u_next[i] / self.dt

        A = A.tocsr()
        u_evaluated = spsolve(A, b)
        return np.asarray(u_evaluated)

    def _policy_improvement(self, u: np.ndarray, m: np.ndarray, t: float) -> None:
        """Improve policy using H.optimal_control (Issue #916)."""
        H = self.network_problem.hamiltonian_class
        for i in range(self.num_nodes):
            neighbors = self.gradient_ops[i] + [i]

            if H is not None:
                rates = H.optimal_control(np.array([i]), m, u, t)
                rates_arr = np.atleast_1d(rates)
                best_action = i
                best_rate = 0.0
                for action in neighbors:
                    if action < len(rates_arr) and rates_arr[action] > best_rate:
                        best_rate = rates_arr[action]
                        best_action = action
                self.current_policy[i] = best_action
            else:
                best_action = self.current_policy[i]
                best_cost = self._compute_action_cost(i, best_action, u, m, t)
                for action in neighbors:
                    cost = self._compute_action_cost(i, action, u, m, t)
                    if cost < best_cost:
                        best_cost = cost
                        best_action = action
                self.current_policy[i] = best_action

    def _compute_action_cost(self, node: int, action: int, u: np.ndarray, m: np.ndarray, t: float) -> float:
        """Legacy: compute cost of taking action from node (no H.optimal_control)."""
        if action == node:
            return self.network_problem.node_potential(node, t) + self.network_problem.density_coupling(node, m, t)
        edge_cost = self.network_problem.edge_cost(node, action, t)
        return edge_cost + u[action]

    def _policies_equal(self, policy1: dict[int, int], policy2: dict[int, int]) -> bool:
        """Check if two policies are equal."""
        return len(policy1) == len(policy2) and all(policy1[node] == policy2.get(node) for node in policy1)


# Factory function for network HJB solvers
def create_network_hjb_solver(problem: NetworkMFGProblem, solver_type: str = "RK45", **kwargs: Any) -> NetworkHJBSolver:
    """
    Create network HJB solver with specified type.

    Args:
        problem: Network MFG problem
        solver_type: Any scipy solve_ivp method, or "policy_iteration"
        **kwargs: Additional solver parameters

    Returns:
        Configured network HJB solver
    """
    if solver_type == "policy_iteration":
        return NetworkPolicyIterationHJBSolver(problem, **kwargs)
    return NetworkHJBSolver(problem, scheme=solver_type, **kwargs)


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing NetworkHJBSolver classes...")

    # Test class availability
    assert NetworkHJBSolver is not None
    assert NetworkPolicyIterationHJBSolver is not None
    assert create_network_hjb_solver is not None
    print("  Network HJB solver classes available")

    # Note: Full smoke test requires NetworkMFGProblem setup
    # See examples/networks/ for usage examples

    print("Smoke tests passed!")

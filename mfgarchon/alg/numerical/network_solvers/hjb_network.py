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

    # Node-BC capability gate (Issue #1468; #1456 network family). The base solver integrates the
    # backward HJB ODE (`_solve_ode` via `solve_ivp`) with terminal data only and never applies
    # `components.boundary_nodes`, so it cannot honor a node BC: `False`. The
    # `NetworkPolicyIterationHJBSolver` subclass — which applies them each step — overrides to `True`.
    _honors_node_bc: bool = False

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

        # Issue #1476: orientation sign for the backward-HJB integration, single-sourced from the wired
        # Hamiltonian object so the solver and the object never disagree on the sense. +1 for MINIMIZE
        # (du/ds = -H_control + source), -1 for MAXIMIZE (+H_control + source). ONLY the control flips
        # with the sense; the source (V + congestion) enters UNFLIPPED for both (see the rhs).
        # Note: the MAXIMIZE integration is anti-dissipative (+H_control grows the reverse-time value, as
        # reward-to-go should), so the explicit RK45 path is less robust than for MINIMIZE on stiff/large
        # problems; the implicit policy-iteration solver is the stable path. A genuine blow-up surfaces
        # via the solve_ivp non-convergence warning below (sol.success is False).
        self._sense_sign = problem.hamiltonian_class.sense_sign

        # Issue #1468/#1471: fail loud on a node BC this solver cannot honor. Node-BC now lives on the
        # geometry (GraphGeometry.has_explicit_boundary_conditions); the base ODE path applies only
        # the terminal condition and never the node-BC, so it would be silently ignored.
        # `NetworkPolicyIterationHJBSolver` (which applies it) sets `_honors_node_bc`.
        if not self._honors_node_bc and problem.geometry.has_explicit_boundary_conditions():
            raise NotImplementedError(
                f"{type(self).__name__} does not support node boundary conditions "
                f"(the graph geometry carries an explicit node-BC config). The base network HJB "
                f"integrates the backward ODE with terminal data only and never applies node-BC, so "
                f"it would be silently ignored (Issue #1468, #1456, #1471). Use "
                f"NetworkPolicyIterationHJBSolver, which applies node-BC, or remove the geometry's BC."
            )

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

    def _source_terms(self, m: np.ndarray, t: float) -> np.ndarray:
        """Per-node RHS source terms V(i, t) + f(i, m, t) — node potential + congestion coupling
        (Issue #1474). In mfgarchon's convention ``-u_t + H_control = source`` these sit on the RHS,
        so in reversed time they enter ``du/ds`` with the OPPOSITE sign to the control Hamiltonian.
        The network Hamiltonian method returns ``control + source``; subtracting this isolates the
        control Hamiltonian.
        """
        # Issue #1470: single-source the source (V + f_m) through the WIRED Hamiltonian object — the
        # SAME computation inside `_evaluate_hamiltonian_batch` (which routes through
        # `network_problem.hamiltonian` -> `hamiltonian_class`). Previously `node_potential +
        # density_coupling` re-derived the coupling on the raw stacked `m`, diverging from the object's
        # `_extract_own_density` for multi-population `m` and corrupting `h_control = h_total - source`.
        H = self.network_problem.hamiltonian_class
        return np.array([H.source_term(i, m, t) for i in range(self.num_nodes)])

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
            # mfgarchon convention -u_t + H_control = source (source = V + coupling on the RHS). In
            # reversed time s = T - t this is du/ds = -H_control + source (Issue #1474). The network
            # method returns control + source, so isolate the control Hamiltonian by subtracting the
            # source. (The old du/ds = +H integrated u_t + H = 0 and self-amplified the one-sided
            # control term (u_i - u_j)_+^2, blowing the value up for any non-trivial terminal data.)
            h_total = self._evaluate_hamiltonian_batch(u_flat, m, t_physical)
            source = self._source_terms(m, t_physical)
            h_control = h_total - source
            # Issue #1476: ONLY the control Hamiltonian flips with sense; the source (V + congestion) is
            # sense-INDEPENDENT (a running payoff accumulates identically whether you min or max — it is
            # added to H without a sense-flip, matching the continuum SeparableHamiltonian). MINIMIZE:
            # du/ds = -H_control + source; MAXIMIZE: +H_control + source. So the control term is
            # -s*h_control and the source enters unflipped.
            return -self._sense_sign * h_control + source

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

    # Issue #1468: unlike the base ODE solver, policy iteration applies
    # `problem.apply_boundary_conditions` (the node-Dirichlet pin over `boundary_nodes`) at every
    # timestep in `solve_hjb_system`, so it honors a node BC and is exempt from the base gate.
    _honors_node_bc: bool = True

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
        # Issue #1426 (S0-25): policy_tolerance is stored but never read — policy iteration
        # terminates on discrete policy stability (`_policies_equal`), not a value tolerance. Fail
        # loud on a non-default value. (max_policy_iterations IS used and bounds the loop.)
        if policy_tolerance != 1e-6:
            raise NotImplementedError(
                f"NetworkPolicyIterationHJBSolver(policy_tolerance={policy_tolerance}) is not "
                "implemented (Issue #1426): policy iteration terminates when the policy stops "
                "changing (exact convergence), not on a value tolerance, so policy_tolerance is "
                "never used. Omit it (default 1e-6); use max_policy_iterations to bound the loop."
            )
        self.hjb_method_name = "NetworkHJB_PolicyIteration"

        # Current policy: the full transition-rate vector alpha*_i per node (Issue #1474), not a
        # single dominant action — finite-state control is a full rate row over neighbors.
        self.current_rates: dict[int, np.ndarray] = {}

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
        """Single backward time step via Howard policy iteration (Issue #1474).

        The "policy" is the full transition-rate vector ``alpha*(u) = H.optimal_control`` (not a single
        dominant action, and sense-oriented via ``sense_sign``). Each iteration solves the linear
        policy-evaluation ``(I/dt + L^pi) u = u_next/dt + s*c(alpha) + source`` (``s = sense_sign``; only
        the control cost flips with the sense — the source does not) then recomputes the rates, until
        they stabilize. At the fixed point this is the backward-Euler discretization of the same HJB the
        RK45 path integrates: the envelope identity (generator action ``= 2*s*H_control``, ``c =
        H_control``) reduces the row to ``-du/dt = -s*H_control + source`` for both senses (Issue #1476).
        """
        self._initialize_policy(u_next, m, t)
        u_current = u_next.copy()
        for _policy_iter in range(self.max_policy_iterations):
            u_new = self._policy_evaluation(u_next, m, t)
            old_rates = self.current_rates
            self._policy_improvement(u_new, m, t)
            u_current = u_new
            if self._policies_equal(old_rates, self.current_rates):
                break
        return u_current

    def _rates_at(self, u: np.ndarray, m: np.ndarray, t: float) -> dict[int, np.ndarray]:
        """Full transition-rate vector ``alpha*_i`` for every node from the single-source Hamiltonian."""
        H = self.network_problem.hamiltonian_class
        if H is None:
            raise RuntimeError(
                "NetworkPolicyIterationHJBSolver requires a wired hamiltonian_class (Issue #1474). "
                "The legacy edge-cost policy evaluation assembled a singular row-sum-zero system "
                "(returned NaN); a NetworkMFGProblem always wires a NetworkHamiltonian."
            )
        return {i: np.atleast_1d(H.optimal_control(np.array([i]), m, u, t)) for i in range(self.num_nodes)}

    def _initialize_policy(self, u: np.ndarray, m: np.ndarray, t: float) -> None:
        self.current_rates = self._rates_at(u, m, t)

    def _policy_improvement(self, u: np.ndarray, m: np.ndarray, t: float) -> None:
        self.current_rates = self._rates_at(u, m, t)

    def _policy_evaluation(self, u_next: np.ndarray, m: np.ndarray, t: float) -> np.ndarray:
        """Solve ``(I/dt + L^pi) u = u_next/dt + c(alpha) + source`` for the frozen rate policy.

        ``L^pi = -Q^pi`` is the policy-weighted graph Laplacian (``A_ii = 1/dt + sum_j alpha_ij``,
        ``A_ij = -alpha_ij``) — a strictly diagonally-dominant M-matrix, non-singular for any
        ``alpha >= 0`` (the old ``A_ii=1/dt, A_ij=-1/dt`` was row-sum zero -> singular -> NaN).
        ``c_i(alpha) = 0.5 * sum_j alpha_ij^2 / w_ij`` is the control cost; ``source = V + coupling``.
        """
        A = sp.lil_matrix((self.num_nodes, self.num_nodes))
        b = np.zeros(self.num_nodes)
        source = self._source_terms(m, t)
        for i in range(self.num_nodes):
            alpha_i = self.current_rates[i]
            A[i, i] = 1.0 / self.dt
            control_cost = 0.0
            for j in self.gradient_ops[i]:
                a = float(alpha_i[j]) if j < len(alpha_i) else 0.0
                if a > 0.0:
                    A[i, i] += a
                    A[i, j] -= a
                    w = self.network_problem.network_data.get_edge_weight(i, j)
                    control_cost += 0.5 * a * a / w
            # Issue #1476: only the control COST flips with sense; the source is sense-INDEPENDENT (same
            # as the RK45 rhs). MINIMIZE b: u_next/dt + c + source; MAXIMIZE b: u_next/dt - c + source.
            # At the optimal-policy fixed point (generator action = 2*s*H_control, c = H_control) this
            # reduces to (u_i-u_next)/dt = -s*H_control + source, matching the RK45 integration. The A
            # matrix is unchanged (an M-matrix for any alpha >= 0, which holds for both senses).
            b[i] = u_next[i] / self.dt + self._sense_sign * control_cost + source[i]
        return np.asarray(spsolve(A.tocsr(), b))

    def _policies_equal(self, rates1: dict[int, np.ndarray], rates2: dict[int, np.ndarray]) -> bool:
        """Two rate policies are equal when every node's rate vector matches."""
        return all(np.allclose(rates1[i], rates2[i], atol=1e-10) for i in range(self.num_nodes))


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

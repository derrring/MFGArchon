"""
Fokker-Planck Solver for Networks/Graphs.

This module implements Fokker-Planck equation solvers for Mean Field Games
on network structures, handling discrete density evolution on graphs.

Mathematical formulation:
∂m/∂t - div_G(m ∇_G H_p) - σ²/2 Δ_G m = 0  on network
m(0, i) = m_0(i)                             initial condition

where:
- div_G: Graph divergence operator
- Δ_G: Graph Laplacian operator
- H_p: Derivative of Hamiltonian w.r.t. momentum
- σ²: Diffusion coefficient on network

Key algorithms:
- Explicit/implicit time stepping on networks
- Flow-based density evolution along edges
- Conservation-preserving schemes
- Network boundary condition handling
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver, DriftConvention
from mfgarchon.utils.deprecation import deprecated_parameter
from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfgarchon.extensions.topology import NetworkMFGProblem


class FPNetworkSolver(BaseFPSolver):
    # Issue #1043 Phase 2: consumes the value function U via `potential_field`
    # (renamed from the old `drift_field`; marked VALUE_FUNCTION for the contract trait).
    _drift_convention = DriftConvention.VALUE_FUNCTION
    """
    Fokker-Planck solver for Mean Field Games on networks.

    Solves the discrete FP equation:
    ∂m/∂t - div_G(m ∇_G H_p) - σ²/2 Δ_G m = 0

    with network-specific operators and mass conservation.

    Required Geometry Traits (Issue #596 Phase 2.4):
        - SupportsGraphLaplacian: Discrete Laplacian Δ_G for diffusion term (σ²/2) Δ_G m
        - SupportsAdjacency: Adjacency matrix for graph divergence operators

    Compatible Geometries:
        - NetworkGeometry (Grid, Random, ScaleFree, Custom networks)
        - MazeGeometry (2D grids with obstacles)
        - Any graph geometry implementing required traits

    Note:
        Uses trait-based graph operators for discrete density evolution on networks.
        Trait validation occurs at problem/geometry level.
        Mass conservation enforced through discrete operators.
    """

    # Node-BC capability gate (Issue #1468; #1456 network family). The network BC model is
    # `NetworkMFGComponents.boundary_nodes` (a node-level Dirichlet pin applied via
    # `NetworkMFGProblem.apply_boundary_conditions`), NOT the continuum `BCType`/segments framework —
    # so the inherited `_validate_bc_support` (which keys on `BoundaryConditions.segments`) is a
    # structural no-op for network problems. This solver ignores `boundary_nodes` entirely and
    # unconditionally renormalizes total mass each step, so it cannot honor a node BC: `False`.
    _honors_node_bc: bool = False

    def __init__(
        self,
        problem: NetworkMFGProblem,
        scheme: str = "explicit",
        diffusion_coefficient: float = 0.1,
        cfl_factor: float = 0.5,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        enforce_mass_conservation: bool = True,
    ):
        """
        Initialize network FP solver.

        Args:
            problem: Network MFG problem instance
            scheme: Time discretization ("explicit", "implicit", "upwind")
            diffusion_coefficient: Network diffusion coefficient σ²/2
            cfl_factor: CFL stability factor for explicit schemes
            max_iterations: Maximum iterations for implicit schemes
            tolerance: Convergence tolerance
            enforce_mass_conservation: Whether to enforce mass conservation
        """
        super().__init__(problem)

        self.network_problem = problem

        # Issue #1468: fail loud on a node BC this solver cannot honor. It ignores
        # `boundary_nodes` and unconditionally renormalizes total mass each step
        # (`enforce_mass_conservation`), so a node BC would be silently dropped and any
        # absorption hidden — manufacturing the mass-conserving answer in place of the
        # intended one (the #1456 silent-wrong-answer class, on the graph).
        if not self._honors_node_bc and problem.components.boundary_nodes:
            raise NotImplementedError(
                f"{type(self).__name__} does not support node boundary conditions "
                f"(problem.components.boundary_nodes is set). It ignores boundary_nodes and "
                f"unconditionally renormalizes total mass each step, so the node BC would be "
                f"silently dropped and any absorption hidden (Issue #1468, #1456). Remove "
                f"boundary_nodes, or use a solver that honors them."
            )

        self.scheme = scheme
        self.diffusion_coefficient = diffusion_coefficient
        self.cfl_factor = cfl_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        # Issue #1426 (S0-25): max_iterations / tolerance are documented "for implicit schemes" but
        # the implicit step is a direct sparse solve (spsolve), not iterative, so they are never
        # read. Fail loud on a non-default value rather than silently ignoring it.
        if max_iterations != 1000:
            raise NotImplementedError(
                f"FPNetworkSolver(max_iterations={max_iterations}) is not implemented (Issue #1426): "
                "the implicit scheme is a direct sparse solve (spsolve), not iterative, so "
                "max_iterations is never used. Omit it (default 1000)."
            )
        if tolerance != 1e-6:
            raise NotImplementedError(
                f"FPNetworkSolver(tolerance={tolerance}) is not implemented (Issue #1426): the "
                "implicit scheme is a direct sparse solve, so there is no iterative tolerance. Omit "
                "it (default 1e-6)."
            )
        self.enforce_mass_conservation = enforce_mass_conservation
        self._current_rates: dict | None = None  # Issue #913: precomputed per timestep

        # Network properties
        self.num_nodes = problem.num_nodes
        self.adjacency_matrix = problem.get_adjacency_matrix()
        self.laplacian_matrix = problem.get_laplacian_matrix()

        # Time discretization
        self.dt = problem.T / problem.Nt
        self.times = np.linspace(0, problem.T, problem.Nt + 1)

        # Solver name
        self.fp_method_name = f"NetworkFP_{scheme}"

        # Initialize discrete operators
        self._initialize_network_operators()

    def _initialize_network_operators(self):
        """Initialize network-specific discrete operators."""
        # Graph divergence operator structure
        self.divergence_ops = {}
        for i in range(self.num_nodes):
            neighbors = self.network_problem.get_node_neighbors(i)
            self.divergence_ops[i] = neighbors

        # Stability constraint for explicit schemes
        if self.scheme == "explicit":
            max_degree = np.max(np.array(self.adjacency_matrix.sum(axis=1)).flatten())
            self.dt_stable = self.cfl_factor / (max_degree * self.diffusion_coefficient + 1e-12)
            if self.dt > self.dt_stable:
                print(f"Warning: dt={self.dt:.2e} > dt_stable={self.dt_stable:.2e}")

    @deprecated_parameter(param_name="m_initial_condition", since="v0.17.0", replacement="M_initial")
    @deprecated_parameter(param_name="drift_field", since="v0.21.0", replacement="potential_field")
    def solve_fp_system(
        self,
        M_initial: np.ndarray | None = None,
        potential_field: np.ndarray | Callable | None = None,
        volatility_field: float | np.ndarray | Callable | None = None,
        show_progress: bool | None = None,
        # Deprecated parameter names for backward compatibility
        m_initial_condition: np.ndarray | None = None,
        drift_field: np.ndarray | Callable | None = None,
    ) -> np.ndarray:
        """
        Solve FP system on network with unified drift and diffusion API.

        Args:
            M_initial: Initial density m_0(i).
            m_initial_condition: DEPRECATED since v0.17.0 — use M_initial.
            potential_field: Value function U(t, i) specification (optional):
                - None: Zero drift (pure diffusion on network).
                - np.ndarray: Value function array, shape (Nt+1, num_nodes).
                  The solver differentiates U internally to compute drift.
                - Callable: Function U(t, i) -> value (Phase 2, not yet supported).
            drift_field: DEPRECATED since v0.21.0 — use potential_field.
                Accepted U under the old drift_field name (U-semantic). The rename
                mirrors the weak-form rename from PR #1205 (Issue #1043 Phase 2).
            volatility_field: SDE volatility sigma (optional); D = sigma^2/2 internally
                (Issue #1429 S0-15 — the base_fp contract, consistent with FDM/FVM/GFDM):
                - None: Use the D-valued self.diffusion_coefficient constructor knob (backward
                  compatible).
                - float: constant SDE volatility sigma; the diffusion is D = sigma^2/2.
                - np.ndarray/Callable: Phase 2 (not yet supported).
            show_progress: Whether to display progress bar for timesteps.

        Returns:
            (Nt+1, num_nodes) density evolution.
        """
        # Handle deprecated m_initial_condition redirect
        if m_initial_condition is not None:
            if M_initial is not None:
                raise ValueError(
                    "Cannot specify both M_initial and m_initial_condition. "
                    "Use M_initial (m_initial_condition is deprecated)."
                )
            M_initial = m_initial_condition

        # Handle deprecated drift_field redirect (Issue #1043 Phase 2).
        # FPNetworkSolver's drift_field was U-semantic (value function), not alpha*;
        # rename to potential_field to make this explicit.
        if drift_field is not None:
            if potential_field is not None:
                raise ValueError(
                    "Cannot specify both potential_field and drift_field. "
                    "Use potential_field (drift_field is deprecated since v0.21.0)."
                )
            potential_field = drift_field

        # Validate required parameter
        if M_initial is None:
            raise ValueError("M_initial is required")

        # Handle potential_field parameter (U-semantic: value function that the
        # solver differentiates internally to compute the drift velocity).
        if potential_field is None:
            # Zero drift (pure diffusion): create zero U field for internal use
            # Use n_time_points = problem.Nt + 1 for consistency
            n_time_points = self.network_problem.Nt + 1
            effective_U = np.zeros((n_time_points, self.num_nodes))
        elif isinstance(potential_field, np.ndarray):
            # Precomputed value function field
            effective_U = potential_field
        elif callable(potential_field):
            # Custom potential function - Phase 2
            raise NotImplementedError(
                "FPNetworkSolver does not yet support callable potential_field. "
                "Pass precomputed value function as np.ndarray. "
                "Support for callable potential_field coming in Phase 2."
            )
        else:
            raise TypeError(f"potential_field must be None, np.ndarray, or Callable, got {type(potential_field)}")

        # Handle volatility_field parameter
        if volatility_field is None:
            # Use self.diffusion_coefficient (backward compatible)
            effective_diffusion = self.diffusion_coefficient
        elif isinstance(volatility_field, (int, float)):
            # Issue #1429 (S0-15): volatility_field is the SDE volatility sigma (the base_fp
            # contract used by FDM/FVM/GFDM), NOT the diffusion D — convert via the single source
            # D = sigma^2/2. (The `diffusion_coefficient` constructor knob stays D-valued.)
            effective_diffusion = float(diffusion_from_volatility(float(volatility_field)))
        elif isinstance(volatility_field, np.ndarray) or callable(volatility_field):
            # Spatially varying or state-dependent - Phase 2
            raise NotImplementedError(
                "FPNetworkSolver does not yet support spatially varying or callable volatility_field. "
                "Pass constant diffusion as float. Support coming in Phase 2."
            )
        else:
            raise TypeError(
                f"volatility_field must be None, float, np.ndarray, or Callable, got {type(volatility_field)}"
            )

        # Temporarily override diffusion_coefficient if custom diffusion provided
        original_diffusion = self.diffusion_coefficient
        if volatility_field is not None:
            self.diffusion_coefficient = effective_diffusion

        try:
            # n_time_points = problem.Nt + 1 (number of time knots including t=0 and t=T)
            # n_intervals = problem.Nt (number of time intervals between knots)
            n_intervals = self.network_problem.Nt
            n_time_points = n_intervals + 1
            M = np.zeros((n_time_points, self.num_nodes))

            # Set initial condition
            M[0, :] = M_initial

            # Normalize initial condition if needed
            if self.enforce_mass_conservation:
                total_mass = np.sum(M[0, :])
                if total_mass > 1e-12:
                    M[0, :] /= total_mass

            # Forward time stepping with progress bar
            from mfgarchon.utils.progress import create_progress_bar, should_show_progress

            # Forward FP loop: n_intervals steps (problem.Nt) from t=0 to t=T
            timestep_range = create_progress_bar(
                range(n_intervals),
                verbose=should_show_progress(show_progress),
                desc="FP (forward)",
            )

            for n in timestep_range:
                t = self.times[n]

                # Current value function for drift computation
                u_current = effective_U[n, :]

                # Issue #913: precompute transition rates via H.optimal_control
                self._current_rates = self._precompute_transition_rates(M[n, :], u_current, t)

                # Solve single time step
                if self.scheme == "explicit":
                    M[n + 1, :] = self._explicit_step(M[n, :], u_current, t)
                elif self.scheme == "implicit":
                    M[n + 1, :] = self._implicit_step(M[n, :], u_current, t)
                elif self.scheme == "upwind":
                    M[n + 1, :] = self._upwind_step(M[n, :], u_current, t)
                else:
                    raise ValueError(f"Unknown scheme: {self.scheme}")

                # Enforce non-negativity and mass conservation
                M[n + 1, :] = np.maximum(M[n + 1, :], 0)

                if self.enforce_mass_conservation:
                    total_mass = np.sum(M[n + 1, :])
                    if total_mass > 1e-12:
                        M[n + 1, :] /= total_mass

            return M
        finally:
            # Restore original diffusion coefficient
            self.diffusion_coefficient = original_diffusion

    def _explicit_step(self, m_current: np.ndarray, u_current: np.ndarray, t: float) -> np.ndarray:
        """Explicit time step for network FP equation."""
        m_next = m_current.copy()

        # Compute flows for each node
        for i in range(self.num_nodes):
            neighbors = self.divergence_ops[i]

            # Diffusion term: -σ²/2 * Δ_G m
            diffusion_term = 0.0
            for j in neighbors:
                edge_weight = self.network_problem.network_data.get_edge_weight(i, j)
                diffusion_term += edge_weight * (m_current[j] - m_current[i])
            diffusion_term *= self.diffusion_coefficient

            # Drift term: -div_G(m ∇_G H_p)
            drift_term = self._compute_drift_term(i, m_current, u_current, t)

            # Forward Euler step
            m_next[i] = m_current[i] + self.dt * (diffusion_term + drift_term)

        return m_next

    def _implicit_step(self, m_current: np.ndarray, u_current: np.ndarray, t: float) -> np.ndarray:
        """Implicit time step for network FP equation."""
        # Build linear system for implicit step
        A = sp.lil_matrix((self.num_nodes, self.num_nodes))
        b = m_current.copy()

        for i in range(self.num_nodes):
            neighbors = self.divergence_ops[i]

            # Diagonal term
            A[i, i] = 1.0

            # Diffusion terms (implicit)
            for j in neighbors:
                edge_weight = self.network_problem.network_data.get_edge_weight(i, j)
                coeff = self.dt * self.diffusion_coefficient * edge_weight

                A[i, i] += coeff
                A[i, j] -= coeff

            # Drift terms (explicit for simplicity)
            drift_term = self._compute_drift_term(i, m_current, u_current, t)
            b[i] += self.dt * drift_term

        # Solve linear system
        A = A.tocsr()
        m_next = spsolve(A, b)
        # Ensure result is numpy array
        # Backend compatibility - scipy.sparse may return matrix (Issue #543 acceptable)
        if hasattr(m_next, "toarray"):
            m_next = m_next.toarray().flatten()
        m_next = np.asarray(m_next)

        return m_next

    def _upwind_step(self, m_current: np.ndarray, u_current: np.ndarray, t: float) -> np.ndarray:
        """Upwind scheme for network FP equation (conservation-preserving)."""
        m_next = m_current.copy()

        # Compute flows between neighboring nodes
        for i in range(self.num_nodes):
            neighbors = self.divergence_ops[i]

            net_flow = 0.0

            for j in neighbors:
                # Compute flow from j to i
                flow_ji = self._compute_edge_flow(j, i, m_current, u_current, t)
                # Compute flow from i to j
                flow_ij = self._compute_edge_flow(i, j, m_current, u_current, t)

                # Net flow into node i
                net_flow += flow_ji - flow_ij

            # Update with net flow
            m_next[i] = m_current[i] + self.dt * net_flow

        return m_next

    def _precompute_transition_rates(self, m: np.ndarray, u: np.ndarray, t: float) -> dict:
        """Precompute transition rates alpha[i][j] for all nodes.

        Issue #913: Uses H.optimal_control() when available.
        Returns dict mapping node i -> {j: alpha_ij} for all neighbors j.
        """
        H_class = self.network_problem.hamiltonian_class
        rates = {}
        for i in range(self.num_nodes):
            neighbors = self.divergence_ops.get(i, [])
            if H_class is not None:
                alpha = H_class.optimal_control(np.array([i]), m, u, t)
                rates[i] = {j: float(alpha[j]) for j in neighbors if alpha[j] > 0}
            else:
                # Legacy: quadratic rates from value gradient
                rates[i] = {}
                for j in neighbors:
                    du = u[j] - u[i]
                    edge_weight = self.network_problem.network_data.get_edge_weight(i, j)
                    rate = edge_weight * max(du, 0.0)
                    if rate > 0:
                        rates[i][j] = rate
        return rates

    def _compute_drift_term(self, node: int, m: np.ndarray, u: np.ndarray, t: float) -> float:
        """Compute drift term for FP equation at given node.

        Uses precomputed rates from _precompute_transition_rates.
        If rates not precomputed, falls back to legacy computation.
        """
        if self._current_rates is not None:
            # Use precomputed rates: inflow - outflow
            outflow = sum(rate * m[node] for rate in self._current_rates.get(node, {}).values())
            inflow = sum(
                self._current_rates.get(j, {}).get(node, 0.0) * m[j] for j in self.divergence_ops.get(node, [])
            )
            return inflow - outflow

        # Legacy fallback
        neighbors = self.divergence_ops.get(node, [])
        drift = 0.0
        for neighbor in neighbors:
            du = u[neighbor] - u[node]
            edge_weight = self.network_problem.network_data.get_edge_weight(node, neighbor)
            drift += edge_weight * m[node] * du
        return drift

    def _compute_edge_flow(self, node_from: int, node_to: int, m: np.ndarray, u: np.ndarray, t: float) -> float:
        """Compute flow along edge from node_from to node_to."""
        if node_from == node_to:
            return 0.0

        # Check if edge exists
        edge_weight = self.network_problem.network_data.get_edge_weight(node_from, node_to)
        if edge_weight == 0:
            return 0.0

        # Upwind density selection
        du = u[node_to] - u[node_from]

        if du > 0:  # Flow from node_from to node_to
            density = m[node_from]
        else:  # Flow from node_to to node_from
            density = m[node_to]
            du = -du

        # Flow magnitude
        flow = edge_weight * density * du

        return flow

    def forward_step(self, m_prev: np.ndarray, u_current: np.ndarray, dt: float) -> np.ndarray:
        """
        Single forward time step (interface compatibility).

        Args:
            m_prev: Density at previous time step
            u_current: Current value function
            dt: Time step size

        Returns:
            Updated density distribution
        """
        # Temporarily adjust dt for this step
        original_dt = self.dt
        self.dt = dt

        t = 0.0  # Dummy time (should be passed properly in full implementation)

        if self.scheme == "explicit":
            result = self._explicit_step(m_prev, u_current, t)
        elif self.scheme == "implicit":
            result = self._implicit_step(m_prev, u_current, t)
        else:
            result = self._upwind_step(m_prev, u_current, t)

        # Restore original dt
        self.dt = original_dt

        # Enforce constraints
        result = np.maximum(result, 0)
        if self.enforce_mass_conservation:
            total_mass = np.sum(result)
            if total_mass > 1e-12:
                result /= total_mass

        return result


# Alias for backward compatibility
NetworkFPSolver = FPNetworkSolver


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing NetworkFPSolver...")

    # Test class availability
    assert NetworkFPSolver is not None
    print("  NetworkFPSolver class available")

    # Note: Full smoke test requires NetworkMFGProblem setup
    # See examples/networks/ for usage examples

    print("Smoke tests passed!")

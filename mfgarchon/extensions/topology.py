"""
Network Mean Field Games Problem Formulation.

This module implements MFG problems on network/graph structures, extending
the continuous MFG framework to discrete network domains.

Mathematical Framework:
- State space: Discrete nodes of the network
- Density evolution: Network flow dynamics on edges
- HJB equation: Discrete optimal control on graphs
- Coupling: Local interactions at nodes and along edges

Key differences from continuous MFG:
- Spatial derivatives → Graph discrete derivatives (differences)
- Laplacian operator → Graph Laplacian matrix
- Continuous density → Discrete node masses/flows
- Boundary conditions → Network boundary nodes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.core.hamiltonian import (
    HamiltonianBase,
    OptimizationSense,
)
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem

if TYPE_CHECKING:
    from collections.abc import Callable

    from scipy.sparse import csr_matrix

    from mfgarchon.geometry.graph.network_geometry import BaseNetworkGeometry


class NetworkHamiltonian(HamiltonianBase):
    """HamiltonianBase subclass for finite-state MFG on graphs.

    Issue #910: Wraps network Hamiltonian logic in the class-based API.
    Accepts either a callable H(node, neighbors, m, p, t) or provides
    a default quadratic Hamiltonian on edges.

    Parameters
    ----------
    network_data : NetworkData
        Graph structure (adjacency, edge weights).
    hamiltonian_func : callable or None
        Custom H(node, neighbors, m, p, t) -> float.
        If None, uses default quadratic: sum_{j in N(i)} w_ij/2 * (p_j - p_i)^2.
    hamiltonian_dm_func : callable or None
        Custom dH/dm.
    node_potential_func : callable or None
        V(node, t) -> float.
    node_interaction_func : callable or None
        f(node, m, t) -> float. Read on the FULL density, matching the live
        ``NetworkMFGProblem.density_coupling`` (Issue #1470 reconciliation).
    """

    def __init__(
        self,
        network_data,
        hamiltonian_func=None,
        hamiltonian_dm_func=None,
        node_potential_func=None,
        node_interaction_func=None,
        sense=OptimizationSense.MINIMIZE,
        population_index: int = 0,
    ):
        super().__init__(sense=sense, population_index=population_index)
        self.network_data = network_data
        self._hamiltonian_func = hamiltonian_func
        self._hamiltonian_dm_func = hamiltonian_dm_func
        self._node_potential = node_potential_func
        self._node_interaction = node_interaction_func
        self._num_nodes: int | None = None

    @property
    def num_nodes(self) -> int:
        if self._num_nodes is None:
            self._num_nodes = self.network_data.num_nodes
        return self._num_nodes

    def _extract_own_density(self, m: np.ndarray) -> np.ndarray:
        """Extract this population's density from stacked m_all.

        If m has length N (single population), return as-is.
        If m has length K*N (stacked multi-population), extract slice.
        """
        N = self.num_nodes
        if len(m) == N:
            return m
        k = self.population_index
        return m[k * N : (k + 1) * N]

    def __call__(self, x, m, p, t=0.0):
        """Evaluate H at node x with density m and costate p.

        x: node index (int or array with single int)
        m: density vector, shape (N,) — this population's density.
        p: costate vector at all nodes, shape (N,)
        """
        node = int(np.asarray(x).flat[0])
        m_arr = np.atleast_1d(m)
        p_arr = np.atleast_1d(p)

        if self._hamiltonian_func is not None:
            neighbors = self.network_data.get_neighbors(node)
            # Single-pop: custom H gets own density only
            return float(self._hamiltonian_func(node, neighbors, m_arr, p_arr, t))

        return self._default_hamiltonian(node, m_arr, p_arr, t)

    def _default_hamiltonian(self, node, m, p, t):
        """Default: quadratic control on edges + potential + congestion.

        For multi-population: congestion uses own density slice.
        Custom hamiltonian_func receives full m_all for cross-coupling.
        """
        neighbors = self.network_data.get_neighbors(node)
        control_cost = 0.0
        for neighbor in neighbors:
            w = self.network_data.get_edge_weight(node, neighbor)
            dp = p[neighbor] - p[node]
            control_cost += 0.5 * w * dp**2

        V = float(self._node_potential(node, t)) if self._node_potential else 0.0
        # Coupling f(node, m, t). Reconciled with the live NetworkMFGProblem.density_coupling
        # (Issue #1470): read node_interaction_func on the FULL density, and default to the
        # quadratic node congestion 0.5 * m_own[node]^2. The previous default of 0.0 (and reading
        # the dead `congestion_func` field) silently diverged from the method used on every live
        # solve — see Issue #910. `_extract_own_density` is the identity for single-population m
        # (so byte-identical to the method) and slices the own population for stacked K*N m.
        if self._node_interaction is not None:
            f_m = float(self._node_interaction(node, m, t))
        else:
            f_m = 0.5 * float(self._extract_own_density(m)[node]) ** 2
        return control_cost + V + f_m

    def optimal_control(self, x, m, p, t=0.0):
        """Optimal transition rates from node x.

        For quadratic H: alpha_{ij} = w_ij * (p_j - p_i)_+ (upwind).
        Transition rates are non-negative by construction, ensuring
        the generator Q preserves positivity of m.
        Returns array of rates to neighbors (zero for non-neighbors).
        """
        node = int(np.asarray(x).flat[0])
        p_arr = np.atleast_1d(p)
        neighbors = self.network_data.get_neighbors(node)

        alpha = np.zeros_like(p_arr)
        for neighbor in neighbors:
            w = self.network_data.get_edge_weight(node, neighbor)
            dp = p_arr[neighbor] - p_arr[node]
            # Issue #914: upwind truncation ensures alpha >= 0
            alpha[neighbor] = w * max(dp, 0.0)
        return alpha

    def dp(self, x, m, p, t=0.0):
        """dH/dp at node x. For quadratic: sum_j w_ij * (p_j - p_i) per component."""
        node = int(np.asarray(x).flat[0])
        p_arr = np.atleast_1d(p)
        neighbors = self.network_data.get_neighbors(node)

        grad = np.zeros_like(p_arr)
        for neighbor in neighbors:
            w = self.network_data.get_edge_weight(node, neighbor)
            dp = p_arr[neighbor] - p_arr[node]
            grad[neighbor] += w * dp
            grad[node] -= w * dp
        return grad

    def dm(self, x, m, p, t=0.0):
        """dH/dm at node x."""
        if self._hamiltonian_dm_func is not None:
            node = int(np.asarray(x).flat[0])
            neighbors = self.network_data.get_neighbors(node)
            return float(self._hamiltonian_dm_func(node, neighbors, np.atleast_1d(m), np.atleast_1d(p), t))
        return self._finite_diff_dm(x, m, p, t)


@dataclass
class NetworkMFGComponents(MFGComponents):
    """
    Components for defining MFG problems on networks.

    This extends the continuous MFGComponents to handle discrete network structures,
    including support for Lagrangian formulations and trajectory measures.
    """

    # Network-specific Hamiltonian (depends on node states and edge flows)
    hamiltonian_func: Callable | None = None  # H(node, neighbors, m, p, t)
    hamiltonian_dm_func: Callable | None = None  # dH/dm at nodes

    # Lagrangian formulation support (based on ArXiv 2207.10908v3)
    lagrangian_func: Callable | None = None  # L(node, velocity, m, t)
    velocity_space_dim: int = 2  # Dimension of velocity space
    trajectory_cost_func: Callable | None = None  # Cost along trajectories
    relaxed_control: bool = False  # Use relaxed equilibria

    # Node-based potential function
    node_potential_func: Callable | None = None  # V(node, t)

    # Edge-based costs/rewards
    edge_cost_func: Callable | None = None  # Cost of moving along edges
    congestion_func: Callable | None = None  # Congestion effects

    # Initial and terminal conditions on network
    initial_node_density_func: Callable | None = None  # m_0(node)
    terminal_node_value_func: Callable | None = None  # u_T(node)

    # Network boundary conditions
    boundary_nodes: list[int] | None = None  # Nodes with boundary conditions
    boundary_values_func: Callable | None = None  # Boundary values

    # Flow dynamics parameters
    diffusion_coefficient: float = 1.0  # Diffusion strength
    drift_coefficient: float = 1.0  # Drift/advection strength

    # Network-specific coupling
    node_interaction_func: Callable | None = None  # Local node interactions
    edge_interaction_func: Callable | None = None  # Edge-based interactions

    # Problem parameters
    problem_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Issue #1470 (Problem/Components unification, Layer Ψ): NetworkMFGComponents IS-A
        # MFGComponents so ``isinstance`` holds and the #1456 BC-capability gate stops silently
        # no-op'ing on network problems. But it specifies the MFG with the network-native fields
        # (hamiltonian_func, node_interaction_func, boundary_nodes, ...), not the continuum ones,
        # so the parent __post_init__ — which requires a class-based Hamiltonian / m_initial /
        # u_terminal at construction — does not apply. The network Hamiltonian needs graph
        # structure and is bound by NetworkMFGProblem, not the components. The init=False
        # ``_hamiltonian_class`` / ``_lagrangian_class`` fields default to None regardless.
        pass


class NetworkMFGProblem(MFGProblem):
    """
    Mean Field Games problem on network structures.

    This class implements MFG formulations on discrete network domains,
    supporting various network topologies and interaction mechanisms.

    Mathematical formulation:

    HJB equation (discrete):
    ∂u/∂t + H_i(m, ∇_G u, t) = 0  at node i
    u(T, i) = g(i)                  terminal condition

    Fokker-Planck equation (discrete):
    ∂m/∂t - div_G(m ∇_G H_p) - Δ_G m = 0  on network
    m(0, i) = m_0(i)                        initial condition

    where:
    - ∇_G: Graph gradient operator
    - div_G: Graph divergence operator
    - Δ_G: Graph Laplacian operator
    - H_i: Hamiltonian at node i
    """

    def __init__(
        self,
        network_geometry: BaseNetworkGeometry,
        T: float = 1.0,
        Nt: int = 100,
        components: NetworkMFGComponents | None = None,
        problem_name: str = "NetworkMFG",
    ):
        """
        Initialize network MFG problem.

        Args:
            network_geometry: Network structure and geometry
            T: Terminal time
            Nt: Number of time steps
            components: Network MFG components (optional)
            problem_name: Problem identifier
        """
        # Network properties - set first before calling super()
        self.network_geometry = network_geometry
        self.network_data = network_geometry.network_data

        # Issue #910: Create NetworkHamiltonian for parent class validation.
        # Uses the real network Hamiltonian instead of a dummy placeholder.
        num_nodes = network_geometry.num_nodes
        net_components = components or NetworkMFGComponents()
        network_hamiltonian = NetworkHamiltonian(
            network_data=network_geometry.network_data,
            hamiltonian_func=net_components.hamiltonian_func,
            hamiltonian_dm_func=net_components.hamiltonian_dm_func,
            node_potential_func=net_components.node_potential_func,
            node_interaction_func=net_components.node_interaction_func,
        )
        parent_components = MFGComponents(
            hamiltonian=network_hamiltonian,
            m_initial=lambda x: 1.0 / num_nodes,
            u_terminal=lambda x: 0.0,
        )

        # Initialize parent with geometry (not deprecated xmin/xmax/Nx)
        super().__init__(
            T=T,
            Nt=Nt,
            geometry=network_geometry,
            components=parent_components,
        )

        self.components = components or NetworkMFGComponents()  # type: ignore[assignment]
        self.problem_name = problem_name

        # Phase 3.1 integration: geometry is already set by parent
        self.dimension = "network"  # Special dimension indicator for network problems

        # Re-detect solver compatibility after overriding dimension
        # (parent __init__ called _detect_solver_compatibility() before this override)
        self._detect_solver_compatibility()

        # Override spatial properties for network
        self.is_network_problem = True
        self.num_nodes = network_geometry.num_nodes
        self.spatial_dimension = 0  # Discrete network, not continuous space

        # Network-specific matrices
        self.adjacency_matrix: csr_matrix | None = None
        self.laplacian_matrix: csr_matrix | None = None
        self.incidence_matrix: csr_matrix | None = None

        self._initialize_network_operators()

    def _initialize_network_operators(self):
        """Initialize network-specific operators and matrices."""
        if self.network_data is None:
            raise ValueError("Network data not available. Create network first.")

        self.adjacency_matrix = self.network_data.adjacency_matrix
        self.laplacian_matrix = self.network_data.laplacian_matrix
        self.incidence_matrix = self.network_data.incidence_matrix

        # Store network properties
        self.is_directed = self.network_data.is_directed
        self.is_weighted = self.network_data.is_weighted
        self.num_edges = self.network_data.num_edges

    # Network-specific MFG components

    def hamiltonian(self, node: int, neighbors: list[int], m: np.ndarray, p: np.ndarray, t: float) -> float:
        """
        Network Hamiltonian function.

        Args:
            node: Current node index
            neighbors: List of neighboring nodes
            m: Density vector at all nodes
            p: Co-state vector at all nodes
            t: Current time

        Returns:
            Hamiltonian value at the node
        """
        if self.components.hamiltonian_func is not None:
            return self.components.hamiltonian_func(node, neighbors, m, p, t)

        # Default quadratic Hamiltonian with network structure
        return self._default_network_hamiltonian(node, neighbors, m, p, t)

    def _default_network_hamiltonian(
        self, node: int, neighbors: list[int], m: np.ndarray, p: np.ndarray, t: float
    ) -> float:
        """Default network Hamiltonian implementation."""
        # Quadratic control cost + potential + density coupling
        control_cost = 0.0

        # Sum over possible moves to neighbors
        for neighbor in neighbors:
            if self.network_data is None:
                edge_weight = 1.0  # Default weight
            else:
                edge_weight = self.network_data.get_edge_weight(node, neighbor)
            # Control cost for moving to neighbor
            dp = p[neighbor] - p[node]  # Discrete gradient
            control_cost += 0.5 * edge_weight * dp**2

        # Node potential
        potential = self.node_potential(node, t)

        # Density coupling (congestion effects)
        coupling = self.density_coupling(node, m, t)

        return control_cost + potential + coupling

    def hamiltonian_dm(self, node: int, neighbors: list[int], m: np.ndarray, p: np.ndarray, t: float) -> float:
        """Derivative of Hamiltonian with respect to density."""
        if self.components.hamiltonian_dm_func is not None:
            return self.components.hamiltonian_dm_func(node, neighbors, m, p, t)

        # Default: derivative of density coupling term
        return self._default_density_coupling_derivative(node, m, t)

    # Lagrangian formulation methods (based on ArXiv 2207.10908v3)

    def lagrangian(self, node: int, velocity: np.ndarray, m: np.ndarray, t: float) -> float:
        """
        Lagrangian function for network MFG.

        Based on the Lagrangian formulation from ArXiv 2207.10908v3,
        this represents the cost of being at a node with given velocity.

        Args:
            node: Current node index
            velocity: Velocity vector in network space
            m: Density distribution over network
            t: Current time

        Returns:
            Lagrangian value L(node, velocity, m, t)
        """
        if self.components.lagrangian_func is not None:  # type: ignore[attr-defined]
            return self.components.lagrangian_func(node, velocity, m, t)  # type: ignore[attr-defined]

        # Default Lagrangian: kinetic energy + potential + interaction
        kinetic_energy = 0.5 * np.linalg.norm(velocity) ** 2
        potential = self.node_potential(node, t)
        interaction = self.density_coupling(node, m, t)

        return float(kinetic_energy + potential + interaction)

    def trajectory_cost(
        self,
        trajectory: list[int],
        velocities: np.ndarray,
        m_evolution: np.ndarray,
        times: np.ndarray,
    ) -> float:
        """
        Compute cost along a network trajectory.

        Args:
            trajectory: Sequence of nodes visited
            velocities: Velocity at each time step
            m_evolution: Density evolution over time
            times: Time points

        Returns:
            Total trajectory cost
        """
        if self.components.trajectory_cost_func is not None:  # type: ignore[attr-defined]
            return self.components.trajectory_cost_func(trajectory, velocities, m_evolution, times)  # type: ignore[attr-defined]

        # Default: integrate Lagrangian along trajectory
        total_cost = 0.0
        dt = times[1] - times[0] if len(times) > 1 else 1.0

        for i, (node, t) in enumerate(zip(trajectory, times, strict=False)):
            if i < len(velocities):
                velocity = velocities[i] if velocities.ndim > 1 else np.array([velocities[i]])
                m_current = m_evolution[i] if m_evolution.ndim > 1 else m_evolution
                lagrangian_value = self.lagrangian(node, velocity, m_current, t)
                total_cost += lagrangian_value * dt

        return total_cost

    def compute_relaxed_equilibrium(self, trajectory_measures: list[Callable]):
        """
        Compute relaxed equilibrium as probability measures on trajectories.

        Based on the relaxed equilibria concept from ArXiv 2207.10908v3.

        Args:
            trajectory_measures: List of probability measures on trajectory space

        Returns:
            NetworkSolveResult with U (value function) and M (density)

        Note:
            Backward compatible: Supports tuple unpacking via `u, m = result`
        """
        # This is a placeholder for advanced trajectory measure computation
        # Full implementation would require sophisticated measure theory

        u = np.zeros((self.Nt + 1, self.num_nodes))
        m = np.zeros((self.Nt + 1, self.num_nodes))

        # Initialize with uniform distribution
        m[0, :] = 1.0 / self.num_nodes

        # Simple trajectory-based computation (to be enhanced)
        for t_idx in range(self.Nt):
            # Update based on trajectory measures
            for node in range(self.num_nodes):
                # Aggregate trajectory contributions
                total_measure = 0.0
                for measure in trajectory_measures:
                    total_measure += measure(node, t_idx)
                m[t_idx + 1, node] = total_measure

        # Normalize density
        for t_idx in range(self.Nt + 1):
            total = np.sum(m[t_idx, :])
            if total > 1e-12:
                m[t_idx, :] /= total

        from mfgarchon.types.solver_types import NetworkSolveResult

        return NetworkSolveResult(U=u, M=m)

    def node_potential(self, node: int, t: float) -> float:
        """Potential function at network nodes."""
        if self.components.node_potential_func is not None:  # type: ignore[attr-defined]
            return self.components.node_potential_func(node, t)  # type: ignore[attr-defined]
        return 0.0

    def density_coupling(self, node: int, m: np.ndarray, t: float) -> float:
        """Density coupling/interaction at nodes."""
        if self.components.node_interaction_func is not None:  # type: ignore[attr-defined]
            return self.components.node_interaction_func(node, m, t)  # type: ignore[attr-defined]

        # Default: quadratic congestion at nodes
        return 0.5 * m[node] ** 2

    def _default_density_coupling_derivative(self, node: int, m: np.ndarray, t: float) -> float:
        """Derivative of default density coupling."""
        return m[node]  # d/dm[i] (0.5 * m[i]^2) = m[i]

    def edge_cost(self, node_from: int, node_to: int, t: float) -> float:
        """Cost of moving along network edges."""
        if self.components.edge_cost_func is not None:  # type: ignore[attr-defined]
            return self.components.edge_cost_func(node_from, node_to, t)  # type: ignore[attr-defined]

        # Default: unit cost weighted by edge weight
        if self.network_data is None:
            return 1.0  # Default weight
        edge_weight = self.network_data.get_edge_weight(node_from, node_to)
        return edge_weight

    # Initial and terminal conditions

    def get_initial_density(self) -> np.ndarray:
        """Initial density distribution on network nodes."""
        if self.components.initial_node_density_func is not None:  # type: ignore[attr-defined]
            return np.array([self.components.initial_node_density_func(i) for i in range(self.num_nodes)])  # type: ignore[attr-defined]

        # Default: uniform distribution
        initial_density = np.ones(self.num_nodes) / self.num_nodes
        return initial_density

    def get_terminal_value(self) -> np.ndarray:
        """Terminal value function on network nodes."""
        if self.components.terminal_node_value_func is not None:  # type: ignore[attr-defined]
            return np.array([self.components.terminal_node_value_func(i) for i in range(self.num_nodes)])  # type: ignore[attr-defined]

        # Default: zero terminal values
        return np.zeros(self.num_nodes)

    # Network-specific operators

    def compute_graph_gradient(self, u: np.ndarray) -> np.ndarray:
        """
        Compute discrete gradient on network.

        For each node, computes differences to neighboring nodes.

        Args:
            u: Node values

        Returns:
            Gradient information (implementation dependent)
        """
        gradients = np.zeros((self.num_nodes, self.num_nodes))

        for i in range(self.num_nodes):
            if self.network_data is None:
                neighbors = []  # No neighbors if no network data
            else:
                neighbors = self.network_data.get_neighbors(i)
            for j in neighbors:
                gradients[i, j] = u[j] - u[i]

        return gradients

    def compute_graph_divergence(self, flow: np.ndarray) -> np.ndarray:
        """
        Compute discrete divergence on network.

        Args:
            flow: Edge flow values

        Returns:
            Divergence at each node
        """
        # Simplified: divergence as sum of incoming/outgoing flows
        divergence = np.zeros(self.num_nodes)

        # This is a simplified implementation
        # Full implementation would handle edge-based flows properly
        for i in range(self.num_nodes):
            if self.network_data is None:
                neighbors = []  # No neighbors if no network data
            else:
                neighbors = self.network_data.get_neighbors(i)
            div_i = 0.0
            for j in neighbors:
                # Flow from j to i minus flow from i to j
                div_i += flow[j] - flow[i]  # Simplified
            divergence[i] = div_i

        return divergence

    def apply_graph_laplacian(self, u: np.ndarray, coefficient: float = 1.0) -> np.ndarray:
        """
        Apply graph Laplacian operator to node values.

        Args:
            u: Node values
            coefficient: Diffusion coefficient

        Returns:
            Laplacian applied to u
        """
        return np.asarray(coefficient * (self.laplacian_matrix @ u))

    # Boundary conditions for networks

    def apply_boundary_conditions(self, u: np.ndarray, t: float) -> np.ndarray:
        """Apply boundary conditions to network nodes."""
        u_bc = u.copy()

        if self.components.boundary_nodes is not None:  # type: ignore[attr-defined]
            for node in self.components.boundary_nodes:  # type: ignore[attr-defined]
                if self.components.boundary_values_func is not None:  # type: ignore[attr-defined]
                    u_bc[node] = self.components.boundary_values_func(node, t)  # type: ignore[attr-defined]
                else:
                    # Default: zero boundary values
                    u_bc[node] = 0.0

        return u_bc

    # Legacy interface compatibility

    def get_initial_m(self) -> np.ndarray:
        """Get initial density (legacy interface)."""
        return self.get_initial_density()

    def get_final_u(self) -> np.ndarray:
        """Get terminal value function (legacy interface)."""
        return self.get_terminal_value()

    @property
    def Nx(self) -> int:
        """Number of spatial points (nodes in network)."""
        return self.num_nodes - 1

    @property
    def xmin(self) -> float:
        """Minimum spatial coordinate (dummy for networks)."""
        return 0.0

    @property
    def xmax(self) -> float:
        """Maximum spatial coordinate (dummy for networks)."""
        return float(self.num_nodes - 1)

    @property
    def Dx(self) -> float:
        """Spatial step size (dummy for networks)."""
        return 1.0

    # Network-specific properties

    def get_network_statistics(self) -> dict[str, Any]:
        """Get comprehensive network statistics."""
        from mfgarchon.geometry.graph.network_geometry import compute_network_statistics

        if self.network_data is None:
            raise ValueError("Network data not initialized")
        return compute_network_statistics(self.network_data)

    def get_adjacency_matrix(self) -> csr_matrix:
        """Get network adjacency matrix."""
        if self.adjacency_matrix is None:
            raise ValueError("Adjacency matrix not initialized")
        return self.adjacency_matrix

    def get_laplacian_matrix(self) -> csr_matrix:
        """Get network Laplacian matrix."""
        if self.laplacian_matrix is None:
            raise ValueError("Laplacian matrix not initialized")
        return self.laplacian_matrix

    def get_node_neighbors(self, node: int) -> list[int]:
        """Get neighbors of a specific node."""
        if self.network_data is None:
            return []  # No neighbors if no network data
        return self.network_data.get_neighbors(node)

    def __str__(self) -> str:
        """String representation of network MFG problem."""
        stats = self.get_network_statistics()
        network_type = "Unknown"
        if self.network_data is not None and hasattr(self.network_data, "network_type"):
            network_type = getattr(self.network_data.network_type, "value", str(self.network_data.network_type))

        return (
            f"NetworkMFGProblem({self.problem_name})\n"
            f"  Network: {network_type}\n"
            f"  Nodes: {self.num_nodes}, Edges: {self.num_edges}\n"
            f"  Time: T={self.T}, Nt={self.Nt}\n"
            f"  Connected: {stats['is_connected']}\n"
            f"  Average degree: {stats['average_degree']:.2f}"
        )


# Factory functions for common network MFG problems


def create_grid_mfg_problem(
    width: int,
    height: int | None = None,
    T: float = 1.0,
    Nt: int = 100,
    periodic: bool = False,
    **kwargs: Any,
) -> NetworkMFGProblem:
    """Create MFG problem on grid network."""
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    height = height or width
    network = GridNetwork(width, height, periodic)
    network.create_network()

    components = NetworkMFGComponents(**kwargs)

    return NetworkMFGProblem(
        network_geometry=network,
        T=T,
        Nt=Nt,
        components=components,
        problem_name=f"GridMFG_{width}x{height}",
    )


def create_random_mfg_problem(
    num_nodes: int,
    connection_prob: float = 0.1,
    T: float = 1.0,
    Nt: int = 100,
    seed: int | None = None,
    **kwargs: Any,
) -> NetworkMFGProblem:
    """Create MFG problem on random network."""
    from mfgarchon.geometry.graph.network_geometry import RandomNetwork

    network = RandomNetwork(num_nodes, connection_prob)
    network.create_network(seed=seed)

    components = NetworkMFGComponents(**kwargs)

    return NetworkMFGProblem(
        network_geometry=network,
        T=T,
        Nt=Nt,
        components=components,
        problem_name=f"RandomMFG_N{num_nodes}_p{connection_prob}",
    )


def create_scale_free_mfg_problem(
    num_nodes: int,
    num_edges_per_node: int = 2,
    T: float = 1.0,
    Nt: int = 100,
    seed: int | None = None,
    **kwargs: Any,
) -> NetworkMFGProblem:
    """Create MFG problem on scale-free network."""
    from mfgarchon.geometry.graph.network_geometry import ScaleFreeNetwork

    network = ScaleFreeNetwork(num_nodes, num_edges_per_node)
    network.create_network(seed=seed)

    components = NetworkMFGComponents(**kwargs)

    return NetworkMFGProblem(
        network_geometry=network,
        T=T,
        Nt=Nt,
        components=components,
        problem_name=f"ScaleFreeMFG_N{num_nodes}_m{num_edges_per_node}",
    )

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

import warnings
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
        # Issue #1474/#1476: the finite-state MFG supports BOTH senses through a single orientation sign
        # `sense_sign` (+1 MINIMIZE / -1 MAXIMIZE). Cost-to-go (MINIMIZE) sends agents DOWNHILL toward
        # lower value; reward-to-go (MAXIMIZE) sends them UPHILL toward higher value. Every sense-
        # dependent piece — control cost, optimal_control, dp, and the base-solver integration sign — is
        # `s * (MINIMIZE form)`, so `sense_sign` is the single source of the mirror. See `sense_sign`.
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

    @property
    def sense_sign(self) -> float:
        """Orientation sign of the finite-state control (Issue #1476): ``+1`` for MINIMIZE (cost-to-go,
        agents move DOWNHILL toward lower value), ``-1`` for MAXIMIZE (reward-to-go, agents move UPHILL
        toward higher value). Every sense-dependent piece — the control cost in ``_default_hamiltonian``,
        ``optimal_control``, ``dp``, and the HJB integration sign in ``hjb_network`` — is ``s * (MINIMIZE
        form)``, so this one property is the single source of the MINIMIZE<->MAXIMIZE mirror.
        """
        return 1.0 if self.sense == OptimizationSense.MINIMIZE else -1.0

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
        # Finite-state MFG control cost (Issue #1474/#1476): one-sided / piecewise-quadratic — the value
        # of the constrained optimum over rates alpha >= 0 (a valid CTMC generator). Only the ACTIVE side
        # contributes, oriented by sense_sign s: MINIMIZE (s=+1) counts downhill edges u_i>u_j giving
        # 0.5*sum w*max(u_i-u_j,0)^2; MAXIMIZE (s=-1) counts uphill edges u_j>u_i giving
        # 0.5*sum w*max(u_j-u_i,0)^2. Consistent with optimal_control (both use max(s*(u_i-u_j),0)).
        s = self.sense_sign
        control_cost = 0.0
        for neighbor in neighbors:
            w = self.network_data.get_edge_weight(node, neighbor)
            du = s * (p[node] - p[neighbor])  # oriented: u_i - u_j (MIN) / u_j - u_i (MAX)
            control_cost += 0.5 * w * max(du, 0.0) ** 2

        # Issue #1470: the p-independent source (V + coupling) is single-sourced in `source_term` and
        # consumed both here (control + source) and by the HJB solver's control isolation.
        return control_cost + self.source_term(node, m, t)

    def source_term(self, x, m, t=0.0):
        """The p-independent source ``V(node) + f(node, m)`` — the RHS in ``-u_t + H_control = source``.

        Single source consumed by ``_default_hamiltonian`` (as ``control + source``) AND by
        ``hjb_network._source_terms`` (Issue #1470): computing it once here removes the multi-population
        fork where the HJB used to re-derive ``V + density_coupling`` on the raw stacked ``m`` while the
        Hamiltonian used ``_extract_own_density`` — so ``h_control = h_total - source`` now isolates the
        control exactly. Coupling ``f(node, m, t)`` reads ``node_interaction_func`` on the FULL density
        (cross-coupling), else defaults to quadratic node congestion ``0.5 * m_own[node]^2``.
        ``_extract_own_density`` is the identity for single-population ``m`` (byte-identical) and slices
        the own population for stacked ``K*N`` ``m``.
        """
        return self.node_potential_value(x, t) + self.coupling_value(x, m, t)

    def node_potential_value(self, x, t=0.0):
        """Node potential ``V(node, t)`` — the single source for ``NetworkMFGProblem.node_potential``
        (Issue #1470 Strand A). ``0.0`` when no ``node_potential_func`` is set.
        """
        node = int(np.asarray(x).flat[0])
        return float(self._node_potential(node, t)) if self._node_potential else 0.0

    def coupling_value(self, x, m, t=0.0):
        """Density coupling ``f(node, m, t)`` — the single source for
        ``NetworkMFGProblem.density_coupling`` (Issue #1470 Strand A). Reads ``node_interaction_func``
        on the FULL density (cross-coupling), else the default quadratic node congestion
        ``0.5 * m_own[node]^2`` on the OWN-population slice. ``_extract_own_density`` is the identity for
        single-population ``m`` (byte-identical to the legacy raw ``m[node]``) and slices the own
        population for stacked ``K*N`` ``m``.
        """
        node = int(np.asarray(x).flat[0])
        if self._node_interaction is not None:
            return float(self._node_interaction(node, m, t))
        return 0.5 * float(self._extract_own_density(m)[node]) ** 2

    def optimal_control(self, x, m, p, t=0.0):
        """Optimal transition rates from node x (Issue #1474/#1476).

        Finite-state MFG: ``alpha*_ij = w_ij * max(s*(u_i - u_j), 0)`` with orientation ``s = sense_sign``
        — the argmax of the one-sided control Hamiltonian. MINIMIZE (s=+1) sends agents DOWNHILL toward
        lower cost-to-go (``max(u_i - u_j, 0)``); MAXIMIZE (s=-1) sends them UPHILL toward higher
        reward-to-go (``max(u_j - u_i, 0)``). Rates are non-negative by construction (a valid
        conservative CTMC generator). Returns an array of rates to neighbors (zero for non-neighbors).
        """
        node = int(np.asarray(x).flat[0])
        p_arr = np.atleast_1d(p)
        neighbors = self.network_data.get_neighbors(node)

        s = self.sense_sign
        alpha = np.zeros_like(p_arr)
        for neighbor in neighbors:
            w = self.network_data.get_edge_weight(node, neighbor)
            du = s * (p_arr[node] - p_arr[neighbor])  # downhill (MIN) / uphill (MAX)
            alpha[neighbor] = w * max(du, 0.0)
        return alpha

    def dp(self, x, m, p, t=0.0):
        """dH/dp at node x (Issue #1474/#1476). Gradient of the one-sided control Hamiltonian
        ``0.5 sum_j w_ij max(s*(u_i-u_j),0)^2`` with ``s = sense_sign``: differentiating gives
        ``dH/du_i = +s*sum_j alpha*_ij`` and ``dH/du_j = -s*alpha*_ij`` where
        ``alpha*_ij = w_ij max(s*(u_i-u_j),0) >= 0``. So both the rate orientation and the gradient sign
        flip with the sense (MINIMIZE s=+1 downhill; MAXIMIZE s=-1 uphill), and ``dp`` equals the
        generator action ``-Q^{alpha*} u``."""
        node = int(np.asarray(x).flat[0])
        p_arr = np.atleast_1d(p)
        neighbors = self.network_data.get_neighbors(node)

        s = self.sense_sign
        grad = np.zeros_like(p_arr)
        for neighbor in neighbors:
            w = self.network_data.get_edge_weight(node, neighbor)
            a = w * max(s * (p_arr[node] - p_arr[neighbor]), 0.0)  # alpha*_{i,neighbor} (oriented) >= 0
            grad[neighbor] -= s * a
            grad[node] += s * a
        return grad

    def dm(self, x, m, p, t=0.0):
        """dH/dm at node x. Issue #1470 Strand A: the default node congestion has the EXACT analytic
        derivative ``d/dm (0.5 * m_own[node]^2) = m_own[node]`` (own-population slice, matching
        ``coupling_value``); a custom ``node_interaction_func`` has no analytic form here, so uses a
        node-wise central finite difference of the coupling. This is the single source for
        ``NetworkMFGProblem.hamiltonian_dm``.
        """
        node = int(np.asarray(x).flat[0])
        if self._hamiltonian_dm_func is not None:
            neighbors = self.network_data.get_neighbors(node)
            return float(self._hamiltonian_dm_func(node, neighbors, np.atleast_1d(m), np.atleast_1d(p), t))
        if self._node_interaction is None:
            return float(self._extract_own_density(np.atleast_1d(m))[node])
        # Custom node_interaction_func: central finite difference of the coupling in the OWN node
        # component of the full density. The base HamiltonianBase._finite_diff_dm collapses m to a
        # scalar (np.mean), which breaks node-indexed interaction funcs (m[node] on a length-1 array
        # -> IndexError) — #1537 review. Difference coupling_value on the full vector instead.
        m_arr = np.atleast_1d(np.asarray(m, dtype=float))
        eps = 1e-7
        m_plus = m_arr.copy()
        m_minus = m_arr.copy()
        m_plus[node] += eps
        m_minus[node] -= eps
        return (self.coupling_value(node, m_plus, t) - self.coupling_value(node, m_minus, t)) / (2.0 * eps)


@dataclass
class NetworkMFGComponents(MFGComponents):
    """
    Components for defining MFG problems on networks.

    This extends the continuous MFGComponents to handle discrete network structures,
    including support for Lagrangian formulations and trajectory measures.

    Default model
    -------------
    When ``hamiltonian_func`` / ``node_interaction_func`` are omitted, the network Hamiltonian defaults
    to the finite-state quadratic-congestion form
    ``H = 0.5*sum_j w_ij*max(u_i - u_j, 0)^2 + V(node) + 0.5*m[node]^2`` (control cost + node potential +
    quadratic node congestion). A ``None`` field means "use this default", **not** "omit the term".
    Provide the callables to override (see ``NetworkHamiltonian``).
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

    # Node boundary conditions are owned by the graph geometry (GraphGeometry), not components
    # (Issue #1471) — construct e.g. GridNetwork(..., boundary_conditions=GraphBCConfig(...)). The
    # former `boundary_nodes` / `boundary_values_func` fields bypassed the #1456 BC single source.

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
        geometry: BaseNetworkGeometry | None = None,
        T: float = 1.0,
        Nt: int = 100,
        components: NetworkMFGComponents | None = None,
        problem_name: str = "NetworkMFG",
        *,
        network_geometry: BaseNetworkGeometry | None = None,
        sense: OptimizationSense = OptimizationSense.MINIMIZE,
    ):
        """
        Initialize network MFG problem.

        Args:
            geometry: Graph geometry (network structure). Named ``geometry`` to align with
                ``MFGProblem`` — geometry is the axis of variation (Issue #1472).
            T: Terminal time
            Nt: Number of time steps
            components: Network MFG components (optional)
            problem_name: Problem identifier
            network_geometry: Deprecated alias for ``geometry`` (Issue #1472); redirects identically.
        """
        # Issue #1472: the canonical constructor param is `geometry` (aligned with MFGProblem, toward
        # the Problem/Components unification). `network_geometry=` is a deprecated alias that redirects
        # identically (equivalence-tested); it will be removed after the deprecation window.
        if network_geometry is not None:
            if geometry is not None:
                raise ValueError(
                    "Pass the graph geometry via geometry=; network_geometry= is a deprecated alias — do not pass both."
                )
            warnings.warn(
                "NetworkMFGProblem(network_geometry=...) is deprecated (Issue #1472); use geometry=. "
                "The alias redirects identically and is removed after the deprecation window.",
                DeprecationWarning,
                stacklevel=2,
            )
            geometry = network_geometry
        if geometry is None:
            raise ValueError("NetworkMFGProblem requires a graph geometry (pass geometry=...).")
        network_geometry = geometry  # local name; the remainder of __init__ uses network_geometry

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
            sense=sense,  # Issue #1476: MINIMIZE (cost-to-go) or MAXIMIZE (reward-to-go)
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

        # Issue #1474: store the SAME NetworkMFGComponents instance the NetworkHamiltonian was built
        # from, and wire the object as the single-source Hamiltonian (`hamiltonian_class`). Previously
        # `self.components` was overwritten with a fresh instance, orphaning the object
        # (`hamiltonian_class == None`), so FP / policy iteration fell to divergent legacy paths while
        # RK45 used the method — the three solved different HJBs. Now all read one Hamiltonian.
        self.components = net_components  # type: ignore[assignment]
        self.components.hamiltonian = network_hamiltonian
        self.components._hamiltonian_class = network_hamiltonian

        # Issue #1471: node boundary conditions are owned by the graph geometry (GraphGeometry),
        # not by components. Resolve the geometry's GraphBCConfig once into the single-source
        # GraphApplicator (the single applier — the pin logic is not re-forked here). Explicit-init
        # None when the geometry carries no node-BC.
        self._node_applicator = None
        node_bc = network_geometry.get_boundary_conditions()
        if node_bc is not None:
            from mfgarchon.geometry.boundary.applicator_graph import GraphApplicator

            self._node_applicator = GraphApplicator.from_config(node_bc, num_nodes=network_geometry.num_spatial_points)

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
        """Network Hamiltonian at a node — delegates to the single-source ``NetworkHamiltonian``.

        Issue #1472: the value is computed by the wired ``hamiltonian_class`` — the SAME object the FP
        and policy-iteration solvers use (via ``optimal_control``) — so the RK45 base solver reads the
        identical Hamiltonian rather than a second hand-synced copy. This removes the former
        ``_default_network_hamiltonian`` duplicate, which had to be kept in lockstep with the object by
        hand (the #1474/N15 divergence risk). ``neighbors`` is accepted for signature compatibility;
        the object recomputes the neighborhood from ``network_data``. Byte-identical (pinned by
        ``test_network_hamiltonian_method_equals_object``).
        """
        return float(self.hamiltonian_class(node, m, p, t))

    def hamiltonian_dm(self, node: int, neighbors: list[int], m: np.ndarray, p: np.ndarray, t: float) -> float:
        """Derivative of the Hamiltonian w.r.t. density dH/dm. Issue #1470 Strand A: delegates to the
        wired single-source Hamiltonian object's ``dm`` (which owns the custom ``hamiltonian_dm_func``,
        the analytic default-congestion derivative, and the finite-difference fallback)."""
        return float(self.hamiltonian_class.dm(node, m, p, t))

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

    def node_potential(self, node: int, t: float) -> float:
        """Potential function V(node, t) at network nodes. Issue #1470 Strand A: delegates to the wired
        single-source Hamiltonian object so every consumer reads ONE computation."""
        return float(self.hamiltonian_class.node_potential_value(node, t))

    def density_coupling(self, node: int, m: np.ndarray, t: float) -> float:
        """Density coupling f(node, m, t) at nodes. Issue #1470 Strand A: delegates to the wired
        single-source Hamiltonian object (``coupling_value``), so the default congestion uses the
        own-population slice (``_extract_own_density``) — matching the Hamiltonian on stacked
        multi-population m instead of re-deriving on the raw ``m[node]``."""
        return float(self.hamiltonian_class.coupling_value(node, m, t))

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

    # Graph operators (gradient / divergence / Laplacian) are owned by the geometry
    # (GraphGeometry, Issue #1472) — the graph structural data lives there, not on the problem. The
    # problem previously reimplemented three of them (compute_graph_gradient / compute_graph_divergence
    # / apply_graph_laplacian); they had zero callers and are removed to keep a single source.

    # Boundary conditions for networks

    def apply_boundary_conditions(self, u: np.ndarray, t: float) -> np.ndarray:
        """Apply the geometry-owned node boundary conditions to the value field (Issue #1471).

        Node-BC lives on the graph geometry and is resolved once into a single-source
        ``GraphApplicator``; this delegates the value-field (HJB) DIRICHLET pin to it (which copies
        the input, so ``u`` is not mutated). No-op when the geometry carries no node-BC. The previous
        ``components.boundary_nodes`` channel (which bypassed the #1456 BC single source) is retired.
        """
        if self._node_applicator is None:
            return u
        return self._node_applicator.apply_hjb(u, t)

    # Legacy interface compatibility

    def get_initial_m(self) -> np.ndarray:
        """Get initial density (legacy interface)."""
        return self.get_initial_density()

    def get_final_u(self) -> np.ndarray:
        """Get terminal value function (legacy interface)."""
        return self.get_terminal_value()

    # The continuum spatial fields Nx / xmin / xmax / Dx are not defined here (Issue #1472). They are
    # not part of the base MFGProblem interface and no network-path consumer reads them — the network
    # solvers use num_nodes; graph geometry has no continuum coordinates. The former dummies (Nx =
    # num_nodes - 1, xmin = 0, xmax = num_nodes - 1, Dx = 1) invited nonsense continuum code paths.

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
        geometry=network,
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
        geometry=network,
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
        geometry=network,
        T=T,
        Nt=Nt,
        components=components,
        problem_name=f"ScaleFreeMFG_N{num_nodes}_m{num_edges_per_node}",
    )

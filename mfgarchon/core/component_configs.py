"""Configuration dataclasses for MFG formulations.

This module provides structured configuration classes for different MFG formulations.
Each config class groups related parameters for a specific formulation type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.geometry import BoundaryConditions


# =============================================================================
# Standard MFG Configuration
# =============================================================================


@dataclass
class StandardMFGConfig:
    """Configuration for standard HJB-FP MFG formulation.

    This is the most common MFG formulation using Hamiltonian H(x,m,p,t)
    to define agent dynamics and coupling through density m.

    Parameters
    ----------
    hamiltonian_func : Callable, optional
        H(x, m, p, t) -> float. Agent Hamiltonian defining dynamics.
    hamiltonian_dm_func : Callable, optional
        dH/dm(x, m, p, t) -> float. Derivative w.r.t. density.
    hamiltonian_dp_func : Callable, optional
        dH/dp(x, m, p, t) -> array. Derivative w.r.t. momentum (for analytic Jacobian).
    hamiltonian_jacobian_func : Callable, optional
        Jacobian contribution for coupling terms.
    potential_func : Callable, optional
        V(x, t) -> float. External potential/forces.
    m_initial : Callable, optional
        m_0(x) -> float. Initial agent distribution.
    u_terminal : Callable, optional
        u_T(x) -> float. Terminal cost/reward.
    boundary_conditions : BoundaryConditions, optional
        Boundary condition specification.
    coupling_func : Callable, optional
        Additional coupling terms F(x, m, t).
    """

    hamiltonian_func: Callable | None = None
    hamiltonian_dm_func: Callable | None = None
    hamiltonian_dp_func: Callable | None = None
    hamiltonian_jacobian_func: Callable | None = None
    potential_func: Callable | None = None
    m_initial: Callable | None = None
    u_terminal: Callable | None = None
    boundary_conditions: BoundaryConditions | None = None
    coupling_func: Callable | None = None


# =============================================================================
# Network/Graph MFG Configuration
# =============================================================================


@dataclass
class NetworkMFGConfig:
    """Configuration for network/graph MFG formulation.

    For MFG on discrete domains where agents move between nodes along edges.

    Parameters
    ----------
    network_geometry : Any, optional
        NetworkGeometry instance defining graph structure.
    node_interaction_func : Callable, optional
        f_node(node_id, density, t) -> float. Interactions at nodes.
    edge_interaction_func : Callable, optional
        f_edge(edge_id, density, t) -> float. Interactions along edges.
    edge_cost_func : Callable, optional
        c(edge_id, density, t) -> float. Cost to traverse edge.
    """

    network_geometry: Any | None = None
    node_interaction_func: Callable | None = None
    edge_interaction_func: Callable | None = None
    edge_cost_func: Callable | None = None


# =============================================================================
# Variational/Lagrangian MFG Configuration
# =============================================================================


@dataclass
class VariationalMFGConfig:
    """Configuration for variational/Lagrangian MFG formulation.

    Optimization-based formulation using Lagrangian L(t,x,v,m) instead of
    Hamiltonian. Related to Hamiltonian via Legendre transform.

    Parameters
    ----------
    lagrangian_func : Callable, optional
        L(t, x, v, m) -> float. Running cost function.
    lagrangian_dx_func : Callable, optional
        ∂L/∂x. Spatial derivative.
    lagrangian_dv_func : Callable, optional
        ∂L/∂v. Velocity derivative.
    lagrangian_dm_func : Callable, optional
        ∂L/∂m. Density derivative (coupling).
    terminal_cost_func : Callable, optional
        g(x) -> float. Terminal cost.
    terminal_cost_dx_func : Callable, optional
        ∂g/∂x. Terminal cost derivative.
    trajectory_cost_func : Callable, optional
        Cost along entire trajectories.
    state_constraints : list[Callable], optional
        Constraints c(t, x) ≤ 0.
    velocity_constraints : list[Callable], optional
        Constraints h(t, x, v) ≤ 0.
    integral_constraints : list[Callable], optional
        Integral constraints ∫ψ(x,m)dx = const.
    """

    lagrangian_func: Callable | None = None
    lagrangian_dx_func: Callable | None = None
    lagrangian_dv_func: Callable | None = None
    lagrangian_dm_func: Callable | None = None
    terminal_cost_func: Callable | None = None
    terminal_cost_dx_func: Callable | None = None
    trajectory_cost_func: Callable | None = None
    state_constraints: list[Callable] | None = None
    velocity_constraints: list[Callable] | None = None
    integral_constraints: list[Callable] | None = None


# =============================================================================
# Stochastic MFG Configuration
# =============================================================================


@dataclass
class StochasticMFGConfig:
    """Configuration for stochastic MFG formulation.

    Adds noise (common and/or idiosyncratic) to MFG dynamics.

    Parameters
    ----------
    noise_intensity : float, default=0.0
        Diffusion coefficient σ.
    common_noise_func : Callable, optional
        W(t) -> float. Noise affecting all agents equally.
    idiosyncratic_noise_func : Callable, optional
        Z_i(t) -> float. Individual agent noise.
    correlation_matrix : NDArray, optional
        Correlation structure between noise dimensions.
    """

    noise_intensity: float = 0.0
    common_noise_func: Callable | None = None
    idiosyncratic_noise_func: Callable | None = None
    correlation_matrix: NDArray | None = None


# =============================================================================
# Neural Network MFG Configuration
# =============================================================================


@dataclass
class NeuralMFGConfig:
    """Configuration for neural network-based MFG solvers.

    For PINN (Physics-Informed Neural Networks) and Deep BSDE methods.

    Parameters
    ----------
    neural_architecture : dict, optional
        Network architecture: {'layers': [64, 64, 64], 'activation': 'tanh'}.
    value_network_config : dict, optional
        Configuration for value function u(t,x) network.
    policy_network_config : dict, optional
        Configuration for optimal control network.
    density_network_config : dict, optional
        Configuration for density m(t,x) network.
    loss_weights : dict, optional
        Loss component weights: {'pde': 1.0, 'ic': 10.0, 'bc': 10.0}.
    physics_loss_func : Callable, optional
        Custom physics-informed loss function.
    network_initializer : Callable, optional
        Custom weight initialization strategy.
    """

    neural_architecture: dict[str, Any] | None = None
    value_network_config: dict[str, Any] | None = None
    policy_network_config: dict[str, Any] | None = None
    density_network_config: dict[str, Any] | None = None
    loss_weights: dict[str, float] | None = None
    physics_loss_func: Callable | None = None
    network_initializer: Callable | None = None


# =============================================================================
# Reinforcement Learning MFG Configuration
# =============================================================================


@dataclass
class RLMFGConfig:
    """Configuration for reinforcement learning-based MFG solvers.

    For PPO, Actor-Critic, and other RL approaches to MFG.

    Parameters
    ----------
    reward_func : Callable, optional
        r(state, action, density, t) -> float. Agent reward function.
    terminal_reward_func : Callable, optional
        r_T(state) -> float. Terminal reward.
    action_space_bounds : list[tuple[float, float]], optional
        Action space bounds: [(a_min, a_max), ...] for each dimension.
    observation_func : Callable, optional
        Maps full state to agent observation.
    action_constraints : list[Callable], optional
        Action constraints: g(state, action) ≤ 0.
    agent_interaction_func : Callable, optional
        How agents interact with each other.
    population_coupling_strength : float, default=0.0
        λ for mean-field coupling term strength.
    """

    reward_func: Callable | None = None
    terminal_reward_func: Callable | None = None
    action_space_bounds: list[tuple[float, float]] | None = None
    observation_func: Callable | None = None
    action_constraints: list[Callable] | None = None
    agent_interaction_func: Callable | None = None
    population_coupling_strength: float = 0.0


# =============================================================================
# Implicit Geometry Configuration
# =============================================================================


@dataclass
class ImplicitGeometryConfig:
    """Configuration for implicit geometry representations.

    For level sets, signed distance functions, obstacles, and manifolds.

    Parameters
    ----------
    level_set_func : Callable, optional
        φ(x) = 0 defines implicit surface.
    signed_distance_func : Callable, optional
        d(x) -> float. Signed distance to surface.
    obstacle_func : Callable, optional
        Returns 1 if point is inside obstacle, 0 otherwise.
    obstacle_penalty : float, default=1e10
        Penalty coefficient for obstacle violations.
    manifold_projection : Callable, optional
        Projects points onto manifold constraint.
    tangent_space_basis : Callable, optional
        Local coordinate system on manifold.
    """

    level_set_func: Callable | None = None
    signed_distance_func: Callable | None = None
    obstacle_func: Callable | None = None
    obstacle_penalty: float = 1e10
    manifold_projection: Callable | None = None
    tangent_space_basis: Callable | None = None


# =============================================================================
# Adaptive Mesh Refinement Configuration
# =============================================================================


@dataclass
class AMRConfig:
    """Configuration for adaptive mesh refinement.

    For dynamic mesh adaptation based on solution features.

    Parameters
    ----------
    refinement_indicator : Callable, optional
        Estimates local error: indicator(u, m) -> float.
    refinement_threshold : float, default=0.1
        Refine cells where indicator > threshold.
    coarsening_threshold : float, default=0.01
        Coarsen cells where indicator < threshold.
    feature_detection_func : Callable, optional
        Detects shocks, fronts, etc. for targeted refinement.
    min_cell_size : float, optional
        Minimum allowed cell size.
    max_cell_size : float, optional
        Maximum allowed cell size.
    max_refinement_level : int, default=5
        Maximum depth of refinement hierarchy.
    """

    refinement_indicator: Callable | None = None
    refinement_threshold: float = 0.1
    coarsening_threshold: float = 0.01
    feature_detection_func: Callable | None = None
    min_cell_size: float | None = None
    max_cell_size: float | None = None
    max_refinement_level: int = 5


# =============================================================================
# Time-Dependent Domain Configuration
# =============================================================================


@dataclass
class TimeDependentDomainConfig:
    """Configuration for time-varying domains.

    For moving boundaries, dynamic obstacles, and domain topology changes.

    Parameters
    ----------
    boundary_motion_func : Callable, optional
        Defines ∂Ω(t) - boundary evolution over time.
    domain_velocity_func : Callable, optional
        v_domain(x, t) - velocity field of moving domain.
    obstacle_trajectory_func : Callable, optional
        x_obstacle(t) - trajectory of moving obstacles.
    domain_split_func : Callable, optional
        Defines when/how domain splits into multiple regions.
    domain_merge_func : Callable, optional
        Defines when/how domains merge together.
    """

    boundary_motion_func: Callable | None = None
    domain_velocity_func: Callable | None = None
    obstacle_trajectory_func: Callable | None = None
    domain_split_func: Callable | None = None
    domain_merge_func: Callable | None = None


# =============================================================================
# Multi-Population MFG Configuration
# =============================================================================


@dataclass
class MultiPopulationMFGConfig:
    """Configuration for multi-population MFG.

    For problems with multiple distinct agent populations interacting.

    Parameters
    ----------
    num_populations : int, default=1
        Number of distinct agent populations.
    population_hamiltonians : list[Callable], optional
        H_i for each population i.
    population_initial_densities : list[Callable], optional
        m_0^i(x) for each population i.
    cross_population_coupling : Callable, optional
        F(m_1, m_2, ..., m_N) defining inter-population interactions.
    population_weights : list[float], optional
        Relative sizes of each population (should sum to 1).
    """

    num_populations: int = 1
    population_hamiltonians: list[Callable] | None = None
    population_initial_densities: list[Callable] | None = None
    cross_population_coupling: Callable | None = None
    population_weights: list[float] | None = None

r"""
Crowd Navigation Mean Field Game Environment.

2D spatial navigation environment where agents navigate toward goals while
avoiding collisions with other agents. Demonstrates:
- 2D spatial dynamics
- Goal-directed behavior
- Collision avoidance via mean field coupling
- Realistic crowd movement patterns

Mathematical Formulation:
- State: $(x, y, v_x, v_y, x_{goal}, y_{goal}) \in \mathbb{R}^6$
- Action: $(a_x, a_y) \in [-a_{\max}, a_{\max}]^2$ (acceleration)
- Dynamics: $\vec{x}' = \vec{x} + \vec{v} \cdot dt$, $\vec{v}' = \vec{v} + \vec{a} \cdot dt + \sigma \vec{\epsilon}$
- Individual Cost: distance to goal + velocity penalty + control effort
- Mean Field Cost: local crowd density (congestion/collision avoidance)
- Goal: Reach goal position while avoiding crowded areas

Use Cases:
- Pedestrian dynamics simulation
- Crowd flow optimization
- Emergency evacuation planning
- Multi-agent navigation benchmarks

Author: MFGarchon Team
Date: October 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    from mfgarchon.alg.reinforcement.environments.continuous_mfg_env_base import ContinuousMFGEnvBase

    BASE_AVAILABLE = True
except ImportError:
    BASE_AVAILABLE = False
    ContinuousMFGEnvBase = object  # type: ignore


class CrowdNavigationEnv(ContinuousMFGEnvBase):
    r"""
    Crowd Navigation Mean Field Game Environment.

    2D spatial navigation with goal-directed movement and collision avoidance.

    State Space:
    - Position: $(x, y) \in [0, L]^2$
    - Velocity: $(v_x, v_y) \in [-v_{\max}, v_{\max}]^2$
    - Goal: $(x_{goal}, y_{goal}) \in [0, L]^2$ (fixed per episode)

    Action Space:
    - Acceleration: $(a_x, a_y) \in [-a_{\max}, a_{\max}]^2$

    Dynamics:
    - Position: $\vec{x}_{t+1} = \vec{x}_t + \vec{v}_t \cdot dt$
    - Velocity: $\vec{v}_{t+1} = \vec{v}_t + \vec{a}_t \cdot dt + \sigma \sqrt{dt} \vec{\epsilon}$

    Reward:
    - Goal distance: $-c_{dist} \|\vec{x} - \vec{x}_{goal}\|^2$
    - Velocity penalty: $-c_{vel} \|\vec{v}\|^2$ (encourage smooth motion)
    - Control cost: $-c_{ctrl} \|\vec{a}\|^2$ (minimize effort)
    - Crowd avoidance: $-c_{crowd} \cdot \rho(\vec{x})$ (local density penalty)
    - Goal bonus: $+R_{goal}$ if within goal radius

    Mean Field Coupling:
    - Repulsive potential from nearby agents
    - Based on kernel: $K(\vec{x}, \vec{y}) = \exp(-\|\vec{x} - \vec{y}\|^2 / 2\sigma^2)$
    - Penalizes crowding and encourages spacing

    Termination:
    - Early: reaching goal (within goal_radius)
    - Truncation: max_steps reached
    """

    def __init__(
        self,
        num_agents: int = 100,
        domain_size: float = 10.0,
        v_max: float = 2.0,
        a_max: float = 1.0,
        cost_distance: float = 1.0,
        cost_velocity: float = 0.1,
        cost_control: float = 0.05,
        cost_crowd: float = 0.5,
        goal_bonus: float = 10.0,
        goal_radius: float = 0.5,
        crowd_kernel_std: float = 1.0,
        dt: float = 0.1,
        max_steps: int = 200,
        noise_std: float = 0.1,
        population_bins: int = 50,
    ):
        """
        Initialize Crowd Navigation environment.

        Args:
            num_agents: Number of agents in population
            domain_size: Size of square domain [0, L]^2
            v_max: Maximum velocity magnitude
            a_max: Maximum acceleration magnitude
            cost_distance: Weight for distance-to-goal penalty
            cost_velocity: Weight for velocity penalty (smooth motion)
            cost_control: Weight for control effort penalty
            cost_crowd: Weight for crowd density penalty
            goal_bonus: Reward for reaching goal
            goal_radius: Distance threshold for goal achievement
            crowd_kernel_std: Standard deviation for crowd repulsion kernel
            dt: Time step size
            max_steps: Maximum episode length
            noise_std: Standard deviation of velocity noise
            population_bins: Number of bins for 2D population histogram
        """
        self.domain_size = domain_size
        self.v_max = v_max
        self.a_max = a_max
        self.cost_distance = cost_distance
        self.cost_velocity = cost_velocity
        self.cost_control = cost_control
        self.cost_crowd = cost_crowd
        self.goal_bonus = goal_bonus
        self.goal_radius = goal_radius
        self.crowd_kernel_std = crowd_kernel_std

        # Goal position (set in reset)
        self.goal_position: NDArray[np.floating[Any]] | None = None

        # Initialize base class
        # State: (x, y, vx, vy, x_goal, y_goal)
        super().__init__(
            num_agents=num_agents,
            state_dim=6,
            action_dim=2,
            action_bounds=(-a_max, a_max),
            population_bins=population_bins,
            dt=dt,
            max_steps=max_steps,
            noise_std=noise_std,
        )

    def _get_state_bounds(self) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """
        Get state space bounds.

        Returns:
            (low, high) arrays with bounds for (x, y, vx, vy, x_goal, y_goal)
        """
        low = np.array(
            [
                0.0,  # x
                0.0,  # y
                -self.v_max,  # vx
                -self.v_max,  # vy
                0.0,  # x_goal
                0.0,  # y_goal
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                self.domain_size,  # x
                self.domain_size,  # y
                self.v_max,  # vx
                self.v_max,  # vy
                self.domain_size,  # x_goal
                self.domain_size,  # y_goal
            ],
            dtype=np.float32,
        )
        return low, high

    def _sample_initial_states(self) -> NDArray[np.floating[Any]]:
        """
        Sample initial states for all agents.

        Initial distribution:
        - Positions: random in domain
        - Velocities: zero (start from rest)
        - Goals: same for all agents in an episode

        Returns:
            Array of shape (num_agents, 6) with initial states
        """
        # Sample random positions in domain
        positions = self.rng.uniform(0, self.domain_size, size=(self.num_agents, 2)).astype(np.float32)

        # Start from rest
        velocities = np.zeros((self.num_agents, 2), dtype=np.float32)

        # Sample single goal for all agents
        self.goal_position = self.rng.uniform(0, self.domain_size, size=2).astype(np.float32)
        goals = np.tile(self.goal_position, (self.num_agents, 1))

        # Stack into (num_agents, 6) array
        states = np.concatenate([positions, velocities, goals], axis=1)
        return states

    def _drift(
        self,
        state: NDArray[np.floating[Any]],
        action: NDArray[np.floating[Any]],
        population: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """
        Compute drift term in dynamics.

        2D kinematic dynamics:
        - Position change: $d\vec{x}/dt = \vec{v}$
        - Velocity change: $d\vec{v}/dt = \vec{a}$

        Args:
            state: Current state (x, y, vx, vy, x_goal, y_goal)
            action: Current action (ax, ay)
            population: Population histogram (unused in kinematics)

        Returns:
            Drift vector (dx/dt, dy/dt, dvx/dt, dvy/dt, 0, 0)
        """
        vx, vy = state[2], state[3]
        ax, ay = action[0], action[1]

        # Kinematic dynamics
        dx_dt = vx
        dy_dt = vy
        dvx_dt = ax
        dvy_dt = ay

        # Goals don't change
        dgoal_dt = np.zeros(2, dtype=np.float32)

        return np.array([dx_dt, dy_dt, dvx_dt, dvy_dt, dgoal_dt[0], dgoal_dt[1]], dtype=np.float32)

    def _noise_mask(self) -> NDArray[np.floating[Any]]:
        """Goal coordinates (dims 4-5) are fixed per episode (``_drift`` returns 0 for them), so they
        must NOT receive Brownian noise (Issue #1600): random-walking goals silently corrupt the
        goal-reached termination and the distance-to-goal reward."""
        mask = np.ones(self.state_dim, dtype=np.float64)
        mask[4:6] = 0.0
        return mask

    def _individual_reward(
        self,
        state: NDArray[np.floating[Any]],
        action: NDArray[np.floating[Any]],
        next_state: NDArray[np.floating[Any]],
    ) -> float:
        r"""
        Compute individual reward (negative cost).

        Components:
        1. Distance to goal: $-c_{dist} \|\vec{x} - \vec{x}_{goal}\|^2$
        2. Velocity penalty: $-c_{vel} \|\vec{v}\|^2$
        3. Control cost: $-c_{ctrl} \|\vec{a}\|^2$
        4. Goal bonus: $+R_{goal}$ if within goal_radius

        Args:
            state: Current state
            action: Current action
            next_state: Next state (used for goal checking)

        Returns:
            Individual reward (negative cost + bonuses)
        """
        # Extract components
        x, y = next_state[0], next_state[1]
        vx, vy = next_state[2], next_state[3]
        x_goal, y_goal = next_state[4], next_state[5]
        ax, ay = action[0], action[1]

        # Distance to goal
        dist_to_goal = np.sqrt((x - x_goal) ** 2 + (y - y_goal) ** 2)
        distance_cost = self.cost_distance * dist_to_goal**2

        # Velocity penalty (encourage smooth motion)
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        velocity_cost = self.cost_velocity * velocity_magnitude**2

        # Control cost (minimize acceleration)
        control_magnitude = np.sqrt(ax**2 + ay**2)
        control_cost = self.cost_control * control_magnitude**2

        # Goal bonus (if within radius)
        goal_reached = dist_to_goal < self.goal_radius
        bonus = self.goal_bonus if goal_reached else 0.0

        # Total reward
        return float(bonus - distance_cost - velocity_cost - control_cost)

    def compute_mean_field_coupling(
        self, state: NDArray[np.floating[Any]], population: NDArray[np.floating[Any]]
    ) -> float:
        r"""
        Compute mean field interaction term.

        Crowd avoidance penalty based on local population density.
        Uses Gaussian kernel to measure crowding:

        $\rho(\vec{x}) = \sum_i K(\vec{x}, \vec{x}_i)$

        where $K(\vec{x}, \vec{y}) = \exp(-\|\vec{x} - \vec{y}\|^2 / 2\sigma^2)$

        Higher density → larger negative reward (encourages spacing).

        Args:
            state: Current state (x, y, vx, vy, x_goal, y_goal)
            population: Population histogram (2D grid)

        Returns:
            Mean field coupling term (negative crowd density penalty)
        """
        # Issue #1600: read the actual population histogram instead of a fixed distance-to-center
        # proxy that ignored `population`. get_population_state() bins the crowd by x-coordinate over
        # [0, domain_size]; the advertised Gaussian-kernel local density is
        # rho(x) = sum_b K(x, y_b) m_b with K(x, y) = exp(-(x - y)^2 / 2 sigma^2). Higher local
        # density -> larger penalty (encourages spacing). Discriminating: a crowd peaked at the
        # agent's x now yields a large penalty, a crowd far away a small one.
        x = float(state[0])
        edges = np.linspace(0.0, self.domain_size, self.population_bins + 1)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        local_density = float(np.sum(np.exp(-((x - bin_centers) ** 2) / (2 * self.crowd_kernel_std**2)) * population))

        # Crowd avoidance penalty
        crowd_cost = self.cost_crowd * local_density

        return float(-crowd_cost)

    def get_population_state(self) -> NDArray[np.floating[Any]]:
        """
        Get population distribution as 2D histogram over position space.

        Bins population by (x, y) coordinates only (ignores velocity and goal).

        Returns:
            Flattened 2D histogram of shape (population_bins,)
            Represents binned density over 2D spatial domain
        """
        if self.agent_states is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Extract positions from all agent states
        positions = self.agent_states[:, :2]  # (num_agents, 2)

        # Create 2D histogram over position space
        # Note: For simplicity, return 1D flattened histogram
        # Full implementation would use 2D bins, but base class expects 1D

        # Bin by x-coordinate only for compatibility with 1D base class histogram
        x_positions = positions[:, 0]
        histogram, _ = np.histogram(x_positions, bins=self.population_bins, range=(0, self.domain_size))

        # Normalize to probability distribution
        histogram = histogram.astype(np.float32)
        total = histogram.sum()
        if total > 0:
            histogram = histogram / total
        else:
            histogram = np.ones(self.population_bins, dtype=np.float32) / self.population_bins

        return histogram

    def _is_terminated(self, state: NDArray[np.floating[Any]]) -> bool:
        """
        Check if episode should terminate early.

        Terminates when agent reaches goal (within goal_radius).

        Args:
            state: Current state (x, y, vx, vy, x_goal, y_goal)

        Returns:
            True if goal reached, False otherwise
        """
        x, y = state[0], state[1]
        x_goal, y_goal = state[4], state[5]

        dist_to_goal = np.sqrt((x - x_goal) ** 2 + (y - y_goal) ** 2)
        return bool(dist_to_goal < self.goal_radius)

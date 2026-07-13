r"""
Traffic Flow Mean Field Game Environment.

Congestion-aware routing environment where agents choose paths through a network
while accounting for congestion effects from other agents. Demonstrates:
- Network navigation and routing
- Congestion dynamics (travel time increases with density)
- Strategic path selection
- Flow equilibrium in transportation networks

Mathematical Formulation:
- State: $(x, v, t_{\text{rem}}) \in [0, L] \times \mathbb{R}_+ \times \mathbb{R}_+$ (position, velocity, time remaining)
- Action: $a \in [-a_{\max}, a_{\max}]$ (acceleration)
- Dynamics: $x' = x + v \cdot dt$, $v' = v + a \cdot dt - \beta \cdot \rho(x)$ (congestion drag)
- Individual Cost: -progress + fuel cost + time penalty
- Mean Field Cost: congestion penalty proportional to local density
- Goal: Reach destination quickly while avoiding congested regions

Use Cases:
- Transportation network optimization
- Traffic flow modeling and control
- Multi-agent routing and navigation
- Congestion games in urban planning

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


class TrafficFlowEnv(ContinuousMFGEnvBase):
    r"""
    Traffic Flow Mean Field Game Environment.

    Congestion-aware navigation where agents travel along a 1D corridor from
    start (x=0) to destination (x=L) while managing velocity and congestion.

    State Space:
    - Position: $x \in [0, L]$
    - Velocity: $v \in [0, v_{\max}]$
    - Time remaining: $t_{\text{rem}} \in [0, T_{\max}]$

    Action Space:
    - Acceleration: $a \in [-a_{\max}, a_{\max}]$

    Dynamics:
    - Position: $x_{t+1} = x_t + v_t \cdot dt$
    - Velocity: $v_{t+1} = v_t + a_t \cdot dt - \beta \cdot \rho(x_t) \cdot v_t \cdot dt$
    - Time remaining: $t_{\text{rem},t+1} = t_{\text{rem},t} - dt$

    where $\rho(x)$ is local density (from mean field coupling) and $\beta$ is
    the congestion coefficient (higher density → velocity drag).

    Reward:
    - Progress: forward movement toward destination
    - Fuel cost: $-\kappa |a|$ (proportional to acceleration)
    - Time penalty: cost for slow travel
    - Arrival bonus: large reward for reaching destination

    Mean Field Coupling:
    - Congestion penalty: Higher local density increases travel time
    - Velocity drag: $-\beta \cdot \rho(x) \cdot v$ reduces speed in crowded regions
    - Strategic routing: Agents may slow down to avoid congested areas

    Termination:
    - Early: reached destination (x ≥ L)
    - Truncation: time limit exceeded (t_rem ≤ 0)
    """

    def __init__(
        self,
        num_agents: int = 100,
        corridor_length: float = 10.0,
        v_max: float = 2.0,
        a_max: float = 1.0,
        time_limit: float = 20.0,
        congestion_coeff: float = 0.5,
        fuel_cost: float = 0.1,
        time_penalty: float = 0.5,
        arrival_bonus: float = 10.0,
        dt: float = 0.1,
        max_steps: int = 200,
        noise_std: float = 0.05,
        population_bins: int = 50,
    ):
        """
        Initialize Traffic Flow environment.

        Args:
            num_agents: Number of agents in population
            corridor_length: Length of travel corridor (L)
            v_max: Maximum velocity
            a_max: Maximum acceleration (absolute value)
            time_limit: Maximum time allowed (T_max)
            congestion_coeff: Congestion drag coefficient (β)
            fuel_cost: Cost per unit acceleration (κ)
            time_penalty: Penalty per unit time
            arrival_bonus: Reward for reaching destination
            dt: Time step size
            max_steps: Maximum episode length
            noise_std: Standard deviation of dynamics noise
            population_bins: Number of bins for population histogram
        """
        self.corridor_length = corridor_length
        self.v_max = v_max
        self.a_max = a_max
        self.time_limit = time_limit
        self.congestion_coeff = congestion_coeff
        self.fuel_cost = fuel_cost
        self.time_penalty = time_penalty
        self.arrival_bonus = arrival_bonus

        # Initialize base class
        # State: (x, v, t_rem)
        super().__init__(
            num_agents=num_agents,
            state_dim=3,
            action_dim=1,
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
            (low, high) arrays with bounds for (x, v, t_rem)
        """
        low = np.array(
            [
                0.0,  # position
                0.0,  # velocity
                0.0,  # time remaining
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                self.corridor_length,  # position
                self.v_max,  # velocity
                self.time_limit,  # time remaining
            ],
            dtype=np.float32,
        )
        return low, high

    def _sample_initial_states(self) -> NDArray[np.floating[Any]]:
        """
        Sample initial states for all agents.

        Initial distribution:
        - Position: x = 0 (start of corridor)
        - Velocity: v = 0 (start from rest)
        - Time remaining: t_rem = time_limit (full time available)

        Returns:
            Array of shape (num_agents, 3) with initial states
        """
        # Start at beginning of corridor
        positions = np.zeros((self.num_agents, 1), dtype=np.float32)

        # Start from rest
        velocities = np.zeros((self.num_agents, 1), dtype=np.float32)

        # Full time available
        time_remaining = np.ones((self.num_agents, 1), dtype=np.float32) * self.time_limit

        # Stack into (num_agents, 3) array
        states = np.concatenate([positions, velocities, time_remaining], axis=1)
        return states

    def _drift(
        self,
        state: NDArray[np.floating[Any]],
        action: NDArray[np.floating[Any]],
        population: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """
        Compute drift term in dynamics.

        Traffic flow dynamics:
        - Position updates with velocity: dx/dt = v
        - Velocity updates with acceleration - congestion drag: dv/dt = a - β·ρ(x)·v
        - Time decreases: dt_rem/dt = -1

        Args:
            state: Current state (x, v, t_rem)
            action: Current action (acceleration a)
            population: Population histogram (used for density ρ(x))

        Returns:
            Drift vector (dx/dt, dv/dt, dt_rem/dt)
        """
        x, v, _t_rem = state[0], state[1], state[2]
        a = action[0]

        # Compute local density at current position
        local_density = self._compute_local_density(x, population)

        # Position dynamics
        dx_dt = v

        # Velocity dynamics (acceleration - congestion drag)
        congestion_drag = self.congestion_coeff * local_density * v
        dv_dt = a - congestion_drag

        # Time remaining decreases
        dt_rem_dt = -1.0

        return np.array([dx_dt, dv_dt, dt_rem_dt], dtype=np.float32)

    def _noise_mask(self) -> NDArray[np.floating[Any]]:
        """Time-remaining (dim 2) is a deterministic countdown (``_drift`` = -1), not a diffusion, so
        it must NOT receive Brownian noise (Issue #1600): noise would make the deadline random."""
        mask = np.ones(self.state_dim, dtype=np.float64)
        mask[2] = 0.0
        return mask

    def _compute_local_density(self, position: float, population: NDArray[np.floating[Any]]) -> float:
        """
        Compute local density at given position from population distribution.

        Args:
            position: Position x ∈ [0, L]
            population: Population histogram (binned by position)

        Returns:
            Local density ρ(x) at given position
        """
        # Map position to population bin
        bin_idx = int(position / self.corridor_length * self.population_bins)
        bin_idx = np.clip(bin_idx, 0, self.population_bins - 1)

        # Return density at this bin
        return float(population[bin_idx])

    def _individual_reward(
        self,
        state: NDArray[np.floating[Any]],
        action: NDArray[np.floating[Any]],
        next_state: NDArray[np.floating[Any]],
    ) -> float:
        r"""
        Compute individual reward (negative cost).

        Components:
        1. Progress reward: distance traveled toward destination
        2. Fuel cost: $-\kappa |a|$ (proportional to acceleration)
        3. Time penalty: cost for slow travel
        4. Arrival bonus: large reward if reached destination

        Args:
            state: Current state (x, v, t_rem)
            action: Current action (acceleration a)
            next_state: Next state (used for progress calculation)

        Returns:
            Individual reward (negative cost + progress + bonuses)
        """
        x_current = state[0]
        x_next = next_state[0]
        a = action[0]

        # Progress reward (distance traveled)
        progress = x_next - x_current
        progress_reward = progress  # Positive for forward movement

        # Fuel cost (proportional to acceleration magnitude)
        fuel_penalty = self.fuel_cost * abs(a)

        # Time penalty (encourage faster travel)
        time_cost = self.time_penalty * self.dt

        # Arrival bonus (large reward for reaching destination)
        arrival_reward = 0.0
        if x_next >= self.corridor_length:
            arrival_reward = self.arrival_bonus

        # Total reward
        return float(progress_reward - fuel_penalty - time_cost + arrival_reward)

    def compute_mean_field_coupling(
        self, state: NDArray[np.floating[Any]], population: NDArray[np.floating[Any]]
    ) -> float:
        r"""
        Compute mean field interaction term.

        Congestion penalty for traveling through crowded regions:
        - Higher local density → stronger penalty
        - Penalty proportional to velocity (faster travel in crowded areas costs more)

        Args:
            state: Current state (x, v, t_rem)
            population: Population histogram (binned by position)

        Returns:
            Mean field coupling term (negative congestion penalty)
        """
        x, v = state[0], state[1]

        # Compute local density
        local_density = self._compute_local_density(x, population)

        # Congestion penalty (proportional to velocity and density)
        # High speed in high density → high penalty
        congestion_cost = self.congestion_coeff * local_density * v

        return float(-congestion_cost)

    def get_population_state(self) -> NDArray[np.floating[Any]]:
        """
        Get population distribution as histogram over position space.

        Bins population by position coordinate only (ignores velocity and time).

        Returns:
            Flattened histogram of shape (population_bins,)
            Represents binned density over position domain [0, L]
        """
        if self.agent_states is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Extract positions from all agent states
        positions = self.agent_states[:, 0]  # (num_agents,)

        # Create histogram over position space
        histogram, _ = np.histogram(positions, bins=self.population_bins, range=(0, self.corridor_length))

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

        Terminates when agent reaches destination (x ≥ L).

        Args:
            state: Current state (x, v, t_rem)

        Returns:
            True if destination reached, False otherwise
        """
        x = state[0]
        return bool(x >= self.corridor_length)

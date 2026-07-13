r"""
Linear-Quadratic Mean Field Game Environment.

Simple MFG environment with linear dynamics and quadratic costs for:
- Validating the ContinuousMFGEnvBase class
- Fast training and algorithm testing
- Known analytical solutions for verification

Mathematical Formulation:
- State: $(x, v) \in \mathbb{R}^2$ (position, velocity)
- Action: $u \in [-u_{\max}, u_{\max}]$ (control/acceleration)
- Dynamics: $x' = x + v \cdot dt$, $v' = v + u \cdot dt + \sigma \epsilon$
- Individual Cost: $c_x x^2 + c_u u^2$ (state regulation + control effort)
- Mean Field Cost: $c_m \int (x - y)^2 m(y) dy$ (quadratic repulsion)
- Goal: Drive position to origin while avoiding congestion

This is the simplest possible MFG environment - a "hello world" for testing.

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


class LQMFGEnv(ContinuousMFGEnvBase):
    r"""
    Linear-Quadratic Mean Field Game Environment.

    Simplest MFG with:
    - Linear dynamics (position-velocity model)
    - Quadratic costs (LQR-style)
    - Known analytical solution (Nash equilibrium)

    State Space:
    - Position: $x \in [-x_{\max}, x_{\max}]$
    - Velocity: $v \in [-v_{\max}, v_{\max}]$

    Action Space:
    - Control: $u \in [-u_{\max}, u_{\max}]$

    Dynamics:
    - $x_{t+1} = x_t + v_t \cdot dt$
    - $v_{t+1} = v_t + u_t \cdot dt + \sigma \sqrt{dt} \epsilon$

    Reward:
    - Individual: $-c_x x^2 - c_u u^2$
    - Mean Field: $-c_m \int (x - y)^2 m(y) dy$
    - Total: minimize distance to origin + control cost + congestion

    Use Case:
    - Validate base class API works correctly
    - Fast training for algorithm sanity checks
    - Baseline for comparing more complex environments
    """

    def __init__(
        self,
        num_agents: int = 50,
        x_max: float = 10.0,
        v_max: float = 5.0,
        u_max: float = 2.0,
        cost_state: float = 1.0,
        cost_control: float = 0.1,
        cost_mean_field: float = 0.5,
        dt: float = 0.05,
        max_steps: int = 200,
        noise_std: float = 0.1,
        population_bins: int = 50,
    ):
        """
        Initialize LQ-MFG environment.

        Args:
            num_agents: Number of agents in population
            x_max: Maximum position (state bounds: [-x_max, x_max])
            v_max: Maximum velocity (state bounds: [-v_max, v_max])
            u_max: Maximum control (action bounds: [-u_max, u_max])
            cost_state: Weight for state cost (penalize distance from origin)
            cost_control: Weight for control cost (penalize large actions)
            cost_mean_field: Weight for mean field interaction (congestion penalty)
            dt: Time step size
            max_steps: Maximum episode length
            noise_std: Standard deviation of velocity noise
            population_bins: Number of bins for population histogram
        """
        self.x_max = x_max
        self.v_max = v_max
        self.u_max = u_max
        self.cost_state = cost_state
        self.cost_control = cost_control
        self.cost_mean_field = cost_mean_field

        # Initialize base class
        super().__init__(
            num_agents=num_agents,
            state_dim=2,  # (x, v)
            action_dim=1,  # u
            action_bounds=(-u_max, u_max),
            population_bins=population_bins,
            dt=dt,
            max_steps=max_steps,
            noise_std=noise_std,
        )

    def _get_state_bounds(self) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """
        Get state space bounds.

        Returns:
            (low, high) arrays with bounds for (x, v)
        """
        low = np.array([-self.x_max, -self.v_max], dtype=np.float32)
        high = np.array([self.x_max, self.v_max], dtype=np.float32)
        return low, high

    def _sample_initial_states(self) -> NDArray[np.floating[Any]]:
        """
        Sample initial states for all agents.

        Initial distribution: random positions and velocities near origin.

        Returns:
            Array of shape (num_agents, 2) with initial (x, v) pairs
        """
        # Sample from Gaussian centered at origin
        # Position: N(0, x_max/3)
        # Velocity: N(0, v_max/3)
        positions = self.rng.normal(0, self.x_max / 3, size=self.num_agents).astype(np.float32)
        velocities = self.rng.normal(0, self.v_max / 3, size=self.num_agents).astype(np.float32)

        # Clip to bounds
        positions = np.clip(positions, -self.x_max, self.x_max)
        velocities = np.clip(velocities, -self.v_max, self.v_max)

        # Stack into (num_agents, 2) array
        states = np.stack([positions, velocities], axis=1)
        return states

    def _drift(
        self,
        state: NDArray[np.floating[Any]],
        action: NDArray[np.floating[Any]],
        population: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """
        Compute drift term in dynamics.

        Linear dynamics:
        - Position change: $dx/dt = v$
        - Velocity change: $dv/dt = u$

        Args:
            state: Current state (x, v)
            action: Current action (u,)
            population: Population histogram (unused in linear dynamics)

        Returns:
            Drift vector (dx/dt, dv/dt)
        """
        v = state[1]
        u = action[0]

        # Linear dynamics
        dx_dt = v
        dv_dt = u

        return np.array([dx_dt, dv_dt], dtype=np.float32)

    def _individual_reward(
        self,
        state: NDArray[np.floating[Any]],
        action: NDArray[np.floating[Any]],
        next_state: NDArray[np.floating[Any]],
    ) -> float:
        """
        Compute individual reward (negative of LQR cost).

        Cost: $c_x x^2 + c_u u^2$
        Reward: $-c_x x^2 - c_u u^2$

        Args:
            state: Current state (x, v)
            action: Current action (u,)
            next_state: Next state (unused, cost is on current state)

        Returns:
            Individual reward (negative cost)
        """
        x = state[0]
        u = action[0]

        # Quadratic cost on position and control
        state_cost = self.cost_state * x**2
        control_cost = self.cost_control * u**2

        # Return negative cost (reward)
        return float(-state_cost - control_cost)

    def compute_mean_field_coupling(
        self, state: NDArray[np.floating[Any]], population: NDArray[np.floating[Any]]
    ) -> float:
        r"""
        Compute mean field interaction term.

        Quadratic coordination cost against the population density $m$:
        $r_{\text{MF}}(x) = -c_m \int (x - y)^2 m(y) \, dy$

        The cost is minimised when the agent sits at the population barycentre, so it rewards
        staying with the crowd (coordination), and grows as the agent moves away from the mass.

        Args:
            state: Current state (x, v)
            population: Population histogram $m$ over position bins (Issue #1600: now read)

        Returns:
            Mean field coupling term (negative coordination cost)
        """
        x = float(state[0])

        # Issue #1600: read the actual population histogram instead of the -c_m*x^2 placeholder.
        # Bin centres match get_population_state's np.histogram(range=(-x_max, x_max)); the discrete
        # form of the advertised integral is c_m * sum_b (x - y_b)^2 m_b. When the population
        # concentrates in one bin b0 (m -> delta_{b0}) it reduces to -c_m*(x - y_{b0})^2, i.e. the old
        # -c_m*x^2 up to the origin bin-centre offset y_{b0} (O(bin_width); zero only if a bin centre
        # lands on 0) -- so it generalises the prior behaviour rather than replacing it.
        edges = np.linspace(-self.x_max, self.x_max, self.population_bins + 1)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        mean_field_cost = self.cost_mean_field * float(np.sum((x - bin_centers) ** 2 * population))

        # Return negative cost (reward)
        return float(-mean_field_cost)

    def get_population_state(self) -> NDArray[np.floating[Any]]:
        """
        Get population distribution as histogram over position space.

        Bins population by position (x coordinate only).

        Returns:
            Histogram of shape (population_bins,) normalized to sum to 1
        """
        if self.agent_states is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Extract positions from all agent states
        positions = self.agent_states[:, 0]

        # Create histogram over position space
        histogram, _ = np.histogram(positions, bins=self.population_bins, range=(-self.x_max, self.x_max))

        # Normalize to probability distribution
        histogram = histogram.astype(np.float32)
        total = histogram.sum()
        if total > 0:
            histogram = histogram / total
        else:
            # Uniform distribution if no agents (shouldn't happen)
            histogram = np.ones(self.population_bins, dtype=np.float32) / self.population_bins

        return histogram

    def _is_terminated(self, state: NDArray[np.floating[Any]]) -> bool:
        """
        Check if episode should terminate early.

        For LQ-MFG, no early termination (always run to max_steps).

        Args:
            state: Current state

        Returns:
            False (no early termination)
        """
        return False

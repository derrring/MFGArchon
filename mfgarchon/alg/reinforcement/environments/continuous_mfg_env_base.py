"""
Base class for continuous action Mean Field Game environments.

Provides Gymnasium-compatible API for single-population MFG environments with:
- Continuous action spaces
- Population distribution tracking via histograms
- Mean field coupling in rewards
- Standardized observation/action spaces

Mathematical Framework:
- State space: $x \\in \\mathcal{X} \\subset \\mathbb{R}^d$
- Action space: $u \\in \\mathcal{U} \\subset \\mathbb{R}^m$
- Population distribution: $m(x) \\in \\mathcal{P}(\\mathcal{X})$
- Dynamics: $dx = f(x, u, m) dt + \\sigma dW$
- Reward: $r(x, u, m) = r_0(x, u) + r_{\text{MF}}(x, m)$

Author: MFGarchon Team
Date: October 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    import gymnasium as gym  # noqa: F401
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError as e:
    raise ImportError("gymnasium required for continuous MFG environments. Install with: pip install gymnasium") from e


class ContinuousMFGEnvBase(ABC):
    """
    Base class for continuous action MFG environments.

    Gymnasium-compatible interface for single-population Mean Field Games with:
    - Continuous state and action spaces
    - Population distribution tracking
    - Mean field coupling in dynamics and rewards
    - Stochastic dynamics with Brownian noise

    Key Design:
    - Each agent observes: (individual_state, population_histogram)
    - Agent acts in continuous action space
    - Reward depends on state, action, and population distribution
    - Population evolves according to all agents' policies

    Mathematical Components:
    1. Individual Dynamics:
       $x_{t+1} = x_t + f(x_t, u_t, m_t) \\cdot dt + \\sigma \\sqrt{dt} \\cdot \\epsilon$

    2. Population Evolution:
       $m_{t+1}(x) = $ empirical distribution of all agents at time $t+1$

    3. Reward Structure:
       $r_t = r_0(x_t, u_t) + r_{\text{MF}}(x_t, m_t)$
    """

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        action_bounds: tuple[float, float] = (-1.0, 1.0),
        population_bins: int = 100,
        dt: float = 0.01,
        max_steps: int = 200,
        noise_std: float = 0.1,
    ):
        """
        Initialize continuous MFG environment.

        Args:
            num_agents: Number of agents in population (for discretization)
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_bounds: (min, max) bounds for actions
            population_bins: Number of bins for population histogram
            dt: Time step size
            max_steps: Maximum episode length
            noise_std: Standard deviation of Brownian noise
        """
        if not GYMNASIUM_AVAILABLE:
            raise ImportError("gymnasium required for continuous MFG environments. Install with: pip install gymnasium")

        if num_agents < 1:
            raise ValueError(f"num_agents must be >= 1, got {num_agents}")
        if state_dim < 1:
            raise ValueError(f"state_dim must be >= 1, got {state_dim}")
        if action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}")
        if population_bins < 1:
            raise ValueError(f"population_bins must be >= 1, got {population_bins}")

        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.population_bins = population_bins
        self.dt = dt
        self.max_steps = max_steps
        self.noise_std = noise_std

        # Gymnasium spaces
        state_low, state_high = self._get_state_bounds()
        self.observation_space = spaces.Box(low=state_low, high=state_high, shape=(state_dim,), dtype=np.float32)

        action_low = np.full(action_dim, action_bounds[0], dtype=np.float32)
        action_high = np.full(action_dim, action_bounds[1], dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(action_dim,), dtype=np.float32)

        # Episode state
        self.current_step = 0
        self.agent_states: NDArray[np.floating[Any]] | None = None
        self.population_histogram: NDArray[np.floating[Any]] | None = None

        # Random number generator
        self.rng = np.random.default_rng()

    @abstractmethod
    def _get_state_bounds(self) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        r"""
        Get bounds for state space.

        Returns:
            Tuple (low, high) where each is array of shape (state_dim,)
            Defines the state space bounds: $x \in [low, high]$
        """

    @abstractmethod
    def _sample_initial_states(self) -> NDArray[np.floating[Any]]:
        """
        Sample initial states for all agents.

        Returns:
            Array of shape (num_agents, state_dim) with initial states
            Should implement problem-specific initialization distribution
        """

    @abstractmethod
    def _drift(
        self, state: NDArray[np.floating[Any]], action: NDArray[np.floating[Any]], population: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        r"""
        Compute drift term in dynamics: $f(x, u, m)$

        Args:
            state: Current state $x \in \mathbb{R}^{state\_dim}$
            action: Current action $u \in \mathbb{R}^{action\_dim}$
            population: Population histogram $m \in \mathbb{R}^{population\_bins}$

        Returns:
            Drift vector $f(x, u, m) \in \mathbb{R}^{state\_dim}$
        """

    @abstractmethod
    def compute_mean_field_coupling(
        self, state: NDArray[np.floating[Any]], population: NDArray[np.floating[Any]]
    ) -> float:
        """
        Compute mean field interaction term for reward.

        This captures how the population distribution affects an agent's reward.
        Common examples:
        - Congestion: penalty proportional to local density
        - Coordination: reward for matching population mean
        - Repulsion: penalty for proximity to other agents

        Args:
            state: Current state $x$
            population: Population histogram $m$

        Returns:
            Mean field coupling term $r_{\text{MF}}(x, m)$
        """

    def _individual_reward(
        self, state: NDArray[np.floating[Any]], action: NDArray[np.floating[Any]], next_state: NDArray[np.floating[Any]]
    ) -> float:
        """
        Compute individual reward term: $r_0(x, u)$

        Default implementation: zero individual reward (pure MF coupling).
        Override for problem-specific individual costs.

        Args:
            state: Current state $x$
            action: Current action $u$
            next_state: Next state $x'$

        Returns:
            Individual reward $r_0(x, u)$
        """
        return 0.0

    def _compute_reward(
        self,
        state: NDArray[np.floating[Any]],
        action: NDArray[np.floating[Any]],
        next_state: NDArray[np.floating[Any]],
        population: NDArray[np.floating[Any]],
    ) -> float:
        """
        Compute total reward: $r = r_0(x, u) + r_{\text{MF}}(x, m)$

        Args:
            state: Current state
            action: Current action
            next_state: Next state
            population: Population histogram

        Returns:
            Total reward
        """
        individual = self._individual_reward(state, action, next_state)
        mean_field = self.compute_mean_field_coupling(state, population)
        return individual + mean_field

    def get_population_state(self) -> NDArray[np.floating[Any]]:
        """
        Get current population distribution as histogram.

        Computes empirical distribution:
        $m(x) \approx \frac{1}{N} \\sum_{i=1}^N \\delta_{x_i}(x)$

        Returns:
            Population histogram of shape (population_bins,)
            Normalized to sum to 1.0
        """
        if self.agent_states is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # For now, return uniform distribution as placeholder
        # Subclasses should implement proper binning based on state space structure
        histogram = np.ones(self.population_bins, dtype=np.float32) / self.population_bins
        return histogram

    def _noise_mask(self) -> NDArray[np.floating[Any]]:
        """Per-dimension 0/1 multiplier selecting which state dims receive Brownian noise.

        Default: every dim is dynamic (all ones). Override to zero the DECLARED-CONSTANT dims --
        those whose ``_drift`` is identically zero (e.g. fixed goal coordinates) or a deterministic
        countdown (drift = -1) -- so they do not random-walk (Issue #1600). Shape (state_dim,); it
        broadcasts over the (num_agents, state_dim) population increment in ``_advance_population``.
        """
        return np.ones(self.state_dim, dtype=np.float64)

    def _postprocess_next_states(self, next_states: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Per-agent post-processing of the advanced population, shape (num_agents, state_dim).

        Default: identity. Override for hard state constraints that must hold for EVERY agent (e.g.
        the resource-allocation simplex projection). Overriding this hook -- rather than the whole
        ``step()`` -- keeps ``_advance_population`` the single source for the Euler-Maruyama update
        (Issue #1600), so the population-frozen and constant-dim-noise fixes cannot silently
        re-diverge into a per-env ``step()`` copy.
        """
        return next_states

    def _advance_population(
        self, action: NDArray[np.floating[Any]], population: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        r"""Advance ALL agents one Euler-Maruyama step against the pre-step ``population``.

        The empirical density ``get_population_state()`` bins is real only if every agent it bins
        actually moves. The prior ``step()`` advanced only ``agent_states[0]``, freezing agents
        ``1..N-1`` at their reset positions, so the coupling density stayed ~= m_0 forever
        (Issue #1600). Each agent takes the SAME ego action -- the single-agent Gym interface
        controls one policy, so the population co-moves as a caricature crowd; a self-consistent
        fictitious-play equilibrium (per-agent best response pi(x_i, m)) is #1570/#887 territory,
        not this fix -- but each agent's drift still differs through its own state. Constant dims
        are held fixed by ``_noise_mask()``; per-agent hard constraints by ``_postprocess_next_states``.
        """
        states = self.agent_states
        assert states is not None
        drift = np.stack([self._drift(states[i], action, population) for i in range(states.shape[0])])
        noise = self.rng.normal(0.0, self.noise_std * np.sqrt(self.dt), size=states.shape) * self._noise_mask()
        next_states = states + drift * self.dt + noise
        state_low, state_high = self._get_state_bounds()
        next_states = np.clip(next_states, state_low, state_high)
        # Keep agent_states float32 (the observation_space dtype); the old in-place agent_states[0]
        # assignment downcast, but we now replace the whole array (float64 from rng/mask).
        return self._postprocess_next_states(next_states).astype(np.float32, copy=False)

    def step(
        self, action: NDArray[np.floating[Any]]
    ) -> tuple[NDArray[np.floating[Any]], float, bool, bool, dict[str, Any]]:
        r"""
        Execute one timestep of the MFG dynamics.

        The controlled ego agent is ``agent_states[0]`` (its next state is the returned observation
        and drives the reward/termination). The whole population advances each step
        (``_advance_population``) so the empirical mean field ``get_population_state()`` bins is real
        rather than frozen at the reset sample (Issue #1600).

        Args:
            action: Action $u \in \mathbb{R}^{action\_dim}$

        Returns:
            Tuple (observation, reward, terminated, truncated, info):
            - observation: Next state $x'$ of the ego agent
            - reward: Reward $r(x, u, m)$ for the ego agent
            - terminated: Whether episode ended naturally
            - truncated: Whether episode hit time limit
            - info: Additional diagnostic information
        """
        if self.agent_states is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Clip action to bounds
        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])

        # Ego (controlled) agent state and the pre-step population the dynamics/reward couple to
        state = self.agent_states[0]
        population = self.get_population_state()

        # Advance the WHOLE population one Euler-Maruyama step (Issue #1600) and read the ego next state
        self.agent_states = self._advance_population(action, population)
        next_state = self.agent_states[0]

        # Reward for the ego agent against the pre-step mean field
        reward = self._compute_reward(state, action, next_state, population)

        # Check termination
        terminated = self._is_terminated(next_state)
        truncated = self.current_step >= self.max_steps - 1

        self.current_step += 1

        info = {
            "step": self.current_step,
            "population_mass": float(np.sum(population)),
        }

        return next_state.astype(np.float32), float(reward), terminated, truncated, info

    def _is_terminated(self, state: NDArray[np.floating[Any]]) -> bool:
        """
        Check if episode should terminate.

        Default: no early termination.
        Override for problem-specific terminal conditions.

        Args:
            state: Current state

        Returns:
            True if episode should end
        """
        return False

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.floating[Any]], dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            Tuple (observation, info):
            - observation: Initial state $x_0$
            - info: Additional information
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0

        # Sample initial states for all agents
        self.agent_states = self._sample_initial_states()

        # Compute initial population distribution
        self.population_histogram = self.get_population_state()

        # Return initial state of first agent (representative)
        initial_state = self.agent_states[0]

        info = {
            "step": 0,
            "population_mass": np.sum(self.population_histogram),
        }

        return initial_state.astype(np.float32), info

    def render(self) -> None:  # noqa: B027
        """
        Render environment state.

        Default: no rendering.
        Override for problem-specific visualization.
        """

    def close(self) -> None:  # noqa: B027
        """Clean up resources."""

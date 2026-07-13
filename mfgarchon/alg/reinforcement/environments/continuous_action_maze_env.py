"""
Continuous Action Maze Environment for Mean Field Games.

Implements:
- Continuous velocity control: a = (vₓ, vᵧ) ∈ [-v_max, v_max]²
- Physics-based movement: x_{t+1} = x_t + dt·a
- Population density tracking
- Gymnasium-compatible API

Mathematical Framework:
- State: s = (x, y) ∈ [0, H] × [0, W]
- Action: a = (vₓ, vᵧ) ∈ [-v_max, v_max]²
- Dynamics: s_{t+1} = s_t + dt·a + ε, ε ~ N(0, σ²)
- Reward: r(s, a, m) = -||s - s_goal|| - λ₁·||a|| - λ₂·m(s)

Author: MFGarchon Team
Date: October 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:  # pragma: no cover
    GYMNASIUM_AVAILABLE = False
    gym = None
    spaces = None


class RewardType(str, Enum):
    """Reward structure types."""

    SPARSE = "sparse"  # Only goal reward
    DENSE = "dense"  # Distance-based shaping
    MFG = "mfg"  # Mean field game reward (goal + distance + congestion + control cost)


@dataclass
class ContinuousActionMazeConfig:
    """Configuration for continuous action maze environment."""

    maze_array: NDArray  # 2D array: 0=free, 1=wall
    num_agents: int = 10
    max_steps: int = 200
    reward_type: RewardType = RewardType.MFG
    dt: float = 0.1  # Time step for physics
    velocity_max: float = 2.0  # Maximum velocity
    noise_std: float = 0.0  # Movement noise (for stochasticity)
    start_positions: list[tuple[int, int]] | None = None
    goal_positions: list[tuple[int, int]] | None = None
    goal_reward: float = 10.0
    collision_penalty: float = -1.0
    congestion_weight: float = 1.0  # Weight for population density penalty
    control_cost_weight: float = 0.1  # Weight for ||a||² penalty
    population_smoothing: float = 0.5  # Gaussian smoothing for density


class PopulationState:
    """Tracks population density for mean field coupling."""

    def __init__(self, maze_shape: tuple[int, int], smoothing: float = 0.5):
        self.maze_shape = maze_shape
        self.smoothing = smoothing
        self.density_histogram = np.zeros(maze_shape, dtype=np.float32)

    def update(self, agent_positions: list[tuple[float, float]]) -> None:
        """Update density from continuous agent positions."""
        self.density_histogram.fill(0)

        H, W = self.maze_shape

        for x, y in agent_positions:
            # Clip to valid range
            x = np.clip(x, 0, H - 1)
            y = np.clip(y, 0, W - 1)

            # Bilinear interpolation for continuous positions
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            x1, y1 = min(x0 + 1, H - 1), min(y0 + 1, W - 1)

            wx = x - x0
            wy = y - y0

            self.density_histogram[x0, y0] += (1 - wx) * (1 - wy)
            self.density_histogram[x0, y1] += (1 - wx) * wy
            self.density_histogram[x1, y0] += wx * (1 - wy)
            self.density_histogram[x1, y1] += wx * wy

        # Normalize
        total = self.density_histogram.sum()
        if total > 0:
            self.density_histogram /= total

        # Apply Gaussian smoothing
        if self.smoothing > 0:
            self.density_histogram = self._gaussian_smooth(self.density_histogram)

    def _gaussian_smooth(self, density: NDArray) -> NDArray:
        """Apply Gaussian smoothing to density."""
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(density, sigma=self.smoothing, mode="constant")

    def get_local_density(self, position: tuple[float, float], radius: int = 1) -> float:
        """Get local density around position."""
        x, y = position
        x_int, y_int = int(np.round(x)), int(np.round(y))

        H, W = self.maze_shape
        x_min = max(0, x_int - radius)
        x_max = min(H, x_int + radius + 1)
        y_min = max(0, y_int - radius)
        y_max = min(W, y_int + radius + 1)

        return self.density_histogram[x_min:x_max, y_min:y_max].sum()


class ContinuousActionMazeEnvironment:
    """
    Gymnasium environment for continuous control in MFG.

    Features:
    - Continuous velocity control a = (vₓ, vᵧ) ∈ [-v_max, v_max]²
    - Physics-based movement with optional noise
    - Population density tracking
    - Mean field reward coupling
    """

    def __init__(self, config: ContinuousActionMazeConfig):
        """Initialize continuous action maze environment."""
        if not GYMNASIUM_AVAILABLE:
            raise ImportError("Gymnasium is required. Install with: pip install gymnasium")

        self.config = config
        self.maze = config.maze_array
        self.H, self.W = self.maze.shape

        # Agent state
        self.num_agents = config.num_agents
        self.agent_positions = np.zeros((self.num_agents, 2), dtype=np.float32)  # Continuous (x, y)
        self.agents_done = np.zeros(self.num_agents, dtype=bool)

        # Start/goal positions
        self.start_positions = config.start_positions or self._get_valid_positions()
        self.goal_positions = config.goal_positions or self._get_valid_positions()

        # Population state
        self.population_state = PopulationState(self.maze.shape, smoothing=config.population_smoothing)

        # Spaces
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0]), high=np.array([float(self.H - 1), float(self.W - 1)]), dtype=np.float32
        )

        self.action_space = spaces.Box(low=-config.velocity_max, high=config.velocity_max, shape=(2,), dtype=np.float32)

        # Episode state
        self.current_step = 0
        self.rng = np.random.RandomState()

    def _get_valid_positions(self) -> list[tuple[int, int]]:
        """Get all valid (free) positions in maze."""
        free_cells = np.argwhere(self.maze == 0)
        return [(int(x), int(y)) for x, y in free_cells]

    def reset(self, seed: int | None = None) -> tuple[NDArray, dict[str, Any]]:
        """Reset environment."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # Reset agents to start positions
        for i in range(self.num_agents):
            pos_idx = i % len(self.start_positions)
            self.agent_positions[i] = np.array(self.start_positions[pos_idx], dtype=np.float32)

        self.agents_done.fill(False)
        self.current_step = 0

        # Update population state
        self.population_state.update(self.agent_positions.tolist())

        return self.agent_positions.copy(), {}

    def step(self, actions: NDArray) -> tuple[NDArray, NDArray, bool, bool, dict[str, Any]]:
        """
        Execute continuous actions.

        Args:
            actions: Continuous velocities [num_agents, 2] in range [-v_max, v_max]²

        Returns:
            observations, rewards, terminated, truncated, info
        """
        actions = np.array(actions, dtype=np.float32)
        assert actions.shape == (
            self.num_agents,
            2,
        ), f"Expected actions shape {(self.num_agents, 2)}, got {actions.shape}"

        rewards = np.zeros(self.num_agents, dtype=np.float32)

        # Execute actions (physics-based movement)
        for i in range(self.num_agents):
            if self.agents_done[i]:
                continue

            # Apply action with time step
            velocity = np.clip(actions[i], -self.config.velocity_max, self.config.velocity_max)

            # Add movement noise if configured
            if self.config.noise_std > 0:
                noise = self.rng.normal(0, self.config.noise_std, size=2)
                velocity = velocity + noise

            # Update position
            new_pos = self.agent_positions[i] + self.config.dt * velocity

            # Clip to bounds
            new_pos[0] = np.clip(new_pos[0], 0, self.H - 1)
            new_pos[1] = np.clip(new_pos[1], 0, self.W - 1)

            # Check wall collision
            x_int, y_int = int(np.round(new_pos[0])), int(np.round(new_pos[1]))
            if self.maze[x_int, y_int] == 1:
                # Collision with wall - don't move, apply penalty
                rewards[i] += self.config.collision_penalty
            else:
                # Valid move
                self.agent_positions[i] = new_pos

            # Compute reward
            rewards[i] += self._compute_reward(i, velocity)

            # Check goal
            if self._is_at_goal(i):
                rewards[i] += self.config.goal_reward
                self.agents_done[i] = True

        # Update population state
        self.population_state.update(self.agent_positions.tolist())

        # Episode termination
        self.current_step += 1
        terminated = self.agents_done.all()
        truncated = self.current_step >= self.config.max_steps

        info = {
            "agents_done": self.agents_done.copy(),
            "population_density": self.population_state.density_histogram.copy(),
        }

        return self.agent_positions.copy(), rewards, terminated, truncated, info

    def _compute_reward(self, agent_idx: int, velocity: NDArray) -> float:
        """Compute reward for agent."""
        pos = self.agent_positions[agent_idx]

        if self.config.reward_type == RewardType.SPARSE:
            return 0.0  # Only goal reward

        # Distance to goal
        goal_pos = np.array(self.goal_positions[agent_idx % len(self.goal_positions)])
        distance = np.linalg.norm(pos - goal_pos)

        reward = -distance

        if self.config.reward_type == RewardType.MFG:
            # Congestion penalty
            local_density = self.population_state.get_local_density(tuple(pos), radius=2)
            reward -= self.config.congestion_weight * local_density

            # Control cost
            control_cost = np.linalg.norm(velocity) ** 2
            reward -= self.config.control_cost_weight * control_cost

        return float(reward)

    def _is_at_goal(self, agent_idx: int) -> bool:
        """Check if agent reached goal."""
        pos = self.agent_positions[agent_idx]
        goal_pos = np.array(self.goal_positions[agent_idx % len(self.goal_positions)])
        return bool(np.linalg.norm(pos - goal_pos) < 1.0)  # Within 1 cell

    def get_population_state(self) -> NDArray[np.floating[Any]]:
        """Current population density as a flat histogram.

        #1570 (Option A): the canonical env->algo population-state contract is a
        flat ``NDArray`` (matching ``ContinuousMFGEnvBase.get_population_state``),
        so mean-field DDPG/TD3/SAC consume every env identically. ``PopulationState``
        remains this env's internal density container (``self.population_state``).
        """
        return self.population_state.density_histogram.flatten()

    def render(self) -> NDArray | None:
        """Render environment (returns maze with agent positions)."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw maze
        ax.imshow(self.maze.T, cmap="binary", origin="lower")

        # Draw population density
        density_plot = np.ma.masked_where(
            self.population_state.density_histogram.T < 0.001, self.population_state.density_histogram.T
        )
        ax.imshow(density_plot, cmap="Reds", origin="lower", alpha=0.5, vmin=0, vmax=density_plot.max())

        # Draw agents
        active_positions = self.agent_positions[~self.agents_done]
        if len(active_positions) > 0:
            ax.scatter(active_positions[:, 0], active_positions[:, 1], c="blue", s=50, label="Agents", zorder=3)

        # Draw goals
        goal_array = np.array(self.goal_positions)
        ax.scatter(goal_array[:, 0], goal_array[:, 1], c="green", s=100, marker="*", label="Goals", zorder=3)

        ax.set_title(f"Step {self.current_step}/{self.config.max_steps}")
        ax.legend()
        ax.set_xlim(-0.5, self.H - 0.5)
        ax.set_ylim(-0.5, self.W - 0.5)

        plt.tight_layout()
        return fig


def create_simple_continuous_maze(size: int = 20) -> ContinuousActionMazeConfig:
    """Create simple continuous action maze for testing."""
    maze = np.ones((size, size), dtype=np.int32)
    maze[1:-1, 1:-1] = 0  # Open interior

    config = ContinuousActionMazeConfig(
        maze_array=maze,
        num_agents=10,
        max_steps=200,
        reward_type=RewardType.MFG,
        dt=0.1,
        velocity_max=2.0,
        start_positions=[(2, 2)],
        goal_positions=[(size - 3, size - 3)],
        goal_reward=10.0,
        congestion_weight=1.0,
        control_cost_weight=0.1,
    )

    return config

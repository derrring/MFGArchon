"""
Comparison of Continuous Control Algorithms for MFG.

Compares three state-of-the-art continuous control algorithms:
- DDPG: Deterministic policy with single critic
- TD3: Deterministic policy with twin critics and delayed updates
- SAC: Stochastic policy with entropy regularization

Problem: Continuous LQ-MFG with crowd aversion
- State: Position in [0,1]
- Action: Continuous velocity in [-1,1]
- Mean field coupling: Quadratic cost for proximity to crowd

Results show:
- DDPG: Fastest training but potential overestimation
- TD3: More stable than DDPG, reduced variance
- SAC: Best exploration and robustness via entropy

Author: MFGarchon Team
Date: October 2025
"""

from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Add package to path
package_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_root))

from mfgarchon.alg.reinforcement.algorithms.mean_field_ddpg import MeanFieldDDPG  # noqa: E402
from mfgarchon.alg.reinforcement.algorithms.mean_field_sac import MeanFieldSAC  # noqa: E402
from mfgarchon.alg.reinforcement.algorithms.mean_field_td3 import MeanFieldTD3  # noqa: E402

# Check for PyTorch
TORCH_AVAILABLE = find_spec("torch") is not None
if not TORCH_AVAILABLE:
    print("PyTorch not available. Install with: pip install torch")
    sys.exit(1)

# Check for Gymnasium
GYMNASIUM_AVAILABLE = find_spec("gymnasium") is not None
if not GYMNASIUM_AVAILABLE:
    print("Gymnasium not available. Install with: pip install gymnasium")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ContinuousLQMFGEnv:
    """
    Continuous Linear-Quadratic Mean Field Game environment.

    State: x ∈ [0,1] (position)
    Action: a ∈ [-1,1] (velocity)
    Dynamics: x_{t+1} = x_t + a_t·dt + σ·dW
    Reward: r(x,a,m) = -c_state·(x-x_goal)² - c_action·a² - c_crowd·∫(x-y)²m(y)dy

    This is a canonical test problem for continuous MFG control.
    """

    def __init__(
        self,
        num_agents: int = 100,
        dt: float = 0.1,
        max_steps: int = 50,
        state_noise: float = 0.05,
        x_goal: float = 0.7,
        c_state: float = 1.0,
        c_action: float = 0.1,
        c_crowd: float = 0.5,
    ):
        """
        Initialize continuous LQ-MFG environment.

        Args:
            num_agents: Population size
            dt: Time step
            max_steps: Episode length
            state_noise: Diffusion coefficient
            x_goal: Target position
            c_state: State tracking cost
            c_action: Action cost
            c_crowd: Crowd aversion cost
        """
        self.num_agents = num_agents
        self.dt = dt
        self.max_steps = max_steps
        self.state_noise = state_noise
        self.x_goal = x_goal
        self.c_state = c_state
        self.c_action = c_action
        self.c_crowd = c_crowd

        # State space
        self.state_dim = 1
        self.action_dim = 1
        self.action_bounds = (-1.0, 1.0)

        # Population state (histogram bins)
        self.num_bins = 20
        self.bin_edges = np.linspace(0, 1, self.num_bins + 1)
        self.population_dim = self.num_bins

        # Episode state
        self.current_step = 0
        self.agent_positions = None
        self.population_density = None

    def reset(self) -> tuple[NDArray, dict[str, Any]]:
        """Reset environment."""
        self.current_step = 0

        # Random initial positions
        self.agent_positions = np.random.uniform(0.2, 0.4, self.num_agents)

        # Compute population density
        self._update_population_density()

        # Return single agent state
        agent_idx = 0
        state = np.array([self.agent_positions[agent_idx]], dtype=np.float32)

        return state, {}

    def step(self, action: NDArray | float) -> tuple[NDArray, float, bool, bool, dict]:
        """
        Execute action for single agent.

        Args:
            action: Continuous action in [-1, 1]

        Returns:
            next_state, reward, terminated, truncated, info
        """
        if isinstance(action, np.ndarray):
            action_val = float(action[0])
        else:
            action_val = float(action)

        # Clip action
        action_val = np.clip(action_val, self.action_bounds[0], self.action_bounds[1])

        # Update agent position
        agent_idx = 0  # Control first agent
        x = self.agent_positions[agent_idx]

        # Dynamics: x_{t+1} = x_t + a·dt + σ·dW
        noise = np.random.normal(0, self.state_noise * np.sqrt(self.dt))
        x_next = x + action_val * self.dt + noise
        x_next = np.clip(x_next, 0.0, 1.0)  # Stay in bounds

        self.agent_positions[agent_idx] = x_next

        # Update population (simple: other agents follow mean)
        for i in range(1, self.num_agents):
            mean_pos = self.agent_positions.mean()
            noise = np.random.normal(0, self.state_noise * np.sqrt(self.dt))
            self.agent_positions[i] += (mean_pos - self.agent_positions[i]) * 0.1 + noise
            self.agent_positions[i] = np.clip(self.agent_positions[i], 0.0, 1.0)

        self._update_population_density()

        # Compute reward
        state_cost = self.c_state * (x_next - self.x_goal) ** 2
        action_cost = self.c_action * action_val**2

        # Crowd cost: distance to population
        crowd_cost = 0.0
        for other_x in self.agent_positions[1:]:
            crowd_cost += (x_next - other_x) ** 2
        crowd_cost *= self.c_crowd / (self.num_agents - 1)

        reward = -(state_cost + action_cost + crowd_cost)

        # Check termination
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps

        next_state = np.array([x_next], dtype=np.float32)

        return next_state, reward, terminated, truncated, {}

    def _update_population_density(self):
        """Update population density histogram."""
        counts, _ = np.histogram(self.agent_positions, bins=self.bin_edges)
        self.population_density = counts / self.num_agents  # Normalize

    def get_population_state(self) -> np.ndarray:
        """Population density as a flat NDArray (#1570 canonical env->algo contract)."""
        return self.population_density.astype(np.float32)


def train_algorithm(algo_class, algo_name: str, env, num_episodes: int, config: dict | None = None) -> dict:
    """
    Train an algorithm and return statistics.

    Args:
        algo_class: Algorithm class (MeanFieldDDPG, MeanFieldTD3, MeanFieldSAC)
        algo_name: Name for logging
        env: Environment
        num_episodes: Training episodes
        config: Algorithm configuration

    Returns:
        Training statistics
    """
    print(f"\n{'=' * 60}")
    print(f"Training {algo_name}")
    print(f"{'=' * 60}")

    algo = algo_class(
        env=env,
        state_dim=1,
        action_dim=1,
        population_dim=env.population_dim,
        action_bounds=env.action_bounds,
        config=config,
    )

    stats = algo.train(num_episodes=num_episodes)

    return stats


def plot_comparison(ddpg_stats: dict, td3_stats: dict, sac_stats: dict, save_path: str | None = None):
    """
    Plot comparison of three algorithms.

    Args:
        ddpg_stats: DDPG training statistics
        td3_stats: TD3 training statistics
        sac_stats: SAC training statistics
        save_path: Optional path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping plots")
        return

    _fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Smooth rewards
    def smooth(data, window=20):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window) / window, mode="valid")

    # 1. Episode Rewards
    ax = axes[0, 0]
    ax.plot(smooth(ddpg_stats["episode_rewards"]), label="DDPG", alpha=0.8)
    ax.plot(smooth(td3_stats["episode_rewards"]), label="TD3", alpha=0.8)
    ax.plot(smooth(sac_stats["episode_rewards"]), label="SAC", alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Rewards (smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Episode Lengths
    ax = axes[0, 1]
    ax.plot(smooth(ddpg_stats["episode_lengths"]), label="DDPG", alpha=0.8)
    ax.plot(smooth(td3_stats["episode_lengths"]), label="TD3", alpha=0.8)
    ax.plot(smooth(sac_stats["episode_lengths"]), label="SAC", alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Episode Lengths (smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Critic Losses
    ax = axes[1, 0]
    if ddpg_stats["critic_losses"]:
        ax.plot(smooth(ddpg_stats["critic_losses"]), label="DDPG Q-loss", alpha=0.8)
    if td3_stats["critic1_losses"]:
        ax.plot(smooth(td3_stats["critic1_losses"]), label="TD3 Q1-loss", alpha=0.8)
    if sac_stats["critic1_losses"]:
        ax.plot(smooth(sac_stats["critic1_losses"]), label="SAC Q1-loss", alpha=0.8)
    ax.set_xlabel("Update Step")
    ax.set_ylabel("Loss")
    ax.set_title("Critic Losses (smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. SAC-specific: Entropy and Temperature
    ax = axes[1, 1]
    if sac_stats["entropy_values"]:
        ax_entropy = ax
        ax_entropy.plot(smooth(sac_stats["entropy_values"]), label="Entropy", color="blue", alpha=0.8)
        ax_entropy.set_xlabel("Update Step")
        ax_entropy.set_ylabel("Entropy", color="blue")
        ax_entropy.tick_params(axis="y", labelcolor="blue")
        ax_entropy.grid(True, alpha=0.3)

        # Twin axis for temperature
        ax_alpha = ax_entropy.twinx()
        ax_alpha.plot(smooth(sac_stats["alpha_values"]), label="Temperature (α)", color="red", alpha=0.8)
        ax_alpha.set_ylabel("Temperature α", color="red")
        ax_alpha.tick_params(axis="y", labelcolor="red")

        ax_entropy.set_title("SAC: Entropy and Temperature")
        ax_entropy.legend(loc="upper left")
        ax_alpha.legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {save_path}")

    plt.show()


def print_summary(ddpg_stats: dict, td3_stats: dict, sac_stats: dict):
    """Print summary statistics."""
    print(f"\n{'=' * 60}")
    print("FINAL PERFORMANCE COMPARISON")
    print(f"{'=' * 60}")

    def get_final_performance(stats):
        rewards = stats["episode_rewards"][-50:]  # Last 50 episodes
        return np.mean(rewards), np.std(rewards)

    ddpg_mean, ddpg_std = get_final_performance(ddpg_stats)
    td3_mean, td3_std = get_final_performance(td3_stats)
    sac_mean, sac_std = get_final_performance(sac_stats)

    print("\nReward (last 50 episodes):")
    print(f"  DDPG: {ddpg_mean:7.2f} ± {ddpg_std:5.2f}")
    print(f"  TD3:  {td3_mean:7.2f} ± {td3_std:5.2f}")
    print(f"  SAC:  {sac_mean:7.2f} ± {sac_std:5.2f}")

    # Determine winner
    best_mean = max(ddpg_mean, td3_mean, sac_mean)
    if sac_mean == best_mean:
        winner = "SAC"
    elif td3_mean == best_mean:
        winner = "TD3"
    else:
        winner = "DDPG"

    print(f"\nBest Algorithm: {winner}")

    # Key differences
    print(f"\n{'=' * 60}")
    print("KEY ALGORITHM PROPERTIES")
    print(f"{'=' * 60}")
    print("\nDDPG (Deep Deterministic Policy Gradient):")
    print("  - Deterministic policy: a = μ(s,m)")
    print("  - Single critic Q(s,a,m)")
    print("  - OU noise for exploration")
    print("  - Fastest training but potential overestimation")

    print("\nTD3 (Twin Delayed DDPG):")
    print("  - Deterministic policy: a = μ(s,m)")
    print("  - Twin critics: min(Q1, Q2) reduces overestimation")
    print("  - Delayed policy updates for stability")
    print("  - Target policy smoothing for robustness")

    print("\nSAC (Soft Actor-Critic):")
    print("  - Stochastic policy: a ~ π(·|s,m)")
    print("  - Maximum entropy objective: J = E[r + α·H(π)]")
    print("  - Automatic temperature tuning")
    print("  - Best exploration and robustness")

    print(f"\n{'=' * 60}\n")


def main():
    """Run comparison experiment."""
    # Create environment
    env = ContinuousLQMFGEnv(
        num_agents=100,
        dt=0.1,
        max_steps=50,
        state_noise=0.05,
        x_goal=0.7,
        c_state=1.0,
        c_action=0.1,
        c_crowd=0.5,
    )

    # Training configuration
    num_episodes = 500

    # Shared config for fair comparison
    base_config = {
        "actor_lr": 1e-4,
        "critic_lr": 1e-3,
        "discount_factor": 0.99,
        "tau": 0.005,
        "batch_size": 128,
        "replay_buffer_size": 50000,
        "hidden_dims": [128, 64],
    }

    # Train DDPG
    ddpg_config = {**base_config, "exploration_noise_std": 0.2}
    ddpg_stats = train_algorithm(MeanFieldDDPG, "DDPG", env, num_episodes, ddpg_config)

    # Reset environment
    env.reset()

    # Train TD3
    td3_config = {
        **base_config,
        "policy_delay": 2,
        "target_noise_std": 0.2,
        "target_noise_clip": 0.5,
        "exploration_noise_std": 0.2,
    }
    td3_stats = train_algorithm(MeanFieldTD3, "TD3", env, num_episodes, td3_config)

    # Reset environment
    env.reset()

    # Train SAC
    sac_config = {
        **base_config,
        "auto_tune_temperature": True,
        "initial_temperature": 0.2,
        "alpha_lr": 1e-4,
    }
    sac_stats = train_algorithm(MeanFieldSAC, "SAC", env, num_episodes, sac_config)

    # Print summary
    print_summary(ddpg_stats, td3_stats, sac_stats)

    # Plot results
    save_path = package_root / "examples" / "advanced" / "continuous_control_comparison.png"
    plot_comparison(ddpg_stats, td3_stats, sac_stats, save_path=str(save_path))


if __name__ == "__main__":
    main()

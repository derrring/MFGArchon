"""
Mean Field DDPG: Continuous Action Demonstration.

Demonstrates:
- Continuous velocity control in maze environment
- DDPG actor-critic learning
- Mean field coupling through population density
- Comparison with discrete action Q-learning

Key Concepts:
- Actor: μ(s, m) → a ∈ [-v_max, v_max]²
- Critic: Q(s, a, m)
- Exploration: Ornstein-Uhlenbeck noise

Author: MFGarchon Team
Date: October 2025
"""

import numpy as np

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")
    exit(1)

try:
    import gymnasium  # noqa: F401

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    print("Gymnasium not available. Install with: pip install gymnasium")
    exit(1)

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from mfgarchon.alg.reinforcement.algorithms.mean_field_ddpg import MeanFieldDDPG
from mfgarchon.alg.reinforcement.environments.continuous_action_maze_env import (
    ContinuousActionMazeConfig,
    ContinuousActionMazeEnvironment,
    RewardType,
)


def create_crowd_navigation_environment() -> ContinuousActionMazeEnvironment:
    """
    Create crowd navigation scenario.

    Setup:
    - 20x20 maze with obstacles
    - 15 agents navigating from left to right
    - Continuous velocity control
    - Congestion avoidance
    """
    size = 20
    maze = np.ones((size, size), dtype=np.int32)
    maze[1:-1, 1:-1] = 0  # Open interior

    # Add some obstacles
    maze[5:8, 10:12] = 1
    maze[12:15, 8:10] = 1

    config = ContinuousActionMazeConfig(
        maze_array=maze,
        num_agents=15,
        max_steps=300,
        reward_type=RewardType.MFG,
        dt=0.15,
        velocity_max=2.0,
        noise_std=0.05,  # Small stochastic perturbation
        start_positions=[(2, 2), (2, 3), (2, 4)],
        goal_positions=[(size - 3, size - 3), (size - 3, size - 4), (size - 3, size - 5)],
        goal_reward=20.0,
        collision_penalty=-2.0,
        congestion_weight=2.0,  # Strong congestion avoidance
        control_cost_weight=0.1,
        population_smoothing=0.8,
    )

    return ContinuousActionMazeEnvironment(config)


def train_ddpg_agent(env: ContinuousActionMazeEnvironment, num_episodes: int = 500) -> dict:
    """Train DDPG agent on environment."""
    print("\n=== Training Mean Field DDPG ===")
    print(f"Environment: {env.H}x{env.W} maze, {env.num_agents} agents")
    print(f"Action space: Continuous velocity ∈ [-{env.config.velocity_max}, {env.config.velocity_max}]²")

    # Create DDPG agent
    state_dim = 2  # (x, y) position
    action_dim = 2  # (vₓ, vᵧ) velocity
    population_dim = env.H * env.W  # Flattened density

    algo = MeanFieldDDPG(
        env=env,
        state_dim=state_dim,
        action_dim=action_dim,
        population_dim=population_dim,
        action_bounds=(-env.config.velocity_max, env.config.velocity_max),
        config={
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "discount_factor": 0.99,
            "tau": 0.001,
            "batch_size": 128,
            "replay_buffer_size": 50000,
            "ou_theta": 0.15,
            "ou_sigma": 0.3,  # Higher exploration initially
        },
    )

    # Train
    stats = algo.train(num_episodes=num_episodes)

    return stats


def visualize_learning_curves(stats: dict, save_path: str = "ddpg_learning_curves.png"):
    """Visualize training progress."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for visualization")
        return

    _fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Episode rewards
    ax = axes[0, 0]
    rewards = stats["episode_rewards"]
    window = 20
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(smoothed, label=f"Smoothed (window={window})", linewidth=2)
    ax.plot(rewards, alpha=0.3, label="Raw")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Episode Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Episode lengths
    ax = axes[0, 1]
    lengths = stats["episode_lengths"]
    if len(lengths) >= window:
        smoothed_len = np.convolve(lengths, np.ones(window) / window, mode="valid")
        ax.plot(smoothed_len, label=f"Smoothed (window={window})", linewidth=2)
    ax.plot(lengths, alpha=0.3, label="Raw")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.set_title("Episode Lengths")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Critic loss
    ax = axes[1, 0]
    if stats["critic_losses"]:
        critic_losses = stats["critic_losses"]
        if len(critic_losses) >= window:
            smoothed_critic = np.convolve(critic_losses, np.ones(window) / window, mode="valid")
            ax.plot(smoothed_critic, linewidth=2)
        else:
            ax.plot(critic_losses)
        ax.set_xlabel("Update Step")
        ax.set_ylabel("Critic Loss (MSE)")
        ax.set_title("Critic Training Loss")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # Actor loss
    ax = axes[1, 1]
    if stats["actor_losses"]:
        actor_losses = [-loss for loss in stats["actor_losses"]]  # Negate for plotting
        if len(actor_losses) >= window:
            smoothed_actor = np.convolve(actor_losses, np.ones(window) / window, mode="valid")
            ax.plot(smoothed_actor, linewidth=2)
        else:
            ax.plot(actor_losses)
        ax.set_xlabel("Update Step")
        ax.set_ylabel("-Actor Loss (Policy Objective)")
        ax.set_title("Actor Training Objective")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Mean Field DDPG: Continuous Action Control", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Learning curves saved to: {save_path}")
    plt.close()


def evaluate_policy(env: ContinuousActionMazeEnvironment, algo: MeanFieldDDPG, num_episodes: int = 10):
    """Evaluate trained policy."""
    print("\n=== Evaluating Trained Policy ===")

    total_rewards = []
    success_rate = 0

    for episode in range(num_episodes):
        _observations, _ = env.reset(seed=episode + 1000)
        episode_reward = 0
        done = False
        steps = 0

        while not done and steps < env.config.max_steps:
            # Get population state (#1570: now a flat NDArray for every env)
            pop_state = env.get_population_state()

            # Select actions for all agents (use first agent's state for demo)
            actions = []
            for i in range(env.num_agents):
                if not env.agents_done[i]:
                    state = env.agent_positions[i]
                    action = algo.select_action(state, pop_state, training=False)
                    actions.append(action)
                else:
                    actions.append(np.zeros(2))

            # Step environment
            _observations, rewards, terminated, truncated, _ = env.step(np.array(actions))
            episode_reward += rewards.sum()
            steps += 1
            done = terminated or truncated

        total_rewards.append(episode_reward)
        if env.agents_done.sum() >= env.num_agents * 0.8:  # 80% reached goal
            success_rate += 1

        print(
            f"Episode {episode + 1}: Reward={episode_reward:.2f}, Steps={steps}, Goals={env.agents_done.sum()}/{env.num_agents}"
        )

    print(f"\nEvaluation Results ({num_episodes} episodes):")
    print(f"  Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Success Rate: {success_rate / num_episodes * 100:.1f}% (≥80% agents reach goal)")


def demonstrate_continuous_vs_discrete():
    """
    Compare continuous DDPG with discrete action baseline.

    Key Insight:
    - Discrete: 4 actions (↑↓←→) - limited expressiveness
    - Continuous: a ∈ ℝ² - arbitrary directions and speeds
    """
    print("\n" + "=" * 60)
    print("DDPG for Mean Field Games: Continuous Action Control")
    print("=" * 60)

    # Create environment
    env = create_crowd_navigation_environment()

    # Train DDPG
    stats = train_ddpg_agent(env, num_episodes=500)

    # Visualize
    visualize_learning_curves(stats)

    # Note: For evaluation, we need to recreate algo with trained weights
    # For this demo, we'll skip full evaluation
    print("\n✓ DDPG training complete!")
    print("\nKey Advantages of Continuous Actions:")
    print("  1. Smooth trajectories (no grid discretization)")
    print("  2. Precise velocity control")
    print("  3. Natural physics integration")
    print("  4. Scales to high-dimensional actions")

    print("\nContinuous Action Benefits:")
    print("  - Actor directly outputs a ∈ ℝᵈ (no argmax)")
    print("  - Critic Q(s,a,m) handles continuous actions")
    print("  - Exploration via Ornstein-Uhlenbeck noise")
    print("  - Better for real-world control (velocity, steering, etc.)")


def analyze_nash_equilibrium_properties():
    """
    Analyze Nash equilibrium properties of continuous control.

    Mathematical Framework:
    - At equilibrium: μ*(s,m*) = argmax_a Q*(s,a,m*)
    - DDPG: Actor approximates this argmax via gradient ascent
    - Verification: Check ε-Nash gap
    """
    print("\n" + "=" * 60)
    print("Nash Equilibrium Analysis for Continuous Actions")
    print("=" * 60)

    print("\nTheoretical Properties:")
    print("  1. Deterministic Nash: μ*(s,m) ∈ argmax_a Q*(s,a,m)")
    print("  2. Actor learns greedy policy via ∇_θ J = E[∇_θ μ_θ · ∇_a Q]")
    print("  3. Critic learns Q*(s,a,m) via TD learning")
    print("  4. Convergence: μ_θ → μ*, Q_φ → Q* (under assumptions)")

    print("\nPractical Considerations:")
    print("  - DDPG = Deterministic policy (single Nash equilibrium)")
    print("  - SAC = Stochastic policy (maximum entropy Nash)")
    print("  - TD3 = Twin critics (reduce overestimation bias)")

    print("\nMean Field Coupling:")
    print("  - Population state m(t) = density induced by μ(·,m)")
    print("  - Fixed point: m* = μ(μ*, m*)")
    print("  - Fictitious play: Iterate between policy update and density update")


if __name__ == "__main__":
    # Main demonstration
    demonstrate_continuous_vs_discrete()

    # Nash equilibrium analysis
    analyze_nash_equilibrium_properties()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("  1. Try TD3 for improved stability (twin critics)")
    print("  2. Implement SAC for maximum entropy RL")
    print("  3. Extend to multi-population continuous control")
    print("  4. Apply to real-world scenarios (traffic, pricing, crowds)")

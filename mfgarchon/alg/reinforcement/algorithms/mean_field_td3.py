"""
Twin Delayed DDPG (TD3) for Mean Field Games.

Improvements over DDPG:
1. Twin critics: Q₁(s,a,m), Q₂(s,a,m) - reduces overestimation bias
2. Delayed policy updates: Update actor every d steps - improves stability
3. Target policy smoothing: Add noise to target actions - better generalization

Mathematical Framework:
- Actor: μ_θ(s, m) → a ∈ ℝᵈ (deterministic)
- Twin Critics: Q_φ₁(s,a,m), Q_φ₂(s,a,m)
- Target: y = r + γ·min(Q'₁(s',ã',m'), Q'₂(s',ã',m'))
  where ã' = μ'(s',m') + clip(ε, -c, c)

Author: MFGarchon Team
Date: October 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None

# Import actor and replay buffer from DDPG
from mfgarchon.alg.reinforcement.algorithms.mean_field_ddpg import DDPGActor, ReplayBuffer


class TD3Critic(nn.Module):
    """
    Q-function network for TD3.

    Identical architecture to DDPG critic, but we use two independent instances.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        population_dim: int,
        hidden_dims: list[int] | None = None,
    ):
        """
        Initialize TD3 critic network.

        Args:
            state_dim: Dimension of individual state
            action_dim: Dimension of continuous action
            population_dim: Dimension of population state
            hidden_dims: Hidden layer dimensions
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        # State encoder
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden_dims[0]), nn.ReLU())

        # Action encoder
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, 64), nn.ReLU())

        # Population encoder
        self.pop_encoder = nn.Sequential(nn.Linear(population_dim, hidden_dims[1]), nn.ReLU())

        # Q-value head
        combined_dim = hidden_dims[0] + 64 + hidden_dims[1]
        self.q_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor, population_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: Individual state [batch, state_dim]
            action: Continuous action [batch, action_dim]
            population_state: Population state [batch, pop_dim]

        Returns:
            Q-value [batch]
        """
        state_feat = self.state_encoder(state)
        action_feat = self.action_encoder(action)
        pop_feat = self.pop_encoder(population_state)
        combined = torch.cat([state_feat, action_feat, pop_feat], dim=1)
        return self.q_head(combined).squeeze(-1)


class MeanFieldTD3:
    """
    Twin Delayed DDPG for Mean Field Games.

    Improvements over DDPG:
    - Twin critics: min(Q₁, Q₂) reduces overestimation
    - Delayed updates: Actor updated every d steps
    - Target smoothing: Noise added to target actions
    """

    def __init__(
        self,
        env: Any,
        state_dim: int,
        action_dim: int,
        population_dim: int,
        action_bounds: tuple[float, float],
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize Mean Field TD3.

        Args:
            env: MFG environment
            state_dim: State dimension
            action_dim: Continuous action dimension
            population_dim: Population state dimension
            action_bounds: (min, max) action bounds
            config: Algorithm configuration
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.population_dim = population_dim
        self.action_bounds = action_bounds

        # Default config (TD3-specific parameters)
        default_config = {
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "discount_factor": 0.99,
            "tau": 0.001,  # Soft update
            "batch_size": 256,
            "replay_buffer_size": 100000,
            "hidden_dims": [256, 128],
            "policy_delay": 2,  # TD3: Delayed policy updates
            "target_noise_std": 0.2,  # TD3: Target policy smoothing noise
            "target_noise_clip": 0.5,  # TD3: Clip range for target noise
            "exploration_noise_std": 0.1,  # Exploration noise during training
        }
        self.config = {**default_config, **(config or {})}

        # Actor network (shared with DDPG)
        self.actor = DDPGActor(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
            action_bounds=action_bounds,
            hidden_dims=self.config["hidden_dims"],
        )

        self.actor_target = DDPGActor(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
            action_bounds=action_bounds,
            hidden_dims=self.config["hidden_dims"],
        )
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Twin critic networks
        self.critic1 = TD3Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
            hidden_dims=self.config["hidden_dims"],
        )

        self.critic2 = TD3Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
            hidden_dims=self.config["hidden_dims"],
        )

        self.critic1_target = TD3Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
            hidden_dims=self.config["hidden_dims"],
        )

        self.critic2_target = TD3Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
            hidden_dims=self.config["hidden_dims"],
        )

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config["actor_lr"])
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.config["critic_lr"])
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.config["critic_lr"])

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.config["replay_buffer_size"],
            state_dim=state_dim,
            action_dim=action_dim,
            pop_dim=population_dim,
        )

        # Training stats
        self.update_count = 0

    def select_action(self, state: NDArray, population_state: NDArray, training: bool = True) -> NDArray:
        """
        Select action using actor with optional exploration noise.

        Args:
            state: Individual state [state_dim]
            population_state: Population state [pop_dim]
            training: If True, add Gaussian exploration noise

        Returns:
            Continuous action [action_dim]
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            pop_state_t = torch.FloatTensor(population_state).unsqueeze(0)
            action = self.actor(state_t, pop_state_t).squeeze(0).numpy()

        if training:
            # Gaussian noise for exploration (simpler than OU for TD3)
            noise = np.random.normal(0, self.config["exploration_noise_std"], size=action.shape)
            action = np.clip(action + noise, self.action_bounds[0], self.action_bounds[1])

        return action

    def update(self) -> tuple[float, float, float] | None:
        """
        Update twin critics and (delayed) actor.

        Returns:
            (critic1_loss, critic2_loss, actor_loss) or None if buffer too small
        """
        if len(self.replay_buffer) < self.config["batch_size"]:
            return None

        # Sample batch
        batch = self.replay_buffer.sample(self.config["batch_size"])

        states = torch.FloatTensor(batch["states"])
        actions = torch.FloatTensor(batch["actions"])
        rewards = torch.FloatTensor(batch["rewards"])
        next_states = torch.FloatTensor(batch["next_states"])
        population_states = torch.FloatTensor(batch["population_states"])
        next_population_states = torch.FloatTensor(batch["next_population_states"])
        dones = torch.BoolTensor(batch["dones"])

        # Compute TD3 target with target policy smoothing
        with torch.no_grad():
            # Target actions with smoothing noise
            next_actions = self.actor_target(next_states, next_population_states)

            # Add clipped noise
            noise = torch.randn_like(next_actions) * self.config["target_noise_std"]
            noise = torch.clamp(noise, -self.config["target_noise_clip"], self.config["target_noise_clip"])

            next_actions_noisy = torch.clamp(next_actions + noise, self.action_bounds[0], self.action_bounds[1])

            # Clipped double Q-learning: min(Q1, Q2)
            target_q1 = self.critic1_target(next_states, next_actions_noisy, next_population_states)
            target_q2 = self.critic2_target(next_states, next_actions_noisy, next_population_states)
            target_q = torch.min(target_q1, target_q2)

            # TD target
            target = rewards + self.config["discount_factor"] * target_q * (~dones)

        # Update critic 1
        current_q1 = self.critic1(states, actions, population_states)
        critic1_loss = nn.functional.mse_loss(current_q1, target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # Update critic 2
        current_q2 = self.critic2(states, actions, population_states)
        critic2_loss = nn.functional.mse_loss(current_q2, target)

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed policy update
        actor_loss = torch.tensor(0.0)
        if self.update_count % self.config["policy_delay"] == 0:
            # Actor loss (use critic1 only, as per TD3 paper)
            actor_actions = self.actor(states, population_states)
            actor_loss = -self.critic1(states, actor_actions, population_states).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

        self.update_count += 1

        return critic1_loss.item(), critic2_loss.item(), actor_loss.item()

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Soft update target network parameters."""
        tau = self.config["tau"]
        for target_param, param in zip(target.parameters(), source.parameters(), strict=False):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def train(self, num_episodes: int = 1000) -> dict[str, Any]:
        """
        Train TD3 agent.

        Args:
            num_episodes: Number of training episodes

        Returns:
            Training statistics
        """
        episode_rewards = []
        episode_lengths = []
        critic1_losses = []
        critic2_losses = []
        actor_losses = []

        for episode in range(num_episodes):
            observations, _ = self.env.reset()
            state = observations[0] if isinstance(observations, list | tuple) else observations
            episode_reward = 0
            episode_length = 0

            done = False
            while not done:
                # Get population state. Issue #1508: get_population_state is a REQUIRED capability of a
                # mean-field env, NOT optional backend detection. The former hasattr -> np.zeros fallback
                # silently trained on an identically-zero mean field (the coupling channel that DEFINES a
                # mean-field game), returning a policy the user trusts while MFG was never active. Fail
                # loud (getattr + callable, per the repo's no-hasattr-duck-typing rule).
                _pop_getter = getattr(self.env, "get_population_state", None)
                if not callable(_pop_getter):
                    raise AttributeError(
                        f"{type(self.env).__name__} has no get_population_state(): a mean-field RL env "
                        f"must expose the population density (the MFG coupling channel). A zero fallback "
                        f"would train on an identically-zero mean field and silently return a non-MFG "
                        f"policy (Issue #1508)."
                    )
                pop_state = _pop_getter().density_histogram.flatten()

                # Select action
                action = self.select_action(state, pop_state, training=True)

                # Execute action
                next_observations, reward, terminated, truncated, _ = self.env.step(action)
                next_state = next_observations[0] if isinstance(next_observations, list | tuple) else next_observations

                # Get next population state (required capability; see above, Issue #1508).
                _pop_getter = getattr(self.env, "get_population_state", None)
                if not callable(_pop_getter):
                    raise AttributeError(f"{type(self.env).__name__} has no get_population_state() (Issue #1508).")
                next_pop_state = _pop_getter().density_histogram.flatten()

                # Store transition
                # Handle both scalar and array rewards
                if isinstance(reward, int | float | np.floating):
                    reward_scalar = float(reward)
                elif isinstance(reward, np.ndarray):
                    reward_scalar = float(reward.item() if reward.size == 1 else reward[0])
                else:
                    reward_scalar = float(reward)

                self.replay_buffer.push(
                    state=state,
                    action=action,
                    reward=reward_scalar,
                    next_state=next_state,
                    population_state=pop_state,
                    next_population_state=next_pop_state,
                    done=terminated or truncated,
                )

                # Update
                losses = self.update()
                if losses is not None:
                    critic1_losses.append(losses[0])
                    critic2_losses.append(losses[1])
                    if losses[2] > 0:  # Actor was updated
                        actor_losses.append(losses[2])

                state = next_state
                episode_reward += reward_scalar
                episode_length += 1

                done = terminated or truncated

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if (episode + 1) % 100 == 0:
                print(
                    f"Episode {episode + 1}/{num_episodes}: "
                    f"Reward={np.mean(episode_rewards[-100:]):.2f}, "
                    f"Length={np.mean(episode_lengths[-100:]):.1f}, "
                    f"Q1 Loss={np.mean(critic1_losses[-100:]) if critic1_losses else 0:.4f}, "
                    f"Q2 Loss={np.mean(critic2_losses[-100:]) if critic2_losses else 0:.4f}"
                )

        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "critic1_losses": critic1_losses,
            "critic2_losses": critic2_losses,
            "actor_losses": actor_losses,
        }

"""
Deep Deterministic Policy Gradient for Mean Field Games.

Implements continuous action control for MFG using:
- Actor: μ(s, m) → a ∈ ℝᵈ (deterministic continuous policy)
- Critic: Q(s, a, m) with action as input
- Mean field consistency: m_t = μ_t(π)
- Ornstein-Uhlenbeck exploration noise

Mathematical Framework:
- Policy: μ_θ: S × P(S) → A ⊂ ℝᵈ
- Q-function: Q_φ(s, a, m) = E[Σ γᵗ r(sₜ, aₜ, mₜ) | s₀=s, a₀=a, m₀=m]
- Policy gradient: ∇_θ J = E[∇_θ μ_θ(s,m) ∇_a Q_φ(s,a,m)|_{a=μ_θ(s,m)}]

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


class DDPGActor(nn.Module):
    """
    Deterministic policy network for continuous actions.

    Architecture:
    - State encoder: s → features
    - Population encoder: m → features
    - Action head: features → a ∈ ℝᵈ (bounded by tanh)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        population_dim: int,
        action_bounds: tuple[float, float],
        hidden_dims: list[int] | None = None,
    ):
        """
        Initialize DDPG actor network.

        Args:
            state_dim: Dimension of individual state
            action_dim: Dimension of continuous action
            population_dim: Dimension of population state
            action_bounds: (min, max) action bounds
            hidden_dims: Hidden layer dimensions
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.population_dim = population_dim

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]), nn.ReLU(), nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU()
        )

        # Population encoder
        self.pop_encoder = nn.Sequential(
            nn.Linear(population_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[1] // 2),
            nn.ReLU(),
        )

        # Action head
        combined_dim = hidden_dims[1] + hidden_dims[1] // 2
        self.action_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dims[1]), nn.ReLU(), nn.Linear(hidden_dims[1], action_dim), nn.Tanh()
        )

        # Action scaling
        self.action_scale = (action_bounds[1] - action_bounds[0]) / 2.0
        self.action_bias = (action_bounds[1] + action_bounds[0]) / 2.0

    def forward(self, state: torch.Tensor, population_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: Individual state [batch, state_dim]
            population_state: Population state [batch, pop_dim]

        Returns:
            Continuous action [batch, action_dim]
        """
        state_feat = self.state_encoder(state)
        pop_feat = self.pop_encoder(population_state)
        combined = torch.cat([state_feat, pop_feat], dim=1)
        action = self.action_head(combined)
        return action * self.action_scale + self.action_bias


class DDPGCritic(nn.Module):
    """
    Q-function network with action as input.

    Architecture:
    - State encoder: s → features
    - Action encoder: a → features
    - Population encoder: m → features
    - Q-head: combined → Q(s, a, m)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        population_dim: int,
        hidden_dims: list[int] | None = None,
    ):
        """
        Initialize DDPG critic network.

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


class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck process for exploration.

    Generates temporally correlated noise for continuous action exploration.
    """

    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        """
        Initialize OU noise process.

        Args:
            action_dim: Dimension of action space
            mu: Mean reversion level
            theta: Mean reversion speed
            sigma: Noise volatility
        """
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu

    def reset(self) -> None:
        """Reset noise to initial state."""
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self) -> NDArray:
        """Sample noise from OU process."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


class ReplayBuffer:
    """Experience replay buffer for DDPG."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int, pop_dim: int):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum buffer size
            state_dim: State dimension
            action_dim: Continuous action dimension
            pop_dim: Population state dimension
        """
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.population_states = np.zeros((capacity, pop_dim), dtype=np.float32)
        self.next_population_states = np.zeros((capacity, pop_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

        self.position = 0
        self.size = 0

    def push(
        self,
        state: NDArray,
        action: NDArray,
        reward: float,
        next_state: NDArray,
        population_state: NDArray,
        next_population_state: NDArray,
        done: bool,
    ) -> None:
        """Add transition to buffer."""
        idx = self.position
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.population_states[idx] = population_state
        self.next_population_states[idx] = next_population_state
        self.dones[idx] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, Any]:
        """Sample batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "population_states": self.population_states[indices],
            "next_population_states": self.next_population_states[indices],
            "dones": self.dones[indices],
        }

    def __len__(self) -> int:
        return self.size


class MeanFieldDDPG:
    """
    Deep Deterministic Policy Gradient for Mean Field Games.

    Implements:
    - Deterministic actor: μ_θ(s, m) → a ∈ ℝᵈ
    - Critic: Q_φ(s, a, m)
    - Target networks with soft updates
    - Ornstein-Uhlenbeck exploration
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
        Initialize Mean Field DDPG.

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

        # Default config
        default_config = {
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "discount_factor": 0.99,
            "tau": 0.001,  # Soft update
            "batch_size": 256,
            "replay_buffer_size": 100000,
            "hidden_dims": [256, 128],
            "ou_theta": 0.15,
            "ou_sigma": 0.2,
        }
        self.config = {**default_config, **(config or {})}

        # Actor networks
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

        # Critic networks
        self.critic = DDPGCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
            hidden_dims=self.config["hidden_dims"],
        )

        self.critic_target = DDPGCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
            hidden_dims=self.config["hidden_dims"],
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config["actor_lr"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config["critic_lr"])

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.config["replay_buffer_size"],
            state_dim=state_dim,
            action_dim=action_dim,
            pop_dim=population_dim,
        )

        # Exploration noise
        self.noise = OrnsteinUhlenbeckNoise(
            action_dim=action_dim, theta=self.config["ou_theta"], sigma=self.config["ou_sigma"]
        )

        # Training stats
        self.update_count = 0

    def select_action(self, state: NDArray, population_state: NDArray, training: bool = True) -> NDArray:
        """
        Select action using actor with optional exploration noise.

        Args:
            state: Individual state [state_dim]
            population_state: Population state [pop_dim]
            training: If True, add exploration noise

        Returns:
            Continuous action [action_dim]
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            pop_state_t = torch.FloatTensor(population_state).unsqueeze(0)
            action = self.actor(state_t, pop_state_t).squeeze(0).numpy()

        if training:
            noise = self.noise.sample()
            action = np.clip(action + noise, self.action_bounds[0], self.action_bounds[1])

        return action

    def update(self) -> tuple[float, float] | None:
        """
        Update actor and critic networks.

        Returns:
            (critic_loss, actor_loss) or None if buffer too small
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

        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states, next_population_states)
            target_q = self.critic_target(next_states, next_actions, next_population_states)
            target_q = rewards + self.config["discount_factor"] * target_q * (~dones)

        current_q = self.critic(states, actions, population_states)
        critic_loss = nn.functional.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_actions = self.actor(states, population_states)
        actor_loss = -self.critic(states, actor_actions, population_states).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        self.update_count += 1

        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Soft update target network parameters."""
        tau = self.config["tau"]
        for target_param, param in zip(target.parameters(), source.parameters(), strict=False):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def train(self, num_episodes: int = 1000) -> dict[str, Any]:
        """
        Train DDPG agent.

        Args:
            num_episodes: Number of training episodes

        Returns:
            Training statistics
        """
        episode_rewards = []
        episode_lengths = []
        critic_losses = []
        actor_losses = []

        for episode in range(num_episodes):
            observations, _ = self.env.reset()
            state = observations[0] if isinstance(observations, list | tuple) else observations
            episode_reward = 0
            episode_length = 0

            self.noise.reset()

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
                pop_state = _pop_getter()  # #1570: get_population_state() returns a flat NDArray for every env

                # Select action
                action = self.select_action(state, pop_state, training=True)

                # Execute action
                next_observations, reward, terminated, truncated, _ = self.env.step(action)
                next_state = next_observations[0] if isinstance(next_observations, list | tuple) else next_observations

                # Get next population state (required capability; see above, Issue #1508).
                _pop_getter = getattr(self.env, "get_population_state", None)
                if not callable(_pop_getter):
                    raise AttributeError(f"{type(self.env).__name__} has no get_population_state() (Issue #1508).")
                next_pop_state = _pop_getter()  # #1570: get_population_state() returns a flat NDArray for every env

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
                    critic_losses.append(losses[0])
                    actor_losses.append(losses[1])

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
                    f"Critic Loss={np.mean(critic_losses[-100:]) if critic_losses else 0:.4f}, "
                    f"Actor Loss={np.mean(actor_losses[-100:]) if actor_losses else 0:.4f}"
                )

        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "critic_losses": critic_losses,
            "actor_losses": actor_losses,
        }

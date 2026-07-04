"""
Soft Actor-Critic (SAC) for Mean Field Games.

Maximum entropy reinforcement learning with:
1. Stochastic policies: π(a|s,m) with entropy regularization
2. Twin soft Q-critics: Q₁(s,a,m), Q₂(s,a,m)
3. Automatic temperature tuning: α dynamically adjusted
4. Reparameterization trick: Enables gradient flow through stochastic sampling

Mathematical Framework:
- Objective: J(π) = E[Σ γᵗ(r(sₜ,aₜ,mₜ) + α𝓗(π(·|sₜ,mₜ)))]
- Soft Bellman: Q(s,a,m) = r + γ E[min(Q₁',Q₂')(s',a',m') - α log π(a'|s',m')]
- Policy: π(a|s,m) = tanh(𝒩(μ_θ(s,m), σ_θ(s,m)))

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
    import torch.nn.functional
    import torch.optim as optim
    from torch.distributions import Normal

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    Normal = None

# Import components from TD3
from mfgarchon.alg.reinforcement.algorithms.mean_field_ddpg import ReplayBuffer
from mfgarchon.alg.reinforcement.algorithms.mean_field_td3 import TD3Critic

# Constants for numerical stability
LOG_STD_MIN = -20
LOG_STD_MAX = 2


class SACStochasticActor(nn.Module):
    """
    Stochastic actor network for SAC.

    Outputs Gaussian policy: π(a|s,m) = 𝒩(μ_θ(s,m), σ_θ(s,m))
    with tanh squashing for bounded actions.
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
        Initialize SAC stochastic actor.

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
            hidden_dims = [256, 256]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.population_dim = population_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + population_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )

        # Mean head
        self.mean_head = nn.Linear(hidden_dims[1], action_dim)

        # Log std head
        self.log_std_head = nn.Linear(hidden_dims[1], action_dim)

        # Action scaling
        self.action_scale = (action_bounds[1] - action_bounds[0]) / 2.0
        self.action_bias = (action_bounds[1] + action_bounds[0]) / 2.0

    def forward(self, state: torch.Tensor, population_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute mean and log_std.

        Args:
            state: Individual state [batch, state_dim]
            population_state: Population state [batch, pop_dim]

        Returns:
            (mean, log_std) tuple
        """
        features = self.encoder(torch.cat([state, population_state], dim=1))
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample(
        self, state: torch.Tensor, population_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action with reparameterization trick.

        Args:
            state: Individual state [batch, state_dim]
            population_state: Population state [batch, pop_dim]

        Returns:
            (action, log_prob, mean) tuple
        """
        mean, log_std = self.forward(state, population_state)
        std = log_std.exp()

        # Reparameterization trick
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Sample with gradient

        # Squash to action bounds with tanh
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # Compute log probability with change of variables
        log_prob = normal.log_prob(x_t)
        # Correction for tanh squashing: log π(a) = log μ(ã) - Σ log(1 - tanh²(ã))
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        # Also return mean for evaluation
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean_action


class MeanFieldSAC:
    """
    Soft Actor-Critic for Mean Field Games.

    Features:
    - Stochastic policy with entropy regularization
    - Twin soft Q-critics (from TD3)
    - Automatic temperature tuning
    - Maximum entropy objective
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
        Initialize Mean Field SAC.

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

        # Default config (SAC-specific parameters)
        default_config = {
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "alpha_lr": 3e-4,
            "discount_factor": 0.99,
            "tau": 0.005,  # Soft update (SAC uses slightly higher than TD3)
            "batch_size": 256,
            "replay_buffer_size": 1000000,
            "hidden_dims": [256, 256],
            "target_entropy": None,  # Auto-computed as -action_dim
            "auto_tune_temperature": True,
            "initial_temperature": 0.2,
        }
        self.config = {**default_config, **(config or {})}

        # Target entropy for automatic temperature tuning
        if self.config["target_entropy"] is None:
            self.config["target_entropy"] = -float(action_dim)

        # Stochastic actor network
        self.actor = SACStochasticActor(
            state_dim=state_dim,
            action_dim=action_dim,
            population_dim=population_dim,
            action_bounds=action_bounds,
            hidden_dims=self.config["hidden_dims"],
        )

        # Twin soft Q-critics (reuse TD3 architecture)
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

        # Temperature (entropy weight)
        if self.config["auto_tune_temperature"]:
            # Learnable log(alpha)
            self.log_alpha = torch.tensor([np.log(self.config["initial_temperature"])], requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config["alpha_lr"])
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = self.config["initial_temperature"]
            self.log_alpha = None

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
        Select action using stochastic policy.

        Args:
            state: Individual state [state_dim]
            population_state: Population state [pop_dim]
            training: If True, sample from policy; else use mean

        Returns:
            Continuous action [action_dim]
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            pop_state_t = torch.FloatTensor(population_state).unsqueeze(0)

            if training:
                action, _, _ = self.actor.sample(state_t, pop_state_t)
                return action.squeeze(0).numpy()
            else:
                # Use mean for evaluation
                _, _, mean_action = self.actor.sample(state_t, pop_state_t)
                return mean_action.squeeze(0).numpy()

    def update(self) -> dict[str, float] | None:
        """
        Update actor, twin critics, and temperature.

        Returns:
            Dictionary of losses or None if buffer too small
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

        # Update critics
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs, _ = self.actor.sample(next_states, next_population_states)

            # Compute soft target: min(Q1', Q2') - α log π
            target_q1 = self.critic1_target(next_states, next_actions, next_population_states)
            target_q2 = self.critic2_target(next_states, next_actions, next_population_states)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs.squeeze(-1)

            # Soft Bellman backup
            target = rewards + self.config["discount_factor"] * target_q * (~dones)

        # Critic 1 loss
        current_q1 = self.critic1(states, actions, population_states)
        critic1_loss = torch.nn.functional.mse_loss(current_q1, target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # Critic 2 loss
        current_q2 = self.critic2(states, actions, population_states)
        critic2_loss = torch.nn.functional.mse_loss(current_q2, target)

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update actor
        sampled_actions, log_probs, _ = self.actor.sample(states, population_states)
        q1 = self.critic1(states, sampled_actions, population_states)
        q2 = self.critic2(states, sampled_actions, population_states)
        q = torch.min(q1, q2)

        # Actor loss: maximize soft Q-value = minimize -(Q - α log π)
        actor_loss = (self.alpha * log_probs.squeeze(-1) - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update temperature (if auto-tuning)
        alpha_loss = 0.0
        if self.config["auto_tune_temperature"]:
            alpha_loss = -(
                self.log_alpha.exp() * (log_probs.squeeze(-1) + self.config["target_entropy"]).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss.item()

        # Soft update target networks
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        self.update_count += 1

        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss,
            "alpha": self.alpha,
            "entropy": -log_probs.mean().item(),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Soft update target network parameters."""
        tau = self.config["tau"]
        for target_param, param in zip(target.parameters(), source.parameters(), strict=False):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def train(self, num_episodes: int = 1000) -> dict[str, Any]:
        """
        Train SAC agent.

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
        alpha_values = []
        entropy_values = []

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

                # Select action (stochastic during training)
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
                    critic1_losses.append(losses["critic1_loss"])
                    critic2_losses.append(losses["critic2_loss"])
                    actor_losses.append(losses["actor_loss"])
                    alpha_values.append(losses["alpha"])
                    entropy_values.append(losses["entropy"])

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
                    f"Alpha={np.mean(alpha_values[-100:]) if alpha_values else self.alpha:.3f}, "
                    f"Entropy={np.mean(entropy_values[-100:]) if entropy_values else 0:.3f}"
                )

        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "critic1_losses": critic1_losses,
            "critic2_losses": critic2_losses,
            "actor_losses": actor_losses,
            "alpha_values": alpha_values,
            "entropy_values": entropy_values,
        }

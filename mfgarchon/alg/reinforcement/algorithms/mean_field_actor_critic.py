"""
Mean Field Actor-Critic Algorithm.

This module implements an actor-critic approach for Mean Field Games, where agents
learn both policy (actor) and value function (critic) that depend on the population
state (mean field).

Key Features:
- Actor network: π(a|s,m) - policy conditioned on state and population
- Critic network: V(s,m) or Q(s,a,m) - value function with mean field
- Population-aware advantage estimation
- Policy gradient methods with population dynamics

Mathematical Framework:
- Policy: π(a|s,m) = P(a|s, population_state)
- Value: V(s,m) = E[∑ γ^t r(s_t, a_t, m_t) | s_0=s, m_0=m, π]
- Advantage: A(s,a,m) = Q(s,a,m) - V(s,m)
- Policy Gradient: ∇θ J(θ) = E[∇θ log π(a|s,m) A(s,a,m)]

Author: MFGarchon Team
Date: October 2025
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as torch_f
    import torch.optim as optim
    from torch.distributions import Categorical

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from mfgarchon.utils.mfg_logging.logger import get_logger

logger = get_logger(__name__)


class ActorNetwork(nn.Module):
    """
    Actor network for Mean Field RL.

    Outputs policy π(a|s,m) conditioned on both individual state
    and population state.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        population_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.population_dim = population_dim

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            self.activation,
            nn.Linear(hidden_dims[0], hidden_dims[0] // 2),
            self.activation,
        )

        # Population encoder
        self.population_encoder = nn.Sequential(
            nn.Linear(population_dim, hidden_dims[0]),
            self.activation,
            nn.Linear(hidden_dims[0], hidden_dims[0] // 2),
            self.activation,
        )

        # Policy head
        fusion_input_dim = hidden_dims[0]
        self.policy_head = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dims[1]),
            self.activation,
            nn.Linear(hidden_dims[1], hidden_dims[1]),
            self.activation,
            nn.Linear(hidden_dims[1], action_dim),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        # Special initialization for policy output layer (smaller scale)
        if isinstance(module, nn.Linear) and module.out_features == self.action_dim:
            nn.init.orthogonal_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor, population_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through actor network.

        Args:
            state: Individual agent state [batch_size, state_dim]
            population_state: Population state [batch_size, population_dim]

        Returns:
            Action logits [batch_size, action_dim]
        """
        # Encode states
        state_features = self.state_encoder(state)
        population_features = self.population_encoder(population_state)

        # Combine features
        combined_features = torch.cat([state_features, population_features], dim=1)

        # Compute policy logits
        logits = self.policy_head(combined_features)

        return logits

    def get_action(
        self, state: torch.Tensor, population_state: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            state: Individual state
            population_state: Population state
            deterministic: If True, return argmax action

        Returns:
            action: Selected action
            log_prob: Log probability of action
        """
        logits = self.forward(state, population_state)
        probs = torch_f.softmax(logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
            log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1)))
        else:
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob


class CriticNetwork(nn.Module):
    """
    Critic network for Mean Field RL.

    Outputs value function V(s,m) or Q(s,a,m) conditioned on
    state and population.
    """

    def __init__(
        self,
        state_dim: int,
        population_dim: int,
        action_dim: int | None = None,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        critic_type: str = "v",  # "v" for V(s,m), "q" for Q(s,a,m)
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.state_dim = state_dim
        self.population_dim = population_dim
        self.action_dim = action_dim
        self.critic_type = critic_type

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Input dimension depends on critic type
        if critic_type == "q":
            if action_dim is None:
                raise ValueError("action_dim required for Q-critic")
            input_dim = state_dim + population_dim + action_dim
        else:  # V-critic
            input_dim = state_dim + population_dim

        # Value network
        self.value_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            self.activation,
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            self.activation,
            nn.Linear(hidden_dims[1], hidden_dims[1]),
            self.activation,
            nn.Linear(hidden_dims[1], 1),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.zeros_(module.bias)

    def forward(
        self, state: torch.Tensor, population_state: torch.Tensor, action: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass through critic network.

        Args:
            state: Individual agent state [batch_size, state_dim]
            population_state: Population state [batch_size, population_dim]
            action: Action (for Q-critic) [batch_size, action_dim]

        Returns:
            Value estimates [batch_size, 1]
        """
        if self.critic_type == "q":
            if action is None:
                raise ValueError("Action required for Q-critic")
            # One-hot encode action if needed
            if action.dim() == 1:
                action_onehot = torch_f.one_hot(action.long(), num_classes=self.action_dim).float()
            else:
                action_onehot = action
            combined_input = torch.cat([state, population_state, action_onehot], dim=1)
        else:  # V-critic
            combined_input = torch.cat([state, population_state], dim=1)

        value = self.value_network(combined_input)
        return value


class MeanFieldActorCritic:
    """
    Mean Field Actor-Critic algorithm for MFG problems.

    Learns both policy (actor) and value function (critic) that depend
    on individual state and population state (mean field).
    """

    def __init__(
        self,
        env: Any,
        state_dim: int,
        action_dim: int,
        population_dim: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        hidden_dims: list[int] | None = None,
        device: str | None = None,
    ) -> None:
        """
        Initialize Mean Field Actor-Critic.

        Args:
            env: MFG environment
            state_dim: Dimension of individual state
            action_dim: Number of discrete actions
            population_dim: Dimension of population state
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_epsilon: PPO clipping epsilon
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            hidden_dims: Hidden layer dimensions
            device: Device to use (cpu/cuda/mps)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Mean Field Actor-Critic")

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.population_dim = population_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, population_dim, hidden_dims).to(self.device)
        self.critic = CriticNetwork(state_dim, population_dim, hidden_dims=hidden_dims, critic_type="v").to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Training stats
        self.training_step = 0

        logger.info(f"Initialized Mean Field Actor-Critic on device: {self.device}")
        logger.info(f"Actor parameters: {sum(p.numel() for p in self.actor.parameters()):,}")
        logger.info(f"Critic parameters: {sum(p.numel() for p in self.critic.parameters()):,}")

    def select_action(
        self, state: np.ndarray, population_state: np.ndarray, deterministic: bool = False
    ) -> tuple[int, float]:
        """
        Select action using current policy.

        Args:
            state: Individual agent state
            population_state: Population state
            deterministic: If True, return greedy action

        Returns:
            action: Selected action
            log_prob: Log probability of action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        pop_tensor = torch.FloatTensor(population_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob = self.actor.get_action(state_tensor, pop_tensor, deterministic)

        return int(action.item()), float(log_prob.item())

    def compute_gae(
        self,
        rewards: list[float],
        values: list[float],
        next_value: float,
        dones: list[bool],
    ) -> tuple[list[float], list[float]]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: List of rewards
            values: List of value estimates
            next_value: Value estimate for next state
            dones: List of done flags

        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        advantages: list[float] = []
        returns: list[float] = []
        gae = 0
        next_value = next_value

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return advantages, returns

    def train(
        self,
        num_episodes: int = 1000,
        max_steps_per_episode: int = 200,
        log_interval: int = 10,
        save_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Train the actor-critic agent.

        Args:
            num_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            log_interval: Logging frequency
            save_path: Path to save model checkpoints

        Returns:
            Training statistics
        """
        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            # Reset environment
            reset_result = self.env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            state = self._extract_state(obs)
            population_state = self._extract_population(obs)

            # Episode data
            states = []
            population_states = []
            actions = []
            rewards = []
            log_probs = []
            values = []
            dones = []

            episode_reward = 0
            episode_length = 0

            # Collect trajectory
            for _ in range(max_steps_per_episode):
                # Select action
                action, log_prob = self.select_action(state, population_state)

                # Get value estimate
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                pop_tensor = torch.FloatTensor(population_state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    value = self.critic(state_tensor, pop_tensor).item()

                # Step environment
                next_obs, reward, done, truncated, _info = self.env.step(action)
                done = done or truncated

                # Convert reward to scalar if needed
                reward_scalar = float(reward) if isinstance(reward, np.ndarray | np.generic) else reward

                # Store experience
                states.append(state)
                population_states.append(population_state)
                actions.append(action)
                rewards.append(reward_scalar)
                log_probs.append(log_prob)
                values.append(value)
                dones.append(done)

                episode_reward += reward_scalar
                episode_length += 1

                # Update state
                state = self._extract_state(next_obs)
                population_state = self._extract_population(next_obs)

                if done:
                    break

            # Compute next value for GAE
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            pop_tensor = torch.FloatTensor(population_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                next_value = self.critic(state_tensor, pop_tensor).item()

            # Compute advantages and returns
            advantages, returns = self.compute_gae(rewards, values, next_value, dones)

            # Update policy
            self._update_policy(states, population_states, actions, log_probs, advantages, returns)

            # Record stats
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Logging
            if (episode + 1) % log_interval == 0:
                mean_reward = np.mean(episode_rewards[-log_interval:])
                mean_length = np.mean(episode_lengths[-log_interval:])
                logger.info(
                    f"Episode {episode + 1}/{num_episodes} | Reward: {mean_reward:.2f} | Length: {mean_length:.1f}"
                )

        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
        }

    def _update_policy(
        self,
        states: list[np.ndarray],
        population_states: list[np.ndarray],
        actions: list[int],
        old_log_probs: list[float],
        advantages: list[float],
        returns: list[float],
    ) -> None:
        """Update actor and critic networks."""
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        pop_states_tensor = torch.FloatTensor(np.array(population_states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages (with numerical stability)
        adv_std = advantages_tensor.std()
        if adv_std > 1e-8:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / adv_std
        else:
            # If all advantages are the same, center them at 0
            advantages_tensor = advantages_tensor - advantages_tensor.mean()

        # Get new log probs and values
        logits = self.actor(states_tensor, pop_states_tensor)

        # Check for NaN in logits (numerical stability)
        if torch.isnan(logits).any():
            logger.warning("NaN detected in actor logits - skipping update")
            return

        probs = torch_f.softmax(logits, dim=-1)
        dist = Categorical(probs)
        new_log_probs = dist.log_prob(actions_tensor)

        # Compute policy loss (PPO-style clipping)
        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor
        policy_loss = -torch.min(surr1, surr2).mean()

        # Compute value loss
        values = self.critic(states_tensor, pop_states_tensor).squeeze()
        value_loss = torch_f.mse_loss(values, returns_tensor)

        # Update actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # Update critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        self.training_step += 1

    def _extract_state(self, obs: dict | np.ndarray) -> np.ndarray:
        """Extract individual state from observation."""
        if isinstance(obs, dict):
            return obs.get("agent_position", obs.get("state", np.zeros(self.state_dim)))
        return obs[: self.state_dim] if len(obs) > self.state_dim else obs

    def _extract_population(self, obs: dict | np.ndarray) -> np.ndarray:
        """Extract the population (mean-field) state from an observation.

        Issue #1568 (extends #1508): a zero-filled population is the silent non-MFG failure that
        #1508 made DDPG/TD3/SAC raise on -- training on an identically-zero mean field returns a
        policy the user trusts as MFG-coupled when the coupling channel was never present. This
        algorithm reads the population from the OBSERVATION (it does not query
        ``env.get_population_state()``), so it must fail loud when the observation carries no
        population channel instead of zero-filling.
        """
        if isinstance(obs, dict):
            pop = obs.get("local_density", obs.get("population"))
            if pop is not None:
                return pop
        elif len(obs) > self.state_dim:
            return obs[self.state_dim :]

        has_env_getter = callable(getattr(self.env, "get_population_state", None))
        hint = (
            "The env exposes get_population_state(), but MeanFieldActorCritic reads the population "
            "from the observation, not from that method -- embed the population channel in the "
            "observation (dict key 'local_density'/'population', or as a tail slice past state_dim)."
            if has_env_getter
            else "The env exposes no get_population_state() and the observation carries no population."
        )
        raise AttributeError(
            "MeanFieldActorCritic observation carries no population (mean-field) channel. A zero "
            "fallback would train on an identically-zero mean field and silently return a non-MFG "
            f"policy (Issue #1508/#1568). {hint}"
        )

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "training_step": self.training_step,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.training_step = checkpoint["training_step"]
        logger.info(f"Model loaded from {path}")

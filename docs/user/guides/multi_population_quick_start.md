# [FROZEN 2026-07-30] Multi-Population Continuous Control - Quick Start Guide

**Status**: ⛔ Frozen prototype — see the banner below

> ⛔ **FROZEN PROTOTYPE.** `mfgarchon.alg.reinforcement` is a design prototype, not under
> development — see `CLAUDE.md` § "FROZEN: alg/neural/ and alg/reinforcement/". The status line
> below predates that decision. Treat everything here as illustrative of intent rather than as a
> supported contract: the behaviour is not pinned, and it may change or be removed without a
> deprecation cycle.

**Version**: Phase 3.4
**Last Updated**: October 2025

---

## 🚀 Quick Start

### 1. Import Algorithms

```python
from mfgarchon.alg.reinforcement.algorithms import (
    MultiPopulationDDPG,  # Deterministic policies
    MultiPopulationTD3,   # Twin critics + delayed updates
    MultiPopulationSAC,   # Stochastic + entropy regularization
)
```

### 2. Create Multi-Population Environment

```python
from mfgarchon.alg.reinforcement.environments.multi_population_env import (
    MultiPopulationMFGEnv,
)

class MyMultiPopEnv(MultiPopulationMFGEnv):
    """Your custom environment with N populations."""

    def _sample_initial_state(self, pop_id: int):
        # Return initial state for population pop_id
        pass

    def _compute_population_distribution(self, pop_id: int):
        # Return normalized distribution for population pop_id
        pass

    def _dynamics(self, pop_id, state, action, population_states):
        # Return next state: s' = f(s, a, m₁, ..., mₙ)
        pass

    def _reward(self, pop_id, state, action, next_state, population_states):
        # Return reward: r(s, a, s', m₁, ..., mₙ)
        pass

# Initialize environment
env = MyMultiPopEnv(
    num_populations=3,
    state_dims=[2, 2, 2],           # Can be heterogeneous: [2, 3, 1]
    action_specs=[
        {"type": "continuous", "dim": 2, "bounds": (-1, 1)},
        {"type": "continuous", "dim": 2, "bounds": (-2, 2)},
        {"type": "continuous", "dim": 1, "bounds": (0, 1)},
    ],
    population_sizes=[100, 100, 100],  # For discretization
)
```

### 3. Train Algorithm

```python
# Initialize algorithm (all 3 have identical API)
algo = MultiPopulationSAC(  # or DDPG, TD3
    env=env,
    num_populations=3,
    state_dims=[2, 2, 2],
    action_dims=[2, 2, 1],
    population_dims=[100, 100, 100],
    action_bounds=[(-1, 1), (-2, 2), (0, 1)],
)

# Train
stats = algo.train(num_episodes=1000)

# Access results
episode_rewards = stats["episode_rewards"]  # {pop_id: [rewards]}
episode_lengths = stats["episode_lengths"]  # [lengths]
```

---

## 📊 Algorithm Comparison

| Algorithm | Policy Type | Critics | Exploration | Best For |
|-----------|-------------|---------|-------------|----------|
| **DDPG** | Deterministic | Single | OU Noise | Fast prototyping |
| **TD3** | Deterministic | Twin | Gaussian Noise | Stable training |
| **SAC** | Stochastic | Twin | Entropy | Exploration, Sample efficiency |

### When to Use Each

**MultiPopulationDDPG**:
- ✅ Fast training, simple implementation
- ✅ Good for well-understood problems
- ❌ Can overestimate Q-values
- ❌ Less stable than TD3/SAC

**MultiPopulationTD3**:
- ✅ Reduced overestimation bias (twin critics)
- ✅ More stable than DDPG (delayed updates)
- ✅ Robust to hyperparameters
- ❌ Deterministic policies (single equilibrium)

**MultiPopulationSAC**:
- ✅ Stochastic policies explore multiple equilibria
- ✅ Automatic temperature tuning
- ✅ Best sample efficiency
- ✅ Most robust to distribution shifts
- ❌ Highest computational cost

---

## 🎛️ Configuration Options

### Common Config (All Algorithms)

```python
config = {
    "actor_lr": 1e-4,                # Actor learning rate
    "critic_lr": 1e-3,               # Critic learning rate
    "discount_factor": 0.99,         # Discount factor γ
    "tau": 0.001,                    # Soft update coefficient
    "batch_size": 256,               # Minibatch size
    "replay_buffer_size": 100000,    # Replay buffer capacity
    "hidden_dims": [256, 128],       # Network architecture
}
```

### DDPG-Specific Config

```python
ddpg_config = {
    **config,
    "ou_theta": 0.15,               # OU noise mean reversion speed
    "ou_sigma": 0.2,                # OU noise volatility
}
```

### TD3-Specific Config

```python
td3_config = {
    **config,
    "policy_delay": 2,              # Delayed policy updates
    "target_noise_std": 0.2,        # Target policy smoothing noise
    "target_noise_clip": 0.5,       # Noise clipping range
    "exploration_noise_std": 0.1,   # Training exploration noise
}
```

### SAC-Specific Config

```python
sac_config = {
    **config,
    "alpha_lr": 3e-4,               # Temperature learning rate
    "target_entropy": None,         # Auto: -action_dim per population
    "auto_tune_temperature": True,  # Enable automatic α tuning
    "initial_temperature": 0.2,     # Starting temperature
}
```

---

## 💡 Best Practices

### 1. Environment Design

```python
class MyMultiPopEnv(MultiPopulationMFGEnv):
    """Best practices for environment implementation."""

    def _dynamics(self, pop_id, state, action, population_states):
        # ✅ Ensure next_state stays in valid bounds
        next_state = state + self.dt * action
        return np.clip(next_state, self.state_bounds[0], self.state_bounds[1])

    def _reward(self, pop_id, state, action, next_state, population_states):
        # ✅ Scale rewards to reasonable range (e.g., [-10, 10])
        # ✅ Include cross-population coupling
        own_cost = self._compute_cost(state, action)
        coupling_cost = self._compute_coupling(pop_id, population_states)
        return -(own_cost + coupling_cost)  # Negative for minimization

    def _compute_population_distribution(self, pop_id):
        # ✅ Always return normalized distribution
        hist = self._compute_histogram(pop_id)
        return hist / np.sum(hist)
```

### 2. Training Tips

```python
# ✅ Use seeded resets for reproducibility
states, _ = env.reset(seed=42)

# ✅ Monitor convergence
for episode in range(num_episodes):
    stats = algo.train(num_episodes=100)
    avg_rewards = {i: np.mean(stats["episode_rewards"][i][-100:])
                   for i in range(num_populations)}
    print(f"Ep {episode}: {avg_rewards}")

    # Check for Nash equilibrium convergence
    if all(abs(avg_rewards[i] - target[i]) < threshold for i in range(N)):
        print("Nash equilibrium reached!")
        break
```

### 3. Hyperparameter Tuning

```python
# Start with defaults, then tune if needed:

# If training is unstable:
config["tau"] = 0.0005          # Slower target updates
config["batch_size"] = 128      # Smaller batches

# If learning is too slow:
config["actor_lr"] = 3e-4       # Higher learning rate
config["batch_size"] = 512      # Larger batches

# If overestimation is an issue:
# Use TD3 or SAC instead of DDPG

# If exploration is insufficient:
# Use SAC with higher initial_temperature
config["initial_temperature"] = 0.5
```

---

## 🔍 Debugging Guide

### Common Issues

#### 1. Training Diverges

**Symptoms**: Rewards decrease or oscillate wildly

**Solutions**:
```python
# Reduce learning rates
config["actor_lr"] = 1e-5
config["critic_lr"] = 1e-4

# Increase tau (slower target updates)
config["tau"] = 0.0005

# Scale rewards
def _reward(self, ...):
    return raw_reward / reward_scale  # e.g., reward_scale = 10
```

#### 2. No Learning Progress

**Symptoms**: Rewards don't improve

**Solutions**:
```python
# Check replay buffer is filling
print(f"Buffer size: {len(algo.replay_buffers[0])}")

# Increase exploration (for SAC)
config["initial_temperature"] = 0.5

# Verify reward function has gradients
# Ensure rewards change with actions
```

#### 3. Populations Don't Converge to Nash

**Symptoms**: Different populations have conflicting behaviors

**Solutions**:
```python
# Ensure cross-population coupling in rewards
def _reward(self, pop_id, ...):
    # Include terms depending on other populations
    coupling = sum(interaction(pop_id, j, population_states[j])
                   for j in range(N) if j != pop_id)
    return own_reward + coupling_weight * coupling

# Increase training time
num_episodes = 5000  # Nash may take longer to reach

# Check population distributions are updating
pop_states = env.get_population_states()
print({i: np.sum(pop_states[i]) for i in range(N)})  # Should be 1.0
```

---

## 📈 Monitoring Training

### Key Metrics to Track

```python
stats = algo.train(num_episodes=1000)

# 1. Episode Rewards (per population)
import matplotlib.pyplot as plt

for pop_id in range(num_populations):
    plt.plot(stats["episode_rewards"][pop_id], label=f"Pop {pop_id}")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.title("Training Progress")
plt.show()

# 2. Critic Losses (convergence indicator)
for pop_id in range(num_populations):
    losses = stats["critic1_losses"][pop_id]  # or critic_losses for DDPG
    plt.plot(losses, label=f"Pop {pop_id}")
plt.xlabel("Update Step")
plt.ylabel("Critic Loss")
plt.legend()
plt.title("Critic Convergence")
plt.show()

# 3. Temperature (SAC only)
if "alphas" in stats:
    for pop_id in range(num_populations):
        plt.plot(stats["alphas"][pop_id], label=f"Pop {pop_id}")
    plt.xlabel("Update Step")
    plt.ylabel("Temperature α")
    plt.legend()
    plt.title("Temperature Evolution")
    plt.show()
```

---

## 🎯 Example: 2-Population Competing Agents

```python
from mfgarchon.alg.reinforcement.environments.multi_population_env import (
    MultiPopulationMFGEnv,
)
from mfgarchon.alg.reinforcement.algorithms import MultiPopulationSAC
import numpy as np

class CompetingAgentsEnv(MultiPopulationMFGEnv):
    """
    Two populations competing for resources.

    State: [position, velocity] for each agent
    Action: acceleration
    Reward: -travel_cost - congestion_penalty
    """

    def _sample_initial_state(self, pop_id):
        # Random position and velocity
        return np.random.uniform(0, 1, size=2)

    def _compute_population_distribution(self, pop_id):
        # Uniform distribution (simplified)
        return np.ones(self.population_sizes[pop_id]) / self.population_sizes[pop_id]

    def _dynamics(self, pop_id, state, action, population_states):
        pos, vel = state
        acc = action[0]

        # Simple kinematic model
        new_vel = vel + self.dt * acc
        new_pos = pos + self.dt * new_vel

        return np.array([np.clip(new_pos, 0, 1), np.clip(new_vel, -1, 1)])

    def _reward(self, pop_id, state, action, next_state, population_states):
        # Travel cost
        travel_cost = np.sum(action**2)

        # Congestion (interaction with other population)
        other_id = 1 - pop_id
        congestion = np.mean(population_states[other_id])  # Simplified

        return -(travel_cost + 0.5 * congestion)

# Create environment
env = CompetingAgentsEnv(
    num_populations=2,
    state_dims=2,
    action_specs=[
        {"type": "continuous", "dim": 1, "bounds": (-1, 1)},
        {"type": "continuous", "dim": 1, "bounds": (-1, 1)},
    ],
    population_sizes=100,
)

# Train with SAC
algo = MultiPopulationSAC(
    env=env,
    num_populations=2,
    state_dims=2,
    action_dims=[1, 1],
    population_dims=100,
    action_bounds=[(-1, 1), (-1, 1)],
)

# Train and monitor
stats = algo.train(num_episodes=1000)

print(f"Final rewards - Pop 0: {np.mean(stats['episode_rewards'][0][-100:]):.2f}")
print(f"Final rewards - Pop 1: {np.mean(stats['episode_rewards'][1][-100:]):.2f}")
```

---

## 📚 References

**Implementation Files**:
- Environment: `mfgarchon/alg/reinforcement/environments/multi_population_env.py`
- DDPG: `mfgarchon/alg/reinforcement/algorithms/multi_population_ddpg.py`
- TD3: `mfgarchon/alg/reinforcement/algorithms/multi_population_td3.py`
- SAC: `mfgarchon/alg/reinforcement/algorithms/multi_population_sac.py`

**Tests**:
- `tests/unit/test_multi_population_env.py`
- `tests/unit/test_multi_population_algorithms_basic.py`

**Documentation**:
- Progress: `docs/development/PHASE_3_4_PROGRESS.md`
- Handoff: `docs/development/PHASE_3_4_HANDOFF.md`

---

**Version**: 1.0
**Status**: ⛔ Frozen prototype — see the banner at the top of this guide
**Last Updated**: October 2025

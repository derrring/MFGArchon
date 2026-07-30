# [FROZEN 2026-07-30] Continuous MFG Environments User Guide

> ⛔ **FROZEN PROTOTYPE.** `mfgarchon.alg.reinforcement` is a design prototype, not under
> development — see `CLAUDE.md` § "FROZEN: alg/neural/ and alg/reinforcement/". Every status, version and readiness marker in this document predates that decision. Treat everything here as illustrative of intent rather than as a
> supported contract: the behaviour is not pinned, and it may change or be removed without a
> deprecation cycle.


Comprehensive guide to the continuous action space environments in MFGArchon's reinforcement learning framework.

## Overview

The continuous environments library provides **5 prototype environments** (described as production-ready before the freeze) for benchmarking RL algorithms (DDPG, TD3, SAC) on diverse Mean Field Game problems. Each environment demonstrates different aspects of MFG theory and provides distinct algorithmic challenges.

**Package**: `mfgarchon.alg.reinforcement.environments`
**Base Class**: `ContinuousMFGEnvBase`
**API Standard**: Gymnasium (OpenAI Gym)

## Environment Catalog

### 1. LQ-MFG Environment

**Module**: `lq_mfg_env.py`
**Class**: `LQMFGEnv`

**Description**: Linear-Quadratic Mean Field Game with analytical solution. Ideal for algorithm validation and baseline comparisons.

**Mathematical Formulation**:
- **State**: $x \in \mathbb{R}^2$ (position + velocity)
- **Action**: $u \in \mathbb{R}^2$ (control input)
- **Dynamics**: $\dot{x} = Ax + Bu + \sigma dW$
- **Cost**: $J = \int_0^T [x^T Q x + u^T R u + \alpha \|x - \bar{x}\|^2] dt$

**State Space** (dim=2):
- Position: $x \in [-10, 10]$
- Velocity: $v \in [-5, 5]$

**Action Space** (dim=2):
- Control: $u \in [-u_{max}, u_{max}]^2$

**Reward Structure**:
```python
r = -state_cost * (x^2 + v^2)           # State penalty
    - control_cost * (u_x^2 + u_y^2)    # Control effort
    - coupling_cost * ||x - mean_x||^2  # Mean field coupling
```

**Use Cases**:
- **Baseline benchmarking**: Analytical solution available for validation
- **Algorithm debugging**: Simple dynamics for testing convergence
- **Hyperparameter tuning**: Quick experiments due to low dimensionality

**Example**:
```python
from mfgarchon.alg.reinforcement.environments import LQMFGEnv
from mfgarchon.alg.reinforcement.algorithms import MeanFieldSAC

# Create environment
env = LQMFGEnv(
    num_agents=100,
    state_cost=1.0,
    control_cost=0.1,
    coupling_cost=0.5,
)

# Train algorithm
algo = MeanFieldSAC(env, state_dim=2, action_dim=2)
stats = algo.train(num_episodes=500)
```

---

### 2. Crowd Navigation Environment

**Module**: `crowd_navigation_env.py`
**Class**: `CrowdNavigationEnv`

**Description**: 2D spatial navigation where agents navigate toward goals while avoiding crowded regions.

**Mathematical Formulation**:
- **State**: $(x, y, v_x, v_y, \theta_{goal}) \in \mathbb{R}^5$
- **Action**: $(a_x, a_y) \in [-a_{max}, a_{max}]^2$ (acceleration)
- **Dynamics**: $\dot{x} = v$, $\dot{v} = a + \text{noise}$
- **Cost**: Distance to goal + velocity penalty + crowd avoidance + control cost

**State Space** (dim=5):
- Position: $(x, y) \in [0, L]^2$
- Velocity: $(v_x, v_y) \in [-v_{max}, v_{max}]^2$
- Goal angle: $\theta_{goal} \in [0, 2\pi]$

**Action Space** (dim=2):
- Acceleration: $(a_x, a_y) \in [-a_{max}, a_{max}]^2$

**Reward Structure**:
```python
r = -distance_cost * ||pos - goal||^2        # Distance to goal
    - velocity_cost * ||velocity||^2         # Velocity penalty
    - control_cost * ||action||^2            # Control effort
    - crowd_cost * density(position)         # Crowd avoidance (MFG coupling)
    + goal_bonus * I(reached_goal)           # Goal arrival bonus
```

**Termination**:
- **Early**: Goal reached (distance < threshold)
- **Truncation**: Max steps exceeded

**Use Cases**:
- **Pedestrian dynamics**: Crowd flow in public spaces
- **Robot navigation**: Multi-agent path planning
- **Evacuation planning**: Emergency scenario modeling

**Example**:
```python
from mfgarchon.alg.reinforcement.environments import CrowdNavigationEnv

env = CrowdNavigationEnv(
    num_agents=100,
    domain_size=10.0,
    goal_threshold=0.5,
    max_velocity=2.0,
)

# All agents share the same randomly sampled goal
state, info = env.reset(seed=42)
print(f"Goal position: {info['goal_position']}")
```

---

### 3. Price Formation Environment

**Module**: `price_formation_env.py`
**Class**: `PriceFormationEnv`

**Description**: Market making where agents set bid/ask spreads while managing inventory risk and liquidity effects.

**Mathematical Formulation**:
- **State**: $(q, p, \dot{p}, m_{depth}) \in \mathbb{R}^4$
- **Action**: $(\delta_{bid}, \delta_{ask}) \in [0, \delta_{max}]^2$
- **Dynamics**:
  - Price: $dp = \sigma dW + \lambda \cdot \text{imbalance}$
  - Inventory: $dq = \text{orders\_filled}$
- **Cost**: -PnL + inventory penalty + spread cost + liquidity depletion

**State Space** (dim=4):
- Inventory: $q \in [-Q_{max}, Q_{max}]$
- Mid-price: $p \in [p_{min}, p_{max}]$
- Price velocity: $\dot{p}$ (recent price change)
- Market depth: $m_{depth} \in [0, 1]$

**Action Space** (dim=2):
- Bid spread: $\delta_{bid} \in [0, \delta_{max}]$
- Ask spread: $\delta_{ask} \in [0, \delta_{max}]$

**Reward Structure**:
```python
r = pnl(spread, fills)                       # Profit from market making
    - inventory_penalty * q^2                # Quadratic inventory risk
    - spread_cost * avg_spread               # Opportunity cost
    - liquidity_penalty * density(price)     # MFG coupling (crowding)
```

**Termination**:
- **Early**: Inventory exceeds 95% of limit (bankruptcy risk)
- **Truncation**: Max steps exceeded

**Use Cases**:
- **Market making strategy**: Optimal spread setting
- **High-frequency trading**: Inventory management under uncertainty
- **Liquidity provision**: Multi-agent market simulation

**Example**:
```python
from mfgarchon.alg.reinforcement.environments import PriceFormationEnv

env = PriceFormationEnv(
    q_max=10.0,
    delta_max=1.0,
    price_volatility=0.5,
    inventory_penalty=0.5,
    liquidity_penalty=0.3,
)

# Market making episode
state, _ = env.reset()
for _ in range(200):
    # Set tight spreads
    action = np.array([0.1, 0.1])  # [bid_spread, ask_spread]
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

---

### 4. Resource Allocation Environment

**Module**: `resource_allocation_env.py`
**Class**: `ResourceAllocationEnv`

**Description**: Portfolio optimization where agents allocate capital across assets with simplex constraints and congestion effects.

**Mathematical Formulation**:
- **State**: $(w, v) \in \Delta^n \times \mathbb{R}_+^n$ (allocation + asset values)
- **Action**: $\Delta w \in \mathbb{R}^n$ with $\sum \Delta w_i = 0$ (rebalancing)
- **Dynamics**:
  - Asset values: $v' = v \odot (1 + \mu + \sigma \epsilon)$
  - Allocation: $w' = \Pi_\Delta(w + \Delta w)$ (simplex projection)
- **Cost**: -returns + risk penalty + transaction cost + congestion

**State Space** (dim=2n, default n=3):
- Allocation: $w \in \Delta^n$ (probability simplex: $\sum w_i = 1$, $w_i \geq 0$)
- Asset values: $v \in \mathbb{R}_+^n$

**Action Space** (dim=n):
- Rebalancing: $\Delta w \in [-\Delta_{max}, \Delta_{max}]^n$ with zero-sum constraint

**Reward Structure**:
```python
r = portfolio_return(w, asset_returns)       # w^T · returns
    - risk_penalty * w^T Σ w                 # Quadratic risk (variance)
    - transaction_cost * ||Δw||_1            # L1 norm of rebalancing
    - congestion_penalty * w^T · density     # MFG coupling (crowding)
```

**Key Constraints**:
- **Simplex constraint**: $\sum w_i = 1$, $w_i \geq 0$ (enforced by projection)
- **Zero-sum rebalancing**: $\sum \Delta w_i = 0$ (no wealth creation)

**Termination**:
- **Early**: Allocation becomes invalid (NaN/Inf)
- **Truncation**: Max steps exceeded

**Use Cases**:
- **Portfolio optimization**: Asset allocation under uncertainty
- **Energy grid management**: Resource distribution with constraints
- **Computational resource allocation**: Job scheduling on clusters

**Example**:
```python
from mfgarchon.alg.reinforcement.environments import ResourceAllocationEnv

env = ResourceAllocationEnv(
    num_assets=3,
    delta_max=0.2,
    base_return=0.05,
    risk_penalty=0.5,
    transaction_cost=0.01,
    congestion_penalty=0.3,
)

# Portfolio management
state, _ = env.reset()
allocations = state[:3]  # Current allocation
values = state[3:]        # Asset values

# Rebalance toward equal weights
action = np.array([0.1, -0.05, -0.05])  # Zero-sum constraint
state, reward, _, _, _ = env.step(action)

# Allocation automatically projected to simplex
new_allocations = state[:3]
assert np.isclose(new_allocations.sum(), 1.0)  # Always sums to 1
```

---

### 5. Traffic Flow Environment

**Module**: `traffic_flow_env.py`
**Class**: `TrafficFlowEnv`

**Description**: Congestion-aware routing where agents travel through a corridor with velocity control and congestion dynamics.

**Mathematical Formulation**:
- **State**: $(x, v, t_{rem}) \in [0, L] \times [0, v_{max}] \times [0, T_{max}]$
- **Action**: $a \in [-a_{max}, a_{max}]$ (acceleration)
- **Dynamics**:
  - Position: $dx/dt = v$
  - Velocity: $dv/dt = a - \beta \cdot \rho(x) \cdot v$ (congestion drag)
  - Time: $dt_{rem}/dt = -1$
- **Cost**: -progress + fuel cost + time penalty - arrival bonus

**State Space** (dim=3):
- Position: $x \in [0, L]$ (corridor length)
- Velocity: $v \in [0, v_{max}]$
- Time remaining: $t_{rem} \in [0, T_{max}]$

**Action Space** (dim=1):
- Acceleration: $a \in [-a_{max}, a_{max}]$

**Reward Structure**:
```python
r = progress(x_new - x_old)                  # Forward movement
    - fuel_cost * |a|                        # Energy consumption
    - time_penalty * dt                      # Time cost
    + arrival_bonus * I(reached_destination) # Completion reward
    - congestion_cost * v · density(x)       # MFG coupling (speed in crowds)
```

**Congestion Dynamics**:
- Velocity drag: $-\beta \cdot \rho(x) \cdot v$ where $\rho(x)$ is local density
- Higher density → slower travel (strategic deceleration)

**Termination**:
- **Early**: Destination reached ($x \geq L$)
- **Truncation**: Time limit exceeded ($t_{rem} \leq 0$)

**Use Cases**:
- **Autonomous vehicle routing**: Traffic-aware path planning
- **Transportation networks**: Congestion management
- **Urban planning**: Traffic flow optimization

**Example**:
```python
from mfgarchon.alg.reinforcement.environments import TrafficFlowEnv

env = TrafficFlowEnv(
    corridor_length=10.0,
    v_max=2.0,
    a_max=1.0,
    time_limit=20.0,
    congestion_coeff=0.5,
)

# Navigate through corridor
state, _ = env.reset()
x, v, t_rem = state[0], state[1], state[2]

# Accelerate to reach destination
action = np.array([1.0])  # Maximum acceleration
state, reward, terminated, truncated, info = env.step(action)

if terminated:
    print("Destination reached!")
```

---

## Comparison Table

| Environment | State Dim | Action Dim | Constraints | MFG Coupling | Difficulty |
|-------------|-----------|------------|-------------|--------------|------------|
| **LQ-MFG** | 2 | 2 | None | Quadratic | ⭐ Easy |
| **Crowd Navigation** | 5 | 2 | Domain bounds | Spatial density | ⭐⭐ Medium |
| **Price Formation** | 4 | 2 | Inventory limits | Price crowding | ⭐⭐⭐ Hard |
| **Resource Allocation** | 6 | 3 | Simplex | Asset congestion | ⭐⭐⭐⭐ Very Hard |
| **Traffic Flow** | 3 | 1 | Time limit | Velocity drag | ⭐⭐ Medium |

## Common Usage Patterns

### Basic Environment Usage

```python
from mfgarchon.alg.reinforcement.environments import CrowdNavigationEnv

# 1. Create environment
env = CrowdNavigationEnv(num_agents=100)

# 2. Reset to initial state
state, info = env.reset(seed=42)

# 3. Interaction loop
for step in range(200):
    # Sample random action
    action = env.action_space.sample()

    # Execute action
    next_state, reward, terminated, truncated, info = env.step(action)

    # Check termination
    if terminated or truncated:
        break

    state = next_state

# 4. Cleanup
env.close()
```

### Training with SAC

```python
from mfgarchon.alg.reinforcement.algorithms import MeanFieldSAC
from mfgarchon.alg.reinforcement.environments import PriceFormationEnv

# Create environment
env = PriceFormationEnv(num_agents=100)

# Create algorithm
algo = MeanFieldSAC(
    env=env,
    state_dim=4,
    action_dim=2,
    population_dim=50,
    action_bounds=(0.0, 1.0),
    hidden_dim=256,
    buffer_size=100000,
)

# Train
stats = algo.train(
    num_episodes=1000,
    log_interval=10,
)

# Analyze results
import matplotlib.pyplot as plt
plt.plot(stats['episode_rewards'])
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('SAC Training on Price Formation')
plt.show()
```

### Benchmarking Multiple Algorithms

```python
from mfgarchon.alg.reinforcement.algorithms import MeanFieldDDPG, MeanFieldTD3, MeanFieldSAC
from mfgarchon.alg.reinforcement.environments import TrafficFlowEnv

env = TrafficFlowEnv(num_agents=100)

algorithms = {
    'DDPG': MeanFieldDDPG(env, state_dim=3, action_dim=1),
    'TD3': MeanFieldTD3(env, state_dim=3, action_dim=1),
    'SAC': MeanFieldSAC(env, state_dim=3, action_dim=1),
}

results = {}
for name, algo in algorithms.items():
    print(f"Training {name}...")
    stats = algo.train(num_episodes=500)
    results[name] = stats['episode_rewards'][-10:]  # Last 10 episodes

# Compare performance
for name, rewards in results.items():
    print(f"{name}: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
```

## Advanced Features

### Custom Environments

To create a custom continuous MFG environment, inherit from `ContinuousMFGEnvBase`:

```python
from mfgarchon.alg.reinforcement.environments import ContinuousMFGEnvBase
import numpy as np

class MyCustomEnv(ContinuousMFGEnvBase):
    def __init__(self, num_agents=100, **kwargs):
        super().__init__(
            num_agents=num_agents,
            state_dim=4,  # Your state dimension
            action_dim=2,  # Your action dimension
            action_bounds=(-1.0, 1.0),
            **kwargs
        )

    def _get_state_bounds(self):
        low = np.array([...], dtype=np.float32)
        high = np.array([...], dtype=np.float32)
        return low, high

    def _sample_initial_states(self):
        # Return (num_agents, state_dim) array
        return np.random.randn(self.num_agents, self.state_dim).astype(np.float32)

    def _drift(self, state, action, population):
        # Return drift vector (state_dim,)
        return np.zeros(self.state_dim, dtype=np.float32)

    def _individual_reward(self, state, action, next_state):
        # Return scalar reward
        return 0.0

    def compute_mean_field_coupling(self, state, population):
        # Return scalar coupling term
        return 0.0

    def get_population_state(self):
        # Return population histogram
        return np.ones(self.population_bins, dtype=np.float32) / self.population_bins
```

### Hyperparameter Tuning

Recommended hyperparameters for each environment:

**LQ-MFG**:
- Learning rate: `3e-4`
- Hidden dim: `128`
- Buffer size: `50000`
- Batch size: `64`

**Crowd Navigation**:
- Learning rate: `1e-4`
- Hidden dim: `256`
- Buffer size: `100000`
- Batch size: `128`

**Price Formation**:
- Learning rate: `5e-5`
- Hidden dim: `256`
- Buffer size: `200000`
- Batch size: `256`
- Target update: `0.001` (soft updates)

**Resource Allocation**:
- Learning rate: `1e-4`
- Hidden dim: `512` (complex constraints)
- Buffer size: `100000`
- Batch size: `256`

**Traffic Flow**:
- Learning rate: `3e-4`
- Hidden dim: `128`
- Buffer size: `50000`
- Batch size: `64`

## Testing

All environments have comprehensive test coverage:

```bash
# Run all environment tests
pytest tests/unit/test_lq_mfg_env.py
pytest tests/unit/test_crowd_navigation_env.py
pytest tests/unit/test_price_formation_env.py
pytest tests/unit/test_resource_allocation_env.py
pytest tests/unit/test_traffic_flow_env.py

# Run all at once (113 tests)
pytest tests/unit/test_*_env.py -v
```

## Examples

See `examples/basic/` for environment demonstrations:
- `lq_mfg_demo.py` - LQ-MFG training with all three algorithms

More advanced examples coming soon in `examples/advanced/`.

## Troubleshooting

### Environment Not Available

If you see `CONTINUOUS_MFG_AVAILABLE = False`:

```bash
pip install gymnasium
```

### Slow Training

- Reduce `num_agents` (default: 100 → 50)
- Increase `dt` (default: 0.1 → 0.2)
- Reduce `population_bins` (default: 50 → 20)

### Unstable Training

- Lower learning rate
- Increase batch size
- Use TD3 instead of DDPG (more stable)
- Enable target network soft updates

### Memory Issues

- Reduce `buffer_size`
- Reduce `num_agents`
- Use smaller `hidden_dim`

## References

- **LQ-MFG**: Lasry & Lions (2007) - Mean Field Games
- **Crowd Navigation**: Lachapelle & Wolfram (2011) - On a mean field game approach modeling congestion and aversion in pedestrian crowds
- **Price Formation**: Cardaliaguet & Lehalle (2018) - Mean field game of controls and an application to trade crowding
- **Resource Allocation**: Bank & Voß (2018) - Linear quadratic stochastic differential games with general noise processes
- **Traffic Flow**: Achdou et al. (2014) - Mean field games: numerical methods for the planning problem

## Related Documentation

- [Quickstart Guide](quickstart.md) - Getting started with MFGArchon
- [Advanced Examples](../../examples/advanced/) - More complex examples

---

**Last Updated**: 2025-10-04
**Version**: 1.5.0 (Phase 3.5 Complete)

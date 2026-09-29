# Stochastic MFG User Guide

> **Status (2026-09-29): stale against the current API** — the first code block raises (#2432).
> Build problems as in the [Quickstart](quickstart.md); the conventions are in
> [`CONVENTIONS.md`](CONVENTIONS.md).

**Target Audience**: Researchers and practitioners using MFGArchon for stochastic problems
**Prerequisites**: Basic understanding of Mean Field Games and HJB-FP systems
**Related Documentation**: [High Dimensional MFG](../theory/high_dimensional_mfg.md)

---

## Quick Start

### Your First Common Noise MFG

```python
from mfgarchon.core.stochastic import StochasticMFGProblem, OrnsteinUhlenbeckProcess
from mfgarchon.alg.numerical.stochastic import CommonNoiseMFGSolver

# 1. Define common noise (mean-reverting process)
noise = OrnsteinUhlenbeckProcess(kappa=2.0, mu=0.0, sigma=0.2)

# 2. Define how noise affects dynamics
def hamiltonian(x, p, m, theta):
    return 0.5 * p**2 + 0.5 * x**2 + 0.1 * theta * x

# 3. Create stochastic problem
problem = StochasticMFGProblem(
    xmin=-2.0, xmax=2.0, Nx=51,
    T=1.0, Nt=26,
    noise_process=noise,
    conditional_hamiltonian=hamiltonian
)

# 4. Solve
solver = CommonNoiseMFGSolver(problem, num_noise_samples=50)
result = solver.solve()

# 5. Access results
print(f"Monte Carlo error: {result.mc_error_u:.6f}")
print(f"Converged: {result.converged}")
```

**Output**:
```
Solving Common Noise MFG with 50 noise realizations...
Variance reduction: True
Parallel execution: True

[1/3] Sampling noise paths...
[2/3] Solving 50 conditional MFG problems...
  Progress: 50/50 (100%)
[3/3] Aggregating solutions via Monte Carlo...

✓ Completed in 12.34s
  MC error (u): 2.456e-03
  MC error (m): 1.234e-03
  Variance reduction factor: 2.34x
  All problems converged: True
```

---

## Common Noise Processes

### 1. Ornstein-Uhlenbeck Process

**Mean-reverting process** - typical for interest rates, volatility

```python
from mfgarchon.core.stochastic import OrnsteinUhlenbeckProcess

# VIX-like volatility process
vix = OrnsteinUhlenbeckProcess(
    kappa=2.0,    # Mean reversion speed (fast)
    mu=20.0,      # Long-term mean (20%)
    sigma=8.0     # Volatility of volatility
)

# Generate sample path
import numpy as np
path = vix.sample_path(T=1.0, Nt=100, seed=42)

# Path properties
print(f"Mean: {np.mean(path[-20:]):.2f}")  # ≈ 20.0 (converges to mu)
print(f"Std: {np.std(path[-20:]):.2f}")    # ≈ sigma/sqrt(2*kappa)
```

**When to use**:
- Financial: Interest rates, volatility indices
- Physics: Damped oscillations with noise
- Biology: Population levels with homeostasis

### 2. Cox-Ingersoll-Ross (CIR) Process

**Always positive** - for quantities that cannot go negative

```python
from mfgarchon.core.stochastic import CoxIngersollRossProcess

# Interest rate (always positive)
interest_rate = CoxIngersollRossProcess(
    kappa=1.5,    # Mean reversion
    mu=0.05,      # Long-term mean (5%)
    sigma=0.1     # Volatility
)
```

**Key property**: $d\theta_t = \kappa(\mu - \theta_t)dt + \sigma\sqrt{\theta_t}dB_t$

The $\sqrt{\theta_t}$ term ensures positivity (Feller condition).

**When to use**:
- Finance: Interest rates, credit spreads
- Epidemiology: Infection rates (always ≥ 0)
- Energy: Commodity prices

### 3. Geometric Brownian Motion

**Exponential growth** - classic for stock prices

```python
from mfgarchon.core.stochastic import GeometricBrownianMotion

# Stock index
market_index = GeometricBrownianMotion(
    mu=0.08,      # Drift (8% annual return)
    sigma=0.20    # Volatility (20%)
)
```

**Key property**: $d\theta_t = \mu \theta_t dt + \sigma \theta_t dB_t$

Solution: $\theta_t = \theta_0 \exp((\mu - \sigma^2/2)t + \sigma B_t)$

**When to use**:
- Finance: Stock prices, indices
- Growth models: GDP, population

### 4. Jump Diffusion Process

**Rare events** - combines diffusion with jumps

```python
from mfgarchon.core.stochastic import JumpDiffusionProcess

# Market with crashes
market = JumpDiffusionProcess(
    mu=0.06,          # Continuous drift
    sigma=0.15,       # Continuous volatility
    jump_intensity=2.0,  # 2 jumps per unit time (average)
    jump_mean=-0.1,   # Average jump size (crash)
    jump_std=0.05     # Jump size volatility
)
```

**Key property**: $d\theta_t = \mu dt + \sigma dB_t + J_t dN_t$

where $N_t$ is Poisson process, $J_t$ are jump sizes.

**When to use**:
- Finance: Market crashes, regime changes
- Insurance: Catastrophic events
- Operations: System failures

---

## Problem Specification

### Conditional Hamiltonian

The Hamiltonian defines how agents optimize given the noise:

```python
def hamiltonian(x, p, m, theta):
    """
    Conditional Hamiltonian H(x, p, m, θ).

    Args:
        x: State (float or array)
        p: Co-state / momentum (same shape as x)
        m: Population density (same shape as x)
        theta: Common noise value (scalar)

    Returns:
        Hamiltonian value (same shape as x)
    """
    # Control cost (depends on noise)
    control_cost = 0.5 * (1.0 + 0.1 * theta) * p**2

    # State cost
    state_cost = 0.5 * x**2

    # Congestion cost (mean field interaction)
    congestion = 0.2 * m

    return control_cost + state_cost + congestion
```

**Design Guidelines**:
1. **Smooth in θ**: Hamiltonian should vary smoothly with noise
2. **Convex in p**: Required for well-posedness (typically)
3. **Bounded**: Avoid singularities for numerical stability

### Initial and Terminal Conditions

```python
import numpy as np

# Initial density (Gaussian)
def initial_density(x):
    return np.exp(-x**2 / 0.5) / np.sqrt(np.pi * 0.5)

# Terminal cost (may depend on noise)
def terminal_cost(x, theta):
    """g(x, θ_T) - terminal cost at final time."""
    base_cost = 0.5 * x**2
    noise_adjustment = 0.1 * theta * x
    return base_cost + noise_adjustment

# Set in problem
problem.rho0 = initial_density(x_grid)
problem.terminal_cost = terminal_cost
```

---

## Solver Configuration

### Basic Configuration

```python
solver = CommonNoiseMFGSolver(
    problem,
    num_noise_samples=100,      # Number of Monte Carlo samples K
    variance_reduction=True,     # Use quasi-Monte Carlo (Sobol)
    parallel=True,               # Solve samples in parallel
    num_workers=None,            # Auto-detect CPU count
    seed=42                      # Reproducibility
)
```

### Advanced: Custom Conditional Solver

For each noise realization, a conditional MFG is solved. Customize this:

```python
from mfgarchon.factory import create_solver

def my_solver_factory(conditional_problem):
    """Create solver for each conditional MFG."""
    return create_solver(
        conditional_problem,
        solver_type="weno",         # High-order spatial discretization
        method="tvd_rk3",           # Time integration
        backend="jax",              # GPU acceleration
        device="gpu",
        weno_variant="weno5"        # 5th order WENO
    )

solver = CommonNoiseMFGSolver(
    problem,
    num_noise_samples=100,
    conditional_solver_factory=my_solver_factory
)
```

### Monte Carlo Configuration

```python
from mfgarchon.utils.numerical.monte_carlo import MCConfig

mc_config = MCConfig(
    num_samples=100,
    sampling_method="sobol",        # or "halton", "uniform"
    use_control_variates=False,     # Advanced variance reduction
    use_antithetic_variables=False, # For symmetric problems
    seed=42
)

solver = CommonNoiseMFGSolver(
    problem,
    mc_config=mc_config
)
```

---

## Result Analysis

### Basic Results

```python
result = solver.solve()

# Mean solutions (averaged over noise realizations)
u_mean = result.u_mean  # Shape: (Nt+1, *Nx_points)
m_mean = result.m_mean  # Shape: (Nt+1, *Nx_points)

# Uncertainty quantification
u_std = result.u_std    # Standard deviation
m_std = result.m_std

# Individual realizations
u_samples = result.u_samples  # List of K solutions
noise_paths = result.noise_paths  # List of K noise paths

# Statistics
mc_error = result.mc_error_u
variance_reduction_factor = result.variance_reduction_factor
print(f"Effective sample size: {variance_reduction_factor * len(u_samples):.0f}")
```

### Confidence Intervals

```python
# 95% confidence interval for value function
lower, upper = result.get_confidence_interval_u(confidence=0.95)

# At specific time and space
t_idx, x_idx = 10, 25
ci_width = upper[t_idx, x_idx] - lower[t_idx, x_idx]
print(f"95% CI width at (t={t_idx}, x={x_idx}): {ci_width:.4f}")
```

### Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Plot mean solution with confidence bands
t_idx = -1  # Terminal time
x = np.linspace(problem.xmin, problem.xmax, problem.Nx)

plt.figure(figsize=(10, 6))
plt.plot(x, u_mean[t_idx], 'b-', linewidth=2, label='Mean u(T,x)')
plt.fill_between(x, lower[t_idx], upper[t_idx],
                 alpha=0.3, label='95% CI')
plt.xlabel('State x')
plt.ylabel('Value function u(T,x)')
plt.legend()
plt.title('Common Noise MFG: Terminal Value Function')
plt.grid(True, alpha=0.3)
plt.show()

# Plot sample paths of noise
plt.figure(figsize=(10, 6))
t_grid = np.linspace(0, problem.T, problem.Nt + 1)
for i, path in enumerate(noise_paths[:10]):  # Plot first 10
    plt.plot(t_grid, path, alpha=0.5)
plt.xlabel('Time t')
plt.ylabel('Common noise θ(t)')
plt.title('Sample Paths of Common Noise Process')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Performance Tuning

### Choosing Number of Samples

**Trade-off**: More samples → better accuracy but longer computation

```python
# Quick exploratory run
solver_quick = CommonNoiseMFGSolver(problem, num_noise_samples=20)
result_quick = solver_quick.solve()

# Production run
solver_prod = CommonNoiseMFGSolver(problem, num_noise_samples=100)
result_prod = solver_prod.solve()

# Compare errors
print(f"Quick MC error: {result_quick.mc_error_u:.6f}")
print(f"Prod MC error: {result_prod.mc_error_u:.6f}")
print(f"Error reduction: {result_quick.mc_error_u / result_prod.mc_error_u:.2f}x")
# Theoretical: sqrt(100/20) ≈ 2.24x
```

**Rule of Thumb**:
- **Exploratory**: K = 20-50
- **Production**: K = 100-200
- **Publication**: K = 200-500 with convergence study

### Parallel Efficiency

```python
import time

# Measure speedup
for num_workers in [1, 2, 4, 8]:
    solver = CommonNoiseMFGSolver(
        problem,
        num_noise_samples=40,  # Use multiple of num_workers
        parallel=(num_workers > 1),
        num_workers=num_workers
    )

    start = time.time()
    result = solver.solve(verbose=False)
    elapsed = time.time() - start

    speedup = (elapsed_1worker / elapsed) if num_workers > 1 else 1.0
    efficiency = speedup / num_workers if num_workers > 1 else 1.0

    print(f"Workers: {num_workers}, Time: {elapsed:.1f}s, "
          f"Speedup: {speedup:.2f}x, Efficiency: {efficiency:.0%}")
```

**Expected Output**:
```
Workers: 1, Time: 60.0s, Speedup: 1.00x, Efficiency: 100%
Workers: 2, Time: 31.5s, Speedup: 1.90x, Efficiency: 95%
Workers: 4, Time: 16.2s, Speedup: 3.70x, Efficiency: 93%
Workers: 8, Time: 8.5s, Speedup: 7.06x, Efficiency: 88%
```

### Memory Management

For large problems, store only mean instead of all samples:

```python
# Standard (stores all K samples)
result_full = solver.solve()  # Memory: O(K * Nx * Nt)

# Custom aggregation (reduce memory)
solver._solve = lambda: custom_solve_with_streaming()
# Only accumulate running mean, not full samples
```

---

## Common Patterns and Recipes

### Pattern 1: Market with Stochastic Volatility

```python
from mfgarchon.core.stochastic import OrnsteinUhlenbeckProcess, StochasticMFGProblem
from mfgarchon.alg.numerical.stochastic import CommonNoiseMFGSolver

# VIX-like volatility
vix = OrnsteinUhlenbeckProcess(kappa=2.0, mu=20.0, sigma=8.0)

def trading_hamiltonian(x, p, m, theta):
    """Trading cost increases with volatility."""
    vol_factor = theta / 20.0  # Normalize around mean
    trading_cost = 0.5 * vol_factor * p**2
    position_cost = 0.5 * x**2
    market_impact = 0.1 * m  # Congestion
    return trading_cost + position_cost + market_impact

problem = StochasticMFGProblem(
    xmin=-3.0, xmax=3.0, Nx=101,
    T=1.0, Nt=51,
    noise_process=vix,
    conditional_hamiltonian=trading_hamiltonian
)

solver = CommonNoiseMFGSolver(problem, num_noise_samples=100)
result = solver.solve()
```

### Pattern 2: Epidemic with Random Policy

```python
from mfgarchon.core.stochastic import JumpDiffusionProcess

# Policy strictness (with sudden changes)
policy = JumpDiffusionProcess(
    mu=0.0, sigma=0.1,
    jump_intensity=3.0,    # 3 policy changes per unit time
    jump_mean=0.5,         # Sudden restrictions
    jump_std=0.2
)

def epidemic_hamiltonian(x, p, m, theta):
    """Infection dynamics affected by policy."""
    # Infection rate decreases with stricter policy
    infection_rate = 0.5 * np.exp(-0.5 * theta)
    recovery_rate = 0.2

    # Agents choose isolation effort (p = effort)
    isolation_cost = 0.5 * p**2
    infection_cost = infection_rate * (1 - p) * m

    return isolation_cost + infection_cost

# Problem setup similar to above
```

### Pattern 3: Sensitivity Analysis

```python
# Study sensitivity to noise strength
noise_levels = [0.05, 0.10, 0.20, 0.40]
results = {}

for sigma in noise_levels:
    noise = OrnsteinUhlenbeckProcess(kappa=2.0, mu=0.0, sigma=sigma)

    problem = StochasticMFGProblem(
        xmin=-2.0, xmax=2.0, Nx=51,
        T=1.0, Nt=26,
        noise_process=noise,
        conditional_hamiltonian=hamiltonian
    )

    solver = CommonNoiseMFGSolver(problem, num_noise_samples=50, seed=42)
    results[sigma] = solver.solve()

# Compare impact of noise
for sigma, result in results.items():
    u_variance = np.mean(result.u_std**2)
    print(f"σ={sigma:.2f}: Var(u) = {u_variance:.6f}")
```

---

## Troubleshooting

### Issue: High Monte Carlo Error

**Symptom**: `mc_error_u` is large (> 0.01)

**Solutions**:
1. Increase `num_noise_samples`
2. Enable `variance_reduction=True`
3. Check if noise variance is too large
4. Use control variates if analytical solution known

```python
# Diagnose
result = solver.solve()
if result.mc_error_u > 0.01:
    print("High MC error detected!")
    print(f"  Current K: {solver.K}")
    print(f"  Recommended K: {int((result.mc_error_u / 0.001)**2 * solver.K)}")
    print(f"  Variance reduction enabled: {solver.variance_reduction}")
```

### Issue: Conditional Solver Fails

**Symptom**: `result.converged == False`

**Solutions**:
1. Check individual noise paths
2. Simplify conditional Hamiltonian
3. Increase grid resolution
4. Use more robust conditional solver

```python
# Debug single noise path
noise_path = solver._sample_noise_paths()[0]
print(f"Noise path statistics:")
print(f"  Min: {noise_path.min():.3f}")
print(f"  Max: {noise_path.max():.3f}")
print(f"  Mean: {noise_path.mean():.3f}")

# Check if extreme values cause issues
if abs(noise_path).max() > 10:
    print("Warning: Noise path has extreme values!")
    print("Consider: (1) reducing σ, (2) checking process parameters")
```

### Issue: Slow Computation

**Symptom**: Solver takes too long

**Solutions**:
1. Enable parallel execution
2. Reduce `num_noise_samples` for initial tests
3. Use coarser spatial grid
4. Switch to GPU backend for conditional solver

```python
# Profile
import time

components = {}

start = time.time()
noise_paths = solver._sample_noise_paths()
components['sampling'] = time.time() - start

start = time.time()
# Single conditional solve
u, m, _ = solver._solve_conditional_mfg(noise_paths[0])
components['single_solve'] = time.time() - start

print(f"Time breakdown:")
print(f"  Noise sampling: {components['sampling']:.2f}s")
print(f"  Single conditional MFG: {components['single_solve']:.2f}s")
print(f"  Estimated total (K={solver.K}): "
      f"{components['sampling'] + solver.K * components['single_solve']:.1f}s")
```

---

## Best Practices Summary

### ✅ Do

- Start with small `num_noise_samples` (K=20) for exploration
- Always enable `variance_reduction=True` for production
- Use `parallel=True` for K > 10
- Set `seed` for reproducibility
- Check `result.converged` before trusting results
- Visualize noise paths to understand problem

### ❌ Don't

- Use K < 20 for publication results
- Ignore Monte Carlo error estimates
- Mix noise processes without understanding
- Set noise variance too large (check sample paths)
- Forget to normalize initial density

---

## Next Steps

- **Theory**: Read [High Dimensional MFG](../theory/high_dimensional_mfg.md)
- **Examples**: See `examples/advanced/` for advanced examples
- **Advanced**: Explore Master Equation formulation (planned for future)

---

**Document Version**: 1.0
**Last Updated**: October 2025
**Feedback**: Report issues at [GitHub Issues](https://github.com/anthropics/mfgarchon/issues)

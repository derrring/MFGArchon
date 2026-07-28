# Tutorial: Obstacle Problems and Variational Inequalities

**Tutorial Level**: Advanced
**Prerequisites**: Basic MFG concepts, familiarity with HJB equations
**Estimated Time**: 45-60 minutes
**Version**: 1.0 (2026-01-18)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Background](#mathematical-background)
3. [Tutorial 1: Basic 1D Obstacle Problem](#tutorial-1-basic-1d-obstacle-problem)
4. [Tutorial 2: Capacity-Constrained MFG](#tutorial-2-capacity-constrained-mfg)
5. [Solver Strategies](#solver-strategies)
6. [Convergence Diagnostics](#convergence-diagnostics)
7. [Common Issues and Solutions](#common-issues-and-solutions)
8. [Advanced Topics](#advanced-topics)

---

## Introduction

This tutorial teaches you how to solve **obstacle problems** and **variational inequalities** using MFGArchon. These problems arise when solutions must satisfy inequality constraints (e.g., $u(t,x) \geq \psi(x)$) rather than just equalities.

### What You'll Learn

- Set up obstacle problems with `ObstacleConstraint`
- Understand active sets and complementarity conditions
- Solve capacity-constrained Mean Field Games
- Choose between penalty and projection methods
- Diagnose convergence issues

### Real-World Applications

- **Capacity constraints**: Pedestrian flow in corridors (density ≤ max capacity)
- **Borrowing constraints**: Portfolio optimization with debt limits
- **State constraints**: Control problems with safety requirements
- **Free boundaries**: Stefan problems, American options

---

## Mathematical Background

### Variational Inequalities (VI)

A **variational inequality** finds $u \in K$ such that:

```
⟨F(u), v - u⟩ ≥ 0    for all v ∈ K
```

where $K$ is a **convex constraint set** (e.g., $K = \{u : u \geq \psi\}$).

**Equivalence to Projection**: The solution satisfies $u = P_K(u - \rho F(u))$ for any $\rho > 0$, where $P_K$ is the projection onto $K$.

### HJB Obstacle Problem

The HJB equation with obstacle constraint:

```
- ∂u/∂t + H(x, m, ∇u) - (σ²/2)Δu = 0    where u ≥ ψ
```

becomes a **complementarity problem**:

```
- ∂u/∂t + H - (σ²/2)Δu ≥ 0
u ≥ ψ
(u - ψ) · (- ∂u/∂t + H - (σ²/2)Δu) = 0    (complementarity)
```

**Active Set**: Region where $u = \psi$ (constraint is active)
**Inactive Set**: Region where $u > \psi$ (constraint is inactive)

### Capacity-Constrained MFG

For crowd dynamics, we want density $m(t,x)$ to satisfy $m \leq m_{\max}$ (capacity constraint). This is enforced through:

1. **Direct projection**: $m \leftarrow \min(m, m_{\max})$ after each FP step
2. **Penalty method**: Add congestion cost $\gamma \cdot g(m/C)$ to Hamiltonian, where $g(\rho) \to \infty$ as $\rho \to 1$

---

## Tutorial 1: Basic 1D Obstacle Problem

### Problem Statement

We solve the HJB equation with a **parabolic obstacle**:

```
- ∂u/∂t + (1/2)|∇u|² + L(x) - (σ²/2)u_xx = 0    where u ≥ ψ(x)

Domain: x ∈ [0, 1], t ∈ [0, 1]
Obstacle: ψ(x) = -κ(x - 0.5)²    (parabolic floor, κ = 0.5)
Terminal: u(T, x) = (x - 0.5)²
Running cost: L(x) = (1/2)(x - 0.5)²
Boundary: Neumann (∂u/∂n = 0)
```

**Physical Interpretation**: The obstacle creates a "floor" on the value function, preventing it from becoming too negative near the center.

### Step 1: Set Up Geometry and Problem

```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import ObstacleConstraint, neumann_bc

# Parameters
x_min, x_max = 0.0, 1.0
Nx = 100
T = 1.0
Nt = 50
sigma = 0.1  # Diffusion
kappa = 0.5  # Obstacle strength

# Create grid and boundary conditions
grid = TensorProductGrid(dimension=1, bounds=[(x_min, x_max)], Nx=[Nx])
bc = neumann_bc(dimension=1)

# Define costs
def running_cost(x_coords, alpha=None):
    """Running cost L(x) = (1/2)(x - 0.5)²."""
    return 0.5 * (x_coords[0] - 0.5) ** 2

def terminal_cost(x_coords):
    """Terminal cost g(x) = (x - 0.5)²."""
    return (x_coords[0] - 0.5) ** 2

# Create problem
problem = MFGProblem(
    geometry=grid,
    T=T,
    Nt=Nt,
    diffusion=sigma,
    bc=bc,
    running_cost=running_cost,
    terminal_cost=terminal_cost,
)

# Get actual grid coordinates
x = grid.coordinates[0]
```

### Step 2: Define Obstacle Function

```python
def obstacle_function(x: np.ndarray, kappa: float = 0.5) -> np.ndarray:
    """
    Parabolic obstacle: ψ(x) = -κ(x - 0.5)².

    Higher κ → stronger floor (more active constraint)
    """
    return -kappa * (x - 0.5) ** 2

# Compute obstacle values on grid
psi = obstacle_function(x, kappa=kappa)

print(f"Obstacle range: [{psi.min():.4f}, {psi.max():.4f}]")
print(f"Obstacle is parabolic with minimum at boundaries")
```

**★ Insight ─────────────────────────────────────**
The obstacle ψ(x) = -κ(x - 0.5)² is most negative at the boundaries (x=0, x=1) and least negative (maximum) at the center (x=0.5). This creates a constraint that is more restrictive away from the center.
**─────────────────────────────────────────────────**

### Step 3: Create Obstacle Constraint

```python
# Create lower obstacle constraint: u ≥ ψ
obstacle = ObstacleConstraint(lower_bound=psi, upper_bound=None)

print("\nObstacle Constraint:")
print(f"  Type: Lower obstacle (u ≥ ψ)")
print(f"  Method: Projection (default)")
```

**Constraint Types**:
- `lower_bound=psi, upper_bound=None` → Lower obstacle ($u \geq \psi$)
- `lower_bound=None, upper_bound=ψ_upper` → Upper obstacle ($u \leq \psi_{\text{upper}}$)
- Both specified → Bilateral obstacle ($\psi_{\text{lower}} \leq u \leq \psi_{\text{upper}}$)

### Step 4: Solve with Obstacle Constraint

```python
# Create solver with obstacle constraint
solver = HJBFDMSolver(
    problem=problem,
    constraint=obstacle,  # Pass constraint here
    tolerance=1e-6,
    max_iterations=100,
)

# Solve backward in time
print("\nSolving HJB with obstacle constraint...")
result = solver.solve()

print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Final error: {result.error_history_U[-1]:.2e}")
```

**What Happens Internally**:
1. Solver computes unconstrained update: $u^* = u^n - \Delta t \cdot (\text{HJB operator})$
2. Projects onto constraint set: $u^{n+1} = \max(u^*, \psi)$ (for lower obstacle)
3. Repeats until convergence

### Step 5: Analyze Active Set

The **active set** is where the constraint is binding ($u = \psi$).

```python
# Extract solution
U = result.U  # Shape: (Nt+1, Nx)
u_final = U[0, :]  # Solution at t=0 (backward time)

# Compute active set (where u ≈ ψ within tolerance)
active_set = np.abs(u_final - psi) < 1e-3

print(f"\nActive Set Analysis:")
print(f"  Active points: {active_set.sum()} / {Nx} ({100*active_set.mean():.1f}%)")
print(f"  Active region: x ∈ [{x[active_set].min():.3f}, {x[active_set].max():.3f}]")
```

**Interpretation**:
- **Large active set**: Constraint is strongly binding (increase κ or T)
- **Small active set**: Constraint is weak (obstacle barely affects solution)
- **No active set**: Constraint is never active (remove obstacle or adjust parameters)

### Step 6: Visualize Results

```python
# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("1D Obstacle Problem - Variational Inequality", fontsize=14, fontweight="bold")

# (a) Solution u(t,x) over time
time_indices = [0, Nt//4, Nt//2, 3*Nt//4, Nt]
t_values = np.linspace(0, T, Nt+1)

for idx in time_indices:
    axes[0, 0].plot(x, U[Nt - idx, :], label=f"t={t_values[idx]:.2f}")
axes[0, 0].plot(x, psi, 'k--', linewidth=2, label="Obstacle ψ(x)")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("u(t, x)")
axes[0, 0].set_title("(a) Solution Evolution (Backward Time)")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# (b) Final solution vs obstacle
axes[0, 1].plot(x, u_final, 'b-', linewidth=2, label="u(0, x)")
axes[0, 1].plot(x, psi, 'k--', linewidth=2, label="Obstacle ψ(x)")
axes[0, 1].fill_between(x, psi, u_final, alpha=0.3, label="Inactive set (u > ψ)")
axes[0, 1].axhline(0, color='gray', linestyle=':', linewidth=1)
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("u(0, x)")
axes[0, 1].set_title("(b) Final Solution vs Obstacle")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# (c) Active set over time
active_sets_over_time = np.zeros((Nt+1, Nx), dtype=bool)
for n in range(Nt+1):
    active_sets_over_time[n, :] = np.abs(U[n, :] - psi) < 1e-3

im = axes[1, 0].imshow(
    active_sets_over_time.T,
    aspect='auto',
    origin='lower',
    extent=[0, T, x_min, x_max],
    cmap='RdYlGn_r',
    interpolation='nearest'
)
axes[1, 0].set_xlabel("Time t")
axes[1, 0].set_ylabel("Space x")
axes[1, 0].set_title("(c) Active Set Evolution (Yellow = Active)")
plt.colorbar(im, ax=axes[1, 0], label="Active (u = ψ)")

# (d) Convergence history
axes[1, 1].semilogy(result.error_history_U, 'b-', linewidth=2)
axes[1, 1].axhline(solver.tolerance, color='r', linestyle='--', label=f"Tolerance = {solver.tolerance:.0e}")
axes[1, 1].set_xlabel("Iteration")
axes[1, 1].set_ylabel("Error ||u^{k+1} - u^k||")
axes[1, 1].set_title("(d) Convergence History")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Expected Observations**:
1. **Panel (a)**: Solution starts at terminal condition and increases backward in time
2. **Panel (b)**: Solution respects obstacle (never dips below ψ)
3. **Panel (c)**: Active set may grow going backward in time
4. **Panel (d)**: Convergence should be monotonic and reach tolerance

### Key Takeaways from Tutorial 1

✅ **One-line projection**: `obstacle.project(u)` enforces constraint
✅ **Active set tracking**: Identify where constraint binds
✅ **Solver integration**: Pass `constraint=` to HJB solver
✅ **Complementarity**: At active points, HJB residual ≥ 0; elsewhere, u > ψ

---

## Tutorial 2: Capacity-Constrained MFG

### Problem Statement

We solve a **Mean Field Game** where agent density is constrained by corridor capacity:

```
HJB: -∂u/∂t + (1/2)|∇u|² + γ·g(m/C) - (σ²/2)Δu = 0
FP:   ∂m/∂t - σ²Δm - div(m·∇u) = 0    with m ≤ C(x)
```

where:
- $C(x)$: **Capacity field** (from corridor geometry)
- $g(\rho)$: **Congestion cost** ($\rho = m/C$, congestion ratio)
- $\gamma$: Congestion weight (trade-off between travel time and crowding)

**Physical Setup**: Agents navigate a maze, avoiding overcrowded corridors.

### Step 1: Generate Maze and Capacity Field

```python
from mfgarchon.geometry.graph import MazeConfig, MazeGeometry
from examples.advanced.capacity_constrained_mfg import CapacityField

# Generate maze
config = MazeConfig(rows=20, cols=20, seed=42)
maze = MazeGeometry.generate(config)
maze_array = maze.to_numpy_array(wall_thickness=3)

print(f"Maze shape: {maze_array.shape}")
print(f"Passages: {100 * np.mean(maze_array == 0):.1f}%")
print(f"Walls: {100 * np.mean(maze_array == 1):.1f}%")

# Compute capacity from corridor width
capacity = CapacityField.from_maze_geometry(
    maze_array,
    wall_thickness=3,
    epsilon=1e-3,         # Minimum capacity in walls
    normalization="max",  # Normalize to [epsilon, 1]
    decay_factor=0.5,     # Smooth falloff near walls
)

print(f"Capacity range: [{capacity.min_capacity:.3f}, {capacity.max_capacity:.3f}]")
print(f"Mean capacity: {capacity.mean_capacity:.3f}")
```

**Capacity Computation**:
1. Compute **distance transform** from walls: $d(x)$ = distance to nearest wall
2. Convert to capacity: $C(x) = \epsilon + (1 - \epsilon) \cdot \tanh(\alpha \cdot d(x))$
3. Wide corridors → high capacity; narrow corridors → low capacity

### Step 2: Choose Congestion Model

MFGArchon provides three congestion models:

| Model | Formula | Derivative | Best For |
|:------|:--------|:-----------|:---------|
| **Quadratic** | $g(\rho) = \frac{1}{2}\rho^2$ | $g'(\rho) = \rho$ | Moderate congestion, stable |
| **Exponential** | $g(\rho) = e^{\beta\rho} - 1$ | $g'(\rho) = \beta e^{\beta\rho}$ | Strong congestion avoidance |
| **LogBarrier** | $g(\rho) = -\log(1 - \rho/\theta)$ | $g'(\rho) = \frac{1}{(\theta - \rho)}$ | Hard capacity limit (ρ → θ) |

```python
from examples.advanced.capacity_constrained_mfg import create_congestion_model

# Choose congestion model
congestion_type = "quadratic"  # Options: "quadratic", "exponential", "logbarrier"
congestion_model = create_congestion_model(congestion_type)

print(f"Congestion model: {congestion_type}")
print(f"  g(0.5) = {congestion_model.cost(np.array([0.5]), np.array([1.0]))[0]:.4f}")
print(f"  g'(0.5) = {congestion_model.derivative(np.array([0.5]), np.array([1.0]))[0]:.4f}")
```

**★ Insight ─────────────────────────────────────**
The **quadratic model** $g(\rho) = \frac{1}{2}\rho^2$ is the most stable choice for beginners. It provides smooth congestion penalties without singularities. Use **logbarrier** for hard capacity limits, but be prepared for potential numerical instabilities near $\rho \to \theta$.
**─────────────────────────────────────────────────**

### Step 3: Create Capacity-Constrained Problem

```python
from examples.advanced.capacity_constrained_mfg import CapacityConstrainedMFGProblem

# Create problem
problem = CapacityConstrainedMFGProblem(
    capacity_field=capacity,
    congestion_model=congestion_model,
    congestion_weight=1.0,        # γ: trade-off parameter
    spatial_bounds=[(0, 1), (0, 1)],
    spatial_discretization=[63, 63],
    T=1.0,
    Nt=50,
    diffusion=0.01,
)

print(f"Problem dimension: {problem.dimension}")
print(f"Grid: {problem.spatial_discretization}")
print(f"Congestion weight γ: {problem.congestion_weight}")
```

**Parameter Selection**:
- **γ = 0**: No congestion (free flow, standard MFG)
- **γ ~ 0.1-1**: Moderate congestion avoidance (realistic)
- **γ > 10**: Strong congestion avoidance (near hard constraint)

### Step 4: Solve MFG System

```python
# TODO: Full solver integration pending
# For now, this demonstrates the API pattern

# Solve the coupled HJB-FP system. The problem's Hamiltonian, including the
# congestion term, is what the solve reads -- there is nothing to configure here.
result = problem.solve()

# Extract solution
U = result.U  # Value function
m = result.m  # Density
```

**What Happens During Solve**:
1. **Picard iteration**: Alternate between HJB (with fixed m) and FP (with fixed u)
2. **HJB step**: Solve $-\partial u/\partial t + H(x, m, \nabla u) = 0$ using congestion Hamiltonian
3. **FP step**: Solve $\partial m/\partial t - \sigma^2 \Delta m - \text{div}(m \nabla u) = 0$
4. **Projection** (optional): Apply $m \leftarrow \min(m, C)$ after FP step
5. **Convergence check**: $\|u^{k+1} - u^k\| < \text{tol}$ and $\|m^{k+1} - m^k\| < \text{tol}$

### Step 5: Analyze Congestion

```python
# Interpolate capacity to match solution grid
Nx, Ny = problem.spatial_discretization
positions = np.array([[i, j] for i in range(Nx) for j in range(Ny)])
C_grid = capacity.interpolate_at_positions(positions).reshape(Nx, Ny)

# Compute congestion ratio ρ(x) = m(x) / C(x)
congestion_ratio = m[0, :, :] / (C_grid + 1e-6)  # At t=0

print(f"Congestion ratio statistics:")
print(f"  Mean ρ: {congestion_ratio.mean():.3f}")
print(f"  Max ρ: {congestion_ratio.max():.3f}")
print(f"  Overcapacity (ρ > 1): {100 * np.mean(congestion_ratio > 1):.1f}%")
```

**Congestion Ratio Interpretation**:
- **ρ < 0.5**: Free flow (low density)
- **0.5 ≤ ρ < 1**: Congested but below capacity
- **ρ ≥ 1**: Overcapacity (density exceeds local capacity)

### Step 6: Visualize Capacity and Congestion

```python
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Capacity-Constrained MFG: Maze Navigation", fontsize=16)

# Row 1: Geometry and Capacity
# (a) Maze
axes[0, 0].imshow(maze_array.T, origin='lower', cmap='binary')
axes[0, 0].set_title("(a) Maze Geometry")

# (b) Capacity field
im1 = axes[0, 1].imshow(capacity.capacity.T, origin='lower', cmap='viridis')
axes[0, 1].set_title("(b) Capacity Field C(x)")
plt.colorbar(im1, ax=axes[0, 1], label="Corridor Capacity")

# (c) Capacity histogram
axes[0, 2].hist(capacity.capacity.flatten(), bins=50, alpha=0.7, edgecolor='black')
axes[0, 2].set_title("(c) Capacity Distribution")
axes[0, 2].set_xlabel("Capacity C(x)")

# Row 2: MFG Solution
# (d) Agent density
im2 = axes[1, 0].imshow(m[0, :, :].T, origin='lower', cmap='hot')
axes[1, 0].set_title("(d) Agent Density m(0,x)")
plt.colorbar(im2, ax=axes[1, 0], label="Density")

# (e) Value function
im3 = axes[1, 1].imshow(U[0, :, :].T, origin='lower', cmap='coolwarm')
axes[1, 1].set_title("(e) Value Function u(0,x)")
plt.colorbar(im3, ax=axes[1, 1], label="Value")

# (f) Congestion ratio
im4 = axes[1, 2].imshow(congestion_ratio.T, origin='lower', cmap='RdYlGn_r', vmin=0, vmax=1.5)
axes[1, 2].set_title("(f) Congestion Ratio ρ=m/C")
cbar = plt.colorbar(im4, ax=axes[1, 2], label="Congestion Ratio")
cbar.ax.axhline(1.0, color='black', linestyle='--', linewidth=2, label="Capacity limit")

plt.tight_layout()
plt.show()
```

**Expected Observations**:
- **Panel (b)**: Wide corridors have high capacity, narrow corridors low
- **Panel (d)**: Density concentrated in wide corridors (high capacity)
- **Panel (f)**: Congestion ratio mostly < 1 (agents avoid overcrowding)

### Key Takeaways from Tutorial 2

✅ **Capacity from geometry**: Distance transform → corridor capacity
✅ **Congestion cost**: Soft penalty $\gamma \cdot g(m/C)$ in Hamiltonian
✅ **Congestion ratio**: $\rho = m/C$ measures crowding level
✅ **Model selection**: Quadratic (stable), LogBarrier (hard limits)

---

## Solver Strategies

### Projection Method (Default)

**How it works**: After computing unconstrained update, project onto constraint set.

```python
# For HJB obstacle: u ≥ ψ
u_new = max(u_unconstrained, psi)  # Element-wise maximum

# For FP capacity: m ≤ m_max
m_new = min(m_unconstrained, m_max)  # Element-wise minimum
```

**Advantages**:
- ✅ Simple and robust
- ✅ One-line implementation
- ✅ No additional parameters
- ✅ Guaranteed to satisfy constraint exactly

**Disadvantages**:
- ⚠️ May slow convergence (constraint discontinuity)
- ⚠️ Active set changes abruptly

**When to use**: First choice for obstacle problems, bilateral constraints, capacity limits.

### Penalty Method

**How it works**: Add penalty term $\frac{1}{2\epsilon} \|\min(0, u - \psi)\|^2$ to objective.

```python
# Not directly exposed in current API
# Penalty is implicit in LogBarrier congestion model

# Example: LogBarrier congestion
congestion_model = LogBarrierCongestion(threshold=0.95)
# As m → 0.95·C, penalty → ∞
```

**Advantages**:
- ✅ Smooth (no discontinuities)
- ✅ Faster convergence (when $\epsilon$ is tuned)
- ✅ Works with standard solvers

**Disadvantages**:
- ⚠️ Requires tuning penalty parameter $\epsilon$
- ⚠️ Constraint satisfied only approximately: $\|u - P_K(u)\| = O(\sqrt{\epsilon})$
- ⚠️ Ill-conditioned for small $\epsilon$

**When to use**: When smoothness is important, or constraint is "soft" (mild violations acceptable).

### Comparison Table

| Feature | Projection | Penalty |
|:--------|:-----------|:--------|
| **Constraint satisfaction** | Exact | Approximate ($O(\sqrt{\epsilon})$) |
| **Smoothness** | Discontinuous at active set | Smooth |
| **Parameters** | None | Penalty parameter $\epsilon$ |
| **Convergence** | May slow near active set | Fast if $\epsilon$ well-tuned |
| **Implementation** | `obstacle.project(u)` | Implicit in congestion cost |
| **Recommended for** | Hard constraints, beginners | Soft constraints, smoothness needed |

**Decision Guide**:
- **Use projection** if: Constraint must be satisfied exactly (e.g., safety), no parameter tuning desired
- **Use penalty** if: Constraint is soft, smoothness is critical, experienced with parameter tuning

---

## Convergence Diagnostics

### 1. Monitor Error History

```python
# After solving
print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Final error: {result.error_history_U[-1]:.2e}")

# Plot convergence
plt.semilogy(result.error_history_U, label="HJB error")
plt.semilogy(result.error_history_m, label="FP error")
plt.axhline(solver.tolerance, color='r', linestyle='--', label="Tolerance")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.show()
```

**Expected Behavior**:
- **Linear convergence**: Error decreases exponentially (straight line on semilogy)
- **Stagnation**: Error plateaus above tolerance → reduce tolerance or increase iterations
- **Oscillation**: Error bounces → reduce relaxation parameter or check CFL condition

### 2. Check Active Set Stability

```python
# Compute active set at each iteration (if you stored iterates)
active_sets = []
for u_iter in u_history:
    active = np.abs(u_iter - psi) < 1e-3
    active_sets.append(active.sum())

# Plot active set evolution
plt.plot(active_sets)
plt.xlabel("Iteration")
plt.ylabel("Active Set Size")
plt.title("Active Set Stability")
plt.grid(True)
plt.show()
```

**Expected Behavior**:
- **Stable**: Active set size stabilizes after initial iterations
- **Oscillating**: Active set grows/shrinks repeatedly → constraint interferes with convergence
- **Empty**: No active set → constraint is never binding (remove or adjust obstacle strength)

### 3. Verify Complementarity

At convergence, the complementarity condition should hold:

```
(u - ψ) · (HJB residual) ≈ 0
```

```python
# Compute HJB residual (simplified for demonstration)
residual = -np.gradient(u_final) + 0.5 * np.gradient(u_final)**2 - running_cost(x)

# Check complementarity
complementarity = (u_final - psi) * residual

print(f"Complementarity violation: {np.abs(complementarity).max():.2e}")
print(f"Should be < {solver.tolerance:.0e}")

# Plot
plt.plot(x, complementarity)
plt.axhline(0, color='k', linestyle='--')
plt.xlabel("x")
plt.ylabel("(u - ψ) · residual")
plt.title("Complementarity Check")
plt.grid(True)
plt.show()
```

**Interpretation**:
- **Small violation** (< tolerance): Converged correctly
- **Large violation**: Either solver didn't converge or implementation error

### 4. Capacity-Constrained Diagnostics

For capacity-constrained MFG, check:

```python
# 1. Congestion ratio range
print(f"Congestion ratio: [{congestion_ratio.min():.3f}, {congestion_ratio.max():.3f}]")
assert congestion_ratio.max() < 2.0, "Excessive overcapacity (m >> C)"

# 2. Mass conservation (FP equation)
mass_initial = m[-1, :, :].sum() * dx * dy
mass_final = m[0, :, :].sum() * dx * dy
print(f"Mass conservation error: {abs(mass_final - mass_initial) / mass_initial:.2e}")

# 3. Hamiltonian consistency
# Compute H(x, m, ∇u) at all points, check for NaN/Inf
H_values = problem.hamiltonian(positions, m.ravel(), grad_u.ravel(), t=0)
print(f"Hamiltonian range: [{H_values.min():.3f}, {H_values.max():.3f}]")
assert np.all(np.isfinite(H_values)), "Hamiltonian contains NaN/Inf"
```

---

## Common Issues and Solutions

### Issue 1: Slow Convergence Near Active Set

**Symptom**: Convergence stalls when active set changes.

**Cause**: Projection creates discontinuity, slowing iterative solver.

**Solutions**:
1. **Increase tolerance**: Accept slightly looser convergence
   ```python
   solver = HJBFDMSolver(problem, constraint=obstacle, tolerance=1e-4)  # Instead of 1e-6
   ```
2. **Use penalty method**: Smooth out constraint
   ```python
   # Use LogBarrier congestion with small threshold
   congestion_model = LogBarrierCongestion(threshold=1.05)  # Soft capacity at 105%
   ```
3. **Adaptive active set**: Freeze active set after initial detection (advanced)

### Issue 2: Constraint Violation

**Symptom**: Solution violates constraint: $u < \psi$ somewhere.

**Cause**:
- Projection not applied correctly
- Numerical precision issues
- Tolerance too loose

**Solutions**:
1. **Verify projection is enabled**:
   ```python
   # Make sure constraint is passed to solver
   solver = HJBFDMSolver(problem, constraint=obstacle)  # NOT missing!
   ```
2. **Check projection correctness**:
   ```python
   # After each iteration, manually verify
   assert np.all(u >= psi - 1e-10), "Constraint violated!"
   ```
3. **Tighten tolerance**:
   ```python
   solver.tolerance = 1e-8  # Stricter convergence
   ```

### Issue 3: Mass Loss in Capacity-Constrained MFG

**Symptom**: Total mass $\int m \, dx$ decreases over time.

**Cause**: Capacity constraint projection $m \leftarrow \min(m, C)$ removes mass.

**Solutions**:
1. **Use penalty instead of projection**: Congestion cost prevents overcrowding without hard projection
   ```python
   # LogBarrier with high penalty weight
   problem = CapacityConstrainedMFGProblem(
       capacity_field=capacity,
       congestion_model=LogBarrierCongestion(threshold=0.95),
       congestion_weight=10.0,  # Strong penalty
   )
   ```
2. **Renormalize after projection** (if mass conservation is critical):
   ```python
   # After FP solve
   m_projected = np.minimum(m, capacity)
   m_projected *= (m.sum() / m_projected.sum())  # Rescale to preserve mass
   ```

### Issue 4: NaN/Inf in Hamiltonian

**Symptom**: Solver crashes with NaN or Inf values.

**Cause**:
- Congestion model evaluates $g(\rho)$ at $\rho \geq \theta$ (LogBarrier)
- Division by zero in capacity interpolation

**Solutions**:
1. **Regularize capacity**:
   ```python
   capacity = CapacityField.from_maze_geometry(
       maze_array,
       epsilon=1e-3,  # Minimum capacity (never zero)
   )
   ```
2. **Clip congestion ratio before evaluation**:
   ```python
   # In Hamiltonian computation
   rho = np.clip(m / C, 0, 0.99 * threshold)  # Stay below barrier threshold
   ```
3. **Switch congestion model**:
   ```python
   # Use quadratic instead of logbarrier for stability
   congestion_model = QuadraticCongestion()
   ```

---

## Advanced Topics

### 1. Bilateral Obstacles

Both upper and lower bounds: $\psi_{\text{lower}} \leq u \leq \psi_{\text{upper}}$

```python
obstacle = ObstacleConstraint(
    lower_bound=psi_lower,
    upper_bound=psi_upper,
)

# Projection: u = clip(u, psi_lower, psi_upper)
```

**Applications**: Portfolio optimization with borrowing/short-selling constraints.

### 2. Time-Dependent Obstacles

Obstacle changes over time: $u(t, x) \geq \psi(t, x)$

```python
# Create obstacle for each timestep
psi_t = lambda t, x: -kappa * (x - 0.5)**2 * (1 - t/T)  # Obstacle relaxes over time

# In solver, update constraint at each timestep
for n in range(Nt):
    psi_current = psi_t(t[n], x)
    obstacle.update_bounds(lower_bound=psi_current)
    u[n] = solver.solve_timestep(u[n+1], obstacle)
```

### 3. Regional Constraints

Obstacle only in part of domain: $u(t, x) \geq \psi(x)$ for $x \in \Omega_{\text{active}}$

```python
# Define obstacle only in active region
psi = np.full(Nx, -np.inf)  # No constraint by default
psi[x > 0.3] = -kappa * (x[x > 0.3] - 0.5)**2  # Constraint only for x > 0.3

obstacle = ObstacleConstraint(lower_bound=psi)
```

**Applications**: State constraints (speed limits in certain zones), safety regions.

### 4. Coupled Capacity and Value Constraints

Capacity depends on value function: $m \leq C(u)$

```python
# Nonlinear coupling (advanced)
for picard_iter in range(max_picard):
    # Solve HJB with current m
    U = solve_hjb(m)

    # Solve FP with capacity depending on U
    C_adaptive = base_capacity * (1 + alpha * U)  # Higher value → more capacity
    m = solve_fp(U)
    m = np.minimum(m, C_adaptive)  # Project onto adaptive capacity
```

**Applications**: Adaptive road widening, dynamic capacity allocation.

---

## Summary and Best Practices

### Quick Start Checklist

1. ✅ **Define obstacle/constraint**: Use `ObstacleConstraint` with `lower_bound` and/or `upper_bound`
2. ✅ **Choose solver method**: Projection (default) or Penalty (via congestion model)
3. ✅ **Pass constraint to solver**: `solver = HJBFDMSolver(problem, constraint=obstacle)`
4. ✅ **Solve**: `result = solver.solve()`
5. ✅ **Check convergence**: Verify `result.converged` and inspect `error_history`
6. ✅ **Analyze active set**: Compute `active_set = (u ≈ ψ)`
7. ✅ **Verify complementarity**: Check $(u - \psi) \cdot \text{residual} \approx 0$

### Best Practices

**DO**:
- ✅ Start with projection method (simple and robust)
- ✅ Visualize active set evolution to understand constraint behavior
- ✅ Use quadratic congestion for first attempts (most stable)
- ✅ Monitor mass conservation in capacity-constrained problems
- ✅ Check complementarity condition at convergence

**DON'T**:
- ❌ Use penalty method without tuning $\epsilon$ carefully
- ❌ Ignore mass loss warnings in capacity-constrained MFG
- ❌ Set LogBarrier threshold = 1.0 (numerical instability)
- ❌ Skip convergence diagnostics (plot error history!)
- ❌ Use overly tight tolerances without checking active set stability

### Further Reading

- **Examples**:
  - `examples/advanced/obstacle_problem_1d.py` (basic HJB obstacle)
  - `examples/advanced/capacity_constrained_mfg/example_maze_mfg.py` (capacity MFG)
- **API Reference**: `docs/user/advanced_boundary_conditions.md`

---

**Last Updated**: 2026-01-18
**Author**: MFGArchon Documentation Team
**Related Issues**: #591 (Variational Constraints), #594 (Documentation)

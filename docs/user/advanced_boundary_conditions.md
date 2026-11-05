# Advanced Boundary Conditions: User Guide

**Audience**: MFG practitioners and researchers
**Level**: Intermediate to Advanced
**Prerequisites**: Basic understanding of Mean Field Games, familiarity with standard BCs (Dirichlet, Neumann, Robin)

---

## Overview

This guide introduces **advanced boundary condition methods** implemented in MFGArchon v0.18.0+, covering:

- **Tier 2 BCs**: Variational inequalities and constraint-based boundary conditions
- **Tier 3 BCs**: Time-dependent boundaries and free boundary problems

**When to use these advanced methods**:
- Your problem involves **capacity constraints** (maximum density limits)
- You need **obstacle avoidance** (keep-out zones)
- Your domain **changes over time** (moving boundaries, melting interfaces)
- Standard Dirichlet/Neumann/Robin BCs are insufficient

---

## Quick Reference: Choosing the Right BC Method

| Problem Type | BC Method | Implementation | Example |
|:-------------|:----------|:---------------|:--------|
| Fixed boundary, fixed value | Dirichlet | Standard | Exit at $U = 0$ |
| Fixed boundary, fixed flux | Neumann | Standard | Reflecting wall |
| Fixed boundary, linear relation | Robin | Standard | Radiation BC |
| **Density cannot exceed limit** | **Obstacle constraint** | **Tier 2** | **Capacity-constrained crowd** |
| **Agents must stay in region** | **Obstacle constraint** | **Tier 2** | **Keep-out zones** |
| **Boundary moves with physics** | **Level Set method** | **Tier 3** | **Melting ice, expanding exit** |

---

## Part I: Tier 2 BCs - Variational Inequalities

### 1.1 What Are Variational Inequalities?

**Standard PDE**: Find $u$ such that $Lu = f$ with BC $Bu = g$.

**Variational Inequality (VI)**: Find $u \in K$ such that:
$$
\langle Lu - f, v - u \rangle \geq 0 \quad \forall v \in K
$$

where $K$ is a **constraint set** (e.g., $K = \{u : u \geq \psi\}$).

**Key Difference**: Solution must satisfy:
1. The PDE ($Lu = f$) **in regions where constraints are inactive**
2. The constraint ($u \in K$) **everywhere**

**Think of it as**: The PDE "wants" to go one way, but the constraint "pushes back" to keep the solution feasible.

### 1.2 Obstacle Problems

**Problem Setup**: Solve PDE with lower bound constraint:
$$
\begin{aligned}
-\Delta u &= f \quad \text{where } u > \psi \quad \text{(free region)} \\
u &\geq \psi \quad \text{everywhere} \quad \text{(obstacle constraint)} \\
u &= g \quad \text{on boundary}
\end{aligned}
$$

**Physical Examples**:
- **Membrane over obstacle**: Elastic membrane cannot penetrate rigid obstacle $\psi(x)$
- **Optimal stopping**: Value function bounded below by stopping payoff
- **Crowd dynamics**: Density cannot exceed physical limit

**Active vs Inactive Sets**:
- **Active set** $\mathcal{A}$: Where $u = \psi$ (constraint is binding)
- **Inactive set** $\mathcal{I}$: Where $u > \psi$ (constraint not binding, PDE holds)

### 1.3 When to Use Obstacle Constraints

✅ **Use obstacle constraints when**:
- You have **physical limits** on state variables (e.g., maximum density, minimum value)
- The constraint **can activate/deactivate** during the solution process
- You want **automatic detection** of where constraints bind

❌ **Don't use obstacle constraints when**:
- Constraint is always active everywhere (just impose as fixed BC)
- You know the active set a priori (use strong BC on that region)
- The constraint is nonlinear or non-convex (obstacle methods require convex constraints)

### 1.4 Implementation in MFGArchon

**Step 1**: Define the obstacle constraint
```python
from mfgarchon.geometry.boundary import ObstacleConstraint

# Example: Density cannot exceed m_max
obstacle = ObstacleConstraint(
    lower_bound=0.0,        # m ≥ 0 (non-negative)
    upper_bound=m_max,      # m ≤ m_max (capacity limit)
)
```

**Step 2**: Use projection in your solver loop
```python
# Standard MFG Picard iteration
for k in range(max_iterations):
    # 1. Solve HJB (backward)
    U = solve_hjb(geometry, m_prev, ...)

    # 2. Solve FP (forward)
    m_unconstrained = solve_fp(geometry, U, ...)

    # 3. Project onto constraint set (NEW!)
    m = obstacle.project(m_unconstrained)

    # 4. Check convergence
    if converged(U, U_prev):
        break

    U_prev, m_prev = U, m
```

**That's it!** The projection happens in **one line** and integrates seamlessly with existing solvers.

### 1.5 Capacity-Constrained MFG Example

**Scenario**: Crowd evacuating a corridor with capacity limit at exit.

**Code**:
```python
# See: examples/advanced/capacity_constrained_mfg_1d.py

from mfgarchon import MFGProblem, TensorProductGrid
from mfgarchon.geometry.boundary import ObstacleConstraint

# Setup
grid = TensorProductGrid(dimension=1, bounds=[(0, 1)], Nx_points=[101])
m_max = 0.5  # Maximum density (capacity)

# Define obstacle
constraint = ObstacleConstraint(lower_bound=0.0, upper_bound=m_max)

# Solve MFG with capacity constraint
problem = MFGProblem(
    geometry=grid,
    terminal_cost=lambda x: x[0],  # Distance to exit
    running_cost=lambda x, m: m,  # Congestion cost
)

# Picard iteration with projection
for iteration in range(max_picard):
    U = problem.solve_hjb(m=m_prev)
    m_unconstrained = problem.solve_fp(U=U)

    # Project to enforce capacity
    m = constraint.project(m_unconstrained)

    # Check convergence...
```

**Result**: Queue forms when density tries to exceed $m_{\max} = 0.5$ at the exit.

**Performance**: Projection overhead < 2% of total solve time (validated in Phase 2).

---

## Part II: Tier 3 BCs - Time-Dependent Boundaries

### 2.1 What Are Time-Dependent Boundaries?

**Standard BC**: Domain $\Omega$ is **fixed** throughout the simulation.

**Time-Dependent BC**: Domain $\Omega(t)$ **evolves** according to some law:
$$
\frac{\partial \Omega}{\partial t} = V_n(t, x) \quad \text{(interface velocity)}
$$

**Examples**:
- **Phase transitions**: Ice melting (Stefan problem)
- **Erosion**: Coastline changing due to waves
- **Adaptive geometry**: Exit expands when congested (MFG)
- **Free boundaries**: Interface position unknown a priori, determined by solution

### 2.2 Level Set Method

**Idea**: Represent moving boundary implicitly via **level set function** $\phi(t, x)$:
$$
\begin{aligned}
\Omega(t) &= \{x : \phi(t, x) < 0\} \quad \text{(interior)} \\
\partial\Omega(t) &= \{x : \phi(t, x) = 0\} \quad \text{(boundary)}
\end{aligned}
$$

**Evolution Equation**:
$$
\frac{\partial \phi}{\partial t} + V_n |\nabla \phi| = 0
$$

where $V_n$ is the **normal velocity** of the interface.

**Advantages**:
- Topology changes (splitting, merging) handled automatically
- Natural extension to any dimension
- No explicit tracking of interface points

### 2.3 When to Use Level Set Method

✅ **Use Level Set when**:
- Boundary **moves** during simulation
- Velocity $V_n$ is **determined by the physics** (not prescribed)
- Topology can change (e.g., obstacles merge, regions split)
- You want **automatic** interface tracking

❌ **Don't use Level Set when**:
- Domain is static (use standard geometry)
- Boundary motion is simple and known (use time-dependent coordinate transform)
- You need **exact** mass conservation (Level Set can drift; use Volume-of-Fluid instead)

### 2.4 Implementation in MFGArchon

**Step 1**: Set up time-dependent domain
```python
from mfgarchon.geometry.level_set import TimeDependentDomain
from mfgarchon.geometry import ImplicitDomain

# Initial interface (e.g., circle at origin with radius 0.5)
def phi_initial(X):
    return np.linalg.norm(X, axis=0) - 0.5

# Wrap in time-dependent container
td_domain = TimeDependentDomain(phi_initial(grid.X), geometry=grid)
```

**Step 2**: Time-stepping loop with geometry updates
```python
for n in range(num_timesteps):
    # 1. Get current static geometry
    geometry_t = td_domain.get_geometry_at_time(t[n])

    # 2. Solve PDE on current geometry (standard solver!)
    T = solve_heat_equation(geometry_t, ...)

    # 3. Compute interface velocity from physics
    # (Example: Stefan condition V_n = -κ·[∂T/∂n])
    velocity = compute_velocity_from_solution(T, geometry_t)

    # 4. Evolve interface
    td_domain.evolve_step(velocity, dt)

    # 5. Reinitialize (maintain numerical accuracy)
    if n % 10 == 0:
        td_domain.reinitialize()
```

**Key Insight**: Existing solvers work **unchanged** - geometry updates happen **between** timesteps.

### 2.5 Stefan Problem Example (1D Ice Melting)

**Physical Setup**: Ice at temperature $T < 0$ melting into water at $T > 0$.

**Governing Equations**:
$$
\begin{aligned}
\frac{\partial T}{\partial t} &= \alpha \Delta T \quad \text{(heat equation)} \\
V_n &= -\kappa \left[\frac{\partial T}{\partial n}\right] \quad \text{(Stefan condition: interface velocity)}
\end{aligned}
$$

**Code**:
```python
# See: examples/advanced/stefan_problem_1d.py

from mfgarchon.geometry.level_set import TimeDependentDomain

# Initial interface at x = 0.5
phi_init = X[0] - 0.5  # 1D: phi(x) = x - 0.5

td_domain = TimeDependentDomain(phi_init, geometry=grid)

for n in range(num_timesteps):
    # Solve heat equation
    T = solve_heat(td_domain.get_geometry_at_time(t[n]))

    # Compute heat flux jump at interface
    heat_flux_jump = compute_gradient_jump(T, interface_location)
    V_n = -kappa * heat_flux_jump

    # Evolve level set
    td_domain.evolve_step(V_n, dt)
```

**Validation**: 1D analytical solution (Neumann problem) error < 5% ✅

---

## Part III: Combining Advanced BCs with MFG

### 3.1 Capacity-Constrained MFG

**Problem**: Agents optimize paths in congested environment with density limit.

**System**:
$$
\begin{aligned}
-\partial_t U - \frac{\sigma^2}{2}\Delta U + H(\nabla U) &= m \\
\partial_t m - \frac{\sigma^2}{2}\Delta m - \text{div}(m \nabla_p H) &= 0 \\
m(t, x) &\leq m_{\max}(x) \quad \text{(capacity constraint)}
\end{aligned}
$$

**Implementation Pattern**:
```python
constraint = ObstacleConstraint(upper_bound=m_max)

for picard_iter in range(max_picard):
    U = solve_hjb(m_constrained)
    m_unconstrained = solve_fp(U)
    m_constrained = constraint.project(m_unconstrained)  # Enforce capacity
```

**Physical Behavior**:
- When $m < m_{\max}$: Agents move optimally (standard MFG)
- When $m = m_{\max}$: **Queue forms**, value function increases (waiting cost)
- Equilibrium: Some agents choose alternative routes to avoid congestion

**Example**: `examples/advanced/capacity_constrained_mfg_1d.py`

### 3.2 MFG with Expanding Exit (Future Work)

**Problem**: Exit expands when too crowded, contracts when empty.

**Exit Dynamics**:
$$
V_n = k \cdot (m|_{\text{exit}} - m_{\text{threshold}})_+
$$

**Coupling**:
1. Solve MFG on current domain → get $(U, m)$
2. Compute exit density $m|_{\text{exit}}$
3. Evolve exit boundary via Level Set
4. Repeat until convergence

**Status**: Deferred from Phase 3 (Issue #592), planned for future extension.

---

## Part IV: Best Practices and Troubleshooting

### 4.1 Obstacle Constraints

**Convergence Issues**:

**Problem**: Picard iteration doesn't converge with constraints.
**Solution**:
- Reduce Picard relaxation parameter: `m_new = (1-ω)·m_old + ω·m_projected` with $\omega \in [0.3, 0.7]$
- Check that $m_{\max}$ is physically reasonable (not too small)
- Verify initial condition satisfies constraint

**Constraint Violations**:

**Problem**: Solution violates $m \leq m_{\max}$ after projection.
**Check**:
```python
assert np.all(m <= m_max + 1e-10), "Projection failed!"
```
**Fix**: Ensure `obstacle.project()` is called **after** every FP solve.

### 4.2 Level Set Method

**Interface Drift**:

**Problem**: Interface moves incorrectly over time.
**Causes**:
- Wrong sign of velocity (check Stefan condition)
- Symmetric initial conditions → zero velocity
- Numerical errors accumulating

**Fix**:
- Validate initial conditions analytically (ensure non-zero heat flux)
- Increase grid resolution (Nx → 2×Nx)
- Reinitialize more frequently (`if n % 5 == 0: reinitialize()`)

**CFL Stability**:

**Problem**: Solution blows up or oscillates.
**Check CFL condition**:
```python
cfl = np.max(np.abs(velocity)) * dt / dx
print(f"CFL = {cfl:.3f}")  # Should be < 0.5 for stability
```

**Fix**: Reduce timestep: `dt = 0.5 * dx / max_velocity`

**Mass Loss**:

**Problem**: Level Set loses mass over time (volume shrinks unphysically).
**Cause**: First-order upwind scheme has numerical diffusion.
**Mitigation**:
- Use finer grid
- Reinitialize frequently
- (Future) Use higher-order schemes (WENO5)

### 4.3 Performance Optimization

**Projection is Slow**:

**Unlikely** - projection is O(N) pointwise operation, typically < 2% overhead.

**If actually slow**:
- Check if you're projecting entire domain instead of just density
- Ensure using NumPy vectorized operations, not Python loops

**Level Set is Slow**:

**Expected** - bottleneck is usually the **number of timesteps**, not level set operations.

**Optimization Priority**:
1. **First**: Use implicit time-stepping for heat equation (30× speedup possible)
2. **Second**: Increase CFL to 0.9 (safe for first-order schemes)
3. **Third**: Optimize level set (narrow band, WENO5) - only after steps 1-2

---

## Part V: Examples and Tutorials

### 5.1 Complete Examples

**Basic Tier 2**:
- `examples/advanced/obstacle_problem_1d.py` - 1D obstacle with analytical validation (< 1% error)

**MFG with Constraints**:
- `examples/advanced/capacity_constrained_mfg_1d.py` - Corridor evacuation with capacity

**Basic Tier 3**:
- `examples/advanced/stefan_problem_1d.py` - 1D ice melting (4.58% error vs analytical)
- `examples/advanced/stefan_problem_2d.py` - 2D circular interface

**Advanced Tier 3**:
- (Planned) `examples/advanced/mfg_expanding_exit.py` - Exit growth driven by congestion

### 5.2 Tutorials

**Step-by-Step Guides**:
- `docs/user/obstacle_problems.md` - VI tutorial with capacity-constrained MFG walkthrough
- `docs/user/level_set_tutorial.md` - Stefan problem step-by-step

**Theory Background** (in mfg-research):
- `docs/archon-notes/theory/variational_inequalities_theory.md` - VI mathematical foundations
- `docs/archon-notes/theory/level_set_method.md` - Level Set theory and numerics

---

## Part VI: API Reference

### 6.1 Obstacle Constraints

**Class**: `mfgarchon.geometry.boundary.ObstacleConstraint`

**Constructor**:
```python
ObstacleConstraint(
    lower_bound: float | np.ndarray = None,
    upper_bound: float | np.ndarray = None,
)
```

**Methods**:
- `project(u)`: Project `u` onto constraint set (element-wise min/max)
- `is_active(u, tol=1e-6)`: Detect active set where constraint binds
- `active_set_indices(u, tol=1e-6)`: Indices where $|u - \psi| < $ tol

**Example**:
```python
constraint = ObstacleConstraint(lower_bound=0.0, upper_bound=1.0)
u_projected = constraint.project(u_unconstrained)
active = constraint.is_active(u_projected)  # Boolean array
```

### 6.2 Time-Dependent Domains

**Class**: `mfgarchon.geometry.level_set.TimeDependentDomain`

**Constructor**:
```python
TimeDependentDomain(
    phi_initial: np.ndarray,
    geometry: GeometryProtocol,
)
```

**Methods**:
- `evolve_step(velocity, dt)`: Evolve interface by one timestep
- `get_geometry_at_time(t)`: Get static geometry at time `t`
- `reinitialize()`: Restore signed distance function property
- `get_curvature()`: Compute mean curvature at interface

**Example**:
```python
td_domain = TimeDependentDomain(phi_init, grid)
td_domain.evolve_step(velocity=V_n, dt=0.01)
geometry_t = td_domain.get_geometry_at_time(t=0.5)
```

### 6.3 Level Set Evolver

**Class**: `mfgarchon.geometry.level_set.LevelSetEvolver`

**Constructor**:
```python
LevelSetEvolver(
    geometry: GeometryProtocol,
    scheme: str = "upwind",  # Godunov upwind (first-order)
)
```

**Methods**:
- `evolve_step(phi, velocity, dt)`: Single timestep evolution
- `compute_cfl(velocity)`: Compute CFL number for adaptive dt

**Low-Level API** (advanced users):
```python
evolver = LevelSetEvolver(grid, scheme="upwind")
phi_new = evolver.evolve_step(phi_old, velocity, dt)
```

---

## Part VII: FAQ

**Q: Can I use obstacle constraints with FEM solvers?**

A: Yes! Projection works with any discretization. Just project the solution vector after solving.

**Q: How do I know if my Level Set is accurate?**

A: Check `max(|∇φ| - 1)` after each timestep. Should be < 0.2. If larger, reinitialize.

**Q: Can I combine Tier 2 and Tier 3 BCs?**

A: Yes, but not yet tested. Example: Capacity constraint on time-dependent domain. Contact developers if needed.

**Q: What if my obstacle constraint is nonlinear?**

A: Current implementation supports only convex constraints (box, half-space). Nonlinear constraints require optimization-based projection (future work).

**Q: Level Set is losing mass. How to fix?**

A: Three options:
1. Use finer grid (reduce dx)
2. Reinitialize more frequently
3. (Future) Use volume correction or particle level set

**Q: Can I use Level Set for MFG free boundaries?**

A: Yes, this is exactly what it's designed for. See `examples/advanced/stefan_problem_*.py` as template.

---

## Part VIII: Version Compatibility

**Introduced**: v0.18.0 (Phases 1-4 complete)

**Backward Compatibility**:
- Standard BCs (Dirichlet, Neumann, Robin) unchanged
- New features are **opt-in** (no breaking changes)
- Examples work with v0.17.1+ (but advanced BCs require v0.18.0+)

**Future Enhancements** (v0.19.0+):
- Projected Newton method for VIs
- WENO5 scheme for Level Set (higher accuracy)
- Narrow band optimization (faster)
- Volume correction (mass conservation)

---

## Summary

**Tier 2 BCs (Variational Inequalities)**:
- ✅ Use for: Capacity constraints, obstacle avoidance
- ✅ Implementation: One-line projection in solver loop
- ✅ Performance: < 2% overhead
- ✅ Examples: Capacity-constrained MFG, 1D obstacle problem

**Tier 3 BCs (Time-Dependent Boundaries)**:
- ✅ Use for: Moving interfaces, phase transitions, adaptive geometry
- ✅ Implementation: TimeDependentDomain wrapper + evolution loop
- ✅ Validation: Stefan problem < 5% error
- ✅ Examples: 1D/2D Stefan problem, (planned) expanding exit MFG

**Getting Started**:
1. Review examples in `examples/advanced/`
2. Follow tutorials in `docs/user/obstacle_problems.md` and `docs/user/level_set_tutorial.md`
3. Check theory docs for mathematical details
4. Ask questions on GitHub Discussions

**Need Help?**
- GitHub Issues: Bug reports, feature requests
- GitHub Discussions: Usage questions, best practices
- Documentation: `docs/user/` (tutorials)

---

**Last Updated**: 2026-01-18
**Version**: v0.18.0
**Related Docs**:
- Theory: `variational_inequalities_theory.md`, `level_set_method.md`
- Tutorials: `obstacle_problems.md`, `level_set_tutorial.md`
- Examples: `examples/advanced/capacity_constrained_mfg_1d.py`, `stefan_problem_1d.py`

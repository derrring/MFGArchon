# MFGArchon Quickstart

**Get started with MFGArchon in 5 minutes**

---

## Prerequisites

MFGArchon assumes you understand:
- **Mean Field Games**: HJB-FP coupled systems, Nash equilibria
- **Numerical PDEs**: Finite difference methods, stability
- **Python**: NumPy and basic scientific computing

---

## Installation

```bash
git clone https://github.com/derrring/mfgarchon.git
cd mfgarchon
pip install -e .
```

---

## Your First MFG Solution

```python
import numpy as np
from mfgarchon import MFGComponents, MFGProblem
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import neumann_bc

# 1. Define Hamiltonian: H(x, p, m) = |p|^2/2 + coupling(m)
H = SeparableHamiltonian(
    control_cost=QuadraticControlCost(control_cost=1.0),
    coupling=lambda m: 0.1 * m,
    coupling_dm=lambda m: 0.1 * np.ones_like(m),
)

# 2. Define terminal/initial conditions
components = MFGComponents(
    hamiltonian=H,
    u_terminal=lambda x: np.zeros_like(x),
    m_initial=lambda x: np.exp(-5 * (x - 0.5) ** 2),
)

# 3. Create geometry with boundary conditions
domain = TensorProductGrid(
    bounds=[(0.0, 1.0)], Nx_points=[51],
    boundary_conditions=neumann_bc(dimension=1),
)

# 4. Create and solve
problem = MFGProblem(geometry=domain, T=1.0, Nt=20, components=components)
result = problem.solve()

print(f"Converged: {result.converged} in {result.iterations} iterations")
print(f"U shape: {result.U.shape}")  # Value function (Nt+1, Nx)
print(f"M shape: {result.M.shape}")  # Density (Nt+1, Nx)
```

### Custom Solver Parameters

```python
result = problem.solve(
    max_iterations=200,
    tolerance=1e-8,
    verbose=True,
)
```

---

## 2D Example

```python
from mfgarchon.geometry.boundary import no_flux_bc

# 2D domain with reflecting boundaries
domain_2d = TensorProductGrid(
    bounds=[(0.0, 1.0), (0.0, 1.0)],
    Nx_points=[51, 51],
    boundary_conditions=no_flux_bc(dimension=2),
)

problem_2d = MFGProblem(geometry=domain_2d, T=1.0, Nt=20, components=components)
result_2d = problem_2d.solve()
```

---

## Visualize Results

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(result.U, aspect="auto", cmap="viridis")
axes[0].set_title("Value Function u(t,x)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("t")

axes[1].imshow(result.M, aspect="auto", cmap="hot")
axes[1].set_title("Density m(t,x)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")

plt.tight_layout()
plt.show()
```

---

## Next Steps

- **[Tutorials](../../examples/tutorials/)** - Step-by-step learning (01-06)
- **[Basic Examples](../../examples/basic/)** - Single-concept demonstrations
- **[Boundary Conditions Guide](guides/boundary_conditions.md)** - BC types and usage
- **[Advanced Examples](../../examples/advanced/)** - Research-grade problems
- **[HJB Solver Selection](HJB_SOLVER_SELECTION_GUIDE.md)** - Choosing the right method

---

## Common Questions

**Q: Which solver does `problem.solve()` use?**
A: By default, FDM upwind with Picard fixed-point coupling. Use `problem.solve(scheme=...)` for other methods.

**Q: How do I check convergence?**
```python
if result.converged:
    print(f"Converged in {result.iterations} iterations")
else:
    print("Did not converge — increase max_iterations or adjust damping")
```

**Q: Where do I get help?**
- [GitHub Issues](https://github.com/derrring/mfgarchon/issues) - Report bugs

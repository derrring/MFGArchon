# MFGArchon User Documentation

**Research-grade Mean Field Games solver**

---

## Get Started

```python
import numpy as np
from mfgarchon import MFGComponents, MFGProblem
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import neumann_bc

H = SeparableHamiltonian(
    control_cost=QuadraticControlCost(control_cost=1.0),
    coupling=lambda m: 0.1 * m,
    coupling_dm=lambda m: 0.1 * np.ones_like(m),
)
components = MFGComponents(
    hamiltonian=H,
    u_terminal=lambda x: np.zeros_like(x),
    m_initial=lambda x: np.exp(-5 * (x - 0.5) ** 2),
)
domain = TensorProductGrid(
    bounds=[(0.0, 1.0)], Nx_points=[51],
    boundary_conditions=neumann_bc(dimension=1),
)

problem = MFGProblem(geometry=domain, T=1.0, Nt=20, components=components)
result = problem.solve()
```

Full tutorial: [Quickstart](quickstart.md)

---

## Documentation Map

### Tutorials (`examples/tutorials/`)

Step-by-step learning from basics to advanced:

| Tutorial | Topic |
|:---------|:------|
| [01 - Hello MFG](../../examples/tutorials/01_hello_mfg.ipynb) | First MFG solve |
| [02 - Custom Hamiltonian](../../examples/tutorials/02_custom_hamiltonian.ipynb) | Non-quadratic control |
| [03 - 2D Geometry](../../examples/tutorials/03_2d_geometry.ipynb) | Multi-dimensional problems |
| [04 - Particle Methods](../../examples/tutorials/04_particle_methods.ipynb) | Monte Carlo FP solver |
| [05 - Config System](../../examples/tutorials/05_config_system.ipynb) | Pydantic |
| [06 - BC Coupling](../../examples/tutorials/06_boundary_condition_coupling.ipynb) | Adjoint-consistent BC |

### Guides

| Guide | Content |
|:------|:--------|
| [Quickstart](quickstart.md) | 5-minute setup |
| [Boundary Conditions](guides/boundary_conditions.md) | BC types, mixed BC, ghost cells, periodic compatibility |
| [Advanced BC](advanced_boundary_conditions.md) | Variational inequalities, moving boundaries |
| [Backend Usage](guides/backend_usage.md) | NumPy, JAX, PyTorch backends |
| [Maze Generation](guides/maze_generation.md) | Graph-based MFG domains |
| [HJB Solver Selection](HJB_SOLVER_SELECTION_GUIDE.md) | Choosing the right numerical method |
| [HJB Solver Selection](HJB_SOLVER_SELECTION_GUIDE.md) | FDM vs GFDM vs SL vs WENO |
| [Migration to v0.19 config](migration_v0.19.md) | Config-schema field mapping |
| [Deprecation Guide](DEPRECATION_MODERNIZATION_GUIDE.md) | Legacy parameter migration paths |

### Examples

| Directory | Content |
|:----------|:--------|
| [Basic](../../examples/basic/) | Single-concept demonstrations |
| [Advanced](../../examples/advanced/) | Research-grade problems |

---

## Key Concepts

### Problem Definition

MFG problems require three components:
1. **Hamiltonian** (`SeparableHamiltonian`): Control cost + coupling term
2. **Conditions** (`MFGComponents`): Terminal value `u_terminal` + initial density `m_initial`
3. **Geometry** (`TensorProductGrid`): Domain + boundary conditions

### Solving

```python
# Default solver (FDM upwind + Picard coupling)
result = problem.solve()

# With scheme selection
from mfgarchon.alg.numerical.adjoint import NumericalScheme
result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)

# With custom parameters
result = problem.solve(max_iterations=200, tolerance=1e-8, verbose=True)
```

### Results

```python
result.U          # Value function u(t,x), shape (Nt+1, Nx)
result.M          # Density m(t,x), shape (Nt+1, Nx)
result.converged  # Boolean
result.iterations # Number of Picard iterations
```

---

## Prerequisites

MFGArchon assumes familiarity with:
- Mean Field Games (HJB-FP coupled systems, Nash equilibria)
- Numerical PDEs (finite difference methods, stability)
- Python (NumPy, scientific computing)

# MFGArchon User Documentation

**Research-grade Mean Field Games solver**

---

## Get Started

```python
import numpy as np
from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import neumann_bc

# 1. The game: H(t, x, p, m) = |p|^2/2 - f(m), where f(m) = 0.1 m is a congestion cost
H = SeparableHamiltonian(
    control_cost=QuadraticControlCost(lambda_=1.0),
    coupling=lambda m: 0.1 * m,
    coupling_dm=lambda m: 0.1 * np.ones_like(m),
)
# At Model's default volatility 0.1 this problem does not converge under the default Picard
# settings; MFGSolverConfig(picard=PicardConfig(relaxation=0.2)) makes it converge there.
model = Model(hamiltonian=H, volatility=0.2)

# 2. The data: terminal value, initial density and horizon
conditions = Conditions(
    u_terminal=lambda x: np.zeros_like(x),
    # Integrates to 0.702328 on this grid, not 1. The library measures the mass, names the
    # measure, and warns -- but does not rescale your density (#1887): whether the initial
    # mass should be 1 is your modelling decision. To make it one, divide by
    # `domain.integrate(...)`, and note that this changes the problem -- the coupling f(m)
    # is then evaluated at 1.4238x these values.
    m_initial=lambda x: np.exp(-5 * (x - 0.5) ** 2),
    T=1.0,
)

# 3. The domain, with its boundary conditions
domain = TensorProductGrid(
    bounds=[(0.0, 1.0)], Nx_points=[51],
    boundary_conditions=neumann_bc(dimension=1),
)

# 4. Assemble and solve
problem = MFGProblem(model=model, domain=domain, conditions=conditions, Nt=20)
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
| [05 - Problem Variations](../../examples/tutorials/05_config_system.ipynb) | Parameter studies |

### Guides

| Guide | Content |
|:------|:--------|
| [Quickstart](quickstart.md) | 5-minute setup |
| [Conventions](CONVENTIONS.md) | Signs, volatility vs diffusion, callable signatures, array layout — the one place each convention is stated |
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

A problem is assembled from three components and a time discretisation:
1. **`Model`** — the game: the Hamiltonian (`SeparableHamiltonian`: control cost, potential,
   coupling) and the volatility
2. **`Conditions`** — the data: terminal value `u_terminal`, initial density `m_initial`, horizon `T`
3. **Domain** (`TensorProductGrid`) — the grid and its boundary conditions
4. **`MFGProblem(model=..., domain=..., conditions=..., Nt=...)`** — the assembly

Which component owns which parameter, and the sign of every term: [Conventions](CONVENTIONS.md).

### Solving

```python
# Default solver (FDM upwind + Picard coupling)
result = problem.solve()

# With scheme selection
from mfgarchon.types import NumericalScheme
result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)

# With custom parameters
result = problem.solve(max_iterations=200, tolerance=1e-8, verbose=True)
```

### Results

```python
result.U          # Value function u(t,x), shape (Nt+1, Nx_points) in 1-D, (Nt+1, *Nx_points) in general
result.M          # Density m(t,x), same shape
result.converged  # Boolean
result.iterations # Number of Picard iterations
```

---

## Prerequisites

MFGArchon assumes familiarity with:
- Mean Field Games (HJB-FP coupled systems, Nash equilibria)
- Numerical PDEs (finite difference methods, stability)
- Python (NumPy, scientific computing)

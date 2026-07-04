# MFGArchon: Mean Field Games Framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CI/CD](https://github.com/derrring/MFGArchon/actions/workflows/ci.yml/badge.svg)](https://github.com/derrring/MFGArchon/actions/workflows/ci.yml)
[![python-compat](https://img.shields.io/github/actions/workflow/status/derrring/MFGArchon/python-compat.yml?branch=main&label=python-compat)](https://github.com/derrring/MFGArchon/actions/workflows/python-compat.yml)
[![codecov](https://codecov.io/gh/derrring/MFGArchon/graph/badge.svg?token=HGZFRSF5V6)](https://codecov.io/gh/derrring/MFGArchon)
[![Release](https://img.shields.io/github/v/release/derrring/MFGArchon)](https://github.com/derrring/MFGArchon/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19251867.svg)](https://doi.org/10.5281/zenodo.19251867)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/derrring/MFGArchon/blob/main/LICENSE)

A Python framework for solving Mean Field Game systems using modern numerical methods, GPU acceleration, and reinforcement learning.

---

## Quick Start

### Installation

```bash
git clone https://github.com/derrring/mfgarchon.git
cd mfgarchon
pip install -e .
```

### Your First MFG Solution

```python
import numpy as np
from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import neumann_bc

# Model: game rules (Hamiltonian + diffusion)
model = Model(
    hamiltonian=SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.1 * m,
        coupling_dm=lambda m: 0.1 * np.ones_like(m),
    ),
    sigma=0.1,
)

# Domain: spatial grid with boundary conditions
domain = TensorProductGrid(
    bounds=[(0.0, 1.0)], Nx_points=[51],
    boundary_conditions=neumann_bc(dimension=1),
)

# Conditions: initial density + terminal cost + time horizon
conditions = Conditions(
    u_terminal=lambda x: np.zeros_like(x),
    m_initial=lambda x: np.exp(-5 * (x - 0.5) ** 2),
    T=1.0,
)

# Create and solve
problem = MFGProblem(model=model, domain=domain, conditions=conditions, Nt=20)
result = problem.solve()
print(f"Converged: {result.converged} in {result.iterations} iterations")
```

---

## Key Features

- **Clean API** - `Model` (game rules) + `Domain` (space) + `Conditions` (data) = `Problem.solve()`
- **Modular** - Mix and match HJB + FP solvers (FDM, GFDM, Semi-Lagrangian, WENO, Particles, FEM, Neural)
- **Multi-Dimensional** - 1D/2D/3D/nD support with TensorProductGrid and implicit domains
- **Geometry Traits** - 12 protocol-based traits for solver-geometry compatibility validation
- **Unified BC Framework** - 4-layer architecture with adjoint-consistent provider pattern
- **Network MFG** - Graph-coupled multi-node solvers with pluggable coupling operators
- **Measure-Dependent MFG** - MeasureField, Lions derivative, Wasserstein distance (Layer 2)
- **Reinforcement Learning** - Complete RL framework (DDPG, TD3, SAC)
- **GPU Acceleration** - PyTorch, JAX, Numba backends

---

## Documentation

**Tutorials** (`examples/tutorials/`) — Jupyter notebooks with math + code:
- [01 - Hello MFG](examples/tutorials/01_hello_mfg.ipynb) - Your first MFG solve
- [02 - Custom Hamiltonian](examples/tutorials/02_custom_hamiltonian.ipynb) - Non-quadratic control
- [03 - 2D Geometry](examples/tutorials/03_2d_geometry.ipynb) - Multi-dimensional problems
- [04 - Particle Methods](examples/tutorials/04_particle_methods.ipynb) - Monte Carlo FP solver
- [05 - Config System](examples/tutorials/05_config_system.ipynb) - Pydantic + OmegaConf
- [06 - BC Coupling](examples/tutorials/06_boundary_condition_coupling.ipynb) - Adjoint-consistent BC

**Guides** (`docs/user/guides/`):
- [Boundary Conditions](docs/user/guides/boundary_conditions.md) - BC types, mixed BC, ghost cells
- [Advanced BC](docs/user/advanced_boundary_conditions.md) - Variational inequalities, moving boundaries
- [Backend Usage](docs/user/guides/backend_usage.md) - NumPy, JAX, PyTorch backends
- [Maze Generation](docs/user/guides/maze_generation.md) - Graph-based MFG domains

---

## Numerical Methods

### HJB Solvers
- **Finite Difference (FDM)** - Standard grid-based with upwind schemes
- **GFDM** - Meshfree generalized FDM with QP-monotonicity enforcement
- **Semi-Lagrangian** - Adaptive time-stepping with periodic BC support
- **WENO** - High-order (5th) shock-capturing with high-order ghost nodes
- **FEM** - scikit-fem based P1/P2 finite elements on unstructured meshes
- **Neural (DGM, PINN)** - Deep learning for high dimensions

### Fokker-Planck Solvers
- **FDM** - Conservative finite difference with dict-dispatched BC
- **Particle Methods** - Monte Carlo, KDE, MCMC sampling
- **FEM** - Mass-conserving Galerkin weak form
- **Semi-Lagrangian Adjoint** - Structure-preserving forward splatting

### Coupling Methods
- **Fixed-Point (Picard)** - With Anderson acceleration and adaptive damping
- **Fictitious Play** - Decaying learning rates for potential games
- **Block Iterators** - Jacobi and Gauss-Seidel with true adjoint mode
- **Newton** - Quadratic convergence near solution
- **Regime Switching** - Markov-chain coupled multi-regime systems
- **Graph MFG** - N-node graph with pluggable coupling (adjacency, Laplacian)
- **Homotopy Continuation** - Predictor-corrector for equilibrium branch tracing

### Geometry & Boundaries
- **TensorProductGrid** - Structured nD grids with 12 trait protocols
- **Implicit Domains** - SDF-based meshfree geometry with CSG operations
- **Unstructured Meshes** - Gmsh integration for FEM (Mesh1D/2D/3D)
- **Graph Networks** - MFG on abstract graphs and mazes
- **Region Predicates** - `box_region()`, `sphere_region()`, `sdf_region()` for spatial marking

See [CHANGELOG.md](CHANGELOG.md) for version history.

---

## Installation

```bash
pip install mfgarchon          # Batteries included (FDM, FEM, GFDM, viz, config)
pip install mfgarchon[nn]      # + PyTorch, RL (DGM, PINN, Actor-Critic, PPO)
pip install mfgarchon[all]     # + JAX, Numba, profiling tools
```

**Default install** includes: NumPy, SciPy, Matplotlib, Rich, scikit-fem, meshio, osqp, igraph, Hydra/OmegaConf, Jupyter, Plotly.

**Requires**: Python 3.12+

---

## Citation

If you use MFGArchon in your research, please cite it. You can use GitHub's
"Cite this repository" button in the sidebar, or use the following BibTeX:

```bibtex
@software{MFGArchon2025,
  title={{MFGArchon}: A Research-Grade Framework for Mean Field Games},
  author={Wang, Jiongyi},
  year={2025},
  doi={10.5281/zenodo.19251867},
  url={https://github.com/derrring/MFGArchon}
}
```

See [CITATION.cff](CITATION.cff) for the machine-readable citation metadata.

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

**Copyright** (c) 2025-2026 Jeremy Jiongyi Wang

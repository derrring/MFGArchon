# MFGArchon Tutorials

A structured 6-step learning path for Mean Field Games in Python.

## Format

Each tutorial is a Python script (`.py`) with inline comments and explanations. Some also have companion Jupyter notebooks (`.ipynb`). The `.py` files are canonical.

## Learning Path

### 01. [Hello MFG](./01_hello_mfg.py) | [notebook](./01_hello_mfg.ipynb)
**Difficulty**: Beginner | **Time**: 10 minutes

Your first Mean Field Game solve.

**You'll learn**: `Model`, `Conditions`, `MFGProblem`, `SeparableHamiltonian`, `TensorProductGrid`, `problem.solve()`

---

### 02. [Custom Hamiltonian](./02_custom_hamiltonian.py) | [notebook](./02_custom_hamiltonian.ipynb)
**Difficulty**: Beginner | **Time**: 15 minutes

Define custom MFG problems with non-standard Hamiltonians.

**You'll learn**: `HamiltonianBase` subclassing, custom coupling terms, congestion effects

---

### 03. [2D Geometry](./03_2d_geometry.py)
**Difficulty**: Intermediate | **Time**: 20 minutes

Move from 1D to 2D spatial domains.

**You'll learn**: Multi-dimensional `TensorProductGrid`, 2D boundary conditions, density visualization

---

### 04. [Particle Methods](./04_particle_methods.py) | [notebook](./04_particle_methods.ipynb)
**Difficulty**: Intermediate | **Time**: 25 minutes

Particle-based FP solvers as alternative to grid methods.

**You'll learn**: FDM vs particle solvers, particle count vs accuracy tradeoff, solver configuration

---

### 05. [Problem Variations and Next Steps](./05_config_system.py) | [notebook](./05_config_system.ipynb)
**Difficulty**: Intermediate | **Time**: 20 minutes

Vary problem parameters and compare the resulting solutions. **Despite the filename, this tutorial
does not cover the configuration system** — it never mentions `MFGSolverConfig`. For that, see
[`docs/user/configuration_system.md`](../../docs/user/configuration_system.md). The file keeps its
name so existing links resolve.

**You'll learn**: building problem variations, comparing solutions across settings, reading MFG
solution outputs

---

### 06. [Boundary Condition Coupling](./06_boundary_condition_coupling.py)
**Difficulty**: Advanced | **Time**: 25 minutes

Adjoint-consistent boundary conditions for reflecting boundaries.

**You'll learn**: `AdjointConsistentProvider`, Robin BC from density gradient, stall point handling

---

## Quick Start

```bash
python examples/tutorials/01_hello_mfg.py
python examples/tutorials/02_custom_hamiltonian.py
python examples/tutorials/03_2d_geometry.py
python examples/tutorials/04_particle_methods.py
python examples/tutorials/05_config_system.py
python examples/tutorials/06_boundary_condition_coupling.py
```

## Prerequisites

- Python 3.12+
- MFGArchon installed (`pip install -e .`)
- Matplotlib (for visualizations)

## After Completing Tutorials

- [Basic Examples](../basic/) - Single-concept demonstrations
- [Advanced Examples](../advanced/) - Research-grade problems
- [Boundary Conditions Guide](../../docs/user/guides/boundary_conditions.md)
- [HJB Solver Selection Guide](../../docs/user/HJB_SOLVER_SELECTION_GUIDE.md)

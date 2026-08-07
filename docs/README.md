# MFGArchon Documentation

**Version**: v0.17.16

Production-ready infrastructure for Mean Field Games research and applications.

## What is MFGArchon?

MFGArchon provides solvers, geometry primitives, boundary condition frameworks, and workflow tools for solving Mean Field Game systems. It supports:

- **HJB solvers**: FDM, GFDM, WENO, Semi-Lagrangian
- **FP solvers**: FDM, GFDM, Particle, Semi-Lagrangian
- **Coupling**: Fixed-point iteration, Newton, fictitious play
- **Geometry**: Tensor product grids, implicit (SDF) domains, meshfree
- **Boundary conditions**: Dirichlet, Neumann, Robin, periodic, no-flux, mixed

## Building the docs

```bash
cd docs/
jupyter-book start .     # Local dev server with hot reload
jupyter-book build .     # Static HTML build
```

## Internal development notes

Architecture design, roadmaps, and issue analysis are in the private `mfg-research` repository under `docs/archon-notes/`.

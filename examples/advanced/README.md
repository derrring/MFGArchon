# Advanced Examples

Multi-feature demonstrations combining multiple concepts and advanced techniques. These examples showcase complex applications and cutting-edge methods.

## Quick Start

```bash
# Run any example
python examples/advanced/applications/traffic_flow_2d_demo.py
python examples/advanced/solvers_advanced/semi_lagrangian_validation.py

# Outputs saved to examples/outputs/advanced/
```

**Prerequisites**: Complete [tutorials/](../tutorials/) and explore [basic/](../basic/) before tackling advanced examples.

---

## Directory Structure

### [comprehensive/](comprehensive/)
Full-featured demonstrations combining multiple systems:
- **`all_maze_algorithms_visualization.py`** - Comparison of maze generation algorithms
- **`pydantic_validation_example.py`** - Configuration validation with Pydantic

### [geometry_advanced/](geometry_advanced/)
Advanced geometric features and high-dimensional problems:
- **`arbitrary_nd_geometry_demo.py`** - nD geometry (3D, 4D, arbitrary dimensions)
- **`maze_implicit_geometry_demo.py`** - Maze + implicit geometry hybrid
- **`triangular_amr_integration.py`** - Triangular mesh AMR
- **`amr_1d_geometry_demo.py`** - 1D adaptive mesh refinement
- **`dual_geometry_fem_mesh.py`** - FEM mesh integration
- **`mfg_2d_geometry_example.py`** - 2D geometry examples

### [solvers_advanced/](solvers_advanced/)
Advanced numerical methods and hybrid approaches:
- **`semi_lagrangian_validation.py`** - Semi-Lagrangian HJB solver
- **`semi_lagrangian_2d_enhancements.py`** - 2D Semi-Lagrangian enhancements
- **`weno_family_comparison_demo.py`** - WENO schemes comparison
- **`particle_collocation_dual_mode_demo.py`** - Particle collocation with dual modes
- **`jax_acceleration_demo.py`** - JAX GPU acceleration

### [optimization/](optimization/)
Constrained optimization and primal-dual methods:
- **`primal_dual_constrained_example.py`** - Primal-dual methods for constrained MFG
- **`lagrangian_constrained_optimization.py`** - Lagrangian optimization with constraints

### [applications/](applications/)
Real-world applications and case studies:
- **`traffic_flow_2d_demo.py`** - Traffic flow modeling
- **`portfolio_optimization_2d_demo.py`** - Financial portfolio optimization
- **`epidemic_modeling_2d_demo.py`** - Epidemic spread dynamics
- **`network_mfg_comparison_example.py`** - Network MFG methods
- **`el_farol_bar_demo.py`** - El Farol Bar coordination game
- **`santa_fe_bar_demo.py`** - Santa Fe Bar problem
- **`towel_beach_demo.py`** - Beach towel placement game

### [visualization/](visualization/)
Advanced visualization techniques:
- **`visualize_2d_density_evolution.py`** - 2D density evolution visualization

---

## Category Guide

### Comprehensive (⭐⭐⭐ Complex)
Full-system demonstrations integrating multiple components.

**When to use**:
- Understanding complete workflows
- Seeing how components integrate
- Production-ready examples

### Geometry Advanced (⭐⭐⭐ Complex)
High-dimensional problems and advanced meshing.

**When to use**:
- 3D, 4D, or higher dimensions
- Complex geometries (mazes, obstacles)
- Adaptive mesh refinement
- FEM/FDM mesh integration

### Solvers Advanced (⭐⭐⭐⭐ Expert)
Cutting-edge numerical methods.

**When to use**:
- High-order accuracy requirements
- Stiff problems (Semi-Lagrangian)
- Hybrid particle-grid methods
- GPU acceleration

### Optimization (⭐⭐⭐ Complex)
Constrained MFG problems.

**When to use**:
- State/control constraints
- Primal-dual methods
- Lagrangian formulations

### Machine Learning (⭐⭐⭐⭐ Expert)
Neural network and deep RL methods.

**When to use**:
- High-dimensional problems (curse of dimensionality)
- Mesh-free solutions
- Data-driven approaches
- Continuous/discrete action RL

### Applications (⭐⭐⭐ Complex)
Real-world use cases and domain-specific problems.

**When to use**:
- Traffic engineering
- Finance and economics
- Epidemiology
- Game theory
- Network dynamics

### Visualization (⭐⭐ Moderate)
Advanced plotting and animation.

**When to use**:
- Publication-quality figures
- Interactive dashboards
- Animation of density evolution

---

## Prerequisites

### Mathematical Background
- Mean Field Games theory (HJB + Fokker-Planck)
- Numerical PDE methods (FDM, FEM, particle methods)
- Optimization theory (for constrained problems)
- Reinforcement learning basics (for RL examples)

### Software Requirements

**Core**:
```bash
pip install numpy scipy matplotlib
```

**Advanced features**:
```bash
pip install jax jaxlib          # GPU acceleration
pip install torch               # Neural networks
pip install gymnasium stable-baselines3  # Reinforcement learning
pip install plotly              # Interactive visualization
pip install networkx            # Network MFG
```

---

## Example Difficulty

- ⭐⭐⭐ **Complex**: Multiple concepts, assumes basic examples completed
- ⭐⭐⭐⭐ **Expert**: Cutting-edge methods, requires domain expertise

---

## Output Files

All examples save to `examples/outputs/advanced/`:
- PNG/HTML visualizations
- Convergence histories (CSV)
- Trained models (PyTorch .pt files)
- Structured logs

---

## Learning Path

**Recommended progression through advanced examples**:

1. **Applications First** (Motivation):
   - `applications/traffic_flow_2d_demo.py`
   - `applications/epidemic_modeling_2d_demo.py`

2. **Geometry** (Spatial complexity):
   - `geometry_advanced/arbitrary_nd_geometry_demo.py`
   - `geometry_advanced/maze_implicit_geometry_demo.py`

3. **Solvers** (Numerical methods):
   - `solvers_advanced/semi_lagrangian_validation.py`

4. **Machine Learning** (Data-driven):

5. **Comprehensive** (Integration):

---

---

## Contributing

When adding advanced examples:
1. **Multi-concept focus** - Demonstrate integration of features
2. **Comprehensive documentation** - Extensive theory references
3. **Production quality** - Error handling, logging, validation
4. **Category placement** - Choose appropriate subdirectory
5. **README entry** - Update this file and category README

See `CLAUDE.md` for full coding standards.

---

## Common Pitfalls

### High-Dimensional Problems
- **Issue**: Curse of dimensionality (exponential grid growth)
- **Solution**: Use particle methods or neural approaches

### GPU Acceleration
- **Issue**: JAX/CuPy not installed
- **Solution**: Install with `pip install jax[cuda]` (NVIDIA) or fall back to CPU

### RL Convergence
- **Issue**: RL methods don't converge
- **Solution**: Tune hyperparameters, increase training episodes, check reward scaling

### Memory Issues
- **Issue**: Out of memory for large grids
- **Solution**: Reduce resolution, use iterative solvers, enable GPU acceleration

---

**Last Updated**: 2025-11-11
**Total Examples**: 35 files across 7 categories
**Coverage**: Advanced solvers, geometry, ML/RL, applications, optimization, visualization

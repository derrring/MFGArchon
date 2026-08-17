# Basic Examples

Single-concept demonstrations organized by category. Each example focuses on one core feature and can be run standalone.

## Quick Start

```bash
# Run any example
python examples/basic/core_infrastructure/solve_mfg_demo.py
python examples/basic/geometry/2d_crowd_motion_fdm.py

# Outputs saved to examples/outputs/basic/
```

**New to MFGArchon?** Start with [tutorials/](../tutorials/) for structured learning path.

---

## Directory Structure

### [core_infrastructure/](core_infrastructure/)
Core API usage and basic MFG setups:
- **`solve_mfg_demo.py`** - Basic `solve_mfg()` function usage
- **`custom_hamiltonian_derivs_demo.py`** - Custom Hamiltonian derivatives

### [geometry/](geometry/)
Domain and geometry examples:
- **`2d_crowd_motion_fdm.py`** - 2D crowd motion with FDM solvers
- **`geometry_first_api_demo.py`** - Geometry-first API pattern
- **`amr_geometry_protocol_demo.py`** - Adaptive mesh refinement
- **`dual_geometry_simple_example.py`** - Dual geometry basics
- **`dual_geometry_multiresolution.py`** - Multi-resolution grids

### [solvers/](solvers/)
Solver method comparisons:
- **`policy_iteration_lq_demo.py`** - Policy iteration for LQ problems
- **`acceleration_comparison.py`** - JAX vs NumPy acceleration

### [utilities/](utilities/)
Tools and utilities:
- **`hdf5_save_load_demo.py`** - HDF5 I/O for results
- **`common_noise_lq_demo.py`** - Stochastic MFG with common noise
- **`solver_result_analysis_demo.py`** - Result analysis tools
- **`utility_demo.py`** - General utility functions

---

## Learning Path

**Recommended order for exploring basic examples**:

2. **API**: `core_infrastructure/solve_mfg_demo.py` - Main solver interface
3. **2D**: `geometry/2d_crowd_motion_fdm.py` - Multi-dimensional problems
4. **Customize**: `core_infrastructure/custom_hamiltonian_derivs_demo.py` - Custom Hamiltonians
5. **Performance**: `solvers/acceleration_comparison.py` - Optimization techniques
6. **I/O**: `utilities/hdf5_save_load_demo.py` - Save/load results
7. **Advanced**: `utilities/common_noise_lq_demo.py` - Stochastic MFG

---

## Example Categories

### Core Infrastructure (⭐ Easiest)
Foundation examples for basic MFG setup and API usage. Start here if you've completed the tutorials.

**When to use**:
- Learning the `solve_mfg()` API
- Understanding Linear-Quadratic problems
- Implementing custom Hamiltonians

### Geometry (⭐⭐ Moderate)
Spatial domain configuration and multi-dimensional problems.

**When to use**:
- 2D/3D problems
- Adaptive mesh refinement
- Multi-resolution grids
- Non-standard geometries

### Solvers (⭐⭐ Moderate)
Different numerical methods and acceleration techniques.

**When to use**:
- Comparing solver performance
- Enabling GPU acceleration
- Policy iteration methods

### Utilities (⭐⭐ Moderate)
Tools for analysis, I/O, and post-processing.

**When to use**:
- Saving/loading large results
- Stochastic MFG problems
- Result analysis and metrics

---

## Dependencies

**Core** (always available):
```bash
pip install numpy scipy matplotlib
```

**Optional** (for specific examples):
```bash
pip install jax jaxlib      # Acceleration
pip install h5py            # HDF5 I/O
pip install plotly          # Interactive plots
```

---

## Output Files

All examples save to `examples/outputs/basic/`:
- PNG plots with visualizations
- Convergence data (CSV)
- Structured logs

**Note**: Output directory is gitignored. Use `examples/outputs/reference/` for tracked reference outputs.

---

## Difficulty Guide

- ⭐ **Easiest**: Basic API usage, good starting point
- ⭐⭐ **Moderate**: One new concept, requires tutorials completed
- ⭐⭐⭐ **Advanced**: Multiple concepts, see `advanced/` directory

---

## Contributing

When adding basic examples:
1. **One concept per example** - Keep focused
2. **Under 300 lines** - Split complex examples into basic + advanced
3. **Comprehensive docstrings** - Include theory references
4. **Save to correct category** - core_infrastructure, geometry, solvers, or utilities
5. **Update this README** - Add entry in appropriate section

See `CLAUDE.md` for full coding standards.

---

---

**Last Updated**: 2025-11-11
**Total Examples**: 14 files across 4 categories
**Coverage**: Core API, 2D geometry, solvers, utilities

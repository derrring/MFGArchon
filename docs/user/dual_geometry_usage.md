# [SUPERSEDED 2026-07-30] Dual Geometry Usage Guide

**Status**: ⛔ **SUPERSEDED — the problem-level API this guide teaches now raises.**
**SUPERSEDED-BY**: `GeometryProjector` (`mfgarchon/operators/interpolation/projection.py`, re-exported as `mfgarchon.geometry.GeometryProjector`); Issue #1765
**Created**: 2025-11-10 · **Superseded**: 2026-07-30
**Audience**: MFGArchon users (researchers, application developers)

> ## Read this before the rest of the page
>
> Every example below passes two *different* geometries to `MFGProblem(hjb_geometry=...,
> fp_geometry=...)`. That construction now raises `NotImplementedError`.
>
> The reason is not a policy change. **Nothing downstream ever resampled between the two
> geometries.** `MFGProblem` accepted both, kept the HJB one, and discarded the FP one, so the FP
> equation silently solved on the HJB grid — a wrong answer with no error and no warning
> (Issue #1765). The capability this guide describes was never implemented; the guide described
> the intent.
>
> **What to do instead.** Drive the projection explicitly, which is what the refusal's error
> message says:
>
```python
from mfgarchon.geometry import GeometryProjector

projector = GeometryProjector(hjb_geometry=hjb_grid, fp_geometry=fp_grid)
m_on_fp = projector.project_hjb_to_fp(m_on_hjb)
u_on_hjb = projector.project_fp_to_hjb(u_on_fp)
```
>
> `GeometryProjector` is unaffected and round-trips correctly; only the automatic problem-level
> wiring is absent.
>
> Two further reasons not to copy from this page: the examples use `dimension=`, deprecated on
> `TensorProductGrid`; and the header claimed `✅ COMPLETED (Issue #257)` for eight months while
> the feature silently did the wrong thing. The text below is kept for the design intent, not as
> working instructions.

## Overview

Dual geometry support allows you to solve the HJB and FP equations on **different discretizations**. This enables:

- **Hybrid methods**: Particle FP + Grid HJB
- **Multi-resolution**: Fine grid HJB + Coarse grid FP
- **Network agents**: Network FP + Grid HJB
- **High-dimensional methods**: Particle FP + Low-dimensional HJB

## Quick Start

### Basic Example: Unified Geometry (Traditional)

```python
from mfgarchon import MFGProblem
from mfgarchon.geometry import TensorProductGrid

# Single geometry for both HJB and FP (traditional approach)
grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx_points=[51, 51])

problem = MFGProblem(
    geometry=grid,  # Same geometry for HJB and FP
    time_domain=(1.0, 50),
    sigma=0.1
)

# Both solvers use the same grid
assert problem.hjb_geometry is grid
assert problem.fp_geometry is grid
assert problem.geometry_projector is None  # No projection needed
```

### Basic Example: Dual Geometry (New)

```python
from mfgarchon import MFGProblem
from mfgarchon.geometry import TensorProductGrid

# Different geometries for HJB and FP
hjb_grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx_points=[101, 101])  # Fine
fp_grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx_points=[21, 21])     # Coarse

problem = MFGProblem(
    hjb_geometry=hjb_grid,  # Fine grid for value iteration
    fp_geometry=fp_grid,     # Coarse grid for density evolution
    time_domain=(1.0, 50),
    sigma=0.1
)

# Automatic projection between geometries
assert problem.hjb_geometry is hjb_grid
assert problem.fp_geometry is fp_grid
assert problem.geometry_projector is not None  # Created automatically
```

## Use Cases

### Use Case 1: Multi-Resolution (Speed vs Accuracy Trade-off)

**Motivation**: Value function needs high resolution, but density evolution is smooth.

```python
from mfgarchon import MFGProblem, solve_mfg
from mfgarchon.geometry import TensorProductGrid

# Fine grid for accurate HJB solution
hjb_grid = TensorProductGrid(dimension=2, bounds=[(0, 10), (0, 10)], Nx_points=[201, 201])

# Coarse grid for fast FP evolution
fp_grid = TensorProductGrid(dimension=2, bounds=[(0, 10), (0, 10)], Nx_points=[51, 51])

problem = MFGProblem(
    hjb_geometry=hjb_grid,
    fp_geometry=fp_grid,
    time_domain=(1.0, 100),
    sigma=0.5,
    final_condition=lambda x, y: (x - 5)**2 + (y - 5)**2,  # Target center
    m0=lambda x, y: np.exp(-((x-2)**2 + (y-2)**2))  # Initial density at corner
)

# Solve with automatic projection
result = solve_mfg(problem, max_iterations=50)

print(f"HJB solution shape: {result.U.shape}")  # (201, 201)
print(f"FP solution shape: {result.M.shape}")   # (51, 51)
```

**Performance**: ~4× speedup compared to uniform fine grid, minimal accuracy loss.

### Use Case 2: Particle-Based FP (Lagrangian Methods)

**Motivation**: Track individual agents while using grid-based value iteration.

```python
import numpy as np
from mfgarchon import MFGProblem
from mfgarchon.geometry import TensorProductGrid

# Grid for HJB (Eulerian)
hjb_grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx_points=[51, 51])

# Particle geometry for FP (Lagrangian)
class ParticleGeometry:
    """Simple particle-based geometry."""

    def __init__(self, num_particles, bounds):
        self.num_particles = num_particles
        self.bounds = bounds
        # Initialize particle positions randomly
        self.positions = np.random.rand(num_particles, 2)

    def get_spatial_grid(self):
        """Return particle positions."""
        return self.positions

    def get_dimension(self):
        return 2

fp_particles = ParticleGeometry(num_particles=5000, bounds=(0, 1, 0, 1))

problem = MFGProblem(
    hjb_geometry=hjb_grid,
    fp_geometry=fp_particles,
    time_domain=(1.0, 50),
    sigma=0.1
)

# Access projector
projector = problem.geometry_projector

# Manual usage in custom solver:
# 1. Particle positions → Grid density (KDE)
particle_masses = np.ones(5000) / 5000  # Uniform masses
M_grid = projector.project_fp_to_hjb(particle_masses, bandwidth="scott")

# 2. Solve HJB on grid
# ... HJB solver ...

# 3. Grid value function → Particle values (interpolation)
U_particles = projector.project_hjb_to_fp(U_grid)

# 4. Update particles using optimal control
# ... particle dynamics ...
```

**Benefits**:
- Natural representation of agent heterogeneity
- Easy obstacle avoidance for particles
- Automatic adaptivity (particles concentrate where needed)

### Use Case 3: Network-Based FP (Urban Planning)

**Motivation**: Agents move on discrete network (roads, corridors) but value function is spatial.

```python
from mfgarchon import MFGProblem
from mfgarchon.geometry import TensorProductGrid, GridNetwork

# Grid for HJB (continuous space)
hjb_grid = TensorProductGrid(dimension=2, bounds=[(0, 10), (0, 10)], Nx_points=[101, 101])

# Network for FP (discrete corridors)
fp_network = GridNetwork(
    nx=10, ny=10,
    bounds=(0, 10, 0, 10),
    network_type="lattice"  # Corridor layout
)

problem = MFGProblem(
    hjb_geometry=hjb_grid,
    fp_geometry=fp_network,
    time_domain=(1.0, 100),
    sigma=0.5,
    final_condition=lambda x, y: -((x - 9)**2 + (y - 9)**2)  # Exit at (9,9)
)

# Projector handles grid ↔ network transformations automatically
projector = problem.geometry_projector

# Grid → Network: Interpolate at node positions
U_grid = ...  # Value function on grid (101, 101)
U_nodes = projector.project_hjb_to_fp(U_grid)  # Values at 100 nodes

# Network → Grid: Spread node densities spatially
M_nodes = ...  # Density on network nodes (100,)
M_grid = projector.project_fp_to_hjb(M_nodes)  # Density on grid (101, 101)
```

**Applications**:
- Building evacuation planning
- Traffic flow on road networks
- Pedestrian dynamics in malls

### Use Case 4: Dimension Reduction (High-D → Low-D)

**Motivation**: Full state space is high-dimensional, but value function has low-dimensional structure.

```python
from mfgarchon import MFGProblem
from mfgarchon.geometry import TensorProductGrid

# High-dimensional particle system (e.g., 10D state space)
class HighDimParticles:
    def __init__(self, num_particles, dimension):
        self.num_particles = num_particles
        self.dimension = dimension
        self.positions = np.random.randn(num_particles, dimension)

    def get_spatial_grid(self):
        # Project to 2D for visualization/value function
        return self.positions[:, :2]  # Use first 2 dimensions

    def get_dimension(self):
        return 2  # Projected dimension

# 2D grid for HJB (value function in reduced space)
hjb_grid = TensorProductGrid(dimension=2, bounds=[(-3, 3), (-3, 3)], Nx_points=[51, 51])

# High-D particles for FP (full state space)
fp_particles = HighDimParticles(num_particles=10000, dimension=10)

problem = MFGProblem(
    hjb_geometry=hjb_grid,
    fp_geometry=fp_particles,
    time_domain=(1.0, 50),
    sigma=0.1
)

# Value function computed in 2D, applied to high-D particles
```

**Advanced**: Implement custom projection with proper dimensionality reduction (PCA, autoencoders).

## Projection Methods

The `GeometryProjector` automatically selects appropriate projection methods based on geometry types.

### Automatic Selection Table

| HJB Geometry | FP Geometry | HJB→FP Method | FP→HJB Method |
|:-------------|:------------|:--------------|:--------------|
| Grid | Grid (fine) | Grid interpolation | Grid restriction |
| Grid | Grid (coarse) | Grid interpolation | Grid restriction |
| Grid | Particles | Interpolation | KDE |
| Grid | Network | Interpolation | KDE (node spreading) |
| Particles | Grid | KDE | Interpolation |
| Network | Grid | KDE (node spreading) | Interpolation |

### Manual Control

```python
from mfgarchon.geometry import GeometryProjector

projector = GeometryProjector(
    hjb_geometry=hjb_grid,
    fp_geometry=fp_particles,
    projection_method="auto"  # "auto" (default), "kde", "interpolation", "nearest"
)

# Inspect selected methods
print(f"HJB→FP: {projector.hjb_to_fp_method}")  # "interpolation"
print(f"FP→HJB: {projector.fp_to_hjb_method}")  # "kde"
```

### KDE Bandwidth Selection

For particle → grid projections, control bandwidth:

```python
# Scott's rule (default): h ∝ n^(-1/(d+4))
M_grid = projector.project_fp_to_hjb(particle_masses, bandwidth="scott")

# Silverman's rule: h ∝ (n(d+2)/4)^(-1/(d+4))
M_grid = projector.project_fp_to_hjb(particle_masses, bandwidth="silverman")

# Manual bandwidth
M_grid = projector.project_fp_to_hjb(particle_masses, bandwidth=0.05)
```

**Rule of thumb**:
- **Small bandwidth**: Preserves local features, may be noisy
- **Large bandwidth**: Smooth, may over-blur
- **Scott's/Silverman's**: Good default for most cases

## Advanced: Custom Projections

### Registering Custom Projector

If you have a custom geometry type, register specialized projections:

```python
from mfgarchon.geometry import ProjectionRegistry

@ProjectionRegistry.register(MyCustomGeometry, TensorProductGrid, "hjb_to_fp")
def custom_to_grid(custom_geo, grid_geo, values, **kwargs):
    """
    Project from MyCustomGeometry to TensorProductGrid.

    Args:
        custom_geo: Source geometry
        grid_geo: Target grid
        values: Values on custom geometry
        **kwargs: Additional arguments

    Returns:
        Values on grid (nx+1, ny+1)
    """
    # Your custom projection logic
    grid_values = ...
    return grid_values

@ProjectionRegistry.register(TensorProductGrid, MyCustomGeometry, "fp_to_hjb")
def grid_to_custom(grid_geo, custom_geo, values, **kwargs):
    """Project from TensorProductGrid to MyCustomGeometry."""
    custom_values = ...
    return custom_values

# Now use in MFGProblem
problem = MFGProblem(
    hjb_geometry=my_custom_geo,
    fp_geometry=grid,
    time_domain=(1.0, 50),
    sigma=0.1
)

# Automatically uses registered projectors
```

### Category-Based Registration

Register for all geometries of a base type:

```python
from mfgarchon.geometry.base_geometry import CartesianGrid

@ProjectionRegistry.register(CartesianGrid, CartesianGrid, "hjb_to_fp")
def conservative_projection(source, target, values, **kwargs):
    """Conservative projection for any CartesianGrid types."""
    # Works for any TensorProductGrid (1D, 2D, 3D, nD)
    return projected_values
```

## Error Handling

### Common Errors

**Error 1: Conflicting parameters**
```python
# ❌ WRONG: Can't specify both
problem = MFGProblem(
    geometry=grid1,
    hjb_geometry=grid2,
    fp_geometry=grid3,
    ...
)
# ValueError: Specify EITHER 'geometry' OR ('hjb_geometry', 'fp_geometry'), not both
```

**Solution**: Choose one mode:
```python
# ✅ Unified mode
problem = MFGProblem(geometry=grid, ...)

# ✅ Dual mode
problem = MFGProblem(hjb_geometry=grid1, fp_geometry=grid2, ...)
```

**Error 2: Partial dual specification**
```python
# ❌ WRONG: Must specify both
problem = MFGProblem(hjb_geometry=grid, ...)
# ValueError: If using dual geometries, both 'hjb_geometry' AND 'fp_geometry' must be specified
```

**Solution**:
```python
# ✅ Specify both
problem = MFGProblem(hjb_geometry=grid1, fp_geometry=grid2, ...)
```

**Error 3: Shape mismatch**
```python
U_hjb = np.zeros((101, 101))
U_fp = projector.project_hjb_to_fp(U_hjb)
# Expected (51, 51), got (2601,)
```

**Cause**: Using wrong projection method or geometry not compatible.

**Solution**: Ensure geometries have `get_grid_shape()` method or use manual reshaping.

## Performance Tips

### Tip 1: Choose Appropriate Resolutions

```python
# HJB needs fine resolution (sharp gradients)
hjb_resolution = (100, 100)

# FP can use coarser resolution (smooth densities)
fp_resolution = (25, 25)  # 4× speedup in FP solver

# Projection overhead is negligible compared to solver time
```

### Tip 2: Cache Projections

```python
# If projecting same geometry repeatedly, cache points
projector = problem.geometry_projector

# First call computes and caches FP points
U_fp_1 = projector.project_hjb_to_fp(U_hjb_1)

# Second call reuses cached points (faster)
U_fp_2 = projector.project_hjb_to_fp(U_hjb_2)
```

### Tip 3: Use Particles for Sparse Domains

```python
# For domains with obstacles, particles automatically avoid obstacles
# Grid methods need explicit obstacle handling

# Particles naturally concentrate where density is high
# No wasted grid points in empty regions
```

### Tip 4: KDE Bandwidth Selection

```python
# For smooth densities: use Scott's rule (default)
M_grid = projector.project_fp_to_hjb(particles, bandwidth="scott")

# For preserving sharp features: use smaller bandwidth
M_grid = projector.project_fp_to_hjb(particles, bandwidth=0.01)

# For noisy particle data: use larger bandwidth
M_grid = projector.project_fp_to_hjb(particles, bandwidth=0.1)
```

## Validation and Debugging

### Check Projection Quality

```python
import matplotlib.pyplot as plt

# Original on coarse grid
U_coarse = ...  # Shape (21, 21)

# Project to fine grid
U_fine = projector.project_hjb_to_fp(U_coarse)  # Shape (101, 101)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(U_coarse, origin='lower', extent=[0, 1, 0, 1])
axes[0].set_title("Original (Coarse)")

axes[1].imshow(U_fine, origin='lower', extent=[0, 1, 0, 1])
axes[1].set_title("Projected (Fine)")

plt.show()
```

### Check Mass Conservation

```python
# Particles → Grid
particle_masses = np.ones(N_particles) / N_particles
total_mass_particles = np.sum(particle_masses)

M_grid = projector.project_fp_to_hjb(particle_masses, bandwidth="scott")
cell_volume = np.prod(hjb_grid.get_cell_sizes())
total_mass_grid = np.sum(M_grid) * cell_volume

rel_error = abs(total_mass_grid - total_mass_particles) / total_mass_particles
print(f"Mass conservation error: {rel_error:.2%}")
# Should be < 5% for good bandwidth choice
```

### Inspect Projection Methods

```python
projector = problem.geometry_projector

print(f"HJB geometry: {type(problem.hjb_geometry).__name__}")
print(f"FP geometry: {type(problem.fp_geometry).__name__}")
print(f"HJB→FP method: {projector.hjb_to_fp_method}")
print(f"FP→HJB method: {projector.fp_to_hjb_method}")

# Check if using custom registered projector
if projector.hjb_to_fp_method == "registry":
    print(f"Using registered function: {projector._hjb_to_fp_func.__name__}")
```

## Complete Example: Crowd Evacuation

```python
import numpy as np
import matplotlib.pyplot as plt
from mfgarchon import MFGProblem, solve_mfg
from mfgarchon.geometry import TensorProductGrid

# Problem: Evacuate building (10m × 10m) with exit at (0, 5)
# Use fine grid for HJB (accurate value function near exit)
# Use coarse grid for FP (fast density evolution)

# Fine grid for HJB (need accuracy near exit)
hjb_grid = TensorProductGrid(dimension=2, bounds=[(0, 10), (0, 10)], Nx_points=[201, 201])

# Coarse grid for FP (density is smooth)
fp_grid = TensorProductGrid(dimension=2, bounds=[(0, 10), (0, 10)], Nx_points=[51, 51])

# Terminal cost: distance to exit at (0, 5)
def g(x, y):
    return np.sqrt(x**2 + (y - 5)**2)

# Initial density: uniform in building
def m0(x, y):
    return np.ones_like(x) / 100.0  # Normalized to integrate to 1

# Running cost: prefer empty areas (congestion penalty)
def f(x, y, m):
    return m  # Congestion cost

problem = MFGProblem(
    hjb_geometry=hjb_grid,
    fp_geometry=fp_grid,
    time_domain=(5.0, 100),  # 5 seconds to evacuate, 100 time steps
    sigma=0.5,
    final_condition=g,
    m0=m0,
    running_cost=f,
    coupling_coefficient=1.0  # Strength of congestion penalty
)

# Solve
result = solve_mfg(problem, max_iterations=20)

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Value function on fine grid
x_hjb = np.linspace(0, 10, 201)
y_hjb = np.linspace(0, 10, 201)
X_hjb, Y_hjb = np.meshgrid(x_hjb, y_hjb)

axes[0].contourf(X_hjb, Y_hjb, result.U, levels=20, cmap='viridis')
axes[0].set_title("Value Function (HJB, Fine Grid 200×200)")
axes[0].set_xlabel("x (m)")
axes[0].set_ylabel("y (m)")
axes[0].plot(0, 5, 'r*', markersize=15, label='Exit')
axes[0].legend()

# Density on coarse grid
x_fp = np.linspace(0, 10, 51)
y_fp = np.linspace(0, 10, 51)
X_fp, Y_fp = np.meshgrid(x_fp, y_fp)

axes[1].contourf(X_fp, Y_fp, result.M, levels=20, cmap='hot')
axes[1].set_title("Final Density (FP, Coarse Grid 50×50)")
axes[1].set_xlabel("x (m)")
axes[1].set_ylabel("y (m)")
axes[1].plot(0, 5, 'r*', markersize=15, label='Exit')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Converged in {result.iterations} iterations")
print(f"HJB solution shape: {result.U.shape}")  # (201, 201)
print(f"FP solution shape: {result.M.shape}")   # (51, 51)
```

## FAQ

### Q1: When should I use dual geometries?

**Use dual geometries when:**
- HJB and FP have different resolution requirements
- Using particle-based methods for FP
- Working with network/graph geometries
- Need computational speedup (coarser FP grid)
- Tracking individual agents (particles)

**Stick with unified geometry when:**
- Both equations have similar resolution needs
- Using standard finite difference methods
- Problem is low-dimensional (1D, 2D)
- Simplicity is more important than performance

### Q2: Does dual geometry affect accuracy?

**Projection introduces error**, but often negligible compared to discretization error:
- **Interpolation error**: $O(h^p)$ where $p$ is interpolation order (usually $p=1$ or $p=2$)
- **KDE error**: Depends on bandwidth choice (Scott's/Silverman's rules are near-optimal)
- **Mass conservation**: KDE approximately conserves mass (within ~5%)

**Trade-off**: Small projection error vs. large speedup (e.g., 4× faster with 2× coarser FP grid)

### Q3: Can I use dual geometries with all solvers?

**Current status**:
- ✅ Infrastructure complete (Issue #257 Phases 1-3)
- ⚠️ Solver integration in progress (Phase 4)

**Solvers with dual geometry support** (after Phase 4):
- `solve_mfg()`
- Custom solvers using `problem.geometry_projector` directly

**Solvers without support yet**:
- Some specialized solvers may assume unified geometry

Check solver documentation for dual geometry compatibility.

### Q4: How do I choose KDE bandwidth?

**Default**: Use `bandwidth="scott"` (automatic, near-optimal)

**Manual tuning**:
1. Try Scott's rule first: `bandwidth="scott"`
2. If density is too smooth: decrease bandwidth by factor of 2
3. If density is too noisy: increase bandwidth by factor of 2
4. Validate with mass conservation check (should be < 5% error)

**Dimension dependence**: Higher dimensions need smaller relative bandwidths due to curse of dimensionality.

### Q5: Can I mix 2D grid with 3D particles?

**Yes**, but projection requires compatible dimensions:

```python
# 3D particles, but use 2D projection
class Particles3D:
    def get_spatial_grid(self):
        # Return only x, y coordinates (ignore z)
        return self.positions[:, :2]

    def get_dimension(self):
        return 2  # Projected dimension

hjb_grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx_points=[51, 51])
fp_particles = Particles3D(...)

problem = MFGProblem(hjb_geometry=hjb_grid, fp_geometry=fp_particles, ...)
```

For true 3D, use `TensorProductGrid` for HJB geometry.

## Further Reading

### Documentation
- `docs/development/GEOMETRY_PROJECTION_IMPLEMENTATION_GUIDE.md`: Developer guide
- API Reference: `mfgarchon.geometry.GeometryProjector`

### Examples
- `examples/basic/dual_geometry_multiresolution.py`: Multi-resolution example
- `examples/advanced/particle_fp_grid_hjb.py`: Particle-based FP
- `examples/advanced/network_evacuation.py`: Network-based agents

### Research Papers
- Benamou, Carlier, Santambrogio: Variational MFG formulations
- Achdou, Capuzzo-Dolcetta: Mean Field Games on networks
- Cottet, Koumoutsakos: Vortex Methods (particle-grid projections)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-10
**Feature Status**: ✅ Available in MFGArchon v1.0+

# [PARTIALLY-SUPERSEDED 2026-07-30] FEM Mesh Projection Guide

**Status**: ⚠️ **The `GeometryProjector` half is current. The `MFGProblem(hjb_geometry=...,
fp_geometry=...)` half now raises.**
**SUPERSEDED-BY**: `GeometryProjector` (`mfgarchon/operators/interpolation/projection.py`,
re-exported as `mfgarchon.geometry.GeometryProjector`); Issue #1765
**Created**: 2025-11-10 · **Partially superseded**: 2026-07-30

> **What changed.** Every `MFGProblem(hjb_geometry=grid, fp_geometry=mesh, ...)` call on this page
> now raises `NotImplementedError`. `MFGProblem` accepted both geometries, kept the HJB one and
> discarded the FP one, so the FP equation silently solved on the HJB grid (Issue #1765). The
> projection machinery this guide describes is real and unaffected — it is the automatic
> problem-level wiring that was never implemented. Drive the projector directly:
>
> ```python
> from mfgarchon.geometry import GeometryProjector
> projector = GeometryProjector(hjb_geometry=grid, fp_geometry=mesh)
> ```
>
> See also `docs/user/dual_geometry_usage.md`, superseded in full for the same reason.

## Quick Answer

**FEM meshes work with `GeometryProjector`.** They do not work as an `fp_geometry=` argument to
`MFGProblem` — see the banner above.

- ✅ **Basic support**: Works out-of-the-box via nearest neighbor fallback
- ✅ **Optimized support**: Easy to add via Delaunay interpolation
- ✅ **Production-ready**: Suitable for mesh-based MFG problems

## Current Support Levels

### Level 1: Basic (Nearest Neighbor) - Out-of-the-Box

The projection system automatically handles FEM meshes using nearest neighbor:

```python
from mfgarchon import MFGProblem
from mfgarchon.geometry import Mesh2D, TensorProductGrid

# FEM mesh (unstructured, handles complex domains)
mesh = Mesh2D(
    domain_type="rectangle",
    bounds=(0, 1, 0, 1),
    holes=[{"type": "circle", "center": (0.5, 0.5), "radius": 0.2}],
    mesh_size=0.05
)
mesh_data = mesh.generate_mesh()

# Regular grid (structured, efficient for HJB)
grid = TensorProductGrid(bounds=(0, 1, 0, 1), resolution=(50, 50))

# Dual geometry problem (works immediately!)
problem = MFGProblem(
    hjb_geometry=grid,
    fp_geometry=mesh,
    time_domain=(1.0, 50),
    sigma=0.1
)

# Projections use nearest neighbor fallback automatically
assert problem.geometry_projector.hjb_to_fp_method == "interpolation"
assert problem.geometry_projector.fp_to_hjb_method == "nearest"
```

**Performance**: O(log N) via KD-tree, adequate for most applications.

**Accuracy**: Piecewise constant, sufficient for coarse estimates.

### Level 2: Optimized (Delaunay) - One-Time Setup

For higher accuracy, register specialized Delaunay interpolation:

```python
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import KDTree
from mfgarchon.geometry import ProjectionRegistry, Mesh2D, TensorProductGrid
import numpy as np

# Register once (e.g., in your project setup script)
@ProjectionRegistry.register(Mesh2D, TensorProductGrid, "fp_to_hjb")
def mesh_to_grid_delaunay(mesh_geo, grid_geo, mesh_values, **kwargs):
    """Project from FEM mesh to grid using Delaunay interpolation."""

    # Get mesh vertices
    vertices = mesh_geo.get_spatial_grid()  # (N_vertices, dimension)

    # Create Delaunay interpolator (respects triangulation)
    interpolator = LinearNDInterpolator(vertices, mesh_values)

    # Evaluate at grid points
    grid_points = grid_geo.get_spatial_grid()
    grid_values_flat = interpolator(grid_points)

    # Handle extrapolation (points outside mesh → nearest neighbor)
    nan_mask = np.isnan(grid_values_flat)
    if np.any(nan_mask):
        tree = KDTree(vertices)
        _, nearest_indices = tree.query(grid_points[nan_mask])
        grid_values_flat[nan_mask] = mesh_values[nearest_indices]

    # Reshape to grid
    return grid_values_flat.reshape(grid_geo.get_grid_shape())

@ProjectionRegistry.register(TensorProductGrid, Mesh2D, "hjb_to_fp")
def grid_to_mesh_interpolation(grid_geo, mesh_geo, grid_values, **kwargs):
    """Project from grid to mesh (bilinear interpolation)."""
    vertices = mesh_geo.get_spatial_grid()
    interpolator = grid_geo.get_interpolator()
    return interpolator(grid_values, vertices)

# Now all subsequent MFGProblems use optimized projections
problem = MFGProblem(hjb_geometry=grid, fp_geometry=mesh, ...)
assert problem.geometry_projector.fp_to_hjb_method == "registry"  # Uses Delaunay!
```

**Performance**: O(N log N) setup, then O(log N) per query.

**Accuracy**: Linear (C⁰ continuous), respects mesh triangulation.

## Comparison: Methods

| Method | Continuity | Accuracy | Speed | Dependencies | Use Case |
|:-------|:-----------|:---------|:------|:-------------|:---------|
| **Nearest Neighbor** | Piecewise constant | Lower | Fast | None | Prototyping, coarse |
| **Delaunay** | C⁰ (linear) | Higher | Moderate | scipy | Production, accuracy |
| **Custom** | Problem-specific | Varies | Varies | Varies | Specialized needs |

## Use Cases

### Use Case 1: Complex Domain with Obstacles

**Problem**: Building evacuation with complex floor plan.

**Solution**: FEM mesh naturally handles obstacles.

```python
# Complex domain (L-shaped building with rooms)
mesh = Mesh2D(
    domain_type="polygon",
    vertices=[(0,0), (2,0), (2,1), (1,1), (1,2), (0,2)],  # L-shape
    holes=[
        {"type": "circle", "center": (0.5, 0.5), "radius": 0.2},  # Column
        {"type": "circle", "center": (1.5, 0.5), "radius": 0.2}   # Column
    ],
    mesh_size=0.05
)

# Regular grid for fast HJB
grid = TensorProductGrid(bounds=mesh.get_bounds(), resolution=(100, 100))

problem = MFGProblem(hjb_geometry=grid, fp_geometry=mesh, ...)
```

### Use Case 2: Adaptive Refinement

**Problem**: Need fine mesh near boundaries, coarse elsewhere.

**Note**: AMR implementation was removed in v0.16.5. For adaptive mesh refinement,
use external libraries directly:
- **pyAMReX**: Block-structured AMR with GPU support
- **Clawpack/AMRClaw**: Hyperbolic PDEs
- **pyAMG**: Mesh adaptation for complex geometries

```python
# For now, use manually refined meshes:
from mfgarchon.geometry import Mesh2D, TensorProductGrid

# Create a fine mesh near boundaries manually
mesh = Mesh2D(
    domain_type="rectangle",
    bounds=(0, 1, 0, 1),
    mesh_size=0.02  # Fine resolution throughout
)

grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx_points=[50, 50])

problem = MFGProblem(hjb_geometry=grid, fp_geometry=mesh, ...)
```

### Use Case 3: CAD Import

**Problem**: Real-world geometry from CAD file.

**Solution**: Import CAD, mesh it, use with dual geometry.

```python
# Import CAD file (e.g., STEP, IGES format)
mesh = Mesh2D(
    domain_type="cad_import",
    cad_file="floor_plan.step",
    mesh_size=0.05
)

grid = TensorProductGrid(bounds=mesh.get_bounds(), resolution=(100, 100))

problem = MFGProblem(hjb_geometry=grid, fp_geometry=mesh, ...)
```

## Implementation Details

### How Mesh Projection Works

**FEM Mesh Structure:**
- Mesh has `N_vertices` vertices at positions `(x_i, y_i)`
- Mesh has `N_elements` triangular elements connecting vertices
- Solutions stored at vertices (FEM nodal values)

**Grid → Mesh (HJB → FP):**
1. Get mesh vertex positions: `mesh.get_spatial_grid()` → (N_vertices, 2)
2. Use grid's bilinear interpolator to evaluate at vertex positions
3. Return interpolated values at vertices

**Mesh → Grid (FP → HJB):**
1. **Nearest Neighbor** (fallback):
   - Build KD-tree from mesh vertices
   - For each grid point, find nearest vertex
   - Assign that vertex's value

2. **Delaunay** (optimal):
   - Use mesh triangulation as Delaunay triangulation
   - Linear interpolation within each triangle
   - Nearest neighbor for points outside mesh

### Why This Works

**Key Insight**: `UnstructuredMesh.get_spatial_grid()` returns vertex positions, which is exactly what we need for interpolation!

```python
# All mesh types have this method
vertices = mesh.get_spatial_grid()  # (N_vertices, dimension)

# This works for:
# - Mesh2D (triangular)
# - Mesh3D (tetrahedral)
# - Any custom mesh implementing UnstructuredMesh
```

## Performance Tips

### Tip 1: Choose Mesh Resolution Wisely

```python
# Coarse mesh for FP (smooth density)
mesh_size = 0.1  # ~100 vertices

# Fine grid for HJB (sharp gradients)
grid_resolution = (100, 100)  # 10,201 points

# Projection overhead: negligible (< 1% solve time)
```

### Tip 2: Cache Projector for Repeated Use

```python
projector = problem.geometry_projector

# First call: builds KD-tree (one-time cost)
M_grid_1 = projector.project_fp_to_hjb(M_mesh_1)

# Subsequent calls: reuse cached tree (fast)
M_grid_2 = projector.project_fp_to_hjb(M_mesh_2)
M_grid_3 = projector.project_fp_to_hjb(M_mesh_3)
```

### Tip 3: Use Appropriate Method for Problem

```python
# Smooth solutions (linear PDE): Delaunay interpolation
# Discontinuous solutions (shocks): Nearest neighbor adequate
# Conservation required: Custom conservative projection
```

## Examples

### Complete Example

See `examples/advanced/dual_geometry_fem_mesh.py` for:
- Basic nearest neighbor projection
- Registering specialized Delaunay projectors
- Accuracy comparison
- Complex domain with holes

### Run Example

```bash
# Requires: gmsh, meshio, scipy
pip install gmsh meshio scipy

# Run example
python examples/advanced/dual_geometry_fem_mesh.py
```

## Future Enhancements

### Conservative Projection (Planned)

For problems requiring strict mass conservation:

```python
@ProjectionRegistry.register(Mesh2D, TensorProductGrid, "fp_to_hjb")
def mesh_to_grid_conservative(mesh_geo, grid_geo, mesh_values, **kwargs):
    """Conservative L2 projection preserving integral."""
    # Compute mass matrix overlap integrals
    # Solve M_grid * u_grid = M_mesh * u_mesh
    # Guarantees: ∫ u_grid = ∫ u_mesh
    pass
```

### High-Order Interpolation (Planned)

For high-accuracy requirements:

```python
@ProjectionRegistry.register(Mesh2D, TensorProductGrid, "fp_to_hjb")
def mesh_to_grid_quadratic(mesh_geo, grid_geo, mesh_values, **kwargs):
    """Quadratic interpolation using element shape functions."""
    # Use P2 finite element basis functions
    # Requires mid-edge values (not just vertices)
    pass
```

## FAQ

### Q: Do I need gmsh for FEM meshes?

**A**: Yes for mesh generation, but not for projection. If you already have mesh data (e.g., from external mesher), you can load it directly.

### Q: Can I use 3D FEM meshes?

**A**: Yes! `Mesh3D` works the same way with tetrahedral elements.

```python
from mfgarchon.geometry import Mesh3D, TensorProductGrid

mesh3d = Mesh3D(domain_type="box", bounds=(0,1,0,1,0,1), mesh_size=0.1)
grid3d = TensorProductGrid(dimension=3, bounds=[(0,1), (0,1), (0,1)], Nx_points=[31, 31, 31])

problem = MFGProblem(hjb_geometry=grid3d, fp_geometry=mesh3d, ...)
```

### Q: What about mesh → mesh projections?

**A**: Currently uses nearest neighbor fallback. For production, register specialized projection respecting both triangulations.

### Q: How accurate is nearest neighbor?

**A**: Error is O(h) where h is mesh size. For h=0.05, expect ~5% relative error for smooth functions. Delaunay reduces this to O(h²).

### Q: Can I reverse the pairing (mesh HJB + grid FP)?

**A**: Yes, but typically less efficient. Grid-based HJB is usually faster than mesh-based HJB, so standard pairing (grid HJB + mesh FP) is recommended.

## Summary

**FEM mesh support in dual geometry system**:

✅ **Basic support**: Works immediately via nearest neighbor fallback
✅ **Optimized support**: Easy one-time registration of Delaunay projectors
✅ **Production-ready**: Suitable for complex domains, obstacles, CAD import
✅ **Flexible**: Registry pattern allows custom projections for specialized needs

**Recommendation**:
1. Start with basic support for rapid prototyping
2. Register Delaunay projectors for production accuracy
3. Validate projection error for your specific problem
4. Consider custom conservative projectors if mass conservation is critical

---

**Document Version**: 1.0
**Related**: `docs/user_guide/dual_geometry_usage.md`, `examples/advanced/dual_geometry_fem_mesh.py`

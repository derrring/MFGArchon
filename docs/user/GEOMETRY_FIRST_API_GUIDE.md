# Geometry-First API Guide

**Status**: ✅ Complete as of v0.16.0
**Date**: 2025-12-11

---

## Overview

The geometry-first API is the **recommended way** to construct MFG problems in MFGArchon. Instead of manually specifying grid parameters in `MFGProblem`, you first create a geometry object and pass it to `MFGProblem`.

As of v0.16.0, `MFGProblem.geometry` is **always non-None** after initialization, and all spatial information is derived from the geometry object.

> **Removed (BREAKING):** The legacy 1D read-properties (`xmin`, `xmax`, `Nx`, `Lx`, `dx`, `xSpace`) were removed in v0.20.1 (Issue #1360), and the legacy 1D constructor kwargs (`xmin=`, `xmax=`, `Nx=`, `Lx=`) plus the `get_u_final` / `get_u_fin` aliases were removed in the following minor release (Issue #1363). Passing a removed kwarg now raises `ValueError`; accessing a removed attribute or alias raises `AttributeError`. Use the geometry-first API below.

## Key Benefits

1. **Type Safety**: Geometry objects are validated at construction time
2. **Single Source of Truth**: All spatial information derived from `problem.geometry`
3. **Reusability**: Same geometry can be used for multiple problems
4. **Clarity**: Separation of spatial discretization from temporal/diffusion parameters
5. **Flexibility**: Supports all geometry types through unified protocol
6. **Helper Properties**: `problem.is_cartesian`, `problem.is_network`, `problem.is_implicit` for type dispatch

## Available Geometry Types

| Geometry Class | Type | Description | Dimensions |
|:---------------|:-----|:------------|:-----------|
| `TensorProductGrid` | CARTESIAN_GRID | Structured regular grid | 1D-nD |
| `Domain1D` | DOMAIN_1D | 1D domain with BC | 1D |
| `Domain2D`, `Domain3D` | UNSTRUCTURED_MESH | Unstructured mesh via Gmsh | 2D, 3D |
| `Hyperrectangle` | IMPLICIT | Box domain via SDF | nD |
| `Hypersphere` | IMPLICIT | Sphere/ball via SDF | nD |
| `Grid` (mazes) | MAZE | Maze-based grid | 2D |
| `NetworkGeometry` | NETWORK | Graph-based network | 2D |

## Quick Start Examples

### 1. TensorProductGrid (Structured Grid)

**New API** (recommended):
```python
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.core import MFGProblem

# Create 2D grid: [0,10] × [0,5]
geometry = TensorProductGrid(
    dimension=2,
    bounds=[(0.0, 10.0), (0.0, 5.0)],
    Nx_points=[101, 51]
)

# Create problem
problem = MFGProblem(geometry=geometry, T=1.0, Nt=100, sigma=0.1)
```

**Old API** (deprecated, still works):
```python
problem = MFGProblem(
    spatial_bounds=[(0.0, 10.0), (0.0, 5.0)],
    spatial_discretization=[101, 51],
    T=1.0,
    Nt=100,
    sigma=0.1
)
```

### 2. Domain1D (1D with Boundary Conditions)

**New API**:
```python
from mfgarchon.geometry import Domain1D
from mfgarchon.core import MFGProblem

# Create 1D periodic domain
domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions="periodic")
domain.create_grid(Nx=101)

problem = MFGProblem(geometry=domain, T=1.0, Nt=100, sigma=0.1)
```

**Removed API** (raises `ValueError` since Issue #1363):
```python
# NO LONGER SUPPORTED -- raises ValueError pointing to the geometry-first API:
#   problem = MFGProblem(xmin=[0.0], xmax=[1.0], Nx=[101], T=1.0, Nt=100, sigma=0.1)
#
# Replace with a TensorProductGrid. NOTE the off-by-one: Nx counts intervals,
# TensorProductGrid takes Nx_points = Nx + 1 grid points.
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

grid = TensorProductGrid(
    bounds=[(0.0, 1.0)],
    Nx_points=[102],  # Nx=101 intervals -> 102 points
    boundary_conditions=no_flux_bc(dimension=1),
)
problem = MFGProblem(geometry=grid, T=1.0, Nt=100, sigma=0.1)
```

### 3. Implicit Domains (Meshfree)

**Hyperrectangle** (box domain via signed distance function):
```python
from mfgarchon.geometry.implicit import Hyperrectangle
from mfgarchon.core import MFGProblem
import numpy as np

# Create [0,1]² box
geometry = Hyperrectangle(bounds=np.array([[0, 1], [0, 1]]))

problem = MFGProblem(geometry=geometry, T=1.0, Nt=100, sigma=0.1)
```

**Hypersphere** (sphere via signed distance function):
```python
from mfgarchon.geometry.implicit import Hypersphere
from mfgarchon.core import MFGProblem

# Create unit sphere centered at origin
geometry = Hypersphere(center=[0, 0], radius=1.0)

problem = MFGProblem(geometry=geometry, T=1.0, Nt=100, sigma=0.1)
```

### 4. Maze Geometry

```python
from mfgarchon.geometry.graph import PerfectMazeGenerator
from mfgarchon.core import MFGProblem

# Generate 10×10 maze
maze_gen = PerfectMazeGenerator(rows=10, cols=10)
geometry = maze_gen.generate()

problem = MFGProblem(geometry=geometry, T=1.0, Nt=100, sigma=0.1)
```

### 5. High-Dimensional Grids

```python
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.core import MFGProblem

# Create 4D grid (will emit performance warning)
geometry = TensorProductGrid(
    dimension=4,
    bounds=[(0.0, 1.0)] * 4,
    Nx_points=[10, 10, 10, 10]  # 10^4 = 10,000 points
)

problem = MFGProblem(geometry=geometry, T=1.0, Nt=100, sigma=0.1)
```

## GeometryProtocol

All geometry objects implement the `GeometryProtocol`:

```python
from typing import Protocol
from numpy.typing import NDArray
from mfgarchon.geometry.protocol import GeometryType

class GeometryProtocol(Protocol):
    """Protocol for geometry objects usable in MFGProblem."""

    @property
    def dimension(self) -> int:
        """Spatial dimension (1, 2, 3, ...)."""
        ...

    @property
    def geometry_type(self) -> GeometryType:
        """Type of geometry (CARTESIAN_GRID, IMPLICIT, etc.)."""
        ...

    @property
    def num_spatial_points(self) -> int:
        """Total number of discrete spatial points."""
        ...

    def get_spatial_grid(self) -> NDArray:
        """Get spatial grid representation as (N, dimension) array."""
        ...

    def get_bounds(self) -> tuple[NDArray, NDArray] | None:
        """Get (min_bounds, max_bounds) arrays, or None if unbounded."""
        ...
```

---

## Accessing Spatial Information (v0.16.0+)

**Modern pattern** - Use geometry methods:

```python
from mfgarchon import MFGProblem
from mfgarchon.geometry import TensorProductGrid

# Create problem with geometry-first API
domain = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[51])
problem = MFGProblem(geometry=domain, T=1.0, Nt=10)

# Access spatial information via geometry (RECOMMENDED)
bounds = problem.geometry.get_bounds()        # (min_array, max_array)
grid = problem.geometry.get_spatial_grid()    # Spatial grid points
num_points = problem.geometry.num_spatial_points  # Total grid points
dim = problem.geometry.dimension              # Spatial dimension

# Helper properties for type dispatch
if problem.is_cartesian:
    # TensorProductGrid-specific logic
    spacing = problem.geometry.spacing  # Available on TensorProductGrid
elif problem.is_network:
    # NetworkGeometry-specific logic
    pass
elif problem.is_implicit:
    # Implicit domain-specific logic
    pass
```

**Removed pattern** - Legacy attributes raise `AttributeError` (Issue #1360):

```python
# These were removed in v0.20.1 -- accessing them raises AttributeError.
x_min = problem.xmin      # Use problem.geometry.get_bounds()[0][0] instead
x_max = problem.xmax      # Use problem.geometry.get_bounds()[1][0] instead
grid_size = problem.Nx    # Use problem.geometry.num_spatial_points - 1 instead
spacing = problem.dx      # Compute from geometry or use TensorProductGrid.spacing
grid = problem.xSpace     # Use problem.geometry.get_spatial_grid() instead
```

### Migration Table

| Legacy Attribute | Modern Equivalent |
|:-----------------|:------------------|
| `problem.xmin` | `problem.geometry.get_bounds()[0][0]` |
| `problem.xmax` | `problem.geometry.get_bounds()[1][0]` |
| `problem.Nx` | `problem.geometry.num_spatial_points - 1` |
| `problem.dx` | `problem.geometry.spacing[0]` (TensorProductGrid) |
| `problem.xSpace` | `problem.geometry.get_spatial_grid()` |
| `problem.Lx` | `bounds[1][0] - bounds[0][0]` where `bounds = problem.geometry.get_bounds()` |

## Migration Strategy

### Phase 1: Update Examples (v0.10.0)
- Convert `examples/basic/` to use geometry-first API
- Keep old API examples in `examples/legacy/` for reference

### Phase 2: Deprecation Period (v0.10.x - v0.99.x)
- Old API emits `DeprecationWarning`
- Both APIs fully functional
- Update documentation to show new API first

### Phase 3: Restriction (v1.0.0+)
- Old manual grid construction restricted (requires explicit flag)
- Geometry-first API becomes mandatory for new code

## Testing Your Migration

Run this script to test geometry-first API with your geometry type:

```python
#!/usr/bin/env python
"""Test geometry-first API."""
import warnings
from mfgarchon.core import MFGProblem
from mfgarchon.geometry import TensorProductGrid  # or your geometry

# Suppress deprecation warnings (we're using new API)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Create geometry
geometry = TensorProductGrid(
    dimension=2,
    bounds=[(0.0, 1.0), (0.0, 1.0)],
    Nx_points=[51, 51]
)

# Create problem
problem = MFGProblem(geometry=geometry, T=1.0, Nt=10, sigma=0.1)

# Verify
assert problem.dimension == 2
assert problem.num_spatial_points == 2601
assert problem.domain_type == "grid"

print("✓ Geometry-first API working correctly!")
```

## Advanced: Custom Geometry Types

You can create custom geometry types by implementing `GeometryProtocol`:

```python
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from mfgarchon.geometry.protocol import GeometryType

@dataclass
class MyCustomGeometry:
    """Custom geometry implementing GeometryProtocol."""

    points: NDArray  # Your spatial discretization
    dim: int

    @property
    def dimension(self) -> int:
        return self.dim

    @property
    def geometry_type(self) -> GeometryType:
        return GeometryType.CUSTOM

    @property
    def num_spatial_points(self) -> int:
        return len(self.points)

    def get_spatial_grid(self) -> NDArray:
        return self.points

# Use it
geometry = MyCustomGeometry(
    points=np.random.rand(100, 2),
    dim=2
)

problem = MFGProblem(geometry=geometry, T=1.0, Nt=10, sigma=0.1)
```

## Common Patterns

### Reusing Geometry

```python
# Create geometry once
geometry = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx_points=[51, 51])

# Use for multiple problems with different parameters
problem1 = MFGProblem(geometry=geometry, T=1.0, Nt=10, sigma=0.1)
problem2 = MFGProblem(geometry=geometry, T=2.0, Nt=20, sigma=0.2)
```

### Geometry Refinement

```python
# Create coarse grid
coarse = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx_points=[11, 11])

# Refine by factor of 2
fine = coarse.refine(factor=2)  # Now 21×21

# Use refined grid
problem = MFGProblem(geometry=fine, T=1.0, Nt=10, sigma=0.1)
```

### Composing Geometries

```python
# Create multiple domains for multi-population problems
domain1 = Hyperrectangle(bounds=np.array([[0, 1], [0, 1]]))
domain2 = Hypersphere(center=[2, 0], radius=0.5)

# Use in multi-population problem (future feature)
# problem = MultiPopulationMFG(geometries=[domain1, domain2], ...)
```

## Performance Considerations

### High-Dimensional Grids
- **Grid-based** (`TensorProductGrid`): O(N^d) memory/computation
  - Practical limit: d ≤ 3 for dense grids
  - For d > 3: Use coarse grids or sparse methods

- **Implicit domains** (`Hyperrectangle`, `Hypersphere`): O(N) memory
  - No mesh generation required
  - Suitable for high dimensions with meshfree solvers

### Mesh Generation
- **Structured grids** (`TensorProductGrid`): Instant creation
- **Unstructured meshes** (`Domain2D/3D`): Requires Gmsh (can be slow)
- **Implicit domains**: No mesh generation (SDF-based)

## FAQ

**Q: Do I need to update my code immediately?**
A: The legacy 1D geometry surfaces (`xmin`/`xmax`/`Nx`/`Lx` read-properties and
constructor kwargs, and the `get_u_final`/`get_u_fin` aliases) have been removed
(Issues #1360, #1363) after their 3-minor-version deprecation window. Code that
still uses them now raises; migrate to the geometry-first API shown above. Other
deprecated surfaces continue to emit `DeprecationWarning`.

**Q: Will my old code break?**
A: Code using the removed 1D geometry kwargs/attributes/aliases will raise. The
`spatial_bounds=` / `spatial_discretization=` n-D grid path and the no-argument
default still work.

**Q: What if I need manual grid construction?**
A: Use `TensorProductGrid` - it provides the same flexibility with better type safety.

**Q: Can I mix old and new APIs?**
A: Yes, but not recommended. Use one pattern consistently per project.

**Q: When should I use implicit domains vs structured grids?**
A: Use implicit domains for:
  - High dimensions (d > 3)
  - Complex shapes (circles, arbitrary SDFs)
  - Meshfree methods (particle-based solvers)

Use structured grids for:
  - Low dimensions (d ≤ 3)
  - Rectangular domains
  - Finite difference methods

## See Also

- `docs/api/geometry_protocol.md` - Full protocol specification
- `examples/basic/geometry_demo.py` - Complete working examples
- `CLAUDE.md` - Repository conventions and standards

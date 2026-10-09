# Geometry Module

Comprehensive geometry and mesh generation support for Mean Field Games.

## Overview

The geometry module provides domain definitions, boundary conditions, and mesh generation capabilities for 2D and 3D MFG problems through the **Gmsh → Meshio → PyVista** pipeline.

## Module Organization

Each entry is the first line of that package's or module's own docstring.

### Subpackages

- **`amr/`** - Adaptive Mesh Refinement (AMR) - API stub for future library integration.
- **`boundary/`** - Boundary condition management for MFG problems.
- **`graph/`** - Graph-based geometries for MFG problems.
- **`grids/`** - Cartesian grid geometries for MFG problems.
- **`implicit/`** - Geometry Infrastructure for Meshfree MFG Methods
- **`level_set/`** - Level Set Methods for Free Boundary Problems in MFG.
- **`meshes/`** - Unstructured mesh geometries for MFG problems.
- **`protocols/`** - Geometry trait protocols for capability-based interface design.

### Top-level modules

- `base.py` - Unified geometry base classes for MFGarchon.
- `cloud_geodesic.py` - Geodesic distance on meshfree clouds with obstacle-aware edge filtering.
- `collocation.py` - Geometry-aware collocation point generation for meshfree methods.
- `masks.py` - Mask generation for boundary condition regions.
- `predicates.py` - Region predicate factories for geometry marking.
- `protocol.py` - Unified geometry protocol for MFG problems.
- `traits.py` - Geometry trait enums, and the one protocol a dispatcher reads.
- `visibility.py` - Visibility queries for meshfree methods in domains with obstacles.

### Import Patterns

```python
# Public API imports (recommended)
from mfgarchon.geometry import (
    TensorProductGrid,  # Unified nD Cartesian grids (1D, 2D, 3D, nD)
    BaseNetworkGeometry,
    GridNetwork,
    MeshData,
    MeshVisualizationMode,
)

# Direct submodule imports (when needed)
from mfgarchon.geometry.grids import TensorProductGrid
from mfgarchon.geometry.graph import BaseNetworkGeometry
from mfgarchon.geometry.meshes import MeshData
from mfgarchon.geometry.boundary import BoundaryConditions, no_flux_bc
```

## Domain Types

### 2D Domains (`Domain2D`)

- **Rectangle**: Axis-aligned rectangular domains
- **Circle**: Circular domains
- **Polygon**: Arbitrary polygonal domains (including non-convex)
- **Custom**: User-defined geometries via Gmsh geometry objects

### 3D Domains (`Domain3D`)

- **Box**: Axis-aligned box domains
- **Sphere**: Spherical domains
- **Cylinder**: Cylindrical domains
- **Polyhedron**: Arbitrary polyhedral domains (including non-convex)
- **Custom**: User-defined 3D geometries via Gmsh geometry objects

## Lipschitz Bounded Domains

The `polygon` (2D) and `polyhedron` (3D) domain types support **arbitrary Lipschitz continuous boundaries**, including non-convex domains.

### Example: L-Shaped Domain (Non-Convex)

```python
import numpy as np
from mfgarchon.geometry import Domain2D

# Define L-shaped domain vertices
vertices = np.array([
    [0.0, 0.0], [1.0, 0.0], [1.0, 0.5],
    [0.5, 0.5], [0.5, 1.0], [0.0, 1.0]
])

domain = Domain2D(
    domain_type="polygon",
    vertices=vertices,
    mesh_size=0.05
)

# Generate mesh
mesh = domain.generate_mesh()
```

### Example: Star-Shaped Domain

```python
# Define star-shaped domain
theta = np.linspace(0, 2*np.pi, 10, endpoint=False)
r = 1.0 + 0.3 * np.sin(5 * theta)  # 5-pointed star
vertices = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

domain = Domain2D(
    domain_type="polygon",
    vertices=vertices,
    mesh_size=0.1
)
```

## Unbounded Domains

For problems on unbounded domains (e.g., $\mathbb{R}^d$, half-spaces), users must apply **computational truncation** or **domain transformation** techniques.

### Strategy 1: Large Bounded Domain with Truncation

```python
from mfgarchon.geometry import Domain2D
from mfgarchon.geometry.boundary_conditions_2d import NeumannBC2D, BoundaryConditionManager2D

# Truncate R^2 to large box [-L, L]^2
L = 100.0
domain = Domain2D(
    domain_type="rectangle",
    bounds=(-L, L, -L, L),
    mesh_size=0.5
)

# Add absorbing or transparent boundary conditions
manager = BoundaryConditionManager2D()
absorbing_bc = NeumannBC2D(0.0, name="Absorbing")  # or custom transparent BC
for edge in range(4):
    manager.add_condition(absorbing_bc, edge)
```

**Notes:**
- Choose $L$ large enough that boundary effects are negligible
- Verify solution decay near boundaries
- Document truncation radius in analysis

### Strategy 2: Domain Transformation

Map unbounded domain to bounded domain via analytic transformation:

```python
import numpy as np
from mfgarchon.geometry import Domain2D

# Example: Map R^2 to disk via transformation
# x = r * cos(θ), y = r * sin(θ)
# where r = tan(ρ * π/2) for ρ ∈ [0, 1)

# Define unit disk domain
domain = Domain2D(
    domain_type="circle",
    center=(0.0, 0.0),
    radius=1.0,
    mesh_size=0.05
)

# User applies transformation in problem setup
def transform_from_unbounded(r_unbounded):
    """Map R^+ to [0, 1) via r = tan(ρ * π/2)."""
    return (2.0 / np.pi) * np.arctan(r_unbounded)

def transform_to_unbounded(rho_bounded):
    """Map [0, 1) to R^+ via r = tan(ρ * π/2)."""
    return np.tan(rho_bounded * np.pi / 2.0)
```

**Common Transformations:**
- **Half-space** $\mathbb{R}^+$: $x = \log(\xi / (1-\xi))$ for $\xi \in (0,1)$
- **Whole space** $\mathbb{R}$: $x = \tan(\pi(\xi - 0.5))$ for $\xi \in (0,1)$
- **Stereographic projection**: Maps $\mathbb{R}^d$ to $\mathbb{S}^d \setminus \{point\}$

### Strategy 3: Localized Solutions

For problems with localized solutions, use **adaptive mesh refinement** near support:

```python
# Create large domain with fine mesh near origin
domain = Domain2D(
    domain_type="rectangle",
    bounds=(-50.0, 50.0, -50.0, 50.0),
    mesh_size=1.0  # Coarse far from origin
)

# User implements mesh refinement callback (future feature)
# or uses Gmsh fields for size control
```

## Bounding Box Property

The `bounds` property returns an **axis-aligned bounding box** (AABB), not the actual domain geometry:

```python
domain = Domain2D(
    domain_type="circle",
    center=(0.0, 0.0),
    radius=1.0
)

min_coords, max_coords = domain.bounds
# Returns: ([-1.0, -1.0], [1.0, 1.0])
# This is the bounding box, NOT the circle itself
```

**Usage:**
- Spatial indexing (quadtree/octree structures)
- Algorithm initialization (grid spacing, etc.)
- Visualization bounds

The actual domain geometry is preserved in the mesh representation generated by Gmsh.

## Boundary Conditions

The geometry module provides comprehensive boundary condition support:

### 2D Boundary Conditions

- **DirichletBC2D**: Fixed values $u = g(x,y,t)$
- **NeumannBC2D**: Flux conditions $\nabla u \cdot n = g(x,y,t)$
- **RobinBC2D**: Mixed conditions $\alpha u + \beta \nabla u \cdot n = g(x,y,t)$
- **PeriodicBC2D**: Periodic conditions $u(x_1,y) = u(x_2,y)$

### 3D Boundary Conditions

Analogous classes for 3D: `DirichletBC3D`, `NeumannBC3D`, `RobinBC3D`, `PeriodicBC3D`

### MFG-Specific Handlers

- **MFGBoundaryHandler2D**: HJB and Fokker-Planck equation boundary conditions
- **MFGBoundaryHandler3D**: 3D MFG boundary condition management

## Mesh Generation Pipeline

The module uses the standard computational geometry pipeline:

1. **Gmsh**: Geometry definition and mesh generation
2. **Meshio**: Format conversion and I/O
3. **PyVista**: Visualization and post-processing

### Example: Complete Workflow

```python
from mfgarchon.geometry import Domain2D
from mfgarchon.geometry.boundary_conditions_2d import create_rectangle_boundary_conditions

# 1. Define geometry
domain = Domain2D(
    domain_type="rectangle",
    bounds=(0.0, 1.0, 0.0, 1.0),
    mesh_size=0.05
)

# 2. Generate mesh
mesh = domain.generate_mesh()

# 3. Setup boundary conditions
bc_manager = create_rectangle_boundary_conditions(
    domain_bounds=(0.0, 1.0, 0.0, 1.0),
    condition_type="dirichlet_zero"
)

# 4. Apply to system matrix and RHS (in solver)
# matrix_mod, rhs_mod = bc_manager.apply_all_conditions(matrix, rhs, mesh)
```

## Advanced Features

### Domains with Holes

```python
# 2D domain with circular hole
outer_vertices = np.array([[0,0], [2,0], [2,2], [0,2]])
hole_center = (1.0, 1.0)
hole_radius = 0.3

domain = Domain2D(
    domain_type="polygon",
    vertices=outer_vertices,
    holes=[(hole_center, hole_radius)]
)
```

### Multi-Region Domains

```python
# Define multiple regions with different material properties
# (Implementation uses Gmsh physical groups)
```

### Mesh Quality Metrics

```python
# Compute mesh quality
mesh = domain.generate_mesh()

quality = mesh.quality_metrics
print(f"Min element quality: {quality['min_quality']}")
print(f"Average element quality: {quality['avg_quality']}")
```

## References

- **Gmsh Documentation**: https://gmsh.info/doc/texinfo/gmsh.html
- **Meshio Documentation**: https://github.com/nschloe/meshio
- **PyVista Documentation**: https://docs.pyvista.org/

## See Also

### Module References

See [Module Organization](#module-organization).

### Examples

- `examples/basic/geometry/`: Simple geometry demonstrations
- `examples/tutorials/`: Step-by-step geometry tutorials

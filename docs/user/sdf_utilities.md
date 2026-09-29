# Signed Distance Function (SDF) Utilities

**Feature**: Convenient functions for computing and manipulating signed distance functions
**Added**: v0.9.0 (Issue #216, Part 2/4)
**Module**: `mfgarchon.utils.numerical.sdf_utils`

---

## Overview

SDF utilities provide simple function-based APIs for working with signed distance functions (SDFs), commonly needed in:
- Obstacle avoidance problems
- Constrained MFG domains
- Boundary condition specification
- Level set methods and visualization

These utilities wrap the full `mfgarchon.geometry.implicit` infrastructure with simpler function-based APIs for quick prototyping.

**Convention**: `φ(x) < 0` inside, `φ(x) = 0` on boundary, `φ(x) > 0` outside — the domain polarity. An obstacle's SDF uses the opposite one; see [`CONVENTIONS.md`](CONVENTIONS.md) § *Geometry*.

---

## Quick Start

```python
from mfgarchon.utils import sdf_sphere, sdf_box, sdf_union, sdf_difference
import numpy as np

# Distance to unit sphere
points = np.array([[0, 0], [1, 0], [2, 0]])
dist = sdf_sphere(points, center=[0, 0], radius=1.0)
# [-1.0, 0.0, 1.0] (inside, on boundary, outside)

# Distance to unit box
dist_box = sdf_box(points, bounds=[[-1, 1], [-1, 1]])

# Box with circular hole (difference)
box = sdf_box(points, bounds=[[-1, 1], [-1, 1]])
hole = sdf_sphere(points, center=[0, 0], radius=0.5)
domain = sdf_difference(box, hole)
```

---

## Basic Primitives

### `sdf_sphere()`: Distance to Sphere/Ball

Compute signed distance to hypersphere (interval in 1D, circle in 2D, ball in 3D, etc.).

#### 1D Example (Interval)

```python
import numpy as np
from mfgarchon.utils import sdf_sphere

# 1D interval [0, 1] = sphere centered at 0.5 with radius 0.5
points = np.array([0.0, 0.5, 1.0, 1.5])
dist = sdf_sphere(points, center=[0.5], radius=0.5)
# [0.0, -0.5, 0.0, 0.5] (on left edge, at center, on right edge, outside)
```

#### 2D Example (Circle)

```python
points = np.array([[0, 0], [1, 0], [2, 0]])
dist = sdf_sphere(points, center=[0, 0], radius=1.0)
# [-1.0, 0.0, 1.0] (at center, on boundary, outside)
```

#### 3D Example (Ball)

```python
points = np.random.uniform(-2, 2, (100, 3))
dist = sdf_sphere(points, center=[0, 0, 0], radius=1.0)
# Negative inside unit ball, zero on surface, positive outside
```

---

### `sdf_box()`: Distance to Axis-Aligned Box

Compute signed distance to hyperrectangle.

#### 1D Example (Interval)

```python
points = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
dist = sdf_box(points, bounds=[[0, 1]])
# [1.0, 0.0, -0.5, 0.0, 1.0] (outside left, on left, inside, on right, outside right)
```

#### 2D Example (Rectangle)

```python
points = np.array([[0.5, 0.5], [0, 0], [2, 2]])
dist = sdf_box(points, bounds=[[0, 1], [0, 1]])
# [-0.5, 0.0, ~1.4] (center, corner, outside)
```

#### 3D Example (Box)

```python
points = np.array([[0.5, 0.5, 0.5], [-1, 0, 0]])
dist = sdf_box(points, bounds=[[0, 1], [0, 1], [0, 1]])
# [-0.5, 1.0] (inside, outside)
```

---

## CSG Operations

### Union: `sdf_union()`

Combine domains by taking minimum SDF (least restrictive).

```python
from mfgarchon.utils import sdf_union

# Two circles forming capsule-like shape
points = np.linspace(-2, 2, 100).reshape(-1, 1)
circle1 = sdf_sphere(points, center=[-0.5], radius=0.5)
circle2 = sdf_sphere(points, center=[0.5], radius=0.5)
union = sdf_union(circle1, circle2)

# Multiple shapes
points_2d = np.random.uniform(-2, 2, (1000, 2))
s1 = sdf_sphere(points_2d, center=[0, 0], radius=0.5)
s2 = sdf_sphere(points_2d, center=[1, 1], radius=0.5)
s3 = sdf_sphere(points_2d, center=[-1, -1], radius=0.5)
union = sdf_union(s1, s2, s3)
```

---

### Intersection: `sdf_intersection()`

Intersect domains by taking maximum SDF (most restrictive).

```python
from mfgarchon.utils import sdf_intersection

# Intersection of two overlapping circles (lens shape)
points = np.linspace(-2, 2, 100).reshape(-1, 1)
circle1 = sdf_sphere(points, center=[-0.3], radius=0.7)
circle2 = sdf_sphere(points, center=[0.3], radius=0.7)
intersection = sdf_intersection(circle1, circle2)

# Box with circular constraint
points_2d = np.random.uniform(-2, 2, (1000, 2))
box = sdf_box(points_2d, bounds=[[-1, 1], [-1, 1]])
circle = sdf_sphere(points_2d, center=[0, 0], radius=0.8)
constrained = sdf_intersection(box, circle)
```

---

### Complement: `sdf_complement()`

Reverse inside/outside by negating SDF.

```python
from mfgarchon.utils import sdf_complement

# Exterior of sphere (everything outside is now "inside")
points = np.array([[0, 0], [1, 0], [2, 0]])
sphere_dist = sdf_sphere(points, center=[0, 0], radius=1.0)
# [-1.0, 0.0, 1.0]

exterior = sdf_complement(sphere_dist)
# [1.0, 0.0, -1.0] (now exterior is negative = "inside")
```

---

### Difference: `sdf_difference()`

Remove one domain from another (A \ B).

```python
from mfgarchon.utils import sdf_difference

# Box with circular hole
points = np.random.uniform(-2, 2, (1000, 2))
box = sdf_box(points, bounds=[[-1, 1], [-1, 1]])
hole = sdf_sphere(points, center=[0, 0], radius=0.5)
domain = sdf_difference(box, hole)
# Points inside box AND outside hole have negative SDF

# Annulus (ring)
outer = sdf_sphere(points, center=[0, 0], radius=1.0)
inner = sdf_sphere(points, center=[0, 0], radius=0.5)
ring = sdf_difference(outer, inner)
```

---

## Smooth Blending

### Smooth Union: `sdf_smooth_union()`

Create smooth blend between shapes instead of sharp seam.

```python
from mfgarchon.utils import sdf_smooth_union

points = np.linspace(-2, 2, 100).reshape(-1, 1)
circle1 = sdf_sphere(points, center=[-0.5], radius=0.5)
circle2 = sdf_sphere(points, center=[0.5], radius=0.5)

# Sharp union
sharp = sdf_union(circle1, circle2)

# Smooth union (k controls smoothing radius)
smooth = sdf_smooth_union(circle1, circle2, smoothing=0.2)
```

**Smoothing parameter**: Larger values = smoother blend. Typical range: 0.01 to 0.5.

---

### Smooth Intersection: `sdf_smooth_intersection()`

Create smooth blend at intersection.

```python
from mfgarchon.utils import sdf_smooth_intersection

points = np.linspace(-2, 2, 100).reshape(-1, 1)
circle1 = sdf_sphere(points, center=[-0.3], radius=0.7)
circle2 = sdf_sphere(points, center=[0.3], radius=0.7)

smooth = sdf_smooth_intersection(circle1, circle2, smoothing=0.1)
```

**Note**: Uses polynomial smooth minimum from [Quilez (2008)](https://iquilezles.org/articles/smin/).

---

## Gradient Computation

### `sdf_gradient()`: Compute SDF Gradient

Compute ∇φ using finite differences. For exact SDFs, |∇φ| = 1.

```python
from mfgarchon.utils import sdf_gradient

# Gradient points outward from sphere center
points = np.array([[0.5, 0], [0, 0.5]])

def sphere_sdf(p):
    return sdf_sphere(p, center=[0, 0], radius=1.0)

grad = sdf_gradient(points, sphere_sdf)
# grad ≈ [[0.5, 0], [0, 0.5]] (normalized radial direction)

# Use for obstacle avoidance
obstacle_grad = sdf_gradient(agent_position, obstacle_sdf)
avoidance_force = -obstacle_grad  # Push away from obstacle
```

---

## Use Cases

### 1. Obstacle Avoidance in MFG

```python
from mfgarchon.utils import sdf_sphere, sdf_box, sdf_union, sdf_gradient

# Define obstacles
points = grid_points  # Your spatial grid
obstacle1 = sdf_sphere(points, center=[0.3, 0.5], radius=0.1)
obstacle2 = sdf_box(points, bounds=[[0.6, 0.8], [0.4, 0.6]])
obstacles = sdf_union(obstacle1, obstacle2)

# Gradient for obstacle avoidance
def obstacle_hamiltonian(x, p):
    dist = obstacles.reshape(x.shape[:-1])
    grad = sdf_gradient(x, lambda pts: obstacles)
    # Add penalty near obstacles
    penalty = np.exp(-10 * dist)
    return 0.5 * np.sum(p**2, axis=-1) + 100 * penalty
```

---

### 2. Domain Specification

```python
# L-shaped domain
box1 = sdf_box(points, bounds=[[0, 1], [0, 1]])
box2 = sdf_box(points, bounds=[[0, 0.5], [0, 0.5]])
l_shape = sdf_difference(box1, box2)

# Use as domain mask
inside_domain = l_shape < 0
```

---

### 3. Boundary Conditions

```python
# Neumann BC on circular boundary
boundary_mask = np.abs(obstacles) < grid_spacing
# Apply Neumann condition only where boundary_mask is True
```

---

### 4. Visualization

```python
import matplotlib.pyplot as plt

# Visualize SDF contours
x = y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y, indexing='ij')
points = np.stack([X, Y], axis=-1).reshape(-1, 2)

box = sdf_box(points, bounds=[[-1, 1], [-1, 1]])
hole = sdf_sphere(points, center=[0, 0], radius=0.5)
domain = sdf_difference(box, hole)
domain_grid = domain.reshape(X.shape)

plt.contour(X, Y, domain_grid, levels=[0], colors='black', linewidths=2)
plt.contourf(X, Y, domain_grid, levels=20, cmap='RdBu')
plt.colorbar(label='Signed Distance')
plt.title('Box with Circular Hole')
plt.axis('equal')
plt.show()
```

---

## API Reference

### Primitives

```python
sdf_sphere(
    points: ndarray,
    center: array_like,
    radius: float,
) -> ndarray

sdf_box(
    points: ndarray,
    bounds: array_like,  # shape (d, 2) or [(xmin, xmax), ...]
) -> ndarray
```

---

### CSG Operations

```python
sdf_union(*sdfs: ndarray) -> ndarray  # min(sdf1, sdf2, ...)

sdf_intersection(*sdfs: ndarray) -> ndarray  # max(sdf1, sdf2, ...)

sdf_complement(sdf: ndarray) -> ndarray  # -sdf

sdf_difference(sdf_a: ndarray, sdf_b: ndarray) -> ndarray  # A \ B
```

---

### Smooth Operations

```python
sdf_smooth_union(
    sdf_a: ndarray,
    sdf_b: ndarray,
    smoothing: float = 0.1,  # Larger = smoother
) -> ndarray

sdf_smooth_intersection(
    sdf_a: ndarray,
    sdf_b: ndarray,
    smoothing: float = 0.1,
) -> ndarray
```

---

### Gradient

```python
sdf_gradient(
    points: ndarray,
    sdf_func: callable,  # Function: points -> distances
    epsilon: float = 1e-5,  # Finite difference step
) -> ndarray  # shape: same as points
```

---

## Relationship to `mfgarchon.geometry.implicit`

SDF utilities provide simple function-based wrappers around the full object-oriented SDF system:

```python
# Full infrastructure (for complex cases)
from mfgarchon.geometry.implicit import Hypersphere, Hyperrectangle, UnionDomain

sphere = Hypersphere(center=[0, 0], radius=1.0)
box = Hyperrectangle(bounds=[[-1, 1], [-1, 1]])
domain = UnionDomain([sphere, box])
dist = domain.signed_distance(points)

# Utility functions (for quick prototyping)
from mfgarchon.utils import sdf_sphere, sdf_box, sdf_union

sphere_dist = sdf_sphere(points, center=[0, 0], radius=1.0)
box_dist = sdf_box(points, bounds=[[-1, 1], [-1, 1]])
dist = sdf_union(sphere_dist, box_dist)
```

Use utilities for quick prototyping, full infrastructure for reusable domain definitions.

---

## Performance Notes

- **Primitives**: O(N) where N = number of points
- **CSG operations**: O(N), just numpy min/max
- **Smooth operations**: O(N), vectorized operations
- **Gradient**: O(N × d) where d = dimension (central differences)

All operations are fully vectorized using numpy for efficiency.

---

## Examples

See `tests/unit/utils/test_sdf_utils.py` for comprehensive examples covering:
- 1D/2D/3D primitives
- CSG operations
- Smooth blending
- Gradient computation
- Edge cases

---

**Documentation Version**: 1.0
**Last Updated**: 2025-11-03
**MFGArchon Version**: v0.9.0+

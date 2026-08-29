# Multi-Dimensional Mean Field Games: User Guide

Complete guide for setting up and solving 2D and 3D Mean Field Game problems using MFGArchon's multi-dimensional infrastructure.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Tensor Product Grids](#tensor-product-grids)
4. [Sparse Matrix Operations](#sparse-matrix-operations)
5. [Multi-Dimensional Visualization](#multi-dimensional-visualization)
6. [Complete Workflows](#complete-workflows)
7. [Application Examples](#application-examples)
8. [Best Practices](#best-practices)

---

## Overview

MFGArchon's multi-dimensional framework enables efficient solution of Mean Field Games on 2D and 3D spatial domains with memory-efficient data structures, sparse linear algebra, and interactive visualizations.

### Key Features

- **Tensor Product Grids**: Memory-efficient O(∑Nᵢ) storage for d-dimensional grids
- **Sparse Operations**: CSR/CSC matrices for large-scale linear systems
- **Multi-Dimensional Visualization**: Interactive 3D plots, animations, and analysis
- **Proven Applications**: Traffic flow, portfolio optimization, epidemic modeling

### When to Use Multi-Dimensional MFG

**2D Problems**:
- Urban traffic routing with spatial congestion
- Financial markets with (wealth, allocation) state space
- Epidemic spread across geographic regions
- Crowd dynamics in 2D venues

**3D Problems**:
- Drone/aerial vehicle navigation (x, y, z)
- Multi-asset portfolio management
- 3D crowd evacuation scenarios

---

## Quick Start

### Minimal 2D Example

```python
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.utils import SparseMatrixBuilder, SparseSolver
from mfgarchon.visualization import MultiDimVisualizer

# 1. Create 2D grid
grid = TensorProductGrid(
    dimension=2,
    bounds=[(0.0, 10.0), (0.0, 10.0)],  # [0,10] × [0,10]
    Nx_points=[51, 51]  # 51×51 grid
)

# 2. Build sparse Laplacian
builder = SparseMatrixBuilder(grid, matrix_format='csr')
L = builder.build_laplacian(boundary_conditions='neumann')

# 3. Solve simple diffusion
import numpy as np
X, Y = grid.meshgrid(indexing='ij')
u0 = np.exp(-((X-5)**2 + (Y-5)**2) / 2)  # Initial condition

# 4. Visualize
viz = MultiDimVisualizer(grid, backend='plotly')
fig = viz.surface_plot(u0, title='Initial Distribution',
                        xlabel='x', ylabel='y', zlabel='Density')
viz.save(fig, 'output.html')
```

**Output**: Interactive 3D surface plot in `output.html`

---

## Tensor Product Grids

Efficient structured grids for multi-dimensional problems.

### Creating Grids

```python
from mfgarchon.geometry import TensorProductGrid

# 2D Grid
grid_2d = TensorProductGrid(
    dimension=2,
    bounds=[(0.0, L_x), (0.0, L_y)],
    Nx_points=[Nx, Ny]
)

# 3D Grid
grid_3d = TensorProductGrid(
    dimension=3,
    bounds=[(0.0, L_x), (0.0, L_y), (0.0, L_z)],
    Nx_points=[Nx, Ny, Nz]
)
```

### Grid Properties

```python
# Dimension and size
print(f"Dimension: {grid.dimension}")  # 2 or 3
print(f"Total points: {grid.total_points()}")  # Nx * Ny * Nz

# Coordinate arrays (memory-efficient)
coords = grid.coordinates  # List of 1D arrays
print(f"Memory: {sum(len(c) for c in coords)} values")  # Nx + Ny + Nz

# Spacing
dx, dy = grid.spacing  # For 2D

# What each node actually owns. The grid is endpoint-inclusive, so the two end
# nodes of every axis own half a cell -- the measure is the trapezoid, not dx*dy.
w_x = grid.quadrature_weights(0)  # shape (Nx,); sums to Lx, not to Nx*dx
mass = grid.integrate(m)          # integral over the domain get_bounds() reports
```

### Meshgrid for Computations

```python
# 2D case
X, Y = grid.meshgrid(indexing='ij')
# Now X[i,j], Y[i,j] are spatial coordinates

# Define functions on grid
u = np.sin(np.pi * X / L_x) * np.sin(np.pi * Y / L_y)

# 3D case
X, Y, Z = grid.meshgrid(indexing='ij')
u = np.exp(-((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2))
```

### Index Mapping

```python
# Multi-index to flat index
i, j = 10, 15  # 2D indices
flat_idx = grid.get_index((i, j))

# Flat index to multi-index
multi_idx = grid.get_multi_index(flat_idx)
```

### Memory Efficiency

```python
# 2D: 100×100 grid
grid = TensorProductGrid(2, [(0, 1), (0, 1)], [100, 100])

# Traditional meshgrid: 100×100×2 = 20,000 values
# Tensor product: 100 + 100 = 200 values (100× less!)

# 3D: 50×50×50 grid
grid = TensorProductGrid(3, [(0, 1)]*3, [50, 50, 50])

# Traditional: 50³ × 3 = 375,000 values
# Tensor product: 50 + 50 + 50 = 150 values (2500× less!)
```

---

## Sparse Matrix Operations

Efficient sparse linear algebra for large-scale systems.

### Building Sparse Operators

```python
from mfgarchon.utils import SparseMatrixBuilder

builder = SparseMatrixBuilder(grid, matrix_format='csr')
```

### Laplacian Operator

```python
# Dirichlet BC: u = 0 on boundary
L_dirichlet = builder.build_laplacian(boundary_conditions='dirichlet')

# Neumann BC: ∂u/∂n = 0 on boundary (no-flux)
L_neumann = builder.build_laplacian(boundary_conditions='neumann')

# Properties
print(f"Matrix shape: {L.shape}")  # (N, N) where N = total_points()
print(f"Nonzeros: {L.nnz}")  # Much less than N²
print(f"Sparsity: {L.nnz / (N*N):.2%}")  # Typically < 1%
```

### Gradient Operators

```python
# 2D gradients
Gx = builder.build_gradient(direction=0, order=2)  # ∂/∂x
Gy = builder.build_gradient(direction=1, order=2)  # ∂/∂y

# 3D gradients
Gz = builder.build_gradient(direction=2, order=2)  # ∂/∂z

# Compute gradient of function u
u_flat = u.flatten()
du_dx = (Gx @ u_flat).reshape(grid.Nx_points)
du_dy = (Gy @ u_flat).reshape(grid.Nx_points)
```

### Solving Linear Systems

```python
from mfgarchon.utils import SparseSolver

# Direct solver (best for small-medium problems)
solver_direct = SparseSolver(method='direct')
u = solver_direct.solve(A, b)

# Iterative solvers (best for large problems)
solver_cg = SparseSolver(method='cg', tol=1e-8, max_iter=1000)
u = solver_cg.solve(A, b)  # For symmetric positive definite A

solver_gmres = SparseSolver(method='gmres', tol=1e-8, max_iter=1000)
u = solver_gmres.solve(A, b)  # For general A
```

### Choosing Solver Methods

| Problem Size | Matrix Type | Recommended Method |
|:-------------|:------------|:-------------------|
| < 10,000 DOF | Any | `direct` |
| 10,000 - 100,000 | SPD | `cg` |
| 10,000 - 100,000 | Non-SPD | `gmres` |
| > 100,000 | SPD | `cg` with preconditioner |
| > 100,000 | Non-SPD | `gmres` with preconditioner |

---

## Multi-Dimensional Visualization

Create publication-quality and interactive visualizations.

### Creating Visualizer

```python
from mfgarchon.visualization import MultiDimVisualizer

# Plotly backend (interactive HTML)
viz = MultiDimVisualizer(grid, backend='plotly', colorscale='Viridis')

# Matplotlib backend (static images)
viz_mpl = MultiDimVisualizer(grid, backend='matplotlib')
```

### Surface Plots (3D view of 2D data)

```python
# Basic surface plot
fig = viz.surface_plot(
    u,  # 2D array (Nx, Ny)
    title='Value Function u(x,y)',
    xlabel='x (km)',
    ylabel='y (km)',
    zlabel='Cost-to-go'
)
viz.save(fig, 'surface.html')
```

### Heatmaps (Top-down view)

```python
fig = viz.heatmap(
    density,
    title='Population Density m(x,y)',
    xlabel='x',
    ylabel='y'
)
viz.save(fig, 'heatmap.html')
```

### Contour Plots (Level sets)

```python
fig = viz.contour_plot(
    u,
    title='Value Function Contours',
    xlabel='x',
    ylabel='y',
    levels=20  # Number of contour lines
)
viz.save(fig, 'contours.html')
```

### Slice Plots (1D cross-sections)

```python
# Slice along y=constant (middle)
fig = viz.slice_plot(
    u,
    slice_dim=1,  # Slice along dimension 1 (y)
    slice_index=Ny//2,  # Middle index
    title='Value at y=5'
)
viz.save(fig, 'slice.html')
```

### Animations (Time evolution)

```python
# u_time has shape (Nt+1, Nx, Ny)
fig = viz.animation(
    u_time,
    title='Density Evolution m(t,x,y)',
    xlabel='x',
    ylabel='y',
    zlabel='Density',
    fps=10  # Frames per second
)
viz.save(fig, 'animation.html')
```

### Colorscale Options

- `'Viridis'` - General purpose (default)
- `'Reds'` - For infection/heat maps
- `'Blues'` - For population/water
- `'RdYlGn'` - For financial gains/losses
- `'Plasma'` - High contrast

---

## Complete Workflows

### 2D MFG Problem Template

```python
import numpy as np
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.utils import SparseMatrixBuilder, SparseSolver
from mfgarchon.visualization import MultiDimVisualizer

# === PROBLEM SETUP ===
L = 10.0  # Domain size
Nx = Ny = 51
T = 1.0  # Time horizon
Nt = 50

grid = TensorProductGrid(2, [(0, L), (0, L)], [Nx, Ny])
dt = T / Nt

# Parameters
sigma = 0.5  # Diffusion
lambda_param = 1.0  # Control cost

# === INITIALIZATION ===
X, Y = grid.meshgrid(indexing='ij')

# Initial density
m0 = np.exp(-((X - L/4)**2 + (Y - L/4)**2) / 2)
m0 /= grid.integrate(m0)

# Terminal cost
g = 0.5 * ((X - L/2)**2 + (Y - L/2)**2)

# === SPARSE OPERATORS ===
builder = SparseMatrixBuilder(grid, matrix_format='csr')
L_op = builder.build_laplacian(boundary_conditions='neumann')
Gx = builder.build_gradient(direction=0, order=2)
Gy = builder.build_gradient(direction=1, order=2)

# === SOLUTION ARRAYS ===
u = np.zeros((Nt+1, Nx, Ny))
m = np.zeros((Nt+1, Nx, Ny))

u[-1, :, :] = g  # Terminal condition
m[0, :, :] = m0  # Initial condition

# === FIXED-POINT ITERATION ===
for iteration in range(max_iter):
    # Backward HJB solve
    for n in range(Nt-1, -1, -1):
        # ... HJB step ...
        pass

    # Forward FP solve
    for n in range(Nt):
        # ... FP step ...
        pass

    # Check convergence
    if converged:
        break

# === VISUALIZATION ===
viz = MultiDimVisualizer(grid, backend='plotly')

# Multiple plots
fig_u = viz.surface_plot(u[-1], title='Value Function')
fig_m0 = viz.heatmap(m[0], title='Initial Density')
fig_mT = viz.heatmap(m[-1], title='Final Density')
fig_anim = viz.animation(m, title='Density Evolution')

viz.save(fig_u, 'value.html')
viz.save(fig_m0, 'density_initial.html')
viz.save(fig_mT, 'density_final.html')
viz.save(fig_anim, 'evolution.html')
```

---

## Application Examples

### Example 1: Traffic Flow (2D Routing)

See `examples/advanced/traffic_flow_2d_demo.py`

**Problem**: Optimal vehicle routing in 20km × 20km urban network

**Key Features**:
- Destination: City center
- Cost: Travel time + congestion
- Output: 6 interactive visualizations

**Run**:
```bash
python examples/advanced/traffic_flow_2d_demo.py
```

### Example 2: Portfolio Optimization (2D State-Control)

See `examples/advanced/portfolio_optimization_2d_demo.py`

**Problem**: Wealth allocation between bonds and stocks with market impact

**State Space**: (W, α) where W = wealth, α = stock fraction

**Output**: Value surfaces, indifference curves, distribution evolution

**Run**:
```bash
python examples/advanced/portfolio_optimization_2d_demo.py
```

### Example 3: Epidemic Modeling (2D Spatial Disease)

See `examples/advanced/epidemic_modeling_2d_demo.py`

**Problem**: Optimal mobility with infection risk and economic activity

**Dynamics**: Coupled population movement + SIR disease spread

**Output**: Infection maps, peak analysis, population response

**Run**:
```bash
python examples/advanced/epidemic_modeling_2d_demo.py
```

---

## Best Practices

### Grid Resolution

**2D Grids**:
- **Coarse**: 21×21 (quick prototyping)
- **Medium**: 51×51 (standard quality)
- **Fine**: 101×101 (high accuracy, slower)

**3D Grids**:
- **Coarse**: 11×11×11 (prototyping)
- **Medium**: 21×21×21 (standard)
- **Fine**: 41×41×41 (expensive!)

### Memory Considerations

```python
# 2D: 101×101 grid
# - Grid coordinates: 202 values
# - Solution u(t,x,y): (Nt+1) × 101 × 101 × 8 bytes
#   Example: 50 time steps = 50 × 101 × 101 × 8 = 4 MB

# 3D: 41×41×41 grid
# - Grid coordinates: 123 values
# - Solution u(t,x,y,z): (Nt+1) × 41³ × 8 bytes
#   Example: 20 time steps = 20 × 68,921 × 8 = 11 MB
```

### Computational Efficiency

1. **Use appropriate grid resolution**: Start coarse, refine as needed
2. **Iterative solvers for large systems**: CG/GMRES for > 10,000 unknowns
3. **Sparse matrix formats**: CSR for operations, CSC for column access
4. **Visualization backends**: Plotly for exploration, Matplotlib for publication

### Debugging Workflow

```python
# 1. Verify grid setup
print(f"Grid: {grid.Nx_points}, Total: {grid.total_points()}")
print(f"Spacing: {grid.spacing}")

# 2. Check sparse matrix properties
print(f"Laplacian: {L.shape}, NNZ: {L.nnz}, Sparsity: {L.nnz/(N*N):.4%}")

# 3. Visualize intermediate results
viz = MultiDimVisualizer(grid, backend='plotly')
fig = viz.heatmap(m[Nt//2], title=f'Density at t={T/2}')
viz.save(fig, 'debug_halfway.html')

# 4. Monitor convergence
for iteration in range(max_iter):
    # ... solver iteration ...
    print(f"Iter {iteration}: Δu = {u_change:.6f}, Δm = {m_change:.6f}")
```

### Common Pitfalls

❌ **Don't**: Create full meshgrids for storage
```python
# Bad: Wastes memory
X_full = np.outer(x_coords, np.ones_like(y_coords))
Y_full = np.outer(np.ones_like(x_coords), y_coords)
```

✅ **Do**: Use tensor product grid's meshgrid only when needed
```python
# Good: Generate on-the-fly
X, Y = grid.meshgrid(indexing='ij')
u = f(X, Y)  # Compute immediately
```

❌ **Don't**: Use dense matrices for large grids
```python
# Bad: O(N²) memory
L_dense = L.toarray()  # Huge for large N!
```

✅ **Do**: Keep sparse format
```python
# Good: O(NNZ) memory
u = SparseSolver(method='cg').solve(L, b)  # Direct sparse solve
```

---

## Further Reading

- **Examples**: `examples/advanced/` directory for complete applications
- **Tests**: `tests/integration/test_multidim_workflow.py` for usage patterns
- **API Reference**: `docs/reference/` for detailed API documentation
- **Theory**: See MFG literature references in examples for mathematical formulations

---

*Last Updated: 2025-10-06*
*MFGArchon Version: 1.4.0+*

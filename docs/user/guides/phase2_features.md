# Phase 2 Features Guide

**New in v0.8**: Simplified interfaces, performance optimizations, and utility enhancements.

This guide covers all Phase 2 improvements:
1. **`problem.solve()`** - One-line high-level interface
2. **QP Utilities** - Caching and warm-starting for quadratic programming
3. **Particle Interpolation** - Grid ↔ particle conversion utilities
4. **Geometry Utilities** - Simplified obstacle and domain creation

---

## 1. High-Level `problem.solve()` Interface (Phase 2.3)

> **`solve_mfg()` does not exist** (#1741). This section taught `from mfgarchon import solve_mfg`,
> which raises `ImportError`. The one-line interface is `problem.solve()` — the three-mode API of
> #580 — and `create_solver()`, the other name once suggested here, is itself deprecated since
> 0.17.0 and points to the same place. Every example below was run before being written down.

### Overview

`problem.solve()` provides a one-line interface for solving MFG problems with automatic configuration and method selection.

**With explicit configuration**:
```python
from mfgarchon import MFGProblem
from mfgarchon.config import MFGSolverConfig

problem = MFGProblem()
config = MFGSolverConfig(picard={"max_iterations": 100, "tolerance": 1e-5})

result = problem.solve(config=config, verbose=True)
```

`MFGSolverConfig` is nested (`picard`, `hjb`, `fp`, `backend`, `logging`); the flat
`config.max_iterations = ...` this block used raises
`ValueError: "MFGSolverConfig" object has no field`.

**After (`problem.solve()` - 1 line)**:
```python
from mfgarchon import MFGProblem

problem = MFGProblem()
result = problem.solve()
```

### Default Parameters

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `max_iterations` | 100 | Maximum fixed-point iterations |
| `tolerance` | 1e-6 | Convergence tolerance |
| `verbose` | True | Show solver progress |

### Custom Parameters

Override defaults as needed:

```python
result = problem.solve(
    max_iterations=200,      # More iterations
    tolerance=1e-8,          # Tighter tolerance
    verbose=True
)
```

### Backend Integration

Backend parameter accepts both strings and objects:

```python
# String (auto-converted to backend object)
# `backend` is NOT a parameter of `solve()`; it lives in the config's backend section.
from mfgarchon.config import MFGSolverConfig

result = problem.solve(config=MFGSolverConfig(backend={"type": "numpy"}))
result = problem.solve(config=MFGSolverConfig(backend={"type": "jax"}))
result = problem.solve(config=MFGSolverConfig(backend={"type": "torch"}))

# Or use backend objects directly
from mfgarchon.backends import create_backend
backend = create_backend("numpy")
result = problem.solve(config=MFGSolverConfig(backend={"type": "numpy"}))
```

### Result Structure

Returns `SolverResult` with:

```python
result = problem.solve()

# Solution arrays
result.U  # Value function (Nt+1, Nx+1) or (Nt+1, Nx+1, Ny+1)
result.M  # Density (Nt+1, Nx+1) or (Nt+1, Nx+1, Ny+1)

# Convergence information
result.converged        # bool: Convergence status
result.iterations       # int: Number of iterations
result.error_history_U  # list[float]: U error per iteration
result.error_history_M  # list[float]: M error per iteration

# Metadata
result.execution_time   # float: Solve time in seconds
result.solver_name      # str: Solver used
result.metadata         # dict: Additional info
```

### When to Use

**Use `problem.solve()` for**:
- Quick prototyping and exploration
- Standard MFG problems
- Getting started with the library
- One-off solves with automatic configuration

**Use Factory API for**:
- Custom solver configurations
- Fine-grained control over HJB/FP methods
- Research method comparisons
- Advanced tuning and diagnostics

### API Reference

The signature below is read off `MFGProblem.solve` rather than transcribed, and every
parameter named here exists.

```python
def solve(
    self,
    Nt: int | None = None,
    max_iterations: int | None = None,
    tolerance: float | None = None,
    verbose: bool | None = None,
    config: MFGSolverConfig | None = None,
    scheme: NumericalScheme | None = None,
    hjb_solver: BaseHJBSolver | None = None,
    fp_solver: BaseFPSolver | None = None,
) -> SolverResult:
```

**Parameters**:
- `Nt`: Number of time steps (default: the problem's own)
- `max_iterations`: Max fixed-point iterations
- `tolerance`: Convergence tolerance
- `verbose`: Print progress
- `config`: `MFGSolverConfig`, whose sections are `picard`, `hjb`, `fp`, `backend`, `logging`
- `scheme`: Safe Mode — e.g. `NumericalScheme.FDM_UPWIND`, which picks an adjoint-consistent pair
- `hjb_solver` / `fp_solver`: Expert Mode — supply both, and the pairing is yours to justify

There is no `method` preset, no `resolution`, and no `**kwargs`; the old block advertised all
three. A backend goes through `config`, not a keyword — `problem.solve(backend="jax")` raises.

**Returns**: `SolverResult` with U, M, convergence info, error histories

**Example**: See `examples/basic/three_mode_api_demo.py`, which `create_solver`'s own deprecation note points at.

---

## 2. QP Utilities (Phase 2.2)

### Overview

Performance optimizations for quadratic programming operations used in particle-based FP solvers.

**Features**:
- **Result Caching**: Hash-based cache with LRU eviction (~9× speedup)
- **Warm-Starting**: Reuse previous solutions for iterative solves (~4% improvement)

### QPCache: Result Caching

**How It Works**:
1. Hash QP problem parameters (A, b, W, bounds) using SHA256
2. Store result in LRU cache (max_size entries)
3. Return cached result on cache hit

**Usage**:
```python
from mfgarchon.utils.numerical.qp_utils import QPCache, QPSolver

# Create cache with size limit
cache = QPCache(max_size=100)

# Create solver with cache
solver = QPSolver(backend="auto", enable_warm_start=False, cache=cache)

# Solve QP problems
for iteration in range(num_iterations):
    result = solver.solve_weighted_least_squares(A, b, W, bounds=bounds)

# Check cache statistics
print(f"Cache hits: {cache.hits}")
print(f"Cache misses: {cache.misses}")
print(f"Hit rate: {cache.hit_rate:.1%}")
```

**Performance**:
- **9× speedup** for repeated identical problems
- 90% hit rate on recurring subproblems
- Negligible overhead on cache misses

**When to Use**:
- Problems with recurring QP substructure
- Iterative algorithms with repeated subproblems
- Spatial grid points with similar problems

**When NOT to Use**:
- RHS changes every iteration (no cache hits)
- Single-solve scenarios
- Memory-constrained environments

### Warm-Starting

**How It Works**:
1. Store previous QP solution (x, y dual variables)
2. Use as initial guess for next solve with same structure
3. Reduces solver iterations (~4% speedup for small problems)

**Usage**:
```python
from mfgarchon.utils.numerical.qp_utils import QPSolver

# Create solver with warm-starting enabled
solver = QPSolver(backend="osqp", enable_warm_start=True, cache=None)

# Solve sequence of related problems
for iteration in range(num_iterations):
    b = compute_rhs(iteration)  # RHS changes

    # Use same point_id to trigger warm-starting
    result = solver.solve_weighted_least_squares(
        A, b, W, bounds=bounds, point_id=0
    )

# Check warm-start statistics
print(f"Cold starts: {solver.stats['cold_starts']}")
print(f"Warm starts: {solver.stats['warm_starts']}")
```

**Performance**:
- **4% speedup** for small problems (50×50)
- More effective for large QP problems (>1000 variables)
- Requires backend support (OSQP ✅, scipy ❌)

**Important**: Use consistent `point_id` for related problems to enable warm-starting.

### Combined Caching + Warm-Starting

```python
from mfgarchon.utils.numerical.qp_utils import QPCache, QPSolver

# Create cache and solver
cache = QPCache(max_size=100)
solver = QPSolver(backend="osqp", enable_warm_start=True, cache=cache)

# MFG scenario: Solve at each spatial grid point across time iterations
for time_iteration in range(num_time_steps):
    for point_id, (A, W, bounds) in enumerate(spatial_problems):
        b = compute_rhs(time_iteration, point_id)

        # Caching checks for identical (A, b, W, bounds)
        # Warm-starting reuses solution from previous time step
        result = solver.solve_weighted_least_squares(
            A, b, W, bounds=bounds, point_id=point_id
        )
```

**Realistic MFG Performance**:
- Warm-starting active but no cache hits (RHS changes)
- Benefits depend on problem size and solver backend
- Monitor cache hit rate to assess effectiveness

### Benchmarks

Run comprehensive performance benchmarks:
```bash
python benchmarks/qp_caching_benchmark.py
```

**Benchmark Scenarios**:
1. Cache performance (10 unique, 100 total solves) → 9× speedup
2. Warm-start performance (50 iterations) → 1.04× speedup
3. Combined MFG scenario (20 grid points, 50 iterations) → 0.98× (no cache hits)

**Findings**:
- Caching highly effective for repeated identical problems
- Warm-starting minimal benefit for small QP problems with OSQP
- MFG scenarios typically don't benefit from caching (RHS changes)

### API Reference

```python
class QPCache:
    def __init__(self, max_size: int = 1000):
        """LRU cache for QP results."""

    @property
    def hits(self) -> int:
        """Number of cache hits."""

    @property
    def misses(self) -> int:
        """Number of cache misses."""

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0-1)."""

class QPSolver:
    def __init__(
        self,
        backend: str = "auto",
        enable_warm_start: bool = True,
        cache: QPCache | None = None,
    ):
        """QP solver with caching and warm-starting."""

    def solve_weighted_least_squares(
        self,
        A: np.ndarray,
        b: np.ndarray,
        W: np.ndarray,
        bounds: list[tuple[float, float]] | None = None,
        point_id: int | None = None,
    ) -> np.ndarray:
        """Solve weighted least squares QP problem."""

    @property
    def stats(self) -> dict[str, int]:
        """Solver statistics (cold_starts, warm_starts)."""
```

**Example**: See `benchmarks/qp_caching_benchmark.py` for performance analysis.

---

## 3. Particle Interpolation Utilities (Phase 2.2)

### Overview

Utilities for converting between grid-based and particle-based representations used in hybrid MFG solvers.

**Features**:
- Grid → Particles: Interpolate grid values to particle positions
- Particles → Grid: Convert particle ensemble to grid density
- Multiple interpolation methods: linear, cubic, KDE

### Grid → Particles

**Use Case**: Initialize particle ensemble from grid-based density.

```python
from mfgarchon.utils.numerical.particle_interpolation import grid_to_particles
import numpy as np

# Grid-based density
x_grid = np.linspace(0, 1, 101)
density_grid = np.exp(-50 * (x_grid - 0.5)**2)  # Gaussian
density_grid /= np.trapz(density_grid, x_grid)  # Normalize

# Convert to particles
particles, weights = grid_to_particles(
    x_grid=x_grid,
    density_grid=density_grid,
    num_particles=1000,
    method="linear"  # or "cubic"
)

# Result: particles ~ density_grid distribution
# weights: all equal (1/num_particles) for uniform sampling
```

**Parameters**:
- `x_grid`: Grid points (1D array)
- `density_grid`: Density values at grid points
- `num_particles`: Number of particles to generate
- `method`: Interpolation method ("linear" or "cubic")

**Returns**:
- `particles`: Particle positions (num_particles,)
- `weights`: Particle weights (num_particles,) - typically uniform

### Particles → Grid (KDE)

**Use Case**: Convert particle ensemble to smooth grid-based density.

```python
from mfgarchon.utils.numerical.particle_interpolation import particles_to_grid_kde
import numpy as np

# Particle ensemble
particles = np.random.randn(1000) * 0.1 + 0.5  # Gaussian around 0.5
weights = np.ones(1000) / 1000  # Equal weights

# Convert to grid using KDE
x_grid = np.linspace(0, 1, 101)
density_grid = particles_to_grid_kde(
    particles=particles,
    weights=weights,
    x_grid=x_grid,
    bandwidth=0.05  # KDE bandwidth
)

# Verify normalization
mass = np.trapz(density_grid, x_grid)  # Should be ≈ 1.0
```

**Parameters**:
- `particles`: Particle positions (num_particles,)
- `weights`: Particle weights (num_particles,)
- `x_grid`: Target grid points
- `bandwidth`: KDE bandwidth (controls smoothness)

**Returns**:
- `density_grid`: Density values at grid points (normalized)

**Bandwidth Selection**:
- Small bandwidth (0.01): Sharp, noisy
- Medium bandwidth (0.05): Balanced (recommended)
- Large bandwidth (0.1): Smooth, over-smoothed

### 2D Support

Both functions support 2D problems:

```python
# 2D Grid → Particles
particles_x, particles_y, weights = grid_to_particles_2d(
    x_grid=x_grid,
    y_grid=y_grid,
    density_grid=density_2d,  # Shape: (len(x_grid), len(y_grid))
    num_particles=5000,
    method="linear"
)

# 2D Particles → Grid
density_2d = particles_to_grid_kde_2d(
    particles_x=particles_x,
    particles_y=particles_y,
    weights=weights,
    x_grid=x_grid,
    y_grid=y_grid,
    bandwidth=0.05
)
```

### Integration with Solvers

Particle interpolation is used internally by hybrid solvers:

```python
from mfgarchon.types import NumericalScheme

# The default pairing is HJB-FDM + FP-Particle (hybrid)
result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)

# Internally:
# 1. HJB solve on grid → value function U
# 2. Optimal controls computed from ∇U
# 3. FP solve: Grid → Particles → Advect → Grid (via KDE)
# 4. Repeat until convergence
```

### API Reference

```python
def grid_to_particles(
    x_grid: np.ndarray,
    density_grid: np.ndarray,
    num_particles: int,
    method: Literal["linear", "cubic"] = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert grid density to particle ensemble."""

def particles_to_grid_kde(
    particles: np.ndarray,
    weights: np.ndarray,
    x_grid: np.ndarray,
    bandwidth: float = 0.05,
) -> np.ndarray:
    """Convert particle ensemble to grid density using KDE."""
```

**Example**: See `examples/basic/utility_demo.py` for demonstrations.

---

## 4. Geometry Utilities (Phase 2.2)

### Overview

Simplified aliases for creating obstacles and domains in MFG problems.

**Before (verbose)**:
```python
from mfgarchon.geometry.obstacles import RectangularObstacle, CircularObstacle
from mfgarchon.geometry.domain import Union, Intersection, Difference

rect = RectangularObstacle(center=(0.5, 0.5), width=0.2, height=0.1)
circle = CircularObstacle(center=(0.3, 0.3), radius=0.1)
combined = Union([rect, circle])
```

**After (aliases)**:
```python
from mfgarchon.utils.geometry import Rectangle, Circle, Union, Intersection, Difference

rect = Rectangle(center=(0.5, 0.5), width=0.2, height=0.1)
circle = Circle(center=(0.3, 0.3), radius=0.1)
combined = Union([rect, circle])
```

### Available Aliases

| Alias | Original | Description |
|:------|:---------|:------------|
| `Rectangle` | `RectangularObstacle` | Rectangular obstacle |
| `Circle` | `CircularObstacle` | Circular obstacle |
| `Union` | `Union` | CSG union (A ∪ B) |
| `Intersection` | `Intersection` | CSG intersection (A ∩ B) |
| `Difference` | `Difference` | CSG difference (A \ B) |

### CSG Operations

Combine obstacles using Constructive Solid Geometry:

```python
from mfgarchon.utils.geometry import Rectangle, Circle, Union, Difference

# Create two obstacles
rect = Rectangle(center=(0.5, 0.5), width=0.3, height=0.2)
circle = Circle(center=(0.6, 0.5), radius=0.15)

# Union: Combined obstacle (A ∪ B)
union = Union([rect, circle])

# Difference: Rectangle with circular hole (A \ B)
difference = Difference(rect, circle)

# Query signed distance function (SDF)
point = (0.5, 0.5)
distance = union.signed_distance(point)  # < 0 inside, > 0 outside
```

### SDF Queries

All geometry objects support signed distance function queries:

```python
# Check if point is inside obstacle
is_inside = obstacle.signed_distance(point) < 0

# Find closest boundary point
distance_to_boundary = abs(obstacle.signed_distance(point))

# Grid-based SDF computation
x = np.linspace(0, 1, 101)
y = np.linspace(0, 1, 101)
X, Y = np.meshgrid(x, y)

sdf_values = np.zeros_like(X)
for i in range(len(x)):
    for j in range(len(y)):
        sdf_values[i, j] = obstacle.signed_distance((X[i, j], Y[i, j]))
```

### Integration with MFG Problems

Use geometry utilities to define spatial domains and obstacles:

```python
from mfgarchon.core import BaseMFGProblem
from mfgarchon.utils.geometry import Rectangle, Circle, Union

class RoomEvacuationProblem(BaseMFGProblem):
    def __init__(self, Nx, Ny, Nt, T):
        super().__init__(Nx, Nt, T, dimension=2)

        # Define room with obstacles
        wall1 = Rectangle(center=(0.5, 0.3), width=0.6, height=0.05)
        wall2 = Rectangle(center=(0.5, 0.7), width=0.6, height=0.05)
        self.obstacles = Union([wall1, wall2])

    def is_obstacle(self, x, y):
        """Check if point is inside obstacle."""
        return self.obstacles.signed_distance((x, y)) < 0
```

### API Reference

```python
class Rectangle:
    def __init__(self, center: tuple[float, float], width: float, height: float):
        """Rectangular obstacle."""

    def signed_distance(self, point: tuple[float, float]) -> float:
        """Compute signed distance to obstacle boundary."""

class Circle:
    def __init__(self, center: tuple[float, float], radius: float):
        """Circular obstacle."""

    def signed_distance(self, point: tuple[float, float]) -> float:
        """Compute signed distance to obstacle boundary."""

class Union:
    def __init__(self, shapes: list):
        """CSG union of shapes (A ∪ B)."""

    def signed_distance(self, point: tuple[float, float]) -> float:
        """Minimum distance to any shape."""

class Intersection:
    def __init__(self, shapes: list):
        """CSG intersection of shapes (A ∩ B)."""

    def signed_distance(self, point: tuple[float, float]) -> float:
        """Maximum distance to all shapes."""

class Difference:
    def __init__(self, shape_a, shape_b):
        """CSG difference (A \ B)."""

    def signed_distance(self, point: tuple[float, float]) -> float:
        """Distance to A with B subtracted."""
```

**Example**: See `examples/basic/utility_demo.py` for geometry demonstrations.

---

## Summary

### Phase 2.2: Infrastructure Improvements
- **QP Utilities**: 9× speedup via caching, warm-starting support
- **Particle Interpolation**: Grid ↔ particle conversion for hybrid solvers
- **Geometry Utilities**: Simplified aliases for obstacles and CSG operations

### Phase 2.3: User Experience Improvements
- **`problem.solve()`**: One-line interface reducing 30 lines → 1 line
- **Method Presets**: Automatic configuration (auto/fast/accurate/research)
- **Backend Integration**: String-to-object auto-conversion

### When to Use

| Feature | Use Case |
|:--------|:---------|
| `problem.solve()` | Quick prototyping, standard problems, getting started |
| Factory API | Custom configs, fine control, research comparisons |
| QP Cache | Repeated identical QP problems |
| Warm-Starting | Large QP problems, iterative solves |
| Particle Interpolation | Hybrid solvers (internal use) |
| Geometry Utilities | Defining obstacles and domains |

### Next Steps

1. **Quick Start**: Try `problem.solve()` on example problems
2. **Performance**: Run QP benchmarks to understand speedups
3. **Advanced**: Use Factory API for custom solver configurations
4. **Research**: Combine utilities for novel MFG algorithms

---

**See Also**:
- [Quickstart Guide](../quickstart.md) - Getting started with `problem.solve()`
- [HJB Solver Selection Guide](../HJB_SOLVER_SELECTION_GUIDE.md) - Choosing the right solver
- [Examples](../../examples/) - Working demonstrations

**Examples**:
- `examples/basic/three_mode_api_demo.py` - High-level interface demonstrations
- `examples/basic/utility_demo.py` - Utility demonstrations
- `benchmarks/qp_caching_benchmark.py` - Performance benchmarks

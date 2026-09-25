# HJB Solver Selection Guide

**Date**: 2025-11-02
**Purpose**: Help users choose the right HJB solver configuration for their problems

---

## Quick Start

### For 1D Problems
```python
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

# Recommended: Newton solver (fast, robust)
solver = HJBFDMSolver(problem, solver_type="newton")
```

### For 2D/3D Problems
```python
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

# Recommended: Fixed-point with damping (much faster)
solver = HJBFDMSolver(
    problem,
    solver_type="fixed_point",
    damping_factor=0.7  # 0.5-0.8 recommended
)
```

---

## Performance Comparison

Based on comprehensive benchmarks (`benchmarks/hjb_solver_comparison.py`):

### 1D Performance

| Grid Size | Fixed-Point | Newton | Winner |
|:----------|:------------|:-------|:-------|
| 50 × 50   | 0.891s     | 0.495s | **Newton** (1.8x faster) |
| 100 × 50  | 0.188s     | 0.187s | Tie (within 1%) |
| 200 × 50  | 0.358s     | 0.361s | Tie (within 1%) |

**Recommendation**: Use **Newton** for 1D (slightly faster, better convergence)

### 2D Performance

| Grid Size | Fixed-Point | Newton | Winner |
|:----------|:------------|:-------|:-------|
| 10×10×10  | 0.087s     | 2.669s | **Fixed-Point** (30.5x faster) |
| 15×15×20  | 0.179s     | 12.907s | **Fixed-Point** (72.2x faster) |
| 20×20×20  | 0.320s     | 41.074s | **Fixed-Point** (128.2x faster) |

**Recommendation**: Use **Fixed-Point** for 2D/3D (orders of magnitude faster)

### Why the Difference?

**Newton's method** requires:
- Jacobian computation: O(N²) for N grid points
- Sparse linear solve: O(N^1.5) to O(N²)
- **1D**: N ≈ 100 → ~10K operations
- **2D**: N ≈ 400 → ~160K-1.6M operations

**Fixed-point** requires:
- Function evaluation: O(N)
- Damped update: O(N)
- **Scales linearly** with grid size

**Result**: Newton's quadratic cost dominates in 2D/3D

---

## Solver Type Selection

### Fixed-Point Iteration

```python
solver = HJBFDMSolver(
    problem,
    solver_type="fixed_point",
    damping_factor=0.7,          # ω ∈ (0,1], recommend 0.5-0.8
    max_newton_iterations=100,   # More iterations than Newton
    newton_tolerance=1e-6,
)
```

**Algorithm**: `u^{k+1} = (1-ω)u^k + ω·G(u^k)`

**When to Use**:
- ✅ 2D/3D/nD problems (much faster than Newton)
- ✅ Large grid sizes (N > 20 per dimension)
- ✅ Smooth problems
- ✅ When you don't have analytical Jacobian

**Tuning Parameters**:
- **damping_factor** (ω):
  - 1.0 = no damping (fastest but may oscillate)
  - 0.7-0.8 = good balance (recommended)
  - 0.5 = heavy damping (stable but slower)
- **max_iterations**: 50-100 typical (needs more than Newton)
- **tolerance**: 1e-6 to 1e-8

**Convergence**: Linear (error reduces by constant factor each iteration)

### Newton's Method

```python
solver = HJBFDMSolver(
    problem,
    solver_type="newton",
    max_newton_iterations=30,    # Fewer iterations than fixed-point
    newton_tolerance=1e-6,
)
```

**Algorithm**: Solves `J·δu = -F(u)` where J is Jacobian

**When to Use**:
- ✅ 1D problems (comparable or faster than fixed-point)
- ✅ Small 2D grids (N < 15 per dimension)
- ✅ Stiff problems (large gradients)
- ✅ When you need high accuracy quickly

**Features**:
- Automatic Jacobian via finite differences
- Sparse matrix solver
- Optional line search (not yet implemented)

**Convergence**: Quadratic (error squares each iteration)

**Limitations**:
- Expensive Jacobian computation in higher dimensions
- Needs good initial guess
- May diverge if problem is highly nonlinear

---

## Dimension-Specific Recommendations

### 1D Problems (d=1)

```python
# Recommended
solver = HJBFDMSolver(problem, solver_type="newton")
```

**Why**: Newton's Jacobian cost is acceptable, convergence is fast

**Alternative**: Use fixed-point if Newton diverges

### 2D Problems (d=2)

```python
# Recommended
solver = HJBFDMSolver(
    problem,
    solver_type="fixed_point",
    damping_factor=0.7
)
```

**Why**: Fixed-point is 30-128x faster for typical grid sizes

**When to use Newton**: Only for very small grids (N < 15) or if you have analytical Jacobian

### 3D Problems (d=3)

```python
# Strongly recommended
solver = HJBFDMSolver(
    problem,
    solver_type="fixed_point",
    damping_factor=0.6  # More stable for 3D
)
```

**Why**: Newton is prohibitively expensive

**Warning**: 3D problems have O(N³) grid points. Consider:
- Sparse grids
- Meshfree methods (GFDM)
- Particle methods

### High-Dimensional Problems (d > 3)

**FDM not recommended** due to curse of dimensionality

**Alternatives**:
- GFDM (meshfree, scales better)
- PINN (neural networks, dimension-independent)
- Semi-Lagrangian (characteristic-based)
- Particle methods

---

## Problem-Specific Guidelines

### Smooth Problems (Low-Gradient)

Examples: LQ games, quadratic Hamiltonians

```python
# Use fixed-point with aggressive damping
solver = HJBFDMSolver(
    problem,
    solver_type="fixed_point",
    damping_factor=0.8  # Can use higher ω
)
```

**Why**: Smooth problems converge well with fixed-point

### Stiff Problems (High-Gradient)

Examples: Thin transition layers, boundary layers

```python
# 1D: Use Newton
solver_1d = HJBFDMSolver(problem_1d, solver_type="newton")

# 2D: Use fixed-point with conservative damping
solver_2d = HJBFDMSolver(
    problem_2d,
    solver_type="fixed_point",
    damping_factor=0.5  # Lower ω for stability
)
```

### Nonsmooth Problems

Examples: Obstacles, state constraints, discontinuous Hamiltonians

```python
# Use fixed-point (more robust)
solver = HJBFDMSolver(
    problem,
    solver_type="fixed_point",
    damping_factor=0.6  # Conservative
)
```

**Why**: Newton may have trouble with non-smooth Jacobian

---

## Advanced Features

### Anderson Acceleration (MFG Level)

For the **outer Picard iteration** (not HJB timestep):

```python
from mfgarchon.alg.numerical.mfg_solvers import FixedPointIterator

mfg_solver = FixedPointIterator(
    problem, hjb_solver, fp_solver,
    use_anderson=True,      # Enable Anderson acceleration
    anderson_depth=5,       # Memory depth (3-10 typical)
    damping_factor=0.5,     # Still damp M for positivity
)
```

**What it does**:
- Accelerates **U** (value function) convergence
- Still uses damping on **M** (density) to preserve positivity and mass
- Typically 2-5x speedup on outer Picard iteration

**Implementation**: `mfgarchon/alg/numerical/mfg_solvers/fixed_point_iterator.py:260-272`

### Custom Jacobian (Advanced)

If you have analytical Jacobian:

```python
def custom_jacobian(U):
    """
    Compute Jacobian analytically.

    Returns:
        J: Sparse matrix (N×N)
    """
    # Your analytical Jacobian here
    return J_sparse

solver = HJBFDMSolver(problem, solver_type="newton")
# Then override jacobian_func attribute (advanced usage)
```

**Speedup**: Can make Newton competitive in 2D if Jacobian is cheap

---

## Troubleshooting

### Fixed-Point Not Converging

**Symptom**: Iterations exceed max without convergence

**Solutions**:
1. **Reduce damping**: Try ω = 0.5 or 0.3
2. **Increase max_iterations**: Try 200-500
3. **Check time step**: Reduce dt if CFL condition violated
4. **Try different initial guess**: Use previous solution

### Newton Diverging

**Symptom**: Residual grows instead of shrinking

**Solutions**:
1. **Better initial guess**: Use fixed-point solution as warmstart
2. **Switch to fixed-point**: More robust for difficult problems
3. **Reduce time step**: Smaller dt = more linear problem
4. **Check Hamiltonian**: Ensure it's smooth

### Slow Convergence (Both)

**Symptom**: Many iterations needed

**Solutions**:
1. **Check CFL condition**: dt too large causes oscillations
2. **Refine grid**: Better resolution may help
3. **Use Anderson** (for outer Picard, not HJB timestep)
4. **Check problem conditioning**: Rescale if needed

### Out of Memory (2D/3D Newton)

**Symptom**: Crashes during Jacobian assembly

**Solutions**:
1. **Switch to fixed-point**: Uses O(N) not O(N²) memory
2. **Reduce grid size**: Fewer points
3. **Use sparse=True**: Already default, but verify

---

## Decision Tree

```
Is problem 1D?
├─ YES → Use Newton
│         - Fast, robust
│         - Quadratic convergence
│
└─ NO (2D/3D/nD)
    │
    Is grid small (N < 15)?
    ├─ YES → Either solver OK
    │         - Newton faster initially
    │         - Fixed-point more robust
    │
    └─ NO (N ≥ 15)
        │
        Is problem smooth?
        ├─ YES → Fixed-point (ω=0.7-0.8)
        │         - Much faster than Newton
        │         - Good convergence
        │
        └─ NO (stiff/nonsmooth)
            └─ Fixed-point (ω=0.5-0.6)
                - More stable
                - Conservative damping
```

---

## Example Code

### Complete 1D Example

```python
from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

# Create problem
problem = MFGProblem(Nx=100, Nt=50, T=1.0)

# Create solver (Newton for 1D)
solver = HJBFDMSolver(
    problem,
    solver_type="newton",
    max_newton_iterations=30,
    newton_tolerance=1e-6,
)

# Solve
import numpy as np
M = np.ones((51, 101)) / 101
U_terminal = 0.5 * np.linspace(0, 1, 101)**2
U_guess = np.zeros((51, 101))

U_solution = solver.solve_hjb_system(M, U_terminal, U_guess)
print(f"Solved! Shape: {U_solution.shape}")
```

### Complete 2D Example

```python
from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
import numpy as np

# Define 2D problem using modern API
class My2DProblem(MFGProblem):
    def __init__(self):
        super().__init__(
            spatial_bounds=[(-1, 1), (-1, 1)],  # 2D domain
            spatial_discretization=[20, 20],     # Grid resolution
            T=1.0,
            Nt=20,
            sigma=0.01,
        )

    def hamiltonian(self, t, x, p, m):
        return 0.5 * np.sum(p**2) + 0.5 * np.sum(x**2)

    def terminal_cost(self, x):
        return 0.5 * np.sum(x**2, axis=1)

    def initial_density(self, x):
        return np.exp(-5 * np.sum(x**2, axis=1))

    def running_cost(self, t, x, m):
        return 0.5 * np.sum(x**2, axis=1)

    def setup_components(self):
        pass

# Create problem and solver
problem = My2DProblem()
solver = HJBFDMSolver(
    problem,
    solver_type="fixed_point",  # Much faster for 2D!
    damping_factor=0.7,
    max_newton_iterations=100,
    newton_tolerance=1e-6,
)

# Solve
M = np.ones((21, 20, 20)) / 400
x = np.linspace(-1, 1, 20)
X, Y = np.meshgrid(x, x, indexing='ij')
U_terminal = 0.5 * (X**2 + Y**2)
U_guess = np.zeros((21, 20, 20))

U_solution = solver.solve_hjb_system(M, U_terminal, U_guess)
print(f"Solved! Shape: {U_solution.shape}")
```

---

## Performance Tips

### 1. Grid Resolution

**Trade-off**: Accuracy vs Speed

```python
# Coarse (fast, less accurate)
N = 20  # ~400 points in 2D

# Medium (balanced)
N = 30  # ~900 points in 2D

# Fine (slow, accurate)
N = 50  # ~2500 points in 2D
```

### 2. Time Steps

**Rule of thumb**: dt ≈ dx² / (2σ²) for stability

```python
# For grid spacing dx and diffusion σ
dt = (dx**2) / (4 * sigma**2)  # Conservative
Nt = int(T / dt)
```

### 3. Warm Starting

Use solution from coarser grid or previous run:

```python
# Solve on coarse grid first
solver_coarse = HJBFDMSolver(problem_coarse, solver_type="fixed_point")
U_coarse = solver_coarse.solve_hjb_system(M_coarse, U_term_coarse, U_guess_coarse)

# Interpolate to fine grid for initial guess
from scipy.interpolate import RegularGridInterpolator
# ... interpolate U_coarse to fine grid ...
U_guess_fine = U_coarse_interpolated

# Solve on fine grid with warm start
solver_fine = HJBFDMSolver(problem_fine, solver_type="fixed_point")
U_fine = solver_fine.solve_hjb_system(M_fine, U_term_fine, U_guess_fine)
```

### 4. Adaptive Damping

Start aggressive, then conservative:

```python
# First attempt
solver1 = HJBFDMSolver(problem, solver_type="fixed_point", damping_factor=0.8)
try:
    U1 = solver1.solve_hjb_system(M, U_term, U_guess)
except:
    # If diverges, retry with conservative damping
    solver2 = HJBFDMSolver(problem, solver_type="fixed_point", damping_factor=0.5)
    U2 = solver2.solve_hjb_system(M, U_term, U_guess)
```

---

## Benchmark Results Summary

From `benchmarks/hjb_solver_comparison.py`:

**1D Conclusion**: Newton and Fixed-Point are comparable
- Small grids: Newton slightly faster
- Large grids: Essentially tied
- **Recommendation**: Use Newton (simpler, fewer parameters)

**2D Conclusion**: Fixed-Point dominates
- All grid sizes: Fixed-Point 30-128x faster
- Gap widens with grid refinement
- **Recommendation**: Always use Fixed-Point for 2D

**3D Extrapolation**: Fixed-Point advantage even larger
- Newton Jacobian: O(N⁶) for 3D grid
- Fixed-Point: O(N³) for 3D grid
- **Recommendation**: Only use Fixed-Point for 3D

---

## Further Reading

- **Implementation**: `mfgarchon/alg/numerical/hjb_solvers/hjb_fdm.py`
- **Theory**: `docs/development/NONLINEAR_SOLVER_ARCHITECTURE.md`
- **Benchmarks**: `benchmarks/hjb_solver_comparison.py`
- **Tests**: `tests/integration/test_hjb_fdm_2d_validation.py`
- **Examples**: `examples/basic/` and `examples/advanced/`

---

**Last Updated**: 2025-11-02
**Version**: MFGArchon 0.8.1+

# MFGArchon Backend Usage Guide

## 🚀 **Quick Start: Choose Your Backend**

MFGArchon automatically selects the best available backend, but you can also manually choose based on your hardware and problem type.

### **One-Line Backend Selection**
```python
from mfgarchon.backends import create_backend

# Automatic optimal selection
backend = create_backend("auto")

# Manual selection
backend = create_backend("torch_mps")  # Apple Silicon
backend = create_backend("jax_gpu")    # NVIDIA GPU with JAX
backend = create_backend("numba")      # CPU optimization
```

## 🎯 **Backend Recommendations by Use Case**

### **🧠 Neural Methods (PINN, Neural Operators)**
```python
# BEST: PyTorch with automatic device selection
from mfgarchon.backends import create_backend
from mfgarchon.alg.neural_solvers import PINNSolver

backend = create_backend("torch")  # Auto-selects CUDA/MPS/CPU
solver = PINNSolver(problem, backend=backend)
result = solver.solve()
```
**Why PyTorch**: Native neural network support, automatic differentiation, optimal tensor operations.

### **📊 Mathematical Kernels (Pure Computation)**
```python
# BEST: JAX for mathematical operations
from mfgarchon.utils.acceleration.jax_utils import compute_hamiltonian
from mfgarchon.alg.mfg_solvers import JAXMFGSolver

solver = JAXMFGSolver(problem, use_gpu=True, jit_compile=True)
result = solver.solve()  # Automatic XLA compilation
```
**Why JAX**: Pure functional code, automatic vectorization, excellent for gradients.

### **🌳 Adaptive Mesh Refinement (AMR)**
```python
# AMR is planned for future integration with external libraries
# Recommended: pyAMReX, Clawpack/AMRClaw, pyAMG, p4est
# See mfgarchon.geometry.amr module for API stub
from mfgarchon.geometry.amr import create_amr_grid, AMRNotImplementedError

# Will raise AMRNotImplementedError with library recommendations
# try:
#     amr = create_amr_grid(...)
# except AMRNotImplementedError as e:
#     print(e)  # Suggests using pyAMReX directly
```
**Note**: AMR implementation was removed in v0.16.5. Use external libraries directly.

### **💾 Large-Scale Problems (Memory Constrained)**
```python
# BEST: Sparse operations with Numba parallelization
from mfgarchon.utils.performance_optimization import SparseMatrixOptimizer

optimizer = SparseMatrixOptimizer()
# Automatically uses Numba if available for parallel operations
laplacian = optimizer.create_laplacian_3d(Nx=200, Ny=200, Nz=200)
solution = optimizer.solve_sparse_system(laplacian, rhs, method="cg")
```

## 🔧 **Backend Configuration Examples**

### **PyTorch Backend Configuration**
```python
from mfgarchon.backends import TorchBackend

# Apple Silicon (M1/M2/M3)
backend = TorchBackend(
    device="mps",              # Metal Performance Shaders
    precision="float32",       # Optimal for MPS
    memory_efficient=True      # Reduce memory fragmentation
)

# NVIDIA CUDA
backend = TorchBackend(
    device="cuda",
    precision="float16",       # Mixed precision for speed
    memory_efficient=False     # More memory for speed
)

# CPU (development/debugging)
backend = TorchBackend(
    device="cpu",
    precision="float64",       # High precision for debugging
    num_threads=4              # Control CPU threading
)
```

### **JAX Backend Configuration**
```python
from mfgarchon.alg.mfg_solvers import JAXMFGSolver

# GPU configuration
solver = JAXMFGSolver(
    problem,
    use_gpu=True,              # Use GPU if available
    jit_compile=True,          # JIT compilation for speed
    finite_diff_order=4,       # Higher-order accuracy
    adaptive_timestep=True     # Dynamic time stepping
)

# CPU configuration (high precision)
solver = JAXMFGSolver(
    problem,
    use_gpu=False,
    jit_compile=True,
    finite_diff_order=6,       # Maximum accuracy
    method="newton"            # Advanced solver method
)
```

### **Numba Optimization Setup**
```python
from mfgarchon.utils.performance_optimization import AccelerationBackend

backend = AccelerationBackend()
backend.set_backend("numba")

# Optimize a custom function
@backend.optimize_function  # Automatically applies numba.jit
def custom_finite_difference(u, dx):
    # Your imperative algorithm here
    result = np.zeros_like(u)
    for i in range(1, len(u)-1):
        result[i] = (u[i+1] - 2*u[i] + u[i-1]) / (dx*dx)
    return result
```

## 🏗️ **Hardware-Specific Optimization**

### **Apple Silicon (M1/M2/M3) Setup**
```python
# Optimal configuration for Mac
import mfgarchon

# Check what's available
from mfgarchon.backends import get_available_backends
available = get_available_backends()
print("Available backends:", available)
# Expected: {'torch_mps': True, 'jax': True, 'numba': True}

# Automatic optimal selection for Mac
backend = create_backend("auto")  # Usually selects torch_mps
print(f"Selected: {backend.name}")  # "torch_mps"
```

### **NVIDIA CUDA Setup**
```python
# Optimal for CUDA systems
backend = create_backend("auto")  # Prefers CUDA if available

# Manual CUDA configuration
if available.get("torch_cuda", False):
    torch_backend = create_backend("torch_cuda")

if available.get("jax_gpu", False):
    jax_backend = create_backend("jax_gpu")

# Use both for different problem components
neural_solver = PINNSolver(problem, backend=torch_backend)
kernel_solver = JAXMFGSolver(problem, use_gpu=True)
```

### **CPU-Only Optimization**
```python
# Optimal for CPU-only systems
from mfgarchon.utils.performance_optimization import ParallelizationHelper

# Use all available CPU cores
parallelization = ParallelizationHelper()
optimal_chunks = parallelization.estimate_optimal_chunks(
    total_size=problem.spatial_points,
    memory_limit_mb=1000  # Adjust based on available RAM
)

# Configure Numba for multiprocessing
backend = create_backend("numba")
# Numba automatically uses multiple cores for suitable operations
```

## 📊 **Performance Monitoring**

### **Built-in Performance Analysis**
```python
from mfgarchon.utils.performance_optimization import PerformanceMonitor

monitor = PerformanceMonitor()

with monitor.monitor_operation("backend_solve", backend_type="torch_mps"):
    result = solver.solve()

# Get detailed performance metrics
summary = monitor.get_summary()
print(f"Execution time: {summary['total_time']:.2f}s")
print(f"Peak memory: {summary['peak_memory']:.1f}MB")
```

### **Backend Benchmarking**
```python
# Compare all available backends
from mfgarchon.utils.performance_optimization import optimize_mfg_problem_performance

performance_analysis = optimize_mfg_problem_performance({
    "spatial_points": 10000,
    "time_steps": 100,
    "dimension": 2
})

print("Recommendations:")
for rec in performance_analysis["recommendations"]:
    print(f"  • {rec}")
```

## 🔄 **Correct Import Paths**

### **Current Structure (Phase 3+)**

The acceleration utilities are organized under `mfgarchon.utils.acceleration`:

```python
# Correct imports
from mfgarchon.alg.mfg_solvers import JAXMFGSolver
from mfgarchon.utils.acceleration.jax_utils import compute_hamiltonian, tridiagonal_solve

# Backend factory
from mfgarchon.backends import create_backend

# Auto-select best backend (torch > jax > numpy)
backend = create_backend()

# Or explicit selection
backend = create_backend("torch")  # PyTorch with CUDA/MPS/CPU
backend = create_backend("jax")    # JAX with GPU/CPU
backend = create_backend("numpy")  # NumPy baseline
```

### **Note on Old Imports**

The old `mfgarchon.accelerated` module has been removed. If you have legacy code:

```python
# REMOVED (no longer available)
# from mfgarchon.accelerated import JAXMFGSolver

# Use this instead:
from mfgarchon.alg.mfg_solvers import JAXMFGSolver
from mfgarchon.utils.acceleration.jax_utils import compute_hamiltonian
```

## 🔧 **Backend Selection Guide**

### **Tiered Auto-Selection (torch > jax > numpy)**

The default `create_backend()` follows a tiered priority for optimal performance:

```python
backend = create_backend()  # Auto-select best available

# Selection logic:
# 1. PyTorch (CUDA > MPS > CPU) - Best for most use cases
# 2. JAX (GPU > CPU) - Scientific computing alternative
# 3. NumPy - Universal fallback
```

**When auto-selection works best:**
- ✅ Vectorizable operations (most MFG solvers)
- ✅ Particle methods (PyTorch KDE)
- ✅ GPU acceleration available
- ✅ Standard grid-based PDEs

### **When to Use Numba Backend Explicitly**

Numba is **NOT** in auto-selection but available via explicit opt-in:

```python
backend = create_backend("numba")
```

**Use Numba for:**

**1. Adaptive Mesh Refinement (AMR)**
```python
# Dynamic grid refinement with irregular patterns
from mfgarchon.backends import create_backend

backend = create_backend("numba")

# AMR solver with imperative mesh traversal
solver = AMRSolver(problem, backend=backend)
solver.refine_mesh(criterion="gradient")
```

**2. Imperative Algorithms with Heavy Branching**
```python
# Algorithms with extensive if/else logic
@numba.njit
def gauss_seidel_step(u, f, dx):
    n = len(u)
    for i in range(1, n-1):
        if f[i] > threshold:  # Conditional logic
            u[i] = relaxation_update(u[i-1], u[i+1], f[i], dx)
        else:
            u[i] = simple_update(u[i-1], u[i+1])
    return u

backend = create_backend("numba")
```

**3. Sequential Iterative Methods**
```python
# Gauss-Seidel, SOR, line relaxation
backend = create_backend("numba")
solver = create_solver(problem, method="gauss_seidel", backend=backend)
```

**4. CPU-Bound Tight Loops**
```python
# When vectorization isn't possible
backend = create_backend("numba")
# Numba JIT compilation provides ~10-100x speedup over pure Python
```

**When NOT to use Numba:**

- ❌ Vectorizable operations → Use PyTorch/JAX instead
- ❌ GPU acceleration needed → Use PyTorch (CUDA/MPS) or JAX (GPU)
- ❌ Particle methods → Use PyTorch KDE (5-6x faster for 50k+ particles)
- ❌ Automatic differentiation needed → Use PyTorch or JAX
- ❌ Neural network components → Use PyTorch (RL infrastructure)

### **Backend Comparison Table**

| Use Case | Best Backend | Reason |
|:---------|:-------------|:-------|
| Particle methods (< 10k) | PyTorch (CPU) | Vectorized KDE |
| Particle methods (> 50k) | PyTorch (MPS/CUDA) | GPU acceleration |
| Standard FDM/FEM | PyTorch/JAX | Vectorized operations |
| AMR solvers | **Numba** | Imperative mesh ops |
| Neural networks | PyTorch | RL infrastructure |
| Auto-differentiation | PyTorch/JAX | Built-in AD |
| Tight loops with conditionals | **Numba** | JIT compilation |
| Linear algebra | JAX | Best precision |

## 🎯 **Common Usage Patterns**

### **Automatic Backend Selection (Recommended)**
```python
# Let MFGArchon choose the best backend
from mfgarchon import MFGProblem
from mfgarchon.factory import create_solver

problem = MFGProblem(Nx=100, Nt=50)
solver = create_solver(problem, method="auto", backend="auto")
result = solver.solve()
```

### **Mixed Backend Strategy**
```python
# Use different backends for different components
torch_backend = create_backend("torch_mps")  # For neural components
jax_backend = create_backend("jax")          # For pure math
numba_backend = create_backend("numba")      # For numerical loops

# Neural solver for learning components
neural_solver = PINNSolver(problem, backend=torch_backend)

# Mathematical solver for PDE kernels
math_solver = JAXMFGSolver(problem, use_gpu=True)

# Numba for loop-heavy algorithms
# (AMR was removed - use external libraries like pyAMReX)
```

### **Development vs Production**
```python
# Development: Maximum compatibility and debugging
if os.getenv("MFG_DEVELOPMENT", "false").lower() == "true":
    backend = create_backend("numpy")  # Predictable, debuggable
    precision = "float64"              # High precision
else:
    # Production: Maximum performance
    backend = create_backend("auto")   # Optimal hardware utilization
    precision = "float32"              # Speed over precision
```

## 🔍 **Debugging Backend Issues**

### **Backend Detection Problems**
```python
from mfgarchon.backends import get_available_backends

available = get_available_backends()
print("Backend availability:")
for name, status in available.items():
    print(f"  {name}: {'✓' if status else '✗'}")

# Force backend for testing
try:
    backend = create_backend("torch_mps")
    print("✓ torch_mps working")
except Exception as e:
    print(f"✗ torch_mps failed: {e}")
```

### **Performance Debugging**
```python
# Enable verbose performance logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check GPU memory usage
if backend.name.endswith("gpu"):
    print(f"GPU memory: {backend.get_memory_usage()}")

# Profile specific operations
from mfgarchon.utils.performance_optimization import profile_jax_function

if backend.name.startswith("jax"):
    metrics = profile_jax_function(compute_hamiltonian, u_grad, density)
    print(f"Compile time: {metrics['compile_time']:.3f}s")
    print(f"Execute time: {metrics['mean_exec_time']:.3f}s")
```

## 📚 **Additional Resources**

### **Backend-Specific Documentation**
- **PyTorch**: See `docs/development/PYTORCH_PINN_INTEGRATION_ANALYSIS.md`
- **JAX**: See `mfgarchon/utils/acceleration/jax_utils.py` docstrings
- **Numba**: See `mfgarchon/utils/performance_optimization.py` examples

### **Hardware Setup Guides**
- **Apple Silicon**: Ensure Metal Performance Shaders enabled
- **NVIDIA CUDA**: Install appropriate CUDA toolkit version
- **CPU Optimization**: Configure OpenMP threads and BLAS libraries

### **Performance Benchmarks**
```python
# Run comprehensive benchmarks
python -m mfgarchon.benchmarks.backend_comparison --all-backends --problem-sizes large
```

---

**Quick Reference:**
- 🧠 **Neural methods** → `torch` backend
- 📊 **Pure math** → `jax` backend
- 🌳 **AMR/loops** → `numba` backend
- 💾 **Large problems** → `sparse` + `numba`
- 🔄 **Unsure** → `"auto"` selection

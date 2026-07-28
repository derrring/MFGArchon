# Plugin Development Guide for MFGArchon

**Date:** July 27, 2025  
**Version:** 1.0  
**Purpose:** Comprehensive guide for developing plugins for the MFGArchon framework  
**Target Audience:** Plugin developers, researchers, and third-party contributors  

## Overview

The MFGArchon plugin system enables researchers and developers to extend the framework with custom solvers, analysis tools, and algorithms without modifying the core codebase. This guide provides everything needed to create, test, and distribute MFGArchon plugins.

## Plugin Architecture

### Plugin Types

MFGArchon supports two main types of plugins:

1. **Solver Plugins** (`SolverPlugin`): Provide new solver algorithms
2. **Analysis Plugins** (`AnalysisPlugin`): Provide post-processing and analysis tools

### Plugin Lifecycle

1. **Discovery**: Plugin manager discovers available plugins
2. **Registration**: Plugin classes are registered with metadata validation
3. **Loading**: Plugin instances are created and initialized
4. **Execution**: Plugin functionality is accessed through the framework
5. **Unloading**: Plugin cleanup and resource release

## Creating a Solver Plugin

### Basic Structure

```python
from mfgarchon.core.plugin_system import SolverPlugin, PluginMetadata
from mfgarchon.config import MFGSolverConfig
from typing import List, Optional, Dict, Any

class YourSolverPlugin(SolverPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="your_plugin_name",
            version="1.0.0",
            description="Description of your plugin",
            author="Your Name",
            email="your.email@example.com",
            license="MIT",
            homepage="https://github.com/yourusername/your-plugin",
            min_mfg_version="1.0.0",
            dependencies=["numpy", "scipy"],  # Required packages
            tags=["custom", "research"]
        )
    
    def get_solver_types(self) -> List[str]:
        return ["your_solver_type"]
    
    def create_solver(self, problem, solver_type: str, 
                     config: Optional[MFGSolverConfig] = None, **kwargs):
        if solver_type == "your_solver_type":
            return YourCustomSolver(problem, **kwargs)
        raise ValueError(f"Unknown solver type: {solver_type}")
    
    def validate_solver_type(self, solver_type: str) -> bool:
        return solver_type in self.get_solver_types()
```

### Implementing Your Solver

Your solver class should follow the standard MFG solver interface:

```python
class YourCustomSolver:
    def __init__(self, problem, **kwargs):
        self.problem = problem
        # Initialize your solver with parameters
    
    def solve(self) -> SolverResult:
        """
        Solve the MFG system and return a SolverResult object.
        
        Returns:
            SolverResult containing U, M, the per-iteration error histories, and metadata
        """
        # Your solving algorithm here

        # Return standardized result. U, M, iterations, error_history_U and
        # error_history_M are required; the rest have defaults.
        return SolverResult(
            U=U_array,
            M=M_array,
            iterations=num_iterations,
            error_history_U=error_history_U,
            error_history_M=error_history_M,
            solver_name="your_solver_type",
            converged=True,
            execution_time=time_taken,
            metadata={"learning_rate": self.learning_rate},
        )
```

## Example: Complete Solver Plugin

Here's a complete example implementing a custom gradient descent solver:

```python
import numpy as np
import time
from typing import List, Optional, Dict, Any

from mfgarchon.core.plugin_system import SolverPlugin, PluginMetadata
from mfgarchon.utils.solver_result import SolverResult
from mfgarchon.config import MFGSolverConfig

class GradientDescentSolver:
    """Custom gradient descent MFG solver."""
    
    def __init__(self, problem, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.problem = problem
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Initialize solution arrays
        self.U_solution = np.zeros((problem.Nt + 1, problem.Nx + 1))
        self.M_solution = np.zeros((problem.Nt + 1, problem.Nx + 1))
        
        # Set boundary conditions
        self.M_solution[0, :] = problem.m_init
        self.U_solution[-1, :] = problem.g_final
    
    def solve(self) -> SolverResult:
        start_time = time.time()
        error_history_U: list[float] = []
        error_history_M: list[float] = []

        for iteration in range(self.max_iterations):
            # Store previous solution
            U_prev = self.U_solution.copy()
            M_prev = self.M_solution.copy()

            # Update value function and density
            self._gradient_step()

            # Track each field separately: the coupled system can converge in one and not
            # the other, and a single combined number cannot show that.
            error_U = float(np.max(np.abs(self.U_solution - U_prev)))
            error_M = float(np.max(np.abs(self.M_solution - M_prev)))
            error_history_U.append(error_U)
            error_history_M.append(error_M)
            error = max(error_U, error_M)

            if error < self.tolerance:
                break

        
        return SolverResult(
            U=self.U_solution,
            M=self.M_solution,
            iterations=iteration + 1,
            error_history_U=np.array(error_history_U),
            error_history_M=np.array(error_history_M),
            solver_name="gradient_descent",
            converged=error < self.tolerance,
            execution_time=time.time() - start_time,
            metadata={"learning_rate": self.learning_rate},
        )
    
    def _gradient_step(self):
        # Implement your gradient descent logic here
        pass
    
    def _compute_error(self, U_prev, M_prev):
        # Compute convergence error
        return np.linalg.norm(self.U_solution - U_prev) + np.linalg.norm(self.M_solution - M_prev)

class CustomSolverPlugin(SolverPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="gradient_descent_plugin",
            version="1.0.0",
            description="Gradient descent solver for MFG systems",
            author="Research Team",
            email="team@research.edu",
            license="MIT",
            homepage="https://github.com/research/mfg-gradient-descent",
            min_mfg_version="1.0.0",
            dependencies=["numpy"]
        )
    
    def get_solver_types(self) -> List[str]:
        return ["gradient_descent"]
    
    def create_solver(self, problem, solver_type: str, 
                     config: Optional[MFGSolverConfig] = None, **kwargs):
        if solver_type == "gradient_descent":
            return GradientDescentSolver(problem, **kwargs)
        raise ValueError(f"Unknown solver type: {solver_type}")
    
    def validate_solver_type(self, solver_type: str) -> bool:
        return solver_type == "gradient_descent"
    
    def get_solver_description(self, solver_type: str) -> str:
        return "Custom gradient descent solver for educational purposes"
    
    def get_solver_parameters(self, solver_type: str) -> Dict[str, Any]:
        return {
            "learning_rate": {
                "type": "float",
                "default": 0.01,
                "range": [1e-6, 1.0],
                "description": "Learning rate for gradient steps"
            },
            "max_iterations": {
                "type": "int",
                "default": 1000,
                "range": [1, 10000],
                "description": "Maximum number of iterations"
            },
            "tolerance": {
                "type": "float",
                "default": 1e-6,
                "range": [1e-12, 1e-2],
                "description": "Convergence tolerance"
            }
        }
```

## Creating an Analysis Plugin

Analysis plugins provide post-processing capabilities for solver results:

```python
from mfgarchon.core.plugin_system import AnalysisPlugin, PluginMetadata
import numpy as np

class CustomAnalysisPlugin(AnalysisPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="custom_analysis_plugin",
            version="1.0.0",
            description="Custom analysis tools for MFG results",
            author="Analysis Team",
            email="analysis@research.edu",
            license="MIT",
            homepage="https://github.com/research/mfg-analysis",
            min_mfg_version="1.0.0"
        )
    
    def get_analysis_types(self) -> List[str]:
        return ["stability_analysis", "performance_metrics"]
    
    def run_analysis(self, result, analysis_type: str, **kwargs):
        if analysis_type == "stability_analysis":
            return self._analyze_stability(result, **kwargs)
        elif analysis_type == "performance_metrics":
            return self._compute_performance_metrics(result, **kwargs)
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    def _analyze_stability(self, result, **kwargs):
        # Implement stability analysis
        eigenvalues = np.linalg.eigvals(result.U_solution)
        return {
            'max_eigenvalue': np.max(np.real(eigenvalues)),
            'stability_margin': np.min(np.real(eigenvalues)),
            'is_stable': np.all(np.real(eigenvalues) <= 0)
        }
    
    def _compute_performance_metrics(self, result, **kwargs):
        # Implement performance metrics computation
        return {
            'convergence_rate': self._estimate_convergence_rate(result),
            'memory_efficiency': self._compute_memory_usage(result),
            'computational_complexity': self._estimate_complexity(result)
        }
```

## Plugin Installation and Distribution

### 1. Setup.py Configuration

Create a `setup.py` file for your plugin:

```python
from setuptools import setup, find_packages

setup(
    name="mfg-custom-solver-plugin",
    version="1.0.0",
    description="Custom solver plugin for MFGArchon",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "mfgarchon>=1.0.0",
        "numpy",
        "scipy"
    ],
    entry_points={
        'mfgarchon.plugins': [
            'custom_solver = your_plugin_package:CustomSolverPlugin',
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
```

### 2. Package Structure

Organize your plugin package:

```
your-plugin/
├── setup.py
├── README.md
├── LICENSE
├── your_plugin_package/
│   ├── __init__.py
│   ├── solver.py          # Your solver implementation
│   ├── plugin.py          # Plugin class
│   └── tests/
│       ├── __init__.py
│       ├── test_solver.py
│       └── test_plugin.py
├── examples/
│   └── basic_usage.py
└── docs/
    └── usage.md
```

### 3. Testing Your Plugin

Create comprehensive tests:

```python
import pytest
from mfgarchon import MFGProblem
from mfgarchon.core.plugin_system import get_plugin_manager
from your_plugin_package import CustomSolverPlugin

class TestCustomSolverPlugin:
    def setup_method(self):
        self.plugin = CustomSolverPlugin()
        self.problem = MFGProblem(Nx=20, Nt=10, T=1.0)
    
    def test_plugin_metadata(self):
        metadata = self.plugin.metadata
        assert metadata.name == "gradient_descent_plugin"
        assert metadata.version == "1.0.0"
    
    def test_solver_creation(self):
        solver = self.plugin.create_solver(self.problem, "gradient_descent")
        assert solver is not None
    
    def test_solver_execution(self):
        solver = self.plugin.create_solver(
            self.problem, 
            "gradient_descent",
            learning_rate=0.02,
            max_iterations=100
        )
        result = solver.solve()
        
        assert result is not None
        assert result.U_solution.shape == (self.problem.Nt + 1, self.problem.Nx + 1)
        assert result.M_solution.shape == (self.problem.Nt + 1, self.problem.Nx + 1)
    
    def test_plugin_integration(self):
        manager = get_plugin_manager()
        assert manager.register_plugin(CustomSolverPlugin)
        assert manager.load_plugin("gradient_descent_plugin")
        
        # Test through plugin manager
        solver = manager.create_solver(self.problem, "gradient_descent")
        result = solver.solve()
        assert result.solver_info['solver_type'] == 'gradient_descent'
```

## Using Plugins

### Loading Plugins Automatically

```python
from mfgarchon.core.plugin_system import discover_and_load_plugins
from mfgarchon import MFGProblem
from mfgarchon.core.plugin_system import create_solver_with_plugins

# Discover and load all available plugins
plugin_results = discover_and_load_plugins()
print("Loaded plugins:", plugin_results)

# Create problem
problem = MFGProblem(Nx=30, Nt=15, T=1.0)

# Use plugin-provided solver
solver = create_solver_with_plugins(
    problem, 
    "gradient_descent",  # Plugin solver type
    learning_rate=0.05,
    max_iterations=500
)

result = solver.solve()
print(f"Plugin solver converged: {result.converged}")
```

### Manual Plugin Loading

```python
from mfgarchon.core.plugin_system import get_plugin_manager
from your_plugin_package import CustomSolverPlugin

# Get plugin manager
manager = get_plugin_manager()

# Register and load plugin manually
manager.register_plugin(CustomSolverPlugin)
manager.load_plugin("gradient_descent_plugin")

# List available solvers (includes plugin solvers)
available_solvers = manager.list_available_solvers()
for solver_type, info in available_solvers.items():
    print(f"{solver_type}: {info['description']}")

# Create solver through manager
solver = manager.create_solver(problem, "gradient_descent", learning_rate=0.1)
```

## Best Practices

### 1. Error Handling

```python
def create_solver(self, problem, solver_type: str, config=None, **kwargs):
    try:
        if not self.validate_solver_type(solver_type):
            raise ValueError(f"Unsupported solver type: {solver_type}")
        
        # Validate parameters
        self._validate_parameters(kwargs)
        
        # Create solver with error handling
        return YourSolver(problem, **kwargs)
        
    except Exception as e:
        logger.error(f"Failed to create solver {solver_type}: {e}")
        raise
```

### 2. Parameter Validation

```python
def _validate_parameters(self, params):
    """Validate solver parameters."""
    learning_rate = params.get('learning_rate', 0.01)
    if not (1e-6 <= learning_rate <= 1.0):
        raise ValueError(f"learning_rate must be in [1e-6, 1.0], got {learning_rate}")
    
    max_iterations = params.get('max_iterations', 1000)
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError(f"max_iterations must be positive integer, got {max_iterations}")
```

### 3. Configuration Integration

```python
def create_solver(self, problem, solver_type: str, config=None, **kwargs):
    # Use config values as defaults, allow kwargs to override
    if config:
        learning_rate = kwargs.get('learning_rate', config.newton.damping_factor)
        tolerance = kwargs.get('tolerance', config.convergence_tolerance)
        max_iterations = kwargs.get('max_iterations', config.newton.max_iterations)
    else:
        learning_rate = kwargs.get('learning_rate', 0.01)
        tolerance = kwargs.get('tolerance', 1e-6)
        max_iterations = kwargs.get('max_iterations', 1000)
    
    return YourSolver(problem, learning_rate, tolerance, max_iterations)
```

### 4. Performance Considerations

```python
def solve(self):
    # Pre-allocate arrays for efficiency
    U_solution = np.zeros((self.problem.Nt + 1, self.problem.Nx + 1))
    M_solution = np.zeros((self.problem.Nt + 1, self.problem.Nx + 1))
    
    # Use vectorized operations when possible
    spatial_derivative = np.gradient(U_solution, self.problem.Dx, axis=1)
    
    # Monitor memory usage for large problems
    if self.problem.Nx * self.problem.Nt > 10000:
        logger.warning("Large problem detected, monitoring memory usage")
```

### 5. Documentation

```python
class YourSolver:
    """
    Custom gradient descent solver for MFG systems.
    
    This solver implements a gradient descent approach to solve the coupled
    Hamilton-Jacobi-Bellman and Fokker-Planck equations that arise in 
    mean field games.
    
    Parameters:
        problem (MFGProblem): The MFG problem to solve
        learning_rate (float): Step size for gradient descent (default: 0.01)
        max_iterations (int): Maximum number of iterations (default: 1000)
        tolerance (float): Convergence tolerance (default: 1e-6)
    
    Attributes:
        convergence_history (List[float]): History of convergence errors
        
    Examples:
        >>> problem = MFGProblem(Nx=20, Nt=10)
        >>> solver = YourSolver(problem, learning_rate=0.05)
        >>> result = solver.solve()
        >>> print(f"Converged: {result.converged}")
    """
    
    def solve(self) -> SolverResult:
        """
        Solve the MFG system using gradient descent.
        
        Returns:
            SolverResult: Complete solution with metadata
            
        Raises:
            ConvergenceError: If solution fails to converge
            ValueError: If problem is ill-posed
        """
        pass
```

## Advanced Features

### 1. GPU Acceleration

```python
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    import numpy as cp
    HAS_GPU = False

class GPUAcceleratedSolver:
    def __init__(self, problem, use_gpu=True):
        self.use_gpu = use_gpu and HAS_GPU
        self.xp = cp if self.use_gpu else np
        
    def solve(self):
        # Arrays automatically on GPU if available
        U = self.xp.zeros((self.problem.Nt + 1, self.problem.Nx + 1))
        M = self.xp.zeros((self.problem.Nt + 1, self.problem.Nx + 1))
        
        # GPU-accelerated computation
        for iteration in range(self.max_iterations):
            U, M = self._gradient_step_gpu(U, M)
        
        # Move back to CPU for result
        if self.use_gpu:
            U = cp.asnumpy(U)
            M = cp.asnumpy(M)
        
        return SolverResult(U_solution=U, M_solution=M, ...)
```

### 2. Adaptive Parameters

```python
class AdaptiveSolver:
    def __init__(self, problem, adaptive_learning=True):
        self.adaptive_learning = adaptive_learning
        self.learning_rate = 0.01
        
    def solve(self):
        convergence_history = []
        
        for iteration in range(self.max_iterations):
            error = self._gradient_step()
            convergence_history.append(error)
            
            # Adapt learning rate based on convergence
            if self.adaptive_learning and len(convergence_history) > 2:
                self._adapt_learning_rate(convergence_history)
    
    def _adapt_learning_rate(self, history):
        if len(history) < 3:
            return
            
        # Increase learning rate if converging well
        if history[-1] < history[-2] < history[-3]:
            self.learning_rate *= 1.1
        # Decrease if diverging
        elif history[-1] > history[-2]:
            self.learning_rate *= 0.9
        
        # Keep within bounds
        self.learning_rate = np.clip(self.learning_rate, 1e-6, 1.0)
```

### 3. Parallel Processing

```python
from multiprocessing import Pool
import numpy as np

class ParallelSolver:
    def __init__(self, problem, num_processes=None):
        self.num_processes = num_processes or os.cpu_count()
        
    def solve(self):
        # Split problem for parallel processing
        subproblems = self._decompose_problem()
        
        with Pool(self.num_processes) as pool:
            partial_solutions = pool.map(self._solve_subproblem, subproblems)
        
        # Combine partial solutions
        return self._combine_solutions(partial_solutions)
```

## Troubleshooting

### Common Issues

1. **Plugin Not Found**
   - Check plugin is in search path
   - Verify entry point configuration
   - Ensure plugin class inherits from correct base class

2. **Version Compatibility**
   - Update min_mfg_version in metadata
   - Check dependency versions
   - Test with target MFGArchon version

3. **Import Errors**
   - Verify all dependencies are installed
   - Check Python path configuration
   - Ensure plugin package is properly structured

4. **Performance Issues**
   - Profile solver with small problems first
   - Use vectorized operations
   - Consider memory allocation patterns

### Debugging Tips

```python
# Enable plugin system logging
import logging
logging.getLogger('mfgarchon.core.plugin_system').setLevel(logging.DEBUG)

# Test plugin manually
plugin = YourPlugin()
solver = plugin.create_solver(problem, "your_solver")
result = solver.solve()

# Check the result. There is no validator function to call: `SolverResult` is a dataclass
# whose required fields are enforced at construction, so a plugin that builds one has
# already been checked. What is worth asserting is the physics your solver claims.
assert result.U.shape == result.M.shape
assert result.converged, f"did not converge in {result.iterations} iterations"
```

## Contributing Plugins to Community

### 1. Code Quality
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add type hints for all functions
- Maintain >90% test coverage

### 2. Documentation
- Provide clear README with examples
- Document all parameters and their ranges
- Include performance characteristics
- Add mathematical background where relevant

### 3. Testing
- Test on multiple problem sizes
- Validate convergence properties
- Include performance benchmarks
- Test plugin registration and loading

### 4. Distribution
- Publish to PyPI with proper versioning
- Create GitHub repository with issues tracking
- Provide example notebooks
- Include citation information for academic use

This guide provides the foundation for creating powerful, extensible plugins for MFGArchon. The plugin system enables the community to collaborate and extend the framework while maintaining code quality and user experience standards.

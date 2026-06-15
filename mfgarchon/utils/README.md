# MFGArchon Utils Module

**Purpose**: Comprehensive utility components for enhanced MFGArchon functionality  
**Location**: `mfgarchon/utils/`  
**Status**: Production Ready  

## 📋 Overview

The `utils` module provides essential utilities that enhance the MFGArchon platform with modern features like progress monitoring, validation, error handling, CLI support, and solver enhancements. These components transform MFGArchon from a research prototype into a professional scientific computing platform.

## 🗂️ Module Components

### Core Utilities

| Component | File | Purpose | Status |
|-----------|------|---------|--------|
| **Validation** | `validation.py` | Centralized validation utilities | ✅ Stable |
| **Exceptions** | `exceptions.py` | Enhanced error handling system | ✅ Stable |
| **Solver Results** | `solver_result.py` | Structured result objects | ✅ Stable |

### Modern Enhancements

| Component | File | Purpose | Status |
|-----------|------|---------|--------|
| **Progress Monitoring** | `progress.py` | Progress bars and timing | [COMPLETE] |
| **Solver Decorators** | `solver_decorators.py` | Solver enhancement decorators | [COMPLETE] |
| **CLI Framework** | `cli.py` | Command-line interface | [COMPLETE] |
| **Professional Logging** | `logging.py` | Structured logging system | [COMPLETE] |
| **Logging Decorators** | `logging_decorators.py` | Logging integration decorators | [COMPLETE] |

## Component Documentation

### 1. Validation (`validation.py`)

**Purpose**: Centralized validation utilities for arrays, solutions, and parameters.

**Key Features**:
- Array validation with dimension checking
- MFG solution validation
- Safe solution return wrappers
- Parameter validation utilities

**Example Usage**:
```python
from mfgarchon.utils.validation import validate_solution_array, validate_mfg_solution

# Validate solution arrays
U, M = validate_mfg_solution(U_raw, M_raw, x_grid)

# Safe array validation
is_valid, error_msg = validate_solution_array(solution, expected_shape=(100, 50))
```

**Benefits**:
- Consistent error messages across the codebase
- Early detection of dimensional mismatches
- Eliminates code duplication
- Provides clear debugging information

---

### 2. Enhanced Exceptions (`exceptions.py`)

**Purpose**: Structured exception handling with actionable guidance.

**Key Features**:
- Specialized exception classes for different error types
- Diagnostic data collection
- Suggested actions for users
- Error codes for programmatic handling

**Exception Classes**:
- `MFGSolverError`: General solver errors
- `MFGConfigurationError`: Configuration issues
- `MFGConvergenceError`: Convergence failures
- `MFGValidationError`: Input validation errors

**Example Usage**:
```python
from mfgarchon.utils.exceptions import MFGSolverError

try:
    result = solver.solve()
except MFGSolverError as e:
    print(f"Solver failed: {e}")
    print(f"Suggestion: {e.suggested_action}")
```

**Benefits**:
- Better user experience with actionable error messages
- Easier debugging with diagnostic information
- Consistent error handling patterns
- Professional error reporting

---

### 3. Solver Results (`solver_result.py`)

**Purpose**: Structured result objects with backward compatibility.

**Key Features**:
- Rich metadata storage
- Tuple-like unpacking for compatibility
- Performance metrics integration
- Serialization support

**Main Classes**:
- `SolverResult`: Primary result dataclass
- `ConvergenceInfo`: Convergence diagnostics
- `PerformanceMetrics`: Timing and iteration data

**Example Usage**:
```python
from mfgarchon.utils.solver_result import SolverResult

# Structured result with metadata
result = SolverResult(
    U=solution_U, M=solution_M, iterations=25,
    converged=True, execution_time=2.5
)

# Backward compatible unpacking
U, M, iterations, err_u, err_m = result
```

**Benefits**:
- Rich result information without breaking existing code
- Better research data collection
- Consistent result format across solvers
- Easy integration with analysis tools

---

### 4. Progress Monitoring (`progress.py`)

**Purpose**: Modern progress bars, timing, and performance monitoring using Rich.

**Key Features**:
- Beautiful progress bars using Rich (v0.16.15+)
- Precise timing utilities
- Iteration progress tracking with convergence monitoring
- `tqdm` alias provided for backward compatibility (now points to `RichProgressBar`)

**Main Components**:
- `SolverTimer`: Context manager for timing operations
- `IterationProgress`: Advanced progress tracking for iterative solvers
- `@timed_operation`: Decorator for automatic timing
- `progress_context`: Context manager for general progress tracking

**Example Usage**:
```python
from mfgarchon.utils.progress import SolverTimer, solver_progress

# Time an operation
with SolverTimer("Matrix Computation") as timer:
    result = expensive_computation()
print(f"Completed in {timer.format_duration()}")

# Track solver iterations
with solver_progress(max_iterations=50, "HJB Solver") as progress:
    for i in range(50):
        # Solver iteration work
        error = solve_iteration()
        progress.update(1, error=error)
```

**Benefits**:
- Professional user experience with real-time feedback
- Performance monitoring and optimization insights
- Minimal overhead on solver performance
- Elegant integration with existing code

---

### 5. Solver Decorators (`solver_decorators.py`)

**Purpose**: Decorator-based enhancement of solver methods.

**Key Features**:
- Automatic progress monitoring integration
- Timing information collection
- Solver method enhancement
- Backward compatibility preservation

**Main Decorators**:
- `@with_progress_monitoring`: Add progress bars to solver methods
- `@enhanced_solver_method`: Comprehensive solver enhancement
- `@time_solver_operation`: Timing-only decoration

**Example Usage**:
```python
from mfgarchon.utils.solver_decorators import enhanced_solver_method

class MySolver:
    @enhanced_solver_method(options=SolverMonitoringOptions.PROGRESS | SolverMonitoringOptions.TIMING)
    def solve(self, max_iterations=100, verbose=True):
        # Solver implementation
        # Progress tracking and timing added automatically
        return result
```

**Benefits**:
- Zero-effort enhancement of existing solvers
- Consistent progress monitoring across all solvers
- Configurable enhancement levels
- Professional solver behavior

---

### 6. CLI Framework (`cli.py`)

**Purpose**: Professional command-line interface for MFGArchon.

**Key Features**:
- Comprehensive argument parsing
- Configuration file support (JSON/YAML)
- Subcommand structure
- Integration with factory patterns

**Main Functions**:
- `create_base_parser()`: Base argument parser setup
- `load_config_file()`: Configuration file loading
- `run_solver_from_cli()`: CLI-based solver execution
- `main()`: Primary CLI entry point

**Command Structure**:
```bash
# Solve MFG problem
python -m mfgarchon.utils.cli solve --solver-type monitored_particle --preset accurate

# Generate configuration
python -m mfgarchon.utils.cli config generate sample_config.json

# Validate configuration
python -m mfgarchon.utils.cli config validate my_config.yaml
```

**Configuration Example**:
```json
{
  "problem": {
    "T": 1.0,
    "Nt": 50,
    "xmin": 0.0,
    "xmax": 1.0,
    "Nx": 100
  },
  "solver": {
    "type": "monitored_particle",
    "preset": "accurate",
    "num_particles": 5000
  },
  "execution": {
    "verbose": true,
    "progress": true,
    "timing": true
  }
}
```

**Benefits**:
- Easy automation and scripting
- Reproducible research through configuration files
- Professional command-line experience
- Integration with existing workflow tools

---

### 7. Professional Logging (`logging.py`)

**Purpose**: Structured logging system for debugging, monitoring, and development.

**Key Features**:
- Configurable logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Optional colored terminal output with colorlog integration
- File logging with automatic log rotation
- Structured logging functions for solvers and validation
- Context managers for timed operations
- Performance metrics logging

**Main Components**:
- `MFGLogger`: Central logging manager with configuration
- `get_logger()`: Get configured logger instance
- `configure_logging()`: Global logging configuration
- `log_solver_start()`, `log_solver_progress()`, `log_solver_completion()`: Solver lifecycle logging
- `LoggedOperation`: Context manager for timed operations

**Example Usage**:
```python
from mfgarchon.utils import get_logger, configure_logging, LoggedOperation

# Configure logging
configure_logging(level="INFO", use_colors=True, log_to_file=True)

# Get logger
logger = get_logger(__name__)

# Basic logging
logger.info("Solver initialized")
logger.debug("Debug information")
logger.warning("Using default parameters")

# Timed operations
with LoggedOperation(logger, "Matrix computation"):
    result = expensive_computation()

# Solver integration
from mfgarchon.utils import log_solver_start, log_solver_completion
log_solver_start(logger, "MySolver", {"max_iter": 50, "tol": 1e-6})
log_solver_completion(logger, "MySolver", 25, 1e-7, 2.3, True)
```

**Benefits**:
- Professional debugging and monitoring capabilities
- Configurable output levels and formatting
- Minimal performance overhead
- Easy integration with existing code
- Consistent logging across all components

---

### 8. Logging Decorators (`logging_decorators.py`)

**Purpose**: Decorators for easy integration of logging into existing code.

**Key Features**:
- `@logged_solver_method`: Automatic solver logging
- `@logged_operation`: General operation logging
- `@logged_validation`: Validation logging with error handling
- `@performance_logged`: Performance metrics collection
- `LoggingMixin`: Mixin class for adding logging to any class

**Example Usage**:
```python
from mfgarchon.utils import logged_solver_method, LoggingMixin

# Decorator approach
class MySolver:
    @logged_solver_method(log_config=True, log_performance=True)
    def solve(self, max_iterations=50):
        # Logging handled automatically
        return result

# Mixin approach
class MyClass(LoggingMixin):
    def __init__(self):
        super().__init__()
        self.log_info("Class initialized")
    
    def compute(self):
        with self.log_operation("Complex computation"):
            return self._do_computation()
```

**Benefits**:
- Zero-modification integration with existing code
- Consistent logging patterns across components
- Configurable logging levels and features
- Easy retrofit of legacy code

---

## Usage Patterns

### Basic Usage

```python
# Import utilities
from mfgarchon.utils import validation, progress, exceptions, configure_logging, get_logger

# Configure logging
configure_logging(level="INFO", use_colors=True)
logger = get_logger(__name__)

# Create a solver with progress monitoring and logging
from mfgarchon.utils.solver_decorators import enhanced_solver_method
from mfgarchon.utils.progress import SolverTimer
from mfgarchon.utils import LoggedOperation

# Use validation with logging
try:
    with LoggedOperation(logger, "Solution validation"):
        validated_solution = validation.validate_mfg_solution(U, M, x_grid)
    logger.info("Validation completed successfully")
except exceptions.MFGValidationError as e:
    logger.error(f"Validation failed: {e.suggested_action}")
```

### Advanced Integration

```python
# Enhanced solver class
from mfgarchon.utils.solver_decorators import SolverProgressMixin, enhanced_solver_method

class MyAdvancedSolver(SolverProgressMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_progress(True)
        self.enable_timing(True)
    
    @enhanced_solver_method(options=SolverMonitoringOptions.CONVERGENCE | SolverMonitoringOptions.PROGRESS)
    def solve(self, **kwargs):
        # Implementation automatically gets progress bars and timing
        return self._solve_implementation(**kwargs)
```

### CLI Integration

```bash
# Run solver from command line
python -m mfgarchon.utils.cli solve \
    --solver-type monitored_particle \
    --preset accurate \
    --num-particles 10000 \
    --max-iterations 100 \
    --output results.json \
    --config my_experiment.yaml \
    --verbose
```

## 🔧 Installation Requirements

### Core Dependencies
```bash
# Essential (already included in MFGArchon)
numpy>=1.19.0
scipy>=1.7.0
```

### Optional Enhancements
```bash
# For YAML configuration support
pip install PyYAML

# For colored terminal output
pip install colorlog
```

### Required Dependencies
```bash
# Rich is required for progress bars (v0.16.15+)
pip install rich

# For full feature set
pip install PyYAML rich colorlog
```

## 📈 Performance Impact

### Progress Monitoring
- **Overhead**: <1% for typical solver operations
- **Benefits**: Real-time feedback, better user experience
- **Recommendation**: Enable by default

### Validation
- **Overhead**: <0.1% for array validation
- **Benefits**: Early error detection, better debugging
- **Recommendation**: Use in development and production

### CLI Framework
- **Overhead**: None during solving (only at startup)
- **Benefits**: Automation, reproducibility, configuration management
- **Recommendation**: Essential for research workflows

## 🎯 Best Practices

### 1. Error Handling
```python
# Good: Use specific exceptions with context
try:
    result = solver.solve()
except MFGConvergenceError as e:
    logger.warning(f"Convergence issue: {e.suggested_action}")
    # Try with different parameters
except MFGConfigurationError as e:
    logger.error(f"Configuration error: {e}")
    # Fix configuration and retry
```

### 2. Progress Monitoring
```python
# Good: Use context managers for automatic cleanup
with solver_progress(max_iterations, "My Solver") as progress:
    for i in range(max_iterations):
        error = solve_iteration()
        progress.update(1, error=error)
        if converged:
            break
```

### 3. Configuration Management
```python
# Good: Use configuration files for reproducibility
config = load_config_file("experiment_config.yaml")
solver = create_solver(**config['solver'])
result = solver.solve(**config['execution'])
```

## 📊 Integration Examples

### Research Workflow
```python
# Complete research workflow with utilities
from mfgarchon import create_research_solver
from mfgarchon.utils import progress, validation, exceptions
from mfgarchon.utils.solver_result import SolverResult

def run_experiment(config_file):
    """Complete experiment with error handling and monitoring."""
    try:
        # Load configuration
        config = load_config_file(config_file)
        
        # Create solver with progress monitoring
        solver = create_research_solver(
            problem, collocation_points,
            **config['solver_params']
        )
        
        # Solve with timing
        with SolverTimer("Experiment") as timer:
            result = solver.solve(verbose=True)
        
        # Validate and return structured result
        validated_result = validation.validate_mfg_solution(
            result.U, result.M, x_grid
        )
        
        return SolverResult(
            **validated_result,
            execution_time=timer.duration,
            config=config
        )
        
    except exceptions.MFGSolverError as e:
        logger.error(f"Experiment failed: {e.suggested_action}")
        raise
```

### Production Pipeline
```python
# Production pipeline with comprehensive monitoring
def production_solve(problem_spec, solver_config):
    """Production solver with full monitoring and validation."""
    
    # Validate inputs
    problem = validation.validate_problem_specification(problem_spec)
    config = validation.validate_solver_configuration(solver_config)
    
    # Create solver with enhancements
    solver = create_solver(**config)
    solver.enable_progress(True)
    solver.enable_timing(True)
    
    # Solve with comprehensive error handling
    try:
        with SolverTimer("Production Solve") as timer:
            result = solver.solve()
            
        # Validate results
        validated_result = validation.validate_solver_result(result)
        
        # Log performance metrics
        logger.info(f"Solve completed in {timer.format_duration()}")
        logger.info(f"Converged: {result.converged}, Iterations: {result.iterations}")
        
        return validated_result
        
    except Exception as e:
        logger.error(f"Production solve failed: {e}")
        raise exceptions.MFGSolverError(
            f"Production solve failed: {e}",
            suggested_action="Check input parameters and solver configuration"
        )
```

## 🔮 Future Extensions

The utils module is designed for extensibility. Planned enhancements include:

1. **Advanced Logging**: Structured logging with multiple handlers
2. **Metrics Collection**: Automatic performance and convergence metrics
3. **Visualization Utilities**: Integrated plotting and analysis tools
4. **Distributed Computing**: Support for parallel and distributed execution
5. **Interactive Tools**: Jupyter notebook integration and widgets

## 📞 Support and Contributing

### Getting Help
- Check the examples in each module for usage patterns
- Review the comprehensive test suites for integration examples
- Consult the main MFGArchon documentation for context

### Contributing
- Follow existing code patterns and documentation standards
- Add comprehensive tests for new utilities
- Update this README when adding new components
- Ensure backward compatibility for existing functionality

---

**The MFGArchon utils module transforms the platform into a modern, professional scientific computing environment while maintaining simplicity and backward compatibility.** 🚀

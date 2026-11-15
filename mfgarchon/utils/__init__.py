"""
MFGarchon Utilities Module.

This module contains core utilities for the MFGarchon package.
All plotting functionality has been permanently moved to mfgarchon.visualization.

For visualization, use for example:
    from mfgarchon.visualization import plot_convergence_summary, plot_mass_history

For logging (preferred import path):
    from mfgarchon.utils.mfg_logging import get_logger, configure_research_logging

Organization:
- mfg_logging/: Logging utilities and decorators (preferred import path)
- performance/: Performance monitoring and optimization
- notebooks/: Jupyter notebook integration
- numerical/: Numerical computation utilities
- data/: Data handling and validation

Deprecation Note (v0.19.0):
    Logging re-exports at this level (e.g., `from mfgarchon.utils import get_logger`)
    are deprecated. Import directly from `mfgarchon.utils.mfg_logging` instead.
"""

# NOTE: The `logging` alias (from . import mfg_logging as logging) was removed in v0.17.0
# Users should import from mfgarchon.utils.mfg_logging or use the re-exported functions directly

# Core utility functions (non-plotting)
from .aux_func import npart, ppart
from .convergence import (
    ConvergenceWrapper,
    DistributionComparator,
    DistributionConvergenceMonitor,
    RollingConvergenceMonitor,
    SolverTypeDetector,
    adaptive_convergence,
    calculate_error,
    compute_norm,
    create_distribution_monitor,
    create_rolling_monitor,
    l2_change,
    sweep_change,
    test_particle_detection,
    wrap_solver_with_adaptive_convergence,
)
from .data.polars_integration import (  # noqa: F401
    POLARS_AVAILABLE,
    MFGDataFrame,
    benchmark_polars_vs_pandas,
    create_data_exporter,
    create_mfg_dataframe,
    create_parameter_sweep_analyzer,
    create_time_series_analyzer,
)

# Data recorder with pluggable backends (NPZ, HDF5)
from .data.recorder import (  # noqa: F401
    BaseRecorder,
    DataRecorderProtocol,
    HDF5Recorder,
    NPZRecorder,
    create_recorder,
    get_convergence_history,
    inspect_experiment,
    load_experiment,
)

# Subdirectory imports
from .data.validation import (
    safe_solution_return,
    validate_convergence_parameters,
    validate_mfg_solution,
    validate_solution_array,
)
from .exceptions import (
    ConfigurationError,
    ConvergenceError,
    DimensionMismatchError,
    MFGSolverError,
    NumericalInstabilityError,
    SolutionNotAvailableError,
    check_numerical_stability,
    validate_array_dimensions,
    validate_parameter_value,
    validate_solver_state,
)
from .memory_management import (  # noqa: F401
    MemoryMonitor,
    check_system_memory_availability,
    estimate_problem_memory_requirements,
    memory_usage_report,
)

# =============================================================================
# Logging utilities - DEPRECATED at this level (Issue #620 Phase 2)
# Prefer: from mfgarchon.utils.mfg_logging import get_logger, configure_research_logging
# These re-exports will be removed in v0.19.0
# =============================================================================
from .mfg_logging.decorators import (
    LoggingMixin,
    add_logging_to_class,
    logged_operation,
    logged_solver_method,
    logged_validation,
    performance_logged,
)
from .mfg_logging.logger import (
    LoggedOperation,
    configure_logging,
    configure_research_logging,
    get_logger,
    log_performance_metric,
    log_solver_completion,
    log_solver_progress,
    log_solver_start,
    log_validation_error,
)
from .notebooks.reporting import (  # noqa: F401
    MFGNotebookReporter,
    create_comparative_analysis,
    create_mfg_research_report,
)
from .numerical.integration import get_integration_info, trapezoid

# NOTE: adaptive_bandwidth_selection alias was removed in v0.17.0
# Use estimate_kde_bandwidth directly instead
from .numerical.particle.interpolation import (
    estimate_kde_bandwidth,
    interpolate_grid_to_particles,
    interpolate_particles_to_grid,
)
from .performance.monitoring import (  # noqa: F401
    PerformanceMonitor,
    benchmark_solver,
    get_performance_report,
    performance_tracked,
)
from .solver_result import ConvergenceResult, MFGSolverResult, SolverResult, create_solver_result

# Public API - Core utilities always available
# Keep organized with comments, don't auto-sort
__all__ = [
    # Availability flag: whether polars is installed (re-exported from data.polars_integration)
    "POLARS_AVAILABLE",
    # Convergence monitoring (new names)
    "ConvergenceWrapper",
    "DistributionConvergenceMonitor",
    "RollingConvergenceMonitor",
    "SolverTypeDetector",
    "calculate_error",
    "l2_change",
    "sweep_change",
    "create_distribution_monitor",
    "create_rolling_monitor",
    # Array norm (#2566)
    "compute_norm",
    # Exception handling
    "ConfigurationError",
    "ConvergenceError",
    # Solver results
    "ConvergenceResult",
    "DimensionMismatchError",
    "DistributionComparator",
    # Callable signature adapter (Issue #684)
    # Hamiltonian signature adapter
    # Logging
    "LoggedOperation",
    # Logging decorators
    "LoggingMixin",
    "MFGSolverError",
    "MFGSolverResult",
    "NumericalInstabilityError",
    "SolutionNotAvailableError",
    "SolverResult",
    # NOTE: adaptive_bandwidth_selection was removed in v0.17.0, use estimate_kde_bandwidth
    "adaptive_convergence",
    "add_logging_to_class",
    "check_numerical_stability",
    "configure_logging",
    "configure_research_logging",
    "create_solver_result",
    # Integration utilities
    "get_integration_info",
    # Particle interpolation
    "interpolate_grid_to_particles",
    "interpolate_particles_to_grid",
    "estimate_kde_bandwidth",
    "get_logger",
    "log_performance_metric",
    "log_solver_completion",
    "log_solver_progress",
    "log_solver_start",
    "log_validation_error",
    "logged_operation",
    "logged_solver_method",
    "logged_validation",
    # Auxiliary functions
    "npart",
    "performance_logged",
    "ppart",
    # Validation
    "safe_solution_return",
    "test_particle_detection",
    "trapezoid",
    "validate_array_dimensions",
    "validate_convergence_parameters",
    "validate_mfg_solution",
    "validate_parameter_value",
    "validate_solution_array",
    "validate_solver_state",
    "wrap_solver_with_adaptive_convergence",
]

# Re-exported submodule APIs
__all__.extend(
    [
        "MFGNotebookReporter",
        "create_comparative_analysis",
        "create_mfg_research_report",
    ]
)

__all__.extend(
    [
        "MFGDataFrame",
        "benchmark_polars_vs_pandas",
        "create_data_exporter",
        "create_mfg_dataframe",
        "create_parameter_sweep_analyzer",
        "create_time_series_analyzer",
    ]
)


__all__.extend(
    [
        "MemoryMonitor",
        "check_system_memory_availability",
        "estimate_problem_memory_requirements",
        "memory_usage_report",
    ]
)

__all__.extend(
    [
        "PerformanceMonitor",
        "benchmark_solver",
        "get_performance_report",
        "performance_tracked",
    ]
)

__all__.extend(
    [
        "BaseRecorder",
        "DataRecorderProtocol",
        "HDF5Recorder",
        "NPZRecorder",
        "create_recorder",
        "load_experiment",
        "get_convergence_history",
        "inspect_experiment",
    ]
)

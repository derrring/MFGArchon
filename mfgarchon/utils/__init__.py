"""
MFGarchon Utilities Module.

This module contains core utilities for the MFGarchon package.
All plotting functionality has been permanently moved to mfgarchon.visualization.

For visualization, use:
    from mfgarchon.visualization import create_visualization_manager
    from mfgarchon.visualization import plot_convergence, plot_results

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

# Adjoint duality validation (Issue #580)
from .adjoint_validation import (
    DualityStatus,
    DualityValidationResult,
    check_solver_duality,
    validate_scheme_config,
)

# Core utility functions (non-plotting)
from .aux_func import npart, ppart
from .callable_adapter import CallableSignature, adapt_ic_callable
from .convergence import (
    # NEW: Convergence checkers
    ConvergenceChecker,
    # New names (preferred)
    ConvergenceConfig,
    ConvergenceWrapper,
    DistributionComparator,
    DistributionConvergenceMonitor,
    FPConvergenceChecker,
    HJBConvergenceChecker,
    MFGConvergenceChecker,
    RollingConvergenceMonitor,
    SolverTypeDetector,
    adaptive_convergence,
    calculate_error,
    calculate_l2_convergence_metrics,
    compute_norm,
    create_distribution_monitor,
    create_rolling_monitor,
    test_particle_detection,
    wrap_solver_with_adaptive_convergence,
)

# Subdirectory imports
from .data.validation import (
    safe_solution_return,
    validate_convergence_parameters,
    validate_mfg_solution,
    validate_solution_array,
)

# Drift field helpers for FP solvers
from .drift_helpers import (
    composite_drift,
    density_dependent_drift,
    optimal_control_drift,
    prescribed_drift,
    zero_drift,
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
from .geometry import (
    BoxObstacle,
    CircleObstacle,
    Complement,
    Difference,
    Hyperrectangle,
    Hypersphere,
    ImplicitDomain,
    Intersection,
    RectangleObstacle,
    SphereObstacle,
    Union,
    create_box_obstacle,
    create_circle_obstacle,
    create_rectangle_obstacle,
    create_sphere_obstacle,
)
from .hamiltonian_adapter import HamiltonianAdapter, adapt_hamiltonian, create_hamiltonian_adapter

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
from .numerical.integration import get_integration_info, trapezoid
from .numerical.particle.interpolation import (
    estimate_kde_bandwidth,
    interpolate_grid_to_particles,
    interpolate_particles_to_grid,
)
from .numerical.qp_utils import QPCache, QPSolver
from .numerical.sdf_utils import (
    sdf_box,
    sdf_complement,
    sdf_difference,
    sdf_gradient,
    sdf_intersection,
    sdf_smooth_intersection,
    sdf_smooth_union,
    sdf_sphere,
    sdf_union,
)

# PDE coefficient handling
from .pde_coefficients import CoefficientField, CoefficientMode, check_adi_compatibility, get_spatial_grid
from .solver_result import ConvergenceResult, MFGSolverResult, SolverResult, create_solver_result
from .sparse_operations import (
    SparseMatrixBuilder,
    SparseSolver,
    estimate_sparsity,
    sparse_matmul,
)

# NOTE: adaptive_bandwidth_selection alias was removed in v0.17.0
# Use estimate_kde_bandwidth directly instead

# Optional modules with graceful handling
try:
    from .notebooks.reporting import (  # noqa: F401
        MFGNotebookReporter,
        create_comparative_analysis,
        create_mfg_research_report,
    )

    NOTEBOOK_REPORTING_AVAILABLE = True
except ImportError:
    NOTEBOOK_REPORTING_AVAILABLE = False

try:
    from .data.polars_integration import (
        MFGDataFrame,  # noqa: F401
        benchmark_polars_vs_pandas,  # noqa: F401
        create_data_exporter,  # noqa: F401
        create_mfg_dataframe,  # noqa: F401
        create_parameter_sweep_analyzer,  # noqa: F401
        create_time_series_analyzer,  # noqa: F401
    )

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

# Utility modules
try:
    from .cli import main as cli_main  # noqa: F401

    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False

try:
    from .experiment_manager import (
        load_experiment_data,  # noqa: F401
        load_experiments_from_dir,  # noqa: F401
        plot_comparison_final_m,  # noqa: F401
        plot_comparison_total_mass,  # noqa: F401
        save_experiment_data,  # noqa: F401
    )

    EXPERIMENT_MANAGER_AVAILABLE = True
except ImportError:
    EXPERIMENT_MANAGER_AVAILABLE = False

# Data recorder with pluggable backends (NPZ, HDF5)
try:
    from .data.recorder import (
        BaseRecorder,  # noqa: F401
        DataRecorderProtocol,  # noqa: F401
        HDF5Recorder,  # noqa: F401
        NPZRecorder,  # noqa: F401
        create_recorder,  # noqa: F401
        get_convergence_history,  # noqa: F401
        inspect_experiment,  # noqa: F401
        load_experiment,  # noqa: F401
    )

    DATA_RECORDER_AVAILABLE = True
except ImportError:
    DATA_RECORDER_AVAILABLE = False

# HDF5_RECORDER_AVAILABLE deprecated - use DATA_RECORDER_AVAILABLE instead
HDF5_RECORDER_AVAILABLE = DATA_RECORDER_AVAILABLE  # Backward compatibility alias

try:
    from .memory_management import (
        MemoryMonitor,  # noqa: F401
        check_system_memory_availability,  # noqa: F401
        estimate_problem_memory_requirements,  # noqa: F401
        memory_usage_report,  # noqa: F401
    )

    MEMORY_MANAGEMENT_AVAILABLE = True
except ImportError:
    MEMORY_MANAGEMENT_AVAILABLE = False

try:
    from .performance.monitoring import (
        PerformanceMonitor,  # noqa: F401
        benchmark_solver,  # noqa: F401
        get_performance_report,  # noqa: F401
        performance_tracked,  # noqa: F401
    )

    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False

# Availability info
AVAILABLE_MODULES = {
    "notebook_reporting": NOTEBOOK_REPORTING_AVAILABLE,
    "polars_integration": POLARS_AVAILABLE,
    "cli": CLI_AVAILABLE,
    "experiment_manager": EXPERIMENT_MANAGER_AVAILABLE,
    "data_recorder": DATA_RECORDER_AVAILABLE,
    "hdf5_recorder": HDF5_RECORDER_AVAILABLE,  # Deprecated, use data_recorder
    "memory_management": MEMORY_MANAGEMENT_AVAILABLE,
    "performance_monitoring": PERFORMANCE_MONITORING_AVAILABLE,
}

# Public API - Core utilities always available
# Keep organized with comments, don't auto-sort
__all__ = [
    # Availability flags
    "AVAILABLE_MODULES",
    "CLI_AVAILABLE",
    "EXPERIMENT_MANAGER_AVAILABLE",
    "HDF5_RECORDER_AVAILABLE",
    "MEMORY_MANAGEMENT_AVAILABLE",
    "NOTEBOOK_REPORTING_AVAILABLE",
    "PERFORMANCE_MONITORING_AVAILABLE",
    "POLARS_AVAILABLE",
    # Adjoint duality validation (Issue #580)
    "DualityStatus",
    "DualityValidationResult",
    "check_solver_duality",
    "validate_scheme_config",
    # PDE coefficient handling
    "CoefficientField",
    "CoefficientMode",
    "check_adi_compatibility",
    "get_spatial_grid",
    # Drift field helpers for FP solvers
    "zero_drift",
    "optimal_control_drift",
    "prescribed_drift",
    "density_dependent_drift",
    "composite_drift",
    # Convergence monitoring (new names)
    "ConvergenceConfig",
    "ConvergenceWrapper",
    "DistributionConvergenceMonitor",
    "RollingConvergenceMonitor",
    "SolverTypeDetector",
    "calculate_error",
    "calculate_l2_convergence_metrics",
    "create_distribution_monitor",
    "create_rolling_monitor",
    # Convergence checkers (Protocol-based)
    "ConvergenceChecker",
    "HJBConvergenceChecker",
    "FPConvergenceChecker",
    "MFGConvergenceChecker",
    "compute_norm",
    # Geometry utilities (obstacles, SDF)
    "BoxObstacle",
    "CircleObstacle",
    "Complement",
    # Exception handling
    "ConfigurationError",
    "ConvergenceError",
    # Solver results
    "ConvergenceResult",
    # CSG operations
    "Difference",
    "DimensionMismatchError",
    "DistributionComparator",
    # Geometry primitives
    "Hyperrectangle",
    "Hypersphere",
    "ImplicitDomain",
    "Intersection",
    # Callable signature adapter (Issue #684)
    "CallableSignature",
    "adapt_ic_callable",
    # Hamiltonian signature adapter
    "HamiltonianAdapter",
    "adapt_hamiltonian",
    "create_hamiltonian_adapter",
    # Logging
    "LoggedOperation",
    # Logging decorators
    "LoggingMixin",
    "MFGSolverError",
    "MFGSolverResult",
    "NumericalInstabilityError",
    "SolutionNotAvailableError",
    "SolverResult",
    "SparseMatrixBuilder",
    "SparseSolver",
    # NOTE: adaptive_bandwidth_selection was removed in v0.17.0, use estimate_kde_bandwidth
    "adaptive_convergence",
    "add_logging_to_class",
    "check_numerical_stability",
    "configure_logging",
    "configure_research_logging",
    "create_solver_result",
    "estimate_sparsity",
    # Integration utilities
    "get_integration_info",
    # Particle interpolation
    "interpolate_grid_to_particles",
    "interpolate_particles_to_grid",
    "estimate_kde_bandwidth",
    # Signed distance functions
    "sdf_box",
    "sdf_complement",
    "sdf_difference",
    "sdf_gradient",
    "sdf_intersection",
    "sdf_smooth_intersection",
    "sdf_smooth_union",
    "sdf_sphere",
    "sdf_union",
    # QP utilities
    "QPCache",
    "QPSolver",
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
    # Geometry aliases
    "RectangleObstacle",
    "SphereObstacle",
    # Validation
    "safe_solution_return",
    "sparse_matmul",
    "test_particle_detection",
    "trapezoid",
    # CSG helpers
    "Union",
    "validate_array_dimensions",
    "validate_convergence_parameters",
    "validate_mfg_solution",
    "validate_parameter_value",
    "validate_solution_array",
    "validate_solver_state",
    "wrap_solver_with_adaptive_convergence",
    # Geometry factory functions
    "create_box_obstacle",
    "create_circle_obstacle",
    "create_rectangle_obstacle",
    "create_sphere_obstacle",
    # Data recorder availability
    "DATA_RECORDER_AVAILABLE",
]

# Add optional modules to public API if available
if NOTEBOOK_REPORTING_AVAILABLE:
    __all__.extend(
        [
            "MFGNotebookReporter",
            "create_comparative_analysis",
            "create_mfg_research_report",
        ]
    )

if POLARS_AVAILABLE:
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

if CLI_AVAILABLE:
    __all__.extend(
        [
            "cli_main",
        ]
    )

if EXPERIMENT_MANAGER_AVAILABLE:
    __all__.extend(
        [
            "load_experiment_data",
            "load_experiments_from_dir",
            "plot_comparison_final_m",
            "plot_comparison_total_mass",
            "save_experiment_data",
        ]
    )

if MEMORY_MANAGEMENT_AVAILABLE:
    __all__.extend(
        [
            "MemoryMonitor",
            "check_system_memory_availability",
            "estimate_problem_memory_requirements",
            "memory_usage_report",
        ]
    )

if PERFORMANCE_MONITORING_AVAILABLE:
    __all__.extend(
        [
            "PerformanceMonitor",
            "benchmark_solver",
            "get_performance_report",
            "performance_tracked",
        ]
    )

if DATA_RECORDER_AVAILABLE:
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

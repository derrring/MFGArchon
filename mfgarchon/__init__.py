from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mfgarchon")  # Matches the name in pyproject.toml
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0-dev"  # Or some other placeholder

# Ensure NumPy 2.0+ compatibility on import
from .utils.numpy_compat import ensure_numpy_compatibility

_numpy_info = ensure_numpy_compatibility()

from .alg.numerical.fp_solvers.fp_particle import KDENormalization  # noqa: E402
from .alg.optimization.variational_problem import (  # noqa: E402
    VariationalMFGComponents,
    VariationalMFGProblem,
)
from .config import MFGSolverConfig  # noqa: E402
from .core.hamiltonian import (  # noqa: E402
    # Control cost classes (original)
    ControlCostBase,
    # Dual classes (Legendre transform)
    DualHamiltonian,
    DualLagrangian,
    # Full MFG Hamiltonian classes (Issue #673)
    HamiltonianBase,
    # Lagrangian classes (Issue #651, #904)
    LagrangianBase,
    OptimizationSense,
    QuadraticControlCost,
    QuadraticMFGHamiltonian,
    SeparableHamiltonian,
    SeparableLagrangian,
    create_hamiltonian,
)
from .core.mfg_problem import (  # noqa: E402
    MFGComponents,
    MFGProblem,
)
from .core.model import Conditions, ErgodicConditions, Model  # noqa: E402
from .extensions.topology import (  # noqa: E402
    NetworkMFGComponents,
    NetworkMFGProblem,
    create_grid_mfg_problem,
    create_random_mfg_problem,
    create_scale_free_mfg_problem,
)
from .factory import (  # noqa: E402
    SolverFactory,
    create_solver,
)
from .geometry import (  # noqa: E402
    BoundaryConditions,
    GridNetwork,
    NetworkBackendType,
    NetworkData,
    NetworkType,
    OperationType,
    RandomNetwork,
    ScaleFreeNetwork,
    compute_network_statistics,
    create_network,
    get_backend_manager,
    set_preferred_backend,
)
from .types import NumericalScheme  # noqa: E402

# Geometry system for 2D/3D complex domains (optional dependency)
try:
    from .geometry import (
        BoundaryManager,  # noqa: F401
        GeometricBoundaryCondition,  # noqa: F401
        Mesh2D,  # noqa: F401
        Mesh3D,  # noqa: F401
        MeshData,  # noqa: F401
        MeshManager,  # noqa: F401
        MeshPipeline,  # noqa: F401
    )

    GEOMETRY_SYSTEM_AVAILABLE = True
except ImportError:
    GEOMETRY_SYSTEM_AVAILABLE = False

# High-dimensional MFG capabilities - REMOVED in v0.14.0
# Use MFGProblem directly with spatial_bounds and spatial_discretization parameters
# See: docs/migration/DEPRECATION_MODERNIZATION_GUIDE.md
HIGHDIM_MFG_AVAILABLE = False

# Interactive research reporting (optional dependency)
try:
    from .utils.notebooks.reporting import (  # noqa: F401
        MFGNotebookReporter,
        create_comparative_analysis,
        create_mfg_research_report,
    )

    NOTEBOOK_REPORTING_AVAILABLE = True
except ImportError:
    NOTEBOOK_REPORTING_AVAILABLE = False


# Optional dependency utilities
def show_optional_features() -> None:
    """
    Display status of all optional features and dependencies.

    Prints a formatted table showing which optional dependencies are installed
    and available, along with installation instructions for missing features.

    Example:
        >>> import mfgarchon
        >>> mfgarchon.show_optional_features()
        MFGarchon Optional Features
        ==================================================
        pytorch        : ✓ Available
        jax            : ✗ Not installed
        ...
    """
    from .utils.dependencies import show_optional_features as _show_features

    _show_features()


# Public API exports
__all__ = [
    # Geometry
    "BoundaryConditions",
    # General MFG factory
    # Network geometry
    "GridNetwork",
    # Solver enums and configuration
    "KDENormalization",
    "MFGComponents",
    "MFGProblem",
    # API v1.0 classes (Issue #875)
    "Model",
    "Conditions",
    "ErgodicConditions",
    # Hamiltonian/Lagrangian classes (Issues #651, #667, #673)
    "ControlCostBase",
    "OptimizationSense",
    "QuadraticControlCost",
    "HamiltonianBase",
    "SeparableHamiltonian",
    "QuadraticMFGHamiltonian",
    "create_hamiltonian",
    "LagrangianBase",
    "SeparableLagrangian",
    "DualHamiltonian",
    "DualLagrangian",
    # Configuration
    "MFGSolverConfig",
    # Numerical schemes (Issue #580)
    "NumericalScheme",
    # Network backend
    "NetworkBackendType",
    "NetworkData",
    # Network MFG
    "NetworkMFGComponents",
    "NetworkMFGProblem",
    "NetworkType",
    "OperationType",
    "RandomNetwork",
    "ScaleFreeNetwork",
    # Factory methods
    "SolverFactory",
    # Variational MFG
    "VariationalMFGComponents",
    "VariationalMFGProblem",
    "compute_network_statistics",
    "create_general_mfg_problem",
    "create_grid_mfg_problem",
    "create_network",
    "create_random_mfg_problem",
    "create_scale_free_mfg_problem",
    "create_solver",
    "get_backend_manager",
    "set_preferred_backend",
    "show_optional_features",
]

# Add conditionally available imports to __all__ when available
if GEOMETRY_SYSTEM_AVAILABLE:
    __all__.extend(
        [
            "BoundaryManager",
            "GeometricBoundaryCondition",
            "Mesh2D",
            "Mesh3D",
            "MeshData",
            "MeshManager",
            "MeshPipeline",
        ]
    )

# GridBasedMFGProblem, HighDimMFGProblem, HybridMFGSolver removed in v0.14.0
# Use MFGProblem directly instead

if NOTEBOOK_REPORTING_AVAILABLE:
    __all__.extend(
        [
            "MFGNotebookReporter",
            "create_comparative_analysis",
            "create_mfg_research_report",
        ]
    )

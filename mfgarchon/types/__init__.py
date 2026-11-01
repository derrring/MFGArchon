"""
Advanced Types for MFGarchon

This module contains type definitions for advanced users who need
to customize solver behavior or implement custom algorithms.

⚠️  Advanced API Warning:
    These types are stable but considered "advanced". They may change
    between major versions. For simple usage, stick to the main API
    in mfgarchon.__init__.py

Basic Types (safe to use):
    from mfgarchon.types import MFGProblem, MFGResult, SolverConfig
    from mfgarchon.types import SolutionArray, SpatialGrid, TimeGrid

Internal Types (use with caution):
    from mfgarchon.types import SpatialTemporalState, HamiltonianFunction
"""

# Re-export the most commonly needed types
from .arrays import SolutionArray, SpatialGrid, TimeGrid
from .callable_protocols import (
    CoefficientFieldType,
    DiffusionFieldCallable,
    DriftFieldCallable,
    FinalValueCallable,
    HamiltonianCallable,
    HamiltonianDerivativeCallable,
    InitialDensityCallable,
    PotentialCallable,
)
from .pde_coefficients import DiffusionCallable, DiffusionField, DriftCallable, DriftField
from .protocols import (
    FPSolverProtocol,
    HJBSolverProtocol,
    MFGProblem,
    MFGResult,
    MFGSolver,
    SolverConfig,
)
from .schemes import NumericalScheme
from .solver_types import HamiltonianJacobians, SolverReturnTuple
from .state import ConvergenceInfo, SpatialTemporalState

__all__ = [
    # Core types
    "ConvergenceInfo",
    "FPSolverProtocol",
    "HamiltonianJacobians",
    "HJBSolverProtocol",
    "MFGProblem",
    "MFGResult",
    "MFGSolver",
    "NumericalScheme",
    "SolutionArray",
    "SolverConfig",
    "SolverReturnTuple",
    "SpatialGrid",
    "SpatialTemporalState",
    "TimeGrid",
    # Legacy coefficient types (kept for backward compatibility)
    "DiffusionCallable",
    "DiffusionField",
    "DriftCallable",
    "DriftField",
    # New callable protocols (Phase 3.5 - enhanced type hints)
    "CoefficientFieldType",
    "DiffusionFieldCallable",
    "DriftFieldCallable",
    "FinalValueCallable",
    "HamiltonianCallable",
    "HamiltonianDerivativeCallable",
    "InitialDensityCallable",
    "PotentialCallable",
]

# Array types available via explicit import:
# from mfgarchon.types.arrays import SpatialArray, TemporalArray, ParticleArray, ...

# Solver types and advanced types available via explicit import:
# from mfgarchon.types.solver_types import JAXSolverReturn, ComplexSolverState, MetadataDict, ...
# from mfgarchon.types.solver_types import HamiltonianFunction, NewtonSolver, ...

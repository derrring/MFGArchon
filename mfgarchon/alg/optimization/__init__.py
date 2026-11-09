"""
Optimization paradigm for MFG problems.

This module contains direct optimization approaches for solving Mean Field Games:
- variational_solvers: Lagrangian formulation and variational methods
- optimal_transport: Wasserstein distance and optimal transport methods

All methods are based on direct optimization of cost functionals,
providing alternatives to the classical HJB-FP system approach.
"""

from mfgarchon.alg.base_solver import BaseOptimizationSolver

# Optimal transport solvers exist only when POT is installed: the subpackage defines them under that condition and
# reports it in POT_AVAILABLE, so they are imported only then.
from .optimal_transport import POT_AVAILABLE, SCIPY_AVAILABLE

# Import variational problem formulation
from .variational_problem import (
    VariationalMFGComponents,
    VariationalMFGProblem,
)

# Import variational solvers
from .variational_solvers import (
    BaseVariationalSolver,
    PrimalDualMFGSolver,
    VariationalMFGSolver,
    VariationalSolverResult,
)

OT_IMPORTS_AVAILABLE = SCIPY_AVAILABLE and POT_AVAILABLE

__all__ = [
    "BaseOptimizationSolver",
    # Variational Problem
    "VariationalMFGComponents",
    "VariationalMFGProblem",
    # Variational Solvers
    "BaseVariationalSolver",
    "PrimalDualMFGSolver",
    "VariationalMFGSolver",
    "VariationalSolverResult",
]

if OT_IMPORTS_AVAILABLE:
    from .optimal_transport import (
        SinkhornMFGSolver,
        SinkhornSolverConfig,
        SinkhornSolverResult,
        WassersteinMFGSolver,
        WassersteinSolverConfig,
        WassersteinSolverResult,
    )

    __all__ += [
        "SinkhornMFGSolver",
        "SinkhornSolverConfig",
        "SinkhornSolverResult",
        "WassersteinMFGSolver",
        "WassersteinSolverConfig",
        "WassersteinSolverResult",
    ]

# Always export availability info
__all__.extend(["OT_IMPORTS_AVAILABLE"])

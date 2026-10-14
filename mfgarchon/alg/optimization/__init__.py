"""
Optimization paradigm for MFG problems.

This module contains direct optimization approaches for solving Mean Field Games:
- variational_solvers: Lagrangian formulation and variational methods
- optimal_transport: Wasserstein distance and optimal transport methods

All methods are based on direct optimization of cost functionals,
providing alternatives to the classical HJB-FP system approach.
"""

from mfgarchon.alg.base_solver import BaseOptimizationSolver

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

# Import optimal transport solvers (conditional on dependencies)
try:
    from .optimal_transport import (
        POT_AVAILABLE,
        SCIPY_AVAILABLE,
        SinkhornMFGSolver,
        SinkhornSolverConfig,
        SinkhornSolverResult,
        WassersteinMFGSolver,
        WassersteinSolverConfig,
        WassersteinSolverResult,
    )

    OT_IMPORTS_AVAILABLE = SCIPY_AVAILABLE and POT_AVAILABLE

    if OT_IMPORTS_AVAILABLE:
        __all__ = [
            "BaseOptimizationSolver",
            # Variational Problem
            "VariationalMFGComponents",
            "VariationalMFGProblem",
            # Variational Solvers
            "BaseVariationalSolver",
            "PrimalDualMFGSolver",
            "SinkhornMFGSolver",
            "SinkhornSolverConfig",
            "SinkhornSolverResult",
            "VariationalMFGSolver",
            "VariationalSolverResult",
            # Optimal Transport Solvers
            "WassersteinMFGSolver",
            "WassersteinSolverConfig",
            "WassersteinSolverResult",
        ]
    else:
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

except ImportError:
    # Optimal transport not available
    OT_IMPORTS_AVAILABLE = False
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

# Always export availability info
__all__.extend(["OT_IMPORTS_AVAILABLE"])

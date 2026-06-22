"""
Numerical methods paradigm for MFG problems.

This module contains classical numerical analysis approaches:
- hjb_solvers: Individual HJB equation solvers (grid-based)
- fp_solvers: Individual Fokker-Planck solvers (grid-based)
- network_solvers: HJB and FP solvers for graph/network domains
- coupling: MFG coupling methods (Picard, Policy iteration, etc.)

All methods are based on discretization and convergence analysis.
Note: FP solvers maintain backward compatibility with original interfaces.
"""

from mfgarchon.alg.base_solver import BaseNumericalSolver

# Import MFG coupling methods
# Note: ParticleCollocationSolver has been removed from core package
from .coupling import (
    BaseMFGSolver,
    FixedPointIterator,
)

# Import FP solvers
from .fp_solvers import (
    BaseFPSolver,
    FPFDMSolver,
    FPFVMSolver,
    FPNetworkSolver,  # Re-exported for backward compat (now in network_solvers)
    FPParticleSolver,
)

# Import HJB solvers
from .hjb_solvers import (
    BaseHJBSolver,
    HJBFDMSolver,
    HJBGFDMSolver,
    HJBSemiLagrangianSolver,
    HJBWENOSolver,
    HJBWenoSolver,
)

# Import Network solvers (graph-based MFG problems)
from .network_solvers import (
    HJBNetworkSolver,
    NetworkHJBSolver,
)

__all__ = [
    # Base Classes
    "BaseFPSolver",
    "BaseHJBSolver",
    "BaseMFGSolver",
    "BaseNumericalSolver",
    # FP Solvers (grid-based)
    "FPFDMSolver",
    "FPFVMSolver",
    "FPParticleSolver",
    # HJB Solvers (grid-based)
    "HJBFDMSolver",
    "HJBGFDMSolver",
    "HJBSemiLagrangianSolver",
    "HJBWENOSolver",
    "HJBWenoSolver",
    # Network Solvers (graph-based)
    "FPNetworkSolver",  # Also in fp_solvers for backward compat
    "HJBNetworkSolver",  # Alias for NetworkHJBSolver
    "NetworkHJBSolver",
    # Coupling Methods
    "FixedPointIterator",
]

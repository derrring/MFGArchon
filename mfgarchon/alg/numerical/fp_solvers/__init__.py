"""
Fokker-Planck equation solvers using numerical methods.

This module provides classical numerical analysis approaches for solving
individual Fokker-Planck equations, including:
- Finite difference methods (FPFDMSolver)
- Particle-based methods (FPParticleSolver)
- Semi-Lagrangian methods (FPSLSolver)
- GFDM meshfree methods (FPGFDMSolver)

Semi-Lagrangian Variants (Issue #710):
- FPSLSolver: Forward SL (scatter/splat) - adjoint of HJB SL, RECOMMENDED
- FPSLJacobianSolver: Backward SL with Jacobian correction - DEPRECATED

Note: FPSLAdjointSolver is a deprecated alias for FPSLSolver (renamed in v0.17.6).

Internal modules (Issue #635 refactoring):
- fp_particle_density: Dimension-agnostic density estimation utilities
- fp_particle_bc: Dimension-agnostic boundary condition handling

Note: Network solvers have been moved to `mfgarchon.alg.numerical.network_solvers`.
FPNetworkSolver is re-exported here for backward compatibility.
"""

# Re-export for backward compatibility (moved to network_solvers)
from mfgarchon.alg.numerical.network_solvers import FPNetworkSolver

from .base_fp import BaseFPSolver
from .fp_fdm import FPFDMSolver
from .fp_fvm import FPFVMSolver
from .fp_gfdm import FPGFDMSolver
from .fp_particle import FPParticleSolver, KDEMethod, KDENormalization

# FPSLJacobianSolver (Backward SL) is deprecated
from .fp_semi_lagrangian import FPSLJacobianSolver

# FPSLSolver (Forward SL) is the recommended solver - exported from adjoint file
from .fp_semi_lagrangian_adjoint import (
    FPSLAdjointSolver,  # Deprecated alias
    FPSLSolver,
)
from .particle_density_query import ParticleDensityQuery
from .particle_result import FPParticleResult

__all__ = [
    "BaseFPSolver",
    "FPFDMSolver",
    "FPFVMSolver",  # Issue #422 - conservative finite-volume FP solver
    "FPGFDMSolver",
    "FPNetworkSolver",  # Backward compat - prefer network_solvers import
    "FPParticleSolver",
    "FPSLSolver",  # Forward SL (adjoint of HJB SL) - RECOMMENDED
    "FPSLJacobianSolver",  # Backward SL with Jacobian - DEPRECATED
    "FPSLAdjointSolver",  # Deprecated alias for FPSLSolver
    "KDEMethod",  # Issue #709 - KDE boundary correction methods
    "KDENormalization",
    "ParticleDensityQuery",  # Issue #489 - Direct particle query
    "FPParticleResult",  # Issue #489 - Result with query support
]

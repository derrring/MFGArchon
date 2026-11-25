"""
Coupled MFG system solvers using numerical methods.

This module provides complete Mean Field Games solvers that combine HJB and FP
solvers to solve the coupled MFG system using classical numerical approaches:
- Fixed point iterators (Picard iteration-based)
- Fictitious Play (decaying learning rate, proven convergence)
- Newton methods (quadratic convergence near solution)
- Hybrid methods combining different techniques

Note: Particle-collocation methods have been removed from core package.
"""

from .base_mfg import BaseCouplingIterator

# Block iterators (Issue #492 Phase 2)
from .block_iterators import (
    BlockGaussSeidelIterator,
    BlockIterator,
    BlockJacobiIterator,
    BlockMethod,
)
from .fictitious_play import FictitiousPlayIterator

# Coupling iterators
from .fixed_point_iterator import FixedPointIterator

# Shared utilities
from .fixed_point_utils import (
    check_convergence_criteria,
    preserve_initial_condition,
    preserve_terminal_condition,
)

# Newton family solvers (Issue #492)
from .mfg_residual import MFGResidual
from .newton_mfg_solver import NewtonMFGSolver

# Note: ParticleCollocationSolver has been removed from core package

# Backward compatibility alias
BaseMFGSolver = BaseCouplingIterator

__all__ = [
    "BaseCouplingIterator",
    "BaseMFGSolver",  # backward compat alias
    "FixedPointIterator",
    "FictitiousPlayIterator",
    # Block iterators (Issue #492 Phase 2)
    "BlockIterator",
    "BlockJacobiIterator",
    "BlockGaussSeidelIterator",
    "BlockMethod",
    # Newton family (Issue #492)
    "MFGResidual",
    "NewtonMFGSolver",
    # Utilities
    "check_convergence_criteria",
    "preserve_initial_condition",
    "preserve_terminal_condition",
]

# Solver categories for factory selection
FIXED_POINT_SOLVERS = [
    "FixedPointIterator",  # Picard iteration with fixed damping
    "FictitiousPlayIterator",  # Fictitious play with decaying learning rate
]

# Block iterators (Issue #492 Phase 2)
BLOCK_SOLVERS = [
    "BlockIterator",  # Unified block iterator (Jacobi or Gauss-Seidel)
    "BlockJacobiIterator",  # Block Jacobi (parallel updates)
    "BlockGaussSeidelIterator",  # Block Gauss-Seidel (sequential updates)
]

# Newton family solvers (Issue #492)
NEWTON_SOLVERS = [
    "NewtonMFGSolver",  # Newton's method for MFG coupling
]

# Note: PARTICLE_SOLVERS category removed - removed from core package

# JAX-accelerated solvers (optional)
JAX_SOLVERS: list[str] = []

ALL_MFG_SOLVERS = FIXED_POINT_SOLVERS + BLOCK_SOLVERS + NEWTON_SOLVERS + JAX_SOLVERS

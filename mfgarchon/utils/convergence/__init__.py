#!/usr/bin/env python3
"""
Convergence monitoring utilities for all solver paradigms.

This package provides convergence monitoring utilities that work across
all MFG solver paradigms (numerical, neural, optimization, reinforcement).

Modules:
- **convergence_metrics.py**: General-purpose utilities
  - DistributionComparator: Wasserstein, KL divergence, moments
  - RollingConvergenceMonitor: Window-based statistical convergence
  - calculate_error: Unified L1/L2/Linf error computation
  - ConvergenceConfig: Configuration dataclass for solver integration

- **convergence_monitors.py**: MFG-specific monitors
  - DistributionConvergenceMonitor: Multi-criteria (Wasserstein, KL, moments)
  - SolverTypeDetector: Detect particle vs grid-based solvers
  - ConvergenceWrapper: Adaptive convergence based on solver type

- **convergence_checkers.py**: Protocol-based convergence checking
  - ConvergenceChecker: Protocol for all checkers
  - HJBConvergenceChecker, FPConvergenceChecker, MFGConvergenceChecker

Usage:
    # General-purpose convergence checking
    from mfgarchon.utils.convergence import calculate_error, RollingConvergenceMonitor

    error = calculate_error(u_new, u_old, dx=0.01, norm='l2')
    monitor = RollingConvergenceMonitor(window_size=10)

    # MFG-specific monitoring
    from mfgarchon.utils.convergence import DistributionConvergenceMonitor

    monitor = DistributionConvergenceMonitor(wasserstein_tol=1e-4)
    diagnostics = monitor.update(u_current, u_previous, m_current, x_grid)

    # Solver integration
    from mfgarchon.utils.convergence import ConvergenceConfig, MFGConvergenceChecker

    config = ConvergenceConfig(tolerance=1e-6, require_both=True)
    checker = MFGConvergenceChecker(config)
    converged, metrics = checker.check(old_state, new_state)
"""

from __future__ import annotations

# =============================================================================
# CONVERGENCE CHECKERS (convergence_checkers.py)
# =============================================================================
from .convergence_checkers import (
    ConvergenceChecker,
    FPConvergenceChecker,
    HJBConvergenceChecker,
    MFGConvergenceChecker,
    compute_norm,
)

# =============================================================================
# GENERAL-PURPOSE IMPORTS (convergence_metrics.py)
# =============================================================================
from .convergence_metrics import (
    ConvergenceConfig,
    # Core utilities
    DistributionComparator,
    MomentConvergenceMonitor,
    RollingConvergenceMonitor,
    calculate_error,
    calculate_l2_convergence_metrics,
    # Factory
    create_moment_monitor,
    create_rolling_monitor,
)

# =============================================================================
# MFG-SPECIFIC IMPORTS (convergence_monitors.py)
# =============================================================================
from .convergence_monitors import (
    ConvergenceWrapper,
    # Core monitors
    DistributionConvergenceMonitor,
    SolverTypeDetector,
    # Internal (exposed for testing)
    _ErrorHistoryTracker,
    adaptive_convergence,
    # Factory and decorators
    create_distribution_monitor,
    test_particle_detection,
    wrap_solver_with_adaptive_convergence,
)

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # General-purpose (convergence_metrics.py)
    "DistributionComparator",
    "MomentConvergenceMonitor",
    "RollingConvergenceMonitor",
    "calculate_error",
    "calculate_l2_convergence_metrics",
    "ConvergenceConfig",
    "create_moment_monitor",
    "create_rolling_monitor",
    # MFG-specific (convergence_monitors.py)
    "DistributionConvergenceMonitor",
    "SolverTypeDetector",
    "ConvergenceWrapper",
    "_ErrorHistoryTracker",
    "create_distribution_monitor",
    "adaptive_convergence",
    "wrap_solver_with_adaptive_convergence",
    "test_particle_detection",
    # Convergence checkers (convergence_checkers.py)
    "ConvergenceChecker",
    "HJBConvergenceChecker",
    "FPConvergenceChecker",
    "MFGConvergenceChecker",
    "compute_norm",
]

"""
Numerical utilities for MFG computations.

This module provides numerical algorithms and helper functions commonly needed
in MFG research projects, including differential operators, kernel functions,
GFDM operators and particle interpolation.

Submodules:
- kernels: Kernel functions (Gaussian, Wendland, B-spline) for GFDM, KDE, SPH
- nonlinear_solvers: Newton, fixed-point, policy iteration solvers
- particle: Particle-based methods (Monte Carlo, MCMC, interpolation)
- tensor_calculus: Internal — regular grid operators (use mfgarchon.operators instead)
"""

# GFDM strategies are NOT re-exported here. They live in
# `mfgarchon.alg.numerical.gfdm_components.gfdm_strategies`; import them from there.
#
# A package `__init__` re-exports names of its own layer or below (#2938), and re-exporting them here
# also closes an import cycle. `mfgarchon.utils` sits BELOW `mfgarchon.alg`, so importing upward from
# here reaches `alg/__init__` -> fp_solvers -> network_solvers ->
# `fp_network.py:36`, which needs a name from `utils.numerical` while this file is still stopped
# at that line:
#
#     ImportError: cannot import name 'clip_nonnegative_or_raise' from partially initialized
#     module 'mfgarchon.utils.numerical' (most likely due to a circular import)
#
# #1930.

# Kernels - general numerical functions (not particle-specific)
from mfgarchon.utils.numerical.kernels import (
    CubicSplineKernel,
    GaussianKernel,
    MultiquadricKernel,
    PHSKernel,
    WendlandKernel,
    create_kernel,
)
from mfgarchon.utils.numerical.mass_fabrication_gate import (
    MAX_CLIP_MASS_FABRICATION,
    clip_nonnegative_or_raise,
    mass_fabricated_by_clip,
)
from mfgarchon.utils.numerical.monotonicity_stats import (
    MonotonicityStats,
    get_m_matrix_diagnostic_string,
    verify_m_matrix_property,
)
from mfgarchon.utils.numerical.nonlinear_solvers import (
    FixedPointSolver,
    NewtonSolver,
    PolicyIterationSolver,
    SolverInfo,
)

# Re-export particle utilities for convenience
from mfgarchon.utils.numerical.particle import (
    HamiltonianMonteCarlo,
    # Monte Carlo
    MCConfig,
    # MCMC
    MCMCConfig,
    MCMCResult,
    MCResult,
    MetropolisHastings,
    monte_carlo_integrate,
)

# Re-export particle interpolation from new location for backward compatibility
from mfgarchon.utils.numerical.particle.interpolation import (
    estimate_kde_bandwidth,
    interpolate_grid_to_particles,
    interpolate_particles_to_grid,
)

# Tensor Calculus functions (gradient, laplacian, etc.) are no longer re-exported here.
# They are internal infrastructure used by geometry/operators/wrappers.py.
# Public API: use mfgarchon.operators (LinearOperator classes) instead.

__all__ = [
    # GFDM operators (legacy; canonical: alg.numerical.gfdm_components.gfdm_strategies)
    # Monotonicity tracking
    "MonotonicityStats",
    "verify_m_matrix_property",
    "get_m_matrix_diagnostic_string",
    # Nonlinear solvers
    "FixedPointSolver",
    "NewtonSolver",
    "MAX_CLIP_MASS_FABRICATION",
    "PolicyIterationSolver",
    "clip_nonnegative_or_raise",
    "mass_fabricated_by_clip",
    "SolverInfo",
    # Particle interpolation (from particle submodule)
    "estimate_kde_bandwidth",
    "interpolate_grid_to_particles",
    "interpolate_particles_to_grid",
    # Kernels (from particle submodule)
    "GaussianKernel",
    "WendlandKernel",
    "CubicSplineKernel",
    "MultiquadricKernel",
    "PHSKernel",
    "create_kernel",
    # Monte Carlo (from particle submodule)
    "MCConfig",
    "MCResult",
    "monte_carlo_integrate",
    # MCMC (from particle submodule)
    "MCMCConfig",
    "MCMCResult",
    "MetropolisHastings",
    "HamiltonianMonteCarlo",
]

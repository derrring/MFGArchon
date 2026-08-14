"""
Numerical utilities for MFG computations.

This module provides numerical algorithms and helper functions commonly needed
in MFG research projects, including differential operators, kernel functions,
GFDM operators, particle interpolation, and signed distance functions.

Submodules:
- kernels: Kernel functions (Gaussian, Wendland, B-spline) for GFDM, KDE, SPH
- nonlinear_solvers: Newton, fixed-point, policy iteration solvers
- particle: Particle-based methods (Monte Carlo, MCMC, interpolation)
- tensor_calculus: Internal — regular grid operators (use mfgarchon.operators instead)
"""

# Flux diagnostics for mass conservation analysis
# GFDMOperator: deprecated, moved to _compat. Import without triggering warning.
# GFDM strategies are NOT re-exported here. They live in
# `mfgarchon.alg.numerical.gfdm_components.gfdm_strategies`; import them from there.
#
# This block was the back-edge of an import cycle. `mfgarchon.utils` sits BELOW `mfgarchon.alg`,
# so importing upward from here reached `alg/__init__` -> fp_solvers -> network_solvers ->
# `fp_network.py:36`, which needs a name from `utils.numerical` while this file is still stopped
# at this line:
#
#     ImportError: cannot import name 'clip_nonnegative_or_raise' from partially initialized
#     module 'mfgarchon.utils.numerical' (most likely due to a circular import)
#
# That error was invisible in normal use because `utils/__init__.py`'s eager
# `from .adjoint_validation import (...)` completes `utils.numerical` by another route first --
# so the cycle only surfaced when someone tried to make that import lazy, which is #1930 step 5.
# Measured: with this block present, deferring that import fails; with it gone, it works.
# No timing is claimed either way -- import totals on one machine drift 0.4s between rounds,
# so a two-row before/after would read as a measurement of this change when it is noise.
#
# Zero files imported these nine names through `utils.numerical` -- verified by AST over
# `mfgarchon/`, `tests/` and `examples/`, for both `from`-imports and attribute access. #1930

# SDF utilities (canonical location: geometry/implicit/)
from mfgarchon.geometry.implicit.sdf_utils import (
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

# Mesh distance metrics (canonical location: geometry/meshes/)
from mfgarchon.geometry.meshes.mesh_distances import (
    MeshDistances,
    compute_distances_for_eoc_study,
    compute_mesh_distances,
)
from mfgarchon.utils.numerical._compat.gfdm_operators import GFDMOperator
from mfgarchon.utils.numerical.flux_diagnostics import (
    BoundaryFluxResult,
    FluxDiagnostics,
    FluxSummary,
    compute_mass_conservation_error,
)
from mfgarchon.utils.numerical.hjb_policy_iteration import (
    HJBPolicyProblem,
    create_lq_policy_problem,
    policy_iteration_hjb,
)

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
    "GFDMOperator",
    # Monotonicity tracking
    "MonotonicityStats",
    "verify_m_matrix_property",
    "get_m_matrix_diagnostic_string",
    # HJB policy iteration
    "HJBPolicyProblem",
    "create_lq_policy_problem",
    "policy_iteration_hjb",
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
    # Flux diagnostics (mass conservation)
    "FluxDiagnostics",
    "BoundaryFluxResult",
    "FluxSummary",
    "compute_mass_conservation_error",
    # Mesh distances (EOC analysis)
    "MeshDistances",
    "compute_mesh_distances",
    "compute_distances_for_eoc_study",
]

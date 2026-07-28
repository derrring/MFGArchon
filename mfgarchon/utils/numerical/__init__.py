"""
Numerical utilities for MFG computations.

This module provides numerical algorithms and helper functions commonly needed
in MFG research projects, including differential operators, kernel functions,
GFDM operators, particle interpolation, and signed distance functions.

Submodules:
- gfdm_strategies: Differential operators for scattered points (GFDM/RBF-FD)
- kernels: Kernel functions (Gaussian, Wendland, B-spline) for GFDM, KDE, SPH
- nonlinear_solvers: Newton, fixed-point, policy iteration solvers
- particle: Particle-based methods (Monte Carlo, MCMC, interpolation)
- tensor_calculus: Internal — regular grid operators (use mfgarchon.operators instead)
"""

# Flux diagnostics for mass conservation analysis
# GFDMOperator: deprecated, moved to _compat. Import without triggering warning.
# GFDM Strategy Pattern (canonical location: alg/numerical/gfdm_components/)
from mfgarchon.alg.numerical.gfdm_components.gfdm_strategies import (
    BoundaryHandler,
    DifferentialOperator,
    DirectCollocationHandler,
    GhostNodeHandler,
    LocalRBFOperator,
    TaylorOperator,
    UpwindOperator,
    create_bc_handler,
    create_operator,
)

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
    # GFDM operators (legacy, use gfdm_strategies)
    "GFDMOperator",
    # GFDM Strategy Pattern (scattered points)
    "DifferentialOperator",
    "BoundaryHandler",
    "TaylorOperator",
    "UpwindOperator",
    "LocalRBFOperator",
    "DirectCollocationHandler",
    "GhostNodeHandler",
    "create_operator",
    "create_bc_handler",
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

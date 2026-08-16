"""
Boundary condition management for MFG problems.

Provides boundary condition specifications for 1D, 2D, and nD domains,
supporting Dirichlet, Neumann, Robin, periodic, and no-flux conditions.

Architecture (4-layer model, see BC_ENFORCEMENT_ARCHITECTURE.md):
- **Layer 1 — Specification**: types.py, conditions.py, providers.py
- **Layer 2 — Resolution**: resolution.py (BCResolver, MathBCType, ResolvedBC)
- **Layer 3 — Enforcement**: calculators.py, enforcement.py, ghost_cells.py
- **Layer 4 — Application**: applicator_*.py, dispatch.py

Primary API (covers 95% of usage)::

    from mfgarchon.geometry.boundary import (
        BCType, BCSegment, BoundaryConditions,
        neumann_bc, dirichlet_bc, periodic_bc, no_flux_bc, mixed_bc, robin_bc,
        pad_array_with_ghosts,
    )

For applicators, import from submodules::

    from mfgarchon.geometry.boundary.applicator_fdm import FDMApplicator
    from mfgarchon.geometry.boundary.bc_adapter import apply_fem_bc  # FEM via scikit-fem
    from mfgarchon.geometry.boundary.applicator_graph import GraphApplicator
    from mfgarchon.geometry.boundary.dispatch import apply_bc
"""

# =============================================================================
# Layer 1: Specification (user-facing)
# =============================================================================

# Core types
# =============================================================================
# Layer 3+4: Primary enforcement/application (most-used)
# =============================================================================
# FDM ghost cell method (structured grids) — the most-used applicator
from .applicator_fdm import (
    FDMApplicator,
    pad_array_with_ghosts,
)

# Unified BC class and factory functions
from .conditions import (
    BoundaryConditions,
    dirichlet_bc,
    mixed_bc,
    mixed_bc_from_regions,
    neumann_bc,
    no_flux_bc,
    periodic_bc,
    robin_bc,
    uniform_bc,
)

# Variational inequality constraints (Issue #591)
from .constraint_protocol import ConstraintProtocol
from .constraints import BilateralConstraint, ObstacleConstraint

# Unified dispatch entry point (Issue #527)
from .dispatch import (
    apply_bc,
    get_applicator_for_geometry,
    validate_bc_compatibility,
)

# Dynamic BC value providers (Issue #625)
from .providers import (
    AdjointConsistentProvider,
    BaseBCValueProvider,
    BCValueProvider,
    ConstantProvider,
    NormalDriftProvider,
    is_provider,
    resolve_provider,
)

# Layer 2: Resolution (Issue #848 Phase 3)
from .resolution import (
    BCResolver,
    FPResolver,
    HJBResolver,
    MathBCType,
    ResolvedBC,
    resolve_bc,
)
from .types import (
    BCSegment,
    BCType,
    BoundaryEntity,
    BoundaryFace,
    PeriodicGridConvention,
    create_standard_boundary_names,
    parse_boundary_face,
)

# =============================================================================
# __all__: Primary public API
#
# For secondary/specialist APIs (applicators, calculators, ghost formulas,
# graph BC, corner handling, etc.), import from submodules:
#   from mfgarchon.geometry.boundary.bc_adapter import apply_fem_bc  # FEM
#   from mfgarchon.geometry.boundary.applicator_graph import GraphApplicator
#   from mfgarchon.geometry.boundary.protocols import BoundaryCapable
#   from mfgarchon.geometry.boundary.calculators import DirichletCalculator
#   from mfgarchon.geometry.boundary.enforcement import enforce_neumann_value_nd
# =============================================================================

__all__ = [
    # Layer 1: Specification
    "BCType",
    "PeriodicGridConvention",
    "BCSegment",
    "BoundaryConditions",
    "BoundaryEntity",
    "BoundaryFace",
    "parse_boundary_face",
    "create_standard_boundary_names",
    # Factory functions
    "uniform_bc",
    "periodic_bc",
    "dirichlet_bc",
    "neumann_bc",
    "no_flux_bc",
    "robin_bc",
    "mixed_bc",
    "mixed_bc_from_regions",
    # Dynamic BC providers
    "BCValueProvider",
    "BaseBCValueProvider",
    "AdjointConsistentProvider",
    "NormalDriftProvider",
    "ConstantProvider",
    "is_provider",
    "resolve_provider",
    # Layer 2: Resolution
    "MathBCType",
    "ResolvedBC",
    "BCResolver",
    "HJBResolver",
    "FPResolver",
    "resolve_bc",
    # Constraints
    "ConstraintProtocol",
    "ObstacleConstraint",
    "BilateralConstraint",
    # Layer 3+4: Primary applicator and dispatch
    "FDMApplicator",
    "pad_array_with_ghosts",
    "apply_bc",
    "get_applicator_for_geometry",
    "validate_bc_compatibility",
]


# =============================================================================
# Backward compatibility: lazy imports for symbols that were in the old __all__
# =============================================================================
# These are no longer in __all__ but remain importable via __getattr__
# for lazy loading (avoids importing everything eagerly).


def __getattr__(name: str):
    """Lazy import for backward-compatible symbols not in __all__."""
    # Applicators
    _applicator_map = {
        # Graph
        "GraphApplicator": ("applicator_graph", "GraphApplicator"),
        "GraphBCConfig": ("applicator_graph", "GraphBCConfig"),
        "GraphBCType": ("applicator_graph", "GraphBCType"),
        "NodeBC": ("applicator_graph", "NodeBC"),
        "EdgeBC": ("applicator_graph", "EdgeBC"),
        "create_graph_applicator": ("applicator_graph", "create_graph_applicator"),
        "create_maze_applicator": ("applicator_graph", "create_maze_applicator"),
        # Implicit
        "ImplicitApplicator": ("applicator_implicit", "ImplicitApplicator"),
        "create_implicit_applicator": ("applicator_implicit", "create_implicit_applicator"),
        # Interpolation
        "InterpolationApplicator": ("applicator_interpolation", "InterpolationApplicator"),
        "create_interpolation_applicator": ("applicator_interpolation", "create_interpolation_applicator"),
        # Meshfree
        "MeshfreeApplicator": ("applicator_meshfree", "MeshfreeApplicator"),
        "ParticleReflector": ("applicator_meshfree", "ParticleReflector"),
        "SDFParticleBCHandler": ("applicator_meshfree", "SDFParticleBCHandler"),
        # Particle
        "ParticleApplicator": ("applicator_particle", "ParticleApplicator"),
        # FDM internals (commonly imported)
        "GhostBuffer": ("applicator_fdm", "GhostBuffer"),
        "GhostCellConfig": ("applicator_fdm", "GhostCellConfig"),
        "PreallocatedGhostBuffer": ("applicator_fdm", "PreallocatedGhostBuffer"),
        "bc_to_topology_calculator": ("applicator_fdm", "bc_to_topology_calculator"),
        "create_ghost_buffer_from_bc": ("applicator_fdm", "create_ghost_buffer_from_bc"),
        # Protocols and base classes
        "BaseBCApplicator": ("protocols", "BaseBCApplicator"),
        "BaseStructuredApplicator": ("protocols", "BaseStructuredApplicator"),
        "BaseUnstructuredApplicator": ("protocols", "BaseUnstructuredApplicator"),
        "BaseMeshfreeApplicator": ("protocols", "BaseMeshfreeApplicator"),
        "BaseGraphApplicator": ("protocols", "BaseGraphApplicator"),
        "BCApplicatorProtocol": ("protocols", "BCApplicatorProtocol"),
        "BoundaryCapable": ("protocols", "BoundaryCapable"),
        "DiscretizationType": ("protocols", "DiscretizationType"),
        "GridType": ("protocols", "GridType"),
        "Topology": ("protocols", "Topology"),
        "BoundaryCalculator": ("protocols", "BoundaryCalculator"),
        "BoundaryHandler": ("protocols", "BoundaryHandler"),
        "AdvancedBoundaryHandler": ("protocols", "AdvancedBoundaryHandler"),
        "validate_boundary_handler": ("protocols", "validate_boundary_handler"),
        # Calculators
        "DirichletCalculator": ("calculators", "DirichletCalculator"),
        "NeumannCalculator": ("calculators", "NeumannCalculator"),
        "RobinCalculator": ("calculators", "RobinCalculator"),
        "ZeroGradientCalculator": ("calculators", "ZeroGradientCalculator"),
        "ZeroFluxCalculator": ("calculators", "ZeroFluxCalculator"),
        "LinearExtrapolationCalculator": ("calculators", "LinearExtrapolationCalculator"),
        "QuadraticExtrapolationCalculator": ("calculators", "QuadraticExtrapolationCalculator"),
        "NoFluxCalculator": ("calculators", "NoFluxCalculator"),
        "FPNoFluxCalculator": ("calculators", "FPNoFluxCalculator"),
        "PeriodicTopology": ("calculators", "PeriodicTopology"),
        "BoundedTopology": ("calculators", "BoundedTopology"),
        # Ghost cell formulas
        "ghost_cell_dirichlet": ("ghost_cells", "ghost_cell_dirichlet"),
        "ghost_cell_neumann": ("ghost_cells", "ghost_cell_neumann"),
        "ghost_cell_robin": ("ghost_cells", "ghost_cell_robin"),
        "high_order_ghost_dirichlet": ("ghost_cells", "high_order_ghost_dirichlet"),
        "high_order_ghost_neumann": ("ghost_cells", "high_order_ghost_neumann"),
        "ghost_cell_fp_no_flux": ("ghost_cells", "ghost_cell_fp_no_flux"),
        "ghost_cell_advection_diffusion_no_flux": ("ghost_cells", "ghost_cell_advection_diffusion_no_flux"),
        "ghost_cell_linear_extrapolation": ("ghost_cells", "ghost_cell_linear_extrapolation"),
        "ghost_cell_quadratic_extrapolation": ("ghost_cells", "ghost_cell_quadratic_extrapolation"),
        # Enforcement utilities
        "enforce_dirichlet_value_nd": ("enforcement", "enforce_dirichlet_value_nd"),
        "enforce_neumann_value_nd": ("enforcement", "enforce_neumann_value_nd"),
        "enforce_periodic_value_nd": ("enforcement", "enforce_periodic_value_nd"),
        "enforce_robin_value_nd": ("enforcement", "enforce_robin_value_nd"),
        # BC utilities
        "bc_type_to_geometric_operation": ("bc_utils", "bc_type_to_geometric_operation"),
        "get_bc_type_string": ("bc_utils", "get_bc_type_string"),
        # BC coupling (deprecated)
        "compute_adjoint_consistent_bc_values": ("bc_coupling", "compute_adjoint_consistent_bc_values"),
        "create_adjoint_consistent_bc_1d": ("bc_coupling", "create_adjoint_consistent_bc_1d"),
        "compute_boundary_log_density_gradient_1d": ("bc_coupling", "compute_boundary_log_density_gradient_1d"),
        # Corner handling
        "reflect_positions": ("corner", "reflect_positions"),
        "absorb_positions": ("corner", "absorb_positions"),
        "reflect_velocity": ("corner", "reflect_velocity"),
        "reflect_velocity_with_normal": ("corner", "reflect_velocity_with_normal"),
        "CornerStrategy": ("corner", "CornerStrategy"),
        "CornerStrategyLiteral": ("corner", "CornerStrategyLiteral"),
        "DEFAULT_CORNER_STRATEGY": ("corner", "DEFAULT_CORNER_STRATEGY"),
        "validate_corner_strategy": ("corner", "validate_corner_strategy"),
        "wrap_positions": ("corner", "wrap_positions"),
        # Ghost point utilities
        "compute_normal_from_bounds": ("ghost", "compute_normal_from_bounds"),
        "compute_normal_from_sdf": ("ghost", "compute_normal_from_sdf"),
        "create_ghost_points_for_kde": ("ghost", "create_ghost_points_for_kde"),
        "create_ghost_stencil": ("ghost", "create_ghost_stencil"),
        "create_reflection_ghost_points": ("ghost", "create_reflection_ghost_points"),
        "reflect_point_across_plane": ("ghost", "reflect_point_across_plane"),
        # Periodic utilities
        "create_periodic_ghost_points": ("periodic", "create_periodic_ghost_points"),
        "periodic_wrap_positions": ("periodic", "wrap_positions"),
        # Legacy/deprecated
        "MixedBoundaryConditions": ("conditions", "MixedBoundaryConditions"),
    }

    if name in _applicator_map:
        module_name, attr_name = _applicator_map[name]
        import importlib

        mod = importlib.import_module(f".{module_name}", __name__)
        return getattr(mod, attr_name)

    # Legacy aliases
    if name == "BoundaryConditions1DFDM":
        from .fdm_bc_1d import BoundaryConditions

        return BoundaryConditions
    if name == "LegacyBoundaryConditions1D":
        from .fdm_bc_1d import BoundaryConditions

        return BoundaryConditions
    if name == "compute_boundary_log_density_gradient":
        from .bc_coupling import compute_boundary_log_density_gradient_1d

        return compute_boundary_log_density_gradient_1d
    if name == "compute_coupled_hjb_bc_values":
        from .bc_coupling import compute_coupled_hjb_bc_values

        return compute_coupled_hjb_bc_values
    if name == "get_ghost_values_nd":
        from ._compat import get_ghost_values_nd

        return get_ghost_values_nd

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from __future__ import annotations

import importlib.util
import inspect
import warnings
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from scipy.linalg import lstsq

# BC types for BoundaryCapable protocol implementation (Issue #527)
from scipy.optimize import approx_fprime

from mfgarchon.alg.numerical.gfdm_components import (
    BoundaryHandler,
    GridCollocationMapper,
    MonotonicityEnforcer,
    NeighborhoodBuilder,
    PrecomputedMonotoneStencils,
)

# GFDM infrastructure (Strategy Pattern)
from mfgarchon.alg.numerical.gfdm_components.gfdm_strategies import (
    DirectCollocationHandler,
    TaylorOperator,
    create_operator,
)
from mfgarchon.alg.numerical.hjb_solvers.h_eval import (
    assemble_hjb_jacobian_diag,
    assemble_hjb_residual,
    eval_dH_dp_batch,
    eval_H_batch,
)
from mfgarchon.geometry.boundary.applicator_base import DiscretizationType
from mfgarchon.geometry.boundary.tolerances import BOUNDARY_TOL
from mfgarchon.geometry.boundary.types import BCSegment, BCType, BoundaryFace
from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.numerical.qp_utils import QPCache, QPSolver
from mfgarchon.utils.pde_coefficients import diffusion_from_volatility, resolve_diffusion_source

from .base_hjb import (
    DEFAULT_NEWTON_MAX_ITERATIONS,
    DEFAULT_NEWTON_TOLERANCE,
    BaseHJBSolver,
)

logger = get_logger(__name__)

# Optional QP solver imports
CVXPY_AVAILABLE = importlib.util.find_spec("cvxpy") is not None
OSQP_AVAILABLE = importlib.util.find_spec("osqp") is not None

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfgarchon.config.mfg_methods import GFDMConfig
    from mfgarchon.core.derivatives import DerivativeTensors
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry import BoundaryConditions


class HJBGFDMSolver(BaseHJBSolver):
    """
    Generalized Finite Difference Method (GFDM) solver for HJB equations using collocation.

    This solver implements meshfree collocation for HJB equations using:
    1. δ-neighborhood search for local support
    2. Taylor expansion with weighted least squares for derivative approximation
    3. Newton iteration for nonlinear HJB equations
    4. Support for various boundary conditions
    5. Optional QP constraints for monotonicity preservation

    QP Optimization Levels:
    - "none": GFDM without QP constraints (fastest, no monotonicity guarantee)
    - "auto": Adaptive QP with M-matrix checking (runtime QP when needed)
    - "always": Force QP at every point (slowest, for debugging)
    - "precompute": Precomputed monotone stencils (fast + monotone, recommended)

    Note: Monotonicity and QP constraint functionality is provided by MonotonicityEnforcer component.

    Implements BoundaryCapable protocol for unified BC handling (Issue #527).

    Collocation Point Strategies (Issue #529):
        Use FIXED collocation points throughout the MFG solve. Moving points
        during iteration causes convergence stall due to interpolation noise
        and stencil weight fluctuations.

        IMPORTANT: Fully Lagrangian MFG (moving collocation with the flow)
        is MATHEMATICALLY INVALID because the optimal control alpha* = -grad(u)
        requires grad(u) at FIXED spatial locations.

        See adaptive collocation analysis for detailed discussion
        of three collocation strategies and why only fixed collocation is valid.
    """

    # Scheme family trait for duality validation (Issue #580)
    from mfgarchon.alg.base_solver import SchemeFamily

    _scheme_family = SchemeFamily.GFDM

    # BoundaryCapable protocol: Supported BC types
    _SUPPORTED_BC_TYPES: frozenset = frozenset(
        {
            BCType.DIRICHLET,
            BCType.NEUMANN,
            BCType.NO_FLUX,  # Same as Neumann with g=0
            BCType.ROBIN,  # adjoint-consistent Robin(0,1); general-Robin sub-cases fail loud in the row builder
            # PERIODIC is honoured, but only on a cloud whose points are not detected as
            # boundary -- seam 2.2e-15 / 3.3e-11 / 6.7e-16 / 6.7e-16 at Nx=11/21/41/81 on a torus
            # cloud, where the Issue #711 wrap does the work. `_detect_boundary_indices` ignores
            # `periodic_dims`, so the default endpoint-inclusive cloud reaches the row builder
            # instead and raises. That is the defect (#1841); the capability is real, so the
            # declaration stays.
            BCType.PERIODIC,
        }
    )

    @property
    def discretization_type(self) -> DiscretizationType:
        """Discretization method (BoundaryCapable protocol)."""
        return DiscretizationType.GFDM

    # Explicitly initialize _neighborhood_builder to None (avoids hasattr)
    _neighborhood_builder: NeighborhoodBuilder | None = None

    @property
    def neighborhoods(self) -> dict:
        """Get neighborhoods from NeighborhoodBuilder or legacy mixin."""
        if self._neighborhood_builder is not None:
            return self._neighborhood_builder.neighborhoods
        # Legacy fallback: direct attribute access
        try:
            return self._neighborhoods
        except AttributeError:
            return {}

    @neighborhoods.setter
    def neighborhoods(self, value: dict) -> None:
        """Set neighborhoods in NeighborhoodBuilder or legacy storage."""
        if self._neighborhood_builder is not None:
            self._neighborhood_builder.neighborhoods = value
        else:
            self._neighborhoods = value

    @property
    def taylor_matrices(self) -> dict:
        """Get Taylor matrices from NeighborhoodBuilder or legacy mixin."""
        if self._neighborhood_builder is not None:
            return self._neighborhood_builder.taylor_matrices
        # Legacy fallback: direct attribute access
        try:
            return self._taylor_matrices
        except AttributeError:
            return {}

    @taylor_matrices.setter
    def taylor_matrices(self, value: dict) -> None:
        """Set Taylor matrices in NeighborhoodBuilder or legacy storage."""
        if self._neighborhood_builder is not None:
            self._neighborhood_builder.taylor_matrices = value
        else:
            self._taylor_matrices = value

    @property
    def adaptive_stats(self) -> dict:
        """Get adaptive neighborhood statistics from NeighborhoodBuilder or legacy mixin."""
        if self._neighborhood_builder is not None:
            return self._neighborhood_builder.adaptive_stats
        # Legacy fallback: direct attribute access
        try:
            return self._adaptive_stats
        except AttributeError:
            return {"n_adapted": 0, "adaptive_enlargements": [], "max_delta_used": 0.0}

    @adaptive_stats.setter
    def adaptive_stats(self, value: dict) -> None:
        """Set adaptive stats in NeighborhoodBuilder or legacy storage."""
        if self._neighborhood_builder is not None:
            self._neighborhood_builder.adaptive_stats = value
        else:
            self._adaptive_stats = value

    @classmethod
    def from_config(
        cls,
        problem: MFGProblem,
        collocation_points: np.ndarray,
        config: GFDMConfig,
        **extra: Any,
    ) -> HJBGFDMSolver:
        """Create solver from GFDMConfig object (Issue #634).

        Converts structured config into constructor kwargs. Additional keyword
        arguments in ``extra`` override config values.

        Args:
            problem: MFG problem instance
            collocation_points: (N_points, d) array of collocation points
            config: Structured GFDM configuration
            **extra: Additional kwargs passed to __init__ (override config)

        Returns:
            Configured HJBGFDMSolver instance
        """
        # Translate QPConfig.optimization_level to the v0.18.0 canonical axes
        # (qp_optimization_level was removed from __init__ in v0.25.0, Issue #1070).
        _qp_level_to_scheme: dict[str, tuple[str, str | None]] = {
            "none": ("none", None),
            "auto": ("qp_m_matrix", "adaptive"),
            "always": ("qp_m_matrix", "always"),
        }
        _mono_scheme, _mono_app = _qp_level_to_scheme.get(config.qp.optimization_level, ("none", None))
        kwargs: dict[str, Any] = {
            "delta": config.delta,
            "taylor_order": config.taylor_order,
            "weight_function": config.weight_function,
            "weight_scale": config.weight_scale,
            "monotonicity_scheme": _mono_scheme,
            "monotonicity_application": _mono_app,
            "qp_solver": config.qp.solver,
            "qp_warm_start": config.qp.warm_start,
            "qp_constraint_mode": config.qp.constraint_mode,
            "neighborhood_mode": config.neighborhood.mode,
            "k_neighbors": config.neighborhood.k_neighbors,
            "adaptive_neighborhoods": config.neighborhood.adaptive,
            "k_min": config.neighborhood.k_min,
            "max_delta_multiplier": config.neighborhood.max_delta_multiplier,
            "derivative_method": config.derivative.method,
            "rbf_kernel": config.derivative.rbf_kernel,
            "rbf_poly_degree": config.derivative.rbf_poly_degree,
            "use_local_coordinate_rotation": config.boundary_accuracy.local_coordinate_rotation,
            "use_ghost_nodes": config.boundary_accuracy.ghost_nodes,
            "use_wind_dependent_bc": config.boundary_accuracy.wind_dependent_bc,
            "congestion_mode": config.congestion_mode,
        }
        kwargs.update(extra)
        return cls(problem, collocation_points, **kwargs)

    def __init__(
        self,
        problem: MFGProblem,
        collocation_points: np.ndarray,
        delta: float = 0.1,
        taylor_order: int = 2,
        weight_function: str = "wendland",
        weight_scale: float = 1.0,
        max_newton_iterations: int | None = None,
        newton_tolerance: float | None = None,
        # Inner HJB solver selector (Issue #1118). 'newton' (default): the existing
        # per-timestep Newton + Armijo line search. 'howard': delegate the backward
        # sweep to HJBHowardSolver (policy iteration; no line search, so it avoids the
        # MIN_ALPHA stall that freezes Newton on advection-dominant / no-flux-BC regimes).
        # 'howard' needs unit control cost and a homogeneous no-flux BC (validated in solve).
        # It no longer requires monotonicity_scheme='joint_socp' (#2066): without SOCP stencils
        # it takes its operators from the collocation operator and warns that monotonicity --
        # a convergence hypothesis, not a runnability one -- is not guaranteed.
        inner_solver: str = "newton",
        boundary_indices: np.ndarray | None = None,
        boundary_conditions: dict | BoundaryConditions | None = None,
        # Monotonicity construction (renamed from qp_optimization_level v0.18.0; Issue #XXXX).
        # Two orthogonal axes:
        #   - monotonicity_scheme: WHICH constraint is enforced
        #   - monotonicity_application: WHEN to enforce it
        # See docstring for full semantics.
        monotonicity_scheme: str | None = None,
        monotonicity_application: str | None = None,
        qp_usage_target: float = 0.1,  # Unused, kept for backward compatibility
        qp_solver: str = "osqp",  # "osqp" or "scipy"
        qp_warm_start: bool = True,  # Enable QP warm-starting
        qp_constraint_mode: str = "indirect",  # "indirect" or "hamiltonian"
        # Adaptive neighborhood parameters
        adaptive_neighborhoods: bool = False,
        k_min: int | None = None,
        max_delta_multiplier: float = 5.0,
        # Hybrid neighborhood parameters
        k_neighbors: int | None = None,
        neighborhood_mode: str = "hybrid",
        # New GFDM infrastructure parameters
        derivative_method: str = "taylor",  # "taylor"; "rbf" unsupported since #1526 (raises, #1553)
        rbf_kernel: str = "phs3",  # For RBF-FD: "phs3", "phs5", "gaussian"
        rbf_poly_degree: int = 2,  # Polynomial augmentation degree for RBF-FD
        use_new_infrastructure: bool = True,  # Use new Strategy Pattern (recommended)
        # Local Coordinate Rotation for boundary accuracy (Issue #531)
        use_local_coordinate_rotation: bool = False,
        # Ghost Nodes for Neumann BC enforcement (Issue #531 - Terminal BC compatibility)
        use_ghost_nodes: bool = False,
        # Wind-Dependent BC for viscosity solution compatibility
        use_wind_dependent_bc: bool = False,
        # Congestion mode for Hamiltonian coupling
        congestion_mode: str = "additive",
        # Collocation geometry for periodic domains (Issue #711)
        collocation_geometry: object | None = None,
        # Obstacle-aware visibility filtering for stencil neighbors
        obstacle_sdf: object | None = None,
        visibility_samples: int = 10,
        visibility_margin: float = 0.0,
        # DMP runtime guard (Issue #1074): warn when the solved drift exceeds the
        # assembled-M-matrix threshold for joint_socp. Off by default (zero overhead,
        # numerically byte-identical) — the assembled discrete-maximum-principle is only
        # diagnostic, not enforced.
        check_dmp: bool = False,
        # LLF (Local Lax-Friedrichs) augmented diffusion (Issue #1059, paper P2 branch).
        # Adds per-node artificial viscosity nu_i to the diffusion term so that the
        # local Peclet condition Pe_h(i) = l_H(i)*h_i / (sigma^2 + 2*nu_i) <= 1/(2C)
        # is satisfied at every node, restoring the discrete comparison principle at
        # high-Pe nodes where P1 (unaugmented) cannot satisfy the theorem hypothesis.
        # Default OFF (byte-identical to current main when off).
        # When on, requires llf_l_H: per-node |dH/dp| Lipschitz bound (fail-loud if None).
        llf_augmentation: bool = False,
        llf_cone_constant: float = 0.5,  # C in nu_i = max(0, C*l_H*h - sigma^2/2)
        llf_l_H: float | np.ndarray | None = None,  # per-node Lipschitz bound |dH/dp|
        # SOCP-infeasibility-triggered adaptive stencil enlargement (Issue #1106),
        # joint_socp scheme only. When > 0, a stencil that is still infeasible
        # after C-bisection is rebuilt with additional next-nearest neighbors and
        # the SOCP retried, up to this many enlargement steps. Default 0 (OFF):
        # the paper / default path is byte-identical. The enlarged neighbor set is
        # contained inside the precomputed SOCP stencils and does NOT mutate the
        # shared runtime neighborhoods (no cascade into operator / FP / BC).
        socp_max_stencil_enlargements: int = 0,
        socp_enlargement_step: int = 2,
    ):
        """
        Initialize the GFDM HJB solver.

        Args:
            problem: MFG problem instance
            collocation_points: (N_points, d) array of collocation points
            delta: Neighborhood radius for collocation
            taylor_order: Order of Taylor expansion (1 or 2)
            weight_function: Weight function type ("wendland", "cubic_spline", "gaussian", "inverse_distance", "uniform")
            weight_scale: Scale parameter for weight function
            max_newton_iterations: Maximum Newton iterations (new parameter name)
            newton_tolerance: Newton convergence tolerance.
            boundary_indices: Indices of boundary collocation points
            boundary_conditions: Dictionary or BoundaryConditions object specifying boundary conditions
            monotonicity_scheme: Which monotonicity construction to enforce on the GFDM
                Laplacian (and, for joint_socp, the per-edge cone on the gradient stencil):
                - "none": no constraints (fastest; no monotonicity guarantee).
                - "qp_m_matrix": classical M-matrix QP — projects unconstrained
                  Wendland-Taylor Laplacian weights onto $L_{ij} \\geq 0$ for $j \\neq i$.
                - "joint_socp": (Phase 1B follow-up) joint SOCP — M-matrix on $-\\Delta_h$
                  + per-edge cone $\\|D_{ij}\\|_2 \\leq C h_i L_{ij}$, closing the discrete
                  comparison principle (audit-major contribution).
                  Scope (Issue #1074): joint_socp enforces the M-matrix property PER
                  STENCIL (each row's off-diagonals satisfy the sign/dominance
                  constraints). This does NOT in general guarantee that the ASSEMBLED
                  HJB iteration matrix $I/dt - D L + \\alpha D_{grad}$ is an M-matrix:
                  the signed drift term can flip an off-diagonal positive at high
                  Peclet, so the discrete maximum principle is not a-priori guaranteed
                  by per-stencil feasibility. Use ``check_dmp=True`` (runtime warning)
                  or ``monotonicity_enforcer.verify_assembled_m_matrix(...)`` to check
                  the assembled M-matrix / DMP property for a specific problem and
                  discretization.
                Default: "none". (Renamed from qp_optimization_level in v0.18.0.)
            monotonicity_application: When the chosen scheme is enforced (only
                meaningful for non-"none" schemes):
                - "adaptive": only at nodes where the unconstrained weights violate
                  the constraint (recommended for qp_m_matrix).
                - "always": at every node, every solve.
                - "precompute": cache feasible weights at construction; reuse for all
                  Picard iterations / time steps. Recommended for joint_socp.
                Default (None): use scheme-recommended default — "adaptive" for
                qp_m_matrix, "precompute" for joint_socp.
            qp_usage_target: Unused parameter, kept for backward compatibility
            qp_solver: QP solver backend (default "osqp"):
                - "osqp": Use OSQP solver (fast convex QP, 5-10× faster than scipy)
                - "scipy": Use scipy.optimize.minimize (SLSQP or L-BFGS-B)
            qp_warm_start: Enable warm-starting for QP solves (default True).
                When True, uses previous QP solution as initial guess for next solve.
                Provides 2-3× additional speedup for OSQP on similar QP problems.
                Only applies to OSQP solver (scipy does not support efficient warm-starting).
            qp_constraint_mode: Type of monotonicity constraints (default "indirect"):
                - "indirect": Constraints on Taylor coefficients (simpler, approximate)
                - "hamiltonian": Direct Hamiltonian gradient constraints dH/du_j >= 0
                  (stricter, better monotonicity guarantees, requires gamma parameter)
            adaptive_neighborhoods: Enable adaptive delta enlargement to guarantee well-posed problems.
                When enabled, points with insufficient neighbors get locally enlarged delta.
                Maintains theoretical soundness while ensuring practical robustness.
                Recommended for irregular particle distributions.
            k_min: Minimum number of neighbors required per point (auto-computed from taylor_order if None).
                For Taylor order p in d dimensions, need C(d+p, p) - 1 derivatives.
            max_delta_multiplier: Maximum allowed delta enlargement factor (default 5.0, conservative).
                Limits delta growth to preserve GFDM locality. For very irregular distributions,
                consider increasing to 10.0 (achieves 98%+ success) or increasing base delta instead.
                Trade-off: Smaller limit = better theory, larger limit = better robustness.
            k_neighbors: Number of neighbors for neighborhood selection (auto-computed if None).
                When None, computed from Taylor order to ensure well-posed least squares.
            neighborhood_mode: Neighborhood selection strategy:
                - "radius": Use all points within delta (classic behavior)
                - "knn": Use exactly k nearest neighbors
                - "hybrid": Use delta, but ensure at least k neighbors (default, most robust)
            derivative_method: Method for computing spatial derivatives:
                - "taylor": Standard GFDM with Taylor polynomial basis (default)
                - "rbf": UNSUPPORTED since #1526 (raises NotImplementedError, Issue #1553) --
                  the RBF branch has no working differentiation-weight builder. Use "taylor".
            rbf_kernel: Kernel for RBF-FD method (reserved for the future "rbf" build-out, #1553):
                - "phs3": r³ polyharmonic spline (most common)
                - "phs5": r⁵ polyharmonic spline (higher accuracy)
                - "gaussian": Gaussian RBF (requires shape parameter tuning)
            rbf_poly_degree: Polynomial augmentation degree for RBF-FD (default 2)
            use_new_infrastructure: Use new Strategy Pattern infrastructure (default True).
                When True, uses TaylorOperator/LocalRBFOperator + DirectCollocationHandler.
                When False, raises ValueError (legacy GFDMOperator removed in v0.17.15).
            use_local_coordinate_rotation: Enable Local Coordinate Rotation (LCR) for
                boundary stencils (default False, Issue #531). When True, rotates
                neighbor offsets at boundary points to align with the boundary normal,
                improving numerical conditioning for normal derivative computation.
                Recommended for domains with complex boundaries or when boundary
                stencils show poor conditioning. Only affects boundary points.
            use_ghost_nodes: Enable Ghost Nodes method for Neumann boundary conditions
                (default False, Issue #531 - Terminal BC compatibility). When True,
                creates mirrored "ghost" neighbors outside the domain for boundary points,
                enforcing ∂u/∂n = 0 structurally through symmetric stencils rather than
                via row replacement. This eliminates terminal cost/BC incompatibility issues
                in MFG problems. Recommended when terminal cost violates Neumann BC
                (e.g., g(x) = ||x - x_exit||² with Neumann BC at walls). Mutually exclusive
                with use_local_coordinate_rotation (ghost nodes take precedence).
            use_wind_dependent_bc: Enable wind-dependent boundary conditions (default False).
                When True (requires use_ghost_nodes=True), ghost nodes are only enforced
                when characteristics flow INTO the boundary (∇u·n > 0). When flow is OUT
                (∇u·n < 0), uses extrapolation instead. This implements the viscosity solution
                approach where BCs are weak constraints, only enforced when the PDE solution
                "wants" to violate them. Recommended for evacuation/exit problems where agents
                need to cross boundaries. Based on Lions & Souganidis theory of discontinuous
                viscosity solutions.
            congestion_mode: Mode for density-velocity coupling (default "additive"):
                - "additive": H = |p|²/(2λ) + γm (standard separable form)
                - "multiplicative": H = (1 + γ|Ω|m)|p|²/(2λ) (velocity reduction by congestion)
                The multiplicative form models agents slowing down in crowded areas, where
                γ|Ω|m ≈ γ × (local_density / average_density). This makes γ dimensionless
                and O(1) for observable effects, unlike additive form where γ ~ 1/|Ω|.
            collocation_geometry: Geometry object for collocation domain (Issue #711).
                If provided and implements SupportsPeriodic (e.g., Hyperrectangle with
                periodic_dims), enables periodic neighbor search for GFDM on torus domains.
                Example: Hyperrectangle(bounds, periodic_dims=(0, 1)) for 2D torus.
            obstacle_sdf: Optional callable ``f(x) -> float`` for visibility-based
                stencil filtering. Convention: ``obstacle_sdf(x) < 0`` means x is INSIDE
                the obstacle. Pass the obstacle's own ``.signed_distance`` directly
                (e.g., ``obstacle_sdf=Hypersphere(...).signed_distance``); do NOT pass
                a ``DifferenceDomain.signed_distance``, which has the opposite convention
                (sd<0 inside the navigable region). See Issue #1038 and the full
                docstring on ``NeighborhoodBuilder.obstacle_sdf``.
            visibility_samples: Number of interior samples along each stencil edge for
                obstacle intersection testing (default 10). Used only when
                ``obstacle_sdf`` is provided.
            visibility_margin: Safety margin for obstacle proximity (default 0.0).
                Stencil edges passing within this distance of an obstacle are filtered.
            llf_augmentation: Enable Local Lax-Friedrichs (LLF) per-node artificial
                diffusion (Issue #1059, paper P2 branch of thm:discrete_comparison).
                When True, augments the diffusion coefficient at each node i by
                ``nu_i = max(0, C * l_H(i) * h_i - sigma^2/2)``, so the effective
                volatility is ``sigma_eff_i = sqrt(sigma^2 + 2*nu_i)``.  This restores
                the discrete comparison principle at high-Pe nodes (``l_H(i)*h_i/sigma^2
                >> 1``) where P1 (unaugmented) cannot satisfy the theorem hypothesis.
                Default OFF (byte-identical to current main when off).
                Requires ``llf_l_H`` to be provided (fail-loud otherwise).
            llf_cone_constant: The constant C in ``nu_i = max(0, C*l_H(i)*h_i -
                sigma^2/2)`` (default 0.5). Per the mfg-research prototype: C=0.5 gives
                structural stability; C=1.0 is catastrophic (vicious Picard feedback).
                See Issue #1059.
            llf_l_H: Per-node Lipschitz bound ``l_H(i) = |dH/dp|(i)``.  May be a scalar
                (broadcast to all nodes) or a shape ``(n_points,)`` array.  Design
                options: (a) cone bound from stencil, (b) from ``dH/dp`` at previous
                Picard iterate, (c) user-supplied (this parameter).  Fail-loud if
                ``llf_augmentation=True`` and this is None.
            socp_max_stencil_enlargements: SOCP-infeasibility-triggered adaptive
                stencil enlargement budget (Issue #1106), ``joint_socp`` scheme
                only. When > 0, a stencil that is still infeasible after the
                C-bisection is rebuilt with ``socp_enlargement_step`` additional
                next-nearest neighbors and the SOCP retried, up to this many
                steps; the extra Taylor degrees of freedom can flip a
                geometrically starved (wall / corner / obstacle-adjacent) stencil
                from infeasible to feasible. Default 0 (OFF): the paper / default
                path is byte-identical. The enlarged neighbor set is contained in
                the precomputed SOCP stencils and does not mutate the shared
                runtime neighborhoods.
            socp_enlargement_step: Number of next-nearest neighbors added per
                enlargement step (Issue #1106). Default 2.
        """
        super().__init__(problem)

        # Issue #1079: GFDM only supports scalar sigma. A full (d,d) tensor stored in
        # problem.volatility_field would silently collapse to a scalar mean in
        # MFGProblem.sigma, then be used as if it were a correct isotropic coefficient.
        # The GFDM Laplacian stencil target (e_lap in joint_socp.py) has zero weight on
        # the cross-derivative column, so D_ij d^2u/dx_i dx_j (i!=j) terms are never
        # discretized. Fail loud at construction rather than returning a silently wrong
        # solution (fail-fast per CLAUDE.md).
        _vf = getattr(self.problem, "volatility_field", None)
        _spatial_shape = tuple(getattr(self.problem, "spatial_shape", ()) or ())
        _problem_dimension = getattr(self.problem, "dimension", None)
        _tensor_shape = (_problem_dimension, _problem_dimension) if isinstance(_problem_dimension, int) else None
        if isinstance(_vf, np.ndarray) and _tensor_shape is not None and _vf.shape == _tensor_shape:
            if _vf.shape == _spatial_shape:
                raise NotImplementedError(
                    "HJBGFDMSolver cannot infer whether problem.volatility_field with shape "
                    f"{_vf.shape} is a scalar-valued grid field or a full-tensor (d,d) sigma; "
                    "the two representations are ambiguous. Keep a scalar sigma on the problem "
                    "and pass the grid field explicitly as "
                    "solve_hjb_system(volatility_field=field) instead."
                )
            raise NotImplementedError(
                "HJBGFDMSolver does not support full-tensor (d,d) sigma. "
                "The GFDM Laplacian target (e_lap in joint_socp.py) has zero weight on "
                "the cross-derivative column, so off-diagonal D_ij d^2u/dx_i dx_j "
                "terms are silently dropped (Issue #1079). Pass scalar sigma or a "
                "scalar-valued spatial field matching problem.spatial_shape instead."
            )

        # --- Resolve (scheme, application) from new API or legacy alias (v0.18.0) ---
        #
        # Two orthogonal axes:
        #   monotonicity_scheme:        WHICH constraint is enforced
        #     "none" | "qp_m_matrix" | "joint_socp"
        #   monotonicity_application:   WHEN it is enforced
        #     "adaptive" | "always" | "precompute"
        #
        # Application defaults per scheme (when application=None):
        #   qp_m_matrix → "adaptive"     (= legacy "auto", recommended runtime check)
        #   joint_socp  → "precompute"   (audit-major default; weights cached at construction)
        #   none        → ignored
        #
        # v0.25.0 (Issue #1070): qp_optimization_level= parameter removed; pass
        # monotonicity_scheme= and monotonicity_application= directly.

        # Issue #1034: warn when user defaults to "none" (bare Wendland-Taylor LSQ).
        # This default produces a method whose M-matrix structure is not enforced;
        # boundary stencils can produce oscillatory derivatives that destabilize
        # FP-Particle coupling on long-time-horizon problems (e.g., 1D ToB at T=8
        # with KL=0.098 and 11 spurious modes — see Issue #1034 for full evidence).
        # Validated in mfg-research/.../exp08_towel_2d_validation/_preflight_1d/
        # post_mortem_1d_tob_debug.md.
        if monotonicity_scheme is None and monotonicity_application is None:
            import warnings as _w

            _w.warn(
                "HJBGFDMSolver: no `monotonicity_scheme` specified; defaulting to "
                "'none' (no QP correction). This produces bare Wendland-Taylor LSQ "
                "stencils whose M-matrix structure is not enforced — boundary "
                "stencils can produce oscillatory derivatives that destabilize "
                "FP-Particle coupling on long-time-horizon problems. For "
                "paper-canonical monotone behavior, pass "
                "`monotonicity_scheme='joint_socp'` (M-matrix + per-edge cone, "
                "discrete comparison principle) or "
                "`monotonicity_scheme='qp_m_matrix'` (M-matrix only, cheaper). "
                "See Issue #1034. Pass `monotonicity_scheme='none'` explicitly "
                "to suppress this warning if the bare scheme is intentional.",
                UserWarning,
                stacklevel=2,
            )

        if monotonicity_scheme is not None or monotonicity_application is not None:
            scheme = monotonicity_scheme if monotonicity_scheme is not None else "none"
            valid_schemes = ("none", "qp_m_matrix", "joint_socp")
            if scheme not in valid_schemes:
                raise ValueError(f"monotonicity_scheme must be one of {valid_schemes}; got '{scheme}'.")
            valid_apps = ("adaptive", "always", "precompute", None)
            if monotonicity_application not in valid_apps:
                raise ValueError(
                    f"monotonicity_application must be one of {valid_apps}; got '{monotonicity_application}'."
                )
            # Resolve application via scheme-default if unspecified
            if monotonicity_application is None:
                application = {
                    "none": "ignored",
                    "qp_m_matrix": "adaptive",
                    "joint_socp": "precompute",
                }[scheme]
            else:
                application = monotonicity_application
        else:
            # Default: no monotonicity constraints
            scheme = "none"
            application = "ignored"

        # Canonical storage
        self.monotonicity_scheme = scheme
        self.monotonicity_application = application

        # Reconstruct legacy `qp_optimization_level` for backward-compat internal branches
        # (lines using self.qp_optimization_level == "auto"/"always"/"precompute"/"none"):
        if scheme == "none":
            self.qp_optimization_level = "none"
        elif scheme == "qp_m_matrix":
            self.qp_optimization_level = application  # adaptive→"auto"-like, etc.
            # Note: "adaptive" maps to legacy "auto" semantically, but the legacy code
            # branches check string == "auto", so we need to translate:
            if application == "adaptive":
                self.qp_optimization_level = "auto"
        elif scheme == "joint_socp":
            # joint_socp precomputes weights at __init__ (per-edge cone + M-matrix
            # via SOCP); semantically this IS a precompute application. Setting
            # legacy `qp_optimization_level = "precompute"` selects the per-point
            # HJB Newton path (line ~2425), matching the qp_m_matrix+precompute
            # path. Setting it to "none" instead would route through the batch
            # Hamiltonian path which evaluates H(x,m,p,t) differently and breaks
            # numerical equivalence with the legacy `precompute_socp_weights +
            # patch_operator` workflow used in research code.
            self.qp_optimization_level = "precompute"
            if application not in ("precompute", "ignored"):
                warnings.warn(
                    f"monotonicity_scheme='joint_socp' currently supports only "
                    f"application='precompute'; got '{application}'. Falling back to "
                    f"'precompute'. Adaptive/always strategies are tracked for a "
                    f"follow-up PR.",
                    stacklevel=2,
                )

        # Method name
        if scheme == "none":
            self.hjb_method_name = "GFDM"
        elif scheme == "qp_m_matrix":
            self.hjb_method_name = {
                "adaptive": "GFDM-QP",
                "always": "GFDM-QP-Always",
                "precompute": "GFDM-Precompute",
            }.get(application, f"GFDM-{application}")
        elif scheme == "joint_socp":
            self.hjb_method_name = f"GFDM-JointSOCP-{application}"
        else:
            self.hjb_method_name = f"GFDM-{self.qp_optimization_level}"

        # Set defaults if still None
        if max_newton_iterations is None:
            max_newton_iterations = DEFAULT_NEWTON_MAX_ITERATIONS
        if newton_tolerance is None:
            newton_tolerance = DEFAULT_NEWTON_TOLERANCE

        # Collocation parameters
        self.collocation_points = collocation_points
        self.n_points = collocation_points.shape[0]
        self.dimension = collocation_points.shape[1]
        self.delta = delta
        self.taylor_order = taylor_order
        self.weight_function = weight_function
        self.weight_scale = weight_scale

        # Newton parameters (canonical names; v0.25.0 removed NiterNewton/l2errBoundNewton)
        self.max_newton_iterations = max_newton_iterations
        self.newton_tolerance = newton_tolerance

        # Issue #1118: inner-solver selector (Newton default; Howard policy iteration opt-in)
        if inner_solver not in ("newton", "howard"):
            raise ValueError(f"inner_solver must be 'newton' or 'howard', got {inner_solver!r}")
        self.inner_solver = inner_solver

        # ONE source for "which domain is this", resolved before anything reads it (#1841).
        # Boundary detection and the operator's periodic wrap both need to know which axes are
        # periodic, and answering that question from two different objects is how a caller who
        # passes a NON-periodic collocation_geometry for a periodic problem got a seam of 1.36e+00
        # in silence: detection saw the problem's periodic axes and skipped them, the operator saw
        # a non-periodic domain and built no wrap. Before #1841 that configuration raised. Reading
        # both from the same object restores the refusal.
        #
        # The fallback: when the caller gives no separate collocation domain, the problem's own
        # geometry describes the same region. Inert for non-periodic problems, whose
        # `periodic_dimensions` is empty, so `_is_periodic` stays False exactly as before.
        self._collocation_geometry = collocation_geometry if collocation_geometry is not None else problem.geometry
        self._refuse_topology_disagreeing_with_bc()

        # Boundary condition parameters
        # Auto-detect boundary indices if not provided (Issue #542 fix)
        if boundary_indices is not None:
            self.boundary_indices = boundary_indices
        else:
            # Try to detect boundary points from domain bounds
            self.boundary_indices = self._detect_boundary_indices(collocation_points)
        # Get BC from parameter, or from problem geometry (Issue #542 fix, Issue #527 centralized BC)
        if boundary_conditions is not None:
            self.boundary_conditions = boundary_conditions
            # Explicit param is the caller's authoritative static choice; never
            # re-read from geometry at solve time (Issue #1118 BC refresh).
            self._bc_from_geometry = False
        else:
            # Use centralized BC resolution from BaseMFGSolver (Issue #527)
            # Checks: cached _boundary_conditions, geometry.boundary_conditions,
            # geometry.get_boundary_conditions(), problem.boundary_conditions,
            # problem.get_boundary_conditions()
            self.boundary_conditions = self.get_boundary_conditions()
            # Geometry-sourced BC may be re-resolved per Picard iteration by the
            # coupling layer (using_resolved_bc); refresh at solve time (Issue #1118).
            self._bc_from_geometry = True
        # Issue #1456: fail loud now if the resolved BC requests a type GFDM cannot honor at all
        # (REFLECTING / EXTRAPOLATION_*). Periodic and general-Robin sub-cases pass the type-level
        # gate and are still rejected by the row builder where they are genuinely unsupported.
        self._validate_bc_support(self.boundary_conditions)
        self.interior_indices = np.setdiff1d(np.arange(self.n_points), self.boundary_indices)

        # DMP runtime guard state (Issue #1074): lazily-computed critical drift and a
        # warn-once latch. Active only when check_dmp=True (default off → no overhead).
        self.check_dmp = check_dmp
        self._dmp_alpha_crit: float | None = None
        self._dmp_warned = False

        # Issue #1316: retain the raw per-solve override for public-state compatibility.
        # Issue #1725: normalize it once per solve into collocation space; every live
        # diffusion consumer reads `_solve_sigma`, so coefficient resolution cannot fork.
        self._volatility_field_override: float | np.ndarray | Callable | None = None
        self._solve_sigma: float | np.ndarray | None = None

        # LLF augmented diffusion (Issue #1059, paper P2 branch of thm:discrete_comparison).
        # nu_i = max(0, C * l_H(i) * h_i - sigma^2/2)
        # sigma_eff_i = sqrt(sigma^2 + 2*nu_i)
        # Stored attributes (None when llf_augmentation=False → zero overhead):
        #   self._llf_l_H      : np.ndarray (n_points,) — per-node |dH/dp| Lipschitz bound
        #   self._llf_sigma_eff: np.ndarray (n_points,) — per-node effective volatility
        self.llf_augmentation: bool = llf_augmentation
        self._llf_cone_constant: float = float(llf_cone_constant)
        if llf_augmentation:
            if llf_l_H is None:
                raise ValueError(
                    "llf_augmentation=True requires llf_l_H: the per-node |dH/dp| "
                    "Lipschitz bound (scalar float or shape (n_points,) array). "
                    "Provide llf_l_H=<value> to enable LLF augmented diffusion. "
                    "See Issue #1059."
                )
            l_H_raw = np.asarray(llf_l_H, dtype=float)
            # Scalar or shape (n_points,) — broadcast to full node array
            self._llf_l_H: np.ndarray = np.broadcast_to(l_H_raw, (self.n_points,)).copy()
            if np.any(self._llf_l_H < 0.0):
                raise ValueError(
                    f"llf_l_H must be non-negative at every node; got min={float(np.min(self._llf_l_H)):.3g} < 0."
                )
            self._llf_sigma_eff: np.ndarray = self._compute_llf_sigma_eff()
        else:
            self._llf_l_H = None  # type: ignore[assignment]
            self._llf_sigma_eff = None  # type: ignore[assignment]

        # SOCP adaptive stencil enlargement (Issue #1106), joint_socp scheme only.
        self._socp_max_stencil_enlargements: int = int(socp_max_stencil_enlargements)
        self._socp_enlargement_step: int = int(socp_enlargement_step)

        # Monotonicity scheme (single source of truth) — already set above (v0.18.0 rename)
        # self.monotonicity_scheme and self.qp_optimization_level both = resolved monotonicity_scheme

        # QP usage target (deprecated, kept for backward compatibility)
        self.qp_usage_target = qp_usage_target

        # QP solver selection
        self.qp_solver = qp_solver
        self.qp_warm_start = qp_warm_start
        self.qp_constraint_mode = qp_constraint_mode

        # Congestion mode for Hamiltonian coupling
        self.congestion_mode = congestion_mode
        # Issue #1426: congestion_mode is stored but never read — 'multiplicative' was a silent
        # no-op. Fail loud rather than silently behaving as 'additive'.
        if congestion_mode != "additive":
            raise NotImplementedError(
                f"congestion_mode={congestion_mode!r} is not implemented (Issue #1426): it is stored but "
                f"never applied, so it would silently behave as 'additive'. Only 'additive' is supported."
            )

        # Initialize QP components (will be fully initialized after neighborhoods are built)
        # Map qp_solver parameter to QPSolver backend
        qp_backend = "auto" if qp_solver == "osqp" else "scipy-slsqp"
        self._qp_cache = QPCache(max_size=1000)
        self._qp_solver_instance = QPSolver(
            backend=qp_backend,
            enable_warm_start=qp_warm_start,
            cache=self._qp_cache,
        )

        # Legacy warm-start cache (kept for backward compatibility, but unused)
        self._qp_warm_start_cache: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}

        # Placeholder for MonotonicityEnforcer - will be initialized after neighborhoods built
        self._monotonicity_enforcer: MonotonicityEnforcer | None = None

        # QP stats placeholder (will be aliased to enforcer.stats after initialization)
        self.qp_stats: dict[str, Any] = {}
        self._current_point_idx = 0

        # Adaptive neighborhood parameters
        self.adaptive_neighborhoods = adaptive_neighborhoods
        self.max_delta_multiplier = max_delta_multiplier

        # Cache grid size info from geometry
        self._n_spatial_grid_points = self._compute_n_spatial_grid_points()

        # Cache domain bounds from geometry
        self.domain_bounds = self._get_domain_bounds()

        # Compute k_min from Taylor order if not provided
        from math import comb

        n_derivatives_required = comb(self.dimension + taylor_order, taylor_order) - 1
        if k_min is None:
            self.k_min = n_derivatives_required
        else:
            # Ensure k_min is at least what's required for Taylor expansion
            if k_min < n_derivatives_required:
                warnings.warn(
                    f"k_min={k_min} is less than required for Taylor order {taylor_order} "
                    f"in {self.dimension}D (need {n_derivatives_required}). "
                    f"Using k_min={n_derivatives_required} instead.",
                    UserWarning,
                    stacklevel=2,
                )
                self.k_min = n_derivatives_required
            else:
                self.k_min = k_min

        # Store new infrastructure parameters
        self._use_new_infrastructure = use_new_infrastructure
        # Issue #1553: fail loud at construction rather than deep in Newton-Jacobian assembly.
        # (An unrecognized derivative_method is already rejected at construction by the operator
        # dispatch's else-branch below, so only 'rbf' needs an explicit guard here.)
        # Since #1526 the non-LCR weight path routes through NeighborhoodBuilder's Taylor-SVD builder,
        # which consumes SVD factors LocalRBFOperator.get_taylor_data does not provide (a dummy shim
        # returning None), so every real 'rbf' solve raises an undiagnostic
        # ``'NoneType' object has no attribute 'T'``. Repairing it also owes threading obstacle_sdf
        # (the pre-#1124 wall-coupling seam). Until a genuine RBF weight-builder + convergence test
        # lands, 'rbf' is unsupported: a half-working parallel path is worse than none.
        if derivative_method == "rbf":
            raise NotImplementedError(
                "HJBGFDMSolver derivative_method='rbf' is not supported (Issue #1553): since #1526 the "
                "RBF branch has no working differentiation-weight builder and would reopen the #1124 "
                "obstacle seam. Use derivative_method='taylor' (the default)."
            )
        self._derivative_method = derivative_method
        self._rbf_kernel = rbf_kernel
        self._rbf_poly_degree = rbf_poly_degree

        # Local Coordinate Rotation for boundary accuracy (Issue #531)
        self._use_local_coordinate_rotation = use_local_coordinate_rotation

        # Ghost Nodes for Neumann BC enforcement (Issue #531 - Terminal BC compatibility)
        self._use_ghost_nodes = use_ghost_nodes

        # Wind-Dependent BC for viscosity solution compatibility
        self._use_wind_dependent_bc = use_wind_dependent_bc

        # Hyperviscosity parameter for wind-dependent BC stabilization
        # epsilon > 0 adds damping: u_ghost = 2u_b - u_m - epsilon*(u_b - u_m)
        # Recommended: 0.0 (no damping) to 0.3 (moderate damping)
        self._wind_bc_hyperviscosity = 0.0  # Default: no hyperviscosity

        # Check for mutual exclusivity (ghost nodes takes precedence)
        if self._use_ghost_nodes and self._use_local_coordinate_rotation:
            warnings.warn(
                "Both use_ghost_nodes and use_local_coordinate_rotation are enabled. "
                "Ghost nodes take precedence and LCR will be disabled for boundary points.",
                UserWarning,
                stacklevel=2,
            )

        # Wind-dependent BC requires ghost nodes
        if self._use_wind_dependent_bc and not self._use_ghost_nodes:
            raise ValueError(
                "use_wind_dependent_bc=True requires use_ghost_nodes=True. "
                "Wind-dependent BC is a modification of the ghost nodes method."
            )

        # DEBUG: Print wind-BC configuration once at initialization
        if self._use_wind_dependent_bc:
            import sys

            print(f"\n[Wind-BC INIT] Enabled with {len(boundary_indices)} boundary points", flush=True, file=sys.stderr)

        # Create differential operator using Strategy Pattern
        if use_new_infrastructure:
            # New infrastructure: TaylorOperator or LocalRBFOperator
            if derivative_method == "taylor":
                self._gfdm_operator = TaylorOperator(
                    points=collocation_points,
                    delta=delta,
                    taylor_order=taylor_order,
                    weight_function=weight_function,
                    k_neighbors=k_neighbors,
                    neighborhood_mode=neighborhood_mode,
                    # Issue #711 periodic support. The resolved geometry, so this and boundary
                    # detection cannot disagree about which axes are periodic (#1841).
                    geometry=self._collocation_geometry,
                    # Issue #1124: visibility filter at operator level so
                    # D_lap / D_grad respect obstacle connectivity (not just
                    # NeighborhoodBuilder's post-filter view).
                    obstacle_sdf=obstacle_sdf,
                    visibility_samples=visibility_samples,
                    visibility_margin=visibility_margin,
                )
            elif derivative_method == "rbf":
                self._gfdm_operator = create_operator(
                    points=collocation_points,
                    delta=delta,
                    method="rbf",
                    kernel=rbf_kernel,
                    poly_degree=rbf_poly_degree,
                    k_neighbors=k_neighbors,
                    neighborhood_mode=neighborhood_mode,
                )
            else:
                raise ValueError(f"Unknown derivative_method: {derivative_method}")

            # Initialize BC handler with Row Replacement pattern
            self._bc_handler = DirectCollocationHandler()

            # Initialize BoundaryHandler component (Issue #545: composition over mixins)
            # This component handles boundary normals, LCR, ghost nodes, etc.
            self._boundary_handler = BoundaryHandler(
                collocation_points=collocation_points,
                dimension=self.dimension,
                domain_bounds=self.domain_bounds,
                boundary_indices=self.boundary_indices,
                neighborhoods={},  # Will be populated by _build_neighborhood_structure
                boundary_conditions=self.boundary_conditions,
                use_ghost_nodes=self._use_ghost_nodes,
                use_wind_dependent_bc=self._use_wind_dependent_bc,
                gfdm_operator=self._gfdm_operator,
                bc_property_getter=lambda prop, default=None: self._get_boundary_condition_property(prop) or default,
                gradient_computer=None,  # Will be set later if needed
            )

            # Compute boundary normals for Neumann BC
            self._boundary_normals = self._boundary_handler.compute_boundary_normals()
            # Store in handler for access by other components
            self._boundary_handler.boundary_normals = self._boundary_normals

            # Create unified BC config (single source of truth)
            self._bc_config = self._boundary_handler.create_bc_config()

            # Pre-classify every boundary collocation point to (BoundaryFace,
            # BCSegment) at construction time. Fails fast if any point cannot
            # be matched — better diagnostic than discovering it as a zero
            # Jacobian row 80 Newton iters later.
            self._preclassify_boundary_points()

            # Initialize NeighborhoodBuilder component (Issue #545: composition over mixins)
            # This component handles stencil construction, Taylor matrices, weight functions
            self._neighborhood_builder = NeighborhoodBuilder(
                collocation_points=collocation_points,
                dimension=self.dimension,
                delta=delta,
                taylor_order=taylor_order,
                weight_function=weight_function,
                weight_scale=weight_scale,
                k_min=self.k_min,
                adaptive_neighborhoods=adaptive_neighborhoods,
                max_delta_multiplier=max_delta_multiplier,
                boundary_indices=self.boundary_indices,
                n_derivatives=0,  # Will be set after multi_indices are determined
                multi_indices=[],  # Will be populated after operator initialization
                gfdm_operator=self._gfdm_operator,
                use_local_coordinate_rotation=self._use_local_coordinate_rotation,
                boundary_handler=self._boundary_handler,
                obstacle_sdf=obstacle_sdf,
                visibility_samples=visibility_samples,
                visibility_margin=visibility_margin,
            )
        else:
            raise ValueError(
                "use_new_infrastructure=False is no longer supported (removed in v0.17.15). "
                "Use use_new_infrastructure=True (default) with TaylorOperator."
            )

        # Get multi-indices from operator
        self.multi_indices = self._gfdm_operator.multi_indices
        self.n_derivatives = len(self.multi_indices)

        # Update neighborhood builder with multi_indices (for new infrastructure)
        if self._neighborhood_builder is not None:
            self._neighborhood_builder.multi_indices = self.multi_indices
            self._neighborhood_builder.n_derivatives = self.n_derivatives

        # Store spatial shape for grid<->collocation interpolation
        # This is needed for _map_grid_to_collocation and _map_collocation_to_grid
        # get_grid_shape() returns node counts (Nx+1, Ny+1), not cell counts
        self._output_spatial_shape = tuple(self.problem.geometry.get_grid_shape())

        # Initialize grid-collocation mapper (Issue #545: composition over mixins)
        self._mapper = GridCollocationMapper(
            collocation_points=collocation_points,
            grid_shape=self._output_spatial_shape,
            domain_bounds=self.domain_bounds,
        )

        # Build neighborhood structure - uses GFDMOperator's neighborhoods as base,
        # only extends for points needing adaptive delta enlargement
        if self._neighborhood_builder is not None:
            self._neighborhood_builder.build_neighborhood_structure()
        else:
            # Legacy fallback
            self._build_neighborhood_structure()

        # Update boundary handler neighborhoods reference (after they're built)
        if self._boundary_handler is not None:
            self._boundary_handler.neighborhoods = self.neighborhoods

        # Apply Ghost Nodes for Neumann BC enforcement (Issue #531 - Terminal BC compatibility)
        # Ghost nodes take precedence over LCR if both are enabled
        # This must be called BEFORE Taylor matrices are built, since it augments neighborhoods
        if self._use_ghost_nodes:
            if self._boundary_handler is not None:
                # Per-point dispatch (Issue #1110 Bug A fix): pass the
                # pre-classified BC type lookup so ghost augmentation
                # correctly handles mixed-BC setups — apply to NEUMANN/
                # NO_FLUX wall points, skip DIRICHLET exit points.
                self._boundary_handler.apply_ghost_nodes_to_neighborhoods(
                    bc_type_for_point=self._get_bc_type_for_point,
                )
            else:
                # Legacy fallback (shouldn't happen with new infrastructure)
                self._apply_ghost_nodes_to_neighborhoods()
        elif self._use_local_coordinate_rotation:
            # Apply Local Coordinate Rotation for boundary stencils (Issue #531)
            # This modifies neighborhoods by adding rotated_offsets for better normal derivatives
            if self._boundary_handler is not None:
                self._boundary_handler.apply_local_coordinate_rotation()
            else:
                # Legacy fallback (shouldn't happen with new infrastructure)
                self._apply_local_coordinate_rotation()

        # Build reverse neighborhood map for sparse Jacobian (point j -> rows affected)
        if self._neighborhood_builder is not None:
            self._neighborhood_builder.build_reverse_neighborhoods()
        else:
            # Legacy fallback
            self._build_reverse_neighborhoods()

        # Build Taylor matrices for extended neighborhoods
        if self._neighborhood_builder is not None:
            self._neighborhood_builder.build_taylor_matrices()
        else:
            # Legacy fallback
            self._build_taylor_matrices()

        # Initialize MonotonicityEnforcer component (Issue #545: composition over mixins)
        # Only create enforcer if QP optimization is enabled
        if self.qp_optimization_level != "none":
            self._monotonicity_enforcer = MonotonicityEnforcer(
                qp_solver=self._qp_solver_instance,
                qp_constraint_mode=self.qp_constraint_mode,
                collocation_points=self.collocation_points,
                neighborhoods=self.neighborhoods,
                multi_indices=self.multi_indices,
                domain_bounds=self.domain_bounds,
                delta=self.delta,
                sigma_function=self._get_sigma_value,
            )
            # Alias qp_stats to enforcer.stats for backward compatibility
            self.qp_stats = self._monotonicity_enforcer.stats
        else:
            self._monotonicity_enforcer = None
            # Initialize empty qp_stats for "none" level
            self.qp_stats = {
                "total_qp_solves": 0,
                "qp_times": [],
                "violations_detected": 0,
                "violation_point_indices": set(),
                "violation_laplacian": 0,
                "violation_gradient": 0,
                "violation_higher_order": 0,
                "points_checked": 0,
                "qp_successes": 0,
                "qp_failures": 0,
                "qp_fallbacks": 0,
                "slsqp_solves": 0,
                "lbfgsb_solves": 0,
                "osqp_solves": 0,
                "osqp_failures": 0,
            }

        # Initialize precomputed joint SOCP stencils first (joint_socp scheme only).
        # Audit-major Phase 1B: enforce M-matrix + per-edge cone at all interior nodes
        # where the joint SOCP is feasible (paper Theorem `thm:joint_socp_feasibility`).
        self._joint_socp_stencils = None
        if self.monotonicity_scheme == "joint_socp":
            from mfgarchon.alg.numerical.gfdm_components.joint_socp import (
                PrecomputedJointSocpStencils,
            )

            interior_indices = np.setdiff1d(np.arange(self.n_points), self.boundary_indices)
            self._joint_socp_stencils = PrecomputedJointSocpStencils(
                points=self.collocation_points,
                interior_indices=interior_indices,
                delta=delta,
                neighborhoods=self.neighborhoods,
                cone_constant_C=8.0,  # higher C → cone less binding, picks fast-path Wendland-LSQ where M-matrix holds
                eps_pos=0.0,
                # C-bisection: retry infeasible stencils with progressively
                # larger C up to C_max. Most marginally infeasible stencils
                # become feasible at C ∈ (1, 8].
                cone_constant_C_max=8.0,
                # Always-feasible relaxed SOCP fallback. Eliminates the
                # discrete scheme switch between joint_socp (overrides L+D)
                # and Phase 2 M-matrix-QP (overrides only L; leaves D as bare
                # W-T) that creates discontinuous discretization on irregular
                # 2D clouds. With this enabled, ALL interior points get
                # consistent (L, D) from the SOCP framework — well-conditioned
                # stencils recover exact joint_socp; marginally infeasible
                # stencils smoothly degrade via slack penalties (continuous
                # mapping cloud→stencil weights). Empirically required for
                # 2D obstacle navigation; latent on 1D / quasi-uniform clouds
                # where infeasibility doesn't fire.
                use_relaxed_fallback=True,
                lambda_M=1.0e4,
                lambda_C=1.0e4,
                # SOCP-infeasibility-triggered adaptive stencil enlargement
                # (Issue #1106). Default 0 (OFF) — paper / default path unchanged.
                max_stencil_enlargements=self._socp_max_stencil_enlargements,
                enlargement_step=self._socp_enlargement_step,
                # Same obstacle visibility filter the base neighborhoods were
                # post-filtered with (Issue #1124 / #1102): enlargement must not
                # re-add a cross-wall / cross-obstacle neighbor that the base
                # filter removed. None (no obstacle) => enlargement unchanged.
                obstacle_sdf=obstacle_sdf,
                visibility_samples=visibility_samples,
                visibility_margin=visibility_margin,
            )
            stats = self._joint_socp_stencils.stats
            relax_C_msg = (
                f"; {stats['n_relaxed_C']} C-relaxed (max C={stats['max_achieved_C']:.2f})"
                if stats.get("n_relaxed_C", 0) > 0
                else ""
            )
            relax_fb_msg = (
                f"; {stats['n_relaxed_fallback']} via relaxed SOCP "
                f"(max ε_M={stats['max_eps_M']:.2e}, ε_C={stats['max_eps_C']:.2e})"
                if stats.get("n_relaxed_fallback", 0) > 0
                else ""
            )
            enlarge_msg = (
                f"; {stats['n_enlarged']} via adaptive enlargement "
                f"(max {stats['max_enlargement_steps']} steps, Issue #1106)"
                if stats.get("n_enlarged", 0) > 0
                else ""
            )
            # Issue #1565: the SOCP loop iterates INTERIOR indices only, and the M-matrix-QP
            # precompute buffer is BOUNDARY-only, so an SOCP-infeasible interior node matches
            # neither has_stencil branch at solve time and falls through to the bare (non-monotone)
            # Wendland-Taylor LSQ weights — NOT a Phase-2 M-matrix QP (which never covers interior
            # nodes). Report the real fallback so the monotone fraction is not over-stated.
            logger.info(
                f"Precomputed joint SOCP stencils: feasible {stats['n_feasible']}/"
                f"{stats['n_interior']} interior "
                f"({stats['n_fast_path']} via Wendland-LSQ fast-path, "
                f"{stats['n_socp']} via CLARABEL SOCP{relax_C_msg}{enlarge_msg}{relax_fb_msg}) in "
                f"{stats['time_ms']:.1f}ms; SOCP-infeasible {stats['n_infeasible']} interior node(s) "
                f"fall through to bare Wendland-Taylor LSQ (NON-MONOTONE; no Phase-2 QP covers "
                f"interior nodes)"
            )

        # Initialize precomputed M-matrix QP stencils at boundary nodes.
        # Activated under both `qp_optimization_level == "precompute"` (legacy
        # qp_m_matrix scheme) and `monotonicity_scheme == "joint_socp"` (which
        # internally aliases qp_optimization_level to "precompute" — see above).
        # SOCP-infeasible interior nodes fall through to bare Wendland-Taylor;
        # extending the buffer set to cover them was empirically destabilizing
        # (Lap-only correction creates Lap/Grad inconsistency at those nodes).
        self._precomputed_stencils: PrecomputedMonotoneStencils | None = None
        if self.qp_optimization_level == "precompute":
            is_buffer = np.zeros(self.n_points, dtype=bool)
            is_buffer[self.boundary_indices] = True
            self._precomputed_stencils = PrecomputedMonotoneStencils(
                is_boundary=is_buffer,
                neighborhoods=self.neighborhoods,
                points=self.collocation_points,
                delta=delta,
                tolerance=1e-6,
            )
            logger.info(
                f"Precomputed monotone stencils: {self._precomputed_stencils.stats['n_monotonized']}/{self._precomputed_stencils.stats['n_boundary']} "
                f"buffer points in {self._precomputed_stencils.stats['time_ms']:.1f}ms"
            )

        # Lazy-initialized cache attributes
        # These are expensive to compute and only created when needed
        self._D_grad: list | None = None  # Gradient differentiation matrices
        self._D_lap: Any | None = None  # Laplacian differentiation matrix
        self._cached_derivative_weights: dict | None = None  # Pre-computed GFDM weights
        self._f_potential_warned: bool = False  # One-time warning for unused f_potential (Issue #766)

    def _compute_n_spatial_grid_points(self) -> int:
        """Compute total number of spatial grid points from geometry."""
        grid_shape = self.problem.geometry.get_grid_shape()
        return int(np.prod(grid_shape))

    def _get_boundary_condition_property(self, property_name: str) -> Any:
        """Get boundary condition property - returns None if not available.

        BC validation is deferred to solve time to allow testing internal mechanics
        without requiring full BC specification.

        For mixed BCs, returns None with a warning (allows fallback to per-point BC).
        """
        # No BC specified - return None (validation deferred to solve time)
        if self.boundary_conditions is None:
            return None

        # Try dictionary access first (doesn't trigger property)
        if isinstance(self.boundary_conditions, dict):
            return self.boundary_conditions.get(property_name)

        # Try attribute access (may raise ValueError for mixed BC properties)
        try:
            return getattr(self.boundary_conditions, property_name)
        except AttributeError:
            return None
        except ValueError:
            # Mixed BC - warn and return None to allow fallback to default_bc
            # Per-point BC types are not yet supported in HJB GFDM solver.
            # The solver will use default_bc for all boundary points.
            if not getattr(self, "_mixed_bc_warned", False):
                logger.info(
                    f"Mixed BC detected: '{property_name}' is not uniform. "
                    f"Per-point BC types will be applied (DIRICHLET at exits, NEUMANN at walls)."
                )
                self._mixed_bc_warned = True
            return None

    def _get_bc_type_for_point(self, point_idx: int) -> str:
        """Determine BC type for a boundary collocation point.

        Resolution order:

        1. **Pre-classified table** (preferred): for mixed BC, the segment
           was resolved at solver __init__ time and stored in
           ``self._bc_segment_per_point``. O(1) lookup, no re-classification.
        2. **Uniform BC**: read global type from ``_bc_config``.

        For mixed BC where the point was *not* pre-classified, this method
        raises ``ValueError`` rather than silently falling back to
        ``self.boundary_conditions.default_bc`` (historically defaulted to
        PERIODIC — the source of silent zero Jacobian rows; Issue #1100 removed
        that implicit default so it is now ``None``/fail-loud). The
        pre-classification at __init__ already raised on unmatched points
        with full diagnostic; reaching here means the boundary_indices set
        was mutated after __init__, which is a programmer error.

        Returns:
            BC type string: "dirichlet", "neumann", or any other BCType
            ``.value.lower()`` for completeness.
        """
        try:
            is_mixed = self.boundary_conditions.is_mixed
        except AttributeError:
            is_mixed = False

        if is_mixed:
            # Pre-classified table is authoritative for mixed BC.
            try:
                segment = self._bc_segment_per_point[point_idx]
            except (AttributeError, KeyError) as exc:
                raise ValueError(
                    f"_get_bc_type_for_point({point_idx}): point not in pre-classified "
                    f"table. boundary_indices was likely mutated after solver __init__, "
                    f"or this method was called before HJBGFDMSolver.__init__ completed. "
                    f"Pre-classified count: "
                    f"{len(getattr(self, '_bc_segment_per_point', {}))}/"
                    f"{len(self.boundary_indices)}."
                ) from exc
            if segment.bc_type == BCType.DIRICHLET:
                return "dirichlet"
            if segment.bc_type in (BCType.NEUMANN, BCType.NO_FLUX):
                return "neumann"
            return segment.bc_type.value.lower()

        # Uniform BC: global type from config.
        bc_type = self._bc_config.get("type") if self._bc_config else None
        if bc_type is None:
            raise ValueError(
                "BC type required but not specified in config (uniform BC path). "
                "Provide boundary_conditions= when constructing HJBGFDMSolver."
            )
        return bc_type

    def _refresh_boundary_conditions_if_changed(self) -> None:
        """Re-sync BC-derived state from geometry if it changed since construction.

        GFDM snapshots ``self.boundary_conditions`` and the preclassified
        ``_bc_segment_per_point`` map at ``__init__``; FDM instead re-reads
        ``get_boundary_conditions()`` every solve (hjb_fdm.py:319). When the
        coupling layer resolves a provider per Picard iteration via
        ``problem.using_resolved_bc`` (Issue #625), it swaps
        ``geometry.boundary_conditions`` for a new object whose segments carry
        the resolved scalar value (e.g. AdjointConsistentProvider's
        ``g = -sigma^2/2 * d ln(m)/dn``). Without this refresh the GFDM/Howard
        path would solve every iteration against the construction-time BC,
        silently freezing the adjoint coupling in the >1000x-impact regime.

        No-op when:
        - BC came from an explicit constructor argument (static, never resolved);
        - the live geometry BC is the same object as the cached snapshot
          (``with_resolved_providers`` fast-paths ``return self`` for
          provider-free BC, so the object is unchanged).
        """
        if not self._bc_from_geometry:
            return
        live_bc = self.get_boundary_conditions()
        if live_bc is None or live_bc is self.boundary_conditions:
            return
        self.boundary_conditions = live_bc
        # Keep the handler's BC reference consistent (its default_bc fallback in
        # create_bc_config reads it); normals are geometry-only and unaffected.
        self._boundary_handler.boundary_conditions = live_bc
        self._bc_config = self._boundary_handler.create_bc_config()
        self._preclassify_boundary_points()

    def _preclassify_boundary_points(self) -> None:
        """Pre-classify every boundary collocation point to a BCSegment + face + normal.

        Called at __init__ time and re-run by ``_refresh_boundary_conditions_if_changed``
        when the geometry BC is resolved per Picard iteration. Populates three
        companion maps:

        - ``self._bc_face_per_point[i]``: BoundaryFace the point lies on.
        - ``self._bc_segment_per_point[i]``: BCSegment that applies to ``i``.
        - ``self._bc_normal_per_point[i]``: outward unit normal (axis-aligned,
          from the face — not from any SDF gradient).

        Raises if any boundary point cannot be classified to a face or matched
        to a segment. This converts a class of latent failures — the historical
        silent ``default_bc=PERIODIC`` fallback (removed in Issue #1100) plus
        zero Jacobian rows discovered 80 Newton iterations later — into a loud,
        diagnosable construction-time error.

        Only runs for mixed-BC setups; uniform BC keeps the global-type fast
        path. Skipped entirely if ``len(self.boundary_indices) == 0``.
        """
        self._bc_face_per_point: dict[int, BoundaryFace] = {}
        self._bc_segment_per_point: dict[int, BCSegment] = {}
        self._bc_normal_per_point: dict[int, np.ndarray] = {}

        if len(self.boundary_indices) == 0 or self.boundary_conditions is None:
            return
        try:
            is_mixed = self.boundary_conditions.is_mixed
        except AttributeError:
            return
        if not is_mixed:
            return

        bounds = self._get_domain_bounds_array()
        sorted_segments = sorted(
            self.boundary_conditions.segments,
            key=lambda seg: seg.priority,
            reverse=True,
        )

        # tolerance for classification: 1e-6 covers ε=1e-6 collocation
        # generators (e.g. SDF-clipped boundary points placed at micron
        # distance from the wall). Users with looser collocation can override
        # via a future kwarg; current default is conservative.
        tol = BOUNDARY_TOL
        unmatched: list[tuple[int, np.ndarray, BoundaryFace | None, str]] = []

        for i in self.boundary_indices:
            i = int(i)
            point = self.collocation_points[i]

            face = self.boundary_conditions.identify_boundary_face(
                point=point,
                tolerance=tol,
                domain_bounds=bounds,
            )
            if face is None:
                unmatched.append((i, point, None, "no BoundaryFace match"))
                continue

            matching_segment: BCSegment | None = None
            for seg in sorted_segments:
                if seg.matches_point(
                    point=point,
                    boundary_id=face.to_string(),
                    domain_bounds=bounds,
                ):
                    matching_segment = seg
                    break

            if matching_segment is None:
                unmatched.append((i, point, face, f"BoundaryFace={face!r} not covered by any segment"))
                continue

            self._bc_face_per_point[i] = face
            self._bc_segment_per_point[i] = matching_segment
            self._bc_normal_per_point[i] = self.boundary_conditions.outward_normal_for_face(
                face, dimension=self.dimension
            )

        if unmatched:
            lines = [
                f"HJBGFDMSolver: BC pre-classification failed for "
                f"{len(unmatched)}/{len(self.boundary_indices)} boundary points.",
                "",
                "Common causes:",
                f"  1. Collocation generator places boundary points >{tol:.0e} off the wall "
                "(e.g. ε=1e-4 with default tol=1e-6) → bump tolerance, or shrink ε.",
                "  2. BoundaryConditions.segments don't cover every geometric face → add "
                "a segment for the missing face, or set boundary='all'.",
                "  3. domain_bounds inferred from problem.geometry differ from collocation "
                "extent (e.g. quirky obstacle-clipping geometry).",
                "",
                "Unmatched points (first 5):",
            ]
            for i, point, face, reason in unmatched[:5]:
                lines.append(f"  pt {i} at {point.tolist()}: face={face!r} -- {reason}")
            if len(unmatched) > 5:
                lines.append(f"  ... and {len(unmatched) - 5} more")
            raise ValueError("\n".join(lines))

    def _detect_boundary_indices(self, collocation_points: np.ndarray) -> np.ndarray:
        """Auto-detect boundary point indices from collocation points and domain bounds.

        Points are classified as boundary if they lie within tolerance of any domain boundary.

        Args:
            collocation_points: Array of shape (n_points, dimension) with collocation coordinates.

        Returns:
            Array of boundary point indices. Empty array if bounds cannot be determined.

        Note:
            Issue #542 fix - enables automatic BC enforcement without explicit boundary_indices.
        """
        # Get domain bounds
        bounds = self._get_domain_bounds_for_detection()
        if bounds is None or len(bounds) == 0:
            return np.array([], dtype=int)

        # A periodic axis has no boundary: its two faces are the same physical place, so a point
        # lying on one is an interior point of the torus. Classifying it as boundary sent it to a
        # row builder with no periodic row, which raised -- so the capability was reachable only by
        # hand-passing an empty `boundary_indices`, i.e. lying to the solver about its own
        # geometry (Issue #1841). Read from the same geometry the bounds came from.
        periodic_dims = self._periodic_dims_for_detection()

        tol = BOUNDARY_TOL
        boundary_mask = np.zeros(len(collocation_points), dtype=bool)

        for d, (d_min, d_max) in enumerate(bounds):
            if d in periodic_dims:
                continue
            if d < collocation_points.shape[1]:
                # Points at min or max boundary in this dimension
                at_min = np.abs(collocation_points[:, d] - d_min) < tol
                at_max = np.abs(collocation_points[:, d] - d_max) < tol
                boundary_mask |= at_min | at_max

        return np.where(boundary_mask)[0]

    def _periodic_dims_for_detection(self) -> tuple[int, ...]:
        """Which axes are periodic, per the geometry that supplied the bounds.

        `SupportsPeriodic.periodic_dimensions` is the declared source; a geometry that does not
        implement it reports nothing periodic, which is the pre-#1841 behaviour.

        Read from `self._collocation_geometry` -- the SAME object the operator's periodic wrap is
        built from, and the same one `_get_domain_bounds_for_detection` takes bounds from -- so no
        two of the three can disagree. `len(...)`, not truthiness: `np.array([0])` is
        falsy on its single element and would silently demote a geometry whose only periodic axis
        is 0, and a multi-element array raises on `if dims` outright.
        """
        dims = getattr(self._collocation_geometry, "periodic_dimensions", None)
        return tuple(dims) if dims is not None and len(dims) > 0 else ()

    def _refuse_topology_disagreeing_with_bc(self) -> None:
        """A periodic axis and a wall on that same axis are different problems; say so.

        The geometry decides topology (a torus has no boundary) and the boundary conditions decide
        what happens at a wall, so the two can contradict each other. They did, silently, in both
        directions (#1841 review):

        - a PERIODIC problem with a non-periodic `collocation_geometry` built no wrap, and returned
          a seam of 1.36e+00 against a tolerance of 1e-9;
        - a NO_FLUX problem with a periodic `collocation_geometry` had both faces skipped by
          detection and was solved with no boundary row anywhere -- the declared wall silently gone.

        Neither is a configuration anyone means, and each answers a different problem than the
        caller posed, which is the failure this whole campaign is about (#1822). Refused at
        construction, where the caller can still act, rather than deep in a row builder or not at
        all.
        """
        if self._collocation_geometry is self.problem.geometry:
            return  # one object cannot disagree with itself

        collocation_periodic = self._periodic_dims_for_detection()
        problem_dims = getattr(self.problem.geometry, "periodic_dimensions", None)
        if problem_dims is None:
            return  # the problem's geometry states no topology, so there is nothing to contradict
        bc_periodic = tuple(problem_dims)
        if set(collocation_periodic) != set(bc_periodic):
            raise ValueError(
                f"collocation_geometry declares periodic axes {collocation_periodic or '()'} while the "
                f"problem's boundary conditions imply {bc_periodic or '()'}. A periodic axis has no "
                f"boundary and a wall is a boundary, so these describe different problems and the "
                f"solver cannot honour both. Pass a collocation_geometry whose periodic_dims match "
                f"the boundary conditions, or omit it and the problem's own geometry is used."
            )

    def _get_domain_bounds_for_detection(self) -> list[tuple[float, float]] | None:
        """Get domain bounds for boundary detection (before full initialization).

        From `self._collocation_geometry`, the SAME object `_periodic_dims_for_detection` reads.
        Detection classifies collocation points, and those live in the collocation domain, so this
        is also the correct source rather than merely the consistent one. Reading bounds from the
        problem while reading periodicity from the collocation geometry left the two disagreeing one
        call frame apart: a no-flux problem given a periodic collocation geometry had every face
        skipped and was solved with no boundary row at all, silently (#1841 review).
        """
        # Try geometry interface first
        try:
            geom = self._collocation_geometry
            if geom is not None:
                try:
                    bounds_result = geom.get_bounds()
                    if bounds_result is not None:
                        min_coords, max_coords = bounds_result
                        return [(float(min_coords[d]), float(max_coords[d])) for d in range(len(min_coords))]
                except AttributeError:
                    pass
                try:
                    return list(geom.bounds)
                except AttributeError:
                    pass
        except AttributeError:
            pass
        return None

    def _get_domain_bounds(self) -> list[tuple[float, float]]:
        """Get domain bounds from geometry, falling back to the collocation cloud extent.

        Returns:
            List of (min, max) tuples for each dimension.

        Note:
            Issue #542 fix - removed hasattr/getattr, using try/except for clearer failure modes.
        """
        # Try geometry interface first (modern API)
        try:
            geom = self.problem.geometry
            if geom is not None:
                try:
                    # Prefer get_bounds() method
                    bounds_result = geom.get_bounds()
                    if bounds_result is not None:
                        min_coords, max_coords = bounds_result
                        return [(float(min_coords[d]), float(max_coords[d])) for d in range(len(min_coords))]
                except AttributeError:
                    pass
                try:
                    # Fallback to .bounds property
                    return list(geom.bounds)
                except AttributeError:
                    pass
        except AttributeError:
            pass

        # Last resort: infer from collocation points
        mins = self.collocation_points.min(axis=0)
        maxs = self.collocation_points.max(axis=0)
        return list(zip(mins.astype(float).tolist(), maxs.astype(float).tolist(), strict=True))

    def _get_domain_bounds_array(self) -> np.ndarray | None:
        """Get domain bounds as numpy array for BCSegment.matches_point().

        Returns:
            Array of shape (dimension, 2) where bounds[i, 0] = min and bounds[i, 1] = max
            for dimension i. Returns None if bounds cannot be determined.

        Note:
            Issue #542 fix - provides bounds in format expected by BCSegment.
        """
        bounds_list = self._get_domain_bounds()
        if not bounds_list:
            return None
        return np.array(bounds_list, dtype=float)

    def _infer_boundary_id(
        self, point: np.ndarray, domain_bounds: np.ndarray | None, tol: float = BOUNDARY_TOL
    ) -> str | None:
        """Infer boundary identifier for a point on rectangular domain boundary.

        This is optional - BCSegment.matches_point() can work without boundary_id
        using SDF or normal matching. For rectangular domains, providing boundary_id
        enables efficient segment matching via the 'boundary' attribute.

        Args:
            point: Spatial coordinates (dimension,)
            domain_bounds: Domain bounds array (dimension, 2) or None
            tol: Tolerance for boundary detection

        Returns:
            Boundary identifier like "x_min", "y_max", or None if not on axis-aligned boundary.

        Note:
            Issue #542 fix - separated boundary inference from BC matching.
            Returns None for non-rectangular domains or interior/corner points.
        """
        if domain_bounds is None:
            return None

        # Check each axis for boundary proximity
        for axis_idx in range(min(len(point), len(domain_bounds))):
            if abs(point[axis_idx] - domain_bounds[axis_idx, 0]) < tol:
                axis_name = ["x", "y", "z"][axis_idx] if axis_idx < 3 else f"dim{axis_idx}"
                return f"{axis_name}_min"
            elif abs(point[axis_idx] - domain_bounds[axis_idx, 1]) < tol:
                axis_name = ["x", "y", "z"][axis_idx] if axis_idx < 3 else f"dim{axis_idx}"
                return f"{axis_name}_max"

        return None

    def _get_domain_sdf(self) -> callable | None:
        """Get signed distance function from geometry if available.

        Returns:
            SDF callable or None if geometry doesn't provide one.

        Note:
            Issue #542 fix - enables BC matching on SDF-based geometries.
        """
        try:
            return self.problem.geometry.sdf
        except AttributeError:
            return None

    def _compute_domain_volume(self) -> float:
        """Compute the volume (area in 2D) of the domain.

        Used for normalizing density in multiplicative congestion mode,
        where H = (1 + γ|Ω|m)|p|²/(2λ). The |Ω|m factor makes γ dimensionless.

        Returns:
            Domain volume/area as a scalar.
        """
        try:
            return self._domain_volume
        except AttributeError:
            pass

        bounds = self.domain_bounds
        volume = 1.0
        for d_min, d_max in bounds:
            volume *= d_max - d_min

        self._domain_volume = volume
        return volume

    # =========================================================================
    # Boundary Methods: Provided by BoundaryHandler component (Issue #545)
    # compute_boundary_normals, build_rotation_matrix, apply_local_coordinate_rotation
    # rotate_derivatives_back, apply_ghost_nodes_to_neighborhoods
    # build_neumann_bc_weights
    # =========================================================================

    def _compute_gradient_at_point(self, u_values: np.ndarray, point_idx: int) -> np.ndarray:
        """
        Compute gradient ∇u at a single point using GFDM weights.

        Args:
            u_values: Solution vector at all collocation points
            point_idx: Index of point where gradient is computed

        Returns:
            Gradient vector ∇u, shape (dimension,)
        """
        # Get neighborhood for this point
        neighborhood = self.neighborhoods[point_idx]
        neighbor_indices = neighborhood["indices"]

        # Get Taylor weights for derivatives
        weights = neighborhood["weights"]

        # Extract gradient weights (first-order derivatives: columns 1 to dimension+1)
        # weights structure: [u, u_x, u_y, u_xx, u_xy, u_yy, ...] for 2D
        grad_weights = weights[:, 1 : 1 + self.dimension]  # Shape: (n_neighbors, dimension)

        # Get neighbor values (using standard ghost mirroring, no wind-dependent check)
        # This avoids circular dependency: we need gradient to check wind direction,
        # but we can't check wind direction while computing the gradient!
        # Solution: use standard ghosts here, wind-dependent BC applies at derivative computation
        if self._use_ghost_nodes and self._boundary_handler is not None:
            # Standard ghost mirroring
            ghost_to_mirror = {}
            for ghost_info in self._boundary_handler.ghost_node_map.values():
                ghost_to_mirror.update(ghost_info["ghost_to_mirror"])

            u_neighbors_list = []
            for idx in neighbor_indices:
                if idx < 0:
                    mirror_idx = ghost_to_mirror.get(int(idx))
                    u_neighbors_list.append(u_values[mirror_idx] if mirror_idx is not None else 0.0)
                else:
                    u_neighbors_list.append(u_values[int(idx)])
            u_neighbors = np.array(u_neighbors_list)
        else:
            u_neighbors = u_values[neighbor_indices]

        # Compute gradient: ∇u = Σ w_i * u_i for each component
        grad_u = grad_weights.T @ u_neighbors  # Shape: (dimension,)

        return grad_u

    def _build_differentiation_matrices(self) -> None:
        """
        Pre-compute sparse differentiation matrices for vectorized derivative computation.

        Builds:
        - D_grad: List of sparse matrices (n_points x n_points) for each gradient component
        - D_lap: Sparse matrix (n_points x n_points) for Laplacian

        After this, derivatives can be computed via matrix-vector multiplication:
            grad_u[d] = D_grad[d] @ u
            lap_u = D_lap @ u

        This converts O(n * k^2) per-point computation to O(n * k) matrix multiplication.
        """
        from scipy.sparse import lil_matrix

        n = self.n_points
        d = self.dimension

        # Initialize sparse matrices in LIL format (efficient for construction)
        D_grad_lil = [lil_matrix((n, n)) for _ in range(d)]
        D_lap_lil = lil_matrix((n, n))

        # Pre-compute LCR boundary points set for fast lookup
        lcr_boundary_set = set()
        if self._use_local_coordinate_rotation and self._boundary_handler is not None:
            lcr_boundary_set = set(self._boundary_handler.boundary_rotations.keys())

        for i in range(n):
            # For LCR boundary points, use our Taylor matrices with rotation
            if i in lcr_boundary_set:
                if self._neighborhood_builder is not None:
                    boundary_rotations = self._boundary_handler.boundary_rotations if self._boundary_handler else None
                    weights = self._neighborhood_builder.compute_derivative_weights_from_taylor(i, boundary_rotations)
                else:
                    # Legacy fallback
                    weights = self._compute_derivative_weights_from_taylor(i)
            else:
                # Issue #1427: route non-LCR points through the adaptive-aware builder so
                # D_lap/D_grad and self.neighborhoods share ONE source. On the default
                # adaptive_neighborhoods=False path the builder reuses the operator's SVD
                # verbatim, so this is byte-identical (verified: non-LCR grad/lap diff == 0.0).
                # When adaptive enlargement fires, the builder reflects the ENLARGED
                # neighborhood (self.neighborhoods[i]); the operator's pre-adaptive weights
                # would silently diverge from it. Fall back to the operator only when the
                # builder has no SVD Taylor data (QR-fallback stencils) — byte-identical there.
                if self._neighborhood_builder is not None:
                    weights = self._neighborhood_builder.compute_derivative_weights_from_taylor(i)
                    if weights is None:
                        weights = self._gfdm_operator.get_derivative_weights(i)
                else:
                    weights = self._gfdm_operator.get_derivative_weights(i)

            if weights is None:
                continue

            neighbor_indices = weights["neighbor_indices"]
            grad_weights = weights["grad_weights"]  # shape: (d, n_neighbors)
            lap_weights = weights["lap_weights"]  # shape: (n_neighbors,)

            # Override Laplacian weights with M-matrix QP precomputed monotone
            # weights if available. Two activation paths both populate
            # self._precomputed_stencils:
            #   (1) qp_m_matrix scheme + precompute application (legacy path)
            #   (2) joint_socp scheme — applies M-matrix QP at boundary buffer
            #       nodes where joint SOCP is infeasible (Phase 2 fallback per
            #       paper §831). The `joint_socp_stencils` override below takes
            #       priority for SOCP-feasible interior nodes.
            if self._precomputed_stencils is not None and self._precomputed_stencils.has_stencil(i):
                precomputed = self._precomputed_stencils.get_laplacian_weights(i)
                if precomputed is not None:
                    lap_weights = precomputed[0]  # (weights, neighbor_indices)

            # Override BOTH Laplacian and gradient weights with joint SOCP weights at
            # interior nodes where SOCP is feasible (audit-major Phase 1B). This takes
            # priority over qp_m_matrix precompute. Boundary buffer nodes where SOCP
            # is infeasible fall back to qp_m_matrix above (or default Wendland-Taylor
            # if qp_m_matrix precompute also doesn't have a stencil there).
            if self._joint_socp_stencils is not None and self._joint_socp_stencils.has_stencil(i):
                socp_weights = self._joint_socp_stencils.get_weights_dict(i)
                if socp_weights is not None:
                    # Single source of truth: consume the SOCP stencil's OWN
                    # neighbor_indices together with its (L, D). With
                    # SOCP-infeasibility-triggered adaptive enlargement (Issue
                    # #1106) the SOCP stencil may have MORE neighbors than the
                    # operator's base stencil, so the fill below must index the
                    # (possibly enlarged) SOCP set — not the operator's. When
                    # enlargement is off these are identical (verified), so this is
                    # byte-identical to the prior behaviour.
                    neighbor_indices = socp_weights["neighbor_indices"]
                    lap_weights = socp_weights["lap_weights"]
                    grad_weights = socp_weights["grad_weights"]
                    assert grad_weights.shape[1] == len(neighbor_indices) == len(lap_weights), (
                        f"joint_socp stencil weight/index length mismatch at point {i}: "
                        f"grad {grad_weights.shape}, lap {len(lap_weights)}, "
                        f"neighbors {len(neighbor_indices)} (Issue #1106 enlargement contract)"
                    )

            # Fill gradient matrices
            for dim in range(d):
                # Neighbor contributions (skip ghost particles with j < 0)
                real_grad_sum = 0.0
                for k, j in enumerate(neighbor_indices):
                    if j >= 0:
                        D_grad_lil[dim][i, j] = grad_weights[dim, k]
                        real_grad_sum += grad_weights[dim, k]
                # Center contribution (sum rule: center weight = -sum of REAL neighbor weights)
                # Note: Must exclude ghost particle weights to maintain row sum = 0
                center_weight = -real_grad_sum
                D_grad_lil[dim][i, i] += center_weight

            # Fill Laplacian matrix (same fix: exclude ghost weights from center)
            real_lap_sum = 0.0
            for k, j in enumerate(neighbor_indices):
                if j >= 0:
                    D_lap_lil[i, j] = lap_weights[k]
                    real_lap_sum += lap_weights[k]
            D_lap_lil[i, i] += -real_lap_sum

        # Convert to CSR format for efficient matrix-vector multiplication
        self._D_grad = [D.tocsr() for D in D_grad_lil]
        self._D_lap = D_lap_lil.tocsr()

    def _compute_derivatives_vectorized(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients and Laplacian for all points via sparse matrix multiplication.

        Args:
            u: Function values at collocation points, shape (n_points,)

        Returns:
            grad_u: Gradient at all points, shape (n_points, dimension)
            lap_u: Laplacian at all points, shape (n_points,)
        """
        # Lazy initialization of differentiation matrices (expensive computation)
        # self._D_grad initialized as None in __init__, computed on first use
        if self._D_grad is None:
            self._build_differentiation_matrices()

        grad_u = np.column_stack([D @ u for D in self._D_grad])
        lap_u = self._D_lap @ u

        # Note: LCR rotation is now applied in _compute_derivative_weights_from_taylor()
        # so gradients are already in the original coordinate frame

        return grad_u, lap_u

    def approximate_derivatives(self, u_values: np.ndarray, point_idx: int) -> dict[tuple[int, ...], float]:
        """
        Approximate derivatives at collocation point using weighted least squares.

        Args:
            u_values: Function values at collocation points
            point_idx: Index of the collocation point

        Returns:
            Dictionary mapping derivative multi-indices to approximated values
        """
        # Track current point for debugging/statistics
        self._current_point_idx = point_idx

        # Get QP level and check for ghost particles
        qp_level = getattr(self, "qp_optimization_level", "none")
        neighborhood = self.neighborhoods[point_idx]
        has_ghost = neighborhood.get("has_ghost", False)

        # Fast path: delegate to GFDMOperator when no ghost particles and no QP needed
        if qp_level == "none" and not has_ghost:
            return self._gfdm_operator.approximate_derivatives_at_point(u_values, point_idx)

        # Slow path: handle ghost particles and/or QP constraints
        if self.taylor_matrices[point_idx] is None:
            return {}

        taylor_data = self.taylor_matrices[point_idx]

        # Extract function values at neighborhood points, handling ghost nodes/particles
        neighbor_indices = neighborhood["indices"]
        u_center = u_values[point_idx]

        # Use ghost-aware value retrieval if ghost nodes method is active
        if self._use_ghost_nodes:
            if self._boundary_handler is not None:
                u_neighbors = self._boundary_handler.get_values_with_ghosts(
                    u_values, neighbor_indices, point_idx=point_idx
                )
            else:
                # Legacy fallback
                u_neighbors = self._get_values_with_ghosts(u_values, neighbor_indices, point_idx=point_idx)
        else:
            # Handle legacy ghost particles based on BC type
            # - Neumann/no-flux: u_ghost = u_center (mirror value)
            # - Dirichlet: u_ghost = BC value (if available)
            bc_type = self._get_boundary_condition_property("type")
            bc_values = self._get_boundary_condition_property("values")

            u_neighbors = []
            for idx in neighbor_indices:  # type: ignore[attr-defined]
                if idx >= 0:
                    # Regular neighbor
                    u_neighbors.append(u_values[idx])
                else:
                    # Legacy ghost particle: value depends on BC type
                    if bc_type == "dirichlet" and bc_values is not None:
                        # Dirichlet BC: use prescribed value
                        # Note: bc_values may be scalar, array, or callable
                        if callable(bc_values):
                            x_pos = self.collocation_points[point_idx]
                            u_neighbors.append(bc_values(x_pos))
                        elif isinstance(bc_values, (list, tuple, np.ndarray)):
                            # Array-like: use value at this point
                            u_neighbors.append(bc_values[point_idx] if point_idx < len(bc_values) else 0.0)
                        else:
                            # Scalar
                            u_neighbors.append(float(bc_values))
                    else:
                        # Neumann/no-flux: mirror u_center
                        u_neighbors.append(u_center)

        u_neighbors = np.array(u_neighbors)  # type: ignore[assignment]

        # Right-hand side: u(x_neighbor) - u(x_center) for Taylor expansion
        # u(x_j) - u(x_0) ≈ ∇u·(x_j - x_0) where A matrix uses (x_j - x_0)
        # For ghost particles: u_ghost = u_center → b = 0, enforcing ∂u/∂n = 0
        b = u_neighbors - u_center

        if qp_level == "always":
            # "always" level: Force QP at every point without checking M-matrix
            derivative_coeffs = self._monotonicity_enforcer.solve_constrained_qp(taylor_data, b, point_idx)  # type: ignore[union-attr]
        elif qp_level == "auto":
            # "auto" level: Adaptive QP with M-matrix checking
            # First try unconstrained solution to check if constraints are needed
            unconstrained_coeffs = self._monotonicity_enforcer._solve_unconstrained_fallback(taylor_data, b)  # type: ignore[union-attr]

            # Check if unconstrained solution violates monotonicity (M-matrix property)
            self.qp_stats["points_checked"] += 1
            needs_constraints = self._monotonicity_enforcer.check_monotonicity_violation(
                unconstrained_coeffs, point_idx
            )  # type: ignore[union-attr]

            if needs_constraints:
                # Apply constrained QP to enforce monotonicity
                self.qp_stats["violations_detected"] += 1
                self.qp_stats["violation_point_indices"].add(point_idx)
                derivative_coeffs = self._monotonicity_enforcer.solve_constrained_qp(taylor_data, b, point_idx)  # type: ignore[union-attr]
            else:
                # Use faster unconstrained solution
                derivative_coeffs = unconstrained_coeffs
        elif taylor_data.get("use_svd", False):  # type: ignore[attr-defined]
            # Use SVD: solve using pseudoinverse with truncated SVD
            sqrt_W = taylor_data["sqrt_W"]
            U = taylor_data["U"]
            S = taylor_data["S"]
            Vt = taylor_data["Vt"]

            # Compute sqrt(W) @ b
            Wb = sqrt_W @ b

            # SVD solution: x = V @ S^{-1} @ U^T @ Wb
            UT_Wb = U.T @ Wb
            S_inv_UT_Wb = UT_Wb / S  # Element-wise division
            derivative_coeffs = Vt.T @ S_inv_UT_Wb

        elif taylor_data.get("use_qr", False):  # type: ignore[attr-defined]
            # Use QR decomposition: solve R @ x = Q^T @ sqrt(W) @ b
            sqrt_W = taylor_data["sqrt_W"]
            Q = taylor_data["Q"]
            R = taylor_data["R"]

            Wb = sqrt_W @ b
            QT_Wb = Q.T @ Wb

            try:
                derivative_coeffs = np.linalg.solve(R, QT_Wb)
            except np.linalg.LinAlgError:
                # Fallback to least squares if R is singular
                A_matrix = taylor_data.get("A")  # type: ignore[attr-defined]
                if A_matrix is not None:
                    lstsq_result = lstsq(A_matrix, b)
                    derivative_coeffs = lstsq_result[0] if lstsq_result is not None else np.zeros(len(b))
                else:
                    derivative_coeffs = np.zeros(len(b))

        else:
            # Final fallback to direct least squares on A. Reached when
            # SVD and QR both failed in NeighborhoodBuilder (#1125 removed
            # the legacy `AtWA_inv` normal-equations branch — see issue
            # for why pseudo-inverse via lstsq is strictly better than
            # inv() on the squared-condition normal-equations matrix).
            A_matrix = taylor_data.get("A")  # type: ignore[attr-defined]
            if A_matrix is not None:
                lstsq_result = lstsq(A_matrix, b)
                derivative_coeffs = lstsq_result[0] if lstsq_result is not None else np.zeros(len(b))
            else:
                derivative_coeffs = np.zeros(len(b))

        # Handle case where coefficient computation failed
        if derivative_coeffs is None:
            derivative_coeffs = np.zeros(len(self.multi_indices))

        # Map coefficients to multi-indices
        derivatives = {}
        for k, beta in enumerate(self.multi_indices):
            derivatives[beta] = derivative_coeffs[k]

        # Apply inverse rotation for LCR boundary points (Issue #531)
        # Derivatives were computed in rotated frame, need to rotate back
        if (
            self._use_local_coordinate_rotation
            and self._boundary_handler is not None
            and point_idx in self._boundary_handler.boundary_rotations
        ):
            derivatives = self._boundary_handler.rotate_derivatives_back(
                derivatives, self._boundary_handler.boundary_rotations[point_idx]
            )

        # Consistency override: when precomputed monotone weights exist for this
        # point, override the corresponding derivative entries so the per-point
        # HJB Newton residual uses the SAME stencil weights as the Jacobian
        # (which is assembled from `_cached_derivative_weights`, populated with
        # SOCP / M-matrix-QP weights at __init__).
        #
        # Without this override, the slow path above computes derivatives via
        # bare Wendland-Taylor LSQ (`taylor_data["U"]/["S"]/["Vt"]` etc.),
        # while the Jacobian uses SOCP-corrected weights. Newton then solves
        #     J · δu = -r
        # with J and r assembled from inconsistent stencil weights, converging
        # to a stationary point of the mongrel system rather than the true
        # discrete-HJB fixed point. Empirically: 12× u_err discrepancy in the
        # exp08 step 4 2D Towel-on-Beach validation at N=100 (joint_socp
        # u_err iter 1 = 48.81 without this fix vs 5.38 with the fix, 9× match
        # to the qp_m_matrix control on the same setup).
        #
        # Precedence (matches `_build_derivative_matrices` / `_cached_derivative_weights`):
        #   joint SOCP at SOCP-feasible interior > M-matrix QP at boundary > bare W-T.
        if self._joint_socp_stencils is not None and self._joint_socp_stencils.has_stencil(point_idx):
            socp = self._joint_socp_stencils.get_weights_dict(point_idx)
            if socp is not None:
                L_w = socp["lap_weights"]  # shape (n_socp_neighbors,)
                D_w = socp["grad_weights"]  # shape (d, n_socp_neighbors)
                # Rebuild b on the SOCP stencil's OWN neighbor_indices. With
                # SOCP-infeasibility-triggered adaptive enlargement (Issue #1106)
                # the SOCP stencil can have MORE neighbors than the runtime
                # `neighborhood["indices"]` that the outer `b` was built on;
                # contracting `D_w @ b` / `L_w @ b` on the mismatched-length outer
                # `b` would raise a matmul size error (the #1102 / G-013 pattern,
                # already handled for the M-matrix elif via `b_precomp`). SOCP
                # applies to interior cloud points only, so these indices are pure
                # cloud indices (no ghosts). When enlargement is off the SOCP
                # neighbor_indices equal the runtime ones, so `b_socp == b`
                # (byte-identical).
                socp_nbr = socp["neighbor_indices"]
                assert D_w.shape[1] == len(socp_nbr) == len(L_w), (
                    f"joint_socp stencil weight/index length mismatch at point {point_idx}: "
                    f"grad {D_w.shape}, lap {len(L_w)}, neighbors {len(socp_nbr)} "
                    f"(Issue #1106 enlargement contract)"
                )
                b_socp = u_values[socp_nbr] - u_center
                # Override gradient: ∂u/∂x_d (i) = sum_j D_w[d, j] * b_j
                for d in range(self.dimension):
                    beta = tuple(1 if k == d else 0 for k in range(self.dimension))
                    derivatives[beta] = float(D_w[d] @ b_socp)
                # Override Laplacian sum (= trace of Hessian) via diagonal split.
                # Preserves bare-WT off-diagonal Hessian entries (e.g. (1,1) in 2D)
                # while enforcing target_lap = L_w · b on the trace.
                target_lap = float(L_w @ b_socp)
                current_lap = sum(
                    float(derivatives.get(beta, 0.0))
                    for beta in self.multi_indices
                    if len(beta) == self.dimension and sum(beta) == 2 and max(beta) == 2
                )
                adjustment = (target_lap - current_lap) / self.dimension
                for d in range(self.dimension):
                    beta = tuple(2 if k == d else 0 for k in range(self.dimension))
                    if beta in derivatives:
                        derivatives[beta] = float(derivatives[beta]) + adjustment
        elif self._precomputed_stencils is not None and self._precomputed_stencils.has_stencil(point_idx):
            precomputed = self._precomputed_stencils.get_laplacian_weights(point_idx)
            if precomputed is not None:
                L_w = precomputed[0]  # shape (n_precomp_neighbors,)
                precomp_nbr = precomputed[1]
                # Rebuild b on the PRECOMP stencil. When runtime
                # ``self.neighborhoods[i]["indices"]`` has been augmented
                # (ghost-node reflection points, or adaptive δ-enlargement),
                # its length differs from the precomp stencil that L_w was
                # built on. Contracting ``L_w @ b`` of mismatched length
                # produced ``ValueError: matmul: size N is different from
                # K`` (Issue #1102, G-013 pattern). The fix re-evaluates
                # ``b = u_neighbors - u_center`` on precomp's stored
                # ``neighbor_indices`` (cloud-only, no ghosts), aligning
                # with L_w's stencil source.
                b_precomp = u_values[precomp_nbr] - u_center
                # M-matrix QP only corrects the Laplacian; gradient stays bare W-T.
                target_lap = float(L_w @ b_precomp)
                current_lap = sum(
                    float(derivatives.get(beta, 0.0))
                    for beta in self.multi_indices
                    if len(beta) == self.dimension and sum(beta) == 2 and max(beta) == 2
                )
                adjustment = (target_lap - current_lap) / self.dimension
                for d in range(self.dimension):
                    beta = tuple(2 if k == d else 0 for k in range(self.dimension))
                    if beta in derivatives:
                        derivatives[beta] = float(derivatives[beta]) + adjustment

        return derivatives

    def compute_all_derivatives(
        self, u: np.ndarray, use_qp: bool | None = None
    ) -> dict[int, dict[tuple[int, ...], float]]:
        """
        Compute derivatives at all collocation points using precomputed Taylor matrices.

        When to use this vs GFDMOperator:
        - Use this method when you need QP constraints for monotonicity (M-matrix)
        - Use GFDMOperator for general GFDM needs (FP solver, one-off computations)

        Example:
            # For QP-constrained derivatives (HJB specific):
            solver = HJBGFDMSolver(problem, points, monotonicity_scheme="auto")
            derivs = solver.compute_all_derivatives(u, use_qp=True)

            # For general GFDM (simpler, no QP):
            from mfgarchon.utils.numerical import GFDMOperator
            gfdm = GFDMOperator(points, delta=0.1)
            grad = gfdm.gradient(u)
            lap = gfdm.laplacian(u)

        Args:
            u: Function values at collocation points, shape (n_points,)
            use_qp: Override QP constraint behavior for this call.
                None: Use solver's qp_optimization_level setting
                True: Force QP constraints at all points
                False: Disable QP constraints for this call

        Returns:
            Dictionary mapping point index to derivative dictionary.
            derivatives[i] = {(1,): du/dx, (2,): d²u/dx², ...} for 1D
            derivatives[i] = {(1,0): du/dx, (0,1): du/dy, (2,0): d²u/dx², ...} for 2D
        """
        # Optionally override QP level for this computation
        saved_qp_level = None
        if use_qp is not None:
            saved_qp_level = self.qp_optimization_level
            self.qp_optimization_level = "always" if use_qp else "none"

        try:
            all_derivatives: dict[int, dict[tuple[int, ...], float]] = {}
            for i in range(self.n_points):
                all_derivatives[i] = self.approximate_derivatives(u, i)
            return all_derivatives
        finally:
            # Restore QP level if overridden
            if saved_qp_level is not None:
                self.qp_optimization_level = saved_qp_level

    # Note: QP methods moved to MonotonicityEnforcer component (Issue #545)
    # - solve_constrained_qp() (was _solve_monotone_constrained_qp)
    # - _solve_unconstrained_fallback()
    # - check_monotonicity_violation() (was _check_monotonicity_violation)
    # - check_m_matrix() (was _check_m_matrix_property)
    # - print_diagnostics() (was print_qp_diagnostics)
    # - compute_fd_weights_from_taylor() (was _compute_fd_weights_from_taylor)

    def _approximate_all_derivatives_cached(self, u: np.ndarray) -> dict[int, dict[tuple[int, ...], float]]:
        """Compute all derivatives at once (for caching between residual/Jacobian)."""
        all_derivs: dict[int, dict[tuple[int, ...], float]] = {}
        for i in range(self.n_points):
            all_derivs[i] = self.approximate_derivatives(u, i)
        return all_derivs

    def _compute_hjb_residual_hamiltonian(
        self,
        u_current: np.ndarray,
        u_n_plus_1: np.ndarray,
        m_n_plus_1: np.ndarray,
        grad_u: np.ndarray,
        lap_u: np.ndarray,
        H_class: Any,
        current_time: float,
        additive_source: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute HJB residual using batch Hamiltonian class (Issue #775).

        Uses H_class(x, m, p, t) for vectorized evaluation over all collocation
        points. Works with any HamiltonianBase subclass.

        Args:
            u_current: Current solution at collocation points, shape (n_points,)
            u_n_plus_1: Solution at next time step, shape (n_points,)
            m_n_plus_1: Density at collocation points, shape (n_points,)
            grad_u: Pre-computed gradient, shape (n_points, dimension)
            lap_u: Pre-computed Laplacian, shape (n_points,)
            H_class: HamiltonianBase instance with batch-polymorphic __call__
            current_time: Current time value for H(x, m, p, t)

        Returns:
            Residual vector, shape (n_points,)
        """
        dt = self.problem.T / self.problem.Nt
        u_t = (u_n_plus_1 - u_current) / dt
        # _get_sigma_value returns σ (not D); the harness applies D = σ²/2 (#1073/#811).
        # Issue #1059/#1071 phase 7: a per-node LLF σ_eff field flows through the single-source
        # assemble_hjb_residual (now field-σ capable, D_i = σ_eff_i²/2 elementwise); a plain solve
        # uses the scalar σ. One assembly path either way.
        sigma = self._sigma_for_assembly()
        return assemble_hjb_residual(
            H_class=H_class,
            x=self.collocation_points,
            m=m_n_plus_1,
            p=grad_u,
            lap_u=lap_u,
            sigma=sigma,
            t=current_time,
            u_t=u_t,
            additive_source=additive_source,
        )

    def _compute_hjb_jacobian_hamiltonian(
        self,
        grad_u: np.ndarray,
        m_n_plus_1: np.ndarray,
        H_class: Any,
        current_time: float,
    ):
        """
        Compute sparse Jacobian using batch H.dp() (Issue #775).

        Uses H_class.dp(x, m, p, t) for vectorized dH/dp computation.
        Jacobian structure: J = (1/dt)I + sum_d diag(dH/dp_d) @ D_grad[d] - (sigma^2/2) D_lap

        Args:
            grad_u: Pre-computed gradient, shape (n_points, dimension)
            m_n_plus_1: Density at collocation points, shape (n_points,)
            H_class: HamiltonianBase instance with batch-polymorphic dp()
            current_time: Current time value for H.dp(x, m, p, t)

        Returns:
            Sparse Jacobian matrix in CSR format
        """
        # Lazy initialization of differentiation matrices
        if self._D_grad is None:
            self._build_differentiation_matrices()

        dt = self.problem.T / self.problem.Nt
        # _get_sigma_value returns σ (not D); the harness applies D = σ²/2 (#1073/#811).
        # Issue #1059/#1071 phase 7: a per-node LLF σ_eff field flows through the single-source
        # assemble_hjb_jacobian_diag (now field-σ capable: it row-scales the Laplacian via
        # diags(D_i) @ D_lap); a plain solve uses the scalar σ. One assembly path either way.
        sigma = self._sigma_for_assembly()
        return assemble_hjb_jacobian_diag(
            H_class=H_class,
            x=self.collocation_points,
            m=m_n_plus_1,
            p=grad_u,
            sigma=sigma,
            t=current_time,
            dt=dt,
            D_grad=self._D_grad,
            D_lap=self._D_lap,
        )

    def _compute_hjb_residual_with_cache(
        self,
        u_current: np.ndarray,
        u_n_plus_1: np.ndarray,
        m_n_plus_1: np.ndarray,
        time_idx: int,
        cached_derivs: dict[int, dict[tuple[int, ...], float]],
        additive_source: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute HJB residual using pre-computed derivatives (per-point path).

        Per-point path via problem.H(). The potential V(x) comes from the
        Hamiltonian class (e.g., SeparableHamiltonian._potential), NOT from
        problem.f_potential. Active for custom problems (is_custom=True) or
        when QP monotonicity is enabled. See Issue #766.
        """
        from mfgarchon.core.derivatives import from_multi_index_dict

        residual = np.zeros(self.n_points)
        dimension = self.problem.dimension
        dt = self.problem.T / self.problem.Nt
        u_t = (u_n_plus_1 - u_current) / dt

        for i in range(self.n_points):
            x_pos = self.collocation_points[i]
            derivs = cached_derivs[i]
            p_derivs = from_multi_index_dict(derivs, dimension=dimension)
            laplacian = p_derivs.laplacian or 0.0

            H = self.problem.H(i, m_n_plus_1[i], derivs=p_derivs, x_position=x_pos)

            # Additive source at this timestep (passed explicitly from the backward loop)
            if additive_source is not None:
                H = H + additive_source[i]

            sigma_val = self._get_sigma_value(i)
            diffusion_term = diffusion_from_volatility(sigma_val) * laplacian
            residual[i] = -u_t[i] + H - diffusion_term

        return residual

    def assemble_hjb_iteration_matrix(self, grad_u: np.ndarray):
        """Assemble the sparse HJB Newton iteration matrix at a given gradient (Issue #1414).

        Returns ``(1/dt)I + Σ_d diag(∂H/∂p_d) @ D_grad[d] - D·D_lap`` (with ``D = σ²/2``,
        per-node for the LLF ``σ_eff`` field) via the single-source assembler
        ``assemble_hjb_jacobian_diag`` → ``H_class.evaluate_dp`` (no inline ``p/λ``
        re-derivation; Issue #1071/#1408). For the separable LQ Hamiltonian ``∂H/∂p = p/λ``
        is independent of ``m`` and ``t``, so ``m=0`` / ``t=0`` reproduce the production
        Newton Jacobian exactly.

        Dedicated entry point for the Issue #1074 M-matrix / discrete-maximum-principle tests,
        which assemble the iteration matrix directly to verify its M-matrix structure. It has
        no production callers — the backward solve calls :meth:`_compute_hjb_jacobian_hamiltonian`
        directly with the live ``m`` / ``t`` (this replaces the former private
        ``_compute_hjb_jacobian_vectorized`` alias).

        Args:
            grad_u: Pre-computed gradient, shape (n_points, dimension)

        Returns:
            Sparse Jacobian (iteration matrix) in CSR format
        """
        H_class = getattr(self.problem, "hamiltonian_class", None)
        if H_class is None:
            raise ValueError(
                "assemble_hjb_iteration_matrix requires problem.hamiltonian_class to derive "
                "∂H/∂p via the single-source assembler (Issue #1071)."
            )
        # Separable LQ: ∂H/∂p = p/λ ignores m and t, so m=0 / t=0 are byte-identical.
        return self._compute_hjb_jacobian_hamiltonian(grad_u, np.zeros(self.n_points), H_class, 0.0)

    def _maybe_warn_dmp(self, u_current: np.ndarray) -> None:
        """Warn (once) when the current drift exceeds the assembled-M-matrix threshold (Issue #1074).

        Per-stencil joint_socp feasibility does NOT imply the assembled HJB iteration matrix
        ``I/dt - D·L + α·D_grad`` is an M-matrix: the signed drift term flips an off-diagonal
        positive once ``|α| > α_crit = D·min_edge(L_ij/‖D_grad_ij‖)`` (see
        :func:`critical_drift_for_dmp`). So the discrete maximum principle holds only in the
        diffusion-dominated regime. This is a *diagnostic* warning (numerically inert); active only
        when ``check_dmp=True`` and the scheme is ``joint_socp``. The drift is computed from the
        current iterate via the assembled gradient operator, consistent with the Jacobian.
        """
        if not self.check_dmp or self._dmp_warned or self.monotonicity_scheme != "joint_socp":
            return
        # Issue #1253 2026-06-10 audit: _D_grad/_D_lap are built lazily and are
        # None on the joint_socp per-point Newton path (_compute_hjb_jacobian_sparse
        # never calls _build_differentiation_matrices). Build them here so the guard
        # actually runs on the real solve path, not just when the vectorized Jacobian
        # path happened to pre-build them as a side effect.
        if self._D_grad is None or self._D_lap is None:
            self._build_differentiation_matrices()
        if self._dmp_alpha_crit is None:
            from mfgarchon.alg.numerical.gfdm_components.monotonicity_enforcer import critical_drift_for_dmp

            sigma = self._sigma_for_assembly()
            diffusion_coeff = (
                diffusion_from_volatility(sigma, kind="field")
                if isinstance(sigma, np.ndarray)
                else diffusion_from_volatility(sigma)
            )
            self._dmp_alpha_crit = critical_drift_for_dmp(
                self._D_lap, self._D_grad, diffusion_coeff, interior_indices=self.interior_indices
            )
        lambda_val = self._control_cost_lambda()
        grad = np.stack([self._D_grad[d] @ u_current for d in range(self.dimension)], axis=1)
        max_alpha = float(np.max(np.linalg.norm(grad, axis=1)) / lambda_val)
        if max_alpha > self._dmp_alpha_crit:
            logger.warning(
                "DMP not guaranteed (joint_socp, Issue #1074): max|drift| = %.3g exceeds the "
                "assembled-M-matrix threshold alpha_crit = %.3g. Per-stencil SOCP feasibility does "
                "not imply an assembled M-matrix once |drift| > alpha_crit (advection-dominated); the "
                "discrete maximum principle holds only for |drift| <= alpha_crit.",
                max_alpha,
                self._dmp_alpha_crit,
            )
            self._dmp_warned = True

    def _compute_hjb_jacobian_sparse(
        self,
        u_current: np.ndarray,
        m_n_plus_1: np.ndarray,
        time_idx: int,
        cached_derivs: dict[int, dict[tuple[int, ...], float]],
    ):
        """Compute sparse Jacobian using pre-computed derivatives and GFDM weights."""
        from scipy.sparse import lil_matrix

        from mfgarchon.core.derivatives import from_multi_index_dict, to_multi_index_dict

        n = self.n_points
        d = self.problem.dimension
        dt = self.problem.T / self.problem.Nt

        # Lazy initialization: Pre-cache all derivative weights (expensive computation)
        # self._cached_derivative_weights initialized as None in __init__
        #
        # Override precedence (joint SOCP > M-matrix QP > default operator weights):
        # this keeps the per-point HJB Jacobian consistent with the differentiation
        # matrices `_D_lap` / `_D_grad` (which apply the same precedence in
        # `_build_derivative_matrices`). v3 research code achieved this by monkey-
        # patching `_gfdm_operator.get_derivative_weights`; the precomputed-stencil
        # path replaces that hack with explicit dispatch here.
        if self._cached_derivative_weights is None:
            self._cached_derivative_weights = [self._gfdm_operator.get_derivative_weights(i) for i in range(n)]
            if self._joint_socp_stencils is not None:
                for i in range(n):
                    if self._joint_socp_stencils.has_stencil(i):
                        socp_w = self._joint_socp_stencils.get_weights_dict(i)
                        if socp_w is not None:
                            self._cached_derivative_weights[i] = socp_w
            if self._precomputed_stencils is not None:
                for i in range(n):
                    if self._precomputed_stencils.has_stencil(i):
                        pre = self._precomputed_stencils.get_laplacian_weights(i)
                        base = self._cached_derivative_weights[i]
                        if pre is not None and base is not None:
                            base["lap_weights"] = pre[0]

        # Use LIL format for efficient construction
        jacobian = lil_matrix((n, n))

        degenerate_rows: list[int] = []
        for i in range(n):
            weights = self._cached_derivative_weights[i]
            if weights is None:
                # Issue #1071: degenerate stencil (singular Taylor matrix) at point i. Use an
                # identity Jacobian row so Newton can continue, but record it — the degenerate
                # stencil must not be masked silently (aggregated warning emitted after the loop).
                jacobian[i, i] = 1.0 / dt
                degenerate_rows.append(i)
                continue

            neighbor_indices = weights["neighbor_indices"]
            grad_weights = weights["grad_weights"]
            lap_weights = weights["lap_weights"]

            p_derivs = from_multi_index_dict(cached_derivs[i], dimension=d)

            dH_dp = self.problem.dH_dp(
                x_idx=i,
                m_at_x=m_n_plus_1[i],
                derivs=to_multi_index_dict(p_derivs),
                t_idx=time_idx,
                x_position=self.collocation_points[i],  # Pass actual position for GFDM
            )
            if dH_dp is None:
                dH_dp = self._compute_dH_dp_fd(i, m_n_plus_1[i], p_derivs, time_idx)

            sigma_val = self._get_sigma_value(i)
            diffusion_coeff = diffusion_from_volatility(sigma_val)

            # Neighbor contributions
            for k, j in enumerate(neighbor_indices):
                if j < 0:
                    continue  # Skip ghost particles
                val = np.dot(dH_dp, grad_weights[:, k]) - diffusion_coeff * lap_weights[k]
                jacobian[i, j] = val

            # Center point contribution
            center_grad_weight = -np.sum(grad_weights, axis=1)
            center_lap_weight = -np.sum(lap_weights)
            jacobian[i, i] += np.dot(dH_dp, center_grad_weight) - diffusion_coeff * center_lap_weight
            jacobian[i, i] += 1.0 / dt  # Time derivative

        if degenerate_rows:
            logger.warning(
                "HJBGFDMSolver: %d collocation point(s) have a degenerate stencil "
                "(singular Taylor matrix); used an identity Jacobian row there, which degrades "
                "Newton convergence at those points. Indices: %s%s",
                len(degenerate_rows),
                degenerate_rows[:10],
                "..." if len(degenerate_rows) > 10 else "",
            )
        return jacobian.tocsr()

    _BC_STR_TO_ENUM: ClassVar[dict[str, BCType]] = {
        "dirichlet": BCType.DIRICHLET,
        "neumann": BCType.NEUMANN,
        "no_flux": BCType.NO_FLUX,
        "periodic": BCType.PERIODIC,
        "robin": BCType.ROBIN,
    }

    def _classify_boundary_point(self, i: int, local_idx: int, use_per_point_bc: bool, global_bc_type, legacy_normals):
        """Resolve ``(bc_enum, segment, normal)`` for boundary point ``i``.

        Mixed BC uses the pre-classified per-point maps; uniform BC uses the global type +
        computed outward normal. Single source shared by the Newton residual-BC path and the
        Howard value-form path (Issue #1118 PR2), so the two never drift in classification.
        """
        if use_per_point_bc:
            segment = self._bc_segment_per_point[i]
            return segment.bc_type, segment, self._bc_normal_per_point[i]
        bc_str = (global_bc_type or "neumann").lower()
        if bc_str not in self._BC_STR_TO_ENUM:
            raise ValueError(
                f"Unknown BC type {bc_str!r} at boundary point {i} (uniform path). "
                f"Supported: {tuple(self._BC_STR_TO_ENUM)}."
            )
        bc_enum = self._BC_STR_TO_ENUM[bc_str]
        if legacy_normals is not None and local_idx < len(legacy_normals):
            normal = legacy_normals[local_idx]
        elif self._boundary_handler is not None:
            normal = self._boundary_handler.compute_outward_normal(i)
        else:
            normal = self._compute_outward_normal(i)
        return bc_enum, None, normal

    def _bc_row_for_point(
        self,
        i: int,
        bc_enum,
        segment,
        normal,
        dimension: int,
        n: int,
        legacy_bc_values,
        current_time: float,
    ) -> tuple[np.ndarray, float]:
        """Build the value-form BC row (coefficients + target) for boundary point ``i``.

        Returns ``(row_coeffs, bc_target)`` WITHOUT any residual subtraction, so it is the
        single coefficient source shared by both BC application paths (Issue #1118 PR2):
        - the Newton path forms the #1116 residual RHS ``row_coeffs @ u_current - bc_target``;
        - the Howard value-form path uses ``bc_target`` as the RHS directly.
        Keeping one source prevents the two paths from drifting (the recurring dual-source BC
        bug class). Raises for BC types with no row builder (PERIODIC/ROBIN).
        """
        match bc_enum:
            case BCType.DIRICHLET:
                new_row = np.zeros(n)
                new_row[i] = 1.0
                bc_target = self._eval_bc_dirichlet_value(i, segment, legacy_bc_values, current_time)
            case BCType.NEUMANN | BCType.NO_FLUX:
                new_row, bc_target = self._build_neumann_bc_row(
                    i, normal, dimension, segment, legacy_bc_values, current_time
                )
            case BCType.PERIODIC:
                raise NotImplementedError(
                    f"PERIODIC BC at boundary point {i} not supported by HJBGFDMSolver "
                    f"via row replacement. Use TensorProductGrid + FDM for periodic "
                    f"geometries, or rephrase as paired Dirichlet/Neumann segments."
                )
            case BCType.ROBIN:
                # Issue #1118 PR2b: the adjoint-consistent BC is ROBIN(alpha=0, beta=1),
                # whose equation beta*(n.grad u) = g reduces to n.grad u = g — exactly the
                # Neumann normal-derivative row with RHS = the resolved scalar g (segment.value
                # is a plain float after with_resolved_providers). Delegate to the SAME builder
                # the NEUMANN/NO_FLUX arm uses: single coefficient source, no second ROBIN
                # stencil (the dual-source BC bug class). General Robin (alpha != 0) adds an
                # alpha*u term the normal-derivative row cannot represent, and beta != 1 would
                # need a 1/beta scaling the builder does not apply — both are reachable via the
                # BCSegment API and would be silently mis-solved, so fail loud instead.
                alpha = getattr(segment, "alpha", 1.0) if segment is not None else 1.0
                beta = getattr(segment, "beta", 0.0) if segment is not None else 0.0
                if abs(alpha) > 0.0:
                    raise NotImplementedError(
                        f"ROBIN BC with alpha={alpha!r} at boundary point {i} is not supported "
                        f"by HJBGFDMSolver: the normal-derivative row encodes only "
                        f"beta*(n.grad u) = g and cannot represent the alpha*u term. Only the "
                        f"adjoint-consistent ROBIN(alpha=0, beta=1) case is supported "
                        f"(Issue #1118 PR2b; see AdjointConsistentProvider)."
                    )
                if abs(beta - 1.0) > 0.0:
                    raise NotImplementedError(
                        f"ROBIN BC with beta={beta!r} at boundary point {i} is not supported: "
                        f"the delegated Neumann row assumes coefficient 1 on n.grad u and does "
                        f"not apply a 1/beta scaling. Only beta=1 (the adjoint-consistent case) "
                        f"is supported (Issue #1118 PR2b)."
                    )
                new_row, bc_target = self._build_neumann_bc_row(
                    i, normal, dimension, segment, legacy_bc_values, current_time
                )
            case _:
                raise ValueError(
                    f"Unhandled BCType {bc_enum!r} at boundary point {i}. "
                    f"This indicates a new BCType value not yet wired into HJBGFDMSolver "
                    f"BC row construction."
                )
        return new_row, float(bc_target)

    def _value_form_bc_rows(self, time_idx: int) -> dict[int, tuple[np.ndarray, float]]:
        """Value-form boundary rows ``{i: (row_coeffs, bc_target)}`` for the Howard inner
        solver (Issue #1118 PR2).

        Uses the SAME per-point classifier (`_classify_boundary_point`) and coefficient
        source (`_bc_row_for_point`) as the Newton residual-BC path, over the solver's
        official `self.boundary_indices`, so the two paths never drift. Howard applies these
        directly (``A[i,:] = row``, ``b[i] = bc_target``); the Newton path forms its #1116
        residual RHS from the same rows instead. Ghost-node-enforced Neumann points are
        omitted (their PDE row stands) — only relevant when ``_use_ghost_nodes`` is on, which
        the joint_socp Howard path does not use.
        """
        try:
            use_per_point_bc = self.boundary_conditions.is_mixed
        except AttributeError:
            use_per_point_bc = False
        global_bc_type = self._get_boundary_condition_property("type") if not use_per_point_bc else None
        legacy_bc_values = self._get_boundary_condition_property("values") if not use_per_point_bc else None
        legacy_normals = self._bc_config.get("normals", None) if not use_per_point_bc and self._bc_config else None
        dimension = self.dimension
        n = self.n_points
        current_time = time_idx * (self.problem.T / self.problem.Nt) if getattr(self.problem, "Nt", 0) > 0 else 0.0

        rows: dict[int, tuple[np.ndarray, float]] = {}
        for local_idx, i in enumerate(self.boundary_indices):
            i = int(i)
            bc_enum, segment, normal = self._classify_boundary_point(
                i, local_idx, use_per_point_bc, global_bc_type, legacy_normals
            )
            if (
                bc_enum in (BCType.NEUMANN, BCType.NO_FLUX)
                and self._use_ghost_nodes
                and self._boundary_handler is not None
                and i in self._boundary_handler.ghost_node_map
            ):
                continue  # symmetric ghost stencils enforce the BC structurally; PDE row stands
            row, target = self._bc_row_for_point(
                i, bc_enum, segment, normal, dimension, n, legacy_bc_values, current_time
            )
            rows[i] = (row, target)
        return rows

    def _apply_boundary_conditions_to_sparse_system(
        self,
        jacobian_sparse,
        residual: np.ndarray,
        time_idx: int,
        u_current: np.ndarray,
    ):
        """Apply boundary conditions to the sparse Jacobian via row replacement.

        Two dispatch paths:

        - **Mixed BC (per-point)**: every boundary point has been pre-classified
          at solver __init__ time to a (BCSegment, outward_normal) pair stored in
          ``self._bc_segment_per_point`` and ``self._bc_normal_per_point``. This
          method consumes those maps for O(1) lookup. If a point is missing from
          the map (only possible if pre-classification raised), we get a KeyError
          rather than a silent zero-row.

        - **Uniform BC**: legacy fast path using ``_bc_config["type"]`` and
          ``_bc_config["normals"]`` arrays indexed by ``local_idx``.

        BC type dispatch is now an exhaustive ``match`` over ``BCType`` with
        ``case _: raise``, so any unhandled enum value surfaces immediately
        rather than silently leaving a cleared zero row. PERIODIC and ROBIN
        raise ``NotImplementedError`` because this solver doesn't support them
        via row replacement (see error messages for alternatives).

        Row construction is **atomic**: we build the full replacement row in a
        local ``np.ndarray`` and assign in one shot, instead of clearing first
        and then conditionally refilling.

        Newton residual semantics (Issue #1116): the BC row's RHS encodes the
        **current violation** ``F_bc(u_current) - target``, not the BC target
        value alone. The pre-#1116 code used the target value, which made the
        boundary half of ``(J, r)`` structurally inconsistent: J rows were the
        Jacobian of ``F_bc``, but r rows were a constant in u, so ``J·δ = -r``
        did not describe a Newton step on the same nonlinear function. The
        pathology was masked for pure Dirichlet ``bc=0`` (post-step projection
        rescues the iterate) and for Neumann ``bc=0`` with ``w·u_init=0``;
        it surfaced on mixed-BC + non-trivial initial state (Stage C v3).
        See ``docs/bug_reports/2026_05_hjb_bc_newton_mismatch.md``.
        """
        if len(self.boundary_indices) == 0:
            return jacobian_sparse, residual

        jac_lil = jacobian_sparse.tolil()
        residual_bc = residual.copy()

        try:
            use_per_point_bc = self.boundary_conditions.is_mixed
        except AttributeError:
            use_per_point_bc = False

        # Legacy uniform-BC scaffold
        global_bc_type = self._get_boundary_condition_property("type") if not use_per_point_bc else None
        legacy_bc_values = self._get_boundary_condition_property("values") if not use_per_point_bc else None
        legacy_normals = self._bc_config.get("normals", None) if not use_per_point_bc and self._bc_config else None

        dimension = self.dimension
        n = self.n_points
        # Time coordinate for callable BC values
        current_time = time_idx * (self.problem.T / self.problem.Nt) if getattr(self.problem, "Nt", 0) > 0 else 0.0

        for local_idx, i in enumerate(self.boundary_indices):
            i = int(i)

            # --- Resolve BC for this point (shared classifier, Issue #1118 PR2) ---
            bc_enum, segment, normal = self._classify_boundary_point(
                i, local_idx, use_per_point_bc, global_bc_type, legacy_normals
            )

            # --- Ghost-nodes structural BC: keep PDE row intact ---
            if bc_enum in (BCType.NEUMANN, BCType.NO_FLUX):
                if (
                    self._use_ghost_nodes
                    and self._boundary_handler is not None
                    and i in self._boundary_handler.ghost_node_map
                ):
                    # Symmetric ghost stencils enforce BC structurally; leave row.
                    continue

            # --- Build replacement row + value-form target, then form the Newton residual. ---
            # `_bc_row_for_point` returns (row coefficients, BC target) WITHOUT the residual
            # subtraction, so the Howard value-form path (Issue #1118 PR2) reuses the SAME
            # coefficient source. Here we form the #1116 residual: F_bc(u_current) - target.
            # (Dirichlet's row is e_i, so `new_row @ u_current == u_current[i]` — unchanged.)
            new_row, bc_target = self._bc_row_for_point(
                i, bc_enum, segment, normal, dimension, n, legacy_bc_values, current_time
            )
            new_rhs = float(new_row @ u_current) - bc_target

            # --- Atomic row replacement ---
            jac_lil[i, :] = new_row
            residual_bc[i] = new_rhs

        return jac_lil.tocsr(), residual_bc

    def _eval_bc_dirichlet_value(
        self,
        point_idx: int,
        segment: BCSegment | None,
        legacy_bc_values,
        current_time: float,
    ) -> float:
        """Resolve a Dirichlet RHS value for boundary point ``point_idx``.

        Prefers ``segment.get_value(point, t)`` when a segment is supplied
        (pre-classified path); falls back to legacy ``bc_values`` (dict,
        callable, or scalar) for the uniform-BC path.
        """
        if segment is not None:
            return float(segment.get_value(self.collocation_points[point_idx], t=current_time))
        if isinstance(legacy_bc_values, dict):
            return float(legacy_bc_values.get(point_idx, 0.0))
        if callable(legacy_bc_values):
            return float(legacy_bc_values(self.collocation_points[point_idx]))
        return float(legacy_bc_values) if legacy_bc_values else 0.0

    def _build_neumann_bc_row(
        self,
        point_idx: int,
        normal: np.ndarray,
        dimension: int,
        segment: BCSegment | None,
        legacy_bc_values,
        current_time: float,
    ) -> tuple[np.ndarray, float]:
        """Build the (row, rhs) for a Neumann / no-flux BC at ``point_idx``.

        Row encodes ``normal · grad(u) ≈ Σ_j (normal · grad_weights[:,k]) u_j``,
        so the linear system row is ``[..., w_j, ..., center_weight, ...]``.
        RHS is the prescribed normal-derivative value (0 for no-flux).

        LCR (local coordinate rotation, Issue #531) and ghost-node paths
        choose the gradient stencil; ghost-node short-circuit happens in the
        caller before this is invoked.
        """
        if (
            self._use_local_coordinate_rotation
            and self._boundary_handler is not None
            and point_idx in self._boundary_handler.boundary_rotations
        ):
            if self._neighborhood_builder is not None:
                boundary_rotations = self._boundary_handler.boundary_rotations
                weights = self._neighborhood_builder.compute_derivative_weights_from_taylor(
                    point_idx, boundary_rotations
                )
            else:
                weights = self._compute_derivative_weights_from_taylor(point_idx)
        else:
            weights = self._gfdm_operator.get_derivative_weights(point_idx)

        n = self.n_points
        new_row = np.zeros(n)

        if weights is None:
            # Degenerate stencil — preserve legacy behavior: pin to identity
            # row with zero RHS rather than producing a zero row. A warning
            # might be more honest, but matches existing semantics.
            new_row[point_idx] = 1.0
            return new_row, 0.0

        neighbor_indices = weights["neighbor_indices"]
        grad_weights = weights["grad_weights"]

        center_weight = 0.0
        for k, j in enumerate(neighbor_indices):
            if j >= 0 and j != point_idx:
                w = sum(normal[d] * grad_weights[d, k] for d in range(dimension))
                new_row[j] = w
                center_weight -= w
        new_row[point_idx] = center_weight

        # RHS: segment.get_value at this point, or legacy dict lookup, else 0
        if segment is not None:
            rhs = float(segment.get_value(self.collocation_points[point_idx], t=current_time))
        elif isinstance(legacy_bc_values, dict):
            rhs = float(legacy_bc_values.get(point_idx, 0.0))
        else:
            rhs = 0.0

        return new_row, rhs

    def _compute_llf_sigma_eff(self) -> np.ndarray:
        """Compute per-node effective sigma for LLF augmentation (Issue #1059, paper P2).

        For each collocation node i:

            nu_i = max(0, C * l_H(i) * h_i - sigma^2/2)
            sigma_eff_i = sqrt(sigma^2 + 2 * nu_i)

        where:
          - sigma = solve-level scalar or collocation-space volatility
          - C = self._llf_cone_constant (paper P2 cone constant, default 0.5)
          - l_H(i) = self._llf_l_H[i] (user-supplied |dH/dp| Lipschitz bound)
          - h_i = self.delta (conservative upper bound for local mesh size)

        When nu_i = 0 at node i (Pe already small enough), sigma_eff_i = sigma exactly
        (no augmentation at that node).  Called once at __init__ and stored in
        self._llf_sigma_eff; recomputed only when a new solve-level volatility is
        normalized, not per Picard/Newton iteration (Issue #1059 stability note:
        holding nu_i fixed avoids the runaway feedback in the prototype).

        Returns:
            shape (n_points,) float64 array of per-node effective volatility values.
        """
        sigma = self._solve_sigma if self._solve_sigma is not None else self._get_sigma_value(None)
        D_base = (
            diffusion_from_volatility(sigma, kind="field")
            if isinstance(sigma, np.ndarray)
            else diffusion_from_volatility(sigma)
        )

        # nu_i = max(0, C * l_H(i) * h_i - D_base)
        # h_i approximated by self.delta (conservative upper bound; actual stencil
        # distances are <= delta, so this may over-estimate nu_i slightly, never under).
        nu_i = np.maximum(0.0, self._llf_cone_constant * self._llf_l_H * self.delta - D_base)

        # sigma_eff_i = sqrt(sigma^2 + 2*nu_i)
        return np.sqrt(sigma**2 + 2.0 * nu_i)

    def _get_sigma_value(self, point_idx: int | None = None) -> float:
        """
        Get diffusion coefficient value, handling both numeric and callable sigma.

        When llf_augmentation=True and point_idx is not None, returns the per-node
        effective sigma sqrt(sigma^2 + 2*nu_i) from LLF augmentation (Issue #1059).
        When point_idx is None, preserves the legacy representative-scalar lookup used
        by pre-solve diagnostics and compatibility tests. Live assembly reads
        :meth:`_sigma_for_assembly` instead.

        Args:
            point_idx: Collocation point index (for callable sigma evaluation or LLF lookup)

        Returns:
            Numeric sigma value

        Handles, in precedence order:
        1. LLF per-node augmentation (point_idx given)
        2. volatility_field override (Issue #1316) — the per-solve spatial diffusion
        3. problem.nu (legacy attribute)
        4. problem.sigma is callable → evaluate at the point / center of domain
        5. problem.sigma is numeric → use directly (fallback: 1.0)
        """
        # LLF per-node override: return sigma_eff_i when augmentation is active (Issue #1059).
        # Only applies when point_idx is not None; solve-level assembly reads the full
        # self._llf_sigma_eff field directly.
        if self.llf_augmentation and point_idx is not None and self._llf_sigma_eff is not None:
            if point_idx < len(self._llf_sigma_eff):
                return float(self._llf_sigma_eff[point_idx])

        # Live solves resolve the coefficient once. Per-point residual and Jacobian
        # assembly index that canonical collocation-space value instead of re-evaluating
        # a callable or independently interpreting an array.
        if point_idx is not None and self._solve_sigma is not None:
            if isinstance(self._solve_sigma, np.ndarray):
                return float(self._solve_sigma[point_idx])
            return self._solve_sigma

        # Issue #1316: the per-solve volatility_field override (the spatial diffusion
        # the coupling layer / FP solver is using) is the authoritative diffusion source,
        # replacing problem.sigma so HJB and FP stay convention-consistent. None (the
        # default) falls through to the byte-identical legacy path below.
        if self._volatility_field_override is not None:
            return self._resolve_diffusion_source(self._volatility_field_override, point_idx)

        # Check for legacy "nu" attribute (optional)
        nu = getattr(self.problem, "nu", None)
        if nu is not None:
            return float(nu)

        sigma = getattr(self.problem, "sigma", None)
        if callable(sigma):
            # Callable sigma: evaluate at the point (per-point path) or at the center
            # of the domain (batch path). Issue #1316: the batch path previously returned
            # a hardcoded 1.0 ("representative value" in name only), silently replacing
            # callable sigma with sigma=1.0 (~2x diffusion error). Now an actual
            # center-of-domain evaluation, matching the documented intent.
            return self._resolve_diffusion_source(sigma, point_idx)
        else:
            # Numeric sigma: use directly (with fallback to default)
            return float(getattr(self.problem, "sigma", 1.0))

    def _sigma_for_assembly(self) -> float | np.ndarray:
        """Return the single solve-level volatility consumed by every assembly path."""
        if self.llf_augmentation and self._llf_sigma_eff is not None:
            return self._llf_sigma_eff
        if self._solve_sigma is not None:
            return self._solve_sigma
        return self._get_sigma_value(None)

    def _resolve_sigma_for_solve(
        self,
        volatility_field: float | np.ndarray | Callable | None,
        *,
        is_meshfree_input: bool,
    ) -> float | np.ndarray:
        """Normalize one volatility source to a scalar or collocation-space ``(N,)`` field."""
        problem_field = getattr(self.problem, "volatility_field", None)
        source_is_problem_field = volatility_field is None or volatility_field is problem_field

        if volatility_field is not None:
            source = volatility_field
        else:
            legacy_nu = getattr(self.problem, "nu", None)
            if legacy_nu is not None:
                source = legacy_nu
                source_is_problem_field = False
            elif problem_field is not None:
                source = problem_field
            else:
                source = getattr(self.problem, "sigma", 1.0)
                source_is_problem_field = False

        if callable(source):
            try:
                parameters = list(inspect.signature(source).parameters.values())
            except (TypeError, ValueError) as exc:
                raise NotImplementedError(
                    "HJBGFDMSolver supports only a space-only volatility callable sigma(x) "
                    "whose one-argument signature can be inspected. Time- or density-dependent "
                    "sigma(t, x, m) is unsupported."
                ) from exc
            if len(parameters) != 1 or parameters[0].kind not in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                raise NotImplementedError(
                    "HJBGFDMSolver supports only a space-only volatility callable sigma(x); "
                    f"got signature {inspect.signature(source)}. Time- or density-dependent "
                    "sigma(t, x, m) cannot be frozen into one solve-level spatial field."
                )
            return np.array(
                [self._resolve_diffusion_source(source, i) for i in range(self.n_points)],
                dtype=float,
            )

        sigma = np.asarray(source, dtype=float)
        if sigma.ndim == 0:
            return float(sigma)

        # An explicit per-solve array follows the representation of M_density. A
        # problem-owned array follows the problem geometry even when a caller supplies
        # collocation-space density directly: grid problems own grid fields, whereas
        # implicit problems own node-indexed fields.
        problem_field_is_grid_indexed = source_is_problem_field and getattr(self.problem, "domain_type", None) == "grid"
        grid_indexed = not is_meshfree_input or problem_field_is_grid_indexed
        if grid_indexed:
            grid_shape = tuple(self._mapper.grid_shape)
            grid_size = int(np.prod(grid_shape))
            if sigma.shape == grid_shape:
                sigma_grid = sigma.reshape(-1)
            elif sigma.shape == (grid_size,):
                sigma_grid = sigma
            else:
                raise ValueError(
                    "volatility_field shape mismatch: a grid-indexed scalar-volatility field "
                    f"must have native shape {grid_shape} or flattened one-dimensional shape "
                    f"({grid_size},), got {sigma.shape}; collocation space has "
                    f"{self.n_points} points."
                )
            resolved = np.asarray(self._mapper.map_grid_to_collocation(sigma_grid), dtype=float)
        else:
            if sigma.shape != (self.n_points,):
                raise ValueError(
                    "volatility_field shape mismatch: a meshfree scalar-volatility field "
                    f"must have one-dimensional collocation shape ({self.n_points},), "
                    f"got {sigma.shape}."
                )
            resolved = sigma

        if resolved.shape != (self.n_points,):
            raise AssertionError(
                "volatility_field normalization violated the collocation-space invariant: "
                f"expected ({self.n_points},), got {resolved.shape}."
            )
        return resolved

    def _resolve_diffusion_source(self, source: float | np.ndarray | Callable, point_idx: int | None) -> float:
        """Resolve a scalar / callable / per-point-array diffusion source to a scalar sigma.

        Thin adapter over the shared single source
        :func:`mfgarchon.utils.pde_coefficients.resolve_diffusion_source` (Issue #1412): the
        collocation points are this solver's spatial points; the batch path (``point_idx=None``)
        collapses an array to its mean / evaluates a callable at the domain center, matching
        ``MFGProblem``'s array -> scalar (sigma = mean) convention.
        """
        return resolve_diffusion_source(source, index=point_idx, points=self.collocation_points)

    # Note: _check_monotonicity_violation moved to MonotonicityEnforcer component
    # Note: _check_m_matrix_property moved to MonotonicityEnforcer component
    # Note: _build_monotonicity_constraints moved to MonotonicityEnforcer component
    # Note: _build_hamiltonian_gradient_constraints moved to MonotonicityEnforcer component

    def solve_hjb_system(
        self,
        M_density: np.ndarray | None = None,
        U_terminal: np.ndarray | None = None,
        U_coupling_prev: np.ndarray | None = None,
        show_progress: bool | None = None,
        volatility_field: float | np.ndarray | Callable | None = None,
        source_term: Callable | None = None,
    ) -> np.ndarray:
        """
        Solve the HJB system using GFDM collocation method.

        Args:
            M_density: (Nt, *spatial_shape) density from FP solver
            U_terminal: (*spatial_shape,) terminal condition u(T,x)
            U_coupling_prev: (Nt, *spatial_shape) previous coupling iteration estimate
            show_progress: Whether to display progress bar for timesteps
            source_term: MMS forcing ``r_u`` of ``-u_t - (sigma^2/2) lap u + H = r_u``, with the
                package-wide contract ``source_term(t, x) -> array``, ``x`` of shape ``(N, d)``
                (:mod:`base_hjb`). This is the ONLY additive channel a caller can reach. The
                alpha-independent part of the Lagrangian -- the potential ``V(x,t)`` and the
                coupling ``f(m)`` -- belongs to the Hamiltonian and arrives through it, so a
                source cannot be confused with a running cost: there is no longer a second slot
                to confuse it with (Issue #1999). Internally the two still meet with OPPOSITE
                signs, since ``h_eval`` assembles ``-u_t + H(+additive_source) - D*lap_u`` while
                the source contract subtracts; the conversion happens once, in ``_source_at``,
                which returns ``-S``. So the effective Hamiltonian is ``H_total = H - S``, NOT
                ``H + S``: migrating a callable from the retired ``running_cost=`` channel
                requires FLIPPING ITS SIGN, since ``running_cost = -source_term``.
            volatility_field: Optional SDE-volatility override. Accepts a scalar,
                an inspectable one-argument space-only callable ``sigma(x)`` evaluated
                at each collocation point, or a
                scalar-valued spatial array. Grid-indexed arrays may use the native
                problem spatial shape or its flattened form and are mapped to
                collocation points; meshfree arrays must have shape ``(n_points,)``.
                If the native grid shape is also ``(d, d)``, pass the array through
                this argument to distinguish a scalar grid field from tensor sigma.
                Time-/density-varying callables, time-varying arrays, and tensor
                volatility are unsupported.

        Returns:
            (Nt, *spatial_shape) solution array
        """
        # Retain the raw public argument and clear the previous solve's normalized value
        # before validating this solve.
        self._volatility_field_override = volatility_field
        self._solve_sigma = None

        # Pick up any per-Picard resolved BC the coupling layer installed on the
        # geometry since construction (Issue #1118; matches FDM's per-solve re-read).
        self._refresh_boundary_conditions_if_changed()

        # Validate required parameters
        if U_terminal is None:
            raise ValueError("U_terminal is required")

        # Validate BC specification if boundary points exist
        if len(self.boundary_indices) > 0 and (self._bc_config is None or self._bc_config.get("type") is None):
            raise ValueError(
                f"Boundary conditions required for solving but not specified. "
                f"Found {len(self.boundary_indices)} boundary points. "
                f"Pass boundary_conditions parameter to solver or set BC on problem.geometry."
            )

        # Determine n_time_points from available data or problem configuration
        # n_time_points = Nt + 1 (number of time knots including t=0 and t=T)
        if M_density is not None:
            n_time_points = M_density.shape[0]
        else:
            n_time_points = self.problem.Nt + 1

        # For standalone HJB (no MFG coupling), use defaults
        if M_density is None:
            # Default: uniform density (no coupling effect)
            M_density = np.ones((n_time_points, *U_terminal.shape))
        if U_coupling_prev is None:
            # Default: zero coupling (pure HJB)
            U_coupling_prev = np.zeros((n_time_points, *U_terminal.shape))

        # Store original spatial shape for reshaping output
        self._output_spatial_shape = M_density.shape[1:]

        # Issue #1999: there is no user running-cost channel. The alpha-independent part of the
        # Lagrangian -- V(x,t) + f(m) -- is owned by the Hamiltonian and already enters through
        # `eval_H_batch` on the Newton path and `howard_running_cost` on the Howard path. A second
        # channel could only carry the same quantity, and adding it on top of a Hamiltonian that
        # already holds a potential double-counted it silently (#2001). The MMS source is now the
        # only additive channel a caller can reach, so a source and a running cost can no longer
        # be confused: there is only one of them.
        self._mms_source_fn: Callable[[int], np.ndarray] | None = None
        if source_term is not None:
            _dt = self.problem.T / self.problem.Nt
            _x = self.collocation_points

            def _source_at(n: int, _dt: float = _dt, _x: np.ndarray = _x) -> np.ndarray:
                # `additive_source = -source_term`: h_eval assembles `-u_t + H(+additive_source)
                # - D*lap_u` while the source contract is `F(u) = (u-u_next)/dt + H - S = 0`.
                # Getting this backwards is not subtle -- measured on the manufactured pair,
                # `-r_u` converges at EOC 2.00/1.99 while `+r_u` sits flat at 1.42.
                s_n = np.asarray(source_term(n * _dt, _x), dtype=float)
                # Shape-check rather than reshape. A 2D source handed back in the wrong point
                # order has the right SIZE and silently yields a different value function --
                # measured, an F-ordered (nx, ny) array is accepted and changes Linf from
                # 6.6433e+00 to 4.6862e+00 with no diagnostic, and nothing downstream
                # validates its callable's output; this is the same contract.
                if s_n.shape != (self.n_points,):
                    # An (N,1) or (1,N) vector has an unambiguous ordering, and `base_hjb`
                    # ravels, `hjb_fdm` reshapes, `hjb_weno` normalises nothing -- rejecting it
                    # here would defeat this argument's own purpose, that one manufactured
                    # solution runs against every solver. Reject only what is genuinely
                    # ambiguous: a 2D array with both extents > 1, whose point order the caller
                    # and the collocation cloud can disagree about silently.
                    if s_n.size == self.n_points and s_n.ndim <= 2 and 1 in s_n.shape:
                        s_n = s_n.reshape(-1)
                if s_n.shape != (self.n_points,):
                    raise ValueError(
                        f"source_term must return shape ({self.n_points},) at the collocation "
                        f"points, got {s_n.shape}. A flattened grid array may be in the wrong "
                        f"point order; index it by the solver's collocation_points."
                    )
                return -s_n

            self._mms_source_fn = _source_at

        # Detect if input is already in collocation format (pure meshfree mode)
        # Grid format: M_density.shape = (Nt, Nx, Ny, ...)
        # Collocation format: M_density.shape = (Nt, n_points)
        is_meshfree_input = M_density.ndim == 2 and M_density.shape[1] == self.n_points

        # Issue #1725: resolve once, after the input representation is known and before
        # any interpolation, Newton probe, sparse assembly, LLF, DMP, or Howard path.
        self._solve_sigma = self._resolve_sigma_for_solve(
            volatility_field,
            is_meshfree_input=is_meshfree_input,
        )
        self._dmp_alpha_crit = None
        if self.llf_augmentation:
            self._llf_sigma_eff = self._compute_llf_sigma_eff()

        # For GFDM, we work directly with collocation points
        U_solution_collocation = np.zeros((n_time_points, self.n_points))

        if is_meshfree_input:
            # Pure meshfree mode: input already at collocation points
            M_collocation = M_density.copy()
            # U_terminal should also be at collocation points
            U_solution_collocation[n_time_points - 1, :] = U_terminal.copy()
        else:
            # Hybrid mode: map grid data to collocation points
            M_collocation = self._mapper.map_grid_to_collocation_batch(M_density)
            # Set final condition at t=T (last time index = n_time_points - 1)
            U_solution_collocation[n_time_points - 1, :] = self._mapper.map_grid_to_collocation(U_terminal.flatten())

        if self.inner_solver == "howard":
            # Issue #1118: delegate the backward sweep to Howard's policy iteration (no
            # Armijo line search -> no MIN_ALPHA stall). Howard owns the full Nt loop and
            # works in collocation space; the grid<->collocation mapping above and below is
            # unchanged. Validation + alpha* synthesis live in the helper.
            U_solution_collocation = self._solve_backward_howard(
                M_collocation, U_solution_collocation[n_time_points - 1, :]
            )
            # Issue #1381: Howard owns the full backward sweep and bypasses _solve_timestep,
            # so the diagnostic DMP guard the Newton path runs per timestep (see
            # _maybe_warn_dmp call below) would otherwise never fire on this path. Run it
            # over the solved time-slices; the warn-once latch stops after the first
            # exceedance. No-op unless check_dmp=True (first line early-returns), and the
            # drift check is numerically inert (Issue #1074).
            for n in range(n_time_points - 1):
                self._maybe_warn_dmp(U_solution_collocation[n, :])
        else:
            # Backward time stepping: Nt steps from index (n_time_points-2) down to 0
            # This covers all Nt intervals in the backward direction (Issue #587 Protocol pattern)
            from mfgarchon.utils.progress import create_progress_bar, should_show_progress

            timestep_range = create_progress_bar(
                range(n_time_points - 2, -1, -1),
                verbose=should_show_progress(show_progress),
                desc="HJB (backward)",
            )

            for n in timestep_range:
                rc_n = self._mms_source_fn(n) if self._mms_source_fn is not None else None

                U_solution_collocation[n, :] = self._solve_timestep(
                    U_solution_collocation[n + 1, :],
                    M_collocation[n, :],  # FIXED: Use m^n, not m^{n+1} (same-time coupling)
                    n,
                    additive_source=rc_n,
                )

                # Update progress bar with QP statistics if available (Issue #587 Protocol - no hasattr needed)
                if self.qp_optimization_level in ["auto", "always"]:
                    timestep_range.update_metrics(qp_solves=self.qp_stats.get("total_qp_solves", 0))

        # Return format depends on input mode
        if is_meshfree_input:
            # Pure meshfree: return collocation data directly
            return U_solution_collocation
        else:
            # Hybrid mode: map back to grid
            U_solution = self._mapper.map_collocation_to_grid_batch(U_solution_collocation)
            return U_solution

    def _solve_backward_howard(self, M_collocation: np.ndarray, U_terminal_colloc: np.ndarray) -> np.ndarray:
        """Issue #1118: backward HJB sweep via Howard's policy iteration (inner_solver='howard').

        Howard has no Armijo line search, so it avoids the MIN_ALPHA stall the per-point Newton
        path hits on advection-dominant / no-flux-BC regimes (#1118). Operates in collocation
        space; the caller handles grid<->collocation mapping. Requires a Hamiltonian exposing
        dp(). SOCP stencils are used when present and are no longer required (#2066).

        Issue #1247 (#1118 PR2): the separable Hamiltonian's potential V(x, t) and density
        coupling f(m), plus any caller-supplied running cost, are wired into Howard's
        running_cost slot (see `running_cost` closure below), so Howard solves the full non-LQ
        HJB ``-d_t u + (1/2)|grad u|^2 + V(x) + f(m) - (sigma^2/2) Lap u = 0``. Still deferred
        (fail-loud below): non-unit control cost lambda, non-quadratic control cost, and the
        MAXIMIZE sense (the Lagrangian-scaling work tracked alongside #1071).
        """
        from mfgarchon.alg.numerical.hjb_solvers.hjb_howard import HJBHowardSolver

        # Issue #2066: this gate used to refuse here, duplicating a check HJBHowardSolver no
        # longer makes. Howard needs D_lap, D_grad and an interior/boundary split; none is
        # SOCP-specific, and the builders close the stencil row whatever produced the weights
        # (#2081), so SOCP weights and operator weights both assemble correctly. Delegating means
        # the caller gets Howard's own diagnosis: a warning naming the lost monotonicity when an
        # operator is available, and a hard error when there is no weight source at all -- which
        # is strictly more informative than refusing on the presence of one particular object.
        H_class = getattr(self.problem, "hamiltonian_class", None)
        if H_class is None:
            raise ValueError(
                "inner_solver='howard' requires problem.hamiltonian_class to derive the optimal "
                "control alpha* = -dH/dp; the legacy no-Hamiltonian LQ path is unsupported."
            )
        # Howard derives alpha* = -dH/dp and now consumes the control-cost Lagrangian
        # L(alpha) = lambda/2 |alpha|^2 from the single source (control_cost.lagrangian, wired
        # below), so any QUADRATIC control cost (unit or lambda != 1) MINIMIZE is faithful. The
        # potential V(x, t), the density coupling f(m), and the MMS source are wired
        # (Issue #1247, below); the user running-cost channel is gone (#1999). What remains unmodelled — NON-quadratic control cost and the
        # MAXIMIZE sense — is failed loud below (validated by
        # tests/unit/test_alg/test_hjb_howard_solver.py::test_integrated_howard_rejects_*).
        control_cost = getattr(H_class, "control_cost", None)
        if control_cost is not None:
            from mfgarchon.core.hamiltonian import QuadraticControlCost

            # Issue #1071: the policy-evaluation RHS now consumes control_cost.lagrangian()
            # (wired below as control_lagrangian=), so any QUADRATIC control cost — unit or
            # lambda != 1 — is exact: L(alpha) = lambda/2 |alpha|^2. Non-quadratic costs (L1,
            # bounded) have a non-smooth Lagrangian whose Howard policy-iteration convergence
            # is unvalidated, so they stay gated.
            if not isinstance(control_cost, QuadraticControlCost):
                raise NotImplementedError(
                    "inner_solver='howard' supports a quadratic control cost (its Lagrangian "
                    "L(alpha) = lambda/2 |alpha|^2 is wired into the policy-evaluation RHS, any "
                    f"lambda), but the Hamiltonian's control cost is {type(control_cost).__name__}. "
                    "Non-quadratic control costs have a non-smooth Lagrangian whose Howard "
                    "convergence is unvalidated; use inner_solver='newton' (Issue #1071)."
                )
            if getattr(control_cost, "sign", 1) != 1:
                raise NotImplementedError(
                    "inner_solver='howard' derives alpha* = -dH/dp (MINIMIZE sense); the Hamiltonian "
                    "uses MAXIMIZE, which needs alpha* = +dH/dp. Use inner_solver='newton' (deferred)."
                )
        # Ordered BEFORE the probe deliberately: the widened probe DOES detect congestion
        # (measured, it departs from the declared quadratic by 1.000e+02), but it would report
        # the generic "assumptions do not hold" message. The specific one names the mechanism
        # and the attribute, which is what a caller can act on.
        # CongestionHamiltonian (Issue #782) carries a MULTIPLICATIVE kinetic factor c(m):
        # H = |p|^2/(2*lambda*c(m)) + V(x, t) + f(m). The control-cost gate above does NOT catch
        # it — c(m) lives in `_congestion_factor`, outside control_cost (a plain unit-quadratic
        # QuadraticControlCost here) — and V(x)/f(m) are wired below, but the congestion factor
        # is not: Howard's policy evaluation hardcodes the unit-quadratic Lagrangian (1/2)|alpha|^2
        # (hjb_howard.py _howard_step RHS) with no c(m) factor, so the value function silently
        # decouples from the congestion. Gate on the factor directly. Fail loud.
        if getattr(H_class, "_congestion_factor", None) is not None:
            raise NotImplementedError(
                "inner_solver='howard' does not support multiplicative kinetic congestion c(m) "
                "(CongestionHamiltonian): Howard hardcodes the unit-quadratic Lagrangian "
                "(1/2)|alpha|^2 with no c(m) factor, so the value function silently decouples from "
                "the congestion. Use inner_solver='newton' (deferred, Issue #1071/#1118 PR2)."
            )

        # Issue #2011, and #2015's review. THE PROBE RUNS FOR EVERY HAMILTONIAN, not only for one
        # that lacks `control_cost`. An earlier version put it in the `else:` of the branch above,
        # which meant any Hamiltonian exposing a QuadraticControlCost -- the ordinary way to write
        # one -- skipped the entire gate. Measured on that version: a SeparableHamiltonian subclass
        # carrying an extra g*(m-1) term in __call__ was ACCEPTED, and #2011's original failure
        # reproduced exactly, max|u(g=2) - u(g=0)| = 0.0000e+00 on Howard against 6.6083e-01 on
        # Newton. That is the third version of this guard repeating the same structural mistake one
        # branch over: v1 keyed the REFUSAL on the attribute, v3 keyed the PROBE'S SCOPE on it.
        #
        # The reason the probe can now run unconditionally is that its kinetic reference comes from
        # the DECLARED control cost when there is one. Against a hard-coded (1/2)|p|^2 it would
        # false-refuse lambda != 1: QuadraticControlCost(control_cost=2.0) gives H(p=1) - H(0) =
        # 0.2500, which the unit quadratic calls a defect and which Howard handles exactly via
        # control_lagrangian.
        if True:
            # Two things happen silently when a Hamiltonian's structure is not what Howard assumes:
            #
            #   1. `control_lagrangian` stays None below, so hjb_howard substitutes the UNIT
            #      quadratic L(alpha) = (1/2)|alpha|^2 whatever the Hamiltonian's actual control cost.
            #   2. `has_H_extra` below is keyed on `_potential` / `_coupling`, also
            #      SeparableHamiltonian internals, so an alpha-free part carried any other way is
            #      dropped BITWISE.
            #
            # GATE ON BEHAVIOUR, AND ON THE PROBLEM'S OWN DATA. Two earlier versions of this guard
            # failed for the same reason one level apart, and both are worth stating because the
            # shape recurs:
            #
            #   - keying on `getattr(H_class, "control_cost", None) is None` refused `H = |p|^2/2`
            #     written as a bare subclass -- a Hamiltonian the substitution is EXACT for. An
            #     attribute standing in for a question about behaviour.
            #   - probing behaviour at `m = ones`, `t = 0`, `p = e_0` accepted six wrong
            #     Hamiltonians, measured: any f(m) with f(1) = 0 (invisible at m = ones), |p|^2/(2m)
            #     congestion with c(1) = 1, (1/2)|p|^4 (agrees with the unit quadratic at |p| = 0 and
            #     1, the only two points sampled), and any anisotropy (only e_0 sampled). A stand-in
            #     for the data standing in for the data.
            #
            # So the probe runs on `M_collocation` at several time slices, at the matching physical
            # times, over several momentum directions AND magnitudes.
            _pot = getattr(H_class, "_potential", None)
            _cpl = getattr(H_class, "_coupling", None)
            _alpha_free_is_wired = _pot is not None or _cpl is not None
            _dt_probe = float(self.problem.T) / int(self.problem.Nt)
            _pts = np.asarray(self.collocation_points, dtype=float)
            _rng = np.random.default_rng(0)
            _slices = np.unique(np.linspace(0, M_collocation.shape[0] - 1, 3).astype(int))

            # MAGNITUDES FROM THE PROBLEM, not hard-coded. The previous version sampled |p| in
            # {0.5, 1, 2} while the solve visits max|grad u| = 6.18 on this PR's own fixture, so
            # H(p) = (1/2)|p|^2 + C*max(0, |p|^2 - 4)^2 -- convex, C^1, and wrong for Howard exactly
            # where the solve lives -- sat in the probe's null space and was accepted at every C.
            #
            # Derived from the terminal datum by a spacing bound rather than through
            # `_compute_gradient_at_point`: that accessor raises KeyError('weights') at this point in
            # the SOCP-precompute path, and an earlier draft of this guard wrapped it in a bare
            # `except` and silently fell back to 1.0 -- so the widening never happened and the
            # super-quadratic above stayed accepted. This form cannot fail, and OVERESTIMATING is the
            # safe direction: the kinetic reference tracks a genuine quadratic exactly at every |p|,
            # so a larger probe cannot produce a false refusal, only a stricter true one.
            # THE DISCRETE LIPSCHITZ CONSTANT OF u_T, not `spread / hmin`. The latter is a spacing
            # bound with the wrong dimensional content: it divides a GLOBAL range by a LOCAL
            # spacing, so it diverges linearly under refinement while the gradient it stands for
            # converges. Measured on u_T = cos(2 pi x): spread/hmin gives 20, 40, 80, 200 at
            # nx = 11, 21, 41, 201 while max|du/dx| goes 3.09 -> 3.14; on a linear ramp it gives
            # 200 where the true gradient is 1. The overestimate reaches 63x by nx = 201.
            #
            # That is why the earlier widening failed. It was told the probe missed |p| = 6.18 and
            # widened the ladder to multiples of a quantity 6.4x too large, moving the new rungs
            # FURTHER from the hole. `max |u_i - u_j| / |x_i - x_j|` is a genuine upper bound on
            # the discrete gradient, converges to it rather than away, and costs nothing: both
            # arrays are already materialised here. (#2072)
            _uT = np.asarray(U_terminal_colloc, dtype=float).ravel()
            _dists = np.linalg.norm(_pts[:, None, :] - _pts[None, :, :], axis=-1)
            np.fill_diagonal(_dists, np.inf)
            if _pts.shape[0] > 1 and _uT.size == _pts.shape[0]:
                _du = np.abs(_uT[:, None] - _uT[None, :])
                with np.errstate(invalid="ignore", divide="ignore"):
                    _ratios = np.where(np.isfinite(_dists) & (_dists > 0.0), _du / _dists, 0.0)
                _gT = float(np.max(_ratios)) if _ratios.size else 0.0
            else:
                _gT = 0.0
            _gT = _gT if np.isfinite(_gT) and _gT > 0.0 else 1.0
            # GEOMETRIC SPAN, not a few multiples of _gT. The previous form was
            # sorted({0.5, 1, 2, 0.5*_gT, _gT, 2*_gT}), which on this file's own fixture is
            # {0.5, 1, 2, 20, 40, 80} -- and `_gT` OVERESTIMATES the true max|grad u_T| (2*pi =
            # 6.2832 here) by 6.4x, because it is a spacing bound rather than a gradient. So the
            # ladder straddled the operating range without ever landing in it, and the widening
            # that was meant to close the |p| = 6.18 hole moved the upper rungs further away from
            # it. Measured: H = |p|^2/2 + bump(|p|^2) with the bump supported on |p| in (3, 10)
            # passed the probe and produced a finite, plausible field 10.9% wrong against Newton.
            #
            # A span cannot have that hole. Overestimating `_gT` stays safe for the same reason as
            # before -- a genuine quadratic matches the kinetic reference at EVERY |p|, so more
            # sample points can only make a true refusal stricter, never invent a false one -- and
            # the span keeps the old ladder's reach while filling what it stepped over. (#2072)
            # The span is DENSER, not hole-free, and an earlier revision of this comment claimed
            # otherwise. Adjacent rungs of a 12-point geomspace over [0.5, 2*_gT] sit at a ratio of
            # ~1.59, so a bump narrower than that in relative width still fits between two. What
            # closes the operating-range hole is `_gT` above now tracking the real gradient, which
            # puts the ladder's own rungs where the solve actually lives.
            #
            # The old ladder's EXACT rungs at 0.5*_gT and _gT are kept in the union rather than
            # dropped: the geomspace grid lands on neither, and a bump centred on one of them was
            # refused by the old form and accepted by a geomspace-only one. Same for the low end --
            # a hard floor of 0.5 stops scaling down when _gT < 1, so the floor is min(0.5, _gT/4).
            _lo = min(0.5, _gT / 4.0)
            _mags = sorted(
                set(np.geomspace(_lo, max(2.0 * _gT, 2.0 * _lo), 12).tolist())
                | {0.5, 1.0, 2.0, 0.5 * _gT, _gT, 2.0 * _gT}
            )
            _dirs = [np.eye(self.dimension)[0] * _m for _m in _mags]
            _dirs += [v / np.linalg.norm(v) * _m for _m in _mags for v in (_rng.normal(size=self.dimension),)]

            # The kinetic reference is what Howard will actually substitute: the DECLARED control
            # cost's H_control(p) when the Hamiltonian exposes one (control_lagrangian is wired from
            # the same object), and the unit quadratic only when nothing is declared.
            def _kinetic_ref(_d):
                if control_cost is not None:
                    return float(np.asarray(control_cost.evaluate(np.asarray(_d, dtype=float))).ravel()[0])
                return 0.5 * float(np.dot(_d, _d))

            # `np.maximum` over the builtin does NOT make this NaN-safe, and an earlier version of
            # this comment claimed it did. Measured: `np.maximum(0.0, nan)` is `nan` and
            # `nan > tol` is False; `max(0.0, nan)` is `0.0` and `0.0 > tol` is False. IDENTICAL
            # acceptance. A quartic returning NaN at one probe magnitude the solve never visits was
            # ACCEPTED and produced a finite field 153% wrong against Newton. The non-finite check
            # after the loop is what actually closes it: a probe that could not be evaluated is not
            # a probe that passed. (#2072)
            # (An earlier version of this comment demonstrated the point with "_af 5.98 -> exactly
            # 0.0". That was the BUILTIN max's behaviour; with np.maximum the value becomes nan,
            # and the acceptance is identical either way -- which is the whole point above.)
            _af = _ke = _scale = 0.0
            try:
                for _n in _slices:
                    _m_n = np.asarray(M_collocation[_n], dtype=float).ravel()
                    _t_n = float(_n) * _dt_probe
                    _h0 = np.asarray(H_class(_pts, _m_n, np.zeros((self.n_points, self.dimension)), _t_n), dtype=float)
                    _af = np.maximum(_af, np.abs(_h0).max())
                    for _d in _dirs:
                        _P = np.tile(np.asarray(_d, dtype=float), (self.n_points, 1))
                        _h = np.asarray(H_class(_pts, _m_n, _P, _t_n), dtype=float)
                        _scale = np.maximum(_scale, np.abs(_h).max())
                        _ke = np.maximum(_ke, np.abs((_h - _h0) - _kinetic_ref(_d)).max())
            except (TypeError, ValueError, AttributeError, IndexError) as _exc:
                # Narrow deliberately: these are what a Hamiltonian that cannot take the batch
                # convention raises. A bare `except Exception` would also swallow a genuine bug
                # inside a working H() and report it as "cannot probe".
                raise NotImplementedError(
                    f"inner_solver='howard' could not probe {type(H_class).__name__} to verify its "
                    f"policy-evaluation assumptions ({type(_exc).__name__}: {_exc}). Howard needs a "
                    f"batch-callable H(x, m, p, t); use inner_solver='newton' (Issue #2011)."
                ) from _exc
            # RELATIVE, not absolute: an algebraically exact unit quadratic whose alpha-free part
            # cancels from terms of magnitude K leaves a residue ~K*eps, and an absolute 1e-10 bound
            # false-refuses it from K ~ 5e5.
            # A PROBE THAT COULD NOT BE EVALUATED IS NOT A PROBE THAT PASSED. Without this, a single
            # non-finite value anywhere in the sweep leaves `_af`/`_ke` as NaN, and every subsequent
            # comparison `nan > tol` is False -- so the guard ACCEPTS, which is the outcome it exists
            # to prevent. Demonstrated on a quartic returning NaN at one probe magnitude the solve
            # never visits: accepted, all-finite output, 153% wrong against Newton. (#2072)
            if not (np.isfinite(_af) and np.isfinite(_ke) and np.isfinite(_scale)):
                raise NotImplementedError(
                    f"inner_solver='howard' could not evaluate {type(H_class).__name__} over this "
                    f"problem's probe: the decomposition check produced a non-finite value "
                    f"(max|H(x,m,0,t)| = {_af}, kinetic departure = {_ke}, |H| scale = {_scale}). "
                    f"A non-finite probe cannot certify that Howard's substituted Lagrangian holds, "
                    f"and comparing it against a tolerance silently ACCEPTS -- `nan > tol` is False. "
                    f"Probed at momentum magnitudes {[round(float(m), 4) for m in _mags]} over "
                    f"{len(_dirs)} directions. Use inner_solver='newton', which reads the "
                    f"Hamiltonian through H() and dp() and needs no decomposition. (Issue #2072)"
                )

            _tol = 1e-10 * max(1.0, _scale)
            _af_bad = (not _alpha_free_is_wired) and _af > _tol
            if _af_bad or _ke > _tol:
                raise NotImplementedError(
                    f"inner_solver='howard' cannot decompose {type(H_class).__name__}: "
                    f"{'it exposes no `control_cost`, and probing' if control_cost is None else 'probing'} "
                    f"it on this problem's own density, times and momentum scale shows Howard's "
                    f"substituted assumptions do not hold "
                    f"(max|H(x,m,0,t)| = {_af:.3e}{' (unwired)' if _af_bad else ' (wired, not gated)'}, "
                    f"and H(x,m,p,t) - H(x,m,0,t) departs from (1/2)|p|^2 by {_ke:.3e}; "
                    f"tolerance {_tol:.1e}, relative to a probed |H| of {_scale:.3e}). Probed at "
                    f"M_collocation slices {list(_slices)}, times {[round(float(n) * _dt_probe, 4) for n in _slices]}, "
                    f"and {len(_dirs)} momentum vectors at |p| in {[round(float(m), 4) for m in _mags]} "
                    f"(scaled to a terminal-gradient bound of {_gT:.4g}). Howard would silently substitute "
                    f"L(alpha) = {'its declared quadratic Lagrangian' if control_cost is not None else '(1/2)|alpha|^2'} "
                    f"and drop the alpha-free part bitwise (Issue #2011). "
                    f"Use inner_solver='newton', which reads the Hamiltonian through H() and dp() "
                    f"and needs no decomposition."
                )

        # BC parity (Issue #1118 PR2a): the howard path now consumes the provider's shared
        # value-form BC rows (`_value_form_bc_rows` -> `_bc_row_for_point`), so it honors
        # Dirichlet VALUES and the real Neumann normal·grad stencil (not the legacy
        # nearest-interior copy). Still deferred to PR2b: ROBIN / PERIODIC (no row builder on
        # either path) and BCValueProvider coupling (e.g. AdjointConsistentProvider). Inspect
        # segments directly: for a mixed BC, `boundary_conditions.type` raises and
        # `_bc_config["type"]` is a meaningless 'periodic' fallback, so neither is reliable.
        # Issue #1118 PR2b: ROBIN is now consumable by the value-form row builder, but only
        # the adjoint-consistent ROBIN(alpha=0, beta=1) case (n.grad u = g). The alpha/beta
        # check lives in exactly one place — _bc_row_for_point fail-louds on ROBIN(alpha != 0)
        # or beta != 1 — so we add "robin" here and let the row builder be the gate.
        allowed_bc = {"no_flux", "neumann", "dirichlet", "robin"}
        segments = getattr(self.boundary_conditions, "segments", None)
        if segments:
            for seg in segments:
                seg_type = seg.bc_type.value if hasattr(seg.bc_type, "value") else str(seg.bc_type)
                if seg_type not in allowed_bc:
                    raise NotImplementedError(
                        f"inner_solver='howard' supports no-flux/Neumann/Dirichlet/Robin BC; "
                        f"segment {getattr(seg, 'name', '?')!r} is {seg_type!r}. PERIODIC parity "
                        f"is deferred (Issue #1118 PR2b)."
                    )
                if callable(getattr(getattr(seg, "value", None), "compute", None)):
                    # Providers are resolved UPSTREAM by the coupling layer
                    # (problem.using_resolved_bc -> with_resolved_providers swaps in a float,
                    # keeping bc_type=ROBIN) and Part 1's per-solve refresh transports that float
                    # into the solver. By the time Howard runs, no segment should still carry a
                    # callable .compute; if one does, the coupling layer failed to resolve it (or
                    # a static BC carries a raw provider) — a real bug, not a deferred feature.
                    raise AssertionError(
                        f"inner_solver='howard': segment {getattr(seg, 'name', '?')!r} still "
                        f"carries an unresolved BCValueProvider (callable .compute) at solve "
                        f"time. Providers must be resolved by the coupling layer "
                        f"(using_resolved_bc / with_resolved_providers) before reaching the "
                        f"solver (Issue #1118 PR2b)."
                    )
        else:
            try:
                uniform_type = self.boundary_conditions.type
            except Exception:
                uniform_type = None
            if uniform_type is not None and uniform_type not in allowed_bc:
                raise NotImplementedError(
                    f"inner_solver='howard' supports no-flux/Neumann/Dirichlet/Robin BC, got "
                    f"{uniform_type!r}. PERIODIC parity is deferred (Issue #1118 PR2b)."
                )

        dt = float(self.problem.T) / int(self.problem.Nt)

        def alpha_star(x_pts, p, m, t_idx):
            # Optimal feedback control alpha* = -dH/dp (the same dp() the Newton Jacobian reads).
            return -eval_dH_dp_batch(H_class, x_pts, m, p, t_idx * dt)

        # Issue #1247 (#1118 PR2): route the Hamiltonian's non-quadratic-in-alpha source terms
        # — potential V(x, t), density coupling f(m^n) — and the MMS source into Howard's
        # running_cost slot, so Howard solves the full non-LQ HJB. There is no user running
        # cost to route: #1999 removed that channel.
        #
        #   SIGN of rc_t (load-bearing). Howard's converged policy-evaluation equation is
        #       (u^n - u^{n+1})/dt + (1/2)|grad u^n|^2 - (sigma^2/2) Lap u^n - rc_t = 0
        #   (substitute alpha* = -grad u^n into b = u^{n+1}/dt + (1/2)|alpha|^2 + rc_t and the
        #   advection operator A_adv u = alpha . grad u; see hjb_howard.py:_howard_step). The
        #   Newton ground truth (h_eval.assemble_hjb_residual, -u_t + H + S_slot - D Lap u = 0
        #   with H = (1/2)|grad u|^2 + V + f(m), and S_slot = -S the MMS source already converted
        #   into h_eval's additive_source convention by _source_at) is
        #       (u^n - u^{n+1})/dt + (1/2)|grad u^n|^2 + V + f(m) + S_slot - (sigma^2/2) Lap u^n = 0.
        #   Matching the two forces rc_t = -(V + f(m) + S_slot): Howard's slot is the Legendre
        #   dual side, which carries the H-additive terms with a FLIPPED sign. Resolved
        #   EMPIRICALLY (not by reasoning alone) by the Howard-vs-Newton agreement gate
        #   test_integrated_howard_matches_newton_nonlq: with Newton as ground truth, s=-1 gives
        #   max-rel-error ~1e-5..5e-4 across V-only / f(m)-only / V+f, while s=+1 gives ~2.0
        #   (a different, wrong value function). The negated user running cost also corrects the
        #   #1247 Defect 2 sign flip (it previously entered Howard's slot un-negated).
        potential = getattr(H_class, "_potential", None)
        coupling = getattr(H_class, "_coupling", None)
        # Issue #1991: the Howard branch must consume the MMS source too. It was read only in the
        # Newton branch, so `solve_hjb_system(source_term=...)` on this path discarded it BITWISE
        # -- measured, |U(source) - U(no source)| = 0.000e+00 at two resolutions, with the Newton
        # path as a positive control at 7.13e-01. The capability gate keys on the parameter NAME,
        # so accepting the name while dropping the argument converts the gate's false negative
        # into a false positive: exactly the silent-wrong-answer #1424 exists to prevent.
        mms_src = self._mms_source_fn
        has_H_extra = potential is not None or coupling is not None

        howard_running_cost = None
        if has_H_extra or mms_src is not None:
            colloc_pts = self.collocation_points
            p_zero = np.zeros((self.n_points, self.dimension))

            def howard_running_cost(t_idx):
                rc = np.zeros(self.n_points)
                if has_H_extra:
                    # V(x, t) + f(m^n) via the SAME eval_H_batch the Newton residual uses
                    # (single source). At p=0 the gate-enforced unit-quadratic control cost
                    # gives H_control(0)=0, so eval_H_batch(..., p=0, ...) == V(x, t) + f(m^n).
                    rc = (
                        rc
                        + np.asarray(
                            eval_H_batch(H_class, colloc_pts, M_collocation[t_idx], p_zero, t_idx * dt),
                            dtype=float,
                        ).ravel()
                    )
                if mms_src is not None:
                    # `_mms_source_fn` already returns -S in the Newton slot's convention, and the
                    # `-rc` below applies Howard's own flip, so it enters here un-negated exactly
                    # in the slot the retired user running cost used.
                    rc = rc + np.asarray(mms_src(t_idx), dtype=float).ravel()
                return -rc  # rc_t = -(V + f(m) + S_mms); see SIGN note above.

        # Issue #1071: the control-cost Lagrangian L(alpha) for the policy-evaluation RHS comes
        # from the single source (control_cost.lagrangian), not a hardcoded (1/2)|alpha|^2. The
        # gate above guarantees a QuadraticControlCost, so L(alpha) = lambda/2 |alpha|^2.
        control_lagrangian = control_cost.lagrangian if control_cost is not None else None
        howard = HJBHowardSolver(
            self.problem,
            stencil_provider=self,
            alpha_star=alpha_star,
            running_cost=howard_running_cost,
            control_lagrangian=control_lagrangian,
            discretisation="central" if self.dimension == 1 else "upwind_projection",
            volatility_field=self._sigma_for_assembly(),
            use_provider_bc_rows=True,  # Issue #1118 PR2a: shared value-form BC rows
        )
        return howard.solve_hjb_system(M_collocation, U_terminal_colloc)

    def _solve_timestep(
        self,
        u_n_plus_1: np.ndarray,
        m_n_plus_1: np.ndarray,
        time_idx: int,
        additive_source: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve HJB at one time step using Newton iteration with backtracking line search.

        Globalization: each Newton iteration tries the full step `delta_u = -J⁻¹·r`,
        then accepts iff sufficient-decrease in residual norm holds (Armijo with
        c₁=1e-4). Otherwise halves α (geometric backtracking) until accepted or
        α drops below `min_alpha=1e-6`. Replaces the legacy hardcoded `max_step=10`
        cap, which prevented Newton from converging on stiff problems with
        |U|=O(100) (e.g., 2D MFG with strong potential — observed 0/150 timesteps
        converging to 1e-6 tolerance at high Pe).

        Reference: Nocedal-Wright "Numerical Optimization" §3.1 (Armijo condition).
        """
        from scipy.sparse.linalg import spsolve

        u_current = u_n_plus_1.copy()

        # Path selection for HJB residual/Jacobian computation (Issue #766, #775):
        #
        # 1. Hamiltonian batch path (use_hamiltonian_batch=True):
        #    Active when: hamiltonian_class is available AND qp_optimization_level="none"
        #    Uses batch H(x, m, p, t) and H.dp(x, m, p, t) from HamiltonianBase
        #    Works with any Hamiltonian subclass (SeparableHamiltonian, etc.)
        #    Fast: numpy vectorized over all collocation points at once
        #
        # 2. Per-point path (fallback):
        #    Active when: NOT the batch path -- QP mode enabled, is_custom=True without a
        #    hamiltonian_class, or no hamiltonian_class at all. Calls problem.H() per-point
        #    over collocation points. (Issue #1071 Phase 5 retired the dead legacy-LQ
        #    vectorized residual branch that ran only for is_custom=False mock problems.)
        H_class = getattr(self.problem, "hamiltonian_class", None)
        use_hamiltonian_batch = H_class is not None and self.qp_optimization_level == "none"

        # Warn if f_potential is set but won't be used (Issue #766)
        if not self._f_potential_warned:
            f_pot = getattr(self.problem, "f_potential", None)
            if f_pot is not None and np.any(f_pot != 0):
                warnings.warn(
                    "f_potential is set but will be ignored because the per-point "
                    "Hamiltonian path is active (is_custom=True or QP mode). "
                    "The potential V(x) comes from the Hamiltonian class instead. "
                    "Use SeparableHamiltonian(potential=...) to set the potential. "
                    "See Issue #766.",
                    UserWarning,
                    stacklevel=2,
                )
                self._f_potential_warned = True

        # Compute actual time for batch Hamiltonian calls
        current_time = time_idx * (self.problem.T / self.problem.Nt)

        # Closure: compute residual norm at any candidate u_trial (used by
        # backtracking line search). Routes through the same path-selection as
        # the Newton step itself, so residual evaluation is consistent.
        def _residual_norm(u_trial: np.ndarray) -> float:
            if use_hamiltonian_batch:
                g_u, l_u = self._compute_derivatives_vectorized(u_trial)
                r = self._compute_hjb_residual_hamiltonian(
                    u_trial,
                    u_n_plus_1,
                    m_n_plus_1,
                    g_u,
                    l_u,
                    H_class,
                    current_time,
                    additive_source=additive_source,
                )
            else:
                derivs = self._approximate_all_derivatives_cached(u_trial)
                r = self._compute_hjb_residual_with_cache(
                    u_trial,
                    u_n_plus_1,
                    m_n_plus_1,
                    time_idx,
                    derivs,
                    additive_source=additive_source,
                )
            return float(np.linalg.norm(r))

        # Armijo backtracking parameters (Nocedal-Wright §3.1)
        ARMIJO_C1 = 1e-4  # sufficient-decrease constant
        BACKTRACK_FACTOR = 0.5  # geometric step reduction
        MIN_ALPHA = 1e-6  # give up below this α

        for _newton_iter in range(self.max_newton_iterations):
            if use_hamiltonian_batch:
                # Batch Hamiltonian path: H(x, m, p, t) vectorized (Issue #775)
                grad_u, lap_u = self._compute_derivatives_vectorized(u_current)

                residual = self._compute_hjb_residual_hamiltonian(
                    u_current,
                    u_n_plus_1,
                    m_n_plus_1,
                    grad_u,
                    lap_u,
                    H_class,
                    current_time,
                    additive_source=additive_source,
                )

                if np.linalg.norm(residual) < self.newton_tolerance:
                    break

                jacobian_sparse = self._compute_hjb_jacobian_hamiltonian(
                    grad_u,
                    m_n_plus_1,
                    H_class,
                    current_time,
                )
            else:
                # Per-point path for QP mode or legacy custom without hamiltonian_class
                all_derivs = self._approximate_all_derivatives_cached(u_current)

                residual = self._compute_hjb_residual_with_cache(
                    u_current,
                    u_n_plus_1,
                    m_n_plus_1,
                    time_idx,
                    all_derivs,
                    additive_source=additive_source,
                )

                if np.linalg.norm(residual) < self.newton_tolerance:
                    break

                jacobian_sparse = self._compute_hjb_jacobian_sparse(u_current, m_n_plus_1, time_idx, all_derivs)

            # Issue #1074: diagnostic DMP guard (no-op unless check_dmp=True). Numerically inert.
            self._maybe_warn_dmp(u_current)

            # Apply boundary conditions (sparse-aware).
            # `u_current` is needed so BC rows encode the Newton residual
            # F_bc(u_current) - target rather than just the target value
            # (Issue #1116 fix).
            jacobian_bc, residual_bc = self._apply_boundary_conditions_to_sparse_system(
                jacobian_sparse, residual, time_idx, u_current
            )

            # Newton update using sparse solver: solve J·δ = -r for δ
            try:
                delta_u = spsolve(jacobian_bc, -residual_bc)
            except Exception as e:
                # Fallback to dense solver
                logger.warning(f"Sparse solver failed in Newton iteration (using dense fallback): {e}")
                delta_u = np.linalg.lstsq(jacobian_bc.toarray(), -residual_bc, rcond=None)[0]

            # Backtracking line search (Armijo). The Newton direction `delta_u`
            # is a descent direction for `½‖r‖²`, but the natural step `α=1`
            # may overshoot on stiff/nonlinear problems. We search for the
            # largest α ∈ {1, 0.5, 0.25, ...} satisfying sufficient-decrease:
            #   ‖r(u + α·δ)‖² ≤ (1 − 2·c₁·α)·‖r(u)‖²
            # which simplifies (for descent direction) to
            #   ‖r(u + α·δ)‖ ≤ (1 − c₁·α)·‖r(u)‖
            # Replaces a hardcoded `max_step=10` cap that was too restrictive
            # for stiff problems with |U|=O(100) and too permissive elsewhere
            # (Issue: HJB Newton non-convergence at high Pe).
            r0_norm = float(np.linalg.norm(residual))
            alpha = 1.0
            u_trial = u_current + alpha * delta_u
            u_trial = self._apply_boundary_conditions_to_solution(u_trial, time_idx)
            r_trial_norm = _residual_norm(u_trial)
            # Guard against NaN/Inf from too-aggressive steps
            if not np.isfinite(r_trial_norm):
                r_trial_norm = float("inf")
            while r_trial_norm > (1.0 - ARMIJO_C1 * alpha) * r0_norm and alpha > MIN_ALPHA:
                alpha *= BACKTRACK_FACTOR
                u_trial = u_current + alpha * delta_u
                u_trial = self._apply_boundary_conditions_to_solution(u_trial, time_idx)
                r_trial_norm = _residual_norm(u_trial)
                if not np.isfinite(r_trial_norm):
                    r_trial_norm = float("inf")

            # Apply accepted update. If line search bottomed out (α<MIN_ALPHA)
            # we still apply the smallest tested step rather than zero, so
            # Newton makes some progress even when sufficient-decrease fails.
            u_current = u_trial

        return u_current

    def _compute_dH_dp_fd(
        self,
        point_idx: int,
        m_at_x: float,
        derivs: DerivativeTensors,
        time_idx: int | None = None,
    ) -> np.ndarray:
        """
        Compute dH/dp - analytical for standard LQ Hamiltonian, FD otherwise.

        For standard LQ Hamiltonian H = |∇u|²/(2λ), dH/dp = p/λ analytically.

        Args:
            point_idx: Collocation point index
            m_at_x: Density value at the point
            derivs: DerivativeTensors with current gradient/hessian
            time_idx: Time index (currently unused, kept for API compatibility)

        Returns:
            dH/dp array, shape (dim,)
        """
        p = derivs.grad if derivs.grad is not None else np.zeros(self.problem.dimension)

        # Fast path: for standard LQ Hamiltonian H = |p|²/(2λ), dH/dp = p/λ.
        # Source λ from the canonical single source (``_control_cost_lambda`` ->
        # ``control_cost.lambda_``, falling back to ``problem.lambda_`` only when the
        # problem carries no Hamiltonian class) rather than reading ``problem.lambda_``
        # directly. This was the last direct ``problem.lambda_`` read in a physics path
        # -- the #1247 desync survivor on the per-point FD-Jacobian fast path (#1071
        # Phase 4). On every reachable construction the two agree (the divergent case is
        # unreachable: ``not is_custom`` implies the problem carries no control cost), so
        # this is byte-identical single-source hygiene, not a behaviour change.
        is_custom = getattr(self.problem, "is_custom", False)
        if not is_custom:
            return p / self._control_cost_lambda()

        # Fallback: finite differences for custom Hamiltonians using scipy
        x_pos = self.collocation_points[point_idx]
        hess = derivs.hess if derivs.hess is not None else np.zeros((len(p), len(p)))

        def H_of_p(p_vec: np.ndarray) -> float:
            """Hamiltonian as function of momentum p only."""
            from mfgarchon.core.derivatives import DerivativeTensors

            d = DerivativeTensors.from_arrays(grad=p_vec, hess=hess)
            return self.problem.H(point_idx, m_at_x, derivs=d, x_position=x_pos)

        # Use scipy's approx_fprime for gradient computation
        return approx_fprime(p, H_of_p, epsilon=1e-7)

    def _apply_boundary_conditions_to_solution(self, u: np.ndarray, time_idx: int) -> np.ndarray:
        """Apply boundary conditions directly to solution array.

        For mixed BC (per-point types), enforces Dirichlet at exit points only.
        """
        if len(self.boundary_indices) == 0:
            return u

        # Check if using per-point BC (mixed BC)
        # Issue #527: Replace hasattr with try/except per CLAUDE.md guidelines
        try:
            use_per_point_bc = self.boundary_conditions.is_mixed
        except AttributeError:
            use_per_point_bc = False

        # Use unified BC config (single source of truth) when using new infrastructure
        if self._use_new_infrastructure and self._bc_config is not None:
            global_bc_type = self._bc_config["type"]
            bc_values = self._bc_config["values"]
        else:
            # Get BC type - will raise error if not specified
            bc_type_val = self._get_boundary_condition_property("type")
            global_bc_type = bc_type_val.lower() if isinstance(bc_type_val, str) else bc_type_val
            bc_values = self._get_boundary_condition_property("value")

        # For per-point BC, apply Dirichlet only at exit points
        if use_per_point_bc:
            for i in self.boundary_indices:
                bc_type = self._get_bc_type_for_point(i)
                if bc_type == "dirichlet":
                    if callable(bc_values):
                        current_time = self.problem.T * time_idx / self.problem.Nt
                        u[i] = bc_values(self.collocation_points[i], current_time)
                    else:
                        u[i] = float(bc_values) if bc_values else 0.0
        elif global_bc_type == "dirichlet":
            # Uniform Dirichlet: apply to all boundary points
            if callable(bc_values):
                current_time = self.problem.T * time_idx / self.problem.Nt
                for i in self.boundary_indices:
                    u[i] = bc_values(self.collocation_points[i], current_time)
            else:
                u[self.boundary_indices] = bc_values
        # For Neumann: no direct solution modification (enforced via residual)

        return u

    def _apply_boundary_conditions_to_system(
        self, jacobian: np.ndarray, residual: np.ndarray, time_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply boundary conditions to the linear system J·δu = -R.

        For Dirichlet BC: Set row to identity and residual to zero.
        For Neumann BC: Row Replacement with normal derivative operator.
        """
        if len(self.boundary_indices) == 0:
            return jacobian, residual

        jacobian_bc = jacobian.copy()
        residual_bc = residual.copy()

        # Use new infrastructure with DirectCollocationHandler
        if self._use_new_infrastructure and self._bc_handler is not None:
            self._bc_handler.apply_to_matrix(
                A=jacobian_bc,
                b=residual_bc,
                boundary_indices=self.boundary_indices,
                operator=self._gfdm_operator,
                bc_config=self._bc_config,
            )
            return jacobian_bc, residual_bc

        # Legacy path (deprecated) - BC required if boundary points exist
        bc_type_val = self._get_boundary_condition_property("type")
        bc_type = bc_type_val.lower() if isinstance(bc_type_val, str) else bc_type_val

        if bc_type == "dirichlet":
            for i in self.boundary_indices:
                jacobian_bc[i, :] = 0.0
                jacobian_bc[i, i] = 1.0
                residual_bc[i] = 0.0

        return jacobian_bc, residual_bc


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing HJBGFDMSolver...")

    import numpy as np

    from mfgarchon import MFGProblem
    from mfgarchon.geometry import TensorProductGrid

    # Test 1D problem with uniform collocation points matching problem grid
    geometry_1d = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21])
    problem_1d = MFGProblem(geometry=geometry_1d, T=1.0, Nt=10, sigma=0.1)

    # Use problem grid points as collocation points to avoid index mismatch
    collocation_points = problem_1d.geometry.get_spatial_grid().reshape(-1, 1)

    solver_1d = HJBGFDMSolver(
        problem_1d,
        collocation_points=collocation_points,
        delta=0.15,
        taylor_order=2,
        weight_function="wendland",
    )

    # Test solver initialization
    assert solver_1d.dimension == 1
    assert solver_1d.n_points == problem_1d.geometry.num_spatial_points
    assert solver_1d.delta == 0.15
    assert solver_1d.taylor_order == 2
    assert solver_1d.hjb_method_name == "GFDM"
    print("  [1D] Solver initialized")
    print(f"       Collocation points: {solver_1d.n_points}, Delta: {solver_1d.delta}")

    # Test derivative computation API (1D)
    # f(x) = x^2 -> df/dx = 2x, d²f/dx² = 2
    x = collocation_points[:, 0]
    u_1d = x**2

    # Test compute_all_derivatives
    all_derivs_1d = solver_1d.compute_all_derivatives(u_1d)
    assert len(all_derivs_1d) == solver_1d.n_points
    # Interior points should have derivatives
    mid_idx = solver_1d.n_points // 2
    assert (1,) in all_derivs_1d[mid_idx], f"Missing gradient key (1,) at point {mid_idx}"
    print(f"  [1D] compute_all_derivatives: {len(all_derivs_1d)} points")
    print(f"       Multi-indices: {solver_1d.multi_indices}")

    # Test 2D problem
    print("\n  [2D] Testing 2D solver...")

    # Create 2D collocation points (grid)
    Nx_2d = 10
    x_grid = np.linspace(0, 1, Nx_2d)
    y_grid = np.linspace(0, 1, Nx_2d)
    xx, yy = np.meshgrid(x_grid, y_grid)
    points_2d = np.column_stack([xx.ravel(), yy.ravel()])

    geometry_2d = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[Nx_2d, Nx_2d])
    problem_2d = MFGProblem(geometry=geometry_2d, T=1.0, Nt=5, sigma=0.1)

    solver_2d = HJBGFDMSolver(
        problem_2d,
        collocation_points=points_2d,
        delta=0.2,
        taylor_order=2,
        weight_function="wendland",
    )
    print(f"       Collocation points: {solver_2d.n_points}, Delta: {solver_2d.delta}")
    print(f"       Multi-indices: {solver_2d.multi_indices}")

    # f(x,y) = x² + y² -> gradient = [2x, 2y], laplacian = 4
    u_2d = points_2d[:, 0] ** 2 + points_2d[:, 1] ** 2

    # Test compute_all_derivatives
    all_derivs_2d = solver_2d.compute_all_derivatives(u_2d)
    assert len(all_derivs_2d) == solver_2d.n_points

    # Find interior point (center of grid)
    mid_idx_2d = 55  # Center of 10x10 grid
    derivs_mid = all_derivs_2d[mid_idx_2d]
    print(f"  [2D] Derivatives at interior point {mid_idx_2d}:")
    print(f"       Keys: {list(derivs_mid.keys())}")

    # Check expected derivatives for f(x,y) = x² + y² at interior point
    if derivs_mid:
        grad_x = derivs_mid.get((1, 0), 0.0)
        grad_y = derivs_mid.get((0, 1), 0.0)
        lap_xx = derivs_mid.get((2, 0), 0.0)
        lap_yy = derivs_mid.get((0, 2), 0.0)
        print(f"       du/dx = {grad_x:.4f} (expected: {2 * points_2d[mid_idx_2d, 0]:.4f})")
        print(f"       du/dy = {grad_y:.4f} (expected: {2 * points_2d[mid_idx_2d, 1]:.4f})")
        print(f"       d²u/dx² = {lap_xx:.4f} (expected: 2.0)")
        print(f"       d²u/dy² = {lap_yy:.4f} (expected: 2.0)")

    print("\nNote: For gradient/laplacian utilities, use mfgarchon.utils.numerical.gfdm_operators")
    print("Smoke tests passed!")

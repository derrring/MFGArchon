"""
MFG solver method configurations.

This module provides configuration classes for all numerical methods used in MFG solvers.
The organization is by method type, not by equation (HJB/FP), since MFG can be solved
via various formulations beyond the classical HJB-FP coupled system.

Method Categories
-----------------
- Numerical: FDM, FEM, GFDM, Semi-Lagrangian, WENO
- Particle: SDE sampling, Monte Carlo
- Iteration: Newton, Picard (in core.py)
- Future: Neural (PINN, DGM), RL (DDPG, TD3)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from mfgarchon.config.core import BaseConfig

# =============================================================================
# ITERATION METHODS
# =============================================================================


class NewtonConfig(BaseConfig):
    """
    Newton iteration configuration for nonlinear solvers.

    Attributes
    ----------
    max_iterations : int
        Maximum number of Newton iterations (default: 10)
    tolerance : float
        Convergence tolerance (default: 1e-6)
    relaxation : float
        Relaxation parameter for Newton updates (default: 1.0)
    """

    max_iterations: int = Field(default=10, ge=1)
    tolerance: float = Field(default=1e-6, gt=0)
    relaxation: float = Field(default=1.0, gt=0, le=1.0)


# =============================================================================
# FINITE DIFFERENCE METHODS
# =============================================================================


class FDMConfig(BaseConfig):
    """
    Finite Difference Method configuration.

    Attributes
    ----------
    scheme : Literal["central", "upwind", "lax_friedrichs"]
        Spatial discretization scheme (default: upwind)
    time_stepping : Literal["explicit", "implicit", "crank_nicolson"]
        Time stepping method (default: implicit)
    """

    scheme: Literal["central", "upwind", "lax_friedrichs"] = "upwind"
    time_stepping: Literal["explicit", "implicit", "crank_nicolson"] = "implicit"


# =============================================================================
# FINITE ELEMENT METHODS
# =============================================================================


class FEMConfig(BaseConfig):
    """
    Finite Element Method configuration.

    FEM solves PDEs on unstructured meshes (triangles in 2D, tetrahedra in 3D)
    using variational formulations. Assembly is delegated to scikit-fem (Issue #773).

    Attributes
    ----------
    element_order : Literal[1, 2]
        Lagrange element polynomial order: 1 (linear, P1) or 2 (quadratic, P2).
        P1 is sufficient for most MFG problems. P2 gives O(h^3) convergence.
    quadrature_order : int | None
        Gauss quadrature order for element integration.
        None selects automatically based on element_order (2*p+1 rule).
    time_stepping : Literal["implicit", "crank_nicolson"]
        Time stepping method for parabolic equations (default: implicit).
    lumped_mass : bool
        Use lumped (diagonal) mass matrix instead of consistent.
        Faster but reduces accuracy order by 1 (default: False).
    """

    element_order: Literal[1, 2] = 1
    quadrature_order: int | None = None
    time_stepping: Literal["implicit", "crank_nicolson"] = "implicit"
    lumped_mass: bool = False

    @model_validator(mode="after")
    def validate_quadrature_order(self) -> FEMConfig:
        """Auto-set quadrature order if not specified."""
        if self.quadrature_order is None:
            # 2p+1 rule: exact for products of basis functions
            self.quadrature_order = 2 * self.element_order + 1
        if self.quadrature_order < 1:
            raise ValueError("quadrature_order must be >= 1")
        return self


# =============================================================================
# MESHFREE / GFDM METHODS
# =============================================================================


class QPConfig(BaseConfig):
    """
    QP monotonicity enforcement configuration for GFDM.

    Controls quadratic programming constraints that ensure monotone schemes,
    important for viscosity solution convergence.

    Attributes
    ----------
    optimization_level : Literal["none", "auto", "always"]
        QP optimization level: "none" (no QP), "auto" (adaptive M-matrix check),
        "always" (force QP everywhere, for debugging).
    solver : Literal["osqp", "scipy"]
        QP solver backend: "osqp" (fast convex QP) or "scipy" (SLSQP).
    warm_start : bool
        Enable warm-starting for OSQP (2-3x speedup on similar QPs).
    constraint_mode : Literal["indirect", "hamiltonian"]
        Constraint type: "indirect" (Taylor coefficient constraints) or
        "hamiltonian" (direct dH/du_j >= 0, stricter).
    """

    optimization_level: Literal["none", "auto", "always"] = "none"
    solver: Literal["osqp", "scipy"] = "osqp"
    warm_start: bool = True
    constraint_mode: Literal["indirect", "hamiltonian"] = "indirect"


class NeighborhoodConfig(BaseConfig):
    """
    Neighborhood construction configuration for GFDM.

    Controls how local point clouds are built for each collocation point.

    Attributes
    ----------
    mode : Literal["radius", "knn", "hybrid"]
        Selection strategy: "radius" (all within delta), "knn" (k nearest),
        "hybrid" (delta with minimum k guarantee, most robust).
    k_neighbors : int | None
        Neighbor count for KNN/hybrid. Auto-computed from Taylor order if None.
    adaptive : bool
        Enable adaptive delta enlargement for points with insufficient neighbors.
    k_min : int | None
        Minimum neighbor count. Auto-computed from Taylor order if None.
    max_delta_multiplier : float
        Maximum delta growth factor for adaptive enlargement.
    """

    mode: Literal["radius", "knn", "hybrid"] = "hybrid"
    k_neighbors: int | None = None
    adaptive: bool = False
    k_min: int | None = None
    max_delta_multiplier: float = Field(default=5.0, gt=1.0)


class DerivativeConfig(BaseConfig):
    """
    Derivative approximation method configuration for GFDM.

    Attributes
    ----------
    method : Literal["taylor", "rbf"]
        Method: "taylor" (standard GFDM) or "rbf" (RBF-FD, better conditioning).
    rbf_kernel : str
        RBF kernel: "phs3" (r^3), "phs5" (r^5), "gaussian".
    rbf_poly_degree : int
        Polynomial augmentation degree for RBF-FD.
    """

    method: Literal["taylor", "rbf"] = "taylor"
    rbf_kernel: str = "phs3"
    rbf_poly_degree: int = Field(default=2, ge=0)


class BoundaryAccuracyConfig(BaseConfig):
    """
    Boundary accuracy techniques for GFDM (Issue #531).

    Attributes
    ----------
    local_coordinate_rotation : bool
        Rotate stencils at boundary to align with normal direction.
    ghost_nodes : bool
        Create mirrored ghost neighbors for Neumann BC enforcement.
    wind_dependent_bc : bool
        Only enforce BC when characteristics flow into boundary.
        Requires ghost_nodes=True.
    """

    local_coordinate_rotation: bool = False
    ghost_nodes: bool = False
    wind_dependent_bc: bool = False

    @model_validator(mode="after")
    def validate_wind_bc(self) -> BoundaryAccuracyConfig:
        """Wind-dependent BC requires ghost nodes."""
        if self.wind_dependent_bc and not self.ghost_nodes:
            raise ValueError("wind_dependent_bc=True requires ghost_nodes=True")
        return self


class GFDMConfig(BaseConfig):
    """
    Generalized Finite Difference Method (meshfree) configuration.

    GFDM is used for particle-based/meshfree discretizations, particularly
    useful for high-dimensional problems and complex geometries.

    Groups 31 constructor parameters into logical sub-configs (Issue #634).

    Attributes
    ----------
    delta : float
        Support radius for local cloud of points (default: 0.1)
    taylor_order : int
        Taylor expansion order (1 or 2, default: 2)
    weight_function : str
        Weight function type (default: "wendland")
    weight_scale : float
        Weight scale parameter (default: 1.0)
    qp : QPConfig
        QP monotonicity enforcement settings
    neighborhood : NeighborhoodConfig
        Neighborhood construction settings
    derivative : DerivativeConfig
        Derivative approximation method settings
    boundary_accuracy : BoundaryAccuracyConfig
        Boundary accuracy technique settings
    congestion_mode : Literal["additive", "multiplicative"]
        Hamiltonian density-velocity coupling mode
    """

    delta: float = Field(default=0.1, gt=0)
    taylor_order: int = Field(default=2, ge=1, le=2)
    weight_function: str = "wendland"
    weight_scale: float = Field(default=1.0, gt=0)
    qp: QPConfig = Field(default_factory=QPConfig)
    neighborhood: NeighborhoodConfig = Field(default_factory=NeighborhoodConfig)
    derivative: DerivativeConfig = Field(default_factory=DerivativeConfig)
    boundary_accuracy: BoundaryAccuracyConfig = Field(default_factory=BoundaryAccuracyConfig)
    congestion_mode: Literal["additive", "multiplicative"] = "additive"


# =============================================================================
# SEMI-LAGRANGIAN METHODS
# =============================================================================


class SLConfig(BaseConfig):
    """
    Semi-Lagrangian method configuration.

    Semi-Lagrangian methods follow characteristics backward in time,
    making them well-suited for convection-dominated problems.

    Attributes
    ----------
    interpolation_method : Literal["linear", "slinear", "nearest", "cubic", "quintic"]
        Interpolation for the semi-Lagrangian lookup. NOT every value is honoured at
        every dimension -- 1D honours only "linear" and "cubic" -- and the solver
        refuses the combinations it would otherwise silently downgrade (Issue #1809).
        "rbf" was accepted here and is not an interpolation method the solver has ever
        implemented; it reached the solver and returned linear.
        Interpolation method for foot-of-characteristic (default: cubic)
    rk_order : Literal[1, 2, 3, 4]
        Runge-Kutta order for characteristic tracing (default: 2)
    cfl_number : float
        CFL number for stability (default: 0.5)
    """

    interpolation_method: Literal["linear", "slinear", "nearest", "cubic", "quintic"] = "cubic"
    rk_order: Literal[1, 2, 3, 4] = 2
    cfl_number: float = Field(default=0.5, gt=0, le=1.0)


# =============================================================================
# HIGH-ORDER METHODS
# =============================================================================


class WENOConfig(BaseConfig):
    """
    WENO (Weighted Essentially Non-Oscillatory) scheme configuration.

    WENO schemes provide high-order accuracy while maintaining shock-capturing
    capability, useful for problems with discontinuities.

    Attributes
    ----------
    weno_order : Literal[3, 5, 7]
        WENO scheme order (default: 5)
    flux_splitting : Literal["lax_friedrichs", "roe", "local_lax"]
        Flux splitting method (default: lax_friedrichs)
    epsilon : float
        Smoothness indicator parameter (default: 1e-6)
    """

    weno_order: Literal[3, 5, 7] = 5
    flux_splitting: Literal["lax_friedrichs", "roe", "local_lax"] = "lax_friedrichs"
    epsilon: float = Field(default=1e-6, gt=0)


# =============================================================================
# PARTICLE METHODS
# =============================================================================


class ParticleConfig(BaseConfig):
    """
    Particle-based method configuration.

    Particle methods sample the density using stochastic differential equations
    and estimate the density via kernel density estimation (KDE) or directly
    on collocation points.

    Attributes
    ----------
    num_particles : int
        Number of particles to sample (default: 5000)
    kde_bandwidth : float | Literal["auto"]
        KDE bandwidth parameter (default: auto)
    normalization : Literal["none", "initial_only", "all"]
        Density normalization strategy (default: initial_only)
    mode : Literal["hybrid", "collocation"]
        Particle mode (default: hybrid):
        - hybrid: Sample particles, output to grid via KDE
        - collocation: Use external particles, output on particles (meshfree)
    external_particles : Any
        External collocation points (ndarray, required for collocation mode)
        NOTE: This field is excluded from serialization (use only programmatically)
    """

    num_particles: int = Field(default=5000, gt=0)
    kde_bandwidth: float | Literal["auto"] = "auto"
    normalization: Literal["none", "initial_only", "all"] = "initial_only"
    mode: Literal["hybrid", "collocation"] = "hybrid"
    external_particles: Any = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_collocation_mode(self) -> ParticleConfig:
        """Validate that external_particles is provided for collocation mode."""
        if self.mode == "collocation" and self.external_particles is None:
            raise ValueError(
                "external_particles must be provided when mode='collocation'. "
                "Note: external_particles must be set programmatically (not via YAML) "
                "as it contains numpy arrays."
            )
        if self.mode == "hybrid" and self.external_particles is not None:
            raise ValueError(
                "external_particles should not be provided for hybrid mode. "
                "Use mode='collocation' if you want to use external particles."
            )
        return self

    @model_validator(mode="after")
    def validate_kde_bandwidth(self) -> ParticleConfig:
        """Validate KDE bandwidth parameter."""
        if isinstance(self.kde_bandwidth, float) and self.kde_bandwidth <= 0:
            raise ValueError("kde_bandwidth must be positive when specified as float")
        return self


# =============================================================================
# NETWORK / GRAPH METHODS
# =============================================================================


class NetworkConfig(BaseConfig):
    """
    Network/graph-based method configuration.

    For MFG problems defined on graphs or networks (e.g., traffic networks).

    Attributes
    ----------
    discretization_method : Literal["finite_volume", "finite_element"]
        Discretization method on network (default: finite_volume)
    """

    discretization_method: Literal["finite_volume", "finite_element"] = "finite_volume"


# =============================================================================
# COMPOSITE SOLVER CONFIGS (for backward compatibility)
# =============================================================================


class HJBConfig(BaseConfig):
    """
    HJB solver configuration (composite).

    Selects and configures a method for solving the HJB equation component.

    Attributes
    ----------
    method : Literal["fdm", "fem", "gfdm", "semi_lagrangian", "weno"]
        Solver method (default: fdm)
    accuracy_order : int
        Numerical accuracy order (1-5, default: 2)
    boundary_conditions : Literal["dirichlet", "neumann", "periodic"]
        Boundary condition type (default: neumann)
    newton : NewtonConfig
        Newton iteration configuration
    fdm : FDMConfig | None
        FDM-specific configuration
    fem : FEMConfig | None
        FEM-specific configuration (Issue #773)
    gfdm : GFDMConfig | None
        GFDM-specific configuration
    sl : SLConfig | None
        Semi-Lagrangian configuration
    weno : WENOConfig | None
        WENO configuration
    """

    method: Literal["fdm", "fem", "gfdm", "semi_lagrangian", "weno"] = "fdm"
    accuracy_order: int = Field(default=2, ge=1, le=5)
    boundary_conditions: Literal["dirichlet", "neumann", "periodic"] = "neumann"
    newton: NewtonConfig = Field(default_factory=NewtonConfig)

    # Method-specific configs
    fdm: FDMConfig | None = None
    fem: FEMConfig | None = None
    gfdm: GFDMConfig | None = None
    sl: SLConfig | None = None
    weno: WENOConfig | None = None

    @model_validator(mode="after")
    def validate_method_config(self) -> HJBConfig:
        """Auto-populate method-specific config if not provided."""
        if self.method == "fdm" and self.fdm is None:
            self.fdm = FDMConfig()
        elif self.method == "fem" and self.fem is None:
            self.fem = FEMConfig()
        elif self.method == "gfdm" and self.gfdm is None:
            self.gfdm = GFDMConfig()
        elif self.method == "semi_lagrangian" and self.sl is None:
            self.sl = SLConfig()
        elif self.method == "weno" and self.weno is None:
            self.weno = WENOConfig()
        return self


class FPConfig(BaseConfig):
    """
    FP solver configuration (composite).

    Selects and configures a method for solving the Fokker-Planck equation component.

    Attributes
    ----------
    method : Literal["fdm", "fem", "particle", "network"]
        Solver method (default: particle)
    fdm : FDMConfig | None
        FDM-specific configuration
    fem : FEMConfig | None
        FEM-specific configuration (Issue #773)
    particle : ParticleConfig | None
        Particle-specific configuration
    network : NetworkConfig | None
        Network-specific configuration
    """

    method: Literal["fdm", "fem", "particle", "network"] = "particle"

    # Method-specific configs
    fdm: FDMConfig | None = None
    fem: FEMConfig | None = None
    particle: ParticleConfig | None = None
    network: NetworkConfig | None = None

    @model_validator(mode="after")
    def validate_method_config(self) -> FPConfig:
        """Auto-populate method-specific config if not provided."""
        if self.method == "fdm" and self.fdm is None:
            self.fdm = FDMConfig()
        elif self.method == "fem" and self.fem is None:
            self.fem = FEMConfig()
        elif self.method == "particle" and self.particle is None:
            self.particle = ParticleConfig()
        elif self.method == "network" and self.network is None:
            self.network = NetworkConfig()
        return self

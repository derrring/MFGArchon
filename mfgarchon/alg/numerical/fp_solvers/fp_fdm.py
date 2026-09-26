from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import scipy.sparse as sparse

from mfgarchon.backends.compat import has_nan_or_inf
from mfgarchon.geometry import BoundaryConditions
from mfgarchon.geometry.base import CartesianGrid
from mfgarchon.geometry.boundary.types import BCType
from mfgarchon.utils.deprecation import deprecated_parameter
from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.numerical import clip_nonnegative_or_raise
from mfgarchon.utils.pde_coefficients import fp_drift_coefficient

from .base_fp import BaseFPSolver
from .fp_fdm_time_stepping import (
    _get_bc_type,
)
from .fp_fdm_time_stepping import (
    solve_fp_nd_full_system as _solve_fp_nd_full_system,
)

logger = get_logger(__name__)

_NO_TIME_DERIVATIVE = 1e300
"""``dt`` for an assembly that wants the operator without the ``1/dt`` diagonal (#2338).

The scheme handlers write ``1/dt`` onto the diagonal because they assemble ``(I/dt + L)`` for one implicit step. A
value this large makes that term vanish to round-off rather than requiring a second assembly path, which is what the
adjoint claim needs: the claim is about ``L``, and ``I/dt`` is symmetric so it would cancel in the comparison anyway --
but only exactly, and only if both sides used the same ``dt``, which the HJB linearisation does not take.
"""

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfgarchon.geometry import TensorProductGrid

# Advection scheme options for FDM (2x2 naming convention)
# Format: {pde_form}_{spatial_scheme}
# - pde_form: "gradient" (v·∇m) or "divergence" (∇·(vm))
# - spatial_scheme: "centered" or "upwind"
AdvectionScheme = Literal[
    "gradient_centered",  # NON-conservative (v.grad m), oscillates for Peclet > 2
    "gradient_upwind",  # NON-conservative at no-flux walls (row sums=1/dt is not mass cons.), stable
    "divergence_centered",  # Conservative (telescoping flux), oscillates for Peclet > 2
    "divergence_upwind",  # Conservative (telescoping flux), stable [factory DEFAULT for FDM]
    # Legacy aliases (DEPRECATED, will be removed in v1.0.0)
    "centered",  # -> gradient_centered
    "upwind",  # -> gradient_upwind
    "flux",  # -> divergence_upwind
]


class FPFDMSolver(BaseFPSolver):
    """
    Finite Difference Method (FDM) solver for Fokker-Planck equations.

    Supports general FP equation: dm/dt + div(v*m) = (sigma^2/2) * Laplacian(m)

    Advection Scheme Options (2x2 classification):

        | Scheme             | PDE Form   | Spatial    | Conservative | Stable |
        |--------------------|------------|------------|--------------|--------|
        | gradient_centered  | v·grad(m)  | Central    | NO           | Pe<2   |
        | gradient_upwind    | v·grad(m)  | Upwind     | YES (rows)   | Always |
        | divergence_centered| div(v*m)   | Central    | YES (flux)   | Pe<2   |
        | divergence_upwind  | div(v*m)   | Upwind     | YES (flux)   | Always |

        **gradient_centered**: Non-conservative form with central differences.
            Second-order accurate but oscillates for Peclet > 2.
            Use to demonstrate why conservative schemes are needed.

        **gradient_upwind**: Non-conservative form with upwind differences.
            Row sums = 1/dt, which is NOT mass conservation: column sums are
            unbalanced at no-flux walls, so mass leaks (up to ~99.8% when the
            drift is wall-directed). Stable but first-order. Use divergence_* for
            no-flux FP; the factory routes FDM_UPWIND to divergence_upwind (#382).

        **divergence_centered**: Conservative form with centered flux averaging.
            Mass-conservative via flux telescoping. Oscillates for Peclet > 2.
            Demonstrates that conservation alone doesn't guarantee stability.

        **divergence_upwind** [default]: Conservative form with upwind flux selection.
            Mass-conservative via flux telescoping. Stable, first-order.
            Best choice for MFG: handles boundary fluxes correctly.

    Legacy Aliases (DEPRECATED, will be removed in v1.0.0):
        - "centered" -> "gradient_centered"
        - "upwind" -> "gradient_upwind"
        - "flux" -> "divergence_upwind"

    Numerical Scheme:
        - Implicit timestepping for stability
        - Central differences for diffusion terms
        - Supports periodic, Dirichlet, and no-flux boundary conditions

    Required Geometry Traits (Issue #596 Phase 2.2A):
        - SupportsLaplacian: Provides Δm operator for diffusion term (σ²/2) Δm

    Compatible Geometries:
        - TensorProductGrid (structured grids)
        - ImplicitDomain (SDF-based domains)
        - Any geometry implementing SupportsLaplacian

    Note:
        Advection operators currently use manual sparse matrix construction.
        Future work (Issue #597) will integrate trait-based advection operators.
    """

    # Scheme family trait for duality validation (Issue #580)
    from mfgarchon.alg.base_solver import SchemeFamily

    _scheme_family = SchemeFamily.FDM

    # BoundaryCapable protocol (Issue #1456): FP-FDM assembles Dirichlet / Neumann / no-flux /
    # periodic boundary rows; Robin has no stencil (fails loud — Issue #1250), and
    # Reflecting/Extrapolation are not field-BC types it handles.
    _SUPPORTED_BC_TYPES: frozenset = frozenset({BCType.DIRICHLET, BCType.NEUMANN, BCType.NO_FLUX, BCType.PERIODIC})

    #: Issue #1686: this family reads a NEUMANN segment's type and drops its value.
    #: On the FP side a Neumann value is a prescribed flux J.n = g, and no FP solver
    #: implements an inhomogeneous flux wall, so a non-zero g is refused rather than
    #: silently discarded. Flip this to True in the same commit that implements it.
    honors_inhomogeneous_neumann: bool = False

    def __init__(
        self,
        problem: Any,
        boundary_conditions: BoundaryConditions | None = None,
        advection_scheme: AdvectionScheme = "divergence_upwind",
    ) -> None:
        """
        Initialize FDM solver for Fokker-Planck equations.

        Parameters
        ----------
        problem : Any
            MFG problem definition
        boundary_conditions : BoundaryConditions | None
            Boundary condition specification (default: no-flux)
        advection_scheme : str
            Advection term discretization (default: "divergence_upwind").

            Scheme names:
            - "gradient_centered": v·grad(m), central diff. NOT mass-conservative at
              no-flux walls -- leaks even for pure diffusion (Issue #1075).
            - "gradient_upwind": v·grad(m), upwind. Also NOT mass-conservative at
              no-flux walls (point-value boundary Laplacian; Issue #1075).
            - "divergence_centered": div(v*m), centered flux. Mass-conservative (telescoping).
            - "divergence_upwind": div(v*m), upwind flux. Mass-conservative (telescoping). [DEFAULT]

            Legacy names (DEPRECATED, will be removed in v1.0.0):
            - "centered" -> gradient_centered
            - "upwind" -> gradient_upwind
            - "flux" -> divergence_upwind
        """
        import warnings

        super().__init__(problem)
        self.fp_method_name = "FDM"

        # Map legacy scheme names to new names
        scheme_aliases = {
            "centered": "gradient_centered",
            "upwind": "gradient_upwind",
            "flux": "divergence_upwind",
        }

        # Emit deprecation warning for legacy aliases
        if advection_scheme in scheme_aliases:
            new_name = scheme_aliases[advection_scheme]
            warnings.warn(
                f"advection_scheme='{advection_scheme}' is deprecated. "
                f"Use advection_scheme='{new_name}' instead. "
                f"Legacy aliases will be removed in v1.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            advection_scheme = new_name

        # Validate scheme name (only new names accepted after mapping)
        valid_schemes = {"gradient_centered", "gradient_upwind", "divergence_centered", "divergence_upwind"}
        if advection_scheme not in valid_schemes:
            raise ValueError(f"Invalid advection_scheme: '{advection_scheme}'. Valid options: {sorted(valid_schemes)}")

        self.advection_scheme = advection_scheme

        # Detect problem dimension first (inherited from BaseNumericalSolver, Issue #633)
        self.dimension = self._detect_dimension()

        # Validate geometry capabilities (Issue #596 Phase 2.2A)
        # FP solver requires Laplacian operator for diffusion term
        from mfgarchon.geometry.protocols import SupportsLaplacian

        if not isinstance(problem.geometry, SupportsLaplacian):
            raise TypeError(
                f"FP FDM solver requires geometry with SupportsLaplacian trait for diffusion term. "
                f"{type(problem.geometry).__name__} does not implement this trait. "
                f"Compatible geometries: TensorProductGrid, ImplicitDomain."
            )

        # Boundary condition resolution hierarchy:
        # Issue #543 Phase 2: Replace hasattr with try/except cascade
        # Issue #527: Align with centralized BC resolution from BaseMFGSolver
        # 1. Explicit boundary_conditions parameter (highest priority)
        # 2. Problem components BC (if available and not None)
        # 3. geometry.boundary_conditions (attribute) - standard path
        # 4. geometry.get_boundary_conditions() (method accessor)
        # 5. Grid geometry boundary handler (legacy, if available)
        # 6. Default no-flux BC (fallback)
        if boundary_conditions is not None:
            self.boundary_conditions = boundary_conditions
        else:
            bc_found = False

            # Try components BC
            try:
                if problem.components is not None and problem.components.boundary_conditions is not None:
                    self.boundary_conditions = problem.components.boundary_conditions
                    bc_found = True
            except AttributeError:
                pass  # No components attribute, continue to next option

            # Try geometry.boundary_conditions (standard path - Issue #527)
            if not bc_found:
                try:
                    bc = problem.geometry.boundary_conditions
                    if bc is not None:
                        self.boundary_conditions = bc
                        bc_found = True
                except AttributeError:
                    pass  # No boundary_conditions attribute

            # Try geometry.get_boundary_conditions() (method accessor - Issue #527)
            if not bc_found:
                try:
                    bc = problem.geometry.get_boundary_conditions()
                    if bc is not None:
                        self.boundary_conditions = bc
                        bc_found = True
                except AttributeError:
                    pass  # No get_boundary_conditions method

            # Try geometry BC handler (legacy support)
            if not bc_found:
                try:
                    self.boundary_conditions = problem.geometry.get_boundary_handler()
                    bc_found = True
                except AttributeError:
                    pass  # No geometry BC handler

            # Default to no-flux if no BC found
            if not bc_found:
                from mfgarchon.geometry.boundary import no_flux_bc

                self.boundary_conditions = no_flux_bc(dimension=self.dimension)

        # This resolution shadows BaseMFGSolver.boundary_conditions, which is where a periodic BC
        # normally picks up the node layout of the grid being solved on (#1822). Without this line
        # a BC handed to the constructor -- or reached through components -- keeps an unstated
        # convention and wraps the historical way, while the same solve through the geometry gets
        # the grid's layout: measured 8.7e-02 against 9.3e-03 of heat-kernel error on one problem,
        # from one library, depending only on which channel supplied the BC. Applied once, after
        # both branches, so no channel can be added below it and miss it.
        self.boundary_conditions = self._with_geometry_periodic_convention(self.boundary_conditions)

        # Issue #1456: fail loud now if the resolved BC requests a type FP-FDM cannot honor
        # (Robin has no stencil; Reflecting/Extrapolation are not field-BC types), instead of
        # silently assembling a default (no-flux) wall.
        self._validate_bc_support(self.boundary_conditions)

        # Issue #1075 / #2007: the non-conservative gradient (point-value) advection schemes do
        # NOT conserve mass at no-flux walls, because the boundary node uses the point-value
        # Neumann Laplacian (sigma^2/dx^2) rather than the finite-volume half-cell closure.
        #
        # THE LEAK IS DRIFT-DEPENDENT, and the previous string understated it by an order of
        # magnitude (#2007). It said "leaks O(1e-2), even with zero drift". Measured:
        #
        #   genuinely zero drift, stationary initial density   -1.7e-14   (no leak at all)
        #   wall-normal drift A = 0.7, D = 1/8, d = 1          -1.4e-1
        #   wall-normal drift A = 0.7, D = 1/8, d = 2          -8.9e-1
        #
        # So the 1.5e-2 figure was a TRANSIENT density, which the string did not say, and a reader
        # budgeting against O(1e-2) was off by 10x exactly where the scheme is used.
        #
        # There is a second defect this warning does not cover, and it is not a wall problem:
        # div(alpha m) = alpha.grad(m) + m div(alpha), and the gradient form drops the second term,
        # so even repointing the wall leaves a scheme that discretizes a different equation
        # (measured on a source-free instance, 5.81e-1 -> 8.02e-1, EOC -0.007 -> 0.108). #2007
        # recommends removing these schemes for that reason; this warning is not a substitute for
        # that decision, and `test_gradient_centered_still_available_and_leaks` records the
        # standing one to keep them explicitly selectable.
        #
        # The conservative path is a 'divergence_*' scheme. Warn once at construction.
        if self.advection_scheme.startswith("gradient") and (
            getattr(self.boundary_conditions, "is_uniform", False)
            and getattr(self.boundary_conditions, "type", None) == "no_flux"
        ):
            warnings.warn(
                f"advection_scheme='{self.advection_scheme}' uses the non-conservative "
                "gradient (point-value) form and does NOT conserve mass at no-flux walls. "
                "The loss is UNBOUNDED in the wall-normal drift -- it is not a tolerance you can "
                "budget for. Measured, n=81, sigma=0.3, T=0.5, drift normal at both walls, by cell "
                "Peclet v*dx/D: 0 -> +0.5%; 0.19 -> -23.6%; 0.89 -> -99.97%. At the last of those "
                "the returned density is a relaxed uniform field, not an under-resolved correct "
                "one. 'divergence_upwind' is 0.0000% at all three. (Earlier revisions of this "
                "warning quoted a single figure -- first 'O(1e-2), even with zero drift', then "
                "'-1.4e-1 at A = 0.7'. Both were true of their fixture and both read as a bounded "
                "error, which is the thing that is false -- Issue #2007.) The same form also drops "
                "the m*div(alpha) term of div(alpha m), so "
                "it does not discretize the FP operator even away from the wall. Use a "
                "'divergence_*' scheme (default 'divergence_upwind') for mass-conservative "
                "no-flux solves. See Issue #1075 and Issue #2007.",
                UserWarning,
                stacklevel=2,
            )

    # _detect_dimension() inherited from BaseNumericalSolver (Issue #633)

    def _log_cfl_diagnostic(self, volatility_field: float | np.ndarray | Callable | None = None) -> None:
        """Log CFL diagnostic for accuracy/convergence guidance (Issue #882, #1052).

        Issue #1052: log once at INFO per solver instance, subsequent calls at
        DEBUG. CFL parameters are static across Picard iterations.
        """
        try:
            dt = self.problem.dt
            dx = self.problem.geometry.get_grid_spacing()[0]
            volatility = volatility_field if volatility_field is not None else self.problem.volatility
            if not isinstance(volatility, (int, float)):
                return  # a per-point or callable volatility has no single diffusive CFL number
            sigma = float(volatility)
            # Diffusive CFL uses the PDE diffusion coefficient D = sigma^2/2 (single source:
            # diffusion_from_volatility, applied at D = 0.5 * sigma**2 below), not the bare sigma^2.
            cfl_diffusive = 0.5 * sigma**2 * dt / dx**2
            if cfl_diffusive > 0.5:
                log_fn = logger.debug if getattr(self, "_cfl_logged", False) else logger.info
                log_fn(
                    "CFL diagnostic (FP FDM): diffusive=%.2f (sigma=%.3g, dt=%.3g, dx=%.3g). "
                    "Implicit scheme is stable but accuracy may degrade for CFL >> 1.",
                    cfl_diffusive,
                    sigma,
                    dt,
                    dx,
                )
                self._cfl_logged = True
        except (AttributeError, IndexError, TypeError):
            pass  # Not enough info to compute CFL — skip silently

    def build_advection_operator(self, U: np.ndarray, coupling_coefficient: float | None = None) -> sparse.csr_matrix:
        """This solver's own advection operator at ``U``, for checking it against the HJB linearisation (#2338).

        The FDM pair is advertised as discretely adjoint (#580, #622, #707): the FP operator should be the transpose
        of `HJBFDMSolver.build_linearized_operator`. `BlockIterator(adjoint_verify=True)` checks that at runtime and
        needs this side of the comparison; before #2338 it asked every FP solver for a `_build_advection_matrix` that
        no FP solver has ever defined, so the check skipped silently for every configuration.

        The operator is the advection block alone: the same per-node stencil this solver's `advection_scheme` uses in
        `solve_timestep_full_nd` -- `_INTERIOR_HANDLERS[scheme]`, one owner, no arithmetic restated here -- evaluated
        with ``dt -> infinity`` so the ``1/dt`` diagonal drops out and ``sigma = 0`` so the diffusion block does. What
        remains is what the adjoint claim is about.

        **Rows.** Only the nodes this scheme assembles through its INTERIOR stencil: strictly interior nodes always,
        and a periodic wall, which dispatches the same way. A no-flux or Dirichlet wall is the boundary handlers'
        business and is left empty here, which costs the comparison nothing -- `build_linearized_operator` zeroes its
        wall rows by design (#1564), so those rows carry no claim on either side and the caller drops them.

        There is no ``time`` parameter, and that is a real asymmetry rather than an oversight: the interior
        handlers take none, so nothing in this assembly is time-dependent, while `build_linearized_operator` on the
        HJB side does take one. A caller comparing the two at a time-dependent boundary is comparing one operator
        that moved with time against one that did not (review of #2344).

        Args:
            U: Value function at one time level, grid-shaped or flat.
            coupling_coefficient: Drift coefficient ``c`` in ``alpha* = -c grad(U)``. Defaults to the problem's own,
                the same `fp_drift_coefficient` the time-stepping resolves. Raises for a Hamiltonian whose scalar
                drift is not ``-c grad(U)`` (#1542), which is a refusal to guess rather than a silent default.

        Returns:
            Sparse operator, ``(N, N)`` over the flattened grid, with unassembled rows empty.

        Raises:
            NotImplementedError: If this solver's `advection_scheme` has no interior handler.
        """
        from .fp_fdm_time_stepping import _INTERIOR_HANDLERS

        handler = _INTERIOR_HANDLERS.get(self.advection_scheme)
        if handler is None:
            raise NotImplementedError(
                f"advection_scheme={self.advection_scheme!r} has no interior stencil to build an operator from "
                f"(#2338). Known: {sorted(_INTERIOR_HANDLERS)}."
            )
        # `CartesianGrid` is the class that GUARANTEES both accessors -- `GeometryProtocol` declares neither, and
        # the constructor does not narrow it for us: it refuses only a geometry without `laplacian` and names
        # `ImplicitDomain` as compatible, which has `get_grid_shape` and no `get_grid_spacing` (review of #2344
        # corrected an earlier comment here that claimed the constructor had already refused such a geometry).
        # `base.py:852` states this isinstance as the way to ask.
        geometry = self.problem.geometry
        if not isinstance(geometry, CartesianGrid):
            raise NotImplementedError(
                f"build_advection_operator needs a structured grid: {type(geometry).__name__} is not a "
                f"`CartesianGrid`, so it guarantees no per-axis spacing for the stencil this delegates to (#2338)."
            )
        shape = tuple(geometry.get_grid_shape())
        spacing = tuple(geometry.get_grid_spacing())
        ndim = len(shape)
        total = int(np.prod(shape))
        u_flat = np.asarray(U, dtype=float).ravel()
        if coupling_coefficient is None:
            coupling_coefficient = fp_drift_coefficient(self.problem)
        # `self.boundary_conditions` and not `geometry.get_boundary_conditions()`: the constructor resolves the BC
        # through components, geometry and the periodic convention (#527), and every real assembly call passes the
        # resolved one. Reading the geometry here gave a second owner -- measured with a periodic BC passed to the
        # constructor over a no-flux geometry, the two disagree (review of #2344).
        boundary_conditions = self.boundary_conditions
        periodic = _get_bc_type(boundary_conditions) == "periodic"

        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        for flat_idx in range(total):
            multi_idx = tuple(int(i) for i in np.unravel_index(flat_idx, shape))
            interior = all(0 < i < n - 1 for i, n in zip(multi_idx, shape, strict=True))
            if not (interior or periodic):
                continue
            handler(
                rows, cols, values, flat_idx, multi_idx, shape, ndim,
                _NO_TIME_DERIVATIVE, 0.0, coupling_coefficient, spacing, u_flat, geometry, boundary_conditions,
            )  # fmt: skip
        return sparse.coo_matrix((values, (rows, cols)), shape=(total, total)).tocsr()

    @deprecated_parameter(param_name="tensor_diffusion_field", since="v0.17.0", replacement="volatility_field")
    @deprecated_parameter(param_name="volatility_matrix", since="v0.17.0", replacement="volatility_field")
    @deprecated_parameter(param_name="velocity_field", since="v0.18.6", replacement="drift_field")
    def solve_fp_system(
        self,
        M_initial: np.ndarray | None = None,
        drift_field: np.ndarray | Callable | None = None,
        volatility_field: float | np.ndarray | Callable | None = None,
        show_progress: bool | None = None,
        progress_callback: Callable[[int], None] | None = None,  # Issue #640
        # Deprecated parameter names for backward compatibility
        tensor_diffusion_field: np.ndarray | Callable | None = None,  # Issue #717: deprecated
        volatility_matrix: np.ndarray | Callable | None = None,  # Deprecated: use volatility_field
        # Deprecated: velocity_field renamed to drift_field (v0.18.6)
        velocity_field: np.ndarray | None = None,
        # Live second channel: value function U (solver forms alpha = -c*grad(U) internally).
        # v0.18.6 (#919) swapped the names -- U moved here from drift_field, which now means
        # the velocity alpha*. This is the rename's destination, not a legacy alias.
        potential_field: np.ndarray | None = None,
        # MMS verification support
        source_term: Callable | None = None,
    ) -> np.ndarray:
        """
        Solve FP system forward in time with general drift and diffusion support.

        Implements BaseFPSolver unified API for both drift and diffusion.
        Automatically routes to 1D or nD solver based on problem dimension.

        Parameters
        ----------
        M_initial : np.ndarray
            Initial density m₀(x). Shape: (Nx,) for 1D or (N1, N2, ...) for nD, where each
            Ni is a POINT count from `TensorProductGrid.Nx_points` (#2235)
        drift_field : np.ndarray or callable, optional
            Drift velocity specification (Issue #573):
            - None: Zero drift (pure diffusion)
            - np.ndarray: Drift velocity α*(t,x), shape (Nt+1, Nx) for 1D or (Nt+1, N1, N2, ...) for nD
              Caller computes α* = -∂_p H(x, ∇U, m) for their Hamiltonian:
                * Quadratic H = (1/2)|p|²: α* = -∇U
                * L1 control H = |p|: α* = -sign(∇U)
                * Quartic H = (1/4)|p|⁴: α* = -sign(∇U) |∇U|^(1/3)
                * Custom H: Any function of ∇U
            - Callable: Custom drift function α(t, x, m) -> drift_vector
              Signature: (t: float, x_coords: list, m: ndarray) -> ndarray
            Default: None
        volatility_field : float, np.ndarray, or callable, optional
            Volatility specification (unified API). Auto-detects scalar vs matrix:
            - None: Use problem.volatility
            - float: Constant isotropic volatility σ → D = σ²/2
            - (d,) array: Diagonal volatility [σ₀, σ₁, ...] → D = diag(σᵢ²)/2
            - (d, d) array: Full volatility matrix Σ → D = ΣΣᵀ/2
            - (*shape, d, d) array: Spatially varying Σ(x) → D(x) = Σ(x)Σ(x)ᵀ/2
            - Callable: State-dependent σ(t, x, m) or Σ(t, x, m)
            Default: None
        tensor_diffusion_field : DEPRECATED, use volatility_field with (d,d) array
        volatility_matrix : DEPRECATED, use volatility_field with (d,d) array
        show_progress : bool
            Whether to show progress bar

        Returns
        -------
        np.ndarray
            Density evolution. Shape: (Nt+1, Nx) for 1D or (Nt+1, N1, N2, ...) for nD.
            Measured: (5, 21) for Nx_points=[21] and (5, 13, 7) for [13, 7] (#2235)

        Examples
        --------
        Pure diffusion (heat equation):
        >>> M = solver.solve_fp_system(m0)

        MFG optimal control:
        >>> drift = -problem.compute_gradient(U_hjb) / problem.control_cost
        >>> M = solver.solve_fp_system(m0, drift_field=drift)

        Custom volatility coefficient:
        >>> M = solver.solve_fp_system(m0, drift_field=drift, volatility_field=0.5)

        Spatially varying volatility (higher at boundaries):
        >>> Nx = problem.geometry.get_grid_shape()[0]
        >>> x_grid = np.linspace(0, 1, Nx)
        >>> volatility_array = 0.1 + 0.2 * np.abs(x_grid - 0.5)
        >>> M = solver.solve_fp_system(m0, drift_field=drift, volatility_field=volatility_array)

        Spatiotemporal volatility (time and space dependent):
        >>> Nt, Nx = problem.Nt + 1, problem.geometry.get_grid_shape()[0]
        >>> volatility_field = np.zeros((Nt, Nx))
        >>> for t in range(Nt):
        ...     volatility_field[t, :] = 0.1 * (1 + 0.5 * t / Nt)  # Increasing over time
        >>> M = solver.solve_fp_system(m0, drift_field=drift, volatility_field=volatility_field)

        State-dependent volatility (porous medium equation):
        >>> def porous_medium(t, x, m):
        ...     return 0.1 * m  # Volatility proportional to density
        >>> M = solver.solve_fp_system(m0, volatility_field=porous_medium)

        Density-dependent volatility with drift:
        >>> def crowd_diffusion(t, x, m):
        ...     return 0.05 + 0.15 * (1 - m / np.max(m))  # Lower volatility in crowds
        >>> M = solver.solve_fp_system(m0, drift_field=drift, volatility_field=crowd_diffusion)

        Pure advection (zero volatility):
        >>> M = solver.solve_fp_system(m0, drift_field=drift, volatility_field=0.0)

        Anisotropic volatility (unified API):
        >>> # Diagonal volatility: faster horizontal diffusion
        >>> Sigma = np.diag([0.2, 0.05])  # σ_x=0.2, σ_y=0.05 → D = diag(0.02, 0.00125)
        >>> M = solver.solve_fp_system(m0, drift_field=drift, volatility_field=Sigma)
        >>>
        >>> # Or pass as 1D array (auto-converted to diagonal matrix):
        >>> M = solver.solve_fp_system(m0, drift_field=drift, volatility_field=[0.2, 0.05])

        Full tensor with cross-diffusion:
        >>> # 2x2 symmetric tensor
        >>> Sigma = np.array([[0.2, 0.05], [0.05, 0.1]])
        >>> M = solver.solve_fp_system(m0, drift_field=drift, tensor_diffusion_field=Sigma)

        State-dependent tensor diffusion:
        >>> def anisotropic_crowd(t, x, m):
        ...     # Reduce perpendicular diffusion in crowds
        ...     sigma_parallel = 0.2
        ...     sigma_perp = 0.05 * (1 - m / np.max(m))
        ...     return np.diag([sigma_parallel, sigma_perp])
        >>> M = solver.solve_fp_system(m0, tensor_diffusion_field=anisotropic_crowd)

        Non-quadratic Hamiltonians (Issue #573):

        Quadratic control (H = (1/2)|p|²):
        >>> U_hjb = hjb_solver.solve(M_density)
        >>> grad_U = problem.compute_gradient(U_hjb)
        >>> alpha_quadratic = -grad_U  # α* = -∇U for quadratic H
        >>> M = solver.solve_fp_system(m0, drift_field=alpha_quadratic)

        L1 control cost (H = |p|, minimal fuel):
        >>> U_hjb = hjb_solver.solve_hjb_L1(M_density)
        >>> grad_U = problem.compute_gradient(U_hjb)
        >>> alpha_L1 = -np.sign(grad_U)  # α* = -sign(∇U) for L1 H
        >>> M = solver.solve_fp_system(m0, drift_field=alpha_L1)

        Quartic control cost (H = (1/4)|p|⁴):
        >>> U_hjb = hjb_solver.solve_hjb_quartic(M_density)
        >>> grad_U = problem.compute_gradient(U_hjb)
        >>> alpha_quartic = -np.sign(grad_U) * np.abs(grad_U) ** (1/3)  # α* = -(∇U)^(1/3)
        >>> M = solver.solve_fp_system(m0, drift_field=alpha_quartic)
        """
        # Validate required parameter
        if M_initial is None:
            raise ValueError("M_initial is required")

        # Handle deprecated velocity_field -> drift_field (v0.18.6)
        if velocity_field is not None:
            if drift_field is not None:
                raise ValueError(
                    "Cannot specify both drift_field and velocity_field. "
                    "Use drift_field (velocity_field is deprecated)."
                )
            drift_field = velocity_field

        # Two live channels, not an alias pair: drift_field carries the velocity alpha*,
        # potential_field carries the value function U (this solver forms alpha = -c*grad(U)
        # internally). Passing both is ambiguous, which is what this refuses -- neither is
        # deprecated (#1771).
        if potential_field is not None:
            if drift_field is not None:
                raise ValueError(
                    "Cannot specify both drift_field and potential_field: they are different "
                    "objects, not two names for one. drift_field is the velocity alpha*; "
                    "potential_field is the value function U, from which this solver forms "
                    "alpha = -coupling_coefficient*grad(U) itself. Pass whichever you have."
                )
            # potential_field is U-potential — route through internal U path
            effective_U = potential_field
        elif drift_field is not None:
            # drift_field is velocity α*(t,x) — route through velocity path
            if isinstance(drift_field, np.ndarray):
                effective_U = None  # Not needed when velocity is provided directly
            elif callable(drift_field):
                # Custom drift function - Phase 2
                # Route to unified nD solver (works for all dimensions including 1D)
                return _solve_fp_nd_full_system(
                    m_initial_condition=M_initial,
                    U_solution_for_drift=None,
                    problem=self.problem,
                    boundary_conditions=self.boundary_conditions,
                    show_progress=show_progress,
                    backend=self.backend,
                    diffusion_field=volatility_field,
                    drift_field=drift_field,  # callable velocity → internal drift_field
                    advection_scheme=self.advection_scheme,
                    progress_callback=progress_callback,
                    source_term=source_term,
                )
            else:
                raise TypeError(f"drift_field must be np.ndarray or Callable, got {type(drift_field)}")
        else:
            # Zero drift (pure diffusion): create zero U field for internal use
            try:
                Nt = self.problem.Nt + 1
            except AttributeError as e:
                raise ValueError("Cannot infer time steps. Ensure problem has Nt attribute.") from e

            # Create zero U field with appropriate shape
            if self.dimension == 1:
                Nx_val = getattr(self.problem, "Nx", None)
                Nx = Nx_val + 1 if Nx_val is not None else self.problem.geometry.get_grid_shape()[0]
                effective_U = np.zeros((Nt, Nx))
            else:
                grid_shape = self.problem.geometry.get_grid_shape()
                effective_U = np.zeros((Nt, *grid_shape))

        # Handle deprecated tensor_diffusion_field → volatility_field
        # Track if input came from tensor-specific parameter (for callable routing)
        _from_tensor_param = False
        if tensor_diffusion_field is not None:
            if volatility_field is not None:
                raise ValueError(
                    "Cannot specify both volatility_field and tensor_diffusion_field. "
                    "Use volatility_field (tensor_diffusion_field is deprecated)."
                )
            volatility_field = tensor_diffusion_field
            _from_tensor_param = True

        # Handle deprecated volatility_matrix → volatility_field
        if volatility_matrix is not None:
            if volatility_field is not None:
                raise ValueError(
                    "Cannot specify both volatility_field and volatility_matrix. "
                    "Use volatility_field (volatility_matrix is deprecated)."
                )
            volatility_field = volatility_matrix
            _from_tensor_param = True

        # Unified volatility_field handling with auto-detection
        # Issue #717: volatility_field is the SDE volatility σ or Σ
        # The solver computes D = σ²/2 (scalar) or D = ΣΣᵀ/2 (matrix) internally
        if volatility_field is None:
            # Issue #1248: with no explicit override, use the problem's full SDE volatility
            # (scalar / per-point array / callable), routed through the same detection below.
            volatility_field = self.problem.volatility

        if volatility_field is None:
            # Defensive: a duck-typed problem without a volatility is malformed.
            raise ValueError(
                "No volatility specified: problem.volatility is None and no volatility_field "
                "override was passed to solve_fp_system."
            )
        if isinstance(volatility_field, (int, float)):
            # Constant isotropic volatility
            effective_sigma = float(volatility_field)
            is_tensor = False
        elif isinstance(volatility_field, np.ndarray):
            # Auto-detect: scalar field vs matrix volatility
            d = self.dimension
            if volatility_field.ndim == 2 and volatility_field.shape == (d, d):
                # Constant volatility matrix Σ (d × d)
                is_tensor = True
                effective_sigma = volatility_field
            elif volatility_field.ndim >= 2 and volatility_field.shape[-2:] == (d, d):
                # Spatially varying volatility Σ(x) with shape (*spatial, d, d)
                is_tensor = True
                effective_sigma = volatility_field
            elif volatility_field.ndim == 1 and len(volatility_field) == d:
                # Diagonal volatility [σ₀, σ₁, ...] → convert to diag matrix
                is_tensor = True
                effective_sigma = np.diag(volatility_field)
            else:
                # Scalar field (spatial or spatiotemporal varying σ)
                is_tensor = False
                effective_sigma = volatility_field
        elif callable(volatility_field):
            # State-dependent volatility - callable σ(t, x, m) or Σ(t, x, m)
            # Issue #641: Always route to unified nD solver (handles 1D too)
            effective_sigma = volatility_field
            # If came from tensor-specific deprecated param, route to tensor path
            # Otherwise, route to scalar path (runtime detection not yet implemented)
            is_tensor = _from_tensor_param
        else:
            raise TypeError(
                f"volatility_field must be None, float, np.ndarray, or Callable, got {type(volatility_field)}"
            )

        # CFL diagnostic (Issue #882)
        self._log_cfl_diagnostic(volatility_field)

        # Resolve velocity for internal routing:
        # drift_field (after deprecation handling) is velocity when effective_U is None
        _internal_velocity = drift_field if (effective_U is None and isinstance(drift_field, np.ndarray)) else None

        # Route tensor volatility to tensor path
        if is_tensor:
            if self.dimension == 1:
                raise NotImplementedError(
                    "Anisotropic volatility not yet implemented for 1D problems. Use scalar volatility_field for 1D."
                )
            # Route to nD solver with tensor volatility
            return _solve_fp_nd_full_system(
                m_initial_condition=M_initial,
                U_solution_for_drift=effective_U,
                problem=self.problem,
                boundary_conditions=self.boundary_conditions,
                show_progress=show_progress,
                backend=self.backend,
                diffusion_field=None,
                tensor_diffusion_field=effective_sigma,  # Internal API uses old name
                advection_scheme=self.advection_scheme,
                progress_callback=progress_callback,
                source_term=source_term,
                velocity_field=_internal_velocity,
            )

        # Issue #641: Always route to unified nD solver (works for all dimensions)
        # Internal API still uses diffusion_field name for backward compatibility
        return _solve_fp_nd_full_system(
            m_initial_condition=M_initial,
            U_solution_for_drift=effective_U,
            problem=self.problem,
            boundary_conditions=self.boundary_conditions,
            show_progress=show_progress,
            backend=self.backend,
            diffusion_field=effective_sigma if volatility_field is not None else None,
            advection_scheme=self.advection_scheme,
            progress_callback=progress_callback,
            source_term=source_term,
            velocity_field=_internal_velocity,
        )

    # =========================================================================
    # Strict Adjoint Mode (Issue #622)
    # =========================================================================

    def solve_fp_step_adjoint_mode(
        self,
        M_current: np.ndarray,
        A_advection_T: sparse.csr_matrix,
        sigma: float | np.ndarray | None = None,
        time: float = 0.0,
    ) -> np.ndarray:
        """
        Solve single FP timestep using externally provided advection matrix.

        This method is used in strict adjoint mode (Issue #622) where the
        FP solver uses A^T from the HJB solver instead of building its own
        advection matrix. This guarantees exact adjoint consistency:
            L_FP = L_HJB^T

        Mathematical Formulation:
            FP equation: dm/dt + ∇·(vm) = (σ²/2) Δm

            Discretized with A_advection_T (transpose of HJB's matrix):
                (I/dt + A_advection_T + D) m^{k+1} = m^k / dt

            where:
            - A_advection_T: Advection matrix from HJB solver (transposed)
            - D: Diffusion matrix (built internally; symmetric for a CONSTANT diffusion, and
              self-adjoint in the control-volume inner product in general -- see the Note)
            - I: Identity matrix

        Args:
            M_current: Current density at timestep k, shape (*spatial_shape)
            A_advection_T: Transposed advection matrix from HJB solver.
                Shape: (N_total, N_total) where N_total = prod(spatial_shape).
                This is A_hjb.T where A_hjb was built by HJBFDMSolver.build_advection_matrix().
            sigma: The SDE volatility (optional); the diffusion is D = sigma^2/2.
                - None: Use problem.volatility
                - float: Constant volatility
                - np.ndarray: Per-point volatility
            time: Current time for time-dependent BCs

        Returns:
            M_next: Density at timestep k+1, shape (*spatial_shape)

        Example:
            >>> # In FixedPointIterator with strict_adjoint=True:
            >>> A_hjb = hjb_solver.build_advection_matrix(U_current)
            >>> M_next = fp_solver.solve_fp_step_adjoint_mode(M_current, A_hjb.T)

        Note:
            ~~The diffusion operator is symmetric (D = D^T)~~ **[CORRECTED 2026-08-28, #2145]**.
            It is self-adjoint in the CONTROL-VOLUME inner product, `W D` symmetric with
            `W = diag(w)`, which is the statement a non-uniform control volume admits: the wall
            rows carry `2/h²` where their neighbours' columns carry `1/h²`, so plain `D = Dᵀ`
            would require the equal-volume mesh this grid is not. Measured on the varying-sigma
            path: `max|D - Dᵀ|` 0.0 before #2145, 197.02 after; `max|WD - (WD)ᵀ|` is 0.0.

            Using this method with A_hjb.T therefore gives `L_FP = L_HJB^T` for the advection
            part, while the diffusion half is adjoint-consistent in `L²(w)` rather than in the
            unweighted inner product. Independent review measured this whole path and found it
            conserves NEITHER measure, on this revision and on its predecessor alike (-13.54%
            rectangle, -15.29% trapezoid, byte-identical), because `A_HJB` has row sums of 60 --
            so the symmetry argument was not what was carrying it.

        See Also:
            - Issue #622: Strict Achdou adjoint mode implementation
            - HJBFDMSolver.build_advection_matrix(): Builds the advection matrix
            - FixedPointIterator: Orchestrates matrix passing between solvers
        """
        # Get problem dimensions
        shape = M_current.shape
        N_total = int(np.prod(shape))
        dt = self.problem.dt

        # Validate matrix shape
        if A_advection_T.shape != (N_total, N_total):
            raise ValueError(
                f"A_advection_T shape {A_advection_T.shape} doesn't match expected "
                f"({N_total}, {N_total}) for density shape {shape}"
            )

        # A scalar or per-point volatility; a callable is refused rather than evaluated at an
        # arbitrary time (#2376).
        sigma_val = self.problem.volatility if sigma is None else sigma
        if callable(sigma_val):
            raise NotImplementedError(
                "FPFDMSolver.solve_fp_step_adjoint_mode takes a scalar or per-point volatility, not a "
                "callable (#2376). With BlockIterator, adjoint_mode='off' solves the FP side through "
                "solve_fp_system, which evaluates a callable volatility per time step."
            )

        # Build diffusion matrix using LaplacianOperator
        from mfgarchon.operators.differential.laplacian import LaplacianOperator

        spacing = list(self.problem.geometry.get_grid_spacing())
        bc = self.boundary_conditions

        sigma_arr = np.asarray(sigma_val)
        varying_sigma = sigma_arr.ndim > 0 and float(np.ptp(sigma_arr)) > 1e-12

        # Build the diffusion matrix.
        if varying_sigma:
            # Issue #1183: per-point variable-coefficient diffusion -- bake the field
            # D(x) = sigma(x)^2/2 into a conservative finite-volume Laplacian (face-averaged
            # D_{i+1/2}, mass-conserving) rather than collapsing sigma to its mean.
            from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

            d_field = diffusion_from_volatility(sigma_arr, kind="field")
            diffusion_matrix = LaplacianOperator(
                spacings=spacing, field_shape=shape, bc=bc, mass_conservative=True, coefficient_field=d_field
            ).as_scipy_sparse()
        else:
            # Scalar (or uniform-array) sigma: scalar D times the (existing) Laplacian.
            scalar_sigma = float(sigma_arr.reshape(-1)[0]) if sigma_arr.ndim > 0 else float(sigma_val)
            D = 0.5 * scalar_sigma**2
            L_matrix = LaplacianOperator(spacings=spacing, field_shape=shape, bc=bc).as_scipy_sparse()
            diffusion_matrix = D * L_matrix

        # Build full system matrix: (I/dt + A_advection_T - diffusion). The Laplacian has a
        # negative diagonal, so the diffusion matrix is SUBTRACTED.
        identity = sparse.eye(N_total)
        A_system = identity / dt + A_advection_T - diffusion_matrix

        # Right-hand side
        b_rhs = M_current.ravel() / dt

        # Solve linear system
        M_next_flat = sparse.linalg.spsolve(A_system, b_rhs)

        # Reshape and ensure non-negativity.
        M_next = M_next_flat.reshape(shape)
        # Issue #1507: A_advection_T (the transposed HJB advection) is NOT an M-matrix, so at high
        # Péclet the solve undershoots negative; clipping to 0 ADDS mass. The caller stores this raw
        # with no mass check, so ∫m drifts up across timesteps and the coupled fixed point converges
        # self-consistently wrong. Renormalize to the pre-step total (the physical density conserves
        # mass; the operator-level adjoint A_FP=A_HJB^T is unchanged) and warn when the clip is
        # non-trivial -- matching fp_semi_lagrangian / fp_fdm_time_stepping (Issues #880/#886), so a
        # diverging solve is surfaced instead of silently reported as a valid conserved density.
        # Issue #1683: this used to clip and then renormalize to the pre-step total, which
        # made a diverging solve indistinguishable from a healthy one -- the result was
        # finite, non-negative and exactly mass-conserving, so every cheap check a caller
        # might run was satisfied by the repair rather than by the physics. Measured on
        # the #1507 configuration, an 8.39% clip came back reporting exact conservation.
        # The shared gate stops the solve instead, and does not renormalize.
        return clip_nonnegative_or_raise(
            M_next,
            context="Strict-adjoint FP step",
            remedy=(
                "The transposed HJB advection operator is not an M-matrix, so at high "
                "Péclet it undershoots negative. Reduce dt (increase Nt), refine the "
                "grid, or add diffusion (sigma > 0)."
            ),
        )

    # NOTE: _solve_fp_1d_with_callable_drift was removed in v0.17.1 (Issue #641)
    # It was dead code - never called from solve_fp_system().
    # Callable drift now routes directly to _solve_fp_nd_full_system() which
    # handles all dimensions including 1D.


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing FPFDMSolver...")
    print("=" * 60)

    from mfgarchon import MFGProblem
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    # Test 1D problem using geometry-based API (unified nD solver, Issue #641)
    print("\n1. Testing 1D FDM (advection schemes)...")

    # Create 1D grid with TensorProductGrid
    Nx = 40  # Number of cells (grid points = Nx + 1)
    grid_1d = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[Nx + 1])
    dx_1d = grid_1d.get_grid_spacing()[0]

    problem_1d = MFGProblem(
        geometry=grid_1d,
        Nt=25,
        T=1.0,
        volatility=0.1,
        coupling_coefficient=1.0,
    )

    # Create initial density (Gaussian) and drift field
    x = np.array(grid_1d.coordinates[0])  # 1D coordinates
    m_init_1d = np.exp(-((x - 0.5) ** 2) / 0.05)
    m_init_1d /= m_init_1d.sum() * dx_1d  # Normalize using sum*dx (consistent with 2D)

    # Create drift pushing mass to right (advection test)
    Nt = problem_1d.Nt + 1
    U_test_1d = np.zeros((Nt, Nx + 1))
    for t in range(Nt):
        U_test_1d[t] = -x  # Drift to the right (alpha = -dU/dx = +1)

    # Test gradient_upwind (gradient form: v·∇m)
    solver_1d_gu = FPFDMSolver(
        problem_1d,
        boundary_conditions=no_flux_bc(dimension=1),
        advection_scheme="gradient_upwind",
    )
    assert solver_1d_gu.dimension == 1
    assert solver_1d_gu.fp_method_name == "FDM"
    assert solver_1d_gu.advection_scheme == "gradient_upwind"

    M_1d_gu = solver_1d_gu.solve_fp_system(m_init_1d, U_test_1d, show_progress=False)
    assert M_1d_gu.shape == (Nt, Nx + 1)
    assert not has_nan_or_inf(M_1d_gu)

    # Test divergence_upwind (divergence form: ∇·(vm))
    solver_1d_du = FPFDMSolver(
        problem_1d,
        boundary_conditions=no_flux_bc(dimension=1),
        advection_scheme="divergence_upwind",
    )
    assert solver_1d_du.advection_scheme == "divergence_upwind"

    M_1d_du = solver_1d_du.solve_fp_system(m_init_1d, U_test_1d, show_progress=False)
    assert M_1d_du.shape == (Nt, Nx + 1)
    assert not has_nan_or_inf(M_1d_du)

    # Calculate mass drift for both (using sum*dx for consistency with 2D)
    initial_mass_1d = m_init_1d.sum() * dx_1d
    final_mass_1d_gu = M_1d_gu[-1].sum() * dx_1d
    final_mass_1d_du = M_1d_du[-1].sum() * dx_1d
    mass_drift_1d_gu = abs(final_mass_1d_gu - initial_mass_1d) / initial_mass_1d
    mass_drift_1d_du = abs(final_mass_1d_du - initial_mass_1d) / initial_mass_1d

    print(f"   Initial mass: {initial_mass_1d:.6f}")
    print(f"   gradient_upwind:   final={final_mass_1d_gu:.6f}, drift={mass_drift_1d_gu:.2%}")
    print(f"   divergence_upwind: final={final_mass_1d_du:.6f}, drift={mass_drift_1d_du:.2%}")

    # Test 2D problem with advection schemes
    print("\n2. Testing 2D FDM (advection schemes)...")

    # Create 2D problem
    grid_2d = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],  # [(xmin, xmax), (ymin, ymax)]
        Nx_points=[11, 11],  # (nx+1, ny+1) grid points
    )
    problem_2d = MFGProblem(
        geometry=grid_2d,
        Nt=20,
        T=0.5,
        volatility=0.2,
        coupling_coefficient=1.0,
    )

    # Create Gaussian initial density
    x, y = grid_2d.coordinates
    X, Y = np.meshgrid(x, y, indexing="ij")
    dx, dy = grid_2d.get_grid_spacing()
    cell_volume = dx * dy
    m_init_2d = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.05)
    m_init_2d /= m_init_2d.sum() * cell_volume

    # Create a drift field that pushes mass to corner (advection test)
    Nt = problem_2d.Nt + 1
    U_drift = np.zeros((Nt, *grid_2d.get_grid_shape()))
    # Potential U = -x - y (drift to upper-right corner)
    for t in range(Nt):
        U_drift[t] = -(X + Y)

    # Test gradient_upwind (non-conservative)
    solver_2d_gu = FPFDMSolver(
        problem_2d,
        boundary_conditions=no_flux_bc(dimension=2),
        advection_scheme="gradient_upwind",
    )
    M_2d_gu = solver_2d_gu.solve_fp_system(m_init_2d, U_drift, show_progress=False)

    # Test divergence_upwind (conservative)
    solver_2d_du = FPFDMSolver(
        problem_2d,
        boundary_conditions=no_flux_bc(dimension=2),
        advection_scheme="divergence_upwind",
    )
    M_2d_du = solver_2d_du.solve_fp_system(m_init_2d, U_drift, show_progress=False)

    # Calculate mass drift for both
    initial_mass_2d = m_init_2d.sum() * cell_volume

    final_mass_gu = M_2d_gu[-1].sum() * cell_volume
    mass_drift_gu = abs(final_mass_gu - initial_mass_2d) / initial_mass_2d

    final_mass_du = M_2d_du[-1].sum() * cell_volume
    mass_drift_du = abs(final_mass_du - initial_mass_2d) / initial_mass_2d

    print(f"   Initial mass: {initial_mass_2d:.6f}")
    print(f"   gradient_upwind:   final={final_mass_gu:.6f}, drift={mass_drift_gu:.2%}")
    print(f"   divergence_upwind: final={final_mass_du:.6f}, drift={mass_drift_du:.2%}")

    # Verify solutions are valid
    assert not has_nan_or_inf(M_2d_gu), "gradient_upwind solution has NaN/Inf"
    assert not has_nan_or_inf(M_2d_du), "divergence_upwind solution has NaN/Inf"
    assert np.all(M_2d_gu >= -1e-10), "gradient_upwind: density should be non-negative"
    assert np.all(M_2d_du >= -1e-10), "divergence_upwind: density should be non-negative"

    # Verify advection_scheme is properly set
    assert solver_2d_gu.advection_scheme == "gradient_upwind"
    assert solver_2d_du.advection_scheme == "divergence_upwind"

    print("\n" + "=" * 60)
    print("All smoke tests passed!")

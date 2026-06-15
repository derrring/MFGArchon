"""
Fixed Point Iterator

Modern fixed-point iterator for MFG systems with full feature support:
- Config-based parameter management with backward compatibility
- Anderson acceleration for faster convergence
- Backend support (GPU/CPU)
- Structured SolverResult output (tuple output for legacy compatibility)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from mfgarchon.utils.deprecation import validate_kwargs
from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.solver_result import SolverResult

from .base_mfg import BaseCouplingIterator
from .fixed_point_utils import (
    check_convergence_criteria,
    compute_fp_velocity_field,
    initialize_cold_start,
    preserve_initial_condition,
    preserve_terminal_condition,
    resolve_fp_drift_kwargs,
)
from .source_composition import compose_fp_source, compose_hjb_source

logger = get_logger(__name__)

if TYPE_CHECKING:
    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfgarchon.config import MFGSolverConfig
    from mfgarchon.problem.base_mfg_problem import MFGProblem

# Type alias for iteration callback (Issue #614)
# Signature: callback(iteration, U, M, error_U, error_M) -> bool
IterationCallback = Callable[[int, np.ndarray, np.ndarray, float, float], bool | None]


class FixedPointIterator(BaseCouplingIterator):
    """
    Fixed-point iterator for MFG systems with full feature support.

    Features:
    - Config-based parameter management with backward compatibility
    - Optional Anderson acceleration for faster convergence
    - GPU/CPU backend support
    - Structured SolverResult output (tuple output for legacy compatibility)
    - Warm start support
    - State-dependent coefficients (Phase 2.3)

    Required Geometry Traits (Issue #596 Phase 2.3):
        This coupling solver requires trait-validated HJB and FP component solvers:
        - HJB solver must use geometry with SupportsGradient trait
        - FP solver must use geometry with SupportsLaplacian trait

        Trait validation occurs in component solvers, not at coupling layer.
        See HJBFDMSolver and FPFDMSolver docstrings for trait details.

    Args:
        problem: MFG problem definition
        hjb_solver: HJB solver instance (must be trait-validated)
        fp_solver: FP solver instance (must be trait-validated)
        config: Configuration object (preferred modern approach)
        use_anderson: Enable Anderson acceleration
        anderson_depth: Anderson acceleration memory depth
        anderson_beta: Anderson acceleration mixing parameter
        backend: Backend name ('numpy', 'torch', 'jax', etc.)
        volatility_field: Optional diffusion override (float, array, or callable)
            - None: Use problem.sigma (default)
            - float: Constant diffusion
            - ndarray: Spatially/temporally varying diffusion
            - Callable: State-dependent diffusion D(t, x, m) -> float | ndarray
        drift_field: Optional drift override for non-MFG problems (array or callable)
            - None: Use MFG drift (default, drift from U)
            - ndarray: Precomputed drift field
            - Callable: State-dependent drift α(t, x, m) -> ndarray
    """

    # v0.25.0 removal: 7 legacy `damping_*` kwargs removed per the 3-version
    # deprecation window (Issue #1070). Caught upfront in `__init__` via
    # `validate_kwargs` with a curated migration message instead of Python's
    # generic "unexpected keyword argument" — matches the mfgarchon
    # convention used by MFGProblem._validate_kwargs.
    _REMOVED_KWARGS: ClassVar[dict[str, str]] = {
        "damping_factor": "Use 'relaxation' instead (v0.25.0 removal, Issue #1070).",
        "damping_factor_M": "Use 'relaxation_M' instead (v0.25.0 removal, Issue #1070).",
        "adaptive_damping": "Use 'adaptive_relaxation' instead (v0.25.0 removal, Issue #1070).",
        "adaptive_damping_decay": "Use 'adaptive_relaxation_decay' instead (v0.25.0 removal, Issue #1070).",
        "adaptive_damping_min": "Use 'adaptive_relaxation_min' instead (v0.25.0 removal, Issue #1070).",
        "damping_schedule": "Use 'relaxation_schedule' instead (v0.25.0 removal, Issue #1070).",
        "damping_schedule_M": "Use 'relaxation_schedule_M' instead (v0.25.0 removal, Issue #1070).",
    }
    _RECOGNIZED_KWARGS: ClassVar[set[str]] = {
        "config",
        "relaxation",
        "relaxation_M",
        "use_anderson",
        "anderson_depth",
        "anderson_beta",
        "backend",
        "volatility_field",
        "drift_field",
        "adaptive_relaxation",
        "adaptive_relaxation_decay",
        "adaptive_relaxation_min",
        "relaxation_schedule",
        "relaxation_schedule_M",
    }

    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
        config: MFGSolverConfig | None = None,
        relaxation: float = 0.5,
        relaxation_M: float | None = None,
        use_anderson: bool = False,
        anderson_depth: int = 5,
        anderson_beta: float = 1.0,
        backend: str | None = None,
        volatility_field: float | np.ndarray | Any | None = None,  # Phase 2.3
        drift_field: np.ndarray | Any | None = None,  # Phase 2.3
        adaptive_relaxation: bool = False,
        adaptive_relaxation_decay: float = 0.5,
        adaptive_relaxation_min: float = 0.05,
        relaxation_schedule: str = "constant",
        relaxation_schedule_M: str | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            relaxation: Under-relaxation factor for U (omega_U) in (0, 1]. Default 0.5.
            relaxation_M: Under-relaxation factor for M (omega_M). If None, uses `relaxation`
                for both. Issue #719: Per-variable relaxation support.
                Recommended for MFG: relaxation=1.0, relaxation_M=0.2 (U adapts fully,
                M filters particle noise).
        """
        # Reject removed kwargs with a curated migration message; warn on
        # unrecognized kwargs that may be typos. See _REMOVED_KWARGS above.
        if kwargs:
            validate_kwargs(
                kwargs=kwargs,
                deprecated_kwargs=self._REMOVED_KWARGS,
                recognized_kwargs=self._RECOGNIZED_KWARGS,
                context="FixedPointIterator",
                error_on_deprecated=True,
                warn_on_unrecognized=True,
            )
        super().__init__(problem)
        self.backend = backend
        self.hjb_solver = hjb_solver
        self.fp_solver = fp_solver
        self.config = config

        # PDE coefficient overrides (Phase 2.3)
        self.volatility_field = volatility_field
        self.drift_field = drift_field

        # Issue #1081: warn on HJB-FP volatility mismatch when both are scalars.
        # If user passes `volatility_field=X` here AND `problem.sigma=Y` with
        # X != Y, HJB sees Y and FP sees X — Picard fixed point corresponds
        # to neither original nor (X, Y)-augmented MFG. Same trap pattern as
        # Issue #811 (`MFGProblem.diffusion` vs `.sigma`). For non-scalar /
        # callable cases, can't compare cheaply — silently allow (research
        # code with augmented diffusion intentionally desyncs these).
        problem_sigma = getattr(problem, "sigma", None)
        if (
            volatility_field is not None
            and problem_sigma is not None
            and isinstance(volatility_field, (int, float))
            and isinstance(problem_sigma, (int, float))
            and abs(float(volatility_field) - float(problem_sigma)) > 1e-12
        ):
            import warnings as _warnings

            _warnings.warn(
                f"FixedPointIterator: volatility_field={volatility_field} differs "
                f"from problem.sigma={problem_sigma}. HJB will use problem.sigma, "
                f"FP will use volatility_field. The Picard fixed point may "
                f"correspond to neither the original nor the augmented MFG. "
                f"For LLF/regularization-augmented FP, suppress this warning "
                f"intentionally; for unintentional desync, set both to the "
                f"same scalar.",
                UserWarning,
                stacklevel=2,
            )

        # Anderson acceleration support
        self.use_anderson = use_anderson
        self.anderson_accelerator = None
        if use_anderson:
            from mfgarchon.alg.numerical.coupling.anderson_acceleration import AndersonAccelerator

            self.anderson_accelerator = AndersonAccelerator(depth=anderson_depth, beta=anderson_beta)

        # Canonical relaxation state (config can override at solve() time)
        # Issue #719: Per-variable relaxation support
        self.relaxation = relaxation
        self.relaxation_M = relaxation_M  # None = use `relaxation` for both

        # Issue #583: Adaptive Picard relaxation
        self.adaptive_relaxation = adaptive_relaxation
        self.adaptive_relaxation_decay = adaptive_relaxation_decay
        self.adaptive_relaxation_min = adaptive_relaxation_min

        # Issue #719 Phase 2: Relaxation schedules
        self.relaxation_schedule = relaxation_schedule
        self.relaxation_schedule_M = relaxation_schedule_M

        # State arrays (initialized in solve)
        self.U: np.ndarray | None = None
        self.M: np.ndarray | None = None

        # Convergence tracking
        self.l2distu_abs: np.ndarray | None = None
        self.l2distm_abs: np.ndarray | None = None
        self.l2distu_rel: np.ndarray | None = None
        self.l2distm_rel: np.ndarray | None = None
        self.iterations_run = 0

        # Warm start support
        self._warm_start_U: np.ndarray | None = None
        self._warm_start_M: np.ndarray | None = None

        # Cache solver signatures via base class (Issue #934)
        self._init_solver_signatures(self.hjb_solver, self.fp_solver)

    def _compose_hjb_source(self, m_current: np.ndarray, u_current: np.ndarray) -> Callable | None:
        """Compose problem-level HJB source terms into a solver-level callable.

        Issue #1361: thin delegate to the single-source
        :func:`source_composition.compose_hjb_source`, shared verbatim with the
        coupled-Newton ``MFGResidual`` path so the source/nonlocal/obstacle
        convention lives in exactly one place (the bug class behind #1259 and
        #1285 was a private second copy). Bridges the problem-level signature
        ``(x, m, v, t)`` to the solver-level ``(t, x)`` by closure binding.

        Returns:
            Callable or None if no HJB source terms are active.
        """
        return compose_hjb_source(self.problem, m_current, u_current)

    def _compose_fp_source(self, m_current: np.ndarray, v_current: np.ndarray) -> Callable | None:
        """Compose problem-level FP source terms into a solver-level callable.

        Issue #1361: thin delegate to the single-source
        :func:`source_composition.compose_fp_source`, shared with the
        coupled-Newton ``MFGResidual`` path.

        Returns:
            Callable or None if no FP source terms are active.
        """
        return compose_fp_source(self.problem, m_current, v_current)

    def _get_initial_and_terminal_conditions(self, shape: tuple) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieve initial density and terminal value function from problem.

        Issue #543 Phase 2: Centralizes initial/terminal condition retrieval
        with 4-priority cascade (eliminates 8 hasattr checks).

        Args:
            shape: Spatial grid shape

        Returns:
            (M_initial, U_terminal): Initial density and terminal value function

        Priority order:
            1. get_m_init() / get_u_terminal() methods (preferred modern API)
            2. m_initial / u_terminal attributes (Issue #670 unified naming)
            3. get_initial_m() / get_final_u() methods (alternate modern API)
            4. initial_density() / terminal_cost() callables (functional API)
        """
        # Priority 1: get_m_init() / get_u_terminal() methods
        try:
            M_initial = self.problem.get_m_init()
            if M_initial.shape != shape:
                M_initial = M_initial.reshape(shape)

            try:
                U_terminal = self.problem.get_u_terminal()
            except AttributeError:
                # No terminal condition - use zeros
                U_terminal = np.zeros(shape)

            if U_terminal.shape != shape:
                U_terminal = U_terminal.reshape(shape)

            return M_initial, U_terminal
        except AttributeError:
            pass  # Try next priority

        # Priority 2: m_initial / u_terminal attributes (Issue #670: unified naming)
        try:
            M_initial = self.problem.m_initial
            if M_initial is not None:
                if M_initial.shape != shape:
                    M_initial = M_initial.reshape(shape)

                try:
                    U_terminal = self.problem.u_terminal
                except AttributeError:
                    U_terminal = np.zeros(shape)

                if U_terminal.shape != shape:
                    U_terminal = U_terminal.reshape(shape)

                return M_initial, U_terminal
        except AttributeError:
            pass  # Try next priority

        # Priority 3: get_initial_m() / get_final_u() methods
        try:
            M_initial = self.problem.get_initial_m()
            U_terminal = self.problem.get_final_u()
            return M_initial, U_terminal
        except AttributeError:
            pass  # Try next priority

        # Priority 4: initial_density() / terminal_cost() callables
        try:
            x_grid = self.problem.geometry.get_spatial_grid()
            M_initial = self.problem.initial_density(x_grid).reshape(shape)
            U_terminal = self.problem.terminal_cost(x_grid).reshape(shape)
            return M_initial, U_terminal
        except AttributeError as e:
            raise ValueError(
                "Problem must provide initial/terminal conditions via one of:\n"
                "  1. get_m_init()/get_u_terminal() methods (preferred)\n"
                "  2. m_initial/u_terminal attributes\n"
                "  3. get_initial_m()/get_final_u() methods\n"
                "  4. initial_density()/terminal_cost() callables"
            ) from e

    def _compute_drift_field(self, U, M, H_class):
        """Compute α* from U via H.optimal_control, return as synthetic U.

        Issue #896: replaces the quadratic assumption (effective_drift = U).
        Computes α* = H.optimal_control(x, m, ∇U, t) at each time step,
        then integrates α* to produce a synthetic U field whose finite
        differences reproduce the correct velocity in the legacy FP solver.

        For quadratic H (α* = -∇U/λ), this is equivalent to U/λ.
        For non-quadratic H, the synthetic U encodes the non-linear control.

        Currently 1D only. For nD problems, falls back to passing U directly
        (correct for quadratic H; TODO: extend synthetic-U to nD).
        """
        import numpy as np

        # For separable H with smooth (quadratic) control cost, the FP solver's
        # internal drift extraction (-coupling_coefficient * ∇U) is already
        # correct: it reproduces α* = -∇U/λ. The synthetic-U reconstruction
        # introduces unnecessary numerical integration error.
        # Only use synthetic-U for non-smooth or non-quadratic H.
        from mfgarchon.core.hamiltonian import SeparableHamiltonian

        if isinstance(H_class, SeparableHamiltonian) and H_class.control_cost.is_smooth():
            return U

        # Synthetic-U integration is 1D only. For nD with non-smooth H,
        # the reconstruction requires solving ∇U_syn = -α*/c (a Poisson-like
        # problem). This is deferred; for now, fall back to U (which is exact
        # for quadratic H and approximate for others).
        if U.ndim > 2:
            logger.warning(
                "Non-smooth H with nD problem: synthetic-U drift not yet supported. "
                "Falling back to U as drift potential (exact only for quadratic H)."
            )
            return U

        geometry = self.problem.geometry
        grid_spacing = geometry.get_grid_spacing()
        dx = grid_spacing[0]
        dt = self.problem.dt
        Nt = U.shape[0]
        Nx = U.shape[-1]
        coupling_coefficient = getattr(self.problem, "coupling_coefficient", 1.0)

        # Compute ∇U via central differences
        grad_U = np.gradient(U, dx, axis=-1)

        # Compute α* at grid points for all time steps
        bounds = geometry.get_bounds()
        x_grid = np.linspace(bounds[0][0], bounds[1][0], Nx).reshape(-1, 1)

        alpha_field = np.zeros_like(grad_U)
        for n in range(Nt):
            p = grad_U[n]
            m_n = M[n] if n < M.shape[0] else M[-1]
            alpha_field[n] = H_class.optimal_control(x_grid, m_n, p.reshape(-1, 1), t=n * dt).ravel()

        # Construct synthetic U such that:
        #   -coupling_coefficient * (U_syn[i+1] - U_syn[i]) / dx ≈ α*[i+1/2]
        # => U_syn[i+1] - U_syn[i] = -α*[i+1/2] * dx / coupling_coefficient
        # Using midpoint α*[i+1/2] ≈ (α*[i] + α*[i+1]) / 2
        alpha_mid = 0.5 * (alpha_field[:, :-1] + alpha_field[:, 1:])  # (Nt, Nx-1)
        increments = -alpha_mid * dx / coupling_coefficient
        U_synthetic = np.zeros_like(U)
        U_synthetic[:, 1:] = np.cumsum(increments, axis=-1)

        return U_synthetic

    def _compute_velocity_field(self, U, M, H_class):
        """Compute face-centered velocity α* via H.optimal_control.

        Issue #1233: single-sourced through
        :func:`fixed_point_utils.compute_fp_velocity_field` (shared with the Newton
        ``MFGResidual``). The face-centered convention (Issue #919) is unchanged.
        """
        return compute_fp_velocity_field(self.problem, U, M, H_class)

    def solve(
        self,
        config: MFGSolverConfig | None = None,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        return_tuple: bool = False,
        iteration_callback: IterationCallback | None = None,
        track_measure_field: bool = False,
        **kwargs: Any,
    ) -> SolverResult | tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
        """
        Solve coupled MFG system using fixed-point iteration.

        Args:
            config: Solver configuration (overrides instance config)
            max_iterations: Maximum iterations (legacy parameter)
            tolerance: Convergence tolerance (legacy parameter)
            return_tuple: Return legacy tuple format instead of SolverResult
            iteration_callback: Optional callback called after each Picard iteration.
                Signature: callback(iteration, U, M, error_U, error_M) -> bool
                Return True to continue, False to stop early.
                If None (default), no callback is invoked.
            track_measure_field: If True, store each Picard iterate as a
                MeasureField snapshot for sensitivity analysis. The resulting
                GridMeasureField is attached to SolverResult.metadata["measure_field"].
                Each snapshot stores (ParticleMeasure from M_k, U_k). Default False.
            **kwargs: Additional parameters for backward compatibility

        Returns:
            SolverResult object (or tuple if return_tuple=True)

        Example:
            >>> def monitor(i, U, M, err_U, err_M):
            ...     print(f"Iteration {i}: err_U={err_U:.2e}, err_M={err_M:.2e}")
            ...     # Save checkpoint every 10 iterations
            ...     if i % 10 == 0:
            ...         np.save(f"checkpoint_{i}.npy", {"U": U, "M": M})
            ...     return True  # Continue
            >>> result = solver.solve(iteration_callback=monitor)
        """
        # Use provided config or fall back to instance config
        solve_config = config or self.config

        # Parameter resolution (config > explicit args > instance defaults)
        if solve_config is not None:
            final_max_iterations = solve_config.picard.max_iterations
            final_tolerance = solve_config.picard.tolerance
            final_damping_factor = solve_config.picard.relaxation
            # Issue #719: Per-variable relaxation and schedules from config
            # relaxation_M / relaxation_schedule_M use `or` because None means "follow U"
            final_damping_factor_M = solve_config.picard.relaxation_M or self.relaxation_M
            final_schedule = solve_config.picard.relaxation_schedule
            final_schedule_M = solve_config.picard.relaxation_schedule_M or self.relaxation_schedule_M
            final_adaptive_damping = solve_config.picard.adaptive_relaxation or self.adaptive_relaxation
            verbose = solve_config.picard.verbose
        else:
            # Legacy parameter precedence
            final_max_iterations = (
                max_iterations or kwargs.get("max_picard_iterations") or kwargs.get("Niter_max") or 100
            )
            final_tolerance = tolerance or kwargs.get("picard_tolerance") or kwargs.get("l2errBoundPicard") or 1e-6
            final_damping_factor = self.relaxation
            final_damping_factor_M = self.relaxation_M  # Issue #719
            final_schedule = self.relaxation_schedule  # Issue #719 Phase 2
            final_schedule_M = self.relaxation_schedule_M
            final_adaptive_damping = self.adaptive_relaxation
            from mfgarchon.utils.progress import should_show_progress

            verbose = should_show_progress()

        # Issue #1081: Picard cannot converge below the inner Newton-solve floor. When the
        # HJB Newton tolerance is looser than the Picard tolerance, each inner solve injects
        # a ~newton_tolerance residual, so the outer Picard residual plateaus there and the
        # run reports "max iterations reached" without ever converging. Warn (strictly
        # looser; equal default tolerances of 1e-6 are fine).
        _newton_tol = getattr(self.hjb_solver, "newton_tolerance", None)
        if (
            isinstance(_newton_tol, (int, float))
            and isinstance(final_tolerance, (int, float))
            and _newton_tol > final_tolerance
        ):
            import warnings as _warnings

            _warnings.warn(
                f"HJB newton_tolerance ({_newton_tol:.1e}) is looser than the Picard "
                f"tolerance ({final_tolerance:.1e}); the outer Picard residual cannot drop "
                f"below the inner Newton floor and may report 'max iterations reached' "
                f"without converging. Set newton_tolerance <= picard tolerance.",
                UserWarning,
                stacklevel=2,
            )

        # Get problem dimensions - handle both old 1D and new nD interfaces
        num_time_steps = self.problem.Nt + 1  # Renamed from Nt

        # Detect problem shape using geometry API
        from mfgarchon.geometry.base import CartesianGrid

        # Issue #543 Phase 2: Replace hasattr with try/except
        try:
            geometry = self.problem.geometry
        except AttributeError as e:
            raise ValueError("Problem must have 'geometry' attribute") from e

        if geometry is None:
            raise ValueError("Problem geometry cannot be None")

        from mfgarchon.geometry import GeometryType

        is_grid = isinstance(geometry, CartesianGrid)
        if is_grid:
            shape = tuple(self.problem.geometry.get_grid_shape())
            grid_spacing = self.problem.geometry.get_grid_spacing()[0]  # For compatibility
        elif getattr(geometry, "geometry_type", None) == GeometryType.UNSTRUCTURED_MESH:
            # Coupled-FEM chain seam 3: unstructured mesh -> flat per-DOF state, no grid spacing.
            # The FEM / meshless-Galerkin solvers assemble their own operators; the iterator only
            # shuttles (Nt+1, N) arrays. grid_spacing is a benign unit weight here -- the L2
            # convergence is a *relative* tolerance, so a constant volume element does not change
            # convergence detection (only the absolute L2 value, which is not compared to anything).
            shape = (int(self.problem.num_spatial_points),)
            grid_spacing = 1.0
        else:
            raise ValueError(
                f"Problem geometry must be a CartesianGrid or unstructured mesh, got {type(geometry).__name__}"
            )
        time_step = self.problem.dt

        # Issue #1285: M_initial / U_terminal are needed by both the warm-start
        # and cold-start paths (HJB solve, FP solve, BC preservation).  Hoist
        # unconditionally so the warm-start branch does not hit NameError.
        # BlockIterator does the same (block_iterators.py:485).
        M_initial, U_terminal = self._get_initial_and_terminal_conditions(shape)

        # Initialize arrays (cold start or warm start)
        warm_start = self.get_warm_start_data()
        if warm_start is not None:
            self.U, self.M = warm_start
        else:
            # Cold start initialization
            if self.backend is not None:
                self.U = self.backend.zeros((num_time_steps, *shape))
                self.M = self.backend.zeros((num_time_steps, *shape))
            else:
                self.U = np.zeros((num_time_steps, *shape))
                self.M = np.zeros((num_time_steps, *shape))

            if num_time_steps > 0:
                # Set boundary conditions
                if len(shape) == 1:
                    self.M[0, :] = M_initial
                    self.U[num_time_steps - 1, :] = U_terminal
                else:
                    self.M[0] = M_initial
                    self.U[num_time_steps - 1] = U_terminal

                # Initialize interior with boundary conditions
                self.U, self.M = initialize_cold_start(self.U, self.M, M_initial, U_terminal, num_time_steps)

        # Initialize error tracking
        self.l2distu_abs = np.ones(final_max_iterations)
        self.l2distm_abs = np.ones(final_max_iterations)
        self.l2distu_rel = np.ones(final_max_iterations)
        self.l2distm_rel = np.ones(final_max_iterations)
        self.iterations_run = 0

        # Issue #583: Adaptive damping state
        _error_history_U: list[float] = []
        _error_history_M: list[float] = []
        _damping_history: list[dict] = []
        _theta_U_initial = final_damping_factor
        _theta_M_initial = final_damping_factor_M if final_damping_factor_M is not None else final_damping_factor

        # Reset Anderson accelerator if using it
        if self.anderson_accelerator is not None:
            self.anderson_accelerator.reset()

        # Layer 2: Initialize MeasureField for sensitivity tracking (#956)
        measure_field = None
        if track_measure_field:
            if not is_grid:
                raise NotImplementedError(
                    "track_measure_field uses GridMeasureField (get_spatial_grid) and is only "
                    "supported for CartesianGrid geometry, not unstructured mesh."
                )
            from mfgarchon.core.measure import ParticleMeasure
            from mfgarchon.core.measure_field import GridMeasureField

            grid_1d = self.problem.geometry.get_spatial_grid().ravel()
            measure_field = GridMeasureField(grid_1d, np.linspace(0, self.problem.T, num_time_steps))
            # Store initial iterate (use terminal density as measure representative)
            mu_init = ParticleMeasure.from_density(self.M[-1], grid_1d)
            measure_field.add_snapshot(mu_init, self.U.copy())

        # Main fixed-point iteration loop
        converged = False
        convergence_reason = "Maximum iterations reached"
        # Hierarchical progress for Picard iterations (Issue #614)
        from mfgarchon.utils.progress import HierarchicalProgress

        with HierarchicalProgress(verbose=verbose) as progress:
            # Add main Picard task with initial metrics
            picard_task = progress.add_task(
                "MFG Picard",
                total=final_max_iterations,
                iter=f"0/{final_max_iterations}",
                err_U=0.0,
                err_M=0.0,
            )

            for iiter in range(final_max_iterations):
                iter_start = time.time()

                U_old = self.U.copy()
                M_old = self.M.copy()

                # Build iteration state for BC provider resolution (Issue #625)
                # This state is passed to BCValueProvider.compute() for dynamic BCs
                bc_resolution_state = {
                    "m_current": M_old,
                    "U_current": U_old,
                    "geometry": self.problem.geometry,
                    "sigma": getattr(self.problem, "sigma", None),
                    "iteration": iiter,
                }

                # 1. Solve HJB backward with current M (transient subtask)
                # Issue #614: Use hierarchical subtask for inner solver visibility
                # Issue #625: Resolve BC providers before HJB solve
                # Issue #922: Compose source terms from problem fields
                # Issue #1259: Pass U_old so nonlocal_operator J[v] uses previous iterate
                hjb_source = self._compose_hjb_source(M_old, U_old)

                # Issue #934: Context routing handles progress automatically —
                # solver's create_progress_bar detects parent HierarchicalProgress
                with self.problem.using_resolved_bc(bc_resolution_state):
                    kwargs = self._build_hjb_kwargs(
                        volatility_field=self.volatility_field,
                        source_term=hjb_source,
                    )
                    U_new = self.hjb_solver.solve_hjb_system(M_old, U_terminal, U_old, **kwargs)

                # 2. Solve FP forward with new U (transient subtask)
                # Issue #614: Use hierarchical subtask for inner solver visibility
                # Issue #922: Compose FP source terms from problem fields
                fp_source = self._compose_fp_source(M_old, U_new)

                # Issue #934: Context routing handles progress automatically
                kwargs = self._build_fp_kwargs(
                    volatility_field=self.volatility_field,
                    source_term=fp_source,
                )

                # Drift/potential logic (FP-specific, not in _build_fp_kwargs).
                # Issue #1233: single-sourced via resolve_fp_drift_kwargs, shared with
                # the Newton MFGResidual so both coupling paths use one convention.
                if self._fp_sig_params is not None:
                    drift_kwargs, use_positional_U = resolve_fp_drift_kwargs(
                        self.problem, self._fp_sig_params, self.drift_field, U_new, M_old
                    )
                    kwargs.update(drift_kwargs)
                    if use_positional_U:
                        M_new = self.fp_solver.solve_fp_system(M_initial, U_new, **kwargs)
                    else:
                        M_new = self.fp_solver.solve_fp_system(M_initial, **kwargs)
                else:
                    M_new = self.fp_solver.solve_fp_system(M_initial, U_new)

                # 3. Apply damping or Anderson acceleration
                # Issue #719: Per-variable damping + Phase 2 schedule support
                from .fixed_point_utils import compute_scheduled_damping

                base_theta_M = final_damping_factor_M if final_damping_factor_M is not None else final_damping_factor
                effective_theta_U = compute_scheduled_damping(
                    iiter,
                    final_damping_factor,
                    final_schedule,
                    self.adaptive_relaxation_min,
                )
                effective_theta_M = compute_scheduled_damping(
                    iiter,
                    base_theta_M,
                    final_schedule_M if final_schedule_M is not None else final_schedule,
                    self.adaptive_relaxation_min,
                )

                if self.use_anderson and self.anderson_accelerator is not None:
                    # Anderson acceleration on U only (M uses standard damping for positivity)
                    x_current_U = U_old.flatten()
                    f_current_U = U_new.flatten()
                    x_next_U = self.anderson_accelerator.update(x_current_U, f_current_U, method="type1")
                    self.U = x_next_U.reshape(U_old.shape)

                    # Standard damping for M (guarantees non-negativity and mass conservation)
                    self.M = effective_theta_M * M_new + (1 - effective_theta_M) * M_old
                else:
                    # Standard damping for both - Issue #719: separate factors + schedules
                    self.U = effective_theta_U * U_new + (1 - effective_theta_U) * U_old
                    self.M = effective_theta_M * M_new + (1 - effective_theta_M) * M_old

                # Preserve boundary conditions
                self.M = preserve_initial_condition(self.M, M_initial)
                self.U = preserve_terminal_condition(self.U, U_terminal)

                # Layer 2: Record MeasureField snapshot (#956)
                # Use terminal density M[-1] as the measure representative —
                # it varies most across Picard iterates (M[0] is fixed by IC).
                if measure_field is not None:
                    from mfgarchon.core.measure import ParticleMeasure

                    grid_1d = self.problem.geometry.get_spatial_grid().ravel()
                    mu_k = ParticleMeasure.from_density(self.M[-1], grid_1d)
                    measure_field.add_snapshot(mu_k, self.U.copy())

                # Issue #688: Early termination on NaN/Inf (runtime safety)
                # Issue #1078: identify HJB vs FP source for triage
                if not np.all(np.isfinite(self.U)) or not np.all(np.isfinite(self.M)):
                    hjb_bad = not np.all(np.isfinite(U_new))
                    fp_bad = not np.all(np.isfinite(M_new))
                    if hjb_bad and not fp_bad:
                        source = "HJB (Newton divergence)"
                    elif fp_bad and not hjb_bad:
                        source = "FP (density blow-up)"
                    elif hjb_bad and fp_bad:
                        source = "both HJB and FP"
                    else:
                        source = "post-damping (likely Anderson acceleration)"
                    convergence_reason = "diverged_nan"
                    logger.warning(
                        "NaN/Inf detected in iteration %d (source: %s). Terminating early.",
                        iiter + 1,
                        source,
                    )
                    self.iterations_run = iiter + 1
                    break

                # Calculate convergence metrics
                from mfgarchon.utils.convergence import calculate_l2_convergence_metrics

                metrics = calculate_l2_convergence_metrics(self.U, U_old, self.M, M_old, grid_spacing, time_step)
                self.l2distu_abs[iiter] = metrics["l2distu_abs"]
                self.l2distu_rel[iiter] = metrics["l2distu_rel"]
                self.l2distm_abs[iiter] = metrics["l2distm_abs"]
                self.l2distm_rel[iiter] = metrics["l2distm_rel"]

                iter_time = time.time() - iter_start
                self.iterations_run = iiter + 1

                # Update main task progress with metrics (Issue #614)
                progress.update(
                    picard_task,
                    iter=f"{iiter + 1}/{final_max_iterations}",
                    err_U=self.l2distu_rel[iiter],
                    err_M=self.l2distm_rel[iiter],
                    time=f"{iter_time:.1f}s",
                )

                # Issue #583: Adaptive damping — adjust based on error behavior
                _error_history_U.append(self.l2distu_rel[iiter])
                _error_history_M.append(self.l2distm_rel[iiter])

                if final_adaptive_damping and iiter >= 1:
                    from .fixed_point_utils import adapt_damping

                    final_damping_factor, theta_M_adapted, warning_msg = adapt_damping(
                        theta_U=final_damping_factor,
                        theta_M=base_theta_M,
                        error_history_U=_error_history_U,
                        error_history_M=_error_history_M,
                        theta_U_initial=_theta_U_initial,
                        theta_M_initial=_theta_M_initial,
                        decay=self.adaptive_relaxation_decay,
                        min_damping=self.adaptive_relaxation_min,
                    )
                    # Update base M damping for next iteration
                    if final_damping_factor_M is not None:
                        final_damping_factor_M = theta_M_adapted
                    # else: base_theta_M follows final_damping_factor via fallback

                    if warning_msg:
                        logger.warning(warning_msg)

                    _damping_history.append(
                        {
                            "iteration": iiter + 1,
                            "theta_U": final_damping_factor,
                            "theta_M": theta_M_adapted,
                        }
                    )

                # Issue #614: Invoke user callback if provided
                if iteration_callback is not None:
                    should_continue = iteration_callback(
                        iiter,
                        self.U,
                        self.M,
                        self.l2distu_rel[iiter],
                        self.l2distm_rel[iiter],
                    )
                    if should_continue is False:
                        converged = True
                        convergence_reason = "callback_stopped"
                        break

                # Check convergence
                converged, convergence_reason = check_convergence_criteria(
                    self.l2distu_rel[iiter],
                    self.l2distm_rel[iiter],
                    self.l2distu_abs[iiter],
                    self.l2distm_abs[iiter],
                    final_tolerance,
                )

                if converged:
                    break

        # Build metadata
        metadata: dict[str, Any] = {
            "convergence_reason": convergence_reason,
            "l2distu_rel": self.l2distu_rel[: self.iterations_run],
            "l2distm_rel": self.l2distm_rel[: self.iterations_run],
            "anderson_used": self.use_anderson,
        }

        # Issue #719 Phase 2: Record schedule info
        if final_schedule != "constant" or (final_schedule_M is not None and final_schedule_M != "constant"):
            metadata["damping_schedule"] = {
                "schedule_U": final_schedule,
                "schedule_M": final_schedule_M if final_schedule_M is not None else final_schedule,
            }

        # Layer 2: Attach MeasureField to metadata (#956)
        if measure_field is not None:
            metadata["measure_field"] = measure_field

        if final_adaptive_damping:
            _final_base_theta_M = final_damping_factor_M if final_damping_factor_M is not None else final_damping_factor
            metadata["adaptive_damping"] = {
                "enabled": True,
                "damping_history": _damping_history,
                "final_theta_U": final_damping_factor,
                "final_theta_M": _final_base_theta_M,
            }

        # Issue #688: Validate final solver output
        from mfgarchon.utils.validation.runtime import validate_solver_output

        output_validation = validate_solver_output(
            self.U,
            self.M,
            check_finite=True,
            check_density_positive=True,
        )
        if not output_validation.is_valid:
            metadata["output_validation"] = {
                "is_valid": False,
                "issues": [str(issue) for issue in output_validation.issues],
            }
            logger.warning(
                "Solver output validation failed: %s",
                "; ".join(str(i) for i in output_validation.issues),
            )

        # Construct result
        result = SolverResult(
            U=self.U,
            M=self.M,
            iterations=self.iterations_run,
            error_history_U=self.l2distu_abs[: self.iterations_run],
            error_history_M=self.l2distm_abs[: self.iterations_run],
            solver_name=self.name,
            converged=converged,
            metadata=metadata,
        )

        # Return tuple for backward compatibility if requested
        if return_tuple:
            import warnings

            warnings.warn(
                "return_tuple=True is deprecated since v0.17.13. "
                "Use SolverResult object instead (the default). "
                "Will be removed in v1.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            return (
                self.U,
                self.M,
                self.iterations_run,
                self.l2distu_rel[: self.iterations_run],
                self.l2distm_rel[: self.iterations_run],
            )
        else:
            return result

    @property
    def name(self) -> str:
        """Solver name for diagnostics."""
        return "FixedPointIterator"

    def get_convergence_data(self) -> dict[str, np.ndarray]:
        """Get convergence diagnostics."""
        return {
            "l2distu_abs": self.l2distu_abs[: self.iterations_run] if self.l2distu_abs is not None else np.array([]),
            "l2distm_abs": self.l2distm_abs[: self.iterations_run] if self.l2distm_abs is not None else np.array([]),
            "l2distu_rel": self.l2distu_rel[: self.iterations_run] if self.l2distu_rel is not None else np.array([]),
            "l2distm_rel": self.l2distm_rel[: self.iterations_run] if self.l2distm_rel is not None else np.array([]),
        }

    def set_warm_start_data(self, U_init: np.ndarray, M_init: np.ndarray) -> None:
        """Set warm start initialization data."""
        self._warm_start_U = U_init
        self._warm_start_M = M_init

    def get_warm_start_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Get warm start initialization data."""
        if self._warm_start_U is not None and self._warm_start_M is not None:
            return (self._warm_start_U, self._warm_start_M)
        return None

    def clear_warm_start_data(self) -> None:
        """Clear warm start data."""
        self._warm_start_U = None
        self._warm_start_M = None

    def get_results(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the computed solution arrays.

        Returns:
            Tuple of (U, M) solution arrays

        Raises:
            RuntimeError: If no solution has been computed yet
        """
        if self.U is None or self.M is None:
            raise RuntimeError("No solution computed. Call solve() first.")
        return self.U, self.M


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing FixedPointIterator...")

    # Test class availability
    assert FixedPointIterator is not None
    print("  FixedPointIterator class available")

    # Full smoke test requires complete solver setup
    # See examples/basic/ for usage examples

    print("Smoke tests passed!")

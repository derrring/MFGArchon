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
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.alg.numerical.coupling.fixed_point_utils import SOLVE_CALLER_STACKLEVEL, verdict_reads_inner_failures
from mfgarchon.config.damping import DampingPolicy
from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.non_finite_read import refuse_a_non_finite_read
from mfgarchon.utils.solver_result import SolverResult

from .base_mfg import (
    BaseCouplingIterator,
    assert_paired_solver_geometry,
    assert_paired_solver_sigma,
    damping_argument,
    damping_for_the_solve,
    refuse_a_capped_exit_in_a_coupled_solve,
    refuse_retired_solve_keywords,
    resolve_backend,
)
from .damping_run import DampingRun
from .fixed_point_utils import (
    check_convergence_criteria,
    diverged_value_function,
    read_inner_solve_failures,
    refuse_convergence_over_failed_inner_solves,
    resolve_fp_drift_kwargs,
)
from .iteration_state import IterationState

logger = get_logger(__name__)

if TYPE_CHECKING:
    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfgarchon.backends.base_backend import BaseBackend
    from mfgarchon.config import MFGSolverConfig
    from mfgarchon.core.mfg_problem import MFGProblem

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
        damping: U's and M's damping, a `DampingPolicy` (#2997), Anderson acceleration on U among its options;
            None is ``DampingPolicy.constant(0.5)``
        backend: Backend NAME (resolved via ``create_backend``) or a backend OBJECT, or None.
            Only backends whose arrays the loop can write into carry a solve -- jax and
            torch are refused at allocation, naming #1922 (see ``iteration_state.allocate_iterate``).
        volatility: Optional VOLATILITY override (float, array, or callable), forwarded to
            both the HJB and FP solvers (#1783) -- the SDE volatility Sigma, not the diffusion.
            - None: the solvers read problem.volatility, with the problem's kind (default)
            - float: Constant volatility
            - ndarray: needs volatility_kind -- 'field' (one value per point, spatial or
              (Nt+1, *grid)) or 'tensor' (a (d, k) noise matrix, constant or per point)
            - Callable: Sigma(t, x, m), read per point unless volatility_kind='tensor'
        volatility_kind: 'field' or 'tensor' for an array override (#2378 part 2a)
        drift_field: Optional drift override for non-MFG problems (array or callable)
            - None: Use MFG drift (default, drift from U)
            - ndarray: Precomputed drift field
            - Callable: State-dependent drift α(t, x, m) -> ndarray
    """

    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
        config: MFGSolverConfig | None = None,
        damping: DampingPolicy | None = None,
        backend: str | BaseBackend | None = None,
        volatility: float | np.ndarray | Any | None = None,  # Phase 2.3
        drift_field: np.ndarray | Any | None = None,  # Phase 2.3
        volatility_kind: str | None = None,
    ):
        """
        Args:
            damping: How U and M are damped, one `DampingPolicy` per field (#2997); None is
                ``DampingPolicy.constant(0.5)``. A solve's config that sets ``PicardConfig.damping`` decides it instead; a
                damping given both here and in that config is refused at the solve.
        """
        super().__init__(problem)
        # #2250: resolved here for a fast failure on an obviously bad value; resolved
        # again in allocate_iterate, at the start of each solve, which also catches a name or bad type assigned
        # to self.backend after construction (a public attribute -- see resolve_backend).
        self.backend = resolve_backend(backend, "FixedPointIterator")
        self.hjb_solver = hjb_solver
        self.fp_solver = fp_solver
        self.config = config

        # Issue #1489 (SD-duality): a weak-form pair must share streamline_diffusion_scale, else the
        # symmetric SD block is added to only one side and the Type-A transpose identity A_FP = A_HJB^T
        # breaks silently. The factory (create_paired_solvers) threads this across the pair; guard the
        # hand-built case here too. Only weak-form solvers carry a NUMERIC `_sd_scale`; others lack the
        # attr (getattr -> None). Compare only real numbers so bare Mock() test doubles -- whose
        # `_sd_scale` auto-resolves to an unequal Mock, not None -- do not trip the guard (#1489).
        hjb_sd = getattr(hjb_solver, "_sd_scale", None)
        fp_sd = getattr(fp_solver, "_sd_scale", None)
        if isinstance(hjb_sd, (int, float)) and isinstance(fp_sd, (int, float)) and hjb_sd != fp_sd:
            raise ValueError(
                f"Paired HJB / FP solvers have different streamline_diffusion_scale (HJB={hjb_sd}, "
                f"FP={fp_sd}); the SD block would be added to only one side and A_FP = A_HJB^T breaks "
                f"(Issue #1489). Use create_paired_solvers (which threads it), or set the same scale on both."
            )

        # PDE coefficient overrides (Phase 2.3)
        self.volatility = volatility
        self.volatility_kind = volatility_kind
        self.drift_field = drift_field

        # Issue #1603: a coupled HJB-FP pair is an adjoint pair and must share the volatility.
        # Single-sourced to assert_paired_solver_sigma (RFC #1574 C14) so every coupling loop shares it.
        # A volatility override needs no guard of its own: resolve_volatility_kwarg forwards it
        # to both sides or refuses the solve (#1783).
        assert_paired_solver_sigma(hjb_solver, fp_solver, "FixedPointIterator", problem=problem, override=volatility)
        assert_paired_solver_geometry(hjb_solver, fp_solver, "FixedPointIterator", problem=problem)

        # The constructor's damping, or None for this route's default; a solve's config may set it instead (#2997).
        self.damping = damping_argument(damping, "FixedPointIterator")

        # State arrays (initialized in solve)
        self.U: np.ndarray | None = None
        self.M: np.ndarray | None = None

        # Convergence tracking
        self.l2distu_abs: np.ndarray | None = None
        self.l2distm_abs: np.ndarray | None = None
        self.l2distu_rel: np.ndarray | None = None
        self.l2distm_rel: np.ndarray | None = None
        self.iterations_run = 0

        # Cache solver signatures via base class (Issue #934)
        self._init_solver_signatures(self.hjb_solver, self.fp_solver)

        # The iterate's owner (#2997), built on first use: the geometry it sizes the state from is checked there.
        self._state: IterationState | None = None

    def _iteration_state(self) -> IterationState:
        if self._state is None:
            self._state = IterationState(
                self.problem,
                self.hjb_solver,
                self.fp_solver,
                reader=type(self).__name__,
                terminal_binding=self._hjb_terminal_binding,
                unstructured_mesh=True,
            )
        return self._state

    @verdict_reads_inner_failures(lambda self: [self.hjb_solver], fp_solvers_of=lambda self: [self.fp_solver])
    def solve(
        self,
        config: MFGSolverConfig | None = None,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        *,
        iteration_callback: IterationCallback | None = None,
        track_measure_field: bool = False,
        absolute_tolerance: float | None = None,
        **kwargs: Any,
    ) -> SolverResult:
        """
        Solve coupled MFG system using fixed-point iteration.

        Args:
            config: Solver configuration (overrides instance config)
            max_iterations: Maximum iterations (legacy parameter)
            tolerance: Bound on the relative L2 change of one sweep (legacy parameter; CONVENTIONS.md § 9)
            iteration_callback: Optional callback called after each Picard iteration.
                Signature: callback(iteration, U, M, error_U, error_M) -> bool
                Return True to continue, False to stop early.
                If None (default), no callback is invoked.
            track_measure_field: If True, store each Picard iterate as a
                MeasureField snapshot for sensitivity analysis. The resulting
                GridMeasureField is attached to SolverResult.metadata["measure_field"].
                Each snapshot stores (ParticleMeasure from M_k, U_k). Default False.
            absolute_tolerance: If given, the absolute L2 change must also fall below it (#2555)
            **kwargs: Not read: ``problem.solve()`` passes ``verbose=`` here. A keyword this method no longer takes
                (``RETIRED_SOLVE_KEYWORDS``), or one that is the constructor's (``CONSTRUCTOR_KEYWORDS``), is refused by
                name.

        Returns:
            SolverResult object

        Example:
            >>> def monitor(i, U, M, err_U, err_M):
            ...     print(f"Iteration {i}: err_U={err_U:.2e}, err_M={err_M:.2e}")
            ...     # Save checkpoint every 10 iterations
            ...     if i % 10 == 0:
            ...         np.save(f"checkpoint_{i}.npy", {"U": U, "M": M})
            ...     return True  # Continue
            >>> result = solver.solve(iteration_callback=monitor)
        """
        refuse_retired_solve_keywords(kwargs, self)
        refuse_a_capped_exit_in_a_coupled_solve([self.fp_solver], "FixedPointIterator.solve")
        # Use provided config or fall back to instance config
        solve_config = config or self.config

        # Parameter resolution (config > explicit args > instance defaults)
        if solve_config is not None:
            final_max_iterations = solve_config.picard.max_iterations
            final_tolerance = solve_config.picard.tolerance
            final_absolute_tolerance = solve_config.picard.absolute_tolerance
            verbose = solve_config.picard.verbose
        else:
            final_max_iterations = max_iterations or 100
            final_tolerance = tolerance or 1e-6
            final_absolute_tolerance = absolute_tolerance
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
                stacklevel=SOLVE_CALLER_STACKLEVEL,
            )

        num_time_steps = self.problem.Nt + 1
        # The iterate's layout, its cold or warm start, and m(0), from its one owner (#2997): the grid's shape, or the
        # pair's DOF count on an unstructured mesh (#1489 S6), checked against the geometry there.
        state = self._iteration_state()
        from mfgarchon.geometry.base import CartesianGrid

        is_grid = isinstance(self.problem.geometry, CartesianGrid)
        time_step = self.problem.dt
        # #2555: the change a sweep makes is measured in the problem's own spatial measure, the one
        # `initial_mass` is measured in, so the same relative accuracy stops at the same sweep in any
        # dimension and on any extent.
        integrate = self.problem.spatial_measure().integrate

        # #2250: the owner's allocation refuses a backend that cannot carry a solve; it reads self.backend here, so a
        # backend assigned after construction, as this repository's own acceleration example does, is checked too.
        self.U, self.M = state.start(self.backend)
        M_initial = state.m0

        # Initialize error tracking
        self.l2distu_abs = np.ones(final_max_iterations)
        self.l2distm_abs = np.ones(final_max_iterations)
        self.l2distu_rel = np.ones(final_max_iterations)
        self.l2distm_rel = np.ones(final_max_iterations)
        self.iterations_run = 0

        # The damping this solve applies, with its own adapted bases, histories and Anderson memory (#2997)
        policy = damping_for_the_solve("FixedPointIterator", self.damping, DampingPolicy.constant(0.5), solve_config)
        run = DampingRun(policy)

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
        # True once self.M holds a density this solve's FP produced. The mass-conservation
        # measurement below is meaningful only then: before that, self.M is the cold start
        # (which no solve produced; with nothing prescribed its mass is constant and measures as
        # exactly 0.0, #2991) or a warm start
        # (some other solve's output). Set where self.M is assigned, not at the exits, so
        # that a break after a completed FP step still reports its real number (Issue #1717).
        fp_output_available = False
        # What the HJB solver said about its inner solves in the latest sweep (#1878).
        inner_failures = None
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
                # u(T) at the density this iteration's HJB solve uses (#1642 stage 2): the solve, the restore after a
                # diverged step and the re-imposition after the update all read this one.
                U_terminal = state.terminal_at(M_old[-1])

                # Build iteration state for BC provider resolution (Issue #625)
                # This state is passed to BCValueProvider.compute() for dynamic BCs
                bc_resolution_state = {
                    "m_current": M_old,
                    "U_current": U_old,
                    "geometry": self.problem.geometry,
                    # The volatility the solve uses, as supplied; a scalar-only provider declares
                    # itself with scalar_volatility (#2376).
                    "volatility": self.volatility if self.volatility is not None else self.problem.volatility,
                    "iteration": iiter,
                }

                # 1. Solve HJB backward with current M (transient subtask)
                # Issue #614: Use hierarchical subtask for inner solver visibility
                # Issue #625: Resolve BC providers before HJB solve
                # Issue #922: Compose source terms from problem fields
                # Issue #1259: Pass U_old so nonlocal_operator J[v] uses previous iterate.
                # Issue #2207: the composition itself moved into _build_hjb_kwargs. Passing the
                # iterates is now the only way to call the builder, which closes the two loops that
                # went through it and composed nothing (FictitiousPlay, Block). It does NOT reach
                # RegimeSwitchingIterator or GraphMFGSolver: both are BaseCouplingIterator
                # subclasses that call solve_*_system directly and never touch the builders, so a
                # required parameter cannot bind them. #2207 names that as out of scope.
                # Issue #934: Context routing handles progress automatically —
                # solver's create_progress_bar detects parent HierarchicalProgress
                with self.problem.using_resolved_bc(bc_resolution_state):
                    kwargs = self._build_hjb_kwargs(
                        M=M_old,
                        U=U_old,
                        volatility=self.volatility,
                        volatility_kind=self.volatility_kind,
                    )
                    U_new = self.hjb_solver.solve_hjb_system(M_old, U_terminal, U_old, **kwargs)
                inner_failures = read_inner_solve_failures(self.hjb_solver)

                # Issue #1717: attribute a diverged HJB to HJB, before FP consumes it. The FP source and
                # drift below are composed from U_new, so a non-finite U_new makes the FP solver fail
                # first and report a CFL condition -- an FP diagnostic for an HJB failure. Worse, that
                # exception escapes the loop, so the #1078 HJB-vs-FP attribution further down never runs.
                #
                # The check itself moved to `diverged_value_function` (#1718): a pre-merge review of
                # #1717 found six more coupling loops with this same shape and no check at all, so the
                # decisions it encodes -- publish the diverged iterate, restore the terminal row, copy
                # before restoring -- now have one owner instead of being restated seven times.
                diverged = diverged_value_function(
                    U_new, U_terminal, site="FixedPointIterator", density=M_old, iteration=iiter
                )
                if diverged is not None:
                    self.U = diverged
                    convergence_reason = "diverged_nan"
                    self.iterations_run = iiter + 1
                    break

                # 2. Solve FP forward with new U (transient subtask)
                # Issue #614: Use hierarchical subtask for inner solver visibility
                # Issue #922 / #2207: the FP source is composed inside the builder, from the
                # iterates handed in -- M_old and the NEW U, which is the pairing this side has
                # always used and differs from the HJB side's (U_old).
                # Issue #934: Context routing handles progress automatically
                kwargs = self._build_fp_kwargs(
                    M=M_old,
                    U=U_new,
                    volatility=self.volatility,
                    volatility_kind=self.volatility_kind,
                )

                # Drift/potential logic (FP-specific, not in _build_fp_kwargs).
                # Issue #1233: single-sourced via resolve_fp_drift_kwargs, shared with
                # the Newton MFGResidual so both coupling paths use one convention.
                if self._fp_sig_params is not None:
                    drift_kwargs, use_positional_U = resolve_fp_drift_kwargs(
                        self.problem,
                        self._fp_sig_params,
                        self.drift_field,
                        U_new,
                        M_old,
                        drift_convention=self._fp_drift_convention,
                        fp_solver=self.fp_solver,
                    )
                    kwargs.update(drift_kwargs)
                    if use_positional_U:
                        M_new = self.fp_solver.solve_fp_system(M_initial, U_new, **kwargs)
                    else:
                        M_new = self.fp_solver.solve_fp_system(M_initial, **kwargs)
                else:
                    M_new = self.fp_solver.solve_fp_system(M_initial, U_new)

                # 3. Damping, per field, or Anderson acceleration on U (the policy, #2997)
                self.U, self.M = run.update(iiter, U_old, U_new, M_old, M_new)

                # self.M now carries a density this solve's FP produced, so the end-of-solve
                # mass measurement has something real to measure -- including if a later
                # iteration breaks before its own FP step (Issue #1717).
                fp_output_available = True

                self.U, self.M = state.write_back(self.U, self.M, U_terminal)

                # Issue #1489 (S5): a FINITE but blown-up density (e.g. the pre-overflow divergence,
                # total mass ~5.9e4 before it becomes inf) passes the isfinite check below and would be
                # reported converged/valid. Guard the magnitude too: a total mass that has GROWN by many
                # orders relative to the initial is diverging (a legitimate solve conserves, or under
                # absorbing BC decreases — an increase of 1e4x is a blow-up, not a valid state).
                m0_mass = float(np.sum(np.abs(self.M[0]))) if self.M.size else 0.0
                m_mass = float(np.sum(np.abs(M_new)))
                if np.isfinite(m_mass) and m_mass > 1e4 * (m0_mass + 1e-300):
                    convergence_reason = "diverged_mass_blowup"
                    logger.warning(
                        "FP density blew up in iteration %d (total |M|=%.2e, %.1e x the initial); the "
                        "coupled solve is diverging — enable stabilization. Terminating early (Issue #1489).",
                        iiter + 1,
                        m_mass,
                        m_mass / (m0_mass + 1e-300),
                    )
                    self.iterations_run = iiter + 1
                    break

                # Issue #688: Early termination on NaN/Inf (runtime safety). It says which array is non-finite and
                # what the sweep was handed, and guesses no cause (#2845).
                if not np.all(np.isfinite(self.U)) or not np.all(np.isfinite(self.M)):
                    hjb_bad = not np.all(np.isfinite(U_new))
                    fp_bad = not np.all(np.isfinite(M_new))
                    bad = [
                        name for name, is_bad in (("the value function", hjb_bad), ("the density", fp_bad)) if is_bad
                    ]
                    convergence_reason = "diverged_nan"
                    logger.warning(
                        "NaN/Inf in %s in iteration %d, reported by FixedPointIterator: the density the sweep was handed has "
                        "a minimum of %.3e. Terminating early.",
                        " and ".join(bad) or "the damped iterate",
                        iiter + 1,
                        float(np.nanmin(M_old)),
                    )
                    self.iterations_run = iiter + 1
                    break

                # Layer 2: Record MeasureField snapshot (#956), after the guards above, which end a
                # sweep that has no measure to record.
                # Use terminal density M[-1] as the measure representative —
                # it varies most across Picard iterates (M[0] is fixed by IC).
                if measure_field is not None:
                    from mfgarchon.core.measure import ParticleMeasure

                    grid_1d = self.problem.geometry.get_spatial_grid().ravel()
                    mu_k = ParticleMeasure.from_density(self.M[-1], grid_1d)
                    measure_field.add_snapshot(mu_k, self.U.copy())

                # Calculate convergence metrics
                from mfgarchon.utils.convergence import sweep_change

                # The Picard residual is between the MAP'S OUTPUT and its input, U_new vs U_old.
                # `self.U` is that output after damping or Anderson, so measuring it conflates
                # "we stopped moving" with "we reached the fixed point" -- and at theta -> 0 the
                # first happens without the second (Issue #1684 item 7, the same shape as item 6
                # one level up). Measured on a 21-point no-flux fixture, iteration 2: the reported
                # l2distu_abs fell 5.825e-01 -> 7.944e-02 as theta went 1.0 -> 0.1, a factor
                # of 7.3 bought by the damping factor rather than by progress toward the fixed
                # point. Convergence is a property of the map; damping is a property of the path.
                metrics = sweep_change(U_new, U_old, M_new, M_old, integrate, time_step)
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

                # Issue #583: an adaptive field adapts on its own history; the run reads the measure from this sweep's change (#2997)
                for message in run.observe(iiter, metrics):
                    logger.warning(message)

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
                        # A user abort is not evidence of convergence, and it is not evidence
                        # against it either -- so evaluate the real criteria at this iterate and
                        # report what they say. Setting `converged = True` here (Issue #1684 item
                        # 2) made an abort at iteration 1 report success at l2distu_rel 1.000e+00
                        # while the same problem run to 3 iterations reported FAILURE at
                        # 3.259e-01: the aborted run claimed convergence at three times the error
                        # of the run that admitted it had not converged.
                        converged, _criteria_reason = refuse_convergence_over_failed_inner_solves(
                            *check_convergence_criteria(
                                self.l2distu_rel[iiter],
                                self.l2distm_rel[iiter],
                                self.l2distu_abs[iiter],
                                self.l2distm_abs[iiter],
                                final_tolerance,
                                final_absolute_tolerance,
                            ),
                            inner_failures,
                        )
                        convergence_reason = (
                            f"callback_stopped ({_criteria_reason})" if _criteria_reason else "callback_stopped"
                        )
                        break

                # Check convergence
                criteria_met, convergence_reason = check_convergence_criteria(
                    self.l2distu_rel[iiter],
                    self.l2distm_rel[iiter],
                    self.l2distu_abs[iiter],
                    self.l2distm_abs[iiter],
                    final_tolerance,
                    final_absolute_tolerance,
                )
                converged, convergence_reason = refuse_convergence_over_failed_inner_solves(
                    criteria_met, convergence_reason, inner_failures
                )

                # A met criterion stops the loop either way: the map has stopped moving, and further
                # sweeps of a map whose HJB half returns non-roots only re-certify the same point (#1878).
                if criteria_met:
                    break

        # Build metadata
        metadata: dict[str, Any] = {
            "convergence_reason": convergence_reason,
            # The final sweep's non-converged HJB time steps (#1878); None when the solver does not track them.
            "inner_hjb_failures": list(inner_failures) if isinstance(inner_failures, tuple) else None,
            "l2distu_rel": self.l2distu_rel[: self.iterations_run],
            "l2distm_rel": self.l2distm_rel[: self.iterations_run],
            "anderson_used": policy.anderson is not None,
        }

        # Issue #719 Phase 2: Record schedule info
        if policy.U.schedule != "constant" or policy.M.schedule != "constant":
            metadata["damping_schedule"] = {"schedule_U": policy.U.schedule, "schedule_M": policy.M.schedule}

        # Layer 2: Attach MeasureField to metadata (#956)
        if measure_field is not None:
            metadata["measure_field"] = measure_field

        if policy.U.adaptive or policy.M.adaptive:
            metadata["adaptive_damping"] = {
                "enabled": True,
                "damping_history": run.adapted,
                "final_theta_U": run.base("U"),
                "final_theta_M": run.base("M"),
            }

        # Issue #688: Validate final solver output
        from mfgarchon.utils.validation.runtime import validate_solver_output

        output_validation = validate_solver_output(
            self.U,
            self.M,
            check_finite=True,
            # A solver that keeps a negative reports the same thing itself, and this check's advice, to make the density
            # non-negative, is the user's to take or not (#2845).
            check_density_positive=not getattr(self.fp_solver, "reports_negative_density", False),
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

        # Issue #1672: the field defaulted to 0.0 and nothing on this path wrote it, so every
        # coupled solve reported perfect mass conservation -- including the FDM_CENTERED case whose
        # mass reached 6378 (Issue #1671). Measured here, where M and the geometry are both in hand,
        # through the geometry's own integral rather than a second hand-rolled dx, so the measure
        # has one owner.
        #
        # Measured as DRIFT from the initial mass, not as deviation from 1.0. A ratio is what the
        # field's name claims to quantify, and 1.0 is not a property of the Fokker-Planck equation --
        # it conserves whatever it started with, and what that is belongs to the caller.
        #
        # The ratio is invariant to a UNIFORM rescaling of the measure and not to this one: the end
        # nodes hold half a cell each (#2145), so `sum(M)` and the grid's integral weight the walls
        # differently, and at a no-flux wall the density piles up exactly there. Reporting the ratio
        # of the counting sum is how a solve reports perfect conservation of a quantity nobody asked
        # about.
        mass_conservation_error: float | None
        if not fp_output_available:
            # Issue #1717: no FP step completed in this solve, so self.M is not this solve's
            # density. Two ways to get here, and the number would be wrong differently in each:
            # from a cold start, self.M is IterationState.start's construction -- M_initial in every
            # row, except where the FP prescribes a value (#2991) -- so its mass describes that
            # construction, and with nothing prescribed measures as exactly 0.0, the best possible value
            # for a solve that never ran; from a warm start, self.M is some other solve's output,
            # so any number describes that solve rather than this one. Neither is a measurement
            # of this solve, and the None sentinel exists so that cannot masquerade as one.
            #
            # The condition is "no FP output yet", not "the solve diverged". A divergence after a
            # completed FP step DOES have a real density to measure, and reporting None there
            # would be this defect's mirror image -- suppressing a genuine number instead of
            # fabricating a zero.
            mass_conservation_error = None
        else:
            # #2188: a solver whose state holds information the grid density discards (e.g. how
            # many particles are still alive) gets to report its own number instead of the
            # generic one below, which cannot see it. None means "nothing solver-specific";
            # every non-particle solver's default returns that, so this changes nothing for them.
            solver_override = self.fp_solver.mass_conservation_error_override() if self.fp_solver is not None else None
            if solver_override is not None:
                mass_conservation_error = solver_override
            else:
                # #2157: the gate used to be `geometry.volume_element()` -- a different method from the
                # one that supplies the number. Only `TensorProductGrid` had it, and only
                # `TensorProductGrid` has `integrate`, so it was a spelling of "is this a
                # TensorProductGrid" that would have refused any future geometry able to integrate.
                # It now asks the question the comment always claimed to ask.
                integrate = getattr(self.problem.geometry, "integrate", None)
                if not callable(integrate):
                    # No measure, so no total mass, so no conservation error to report. None says "not
                    # measured"; 0.0 would be a valid measurement and must not stand in for one.
                    #
                    # The `np.sum(M)` fallback that used to sit here went with the old gate. Reaching it
                    # required a geometry with a volume element and no integral, which no class in this
                    # package has -- and it was the counting measure #2145 removed everywhere else.
                    mass_conservation_error = None
                else:
                    try:
                        mass_per_step = np.asarray(integrate(self.M), dtype=float)
                    except (ValueError, NotImplementedError):
                        # `integrate` refuses a geometry that has no measure -- a single-node axis is the
                        # case that exists. Found by independent review of #2145: without this, a
                        # completed solve on a one-point grid threw from result construction.
                        #
                        # The try is narrowed to this ONE call, which is what lets the deliberate
                        # non-positive-mass raise below live outside it and escape. That used to need a
                        # private `_MassNotMeasurableError` to climb out of a wider try; the sentinel is
                        # gone because the structure now carries the distinction instead of a class.
                        mass_conservation_error = None
                    else:
                        initial_mass = float(mass_per_step[0])
                        refuse_a_non_finite_read(
                            "FixedPointIterator", "the initial density's total mass", {"the initial mass": initial_mass}
                        )
                        if initial_mass <= 0.0:
                            raise ValueError(
                                f"initial density has non-positive total mass ({initial_mass!r}); "
                                "mass conservation is undefined and the solve that produced it is already wrong"
                            )
                        mass_conservation_error = float(np.max(np.abs(mass_per_step / initial_mass - 1.0)))

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
            mass_conservation_error=mass_conservation_error,
        )

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

"""
Fixed Point Iterator

Modern fixed-point iterator for MFG systems with full feature support:
- Config-based parameter management with backward compatibility
- Anderson acceleration for faster convergence
- Backend support (GPU/CPU)
- Structured SolverResult output (tuple output for legacy compatibility)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.alg.numerical.coupling.fixed_point_utils import SOLVE_CALLER_STACKLEVEL, verdict_reads_inner_failures
from mfgarchon.config.damping import DampingPolicy
from mfgarchon.utils.mfg_logging import get_logger

from .base_mfg import (
    BaseCouplingIterator,
    assert_paired_solver_geometry,
    assert_paired_solver_sigma,
    damping_argument,
    damping_for_the_solve,
    refuse_a_capped_exit_in_a_coupled_solve,
    resolve_backend,
)
from .coupled_driver import CoupledDriver, FixedPointStep
from .coupled_result import SweepRecord, coupled_result
from .damping_run import DampingRun
from .iteration_state import IterationState

logger = get_logger(__name__)

if TYPE_CHECKING:
    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfgarchon.backends.base_backend import BaseBackend
    from mfgarchon.config import MFGSolverConfig
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.utils.solver_result import SolverResult

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
        self._record: SweepRecord | None = None

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
        track_rate_history: bool = False,
        absolute_tolerance: float | None = None,
        verbose: bool | None = None,
    ) -> SolverResult:
        """
        Solve coupled MFG system using fixed-point iteration.

        Args:
            config: Solver configuration (overrides instance config)
            max_iterations: The sweeps the solve may run (legacy parameter); None is 100, and 0 runs none (#3002)
            tolerance: Bound on the relative L2 change of one sweep (legacy parameter; CONVENTIONS.md § 9)
            iteration_callback: Optional callback called after each Picard iteration.
                Signature: callback(iteration, U, M, error_U, error_M) -> bool | None
                Return False to stop early; True or None continues. The convention every coupled route that takes
                a callback reads (#2886). If None (default), no callback is invoked.
            track_measure_field: If True, store each Picard iterate as a
                MeasureField snapshot for sensitivity analysis. The resulting
                GridMeasureField is attached to SolverResult.metadata["measure_field"].
                Each snapshot stores (ParticleMeasure from M_k, U_k). Default False.
            track_rate_history: If True, record each sweep's M damping rate in
                SolverResult.metadata["learning_rate_history"]. Default False.
            absolute_tolerance: If given, the absolute L2 change must also fall below it (#2555)
            verbose: Show the progress display; None decides by the terminal (``should_show_progress``). A config in
                force governs progress instead, through its ``picard.verbose``, as its ``picard`` fields decide the budget
                and the tolerances (#2922). ``problem.solve()`` always solves from a config, into whose ``picard.verbose``
                it writes its own ``verbose=``, so there that is what governs.

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
            final_max_iterations = 100 if max_iterations is None else max_iterations
            final_tolerance = tolerance or 1e-6
            final_absolute_tolerance = absolute_tolerance
            from mfgarchon.utils.progress import should_show_progress

            verbose = should_show_progress(verbose)

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

        # The iterate's layout, its cold or warm start, and m(0), from its one owner (#2997): the grid's shape, or the
        # pair's DOF count on an unstructured mesh (#1489 S6), checked against the geometry there.
        state = self._iteration_state()
        # #2250: the owner's allocation refuses a backend that cannot carry a solve; it reads self.backend here, so a
        # backend assigned after construction, as this repository's own acceleration example does, is checked too.
        self.U, self.M = state.start(self.backend)

        # The damping this solve applies, with its own adapted bases, histories and Anderson memory (#2997)
        policy = damping_for_the_solve("FixedPointIterator", self.damping, DampingPolicy.constant(0.5), solve_config)
        run = DampingRun(policy)

        # One driver loop for every coupled route (#3002 stage 2a): the Gauss-Seidel sweep map, then the damping
        with CoupledDriver(
            "MFG Picard",
            state,
            self.hjb_solver,
            budget=final_max_iterations,
            tolerance=final_tolerance,
            absolute_tolerance=final_absolute_tolerance,
            iteration_callback=iteration_callback,
            verbose=verbose,
            track_measure_field=track_measure_field,
            track_rates=track_rate_history,
        ) as driver:
            self._record = driver.record
            # Layer 2: the initial iterate's snapshot, its terminal density as the measure's representative (#956)
            driver.snapshot(self.U, self.M)
            step = FixedPointStep(self, state, run, order="gauss_seidel", site="FixedPointIterator")
            outcome = driver.run_fixed_point(step, self.U, self.M, driver.budget)
        self.U, self.M = outcome.U, outcome.M
        self.iterations_run = driver.sweeps
        self.l2distu_abs = driver.histories["l2distu_abs"]
        self.l2distm_abs = driver.histories["l2distm_abs"]
        self.l2distu_rel = driver.histories["l2distu_rel"]
        self.l2distm_rel = driver.histories["l2distm_rel"]
        measure_field = driver.measure_field

        # FixedPoint's own metadata; the fields every route has are the result owner's (#2997)
        metadata: dict[str, Any] = {"anderson_used": policy.anderson is not None}

        # Issue #719 Phase 2: Record schedule info
        if policy.U.schedule != "constant" or policy.M.schedule != "constant":
            metadata["damping_schedule"] = {"schedule_U": policy.U.schedule, "schedule_M": policy.M.schedule}

        # Layer 2: Attach MeasureField to metadata (#956)
        if measure_field is not None:
            metadata["measure_field"] = measure_field
        if driver.rates is not None:
            metadata["learning_rate_history"] = driver.rates

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

        # The result, through the one owner every coupled route reads (#2997): its histories from the sweeps recorded, and
        # its mass_conservation_error measured only once an FP step of this solve completed (#1672, #1717).
        return coupled_result(
            U=self.U,
            M=self.M,
            record=driver.record,
            converged=driver.converged,
            reason=driver.reason,
            solver_name=self.name,
            damping=policy,
            problems=[self.problem],
            fp_solvers=[self.fp_solver],
            fp_output_available=driver.fp_output_available,
            metadata=metadata,
        )

    @property
    def name(self) -> str:
        """Solver name for diagnostics."""
        return "FixedPointIterator"

    def get_convergence_data(self) -> dict[str, np.ndarray]:
        """The last solve's change per sweep, from the record its result reads (#2997); empty before a solve."""
        return {
            key: np.array([]) if self._record is None else self._record.history(key)
            for key in ("l2distu_abs", "l2distm_abs", "l2distu_rel", "l2distm_rel")
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

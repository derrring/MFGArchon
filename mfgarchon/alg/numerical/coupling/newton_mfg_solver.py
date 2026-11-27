"""
Newton MFG Solver.

Issue #492 Phase 1: Newton's method for MFG coupling systems.

Solves the coupled MFG system using Newton's method for faster
convergence near the solution. Falls back to Picard iteration
for robustness during initial iterations.

Algorithm:
    1. (Optional) Run Picard iterations for warm-up
    2. Apply Newton iteration: J·δx = -F(x), x += δx
    3. Where F(x) = [HJB_solve(M) - U, FP_solve(U) - M]

Features:
    - Leverages existing NewtonSolver from utils/numerical/
    - Hybrid strategy: Picard warm-up + Newton finish
    - Automatic Jacobian via finite differences or JAX autodiff
    - Line search for robustness
    - Structured SolverResult output

References:
    - Achdou & Capuzzo-Dolcetta (2010): Mean field games: Numerical methods
    - Nocedal & Wright (2006): Numerical Optimization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mfgarchon.config.damping import DampingPolicy
from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.numerical.nonlinear_solvers import NewtonSolver, SolverInfo

from .base_mfg import (
    BaseCouplingIterator,
    assert_paired_solver_geometry,
    assert_paired_solver_sigma,
    damping_argument,
    refuse_a_capped_exit_in_a_coupled_solve,
)
from .coupled_driver import CoupledDriver, FixedPointStep, ResidualMap
from .coupled_result import SweepRecord, coupled_result
from .damping_run import DampingRun
from .fixed_point_utils import (
    BUDGET_SPENT,
    read_inner_solve_failures,
    validate_budget,
    verdict_reads_inner_failures,
)
from .mfg_residual import MFGResidual

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.utils.solver_result import SolverResult

    from .iteration_state import IterationState

logger = get_logger(__name__)


#: The verdict's read of the inner HJB record, before any verdict has run.
_NOT_READ = object()


class NewtonMFGSolver(BaseCouplingIterator):
    """
    Newton's method solver for coupled MFG systems.

    Uses Newton iteration to solve F(U, M) = 0 where:
        F_HJB = HJB_solve(M) - U
        F_FP = FP_solve(U) - M

    Features:
        - Quadratic convergence near solution
        - Optional Picard warm-up for robustness
        - Automatic Jacobian (finite diff or JAX)
        - Line search for globalization

    Source / nonlocal terms (Issue #1361):
        ``source_term_hjb``, a bound ``coupling_cost``, ``nonlocal_operator`` and
        ``source_term_fp`` flow through the residual ``MFGResidual`` and are composed
        from the ``(U, M)`` residual arguments via the single-source
        ``source_composition`` helpers shared with the Picard
        ``FixedPointIterator``. The FD Jacobian differentiates through the source
        dependence (option A; see ``MFGResidual``), so Newton converges to the
        same equilibrium as Picard. An obstacle is no source term: since #2002 a soft wall is
        ``state_penalty``, a potential composed into the Hamiltonian, and ``v >= Psi`` is
        ``HJBFDMSolver(constraint=...)`` (#591), a projection splitting with its own limits in #2046.

    Performance / solver selection:
        NewtonMFGSolver is research-grade for d>=2 / Nx>~50 — it is ~135x slower
        than the fixed-point (Picard) coupler in 2D and scales poorly with
        dimension and grid size; prefer ``FixedPointIterator`` (Picard) or Howard
        for production coupled solves. Newton is appropriate for small 1D problems
        or research on the coupled-Newton method.

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
        picard_warmup: Number of Picard iterations before Newton (default: 3). The
            warmup must bring the iterate into the basin of the physical equilibrium:
            Newton is a *local* root-finder and (even with line search) cannot escape a
            basin, so a too-short warmup can converge to a spurious near-trivial fixed
            point of the discrete MFG map. For stiff couplings increase this until the
            warmup residual is well past its transient peak (Issue #1233). Every warm-up sweep's
            change is judged, and the first that meets the criteria ends the solve, converged unless
            its HJB steps failed, as FixedPoint's loop stops (#1878, #2997).
        damping: The Picard warm-up's damping, a `DampingPolicy` (#2997); None is ``DampingPolicy.constant(0.5)``.
            The Newton phase is not damped; its line search sets the step.
        newton_max_iterations: Maximum Newton iterations (default: 20)
        line_search: Enable backtracking line search (default: True)
        use_jax_autodiff: Use JAX autodiff for the Jacobian (default: False). The MFG
            residual wraps the black-box numpy/scipy HJB and FP solvers, which JAX
            cannot trace — autodiff therefore raises ``TracerArrayConversionError`` and
            falls back to the finite-difference Jacobian. The FD-Jacobian path is the
            supported one; ``'auto'``/``True`` are retained only for a hypothetical
            JAX-traceable residual and will warn-and-fall-back with the standard solvers
            (Issue #1233).
        volatility: Optional volatility override -- the SDE volatility, not a diffusion
        volatility_kind: 'field' or 'tensor' for an array override (#2378 part 2a)
        drift_field: Optional drift override

    Example:
        >>> solver = NewtonMFGSolver(problem, hjb_solver, fp_solver)
        >>> result = solver.solve(max_iterations=30)  # result.U, result.M, result.metadata
        >>> print(f"Converged: {result.converged}, iterations: {result.iterations}")
    """

    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
        *,
        picard_warmup: int = 3,
        damping: DampingPolicy | None = None,
        newton_max_iterations: int = 20,
        line_search: bool = True,
        use_jax_autodiff: bool | str = False,
        volatility: float | NDArray | Any | None = None,
        volatility_kind: str | None = None,
        drift_field: NDArray | Any | None = None,
    ):
        """Initialize Newton MFG solver."""
        super().__init__(problem)

        self.hjb_solver = hjb_solver
        self.fp_solver = fp_solver
        assert_paired_solver_sigma(hjb_solver, fp_solver, "NewtonMFGSolver", problem=problem, override=volatility)
        assert_paired_solver_geometry(hjb_solver, fp_solver, "NewtonMFGSolver", problem=problem)

        # Picard warm-up parameters
        self.picard_warmup = picard_warmup
        self.damping = damping_argument(damping, "NewtonMFGSolver") or DampingPolicy.constant(0.5)

        # Newton parameters
        self.newton_max_iterations = newton_max_iterations
        self.line_search = line_search
        self.use_jax_autodiff = use_jax_autodiff

        # PDE coefficient overrides
        self.volatility = volatility
        self.volatility_kind = volatility_kind
        self.drift_field = drift_field

        # Create MFG residual computer
        self.mfg_residual = MFGResidual(
            problem,
            hjb_solver,
            fp_solver,
            volatility=volatility,
            volatility_kind=volatility_kind,
            drift_field=drift_field,
        )

        # Solution state
        self.U: NDArray | None = None
        self.M: NDArray | None = None

        # Convergence history
        self.newton_residuals: list[float] = []
        # The change of the map's output against its input at each sweep: the warm-up's sweeps, then each iterate the
        # Newton phase's verdict judges, a diverged one as NaN (#2997 ruling 7, stage 1c).
        self._record = SweepRecord(1)
        self._driver: CoupledDriver | None = None
        # The warm-up sweeps the last solve ran: the Newton phase's iterations count from there
        self._warmup_sweeps = 0

        # The warm-up's sweep map reads the solvers' signatures through the base class (Issue #934)
        self._init_solver_signatures(self.hjb_solver, self.fp_solver)

    @verdict_reads_inner_failures(
        lambda self: [self.hjb_solver],
        fp_solvers_of=lambda self: [self.fp_solver],
        check=False,
        failures_of=lambda self: self._verdict_failures(),
    )
    def solve(
        self,
        max_iterations: int | None = None,
        tolerance: float = 1e-5,
        verbose: bool = True,
        absolute_tolerance: float | None = None,
        *,
        iteration_callback: Callable[[int, NDArray, NDArray, float, float], bool | None] | None = None,
        track_measure_field: bool = False,
        track_rate_history: bool = False,
    ) -> SolverResult:
        """
        Solve MFG system using Newton's method.

        Args:
            max_iterations: The sweeps the solve may run, the warm-up's and the Newton phase's together; None is 30,
                and 0 runs none (#3002). ``newton_max_iterations`` bounds the Newton phase within it.
            tolerance: The outer coupling tolerance (docs/user/CONVENTIONS.md § 9): it bounds the Picard
                residual Phi(x) - x relative to the map's output, in the problem's own measure, through
                `sweep_change` and `check_convergence_criteria` (#2565). It is checked at every warm-up sweep, on
                that sweep's change, which ends the solve when it meets the criteria (#2997), after the warm-up and
                at each iterate Newton evaluates before stepping; the iterate returned when Newton's budget
                runs out is not judged, so it reports not converged.
            absolute_tolerance: Opt-in absolute bound on the same change, as for the other coupling
                iterators.
            verbose: Show the progress display
            iteration_callback: Optional ``callback(iteration, U, M, error_U, error_M)``. It is called after warm-up
                sweep k's update, with ``iteration=k`` and the iterate the update left, and after the Newton phase's
                verdict on each iterate, before its step, with the evaluation's index; an evaluation, update or verdict
                that ends the solve before the call makes none. The post-warm-up check, index W after W warm-up sweeps,
                makes no call: the Newton phase's first call, at W + 1, receives the iterate the check judged. A return
                of False stops the solve, True or None continues (#2886)
            track_measure_field: If True, store each warm-up iterate as a MeasureField snapshot in
                SolverResult.metadata["measure_field"] (#956)
            track_rate_history: If True, record each warm-up sweep's M damping rate in
                SolverResult.metadata["learning_rate_history"]

        Returns:
            SolverResult (#2997 ruling 4): U, M, the absolute change per sweep, a diverged one as NaN, and the convergence
            information in its metadata (convergence_reason, newton residuals, ...). The iteration count is
            `SolverResult.iterations` alone: the warm-up's sweeps and the iterates the Newton phase judged.
        """
        # A capped exit is offered for a standalone FP solve only, as on every coupled route (#2581)
        refuse_a_capped_exit_in_a_coupled_solve([self.fp_solver], "NewtonMFGSolver.solve")
        budget = 30 if max_iterations is None else max_iterations
        self._warmup_sweeps = 0
        self.newton_residuals = []
        self._verdict_inner_failures: object = _NOT_READ
        last_reason = [""]

        def on_verdict(converged: bool, reason: str) -> None:
            # The last verdict's reason and the inner HJB steps it read, for the final reason and the result (#2644)
            last_reason[0] = reason
            self._verdict_inner_failures = driver.inner_failures

        # The warm start, or the owner's cold start (#2997): u(T) at m(0) in every row of U, and m(0) with the FP's
        # prescribed values in M.
        state = self.mfg_residual.state
        U, M = state.start()
        residual_map = ResidualMap(self.mfg_residual, "NewtonMFGSolver")

        # One driver loop for every coupled route (#3002 stage 2a): a warm-up phase of FixedPoint's sweep map and
        # damping, then a Newton phase of `NewtonRun.step` on the residual of the Jacobi map. The budget counts both, and
        # every evaluation whose change a verdict reads goes through the driver's one pipeline (plan v1.1).
        with CoupledDriver(
            "Newton",
            state,
            self.mfg_residual.hjb_solver,
            reader="NewtonMFGSolver",
            budget=budget,
            tolerance=tolerance,
            absolute_tolerance=absolute_tolerance,
            iteration_callback=iteration_callback,
            verbose=verbose,
            track_measure_field=track_measure_field,
            track_rates=track_rate_history,
        ) as driver:
            self._record = driver.record
            self._driver = driver
            driver.snapshot(U, M)
            # The Newton phase's bound, refused before any sweep runs, by the rule the budget is (#3002 class line 2)
            newton_bound = validate_budget(self.newton_max_iterations, type(self).__name__, "newton_max_iterations")

            # Phase 1: the Picard warm-up, for robustness. A met criterion ends the solve with the iterate that sweep's
            # update produced, whether or not its verdict is refused over failed inner HJB steps (#1878).
            warmup = min(self.picard_warmup, driver.budget)
            if warmup > 0:
                step = FixedPointStep(
                    self, state, DampingRun(self.damping), order="gauss_seidel", site="NewtonMFGSolver"
                )
                outcome = driver.run_fixed_point(step, U, M, warmup, reason_prefix="Picard warm-up: ")
                U, M = outcome.U, outcome.M
                self._warmup_sweeps = outcome.sweeps
                if outcome.stop is not None:
                    # A stopped warm-up publishes what the pipeline published, and the warm-up count says how far it got
                    # (#1718, #1672)
                    self.U, self.M = U, M
                    self._solution_computed = True
                    if outcome.stop == "diverged":
                        self.picard_warmup = outcome.sweeps
                    return self._create_result(driver.converged, driver.reason)

            # The Newton phase gets what the warm-up left of the budget, bounded by newton_max_iterations (#3002)
            remaining = min(driver.remaining, newton_bound)
            if remaining <= 0:
                self.U, self.M = U, M
                self._solution_computed = True
                # The bound that ran out names the stop: the solve's budget, or newton_max_iterations within it
                reason = BUDGET_SPENT if driver.remaining <= 0 else "Newton max iterations"
                return self._create_result(False, reason)

            # The post-warm-up check: the outer verdict on the Picard residual at the warm-up's iterate (#2565), an
            # evaluation of the map like any other, so it goes through the pipeline. It is recorded only if it ends the
            # solve; otherwise the Newton phase's first verdict judges this same iterate and records it.
            check_index = driver.sweeps
            evaluation = driver.evaluate(check_index, residual_map, U, M)
            if evaluation.stop is not None:
                return self._stop_on_a_diverged_map(*evaluation.published, check_index, driver)
            _met, converged_after_warmup, reason = driver._verdict(evaluation.metrics)
            on_verdict(converged_after_warmup, reason)
            if converged_after_warmup:
                driver.record.measured([evaluation.metrics])
                driver.sweeps += 1
                self.U = U
                self.M = M
                self._solution_computed = True
                return self._create_result(converged=True, reason="Converged during Picard warm-up")

            # Phase 2: Newton iterations. Newton's own norm tolerance is not consulted: the stop is the outer verdict
            # (#2565), so it keeps its default.
            newton_solver = NewtonSolver(
                max_iterations=remaining,
                sparse=True,
                line_search=self.line_search,
                use_jax_autodiff=self.use_jax_autodiff,
            )
            self.mfg_residual.reset_evaluation_count()
            newton = newton_solver.run(self.mfg_residual.residual_function, self.mfg_residual.pack_state(U, M))
            outcome = driver.run_newton(
                newton,
                remaining,
                residual_map,
                check_index + 1,
                self.mfg_residual.state.pack,
                self.mfg_residual.unpack_state,
                on_verdict,
            )
            if outcome.stop == "diverged":
                return self._stop_on_a_diverged_map(outcome.U, outcome.M, outcome.detail["sweep"], driver)

        self.U, self.M = outcome.U, outcome.M
        self.newton_residuals = newton.residual_history
        self._solution_computed = True
        if outcome.stop == "callback":
            return self._create_result(driver.converged, driver.reason)

        # A last verdict refused over failed inner steps says so, not only how Newton stopped. A named stop
        # (line_search_failed, diverged) returns the iterate that verdict judged; when the budget runs out, Newton
        # evaluates F at its last iterate without a verdict, so the verdict read the iterate before it (#2644).
        newton_info = outcome.newton_info
        refused = last_reason[0] if last_reason[0].startswith("inner_hjb_not_converged") else ""
        stop = newton_info.extra.get("reason")
        if newton_info.converged:
            reason = "Newton converged"
        elif stop:
            reason = f"Newton stopped: {stop}" + (
                f"; its last verdict, on the iterate returned: {refused}" if refused else ""
            )
        else:
            # The solve's budget spent reads as on every route; newton_max_iterations spent within it, as the phase's
            # own bound (#3002, ruling 2)
            reason = (BUDGET_SPENT if driver.remaining <= 0 else "Newton max iterations") + (
                f"; its last verdict, on the iterate before the one returned: {refused}" if refused else ""
            )
        return self._create_result(converged=newton_info.converged, reason=reason, newton_info=newton_info)

    def _iteration_state(self) -> IterationState:
        return self.mfg_residual.state

    def _stop_on_a_diverged_map(self, U: NDArray, M: NDArray, sweep: int, driver: CoupledDriver) -> SolverResult:
        """Publish what the driver published and stop, with the reason it gave: the HJB guard's (#1718) or the FP-output
        rule's pair at evaluation ``sweep`` (the approver's ruling E), whose change counts as NaN (#2997), or the iterate
        the post-update rule judged after the step from evaluation ``sweep``. The Newton iterations reported are
        ``sweep`` less the warm-up's sweeps. Whether the published density is measured is the driver's one decision
        (#3002 class line 6)."""
        self.U, self.M = U, M
        self.newton_residuals = []
        self._solution_computed = True
        return self._create_result(False, driver.reason, newton_iterations=sweep - self._warmup_sweeps)

    def _verdict_failures(self) -> object:
        """The inner HJB steps the last verdict read, or the latest solve's where no verdict ran; None when the solver
        does not track them (#2644 review 2, F4): what the result's ``inner_hjb_failures`` holds."""
        failures = getattr(self, "_verdict_inner_failures", _NOT_READ)
        return read_inner_solve_failures(self.mfg_residual.hjb_solver) if failures is _NOT_READ else failures

    def _create_result(
        self,
        converged: bool,
        reason: str,
        newton_info: SolverInfo | None = None,
        newton_iterations: int | None = None,
        picard_iterations: int | None = None,
    ) -> SolverResult:
        """The one result type (#2997 ruling 4), through the owner every coupled route reads, with Newton's own counters
        in its metadata. ``newton_iterations`` overrides the count read from ``newton_residuals``, for a stop that has no
        residual history, and ``picard_iterations`` the count of warm-up sweeps the solve ran. Its histories hold every
        sweep recorded, so a warm-up that diverged at sweep i reports i + 1 iterations, the last NaN."""
        info: dict[str, Any] = {
            "picard_iterations": self._warmup_sweeps if picard_iterations is None else picard_iterations,
            "newton_iterations": len(self.newton_residuals) if newton_iterations is None else newton_iterations,
            "newton_residuals": self.newton_residuals,
            "residual_evaluations": self.mfg_residual.residual_evaluations,
        }

        if newton_info is not None:
            info["final_residual"] = newton_info.residual
            info["jacobian_evaluations"] = newton_info.extra.get("jacobian_evals", 0)
        # The driver's records any route may take (#3002, ruling 7)
        driver = self._driver
        if driver is not None and driver.measure_field is not None:
            info["measure_field"] = driver.measure_field
        if driver is not None and driver.rates is not None:
            info["learning_rate_history"] = driver.rates

        return coupled_result(
            U=self.U,
            M=self.M,
            record=self._record,
            converged=converged,
            reason=reason,
            solver_name="NewtonMFGSolver",
            damping=self.damping,
            problems=[self.problem],
            fp_solvers=[self.fp_solver],
            fp_output_available=driver is not None and driver.fp_output_available,
            metadata=info,
        )

    def get_results(self) -> tuple[NDArray, NDArray]:
        """Get computed solution arrays."""
        if self.U is None or self.M is None:
            raise RuntimeError("Solver has not been run yet. Call solve() first.")
        return self.U, self.M

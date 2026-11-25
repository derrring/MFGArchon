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

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.config.damping import DampingPolicy
from mfgarchon.utils.convergence import sweep_change
from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.numerical.nonlinear_solvers import NewtonSolver, SolverInfo
from mfgarchon.utils.solver_result import SolverResult

from .base_mfg import (
    BaseCouplingIterator,
    assert_paired_solver_geometry,
    assert_paired_solver_sigma,
    damping_argument,
)
from .damping_run import DampingRun
from .fixed_point_utils import (
    check_convergence_criteria,
    diverged_value_function,
    read_inner_solve_failures,
    refuse_convergence_over_failed_inner_solves,
    verdict_reads_inner_failures,
)
from .mfg_residual import MFGResidual

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfgarchon.core.mfg_problem import MFGProblem

    from .iteration_state import IterationState

logger = get_logger(__name__)


#: The verdict's read of the inner HJB record, before any verdict has run.
_NOT_READ = object()


class _MapDivergedError(Exception):
    """A verdict's map output, the HJB half, is non-finite: the shared guard has published it (#2997 ruling 10)."""

    def __init__(self, U: NDArray, M: NDArray, sweep: int) -> None:
        super().__init__(sweep)
        self.U, self.M, self.sweep = U, M, sweep


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
            warmup residual is well past its transient peak (Issue #1233). Every warm-up sweep is
            judged as every route judges a sweep, and one whose change passes the tolerance ends
            the solve as converged (#2997).
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
        >>> print(f"Converged: {info['converged']}, iterations: {info['iterations']}")
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
        self.picard_residuals: list[float] = []
        self.newton_residuals: list[float] = []
        # The absolute L2 change of the map's output against its input, U's and M's, at each sweep measured: the
        # warm-up's sweeps, then each iterate the Newton phase's verdict judges (#2997 ruling 7).
        self._changes: tuple[list[float], list[float]] = ([], [])

    def _run_picard_warmup(
        self,
        U: NDArray,
        M: NDArray,
        num_iterations: int,
        judge: Callable[[dict[str, float]], bool],
    ) -> tuple[NDArray, NDArray, list[float], int | None, int | None]:
        """
        Run Picard fixed-point iterations for warm-up.

        Args:
            U: Initial value function
            M: Initial density
            num_iterations: Number of Picard iterations
            judge: The verdict on a sweep's change (`sweep_change`'s metrics), the one every verdict of the solve
                reads: True when the sweep's change passes (#2997, the approver's ruling of 2026-11-25).

        Returns:
            (U, M, residuals, diverged_at, converged_at): updated state, residual history, the 0-based
            warmup iteration at which the value function went non-finite -- or None if it did
            not -- and the 0-based warmup iteration whose change passed `judge`, after which the
            warmup stopped, or None. The caller MUST stop on a non-None ``diverged_at``: the statement
            after the warmup call composes an FP solve from U (#1718).
        """
        residuals = []
        run = DampingRun(self.damping)
        # The warmup's `break` leaves only this loop. Its caller runs compute_residual_norm
        # immediately, which composes an FP solve from U -- so a diverged U escaped the guard
        # below and reached FP anyway, with the identical CFL misattribution #1718 removes
        # everywhere else. The flag carries the stop out (#1718 review).
        diverged_at: int | None = None
        converged_at: int | None = None

        for i in range(num_iterations):
            # Store old values
            U_old = U.copy()
            M_old = M.copy()

            # HJB solve: U_new = HJB(M_old), whose terminal is u(T) at M_old's m(T) (#1642)
            U_terminal = self.mfg_residual.terminal_at(M_old)
            U_new = self.mfg_residual.compute_hjb_output(M_old, U_old)

            # Issue #1718: compute_fp_output below builds its drift from U_new, so a non-finite
            # U_new makes the FP solver raise a CFL diagnostic for an HJB failure. The warmup
            # stops and hands back what it has; the residual history it returns is the record of
            # how far it got.
            diverged = diverged_value_function(
                U_new,
                U_terminal,
                site="NewtonMFGSolver._run_picard_warmup",
                density=M_old,
                iteration=i,
            )
            if diverged is not None:
                U = diverged
                diverged_at = i
                break

            # FP solve: M_new = FP(U_new)
            M_new = self.mfg_residual.compute_fp_output(U_new, M_old)
            change = sweep_change(U_new, U_old, M_new, M_old, self.problem.spatial_measure().integrate, self.problem.dt)
            self._changes[0].append(change["l2distu_abs"])
            self._changes[1].append(change["l2distm_abs"])
            converged = judge(change)
            # Apply damping
            U, M = run.update(i, U_old, U_new, M_old, M_new)
            # An adaptive field adapts on its own history, after this sweep's blend as on every route, so a decay takes
            # effect at the next sweep; the run reads the measure from this sweep's change (#2997)
            for message in run.observe(i, change):
                logger.warning(message)

            U, M = self.mfg_residual.state.write_back(U, M, U_terminal)

            # Compute residual norm
            res_norm = self.mfg_residual.compute_residual_norm(U, M)
            residuals.append(res_norm)

            logger.debug(f"Picard iteration {i + 1}: residual = {res_norm:.2e}")
            if converged:
                converged_at = i
                break

        return U, M, residuals, diverged_at, converged_at

    @verdict_reads_inner_failures(
        lambda self: [self.hjb_solver], fp_solvers_of=lambda self: [self.fp_solver], check=False
    )
    def solve(
        self,
        max_iterations: int = 30,
        tolerance: float = 1e-5,
        verbose: bool = True,
        absolute_tolerance: float | None = None,
    ) -> SolverResult:
        """
        Solve MFG system using Newton's method.

        Args:
            max_iterations: Maximum total iterations (Picard + Newton)
            tolerance: The outer coupling tolerance (docs/user/CONVENTIONS.md § 9): it bounds the Picard
                residual Phi(x) - x relative to the map's output, in the problem's own measure, through
                `sweep_change` and `check_convergence_criteria` (#2565). It is checked at every warm-up sweep,
                which ends the solve when its change passes, as every route's sweep does (#2997), after the warm-up and
                at each iterate Newton evaluates before stepping; the iterate returned when Newton's budget
                runs out is not judged, so it reports not converged.
            absolute_tolerance: Opt-in absolute bound on the same change, as for the other coupling
                iterators.
            verbose: Print progress information

        Returns:
            SolverResult (#2997 ruling 4): U, M, the absolute change per measured sweep, and the convergence information
            in its metadata (convergence_reason, picard and newton residuals, ...). The iteration count is
            `SolverResult.iterations` alone: the sweeps whose change was measured.
        """
        start_time = time.time()
        self._changes = ([], [])
        outer_verdict = self._outer_verdict(tolerance, absolute_tolerance)

        last_reason = [""]
        self._verdict_inner_failures: object = _NOT_READ
        # The 0-based index of the sweep the next verdict judges: the warm-up's sweeps come first, then one per iterate.
        judged = [self.picard_warmup]
        # The change the latest verdict read, so the exit that ends on the post-warm-up verdict can record it.
        last_change: list[dict[str, float]] = []

        def verdict(U: NDArray, M: NDArray, F_HJB: NDArray, F_FP: NDArray, record: bool = True) -> tuple[bool, str]:
            # The verdict reads the map's change, and U + F_HJB is the map's HJB output: the shared guard sees it first,
            # as on every other route, so a non-finite one stops the solve instead of reaching #2578's refusal of a
            # non-finite change (#2997 ruling 10).
            sweep = judged[0]
            judged[0] += 1
            published = diverged_value_function(
                U + F_HJB, self.mfg_residual.terminal_at(M), site="NewtonMFGSolver", density=M, iteration=sweep
            )
            if published is not None:
                raise _MapDivergedError(published, M, sweep)
            # Read right after the residual that produced F_HJB, so the HJB record is that solve's: a residual whose
            # HJB half has non-roots is not converged, whatever its norm (#1878, #2644).
            converged, reason, metrics = outer_verdict(U, M, F_HJB, F_FP)
            last_change[:] = [metrics]
            if record:
                self._changes[0].append(metrics["l2distu_abs"])
                self._changes[1].append(metrics["l2distm_abs"])
            self._verdict_inner_failures = read_inner_solve_failures(self.mfg_residual.hjb_solver)
            converged, last_reason[0] = refuse_convergence_over_failed_inner_solves(
                converged, reason, self._verdict_inner_failures
            )
            return converged, last_reason[0]

        def judge_sweep(change: dict[str, float]) -> bool:
            # A warm-up sweep's change, judged as the verdict above judges the map's: one verdict measurement for every
            # sweep of every phase (the approver's ruling of 2026-11-25, #2997).
            converged, reason = self._judge(change, tolerance, absolute_tolerance)
            self._verdict_inner_failures = read_inner_solve_failures(self.mfg_residual.hjb_solver)
            converged, last_reason[0] = refuse_convergence_over_failed_inner_solves(
                converged, reason, self._verdict_inner_failures
            )
            return converged

        # The warm start, or the owner's cold start (#2997): u(T) at m(0) in every row of U, and m(0) with the FP's
        # prescribed values in M.
        state = self.mfg_residual.state
        if verbose and state.warm_start() is not None:
            print("Using warm start from previous solution")
        U, M = state.start()

        # Phase 1: Picard warm-up (for robustness)
        if self.picard_warmup > 0:
            if verbose:
                print(f"Phase 1: Picard warm-up ({self.picard_warmup} iterations)")

            U, M, picard_res, picard_diverged_at, picard_converged_at = self._run_picard_warmup(
                U, M, self.picard_warmup, judge_sweep
            )
            self.picard_residuals = picard_res

            if picard_diverged_at is not None:
                # Stop here. `compute_residual_norm` below composes an FP solve from U, so
                # continuing would hand the diverged value function to FP and get the CFL
                # diagnostic this issue exists to eliminate -- and if it did not raise, Newton
                # would run on a NaN iterate. Publishing U unchanged keeps the diverged iterate
                # visible, which is the owner's contract (#1718).
                self.U, self.M = U, M
                # `picard_warmup` is the CONFIGURED count and `newton_residuals` may hold a
                # previous solve's history on this instance, so reporting either unchanged would
                # publish numbers for iterations this solve never ran -- the #1672 shape, at a
                # fourth site. Both are set to what actually happened.
                self.picard_warmup = picard_diverged_at + 1
                self.newton_residuals = []
                self._solution_computed = True
                return self._create_result(False, "diverged_nan", time.time() - start_time)

            if verbose and picard_res:
                print(f"  Final Picard residual: {picard_res[-1]:.2e}")

            if picard_converged_at is not None:
                # A warm-up sweep whose change passed ends the solve, with the iterate that sweep's update produced; the
                # metadata's picard_iterations says which sweep.
                self.U, self.M = U, M
                self.newton_residuals = []
                self._solution_computed = True
                return self._create_result(
                    converged=True,
                    reason="Converged during Picard warm-up",
                    elapsed=time.time() - start_time,
                    picard_iterations=picard_converged_at + 1,
                )

        # Check if already converged after Picard: the outer verdict on the Picard residual (#2565)
        _, F_HJB, F_FP = self.mfg_residual.compute_residual(U, M, return_components=True)
        try:
            # Recorded only if it ends the solve (below): otherwise the Newton phase's first verdict judges this same
            # iterate and records it.
            converged_after_warmup = verdict(U, M, F_HJB, F_FP, record=False)[0]
        except _MapDivergedError as stop:
            return self._stop_on_a_diverged_map(stop, start_time)
        if converged_after_warmup:
            self._changes[0].append(last_change[0]["l2distu_abs"])
            self._changes[1].append(last_change[0]["l2distm_abs"])
            self.U = U
            self.M = M
            self._solution_computed = True

            elapsed = time.time() - start_time
            return self._create_result(
                converged=True,
                reason="Converged during Picard warm-up",
                elapsed=elapsed,
            )

        # Phase 2: Newton iterations
        remaining_iterations = max_iterations - self.picard_warmup
        if remaining_iterations <= 0:
            remaining_iterations = self.newton_max_iterations

        if verbose:
            print(f"Phase 2: Newton iterations (max {remaining_iterations})")

        # Create Newton solver. Its own norm tolerance is not consulted: the stop is the outer verdict
        # passed to solve() below (#2565), so it keeps its default.
        newton_solver = NewtonSolver(
            max_iterations=min(remaining_iterations, self.newton_max_iterations),
            sparse=True,
            line_search=self.line_search,
            use_jax_autodiff=self.use_jax_autodiff,
        )

        # Pack current state
        x_current = self.mfg_residual.pack_state(U, M)

        # Run Newton solver
        self.mfg_residual.reset_evaluation_count()
        try:
            x_solution, newton_info = newton_solver.solve(
                self.mfg_residual.residual_function,
                x_current,
                converged=lambda x, F: verdict(*self.mfg_residual.unpack_state(x), *self.mfg_residual.unpack_state(F)),
            )
        except _MapDivergedError as stop:
            return self._stop_on_a_diverged_map(stop, start_time)

        # Unpack solution
        self.U, self.M = self.mfg_residual.unpack_state(x_solution)
        self.newton_residuals = newton_info.residual_history

        elapsed = time.time() - start_time

        if verbose:
            status = "converged" if newton_info.converged else "not converged"
            print(f"Newton {status} in {newton_info.iterations} iterations")
            print(f"  Final residual: {newton_info.residual:.2e}")
            print(f"  Jacobian evaluations: {newton_info.extra.get('jacobian_evals', 'N/A')}")
            print(f"  Total time: {elapsed:.2f}s")

        self._solution_computed = True

        # A last verdict refused over failed inner steps says so, not only how Newton stopped. A named stop
        # (line_search_failed, diverged) returns the iterate that verdict judged; when the budget runs out, Newton
        # evaluates F at its last iterate without a verdict, so the verdict read the iterate before it (#2644).
        refused = last_reason[0] if last_reason[0].startswith("inner_hjb_not_converged") else ""
        stop = newton_info.extra.get("reason")
        if newton_info.converged:
            reason = "Newton converged"
        elif stop:
            reason = f"Newton stopped: {stop}" + (
                f"; its last verdict, on the iterate returned: {refused}" if refused else ""
            )
        else:
            reason = "Newton max iterations" + (
                f"; its last verdict, on the iterate before the one returned: {refused}" if refused else ""
            )
        return self._create_result(
            converged=newton_info.converged,
            reason=reason,
            elapsed=elapsed,
            newton_info=newton_info,
        )

    def _outer_verdict(
        self, tolerance: float, absolute_tolerance: float | None
    ) -> Callable[[NDArray, NDArray, NDArray, NDArray], tuple[bool, str, dict[str, float]]]:
        """``(U, M, F_HJB, F_FP) -> (converged, reason, metrics)``: the owners' verdict on the Picard residual (#2565), and
        the change it read.

        The residual ``F = Phi(x) - x`` is the map's output against its input, the kind of pair `sweep_change`
        measures for every other coupling iterator; so ``U + F_HJB`` and ``M + F_FP`` are the map's output.
        Newton's map is the Jacobi one -- the FP is fed the input U -- where FixedPointIterator's is
        Gauss-Seidel; the two have the same fixed point.
        """
        integrate = self.problem.spatial_measure().integrate
        dt = self.problem.dt

        def verdict(U: NDArray, M: NDArray, F_HJB: NDArray, F_FP: NDArray) -> tuple[bool, str, dict[str, float]]:
            metrics = sweep_change(U + F_HJB, U, M + F_FP, M, integrate, dt)
            converged, reason = self._judge(metrics, tolerance, absolute_tolerance)
            return converged, reason, metrics

        return verdict

    @staticmethod
    def _judge(metrics: dict[str, float], tolerance: float, absolute_tolerance: float | None) -> tuple[bool, str]:
        """The criteria on one sweep's change (`check_convergence_criteria`), which every verdict of the solve reads: a
        warm-up sweep's and the map's at each iterate Newton evaluates (#2997)."""
        return check_convergence_criteria(
            metrics["l2distu_rel"],
            metrics["l2distm_rel"],
            metrics["l2distu_abs"],
            metrics["l2distm_abs"],
            tolerance,
            absolute_tolerance,
        )

    def _iteration_state(self) -> IterationState:
        return self.mfg_residual.state

    def _stop_on_a_diverged_map(self, stop: _MapDivergedError, start_time: float) -> SolverResult:
        """Publish what the guard published and stop: the iterate the verdict judged, with the map's non-finite HJB
        output as U (#1718's contract), after ``stop.sweep - picard_warmup`` Newton iterations."""
        self.U, self.M = stop.U, stop.M
        self.newton_residuals = []
        self._solution_computed = True
        return self._create_result(
            False, "diverged_nan", time.time() - start_time, newton_iterations=stop.sweep - self.picard_warmup
        )

    def _create_result(
        self,
        converged: bool,
        reason: str,
        elapsed: float,
        newton_info: SolverInfo | None = None,
        newton_iterations: int | None = None,
        picard_iterations: int | None = None,
    ) -> SolverResult:
        """The one result type (#2997 ruling 4), its metadata the convergence information. ``newton_iterations``
        overrides the count read from ``newton_residuals``, for a stop that has no residual history, and
        ``picard_iterations`` the configured warm-up count, for a warm-up that converged before it. Its error histories
        are the absolute change at each sweep measured, so a warm-up that diverged at sweep i reports i iterations."""
        info = {
            "converged": converged,
            "convergence_reason": reason,
            "picard_iterations": self.picard_warmup if picard_iterations is None else picard_iterations,
            "picard_residuals": self.picard_residuals,
            "newton_iterations": len(self.newton_residuals) if newton_iterations is None else newton_iterations,
            "newton_residuals": self.newton_residuals,
            "residual_evaluations": self.mfg_residual.residual_evaluations,
            "elapsed_time": elapsed,
        }

        if newton_info is not None:
            info["final_residual"] = newton_info.residual
            info["jacobian_evaluations"] = newton_info.extra.get("jacobian_evals", 0)

        # As on every coupler's result: the inner HJB steps the last verdict read, or the latest solve's where no
        # verdict ran; None when the solver does not track them (#2644 review 2, F4).
        failures = getattr(self, "_verdict_inner_failures", _NOT_READ)
        if failures is _NOT_READ:
            failures = read_inner_solve_failures(self.mfg_residual.hjb_solver)
        info["inner_hjb_failures"] = list(failures) if isinstance(failures, tuple) else None

        changes_U, changes_M = self._changes
        return SolverResult(
            U=self.U,
            M=self.M,
            iterations=len(changes_U),
            error_history_U=np.asarray(changes_U, dtype=float),
            error_history_M=np.asarray(changes_M, dtype=float),
            solver_name="NewtonMFGSolver",
            converged=converged,
            metadata=info,
        )

    def get_results(self) -> tuple[NDArray, NDArray]:
        """Get computed solution arrays."""
        if self.U is None or self.M is None:
            raise RuntimeError("Solver has not been run yet. Call solve() first.")
        return self.U, self.M

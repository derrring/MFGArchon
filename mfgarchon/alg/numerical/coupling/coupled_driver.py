"""One driver loop for the coupled routes (#3002, stage 2a).

FixedPointIterator, FictitiousPlayIterator, the Block iterators and NewtonMFGSolver each run their solve here. A route
is a configuration of the driver: its phases, the map each phase evaluates, its share of the budget, and a damping
policy.

**One per-step pipeline** (`CoupledDriver.evaluate`, plans v1.1 and v1.2 on #3002). Every evaluation of the sweep map
T whose output reaches a measure or a verdict goes through it, in every phase: a fixed-point phase's sweep, Newton's
warm-up sweep, its post-warm-up check, and the iterate each Newton step's verdict judges. The phase supplies T's two
halves and their order. The driver reads the owner (`IterationState.read_owner`, #3005), then runs T's HJB half and
records its inner failures (#1878). On a Gauss-Seidel map it then applies the HJB guard (#1718) and runs T's FP half; on
a Jacobi map it runs the FP half, fed the input U, and then the guard. The FP-output rule (`FPOutputRule`, #2997) and
the measure follow. A stop publishes the evaluation's HJB output and, after the FP half, its FP output with its bad
values kept (the approver's ruling E). After each update, a fixed-point sweep's or a Newton step's, the post-update
rule (`updated_iterate_stop`) reads the iterate, and a stop publishes it as it is. The Newton step's own evaluations of
F, the finite-difference Jacobian's probes and the line search's trials, feed the Jacobian and the Armijo test and are
outside the verdict rules (ruling D); a non-finite probe is #2996's folded item. The owner is read before each Newton
step, so its probes and trials are evaluated under the owner too, and before the F of an exhausted budget.

- **The fixed-point step** (`FixedPointStep`): T through the iterator's hand-off (`hjb_solve_kwargs`,
  `fp_solve_kwargs`, `resolve_fp_drift_kwargs`), Gauss-Seidel or Jacobi between the fields, then `DampingRun.update`.
- **Newton's residual map** (`ResidualMap`): `MFGResidual.compute_hjb_output` then `compute_fp_output`, Jacobi, as
  `MFGResidual.compute_residual` runs them; the Newton step is `NewtonRun.step`, the step `NewtonSolver.solve` is made
  of.

The driver also owns the budget (`max_iterations` counts every evaluation that is a sweep of the solve, every
phase's, and 0 runs none), the record and the verdict, the callback's one convention (False stops, #2886), the stops
and their reasons, one progress display, and the measure-field and damping-rate records any route may take.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.utils.convergence import sweep_change
from mfgarchon.utils.mfg_logging import get_logger

from .base_mfg import call_handed
from .coupled_result import DensityRecord, SweepRecord
from .fixed_point_utils import (
    BUDGET_SPENT,
    FPOutputRule,
    check_convergence_criteria,
    diverged_value_function,
    read_inner_solve_failures,
    refuse_convergence_over_failed_inner_solves,
    resolve_fp_drift_kwargs,
    updated_iterate_stop,
    validate_budget,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.utils.numerical.nonlinear_solvers import NewtonRun, SolverInfo

    from .base_mfg import BaseCouplingIterator
    from .damping_run import DampingRun
    from .iteration_state import IterationState
    from .mfg_residual import MFGResidual

logger = get_logger(__name__)


@dataclass
class Evaluation:
    """One evaluation of the sweep map T through the pipeline: T's outputs, and their change, or the stop it met.

    ``stop`` is the HJB guard's or the FP-output rule's reason, and ``published`` the pair a stop publishes: the
    published HJB output (#1718) with the density the evaluation was handed, or, after the FP half, the HJB output with
    the FP output, its bad values kept (ruling E). ``M_new`` is None when the guard stopped before the FP half.
    """

    U_new: NDArray
    M_new: NDArray | None
    U_terminal: NDArray
    metrics: dict[str, float] | None = None
    stop: str | None = None
    published: tuple[NDArray, NDArray] | None = None


class FixedPointStep:
    """A fixed-point phase's map T, for one component, and its update.

    ``iterator`` is the route: a `BaseCouplingIterator` that holds its pair (``hjb_solver``, ``fp_solver``), its
    overrides (``volatility``, ``volatility_kind``, ``drift_field``) and its solvers' signatures
    (``_init_solver_signatures``). ``order`` is ``"gauss_seidel"`` or
    ``"jacobi"``. ``site`` names the route in the guards' warnings. ``fp_solve``, when given, replaces the FP half's
    hand-off: Block's ``adjoint_mode`` solves the FP with the transposed HJB operator (#622).
    """

    def __init__(
        self,
        iterator: BaseCouplingIterator,
        state: IterationState,
        run: DampingRun,
        *,
        order: str,
        site: str,
        fp_solve: Callable[[NDArray, NDArray], NDArray] | None = None,
    ) -> None:
        if order not in ("gauss_seidel", "jacobi"):
            raise ValueError(f"FixedPointStep: order is 'gauss_seidel' or 'jacobi', not {order!r}")
        self.iterator = iterator
        self.state = state
        self.run = run
        self.order = order
        self.site = site
        self.fp_solve = fp_solve
        self.integrate = iterator.problem.spatial_measure().integrate  # #2555
        self.time_step = iterator.problem.dt

    def terminal(self, M_in: NDArray) -> NDArray:
        """u(T) at the density this evaluation's HJB solve uses (#1642 stage 2): the solve, the restore after a diverged
        step and the re-imposition after the update all read this one."""
        return self.state.terminal_at(M_in[-1])

    def hjb(self, U_in: NDArray, M_in: NDArray, U_terminal: NDArray) -> NDArray:
        """T's HJB half: u solved backward at the density ``M_in``, with the problem's sources composed (#2207)."""
        it = self.iterator
        kwargs = it._build_hjb_kwargs(M=M_in, U=U_in, volatility=it.volatility, volatility_kind=it.volatility_kind)
        return call_handed(it.hjb_solver.solve_hjb_system, M_in, U_terminal, U_in, **kwargs)

    def fp(self, U_for_fp: NDArray, M_in: NDArray) -> NDArray:
        """T's FP half: m solved forward from m(0), driven by ``U_for_fp`` through the one drift convention (#1233). The
        FP solver's signature was read when the route was built, which refuses one it cannot read or that names neither
        ``potential_field`` nor ``drift_field`` (`solver_signature`)."""
        M_initial = self.state.m0
        if self.fp_solve is not None:
            return call_handed(self.fp_solve, M_initial, U_for_fp)
        it = self.iterator
        kwargs = it._build_fp_kwargs(M=M_in, U=U_for_fp, volatility=it.volatility, volatility_kind=it.volatility_kind)
        drift_kwargs = resolve_fp_drift_kwargs(
            it.problem,
            it._fp_sig_params,
            it.drift_field,
            U_for_fp,
            M_in,
            drift_convention=it._fp_drift_convention,
            fp_solver=it.fp_solver,
        )
        kwargs.update(drift_kwargs)
        return call_handed(it.fp_solver.solve_fp_system, M_initial, **kwargs)

    def measured(self, U_new: NDArray, M_new: NDArray, U_in: NDArray, M_in: NDArray) -> tuple[NDArray, NDArray]:
        """The outputs the change is measured on: T's own."""
        return U_new, M_new

    def advance(self, k: int, U_in: NDArray, M_in: NDArray, evaluation: Evaluation) -> tuple[NDArray, NDArray]:
        """The next iterate: the damping, the write-back (level 0 to m(0), U's last row to this evaluation's u(T)), and an
        adaptive field's adaptation on its own history, from this sweep's change (#2997)."""
        U_next, M_next = self.run.update(k, U_in, evaluation.U_new, M_in, evaluation.M_new)
        U_next, M_next = self.state.write_back(U_next, M_next, evaluation.U_terminal)
        for message in self.run.observe(k, evaluation.metrics):
            logger.warning(message)
        return U_next, M_next


class ResidualMap:
    """Newton's map at an iterate: `MFGResidual.compute_hjb_output` then `compute_fp_output`, the FP fed the input U
    (Jacobi), as `MFGResidual.compute_residual` runs them and counts them. The change is measured on ``U + F_HJB`` and
    ``M + F_FP``, the map's outputs as the residual ``F`` carries them (#2565)."""

    order = "jacobi"

    def __init__(self, residual: MFGResidual, site: str) -> None:
        self.residual = residual
        self.site = site
        self.integrate = residual.problem.spatial_measure().integrate
        self.time_step = residual.problem.dt

    def terminal(self, M_in: NDArray) -> NDArray:
        return self.residual.terminal_at(M_in)

    def hjb(self, U_in: NDArray, M_in: NDArray, U_terminal: NDArray) -> NDArray:
        self.residual.residual_evaluations += 1
        return self.residual.compute_hjb_output(M_in, U_in)

    def fp(self, U_for_fp: NDArray, M_in: NDArray) -> NDArray:
        return self.residual.compute_fp_output(U_for_fp, M_in)

    def measured(self, U_new: NDArray, M_new: NDArray, U_in: NDArray, M_in: NDArray) -> tuple[NDArray, NDArray]:
        return U_in + (U_new - U_in), M_in + (M_new - M_in)


@dataclass
class Outcome:
    """How a phase ended: ``stop`` is None when its budget ran out, else ``"met"``, ``"diverged"`` (the HJB guard, the
    FP-output rule or the post-update rule, its reason the driver's), ``"callback"``, or a Newton phase's named stop.
    ``newton_info`` is the Newton phase's `SolverInfo`; ``detail["sweep"]`` is the index of the evaluation a
    ``"diverged"`` stop met, or of the evaluation whose update the post-update rule stopped."""

    U: NDArray
    M: NDArray
    stop: str | None
    sweeps: int
    newton_info: SolverInfo | None = None
    detail: dict[str, Any] = field(default_factory=dict)


class CoupledDriver:
    """The driver of one solve, used as a context manager around its phases: its progress display opens and closes with
    it. The record, the verdict, the reason and the histories are the solve's, across its phases.

    Args:
        route: The route's name, in the progress display.
        reader: The route's class, which a refusal names.
        state: The iterate's owner (#2997), whose `read_owner` runs before every step (#3005).
        hjb_solver: The HJB solver whose inner failures the verdict reads (#1878).
        budget: Sweeps the solve may run, every phase's; 0 runs none.
        tolerance, absolute_tolerance: The verdict's criteria (`check_convergence_criteria`).
        iteration_callback: ``callback(iteration, U, M, error_U, error_M)``; a return of False stops the solve, and the
            verdict is then read at that iterate (#1684, #2886). True and None continue.
        verbose: Whether the progress display shows.
        track_measure_field: Record each sweep's iterate as a measure-field snapshot (#956); a `CartesianGrid` only.
        track_rates: Record each sweep's M damping rate.
    """

    def __init__(
        self,
        route: str,
        state: IterationState,
        hjb_solver: Any,
        *,
        reader: str,
        budget: int,
        tolerance: float,
        absolute_tolerance: float | None,
        iteration_callback: Any = None,
        verbose: bool = False,
        track_measure_field: bool = False,
        track_rates: bool = False,
    ) -> None:
        self.budget = validate_budget(budget, reader)
        self.route = route
        self.state = state
        self.hjb_solver = hjb_solver
        self.tolerance = tolerance
        self.absolute_tolerance = absolute_tolerance
        self.iteration_callback = iteration_callback
        self.verbose = verbose
        self.record = SweepRecord(1)
        self.converged = False
        self.reason = BUDGET_SPENT
        self.sweeps = 0
        # What produced the density the solve publishes: the one decider of whether it is measured (#1717; #3008)
        self.density = DensityRecord()
        # The FP-output rule, which holds this solve's references (the user's ruling of 2026-11-27)
        self.fp_output_rule = FPOutputRule([state.problem.spatial_measure().integrate])
        self.inner_failures: object = None
        self.rates: list[float] | None = [] if track_rates else None
        # The change of each sweep, NaN until measured, as the routes' l2dist* attributes report it
        self.histories = {
            key: np.full(self.budget, np.nan) for key in ("l2distu_abs", "l2distm_abs", "l2distu_rel", "l2distm_rel")
        }
        self.measure_field = None
        if track_measure_field:
            from mfgarchon.geometry.base import CartesianGrid

            problem = state.problem
            if not isinstance(problem.geometry, CartesianGrid):
                raise NotImplementedError(
                    "track_measure_field uses GridMeasureField (get_spatial_grid) and is only "
                    "supported for CartesianGrid geometry, not unstructured mesh."
                )
            from mfgarchon.core.measure_field import GridMeasureField

            self._grid_1d = problem.geometry.get_spatial_grid().ravel()
            self.measure_field = GridMeasureField(self._grid_1d, np.linspace(0, problem.T, problem.Nt + 1))
        self._progress: Any = None
        self._task: Any = None

    def __enter__(self) -> CoupledDriver:
        from mfgarchon.utils.progress import HierarchicalProgress

        self._progress_cm = HierarchicalProgress(verbose=self.verbose)
        self._progress = self._progress_cm.__enter__()
        self._task = self._progress.add_task(
            self.route, total=self.budget, iter=f"0/{self.budget}", err_U=0.0, err_M=0.0
        )
        return self

    def __exit__(self, *exc: object) -> None:
        self._progress_cm.__exit__(*exc)

    @property
    def fp_output_available(self) -> bool:
        """Whether the density the solve publishes is one it produced (`DensityRecord.measured`)."""
        return self.density.measured

    @property
    def remaining(self) -> int:
        return self.budget - self.sweeps

    def snapshot(self, U: NDArray, M: NDArray) -> None:
        """A measure-field snapshot of the iterate (#956), its terminal density as the measure's representative: it
        varies most across iterates, where m(0) is fixed."""
        if self.measure_field is not None:
            from mfgarchon.core.measure import ParticleMeasure

            self.measure_field.add_snapshot(ParticleMeasure.from_density(M[-1], self._grid_1d), U.copy())

    def _measured(self, k: int, metrics: dict[str, float], started: float) -> None:
        for key, history in self.histories.items():
            history[k] = metrics[key]
        self.record.measured([metrics])
        self._progress.update(
            self._task,
            iter=f"{k + 1}/{self.budget}",
            err_U=metrics["l2distu_rel"],
            err_M=metrics["l2distm_rel"],
            time=f"{time.time() - started:.1f}s",
        )

    def _verdict(self, metrics: dict[str, float], prefix: str = "") -> tuple[bool, bool, str]:
        """``(met, converged, reason)``: the criteria on this change, and the verdict refused over failed inner HJB
        steps of this sweep (#1878). ``prefix`` names the phase in a met criterion's reason, inside a refusal's text."""
        met, reason = check_convergence_criteria(
            metrics["l2distu_rel"],
            metrics["l2distm_rel"],
            metrics["l2distu_abs"],
            metrics["l2distm_abs"],
            self.tolerance,
            self.absolute_tolerance,
        )
        if met and prefix:
            reason = f"{prefix}{reason}"
        converged, reason = refuse_convergence_over_failed_inner_solves(met, reason, self.inner_failures)
        return met, converged, reason

    def _callback_stops(self, k: int, U: NDArray, M: NDArray, metrics: dict[str, float]) -> bool:
        """The callback's one convention (#2886): False stops, and the verdict is read at this iterate, so an abort is
        evidence neither for convergence nor against it (#1684)."""
        if self.iteration_callback is None:
            return False
        if self.iteration_callback(k, U, M, metrics["l2distu_rel"], metrics["l2distm_rel"]) is not False:
            return False
        _met, self.converged, criteria_reason = self._verdict(metrics)
        self.reason = f"callback_stopped ({criteria_reason})" if criteria_reason else "callback_stopped"
        return True

    def evaluate(self, k: int, the_map: Any, U_in: NDArray, M_in: NDArray) -> Evaluation:
        """One evaluation of ``the_map`` at ``(U_in, M_in)``, through every per-step rule: the owner read (#3005), the HJB
        half and the inner-failure record (#1878); then the HJB guard (#1718) and the FP half on a Gauss-Seidel map, or
        the FP half and then the guard on a Jacobi map; the FP-output rule (#2997), and the measure. ``k`` is the
        evaluation's 0-based index, which the guards' warnings report."""
        self.state.read_owner()
        U_terminal = the_map.terminal(M_in)
        U_new = the_map.hjb(U_in, M_in, U_terminal)
        self.inner_failures = read_inner_solve_failures(self.hjb_solver)
        # Under Jacobi the FP is fed the input U, so a non-finite U_new reaches nothing before the guard below stops.
        M_new = the_map.fp(U_in, M_in) if the_map.order == "jacobi" else None
        diverged = diverged_value_function(U_new, U_terminal, site=the_map.site, density=M_in, iteration=k)
        if diverged is not None:
            self.reason = "diverged_nan"
            self.record.diverged()
            return Evaluation(U_new, None, U_terminal, stop="diverged_nan", published=(diverged, M_in))
        if M_new is None:
            M_new = the_map.fp(U_new, M_in)
        stop = self.fp_output_rule.stop([M_new], [M_in], site=the_map.site, iteration=k)
        if stop is not None:
            self.reason = stop
            self.record.diverged()
            self.density.fp_outputs_published += 1
            return Evaluation(U_new, M_new, U_terminal, stop=stop, published=(U_new, M_new))
        U_measured, M_measured = the_map.measured(U_new, M_new, U_in, M_in)
        metrics = sweep_change(U_measured, U_in, M_measured, M_in, the_map.integrate, the_map.time_step)
        return Evaluation(U_new, M_new, U_terminal, metrics=metrics)

    def run_fixed_point(
        self, step: FixedPointStep, U: NDArray, M: NDArray, budget: int, *, reason_prefix: str = ""
    ) -> Outcome:
        """Up to ``budget`` sweeps of ``step``, from ``(U, M)``. A met criterion stops the phase whether or not its
        verdict is refused over failed inner steps: further sweeps only re-certify the same point (#1878).
        ``reason_prefix`` names the phase in a met criterion's reason."""
        ran = 0
        for _ in range(budget):
            k = self.sweeps
            started = time.time()
            if self.rates is not None:
                self.rates.append(step.run.theta("M", k))
            evaluation = self.evaluate(k, step, U, M)
            self.sweeps += 1
            ran += 1
            if evaluation.stop is not None:
                U, M = evaluation.published
                return Outcome(U, M, "diverged", ran, detail={"sweep": k})
            U, M = step.advance(k, U, M, evaluation)
            self.density.updates += 1
            stop = updated_iterate_stop([U], [M], producer="by the sweep's update", site=step.site, iteration=k)
            if stop is not None:
                self.reason = stop
                self.record.diverged()
                return Outcome(U, M, "diverged", ran, detail={"sweep": k})
            self.snapshot(U, M)
            self._measured(k, evaluation.metrics, started)
            if self._callback_stops(k, U, M, evaluation.metrics):
                return Outcome(U, M, "callback", ran)
            met, self.converged, self.reason = self._verdict(evaluation.metrics, reason_prefix)
            if met:
                return Outcome(U, M, "met", ran)
        if self.reason == "" or not ran:
            self.reason = BUDGET_SPENT
        return Outcome(U, M, None, ran)

    def run_newton(
        self,
        newton: NewtonRun,
        budget: int,
        the_map: ResidualMap,
        first_index: int,
        pack: Callable[[NDArray, NDArray], NDArray],
        unpack: Callable[[NDArray], tuple[NDArray, NDArray]],
        on_verdict: Callable[[bool, str], None],
    ) -> Outcome:
        """Up to ``budget`` Newton steps of ``newton``. At each, the iterate's evaluation goes through the pipeline, its
        residual ``F = (U + F_HJB, M + F_FP) - (U, M)`` is packed for Newton, the verdict judges its change (refused over
        the failed inner steps of that evaluation), the callback runs, the owner is read, then the step, and the
        post-update rule reads the iterate it moved. ``first_index`` is the index of the first iterate's evaluation.
        When the budget is spent, the owner is read and F is evaluated at the last iterate without a verdict, as
        `NewtonSolver.solve` does."""
        for iteration in range(budget):
            started = time.time()
            k = first_index + iteration
            U_x, M_x = unpack(newton.x)
            evaluation = self.evaluate(k, the_map, U_x, M_x)
            self.sweeps += 1
            if evaluation.stop is not None:
                U, M = evaluation.published
                return Outcome(U, M, "diverged", iteration + 1, detail={"sweep": k})
            F_current = pack(evaluation.U_new - U_x, evaluation.M_new - M_x)
            residual_norm = newton.accept(F_current)
            self.record.measured([evaluation.metrics])
            _met, converged, reason = self._verdict(evaluation.metrics)
            on_verdict(converged, reason)
            if converged:
                x, info = newton.converged(iteration, residual_norm)
                U, M = unpack(x)
                return Outcome(U, M, "met", iteration + 1, newton_info=info)
            self._progress.update(
                self._task,
                iter=f"{self.sweeps}/{self.budget}",
                err_U=evaluation.metrics["l2distu_rel"],
                err_M=evaluation.metrics["l2distm_rel"],
                time=f"{time.time() - started:.1f}s",
            )
            if self._callback_stops(k, U_x, M_x, evaluation.metrics):
                return Outcome(U_x, M_x, "callback", iteration + 1)
            # The step's probes and line-search trials evaluate the map under the owner too (#3005)
            self.state.read_owner()
            stopped = newton.step(F_current, residual_norm, iteration)
            if stopped is not None:
                x, info = stopped
                U, M = unpack(x)
                return Outcome(U, M, str(info.extra.get("reason")), iteration + 1, newton_info=info)
            self.density.newton_steps = newton.steps_taken
            U_step, M_step = unpack(newton.x)
            stop = updated_iterate_stop([U_step], [M_step], producer="by Newton's step", site=the_map.site, iteration=k)
            if stop is not None:
                self.reason = stop
                return Outcome(U_step, M_step, "diverged", iteration + 1, detail={"sweep": k})
        self.state.read_owner()
        x, info = newton.exhausted(budget)
        U, M = unpack(x)
        return Outcome(U, M, None, budget, newton_info=info)

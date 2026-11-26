"""One driver loop for the coupled routes (#3002, stage 2a).

FixedPointIterator, FictitiousPlayIterator, the Block iterators and NewtonMFGSolver each run their sweeps here. A route
is a configuration of the driver: a step function per phase, its share of the budget, and a damping policy.

- **The fixed-point step** (`FixedPointStep`) is T, then `DampingRun.update`. T is an HJB solve, then an FP solve, for
  one component, through the iterator's hand-off (`hjb_solve_kwargs`, `fp_solve_kwargs` and `resolve_fp_drift_kwargs`).
  Its option is the order between the fields: Gauss-Seidel feeds the FP the HJB's output, Jacobi the input.
- **The Newton step** is one `NewtonRun.step` on the residual of the Jacobi map (`MFGResidual`), the same step
  `NewtonSolver.solve` is made of.

The driver owns, for every route: the budget (`max_iterations` counts every sweep of the solve, every phase's, and 0
runs none); the owner read before every step (`IterationState.read_owner`, #3005); the HJB guard (#1718) and the
FP-output rule (`fp_output_stop`, #2997); the measure, the record and the verdict; the callback's one convention
(False stops, #2886); the stops and their reasons; one progress display; and the measure-field and damping-rate
records any route may take.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.utils.convergence import sweep_change
from mfgarchon.utils.mfg_logging import get_logger

from .coupled_result import SweepRecord
from .fixed_point_utils import (
    BUDGET_SPENT,
    check_convergence_criteria,
    diverged_value_function,
    fp_output_stop,
    read_inner_solve_failures,
    refuse_convergence_over_failed_inner_solves,
    resolve_fp_drift_kwargs,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.utils.numerical.nonlinear_solvers import NewtonRun, SolverInfo

    from .damping_run import DampingRun
    from .iteration_state import IterationState

logger = get_logger(__name__)


@dataclass
class Sweep:
    """What one fixed-point step produced: the map's outputs, and the iterate after the update and the write-back.

    ``diverged`` holds the published form of a non-finite HJB output (#1718), and then ``M_new`` is None: the FP was
    not solved with it.
    """

    U_old: NDArray
    M_old: NDArray
    U_new: NDArray
    M_new: NDArray | None
    U: NDArray
    M: NDArray
    diverged: NDArray | None = None


class FixedPointStep:
    """T, then `DampingRun.update`: one sweep of a fixed-point phase, for one component.

    ``iterator`` is the route: a `BaseCouplingIterator` that holds its pair (``hjb_solver``, ``fp_solver``), its
    overrides (``volatility``, ``volatility_kind``, ``drift_field``) and its solvers' signatures
    (``_init_solver_signatures``), which the base class does not declare. ``order`` is ``"gauss_seidel"`` or
    ``"jacobi"``. ``site`` names the route in the guards' warnings. ``fp_solve``, when given, replaces the FP half's
    hand-off: Block's ``adjoint_mode`` solves the FP with the transposed HJB operator (#622).
    """

    def __init__(
        self,
        iterator: Any,
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

    def hjb(self, M_old: NDArray, U_old: NDArray, U_terminal: NDArray) -> NDArray:
        """The HJB half: u solved backward at the density ``M_old``, with the problem's sources composed (#2207)."""
        it = self.iterator
        kwargs = it._build_hjb_kwargs(M=M_old, U=U_old, volatility=it.volatility, volatility_kind=it.volatility_kind)
        return it.hjb_solver.solve_hjb_system(M_old, U_terminal, U_old, **kwargs)

    def fp(self, M_initial: NDArray, U_in: NDArray, M_old: NDArray) -> NDArray:
        """The FP half: m solved forward from m(0), driven by ``U_in`` through the one drift convention (#1233)."""
        if self.fp_solve is not None:
            return self.fp_solve(M_initial, U_in)
        it = self.iterator
        if it._fp_sig_params is None:
            return it.fp_solver.solve_fp_system(M_initial, U_in)
        kwargs = it._build_fp_kwargs(M=M_old, U=U_in, volatility=it.volatility, volatility_kind=it.volatility_kind)
        drift_kwargs, use_positional_U = resolve_fp_drift_kwargs(
            it.problem,
            it._fp_sig_params,
            it.drift_field,
            U_in,
            M_old,
            drift_convention=it._fp_drift_convention,
            fp_solver=it.fp_solver,
        )
        kwargs.update(drift_kwargs)
        if use_positional_U:
            return it.fp_solver.solve_fp_system(M_initial, U_in, **kwargs)
        return it.fp_solver.solve_fp_system(M_initial, **kwargs)

    def __call__(self, k: int, U: NDArray, M: NDArray) -> Sweep:
        U_old = U.copy()
        M_old = M.copy()
        # u(T) at the density this sweep's HJB solve uses (#1642 stage 2): the solve, the restore after a diverged step
        # and the re-imposition after the update all read this one.
        U_terminal = self.state.terminal_at(M_old[-1])
        U_new = self.hjb(M_old, U_old, U_terminal)
        if self.order == "jacobi":
            # The FP is fed the input U, so a non-finite U_new reaches nothing before the guard below stops.
            M_new = self.fp(self.state.m0, U_old, M_old)
        diverged = diverged_value_function(U_new, U_terminal, site=self.site, density=M_old, iteration=k)
        if diverged is not None:
            return Sweep(U_old, M_old, U_new, None, U_old, M_old, diverged=diverged)
        if self.order == "gauss_seidel":
            M_new = self.fp(self.state.m0, U_new, M_old)
        U_next, M_next = self.run.update(k, U_old, U_new, M_old, M_new)
        U_next, M_next = self.state.write_back(U_next, M_next, U_terminal)
        return Sweep(U_old, M_old, U_new, M_new, U_next, M_next)

    def measure(self, k: int, sweep: Sweep) -> dict[str, float]:
        """The map's output against its input (#1684 item 7), in the problem's own measure (#2555); an adaptive field
        adapts on its own history, from this sweep's change (#2997)."""
        metrics = sweep_change(sweep.U_new, sweep.U_old, sweep.M_new, sweep.M_old, self.integrate, self.time_step)
        for message in self.run.observe(k, metrics):
            logger.warning(message)
        return metrics


@dataclass
class Outcome:
    """How a phase ended: ``stop`` is None when its budget ran out, else ``"met"``, ``"diverged"``, ``"callback"``, or
    a Newton phase's named stop. ``newton_info`` is the Newton phase's `SolverInfo`."""

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
        budget: int,
        tolerance: float,
        absolute_tolerance: float | None,
        iteration_callback: Any = None,
        verbose: bool = False,
        track_measure_field: bool = False,
        track_rates: bool = False,
    ) -> None:
        if not isinstance(budget, (int, np.integer)) or isinstance(budget, bool) or budget < 0:
            raise ValueError(f"{route}: max_iterations is a count of sweeps, an int >= 0, not {budget!r}")
        self.route = route
        self.state = state
        self.hjb_solver = hjb_solver
        self.budget = int(budget)
        self.tolerance = tolerance
        self.absolute_tolerance = absolute_tolerance
        self.iteration_callback = iteration_callback
        self.verbose = verbose
        self.record = SweepRecord(1)
        self.converged = False
        self.reason = BUDGET_SPENT
        self.sweeps = 0
        # True once the iterate holds a density this solve's FP produced: what mass_conservation_error measures (#1717)
        self.fp_output_available = False
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
            self.state.read_owner()
            if self.rates is not None:
                self.rates.append(step.run.theta("M", k))
            sweep = step(k, U, M)
            self.inner_failures = read_inner_solve_failures(self.hjb_solver)
            self.sweeps += 1
            ran += 1
            if sweep.diverged is not None:
                self.reason = "diverged_nan"
                self.record.diverged()
                return Outcome(sweep.diverged, sweep.M_old, "diverged", ran)
            U, M = sweep.U, sweep.M
            self.fp_output_available = True
            stop = fp_output_stop(
                [sweep.U_new], [sweep.M_new], [U], [M], [sweep.M_old], each_component=True, site=step.site, iteration=k
            )
            if stop is not None:
                self.reason = stop
                self.record.diverged()
                return Outcome(U, M, "diverged", ran)
            self.snapshot(U, M)
            metrics = step.measure(k, sweep)
            self._measured(k, metrics, started)
            if self._callback_stops(k, U, M, metrics):
                return Outcome(U, M, "callback", ran)
            met, self.converged, self.reason = self._verdict(metrics, reason_prefix)
            if met:
                return Outcome(U, M, "met", ran)
        if self.reason == "" or not ran:
            self.reason = BUDGET_SPENT
        return Outcome(U, M, None, ran)

    def run_newton(
        self,
        newton: NewtonRun,
        budget: int,
        verdict: Callable[[NDArray, NDArray], tuple[bool, str]],
        unpack: Callable[[NDArray], tuple[NDArray, NDArray]],
        metrics_of_last_verdict: Callable[[], dict[str, float]],
    ) -> Outcome:
        """Up to ``budget`` Newton steps of ``newton``: at each, the owner read, F at the iterate, ``verdict`` on it, the
        callback, then the step. ``verdict(x, F)`` is the route's, which records each iterate it judges. When the budget
        is spent, F is evaluated at the last iterate without a verdict, as `NewtonSolver.solve` does."""
        for iteration in range(budget):
            started = time.time()
            self.state.read_owner()
            F_current, residual_norm = newton.evaluate()
            self.sweeps += 1
            if verdict(newton.x, F_current)[0]:
                x, info = newton.converged(iteration, residual_norm)
                U, M = unpack(x)
                return Outcome(U, M, "met", iteration + 1, newton_info=info)
            metrics = metrics_of_last_verdict()
            U, M = unpack(newton.x)
            self._progress.update(
                self._task,
                iter=f"{self.sweeps}/{self.budget}",
                err_U=metrics["l2distu_rel"],
                err_M=metrics["l2distm_rel"],
                time=f"{time.time() - started:.1f}s",
            )
            if self._callback_stops(self.sweeps - 1, U, M, metrics):
                return Outcome(U, M, "callback", iteration + 1)
            stopped = newton.step(F_current, residual_norm, iteration)
            if stopped is not None:
                x, info = stopped
                U, M = unpack(x)
                return Outcome(U, M, str(info.extra.get("reason")), iteration + 1, newton_info=info)
        x, info = newton.exhausted(budget)
        U, M = unpack(x)
        return Outcome(U, M, None, budget, newton_info=info)

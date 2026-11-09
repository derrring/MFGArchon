"""
Fixed Point Iterator Utilities

Shared helper functions for fixed-point iteration solvers to eliminate code duplication.
These utilities are used by both legacy and config-aware fixed-point iterator implementations.
"""

from __future__ import annotations

import functools
import inspect
from collections import Counter
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar

import numpy as np

from mfgarchon.core.hamiltonian import HEvalState, _population_rows
from mfgarchon.geometry.grids.tensor_grid import geometry_grid_axes
from mfgarchon.utils.convergence.convergence_metrics import refuse_non_finite_change
from mfgarchon.utils.mfg_logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from mfgarchon.alg.numerical.fp_solvers.base_fp import DriftConvention
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.utils.solver_result import SolverResult

_Solver = TypeVar("_Solver")
_Params = ParamSpec("_Params")
_Result = TypeVar("_Result")


def fp_solver_sig_params(fp_solver: object) -> set[str] | None:
    """Return the parameter names of ``fp_solver.solve_fp_system`` (or ``None``).

    Mirrors :meth:`BaseCouplingIterator._init_solver_signatures` for iterators that hold a
    *list* of per-subsystem FP solvers (regime-switching, graph MFG) and therefore cache one
    signature set per solver rather than a single shared one. The cached set is the
    ``fp_sig_params`` argument of :func:`resolve_fp_drift_kwargs`, which decides whether the
    value function is routed via ``potential_field`` (smooth separable $H$, FP derives the
    velocity) or ``drift_field`` (Issue #1315).
    """
    try:
        return set(inspect.signature(fp_solver.solve_fp_system).parameters.keys())
    except (AttributeError, ValueError):
        return None


def hjb_solver_sig_params(hjb_solver: BaseHJBSolver) -> set[str] | None:
    """Return the parameter names of ``hjb_solver.solve_hjb_system`` (or ``None``).

    :func:`fp_solver_sig_params`'s HJB twin, for an iterator that holds one HJB solver per
    population and hands each set to :func:`~.base_mfg.hjb_solve_kwargs` (#2666).
    """
    try:
        return set(inspect.signature(hjb_solver.solve_hjb_system).parameters.keys())
    except (AttributeError, ValueError):
        return None


def initialize_cold_start(
    U: np.ndarray,
    M: np.ndarray,
    M_initial: np.ndarray,
    U_terminal: np.ndarray,
    Nt: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize U and M arrays for cold start (no warm start provided).

    Cold start initialization sets:
    - U[Nt-1, :] = U_terminal (terminal condition)
    - M[0, :] = M_initial (initial condition)
    - U[0:Nt-1, :] = U_terminal (interior initialized with terminal condition)
    - M[1:Nt, :] = M_initial (interior initialized with initial condition)

    Args:
        U: Value function array to initialize (modified in-place)
        M: Density array to initialize (modified in-place)
        M_initial: Initial density distribution m_0(x)
        U_terminal: The terminal value at the initial density, ``problem.terminal_value(M_initial)``
        Nt: Number of time steps

    Returns:
        Tuple of (initialized U, initialized M)

    Note:
        This is the standard initialization used when no warm start is provided.
        Interior points are set to boundary conditions as a simple initial guess.
    """
    # Set boundary conditions
    U[Nt - 1, :] = U_terminal
    M[0, :] = M_initial

    # Initialize interior with boundary conditions
    for n_time_idx in range(Nt - 1):
        U[n_time_idx, :] = U_terminal
    for n_time_idx in range(1, Nt):
        M[n_time_idx, :] = M_initial

    return U, M


def construct_solver_result(
    U: np.ndarray,
    M: np.ndarray,
    iterations_run: int,
    l2distu_abs: np.ndarray,
    l2distm_abs: np.ndarray,
    l2distu_rel: np.ndarray,
    l2distm_rel: np.ndarray,
    solver_name: str,
    converged: bool,
    convergence_reason: str,
) -> SolverResult:
    """
    Construct a SolverResult object from fixed-point iteration data.

    Args:
        U: Final value function array
        M: Final density array
        iterations_run: Number of iterations executed
        l2distu_abs: Absolute L2 errors for U (full array)
        l2distm_abs: Absolute L2 errors for M (full array)
        l2distu_rel: Relative L2 errors for U (full array)
        l2distm_rel: Relative L2 errors for M (full array)
        solver_name: Name of the solver
        converged: Whether the solver converged
        convergence_reason: Reason for convergence/termination

    Returns:
        SolverResult object with all diagnostic information

    Note:
        Error arrays are truncated to iterations_run length before storage.
    """
    from mfgarchon.utils.solver_result import SolverResult

    # Truncate error arrays to actual iterations run
    l2distu_abs_truncated = l2distu_abs[:iterations_run]
    l2distm_abs_truncated = l2distm_abs[:iterations_run]
    l2distu_rel_truncated = l2distu_rel[:iterations_run]
    l2distm_rel_truncated = l2distm_rel[:iterations_run]

    # Construct result object
    result = SolverResult(
        U=U,
        M=M,
        converged=converged,
        iterations=iterations_run,
        convergence_reason=convergence_reason,
        diagnostics={
            "l2distu_abs": l2distu_abs_truncated,
            "l2distm_abs": l2distm_abs_truncated,
            "l2distu_rel": l2distu_rel_truncated,
            "l2distm_rel": l2distm_rel_truncated,
            "solver_name": solver_name,
        },
    )

    return result


def apply_damping(
    U_new: np.ndarray,
    U_old: np.ndarray,
    M_new: np.ndarray,
    M_old: np.ndarray,
    theta: float,
    theta_M: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply damping to fixed-point iteration updates.

    Damping formula:
        U_damped = theta * U_new + (1 - theta) * U_old
        M_damped = theta_M * M_new + (1 - theta_M) * M_old

    Args:
        U_new: New value function from HJB solve
        U_old: Previous value function
        M_new: New density from FP solve
        M_old: Previous density
        theta: Damping parameter for U in [0, 1] (0=no update, 1=full update)
            Also used for M if theta_M is None (backward compatible).
        theta_M: Optional separate damping parameter for M in [0, 1].
            If None, uses theta for both U and M (backward compatible).
            Issue #719: Per-variable damping support.

    Returns:
        Tuple of (U_damped, M_damped)

    Note:
        theta=1 corresponds to no damping (full update)
        theta=0.5 is a common choice for stability
        Smaller theta increases stability but slows convergence

    Example:
        # Same damping for both (backward compatible)
        U, M = apply_damping(U_new, U_old, M_new, M_old, theta=0.5)

        # Different damping: full update for U, conservative for M
        # Recommended for MFG: U adapts quickly, M filters particle noise
        U, M = apply_damping(U_new, U_old, M_new, M_old, theta=1.0, theta_M=0.2)
    """
    # Issue #719: Support per-variable damping
    theta_U = theta
    if theta_M is None:
        theta_M = theta  # Backward compatible: same damping for both

    U_damped = theta_U * U_new + (1 - theta_U) * U_old
    M_damped = theta_M * M_new + (1 - theta_M) * M_old

    return U_damped, M_damped


def check_convergence_criteria(
    l2distu_rel: float,
    l2distm_rel: float,
    l2distu_abs: float,
    l2distm_abs: float,
    tol_picard: float,
    absolute_tolerance: float | None = None,
) -> tuple[bool, str]:
    """Whether a coupling sweep's change meets the outer tolerance: the one owner of the verdict (#2555).

    The inputs are what ``sweep_change`` returns for the map's output against its
    input, so the pair compared, and the measure it is compared in, are fixed there.

    - ``tol_picard`` bounds the larger RELATIVE change, ``max(l2distu_rel, l2distm_rel)``
      (``docs/user/CONVENTIONS.md`` § 9). The ``max`` over the two fields is half the criterion: a
      density settling while the value function diverges is not convergence (#1914).
    - ``absolute_tolerance``, when given, must ALSO bound ``max(l2distu_abs, l2distm_abs)``. It is
      ``None`` by default, and then no absolute quantity is compared.

    A multi-field iterator (populations, regimes, graph nodes) passes, for each of the four values, the
    max over its fields of that field's change in its own measure.

    A non-finite change is refused, naming the value (#2578). ``max`` drops a NaN in its second argument,
    since every comparison with NaN is false, so a NaN density change used to leave the verdict to the
    value function's change alone and could report convergence. A multi-field iterator aggregates
    through :func:`~mfgarchon.utils.convergence.worst_sweep_change`, which refuses first and names the
    field.

    Raises:
        ValueError: one of the four changes is NaN or infinite.

    Returns:
        ``(converged, reason)``; ``reason`` is empty when not converged.

    Example:
        >>> check_convergence_criteria(1e-7, 1e-8, 3.0, 2.0, 1e-6)
        (True, 'Converged: Rel err 1.0e-07 < tol 1.0e-06')
        >>> check_convergence_criteria(1e-7, 1e-8, 3.0, 2.0, 1e-6, absolute_tolerance=1e-6)
        (False, '')
    """
    refuse_non_finite_change(
        {"l2distu_rel": l2distu_rel, "l2distm_rel": l2distm_rel, "l2distu_abs": l2distu_abs, "l2distm_abs": l2distm_abs}
    )
    max_rel_err = max(l2distu_rel, l2distm_rel)
    max_abs_err = max(l2distu_abs, l2distm_abs)
    absolute_met = absolute_tolerance is None or max_abs_err < absolute_tolerance
    if max_rel_err < tol_picard and absolute_met:
        reason = f"Converged: Rel err {max_rel_err:.1e} < tol {tol_picard:.1e}"
        if absolute_tolerance is not None:
            reason += f", Abs err {max_abs_err:.1e} < absolute_tolerance {absolute_tolerance:.1e}"
        return True, reason
    return False, ""


def read_inner_solve_failures(hjb_solver: object) -> object:
    """What an HJB solver says about its latest solve's inner solves, or ``None`` if it cannot say (#1878).

    The coupling layer duck-types its solvers: a plain object with a ``solve_hjb_system`` is a supported
    HJB solver, and it has no ``inner_solve_failures``. Such a solver does not track its inner solves, which
    is what ``None`` already means.
    """
    method = getattr(hjb_solver, "inner_solve_failures", None)
    return method() if callable(method) else None


def refuse_a_shared_hjb_solver(hjb_solvers: Iterable[object], coupler: str) -> None:
    """A coupled solve takes one HJB solver object per slot, and refuses one object in several slots by name (#2644).

    A solver's record is its latest solve, so an object filling two nodes or regimes reports only its last slot's, and
    a coupler reading it certified over the first slot's failed steps (#2644 review 1, M1). Nothing in the library
    shares a solver, so sharing is refused rather than read solve by solve. The check is by identity, of the objects the
    coupler is handed: wrappers that hide one solver behind distinct objects are not seen.
    """
    slots = list(hjb_solvers)
    filled = Counter(id(solver) for solver in slots)
    for solver in slots:
        if filled[id(solver)] > 1:
            raise ValueError(
                f"{coupler}: one HJB solver object, a {type(solver).__name__}, fills {filled[id(solver)]} of its slots. "
                "Pass one HJB solver per slot (#2644): a solver's record is its latest solve, so the earlier slots of a "
                "shared one could not be read."
            )


def read_inner_solve_failures_of(hjb_solvers: Iterable[object]) -> tuple | None:
    """What a coupled solve's HJB solvers say about their latest solves, together (#2644): ``None`` when none of them
    tracks its inner solves, else every recorded step of every one that does. Each slot holds its own solver
    (`refuse_a_shared_hjb_solver`), so each one's latest solve is its slot's in the last sweep, where the sweep reached
    it. A sweep that a diverged HJB ends early (#1718) leaves the later slots unsolved, and their solvers report their
    previous solve, which can predate this coupled solve; that result is not converged."""
    records = [read_inner_solve_failures(solver) for solver in hjb_solvers]
    tracked = [record for record in records if isinstance(record, tuple)]
    return None if not tracked else tuple(failure for record in tracked for failure in record)


@contextmanager
def inner_failures_recorded(hjb_solvers: Iterable[Any]) -> Iterator[None]:
    """For a coupled solve: each HJB solver whose ``on_inner_failure`` is "raise" records instead, while it runs.

    A sweep's inner failure is then the coupler's to read at its verdict (#1878), not an exception partway through
    the solve; the policy is restored after (#2644). A duck-typed solver without the attribute is left alone.
    """
    switched: list = []
    # Every write is inside the try, and only what was written is restored (#2644 review 2, F7).
    try:
        for solver in hjb_solvers:
            if getattr(solver, "on_inner_failure", None) == "raise":
                solver.on_inner_failure = "record"
                switched.append(solver)
        yield
    finally:
        for solver in switched:
            solver.on_inner_failure = "raise"


@contextmanager
def negative_density_recorded(fp_solvers: Iterable[Any]) -> Iterator[None]:
    """For a coupled solve: each FP solver that keeps a negative value records it instead of warning at each sweep, and
    the coupler reports the density it returns (#2845). Its record starts empty, so a solve that raises before any FP
    step has nothing of an earlier solve's to report. A duck-typed solver without the attribute is left alone."""
    switched: list = []
    try:
        for solver in fp_solvers:
            if getattr(solver, "reports_negative_density", False):
                solver.last_negative_density = None
            if getattr(solver, "on_negative_density", None) == "warn":
                solver.on_negative_density = "record"
                switched.append(solver)
        yield
    finally:
        for solver in switched:
            solver.on_negative_density = "warn"


def _report_negative_density(result: Any, fp_solvers: list[Any]) -> None:
    """One warning per solver, and ``fp_negative_density`` in the result's record, for the density the result returns
    rather than the last sweep's step, since a coupler can damp it (#2845). Newton returns ``(U, M, info)``."""
    if isinstance(result, tuple):
        densities, record = result[1], result[2]
    else:
        densities = getattr(result, "densities", None)
        densities = result.M if densities is None else densities
        record = getattr(result, "metadata", None)
    per_solver = [densities] if len(fp_solvers) == 1 else list(densities)
    reports: list[dict[str, Any] | None] = []
    for solver, density in zip(fp_solvers, per_solver, strict=True):
        if not getattr(solver, "reports_negative_density", False):
            reports.append(None)
            continue
        report = solver.negative_density_of(density)
        solver.warn_negative_density(report)
        reports.append(report)
    if any(report is not None for report in reports) and isinstance(record, dict):
        record["fp_negative_density"] = reports[0] if len(reports) == 1 else reports


#: The ``stacklevel`` of a warning a decorated ``solve`` raises for its caller: `verdict_reads_inner_failures`' wrapper
#: is one frame between the two (#2644).
SOLVE_CALLER_STACKLEVEL = 3


def verdict_reads_inner_failures(
    hjb_solvers_of: Callable[[Any], list[object]],
    *,
    fp_solvers_of: Callable[[Any], list[Any]],
    check: bool = True,
) -> Callable[
    [Callable[Concatenate[_Solver, _Params], _Result]],
    Callable[Concatenate[_Solver, _Params], _Result],
]:
    """Decorate a coupled solver's ``solve`` so that its HJB solvers record inner failures while it runs, and a result
    reported converged over a failed inner step of the last sweep is not (#1878, #2644).

    The one place every coupler's verdict reads the record. ``FixedPointIterator`` read it alone before, so a
    ``BlockIterator`` over the same failed steps reported ``converged=True``. One HJB solver object in several slots is
    refused before the solve (`refuse_a_shared_hjb_solver`), so each solver's latest record is its slot's in the last
    sweep. Newton takes ``check=False``: its last HJB solve can be a Jacobian probe, so it reads the record in its own
    verdict, where the latest solve is the accepted iterate's. ``fp_solvers_of`` gives one FP solver per density the
    result returns, for its report of a kept negative (#2845).
    """

    def decorate(
        solve: Callable[Concatenate[_Solver, _Params], _Result],
    ) -> Callable[Concatenate[_Solver, _Params], _Result]:
        @functools.wraps(solve)
        def wrapper(self: _Solver, /, *args: _Params.args, **kwargs: _Params.kwargs) -> _Result:
            solvers = hjb_solvers_of(self)
            refuse_a_shared_hjb_solver(solvers, type(self).__name__)
            fp_solvers = fp_solvers_of(self)
            with inner_failures_recorded(solvers), negative_density_recorded(fp_solvers):
                returned = False
                try:
                    result = solve(self, *args, **kwargs)
                    returned = True
                finally:
                    if not returned:
                        # There is no result to measure, so this says what the last FP solve of this solve left, if one
                        # ran (#2845).
                        for solver in fp_solvers:
                            if getattr(solver, "reports_negative_density", False):
                                report = solver.last_negative_density
                                if report is not None:
                                    solver.warn_negative_density(report)
            _report_negative_density(result, fp_solvers)
            if not check:
                return result
            failures = read_inner_solve_failures_of(solvers)
            metadata = getattr(result, "metadata", None)
            if isinstance(metadata, dict):
                # Every coupler's result says what its solvers' latest solves recorded, converged or not (#2644).
                metadata["inner_hjb_failures"] = None if failures is None else list(failures)
            if failures:
                _refuse_a_converged_result(result, failures)
            return result

        return wrapper

    return decorate


def _refuse_a_converged_result(result: Any, failures: tuple) -> None:
    """A result reported converged over ``failures``, failed inner steps of its last sweep, is not (#1878)."""
    if not getattr(result, "converged", False):
        return
    metadata = getattr(result, "metadata", None)
    previous = metadata.get("convergence_reason", "") if isinstance(metadata, dict) else ""
    _, reason = refuse_convergence_over_failed_inner_solves(True, previous, failures)
    result.converged = False
    if isinstance(metadata, dict):
        metadata["convergence_reason"] = reason
        metadata["inner_hjb_failures"] = list(failures)


def refuse_convergence_over_failed_inner_solves(
    converged: bool,
    reason: str,
    inner_failures: object,
) -> tuple[bool, str]:
    """A fixed point is not a converged MFG solution if its HJB half returned non-roots (#1878).

    The Picard criteria test whether the map stopped moving. When the HJB solves of the sweep that
    stopped it did not converge, the map being iterated is not the discrete MFG map, and its fixed
    point is not a solution of the discrete system -- measured on the 1-D smoke fixture at 6c0610d2,
    ``converged=True`` at sweep 38 with three of ten backward steps at residuals 7.5e+01 to 2.8e+02
    against 1e-06. So ``converged`` stays False and the reason names the steps.

    ``inner_failures`` is what ``BaseHJBSolver.inner_solve_failures()`` returned for that sweep. Only a
    non-empty tuple refuses: ``None`` means the solver does not track its inner solves, and anything
    that is not a tuple is treated the same way rather than read as truthy.
    """
    if not converged or not isinstance(inner_failures, tuple) or not inner_failures:
        return converged, reason
    # Members measure different quantities (InnerSolveFailure.measure), so the worst step is the one furthest past its
    # own tolerance, not the largest number (#2644).
    worst = max(
        inner_failures, key=lambda failure: failure.residual / failure.tolerance if failure.tolerance > 0 else np.inf
    )
    steps = ", ".join(str(failure.t_idx) for failure in inner_failures)
    return False, (
        f"inner_hjb_not_converged: the Picard criteria were met ({reason}), but {len(inner_failures)} "
        f"HJB time step(s) of that sweep did not converge (t_idx {steps}; worst residual {worst.residual:.3e} "
        f"against {worst.tolerance:.3e}, {worst.measure}: {worst.reason}). The fixed point is not a solution of the "
        f"discrete MFG system (#1878)."
    )


def preserve_initial_condition(
    M: np.ndarray,
    M_initial: np.ndarray,
) -> np.ndarray:
    """
    Preserve initial condition for density after updates.

    After damping or Anderson acceleration, M[0, :] may be modified,
    but the initial condition must remain fixed.

    Args:
        M: Density array (modified in-place)
        M_initial: Initial density distribution m_0(x)

    Returns:
        Modified M with preserved initial condition

    Note:
        This is critical for maintaining physical correctness of the solution.
        The initial density distribution is a boundary condition that must not change.
    """
    M[0, :] = M_initial
    return M


def adapt_damping(
    theta_U: float,
    theta_M: float,
    error_history_U: list[float],
    error_history_M: list[float],
    *,
    theta_U_initial: float,
    theta_M_initial: float,
    decay: float = 0.5,
    min_damping: float = 0.05,
    increase_threshold: float = 1.2,
    recovery_rate: float = 1.05,
    stable_window: int = 3,
) -> tuple[float, float, str | None]:
    """
    Adapt Picard damping factors based on error history (Issue #583).

    Detects oscillation (error increasing) and reduces damping to stabilize.
    After sustained convergence, cautiously recovers toward initial damping.

    U and M are adapted independently since U gradient explosion is the
    primary pathology in strongly-coupled MFG systems.

    Args:
        theta_U: Current damping factor for U.
        theta_M: Current damping factor for M.
        error_history_U: Relative L2 error history for U (all iterations so far).
        error_history_M: Relative L2 error history for M (all iterations so far).
        theta_U_initial: Initial damping factor for U (recovery ceiling).
        theta_M_initial: Initial damping factor for M (recovery ceiling).
        decay: Multiplicative decay on oscillation (e.g., 0.5 halves damping).
        min_damping: Minimum damping bound (prevents stalling).
        increase_threshold: Error ratio above which oscillation is detected.
        recovery_rate: Multiplicative increase during stable convergence.
        stable_window: Consecutive decreasing iterations required for recovery.

    Returns:
        (theta_U, theta_M, warning_msg): Updated damping factors and optional
        warning message (None if no oscillation detected).
    """
    warning_msg = None

    # Need at least 2 data points to detect oscillation
    if len(error_history_U) < 2:
        return theta_U, theta_M, None

    # --- Adapt U ---
    ratio_U = error_history_U[-1] / error_history_U[-2] if error_history_U[-2] > 0 else 1.0
    if ratio_U > increase_threshold:
        theta_U = max(theta_U * decay, min_damping)
        warning_msg = f"Adaptive damping: U error increased by {ratio_U:.2f}x. Reduced theta_U to {theta_U:.4f}."
    elif len(error_history_U) >= stable_window and all(
        error_history_U[-(i + 1)] < error_history_U[-(i + 2)] for i in range(stable_window - 1)
    ):
        theta_U = min(theta_U * recovery_rate, theta_U_initial)

    # --- Adapt M ---
    ratio_M = error_history_M[-1] / error_history_M[-2] if error_history_M[-2] > 0 else 1.0
    if ratio_M > increase_threshold:
        theta_M = max(theta_M * decay, min_damping)
        msg_M = f"Adaptive damping: M error increased by {ratio_M:.2f}x. Reduced theta_M to {theta_M:.4f}."
        warning_msg = f"{warning_msg} {msg_M}" if warning_msg else msg_M
    elif len(error_history_M) >= stable_window and all(
        error_history_M[-(i + 1)] < error_history_M[-(i + 2)] for i in range(stable_window - 1)
    ):
        theta_M = min(theta_M * recovery_rate, theta_M_initial)

    return theta_U, theta_M, warning_msg


def compute_scheduled_damping(
    iteration: int,
    base_damping: float,
    schedule: str = "constant",
    min_damping: float = 0.01,
) -> float:
    """Compute damping factor for a given iteration using a named schedule.

    The base_damping acts as the initial/maximum value. Decay schedules
    modulate it: theta(k) = base_damping * schedule(k), clamped to min_damping.

    Args:
        iteration: Current iteration (0-indexed).
        base_damping: Base damping factor (theta at k=0 for decay schedules).
        schedule: "constant", "harmonic", "sqrt", or "exponential".
        min_damping: Floor for damping value.

    Returns:
        Damping factor for this iteration.

    Raises:
        ValueError: If schedule name is not recognized.
    """
    if schedule == "constant":
        return base_damping

    from mfgarchon.utils.iteration.schedules import harmonic_schedule, sqrt_schedule

    if schedule == "harmonic":
        value = base_damping * harmonic_schedule(iteration)
    elif schedule == "sqrt":
        value = base_damping * sqrt_schedule(iteration)
    elif schedule == "exponential":
        # Geometric decay: theta(k) = base_damping^(k+1)
        value = base_damping ** (iteration + 1)
    else:
        raise ValueError(
            f"Unknown damping schedule: {schedule!r}. Available: 'constant', 'harmonic', 'sqrt', 'exponential'."
        )

    return max(value, min_damping)


def preserve_terminal_condition(
    U: np.ndarray,
    U_terminal: np.ndarray,
) -> np.ndarray:
    """
    Preserve terminal condition for value function after updates.

    After damping or Anderson acceleration, U[-1, :] may be modified, but it must be the terminal value of
    the HJB solve this sweep made.

    Args:
        U: Value function array (modified in-place)
        U_terminal: The terminal value that HJB solve was given, ``problem.terminal_value`` at its m(T) (#1642)

    Returns:
        Modified U with preserved terminal condition

    Note:
        This is critical for maintaining physical correctness of the solution.
        The terminal condition is a boundary condition; damping must not change it. With a terminal coupling it
        moves with m(T) from sweep to sweep, and the row takes this sweep's value. Without this, damping dilutes the
        terminal condition, causing the
        value function gradient to vanish and agents to not move toward targets.
    """
    U[-1] = U_terminal
    return U


def value_function_is_finite(U: np.ndarray) -> bool:
    """Whether a value function is usable as an FP drift.

    Split out of :func:`diverged_value_function` for the one caller that needs the question
    without the answer: ``BlockIterator._gauss_seidel_step`` must SKIP its FP solve, while the
    loop above it publishes and stops. Two sites, two jobs, one predicate -- restating
    ``np.all(np.isfinite(...))`` at the second one is the shape #1718 exists to remove.
    """
    return bool(np.all(np.isfinite(U)))


def diverged_value_function(
    U_new: np.ndarray,
    U_terminal: np.ndarray,
    *,
    site: str,
    density: np.ndarray,
    iteration: int | None = None,
) -> np.ndarray | None:
    """The publishable form of a non-finite value function, or ``None`` when it is finite.

    One owner for the check #1717 wrote inline and #1718 found missing at six more coupling loops.
    Every one of them has the same shape -- HJB is solved, the result is consumed by an FP solve,
    and nothing looks in between -- and the same failure: the FP solver composes a drift from NaN
    and raises ``"Check CFL condition: dt * sigma^2 / dx^2 should be < 0.5"``. That is an FP
    diagnostic, carrying actively wrong advice, for an HJB failure.

    Returning the array rather than assigning it is deliberate: the seven sites publish to seven
    different places (``self.U``, ``Us_new[k]``, ``U[k]``, a local), and a helper that assigned
    would need to know which. The caller keeps that; the three decisions that must not vary live
    here.

    **Publish the diverged iterate, not the last finite one.** Keeping the last finite iterate
    leaves the result looking valid to any caller that reads output validation rather than the
    convergence reason -- trading a misattributed diagnostic for a silent one, which is worse.

    **Restore the terminal row.** ``U[-1]`` is boundary data the solve never had licence to
    overwrite, and it is restored here for the same reason the normal path restores it. A NaN
    terminal row would additionally make the published array useless as a diagnostic.

    **Copy before restoring.** ``preserve_terminal_condition`` writes ``U[-1]`` in place, and
    ``U_new`` is the HJB solver's own return value rather than a freshly damped array the caller
    owns. Mutating it would reach back into the solver's internals at five of the seven sites.

    Args:
        U_new: The value function the HJB solve just produced.
        U_terminal: The terminal value the HJB solve was given, u(T) at that solve's m(T) (#1642).
        site: Which coupling loop is reporting, named in the log line. The whole point of the
            check is attribution, so an unnamed one would be half a fix.
        density: The density this HJB solve was handed. The log line states its minimum, what was measured, and
            names no cause, so it holds on every solver (#2845).
        iteration: Iteration index, 0-based, if the site has one. Reported 1-based.

    Returns:
        ``None`` when ``U_new`` is finite everywhere. Otherwise a copy with the terminal row
        restored, which the caller should publish before stopping the iteration.
    """
    if value_function_is_finite(U_new):
        return None
    where = f" at iteration {iteration + 1}" if iteration is not None else ""
    logger.warning(
        "NaN/Inf in the value function%s, reported by %s: the density this solve was handed has a minimum of %.3e. "
        "Terminating early; FP was not solved with this iterate.",
        where,
        site,
        float(np.nanmin(np.asarray(density, dtype=float))),
    )
    published = np.asarray(U_new).copy()
    return preserve_terminal_condition(published, U_terminal)


def _face_neighbours(n: int, span: int | None) -> np.ndarray:
    """The node across the face (i + 1/2) from node i, along an axis of n nodes, as FP-FDM's divergence-upwind stencil
    takes it (#2904): i + 1, wrapped modulo the periodic span (`periodic_axis_span`), so that the face across the wrap
    reads node 0, never the node an endpoint-inclusive axis repeats. The last node of an axis that does not wrap has no
    face, and is mapped to itself."""
    nodes = np.arange(n)
    return np.minimum(nodes + 1, n - 1) if span is None else (nodes + 1) % span


def _density_at_faces(m: np.ndarray, across: np.ndarray) -> np.ndarray:
    """A 1-D density at the faces $x_{i+1/2}$, the mean of its two nodes, along the last axis: the points the 1-D
    velocity reads it at, for the own density and for each block of a multi-population stack alike (#2335). ``across``
    is each face's second node (`_face_neighbours`, #2904)."""
    return 0.5 * (m[..., : across.size] + m[..., across])


def _centred_gradient(u: np.ndarray, axis: int, dx: float, span: int | None) -> np.ndarray:
    """The centred difference of u along ``axis`` at every node: `np.gradient`'s on an axis that does not wrap, with
    second-order one-sided ends where the axis has three nodes or more and first-order ones on an axis of two, and
    across the wrap through `periodic_axis_span`'s span on one that does (#2904)."""
    if span is None:
        return np.gradient(u, dx, axis=axis, edge_order=2 if u.shape[axis] >= 3 else 1)
    nodes = np.arange(u.shape[axis])
    return (np.take(u, (nodes + 1) % span, axis=axis) - np.take(u, (nodes - 1) % span, axis=axis)) / (2.0 * dx)


def compute_fp_velocity_field(
    problem: MFGProblem,
    U: np.ndarray,
    M: np.ndarray,
    H_class: object,
    cross_density: np.ndarray | None = None,
    boundary_conditions: object | None = None,
) -> np.ndarray:
    """Compute the FP advection velocity $\\alpha^*$ from the value function, at the faces FP-FDM's divergence-upwind
    stencil reads it at (#919, #2904).

    Single-source for the velocity convention shared by the Picard
    ``FixedPointIterator`` and the Newton ``MFGResidual`` (Issue #1233). Evaluates
    ``H.optimal_control`` at cell interfaces $x_{i+1/2}$, with the normal component of the gradient the stencil's own
    forward difference $p_{i+1/2} = (U_{j} - U_i)/\\Delta x$, where $j$ is the node across the face as the stencil
    takes it (`_face_neighbours`): $i + 1$, wrapped through `periodic_axis_span` on an axis that wraps. On the quadratic
    control cost the velocity route then equals the value-function route the stencil forms from U (#2904).

    Args:
        problem: MFG problem (provides ``geometry``, ``dt`` and, when no BC is handed, ``get_boundary_conditions``).
        U: Value function, shape ``(Nt+1, *spatial_shape)``.
        M: Density, shape ``(Nt+1, *spatial_shape)`` (only the own-population density).
        H_class: Hamiltonian exposing ``optimal_control(t, x, p, m)``.
        cross_density: Optional stacked multi-population density trajectory
            ``(Nt+1, K*Nx)`` (Issue #1071, lock-faithful). The Hamiltonian is read through
            ``evaluate_optimal_control`` at the own density as ``m``. In 1-D this is handed too, as the state's
            ``cross_density``: at integer timestep ``n``, one row per population, at the faces as ``m`` is, each row
            averaged by ``_density_at_faces``, for the density terms that declare it (#2335). ``None`` =>
            single-population. In nD only a stack with one value per node (K = 1) is accepted, and it is read as
            ``m`` (#2330).
        boundary_conditions: The BC of the FP solver the velocity is for, which decides where an axis wraps
            (`periodic_axis_span`). ``None`` => the problem's shared BC as an FP solver reads it
            (`fp_view_of_shared_bc`), which refuses a Neumann value that is not zero and a Robin condition. On a
            `TensorProductGrid` either is read with the grid's periodic layout, which the grid's coordinates own:
            through the grid's one check (`bind_periodic_convention`), an unstated convention is completed with the
            grid's and a stated one that contradicts the grid is refused with the grid's message. On any other
            geometry the convention is read as handed, and an unstated one wraps every node as distinct (#2910). A
            BC periodic on some faces only is refused (`_refuse_partial_periodic`).

    Returns:
        Velocity $\\alpha^*$.
        1D: shape ``(U.shape[0], Nx - 1)``, at the faces $(i+1/2)$; on a `TensorProductGrid`, on an axis that
        wraps, the last face reads node 0.
        nD: shape ``(U.shape[0], ndim, *spatial_shape)``, one face array per axis: entry ``[d][i]`` is the velocity at
        the face $(i + 1/2)$ along ``d``, the face the stencil reads at node ``i``. The last node of an axis that does
        not wrap has no face and carries 0; a node a wrapping axis repeats carries its twin's face, which is the same
        face. ``optimal_control`` is called once per axis and time step, on the ``(N, d)`` batch of the faces that
        exist along that axis, each inside the domain on a `TensorProductGrid`: the normal component of p is the
        forward difference across the face, the others the mean of the two nodes' centred differences, and ``m``
        the mean of the two nodes (#2330, #2904).

    Raises:
        ValueError: when the geometry's `get_grid_spacing` refuses a grid that is not uniform; when U's shape has
            no tensor grid of the problem's geometry to sit on (#2756); when a stated periodic convention contradicts
            the grid's layout; and in nD, when ``optimal_control`` does not return
            shape ``(N, d)``.
        NotImplementedError: handed no BC, when the problem's shared BC has a Neumann value that is not zero or a
            Robin condition; when the BC is periodic on some faces only; and in nD, when ``cross_density[n]`` does not
            hold exactly one value per node.
    """
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import _refuse_partial_periodic
    from mfgarchon.geometry.boundary.bc_utils import fp_view_of_shared_bc
    from mfgarchon.geometry.boundary.conditions import periodic_axis_span
    from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid, bind_periodic_convention

    geometry = problem.geometry
    grid_spacing = geometry.get_grid_spacing()
    dt = problem.dt
    Nt = U.shape[0]
    spatial_shape = U.shape[1:]
    ndim = len(spatial_shape)
    bc = (
        fp_view_of_shared_bc(problem.get_boundary_conditions(), consumer="compute_fp_velocity_field")
        if boundary_conditions is None
        else boundary_conditions
    )
    # The grid's coordinates own the periodic layout (#2904, user ruling 2026-11-08): through the grid's one check, an
    # unstated convention takes the grid's and a stated one that contradicts the grid is refused with the grid's
    # message; a BC periodic on some faces only is refused, since FP-FDM's stencil wraps every axis or none.
    if isinstance(geometry, TensorProductGrid):
        bc = bind_periodic_convention(geometry, bc)
    _refuse_partial_periodic(bc, ndim, geometry, consumer="compute_fp_velocity_field")
    spans = [periodic_axis_span(bc, spatial_shape[d], ndim, geometry) for d in range(ndim)]

    # The grid's own node axes, from their one owner (#2756).
    axes = geometry_grid_axes(geometry)
    if axes is None or tuple(axis.size for axis in axes) != tuple(spatial_shape):
        raise ValueError(
            f"compute_fp_velocity_field: U has spatial shape {tuple(spatial_shape)}, and the problem's geometry "
            f"({type(geometry).__name__}) has no tensor grid of that shape to place it on (#2756)."
        )

    if ndim == 1:
        dx = grid_spacing[0]
        Nx = spatial_shape[0]
        across = _face_neighbours(Nx, spans[0])[: Nx - 1]

        # Face centers: x_{i+1/2}
        x_nodes = axes[0]
        x_faces = 0.5 * (x_nodes[:-1] + x_nodes[1:])  # (Nx-1,)

        # Face gradient: p_{i+1/2} = (U[j] - U[i]) / dx, j the node across the face (#2904)
        p_faces = (U[:, across] - U[:, : Nx - 1]) / dx  # (Nt, Nx-1)

        # Face density: m_{i+1/2} = (m[i] + m[j]) / 2, the own density's, in a multi-population solve too (#2335). The
        # stacked cross-density trajectory a multi-population solve passes (#1071) goes beside it as cross_density, read
        # at the faces too: each population's row averaged by the same function, so at K = 1 its one row is m.
        m_faces = _density_at_faces(M, across)  # (Nt, Nx-1)
        if cross_density is not None:
            stacked = np.asarray(cross_density)
            rows = np.stack([_population_rows(row, Nx, "compute_fp_velocity_field") for row in stacked])
            cross_faces = _density_at_faces(rows, across)  # (Nt, K, Nx-1)

        alpha_faces = np.zeros((Nt, Nx - 1))
        x_arr = x_faces.reshape(-1, 1)
        for n in range(Nt):
            m_n = m_faces[n] if n < m_faces.shape[0] else m_faces[-1]
            cross_n = None if cross_density is None else cross_faces[min(n, cross_faces.shape[0] - 1)]
            p_n = p_faces[n].reshape(-1, 1)
            state = HEvalState(t=n * dt, x=x_arr, p=p_n, m=m_n, cross_density=cross_n)
            alpha_faces[n] = np.ravel(H_class.evaluate_optimal_control(state))

        return alpha_faces
    else:
        # Issue #2330: `HamiltonianBase` documents a point ``(d,)`` or a batch ``(N, d)``, with ``m`` one value per
        # point, so each axis's faces go as one flattened batch, and a return of any other shape is refused.
        x_nodes = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)  # (*shape, ndim)
        n_points = int(np.prod(spatial_shape))

        alpha_field = np.zeros((Nt, ndim, *spatial_shape))
        for n in range(Nt):
            u = U[n]
            # Issue #1071: stacked cross-density at this integer timestep for multi-pop (see 1D path).
            if cross_density is not None:
                stacked = np.asarray(cross_density[n])
                if stacked.size != n_points:
                    # A stack of another size is more than one population, which this door does not pair with an nD
                    # batch, nor hand on as cross_density (#2335). This also refuses a Hamiltonian that would ignore m,
                    # the price of not guessing (#2330 review).
                    raise NotImplementedError(
                        f"compute_fp_velocity_field: a cross density of {stacked.size} values for {n_points} nodes "
                        f"has no per-node form for an nD optimal_control batch; only one value per node is "
                        f"accepted (#2330)."
                    )
                m_nodes = stacked.reshape(spatial_shape)
            else:
                m_nodes = M[n] if n < M.shape[0] else M[-1]
            centred = [_centred_gradient(u, e, grid_spacing[e], spans[e]) for e in range(ndim)]
            for d in range(ndim):
                # The faces that exist along d: every node but the last on an axis that does not wrap, and the span's
                # distinct nodes on one that does, so on a TensorProductGrid each face sits inside the domain (#2904).
                faces = spatial_shape[d] - 1 if spans[d] is None else spans[d]
                own = np.arange(faces)
                across = _face_neighbours(spatial_shape[d], spans[d])[:faces]
                # The normal component is the stencil's forward difference across the face; the others are the mean of
                # the two nodes' centred differences (#2904).
                p_face = np.stack(
                    [
                        (np.take(u, across, axis=d) - np.take(u, own, axis=d)) / grid_spacing[d]
                        if e == d
                        else 0.5 * (np.take(centred[e], own, axis=d) + np.take(centred[e], across, axis=d))
                        for e in range(ndim)
                    ],
                    axis=-1,
                )
                x_face = np.take(x_nodes, own, axis=d)
                x_face[..., d] += 0.5 * grid_spacing[d]
                m_face = 0.5 * (np.take(m_nodes, own, axis=d) + np.take(m_nodes, across, axis=d))
                face_shape = x_face.shape[:-1]
                n_faces = int(np.prod(face_shape))
                alpha_n = np.asarray(
                    H_class.evaluate_optimal_control(
                        HEvalState(
                            t=n * dt,
                            x=x_face.reshape(n_faces, ndim),
                            p=p_face.reshape(n_faces, ndim),
                            m=m_face.reshape(n_faces),
                        )
                    )
                )
                if alpha_n.shape != (n_faces, ndim):
                    raise ValueError(
                        f"{type(H_class).__name__}.optimal_control returned shape {alpha_n.shape} for a batch of "
                        f"{n_faces} points in {ndim}-D; it must return one control vector per point, shape "
                        f"({n_faces}, {ndim}) (#2330)."
                    )
                # The last node of an axis that does not wrap has no face along it and keeps 0; a node a wrapping
                # axis repeats takes its twin's face, the same face.
                component = np.zeros(spatial_shape)
                placed: list[slice | int] = [slice(None)] * ndim
                placed[d] = slice(0, faces)
                component[tuple(placed)] = alpha_n[:, d].reshape(face_shape)
                if spans[d] is not None and faces < spatial_shape[d]:
                    repeated: list[slice | int] = [slice(None)] * ndim
                    twin: list[slice | int] = [slice(None)] * ndim
                    repeated[d] = slice(faces, spatial_shape[d])
                    twin[d] = slice(0, spatial_shape[d] - faces)
                    component[tuple(repeated)] = component[tuple(twin)]
                alpha_field[n, d] = component

        return alpha_field


def _why_an_own_control_is_not_a_velocity_here(
    problem: Any, fp_solver: Any, M: np.ndarray, takes_a_velocity: bool
) -> list[str]:
    """Why ``fp_solver``, which forms its drift from U (``_forms_its_drift_from_u``), cannot take an own optimal control
    as a velocity here, or ``[]`` where it can (#2900, #2908, #2915). The set where it can is closed and stated
    positively: a tensor grid, a density trajectory of shape ``(Nt + 1, *grid shape)``, and a solver that reads the
    velocity there, by its own answer (``why_it_reads_no_velocity``). Three more conditions are enforced before this
    runs. FP-FDM accepts walls of four kinds only, NO_FLUX, PERIODIC, DIRICHLET and FLUX, a problem's zero NEUMANN
    reaching it as NO_FLUX (#1456, #2512 row B3), and on each of them the censuses in
    ``test_fp_velocity_at_the_stencils_faces_2904.py`` and ``test_fp_drift_from_optimal_control_2900.py`` measure the
    velocity route equal to the value-function route. FP-FDM refuses a boundary condition periodic on some faces
    only when it is built (#2495). ``MultiPopulationIterator`` refuses a grid that is not 1-D, so several
    populations reach here in 1-D only (#2330, #2335)."""
    from mfgarchon.geometry.grids.tensor_grid import geometry_grid_axes

    name = type(fp_solver).__name__
    if not takes_a_velocity:
        return [f"{name} takes the value function only"]
    reasons = []
    geometry = problem.geometry
    dimension = problem.dimension
    shape = tuple(int(n) for n in geometry.get_grid_shape())
    if geometry_grid_axes(geometry) is None:
        reasons.append(f"the problem is not a tensor grid ({type(geometry).__name__}, {dimension}-D)")
    unread = fp_solver.why_it_reads_no_velocity()
    if unread is not None:
        reasons.append(f"it reads no velocity: {unread}")
    trajectory = (problem.Nt + 1, *shape)
    if np.shape(M) != trajectory:
        reasons.append(f"it is handed M of shape {np.shape(M)}, not a density trajectory of shape {trajectory}")
    return reasons


def _what_the_router_read(problem: Any, fp_solver: Any, M: np.ndarray, cross_density: np.ndarray | None) -> str:
    """What the closed set read, for its refusal to state (#2900): the grid's dimension, whether one population is
    solved, and for a solver that holds its walls, those, then the shape of M. Whether the solver reads a velocity
    is its own answer, which the reasons carry (#2915)."""
    from mfgarchon.geometry.boundary.types import BoundaryFace

    read = [
        f"a {problem.dimension}-D grid",
        "one population" if cross_density is None else "a MultiPopulationIterator solve",
    ]
    bc = getattr(fp_solver, "boundary_conditions", None)
    if bc is not None:
        faces = [BoundaryFace(axis, side) for axis in range(problem.dimension) for side in ("min", "max")]
        read.append(f"walls {[bc.segment_for_face(face, problem.geometry).bc_type.name for face in faces]}")
    read.append(f"M of shape {np.shape(M)}")
    return ", ".join(read)


def resolve_fp_drift_kwargs(
    problem: object,
    fp_sig_params: set[str] | None,
    drift_field_override: object | None,
    U: np.ndarray,
    M: np.ndarray,
    *,
    h_class: object | None = None,
    cross_density: np.ndarray | None = None,
    drift_convention: DriftConvention | None = None,
    fp_solver: object | None = None,
) -> tuple[dict, bool]:
    """Resolve how the value function enters ``solve_fp_system`` (drift vs potential).

    Single-source for the FP drift/potential convention shared by the Picard
    ``FixedPointIterator`` and the Newton ``MFGResidual`` (Issue #1233). Before this
    helper existed the two code paths diverged: ``MFGResidual`` still passed the value
    function as ``drift_field`` (which the v0.18.6 rename redefined as the *velocity*
    $\\alpha^*$, not the potential), so the Newton residual was inconsistent with the
    Picard fixed point and the two solvers converged to different roots.

    Resolution rules (matching the FP-solver semantics post-v0.18.6):
        - Explicit ``drift_field`` override → pass it through verbatim (velocity).
        - Smooth separable $H$ → pass $U$ as ``potential_field`` (the FP solver derives
          the velocity $\\alpha = -c\\,\\nabla U$ internally; ``potential_field`` is the live
          U-potential channel -- it was marked deprecated until #1771, in the wrong direction,
          while this very line emitted it on the default path).
        - Smooth separable $H$ for which ``drift_is_the_control_costs`` does not hold (#2900), handed
          to a solver whose class sets ``_forms_its_drift_from_u`` (FP-FDM, FP-SL) → the velocity
          path below, which reads its ``optimal_control``, only where
          ``_why_an_own_control_is_not_a_velocity_here`` finds no reason against it: a tensor grid whose
          every face is ``NO_FLUX``, ``PERIODIC``, ``DIRICHLET`` or ``FLUX`` (#2908, #2915), a density
          trajectory, a solver that reads the velocity there by its own answer, and one population above
          1-D (in 1-D any number, #2335). Anywhere else this refuses, stating what it read and why, and names no other
          scheme. Every other solver (FVM,
          particle, the weak-form family) keeps ``potential_field`` and reads ``optimal_control``
          itself. With ``drift_convention=None`` the pre-#1489 gate below decides, and a solver that
          forms $-c\\,\\nabla U$ refuses through ``fp_drift_coefficient``.
        - Non-smooth $H$ on a ``VELOCITY``-channel solver → pass the face-centered velocity
          $\\alpha^*$ as ``drift_field`` (see :func:`compute_fp_velocity_field`).
        - Non-smooth $H$ on a ``VALUE_FUNCTION`` solver (only ``potential_field=U``) → raise:
          the solver forms $\\alpha = -c\\,\\nabla U$ internally, but a non-smooth $H$'s FP
          drift is the Clarke optimal control $\\alpha^*$, not $-c\\,\\nabla U$, so $U$ cannot
          represent it (Issue #1489 S1).

    Issue #1489 (S1): the drift convention cannot be inferred from parameter presence.
    ``drift_field`` is a real velocity channel on some solvers (fp_fvm / FPFDM)
    but a DEPRECATED ALIAS for ``potential_field=U`` on the weak-form family
    (``DriftConvention.VALUE_FUNCTION``). The routing therefore keys on the solver-declared
    ``drift_convention`` when the caller supplies it, and falls back byte-for-byte to the
    pre-#1489 param-presence gate when it is ``None`` (un-updated callers are unaffected).

    Args:
        problem: MFG problem (provides ``hamiltonian_class``, ``geometry``, ``dt``).
        fp_sig_params: Parameter names of ``fp_solver.solve_fp_system`` (or ``None``).
        drift_field_override: Caller-supplied drift override (array/callable), or ``None``.
        U: Current value function, shape ``(Nt+1, *spatial_shape)``.
        M: Current density, shape ``(Nt+1, *spatial_shape)`` (used only for non-smooth $H$).
        h_class: Hamiltonian to use for the smoothness dispatch and velocity computation,
            overriding ``problem.hamiltonian_class``. The multi-population iterator passes the
            population's own (unbound) Hamiltonian here (Issue #1043) so K populations resolve drift
            the same way single-pop does; ``None`` uses ``problem.hamiltonian_class`` (single-pop).
        cross_density: Optional stacked multi-population density trajectory ``(Nt+1, K*Nx)``
            (Issue #1071, lock-faithful). Forwarded to :func:`compute_fp_velocity_field` so the
            non-smooth velocity path sees the other populations' density — replacing the retired
            ``BoundHamiltonian`` wrapper. ``None`` => single-population own density.
        drift_convention: The FP solver's declared ``_drift_convention`` (Issue #1489 S1). When
            given, it disambiguates the ``drift_field`` channel: ``DriftConvention.VELOCITY`` means
            ``drift_field`` is a true velocity $\\alpha^*$; ``DriftConvention.VALUE_FUNCTION`` means
            the solver only consumes $U$ (its ``drift_field`` is a deprecated alias for
            ``potential_field``), so a non-smooth $H$ raises instead of silently advecting $U$.
            ``None`` (caller did not thread it) => the pre-#1489 param-presence gate, byte-identical.
        fp_solver: The FP solver the kwargs are for. Its class's ``_forms_its_drift_from_u`` and, for such a
            solver, its walls and advection scheme decide where an own ``optimal_control`` goes (#2900). ``None``
            => no solver is known, and such a Hamiltonian keeps ``potential_field``.

    Returns:
        ``(drift_kwargs, use_positional_U)`` where ``drift_kwargs`` is one of ``{}``,
        ``{"drift_field": ...}`` or ``{"potential_field": ...}`` to merge into the
        solver kwargs, and ``use_positional_U`` is ``True`` iff the FP solver exposes
        neither ``drift_field`` nor ``potential_field`` (legacy positional-U interface),
        in which case the caller passes ``U`` positionally.
    """
    if fp_sig_params is None:
        return {}, True

    params = fp_sig_params
    drift_kwargs: dict = {}

    if drift_field_override is not None:
        if "drift_field" in params:
            drift_kwargs["drift_field"] = drift_field_override
    else:
        from mfgarchon.alg.numerical.fp_solvers.base_fp import DriftConvention
        from mfgarchon.core.hamiltonian import SeparableHamiltonian
        from mfgarchon.utils.pde_coefficients import drift_is_the_control_costs, refuse_the_scalar_drift

        H_class = h_class if h_class is not None else problem.hamiltonian_class
        # Issue #1071: the multi-population iterator now passes the population's own (unbound)
        # Hamiltonian plus the cross_density trajectory, so the smoothness dispatch reads H_class
        # directly — a smooth-separable H takes potential_field=U (cross-coupling enters via the
        # HJB, not the FP drift). The former BoundHamiltonian-unwrap (getattr `_inner`) is gone.
        #
        # A smooth separable H is the only case whose FP drift the potential_field=U channel can
        # represent: the solver recovers alpha = -c*grad(U) internally. A non-smooth (or
        # non-separable) H has a set-valued / Clarke optimal control alpha* that is NOT -c*grad(U),
        # so it can only enter a VELOCITY-channel solver as a precomputed alpha*.
        h_is_smooth_separable = isinstance(H_class, SeparableHamiltonian) and H_class.control_cost.is_smooth()
        # #2900: -c*grad(U) is a smooth separable H's drift only where drift_is_the_control_costs holds. Only a solver
        # whose class says it forms its drift from U is affected: FVM, particle and the weak-form family read
        # optimal_control on the potential_field route. Such a solver takes the velocity path inside the closed set
        # `_why_an_own_control_is_not_a_velocity_here` checks, and is refused outside it. With drift_convention=None
        # the pre-#1489 gate below decides, and fp_drift_coefficient refuses where the scalar is formed.
        drift_from_u = h_is_smooth_separable
        if (
            h_is_smooth_separable
            and drift_convention is not None
            and getattr(type(fp_solver), "_forms_its_drift_from_u", False)
            and not drift_is_the_control_costs(H_class)
        ):
            takes_a_velocity = drift_convention == DriftConvention.VELOCITY and "drift_field" in params
            reasons = _why_an_own_control_is_not_a_velocity_here(problem, fp_solver, M, takes_a_velocity)
            if reasons:
                name = type(fp_solver).__name__
                refuse_the_scalar_drift(
                    H_class,
                    f"{name}.solve_fp_system(potential_field=U)",
                    _what_the_router_read(problem, fp_solver, M, cross_density),
                    (
                        f"{name} takes its optimal_control as a velocity only on a tensor grid whose every face is "
                        "NO_FLUX, PERIODIC, DIRICHLET or FLUX, handed the density trajectory, one population above "
                        "1-D, where it reads the velocity (#2900, #2908, #2915). "
                        f"Here {'; '.join(reasons)}."
                        if takes_a_velocity
                        else f"{'; '.join(reasons)}."
                    ),
                )
            drift_from_u = False

        # Issue #1489 (S1): param presence cannot disambiguate the drift convention — `drift_field`
        # is a real velocity channel on some solvers but a deprecated alias for `potential_field=U`
        # on the weak-form family (VALUE_FUNCTION). Key on the solver-declared `drift_convention`
        # when the caller threads it; when it is None, reproduce the pre-#1489 param-presence gate
        # byte-for-byte so un-updated callers are unaffected.
        if drift_convention is None:
            use_velocity = H_class is not None and "drift_field" in params and not h_is_smooth_separable
        else:
            use_velocity = (
                drift_convention == DriftConvention.VELOCITY
                and H_class is not None
                and "drift_field" in params
                and not drift_from_u
            )

        if use_velocity:
            drift_kwargs["drift_field"] = compute_fp_velocity_field(
                problem,
                U,
                M,
                H_class,
                cross_density=cross_density,
                boundary_conditions=getattr(fp_solver, "boundary_conditions", None),
            )
        elif "potential_field" in params:
            # Issue #1489 (S1): a VALUE_FUNCTION solver consumes `potential_field=U` and forms its
            # drift as alpha = -c*grad(U) internally. That is valid ONLY for a smooth separable H.
            # For a non-smooth / non-separable H the true FP drift is the Clarke optimal control
            # alpha* (set-valued at kinks), NOT -c*grad(U), which U cannot represent — passing it
            # would silently solve the wrong problem, so fail loud. The determination is only safe
            # when the caller declared a convention; with drift_convention=None we keep the legacy
            # potential_field=U behavior (byte-identical pre-#1489 fallback).
            if drift_convention is not None and not h_is_smooth_separable:
                raise ValueError(
                    "Cannot route the value function as FP drift: this FP solver takes "
                    "`potential_field=U` (DriftConvention.VALUE_FUNCTION) and forms its drift as "
                    "alpha = -c*grad(U) internally, but the problem's Hamiltonian is non-smooth / "
                    "non-separable, whose optimal control alpha* is the (set-valued) Clarke control, "
                    "not -c*grad(U). U cannot represent this drift, so passing it would silently solve "
                    "the wrong problem. Use a VELOCITY-channel FP solver that accepts a precomputed "
                    "alpha* (e.g. FPFVMSolver / FPFDMSolver) with an explicit "
                    "`drift_field=alpha*`, or use a smooth separable Hamiltonian. Issue #1489 (S1); "
                    "see also #1420."
                )
            drift_kwargs["potential_field"] = U
        elif "drift_field" in params:
            # Issue #1420 (G-017 V2): the FP solver exposes only `drift_field` (the velocity α*,
            # DriftConvention.VELOCITY) and no `potential_field`. We hold the value function U, not a
            # velocity, and for a meshfree/collocation solver the coupling layer
            # cannot derive α* at the solver's own points (compute_fp_velocity_field is grid-based).
            # Passing U as `drift_field` would silently advect the value function as a velocity — the
            # exact #1043/V2 bug. Fail loud instead of solving the wrong problem.
            raise ValueError(
                "Cannot auto-route the value function as FP drift: the FP solver exposes only "
                "`drift_field` (velocity α*) and no `potential_field`, so U cannot be passed as a "
                "velocity (this would silently advect the value function as a velocity). Pass an "
                "explicit `drift_field=α*` (the precomputed optimal control, e.g. -∇U/control_cost via "
                "`compute_fp_velocity_field` at the solver's own points), or use an FP solver with "
                "`potential_field` support (e.g. FPFDMSolver). Issue #1420 (G-017 V2)."
            )

    use_positional_U = not ("drift_field" in params or "potential_field" in params)
    return drift_kwargs, use_positional_U

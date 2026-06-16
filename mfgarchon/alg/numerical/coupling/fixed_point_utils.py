"""
Fixed Point Iterator Utilities

Shared helper functions for fixed-point iteration solvers to eliminate code duplication.
These utilities are used by both legacy and config-aware fixed-point iterator implementations.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mfgarchon.utils.solver_result import SolverResult


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
        U_terminal: Terminal condition g(x) for value function
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
) -> tuple[bool, str]:
    """
    Check if fixed-point iteration has converged.

    Convergence criteria (both must be satisfied):
    1. Relative errors: max(l2distu_rel, l2distm_rel) < tol_picard
    2. Absolute errors: max(l2distu_abs, l2distm_abs) < tol_picard

    Args:
        l2distu_rel: Relative L2 error for U
        l2distm_rel: Relative L2 error for M
        l2distu_abs: Absolute L2 error for U
        l2distm_abs: Absolute L2 error for M
        tol_picard: Convergence tolerance

    Returns:
        Tuple of (converged: bool, reason: str)

    Example:
        >>> converged, reason = check_convergence_criteria(1e-7, 1e-8, 1e-6, 1e-7, 1e-6)
        >>> print(converged)  # True
        >>> print(reason)  # "Converged: Rel err 1.0e-07, Abs err 1.0e-06 < tol 1.0e-06"
    """
    max_rel_err = max(l2distu_rel, l2distm_rel)
    max_abs_err = max(l2distu_abs, l2distm_abs)

    # Both relative and absolute errors must be below tolerance
    if max_rel_err < tol_picard and max_abs_err < tol_picard:
        reason = f"Converged: Rel err {max_rel_err:.1e}, Abs err {max_abs_err:.1e} < tol {tol_picard:.1e}"
        return True, reason
    else:
        return False, ""


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

    After damping or Anderson acceleration, U[-1, :] may be modified,
    but the terminal condition must remain fixed.

    Args:
        U: Value function array (modified in-place)
        U_terminal: Terminal condition g(x) at t=T

    Returns:
        Modified U with preserved terminal condition

    Note:
        This is critical for maintaining physical correctness of the solution.
        The terminal condition is a boundary condition that must not change.
        Without this, damping dilutes the terminal condition, causing the
        value function gradient to vanish and agents to not move toward targets.
    """
    U[-1] = U_terminal
    return U


def compute_fp_velocity_field(
    problem: object,
    U: np.ndarray,
    M: np.ndarray,
    H_class: object,
    cross_density: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the face-centered FP advection velocity $\\alpha^*$ from the value function.

    Single-source for the velocity convention shared by the Picard
    ``FixedPointIterator`` and the Newton ``MFGResidual`` (Issue #1233). Evaluates
    ``H.optimal_control`` at cell interfaces $x_{i+1/2}$ using the forward-difference
    gradient $p_{i+1/2} = (U_{i+1} - U_i)/\\Delta x$ (Issue #919), which matches the
    FDM divergence-upwind stencil exactly.

    Args:
        problem: MFG problem (provides ``geometry`` and ``dt``).
        U: Value function, shape ``(Nt+1, *spatial_shape)``.
        M: Density, shape ``(Nt+1, *spatial_shape)`` (only the own-population density).
        H_class: Hamiltonian exposing ``optimal_control(x, m, p, t)``.
        cross_density: Optional stacked multi-population density trajectory
            ``(Nt+1, K*Nx)`` (Issue #1071, lock-faithful). When given, ``optimal_control``
            receives ``cross_density[n]`` (the stacked density at integer timestep ``n``,
            sliced per-population via ``population_index``) instead of the own-population
            density — replacing the ``BoundHamiltonian`` wrapper's ``m_all[round(t/dt)]``
            (``n*dt/dt == n``, so byte-identical). ``None`` => single-population own density.

    Returns:
        Face-centered velocity $\\alpha^*$.
        1D: shape ``(Nt, Nx-1)`` — velocity at each face $(i+1/2)$.
        nD: shape ``(Nt, ndim, *spatial_shape)`` — node-centered (nD fallback).
    """
    geometry = problem.geometry
    grid_spacing = geometry.get_grid_spacing()
    dt = problem.dt
    Nt = U.shape[0]
    spatial_shape = U.shape[1:]
    ndim = len(spatial_shape)

    bounds = geometry.get_bounds()

    if ndim == 1:
        dx = grid_spacing[0]
        Nx = spatial_shape[0]

        # Face centers: x_{i+1/2}
        x_nodes = np.linspace(bounds[0][0], bounds[1][0], Nx)
        x_faces = 0.5 * (x_nodes[:-1] + x_nodes[1:])  # (Nx-1,)

        # Face gradient: p_{i+1/2} = (U[i+1] - U[i]) / dx
        p_faces = np.diff(U, axis=-1) / dx  # (Nt, Nx-1)

        # Face density: m_{i+1/2} = (m[i] + m[i+1]) / 2
        m_faces = 0.5 * (M[:, :-1] + M[:, 1:])  # (Nt, Nx-1)

        alpha_faces = np.zeros((Nt, Nx - 1))
        x_arr = x_faces.reshape(-1, 1)
        for n in range(Nt):
            # Issue #1071: a multi-population solve passes the stacked cross-density trajectory;
            # optimal_control then sees the other populations' density (sliced via population_index)
            # at this integer timestep, replacing the BoundHamiltonian wrapper's m_all[round(t/dt)]
            # (n*dt/dt == n, byte-identical). Single-population: the own face-centered density.
            if cross_density is not None:
                m_n = cross_density[n]
            else:
                m_n = m_faces[n] if n < m_faces.shape[0] else m_faces[-1]
            p_n = p_faces[n].reshape(-1, 1)
            alpha_faces[n] = H_class.optimal_control(x_arr, m_n, p_n, t=n * dt).ravel()

        return alpha_faces
    else:
        # nD: node-centered fallback (face-centered nD deferred)
        grad_components = []
        for d in range(ndim):
            grad_d = np.gradient(U, grid_spacing[d], axis=d + 1)
            grad_components.append(grad_d)

        coords = [np.linspace(bounds[0][d], bounds[1][d], spatial_shape[d]) for d in range(ndim)]
        mesh = np.meshgrid(*coords, indexing="ij")
        x_grid = np.stack(mesh, axis=-1)

        alpha_field = np.zeros((Nt, ndim, *spatial_shape))
        for n in range(Nt):
            p_n = np.stack([grad_components[d][n] for d in range(ndim)], axis=-1)
            # Issue #1071: stacked cross-density at this integer timestep for multi-pop (see 1D path).
            if cross_density is not None:
                m_n = cross_density[n]
            else:
                m_n = M[n] if n < M.shape[0] else M[-1]
            alpha_n = H_class.optimal_control(x_grid, m_n, p_n, t=n * dt)
            if alpha_n.ndim == ndim + 1:
                alpha_field[n] = np.moveaxis(alpha_n, -1, 0)
            else:
                for d in range(ndim):
                    alpha_field[n, d] = alpha_n

        return alpha_field


def resolve_fp_drift_kwargs(
    problem: object,
    fp_sig_params: set[str] | None,
    drift_field_override: object | None,
    U: np.ndarray,
    M: np.ndarray,
    *,
    h_class: object | None = None,
    cross_density: np.ndarray | None = None,
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
          the velocity internally; ``potential_field`` is the deprecated-but-routing
          U-potential entry point).
        - Non-smooth $H$ → pass the face-centered velocity $\\alpha^*$ as ``drift_field``
          (see :func:`compute_fp_velocity_field`).

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
        from mfgarchon.core.hamiltonian import SeparableHamiltonian

        H_class = h_class if h_class is not None else problem.hamiltonian_class
        # Issue #1043: unwrap a cross-density-bound Hamiltonian (BoundHamiltonian, multi-pop) to
        # its inner H for the smoothness dispatch. The wrapper delegates optimal_control to the
        # inner H, so a bound smooth-separable H still has the momentum-only optimal control that
        # makes potential_field=U correct (the cross-coupling enters via the HJB, not the FP
        # drift). Without this, the wrapper fails `isinstance(SeparableHamiltonian)` and a K=1
        # multi-population solve takes the velocity path while single-pop takes potential → the
        # two converge to different fixed points (||F_FP|| ~ O(1)). For single-pop H_class has no
        # `_inner`, so this is a no-op (byte-identical).
        H_base = getattr(H_class, "_inner", H_class)
        use_velocity = (
            H_class is not None
            and "drift_field" in params
            and not (isinstance(H_base, SeparableHamiltonian) and H_base.control_cost.is_smooth())
        )
        if use_velocity:
            drift_kwargs["drift_field"] = compute_fp_velocity_field(problem, U, M, H_class, cross_density=cross_density)
        elif "potential_field" in params:
            drift_kwargs["potential_field"] = U
        elif "drift_field" in params:
            drift_kwargs["drift_field"] = U

    use_positional_U = not ("drift_field" in params or "potential_field" in params)
    return drift_kwargs, use_positional_U

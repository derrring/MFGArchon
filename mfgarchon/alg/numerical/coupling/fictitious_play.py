"""
Fictitious Play Iterator for Mean Field Games

Implements the Fictitious Play algorithm for solving MFG systems.
Unlike standard Picard iteration with fixed damping, Fictitious Play uses
a decaying learning rate that provably converges for potential MFGs.

Key advantages over fixed-point iteration:
- Proven convergence for potential MFGs (even when Picard fails)
- Noise suppression through Cesaro averaging
- Robust for long time horizons where fixed damping oscillates

Mathematical background:
    Standard Picard: m_{n+1} = (1-alpha) * m_n + alpha * L(m_n)  [fixed alpha]
    Fictitious Play: m_{n+1} = (1 - 1/(n+1)) * m_n + (1/(n+1)) * L(m_n)

    This is equivalent to the Cesaro mean: m_n = (1/n) * sum_{k=1}^{n} L(m_{k-1})

    The decaying step size suppresses stochastic noise from particle methods.

References:
    Cardaliaguet, P., & Hadikhanloo, S. (2017). Learning in mean field games:
        the fictitious play. ESAIM: Control, Optimisation and Calculus of
        Variations, 23(2), 569-591.

    Perrin, S., et al. (2020). Fictitious play for mean field games:
        Continuous time analysis and applications. NeurIPS 2020.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.alg.numerical.coupling.fixed_point_utils import verdict_reads_inner_failures
from mfgarchon.config.damping import DampingPolicy
from mfgarchon.utils.solver_result import SolverResult

from .base_mfg import (
    BaseCouplingIterator,
    assert_bc_providers_resolvable,
    assert_paired_solver_geometry,
    assert_paired_solver_sigma,
    refuse_a_capped_exit_in_a_coupled_solve,
    refuse_retired_solve_keywords,
    resolve_backend,
)
from .damping_run import DampingRun
from .fixed_point_utils import (
    check_convergence_criteria,
    diverged_value_function,
    resolve_fp_drift_kwargs,
)
from .iteration_state import IterationState

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfgarchon.backends.base_backend import BaseBackend
    from mfgarchon.config import MFGSolverConfig
    from mfgarchon.core.mfg_problem import MFGProblem


class FictitiousPlayIterator(BaseCouplingIterator):
    """
    Fictitious Play iterator for MFG systems.

    Implements the Fictitious Play algorithm which is proven to converge for
    potential Mean Field Games, even when standard Picard iteration fails.

    Key differences from FixedPointIterator:
    1. Decaying learning rate: alpha(k) = 1/(k+1) instead of fixed damping
    2. Only average M (density): HJB solves full best-response each iteration
    3. Cesaro averaging suppresses particle noise over iterations

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
        config: Configuration object (optional)
        damping: U's and M's damping, a `DampingPolicy` (#2997). None is ``DampingPolicy.fictitious_play()``: M averaged
            with alpha_k = 1/(k+1) and U the best response taken whole. ``fictitious_play(alpha0, schedule, floor=,
            damp_value_function=)`` sets the first rate, the schedule ("harmonic", "sqrt", "polynomial" or a callable),
            a floor, and whether U is averaged too. A positive floor turns the schedule into constant damping once the
            schedule falls below it, and constant damping is not fictitious play: the convergence results are for the
            unfloored 1/(n+1) average (Cardaliaguet-Hadikhanloo 2017, (2.2) with Theorem 2.1 for the second-order
            system, (3.3) with Theorem 3.1 for the first-order one). On #1914's sigma=0 fixture, measured at 0f937601, a
            0.01 floor left the fixed-point residual at 2.45e-01 after 300 sweeps, against 4.67e-04 unfloored (#2415).
        backend: Backend NAME (resolved via ``create_backend``) or a backend OBJECT, or None.
            Only backends whose arrays the loop can write into carry a solve -- jax and
            torch are refused at allocation, naming #1922 (see ``iteration_state.allocate_iterate``).
        volatility: Optional volatility override -- the SDE volatility, not a diffusion
        volatility_kind: 'field' or 'tensor' for an array override (#2378 part 2a)
        drift_field: Optional drift override for non-MFG problems

    Example:
        >>> from mfgarchon.alg.numerical.coupling import FictitiousPlayIterator
        >>>
        >>> solver = FictitiousPlayIterator(
        ...     problem=problem,
        ...     hjb_solver=hjb_solver,
        ...     fp_solver=fp_solver,
        ... )
        >>> result = solver.solve(max_iterations=100, tolerance=1e-4)
    """

    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
        config: MFGSolverConfig | None = None,
        damping: DampingPolicy | None = None,
        backend: str | BaseBackend | None = None,
        volatility: float | np.ndarray | Any | None = None,
        drift_field: np.ndarray | Any | None = None,
        volatility_kind: str | None = None,
    ):
        super().__init__(problem)
        assert_bc_providers_resolvable(self.problem, "FictitiousPlayIterator")
        # #2250: resolved here for a fast failure on an obviously bad value; resolved
        # again in allocate_iterate, at the start of each solve, which also catches a name or bad type assigned
        # to self.backend after construction (a public attribute -- see resolve_backend).
        self.backend = resolve_backend(backend, "FictitiousPlayIterator")
        self.hjb_solver = hjb_solver
        self.fp_solver = fp_solver
        assert_paired_solver_sigma(
            hjb_solver, fp_solver, "FictitiousPlayIterator", problem=problem, override=volatility
        )
        assert_paired_solver_geometry(hjb_solver, fp_solver, "FictitiousPlayIterator", problem=problem)
        self.config = config

        self.damping = damping if damping is not None else DampingPolicy.fictitious_play()

        # PDE coefficient overrides
        self.volatility = volatility
        self.volatility_kind = volatility_kind
        self.drift_field = drift_field

        # State arrays (initialized in solve)
        self.U: np.ndarray | None = None
        self.M: np.ndarray | None = None

        # Convergence tracking
        self.l2distu_abs: np.ndarray | None = None
        self.l2distm_abs: np.ndarray | None = None
        self.l2distu_rel: np.ndarray | None = None
        self.l2distm_rel: np.ndarray | None = None
        self.learning_rate_history: list[float] = []
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
            )
        return self._state

    @verdict_reads_inner_failures(lambda self: [self.hjb_solver], fp_solvers_of=lambda self: [self.fp_solver])
    def solve(
        self,
        config: MFGSolverConfig | None = None,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        *,
        iteration_callback: Callable[[int, np.ndarray, np.ndarray, float, float], bool | None] | None = None,
        absolute_tolerance: float | None = None,
        **kwargs: Any,
    ) -> SolverResult:
        """
        Solve coupled MFG system using Fictitious Play.

        Args:
            config: Solver configuration (overrides instance config)
            max_iterations: Maximum iterations (legacy parameter)
            tolerance: Bound on the relative L2 change of one sweep (legacy parameter; CONVENTIONS.md § 9)
            iteration_callback: Optional callback for custom logging per iteration.
                Signature: (iteration, U, M, error_U, error_M) -> bool | None
                Return True to stop early, None/False to continue.
            absolute_tolerance: If given, the absolute L2 change must also fall below it (#2555)
            **kwargs: ``verbose`` (default True). A keyword this method no longer takes (``RETIRED_SOLVE_KEYWORDS``)
                is refused by name.

        Returns:
            SolverResult object
        """
        refuse_retired_solve_keywords(kwargs, "FictitiousPlayIterator.solve")
        refuse_a_capped_exit_in_a_coupled_solve([self.fp_solver], "FictitiousPlayIterator.solve")
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
            verbose = kwargs.get("verbose", True)

        # The iterate's layout, its cold or warm start, and m(0), from its one owner (#2997), which refuses a geometry
        # other than a CartesianGrid for this route.
        state = self._iteration_state()
        time_step = self.problem.dt
        integrate = self.problem.spatial_measure().integrate  # #2555

        # #2250: the owner's allocation refuses a backend that cannot carry a solve, reading self.backend here.
        self.U, self.M = state.start(self.backend)
        M_initial = state.m0

        # Initialize error tracking
        self.l2distu_abs = np.ones(final_max_iterations)
        self.l2distm_abs = np.ones(final_max_iterations)
        self.l2distu_rel = np.ones(final_max_iterations)
        self.l2distm_rel = np.ones(final_max_iterations)
        self.learning_rate_history = []
        run = DampingRun(self.damping)
        self.iterations_run = 0

        # Main Fictitious Play iteration loop
        converged = False
        convergence_reason = "Maximum iterations reached"

        # Progress bar (Issue #587 Protocol pattern)
        from mfgarchon.utils.progress import create_progress_bar

        iter_range = create_progress_bar(
            range(final_max_iterations),
            verbose=verbose,
            desc="Fictitious Play",
        )

        for k in iter_range:
            iter_start = time.time()

            U_old = self.U.copy()
            M_old = self.M.copy()
            # u(T) at the density this iteration's HJB solve uses (#1642 stage 2), for the solve, the restore and the
            # re-imposition alike.
            U_terminal = state.terminal_at(M_old[-1])

            # Compute learning rate for this iteration
            alpha = run.theta("M", k)
            self.learning_rate_history.append(alpha)

            # 1. Solve HJB backward — progress handled via context routing (#934)
            # Issue #2207: this call composed no source at all before -- a problem carrying
            # source_term_hjb / nonlocal_operator solved here without it, bit-identically
            # to one that carried neither. The iterates are now required and the builder
            # composes.
            hjb_kwargs = self._build_hjb_kwargs(
                M=M_old, U=U_old, volatility=self.volatility, volatility_kind=self.volatility_kind
            )
            U_new = self.hjb_solver.solve_hjb_system(M_old, U_terminal, U_old, **hjb_kwargs)

            # Issue #1718: the FP call below composes its drift from U_new, so a non-finite U_new
            # makes the FP solver raise a CFL diagnostic for an HJB failure. One owner for the
            # check, the terminal-condition restore, and which iterate gets published.
            diverged = diverged_value_function(
                U_new, U_terminal, site="FictitiousPlayIterator", density=M_old, iteration=k
            )
            if diverged is not None:
                self.U = diverged
                converged = False
                convergence_reason = "diverged_nan"
                self.iterations_run = k + 1
                break

            # 2. Solve FP forward with new U.
            # Issue #1043 Phase 2: route through single-source convention, same as
            # FixedPointIterator.solve().
            # Previously FictitiousPlay called _build_fp_kwargs(drift_field=U_new) which
            # silently passed the value function U as the velocity alpha* to FPFDMSolver
            # (wrong physics: FPFDMSolver treats drift_field as alpha, not potential).
            fp_kwargs = self._build_fp_kwargs(
                M=M_old, U=U_new, volatility=self.volatility, volatility_kind=self.volatility_kind
            )
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
                fp_kwargs.update(drift_kwargs)
                if use_positional_U:
                    M_candidate = self.fp_solver.solve_fp_system(M_initial, U_new, **fp_kwargs)
                else:
                    M_candidate = self.fp_solver.solve_fp_system(M_initial, **fp_kwargs)
            else:
                effective_drift = self.drift_field if self.drift_field is not None else U_new
                M_candidate = self.fp_solver.solve_fp_system(M_initial, effective_drift, **fp_kwargs)

            # 3. Fictitious Play update: m_{n+1} = (1 - alpha_n) m_n + alpha_n m_candidate, and U the best response
            # taken whole unless the policy averages it too (the policy, #2997)
            self.U, self.M = run.update(k, U_old, U_new, M_old, M_candidate)

            self.U, self.M = state.write_back(self.U, self.M, U_terminal)

            # Calculate convergence metrics
            from mfgarchon.utils.convergence import sweep_change

            # The residual of the map, M_candidate vs M_old -- not the averaged step. The step is
            # exactly alpha * (M_candidate - M_old), so under a decaying alpha it shrinks whether or
            # not the belief is near a fixed point: measured at 0f937601 with the floor removed, the
            # default tolerance reported converged=True at sweep 359 with the map residual 3.57e-04
            # on #1914's sigma=0 fixture (#2415). This is
            # #1684 item 7, which FixedPointIterator already measures this way.
            metrics = sweep_change(U_new, U_old, M_candidate, M_old, integrate, time_step)
            self.l2distu_abs[k] = metrics["l2distu_abs"]
            self.l2distu_rel[k] = metrics["l2distu_rel"]
            self.l2distm_abs[k] = metrics["l2distm_abs"]
            self.l2distm_rel[k] = metrics["l2distm_rel"]

            iter_time = time.time() - iter_start
            self.iterations_run = k + 1

            # Invoke iteration callback if provided (for text logging in non-TTY mode)
            if iteration_callback is not None:
                should_stop = iteration_callback(k, self.U, self.M, self.l2distu_rel[k], self.l2distm_rel[k])
                if should_stop:
                    convergence_reason = "Stopped by iteration callback"
                    converged = False
                    break

            # Update progress metrics (Issue #587 Protocol - no hasattr needed)
            iter_range.update_metrics(
                U_err=f"{self.l2distu_rel[k]:.2e}",
                M_err=f"{self.l2distm_rel[k]:.2e}",
                alpha=f"{alpha:.3f}",
                t=f"{iter_time:.1f}s",
            )

            # Check convergence
            converged, convergence_reason = check_convergence_criteria(
                self.l2distu_rel[k],
                self.l2distm_rel[k],
                self.l2distu_abs[k],
                self.l2distm_abs[k],
                final_tolerance,
                final_absolute_tolerance,
            )

            if converged:
                # Log convergence message (Issue #587 Protocol - no hasattr needed)
                iter_range.log(convergence_reason)
                break

        # Construct result
        result = SolverResult(
            U=self.U,
            M=self.M,
            iterations=self.iterations_run,
            error_history_U=self.l2distu_abs[: self.iterations_run],
            error_history_M=self.l2distm_abs[: self.iterations_run],
            solver_name=self.name,
            converged=converged,
            metadata={
                "convergence_reason": convergence_reason,
                "l2distu_rel": self.l2distu_rel[: self.iterations_run],
                "l2distm_rel": self.l2distm_rel[: self.iterations_run],
                "damping": self.damping,
                "learning_rate_history": self.learning_rate_history,
            },
        )

        return result

    @property
    def name(self) -> str:
        """Solver name for diagnostics."""
        return "FictitiousPlayIterator"

    def get_convergence_data(self) -> dict[str, np.ndarray | list[float]]:
        """Get convergence diagnostics including learning rate history."""
        return {
            "l2distu_abs": self.l2distu_abs[: self.iterations_run] if self.l2distu_abs is not None else np.array([]),
            "l2distm_abs": self.l2distm_abs[: self.iterations_run] if self.l2distm_abs is not None else np.array([]),
            "l2distu_rel": self.l2distu_rel[: self.iterations_run] if self.l2distu_rel is not None else np.array([]),
            "l2distm_rel": self.l2distm_rel[: self.iterations_run] if self.l2distm_rel is not None else np.array([]),
            "learning_rate_history": self.learning_rate_history,
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

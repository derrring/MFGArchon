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

# Import shared iteration utilities (Issue #630, migrated #985)
from mfgarchon.utils.iteration.schedules import (
    LEARNING_RATE_SCHEDULES,
    get_schedule,
    harmonic_schedule,
)
from mfgarchon.utils.solver_result import SolverResult

from .base_mfg import BaseCouplingIterator
from .fixed_point_utils import (
    check_convergence_criteria,
    initialize_cold_start,
    preserve_initial_condition,
    preserve_terminal_condition,
    resolve_fp_drift_kwargs,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfgarchon.config import MFGSolverConfig
    from mfgarchon.problem.base_mfg_problem import MFGProblem


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
        learning_rate_schedule: Schedule type or callable
            - "harmonic": 1/(k+1) - standard fictitious play (default)
            - "sqrt": 1/sqrt(k+1) - faster initial progress
            - "polynomial": 1/(k+1)^0.6 - balanced rate
            - Callable[[int], float]: Custom schedule function
        config: Configuration object (optional)
        initial_learning_rate: Learning rate for first iteration (default 1.0)
        min_learning_rate: Minimum learning rate floor (default 0.01)
        damp_value_function: Whether to also damp U (default False)
            - False: Pure fictitious play (damp only M)
            - True: Hybrid approach (damp both U and M)
        backend: Backend name ('numpy', 'torch', 'jax', etc.)
        volatility_field: Optional diffusion override
        drift_field: Optional drift override for non-MFG problems

    Example:
        >>> from mfgarchon.alg.numerical.coupling import FictitiousPlayIterator
        >>>
        >>> solver = FictitiousPlayIterator(
        ...     problem=problem,
        ...     hjb_solver=hjb_solver,
        ...     fp_solver=fp_solver,
        ...     learning_rate_schedule="harmonic",
        ... )
        >>> result = solver.solve(max_iterations=100, tolerance=1e-4)
    """

    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
        learning_rate_schedule: str | Callable[[int], float] = "harmonic",
        config: MFGSolverConfig | None = None,
        initial_learning_rate: float = 1.0,
        min_learning_rate: float = 0.01,
        damp_value_function: bool = False,
        backend: str | None = None,
        volatility_field: float | np.ndarray | Any | None = None,
        drift_field: np.ndarray | Any | None = None,
    ):
        super().__init__(problem)
        self.backend = backend
        self.hjb_solver = hjb_solver
        self.fp_solver = fp_solver
        self.config = config

        # Fictitious play parameters
        self.initial_learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.damp_value_function = damp_value_function

        # Set up learning rate schedule (Issue #630: use shared schedules)
        self._lr_schedule_fn = get_schedule(learning_rate_schedule)
        self._lr_schedule_name = learning_rate_schedule if isinstance(learning_rate_schedule, str) else "custom"

        # PDE coefficient overrides
        self.volatility_field = volatility_field
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

        # Warm start support
        self._warm_start_U: np.ndarray | None = None
        self._warm_start_M: np.ndarray | None = None

        # Cache solver signatures via base class (Issue #934)
        self._init_solver_signatures(self.hjb_solver, self.fp_solver)

    def get_learning_rate(self, iteration: int) -> float:
        """
        Compute learning rate for given iteration.

        Args:
            iteration: Current iteration number (0-indexed)

        Returns:
            Learning rate alpha in (0, 1]
        """
        # Apply schedule
        alpha = self._lr_schedule_fn(iteration)

        # Scale by initial rate and clamp to minimum
        alpha = self.initial_learning_rate * alpha
        alpha = max(alpha, self.min_learning_rate)

        return alpha

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

    def solve(
        self,
        config: MFGSolverConfig | None = None,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        return_tuple: bool = False,
        iteration_callback: Callable[[int, np.ndarray, np.ndarray, float, float], bool | None] | None = None,
        **kwargs: Any,
    ) -> SolverResult | tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
        """
        Solve coupled MFG system using Fictitious Play.

        Args:
            config: Solver configuration (overrides instance config)
            max_iterations: Maximum iterations (legacy parameter)
            tolerance: Convergence tolerance (legacy parameter)
            return_tuple: Return legacy tuple format instead of SolverResult
            iteration_callback: Optional callback for custom logging per iteration.
                Signature: (iteration, U, M, error_U, error_M) -> bool | None
                Return True to stop early, None/False to continue.
            **kwargs: Additional parameters for backward compatibility

        Returns:
            SolverResult object (or tuple if return_tuple=True)
        """
        # Use provided config or fall back to instance config
        solve_config = config or self.config

        # Parameter resolution (config > explicit args > instance defaults)
        if solve_config is not None:
            final_max_iterations = solve_config.picard.max_iterations
            final_tolerance = solve_config.picard.tolerance
            verbose = solve_config.picard.verbose
        else:
            final_max_iterations = (
                max_iterations or kwargs.get("max_picard_iterations") or kwargs.get("Niter_max") or 100
            )
            final_tolerance = tolerance or kwargs.get("picard_tolerance") or kwargs.get("l2errBoundPicard") or 1e-6
            verbose = kwargs.get("verbose", True)

        # Get problem dimensions
        num_time_steps = self.problem.Nt + 1

        # Detect problem shape using geometry API
        from mfgarchon.geometry.base import CartesianGrid

        # Issue #543 Phase 2: Replace hasattr with try/except
        try:
            geometry = self.problem.geometry
        except AttributeError as e:
            raise ValueError("Problem must have 'geometry' attribute") from e

        if geometry is None:
            raise ValueError("Problem geometry cannot be None")

        if not isinstance(geometry, CartesianGrid):
            raise ValueError("Problem geometry must be CartesianGrid")

        shape = tuple(self.problem.geometry.get_grid_shape())
        grid_spacing = self.problem.geometry.get_grid_spacing()[0]
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
                if len(shape) == 1:
                    self.M[0, :] = M_initial
                    self.U[num_time_steps - 1, :] = U_terminal
                else:
                    self.M[0] = M_initial
                    self.U[num_time_steps - 1] = U_terminal

                self.U, self.M = initialize_cold_start(self.U, self.M, M_initial, U_terminal, num_time_steps)

        # Initialize error tracking
        self.l2distu_abs = np.ones(final_max_iterations)
        self.l2distm_abs = np.ones(final_max_iterations)
        self.l2distu_rel = np.ones(final_max_iterations)
        self.l2distm_rel = np.ones(final_max_iterations)
        self.learning_rate_history = []
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

            # Compute learning rate for this iteration
            alpha = self.get_learning_rate(k)
            self.learning_rate_history.append(alpha)

            # 1. Solve HJB backward — progress handled via context routing (#934)
            hjb_kwargs = self._build_hjb_kwargs(volatility_field=self.volatility_field)
            U_new = self.hjb_solver.solve_hjb_system(M_old, U_terminal, U_old, **hjb_kwargs)

            # 2. Solve FP forward with new U.
            # Issue #1043 Phase 2: route through single-source convention, same as
            # FixedPointIterator.solve() (fixed_point_iterator.py:720-728).
            # Previously FictitiousPlay called _build_fp_kwargs(drift_field=U_new) which
            # silently passed the value function U as the velocity alpha* to FPFDMSolver
            # (wrong physics: FPFDMSolver treats drift_field as alpha, not potential).
            fp_kwargs = self._build_fp_kwargs(volatility_field=self.volatility_field)
            if self._fp_sig_params is not None:
                drift_kwargs, use_positional_U = resolve_fp_drift_kwargs(
                    self.problem, self._fp_sig_params, self.drift_field, U_new, M_old
                )
                fp_kwargs.update(drift_kwargs)
                if use_positional_U:
                    M_candidate = self.fp_solver.solve_fp_system(M_initial, U_new, **fp_kwargs)
                else:
                    M_candidate = self.fp_solver.solve_fp_system(M_initial, **fp_kwargs)
            else:
                effective_drift = self.drift_field if self.drift_field is not None else U_new
                M_candidate = self.fp_solver.solve_fp_system(M_initial, effective_drift, **fp_kwargs)

            # 3. Fictitious Play update: average M with decaying learning rate
            # This is the key difference from fixed-point iteration
            # m_{n+1} = (1 - alpha_n) * m_n + alpha_n * m_candidate
            self.M = (1 - alpha) * M_old + alpha * M_candidate

            # Optionally damp U as well (hybrid mode)
            if self.damp_value_function:
                self.U = (1 - alpha) * U_old + alpha * U_new
            else:
                # Pure fictitious play: use full best response for U
                self.U = U_new

            # Preserve boundary conditions
            self.M = preserve_initial_condition(self.M, M_initial)
            self.U = preserve_terminal_condition(self.U, U_terminal)

            # Calculate convergence metrics
            from mfgarchon.utils.convergence import calculate_l2_convergence_metrics

            metrics = calculate_l2_convergence_metrics(self.U, U_old, self.M, M_old, grid_spacing, time_step)
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
                "learning_rate_schedule": self._lr_schedule_name,
                "learning_rate_history": self.learning_rate_history,
                "damp_value_function": self.damp_value_function,
            },
        )

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
    print("Testing FictitiousPlayIterator...")

    # Test class availability
    assert FictitiousPlayIterator is not None
    print("  FictitiousPlayIterator class available")

    # Test learning rate schedules
    print("\n  Learning rate schedules:")
    for name, fn in LEARNING_RATE_SCHEDULES.items():
        rates = [fn(k) for k in [0, 1, 4, 9, 29, 99]]
        print(
            f"    {name}: k=0:{rates[0]:.3f}, k=1:{rates[1]:.3f}, k=4:{rates[2]:.3f}, "
            f"k=9:{rates[3]:.3f}, k=29:{rates[4]:.3f}, k=99:{rates[5]:.3f}"
        )

    # Verify harmonic schedule matches theory
    alpha_1 = harmonic_schedule(0)  # k=0 -> alpha = 1/(0+1) = 1.0
    alpha_2 = harmonic_schedule(1)  # k=1 -> alpha = 1/(1+1) = 0.5
    alpha_10 = harmonic_schedule(9)  # k=9 -> alpha = 1/10 = 0.1
    alpha_100 = harmonic_schedule(99)  # k=99 -> alpha = 1/100 = 0.01

    assert abs(alpha_1 - 1.0) < 1e-10, f"Expected 1.0, got {alpha_1}"
    assert abs(alpha_2 - 0.5) < 1e-10, f"Expected 0.5, got {alpha_2}"
    assert abs(alpha_10 - 0.1) < 1e-10, f"Expected 0.1, got {alpha_10}"
    assert abs(alpha_100 - 0.01) < 1e-10, f"Expected 0.01, got {alpha_100}"
    print("  Harmonic schedule verified")

    print("\nSmoke tests passed!")

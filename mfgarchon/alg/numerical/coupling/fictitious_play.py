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

from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.alg.numerical.coupling.fixed_point_utils import verdict_reads_inner_failures
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
    from collections.abc import Callable

    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfgarchon.backends.base_backend import BaseBackend
    from mfgarchon.config import MFGSolverConfig
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.utils.solver_result import SolverResult


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
            A solve's config that sets ``PicardConfig.damping`` decides it instead; a damping given both here and in that
            config is refused at the solve.
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

        # The constructor's damping, or None for fictitious play's own; a solve's config may set it instead (#2997).
        self.damping = damping_argument(damping, "FictitiousPlayIterator")

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
        track_measure_field: bool = False,
        absolute_tolerance: float | None = None,
        verbose: bool = True,
    ) -> SolverResult:
        """
        Solve coupled MFG system using Fictitious Play.

        Args:
            config: Solver configuration (overrides instance config)
            max_iterations: The sweeps the solve may run (legacy parameter); None is 100, and 0 runs none (#3002)
            tolerance: Bound on the relative L2 change of one sweep (legacy parameter; CONVENTIONS.md § 9)
            iteration_callback: Optional callback called after each iteration.
                Signature: (iteration, U, M, error_U, error_M) -> bool | None
                Return False to stop early; True or None continues: the convention every coupled route reads (#2886).
                Before #3002 stage 2a this route read the opposite: True stopped and False continued.
            track_measure_field: If True, store each iterate as a MeasureField snapshot in
                SolverResult.metadata["measure_field"] (#956). Default False.
            absolute_tolerance: If given, the absolute L2 change must also fall below it (#2555)
            verbose: Show progress (default True). A config in force governs progress instead, through its
                ``picard.verbose``, as its ``picard`` fields decide the budget and the tolerances (#2922).

        Returns:
            SolverResult object
        """
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
            final_max_iterations = 100 if max_iterations is None else max_iterations
            final_tolerance = tolerance or 1e-6
            final_absolute_tolerance = absolute_tolerance

        # The iterate's layout, its cold or warm start, and m(0), from its one owner (#2997), which refuses a geometry
        # other than a CartesianGrid for this route.
        state = self._iteration_state()
        # #2250: the owner's allocation refuses a backend that cannot carry a solve, reading self.backend here.
        self.U, self.M = state.start(self.backend)
        policy = damping_for_the_solve(
            "FictitiousPlayIterator", self.damping, DampingPolicy.fictitious_play(), solve_config
        )
        run = DampingRun(policy)

        # One driver loop for every coupled route (#3002 stage 2a). The sweep map is FixedPoint's; fictitious play is
        # its damping, m_{n+1} = (1 - alpha_n) m_n + alpha_n m_candidate with U the best response taken whole unless
        # the policy averages it too, and its rates are recorded (#2997). The verdict reads the map's residual, not
        # the averaged step, which shrinks under a decaying alpha whether or not the belief is near a fixed point
        # (#2415, #1684 item 7).
        with CoupledDriver(
            "Fictitious Play",
            state,
            self.hjb_solver,
            budget=final_max_iterations,
            tolerance=final_tolerance,
            absolute_tolerance=final_absolute_tolerance,
            iteration_callback=iteration_callback,
            verbose=verbose,
            track_measure_field=track_measure_field,
            track_rates=True,
        ) as driver:
            self._record = driver.record
            driver.snapshot(self.U, self.M)
            step = FixedPointStep(self, state, run, order="gauss_seidel", site="FictitiousPlayIterator")
            outcome = driver.run_fixed_point(step, self.U, self.M, driver.budget)
        self.U, self.M = outcome.U, outcome.M
        self.iterations_run = driver.sweeps
        self.l2distu_abs = driver.histories["l2distu_abs"]
        self.l2distm_abs = driver.histories["l2distm_abs"]
        self.l2distu_rel = driver.histories["l2distu_rel"]
        self.l2distm_rel = driver.histories["l2distm_rel"]
        self.learning_rate_history = driver.rates
        metadata: dict[str, Any] = {"learning_rate_history": self.learning_rate_history}
        if driver.measure_field is not None:
            metadata["measure_field"] = driver.measure_field

        # The result, through the one owner every coupled route reads (#2997)
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
        return "FictitiousPlayIterator"

    def get_convergence_data(self) -> dict[str, np.ndarray | list[float]]:
        """The last solve's change per sweep, from the record its result reads (#2997), and its learning rates; empty
        before a solve."""
        histories = {
            key: np.array([]) if self._record is None else self._record.history(key)
            for key in ("l2distu_abs", "l2distm_abs", "l2distu_rel", "l2distm_rel")
        }
        return {**histories, "learning_rate_history": self.learning_rate_history}

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

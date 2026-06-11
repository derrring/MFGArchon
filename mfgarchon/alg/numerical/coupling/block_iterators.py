"""
Block Iteration Solvers for MFG Systems.

Issue #492 Phase 2: Block methods for coupled HJB-FP systems.

Provides two classical block iteration methods:
- Block Jacobi: Update U and M using values from previous iteration
- Block Gauss-Seidel: Use updated U immediately when computing M

Mathematical Background:
    The MFG system at equilibrium satisfies:
        U = HJB_solve(M)  (value function)
        M = FP_solve(U)   (density)

    Block Jacobi iteration (parallel updates):
        U^{k+1} = HJB_solve(M^k)
        M^{k+1} = FP_solve(U^k)  <- uses OLD U

    Block Gauss-Seidel iteration (sequential updates):
        U^{k+1} = HJB_solve(M^k)
        M^{k+1} = FP_solve(U^{k+1})  <- uses NEW U

    Gauss-Seidel typically converges faster due to using fresh information,
    but Jacobi can be parallelized across HJB and FP solves.

References:
    - Achdou & Capuzzo-Dolcetta (2010): Mean field games: Numerical methods
    - Saad (2003): Iterative Methods for Sparse Linear Systems
"""

from __future__ import annotations

import time
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.deprecation import deprecated_parameter
from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.solver_result import SolverResult

from .base_mfg import BaseCouplingIterator
from .fixed_point_utils import (
    apply_damping,
    check_convergence_criteria,
    initialize_cold_start,
    preserve_initial_condition,
    preserve_terminal_condition,
    resolve_fp_drift_kwargs,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfgarchon.core.mfg_problem import MFGProblem

logger = get_logger(__name__)


class BlockMethod(Enum):
    """Block iteration method selection."""

    JACOBI = "jacobi"
    GAUSS_SEIDEL = "gauss_seidel"


class BlockIterator(BaseCouplingIterator):
    """
    Block iteration solver for coupled MFG systems.

    Provides unified implementation of Jacobi and Gauss-Seidel block methods
    with optional damping for robustness.

    Required Geometry Traits (Issue #596 Phase 2.3):
        This coupling solver requires trait-validated HJB and FP component solvers:
        - HJB solver must use geometry with SupportsGradient trait
        - FP solver must use geometry with SupportsLaplacian trait

        Trait validation occurs in component solvers, not at coupling layer.
        See HJBFDMSolver and FPFDMSolver docstrings for trait details.

    Args:
        problem: MFG problem definition
        hjb_solver: HJB solver instance
        fp_solver: FP solver instance
        method: Block iteration method ('jacobi' or 'gauss_seidel')
        damping_factor: Damping parameter in (0, 1] (default: 1.0 = no damping)
        volatility_field: Optional diffusion override
        drift_field: Optional drift override for non-MFG problems
        adjoint_mode: Adjoint enforcement mode (Issue #622, #704, #707).
            - "off": Independent matrix construction (default)
            - "transpose": FP uses A_hjb^T directly (velocity-based, only valid for symmetric BCs)
            - "jacobian_transpose": FP uses A_Jac^T from linearized HJB operator (Issue #707).
              Correct for arbitrary Hamiltonians. Requires class-based Hamiltonian.
            - "auto": Detects BC types, uses transpose for symmetric, correction for Dirichlet
        adjoint_verify: Enable runtime adjoint verification for debugging.

    Example:
        >>> # Block Gauss-Seidel (faster convergence)
        >>> solver = BlockIterator(problem, hjb_solver, fp_solver, method='gauss_seidel')
        >>> result = solver.solve(max_iterations=50, tolerance=1e-5)
        >>>
        >>> # Block Jacobi (can be parallelized)
        >>> solver = BlockIterator(problem, hjb_solver, fp_solver, method='jacobi')
        >>> result = solver.solve(max_iterations=100, tolerance=1e-5)
    """

    @deprecated_parameter(param_name="damping_factor", since="v0.19.2", replacement="relaxation")
    @deprecated_parameter(param_name="damping_factor_M", since="v0.19.2", replacement="relaxation_M")
    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
        *,
        method: str | BlockMethod = BlockMethod.GAUSS_SEIDEL,
        relaxation: float = 1.0,
        relaxation_M: float | None = None,
        volatility_field: float | NDArray | Any | None = None,
        drift_field: NDArray | Any | None = None,
        adjoint_mode: str = "off",
        adjoint_verify: bool = False,
        # Legacy damping_* kwargs (deprecated since v0.19.2, removal v0.25.0)
        damping_factor: float | None = None,
        damping_factor_M: float | None = None,
    ):
        """Initialize block iterator.

        Args:
            relaxation: Under-relaxation factor for U (omega_U). Default 1.0 = no relaxation.
            relaxation_M: Under-relaxation factor for M (omega_M). If None, uses `relaxation`.
                Issue #719: Per-variable relaxation support.
                Recommended for MFG: relaxation=1.0, relaxation_M=0.2
                (U adapts fully, M filters particle noise).

            Legacy `damping_factor` / `damping_factor_M` kwargs still accepted with
            DeprecationWarning.
        """
        # Redirect legacy kwargs -> canonical (decorator already warned)
        if damping_factor is not None:
            relaxation = damping_factor
        if damping_factor_M is not None:
            relaxation_M = damping_factor_M

        super().__init__(problem)

        self.hjb_solver = hjb_solver
        self.fp_solver = fp_solver

        # Parse method
        if isinstance(method, str):
            method = BlockMethod(method.lower())
        self.method = method

        # Canonical relaxation state - Issue #719: Per-variable relaxation
        self.relaxation = relaxation
        self.relaxation_M = relaxation_M  # None = use `relaxation` for both

        # PDE coefficient overrides
        self.volatility_field = volatility_field
        self.drift_field = drift_field

        # Issue #622, #704, #707: Adjoint mode
        _valid_adjoint_modes = ("off", "transpose", "jacobian_transpose", "auto")
        if adjoint_mode not in _valid_adjoint_modes:
            raise ValueError(f"adjoint_mode must be one of {_valid_adjoint_modes}, got '{adjoint_mode}'")

        # Deprecate old modes (#706, #707)
        if adjoint_mode in ("transpose", "auto"):
            import warnings

            warnings.warn(
                f"adjoint_mode='{adjoint_mode}' is deprecated since v0.17.13. "
                "Use adjoint_mode='jacobian_transpose' instead, which correctly "
                "transposes the linearized HJB Jacobian (Issue #707). "
                f"The '{adjoint_mode}' mode will be removed in v1.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.adjoint_mode = adjoint_mode
        self.adjoint_verify = adjoint_verify
        self._adjoint_mismatch_count = 0

        if adjoint_mode != "off":
            self._validate_strict_adjoint_capability()

        # Solution state
        self.U: NDArray | None = None
        self.M: NDArray | None = None

        # Convergence tracking
        self.error_history_U: list[float] = []
        self.error_history_M: list[float] = []
        self.iterations_run = 0

        # Cache solver signatures via base class (Issue #934)
        self._init_solver_signatures(self.hjb_solver, self.fp_solver)

        # Cache initial/terminal conditions
        self._M_initial: NDArray | None = None
        self._U_terminal: NDArray | None = None

    # ------------------------------------------------------------------
    # Backward-compat attribute aliases (damping_* -> relaxation_*)
    # Added in v0.19.2. Silent (no DeprecationWarning on read) to avoid
    # log flooding in block iteration loops. Removal in v0.25.0;
    # the ctor kwargs already emit DeprecationWarning via @deprecated_parameter.
    # ------------------------------------------------------------------

    @property
    def damping_factor(self) -> float:
        """Deprecated alias for `relaxation` (v0.19.2+). Removal in v0.25.0."""
        return self.relaxation

    @property
    def damping_factor_M(self) -> float | None:
        """Deprecated alias for `relaxation_M` (v0.19.2+). Removal in v0.25.0."""
        return self.relaxation_M

    def _validate_strict_adjoint_capability(self) -> None:
        """
        Validate that solvers support strict adjoint mode (Issue #622, #704).

        Uses protocol-based validation to ensure solvers implement required methods.
        """
        from mfgarchon.alg.numerical.adjoint import validate_adjoint_capability

        validate_adjoint_capability(self.hjb_solver, self.fp_solver, strict=True)

    def _initialize_conditions(self, shape: tuple[int, ...]) -> tuple[NDArray, NDArray]:
        """Initialize initial density and terminal value from problem."""
        # Try get_m_init() / get_u_fin() methods (preferred)
        try:
            M_initial = self.problem.get_m_initial()
            if M_initial.shape != shape:
                M_initial = M_initial.reshape(shape)
        except AttributeError:
            try:
                M_initial = self.problem.m_initial  # Issue #670: unified naming
                if M_initial is not None and M_initial.shape != shape:
                    M_initial = M_initial.reshape(shape)
            except AttributeError:
                M_initial = np.ones(shape) / np.prod(shape)
                logger.warning("No initial density found, using uniform")

        try:
            U_terminal = self.problem.get_u_terminal()
            if U_terminal.shape != shape:
                U_terminal = U_terminal.reshape(shape)
        except AttributeError:
            try:
                U_terminal = self.problem.u_terminal  # Issue #670: unified naming
                if U_terminal is not None and U_terminal.shape != shape:
                    U_terminal = U_terminal.reshape(shape)
            except AttributeError:
                U_terminal = np.zeros(shape)

        return M_initial, U_terminal

    def _solve_hjb(
        self,
        M: NDArray,
        U_terminal: NDArray,
        U_prev: NDArray,
    ) -> NDArray:
        """Solve HJB equation with given density."""
        # Issue #934: Progress handled via context routing
        kwargs = self._build_hjb_kwargs(volatility_field=self.volatility_field)
        return self.hjb_solver.solve_hjb_system(M, U_terminal, U_prev, **kwargs)

    def _solve_fp(self, M_initial: NDArray, U: NDArray, M_current: NDArray | None = None) -> NDArray:
        """Solve FP equation with given value function.

        Args:
            M_initial: Initial density m_0 for the FP forward solve.
            U: Value function (potential or velocity source, resolved by convention).
            M_current: Current density estimate at this iteration.  Used only for
                non-smooth Hamiltonians where the optimal control alpha* depends on m
                (via ``compute_fp_velocity_field``).  If ``None``, falls back to
                ``M_initial`` (safe default for smooth-separable H).
        """
        # Issue #622: Use adjoint mode if enabled
        if self.adjoint_mode != "off":
            return self._solve_fp_strict_adjoint(M_initial, U)

        _M_curr = M_current if M_current is not None else M_initial

        if self._fp_sig_params is not None:
            kwargs = self._build_fp_kwargs(volatility_field=self.volatility_field)
            # Issue #1043 Phase 2: route through single-source convention, same as
            # FixedPointIterator.solve() (fixed_point_iterator.py:720-728).
            # Previously used a sig-probe heuristic (Issue #919) that put U directly
            # into potential_field/drift_field without Hamiltonian-smoothness dispatch —
            # incorrect for solvers where drift_field is alpha* (e.g. FPGFDMSolver).
            drift_kwargs, use_positional_U = resolve_fp_drift_kwargs(
                self.problem, self._fp_sig_params, self.drift_field, U, _M_curr
            )
            kwargs.update(drift_kwargs)
            if use_positional_U:
                return self.fp_solver.solve_fp_system(M_initial, U, **kwargs)
            else:
                return self.fp_solver.solve_fp_system(M_initial, **kwargs)
        else:
            return self.fp_solver.solve_fp_system(M_initial, U)

    def _solve_fp_strict_adjoint(self, M_initial: NDArray, U: NDArray) -> NDArray:
        """
        Solve FP forward using adjoint mode (Issue #622, #704).

        Args:
            M_initial: Initial density at t=0
            U: Full value function trajectory, shape (Nt+1, *spatial_shape)

        Returns:
            M_solution: Full density trajectory, shape (Nt+1, *spatial_shape)
        """
        num_time_steps = U.shape[0]
        spatial_shape = U.shape[1:]

        M_solution = np.zeros((num_time_steps, *spatial_shape))
        M_solution[0] = M_initial

        sigma = self.volatility_field if self.volatility_field is not None else self.problem.sigma

        # Import utilities
        if self.adjoint_verify:
            from mfgarchon.alg.numerical.adjoint import check_operator_adjoint
        if self.adjoint_mode == "auto":
            from mfgarchon.alg.numerical.adjoint import build_bc_aware_adjoint_matrix

        for k in range(num_time_steps - 1):
            U_k = U[k]
            M_current = M_solution[k]

            # Build FP matrix based on mode
            if self.adjoint_mode == "jacobian_transpose":
                # Issue #707: True adjoint via linearized HJB Jacobian
                # A_fp = A_Jac^T where A_Jac = dH/dp * dp/dU (correct for any Hamiltonian)
                A_jac = self.hjb_solver.build_linearized_operator(U_k, M_current, time=k * self.problem.dt)
                A_fp = A_jac.T.tocsr()
            else:
                # Velocity-based modes use build_advection_matrix
                A_hjb = self.hjb_solver.build_advection_matrix(U_k)

                # Optional verification
                if self.adjoint_verify:
                    fp_build_matrix = getattr(self.fp_solver, "_build_advection_matrix", None)
                    if callable(fp_build_matrix):
                        A_fp_ind = fp_build_matrix(U_k)
                        is_adj, error = check_operator_adjoint(A_hjb, A_fp_ind.T, rtol=1e-10)
                        if not is_adj:
                            self._adjoint_mismatch_count += 1
                            if self._adjoint_mismatch_count <= 5:
                                logger.warning(f"Adjoint mismatch at step {k}: error={error:.2e}")

                if self.adjoint_mode == "transpose":
                    A_fp = A_hjb.T.tocsr()
                else:  # "auto"
                    bc_types = self._get_boundary_bc_types()
                    grid_shape = self.problem.geometry.get_grid_shape()
                    dx = self.problem.geometry.get_grid_spacing()
                    dt = self.problem.dt
                    A_fp = build_bc_aware_adjoint_matrix(A_hjb, bc_types, grid_shape, dx, dt)

            M_next = self.fp_solver.solve_fp_step_adjoint_mode(M_current, A_fp, sigma=sigma, time=k * self.problem.dt)
            M_solution[k + 1] = M_next

        return M_solution

    def _get_boundary_bc_types(self) -> dict[str, str]:
        """Extract BC types from problem geometry."""
        bc_types = {}
        geometry = self.problem.geometry

        try:
            bc = geometry.boundary_conditions
            if bc is not None:
                for segment in getattr(bc, "segments", []):
                    name = getattr(segment, "name", None)
                    bc_type = getattr(segment, "bc_type", None)
                    if name and bc_type:
                        bc_types[name] = bc_type.value if hasattr(bc_type, "value") else str(bc_type)
        except AttributeError:
            pass

        if not bc_types:
            dim = geometry.dimension
            for d in range(dim):
                axis = ["x", "y", "z"][d] if dim <= 3 else f"x_{d + 1}"
                bc_types[f"{axis}_min"] = "reflecting"
                bc_types[f"{axis}_max"] = "reflecting"

        return bc_types

    def _generate_adjoint_report(self):
        """
        Generate comprehensive adjoint diagnostic report (Issue #704).

        Returns:
            AdjointDiagnosticReport if both solvers support matrix building,
            None otherwise.
        """
        from mfgarchon.alg.numerical.adjoint import diagnose_adjoint_error

        fp_build_matrix = getattr(self.fp_solver, "_build_advection_matrix", None)
        if not callable(fp_build_matrix):
            return None

        if self.U is None or len(self.U) < 2:
            return None

        U_for_diagnosis = self.U[-2]

        try:
            A_hjb = self.hjb_solver.build_advection_matrix(U_for_diagnosis)
            A_fp_ind = fp_build_matrix(U_for_diagnosis)

            geometry = getattr(self.problem, "geometry", None)

            return diagnose_adjoint_error(A_hjb, A_fp_ind.T, geometry=geometry)

        except Exception as e:
            logger.warning(f"Failed to generate adjoint report: {e}")
            return None

    def _jacobi_step(
        self,
        U_old: NDArray,
        M_old: NDArray,
        M_initial: NDArray,
        U_terminal: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """
        Perform one Jacobi block iteration.

        Both HJB and FP use values from the previous iteration.

        Returns:
            (U_new, M_new): Updated value function and density
        """
        # Both solves use OLD values (can be parallelized)
        U_new = self._solve_hjb(M_old, U_terminal, U_old)
        M_new = self._solve_fp(M_initial, U_old, M_current=M_old)  # Uses U_old, not U_new

        return U_new, M_new

    def _gauss_seidel_step(
        self,
        U_old: NDArray,
        M_old: NDArray,
        M_initial: NDArray,
        U_terminal: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """
        Perform one Gauss-Seidel block iteration.

        FP uses the freshly computed U from HJB.

        Returns:
            (U_new, M_new): Updated value function and density
        """
        # First solve HJB with old M
        U_new = self._solve_hjb(M_old, U_terminal, U_old)

        # Then solve FP with NEW U (sequential dependency)
        M_new = self._solve_fp(M_initial, U_new, M_current=M_old)  # Uses U_new

        return U_new, M_new

    def solve(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-5,
        verbose: bool = True,
        **kwargs: Any,
    ) -> SolverResult | tuple[NDArray, NDArray, dict[str, Any]]:
        """
        Solve MFG system using block iteration.

        Args:
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            verbose: Print progress information
            **kwargs: Additional parameters

        Returns:
            SolverResult with solution and convergence info
        """
        start_time = time.time()

        # Get problem dimensions
        num_time_steps = self.problem.Nt + 1
        shape = tuple(self.problem.geometry.get_grid_shape())
        grid_spacing = self.problem.geometry.get_grid_spacing()[0]
        time_step = self.problem.dt

        # Initialize conditions
        self._M_initial, self._U_terminal = self._initialize_conditions(shape)

        # Initialize solution arrays
        warm_start = self.get_warm_start_data()
        if warm_start is not None:
            self.U, self.M = warm_start
            if verbose:
                print(f"Using warm start ({self.method.value})")
        else:
            self.U = np.zeros((num_time_steps, *shape))
            self.M = np.zeros((num_time_steps, *shape))

            # Set boundary conditions
            self.M[0] = self._M_initial
            self.U[-1] = self._U_terminal

            # Initialize interior
            self.U, self.M = initialize_cold_start(self.U, self.M, self._M_initial, self._U_terminal, num_time_steps)

        # Reset tracking
        self.error_history_U = []
        self.error_history_M = []
        self.iterations_run = 0

        # Issue #704: Reset adjoint verification counter
        if self.adjoint_verify:
            self._adjoint_mismatch_count = 0

        # Select iteration step function
        if self.method == BlockMethod.JACOBI:
            step_fn = self._jacobi_step
            method_name = "Block Jacobi"
        else:
            step_fn = self._gauss_seidel_step
            method_name = "Block Gauss-Seidel"

        if verbose:
            print(f"Starting {method_name} iteration (relaxation={self.relaxation})")

        # Progress bar (Issue #587 Protocol pattern)
        from mfgarchon.utils.progress import create_progress_bar

        iter_range = create_progress_bar(
            range(max_iterations),
            verbose=verbose,
            desc=method_name,
        )

        converged = False
        convergence_reason = "Maximum iterations reached"

        for iiter in iter_range:
            iter_start = time.time()

            U_old = self.U.copy()
            M_old = self.M.copy()

            # Perform block iteration step
            U_new, M_new = step_fn(U_old, M_old, self._M_initial, self._U_terminal)

            # Apply damping - Issue #719: Per-variable damping
            # Note: apply_damping expects (new, old) order, not (old, new)
            self.U, self.M = apply_damping(
                U_new,
                U_old,
                M_new,
                M_old,
                theta=self.relaxation,
                theta_M=self.relaxation_M,
            )

            # Preserve boundary conditions
            self.M = preserve_initial_condition(self.M, self._M_initial)
            self.U = preserve_terminal_condition(self.U, self._U_terminal)

            # Calculate convergence metrics
            from mfgarchon.utils.convergence import calculate_l2_convergence_metrics

            metrics = calculate_l2_convergence_metrics(self.U, U_old, self.M, M_old, grid_spacing, time_step)

            self.error_history_U.append(metrics["l2distu_rel"])
            self.error_history_M.append(metrics["l2distm_rel"])

            iter_time = time.time() - iter_start
            self.iterations_run = iiter + 1

            # Update progress metrics (Issue #587 Protocol - no hasattr needed)
            iter_range.update_metrics(
                U_err=f"{metrics['l2distu_rel']:.2e}",
                M_err=f"{metrics['l2distm_rel']:.2e}",
                t=f"{iter_time:.1f}s",
            )

            # Check convergence
            converged, convergence_reason = check_convergence_criteria(
                metrics["l2distu_rel"],
                metrics["l2distm_rel"],
                metrics["l2distu_abs"],
                metrics["l2distm_abs"],
                tolerance,
            )

            if converged:
                # Log convergence message (Issue #587 Protocol - no hasattr needed)
                iter_range.log(convergence_reason)
                break

        elapsed = time.time() - start_time

        if verbose:
            status = "converged" if converged else "not converged"
            print(f"{method_name} {status} in {self.iterations_run} iterations ({elapsed:.2f}s)")

        self._solution_computed = True

        # Build metadata
        metadata = {
            "method": self.method.value,
            "relaxation": self.relaxation,
            "convergence_reason": convergence_reason,
            "elapsed_time": elapsed,
            "adjoint_mode": self.adjoint_mode,
        }

        # Issue #704: Include adjoint verification results
        if self.adjoint_verify:
            metadata["adjoint_verify"] = True
            metadata["adjoint_mismatch_count"] = self._adjoint_mismatch_count

            adjoint_report = self._generate_adjoint_report()
            if adjoint_report is not None:
                metadata["adjoint_report"] = adjoint_report

            if self._adjoint_mismatch_count == 0:
                logger.info("Adjoint verification passed")
            else:
                logger.warning(f"Adjoint verification: {self._adjoint_mismatch_count} mismatches")

        return SolverResult(
            U=self.U,
            M=self.M,
            iterations=self.iterations_run,
            error_history_U=np.array(self.error_history_U),
            error_history_M=np.array(self.error_history_M),
            solver_name=self.name,
            converged=converged,
            metadata=metadata,
        )

    @property
    def name(self) -> str:
        """Solver name for diagnostics."""
        return f"BlockIterator({self.method.value})"

    def get_results(self) -> tuple[NDArray, NDArray]:
        """Get computed solution arrays."""
        if self.U is None or self.M is None:
            raise RuntimeError("Solver has not been run yet. Call solve() first.")
        return self.U, self.M


# Convenience aliases
class BlockJacobiIterator(BlockIterator):
    """
    Block Jacobi iterator for MFG systems.

    Updates U and M using values from the previous iteration only.
    Can be parallelized since HJB and FP solves are independent.

    Example:
        >>> solver = BlockJacobiIterator(problem, hjb_solver, fp_solver)
        >>> result = solver.solve(max_iterations=100)
    """

    @deprecated_parameter(param_name="damping_factor", since="v0.19.2", replacement="relaxation")
    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
        *,
        relaxation: float = 0.5,
        damping_factor: float | None = None,
        **kwargs: Any,
    ):
        if damping_factor is not None:
            relaxation = damping_factor
        super().__init__(
            problem,
            hjb_solver,
            fp_solver,
            method=BlockMethod.JACOBI,
            relaxation=relaxation,
            **kwargs,
        )


class BlockGaussSeidelIterator(BlockIterator):
    """
    Block Gauss-Seidel iterator for MFG systems.

    Uses freshly computed U when solving FP (sequential dependency).
    Typically converges faster than Jacobi.

    Example:
        >>> solver = BlockGaussSeidelIterator(problem, hjb_solver, fp_solver)
        >>> result = solver.solve(max_iterations=50)
    """

    @deprecated_parameter(param_name="damping_factor", since="v0.19.2", replacement="relaxation")
    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
        *,
        relaxation: float = 0.5,
        damping_factor: float | None = None,
        **kwargs: Any,
    ):
        if damping_factor is not None:
            relaxation = damping_factor
        super().__init__(
            problem,
            hjb_solver,
            fp_solver,
            method=BlockMethod.GAUSS_SEIDEL,
            relaxation=relaxation,
            **kwargs,
        )


if __name__ == "__main__":
    """Quick smoke test for development.

    Run with: python -m mfgarchon.alg.numerical.coupling.block_iterators
    Or ensure mfgarchon is installed: pip install -e .
    """
    # Issue #757: Removed sys.path manipulation - use proper package installation
    print("Testing BlockIterator...")

    from mfgarchon.alg.numerical.coupling.base_mfg import BaseCouplingIterator
    from mfgarchon.alg.numerical.coupling.block_iterators import (
        BlockGaussSeidelIterator,
        BlockJacobiIterator,
    )
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry import TensorProductGrid

    # Create simple 1D problem
    geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(geometry=geometry, T=0.5, Nt=10, sigma=0.2)

    # Create solvers
    hjb_solver = HJBFDMSolver(problem)
    fp_solver = FPFDMSolver(problem)

    # Test Block Gauss-Seidel
    print("\nTesting Block Gauss-Seidel...")
    gs_solver = BlockGaussSeidelIterator(problem, hjb_solver, fp_solver, relaxation=0.5)
    result_gs = gs_solver.solve(max_iterations=10, tolerance=1e-4, verbose=True)
    print(f"  Converged: {result_gs.converged}, iterations: {result_gs.iterations}")

    # Test Block Jacobi
    print("\nTesting Block Jacobi...")
    jacobi_solver = BlockJacobiIterator(problem, hjb_solver, fp_solver, damping_factor=0.5)
    result_jacobi = jacobi_solver.solve(max_iterations=10, tolerance=1e-4, verbose=True)
    print(f"  Converged: {result_jacobi.converged}, iterations: {result_jacobi.iterations}")

    # Verify shapes
    assert result_gs.U.shape == (11, 21)
    assert result_gs.M.shape == (11, 21)
    assert np.all(np.isfinite(result_gs.U))
    assert np.all(np.isfinite(result_gs.M))

    print("\nBlockIterator smoke tests passed!")

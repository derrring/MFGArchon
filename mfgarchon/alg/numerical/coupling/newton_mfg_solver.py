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

from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.numerical.nonlinear_solvers import NewtonSolver, SolverInfo
from mfgarchon.utils.solver_result import SolverResult

from .base_mfg import BaseCouplingIterator
from .mfg_residual import MFGResidual

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfgarchon.core.mfg_problem import MFGProblem

logger = get_logger(__name__)


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
            warmup residual is well past its transient peak (Issue #1233).
        newton_tolerance: Convergence tolerance for Newton (default: 1e-6)
        newton_max_iterations: Maximum Newton iterations (default: 20)
        line_search: Enable backtracking line search (default: True)
        use_jax_autodiff: Use JAX autodiff for the Jacobian (default: False). The MFG
            residual wraps the black-box numpy/scipy HJB and FP solvers, which JAX
            cannot trace — autodiff therefore raises ``TracerArrayConversionError`` and
            falls back to the finite-difference Jacobian. The FD-Jacobian path is the
            supported one; ``'auto'``/``True`` are retained only for a hypothetical
            JAX-traceable residual and will warn-and-fall-back with the standard solvers
            (Issue #1233).
        volatility_field: Optional diffusion override
        drift_field: Optional drift override

    Example:
        >>> solver = NewtonMFGSolver(problem, hjb_solver, fp_solver)
        >>> U, M, info = solver.solve(max_iterations=30)
        >>> print(f"Converged: {info['converged']}, iterations: {info['iterations']}")
    """

    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
        *,
        picard_warmup: int = 3,
        picard_damping: float = 0.5,
        newton_tolerance: float = 1e-6,
        newton_max_iterations: int = 20,
        line_search: bool = True,
        use_jax_autodiff: bool | str = False,
        volatility_field: float | NDArray | Any | None = None,
        drift_field: NDArray | Any | None = None,
    ):
        """Initialize Newton MFG solver."""
        super().__init__(problem)

        self.hjb_solver = hjb_solver
        self.fp_solver = fp_solver

        # Picard warm-up parameters
        self.picard_warmup = picard_warmup
        self.picard_damping = picard_damping

        # Newton parameters
        self.newton_tolerance = newton_tolerance
        self.newton_max_iterations = newton_max_iterations
        self.line_search = line_search
        self.use_jax_autodiff = use_jax_autodiff

        # PDE coefficient overrides
        self.volatility_field = volatility_field
        self.drift_field = drift_field

        # Create MFG residual computer
        self.mfg_residual = MFGResidual(
            problem,
            hjb_solver,
            fp_solver,
            volatility_field=volatility_field,
            drift_field=drift_field,
        )

        # Solution state
        self.U: NDArray | None = None
        self.M: NDArray | None = None

        # Convergence history
        self.picard_residuals: list[float] = []
        self.newton_residuals: list[float] = []
        self.total_iterations = 0

    def _run_picard_warmup(
        self,
        U: NDArray,
        M: NDArray,
        num_iterations: int,
    ) -> tuple[NDArray, NDArray, list[float]]:
        """
        Run Picard fixed-point iterations for warm-up.

        Args:
            U: Initial value function
            M: Initial density
            num_iterations: Number of Picard iterations

        Returns:
            (U, M, residuals): Updated state and residual history
        """
        residuals = []

        for i in range(num_iterations):
            # Store old values
            U_old = U.copy()
            M_old = M.copy()

            # HJB solve: U_new = HJB(M_old)
            U_new = self.mfg_residual.compute_hjb_output(M_old, U_old)

            # FP solve: M_new = FP(U_new)
            M_new = self.mfg_residual.compute_fp_output(U_new, M_old)

            # Apply damping
            U = self.picard_damping * U_new + (1 - self.picard_damping) * U_old
            M = self.picard_damping * M_new + (1 - self.picard_damping) * M_old

            # Preserve boundary conditions
            if self.mfg_residual.M_initial is not None:
                M[0] = self.mfg_residual.M_initial
            if self.mfg_residual.U_terminal is not None:
                U[-1] = self.mfg_residual.U_terminal

            # Compute residual norm
            res_norm = self.mfg_residual.compute_residual_norm(U, M)
            residuals.append(res_norm)

            logger.debug(f"Picard iteration {i + 1}: residual = {res_norm:.2e}")

        return U, M, residuals

    def solve(
        self,
        max_iterations: int = 30,
        tolerance: float = 1e-5,
        verbose: bool = True,
        **kwargs: Any,
    ) -> tuple[NDArray, NDArray, dict[str, Any]]:
        """
        Solve MFG system using Newton's method.

        Args:
            max_iterations: Maximum total iterations (Picard + Newton)
            tolerance: Convergence tolerance
            verbose: Print progress information
            **kwargs: Additional solver parameters

        Returns:
            (U, M, info): Solution and convergence information
        """
        start_time = time.time()

        # Initialize from warm start or cold start
        warm_start = self.get_warm_start_data()
        if warm_start is not None:
            U, M = warm_start
            if verbose:
                print("Using warm start from previous solution")
        else:
            x0 = self.mfg_residual.get_initial_guess()
            U, M = self.mfg_residual.unpack_state(x0)

        # Phase 1: Picard warm-up (for robustness)
        if self.picard_warmup > 0:
            if verbose:
                print(f"Phase 1: Picard warm-up ({self.picard_warmup} iterations)")

            U, M, picard_res = self._run_picard_warmup(U, M, self.picard_warmup)
            self.picard_residuals = picard_res

            if verbose and picard_res:
                print(f"  Final Picard residual: {picard_res[-1]:.2e}")

        # Check if already converged after Picard
        current_residual = self.mfg_residual.compute_residual_norm(U, M)
        if current_residual < tolerance:
            self.U = U
            self.M = M
            self.total_iterations = self.picard_warmup
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

        # Create Newton solver
        newton_solver = NewtonSolver(
            max_iterations=min(remaining_iterations, self.newton_max_iterations),
            tolerance=self.newton_tolerance,
            sparse=True,
            line_search=self.line_search,
            use_jax_autodiff=self.use_jax_autodiff,
        )

        # Pack current state
        x_current = self.mfg_residual.pack_state(U, M)

        # Run Newton solver
        self.mfg_residual.reset_evaluation_count()
        x_solution, newton_info = newton_solver.solve(
            self.mfg_residual.residual_function,
            x_current,
        )

        # Unpack solution
        self.U, self.M = self.mfg_residual.unpack_state(x_solution)
        self.newton_residuals = newton_info.residual_history
        self.total_iterations = self.picard_warmup + newton_info.iterations

        elapsed = time.time() - start_time

        if verbose:
            status = "converged" if newton_info.converged else "not converged"
            print(f"Newton {status} in {newton_info.iterations} iterations")
            print(f"  Final residual: {newton_info.residual:.2e}")
            print(f"  Jacobian evaluations: {newton_info.extra.get('jacobian_evals', 'N/A')}")
            print(f"  Total time: {elapsed:.2f}s")

        self._solution_computed = True

        return self._create_result(
            converged=newton_info.converged,
            reason="Newton converged" if newton_info.converged else "Newton max iterations",
            elapsed=elapsed,
            newton_info=newton_info,
        )

    def _create_result(
        self,
        converged: bool,
        reason: str,
        elapsed: float,
        newton_info: SolverInfo | None = None,
    ) -> tuple[NDArray, NDArray, dict[str, Any]]:
        """Create result tuple with convergence information."""
        info = {
            "converged": converged,
            "convergence_reason": reason,
            "total_iterations": self.total_iterations,
            "picard_iterations": self.picard_warmup,
            "picard_residuals": self.picard_residuals,
            "newton_iterations": len(self.newton_residuals),
            "newton_residuals": self.newton_residuals,
            "residual_evaluations": self.mfg_residual.residual_evaluations,
            "elapsed_time": elapsed,
        }

        if newton_info is not None:
            info["final_residual"] = newton_info.residual
            info["jacobian_evaluations"] = newton_info.extra.get("jacobian_evals", 0)

        return self.U, self.M, info

    def get_results(self) -> tuple[NDArray, NDArray]:
        """Get computed solution arrays."""
        if self.U is None or self.M is None:
            raise RuntimeError("Solver has not been run yet. Call solve() first.")
        return self.U, self.M


if __name__ == "__main__":
    """Quick smoke test for development.

    Run with: python -m mfgarchon.alg.numerical.coupling.newton_mfg_solver
    Or ensure mfgarchon is installed: pip install -e .
    """
    # Issue #757: Removed sys.path manipulation - use proper package installation
    print("Testing NewtonMFGSolver...")

    from mfgarchon.alg.numerical.coupling.base_mfg import BaseCouplingIterator
    from mfgarchon.alg.numerical.coupling.mfg_residual import MFGResidual
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.utils.numerical.nonlinear_solvers import NewtonSolver, SolverInfo
    from mfgarchon.utils.solver_result import SolverResult  # noqa: F401

    # Create simple 1D problem
    geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(geometry=geometry, T=0.5, Nt=10, sigma=0.2)

    # Create solvers
    hjb_solver = HJBFDMSolver(problem)
    fp_solver = FPFDMSolver(problem)

    # Create Newton MFG solver
    newton_solver = NewtonMFGSolver(
        problem,
        hjb_solver,
        fp_solver,
        picard_warmup=2,
        newton_max_iterations=5,
        line_search=True,
    )

    print(f"  Problem shape: {newton_solver.mfg_residual.solution_shape}")

    # Solve
    U, M, info = newton_solver.solve(max_iterations=10, tolerance=1e-4, verbose=True)

    print("\nResults:")
    print(f"  U shape: {U.shape}")
    print(f"  M shape: {M.shape}")
    print(f"  Converged: {info['converged']}")
    print(f"  Total iterations: {info['total_iterations']}")
    print(f"  Final residual: {info.get('final_residual', 'N/A')}")

    # Verify shapes
    assert U.shape == newton_solver.mfg_residual.solution_shape
    assert M.shape == newton_solver.mfg_residual.solution_shape
    assert np.all(np.isfinite(U)), "U contains inf/nan"
    assert np.all(np.isfinite(M)), "M contains inf/nan"

    print("\nNewtonMFGSolver smoke tests passed!")

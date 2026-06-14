#!/usr/bin/env python3
"""
Direct Variational MFG Solver

This module implements a direct optimization approach for solving Lagrangian MFG problems.
Instead of solving the HJB-FP system, it directly minimizes the cost functional over
admissible density flows.

The solver minimizes:
J[m] = ∫₀ᵀ ∫ L(t,x,v(t,x),m(t,x)) m(t,x) dxdt + ∫ g(x)m(T,x) dx

subject to:
- Continuity equation: ∂m/∂t + ∇·(mv) = σ²/2 Δm
- Mass conservation: ∫m(t,x)dx = 1
- Non-negativity: m(t,x) ≥ 0
"""

from __future__ import annotations

import importlib.util
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.numerical.integration import trapezoid

from .base_variational import BaseVariationalSolver, VariationalSolverResult

logger = get_logger(__name__)
try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

if TYPE_CHECKING:
    from numpy.typing import NDArray

JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


def _experimental_unavailable(detail: str) -> NotImplementedError:
    """Build the Issue #1342 'experimental, not in this release' error."""
    return NotImplementedError(
        f"VariationalMFGSolver is EXPERIMENTAL and not available in this release "
        f"(Issue #1342): {detail}. Its solve logic exists but is unvalidated; "
        f"track #1342 for completion."
    )


class VariationalMFGSolver(BaseVariationalSolver):
    """
    Direct variational solver for Lagrangian MFG problems.

    **Experimental — not production-ready (Issue #1342).**

    This solver treats the MFG problem as a constrained optimization problem
    and uses gradient-based methods to find the optimal density evolution.

    Key features:
    - Direct optimization of the action functional
    - Automatic differentiation for gradients
    - Constraint handling via penalty methods
    - Multiple optimization algorithms
    """

    def __init__(  # type: ignore[no-untyped-def]
        self,
        problem,
        optimization_method: str = "L-BFGS-B",
        penalty_weight: float = 1000.0,
        use_jax: bool | None = None,
        constraint_tolerance: float = 1e-6,
    ):
        """
        Initialize variational MFG solver.

        Args:
            problem: VariationalMFGProblem instance
            optimization_method: Scipy optimization method ('L-BFGS-B', 'CG', 'SLSQP')
            penalty_weight: Weight for constraint penalty terms
            use_jax: Use JAX for automatic differentiation (auto-detect if None)
            constraint_tolerance: Tolerance for constraint satisfaction
        """
        # Issue #1342: EXPERIMENTAL solver demoted for v1.0 — block construction with a
        # clear, actionable error BEFORE super().__init__ (which also reaches an
        # unimplemented problem.geometry path). The original implementation is preserved
        # below (currently unreachable) for future completion under #1342.
        raise _experimental_unavailable(
            "construction is intentionally blocked because the required solver hooks "
            "(compute_objective, compute_gradient, validate_solution) are not implemented"
        )
        super().__init__(problem)
        self.solver_name = "VariationalMFG"

        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required for VariationalMFGSolver")

        # Solver configuration
        self.optimization_method = optimization_method
        self.penalty_weight = penalty_weight
        self.constraint_tolerance = constraint_tolerance

        # JAX configuration
        self.use_jax = use_jax if use_jax is not None else JAX_AVAILABLE
        if self.use_jax and not JAX_AVAILABLE:
            logger.warning("JAX not available, falling back to finite differences")
            self.use_jax = False

        # Optimization state
        self.iteration_count = 0
        self.cost_history: list[float] = []
        self.constraint_violation_history: list[float] = []

        logger.info(f"Created {self.solver_name} solver")
        logger.info(f"  Optimization method: {optimization_method}")
        logger.info(f"  Penalty weight: {penalty_weight}")
        logger.info(f"  JAX acceleration: {self.use_jax}")

    # Issue #1342: abstract-method stubs. Overriding the abstract hooks makes the
    # class concrete so construction reaches the clear __init__ guard above instead
    # of the cryptic "Can't instantiate abstract class" TypeError. They remain
    # unimplemented (experimental) and raise the same actionable error.
    def compute_objective(self, variables: Any) -> float:
        raise _experimental_unavailable("the compute_objective hook is not implemented")

    def compute_gradient(self, variables: Any) -> Any:
        raise _experimental_unavailable("the compute_gradient hook is not implemented")

    def validate_solution(self) -> dict[str, float]:
        raise _experimental_unavailable("the validate_solution hook is not implemented")

    def solve(
        self,
        initial_guess: NDArray | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        **kwargs: Any,
    ) -> VariationalSolverResult:
        """
        Solve the variational MFG problem using direct optimization.

        Args:
            initial_guess: Initial density evolution (Nt+1, Nx_points)
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance for optimization
            verbose: Enable detailed output
            **kwargs: Additional optimization parameters

        Returns:
            VariationalSolverResult with optimal solution
        """
        start_time = time.time()

        if verbose:
            logger.info("Starting variational MFG optimization...")
            logger.info(f"  Max iterations: {max_iterations}")
            logger.info(f"  Tolerance: {tolerance}")

        # Create initial guess
        if initial_guess is None:
            initial_guess = self.create_initial_guess("gaussian")
            if verbose:
                logger.info("  Using Gaussian initial guess")

        # Validate initial guess: Nt+1 time points, Nx_points space points
        n_time_points = self.Nt + 1
        if initial_guess.shape != (n_time_points, self.Nx_points):
            raise ValueError(f"Initial guess must have shape ({n_time_points}, {self.Nx_points})")

        # Reset optimization state
        self.iteration_count = 0
        self.cost_history = []
        self.constraint_violation_history = []

        # Flatten initial guess for optimization
        x0 = initial_guess.flatten()

        # Set up optimization bounds (density must be non-negative)
        bounds = [(1e-10, None) for _ in range(len(x0))]

        # Optimization options
        options = {
            "maxiter": max_iterations,
            "ftol": tolerance,
            "gtol": tolerance,
            "disp": verbose,
            **kwargs,
        }

        if verbose:
            logger.info(f"  Optimization variables: {len(x0)}")
            logger.info(f"  Starting optimization with {self.optimization_method}...")

        try:
            # Run optimization
            result = minimize(
                fun=self._objective_function,
                x0=x0,
                method=self.optimization_method,
                bounds=bounds,
                jac=self._gradient_function if not self.use_jax else None,
                options=options,
                callback=self._optimization_callback if verbose else None,
            )

            # Extract optimal solution: Nt+1 time points, Nx_points space points
            n_time_points = self.Nt + 1
            optimal_density = result.x.reshape((n_time_points, self.Nx_points))

            # Compute optimal velocity field
            optimal_velocity = self._compute_velocity_from_density(optimal_density)

            # Compute representative trajectory
            representative_trajectory = self._compute_representative_trajectory(optimal_density)
            representative_velocity_traj = np.gradient(representative_trajectory, self.dt)

            # Final cost evaluation
            final_cost = self.evaluate_cost_functional(optimal_density, optimal_velocity)

            # Check constraints
            constraint_violations = self._evaluate_constraints(optimal_density)

            # Create result
            solve_time = time.time() - start_time

            variational_result = VariationalSolverResult(
                optimal_flow=optimal_density,
                representative_trajectory=representative_trajectory,
                representative_velocity=representative_velocity_traj,
                final_cost=final_cost,
                cost_history=self.cost_history.copy(),
                converged=result.success,
                num_iterations=self.iteration_count,
                constraint_violations=constraint_violations,
                solve_time=solve_time,
                solver_info={
                    "optimization_result": result,
                    "optimization_method": self.optimization_method,
                    "penalty_weight": self.penalty_weight,
                    "final_objective": result.fun,
                    # Backend compatibility - scipy.optimize result may not have jac (Issue #543 acceptable)
                    "gradient_norm": (np.linalg.norm(result.jac) if hasattr(result, "jac") else None),
                },
            )

            if verbose:
                logger.info(f"  Optimization completed in {solve_time:.2f}s")
                logger.info(f"  Converged: {result.success}")
                logger.info(f"  Final cost: {final_cost:.6e}")
                logger.info(f"  Iterations: {self.iteration_count}")

                # Log constraint violations
                max_violation = max(constraint_violations.values()) if constraint_violations else 0.0
                logger.info(f"  Max constraint violation: {max_violation:.2e}")

            return variational_result

        except Exception as e:
            logger.error(f"Optimization failed: {e}")

            # Return failure result
            solve_time = time.time() - start_time
            return VariationalSolverResult(
                final_cost=np.inf,
                converged=False,
                num_iterations=self.iteration_count,
                solve_time=solve_time,
                solver_info={"error": str(e)},
            )

    def _objective_function(self, x: NDArray) -> float:
        """
        Evaluate objective function (cost + penalty terms).

        Args:
            x: Flattened density evolution

        Returns:
            Objective function value
        """
        # Reshape to density field: Nt+1 time points, Nx_points space points
        n_time_points = self.Nt + 1
        density = x.reshape((n_time_points, self.Nx_points))

        # Compute velocity field
        velocity = self._compute_velocity_from_density(density)

        # Evaluate cost functional
        cost = self.evaluate_cost_functional(density, velocity)

        # Add penalty terms for constraints
        penalty = self._compute_penalty_terms(density)

        objective = cost + penalty

        # Store for history
        self.cost_history.append(cost)

        return objective

    def _gradient_function(self, x: NDArray) -> NDArray:
        """
        Compute gradient of objective function using finite differences.

        Args:
            x: Flattened density evolution

        Returns:
            Gradient vector
        """
        if self.use_jax:
            # JAX automatic differentiation would go here
            # For now, fall back to finite differences
            pass

        # Finite difference gradient
        eps = 1e-8
        grad = np.zeros_like(x)

        f0 = self._objective_function(x)

        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps

            f_plus = self._objective_function(x_plus)
            grad[i] = (f_plus - f0) / eps

        return grad

    def _compute_penalty_terms(self, density: NDArray) -> float:
        """
        Compute penalty terms for constraint violations.

        Args:
            density: Density field shape (Nt+1, Nx_points) - Nt+1 time points

        Returns:
            Total penalty value
        """
        penalty = 0.0
        n_time_points = self.Nt + 1

        # Mass conservation penalty
        for i in range(n_time_points):
            total_mass = trapezoid(density[i, :], x=self.x_grid)
            mass_violation = abs(total_mass - 1.0)
            penalty += self.penalty_weight * mass_violation**2

        # Non-negativity penalty (soft constraint)
        negative_values = np.minimum(density, 0.0)
        penalty += self.penalty_weight * np.sum(negative_values**2)

        # Continuity equation penalty (simplified)
        velocity = self._compute_velocity_from_density(density)
        continuity_residual = self.check_continuity_equation(density, velocity)
        penalty += 0.1 * self.penalty_weight * continuity_residual**2

        return penalty

    def _evaluate_constraints(self, density: NDArray) -> dict[str, float]:
        """
        Evaluate constraint violations.

        Args:
            density: Density field

        Returns:
            Dictionary with constraint violation measures
        """
        violations = {}
        n_time_points = self.Nt + 1

        # Mass conservation
        mass_errors = []
        for i in range(n_time_points):
            total_mass = trapezoid(density[i, :], x=self.x_grid)
            mass_errors.append(abs(total_mass - 1.0))
        violations["mass_conservation"] = max(mass_errors)

        # Non-negativity
        min_density = np.min(density)
        violations["non_negativity"] = abs(min(min_density, 0.0))

        # Continuity equation
        velocity = self._compute_velocity_from_density(density)
        violations["continuity_equation"] = self.check_continuity_equation(density, velocity)

        return violations

    def _compute_representative_trajectory(self, density: NDArray) -> NDArray:
        """
        Compute a representative trajectory from the optimal density.

        Uses the center of mass as a representative trajectory.

        Args:
            density: Optimal density field shape (Nt+1, Nx_points)

        Returns:
            Representative trajectory x(t) shape (Nt+1,) - one value per time point
        """
        n_time_points = self.Nt + 1
        trajectory = np.zeros(n_time_points)

        bounds = self.problem.geometry.get_bounds()
        xmin, xmax = bounds[0][0], bounds[1][0]

        for i in range(n_time_points):
            # Compute center of mass
            total_mass = trapezoid(density[i, :], x=self.x_grid)
            if total_mass > 1e-12:
                center_of_mass = trapezoid(self.x_grid * density[i, :], x=self.x_grid) / total_mass
                trajectory[i] = center_of_mass
            else:
                trajectory[i] = 0.5 * (xmin + xmax)

        return trajectory

    def _optimization_callback(self, x: NDArray) -> None:
        """
        Callback function called during optimization.

        Args:
            x: Current optimization variables
        """
        self.iteration_count += 1

        if self.iteration_count % 10 == 0:
            current_cost = self.cost_history[-1] if self.cost_history else np.inf
            logger.info(f"    Iteration {self.iteration_count}: cost = {current_cost:.6e}")

    def create_comparison_with_hamiltonian(self) -> dict[str, Any]:
        """
        Create equivalent Hamiltonian problem for comparison.

        Returns:
            Dictionary with converted Hamiltonian problem and solver
        """
        # Convert Lagrangian to Hamiltonian formulation
        hamiltonian_problem = self.problem.create_compatible_mfg_problem()

        # Create HJB-FP solver for comparison using problem.solve() API
        # Note: The solver is accessed via problem.solve() which creates the appropriate solver internally

        return {
            "hamiltonian_problem": hamiltonian_problem,
            "hamiltonian_solver": None,  # Use hamiltonian_problem.solve() instead
            "lagrangian_problem": self.problem,
            "lagrangian_solver": self,
        }

    def get_solver_info(self) -> dict[str, Any]:
        """Return detailed solver information."""
        base_info = super().get_solver_info()

        variational_info = {
            "optimization_method": self.optimization_method,
            "penalty_weight": self.penalty_weight,
            "constraint_tolerance": self.constraint_tolerance,
            "use_jax": self.use_jax,
            "current_iterations": self.iteration_count,
        }

        return {**base_info, **variational_info}

#!/usr/bin/env python3
"""
Primal-Dual Variational MFG Solver

This module implements primal-dual methods for solving constrained Lagrangian MFG problems.
Primal-dual methods are particularly effective for problems with equality and inequality constraints.

The solver addresses the constrained optimization problem:
min J[m,v] = ∫₀ᵀ ∫ L(t,x,v,m) m dxdt + ∫ g(x)m(T,x) dx

subject to:
- Continuity equation: ∂m/∂t + ∇·(mv) = σ²/2 Δm
- State constraints: c(t,x) ≤ 0
- Velocity constraints: h(t,x,v) ≤ 0
- Integral constraints: ∫ ψ(x,m) dx = constant

Key advantages of primal-dual methods:
- Better constraint handling than penalty methods
- Automatic dual variable updates
- Convergence guarantees for convex problems
- Separation of primal and dual subproblems
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.numerical.integration import trapezoid

from .base_variational import BaseVariationalSolver, VariationalSolverResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)
try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _experimental_unavailable(detail: str) -> NotImplementedError:
    """Build the Issue #1342 'experimental, not in this release' error."""
    return NotImplementedError(
        f"PrimalDualMFGSolver is EXPERIMENTAL and not available in this release "
        f"(Issue #1342): {detail}. Its solve logic exists but is unvalidated; "
        f"track #1342 for completion."
    )


class PrimalDualMFGSolver(BaseVariationalSolver):
    """
    Primal-dual solver for constrained Lagrangian MFG problems.

    **Experimental — not production-ready (Issue #1342).**

    Uses augmented Lagrangian method with automatic dual variable updates:
    L(m,v,λ,μ) = J[m,v] + ⟨λ, constraints⟩ + ρ/2 ||constraints||²

    The method alternates between:
    1. Primal step: minimize over (m,v) with fixed dual variables (λ,μ)
    2. Dual step: update dual variables based on constraint violations
    """

    def __init__(
        self,
        problem: Any,
        primal_solver: str = "L-BFGS-B",
        dual_update_method: str = "gradient_ascent",
        augmented_penalty: float = 10.0,
        dual_step_size: float = 0.1,
        use_adaptive_penalty: bool = True,
        constraint_tolerance: float = 1e-6,
    ) -> None:
        """
        Initialize primal-dual MFG solver.

        Args:
            problem: VariationalMFGProblem instance
            primal_solver: Optimization method for primal subproblem
            dual_update_method: Method for dual variable updates
            augmented_penalty: Initial penalty parameter ρ
            dual_step_size: Step size for dual updates
            use_adaptive_penalty: Automatically adjust penalty parameter
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
        self.solver_name = "PrimalDualMFG"

        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required for PrimalDualMFGSolver")

        # Solver configuration
        self.primal_solver = primal_solver
        self.dual_update_method = dual_update_method
        self.augmented_penalty = augmented_penalty
        self.dual_step_size = dual_step_size
        self.use_adaptive_penalty = use_adaptive_penalty
        self.constraint_tolerance = constraint_tolerance

        # Dual variables (Lagrange multipliers)
        self.dual_vars: dict[str, Any] = {}
        self._initialize_dual_variables()

        # Algorithm state
        self.primal_dual_history: list[dict[str, float]] = []
        self.constraint_violation_history: list[float] = []
        self.penalty_history = [augmented_penalty]

        logger.info(f"Created {self.solver_name} solver")
        logger.info(f"  Primal solver: {primal_solver}")
        logger.info(f"  Dual update: {dual_update_method}")
        logger.info(f"  Initial penalty: {augmented_penalty}")
        logger.info(f"  Adaptive penalty: {use_adaptive_penalty}")

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

    def solve(  # type: ignore[override]
        self,
        initial_guess: NDArray | None = None,
        max_outer_iterations: int = 20,
        max_inner_iterations: int = 50,
        tolerance: float = 1e-6,
        verbose: bool = True,
        **kwargs: Any,
    ) -> VariationalSolverResult:
        """
        Solve using primal-dual method.

        Args:
            initial_guess: Initial density evolution
            max_outer_iterations: Maximum primal-dual iterations
            max_inner_iterations: Maximum iterations per primal subproblem
            tolerance: Convergence tolerance
            verbose: Enable detailed output
            **kwargs: Additional solver parameters

        Returns:
            VariationalSolverResult with optimal solution
        """
        start_time = time.time()

        if verbose:
            logger.info("Starting primal-dual MFG optimization...")
            logger.info(f"  Max outer iterations: {max_outer_iterations}")
            logger.info(f"  Max inner iterations: {max_inner_iterations}")
            logger.info(f"  Tolerance: {tolerance}")

        # Create initial guess
        if initial_guess is None:
            initial_guess = self.create_initial_guess("gaussian")
            if verbose:
                logger.info("  Using Gaussian initial guess")

        # Initialize algorithm state
        current_density = initial_guess.copy()
        self.primal_dual_history = []
        self.constraint_violation_history = []

        converged = False
        outer_iteration = 0

        # Main primal-dual iteration
        for outer_iteration in range(max_outer_iterations):
            if verbose:
                logger.info(f"\n--- Primal-Dual Iteration {outer_iteration + 1} ---")

            # Store previous solution for convergence check
            prev_density = current_density.copy()

            # PRIMAL STEP: Solve augmented Lagrangian subproblem
            primal_result = self._solve_primal_subproblem(
                current_density,
                max_iterations=max_inner_iterations,
                tolerance=tolerance * 10,  # Looser tolerance for inner problem
                verbose=verbose,
            )

            if primal_result is not None:
                current_density = primal_result
            else:
                logger.warning(f"  Primal subproblem failed at iteration {outer_iteration + 1}")
                break

            # Evaluate constraints
            constraint_violations = self._evaluate_all_constraints(current_density)
            max_violation = max(constraint_violations.values()) if constraint_violations else 0.0

            # DUAL STEP: Update dual variables
            self._update_dual_variables(constraint_violations)

            # PENALTY UPDATE: Adapt penalty parameter if needed
            if self.use_adaptive_penalty:
                self._update_penalty_parameter(constraint_violations)

            # Store iteration history
            iteration_info = {
                "outer_iteration": outer_iteration + 1,
                "max_constraint_violation": max_violation,
                "augmented_penalty": self.augmented_penalty,
                "dual_vars": {k: np.linalg.norm(v) for k, v in self.dual_vars.items()},
                "primal_change": np.linalg.norm(current_density - prev_density),
            }
            self.primal_dual_history.append(iteration_info)  # type: ignore[arg-type]
            self.constraint_violation_history.append(max_violation)

            if verbose:
                logger.info(f"  Max constraint violation: {max_violation:.2e}")
                logger.info(f"  Primal change: {iteration_info['primal_change']:.2e}")
                logger.info(f"  Current penalty: {self.augmented_penalty:.1f}")

            # Convergence check
            if max_violation < self.constraint_tolerance and float(iteration_info["primal_change"]) < tolerance:  # type: ignore[arg-type]
                converged = True
                if verbose:
                    logger.info(f"  ✓ Converged at iteration {outer_iteration + 1}")
                break

        # Compute final solution properties
        # Note: Simplified implementation - methods not yet implemented
        # optimal_velocity = None  # Would be computed from current_density
        representative_trajectory = None  # Would be computed from current_density
        representative_velocity_traj = None  # Would be gradient of trajectory
        final_cost = 0.0  # Would evaluate cost functional

        # Create result
        solve_time = time.time() - start_time

        result = VariationalSolverResult(
            optimal_flow=current_density,
            representative_trajectory=representative_trajectory,
            representative_velocity=representative_velocity_traj,
            final_cost=final_cost,
            cost_history=[info["max_constraint_violation"] for info in self.primal_dual_history],
            converged=converged,
            num_iterations=outer_iteration + 1,
            constraint_violations=constraint_violations if converged else {},
            solve_time=solve_time,
            solver_info={
                "primal_dual_history": self.primal_dual_history,
                "final_penalty": self.augmented_penalty,
                "dual_variables": self.dual_vars.copy(),
                "constraint_violation_history": self.constraint_violation_history,
                "penalty_history": self.penalty_history.copy(),
            },
        )

        if verbose:
            logger.info(f"\nPrimal-dual optimization completed in {solve_time:.2f}s")
            logger.info(f"  Converged: {converged}")
            logger.info(f"  Final cost: {final_cost:.6e}")
            logger.info(f"  Outer iterations: {outer_iteration + 1}")
            if converged:
                logger.info(f"  Final constraint violation: {max_violation:.2e}")

        return result

    def _initialize_dual_variables(self):
        """Initialize dual variables (Lagrange multipliers)."""
        # Dual variables for continuity equation (one per grid point per time step)
        self.dual_vars["continuity"] = np.zeros((self.Nt - 1, self.Nx - 1))

        # Dual variables for mass conservation (one per time step)
        self.dual_vars["mass_conservation"] = np.zeros(self.Nt)

        # Dual variables for state constraints (if any)
        # Problem API: use getattr instead of hasattr (Issue #543 fix)
        state_constraints = getattr(self.problem.components, "state_constraints", None)
        if state_constraints:
            self.dual_vars["state_constraints"] = np.zeros((self.Nt, self.Nx + 1))

        # Dual variables for velocity constraints (if any)
        velocity_constraints = getattr(self.problem.components, "velocity_constraints", None)
        if velocity_constraints:
            self.dual_vars["velocity_constraints"] = np.zeros((self.Nt, self.Nx + 1))

        # Dual variables for integral constraints (if any)
        integral_constraints = getattr(self.problem.components, "integral_constraints", None)
        if integral_constraints:
            num_integral_constraints = len(integral_constraints)
            self.dual_vars["integral_constraints"] = np.zeros(num_integral_constraints)

        logger.info("Initialized dual variables:")
        for name, dual_var in self.dual_vars.items():
            logger.info(f"  {name}: shape {dual_var.shape}")

    def _solve_primal_subproblem(
        self,
        current_density: NDArray,
        max_iterations: int,
        tolerance: float,
        verbose: bool,
    ) -> NDArray | None:
        """
        Solve the primal subproblem with fixed dual variables.

        Minimizes the augmented Lagrangian:
        L_ρ(m,v) = J[m,v] + ⟨λ, constraints⟩ + ρ/2 ||constraints||²
        """
        if verbose:
            logger.info("  Solving primal subproblem...")

        # Flatten density for optimization
        x0 = current_density.flatten()

        # Set up bounds (density must be non-negative)
        bounds = [(1e-10, None) for _ in range(len(x0))]

        # Optimization options
        options = {
            "maxiter": max_iterations,
            "ftol": tolerance,
            "gtol": tolerance,
            "disp": False,
        }

        try:
            # Solve augmented Lagrangian subproblem
            result = minimize(
                fun=self._augmented_lagrangian_objective,
                x0=x0,
                method=self.primal_solver,
                bounds=bounds,
                options=options,
            )

            if result.success:
                optimal_density = result.x.reshape((self.Nt, self.Nx + 1))
                if verbose:
                    logger.info(f"    ✓ Primal subproblem converged in {result.nit} iterations")
                return optimal_density
            else:
                if verbose:
                    logger.warning(f"    ⚠ Primal subproblem did not converge: {result.message}")
                return None

        except Exception as e:
            logger.error(f"    ✗ Primal subproblem failed: {e}")
            return None

    def _augmented_lagrangian_objective(self, x: NDArray) -> float:
        """
        Evaluate augmented Lagrangian objective function.

        L_ρ(m) = J[m] + ⟨λ, constraints⟩ + ρ/2 ||constraints||²
        """
        # Reshape to density field
        density = x.reshape((self.Nt, self.Nx + 1))

        # Compute velocity field
        velocity = self._compute_velocity_from_density(density)

        # Original cost functional
        cost = self.evaluate_cost_functional(density, velocity)

        # Constraint evaluations
        constraints = self._evaluate_all_constraints(density)

        # Augmented Lagrangian terms
        augmented_terms = 0.0

        # Continuity equation terms
        continuity_residual = self.check_continuity_equation(density, velocity)
        if "continuity" in self.dual_vars:
            # Simplified: use scalar dual variable for continuity
            dual_continuity = np.mean(self.dual_vars["continuity"])
            augmented_terms += dual_continuity * continuity_residual
            augmented_terms += 0.5 * self.augmented_penalty * continuity_residual**2

        # Mass conservation terms
        for i in range(self.Nt):
            mass_violation = constraints.get("mass_conservation", 0.0)
            if "mass_conservation" in self.dual_vars:
                augmented_terms += self.dual_vars["mass_conservation"][i] * mass_violation
                augmented_terms += 0.5 * self.augmented_penalty * mass_violation**2

        # State constraint terms
        if "state_constraints" in constraints and "state_constraints" in self.dual_vars:
            state_violation = constraints["state_constraints"]
            dual_state = np.mean(self.dual_vars["state_constraints"])
            augmented_terms += dual_state * state_violation
            augmented_terms += 0.5 * self.augmented_penalty * state_violation**2

        return cost + augmented_terms

    def _evaluate_all_constraints(self, density: NDArray) -> dict[str, float]:
        """Evaluate all constraint violations."""
        constraints = {}

        # Mass conservation
        mass_errors = []
        for i in range(self.Nt):
            total_mass = trapezoid(density[i, :], x=self.x_grid)
            mass_errors.append(abs(total_mass - 1.0))
        constraints["mass_conservation"] = max(mass_errors)

        # Non-negativity
        min_density = np.min(density)
        constraints["non_negativity"] = abs(min(min_density, 0.0))

        # Continuity equation
        # Note: _compute_velocity_from_density not yet implemented
        velocity = None  # Would be computed from density
        constraints["continuity_equation"] = 0.0  # Would check continuity equation

        # State constraints (if any)
        # Problem API: use getattr instead of hasattr (Issue #543 fix)
        state_constraints = getattr(self.problem.components, "state_constraints", None)
        if state_constraints:
            max_state_violation = 0.0
            for constraint_func in state_constraints:
                for i in range(self.Nt):
                    for x in self.x_grid:
                        t = self.t_grid[i]
                        violation = max(0.0, constraint_func(t, x))
                        max_state_violation = max(max_state_violation, violation)
            constraints["state_constraints"] = max_state_violation

        # Velocity constraints (if any)
        velocity_constraints = getattr(self.problem.components, "velocity_constraints", None)
        if velocity_constraints and velocity is not None:
            max_velocity_violation = 0.0
            for constraint_func in velocity_constraints:
                for i in range(self.Nt):
                    for j, x in enumerate(self.x_grid):
                        t = self.t_grid[i]
                        v = velocity[i, j]
                        violation = max(0.0, constraint_func(t, x, v))
                        max_velocity_violation = max(max_velocity_violation, violation)
            constraints["velocity_constraints"] = max_velocity_violation
        else:
            constraints["velocity_constraints"] = 0.0

        return constraints

    def _update_dual_variables(self, constraint_violations: dict[str, float]) -> None:
        """Update dual variables based on constraint violations."""
        if self.dual_update_method == "gradient_ascent":
            # Gradient ascent: λ ← λ + α * constraints

            # Update mass conservation dual variables
            if "mass_conservation" in self.dual_vars:
                mass_violation = constraint_violations.get("mass_conservation", 0.0)
                self.dual_vars["mass_conservation"] += self.dual_step_size * mass_violation

            # Update continuity dual variables
            if "continuity" in self.dual_vars:
                continuity_violation = constraint_violations.get("continuity_equation", 0.0)
                self.dual_vars["continuity"] += self.dual_step_size * continuity_violation

            # Update state constraint dual variables
            if "state_constraints" in self.dual_vars and "state_constraints" in constraint_violations:
                state_violation = constraint_violations["state_constraints"]
                self.dual_vars["state_constraints"] += self.dual_step_size * state_violation

            # Update velocity constraint dual variables
            if "velocity_constraints" in self.dual_vars and "velocity_constraints" in constraint_violations:
                velocity_violation = constraint_violations["velocity_constraints"]
                self.dual_vars["velocity_constraints"] += self.dual_step_size * velocity_violation

        elif self.dual_update_method == "multiplier_method":
            # Classic multiplier method with exact penalty update
            # This would require more sophisticated implementation
            pass

    def _update_penalty_parameter(self, constraint_violations: dict[str, float]) -> None:
        """Adapt penalty parameter based on constraint violation progress."""
        if not self.use_adaptive_penalty:
            return

        max_violation = max(constraint_violations.values()) if constraint_violations else 0.0

        # If constraints are not improving, increase penalty
        if len(self.constraint_violation_history) >= 2:
            prev_violation = self.constraint_violation_history[-2]

            if max_violation >= 0.9 * prev_violation:  # Less than 10% improvement
                self.augmented_penalty *= 2.0
                logger.info(f"    Increased penalty to {self.augmented_penalty:.1f}")
            elif max_violation < 0.1 * prev_violation:  # Significant improvement
                self.augmented_penalty = max(self.augmented_penalty * 0.8, 1.0)

        self.penalty_history.append(self.augmented_penalty)

    def get_solver_info(self) -> dict[str, Any]:
        """Return detailed solver information."""
        base_info = super().get_solver_info()

        primal_dual_info = {
            "primal_solver": self.primal_solver,
            "dual_update_method": self.dual_update_method,
            "augmented_penalty": self.augmented_penalty,
            "dual_step_size": self.dual_step_size,
            "use_adaptive_penalty": self.use_adaptive_penalty,
            "constraint_tolerance": self.constraint_tolerance,
            "dual_variable_shapes": {name: dual_var.shape for name, dual_var in self.dual_vars.items()},
        }

        return {**base_info, **primal_dual_info}

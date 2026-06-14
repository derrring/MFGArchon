"""
Sinkhorn Algorithm-Based MFG Solver.

This module implements entropic regularized optimal transport for MFG problems
using the Sinkhorn algorithm. This approach provides fast, scalable solutions
for large-scale MFG problems through efficient iterative algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.alg.base_solver import BaseOptimizationSolver
from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.core.mfg_problem import MFGProblem

logger = get_logger(__name__)


@dataclass
class SinkhornSolverConfig:
    """Configuration for Sinkhorn MFG solver."""

    # Sinkhorn algorithm parameters
    regularization: float = 0.01  # Entropic regularization parameter
    max_sinkhorn_iterations: int = 1000
    sinkhorn_tolerance: float = 1e-8

    # MFG optimization parameters
    max_mfg_iterations: int = 500
    mfg_tolerance: float = 1e-6
    step_size: float = 0.1

    # Discretization
    num_time_steps: int = 50
    num_spatial_points: int = 100

    # Adaptive parameters
    adaptive_regularization: bool = True
    min_regularization: float = 1e-4
    regularization_decay: float = 0.95

    # Stabilization
    log_domain: bool = True  # Use log-domain Sinkhorn for numerical stability
    stabilization_threshold: float = 1e2


@dataclass
class SinkhornSolverResult:
    """Result container for Sinkhorn MFG solver."""

    # Optimal solution
    optimal_densities: NDArray | None = None
    optimal_potentials: NDArray | None = None  # Dual potentials (u, v)
    transport_matrices: list[NDArray] | None = None

    # Algorithm performance
    final_cost: float = np.inf
    regularization_path: list[float] | None = None
    sinkhorn_iterations: list[int] | None = None  # Sinkhorn iterations per MFG step
    cost_history: list[float] | None = None

    # Convergence information
    converged: bool = False
    num_mfg_iterations: int = 0
    total_sinkhorn_iterations: int = 0

    # Solution quality
    mass_conservation_error: float = 0.0
    entropic_cost: float = 0.0


def _experimental_unavailable(detail: str) -> NotImplementedError:
    """Build the Issue #1342 'experimental, not in this release' error."""
    return NotImplementedError(
        f"SinkhornMFGSolver is EXPERIMENTAL and not available in this release "
        f"(Issue #1342): {detail}. Its solve logic exists but is unvalidated; "
        f"track #1342 for completion."
    )


class SinkhornMFGSolver(BaseOptimizationSolver):
    """
    Sinkhorn algorithm-based solver for Mean Field Games.

    **Experimental — not production-ready (Issue #1342).**

    This solver uses entropic regularized optimal transport with the Sinkhorn
    algorithm to efficiently solve MFG problems. The entropic regularization
    allows for fast, parallelizable computation while maintaining the
    geometric structure of optimal transport.

    Mathematical Framework:
    - Regularized transport: min_{π} ⟨C,π⟩ + ε⋅H(π)
    - Sinkhorn iterations: u^(k+1) = a / (K^T v^(k)), v^(k+1) = b / (K u^(k+1))
    - Transport matrix: π = diag(u) K diag(v)
    - MFG coupling: Density evolution via regularized transport

    Key Features:
    - Fast O(n²) complexity per iteration
    - Numerical stability through log-domain arithmetic
    - Adaptive regularization scheduling
    - GPU-friendly parallelizable operations
    """

    def __init__(
        self,
        problem: MFGProblem,
        config: SinkhornSolverConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize Sinkhorn MFG solver."""
        # Issue #1342: EXPERIMENTAL solver demoted for v1.0 — block construction with a
        # clear, actionable error. The original setup/solve implementation is preserved
        # below (currently unreachable) for future completion under #1342.
        raise _experimental_unavailable(
            "construction is intentionally blocked because the required solver hooks "
            "(compute_objective, compute_gradient, validate_solution) are not implemented"
        )
        super().__init__(problem, **kwargs)
        self.config = config or SinkhornSolverConfig()
        self.logger = get_logger(__name__)

        # Validate dependencies
        self._check_dependencies()

        # Setup discretization and transport kernel
        self._setup_discretization()
        self._setup_transport_kernel()

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

    def _check_dependencies(self) -> None:
        """Check required dependencies."""
        try:
            import ot

            self.ot = ot
        except ImportError as e:
            raise ImportError(
                f"Sinkhorn solver requires POT library: {e}. Install with: pip install mfgarchon[optimization]"
            ) from e

    def _setup_discretization(self) -> None:
        """Setup temporal and spatial discretization."""
        self.time_grid = np.linspace(0, self.problem.T, self.config.num_time_steps)
        self.dt = self.time_grid[1] - self.time_grid[0]

        bounds = self.problem.geometry.get_bounds()
        xmin, xmax = bounds[0][0], bounds[1][0]
        self.spatial_grid = np.linspace(xmin, xmax, self.config.num_spatial_points)
        self.dx = self.spatial_grid[1] - self.spatial_grid[0]

        self.logger.info(f"Sinkhorn discretization: {len(self.time_grid)}×{len(self.spatial_grid)} grid")

    def _setup_transport_kernel(self) -> None:
        """Setup Gibbs kernel for Sinkhorn iterations."""
        # Ground cost matrix
        x, y = np.meshgrid(self.spatial_grid, self.spatial_grid)
        self.cost_matrix = (x - y) ** 2  # Squared Euclidean distance

        # Pre-compute Gibbs kernel for efficiency
        self.current_regularization = self.config.regularization
        self._update_gibbs_kernel()

    def _update_gibbs_kernel(self) -> None:
        """Update Gibbs kernel with current regularization parameter."""
        if self.config.log_domain:
            # Log-domain kernel for numerical stability
            self.log_kernel = -self.cost_matrix / self.current_regularization
            # Normalize to prevent overflow
            self.log_kernel -= np.max(self.log_kernel)
        else:
            # Standard kernel
            self.kernel = np.exp(-self.cost_matrix / self.current_regularization)

    def solve(self) -> SinkhornSolverResult:
        """Solve MFG problem using Sinkhorn algorithm."""
        self.logger.info("Starting Sinkhorn MFG solver")

        result = SinkhornSolverResult()

        try:
            # Initialize densities and potentials
            densities = self._initialize_densities()
            potentials_u = np.zeros((len(self.time_grid), len(self.spatial_grid)))
            potentials_v = np.zeros((len(self.time_grid), len(self.spatial_grid)))

            # Algorithm tracking
            cost_history = []
            regularization_path = [self.current_regularization]
            sinkhorn_iterations_history = []

            for mfg_iter in range(self.config.max_mfg_iterations):
                self.logger.debug(f"MFG iteration {mfg_iter + 1}/{self.config.max_mfg_iterations}")

                # Solve transport sequence with current densities
                transport_matrices, sinkhorn_iters = self._solve_transport_sequence(densities)
                sinkhorn_iterations_history.extend(sinkhorn_iters)

                # Update densities using transport-based dynamics
                new_densities = self._update_densities_sinkhorn(densities, transport_matrices)

                # Update dual potentials
                potentials_u, potentials_v = self._update_potentials(transport_matrices)

                # Compute current cost
                current_cost = self._compute_mfg_cost(new_densities, transport_matrices)
                cost_history.append(current_cost)

                # Adaptive regularization
                if self.config.adaptive_regularization:
                    self._update_regularization(mfg_iter)
                    regularization_path.append(self.current_regularization)

                # Check convergence
                if self._check_mfg_convergence(densities, new_densities, cost_history):
                    self.logger.info(f"Sinkhorn MFG converged after {mfg_iter + 1} iterations")
                    result.converged = True
                    break

                densities = new_densities

                # Progress logging
                if (mfg_iter + 1) % 25 == 0:
                    avg_sinkhorn = np.mean(sinkhorn_iters) if sinkhorn_iters else 0
                    self.logger.info(
                        f"  MFG iter {mfg_iter + 1}: Cost={current_cost:.6e}, "
                        f"Reg={self.current_regularization:.6e}, Avg Sinkhorn={avg_sinkhorn:.1f}"
                    )

            else:
                self.logger.warning(f"Did not converge after {self.config.max_mfg_iterations} MFG iterations")

            # Finalize results
            result.optimal_densities = densities
            result.optimal_potentials = (potentials_u, potentials_v)
            result.transport_matrices = transport_matrices
            result.final_cost = cost_history[-1] if cost_history else np.inf
            result.cost_history = cost_history
            result.regularization_path = regularization_path
            result.sinkhorn_iterations = sinkhorn_iterations_history
            result.num_mfg_iterations = len(cost_history)
            result.total_sinkhorn_iterations = sum(sinkhorn_iterations_history)

            # Compute solution quality metrics
            result.mass_conservation_error = self._compute_mass_conservation_error(densities)
            result.entropic_cost = self._compute_entropic_cost(transport_matrices)

            self.logger.info(
                f"Sinkhorn solver completed: Cost={result.final_cost:.6e}, "
                f"Total Sinkhorn iters={result.total_sinkhorn_iterations}"
            )

        except Exception as e:
            self.logger.error(f"Sinkhorn solver failed: {e}")
            raise

        return result

    def _initialize_densities(self) -> NDArray:
        """Initialize density evolution."""
        densities = np.zeros((len(self.time_grid), len(self.spatial_grid)))

        bounds = self.problem.geometry.get_bounds()
        xmin, xmax = bounds[0][0], bounds[1][0]

        # Initial condition
        # Problem API: use getattr instead of hasattr (Issue #543 fix)
        initial_density = getattr(self.problem, "initial_density", None)
        if initial_density is not None:
            for i, x in enumerate(self.spatial_grid):
                densities[0, i] = initial_density(x)
        else:
            # Default Gaussian
            center = (xmin + xmax) / 2
            width = (xmax - xmin) / 4
            densities[0, :] = np.exp(-((self.spatial_grid - center) ** 2) / (2 * width**2))

        # Normalize
        densities[0, :] /= np.trapezoid(densities[0, :], self.spatial_grid)

        return densities

    def _solve_transport_sequence(self, densities: NDArray) -> tuple[list[NDArray], list[int]]:
        """Solve optimal transport for each consecutive time pair."""
        transport_matrices = []
        sinkhorn_iterations = []

        for t in range(len(self.time_grid) - 1):
            source = np.maximum(densities[t, :], 1e-12)
            target = np.maximum(densities[t + 1, :], 1e-12) if t + 1 < len(densities) else source

            # Normalize masses
            source /= np.sum(source)
            target /= np.sum(target)

            # Solve transport with Sinkhorn
            if self.config.log_domain:
                transport_matrix, n_iter = self._sinkhorn_log_domain(source, target)
            else:
                transport_matrix, n_iter = self._sinkhorn_standard(source, target)

            transport_matrices.append(transport_matrix)
            sinkhorn_iterations.append(n_iter)

        return transport_matrices, sinkhorn_iterations

    def _sinkhorn_log_domain(self, a: NDArray, b: NDArray) -> tuple[NDArray, int]:
        """Log-domain Sinkhorn for numerical stability."""
        log_a = np.log(a)
        log_b = np.log(b)

        # Initialize potentials
        u = np.zeros_like(a)
        v = np.zeros_like(b)

        for _iteration in range(self.config.max_sinkhorn_iterations):
            u_old = u.copy()

            # Sinkhorn updates in log domain
            lse_1 = self._log_sum_exp(self.log_kernel + v[np.newaxis, :], axis=1)
            u = log_a - lse_1

            lse_2 = self._log_sum_exp(self.log_kernel + u[:, np.newaxis], axis=0)
            v = log_b - lse_2

            # Check convergence
            if np.max(np.abs(u - u_old)) < self.config.sinkhorn_tolerance:
                break

            # Stabilization
            if _iteration % 10 == 0 and np.max(np.abs(u)) > self.config.stabilization_threshold:
                u -= np.mean(u)
                v -= np.mean(v)

        # Compute transport matrix
        transport_matrix = np.exp(u[:, np.newaxis] + self.log_kernel + v[np.newaxis, :])

        return transport_matrix, _iteration + 1

    def _sinkhorn_standard(self, a: NDArray, b: NDArray) -> tuple[NDArray, int]:
        """Standard Sinkhorn algorithm."""
        u = np.ones_like(a)
        v = np.ones_like(b)

        for _iteration in range(self.config.max_sinkhorn_iterations):
            u_old = u.copy()

            # Sinkhorn updates
            u = a / (self.kernel @ v)
            v = b / (self.kernel.T @ u)

            # Check convergence
            if np.max(np.abs(u - u_old)) < self.config.sinkhorn_tolerance:
                break

        # Transport matrix
        transport_matrix = np.diag(u) @ self.kernel @ np.diag(v)

        return transport_matrix, _iteration + 1

    def _log_sum_exp(self, x: NDArray, axis: int) -> NDArray:
        """Numerically stable log-sum-exp computation."""
        x_max = np.max(x, axis=axis, keepdims=True)
        return x_max.squeeze() + np.log(np.sum(np.exp(x - x_max), axis=axis))

    def _update_densities_sinkhorn(self, densities: NDArray, transport_matrices: list[NDArray]) -> NDArray:
        """Update density evolution using Sinkhorn transport."""
        new_densities = densities.copy()

        for t in range(1, len(self.time_grid)):
            if t - 1 < len(transport_matrices):
                # Forward transport
                new_densities[t, :] = transport_matrices[t - 1].T @ densities[t - 1, :]

        return new_densities

    def _update_potentials(self, transport_matrices: list[NDArray]) -> tuple[NDArray, NDArray]:
        """Update dual potentials from transport matrices."""
        n_t, n_x = len(self.time_grid), len(self.spatial_grid)
        potentials_u = np.zeros((n_t, n_x))
        potentials_v = np.zeros((n_t, n_x))

        # Extract potentials from transport structure (simplified)
        for t, transport in enumerate(transport_matrices):
            if t + 1 < n_t:
                # Approximate potentials from transport marginals
                marginal_1 = np.sum(transport, axis=1)
                marginal_2 = np.sum(transport, axis=0)

                potentials_u[t + 1, :] = -self.current_regularization * np.log(marginal_1 + 1e-12)
                potentials_v[t + 1, :] = -self.current_regularization * np.log(marginal_2 + 1e-12)

        return potentials_u, potentials_v

    def _compute_mfg_cost(self, densities: NDArray, transport_matrices: list[NDArray]) -> float:
        """Compute total MFG cost."""
        cost = 0.0

        # Transport cost
        for _t, transport in enumerate(transport_matrices):
            cost += np.sum(transport * self.cost_matrix) * self.dt

        # Entropic cost
        cost += self.current_regularization * self._compute_entropic_cost(transport_matrices)

        return cost

    def _compute_entropic_cost(self, transport_matrices: list[NDArray]) -> float:
        """Compute entropic regularization cost."""
        entropy = 0.0
        for transport in transport_matrices:
            # Entropy: -sum(π log π)
            log_transport = np.log(transport + 1e-12)
            entropy -= np.sum(transport * log_transport) * self.dt
        return entropy

    def _update_regularization(self, iteration: int) -> None:
        """Update regularization parameter adaptively."""
        if iteration > 0:
            self.current_regularization = max(
                self.config.min_regularization,
                self.current_regularization * self.config.regularization_decay,
            )
            self._update_gibbs_kernel()

    def _check_mfg_convergence(self, old_densities: NDArray, new_densities: NDArray, cost_history: list[float]) -> bool:
        """Check MFG convergence."""
        if len(cost_history) < 2:
            return False

        # Density change
        density_change = np.linalg.norm(new_densities - old_densities)

        # Cost change
        cost_change = abs(cost_history[-1] - cost_history[-2]) if len(cost_history) > 1 else float("inf")

        return density_change < self.config.mfg_tolerance and cost_change < self.config.mfg_tolerance

    def _compute_mass_conservation_error(self, densities: NDArray) -> float:
        """Compute mass conservation error."""
        total_masses = [np.trapezoid(density, self.spatial_grid) for density in densities]
        return np.std(total_masses)

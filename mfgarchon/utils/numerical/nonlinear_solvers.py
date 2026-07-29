"""
Fundamental Nonlinear System Solvers.

This module provides generic, reusable solvers for nonlinear systems that
appear throughout MFG algorithms. These are the building blocks for value
iteration, policy iteration, and Newton-based methods.

Control Theory Context:
    - FixedPointSolver: Value iteration
    - NewtonSolver: Newton-Raphson dynamic programming
    - PolicyIterationSolver: Howard's policy iteration algorithm

All solvers are dimension-agnostic and preserve array shapes.

References:
    - Bertsekas (2012): Dynamic Programming and Optimal Control
    - Nocedal & Wright (2006): Numerical Optimization
    - Walker & Ni (2011): Anderson acceleration for fixed-point iterations
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

from mfgarchon.utils.deprecation import deprecated_parameter
from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)
# Try to import JAX autodiff utilities
_JAX_AVAILABLE = False
try:
    from mfgarchon.utils.acceleration import HAS_JAX
    from mfgarchon.utils.acceleration.jax_utils import compute_jacobian

    _JAX_AVAILABLE = HAS_JAX
except ImportError:
    logger.debug("JAX acceleration utilities not available")


class SolverInfo:
    """
    Container for solver convergence information.

    Attributes:
        converged: Whether solver converged within tolerance
        iterations: Number of iterations performed
        residual: Final residual norm
        residual_history: Residual at each iteration
        solver_time: Total time in seconds (if tracked)
        extra: Additional solver-specific information
    """

    def __init__(
        self,
        converged: bool,
        iterations: int,
        residual: float,
        residual_history: list[float],
        solver_time: float = 0.0,
        **extra: Any,
    ):
        self.converged = converged
        self.iterations = iterations
        self.residual = residual
        self.residual_history = residual_history
        self.solver_time = solver_time
        self.extra = extra

    def __repr__(self) -> str:
        status = "converged" if self.converged else "not converged"
        return f"SolverInfo({status}, iterations={self.iterations}, residual={self.residual:.2e})"


class NonlinearSolver(ABC):
    """
    Abstract base class for nonlinear system solvers.

    All solvers follow the same interface:
        x_solution, info = solver.solve(F_or_G, x0)

    where F_or_G is either:
        - F: x → residual (for Newton: solve F(x) = 0)
        - G: x → next_x (for fixed-point: solve x = G(x))

    Shape preservation: Output x has same shape as input x0.
    """

    @abstractmethod
    def solve(
        self,
        func: Callable[[NDArray], NDArray],
        x0: NDArray,
        **kwargs: Any,
    ) -> tuple[NDArray, SolverInfo]:
        """
        Solve nonlinear system.

        Args:
            func: Function defining the system
            x0: Initial guess (any shape)
            **kwargs: Solver-specific options

        Returns:
            x: Solution (same shape as x0)
            info: Convergence information
        """


class FixedPointSolver(NonlinearSolver):
    """
    Damped fixed-point iteration solver.

    Solves: x = G(x) via iteration
        x^{k+1} = (1-ω)x^k + ω·G(x^k)

    where ω ∈ (0,1] is the damping factor:
        - ω = 1.0: Full update (no damping)
        - ω < 1.0: Damped update (more stable, slower)

    Control theory: This is value iteration with relaxation.

    Example:
        >>> G = lambda x: np.cos(x)  # Solve x = cos(x)
        >>> solver = FixedPointSolver(relaxation=0.8, tol=1e-6)
        >>> x, info = solver.solve(G, x0=0.5)
        >>> print(f"Solution: {x}, converged: {info.converged}")

    Notes:
        - Works for any array shape (scalars, vectors, matrices, tensors)
        - Damping recommended for stability (ω ≈ 0.5-0.8)
        - Use Anderson acceleration for faster convergence
    """

    @deprecated_parameter(param_name="damping_factor", since="v0.19.2", replacement="relaxation")
    def __init__(
        self,
        relaxation: float = 1.0,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        norm_type: Literal["relative", "absolute"] = "relative",
        # Legacy kwarg (deprecated since v0.19.2, removal v0.25.0)
        damping_factor: float | None = None,
    ):
        """
        Initialize fixed-point solver.

        Args:
            relaxation: Under-relaxation factor ω ∈ (0, 1].
                ω = 1.0 applies the full update; ω < 1 averages new and old iterates.
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            norm_type: 'relative' or 'absolute' residual norm

            Legacy `damping_factor` kwarg still accepted with DeprecationWarning.
        """
        if damping_factor is not None:
            relaxation = damping_factor
        if not 0 < relaxation <= 1.0:
            raise ValueError(f"relaxation must be in (0,1], got {relaxation}")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be > 0, got {tolerance}")

        self.relaxation = relaxation
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.norm_type = norm_type

    @property
    def damping_factor(self) -> float:
        """Deprecated alias for `relaxation` (v0.19.2+). Removal in v0.25.0."""
        return self.relaxation

    def solve(  # type: ignore[override]
        self,
        G: Callable[[NDArray], NDArray],
        x0: NDArray,
        **kwargs: Any,
    ) -> tuple[NDArray, SolverInfo]:
        """
        Solve fixed-point problem x = G(x).

        Args:
            G: Fixed-point map G: x → x
            x0: Initial guess (any shape)

        Returns:
            x: Solution (same shape as x0)
            info: Convergence information

        Note:
            Parameter name 'G' intentionally differs from base class 'func'
            for better API clarity in fixed-point context.
        """
        import time

        start_time = time.time()
        # Convert to numpy array to handle scalars
        x_current = np.asarray(x0, dtype=float).copy()
        original_shape = np.asarray(x0).shape
        is_scalar = original_shape == ()

        residual_history = []

        for iteration in range(self.max_iterations):
            # Evaluate fixed-point map
            x_new = G(x_current)

            # Apply under-relaxation
            x_updated = (1 - self.relaxation) * x_current + self.relaxation * x_new

            # Compute residual: ||x_updated - x_current||
            diff = x_updated - x_current
            residual_abs = np.linalg.norm(diff.flatten())

            if self.norm_type == "relative":
                residual = residual_abs / (np.linalg.norm(x_current.flatten()) + 1e-10)
            else:  # absolute
                residual = residual_abs

            residual_history.append(float(residual))

            # Check convergence
            if residual < self.tolerance:
                solver_time = time.time() - start_time
                result = x_updated.item() if is_scalar else x_updated
                return result, SolverInfo(
                    converged=True,
                    iterations=iteration + 1,
                    residual=residual,
                    residual_history=residual_history,
                    solver_time=solver_time,
                )

            x_current = x_updated

        # Maximum iterations reached
        solver_time = time.time() - start_time
        result = x_current.item() if is_scalar else x_current
        return result, SolverInfo(
            converged=False,
            iterations=self.max_iterations,
            residual=residual_history[-1] if residual_history else float("inf"),
            residual_history=residual_history,
            solver_time=solver_time,
        )


class NewtonSolver(NonlinearSolver):
    """
    Newton's method for nonlinear systems.

    Solves: F(x) = 0 via Newton iteration
        J(x^k) δx = -F(x^k)
        x^{k+1} = x^k + α·δx

    where:
        - J = ∂F/∂x is the Jacobian
        - α is the line search parameter (optional)

    Control theory: Newton-Raphson dynamic programming.

    Features:
        - Automatic Jacobian via finite differences
        - User-provided Jacobian (faster if available)
        - Sparse matrix support for large systems
        - Optional line search for robustness

    Example:
        >>> F = lambda x: x**2 - 4  # Solve x^2 = 4
        >>> solver = NewtonSolver(tol=1e-8)
        >>> x, info = solver.solve(F, x0=1.0)
        >>> print(f"Solution: {x}, iterations: {info.iterations}")

    Notes:
        - Quadratic convergence near solution
        - Requires good initial guess
        - May fail for poor x0 (use line search)
    """

    def __init__(
        self,
        max_iterations: int = 30,
        tolerance: float = 1e-6,
        jacobian: Callable[[NDArray], NDArray | sparse.spmatrix] | None = None,
        sparse: bool = True,
        line_search: bool = True,
        divergence_tolerance: float = 1e4,
        finite_diff_epsilon: float = 1e-7,
        use_jax_autodiff: bool | Literal["auto"] = "auto",
    ):
        """
        Initialize Newton solver.

        Args:
            max_iterations: Maximum Newton iterations
            tolerance: Convergence tolerance on ||F(x)||
            jacobian: Optional Jacobian function J: x → ∂F/∂x
                     If None, uses autodiff (if JAX available) or finite differences
            sparse: Use sparse linear solver (for large systems)
            line_search: Backtracking line search. **On by default (Issue #1745)** -- a
                Newton that accepts a step increasing the residual can walk away from a root
                it had almost reached. Measured on a 2-D HJB solve: the residual converged to
                1.57e-03 in five iterations, then diverged geometrically to 2.42e-01 because
                the full step increased it (1.63e-03) where a half step would have quartered
                it (3.95e-04). No surveyed library -- PETSc, KINSOL, SciPy, Optimistix,
                Trilinos NOX -- ships an unchecked full step as what a caller gets by
                omission. Pass False only with a reason.
            divergence_tolerance: stop when the residual exceeds this multiple of the initial
                one. Always applied, line search or not (PETSc default, 1e4).
            finite_diff_epsilon: Step size for finite differences
            use_jax_autodiff: Whether to use JAX automatic differentiation:
                - True: Use JAX autodiff (raises error if JAX not available)
                - False: Use finite differences
                - "auto" (default): Use JAX if available, else finite differences
        """
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be > 0, got {tolerance}")
        if finite_diff_epsilon <= 0:
            raise ValueError(f"finite_diff_epsilon must be > 0, got {finite_diff_epsilon}")

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.jacobian_func = jacobian
        self.use_sparse = sparse
        self.line_search = line_search
        # PETSc's SNESSetDivergenceTolerance default, and always on there for the same reason.
        self.divergence_tolerance = divergence_tolerance
        self.epsilon = finite_diff_epsilon

        # Configure JAX autodiff
        if use_jax_autodiff is True and not _JAX_AVAILABLE:
            raise ImportError("JAX is required for autodiff Jacobian but not installed")
        if use_jax_autodiff == "auto":
            self.use_jax = _JAX_AVAILABLE
        else:
            self.use_jax = bool(use_jax_autodiff) and _JAX_AVAILABLE

        # Cache for JIT-compiled Jacobian function
        self._jax_jacobian_fn: Any = None

    def solve(  # type: ignore[override]
        self,
        F: Callable[[NDArray], NDArray],
        x0: NDArray,
        jacobian: Callable[[NDArray], NDArray | sparse.spmatrix] | None = None,
        **kwargs: Any,
    ) -> tuple[NDArray, SolverInfo]:
        """
        Solve F(x) = 0 using Newton's method.

        Args:
            F: Residual function F: x → ℝ^n (same shape as x)
            x0: Initial guess (any shape)
            jacobian: Optional Jacobian (overrides self.jacobian_func)

        Returns:
            x: Solution (same shape as x0)
            info: Convergence information

        Note:
            Parameter names 'F' and 'jacobian' intentionally differ from
            base class 'func' for better API clarity in Newton context.
        """
        import time

        start_time = time.time()
        # Convert to numpy array to handle scalars
        x_current = np.asarray(x0, dtype=float).copy()
        original_shape = np.asarray(x0).shape
        is_scalar = original_shape == ()

        residual_history = []
        jacobian_evals = 0

        # Use provided jacobian or default
        jac_func = jacobian or self.jacobian_func

        # Local flag for JAX usage (can be disabled if tracing fails)
        use_jax_autodiff = self.use_jax

        for iteration in range(self.max_iterations):
            # Evaluate residual
            F_current = F(x_current)
            residual_norm = np.linalg.norm(F_current.flatten())
            residual_history.append(float(residual_norm))

            # Check convergence
            if residual_norm < self.tolerance:
                solver_time = time.time() - start_time
                result = x_current.item() if is_scalar else x_current
                return result, SolverInfo(
                    converged=True,
                    iterations=iteration + 1,
                    residual=residual_norm,
                    residual_history=residual_history,
                    solver_time=solver_time,
                    jacobian_evals=jacobian_evals,
                )

            # Compute Jacobian
            if jac_func is not None:
                J = jac_func(x_current)
                jacobian_evals += 1
            elif use_jax_autodiff:
                # Use JAX automatic differentiation (O(1) Jacobian)
                try:
                    J = compute_jacobian(F, x_current, original_shape, mode="reverse")
                    jacobian_evals += 1
                except Exception as e:
                    # JAX tracing failed (e.g., function uses numpy operations)
                    # Fall back to finite differences for this iteration
                    # and disable JAX for remaining iterations
                    use_jax_autodiff = False
                    warnings.warn(
                        f"JAX autodiff failed: {e!r}. "
                        "Falling back to finite-difference Jacobian (O(N) complexity). "
                        "For better performance, use JAX-compatible functions (jax.numpy).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    J = self._finite_difference_jacobian(F, x_current, F_current)
                    jacobian_evals += 1
            else:
                # Fallback: finite difference Jacobian (O(N) complexity)
                J = self._finite_difference_jacobian(F, x_current, F_current)
                jacobian_evals += 1

            # Solve linear system: J δx = -F
            delta_x = self._solve_linear_system(J, -F_current.flatten(), original_shape)

            # Line search (on by default; Issue #1745)
            if self.line_search:
                alpha = self._backtracking_line_search(F, x_current, delta_x, F_current)
                if alpha == 0.0:
                    return self._stop(
                        x_current,
                        is_scalar,
                        iteration,
                        residual_norm,
                        residual_history,
                        start_time,
                        jacobian_evals,
                        reason="line_search_failed",
                        detail=(
                            "no step along the Newton direction reduced the residual. The "
                            "direction may be wrong (check the Jacobian) or the iterate may "
                            "be at a local minimum of ||F|| that is not a root."
                        ),
                    )
            else:
                alpha = 1.0

            # Divergence guard, always on. PETSc keeps SNESSetDivergenceTolerance active
            # regardless of line search for the same reason: a run that has grown this far past
            # its starting residual is not going to recover, and the remaining iterations only
            # delay the diagnosis.
            if residual_norm > self.divergence_tolerance * residual_history[0]:
                return self._stop(
                    x_current,
                    is_scalar,
                    iteration,
                    residual_norm,
                    residual_history,
                    start_time,
                    jacobian_evals,
                    reason="diverged",
                    detail=(
                        f"residual grew to {residual_norm:.3e}, more than "
                        f"{self.divergence_tolerance:g}x the initial {residual_history[0]:.3e}."
                    ),
                )

            # Update
            x_current = x_current + alpha * delta_x

        # Maximum iterations reached
        solver_time = time.time() - start_time
        F_final = F(x_current)
        residual_final = np.linalg.norm(F_final.flatten())

        result = x_current.item() if is_scalar else x_current
        return result, SolverInfo(
            converged=False,
            iterations=self.max_iterations,
            residual=float(residual_final),
            residual_history=residual_history,
            solver_time=solver_time,
            jacobian_evals=jacobian_evals,
        )

    def _finite_difference_jacobian(
        self,
        F: Callable[[NDArray], NDArray],
        x: NDArray,
        F_x: NDArray,
    ) -> NDArray | sparse.spmatrix:
        """
        Compute Jacobian via forward finite differences.

        J[i,j] ≈ (F(x + ε·e_j)[i] - F(x)[i]) / ε
        """
        x_flat = x.flatten()
        F_flat = F_x.flatten()
        n = len(x_flat)

        if self.use_sparse:
            # Build sparse Jacobian (assumes sparsity pattern)
            J = sparse.lil_matrix((n, n), dtype=np.float64)
        else:
            J = np.zeros((n, n), dtype=np.float64)

        for j in range(n):
            # Perturb j-th component
            x_pert = x_flat.copy()
            x_pert[j] += self.epsilon

            # Evaluate F at perturbed point
            F_pert = F(x_pert.reshape(x.shape)).flatten()

            # Finite difference
            J[:, j] = (F_pert - F_flat) / self.epsilon

        return J.tocsr() if self.use_sparse else J

    def _solve_linear_system(
        self,
        J: NDArray | sparse.spmatrix,
        rhs: NDArray,
        original_shape: tuple,
    ) -> NDArray:
        """Solve J·δx = rhs and reshape to original_shape."""
        if sparse.issparse(J):
            delta_x_flat = spsolve(J, rhs)
        else:
            delta_x_flat = np.linalg.solve(J, rhs)

        return delta_x_flat.reshape(original_shape)

    def _stop(
        self,
        x_current,
        is_scalar: bool,
        iteration: int,
        residual: float,
        residual_history: list[float],
        start_time: float,
        jacobian_evals: int,
        *,
        reason: str,
        detail: str,
    ):
        """Stop with a NAMED outcome rather than exhausting the iteration budget.

        Issue #1745: "no acceptable step" and "diverged" are different from "ran out of
        iterations", and a caller that cannot tell them apart cannot act on any of them. Every
        surveyed library types these separately -- PETSc SNES_DIVERGED_LINE_SEARCH /
        SNES_DIVERGED_DTOL, KINSOL KIN_LINESEARCH_NONCONV. The reason travels in `extra` so the
        SolverInfo signature is unchanged.
        """
        warnings.warn(f"Newton stopped: {reason}. {detail}", RuntimeWarning, stacklevel=3)
        return (
            x_current.item() if is_scalar else x_current,
            SolverInfo(
                converged=False,
                # `iteration + 1`, matching the converged and budget-exhausted paths: the
                # invariant across all three is `iterations == len(residual_history)`. Passing
                # the raw 0-based index produced "failed to converge after 0 iterations", which
                # reads as though the solver never ran.
                iterations=iteration + 1,
                residual=float(residual),
                residual_history=residual_history,
                solver_time=time.time() - start_time,
                # Splatted, NOT `extra={...}`. `SolverInfo.__init__` takes `**extra`, so a
                # keyword literally named `extra` nests one level: callers found
                # `info.extra["extra"]["reason"]` while every doc said `info.extra["reason"]`.
                # It also shadowed `jacobian_evals`, so `newton_mfg_solver.py`'s
                # `extra.get("jacobian_evals", 0)` reported 0 on every stop path.
                reason=reason,
                detail=detail,
                jacobian_evals=jacobian_evals,
            ),
        )

    def _backtracking_line_search(
        self,
        F: Callable[[NDArray], NDArray],
        x: NDArray,
        delta_x: NDArray,
        F_x: NDArray,
        alpha_init: float = 1.0,
        rho: float = 0.5,
        c: float = 1e-4,
    ) -> float:
        """
        Backtracking line search (Armijo rule).

        Finds α such that: ||F(x + α·δx)|| < ||F(x)|| - c·α·||δx||²
        """
        alpha = alpha_init
        norm_F_x = np.linalg.norm(F_x.flatten())

        for _ in range(20):  # Max 20 backtracking steps
            x_trial = x + alpha * delta_x
            F_trial = F(x_trial)
            norm_F_trial = np.linalg.norm(F_trial.flatten())

            # Armijo sufficient decrease, in the standard form for a residual norm:
            #   ||F(x + a*d)|| <= (1 - c*a) * ||F(x)||
            # The previous form subtracted `c * alpha * ||delta_x||^2` from ||F(x)||, which
            # compares a residual norm against a STEP norm -- different units, and the term is
            # so small near a solution that it accepted essentially any decrease.
            if norm_F_trial <= (1.0 - c * alpha) * norm_F_x:
                return alpha

            alpha *= rho

        # No acceptable step. Return 0.0 rather than the last tiny alpha: a step that does not
        # reduce the residual is not worth taking, and reporting the failure is the caller's
        # only chance to act on it. PETSc types this as SNES_DIVERGED_LINE_SEARCH rather than
        # continuing; the previous code returned ~1e-6 and burned the remaining iterations.
        return 0.0


class PolicyIterationSolver(NonlinearSolver):
    """
    Policy iteration (Howard's algorithm) for control problems.

    Solves optimal control problems via:
        1. Policy evaluation: Solve value function V^π for fixed policy π
        2. Policy improvement: Update π' = argmax_a Q(s, a, V^π)
        3. Repeat until π = π'

    Control theory: This is Howard's policy iteration algorithm.
    MFG context: Alternates between solving HJB for fixed control and
                 improving control based on value function.

    Algorithm:
        Given: policy_eval(π) → V, policy_improve(V) → π'
        1. Initialize π_0
        2. Loop:
            a. V_k = policy_eval(π_k)
            b. π_{k+1} = policy_improve(V_k)
            c. If π_{k+1} = π_k, stop

    Example:
        >>> def eval_policy(policy):
        ...     # Solve linear system for value given policy
        ...     return value
        >>> def improve_policy(value):
        ...     # Compute optimal policy from value
        ...     return new_policy
        >>> solver = PolicyIterationSolver(max_iterations=20)
        >>> value, policy, info = solver.solve(eval_policy, improve_policy, policy0)

    Notes:
        - Typically faster than value iteration
        - Each step solves linear (not nonlinear) system
        - Natural for control problems with discrete actions
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ):
        """
        Initialize policy iteration solver.

        Args:
            max_iterations: Maximum policy iteration steps
            tolerance: Convergence tolerance on policy change
        """
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be > 0, got {tolerance}")

        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def solve(  # type: ignore[override]
        self,
        policy_eval: Callable[[NDArray], NDArray],
        policy_improve: Callable[[NDArray], NDArray],
        policy0: NDArray,
        **kwargs: Any,
    ) -> tuple[NDArray, NDArray, SolverInfo]:
        """
        Solve optimal control via policy iteration.

        Args:
            policy_eval: Function π → V (evaluate value for policy)
            policy_improve: Function V → π' (improve policy from value)
            policy0: Initial policy guess

        Returns:
            value: Optimal value function
            policy: Optimal policy
            info: Convergence information

        Note:
            Signature intentionally differs from base class (returns 3 values
            instead of 2, has 2 callables instead of 1) for policy iteration
            API clarity.
        """
        import time

        start_time = time.time()
        policy_current = policy0.copy()
        residual_history = []

        for iteration in range(self.max_iterations):
            # Step 1: Policy evaluation
            value_current = policy_eval(policy_current)

            # Step 2: Policy improvement
            policy_new = policy_improve(value_current)

            # Check convergence: ||π_new - π_current||
            policy_diff = policy_new - policy_current
            residual = np.linalg.norm(policy_diff.flatten())
            residual_history.append(float(residual))

            if residual < self.tolerance:
                solver_time = time.time() - start_time
                return (
                    value_current,
                    policy_new,
                    SolverInfo(
                        converged=True,
                        iterations=iteration + 1,
                        residual=residual,
                        residual_history=residual_history,
                        solver_time=solver_time,
                    ),
                )

            policy_current = policy_new

        # Maximum iterations reached
        solver_time = time.time() - start_time
        value_final = policy_eval(policy_current)

        return (
            value_final,
            policy_current,
            SolverInfo(
                converged=False,
                iterations=self.max_iterations,
                residual=residual_history[-1] if residual_history else float("inf"),
                residual_history=residual_history,
                solver_time=solver_time,
            ),
        )


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing Nonlinear Solvers...")

    import numpy as np

    # Test 1: Fixed-point solver
    print("\n1. Testing FixedPointSolver...")
    fp_solver = FixedPointSolver(relaxation=0.5, max_iterations=100, tolerance=1e-6)

    # Simple fixed-point problem: x = cos(x)
    def G_fixedpoint(x):
        return np.cos(x)

    x0 = 1.0
    x_sol, info = fp_solver.solve(G_fixedpoint, x0)

    assert info.converged, "Fixed-point solver should converge"
    assert abs(x_sol - np.cos(x_sol)) < 1e-5, "Solution should satisfy x = cos(x)"

    print(f"  Converged: {info.converged}, iterations: {info.iterations}")
    print(f"  Solution: x = {x_sol:.6f}, residual: {info.residual:.2e}")

    # Test 2: Newton solver
    print("\n2. Testing NewtonSolver...")
    newton_solver = NewtonSolver(max_iterations=50, tolerance=1e-8, sparse=False)

    # Nonlinear equation: x^2 - 2 = 0 (solution x = sqrt(2))
    def F_newton(x):
        return x**2 - 2.0

    x0 = 1.0
    x_sol, info = newton_solver.solve(F_newton, x0)

    assert info.converged, "Newton solver should converge"
    assert abs(x_sol - np.sqrt(2)) < 1e-6, "Solution should be sqrt(2)"

    print(f"  Converged: {info.converged}, iterations: {info.iterations}")
    print(f"  Solution: x = {x_sol:.6f}, residual: {info.residual:.2e}")
    print(f"  Expected: {np.sqrt(2):.6f}")

    # Test 3: Policy Iteration solver
    print("\n3. Testing PolicyIterationSolver...")
    pi_solver = PolicyIterationSolver(max_iterations=20, tolerance=1e-5)

    # Simple MDP: 2 states, 2 actions
    # State transition probability
    def P(state, action):
        # Deterministic transitions
        if action == 0:
            return {0: 1.0} if state == 0 else {1: 1.0}
        else:
            return {1: 1.0} if state == 0 else {0: 1.0}

    # Reward function
    def R(state, action):
        return 1.0 if (state == 0 and action == 1) else 0.0

    # Policy evaluation (simplified)
    def policy_eval(policy):
        # For this test, just return dummy value function
        return np.array([1.0, 0.5])

    # Policy improvement (simplified)
    def policy_improve(value):
        # For this test, just return dummy policy
        return np.array([1, 0])

    # Note: PolicyIterationSolver requires proper MDP setup, so this is a minimal test
    print("  PolicyIterationSolver requires full MDP setup - skipping detailed test")

    print("\nAll smoke tests passed!")

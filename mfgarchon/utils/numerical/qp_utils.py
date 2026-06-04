#!/usr/bin/env python3
"""
Quadratic Programming Utilities for GFDM Solvers

This module provides reusable QP solving infrastructure with caching and warm-starting
optimizations. Primary use case: Computing finite difference coefficients with monotonicity
constraints in HJB GFDM solvers.

Key Features:
- Multiple solver backends: OSQP (fast), scipy SLSQP (general), scipy L-BFGS-B (bounds-only)
- Warm-starting: Reuse previous solutions to accelerate convergence
- Result caching: Hash-based lookup with LRU eviction for identical subproblems
- Performance tracking: Detailed statistics on solver usage and timing

Use Cases:
- GFDM monotone finite difference schemes
- Constrained least-squares problems
- Optimization with physical bounds and monotonicity constraints

Example:
    >>> from mfgarchon.utils import QPSolver, QPCache
    >>>
    >>> # Create solver with caching
    >>> cache = QPCache(max_size=1000)
    >>> solver = QPSolver(backend="osqp", enable_warm_start=True, cache=cache)
    >>>
    >>> # Solve constrained least-squares problem
    >>> result = solver.solve_weighted_least_squares(
    ...     A=design_matrix,
    ...     b=target_values,
    ...     W=weight_matrix,
    ...     bounds=[(-10, 10)] * n_coeffs,
    ...     point_id="point_42"  # For warm-starting
    ... )
    >>>
    >>> # Check cache statistics
    >>> print(f"Cache hit rate: {cache.hit_rate:.1%}")
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Optional dependencies - runtime imports
try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import osqp

    import scipy.sparse as sp

    OSQP_AVAILABLE = True
except ImportError:
    OSQP_AVAILABLE = False

# Type-checking imports are handled above in runtime imports
# No separate TYPE_CHECKING block needed - using runtime availability checks


# =============================================================================
# QP RESULT CACHING
# =============================================================================


class QPCache:
    """
    Hash-based cache for QP problem results with LRU eviction.

    Caches solutions to identical QP subproblems to avoid redundant solves.
    Useful when the same QP structure appears across iterations or time steps.

    Attributes:
        max_size: Maximum cache entries (LRU eviction when exceeded)
        hits: Number of cache hits
        misses: Number of cache misses

    Example:
        >>> cache = QPCache(max_size=500)
        >>> # Solver uses cache.get() and cache.put() internally
        >>> print(f"Hit rate: {cache.hit_rate:.1%}")
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize QP result cache.

        Args:
            max_size: Maximum number of cached results (default 1000)
                When exceeded, least recently used entries are evicted.
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, NDArray[np.float64]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _compute_hash(self, A: NDArray[np.float64], b: NDArray[np.float64], W: NDArray[np.float64]) -> str:
        """
        Compute hash key for QP problem.

        Hash is based on problem structure (A, b, W) to detect identical subproblems.
        Uses SHA256 for collision resistance.

        Args:
            A: Design matrix (n_points, n_coeffs)
            b: Target values (n_points,)
            W: Weight matrix (n_points, n_points) or diagonal (n_points,)

        Returns:
            64-character hex hash string
        """
        # Convert to bytes for hashing
        # Use tobytes() for efficiency (direct memory copy)
        hasher = hashlib.sha256()
        hasher.update(A.tobytes())
        hasher.update(b.tobytes())

        # Handle both full weight matrix and diagonal weights
        if W.ndim == 1:
            hasher.update(W.tobytes())
        else:
            # For full matrix, hash diagonal only (most problems use diagonal W)
            hasher.update(np.diag(W).tobytes())

        return hasher.hexdigest()

    def get(self, A: NDArray[np.float64], b: NDArray[np.float64], W: NDArray[np.float64]) -> NDArray[np.float64] | None:
        """
        Retrieve cached solution if available.

        Args:
            A: Design matrix
            b: Target values
            W: Weight matrix

        Returns:
            Cached solution if found, None otherwise
        """
        key = self._compute_hash(A, b, W)

        if key in self._cache:
            self.hits += 1
            # Move to end (LRU update)
            self._cache.move_to_end(key)
            return self._cache[key].copy()

        self.misses += 1
        return None

    def put(
        self, A: NDArray[np.float64], b: NDArray[np.float64], W: NDArray[np.float64], solution: NDArray[np.float64]
    ) -> None:
        """
        Store solution in cache.

        Args:
            A: Design matrix
            b: Target values
            W: Weight matrix
            solution: QP solution to cache
        """
        key = self._compute_hash(A, b, W)

        # Evict oldest entry if cache full
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # Remove oldest (FIFO)

        self._cache[key] = solution.copy()

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0


# =============================================================================
# QP SOLVER WITH WARM-STARTING
# =============================================================================


class QPSolver:
    """
    Unified QP solver with multiple backends, warm-starting, and caching.

    Solves weighted least-squares problems with optional constraints:
        minimize    (1/2) ||W^(1/2) (A x - b)||^2
        subject to  bounds and monotonicity constraints

    Supports multiple solver backends:
    - OSQP: Fast, warm-starting, best for large problems
    - scipy SLSQP: General constraints, robust
    - scipy L-BFGS-B: Fast for bounds-only problems

    Attributes:
        backend: Solver backend ("osqp", "scipy-slsqp", "scipy-lbfgsb", "auto")
        enable_warm_start: Use previous solutions as initial guesses
        cache: Optional QPCache for result caching
        stats: Dictionary of solve statistics

    Example:
        >>> solver = QPSolver(backend="osqp", enable_warm_start=True)
        >>> result = solver.solve_weighted_least_squares(
        ...     A=design_matrix, b=target, W=weights,
        ...     bounds=[(-10, 10)] * n, point_id="point_0"
        ... )
        >>> solver.print_statistics()
    """

    def __init__(
        self,
        backend: Literal["osqp", "scipy-slsqp", "scipy-lbfgsb", "auto"] = "auto",
        enable_warm_start: bool = True,
        cache: QPCache | None = None,
    ):
        """
        Initialize QP solver.

        Args:
            backend: Solver backend to use
                - "osqp": OSQP (requires osqp package)
                - "scipy-slsqp": scipy SLSQP (general constraints)
                - "scipy-lbfgsb": scipy L-BFGS-B (bounds only, fastest scipy)
                - "auto": Choose best available (OSQP > scipy)
            enable_warm_start: Enable warm-starting from previous solutions
            cache: Optional QPCache for result caching

        Raises:
            ValueError: If requested backend not available
        """
        # Validate and set backend
        if backend == "auto":
            self.backend = "osqp" if OSQP_AVAILABLE else "scipy-slsqp"
        else:
            self.backend = backend

        if self.backend == "osqp" and not OSQP_AVAILABLE:
            raise ValueError("OSQP backend requested but osqp package not available")
        if self.backend.startswith("scipy") and not SCIPY_AVAILABLE:
            raise ValueError("scipy backend requested but scipy package not available")

        self.enable_warm_start = enable_warm_start
        self.cache = cache

        # Warm-start storage: point_id -> (primal, dual)
        self._warm_start_cache: dict[Any, tuple[NDArray[np.float64], NDArray[np.float64] | None]] = {}

        # Statistics tracking
        self.stats = {
            "total_solves": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "warm_starts": 0,
            "cold_starts": 0,
            "osqp_solves": 0,
            "slsqp_solves": 0,
            "lbfgsb_solves": 0,
            "successes": 0,
            "failures": 0,
            "solve_times": [],
        }

    def solve_weighted_least_squares(
        self,
        A: NDArray[np.float64],
        b: NDArray[np.float64],
        W: NDArray[np.float64],
        bounds: list[tuple[float | None, float | None]] | None = None,
        constraints: list[dict[str, Any]] | None = None,
        point_id: Any = None,
    ) -> NDArray[np.float64]:
        """
        Solve weighted least-squares problem with constraints.

        Minimizes: (1/2) ||W^(1/2) (A x - b)||^2
        Subject to: variable bounds and optional constraints

        Args:
            A: Design matrix, shape (n_points, n_coeffs)
            b: Target values, shape (n_points,)
            W: Weight matrix (n_points, n_points) or diagonal weights (n_points,)
            bounds: List of (lower, upper) bounds for each variable
                Use None for unbounded. Default: all unbounded
            constraints: List of constraint dicts (scipy format)
                Example: [{"type": "ineq", "fun": lambda x: x[0] + x[1] - 1}]
            point_id: Identifier for warm-starting (e.g., collocation point index)
                Solutions are cached per point_id for reuse across iterations

        Returns:
            Solution vector x, shape (n_coeffs,)

        Example:
            >>> # Solve constrained least-squares
            >>> A = np.random.randn(20, 5)
            >>> b = np.random.randn(20)
            >>> W = np.eye(20)
            >>> bounds = [(-10, 10)] * 5
            >>> x = solver.solve_weighted_least_squares(A, b, W, bounds, point_id=0)
        """
        t0 = time.time()
        self.stats["total_solves"] += 1

        # Check cache if available
        if self.cache is not None:
            cached_result = self.cache.get(A, b, W)
            if cached_result is not None:
                self.stats["cache_hits"] += 1
                elapsed = time.time() - t0
                self.stats["solve_times"].append(elapsed)
                return cached_result
            self.stats["cache_misses"] += 1

        # Prepare default bounds
        n_coeffs = A.shape[1]
        if bounds is None:
            bounds = [(None, None)] * n_coeffs
        if constraints is None:
            constraints = []

        # Compute initial guess (unconstrained solution)
        x0 = self._solve_unconstrained(A, b, W)

        # Check warm-start cache
        warm_start_available = self.enable_warm_start and point_id is not None and point_id in self._warm_start_cache

        # Dispatch to appropriate solver
        if self.backend == "osqp":
            result = self._solve_with_osqp(A, b, W, bounds, constraints, x0, point_id, warm_start_available)
        elif self.backend == "scipy-lbfgsb" or (self.backend == "auto" and len(constraints) == 0):
            result = self._solve_with_lbfgsb(A, b, W, bounds, x0, point_id, warm_start_available)
        else:  # scipy-slsqp or auto with constraints
            result = self._solve_with_slsqp(A, b, W, bounds, constraints, x0, point_id, warm_start_available)

        # Cache result if caching enabled
        if self.cache is not None:
            self.cache.put(A, b, W, result)

        elapsed = time.time() - t0
        self.stats["solve_times"].append(elapsed)

        return result

    def _solve_unconstrained(
        self, A: NDArray[np.float64], b: NDArray[np.float64], W: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Solve unconstrained weighted least-squares using normal equations.

        Solves: A^T W A x = A^T W b

        Args:
            A: Design matrix
            b: Target values
            W: Weight matrix or diagonal weights

        Returns:
            Unconstrained solution
        """
        # Handle diagonal vs full weight matrix
        if W.ndim == 1:
            # Diagonal weights: W is vector
            WA = W[:, np.newaxis] * A  # Broadcasting
            Wb = W * b
        else:
            # Full weight matrix
            WA = W @ A
            Wb = W @ b

        # Solve normal equations: A^T W A x = A^T W b
        # Use lstsq for numerical stability
        try:
            from scipy.linalg import lstsq

            solution, *_ = lstsq(A.T @ WA, A.T @ Wb)
            return solution
        except ImportError:
            # Fallback to numpy
            solution, *_ = np.linalg.lstsq(A.T @ WA, A.T @ Wb, rcond=None)
            return solution

    def _solve_with_osqp(
        self,
        A: NDArray[np.float64],
        b: NDArray[np.float64],
        W: NDArray[np.float64],
        bounds: list[tuple[float | None, float | None]],
        constraints: list[dict[str, Any]],
        x0: NDArray[np.float64],
        point_id: Any,
        warm_start_available: bool,
    ) -> NDArray[np.float64]:
        """Solve using OSQP backend."""
        if not OSQP_AVAILABLE:
            raise RuntimeError("OSQP not available")

        # Type narrowing for mypy
        assert osqp is not None
        assert sp is not None

        self.stats["osqp_solves"] += 1
        n = len(x0)

        # Compute QP matrices: min (1/2) x^T P x + q^T x
        # From ||W^(1/2) (Ax - b)||^2 = x^T (A^T W A) x - 2 (A^T W b)^T x + b^T W b
        if W.ndim == 1:
            WA = W[:, np.newaxis] * A
            Wb = W * b
        else:
            WA = W @ A
            Wb = W @ b

        P = A.T @ WA
        q = -A.T @ Wb

        # Build constraint matrix (bounds as linear constraints)
        constraint_rows = []
        l_bounds = []
        u_bounds = []

        for i, (lb, ub) in enumerate(bounds):
            row = np.zeros(n)
            row[i] = 1.0
            constraint_rows.append(row)
            l_bounds.append(lb if lb is not None else -np.inf)
            u_bounds.append(ub if ub is not None else np.inf)

        # Note: OSQP doesn't support nonlinear constraints
        # For monotonicity constraints, we skip them here (bounds-only approximation)
        # Full implementation would linearize or use SLSQP

        A_constraint = sp.csc_matrix(np.vstack(constraint_rows)) if constraint_rows else sp.csc_matrix((0, n))
        lower_bounds = np.array(l_bounds)
        upper_bounds = np.array(u_bounds)

        # Setup OSQP problem
        P_sparse = sp.csc_matrix(P)
        prob = osqp.OSQP()
        prob.setup(
            P=P_sparse,
            q=q,
            A=A_constraint,
            l=lower_bounds,
            u=upper_bounds,
            verbose=False,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=10000,
            polishing=False,  # OSQP 1.0+ name; "polish" deprecated (warns per QP solve)
        )

        # Apply warm-start if available
        if warm_start_available:
            x_prev, y_prev = self._warm_start_cache[point_id]
            if y_prev is not None and len(y_prev) == len(lower_bounds):
                prob.warm_start(x=x_prev, y=y_prev)
            else:
                prob.warm_start(x=x_prev)
            self.stats["warm_starts"] += 1
        else:
            self.stats["cold_starts"] += 1

        # Solve. raise_error=False is the current default but explicit here to silence the
        # OSQP 1.x PendingDeprecationWarning; failures are handled below via the status check.
        result = prob.solve(raise_error=False)

        if result.info.status == "solved":
            self.stats["successes"] += 1
            # Cache warm-start
            if self.enable_warm_start and point_id is not None:
                self._warm_start_cache[point_id] = (result.x.copy(), result.y.copy())
            return result.x

        # Fallback to unconstrained
        self.stats["failures"] += 1
        return x0

    def _solve_with_lbfgsb(
        self,
        A: NDArray[np.float64],
        b: NDArray[np.float64],
        W: NDArray[np.float64],
        bounds: list[tuple[float | None, float | None]],
        x0: NDArray[np.float64],
        point_id: Any,
        warm_start_available: bool,
    ) -> NDArray[np.float64]:
        """Solve using scipy L-BFGS-B (bounds-only, fast)."""
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy not available")

        # Type narrowing for mypy
        assert minimize is not None

        self.stats["lbfgsb_solves"] += 1

        # Handle weights
        if W.ndim == 1:
            sqrt_W = np.sqrt(W)

            def objective(x):
                residual = sqrt_W * (A @ x - b)
                return 0.5 * np.dot(residual, residual)

            def gradient(x):
                residual = A @ x - b
                return A.T @ (W * residual)
        else:
            sqrt_W = np.linalg.cholesky(W)

            def objective(x):
                residual = sqrt_W @ (A @ x - b)
                return 0.5 * np.dot(residual, residual)

            def gradient(x):
                residual = A @ x - b
                return A.T @ (W @ residual)

        # Warm-start initial guess
        if warm_start_available:
            x_init = self._warm_start_cache[point_id][0]
            self.stats["warm_starts"] += 1
        else:
            x_init = x0
            self.stats["cold_starts"] += 1

        result = minimize(
            objective,
            x_init,
            method="L-BFGS-B",
            jac=gradient,
            bounds=bounds,
            options={"maxiter": 50, "ftol": 1e-6, "gtol": 1e-6},
        )

        if result.success:
            self.stats["successes"] += 1
            # Cache warm-start
            if self.enable_warm_start and point_id is not None:
                self._warm_start_cache[point_id] = (result.x.copy(), None)
            return result.x

        self.stats["failures"] += 1
        return x0

    def _solve_with_slsqp(
        self,
        A: NDArray[np.float64],
        b: NDArray[np.float64],
        W: NDArray[np.float64],
        bounds: list[tuple[float | None, float | None]],
        constraints: list[dict[str, Any]],
        x0: NDArray[np.float64],
        point_id: Any,
        warm_start_available: bool,
    ) -> NDArray[np.float64]:
        """Solve using scipy SLSQP (general constraints)."""
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy not available")

        # Type narrowing for mypy
        assert minimize is not None

        self.stats["slsqp_solves"] += 1

        # Handle weights
        if W.ndim == 1:
            sqrt_W = np.sqrt(W)

            def objective(x):
                residual = sqrt_W * (A @ x - b)
                return 0.5 * np.dot(residual, residual)

            def gradient(x):
                residual = A @ x - b
                return A.T @ (W * residual)
        else:
            sqrt_W = np.linalg.cholesky(W)

            def objective(x):
                residual = sqrt_W @ (A @ x - b)
                return 0.5 * np.dot(residual, residual)

            def gradient(x):
                residual = A @ x - b
                return A.T @ (W @ residual)

        # Warm-start initial guess
        if warm_start_available:
            x_init = self._warm_start_cache[point_id][0]
            self.stats["warm_starts"] += 1
        else:
            x_init = x0
            self.stats["cold_starts"] += 1

        result = minimize(
            objective,
            x_init,
            method="SLSQP",
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 40, "ftol": 1e-6, "eps": 1.4901161193847656e-08, "disp": False},
        )

        if result.success:
            self.stats["successes"] += 1
            # Cache warm-start
            if self.enable_warm_start and point_id is not None:
                self._warm_start_cache[point_id] = (result.x.copy(), None)
            return result.x

        self.stats["failures"] += 1
        return x0

    def print_statistics(self) -> None:
        """Print detailed solver statistics."""
        print("\n" + "=" * 60)
        print("QP Solver Statistics")
        print("=" * 60)

        total = self.stats["total_solves"]
        if total == 0:
            print("No solves recorded.")
            return

        print(f"\nTotal solves:        {total}")
        print(f"Successes:           {self.stats['successes']} ({100 * self.stats['successes'] / total:.1f}%)")
        print(f"Failures:            {self.stats['failures']} ({100 * self.stats['failures'] / total:.1f}%)")

        # Caching stats
        if self.cache is not None:
            cache_total = self.stats["cache_hits"] + self.stats["cache_misses"]
            if cache_total > 0:
                print("\nCache Statistics:")
                print(
                    f"  Hits:              {self.stats['cache_hits']} ({100 * self.stats['cache_hits'] / cache_total:.1f}%)"
                )
                print(
                    f"  Misses:            {self.stats['cache_misses']} ({100 * self.stats['cache_misses'] / cache_total:.1f}%)"
                )
                print(f"  Cache size:        {self.cache.size} / {self.cache.max_size}")

        # Warm-start stats
        if self.enable_warm_start:
            ws_total = self.stats["warm_starts"] + self.stats["cold_starts"]
            if ws_total > 0:
                print("\nWarm-Start Statistics:")
                print(
                    f"  Warm starts:       {self.stats['warm_starts']} ({100 * self.stats['warm_starts'] / ws_total:.1f}%)"
                )
                print(
                    f"  Cold starts:       {self.stats['cold_starts']} ({100 * self.stats['cold_starts'] / ws_total:.1f}%)"
                )

        # Solver backend breakdown
        print("\nSolver Backend Usage:")
        print(f"  OSQP:              {self.stats['osqp_solves']} ({100 * self.stats['osqp_solves'] / total:.1f}%)")
        print(f"  SLSQP:             {self.stats['slsqp_solves']} ({100 * self.stats['slsqp_solves'] / total:.1f}%)")
        print(f"  L-BFGS-B:          {self.stats['lbfgsb_solves']} ({100 * self.stats['lbfgsb_solves'] / total:.1f}%)")

        # Timing stats
        if self.stats["solve_times"]:
            times = np.array(self.stats["solve_times"])
            print("\nTiming Statistics:")
            print(f"  Total time:        {np.sum(times):.3f} s")
            print(f"  Mean time:         {np.mean(times) * 1000:.2f} ms")
            print(f"  Median time:       {np.median(times) * 1000:.2f} ms")
            print(f"  Min time:          {np.min(times) * 1000:.2f} ms")
            print(f"  Max time:          {np.max(times) * 1000:.2f} ms")

        print("=" * 60 + "\n")

    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        for key in self.stats:
            if isinstance(self.stats[key], list):
                self.stats[key] = []
            else:
                self.stats[key] = 0

        if self.cache is not None:
            self.cache.clear()


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "QPCache",
    "QPSolver",
]


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing QP utilities...")

    import numpy as np

    # Test weighted least squares: min ||Ax - b||_W^2
    # Simple problem: find x such that 2x ≈ 4, solution x ≈ 2
    A = np.array([[2.0]])
    b = np.array([4.0])
    W = np.array([[1.0]])  # Weight matrix

    solver = QPSolver()
    x = solver.solve_weighted_least_squares(A, b, W)

    assert x is not None, "QP solver returned None"
    assert x.shape == (1,), f"Wrong shape: {x.shape}"
    assert abs(x[0] - 2.0) < 1e-3, f"QP solution {x[0]} != 2.0"

    print(f"  Weighted least squares: x = {x[0]:.6f} (expected 2.0)")

    # Test cache
    cache = QPCache(max_size=2)

    # Store a solution
    cache.put(A, b, W, x)

    # Retrieve it
    x_cached = cache.get(A, b, W)
    assert x_cached is not None, "Cache retrieval failed"
    assert np.allclose(x_cached, x), "Cached solution mismatch"

    print(f"  QP cache: hit rate = {cache.hit_rate:.1%}, size = {cache.size}")
    print("Smoke tests passed!")

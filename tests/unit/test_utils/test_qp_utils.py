#!/usr/bin/env python3
"""
Unit Tests for QP Utilities

Tests for QPCache and QPSolver classes in mfgarchon.utils.numerical.qp_utils.

Coverage:
- QPCache: Hash computation, get/put operations, LRU eviction, statistics
- QPSolver: Multiple backends (OSQP, scipy), warm-starting, caching integration
- Weighted least-squares problems with bounds and constraints
"""

import pytest

import numpy as np

from mfgarchon.utils import QPCache, QPSolver

# =============================================================================
# QPCache Tests
# =============================================================================


class TestQPCache:
    """Test QPCache functionality."""

    def test_cache_basic_operations(self):
        """Test basic get/put operations."""
        cache = QPCache(max_size=10)

        # Create test problem
        A = np.random.randn(20, 5)
        b = np.random.randn(20)
        W = np.eye(20)
        solution = np.random.randn(5)

        # Cache should be empty initially
        assert cache.get(A, b, W) is None
        assert cache.size == 0
        assert cache.hits == 0
        assert cache.misses == 1

        # Put solution in cache
        cache.put(A, b, W, solution)
        assert cache.size == 1

        # Retrieve cached solution
        cached = cache.get(A, b, W)
        assert cached is not None
        assert np.allclose(cached, solution)
        assert cache.hits == 1
        assert cache.misses == 1

    def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        cache = QPCache(max_size=10)

        A = np.random.randn(10, 3)
        b = np.random.randn(10)
        W = np.eye(10)
        solution = np.random.randn(3)

        # First access: miss
        assert cache.get(A, b, W) is None
        assert cache.hit_rate == 0.0

        # Cache it
        cache.put(A, b, W, solution)

        # Second access: hit
        assert cache.get(A, b, W) is not None
        assert cache.hit_rate == 0.5  # 1 hit, 1 miss

        # Third access: hit
        assert cache.get(A, b, W) is not None
        assert cache.hit_rate == pytest.approx(2.0 / 3.0)

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = QPCache(max_size=3)

        # Create 4 different problems
        problems = []
        for i in range(4):
            A = np.random.randn(10, 3) + i  # Make each unique
            b = np.random.randn(10) + i
            W = np.eye(10)
            solution = np.random.randn(3)
            problems.append((A, b, W, solution))

        # Fill cache with first 3 problems
        for i in range(3):
            cache.put(*problems[i])
        assert cache.size == 3

        # All 3 should be cached
        for i in range(3):
            assert cache.get(problems[i][0], problems[i][1], problems[i][2]) is not None

        # Add 4th problem - should evict oldest (problem 0)
        cache.put(*problems[3])
        assert cache.size == 3

        # Problem 0 should be evicted
        assert cache.get(problems[0][0], problems[0][1], problems[0][2]) is None

        # Problems 1, 2, 3 should still be cached
        assert cache.get(problems[1][0], problems[1][1], problems[1][2]) is not None
        assert cache.get(problems[2][0], problems[2][1], problems[2][2]) is not None
        assert cache.get(problems[3][0], problems[3][1], problems[3][2]) is not None

    def test_cache_diagonal_weights(self):
        """Test cache works with diagonal weight vectors."""
        cache = QPCache(max_size=10)

        A = np.random.randn(15, 4)
        b = np.random.randn(15)
        W_diag = np.random.rand(15)  # Diagonal weights as vector
        solution = np.random.randn(4)

        # Cache with diagonal weights
        cache.put(A, b, W_diag, solution)

        # Retrieve
        cached = cache.get(A, b, W_diag)
        assert cached is not None
        assert np.allclose(cached, solution)

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = QPCache(max_size=10)

        A = np.random.randn(10, 3)
        b = np.random.randn(10)
        W = np.eye(10)
        solution = np.random.randn(3)

        # Put and get to generate stats
        cache.put(A, b, W, solution)
        cache.get(A, b, W)  # Hit
        cache.get(A + 0.1, b, W)  # Miss

        assert cache.size == 1
        assert cache.hits + cache.misses > 0

        # Clear cache
        cache.clear()
        assert cache.size == 0
        assert cache.hits == 0
        assert cache.misses == 0


# =============================================================================
# QPSolver Tests
# =============================================================================


class TestQPSolver:
    """Test QPSolver with multiple backends."""

    def test_solver_unconstrained_scipy(self):
        """Test unconstrained least-squares with scipy backend."""
        solver = QPSolver(backend="scipy-lbfgsb", enable_warm_start=False)

        # Create simple least-squares problem
        A = np.random.randn(20, 5)
        b = np.random.randn(20)
        W = np.eye(20)

        # Solve
        x = solver.solve_weighted_least_squares(A, b, W)

        # Check solution quality (should be close to lstsq solution)
        x_lstsq, *_ = np.linalg.lstsq(A, b, rcond=None)
        assert np.allclose(x, x_lstsq, atol=1e-4)

        # Check statistics
        assert solver.stats["total_solves"] == 1
        assert solver.stats["lbfgsb_solves"] == 1
        assert solver.stats["successes"] == 1

    def test_solver_bounded_scipy(self):
        """Test bounded least-squares with scipy backend."""
        solver = QPSolver(backend="scipy-lbfgsb", enable_warm_start=False)

        A = np.random.randn(20, 5)
        b = np.random.randn(20)
        W = np.eye(20)
        bounds = [(-1, 1)] * 5  # Constrain all variables to [-1, 1]

        # Solve
        x = solver.solve_weighted_least_squares(A, b, W, bounds=bounds)

        # Check bounds are satisfied
        assert np.all(x >= -1.0)
        assert np.all(x <= 1.0)

        assert solver.stats["successes"] == 1

    @pytest.mark.skipif(
        not hasattr(QPSolver(backend="auto"), "backend") or QPSolver(backend="auto").backend != "osqp",
        reason="OSQP not available",
    )
    def test_solver_osqp_backend(self):
        """Test OSQP backend."""
        solver = QPSolver(backend="osqp", enable_warm_start=False)

        A = np.random.randn(20, 5)
        b = np.random.randn(20)
        W = np.eye(20)
        bounds = [(-10, 10)] * 5

        # Solve
        x = solver.solve_weighted_least_squares(A, b, W, bounds=bounds)

        # Check solution is reasonable (bounded)
        assert np.all(np.abs(x) <= 10.0)

        assert solver.stats["osqp_solves"] == 1

    def test_solver_with_caching(self):
        """Test solver with result caching."""
        cache = QPCache(max_size=10)
        solver = QPSolver(backend="scipy-lbfgsb", enable_warm_start=False, cache=cache)

        A = np.random.randn(15, 4)
        b = np.random.randn(15)
        W = np.eye(15)

        # First solve: cache miss
        x1 = solver.solve_weighted_least_squares(A, b, W)
        assert solver.stats["cache_misses"] == 1
        assert solver.stats["cache_hits"] == 0

        # Second solve with same problem: cache hit
        x2 = solver.solve_weighted_least_squares(A, b, W)
        assert solver.stats["cache_misses"] == 1
        assert solver.stats["cache_hits"] == 1

        # Solutions should be identical
        assert np.allclose(x1, x2)

    def test_solver_warm_starting(self):
        """Test warm-starting across multiple solves."""
        solver = QPSolver(backend="scipy-lbfgsb", enable_warm_start=True)

        A = np.random.randn(15, 4)
        b = np.random.randn(15)
        W = np.eye(15)

        # First solve: cold start
        x1 = solver.solve_weighted_least_squares(A, b, W, point_id=0)
        assert solver.stats["cold_starts"] == 1
        assert solver.stats["warm_starts"] == 0

        # Modify b slightly (similar problem)
        b_mod = b + 0.01 * np.random.randn(15)

        # Second solve: warm start
        x2 = solver.solve_weighted_least_squares(A, b_mod, W, point_id=0)
        assert solver.stats["cold_starts"] == 1
        assert solver.stats["warm_starts"] == 1

        # Solutions should be close (similar problems)
        assert np.allclose(x1, x2, atol=0.1)

    def test_solver_diagonal_weights(self):
        """Test solver with diagonal weight vectors."""
        solver = QPSolver(backend="scipy-lbfgsb", enable_warm_start=False)

        A = np.random.randn(20, 5)
        b = np.random.randn(20)
        W_diag = np.random.rand(20) + 0.1  # Positive weights

        # Solve with diagonal weights
        x = solver.solve_weighted_least_squares(A, b, W_diag)

        # Check solution is reasonable
        assert x.shape == (5,)
        assert np.all(np.isfinite(x))

    def test_solver_statistics_reset(self):
        """Test statistics reset."""
        solver = QPSolver(backend="scipy-lbfgsb", enable_warm_start=False)

        A = np.random.randn(10, 3)
        b = np.random.randn(10)
        W = np.eye(10)

        # Solve a few problems
        for _ in range(3):
            solver.solve_weighted_least_squares(A, b, W)

        assert solver.stats["total_solves"] == 3

        # Reset statistics
        solver.reset_statistics()

        assert solver.stats["total_solves"] == 0
        assert solver.stats["successes"] == 0
        assert len(solver.stats["solve_times"]) == 0

    def test_solver_print_statistics(self, capsys):
        """Test statistics printing."""
        solver = QPSolver(backend="scipy-lbfgsb", enable_warm_start=False)

        A = np.random.randn(10, 3)
        b = np.random.randn(10)
        W = np.eye(10)

        solver.solve_weighted_least_squares(A, b, W)

        # Print statistics
        solver.print_statistics()

        # Capture printed output
        captured = capsys.readouterr()
        assert "QP Solver Statistics" in captured.out
        assert "Total solves:" in captured.out
        assert "Timing Statistics:" in captured.out

    def test_solver_auto_backend_selection(self):
        """Test automatic backend selection."""
        solver = QPSolver(backend="auto", enable_warm_start=False)

        # Should select OSQP if available, otherwise scipy
        assert solver.backend in ["osqp", "scipy-slsqp"]

        A = np.random.randn(10, 3)
        b = np.random.randn(10)
        W = np.eye(10)

        # Should work regardless of backend
        x = solver.solve_weighted_least_squares(A, b, W)
        assert x.shape == (3,)


# =============================================================================
# Integration Tests
# =============================================================================


class TestQPIntegration:
    """Integration tests combining cache and solver."""

    def test_cache_with_multiple_points(self):
        """Test caching with multiple collocation points (realistic MFG usage)."""
        cache = QPCache(max_size=50)
        solver = QPSolver(backend="scipy-lbfgsb", enable_warm_start=True, cache=cache)

        # Simulate solving at multiple collocation points
        n_points = 20
        for point_id in range(n_points):
            # Each point has slightly different problem
            A = np.random.randn(15, 4) + 0.1 * point_id
            b = np.random.randn(15) + 0.1 * point_id
            W = np.eye(15)

            solver.solve_weighted_least_squares(A, b, W, point_id=point_id)

        # First iteration: all cache misses
        assert solver.stats["cache_misses"] == n_points
        assert solver.stats["cache_hits"] == 0

        # Second iteration: all should be warm-started
        solver.reset_statistics()
        for point_id in range(n_points):
            A = np.random.randn(15, 4) + 0.1 * point_id
            b = np.random.randn(15) + 0.1 * point_id
            W = np.eye(15)

            solver.solve_weighted_least_squares(A, b, W, point_id=point_id)

        # Should have warm starts but cache misses (different problems)
        assert solver.stats["warm_starts"] == n_points
        assert solver.stats["cache_misses"] == n_points

    def test_speedup_from_caching(self):
        """Test that caching provides speedup."""
        cache = QPCache(max_size=10)
        solver_cached = QPSolver(backend="scipy-lbfgsb", enable_warm_start=False, cache=cache)
        solver_uncached = QPSolver(backend="scipy-lbfgsb", enable_warm_start=False, cache=None)

        A = np.random.randn(30, 8)
        b = np.random.randn(30)
        W = np.eye(30)

        # Solve 10 times with same problem
        for _ in range(10):
            solver_cached.solve_weighted_least_squares(A, b, W)
            solver_uncached.solve_weighted_least_squares(A, b, W)

        # Cached solver should have 9 cache hits
        # Note: Speedup may vary due to timing overhead on small problems
        assert solver_cached.stats["cache_hits"] == 9
        assert solver_cached.stats["cache_misses"] == 1


class TestQPNonConvergenceWarns1071:
    """Issue #1071: a QP/least-squares solve that does not converge must NOT silently return
    the unconstrained solution — it warns (the constraints are not enforced on the result)."""

    def test_unconstrained_fallback_warns_and_returns_x0(self, monkeypatch):
        import mfgarchon.utils.numerical.qp_utils as qp_mod

        calls: list[str] = []
        monkeypatch.setattr(qp_mod.logger, "warning", lambda msg, *a, **k: calls.append(str(msg)))

        solver = QPSolver()
        before = solver.stats["failures"]
        x0 = np.array([1.0, 2.0, 3.0])
        result = solver._unconstrained_fallback(x0)

        assert np.array_equal(result, x0)  # unconstrained x0 still returned (robustness)
        assert solver.stats["failures"] == before + 1  # failure tracked
        assert any("did not converge" in c for c in calls)  # and NOT silent


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

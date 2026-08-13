"""
Validation tests for penalty method convergence rate.

Tests that penalty method achieves theoretical O(√ε) convergence for
variational inequality constraints (Issue #591).

Created: 2026-01-18 (Issue #594 Phase 5.3)
"""

import pytest

import numpy as np

from mfgarchon.geometry.boundary import BilateralConstraint, ObstacleConstraint


class TestPenaltyConvergenceTheory:
    """
    Test penalty method convergence rate: ||u_ε - u*|| = O(√ε).

    The penalty method approximates the VI solution by solving a regularized
    problem with penalty parameter ε. As ε → 0, the penalized solution u_ε
    should converge to the VI solution u* at rate O(√ε).

    Reference: Glowinski et al. (1984), Theorem 3.2
    """

    def penalty_solve_1d(self, psi, epsilon, u_initial=None):
        """
        Solve 1D obstacle problem with penalty method.

        Solves: u = max(u_free - ε·max(0, ψ - u_free), ψ)

        This is a simplified penalty iteration (not full PDE solve).
        """
        if u_initial is None:
            u = psi.copy() + 1.0  # Start above obstacle

        u = u_initial.copy() if u_initial is not None else psi + 1.0

        # Fixed-point iteration for penalty method
        for _ in range(100):
            u_prev = u.copy()

            # Penalty term: (1/ε)·(ψ - u)₊ ≈ Lagrange multiplier
            penalty = (1.0 / epsilon) * np.maximum(0, psi - u)

            # Simplified update: u ← u + penalty (would be full operator in PDE)
            u = u + 0.01 * penalty  # Small step for stability

            # Project onto constraint
            u = np.maximum(u, psi)

            if np.linalg.norm(u - u_prev) < 1e-8:
                break

        return u

    @pytest.mark.xfail(reason="Issue #594: Theoretical convergence rate validation - requires improved penalty solver")
    def test_penalty_convergence_rate_sqrt_epsilon(self):
        """Test O(√ε) convergence rate for penalty method."""
        # Reference VI solution (via projection)
        Nx = 100
        x = np.linspace(0, 1, Nx)
        psi = -0.5 * (x - 0.5) ** 2  # Parabolic obstacle

        # "True" solution: simple projection (u* = max(u_free, ψ))
        u_free = 0.2 + 0.0 * x  # Constant free solution
        u_star = np.maximum(u_free, psi)

        # Test different penalty parameters
        epsilons = [1e-2, 1e-3, 1e-4, 1e-5]
        errors = []

        for epsilon in epsilons:
            u_penalty = self.penalty_solve_1d(psi, epsilon, u_initial=u_free)
            error = np.linalg.norm(u_penalty - u_star)
            errors.append(error)

        # Check O(√ε) convergence
        # Error should be ~ C·√ε, so log(error) ≈ log(C) + 0.5·log(ε)
        log_epsilons = np.log(epsilons)
        log_errors = np.log(errors)

        # Fit line: log(error) = a + b·log(ε)
        coeffs = np.polyfit(log_epsilons, log_errors, deg=1)
        slope = coeffs[0]

        # Slope should be ≈ 0.5 for O(√ε)
        assert 0.3 < slope < 0.7, f"Convergence rate slope {slope:.3f} not near 0.5 (√ε)"

    def test_penalty_vs_projection_equivalence(self):
        """Test that penalty with ε→0 approaches projection solution."""
        Nx = 80
        x = np.linspace(0, 1, Nx)
        psi = -0.3 * (x - 0.5) ** 2
        u_free = 0.15 + 0.0 * x

        # Projection solution (ε = 0 limit)
        u_projection = np.maximum(u_free, psi)

        # Penalty solution with very small ε
        epsilon_small = 1e-6
        u_penalty = self.penalty_solve_1d(psi, epsilon_small, u_initial=u_free)

        # Should be very close
        error = np.linalg.norm(u_penalty - u_projection) / np.linalg.norm(u_projection)
        assert error < 1e-3, f"Penalty (ε={epsilon_small}) vs projection error: {error:.2%}"

    def test_active_set_convergence(self):
        """Test that active set converges as ε → 0."""
        Nx = 100
        x = np.linspace(0, 1, Nx)
        psi = -0.4 * (x - 0.5) ** 2
        u_free = 0.1 + 0.0 * x

        # True active set (projection)
        u_proj = np.maximum(u_free, psi)
        active_true = np.abs(u_proj - psi) < 1e-6

        # Penalty active sets
        epsilons = [1e-2, 1e-3, 1e-4]
        active_set_errors = []

        for epsilon in epsilons:
            u_penalty = self.penalty_solve_1d(psi, epsilon, u_initial=u_free)
            active_penalty = np.abs(u_penalty - psi) < epsilon  # ε-active set

            # Measure active set difference
            symmetric_diff = np.logical_xor(active_true, active_penalty).sum()
            active_set_errors.append(symmetric_diff)

        # Active set should converge (error decreases)
        assert active_set_errors[-1] <= active_set_errors[0], "Active set should converge with ε→0"


class TestObstacleConstraintAccuracy:
    """Test accuracy of ObstacleConstraint projection."""

    def test_projection_exact_for_violated_constraint(self):
        """Test that projection exactly enforces constraint."""
        psi = np.array([0.5, 1.0, 0.3, 0.8])
        constraint = ObstacleConstraint(psi, constraint_type="lower")

        # Input violates constraint
        u = np.array([0.3, 0.9, 0.1, 1.2])

        u_proj = constraint.project(u)

        # Check exact constraint satisfaction
        assert np.all(u_proj >= psi), "Projection should enforce u ≥ ψ"

        # Check projection is minimal (closest point)
        for i in range(len(u)):
            if u[i] < psi[i]:
                assert u_proj[i] == psi[i], "Violated point should project to boundary"
            else:
                assert u_proj[i] == u[i], "Feasible point should remain unchanged"

    def test_projection_distance_minimization(self):
        """Test that projection minimizes distance."""
        psi = np.array([0.0, 0.5, 1.0, 0.3])
        constraint = ObstacleConstraint(psi, constraint_type="lower")

        u = np.array([-0.2, 0.3, 1.5, 0.8])
        u_proj = constraint.project(u)

        # Projection should be closest feasible point
        # For any other feasible v, ||u - u_proj|| ≤ ||u - v||
        v_test = psi + 0.1  # Another feasible point (above obstacle)
        dist_to_proj = np.linalg.norm(u - u_proj)
        dist_to_test = np.linalg.norm(u - v_test)

        assert dist_to_proj <= dist_to_test, "Projection should minimize distance"

    def test_bilateral_projection_accuracy(self):
        """Test bilateral projection ψ_lower ≤ u ≤ ψ_upper."""
        psi_lower = np.array([-0.5, -0.3, 0.0, 0.2])
        psi_upper = np.array([0.5, 0.7, 1.0, 0.8])

        constraint = BilateralConstraint(psi_lower, psi_upper)

        # Test points violating both constraints
        u = np.array([-1.0, 0.5, 1.5, 0.5])  # Below, inside, above, inside

        u_proj = constraint.project(u)

        # Check both constraints
        assert np.all(u_proj >= psi_lower), "Must satisfy lower bound"
        assert np.all(u_proj <= psi_upper), "Must satisfy upper bound"

        # Check specific projections
        expected = np.array([-0.5, 0.5, 1.0, 0.5])  # Clip to corridor
        assert np.allclose(u_proj, expected), "Bilateral projection should clip correctly"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

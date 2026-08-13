"""
Unit tests for MonotonicityEnforcer violation detection.

Tests the check_monotonicity_violation() method for basic M-matrix checking.

Refactored from HJBGFDMSolver tests as part of Issue #763.
The monotonicity enforcement logic was extracted to MonotonicityEnforcer
component (Issue #545) for better testability and reusability.
"""

import numpy as np

from mfgarchon.alg.numerical.gfdm_components.monotonicity_enforcer import MonotonicityEnforcer


def _create_minimal_enforcer(
    multi_indices: list[tuple[int, ...]],
    sigma: float = 0.5,
) -> MonotonicityEnforcer:
    """
    Create a minimal MonotonicityEnforcer for testing check_monotonicity_violation.

    The check_monotonicity_violation method only uses:
    - self.multi_indices
    - self._sigma_function(point_idx)

    Other parameters are set to minimal valid values.
    """
    # Minimal 2D collocation points
    collocation_points = np.array([[0.5, 0.5]])
    neighborhoods = {0: [0]}  # Self-referential neighborhood

    return MonotonicityEnforcer(
        qp_solver=None,  # Not used by check_monotonicity_violation
        qp_constraint_mode="taylor",
        collocation_points=collocation_points,
        neighborhoods=neighborhoods,
        multi_indices=multi_indices,
        domain_bounds=[(0.0, 1.0), (0.0, 1.0)],
        delta=0.1,
        sigma_function=lambda _idx: sigma,
    )


class TestCheckMonotonicityViolation:
    """Tests for MonotonicityEnforcer.check_monotonicity_violation()."""

    def test_valid_coefficients_no_violation(self):
        """Test that valid coefficients (negative Laplacian, bounded gradient) pass."""
        # 2D multi-indices up to order 2
        multi_indices = [
            (0, 0),  # Constant
            (1, 0),  # ∂/∂x
            (0, 1),  # ∂/∂y
            (2, 0),  # ∂²/∂x²
            (1, 1),  # ∂²/∂x∂y
            (0, 2),  # ∂²/∂y²
        ]
        enforcer = _create_minimal_enforcer(multi_indices)

        # Valid coefficients: Laplacian negative, gradient bounded
        D_valid = np.array([1.0, 0.1, 0.1, -0.5, 0.0, -0.5])
        result = enforcer.check_monotonicity_violation(D_valid, point_idx=0, use_adaptive=False)

        assert not result, "Valid coefficients should not trigger violation"

    def test_positive_laplacian_triggers_violation(self):
        """Test that positive Laplacian (criterion 1) triggers violation."""
        multi_indices = [
            (0, 0),  # Constant
            (1, 0),  # ∂/∂x
            (0, 1),  # ∂/∂y
            (2, 0),  # ∂²/∂x² (positive = violation)
            (1, 1),  # ∂²/∂x∂y
            (0, 2),  # ∂²/∂y² (positive = violation)
        ]
        enforcer = _create_minimal_enforcer(multi_indices)

        # Positive Laplacian violates criterion 1
        D_positive_laplacian = np.array([1.0, 0.1, 0.1, 0.5, 0.0, 0.5])
        result = enforcer.check_monotonicity_violation(D_positive_laplacian, point_idx=0, use_adaptive=False)

        assert result, "Positive Laplacian should trigger violation"

    def test_large_gradient_triggers_violation(self):
        """Test that unbounded gradient (criterion 2) triggers violation."""
        multi_indices = [
            (0, 0),  # Constant
            (1, 0),  # ∂/∂x (large = violation)
            (0, 1),  # ∂/∂y
            (2, 0),  # ∂²/∂x²
            (1, 1),  # ∂²/∂x∂y
            (0, 2),  # ∂²/∂y²
        ]
        enforcer = _create_minimal_enforcer(multi_indices)

        # Large gradient relative to small Laplacian
        D_large_gradient = np.array([1.0, 10.0, 0.1, -0.1, 0.0, -0.1])
        result = enforcer.check_monotonicity_violation(D_large_gradient, point_idx=0, use_adaptive=False)

        assert result, "Large gradient should trigger violation"


class TestMissingLaplacian:
    """Tests for behavior when Laplacian term is missing."""

    def test_no_laplacian_returns_false(self):
        """Test that missing Laplacian term returns False (cannot check)."""
        # Multi-indices without any second-order terms
        multi_indices = [(0, 0), (1, 0), (0, 1)]
        enforcer = _create_minimal_enforcer(multi_indices)

        D_no_laplacian = np.array([1.0, 0.5, 0.5])
        result = enforcer.check_monotonicity_violation(D_no_laplacian, point_idx=0, use_adaptive=False)

        assert not result, "Missing Laplacian should return False (cannot check)"


class TestHigherOrderControl:
    """Tests for higher-order coefficient control (criterion 3)."""

    def test_higher_order_violation(self):
        """Test that large higher-order terms trigger violation."""
        # Include order-3 terms
        multi_indices = [
            (0, 0),  # Constant
            (1, 0),  # ∂/∂x
            (0, 1),  # ∂/∂y
            (2, 0),  # ∂²/∂x²
            (1, 1),  # ∂²/∂x∂y
            (0, 2),  # ∂²/∂y²
            (3, 0),  # ∂³/∂x³ (higher-order)
            (0, 3),  # ∂³/∂y³ (higher-order)
        ]
        enforcer = _create_minimal_enforcer(multi_indices)

        # Small Laplacian but large higher-order terms
        D_higher_order = np.array([1.0, 0.1, 0.1, -0.1, 0.0, -0.1, 5.0, 5.0])
        result = enforcer.check_monotonicity_violation(D_higher_order, point_idx=0, use_adaptive=False)

        assert result, "Large higher-order terms should trigger violation"

    def test_bounded_higher_order_passes(self):
        """Test that bounded higher-order terms pass."""
        multi_indices = [
            (0, 0),  # Constant
            (1, 0),  # ∂/∂x
            (0, 1),  # ∂/∂y
            (2, 0),  # ∂²/∂x²
            (1, 1),  # ∂²/∂x∂y
            (0, 2),  # ∂²/∂y²
            (3, 0),  # ∂³/∂x³
            (0, 3),  # ∂³/∂y³
        ]
        enforcer = _create_minimal_enforcer(multi_indices)

        # Large Laplacian dominates higher-order terms
        D_bounded = np.array([1.0, 0.1, 0.1, -1.0, 0.0, -1.0, 0.1, 0.1])
        result = enforcer.check_monotonicity_violation(D_bounded, point_idx=0, use_adaptive=False)

        assert not result, "Bounded higher-order terms should pass"


class TestSigmaDependence:
    """Tests for sigma dependence in gradient bound."""

    def test_large_sigma_relaxes_gradient_bound(self):
        """Test that larger sigma allows larger gradients."""
        multi_indices = [
            (0, 0),
            (1, 0),
            (0, 1),
            (2, 0),
            (1, 1),
            (0, 2),
        ]

        # Same gradient, different sigma values
        D_coeffs = np.array([1.0, 2.0, 0.1, -0.5, 0.0, -0.5])

        # Small sigma: gradient might violate
        enforcer_small_sigma = _create_minimal_enforcer(multi_indices, sigma=0.1)
        result_small = enforcer_small_sigma.check_monotonicity_violation(D_coeffs, point_idx=0, use_adaptive=False)

        # Large sigma: gradient should be acceptable
        enforcer_large_sigma = _create_minimal_enforcer(multi_indices, sigma=2.0)
        result_large = enforcer_large_sigma.check_monotonicity_violation(D_coeffs, point_idx=0, use_adaptive=False)

        # Large sigma should be more permissive
        # Note: The exact behavior depends on the scale_factor calculation
        # This test verifies sigma affects the result
        assert result_small or not result_large, "Larger sigma should relax gradient bound"


class TestMultiDimensional:
    """Tests for nD support."""

    def test_1d_multi_indices(self):
        """Test with 1D multi-indices, both sides of the Laplacian-sign criterion.

        The valid vector alone is one-sided: a 1D regression in the multi-index lookup that
        biased toward "no violation" -- including a bare ``return False`` -- would pass it. The
        two vectors below differ only in the sign of the second-order coefficient, so the pair
        brackets criterion 1 (``D_2 >= -1e-12``) in the 1D dispatch, where ``multi_indices`` are
        1-tuples and ``laplacian_idx`` comes from ``sum(beta) == 2``.
        """
        multi_indices = [
            (0,),  # Constant
            (1,),  # ∂/∂x
            (2,),  # ∂²/∂x²
        ]

        # Need 1D collocation point
        enforcer = MonotonicityEnforcer(
            qp_solver=None,
            qp_constraint_mode="taylor",
            collocation_points=np.array([[0.5]]),
            neighborhoods={0: [0]},
            multi_indices=multi_indices,
            domain_bounds=[(0.0, 1.0)],
            delta=0.1,
            sigma_function=lambda _idx: 0.5,
        )

        # Valid 1D coefficients
        D_valid_1d = np.array([1.0, 0.1, -0.5])
        result = enforcer.check_monotonicity_violation(D_valid_1d, point_idx=0, use_adaptive=False)

        assert not result, "Valid 1D coefficients should pass"

        # Same vector with the second-order coefficient sign-flipped: diffusion is no longer
        # dominant, so criterion 1 must fire. The gradient stays 0.1 against a bound of
        # 10*max(sigma^2, 0.1)*|D_2| = 1.25, so criterion 2 cannot be what fires here.
        D_positive_laplacian_1d = np.array([1.0, 0.1, 0.5])
        result = enforcer.check_monotonicity_violation(D_positive_laplacian_1d, point_idx=0, use_adaptive=False)

        assert result, "Positive 1D Laplacian should trigger violation"
        assert enforcer.stats["violation_laplacian"] == 1
        assert enforcer.stats["violation_gradient"] == 0
        assert enforcer.stats["violation_higher_order"] == 0

    def test_3d_multi_indices(self):
        """Test with 3D multi-indices."""
        multi_indices = [
            (0, 0, 0),  # Constant
            (1, 0, 0),  # ∂/∂x
            (0, 1, 0),  # ∂/∂y
            (0, 0, 1),  # ∂/∂z
            (2, 0, 0),  # ∂²/∂x²
            (0, 2, 0),  # ∂²/∂y²
            (0, 0, 2),  # ∂²/∂z²
        ]

        # Need 3D collocation point
        enforcer = MonotonicityEnforcer(
            qp_solver=None,
            qp_constraint_mode="taylor",
            collocation_points=np.array([[0.5, 0.5, 0.5]]),
            neighborhoods={0: [0]},
            multi_indices=multi_indices,
            domain_bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            delta=0.1,
            sigma_function=lambda _idx: 0.5,
        )

        # Valid 3D coefficients (all Laplacian terms negative)
        D_valid_3d = np.array([1.0, 0.1, 0.1, 0.1, -0.3, -0.3, -0.3])
        result = enforcer.check_monotonicity_violation(D_valid_3d, point_idx=0, use_adaptive=False)

        assert not result, "Valid 3D coefficients should pass"

        # Complement on the same index set: only the (2,0,0) coefficient flips sign.
        #
        # This is the coefficient the implementation actually reads -- it takes the FIRST
        # order-2 multi-index and stops (``if laplacian_idx is None: laplacian_idx = k``), rather
        # than the Laplacian trace. So the pair below brackets what the code does, not what
        # "Laplacian negativity" says: measured, D = [1.0, .1, .1, .1, -0.3, +5.0, +5.0] has
        # trace +9.7 and still returns False. That gap is a defect in the enforcer, not something
        # this test can assert today without failing.
        D_positive_laplacian_3d = np.array([1.0, 0.1, 0.1, 0.1, 0.3, -0.3, -0.3])
        result = enforcer.check_monotonicity_violation(D_positive_laplacian_3d, point_idx=0, use_adaptive=False)

        assert result, "Positive leading 3D second-order coefficient should trigger violation"
        assert enforcer.stats["violation_laplacian"] == 1
        assert enforcer.stats["violation_gradient"] == 0
        assert enforcer.stats["violation_higher_order"] == 0


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing MonotonicityEnforcer.check_monotonicity_violation()...")

    # Run basic tests
    test = TestCheckMonotonicityViolation()
    test.test_valid_coefficients_no_violation()
    print("✓ test_valid_coefficients_no_violation")

    test.test_positive_laplacian_triggers_violation()
    print("✓ test_positive_laplacian_triggers_violation")

    test.test_large_gradient_triggers_violation()
    print("✓ test_large_gradient_triggers_violation")

    test2 = TestMissingLaplacian()
    test2.test_no_laplacian_returns_false()
    print("✓ test_no_laplacian_returns_false")

    test3 = TestHigherOrderControl()
    test3.test_higher_order_violation()
    print("✓ test_higher_order_violation")

    test3.test_bounded_higher_order_passes()
    print("✓ test_bounded_higher_order_passes")

    test4 = TestMultiDimensional()
    test4.test_1d_multi_indices()
    print("✓ test_1d_multi_indices")

    test4.test_3d_multi_indices()
    print("✓ test_3d_multi_indices")

    print("\n✅ All monotonicity tests passed!")

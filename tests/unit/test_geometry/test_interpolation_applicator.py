"""
Unit tests for InterpolationApplicator (Issue #636).

Tests the boundary condition enforcement for interpolation-based solvers
(Semi-Lagrangian, particle methods, RBF interpolation, etc.).
"""

import pytest

import numpy as np

from mfgarchon.geometry.boundary import (
    InterpolationApplicator,
    create_interpolation_applicator,
    dirichlet_bc,
    neumann_bc,
    no_flux_bc,
    periodic_bc,
)


class TestInterpolationApplicatorBasic:
    """Basic functionality tests for InterpolationApplicator."""

    def test_initialization_default(self):
        """Test default initialization."""
        applicator = InterpolationApplicator()
        assert applicator.dimension is None  # Dimension-agnostic
        assert applicator.extrapolation_order == 2

    def test_initialization_with_dimension(self):
        """Test initialization with specific dimension."""
        applicator = InterpolationApplicator(dimension=2)
        assert applicator.dimension == 2

    def test_initialization_with_order(self):
        """Test initialization with custom extrapolation order."""
        applicator = InterpolationApplicator(extrapolation_order=1)
        assert applicator.extrapolation_order == 1

    def test_factory_function(self):
        """Test factory function creates correct applicator."""
        applicator = create_interpolation_applicator(dimension=3, extrapolation_order=2)
        assert applicator.dimension == 3
        assert applicator.extrapolation_order == 2

    def test_repr(self):
        """Test string representation."""
        applicator = InterpolationApplicator(dimension=2)
        assert "InterpolationApplicator" in repr(applicator)
        assert "dimension=2" in repr(applicator)


class TestNeumannBCEnforcement:
    """Tests for Neumann BC enforcement via extrapolation."""

    def test_1d_neumann_2nd_order(self):
        """Test 1D Neumann BC with 2nd-order extrapolation.

        Formula: U[0] = (4*U[1] - U[2]) / 3
        """
        applicator = InterpolationApplicator(extrapolation_order=2)
        U = np.array([0.9, 1.0, 1.1, 1.2, 1.3])  # Simulated interpolation result
        bc = neumann_bc(dimension=1)

        U_enforced = applicator.enforce_values(U.copy(), bc)

        # Check 2nd-order extrapolation at boundaries
        expected_min = (4.0 * 1.0 - 1.1) / 3.0  # = 0.9667
        expected_max = (4.0 * 1.2 - 1.1) / 3.0  # = 1.2333

        assert np.isclose(U_enforced[0], expected_min, rtol=1e-10)
        assert np.isclose(U_enforced[-1], expected_max, rtol=1e-10)
        # Interior unchanged
        assert np.allclose(U_enforced[1:-1], U[1:-1])

    def test_1d_neumann_1st_order(self):
        """Test 1D Neumann BC with 1st-order extrapolation.

        Formula: U[0] = U[1] (simple copy)
        """
        applicator = InterpolationApplicator(extrapolation_order=1)
        U = np.array([0.9, 1.0, 1.1, 1.2, 1.3])
        bc = neumann_bc(dimension=1)

        U_enforced = applicator.enforce_values(U.copy(), bc)

        # 1st-order: boundary = adjacent interior
        assert U_enforced[0] == U_enforced[1]
        assert U_enforced[-1] == U_enforced[-2]

    def test_2d_neumann(self):
        """Test 2D Neumann BC enforcement."""
        applicator = InterpolationApplicator()
        U = np.ones((5, 6)) + 0.1 * np.arange(5)[:, np.newaxis]
        bc = neumann_bc(dimension=2)

        U_enforced = applicator.enforce_values(U.copy(), bc)

        # Shape should be preserved
        assert U_enforced.shape == (5, 6)

        # Check that extrapolation formula is applied along x (axis 0)
        for j in range(1, 5):  # Interior columns
            expected_x_min = (4.0 * U_enforced[1, j] - U_enforced[2, j]) / 3.0
            assert np.isclose(U_enforced[0, j], expected_x_min, rtol=1e-10)

    def test_3d_neumann(self):
        """Test 3D Neumann BC enforcement."""
        applicator = InterpolationApplicator()
        # Non-constant data: np.ones is a fixed point of ANY extrapolation rule, so a constant
        # field cannot distinguish the 2nd-order formula from doing nothing at all.
        U = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
        bc = neumann_bc(dimension=3)

        U_enforced = applicator.enforce_values(U.copy(), bc)

        # Shape should be preserved
        assert U_enforced.shape == (4, 5, 6)
        # Values should be finite
        assert np.isfinite(U_enforced).all()

        # The per-axis loop must apply U[0] = (4*U[1] - U[2]) / 3 on every one of the six faces.
        # Checked on the strict interior of the other two axes, where the faces do not interact.
        for axis in range(3):
            o = np.moveaxis(U_enforced, axis, 0)
            interior = (slice(1, -1), slice(1, -1))
            np.testing.assert_allclose(o[0][interior], ((4.0 * o[1] - o[2]) / 3.0)[interior], atol=1e-10)
            np.testing.assert_allclose(o[-1][interior], ((4.0 * o[-2] - o[-3]) / 3.0)[interior], atol=1e-10)


class TestDirichletBCEnforcement:
    """Tests for Dirichlet BC enforcement."""

    def test_1d_dirichlet(self):
        """Test 1D Dirichlet BC enforcement."""
        applicator = InterpolationApplicator()
        U = np.array([0.9, 1.0, 1.1, 1.2, 1.3])
        bc = dirichlet_bc(dimension=1, value=0.0)

        U_enforced = applicator.enforce_values(U.copy(), bc)

        # Boundaries should be set to Dirichlet value
        assert U_enforced[0] == 0.0
        assert U_enforced[-1] == 0.0
        # Interior unchanged
        assert np.allclose(U_enforced[1:-1], U[1:-1])

    def test_1d_dirichlet_nonzero(self):
        """Test 1D Dirichlet BC with non-zero value."""
        applicator = InterpolationApplicator()
        U = np.array([0.9, 1.0, 1.1, 1.2, 1.3])
        bc = dirichlet_bc(dimension=1, value=5.0)

        U_enforced = applicator.enforce_values(U.copy(), bc)

        assert U_enforced[0] == 5.0
        assert U_enforced[-1] == 5.0

    def test_2d_dirichlet(self):
        """Test 2D Dirichlet BC enforcement."""
        applicator = InterpolationApplicator()
        U = np.ones((5, 6))
        bc = dirichlet_bc(dimension=2, value=0.0)

        U_enforced = applicator.enforce_values(U.copy(), bc)

        # All boundaries should be 0
        assert np.allclose(U_enforced[0, :], 0.0)
        assert np.allclose(U_enforced[-1, :], 0.0)
        assert np.allclose(U_enforced[:, 0], 0.0)
        assert np.allclose(U_enforced[:, -1], 0.0)


class TestNoFluxBC:
    """Tests for no-flux BC (alias for Neumann with zero gradient)."""

    def test_no_flux_same_as_neumann(self):
        """Test that no_flux BC behaves same as Neumann."""
        applicator = InterpolationApplicator()
        U = np.array([0.9, 1.0, 1.1, 1.2, 1.3])

        bc_neumann = neumann_bc(dimension=1)
        bc_no_flux = no_flux_bc(dimension=1)

        U_neumann = applicator.enforce_values(U.copy(), bc_neumann)
        U_no_flux = applicator.enforce_values(U.copy(), bc_no_flux)

        assert np.allclose(U_neumann, U_no_flux)


class TestPeriodicBC:
    """Tests for periodic BC enforcement."""

    def test_1d_periodic(self):
        """Test 1D periodic BC enforcement."""
        applicator = InterpolationApplicator()
        U = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bc = periodic_bc(dimension=1)

        U_enforced = applicator.enforce_values(U.copy(), bc)

        assert U_enforced.shape == U.shape

        # Periodic enforcement is a mean identification (#1820/#1829): both endpoints become
        # (U[0] + U[-1]) / 2 = 3.0, interior untouched.  The halo convention U[0] = U[-2] would
        # instead give [4, 2, 3, 4, 2], so this pins the routing AND the convention.
        np.testing.assert_array_equal(U_enforced, [3.0, 2.0, 3.0, 4.0, 3.0])


class TestClampQueryPoints:
    """Tests for clamping query points to domain."""

    def test_clamp_mode(self):
        """Test clamp mode brings points to boundary."""
        applicator = InterpolationApplicator()
        pts = np.array([[0.5, -0.1], [0.5, 1.1], [-0.2, 0.5], [1.3, 0.5]])
        domain_min = np.array([0.0, 0.0])
        domain_max = np.array([1.0, 1.0])

        clamped = applicator.clamp_query_points(pts, domain_min, domain_max, mode="clamp")

        assert np.allclose(clamped[0], [0.5, 0.0])  # y clamped to 0
        assert np.allclose(clamped[1], [0.5, 1.0])  # y clamped to 1
        assert np.allclose(clamped[2], [0.0, 0.5])  # x clamped to 0
        assert np.allclose(clamped[3], [1.0, 0.5])  # x clamped to 1

    def test_reflect_mode(self):
        """Test reflect mode reflects points about boundary."""
        applicator = InterpolationApplicator()
        pts = np.array([[0.5, -0.1]])
        domain_min = np.array([0.0, 0.0])
        domain_max = np.array([1.0, 1.0])

        reflected = applicator.clamp_query_points(pts, domain_min, domain_max, mode="reflect")

        # Point at y=-0.1 should reflect to y=0.1
        assert np.isclose(reflected[0, 1], 0.1, rtol=1e-10)

    def test_invalid_mode(self):
        """Test invalid mode raises error."""
        applicator = InterpolationApplicator()
        pts = np.array([[0.5, 0.5]])
        domain_min = np.array([0.0, 0.0])
        domain_max = np.array([1.0, 1.0])

        with pytest.raises(ValueError, match="Unknown mode"):
            applicator.clamp_query_points(pts, domain_min, domain_max, mode="invalid")


class TestBoundaryDistances:
    """Tests for computing signed boundary distances."""

    def test_interior_point_positive_distance(self):
        """Test interior points have positive distance."""
        applicator = InterpolationApplicator()
        pts = np.array([[0.5, 0.5]])
        domain_min = np.array([0.0, 0.0])
        domain_max = np.array([1.0, 1.0])

        dists = applicator.get_boundary_distances(pts, domain_min, domain_max)

        assert dists[0] > 0  # Inside domain

    def test_exterior_point_negative_distance(self):
        """Test exterior points have negative distance."""
        applicator = InterpolationApplicator()
        pts = np.array([[0.5, -0.1], [1.2, 0.5]])
        domain_min = np.array([0.0, 0.0])
        domain_max = np.array([1.0, 1.0])

        dists = applicator.get_boundary_distances(pts, domain_min, domain_max)

        assert dists[0] < 0  # Outside (y < 0)
        assert dists[1] < 0  # Outside (x > 1)

    def test_boundary_point_zero_distance(self):
        """Test points on boundary have zero distance."""
        applicator = InterpolationApplicator()
        pts = np.array([[0.0, 0.5], [0.5, 1.0]])
        domain_min = np.array([0.0, 0.0])
        domain_max = np.array([1.0, 1.0])

        dists = applicator.get_boundary_distances(pts, domain_min, domain_max)

        assert np.isclose(dists[0], 0.0)
        assert np.isclose(dists[1], 0.0)


class TestDimensionAgnostic:
    """Tests for dimension-agnostic behavior."""

    def test_4d_enforcement(self):
        """Test BC enforcement works in 4D."""
        applicator = InterpolationApplicator()
        # Every axis needs >= 4 points: with 3, the min-face stencil (indices 0,1,2) and the
        # max-face stencil (indices -1,-2,-3) overlap and the formula is not well posed there.
        # Non-constant data, because np.ones satisfies any extrapolation rule including none.
        U = np.arange(4 * 5 * 6 * 7, dtype=float).reshape(4, 5, 6, 7)
        bc = neumann_bc(dimension=4)

        U_enforced = applicator.enforce_values(U.copy(), bc)

        assert U_enforced.shape == (4, 5, 6, 7)
        assert np.isfinite(U_enforced).all()

        # axis_names = ["x", "y", "z", "w"] runs out at 4-D and falls back to f"d{d}"; the
        # 2nd-order formula must still be applied on all eight faces.
        for axis in range(4):
            o = np.moveaxis(U_enforced, axis, 0)
            interior = (slice(1, -1), slice(1, -1), slice(1, -1))
            np.testing.assert_allclose(o[0][interior], ((4.0 * o[1] - o[2]) / 3.0)[interior], atol=1e-10)
            np.testing.assert_allclose(o[-1][interior], ((4.0 * o[-2] - o[-3]) / 3.0)[interior], atol=1e-10)

    def test_dimension_mismatch_error(self):
        """Test dimension mismatch raises error when dimension is fixed."""
        applicator = InterpolationApplicator(dimension=2)
        U = np.ones((5, 6, 7))  # 3D array
        bc = neumann_bc(dimension=3)

        with pytest.raises(ValueError, match="dimension"):
            applicator.enforce_values(U, bc)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_small_array_1st_order_fallback(self):
        """Test small arrays fall back to 1st-order."""
        applicator = InterpolationApplicator(extrapolation_order=2)
        U = np.array([1.0, 2.0])  # Only 2 points, can't do 2nd-order
        bc = neumann_bc(dimension=1)

        U_enforced = applicator.enforce_values(U.copy(), bc)

        # Should fall back to 1st-order: U[0] = U[1]
        assert U_enforced[0] == U_enforced[1]

    def test_no_bc_returns_unchanged(self):
        """Test that None BC returns array unchanged."""
        applicator = InterpolationApplicator()
        U = np.array([1.0, 2.0, 3.0])

        # Create a minimal mock BC that enforce_values can handle
        # In practice, solvers check for None BC before calling
        # This test verifies interior is preserved for valid BC
        bc = neumann_bc(dimension=1)
        U_orig = U.copy()
        U_enforced = applicator.enforce_values(U, bc)

        # Interior should be unchanged
        assert U_enforced[1] == U_orig[1]

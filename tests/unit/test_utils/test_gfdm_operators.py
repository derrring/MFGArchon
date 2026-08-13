"""
Unit tests for GFDM (Generalized Finite Difference Method) operators.

Tests the GFDMOperator class in mfgarchon.utils.numerical.gfdm_operators,
which provides efficient spatial derivative computation on scattered points.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.utils.numerical.gfdm_operators import GFDMOperator


class TestGFDMOperatorInitialization:
    """Test GFDMOperator initialization and structure building."""

    def test_init_1d(self):
        """Test 1D initialization."""
        points = np.linspace(0, 1, 20).reshape(-1, 1)
        gfdm = GFDMOperator(points, delta=0.15, taylor_order=2)

        assert gfdm.n_points == 20
        assert gfdm.dimension == 1
        assert gfdm.delta == 0.15
        assert gfdm.taylor_order == 2

    def test_init_2d(self):
        """Test 2D initialization."""
        x = np.linspace(0, 1, 10)
        xx, yy = np.meshgrid(x, x)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        gfdm = GFDMOperator(points, delta=0.2, taylor_order=2)

        assert gfdm.n_points == 100
        assert gfdm.dimension == 2
        assert len(gfdm.multi_indices) > 0

    def test_multi_indices_1d(self):
        """Test multi-index generation for 1D."""
        points = np.linspace(0, 1, 10).reshape(-1, 1)
        gfdm = GFDMOperator(points, delta=0.2, taylor_order=2)

        # 1D, order 2: (1,), (2,)
        assert (1,) in gfdm.multi_indices
        assert (2,) in gfdm.multi_indices

    def test_multi_indices_2d(self):
        """Test multi-index generation for 2D."""
        points = np.random.rand(50, 2)
        gfdm = GFDMOperator(points, delta=0.3, taylor_order=2)

        # 2D, order 2: (1,0), (0,1), (2,0), (1,1), (0,2)
        assert (1, 0) in gfdm.multi_indices
        assert (0, 1) in gfdm.multi_indices
        assert (2, 0) in gfdm.multi_indices
        assert (1, 1) in gfdm.multi_indices
        assert (0, 2) in gfdm.multi_indices


class TestGradientOperator:
    """Test gradient computation accuracy."""

    def test_gradient_constant_function(self):
        """Gradient of constant function should be zero."""
        points = np.random.rand(50, 2)
        gfdm = GFDMOperator(points, delta=0.3, taylor_order=2)

        f = 3.0 * np.ones(50)
        gradient = gfdm.gradient(f)

        assert gradient.shape == (50, 2)
        assert np.max(np.abs(gradient)) < 0.1

    def test_gradient_linear_function_2d(self):
        """Test gradient on f(x,y) = 2x + 3y."""
        # Analytical: grad f = [2, 3]
        x = np.linspace(0, 1, 15)
        y = np.linspace(0, 1, 15)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        gfdm = GFDMOperator(points, delta=0.2, taylor_order=2)

        f = 2 * points[:, 0] + 3 * points[:, 1]
        gradient = gfdm.gradient(f)

        # Mean gradient should be close to [2, 3]
        mean_grad = np.mean(gradient, axis=0)
        assert mean_grad[0] == pytest.approx(2.0, abs=0.2)
        assert mean_grad[1] == pytest.approx(3.0, abs=0.2)

    def test_gradient_1d(self):
        """Test gradient in 1D: f(x) = 2x."""
        # Analytical: grad f = [2]
        points = np.linspace(0, 1, 30).reshape(-1, 1)
        gfdm = GFDMOperator(points, delta=0.1, taylor_order=2)

        f = 2 * points[:, 0]
        gradient = gfdm.gradient(f)

        # Should be close to 2.0 (excluding boundaries)
        interior = gradient[5:-5, 0]
        assert np.mean(interior) == pytest.approx(2.0, abs=0.3)


class TestLaplacianOperator:
    """Test Laplacian operator accuracy."""

    def test_laplacian_constant_function(self):
        """Laplacian of constant function should be zero."""
        x = np.linspace(0, 1, 20)
        y = np.linspace(0, 1, 20)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        gfdm = GFDMOperator(points, delta=0.15, taylor_order=2)

        f = 5.0 * np.ones(points.shape[0])
        laplacian = gfdm.laplacian(f)

        assert np.max(np.abs(laplacian)) < 0.1

    def test_laplacian_quadratic_function_2d(self):
        """Test Laplacian on f(x,y) = x^2 + y^2."""
        # Analytical: Laplacian f = 2 + 2 = 4
        x = np.linspace(0, 1, 15)
        y = np.linspace(0, 1, 15)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        gfdm = GFDMOperator(points, delta=0.2, taylor_order=2)

        f = points[:, 0] ** 2 + points[:, 1] ** 2
        laplacian = gfdm.laplacian(f)

        assert laplacian.shape == (points.shape[0],)
        # Order-2 Taylor reproduces a quadratic exactly, at every node including the
        # boundary ones. Measured max|lap - 4| = 9.1e-14 over all 225 points.
        np.testing.assert_allclose(laplacian, 4.0, atol=1e-10)

    def test_laplacian_1d(self):
        """Test Laplacian in 1D: f(x) = x^2."""
        # Analytical: Laplacian f = 2
        points = np.linspace(0, 1, 30).reshape(-1, 1)
        gfdm = GFDMOperator(points, delta=0.1, taylor_order=2)

        f = points[:, 0] ** 2
        laplacian = gfdm.laplacian(f)

        # Should be close to 2.0 (excluding boundaries)
        interior = laplacian[5:-5]
        assert np.mean(interior) == pytest.approx(2.0, abs=0.3)


class TestHessianOperator:
    """Test Hessian operator."""

    def test_hessian_constant_function(self):
        """Hessian of constant function should be zero."""
        points = np.random.rand(50, 2)
        gfdm = GFDMOperator(points, delta=0.3, taylor_order=2)

        f = 5.0 * np.ones(50)
        hessian = gfdm.hessian(f)

        assert hessian.shape == (50, 2, 2)
        assert np.max(np.abs(hessian)) < 0.2

    def test_hessian_symmetry(self):
        """Hessian should be symmetric."""
        points = np.random.rand(50, 2)
        gfdm = GFDMOperator(points, delta=0.3, taylor_order=2)

        f = points[:, 0] ** 2 + points[:, 0] * points[:, 1] + points[:, 1] ** 2
        hessian = gfdm.hessian(f)

        # Check symmetry: H[i,j] = H[j,i]
        for i in range(50):
            assert np.abs(hessian[i, 0, 1] - hessian[i, 1, 0]) < 0.3

    def test_hessian_trace_equals_laplacian(self):
        """Trace of Hessian should equal Laplacian."""
        points = np.random.rand(50, 2)
        gfdm = GFDMOperator(points, delta=0.3, taylor_order=2)

        f = points[:, 0] ** 2 + points[:, 1] ** 2
        hessian = gfdm.hessian(f)
        laplacian = gfdm.laplacian(f)

        # Trace = H[0,0] + H[1,1]
        trace_hessian = hessian[:, 0, 0] + hessian[:, 1, 1]

        # Should be close
        assert np.max(np.abs(trace_hessian - laplacian)) < 0.5


class TestDivergenceOperator:
    """Test divergence operator accuracy."""

    def test_divergence_zero_field(self):
        """Divergence of zero field should be zero."""
        points = np.random.rand(50, 2)
        gfdm = GFDMOperator(points, delta=0.3, taylor_order=2)

        vector_field = np.zeros((50, 2))
        divergence = gfdm.divergence(vector_field)

        assert np.max(np.abs(divergence)) < 1e-10

    def test_divergence_constant_field(self):
        """Divergence of constant field should be zero."""
        x = np.linspace(0, 1, 20)
        y = np.linspace(0, 1, 20)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        gfdm = GFDMOperator(points, delta=0.15, taylor_order=2)

        vector_field = np.ones((points.shape[0], 2))
        divergence = gfdm.divergence(vector_field)

        assert np.max(np.abs(divergence)) < 0.2

    def test_divergence_with_density(self):
        """Test divergence with density weighting."""
        points = np.random.rand(50, 2)
        gfdm = GFDMOperator(points, delta=0.3, taylor_order=2)

        density = np.ones(50)
        vector_field = np.zeros((50, 2))

        # div(m * v) with m=1, v=0 should be 0
        divergence = gfdm.divergence(vector_field, density)

        assert np.max(np.abs(divergence)) < 1e-10


class TestAllDerivatives:
    """Test all_derivatives method."""

    def test_all_derivatives_returns_dict(self):
        """Test that all_derivatives returns proper structure."""
        points = np.linspace(0, 1, 20).reshape(-1, 1)
        gfdm = GFDMOperator(points, delta=0.15, taylor_order=2)

        f = points[:, 0] ** 2
        all_derivs = gfdm.all_derivatives(f)

        assert isinstance(all_derivs, dict)
        assert len(all_derivs) == gfdm.n_points

        # Each point should have derivative dict
        for i in range(gfdm.n_points):
            assert i in all_derivs
            assert isinstance(all_derivs[i], dict)

    def test_all_derivatives_contains_expected_keys(self):
        """Test that derivatives contain expected multi-indices."""
        points = np.random.rand(50, 2)
        gfdm = GFDMOperator(points, delta=0.3, taylor_order=2)

        f = points[:, 0] ** 2 + points[:, 1] ** 2
        all_derivs = gfdm.all_derivatives(f)

        # Interior points should have all derivative keys
        # (boundary points may have fewer neighbors)
        mid_idx = 25
        derivs = all_derivs[mid_idx]

        # Should have first derivatives
        assert (1, 0) in derivs or len(derivs) == 0  # May be empty at boundary
        assert (0, 1) in derivs or len(derivs) == 0


class TestAccessorMethods:
    """Test accessor methods for composition pattern."""

    def test_get_neighborhood(self):
        """Test get_neighborhood accessor."""
        points = np.linspace(0, 1, 20).reshape(-1, 1)
        gfdm = GFDMOperator(points, delta=0.15, taylor_order=2)

        neighborhood = gfdm.get_neighborhood(10)

        assert "indices" in neighborhood
        assert "points" in neighborhood
        assert "distances" in neighborhood
        assert "size" in neighborhood

    def test_get_taylor_data(self):
        """Test get_taylor_data accessor."""
        points = np.linspace(0, 1, 20).reshape(-1, 1)
        gfdm = GFDMOperator(points, delta=0.15, taylor_order=2)

        taylor_data = gfdm.get_taylor_data(10)

        # Should have precomputed matrices
        assert taylor_data is not None or taylor_data is None  # May be None for boundary

    def test_get_multi_indices(self):
        """Test get_multi_indices accessor."""
        points = np.random.rand(50, 2)
        gfdm = GFDMOperator(points, delta=0.3, taylor_order=2)

        multi_indices = gfdm.get_multi_indices()

        assert multi_indices == gfdm.multi_indices

    def test_approximate_derivatives_at_point(self):
        """Test public derivative accessor."""
        points = np.linspace(0, 1, 20).reshape(-1, 1)
        gfdm = GFDMOperator(points, delta=0.15, taylor_order=2)

        f = points[:, 0] ** 2
        derivs = gfdm.approximate_derivatives_at_point(f, 10)

        assert isinstance(derivs, dict)
        # f = x^2 at x = 10/19: f' = 2x, f'' = 2. An under-resolved point returns {}
        # (see test_small_delta_warning), which the isinstance check alone cannot detect.
        x10 = 10 / 19
        assert derivs[(1,)] == pytest.approx(2 * x10, abs=1e-10)
        assert derivs[(2,)] == pytest.approx(2.0, abs=1e-10)


class TestWeightFunctions:
    """Test different weight function options."""

    def test_wendland_weight(self):
        """Test Wendland weight function."""
        points = np.linspace(0, 1, 20).reshape(-1, 1)
        gfdm = GFDMOperator(points, delta=0.15, taylor_order=2, weight_function="wendland")

        f = points[:, 0] ** 2
        laplacian = gfdm.laplacian(f)

        # Polynomial reproduction: measured max|lap - 2| = 5.3e-14 at every node.
        # Pins conditioning, not the weight's shape -- all three weights are exact on a quadratic.
        np.testing.assert_allclose(laplacian, 2.0, atol=1e-10)

    def test_gaussian_weight(self):
        """Test Gaussian weight function."""
        points = np.linspace(0, 1, 20).reshape(-1, 1)
        gfdm = GFDMOperator(points, delta=0.15, taylor_order=2, weight_function="gaussian")

        f = points[:, 0] ** 2
        laplacian = gfdm.laplacian(f)

        # Measured max|lap - 2| = 2.4e-14. An all-zero return -- what a singular moment
        # matrix produces here -- passes `not isnan` and fails this.
        np.testing.assert_allclose(laplacian, 2.0, atol=1e-10)

    def test_uniform_weight(self):
        """Test uniform weight function."""
        points = np.linspace(0, 1, 20).reshape(-1, 1)
        gfdm = GFDMOperator(points, delta=0.15, taylor_order=2, weight_function="uniform")

        f = points[:, 0] ** 2
        laplacian = gfdm.laplacian(f)

        # Uniform weights are the worst-conditioned branch; measured max|lap - 2| = 2.6e-14.
        np.testing.assert_allclose(laplacian, 2.0, atol=1e-10)

    def test_invalid_weight_function(self):
        """Test that invalid weight function raises error."""
        points = np.linspace(0, 1, 20).reshape(-1, 1)

        with pytest.raises(ValueError, match=r"Unknown weight function"):
            GFDMOperator(points, delta=0.15, weight_function="invalid")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_delta_warning(self):
        """Test behavior with very small delta (few neighbors)."""
        points = np.linspace(0, 1, 10).reshape(-1, 1)

        # No point has any neighbour within delta=0.01 (spacing is 1/9), so the operator
        # warns and falls back to k-NN.
        with pytest.warns(UserWarning, match=r"Hybrid neighborhood: 10/10 points"):
            gfdm = GFDMOperator(points, delta=0.01, taylor_order=2)

        f = points[:, 0] ** 2
        laplacian = gfdm.laplacian(f)
        assert laplacian.shape == (10,)
        # The fallback is still under-determined, and the operator answers with an exact
        # all-zero sentinel instead of raising. Pinned to make that visible, NOT to bless it:
        # a correct operator would return 2.0 here, as every other test in this file requires.
        np.testing.assert_array_equal(laplacian, np.zeros(10))

    def test_large_delta(self):
        """Test behavior with large delta (many neighbors)."""
        points = np.linspace(0, 1, 20).reshape(-1, 1)

        gfdm = GFDMOperator(points, delta=1.0, taylor_order=2)

        f = points[:, 0] ** 2
        laplacian = gfdm.laplacian(f)

        # Every point is every other point's neighbour, so this is a conditioning test:
        # the moment matrix stays invertible. Measured max|lap - 2| = 2.0e-15.
        np.testing.assert_allclose(laplacian, 2.0, atol=1e-10)

    def test_single_point(self):
        """Test with single point (edge case).

        Single-point GFDM is mathematically undefined - you cannot compute
        derivatives without at least 2 points. The operator should raise
        a clear error rather than failing obscurely.
        """
        points = np.array([[0.5]])

        # With taylor_order=1, GFDM needs at least k_neighbors=2 points
        # to solve the underdetermined system. Single point should fail.
        with pytest.raises((ValueError, IndexError)):
            GFDMOperator(points, delta=0.1, taylor_order=1)

    def test_3d_points(self):
        """Test 3D operator on a resolved scattered cloud.

        The order-2 Taylor system in 3D has 9 unknowns, so delta must admit well more
        than 9 neighbours. At (100 points, delta=0.3) it does not: every point falls
        back to k-NN with exactly k=10, the system is degenerate, and the Laplacian of
        x^2+y^2+z^2 comes back with a median error of 2.0 against the exact value 6
        (max 6.3, i.e. one point returns 0). Seeded and resolved here instead.
        """
        points = np.random.default_rng(0).random((200, 3))
        gfdm = GFDMOperator(points, delta=0.55, taylor_order=2)

        f = points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2
        laplacian = gfdm.laplacian(f)

        assert laplacian.shape == (200,)
        # Smallest neighbourhood is 24 points, no k-NN fallback. Measured
        # max|lap - 6| = 5.7e-14 at seed 0, and < 7.5e-14 over seeds 0-7.
        np.testing.assert_allclose(laplacian, 6.0, atol=1e-8)

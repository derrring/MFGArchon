"""Tests for the radial interaction kernel zoo (Issue #1023)."""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.operators.interaction.kernels import (
    DipoleKernel,
    GaussianKernel,
    PowerLawKernel,
    RadialKernel,
    TentKernel,
    WendlandKernel,
)

ALL_KERNELS = [
    GaussianKernel(amplitude=1.0, length_scale=0.2),
    TentKernel(amplitude=1.0, length_scale=0.2),
    WendlandKernel(amplitude=1.0, length_scale=0.2),
    DipoleKernel(),
    PowerLawKernel(),
]


class TestRadialKernelInterface:
    @pytest.mark.parametrize("kernel", ALL_KERNELS)
    def test_is_radial_kernel(self, kernel):
        assert isinstance(kernel, RadialKernel)

    @pytest.mark.parametrize("kernel", ALL_KERNELS)
    def test_translational(self, kernel):
        assert kernel.is_translational

    @pytest.mark.parametrize("kernel", ALL_KERNELS)
    def test_callable_returns_array(self, kernel):
        r = np.linspace(0.0, 1.0, 11)
        out = kernel(r)
        assert out.shape == r.shape

    @pytest.mark.parametrize("kernel", ALL_KERNELS)
    def test_matrix_symmetric(self, kernel):
        """W_ij = K(|x_i - x_j|) is symmetric (radial kernel)."""
        x = np.linspace(0.0, 1.0, 15)
        W = kernel.matrix(x)
        assert W.shape == (15, 15)
        np.testing.assert_allclose(W, W.T, atol=1e-14)

    @pytest.mark.parametrize("kernel", ALL_KERNELS)
    def test_matrix_2d_points(self, kernel):
        """matrix() handles (N, d) point sets via Euclidean distance."""
        rng = np.random.RandomState(0)
        pts = rng.rand(12, 2)
        W = kernel.matrix(pts)
        assert W.shape == (12, 12)
        np.testing.assert_allclose(W, W.T, atol=1e-14)
        # Diagonal is K(0).
        np.testing.assert_allclose(np.diag(W), kernel(np.zeros(12)), atol=1e-14)


class TestGaussianKernel:
    def test_peak_at_zero(self):
        K = GaussianKernel(amplitude=3.0, length_scale=0.1)
        assert K(np.array([0.0]))[0] == pytest.approx(3.0)

    def test_decay(self):
        K = GaussianKernel(amplitude=1.0, length_scale=0.1)
        vals = K(np.array([0.0, 0.05, 0.2, 0.5]))
        assert vals[0] > vals[1] > vals[2] > vals[3]

    def test_repulsive_sign(self):
        assert GaussianKernel(amplitude=1.0).is_repulsive
        assert not GaussianKernel(amplitude=-1.0).is_repulsive

    def test_invalid_length_scale(self):
        with pytest.raises(ValueError):
            GaussianKernel(length_scale=0.0)


class TestTentKernel:
    def test_compact_support(self):
        K = TentKernel(amplitude=1.0, length_scale=0.3)
        assert K(np.array([0.3]))[0] == pytest.approx(0.0)
        assert K(np.array([0.5]))[0] == pytest.approx(0.0)

    def test_linear_decay(self):
        K = TentKernel(amplitude=2.0, length_scale=1.0)
        # K(r) = 2*(1 - r) on [0,1]
        np.testing.assert_allclose(K(np.array([0.0, 0.5, 1.0])), [2.0, 1.0, 0.0])


class TestWendlandKernel:
    def test_compact_support(self):
        K = WendlandKernel(amplitude=1.0, length_scale=0.25)
        assert K(np.array([0.25]))[0] == pytest.approx(0.0)
        assert K(np.array([0.4]))[0] == pytest.approx(0.0)

    def test_peak_at_zero(self):
        K = WendlandKernel(amplitude=2.5, length_scale=0.2)
        assert K(np.array([0.0]))[0] == pytest.approx(2.5)

    def test_c2_smoothness_first_derivative_zero_at_origin(self):
        """Wendland C^2 has vanishing slope at r=0 (smooth, unlike the tent).

        The Wendland forward-difference slope ~ 10 h / ell^2 -> 0, whereas the
        tent kernel has an O(1/ell) corner slope; compare the two.
        """
        h = 1e-6
        ell = 0.3
        wendland = WendlandKernel(amplitude=1.0, length_scale=ell)
        tent = TentKernel(amplitude=1.0, length_scale=ell)
        slope_w = abs((wendland(np.array([h]))[0] - wendland(np.array([0.0]))[0]) / h)
        slope_t = abs((tent(np.array([h]))[0] - tent(np.array([0.0]))[0]) / h)
        assert slope_w < 1e-3 * slope_t


class TestDipoleKernel:
    def test_short_range_repulsion_long_range_attraction(self):
        K = DipoleKernel(rep_amplitude=1.0, rep_scale=0.05, att_amplitude=0.6, att_scale=0.3)
        # Near zero: repulsion dominates (positive).
        assert K(np.array([0.0]))[0] > 0
        # At intermediate range: attraction dominates (negative).
        assert K(np.array([0.2]))[0] < 0

    def test_mixed_sign_is_repulsive_at_origin(self):
        assert DipoleKernel(rep_amplitude=1.0, att_amplitude=0.5).is_repulsive
        assert not DipoleKernel(rep_amplitude=0.5, att_amplitude=1.0).is_repulsive


class TestPowerLawKernel:
    def test_softened_finite_at_zero(self):
        K = PowerLawKernel(amplitude=1.0, exponent=1.0, softening=0.1)
        assert np.isfinite(K(np.array([0.0]))[0])
        assert K(np.array([0.0]))[0] == pytest.approx(0.1 ** (-1.0))

    def test_power_decay(self):
        K = PowerLawKernel(amplitude=1.0, exponent=2.0, softening=1e-3)
        # Far from origin, ~ r^{-2}: ratio of K(1)/K(2) ~ 4.
        ratio = K(np.array([1.0]))[0] / K(np.array([2.0]))[0]
        assert ratio == pytest.approx(4.0, rel=0.01)

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            PowerLawKernel(exponent=0.0)
        with pytest.raises(ValueError):
            PowerLawKernel(softening=0.0)

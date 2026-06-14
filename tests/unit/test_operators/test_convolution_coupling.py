"""Tests for ConvolutionCouplingOperator (Issue #1023).

Gate 1 (convolution correctness):
  (a) FFT matvec == direct-quadrature matvec on the same regular grid;
  (b) direct matvec == hand-computed integral on a tiny grid.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.operators.interaction.convolution import ConvolutionCouplingOperator
from mfgarchon.operators.interaction.kernels import (
    GaussianKernel,
    TentKernel,
    WendlandKernel,
)


class TestGate1FFTvsDirect:
    """Gate 1(a): the FFT path equals the direct-quadrature path."""

    def test_fft_equals_direct_1d(self):
        N = 128
        dx = 1.0 / (N - 1)
        kernel = GaussianKernel(amplitude=2.0, length_scale=0.08)
        F_fft = ConvolutionCouplingOperator(kernel, grid_shape=(N,), spacings=[dx], use_fft=True)
        F_dir = ConvolutionCouplingOperator(kernel, grid_shape=(N,), spacings=[dx], use_fft=False)
        assert F_fft.uses_fft
        assert not F_dir.uses_fft

        rng = np.random.RandomState(0)
        m = np.abs(rng.randn(N)) + 0.1
        np.testing.assert_allclose(F_fft @ m, F_dir @ m, atol=1e-12)

    def test_fft_equals_direct_2d(self):
        grid_shape = (24, 20)
        spacings = [0.04, 0.05]
        kernel = WendlandKernel(amplitude=1.5, length_scale=0.15)
        F_fft = ConvolutionCouplingOperator(kernel, grid_shape=grid_shape, spacings=spacings, use_fft=True)
        F_dir = ConvolutionCouplingOperator(kernel, grid_shape=grid_shape, spacings=spacings, use_fft=False)
        rng = np.random.RandomState(1)
        m = np.abs(rng.randn(grid_shape[0] * grid_shape[1])) + 0.05
        np.testing.assert_allclose(F_fft @ m, F_dir @ m, atol=1e-12)


class TestGate1DirectVsHand:
    """Gate 1(b): the direct path equals the hand-computed integral."""

    def test_tent_kernel_tiny_grid(self):
        # 3-point grid x = [0, 1, 2], spacing 1, K(r) = 1 - r/3 (tent, support 3).
        kernel = TentKernel(amplitude=1.0, length_scale=3.0)
        F = ConvolutionCouplingOperator(kernel, grid_shape=(3,), spacings=[1.0], use_fft=False)
        m = np.array([1.0, 2.0, 3.0])
        # F_i = sum_j K(|i-j|) m_j * cell_volume(=1).
        Kmat = np.array([[1 - abs(i - j) / 3 for j in range(3)] for i in range(3)])
        np.testing.assert_allclose(F @ m, Kmat @ m, atol=1e-14)

    def test_constant_density_uniform_kernel(self):
        # Uniform kernel K=1 on a grid: F[m]_i = integral m dy = sum_j m_j * dx.
        N = 50
        dx = 1.0 / (N - 1)
        kernel = GaussianKernel(amplitude=1.0, length_scale=1e6)  # effectively constant ~1
        F = ConvolutionCouplingOperator(kernel, grid_shape=(N,), spacings=[dx], use_fft=False)
        m = np.ones(N)
        expected = np.full(N, N * dx)  # sum_j 1 * dx
        np.testing.assert_allclose(F @ m, expected, rtol=1e-6)


class TestLinearOperatorContract:
    def test_linearity(self):
        N = 64
        dx = 1.0 / (N - 1)
        F = ConvolutionCouplingOperator(GaussianKernel(1.0, 0.1), grid_shape=(N,), spacings=[dx])
        rng = np.random.RandomState(2)
        f, g = rng.randn(N), rng.randn(N)
        a, b = 1.7, -0.4
        np.testing.assert_allclose(F @ (a * f + b * g), a * (F @ f) + b * (F @ g), atol=1e-10)

    def test_adjoint_consistency(self):
        """<F m, g> == <m, F^T g> (operator is self-adjoint for uniform weights)."""
        N = 50
        dx = 1.0 / (N - 1)
        F = ConvolutionCouplingOperator(WendlandKernel(1.0, 0.2), grid_shape=(N,), spacings=[dx], use_fft=False)
        rng = np.random.RandomState(3)
        m, g = rng.randn(N), rng.randn(N)
        np.testing.assert_allclose(np.dot(F @ m, g), np.dot(m, F.rmatvec(g)), atol=1e-12)

    def test_as_dense_matches_matvec(self):
        N = 40
        dx = 1.0 / (N - 1)
        F = ConvolutionCouplingOperator(GaussianKernel(2.0, 0.15), grid_shape=(N,), spacings=[dx])
        D = F.as_dense()
        assert D.shape == (N, N)
        rng = np.random.RandomState(4)
        m = rng.randn(N)
        np.testing.assert_allclose(D @ m, F @ m, atol=1e-12)

    def test_dense_symmetric_uniform_grid(self):
        N = 30
        dx = 1.0 / (N - 1)
        F = ConvolutionCouplingOperator(GaussianKernel(1.0, 0.2), grid_shape=(N,), spacings=[dx])
        D = F.as_dense()
        np.testing.assert_allclose(D, D.T, atol=1e-14)


class TestIrregularCloud:
    def test_direct_on_irregular_points(self):
        rng = np.random.RandomState(5)
        pts = np.sort(rng.rand(40))
        kernel = GaussianKernel(1.0, 0.1)
        cell = 1.0 / 40
        F = ConvolutionCouplingOperator(kernel, points=pts, cell_volume=cell)
        m = np.abs(rng.randn(40)) + 0.1
        W = kernel.matrix(pts)
        np.testing.assert_allclose(F @ m, (W @ m) * cell, atol=1e-12)

    def test_per_point_weights_adjoint(self):
        rng = np.random.RandomState(6)
        pts = rng.rand(20, 2)
        kernel = WendlandKernel(1.0, 0.4)
        w = np.abs(rng.rand(20)) + 0.01
        F = ConvolutionCouplingOperator(kernel, points=pts, cell_volume=w)
        m, g = rng.randn(20), rng.randn(20)
        # Operator matrix M = W @ diag(w); adjoint is diag(w) @ W.
        np.testing.assert_allclose(np.dot(F @ m, g), np.dot(m, F.rmatvec(g)), atol=1e-12)


class TestConstructionValidation:
    def test_both_modes_raises(self):
        with pytest.raises(ValueError):
            ConvolutionCouplingOperator(GaussianKernel(), points=np.zeros(3), grid_shape=(3,), spacings=[1.0])

    def test_neither_mode_raises(self):
        with pytest.raises(ValueError):
            ConvolutionCouplingOperator(GaussianKernel())

    def test_regular_requires_spacings(self):
        with pytest.raises(ValueError):
            ConvolutionCouplingOperator(GaussianKernel(), grid_shape=(10,))

    def test_irregular_requires_cell_volume(self):
        with pytest.raises(ValueError):
            ConvolutionCouplingOperator(GaussianKernel(), points=np.zeros(3))

    def test_fft_on_irregular_raises(self):
        with pytest.raises(ValueError):
            ConvolutionCouplingOperator(GaussianKernel(), points=np.zeros(3), cell_volume=0.1, use_fft=True)

    def test_cell_volume_on_regular_raises(self):
        with pytest.raises(ValueError):
            ConvolutionCouplingOperator(GaussianKernel(), grid_shape=(5,), spacings=[0.1], cell_volume=0.1)

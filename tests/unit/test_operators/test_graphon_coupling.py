"""Tests for GraphonCouplingOperator and graphon kernels."""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.operators.integro_diff.graphon_coupling import GraphonCouplingOperator
from mfgarchon.operators.integro_diff.graphon_kernels import (
    ConstantGraphon,
    GeometricGraphon,
    GraphonKernel,
    StochasticBlockModelGraphon,
)


class TestGraphonKernelProtocol:
    def test_constant_satisfies_protocol(self):
        assert isinstance(ConstantGraphon(0.5), GraphonKernel)

    def test_sbm_satisfies_protocol(self):
        B = np.array([[0.8, 0.2], [0.2, 0.8]])
        assert isinstance(StochasticBlockModelGraphon(B), GraphonKernel)

    def test_geometric_satisfies_protocol(self):
        assert isinstance(GeometricGraphon(0.1), GraphonKernel)


class TestConstantGraphon:
    def test_evaluate_shape(self):
        W = ConstantGraphon(0.5)
        result = W.evaluate(np.linspace(0, 1, 10), np.linspace(0, 1, 20))
        assert result.shape == (10, 20)

    def test_evaluate_value(self):
        W = ConstantGraphon(0.3)
        result = W.evaluate(np.array([0.0, 0.5, 1.0]), np.array([0.2, 0.8]))
        np.testing.assert_allclose(result, 0.3)

    def test_symmetric(self):
        assert ConstantGraphon(0.5).is_symmetric

    def test_invalid_p(self):
        with pytest.raises(ValueError):
            ConstantGraphon(1.5)


class TestStochasticBlockModelGraphon:
    def test_2block(self):
        B = np.array([[0.9, 0.1], [0.1, 0.9]])
        W = StochasticBlockModelGraphon(B)
        # x=0.2, y=0.8 → different blocks → B[0,1] = 0.1
        result = W.evaluate(np.array([0.2]), np.array([0.8]))
        assert result[0, 0] == pytest.approx(0.1)

    def test_same_block(self):
        B = np.array([[0.9, 0.1], [0.1, 0.9]])
        W = StochasticBlockModelGraphon(B)
        # x=0.2, y=0.3 → same block → B[0,0] = 0.9
        result = W.evaluate(np.array([0.2]), np.array([0.3]))
        assert result[0, 0] == pytest.approx(0.9)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            StochasticBlockModelGraphon(np.zeros((2, 3)))

    def test_symmetric(self):
        B = np.array([[0.8, 0.2], [0.2, 0.8]])
        assert StochasticBlockModelGraphon(B).is_symmetric


class TestGeometricGraphon:
    def test_diagonal_is_one(self):
        """W(x, x) = exp(0) = 1 for geometric graphon."""
        W = GeometricGraphon(0.1)
        x = np.linspace(0, 1, 11)
        result = W.evaluate(x, x)
        np.testing.assert_allclose(np.diag(result), 1.0)

    def test_decay_with_distance(self):
        W = GeometricGraphon(0.1)
        result = W.evaluate(np.array([0.0]), np.array([0.0, 0.1, 0.5, 1.0]))
        # Should decrease with distance
        assert result[0, 0] > result[0, 1] > result[0, 2] > result[0, 3]

    def test_translational(self):
        assert GeometricGraphon(0.1).is_translational


class TestGraphonCouplingOperator:
    def test_constant_kernel_integral(self):
        """W[f](x) = p * integral(f) for constant graphon."""
        x = np.linspace(0, 1, 101)
        W_op = GraphonCouplingOperator(x, ConstantGraphon(p=0.5), quadrature_order=32)
        f = np.ones(101)
        Wf = W_op @ f
        # integral of 1 over [0,1] = 1, so W[f] = 0.5 * 1 = 0.5
        np.testing.assert_allclose(Wf, 0.5, atol=0.02)

    def test_shape(self):
        x = np.linspace(0, 1, 51)
        W_op = GraphonCouplingOperator(x, ConstantGraphon(0.5))
        f = np.ones(51)
        assert (W_op @ f).shape == (51,)

    def test_linearity(self):
        """W[a*f + b*g] = a*W[f] + b*W[g]."""
        x = np.linspace(0, 1, 51)
        W_op = GraphonCouplingOperator(x, GeometricGraphon(0.2), use_fft=False)
        rng = np.random.RandomState(42)
        f = rng.randn(51)
        g = rng.randn(51)
        a, b = 2.3, -0.7
        lhs = W_op @ (a * f + b * g)
        rhs = a * (W_op @ f) + b * (W_op @ g)
        np.testing.assert_allclose(lhs, rhs, atol=1e-10)

    def test_adjoint_consistency_via_dense(self):
        """Dense matrix W should be symmetric for symmetric kernel."""
        x = np.linspace(0, 1, 31)
        W_op = GraphonCouplingOperator(x, ConstantGraphon(0.5))
        D = W_op.as_dense()
        np.testing.assert_allclose(D, D.T, atol=1e-10)

    def test_adjoint_consistency_geometric(self):
        """Dense matrix should be symmetric for geometric (symmetric) kernel."""
        x = np.linspace(0, 1, 31)
        W_op = GraphonCouplingOperator(x, GeometricGraphon(0.2))
        D = W_op.as_dense()
        np.testing.assert_allclose(D, D.T, atol=1e-10)

    def test_sbm_piecewise(self):
        """SBM graphon should produce different values in different blocks."""
        x = np.linspace(0, 1, 101)
        B = np.array([[0.9, 0.1], [0.1, 0.9]])
        W_op = GraphonCouplingOperator(x, StochasticBlockModelGraphon(B))
        # f = 1 in block 0, 0 in block 1
        f = np.zeros(101)
        f[:50] = 1.0
        Wf = W_op @ f
        # Block 0 (x < 0.5): high coupling to f → high value
        # Block 1 (x >= 0.5): low coupling to f → low value
        assert Wf[:25].mean() > Wf[75:].mean()

    def test_as_dense(self):
        x = np.linspace(0, 1, 21)
        W_op = GraphonCouplingOperator(x, ConstantGraphon(0.5))
        D = W_op.as_dense()
        assert D.shape == (21, 21)
        # For constant kernel: D_{ij} = 0.5 * dx
        dx = x[1] - x[0]
        np.testing.assert_allclose(D, 0.5 * dx, atol=1e-10)

    def test_fft_path(self):
        """FFT path should give similar results to quadrature for geometric kernel."""
        x = np.linspace(0, 1, 101)
        kernel = GeometricGraphon(0.2)
        W_quad = GraphonCouplingOperator(x, kernel, use_fft=False, quadrature_order=64)
        W_fft = GraphonCouplingOperator(x, kernel, use_fft=True)
        f = np.sin(np.pi * x)
        result_quad = W_quad @ f
        result_fft = W_fft @ f
        # FFT uses grid points directly (different discretization), so allow tolerance
        np.testing.assert_allclose(result_fft, result_quad, rtol=0.15)


class TestGraphonConvergenceToFiniteGraph:
    def test_constant_graphon_matches_complete_graph(self):
        """Constant graphon W=p is the limit of complete graph G(n,p).

        For f = constant, W[f] = p * integral(f) = p * f * L
        where L is the domain length.
        """
        x = np.linspace(0, 1, 201)
        p = 0.7
        W_op = GraphonCouplingOperator(x, ConstantGraphon(p), quadrature_order=64)
        f = np.ones(201) * 3.0
        Wf = W_op @ f
        # Expected: p * integral(3.0 over [0,1]) = 0.7 * 3.0 = 2.1
        np.testing.assert_allclose(Wf, 2.1, atol=0.05)

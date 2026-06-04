"""Tests for the canonical volatility -> diffusion converter (Issue #811).

``diffusion_from_volatility`` is the single source of truth for ``D = (1/2) Sigma Sigma^T``
(scalar: ``D = sigma^2/2``), replacing ~38 ad-hoc literal ``0.5 * sigma**2`` sites. These
tests pin the contract documented in NAMING_CONVENTIONS.md "Volatility vs Diffusion":
tensor-first, ``Sigma Sigma^T`` (NOT ``Sigma^T Sigma``), and byte-identity for the scalar /
field cases it replaces.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.utils.pde_coefficients import diffusion_from_volatility


class TestScalarAndField:
    def test_scalar_is_half_sigma_squared(self):
        assert diffusion_from_volatility(0.5) == pytest.approx(0.125)  # 0.5^2/2

    def test_scalar_byte_identical_to_literal(self):
        # the literal it replaces is `0.5 * sigma**2`; IEEE-754: 0.5*x == x/2 exactly
        for sigma in (0.3, 0.05, 1.0, 0.123456789):
            assert diffusion_from_volatility(sigma) == 0.5 * sigma**2

    def test_spatial_field_is_elementwise(self):
        sigma = np.array([0.05, 0.1, 0.2, 0.3])  # (*spatial,) isotropic per-point
        D = diffusion_from_volatility(sigma)
        assert np.array_equal(D, 0.5 * sigma**2)
        assert D.shape == sigma.shape

    def test_2d_spatial_field_with_dimension_is_elementwise(self):
        # a (Nx, Ny) spatial field for a 2D problem: pass dimension so it is NOT read as a tensor
        sigma = np.linspace(0.05, 0.3, 35).reshape(5, 7)
        D = diffusion_from_volatility(sigma, dimension=2)  # shape[0]=5 != 2 -> field
        assert np.array_equal(D, 0.5 * sigma**2)


class TestTensor:
    def test_symmetric_tensor(self):
        Sigma = np.array([[0.3, 0.1], [0.1, 0.2]])  # symmetric
        D = diffusion_from_volatility(Sigma)
        assert np.allclose(D, 0.5 * Sigma @ Sigma.T)
        assert np.allclose(D, D.T)  # D is symmetric PSD

    def test_uses_sigma_sigmaT_not_sigmaT_sigma(self):
        # LOAD-BEARING: non-symmetric Sigma distinguishes Sigma@Sigma.T from Sigma.T@Sigma
        Sigma = np.array([[1.0, 2.0], [0.0, 1.0]])
        D = diffusion_from_volatility(Sigma)
        assert np.allclose(D, 0.5 * Sigma @ Sigma.T), "must use Sigma @ Sigma.T"
        assert not np.allclose(D, 0.5 * Sigma.T @ Sigma), "must NOT use Sigma.T @ Sigma"

    def test_nonsquare_dk_matrix_gives_dxd(self):
        # (d, k) = (2, 3) volatility -> (d, d) = (2, 2) diffusion via Sigma @ Sigma.T
        Sigma = np.array([[0.2, 0.1, 0.05], [0.0, 0.3, 0.1]])  # (2, 3)
        D = diffusion_from_volatility(Sigma, dimension=2)
        assert D.shape == (2, 2)
        assert np.allclose(D, 0.5 * Sigma @ Sigma.T)

    def test_default_treats_square_2d_as_tensor(self):
        # without a dimension hint a square 2-D array is a Sigma tensor (volatility is tensor by default)
        Sigma = np.array([[0.4, 0.0], [0.2, 0.3]])
        assert np.allclose(diffusion_from_volatility(Sigma), 0.5 * Sigma @ Sigma.T)

    def test_spatially_varying_tensor_field(self):
        # (*spatial, d, k) -> per-point 0.5 * Sigma Sigma^T over the trailing axes
        nx, d = 4, 2
        rng = np.random.default_rng(0)
        Sig = rng.uniform(0.05, 0.3, size=(nx, d, d))
        D = diffusion_from_volatility(Sig)
        assert D.shape == (nx, d, d)
        for i in range(nx):
            assert np.allclose(D[i], 0.5 * Sig[i] @ Sig[i].T)

"""RFC #1596: single-source sigma->D tensor convention (folds #1548, #1549; re-founds #1079).

Convention pinned here (so a future private re-fork fails loudly -- the silent-divergence bug class):

- ``sigma`` is the SDE VOLATILITY (a standard deviation) everywhere. Conversion sigma->D is owned by
  ``diffusion_from_volatility`` (Issue #811): scalar -> D = sigma^2/2; (d,d) -> D = 1/2 S S^T where
  the (d,d) input is the SYMMETRIC standard-deviation matrix S (symmetric square root of covariance).
- Squaring is UNIVERSAL (a diagonal std-dev tensor must match the scalar path -- the #1079/#1548
  divergence). Symmetry is a CONSUMER admissibility gate (grid solvers reject an asymmetric (d,d)).
- ``DiffusionOperator`` (Path A) is a pure div(D grad u): the coefficient is the already-converted D,
  no branch squares, so scalar / diagonal-vector / diagonal-tensor for identical isotropic physics
  agree (the #1549 10x shape-flip).
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_adi import adi_diffusion_step
from mfgarchon.core.mfg_problem import _diffusion_to_volatility
from mfgarchon.operators.differential.diffusion import DiffusionOperator
from mfgarchon.utils.pde_coefficients import diffusion_from_volatility


class TestKernelConvention:
    """The single-source converter defines D = 1/2 S S^T for a (d,d) volatility."""

    def test_tensor_is_half_SSt_and_symmetric(self) -> None:
        S = np.array([[0.2, 0.05], [0.05, 0.3]])
        D = diffusion_from_volatility(S, kind="tensor")
        np.testing.assert_allclose(D, 0.5 * S @ S.T)
        np.testing.assert_allclose(D, D.T)  # D is always symmetric

    def test_scalar_is_half_sigma_squared(self) -> None:
        assert diffusion_from_volatility(0.3) == pytest.approx(0.5 * 0.3**2)


class TestSLADIVolatilityConvention:
    """SL-ADI tensor branch squares (routes through the converter) and gates symmetry."""

    def test_diagonal_tensor_matches_scalar_magnitude(self) -> None:
        """The pin that catches the #1079(no-square) vs #1548(square) divergence: a diagonal
        std-dev tensor diag([s, s]) MUST produce the exact same diffusion as the scalar volatility
        s (both are D = s^2/2 per axis, zero cross term). Under the old covariance/no-square reading
        the diagonal tensor gave D = s (2x off)."""
        rng = np.random.default_rng(0)
        U = rng.standard_normal((14, 14))
        spacing = np.array([0.1, 0.1])
        s = 0.35
        out_scalar = adi_diffusion_step(U.copy(), dt=0.01, sigma=s, spacing=spacing, grid_shape=(14, 14))
        out_tensor = adi_diffusion_step(U.copy(), dt=0.01, sigma=np.diag([s, s]), spacing=spacing, grid_shape=(14, 14))
        np.testing.assert_allclose(out_tensor, out_scalar, rtol=0, atol=1e-13)

    def test_symmetric_offdiagonal_accepted(self) -> None:
        rng = np.random.default_rng(1)
        U = rng.standard_normal((12, 12))
        S = np.array([[0.2, 0.04], [0.04, 0.25]])  # symmetric PSD
        out = adi_diffusion_step(U, dt=0.01, sigma=S, spacing=np.array([0.1, 0.1]), grid_shape=(12, 12))
        assert out.shape == (12, 12)
        assert np.all(np.isfinite(out))

    def test_asymmetric_rejected(self) -> None:
        U = np.zeros((8, 8))
        with pytest.raises(ValueError, match="symmetric"):
            adi_diffusion_step(
                U, dt=0.01, sigma=np.array([[0.3, 0.1], [0.0, 0.2]]), spacing=np.array([0.1, 0.1]), grid_shape=(8, 8)
            )


class TestDiffusionOperatorShapeConsistency:
    """Path A: scalar / diagonal-vector / diagonal-tensor for identical isotropic physics agree
    (the #1549 10x shape-flip); from_volatility routes sigma->D through the single source."""

    def test_scalar_vector_tensor_agree(self) -> None:
        rng = np.random.default_rng(2)
        u = rng.standard_normal((18, 18))
        spac = [0.1, 0.1]
        fs = (18, 18)
        D = 0.005
        a = DiffusionOperator(D, spac, fs)(u)
        b = DiffusionOperator(np.array([D, D]), spac, fs)(u)
        c = DiffusionOperator(np.diag([D, D]), spac, fs)(u)
        np.testing.assert_allclose(a, b, atol=1e-12)
        np.testing.assert_allclose(a, c, atol=1e-12)

    def test_from_volatility_matches_manual_D(self) -> None:
        rng = np.random.default_rng(3)
        u = rng.standard_normal((16, 16))
        spac = [0.1, 0.1]
        fs = (16, 16)
        sigma = 0.2
        from_vol = DiffusionOperator.from_volatility(sigma, spac, fs)(u)
        manual = DiffusionOperator(0.5 * sigma**2, spac, fs)(u)
        np.testing.assert_allclose(from_vol, manual, atol=1e-14)

    def test_tensor_coefficient_asymmetric_rejected(self) -> None:
        with pytest.raises(ValueError, match="symmetric"):
            DiffusionOperator(np.array([[0.02, 0.01], [0.0, 0.03]]), [0.1, 0.1], (10, 10))

    def test_from_volatility_asymmetric_rejected(self) -> None:
        with pytest.raises(ValueError, match="symmetric"):
            DiffusionOperator.from_volatility(np.array([[0.2, 0.1], [0.0, 0.3]]), [0.1, 0.1], (10, 10))


class TestInverseRoundTrip:
    """volatility->D->volatility lands back on a SYMMETRIC S that passes the consumer gate."""

    def test_roundtrip_symmetric(self) -> None:
        D = np.array([[0.02, 0.005], [0.005, 0.045]])
        S = _diffusion_to_volatility(D)
        np.testing.assert_allclose(S, S.T)  # symmetric square root, not a Cholesky factor
        np.testing.assert_allclose(diffusion_from_volatility(S, kind="tensor"), D, atol=1e-14)

    def test_diagonal_D_roundtrips_to_sqrt(self) -> None:
        D = np.diag([0.02, 0.045])
        S = _diffusion_to_volatility(D)
        # symmetric sqrt of 2D on the diagonal = sqrt(2 * D_ii)
        np.testing.assert_allclose(np.diag(S), np.sqrt(2.0 * np.diag(D)), atol=1e-12)

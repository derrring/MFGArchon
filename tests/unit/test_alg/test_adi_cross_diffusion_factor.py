"""
Pinning test for Issue #1261: ADI explicit cross-diffusion term was 2x too large.

Convention: sigma_tensor is the covariance matrix. Diagonal ADI uses
D_d = sigma_tensor[d,d]/2 (via diffusion_from_volatility on sqrt of diagonal).
The off-diagonal explicit contribution per pair (i<j) must be
  dt * sigma_tensor[i,j] * d^2u/dx_i dx_j
NOT dt * 2 * sigma_tensor[i,j] * d^2u/dx_i dx_j.

The factor-of-2 from symmetry (sum over all i!=j = 2*sum_{i<j}) and the 1/2 in
D_ij = sigma_tensor[i,j]/2 cancel; no residual factor of 2 remains.
"""

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_adi import apply_cross_diffusion_explicit


def _make_quadratic_xy(Nx: int, Ny: int, dx: float, dy: float) -> np.ndarray:
    """u(x,y) = x * y on a uniform grid.  Mixed derivative d^2u/dx dy = 1 everywhere."""
    xs = np.arange(Nx) * dx
    ys = np.arange(Ny) * dy
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return X * Y


class TestADICrossDiffusionFactor:
    """Issue #1261, 2026-06-10 audit: off-diagonal factor must be sigma_ij, not 2*sigma_ij."""

    def test_cross_diffusion_increment_matches_sigma_not_2sigma(self) -> None:
        """
        For u(x,y) = x*y the mixed derivative d^2u/dx dy = 1 everywhere (interior).
        With sigma_tensor = [[a, b], [b, a]] (symmetric, b != 0) and dt=1,
        the increment at interior points must equal b * 1 = b,
        NOT 2*b.

        This test FAILS on buggy code (returns 2*b) and PASSES after the fix (returns b).
        """
        Nx, Ny = 10, 10
        dx, dy = 0.1, 0.1
        dt = 1.0

        a = 1.0  # diagonal; irrelevant to cross term
        b = 0.5  # off-diagonal
        sigma_tensor = np.array([[a, b], [b, a]])
        spacing = np.array([dx, dy])

        U = _make_quadratic_xy(Nx, Ny, dx, dy)
        U_new = apply_cross_diffusion_explicit(U, sigma_tensor, dt, spacing)

        increment = U_new - U  # should be dt * b * mixed_deriv_approx

        # Interior region: rows/cols 1:-1 (boundary is zero-padded in mixed deriv)
        interior = increment[1:-1, 1:-1]

        # The exact mixed second derivative of x*y is 1 everywhere.
        # Central-difference approximation of d^2(xy)/dxdy on a uniform grid = 1 exactly.
        # So dt * sigma_ij * d^2u/dxdy = 1.0 * 0.5 * 1.0 = 0.5 at every interior point.
        expected = dt * b * 1.0  # = 0.5

        np.testing.assert_allclose(
            interior,
            expected,
            rtol=1e-10,
            err_msg=(
                f"Expected cross-diffusion increment {expected} (= dt*sigma_ij*mixed_deriv), "
                f"got {interior[0, 0]:.6f}. "
                "If 2x too large, this is Issue #1261 (factor-of-2 bug)."
            ),
        )

    def test_cross_diffusion_coefficient_is_sigma_not_2sigma_parametric(self) -> None:
        """
        Parametric sweep over several b values to confirm linearity and correct coefficient.
        """
        Nx, Ny = 8, 8
        dx, dy = 0.2, 0.2
        dt = 0.5
        spacing = np.array([dx, dy])

        for b in [0.1, 0.3, 0.7, 1.0]:
            sigma_tensor = np.array([[1.0, b], [b, 1.0]])
            U = _make_quadratic_xy(Nx, Ny, dx, dy)
            U_new = apply_cross_diffusion_explicit(U, sigma_tensor, dt, spacing)
            interior_increment = (U_new - U)[1:-1, 1:-1]
            expected = dt * b * 1.0  # d^2(xy)/dxdy = 1
            np.testing.assert_allclose(
                interior_increment,
                expected,
                rtol=1e-10,
                err_msg=f"b={b}: expected {expected}, got {interior_increment[0, 0]:.6f}",
            )

    def test_zero_off_diagonal_no_change(self) -> None:
        """Diagonal sigma_tensor: cross term is zero, U unchanged."""
        Nx, Ny = 6, 6
        dx, dy = 0.1, 0.1
        sigma_tensor = np.array([[1.0, 0.0], [0.0, 1.0]])
        spacing = np.array([dx, dy])
        U = _make_quadratic_xy(Nx, Ny, dx, dy)
        U_new = apply_cross_diffusion_explicit(U, sigma_tensor, 1.0, spacing)
        np.testing.assert_array_equal(U_new, U, err_msg="Zero off-diagonal: U must be unchanged.")

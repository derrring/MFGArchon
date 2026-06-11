"""
Regression tests for #1282: backend fpk_step physics defects.

(1) torch_backend.py fpk_step applied div_term with PLUS instead of MINUS,
    reversing transport direction vs numpy/jax/numba siblings.

(2) numba_backend.py fpk_step_kernel reused U_x[i] for flux at i-1 and i+1,
    dropping the m*U_xx term; conservative flux form F[j]=M[j]*v[j] with
    central divergence (F[i+1]-F[i-1])/(2*dx) is the correct form.

Each pinning test FAILs on the pre-fix code and PASSES after the fix.
"""

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# Shared problem setup
# ---------------------------------------------------------------------------


def _trapz(M, dx):
    """np.trapezoid (numpy>=2) or np.trapz (numpy<2)."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return fn(M, dx=dx)


def _linear_U_params(nx: int = 40):
    """Linear U so that U_xx=0 and numpy/numba results agree exactly."""
    dx = 1.0 / (nx - 1)
    x = np.linspace(0.0, 1.0, nx)
    U = x.copy()  # U_x = 1 everywhere, drift a* = -1, flux = -M
    # Start with Gaussian density
    M = np.exp(-0.5 * ((x - 0.5) / 0.15) ** 2)
    M /= _trapz(M, dx=dx)
    dt = 1e-3
    problem_params = {"sigma_sq": 0.0, "sigma": 0.0, "x_grid": x}
    return M, U, dt, dx, problem_params


def _nonlinear_U_params(nx: int = 40):
    """Quadratic U so U_xx != 0; exposes the dropped m*U_xx term in numba."""
    dx = 1.0 / (nx - 1)
    x = np.linspace(0.0, 1.0, nx)
    U = x**2  # U_x = 2x, U_xx = 2; drift a* = -2x
    M = np.exp(-0.5 * ((x - 0.5) / 0.15) ** 2)
    M /= _trapz(M, dx=dx)
    dt = 1e-3
    problem_params = {"sigma_sq": 0.0, "sigma": 0.0, "x_grid": x}
    return M, U, dt, dx, problem_params


# ---------------------------------------------------------------------------
# Bug (1): torch fpk_step sign inverted
# ---------------------------------------------------------------------------


class TestTorchFpkSign:
    """
    Verify torch fpk_step evolves density in the SAME direction as numpy.

    Pre-fix code had ``M_new = M_tensor + dt * (div_term + diffusion_term)``
    (PLUS on div_term); correct is MINUS.  Under linear U with zero diffusion
    the density should shift leftward (drift = -1).  With the bug it shifts
    rightward — opposite to numpy.
    """

    @pytest.fixture
    def torch_backend(self):
        pytest.importorskip("torch")
        from mfgarchon.backends.torch_backend import TorchBackend

        return TorchBackend()

    @pytest.fixture
    def numpy_backend(self):
        from mfgarchon.backends.numpy_backend import NumPyBackend

        return NumPyBackend()

    def test_torch_fpk_same_direction_as_numpy(self, torch_backend, numpy_backend):
        """
        Pinning test for #1282 item 1.

        The centre-of-mass of the density should shift in the SAME direction
        for torch and numpy after one fpk_step.
        Pre-fix: torch CoM shifts opposite to numpy (sign error).
        Post-fix: shifts agree in sign.
        """
        M, U, dt, dx, problem_params = _linear_U_params(nx=40)

        M_np = numpy_backend.fpk_step(M, U, dt, dx, problem_params)

        # Torch backend reads "diffusion" key with default 0.1 (issue #1282
        # item 3 is a separate key-name concern; keep zero diffusion via
        # sigma_sq=0 so the drift-sign difference is isolated here).
        torch_params = dict(problem_params)
        torch_params["diffusion"] = 0.0  # override torch's key to zero
        M_torch = torch_backend.fpk_step(M, U, dt, dx, torch_params)
        if hasattr(M_torch, "numpy"):
            M_torch = M_torch.detach().cpu().numpy()

        x = problem_params["x_grid"]
        com_before = np.sum(x * M) / np.sum(M)
        com_np = np.sum(x * M_np) / np.sum(M_np)
        com_torch = np.sum(x * M_torch) / np.sum(M_torch)

        shift_np = com_np - com_before
        shift_torch = com_torch - com_before

        # Both must shift in the same direction (same sign).
        assert shift_np * shift_torch > 0, (
            f"torch and numpy fpk_step shift in opposite directions: "
            f"shift_numpy={shift_np:.6g}, shift_torch={shift_torch:.6g}. "
            f"This is the #1282 sign inversion bug."
        )

    def test_torch_fpk_close_to_numpy_one_step(self, torch_backend, numpy_backend):
        """
        After fixing the sign, one-step outputs should be close.

        Note: torch uses torch.gradient (O(dx^2) central differences on the
        full array) while numpy uses slightly different boundary discretisation,
        so we test closeness only in the interior, not exact equality.
        """
        M, U, dt, dx, problem_params = _linear_U_params(nx=40)
        M_np = numpy_backend.fpk_step(M, U, dt, dx, problem_params)

        torch_params = dict(problem_params)
        torch_params["diffusion"] = 0.0
        M_torch = torch_backend.fpk_step(M, U, dt, dx, torch_params)
        if hasattr(M_torch, "numpy"):
            M_torch = M_torch.detach().cpu().numpy()

        # Interior (away from boundary artefacts from one-sided FD)
        np.testing.assert_allclose(
            M_torch[5:-5],
            M_np[5:-5],
            rtol=1e-2,
            err_msg=("torch fpk_step interior deviates from numpy after sign fix. Refs #1282."),
        )


# ---------------------------------------------------------------------------
# Bug (2): numba drops m*U_xx in fpk_step_kernel
# ---------------------------------------------------------------------------


class TestNumbaFpkFluxDivergence:
    """
    Verify numba fpk_step produces the same flux divergence as numpy,
    including the m*U_xx term that is dropped by the pre-fix kernel.

    The bug: the pre-fix kernel uses U_x[i] for both flux_left and flux_right,
    giving div = (M[i+1]*(-U_x[i]) - M[i-1]*(-U_x[i])) / (2*dx)
              = -U_x[i] * (M[i+1] - M[i-1]) / (2*dx)
              = -U_x[i] * M_x[i].
    The correct form is (F[i+1] - F[i-1])/(2*dx) where F[j] = M[j]*(-U_x[j]),
    which expands to -U_x[i]*M_x[i] - M[i]*U_xx[i] (includes the second term).
    For nonlinear U (U_xx != 0) the two differ; for linear U they agree.
    """

    @pytest.fixture
    def numba_backend(self):
        pytest.importorskip("numba")
        from mfgarchon.backends.numba_backend import NumbaBackend

        return NumbaBackend()

    @pytest.fixture
    def numpy_backend(self):
        from mfgarchon.backends.numpy_backend import NumPyBackend

        return NumPyBackend()

    def test_numba_linear_U_agrees_with_numpy(self, numba_backend, numpy_backend):
        """
        Linear U: U_xx=0, so both buggy and fixed kernels give the same result.
        This test confirms the basic transport structure is intact after the fix.
        """
        M, U, dt, dx, problem_params = _linear_U_params(nx=40)
        # Numba backend reads key "sigma" not "sigma_sq"
        numba_params = dict(problem_params)
        numba_params["sigma"] = 0.0

        M_np = numpy_backend.fpk_step(M, U, dt, dx, problem_params)
        M_nb = numba_backend.fpk_step(M, U, dt, dx, numba_params)

        np.testing.assert_allclose(
            M_nb[1:-1],
            M_np[1:-1],
            atol=1e-10,
            err_msg=(
                "numba fpk_step disagrees with numpy on linear U. "
                "Interior points must match exactly (zero diffusion, zero U_xx). "
                "Refs #1282."
            ),
        )

    def test_numba_nonlinear_U_agrees_with_numpy(self, numba_backend, numpy_backend):
        """
        Pinning test for #1282 item 2.

        Quadratic U => U_xx = 2 everywhere; the dropped m*U_xx term is
        non-zero.  Pre-fix: results differ; post-fix: results match numpy.
        """
        M, U, dt, dx, problem_params = _nonlinear_U_params(nx=40)
        numba_params = dict(problem_params)
        numba_params["sigma"] = 0.0

        M_np = numpy_backend.fpk_step(M, U, dt, dx, problem_params)
        M_nb = numba_backend.fpk_step(M, U, dt, dx, numba_params)

        # Numba uses fastmath=True by default, so floating-point reordering
        # introduces small differences vs numpy even for identical formulas.
        # Pre-fix: max relative error ~2e-3 (m*U_xx term dropped).
        # Post-fix: max relative error <1e-4 (fastmath noise only).
        # rtol=1e-3 discriminates the bug without being fragile to fastmath.
        np.testing.assert_allclose(
            M_nb[2:-2],
            M_np[2:-2],
            rtol=1e-3,
            err_msg=(
                "numba fpk_step disagrees with numpy on nonlinear U "
                "(quadratic U; U_xx=2). "
                "The m*U_xx term is dropped in the pre-fix kernel. "
                "Refs #1282."
            ),
        )

    def test_numba_nonlinear_U_not_equal_to_linear_approximation(self, numba_backend, numpy_backend):
        """
        Sanity check: for nonlinear U the result differs from the simplified
        -U_x[i]*m_x form (the pre-fix approximation), confirming the test
        actually exercises the m*U_xx path.
        """
        M, U, dt, dx, problem_params = _nonlinear_U_params(nx=40)

        # Compute the simplified (buggy) divergence manually
        dU_dx = np.zeros_like(U)
        dU_dx[1:-1] = (U[2:] - U[:-2]) / (2 * dx)
        dU_dx[0] = (U[1] - U[0]) / dx
        dU_dx[-1] = (U[-1] - U[-2]) / dx

        dM_dx = np.zeros_like(M)
        dM_dx[1:-1] = (M[2:] - M[:-2]) / (2 * dx)

        # Simplified (buggy) one-step — uses -U_x * m_x only
        M_buggy = M.copy()
        M_buggy[1:-1] = M[1:-1] + dt * (dU_dx[1:-1] * dM_dx[1:-1])  # -(-1)*...

        numba_params = dict(problem_params)
        numba_params["sigma"] = 0.0
        M_nb = numba_backend.fpk_step(M, U, dt, dx, numba_params)

        # After fix the numba result should NOT equal the simplified form
        # (the m*U_xx term makes them differ at interior points far from 0 density)
        mid = len(M) // 2
        diff = abs(float(M_nb[mid]) - M_buggy[mid])
        assert diff > 1e-8, (
            "numba fpk_step result equals the simplified -U_x*m_x form; "
            "the m*U_xx term appears to still be dropped. Refs #1282."
        )

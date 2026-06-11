"""
Pinning tests for Issue #1286 (2026-06-11 survey):

1. adi_diffusion_step with an unrecognised sigma type must raise ValueError,
   not silently fall back to sigma=0.1.
2. _compute_upwind_divergence must return finite values (no nan) when the
   drift field is zero at one or more collocation points.
"""

from __future__ import annotations

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# Bug 1: adi_diffusion_step silently used sigma=0.1 for bad sigma type
# ---------------------------------------------------------------------------


def test_adi_diffusion_step_bad_sigma_type_raises():
    """Unrecognised sigma type (dict) must raise ValueError (was: silent sigma=0.1)."""
    from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_adi import adi_diffusion_step

    U = np.zeros((4, 4))
    bad_sigma = {"value": 0.5}  # dict -- never a valid sigma type

    with pytest.raises(ValueError, match="unsupported sigma type"):
        adi_diffusion_step(
            U_star=U,
            dt=0.01,
            sigma=bad_sigma,
            spacing=np.array([0.25, 0.25]),
            grid_shape=(4, 4),
        )


def test_adi_diffusion_step_bad_sigma_wrong_ndim_raises():
    """3-D sigma array (ndim=3) must raise ValueError."""
    from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_adi import adi_diffusion_step

    U = np.zeros((4, 4))
    bad_sigma = np.zeros((2, 2, 2))  # ndim=3 -- not scalar, 1-D, or 2-D

    with pytest.raises(ValueError, match="unsupported sigma type"):
        adi_diffusion_step(
            U_star=U,
            dt=0.01,
            sigma=bad_sigma,
            spacing=np.array([0.25, 0.25]),
            grid_shape=(4, 4),
        )


def test_adi_diffusion_step_bad_sigma_wrong_1d_length_raises():
    """1-D sigma with wrong length (3 for 2-D grid) must raise ValueError."""
    from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_adi import adi_diffusion_step

    U = np.zeros((4, 4))
    bad_sigma = np.array([0.5, 0.5, 0.5])  # length 3 for a 2-D grid

    with pytest.raises(ValueError, match="unsupported sigma type"):
        adi_diffusion_step(
            U_star=U,
            dt=0.01,
            sigma=bad_sigma,
            spacing=np.array([0.25, 0.25]),
            grid_shape=(4, 4),
        )


def test_adi_diffusion_step_valid_sigma_types_do_not_raise():
    """Valid sigma types (scalar, 1-D matching dimension, 2-D tensor) must not raise."""
    from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_adi import adi_diffusion_step

    U = np.zeros((4, 4))
    spacing = np.array([0.25, 0.25])

    # scalar
    adi_diffusion_step(U, dt=0.01, sigma=0.5, spacing=spacing, grid_shape=(4, 4))
    # 1-D array matching dimension
    adi_diffusion_step(
        U,
        dt=0.01,
        sigma=np.array([0.5, 0.5]),
        spacing=spacing,
        grid_shape=(4, 4),
    )
    # 2-D tensor
    adi_diffusion_step(
        U,
        dt=0.01,
        sigma=np.diag([0.5, 0.5]),
        spacing=spacing,
        grid_shape=(4, 4),
    )


# ---------------------------------------------------------------------------
# Bug 2: _compute_upwind_divergence produces nan when drift is zero
# ---------------------------------------------------------------------------


def _make_problem():
    """Build a minimal 1-D MFGProblem for FPGFDMSolver tests."""
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[10],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    comp = MFGComponents(
        hamiltonian=H,
        m_initial=lambda x: np.exp(-20 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.5 * (x - 0.5) ** 2,
    )
    return MFGProblem(
        geometry=grid,
        components=comp,
        T=0.5,
        Nt=5,
        sigma=0.3,
        coupling_coefficient=0.5,
    )


def _make_fp_gfdm_solver():
    """Construct FPGFDMSolver on a small 1-D point cloud."""
    from mfgarchon.alg.numerical.fp_solvers.fp_gfdm import FPGFDMSolver

    points = np.linspace(0.0, 1.0, 12).reshape(-1, 1)
    problem = _make_problem()
    solver = FPGFDMSolver(
        problem=problem,
        collocation_points=points,
        upwind_scheme="exponential",
        upwind_strength=1.0,
    )
    return solver


def test_upwind_divergence_zero_drift_no_invalid_float_op():
    """_compute_upwind_divergence must not produce an invalid float operation
    (nan/inf in intermediate values) when drift is entirely zero.

    On buggy code: cos_theta = dot(0, r) / (0 * r_norm) raises RuntimeWarning
    (invalid value encountered in scalar divide).  With errstate(invalid='raise')
    that becomes FloatingPointError, so the test FAILS on the unfixed code.
    After fix: zero-drift path bypasses the divide entirely; no warning raised.
    """
    solver = _make_fp_gfdm_solver()
    N = solver.n_points

    drift = np.zeros((N, 1))  # all-zero drift -- triggers division-by-zero bug
    density = np.ones(N) / N

    with np.errstate(invalid="raise"):
        result = solver._compute_upwind_divergence(drift_field=drift, density=density)

    assert np.all(np.isfinite(result)), (
        f"_compute_upwind_divergence returned non-finite values with zero drift: {result}"
    )


def test_upwind_divergence_partial_zero_drift_no_invalid_float_op():
    """Same invariant when only some points have zero drift."""
    solver = _make_fp_gfdm_solver()
    N = solver.n_points

    rng = np.random.default_rng(42)
    drift = rng.standard_normal((N, 1)) * 0.5
    drift[::2] = 0.0  # zero drift at every other point

    density = np.ones(N) / N

    with np.errstate(invalid="raise"):
        result = solver._compute_upwind_divergence(drift_field=drift, density=density)

    assert np.all(np.isfinite(result)), (
        f"_compute_upwind_divergence returned non-finite values with partial zero drift: {result}"
    )

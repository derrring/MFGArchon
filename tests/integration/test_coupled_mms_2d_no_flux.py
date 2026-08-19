"""Coupled 2D MMS on a no-flux box: the GFDM paper's manufactured pair, run through production.

The gap this fills. `test_coupled_mfg_mms.py` measures coupled EOC, but in **1D** and under
**periodic** BC (deliberately — its own note says periodic "avoids the no-flux conservative-Laplacian
boundary handling (#1075) so the measured error" is the interior scheme). So nothing measured coupled
convergence in 2D, and nothing measured it with a wall present at all.

THE MANUFACTURED PAIR
---------------------
Verbatim from the GFDM paper, `eq:mms_reference` / `eq:mms_system`, on Ω = (0,L)² with L = 20, T = 4,
σ = 1, c = 2π/L, ζ = 1/2:

    u(t,x) = a1(t)(cos(c x1) + cos(c x2)),          a1 = 1 + (T-t)/(2T)
    m(t,x) = L^-2 (1 + a2(t) cos(2c x1) cos(2c x2)), a2 = (2/5) cos(π t / (2T))

    -d_t u - (σ²/2)Δu + (1/2)|∇u|² + ζm = r_u
     d_t m - div(m ∇u) - (σ²/2)Δm       = r_m

The sources were derived symbolically (sympy) and cross-checked against a hand-written vectorised
form to 1.1e-16 / 3.3e-19 over 200 random (t,x). Two properties the paper states are reproduced as
independent confirmation of the transcription:

- ∫_Ω r_m dx = 0 exactly (the paper: "the manufactured forward equation is mass-consistent")
- ∂u/∂x_k = ∂m/∂x_k = 0 on every wall

The second matters here: α = -∇u, so α·n = 0 and hence J·n = αm - D∇m = 0. **The pair is exactly
compatible with the no-flux wall**, which is what lets it run on this box at all.

SIGN CONVENTIONS (read off the working tree, not memory; same as test_coupled_mfg_mms.py)
------------------------------------------------------------------------------------------
S_HJB = -d_t u + H(x, m, ∇u) - (σ²/2)Δu   with H = |p|²/(2λ) + f(m), λ = 1, f(m) = ζm
     => identical to the paper's r_u.
S_FP  =  d_t m + div(α m) - (σ²/2)Δm      with α = -coupling_coefficient·∇U, coefficient = 1
     => identical to the paper's r_m.

WHAT THIS STUDY CANNOT SEE
--------------------------
- **The wall is TANGENTIAL.** ∂_ν u = 0 makes α·n = 0, so this exercises the wall's *compatibility*
  and not its treatment of a normal drift. The non-tangential wall is
  `tests/unit/test_alg/test_fp_mms_wall_order_1728.py` (#2006), FP-only.
- **It does not measure the coupling direction**, and the paper says so of its own instance: "this
  leg measures discretization order and not the coupling direction analyzed above". Measured here,
  ζm is ~2.5% of the HJB residual (RMS 1.28e-03 against 1.24e-01 for |∇u|²/2), so a *model-side*
  perturbation of ζ is swamped by the discretization error, which is ~30% relative at these
  resolutions. That is a statement about resolution, not about the fixture: a *solver-side* error
  breaks it decisively, which is what the discrimination measurement below shows.
- Only `FDM_UPWIND`. The pair is method-agnostic; other schemes are separate parametrisations.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

L, T, SIGMA, ZETA = 20.0, 4.0, 1.0, 0.5
C = 2.0 * np.pi / L
LEVELS = ((11, 8), (21, 16), (31, 24))


def _a1(t):
    return 1.0 + 0.5 * (T - t) / T


def _a1p(_t):
    return -0.5 / T


def _a2(t):
    return 0.4 * np.cos(np.pi * t / (2.0 * T))


def _a2p(t):
    return -0.4 * (np.pi / (2.0 * T)) * np.sin(np.pi * t / (2.0 * T))


def u_star(t, x1, x2):
    return _a1(t) * (np.cos(C * x1) + np.cos(C * x2))


def m_star(t, x1, x2):
    return (1.0 + _a2(t) * np.cos(2 * C * x1) * np.cos(2 * C * x2)) / L**2


def s_hjb(t, x1, x2):
    """S_HJB = -d_t u + (1/2)|grad u|^2 + zeta*m - (sigma^2/2) Lap u."""
    du_dt = _a1p(t) * (np.cos(C * x1) + np.cos(C * x2))
    grad_sq = (_a1(t) * C) ** 2 * (np.sin(C * x1) ** 2 + np.sin(C * x2) ** 2)
    lap_u = -_a1(t) * C**2 * (np.cos(C * x1) + np.cos(C * x2))
    return -du_dt + 0.5 * grad_sq + ZETA * m_star(t, x1, x2) - 0.5 * SIGMA**2 * lap_u


def s_fp(t, x1, x2):
    """S_FP = d_t m - div(m grad u) - (sigma^2/2) Lap m."""
    dm_dt = _a2p(t) * np.cos(2 * C * x1) * np.cos(2 * C * x2) / L**2
    m = m_star(t, x1, x2)
    gm1 = -2 * C * _a2(t) * np.sin(2 * C * x1) * np.cos(2 * C * x2) / L**2
    gm2 = -2 * C * _a2(t) * np.cos(2 * C * x1) * np.sin(2 * C * x2) / L**2
    gu1 = -_a1(t) * C * np.sin(C * x1)
    gu2 = -_a1(t) * C * np.sin(C * x2)
    lap_u = -_a1(t) * C**2 * (np.cos(C * x1) + np.cos(C * x2))
    lap_m = -8 * C**2 * _a2(t) * np.cos(2 * C * x1) * np.cos(2 * C * x2) / L**2
    return dm_dt - (gm1 * gu1 + gm2 * gu2 + m * lap_u) - 0.5 * SIGMA**2 * lap_m


def _split(x):
    a = np.asarray(x, dtype=float)
    return (a[..., 0], a[..., 1]) if a.ndim >= 2 and a.shape[-1] == 2 else (a.ravel(), a.ravel())


def _solve(nx: int, nt: int):
    grid = TensorProductGrid(bounds=[(0.0, L)] * 2, Nx_points=[nx] * 2, boundary_conditions=no_flux_bc(dimension=2))
    components = MFGComponents(
        m_initial=lambda p: float(m_star(0.0, np.asarray(p).ravel()[0], np.asarray(p).ravel()[1])),
        u_terminal=lambda p: float(u_star(T, np.asarray(p).ravel()[0], np.asarray(p).ravel()[1])),
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: ZETA * m,
            coupling_dm=lambda _m: ZETA,
        ),
    )
    problem = MFGProblem(
        geometry=grid,
        T=T,
        Nt=nt,
        sigma=SIGMA,
        coupling_coefficient=1.0,  # alpha = -grad U, matching the paper's div(m grad u)
        components=components,
        source_term_hjb=lambda x, m, v, t: s_hjb(t, *_split(x)).ravel(),
        source_term_fp=lambda x, m, v, t: s_fp(t, *_split(x)).ravel(),
    )
    result = FixedPointIterator(
        problem, hjb_solver=HJBFDMSolver(problem), fp_solver=FPFDMSolver(problem), relaxation=1.0
    ).solve(max_iterations=200, tolerance=1e-6, verbose=False)
    # BEFORE any error metric: an EOC read off an unconverged outer iteration measures how far the
    # iteration got in N steps, not a discretization order (#1998/#1728).
    assert result.converged, (
        f"Picard did not converge at Nx={nx} (iters={result.iterations}); an EOC measured on an "
        "unconverged solve is a statistic about the iteration, not about the scheme."
    )
    points = problem.geometry.get_spatial_grid()
    shape = tuple(problem.geometry.Nx_points)
    dx = L / (nx - 1)
    x1, x2 = points[:, 0].reshape(shape), points[:, 1].reshape(shape)
    u_err = np.asarray(result.U)[0] - u_star(0.0, x1, x2)
    m_err = np.asarray(result.M)[-1] - m_star(T, x1, x2)
    return float(np.sqrt((u_err**2).sum()) * dx), float(np.sqrt((m_err**2).sum()) * dx)


def _eoc(errors, levels):
    return [
        float(np.log(errors[i] / errors[i + 1]) / np.log(levels[i + 1][0] / levels[i][0]))
        for i in range(len(errors) - 1)
    ]


@pytest.mark.integration
@pytest.mark.slow
def test_coupled_2d_no_flux_converges_at_first_order():
    """First order in both fields, with an error LEVEL beside the order.

    The level is not decoration. An order alone is a min over the HJB scheme, the FP scheme and the
    wall closure, so it cannot say which produced the O(h); the level separates them. Bounds are set
    from measurement, not chosen: baseline `eu` at the finest level is 5.959e-01, and a solver whose
    D is off by 1.21x gives 1.326e+00 (EOC u collapsing 1.01 -> 0.38), by 2x gives 4.465e+00
    (EOC u -> 0.06). 9e-01 separates them with margin on both sides.
    """
    errors = [_solve(nx, nt) for nx, nt in LEVELS]
    eu = [e[0] for e in errors]
    em = [e[1] for e in errors]
    order_u, order_m = _eoc(eu, LEVELS), _eoc(em, LEVELS)

    assert all(0.8 <= o <= 1.3 for o in order_u), f"u is not first order: {order_u} (errors {eu})"
    assert all(0.8 <= o <= 1.3 for o in order_m), f"m is not first order: {order_m} (errors {em})"
    assert eu[-1] < 9e-01, (
        f"u error level {eu[-1]:.4e} at the finest grid exceeds 9e-01. The order can stay near 1 "
        f"while the constant moves, which is what a wrong diffusion coefficient looks like here."
    )

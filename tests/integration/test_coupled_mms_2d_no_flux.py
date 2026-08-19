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

The pair actually used here is NOT the paper's verbatim one -- see C and BETA below for why and for
what each change buys. The sources were re-derived for THIS pair symbolically (sympy, from the
definitions of u and m) and cross-checked against the hand-written vectorised forms in this file to
5.6e-17 / 1.1e-19 over 400 random (t,x), against residual scales 2.3e-01 / 4.5e-04. That comparison
can fail: perturbing s_fp by 1e-7 relative registers as 4.5e-11.

Two properties the paper states also hold for this pair, verified symbolically on it:

- ∫_Ω r_m dx = 0 exactly (the paper: "the manufactured forward equation is mass-consistent")
- ∂u/∂x_k = ∂m/∂x_k = 0 on each of the four walls

**Neither is a check on the transcription**, and an earlier version of this docstring called them
"independent confirmation" of it. The first is one scalar per time and is blind to any error that
integrates away: an r_m carrying a spurious (1/7)cos(2c x1)cos(4c x2) integrates to exactly 0 as
well. The second is a property of u and m alone and never touches the sources. Only the sympy
cross-check above bears on the transcription.

The second property earns its place for a different reason: α = -∇u, so α·n = 0 and hence
J·n = αm - D∇m = 0. **The pair is exactly compatible with the no-flux wall**, which is what lets
it run on this box at all.

SIGN CONVENTIONS (read off the working tree, not memory; same as test_coupled_mfg_mms.py)
------------------------------------------------------------------------------------------
S_HJB = -d_t u + H(x, m, ∇u) - (σ²/2)Δu   with H = |p|²/(2λ) + f(m), λ = 1, f(m) = ζm
     => identical to the paper's r_u.
S_FP  =  d_t m + div(α m) - (σ²/2)Δm      with α = -c·∇U, c = 1
     => identical to the paper's r_m.

c is NOT `problem.coupling_coefficient`. It comes from `fp_drift_coefficient`
(`utils/pde_coefficients.py:118`), which for a quadratic-MINIMIZE SeparableHamiltonian returns
1/control_cost.lambda_ and never reaches the `coupling_coefficient` fallback (#1420 / G-017). Here
λ = 1, so c = 1. An earlier version of this file credited the value to the `coupling_coefficient=1.0`
argument below -- wrong source, right number, which is exactly why the misattribution left no trace.

WHAT THIS STUDY CANNOT SEE
--------------------------
- **The wall is TANGENTIAL.** ∂_ν u = 0 makes α·n = 0, so this exercises the wall's *compatibility*
  and not its treatment of a normal drift. The non-tangential wall is
  `tests/unit/test_alg/test_fp_mms_wall_order_1728.py` (#2006), FP-only.
- **It does not measure the coupling direction**, and the paper says so of its own instance: "this
  leg measures discretization order and not the coupling direction analyzed above". Measured on THIS
  pair over the space-time box: ζm has RMS 1.26e-03 — 8.1% of the |∇u|²/2 term (1.55e-02) and 1.1%
  of the whole HJB residual (1.17e-01, dominated by -∂_t u at 1.03e-01). Against that, the
  discretization error at the finest level is 0.41% relative in u and 6.6% in m. So a *model-side*
  perturbation of ζ is comparable to or smaller than the error already present, and this study
  cannot resolve it. That is a statement about resolution, not about the fixture: a *solver-side*
  error breaks it decisively, which is what the discrimination measurement below shows.
- Only `FDM_UPWIND`. The pair is method-agnostic, but the LIBRARY mostly is not: measured at
  Nx=21 on this exact fixture, 2 of 8 solver pairings run at all.

      HJB x FP                 outcome
      FDM  x FDM               converged, 4 iter, eu = 1.5759e-01
      GFDM x FDM               converged, 4 iter, eu = 1.9699e-02
      WENO x FDM               NotImplementedError: source_term is 1D-path only, this is 2D
      SemiLagrangian x FDM     NotImplementedError: solve_hjb_system takes no source_term
      FDM  x GFDM/Particle/SL  NotImplementedError: solve_fp_system takes no source_term
      FDM  x FVM               ValueError: cannot broadcast (21,21) against (882,)

  The NotImplementedErrors are fail-loud guards, not bugs -- but they mean an MMS cannot reach
  those solvers at all (#1991 for the HJB side). GFDM's source WAS verified to reach it: zeroing
  s_hjb moves eu from 1.9699e-02 to 9.7640e+00, a factor of 496. The FVM crash is specific to the
  2D-and-source cell -- 1D+source, 2D-no-source and both FDM cells all run -- so it is neither a
  general 2D defect nor a general source defect.
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
# HALF a period, not the paper's full one. sin(cL) = 0 either way, so the pair stays Neumann
# compatible -- but cos(pi x/L) is NOT L-periodic, so the mirror ghost and the periodic ghost no
# longer coincide and the wall closure is actually exercised. Measured: with the paper's
# c = 2 pi / L, swapping the entire boundary-condition family for `periodic_bc` left every
# assertion green; with c = pi / L it moves eu from 3.0125e-01 to 1.2939e+01 at Nx=11 (42x) and
# collapses EOC u to -0.118 / 0.067. That difference IS the wall closure.
C = np.pi / L
# Asymmetry between the axes, deliberately. With the paper's pair, u, m and BOTH sources are exactly
# transpose-symmetric, so `eu` computed from U[0] and from U[0].T is bit-identical and the fixture
# cannot express "read the potential along the wrong axis" -- one of the three mutants #2017 cites as
# the reason to work in 2D. BETA breaks it in u; the 2c/4c pair breaks it in m. Measured: eu from
# U[0].T is now 45.7x / 81.8x / 119.5x the eu from U[0] across the three levels.
BETA = 0.6
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
    return _a1(t) * (np.cos(C * x1) + BETA * np.cos(C * x2))


def m_star(t, x1, x2):
    return (1.0 + _a2(t) * np.cos(2 * C * x1) * np.cos(4 * C * x2)) / L**2


def s_hjb(t, x1, x2):
    """S_HJB = -d_t u + (1/2)|grad u|^2 + zeta*m - (sigma^2/2) Lap u."""
    du_dt = _a1p(t) * (np.cos(C * x1) + BETA * np.cos(C * x2))
    grad_sq = (_a1(t) * C) ** 2 * (np.sin(C * x1) ** 2 + BETA**2 * np.sin(C * x2) ** 2)
    lap_u = -_a1(t) * C**2 * (np.cos(C * x1) + BETA * np.cos(C * x2))
    return -du_dt + 0.5 * grad_sq + ZETA * m_star(t, x1, x2) - 0.5 * SIGMA**2 * lap_u


def s_fp(t, x1, x2):
    """S_FP = d_t m - div(m grad u) - (sigma^2/2) Lap m."""
    dm_dt = _a2p(t) * np.cos(2 * C * x1) * np.cos(4 * C * x2) / L**2
    m = m_star(t, x1, x2)
    gm1 = -2 * C * _a2(t) * np.sin(2 * C * x1) * np.cos(4 * C * x2) / L**2
    gm2 = -4 * C * _a2(t) * np.cos(2 * C * x1) * np.sin(4 * C * x2) / L**2
    gu1 = -_a1(t) * C * np.sin(C * x1)
    gu2 = -BETA * _a1(t) * C * np.sin(C * x2)
    lap_u = -_a1(t) * C**2 * (np.cos(C * x1) + BETA * np.cos(C * x2))
    lap_m = -20 * C**2 * _a2(t) * np.cos(2 * C * x1) * np.cos(4 * C * x2) / L**2
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
    """Order against the MESH ratio h_i/h_{i+1}, not the point-count ratio.

    `dx = L/(nx-1)`, so refining 11 -> 21 -> 31 halves h and then takes 2/3 of it: ratios 2.0 and
    1.5, not 21/11 = 1.909 and 31/21 = 1.476. A first draft divided by the point counts and inflated
    every order by 7.2% and 4.1% -- while `_solve` twelve lines up computed `dx` with `nx - 1`
    correctly, so the file disagreed with itself.
    """
    h = [L / (nx - 1) for nx, _ in levels]
    return [float(np.log(errors[i] / errors[i + 1]) / np.log(h[i] / h[i + 1])) for i in range(len(errors) - 1)]


@pytest.mark.integration
@pytest.mark.slow
def test_coupled_2d_no_flux_converges_at_first_order():
    """First order in both fields, with an error LEVEL beside the order.

    The level is not decoration. An order alone is a min over the HJB scheme, the FP scheme and the
    wall closure, so it cannot say which produced the O(h); the level separates them.

    Bounds are set from measurement. Baseline against two solver-side diffusion errors -- sigma is
    wrong at problem CONSTRUCTION while s_hjb/s_fp stay closed over the true SIGMA, so the
    manufactured pair is untouched and only the PDE the scheme discretizes is wrong:

        D factor   eu(Nx=11)   eu(Nx=21)   eu(Nx=31)   EOC u
        1.00       3.0125e-01  1.5759e-01  1.0534e-01   0.935,  0.993
        1.21       3.5252e-01  2.4572e-01  2.2015e-01   0.521,  0.271
        2.00       9.6882e-01  9.5405e-01  9.5607e-01   0.022, -0.005

    So 1.5e-01 sits 1.42x above the baseline and 1.47x below the 1.21x mutant. The previous bound
    of 9e-01 came from a different fixture and would have PASSED the 1.21x mutant outright.

    `em` is not the discriminating field for this defect and no level bound is asserted on it: at
    1.21x it moves only 3.383e-03 -> 3.487e-03 and its EOC stays inside the band at 0.956 / 0.945.
    Only at 2x does it leave, at 0.886 / 0.777.

    Do NOT reproduce these by mutating `problem.sigma` after construction -- that is inert and
    silently so. `get_diffusion_coefficient_field` (mfg_problem.py:1416) resolves
    override -> volatility_field -> self.sigma, and `volatility_field` is snapshotted at
    construction (:560), so the solve comes back byte-identical while `problem.diffusion` reports
    the mutated value. Measured: 1.21x and 2.0x that way both reproduce the baseline to every
    printed digit.
    """
    errors = [_solve(nx, nt) for nx, nt in LEVELS]
    eu = [e[0] for e in errors]
    em = [e[1] for e in errors]
    order_u, order_m = _eoc(eu, LEVELS), _eoc(em, LEVELS)

    assert all(0.8 <= o <= 1.3 for o in order_u), f"u is not first order: {order_u} (errors {eu})"
    assert all(0.8 <= o <= 1.3 for o in order_m), f"m is not first order: {order_m} (errors {em})"
    assert eu[-1] < 1.5e-01, (
        f"u error level {eu[-1]:.4e} at the finest grid exceeds 1.5e-01. The order can stay near 1 "
        f"while the constant moves, which is what a wrong diffusion coefficient looks like here."
    )

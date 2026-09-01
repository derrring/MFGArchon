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
definitions of u and m) and cross-checked against the hand-written vectorised forms this file used to carry (#2201 replaced
them with the shared assembly in `mfgarchon.utils.manufactured`; the cross-check stands as the
provenance of the pair, not as a description of code still here) to
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
  discretization error at the finest level is 0.41% relative in u and ~~6.6%~~ **0.735%** in m
  [SUPERSEDED 2026-08-31] SUPERSEDED-BY: #2189 -- #2145 cut the m error tenfold and this
  argument rests on its size, so the figure is restated rather than left standing. It does not
  reverse the conclusion: 0.735% is still comparable to the 1.1% zeta-m share, so the study
  still cannot resolve a model-side perturbation of zeta. u is unchanged (measured 0.406%). So a *model-side*
  perturbation of ζ is comparable to or smaller than the error already present, and this study
  cannot resolve it. That is a statement about resolution, not about the fixture: a *solver-side*
  error breaks it decisively, which is what the discrimination measurement below shows.
- **It cannot resolve a coefficient error below ~1.5% (diffusion) or ~5% (drift scale).**
  ~~Measured: a 5% error in the diffusion coefficient (k = 1.05) or in the drift scale
  (lambda = 1.05) passes every assertion in this file.~~ [SUPERSEDED 2026-08-31]
  SUPERSEDED-BY: #2189. Both of those now FAIL, and on the m order rather than the u order:
  k = 1.05 gives EOC(m) 0.606 / 0.636 and lambda = 1.05 gives 0.696 / 0.702, against a bound
  of 0.70 -- while EOC(u) passes in both (0.915 / 0.909 and 0.868 / 0.827). Removing the
  rectangle-rule error made em the sensitive column. The surviving floor is measured at the
  order assertion below. Any order reported here is an order at that sensitivity, not a
  certificate below it.
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
from mfgarchon.utils.manufactured import ManufacturedPair, check_pair, fp_source, hjb_source

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


LAMBDA = 1.0

# The Hamiltonian the SOURCE is assembled from, deliberately SEPARATE from the one `_solve` puts on
# the problem. Every discrimination table in this file supplies a wrong coefficient at problem
# CONSTRUCTION while the source stays closed over the true constants; routing both through one
# object would make each mutant self-consistent and the test unable to fail at all.
_TRUE_HAMILTONIAN = SeparableHamiltonian(
    control_cost=QuadraticControlCost(lambda_=LAMBDA),
    coupling=lambda m: ZETA * m,
    coupling_dm=lambda _m: ZETA,
)

# The exact pair and its analytic derivatives. The ASSEMBLY of these into S_HJB / S_FP is owned by
# `mfgarchon.utils.manufactured` (#2201) and is no longer written here: the sign conventions in this
# file's header are what that module encodes, and stating them twice is how they drift apart.
PAIR = ManufacturedPair(
    u=lambda t, x: u_star(t, x[..., 0], x[..., 1]),
    u_t=lambda t, x: _a1p(t) * (np.cos(C * x[..., 0]) + BETA * np.cos(C * x[..., 1])),
    grad_u=lambda t, x: np.stack(
        [-_a1(t) * C * np.sin(C * x[..., 0]), -BETA * _a1(t) * C * np.sin(C * x[..., 1])], axis=-1
    ),
    hess_u=lambda t, x: _diag_hessian(
        -_a1(t) * C**2 * np.cos(C * x[..., 0]), -BETA * _a1(t) * C**2 * np.cos(C * x[..., 1])
    ),
    m=lambda t, x: m_star(t, x[..., 0], x[..., 1]),
    m_t=lambda t, x: _a2p(t) * np.cos(2 * C * x[..., 0]) * np.cos(4 * C * x[..., 1]) / L**2,
    grad_m=lambda t, x: np.stack(
        [
            -2 * C * _a2(t) * np.sin(2 * C * x[..., 0]) * np.cos(4 * C * x[..., 1]) / L**2,
            -4 * C * _a2(t) * np.cos(2 * C * x[..., 0]) * np.sin(4 * C * x[..., 1]) / L**2,
        ],
        axis=-1,
    ),
    hess_m=lambda t, x: _hess_m(t, x),
    name="coupled_2d_no_flux",
)


def _diag_hessian(d00, d11):
    """A ``(N, 2, 2)`` Hessian with zero cross term -- correct for u*, which is a separable SUM."""
    hess = np.zeros((len(np.atleast_1d(d00)), 2, 2))
    hess[:, 0, 0], hess[:, 1, 1] = d00, d11
    return hess


def _hess_m(t, x):
    """m* is a PRODUCT, so its cross derivative is non-zero -- the one this file supplies that an
    isotropic sigma multiplies by exactly zero. `check_pair` is what audits it; see the test below."""
    hess = np.zeros((len(x), 2, 2))
    common = _a2(t) * np.cos(2 * C * x[..., 0]) * np.cos(4 * C * x[..., 1]) / L**2
    hess[:, 0, 0] = -4 * C**2 * common
    hess[:, 1, 1] = -16 * C**2 * common
    hess[:, 0, 1] = hess[:, 1, 0] = 8 * C**2 * _a2(t) * np.sin(2 * C * x[..., 0]) * np.sin(4 * C * x[..., 1]) / L**2
    return hess


s_hjb = hjb_source(PAIR, _TRUE_HAMILTONIAN, SIGMA)
s_fp = fp_source(PAIR, _TRUE_HAMILTONIAN, SIGMA)


def _points(x):
    """Return the solver's point array as ``(N, 2)``. Raises rather than guessing.

    A previous version split this into ``(x1, x2)`` and fell back to ``(a.ravel(), a.ravel())`` for
    any other shape, which aliases x2 := x1 and evaluates BOTH manufactured sources on the
    degenerate diagonal x1 == x2 -- a plausible, wrong source field, silently. In an oracle that is
    the one failure that cannot be afforded: the test would keep passing while measuring the scheme
    against the wrong exact solution. Instrumented over a full converged solve, that branch was
    taken 0 times out of 64, so it bought nothing and risked everything. The repo's own rule is at
    `utils/pde_coefficients.py:114` -- "the prior silent fallback masked a malformed problem".

    Note this convention is not universal in the library: `FPFVMSolver` passes
    `geometry.meshgrid()`, a (d, *shape) tuple, where FDM and the whole HJB side pass (N, d)
    points (#2019). Raising here is what makes that difference visible instead of silently
    diagonal.
    """
    a = np.asarray(x, dtype=float)
    if a.ndim >= 2 and a.shape[-1] == 2:
        return a.reshape(-1, 2)
    raise TypeError(
        f"_points expected an (N, 2) point array from the solver, got shape {a.shape}. "
        "Aliasing x2 := x1 would evaluate the manufactured sources on the diagonal and the test "
        "would pass while measuring the wrong exact solution (see #2019 for the FVM convention)."
    )


def _build_problem(nx: int, nt: int):
    grid = TensorProductGrid(bounds=[(0.0, L)] * 2, Nx_points=[nx] * 2, boundary_conditions=no_flux_bc(dimension=2))
    components = MFGComponents(
        m_initial=lambda p: float(m_star(0.0, np.asarray(p).ravel()[0], np.asarray(p).ravel()[1])),
        u_terminal=lambda p: float(u_star(T, np.asarray(p).ravel()[0], np.asarray(p).ravel()[1])),
        hamiltonian=SeparableHamiltonian(
            # LAMBDA, not a literal: the source's Hamiltonian is a separate OBJECT by design
            # (see _TRUE_HAMILTONIAN) but the two must agree on the VALUE, or the source
            # manufactures a different equation than the solver integrates. `control_cost=` and
            # `lambda_=` are aliases for the same parameter, so the divergence was invisible by
            # eye. The documented mutation protocol still works: replace this with 1.05 * LAMBDA.
            control_cost=QuadraticControlCost(control_cost=LAMBDA),
            coupling=lambda m: ZETA * m,
            coupling_dm=lambda _m: ZETA,
        ),
    )
    problem = MFGProblem(
        geometry=grid,
        T=T,
        Nt=nt,
        sigma=SIGMA,
        # INERT here, and kept only because MFGProblem's own default (0.5) would be equally
        # inert and more confusing. The drift scale comes from `fp_drift_coefficient` = 1/lambda,
        # never from this argument (see SIGN CONVENTIONS above). Measured: solves at
        # coupling_coefficient = 0.5 / 1.0 / 7.0 are bit-identical, max|dU| = max|dM| = 0.000e+00,
        # against a control where sigma=1.1 moves the same solve by 1.672e-02.
        coupling_coefficient=1.0,
        components=components,
        source_term_hjb=lambda x, m, v, t: s_hjb(t, _points(x)),
        source_term_fp=lambda x, m, v, t: s_fp(t, _points(x)),
    )
    return problem


def _solve(nx: int, nt: int):
    problem = _build_problem(nx, nt)
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

    WHAT EACH ASSERTION ACTUALLY CATCHES, measured over two independent solver-side defect
    families. In both, the manufactured pair is untouched -- the wrong coefficient is supplied at
    problem CONSTRUCTION while s_hjb/s_fp stay closed over the true constants -- so only the PDE
    the scheme discretizes is wrong.

    Family 1, wrong diffusion (sigma at construction, so D = k*sigma^2/2):

        k       eu(11)      eu(21)      eu(31)      EOC u           EOC-assert  level-assert
        1.00    ~~3.0125e-01  1.5759e-01  1.0534e-01   0.935,  0.993~~ pass    pass
                3.0111e-01  1.5775e-01  1.0549e-01   0.933,  0.992  pass        pass
        # [SUPERSEDED 2026-08-31] SUPERSEDED-BY: #2201. The CLEAN row is the reference row of
        # BOTH tables below and had gone stale at #2145; re-measured on the current tree and
        # updated in BOTH -- an earlier revision of this note said 'both' while updating only
        # Family 1, leaving Family 2's clean row byte-identical to the string struck above it.
        # Every
        # pass/FAIL verdict in both tables is unaffected. The mutant rows are NOT re-measured
        # here and remain pre-#2145 -- lambda=1.50's eu(11) reads 5.4156e-01 against a measured
        # 5.2801e-01, so read the mutant columns as ordinal, not as current levels.
        1.05    3.0241e-01  1.6042e-01  1.1106e-01   0.915,  0.907  pass        pass
        1.10    3.1101e-01  1.7715e-01  1.3559e-01   0.812,  0.659  FAIL        pass
        1.15    3.2640e-01  2.0434e-01  1.7092e-01   0.676,  0.440  FAIL        FAIL
        1.21    3.5252e-01  2.4572e-01  2.2015e-01   0.521,  0.271  FAIL        FAIL
        2.00    9.6882e-01  9.5405e-01  9.5607e-01   0.022, -0.005  FAIL        FAIL

    Family 2, wrong drift scale (control_cost lambda, so alpha = -grad U / lambda):

        lambda  eu(11)      eu(21)      eu(31)      EOC u           EOC-assert  level-assert
        1.00    3.0111e-01  1.5775e-01  1.0549e-01   0.933,  0.992  pass        pass
        1.05    3.1276e-01  1.7232e-01  1.2388e-01   0.860,  0.814  pass        pass
        1.20    3.8375e-01  2.6818e-01  2.3543e-01   0.517,  0.321  FAIL        FAIL
        1.50    5.4156e-01  4.5111e-01  4.2713e-01   0.264,  0.135  FAIL        FAIL

    Two things follow, and the first corrects an earlier version of this docstring.

    **The level bound is not a discriminator for coefficient errors, and claiming it was is what
    an earlier version of this file did.** Across 10 mutants in two families it never fails alone:
    every row where it fails, the EOC assertion has already failed, and at k = 1.10 the EOC fails
    while the level passes. The retracted claim -- "the previous bound of 9e-01 would have PASSED
    the 1.21x mutant" -- is false for the same reason: EOC rejects that mutant on 0.521 / 0.271
    whatever the level bound is.

    It is kept, relabelled, as a **regression guard on the constant**: EOC is a ratio and is blind
    to an error scaled uniformly across levels, so a future change that preserves first order while
    multiplying the constant would pass the EOC assertion. No mutant here exhibits that, so this is
    an unexercised guard and is labelled as one rather than sold as discrimination.

    ~~**The measured detection floor is between 5% and 10%.** A 5% error in either coefficient
    passes the ENTIRE test (k = 1.05 and lambda = 1.05 both pass both assertions).~~
    [SUPERSEDED 2026-08-31] SUPERSEDED-BY: #2189. **Both of those now fail**, on the m order:
    k = 1.05 gives EOC(m) 0.606 / 0.636 and lambda = 1.05 gives 0.696 / 0.702 against the 0.70
    bound, while EOC(u) passes in both (0.915 / 0.909 and 0.868 / 0.827). The floor is now ~1.5%
    in the diffusion coefficient -- k = 1.015 survives at 0.707 and k = 1.020 fails at 0.691 --
    and ~5% in the drift scale, where lambda = 1.02 survives at 0.760 and lambda = 1.05 fails by
    0.004. That is a fact about these resolutions, not about the fixture, and it belongs beside
    any order this test reports.

    ~~`em` is weaker still: at k = 1.21 it moves only 3.383e-03 -> 3.487e-03 with EOC 0.956 / 0.945,
    inside the band.~~ [SUPERSEDED 2026-08-31] SUPERSEDED-BY: #2189. Those figures were measured
    before #2145 and the em column above is stale throughout: em is now ~10x smaller (9.187e-04 /
    5.434e-04 / 3.800e-04) and its EOC is 0.758 / 0.882. The eu column moved only in the fourth
    digit ON THE CLEAN ROW (3.0125e-01 -> 3.0111e-01); do not read that as "the eu tables are
    unaffected", which is a generalisation from one row to ten and false for the mutant rows --
    lambda = 1.50 moved in the second digit (5.4156e-01 -> 5.2801e-01, 2.5% relative). Their
    pass/FAIL conclusions all still hold. The retracted sentence also had the conclusion
    backwards: em is now the STRONGER order column, reading 0.339 / 0.308 at k = 1.21 where it once
    read 0.956 / 0.945. See the measured table at the order assertion. No level bound is asserted
    on em.

    Do NOT reproduce any of this by mutating `problem.sigma` after construction -- that is inert
    and silently so. `get_diffusion_coefficient_field` (mfg_problem.py:1416) resolves
    override -> volatility_field -> self.sigma, and `volatility_field` is snapshotted at
    construction (:560), so the solve comes back byte-identical while `problem.diffusion` reports
    the mutated value. Measured: k = 1.21 and k = 2.0 that way both reproduce the baseline to
    every printed digit.
    """
    errors = [_solve(nx, nt) for nx, nt in LEVELS]
    eu = [e[0] for e in errors]
    em = [e[1] for e in errors]
    order_u, order_m = _eoc(eu, LEVELS), _eoc(em, LEVELS)

    assert all(0.8 <= o <= 1.3 for o in order_u), f"u is not first order: {order_u} (errors {eu})"
    # 0.70, not 0.80, and the gap is measured rather than granted. After #2145 the m error fell by
    # ~10x (9.876e-03 -> 9.187e-04 at Nx=11) because the rectangle-rule measure that dominated it is
    # gone, and what remains has not reached its asymptotic range at these resolutions: the order
    # RISES 0.758 -> 0.882 -> 0.916 across Nx = 11/21/31/41. First order approached from below, not
    # a defect -- but the old bound read it as one and turned the improvement red (#2189).
    #
    # The widening costs no kill, and that is measured, not assumed. Re-running the diffusion
    # family against the post-#2145 em (sigma mutated at CONSTRUCTION, source terms untouched):
    #
    #     mutant          EOC m          min      0.80   0.70
    #     clean           0.758, 0.882   0.758    FAIL   pass    <- the old bound rejects CLEAN
    #     sigma k=1.015   0.707, 0.795   0.707    FAIL   pass    <- nearest survivor
    #     sigma k=1.020   0.691, 0.768   0.691    FAIL   FAIL
    #     sigma k=1.05    0.606, 0.636   0.606    FAIL   FAIL
    #     sigma k=1.10    0.495, 0.485   0.485    FAIL   FAIL
    #     sigma k=1.21    0.339, 0.308   0.308    FAIL   FAIL
    #     lambda 1.02     0.760, 0.874   0.760    FAIL   pass    <- ABOVE clean
    #     lambda 1.05     0.696, 0.702   0.696    FAIL   FAIL    <- nearest kill, by 0.004
    #
    # An earlier version of this comment measured only k = 1.00 / 1.10 / 1.21 and concluded "0.70
    # falls in a wide gap". It does not. The gap between measured points is not measured, and the
    # unsampled region is where the mutants are: the nearest survivor sits at 0.707 and the nearest
    # kill at 0.696, so the live window is 0.011 wide, not 0.263.
    #
    # What actually justifies 0.70, and it is a weaker claim than the retracted one: the CLEAN run
    # is at 0.758, so no admissible bound exists above it and the old 0.80 was not a discriminator
    # at all -- it rejected the correct solution. Against the tightest admissible bound (~0.75),
    # 0.70 costs the sigma mutants in 0.5%-1.5%, and buys 0.058 of headroom over clean. That trade
    # is deliberate: with 0.058 of margin a platform whose EOC shifts by more re-reds this test in
    # exactly the way #2189 is fixing, and every other assertion here has a floor above 1.5%.
    # This also corrects the docstring: em is no longer the weak column. Before #2145 a 21% error
    # left its EOC at 0.956 / 0.945, inside the band and killing nothing; it now reads 0.339 / 0.308.
    # Removing the rectangle-rule error turned em from an unexercised guard into a working one.
    assert all(0.70 <= o <= 1.3 for o in order_m), f"m is not first order: {order_m} (errors {em})"
    # Regression guard on the constant, NOT a discriminator -- see the tables above. Across ten
    # measured coefficient mutants this never fails without the EOC assertion failing first.
    assert eu[-1] < 1.5e-01, (
        f"u error level {eu[-1]:.4e} at the finest grid exceeds 1.5e-01 while the order stayed in "
        f"band (order_u={order_u}). EOC is a ratio and cannot see an error scaled uniformly across "
        f"levels; this bound is the only thing that can."
    )


@pytest.mark.integration
def test_the_manufactured_pair_is_its_own_derivatives():
    """The pair's six analytic derivatives, audited against a finite difference of u* and m*.

    This is the ONLY check here whose oracle is outside the scheme and outside the assembly. The
    convergence study above cannot do it: an isotropic sigma contracts `tr(D . Hess)` over the
    diagonal only, so `hess_m`'s cross term -- the one derivative in this file that no other line
    states -- is multiplied by exactly zero and a wrong value for it is invisible to every
    assertion above. Measured: flipping its sign moves s_fp by 0.000e+00 at this sigma.

    It also covers the direction the study is blind to for a different reason: `u*` is a separable
    SUM, so `hess_u`'s cross term is structurally zero and no choice of sigma can exercise it.
    """
    rng = np.random.default_rng(20260831)
    x = rng.uniform(0.0, L, size=(200, 2))
    check_pair(PAIR, 0.37 * T, x)


@pytest.mark.integration
def test_the_pair_audit_would_catch_a_wrong_cross_derivative():
    """Discrimination for the test above: it must FAIL on the defect it claims to catch, and that
    defect must be invisible to the source assembly -- otherwise the audit is redundant, not the
    only check."""
    rng = np.random.default_rng(20260831)
    x = rng.uniform(0.0, L, size=(200, 2))
    broken = ManufacturedPair(
        **{**PAIR.__dict__, "hess_m": lambda t, xx: _hess_m(t, xx) * np.array([[1, -1], [-1, 1]])}
    )

    assert np.max(np.abs(fp_source(broken, _TRUE_HAMILTONIAN, SIGMA)(0.37 * T, x) - s_fp(0.37 * T, x))) == 0.0
    with pytest.raises(ValueError, match="hess_m"):
        check_pair(broken, 0.37 * T, x)


@pytest.mark.integration
def test_the_source_and_the_solver_agree_on_the_coefficients():
    """The two Hamiltonians are separate OBJECTS on purpose; they must agree on the VALUES.

    `_TRUE_HAMILTONIAN` must not be the problem's Hamiltonian -- every mutant in the tables above
    perturbs the problem while the source stays closed over the true constants, and one object would
    make each mutant self-consistent and this whole file unable to fail. But separateness is exactly
    what lets the two drift: `control_cost=` and `lambda_=` are aliases for one parameter, so a
    literal at one site and a named constant at the other look identical by eye.

    Detection belonged only to the slow tier before this test existed. Measured: with the source at
    LAMBDA = 1.0 and the solver at a literal 1.0, changing LAMBDA alone left the CI-selected tests
    green (2 passed, 1 deselected) while moving the manufactured source by 1.888e-02 against a scale
    of 2.314e-01 -- the scheme verified against an equation nobody solves, which is the failure
    `mfgarchon.utils.manufactured` exists to prevent.
    """
    problem = _build_problem(*LEVELS[0])
    solver_side = problem.components.hamiltonian.control_cost.lambda_
    source_side = _TRUE_HAMILTONIAN.control_cost.lambda_
    assert solver_side == source_side, (
        f"the solver integrates lambda={solver_side} while the manufactured source was assembled "
        f"with lambda={source_side}; the study would measure the scheme against an equation nobody "
        f"solves, and converge cleanly doing it."
    )
    assert problem.sigma == SIGMA, f"solver sigma={problem.sigma} vs source sigma={SIGMA}"

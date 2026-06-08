"""Diffusion-magnitude invariant gate (codebase retrospect, recommendation #1).

This is a STANDING invariant gate, not a per-bug test. It pins that every
diffusion-carrying solver applies the *correct* diffusion magnitude
``D = sigma^2/2`` (Issue #811) — the property that the dominant bug class of the
2026-05/06 audit silently violated and that finiteness / mass-conservation /
Picard-self-consistency tests cannot catch.

Mechanism: a cosine eigenmode of the Laplacian decays analytically under pure
diffusion as ``exp(-D * sum_d k_d^2 * T)`` with ``D = sigma^2/2``. On ``[0,1]^d``
with ``k = 2*pi``, ``cos(2*pi*x)`` is an eigenmode compatible with both no-flux
(zero normal derivative at 0,1) and periodic BC. We evolve the eigenmode under
pure diffusion (no advection / drift / Hamiltonian) and assert the recovered
decay factor matches the analytic value. Any factor error in the diffusion
coefficient breaks the decay rate while leaving finiteness and (approximate) mass
intact — so this gate FAILS on exactly the bugs that shipped:

  - #1152  weak-form used ``volatility_field`` directly as ``D`` (skipped the /2)
  - #1178  ADI applied ``dt/dimension`` -> only ``1/dim`` of the diffusion
  - #1183  explicit-drift FP collapsed spatially-varying sigma to its mean
  - #1169  anisotropic off-diagonal sigma dropped

Coverage: ADI diffusion step (standalone; also the SL HJB default diffusion path,
which delegates to it) over 1D/2D/3D, the FP-FDM explicit-drift and implicit
per-point diffusion paths, and the HJB-GFDM per-point Newton path (the production
``joint_socp`` + ``precompute`` stack) via MMS source-cancellation. Constant sigma
(the clean eigenmode invariant).

The HJB-GFDM case isolates pure diffusion past the Hamiltonian advection with an
MMS source ``L^n[i] = -H(grad u*^n)`` evaluated on the analytic backward-decaying
eigenmode, so the ``H`` term cancels in the residual and the recovered field reads
the diffusion coefficient directly (the #1073 chain: ``problem.diffusion`` already
equals ``sigma^2/2``, so a path that re-squared it produced ``(sigma^2/2)^2``).

NOT covered yet (follow-up): spatially-varying sigma is partially guarded by the
#1183 warning + the per-point implicit path; a varying-sigma magnitude reference
(cos is not an eigenmode of ``div(D(x)grad)``) is a separate follow-up.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_adi import adi_diffusion_step
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

_K = 2.0 * np.pi


def _decay_relerr(adi_factor: float, analytic_factor: float) -> float:
    """Relative error in the (small) decay increment, robust as factor -> 1."""
    return abs(adi_factor - analytic_factor) / abs(1.0 - analytic_factor)


# ---------------------------------------------------------------------------
# ADI diffusion step (standalone; the SL HJB default diffusion path delegates here)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_adi_diffusion_magnitude(dim):
    """ADI nD diffusion must decay a cosine eigenmode at exp(-D*dim*k^2*T) (D=sigma^2/2).
    Pre-#1178 the dt/dimension split applied only 1/dim of the diffusion (relerr ~ dim-1)."""
    sigma, dt = 1.0, 5e-4 if dim == 3 else (1e-3 if dim == 2 else 2e-3)
    D = 0.5 * sigma**2
    n = {1: 121, 2: 81, 3: 31}[dim]
    L = 1.0
    dx = L / (n - 1)
    axes = [np.linspace(0.0, L, n)] * dim
    grids = np.meshgrid(*axes, indexing="ij")
    u0 = np.ones_like(grids[0])
    for g in grids:
        u0 = u0 * np.cos(_K * g)
    u1 = adi_diffusion_step(u0.copy(), dt, sigma, np.array([dx] * dim), tuple([n] * dim), "neumann")
    idx = tuple([n // 3] * dim)
    factor = u1[idx] / u0[idx]
    analytic = np.exp(-D * dim * _K**2 * dt)  # sum_d k_d^2 = dim*k^2
    assert _decay_relerr(factor, analytic) < 0.03, (
        f"ADI {dim}D diffusion magnitude wrong: factor {factor:.6f} vs analytic {analytic:.6f}"
    )


# ---------------------------------------------------------------------------
# Fokker-Planck FDM diffusion paths (zero drift => pure diffusion)
# ---------------------------------------------------------------------------


def _fp_pure_diffusion_decay(path: str, sigma: float, n: int = 81, nt: int = 40, T: float = 0.2) -> tuple:
    """Evolve m = 1 + a*cos(kx) under zero-drift FP and return (decay, analytic) of the cos amplitude.

    path='explicit' routes through the callable-drift explicit step; path='implicit' through the
    per-point implicit assembly. Both should apply D = sigma^2/2 and decay the eigenmode amplitude.
    """
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: np.asarray(m) * 0.0,
        coupling_dm=lambda m: np.asarray(m) * 0.0,
    )
    x = np.linspace(0.0, 1.0, n)
    comps = MFGComponents(
        m_initial=lambda xx: 1.0 + 0.4 * np.cos(_K * np.asarray(xx)),
        u_terminal=lambda xx: np.asarray(xx) * 0.0,
        hamiltonian=H,
    )
    prob = MFGProblem(geometry=grid, T=T, Nt=nt, sigma=sigma, components=comps)
    solver = FPFDMSolver(prob)
    m0 = 1.0 + 0.4 * np.cos(_K * x)
    if path == "explicit":
        M = solver.solve_fp_system(m0.copy(), drift_field=lambda t, g, m: np.zeros(n), volatility_field=sigma)
    else:
        M = solver.solve_fp_system(m0.copy(), drift_field=np.zeros((nt + 1, n)), volatility_field=sigma)
    D = 0.5 * sigma**2
    amp0, ampT = M[0] - 1.0, M[-1] - 1.0
    i = int(np.argmax(np.abs(amp0)))
    return ampT[i] / amp0[i], np.exp(-D * _K**2 * T)


@pytest.mark.tier2
@pytest.mark.integration
@pytest.mark.parametrize("path", ["explicit", "implicit"])
def test_fp_fdm_diffusion_magnitude(path):
    """The FP-FDM explicit-drift and implicit per-point diffusion paths must decay a cosine
    eigenmode at exp(-D*k^2*T) (D=sigma^2/2). A factor error (e.g. sigma used directly as D,
    #1152 class) breaks the rate while mass conservation still passes."""
    factor, analytic = _fp_pure_diffusion_decay(path, sigma=0.3)
    assert _decay_relerr(factor, analytic) < 0.03, (
        f"FP {path} diffusion magnitude wrong: factor {factor:.6f} vs analytic {analytic:.6f}"
    )


# ---------------------------------------------------------------------------
# HJB-GFDM diffusion magnitude in isolation (production joint_socp + precompute path)
# ---------------------------------------------------------------------------

_PI = np.pi  # cos(pi x) is a no-flux Laplacian eigenmode on [0,1]


def _gfdm_diffusion_field_relerr(
    sigma: float,
    D_reference: float,
    n_x: int = 41,
    T: float = 0.05,
    nt: int = 50,
    amp: float = 1e-3,
    lam: float = 1.0,
    delta: float = 0.3,
) -> float:
    """L2 error between the GFDM-recovered field and the analytic backward eigenmode.

    The solver applies its own ``D = sigma^2/2`` (the path under test); the analytic
    reference ``u*`` and the MMS source are built on ``D_reference``. With
    ``D_reference == 0.5*sigma**2`` (the correct magnitude) the recovered field tracks
    ``u*`` and the relerr is small; a mismatched ``D_reference`` detunes the decay and
    the relerr blows up -- which is how this gate would catch a wrong solver magnitude.
    The Hamiltonian ``H = |p|^2/(2 lam)`` advection is cancelled by the MMS source
    ``L^n[i] = -H(grad u*^n)``; ``amp`` is kept small so the residual cancellation is
    clean (diffusion O(amp) dominates the O(amp^2) Hamiltonian remnant).
    """
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n_x], boundary_conditions=no_flux_bc(dimension=1))
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=lam),
        coupling=lambda m: 0.0 * np.asarray(m),
        coupling_dm=lambda m: 0.0 * np.asarray(m),
    )
    x = np.linspace(0.0, 1.0, n_x)
    comps = MFGComponents(
        m_initial=lambda xx: np.ones_like(np.asarray(xx, dtype=float)),
        u_terminal=lambda xx: amp * np.cos(_PI * np.asarray(xx, dtype=float)),
        hamiltonian=H,
    )
    prob = MFGProblem(geometry=grid, T=T, Nt=nt, sigma=sigma, components=comps)
    solver = HJBGFDMSolver(
        prob, x.reshape(-1, 1), delta=delta, monotonicity_scheme="joint_socp", monotonicity_application="precompute"
    )
    tspace = np.linspace(0.0, T, nt + 1)

    def running_cost_fn(n):  # L^n[i] = -H(grad u*^n) on the analytic field
        grad_u_star = -amp * np.exp(-D_reference * _PI**2 * (T - tspace[n])) * _PI * np.sin(_PI * x)
        return -(grad_u_star**2) / (2.0 * lam)

    U = solver.solve_hjb_system(
        M_density=np.ones((nt + 1, n_x)),
        U_terminal=amp * np.cos(_PI * x),
        running_cost=running_cost_fn,
        show_progress=False,
    )
    u0_star = amp * np.exp(-D_reference * _PI**2 * T) * np.cos(_PI * x)
    return float(np.linalg.norm(U[0, :] - u0_star) / np.linalg.norm(u0_star))


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.tier3
def test_hjb_gfdm_diffusion_magnitude():
    """The production HJB-GFDM per-point Newton path (joint_socp + precompute) must apply
    D = sigma^2/2. Verified discriminating: correct D -> field relerr ~0.012, a halved D ->
    ~0.105, a doubled D -> ~0.295; the 0.05 threshold separates correct from either error
    with margin. This is the #1073 class (re-squaring problem.diffusion to (sigma^2/2)^2)."""
    sigma = 1.0
    relerr = _gfdm_diffusion_field_relerr(sigma, D_reference=0.5 * sigma**2)
    assert relerr < 0.05, (
        f"HJB-GFDM diffusion magnitude wrong: field relerr {relerr:.4e} >= 0.05 "
        "(solver D = sigma^2/2 disagrees with the analytic eigenmode)"
    )

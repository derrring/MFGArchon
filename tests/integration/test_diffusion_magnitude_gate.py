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
which delegates to it) over 1D/2D/3D, and the FP-FDM explicit-drift and implicit
per-point diffusion paths. Constant sigma (the clean eigenmode invariant).

NOT covered yet (follow-up): the HJB FDM/GFDM diffusion magnitude in isolation —
isolating pure diffusion past the Hamiltonian advection needs an MMS
source-cancellation harness; the HJB-side bugs this session were BC/coupling, not
diffusion-magnitude. Spatially-varying sigma is partially guarded by the #1183
warning + the per-point implicit path; a varying-sigma magnitude reference (cos is
not an eigenmode of ``div(D(x)grad)``) is a separate follow-up.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
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

"""Issue #1543: nD semi-Lagrangian steps under-applied diffusion by a factor d.

Both SL diffusion methods ('stochastic' and 'canonical_cs') place 2d Brownian feet x +- c*e_ax
and average them with weight 1/(2d), recovering (c^2/2d)*Lap(u). With the shipped offset
c = sigma*sqrt(dt) that is (sigma^2/2d)*Lap(u)*dt -- a factor-1/d deficit vs the canonical
(sigma^2/2)*Lap(u)*dt (2x under-diffusion in 2D, 3x in 3D; 1D exact, which is why the bug hid).
The fix single-sources the offset in HJBSemiLagrangianSolver._brownian_foot_offset to
c = sqrt(d)*sigma*sqrt(dt) (weak-Euler direction tree, E[xi xi^T] = I dt).

Pinned invariants:
  1. The owner returns sqrt(d)*sigma*sqrt(dt) (exact identity at d=1). Direct pin of the sqrt(d).
  2. The owner is WIRED into both real SL paths: driving one SL step (large control cost + zero
     potential => the step collapses to the pure diffusion average u^n = u_avg) recovers the FULL
     (sigma^2/2)*Lap(u), not 1/d of it, for d=2 and d=3.
  3. Convention agreement SL-vs-ADI: the SL diffusion increment matches the ADI sibling's (ADI
     independently applies the full D per axis) at interior points.

Discriminating: reverting the sqrt(d) in _brownian_foot_offset (the fix) makes every assertion here
fail (owner returns sigma*sqrt(dt); the recovered ratios collapse to 1/d).
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_adi import adi_diffusion_step
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

L = 2.0 * np.pi  # u = sum_k cos(x_k) on [0, L]^d has zero normal derivative at 0 and L
SIGMA = 0.6
# Fixed code recovers ~1.0 (2D ~1.00, 3D ~1.08, canonical_cs ~1.07); buggy code recovers 1/d
# (0.5, 0.33). The band passes the former with O(c^4)/interp headroom and rejects the latter.
_LO, _HI = 0.7, 1.3


def _diffusion_problem(d: int, N: int, Nt: int, sigma: float = SIGMA) -> MFGProblem:
    """Pure-diffusion setup: huge control cost => drift alpha* = -grad(u)/lambda -> 0 and the kinetic
    value term dt*|grad u|^2/(2 lambda) -> 0; zero potential/coupling => H(x,0,m) = 0. The Lax-Oleinik
    update u^n = u_avg + dt*(H(p) - 2 H(0)) then collapses to u^n = u_avg (the diffusion average)."""
    grid = TensorProductGrid(
        dimension=d, bounds=[(0.0, L)] * d, Nx_points=[N] * d, boundary_conditions=no_flux_bc(dimension=d)
    )
    return MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1e8)), sigma=sigma),
        domain=grid,
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=1.0),
        Nt=Nt,
    )


def _analytic_field(d: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    coords = [np.linspace(0.0, L, N) for _ in range(d)]
    mesh = np.meshgrid(*coords, indexing="ij")
    u = sum(np.cos(m) for m in mesh)
    return u, -u  # Lap(sum cos x_k) = -sum cos x_k


def _interior_median(ratio: np.ndarray, d: int) -> float:
    sl = tuple(slice(4, -4) for _ in range(d))
    r = ratio[sl]
    return float(np.median(r[np.isfinite(r)]))


def _sl_effective_ratio(d: int, N: int, Nt: int, method: str) -> float:
    """Median interior ratio (SL diffusion increment) / (intended (sigma^2/2)*Lap(u)*dt)."""
    problem = _diffusion_problem(d, N, Nt)
    dt = 1.0 / Nt
    u_next, lap = _analytic_field(d, N)
    m = np.ones([N] * d)
    solver = HJBSemiLagrangianSolver(problem, interpolation_method="linear", diffusion_method=method)
    step = solver._solve_timestep_stochastic_sl if method == "stochastic" else solver._solve_timestep_canonical_cs
    u_n = step(u_next, m, Nt - 1, dt)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _interior_median((u_n - u_next) / (dt * (SIGMA**2 / 2.0) * lap), d)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_brownian_foot_offset_is_sqrt_d_scaled(d):
    """Owner returns sqrt(d)*sigma*sqrt(dt) (invariant 1). Mutation: dropping sqrt(d) fails d=2,3."""
    solver = HJBSemiLagrangianSolver(_diffusion_problem(d, 5, 4), diffusion_method="stochastic")
    sqrt_dt = 0.1
    offset = solver._brownian_foot_offset(sqrt_dt)
    expected = np.full(d, np.sqrt(d) * SIGMA * sqrt_dt)
    assert np.allclose(offset, expected), f"d={d}: expected {expected}, got {offset}"


@pytest.mark.parametrize(
    ("d", "N", "Nt", "method"),
    [
        (2, 41, 6, "stochastic"),
        (3, 21, 8, "stochastic"),
        (2, 15, 2, "canonical_cs"),
    ],
)
def test_sl_recovers_full_diffusion(d, N, Nt, method):
    """The owner is wired into both real SL paths: one step recovers the FULL (sigma^2/2)*Lap(u), not
    1/d of it (invariant 2). Mutation: reverting the sqrt(d) collapses the ratio to 1/d (< _LO)."""
    ratio = _sl_effective_ratio(d, N, Nt, method)
    assert _LO <= ratio <= _HI, (
        f"{method} d={d}: effective-diffusion ratio {ratio:.3f} not in [{_LO}, {_HI}] (1/d={1 / d:.3f} = the bug)"
    )


def test_sl_diffusion_agrees_with_adi_2d():
    """Convention agreement SL-vs-ADI (invariant 3): the SL diffusion increment matches the ADI
    sibling's on the same field. ADI independently applies the full D per axis, so agreement confirms
    the SL paths now share that convention. Mutation: buggy SL diffuses half as much -> ratio ~0.5."""
    d, N, Nt = 2, 41, 6
    dt = 1.0 / Nt
    u_next, _ = _analytic_field(d, N)
    dx = L / (N - 1)
    adi = adi_diffusion_step(u_next, dt, SIGMA, np.full(d, dx), (N,) * d, "neumann")
    problem = _diffusion_problem(d, N, Nt)
    solver = HJBSemiLagrangianSolver(problem, interpolation_method="linear", diffusion_method="stochastic")
    sl = solver._solve_timestep_stochastic_sl(u_next, np.ones([N] * d), Nt - 1, dt)
    with np.errstate(divide="ignore", invalid="ignore"):
        sl_over_adi = _interior_median((sl - u_next) / (adi - u_next), d)
    assert 0.8 <= sl_over_adi <= 1.3, f"SL/ADI diffusion increment {sl_over_adi:.3f} not ~1 (buggy SL ~0.5)"

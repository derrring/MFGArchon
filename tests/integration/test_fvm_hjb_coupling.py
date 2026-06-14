"""Coupled FVM-FP + HJB-FDM fixed-point integration tests (Issue #422).

The conservative finite-volume FP solver (``FPFVMSolver``) ships with standalone unit
tests (``tests/unit/test_fp_fvm.py``) that exercise the FP operator with a *prescribed*
drift/potential field. Its intended use, however, is inside the MFG fixed-point loop, where
the velocity ``alpha = -coupling * grad(U)`` is produced by an HJB solve each Picard
iteration. That coupled path had no CI coverage. These tests lock it in:

1. **1D LQ coupled convergence + conservation** -- ``problem.solve(scheme=FVM_MUSCL)`` (the
   Safe-Mode pairing FVM-FP + upwind HJB-FDM) reaches the tolerance in a sane iteration
   count, conserves the finite-volume cell-sum mass to machine precision at every step, and
   keeps the density non-negative (MUSCL minmod limiter).
2. **FVM-vs-FDM agreement** -- the same 1D LQ problem solved with ``FVM_MUSCL`` and with
   ``FDM_UPWIND`` agree to a few percent in ``L2`` (they discretize the same PDE; the
   difference is the spatial/temporal truncation, ``O(dx)`` between upwind-FDM and
   MUSCL-FVM, plus the Strang-split-vs-implicit time stepping).
3. **2D LQ coupled** -- ``FVM_MUSCL`` + fixed-point HJB-FDM via ``FixedPointIterator`` runs,
   conserves cell-sum mass to machine precision, and stays finite/non-negative. (The 2D LQ
   coupled fixed point is stiff and is not required to converge here -- as in
   ``test_coupled_hjb_fp_2d.py`` -- so only mass/finiteness are asserted.)

Mass convention: ``FPFVMSolver`` conserves the rectangular cell-sum ``sum(M) * prod(dx)``
(the flux telescopes over cells), NOT the trapezoidal integral (which half-weights the wall
cells and therefore drifts as mass moves toward a boundary). All mass checks below use the
rectangular cell-sum, matching the solver's invariant and the unit tests.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.coupling import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers import FPFVMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import dirichlet_bc, no_flux_bc
from mfgarchon.types import NumericalScheme


# ---------------------------------------------------------------------------
# Problem builders (LQ MFG: H = |p|^2/2 + m, quadratic terminal well)
# ---------------------------------------------------------------------------
def _lq_hamiltonian():
    """Separable LQ Hamiltonian H = |p|^2/2 + m (smooth control cost, linear coupling)."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


# 1D LQ regime. Weak coupling + moderate diffusion keep the coupled velocity small so the
# Picard loop converges in <20 iterations; the cost is dominated by the per-iteration 1D HJB
# Newton solve, so N/Nt are kept modest to bound the wall-clock.
_N_1D = 25
_T_1D = 0.3
_NT_1D = 12
_SIGMA_1D = 0.4
_COUPLING_1D = 0.3
_X0_1D, _S0_1D = 0.4, 0.13  # Gaussian IC center / width
_XT_1D, _KT_1D = 0.6, 0.2  # terminal cost 0.2*(x - 0.6)^2


def _build_problem_1d():
    components = MFGComponents(
        m_initial=lambda x: np.exp(-((np.asarray(x) - _X0_1D) ** 2) / (2 * _S0_1D**2)),
        u_terminal=lambda x: _KT_1D * (np.asarray(x) - _XT_1D) ** 2,
        hamiltonian=_lq_hamiltonian(),
    )
    geom = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[_N_1D],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    return MFGProblem(
        geometry=geom,
        T=_T_1D,
        Nt=_NT_1D,
        sigma=_SIGMA_1D,
        coupling_coefficient=_COUPLING_1D,
        components=components,
    )


def _rel_l2(a: np.ndarray, b: np.ndarray, dx: float) -> float:
    """Relative discrete L2 norm ||a - b|| / ||b|| with cell measure dx."""
    return float(np.sqrt(dx * np.sum((a - b) ** 2)) / np.sqrt(dx * np.sum(b**2)))


# Solved once per module (each Picard solve takes several seconds): the convergence,
# conservation, and FVM-vs-FDM agreement tests all read from these.
@pytest.fixture(scope="module")
def fvm_1d():
    prob = _build_problem_1d()
    dx = prob.geometry.get_grid_spacing()[0]
    result = prob.solve(scheme=NumericalScheme.FVM_MUSCL, max_iterations=40, tolerance=1e-4)
    return result, dx


@pytest.fixture(scope="module")
def fdm_1d():
    prob = _build_problem_1d()
    dx = prob.geometry.get_grid_spacing()[0]
    result = prob.solve(scheme=NumericalScheme.FDM_UPWIND, max_iterations=40, tolerance=1e-4)
    return result, dx


# ===========================================================================
# 1D LQ coupled: convergence, mass conservation, positivity
# ===========================================================================
@pytest.mark.slow
@pytest.mark.integration
def test_1d_fvm_coupled_converges(fvm_1d):
    """FVM_MUSCL + HJB-FDM Picard loop reaches tol in a sane iteration count, error decreasing."""
    result, _dx = fvm_1d

    eh = np.asarray(result.error_history_M, dtype=float)
    assert eh.size >= 2, "error history too short to assess convergence"
    # Error decreases substantially over the run (transient hump allowed; min must collapse).
    assert eh.min() < 0.1 * eh[0], f"residual M did not decrease: {eh[0]:.3e} -> min {eh.min():.3e}"
    # Reaches the requested tolerance (1e-4) within a sane iteration budget.
    assert result.final_error_M <= 1e-4, f"did not reach tol: final_error_M={result.final_error_M:.3e}"
    assert result.iterations <= 30, f"too many iterations to converge: {result.iterations}"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.mathematical
def test_1d_fvm_coupled_mass_conserved_and_positive(fvm_1d):
    """Cell-sum mass is conserved to ~machine precision at every step; density stays non-negative."""
    result, dx = fvm_1d
    M = result.M

    assert np.all(np.isfinite(M)), "density contains non-finite values"
    # FVM conserves the rectangular cell-sum (flux telescoping), not the trapezoidal integral.
    mass = M.sum(axis=1) * dx
    mass_drift = float(np.max(np.abs(mass - mass[0])))
    assert mass_drift < 1e-10, f"cell-sum mass drift {mass_drift:.3e} (expected ~1e-15)"
    # MUSCL minmod limiter -> no negative-density ringing.
    assert M.min() >= -1e-12, f"density went negative: {M.min():.3e}"


# ===========================================================================
# 1D LQ coupled: FVM_MUSCL vs FDM_UPWIND agreement (same PDE)
# ===========================================================================
@pytest.mark.slow
@pytest.mark.integration
def test_1d_fvm_fdm_agreement(fvm_1d, fdm_1d):
    """FVM_MUSCL and FDM_UPWIND solve the same LQ MFG; converged U and M agree to a few percent."""
    res_fvm, dx = fvm_1d
    res_fdm, dx_fdm = fdm_1d
    assert dx == dx_fdm

    rel_m = _rel_l2(res_fvm.M, res_fdm.M, dx)
    rel_u = _rel_l2(res_fvm.U, res_fdm.U, dx)
    rel_m_terminal = _rel_l2(res_fvm.M[-1], res_fdm.M[-1], dx)

    # They discretize the same PDE: difference is O(dx) (upwind-FDM vs MUSCL-FVM) plus the
    # Strang-split-vs-implicit time stepping. A few percent at this resolution.
    assert rel_m < 0.07, f"FVM-vs-FDM full-field M disagreement {rel_m:.3%} (>7%)"
    assert rel_u < 0.07, f"FVM-vs-FDM full-field U disagreement {rel_u:.3%} (>7%)"
    assert rel_m_terminal < 0.07, f"FVM-vs-FDM terminal M disagreement {rel_m_terminal:.3%} (>7%)"


# ===========================================================================
# 2D LQ coupled: FVM_MUSCL runs, conserves mass, stays finite/non-negative
# ===========================================================================
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.mathematical
def test_2d_fvm_coupled_mass_conserved_finite():
    """2D FVM_MUSCL + fixed-point HJB-FDM: cell-sum mass conserved, density finite/non-negative.

    The 2D LQ coupled fixed point is stiff (Newton Jacobians degrade as the density evolves --
    see test_coupled_hjb_fp_2d.py), so convergence is not required here; only the structural
    invariants the FVM scheme guarantees per FP solve (exact cell-sum mass, finiteness,
    positivity) are asserted. A low Picard-iteration cap keeps the run bounded and well inside
    the finite regime.
    """
    n = 10  # 11 x 11 grid on [-1, 1]^2
    x = np.linspace(-1.0, 1.0, n + 1)
    X, Y = np.meshgrid(x, x, indexing="ij")
    m0 = np.exp(-5.0 * (X**2 + Y**2))
    components = MFGComponents(
        m_initial=m0,
        u_terminal=lambda xy: 0.5 * np.sum(np.asarray(xy) ** 2),
        hamiltonian=_lq_hamiltonian(),
    )
    geom = TensorProductGrid(
        bounds=[(-1.0, 1.0), (-1.0, 1.0)],
        Nx_points=[n + 1, n + 1],
        boundary_conditions=no_flux_bc(dimension=2),
    )
    prob = MFGProblem(geometry=geom, T=0.2, Nt=10, sigma=0.1, components=components)
    dx, dy = prob.geometry.get_grid_spacing()

    hjb = HJBFDMSolver(prob, solver_type="fixed_point", damping_factor=0.8, max_newton_iterations=50)
    fp = FPFVMSolver(prob, reconstruction="muscl")
    iterator = FixedPointIterator(prob, hjb_solver=hjb, fp_solver=fp, relaxation=0.3)

    result = iterator.solve(max_iterations=5, tolerance=1e-4, show_progress=False)
    M = result.M

    grid_shape = prob.geometry.get_grid_shape()
    assert M.shape == (prob.Nt + 1, *grid_shape), f"unexpected M shape {M.shape}"
    assert np.all(np.isfinite(M)), "2D density contains non-finite values"
    mass = M.sum(axis=(1, 2)) * dx * dy
    mass_drift = float(np.max(np.abs(mass - mass[0])))
    assert mass_drift < 1e-10, f"2D cell-sum mass drift {mass_drift:.3e} (expected ~1e-15)"
    assert M.min() >= -1e-12, f"2D density went negative: {M.min():.3e}"


# ===========================================================================
# Construction-time fail-loud: Dirichlet BC is out of FP FVM v1 scope (Issue #422)
# ===========================================================================
def test_fvm_dirichlet_bc_raises_at_construction():
    """FPFVMSolver must reject Dirichlet BC at construction, not silently at solve-time.

    The advective flux closure has no Dirichlet inflow handling (deferred, Issue #422); without
    the __init__ guard the failure would only surface from fp_fvm_flux.axis_flux_divergence once
    an advected solve runs.
    """
    components = MFGComponents(
        m_initial=lambda x: np.ones_like(np.asarray(x, dtype=float)),
        u_terminal=lambda x: 0.0 * np.asarray(x, dtype=float),
        hamiltonian=_lq_hamiltonian(),
    )
    geom = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[21],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    prob = MFGProblem(geometry=geom, T=0.1, Nt=10, sigma=0.2, components=components)

    with pytest.raises(NotImplementedError, match="Dirichlet"):
        FPFVMSolver(prob, boundary_conditions=dirichlet_bc(dimension=1))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

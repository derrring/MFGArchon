"""Pinning tests for Issue #1412: scalar-only FP solvers fail loud on a spatial override.

Context — the σ-value single-source (Issue #1412, generalizing #1071). The shared
``mfgarchon.utils.pde_coefficients.resolve_diffusion_source`` resolves a scalar / array /
callable volatility (``sigma``) source to ONE scalar; its batch path (``index=None``)
collapses an array to its **mean**. That mean-collapse is correct for a solver that genuinely
applies a single global ``sigma`` — but it is exactly the wrong thing for the *coupling*:
if a scalar-only FP solver silently mean-collapsed a spatially-varying ``volatility_field``
while the HJB solver consumed the full ``sigma(x)`` (HJBGFDMSolver does, via #1316), HJB and
FP would solve different diffusions and the Picard fixed point would correspond to neither —
the silent cross-path divergence #1412 / #1316 exist to kill.

The defense is that every scalar-only FP solver **fails loud** (``NotImplementedError``) on an
array/callable ``volatility_field`` instead of collapsing it — the same disposition
``HJBSemiLagrangianSolver`` / ``HJBWENOSolver`` adopt on the HJB side
(``test_issue_1316_hjb_volatility_consumption``). FPFVMSolver's guard is already pinned
(``test_fp_fvm`` — ``_scalar_diffusion`` raises on a non-constant array); the GFDM, backward-SL
and forward-SL (adjoint) FP guards were **not** pinned anywhere. This module pins them so a
future "single-source it through ``resolve_diffusion_source``" refactor that drops a guard and
reintroduces the silent mean-collapse **fails CI**.

A matched scalar ``volatility_field == problem.sigma`` (the iterator's redundant forwarding,
Issue #1248) must stay a no-op and NOT raise.

Refs #1412 (parent #1071); guard pattern from #1316.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_gfdm import FPGFDMSolver
from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian import FPSLJacobianSolver
from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

# Legacy MFGProblem API + FPSLJacobianSolver deprecation warnings are incidental to these tests.
pytestmark = pytest.mark.filterwarnings("ignore")

N = 21


def _problem(sigma: float = 0.3) -> MFGProblem:
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[N], boundary_conditions=no_flux_bc(dimension=1))
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.0,
        coupling_dm=lambda m: 0.0,
    )
    comp = MFGComponents(
        hamiltonian=H,
        m_initial=lambda x: np.ones_like(x),
        u_terminal=lambda x: x * 0,
    )
    return MFGProblem(geometry=grid, components=comp, T=0.2, Nt=10, sigma=sigma)


def _m_initial() -> np.ndarray:
    return np.ones(N) / N


def _potential_field(nt_points: int = 11) -> np.ndarray:
    """A non-callable potential U(t, x) so the SL solvers pass their potential-callable guard
    and reach the volatility resolution (the site under test)."""
    return np.zeros((nt_points, N))


def _spatial_volatility() -> np.ndarray:
    """A genuinely spatially-varying sigma(x) — the override a scalar-only solver must refuse,
    not silently mean-collapse."""
    return np.linspace(0.2, 0.8, N)


def _gfdm_solver(problem: MFGProblem) -> FPGFDMSolver:
    points = np.linspace(0.0, 1.0, N).reshape(-1, 1)
    return FPGFDMSolver(problem, collocation_points=points, delta=0.25)


# ---------------------------------------------------------------------------
# Fail-loud on a spatially-varying volatility_field (no silent mean-collapse)
# ---------------------------------------------------------------------------


def test_fp_sl_jacobian_fails_loud_on_spatial_volatility():
    """Backward-SL FP (FPSLJacobianSolver) must refuse an array volatility_field rather than
    silently using a single scalar — HJB would see sigma(x), FP a collapsed scalar."""
    solver = FPSLJacobianSolver(_problem())
    with pytest.raises(NotImplementedError, match="volatility"):
        solver.solve_fp_system(
            M_initial=_m_initial(), potential_field=_potential_field(), volatility_field=_spatial_volatility()
        )


def test_fp_sl_adjoint_fails_loud_on_spatial_volatility():
    """Forward-SL (adjoint) FP (FPSLSolver) must refuse an array volatility_field."""
    solver = FPSLSolver(_problem())
    with pytest.raises(NotImplementedError, match="volatility"):
        solver.solve_fp_system(
            M_initial=_m_initial(), potential_field=_potential_field(), volatility_field=_spatial_volatility()
        )


def test_fp_gfdm_fails_loud_on_spatial_volatility():
    """GFDM FP must refuse an array volatility_field (its forward-Euler loop applies one scalar
    diffusion_coeff; a per-point sigma(x) would need a variable-coefficient operator)."""
    solver = _gfdm_solver(_problem())
    with pytest.raises(NotImplementedError, match="volatility"):
        solver.solve_fp_system(_m_initial(), drift_field=None, volatility_field=_spatial_volatility())


# ---------------------------------------------------------------------------
# A matched scalar override stays a no-op (must NOT raise)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver_cls", [FPSLJacobianSolver, FPSLSolver])
def test_scalar_override_equal_to_sigma_does_not_raise(solver_cls):
    """The iterator's redundant forwarding of a scalar problem.sigma as volatility_field
    (Issue #1248) is a no-op for the scalar-only SL solvers and must not fail loud."""
    solver = solver_cls(_problem(sigma=0.3))
    result = solver.solve_fp_system(M_initial=_m_initial(), potential_field=_potential_field(), volatility_field=0.3)
    assert np.asarray(result).shape[0] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

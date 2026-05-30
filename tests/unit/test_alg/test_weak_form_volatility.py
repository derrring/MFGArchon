"""The weak-form FP/HJB solvers must treat `volatility_field` as the SDE volatility
`sigma`, converting to the PDE diffusion `D = sigma^2 / 2` (Conventions Index; Issue #811).

Previously the scalar/array branches of `_diffusion_coefficient` (FP) and the HJB D
computation returned the input directly as D, skipping the conversion that the `None`
branch, `solve_fp_step_adjoint_mode`, and every other solver (FDM, particle, GFDM) apply
-- so a passed volatility was used with the wrong (squared-too-large) diffusion.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver
from mfgarchon.alg.numerical.meshless_galerkin.hjb_solver import MeshlessGalerkinHJBSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _problem(sigma, n=15):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    comp = MFGComponents(
        hamiltonian=H, m_initial=lambda x: np.exp(-20 * (x - 0.5) ** 2), u_terminal=lambda x: 0.5 * (x - 0.5) ** 2
    )
    return MFGProblem(geometry=grid, components=comp, T=0.5, Nt=10, sigma=sigma, coupling_coefficient=0.5)


def _cloud(n=15):
    return np.linspace(0.0, 1.0, n)[:, None]


def test_diffusion_coefficient_converts_sigma_to_D():
    """volatility_field (scalar or array) is sigma -> D = sigma^2 / 2."""
    fp = MeshlessGalerkinFPSolver(_problem(sigma=0.5), collocation_points=_cloud(), delta=3.5 / 14)
    assert fp._diffusion_coefficient(None) == pytest.approx(0.5 * 0.5**2)  # uses problem.sigma
    assert fp._diffusion_coefficient(0.3) == pytest.approx(0.5 * 0.3**2)  # NOT 0.3
    assert fp._diffusion_coefficient(np.full(fp.n_dof, 0.3)) == pytest.approx(0.5 * 0.3**2)


def test_fp_volatility_override_equals_problem_with_that_sigma():
    """An FP solve with volatility_field=s must match a problem built with sigma=s
    (the equivalence the sigma->D convention guarantees)."""
    x = np.linspace(0.0, 1.0, 15)
    m0 = np.exp(-20 * (x - 0.5) ** 2)
    drift = np.tile(0.5 * (x - 0.5) ** 2, (11, 1))

    fp_override = MeshlessGalerkinFPSolver(_problem(sigma=0.9), collocation_points=_cloud(), delta=3.5 / 14)
    fp_native = MeshlessGalerkinFPSolver(_problem(sigma=0.3), collocation_points=_cloud(), delta=3.5 / 14)
    traj_override = fp_override.solve_fp_system(m0, drift_field=drift, volatility_field=0.3)
    traj_native = fp_native.solve_fp_system(m0, drift_field=drift, volatility_field=None)
    assert np.allclose(traj_override, traj_native, atol=1e-12)


def test_hjb_volatility_override_equals_problem_with_that_sigma():
    """An HJB solve with volatility_field=s must match a problem built with sigma=s."""
    x = np.linspace(0.0, 1.0, 15)
    m = np.ones((11, 15)) / 15
    u_T = 0.5 * (x - 0.5) ** 2

    hjb_override = MeshlessGalerkinHJBSolver(_problem(sigma=0.9), collocation_points=_cloud(), delta=3.5 / 14)
    hjb_native = MeshlessGalerkinHJBSolver(_problem(sigma=0.3), collocation_points=_cloud(), delta=3.5 / 14)
    U_override = hjb_override.solve_hjb_system(M_density=m, U_terminal=u_T, volatility_field=0.3)
    U_native = hjb_native.solve_hjb_system(M_density=m, U_terminal=u_T, volatility_field=None)
    assert np.allclose(U_override, U_native, atol=1e-12)

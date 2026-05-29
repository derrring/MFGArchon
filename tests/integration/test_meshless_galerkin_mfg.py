"""
Integration tests for the meshless Galerkin (MLS) solver pair (Issue #1131).

Validates the meshfree weak-form path end-to-end: scheme registration + duality,
the adjoint-consistent mass-conserving FP advection (FP = HJB^T), and a finite
HJB backward solve. 1D Neumann LQ-MFG.
"""

from __future__ import annotations

import numpy as np
import pytest

from mfgarchon import MFGProblem
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents
from mfgarchon.factory.scheme_factory import create_paired_solvers
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.types.schemes import NumericalScheme


def _problem(n=21, T=0.5, nt=20, sigma=0.3):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: -(m**2),
        coupling_dm=lambda m: -2 * m,
    )
    comp = MFGComponents(
        hamiltonian=H,
        m_initial=lambda x: np.exp(-40 * (x - 0.35) ** 2),
        u_terminal=lambda x: 0.5 * (x - 0.5) ** 2,
    )
    return MFGProblem(geometry=grid, components=comp, T=T, Nt=nt, sigma=sigma, coupling_coefficient=0.5)


def _pair(problem, n=21):
    cloud = np.linspace(0.0, 1.0, n)[:, None]
    delta = 3.5 / (n - 1)
    return create_paired_solvers(
        problem,
        NumericalScheme.MESHLESS_GALERKIN,
        hjb_config={"collocation_points": cloud, "delta": delta},
    )


@pytest.mark.integration
class TestMeshlessGalerkinMFG:
    def test_scheme_is_discrete_dual(self):
        assert NumericalScheme.MESHLESS_GALERKIN.is_discrete_dual()
        assert not NumericalScheme.MESHLESS_GALERKIN.requires_renormalization()

    def test_pair_creation_and_duality(self):
        # create_paired_solvers raises if the pair is not a validated dual.
        hjb, fp = _pair(_problem())
        assert type(hjb).__name__ == "MeshlessGalerkinHJBSolver"
        assert type(fp).__name__ == "MeshlessGalerkinFPSolver"

    def test_fp_pure_diffusion_conserves_mass(self):
        problem = _problem()
        _, fp = _pair(problem)
        M_mat = fp._M
        x = fp._disc.dof_coordinates[:, 0]
        m0 = np.exp(-40 * (x - 0.5) ** 2)
        m0 /= float((M_mat @ m0).sum())
        traj = fp.solve_fp_system(m0, drift_field=None)  # pure diffusion
        mass = np.array([float((M_mat @ traj[n]).sum()) for n in range(problem.Nt + 1)])
        # No advection, no clipping needed: mass conserved structurally.
        assert np.max(np.abs(mass - 1.0)) < 1e-8

    def test_fp_with_drift_conserves_mass_no_blowup(self):
        problem = _problem()
        _, fp = _pair(problem)
        M_mat = fp._M
        x = fp._disc.dof_coordinates[:, 0]
        m0 = np.exp(-40 * (x - 0.35) ** 2)
        m0 /= float((M_mat @ m0).sum())
        U = np.tile(0.5 * (x - 0.5) ** 2, (problem.Nt + 1, 1))
        traj = fp.solve_fp_system(m0, drift_field=U)
        mass = np.array([float((M_mat @ traj[n]).sum()) for n in range(problem.Nt + 1)])
        assert np.all(np.isfinite(traj)), "FP blew up"
        # Adjoint-consistent transpose advection conserves mass; residual is the
        # positivity clip (Galerkin is not an M-matrix), bounded well below O(1).
        assert np.max(np.abs(mass - 1.0)) < 1e-2

    def test_hjb_solve_finite(self):
        problem = _problem()
        hjb, _ = _pair(problem)
        x = hjb._disc.dof_coordinates[:, 0]
        m = np.ones((problem.Nt + 1, hjb.n_dof)) / hjb.n_dof
        U = hjb.solve_hjb_system(M_density=m, U_terminal=0.5 * (x - 0.5) ** 2)
        assert U.shape == (problem.Nt + 1, hjb.n_dof)
        assert np.all(np.isfinite(U))

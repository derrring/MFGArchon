"""
Integration tests for the meshless Galerkin (MLS) solver pair (Issue #1131).

Validates the meshfree weak-form path end-to-end: scheme registration + duality,
the adjoint-consistent mass-conserving FP advection (FP = HJB^T), and a finite
HJB backward solve. 1D Neumann LQ-MFG.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents
from mfgarchon.factory.scheme_factory import create_paired_solvers
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BoundaryConditions, no_flux_bc
from mfgarchon.geometry.boundary.types import BCSegment, BCType
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


def _dirichlet_problem(n=31, T=0.5, nt=20, sigma=0.3):
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="x_min", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_min"),
            BCSegment(name="x_max", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max"),
        ],
        dimension=1,
    )
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=bc)
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: -(m**2),
        coupling_dm=lambda m: -2 * m,
    )
    comp = MFGComponents(
        hamiltonian=H,
        m_initial=lambda x: np.exp(-40 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.5 * (x - 0.5) ** 2,
    )
    return MFGProblem(geometry=grid, components=comp, T=T, Nt=nt, sigma=sigma, coupling_coefficient=0.5)


@pytest.mark.integration
class TestMeshlessGalerkinNitsche:
    """Dirichlet/absorbing BC imposed weakly via symmetric Nitsche (#1138)."""

    _NB = (np.array([[0.0], [1.0]]), np.array([[-1.0], [1.0]]))  # boundary points + outward normals

    def test_hjb_dirichlet_weak_trace_picard(self):
        n = 31
        hjb, _ = _pair(_dirichlet_problem(n=n), n=n)
        x = hjb._disc.dof_coordinates[:, 0]
        m = np.ones((hjb.problem.Nt + 1, hjb.n_dof)) / hjb.n_dof
        U = hjb.solve_hjb_system(M_density=m, U_terminal=0.5 * (x - 0.5) ** 2, use_newton=False)
        assert np.all(np.isfinite(U))
        phi_b, _gn = hjb._disc.boundary_shape_data(*self._NB)
        trace = phi_b @ U[0]  # weak boundary trace u_h(0), u_h(1)
        assert np.max(np.abs(trace)) < 2e-2, f"Nitsche should drive u_h->g=0 on the boundary, got {trace}"

        hjb_nf, _ = _pair(_problem(n=n), n=n)  # no_flux: boundary trace is unconstrained
        U_nf = hjb_nf.solve_hjb_system(M_density=m, U_terminal=0.5 * (x - 0.5) ** 2, use_newton=False)
        assert np.max(np.abs(trace)) < np.max(np.abs(phi_b @ U_nf[0]))

    def test_fp_absorbing_loses_mass(self):
        n = 31
        problem = _dirichlet_problem(n=n)
        _, fp = _pair(problem, n=n)
        M_mat = fp._M
        x = fp._disc.dof_coordinates[:, 0]
        m0 = np.exp(-40 * (x - 0.5) ** 2)
        m0 /= float((M_mat @ m0).sum())
        traj = fp.solve_fp_system(m0, drift_field=None)
        mass = np.array([float((M_mat @ traj[k]).sum()) for k in range(problem.Nt + 1)])
        assert np.all(np.isfinite(traj))
        assert traj.min() >= -1e-12
        assert mass[-1] < 0.99 * mass[0], "mass must leave through the absorbing boundary"
        assert mass[-1] <= mass[0] + 1e-9, "absorbing mass must not be created"
        phi_b, _gn = fp._disc.boundary_shape_data(*self._NB)
        assert np.max(np.abs(phi_b @ traj[-1])) < 0.05, "m_h should be driven toward 0 on Gamma_D"

        _, fp_nf = _pair(_problem(n=n), n=n)  # no_flux conserves mass
        traj_nf = fp_nf.solve_fp_system(m0, drift_field=None)
        assert abs(float((fp_nf._M @ traj_nf[-1]).sum()) - 1.0) < 1e-2

    def test_newton_nitsche_jacobian_consistent(self):
        """Analytic Jacobian (incl. Nitsche block) matches the FD Jacobian of the residual.

        Proves the Newton integration is correct independent of global convergence
        (undamped Newton on this stiff LQ Hamiltonian + boundary layer is fragile;
        Picard is the recommended inner solver -- see project notes)."""
        from scipy import sparse

        problem = _dirichlet_problem(n=31)
        hjb, _ = _pair(problem, n=31)
        D = 0.5 * problem.sigma**2
        dt = problem.dt
        hjb._build_gradient_operators()
        G, M, K = hjb._G_grad, hjb._M, hjb._K
        N_block, rhs_extra = hjb._weak_bc_terms(D)
        Hc = problem.hamiltonian_class
        x = hjb._disc.dof_coordinates
        m_n = np.ones(hjb.n_dof) / hjb.n_dof
        U0 = 0.5 * (x[:, 0] - 0.5) ** 2
        t = 0.2

        def residual(U):
            p = np.column_stack([Gd @ U for Gd in G])
            Hv = np.asarray(Hc(x, m_n, p, t=t), dtype=float).ravel()
            r = (M / dt) @ (U - U0) + D * (K @ U) + M @ Hv + N_block @ U
            return r - rhs_extra if rhs_extra is not None else r

        p0 = np.column_stack([Gd @ U0 for Gd in G])
        dHdp = np.asarray(Hc.dp(x, m_n, p0, t=t), dtype=float)
        if dHdp.ndim == 1:
            dHdp = dHdp.reshape(-1, 1)
        J = (M / dt + D * K + N_block).copy()
        for d in range(len(G)):
            J = J + M @ sparse.diags(dHdp[:, d]) @ G[d]
        J = J.toarray()

        rng = np.random.default_rng(0)
        e = rng.standard_normal(hjb.n_dof)
        eps = 1e-6
        fd = (residual(U0 + eps * e) - residual(U0 - eps * e)) / (2 * eps)
        assert np.linalg.norm(fd - J @ e) / np.linalg.norm(J @ e) < 1e-7

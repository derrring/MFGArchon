"""
Validation tests for Issue #1256 Part 1: anisotropic (d,d) volatility Σ in the
meshfree particle FP solver (FPParticleSolver, nD CPU path).

SDE convention (must match the FDM tensor-diffusion path, Issue #1249/#1276):

    dX = v dt + Σ dW,   dW ~ N(0, dt·I_d)   =>   diffusion tensor D = (1/2) Σ Σ^T.

Correctness proof (the gate): a free-diffusion ensemble (drift v = 0) started from a
single point, evolved for time T under a CONSTANT anisotropic Σ, must have empirical
particle covariance ≈ Σ Σ^T · T. A wrong apply (Σ Σ^T vs Σ, a transpose Σ^T Σ, or a
double-σ) fails this. The isotropic reduction Σ = σ I must give Cov ≈ σ² T · I, matching
the existing scalar-σ path.
"""

from __future__ import annotations

import numpy as np

from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import neumann_bc


def _hamiltonian():
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _free_diffusion_problem(T: float, Nt: int):
    """Large-domain 2D problem so a tight central cloud never touches the boundary
    over [0, T] (pure free diffusion, no BC interaction)."""
    geometry = TensorProductGrid(
        bounds=[(-3.0, 3.0), (-3.0, 3.0)],
        Nx_points=[16, 16],
        boundary_conditions=neumann_bc(dimension=2),
    )
    return MFGProblem(
        geometry=geometry,
        T=T,
        Nt=Nt,
        sigma=0.1,
        components=MFGComponents(
            m_initial=lambda x: np.exp(-np.sum(np.asarray(x) ** 2)),
            u_terminal=lambda x: 0.0,
            hamiltonian=_hamiltonian(),
        ),
    )


def _zero_drift(t, x, m):
    return np.zeros_like(x)


def _final_cloud_covariance(volatility_field, *, T=0.5, Nt=50, num_particles=20000, seed=11):
    """Run free diffusion from a single point (all particles at the origin) and return
    the empirical covariance of the final particle cloud.

    Only randomness is the per-step Brownian increment (global np.random seeded);
    initial particles are deterministic, density_mode='query_only' skips per-step KDE.
    """
    problem = _free_diffusion_problem(T, Nt)
    grid_shape = problem.geometry.get_grid_shape()
    m0 = np.ones(grid_shape) / np.prod(grid_shape)
    initial_particles = np.zeros((num_particles, 2))  # all at the origin (tight cloud)

    np.random.seed(seed)
    solver = FPParticleSolver(problem, num_particles=num_particles, density_mode="query_only")
    solver.solve_fp_system(
        M_initial=m0,
        drift_field=_zero_drift,
        volatility_field=volatility_field,
        initial_particles=initial_particles,
        show_progress=False,
    )
    final_cloud = solver.M_particles_trajectory[-1]
    assert final_cloud.shape == (num_particles, 2)
    return np.cov(final_cloud.T)


class TestAnisotropicCovarianceGate:
    """The correctness proof: empirical covariance ≈ Σ Σ^T · T."""

    def test_constant_anisotropic_covariance(self):
        """
        Σ = [[0.3, 0.1], [0.0, 0.2]] (non-symmetric), drift v=0, point start.

        Asserts Cov(X_T) ≈ Σ Σ^T · T entrywise, AND that it is distinguishable from the
        transpose-bug target Σ^T Σ · T (which differs in the off-diagonal and (1,1) entry).
        """
        T = 0.5
        Sigma = np.array([[0.3, 0.1], [0.0, 0.2]])
        emp_cov = _final_cloud_covariance(Sigma, T=T)

        expected = T * (Sigma @ Sigma.T)  # correct: dt·Σ Σ^T accumulated over T
        transpose_bug = T * (Sigma.T @ Sigma)  # what a dW @ Σ (transpose) apply would give

        # Gate: matches Σ Σ^T · T within Monte-Carlo tolerance (N=20000 ⇒ ~1-2% on entries).
        assert np.allclose(emp_cov, expected, rtol=0.08, atol=0.0025), (
            f"empirical cov\n{emp_cov}\nnot ~ Sigma Sigma^T * T\n{expected}"
        )
        # Discrimination: a transposed apply would NOT pass the same tolerance.
        assert not np.allclose(emp_cov, transpose_bug, rtol=0.08, atol=0.0025), (
            "empirical covariance is indistinguishable from the transpose-bug target; "
            "the test cannot discriminate Sigma Sigma^T from Sigma^T Sigma."
        )
        # The off-diagonal is the sharpest discriminator (0.01 correct vs 0.015 transposed).
        assert abs(emp_cov[0, 1] - emp_cov[1, 0]) < 1e-12  # covariance is symmetric
        assert abs(emp_cov[0, 1] - expected[0, 1]) < 0.0025

    def test_isotropic_reduction_matches_scalar(self):
        """
        Σ = σ I reduces to the existing isotropic path: Cov ≈ σ² T · I (off-diagonal ≈ 0).

        Also checks the matrix path and the scalar-float path give statistically the same
        covariance (the (d,d) branch is a strict generalization, not a divergent path).
        """
        T = 0.5
        sigma = 0.25
        Sigma = sigma * np.eye(2)

        cov_matrix = _final_cloud_covariance(Sigma, T=T, seed=7)
        cov_scalar = _final_cloud_covariance(float(sigma), T=T, seed=7)

        expected_diag = sigma**2 * T  # = 0.03125
        # Diagonal matches σ² T; off-diagonal ≈ 0.
        assert np.allclose(np.diag(cov_matrix), expected_diag, rtol=0.08)
        assert abs(cov_matrix[0, 1]) < 0.0025
        # Matrix Σ = σ I and scalar σ agree statistically (same covariance structure).
        assert np.allclose(cov_matrix, cov_scalar, rtol=0.10, atol=0.0025)

    def test_spatial_anisotropic_field_matches_constant(self):
        """
        A spatially CONSTANT Σ supplied as a full (*grid_shape, d, d) field must reproduce
        the same covariance as the constant (d, d) matrix. Exercises the spatial-matrix
        interpolation branch (_interpolate_matrix_field_at_particles).
        """
        T = 0.5
        Sigma = np.array([[0.3, 0.1], [0.0, 0.2]])
        grid_shape = (16, 16)
        # Broadcast the constant matrix over the whole grid: shape (16, 16, 2, 2).
        Sigma_field = np.broadcast_to(Sigma, (*grid_shape, 2, 2)).copy()

        cov_field = _final_cloud_covariance(Sigma_field, T=T, seed=11)
        expected = T * (Sigma @ Sigma.T)
        assert np.allclose(cov_field, expected, rtol=0.08, atol=0.0025), (
            f"spatial-field covariance\n{cov_field}\nnot ~ Sigma Sigma^T * T\n{expected}"
        )


if __name__ == "__main__":
    t = TestAnisotropicCovarianceGate()
    t.test_constant_anisotropic_covariance()
    t.test_isotropic_reduction_matches_scalar()
    t.test_spatial_anisotropic_field_matches_constant()
    print("All anisotropic-sigma validation tests passed.")

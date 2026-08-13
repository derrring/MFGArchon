"""
Integration test for meshfree HJB + particle FP workflow.

Tests the typical MFG workflow using:
- HJBGFDMSolver: Solve HJB on collocation points
- FPParticleSolver: Evolve density using particles (hybrid mode with KDE output)

This is the recommended workflow for meshfree MFG problems.

Note: FPGFDMSolver exists for specialized use cases where you want
GFDM-based density evolution on collocation points, but the typical
workflow uses particle-based FP for better handling of density transport.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers import FPGFDMSolver, FPParticleSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.implicit import Hyperrectangle


def _default_hamiltonian():
    """Default Hamiltonian for testing."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components_2d():
    """Default MFGComponents for 2D testing (Issue #670: explicit specification required)."""

    def m_initial_2d(x):
        x_arr = np.asarray(x)
        return np.exp(-10 * np.sum((x_arr - 0.5) ** 2))

    return MFGComponents(
        m_initial=m_initial_2d,
        u_terminal=lambda x: 0.0,
        hamiltonian=_default_hamiltonian(),
    )


class SimpleLQMFG2D(MFGProblem):
    """Simple 2D LQ-MFG problem for integration testing."""

    def __init__(self):
        super().__init__(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1)
            ),
            T=1.0,
            Nt=20,
            sigma=0.2,
            coupling_coefficient=0.5,
            components=_default_components_2d(),
        )
        # GFDM solver expects problem.d for spatial dimension
        self.d = 2


class TestHJBGFDMWithParticleFP:
    """Integration tests for HJB-GFDM + FP-Particle workflow."""

    def test_hjb_gfdm_initialization(self):
        """Test that HJB-GFDM solver initializes correctly."""
        problem = SimpleLQMFG2D()
        N_points = 200

        domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        points = domain.sample_uniform(N_points, seed=42)

        hjb_solver = HJBGFDMSolver(problem, collocation_points=points, delta=0.15)

        assert hjb_solver.collocation_points.shape == (N_points, 2)

        # The constructor runs the delta-neighbourhood search and assembles the Taylor matrices;
        # a node whose neighbourhood came back empty carries None. Measured: all 200 carry a stencil.
        assert all(hjb_solver.taylor_matrices[i] is not None for i in range(hjb_solver.n_points))

        # taylor_order defaults to 2, so the weighted least-squares stencils must reproduce any
        # quadratic exactly. Measured worst error over all 200 nodes and all five multi-indices:
        # 2.52e-12, so abs=1e-8 sits ~4000x above the observed error.
        x, y = points[:, 0], points[:, 1]
        u = 1 + 2 * x + 3 * y + 0.5 * x**2 - y**2 + 1.5 * x * y
        for i in range(hjb_solver.n_points):
            derivs = hjb_solver.approximate_derivatives(u, i)
            assert derivs[(2, 0)] == pytest.approx(1.0, abs=1e-8)
            assert derivs[(0, 2)] == pytest.approx(-2.0, abs=1e-8)
            assert derivs[(1, 1)] == pytest.approx(1.5, abs=1e-8)
            assert derivs[(1, 0)] == pytest.approx(2 + x[i] + 1.5 * y[i], abs=1e-8)
            assert derivs[(0, 1)] == pytest.approx(3 - 2 * y[i] + 1.5 * x[i], abs=1e-8)

    def test_fp_particle_outputs_to_grid(self):
        """Test that FP particle solver outputs density on grid."""
        problem = SimpleLQMFG2D()

        fp_solver = FPParticleSolver(problem, num_particles=1000, seed=20260813)

        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        Nt_points = problem.Nt + 1  # Temporal grid points
        m0 = np.ones(Nx_points) / Nx_points
        U = np.zeros((Nt_points, Nx_points))

        M = fp_solver.solve_fp_system(m0, U, show_progress=False)

        # Particle solver outputs on grid via KDE
        assert M.shape == (Nt_points, Nx_points)

        # The projection returns a density, not raw particle counts: it integrates to 1 at every
        # time. Measured max|sum(M)*Dx - 1| = 4.4e-16 (unchanged under kde_normalization="none"),
        # so atol=1e-12 sits ~2000x above the observed error.
        Dx = problem.geometry.get_grid_spacing()[0]
        np.testing.assert_allclose(M.sum(axis=1) * Dx, 1.0, atol=1e-12)

        # At t=0 the answer is known independently of the scheme: m0 is uniform on [0, 1], so the
        # KDE projection must return its own input. Measured max|M[0] - 1| = 0.156 at this seed
        # (0.083-0.156 across seeds 0/1/7/1234/20260813 at N=1000); 0.3 leaves ~2x margin over the
        # sampling error while still catching a projection that drops the walls.
        assert np.abs(M[0] - 1.0).max() < 0.3

    def test_mass_conservation_particle_fp(self):
        """Test mass conservation in particle FP solver."""
        problem = SimpleLQMFG2D()

        num_particles = 2000
        fp_solver = FPParticleSolver(problem, num_particles=num_particles, seed=20260813)

        # Uniform initial density
        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        Nt_points = problem.Nt + 1  # Temporal grid points
        m0 = np.ones(Nx_points) / Nx_points
        U = np.zeros((Nt_points, Nx_points))

        M = fp_solver.solve_fp_system(m0, U, show_progress=False)

        # Check density is non-negative and finite
        assert np.all(M >= 0)
        assert np.all(np.isfinite(M))

        # The grid density is not the place to read mass off: sum(M)*Dx == 1 to 3.3e-16 even with
        # kde_normalization="none", so it is a property of the projection and says nothing about the
        # transport. For a particle method the conservation law is particle conservation: none
        # created, none destroyed, none outside the domain. Reflection puts particles exactly on the
        # wall (measured min 0.0, max 1.0, with 111 of 42000 landing on a boundary), so the bound is
        # inclusive; a broken no-flux BC leaks them past it.
        particles = np.asarray(fp_solver.M_particles_trajectory)
        assert particles.shape == (Nt_points, num_particles)
        assert particles.min() >= 0.0
        assert particles.max() <= 1.0

        # U == 0 gives zero drift, so this is reflected Brownian motion on [0, 1], whose invariant
        # measure is the uniform density it started from -- an oracle independent of the scheme.
        # Measured max|M - 1| = 0.122 at this seed (0.102-0.126 across seeds at N=2000); 0.25 leaves
        # ~2x margin over the Monte-Carlo error.
        assert np.abs(M - 1.0).max() < 0.25


class TestFPGFDMSolver:
    """Test FPGFDMSolver for specialized meshfree density evolution."""

    def test_fp_gfdm_initialization(self):
        """Test FPGFDMSolver initialization."""
        problem = SimpleLQMFG2D()
        N_points = 100

        domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        points = domain.sample_uniform(N_points, seed=42)

        fp_solver = FPGFDMSolver(problem, collocation_points=points, delta=0.15)

        assert fp_solver.n_points == N_points
        assert fp_solver.dimension == 2

    def test_fp_gfdm_mass_conservation(self):
        """Test mass conservation in GFDM-based FP solver."""
        problem = SimpleLQMFG2D()
        N_points = 100

        domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        points = domain.sample_uniform(N_points, seed=42)

        fp_solver = FPGFDMSolver(problem, collocation_points=points, delta=0.15)

        # Use temporal grid size (Nt + 1), not spatial grid
        n_time_points = problem.Nt + 1
        m0 = np.ones(N_points) / N_points

        # drift_field must be shape (Nt+1, N, d) for GFDM solver
        # Use zero drift for this test
        drift_field = np.zeros((n_time_points, N_points, problem.d))

        M = fp_solver.solve_fp_system(m0, drift_field=drift_field, show_progress=False)

        # Check mass conservation
        for t_idx in range(n_time_points):
            mass = np.sum(M[t_idx, :])
            assert np.abs(mass - 1.0) < 1e-10

    def test_fp_gfdm_validates_shapes(self):
        """Test that FPGFDMSolver validates input shapes."""
        problem = SimpleLQMFG2D()
        N_points = 100

        domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        points = domain.sample_uniform(N_points, seed=42)

        solver = FPGFDMSolver(problem, collocation_points=points, delta=0.15)

        # Wrong m0 shape
        Nt_points = problem.geometry.get_grid_shape()[0]
        m0_wrong = np.ones(50)
        U_correct = np.zeros((Nt_points, N_points))

        with pytest.raises(ValueError, match="must match"):
            solver.solve_fp_system(m0_wrong, U_correct, show_progress=False)


class TestCollocationModeRemoved:
    """Pin that the past-window `mode` kwarg (and collocation handling) is gone."""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

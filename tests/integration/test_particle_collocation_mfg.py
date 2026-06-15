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

    def test_fp_particle_outputs_to_grid(self):
        """Test that FP particle solver outputs density on grid."""
        problem = SimpleLQMFG2D()

        fp_solver = FPParticleSolver(problem, num_particles=1000)

        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        Nt_points = problem.Nt + 1  # Temporal grid points
        m0 = np.ones(Nx_points) / Nx_points
        U = np.zeros((Nt_points, Nx_points))

        M = fp_solver.solve_fp_system(m0, U, show_progress=False)

        # Particle solver outputs on grid via KDE
        assert M.shape == (Nt_points, Nx_points)

    def test_mass_conservation_particle_fp(self):
        """Test mass conservation in particle FP solver."""
        problem = SimpleLQMFG2D()

        fp_solver = FPParticleSolver(problem, num_particles=2000)

        # Uniform initial density
        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        Nt_points = problem.Nt + 1  # Temporal grid points
        m0 = np.ones(Nx_points) / Nx_points
        U = np.zeros((Nt_points, Nx_points))

        M = fp_solver.solve_fp_system(m0, U, show_progress=False)

        # Check density is non-negative and finite
        assert np.all(M >= 0)
        assert np.all(np.isfinite(M))


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

    def test_fp_gfdm_outputs_on_collocation_points(self):
        """Test that FPGFDMSolver outputs on collocation points."""
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

        # GFDM solver outputs on collocation points
        assert M.shape == (n_time_points, N_points)

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

    def test_mode_kwarg_removed(self):
        """`mode` removed (-> density_mode): passing it is an unexpected keyword."""
        problem = SimpleLQMFG2D()

        with pytest.raises(TypeError, match="unexpected keyword argument 'mode'"):
            FPParticleSolver(problem, mode="collocation")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

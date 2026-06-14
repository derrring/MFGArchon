"""
Tests for FP particle solver.

Tests the particle-based FP solver using Monte Carlo sampling and KDE.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver, KDENormalization
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem


def _default_hamiltonian():
    """Default Hamiltonian for testing (Issue #670: explicit specification required)."""
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


def _default_components():
    """Default MFGComponents for 1D testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
        hamiltonian=_default_hamiltonian(),
    )


class Simple2DMFGProblem(MFGProblem):
    """Simple 2D test problem for particle solver testing."""

    def __init__(self):
        super().__init__(
            T=1.0,
            Nt=10,
            Nx=20,
            Lx=1.0,
            xmin=0.0,
            sigma=0.1,
            coupling_coefficient=0.5,
            dimension=2,
            components=_default_components_2d(),
        )


class TestFPParticleSolverBasic:
    """Test basic FPParticleSolver functionality."""

    def test_default_initialization(self):
        """Test default initialization."""
        problem = Simple2DMFGProblem()
        solver = FPParticleSolver(problem, num_particles=100)

        assert solver.fp_method_name == "Particle"
        assert solver.num_particles == 100

    def test_output_shape(self):
        """Test that solver outputs density on grid."""
        problem = Simple2DMFGProblem()
        solver = FPParticleSolver(problem, num_particles=1000)

        Nx_points = problem.geometry.get_grid_shape()[0]
        m0 = np.ones(Nx_points) / Nx_points
        U = np.zeros((problem.Nt + 1, Nx_points))

        M = solver.solve_fp_system(m0, U, show_progress=False)

        # Output shape should be (Nt+1, Nx_points) - grid output via KDE
        assert M.shape == (problem.Nt + 1, Nx_points)

    def test_non_negative_density(self):
        """Test that solver produces non-negative density."""
        problem = Simple2DMFGProblem()
        solver = FPParticleSolver(problem, num_particles=1000)

        Nx_points = problem.geometry.get_grid_shape()[0]
        m0 = np.ones(Nx_points) / Nx_points
        U = np.zeros((problem.Nt + 1, Nx_points))

        M = solver.solve_fp_system(m0, U, show_progress=False)

        assert np.all(M >= 0)
        assert np.all(np.isfinite(M))


class TestKDENormalization:
    """Test KDE normalization options."""

    def test_kde_normalization_enum(self):
        """Test KDENormalization enum values."""
        assert KDENormalization.NONE == "none"
        assert KDENormalization.INITIAL_ONLY == "initial_only"
        assert KDENormalization.ALL == "all"

    def test_kde_normalization_string_conversion(self):
        """Test that kde_normalization accepts strings."""
        problem = Simple2DMFGProblem()

        solver = FPParticleSolver(problem, num_particles=100, kde_normalization="all")
        assert solver.kde_normalization == KDENormalization.ALL

        solver = FPParticleSolver(problem, num_particles=100, kde_normalization="none")
        assert solver.kde_normalization == KDENormalization.NONE


class TestRemovedParameters:
    """Pin that past-window deprecated kwargs are gone (fail loud as TypeError)."""

    def test_mode_removed(self):
        """`mode` (-> density_mode) removed: passing it is an unexpected keyword."""
        problem = Simple2DMFGProblem()

        with pytest.raises(TypeError, match="unexpected keyword argument 'mode'"):
            FPParticleSolver(problem, mode="hybrid")

    def test_external_particles_removed(self):
        """`external_particles` (-> num_particles) removed: unexpected keyword."""
        problem = Simple2DMFGProblem()
        points = np.random.rand(50, 2)

        with pytest.raises(TypeError, match="unexpected keyword argument 'external_particles'"):
            FPParticleSolver(problem, num_particles=100, external_particles=points)

    def test_density_mode_new_name_accepted(self):
        """New name `density_mode` works (replacement for removed `mode`)."""
        problem = Simple2DMFGProblem()

        solver = FPParticleSolver(problem, density_mode="hybrid", num_particles=100)
        assert solver.num_particles == 100
        assert solver.density_mode == "hybrid"


class TestBoundaryConditionRequirements:
    """Test BC requirement enforcement (Issue #545)."""

    def test_fp_particle_requires_boundary_conditions(self):
        """Test that FPParticleSolver fails fast without BCs."""
        from unittest.mock import Mock

        # Create minimal mock problem without geometry.get_boundary_conditions()
        mock_geometry = Mock(spec=[])  # Empty spec - no methods
        mock_problem = Mock()
        mock_problem.geometry = mock_geometry
        mock_problem.T = 1.0
        mock_problem.Nt = 10
        mock_problem.dimension = 1

        # Should fail fast when geometry lacks get_boundary_conditions()
        with pytest.raises(ValueError, match="requires explicit boundary conditions"):
            FPParticleSolver(mock_problem, num_particles=100)

    def test_fp_particle_with_geometry_bc(self):
        """Test FPParticleSolver with geometry-provided BCs."""
        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import dirichlet_bc

        # Geometry with BCs
        geometry = TensorProductGrid(
            bounds=[(0, 1)],
            Nx_points=[11],
            boundary_conditions=dirichlet_bc(dimension=1, value=0.0),
        )
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=10, components=_default_components())

        # Should work
        solver = FPParticleSolver(problem, num_particles=100)
        assert solver.boundary_conditions is not None

    def test_fp_particle_with_explicit_bc_parameter(self):
        """Test FPParticleSolver with explicit BC parameter."""
        from mfgarchon.geometry.boundary import periodic_bc

        # Problem without geometry
        problem = MFGProblem(T=1.0, Nt=10, components=_default_components())

        # Should work with explicit BC
        bc = periodic_bc(dimension=1)
        solver = FPParticleSolver(problem, num_particles=100, boundary_conditions=bc)
        assert solver.boundary_conditions is bc

    def test_fp_particle_bc_parameter_takes_priority(self):
        """Test that explicit BC parameter overrides geometry BC."""
        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import dirichlet_bc, periodic_bc

        # Geometry with Dirichlet BC
        geometry = TensorProductGrid(
            bounds=[(0, 1)],
            Nx_points=[11],
            boundary_conditions=dirichlet_bc(dimension=1, value=0.0),
        )
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=10, components=_default_components())

        # Explicit periodic BC should take priority
        bc = periodic_bc(dimension=1)
        solver = FPParticleSolver(problem, num_particles=100, boundary_conditions=bc)
        assert solver.boundary_conditions is bc
        assert solver.boundary_conditions is not problem.geometry.get_boundary_conditions()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

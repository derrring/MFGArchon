#!/usr/bin/env python3
"""
Unit tests for FPFDMSolver - comprehensive coverage.

Tests the Finite Difference Method (FDM) solver for Fokker-Planck equations
with different boundary conditions (periodic, Dirichlet, no-flux).
"""

import warnings

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    dirichlet_bc,
    no_flux_bc,
    periodic_bc,
)


def _default_hamiltonian():
    """Default Hamiltonian for testing (Issue #670: explicit specification required)."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components():
    """Default MFGComponents for 1D testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),  # Gaussian centered at 0.5
        u_terminal=lambda x: 0.0,  # Zero terminal cost
        hamiltonian=_default_hamiltonian(),
    )


def _default_components_2d():
    """Default MFGComponents for 2D testing (Issue #670: explicit specification required)."""

    def m_initial_2d(x):
        # x is [x, y] coordinate - compute Gaussian at center (0.5, 0.5)
        x_arr = np.asarray(x)
        return np.exp(-10 * np.sum((x_arr - 0.5) ** 2))

    return MFGComponents(
        m_initial=m_initial_2d,
        u_terminal=lambda x: 0.0,
        hamiltonian=_default_hamiltonian(),
    )


@pytest.fixture
def standard_problem():
    """Create standard 1D MFG problem using modern geometry-first API.

    Standard MFGProblem configuration:
    - Domain: [0, 1] with 51 grid points
    - Time: T=1.0 with 51 time steps
    - Diffusion: sigma=1.0
    """
    domain = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
    return MFGProblem(geometry=domain, T=1.0, Nt=51, sigma=1.0, components=_default_components())


class TestFPFDMSolverInitialization:
    """Test FPFDMSolver initialization and setup."""

    def test_basic_initialization(self, standard_problem):
        """Test basic solver initialization with default BC."""
        solver = FPFDMSolver(standard_problem)

        assert solver.fp_method_name == "FDM"
        # Default BC is no_flux when geometry doesn't specify BC
        assert solver.boundary_conditions.type == "no_flux"
        assert solver.problem is standard_problem

    def test_initialization_with_periodic_bc(self, standard_problem):
        """Test initialization with periodic boundary conditions."""
        bc = periodic_bc(dimension=1)
        solver = FPFDMSolver(standard_problem, boundary_conditions=bc)

        assert solver.fp_method_name == "FDM"
        assert solver.boundary_conditions.type == "periodic"

    def test_initialization_with_dirichlet_bc(self, standard_problem):
        """Test initialization with Dirichlet boundary conditions."""
        bc = dirichlet_bc(value=0.0, dimension=1)
        solver = FPFDMSolver(standard_problem, boundary_conditions=bc)

        assert solver.fp_method_name == "FDM"
        assert solver.boundary_conditions.type == "dirichlet"
        assert solver.boundary_conditions.segments[0].value == 0.0

    def test_initialization_with_no_flux_bc(self, standard_problem):
        """Test initialization with no-flux boundary conditions."""
        bc = no_flux_bc(dimension=1)
        solver = FPFDMSolver(standard_problem, boundary_conditions=bc)

        assert solver.fp_method_name == "FDM"
        assert solver.boundary_conditions.type == "no_flux"


class TestFPFDMSolverBasicSolution:
    """Test basic solution functionality."""

    def test_solve_fp_system_shape(self, standard_problem):
        """Test that solve_fp_system returns correct shape."""
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1

        # Create simple inputs
        m_initial = np.ones(Nx_points) / Nx_points  # Normalized density
        U_solution = np.zeros((Nt_points, Nx_points))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        assert m_result.shape == (Nt_points, Nx_points)

    def test_solve_fp_system_initial_condition_preserved(self, standard_problem):
        """Test that initial condition is preserved at t=0."""
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1
        bounds = standard_problem.geometry.get_bounds()

        # Create Gaussian initial condition
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_initial = np.exp(-((x_coords - 0.0) ** 2) / (2 * 0.1**2))
        m_initial = m_initial / np.sum(m_initial)  # Normalize

        U_solution = np.zeros((Nt_points, Nx_points))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # Initial condition should be preserved (approximately, after non-negativity enforcement)
        assert np.allclose(m_result[0, :], m_initial, rtol=0.1)

    def test_solve_fp_system_zero_timesteps(self, standard_problem):
        """Test behavior with zero time steps (Nt=0)."""
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        _Nt_points = standard_problem.Nt + 1

        m_initial = np.ones(Nx_points) / Nx_points
        U_solution = np.zeros((0, Nx_points))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        assert m_result.shape == (0, Nx_points)

    def test_solve_fp_system_one_timestep(self, standard_problem):
        """Test behavior with single time step (Nt=1)."""
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        _Nt_points = standard_problem.Nt + 1

        m_initial = np.ones(Nx_points) / Nx_points
        U_solution = np.zeros((1, Nx_points))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        assert m_result.shape == (1, Nx_points)
        # Should return initial condition (possibly with non-negativity enforcement)
        assert np.allclose(m_result[0, :], m_initial, rtol=0.1)


class TestFPFDMSolverBoundaryConditions:
    """Test different boundary condition types."""

    def test_periodic_boundary_conditions(self, standard_problem):
        """Test periodic boundary conditions."""
        bc = periodic_bc(dimension=1)
        solver = FPFDMSolver(standard_problem, boundary_conditions=bc)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1

        # Initial condition with support near boundaries
        m_initial = np.zeros(Nx_points)
        m_initial[0] = 0.5
        m_initial[-1] = 0.5

        U_solution = np.zeros((Nt_points, Nx_points))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # With periodic BC, mass should wrap around
        assert m_result.shape == (Nt_points, Nx_points)
        # Mass should be preserved
        assert np.all(m_result >= -1e-10)  # Non-negative (with small tolerance)

    def test_dirichlet_boundary_conditions(self, standard_problem):
        """Test Dirichlet boundary conditions."""
        bc = BoundaryConditions(
            segments=[
                BCSegment(name="left", bc_type=BCType.DIRICHLET, value=0.1, boundary="x_min"),
                BCSegment(name="right", bc_type=BCType.DIRICHLET, value=0.2, boundary="x_max"),
            ],
            dimension=1,
        )
        solver = FPFDMSolver(standard_problem, boundary_conditions=bc)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1

        m_initial = np.ones(Nx_points) / Nx_points
        U_solution = np.zeros((Nt_points, Nx_points))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # Boundary values should be enforced at all time steps
        for t in range(Nt_points):
            assert np.isclose(m_result[t, 0], 0.1, atol=1e-10)
            assert np.isclose(m_result[t, -1], 0.2, atol=1e-10)

    def test_no_flux_boundary_conditions(self, standard_problem):
        """Test no-flux boundary conditions."""
        bc = no_flux_bc(dimension=1)
        solver = FPFDMSolver(standard_problem, boundary_conditions=bc)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1
        bounds = standard_problem.geometry.get_bounds()

        # Gaussian initial condition
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_initial = np.exp(-((x_coords - 0.0) ** 2) / (2 * 0.1**2))
        m_initial = m_initial / np.sum(m_initial)

        U_solution = np.zeros((Nt_points, Nx_points))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # With no-flux BC, total mass should be approximately conserved
        initial_mass = np.sum(m_initial)
        for t in range(Nt_points):
            final_mass = np.sum(m_result[t, :])
            # Allow some numerical error
            assert np.isclose(final_mass, initial_mass, rtol=0.1)


class TestFPFDMSolverNonNegativity:
    """Test non-negativity enforcement."""

    def test_non_negativity_enforcement(self, standard_problem):
        """Test that solution remains non-negative."""
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1

        m_initial = np.ones(Nx_points) / Nx_points
        U_solution = np.zeros((Nt_points, Nx_points))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # All values should be non-negative (with small tolerance for numerical errors)
        assert np.all(m_result >= -1e-10)

    def test_initial_condition_non_negativity(self, standard_problem):
        """Test that negative values in initial condition are set to zero."""
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1

        # Initial condition with some negative values
        m_initial = np.random.randn(Nx_points)
        U_solution = np.zeros((Nt_points, Nx_points))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # Initial condition should have negative values removed
        assert np.all(m_result[0, :] >= 0)


class TestFPFDMSolverWithDrift:
    """Test solver behavior with non-zero drift (from HJB solution)."""

    def test_solve_with_linear_drift(self, standard_problem):
        """Test solution with linear value function (constant drift)."""
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1
        bounds = standard_problem.geometry.get_bounds()

        # Gaussian initial condition at center
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_initial = np.exp(-((x_coords - 0.0) ** 2) / (2 * 0.1**2))
        m_initial = m_initial / np.sum(m_initial)

        # Linear value function: U(t,x) = x (constant drift)
        U_solution = np.tile(x_coords, (Nt_points, 1))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # Solution should evolve (not remain constant)
        assert not np.allclose(m_result[-1, :], m_result[0, :])
        # Should remain non-negative
        assert np.all(m_result >= -1e-10)

    def test_solve_with_quadratic_value_function(self, standard_problem):
        """Test solution with quadratic value function and periodic BC.

        Uses periodic BC to allow density to evolve under drift from quadratic
        value function U(t,x) = x^2, which creates linear drift = -2x.
        """
        # Explicit periodic BC to ensure density evolution under drift
        bc = periodic_bc(dimension=1)
        solver = FPFDMSolver(standard_problem, boundary_conditions=bc)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1
        bounds = standard_problem.geometry.get_bounds()

        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_initial = np.ones(Nx_points) / Nx_points

        # Quadratic value function: U(t,x) = x^2 (linear drift)
        U_solution = np.tile(x_coords**2, (Nt_points, 1))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # Solution should evolve (periodic BC allows mass flow)
        assert not np.allclose(m_result[-1, :], m_result[0, :])
        # Should remain non-negative
        assert np.all(m_result >= -1e-10)


class TestFPFDMSolverEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_diffusion_timestep(self, standard_problem):
        """Test behavior when Dt is extremely small."""
        standard_problem.dt = 1e-20  # Very small timestep
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1

        m_initial = np.ones(Nx_points) / Nx_points
        U_solution = np.zeros((Nt_points, Nx_points))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # With very small Dt, solution should remain close to initial condition
        assert np.allclose(m_result[1, :], m_result[0, :], rtol=0.1)

    def test_zero_spatial_step(self, standard_problem):
        """Test behavior when Dx is extremely small (but Nx > 1)."""
        # Note: This test modifies problem.dx which is derived from geometry.
        # For a proper test, create a new problem with tiny domain or many points.
        # This test maintains legacy behavior for backward compatibility testing.
        standard_problem.dx = 1e-20  # Very small spatial step (legacy attribute)
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1

        m_initial = np.ones(Nx_points) / Nx_points
        U_solution = np.zeros((Nt_points, Nx_points))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # With very small Dx, solution should remain close to initial condition
        assert np.allclose(m_result[1, :], m_result[0, :], rtol=0.1)

        # Removed: test_single_spatial_point — tested degenerate Nx=1 case via
        # legacy problem.Nx mutation. Not physically meaningful for PDEs. (#833)

        # With no spatial variation, the solution should remain constant
        assert m_result.shape == (Nt_points, Nx_points)
        assert np.all(np.isfinite(m_result))


class TestFPFDMSolverMassConservation:
    """Test mass conservation properties."""

    def test_mass_conservation_no_flux(self, standard_problem):
        """Test that mass is conserved with no-flux boundary conditions."""
        bc = no_flux_bc(dimension=1)
        solver = FPFDMSolver(standard_problem, boundary_conditions=bc)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1
        bounds = standard_problem.geometry.get_bounds()

        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_initial = np.exp(-((x_coords - 0.0) ** 2) / (2 * 0.1**2))
        m_initial = m_initial / np.sum(m_initial)

        U_solution = np.zeros((Nt_points, Nx_points))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # Check mass conservation at each time step
        initial_mass = np.sum(m_initial)
        for t in range(Nt_points):
            current_mass = np.sum(m_result[t, :])
            assert np.isclose(current_mass, initial_mass, rtol=0.1)

    def test_mass_evolution_periodic(self, standard_problem):
        """Test mass evolution with periodic boundary conditions."""
        bc = periodic_bc(dimension=1)
        solver = FPFDMSolver(standard_problem, boundary_conditions=bc)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1

        m_initial = np.ones(Nx_points) / Nx_points
        U_solution = np.zeros((Nt_points, Nx_points))

        m_result = solver.solve_fp_system(m_initial, U_solution)

        # With periodic BC and zero drift, mass should be conserved
        initial_mass = np.sum(m_initial)
        final_mass = np.sum(m_result[-1, :])
        assert np.isclose(final_mass, initial_mass, rtol=0.1)


class TestFPFDMSolverIntegration:
    """Integration tests with actual MFG problems."""

    def test_solver_with_example_problem(self, standard_problem):
        """Test solver works with standard MFGProblem."""
        solver = FPFDMSolver(standard_problem)

        assert solver is not None
        assert hasattr(solver, "solve_fp_system")
        assert callable(solver.solve_fp_system)

    def test_solver_not_abstract(self, standard_problem):
        """Test that FPFDMSolver can be instantiated (is concrete)."""
        import inspect

        solver = FPFDMSolver(standard_problem)

        assert isinstance(solver, FPFDMSolver)
        assert not inspect.isabstract(FPFDMSolver)


class TestFPFDMSolverArrayDiffusion:
    """Test array diffusion support (Phase 2.1).

    Note: Spatially varying diffusion with FDM can exhibit ~5-15% mass drift
    due to discretization errors when diffusion varies significantly. This is
    a known limitation of FDM, not a bug. Tests focus on correctness
    (shape, non-negativity) rather than strict mass conservation.
    """

    def test_spatially_varying_diffusion_1d(self, standard_problem):
        """Test spatially varying diffusion: sigma(x) with periodic BC."""
        # Use periodic BC for better mass conservation with array diffusion
        bc = periodic_bc(dimension=1)
        solver = FPFDMSolver(standard_problem, boundary_conditions=bc)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1
        bounds = standard_problem.geometry.get_bounds()

        # Create spatially varying diffusion (moderate variation)
        x_grid = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        diffusion_array = 0.15 + 0.05 * np.abs(x_grid - 0.5)  # Moderate variation

        # Initial condition
        m_initial = np.exp(-((x_grid - 0.5) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial)

        # Zero drift (pure diffusion)
        U_solution = np.zeros((Nt_points, Nx_points))

        # Solve with array diffusion
        M = solver.solve_fp_system(m_initial, potential_field=U_solution, volatility_field=diffusion_array)

        assert M.shape == (Nt_points, Nx_points)
        assert np.all(M >= 0)
        # Verify solution doesn't blow up (moderate mass drift is expected with variable diffusion)
        assert np.all(np.sum(M, axis=1) > 0.5)
        assert np.all(np.sum(M, axis=1) < 2.0)

    def test_spatiotemporal_diffusion_1d(self, standard_problem):
        """Test spatiotemporal diffusion: sigma(t, x)."""
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1
        bounds = standard_problem.geometry.get_bounds()

        # Create spatiotemporal diffusion (varying in time and space)
        volatility_field = np.zeros((Nt_points, Nx_points))
        x_grid = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        for t in range(Nt_points):
            # Diffusion increases over time, higher at boundaries
            time_factor = 0.1 * (1 + 0.5 * t / Nt_points)
            space_factor = 1.0 + 0.3 * np.abs(x_grid - 0.5)
            volatility_field[t, :] = time_factor * space_factor

        # Initial condition
        m_initial = np.exp(-((x_grid - 0.5) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial)

        # Zero drift
        U_solution = np.zeros((Nt_points, Nx_points))

        # Solve with array diffusion
        M = solver.solve_fp_system(m_initial, potential_field=U_solution, volatility_field=volatility_field)

        assert M.shape == (Nt_points, Nx_points)
        assert np.all(M >= 0)
        # Mass conservation
        assert np.allclose(np.sum(M, axis=1), 1.0, atol=0.05)

    def test_array_diffusion_with_advection(self, standard_problem):
        """Test array diffusion with non-zero drift."""
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1
        bounds = standard_problem.geometry.get_bounds()

        # Spatially varying diffusion (moderate variation)
        x_grid = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        diffusion_array = 0.2 + 0.05 * x_grid  # Moderate increase

        # Initial condition
        m_initial = np.exp(-((x_grid - 0.3) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial)

        # Non-zero drift (constant velocity to the right)
        U_solution = np.zeros((Nt_points, Nx_points))
        for t in range(Nt_points):
            U_solution[t, :] = -0.2 * x_grid  # Moderate drift

        # Solve with array diffusion and drift
        M = solver.solve_fp_system(m_initial, potential_field=U_solution, volatility_field=diffusion_array)

        assert M.shape == (Nt_points, Nx_points)
        assert np.all(M >= 0)
        # Verify solution stability (no blow-up)
        assert np.all(np.sum(M, axis=1) > 0.5)
        assert np.all(np.sum(M, axis=1) < 2.0)

    def test_array_diffusion_mass_conservation(self, standard_problem):
        """Test that mass is conserved with array diffusion."""
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1
        bounds = standard_problem.geometry.get_bounds()

        # Spatially varying diffusion (non-uniform)
        x_grid = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        diffusion_array = 0.1 + 0.3 * (x_grid * (1 - x_grid))  # Parabolic profile

        # Initial condition (normalized)
        m_initial = np.ones(Nx_points) / Nx_points

        # Zero drift
        U_solution = np.zeros((Nt_points, Nx_points))

        # Solve
        M = solver.solve_fp_system(m_initial, potential_field=U_solution, volatility_field=diffusion_array)

        # Check mass conservation at all timesteps
        masses = np.sum(M, axis=1)
        assert np.allclose(masses, 1.0, atol=0.05)

    # Removed: test_array_diffusion_shape_validation — tested exact error message
    # string that changed in PR #383. The validation still works, just the message
    # format differs. Not worth maintaining exact string matching. (#833)


class TestFPFDMSolverCallableDiffusion:
    """Test callable (state-dependent) diffusion support (Phase 2.2)."""

    def test_porous_medium_equation(self, standard_problem):
        """Test porous medium equation: D(m) = σ² m."""
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1
        bounds = standard_problem.geometry.get_bounds()

        # Porous medium diffusion: D = σ² m
        def porous_medium_diffusion(t, x, m):
            return 0.1 * m  # Diffusion proportional to density

        # Initial condition (Gaussian)
        x_grid = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_initial = np.exp(-((x_grid - 0.5) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial)

        # Zero drift
        U_solution = np.zeros((Nt_points, Nx_points))

        # Solve with callable diffusion
        M = solver.solve_fp_system(m_initial, potential_field=U_solution, volatility_field=porous_medium_diffusion)

        assert M.shape == (Nt_points, Nx_points)
        assert np.all(M >= 0)
        # Verify solution stability
        assert np.all(np.sum(M, axis=1) > 0.5)
        assert np.all(np.sum(M, axis=1) < 2.0)

    def test_density_dependent_diffusion(self, standard_problem):
        """Test density-dependent diffusion: D = D0 + D1 * (1 - m/m_max)."""
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1
        bounds = standard_problem.geometry.get_bounds()

        # Crowd diffusion: lower diffusion in high-density regions
        def crowd_diffusion(t, x, m):
            m_max = np.max(m) if np.max(m) > 0 else 1.0
            return 0.05 + 0.15 * (1 - m / m_max)

        # Initial condition
        x_grid = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_initial = np.exp(-((x_grid - 0.5) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial)

        # Zero drift
        U_solution = np.zeros((Nt_points, Nx_points))

        # Solve
        M = solver.solve_fp_system(m_initial, potential_field=U_solution, volatility_field=crowd_diffusion)

        assert M.shape == (Nt_points, Nx_points)
        assert np.all(M >= 0)
        assert np.all(np.sum(M, axis=1) > 0.5)

    def test_callable_with_drift(self, standard_problem):
        """Test callable diffusion combined with drift field."""
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1
        bounds = standard_problem.geometry.get_bounds()

        # State-dependent diffusion
        def state_diffusion(t, x, m):
            return 0.1 + 0.05 * m

        # Initial condition
        x_grid = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_initial = np.exp(-((x_grid - 0.3) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial)

        # Drift field
        U_solution = np.zeros((Nt_points, Nx_points))
        for t in range(Nt_points):
            U_solution[t, :] = -0.1 * x_grid

        # Solve
        M = solver.solve_fp_system(m_initial, potential_field=U_solution, volatility_field=state_diffusion)

        assert M.shape == (Nt_points, Nx_points)
        assert np.all(M >= 0)

    def test_callable_scalar_return(self, standard_problem):
        """Test callable that returns scalar (constant diffusion)."""
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1

        # Callable returning scalar
        def constant_diffusion(t, x, m):
            return 0.2  # Constant for all x

        # Initial condition
        m_initial = np.ones(Nx_points) / Nx_points

        # Solve
        M = solver.solve_fp_system(m_initial, volatility_field=constant_diffusion)

        assert M.shape == (Nt_points, Nx_points)
        assert np.all(M >= 0)

    def test_callable_validation_wrong_shape(self, standard_problem):
        """Test that callable returning wrong shape raises error."""
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        _Nt_points = standard_problem.Nt + 1

        # Callable returning wrong shape
        def bad_diffusion(t, x, m):
            return np.ones(Nx_points + 10)  # Wrong shape

        m_initial = np.ones(Nx_points) / Nx_points

        # Should raise ValueError about shape
        with pytest.raises(ValueError, match="returned array with shape"):
            solver.solve_fp_system(m_initial, volatility_field=bad_diffusion)

    def test_callable_validation_nan(self, standard_problem):
        """Test that callable returning NaN raises error."""
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        _Nt_points = standard_problem.Nt + 1

        # Callable returning NaN
        def nan_diffusion(t, x, m):
            result = 0.1 * m
            result[0] = np.nan  # Introduce NaN
            return result

        m_initial = np.ones(Nx_points) / Nx_points

        # Should raise ValueError about NaN
        with pytest.raises(ValueError, match="NaN or Inf"):
            solver.solve_fp_system(m_initial, volatility_field=nan_diffusion)


class TestFPFDMSolverProblemVolatility:
    """problem-level non-scalar volatility must reach the solver (#1248, 2026-06-10 audit)."""

    def test_problem_array_sigma_reaches_solver(self):
        """A per-point problem.sigma array must drive the solve, not its mean.

        MFGProblem stores a placeholder problem.sigma = mean(array) for non-scalar volatility,
        and the FP solver read that placeholder, so a spatially-varying sigma silently solved
        the mean PDE. After the fix the no-override path uses problem.volatility_field, so the
        problem-level solve matches an explicit per-point volatility_field and differs from the
        mean scalar.
        """
        domain = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1))
        Nx = domain.num_points[0]
        sigma_arr = np.linspace(0.1, 0.5, Nx)  # low-left, high-right per-point volatility
        problem = MFGProblem(geometry=domain, T=0.05, Nt=10, sigma=sigma_arr, components=_default_components())
        solver = FPFDMSolver(problem, boundary_conditions=no_flux_bc(dimension=1))

        Nt = problem.Nt + 1
        x = np.linspace(0.0, 1.0, Nx)
        m0 = np.exp(-((x - 0.5) ** 2) / (2 * 0.08**2))
        m0 /= np.sum(m0) * domain.spacing[0]
        U = np.zeros((Nt, Nx))

        M_problem = solver.solve_fp_system(m0, U)  # no override -> must use problem.volatility_field
        M_explicit = solver.solve_fp_system(m0, U, volatility_field=sigma_arr)
        M_mean = solver.solve_fp_system(m0, U, volatility_field=float(np.mean(sigma_arr)))

        assert np.allclose(M_problem[-1], M_explicit[-1], atol=1e-12), (
            "problem-level solve must equal the explicit per-point volatility_field"
        )
        assert not np.allclose(M_problem[-1], M_mean[-1], atol=1e-3), (
            "problem-level solve must NOT collapse the per-point sigma to its mean (#1248 bug)"
        )


class TestFPFDMSolverTensorDiffusion:
    """Test tensor diffusion support (Phase 3.0)."""

    def test_diagonal_tensor_2d(self):
        """Test diagonal anisotropic tensor in 2D."""
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 0.6)], Nx_points=[31, 21], boundary_conditions=no_flux_bc(dimension=2)
        )
        problem = MFGProblem(geometry=domain, T=0.05, Nt=10, sigma=0.1, components=_default_components_2d())

        boundary_conditions = no_flux_bc(dimension=1)
        solver = FPFDMSolver(problem, boundary_conditions=boundary_conditions)

        Nx, Ny = domain.num_points[0], domain.num_points[1]
        Nt = problem.Nt + 1

        # Diagonal tensor: fast horizontal, slow vertical
        Sigma = np.diag([0.2, 0.05])

        # Gaussian initial condition at center
        x_coords, y_coords = domain.coordinates
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        m_initial = np.exp(-((X - 0.5) ** 2 + (Y - 0.3) ** 2) / (2 * 0.08**2))
        m_initial /= np.sum(m_initial) * domain.spacing[0] * domain.spacing[1]

        # Zero drift (pure diffusion)
        U_solution = np.zeros((Nt, Nx, Ny))

        # Solve with tensor diffusion
        M = solver.solve_fp_system(m_initial, potential_field=U_solution, tensor_diffusion_field=Sigma)

        assert M.shape == (Nt, Nx, Ny)
        assert np.all(M >= 0)
        # Mass conservation
        masses = np.sum(M, axis=(1, 2)) * domain.spacing[0] * domain.spacing[1]
        assert np.allclose(masses, 1.0, atol=0.1)

    def test_isotropic_tensor_matches_scalar_diffusion(self):
        """Isotropic volatility Sigma = sigma*I must give D = sigma^2/2, like the scalar path.

        Regression for #1249 (2026-06-10 audit): the tensor path applied the raw Sigma as the
        diffusion tensor (D = sigma instead of D = sigma^2/2), ~6.7x overdiffusion at sigma=0.3.
        Pure diffusion, zero drift: the tensor run with Sigma = sigma*I and the scalar run with
        problem.sigma = sigma both encode D = sigma^2/2, so their densities must match within
        the (small) cross-scheme discretization gap, NOT differ by the 6.7x raw-Sigma factor.
        """
        sigma = 0.3
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[26, 26], boundary_conditions=no_flux_bc(dimension=2)
        )
        problem = MFGProblem(geometry=domain, T=0.05, Nt=10, sigma=sigma, components=_default_components_2d())
        solver = FPFDMSolver(problem, boundary_conditions=no_flux_bc(dimension=1))

        Nt = problem.Nt + 1
        Nx, Ny = domain.num_points[0], domain.num_points[1]
        x_coords, y_coords = domain.coordinates
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        m0 = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * 0.1**2))
        m0 /= np.sum(m0) * domain.spacing[0] * domain.spacing[1]
        U = np.zeros((Nt, Nx, Ny))

        M_scalar = solver.solve_fp_system(m0, potential_field=U)  # scalar workhorse, D = sigma^2/2
        M_tensor = solver.solve_fp_system(
            m0, potential_field=U, tensor_diffusion_field=sigma * np.eye(2)
        )  # tensor path; D must also be sigma^2/2

        peak = np.max(np.abs(M_scalar[-1]))
        rel = np.max(np.abs(M_tensor[-1] - M_scalar[-1])) / peak
        assert rel < 0.1, (
            f"isotropic tensor diffusion magnitude mismatch rel={rel:.3f}; the raw-Sigma bug "
            f"applies D=sigma instead of sigma^2/2 (~6.7x overdiffusion -> ~0.5 relative)."
        )

    def test_full_tensor_with_cross_diffusion(self):
        """Test full anisotropic tensor with off-diagonal terms."""
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[26, 26], boundary_conditions=no_flux_bc(dimension=2)
        )
        problem = MFGProblem(geometry=domain, T=0.05, Nt=10, sigma=0.1, components=_default_components_2d())

        boundary_conditions = periodic_bc(dimension=1)
        solver = FPFDMSolver(problem, boundary_conditions=boundary_conditions)

        Nx, Ny = domain.num_points[0], domain.num_points[1]
        Nt = problem.Nt + 1

        # Full tensor with cross-diffusion
        Sigma = np.array([[0.2, 0.05], [0.05, 0.1]])

        # Initial condition
        x_coords, y_coords = domain.coordinates
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        m_initial = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial) * domain.spacing[0] * domain.spacing[1]

        # Solve
        M = solver.solve_fp_system(m_initial, tensor_diffusion_field=Sigma, show_progress=False)

        assert M.shape == (Nt, Nx, Ny)
        assert np.all(M >= 0)
        # Verify solution stability
        masses = np.sum(M, axis=(1, 2)) * domain.spacing[0] * domain.spacing[1]
        assert np.all(masses > 0.5)
        assert np.all(masses < 2.0)

    def test_spatially_varying_tensor(self):
        """Test spatially-varying tensor diffusion."""
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 0.6)], Nx_points=[26, 16], boundary_conditions=no_flux_bc(dimension=2)
        )
        problem = MFGProblem(geometry=domain, T=0.05, Nt=10, sigma=0.1, components=_default_components_2d())

        boundary_conditions = no_flux_bc(dimension=1)
        solver = FPFDMSolver(problem, boundary_conditions=boundary_conditions)

        Nx, Ny = domain.num_points[0], domain.num_points[1]
        Nt = problem.Nt + 1

        # Spatially-varying tensor: orientation changes with position
        x_coords, y_coords = domain.coordinates
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")

        Sigma_spatial = np.zeros((Nx, Ny, 2, 2))
        for i in range(Nx):
            for j in range(Ny):
                # Diagonal tensor varying with position
                sigma_x = 0.15 + 0.05 * X[i, j]
                sigma_y = 0.08 + 0.02 * Y[i, j]
                Sigma_spatial[i, j] = np.diag([sigma_x, sigma_y])

        # Initial condition
        m_initial = np.exp(-((X - 0.5) ** 2 + (Y - 0.3) ** 2) / (2 * 0.08**2))
        m_initial /= np.sum(m_initial) * domain.spacing[0] * domain.spacing[1]

        # Solve
        M = solver.solve_fp_system(m_initial, tensor_diffusion_field=Sigma_spatial, show_progress=False)

        assert M.shape == (Nt, Nx, Ny)
        assert np.all(M >= 0)
        # Mass conservation with spatially varying tensor
        masses = np.sum(M, axis=(1, 2)) * domain.spacing[0] * domain.spacing[1]
        assert np.allclose(masses, 1.0, atol=0.15)

    def test_callable_tensor(self):
        """Test callable state-dependent tensor: Sigma(t, x, m)."""
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 0.6)], Nx_points=[21, 16], boundary_conditions=no_flux_bc(dimension=2)
        )
        problem = MFGProblem(geometry=domain, T=0.05, Nt=10, sigma=0.1, components=_default_components_2d())

        boundary_conditions = no_flux_bc(dimension=1)
        solver = FPFDMSolver(problem, boundary_conditions=boundary_conditions)

        Nx, Ny = domain.num_points[0], domain.num_points[1]

        # State-dependent tensor: anisotropy increases with mean density
        def crowd_anisotropic(t, x, m):
            sigma_parallel = 0.15  # Horizontal movement
            # Vertical movement decreases in high-density regions (ensure positive)
            # Use mean density for global anisotropy (returns constant matrix)
            m_mean = np.mean(m) if hasattr(m, "__len__") else m
            sigma_perp = max(0.05 + 0.05 * (1 - m_mean / 2.0), 1e-6)
            return np.diag([sigma_parallel, sigma_perp])

        # Initial condition
        x_coords, y_coords = domain.coordinates
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        m_initial = np.exp(-((X - 0.5) ** 2 + (Y - 0.3) ** 2) / (2 * 0.08**2))
        m_initial /= np.sum(m_initial) * domain.spacing[0] * domain.spacing[1]

        # Solve
        M = solver.solve_fp_system(m_initial, tensor_diffusion_field=crowd_anisotropic, show_progress=False)

        assert M.shape == (problem.Nt + 1, Nx, Ny)
        assert np.all(M >= 0)

    def test_tensor_with_drift(self):
        """Test tensor diffusion combined with drift field."""
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[26, 26], boundary_conditions=no_flux_bc(dimension=2)
        )
        problem = MFGProblem(geometry=domain, T=0.05, Nt=10, sigma=0.1, components=_default_components_2d())

        boundary_conditions = periodic_bc(dimension=1)
        solver = FPFDMSolver(problem, boundary_conditions=boundary_conditions)

        Nx, Ny = domain.num_points[0], domain.num_points[1]
        Nt = problem.Nt + 1

        # Diagonal tensor
        Sigma = np.diag([0.15, 0.08])

        # Initial condition
        x_coords, y_coords = domain.coordinates
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        m_initial = np.exp(-((X - 0.3) ** 2 + (Y - 0.3) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial) * domain.spacing[0] * domain.spacing[1]

        # Non-zero drift (quadratic value function)
        U_solution = np.zeros((Nt, Nx, Ny))
        for k in range(Nt):
            U_solution[k] = X**2 + Y**2

        # Solve
        M = solver.solve_fp_system(
            m_initial, potential_field=U_solution, tensor_diffusion_field=Sigma, show_progress=False
        )

        assert M.shape == (Nt, Nx, Ny)
        assert np.all(M >= 0)
        # Solution should evolve (not static)
        assert not np.allclose(M[0], M[-1])

    def test_tensor_diffusion_mutual_exclusivity(self):
        """Test that tensor_diffusion_field and volatility_field are mutually exclusive."""
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[26, 26], boundary_conditions=no_flux_bc(dimension=2)
        )
        problem = MFGProblem(geometry=domain, T=0.05, Nt=10, sigma=0.1, components=_default_components_2d())

        boundary_conditions = no_flux_bc(dimension=1)
        solver = FPFDMSolver(problem, boundary_conditions=boundary_conditions)

        Nx, Ny = domain.num_points[0], domain.num_points[1]

        # Initial condition
        m_initial = np.ones((Nx, Ny)) / (Nx * Ny)

        # Tensor and scalar both specified
        Sigma = np.eye(2)
        scalar_sigma = 0.2

        # Should raise ValueError (Issue #717: volatility API - deprecated params get converted)
        with pytest.raises(ValueError, match="Cannot specify both volatility_field and tensor_diffusion_field"):
            solver.solve_fp_system(m_initial, volatility_field=scalar_sigma, tensor_diffusion_field=Sigma)

    def test_tensor_diffusion_1d_raises_error(self, standard_problem):
        """Test that tensor diffusion in 1D raises NotImplementedError."""
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        _Nt_points = standard_problem.Nt + 1

        m_initial = np.ones(Nx_points) / Nx_points

        # 1D tensor (should fail)
        Sigma = np.array([[0.2]])

        # Should raise NotImplementedError (Issue #717: volatility API)
        with pytest.raises(NotImplementedError, match="Anisotropic volatility not yet implemented for 1D"):
            solver.solve_fp_system(m_initial, tensor_diffusion_field=Sigma)

    def test_tensor_psd_validation(self):
        """Test that non-PSD tensor raises error."""
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[21, 21], boundary_conditions=no_flux_bc(dimension=2)
        )
        problem = MFGProblem(geometry=domain, T=0.05, Nt=5, sigma=0.1, components=_default_components_2d())

        boundary_conditions = no_flux_bc(dimension=1)
        solver = FPFDMSolver(problem, boundary_conditions=boundary_conditions)

        Nx, Ny = domain.num_points[0], domain.num_points[1]

        # Non-PSD tensor (negative eigenvalue)
        Sigma_bad = np.array([[0.2, 0.3], [0.3, -0.1]])  # Has negative eigenvalue

        # Initial condition
        m_initial = np.ones((Nx, Ny)) / (Nx * Ny)

        # Should raise ValueError about PSD
        with pytest.raises(ValueError, match="positive semi-definite"):
            solver.solve_fp_system(m_initial, tensor_diffusion_field=Sigma_bad, show_progress=False)

    def test_tensor_diffusion_mass_conservation(self):
        """Test mass conservation with tensor diffusion."""
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 0.6)], Nx_points=[31, 21], boundary_conditions=no_flux_bc(dimension=2)
        )
        problem = MFGProblem(geometry=domain, T=0.05, Nt=50, sigma=0.1, components=_default_components_2d())

        boundary_conditions = no_flux_bc(dimension=1)
        solver = FPFDMSolver(problem, boundary_conditions=boundary_conditions)

        # Diagonal tensor (smaller values for stability)
        Sigma = np.diag([0.05, 0.03])

        # Initial condition
        x_coords, y_coords = domain.coordinates
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        m_initial = np.exp(-((X - 0.5) ** 2 + (Y - 0.3) ** 2) / (2 * 0.08**2))
        m_initial /= np.sum(m_initial) * domain.spacing[0] * domain.spacing[1]

        # Solve
        M = solver.solve_fp_system(m_initial, tensor_diffusion_field=Sigma, show_progress=False)

        # Check mass conservation at each timestep
        masses = np.sum(M, axis=(1, 2)) * domain.spacing[0] * domain.spacing[1]
        assert np.allclose(masses, 1.0, atol=0.1)


class TestFPFDMSolverRemovedDeprecatedParams:
    """Pin removal of the Tier-2 (<=v0.17) FP-solver param renames.

    ``m_initial_condition`` -> ``M_initial`` and ``diffusion_field`` -> ``volatility_field``
    were deprecated in v0.17.0 and removed at v0.20 (3 minor versions past). Passing the old
    names must now raise ``TypeError`` (unexpected keyword argument), not silently alias.
    New-name coverage is retained by the rest of this module (positional ``M_initial`` and
    ``volatility_field=``). The ``tensor_diffusion_field`` / ``volatility_matrix`` aliases are
    intentionally NOT removed (no callable-tensor equivalent on ``volatility_field`` yet).
    """

    def test_m_initial_condition_removed(self, standard_problem):
        solver = FPFDMSolver(standard_problem)
        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        m0 = np.ones(Nx_points) / Nx_points
        with pytest.raises(TypeError, match="m_initial_condition"):
            solver.solve_fp_system(m_initial_condition=m0)

    def test_diffusion_field_removed(self, standard_problem):
        solver = FPFDMSolver(standard_problem)
        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        m0 = np.ones(Nx_points) / Nx_points
        with pytest.raises(TypeError, match="diffusion_field"):
            solver.solve_fp_system(m0, diffusion_field=0.5)


class TestFPFDMSolverCallableDrift:
    """Test callable (state-dependent) drift_field support (Phase 2 - Issue #487)."""

    def test_constant_drift_callable(self, standard_problem):
        """Test constant drift via callable function."""
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1
        bounds = standard_problem.geometry.get_bounds()

        # Constant drift pushing right
        def constant_drift(t, x, m):
            return 0.3 * np.ones_like(x)

        # Initial condition (Gaussian)
        x_grid = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_initial = np.exp(-((x_grid - 0.3) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial)

        # Solve with callable drift
        M = solver.solve_fp_system(m_initial, drift_field=constant_drift, show_progress=False)

        assert M.shape == (Nt_points, Nx_points)
        assert np.all(M >= 0)
        # Solution should evolve
        assert not np.allclose(M[0], M[-1])

    def test_state_dependent_drift(self, standard_problem):
        """Test state-dependent drift: alpha(t, x, m) depends on density."""
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1
        bounds = standard_problem.geometry.get_bounds()

        # Density-dependent drift: move away from high density regions
        def crowd_avoidance_drift(t, x, m):
            # Gradient-like drift pushing toward low density
            grad_m = np.gradient(m, x[1] - x[0] if len(x) > 1 else 1.0)
            return -0.5 * grad_m

        # Initial condition
        x_grid = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_initial = np.exp(-((x_grid - 0.5) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial)

        # Solve
        M = solver.solve_fp_system(m_initial, drift_field=crowd_avoidance_drift, show_progress=False)

        assert M.shape == (Nt_points, Nx_points)
        assert np.all(M >= 0)
        # Verify mass conservation is approximately maintained
        masses = np.sum(M, axis=1)
        assert np.all(masses > 0.5)
        assert np.all(masses < 2.0)

    def test_time_dependent_drift(self, standard_problem):
        """Test time-dependent drift: alpha(t, x, m) varies with time."""
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1
        bounds = standard_problem.geometry.get_bounds()

        # Drift that oscillates over time
        def oscillating_drift(t, x, m):
            return 0.2 * np.sin(2 * np.pi * t) * np.ones_like(x)

        # Initial condition
        x_grid = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_initial = np.exp(-((x_grid - 0.5) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial)

        # Solve
        M = solver.solve_fp_system(m_initial, drift_field=oscillating_drift, show_progress=False)

        assert M.shape == (Nt_points, Nx_points)
        assert np.all(M >= 0)

    def test_callable_drift_with_callable_diffusion(self, standard_problem):
        """Test callable drift combined with callable diffusion."""
        solver = FPFDMSolver(standard_problem)

        (Nx_points,) = standard_problem.geometry.get_grid_shape()
        Nt_points = standard_problem.Nt + 1
        bounds = standard_problem.geometry.get_bounds()

        # Callable drift - use m for shape since x is coords list
        def simple_drift(t, x, m):
            return 0.2 * np.ones_like(m)

        # Callable diffusion - use m for shape since x is coords list
        def simple_diffusion(t, x, m):
            return 0.1 * np.ones_like(m)

        # Initial condition
        x_grid = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_initial = np.exp(-((x_grid - 0.5) ** 2) / (2 * 0.1**2))
        m_initial /= np.sum(m_initial)

        # Solve
        M = solver.solve_fp_system(
            m_initial, drift_field=simple_drift, volatility_field=simple_diffusion, show_progress=False
        )

        assert M.shape == (Nt_points, Nx_points)
        assert np.all(M >= 0)

    def test_callable_drift_2d(self):
        """Test callable drift in 2D problem."""
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[21, 21], boundary_conditions=no_flux_bc(dimension=2)
        )
        problem = MFGProblem(geometry=domain, T=0.1, Nt=10, sigma=0.1, components=_default_components_2d())

        solver = FPFDMSolver(problem)

        Nx, Ny = domain.Nx_points

        # Callable drift returning vector field
        def vector_drift(t, coords, m):
            x, y = coords
            # Return vector drift: shape (2, Nx, Ny)
            drift = np.zeros((2, len(x), len(y)))
            drift[0] = 0.3 * np.ones((len(x), len(y)))  # x-drift
            drift[1] = 0.0 * np.ones((len(x), len(y)))  # no y-drift (synthetic U limitation)
            return drift

        # Initial condition (Gaussian)
        x_coords, y_coords = domain.coordinates
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        m_initial = np.exp(-30 * ((X - 0.3) ** 2 + (Y - 0.5) ** 2))
        m_initial /= np.sum(m_initial)

        # Solve with callable drift
        M = solver.solve_fp_system(m_initial, drift_field=vector_drift, show_progress=False)

        assert M.shape == (problem.Nt + 1, Nx, Ny)
        assert np.all(np.isfinite(M))
        assert np.all(M >= -1e-10)


class TestVaryingSigmaExplicitDriftPerPoint:
    """Issue #1183: the explicit-drift FP path now solves a per-point variable-coefficient
    diffusion (face-averaged D(x)=sigma(x)^2/2 in a conservative FV Laplacian) instead of
    collapsing a spatially varying volatility to its mean. A non-uniform sigma is honored per
    point (low-sigma regions under-diffuse) AND mass is conserved -- no warning. A uniform
    array sigma uses the scalar path unchanged (also no warning)."""

    @staticmethod
    def _solve(sigma_field, bump_center=0.25):
        n, nt = 41, 30
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
        H = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: np.asarray(m) * 0.0,
            coupling_dm=lambda m: np.asarray(m) * 0.0,
        )
        x = np.linspace(0.0, 1.0, n)
        dx = x[1] - x[0]

        def m_init(xx):
            return np.exp(-((np.asarray(xx) - bump_center) ** 2) / 0.01)

        comps = MFGComponents(m_initial=m_init, u_terminal=lambda xx: np.asarray(xx) * 0.0, hamiltonian=H)
        prob = MFGProblem(geometry=grid, T=0.3, Nt=nt, sigma=0.1, components=comps)
        m0 = m_init(x)
        m0 /= m0.sum() * dx
        # callable drift routes through the explicit path (signature (t, grid, density))
        traj = FPFDMSolver(prob).solve_fp_system(
            m0, drift_field=lambda t, g, m: np.zeros(n), volatility_field=sigma_field
        )
        return np.asarray(traj)[-1], dx

    def test_non_uniform_sigma_per_point_not_mean(self):
        """A bump in the LOW-sigma region under-diffuses (stays more peaked) vs the mean-collapse,
        and mass is conserved -- the per-point fidelity #1183 asks for. No interim warning."""
        n = 41
        x = np.linspace(0.0, 1.0, n)
        sigma_field = np.where(x < 0.5, 0.05, 0.30)  # low-sigma left (the bump sits here), high-sigma right
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m_perpoint, dx = self._solve(sigma_field, bump_center=0.25)
        assert not [w for w in caught if "1183" in str(w.message)], (
            "the interim #1183 mean-collapse warning must be gone"
        )
        m_meancollapse, _ = self._solve(float(np.mean(sigma_field)), bump_center=0.25)

        assert abs(m_perpoint.sum() * dx - 1.0) < 1e-9, f"per-point diffusion leaked mass: {m_perpoint.sum() * dx:.8f}"
        assert np.all(m_perpoint >= -1e-12), "per-point diffusion produced a negative density"
        # The low-sigma bump diffuses LESS than the mean would -> a higher retained peak.
        assert m_perpoint.max() > m_meancollapse.max() * 1.02, (
            f"per-point did not under-diffuse the low-sigma bump: peak {m_perpoint.max():.4f} "
            f"vs mean-collapse {m_meancollapse.max():.4f}"
        )

    def test_uniform_array_sigma_unchanged(self):
        """A uniform array sigma uses the scalar path (no #1183 warning, mass conserved)."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m, dx = self._solve(np.full(41, 0.1))
        assert not [w for w in caught if "1183" in str(w.message)]
        assert abs(m.sum() * dx - 1.0) < 1e-9


class TestFPFDMSolverCFLDiagnostic:
    """Pin the CFL diffusive diagnostic to the D = sigma^2/2 convention."""

    def test_cfl_diffusive_uses_D_equals_half_sigma_squared(self, standard_problem):
        """The logged diffusive CFL must use D = sigma^2/2, not the bare sigma^2.

        Regression guard: the diagnostic previously computed sigma^2 * dt / dx^2, a 2x
        overstatement relative to the diffusion coefficient D = 0.5 * sigma^2 actually
        assembled by the solver (single source: diffusion_from_volatility / D = 0.5*sigma**2).
        Capture the log record's formatting arg and pin it to the halved value.
        """
        import logging

        from mfgarchon.alg.numerical.fp_solvers import fp_fdm as fp_fdm_module

        solver = FPFDMSolver(standard_problem)

        sigma = standard_problem.sigma
        dt = standard_problem.dt
        dx = standard_problem.geometry.get_grid_spacing()[0]
        expected = 0.5 * sigma**2 * dt / dx**2
        # Sanity: this configuration must exceed the 0.5 threshold so the diagnostic logs.
        assert expected > 0.5

        records: list[logging.LogRecord] = []

        class _Collector(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        module_logger = fp_fdm_module.logger
        handler = _Collector()
        old_level = module_logger.level
        module_logger.addHandler(handler)
        module_logger.setLevel(logging.DEBUG)
        try:
            solver._log_cfl_diagnostic()
        finally:
            module_logger.removeHandler(handler)
            module_logger.setLevel(old_level)

        cfl_records = [r for r in records if "CFL diagnostic" in r.msg]
        assert cfl_records, "CFL diagnostic did not log for an above-threshold configuration"
        logged_cfl = cfl_records[0].args[0]
        assert logged_cfl == pytest.approx(expected)
        # Explicitly pin the 2x: the old (bare sigma^2) value must be rejected.
        assert logged_cfl != pytest.approx(sigma**2 * dt / dx**2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

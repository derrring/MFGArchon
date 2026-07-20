#!/usr/bin/env python3
"""
Unit tests for HJBSemiLagrangianSolver.

Tests the semi-Lagrangian method for solving Hamilton-Jacobi-Bellman equations
in Mean Field Games, including characteristic-following schemes and interpolation.
"""

import warnings

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _default_hamiltonian():
    """Default Hamiltonian for testing (Issue #670: explicit specification required)."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components():
    """Default MFGComponents for testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),  # Gaussian centered at 0.5
        u_terminal=lambda x: 0.0,  # Zero terminal cost
        hamiltonian=_default_hamiltonian(),
    )


class TestHJBSemiLagrangianInitialization:
    """Test HJBSemiLagrangianSolver initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic solver initialization with default parameters."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem)

        assert solver.hjb_method_name == "Semi-Lagrangian"
        assert solver.interpolation_method == "linear"
        assert solver.optimization_method == "brent"
        assert solver.characteristic_solver == "explicit_euler"
        assert solver.tolerance == 1e-8

    def test_custom_interpolation_method(self):
        """Test initialization with custom interpolation method."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, interpolation_method="cubic")

        assert solver.interpolation_method == "cubic"

    def test_custom_optimization_method(self):
        """Test initialization with custom optimization method."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, optimization_method="golden")

        assert solver.optimization_method == "golden"

    def test_custom_characteristic_solver(self):
        """Test initialization with custom characteristic solver."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="rk2")

        assert solver.characteristic_solver == "rk2"

    def test_custom_tolerance(self):
        """Test initialization with custom tolerance."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, tolerance=1e-10)

        assert solver.tolerance == 1e-10

    def test_grid_parameters_computed(self):
        """Test that grid parameters are properly computed."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem)

        assert hasattr(solver, "x_grid")
        assert hasattr(solver, "dt")
        assert hasattr(solver, "dx")
        assert len(solver.x_grid) == problem.geometry.get_grid_shape()[0]
        assert np.isclose(solver.dt, problem.dt)
        assert np.isclose(solver.dx, problem.geometry.get_grid_spacing()[0])


class TestHJBSemiLagrangianSolveHJBSystem:
    """Test the main solve_hjb_system method."""

    def test_solve_hjb_system_shape(self):
        """Test that solve_hjb_system returns correct shape."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem)

        # Create inputs: Nx, Nt are intervals; knots = intervals + 1
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points))
        U_final = np.zeros(Nx_points)
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Output: same shape as input density (Nt+1 time points)
        assert U_solution.shape == (problem.Nt + 1, Nx_points)
        assert np.all(np.isfinite(U_solution))

    def test_solve_hjb_system_final_condition(self):
        """Test that final condition is preserved."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem)

        # Create inputs with specific final condition
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points))
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - bounds[1][0]) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Final time step should match final condition
        assert np.allclose(U_solution[-1, :], U_final, rtol=0.1)

    def test_solve_hjb_system_backward_propagation(self):
        """Test that solution propagates backward in time."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem)

        # Create inputs
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points))
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = x_coords**2  # Quadratic final condition
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Solution should propagate backward (values at earlier times should be influenced by final condition)
        # Check that solution at t=0 is different from zero
        assert not np.allclose(U_solution[0, :], 0.0)


class TestHJBSemiLagrangianNumericalProperties:
    """Test numerical properties of the semi-Lagrangian method."""

    def test_solution_finiteness(self):
        """Test that solution remains finite throughout."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=40, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem)

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) * 0.5
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = np.sin(2 * np.pi * x_coords)
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # All values should be finite
        assert np.all(np.isfinite(U_solution))

    def test_solution_smoothness(self):
        """Test that solution has reasonable smoothness."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem)

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points))
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # The solution should be no rougher than the terminal data it propagates back from.
        # A hardcoded bound does not express this: the previous 100.0 sat 9626x above the
        # measured value, so the solve had to degrade by four orders of magnitude to fail it.
        # Scaling to the terminal condition tracks the grid and stays discriminating.
        terminal_roughness = np.max(np.abs(np.diff(U_final)))
        solution_roughness = np.max(np.abs(np.diff(U_solution, axis=1)))
        assert solution_roughness < 10 * terminal_roughness, (
            f"solution roughness {solution_roughness:.3e} exceeds 10x the terminal data's "
            f"{terminal_roughness:.3e}; measured ratio on a healthy solve is 1.06"
        )


class TestHJBSemiLagrangianIntegration:
    """Integration tests with actual MFG problems."""

    def test_solver_with_uniform_density(self):
        """Test solver with uniform density distribution."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem)

        # Uniform density
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)

        # Simple final condition
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = (x_coords - 0.5) ** 2

        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Should produce valid solution
        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_solver_with_gaussian_density(self):
        """Test solver with Gaussian density distribution."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem)

        # Gaussian density
        bounds = problem.geometry.get_bounds()
        Nx_points = problem.geometry.get_grid_shape()[0]
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_profile = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        m_profile = m_profile / np.sum(m_profile)
        M_density = np.tile(m_profile, (problem.Nt + 1, 1))

        U_final = np.zeros(Nx_points)
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Should produce valid solution
        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)


class TestHJBSemiLagrangianSolverNotAbstract:
    """Test that HJBSemiLagrangianSolver is concrete (not abstract)."""

    def test_solver_not_abstract(self):
        """Test that HJBSemiLagrangianSolver can be instantiated."""
        import inspect

        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())

        # Should not raise TypeError about abstract methods
        solver = HJBSemiLagrangianSolver(problem)
        assert isinstance(solver, HJBSemiLagrangianSolver)

        # Should not have abstract methods
        assert not inspect.isabstract(HJBSemiLagrangianSolver)


class TestCharacteristicTracingMethods:
    """Test different characteristic tracing methods (explicit_euler, rk2, rk4)."""

    def test_explicit_euler_initialization(self):
        """Test that explicit_euler method initializes correctly."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="explicit_euler")

        assert solver.characteristic_solver == "explicit_euler"

    def test_rk2_initialization(self):
        """Test that rk2 method initializes correctly."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="rk2")

        assert solver.characteristic_solver == "rk2"

    def test_rk4_initialization(self):
        """Test that rk4 method initializes correctly."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="rk4")

        assert solver.characteristic_solver == "rk4"

    def test_euler_produces_valid_solution(self):
        """Test that explicit_euler produces valid solution."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="explicit_euler", use_jax=False)

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_rk2_produces_valid_solution(self):
        """Test that rk2 produces valid solution."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="rk2", use_jax=False)

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_rk4_produces_valid_solution(self):
        """Test that rk4 with scipy.solve_ivp produces valid solution."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="rk4", use_jax=False)

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_rk2_consistency_with_euler(self):
        """Test that rk2 produces consistent results with euler on smooth problems."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.2, Nt=20, components=_default_components())

        # Solve with euler
        solver_euler = HJBSemiLagrangianSolver(problem, characteristic_solver="explicit_euler", use_jax=False)
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))
        U_euler = solver_euler.solve_hjb_system(M_density, U_final, U_prev)

        # Solve with rk2
        solver_rk2 = HJBSemiLagrangianSolver(problem, characteristic_solver="rk2", use_jax=False)
        U_rk2 = solver_rk2.solve_hjb_system(M_density, U_final, U_prev)

        # On smooth problems with small dt, should be very similar
        rel_error = np.linalg.norm(U_rk2 - U_euler) / np.linalg.norm(U_euler)
        assert rel_error < 0.1  # Within 10%

    def test_rk4_consistency_with_euler(self):
        """Test that rk4 produces consistent results with euler on smooth problems."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.2, Nt=20, components=_default_components())

        # Solve with euler
        solver_euler = HJBSemiLagrangianSolver(problem, characteristic_solver="explicit_euler", use_jax=False)
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))
        U_euler = solver_euler.solve_hjb_system(M_density, U_final, U_prev)

        # Solve with rk4
        solver_rk4 = HJBSemiLagrangianSolver(problem, characteristic_solver="rk4", use_jax=False)
        U_rk4 = solver_rk4.solve_hjb_system(M_density, U_final, U_prev)

        # On smooth problems with small dt, should be similar
        rel_error = np.linalg.norm(U_rk4 - U_euler) / np.linalg.norm(U_euler)
        assert rel_error < 0.1  # Within 10%

    def test_trace_characteristic_backward_1d(self):
        """Test _trace_characteristic_backward method directly in 1D."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, characteristic_solver="rk4", use_jax=False)

        # Test characteristic tracing
        x_current = 0.5
        p_optimal = 0.1
        dt = 0.01

        x_departure = solver._trace_characteristic_backward(x_current, p_optimal, dt)

        # Should return a scalar
        assert isinstance(x_departure, (float, np.floating))
        # Should be finite
        assert np.isfinite(x_departure)
        # Should be within domain
        bounds = problem.geometry.get_bounds()
        assert bounds[0][0] <= x_departure <= bounds[1][0]


class TestInterpolationMethods:
    """Test different interpolation methods (linear, cubic, quintic)."""

    def test_linear_interpolation_initialization(self):
        """Test that linear interpolation initializes correctly."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, interpolation_method="linear")

        assert solver.interpolation_method == "linear"

    def test_cubic_interpolation_initialization(self):
        """Test that cubic interpolation initializes correctly."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, interpolation_method="cubic")

        assert solver.interpolation_method == "cubic"

    def test_cubic_produces_valid_solution_1d(self):
        """Test that cubic interpolation produces valid solution in 1D."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = HJBSemiLagrangianSolver(
            problem, interpolation_method="cubic", characteristic_solver="rk2", use_jax=False
        )

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_cubic_consistency_with_linear(self):
        """Test that cubic interpolation is consistent with linear on smooth problems."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=20, components=_default_components())

        # Solve with linear
        solver_linear = HJBSemiLagrangianSolver(
            problem, interpolation_method="linear", characteristic_solver="rk2", use_jax=False
        )
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))
        U_linear = solver_linear.solve_hjb_system(M_density, U_final, U_prev)

        # Solve with cubic
        solver_cubic = HJBSemiLagrangianSolver(
            problem, interpolation_method="cubic", characteristic_solver="rk2", use_jax=False
        )
        U_cubic = solver_cubic.solve_hjb_system(M_density, U_final, U_prev)

        # On smooth problems with fine grid, should be reasonably similar
        # Note: With gradient-based optimal control (Issue #298 fix), interpolation
        # method has more impact since characteristics now move correctly
        rel_error = np.linalg.norm(U_cubic - U_linear) / np.linalg.norm(U_linear)
        assert rel_error < 0.25  # Within 25% (updated after gradient fix)

    def test_cubic_improves_smoothness(self):
        """Cubic interpolation is not markedly rougher than linear, measured on second differences.

        Formerly ``@pytest.mark.xfail(reason="Cubic interpolation produces NaN values - see
        issue #583")``. #583 is CLOSED and the test passes, so the marker was stale; it had been
        XPASSing unnoticed because pytest.ini set no ``xfail_strict`` (Issue #1663).

        **Why it passes is not established.** An earlier version of this docstring asserted that
        #583's fix -- routing ``interpolation_method='cubic'`` through PCHIP rather than a natural
        cubic spline (``hjb_sl_interpolation.py:79-86``) -- was the reason. That is refuted by
        counterfactual: aliasing ``PchipInterpolator`` to ``scipy.interpolate.CubicSpline`` and
        re-running leaves this class **passing** (5 passed). PCHIP dispatch is real and
        deliberate, but it is not what makes this test green. #583's own remaining-work note
        attributes the NaNs to ``p**2`` overflow rather than to Runge oscillations. Do not
        substitute a mechanism here without testing it.

        The name is restored: a previous rename to ``..._produces_a_finite_solution`` described
        the body as asserting "finiteness and a shape, never smoothness, and never comparing
        against a non-cubic run". All three clauses were false -- the body builds a
        ``interpolation_method='linear'`` solver, computes ``mean|second difference|`` for both,
        and asserts ``smoothness_cubic < smoothness_linear * 2.0``. The original name was the
        more accurate of the two.

        What is fair to say about the assertion: the ``* 2.0`` slack makes it a
        not-much-worse check rather than an improves check.
        """
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=20, components=_default_components())

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        # Use steep gradients to test interpolation quality
        U_final = np.exp(-20 * (x_coords - 0.5) ** 2)
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        # Solve with linear
        solver_linear = HJBSemiLagrangianSolver(
            problem, interpolation_method="linear", characteristic_solver="rk2", use_jax=False
        )
        U_linear = solver_linear.solve_hjb_system(M_density, U_final, U_prev)

        # Solve with cubic
        solver_cubic = HJBSemiLagrangianSolver(
            problem, interpolation_method="cubic", characteristic_solver="rk2", use_jax=False
        )
        U_cubic = solver_cubic.solve_hjb_system(M_density, U_final, U_prev)

        # Measure smoothness via second derivative
        smoothness_linear = np.mean(np.abs(np.diff(U_linear, n=2, axis=1)))
        smoothness_cubic = np.mean(np.abs(np.diff(U_cubic, n=2, axis=1)))

        # Both should be finite
        assert np.isfinite(smoothness_linear)
        assert np.isfinite(smoothness_cubic)
        # Cubic should generally be smoother (smaller second derivatives)
        # This is not always true but should hold for most cases
        # We just check that cubic doesn't make things dramatically worse
        assert smoothness_cubic < smoothness_linear * 2.0


class TestRBFInterpolationFallback:
    """Test RBF interpolation fallback functionality."""

    def test_rbf_fallback_initialization_enabled(self):
        """Test that RBF fallback can be enabled."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, use_rbf_fallback=True, rbf_kernel="thin_plate_spline")

        assert solver.use_rbf_fallback is True
        assert solver.rbf_kernel == "thin_plate_spline"

    def test_rbf_fallback_initialization_disabled(self):
        """Test that RBF fallback can be disabled."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem, use_rbf_fallback=False)

        assert solver.use_rbf_fallback is False

    def test_rbf_kernel_options(self):
        """Test different RBF kernel options."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())

        kernels = ["thin_plate_spline", "multiquadric", "gaussian"]

        for kernel in kernels:
            solver = HJBSemiLagrangianSolver(problem, use_rbf_fallback=True, rbf_kernel=kernel)
            assert solver.rbf_kernel == kernel

    def test_enabling_rbf_fallback_does_not_break_the_solve(self):
        """Enabling ``use_rbf_fallback`` leaves an ordinary 1D solve finite -- and the flag is INERT here.

        Formerly ``@pytest.mark.xfail(reason="Numerical instability with RBF thin_plate_spline on
        steep gradients - see Issue #583")``, XPASSing silently for want of ``xfail_strict``
        (Issue #1663). That reason described a path this test never enters.

        Counter instrumentation over this exact configuration records
        ``interpolate_value_rbf_fallback`` invoked **0** times against ``interpolate_value_1d``
        **5177** times. Code-traced: ``hjb_semi_lagrangian.py`` returns ``interpolate_value_1d``
        unconditionally on the 1D branch, and the only RBF call site is in the nD ``else``. So in
        1D the flag is structurally inert -- this test would pass identically with
        ``use_rbf_fallback=False``, or with the RBF function deleted.

        Kept, narrowly, as a regression pin that constructing with the flag does not break the
        solve. It is largely redundant with ``test_rbf_consistency_with_no_fallback``, which at
        least compares two runs. That RBF has no test reaching it at all is Issue #1664.
        """
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = HJBSemiLagrangianSolver(
            problem, use_rbf_fallback=True, rbf_kernel="thin_plate_spline", characteristic_solver="rk2", use_jax=False
        )

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        # Use steep gradient to potentially trigger RBF fallback
        U_final = np.exp(-20 * (x_coords - 0.5) ** 2)
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_rbf_consistency_with_no_fallback(self):
        """Test that RBF fallback doesn't change results on well-behaved problems."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=20, components=_default_components())

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        # Solve without RBF
        solver_no_rbf = HJBSemiLagrangianSolver(
            problem, use_rbf_fallback=False, characteristic_solver="rk2", use_jax=False
        )
        U_no_rbf = solver_no_rbf.solve_hjb_system(M_density, U_final, U_prev)

        # Solve with RBF
        solver_rbf = HJBSemiLagrangianSolver(
            problem, use_rbf_fallback=True, rbf_kernel="thin_plate_spline", characteristic_solver="rk2", use_jax=False
        )
        U_rbf = solver_rbf.solve_hjb_system(M_density, U_final, U_prev)

        # On well-behaved problems, RBF fallback shouldn't trigger
        # Results should be identical or very close
        rel_error = np.linalg.norm(U_rbf - U_no_rbf) / np.linalg.norm(U_no_rbf)
        assert rel_error < 1e-10  # Should be machine precision


class TestEnhancementsIntegration:
    """Test combinations of enhancements working together."""

    def test_rk4_with_cubic_interpolation(self):
        """Test RK4 characteristic tracing with cubic interpolation."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = HJBSemiLagrangianSolver(
            problem, characteristic_solver="rk4", interpolation_method="cubic", use_jax=False
        )

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_rk4_with_rbf_fallback(self):
        """Test RK4 characteristic tracing with RBF fallback."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = HJBSemiLagrangianSolver(
            problem, characteristic_solver="rk4", use_rbf_fallback=True, rbf_kernel="thin_plate_spline", use_jax=False
        )

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_all_enhancements_together(self):
        """Test all enhancements working together: RK4 + cubic + RBF."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        solver = HJBSemiLagrangianSolver(
            problem,
            characteristic_solver="rk4",
            interpolation_method="cubic",
            use_rbf_fallback=True,
            rbf_kernel="thin_plate_spline",
            use_jax=False,
        )

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (problem.Nt + 1, Nx_points)

    def test_enhanced_vs_baseline_consistency(self):
        """Test that enhanced configuration produces consistent results with baseline."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=20, components=_default_components())

        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((problem.Nt + 1, Nx_points)) / (Nx_points - 1)
        bounds = problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((problem.Nt + 1, Nx_points))

        # Baseline configuration
        solver_baseline = HJBSemiLagrangianSolver(
            problem,
            characteristic_solver="explicit_euler",
            interpolation_method="linear",
            use_rbf_fallback=False,
            use_jax=False,
        )
        U_baseline = solver_baseline.solve_hjb_system(M_density, U_final, U_prev)

        # Enhanced configuration
        solver_enhanced = HJBSemiLagrangianSolver(
            problem,
            characteristic_solver="rk4",
            interpolation_method="cubic",
            use_rbf_fallback=True,
            rbf_kernel="thin_plate_spline",
            use_jax=False,
        )
        U_enhanced = solver_enhanced.solve_hjb_system(M_density, U_final, U_prev)

        # On smooth problems with fine grid, should be reasonably consistent
        # Note: With gradient-based optimal control (Issue #298 fix), method differences
        # are more pronounced since characteristics now move correctly
        rel_error = np.linalg.norm(U_enhanced - U_baseline) / np.linalg.norm(U_baseline)
        assert rel_error < 0.20  # Within 20% (updated after gradient fix)


class TestStochasticCharacteristicSL:
    """Issue #1026: Carlini-Silva (2014) stochastic-characteristic SL.

    Tests the diffusion_method="stochastic" branch that incorporates the
    diffusion term into the SL update via 2*d Brownian departure points,
    instead of the operator-splitting (ADI/Crank-Nicolson) default.

    Validation experiment: mfg-research/experiments/crowd_evacuation_2d/
    minors/archive/exp14_towel_1d_benchmark/subs/exp14e_solver_comparison/
    """

    def test_linear_plus_stochastic_accepted(self):
        """Issue #1049: linear+stochastic IS the canonical Carlini-Silva 2014 scheme.

        Previously rejected by validation (`test_linear_plus_stochastic_rejected`).
        That validation was inverted from CS 2014's stability requirement: the
        rejected combination IS the proven-stable canonical scheme, while the
        forced cubic combination is non-monotone (Issue #1033). Test renamed and
        inverted to assert the corrected behavior.
        """
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)],
            Nx_points=[51],
            boundary_conditions=no_flux_bc(dimension=1),
        )
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())

        # Should NOT raise — linear+stochastic is now allowed and recommended.
        solver = HJBSemiLagrangianSolver(
            problem,
            interpolation_method="linear",
            diffusion_method="stochastic",
        )
        assert solver.diffusion_method == "stochastic"
        assert solver.interpolation_method == "linear"

    def test_cubic_plus_stochastic_warns(self):
        """Issue #1049: cubic+stochastic emits a UserWarning (CS 2014 proof doesn't apply)."""
        import warnings as _w

        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)],
            Nx_points=[51],
            boundary_conditions=no_flux_bc(dimension=1),
        )
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            solver = HJBSemiLagrangianSolver(
                problem,
                interpolation_method="cubic",
                diffusion_method="stochastic",
                check_cfl=False,
            )
            cs_warnings = [m for m in caught if "Carlini-Silva" in str(m.message)]

        assert len(cs_warnings) == 1, f"expected 1 CS UserWarning, got {len(cs_warnings)}"
        assert solver.diffusion_method == "stochastic"
        assert solver.interpolation_method == "cubic"

    def test_apply_diffusion_raises_under_stochastic(self):
        """Reaching _apply_diffusion under stochastic dispatch is a programming error."""
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)],
            Nx_points=[51],
            boundary_conditions=no_flux_bc(dimension=1),
        )
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBSemiLagrangianSolver(
            problem,
            interpolation_method="cubic",
            diffusion_method="stochastic",
            check_cfl=False,
        )

        with pytest.raises(NotImplementedError, match="should not be called"):
            solver._apply_diffusion(np.zeros(51), 0.01)

    def test_constant_terminal_preserved(self):
        """H=0 with constant U_T must give constant U[0] (no spurious drift)."""
        from mfgarchon.core.hamiltonian import HamiltonianBase, OptimizationSense

        class ZeroH(HamiltonianBase):
            def __init__(self):
                super().__init__(sense=OptimizationSense.MINIMIZE)

            def __call__(self, x, m, p, t=0.0):
                p_arr = np.atleast_1d(np.asarray(p, dtype=float))
                if p_arr.ndim > 0:
                    return np.zeros(p_arr.shape[:-1])
                return 0.0

            def gradient_p(self, x, m, p, t=0.0):
                return np.zeros_like(np.asarray(p, dtype=float))

            def density_derivative(self, x, m, p, t=0.0):
                return 0.0

        geometry = TensorProductGrid(
            dimension=1,
            bounds=[(-1.0, 1.0)],
            Nx_points=[31],
            boundary_conditions=no_flux_bc(dimension=1),
        )
        components = MFGComponents(
            hamiltonian=ZeroH(),
            m_initial=lambda x: 1.0,
            u_terminal=lambda x: 1.0,
        )
        problem = MFGProblem(
            geometry=geometry,
            T=0.1,
            Nt=10,
            diffusion=0.045,
            components=components,
        )
        solver = HJBSemiLagrangianSolver(
            problem,
            interpolation_method="cubic",
            diffusion_method="stochastic",
            check_cfl=False,
        )

        Nx = 31
        Nt = problem.Nt
        M_density = np.ones((Nt + 1, Nx))
        U_terminal = np.ones(Nx)

        U = solver.solve_hjb_system(
            M_density=M_density,
            U_terminal=U_terminal,
            U_coupling_prev=np.zeros((Nt + 1, Nx)),
        )

        np.testing.assert_allclose(U[0], 1.0, atol=1e-10)

    def test_consistency_with_default_adi(self):
        """Stochastic and default ADI must converge to the same numerical solution.

        Both schemes solve the same backward HJB; only the discretization
        path differs (Brownian quadrature vs. operator splitting). On a
        smooth Gaussian terminal with H=0, the difference should be
        within a few units of the local truncation error of either scheme.
        """
        from mfgarchon.core.hamiltonian import HamiltonianBase, OptimizationSense

        class ZeroH(HamiltonianBase):
            def __init__(self):
                super().__init__(sense=OptimizationSense.MINIMIZE)

            def __call__(self, x, m, p, t=0.0):
                p_arr = np.atleast_1d(np.asarray(p, dtype=float))
                if p_arr.ndim > 0:
                    return np.zeros(p_arr.shape[:-1])
                return 0.0

            def gradient_p(self, x, m, p, t=0.0):
                return np.zeros_like(np.asarray(p, dtype=float))

            def density_derivative(self, x, m, p, t=0.0):
                return 0.0

        sigma_test = 0.3
        T_test = 0.5
        beta_T = 1.0
        N, Nt = 100, 200

        geometry = TensorProductGrid(
            dimension=1,
            bounds=[(-5.0, 5.0)],
            Nx_points=[N + 1],
            boundary_conditions=no_flux_bc(dimension=1),
        )
        x_grid = geometry.get_spatial_grid().flatten()
        components = MFGComponents(
            hamiltonian=ZeroH(),
            m_initial=lambda x: 1.0,
            u_terminal=lambda x: float(np.exp(-(x[0] ** 2) / (2 * beta_T)) / np.sqrt(2 * np.pi * beta_T)),
        )
        problem = MFGProblem(
            geometry=geometry,
            T=T_test,
            Nt=Nt,
            diffusion=sigma_test**2 / 2,
            components=components,
        )

        U_terminal = np.exp(-(x_grid**2) / (2 * beta_T)) / np.sqrt(2 * np.pi * beta_T)
        M_density = np.ones((Nt + 1, N + 1))

        solver_st = HJBSemiLagrangianSolver(
            problem,
            interpolation_method="cubic",
            diffusion_method="stochastic",
            check_cfl=False,
        )
        solver_adi = HJBSemiLagrangianSolver(
            problem,
            interpolation_method="cubic",
            diffusion_method="adi",
            check_cfl=False,
        )

        U_st = solver_st.solve_hjb_system(
            M_density=M_density,
            U_terminal=U_terminal,
            U_coupling_prev=np.zeros((Nt + 1, N + 1)),
        )
        U_adi = solver_adi.solve_hjb_system(
            M_density=M_density,
            U_terminal=U_terminal,
            U_coupling_prev=np.zeros((Nt + 1, N + 1)),
        )

        max_diff = np.max(np.abs(U_st[0] - U_adi[0]))
        # Both schemes are 2nd-order accurate on smooth Gaussians; their
        # difference should be a few units of the local truncation error.
        assert max_diff < 5e-3, f"Stochastic and ADI diverge on smooth Gaussian: max diff = {max_diff:.3e}"


class TestStochasticSLUnificationPinning:
    """Issue #1050: pin the unified `_stochastic_sl_step` to the pre-merge 1D algorithm.

    `_stochastic_sl_step_1d` and `_stochastic_sl_step_nd` were merged into one
    dimension-agnostic `_stochastic_sl_step`. The merge must be byte-identical
    for 1D (orchestrator gate). This reconstructs the former 1D algorithm
    independently and asserts exact equality, so a future regression that
    silently changes the 1D path (e.g. swapping numpy.interp for RGI, or the
    FDM final-BC applicator for the Interpolation one) fails loudly here.
    """

    @staticmethod
    def _legacy_1d_step(solver, U_next, M_next, time_idx, dt):
        """The pre-#1050 `_stochastic_sl_step_1d`, inlined as the reference."""
        from scipy.interpolate import PchipInterpolator

        from mfgarchon.alg.numerical.hjb_solvers.h_eval import eval_H_batch
        from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_characteristics import reflect_into_domain
        from mfgarchon.geometry.boundary.bc_utils import (
            bc_type_to_geometric_operation,
            get_bc_type_string,
        )

        sigma = solver.problem.sigma
        sqrt_dt = float(np.sqrt(dt))
        diffusion_offset = sigma * sqrt_dt
        grad_u = solver._compute_gradient(U_next, check_cfl=True, t_idx=time_idx, m_density=M_next)
        # Issue #1413: drift along the characteristic velocity dH/dp = p/lambda (not raw p).
        lam = solver._control_cost_lambda()
        x_drift = solver.x_grid - (grad_u / lam) * dt
        y_plus = x_drift + diffusion_offset
        y_minus = x_drift - diffusion_offset
        bc = solver.get_boundary_conditions()
        bc_op = bc_type_to_geometric_operation(get_bc_type_string(bc))
        bounds = solver.problem.geometry.get_bounds()
        xmin, xmax = bounds[0][0], bounds[1][0]
        if bc_op == "reflect":
            y_plus = reflect_into_domain(y_plus, xmin, xmax)
            y_minus = reflect_into_domain(y_minus, xmin, xmax)
        elif bc_op == "wrap":
            L = xmax - xmin
            y_plus = xmin + (y_plus - xmin) % L
            y_minus = xmin + (y_minus - xmin) % L
        if solver.interpolation_method == "linear":
            u_plus = np.interp(y_plus, solver.x_grid, U_next)
            u_minus = np.interp(y_minus, solver.x_grid, U_next)
        else:
            interp_fn = PchipInterpolator(solver.x_grid, U_next, extrapolate=False)
            u_plus = interp_fn(y_plus)
            u_minus = interp_fn(y_minus)
        u_avg = 0.5 * (u_plus + u_minus)
        x_batch = solver.x_grid.reshape(-1, 1)
        p_batch = grad_u.reshape(-1, 1)
        H_class = solver.problem.hamiltonian_class
        # Issue #1413: Lax-Oleinik value update u_avg + dt*(H(p) - 2*H(0)), where H(0) = V+f.
        H_values = eval_H_batch(H_class, x_batch, M_next, p_batch, time_idx * dt).ravel()
        H_0 = eval_H_batch(H_class, x_batch, M_next, np.zeros_like(p_batch), time_idx * dt).ravel()
        U_current = u_avg + dt * (H_values - 2.0 * H_0)
        if bc:
            U_current = solver.bc_applicator.enforce_values(
                U_current, boundary_conditions=bc, spacing=(solver.dx,), time=time_idx * dt
            )
        return U_current

    @pytest.mark.parametrize("method", ["linear", "cubic"])
    def test_1d_step_byte_identical_to_legacy(self, method):
        import warnings as _w

        geometry = TensorProductGrid(
            dimension=1,
            bounds=[(0.0, 1.0)],
            Nx_points=[51],
            boundary_conditions=no_flux_bc(dimension=1),
        )
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, diffusion=0.045, components=_default_components())
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            solver = HJBSemiLagrangianSolver(
                problem,
                interpolation_method=method,
                diffusion_method="stochastic",
                check_cfl=False,
            )

        x_grid = solver.x_grid
        # Non-trivial U (curvature) and varying M so interpolation + Hamiltonian
        # both contribute; time_idx mid-horizon.
        U_next = np.sin(3.0 * x_grid) + 0.2 * np.cos(7.0 * x_grid)
        M_next = 1.0 + 0.1 * np.cos(2.0 * np.pi * x_grid)
        time_idx, dt = 25, problem.T / problem.Nt

        got = solver._stochastic_sl_step(U_next.copy(), M_next.copy(), time_idx, dt)
        ref = self._legacy_1d_step(solver, U_next.copy(), M_next.copy(), time_idx, dt)
        np.testing.assert_array_equal(got, ref)


class TestStochasticCharacteristicSL_nD:  # noqa: N801 — SL_nD = semi-Lagrangian, n-Dimensional (deliberate test-group name)
    """Issue #1054: nD stochastic SL companion fixes (analogous to 1D #1033/#1048/#1049)."""

    def _make_2d_problem(self, sigma=0.3, T=0.1, Nt=4, N=15):
        from mfgarchon.core.hamiltonian import HamiltonianBase, OptimizationSense

        class ZeroH(HamiltonianBase):
            def __init__(self):
                super().__init__(sense=OptimizationSense.MINIMIZE)

            def __call__(self, x, m, p, t=0.0):
                p_arr = np.atleast_1d(np.asarray(p, dtype=float))
                if p_arr.ndim > 0:
                    return np.zeros(p_arr.shape[:-1])
                return 0.0

            def gradient_p(self, x, m, p, t=0.0):
                return np.zeros_like(np.asarray(p, dtype=float))

            def density_derivative(self, x, m, p, t=0.0):
                return 0.0

        bc = no_flux_bc(dimension=2)
        grid = TensorProductGrid(
            dimension=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[N, N],
            boundary_conditions=bc,
        )

        def m0(x):
            return np.exp(-10.0 * ((x[..., 0] - 0.5) ** 2 + (x[..., 1] - 0.5) ** 2))

        components = MFGComponents(
            hamiltonian=ZeroH(),
            m_initial=m0,
            u_terminal=lambda x: 1.0,
        )
        return (
            MFGProblem(
                geometry=grid,
                T=T,
                Nt=Nt,
                sigma=sigma,
                components=components,
                boundary_conditions=bc,
            ),
            grid,
            m0,
        )

    def test_2d_linear_stochastic_finite(self):
        """Issue #1054: linear+stochastic on 2D no-flux must produce finite output."""
        problem, grid, m0 = self._make_2d_problem()
        solver = HJBSemiLagrangianSolver(
            problem,
            diffusion_method="stochastic",
            interpolation_method="linear",
            check_cfl=False,
        )
        U_terminal = np.zeros(tuple(grid.Nx_points))
        M_init = m0(np.stack(np.meshgrid(*grid.coordinates, indexing="ij"), axis=-1))
        U_step = solver._stochastic_sl_step(U_terminal, M_init, time_idx=problem.Nt - 1, dt=0.025)
        assert U_step.shape == tuple(grid.Nx_points)
        assert np.isfinite(U_step).all()

    def test_2d_cubic_stochastic_uses_pchip(self):
        """Issue #1054: cubic+stochastic in nD routes through monotone PCHIP (no NaN)."""
        problem, grid, m0 = self._make_2d_problem()
        with __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            solver = HJBSemiLagrangianSolver(
                problem,
                diffusion_method="stochastic",
                interpolation_method="cubic",
                check_cfl=False,
            )
        U_terminal = np.zeros(tuple(grid.Nx_points))
        M_init = m0(np.stack(np.meshgrid(*grid.coordinates, indexing="ij"), axis=-1))
        U_step = solver._stochastic_sl_step(U_terminal, M_init, time_idx=problem.Nt - 1, dt=0.025)
        assert np.isfinite(U_step).all()

    def test_2d_reflect_bc_no_extrapolation(self):
        """Issue #1054: high-curvature U near walls must not produce NaN under no-flux."""
        problem, grid, m0 = self._make_2d_problem()
        solver = HJBSemiLagrangianSolver(
            problem,
            diffusion_method="stochastic",
            interpolation_method="linear",
            check_cfl=False,
        )
        # Stress-test: bowl U_T peaking near walls — Brownian feet would otherwise
        # extrapolate (silent fill_value=None) and produce nonsense values.
        Nx = grid.Nx_points[0]
        U_terminal = np.fromfunction(
            lambda i, j: (i / (Nx - 1) - 0.5) ** 2 + (j / (Nx - 1) - 0.5) ** 2,
            (Nx, Nx),
        ).astype(float)
        M_init = m0(np.stack(np.meshgrid(*grid.coordinates, indexing="ij"), axis=-1))
        U_step = solver._stochastic_sl_step(U_terminal, M_init, time_idx=problem.Nt - 1, dt=0.025)
        assert np.isfinite(U_step).all()


class TestADIDiffusionMagnitude:
    """Regression: the nD ADI diffusion step must apply the FULL prescribed diffusion.

    A cosine eigenmode of the Laplacian decays analytically as
    ``exp(-D (sum_d k_d^2) t)`` with ``D = sigma^2/2``. The sequential (Lie) ADI split
    is exact for this separable mode up to Crank-Nicolson time truncation, so the ADI
    decay must match the analytic decay. The pre-fix code used ``dt/dimension`` per
    directional sweep, applying only ``1/dimension`` of the diffusion (2x under in 2D,
    3x in 3D) — a silent magnitude error no prior test caught (they asserted only
    finiteness / loose mass). These tests fail on that bug and pass on the full-dt fix.
    """

    @staticmethod
    def _decay_relerr(adi_fac, analytic_fac):
        # relative error in the (small) decay increment, robust near fac~1
        return abs(adi_fac - analytic_fac) / abs(1.0 - analytic_fac)

    def test_adi_2d_diffusion_matches_analytic(self):
        from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_adi import adi_diffusion_step

        N, L, sigma, dt = 81, 1.0, 1.0, 1e-3
        D = 0.5 * sigma**2
        x = np.linspace(0.0, L, N)
        X, Y = np.meshgrid(x, x, indexing="ij")
        k = np.pi
        u0 = np.cos(k * X) * np.cos(k * Y)
        dx = L / (N - 1)
        u1 = adi_diffusion_step(u0.copy(), dt, sigma, np.array([dx, dx]), (N, N), "neumann")
        i, j = N // 3, N // 4
        adi_fac = u1[i, j] / u0[i, j]
        analytic_fac = np.exp(-D * (2 * k**2) * dt)  # full 2D decay
        # pre-fix this would be exp(-D*k^2*dt) (half exponent) -> rel error ~1.0
        assert self._decay_relerr(adi_fac, analytic_fac) < 0.02, (
            f"ADI 2D under/over-diffuses: factor {adi_fac:.6f} vs analytic {analytic_fac:.6f}"
        )

    def test_adi_3d_diffusion_matches_analytic(self):
        from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_adi import adi_diffusion_step

        N, L, sigma, dt = 31, 1.0, 1.0, 5e-4
        D = 0.5 * sigma**2
        x = np.linspace(0.0, L, N)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        k = np.pi
        u0 = np.cos(k * X) * np.cos(k * Y) * np.cos(k * Z)
        dx = L / (N - 1)
        u1 = adi_diffusion_step(u0.copy(), dt, sigma, np.array([dx, dx, dx]), (N, N, N), "neumann")
        i = N // 3
        adi_fac = u1[i, i, i] / u0[i, i, i]
        analytic_fac = np.exp(-D * (3 * k**2) * dt)  # full 3D decay
        # pre-fix this would be exp(-D*k^2*dt) (one-third exponent) -> rel error ~2.0
        assert self._decay_relerr(adi_fac, analytic_fac) < 0.03, (
            f"ADI 3D under/over-diffuses: factor {adi_fac:.6f} vs analytic {analytic_fac:.6f}"
        )


class TestReflectIntoDomain:
    """Regression guard for the reflecting-BC characteristic-foot fold (Issues #1161/#1048/#1054).

    The vectorized fold must equal the trusted iterated scalar reflection
    (``apply_boundary_conditions_1d``) on ASYMMETRIC domains. The bug that shipped used
    ``xmin + |((x-xmin) % 2L) - L|`` (a center-flip about the midpoint), which is correct only
    on domains symmetric about their center — exactly why the prior [0,1]-only tests missed it.
    """

    def test_matches_trusted_scalar_reference_asymmetric(self):
        from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_characteristics import (
            apply_boundary_conditions_1d,
            reflect_into_domain,
        )

        xmin, xmax = 2.0, 5.0  # asymmetric, off-origin: center-flip != reflection here
        pts = np.array([2.5, 1.8, 1.0, 5.0, 2.0, 3.5, 6.2, -1.0, 8.3, 4.99])
        new = reflect_into_domain(pts, xmin, xmax)
        ref = np.array([apply_boundary_conditions_1d(float(x), xmin, xmax, "reflect") for x in pts])
        np.testing.assert_allclose(new, ref, atol=1e-12)

    def test_in_bounds_is_identity(self):
        from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_characteristics import reflect_into_domain

        xmin, xmax = 2.0, 5.0
        interior = np.array([2.0, 2.5, 3.5, 4.99, 5.0])
        np.testing.assert_allclose(reflect_into_domain(interior, xmin, xmax), interior, atol=1e-12)

    def test_single_and_multi_bounce(self):
        from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_characteristics import reflect_into_domain

        xmin, xmax = 2.0, 5.0  # L = 3
        # 6.2 over by 1.2 -> 3.8 (one bounce); 8.3 -> two bounces -> 2.3; -1.0 -> 5.0
        out = reflect_into_domain(np.array([6.2, 8.3, -1.0]), xmin, xmax)
        np.testing.assert_allclose(out, [3.8, 2.3, 5.0], atol=1e-12)
        assert np.all((out >= xmin - 1e-12) & (out <= xmax + 1e-12))

    def test_not_center_flip(self):
        """The specific bug: an interior point must NOT be mirrored about the midpoint."""
        from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_characteristics import reflect_into_domain

        xmin, xmax = 2.0, 5.0
        x = 2.5
        flipped = xmin + xmax - x  # 4.5 — what the broken formula produced
        out = reflect_into_domain(np.array([x]), xmin, xmax)[0]
        assert out == pytest.approx(x)
        assert out != pytest.approx(flipped)

    def test_per_axis_broadcast_nd(self):
        from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_characteristics import reflect_into_domain

        xmn = np.array([0.0, 2.0])
        xmx = np.array([1.0, 5.0])
        pts = np.array([[0.3, 2.5], [1.2, 6.2], [-0.3, 1.0]])
        out = reflect_into_domain(pts, xmn, xmx)
        np.testing.assert_allclose(out, [[0.3, 2.5], [0.8, 3.8], [0.3, 3.0]], atol=1e-12)
        assert np.all((out >= xmn - 1e-9) & (out <= xmx + 1e-9))


class TestSLHamiltonianSingleSource:
    """Issue #1071: _evaluate_hamiltonian routes the H value through the single-source
    batch shim (eval_H_batch -> HamiltonianBase.evaluate_H) instead of calling H_class
    directly."""

    def test_evaluate_hamiltonian_byte_identical_to_inline_call(self):
        """Pin: _evaluate_hamiltonian(x, p, m, t_idx) is byte-identical (exact IEEE-754) to
        the inline ``float(H_class(x_vec, m, p_vec, t))`` it replaced. evaluate_H wraps
        ``__call__`` (np.asarray(self(...), dtype=float)), so float() of the shim equals the
        direct call; this locks SL against a future divergence of the shim from __call__."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBSemiLagrangianSolver(problem)
        H_class = problem.hamiltonian_class

        rng = np.random.default_rng(1071)
        for _ in range(100):
            x = float(rng.uniform(0.0, 1.0))
            p = float(rng.standard_normal() * 3.0)
            m = float(rng.uniform(1e-3, 5.0))
            t_idx = int(rng.integers(0, problem.Nt + 1))
            t_value = t_idx * problem.T / problem.Nt
            inline = float(H_class(np.atleast_1d(x), m, np.atleast_1d(p), t_value))
            routed = solver._evaluate_hamiltonian(x, p, m, t_idx)
            assert routed.hex() == inline.hex(), (
                f"x={x} p={p} m={m} t_idx={t_idx}: routed {routed!r} != inline {inline!r}"
            )


class TestSLHJBConsistency:
    """Issue #1413: the semi-Lagrangian scheme must solve the correct HJB.

    Pins the corrected Lax-Oleinik update ``u_dep + dt*(H(p) - 2*H(0))`` with a λ-scaled
    departure foot against (a) the analytic Hopf-Lax solution for σ→0 pure-LQ, and (b) FDM
    (the reference solver) with a potential. Before the fix the H-based SL was ~24% off the
    analytic solution even at λ=1 AND non-convergent (the foot + ``-dt*H`` double-counted the
    kinetic term ≈3×); the foot was also λ=1-only. Issue #575 corrected the state term but
    left the kinetic error. FDM matches the analytic Hopf-Lax to ~0.6%, so it is the oracle.
    """

    @staticmethod
    def _build(lam, potential, coupling, coupling_dm, sigma, nx, Nt, T):
        from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian

        geom = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=no_flux_bc(dimension=1))
        comp = MFGComponents(
            m_initial=lambda x: 0.5 + np.exp(-10 * (x - 0.5) ** 2),
            u_terminal=lambda x: 0.5 * (x - 0.5) ** 2,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=lam),
                potential=potential,
                coupling=coupling,
                coupling_dm=coupling_dm,
            ),
        )
        problem = MFGProblem(geometry=geom, T=T, Nt=Nt, sigma=sigma, components=comp)
        x = np.linspace(0.0, 1.0, nx)
        U_T = 0.5 * (x - 0.5) ** 2
        M = np.tile(0.5 + np.exp(-10 * (x - 0.5) ** 2), (Nt + 1, 1))
        U_prev = np.tile(U_T, (Nt + 1, 1))
        return problem, x, U_T, M, U_prev

    @staticmethod
    def _solve(solver_cls, problem, U_T, M, U_prev):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = solver_cls(problem)
            U = np.asarray(s.solve_hjb_system(M_density=M, U_terminal=U_T, U_coupling_prev=U_prev))
        return U[0] if U.ndim == 2 else U

    @pytest.mark.parametrize("lam", [1.0, 2.0])
    def test_matches_analytic_hopf_lax(self, lam):
        """σ→0 pure-LQ: SL must match the λ-dependent Hopf-Lax solution
        ``u(0,x) = min_y[u_T(y) + λ(x-y)²/(2T)]`` — and for λ≠1 must be CLEARLY distinct
        from the λ=1 solution (so a regression to the λ=1-foot fails here)."""
        T, nx, Nt = 0.2, 121, 200
        problem, x, U_T, M, U_prev = self._build(lam, None, None, None, sigma=1e-3, nx=nx, Nt=Nt, T=T)
        U_sl = self._solve(HJBSemiLagrangianSolver, problem, U_T, M, U_prev)

        y = np.linspace(0.0, 1.0, 4001)

        def hopf_lax(xq, lq):
            return np.array([np.min(0.5 * (y - 0.5) ** 2 + lq * (xi - y) ** 2 / (2.0 * T)) for xi in xq])

        interior = (x > 0.25) & (x < 0.75)
        a_lam = hopf_lax(x, lam)
        denom = max(float(np.max(np.abs(a_lam[interior]))), 1e-12)
        err = float(np.max(np.abs(U_sl[interior] - a_lam[interior]))) / denom
        assert err < 0.05, f"λ={lam}: SL vs analytic Hopf-Lax rel-err={err:.3e} (expected <5%)"
        if lam != 1.0:
            a_1 = hopf_lax(x, 1.0)
            err_1 = float(np.max(np.abs(U_sl[interior] - a_1[interior]))) / denom
            assert err_1 > 0.05, (
                f"λ={lam}: SL is indistinguishable from the λ=1 solution (rel-err vs analytic(1)="
                f"{err_1:.3e}) — the λ-scaled foot regressed (Issue #1413)."
            )

    @pytest.mark.parametrize("lam", [1.0, 2.0])
    def test_matches_fdm_with_potential(self, lam):
        """σ>0 with a potential V(x): SL must match FDM (the reference). Pins the state-term
        sign — the corrected scheme is ``+dt*H_control - dt*(V+f)`` (Issue #575/#1413)."""
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

        def V(x, t):
            return -0.5 * (np.asarray(x)[..., 0] - 0.5) ** 2

        T, nx, Nt = 0.3, 101, 120
        problem, x, U_T, M, U_prev = self._build(lam, V, None, None, sigma=0.25, nx=nx, Nt=Nt, T=T)
        U_sl = self._solve(HJBSemiLagrangianSolver, problem, U_T, M, U_prev)
        problem_f, _, _, _, _ = self._build(lam, V, None, None, sigma=0.25, nx=nx, Nt=Nt, T=T)
        U_fdm = self._solve(HJBFDMSolver, problem_f, U_T, M, U_prev)
        interior = (x > 0.2) & (x < 0.8)
        denom = max(float(np.max(np.abs(U_fdm[interior]))), 1e-12)
        err = float(np.max(np.abs(U_sl[interior] - U_fdm[interior]))) / denom
        assert err < 0.05, f"λ={lam}: SL(+V) vs FDM rel-err={err:.3e} (expected <5%)"


class TestSLValueUpdateND:
    """Issue #1413/#1417: pin the nD Lax-Oleinik value combination ``u_foot + dt*(H(p) - 2*H(0))``.

    The 1D update is covered by TestSLHJBConsistency (analytic Hopf-Lax). nD previously had
    only finiteness checks, so the corrected λ-aware kinetic term was invisible in every nD
    test (audit finding S0-28). This pins ``_sl_value_update`` directly on a 2D batch against
    the analytic LQ closed form ``u_foot + dt*(|p|²/(2λ) - V - f)`` for λ≠1 with V,f≠0, and
    asserts it stays distinct from the pre-#575/#1413 scheme (kinetic 3x / λ=1-only foot).
    """

    @staticmethod
    def _build_2d(lam):
        def V(x, t):
            return 0.2 * (x[:, 0] ** 2 + x[:, 1] ** 2)

        geom = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[6, 5],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        comp = MFGComponents(
            m_initial=lambda x: 1.0,
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=lam),
                potential=V,
                coupling=lambda m: 0.7 * m,
                coupling_dm=lambda m: 0.7 * np.ones_like(m),
            ),
        )
        problem = MFGProblem(geometry=geom, T=0.1, Nt=10, sigma=0.1, components=comp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver = HJBSemiLagrangianSolver(problem)
        return solver, V

    @pytest.mark.parametrize("lam", [1.0, 2.0, 0.5])
    def test_nd_value_update_matches_analytic_lq(self, lam):
        solver, V = self._build_2d(lam)
        xs = np.linspace(0.0, 1.0, 6)
        ys = np.linspace(0.0, 1.0, 5)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        n = pts.shape[0]
        idx = np.arange(n)
        p = np.stack([0.1 + 0.03 * idx, -0.2 + 0.02 * idx], axis=1)
        m = 0.5 + 0.01 * idx
        u_foot = np.cos(idx * 0.3)
        t, dt = 0.05, 0.01

        out = np.asarray(solver._sl_value_update(u_foot, pts, m, p, t, dt))

        # Independent analytic LQ form: H = |p|^2/(2λ) + V(x) + f(m), so
        # H(p) - 2*H(0) = |p|^2/(2λ) - (V + f).
        h_control = np.sum(p**2, axis=1) / (2.0 * lam)
        h_state = V(pts, t) + 0.7 * m
        expected = u_foot + dt * (h_control - h_state)
        old_scheme = u_foot - dt * (h_control + h_state)  # pre-#575/#1413

        assert out.shape == (n,)
        np.testing.assert_allclose(
            out, expected, atol=1e-10, err_msg=f"λ={lam}: nD value update != analytic LQ closed form"
        )
        assert not np.allclose(out, old_scheme, atol=1e-6), (
            f"λ={lam}: nD value update matches the pre-#1413 scheme — corrected kinetic term regressed"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

#!/usr/bin/env python3
"""
Unit tests for HJBFDMSolver.

Tests the Finite Difference Method (FDM) solver for Hamilton-Jacobi-Bellman equations
using upwind schemes and Newton iteration.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
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


class TestHJBFDMSolverInitialization:
    """Test HJBFDMSolver initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic solver initialization with default parameters."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBFDMSolver(problem)

        assert solver.hjb_method_name == "FDM"
        assert solver.max_newton_iterations == 30
        assert solver.newton_tolerance == 1e-6
        assert solver.problem is problem

    def test_custom_newton_parameters(self):
        """Test initialization with custom Newton parameters."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBFDMSolver(
            problem,
            max_newton_iterations=50,
            newton_tolerance=1e-8,
        )

        assert solver.max_newton_iterations == 50
        assert solver.newton_tolerance == 1e-8

    def test_removed_NiterNewton_raises(self):
        """v0.16 deprecation removed: NiterNewton= no longer in __init__ signature -> TypeError."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())

        with pytest.raises(TypeError, match="NiterNewton"):
            HJBFDMSolver(problem, NiterNewton=40)

    def test_removed_l2errBoundNewton_raises(self):
        """v0.16 deprecation removed: l2errBoundNewton= no longer in __init__ signature -> TypeError."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())

        with pytest.raises(TypeError, match="l2errBoundNewton"):
            HJBFDMSolver(problem, l2errBoundNewton=1e-5)

    def test_invalid_max_iterations(self):
        """Test that invalid max_newton_iterations raises error."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())

        with pytest.raises(ValueError, match="max_newton_iterations must be >= 1"):
            HJBFDMSolver(problem, max_newton_iterations=0)

        with pytest.raises(ValueError, match="max_newton_iterations must be >= 1"):
            HJBFDMSolver(problem, max_newton_iterations=-5)

    def test_invalid_tolerance(self):
        """Test that invalid newton_tolerance raises error."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())

        with pytest.raises(ValueError, match="newton_tolerance must be > 0"):
            HJBFDMSolver(problem, newton_tolerance=0.0)

        with pytest.raises(ValueError, match="newton_tolerance must be > 0"):
            HJBFDMSolver(problem, newton_tolerance=-1e-6)

    def test_newton_config_storage(self):
        """Test that Newton configuration is properly stored."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBFDMSolver(
            problem,
            max_newton_iterations=35,
            newton_tolerance=1e-7,
        )

        assert hasattr(solver, "_newton_config")
        assert solver._newton_config["max_iterations"] == 35
        assert solver._newton_config["tolerance"] == 1e-7

    def test_backend_initialization(self):
        """Test that backend is properly initialized."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBFDMSolver(problem)

        assert hasattr(solver, "backend")
        assert solver.backend is not None

    def test_custom_backend(self):
        """Test initialization with custom backend."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBFDMSolver(problem, backend="numpy")

        assert solver.backend is not None
        # Backend should be NumPy backend
        assert hasattr(solver.backend, "array")

    def test_advection_scheme_default(self):
        """Test default advection_scheme is gradient_upwind."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBFDMSolver(problem)

        assert solver.advection_scheme == "gradient_upwind"
        assert solver.use_upwind is True

    def test_advection_scheme_gradient_upwind(self):
        """Test explicit gradient_upwind advection scheme."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBFDMSolver(problem, advection_scheme="gradient_upwind")

        assert solver.advection_scheme == "gradient_upwind"
        assert solver.use_upwind is True

    def test_advection_scheme_gradient_centered(self):
        """Test gradient_centered advection scheme."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBFDMSolver(problem, advection_scheme="gradient_centered")

        assert solver.advection_scheme == "gradient_centered"
        assert solver.use_upwind is False

    def test_advection_scheme_invalid(self):
        """Test that invalid advection_scheme raises ValueError."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())

        with pytest.raises(ValueError, match="Invalid advection_scheme"):
            HJBFDMSolver(problem, advection_scheme="invalid_scheme")

        with pytest.raises(ValueError, match="Invalid advection_scheme"):
            HJBFDMSolver(problem, advection_scheme="divergence_upwind")  # FP scheme


class TestRemovedSolveHJBSystemKwargs:
    """v0.17 deprecations removed: solve_hjb_system no longer accepts the legacy
    density/terminal/coupling kwarg names or bc_values -> TypeError (unexpected kwarg)."""

    @pytest.fixture(scope="class")
    def solver(self):
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=10, components=_default_components())
        return HJBFDMSolver(problem)

    @pytest.mark.parametrize(
        "kwarg",
        [
            "M_density_evolution_from_FP",
            "M_density_evolution",
            "U_final_condition_at_T",
            "U_final_condition",
            "U_from_prev_picard",
            "bc_values",
        ],
    )
    def test_removed_kwarg_raises_type_error(self, solver, kwarg):
        """Each removed legacy kwarg now raises TypeError when passed to solve_hjb_system."""
        with pytest.raises(TypeError, match=kwarg):
            solver.solve_hjb_system(**{kwarg: None})


class TestHJBFDMSolverSolveHJBSystem:
    """Test the main solve_hjb_system method."""

    def test_solve_hjb_system_shape(self):
        """Test that solve_hjb_system returns correct shape."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBFDMSolver(problem)

        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]

        # Create inputs
        M_density = np.ones((Nt_points, Nx_points))
        U_final = np.zeros(Nx_points)
        U_prev = np.zeros((Nt_points, Nx_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert U_solution.shape == (Nt_points, Nx_points)
        assert np.all(np.isfinite(U_solution))

    def test_solve_hjb_system_final_condition(self):
        """Test that final condition is preserved."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBFDMSolver(problem)

        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]
        bounds = problem.geometry.get_bounds()

        # Create inputs with specific final condition
        M_density = np.ones((Nt_points, Nx_points))
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - bounds[1][0]) ** 2
        U_prev = np.zeros((Nt_points, Nx_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Final time step should match final condition
        assert np.allclose(U_solution[-1, :], U_final, rtol=0.1)

    def test_solve_hjb_system_backward_propagation(self):
        """Test that solution propagates backward in time."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBFDMSolver(problem)

        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]
        bounds = problem.geometry.get_bounds()

        # Create inputs
        M_density = np.ones((Nt_points, Nx_points))
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = x_coords**2  # Quadratic final condition
        U_prev = np.zeros((Nt_points, Nx_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Solution should propagate backward (values at earlier times influenced by final)
        # Check that solution at t=0 is different from zero
        assert not np.allclose(U_solution[0, :], 0.0)

    def test_solve_hjb_system_with_density_variation(self):
        """Test solving with non-uniform density."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBFDMSolver(problem)

        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]
        bounds = problem.geometry.get_bounds()

        # Create Gaussian density
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_profile = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        M_density = np.tile(m_profile, (Nt_points, 1))

        U_final = np.zeros(Nx_points)
        U_prev = np.zeros((Nt_points, Nx_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Should produce valid solution
        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (Nt_points, Nx_points)


class TestHJBFDMSolverNumericalProperties:
    """Test numerical properties of the FDM method."""

    def test_solution_finiteness(self):
        """Test that solution remains finite throughout."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=40, components=_default_components())
        solver = HJBFDMSolver(problem)

        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]
        bounds = problem.geometry.get_bounds()

        M_density = np.ones((Nt_points, Nx_points)) * 0.5
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = np.sin(2 * np.pi * x_coords)
        U_prev = np.zeros((Nt_points, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # All values should be finite
        assert np.all(np.isfinite(U_solution))

    def test_solution_smoothness(self):
        """Test that solution has reasonable smoothness."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, components=_default_components())
        solver = HJBFDMSolver(problem)

        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]
        bounds = problem.geometry.get_bounds()

        M_density = np.ones((Nt_points, Nx_points))
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = 0.5 * (x_coords - 0.5) ** 2
        U_prev = np.zeros((Nt_points, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Check spatial smoothness - finite differences shouldn't be too large
        # With dx ≈ 0.02, max gradient differences can be significant for HJB solutions
        U_diff = np.diff(U_solution, axis=1)
        assert np.max(np.abs(U_diff)) < 1000.0


class TestHJBFDMSolverParameterSensitivity:
    """Test solver behavior with different parameter configurations."""

    def test_different_newton_iterations(self):
        """Test solver with different Newton iteration counts."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]

        for n_iter in [10, 20, 30, 50]:
            solver = HJBFDMSolver(problem, max_newton_iterations=n_iter)

            M_density = np.ones((Nt_points, Nx_points))
            U_final = np.zeros(Nx_points)
            U_prev = np.zeros((Nt_points, Nx_points))

            U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

            assert np.all(np.isfinite(U_solution))

    def test_different_tolerances(self):
        """Test solver with different Newton tolerances."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=20, components=_default_components())
        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]

        for tol in [1e-4, 1e-6, 1e-8]:
            solver = HJBFDMSolver(problem, newton_tolerance=tol)

            M_density = np.ones((Nt_points, Nx_points))
            U_final = np.zeros(Nx_points)
            U_prev = np.zeros((Nt_points, Nx_points))

            U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

            assert np.all(np.isfinite(U_solution))


class TestHJBFDMSolverIntegration:
    """Integration tests with actual MFG problems."""

    def test_solver_with_uniform_density(self):
        """Test solver with uniform density distribution."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBFDMSolver(problem)

        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]
        bounds = problem.geometry.get_bounds()

        # Uniform density
        M_density = np.ones((Nt_points, Nx_points)) / Nx_points

        # Simple final condition
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        U_final = (x_coords - 0.5) ** 2

        U_prev = np.zeros((Nt_points, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Should produce valid solution
        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (Nt_points, Nx_points)

    def test_solver_with_gaussian_density(self):
        """Test solver with Gaussian density distribution."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())
        solver = HJBFDMSolver(problem)

        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]
        bounds = problem.geometry.get_bounds()

        # Gaussian density
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_profile = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        m_profile = m_profile / np.sum(m_profile)
        M_density = np.tile(m_profile, (Nt_points, 1))

        U_final = np.zeros(Nx_points)
        U_prev = np.zeros((Nt_points, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Should produce valid solution
        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (Nt_points, Nx_points)


class TestHJBFDMSolverNotAbstract:
    """Test that HJBFDMSolver is concrete (not abstract)."""

    def test_solver_not_abstract(self):
        """Test that HJBFDMSolver can be instantiated."""
        import inspect

        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=30, components=_default_components())

        # Should not raise TypeError about abstract methods
        solver = HJBFDMSolver(problem)
        assert isinstance(solver, HJBFDMSolver)

        # Should not have abstract methods
        assert not inspect.isabstract(HJBFDMSolver)


class TestHJBFDMSolverDiagonalTensor:
    """Test HJBFDMSolver with diagonal tensor diffusion (Phase 3.1)."""

    def test_diagonal_tensor_2d(self):
        """Test HJB solver with constant diagonal tensor in 2D."""
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 0.6)], Nx_points=[16, 11], boundary_conditions=no_flux_bc(dimension=2)
        )
        problem = MFGProblem(geometry=domain, T=0.05, Nt=5, sigma=0.1, components=_default_components_2d())

        solver = HJBFDMSolver(problem, solver_type="newton")

        # Get grid shape
        Nx, Ny = domain.get_grid_shape()
        Nt_points = problem.Nt_points

        # Create dummy density and initial conditions
        M_density = np.ones((Nt_points, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt_points, Nx, Ny))

        # Diagonal tensor: fast horizontal, slow vertical
        Sigma = np.diag([0.15, 0.05])

        # Should not warn for diagonal tensor
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            U_solution = solver.solve_hjb_system(M_density, U_final, U_prev, tensor_volatility_field=Sigma)

            # Check that no non-deprecation warning was raised for diagonal tensor
            # (DeprecationWarning from @deprecated_parameter is expected)
            tensor_warnings = [
                warning
                for warning in w
                if "tensor_volatility_field" in str(warning.message)
                and not issubclass(warning.category, DeprecationWarning)
            ]
            assert len(tensor_warnings) == 0, "Should not warn for diagonal tensor"

        # Verify solution shape and validity
        assert U_solution.shape == (Nt_points, Nx, Ny)
        assert not np.any(np.isnan(U_solution))
        assert not np.any(np.isinf(U_solution))

    def test_non_diagonal_tensor_warning(self):
        """Test that HJB solver warns for non-diagonal tensor."""
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 0.6)], Nx_points=[16, 11], boundary_conditions=no_flux_bc(dimension=2)
        )
        problem = MFGProblem(geometry=domain, T=0.05, Nt=5, sigma=0.1, components=_default_components_2d())

        solver = HJBFDMSolver(problem, solver_type="newton")

        # Get grid shape
        Nx, Ny = domain.get_grid_shape()
        Nt_points = problem.Nt_points

        # Create dummy density and initial conditions
        M_density = np.ones((Nt_points, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt_points, Nx, Ny))

        # Non-diagonal tensor with off-diagonal elements
        Sigma = np.array([[0.15, 0.02], [0.02, 0.05]])

        # Should warn for non-diagonal tensor
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            U_solution = solver.solve_hjb_system(M_density, U_final, U_prev, tensor_volatility_field=Sigma)

            # Check that warning was raised
            tensor_warnings = [warning for warning in w if "non-diagonal" in str(warning.message).lower()]
            assert len(tensor_warnings) > 0, "Should warn for non-diagonal tensor"
            # Issue #1079: the warning must convey the O(1) severity (dropping off-diagonal
            # sigma_ij is not a small correction), not merely "using diagonal approximation".
            assert any(
                "o(1)" in str(warning.message).lower() or "1079" in str(warning.message) for warning in tensor_warnings
            ), "Warning should state the off-diagonal drop is an O(1) error (Issue #1079)"

        # Solution should still be computed (fallback to diagonal)
        assert U_solution.shape == (Nt_points, Nx, Ny)
        assert not np.any(np.isnan(U_solution))

    def test_spatially_varying_tensor_averaging_warning(self):
        """Issue #1079: a spatially-varying *diagonal* tensor is averaged to constant per-axis
        weights — previously silent. Must warn (and must NOT raise the non-diagonal warning)."""
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 0.6)], Nx_points=[12, 9], boundary_conditions=no_flux_bc(dimension=2)
        )
        problem = MFGProblem(geometry=domain, T=0.05, Nt=4, sigma=0.1, components=_default_components_2d())
        solver = HJBFDMSolver(problem, solver_type="newton")

        Nx, Ny = domain.get_grid_shape()
        Nt_points = problem.Nt_points
        M_density = np.ones((Nt_points, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt_points, Nx, Ny))

        # Spatially-varying DIAGONAL tensor field (Nx, Ny, 2, 2); off-diagonals zero,
        # diagonal varies in x → forces the "average to constant" reduction.
        field = np.zeros((Nx, Ny, 2, 2))
        field[..., 0, 0] = 0.05 + 0.1 * np.linspace(0.0, 1.0, Nx)[:, None]
        field[..., 1, 1] = 0.05

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver.solve_hjb_system(M_density, U_final, U_prev, tensor_volatility_field=field)

            averaging = [x for x in w if "averaged to constant" in str(x.message)]
            assert len(averaging) > 0, "Should warn that a spatially-varying tensor is averaged to constant"
            assert any("1079" in str(x.message) for x in averaging)
            # The field is diagonal, so the non-diagonal-drop warning must NOT fire.
            non_diag = [x for x in w if "non-diagonal" in str(x.message).lower()]
            assert len(non_diag) == 0, "Diagonal field must not trigger the off-diagonal warning"

    def test_diagonal_tensor_is_diagonal_helper(self):
        """Test is_diagonal_tensor helper function."""
        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import is_diagonal_tensor

        # Test diagonal tensor
        Sigma_diag = np.diag([0.15, 0.05])
        assert is_diagonal_tensor(Sigma_diag), "Should detect diagonal tensor"

        # Test non-diagonal tensor
        Sigma_full = np.array([[0.15, 0.02], [0.02, 0.05]])
        assert not is_diagonal_tensor(Sigma_full), "Should detect non-diagonal tensor"

        # Test nearly-diagonal tensor (within tolerance)
        Sigma_nearly = np.diag([0.15, 0.05])
        Sigma_nearly[0, 1] = 1e-12  # Very small off-diagonal
        Sigma_nearly[1, 0] = 1e-12
        assert is_diagonal_tensor(Sigma_nearly), "Should treat nearly-diagonal as diagonal"

        # Test spatially-varying diagonal tensor
        Sigma_spatial = np.zeros((10, 10, 2, 2))
        for i in range(10):
            for j in range(10):
                Sigma_spatial[i, j] = np.diag([0.15, 0.05])
        assert is_diagonal_tensor(Sigma_spatial), "Should detect spatially-varying diagonal"

    def test_diagonal_tensor_spatially_varying(self):
        """Test HJB solver with spatially-varying diagonal tensor."""
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 0.6)], Nx_points=[16, 11], boundary_conditions=no_flux_bc(dimension=2)
        )
        problem = MFGProblem(geometry=domain, T=0.05, Nt=3, sigma=0.1, components=_default_components_2d())

        solver = HJBFDMSolver(problem, solver_type="newton")

        # Get grid shape
        Nx, Ny = domain.get_grid_shape()
        Nt_points = problem.Nt_points

        # Create dummy density and initial conditions
        M_density = np.ones((Nt_points, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt_points, Nx, Ny))

        # Spatially-varying diagonal tensor
        Sigma_spatial = np.zeros((Nx, Ny, 2, 2))
        for i in range(Nx):
            for j in range(Ny):
                # Vary diffusion based on position
                sigma_x = 0.1 + 0.1 * (i / Nx)
                sigma_y = 0.05
                Sigma_spatial[i, j] = np.diag([sigma_x**2, sigma_y**2])

        # Should not warn for spatially-varying diagonal tensor
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            U_solution = solver.solve_hjb_system(M_density, U_final, U_prev, tensor_volatility_field=Sigma_spatial)

            # No non-deprecation warnings for spatially-varying diagonal
            tensor_warnings = [
                warning
                for warning in w
                if "tensor_volatility_field" in str(warning.message)
                and not issubclass(warning.category, DeprecationWarning)
            ]
            assert len(tensor_warnings) == 0, "Should not warn for spatially-varying diagonal tensor"

        # Verify solution
        assert U_solution.shape == (Nt_points, Nx, Ny)
        assert not np.any(np.isnan(U_solution))

    def test_diagonal_tensor_callable(self):
        """Test HJB solver with callable diagonal tensor Σ(t, x, m)."""
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 0.6)], Nx_points=[11, 9], boundary_conditions=no_flux_bc(dimension=2)
        )
        problem = MFGProblem(geometry=domain, T=0.05, Nt=3, sigma=0.1, components=_default_components_2d())

        solver = HJBFDMSolver(problem, solver_type="newton")

        # Get grid shape
        Nx, Ny = domain.get_grid_shape()
        Nt_points = problem.Nt_points

        # Create dummy density and initial conditions
        M_density = np.ones((Nt_points, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt_points, Nx, Ny))

        # Callable diagonal tensor that varies with density
        def tensor_func(t, x, m):
            sigma_x = 0.15
            sigma_y = 0.05 * (1.0 + 0.5 * m)  # Increases with density
            return np.diag([sigma_x**2, sigma_y**2])

        # Should warn for callable (can't check if diagonal without evaluation)
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            U_solution = solver.solve_hjb_system(M_density, U_final, U_prev, tensor_volatility_field=tensor_func)

            # Callable tensors should produce a warning
            tensor_warnings = [warning for warning in w if "callable" in str(warning.message).lower()]
            assert len(tensor_warnings) > 0, "Should warn for callable tensor"

        # Verify solution
        assert U_solution.shape == (Nt_points, Nx, Ny)
        assert not np.any(np.isnan(U_solution))

    def test_diagonal_tensor_mutual_exclusivity(self):
        """Test that tensor_volatility_field and volatility_field are mutually exclusive."""
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 0.6)], Nx_points=[11, 9], boundary_conditions=no_flux_bc(dimension=2)
        )
        problem = MFGProblem(geometry=domain, T=0.05, Nt=3, sigma=0.1, components=_default_components_2d())

        solver = HJBFDMSolver(problem, solver_type="newton")

        # Get grid shape
        Nx, Ny = domain.get_grid_shape()
        Nt_points = problem.Nt_points

        # Create dummy density and initial conditions
        M_density = np.ones((Nt_points, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt_points, Nx, Ny))

        # Both tensor and scalar diffusion
        Sigma = np.diag([0.15, 0.05])
        sigma_scalar = 0.2

        # Should raise ValueError
        with pytest.raises(ValueError, match="Cannot specify both"):
            solver.solve_hjb_system(
                M_density, U_final, U_prev, volatility_field=sigma_scalar, tensor_volatility_field=Sigma
            )


class TestHJBFDMSolverGhostValueBC:
    """Test HJB FDM solver with ghost value boundary conditions (Issue #494)."""

    def test_get_ghost_values_nd_dirichlet(self):
        """Test get_ghost_values_nd for Dirichlet BC."""
        from mfgarchon.geometry.boundary import dirichlet_bc, get_ghost_values_nd

        # Create 2D field
        field = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        bc = dirichlet_bc(dimension=2, value=0.0)
        spacing = (0.1, 0.1)

        ghosts = get_ghost_values_nd(field, bc, spacing)

        # Check all boundaries have ghost values
        assert (0, 0) in ghosts  # x_min
        assert (0, 1) in ghosts  # x_max
        assert (1, 0) in ghosts  # y_min
        assert (1, 1) in ghosts  # y_max

        # For Dirichlet g=0, cell-centered: u_ghost = 2*0 - u_interior = -u_interior
        # Left boundary (axis 0): interior values are [1, 2, 3]
        np.testing.assert_allclose(ghosts[(0, 0)], -np.array([1.0, 2.0, 3.0]))
        # Right boundary (axis 0): interior values are [4, 5, 6]
        np.testing.assert_allclose(ghosts[(0, 1)], -np.array([4.0, 5.0, 6.0]))

    def test_get_ghost_values_nd_neumann(self):
        """Test get_ghost_values_nd for Neumann/no-flux BC."""
        from mfgarchon.geometry.boundary import get_ghost_values_nd, no_flux_bc

        # Create 2D field
        field = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        bc = no_flux_bc(dimension=2)
        spacing = (0.1, 0.1)

        ghosts = get_ghost_values_nd(field, bc, spacing)

        # For no-flux (Neumann g=0): use reflection formula (Issue #542 fix)
        # ghost[k] = interior[g-1-k] where g=1
        # Left boundary (axis 0): ghost = field[1, :] (reflection of next interior)
        np.testing.assert_allclose(ghosts[(0, 0)], np.array([4.0, 5.0, 6.0]))
        # Right boundary (axis 0): ghost = field[0, :] (reflection of next interior)
        np.testing.assert_allclose(ghosts[(0, 1)], np.array([1.0, 2.0, 3.0]))

    def test_get_ghost_values_nd_periodic(self):
        """Test get_ghost_values_nd for periodic BC."""
        from mfgarchon.geometry.boundary import get_ghost_values_nd, periodic_bc

        # Create 2D field
        field = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        bc = periodic_bc(dimension=2)
        spacing = (0.1, 0.1)

        ghosts = get_ghost_values_nd(field, bc, spacing)

        # For periodic: ghost = opposite boundary value
        # Left ghost (axis 0) = right interior [4, 5, 6]
        np.testing.assert_allclose(ghosts[(0, 0)], np.array([4.0, 5.0, 6.0]))
        # Right ghost (axis 0) = left interior [1, 2, 3]
        np.testing.assert_allclose(ghosts[(0, 1)], np.array([1.0, 2.0, 3.0]))

    def test_hjb_solver_uses_ghost_values_with_geometry(self):
        """Test that HJB solver uses ghost values when geometry has BCs."""
        from mfgarchon.geometry.boundary import no_flux_bc

        # Create problem with geometry that has BC
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 10], boundary_conditions=no_flux_bc(dimension=2)
        )

        problem = MFGProblem(geometry=domain, T=0.1, Nt=5, sigma=0.1, components=_default_components_2d())
        solver = HJBFDMSolver(problem, solver_type="fixed_point", advection_scheme="gradient_upwind")

        # Get grid shape
        Nx, Ny = domain.get_grid_shape()
        Nt_points = problem.Nt_points

        # Create density and initial conditions
        M_density = np.ones((Nt_points, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt_points, Nx, Ny))

        # Solve - should use ghost values for gradient computation at boundaries
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Verify solution is valid
        assert U_solution.shape == (Nt_points, Nx, Ny)
        assert np.all(np.isfinite(U_solution))

    def test_hjb_gradient_computation_with_ghost_values(self):
        """Test that _get_ghost_values integrates correctly in gradient computation."""
        from mfgarchon.geometry.boundary import dirichlet_bc

        # Create 2D problem with Dirichlet BC
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[8, 8], boundary_conditions=dirichlet_bc(dimension=2, value=0.0)
        )

        problem = MFGProblem(geometry=domain, T=0.1, Nt=3, sigma=0.1, components=_default_components_2d())
        solver = HJBFDMSolver(problem, solver_type="fixed_point", advection_scheme="gradient_upwind")

        # Test gradient computation directly
        Nx, Ny = domain.get_grid_shape()
        U_test = np.random.rand(Nx, Ny)

        # Call _compute_gradients_nd which should use ghost values
        gradients = solver._compute_gradients_nd(U_test)

        # Check gradient structure
        assert 0 in gradients  # x gradient
        assert 1 in gradients  # y gradient
        assert -1 in gradients  # function value

        # Verify gradient shapes
        assert gradients[0].shape == U_test.shape
        assert gradients[1].shape == U_test.shape

        # Verify gradients are finite
        assert np.all(np.isfinite(gradients[0]))
        assert np.all(np.isfinite(gradients[1]))

    def test_hjb_solver_centered_scheme_with_ghost_values(self):
        """Test centered scheme gradient computation with ghost values."""
        from mfgarchon.geometry.boundary import no_flux_bc

        # Create 2D problem
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[10, 10], boundary_conditions=no_flux_bc(dimension=2)
        )

        problem = MFGProblem(geometry=domain, T=0.1, Nt=3, sigma=0.1, components=_default_components_2d())
        # Use centered scheme (non-upwind)
        solver = HJBFDMSolver(problem, solver_type="fixed_point", advection_scheme="gradient_centered")

        Nx, Ny = domain.get_grid_shape()
        Nt_points = problem.Nt_points

        M_density = np.ones((Nt_points, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt_points, Nx, Ny))

        # Should work without issues
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert U_solution.shape == (Nt_points, Nx, Ny)
        assert np.all(np.isfinite(U_solution))

    def test_time_varying_dirichlet_bc(self):
        """Test that time-varying Dirichlet BC receives correct time (Issue #496)."""
        from mfgarchon.geometry.boundary import dirichlet_bc, get_ghost_values_nd

        # Track times when BC is called
        call_times = []

        def time_varying_value(t):
            call_times.append(t)
            return np.sin(2 * np.pi * t)

        # Create BC with time-varying value
        bc = dirichlet_bc(dimension=2, value=time_varying_value)

        # Create test field
        field = np.array([[1.0, 2.0], [3.0, 4.0]])
        spacing = (0.1, 0.1)

        # Call at different times
        for t in [0.0, 0.25, 0.5, 0.75]:
            call_times.clear()
            _ = get_ghost_values_nd(field, bc, spacing, time=t)

            # Verify the function was called with correct time
            # (Called once per dimension for Dirichlet)
            assert len(call_times) == 2, f"Expected 2 calls at t={t}, got {len(call_times)}"
            for call_t in call_times:
                assert call_t == t, f"Expected time {t}, got {call_t}"

    def test_hjb_solver_time_varying_bc(self):
        """Test HJB solver with time-varying Dirichlet BC (Issue #496)."""
        from mfgarchon.geometry.boundary import dirichlet_bc

        # Track times when BC is called
        call_times = []

        def time_varying_value(t):
            call_times.append(t)
            return 0.0  # Simple value, we just want to track times

        # Create 2D problem with time-varying BC
        domain = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[8, 8],
            boundary_conditions=dirichlet_bc(dimension=2, value=time_varying_value),
        )

        T = 0.4
        Nt = 4
        problem = MFGProblem(geometry=domain, T=T, Nt=Nt, sigma=0.1, components=_default_components_2d())
        solver = HJBFDMSolver(problem, solver_type="fixed_point", advection_scheme="gradient_upwind")

        Nx, Ny = domain.get_grid_shape()
        n_time_points = Nt + 1

        M_density = np.ones((n_time_points, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((n_time_points, Nx, Ny))

        # Clear call times and solve
        call_times.clear()
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Verify solution is valid
        assert U_solution.shape == (n_time_points, Nx, Ny)
        assert np.all(np.isfinite(U_solution))

        # Verify that BC was called at different times (not always t=0)
        # HJB solves backward, so times should be n*dt for n in [Nt-1, ..., 0]
        # Check that we have calls at different times
        unique_times = sorted(set(call_times))
        assert len(unique_times) >= 2, (
            f"Expected BC calls at multiple times, got times: {unique_times}. "
            "This suggests time is not being passed correctly to BC."
        )


class TestHJBFDMSolverNewtonFallback:
    """Test Newton-to-Value-Iteration adaptive fallback (Issue #669)."""

    def test_on_newton_failure_default_is_raise(self):
        """Test that default on_newton_failure is 'raise'."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=10, components=_default_components())
        solver = HJBFDMSolver(problem)

        assert solver.on_newton_failure == "raise"

    def test_on_newton_failure_raise_on_divergence(self):
        """Test that ConvergenceError is raised when Newton diverges with on_newton_failure='raise'."""
        from unittest.mock import patch

        from mfgarchon.alg.numerical.hjb_solvers import ConvergenceError
        from mfgarchon.utils.numerical import SolverInfo

        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[8, 8], boundary_conditions=no_flux_bc(dimension=2)
        )
        problem = MFGProblem(geometry=geometry, T=0.1, Nt=2, sigma=0.1, components=_default_components_2d())
        solver = HJBFDMSolver(problem, solver_type="newton", on_newton_failure="raise")

        Nx, Ny = 8, 8
        Nt_points = problem.Nt_points
        M_density = np.ones((Nt_points, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt_points, Nx, Ny))

        # Mock the nonlinear solver to return non-converged result
        failed_info = SolverInfo(converged=False, iterations=30, residual=1.5, residual_history=[])
        with (
            patch.object(solver.nonlinear_solver, "solve", return_value=(np.zeros((8, 8)), failed_info)),
            pytest.raises(ConvergenceError, match="newton solver failed to converge"),
        ):
            solver.solve_hjb_system(M_density, U_final, U_prev)

    def test_on_newton_failure_raise_on_exception(self):
        """Test that ConvergenceError wraps LinAlgError when on_newton_failure='raise'."""
        from unittest.mock import patch

        from mfgarchon.alg.numerical.hjb_solvers import ConvergenceError

        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[8, 8], boundary_conditions=no_flux_bc(dimension=2)
        )
        problem = MFGProblem(geometry=geometry, T=0.1, Nt=2, sigma=0.1, components=_default_components_2d())
        solver = HJBFDMSolver(problem, solver_type="newton", on_newton_failure="raise")

        Nx, Ny = 8, 8
        Nt_points = problem.Nt_points
        M_density = np.ones((Nt_points, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt_points, Nx, Ny))

        # Mock the nonlinear solver to raise LinAlgError (singular Jacobian)
        with (
            patch.object(solver.nonlinear_solver, "solve", side_effect=np.linalg.LinAlgError("Singular matrix")),
            pytest.raises(ConvergenceError, match=r"Newton solver failed.*Singular matrix"),
        ):
            solver.solve_hjb_system(M_density, U_final, U_prev)

    def test_on_newton_failure_fallback_on_divergence(self):
        """Test fallback to Value Iteration when Newton diverges with warn_and_fallback."""
        import warnings
        from unittest.mock import patch

        from mfgarchon.utils.numerical import SolverInfo

        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[8, 8], boundary_conditions=no_flux_bc(dimension=2)
        )
        problem = MFGProblem(geometry=geometry, T=0.1, Nt=2, sigma=0.1, components=_default_components_2d())
        solver = HJBFDMSolver(problem, solver_type="newton", on_newton_failure="warn_and_fallback")

        # Mock Newton solver to return non-converged result
        failed_info = SolverInfo(converged=False, iterations=30, residual=1.5, residual_history=[])

        # Create a successful fallback result
        success_info = SolverInfo(converged=True, iterations=50, residual=1e-7, residual_history=[])
        fallback_result = (np.zeros((8, 8)), success_info)

        Nx, Ny = 8, 8
        Nt_points = problem.Nt_points
        M_density = np.ones((Nt_points, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt_points, Nx, Ny))

        with (
            patch.object(solver.nonlinear_solver, "solve", return_value=(np.zeros((8, 8)), failed_info)),
            patch("mfgarchon.alg.numerical.hjb_solvers.hjb_fdm.FixedPointSolver") as MockFPSolver,
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            mock_fp_instance = MockFPSolver.return_value
            mock_fp_instance.solve.return_value = fallback_result

            solver.solve_hjb_system(M_density, U_final, U_prev)

        # Check that fallback warning was emitted
        fallback_warnings = [x for x in w if "Falling back to Value Iteration" in str(x.message)]
        assert len(fallback_warnings) >= 1, f"Expected fallback warning, got: {[str(x.message) for x in w]}"

        # Verify FixedPointSolver was constructed with 10x iterations
        MockFPSolver.assert_called()
        call_kwargs = MockFPSolver.call_args[1]
        assert call_kwargs["max_iterations"] == solver.max_newton_iterations * 10

    def test_on_newton_failure_ignored_for_fixed_point(self):
        """Test that on_newton_failure is irrelevant when solver_type='fixed_point'."""
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[8, 8], boundary_conditions=no_flux_bc(dimension=2)
        )
        problem = MFGProblem(geometry=geometry, T=0.1, Nt=2, sigma=0.1, components=_default_components_2d())

        # Should accept the parameter without error even for fixed_point
        solver = HJBFDMSolver(problem, solver_type="fixed_point", on_newton_failure="warn_and_fallback")
        assert solver.on_newton_failure == "warn_and_fallback"

        # Solving should work normally (no fallback logic triggered)
        Nx, Ny = 8, 8
        Nt_points = problem.Nt_points
        M_density = np.ones((Nt_points, Nx, Ny)) * 0.5
        U_final = np.zeros((Nx, Ny))
        U_prev = np.zeros((Nt_points, Nx, Ny))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)
        assert U_solution.shape == (Nt_points, Nx, Ny)
        assert np.all(np.isfinite(U_solution))

    def test_convergence_error_attributes(self):
        """Test ConvergenceError exception has correct attributes."""
        from mfgarchon.alg.numerical.hjb_solvers import ConvergenceError

        err = ConvergenceError("newton", iterations=42, residual=3.14)
        assert err.solver_type == "newton"
        assert err.iterations == 42
        assert err.residual == 3.14
        assert "newton" in str(err)
        assert "42" in str(err)
        assert "3.14" in str(err)

    def test_convergence_error_custom_message(self):
        """Test ConvergenceError with custom message."""
        from mfgarchon.alg.numerical.hjb_solvers import ConvergenceError

        err = ConvergenceError("newton", 10, 1e-2, message="Custom failure message")
        assert str(err) == "Custom failure message"
        assert err.solver_type == "newton"


class TestBoundaryGradientBCAware1384:
    """Issue #1384: the default single-population 1D HJB must use the BC-aware boundary
    gradient (residual AND Jacobian), not the periodic %Nx wraparound.

    Coverage gap these tests close: the prior suite's only non-periodic FDM solve was a
    fully symmetric no_flux problem, for which the periodic and BC-aware boundary gradients
    coincide — so the bug (and the residual/Jacobian boundary inconsistency it would expose)
    was invisible. These exercise (a) asymmetric no_flux, where the fix changes the result
    toward BC-respecting, (b) periodic, which must stay byte-identical, and (c) Dirichlet
    with a steep terminal, which diverged when only the residual was made BC-aware.
    """

    def _solve(self, bc, *, Nx=42, sigma=0.3, Nt=20, T=0.5, terminal):
        components = MFGComponents(
            m_initial=lambda x: 1.0,
            u_terminal=terminal,
            hamiltonian=_default_hamiltonian(),
        )
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[Nx], boundary_conditions=bc)
        problem = MFGProblem(geometry=geometry, T=T, Nt=Nt, sigma=sigma, components=components)
        solver = HJBFDMSolver(problem)
        Nx_points = problem.geometry.get_grid_shape()[0]
        Nt_points = problem.Nt_points
        x = problem.geometry.get_spatial_grid().ravel()
        M = np.ones((Nt_points, Nx_points)) / Nx_points
        U_T = terminal(x)
        return solver.solve_hjb_system(M, U_T, np.tile(U_T, (Nt_points, 1)))

    @staticmethod
    def _legacy_periodic_grad(U, dx):
        """The pre-fix per-point gradient (legacy %Nx wraparound), for discrimination."""
        from mfgarchon.alg.numerical.hjb_solvers import base_hjb

        Nx = len(U)
        return np.array(
            [
                base_hjb._calculate_derivatives(U, i, dx, Nx, upwind=False, precomputed_gradient=None)[(1,)]
                for i in range(Nx)
            ]
        )

    def test_boundary_gradient_is_bc_aware_not_periodic(self):
        """Mechanism: for non-periodic BC the boundary momentum differs materially from the
        legacy %Nx periodic stencil, while the interior agrees to <=1 ULP. This is exactly
        the #1384 corruption; the test fails if the boundary gradient is not BC-aware."""
        from mfgarchon.alg.numerical.hjb_solvers import base_hjb
        from mfgarchon.geometry.boundary import no_flux_bc as _nf

        Nx = 21
        x = np.linspace(0.0, 1.0, Nx)
        dx = x[1] - x[0]
        U = x**2  # asymmetric on [0,1]; analytic du/dx = 2x (0 at left, 2 at right)
        legacy = self._legacy_periodic_grad(U, dx)
        bc_aware = base_hjb._compute_gradient_array_1d(U, dx, bc=_nf(dimension=1), upwind=False)
        # interior agrees to <=1 ULP (only the boundary stencil changes)
        assert np.allclose(bc_aware[1:-1], legacy[1:-1], rtol=0, atol=1e-12)
        # boundary is materially different: legacy wraps to the opposite end (~ -9.97)
        assert abs(bc_aware[0] - legacy[0]) > 1.0
        assert abs(bc_aware[-1] - legacy[-1]) > 1.0

    def test_periodic_gradient_byte_identical_to_legacy(self):
        """Mechanism: for PERIODIC BC the BC-aware gradient equals the legacy %Nx stencil
        (Δ <= 1 ULP everywhere) — the discrimination the fix relies on, so periodic
        baselines do not move."""
        from mfgarchon.alg.numerical.hjb_solvers import base_hjb
        from mfgarchon.geometry.boundary import periodic_bc as _pb

        Nx = 21
        x = np.linspace(0.0, 1.0, Nx)
        dx = x[1] - x[0]
        U = np.sin(2 * np.pi * x)
        legacy = self._legacy_periodic_grad(U, dx)
        bc_aware = base_hjb._compute_gradient_array_1d(U, dx, bc=_pb(dimension=1), upwind=False)
        assert np.max(np.abs(bc_aware - legacy)) <= 1e-12

    def test_dirichlet_steep_terminal_converges(self):
        """Dirichlet with a steep terminal: making only the residual BC-aware while the FD
        Jacobian stayed periodic made Newton diverge (J != dF/dU at the pinned boundary).
        With the Jacobian also BC-aware the solve stays finite and bounded."""
        from mfgarchon.geometry.boundary import dirichlet_bc as _db

        U = self._solve(_db(0.0, dimension=1), Nx=42, sigma=0.05, Nt=3, T=2.0, terminal=lambda x: np.sin(3 * np.pi * x))
        assert np.all(np.isfinite(U)), "Dirichlet steep-terminal solve produced NaN/Inf"
        assert np.max(np.abs(U)) < 10.0, "Dirichlet solve blew up (Newton J/residual boundary mismatch)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

#!/usr/bin/env python3
"""
Integration tests for 2D HJB FDM solver validation.

Tests the dimension-agnostic HJB FDM solver on 2D problems with:
- Both fixed-point and Newton solver modes
- Convergence studies
- Comparison between solver types
- Mass-conserving density evolution

NOTE: These tests use MFGProblem with spatial_bounds for 2D support.
API consistency improvements tracked in Issue #277.
"""

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents


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
        """Gaussian centered at origin for 2D."""
        x_arr = np.asarray(x)
        return np.exp(-5 * np.sum(x_arr**2))

    def u_final_2d(x):
        """Quadratic terminal cost for 2D."""
        x_arr = np.asarray(x)
        return 0.5 * np.sum(x_arr**2)

    return MFGComponents(
        m_initial=m_initial_2d,
        u_terminal=u_final_2d,
        hamiltonian=_default_hamiltonian(),
    )


class QuadraticHamiltonian2D(MFGProblem):
    """
    2D LQ problem with quadratic Hamiltonian.

    Hamiltonian: H(x, p, m) = 0.5·|p|² + 0.5·|x|²

    This has a simple structure that should converge well.
    """

    def __init__(self, N=20, T=1.0, Nt=20, nu=0.01):
        super().__init__(
            spatial_bounds=[(-1, 1), (-1, 1)],
            spatial_discretization=[N, N],
            T=T,
            Nt=Nt,
            sigma=nu,
            components=_default_components_2d(),
        )
        self.grid_resolution = N

    def hamiltonian(self, x, m, p, t):
        """Quadratic Hamiltonian: H = 0.5·|p|² + 0.5·|x|²"""
        return 0.5 * np.sum(p**2) + 0.5 * np.sum(x**2)

    def terminal_cost(self, x):
        """Terminal cost: g(x) = 0.5·|x|²"""
        return 0.5 * np.sum(x**2)

    def initial_density(self, x):
        """Initial density: Gaussian centered at origin"""
        return np.exp(-5 * np.sum(x**2))

    def running_cost(self, x, m, t):
        """Running cost: f(x,m) = 0.5·|x|²"""
        return 0.5 * np.sum(x**2)

    def setup_components(self):
        """No additional setup needed."""


class TestHJBFDM2DBasic:
    """Basic 2D HJB FDM functionality tests."""

    def test_2d_initialization_fixed_point(self):
        """Test that 2D solver initializes correctly in fixed-point mode."""
        problem = QuadraticHamiltonian2D(N=10, T=0.5, Nt=10)
        solver = HJBFDMSolver(problem, solver_type="fixed_point", damping_factor=0.7)

        assert solver.dimension == 2
        assert solver.hjb_method_name == "FDM-2D-fixed_point"
        assert solver.shape == (11, 11)  # N+1 grid points in each dimension
        assert solver.solver_type == "fixed_point"

    def test_2d_initialization_newton(self):
        """Test that 2D solver initializes correctly in Newton mode."""
        problem = QuadraticHamiltonian2D(N=10, T=0.5, Nt=10)
        solver = HJBFDMSolver(problem, solver_type="newton")

        assert solver.dimension == 2
        assert solver.hjb_method_name == "FDM-2D-newton"
        assert solver.shape == (11, 11)  # N+1 grid points in each dimension
        assert solver.solver_type == "newton"

    def test_2d_gradient_computation(self):
        """Test gradient computation in 2D."""
        problem = QuadraticHamiltonian2D(N=10, T=0.5, Nt=10)
        solver = HJBFDMSolver(problem, solver_type="newton")

        # Create test function: u(x,y) = x² + y²
        # N=10 gives N+1=11 grid points
        x = np.linspace(-1, 1, 11)
        y = np.linspace(-1, 1, 11)
        X, Y = np.meshgrid(x, y, indexing="ij")
        U = X**2 + Y**2

        # Compute gradients
        gradients = solver._compute_gradients_nd(U)

        # Check gradient keys (new format uses integer dimension indices)
        # Key -1 = function value U, Key 0 = ∂u/∂x₀, Key 1 = ∂u/∂x₁
        assert -1 in gradients  # Value
        assert 0 in gradients  # ∂u/∂x
        assert 1 in gradients  # ∂u/∂y

        # Check gradient values (should be approximately 2x and 2y)
        grad_x = gradients[0]
        grad_y = gradients[1]

        # Interior points should have gradients ≈ 2x, 2y
        interior_x = grad_x[5, 5]
        interior_y = grad_y[5, 5]
        expected_x = 2 * X[5, 5]
        expected_y = 2 * Y[5, 5]

        assert np.abs(interior_x - expected_x) < 0.5  # Loose check due to discretization
        assert np.abs(interior_y - expected_y) < 0.5


class TestHJBFDM2DSolving:
    """Test 2D HJB solving with both solver types."""

    def test_2d_solve_fixed_point(self):
        """Test solving 2D HJB with fixed-point solver."""
        problem = QuadraticHamiltonian2D(N=15, T=0.5, Nt=10)
        solver = HJBFDMSolver(
            problem,
            solver_type="fixed_point",
            damping_factor=0.8,
            max_newton_iterations=50,
            newton_tolerance=1e-5,
        )

        # N=15 gives N+1=16 grid points, Nt=10 gives 11 time steps
        # Create test density (uniform)
        M_test = np.ones((11, 16, 16)) / 256  # Normalized uniform density

        # Terminal condition
        x = np.linspace(-1, 1, 16)
        y = np.linspace(-1, 1, 16)
        X, Y = np.meshgrid(x, y, indexing="ij")
        U_terminal = 0.5 * (X**2 + Y**2)

        # Initial guess
        U_guess = np.zeros((11, 16, 16))

        # Solve
        U_solution = solver.solve_hjb_system(M_test, U_terminal, U_guess)

        # Check output shape
        assert U_solution.shape == (11, 16, 16)

        # Check terminal condition preserved
        assert np.allclose(U_solution[10], U_terminal, atol=1e-10)

        # Check finiteness
        assert np.all(np.isfinite(U_solution))

        # Value function should be non-negative for this problem
        assert np.all(U_solution >= -1e-6)  # Small tolerance for numerical errors

    @pytest.mark.slow
    def test_2d_solve_newton(self):
        """Test solving 2D HJB with Newton solver."""
        problem = QuadraticHamiltonian2D(N=15, T=0.5, Nt=10)
        solver = HJBFDMSolver(problem, solver_type="newton", max_newton_iterations=30, newton_tolerance=1e-5)

        # N=15 gives N+1=16 grid points, Nt=10 gives 11 time steps
        # Create test density (uniform)
        M_test = np.ones((11, 16, 16)) / 256

        # Terminal condition
        x = np.linspace(-1, 1, 16)
        y = np.linspace(-1, 1, 16)
        X, Y = np.meshgrid(x, y, indexing="ij")
        U_terminal = 0.5 * (X**2 + Y**2)

        # Initial guess
        U_guess = np.zeros((11, 16, 16))

        # Solve
        U_solution = solver.solve_hjb_system(M_test, U_terminal, U_guess)

        # Check output shape
        assert U_solution.shape == (11, 16, 16)

        # Check terminal condition preserved
        assert np.allclose(U_solution[10], U_terminal, atol=1e-10)

        # Check finiteness
        assert np.all(np.isfinite(U_solution))

        # Value function should be non-negative
        assert np.all(U_solution >= -1e-6)


class TestHJBFDM2DConvergence:
    """Test convergence properties of 2D solver."""

    @pytest.mark.slow
    def test_fixed_point_vs_newton_consistency(self):
        """Test that fixed-point and Newton give similar results."""
        problem = QuadraticHamiltonian2D(N=12, T=0.3, Nt=8)

        # Solve with fixed-point
        solver_fp = HJBFDMSolver(
            problem,
            solver_type="fixed_point",
            damping_factor=0.8,
            max_newton_iterations=100,
            newton_tolerance=1e-6,
        )

        # Solve with Newton
        solver_newton = HJBFDMSolver(problem, solver_type="newton", max_newton_iterations=30, newton_tolerance=1e-6)

        # N=12 gives 13 grid points, Nt=8 gives 9 time steps
        # Create test inputs
        M_test = np.ones((9, 13, 13)) / 169
        x = np.linspace(-1, 1, 13)
        y = np.linspace(-1, 1, 13)
        X, Y = np.meshgrid(x, y, indexing="ij")
        U_terminal = 0.5 * (X**2 + Y**2)
        U_guess = np.zeros((9, 13, 13))

        # Solve both
        U_fp = solver_fp.solve_hjb_system(M_test, U_terminal, U_guess)
        U_newton = solver_newton.solve_hjb_system(M_test, U_terminal, U_guess)

        # Solutions should be close (not identical due to different convergence paths)
        rel_error = np.linalg.norm(U_fp - U_newton) / np.linalg.norm(U_newton)
        assert rel_error < 0.1  # Within 10% (loose check, different solvers)

        # Both should satisfy terminal condition
        assert np.allclose(U_fp[8], U_terminal, atol=1e-6)
        assert np.allclose(U_newton[8], U_terminal, atol=1e-6)

    @pytest.mark.slow
    def test_spatial_convergence(self):
        """Test that solution converges with grid refinement."""
        T, Nt = 0.3, 10

        # Solve on coarse grid
        # N=10 gives 11 grid points
        problem_coarse = QuadraticHamiltonian2D(N=10, T=T, Nt=Nt)
        solver_coarse = HJBFDMSolver(
            problem_coarse, solver_type="newton", max_newton_iterations=30, newton_tolerance=1e-6
        )

        M_coarse = np.ones((11, 11, 11)) / 121
        x_c = np.linspace(-1, 1, 11)
        X_c, Y_c = np.meshgrid(x_c, x_c, indexing="ij")
        U_terminal_coarse = 0.5 * (X_c**2 + Y_c**2)
        U_guess_coarse = np.zeros((11, 11, 11))

        U_coarse = solver_coarse.solve_hjb_system(M_coarse, U_terminal_coarse, U_guess_coarse)

        # Solve on fine grid
        # N=20 gives 21 grid points
        problem_fine = QuadraticHamiltonian2D(N=20, T=T, Nt=Nt)
        solver_fine = HJBFDMSolver(problem_fine, solver_type="newton", max_newton_iterations=30, newton_tolerance=1e-6)

        M_fine = np.ones((11, 21, 21)) / 441
        x_f = np.linspace(-1, 1, 21)
        X_f, Y_f = np.meshgrid(x_f, x_f, indexing="ij")
        U_terminal_fine = 0.5 * (X_f**2 + Y_f**2)
        U_guess_fine = np.zeros((11, 21, 21))

        U_fine = solver_fine.solve_hjb_system(M_fine, U_terminal_fine, U_guess_fine)

        # Interpolate coarse solution to fine grid for comparison
        from scipy.interpolate import RegularGridInterpolator

        # Compare at initial time
        interpolator = RegularGridInterpolator(
            (x_c, x_c), U_coarse[0], method="linear", bounds_error=False, fill_value=None
        )
        points_fine = np.array([[x, y] for x in x_f for y in x_f])
        U_coarse_interp = interpolator(points_fine).reshape(21, 21)

        # Fine grid should be more accurate (lower values typically for this problem)
        # Just check that they're reasonably close
        rel_diff = np.linalg.norm(U_fine[0] - U_coarse_interp) / np.linalg.norm(U_fine[0])
        assert rel_diff < 0.5  # Should converge with refinement


class TestHJBFDM2DPhysicalProperties:
    """Test physical properties of 2D solutions."""

    @pytest.mark.slow
    def test_monotonicity_in_time(self):
        """At the symmetric centre the value function decreases forward in time.

        grad u = 0 there, so the HJB reads u_t = -D Lap u - f(m) with f(m) = m > 0 a cost (#2375 ruling
        3): the running cost accumulates backward, and the convex terminal adds to it through diffusion.
        """
        # N=12 gives 13 grid points
        problem = QuadraticHamiltonian2D(N=12, T=0.5, Nt=10)
        solver = HJBFDMSolver(problem, solver_type="newton")

        M_test = np.ones((11, 13, 13)) / 169
        x = np.linspace(-1, 1, 13)
        X, Y = np.meshgrid(x, x, indexing="ij")
        U_terminal = 0.5 * (X**2 + Y**2)
        U_guess = np.zeros((11, 13, 13))

        U_solution = solver.solve_hjb_system(M_test, U_terminal, U_guess)

        center_values = U_solution[:, 6, 6]
        diffs = np.diff(center_values)
        # Measured: all 10 steps lie in [-3.008e-4, -3.003e-4], against -(f + D Lap U_T) * dt = -3.009e-4
        # with f = 1/169, D = 5e-5, Lap U_T = 2 and dt = 0.05.
        assert np.all(diffs < 0), f"the centre value must fall forward in time; steps {diffs}"

    @pytest.mark.slow
    def test_symmetry(self):
        """Test that solution respects problem symmetry."""
        # N=20 gives 21 grid points
        problem = QuadraticHamiltonian2D(N=20, T=0.3, Nt=10)
        solver = HJBFDMSolver(problem, solver_type="newton")

        # Symmetric initial density
        x = np.linspace(-1, 1, 21)
        X, Y = np.meshgrid(x, x, indexing="ij")
        m_init_2d = np.exp(-5 * (X**2 + Y**2))
        m_init_2d /= np.trapezoid(np.trapezoid(m_init_2d, x), x)

        M_test = np.tile(m_init_2d, (11, 1, 1))

        # Symmetric terminal condition
        U_terminal = 0.5 * (X**2 + Y**2)
        U_guess = np.zeros((11, 21, 21))

        U_solution = solver.solve_hjb_system(M_test, U_terminal, U_guess)

        # Solution should be symmetric about x=0 and y=0
        # Check at initial time
        U_0 = U_solution[0]

        # Check x-symmetry: U(x,y) ≈ U(-x,y)
        x_sym_error = np.linalg.norm(U_0 - np.flip(U_0, axis=0)) / np.linalg.norm(U_0)
        assert x_sym_error < 0.05  # Within 5%

        # Check y-symmetry: U(x,y) ≈ U(x,-y)
        y_sym_error = np.linalg.norm(U_0 - np.flip(U_0, axis=1)) / np.linalg.norm(U_0)
        assert y_sym_error < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

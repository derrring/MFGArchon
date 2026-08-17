"""
Integration tests for HJB solvers with obstacle constraints.

Tests that HJB FDM solver correctly handles ObstacleConstraint (Tier 2 BCs)
through complete solve workflows.

Created: 2026-01-18 (Issue #594 Phase 5.3)
"""

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BilateralConstraint, ObstacleConstraint, neumann_bc, no_flux_bc


def _default_hamiltonian():
    """Default Hamiltonian for testing."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components():
    """Default MFGComponents for testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        m_initial=lambda x: np.exp(
            -10 * (np.asarray(x) - 0.5) ** 2 if np.ndim(x) == 0 else -10 * np.sum((np.asarray(x) - 0.5) ** 2)
        ),
        u_terminal=lambda x: 0.0,
        hamiltonian=_default_hamiltonian(),
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


@pytest.mark.slow
class TestHJBWithLowerObstacle:
    """Test HJB solver with lower obstacle constraint (u ≥ ψ)."""

    def test_1d_parabolic_obstacle_convergence(self):
        """Test HJB solver converges with parabolic obstacle."""
        # Setup
        x_min, x_max = 0.0, 1.0
        Nx = 100
        T = 1.0
        Nt = 50
        sigma = 0.1
        kappa = 0.5

        grid = TensorProductGrid(bounds=[(x_min, x_max)], boundary_conditions=no_flux_bc(dimension=1), Nx=[Nx])

        # Terminal cost function (used locally for computing terminal values)
        def terminal_cost(x_coords):
            return (x_coords[0] - 0.5) ** 2

        # Create MFGProblem with minimal parameters (HJB solver uses explicit inputs)
        problem = MFGProblem(geometry=grid, T=T, Nt=Nt, sigma=sigma, components=_default_components())

        # Obstacle: ψ(x) = -κ(x - 0.5)²
        x = grid.coordinates[0]
        psi = -kappa * (x - 0.5) ** 2
        obstacle = ObstacleConstraint(psi, constraint_type="lower")

        # Solve
        solver = HJBFDMSolver(problem, constraint=obstacle, newton_tolerance=1e-6, max_newton_iterations=100)

        # Setup inputs for solve_hjb_system
        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((Nt_points, Nx_points)) / Nx_points
        U_terminal = terminal_cost(grid.coordinates)
        U_prev = np.zeros((Nt_points, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_terminal, U_prev)

        # Assertions
        assert U_solution.shape == (Nt_points, Nx_points), "Solution has correct shape"
        assert np.all(np.isfinite(U_solution)), "Solution is finite"

        # Check constraint satisfaction
        u_final = U_solution[-1, :]
        assert np.all(u_final >= psi - 1e-8), "Solution must satisfy u ≥ ψ"

        # Check active set exists
        active_set = np.abs(u_final - psi) < 1e-3
        assert np.sum(active_set) > 0, "Active set should be non-empty"

    def test_2d_obstacle_solver_integration(self):
        """Test HJB solver with 2D obstacle."""
        Nx, Ny = 10, 10  # Reduced for speed
        T = 0.3
        Nt = 10  # Reduced for speed
        sigma = 0.05

        grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], boundary_conditions=no_flux_bc(dimension=2), Nx=[Nx, Ny])

        # Terminal cost function (used locally for computing terminal values)
        def terminal_cost_2d(x_coords):
            return (x_coords[0] - 0.5) ** 2 + (x_coords[1] - 0.5) ** 2

        # Create MFGProblem with minimal parameters (HJB solver uses explicit inputs)
        problem = MFGProblem(geometry=grid, T=T, Nt=Nt, sigma=sigma, components=_default_components_2d())

        # Obstacle: Bowl-shaped
        X, Y = grid.meshgrid()
        psi = -0.2 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2)
        obstacle = ObstacleConstraint(psi, constraint_type="lower")  # Keep 2D shape

        # Solve
        solver = HJBFDMSolver(problem, constraint=obstacle, newton_tolerance=1e-5, max_newton_iterations=80)

        # Setup inputs
        Nt_points = problem.Nt_points
        shape = problem.geometry.get_grid_shape()
        M_density = np.ones((Nt_points, *shape)) / np.prod(shape)
        # For 2D, need to use meshgrid to get 2D arrays
        X, Y = grid.meshgrid()
        U_terminal = (X - 0.5) ** 2 + (Y - 0.5) ** 2
        U_prev = np.zeros((Nt_points, *shape))

        U_solution = solver.solve_hjb_system(M_density, U_terminal, U_prev)

        # Assertions
        assert U_solution.shape == (Nt_points, *shape), "2D solution has correct shape"
        assert np.all(np.isfinite(U_solution)), "2D solution is finite"
        # Over the whole sweep, not the terminal slice alone -- that slice is the array the
        # test just built, and the obstacle does not bind there. Measured min(U - psi) = 0.0.
        assert np.all(U_solution >= psi - 1e-12), "2D solution must respect obstacle"

        # The obstacle genuinely binds here: the unconstrained solve on the same inputs dips
        # 3.95e-03 below psi at t = 0, and the constrained one is lifted by exactly that much.
        solver_free = HJBFDMSolver(problem, newton_tolerance=1e-5, max_newton_iterations=80)
        U_free = solver_free.solve_hjb_system(M_density, U_terminal, U_prev)
        u0 = U_solution[0]
        assert np.all(u0 >= U_free[0] - 1e-12), "obstacle must raise the solution, never lower it"
        assert np.max(u0 - U_free[0]) > 1e-3, "obstacle never binds; a broken projection would go unnoticed"

        # Square domain, symmetric terminal cost, radially symmetric obstacle: the solution
        # must be exactly transpose-symmetric. Measured max|u0 - u0.T| = 3.7e-15, which an
        # x/y axis swap anywhere in the 2D assembly would destroy.
        assert np.max(np.abs(u0 - u0.T)) < 1e-12, "isotropic setup must give an x<->y symmetric solution"

    def test_obstacle_without_constraint_comparison(self):
        """Compare solution with and without obstacle constraint."""
        Nx = 80
        T = 0.8
        Nt = 40
        sigma = 0.08
        kappa = 0.3

        grid = TensorProductGrid(bounds=[(0.0, 1.0)], boundary_conditions=no_flux_bc(dimension=1), Nx=[Nx])

        # Terminal cost function (used locally)
        def terminal_cost(x_coords):
            return (x_coords[0] - 0.5) ** 2

        # Create MFGProblem with minimal parameters
        problem = MFGProblem(geometry=grid, T=T, Nt=Nt, sigma=sigma, components=_default_components())

        x = grid.coordinates[0]
        psi = -kappa * (x - 0.5) ** 2

        # Setup inputs
        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((Nt_points, Nx_points)) / Nx_points
        U_terminal = terminal_cost(grid.coordinates)
        U_prev = np.zeros((Nt_points, Nx_points))

        # Solve without obstacle
        solver_free = HJBFDMSolver(problem)
        U_free = solver_free.solve_hjb_system(M_density, U_terminal, U_prev)

        # Solve with obstacle
        obstacle = ObstacleConstraint(psi, constraint_type="lower")
        solver_constrained = HJBFDMSolver(problem, constraint=obstacle)
        U_constrained = solver_constrained.solve_hjb_system(M_density, U_terminal, U_prev)

        # Compare at t = 0. At the terminal index both arrays are the terminal datum the two
        # solvers were handed -- the obstacle does not bind there -- so comparing that slice
        # compares an array with itself.
        u_free = U_free[0, :]
        u_constrained = U_constrained[0, :]

        # Constrained solution should be ≥ free solution (obstacle raises floor)
        assert np.all(u_constrained >= u_free - 1e-12), "Obstacle should raise solution"

        # Positive control, without which the line above is vacuous: the obstacle does bind
        # on this configuration. Measured max(U_constrained[0] - U_free[0]) = 6.59e-03, and
        # the free solution dips 6.59e-03 below psi at t = 0.
        assert np.max(u_constrained - u_free) > 1e-3, "obstacle never binds; a broken projection would go unnoticed"
        assert np.min(u_free - psi) < -1e-3, "free solution already satisfies the obstacle"

        # Constraint satisfaction over the whole sweep, not one slice. Measured min = 0.0.
        assert np.all(U_constrained >= psi - 1e-12), "Constrained solution respects obstacle"


@pytest.mark.slow
class TestHJBWithUpperObstacle:
    """Test HJB solver with upper obstacle constraint (u ≤ ψ_upper)."""

    def test_1d_upper_ceiling(self):
        """Test HJB solver with upper obstacle (ceiling)."""
        Nx = 100
        T = 1.0
        Nt = 50
        sigma = 0.1

        grid = TensorProductGrid(bounds=[(0.0, 1.0)], boundary_conditions=no_flux_bc(dimension=1), Nx=[Nx])

        # Terminal cost function (used locally)
        def terminal_cost(x_coords):
            return (x_coords[0] - 0.5) ** 2

        # Create MFGProblem with minimal parameters
        problem = MFGProblem(geometry=grid, T=T, Nt=Nt, sigma=sigma, components=_default_components())

        # Upper obstacle: a ceiling low enough to bind. The free solution reaches 0.0733 at
        # t = 0, so the original 0.3 ceiling was never touched and the constrained output was
        # byte-identical to the unconstrained one.
        x = grid.coordinates[0]
        psi_upper = 0.05 + 0.0 * x  # Constant ceiling
        obstacle = ObstacleConstraint(psi_upper, constraint_type="upper")

        # Solve
        solver = HJBFDMSolver(problem, constraint=obstacle)

        # Setup inputs
        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((Nt_points, Nx_points)) / Nx_points
        U_terminal = terminal_cost(grid.coordinates)
        U_prev = np.zeros((Nt_points, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_terminal, U_prev)

        # Assertions
        assert U_solution.shape == (Nt_points, Nx_points), "Solution has correct shape"
        assert np.all(np.isfinite(U_solution)), "Solution is finite"

        # The ceiling holds at every time level, the terminal one included: the terminal
        # datum reaches 0.25 and comes back projected to 0.05. Measured sweep max 0.05 exactly.
        assert np.all(U_solution <= psi_upper + 1e-10), "Solution must satisfy u ≤ ψ_upper"

        # The contact set must be non-empty, or the ceiling is decoration: measured 22 of the
        # 101 nodes sitting exactly on it at t = 0.
        u0 = U_solution[0, :]
        assert np.sum(np.abs(u0 - psi_upper) < 1e-9) >= 10, "upper obstacle never binds"

        # Positive control: the unconstrained solve on the same inputs exceeds the ceiling
        # (measured max 0.0733 at t = 0), so this configuration can see a broken projection.
        U_free = HJBFDMSolver(problem).solve_hjb_system(M_density, U_terminal, U_prev)
        assert U_free[0, :].max() > 0.07, "free solution stays under the ceiling on its own"


@pytest.mark.slow
class TestHJBWithBilateralObstacle:
    """Test HJB solver with bilateral obstacle (ψ_lower ≤ u ≤ ψ_upper)."""

    def test_1d_corridor_constraint(self):
        """Test bilateral obstacle creating solution corridor."""
        Nx = 100
        T = 1.0
        Nt = 50
        sigma = 0.1

        grid = TensorProductGrid(bounds=[(0.0, 1.0)], boundary_conditions=no_flux_bc(dimension=1), Nx=[Nx])

        # Terminal cost function (used locally)
        def terminal_cost(x_coords):
            return 0.5 * (x_coords[0] - 0.5) ** 2

        # Create MFGProblem with minimal parameters
        problem = MFGProblem(geometry=grid, T=T, Nt=Nt, sigma=sigma, components=_default_components())

        # Bilateral obstacle: a corridor narrow enough to bind on BOTH faces. The free
        # solution runs over [-0.0063, 0.125] on this configuration, so the original
        # [-0.2, 0.3] corridor contained it entirely: the constrained output was
        # byte-identical to the unconstrained one and the projection never fired.
        x = grid.coordinates[0]
        psi_lower = -0.004 + 0.0 * x
        psi_upper = 0.04 + 0.0 * x
        obstacle = BilateralConstraint(psi_lower, psi_upper)

        # Solve
        solver = HJBFDMSolver(problem, constraint=obstacle)

        # Setup inputs
        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((Nt_points, Nx_points)) / Nx_points
        U_terminal = terminal_cost(grid.coordinates)
        U_prev = np.zeros((Nt_points, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_terminal, U_prev)

        # Assertions
        assert U_solution.shape == (Nt_points, Nx_points), "Solution has correct shape"
        assert np.all(np.isfinite(U_solution)), "Solution is finite"

        # The corridor is enforced at every time level, the terminal one included: the
        # terminal datum reaches 0.125 and comes back projected. Measured violation 0.0 on
        # both faces, sweep range exactly [-0.004, 0.04].
        assert np.all(U_solution >= psi_lower - 1e-10), "Must satisfy lower bound"
        assert np.all(U_solution <= psi_upper + 1e-10), "Must satisfy upper bound"

        # Both faces must carry an active set, or the corridor tests nothing.
        # Measured at t = 0: 19 nodes on the floor, 16 on the ceiling.
        u0 = U_solution[0, :]
        assert np.sum(np.abs(u0 - psi_lower) < 1e-9) > 0, "lower obstacle never binds"
        assert np.sum(np.abs(u0 - psi_upper) < 1e-9) > 0, "upper obstacle never binds"

        # Positive control: on the same inputs without the constraint the solution leaves the
        # corridor on both sides (19 nodes below the floor, 16 above the ceiling at t = 0),
        # so the assertions above separate a working projection from a disabled one.
        U_free = HJBFDMSolver(problem).solve_hjb_system(M_density, U_terminal, U_prev)
        assert np.any(U_free[0, :] < psi_lower), "free solution never leaves the corridor below"
        assert np.any(U_free[0, :] > psi_upper), "free solution never leaves the corridor above"


class TestObstacleConvergenceProperties:
    """Test convergence properties with obstacles."""

    def test_tolerance_scaling(self):
        """Test that tighter tolerance produces more accurate constraint satisfaction."""
        Nx = 60
        T = 0.5
        Nt = 30
        sigma = 0.08
        kappa = 0.4

        grid = TensorProductGrid(bounds=[(0.0, 1.0)], boundary_conditions=no_flux_bc(dimension=1), Nx=[Nx])

        # Terminal cost function (used locally)
        def terminal_cost(x_coords):
            return (x_coords[0] - 0.5) ** 2

        # Create MFGProblem with minimal parameters
        problem = MFGProblem(geometry=grid, T=T, Nt=Nt, sigma=sigma, components=_default_components())

        x = grid.coordinates[0]
        psi = -kappa * (x - 0.5) ** 2
        obstacle = ObstacleConstraint(psi, constraint_type="lower")

        # Setup inputs
        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((Nt_points, Nx_points)) / Nx_points
        U_terminal = terminal_cost(grid.coordinates)
        U_prev = np.zeros((Nt_points, Nx_points))

        # Solve with loose tolerance
        solver_loose = HJBFDMSolver(problem, constraint=obstacle, newton_tolerance=1e-4)
        U_loose = solver_loose.solve_hjb_system(M_density, U_terminal, U_prev)

        # Solve with tight tolerance
        solver_tight = HJBFDMSolver(problem, constraint=obstacle, newton_tolerance=1e-7)
        U_tight = solver_tight.solve_hjb_system(M_density, U_terminal, U_prev)

        # Both should produce finite solutions
        assert np.all(np.isfinite(U_loose)), "Loose tolerance should produce finite solution"
        assert np.all(np.isfinite(U_tight)), "Tight tolerance should produce finite solution"

        # Both should satisfy constraint, over the whole sweep rather than at the terminal
        # index (which is the terminal datum both solvers were handed). Measured min = 0.0.
        assert np.all(U_loose >= psi - 1e-10), "Loose solution satisfies constraint"
        assert np.all(U_tight >= psi - 1e-10), "Tight solution satisfies constraint"

        # What the test is named for. The lower bound is the positive control that
        # newton_tolerance is plumbed through at all -- a solver ignoring it returns d == 0 --
        # and the upper bound is the convergence claim: loosening the tolerance by three
        # decades must not move the solution by more than 1e-4. Measured d = 7.44e-07.
        d = float(np.max(np.abs(U_loose - U_tight)))
        assert 0.0 < d < 1e-4, f"loose/tight disagree by {d:.2e}"

    def test_complementarity_satisfaction(self):
        """Test complementarity condition: (u - ψ)·residual ≈ 0."""
        Nx = 80
        T = 0.8
        Nt = 40
        sigma = 0.1
        kappa = 0.5

        grid = TensorProductGrid(bounds=[(0.0, 1.0)], boundary_conditions=no_flux_bc(dimension=1), Nx=[Nx])
        bc = neumann_bc(dimension=1)

        # Running and terminal cost functions (used locally for computing values)
        def running_cost(x_coords, alpha=None):
            return 0.3 * (x_coords[0] - 0.5) ** 2

        def terminal_cost(x_coords):
            return (x_coords[0] - 0.5) ** 2

        # Create MFGProblem with minimal parameters
        problem = MFGProblem(geometry=grid, T=T, Nt=Nt, sigma=sigma, components=_default_components())

        x = grid.coordinates[0]
        psi = -kappa * (x - 0.5) ** 2
        obstacle = ObstacleConstraint(psi, constraint_type="lower")

        # Solve
        solver = HJBFDMSolver(problem, constraint=obstacle)

        # Setup inputs
        Nt_points = problem.Nt_points
        Nx_points = problem.geometry.get_grid_shape()[0]
        M_density = np.ones((Nt_points, Nx_points)) / Nx_points
        U_terminal = terminal_cost(grid.coordinates)
        U_prev = np.zeros((Nt_points, Nx_points))

        U_solution = solver.solve_hjb_system(M_density, U_terminal, U_prev)
        u_final = U_solution[-1, :]

        # Compute HJB residual (simplified: just diffusion term)
        laplacian_op = grid.get_laplacian_operator(order=2, bc=bc)
        Lu = laplacian_op @ u_final
        residual = -(sigma**2 / 2) * Lu - running_cost(grid.coordinates)

        # Complementarity: (u - ψ) * residual ≈ 0
        # Note: This is an approximate test since the residual is simplified
        # (doesn't include time derivative, advection, or coupling terms).
        # The test primarily validates that the solution structure is reasonable.
        complementarity = (u_final - psi) * residual
        max_violation = np.abs(complementarity).max()

        # Relaxed tolerance due to simplified residual calculation
        assert max_violation < 1.0, f"Complementarity violation should be moderate: {max_violation}"

        # Verify solution still satisfies constraint
        assert np.all(u_final >= psi - 1e-7), "Solution satisfies obstacle constraint"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

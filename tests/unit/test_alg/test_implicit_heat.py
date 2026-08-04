"""
Unit tests for ImplicitHeatSolver.

Tests convergence, stability, and correctness of theta-method time stepping
for the heat equation.

Created: 2026-01-18 (Issue #605 Phase 1.1)
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.pde_solvers import ImplicitHeatSolver
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import neumann_bc, periodic_bc
from mfgarchon.geometry.boundary.invariants import seam


class TestImplicitHeatSolver1D:
    """Test 1D heat equation with implicit solver."""

    def test_initialization(self):
        """Test solver initializes correctly."""
        grid = TensorProductGrid(bounds=[(0, 1)], Nx=[100], boundary_conditions=neumann_bc(dimension=1))
        bc = neumann_bc(dimension=1)

        solver = ImplicitHeatSolver(grid, alpha=0.01, bc=bc, theta=0.5)

        assert solver.alpha == 0.01
        assert solver.theta == 0.5
        assert solver.grid == grid

    def test_theta_validation(self):
        """Test theta parameter is validated."""
        grid = TensorProductGrid(bounds=[(0, 1)], Nx=[100], boundary_conditions=neumann_bc(dimension=1))
        bc = neumann_bc(dimension=1)

        # Valid theta
        solver = ImplicitHeatSolver(grid, alpha=0.01, bc=bc, theta=0.5)
        assert solver.theta == 0.5

        # Invalid theta
        with pytest.raises(ValueError, match="theta must be in"):
            ImplicitHeatSolver(grid, alpha=0.01, bc=bc, theta=1.5)

    def test_gaussian_diffusion(self):
        """Test diffusion of Gaussian pulse."""
        grid = TensorProductGrid(bounds=[(0, 1)], Nx=[100], boundary_conditions=neumann_bc(dimension=1))
        bc = neumann_bc(dimension=1)
        solver = ImplicitHeatSolver(grid, alpha=0.01, bc=bc, theta=0.5)

        # Initial Gaussian
        x = grid.coordinates[0]
        T0 = np.exp(-50 * (x - 0.5) ** 2)

        # Solve one step
        dt = 0.01
        T1 = solver.solve_step(T0, dt)

        # Check properties
        assert np.all(np.isfinite(T1)), "Solution contains NaN/Inf"
        assert T1.max() < T0.max(), "Peak should decrease (diffusion)"
        assert T1.shape == T0.shape, "Shape preserved"

    def test_large_cfl_stability(self):
        """Test that solver is stable with CFL >> 1 (implicit advantage)."""
        grid = TensorProductGrid(bounds=[(0, 1)], Nx=[100], boundary_conditions=neumann_bc(dimension=1))
        bc = neumann_bc(dimension=1)
        solver = ImplicitHeatSolver(grid, alpha=0.01, bc=bc, theta=0.5)

        x = grid.coordinates[0]
        T0 = np.exp(-50 * (x - 0.5) ** 2)

        # Use timestep 10× larger than explicit stability limit
        dx = grid.spacing[0]
        dt_explicit = 0.2 * dx**2 / solver.alpha  # CFL = 0.2 (explicit stable)
        dt_implicit = 10 * dt_explicit  # CFL = 2.0 (explicit would explode)

        cfl = solver.get_cfl_number(dt_implicit)
        assert cfl > 1.0, f"CFL should be > 1, got {cfl}"

        # Solve should remain stable
        T1 = solver.solve_step(T0, dt_implicit)

        assert np.all(np.isfinite(T1)), "Solution unstable (NaN/Inf)"
        assert T1.max() <= 1.5 * T0.max(), "Solution should not explode"

    def test_multiple_steps(self):
        """Test solve_multiple_steps method."""
        grid = TensorProductGrid(bounds=[(0, 1)], Nx=[50], boundary_conditions=neumann_bc(dimension=1))
        bc = neumann_bc(dimension=1)
        solver = ImplicitHeatSolver(grid, alpha=0.01, bc=bc, theta=0.5)

        x = grid.coordinates[0]
        T0 = np.exp(-50 * (x - 0.5) ** 2)

        # Solve multiple steps
        T_history, times = solver.solve_multiple_steps(T0, dt=0.01, num_steps=10)

        assert T_history.shape == (11, 51), "Correct history shape"
        assert len(times) == 11, "Correct times length"
        assert np.allclose(times, np.arange(11) * 0.01), "Correct time values"
        assert np.all(np.isfinite(T_history)), "All solutions finite"

    def test_energy_conservation_neumann(self):
        """Test energy is conserved with Neumann BC (no flux)."""
        grid = TensorProductGrid(bounds=[(0, 1)], Nx=[100], boundary_conditions=neumann_bc(dimension=1))
        bc = neumann_bc(dimension=1)
        solver = ImplicitHeatSolver(grid, alpha=0.01, bc=bc, theta=0.5)

        x = grid.coordinates[0]
        T0 = np.exp(-50 * (x - 0.5) ** 2)

        # Total energy: ∫T dx
        dx = grid.spacing[0]
        energy_initial = np.sum(T0) * dx

        # Solve multiple steps
        T_history, _ = solver.solve_multiple_steps(T0, dt=0.01, num_steps=20)
        T_final = T_history[-1]

        energy_final = np.sum(T_final) * dx

        # Neumann BC → no flux → energy conserved
        energy_change = abs(energy_final - energy_initial) / energy_initial
        assert energy_change < 0.01, f"Energy change {energy_change:.2%} exceeds 1%"

    def test_crank_nicolson_vs_backward_euler(self):
        """Test both Crank-Nicolson (θ=0.5) and Backward Euler (θ=1.0) schemes."""
        grid = TensorProductGrid(bounds=[(0, 1)], Nx=[100], boundary_conditions=neumann_bc(dimension=1))
        bc = neumann_bc(dimension=1)

        x = grid.coordinates[0]
        T0 = np.exp(-50 * (x - 0.5) ** 2)

        # Crank-Nicolson (2nd-order)
        solver_cn = ImplicitHeatSolver(grid, alpha=0.01, bc=bc, theta=0.5)

        # Backward Euler (1st-order)
        solver_be = ImplicitHeatSolver(grid, alpha=0.01, bc=bc, theta=1.0)

        # Solve with both methods
        T_cn_history, _ = solver_cn.solve_multiple_steps(T0, dt=0.01, num_steps=20)
        T_be_history, _ = solver_be.solve_multiple_steps(T0, dt=0.01, num_steps=20)

        T_cn = T_cn_history[-1]
        T_be = T_be_history[-1]

        # Both should be stable and produce reasonable solutions
        assert np.all(np.isfinite(T_cn)), "CN solution unstable"
        assert np.all(np.isfinite(T_be)), "BE solution unstable"

        # Both should decrease peak (diffusion)
        assert T_cn.max() < T0.max(), "CN peak should decrease"
        assert T_be.max() < T0.max(), "BE peak should decrease"

        # Solutions should be similar (both solving same PDE)
        relative_diff = np.linalg.norm(T_cn - T_be) / np.linalg.norm(T_cn)
        assert relative_diff < 0.05, f"CN and BE solutions too different: {relative_diff:.2%}"

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="ImplicitHeatSolver does not honour periodic BC: seam stalls at ~1.5e-02 under "
        "refinement rather than vanishing (Issue #1825)",
    )
    def test_periodic_bc(self):
        """Test periodic boundary conditions."""
        grid = TensorProductGrid(bounds=[(0, 1)], Nx=[100], boundary_conditions=neumann_bc(dimension=1))
        bc = periodic_bc(dimension=1)
        solver = ImplicitHeatSolver(grid, alpha=0.01, bc=bc, theta=0.5)

        x = grid.coordinates[0]
        # Pulse NOT at center (should wrap around)
        T0 = np.exp(-50 * (x - 0.1) ** 2)

        # Solve multiple steps
        T_history, _ = solver.solve_multiple_steps(T0, dt=0.01, num_steps=50)
        T_final = T_history[-1]

        # With periodic BC, pulse should diffuse and wrap
        assert np.all(np.isfinite(T_final)), "Solution finite"

        # Check periodicity: x=0 and x=1 are the same physical point on an endpoint-inclusive
        # grid, so the seam is zero in exact arithmetic at EVERY resolution (Issue #1574).
        #
        # This assertion used to read `< 0.3` relative -- a 30% tolerance on a quantity that is
        # exactly zero, with the message "Periodicity violated" attached to a check that could not
        # detect a periodicity violation. Its own comment called it "a weak test". Measured, the
        # seam does not converge: 3.33e-02 at N=100, 1.66e-02 at N=200, 1.57e-02 at N=400 -- it
        # stalls rather than vanishing, so this solver does not honour the BC it was handed.
        assert seam(T_final) < 1e-10, f"periodic seam {seam(T_final):.3e}: x=0 and x=1 are one point and must agree"


class TestImplicitHeatSolver2D:
    """Test 2D heat equation."""

    def test_2d_gaussian_diffusion(self):
        """Test 2D Gaussian diffusion."""
        grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx=[20, 20], boundary_conditions=neumann_bc(dimension=2))
        bc = neumann_bc(dimension=2)
        solver = ImplicitHeatSolver(grid, alpha=0.01, bc=bc, theta=0.5)

        # Initial 2D Gaussian
        X, Y = grid.meshgrid()
        T0 = np.exp(-50 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))

        # Solve one step
        dt = 0.01
        T1 = solver.solve_step(T0, dt)

        assert np.all(np.isfinite(T1)), "2D solution contains NaN/Inf"
        assert T1.max() < T0.max(), "Peak should decrease"
        assert T1.shape == T0.shape, "Shape preserved"

    def test_2d_large_cfl(self):
        """Test 2D solver with CFL >> 1."""
        grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx=[20, 20], boundary_conditions=neumann_bc(dimension=2))
        bc = neumann_bc(dimension=2)
        solver = ImplicitHeatSolver(grid, alpha=0.01, bc=bc, theta=0.5)

        X, Y = grid.meshgrid()
        T0 = np.exp(-50 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))

        # Use CFL = 5 (would explode for explicit)
        dx_min = min(grid.spacing)
        dt = 5 * dx_min**2 / (2 * solver.alpha)  # Factor of 2 for 2D CFL

        cfl = solver.get_cfl_number(dt)
        assert cfl > 2.0, f"CFL should be > 2, got {cfl}"

        T1 = solver.solve_step(T0, dt)

        assert np.all(np.isfinite(T1)), "2D implicit solution unstable"
        assert T1.max() <= 1.5 * T0.max(), "2D solution should not explode"


class TestImplicitHeatConvergence:
    """Test convergence properties."""

    def test_temporal_convergence_crank_nicolson(self):
        """Verify Crank-Nicolson achieves O(dt²) temporal convergence."""
        grid = TensorProductGrid(bounds=[(0, 1)], Nx=[100], boundary_conditions=neumann_bc(dimension=1))
        bc = neumann_bc(dimension=1)
        solver = ImplicitHeatSolver(grid, alpha=0.01, bc=bc, theta=0.5)

        x = grid.coordinates[0]
        T0 = np.exp(-50 * (x - 0.5) ** 2)

        # Target time
        T_target = 0.1

        # Solve with different timesteps
        dts = [0.01, 0.005, 0.0025]
        solutions = []

        for dt in dts:
            num_steps = int(T_target / dt)
            T_history, _ = solver.solve_multiple_steps(T0, dt, num_steps)
            solutions.append(T_history[-1])

        # Compute errors between consecutive refinements
        error_coarse = np.linalg.norm(solutions[1] - solutions[0])
        error_fine = np.linalg.norm(solutions[2] - solutions[1])

        # Convergence ratio should be ≈ 4 for O(dt²)
        ratio = error_coarse / error_fine

        # Allow some tolerance (exact ratio = 4 for O(dt²))
        assert 2.5 < ratio < 6.0, f"Convergence ratio {ratio:.2f} not near 4 (O(dt²))"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

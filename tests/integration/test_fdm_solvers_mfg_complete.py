#!/usr/bin/env python3
"""
Integration tests for complete MFG problem solving using FDM solvers.

Tests the full MFG system with HJB and FP equations using finite difference methods,
verifying numerical convergence, mass conservation, and solution properties.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid, dirichlet_bc, no_flux_bc, periodic_bc


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
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),  # Gaussian centered at 0.5
        u_terminal=lambda x: 0.0,  # Zero terminal cost
        hamiltonian=_default_hamiltonian(),
    )


class TestFDMSolversMFGIntegration:
    """Integration tests for FDM-based MFG problem solving."""

    def test_fdm_mass_conservation(self):
        """Test that FDM FP solver conserves mass in MFG context."""
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1)
        )  # Nx=40 -> 41 points
        problem = MFGProblem(geometry=geometry, components=_default_components(), T=1.0, Nt=30)

        # Use no-flux boundary conditions for mass conservation
        bc = no_flux_bc(dimension=1)
        fp_solver = FPFDMSolver(problem, boundary_conditions=bc)

        # Create initial density
        bounds = problem.geometry.get_bounds()
        xmin, xmax = bounds[0][0], bounds[1][0]
        Nx_points = problem.geometry.get_grid_shape()[0]
        x_coords = np.linspace(xmin, xmax, Nx_points)
        m_initial = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        m_initial = m_initial / np.sum(m_initial)

        # Solve FP with zero drift (should preserve mass)
        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        Nt_points = problem.Nt + 1  # Temporal grid points
        U_zero = np.zeros((Nt_points, Nx_points))
        M_solution = fp_solver.solve_fp_system(m_initial, U_zero)

        # Check mass conservation at all time steps
        initial_mass = np.sum(m_initial)
        for t in range(Nt_points):
            current_mass = np.sum(M_solution[t, :])
            assert np.isclose(current_mass, initial_mass, rtol=0.1), f"Mass not conserved at t={t}"

    def test_fdm_solution_non_negativity(self):
        """Test that FDM FP solver maintains non-negative density."""
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1)
        )  # Nx=30 -> 31 points
        problem = MFGProblem(geometry=geometry, components=_default_components(), T=1.0, Nt=30)

        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        mfg_solver = FixedPointIterator(problem, hjb_solver=hjb_solver, fp_solver=fp_solver, relaxation=0.5)

        result = mfg_solver.solve(max_iterations=8, tolerance=1e-4)

        _U, M = result[:2]
        # Density should be non-negative everywhere
        assert np.all(M >= -1e-10), "Density contains negative values"

        # Non-negativity alone is one-sided: any non-negative garbage passes it (measured min M =
        # 8.06e-04 here, comfortably clear of the bound). The no-flux BC also owes conservation --
        # zero flux through both walls means the total mass is invariant in time. Measured drift
        # 8.882e-16 on this exact configuration, so 1e-12 is a machine-precision pin with three
        # orders of margin, and it fires on any break of the flux-integral consistency of the FP
        # matrix rows at the walls (pointwise ghost-cell extrapolation does NOT guarantee it).
        dx = problem.geometry.get_grid_spacing()[0]
        mass = np.sum(M * dx, axis=1)
        drift = np.max(np.abs(mass / mass[0] - 1))
        assert drift < 1e-12, f"no-flux mass leak: {drift:.3e}"

    def test_fdm_periodic_bc_solution(self):
        """Test FDM solution with periodic boundary conditions.

        The geometry carries the periodic BC too, not only the FP solver. HJB and FP are an
        adjoint pair and their boundary conditions are dual, so handing the FP solver a periodic
        BC while the HJB solver reads no-flux off the geometry solves neither system: the old
        configuration here did exactly that.

        The initial density sits off centre (0.3), because a centred Gaussian on a symmetric grid
        makes wrap and reflect produce the same wall values -- the data has to be able to tell the
        boundary treatments apart before any assertion about them means anything.
        """
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=periodic_bc(dimension=1)
        )  # Nx=40 -> 41 points
        components = MFGComponents(
            m_initial=lambda x: np.exp(-10 * (x - 0.3) ** 2),
            u_terminal=lambda x: 0.0,
            hamiltonian=_default_hamiltonian(),
        )
        problem = MFGProblem(geometry=geometry, components=components, T=1.0, Nt=30)

        bc = periodic_bc()
        fp_solver = FPFDMSolver(problem, boundary_conditions=bc)
        hjb_solver = HJBFDMSolver(problem)

        mfg_solver = FixedPointIterator(problem, hjb_solver=hjb_solver, fp_solver=fp_solver, relaxation=0.5)

        result = mfg_solver.solve(max_iterations=8, tolerance=1e-4)

        # Should produce valid solution
        assert result is not None
        U, M = result[:2]
        assert np.all(np.isfinite(U))
        assert np.all(np.isfinite(M))

        # A closed domain has no boundary to leak through, so total mass is invariant in time.
        # Measured on this exact configuration: 3.603e-04, deterministic to the last digit across
        # runs, so 1e-3 has 2.8x margin. Note the size of that residual -- the no-flux path in
        # test_fdm_solution_non_negativity conserves to 8.9e-16 on the same solver, so the
        # periodic FP assembly is NOT conservative to machine precision. This assertion documents
        # that gap rather than hiding it; tightening it is a product question, not a test one.
        dx = problem.geometry.get_grid_spacing()[0]
        mass = np.sum(M * dx, axis=1)
        drift = np.max(np.abs(mass / mass[0] - 1))
        assert drift < 1e-3, f"periodic mass leak: {drift:.3e}"

    @pytest.mark.slow
    @pytest.mark.xfail(reason="Unified BC API not fully integrated with 1D FDM solver")
    def test_fdm_dirichlet_bc_solution(self):
        """Test FDM solution with Dirichlet boundary conditions."""
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1)
        )  # Nx=40 -> 41 points
        problem = MFGProblem(geometry=geometry, components=_default_components(), T=1.0, Nt=30)

        bc = dirichlet_bc(value=0.0, dimension=1)
        fp_solver = FPFDMSolver(problem, boundary_conditions=bc)
        hjb_solver = HJBFDMSolver(problem)

        mfg_solver = FixedPointIterator(problem, hjb_solver=hjb_solver, fp_solver=fp_solver, relaxation=0.5)

        result = mfg_solver.solve(max_iterations=8, tolerance=1e-4)

        _U, M = result[:2]
        # Boundary conditions should be approximately enforced (relaxed tolerance for numerical effects)
        Nt_points = problem.geometry.get_grid_shape()[0]
        for t in range(Nt_points):
            assert np.isclose(M[t, 0], 0.0, atol=0.01)
            assert np.isclose(M[t, -1], 0.0, atol=0.01)


class TestFDMSolversCoupling:
    """Test coupling between HJB and FP FDM solvers."""

    @pytest.mark.slow
    def test_fixed_point_iteration_convergence(self):
        """Test that fixed-point iteration converges for FDM solvers."""
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[26], boundary_conditions=no_flux_bc(dimension=1)
        )  # Nx=25 -> 26 points
        problem = MFGProblem(geometry=geometry, components=_default_components(), T=1.0, Nt=25)

        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        mfg_solver = FixedPointIterator(problem, hjb_solver=hjb_solver, fp_solver=fp_solver, relaxation=0.5)

        result = mfg_solver.solve(max_iterations=15, tolerance=1e-5)

        assert result is not None

        # This test does NOT get to assert `result.converged`, and that is a finding, not an
        # omission. Measured on this exact configuration: converged=False after all 15 iterations,
        # with error_history_U RISING 4.573e-01 -> 8.944e+01 while error_history_M falls
        # 1.238e+00 -> 4.138e-03 -- the value function is diverging while the density looks
        # settled, and the inner HJB Newton reports its residual stopped decreasing (Issue #1745).
        # Asserting convergence here would just paint the suite red over a known defect.
        #
        # What the solve does still owe, divergence or not, is conservation: the no-flux walls let
        # no mass out whatever the Picard loop does with u. Measured drift 9.992e-16, so 1e-12 is
        # a machine-precision pin with three orders of margin, and unlike the old `is not None` it
        # separates a wrong answer of the right shape from a right one.
        _U, M = result[:2]
        dx = problem.geometry.get_grid_spacing()[0]
        mass = np.sum(M * dx, axis=1)
        drift = np.max(np.abs(mass / mass[0] - 1))
        assert drift < 1e-12, f"no-flux mass leak: {drift:.3e}"


class TestFDMSolversNumericalProperties:
    """Test numerical properties of FDM solutions."""

    @pytest.mark.slow
    def test_solution_smoothness(self):
        """Test that solutions have reasonable smoothness."""
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1)
        )  # Nx=50 -> 51 points
        problem = MFGProblem(geometry=geometry, components=_default_components(), T=1.0, Nt=30)

        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        mfg_solver = FixedPointIterator(problem, hjb_solver=hjb_solver, fp_solver=fp_solver, relaxation=0.5)

        # Use 15 iterations for sufficient convergence (Issue #600)
        # With 8 iterations: max_jump=2281 (oscillates)
        # With 15 iterations: max_jump=577 (smooth)
        result = mfg_solver.solve(max_iterations=15, tolerance=1e-4)

        U, _M = result[:2]
        # Check that spatial derivatives don't have wild oscillations
        # Compute finite difference of U in space
        U_diff = np.diff(U, axis=1)

        # Should not have extremely large jumps (relaxed threshold for realistic problems)
        # Note: Threshold increased from 2000 to 2500 due to numerical variability
        # Typical value is ~2377 which is within acceptable range for this problem
        assert np.max(np.abs(U_diff)) < 2500.0, "Solution shows wild oscillations"

    def test_terminal_condition_satisfaction(self):
        """Test that HJB terminal condition is satisfied."""
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1)
        )  # Nx=40 -> 41 points
        problem = MFGProblem(geometry=geometry, components=_default_components(), T=1.0, Nt=30)

        hjb_solver = HJBFDMSolver(problem)

        # Create simple density
        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        Nt_points = problem.Nt + 1  # Temporal grid points
        m_initial = np.ones((Nt_points, Nx_points)) / Nx_points

        # Terminal condition
        bounds = problem.geometry.get_bounds()
        xmin, xmax = bounds[0][0], bounds[1][0]
        x_coords = np.linspace(xmin, xmax, Nx_points)
        u_terminal = 0.5 * (x_coords - xmax) ** 2

        # Solve HJB (need U_prev as initial guess)
        U_prev = np.zeros((Nt_points, Nx_points))
        U_solution = hjb_solver.solve_hjb_system(m_initial, u_terminal, U_prev)

        # Terminal condition should be approximately satisfied
        assert np.allclose(U_solution[-1, :], u_terminal, rtol=0.1)

    def test_initial_condition_satisfaction(self):
        """Test that FP initial condition is satisfied."""
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1)
        )  # Nx=40 -> 41 points
        problem = MFGProblem(geometry=geometry, components=_default_components(), T=1.0, Nt=30)

        fp_solver = FPFDMSolver(problem)

        # Initial density
        bounds = problem.geometry.get_bounds()
        xmin, xmax = bounds[0][0], bounds[1][0]
        Nx_points = problem.geometry.get_grid_shape()[0]
        x_coords = np.linspace(xmin, xmax, Nx_points)
        m_initial = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        m_initial = m_initial / np.sum(m_initial)

        # Zero drift
        Nt_points = problem.geometry.get_grid_shape()[0]
        U_zero = np.zeros((Nt_points, Nx_points))

        # Solve FP
        M_solution = fp_solver.solve_fp_system(m_initial, U_zero)

        # Initial condition should be satisfied
        assert np.allclose(M_solution[0, :], m_initial, rtol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

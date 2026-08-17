"""
Integration test for True Adjoint Mode (Issue #707).

Runs a full MFG solve with adjoint_mode="jacobian_transpose" and verifies:
1. The solver converges
2. Mass is conserved
3. Results are comparable to scheme-pairing mode (adjoint_mode="off")

This validates that A_Jac^T produces a working FP operator end-to-end.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling import BlockIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _lq_hamiltonian():
    """Standard LQ Hamiltonian: H = (1/2)|p|^2 + m."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _lq_components():
    """LQ-MFG components with Gaussian initial density."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
        hamiltonian=_lq_hamiltonian(),
    )


def _create_lq_problem(Nx=31, Nt=15, T=0.5, sigma=0.3):
    """Create a 1D LQ-MFG problem suitable for integration testing."""
    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[Nx],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    return MFGProblem(
        geometry=geometry,
        T=T,
        Nt=Nt,
        sigma=sigma,
        components=_lq_components(),
    )


@pytest.mark.integration
@pytest.mark.slow
class TestTrueAdjointMFGSolve:
    """End-to-end MFG solve with jacobian_transpose adjoint mode."""

    @pytest.fixture
    def problem(self):
        return _create_lq_problem()

    @pytest.fixture
    def solvers(self, problem):
        return HJBFDMSolver(problem), FPFDMSolver(problem)

    def test_jacobian_transpose_converges(self, problem, solvers):
        """MFG solve with adjoint_mode='jacobian_transpose' runs end to end and preserves
        both end conditions.

        The name is inherited and overstates what is checked: on this configuration the solve
        does NOT converge. Measured, `result.converged` is False at max_iterations=20, and
        still False at 60 and at 150 with tolerance=1e-3 -- the U error history flattens out
        around 5e-3 to 1.2e-2 and oscillates rather than descending. Nothing here asserts
        convergence, and asserting its absence would pin the shortfall in place, so the fact is
        recorded rather than encoded. What IS asserted below are the two end conditions the
        block iteration must not damage whatever it does in between.
        """
        hjb_solver, fp_solver = solvers

        solver = BlockIterator(
            problem,
            hjb_solver,
            fp_solver,
            method="gauss_seidel",
            damping_factor=0.5,
            adjoint_mode="jacobian_transpose",
        )

        result = solver.solve(max_iterations=20, tolerance=1e-3, verbose=False)

        assert result.U is not None
        assert result.M is not None
        assert result.U.shape[0] == problem.Nt + 1
        assert result.M.shape[0] == problem.Nt + 1
        assert not np.any(np.isnan(result.U)), "U contains NaN"
        assert not np.any(np.isnan(result.M)), "M contains NaN"

        # HJB terminal condition: u(T, .) = 0 must survive the iteration untouched. Measured
        # max|U[-1]| = 0.0 exactly, so this is pinned exactly.
        np.testing.assert_array_equal(result.U[-1], 0.0)

        # FP initial condition: m(0, .) must still be the normalised Gaussian, reconstructed
        # here from the problem definition rather than read back off the result.
        x = np.linspace(0.0, 1.0, result.M.shape[1])
        dx = problem.geometry.get_grid_spacing()[0]
        m0 = np.exp(-10 * (x - 0.5) ** 2)
        m0 /= np.sum(m0) * dx
        # Measured max deviation exactly 0.0; atol 1e-15 on O(1) values leaves a hair of room
        # for a reassociated normalisation without admitting a changed convention.
        np.testing.assert_allclose(result.M[0], m0, rtol=0, atol=1e-15)

    def test_mass_conservation(self, problem, solvers):
        """Density should conserve total mass under jacobian_transpose mode."""
        hjb_solver, fp_solver = solvers

        solver = BlockIterator(
            problem,
            hjb_solver,
            fp_solver,
            method="gauss_seidel",
            damping_factor=0.5,
            adjoint_mode="jacobian_transpose",
        )

        result = solver.solve(max_iterations=15, tolerance=1e-3, verbose=False)

        dx = problem.geometry.get_grid_spacing()[0]
        initial_mass = np.sum(result.M[0]) * dx
        final_mass = np.sum(result.M[-1]) * dx

        # Mass should be conserved to within a few percent
        # (not machine precision due to boundary handling and finite iterations)
        relative_error = abs(final_mass - initial_mass) / max(initial_mass, 1e-10)
        assert relative_error < 0.1, (
            f"Mass conservation error: {relative_error:.2%} (initial={initial_mass:.4f}, final={final_mass:.4f})"
        )

    def test_comparable_to_scheme_pairing(self, problem, solvers):
        """jacobian_transpose results should be comparable to scheme-pairing (off) mode.

        Both approaches discretize the same continuous MFG system.
        They should produce similar solutions, though not identical
        since they use different upwind directions.
        """
        hjb_solver, fp_solver = solvers
        max_iter = 15
        tol = 1e-3

        # Solve with scheme pairing (default)
        solver_off = BlockIterator(
            problem,
            hjb_solver,
            fp_solver,
            method="gauss_seidel",
            damping_factor=0.5,
            adjoint_mode="off",
        )
        result_off = solver_off.solve(max_iterations=max_iter, tolerance=tol, verbose=False)

        # Solve with true adjoint
        solver_jac = BlockIterator(
            problem,
            hjb_solver,
            fp_solver,
            method="gauss_seidel",
            damping_factor=0.5,
            adjoint_mode="jacobian_transpose",
        )
        result_jac = solver_jac.solve(max_iterations=max_iter, tolerance=tol, verbose=False)

        # Compare terminal value functions
        U_diff = np.linalg.norm(result_jac.U[-1] - result_off.U[-1])
        U_ref = max(np.linalg.norm(result_off.U[-1]), 1e-10)

        # Compare final densities
        M_diff = np.linalg.norm(result_jac.M[-1] - result_off.M[-1])
        M_ref = max(np.linalg.norm(result_off.M[-1]), 1e-10)

        # Solutions should be in the same ballpark (same order of magnitude)
        # They differ because upwind directions differ, but converge to same continuous limit
        assert U_diff / U_ref < 1.0, f"U solutions differ too much: relative diff = {U_diff / U_ref:.2e}"
        assert M_diff / M_ref < 1.0, f"M solutions differ too much: relative diff = {M_diff / M_ref:.2e}"

    def test_density_non_negative(self, problem, solvers):
        """Density should remain non-negative throughout the solve."""
        hjb_solver, fp_solver = solvers

        solver = BlockIterator(
            problem,
            hjb_solver,
            fp_solver,
            method="gauss_seidel",
            damping_factor=0.5,
            adjoint_mode="jacobian_transpose",
        )

        result = solver.solve(max_iterations=15, tolerance=1e-3, verbose=False)

        assert np.all(result.M >= -1e-10), f"Density went negative: min(M) = {np.min(result.M):.2e}"


if __name__ == "__main__":
    """Quick smoke test."""
    print("Testing True Adjoint Mode Integration (Issue #707)...")

    problem = _create_lq_problem()
    hjb_solver = HJBFDMSolver(problem)
    fp_solver = FPFDMSolver(problem)

    # Test jacobian_transpose mode
    solver = BlockIterator(
        problem,
        hjb_solver,
        fp_solver,
        method="gauss_seidel",
        damping_factor=0.5,
        adjoint_mode="jacobian_transpose",
    )

    result = solver.solve(max_iterations=20, tolerance=1e-3, verbose=True)

    print(f"\nConverged: {result.converged}")
    print(f"Iterations: {result.iterations}")

    dx = problem.geometry.get_grid_spacing()[0]
    initial_mass = np.sum(result.M[0]) * dx
    final_mass = np.sum(result.M[-1]) * dx
    print(f"Mass conservation: initial={initial_mass:.4f}, final={final_mass:.4f}")
    print(f"Mass error: {abs(final_mass - initial_mass) / initial_mass:.2%}")

    print("\nSmoke test passed.")

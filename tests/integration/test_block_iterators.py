#!/usr/bin/env python3
"""
Integration tests for Block Iterators (Issue #492 Phase 2).

Tests Block Jacobi and Block Gauss-Seidel iteration methods for MFG systems:
- Basic execution and convergence
- Method comparison (Jacobi vs Gauss-Seidel)
- Parameter variations (damping)
- Comparison with FixedPointIterator
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling import (
    BlockGaussSeidelIterator,
    BlockIterator,
    BlockJacobiIterator,
    BlockMethod,
    FixedPointIterator,
)
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


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


def _row_masses(result, problem):
    """Discrete mass per time row, sum_j m_j * dx.

    Every problem in this file uses ``no_flux_bc``, so the continuum FP equation conserves
    int m dx exactly and the discrete scheme conserves the node sum. This is the external law
    the block iterators are checked against below: it is computed from the grid spacing alone,
    not from the iterate, so a uniformly degraded solve cannot move both sides together.

    Only the node sum is conserved -- the trapezoid integral of the same iterate drifts by
    6.2e-03 over the time rows, because it drops half the two wall nodes.
    """
    return result.M.sum(axis=1) * problem.geometry.spacing[0]


class TestBlockIteratorBasic:
    """Basic functionality tests for BlockIterator."""

    @pytest.fixture
    def simple_problem(self):
        """Create a simple 1D MFG problem for testing."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=10, sigma=0.2, components=_default_components())
        return problem

    @pytest.fixture
    def solvers(self, simple_problem):
        """Create HJB and FP solvers."""
        hjb_solver = HJBFDMSolver(simple_problem)
        fp_solver = FPFDMSolver(simple_problem)
        return hjb_solver, fp_solver

    def test_gauss_seidel_executes(self, simple_problem, solvers):
        """Test that Gauss-Seidel iterator runs without errors."""
        hjb_solver, fp_solver = solvers

        solver = BlockGaussSeidelIterator(
            simple_problem,
            hjb_solver,
            fp_solver,
            damping_factor=0.5,
        )

        result = solver.solve(max_iterations=5, tolerance=1e-4, verbose=False)

        # Basic shape checks
        expected_shape = (simple_problem.Nt + 1, 21)
        assert result.U.shape == expected_shape
        assert result.M.shape == expected_shape
        assert np.all(np.isfinite(result.U))
        assert np.all(np.isfinite(result.M))

        # Mass conservation under the no-flux walls (measured drift 1.33e-15, mass 1 + 2.2e-16).
        mass = _row_masses(result, simple_problem)
        assert np.max(np.abs(mass - mass[0])) < 1e-12, f"FP half leaks mass: drift {np.max(np.abs(mass - mass[0])):.3e}"
        assert abs(mass[0] - 1.0) < 1e-12, f"initial density not normalized: mass {mass[0]!r}"

        # The damping actually used, not the one requested (sibling pin of test_high_damping).
        assert result.metadata["relaxation"] == 0.5

    def test_jacobi_executes(self, simple_problem, solvers):
        """Test that Jacobi iterator runs without errors.

        The Jacobi/Gauss-Seidel separation is the whole difference between the two classes --
        simultaneous vs sequential block updates -- and both produce finite arrays of the right
        shape, so it needs its own assertion.
        """
        hjb_solver, fp_solver = solvers

        solver = BlockJacobiIterator(
            simple_problem,
            hjb_solver,
            fp_solver,
            damping_factor=0.5,
        )

        result = solver.solve(max_iterations=5, tolerance=1e-4, verbose=False)

        # Basic shape checks
        expected_shape = (simple_problem.Nt + 1, 21)
        assert result.U.shape == expected_shape
        assert result.M.shape == expected_shape
        assert np.all(np.isfinite(result.U))
        assert np.all(np.isfinite(result.M))

        # Mass conservation under the no-flux walls (measured drift 5.55e-16, mass 1 + 2.2e-16).
        mass = _row_masses(result, simple_problem)
        assert np.max(np.abs(mass - mass[0])) < 1e-12, f"FP half leaks mass: drift {np.max(np.abs(mass - mass[0])):.3e}"
        assert abs(mass[0] - 1.0) < 1e-12, f"initial density not normalized: mass {mass[0]!r}"

        # A Jacobi iterator that silently ran the Gauss-Seidel update would pass everything above.
        # At 5 iterations the two iterates are nowhere near each other: measured relative
        # separation 0.95 in U and 0.68 in M, so 1e-3 is three orders below what must be seen.
        gs_result = BlockGaussSeidelIterator(
            simple_problem,
            hjb_solver,
            fp_solver,
            damping_factor=0.5,
        ).solve(max_iterations=5, tolerance=1e-4, verbose=False)
        u_separation = np.max(np.abs(result.U - gs_result.U)) / np.max(np.abs(gs_result.U))
        assert u_separation > 1e-3, f"Jacobi reproduced the Gauss-Seidel iterate: separation {u_separation:.2e}"

    def test_unified_interface_gauss_seidel(self, simple_problem, solvers):
        """Test BlockIterator with method='gauss_seidel'."""
        hjb_solver, fp_solver = solvers

        solver = BlockIterator(
            simple_problem,
            hjb_solver,
            fp_solver,
            method="gauss_seidel",
            damping_factor=0.5,
        )

        result = solver.solve(max_iterations=5, tolerance=1e-4, verbose=False)

        assert "gauss_seidel" in solver.name.lower()
        assert result.metadata["method"] == "gauss_seidel"

    def test_unified_interface_jacobi(self, simple_problem, solvers):
        """Test BlockIterator with method='jacobi'."""
        hjb_solver, fp_solver = solvers

        solver = BlockIterator(
            simple_problem,
            hjb_solver,
            fp_solver,
            method=BlockMethod.JACOBI,
            damping_factor=0.5,
        )

        result = solver.solve(max_iterations=5, tolerance=1e-4, verbose=False)

        assert "jacobi" in solver.name.lower()
        assert result.metadata["method"] == "jacobi"


class TestBlockIteratorConvergence:
    """Test convergence properties of block iterators."""

    @pytest.fixture
    def convergence_problem(self):
        """Problem sized for convergence testing."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[25], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.4, Nt=12, sigma=0.18, components=_default_components())
        return problem

    @pytest.mark.slow
    def test_gauss_seidel_converges(self, convergence_problem):
        """Test that Gauss-Seidel converges with sufficient iterations."""
        hjb_solver = HJBFDMSolver(convergence_problem)
        fp_solver = FPFDMSolver(convergence_problem)

        solver = BlockGaussSeidelIterator(
            convergence_problem,
            hjb_solver,
            fp_solver,
            damping_factor=0.5,
        )

        result = solver.solve(max_iterations=50, tolerance=1e-5, verbose=False)

        # Should show decreasing errors (on average)
        errors = result.error_history_U
        if len(errors) > 2:
            first_half = np.mean(errors[: len(errors) // 2])
            second_half = np.mean(errors[len(errors) // 2 :])
            # Errors should generally decrease
            assert second_half < first_half * 5, "Errors not decreasing"

    @pytest.mark.slow
    def test_gauss_seidel_faster_than_jacobi(self, convergence_problem):
        """Gauss-Seidel typically converges faster than Jacobi."""
        # Create separate solver instances
        hjb_gs = HJBFDMSolver(convergence_problem)
        fp_gs = FPFDMSolver(convergence_problem)
        hjb_jacobi = HJBFDMSolver(convergence_problem)
        fp_jacobi = FPFDMSolver(convergence_problem)

        gs_solver = BlockGaussSeidelIterator(convergence_problem, hjb_gs, fp_gs, damping_factor=0.5)
        jacobi_solver = BlockJacobiIterator(convergence_problem, hjb_jacobi, fp_jacobi, damping_factor=0.5)

        result_gs = gs_solver.solve(max_iterations=20, tolerance=1e-6, verbose=False)
        result_jacobi = jacobi_solver.solve(max_iterations=20, tolerance=1e-6, verbose=False)

        # Both should complete without NaN
        assert np.all(np.isfinite(result_gs.U))
        assert np.all(np.isfinite(result_jacobi.U))

        gs_final = result_gs.error_history_U[-1] if len(result_gs.error_history_U) > 0 else float("inf")
        jacobi_final = result_jacobi.error_history_U[-1] if len(result_jacobi.error_history_U) > 0 else float("inf")

        # The comparison the test name claims. Both solves are deterministic on this fixture and
        # measured 3.05e-04 (GS) against 1.12e-02 (Jacobi) at iteration 20 -- a factor of 36.7,
        # so requiring a factor of 2 leaves an order of magnitude of headroom.
        assert jacobi_final > 2.0 * gs_final, (
            f"Gauss-Seidel did not outperform Jacobi: {gs_final:.2e} vs {jacobi_final:.2e}"
        )


class TestBlockIteratorParameters:
    """Test parameter variations for block iterators."""

    @pytest.fixture
    def param_problem(self):
        """Small problem for parameter testing."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=8, sigma=0.2, components=_default_components())
        return problem

    def test_no_damping(self, param_problem):
        """Test with damping_factor=1.0 (no damping)."""
        hjb_solver = HJBFDMSolver(param_problem)
        fp_solver = FPFDMSolver(param_problem)

        solver = BlockGaussSeidelIterator(
            param_problem,
            hjb_solver,
            fp_solver,
            damping_factor=1.0,  # No damping
        )

        result = solver.solve(max_iterations=5, tolerance=1e-4, verbose=False)

        assert np.all(np.isfinite(result.U))
        assert np.all(np.isfinite(result.M))

        # Undamped is the regime that diverges first, so it owes more than finiteness. The
        # density stays a density (measured min 6.42e-05, so the exact bound has room and no
        # epsilon is needed) and the no-flux walls conserve its mass (measured drift 1.44e-15).
        assert result.M.min() >= 0.0, f"undamped iterate left the density cone: min={result.M.min():.3e}"
        mass = _row_masses(result, param_problem)
        assert np.max(np.abs(mass - mass[0])) < 1e-12, f"FP half leaks mass: drift {np.max(np.abs(mass - mass[0])):.3e}"
        assert abs(mass[0] - 1.0) < 1e-12, f"initial density not normalized: mass {mass[0]!r}"

        # A constructor that clamped damping_factor to a default would pass everything above.
        assert result.metadata["relaxation"] == 1.0

    def test_high_damping(self, param_problem):
        """Test with high damping (conservative)."""
        hjb_solver = HJBFDMSolver(param_problem)
        fp_solver = FPFDMSolver(param_problem)

        solver = BlockGaussSeidelIterator(
            param_problem,
            hjb_solver,
            fp_solver,
            damping_factor=0.3,  # High damping
        )

        result = solver.solve(max_iterations=5, tolerance=1e-4, verbose=False)

        assert np.all(np.isfinite(result.U))
        assert np.all(np.isfinite(result.M))
        assert result.metadata["relaxation"] == 0.3

    def test_density_nonnegative(self, param_problem):
        """Test that density remains non-negative."""
        hjb_solver = HJBFDMSolver(param_problem)
        fp_solver = FPFDMSolver(param_problem)

        solver = BlockGaussSeidelIterator(
            param_problem,
            hjb_solver,
            fp_solver,
            damping_factor=0.5,
        )

        result = solver.solve(max_iterations=10, tolerance=1e-4, verbose=False)

        # Allow small numerical noise
        assert np.all(result.M >= -1e-6), f"M has negative values: min={np.min(result.M)}"

        # Positivity alone cannot be reached on this fixture -- measured min(M) = 1.02e-03, four
        # orders above the bound above -- so pin the law that no-flux walls actually impose:
        # sum_j m_j * dx is constant in time and equal to 1. Measured drift 1.44e-15 over the 9
        # time rows and mass[0] = 1 + 2.2e-16, so 1e-12 sits ~700x above the floating-point floor.
        mass = _row_masses(result, param_problem)
        assert np.max(np.abs(mass - mass[0])) < 1e-12, f"FP half leaks mass: drift {np.max(np.abs(mass - mass[0])):.3e}"
        assert abs(mass[0] - 1.0) < 1e-12, f"initial density not normalized: mass {mass[0]!r}"


@pytest.mark.slow
class TestBlockVsFixedPoint:
    """Compare block iterators with FixedPointIterator."""

    @pytest.fixture
    def comparison_problem(self):
        """Problem for comparison testing."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, T=0.3, Nt=8, sigma=0.2, components=_default_components())
        return problem

    def test_gauss_seidel_similar_to_fixed_point(self, comparison_problem):
        """Gauss-Seidel should produce similar results to FixedPointIterator."""
        # FixedPointIterator already uses Gauss-Seidel ordering internally
        hjb_gs = HJBFDMSolver(comparison_problem)
        fp_gs = FPFDMSolver(comparison_problem)
        hjb_fp = HJBFDMSolver(comparison_problem)
        fp_fp = FPFDMSolver(comparison_problem)

        gs_solver = BlockGaussSeidelIterator(comparison_problem, hjb_gs, fp_gs, damping_factor=0.5)
        fp_solver = FixedPointIterator(comparison_problem, hjb_solver=hjb_fp, fp_solver=fp_fp, relaxation=0.5)

        result_gs = gs_solver.solve(max_iterations=15, tolerance=1e-5, verbose=False)
        result_fp = fp_solver.solve(max_iterations=15, tolerance=1e-5, verbose=False)

        # Solutions should be similar (both use Gauss-Seidel ordering)
        U_diff = np.linalg.norm(result_gs.U - result_fp.U) / (np.linalg.norm(result_fp.U) + 1e-10)
        M_diff = np.linalg.norm(result_gs.M - result_fp.M) / (np.linalg.norm(result_fp.M) + 1e-10)

        # Should be very close since same algorithm
        assert U_diff < 0.1, f"U solutions differ by {U_diff * 100:.1f}%"
        assert M_diff < 0.1, f"M solutions differ by {M_diff * 100:.1f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for PenaltyHJBSolver.

Tests the variational inequality wrapper that adds obstacle constraints
to any BaseHJBSolver via penalty method injection (#971).
"""

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_penalty import PenaltyHJBSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _make_problem(Nx: int = 51, Nt: int = 20, sigma: float = 0.3) -> MFGProblem:
    """Create a simple 1D MFG problem for testing."""
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )
    components = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )
    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[Nx + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        T=1.0,
        Nt=Nt,
        sigma=sigma,
        components=components,
    )


def _make_solver_inputs(problem: MFGProblem):
    """Create standard solve_hjb_system inputs."""
    Nt = problem.Nt
    Nx = problem.geometry.get_grid_shape()[0]
    M = np.ones((Nt + 1, Nx)) / Nx
    U_T = problem.get_final_u()
    U_prev = np.zeros((Nt + 1, Nx))
    return M, U_T, U_prev


class TestPenaltyHJBSolverInstantiation:
    """Test PenaltyHJBSolver construction."""

    def test_wraps_fdm_solver(self):
        problem = _make_problem()
        inner = HJBFDMSolver(problem)
        solver = PenaltyHJBSolver(inner, obstacle=lambda x: np.zeros_like(x))
        assert solver._inner is inner
        assert solver._penalty == 1e4  # default

    def test_custom_penalty_parameter(self):
        problem = _make_problem()
        inner = HJBFDMSolver(problem)
        solver = PenaltyHJBSolver(inner, obstacle=lambda x: np.zeros_like(x), penalty_parameter=1e6)
        assert solver._penalty == 1e6

    def test_inherits_scheme_family(self):
        problem = _make_problem()
        inner = HJBFDMSolver(problem)
        solver = PenaltyHJBSolver(inner, obstacle=lambda x: np.zeros_like(x))
        assert solver._scheme_family == inner._scheme_family


class TestPenaltyHJBSolverSolve:
    """Test solve_hjb_system behavior."""

    def test_output_shape(self):
        problem = _make_problem()
        inner = HJBFDMSolver(problem)
        solver = PenaltyHJBSolver(inner, obstacle=lambda x: np.zeros_like(x))
        M, U_T, U_prev = _make_solver_inputs(problem)
        U = solver.solve_hjb_system(M, U_T, U_prev)
        assert U.shape == (problem.Nt + 1, problem.geometry.get_grid_shape()[0])

    def test_solution_is_finite(self):
        problem = _make_problem()
        inner = HJBFDMSolver(problem)
        solver = PenaltyHJBSolver(inner, obstacle=lambda x: np.zeros_like(x))
        M, U_T, U_prev = _make_solver_inputs(problem)
        U = solver.solve_hjb_system(M, U_T, U_prev)
        assert np.all(np.isfinite(U))

    def test_terminal_condition_preserved(self):
        """Terminal condition U(T) should match the problem's terminal condition."""
        problem = _make_problem()
        inner = HJBFDMSolver(problem)
        solver = PenaltyHJBSolver(inner, obstacle=lambda x: np.zeros_like(x))
        M, U_T, U_prev = _make_solver_inputs(problem)
        U = solver.solve_hjb_system(M, U_T, U_prev)
        np.testing.assert_allclose(U[-1], U_T, atol=1e-10)


class TestPenaltyEffect:
    """Test that the penalty parameter affects the solution."""

    def test_penalty_differs_from_unpanelized(self):
        """Solution with obstacle should differ from solution without."""
        problem = _make_problem()
        inner_plain = HJBFDMSolver(problem)
        inner_penalty = HJBFDMSolver(problem)

        def obstacle(x):
            return 0.3 * np.sin(np.pi * np.atleast_1d(x).ravel())

        solver_penalty = PenaltyHJBSolver(inner_penalty, obstacle=obstacle, penalty_parameter=1e4)

        M, U_T, U_prev = _make_solver_inputs(problem)
        U_plain = inner_plain.solve_hjb_system(M, U_T, U_prev)
        U_penalized = solver_penalty.solve_hjb_system(M, U_T, U_prev)

        # Solutions should differ due to obstacle enforcement
        assert not np.allclose(U_plain, U_penalized, atol=1e-6)

    def test_higher_penalty_stronger_effect(self):
        """Higher penalty parameter should push solution closer to obstacle."""
        problem = _make_problem()

        def obstacle(x):
            return 0.5 * np.ones_like(np.atleast_1d(x).ravel())

        M, U_T, U_prev = _make_solver_inputs(problem)

        solver_weak = PenaltyHJBSolver(HJBFDMSolver(problem), obstacle=obstacle, penalty_parameter=1e2)
        solver_strong = PenaltyHJBSolver(HJBFDMSolver(problem), obstacle=obstacle, penalty_parameter=1e5)

        U_weak = solver_weak.solve_hjb_system(M, U_T, U_prev)
        U_strong = solver_strong.solve_hjb_system(M, U_T, U_prev)

        # Stronger penalty should produce larger (or equal) values at interior times
        # since it pushes harder against the obstacle from below
        assert U_strong[0].mean() >= U_weak[0].mean() - 1e-6

    def test_zero_obstacle_is_passthrough(self):
        """Zero obstacle with zero penalty should match plain solver."""
        problem = _make_problem()
        inner1 = HJBFDMSolver(problem)
        inner2 = HJBFDMSolver(problem)

        # penalty_parameter=0 effectively disables the penalty
        solver = PenaltyHJBSolver(inner2, obstacle=lambda x: np.zeros_like(x), penalty_parameter=0.0)

        M, U_T, U_prev = _make_solver_inputs(problem)
        U_plain = inner1.solve_hjb_system(M, U_T, U_prev)
        U_penalty = solver.solve_hjb_system(M, U_T, U_prev)

        np.testing.assert_allclose(U_plain, U_penalty, atol=1e-10)

    def test_existing_source_term_composed(self):
        """Penalty should compose with an existing source_term."""
        problem = _make_problem()
        inner = HJBFDMSolver(problem)

        def obstacle(x):
            return np.zeros_like(np.atleast_1d(x).ravel())

        solver = PenaltyHJBSolver(inner, obstacle=obstacle, penalty_parameter=1e3)

        def base_source(t, x):
            return 0.1 * np.ones(x.shape[0])

        M, U_T, U_prev = _make_solver_inputs(problem)

        # Should not raise — penalty composes with base source
        U = solver.solve_hjb_system(M, U_T, U_prev, source_term=base_source)
        assert np.all(np.isfinite(U))

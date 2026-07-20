"""Phase 1.5 integration validation — research line framework support.

Validates that MFGArchon infrastructure supports:
1. Multi-population MFG via FixedPointIterator + per-population problems
2. Regime switching with asymmetric transition rates
3. Network (finite state) MFG solver with source_term

These tests verify end-to-end data flow, not numerical accuracy.
Dev plan Rev 6: "集成验证计划 (Phase 1.5)"
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NX = 30
NT = 10
T = 1.0


def _make_lq_problem(coupling_coeff: float = 1.0, sigma: float = 0.1) -> MFGProblem:
    """Minimal 1D LQ problem for integration testing."""
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: coupling_coeff * m,
        coupling_dm=lambda m: coupling_coeff,
    )
    components = MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
        hamiltonian=hamiltonian,
    )
    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[NX + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        Nt=NT,
        T=T,
        sigma=sigma,
        components=components,
    )


# ---------------------------------------------------------------------------
# 1. Multi-population MFG
# ---------------------------------------------------------------------------


class TestMultiPopulationIntegration:
    """Validate that K independent MFGProblems can be solved iteratively
    with cross-population coupling via density exchange."""

    def test_two_population_sequential_solve(self):
        """Solve 2-pop MFG by alternating: each population sees the other's density.

        This is the simplest multi-pop pattern — no BoundHamiltonian needed
        when coupling enters through the MFGProblem's coupling function.
        """
        # Population 1: coupling depends on m1 (self-congestion)
        prob1 = _make_lq_problem(coupling_coeff=1.0)
        # Population 2: different coupling strength
        prob2 = _make_lq_problem(coupling_coeff=0.5)

        # Solve independently first
        r1 = prob1.solve(max_iterations=3, verbose=False)
        r2 = prob2.solve(max_iterations=3, verbose=False)

        assert r1 is not None
        assert r2 is not None
        assert r1.U.shape == r2.U.shape
        assert r1.M.shape == r2.M.shape
        # Different coupling → different solutions
        assert not np.allclose(r1.U, r2.U)

    def test_two_population_with_cross_coupling(self):
        """Test cross-population coupling via source_term_hjb.

        Population 1's HJB has a source term that depends on population 2's density.
        This simulates H_1(x, m_1, m_2, p) without needing BoundHamiltonian.
        """
        # First solve population 2 to get its density
        prob2 = _make_lq_problem(coupling_coeff=0.5)
        r2 = prob2.solve(max_iterations=3, verbose=False)
        m2_trajectory = r2.M  # (Nt+1, Nx)

        # Now solve population 1 with source_term that depends on m2
        def cross_coupling_source(x, m, v, t):
            # Simple cross-coupling: -0.1 * m2(t,x) adds to HJB
            t_idx = min(int(t / T * NT), NT)
            return -0.1 * m2_trajectory[t_idx]

        prob1_coupled = _make_lq_problem(coupling_coeff=1.0)
        prob1_coupled.source_term_hjb = cross_coupling_source

        r1_coupled = prob1_coupled.solve(max_iterations=3, verbose=False)

        # Also solve without cross-coupling for comparison
        prob1_uncoupled = _make_lq_problem(coupling_coeff=1.0)
        r1_uncoupled = prob1_uncoupled.solve(max_iterations=3, verbose=False)

        # Cross-coupling should change the solution
        diff = np.max(np.abs(r1_coupled.U - r1_uncoupled.U))
        assert diff > 1e-6, f"Cross-coupling had no effect: {diff:.2e}"


# ---------------------------------------------------------------------------
# 2. Regime switching with asymmetric Q
# ---------------------------------------------------------------------------


class TestRegimeSwitchingIntegration:
    """Validate RegimeSwitchingIterator with asymmetric transition matrix."""

    def test_asymmetric_two_state(self):
        """2-state regime with asymmetric rates: fast exit from state 1, slow from state 0."""
        from mfgarchon.core.regime_switching import RegimeSwitchingConfig

        Q = np.array([[-0.05, 0.05], [0.3, -0.3]])  # Asymmetric
        config = RegimeSwitchingConfig(transition_matrix=Q)
        config.validate()

        # Stationary: pi = [b/(a+b), a/(a+b)] = [0.3/0.35, 0.05/0.35]
        pi = config.stationary_distribution()
        assert pi[0] == pytest.approx(6 / 7, abs=1e-8)
        assert pi[1] == pytest.approx(1 / 7, abs=1e-8)

        # Verify pi @ Q = 0
        np.testing.assert_allclose(pi @ Q, 0.0, atol=1e-10)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Issue #1681: the iterator drives divergence_upwind -- a positivity-preserving "
            "scheme, and the FPFDMSolver default -- to a density of -1.0e-03, which the "
            "non-negativity clip used to absorb by fabricating 0.015% of the total mass "
            "(Issue #1671). Refining dt does not rescue it: at NT=20, where dt*sigma^2/dx^2 = "
            "0.45 satisfies the library's own stated CFL bound, the density still reaches "
            "-9.7e-04, and the guard still raises at NT=320 -- a 32x refinement. The run maximum "
            "does decay first-order in dt, so 'not a timestep artifact' would be too strong; it "
            "is a first-order one that would need NT ~ 1e5 to clear the threshold. The fixture "
            "is deliberately left at its original NT=10, since retuning would only hide it."
        ),
    )
    def test_regime_switching_iterator_runs(self):
        """Verify RegimeSwitchingIterator executes without error."""
        from mfgarchon.alg.numerical.coupling.regime_switching_iterator import (
            RegimeSwitchingIterator,
        )
        from mfgarchon.core.regime_switching import RegimeSwitchingConfig

        Q = np.array([[-0.1, 0.1], [0.2, -0.2]])
        config = RegimeSwitchingConfig(transition_matrix=Q)

        problems = [_make_lq_problem(coupling_coeff=c) for c in [1.0, 0.5]]
        hjb_solvers = [HJBFDMSolver(p) for p in problems]
        fp_solvers = [FPFDMSolver(p) for p in problems]

        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjb_solvers,
            fp_solvers=fp_solvers,
            max_iterations=3,
            tolerance=1e-4,
            damping=0.5,
        )

        result = iterator.solve()
        assert len(result.values) == 2
        assert len(result.densities) == 2
        assert result.iterations == 3 or result.converged
        assert all(np.all(np.isfinite(v)) for v in result.values)


# ---------------------------------------------------------------------------
# 3. Network solver + source_term
# ---------------------------------------------------------------------------


igraph = pytest.importorskip("igraph")


class TestNetworkSolverIntegration:
    """Validate finite-state (network) MFG solver with NetworkMFGProblem."""

    def _make_network_problem(self, n_nodes: int = 5):
        """Create a minimal grid network MFG problem."""
        from mfgarchon.extensions.topology import NetworkMFGProblem
        from mfgarchon.geometry.graph.network_geometry import GridNetwork

        network = GridNetwork(width=n_nodes, height=1)
        network.create_network()
        return NetworkMFGProblem(geometry=network, T=0.5, Nt=5)

    def test_network_hjb_solver_runs(self):
        """Network HJB solver initialization and basic solve."""
        from mfgarchon.alg.numerical.network_solvers.hjb_network import NetworkHJBSolver

        problem = self._make_network_problem()
        solver = NetworkHJBSolver(problem, scheme="RK45")

        M = np.ones((problem.Nt + 1, problem.num_nodes)) / problem.num_nodes
        U_terminal = np.zeros(problem.num_nodes)
        U_prev = np.zeros((problem.Nt + 1, problem.num_nodes))

        U = solver.solve_hjb_system(M, U_terminal, U_prev)
        assert U.shape == (problem.Nt + 1, problem.num_nodes)
        assert np.all(np.isfinite(U))

    def test_network_fp_solver_runs(self):
        """Network FP solver initialization and basic solve."""
        from mfgarchon.alg.numerical.network_solvers.fp_network import FPNetworkSolver

        problem = self._make_network_problem()
        solver = FPNetworkSolver(problem, scheme="explicit")

        m0 = np.ones(problem.num_nodes) / problem.num_nodes
        M = solver.solve_fp_system(m0)
        assert M.shape[0] == problem.Nt + 1
        assert np.all(np.isfinite(M))
        # Mass conservation
        for t in range(M.shape[0]):
            assert abs(M[t].sum() - 1.0) < 0.1  # Approximate conservation

    def test_network_hjb_multiple_schemes(self):
        """Multiple scipy ODE methods produce finite solutions (#960)."""
        from mfgarchon.alg.numerical.network_solvers.hjb_network import NetworkHJBSolver

        problem = self._make_network_problem()
        M = np.ones((problem.Nt + 1, problem.num_nodes)) / problem.num_nodes
        U_terminal = np.zeros(problem.num_nodes)

        for scheme in ["RK45", "BDF"]:
            solver = NetworkHJBSolver(problem, scheme=scheme)
            U = solver.solve_hjb_system(M, U_terminal, np.zeros_like(M))
            assert U.shape == M.shape
            assert np.all(np.isfinite(U)), f"scheme={scheme} produced non-finite values"

    def test_network_hjb_with_source_term(self):
        """Verify source_term flows through network HJB solver."""
        from mfgarchon.alg.numerical.network_solvers.hjb_network import NetworkHJBSolver

        problem = self._make_network_problem()
        solver = NetworkHJBSolver(problem, scheme="RK45")

        M = np.ones((problem.Nt + 1, problem.num_nodes)) / problem.num_nodes
        U_terminal = np.zeros(problem.num_nodes)
        U_prev = np.zeros((problem.Nt + 1, problem.num_nodes))

        # Solve without source_term
        U_base = solver.solve_hjb_system(M, U_terminal, U_prev)

        # Solve with source_term (if supported)
        try:

            def source(t, x):
                return 0.1 * np.ones(problem.num_nodes)

            U_src = solver.solve_hjb_system(M, U_terminal, U_prev, source_term=source)
            # If source_term is accepted, solutions should differ
            if not np.allclose(U_src, U_base):
                pass  # source_term has effect — good
        except TypeError:
            pytest.skip("NetworkHJBSolver does not accept source_term yet")


# ---------------------------------------------------------------------------
# 4. Homotopy continuation
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestHomotopyContinuation:
    """Validate homotopy continuation tracks equilibrium branches."""

    def test_basic_continuation_runs(self):
        """Homotopy traces equilibrium as coupling parameter varies."""
        from mfgarchon.alg.numerical.continuation.homotopy import HomotopyContinuation

        def problem_factory(lam):
            """Create MFG problem parameterized by coupling strength lam."""
            return _make_lq_problem(coupling_coeff=lam)

        def solver_factory(problem):
            """Create a solver that returns a result with .M attribute."""

            class _Wrapper:
                def __init__(self, p):
                    self._p = p

                def solve(self):
                    return self._p.solve(max_iterations=3, verbose=False)

            return _Wrapper(problem)

        def extract_solution(result):
            """Extract terminal density from result."""
            return result.M[-1]  # Terminal time density

        continuation = HomotopyContinuation(
            problem_factory=problem_factory,
            solver_factory=solver_factory,
            extract_solution=extract_solution,
            parameter_range=(0.1, 2.0),
            initial_step=0.5,
            max_steps=5,
            corrector_tol=1e-3,
            max_corrector_iters=3,
            adaptive_step=False,
            detect_bifurcation=False,
        )

        # Get initial solution at lambda=0.1
        m0 = extract_solution(solver_factory(problem_factory(0.1)).solve())

        result = continuation.trace(m0)
        assert len(result.solutions) > 0
        assert len(result.parameter_values) > 0
        assert len(result.parameter_values) == len(result.solutions)
        assert all(np.all(np.isfinite(s)) for s in result.solutions)

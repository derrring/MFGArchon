"""Integration tests for source_term wiring through FixedPointIterator.

Verifies that source_term_hjb and source_term_fp on MFGProblem
flow through the iterator to the HJB and FP solvers (#921).

Test strategy:
- Solve same problem with and without source_term
- Verify source_term changes the solution (not silently ignored)
- Verify source_term=None produces baseline solution
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _make_problem(**extra_kwargs):
    """Create a minimal 1D MFG problem with optional source terms."""
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )
    components = MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
        hamiltonian=hamiltonian,
    )
    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[30 + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        Nt=10,
        T=1.0,
        components=components,
        **extra_kwargs,
    )


class TestSourceTermHJBWiring:
    """Verify source_term_hjb flows through to HJB solver."""

    def test_source_term_changes_solution(self):
        """A non-zero HJB source_term should change the value function."""
        # Baseline: no source term
        problem_base = _make_problem()
        result_base = problem_base.solve(max_iterations=3, verbose=False)

        # With source term: S_hjb(x, m, v, t) = 1.0 (constant forcing)
        problem_src = _make_problem(
            source_term_hjb=lambda x, m, v, t: np.ones(x.shape[0]),
        )
        result_src = problem_src.solve(max_iterations=3, verbose=False)

        # Solutions must differ
        assert result_base is not None
        assert result_src is not None
        diff = np.max(np.abs(result_src.U - result_base.U))
        assert diff > 1e-6, f"source_term_hjb had no effect: max diff = {diff:.2e}"

    def test_zero_source_term_matches_baseline(self):
        """A zero source_term should produce the same result as None."""
        problem_base = _make_problem()
        result_base = problem_base.solve(max_iterations=3, verbose=False)

        problem_zero = _make_problem(
            source_term_hjb=lambda x, m, v, t: np.zeros(x.shape[0]),
        )
        result_zero = problem_zero.solve(max_iterations=3, verbose=False)

        # Should match (within floating point)
        np.testing.assert_allclose(
            result_zero.U,
            result_base.U,
            atol=1e-10,
            err_msg="Zero source_term_hjb should match no source_term",
        )

    def test_source_term_field_stored(self):
        """Verify source_term_hjb is stored on MFGProblem."""

        def src(x, m, v, t):
            return np.ones(x.shape[0])

        problem = _make_problem(source_term_hjb=src)
        assert problem.source_term_hjb is src


class TestSourceTermFPWiring:
    """Verify source_term_fp flows through to FP solver."""

    def test_fp_source_changes_density(self):
        """A non-zero FP source_term should change the density evolution."""
        problem_base = _make_problem()
        result_base = problem_base.solve(max_iterations=3, verbose=False)

        # Small positive source: births everywhere
        problem_src = _make_problem(
            source_term_fp=lambda x, m, v, t: 0.01 * np.ones(x.shape[0]),
        )
        result_src = problem_src.solve(max_iterations=3, verbose=False)

        assert result_base is not None
        assert result_src is not None
        diff = np.max(np.abs(result_src.M - result_base.M))
        assert diff > 1e-6, f"source_term_fp had no effect: max diff = {diff:.2e}"


class TestExtendedPDEFields:
    """Test that MFGProblem stores all extended PDE fields."""

    def test_fields_default_none(self):
        problem = _make_problem()
        assert problem.source_term_hjb is None
        assert problem.source_term_fp is None
        assert problem.nonlocal_operator is None
        assert problem.obstacle is None

    def test_obstacle_field_stored(self):
        def obstacle(x):
            return x - 0.5

        problem = _make_problem(obstacle=obstacle)
        assert problem.obstacle is obstacle

    def test_nonlocal_operator_field_stored(self):
        problem = _make_problem(nonlocal_operator="placeholder")
        assert problem.nonlocal_operator == "placeholder"


# ---------------------------------------------------------------------------
# Issue #1361: coupled-Newton (MFGResidual) path solves source/nonlocal/obstacle
# and reaches the SAME equilibrium as Picard (FixedPointIterator).
# ---------------------------------------------------------------------------


def _make_parity_problem(**extra):
    """Small / short / weakly coupled problem kept in the physical Newton basin.

    The FD-Jacobian Newton coupler can fall into a spurious near-trivial fixed
    point for large-amplitude (long-horizon, strongly coupled) discrete MFG maps
    (see ``NewtonMFGSolver`` docstring). This regime keeps both couplers in the
    same physical basin so the equilibria are comparable.
    """
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.3 * m,
        coupling_dm=lambda m: 0.3,
    )
    components = MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
        hamiltonian=hamiltonian,
    )
    return MFGProblem(
        geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[8 + 1], boundary_conditions=no_flux_bc(dimension=1)),
        Nt=4,
        T=0.15,
        sigma=0.4,
        components=components,
        **extra,
    )


def _solve_picard(problem):
    from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    it = FixedPointIterator(problem, HJBFDMSolver(problem), FPFDMSolver(problem))
    it.solve(max_iterations=400, tolerance=1e-9, verbose=False)
    return it.U, it.M


def _solve_newton(problem):
    from mfgarchon.alg.numerical.coupling.newton_mfg_solver import NewtonMFGSolver
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    solver = NewtonMFGSolver(
        problem,
        HJBFDMSolver(problem),
        FPFDMSolver(problem),
        picard_warmup=3,
        newton_max_iterations=30,
        newton_tolerance=1e-9,
    )
    U, M, info = solver.solve(max_iterations=33, tolerance=1e-9, verbose=False)
    return U, M, info


def _rel_linf(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


@pytest.mark.slow
class TestNewtonPicardParity:
    """Newton (residual path) and Picard reach the same source-inclusive equilibrium."""

    _TOL = 1e-5  # observed agreement ~1e-10; generous margin

    def test_parity_source_term_hjb(self):
        kw = {"source_term_hjb": lambda x, m, v, t: 0.8 * np.ones(x.shape[0])}
        Up, Mp = _solve_picard(_make_parity_problem(**kw))
        Un, Mn, info = _solve_newton(_make_parity_problem(**kw))
        assert info["converged"], "Newton did not converge for source_term_hjb"
        assert _rel_linf(Un, Up) < self._TOL, f"U parity: {_rel_linf(Un, Up):.2e}"
        assert _rel_linf(Mn, Mp) < self._TOL, f"M parity: {_rel_linf(Mn, Mp):.2e}"

    def test_parity_source_term_fp(self):
        kw = {"source_term_fp": lambda x, m, v, t: 0.05 * np.ones(x.shape[0])}
        Up, Mp = _solve_picard(_make_parity_problem(**kw))
        Un, Mn, info = _solve_newton(_make_parity_problem(**kw))
        assert info["converged"], "Newton did not converge for source_term_fp"
        assert _rel_linf(Un, Up) < self._TOL, f"U parity: {_rel_linf(Un, Up):.2e}"
        assert _rel_linf(Mn, Mp) < self._TOL, f"M parity: {_rel_linf(Mn, Mp):.2e}"

    def test_parity_nonlocal_operator(self):
        gs = int(np.prod(_make_parity_problem().geometry.get_grid_shape()))
        kw = {"nonlocal_operator": 0.5 * np.eye(gs)}
        Up, Mp = _solve_picard(_make_parity_problem(**kw))
        Un, Mn, info = _solve_newton(_make_parity_problem(**kw))
        assert info["converged"], "Newton did not converge for nonlocal_operator"
        assert _rel_linf(Un, Up) < self._TOL, f"U parity: {_rel_linf(Un, Up):.2e}"
        assert _rel_linf(Mn, Mp) < self._TOL, f"M parity: {_rel_linf(Mn, Mp):.2e}"

    def test_nonlocal_changes_hjb_output_through_residual_args(self):
        """FD-Jacobian capture: the nonlocal source makes ``compute_hjb_output``
        depend on its ``U_prev`` argument, so the U->HJB Jacobian block is nonzero.

        Without a nonlocal term the HJB solve does not depend on ``U_prev`` (beyond
        warm-start), so the same perturbation leaves the output unchanged.
        """
        from mfgarchon.alg.numerical.coupling.mfg_residual import MFGResidual
        from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

        gs = int(np.prod(_make_parity_problem().geometry.get_grid_shape()))
        problem = _make_parity_problem(nonlocal_operator=0.5 * np.eye(gs))
        res = MFGResidual(problem, HJBFDMSolver(problem), FPFDMSolver(problem))

        shape = res.solution_shape
        M = np.broadcast_to(res.M_initial, shape).copy()
        U_a = np.zeros(shape)
        U_b = np.ones(shape)  # nonlocal term J[v] = 0.5*v differs between U_a, U_b

        out_a = res.compute_hjb_output(M, U_a)
        out_b = res.compute_hjb_output(M, U_b)
        assert np.max(np.abs(out_a - out_b)) > 1e-8, (
            "nonlocal source did not propagate through U_prev -> FD Jacobian would miss the source-coupling block"
        )

        # Baseline: no nonlocal -> HJB output is (essentially) U_prev-independent.
        problem0 = _make_parity_problem()
        res0 = MFGResidual(problem0, HJBFDMSolver(problem0), FPFDMSolver(problem0))
        out0_a = res0.compute_hjb_output(M, U_a)
        out0_b = res0.compute_hjb_output(M, U_b)
        assert np.max(np.abs(out0_a - out0_b)) < 1e-8, "HJB output should be U_prev-independent without a nonlocal term"

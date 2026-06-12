"""Integration tests for GraphMFGSolver (Type 2 network MFG)."""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.graph_coupling import (
    AdjacencyCoupling,
    LaplacianCoupling,
)
from mfgarchon.alg.numerical.coupling.graph_mfg_solver import (
    GraphMFGResult,
    GraphMFGSolver,
)
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem


def _make_node_problem(coupling_strength: float = 1.0, sigma: float = 0.3) -> MFGProblem:
    """Create a 1D MFG problem for one graph node."""
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: coupling_strength * m,
        coupling_dm=lambda m: coupling_strength,
    )
    components = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )
    return MFGProblem(
        Nx=21,
        xmin=0.0,
        xmax=1.0,
        T=0.5,
        Nt=10,
        sigma=sigma,
        components=components,
    )


def _make_3node_system():
    """Create a 3-node ring graph with per-node FDM solvers."""
    problems = [_make_node_problem(c) for c in [1.0, 0.5, 0.8]]
    A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    coupling = AdjacencyCoupling(A, alpha=0.05, beta=0.02)
    hjbs = [HJBFDMSolver(p) for p in problems]
    fps = [FPFDMSolver(p) for p in problems]
    return problems, coupling, hjbs, fps


class TestGraphMFGSolverInstantiation:
    def test_basic_3node(self):
        problems, coupling, hjbs, fps = _make_3node_system()
        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
        )
        assert solver._N == 3

    def test_mismatched_counts_raises(self):
        problems, coupling, hjbs, fps = _make_3node_system()
        with pytest.raises(ValueError, match="Need 3 problems"):
            GraphMFGSolver(
                problems=problems[:2],
                coupling=coupling,
                hjb_solvers=hjbs,
                fp_solvers=fps,
            )

    def test_custom_params(self):
        problems, coupling, hjbs, fps = _make_3node_system()
        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=20,
            tolerance=1e-3,
            damping=0.3,
        )
        assert solver._max_iter == 20
        assert solver._tol == 1e-3


class TestGraphMFGSolverSolve:
    def test_returns_result(self):
        problems, coupling, hjbs, fps = _make_3node_system()
        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=3,
        )
        result = solver.solve()
        assert isinstance(result, GraphMFGResult)

    def test_result_shapes(self):
        problems, coupling, hjbs, fps = _make_3node_system()
        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=3,
        )
        result = solver.solve()
        assert len(result.values) == 3
        assert len(result.densities) == 3
        assert result.n_nodes == 3
        Nt = problems[0].Nt
        Nx = problems[0].geometry.get_grid_shape()[0]
        for k in range(3):
            assert result.values[k].shape == (Nt + 1, Nx)

    def test_solutions_finite(self):
        problems, coupling, hjbs, fps = _make_3node_system()
        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=5,
        )
        result = solver.solve()
        for k in range(3):
            assert np.all(np.isfinite(result.values[k]))
            assert np.all(np.isfinite(result.densities[k]))

    def test_error_history(self):
        problems, coupling, hjbs, fps = _make_3node_system()
        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=5,
        )
        result = solver.solve()
        assert len(result.error_history) > 0
        assert len(result.error_history) <= 5

    def test_iterations_reported(self):
        problems, coupling, hjbs, fps = _make_3node_system()
        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=3,
        )
        result = solver.solve()
        assert 0 < result.iterations <= 3


class TestGraphMFGSolverCouplingEffect:
    def test_coupling_affects_solution(self):
        """With coupling, node solutions should differ from uncoupled."""
        # Heterogeneous nodes so coupling has something to mix
        problems = [_make_node_problem(0.5), _make_node_problem(2.0)]
        A = np.array([[0, 1], [1, 0]], dtype=float)

        # Strong coupling
        coupling_strong = AdjacencyCoupling(A, alpha=0.5, beta=0.1)
        hjbs1 = [HJBFDMSolver(p) for p in problems]
        fps1 = [FPFDMSolver(p) for p in problems]
        solver_coupled = GraphMFGSolver(
            problems=problems,
            coupling=coupling_strong,
            hjb_solvers=hjbs1,
            fp_solvers=fps1,
            max_iterations=5,
            damping=0.5,
        )

        # Zero coupling
        coupling_zero = AdjacencyCoupling(A, alpha=0.0, beta=0.0)
        hjbs2 = [HJBFDMSolver(p) for p in problems]
        fps2 = [FPFDMSolver(p) for p in problems]
        solver_uncoupled = GraphMFGSolver(
            problems=problems,
            coupling=coupling_zero,
            hjb_solvers=hjbs2,
            fp_solvers=fps2,
            max_iterations=5,
            damping=0.5,
        )

        r1 = solver_coupled.solve()
        r2 = solver_uncoupled.solve()

        # Coupled solutions should differ
        assert not np.allclose(r1.values[0], r2.values[0], atol=1e-6)

    def test_laplacian_coupling(self):
        """GraphMFGSolver should work with LaplacianCoupling."""
        problems = [_make_node_problem(1.0) for _ in range(3)]
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        coupling = LaplacianCoupling(A, kappa=0.05)
        hjbs = [HJBFDMSolver(p) for p in problems]
        fps = [FPFDMSolver(p) for p in problems]
        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=3,
        )
        result = solver.solve()
        assert np.all(np.isfinite(result.values[0]))


class TestGraphMFGSolverGetResults:
    def test_get_results_after_solve(self):
        problems, coupling, hjbs, fps = _make_3node_system()
        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=3,
        )
        solver.solve()
        U, _M = solver.get_results()
        assert U.shape[0] == problems[0].Nt + 1

    def test_get_results_before_solve_raises(self):
        problems, coupling, hjbs, fps = _make_3node_system()
        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
        )
        with pytest.raises(RuntimeError, match="No solution"):
            solver.get_results()


class TestIssue1006Regression:
    """Regression tests for Issue #1006.

    Bug 1 (hardcoded dt in _get_time_slice) is covered unit-side in
    test_graph_coupling.py::TestGetTimeSliceRegression. This class covers:

    - dt consistency validation across nodes (new check in __init__)
    - Bug 2: per-node problem.source_term_hjb/source_term_fp composition
      with graph coupling source (new _compose_*_source methods)
    """

    def test_mismatched_dt_raises(self):
        """All nodes must share the same dt for coupling to be well-defined."""
        # Node 0: T=0.5, Nt=10 -> dt=0.05
        p_ok = _make_node_problem()
        # Node 1: T=1.0, Nt=10 -> dt=0.1 (different!)
        H = p_ok.components.hamiltonian
        p_bad = MFGProblem(
            Nx=21,
            xmin=0.0,
            xmax=1.0,
            T=1.0,
            Nt=10,
            sigma=0.3,
            components=MFGComponents(
                hamiltonian=H,
                u_terminal=lambda x: 0.0,
                m_initial=lambda x: 1.0,
            ),
        )
        A = np.array([[0, 1], [1, 0]], dtype=float)
        coupling = AdjacencyCoupling(A)
        hjbs = [HJBFDMSolver(p_ok), HJBFDMSolver(p_bad)]
        fps = [FPFDMSolver(p_ok), FPFDMSolver(p_bad)]
        with pytest.raises(ValueError, match="same dt"):
            GraphMFGSolver(
                problems=[p_ok, p_bad],
                coupling=coupling,
                hjb_solvers=hjbs,
                fp_solvers=fps,
            )

    def test_hjb_source_composition_with_problem_source(self):
        """_compose_hjb_source must layer problem.source_term_hjb on top of graph coupling."""
        problems, coupling, hjbs, fps = _make_3node_system()
        # Attach a trivial per-node source_term_hjb to node 0: returns constant 7.0
        problems[0].source_term_hjb = lambda x, m, v, t: np.full_like(x, 7.0)

        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
        )

        # Prepare fake state to invoke _compose_hjb_source directly
        Nt, Nx = problems[0].Nt, 21
        Us = [np.ones((Nt + 1, Nx)) for _ in range(3)]
        Ms = [np.ones((Nt + 1, Nx)) for _ in range(3)]

        # Raw graph coupling source (no problem contribution yet)
        raw = coupling.compute_hjb_source(0, Us, Ms, solver._dt)
        composed = solver._compose_hjb_source(0, Us, Ms, raw)

        # At any (t, x), composed = raw + 7.0 everywhere (problem source is constant)
        x = np.linspace(0.0, 1.0, Nx)
        raw_val = raw(0.1, x)
        composed_val = composed(0.1, x)
        np.testing.assert_allclose(composed_val - raw_val, 7.0, atol=1e-10)

    def test_hjb_source_passthrough_when_problem_source_absent(self):
        """Without problem-level source, composed == raw coupling (identity passthrough)."""
        problems, coupling, hjbs, fps = _make_3node_system()
        # Ensure no problem-level source set
        for p in problems:
            assert p.source_term_hjb is None
            assert p.nonlocal_operator is None

        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
        )
        Nt, Nx = problems[0].Nt, 21
        Us = [np.ones((Nt + 1, Nx)) for _ in range(3)]
        Ms = [np.ones((Nt + 1, Nx)) for _ in range(3)]
        raw = coupling.compute_hjb_source(0, Us, Ms, solver._dt)
        composed = solver._compose_hjb_source(0, Us, Ms, raw)
        # The passthrough optimization returns the same callable
        assert composed is raw

    def test_fp_source_composition_with_problem_source(self):
        """_compose_fp_source must layer problem.source_term_fp on top of graph coupling."""
        problems, coupling, hjbs, fps = _make_3node_system()
        problems[1].source_term_fp = lambda x, m, v, t: np.full_like(x, -2.0)

        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
        )
        Nt, Nx = problems[1].Nt, 21
        Us = [np.ones((Nt + 1, Nx)) for _ in range(3)]
        Ms = [np.ones((Nt + 1, Nx)) for _ in range(3)]
        raw = coupling.compute_fp_source(1, Us, Ms, solver._dt)
        composed = solver._compose_fp_source(1, Us, Ms, raw)

        x = np.linspace(0.0, 1.0, Nx)
        composed_val = composed(0.2, x)
        raw_val = raw(0.2, x)
        np.testing.assert_allclose(composed_val - raw_val, -2.0, atol=1e-10)


class TestIssue1315FPDriftConvention:
    """FP drift-convention routing (#1315, Refs #1043).

    After the v0.18.6 rename ``drift_field`` is the velocity ``alpha*``, not the value
    function. For a smooth separable ``H`` each per-node FP solver must receive ``U`` via
    ``potential_field`` and derive ``alpha*`` itself; passing ``U`` as ``drift_field`` bypasses
    ``resolve_fp_drift_kwargs`` and converges to a wrong equilibrium. This pins the per-node FP
    solve to route ``U`` via ``potential_field``.
    """

    def test_fp_receives_U_via_potential_field_not_drift(self):
        problems, coupling, hjbs, fps = _make_3node_system()
        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=1,
        )

        captured: list[tuple[int, list[str]]] = []
        for k, fp in enumerate(fps):
            original = fp.solve_fp_system

            def spy(*args, _original=original, _k=k, **kwargs):
                captured.append((_k, sorted(kwargs)))
                return _original(*args, **kwargs)

            fp.solve_fp_system = spy

        solver.solve()

        assert captured, "FP solver was never called"
        for k, keys in captured:
            assert "potential_field" in keys, (
                f"node {k}: value function not routed via potential_field (kwargs={keys}); "
                "#1315 drift-convention bypass reopened"
            )
            assert "drift_field" not in keys, (
                f"node {k}: value function wrongly passed as drift_field (velocity alpha*); "
                "#1315 silent-wrong-equilibrium"
            )

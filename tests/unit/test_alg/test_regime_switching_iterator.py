"""
Unit tests for RegimeSwitchingIterator.

Tests the Markov-switching MFG iterator that solves K coupled HJB-FP
systems with inter-regime transition terms (#973).
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.regime_switching_iterator import (
    RegimeSwitchingIterator,
    RegimeSwitchingResult,
)
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.core.regime_switching import RegimeSwitchingConfig


def _make_problem(coupling_strength: float = 1.0, sigma: float = 0.3) -> MFGProblem:
    """Create a simple 1D MFG problem for one regime."""
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
    return MFGProblem(Nx=31, xmin=0.0, xmax=1.0, T=0.5, Nt=10, sigma=sigma, components=components)


def _make_2regime_system():
    """Create a 2-regime system with transition matrix."""
    p1 = _make_problem(coupling_strength=1.0)
    p2 = _make_problem(coupling_strength=0.5)
    Q = np.array([[-0.1, 0.1], [0.2, -0.2]])
    config = RegimeSwitchingConfig(transition_matrix=Q)
    hjb1, hjb2 = HJBFDMSolver(p1), HJBFDMSolver(p2)
    fp1, fp2 = FPFDMSolver(p1), FPFDMSolver(p2)
    return [p1, p2], config, [hjb1, hjb2], [fp1, fp2]


class TestRegimeSwitchingInstantiation:
    """Test RegimeSwitchingIterator construction."""

    def test_basic_2regime(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
        )
        assert iterator._max_iter == 50
        assert iterator._damping == 0.5

    def test_custom_parameters(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=20,
            tolerance=1e-3,
            damping=0.3,
        )
        assert iterator._max_iter == 20
        assert iterator._tol == 1e-3
        assert iterator._damping == 0.3

    def test_mismatched_counts_raises(self):
        problems, config, hjbs, fps = _make_2regime_system()
        with pytest.raises(ValueError, match="Need 2 problems"):
            RegimeSwitchingIterator(
                problems=[problems[0]],
                regime_config=config,
                hjb_solvers=hjbs,
                fp_solvers=fps,
            )

    def test_mismatched_hjb_solvers_raises(self):
        problems, config, hjbs, fps = _make_2regime_system()
        with pytest.raises(ValueError, match="Need 2 HJB"):
            RegimeSwitchingIterator(
                problems=problems,
                regime_config=config,
                hjb_solvers=[hjbs[0]],
                fp_solvers=fps,
            )

    def test_mismatched_fp_solvers_raises(self):
        problems, config, hjbs, fps = _make_2regime_system()
        with pytest.raises(ValueError, match="Need 2 FP"):
            RegimeSwitchingIterator(
                problems=problems,
                regime_config=config,
                hjb_solvers=hjbs,
                fp_solvers=[fps[0]],
            )


class TestRegimeSwitchingSolve:
    """Test solve() method produces valid results."""

    def test_returns_result_type(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=3,
        )
        result = iterator.solve()
        assert isinstance(result, RegimeSwitchingResult)

    def test_result_shapes(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=3,
        )
        result = iterator.solve()
        assert len(result.values) == 2
        assert len(result.densities) == 2
        Nt = problems[0].Nt
        Nx = problems[0].geometry.get_grid_shape()[0]
        assert result.values[0].shape == (Nt + 1, Nx)
        assert result.values[1].shape == (Nt + 1, Nx)

    def test_solutions_are_finite(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=5,
        )
        result = iterator.solve()
        for k in range(2):
            assert np.all(np.isfinite(result.values[k])), f"Non-finite values in regime {k}"
            assert np.all(np.isfinite(result.densities[k])), f"Non-finite densities in regime {k}"

    def test_error_history_recorded(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=5,
        )
        result = iterator.solve()
        assert len(result.error_history) > 0
        assert len(result.error_history) <= 5

    def test_iterations_reported(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=3,
        )
        result = iterator.solve()
        assert result.iterations > 0
        assert result.iterations <= 3


class TestRegimeSwitchingUpdateSchemes:
    """Test Jacobi vs Gauss-Seidel update schemes."""

    def test_gauss_seidel_default(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
        )
        assert iterator._update_scheme == "gauss_seidel"

    def test_jacobi_scheme(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            update_scheme="jacobi",
            max_iterations=3,
        )
        result = iterator.solve()
        assert isinstance(result, RegimeSwitchingResult)
        assert np.all(np.isfinite(result.values[0]))


class TestRegimeSwitchingGetResults:
    """Test get_results() interface."""

    def test_get_results_after_solve(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=3,
        )
        iterator.solve()
        U, _M = iterator.get_results()
        assert U.shape[0] == problems[0].Nt + 1

    def test_get_results_before_solve_raises(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
        )
        with pytest.raises(RuntimeError, match="No solution computed"):
            iterator.get_results()


class TestRegimeSwitchingCrossTermSign:
    """Inter-regime coupling sign (#1251, 2026-06-10 audit).

    The DPP chain term ``sum_j Q[k,j](v^k - v^j)`` sits on the HJB LHS, and the HJB
    solver subtracts the source (``Phi_U -= source_term``), so the source must equal
    ``-cross``. A ``+cross`` source flips the inter-regime value coupling. The FP source
    (``+inflow Q[j,k]m^j - outflow Q[k,j]m^k``) was already correct and must be unchanged.
    """

    def _iterator(self):
        problems, config, hjbs, fps = _make_2regime_system()
        return (
            RegimeSwitchingIterator(problems=problems, regime_config=config, hjb_solvers=hjbs, fp_solvers=fps),
            problems,
            config.transition_matrix,
        )

    def test_hjb_source_is_negative_cross_term(self):
        iterator, problems, Q = self._iterator()
        Nt, N = problems[0].Nt, 5
        x = np.linspace(0.0, 1.0, N)
        v0, v1 = 2.0, 1.0
        Us_full = [v0 * np.ones((Nt + 1, N)), v1 * np.ones((Nt + 1, N))]

        src0 = iterator._make_hjb_source(0, 2, Q, Us_full, [None, None])
        np.testing.assert_allclose(src0(0.0, x), -Q[0, 1] * (v0 - v1) * np.ones(N))

        src1 = iterator._make_hjb_source(1, 2, Q, Us_full, [None, None])
        np.testing.assert_allclose(src1(0.0, x), -Q[1, 0] * (v1 - v0) * np.ones(N))

    def test_fp_source_sign_unchanged(self):
        iterator, problems, Q = self._iterator()
        Nt, N = problems[0].Nt, 5
        x = np.linspace(0.0, 1.0, N)
        m0, m1 = 0.7, 0.3
        Ms = [m0 * np.ones((Nt + 1, N)), m1 * np.ones((Nt + 1, N))]

        src0 = iterator._make_fp_source(0, 2, Q, Ms)
        np.testing.assert_allclose(src0(0.0, x), (Q[1, 0] * m1 - Q[0, 1] * m0) * np.ones(N))


class TestRegimeFPDriftConvention:
    """FP drift-convention routing (#1315, Refs #1043).

    After the v0.18.6 rename ``drift_field`` is the velocity ``alpha*``, not the value
    function. For a smooth separable ``H`` the FP solver must receive ``U`` via
    ``potential_field`` and derive ``alpha*`` itself; passing ``U`` as ``drift_field`` bypasses
    ``resolve_fp_drift_kwargs`` and converges to a wrong equilibrium (silent-wrong-physics).
    This pins the per-regime FP solve to route ``U`` via ``potential_field`` so the bypass
    cannot silently reopen.
    """

    def test_fp_receives_U_via_potential_field_not_drift(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
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

        iterator.solve()

        assert captured, "FP solver was never called"
        for k, keys in captured:
            assert "potential_field" in keys, (
                f"regime {k}: value function not routed via potential_field (kwargs={keys}); "
                "#1315 drift-convention bypass reopened"
            )
            assert "drift_field" not in keys, (
                f"regime {k}: value function wrongly passed as drift_field (velocity alpha*); "
                "#1315 silent-wrong-equilibrium"
            )

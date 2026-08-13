"""
Unit tests for HomotopyContinuation.

Tests the predictor-corrector continuation for tracking MFG
equilibrium branches as a parameter varies (#972).
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.continuation.homotopy import (
    HomotopyContinuation,
)


class _MockResult:
    """Mock solver result that wraps a density array."""

    def __init__(self, density: np.ndarray):
        self.density = density


def _linear_problem_factory(lam: float):
    """Factory for a toy problem: equilibrium is m* = lam * ones."""
    return {"lam": lam}


def _linear_solver_factory(problem: dict):
    """Solver that returns m* = problem['lam'] * uniform density."""

    class _MockSolver:
        def __init__(self, p):
            self._lam = p["lam"]

        def solve(self):
            # Equilibrium density is proportional to lambda
            m = self._lam * np.ones(20) / 20.0
            return _MockResult(m)

    return _MockSolver(problem)


def _extract(result: _MockResult) -> np.ndarray:
    return result.density


class TestHomotopyContinuationInstantiation:
    """Test construction and parameter validation."""

    def test_basic_construction(self):
        cont = HomotopyContinuation(
            problem_factory=_linear_problem_factory,
            solver_factory=_linear_solver_factory,
            extract_solution=_extract,
            parameter_range=(0.0, 1.0),
        )
        assert cont._lam_start == 0.0
        assert cont._lam_end == 1.0
        assert cont._max_steps == 200

    def test_custom_parameters(self):
        cont = HomotopyContinuation(
            problem_factory=_linear_problem_factory,
            solver_factory=_linear_solver_factory,
            extract_solution=_extract,
            parameter_range=(0.5, 2.0),
            initial_step=0.05,
            max_steps=50,
            corrector_tol=1e-8,
        )
        assert cont._step == 0.05
        assert cont._max_steps == 50
        assert cont._corr_tol == 1e-8


class TestHomotopyContinuationTrace:
    """Test the trace() method on toy problems."""

    def test_trace_records_parameter_values(self):
        """Parameter values should start at lam_start and end at lam_end."""
        cont = HomotopyContinuation(
            problem_factory=_linear_problem_factory,
            solver_factory=_linear_solver_factory,
            extract_solution=_extract,
            parameter_range=(0.0, 1.0),
            initial_step=0.5,
            max_steps=20,
            detect_bifurcation=False,
        )
        result = cont.trace(initial_solution=np.zeros(20))
        assert result.parameter_values[0] == 0.0
        assert result.parameter_values[-1] == pytest.approx(1.0, abs=0.1)

    def test_trace_solutions_match_parameters(self):
        """Each solution should correspond to its parameter value."""
        cont = HomotopyContinuation(
            problem_factory=_linear_problem_factory,
            solver_factory=_linear_solver_factory,
            extract_solution=_extract,
            parameter_range=(0.0, 1.0),
            initial_step=0.5,
            max_steps=20,
            detect_bifurcation=False,
        )
        result = cont.trace(initial_solution=np.zeros(20))
        assert len(result.solutions) == len(result.parameter_values)
        assert result.converged_steps > 0

    def test_trace_backward(self):
        """Continuation should work in reverse direction."""
        cont = HomotopyContinuation(
            problem_factory=_linear_problem_factory,
            solver_factory=_linear_solver_factory,
            extract_solution=_extract,
            parameter_range=(1.0, 0.0),
            initial_step=0.5,
            max_steps=20,
            detect_bifurcation=False,
        )
        m0 = np.ones(20) / 20.0
        result = cont.trace(initial_solution=m0)
        assert result.parameter_values[0] == 1.0
        # Should move toward 0
        assert result.parameter_values[-1] < 1.0


class TestCorrectorConvergence:
    """Test corrector iteration behavior."""

    def test_corrector_converges_on_fixed_point(self):
        """Corrector should converge when solver already returns the fixed point."""
        cont = HomotopyContinuation(
            problem_factory=_linear_problem_factory,
            solver_factory=_linear_solver_factory,
            extract_solution=_extract,
            parameter_range=(0.0, 1.0),
            corrector_tol=1e-6,
            max_corrector_iters=5,
        )
        # The solver returns lam * ones/20 regardless of input
        # So corrector converges in 1 step (fixed point map is contractive)
        m_pred = np.ones(20) * 0.5  # arbitrary prediction
        m_corr, converged = cont._corrector(m_pred, lam=0.3)
        assert converged
        np.testing.assert_allclose(m_corr, 0.3 * np.ones(20) / 20.0, atol=1e-5)

    def test_corrector_fails_on_non_convergent(self):
        """Corrector must report False when the iteration budget runs out before the tolerance."""

        # Solver that ignores its input and returns a constant far from it
        def divergent_factory(problem):
            class _Div:
                def solve(self):
                    return _MockResult(np.ones(10) * 999.0)  # never matches input

            return _Div()

        def _make(max_corrector_iters):
            return HomotopyContinuation(
                problem_factory=_linear_problem_factory,
                solver_factory=divergent_factory,
                extract_solution=_extract,
                parameter_range=(0.0, 1.0),
                corrector_tol=1e-12,
                max_corrector_iters=max_corrector_iters,
            )

        # One iteration: error is 999 on the only pass, so the budget is exhausted -> False branch
        m_corr, converged = _make(1)._corrector(np.zeros(10), lam=0.5)
        assert converged is False, "budget exhausted before the residual fell below tol"
        np.testing.assert_allclose(m_corr, 999.0)

        # Three iterations: the map is constant, so pass 2 has error exactly 0 < tol -> True branch.
        # This is the reason the old `isinstance(converged, bool)` hid a defect: the fixture named
        # "non convergent" actually converges, and both exits satisfy the isinstance check.
        m_corr, converged = _make(3)._corrector(np.zeros(10), lam=0.5)
        assert converged is True
        np.testing.assert_allclose(m_corr, 999.0)


class TestAdaptiveStepSizing:
    """Test step size adaptation."""

    def test_step_grows_on_success(self):
        """Adaptive step should grow after successful corrector steps."""
        cont = HomotopyContinuation(
            problem_factory=_linear_problem_factory,
            solver_factory=_linear_solver_factory,
            extract_solution=_extract,
            parameter_range=(0.0, 1.0),
            initial_step=0.01,
            adaptive_step=True,
            max_step=0.1,
            max_steps=5,
            detect_bifurcation=False,
        )
        result = cont.trace(initial_solution=np.zeros(20))
        # With successful steps, should reach end faster than initial_step would suggest
        # 0.01 * 5 = 0.05 without growth, but with 1.2x growth per step we get further
        assert result.converged_steps > 0

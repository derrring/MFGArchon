"""Issue #1546: FPNetworkSolver.forward_step must fail loud, not silently mis-step.

forward_step never precomputed transition rates, so a fresh solver hit the wrong-signed legacy drift
and a post-solve call reused stale rates; it also skipped the node-BC / mass-renorm gate. It has no
callers. It now raises NotImplementedError, and the legacy fallback in _compute_drift_term (reachable
only off the solve_fp_system path) also raises rather than resurrecting the wrong-signed drift.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.network_solvers.fp_network import FPNetworkSolver
from mfgarchon.extensions.topology import NetworkMFGProblem
from mfgarchon.geometry.graph.network_geometry import GridNetwork


def _solver() -> tuple[FPNetworkSolver, int]:
    problem = NetworkMFGProblem(geometry=GridNetwork(width=3, height=1), T=0.5, Nt=5)
    return FPNetworkSolver(problem), problem.num_nodes


def test_forward_step_fails_loud_1546():
    solver, n = _solver()
    m0 = np.ones(n) / n
    with pytest.raises(NotImplementedError, match=r"forward_step|1546"):
        solver.forward_step(m0, np.zeros(n), 0.1)


def test_compute_drift_term_fails_loud_without_precomputed_rates_1546():
    """The legacy fallback (fresh solver, no precomputed rates) must raise, not use the wrong-signed drift."""
    solver, n = _solver()
    assert solver._current_rates is None
    with pytest.raises(RuntimeError, match=r"not precomputed|1546|1474"):
        solver._compute_drift_term(0, np.ones(n) / n, np.zeros(n), 0.0)


def test_solve_fp_system_still_works_1546():
    """solve_fp_system precomputes rates each step, so it is unaffected by the fail-loud fallback."""
    solver, n = _solver()
    m0 = np.ones(n) / n
    U = np.zeros((6, n))
    M = solver.solve_fp_system(M_initial=m0, potential_field=U, show_progress=False)
    assert np.isfinite(M).all()
    assert M.shape[0] == 6

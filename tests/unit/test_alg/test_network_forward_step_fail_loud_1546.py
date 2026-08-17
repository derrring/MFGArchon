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


def _solver(diffusion_coefficient: float | None = None) -> tuple[FPNetworkSolver, int]:
    problem = NetworkMFGProblem(geometry=GridNetwork(width=3, height=1), T=0.5, Nt=5)
    return FPNetworkSolver(problem, diffusion_coefficient=diffusion_coefficient), problem.num_nodes


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
    """solve_fp_system precomputes rates each step, so it is unaffected by the fail-loud fallback.

    Checked against the closed-form graph heat kernel.  The uniform m0 this test used to start
    from is already the stationary law of the 3-node path graph, so *every* assertion below --
    including mass conservation -- was satisfied by a solver that returned M_initial unchanged,
    which is the mass-leak / dead-solve failure mode #1546 is about.  A concentrated m0 makes
    the transport real.
    """
    T, D = 0.5, 0.1
    solver, n = _solver(diffusion_coefficient=D)
    m0 = np.zeros(n)
    m0[0] = 1.0
    U = np.zeros((6, n))
    M = solver.solve_fp_system(M_initial=m0, potential_field=U, show_progress=False)
    assert np.isfinite(M).all()
    assert M.shape[0] == 6

    # Closed graph, no sources: total mass is conserved exactly, and the density stays a density.
    np.testing.assert_allclose(M.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_array_equal(M[0], m0)
    assert (M >= 0.0).all()

    # External oracle: U == 0 means no drift, so the generator is the graph Laplacian L = deg - A
    # and m(T) = m0 @ exp(-D*L*T).  The path graph P3 has eigenpairs (0, [1,1,1]/3),
    # (1, [1,0,-1]/2), (3, [1,-2,1]/6), giving m(T) in closed form with no matrix exponential.
    e1, e3 = np.exp(-D * T), np.exp(-3.0 * D * T)
    m_exact = np.array([1 / 3 + e1 / 2 + e3 / 6, 1 / 3 - e3 / 3, 1 / 3 - e1 / 2 + e3 / 6])
    # Measured gap 6.6e-4 at Nt=5, and it halves under each doubling of Nt (first order in dt),
    # so 5e-3 is ~7x margin.  A solver that returned m0 unchanged would sit 4.8e-2 away.
    np.testing.assert_allclose(M[-1], m_exact, atol=5e-3)

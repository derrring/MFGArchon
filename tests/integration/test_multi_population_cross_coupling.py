"""Multi-population HJB cross-coupling (Issue #1157).

Before the fix, `MultiPopulationIterator` computed the cross-population-bound
Hamiltonian but never passed it to `solve_hjb_system`, so each population's HJB
solved against the uncoupled `problem.hamiltonian_class` — the cross-density
coupling reached the FP drift but not the value function (a silently wrong,
half-coupled equilibrium). These tests pin the fix:

- the HJB now responds to the other population's density (coupled != decoupled);
- a single-population run is byte-identical (the override is not sent for K==1);
- a backend that does not thread the override fails loud rather than silently
  decoupling.

The cross-coupling here enters EXCLUSIVELY through the HJB Hamiltonian term:
``SeparableHamiltonian.optimal_control`` is momentum-only, so the FP drift never
sees the bound density. That isolates the HJB-coupling path being fixed.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.multi_population import MultiPopulationProblem

_NX, _NT, _T, _SIG = 20, 8, 1.0, 0.15


def _make_problem(k, cross, K):
    """Population-k problem whose coupling f_k(m) = cross * (other population's density).

    The coupling distinguishes the stacked cross-density (length K*grid) from a
    single-population density and is scalar-safe (so MFGComponents validation passes).
    """

    def coupling(m, pop_idx=k, cross=cross, K=K):
        m = np.asarray(m, float)
        if m.ndim >= 1 and m.shape[-1] % K == 0 and m.shape[-1] >= 2 * K:
            grid = m.shape[-1] // K
            return cross * m.reshape(*m.shape[:-1], K, grid)[..., 1 - pop_idx, :]
        return np.zeros_like(m)

    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=coupling,
        coupling_dm=lambda m: np.zeros_like(np.asarray(m, float)),
        population_index=k,
    )
    comps = MFGComponents(
        m_initial=lambda xx, kk=k: np.exp(-((np.asarray(xx) - (0.3 + 0.4 * kk)) ** 2) / 0.02),
        u_terminal=lambda xx: np.asarray(xx) * 0.0,
        hamiltonian=H,
    )
    return MFGProblem(Nx=[_NX], Nt=_NT, T=_T, sigma=_SIG, components=comps)


def _solve(K, cross, max_iterations=6):
    probs = [_make_problem(k, cross, K) for k in range(K)]
    multi = MultiPopulationProblem(populations=probs, population_names=[f"P{k}" for k in range(K)])
    it = MultiPopulationIterator(
        multi,
        [HJBFDMSolver(p) for p in probs],
        [FPFDMSolver(p) for p in probs],
        relaxation=0.5,
    )
    return it.solve(max_iterations=max_iterations, tolerance=1e-10)


def test_hjb_sees_cross_density_bug_1157():
    """LOAD-BEARING: a genuinely cross-coupled 2-population MFG must differ from the
    decoupled (cross=0) solve. FAILS on the pre-#1157 code (coupled == decoupled bit-for-bit,
    because the bound Hamiltonian never reached the HJB)."""
    K = 2
    coupled = _solve(K, cross=2.0)
    decoupled = _solve(K, cross=0.0)
    dU = max(np.max(np.abs(np.asarray(coupled.U[k]) - np.asarray(decoupled.U[k]))) for k in range(K))
    assert dU > 1e-6, f"cross-coupling had no effect on the HJB value function (bug #1157): dU={dU:.3e}"


def test_single_population_byte_identical():
    """K==1 has no cross-coupling: the iterator must not send an override, so the solve is
    byte-identical regardless of the (irrelevant) coupling strength."""
    c = _solve(1, cross=2.0, max_iterations=4)
    d = _solve(1, cross=0.0, max_iterations=4)
    assert np.array_equal(np.asarray(c.U[0]), np.asarray(d.U[0]))
    assert np.array_equal(np.asarray(c.M[0]), np.asarray(d.M[0]))


def test_nonfdm_backend_multipop_fails_loud():
    """A K>1 run on an HJB backend that does not thread the cross-density override must fail
    loud (the half-coupled silent-wrong equilibrium is the bug), not run silently."""

    class _StubHJB:
        # Deliberately lacks _honors_multipop_hamiltonian_override.
        def solve_hjb_system(self, *args, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("solve_hjb_system should not be called; iterator must fail loud first")

    K = 2
    probs = [_make_problem(k, 2.0, K) for k in range(K)]
    multi = MultiPopulationProblem(populations=probs, population_names=["A", "B"])
    it = MultiPopulationIterator(
        multi,
        [_StubHJB() for _ in range(K)],
        [FPFDMSolver(p) for p in probs],
        relaxation=0.5,
    )
    with pytest.raises(NotImplementedError, match="1157"):
        it.solve(max_iterations=2, tolerance=1e-10)

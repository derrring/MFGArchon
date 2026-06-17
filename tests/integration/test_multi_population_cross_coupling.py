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
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

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
    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[_NX + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        Nt=_NT,
        T=_T,
        sigma=_SIG,
        components=comps,
    )


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


def test_k1_matches_single_population_fp_convention():
    """A K=1 multi-pop solve must converge to the SAME fixed point as the single-pop
    FixedPointIterator (Issue #1043).

    Two stacked convention bugs made K=1 diverge from single-pop. (1) The iterator computed its
    own *node*-centered velocity and always passed it as an explicit ``drift_field`` (~116% off);
    routing through the shared ``resolve_fp_drift_kwargs`` fixed the convention. (2) Historically
    the iterator bound the Hamiltonian (``BoundHamiltonian``), which failed
    ``isinstance(SeparableHamiltonian)`` so the resolver took the *velocity* path while single-pop
    (unbound H) took *potential* → ``||F_FP|| ~ O(1)`` (not a coupled fixed point). The lock-faithful
    migration (Issue #1071) retired the wrapper: the iterator now passes the population's own
    (unbound) H + a ``cross_density`` trajectory, so the smoothness dispatch reads the plain H. With
    both fixes K=1 matches single-pop **exactly** (bounded only by the Picard tolerance), so a tight
    threshold catches any reintroduction of either bug."""
    from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator

    prob = _make_problem(0, cross=0.0, K=1)
    sp = FixedPointIterator(prob, hjb_solver=HJBFDMSolver(prob), fp_solver=FPFDMSolver(prob), relaxation=0.5).solve(
        max_iterations=120, tolerance=1e-6, verbose=False
    )
    U_sp, M_sp = sp[0], sp[1]

    mp = _solve(1, cross=0.0, max_iterations=200)
    U_mp, M_mp = np.asarray(mp.U[0]), np.asarray(mp.M[0])

    u_diff = np.linalg.norm(U_mp - U_sp) / (np.linalg.norm(U_sp) + 1e-12)
    m_diff = np.linalg.norm(M_mp - M_sp) / (np.linalg.norm(M_sp) + 1e-12)
    assert u_diff < 1e-4, f"K=1 multi-pop U {u_diff * 100:.3f}% off single-pop (FP convention re-forked?)"
    assert m_diff < 1e-4, f"K=1 multi-pop density {m_diff * 100:.3f}% off single-pop (FP convention re-forked?)"
    # mass conservation preserved
    assert np.allclose(M_mp.sum(axis=-1), M_mp[0].sum(), rtol=1e-6)


def test_nonfdm_backend_multipop_fails_loud():
    """A K>1 run on an HJB backend that does not thread the cross-density override must fail
    loud (the half-coupled silent-wrong equilibrium is the bug), not run silently."""

    class _StubHJB:
        # Deliberately lacks _honors_multipop_cross_density.
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
    with pytest.raises(NotImplementedError, match="1071"):
        it.solve(max_iterations=2, tolerance=1e-10)


def test_fp_velocity_consumes_cross_density_1071():
    """Issue #1071: the FP drift velocity must consume the ``cross_density`` trajectory — at each
    integer timestep ``optimal_control`` sees the stacked density (the other populations), not the
    own-population density. The integration test's separable H has a momentum-only
    ``optimal_control`` (so it would not exercise this), hence a deliberately ``m``-dependent test
    Hamiltonian; the velocity with a cross-density must differ from the own-density velocity."""
    from mfgarchon.alg.numerical.coupling.fixed_point_utils import compute_fp_velocity_field
    from mfgarchon.core.hamiltonian import HamiltonianBase

    K, Nx, Nt, T = 2, _NX + 1, _NT, _T
    dt = T / Nt

    class _MDepH(HamiltonianBase):
        """optimal_control reads the stacked density (the OTHER population) so the velocity
        genuinely depends on the cross-density (gives the flow test teeth)."""

        def __init__(self, population_index, k_pops):
            self.population_index = population_index
            self._K = k_pops

        def __call__(self, x, m, p, t=0.0):
            return 0.5 * np.asarray(p, float) ** 2

        def optimal_control(self, x, m, p, t=0.0):
            m = np.asarray(m, float)
            p = np.asarray(p, float)
            other = 0.0
            if m.ndim >= 1 and m.shape[-1] % self._K == 0 and m.shape[-1] >= 2 * self._K:
                grid = m.shape[-1] // self._K
                other = float(m.reshape(*m.shape[:-1], self._K, grid)[..., 1 - self.population_index, :].mean())
            return p.ravel() + other

    class _Geom:
        def get_grid_spacing(self):
            return [1.0 / _NX]

        def get_bounds(self):
            return [(0.0,), (1.0,)]

    class _Prob:
        geometry = _Geom()

    prob = _Prob()
    prob.dt = dt
    rng = np.random.RandomState(1)
    M = [np.abs(rng.rand(Nt + 1, Nx)) + 0.1 for _ in range(K)]
    M = [m / m.sum(axis=-1, keepdims=True) for m in M]
    m_all = np.concatenate(M, axis=-1)  # (Nt+1, K*Nx)
    U = rng.rand(Nt + 1, Nx)

    for k in range(K):
        h = _MDepH(k, K)
        v_cross = compute_fp_velocity_field(prob, U, M[k], h, cross_density=m_all)
        v_own = compute_fp_velocity_field(prob, U, M[k], h)  # own face density (no cross)
        assert not np.array_equal(v_cross, v_own), (
            f"pop {k}: cross_density did not flow into the FP velocity (== own-density result)"
        )

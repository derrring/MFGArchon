#!/usr/bin/env python3
"""Issue #1430 (Strand C — cross-path pinning): the 1D and nD HJB-FDM specializations must couple
U^n to the density at the SAME time level, M[n].

Context. The two paths are distinct code — ``base_hjb.solve_hjb_system_backward`` (1D) and
``HJBFDMSolver._solve_hjb_nd`` (nD) — and are *intentionally different discretizations* (the 1D path
is an optimized, BC-aware solver; Issue #1430 keeps the fork). They converge to the same continuous
solution only at O(h): on a y-invariant LQ problem the 1D-vs-nD value gap is ~6% at N=16 and falls
first-order under refinement (measured max|U_1d - U_2d|/|U| = 6.3e-2, 4.0e-2, 2.8e-2, 1.7e-2 at
N=16/24/32/48). So an end-to-end *value* comparison cannot sharply detect an O(dt) *convention* drift
between the paths — the O(h) scheme gap dominates it. This pin therefore asserts the shared
CONVENTION at the seam, not the discretized value.

Issue #1423/#1437: the nD path coupled U^n to M[n+1] while the 1D path used M[n] ("BUG #7 FIX") — a
silent O(dt) cross-path off-by-one. ``test_hjb_fdm_nd_coupling_time_level.py`` pins only the nD path;
this asserts the 1D path AND that the two AGREE, so a future divergence in *either* path's coupling
index fails CI. It would have caught #1423.

Method. Feed a density whose time slices are strictly distinct (M[k] is the constant k+1). Spy each
path's per-timestep solve and record which slice it consumes when solving U^n: the 1D path by
monkeypatching ``base_hjb.solve_hjb_timestep_newton`` (3rd positional arg is M[n]; the ``t_idx_n``
kwarg names the step), the nD path by wrapping ``solver._solve_single_timestep``. Both must yield the
M[n] sequence ``[Nt, Nt-1, ..., 1]``; a pre-#1437 nD path would yield ``[Nt+1, Nt, ..., 2]`` (M[n+1]).

Why B0 (no baseline risk): pure test addition validating existing (post-#1437) behavior; no
production code is touched. Verified passing on current main. Refs #1430, #1423, #1071. Generalizes
the #1421/#1422 pinning pattern.
"""

import warnings

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver, base_hjb
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

N = 6
T = 0.2
SIGMA = 0.1


def _components() -> MFGComponents:
    """Separable LQ Hamiltonian with a linear running cost f(m) = m, so the density enters the HJB
    and the coupling time index is observable. Dimension-agnostic (sums over spatial dimensions)."""
    return MFGComponents(
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: m,
            coupling_dm=lambda m: 1.0,
        ),
    )


def _problem_1d(Nt: int) -> MFGProblem:
    geom = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[N + 1],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    return MFGProblem(geometry=geom, components=_components(), T=T, Nt=Nt, sigma=SIGMA)


def _problem_2d(Nt: int) -> MFGProblem:
    geom = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[N + 1, N + 1],
        boundary_conditions=no_flux_bc(dimension=2),
    )
    return MFGProblem(geometry=geom, components=_components(), T=T, Nt=Nt, sigma=SIGMA)


class TestHJBFDMCouplingIndexCrossPath:
    """The 1D and nD HJB-FDM paths must agree on the coupling time index (Issue #1423 convention)."""

    @pytest.mark.parametrize("Nt", [5, 7])
    def test_1d_and_nd_couple_un_to_mn(self, monkeypatch, Nt):
        p1 = _problem_1d(Nt)
        p2 = _problem_2d(Nt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s1 = HJBFDMSolver(p1, solver_type="newton")
            s2 = HJBFDMSolver(p2, solver_type="fixed_point")

        # Density with strictly distinct, identifiable time slices: M[k] is the constant (k+1).
        shape_nd = s2.shape
        m1 = np.stack([np.full(N + 1, float(k + 1)) for k in range(Nt + 1)])
        m2 = np.stack([np.full(shape_nd, float(k + 1)) for k in range(Nt + 1)])
        u_terminal_1d = np.zeros(N + 1)
        u_terminal_nd = np.zeros(shape_nd)

        # --- 1D path: spy the module-level per-timestep solver. Positional args are
        # (U_{n+1}, U_n_prev, M_n_prev_picard, problem, ...); t_idx_n names the step being solved. ---
        captured_1d: list[tuple[int, float]] = []
        orig_1d = base_hjb.solve_hjb_timestep_newton

        def spy_1d(*args, **kwargs):
            captured_1d.append((kwargs.get("t_idx_n"), float(np.asarray(args[2]).flat[0])))
            return orig_1d(*args, **kwargs)

        monkeypatch.setattr(base_hjb, "solve_hjb_timestep_newton", spy_1d)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s1.solve_hjb_system(m1, u_terminal_1d, U_coupling_prev=np.zeros((Nt + 1, N + 1)))

        # --- nD path: wrap the per-timestep solve; record the coupling slice in backward order. ---
        captured_nd: list[float] = []
        orig_nd = s2._solve_single_timestep

        def spy_nd(u_next, m_coupling, u_guess, sigma_at_n, Sigma_at_n, **kw):
            captured_nd.append(float(np.asarray(m_coupling).flat[0]))
            return orig_nd(u_next, m_coupling, u_guess, sigma_at_n, Sigma_at_n, **kw)

        monkeypatch.setattr(s2, "_solve_single_timestep", spy_nd)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s2.solve_hjb_system(m2, u_terminal_nd, U_coupling_prev=np.zeros((Nt + 1, *shape_nd)))

        # The backward loop solves U^n for n = Nt-1, ..., 0; the correct coupling slice is M[n], whose
        # constant value is (n+1). So both paths must produce [Nt, Nt-1, ..., 1].
        expected = [float(n + 1) for n in range(Nt - 1, -1, -1)]
        buggy = [float(n + 2) for n in range(Nt - 1, -1, -1)]  # pre-#1437 nD path (M[n+1])

        # (i) 1D per-call invariant: solving U^{t_idx} consumes M[t_idx] (value t_idx + 1), never M[t_idx+1].
        for t_idx, m_value in captured_1d:
            assert m_value == float(t_idx + 1), (
                f"1D HJB-FDM coupled U^{t_idx} to M[{int(m_value) - 1}] (value {m_value}), not M[{t_idx}] "
                f"— violates the Issue #1423 same-time-level coupling convention."
            )

        sequence_1d = [m_value for _, m_value in captured_1d]
        assert sequence_1d == expected, f"1D coupling sequence {sequence_1d} != M[n] {expected}."
        assert captured_nd == expected, (
            f"nD HJB-FDM coupling sequence {captured_nd} != M[n] {expected} "
            f"(a pre-#1437 M[n+1] path would yield {buggy}). Issue #1423/#1437."
        )

        # (ii) cross-path agreement: the two specializations use the SAME coupling time index. A future
        # fix or refactor that shifts one path's index without the other fails here.
        assert sequence_1d == captured_nd, (
            f"1D and nD HJB-FDM disagree on the coupling time index: 1D {sequence_1d} vs nD "
            f"{captured_nd}. Cross-path shared-logic divergence (Issue #1430 / #1423)."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

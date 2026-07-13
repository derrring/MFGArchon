"""Issue #1607: opt-in analytic inner-Newton Jacobian for HJBFDMSolver.

``HJBFDMSolver.__init__`` always injects a NumPy backend, so ``compute_hjb_jacobian``'s fast
analytic chain-rule path (gated on ``backend is None``) never fires for a single-population solve --
every inner Newton step falls back to the O(Nx^2) per-point finite-difference Jacobian. The
``analytic_jacobian=True`` flag routes the single-pop solve through the same ``backend=None`` path the
multi-population branch already uses (batch residual + analytic Jacobian), ~17x faster per solve.

Pinned invariants:
  1. Default is unchanged (flag defaults False -> FD path; the flag is off unless explicitly set).
  2. The flag actually ROUTES: True -> the solver calls solve_hjb_system_backward with backend=None
     (analytic path); False -> backend is the NumPy backend (FD path). This discriminates the flag
     being a silent no-op -- the equivalence test below cannot (both paths agree, so a dead flag
     still "agrees"). Reverting the gate in hjb_fdm.py to ignore self._analytic_jacobian makes THIS
     test fail while leaving the equivalence test green.
  3. The flag is NumPy-only and fails loud on any other backend (the analytic assembly is a NumPy
     kernel; silently ignoring it would train a false "it's faster" belief).
  4. The analytic path converges to the SAME fixed point as the FD path (to tolerance): opt-in speed
     must not buy a different solution.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.backends import NumPyBackend
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _tiny_problem(Nx: int = 13, Nt: int = 6) -> MFGProblem:
    """A 1D LQ-MFG small enough that even the O(Nx^2) FD-Jacobian path solves in a few seconds."""
    ham = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.05 * m,
        coupling_dm=lambda m: 0.05,
    )
    return MFGProblem(
        model=Model(hamiltonian=ham, sigma=0.2),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[Nx], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(
            u_terminal=lambda x: (x - 0.5) ** 2,
            m_initial=lambda x: np.exp(-20 * (x - 0.5) ** 2),
            T=0.5,
        ),
        Nt=Nt,
    )


def test_analytic_jacobian_defaults_off():
    """Default construction leaves the flag off -> the existing FD-Jacobian path (unchanged behavior)."""
    solver = HJBFDMSolver(_tiny_problem())
    assert solver._analytic_jacobian is False


def test_analytic_jacobian_routes_backend_none(monkeypatch):
    """The flag must actually ROUTE the solve: True -> solve_hjb_system_backward(backend=None) (analytic
    path); False -> backend is a NumPyBackend (FD path). Spy on the kwarg to pin the wiring directly.

    DISCRIMINATING (the reason this test exists): the equivalence test below passes even if the flag is
    a silent no-op, because a dead flag leaves both solvers on the FD path -> they trivially "agree." A
    no-op is the likelier regression than a wrong-operator wiring. Reverting the gate in hjb_fdm.py
    (`effective_backend = None if cross_density is not None else self.backend`, dropping the
    `self._analytic_jacobian` term) makes the flag-True assertion here FAIL (backend would be a
    NumPyBackend, not None) while the equivalence test stays green."""
    import mfgarchon.alg.numerical.hjb_solvers.base_hjb as base_hjb

    def _capture_backends(analytic: bool) -> list:
        seen: list = []
        original = base_hjb.solve_hjb_system_backward

        def _spy(*args, **kwargs):
            seen.append(kwargs.get("backend"))
            return original(*args, **kwargs)

        monkeypatch.setattr(base_hjb, "solve_hjb_system_backward", _spy)
        p = _tiny_problem()
        FixedPointIterator(p, HJBFDMSolver(p, analytic_jacobian=analytic), FPFDMSolver(p)).solve(
            max_iterations=1, tolerance=1e-6, verbose=False
        )
        monkeypatch.setattr(base_hjb, "solve_hjb_system_backward", original)
        return seen

    on = _capture_backends(analytic=True)
    off = _capture_backends(analytic=False)

    assert on, "the HJB backward solver must be called at least once"
    assert all(b is None for b in on), f"analytic_jacobian=True must route backend=None, saw {on}"
    assert off, "the HJB backward solver must be called at least once"
    assert all(isinstance(b, NumPyBackend) for b in off), (
        f"analytic_jacobian=False must route the NumPy backend (FD path), saw {off}"
    )


def test_analytic_jacobian_rejects_non_numpy_backend(monkeypatch):
    """The analytic assembly is a NumPy-only kernel; requesting it on a non-NumPy backend must fail loud,
    not silently fall back (which would make ``analytic_jacobian=True`` a no-op the caller trusts).
    Torch-free: stub ``create_backend`` to return a non-NumPy backend so the guard is exercised on every
    runner, not only those with torch installed."""

    class _StubBackend:
        """A non-NumPyBackend; the guard must reject it under analytic_jacobian=True."""

    monkeypatch.setattr("mfgarchon.backends.create_backend", lambda *a, **k: _StubBackend())

    with pytest.raises(ValueError, match="NumPy-only"):
        HJBFDMSolver(_tiny_problem(), analytic_jacobian=True)


def test_analytic_jacobian_matches_fd_solution():
    """The opt-in analytic Jacobian must reach the SAME fixed point as the default FD Jacobian (both
    solve the identical residual; only the Newton Jacobian approximation differs, so they share the
    root). Guards the numerics (invariant 4), NOT the flag wiring -- a silent no-op also passes this
    (both run FD -> trivially agree); the routing test above pins the wiring. (Observed on this
    problem: max|dU|~7e-8, max|dM|~2e-6; bounds are generous headroom, tight enough to catch a real
    divergence.)"""
    p_fd, p_an = _tiny_problem(), _tiny_problem()

    res_fd = FixedPointIterator(p_fd, HJBFDMSolver(p_fd, analytic_jacobian=False), FPFDMSolver(p_fd)).solve(
        max_iterations=120, tolerance=1e-6, verbose=False
    )
    res_an = FixedPointIterator(p_an, HJBFDMSolver(p_an, analytic_jacobian=True), FPFDMSolver(p_an)).solve(
        max_iterations=120, tolerance=1e-6, verbose=False
    )

    assert res_fd.converged, "FD baseline must converge on the tiny problem"
    assert res_an.converged, "analytic path must converge on the tiny problem"
    assert np.max(np.abs(res_fd.U - res_an.U)) < 1e-4, "value function diverges between paths"
    assert np.max(np.abs(res_fd.M - res_an.M)) < 1e-3, "density diverges between paths"

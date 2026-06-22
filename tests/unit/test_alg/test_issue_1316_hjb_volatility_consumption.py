"""Pinning tests for Issue #1316: HJB solvers accept volatility_field but ignore it.

The coupling iterator forwards ``volatility_field`` to ``solve_hjb_system`` for solvers
that declare it (signature-gated, ``coupling/base_mfg.py``). Before this fix:

  * ``HJBGFDMSolver`` declared ``volatility_field`` but never consumed it — the body
    resolved diffusion via ``_get_sigma_value()`` / ``problem.sigma`` only. So a
    spatially-varying field passed to FP was silently dropped by HJB, breaking the
    Picard fixed-point correspondence.
  * ``HJBGFDMSolver._get_sigma_value`` returned a HARDCODED ``1.0`` for a callable
    ``problem.sigma`` on the batch path (``point_idx=None``), despite the comment
    "use representative value (center of domain)" — a ~2x diffusion error.

This module pins:

  FIX-B — callable sigma batch path evaluates at the domain center, not 1.0.
  FIX-A — a volatility_field override is the authoritative diffusion source
          (replacing problem.sigma) and is wired in by solve_hjb_system.
  Default (volatility_field=None) path is unchanged (problem.sigma).
  SL / WENO fail loud (NotImplementedError) on a mismatched volatility_field
          instead of silently accept-and-ignore.

Refs #1316 (Refs #1248).
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import (
    HJBGFDMSolver,
    HJBSemiLagrangianSolver,
    HJBWENOSolver,
)
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _components():
    return MFGComponents(
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
    )


def _problem(sigma=0.3):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    return MFGProblem(geometry=grid, T=0.2, Nt=10, sigma=sigma, components=_components())


def _gfdm_solver(problem):
    pts = np.linspace(0.0, 1.0, 21).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HJBGFDMSolver(problem, collocation_points=pts, delta=0.2)


# ---------------------------------------------------------------------------
# FIX-B: callable sigma batch path -> center-of-domain eval (not 1.0)
# ---------------------------------------------------------------------------


def test_callable_sigma_batch_path_is_center_eval_not_one():
    """Issue #1316 FIX-B — _get_sigma_value(None) for callable sigma(x)=0.5+0.5x must
    evaluate at the domain center (~0.75 over [0,1]), NOT return the hardcoded 1.0."""
    solver = _gfdm_solver(_problem())
    # Make problem.sigma callable; collocation points span [0,1] -> center mean = 0.5.
    solver.problem.sigma = lambda x: 0.5 + 0.5 * float(np.atleast_1d(x)[0])

    resolved = solver._get_sigma_value(None)

    assert resolved == pytest.approx(0.75, abs=1e-12), (
        f"callable sigma batch path resolved to {resolved}; expected the center eval 0.75. "
        f"If it returned 1.0 the Issue #1316 hardcoded fallback has returned."
    )
    assert resolved != pytest.approx(1.0, abs=1e-9), "regressed to the hardcoded 1.0 fallback"


def test_callable_sigma_per_point_path_unchanged():
    """Sanity: the per-point path (point_idx given) still evaluates at that point."""
    solver = _gfdm_solver(_problem())
    solver.problem.sigma = lambda x: 0.5 + 0.5 * float(np.atleast_1d(x)[0])
    # Point 0 is x=0 -> 0.5; last point x=1 -> 1.0.
    assert solver._get_sigma_value(0) == pytest.approx(0.5, abs=1e-12)
    assert solver._get_sigma_value(solver.n_points - 1) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# FIX-A: volatility_field override is the authoritative diffusion source
# ---------------------------------------------------------------------------


def test_array_override_replaces_problem_sigma():
    """Issue #1316 FIX-A — a per-point array override is used (not problem.sigma): the
    batch path collapses to its mean, the per-point path indexes it; both differ from
    problem.sigma = 0.3."""
    problem = _problem(sigma=0.3)
    solver = _gfdm_solver(problem)
    field = np.full(solver.n_points, 0.8)
    solver._volatility_field_override = field

    batch = solver._get_sigma_value(None)
    assert batch == pytest.approx(0.8, abs=1e-12)
    assert batch != pytest.approx(float(problem.sigma), abs=1e-9), "override ignored; used problem.sigma"

    assert solver._get_sigma_value(5) == pytest.approx(0.8, abs=1e-12)


def test_callable_override_replaces_problem_sigma():
    """Issue #1316 FIX-A — a callable override evaluates at the center on the batch path."""
    problem = _problem(sigma=0.3)
    solver = _gfdm_solver(problem)
    solver._volatility_field_override = lambda x: 0.5 + 0.5 * float(np.atleast_1d(x)[0])

    batch = solver._get_sigma_value(None)
    assert batch == pytest.approx(0.75, abs=1e-12)
    assert batch != pytest.approx(float(problem.sigma), abs=1e-9), "override ignored; used problem.sigma"


def test_solve_hjb_system_wires_in_override():
    """Issue #1316 FIX-A — solve_hjb_system stores the volatility_field arg into the
    override attribute (the wiring that makes _get_sigma_value see it), and a default
    None solve leaves it None."""
    problem = _problem(sigma=0.3)
    solver = _gfdm_solver(problem)
    U_T = 0.5 * (np.linspace(0.0, 1.0, 21) - 0.5) ** 2
    M = np.ones((11, 21)) / 21
    field = np.full(solver.n_points, 0.8)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.solve_hjb_system(M_density=M, U_terminal=U_T, volatility_field=field)
    assert solver._volatility_field_override is field, "solve_hjb_system did not install the override"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.solve_hjb_system(M_density=M, U_terminal=U_T)
    assert solver._volatility_field_override is None, "override not reset on a default (None) solve"


def test_scalar_override_equal_to_sigma_is_byte_identical():
    """Convention agreement — forwarding the scalar problem.sigma as volatility_field
    (the iterator's redundant Issue #1248 forwarding) is byte-identical to not forwarding."""
    U_T = 0.5 * (np.linspace(0.0, 1.0, 21) - 0.5) ** 2
    M = np.ones((11, 21)) / 21

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u_none = np.asarray(_gfdm_solver(_problem(0.3)).solve_hjb_system(M_density=M, U_terminal=U_T))
        u_fwd = np.asarray(
            _gfdm_solver(_problem(0.3)).solve_hjb_system(M_density=M, U_terminal=U_T, volatility_field=0.3)
        )
    np.testing.assert_array_equal(u_fwd, u_none, err_msg="scalar volatility_field == sigma changed the solution")


# ---------------------------------------------------------------------------
# SL / WENO: fail loud on a mismatched volatility_field (no clean chokepoint)
# ---------------------------------------------------------------------------


def _sl_problem():
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    return MFGProblem(geometry=grid, T=0.2, Nt=10, sigma=0.3, components=_components())


@pytest.mark.parametrize("solver_cls", [HJBSemiLagrangianSolver, HJBWENOSolver])
def test_sl_weno_fail_loud_on_spatial_volatility(solver_cls):
    """Issue #1316 — SL/WENO read problem.sigma at scattered sites (no chokepoint); a
    spatial volatility_field must fail loud, not be silently ignored."""
    problem = _sl_problem()
    solver = solver_cls(problem)
    U_T = np.zeros(21)
    M = np.ones((11, 21)) / 21
    field = np.linspace(0.2, 0.8, 21)

    with pytest.raises(NotImplementedError, match="volatility_field"):
        solver.solve_hjb_system(M_density=M, U_terminal=U_T, volatility_field=field)


@pytest.mark.parametrize("solver_cls", [HJBSemiLagrangianSolver, HJBWENOSolver])
def test_sl_weno_fail_loud_on_mismatched_scalar(solver_cls):
    """Issue #1316 — a scalar volatility_field that differs from problem.sigma also fails
    loud (HJB would silently solve a different diffusion than FP)."""
    problem = _sl_problem()
    solver = solver_cls(problem)
    U_T = np.zeros(21)
    M = np.ones((11, 21)) / 21

    with pytest.raises(NotImplementedError, match="volatility_field"):
        solver.solve_hjb_system(M_density=M, U_terminal=U_T, volatility_field=0.9)


@pytest.mark.parametrize("solver_cls", [HJBSemiLagrangianSolver, HJBWENOSolver])
def test_sl_weno_accept_scalar_equal_to_sigma(solver_cls):
    """Issue #1316 — the iterator's redundant forwarding of a scalar problem.sigma as
    volatility_field is a no-op and must NOT raise."""
    problem = _sl_problem()
    solver = solver_cls(problem)
    U_T = np.zeros(21)
    M = np.ones((11, 21)) / 21
    U_prev = np.zeros((11, 21))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        U = solver.solve_hjb_system(M_density=M, U_terminal=U_T, U_coupling_prev=U_prev, volatility_field=0.3)
    assert np.asarray(U).shape[0] == 11

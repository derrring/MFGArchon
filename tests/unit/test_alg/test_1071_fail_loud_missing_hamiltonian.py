"""Issue #1071 / fail-fast: HJB solvers must NOT silently substitute a hardcoded LQ
Hamiltonian when none is available — they must raise.

Pins the removal of two silent-wrong-physics fallbacks:
- the semi-Lagrangian ``_default_hamiltonian`` (``H = 0.5*|p|^2 + C*m``), and
- the WENO ``0.5*grad**2 + m_val*grad`` fallback.

These were dead for any normally-constructed ``MFGProblem`` (construction requires a
Hamiltonian), so the change is byte-identical for real usage; the raise guards
duck-typed / externally-nulled Hamiltonian misuse and forbids silent re-introduction.
"""

from __future__ import annotations

import numpy as np

import pytest

from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver, HJBWenoSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _problem() -> MFGProblem:
    components = MFGComponents(
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: m,
            coupling_dm=lambda m: 1.0,
        ),
    )
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    return MFGProblem(geometry=grid, T=0.5, Nt=10, sigma=0.3, components=components)


def _raise_attribute_error(*_args, **_kwargs):
    raise AttributeError("no Hamiltonian (test)")


def test_semi_lagrangian_no_hamiltonian_fails_loud(monkeypatch):
    """SL: with no hamiltonian_class and no legacy H, _evaluate_hamiltonian raises rather
    than silently returning the LQ default (the removed _default_hamiltonian)."""
    problem = _problem()
    solver = HJBSemiLagrangianSolver(problem)

    # Force the no-Hamiltonian state the silent fallback used to swallow.
    monkeypatch.setattr(problem.components, "_hamiltonian_class", None, raising=False)
    monkeypatch.setattr(problem, "H", _raise_attribute_error, raising=False)
    monkeypatch.setattr(problem, "hamiltonian", _raise_attribute_error, raising=False)

    with pytest.raises(ValueError, match="silently substitute|fail-fast"):
        solver._evaluate_hamiltonian(x=0.5, p=0.3, m=1.0, time_idx=0)


def test_semi_lagrangian_default_hamiltonian_method_removed():
    """The silent LQ fallback method must stay gone (no silent re-introduction)."""
    assert not hasattr(HJBSemiLagrangianSolver, "_default_hamiltonian")


def test_semi_lagrangian_solve_no_hamiltonian_fails_loud(monkeypatch):
    """The REAL solve path must fail loud, not just _evaluate_hamiltonian in isolation.

    The batch path used to zero H silently and the per-point loops' broad except would
    swallow the per-point raise; the solve-entry guard catches the missing Hamiltonian
    before either can produce a silent pure-transport solution.
    """
    problem = _problem()
    solver = HJBSemiLagrangianSolver(problem)
    monkeypatch.setattr(problem.components, "_hamiltonian_class", None, raising=False)

    nx = problem.geometry.get_grid_shape()[0]
    nt_points = problem.Nt + 1
    x = problem.geometry.get_spatial_grid().ravel()
    m_density = np.ones((nt_points, nx)) / nx
    u_terminal = 0.5 * (x - 0.5) ** 2
    u_prev = np.tile(u_terminal, (nt_points, 1))

    with pytest.raises(ValueError, match="silently substitute|fail-fast"):
        solver.solve_hjb_system(M_density=m_density, U_terminal=u_terminal, U_coupling_prev=u_prev)


def test_weno_no_hamiltonian_fails_loud(monkeypatch):
    """WENO: with problem.H unavailable, _evaluate_hamiltonian raises rather than silently
    returning the hardcoded 0.5*grad**2 + m_val*grad."""
    problem = _problem()
    solver = HJBWenoSolver(problem=problem)

    monkeypatch.setattr(problem, "H", _raise_attribute_error, raising=False)

    with pytest.raises(ValueError, match="silently substitute|fail-fast"):
        solver._evaluate_hamiltonian(x_idx=0, m_val=1.0, grad=0.3)


def test_normal_problem_still_evaluates(monkeypatch):
    """Sanity: a properly-specified Hamiltonian still evaluates (the fail-loud does not
    fire on the happy path)."""
    problem = _problem()
    solver = HJBSemiLagrangianSolver(problem)
    val = solver._evaluate_hamiltonian(x=0.5, p=0.3, m=1.0, time_idx=0)
    assert np.isfinite(val)

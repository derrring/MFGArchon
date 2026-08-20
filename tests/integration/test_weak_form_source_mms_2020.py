"""MMS for the `source_term` channel of the weak-form family (#2020).

The channel was added because six solvers silently discarded `source_term`, so an MMS could not
reach them. "The answer moves when a source is passed" is a liveness check, not a correctness one;
what a source channel owes is an ORDER against an exact solution.

TWO STUDIES, DELIBERATELY SEPARATED
-----------------------------------
Measured together the smaller order caps the larger and neither is visible.

- SPACE: a STEADY manufactured solution. ``d_t u* == 0``, so backward Euler is exact on the time
  term and the residual is purely spatial. P1 elements should give O(h^2).
- TIME: a time-dependent solution on a mesh fine enough that the spatial error sits below the
  temporal one. Backward Euler should give O(dt).

THE FIXTURE TRAP THIS FILE ENCODES
----------------------------------
The first version of the time study used ``a(t) = 1 + (T - t)``, LINEAR in t. Backward Euler's local
truncation error is ``(dt/2) * u_tt``, which is identically zero for a linear-in-t solution, so the
study measured the spatial floor and reported order 0.02 -- an apparently catastrophic result from a
correct implementation. ``test_the_time_fixture_is_not_degenerate`` pins that the fixture has
``u_tt != 0``, because a fixture that cannot express the error is indistinguishable from a scheme
that has none.

u*(t,x) = a(t) cos(pi x) on [0,1]: ``d_x u* = -a pi sin(pi x)`` vanishes at both walls, so the pair
is exactly no-flux compatible and the wall contributes no error of its own.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import Mesh1D
from mfgarchon.geometry.boundary import no_flux_bc

_T, _SIGMA, _LAM = 0.5, 0.4, 1.0
_D = 0.5 * _SIGMA**2
_PI = np.pi


def _a(t, steady):
    return 1.0 if steady else float(np.exp(-2.0 * t))


def _ap(t, steady):
    return 0.0 if steady else float(-2.0 * np.exp(-2.0 * t))


def _u_star(t, x, steady):
    return _a(t, steady) * np.cos(_PI * x)


def _s_hjb(t, x, steady):
    """S of -u_t + |u_x|^2/(2 lam) - D u_xx = S."""
    ux = -_a(t, steady) * _PI * np.sin(_PI * x)
    uxx = -_a(t, steady) * _PI**2 * np.cos(_PI * x)
    return -_ap(t, steady) * np.cos(_PI * x) + ux**2 / (2.0 * _LAM) - _D * uxx


def _b(t, steady):
    return 0.3 if steady else float(0.3 * np.exp(-2.0 * t))


def _bp(t, steady):
    return 0.0 if steady else float(-0.6 * np.exp(-2.0 * t))


def _m_star(t, x, steady):
    return 1.0 + _b(t, steady) * np.cos(2 * _PI * x)


def _s_fp(t, x, steady):
    """S of m_t - D m_xx = S, with no potential so the drift is absent."""
    return _bp(t, steady) * np.cos(2 * _PI * x) + _D * _b(t, steady) * (2 * _PI) ** 2 * np.cos(2 * _PI * x)


def _problem(ne, nt):
    mesh = Mesh1D(bounds=(0.0, 1.0), num_elements=ne)
    mesh.generate_mesh()
    mesh.boundary_conditions = no_flux_bc(dimension=1)
    return MFGProblem(
        geometry=mesh,
        T=_T,
        Nt=nt,
        sigma=_SIGMA,
        coupling_coefficient=0.0,
        components=MFGComponents(
            m_initial=lambda x: np.ones_like(np.asarray(x, dtype=float)),
            u_terminal=lambda x: np.asarray(x, dtype=float) * 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=_LAM),
                coupling=lambda m: np.asarray(m) * 0.0,
                coupling_dm=lambda m: np.asarray(m) * 0.0,
            ),
        ),
    )


def _hjb_error(ne, nt, steady, use_newton):
    pytest.importorskip("skfem", reason="scikit-fem required for the weak-form FEM solvers")
    from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver

    problem = _problem(ne, nt)
    solver = HJBFEMSolver(problem, order=1)
    x = solver._disc.dof_coordinates[:, 0]
    n = len(x)
    # The Picard branch linearises H at the previous iterate; handing it the EXACT one makes the
    # only remaining error the discretisation, which is what an order study must isolate.
    u_prev = np.array([_u_star(k * problem.dt, x, steady) for k in range(nt + 1)])
    u = np.asarray(
        solver.solve_hjb_system(
            M_density=np.ones((nt + 1, n)),
            U_terminal=_u_star(_T, x, steady),
            U_coupling_prev=u_prev,
            source_term=lambda t, pts: _s_hjb(t, np.asarray(pts)[:, 0], steady),
            use_newton=use_newton,
        )
    )
    return float(np.sqrt(((u[0] - _u_star(0.0, x, steady)) ** 2).sum() / n))


def _fp_error(ne, nt, steady):
    pytest.importorskip("skfem", reason="scikit-fem required for the weak-form FEM solvers")
    from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver

    problem = _problem(ne, nt)
    solver = FPFEMSolver(problem, order=1)
    x = solver._disc.dof_coordinates[:, 0]
    n = len(x)
    m = np.asarray(
        solver.solve_fp_system(
            _m_star(0.0, x, steady),
            potential_field=None,
            volatility_field=_SIGMA,
            source_term=lambda t, pts: _s_fp(t, np.asarray(pts)[:, 0], steady),
        )
    )
    return float(np.sqrt(((m[-1] - _m_star(_T, x, steady)) ** 2).sum() / n))


def _order(errors, ratios):
    return [float(np.log(errors[i] / errors[i + 1]) / np.log(ratios[i])) for i in range(len(errors) - 1)]


def test_the_time_fixture_is_not_degenerate():
    """A linear-in-t manufactured solution has u_tt == 0, killing backward Euler's whole truncation
    error. The first draft of the time study did exactly that and reported order 0.02 from a correct
    implementation. Pin that the fixture can express what the study claims to measure."""
    h = 1e-4
    for f, label in ((lambda t: _a(t, False), "a"), (lambda t: _b(t, False), "b")):
        second = (f(0.2 + h) - 2 * f(0.2) + f(0.2 - h)) / h**2
        assert abs(second) > 1e-2, (
            f"{label}''(t) = {second:.3e} is ~0, so backward Euler has no truncation error on this "
            f"fixture and the temporal study below would measure the spatial floor instead."
        )
    # And the steady fixture must genuinely be steady, or the space study is capped by O(dt).
    assert _a(0.0, True) == _a(_T, True), "the steady u* fixture is not steady"
    assert _b(0.0, True) == _b(_T, True), "the steady m* fixture is not steady"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("use_newton", [False, True], ids=["picard", "newton"])
def test_the_hjb_source_is_second_order_in_space(use_newton):
    """P1 elements on a steady manufactured solution. Measured 1.883 / 1.945 / 1.974 (picard) and
    1.818 / 1.805 / 1.895 (newton) over ne = 10..80 at Nt = 400."""
    levels = (10, 20, 40)
    errs = [_hjb_error(ne, 200, True, use_newton) for ne in levels]
    orders = _order(errs, [levels[i + 1] / levels[i] for i in range(len(levels) - 1)])
    assert all(o > 1.6 for o in orders), f"HJB spatial order {orders} (errors {errs}), expected ~2"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("use_newton", [False, True], ids=["picard", "newton"])
def test_the_hjb_source_is_first_order_in_time(use_newton):
    """Backward Euler. Measured 0.999 / 1.000 / 1.000 (picard) and 1.039 / 1.018 / 1.009 (newton)
    over Nt = 5..40 at ne = 640."""
    levels = (5, 10, 20)
    errs = [_hjb_error(320, nt, False, use_newton) for nt in levels]
    orders = _order(errs, [levels[i + 1] / levels[i] for i in range(len(levels) - 1)])
    assert all(0.8 < o < 1.3 for o in orders), f"HJB temporal order {orders} (errors {errs}), expected ~1"


@pytest.mark.integration
@pytest.mark.slow
def test_the_fp_source_is_second_order_in_space():
    """Measured 2.021 / 2.014 / 2.008 over ne = 10..80 at Nt = 400."""
    levels = (10, 20, 40)
    errs = [_fp_error(ne, 200, True) for ne in levels]
    orders = _order(errs, [levels[i + 1] / levels[i] for i in range(len(levels) - 1)])
    assert all(o > 1.6 for o in orders), f"FP spatial order {orders} (errors {errs}), expected ~2"


@pytest.mark.integration
@pytest.mark.slow
def test_the_fp_source_is_first_order_in_time():
    """Measured 0.948 / 0.973 / 0.987 over Nt = 5..40 at ne = 640."""
    levels = (5, 10, 20)
    errs = [_fp_error(320, nt, False) for nt in levels]
    orders = _order(errs, [levels[i + 1] / levels[i] for i in range(len(levels) - 1)])
    assert all(0.8 < o < 1.3 for o in orders), f"FP temporal order {orders} (errors {errs}), expected ~1"

"""Every semi-Lagrangian value path reads the exact solution, for any control cost that reaches it (#2866).

With terminal ``c.x`` and no state term, ``p = c`` everywhere, so ``-u_t + H_control(p) = V0 + f0`` has
``u(0, x) = c.x - T*(H_control(c) - V0 - f0)``. The value update ``H(p) - 2*H(0)`` holds only for a control cost
2-homogeneous in ``p``, up to a constant; for any other it is off by ``T*(2*(H_control(p) - H_control(0)) - p*dH/dp)`` at every refinement.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import numpy as np
from scipy.integrate import quad

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver, HJBSemiLagrangianSolver
from mfgarchon.alg.numerical.hjb_solvers.h_eval import eval_H_batch
from mfgarchon.core.hamiltonian import (
    BoundedControlCost,
    CongestionHamiltonian,
    ControlCostBase,
    L1ControlCost,
    QuadraticControlCost,
    SeparableHamiltonian,
)
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.pde_coefficients import assert_quadratic_drift

_T = 0.2
_V0, _F0 = 0.7, 0.3


class _Cubic(ControlCostBase):
    """``H_control = |p|^3/3``, smooth and not 2-homogeneous."""

    def evaluate(self, p):
        return np.linalg.norm(p, axis=-1) ** 3 / 3.0

    def dp(self, p):
        return p * np.linalg.norm(p, axis=-1, keepdims=True)

    def optimal_control(self, p):
        return -self.dp(p)

    def lagrangian(self, alpha):
        return 2.0 * np.linalg.norm(alpha, axis=-1) ** 1.5 / 3.0


_COSTS = {
    "quadratic": (QuadraticControlCost(lambda_=2.0), 1.5),
    "cubic": (_Cubic(), 1.0),
    "L1, regularized": (L1ControlCost(lambda_=0.5).regularize(0.1), 1.5),
    "Bounded, regularized": (BoundedControlCost(lambda_=1.0, max_control=0.5).regularize(0.1), 1.5),
}
# The call site of the update each path holds, by function, so a row cannot quietly stop reaching its path.
_PATHS = {
    "adi": ({}, "_solve_timestep_semi_lagrangian"),
    "rk4, per point": ({"characteristic_solver": "rk4"}, "_advect_pointwise"),
    "stochastic": ({"diffusion_method": "stochastic"}, "_stochastic_sl_step"),
}
_CANONICAL_CS = {"diffusion_method": "canonical_cs", "interpolation_method": "linear"}


def _read(hamiltonian, slope, n, nt, **solver_options):
    """``U[0]`` over the interior, away from what the walls reach, its ``c.x``, and the callers of the value update."""
    slope = np.atleast_1d(np.asarray(slope, dtype=float))
    d = slope.size
    problem = MFGProblem(
        model=Model(hamiltonian=hamiltonian, volatility=0.05),
        domain=TensorProductGrid(
            bounds=[(0.0, 1.0)] * d, Nx_points=[n] * d, boundary_conditions=no_flux_bc(dimension=d)
        ),
        conditions=Conditions(
            m_initial=lambda x: 1.0, u_terminal=lambda x: np.sum(slope * np.asarray(x, dtype=float), axis=-1), T=_T
        ),
        Nt=nt,
    )
    mesh = np.stack(np.meshgrid(*[np.linspace(0.0, 1.0, n)] * d, indexing="ij"), axis=-1)
    terminal = np.sum(slope * mesh, axis=-1)
    solver = HJBSemiLagrangianSolver(problem, **solver_options)
    callers = set()
    update = solver._sl_value_update

    def recording(*args, **kwargs):
        callers.add(sys._getframe(1).f_code.co_name)
        return update(*args, **kwargs)

    solver._sl_value_update = recording
    U = np.asarray(
        solver.solve_hjb_system(np.ones((nt + 1, *[n] * d)), terminal, np.tile(terminal, (nt + 1, *[1] * d)))
    )
    interior = np.all((mesh > 0.55) & (mesh < 0.75), axis=-1)
    return U[0][interior], terminal[interior], callers


def _separable(cost, *, state=False):
    if not state:
        return SeparableHamiltonian(control_cost=cost, coupling=lambda m: 0.0 * m)
    return SeparableHamiltonian(
        control_cost=cost,
        coupling=lambda m: _F0 + 0.0 * m,
        potential=lambda t, x: _V0 + 0.0 * np.asarray(x, dtype=float)[..., 0],
    )


def _exact(cost, slope, terminal, *, state=False):
    H_control = float(np.ravel(cost.evaluate(np.atleast_1d(np.asarray(slope, dtype=float))[None, :]))[0])
    return terminal - _T * (H_control - (_V0 + _F0 if state else 0.0))


@pytest.mark.parametrize("state", [False, True], ids=["no state term", "V0 and f0"])
@pytest.mark.parametrize("path", list(_PATHS))
@pytest.mark.parametrize("cost", list(_COSTS))
def test_each_path_reads_the_exact_solution_in_1d(cost, path, state):
    control_cost, slope = _COSTS[cost]
    options, caller = _PATHS[path]
    U, terminal, callers = _read(_separable(control_cost, state=state), slope, 41, 20, **options)
    assert caller in callers, callers
    np.testing.assert_allclose(U, _exact(control_cost, slope, terminal, state=state), rtol=0, atol=2e-5)


@pytest.mark.parametrize("path", list(_PATHS))
@pytest.mark.parametrize("cost", list(_COSTS))
def test_each_path_reads_the_exact_solution_in_2d(cost, path):
    """Two components of ``p``, so ``p*dH/dp`` is a sum: a 1-D row cannot see one read along the wrong axis."""
    control_cost, _ = _COSTS[cost]
    options, _caller = _PATHS[path]
    slope = (1.0, 0.5)
    U, terminal, _callers = _read(_separable(control_cost), slope, 15, 6, **options)
    np.testing.assert_allclose(U, _exact(control_cost, slope, terminal), rtol=0, atol=2e-5)


def test_the_substepped_path_reads_the_exact_solution():
    """Substepped when one step crosses more than ``cfl_target`` cells: the update's fifth call site."""
    control_cost = _Cubic()
    slope = (1.0, 0.5)
    U, terminal, callers = _read(_separable(control_cost), slope, 21, 2, cfl_target=0.5)
    assert "_solve_timestep_semi_lagrangian_with_dt" in callers, callers
    np.testing.assert_allclose(U, _exact(control_cost, slope, terminal), rtol=0, atol=2e-5)


def _congestion():
    return CongestionHamiltonian(QuadraticControlCost(lambda_=1.0), congestion_factor=lambda m: 1.0 + 3.0 * m)


@pytest.mark.parametrize("path", ["adi", "rk4, per point", "stochastic"])
def test_a_non_separable_hamiltonian_reads_right_where_the_update_reads_its_own_derivative(path):
    """``dH/dp`` is ``p/(lambda*c(m))`` here, not the control cost's ``p/lambda``: uniform m = 1, c = 4, exact ``c^2/8``."""
    U, terminal, _ = _read(_congestion(), 1.5, 41, 20, **_PATHS[path][0])
    np.testing.assert_allclose(U, terminal - _T * 1.5 * 1.5 / 8.0, rtol=0, atol=2e-5)


def _problem(hamiltonian, slope=1.5, n=21, nt=5):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    return MFGProblem(
        model=Model(hamiltonian=hamiltonian, volatility=0.05),
        domain=grid,
        conditions=Conditions(
            m_initial=lambda x: 1.0, u_terminal=lambda x: slope * np.asarray(x, dtype=float)[..., 0], T=_T
        ),
        Nt=nt,
    )


_NOT_QUADRATIC = {
    "cubic": _separable(_Cubic()),
    "L1, regularized": _separable(L1ControlCost(lambda_=0.5).regularize(0.1)),
    "Bounded, regularized": _separable(BoundedControlCost(lambda_=1.0, max_control=0.5).regularize(0.1)),
    "L1": _separable(L1ControlCost(lambda_=0.5)),
    "Bounded": _separable(BoundedControlCost(lambda_=1.0, max_control=0.5)),
    "congestion": _congestion(),
}


@pytest.mark.parametrize("hamiltonian", list(_NOT_QUADRATIC))
def test_canonical_cs_refuses_what_it_cannot_read_when_it_is_built(hamiltonian):
    """Its hand-coded quadratic read plain L1 at -0.248 and congestion at -0.169 from the exact solution, with no error.

    The names are the point of the match: congestion holds a quadratic control cost, and that alone is not what is refused.
    """
    given = _NOT_QUADRATIC[hamiltonian]
    named = f"{type(given).__name__} with {type(given.control_cost).__name__}"
    with pytest.raises(
        NotImplementedError,
        match=rf"needs a SeparableHamiltonian with a QuadraticControlCost.*Hamiltonian is {named} .*#2866.*"
        r"Use HJBFDMSolver, or, where the Hamiltonian's is_smooth\(\) is True, diffusion_method='adi' or 'stochastic'",
    ):
        HJBSemiLagrangianSolver(_problem(given), **_CANONICAL_CS)


def test_the_fp_drift_guard_sharing_that_test_still_passes_a_non_separable_hamiltonian():
    """``assert_quadratic_drift`` now reads the test this refusal reads, and its no-op there is another caller's to refuse."""
    assert assert_quadratic_drift(SimpleNamespace(hamiltonian_class=_congestion()), context="row") is None


@pytest.mark.parametrize("cost", ["cubic", "L1, regularized", "Bounded, regularized"])
@pytest.mark.parametrize("path", ["adi", "stochastic"])
def test_each_cure_the_refusal_names_for_a_smooth_cost_reads_it(cost, path):
    control_cost, slope = _COSTS[cost]
    assert _separable(control_cost).is_smooth()
    U, terminal, _ = _read(_separable(control_cost), slope, 41, 20, **_PATHS[path][0])
    np.testing.assert_allclose(U, _exact(control_cost, slope, terminal), rtol=0, atol=2e-5)


@pytest.mark.parametrize(
    ("control_cost", "slope"),
    [(L1ControlCost(lambda_=0.5), 0.75), (BoundedControlCost(lambda_=1.0, max_control=5.0), 0.5)],
    ids=["L1", "Bounded"],
)
def test_the_cure_the_refusal_names_for_a_cost_that_is_not_smooth_reads_it(control_cost, slope):
    """``stochastic`` raises on these, and ``adi`` runs the DPP search: this Bounded optimum is between its 11 samples."""
    hamiltonian = _separable(control_cost)
    assert not hamiltonian.is_smooth()
    n, nt = 81, 40
    x = np.linspace(0.0, 1.0, n)
    terminal = slope * x
    solver = HJBFDMSolver(_problem(hamiltonian, slope, n, nt))
    U = np.asarray(solver.solve_hjb_system(np.ones((nt + 1, n)), terminal, np.tile(terminal, (nt + 1, 1))))
    interior = (x > 0.55) & (x < 0.75)
    np.testing.assert_allclose(U[0][interior], _exact(control_cost, slope, terminal[interior]), rtol=0, atol=1e-4)
    with pytest.raises(NotImplementedError):
        _read(hamiltonian, slope, 21, 5, **_PATHS["stochastic"][0])


@pytest.mark.parametrize(("slope", "n", "nt"), [(1.5, 41, 20), ((1.0, 0.5), 15, 6)], ids=["1-D", "2-D"])
def test_canonical_cs_keeps_a_quadratic_cost(slope, n, nt):
    control_cost = QuadraticControlCost(lambda_=2.0)
    U, terminal, _ = _read(_separable(control_cost), slope, n, nt, **_CANONICAL_CS)
    np.testing.assert_allclose(U, _exact(control_cost, slope, terminal), rtol=0, atol=2e-5)


@pytest.mark.parametrize("path", list(_PATHS))
def test_the_update_refines_with_a_potential_and_a_coupling_that_vary(path):
    """``V = (1 + 2t)x + 0.3t`` and ``f = 0.4*m`` on ``m = 0.5 + x``, terminal ``0.8x``. Then ``u = c(t)x + e(t)``, with
    ``c' = -(1 + 2t + 0.4)`` and ``e' = c^3/3 - 0.3t - 0.2`` for the cubic cost. The old update's plateau cannot halve."""
    control_cost = _Cubic()
    hamiltonian = SeparableHamiltonian(
        control_cost=control_cost,
        potential=lambda t, x: (1.0 + 2.0 * t) * np.asarray(x, dtype=float)[..., 0] + 0.3 * t,
        coupling=lambda m: 0.4 * m,
    )

    def slope_at(t):
        return 0.8 + 1.4 * (_T - t) + (_T**2 - t**2)

    drop = quad(lambda s: slope_at(s) ** 3 / 3.0 - 0.3 * s - 0.2, 0.0, _T, epsabs=1e-11)[0]
    errors = []
    for n, nt in ((81, 40), (161, 80)):
        x = np.linspace(0.0, 1.0, n)
        problem = MFGProblem(
            model=Model(hamiltonian=hamiltonian, volatility=0.05),
            domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1)),
            conditions=Conditions(
                m_initial=lambda x: 1.0, u_terminal=lambda x: 0.8 * np.asarray(x, dtype=float)[..., 0], T=_T
            ),
            Nt=nt,
        )
        solver = HJBSemiLagrangianSolver(problem, **_PATHS[path][0])
        U = np.asarray(solver.solve_hjb_system(np.tile(0.5 + x, (nt + 1, 1)), 0.8 * x, np.tile(0.8 * x, (nt + 1, 1))))
        box = (x > 0.55) & (x < 0.75)
        errors.append(np.abs(U[0][box] - (slope_at(0.0) * x[box] - drop)).max())
    assert errors[1] < 0.65 * errors[0], errors


def _curved(x):
    return 0.25 * np.sin(2.0 * np.pi * x)


@pytest.mark.parametrize("cost", ["cubic", "L1, regularized"])
@pytest.mark.parametrize("path", ["adi", "stochastic"])
def test_the_update_agrees_with_fdm_where_p_varies_in_x(cost, path):
    """The other rows have ``u`` linear in ``x``, where the foot and the update cancel ``dH/dp``, whatever it is:
    ``c*(x - dt*v) + dt*(c*v - H) = c*x - dt*H``. FDM is a second solver and not an exact solution, so the bound sits between
    this update's distance from it, 0.009 at most, and the old update's, 0.06 on the regularized L1 and 0.11 on the cubic.
    """
    n, nt = 161, 40
    problem = MFGProblem(
        model=Model(hamiltonian=_separable(_COSTS[cost][0]), volatility=0.05),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(
            m_initial=lambda x: 1.0, u_terminal=lambda x: _curved(np.asarray(x, dtype=float)[..., 0]), T=_T
        ),
        Nt=nt,
    )
    x = np.linspace(0.0, 1.0, n)
    terminal = _curved(x)
    semi_lagrangian, fdm = (
        np.asarray(solver.solve_hjb_system(np.ones((nt + 1, n)), terminal, np.tile(terminal, (nt + 1, 1))))[0]
        for solver in (HJBSemiLagrangianSolver(problem, **_PATHS[path][0]), HJBFDMSolver(problem))
    )
    interior = (x > 0.2) & (x < 0.8)
    np.testing.assert_allclose(semi_lagrangian[interior], fdm[interior], rtol=0, atol=0.015)


@pytest.mark.parametrize(
    "hamiltonian",
    [
        _separable(QuadraticControlCost(lambda_=2.0), state=True),
        CongestionHamiltonian(
            QuadraticControlCost(lambda_=1.5), congestion_factor=lambda m: 1.0 + 3.0 * m, potential=lambda t, x: 0.7 * t
        ),
    ],
    ids=["separable", "congestion"],
)
def test_a_2_homogeneous_kinetic_term_keeps_the_update_it_had(hamiltonian):
    """Where the two forms agree, they agree to round-off, at varied x, p and m."""
    rng = np.random.default_rng(2866)
    x, p, m = rng.uniform(0, 1, (50, 1)), rng.normal(0, 2, (50, 1)), rng.uniform(0.1, 3, 50)
    problem = MFGProblem(
        model=Model(hamiltonian=hamiltonian, volatility=0.05),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=_T),
        Nt=2,
    )
    solver = HJBSemiLagrangianSolver(problem)
    H_class = problem.hamiltonian_class
    old = 1.0 + 0.01 * (eval_H_batch(H_class, x, m, p, 0.3) - 2.0 * eval_H_batch(H_class, x, m, np.zeros_like(p), 0.3))
    np.testing.assert_allclose(solver._sl_value_update(1.0, x, m, p, 0.3, 0.01), old.ravel(), rtol=0, atol=1e-13)

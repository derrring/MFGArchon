"""`HJBFDMSolver(constraint=)` projects after each time step, in 1-D as in nD, and refuses infeasible terminal data (#2036).

The constructor documents the constraint as "applied after each timestep solve via projection". The 1-D path
instead clipped the finished array, so the backward sweep never saw the obstacle: the result was exactly
max(U_free, psi), a feasible array that is not a solution of the obstacle problem. The nD path projected each
step, but returned the caller's terminal slice unprojected and infeasible.

Oracle for the first: the contract itself. Each returned slice must be the projection of one unconstrained step
taken from the next returned slice, U^n = P_K(N(U^{n+1})), with N computed independently as a one-step solve of
the same step length. A post-hoc clip breaks this wherever the obstacle binds at t_{n+1}, because its U^n came
from the unprojected U^{n+1}.

For the second, the user ruling is to refuse: the terminal condition is the caller's data, returned as given.
"""

from __future__ import annotations

import logging
import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import ObstacleConstraint, no_flux_bc

NX, NT, T = 41, 20, 1.0
X = np.linspace(0.0, 1.0, NX)
M = np.ones((NT + 1, NX))


def _problem(dim=1, nt=NT, horizon=T):
    # A running cost that is positive near the walls, so the value rises backward there and crosses a ceiling
    # placed above the zero terminal datum.
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(lambda_=1.0),
        potential=lambda t, x: 0.1 * np.cos(2 * np.pi * np.asarray(x, dtype=float)[..., 0]),
    )
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * dim, Nx_points=[NX] * dim, boundary_conditions=no_flux_bc(dimension=dim)
    )
    return MFGProblem(
        model=Model(hamiltonian=hamiltonian, volatility=0.1),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=horizon),
        Nt=nt,
    )


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.WARNING)
        try:
            return fn()
        finally:
            logging.disable(logging.NOTSET)


def test_each_returned_step_is_the_projection_of_one_step_from_the_next():
    ceiling = ObstacleConstraint(0.02 + 0.0 * X, constraint_type="upper")
    u_terminal = np.zeros(NX)
    U = _quiet(
        lambda: HJBFDMSolver(_problem(), constraint=ceiling).solve_hjb_system(M, u_terminal, np.zeros((NT + 1, NX)))
    )
    assert np.sum(np.abs(U[0] - 0.02) < 1e-12) >= 5, "the ceiling never binds, so the test cannot see a clip"

    one_step = _problem(nt=1, horizon=T / NT)
    worst = 0.0
    for n in range(NT):
        unconstrained = _quiet(
            lambda n=n: HJBFDMSolver(one_step).solve_hjb_system(np.ones((2, NX)), U[n + 1], np.zeros((2, NX)))
        )[0]
        worst = max(worst, float(np.max(np.abs(U[n] - ceiling.project(unconstrained)))))
    assert worst < 1e-10, (
        f"a returned step is not P_K of one step from the next returned step (max deviation {worst:.3e}): the "
        "constraint is not applied inside the backward sweep (#2036)"
    )


@pytest.mark.parametrize("dim", [1, 2], ids=["1d", "2d"])
def test_an_infeasible_terminal_condition_is_refused(dim):
    shape = (NX,) * dim
    floor = ObstacleConstraint(np.full(shape, 0.1), constraint_type="lower")
    solver = _quiet(lambda: HJBFDMSolver(_problem(dim=dim, nt=2), constraint=floor))
    with pytest.raises(ValueError, match="U_terminal lies outside the constraint set"):
        solver.solve_hjb_system(np.ones((3, *shape)), np.zeros(shape), np.zeros((3, *shape)))

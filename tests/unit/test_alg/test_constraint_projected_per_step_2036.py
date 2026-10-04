"""`HJBFDMSolver(constraint=)` projects after each time step, in 1-D as in nD, and refuses infeasible terminal data (#2036).

The constructor documents the constraint as "applied after each timestep solve via projection". The 1-D path
instead clipped the finished array, so the backward sweep never saw the obstacle: the result was exactly
max(U_free, psi), a feasible array that is not a solution of the obstacle problem. The nD path projected each
step, but returned the caller's terminal slice unprojected and infeasible.

Two pins for the first:

- the order of operations, from the contract itself. Each returned slice must be the projection of one
  unconstrained step taken from the next returned slice, U^n = P_K(N(U^{n+1})), with N a one-step solve of the
  same step length. That reruns the same step function, so it is a structural pin, not an external oracle. It
  holds because the fixture is autonomous: the one-step solve starts at t = 0 and the sweep's step n at n*dt,
  which differ once the data depend on time. A post-hoc clip breaks it wherever the obstacle binds at t_{n+1},
  because its U^n came from the unprojected U^{n+1}.
- the nD path, which no fast test covered: a 2-D solve whose data do not depend on y must reproduce the 1-D
  solve at every y. Measured on this fixture, 1.4e-8 after #2036 and 3.45e-3 before, when only the nD half
  projected per step.

For the second, the user ruling is to refuse: the terminal condition is the caller's data, returned as given.
The tolerance is round-off scale, so its edges are pinned too.
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


def _problem(dim=1, nt=NT, horizon=T, points=None):
    # A running cost that is positive near the walls, so the value rises backward there and crosses a ceiling
    # placed above the zero terminal datum.
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(lambda_=1.0),
        potential=lambda t, x: 0.1 * np.cos(2 * np.pi * np.asarray(x, dtype=float)[..., 0]),
    )
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * dim, Nx_points=points or [NX] * dim, boundary_conditions=no_flux_bc(dimension=dim)
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


def test_a_y_independent_2d_solve_reproduces_the_1d_solve():
    ny = 5
    ceiling_1d = ObstacleConstraint(np.full(NX, 0.02), constraint_type="upper")
    ceiling_2d = ObstacleConstraint(np.full((NX, ny), 0.02), constraint_type="upper")
    U1 = _quiet(
        lambda: HJBFDMSolver(_problem(), constraint=ceiling_1d).solve_hjb_system(
            M, np.zeros(NX), np.zeros((NT + 1, NX))
        )
    )
    U2 = _quiet(
        lambda: HJBFDMSolver(_problem(dim=2, points=[NX, ny]), constraint=ceiling_2d).solve_hjb_system(
            np.ones((NT + 1, NX, ny)), np.zeros((NX, ny)), np.zeros((NT + 1, NX, ny))
        )
    )
    deviation = float(np.max(np.abs(U2 - U1[:, :, None])))
    assert deviation < 1e-6, (
        f"the 2-D solve departs from the 1-D one by {deviation:.3e} on y-independent data: the two paths do not "
        "apply the constraint at the same point of the step (#2036)"
    )


@pytest.mark.parametrize("dim", [1, 2], ids=["1d", "2d"])
@pytest.mark.parametrize(
    ("below_floor", "refusal"),
    [
        (1e-13, None),
        (1e-9, "U_terminal lies outside the constraint set"),
        (np.nan, "non-finite entries"),
    ],
    ids=["round_off_accepted", "violation_refused", "nan_refused"],
)
def test_the_terminal_condition_must_lie_in_the_constraint_set(dim, below_floor, refusal):
    shape = (NX,) if dim == 1 else (9, 9)
    floor = ObstacleConstraint(np.full(shape, 0.1), constraint_type="lower")
    u_terminal = np.full(shape, 0.1)
    u_terminal.flat[0] = np.nan if np.isnan(below_floor) else 0.1 - below_floor
    solver = _quiet(lambda: HJBFDMSolver(_problem(dim=dim, nt=2, points=list(shape)), constraint=floor))
    solve = lambda: solver.solve_hjb_system(np.ones((3, *shape)), u_terminal, np.zeros((3, *shape)))  # noqa: E731
    if refusal is None:
        assert np.all(np.isfinite(_quiet(solve)))
    else:
        with pytest.raises(ValueError, match=refusal):
            solve()

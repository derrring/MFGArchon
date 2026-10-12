"""n-D HJB-FDM's cached Laplacian reads a time-dependent Neumann value at the step's time (#2543's time half).

The operator is built once, and it carried the clock it was built with, t = 0. The gradient operators are rebuilt
each step with the step's time. So the two halves of the residual read the wall datum at different times, and the
Neumann wall, held by the ghosts alone since #2537, converged to the t = 0 datum.

The fixture is #2543's: 2-D, with `neumann_bc` value g(t) = -0.5 - 4t on every face, sigma = 0.5, T = 0.4, Nt = 4,
quadratic H and no coupling. It measures the mean over x_min of the second-order one-sided outward slope of the
t = 0.3 slice, where g(0.3) = -1.70 and g(0) = -0.50:

| | 21^2 | 41^2 | 81^2 |
|---|---|---|---|
| 52219be2 (main) | -0.983 | -0.806 | -0.675 (-> g(0)) |
| this change | -1.680 | -1.695 | -1.699 (-> g(0.3)) |

#2543's other half, a BC object replaced after the operator was built, is #2475's class and is not this pin's.
"""

from __future__ import annotations

import warnings

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import neumann_bc

T, NT, SIGMA, N = 0.4, 4, 0.5, 21


def _g(t):
    return -0.5 - 4.0 * t


def test_a_time_dependent_neumann_wall_holds_the_steps_datum():
    zero = lambda m: 0.0 * np.asarray(m)  # noqa: E731
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * 2, Nx_points=[N] * 2, boundary_conditions=neumann_bc(dimension=2, value=_g)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = MFGProblem(
            model=Model(
                hamiltonian=SeparableHamiltonian(
                    control_cost=QuadraticControlCost(lambda_=1.0), coupling=zero, coupling_dm=zero
                ),
                volatility=SIGMA,
            ),
            domain=grid,
            conditions=Conditions(m_initial=lambda p: 1.0, u_terminal=lambda p: 0.0, T=T),
            Nt=NT,
        )
    U = np.asarray(
        HJBFDMSolver(problem).solve_hjb_system(
            M_density=np.ones((NT + 1, N, N)), U_terminal=np.zeros((N, N)), U_coupling_prev=np.zeros((NT + 1, N, N))
        )
    )
    h, level = 1.0 / (N - 1), 3  # t = 0.3
    outward = -(-3.0 * U[level, 0, :] + 4.0 * U[level, 1, :] - U[level, 2, :]) / (2.0 * h)
    assert abs(outward.mean() - _g(0.3)) < 0.05, outward.mean()

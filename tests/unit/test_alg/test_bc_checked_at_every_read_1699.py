"""A BC set on the geometry after construction is checked before a solver solves it (#1699).

`_validate_bc_support` ran only in constructors, while HJB-FDM, HJB-SL and FP-SL read the geometry's BC
again at every solve. A BC set on the geometry afterwards was therefore solved unchecked, a type the
constructor refuses included. Measured at e7a4700b on this file's fixture: all three solved a swapped-in
ROBIN, and HJB-FDM's answer moved by 6.1e-02 against the unswapped solve, so the BC reached the
discretisation.

The check now sits in `get_boundary_conditions`, so it runs at every read.

Each case also solves after swapping to a SUPPORTED BC that changes the answer. That shows the solver reads
the geometry's BC live, so the refusal comes from reading the swapped BC and not from refusing every BC. A
solver that reads only a construction-time snapshot would fail that check instead of passing vacuously.
"""

from __future__ import annotations

import logging
import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver, HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import dirichlet_bc, neumann_bc, no_flux_bc, periodic_bc, robin_bc

N, NT = 21, 4
X = np.linspace(0.0, 1.0, N)


def _solver(cls):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[N], boundary_conditions=no_flux_bc(dimension=1))
    model = Model(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0), potential=lambda t, x: 0.0),
        volatility=0.2,
    )
    conditions = Conditions(m_initial=lambda x: 1.0 + 0.5 * x, u_terminal=lambda x: (x - 0.3) ** 2, T=0.2)
    return grid, cls(MFGProblem(model=model, domain=grid, conditions=conditions, Nt=NT))


def _solve(solver):
    u = np.tile((X - 0.3) ** 2, (NT + 1, 1))
    if isinstance(solver, FPSLSolver):
        return solver.solve_fp_system(M_initial=1.0 + 0.5 * X, potential_field=u)
    return solver.solve_hjb_system(np.tile(1.0 + 0.5 * X, (NT + 1, 1)), u[-1], u)


@pytest.mark.parametrize(
    ("cls", "unsupported", "refusal"),
    [
        (HJBFDMSolver, lambda: robin_bc(alpha=1.0, beta=1.0, dimension=1), "ROBIN"),
        (HJBSemiLagrangianSolver, lambda: dirichlet_bc(dimension=1), "DIRICHLET"),
        (FPSLSolver, lambda: dirichlet_bc(dimension=1), "DIRICHLET"),
        # The validator's other half: a Neumann value these solvers would drop (#1686).
        (HJBSemiLagrangianSolver, lambda: neumann_bc(value=-0.3, dimension=1), "honours only the homogeneous"),
    ],
    ids=["hjb_fdm-robin", "hjb_sl-dirichlet", "fp_sl-dirichlet", "hjb_sl-neumann_value"],
)
def test_a_bc_swapped_onto_the_geometry_is_checked_before_it_is_solved(cls, unsupported, refusal):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.WARNING)
        try:
            grid, solver = _solver(cls)
            reference = _solve(solver)

            grid.set_boundary_conditions(periodic_bc(dimension=1))
            assert np.abs(_solve(solver) - reference).max() > 1e-3, (
                f"{cls.__name__} did not move under a supported swap: it no longer reads the geometry's BC "
                "at solve time, so a refusal here would be vacuous"
            )

            grid.set_boundary_conditions(unsupported())
            with pytest.raises(NotImplementedError, match=refusal):
                _solve(solver)
        finally:
            logging.disable(logging.NOTSET)

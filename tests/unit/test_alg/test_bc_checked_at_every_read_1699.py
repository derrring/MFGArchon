"""A BC set on the geometry after construction is checked before a solver solves it (#1699).

`_validate_bc_support` ran only in constructors, while HJB-FDM, HJB-SL and FP-SL read the geometry's BC
again at every solve. A BC set on the geometry afterwards was therefore solved unchecked, a type the
constructor refuses included. Measured on this file's 1-D fixture, with the check removed: a swapped-in
ROBIN moved HJB-FDM's answer by 4.27e-01 and a Dirichlet moved HJB-SL's by 4.00e-01. A Neumann value
moved FP-SL's by 0: FP-SL drops it.

The check now sits in `get_boundary_conditions`, so it runs at every read. HJB-FDM also reads the accessor
once at its solve entry. Otherwise, on a first solve, the n-D path's first read is inside the Newton
residual, whose handler retypes the refusal as a ConvergenceError (#2477).

Each case is refused twice: once before the first solve, which is the order that reaches that residual
read, and once after solves have succeeded. In between, it swaps to a SUPPORTED BC that changes the answer.
This confirms the solver reads the geometry's BC live, and it rules out a refusal that fires on every swap.
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
from mfgarchon.geometry.boundary import dirichlet_bc, no_flux_bc, periodic_bc, robin_bc

NT = 4
N = {1: 21, 2: 9}


def _terminal(dim):
    if dim == 1:
        return lambda x: (np.asarray(x, dtype=float) - 0.3) ** 2
    return lambda x: (np.asarray(x, dtype=float)[..., 0] - 0.3) ** 2 + 0.5 * np.asarray(x, dtype=float)[..., 1]


def _solver(cls, dim):
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * dim, Nx_points=[N[dim]] * dim, boundary_conditions=no_flux_bc(dimension=dim)
    )
    model = Model(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0), potential=lambda t, x: 0.0),
        volatility=0.2,
    )
    conditions = Conditions(m_initial=lambda x: 1.0, u_terminal=_terminal(dim), T=0.2)
    return grid, cls(MFGProblem(model=model, domain=grid, conditions=conditions, Nt=NT))


def _solve(solver, dim):
    nodes = np.stack(np.meshgrid(*[np.linspace(0.0, 1.0, N[dim])] * dim, indexing="ij"), axis=-1)
    u_terminal = _terminal(dim)(nodes[..., 0] if dim == 1 else nodes)
    u = np.broadcast_to(u_terminal, (NT + 1, *u_terminal.shape)).copy()
    m = np.ones_like(u) if dim > 1 else 1.0 + 0.5 * np.broadcast_to(nodes[..., 0], u.shape)
    if isinstance(solver, FPSLSolver):
        return solver.solve_fp_system(M_initial=m[0], potential_field=u)
    return solver.solve_hjb_system(m, u_terminal, u)


@pytest.mark.parametrize(
    ("cls", "dim", "unsupported", "refusal"),
    [
        (HJBFDMSolver, 1, robin_bc(alpha=1.0, beta=1.0, dimension=1), "ROBIN"),
        (HJBFDMSolver, 2, robin_bc(alpha=1.0, beta=1.0, dimension=2), "ROBIN"),
        (HJBSemiLagrangianSolver, 1, dirichlet_bc(dimension=1), "DIRICHLET"),
        (FPSLSolver, 1, dirichlet_bc(dimension=1), "DIRICHLET"),
    ],
    ids=["hjb_fdm-1d-robin", "hjb_fdm-2d-robin", "hjb_sl-dirichlet", "fp_sl-dirichlet"],
)
def test_a_bc_swapped_onto_the_geometry_is_checked_before_it_is_solved(cls, dim, unsupported, refusal):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.WARNING)
        try:
            grid, solver = _solver(cls, dim)
            grid.set_boundary_conditions(unsupported)
            with pytest.raises(NotImplementedError, match=refusal):
                _solve(solver, dim)

            grid.set_boundary_conditions(no_flux_bc(dimension=dim))
            reference = _solve(solver, dim)
            grid.set_boundary_conditions(periodic_bc(dimension=dim))
            assert np.abs(_solve(solver, dim) - reference).max() > 1e-3, (
                f"{cls.__name__} did not move under a supported swap: it no longer reads the geometry's BC at "
                "solve time, so the refusal above does not show that a swapped BC is checked"
            )

            # And after solves have succeeded: a check that lapses once the solver has run is also a bypass.
            grid.set_boundary_conditions(unsupported)
            with pytest.raises(NotImplementedError, match=refusal):
                _solve(solver, dim)
        finally:
            logging.disable(logging.NOTSET)

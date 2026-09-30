"""The HJB half of the SL pair sizes its sub-steps by the speed its foot moves at (#2439).

The foot moves at dH/dp, which is p/lambda for the quadratic control cost. The CFL number used to be
max|grad u| * dt / dx, which is that speed only at lambda = 1. At lambda = 0.25 each sub-step crossed
about 3.3 cells, and on #1880's fixture the coupled solve ended in the NaN #2438 describes, with no
cap involved. The property pinned here is read off the feet themselves, not off the schedule.
"""

from __future__ import annotations

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def test_no_hjb_substep_moves_its_foot_more_than_one_cell_below_lambda_one():
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(lambda_=0.25), coupling=lambda m: -m, coupling_dm=lambda m: -1.0
    )
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(
        model=Model(hamiltonian=hamiltonian, volatility=0.2),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=1.0),
        Nt=10,
    )
    u_terminal = 3.0 * (grid.coordinates[0] - 0.4) ** 2
    solver = HJBSemiLagrangianSolver(problem)

    crossed = []
    trace = solver._trace_characteristic_backward

    def recording_trace(x, velocity, dt):
        crossed.append(abs(float(velocity)) * dt / solver.dx)
        return trace(x, velocity, dt)

    solver._trace_characteristic_backward = recording_trace
    solver.solve_hjb_system(np.ones((11, 21)), u_terminal, np.tile(u_terminal, (11, 1)))
    # Measured 0.888 with the fix and 3.300 without it. The lower bound is the control that the
    # recorder saw the sub-stepped feet at all.
    assert 0.5 < max(crossed) <= 1.0

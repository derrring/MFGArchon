"""The HJB half of the SL pair sizes its sub-steps by the speed its foot moves at (#2439).

The foot moves at dH/dp: p/lambda for the quadratic control cost, and density-dependent for a
congestion Hamiltonian. The CFL number used to be max|grad u| * dt / dx, which is that speed only for
the quadratic cost at lambda = 1. At lambda = 0.25 a sub-step crossed about 3.3 cells, and on #1880's
fixture the coupled solve ended in the NaN #2438 describes, with no cap involved.

The property is read off the feet the solver traces, not off the schedule. It is "about one cell,
not 1/lambda": the schedule plans at most cfl_target = 0.9 cells, but the gradient steepens within a
step, and in a coupled solve the feet reach ~1.15-1.22 cells (#2439's velocity time level), so the
bound is 1.5 rather than 1.
"""

from __future__ import annotations

import logging

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfgarchon.core import CongestionHamiltonian
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _largest_crossing(hamiltonian, shape, u_terminal, density):
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * len(shape), Nx_points=list(shape), boundary_conditions=no_flux_bc(dimension=len(shape))
    )
    problem = MFGProblem(
        model=Model(hamiltonian=hamiltonian, volatility=0.2),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=1.0),
        Nt=10,
    )
    axes = np.meshgrid(*grid.coordinates, indexing="ij")
    ut = u_terminal(axes)
    solver = HJBSemiLagrangianSolver(problem)
    crossed = []
    trace = solver._trace_characteristic_backward
    dx_min = float(np.min(solver.dx))

    def recording_trace(x, velocity, dt):
        crossed.append(float(np.linalg.norm(velocity)) * dt / dx_min)
        return trace(x, velocity, dt)

    solver._trace_characteristic_backward = recording_trace
    m = density(axes)
    solver.solve_hjb_system(m, ut, np.broadcast_to(ut, m.shape).copy())
    return max(crossed)


def _quadratic(lam):
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(lambda_=lam), coupling=lambda m: -m, coupling_dm=lambda m: -1.0
    )


def _congestion():
    return CongestionHamiltonian(
        control_cost=QuadraticControlCost(lambda_=1.0),
        congestion_factor=lambda m: 0.2 + np.asarray(m),
        congestion_factor_dm=lambda m: 1.0 + 0 * np.asarray(m),
        coupling=lambda m: -m,
        coupling_dm=lambda m: -1.0,
    )


def _uniform(axes):
    return np.ones((11, *axes[0].shape))


def _falling(axes):
    # Density that falls with time, so the congestion factor falls and the foot speeds up: a schedule
    # that reads an earlier step's density plans too few sub-steps for the later ones.
    t = np.linspace(0.0, 1.0, 11).reshape(-1, *([1] * len(axes)))
    return (2.1 - 2.0 * t) * (1 + 0.5 * np.cos(np.pi * axes[0]))[None]


@pytest.mark.parametrize(
    ("hamiltonian", "shape", "u_terminal", "density"),
    [
        (_quadratic(0.25), (21,), lambda a: 3.0 * (a[0] - 0.4) ** 2, _uniform),
        (_quadratic(0.25), (13, 11), lambda a: 3.0 * ((a[0] - 0.4) ** 2 + 0.5 * (a[1] - 0.3) ** 2), _uniform),
        (_congestion(), (21,), lambda a: 3.0 * (a[0] - 0.4) ** 2, _falling),
    ],
    ids=["1d-lambda-quarter", "2d-lambda-quarter", "1d-congestion"],
)
def test_an_hjb_substep_crosses_about_one_cell_whatever_the_foot_speed(hamiltonian, shape, u_terminal, density):
    # Measured, fixed / not: 1-D 0.888 / 3.300 (|grad u| measure), 2-D 0.952 / 3.441 (the nD measure),
    # congestion 0.893 / 1.773 (|grad u|) and 4.389 (the schedule reading step 0's density and time).
    # The lower bound is the control that the recorder saw the sub-stepped feet at all.
    assert 0.5 < _largest_crossing(hamiltonian, shape, u_terminal, density) <= 1.5


def test_the_cfl_warning_reads_the_same_foot_speed_as_the_schedule(mfg_caplog):
    """At lambda = 4 the foot moves at a quarter of |grad u|. The warning used |grad u| after the
    schedule stopped doing so, and fired on steps the schedule had rightly left whole (#2449's review)."""

    def solve(lam, **config):
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(
            model=Model(hamiltonian=_quadratic(lam), volatility=0.2),
            domain=grid,
            conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=1.0),
            Nt=10,
        )
        ut = 3.0 * (grid.coordinates[0] - 0.4) ** 2
        HJBSemiLagrangianSolver(problem, **config).solve_hjb_system(np.ones((11, 21)), ut, np.tile(ut, (11, 1)))

    logger = "mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian"
    with mfg_caplog.at_level(logging.WARNING, logger=logger):
        solve(4.0)
    assert not [m for m in mfg_caplog.messages if "CFL condition violated" in m]
    # Control, same logger: with sub-stepping off at lambda = 1 the warning does fire.
    with mfg_caplog.at_level(logging.WARNING, logger=logger):
        solve(1.0, enable_adaptive_substepping=False)
    assert [m for m in mfg_caplog.messages if "CFL condition violated" in m]

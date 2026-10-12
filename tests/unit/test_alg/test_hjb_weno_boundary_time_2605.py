"""HJB-WENO reads a time-dependent boundary value at each RK stage's own time (#2605, #2512 row B2).

The ghost fill had no time. `PreallocatedGhostBuffer.update_ghosts` defaulted to t = 0, so a g(t) Neumann value was
read at t = 0 at every stage of every step, silently. Each stage now reads it on the source's clock: the backward
SSP-RK3 stages at t, t - dt and t - dt/2, from the later end of the sub-step.

**Order oracle (1-D).** The manufactured solution is u = a(t) cos(pi x) + b(t) x, with a = 1 + t/2 and b = 1/2 + t.
Its outward derivative is -b(t) at x_min and +b(t) at x_max, and the MMS source is
r = -u_t - D u_xx + u_x^2 / 2.

Max error at t = 0 (T = 0.5, Nt = 10, sigma = 0.4):

| tree | n = 11 | n = 21 | n = 41 | EOC 21 -> 41 |
|---|---|---|---|---|
| 52219be2 (main) | 2.21e-2 | 4.26e-2 | 4.63e-2 | -0.12 |
| this change | 1.82e-3 | 8.05e-4 | 2.13e-4 | 1.92 |

The order with time-dependent data is the order without it. At n = 21, 41, 81, 161 the EOC was 1.92, 1.99, 2.00 with
g(t), and 1.91, 1.99, 2.00 with b constant. So reading the value at the stage times costs no order here. Reading every
stage at the step's start time also gave second order, with errors 0.68 times these. The diffusion-limited sub-step
keeps the time error far below the spatial one, so this oracle does not discriminate between the two stage-time
conventions. It discriminates t = 0 from either.

**Clock, with no source.** The 1-D loop kept a clock only with a source, and the multi-D path, a dimensional split,
takes none (#1991). So the ordinary path has no exact solution here, and its case pins the clock: the value is read
over [0, T] and nowhere else, at both ends, in 1-D and under each split.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers.hjb_weno import HJBWENOSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, neumann_bc

T, NT, SIGMA = 0.5, 10, 0.4
D = SIGMA**2 / 2


def _a(t):
    return 1.0 + 0.5 * t


def _b(t):
    return 0.5 + t


def _u(t, x):
    return _a(t) * np.cos(np.pi * x) + _b(t) * x


def _source(t, x):
    x = np.asarray(x, dtype=float).reshape(-1)
    u_t = 0.5 * np.cos(np.pi * x) + x
    u_x = -np.pi * _a(t) * np.sin(np.pi * x) + _b(t)
    u_xx = -(np.pi**2) * _a(t) * np.cos(np.pi * x)
    return -u_t - D * u_xx + 0.5 * u_x**2


def _problem(bc, d=1, n=21):
    zero = lambda m: 0.0 * np.asarray(m)  # noqa: E731
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * d, Nx_points=[n] * d, boundary_conditions=bc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MFGProblem(
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


def _error(n):
    bc = BoundaryConditions(
        dimension=1,
        segments=[
            BCSegment(name="left", bc_type=BCType.NEUMANN, value=lambda t: -_b(t), boundary="x_min"),
            BCSegment(name="right", bc_type=BCType.NEUMANN, value=lambda t: _b(t), boundary="x_max"),
        ],
        default_bc=BCType.NO_FLUX,
    )
    x = np.linspace(0.0, 1.0, n)
    U = HJBWENOSolver(_problem(bc, n=n)).solve_hjb_system(
        M_density=np.ones((NT + 1, n)),
        U_terminal=_u(T, x),
        U_coupling_prev=np.zeros((NT + 1, n)),
        source_term=_source,
    )
    return float(np.abs(np.asarray(U)[0] - _u(0.0, x)).max())


def test_a_time_dependent_neumann_value_converges_at_second_order():
    errors = [_error(n) for n in (11, 21, 41)]
    assert errors[-1] < 3e-4, errors
    assert np.log2(errors[1] / errors[2]) > 1.8, errors


@pytest.mark.parametrize(("d", "splitting"), [(1, "strang"), (2, "strang"), (2, "godunov")])
def test_every_path_reads_the_value_over_the_whole_interval(d, splitting):
    """With no source: the 1-D loop, which kept no clock without one, and the dimensional split under each method."""
    times = []

    def g(t):
        times.append(float(t))
        return 0.1 * t

    n = 9
    shape = (n,) * d
    solver = HJBWENOSolver(_problem(neumann_bc(dimension=d, value=g), d=d, n=n), splitting_method=splitting)
    times.clear()
    solver.solve_hjb_system(
        M_density=np.ones((NT + 1, *shape)), U_terminal=np.zeros(shape), U_coupling_prev=np.zeros((NT + 1, *shape))
    )
    assert min(times) == pytest.approx(0.0, abs=1e-12)
    assert max(times) == pytest.approx(T, abs=1e-12)
    assert all(-1e-12 <= t <= T + 1e-12 for t in times)

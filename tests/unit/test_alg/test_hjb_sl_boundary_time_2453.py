"""A semi-Lagrangian HJB step reads time-dependent boundary data at the step's own time (#2453).

The gradient's ghost cells (``geometry.get_gradient_operator``, whose ``time`` defaults to 0) and the
post-step enforcement of sub-steps, of every n-D stochastic step and of the n-D DPP step all read a
time-dependent boundary value at t = 0, whatever the step. Measured on these fixtures before the fix:
1-D ADI read 54 of its 64 values at t = 0, 2-D ADI and 2-D DPP all of theirs.

The pin is an invariant of the clock, not a value. The solver marches backward, one step at a time,
so the times at which the boundary value is read, in call order, never increase: a site that reads
at t = 0 inside an earlier step puts a 0 before a later positive time. Mutation-verified: each of the
call sites, reverted alone to the default time, fails its path's case.
"""

from __future__ import annotations

import itertools
import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import HamiltonianBase, L1ControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import neumann_bc

NT = 10


class _TimeDependentH(HamiltonianBase):
    """H = (1 + 4t)|p|^2/2 - m, so the foot speed grows with t and some steps sub-step."""

    def __call__(self, t, x, p, m):
        return (1 + 4 * t) * 0.5 * np.sum(np.asarray(p, dtype=float) ** 2, axis=-1) - np.asarray(m, dtype=float)

    def dp(self, t, x, p, m):
        return (1 + 4 * t) * np.asarray(p, dtype=float)


def _l1():
    return SeparableHamiltonian(control_cost=L1ControlCost(lambda_=1.0), potential=lambda t, x: 0.5)


CASES = {
    "1d-adi": (1, _TimeDependentH, {"cfl_target": 0.5}, 1.0),
    "2d-adi-whole": (2, _TimeDependentH, {}, 0.4),
    "2d-adi-substep": (2, _TimeDependentH, {"cfl_target": 0.3}, 3.0),
    "1d-stochastic": (1, _TimeDependentH, {"diffusion_method": "stochastic"}, 0.4),
    "2d-stochastic": (2, _TimeDependentH, {"diffusion_method": "stochastic"}, 0.4),
    "2d-dpp": (2, _l1, {}, 0.4),
}


@pytest.mark.parametrize("case", list(CASES))
def test_boundary_data_is_read_at_the_steps_own_time(case):
    dim, hamiltonian, kwargs, amp = CASES[case]
    reads = []

    def value(*args, **kw):
        t = float(kw.get("time", args[-1] if args else 0.0))
        reads.append(round(t, 9))
        return 0.05 * t

    n = {1: 21, 2: 9}[dim]
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * dim, Nx_points=[n] * dim, boundary_conditions=neumann_bc(value=value, dimension=dim)
    )
    problem = MFGProblem(
        model=Model(hamiltonian=hamiltonian(), volatility=0.2),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=1.0),
        Nt=NT,
    )
    axes = np.meshgrid(*grid.coordinates, indexing="ij")
    u_terminal = amp * sum((X - 0.4) ** 2 for X in axes)
    m = np.ones((NT + 1, *u_terminal.shape))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        HJBSemiLagrangianSolver(problem, **kwargs).solve_hjb_system(
            m, u_terminal, np.broadcast_to(u_terminal, m.shape).copy()
        )

    step_times = {round(k * problem.dt, 9) for k in range(NT)}
    assert set(reads) == step_times, f"read at {sorted(set(reads))}, steps at {sorted(step_times)}"
    rises = [(i, a, b) for i, (a, b) in enumerate(itertools.pairwise(reads)) if b > a]
    assert not rises, (
        f"the read time rose {len(rises)} times, first at read {rises[0][0]}: {rises[0][1]} then {rises[0][2]}. "
        f"A site read the boundary value at a time other than its step's (#2453)."
    )

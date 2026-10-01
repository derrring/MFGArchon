"""A semi-Lagrangian HJB step reads time-dependent boundary data at the step's own time (#2453).

The gradient's ghost cells (``geometry.get_gradient_operator``, whose ``time`` defaults to 0) and the
post-step enforcement of operator-splitting sub-steps, of every n-D stochastic step and of the n-D DPP
step read a time-dependent boundary value at t = 0, whatever the step. Measured on these cases before
the fix: ``1d-adi`` read 146 of its 154 values at t = 0, ``1d-stochastic`` 30 of 64, and the 2-D cases
all of theirs.

The clock is one time per step, the step's ``time_idx * dt``, for every read the step makes: the same
clock #2452 gave H, the foot and the value update. It is a decision, not a measured optimum: the step
differentiates U^{n+1}, whose own time is t_{n+1}, and no oracle here separates the two: the solver refuses
inhomogeneous Neumann data (#1936), and with diffusion it did not hold even a constant value. #2453
records the alternative.

The solver refuses a time-dependent Neumann value (#1936: on no path does it reach every place it must),
so no boundary condition accepted at construction reaches these reads today; one set on the geometry
afterwards is not checked again. The pin holds the clock an implementation of
that data will use, with the refusal lifted for the test; the second test pins the refusal itself.

The pin is an invariant of that clock, not a value. Each read is tagged with the step it belongs to: the
step functions receive their index, and the CFL schedule, which runs just before step n, takes the index
of the next step entered. Every read must be at its tag's time. Mutation-verified: each call site
reverted alone to t = 0, or moved to the next or the previous step's time, fails its path's case.
"""

from __future__ import annotations

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
    "1d-dpp": (1, _l1, {}, 0.4),
    "2d-dpp": (2, _l1, {}, 0.4),
}


def _tag_by_step(monkeypatch, events):
    """Log, in call order, entry to each step function (with its index) and to the CFL schedule."""
    for name in ("_solve_timestep_semi_lagrangian", "_solve_timestep_semi_lagrangian_with_dt"):
        original = getattr(HJBSemiLagrangianSolver, name)

        def entered(self, U_next, M_next, U_coupling, time_idx, *rest, _original=original):
            events.append(("step", time_idx))
            return _original(self, U_next, M_next, U_coupling, time_idx, *rest)

        monkeypatch.setattr(HJBSemiLagrangianSolver, name, entered)
    schedule = HJBSemiLagrangianSolver._compute_cfl_and_substeps

    def scheduled(self, *args):
        events.append(("schedule", None))
        return schedule(self, *args)

    monkeypatch.setattr(HJBSemiLagrangianSolver, "_compute_cfl_and_substeps", scheduled)


def _reads_with_their_step(events):
    """(time read, index of the step it belongs to): a schedule's reads belong to the next step entered."""
    tagged, pending, current = [], [], None
    for kind, value in events:
        if kind == "step":
            current = value
            tagged.extend((t, current) for t in pending)
            pending = []
        elif kind == "schedule":
            current = None
        elif current is None:
            pending.append(value)
        else:
            tagged.append((value, current))
    assert not pending, f"{len(pending)} reads after the last step"
    return tagged


@pytest.mark.parametrize("case", list(CASES))
def test_boundary_data_is_read_at_the_steps_own_time(case, monkeypatch):
    dim, hamiltonian, kwargs, amp = CASES[case]
    monkeypatch.setattr(HJBSemiLagrangianSolver, "honors_inhomogeneous_neumann", True)  # refused since #1936
    events = []
    _tag_by_step(monkeypatch, events)

    def value(*args, **kw):
        t = float(kw.get("time", args[-1] if args else 0.0))
        events.append(("read", t))
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

    tagged = _reads_with_their_step(events)
    assert {n for _, n in tagged} == set(range(NT)), "some step read no boundary value"
    wrong = [(t, n) for t, n in tagged if not np.isclose(t, n * problem.dt, rtol=0, atol=1e-12)]
    assert not wrong, (
        f"{len(wrong)} of {len(tagged)} boundary reads are not at their step's time; first: read at "
        f"t = {wrong[0][0]:.6g} in step {wrong[0][1]} (t = {wrong[0][1] * problem.dt:.6g}) (#2453)"
    )


@pytest.mark.parametrize("value", [0.3, lambda t: 0.05 * t])
def test_a_neumann_value_the_solver_does_not_apply_is_refused(value):
    """RECORDED DEFECT, not a contract (#1936). On no path does the solver carry a Neumann value
    everywhere it touches a wall, and on an exact solution with sigma = 0.2 no path measured converged
    with g. Carrying g on every path retires this test: set ``honors_inhomogeneous_neumann = True`` on
    HJBSemiLagrangianSolver and delete it."""
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=neumann_bc(value=value, dimension=1)
    )
    problem = MFGProblem(
        model=Model(hamiltonian=_TimeDependentH(), volatility=0.2),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=1.0),
        Nt=NT,
    )
    with pytest.raises(NotImplementedError, match="#1936"):
        HJBSemiLagrangianSolver(problem)

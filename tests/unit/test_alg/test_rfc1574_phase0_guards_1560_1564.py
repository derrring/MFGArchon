"""RFC #1574 Phase 0 capability-honesty guards: fail loud where a declared/dispatched surface is
broader than the code that honors it.

- #1564: HJBFDMSolver.build_linearized_operator (the strict-adjoint FP operator, #707) hardcodes
  no-flux at every boundary while HJBFDMSolver declares DIRICHLET supported for the normal solve.
- #1560: HJBSemiLagrangianSolver collapsed a mixed per-axis BC to one operation on every axis. Its
  refusal pin retired when per-axis handling landed, as the pin's own message asked; the law it stood
  in for is `tests/unit/test_alg/test_sl_channel_separates_by_axis_1560_1697.py`, and its control
  `test_sl_uniform_bc_still_constructs_1560` stays here.

ADMISSION (#2257). Class 3, a defect pin, TEMPORARY by construction: RFC #1574 has later phases,
and the guard is a refusal standing in for an implementation that phase is meant to supply. So it
carries a retirement condition that fires when its own capability lands -- the refusal stops being
raised, the test goes red, and the message says to delete the pin rather than restore the raise.

A pin without that is what makes a capability harder to add than to leave missing: the next person
implementing the capability finds a red test asserting the refusal and cannot tell whether it is a
regression they caused or the pin's own success.
"""

from __future__ import annotations

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian import HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
from mfgarchon.geometry.boundary import dirichlet_bc, no_flux_bc
from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid

_OBSERVED_1564 = "build_linearized_operator did not refuse a Dirichlet BC."

_CAUSES_1564 = {
    "the operator now assembles a Dirichlet boundary": (
        "do NOT restore the raise. Delete this test and replace it with one checking the operator is "
        "no longer mass-conserving at a Dirichlet boundary -- the row sums must show outflow, which "
        "is what the hardcoded no-flux assembly could not express. `test_build_linearized_operator_ok_on_no_flux_1564` stays either way. See #1564"
    ),
    "the guard was moved or dropped with the assembly unchanged": (
        "the operator is silently reporting a mass-conserving wall for an absorbing one, which is "
        "the defect the refusal replaced -- restore it. See #1564"
    ),
}

# `test_sl_uniform_bc_still_constructs_1560` and
# `test_build_linearized_operator_ok_on_no_flux_1564` stay under either disposition.


def _components() -> MFGComponents:
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0))
    return MFGComponents(hamiltonian=H, u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0)


def _problem(bc, bounds, npts) -> MFGProblem:
    grid = TensorProductGrid(bounds=bounds, Nx_points=npts, boundary_conditions=bc)
    return MFGProblem(geometry=grid, T=0.2, Nt=2, volatility=0.1, components=_components())


def test_sl_uniform_bc_still_constructs_1560():
    """A single BC type across all axes constructs.

    The control the #1560 refusal pin had, kept when per-axis handling retired that pin: the per-axis
    read still refuses an axis whose two faces disagree, and must not refuse a uniform BC.
    """
    solver = HJBSemiLagrangianSolver(problem=_problem(no_flux_bc(dimension=2), [(0.0, 1.0), (0.0, 1.0)], [6, 6]))
    assert solver is not None


def test_build_linearized_operator_fails_loud_on_dirichlet_1564(still_refused):
    """The strict-adjoint FP operator hardcodes no-flux; a Dirichlet BC must raise, not be silently
    treated as mass-conserving no-flux."""
    solver = HJBFDMSolver(problem=_problem(dirichlet_bc(dimension=1), [(0.0, 1.0)], [11]))
    U, M = np.zeros(11), np.ones(11) / 11
    with still_refused("1564", observed=_OBSERVED_1564, causes=_CAUSES_1564):
        solver.build_linearized_operator(U, M, time=0.0)


def test_build_linearized_operator_ok_on_no_flux_1564():
    """No-flux (the honored BC) must still build the operator.

    The control for the pin above, and the test that survives its retirement.
    """
    solver = HJBFDMSolver(problem=_problem(no_flux_bc(dimension=1), [(0.0, 1.0)], [11]))
    U, M = np.zeros(11), np.ones(11) / 11
    A = solver.build_linearized_operator(U, M, time=0.0)
    assert A.shape == (11, 11)

"""RFC #1574 Phase 0 capability-honesty guards: fail loud where a declared/dispatched surface is
broader than the code that honors it.

- #1560: HJBSemiLagrangianSolver collapses a mixed per-axis BC (segments mapping to different
  geometric operations, e.g. no-flux + periodic) to the first segment's single op applied to all
  axes.
- #1564: HJBFDMSolver.build_linearized_operator (the strict-adjoint FP operator, #707) hardcodes
  no-flux at every boundary while HJBFDMSolver declares DIRICHLET supported for the normal solve.

ADMISSION (#2257). Class 3, two defect pins in one file because they are one RFC phase. Both are
TEMPORARY by construction: RFC #1574 has later phases, and each guard is a refusal standing in for
an implementation that phase is meant to supply. So each carries a retirement condition that fires
when its own capability lands -- the refusal stops being raised, the test goes red, and the message
says to delete the pin rather than restore the raise.

A pin without that is what makes a capability harder to add than to leave missing: the next person
implementing per-axis BC in the SL solver finds a red test asserting the refusal and cannot tell
whether it is a regression they caused or the pin's own success.

WHAT THE #1560 PIN COVERS
-------------------------
One owner: `bc_utils.refuse_mixed_per_axis`, which `HJBSemiLagrangianSolver` calls at construction
and, through the `_checked_bc_type_string` wrapper, at every solve-time site.

The predicate existed TWICE until #2284 -- an inline copy in `__init__`, which had already diverged
from the owner -- and this pin then held that copy and nothing else. It now holds the owner.
Measured: disabling the owner's raise turns `test_sl_mixed_per_axis_bc_fails_loud_1560` red, where
before the consolidation the same mutation left this file green (4 passed either way).

That closes the reach gap this file used to record. Per-axis handling landing in the owner alone can
no longer leave the pin green, because there is no second copy left to keep raising.
"""

from __future__ import annotations

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian import HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, dirichlet_bc, no_flux_bc
from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid

_RETIRE_1560 = """HJBSemiLagrangianSolver did not refuse a mixed per-axis BC at construction.

That is the retirement condition ONLY if the solver now HONOURS per-axis BC, which is the RFC #1574
phase this pin existed to demand. Check which happened:

  * per-axis handling landed -- do NOT restore the raise. Delete this test and replace it with one
    that checks the ops are applied PER AXIS: no-flux on x must reflect while periodic on y wraps,
    which is the collapse the refusal was standing in for.
  * the guard moved -- `bc_utils.refuse_mixed_per_axis` has been its one owner since #2284, so the
    refusal is still the correct behaviour and this pin wants re-pointing, not deleting.

`test_sl_uniform_bc_still_constructs_1560` stays either way. See #1560, #2284."""

_RETIRE_1564 = """build_linearized_operator did not refuse a Dirichlet BC.

That is the retirement condition ONLY if the operator now ASSEMBLES a Dirichlet boundary rather
than hardcoding no-flux. If instead the guard was moved or dropped with the assembly unchanged, the
operator is silently reporting a mass-conserving wall for an absorbing one, which is the defect the
refusal replaced -- restore it.

If the capability landed: do NOT restore the raise. Delete this test and replace it with one that
checks the operator is no longer mass-conserving at a Dirichlet boundary -- the row sums must show
outflow, which is precisely what the hardcoded no-flux assembly could not express.

`test_build_linearized_operator_ok_on_no_flux_1564` stays either way. See #1564."""


def _components() -> MFGComponents:
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0))
    return MFGComponents(hamiltonian=H, u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0)


def _problem(bc, bounds, npts) -> MFGProblem:
    grid = TensorProductGrid(bounds=bounds, Nx_points=npts, boundary_conditions=bc)
    return MFGProblem(geometry=grid, T=0.2, Nt=2, sigma=0.1, components=_components())


def test_sl_mixed_per_axis_bc_fails_loud_1560(still_refused):
    """no-flux (x) + periodic (y) map to different geometric ops (reflect vs periodic); SL collapses
    them to one op on all axes, so construction must raise rather than silently pick the first."""
    mixed = BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="wx", boundary="x_min", bc_type=BCType.NO_FLUX),
            BCSegment(name="ex", boundary="x_max", bc_type=BCType.NO_FLUX),
            BCSegment(name="py0", boundary="y_min", bc_type=BCType.PERIODIC),
            BCSegment(name="py1", boundary="y_max", bc_type=BCType.PERIODIC),
        ],
        # `mixed_bc` is deprecated since v0.18.0 -- three minors back against 0.22.0.dev0 -- and
        # was this file's only use of it (#2257). Its body (conditions.py:1265-1273) forwards
        # every argument to this constructor unchanged, so the migration is verbatim; the one
        # thing not carried by the constructor's own defaults is `default_bc`, which the factory
        # supplied as NEUMANN and the constructor leaves None. Passed explicitly for that reason.
        default_bc=BCType.NEUMANN,
    )
    with still_refused("mixed per-axis", _RETIRE_1560):
        HJBSemiLagrangianSolver(problem=_problem(mixed, [(0.0, 1.0), (0.0, 1.0)], [6, 6]))


def test_sl_uniform_bc_still_constructs_1560():
    """A single BC type across all axes must be unaffected by the mixed-BC guard.

    The control. Without it a guard that refused every BC would satisfy the test above.
    """
    solver = HJBSemiLagrangianSolver(problem=_problem(no_flux_bc(dimension=2), [(0.0, 1.0), (0.0, 1.0)], [6, 6]))
    assert solver is not None


def test_build_linearized_operator_fails_loud_on_dirichlet_1564(still_refused):
    """The strict-adjoint FP operator hardcodes no-flux; a Dirichlet BC must raise, not be silently
    treated as mass-conserving no-flux."""
    solver = HJBFDMSolver(problem=_problem(dirichlet_bc(dimension=1), [(0.0, 1.0)], [11]))
    U, M = np.zeros(11), np.ones(11) / 11
    with still_refused("1564", _RETIRE_1564):
        solver.build_linearized_operator(U, M, time=0.0)


def test_build_linearized_operator_ok_on_no_flux_1564():
    """No-flux (the honored BC) must still build the operator.

    The control for the pin above, and the test that survives its retirement.
    """
    solver = HJBFDMSolver(problem=_problem(no_flux_bc(dimension=1), [(0.0, 1.0)], [11]))
    U, M = np.zeros(11), np.ones(11) / 11
    A = solver.build_linearized_operator(U, M, time=0.0)
    assert A.shape == (11, 11)

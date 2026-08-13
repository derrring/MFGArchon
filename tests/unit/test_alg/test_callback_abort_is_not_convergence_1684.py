"""A user abort is not convergence (#1684 item 2).

`FixedPointIterator.solve(iteration_callback=...)` let a callback returning False set
`converged = True` with `reason="callback_stopped"`, before the real criteria were ever
evaluated. Measured on the fixture below at HEAD~:

    no callback       3 iterations   converged=False   l2distu_rel = 3.259e-01
    callback aborts   1 iteration    converged=True    l2distu_rel = 1.000e+00

The aborted run reported success at THREE TIMES the error of the run that admitted failure.

The fix is not "report False on abort" -- that is one wrong constant replacing another. An abort
is evidence of neither outcome, so the criteria are evaluated at that iterate and the flag reports
what they say. Both directions are asserted below; a hard-coded False passes the first test and
fails the second.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _problem():
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    return MFGProblem(
        geometry=grid,
        T=1.0,
        Nt=10,
        sigma=0.3,
        components=MFGComponents(
            m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2),
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
        ),
    )


def _solve(tolerance: float, callback):
    problem = _problem()
    iterator = FixedPointIterator(
        problem, hjb_solver=HJBFDMSolver(problem), fp_solver=FPFDMSolver(problem), relaxation=0.5
    )
    result = iterator.solve(max_iterations=3, tolerance=tolerance, iteration_callback=callback)
    return iterator, result


def test_an_abort_at_an_unconverged_iterate_does_not_claim_convergence():
    """The defect, directly: abort at iteration 1, where the error is LARGER than at iteration 3."""
    aborting, result = _solve(1e-10, lambda *_: False)
    error_at_abort = aborting.l2distu_rel[aborting.iterations_run - 1]

    assert aborting.iterations_run == 1, "the callback must actually have stopped the loop"
    assert result.converged is False, (
        f"aborted at l2distu_rel = {error_at_abort:.3e} against tolerance 1e-10 and reported convergence"
    )


def test_the_aborted_error_really_is_worse_than_the_run_that_reports_failure():
    """The control that makes the test above mean something.

    Without it, `converged is False` on the aborted run is satisfied by a solver that simply never
    converges, and the test would not be about the abort at all.
    """
    aborting, _ = _solve(1e-10, lambda *_: False)
    running, running_result = _solve(1e-10, None)

    err_abort = aborting.l2distu_rel[aborting.iterations_run - 1]
    err_full = running.l2distu_rel[running.iterations_run - 1]

    assert running_result.converged is False, "the un-aborted run is supposed to fail to converge here"
    assert err_abort > err_full, (
        f"the abort must land on a WORSE iterate for this comparison to bite: {err_abort:.3e} vs {err_full:.3e}"
    )


def test_an_abort_at_a_converged_iterate_still_reports_convergence():
    """The other direction, and the reason the fix is not `converged = False`.

    With a tolerance the first iterate already satisfies, aborting there must still report True --
    and say why. A hard-coded False passes the first test in this file and fails this one.
    """
    aborting, result = _solve(10.0, lambda *_: False)

    assert aborting.iterations_run == 1
    assert result.converged is True, "the criteria are met at this iterate; the abort does not unmake that"
    reason = (result.metadata or {}).get("convergence_reason", "")
    assert "callback_stopped" in reason, f"the abort must still be recorded: {reason!r}"
    assert "Converged" in reason, f"and the criteria's own verdict carried with it: {reason!r}"


@pytest.mark.parametrize("returned", [True, None])
def test_a_callback_that_does_not_return_false_does_not_stop_the_loop(returned):
    """`should_continue is False` is the stop signal, not falsiness: None must not stop it."""
    iterator, _ = _solve(1e-10, lambda *_: returned)
    assert iterator.iterations_run == 3

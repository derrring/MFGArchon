"""A coupled result whose returned solution fails output validation is not reported converged (#2323).

At 5b97f465 a 2-D FVM_MUSCL solve returned ``converged=True`` with density -3.603e-10 at 20 nodes. The iterator
had already run `validate_solver_output`, which judged the density invalid, and it logged that at WARNING and
kept the verdict. Under the ruling recorded on #1878 -- a coupled result reports what its parts failed -- the
verdict is now False.

That fixture is a minutes-scale 2-D solve, so the contract is pinned on the 1-D smoke fixture, with an FP solver
that writes one negative entry into the density it returns. The Picard map stays deterministic, so the criteria
are still met: what the test isolates is whether the verdict reads the validation.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid, no_flux_bc


def _problem() -> MFGProblem:
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MFGProblem(
            model=Model(hamiltonian=hamiltonian, sigma=0.0),
            domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1)),
            conditions=Conditions(
                m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2), u_terminal=lambda x: 0.0, T=1.0
            ),
            Nt=10,
        )


def _solve(negative_entry: float | None):
    problem = _problem()
    fp_solver = FPFDMSolver(problem)
    if negative_entry is not None:
        original = fp_solver.solve_fp_system

        @functools.wraps(original)  # keeps the signature the iterator inspects to route the drift
        def returns_a_negative_density(*args, **kwargs):
            density = np.array(original(*args, **kwargs), dtype=float)
            density[-1, 0] = negative_entry
            return density

        fp_solver.solve_fp_system = returns_a_negative_density
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FixedPointIterator(problem, hjb_solver=HJBFDMSolver(problem), fp_solver=fp_solver).solve(
            max_iterations=150, tolerance=1e-3, verbose=False
        )


def test_a_converged_solve_with_an_invalid_density_is_not_reported_converged():
    result = _solve(negative_entry=-1e-3)
    reason = result.metadata["convergence_reason"]
    assert result.converged is False, reason
    assert reason.startswith("output_invalid"), reason
    assert "negative" in reason, reason
    assert result.metadata["output_validation"]["is_valid"] is False


def test_the_same_solve_with_a_valid_density_is_converged():
    """The presence half: the fixture converges when the density is left alone."""
    result = _solve(negative_entry=None)
    assert result.converged is True, result.metadata["convergence_reason"]
    assert "output_validation" not in result.metadata

"""Inverted pins for Issue #1285 / Issue #1361.

History:
- #1285 added a fail-loud ``NotImplementedError`` guard to ``MFGResidual`` because
  the coupled-Newton path silently ignored ``source_term_hjb``,
  ``source_term_fp``, ``nonlocal_operator``, and ``obstacle`` — converging to a
  wrong equilibrium.
- #1361 wired those four terms into the residual path (composed from the
  ``(U, M)`` residual arguments via the single-source ``source_composition``
  helpers shared with Picard), so the guard is removed and the Newton path
  *solves* these problems instead of refusing them.

These tests were previously ``pytest.raises(NotImplementedError)`` pins. They are
inverted here: ``MFGResidual`` / ``NewtonMFGSolver`` must construct cleanly and
produce a finite solve for each extended field. Newton-vs-Picard equilibrium
parity is pinned in ``tests/integration/test_source_term_wiring.py``.

Refs #1285 #1361 #1043.
"""

from __future__ import annotations

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

# ---------------------------------------------------------------------------
# Minimal problem factory (small/short/weakly coupled: keeps the FD-Jacobian
# Newton solve in the physical basin and fast).
# ---------------------------------------------------------------------------

_NX = 5  # Nx=5 intervals -> 6 grid points
_NT = 3
_GRID = _NX + 1


def _components() -> MFGComponents:
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.2 * m,
        coupling_dm=lambda m: 0.2,
    )
    return MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: np.exp(-10.0 * (np.asarray(x) - 0.5) ** 2),
    )


def _make(**extra) -> MFGProblem:
    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[_NX + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        T=0.15,
        Nt=_NT,
        sigma=0.3,
        components=_components(),
        **extra,
    )


def _make_plain_problem() -> MFGProblem:
    return _make()


def _make_problem_with_nonlocal() -> MFGProblem:
    return _make(nonlocal_operator=0.3 * np.eye(_GRID))


def _make_problem_with_source_hjb() -> MFGProblem:
    return _make(source_term_hjb=lambda x, m, v, t: 0.5 * np.ones(len(x)))


def _make_problem_with_source_fp() -> MFGProblem:
    return _make(source_term_fp=lambda x, m, v, t: 0.02 * np.ones(len(x)))


def _make_problem_with_obstacle() -> MFGProblem:
    return _make(obstacle=lambda x: np.asarray(x) - 0.5)


def _make_solvers(problem: MFGProblem):
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    return HJBFDMSolver(problem), FPFDMSolver(problem)


def _assert_finite_solve(problem: MFGProblem) -> tuple[np.ndarray, np.ndarray]:
    """NewtonMFGSolver must produce a finite (U, M) for the given problem."""
    from mfgarchon.alg.numerical.coupling.newton_mfg_solver import NewtonMFGSolver

    hjb_solver, fp_solver = _make_solvers(problem)
    solver = NewtonMFGSolver(
        problem, hjb_solver, fp_solver, picard_warmup=3, newton_max_iterations=15, newton_tolerance=1e-8
    )
    U, M, _info = solver.solve(max_iterations=18, tolerance=1e-8, verbose=False)
    assert np.all(np.isfinite(U)), "U not finite"
    assert np.all(np.isfinite(M)), "M not finite"
    assert U.shape == solver.mfg_residual.solution_shape
    assert M.shape == solver.mfg_residual.solution_shape
    return U, M


def _picard_solve(problem: MFGProblem) -> tuple[np.ndarray, np.ndarray]:
    """The same problem through the Picard path, as the oracle for the Newton equilibrium."""
    from mfgarchon.alg.numerical.coupling import FixedPointIterator

    hjb_solver, fp_solver = _make_solvers(problem)
    iterator = FixedPointIterator(problem, hjb_solver=hjb_solver, fp_solver=fp_solver)
    result = iterator.solve(max_iterations=200, tolerance=1e-10, verbose=False)
    return result.U, result.M


# ---------------------------------------------------------------------------
# MFGResidual must CONSTRUCT (no NotImplementedError) for each field (#1361)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# NewtonMFGSolver must CONSTRUCT and SOLVE (finite output) for each field
# ---------------------------------------------------------------------------


def test_newton_solver_solves_with_obstacle():
    """Newton must reach the SAME equilibrium as Picard on an obstacle problem.

    This is the only end-to-end obstacle coverage of the Newton path anywhere -- the parity
    class in ``tests/integration/test_source_term_wiring.py`` covers source_term_hjb,
    source_term_fp and nonlocal_operator but has no obstacle case, and
    ``test_issue1361_source_composition.py`` pins the obstacle only at the composition level,
    never through a solve. Finiteness and shape are satisfied by a Newton path that ignores the
    obstacle entirely, which is the #1285 defect.

    Threshold. The obstacle enters as ``(1/eps)*max(0, psi(x))`` with ``eps = 1e6``
    (source_composition.py:130-133), so it is a small perturbation and the tolerance has to sit
    below it or it certifies nothing. Measured on this fixture:

    - Newton-vs-Picard relative gap with the obstacle wired to both: 2.4e-10 (U), 7.6e-11 (M).
    - Effect of the obstacle itself, same solver with it on vs off: 9.5e-07 (U), 2.9e-08 (M).

    1e-8 sits between them: 41x above the measured agreement in U and 95x below the gap a
    Newton path that silently dropped the obstacle would open. A tolerance of 1e-5 would be
    above the obstacle's whole effect and would pass with the term deleted.
    """
    problem = _make_problem_with_obstacle()
    U_newton, M_newton = _assert_finite_solve(problem)
    U_picard, M_picard = _picard_solve(problem)

    u_gap = np.max(np.abs(U_newton - U_picard)) / np.max(np.abs(U_picard))
    m_gap = np.max(np.abs(M_newton - M_picard)) / np.max(np.abs(M_picard))
    assert u_gap < 1e-8, f"Newton and Picard disagree on the obstacle equilibrium: U gap {u_gap:.3e}"
    assert m_gap < 1e-8, f"Newton and Picard disagree on the obstacle equilibrium: M gap {m_gap:.3e}"


# ---------------------------------------------------------------------------
# Plain problem (no extended fields) must still construct and solve
# ---------------------------------------------------------------------------


def test_mfg_residual_plain_problem_does_not_raise():
    from mfgarchon.alg.numerical.coupling.mfg_residual import MFGResidual

    problem = _make_plain_problem()
    hjb_solver, fp_solver = _make_solvers(problem)
    residual = MFGResidual(problem, hjb_solver, fp_solver)
    assert residual.problem is problem
    assert residual.num_time_steps == problem.Nt + 1

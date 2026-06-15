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


def _assert_finite_solve(problem: MFGProblem) -> None:
    """NewtonMFGSolver must produce a finite (U, M) for the given problem."""
    from mfgarchon.alg.numerical.coupling.newton_mfg_solver import NewtonMFGSolver

    hjb_solver, fp_solver = _make_solvers(problem)
    solver = NewtonMFGSolver(
        problem, hjb_solver, fp_solver, picard_warmup=3, newton_max_iterations=15, newton_tolerance=1e-8
    )
    U, M, info = solver.solve(max_iterations=18, tolerance=1e-8, verbose=False)
    assert np.all(np.isfinite(U)), "U not finite"
    assert np.all(np.isfinite(M)), "M not finite"
    assert U.shape == solver.mfg_residual.solution_shape
    assert M.shape == solver.mfg_residual.solution_shape


# ---------------------------------------------------------------------------
# MFGResidual must CONSTRUCT (no NotImplementedError) for each field (#1361)
# ---------------------------------------------------------------------------


def test_mfg_residual_constructs_with_nonlocal_operator():
    from mfgarchon.alg.numerical.coupling.mfg_residual import MFGResidual

    problem = _make_problem_with_nonlocal()
    hjb_solver, fp_solver = _make_solvers(problem)
    residual = MFGResidual(problem, hjb_solver, fp_solver)
    assert residual.problem is problem


def test_mfg_residual_constructs_with_source_term_hjb():
    from mfgarchon.alg.numerical.coupling.mfg_residual import MFGResidual

    problem = _make_problem_with_source_hjb()
    hjb_solver, fp_solver = _make_solvers(problem)
    residual = MFGResidual(problem, hjb_solver, fp_solver)
    assert residual.problem is problem


def test_mfg_residual_constructs_with_source_term_fp():
    from mfgarchon.alg.numerical.coupling.mfg_residual import MFGResidual

    problem = _make_problem_with_source_fp()
    hjb_solver, fp_solver = _make_solvers(problem)
    residual = MFGResidual(problem, hjb_solver, fp_solver)
    assert residual.problem is problem


def test_mfg_residual_constructs_with_obstacle():
    from mfgarchon.alg.numerical.coupling.mfg_residual import MFGResidual

    problem = _make_problem_with_obstacle()
    hjb_solver, fp_solver = _make_solvers(problem)
    residual = MFGResidual(problem, hjb_solver, fp_solver)
    assert residual.problem is problem


# ---------------------------------------------------------------------------
# NewtonMFGSolver must CONSTRUCT and SOLVE (finite output) for each field
# ---------------------------------------------------------------------------


def test_newton_solver_solves_with_nonlocal_operator():
    _assert_finite_solve(_make_problem_with_nonlocal())


def test_newton_solver_solves_with_source_term_hjb():
    _assert_finite_solve(_make_problem_with_source_hjb())


def test_newton_solver_solves_with_source_term_fp():
    _assert_finite_solve(_make_problem_with_source_fp())


def test_newton_solver_solves_with_obstacle():
    _assert_finite_solve(_make_problem_with_obstacle())


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


def test_newton_solver_plain_problem_does_not_raise():
    from mfgarchon.alg.numerical.coupling.newton_mfg_solver import NewtonMFGSolver

    problem = _make_plain_problem()
    hjb_solver, fp_solver = _make_solvers(problem)
    solver = NewtonMFGSolver(problem, hjb_solver, fp_solver)
    assert solver.problem is problem

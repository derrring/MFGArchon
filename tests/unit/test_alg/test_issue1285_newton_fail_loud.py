"""Pinning tests for Issue #1285 (2026-06-11 survey).

MFGResidual / NewtonMFGSolver silently ignored source_term_hjb,
source_term_fp, nonlocal_operator, and obstacle, causing Newton to
converge to a wrong equilibrium whenever any of those problem fields
is set.

Fix: MFGResidual.__init__ raises NotImplementedError with a clear
message when any of the four unsupported fields is non-None.
FixedPointIterator correctly composes all four terms and should be
used instead.

Each test here asserts that NotImplementedError is raised.
On unfixed code (no guard in __init__): the object constructs silently
-- no error -- so pytest.raises(...) catches no exception and the test
FAILS.  After the fix the guard fires and the tests PASS.

Refs #1285 #1043.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem

# ---------------------------------------------------------------------------
# Minimal problem factory
# ---------------------------------------------------------------------------


def _make_plain_problem(Nx: int = 5, Nt: int = 3) -> MFGProblem:
    """1-D problem without any extended fields."""
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    comp = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )
    return MFGProblem(Nx=Nx, xmin=0.0, xmax=1.0, T=0.3, Nt=Nt, sigma=0.2, components=comp)


def _make_problem_with_nonlocal(Nx: int = 5, Nt: int = 3) -> MFGProblem:
    """1-D problem with a diagonal nonlocal_operator."""
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    comp = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )
    return MFGProblem(
        Nx=Nx,
        xmin=0.0,
        xmax=1.0,
        T=0.3,
        Nt=Nt,
        sigma=0.2,
        components=comp,
        nonlocal_operator=np.eye(Nx),
    )


def _make_problem_with_source_hjb(Nx: int = 5, Nt: int = 3) -> MFGProblem:
    """1-D problem with source_term_hjb."""
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    comp = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )
    return MFGProblem(
        Nx=Nx,
        xmin=0.0,
        xmax=1.0,
        T=0.3,
        Nt=Nt,
        sigma=0.2,
        components=comp,
        source_term_hjb=lambda x, m, v, t: np.zeros(len(x)),
    )


def _make_problem_with_source_fp(Nx: int = 5, Nt: int = 3) -> MFGProblem:
    """1-D problem with source_term_fp."""
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    comp = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )
    return MFGProblem(
        Nx=Nx,
        xmin=0.0,
        xmax=1.0,
        T=0.3,
        Nt=Nt,
        sigma=0.2,
        components=comp,
        source_term_fp=lambda x, m, v, t: np.zeros(len(x)),
    )


def _make_problem_with_obstacle(Nx: int = 5, Nt: int = 3) -> MFGProblem:
    """1-D problem with obstacle."""
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    comp = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )
    return MFGProblem(
        Nx=Nx,
        xmin=0.0,
        xmax=1.0,
        T=0.3,
        Nt=Nt,
        sigma=0.2,
        components=comp,
        obstacle=lambda x: np.zeros(len(x)),
    )


def _make_solvers(problem: MFGProblem):
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    return HJBFDMSolver(problem), FPFDMSolver(problem)


# ---------------------------------------------------------------------------
# MFGResidual must raise NotImplementedError for each unsupported field
# ---------------------------------------------------------------------------


def test_mfg_residual_raises_on_nonlocal_operator():
    """MFGResidual must raise NotImplementedError when nonlocal_operator is set.

    Pre-fix: object constructs silently, no exception raised.
    Post-fix: NotImplementedError is raised with '#1285' in the message.
    """
    from mfgarchon.alg.numerical.coupling.mfg_residual import MFGResidual

    problem = _make_problem_with_nonlocal()
    hjb_solver, fp_solver = _make_solvers(problem)

    with pytest.raises(NotImplementedError, match="#1285"):
        MFGResidual(problem, hjb_solver, fp_solver)


def test_mfg_residual_raises_on_source_term_hjb():
    """MFGResidual must raise NotImplementedError when source_term_hjb is set."""
    from mfgarchon.alg.numerical.coupling.mfg_residual import MFGResidual

    problem = _make_problem_with_source_hjb()
    hjb_solver, fp_solver = _make_solvers(problem)

    with pytest.raises(NotImplementedError, match="#1285"):
        MFGResidual(problem, hjb_solver, fp_solver)


def test_mfg_residual_raises_on_source_term_fp():
    """MFGResidual must raise NotImplementedError when source_term_fp is set."""
    from mfgarchon.alg.numerical.coupling.mfg_residual import MFGResidual

    problem = _make_problem_with_source_fp()
    hjb_solver, fp_solver = _make_solvers(problem)

    with pytest.raises(NotImplementedError, match="#1285"):
        MFGResidual(problem, hjb_solver, fp_solver)


def test_mfg_residual_raises_on_obstacle():
    """MFGResidual must raise NotImplementedError when obstacle is set."""
    from mfgarchon.alg.numerical.coupling.mfg_residual import MFGResidual

    problem = _make_problem_with_obstacle()
    hjb_solver, fp_solver = _make_solvers(problem)

    with pytest.raises(NotImplementedError, match="#1285"):
        MFGResidual(problem, hjb_solver, fp_solver)


# ---------------------------------------------------------------------------
# NewtonMFGSolver (builds MFGResidual internally) must propagate the error
# ---------------------------------------------------------------------------


def test_newton_solver_raises_on_nonlocal_operator():
    """NewtonMFGSolver must raise NotImplementedError when nonlocal_operator is set.

    NewtonMFGSolver constructs MFGResidual internally; the guard in
    MFGResidual.__init__ must propagate through.
    """
    from mfgarchon.alg.numerical.coupling.newton_mfg_solver import NewtonMFGSolver

    problem = _make_problem_with_nonlocal()
    hjb_solver, fp_solver = _make_solvers(problem)

    with pytest.raises(NotImplementedError, match="#1285"):
        NewtonMFGSolver(problem, hjb_solver, fp_solver)


def test_newton_solver_raises_on_source_term_hjb():
    """NewtonMFGSolver must raise NotImplementedError when source_term_hjb is set."""
    from mfgarchon.alg.numerical.coupling.newton_mfg_solver import NewtonMFGSolver

    problem = _make_problem_with_source_hjb()
    hjb_solver, fp_solver = _make_solvers(problem)

    with pytest.raises(NotImplementedError, match="#1285"):
        NewtonMFGSolver(problem, hjb_solver, fp_solver)


# ---------------------------------------------------------------------------
# Plain problem (no extended fields) must NOT raise
# ---------------------------------------------------------------------------


def test_mfg_residual_plain_problem_does_not_raise():
    """MFGResidual must construct cleanly for a plain problem with no extended fields."""
    from mfgarchon.alg.numerical.coupling.mfg_residual import MFGResidual

    problem = _make_plain_problem()
    hjb_solver, fp_solver = _make_solvers(problem)

    residual = MFGResidual(problem, hjb_solver, fp_solver)
    # Smoke-check: basic attributes are set
    assert residual.problem is problem
    assert residual.num_time_steps == problem.Nt + 1


def test_newton_solver_plain_problem_does_not_raise():
    """NewtonMFGSolver must construct cleanly for a plain problem."""
    from mfgarchon.alg.numerical.coupling.newton_mfg_solver import NewtonMFGSolver

    problem = _make_plain_problem()
    hjb_solver, fp_solver = _make_solvers(problem)

    solver = NewtonMFGSolver(problem, hjb_solver, fp_solver)
    assert solver.problem is problem

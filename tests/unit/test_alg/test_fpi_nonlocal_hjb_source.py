"""Issue #1259: FixedPointIterator._compose_hjb_source silently dropped nonlocal_operator.

The `has_nonlocal` flag was computed but never used inside the `composed`
closure.  This test builds a tiny problem with a known LinearOperator
nonlocal_operator and asserts that the composed HJB source includes the J[v]
contribution.  It fails on pre-fix code (source omits nonlocal term) and
passes after the fix.

2026-06-10 audit.
"""

from __future__ import annotations

import numpy as np

from mfgarchon.alg.numerical.coupling import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem


def _make_problem_with_nonlocal(Nx: int = 5, Nt: int = 3) -> MFGProblem:
    """Minimal 1-D MFG problem with a diagonal nonlocal_operator."""
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )
    components = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )
    # nonlocal_operator: 2*I (applied to v_t it returns 2*v_t)
    nonlocal_op = 2.0 * np.eye(Nx)
    return MFGProblem(
        Nx=Nx,
        xmin=0.0,
        xmax=1.0,
        T=0.3,
        Nt=Nt,
        sigma=0.2,
        components=components,
        nonlocal_operator=nonlocal_op,
    )


def test_nonlocal_term_present_in_hjb_source():
    """Composed HJB source must include nonlocal_operator @ v_t (Issue #1259).

    Setup: nonlocal_operator = 2*I, so J[v_t] = 2*v_t.
    For t=0 and a known u_current, the source must equal 2 * v_t[0, :].
    Pre-fix: source returns zeros (nonlocal branch absent).
    Post-fix: source returns 2 * v_t.
    """
    Nx = 5
    Nt = 3
    problem = _make_problem_with_nonlocal(Nx=Nx, Nt=Nt)
    hjb_solver = HJBFDMSolver(problem)
    fp_solver = FPFDMSolver(problem)
    iterator = FixedPointIterator(problem, hjb_solver, fp_solver)

    # Build m_current (flat spatial) and u_current (Nt+1, Nx)
    m_current = np.ones(Nx) / Nx
    rng = np.random.default_rng(42)
    u_current = rng.standard_normal((Nt + 1, Nx))

    # The composed source at t=0 with x_grid irrelevant to the nonlocal term
    x_grid = np.linspace(0.0, 1.0, Nx).reshape(-1, 1)
    source_fn = iterator._compose_hjb_source(m_current, u_current)

    assert source_fn is not None, "source_fn must not be None when nonlocal_operator is set"

    result = source_fn(t=0.0, x=x_grid)

    # Expected: nonlocal_operator @ v_t(0) = 2*I @ u_current[0] = 2*u_current[0]
    expected_nonlocal = 2.0 * u_current[0]

    np.testing.assert_allclose(
        result,
        expected_nonlocal,
        rtol=1e-12,
        err_msg=(
            "HJB source did not include nonlocal_operator @ v_t. Pre-fix code returns zeros; post-fix returns 2*v_t."
        ),
    )


def test_nonlocal_term_present_at_interior_time():
    """Nonlocal term is correctly extracted at an interior time step.

    Checks that _get_time_slice selects the right row of u_current.
    """
    Nx = 5
    Nt = 4
    problem = _make_problem_with_nonlocal(Nx=Nx, Nt=Nt)
    hjb_solver = HJBFDMSolver(problem)
    fp_solver = FPFDMSolver(problem)
    iterator = FixedPointIterator(problem, hjb_solver, fp_solver)

    m_current = np.ones(Nx) / Nx
    rng = np.random.default_rng(7)
    u_current = rng.standard_normal((Nt + 1, Nx))

    x_grid = np.linspace(0.0, 1.0, Nx).reshape(-1, 1)
    source_fn = iterator._compose_hjb_source(m_current, u_current)

    # t = T/2 -> time index n = round(t / dt) = round(Nt/2)
    dt = problem.dt
    t_half = dt * (Nt // 2)
    n_half = Nt // 2

    result = source_fn(t=t_half, x=x_grid)
    expected_nonlocal = 2.0 * u_current[n_half]

    np.testing.assert_allclose(
        result,
        expected_nonlocal,
        rtol=1e-12,
        err_msg=(
            f"At t={t_half} (index {n_half}), expected 2*u_current[{n_half}]. nonlocal time-slice extraction failed."
        ),
    )

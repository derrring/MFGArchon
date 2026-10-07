"""FP-FDM Dirichlet rows hold their value under a source, and the adjoint step refuses what it cannot hold (#2531).

A Dirichlet row reads m = g, a constraint rather than a balance. The source was added to every row after
the Dirichlet values, so the wall held g + dt*S: under `problem.solve(scheme="fdm_upwind")` with a shared
exit (absorbing since #2528) and a problem-level `source_term_fp`, the exit held dt*S -- 0.025 for S = 1 at
dt = 0.025 -- where it must hold 0. `solve_fp_step_adjoint_mode` solves I/dt + A^T - D with right-hand
side m/dt and has no Dirichlet row at all: a uniform Dirichlet wall held neither g nor 0, silently. It is
refused for a Dirichlet wall.

The fixtures put the Dirichlet wall on x_max only, with no-flux on x_min.
"""

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, dirichlet_bc, no_flux_bc

N, NT, T, SIGMA = 21, 10, 0.5, 0.4
DT = T / NT


def _wall_on_x_max(g: float = 0.7) -> BoundaryConditions:
    return BoundaryConditions(
        dimension=1,
        segments=[
            BCSegment(name="wall", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, boundary="x_max", value=g),
        ],
    )


def _problem(bc, nt: int = NT, **problem_kw) -> MFGProblem:
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[N], boundary_conditions=bc)
    x = np.linspace(0.0, 1.0, N)
    bump = lambda z: np.exp(-20 * (np.asarray(z, dtype=float) - 0.6) ** 2)  # noqa: E731
    scale = 1.0 / np.trapezoid(bump(x), x)
    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=SIGMA
        ),
        domain=grid,
        conditions=Conditions(
            m_initial=lambda z: scale * bump(z), u_terminal=lambda z: 0.0 * np.asarray(z, dtype=float), T=T
        ),
        Nt=nt,
        **problem_kw,
    )


def _constant(value: float):
    return lambda t, x: np.full(np.asarray(x).shape[0], value)


def _solve(solver, source):
    x = np.linspace(0.0, 1.0, N)
    return np.asarray(
        solver.solve_fp_system(np.exp(-20 * (x - 0.6) ** 2), drift_field=np.zeros((NT + 1, N)), source_term=source)
    )


def test_a_shared_exit_holds_zero_under_a_source():
    """Main held it at dt*S = 0.05 for S = 1 at dt = 0.05."""
    M = _solve(FPFDMSolver(_problem(_wall_on_x_max())), _constant(1.0))
    assert np.all(M[1:, -1] == 0.0)


def test_an_explicit_dirichlet_holds_g_under_a_source():
    """The FP's own DIRICHLET(0.7) is a prescribed density; main held it at 0.7 + dt*S = 0.75."""
    solver = FPFDMSolver(_problem(no_flux_bc(dimension=1)), boundary_conditions=_wall_on_x_max(0.7))
    M = _solve(solver, _constant(1.0))
    np.testing.assert_allclose(M[1:, -1], 0.7, rtol=0, atol=1e-12)


def test_the_source_still_enters_every_balance_row():
    """Moving the Dirichlet rows after the source must not drop it elsewhere: on no-flux walls the
    divergence form conserves, so the mass grows by exactly dt*S*|domain| per step."""
    grid_problem = _problem(no_flux_bc(dimension=1))
    M = _solve(FPFDMSolver(grid_problem), _constant(1.0))
    mass = np.array([grid_problem.geometry.integrate(m) for m in M])
    np.testing.assert_allclose(np.diff(mass), DT * 1.0, rtol=1e-10)


def test_problem_solve_holds_a_shared_exit_at_zero_under_a_problem_source():
    """The route #2531 was measured on: plain problem.solve, a shared exit, a problem-level source."""
    problem = _problem(_wall_on_x_max(), nt=20, source_term_fp=lambda t, x, v, m: np.full(np.asarray(m).shape, 1.0))
    result = problem.solve(scheme="fdm_upwind", max_iterations=40, tolerance=1e-6)
    M = np.asarray(result.M)
    # Picard damps each iterate toward the initial guess, which is nonzero at the exit; read before the fix
    # the exit held 0.025 here.
    assert np.abs(M[1:, -1]).max() < 1e-6


@pytest.mark.parametrize(
    "bc",
    [dirichlet_bc(dimension=1, value=0.0), dirichlet_bc(dimension=1, value=0.7), _wall_on_x_max(0.7)],
    ids=["uniform_absorbing", "uniform_prescribed", "mixed"],
)
def test_the_adjoint_step_refuses_a_dirichlet_wall(bc):
    """It has no Dirichlet row. On main, from cos(pi x / 2), a uniform Dirichlet left the walls at
    0.5338 / 0.0359 after one step for g = 0 and g = 0.7 alike -- neither absorbing nor prescribed, and
    silent; a mixed one raised a ValueError from LaplacianOperator."""
    from scipy import sparse

    solver = FPFDMSolver(_problem(no_flux_bc(dimension=1)), boundary_conditions=bc)
    with pytest.raises(NotImplementedError, match="#2531"):
        solver.solve_fp_step_adjoint_mode(np.ones(N), sparse.csr_matrix((N, N)))


def test_the_adjoint_step_still_runs_on_no_flux():
    from scipy import sparse

    solver = FPFDMSolver(_problem(no_flux_bc(dimension=1)))
    M_next = solver.solve_fp_step_adjoint_mode(np.ones(N), sparse.csr_matrix((N, N)))
    assert np.all(np.isfinite(M_next))

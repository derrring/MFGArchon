"""HJB-GFDM's Newton step imposes a Dirichlet value, and returns a root of the system it solves (#2599).

The step tested convergence, and measured its Armijo line search, on the PDE residual alone. The system it solves
replaces each boundary row with the BC's, u - g at a Dirichlet point. Two failures followed:
- **A spatially constant iterate with no source has a zero PDE residual.** The step stopped at iteration 0 and
  returned the terminal unchanged, imposing no Dirichlet value.
- **A trial's walls are projected to g, so the merit compared a different function across each step.** The search
  backtracked to its floor, and every step ran all 30 iterations without converging, silently.

Measured at 52219be2 (main), on the issue's 1-D fixture:

| terminal | U(0) at x_min, x_max | final residual, boundary rows included |
|---|---|---|
| 0 | 0, 0 | not formed |
| 0.1 | 0.1, 0.1 | not formed |
| 0.1 + 0.05 x | 0.7, 0.4 | 3e-2 to 7e-3 |

In 2-D (6 x 6, g = x + 2y), the max |U - g| on the boundary was 2.9 under Newton with a constant terminal, and 0
under Howard. Both now read the residual with its boundary rows, and every case here imposes g.

Convergence is not uniform. The constant and linear 1-D cases converge at every step, to below 1e-6, and a step
that does not now says so. Others still do not converge, and are now logged:
- the zero terminal, where the walls jump from 0 to 0.7 and 0.4: 8 of 10 steps, final residual about 4;
- the 2-D Newton cases: 10 of 10 steps, about 29.
On 52219be2 every step of each of these ended unconverged too, in silence: 0.81 for the zero terminal, and 7.5 and
25 in 2-D, where the walls were not imposed. So the root test below covers the constant and linear cases only.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, dirichlet_bc

L, T, NX, NT = 2.0, 1.0, 21, 10
X = np.linspace(0.0, L, NX)
TERMINALS = {"zero": np.zeros(NX), "constant": np.full(NX, 0.1), "linear": 0.1 + 0.05 * X}
HOWARD = {"inner_solver": "howard", "monotonicity_scheme": "qp_m_matrix", "monotonicity_application": "always"}


def _problem(grid):
    comps = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        m_initial=lambda xx: 1.0,
        u_terminal=lambda xx: 0.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MFGProblem(geometry=grid, components=comps, T=T, Nt=NT, volatility=0.5)


def _solve_1d(terminal, **solver_kwargs):
    bc = BoundaryConditions(
        dimension=1,
        segments=[
            BCSegment(name="a", bc_type=BCType.DIRICHLET, value=0.7, boundary="x_min"),
            BCSegment(name="b", bc_type=BCType.DIRICHLET, value=0.4, boundary="x_max"),
        ],
    )
    grid = TensorProductGrid(bounds=[(0.0, L)], Nx_points=[NX], boundary_conditions=bc)
    solver = HJBGFDMSolver(
        _problem(grid),
        collocation_points=X.reshape(-1, 1),
        delta=3.0 * L / (NX - 1),
        monotonicity_scheme="none",
        **solver_kwargs,
    )
    U_T = TERMINALS[terminal]
    M = np.ones((NT + 1, NX)) / L
    return (
        solver,
        M,
        np.asarray(solver.solve_hjb_system(M_density=M, U_terminal=U_T, U_coupling_prev=np.tile(U_T, (NT + 1, 1)))),
    )


@pytest.mark.parametrize("terminal", sorted(TERMINALS))
def test_newton_imposes_the_dirichlet_value_whatever_the_terminal(terminal):
    _, _, U = _solve_1d(terminal)
    np.testing.assert_allclose(U[:NT, 0], 0.7, rtol=0, atol=1e-12)
    np.testing.assert_allclose(U[:NT, -1], 0.4, rtol=0, atol=1e-12)


@pytest.mark.parametrize("terminal", ["constant", "linear"])
def test_each_level_is_a_root_of_the_system_its_step_solves(terminal):
    """The residual with its boundary rows, at each level returned, is below the Newton tolerance: the step
    converged rather than running out of iterations (#2599)."""
    solver, M, U = _solve_1d(terminal)
    H = solver.problem.hamiltonian_class
    for n in range(NT):
        grad, lap = solver._compute_derivatives_vectorized(U[n])
        pde = solver._compute_hjb_residual_hamiltonian(U[n], U[n + 1], M[n + 1], grad, lap, H, n * T / NT)
        _, residual = solver._apply_boundary_conditions_to_sparse_system(None, pde, n, U[n])
        assert np.linalg.norm(residual) < solver.newton_tolerance, (n, np.linalg.norm(residual))


@pytest.mark.parametrize("iterations", [1, None], ids=["one iteration", "the default"])
def test_newton_says_when_it_runs_out_of_iterations(iterations, caplog):
    """It returned its last iterate in silence, which hid #2599. The constant case needs three iterations a step: given
    one, it warns; given the default, it does not."""
    kwargs = {} if iterations is None else {"max_newton_iterations": iterations}
    with caplog.at_level("WARNING", logger="mfgarchon"):
        _solve_1d("constant", **kwargs)
    warned = [r for r in caplog.records if "HJB-GFDM Newton did not converge" in r.getMessage()]
    assert bool(warned) == (iterations == 1), [r.getMessage() for r in warned][:2]


@pytest.mark.parametrize("inner", ["newton", "howard"])
@pytest.mark.parametrize("terminal", ["constant", "linear"])
def test_both_inner_solvers_impose_it_in_2d(inner, terminal):
    n = 6
    xs = np.linspace(0.0, 1.0, n)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[n, n], boundary_conditions=dirichlet_bc(dimension=2, value=0.0)
    )
    kwargs = HOWARD if inner == "howard" else {"monotonicity_scheme": "none"}
    solver = HJBGFDMSolver(
        _problem(grid),
        collocation_points=pts,
        delta=0.5,
        boundary_conditions=dirichlet_bc(dimension=2, value=lambda x: x[0] + 2 * x[1]),
        **kwargs,
    )
    U_T = np.full(n * n, 0.1) if terminal == "constant" else 0.1 + 0.05 * pts[:, 0]
    U = np.asarray(
        solver.solve_hjb_system(
            M_density=np.ones((NT + 1, n * n)), U_terminal=U_T, U_coupling_prev=np.tile(U_T, (NT + 1, 1))
        )
    )
    walls = solver.boundary_indices
    g = pts[walls, 0] + 2 * pts[walls, 1]
    assert np.abs(U[:NT][:, walls] - g).max() < 1e-12

"""1-D HJB-FDM on an exact solution with a nonzero Dirichlet wall (Issue #2515, #2512 (a)).

`dirichlet_bc` is declared by `HJBFDMSolver` and is reached by `problem.solve()` on a grid, and no
test compared a Dirichlet solve against an exact solution. The backward heat mode

    u(t, x) = g + exp(-D pi^2 (T - t)) sin(pi x),   D = sigma^2 / 2,

solves -u_t - D u_xx = 0 with u = g on both walls, so with H(p) = |p|^2/2 the source S = H(u_x)
makes it exact for the full HJB, as in `test_mms_validation.py`'s periodic case.

What this pins and what it does not. The solution was already right to first order before #2515:
2.58e-2 / 1.31e-2 / 6.56e-3 at Nx = 41 / 81 / 161 at `93c490ba`, and the same to three digits after.
#2515 fixed the `converged` claim, which `test_hjb_bc_enforcement_1900.py`'s law pins. This test pins
the boundary VALUE reaching the solution: dropping g from the wall row (u_wall = 0) gives an
O(1) error that does not fall with h.
"""

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import dirichlet_bc

SIGMA = 0.2
T = 0.3
NT = 20
G = 0.3
D = SIGMA**2 / 2
K = np.pi


def _u(t, x):
    return G + np.exp(-D * K**2 * (T - t)) * np.sin(K * x)


def _source(t, x):
    u_x = np.exp(-D * K**2 * (T - t)) * K * np.cos(K * np.asarray(x, dtype=float))
    return 0.5 * u_x**2


def _error(nx: int) -> tuple[float, np.ndarray]:
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=dirichlet_bc(dimension=1, value=G)
    )
    comps = MFGComponents(
        m_initial=lambda x: np.ones_like(np.asarray(x, dtype=float)),
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
    )
    problem = MFGProblem(geometry=grid, T=T, Nt=NT, volatility=SIGMA, components=comps)
    x = grid.coordinates[0]
    U = HJBFDMSolver(problem).solve_hjb_system(
        M_density=np.ones((NT + 1, nx)),
        U_terminal=_u(T, x),
        U_coupling_prev=np.zeros((NT + 1, nx)),
        source_term=_source,
    )
    U0 = np.asarray(U)[0]
    return float(np.abs(U0 - _u(0.0, x)).max()), U0


def test_a_dirichlet_solve_converges_to_the_exact_solution_at_first_order():
    errors = []
    for nx in (41, 81, 161):
        err, U0 = _error(nx)
        assert U0[0] == G, f"left wall {U0[0]} against g = {G}"
        assert U0[-1] == G, f"right wall {U0[-1]} against g = {G}"
        errors.append(err)
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    assert np.all((ratios > 1.6) & (ratios < 2.6)), (
        f"error ratios {ratios} (errors {errors}); first order gives about 2"
    )

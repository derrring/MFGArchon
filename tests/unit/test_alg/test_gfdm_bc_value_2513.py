"""HJB-GFDM honours the value of a Neumann or Dirichlet BC, however it is spelt (Issue #2513).

The uniform path handed the row builders no segment, so the target fell back to a `values`
attribute `BoundaryConditions` does not have: `neumann_bc(value=g)` solved as g = 0 and
`dirichlet_bc(value=g)` pinned 0. The post-step Dirichlet overwrite read the same missing value
and wrote NaN (uniform) or 0.0 (per face) over what the row had just imposed.

External oracle, the 1-D setting of `test_gfdm_mms_source_1991.py` (L = 20, T = 4, sigma = 1,
`a1` linear in t so backward Euler carries no temporal error), with g = 0.7:

- Neumann: u = a1(t) cos(2 pi x / L) + (g / L)(x - L/2)^2, whose outward derivative is g on both walls;
- Dirichlet: u = g + a1(t) sin(pi x / L).

Measured at `ca72355b`, max error at t = 0, nx 21 -> 41: uniform Neumann 5.0e-1 -> 7.6e-1 (Newton)
and 5.5e-1 -> 5.6e-1 (Howard); uniform Dirichlet NaN (Newton) and 7.0e-1 flat (Howard); per-face
Dirichlet 7.0e-1 flat (Newton). Per-face Neumann and per-face Howard Dirichlet already gave EOC 2.
After the fix every case gives EOC 2.00 (Neumann) or 2.11 (Dirichlet), and the two spellings agree
bit for bit.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, dirichlet_bc, neumann_bc

L = 20.0
T = 4.0
SIGMA = 1.0
G = 0.7
C = 2.0 * np.pi / L
K = np.pi / L
Q = G / L


def _a1(t):
    return 1.0 + (T - t) / (2.0 * T)


_DA1 = -1.0 / (2.0 * T)

# (u, (u_t, u_x, u_xx)) for each BC kind.
_EXACT = {
    "neumann": (
        lambda t, x: _a1(t) * np.cos(C * x) + Q * (x - L / 2) ** 2,
        lambda t, x: (
            _DA1 * np.cos(C * x),
            -_a1(t) * C * np.sin(C * x) + 2 * Q * (x - L / 2),
            -_a1(t) * C**2 * np.cos(C * x) + 2 * Q,
        ),
    ),
    "dirichlet": (
        lambda t, x: G + _a1(t) * np.sin(K * x),
        lambda t, x: (_DA1 * np.sin(K * x), _a1(t) * K * np.cos(K * x), -_a1(t) * K**2 * np.sin(K * x)),
    ),
}

_HOWARD = {"inner_solver": "howard", "monotonicity_scheme": "qp_m_matrix", "monotonicity_application": "always"}


def _bc(kind: str, per_face: bool) -> BoundaryConditions:
    bc_type = BCType.NEUMANN if kind == "neumann" else BCType.DIRICHLET
    if per_face:
        return BoundaryConditions(
            dimension=1,
            segments=[BCSegment(name=f, bc_type=bc_type, value=G, boundary=f) for f in ("x_min", "x_max")],
        )
    return neumann_bc(dimension=1, value=G) if kind == "neumann" else dirichlet_bc(dimension=1, value=G)


def _solve(kind: str, per_face: bool, nx: int, nt: int = 20, **solver_kw) -> tuple[np.ndarray, np.ndarray]:
    u, derivs = _EXACT[kind]
    x = np.linspace(0.0, L, nx)

    def source(t, xx):
        xx = np.asarray(xx, dtype=float).reshape(-1)
        u_t, u_x, u_xx = derivs(t, xx)
        return -u_t - 0.5 * SIGMA**2 * u_xx + 0.5 * u_x**2

    grid = TensorProductGrid(bounds=[(0.0, L)], Nx_points=[nx], boundary_conditions=_bc(kind, per_face))
    comps = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        m_initial=lambda xx: np.ones_like(np.asarray(xx, dtype=float)) / L,
        u_terminal=lambda xx: u(T, np.asarray(xx, dtype=float)),
    )
    problem = MFGProblem(geometry=grid, components=comps, T=T, Nt=nt, volatility=SIGMA)
    solver = HJBGFDMSolver(problem, collocation_points=x.reshape(-1, 1), delta=3.0 * L / (nx - 1), **solver_kw)
    u_T = u(T, x)
    U = solver.solve_hjb_system(
        M_density=np.ones((nt + 1, nx)) / L,
        U_terminal=u_T,
        U_coupling_prev=np.tile(u_T, (nt + 1, 1)),
        source_term=source,
    )
    return np.asarray(U)[0].reshape(-1), u(0.0, x)


@pytest.mark.parametrize("inner", ["newton", "howard"])
@pytest.mark.parametrize("per_face", [False, True], ids=["uniform", "per_face"])
@pytest.mark.parametrize("kind", ["neumann", "dirichlet"])
def test_the_bc_value_reaches_the_solution_at_second_order(kind: str, per_face: bool, inner: str):
    kw = _HOWARD if inner == "howard" else {"monotonicity_scheme": "none"}
    errors = []
    for nx in (21, 41):
        U0, u0 = _solve(kind, per_face, nx, **kw)
        errors.append(float(np.abs(U0 - u0).max()))
    order = np.log2(errors[0] / errors[1])
    assert 1.7 < order < 2.4, (
        f"{kind}/{'per-face' if per_face else 'uniform'}/{inner}: EOC {order:.2f}, errors {errors}"
    )


@pytest.mark.parametrize("kind", ["neumann", "dirichlet"])
def test_the_two_spellings_of_one_bc_solve_identically(kind: str):
    """A uniform BC and the same value on every named face are one condition."""
    uniform, _ = _solve(kind, per_face=False, nx=21, monotonicity_scheme="none")
    per_face, _ = _solve(kind, per_face=True, nx=21, monotonicity_scheme="none")
    np.testing.assert_array_equal(uniform, per_face)

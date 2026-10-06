"""HJB-GFDM honours the value of a Neumann or Dirichlet BC, however it is spelt (Issue #2513).

The uniform path handed the row builders no segment, so the target fell back to a `values`
attribute `BoundaryConditions` does not have: `neumann_bc(value=g)` solved as g = 0 and
`dirichlet_bc(value=g)` pinned 0. The post-step Dirichlet overwrite read the same missing value
and wrote NaN (uniform) or 0.0 (per face) over what the row had just imposed.

External oracle, the 1-D setting of `test_gfdm_mms_source_1991.py` (L = 20, T = 4, sigma = 1,
`a1` linear in t so backward Euler carries no temporal error), with values gL at x = 0 and gR at
x = L -- equal for a uniform BC, 0.7 and 0.2 per face, so a face mix-up cannot pass:

- Neumann: u = a1(t) cos(2 pi x / L) + q (x - L/2)^2 + b x, q = (gL + gR) / 2L, b = (gR - gL) / 2,
  whose outward derivative is gL on the left wall and gR on the right;
- Dirichlet: u = gL + (gR - gL) x / L + a1(t) sin(pi x / L).

Measured at `ca72355b` with gL = gR = 0.7, max error at t = 0, nx 21 -> 41: uniform Neumann
5.0e-1 -> 7.6e-1 (Newton) and 5.5e-1 -> 5.6e-1 (Howard); uniform Dirichlet NaN (Newton) and 7.0e-1
flat (Howard); per-face Dirichlet 7.0e-1 flat (Newton). After the fix every case converges at
second order on the default (no ghost-node) path.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    dirichlet_bc,
    neumann_bc,
    robin_bc,
)

L = 20.0
T = 4.0
SIGMA = 1.0
G = 0.7
G_RIGHT = 0.2
C = 2.0 * np.pi / L
K = np.pi / L
DA1 = -1.0 / (2.0 * T)

_HOWARD = {"inner_solver": "howard", "monotonicity_scheme": "qp_m_matrix", "monotonicity_application": "always"}
_NEWTON = {"monotonicity_scheme": "none"}


def _a1(t):
    return 1.0 + (T - t) / (2.0 * T)


def _exact(kind: str, g_left: float, g_right: float):
    """(u, (u_t, u_x, u_xx)) with the given wall data."""
    if kind == "neumann":
        q, b = (g_left + g_right) / (2 * L), (g_right - g_left) / 2
        return (
            lambda t, x: _a1(t) * np.cos(C * x) + q * (x - L / 2) ** 2 + b * x,
            lambda t, x: (
                DA1 * np.cos(C * x),
                -_a1(t) * C * np.sin(C * x) + 2 * q * (x - L / 2) + b,
                -_a1(t) * C**2 * np.cos(C * x) + 2 * q,
            ),
        )
    slope = (g_right - g_left) / L
    return (
        lambda t, x: g_left + slope * x + _a1(t) * np.sin(K * x),
        lambda t, x: (DA1 * np.sin(K * x), slope + _a1(t) * K * np.cos(K * x), -_a1(t) * K**2 * np.sin(K * x)),
    )


def _per_face(bc_type: BCType, g_left: float, g_right: float) -> BoundaryConditions:
    return BoundaryConditions(
        dimension=1,
        segments=[
            BCSegment(name="x_min", bc_type=bc_type, value=g_left, boundary="x_min"),
            BCSegment(name="x_max", bc_type=bc_type, value=g_right, boundary="x_max"),
        ],
    )


def _solve(kind: str, bc, g_left: float, g_right: float, nx: int, nt: int = 20, **solver_kw):
    u, derivs = _exact(kind, g_left, g_right)
    x = np.linspace(0.0, L, nx)

    def source(t, xx):
        xx = np.asarray(xx, dtype=float).reshape(-1)
        u_t, u_x, u_xx = derivs(t, xx)
        return -u_t - 0.5 * SIGMA**2 * u_xx + 0.5 * u_x**2

    grid = TensorProductGrid(bounds=[(0.0, L)], Nx_points=[nx], boundary_conditions=bc)
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


def _cases(kind: str, per_face: bool):
    bc_type = BCType.NEUMANN if kind == "neumann" else BCType.DIRICHLET
    if per_face:
        return _per_face(bc_type, G, G_RIGHT), G, G_RIGHT
    uniform = neumann_bc(dimension=1, value=G) if kind == "neumann" else dirichlet_bc(dimension=1, value=G)
    return uniform, G, G


@pytest.mark.parametrize("inner", ["newton", "howard"])
@pytest.mark.parametrize("per_face", [False, True], ids=["uniform", "per_face"])
@pytest.mark.parametrize("kind", ["neumann", "dirichlet"])
def test_the_bc_value_reaches_the_solution_at_second_order(kind: str, per_face: bool, inner: str):
    kw = _HOWARD if inner == "howard" else _NEWTON
    bc, g_left, g_right = _cases(kind, per_face)
    errors = []
    for nx in (21, 41):
        U0, u0 = _solve(kind, bc, g_left, g_right, nx, **kw)
        errors.append(float(np.abs(U0 - u0).max()))
    order = np.log2(errors[0] / errors[1])
    assert 1.7 < order < 2.4, (
        f"{kind}/{'per-face' if per_face else 'uniform'}/{inner}: EOC {order:.2f}, errors {errors}"
    )


@pytest.mark.parametrize("kind", ["neumann", "dirichlet"])
def test_the_two_spellings_of_one_bc_solve_identically(kind: str):
    """A uniform BC and the same value on every named face are one condition."""
    bc_type = BCType.NEUMANN if kind == "neumann" else BCType.DIRICHLET
    uniform, _ = _solve(kind, _cases(kind, per_face=False)[0], G, G, nx=21, **_NEWTON)
    per_face, _ = _solve(kind, _per_face(bc_type, G, G), G, G, nx=21, **_NEWTON)
    np.testing.assert_array_equal(uniform, per_face)


@pytest.mark.parametrize("inner", ["newton", "howard"])
def test_a_uniform_robin_with_no_value_term_is_the_neumann_condition(inner: str):
    """Robin(alpha=0, beta=1, g) is du/dn = g. A uniform one used to reach the row with no segment,
    so alpha defaulted to 1 and it was refused; it now reads its own coefficients."""
    kw = _HOWARD if inner == "howard" else _NEWTON
    robin, _ = _solve("neumann", robin_bc(dimension=1, alpha=0.0, beta=1.0, value=G), G, G, nx=21, **kw)
    neumann, _ = _solve("neumann", neumann_bc(dimension=1, value=G), G, G, nx=21, **kw)
    np.testing.assert_array_equal(robin, neumann)


def test_a_bc_value_no_row_reads_is_refused_not_solved_as_zero():
    grid = TensorProductGrid(
        bounds=[(0.0, L)], Nx_points=[11], boundary_conditions=dirichlet_bc(dimension=1, value=0.0)
    )
    comps = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        m_initial=lambda xx: np.ones_like(np.asarray(xx, dtype=float)) / L,
        u_terminal=lambda xx: np.zeros_like(np.asarray(xx, dtype=float)),
    )
    problem = MFGProblem(geometry=grid, components=comps, T=T, Nt=4, volatility=SIGMA)
    x = np.linspace(0.0, L, 11).reshape(-1, 1)
    refused = ({"value": 0.3}, {"values": 0.2, "value": 0.3})
    accepted = ({"value": 0.0}, {"values": 0.3}, {"values": 0.3, "value": 0.3})
    for extra in refused:
        with pytest.raises(NotImplementedError, match="#2513"):
            HJBGFDMSolver(problem, collocation_points=x, boundary_conditions={"type": "dirichlet", **extra}, **_NEWTON)
    for extra in accepted:
        HJBGFDMSolver(problem, collocation_points=x, boundary_conditions={"type": "dirichlet", **extra}, **_NEWTON)

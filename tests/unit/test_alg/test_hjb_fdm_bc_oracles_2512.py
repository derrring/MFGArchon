"""HJB-FDM against exact solutions, on every boundary condition it declares, in 1-D and 2-D (#2512 (a)).

One cell per BC type and dimension, and in 2-D one per path of the ghost buffer: a uniform BC takes
`_apply_linear_reflection`, a BC with a segment per face `_apply_ghost_for_face`.

**Each fixture is chosen so that rival wall treatments give different numbers.**
- In the per-face cells, data that carry a value differ at the two walls of an axis. A uniform BC carries one
  value on every wall, so its cells rely on the solution instead.
- Where the data carry none (no-flux, periodic), the solution is not even about the midline, nor about the
  seam.
- The optimal drift ``-grad u`` points into every Neumann wall, which is what carries a flux datum into the
  solution. With it pointing away, swapping a Neumann wall's data left the error unchanged to the printed
  digit (#2512, comment 6042778010). Dirichlet walls need no drift condition: the wall value is asserted
  exactly.

**Each cell asserts more than a convergence rate.** Wall closures that are wrong at O(h) keep the rate and
move the level -- the cell-centred no-flux mirror (#1935) LOWERS it -- so each cell asserts:
- every inner step converged;
- the error ratio under refinement;
- each level's error, two-sided, within a band around the value measured when this file was written;
- the wall itself: Dirichlet nodes equal ``g``, and a Neumann or no-flux wall's slope, read off the returned
  field to second order, matches its datum.

A level outside its band in either direction is a change in the wall closure. Re-measure and record it, as
for the discrimination ratchet.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    dirichlet_bc,
    neumann_bc,
    no_flux_bc,
    periodic_bc,
)
from mfgarchon.geometry.boundary.types import PeriodicGridConvention

if TYPE_CHECKING:
    from collections.abc import Callable

P = np.pi


# --------------------------------------------------------------------------------------------- exact solutions
# u(t, x) = sum_i a_i theta_i(T - t) prod_d f_i^d(x_d), theta(tau) in {1, tau, exp(-lam tau)},
# f in {1, z, z^2, sin(k z + c), cos(k z + c)}. With H = |p|^2 / 2 the source that makes u exact for
# -u_t + H(grad u) - D lap u = S is S = -u_t + |grad u|^2 / 2 - D lap u.


def _factor(name: str, k: float = 0.0, c: float = 0.0):
    """(f, f', f'') of one spatial factor."""
    if name == "one":
        return (lambda z: np.ones_like(z), lambda z: np.zeros_like(z), lambda z: np.zeros_like(z))
    if name == "lin":
        return (lambda z: z, lambda z: np.ones_like(z), lambda z: np.zeros_like(z))
    if name == "sq":
        return (lambda z: z**2, lambda z: 2 * z, lambda z: 2 * np.ones_like(z))
    if name == "sin":
        return (lambda z: np.sin(k * z + c), lambda z: k * np.cos(k * z + c), lambda z: -(k**2) * np.sin(k * z + c))
    if name == "cos":
        return (lambda z: np.cos(k * z + c), lambda z: -k * np.sin(k * z + c), lambda z: -(k**2) * np.cos(k * z + c))
    raise ValueError(name)


def _theta(kind: str, lam: float, tau: float) -> tuple[float, float]:
    """(theta, d theta / dt) with tau = T - t."""
    if kind == "const":
        return 1.0, 0.0
    if kind == "tau":
        return tau, -1.0
    if kind == "exp":
        return float(np.exp(-lam * tau)), lam * float(np.exp(-lam * tau))
    raise ValueError(kind)


@dataclass
class Exact:
    T: float
    D: float
    terms: list  # (a, (kind, lam), [(name, k, c) per axis])

    def parts(self, t: float, X: list[np.ndarray]):
        tau = self.T - t
        u = np.zeros_like(X[0], dtype=float)
        ut = np.zeros_like(u)
        grad = [np.zeros_like(u) for _ in X]
        lap = np.zeros_like(u)
        for a, (kind, lam), factors in self.terms:
            th, dth = _theta(kind, lam, tau)
            fs = [_factor(*spec) for spec in factors]
            vals = [f[0](X[i]) for i, f in enumerate(fs)]
            whole = np.prod(vals, axis=0)
            u += a * th * whole
            ut += a * dth * whole
            for i, f in enumerate(fs):
                others = np.prod([vals[j] for j in range(len(X)) if j != i], axis=0) if len(X) > 1 else 1.0
                grad[i] += a * th * f[1](X[i]) * others
                lap += a * th * f[2](X[i]) * others
        return u, ut, grad, lap

    def u(self, t: float, X: list[np.ndarray]) -> np.ndarray:
        return self.parts(t, X)[0]

    def source(self, t: float, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1, len(self.terms[0][2]))
        _, ut, grad, lap = self.parts(t, [x[:, i] for i in range(x.shape[1])])
        return -ut + 0.5 * sum(g**2 for g in grad) - self.D * lap


# ---------------------------------------------------------------------------------------------------- cells


@dataclass
class Cell:
    bc: Callable[[], BoundaryConditions]
    exact: Exact
    sigma: float
    levels: tuple[int, ...]
    nts: tuple[int, ...]
    values: dict = field(default_factory=dict)  # Dirichlet face -> g
    slopes: dict = field(default_factory=dict)  # Neumann / no-flux face -> outward du/dn
    seam: bool = False  # periodic: the duplicated endpoints must come back equal

    @property
    def dim(self) -> int:
        return len(self.exact.terms[0][2])


def _faces(dim: int, spec: dict) -> BoundaryConditions:
    return BoundaryConditions(
        dimension=dim, segments=[BCSegment(name=f, bc_type=t, boundary=f, value=v) for f, (t, v) in spec.items()]
    )


ONE, CONST, TAU = ("one",), ("const", 0.0), ("tau", 0.0)

# 1-D: sigma = 0.2, T = 0.3. The heat modes solve -u_t - D u_xx = 0 exactly.
T1, S1 = 0.3, 0.2
D1 = S1**2 / 2
_g = {"x_min": 0.3, "x_max": 0.8}  # Dirichlet
_n = {"x_min": -0.4, "x_max": -1.0}  # outward Neumann slopes: u_x(0) = 0.4, u_x(1) = -1.0, drift -u_x into both walls
_alpha, _beta = -_n["x_min"], (_n["x_min"] + _n["x_max"]) / 2

# 2-D: sigma = 0.5, T = 0.3, Nt = 2. Linear in t, so implicit Euler is exact in time and the ratios are spatial.
# The domain is [0, 1] x [0, LY] with LY = 0.6 on n x n points, so dy = 0.6 dx at every level and the extent
# is not square: a square grid is a symmetry, and it hid a ghost reading the other axis's spacing
# (#2547). KY = pi / LY puts the y modes' walls and seam at y = 0 and y = LY.
T2, S2 = 0.3, 0.5
D2 = S2**2 / 2
LY = 0.6
KY = P / LY
_g2 = -0.5  # uniform outward Neumann slope; q(z) = g (z^2 / L - z) has -q'(0) = q'(L) = g on [0, L]


def _sin(k, c=0.0):
    return ("sin", k, c)


def _cos(k, c=0.0):
    return ("cos", k, c)


CELLS: dict[str, Cell] = {
    # ------------------------------------------------------------------------------------------------ 1-D
    "dirichlet_1d": Cell(
        lambda: _faces(1, {f: (BCType.DIRICHLET, g) for f, g in _g.items()}),
        Exact(
            T1,
            D1,
            [
                (_g["x_min"], CONST, [ONE]),
                (_g["x_max"] - _g["x_min"], CONST, [("lin",)]),
                (1.0, ("exp", D1 * P**2), [_sin(P)]),
                (0.5, ("exp", 4 * D1 * P**2), [_sin(2 * P)]),
            ],
        ),
        S1,
        (41, 81, 161),
        (20, 20, 20),
        values=_g,
    ),
    "neumann_1d": Cell(
        lambda: _faces(1, {f: (BCType.NEUMANN, g) for f, g in _n.items()}),
        Exact(
            T1,
            D1,
            [
                (_alpha, CONST, [("lin",)]),
                (_beta, CONST, [("sq",)]),
                (1.0, ("exp", D1 * P**2), [_cos(P)]),
                (0.5, ("exp", 4 * D1 * P**2), [_cos(2 * P)]),
            ],
        ),
        S1,
        (41, 81, 161),
        (20, 20, 20),
        slopes=_n,
    ),
    "no_flux_1d": Cell(
        # u_x = 0 at both walls, and with a1 = 1, a2 = -1/2 the drift points toward each wall near it.
        lambda: no_flux_bc(dimension=1),
        Exact(T1, D1, [(1.0, ("exp", D1 * P**2), [_cos(P)]), (-0.5, ("exp", 4 * D1 * P**2), [_cos(2 * P)])]),
        S1,
        (41, 81, 161),
        (20, 20, 20),
        slopes={"x_min": 0.0, "x_max": 0.0},
    ),
    "periodic_1d": Cell(
        # At the seam u != 0 (1/2 at t = T, decaying) and u_x != 0, so neither an odd reflection nor a mirror
        # reproduces it. Time and
        # space errors have opposite signs here, so dt ~ h.
        lambda: periodic_bc(dimension=1),
        Exact(T1, D1, [(1.0, ("exp", 4 * D1 * P**2), [_sin(2 * P)]), (0.5, ("exp", 16 * D1 * P**2), [_cos(4 * P)])]),
        S1,
        (81, 161, 321),
        (20, 40, 80),
        seam=True,
    ),
    # ------------------------------------------------------------------------------------------------ 2-D
    "dirichlet_uniform_2d": Cell(
        lambda: dirichlet_bc(dimension=2, value=0.4),
        Exact(
            T2,
            D2,
            [
                (0.4, CONST, [ONE, ONE]),
                (0.5, CONST, [_sin(P), _sin(KY)]),
                (0.25, CONST, [_sin(2 * P), _sin(KY)]),
                (0.6, TAU, [_sin(P), _sin(2 * KY)]),
            ],
        ),
        S2,
        (11, 21),
        (2, 2),
        values=dict.fromkeys(("x_min", "x_max", "y_min", "y_max"), 0.4),
    ),
    "dirichlet_faced_x_2d": Cell(
        lambda: _faces(
            2,
            {
                "x_min": (BCType.DIRICHLET, 0.3),
                "x_max": (BCType.DIRICHLET, 0.8),
                "y_min": (BCType.NO_FLUX, 0.0),
                "y_max": (BCType.NO_FLUX, 0.0),
            },
        ),
        Exact(
            T2,
            D2,
            [
                (0.3, CONST, [ONE, ONE]),
                (0.5, CONST, [("lin",), ONE]),
                (0.5, CONST, [_sin(P), _cos(KY)]),
                (0.25, CONST, [_sin(2 * P), ONE]),
                (0.6, TAU, [_sin(P), ONE]),
                (0.3, TAU, [_sin(2 * P), _cos(KY)]),
            ],
        ),
        S2,
        (11, 21),
        (2, 2),
        values={"x_min": 0.3, "x_max": 0.8},
        slopes={"y_min": 0.0, "y_max": 0.0},
    ),
    "dirichlet_faced_y_2d": Cell(
        lambda: _faces(
            2,
            {
                "x_min": (BCType.NO_FLUX, 0.0),
                "x_max": (BCType.NO_FLUX, 0.0),
                "y_min": (BCType.DIRICHLET, 0.2),
                "y_max": (BCType.DIRICHLET, 0.9),
            },
        ),
        Exact(
            T2,
            D2,
            [
                (0.2, CONST, [ONE, ONE]),
                (0.7 / LY, CONST, [ONE, ("lin",)]),
                (0.4, CONST, [_cos(P), _sin(KY)]),
                (-0.2, CONST, [ONE, _sin(2 * KY)]),
                (-0.5, TAU, [ONE, _sin(KY)]),
                (0.3, TAU, [_cos(P), _sin(2 * KY)]),
            ],
        ),
        S2,
        (11, 21),
        (2, 2),
        values={"y_min": 0.2, "y_max": 0.9},
        slopes={"x_min": 0.0, "x_max": 0.0},
    ),
    "neumann_uniform_2d": Cell(
        lambda: neumann_bc(dimension=2, value=_g2),
        Exact(
            T2,
            D2,
            [
                (_g2, CONST, [("sq",), ONE]),
                (-_g2, CONST, [("lin",), ONE]),
                (_g2 / LY, CONST, [ONE, ("sq",)]),
                (-_g2, CONST, [ONE, ("lin",)]),
                (0.4, CONST, [_cos(P), _cos(KY)]),
                (0.2, CONST, [_cos(2 * P), ONE]),
                (0.5, TAU, [_cos(P), ONE]),
            ],
        ),
        S2,
        (11, 21),
        (2, 2),
        slopes=dict.fromkeys(("x_min", "x_max", "y_min", "y_max"), _g2),
    ),
    "neumann_faced_2d": Cell(
        lambda: _faces(
            2,
            {
                "x_min": (BCType.NEUMANN, -0.4),
                "x_max": (BCType.NEUMANN, -0.9),
                "y_min": (BCType.NEUMANN, -0.6),
                "y_max": (BCType.NEUMANN, -0.2),
            },
        ),
        Exact(
            T2,
            D2,
            [
                (0.4, CONST, [("lin",), ONE]),
                (-0.65, CONST, [("sq",), ONE]),
                (0.6, CONST, [ONE, ("lin",)]),
                (-0.4 / LY, CONST, [ONE, ("sq",)]),
                (0.4, CONST, [_cos(P), _cos(KY)]),
                (0.5, TAU, [_cos(P), ONE]),
            ],
        ),
        S2,
        (11, 21),
        (2, 2),
        slopes={"x_min": -0.4, "x_max": -0.9, "y_min": -0.6, "y_max": -0.2},
    ),
}

# The no-flux and periodic 2-D solutions, each run through both paths of the ghost buffer.
_NO_FLUX_2D = Exact(
    T2,
    D2,
    [
        (0.5, CONST, [_cos(P), ONE]),
        (-0.25, CONST, [_cos(2 * P), ONE]),
        (-0.3, CONST, [ONE, _cos(KY)]),
        (0.15, CONST, [ONE, _cos(2 * KY)]),
        (0.15, CONST, [_cos(P), _cos(KY)]),
        (0.6, TAU, [_cos(P), ONE]),
        (-0.4, TAU, [_cos(2 * P), _cos(KY)]),
    ],
)
_PERIODIC_2D = Exact(
    T2,
    D2,
    [
        (0.5, CONST, [_sin(2 * P, 0.3), ONE]),
        (0.3, CONST, [ONE, _sin(2 * KY, 1.1)]),
        (0.15, CONST, [_sin(2 * P), _cos(2 * KY)]),
        (0.4, TAU, [_sin(2 * P, 0.2), ONE]),
        (-0.3, TAU, [ONE, _cos(2 * KY, 0.7)]),
    ],
)
_ALL_FACES = ("x_min", "x_max", "y_min", "y_max")
CELLS["no_flux_uniform_2d"] = Cell(
    lambda: no_flux_bc(dimension=2), _NO_FLUX_2D, S2, (11, 21), (2, 2), slopes=dict.fromkeys(_ALL_FACES, 0.0)
)
CELLS["no_flux_faced_2d"] = Cell(
    lambda: _faces(2, dict.fromkeys(_ALL_FACES, (BCType.NO_FLUX, 0.0))),
    _NO_FLUX_2D,
    S2,
    (11, 21),
    (2, 2),
    slopes=dict.fromkeys(_ALL_FACES, 0.0),
)
CELLS["periodic_uniform_2d"] = Cell(lambda: periodic_bc(dimension=2), _PERIODIC_2D, S2, (11, 21), (2, 2), seam=True)
CELLS["periodic_faced_2d"] = Cell(
    lambda: _faces(2, dict.fromkeys(_ALL_FACES, (BCType.PERIODIC, 0.0))), _PERIODIC_2D, S2, (11, 21), (2, 2), seam=True
)


# ------------------------------------------------------------------------------------------------- harness


@dataclass
class Level:
    error: float
    values: dict  # Dirichlet face -> max |U - g| at t = 0
    slopes: dict  # face -> max |second-order outward slope - datum| at t = 0
    seam: float  # max |u(0) - u(last)| along each periodic axis at t = 0; 0.0 for a cell without a seam
    failures: tuple


def _face_index(dim: int, face: str) -> tuple:
    axis = {"x": 0, "y": 1}[face[0]]
    index: list = [slice(None)] * dim
    index[axis] = 0 if face.endswith("min") else -1
    return tuple(index)


def _outward_slope(u: np.ndarray, face: str, h: float) -> np.ndarray:
    """Second-order one-sided outward normal derivative on ``face``."""
    axis = {"x": 0, "y": 1}[face[0]]
    v = np.moveaxis(u, axis, 0)
    if face.endswith("min"):
        return (3 * v[0] - 4 * v[1] + v[2]) / (2 * h)
    return (3 * v[-1] - 4 * v[-2] + v[-3]) / (2 * h)


def _grid(dim: int, n: int, bc: BoundaryConditions) -> TensorProductGrid:
    if dim == 1:
        return TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=bc)
    return TensorProductGrid(bounds=[(0.0, 1.0), (0.0, LY)], Nx_points=[n, n], boundary_conditions=bc)


def solve_level(cell: Cell, n: int, nt: int) -> Level:
    grid = _grid(cell.dim, n, cell.bc())
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=cell.sigma
        ),
        domain=grid,
        conditions=Conditions(
            m_initial=lambda x: 1.0 + 0.0 * float(np.sum(x)), u_terminal=lambda x: 0.0, T=cell.exact.T
        ),
        Nt=nt,
    )
    X = list(np.meshgrid(*grid.coordinates, indexing="ij"))
    u_T = cell.exact.u(cell.exact.T, X)
    shape = (nt + 1, *X[0].shape)
    solver = HJBFDMSolver(problem)
    with warnings.catch_warnings():
        # The 2-D Newton tries JAX autodiff on a NumPy residual and falls back to finite differences, at
        # every step; the warning is hjb_fdm's and is not what this file measures.
        warnings.filterwarnings("ignore", message="JAX autodiff failed", category=RuntimeWarning)
        U = np.asarray(
            solver.solve_hjb_system(
                M_density=np.ones(shape),
                U_terminal=u_T,
                U_coupling_prev=np.broadcast_to(u_T, shape).copy(),
                source_term=cell.exact.source,
            )
        )
    u0 = U[0]
    h = [float(c[1] - c[0]) for c in grid.coordinates]
    seam = 0.0
    if cell.seam:
        # The duplicated endpoint is one point only on an endpoint-inclusive grid (#1822).
        assert grid.periodic_convention is PeriodicGridConvention.ENDPOINT_INCLUSIVE, grid.periodic_convention
        seam = max(float(np.abs(u0.take(0, axis=a) - u0.take(-1, axis=a)).max()) for a in range(cell.dim))
    return Level(
        error=float(np.abs(u0 - cell.exact.u(0.0, X)).max()),
        values={f: float(np.abs(u0[_face_index(cell.dim, f)] - g).max()) for f, g in cell.values.items()},
        slopes={
            f: float(np.abs(_outward_slope(u0, f, h[{"x": 0, "y": 1}[f[0]]]) - g).max()) for f, g in cell.slopes.items()
        },
        seam=seam,
        failures=tuple(solver.inner_solve_failures() or ()),
    )


def ladder(cell: Cell) -> list[Level]:
    return [solve_level(cell, n, nt) for n, nt in zip(cell.levels, cell.nts, strict=True)]


# ---------------------------------------------------------------------------------------------- the record
# Errors at t = 0 per level, and each wall-carrying cell's largest second-order slope miss at the finest level:
# the 1-D cells measured at d1a9a1e8 (#2537), the 2-D cells on the rectangle at 711204a6 (#2547). The bands below
# are two-sided: a closure that is wrong at O(h) can keep the rate and move the level either way.
MEASURED = {
    "dirichlet_1d": (5.8975e-02, 2.9356e-02, 1.4653e-02),
    "neumann_1d": (5.8333e-02, 2.9202e-02, 1.4486e-02),
    "no_flux_1d": (8.9860e-02, 4.5278e-02, 2.2454e-02),
    "periodic_1d": (1.7122e-01, 8.8253e-02, 4.4669e-02),
    "dirichlet_uniform_2d": (7.2590e-02, 3.8216e-02),
    "dirichlet_faced_x_2d": (9.2265e-02, 5.5684e-02),
    "dirichlet_faced_y_2d": (2.3146e-01, 1.4634e-01),
    "neumann_uniform_2d": (5.7708e-02, 3.3905e-02),
    "neumann_faced_2d": (1.0539e-01, 5.9103e-02),
    "no_flux_uniform_2d": (3.3089e-01, 1.8712e-01),
    "no_flux_faced_2d": (3.3089e-01, 1.8712e-01),
    "periodic_uniform_2d": (3.6383e-01, 2.1056e-01),
    "periodic_faced_2d": (3.6383e-01, 2.1056e-01),
}
SLOPE_MEASURED = {
    "neumann_1d": 9.03e-04,
    "no_flux_1d": 1.38e-03,
    "dirichlet_faced_x_2d": 1.03e-02,
    "dirichlet_faced_y_2d": 3.38e-03,
    "neumann_uniform_2d": 7.55e-03,
    "neumann_faced_2d": 6.84e-03,
    "no_flux_uniform_2d": 9.76e-02,
    "no_flux_faced_2d": 9.76e-02,
}
LEVEL_BAND = 1.25
RATIO_BAND = {1: (1.6, 2.6), 2: (1.5, 2.5)}


def test_every_declared_bc_type_has_a_cell_in_both_dimensions():
    """The cells cover HJBFDMSolver's declared set, in 1-D and 2-D, so a type added to the declaration
    without an oracle fails here."""
    declared = {t.value for t in HJBFDMSolver._SUPPORTED_BC_TYPES}
    for dim in (1, 2):
        covered = {name.split("_")[0] for name in CELLS if name.endswith(f"_{dim}d")}
        covered = {"no_flux" if c == "no" else c for c in covered}
        assert declared <= covered, f"{dim}-D: declared {sorted(declared)}, oracle cells for {sorted(covered)}"
    assert set(CELLS) == set(MEASURED)


@pytest.mark.parametrize("name", sorted(CELLS))
def test_an_exact_solution_is_reproduced_to_its_order_level_and_wall(name: str):
    cell = CELLS[name]
    levels = ladder(cell)
    errors = [level.error for level in levels]

    failed = [level.failures for level in levels if level.failures]
    assert not failed, f"{name}: inner solves failed: {failed}"

    lo, hi = RATIO_BAND[cell.dim]
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(lo < r < hi for r in ratios), f"{name}: error ratios {ratios} outside ({lo}, {hi}); errors {errors}"

    for n, got, want in zip(cell.levels, errors, MEASURED[name], strict=True):
        assert want / LEVEL_BAND <= got <= want * LEVEL_BAND, (
            f"{name}: error {got:.4e} at n = {n} is outside [{want / LEVEL_BAND:.4e}, {want * LEVEL_BAND:.4e}] "
            f"around the recorded {want:.4e}; the wall closure moved"
        )

    for level in levels:
        for face, miss in level.values.items():
            assert miss == 0.0, f"{name}: the Dirichlet face {face} misses g by {miss:.3e}"
        assert level.seam < 1e-10, f"{name}: the duplicated seam endpoints differ by {level.seam:.3e}"

    if cell.slopes:
        fine, coarse = max(levels[-1].slopes.values()), max(levels[-2].slopes.values())
        assert fine <= 2.0 * SLOPE_MEASURED[name], (
            f"{name}: the wall slope misses its datum by {fine:.3e} at the finest level, recorded {SLOPE_MEASURED[name]:.3e}"
        )
        assert coarse >= 3.0 * fine, f"{name}: wall slope miss {coarse:.3e} -> {fine:.3e} is not second order"

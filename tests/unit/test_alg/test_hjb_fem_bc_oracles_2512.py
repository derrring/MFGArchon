"""HJB-FEM against exact solutions, on every boundary condition it declares, in 1-D and 2-D (#2512 (a)).

P1 elements, so the error falls at second order. Each cell is run on both arms of the step:

- **Picard**, the default, is handed the exact solution as its previous iterate. That isolates the step's
  boundary treatment, including the Dirichlet lift, which only Picard reads.
- **Newton** solves the nonlinear step itself. It is the full oracle.

**Fixtures.** Each solution is ``u = psi(x) + w(t) phi(x)`` with ``w`` linear in t, so backward Euler is
exact in time and the ratios are spatial (``Nt`` 5 -> 40 moves the finest 1-D error at 40 cells by under 1.8%).
- Data that carry a value differ from wall to wall.
- The optimal drift ``-grad u`` points into every wall that carries a flux datum (NEUMANN, ROBIN).
- The no-flux and reflecting solutions are even about no midline.
- Dirichlet data are time-independent: the solver reads ``value(x)``, with no ``t``. In 2-D they are per-face
  callables that agree at the corners, as the corner guard requires.
- The 2-D domain is [0, 1] x [0, LY] with LY = 0.6, so dx != dy and the extent is not square (#2547).

**Assertions per arm.**
- The error ratio, in (3.0, 5.0).
- Each level's error, two-sided, within a factor of 1.25 either way of the value recorded here.
- Dirichlet nodes equal ``g`` exactly at every time.

A level outside its band in either direction is a change in the boundary treatment. Re-measure and record it.
REFLECTING has no code of its own here: it takes the NEUMANN/NO_FLUX arm. Its cells are the NO_FLUX
solutions under REFLECTING segments, so they pin that the type keeps reaching that arm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
import skfem

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver
from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
from mfgarchon.geometry.meshes.mesh_1d import Mesh1D
from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

if TYPE_CHECKING:
    from collections.abc import Callable

T, SIGMA = 0.5, 0.8
D = SIGMA**2 / 2
P = np.pi


def _w(t):
    return 0.5 * (1.0 + 0.8 * (T - t))


_DW = -0.4  # dw/dt


@dataclass
class Field:
    """A scalar field on points X of shape (N, d): value, gradient (N, d), Laplacian."""

    f: Callable
    grad: Callable
    lap: Callable


@dataclass
class Cell:
    dim: int
    segments: Callable[[], list]
    psi: Field
    phi: Field
    levels: tuple
    nt: int
    dirichlet: dict = field(default_factory=dict)  # face -> g(X) for the exact-wall check

    def u(self, t, X):
        return self.psi.f(X) + _w(t) * self.phi.f(X)

    def source(self, t, x):
        X = np.asarray(x, dtype=float).reshape(-1, self.dim)
        grad = self.psi.grad(X) + _w(t) * self.phi.grad(X)
        lap = self.psi.lap(X) + _w(t) * self.phi.lap(X)
        return -_DW * self.phi.f(X) + 0.5 * np.sum(grad**2, axis=1) - D * lap


def _seg(face, bc_type, value=0.0, alpha=1.0, beta=0.0):
    return BCSegment(name=face, bc_type=bc_type, boundary=face, value=value, alpha=alpha, beta=beta)


def _x(X):
    return X[:, 0]


def _y(X):
    return X[:, 1]


def _col(*cols):
    return np.stack(cols, axis=1)


# ------------------------------------------------------------------------------------------------- 1-D fields
_SIN1 = Field(
    lambda X: np.sin(P * _x(X)) + 0.5 * np.sin(2 * P * _x(X)),
    lambda X: _col(P * np.cos(P * _x(X)) + P * np.cos(2 * P * _x(X))),
    lambda X: -(P**2) * np.sin(P * _x(X)) - 2 * P**2 * np.sin(2 * P * _x(X)),
)
_COS1 = Field(
    lambda X: np.cos(P * _x(X)) + 0.5 * np.cos(2 * P * _x(X)),
    lambda X: _col(-P * np.sin(P * _x(X)) - P * np.sin(2 * P * _x(X))),
    lambda X: -(P**2) * np.cos(P * _x(X)) - 2 * P**2 * np.cos(2 * P * _x(X)),
)


def _poly1(c0, c1, c2):
    return Field(
        lambda X: c0 + c1 * _x(X) + c2 * _x(X) ** 2,
        lambda X: _col(c1 + 2 * c2 * _x(X)),
        lambda X: 2 * c2 + 0.0 * _x(X),
    )


# p satisfies the homogeneous Robin pair (1, 1) at x = 0 and (2, 0.5) at x = 1.
_P1D = Field(
    lambda X: 1 + _x(X) - 1.5 * _x(X) ** 2,
    lambda X: _col(1 - 3 * _x(X)),
    lambda X: -3.0 + 0.0 * _x(X),
)

# ------------------------------------------------------------------------------------------------- 2-D fields
# The 2-D domain is [0, 1] x [0, LY] with n cells each way, so dx != dy and the extent is not square: a square
# grid is a symmetry, and it hides any defect that confuses the two axes (#2547). KY = pi / LY.
LY = 0.6
KY = P / LY
_SIN2 = Field(
    lambda X: (
        np.sin(P * _x(X)) * np.sin(KY * _y(X))
        + 0.5 * np.sin(2 * P * _x(X)) * np.sin(KY * _y(X))
        + 0.3 * np.sin(P * _x(X)) * np.sin(2 * KY * _y(X))
    ),
    lambda X: _col(
        P * np.cos(P * _x(X)) * np.sin(KY * _y(X))
        + P * np.cos(2 * P * _x(X)) * np.sin(KY * _y(X))
        + 0.3 * P * np.cos(P * _x(X)) * np.sin(2 * KY * _y(X)),
        KY * np.sin(P * _x(X)) * np.cos(KY * _y(X))
        + 0.5 * KY * np.sin(2 * P * _x(X)) * np.cos(KY * _y(X))
        + 0.6 * KY * np.sin(P * _x(X)) * np.cos(2 * KY * _y(X)),
    ),
    lambda X: (
        -(P**2 + KY**2) * np.sin(P * _x(X)) * np.sin(KY * _y(X))
        - 0.5 * (4 * P**2 + KY**2) * np.sin(2 * P * _x(X)) * np.sin(KY * _y(X))
        - 0.3 * (P**2 + 4 * KY**2) * np.sin(P * _x(X)) * np.sin(2 * KY * _y(X))
    ),
)
_COS2 = Field(
    lambda X: np.cos(P * _x(X)) + 0.5 * np.cos(KY * _y(X)) + 0.4 * np.cos(2 * P * _x(X)) * np.cos(KY * _y(X)),
    lambda X: _col(
        -P * np.sin(P * _x(X)) - 0.8 * P * np.sin(2 * P * _x(X)) * np.cos(KY * _y(X)),
        -0.5 * KY * np.sin(KY * _y(X)) - 0.4 * KY * np.cos(2 * P * _x(X)) * np.sin(KY * _y(X)),
    ),
    lambda X: (
        -(P**2) * np.cos(P * _x(X))
        - 0.5 * KY**2 * np.cos(KY * _y(X))
        - 0.4 * (4 * P**2 + KY**2) * np.cos(2 * P * _x(X)) * np.cos(KY * _y(X))
    ),
)


def _poly2(c0, a1, b1, a2, b2, cxy=0.0):
    return Field(
        lambda X: c0 + a1 * _x(X) + b1 * _x(X) ** 2 + a2 * _y(X) + b2 * _y(X) ** 2 + cxy * _x(X) * _y(X),
        lambda X: _col(a1 + 2 * b1 * _x(X) + cxy * _y(X), a2 + 2 * b2 * _y(X) + cxy * _x(X)),
        lambda X: 2 * b1 + 2 * b2 + 0.0 * _x(X),
    )


# q(y) = 1 + 0.5 y + QC y^2 satisfies the homogeneous Robin pair (0.5, 1) at y = 0 and (1.5, 2) at y = LY.
_QC = -(2.5 + 0.75 * LY) / (1.5 * LY**2 + 4 * LY)
_PQ = Field(
    lambda X: (1 + _x(X) - 1.5 * _x(X) ** 2) * (1 + 0.5 * _y(X) + _QC * _y(X) ** 2),
    lambda X: _col(
        (1 - 3 * _x(X)) * (1 + 0.5 * _y(X) + _QC * _y(X) ** 2),
        (1 + _x(X) - 1.5 * _x(X) ** 2) * (0.5 + 2 * _QC * _y(X)),
    ),
    lambda X: -3.0 * (1 + 0.5 * _y(X) + _QC * _y(X) ** 2) + (1 + _x(X) - 1.5 * _x(X) ** 2) * (2 * _QC),
)
_ALL = ("x_min", "x_max", "y_min", "y_max")

_LEVELS_1D, _NT_1D = (10, 20, 40), 10
_LEVELS_2D, _NT_2D = (16, 32, 64), 6

CELLS: dict[str, Cell] = {
    "dirichlet_1d": Cell(
        1,
        lambda: [_seg("x_min", BCType.DIRICHLET, 0.4), _seg("x_max", BCType.DIRICHLET, -0.7)],
        _poly1(0.4, -1.1, 0.0),
        _SIN1,
        _LEVELS_1D,
        _NT_1D,
        dirichlet={"x_min": lambda X: 0.4 + 0.0 * _x(X), "x_max": lambda X: -0.7 + 0.0 * _x(X)},
    ),
    "neumann_1d": Cell(
        # outward du/dn: x_min -0.6, x_max -0.25, so the drift points into both walls
        1,
        lambda: [
            _seg("x_min", BCType.NEUMANN, -0.6, alpha=0.0, beta=1.0),
            _seg("x_max", BCType.NEUMANN, -0.25, alpha=0.0, beta=1.0),
        ],
        _poly1(0.3, 0.6, -0.425),
        _COS1,
        _LEVELS_1D,
        _NT_1D,
    ),
    "no_flux_1d": Cell(
        1,
        lambda: [_seg("x_min", BCType.NO_FLUX), _seg("x_max", BCType.NO_FLUX)],
        _poly1(0.3, 0.0, 0.0),
        _COS1,
        _LEVELS_1D,
        _NT_1D,
    ),
    "reflecting_1d": Cell(
        1,
        lambda: [_seg("x_min", BCType.REFLECTING), _seg("x_max", BCType.REFLECTING)],
        _poly1(0.3, 0.0, 0.0),
        _COS1,
        _LEVELS_1D,
        _NT_1D,
    ),
    "robin_1d": Cell(
        # (alpha, beta, g): x_min (1, 1, -0.5), x_max (2, 0.5, 2.15); u_x > 0 at 0, < 0 at 1: drift into both walls
        1,
        lambda: [
            _seg("x_min", BCType.ROBIN, -0.5, alpha=1.0, beta=1.0),
            _seg("x_max", BCType.ROBIN, 2.15, alpha=2.0, beta=0.5),
        ],
        _poly1(0.2, 0.7, 0.0),
        _P1D,
        _LEVELS_1D,
        _NT_1D,
    ),
    "dirichlet_2d": Cell(
        # psi is harmonic and gives each face its own linear datum; the four agree at the corners
        2,
        lambda: [
            BCSegment(name="x_min", bc_type=BCType.DIRICHLET, boundary="x_min", value=lambda x: 0.3 - 0.2 * x[1]),
            BCSegment(name="x_max", bc_type=BCType.DIRICHLET, boundary="x_max", value=lambda x: 0.8 + 0.2 * x[1]),
            BCSegment(name="y_min", bc_type=BCType.DIRICHLET, boundary="y_min", value=lambda x: 0.3 + 0.5 * x[0]),
            BCSegment(
                name="y_max",
                bc_type=BCType.DIRICHLET,
                boundary="y_max",
                value=lambda x: (0.3 - 0.2 * LY) + (0.5 + 0.4 * LY) * x[0],
            ),
        ],
        _poly2(0.3, 0.5, 0.0, -0.2, 0.0, cxy=0.4),
        _SIN2,
        _LEVELS_2D,
        _NT_2D,
        dirichlet={
            "x_min": lambda X: 0.3 - 0.2 * _y(X),
            "x_max": lambda X: 0.8 + 0.2 * _y(X),
            "y_min": lambda X: 0.3 + 0.5 * _x(X),
            "y_max": lambda X: (0.3 - 0.2 * LY) + (0.5 + 0.4 * LY) * _x(X),
        },
    ),
    "neumann_2d": Cell(
        # outward du/dn: x_min -0.6, x_max -0.25, y_min -0.3, y_max -0.45 -- the drift points into every wall
        2,
        lambda: [
            _seg(f, BCType.NEUMANN, g, alpha=0.0, beta=1.0)
            for f, g in (("x_min", -0.6), ("x_max", -0.25), ("y_min", -0.3), ("y_max", -0.45))
        ],
        _poly2(0.3, 0.6, -0.425, 0.3, (-0.45 - 0.3) / (2 * LY)),
        _COS2,
        _LEVELS_2D,
        _NT_2D,
    ),
    "no_flux_2d": Cell(
        2, lambda: [_seg(f, BCType.NO_FLUX) for f in _ALL], _poly2(0.3, 0, 0, 0, 0), _COS2, _LEVELS_2D, _NT_2D
    ),
    "reflecting_2d": Cell(
        2, lambda: [_seg(f, BCType.REFLECTING) for f in _ALL], _poly2(0.3, 0, 0, 0, 0), _COS2, _LEVELS_2D, _NT_2D
    ),
    "robin_2d": Cell(
        # alpha u + beta du/dn = alpha * 0.5 on every face, with a different (alpha, beta) per face
        2,
        lambda: [
            _seg("x_min", BCType.ROBIN, 0.5, alpha=1.0, beta=1.0),
            _seg("x_max", BCType.ROBIN, 1.0, alpha=2.0, beta=0.5),
            _seg("y_min", BCType.ROBIN, 0.25, alpha=0.5, beta=1.0),
            _seg("y_max", BCType.ROBIN, 0.75, alpha=1.5, beta=2.0),
        ],
        _poly2(0.5, 0, 0, 0, 0),
        _PQ,
        _LEVELS_2D,
        _NT_2D,
    ),
}


# ------------------------------------------------------------------------------------------------- harness


def _geometry(dim: int, n: int):
    if dim == 1:
        geometry = Mesh1D(bounds=(0.0, 1.0), num_elements=n)
        geometry.generate_mesh()
        return geometry
    xs, ys = np.linspace(0.0, 1.0, n + 1), np.linspace(0.0, LY, n + 1)
    geometry = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, LY))
    geometry.mesh_data = skfem_to_meshdata(skfem.MeshTri.init_tensor(xs, ys))
    return geometry


def _on_face(X: np.ndarray, face: str) -> np.ndarray:
    axis = {"x": 0, "y": 1}[face[0]]
    top = 1.0 if axis == 0 else LY
    return np.isclose(X[:, axis], 0.0 if face.endswith("min") else top)


def solve_level(cell: Cell, n: int, newton: bool) -> tuple[float, float]:
    """(max error over every node and time level, max |U - g| over the Dirichlet nodes)."""
    geometry = _geometry(cell.dim, n)
    geometry.boundary_conditions = BoundaryConditions(dimension=cell.dim, segments=cell.segments())
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(lambda_=1.0),
                coupling=lambda m: 0.0 * np.asarray(m),
                coupling_dm=lambda m: 0.0 * np.asarray(m),
            ),
            volatility=SIGMA,
        ),
        domain=geometry,
        # Unit mass on the mesh's own measure: Mesh1D's uniform-cell measure gives a constant c the mass c (n+1)/n.
        conditions=Conditions(
            m_initial=lambda p, c=(n / (n + 1) if cell.dim == 1 else 1.0): c, u_terminal=lambda p: 0.0, T=T
        ),
        Nt=cell.nt,
    )
    solver = HJBFEMSolver(problem, order=1)
    X = solver._disc.dof_coordinates
    dt = problem.dt
    exact = np.array([cell.u(k * dt, X) for k in range(cell.nt + 1)])
    U = np.asarray(
        solver.solve_hjb_system(
            M_density=np.ones((cell.nt + 1, X.shape[0])),
            U_terminal=exact[-1],
            U_coupling_prev=None if newton else exact,
            source_term=cell.source,
            use_newton=newton,
            newton_tolerance=1e-12,
        )
    )
    wall = 0.0
    for face, g in cell.dirichlet.items():
        on = _on_face(X, face)
        wall = max(wall, float(np.abs(U[:, on] - g(X[on])).max()))
    return float(np.abs(U - exact).max()), wall


@pytest.fixture(autouse=True)
def _quiet():
    logging.getLogger("mfgarchon").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------------------------- the record
# The max error over every node and time level, per level, measured at e8905019 (picard, newton).
MEASURED = {
    "dirichlet_1d": ((6.0141e-02, 1.5629e-02, 3.9632e-03), (3.9455e-02, 9.8267e-03, 2.4625e-03)),
    "neumann_1d": ((9.6124e-02, 2.6721e-02, 7.0035e-03), (7.6583e-02, 2.0180e-02, 5.2771e-03)),
    "no_flux_1d": ((9.2005e-02, 2.5571e-02, 6.6673e-03), (7.7766e-02, 2.0181e-02, 5.1132e-03)),
    "reflecting_1d": ((9.2005e-02, 2.5571e-02, 6.6673e-03), (7.7766e-02, 2.0181e-02, 5.1132e-03)),
    "robin_1d": ((6.0632e-03, 1.6027e-03, 4.1217e-04), (8.2084e-03, 2.2413e-03, 5.8807e-04)),
    "dirichlet_2d": ((1.8884e-02, 4.8449e-03, 1.2211e-03), (1.8269e-02, 4.6720e-03, 1.1742e-03)),
    "neumann_2d": ((3.5555e-02, 1.0471e-02, 2.9211e-03), (4.0363e-02, 1.1029e-02, 3.0541e-03)),
    "no_flux_2d": ((3.5750e-02, 9.7860e-03, 2.6039e-03), (4.0372e-02, 1.1100e-02, 2.9513e-03)),
    "reflecting_2d": ((3.5750e-02, 9.7860e-03, 2.6039e-03), (4.0372e-02, 1.1100e-02, 2.9513e-03)),
    "robin_2d": ((2.5128e-03, 6.8695e-04, 1.8448e-04), (2.7862e-03, 7.6634e-04, 2.0586e-04)),
}
LEVEL_BAND = 1.25
RATIO_BAND = (3.0, 5.0)


def test_every_declared_bc_type_has_a_cell_in_both_dimensions():
    declared = {t.value for t in HJBFEMSolver._SUPPORTED_BC_TYPES}
    for dim in (1, 2):
        covered = {name.rsplit("_", 1)[0] for name in CELLS if name.endswith(f"_{dim}d")}
        assert declared <= covered, f"{dim}-D: declared {sorted(declared)}, oracle cells for {sorted(covered)}"
    assert set(CELLS) == set(MEASURED)


@pytest.mark.parametrize("arm", ["picard", "newton"])
@pytest.mark.parametrize("name", sorted(CELLS))
def test_an_exact_solution_is_reproduced_at_second_order(name: str, arm: str):
    cell = CELLS[name]
    rows = [solve_level(cell, n, newton=arm == "newton") for n in cell.levels]
    errors = [r[0] for r in rows]

    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    lo, hi = RATIO_BAND
    assert all(lo < r < hi for r in ratios), f"{name} [{arm}]: ratios {ratios} outside ({lo}, {hi}); errors {errors}"

    recorded = MEASURED[name][arm == "newton"]
    for n, got, want in zip(cell.levels, errors, recorded, strict=True):
        assert want / LEVEL_BAND <= got <= want * LEVEL_BAND, (
            f"{name} [{arm}]: error {got:.4e} at n = {n} is outside [{want / LEVEL_BAND:.4e}, {want * LEVEL_BAND:.4e}] "
            f"around the recorded {want:.4e}; the boundary treatment moved"
        )

    for n, (_, wall) in zip(cell.levels, rows, strict=True):
        assert wall < 1e-12, f"{name} [{arm}]: a Dirichlet node misses g by {wall:.3e} at n = {n}"

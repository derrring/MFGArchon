"""FP-FEM against exact densities, on its no-flux and reflecting walls, its shared Dirichlet exit and its own Robin
wall, in 1-D and 2-D (#2512 (a)).

P1 elements, so the density error falls at second order. The FP equation is ``d_t m + div(v m) - D Lap m = S`` with
``v = -grad(phi)`` from the potential the test hands in (quadratic H, control cost 1), and ``S`` the manufactured source.

**Fixtures.** Each density is ``m = w(t) exp(-phi/D) g(x)`` with ``phi`` linear, so the total flux is
``J = v m - D grad m = -D w exp(-phi/D) grad g``.
- **No flux.** ``dg/dn = 0`` on every wall, so ``J.n = 0`` there, while the drift ``v = -grad(phi)`` is nonzero
  and crosses every wall. A density chosen only to satisfy the condition would be symmetric, and symmetry is where
  ``J.n = 0`` and ``dm/dn = 0`` coincide; here they differ, by ``(v.n) m``.
- **Exit.** The shared BC is ``DIRICHLET(0.7)`` on ``x_max``, read at the FP as absorbing with its value dropped
  (#2512 row B3): ``g = 0`` there and ``dg/dn = 0`` on the other walls, with an outward flux at the exit.
- **Reflecting.** The no-flux density under REFLECTING segments. REFLECTING has no code of its own at the FP: it takes
  the NO_FLUX wall, and these cells solve bit-identically to the no-flux ones (measured, max difference 0.0). They
  pin that the type keeps reaching that wall; by #2512 comment 6079979873 they count once the no-flux cells do.
- **Robin.** The FP's own ROBIN, handed to the solver (a shared ROBIN is the HJB's and is refused at the FP), read as
  ``J.n = (D/beta)(alpha m - g)`` (user ruling 2026-10-10), with ``alpha`` and ``beta`` different on every face. The
  density adds a constant, ``m = w exp(-phi/D) g + Q0``, and ``g`` and its normal derivative vanish on every wall,
  so there ``m = Q0`` and ``J.n = (v.n) Q0``: each face's datum ``alpha Q0 - (beta/D)(v.n) Q0`` is a constant, as the
  FEM Robin load requires, while the drift crosses every wall. A constant lies in the P1 space, so ``Q0`` adds no
  error of its own (measured: the errors agree to four digits at Q0 = 0.4 and 2.0) and only scales the rival's miss.
- ``w`` is linear in t and ``phi`` does not depend on t, so the implicit-Euler time error vanishes and the ratios are
  spatial. The 2-D domain is [0, 1] x [0, LY], LY = 0.6, and the drift differs per axis.

**Assertions.** Per cell: the error ratio in (3.0, 5.0), and each level within a factor of 1.25 either way of the value
recorded here. Per wall: what the cell is named for, measured, and a rival that must miss.
- No flux: ``||J_h.n||`` on the boundary falls at first order (P1 gradients), and the rival, the same solve with
  the wall written as ``dm/dn = 0`` (the advection without its by-parts form), misses the density by a constant.
- Robin: the residual ``J_h.n - (D/beta)(alpha m_h - g)`` on the boundary falls at first order, and the rival that
  reads the same coefficients as ``alpha m + beta dm/dn = g`` -- the same solve with the advection's facet term
  removed -- misses the density by a constant: the two differ by ``(beta/D)(v.n) m``.
- Exit: the exit nodes hold 0 at every time, and the rival that reads the shared value as a density Dirichlet
  (lift and write both keep 0.7) misses away from the exit nodes. The mass identity -- the change in mass equals the source
  minus the integrated exit flux -- is reported beside the oracle, not as it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
import skfem
from skfem.helpers import dot, grad

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver
from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
from mfgarchon.geometry.meshes.mesh_1d import Mesh1D
from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

if TYPE_CHECKING:
    from collections.abc import Callable

T, SIGMA, NT, LY, EPS = 0.5, 0.8, 20, 0.6, 0.5
D = SIGMA**2 / 2
KX, KY = np.pi, np.pi / LY
VEL = np.array([-1.0, 0.6])  # v = -grad(phi), phi = 1.0 x - 0.6 y: it crosses every wall
EXIT_VALUE = 0.7  # the shared Dirichlet's value, which the FP drops


def _w(t):
    return 0.5 * (1.0 + 0.8 * (T - t))


_DW = -0.4  # dw/dt


@dataclass
class Shape:
    """``g`` on points X of shape (N, d): value, gradient (N, d), Laplacian."""

    f: Callable
    grad: Callable
    lap: Callable


def _cos_x(k: float, eps: float) -> tuple[Callable, Callable, Callable]:
    """``1 + eps cos(k x)``, or ``cos(k x)`` when ``eps`` is None, with its first and second derivatives."""
    if eps is None:
        return (lambda x: np.cos(k * x), lambda x: -k * np.sin(k * x), lambda x: -(k**2) * np.cos(k * x))
    return (
        lambda x: 1 + eps * np.cos(k * x),
        lambda x: -eps * k * np.sin(k * x),
        lambda x: -eps * k**2 * np.cos(k * x),
    )


def _bump(k: float) -> tuple[Callable, Callable, Callable]:
    """``1 - cos(k x)``: with ``k = 2 pi / L`` it and its derivative vanish at both ends of [0, L]."""
    return (lambda x: 1 - np.cos(k * x), lambda x: k * np.sin(k * x), lambda x: k**2 * np.cos(k * x))


def _shape(dim: int, x_factor, y_factor=None) -> Shape:
    fx, dfx, d2fx = x_factor
    if dim == 1:
        return Shape(
            lambda X: fx(X[:, 0]),
            lambda X: dfx(X[:, 0])[:, None],
            lambda X: d2fx(X[:, 0]),
        )
    fy, dfy, d2fy = _cos_x(KY, EPS) if y_factor is None else y_factor
    return Shape(
        lambda X: fx(X[:, 0]) * fy(X[:, 1]),
        lambda X: np.stack([dfx(X[:, 0]) * fy(X[:, 1]), fx(X[:, 0]) * dfy(X[:, 1])], axis=1),
        lambda X: d2fx(X[:, 0]) * fy(X[:, 1]) + fx(X[:, 0]) * d2fy(X[:, 1]),
    )


@dataclass
class Cell:
    dim: int
    faces: Callable[[], list]
    g: Shape
    ns: tuple
    q0: float = 0.0  # a constant added to the density: steady, with flux v q0 and no source
    own: bool = False  # the faces are the FP's own BC, handed to the solver; the shared BC is then no-flux

    def _e(self, X):
        return np.exp((X @ VEL[: self.dim]) / D)

    def m(self, t, X):
        return _w(t) * self._e(X) * self.g.f(X) + self.q0

    def flux(self, t, X):
        """J = v m - D grad m = -D w exp(-phi/D) grad g + v q0."""
        return -D * _w(t) * self._e(X)[:, None] * self.g.grad(X) + self.q0 * VEL[: self.dim]

    def source(self, t, x):
        X = np.asarray(x, dtype=float).reshape(-1, self.dim)
        e, gv = self._e(X), self.g.f(X)
        # S = d_t m + div J, div J = -D w e (Lap g - grad(phi).grad g / D) and grad(phi) = -v.
        return _DW * e * gv - D * _w(t) * e * (self.g.lap(X) + (self.g.grad(X) @ VEL[: self.dim]) / D)


def _seg(face, bc_type, value=0.0, **coefficients):
    return BCSegment(name=face, bc_type=bc_type, boundary=face, value=value, **coefficients)


_FACES = {1: ("x_min", "x_max"), 2: ("x_min", "x_max", "y_min", "y_max")}


def _no_flux(dim):
    return lambda: [_seg(f, BCType.NO_FLUX) for f in _FACES[dim]]


def _reflecting(dim):
    return lambda: [_seg(f, BCType.REFLECTING) for f in _FACES[dim]]


# Outward normals, and each face's Robin coefficients (alpha, beta), different per face.
_NORMAL = {"x_min": (-1.0, 0.0), "x_max": (1.0, 0.0), "y_min": (0.0, -1.0), "y_max": (0.0, 1.0)}
_ROBIN_AB = {"x_min": (1.0, 1.0), "x_max": (2.0, 0.5), "y_min": (0.5, 1.0), "y_max": (1.5, 2.0)}
Q0 = 2.0


def _robin_value(face: str, dim: int) -> float:
    """The FP Robin datum the density satisfies on ``face``: ``J.n = (D/beta)(alpha m - g)``. There ``m = Q0`` and
    ``J.n = (v.n) Q0``, so ``g = alpha Q0 - (beta/D)(v.n) Q0``."""
    alpha, beta = _ROBIN_AB[face]
    vn = float(np.dot(VEL[:dim], _NORMAL[face][:dim]))
    return alpha * Q0 - (beta / D) * vn * Q0


def _robin(dim):
    return lambda: [
        _seg(f, BCType.ROBIN, _robin_value(f, dim), alpha=_ROBIN_AB[f][0], beta=_ROBIN_AB[f][1]) for f in _FACES[dim]
    ]


def _exit(dim):
    return lambda: (
        [_seg("x_max", BCType.DIRICHLET, EXIT_VALUE)] + [_seg(f, BCType.NO_FLUX) for f in _FACES[dim] if f != "x_max"]
    )


CELLS = {
    "no_flux_1d": Cell(1, _no_flux(1), _shape(1, _cos_x(KX, EPS)), (10, 20, 40)),
    "no_flux_2d": Cell(2, _no_flux(2), _shape(2, _cos_x(KX, EPS)), (8, 16, 32)),
    "dirichlet_exit_1d": Cell(1, _exit(1), _shape(1, _cos_x(KX / 2, None)), (10, 20, 40)),
    "dirichlet_exit_2d": Cell(2, _exit(2), _shape(2, _cos_x(KX / 2, None)), (8, 16, 32)),
    "reflecting_1d": Cell(1, _reflecting(1), _shape(1, _cos_x(KX, EPS)), (10, 20, 40)),
    "reflecting_2d": Cell(2, _reflecting(2), _shape(2, _cos_x(KX, EPS)), (8, 16, 32)),
    "robin_1d": Cell(1, _robin(1), _shape(1, _bump(2 * KX)), (10, 20, 40), q0=Q0, own=True),
    "robin_2d": Cell(2, _robin(2), _shape(2, _bump(2 * KX), _bump(2 * KY)), (8, 16, 32), q0=Q0, own=True),
}


class _WallAsZeroNormalDerivative(FPFEMSolver):
    """The rival no-flux wall: ``dm/dn = 0``. Transposing the solver's by-parts advection back gives
    ``int phi_i (v . grad phi_j)``, which for a constant drift is ``div(v m)`` without its facet term, so the natural
    condition becomes ``D dm/dn = 0`` instead of ``J.n = 0``. Under a Robin wall the same Robin terms then impose
    ``alpha m + beta dm/dn = g`` instead of ``J.n = (D/beta)(alpha m - g)``: the rival Robin reading."""

    def _build_advection(self, U_n, D=0.0):
        return -super()._build_advection(U_n, D).T


class _ExitKeepsTheValue(FPFEMSolver):
    """The rival exit: the shared Dirichlet read as a density Dirichlet, ``m = 0.7`` at the exit. Both the
    condensation's lift and the post-solve write take the raw shared BC instead of the FP view, so the value is
    kept throughout the solve, not only written into the exit nodes afterwards."""

    def _shared(self):
        return self.problem.geometry.boundary_conditions

    def _apply_bc_to_system(self, matrix, rhs):
        from mfgarchon.alg.numerical.fem.bc_adapter import apply_bc_to_fem_system

        return apply_bc_to_fem_system(matrix, rhs, self._basis, self._shared())

    def _dirichlet_dofs_and_values(self):
        from mfgarchon.alg.numerical.fem.bc_adapter import get_dirichlet_dofs_and_values

        return get_dirichlet_dofs_and_values(self._basis, self._shared())


def _geometry(cell: Cell, n: int):
    if cell.dim == 1:
        geometry = Mesh1D(bounds=(0.0, 1.0), num_elements=n)
        geometry.generate_mesh()
    else:
        geometry = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, LY))
        geometry.mesh_data = skfem_to_meshdata(
            skfem.MeshTri.init_tensor(np.linspace(0.0, 1.0, n + 1), np.linspace(0.0, LY, n + 1))
        )
    shared = _no_flux(cell.dim)() if cell.own else cell.faces()
    geometry.boundary_conditions = BoundaryConditions(dimension=cell.dim, segments=shared)
    return geometry


def _solve(cell: Cell, n: int, solver_cls=FPFEMSolver):
    """(solver, dof coordinates, density at every level) for ``cell`` at ``n`` cells per axis."""
    problem = MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=SIGMA),
        domain=_geometry(cell, n),
        # The problem's own initial density is evaluated at construction, but the solve is handed the manufactured one;
        # the constant only makes MFGProblem's mass report read 1 on Mesh1D's uniform-cell measure.
        conditions=Conditions(
            m_initial=lambda p, c=(n / (n + 1) if cell.dim == 1 else 1.0): c, u_terminal=lambda p: 0.0, T=T
        ),
        Nt=NT,
    )
    own = BoundaryConditions(dimension=cell.dim, segments=cell.faces()) if cell.own else None
    solver = solver_cls(problem, boundary_conditions=own)
    X = solver._disc.dof_coordinates
    phi = -(X @ VEL[: cell.dim])
    M = solver.solve_fp_system(
        M_initial=cell.m(0.0, X),
        potential_field=np.broadcast_to(phi, (NT + 1, X.shape[0])).copy(),
        source_term=cell.source,
    )
    return solver, X, np.asarray(M)


def _error(cell: Cell, X, M) -> float:
    """The max error over every node and time level."""
    return float(max(np.abs(M[k] - cell.m(k * T / NT, X)).max() for k in range(NT + 1)))


def _wall_flux(cell: Cell, solver, m_h) -> float:
    """``||J_h.n||_{L2}`` over the boundary, with ``J_h = v m_h - D grad m_h`` from the P1 density."""
    facets = skfem.FacetBasis(solver._skfem_mesh, solver.basis.elem)
    v = VEL[: cell.dim]

    def squared(w):
        vn = sum(v[i] * w.n[i] for i in range(cell.dim))
        return (vn * w["m"] - D * dot(grad(w["m"]), w.n)) ** 2

    return float(np.sqrt(skfem.Functional(squared).assemble(facets, m=facets.interpolate(m_h))))


def _robin_residual(cell: Cell, solver, m_h) -> float:
    """``||J_h.n - (D/beta)(alpha m_h - g)||_{L2}`` over the boundary, face by face, with
    ``J_h = v m_h - D grad m_h``."""
    mesh, v = solver._skfem_mesh, VEL[: cell.dim]
    total = 0.0
    for segment in cell.faces():
        facets = skfem.FacetBasis(mesh, solver.basis.elem, facets=mesh.boundaries[segment.boundary])

        def squared(w, a=segment.alpha, b=segment.beta, g=segment.value):
            vn = sum(v[i] * w.n[i] for i in range(cell.dim))
            return (vn * w["m"] - D * dot(grad(w["m"]), w.n) - (D / b) * (a * w["m"] - g)) ** 2

        total += float(skfem.Functional(squared).assemble(facets, m=facets.interpolate(m_h)))
    return float(np.sqrt(total))


@pytest.fixture(autouse=True)
def _quiet(caplog):
    caplog.set_level(logging.WARNING, logger="mfgarchon")


# ---------------------------------------------------------------------------------------------- the record
# The max error over every node and time level, per level, measured at 1290195a.
MEASURED = {
    "no_flux_1d": (1.4227e-02, 3.5673e-03, 8.9248e-04),
    "no_flux_2d": (1.1853e-01, 3.2328e-02, 8.5018e-03),
    "dirichlet_exit_1d": (8.1297e-03, 2.0416e-03, 5.1097e-04),
    "dirichlet_exit_2d": (7.3374e-02, 2.0187e-02, 5.3398e-03),
    # Measured on #2512's PR 2b, based on d74160d1. The reflecting cells equal the no-flux ones bit for bit.
    "reflecting_1d": (1.4227e-02, 3.5673e-03, 8.9248e-04),
    "reflecting_2d": (1.1853e-01, 3.2328e-02, 8.5018e-03),
    "robin_1d": (4.8605e-02, 1.2051e-02, 3.0065e-03),
    "robin_2d": (3.6748e-01, 9.7395e-02, 2.4890e-02),
}
# ||J_h.n|| on the no-flux boundary at T, per level, measured at 1290195a; the reflecting rows on PR 2b, equal to the
# no-flux ones bit for bit.
MEASURED_WALL_FLUX = {
    "no_flux_1d": (7.5829e-02, 3.8183e-02, 1.9238e-02),
    "no_flux_2d": (1.8420e-01, 9.4860e-02, 4.7817e-02),
    "reflecting_1d": (7.5829e-02, 3.8183e-02, 1.9238e-02),
    "reflecting_2d": (1.8420e-01, 9.4860e-02, 4.7817e-02),
}
# ||J_h.n - (D/beta)(alpha m_h - g)|| on the Robin boundary at T, per level, measured on PR 2b. It falls at first
# order from below: in 1-D the ratios run 1.62, 1.81, 1.91, 1.95, 1.98 out to 320 cells.
MEASURED_ROBIN_RESIDUAL = {
    "robin_1d": (2.1107e-01, 1.3035e-01, 7.1963e-02),
    "robin_2d": (6.5469e-01, 3.9902e-01, 2.1818e-01),
}
LEVEL_BAND = 1.25
MASS_BAND = 2e-3  # relative to the expected change; measured 2.77e-4 (1-D) and 2.56e-4 (2-D) at the finest level
RATIO_BAND = (3.0, 5.0)


@pytest.mark.parametrize("name", sorted(CELLS))
def test_an_exact_density_is_reproduced_at_second_order(name: str):
    cell = CELLS[name]
    errors = [_error(cell, *_solve(cell, n)[1:]) for n in cell.ns]
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(RATIO_BAND[0] < r < RATIO_BAND[1] for r in ratios), (name, errors, ratios)
    for level, (got, recorded) in enumerate(zip(errors, MEASURED[name], strict=True)):
        assert recorded / LEVEL_BAND < got < recorded * LEVEL_BAND, (name, level, got, recorded)


@pytest.mark.parametrize("name", ["no_flux_1d", "no_flux_2d", "reflecting_1d", "reflecting_2d"])
def test_the_wall_carries_no_total_flux_and_a_neumann_wall_misses(name: str):
    """``J_h.n`` on the boundary falls at first order. The rival wall, ``dm/dn = 0``, keeps a flux ``(v.n) m`` and
    misses the density by a constant at every resolution, which is what separates it."""
    cell = CELLS[name]
    flux = []
    for n in cell.ns:
        solver, _X, M = _solve(cell, n)
        flux.append(_wall_flux(cell, solver, M[-1]))
    assert all(1.7 < flux[i] / flux[i + 1] < 2.3 for i in range(len(flux) - 1)), flux
    for level, (got, recorded) in enumerate(zip(flux, MEASURED_WALL_FLUX[name], strict=True)):
        assert recorded / LEVEL_BAND < got < recorded * LEVEL_BAND, (name, level, got, recorded)
    rival = [_error(cell, *_solve(cell, n, _WallAsZeroNormalDerivative)[1:]) for n in cell.ns[-2:]]
    assert rival[-1] > 50 * MEASURED[name][-1], rival
    assert rival[0] / rival[1] < 1.2, f"the rival converges, so the cell does not separate the walls: {rival}"


@pytest.mark.parametrize("name", ["robin_1d", "robin_2d"])
def test_the_robin_wall_is_on_the_total_flux_and_a_robin_on_the_density_misses(name: str):
    """The FP's own ROBIN(alpha, beta, g) is ``J.n = (D/beta)(alpha m - g)``: its residual on the boundary falls at
    first order, from below (each ratio in (1.5, 2.3), the finest in (1.7, 2.3)). The rival reads the same
    coefficients as ``alpha m + beta dm/dn = g``, the HJB's shape. Where ``v.n`` is nonzero -- every wall here -- the
    two differ by ``(beta/D)(v.n) m``, and the rival misses the density by a constant at every resolution."""
    cell = CELLS[name]
    residual = []
    for n in cell.ns:
        solver, _X, M = _solve(cell, n)
        residual.append(_robin_residual(cell, solver, M[-1]))
    ratios = [residual[i] / residual[i + 1] for i in range(len(residual) - 1)]
    assert all(1.5 < r < 2.3 for r in ratios), (residual, ratios)
    assert 1.7 < ratios[-1] < 2.3, (residual, ratios)
    for level, (got, recorded) in enumerate(zip(residual, MEASURED_ROBIN_RESIDUAL[name], strict=True)):
        assert recorded / LEVEL_BAND < got < recorded * LEVEL_BAND, (name, level, got, recorded)
    rival = [_error(cell, *_solve(cell, n, _WallAsZeroNormalDerivative)[1:]) for n in cell.ns[-2:]]
    assert rival[-1] > 50 * MEASURED[name][-1], rival
    assert rival[0] / rival[1] < 1.2, f"the rival converges, so the cell does not separate the readings: {rival}"


@pytest.mark.parametrize("name", ["dirichlet_exit_1d", "dirichlet_exit_2d"])
def test_the_exit_drops_the_shared_value_and_a_wall_that_keeps_it_misses(name: str):
    cell = CELLS[name]
    n = cell.ns[-1]
    _solver, X, M = _solve(cell, n)
    exit_nodes = np.isclose(X[:, 0], 1.0)
    assert exit_nodes.any()
    assert np.abs(M[1:, exit_nodes]).max() == 0.0
    # The rival's exit nodes hold 0.7 by construction, so its miss is measured on every other node (the other
    # walls included).
    _rival, X, R = _solve(cell, n, _ExitKeepsTheValue)
    interior = ~np.isclose(X[:, 0], 1.0)  # every node but the exit's
    miss = max(np.abs(R[k][interior] - cell.m(k * T / NT, X)[interior]).max() for k in range(NT + 1))
    assert miss > 50 * MEASURED[name][-1], miss


@pytest.mark.parametrize("name", ["dirichlet_exit_1d", "dirichlet_exit_2d"])
def test_the_exit_loses_the_integrated_boundary_flux(name: str):
    """The mass identity beside the oracle: ``M(T) - M(0) = int S - int_exit J.n`` over the run, with the exact flux.
    The discrete change differs from it at the discretisation's order, so the band is relative to it."""
    cell = CELLS[name]
    solver, X, M = _solve(cell, cell.ns[-1])
    mass = lambda v: float((solver._M @ v).sum())  # noqa: E731
    dt = T / NT
    times = [(k + 1) * dt for k in range(NT)]
    sourced = sum(mass(np.asarray(cell.source(t, X), dtype=float)) for t in times) * dt
    if cell.dim == 1:
        out = sum(float(cell.flux(t, np.array([[1.0]]))[0, 0]) for t in times) * dt
    else:
        ys = np.linspace(0.0, LY, 401)
        face = np.stack([np.ones_like(ys), ys], axis=1)
        out = sum(float(np.trapezoid(cell.flux(t, face)[:, 0], ys)) for t in times) * dt
    loss = mass(M[-1]) - mass(M[0])
    expected = sourced - out
    assert abs(loss - expected) < MASS_BAND * abs(expected), (loss, expected)

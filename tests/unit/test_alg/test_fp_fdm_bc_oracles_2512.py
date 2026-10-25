"""FP-FDM against exact densities: its 2-D exit, its no-flux walls and its periodic seam, in 1-D and 2-D (#2512 (a)).

The FP equation is ``d_t m + div(alpha m) - D Lap m = S`` with ``alpha = -grad U`` from the potential the test hands
in (quadratic H, control cost 1), ``D = sigma^2 / 2`` and ``S`` the manufactured source, written in closed form (derived
with sympy and checked against it to 3e-15). Every solve takes the default path's drift arm, ``potential_field=U``,
and the default scheme, ``divergence_upwind``.

**Fixtures.**
- **No flux.** ``m = exp(a x/D) (1 + c t cos(pi x) [cos(pi y)])``, ``U = -a x``: the total flux is
  ``J = alpha m - D grad m = -D exp(a x/D) c t grad(cos(pi x) [cos(pi y)])``, zero on every wall, while the drift
  ``a = 0.4`` points into ``x_max`` and ``dm/dn != 0`` on both x walls.
- **Periodic.** ``m = 1 + 0.4 (1 + t) sin(2 pi x + 0.7) [sin(2 pi y + 0.3)]``, ``U = 0.1 sin(2 pi x) [+ 0.05 sin(2 pi
  y)]``: neither is even about the seam, and the drift crosses it, so the seam's velocity comes from differences of
  ``U`` across it.
- Both are linear in t and ``U`` does not depend on t, so implicit Euler adds no time error and the ratios are spatial.
- **Exit (2-D).** The shared BC is ``DIRICHLET(0.7)`` on ``x_max``, read at the FP as an absorbing exit with its value
  dropped (#2525), and NO_FLUX elsewhere; drift-free, ``m = cos(pi x/2) [exp(-D (pi/2)^2 t) + 0.5 cos(pi y) exp(-D
  ((pi/2)^2 + pi^2) t)]``, two decaying modes, with dt shrinking as h^2. The 1-D exit is
  `test_fp_shared_dirichlet_exit_2525`'s.

**Assertions.** Per cell: the error ratio between levels in a band, each level within a factor of 1.25 either way of the
value recorded here, and the wall-node error (the seam's, for periodic) within the same factor. The exit holds m = 0
at every step. The no-flux and periodic cells converge at first order: that is the upwind advection's, since the same
1-D no-flux fixture gives order 2.0 with ``a = 0`` and with ``divergence_centered`` (measured at cae35fa6).

**Rivals,** each measured on these inputs at cae35fa6 on a copy of the package, the finest level against the main
line's value:
- no_flux_1d (5.12e-2 at N = 81): the full-cell wall (``wall_vol = dx``) 9.02e-2; a ``dm/dn = 0`` wall, the drift's
  flux through the wall kept, 4.13. no_flux_2d (0.163 at N = 41): 0.220 and 5.92.
- periodic_1d (7.55e-3 at N = 81): the torus one cell too long (#1822: the axis span ``n`` for ``n - 1``) 6.73e-2;
  ``U``'s seam difference dropped 3.43e-2; no wrap at all stops in the mass gate (the density goes negative).
  periodic_2d (1.45e-2 at N = 41): 0.139 and 8.55e-2.
- dirichlet_exit_2d (3.11e-3 at N = 41): the shared value read literally (m = 0.7 at the exit) 0.70; the exit made a
  wall 0.55.

Held fixed: a scalar constant volatility (a tensor one is #2801), the ``potential_field`` arm (a ``drift_field`` array
takes the face-velocity arm, which the 1-D exit and the #1822 torus tests use; a callable ``drift_field`` is #2539),
``divergence_upwind``, implicit time stepping, a ``TensorProductGrid`` on the unit square.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc, periodic_bc

if TYPE_CHECKING:
    from collections.abc import Callable

SIGMA, T, NT = 0.6, 0.5, 10
D = SIGMA**2 / 2
A, C = 0.4, 0.5
LEVEL_FACTOR = 1.25


def _noflux_m(t: float, x: list[np.ndarray], d: int) -> np.ndarray:
    y = np.cos(np.pi * x[1]) if d == 2 else 1.0
    return np.exp(A * x[0] / D) * (1 + C * t * np.cos(np.pi * x[0]) * y)


def _noflux_source(t: float, x: list[np.ndarray], d: int) -> np.ndarray:
    y = np.cos(np.pi * x[1]) if d == 2 else 1.0
    psi = np.cos(np.pi * x[0]) * y
    psi_x = -np.pi * np.sin(np.pi * x[0]) * y
    return np.exp(A * x[0] / D) * (C * psi - C * t * (A * psi_x - D * np.pi**2 * d * psi))


def _noflux_u(x: list[np.ndarray], d: int) -> np.ndarray:
    return -A * x[0]


def _periodic_m(t: float, x: list[np.ndarray], d: int) -> np.ndarray:
    q = np.sin(2 * np.pi * x[1] + 0.3) if d == 2 else 1.0
    return 1 + 0.4 * (1 + t) * np.sin(2 * np.pi * x[0] + 0.7) * q


def _periodic_source(t: float, x: list[np.ndarray], d: int) -> np.ndarray:
    q = np.sin(2 * np.pi * x[1] + 0.3) if d == 2 else 1.0
    s = np.sin(2 * np.pi * x[0] + 0.7) * q
    grad_s = [2 * np.pi * np.cos(2 * np.pi * x[0] + 0.7) * q]
    alpha = [-0.2 * np.pi * np.cos(2 * np.pi * x[0])]
    div_alpha = 0.4 * np.pi**2 * np.sin(2 * np.pi * x[0])
    if d == 2:
        grad_s.append(np.sin(2 * np.pi * x[0] + 0.7) * 2 * np.pi * np.cos(2 * np.pi * x[1] + 0.3))
        alpha.append(-0.1 * np.pi * np.cos(2 * np.pi * x[1]))
        div_alpha = div_alpha + 0.2 * np.pi**2 * np.sin(2 * np.pi * x[1])
    m = 1 + 0.4 * (1 + t) * s
    advection = div_alpha * m + 0.4 * (1 + t) * sum(a * g for a, g in zip(alpha, grad_s, strict=True))
    return 0.4 * s + advection + D * 0.4 * (1 + t) * 4 * np.pi**2 * d * s


def _periodic_u(x: list[np.ndarray], d: int) -> np.ndarray:
    return 0.1 * np.sin(2 * np.pi * x[0]) + (0.05 * np.sin(2 * np.pi * x[1]) if d == 2 else 0.0)


@dataclass(frozen=True)
class _Cell:
    d: int
    bc: Callable[[], BoundaryConditions]
    m: Callable
    source: Callable
    u: Callable
    levels: tuple[int, ...]
    errors: tuple[float, ...]  # max error at T, recorded at cae35fa6
    wall_errors: tuple[float, ...]  # max error at T on the wall (seam) nodes, recorded at cae35fa6
    ratio: tuple[float, float]


CELLS = {
    "no_flux_1d": _Cell(
        d=1,
        bc=lambda: no_flux_bc(dimension=1),
        m=_noflux_m,
        source=_noflux_source,
        u=_noflux_u,
        levels=(21, 41, 81),
        errors=(2.0099e-01, 1.0175e-01, 5.1186e-02),
        wall_errors=(2.0099e-01, 1.0175e-01, 5.1186e-02),
        ratio=(1.6, 2.4),
    ),
    "no_flux_2d": _Cell(
        d=2,
        bc=lambda: no_flux_bc(dimension=2),
        m=_noflux_m,
        source=_noflux_source,
        u=_noflux_u,
        levels=(11, 21, 41),
        errors=(5.6022e-01, 3.1060e-01, 1.6302e-01),
        wall_errors=(5.6022e-01, 3.1060e-01, 1.6302e-01),
        ratio=(1.6, 2.4),
    ),
    "periodic_1d": _Cell(
        d=1,
        bc=lambda: periodic_bc(dimension=1),
        m=_periodic_m,
        source=_periodic_source,
        u=_periodic_u,
        levels=(21, 41, 81),
        errors=(2.4077e-02, 1.4045e-02, 7.5478e-03),
        wall_errors=(1.5558e-02, 8.5149e-03, 4.4592e-03),
        ratio=(1.5, 2.5),
    ),
    "periodic_2d": _Cell(
        d=2,
        bc=lambda: periodic_bc(dimension=2),
        m=_periodic_m,
        source=_periodic_source,
        u=_periodic_u,
        levels=(11, 21, 41),
        errors=(6.5435e-02, 3.0335e-02, 1.4547e-02),
        wall_errors=(1.9692e-02, 1.4166e-02, 8.0239e-03),
        ratio=(1.5, 2.5),
    ),
}


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)


def _problem(d: int, n: int, nt: int, bc: BoundaryConditions) -> tuple[TensorProductGrid, MFGProblem]:
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * d, Nx_points=[n] * d, boundary_conditions=bc)
    model = Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=SIGMA)
    conditions = Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=T)
    return grid, MFGProblem(model=model, domain=grid, conditions=conditions, Nt=nt)


def _on_wall(shape: tuple[int, ...]) -> np.ndarray:
    idx = np.indices(shape)
    return np.any((idx == 0) | (idx == np.array(shape).reshape(-1, *([1] * len(shape))) - 1), axis=0)


def _solve(cell: _Cell, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The density at T and the exact one, on the grid's nodes."""
    _, problem = _problem(cell.d, n, NT, cell.bc())
    shape = (n,) * cell.d
    axes = np.meshgrid(*[np.linspace(0.0, 1.0, n)] * cell.d, indexing="ij")
    u = np.broadcast_to(np.asarray(cell.u(axes, cell.d), dtype=float), shape)

    def source(t: float, x: np.ndarray) -> np.ndarray:
        points = np.asarray(x, dtype=float).reshape(-1, cell.d)
        return np.asarray(cell.source(t, list(points.T), cell.d), dtype=float).reshape(-1)

    m = FPFDMSolver(problem).solve_fp_system(
        cell.m(0.0, axes, cell.d),
        potential_field=np.broadcast_to(u, (NT + 1, *shape)).copy(),
        source_term=source,
        show_progress=False,
    )
    return np.asarray(m)[-1], cell.m(T, axes, cell.d)


@pytest.mark.parametrize("name", sorted(CELLS))
def test_an_exact_density_is_reproduced_at_its_rate_level_and_wall(name: str):
    cell = CELLS[name]
    errors, walls = [], []
    for n in cell.levels:
        m, exact = _solve(cell, n)
        error = np.abs(m - exact)
        errors.append(error.max())
        walls.append(error[_on_wall(error.shape)].max())
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(cell.ratio[0] < r < cell.ratio[1] for r in ratios), (name, ratios)
    for got, recorded in zip(errors + walls, cell.errors + cell.wall_errors, strict=True):
        assert recorded / LEVEL_FACTOR < got < recorded * LEVEL_FACTOR, (name, errors, walls)


def _exit_bc() -> BoundaryConditions:
    return BoundaryConditions(
        dimension=2,
        segments=[BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.7, boundary="x_max")],
        default_bc=BCType.NO_FLUX,
    )


def _exit_m(t: float, x: list[np.ndarray]) -> np.ndarray:
    k1, k2 = (np.pi / 2) ** 2, (np.pi / 2) ** 2 + np.pi**2
    return np.cos(np.pi * x[0] / 2) * (np.exp(-D * k1 * t) + 0.5 * np.cos(np.pi * x[1]) * np.exp(-D * k2 * t))


def test_the_2d_exit_absorbs_and_matches_the_decaying_modes():
    """The exit nodes hold 0 at every step after t = 0 (the shared 0.7 read literally holds 0.7 there); the density
    matches at second order, each level within the factor of its recorded value (1.2079e-2, 3.1051e-3 at cae35fa6)."""
    errors = []
    for n, nt in ((21, 10), (41, 40)):
        _, problem = _problem(2, n, nt, _exit_bc())
        axes = np.meshgrid(*[np.linspace(0.0, 1.0, n)] * 2, indexing="ij")
        m = np.asarray(
            FPFDMSolver(problem).solve_fp_system(
                _exit_m(0.0, axes), potential_field=np.zeros((nt + 1, n, n)), show_progress=False
            )
        )
        assert np.all(m[1:, -1, :] == 0.0), "the exit wall must hold m = 0 at every step"
        errors.append(np.abs(m[-1] - _exit_m(T, axes)).max())
    assert 3.0 < errors[0] / errors[1] < 5.0, errors
    for got, recorded in zip(errors, (1.2079e-02, 3.1051e-03), strict=True):
        assert recorded / LEVEL_FACTOR < got < recorded * LEVEL_FACTOR, errors

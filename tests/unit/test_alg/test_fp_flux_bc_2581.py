"""A prescribed flux, ``J.n = q`` (``BCType.FLUX``, ``flux_bc``), on FP-FDM and FP-FEM (#2581 (B), #2512 (a)).

``J = alpha m - D grad m`` with ``alpha = -grad U`` (quadratic H, control cost 1) and ``D = sigma^2 / 2``; ``q`` is mass
per unit time per unit boundary measure, outward positive (user ruling 2026-10-09, B-1). Each solve hands the FP solver
its own BC; the problem's geometry carries NO_FLUX, the HJB's, since a FLUX on the shared BC is refused (B-2).

**The cells** (``fp_fdm/flux_{1,2}d``, ``fp_fem/flux_{1,2}d``). A constant drift ``v`` along x, ``J.n = -q`` on x_min
(inflow) and ``+q`` on x_max (outflow), NO_FLUX on y. The steady state of ``v m - D m' = q`` is
``m = q/v + C exp(v x / D)``, exact, with ``C`` set by the mass per unit length in y. Each solve starts there and runs
to T; the error is the density's distance from it at T, which is the scheme's steady error plus what remains of the
transient. FP-FDM converges at first order (the upwind advection's), FP-FEM P1 at second order in 1-D and near it in
2-D. **The rival**, in each cell: the drift dropped at the wall, so that the face carries ``q`` plus the advective
flux ``v.n m`` -- a Neumann condition on m, ``-D dm/dn = q``. On FP-FDM it adds ``v_out/(dx/2)`` to each wall row's
diagonal (``_BOUNDARY_HANDLERS``); on FP-FEM it is the transposed advection of `test_fp_fem_bc_oracles_2512`, whose
natural condition is ``D dm/dn`` in place of ``J.n``. It misses by O(1) at every level. **Pure diffusion** is a second
control: with ``v = 0`` the rival and the FLUX solve coincide bit for bit, which is why the cells carry a drift.

**The mass** follows the flux exactly, ``M(t_n) = M(0) - sum_k dt int_Gamma q(t_k)``, at the level each implicit step
reads, ``t_{k+1}``, and on the scheme's own face quadrature: FP-FDM's nodal values on the trapezoid face weights,
FP-FEM's facet quadrature. ``q = 0`` solves as NO_FLUX bit for bit.

**A manufactured solution** carries FLUX on every face, with q the exact ``J.n`` in t and along each face, through each
face's own q at the corners: the mass tests integrate q over a face, so a q read at a place along it that keeps the
integral -- reflected along the face -- passes them.

**B-3** (user rulings 2026-10-09 and 2026-10-30; the approver's on #2581's plan, Q3, and on #2840): a step is refused,
not clipped past the gate, where the clip would fabricate mass the flux decides -- first, (1) the negatives the flux's
part makes at nodes the step without it leaves non-negative fabricate more than the gate allows; else (2) the step
without it passes the clip gate and either (a) the flux deepens its negatives past it, counted without netting, on the
measure of the step without the flux, or (b) the step with the flux fails the gate itself. Any other step whose own clip
is past the gate is the scheme's, flux included.
One owner reads it for both families (`refuse_what_a_flux_takes_negative`). On ``divergence_upwind`` the step's matrix
A has non-positive off-diagonals and, as the step conserves the trapezoid mass, ``w^T A = w^T/dt > 0`` on the
trapezoid weights w: it is an M-matrix, so the step without the flux, right-hand side ``m_old/dt >= 0``, is
non-negative. Its rows need not be dominant (a row sum of -132 under a drift that piles mass up), so no discrete
maximum principle puts the first negative on a FLUX face. Measured, not proved: none off the face in a seeded 1-D scan
of 300 steps (review 1 of #2840), against 3 on ``divergence_centered``. ``divergence_centered`` and FP-FEM
are not M-matrices, and the rows below take a step whose first negative is off the face, at the opposite wall, while
every FLUX node stays non-negative: there a check of the FLUX nodes alone would solve on.

Held fixed: a constant scalar volatility; a ``U`` linear in x and constant in t; FLUX on whole box faces (FP-FDM reads one
condition per face, #2512 row B1). The fuzz before review 1 covers the rest.
"""

from __future__ import annotations

import logging

import pytest
import skfem

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver
from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
from mfgarchon.alg.numerical.fp_solvers import fp_fdm_time_stepping
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid, flux_bc, no_flux_bc
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
from mfgarchon.geometry.meshes.mesh_1d import Mesh1D
from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

SIGMA, V, Q, LY, T, NT = 0.6, 0.5, 0.2, 0.5, 2.0, 40
D = SIGMA**2 / 2
LEVEL_FACTOR = 1.25
FACES = {1: ("x_min", "x_max"), 2: ("x_min", "x_max", "y_min", "y_max")}

#: (family, dimension, levels, errors at T recorded at f7d6f1cf/35452acd's library, ratio band)
CELLS = {
    "fdm_1d": ("fdm", 1, (21, 41, 81), (9.0306e-02, 4.7406e-02, 2.4294e-02), (1.6, 2.4)),
    "fdm_2d": ("fdm", 2, (11, 21, 41), (4.3772e-01, 2.4082e-01, 1.2642e-01), (1.6, 2.4)),
    "fem_1d": ("fem", 1, (10, 20, 40), (9.1907e-03, 2.3038e-03, 5.7634e-04), (3.5, 4.5)),
    "fem_2d": ("fem", 2, (8, 16, 32), (7.2057e-02, 2.0746e-02, 5.8597e-03), (3.0, 4.2)),
}


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)


def _model(sigma: float = SIGMA) -> Model:
    return Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=sigma)


def _problem(domain, nt: int = NT, t_end: float = T, sigma: float = SIGMA) -> MFGProblem:
    """Each solve is handed its own density; the problem's is a constant whose mass reads 1 on the domain's measure:
    the grid's area, Mesh1D's uniform-cell measure, (n + 1)/n for n cells, and Mesh2D's point average."""
    if isinstance(domain, Mesh1D):
        level = domain.num_elements / (domain.num_elements + 1)
    elif isinstance(domain, Mesh2D):
        level = 1.0
    else:
        level = 1.0 / (LY if domain.dimension == 2 else 1.0)
    conditions = Conditions(m_initial=lambda x, c=level: c, u_terminal=lambda x: 0.0, T=t_end)
    return MFGProblem(model=_model(sigma), domain=domain, conditions=conditions, Nt=nt)


def _segments(dim: int, values: dict[str, float]) -> BoundaryConditions:
    """FLUX with ``values[face]`` on the faces it names, NO_FLUX on the others."""
    return BoundaryConditions(
        dimension=dim,
        segments=[
            BCSegment(
                name=f, bc_type=BCType.FLUX if f in values else BCType.NO_FLUX, value=values.get(f, 0.0), boundary=f
            )
            for f in FACES[dim]
        ],
    )


def _steady(x: np.ndarray, v: float, per_length: float) -> np.ndarray:
    """``m = q/v + C exp(v x / D)``, J = q along x, with mass ``per_length`` per unit length in y (v != 0), or the
    linear ``m = a - (q/D) x`` at v = 0."""
    if v == 0.0:
        return per_length + Q / (2 * D) - (Q / D) * x
    c = (per_length - Q / v) / ((D / v) * (np.exp(v / D) - 1))
    return Q / v + c * np.exp(v * x / D)


def _grid(dim: int, n: int) -> TensorProductGrid:
    return TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, LY)][:dim], Nx_points=[n] * dim, boundary_conditions=no_flux_bc(dimension=dim)
    )


def _mesh(dim: int, n: int, ly: float = LY):
    if dim == 1:
        mesh = Mesh1D(bounds=(0.0, 1.0), num_elements=n)
        mesh.generate_mesh()
    else:
        mesh = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, ly))
        mesh.mesh_data = skfem_to_meshdata(
            skfem.MeshTri.init_tensor(np.linspace(0.0, 1.0, n + 1), np.linspace(0.0, ly, n + 1))
        )
    mesh.boundary_conditions = BoundaryConditions(
        dimension=dim, segments=[BCSegment(name=f, bc_type=BCType.NO_FLUX, boundary=f) for f in FACES[dim]]
    )
    return mesh


def _fdm(dim: int, n: int, bc: BoundaryConditions, v: float = V, m0=None, **kwargs):
    """(the grid's node axes as meshgrid arrays, the solve's densities)."""
    grid = _grid(dim, n)
    axes = np.meshgrid(*[np.asarray(a) for a in grid.coordinates], indexing="ij")
    if m0 is None:
        m0 = _steady(axes[0], v, 1.0 / (LY if dim == 2 else 1.0))
    u = np.broadcast_to(-v * axes[0], (NT + 1, *axes[0].shape)).copy()
    m = FPFDMSolver(_problem(grid), boundary_conditions=bc, **kwargs).solve_fp_system(
        m0, potential_field=u, show_progress=False
    )
    return axes, np.asarray(m)


def _fem(dim: int, n: int, bc: BoundaryConditions, v: float = V, solver_cls=FPFEMSolver, m0=None):
    """(the solver, its DOF coordinates, the solve's densities)."""
    solver = solver_cls(_problem(_mesh(dim, n)), boundary_conditions=bc)
    x = solver._disc.dof_coordinates
    if m0 is None:
        m0 = _steady(x[:, 0], v, 1.0 / (LY if dim == 2 else 1.0))
    potential = np.broadcast_to(-v * x[:, 0], (NT + 1, x.shape[0])).copy()
    return solver, x, np.asarray(solver.solve_fp_system(M_initial=m0, potential_field=potential))


def _cell_error(name: str, n: int, v: float = V, fem_cls=FPFEMSolver) -> float:
    family, dim = CELLS[name][0], CELLS[name][1]
    bc = _segments(dim, {"x_min": -Q, "x_max": Q})
    if family == "fdm":
        _, m = _fdm(dim, n, bc, v)
        return float(np.abs(m[-1] - m[0]).max())
    _, _, m = _fem(dim, n, bc, v, solver_cls=fem_cls)
    return float(np.abs(m[-1] - m[0]).max())


class _FemWallDropsTheDrift(FPFEMSolver):
    """The rival FP-FEM wall: the advection transposed back to ``int phi_i (v . grad phi_j)``, whose natural condition is
    ``D dm/dn``, so the FLUX load imposes ``-D dm/dn = q`` (`test_fp_fem_bc_oracles_2512`'s ``dm/dn`` rival)."""

    def _build_advection(self, U_n, D=0.0):
        return -super()._build_advection(U_n, D).T


def _fdm_wall_drops_the_drift(monkeypatch) -> None:
    """The rival FP-FDM wall: each wall row also takes the advective flux through the wall, ``v_out m``, with ``v_out``
    that wall's outward velocity, so the FLUX source imposes the diffusive flux alone. Its mass then moves at
    ``v (m(x_min) - m(x_max))`` per unit time beside the FLUX's, as FP-FEM's rival's does."""
    original = fp_fdm_time_stepping._BOUNDARY_HANDLERS["divergence_upwind"]

    def handler(rows, cols, data, flat, index, shape, ndim, dt, sigma, c, spacing, u, grid, interface_velocity=None):
        original(rows, cols, data, flat, index, shape, ndim, dt, sigma, c, spacing, u, grid, interface_velocity)
        for d in range(ndim):
            dx = spacing[d]
            if index[d] in (0, shape[d] - 1):
                neighbour = list(index)
                neighbour[d] += 1 if index[d] == 0 else -1
                # alpha.n = -c du/dn, one-sided: (u_neighbour - u_wall)/dx at either wall
                v_out = c * (u[grid.get_index(tuple(neighbour))] - u[flat]) / dx
                rows.append(flat)
                cols.append(flat)
                data.append(v_out / (dx / 2))

    monkeypatch.setitem(fp_fdm_time_stepping._BOUNDARY_HANDLERS, "divergence_upwind", handler)


# ---------------------------------------------------------------------------------------------------------- the cells


@pytest.mark.parametrize("name", sorted(CELLS))
def test_a_prescribed_flux_holds_the_steady_profile_at_its_order(name: str):
    _, _, levels, recorded, (low, high) = CELLS[name]
    errors = [_cell_error(name, n) for n in levels]
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(low < r < high for r in ratios), (name, errors, ratios)
    for got, want in zip(errors, recorded, strict=True):
        assert want / LEVEL_FACTOR < got < want * LEVEL_FACTOR, (name, errors)


@pytest.mark.parametrize("name", sorted(CELLS))
def test_the_flux_read_with_the_drift_dropped_at_the_wall_misses(name: str, monkeypatch):
    family, _, levels, _, _ = CELLS[name]
    if family == "fdm":
        _fdm_wall_drops_the_drift(monkeypatch)
        errors = [_cell_error(name, n) for n in levels]
    else:
        errors = [_cell_error(name, n, fem_cls=_FemWallDropsTheDrift) for n in levels]
    # Measured, both families: 1.58 to 1.65 (1-D) and 3.47 to 3.48 (2-D) at every level, flat, against the cell's own
    # 5.8e-4 to 0.44. A rival built with the wrong sign at one wall grew from 829 to 2459 and passed a bare `> 1`.
    low, high = (1.4, 1.8) if CELLS[name][1] == 1 else (3.2, 3.8)
    assert all(low < e < high for e in errors), (name, errors)
    assert max(errors) < 1.1 * min(errors), (name, errors)


@pytest.mark.parametrize("family", ["fdm", "fem"])
def test_with_no_drift_the_flux_and_the_drift_dropped_wall_coincide(family: str, monkeypatch):
    """The cells' second control: with v = 0 there is no advective flux to drop, so the rival is the FLUX solve."""
    bc = _segments(1, {"x_min": -Q, "x_max": Q})
    if family == "fdm":
        _, kept = _fdm(1, 21, bc, v=0.0)
        _fdm_wall_drops_the_drift(monkeypatch)
        _, dropped = _fdm(1, 21, bc, v=0.0)
    else:
        _, _, kept = _fem(1, 10, bc, v=0.0)
        _, _, dropped = _fem(1, 10, bc, v=0.0, solver_cls=_FemWallDropsTheDrift)
    assert np.array_equal(kept, dropped)
    assert float(np.abs(kept[-1] - kept[0]).max()) < 1e-2


# ----------------------------------------------------------------------------------------------------------- the mass


QM = 0.05  # the mass tests' rate: the outflow takes 0.2 of the mass by T, short of draining the wall (B-3)


def _q_t(t, x):
    return QM * (1.0 + t)


def _q_y(t, x):
    return QM * (1.0 + 2.0 * float(np.asarray(x)[-1]))


@pytest.mark.parametrize(
    ("family", "dim", "q"), [("fdm", 1, _q_t), ("fdm", 2, _q_y), ("fem", 1, _q_t), ("fem", 2, _q_y)]
)
def test_the_mass_leaves_at_the_prescribed_rate(family: str, dim: int, q):
    """Outflow on x_max only: the mass loses sum_k dt int q(t_{k+1}) exactly, on the scheme's own face quadrature."""
    bc = _segments(dim, {"x_max": q})
    dt = T / NT
    if family == "fdm":
        _, m = _fdm(dim, 21, bc)
        grid = _grid(dim, 21)
        mass = [float(np.sum(level * grid.measure_weights())) for level in m]
        # FP-FDM's face quadrature: q at the face's nodes, on the trapezoid weights along it.
        y = np.asarray(grid.coordinates[1]) if dim == 2 else np.zeros(1)
        face = np.ones(1) if dim == 1 else np.full(y.shape, y[1] - y[0])
        if dim == 2:
            face[[0, -1]] /= 2
        rate = [
            sum(face[j] * q(k * dt, np.array([1.0, y[j]][:dim])) for j in range(face.size)) for k in range(1, NT + 1)
        ]
    else:
        solver, _, m = _fem(dim, 10, bc)
        mass = [float((solver._M @ level).sum()) for level in m]
        # Constant on a 1-D point, linear in y on the 2-D face: the facet quadrature integrates both exactly.
        rate = [q(k * dt, np.ones(1)) if dim == 1 else QM * (LY + LY**2) for k in range(1, NT + 1)]
    expected = mass[0] - np.concatenate([[0.0], np.cumsum(np.asarray(rate) * dt)])
    assert np.max(np.abs(np.asarray(mass) - expected)) < 1e-12, (family, dim)


@pytest.mark.parametrize(("family", "dim"), [("fdm", 1), ("fdm", 2), ("fem", 1), ("fem", 2)])
def test_a_zero_flux_solves_as_no_flux_bit_for_bit(family: str, dim: int):
    zero = _segments(dim, {"x_min": 0.0, "x_max": 0.0})
    wall = _segments(dim, {})
    if family == "fdm":
        assert np.array_equal(_fdm(dim, 11, zero)[1], _fdm(dim, 11, wall)[1])
    else:
        assert np.array_equal(_fem(dim, 8, zero)[2], _fem(dim, 8, wall)[2])


@pytest.mark.parametrize("family", ["fdm", "fem"])
def test_a_dirichlet_face_beside_a_flux_face_takes_nothing_from_it(family: str):
    """Dirichlet 0.3 on y_min, FLUX on x_max, meeting at a corner. The corner is the Dirichlet face's -- FP-FDM's
    identity row takes no FLUX source, and FP-FEM solves the flux's part with the Dirichlet values homogeneous -- so
    every node on y_min holds, bit for bit, what it holds with q = 0: 0.3 to round-off."""

    def bc(q: float) -> BoundaryConditions:
        kinds = {"x_min": BCType.NO_FLUX, "x_max": BCType.FLUX, "y_min": BCType.DIRICHLET, "y_max": BCType.NO_FLUX}
        values = {"x_max": q, "y_min": 0.3}
        return BoundaryConditions(
            dimension=2,
            segments=[BCSegment(name=f, bc_type=k, value=values.get(f, 0.0), boundary=f) for f, k in kinds.items()],
        )

    if family == "fdm":
        axes, m = _fdm(2, 11, bc(QM))
        on_wall = np.isclose(axes[1], 0.0)
        held, without = m[1:, on_wall], _fdm(2, 11, bc(0.0))[1][1:, on_wall]
    else:
        _, x, m = _fem(2, 8, bc(QM))
        on_wall = np.isclose(x[:, 1], 0.0)
        held, without = m[1:, on_wall], _fem(2, 8, bc(0.0))[2][1:, on_wall]
    assert held.size > 0
    assert np.array_equal(held, without), float(np.abs(held - without).max())
    assert np.abs(held - 0.3).max() < 1e-12


@pytest.mark.parametrize(("family", "dim"), [("fdm", 1), ("fdm", 2), ("fem", 1), ("fem", 2)])
def test_a_uniform_flux_bc_takes_q_through_every_face(family: str, dim: int):
    """``flux_bc(q)``, the form the shared BC's refusal tells the caller to pass: the mass leaves at q times the
    boundary's measure. On arrival FP-FDM cached a Laplacian for every uniform BC, which has no reading of FLUX, and
    so refused it on the step that reads it. q is a quarter of the mass tests': every face drains, a corner from two
    sides, and B-3 refuses the 2-D FP-FEM corner at QM before T."""
    q = QM / 4
    bc = flux_bc(q, dimension=dim)
    boundary = 2.0 if dim == 1 else 2.0 * (1.0 + LY)
    if family == "fdm":
        grid = _grid(dim, 21)
        _, m = _fdm(dim, 21, bc)
        mass = [float(np.sum(level * grid.measure_weights())) for level in m]
    else:
        solver, _, m = _fem(dim, 10, bc)
        mass = [float((solver._M @ level).sum()) for level in m]
    assert np.max(np.abs(np.diff(mass) + q * boundary * T / NT)) < 1e-12, (family, dim)


# ---------------------------------------------------------------------------------------------- a manufactured solution

#: ``m = (1.2 + sin(pi x/2 + 0.3) cos(pi y + 0.2)) e^{-0.3 t}`` under the drift ``(0.7, -0.4)``, with sigma 0.4: no
#: symmetry in y, so a q read at the wrong place along a face, at the wrong level, or a corner given one face's q, is
#: off the solution. (family, levels, steps per level, errors at T recorded at this file's commit, ratio band)
MMS_DRIFT, MMS_SIGMA, MMS_T = (0.7, -0.4), 0.4, 0.5
MMS = {
    "fdm": ((11, 21, 41), lambda n: 2 * (n - 1), (1.9973e-01, 1.1046e-01, 5.7990e-02), (1.6, 2.2)),
    "fem": ((8, 16, 32), lambda n: 2 * n, (6.7995e-02, 1.6598e-02, 3.3813e-03), (3.5, 5.5)),
}


def _mms(t, x, y):
    """(m, dm/dx, dm/dy, laplacian of m)."""
    e = np.exp(-0.3 * t)
    sx, cx = np.sin(np.pi * x / 2 + 0.3), np.cos(np.pi * x / 2 + 0.3)
    sy, cy = np.sin(np.pi * y + 0.2), np.cos(np.pi * y + 0.2)
    return (1.2 + sx * cy) * e, (np.pi / 2) * cx * cy * e, -np.pi * sx * sy * e, -1.25 * np.pi**2 * sx * cy * e


def _mms_source(t, x):
    """``S = m_t + alpha . grad m - D lap m``, for a constant alpha."""
    x = np.asarray(x)
    m, mx, my, lap = _mms(t, x[:, 0], x[:, 1])
    return -0.3 * m + MMS_DRIFT[0] * mx + MMS_DRIFT[1] * my - (MMS_SIGMA**2 / 2) * lap


def _mms_flux(face: str):
    """``q = (alpha m - D grad m) . n`` on ``face``, n outward: the exact flux, each face its own at a corner."""
    normal = {"x_min": (-1.0, 0.0), "x_max": (1.0, 0.0), "y_min": (0.0, -1.0), "y_max": (0.0, 1.0)}[face]

    def q(t, x):
        m, mx, my, _ = _mms(t, float(x[0]), float(x[1]))
        d = MMS_SIGMA**2 / 2
        return (MMS_DRIFT[0] * m - d * mx) * normal[0] + (MMS_DRIFT[1] * m - d * my) * normal[1]

    return q


def _mms_error(family: str, n: int) -> float:
    _, steps, _, _ = MMS[family]
    nt = steps(n)
    bc = BoundaryConditions(
        dimension=2, segments=[BCSegment(name=f, bc_type=BCType.FLUX, value=_mms_flux(f), boundary=f) for f in FACES[2]]
    )
    if family == "fdm":
        grid = _grid(2, n)
        x, y = np.meshgrid(*[np.asarray(a) for a in grid.coordinates], indexing="ij")
        u = np.broadcast_to(-(MMS_DRIFT[0] * x + MMS_DRIFT[1] * y), (nt + 1, *x.shape)).copy()
        solver = FPFDMSolver(_problem(grid, nt=nt, t_end=MMS_T, sigma=MMS_SIGMA), boundary_conditions=bc)
        m = solver.solve_fp_system(_mms(0.0, x, y)[0], potential_field=u, source_term=_mms_source, show_progress=False)
    else:
        solver = FPFEMSolver(_problem(_mesh(2, n), nt=nt, t_end=MMS_T, sigma=MMS_SIGMA), boundary_conditions=bc)
        x, y = solver._disc.dof_coordinates[:, 0], solver._disc.dof_coordinates[:, 1]
        u = np.broadcast_to(-(MMS_DRIFT[0] * x + MMS_DRIFT[1] * y), (nt + 1, x.size)).copy()
        m = solver.solve_fp_system(M_initial=_mms(0.0, x, y)[0], potential_field=u, source_term=_mms_source)
    return float(np.abs(np.asarray(m)[-1] - _mms(MMS_T, x, y)[0]).max())


@pytest.mark.parametrize("family", sorted(MMS))
def test_a_manufactured_solution_with_a_flux_on_every_face_converges(family: str):
    """FLUX on all four faces, q the exact ``J.n`` varying in t and along each face, dt proportional to h. The mass tests
    integrate q over a face, so a q read reflected along it passes them; this solution does not (reviews 1 and 2 of
    #2840)."""
    levels, _, recorded, (low, high) = MMS[family]
    errors = [_mms_error(family, n) for n in levels]
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(low < r < high for r in ratios), (family, errors, ratios)
    for got, want in zip(errors, recorded, strict=True):
        assert want / LEVEL_FACTOR < got < want * LEVEL_FACTOR, (family, errors)


# ------------------------------------------------------------------------------------------------------------- B-3


def test_the_b3_owner_refuses_the_clip_the_flux_decides_and_leaves_the_scheme_its_own():
    """B-3's one owner, read by both families (user rulings 2026-10-09 and 2026-10-30; the approver's on #2840), on steps
    whose verdict the rule fixes. With F the negatives the flux alone makes, D its deepening of the step's own, both on
    the caller's weights: (1) F / pos(step) > thr, first; (2) s0 <= thr and either (a) s0 + D / pos(step) > thr, or (b)
    the step with the flux fails the gate itself. Otherwise the step is the scheme's, returned unclipped. Each step
    below is chosen so that one wrong reading gives the other verdict, or another message."""
    from mfgarchon.utils.numerical.mass_fabrication_gate import refuse_what_a_flux_takes_negative as owner

    kw = {
        "context": "C",
        "time": 0.1,
        "unit": "node",
        "locate": lambda i: f"node {i}",
        "remedy": "R",
        "remedy_past_gate": "P",
    }
    adds = r"the flux's part deepens its negatives by {} more, past it.*B-3"
    fails = r"clipping the step with it would fabricate {}, past it.*B-3"
    # node 3, which the flux takes to 0 and not below it, is not counted
    with pytest.raises(ValueError, match=r"at 1 node\(s\) in the step to t=0\.1, the most at node 1 .*B-3\)\. R$"):
        owner(np.array([1.0, 0.0, 1.0, 1e-3]), np.array([0.0, -1e-3, 0.0, -1e-3]), **kw)
    # (1) at its boundary: F / pos(step) = thr exactly is not past it (nor are (2a) and (2b)); 2e-8 is
    assert np.array_equal(owner(np.array([1.0, 0.0]), np.array([0.0, -1e-8]), **kw), np.array([1.0, -1e-8]))
    with pytest.raises(ValueError, match=r"fabricate 2\.000e-08 of the mass, past the threshold.*B-3\)\. R$"):
        owner(np.array([1.0, 0.0]), np.array([0.0, -2e-8]), **kw)
    # round-off at a node the step without the flux leaves at 0 is within the gate, and clipped as any round-off
    assert np.array_equal(owner(np.array([1.0, 0.0]), np.array([0.0, -1e-22]), **kw), np.array([1.0, -1e-22]))
    # a node the flux takes to zero, not below it (1e-3 - 1e-3 is 0.0 exactly)
    assert np.array_equal(owner(np.array([1.0, 1e-3]), np.array([0.0, -1e-3]), **kw), np.array([1.0, 0.0]))
    # (1) before the scheme: s0 = 1e-3 is past the gate; the caller's clip decides what a change of q does there
    with pytest.raises(ValueError, match=r"The step without the flux would itself fabricate 1\.000e-03") as caught:
        owner(np.array([1.0, 0.0, -1e-3]), np.array([0.0, -1e-3, 0.0]), **kw)
    assert str(caught.value).endswith("before this step's flux is added. P")
    # s0 = thr exactly is not past the gate, so (1)'s message names the flux's remedy
    with pytest.raises(ValueError, match=r"fabricate 1\.000e-03 of the mass, past the threshold.*B-3\)\. R$"):
        owner(np.array([1.0, 0.0, -1e-8]), np.array([0.0, -1e-3, 0.0]), **kw)
    # F on a step with no positive mass is infinite, not 0
    with pytest.raises(ValueError, match=r"Clipping that would fabricate inf of the mass"):
        owner(np.zeros(2), np.array([0.0, -1.0]), **kw)
    # (2a) at the gate: s0 = thr exactly is the owner's, and the flux's 1e-12 takes it past
    with pytest.raises(ValueError, match=adds.format(r"1\.000e-12")):
        owner(np.array([1.0, -1e-8]), np.array([0.0, -1e-12]), **kw)
    # exactly at the gate is not past it, by either branch
    assert np.array_equal(owner(np.array([1.0, -1e-8]), np.array([0.0, 0.0]), **kw), np.array([1.0, -1e-8]))
    assert np.array_equal(owner(np.array([1.0, -5e-9]), np.array([0.0, -5e-9]), **kw), np.array([1.0, -1e-8]))
    # D on the step that holds it: deepening by 6e-9 while doubling the mass gives s0 + D / pos(step) = 8e-9 and
    # s_step = 5.5e-9, within the gate; on pos(m0), withdrawn, it was 1.1e-8 (the user's ruling, review 4 of #2840)
    m0, m_q = np.array([1.0, -0.5e-8]), np.array([1.0, -0.6e-8])
    assert np.array_equal(owner(m0, m_q, **kw), m0 + m_q)
    # and where the flux drains the step: D = 3e-9 on pos(step) = 0.5 takes s0 = 6e-9 to 1.2e-8, though s_step = 8e-9;
    # on pos(m0) it was 9e-9 (review 5 of #2840)
    with pytest.raises(ValueError, match=adds.format(r"6\.000e-09")):
        owner(np.array([1.0, -1e-9, -5e-9]), np.array([-0.5, -3e-9, 5e-9]), **kw)
    # D on a step with no positive mass is infinite, as F is
    with pytest.raises(ValueError, match=adds.format("inf")):
        owner(np.array([1.0, -1e-9]), np.array([-1.0, -1e-9]), **kw)
    # F is counted once, by (1): F / pos(step) = 3e-9, s0 + D / pos(step) = 9e-9 and s_step = 9e-9 are each within the
    # gate; F in (2a) too would make it 1.2e-8 (the approver's ruling on #2840)
    m0, m_q = np.array([1.0, -0.3e-8, 0.0, -0.3e-8]), np.array([0.0, -0.3e-8, -0.3e-8, 0.3e-8])
    assert np.array_equal(owner(m0, m_q, **kw), m0 + m_q)
    # (2b) alone: a flux that drains 60% of the positive mass, deepening nothing (s0 + D = 5e-9), fails the gate
    with pytest.raises(ValueError, match=fails.format(r"1\.250e-08")):
        owner(np.array([1.0, -5e-9]), np.array([-0.6, 0.0]), **kw)
    # neither: a flux that fills the negative and drains the rest leaves nothing to clip; nor does an empty step
    m0, m_q = np.array([1.0, -0.9e-8]), np.array([-0.2, 1e-8])
    assert np.array_equal(owner(m0, m_q, **kw), m0 + m_q)
    assert np.array_equal(owner(np.zeros(2), np.zeros(2), **kw), np.zeros(2))
    # no netting: filling node 2 by as much as node 1 is deepened leaves D = 4e-9, not 0, though s_step = 8e-9
    with pytest.raises(ValueError, match=adds.format(r"4\.000e-09")):
        owner(np.array([1.0, -4e-9, -4e-9]), np.array([0.0, -4e-9, 4e-9]), **kw)
    # the caller's weights: uniform, s0 = 3.3e-7 is past the gate and the step is the scheme's; weighted, s0 = 3.3e-10
    m0, m_q = np.array([1.0, -1e-6, 1.0, 1.0]), np.array([0.0, -1e-3, 0.0, 0.0])
    assert np.array_equal(owner(m0, m_q, **kw), m0 + m_q)
    with pytest.raises(ValueError, match=adds.format(r"3\.333e-07")):
        owner(m0, m_q, weights=np.array([1.0, 1e-3, 1.0, 1.0]), **kw)
    # and on F, pos(step) and s_step: F = 1e-9 is within the gate weighted, 1e-6 uniform; ...
    m0, m_q = np.array([1.0, 0.0]), np.array([0.0, -1e-6])
    assert np.array_equal(owner(m0, m_q, weights=np.array([1.0, 1e-3]), **kw), m0 + m_q)
    # ... F / pos(step) = 1.998e-8 weighted, 1e-8 uniform; ...
    m0, m_q, w = np.array([1.0, 0.0, 1.0, -1e-3]), np.array([0.0, -2e-8, 0.0, 0.0]), np.array([1.0, 1.0, 1e-3, 1.0])
    with pytest.raises(ValueError, match=r"fabricate 1\.998e-08 of the mass"):
        owner(m0, m_q, weights=w, **kw)
    # ... and s_step = 1.247e-8 weighted, 3.6e-9 uniform
    with pytest.raises(ValueError, match=fails.format(r"1\.247e-08")):
        owner(np.array([1.0, -5e-9, 1.0]), np.array([-0.6, 0.0, 0.0]), weights=np.array([1.0, 1.0, 1e-3]), **kw)


def test_b3_lets_an_upwind_inflow_round_off_pass_within_the_gate():
    """Upwind's matrix is an M-matrix, so an inflow's part is non-negative in exact arithmetic; spsolve's carries
    round-off negatives, 4.7e-38 at a node the step without the flux leaves at 0, which strict clause (1) refused (review
    3 of #2840). Within the gate (user ruling 2026-10-30) the step solves, and the inflow adds its mass."""
    n, sigma, dt, nt = 41, 0.05, 0.2, 3
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[n, n], boundary_conditions=no_flux_bc(dimension=2)
    )
    x, y = np.meshgrid(*[np.asarray(a) for a in grid.coordinates], indexing="ij")
    m_init = np.where(np.abs(x - 0.5) + np.abs(y - 0.5) < 0.15, 1.0, 0.0)
    bc = BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(
                name=f,
                bc_type=BCType.FLUX if f == "x_min" else BCType.NO_FLUX,
                value=-1.0 if f == "x_min" else 0.0,
                boundary=f,
            )
            for f in FACES[2]
        ],
    )
    conditions = Conditions(m_initial=lambda p: 1.0, u_terminal=lambda p: 0.0, T=nt * dt)
    problem = MFGProblem(model=_model(sigma), domain=grid, conditions=conditions, Nt=nt)
    m = np.asarray(
        FPFDMSolver(problem, boundary_conditions=bc).solve_fp_system(
            m_init, potential_field=np.broadcast_to(3.0 * y, (nt + 1, n, n)).copy(), show_progress=False
        )
    )
    mass = np.array([float(np.sum(level * grid.measure_weights())) for level in m])
    assert np.allclose(np.diff(mass), 1.0 * 1.0 * dt, rtol=0, atol=1e-12), np.diff(mass)


@pytest.mark.parametrize("level", [0.0, 1e-20])
def test_b3_lets_an_upwind_inflow_into_an_empty_domain_solve(level: float):
    """An inflow into a domain left empty, or at 1e-20: the flux's round-off negatives, 6.8e-15 of the first step's 321
    on the guard's uniform weights, measured on the step without the flux, whose mass is 0 or 2e-17, were refused at
    any q. (1) measures them on the step that holds them and (2a) counts only the deepening (the approver's rows on
    #2840, 2026-10-30), so the step solves and the inflow adds its mass."""
    n, dt, nt = 41, 0.1, 2
    grid = _grid(2, n)
    x = np.meshgrid(*[np.asarray(a) for a in grid.coordinates], indexing="ij")[0]
    solver = FPFDMSolver(
        _problem(grid, nt=nt, t_end=nt * dt, sigma=0.05), boundary_conditions=_segments(2, {"x_min": -1.0})
    )
    m = np.asarray(
        solver.solve_fp_system(
            np.full((n, n), level), potential_field=np.broadcast_to(2.0 * x, (nt + 1, n, n)).copy(), show_progress=False
        )
    )  # v = -2
    mass = np.array([float(np.sum(m_k * grid.measure_weights())) for m_k in m])
    assert np.allclose(np.diff(mass), 1.0 * LY * dt, rtol=0, atol=1e-12), np.diff(mass)


@pytest.mark.parametrize("family", ["fdm", "fem"])
def test_b3_clause_two_refuses_a_step_the_flux_takes_past_the_gate(family: str):
    """Clause (2) at the solver level (reviews 2 and 3 of #2840). FP-FDM centered, branch (b): an outflow drains 32% of
    the positive mass, deepening nothing, and takes the step's own clip from s0 = 8.8e-9 past the gate; with q = 0 the
    step solves. FP-FEM P2, branch (a): an inflow deepens the Galerkin step's own negatives, s0 = 7.1e-10, by 3.9e-8 of
    the mass. P2, which keeps the consistent mass: P1's time term is lumped since #2845, and on lumped 1-D P1 the one
    row found for this branch (an inflow at x_max, n 22, sigma 0.14, v 6.2; review 1 of #2848) sits in a dt band of
    relative width 2.4e-8, too narrow for a pin to stay inside across platforms, against 6e-5 for this P2 row."""
    if family == "fdm":
        n, dt = 14, 0.1
        grid = _grid(1, n)
        x = np.asarray(grid.coordinates[0])
        m_init = np.exp(-((x - 0.19) ** 2) / 0.044) + 3e-10
        u = np.broadcast_to(x, (2, n)).copy()  # v = -1
        for q, refused in ((0.6, True), (0.0, False)):
            solver = FPFDMSolver(
                _problem(grid, nt=1, t_end=dt, sigma=0.2),
                boundary_conditions=_segments(1, {"x_min": q}),
                advection_scheme="divergence_centered",
            )
            if refused:
                with pytest.raises(
                    ValueError, match=r"clipping the step with it would fabricate 1\.2\d\de-08, past it.*B-3"
                ):
                    solver.solve_fp_system(m_init, potential_field=u, show_progress=False)
            else:
                solver.solve_fp_system(m_init, potential_field=u, show_progress=False)
    else:
        n, dt, q = 8, 0.00032432, -0.0026
        solver = FPFEMSolver(
            _problem(_mesh(1, n), nt=1, t_end=dt, sigma=0.05), order=2, boundary_conditions=_segments(1, {"x_min": q})
        )
        x = solver._disc.dof_coordinates[:, 0]
        m_init = np.exp(-((x - 0.77) ** 2) / 0.018) + 2.4e-5
        with pytest.raises(
            ValueError,
            match=r"would fabricate 7\.10\de-10 of the mass, within the threshold.*deepens its negatives by 3\.89\de-08 more",
        ):
            solver.solve_fp_system(M_initial=m_init, potential_field=np.broadcast_to(-10.0 * x, (2, x.size)).copy())


def test_an_outflow_that_drains_the_wall_is_refused_on_upwind():
    bc = _segments(1, {"x_max": 10.0})
    # upwind's remedy names the outflow: no upwind inflow row measured under this rule is refused (review 4 of #2840)
    with pytest.raises(ValueError, match=r"B-3\)\. Lower \|q\|, or end the outflow before the mass runs out\.$"):
        _fdm(1, 21, bc, m0=np.ones(21))
    _, m = _fdm(1, 21, _segments(1, {"x_max": 0.05}), m0=np.ones(21))
    assert float(m.min()) >= 0.0


def _bump(x: np.ndarray) -> np.ndarray:
    return np.exp(-((x - 0.5) ** 2) / 0.02) + 1e-3


def test_centered_fdm_is_refused_where_the_first_negative_is_off_the_flux_face():
    """n 11, v 2, sigma 0.05, dt 0.1: the first node to go negative as q grows is x_min's, at q* = 0.2447, and the FLUX
    node's is 0.2574 (`p_b3_rows.py`); at q = 0.25 the step is refused at x_min alone, its FLUX node non-negative."""
    grid = _grid(1, 11)
    x = np.asarray(grid.coordinates[0])
    solver = FPFDMSolver(
        _problem(grid, nt=2, t_end=0.2, sigma=0.05),
        boundary_conditions=_segments(1, {"x_max": 0.25}),
        advection_scheme="divergence_centered",
    )
    u = np.broadcast_to(-2.0 * x, (3, 11)).copy()
    with pytest.raises(ValueError, match=r"at 1 node\(s\) in the step to t=0\.1, the most at node \(0,\)") as caught:
        solver.solve_fp_system(_bump(x), potential_field=u, show_progress=False)
    # the step without the flux is past the gate (s0 = 0.23): the message says so, hands a lower q to the guard, and at
    # q = 0 the guard refuses the step, as it says (reviews 3 and 4 of #2840)
    assert "B-3" in str(caught.value)
    assert "hands the step to FP-FDM's guard, which refuses it unless the step with the flux passes the gate" in str(
        caught.value
    )
    assert "at q = 0 on this step it does not. The state this step starts from holds what the flux did" in str(
        caught.value
    )
    unfluxed = FPFDMSolver(
        _problem(grid, nt=2, t_end=0.2, sigma=0.05),
        boundary_conditions=_segments(1, {"x_max": 0.0}),
        advection_scheme="divergence_centered",
    )
    with pytest.raises(ValueError, match=r"advection_scheme='divergence_centered': density went to"):
        unfluxed.solve_fp_system(_bump(x), potential_field=u, show_progress=False)


def test_fem_is_refused_where_the_first_negative_is_off_the_flux_face():
    """P1, 10 cells, v 2, sigma 0.12, dt 0.1: the first DOF to go negative as q grows is x_min's, at q* = 1.0991, and the
    FLUX DOF's is 1.2098 (`p_b3_pick_rows.py`, re-found on the lumped P1 step of #2845); at q = 1.15 the step is refused
    at x_min's DOF alone."""
    mesh = _mesh(1, 10)
    solver = FPFEMSolver(_problem(mesh, nt=2, t_end=0.2, sigma=0.12), boundary_conditions=_segments(1, {"x_max": 1.15}))
    x = solver._disc.dof_coordinates[:, 0]
    first = int(np.argmin(x))
    u = np.broadcast_to(-2.0 * x, (3, x.size)).copy()
    m_init = np.exp(-((x - 0.66) ** 2) / 0.08) + 1.5e-3
    with pytest.raises(ValueError, match=rf"at 1 DOF\(s\) in the step to t=0\.1, the most at DOF {first} ") as caught:
        solver.solve_fp_system(M_initial=m_init, potential_field=u)
    # the step without the flux is past the gate (s0 = 0.45): lowering q below q* hands the step to FP-FEM's clip, as the
    # message says, and the solve runs (review 3 of #2840)
    assert (
        "A q on this step with which the flux takes no DOF negative past the threshold hands the step to FP-FEM's"
        in str(caught.value)
    )
    lowered = FPFEMSolver(
        _problem(mesh, nt=2, t_end=0.2, sigma=0.12), boundary_conditions=_segments(1, {"x_max": 0.85})
    )
    assert float(np.min(lowered.solve_fp_system(M_initial=m_init, potential_field=u))) >= 0.0


def test_fp_fdm_names_the_inflow_remedy_on_centered_and_it_clears_the_row():
    """An inflow, q = -3, into x_max on centered: its flux part goes negative off the wall, the step without it within
    the gate, so B-3 names the scheme's switch to upwind; on upwind the same row solves (review 3 of #2840)."""
    grid = _grid(1, 11)
    x = np.asarray(grid.coordinates[0])
    m_init = 1.0 + np.exp(-((x - 0.5) ** 2) / 0.02)
    u = np.broadcast_to(-2.0 * x, (2, 11)).copy()

    def solve(scheme):
        solver = FPFDMSolver(
            _problem(grid, nt=1, t_end=0.05, sigma=0.3),
            boundary_conditions=_segments(1, {"x_max": -3.0}),
            advection_scheme=scheme,
        )
        return solver.solve_fp_system(m_init, potential_field=u, show_progress=False)

    with pytest.raises(
        ValueError,
        match=r"B-3\)\. 'divergence_centered' is not monotone.*for an inflow, use advection_scheme='divergence_upwind'",
    ):
        solve("divergence_centered")
    assert float(np.min(solve("divergence_upwind"))) >= 0.0


# ------------------------------------------------------------------------------------------------------ the refusals


def test_a_flux_on_the_shared_bc_is_refused_naming_the_fp_route():
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=flux_bc(0.1, dimension=1))
    with pytest.raises(ValueError, match=r"pass flux_bc\(\.\.\.\) to the FP solver as its own boundary_conditions"):
        FPFDMSolver(_problem(grid))


def test_fp_fdm_refuses_a_flux_off_its_implicit_step():
    grid = _grid(1, 11)
    solver = FPFDMSolver(_problem(grid), boundary_conditions=_segments(1, {"x_max": Q}))
    with pytest.raises(NotImplementedError, match=r"explicit step for a callable drift"):
        solver.solve_fp_system(np.ones(11), drift_field=lambda t, x, m: np.zeros_like(x), show_progress=False)
    with pytest.raises(NotImplementedError, match=r"solve_fp_step_adjoint_mode"):
        solver.solve_fp_step_adjoint_mode(np.ones(11), np.zeros((11, 11)))


def test_fp_fem_refuses_a_flux_on_its_adjoint_step():
    solver = FPFEMSolver(_problem(_mesh(1, 8)), boundary_conditions=_segments(1, {"x_max": Q}))
    with pytest.raises(NotImplementedError, match=r"builds no prescribed-flux load"):
        solver.solve_fp_step_adjoint_mode(np.ones(solver.n_dof), np.zeros((solver.n_dof, solver.n_dof)))


def test_the_fdm_readers_that_cannot_impose_a_flux_refuse_it():
    """A ghost layer, a ghost calculator and a geometric fold read a condition on the field; FLUX is on the total flux.
    On arrival the ghost layer wrote a reflection (mixed) or zeros (uniform), the calculator route a zero gradient, the
    fold a clamp, the legacy ghost values a mirror and the nD tensor diffusion a zero wall value (review 1 of #2840);
    each now refuses by name (#2581), as the particle action does, saying what FLUX is."""
    from mfgarchon.geometry.boundary._compat import get_ghost_values_nd
    from mfgarchon.geometry.boundary.applicator_fdm import bc_to_topology_calculator
    from mfgarchon.geometry.boundary.applicator_particle import particle_action_for_bc_type
    from mfgarchon.geometry.boundary.bc_utils import bc_type_to_geometric_operation
    from mfgarchon.operators.differential import DiffusionOperator
    from mfgarchon.operators.differential.laplacian import LaplacianOperator

    bc = flux_bc(0.3, dimension=1)
    with pytest.raises(NotImplementedError, match=r"The FDM ghost layer has no reading of BCType\.FLUX"):
        LaplacianOperator(spacings=[0.1], field_shape=(11,), bc=bc) @ np.ones(11)
    with pytest.raises(NotImplementedError, match=r"calculator route .* no reading of BCType\.FLUX"):
        bc_to_topology_calculator(bc, (11,))
    with pytest.raises(NotImplementedError, match=r"FLUX boundary .* has no geometric operation"):
        bc_type_to_geometric_operation("flux")
    # the legacy ghost values are deprecated, and warn so before they refuse
    with (
        pytest.deprecated_call(),
        pytest.raises(NotImplementedError, match=r"A ghost value has no reading of BCType\.FLUX"),
    ):
        get_ghost_values_nd(np.ones(11), bc, spacing=(0.1,))
    tensor = DiffusionOperator(coefficient=0.1 * np.eye(3), spacings=[0.2] * 3, field_shape=(6, 6, 6), bc=flux_bc(0.3))
    with pytest.raises(
        NotImplementedError, match=r"nD tensor diffusion's ghost padding has no reading of BCType\.FLUX"
    ):
        tensor @ np.ones(216)
    with pytest.raises(
        ValueError, match=r"BCType\.FLUX \(a prescribed mass flux J\.n = q\) has no per-particle action"
    ):
        particle_action_for_bc_type(BCType.FLUX)


def test_a_flux_needs_its_q():
    """A FLUX's value is q, and a None constant reads as 0: a FLUX with no q solved as NO_FLUX on both families (reviews
    1 and 2 of #2840). It is refused at the read every FLUX reader makes (`segment_value`), however and whenever the
    value became None: on a segment, by ``flux_bc(None)``, on a FLUX ``default_bc``, or reassigned after the BC was
    built."""
    with pytest.raises(ValueError, match=r"Segment 'x_max' with flux BC requires a value"):
        _fdm(1, 11, _segments(1, {"x_max": None}))
    with pytest.raises(ValueError, match=r"Segment 'uniform' with flux BC requires a value"):
        _fdm(1, 11, flux_bc(None, dimension=1))
    with pytest.raises(ValueError, match=r"Segment 'default' with flux BC requires a value"):
        _fem(1, 8, BoundaryConditions(dimension=1, segments=[], default_bc=BCType.FLUX, default_value=None))
    reassigned = _segments(1, {"x_max": 0.2})
    reassigned.segments[1].value = None
    with pytest.raises(ValueError, match=r"Segment 'x_max' with flux BC requires a value"):
        _fem(1, 8, reassigned)
    with pytest.raises(ValueError, match=r"Segment 'x_max' with flux BC requires a value"):
        reassigned.validate_values()

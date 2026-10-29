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

**B-3** (user ruling 2026-10-09; the audit's ruling on the test, #2581's plan, Q3): a step is refused where its flux part
takes the density negative at a node that the step without it leaves non-negative, never clipped. On
``divergence_upwind`` the matrix is an M-matrix, the step's right-hand side without the flux is ``m_old/dt >= 0``, and
the flux's right-hand side is zero off the FLUX faces; under those three conditions the first node to go negative is on a
FLUX face or ties with one (a discrete maximum principle), so there a check of the FLUX nodes alone refuses at the same
step. ``divergence_centered`` and FP-FEM have no M-matrix, and the rows below take a step whose first negative is off
the face, at the opposite wall, while every FLUX node stays non-negative: there a FLUX-node check would solve on.

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
    the first face's outward velocity, so the FLUX source imposes the diffusive flux alone."""
    original = fp_fdm_time_stepping._BOUNDARY_HANDLERS["divergence_upwind"]

    def handler(rows, cols, data, flat, index, shape, ndim, dt, sigma, c, spacing, u, grid, interface_velocity=None):
        original(rows, cols, data, flat, index, shape, ndim, dt, sigma, c, spacing, u, grid, interface_velocity)
        for d in range(ndim):
            dx = spacing[d]
            if index[d] in (0, shape[d] - 1):
                step = 1 if index[d] == 0 else -1
                neighbour = list(index)
                neighbour[d] += step
                v_out = step * c * (u[grid.get_index(tuple(neighbour))] - u[flat]) / dx
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
    # Measured: 1.6 (1-D) and 3.5 (2-D) at every level, flat, against the cell's own 0.02 to 0.09.
    assert all(e > 1.0 for e in errors), (name, errors)
    assert errors[-1] > 0.9 * errors[0], (name, errors)


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


# ------------------------------------------------------------------------------------------------------------- B-3


def test_an_outflow_that_drains_the_wall_is_refused_on_upwind():
    bc = _segments(1, {"x_max": 10.0})
    with pytest.raises(ValueError, match=r"B-3"):
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
    with pytest.raises(ValueError, match=r"at 1 node\(s\) in the step to t=0\.1, the most at node \(0,\)") as caught:
        solver.solve_fp_system(_bump(x), potential_field=np.broadcast_to(-2.0 * x, (3, 11)).copy(), show_progress=False)
    assert "B-3" in str(caught.value)


def test_fem_is_refused_where_the_first_negative_is_off_the_flux_face():
    """P1, 8 cells, v 2, sigma 0.3, dt 0.1: the first DOF to go negative as q grows is x_min's, at q* = 0.2677, and the
    FLUX DOF's is 0.3039 (`p_b3_rows.py`); at q = 0.285 the step is refused at x_min's DOF alone."""
    mesh = _mesh(1, 8)
    solver = FPFEMSolver(_problem(mesh, nt=2, t_end=0.2, sigma=0.3), boundary_conditions=_segments(1, {"x_max": 0.285}))
    x = solver._disc.dof_coordinates[:, 0]
    first = int(np.argmin(x))
    with pytest.raises(ValueError, match=rf"at 1 DOF\(s\) in the step to t=0\.1, the most at DOF {first} "):
        solver.solve_fp_system(M_initial=_bump(x), potential_field=np.broadcast_to(-2.0 * x, (3, x.size)).copy())


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

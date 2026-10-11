"""A Dirichlet value is read at the time of the level solved for: FEM, meshless Nitsche and HJB-GFDM (#2512 row B2,
#2538).

External oracle: u = a0 + c t + b.x on every wall.
- **HJB**, with H = |p|^2 / 2 and no coupling: c = H(b), read from the problem's own Hamiltonian, so
  -u_t + H(grad u) - D lap u = 0 holds exactly.
- **FP**, with zero drift and the source c: m_t - D lap m = c holds exactly.

A linear function lies in the P1 space and in what GFDM's stencils reproduce, and implicit Euler is exact for a
solution linear in t. So every case reproduces u to roundoff, provided each level reads its wall at its own time. A
wall read one level off misses by c dt.

Measured at `91f90824` (main): FEM and meshless Nitsche refused every g(t, x), since they had no time to pass. HJB-GFDM
already read each level at its own time. Its cases pin that.

Meshless Nitsche imposes the wall weakly, so it has no roundoff oracle. Its case checks instead that the wall tracks
g(t_n) at each level, on both arms of the step, and that g is read at t_0 .. t_{Nt-1} and nowhere else.
"""

from __future__ import annotations

import warnings

import pytest
import skfem

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fem import FPFEMSolver, HJBFEMSolver
from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfgarchon.alg.numerical.meshless_galerkin.hjb_solver import MeshlessGalerkinHJBSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import dirichlet_bc
from mfgarchon.geometry.meshes.mesh_1d import Mesh1D
from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

T, NT, SIGMA, A0, LY = 0.5, 10, 0.4, 1.0, 0.6
B = {1: np.array([0.7]), 2: np.array([0.7, -0.4])}
FP_RATE = 0.3


def _hamiltonian():
    zero = lambda m: 0.0 * np.asarray(m)  # noqa: E731
    return SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0), coupling=zero, coupling_dm=zero)


def _problem(geometry, nt=NT):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MFGProblem(
            model=Model(hamiltonian=_hamiltonian(), volatility=SIGMA),
            domain=geometry,
            conditions=Conditions(m_initial=lambda p: 1.0, u_terminal=lambda p: 0.0, T=T),
            Nt=nt,
        )


def _rate(problem, d):
    """c = H(b), read from the problem's Hamiltonian, so the exact solution follows its sign convention."""
    p = B[d][None, :]
    return float(np.asarray(problem.hamiltonian_class(x=np.zeros((1, d)), m=np.ones(1), p=p, t=0.0)).ravel()[0])


def _mesh(d, n):
    if d == 1:
        geometry = Mesh1D(bounds=(0.0, 1.0), num_elements=n)
        geometry.generate_mesh()
        return geometry
    geometry = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, LY))
    geometry.mesh_data = skfem_to_meshdata(
        skfem.MeshTri.init_tensor(np.linspace(0.0, 1.0, n + 1), np.linspace(0.0, LY, n + 1))
    )
    return geometry


def _exact(c, d):
    b = B[d]
    return lambda t, x: A0 + c * t + np.asarray(x, dtype=float) @ b


@pytest.mark.parametrize("d", [1, 2])
@pytest.mark.parametrize("arm", ["picard", "newton", "fp"])
def test_fem_reads_a_dirichlet_value_at_its_level(arm, d):
    n = 8 if d == 1 else 6
    geometry = _mesh(d, n)
    geometry.boundary_conditions = dirichlet_bc(dimension=d, value=0.0)
    c = FP_RATE if arm == "fp" else _rate(_problem(geometry), d)
    u = _exact(c, d)
    g = dirichlet_bc(dimension=d, value=lambda t, x: float(u(t, x)))
    if arm != "fp":
        geometry.boundary_conditions = g
    problem = _problem(geometry)
    levels = np.linspace(0.0, T, NT + 1)
    if arm == "fp":
        solver = FPFEMSolver(problem, order=1, boundary_conditions=g)
        X = solver._disc.dof_coordinates
        exact = np.stack([u(t, X) for t in levels])
        result = solver.solve_fp_system(
            exact[0], np.zeros((NT + 1, len(X))), source_term=lambda t, x: np.full(len(x), FP_RATE)
        )
    else:
        solver = HJBFEMSolver(problem, order=1)
        X = solver._disc.dof_coordinates
        exact = np.stack([u(t, X) for t in levels])
        result = solver.solve_hjb_system(
            M_density=np.ones((NT + 1, len(X))),
            U_terminal=exact[-1],
            U_coupling_prev=None if arm == "newton" else exact,
            use_newton=arm == "newton",
            newton_tolerance=1e-12,
        )
    assert np.abs(np.asarray(result) - exact).max() < 1e-12


@pytest.mark.parametrize(("d", "inner"), [(1, "newton"), (1, "howard"), (2, "newton")])
def test_gfdm_reads_a_dirichlet_value_at_its_level(d, inner):
    n = 11 if d == 1 else 7
    kwargs = (
        {"monotonicity_scheme": "none"}
        if inner == "newton"
        else {"inner_solver": "howard", "monotonicity_scheme": "qp_m_matrix", "monotonicity_application": "always"}
    )
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * d, Nx_points=[n] * d, boundary_conditions=dirichlet_bc(dimension=d))
    u = _exact(_rate(_problem(grid), d), d)
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * d,
        Nx_points=[n] * d,
        boundary_conditions=dirichlet_bc(dimension=d, value=lambda t, x: float(u(t, x))),
    )
    points = np.stack(np.meshgrid(*[np.linspace(0.0, 1.0, n)] * d, indexing="ij"), axis=-1).reshape(-1, d)
    exact = np.stack([u(t, points) for t in np.linspace(0.0, T, NT + 1)])
    solver = HJBGFDMSolver(_problem(grid), collocation_points=points, delta=3.0 / (n - 1), **kwargs)
    result = solver.solve_hjb_system(
        M_density=np.ones((NT + 1, len(points))), U_terminal=exact[-1], U_coupling_prev=exact
    )
    assert np.abs(np.asarray(result).reshape(exact.shape) - exact).max() < 1e-12


@pytest.mark.parametrize("newton", [False, True], ids=["picard", "newton"])
def test_meshless_nitsche_reads_its_data_at_each_level(newton):
    """Each level's wall holds g at that level's time, g rising by 2 per unit time: within 0.05, where a load one
    level off misses by 2 dt = 0.25. Measured at 6d1dd200, the miss is at most 0.0057 (Picard) and 0.0043 (Newton).
    With every level served the first level's load, it is 0.75. The block is the same at every level. The terms
    are assembled again at each level only because the value declares t."""
    nt, nx, rate = 4, 11, 2.0
    times = []

    def g(t, x):
        times.append(round(float(t), 12))
        return A0 + rate * t + 0.7 * x[0]

    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=dirichlet_bc(dimension=1, value=g)
    )
    problem = _problem(grid, nt=nt)
    xs = np.linspace(0.0, 1.0, nx)[:, None]
    solver = MeshlessGalerkinHJBSolver(problem, xs, delta=0.35)
    times.clear()
    U = np.asarray(
        solver.solve_hjb_system(
            M_density=np.ones((nt + 1, nx)),
            U_terminal=A0 + rate * T + 0.7 * xs[:, 0],
            U_coupling_prev=np.zeros((nt + 1, nx)),
            use_newton=newton,
        )
    )
    levels = np.linspace(0.0, T, nt + 1)[:nt]
    walls = U[:nt][:, [0, -1]] - (A0 + rate * levels[:, None] + np.array([0.0, 0.7]))
    assert np.abs(walls).max() < 0.05, walls
    assert sorted(set(times)) == [round(t, 12) for t in levels]

"""HJB-GFDM moves a grid density to its collocation points by position, on the grid's own nodes (Issue #2756).

Two defects of one owner, the solver's grid-to-collocation transfer:

1. The representation was chosen by shape. A grid density whose node count equals the number of collocation points
   was read as already at the points, so grid node i's value went to collocation point i, and the terminal value's
   with it. With the 21 nodes reversed as the collocation set, at f74312f9, in a run with no coupling term: U came back
   4.592e-01 off, in point order, and 4.000e-01 off with that order undone, which is the terminal value read by index.
2. The mapper rebuilt the grid's axes as ``linspace(bounds, shape)``. On a graded grid, m = x mapped to 31 points
   was 2.500e-01 off x.

Both oracles are exact up to round-off, so they fail by orders of magnitude when either defect returns. The density and the terminal
value are asymmetric in x and enter the HJB through a coupling f(m) = m, so a reversal and a uniform-axes reading
both move U.
"""

import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical import HJBGFDMSolver
from mfgarchon.alg.numerical.gfdm_components import GridCollocationMapper
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import Hyperrectangle, TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.grids.tensor_grid import geometry_grid_axes, tensor_grid_axes

_NT, _T = 10, 0.5


def _potential(t, x):
    x = np.asarray(x, dtype=float)[..., 0]
    return 0.8 * np.sin(2 * np.pi * x) + 0.5 * x


def _problem(geometry, u_terminal):
    # The potential breaks the problem's mirror symmetry: without it, nodes read by index in reversed order solve the
    # mirrored problem and return exactly U(nodes), so a reversal could not see the defect.
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        potential=_potential,
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )
    return MFGProblem(
        model=Model(hamiltonian=H, volatility=0.3),
        domain=geometry,
        conditions=Conditions(u_terminal=u_terminal, m_initial=lambda x: 1.0, T=_T),
        Nt=_NT,
    )


def _solve(problem, points, M, u_terminal_values, **solver_kwargs):
    solver = HJBGFDMSolver(
        problem,
        collocation_points=np.asarray(points, dtype=float).reshape(len(points), -1),
        monotonicity_scheme="none",
        **solver_kwargs,
    )
    U0 = np.tile(u_terminal_values, (_NT + 1, 1))
    return solver.solve_hjb_system(M, u_terminal_values, U0)


def test_a_grid_density_reaches_permuted_collocation_points_by_position_not_index():
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    problem = _problem(grid, lambda x: (np.asarray(x, dtype=float) - 0.7) ** 2)
    x = np.asarray(grid.get_spatial_grid(), dtype=float).ravel()
    M = np.tile(1.0 + 0.5 * np.cos(np.pi * x), (_NT + 1, 1))
    u_T = (x - 0.7) ** 2

    U_in_order = _solve(problem, x, M, u_T)
    for order in (np.arange(21)[::-1], np.random.default_rng(3).permutation(21)):
        U_permuted = _solve(problem, x[order], M, u_T)

        # The same points in another order are the same discretisation, and the output is on the grid's nodes either
        # way. At f74312f9 the reversed nodes gave U 4.000e-01 off once its point order was undone, from the terminal
        # value read by index (a run with no coupling term), and 4.592e-01 off as returned.
        # The two solves differ in round-off only, by an amount that depends on the order and the BLAS: on macOS
        # 2.2e-16 reversed and 2.1e-12 permuted. Unmeasured on the Ubuntu runner, where the 2-D row below is 270 times
        # its macOS value.
        # Reading the density by index, at 2034e5a9 before #2756's fix, puts U 0.480 off reversed and 0.602 permuted.
        # The bound sits between the two.
        assert U_permuted.shape == U_in_order.shape == (_NT + 1, 21)
        assert np.max(np.abs(U_permuted - U_in_order)) < 1e-6


def test_a_2d_grid_density_reaches_its_nodes_in_f_order_by_position():
    # The 2-D case of the test above, for the C-order node test: the grid's nodes in F order are not its nodes in its
    # order, so a flattened density of the grid's count is the grid's and is mapped. A node test against F order would
    # read it by index.
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[6, 5], boundary_conditions=no_flux_bc(dimension=2)
    )
    problem = _problem(grid, lambda x: float((np.asarray(x, dtype=float)[0] - 0.7) ** 2 + 0.5 * np.asarray(x)[1]))
    nodes = np.asarray(grid.get_spatial_grid(), dtype=float)
    M = np.tile(1.0 + 0.5 * np.cos(np.pi * nodes[:, 0]) + 0.3 * nodes[:, 1], (_NT + 1, 1))
    u_T = (nodes[:, 0] - 0.7) ** 2 + 0.5 * nodes[:, 1]
    f_order = np.arange(30).reshape(6, 5).T.ravel()

    U_in_order = _solve(problem, nodes, M, u_T, boundary_conditions=no_flux_bc(dimension=2))
    U_f_order = _solve(problem, nodes[f_order], M, u_T, boundary_conditions=no_flux_bc(dimension=2))

    assert U_in_order.shape == (_NT + 1, 30)
    assert U_f_order.shape == (_NT + 1, 6, 5)
    # The two solves differ in round-off only, and how much depends on the point order and the BLAS: 7.6e-12 on macOS
    # (Accelerate) and 2.05e-9 on the Ubuntu runner (discrimination run 39231402975), both at the fix. Reading the
    # density by index, at 2034e5a9 before #2756's fix, puts it 1.331 off. The bound sits between the two.
    assert np.max(np.abs(U_f_order.reshape(_NT + 1, 30) - U_in_order)) < 1e-6


def test_a_graded_grid_density_is_interpolated_on_its_own_nodes():
    nodes = np.linspace(0.0, 1.0, 21) ** 2
    graded = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[21],
        spacing_type="custom",
        custom_coordinates=[nodes],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    assert not graded.is_uniform
    problem = _problem(graded, lambda x: 0.3 * np.asarray(x, dtype=float) + 0.1)
    points = np.linspace(0.0, 1.0, 31)

    # Linear m and u_T: interpolation on the grid's nodes reproduces them at the points exactly, so the solve from the
    # grid must equal the solve from their exact values at the points. On the bounds' uniform axes m = x was 2.500e-01
    # off at the points.
    U_from_grid = _solve(problem, points, np.tile(nodes, (_NT + 1, 1)), 0.3 * nodes + 0.1)
    U_at_points = _solve(problem, points, np.tile(points, (_NT + 1, 1)), 0.3 * points + 0.1)

    # The grid route returns U on the graded nodes, linearly interpolated from the points.
    assert U_from_grid.shape == (_NT + 1, 21)
    expected = np.stack([np.interp(nodes, points, row) for row in U_at_points])
    assert np.max(np.abs(U_from_grid - expected)) < 1e-12


def test_the_mapper_reads_a_graded_2d_grid_from_its_nodes():
    xs, ys = np.linspace(0.0, 1.0, 9) ** 2, np.sqrt(np.linspace(0.0, 1.0, 7))
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[9, 7],
        spacing_type="custom",
        custom_coordinates=[xs, ys],
        boundary_conditions=no_flux_bc(dimension=2),
    )
    nodes = np.asarray(grid.get_spatial_grid(), dtype=float)
    axes = geometry_grid_axes(grid)
    assert axes is not None
    assert np.array_equal(axes[0], xs)
    assert np.array_equal(axes[1], ys)

    points = np.random.default_rng(0).uniform(0.05, 0.95, (40, 2))
    mapper = GridCollocationMapper(points, grid_shape=(9, 7), grid_axes=axes)
    m = nodes[:, 0] + 2.0 * nodes[:, 1]
    assert np.max(np.abs(mapper.map_grid_to_collocation(m) - (points[:, 0] + 2.0 * points[:, 1]))) < 1e-12

    # Sampled points, or a C-order grid read with the wrong shape, are no tensor grid; an implicit domain declares none.
    assert tensor_grid_axes(points, (40,)) is None
    assert tensor_grid_axes(nodes, (7, 9)) is None
    assert geometry_grid_axes(Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]]))) is None


def test_a_density_that_fits_neither_the_points_nor_a_grid_is_refused():
    with warnings.catch_warnings():
        # m = 1 has mass n*dx on the domain's uniform-cell measure, not 1; the refusal below does not depend on it.
        warnings.filterwarnings("ignore", message="initial density mass")
        problem = _problem(Hyperrectangle(np.array([[0.0, 1.0]])), lambda x: 0.0 * np.asarray(x, dtype=float))
    n_samples = problem.geometry.get_grid_shape()[0]
    points = np.linspace(0.0, 1.0, 31)
    assert n_samples != 31

    # An implicit domain's sampled points are no grid to map from: before #2756 they were read as uniform axes.
    with pytest.raises(ValueError, match=r"neither the 31 collocation points' \(31,\) nor a grid"):
        _solve(
            problem,
            points,
            np.ones((_NT + 1, n_samples)),
            np.zeros(n_samples),
            boundary_conditions=no_flux_bc(dimension=1),
        )


def test_the_particle_drift_reaches_a_particle_on_the_last_node():
    # FPParticleSolver's 1-D drift is read on the grid's axes (#2756). At 2034e5a9 it was read on
    # linspace(xmin, xmin + Lx), which over (0.2, 0.9) ends at 0.8999999999999999: a particle sampled on the last node
    # lay past the axis and got the fill value, zero drift. With no noise every particle moves by exactly drift * dt.
    from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver

    grid = TensorProductGrid([(0.2, 0.9)], Nx_points=[15], boundary_conditions=no_flux_bc(dimension=1))
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    with warnings.catch_warnings():
        # m = 1 has mass 0.7 on (0.2, 0.9); the drift below does not depend on it.
        warnings.filterwarnings("ignore", message="initial density mass")
        problem = MFGProblem(
            model=Model(hamiltonian=H, volatility=0.0),
            domain=grid,
            conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.2),
            Nt=4,
        )
    x = np.asarray(grid.get_spatial_grid(), dtype=float).ravel()
    solver = FPParticleSolver(problem, num_particles=3000, seed=7)
    # U = x/2: the drift -grad U is -0.5 everywhere
    solver.solve_fp_system(M_initial=np.ones(15), drift_field=np.tile(0.5 * x, (5, 1)), show_progress=False)
    first, second = (np.asarray(solver.M_particles_trajectory[k], dtype=float).ravel() for k in (0, 1))

    assert int(np.isclose(first, 0.9, rtol=0, atol=1e-12).sum()) > 0, (
        "the fixture must start particles on the last node"
    )
    # The particles that start within a step of the reflecting x_min are placed back by the wall; every other one moves.
    away = first >= 0.2 + 0.5 * 0.2 / 4 + 1e-9
    assert np.max(np.abs((second[away] - first[away]) - (-0.5 * 0.2 / 4))) < 1e-12

"""HJB-GFDM moves a grid density to its collocation points by position, on the grid's own nodes (Issue #2756).

Two defects of one owner, the solver's grid-to-collocation transfer:

1. The representation was chosen by shape. A grid density whose node count equals the number of collocation points
   was read as already at the points, so grid node i's value went to collocation point i. With the 21 nodes reversed
   as the collocation set, U moved by 4.592e-01 (main at f74312f9).
2. The mapper rebuilt the grid's axes as ``linspace(bounds, shape)``. On a graded grid, m = x mapped to 31 points
   was 2.500e-01 off x.

Both oracles are exact, so they fail by orders of magnitude when either defect returns. The density and the terminal
value are asymmetric in x and enter the HJB through a coupling f(m) = m, so a reversal and a uniform-axes reading
both move U.
"""

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical import HJBGFDMSolver
from mfgarchon.alg.numerical.gfdm_components import GridCollocationMapper
from mfgarchon.alg.numerical.gfdm_components.grid_collocation_mapper import tensor_grid_axes
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import Hyperrectangle, TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

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
        conditions=Conditions(u_terminal=u_terminal, m_initial=lambda x: 1.0 + 0.0 * np.asarray(x, dtype=float), T=_T),
        Nt=_NT,
    )


def _solve(problem, points, M, u_terminal_values, **solver_kwargs):
    solver = HJBGFDMSolver(problem, collocation_points=np.asarray(points, dtype=float).reshape(-1, 1), **solver_kwargs)
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
        # way. Reading the density by index gave 4.592e-01 on the reversed nodes (no coupling term) at f74312f9.
        assert U_permuted.shape == U_in_order.shape == (_NT + 1, 21)
        assert np.max(np.abs(U_permuted - U_in_order)) < 1e-10


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
    axes = tensor_grid_axes(nodes, tuple(grid.get_grid_shape()))
    assert axes is not None
    assert np.array_equal(axes[0], xs)
    assert np.array_equal(axes[1], ys)

    points = np.random.default_rng(0).uniform(0.05, 0.95, (40, 2))
    mapper = GridCollocationMapper(points, grid_shape=(9, 7), grid_axes=axes)
    m = nodes[:, 0] + 2.0 * nodes[:, 1]
    assert np.max(np.abs(mapper.map_grid_to_collocation(m) - (points[:, 0] + 2.0 * points[:, 1]))) < 1e-12

    # Sampled points, or a C-order grid read with the wrong shape, are no tensor grid.
    assert tensor_grid_axes(points, (40,)) is None
    assert tensor_grid_axes(nodes, (7, 9)) is None


def test_a_density_that_fits_neither_the_points_nor_a_grid_is_refused():
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

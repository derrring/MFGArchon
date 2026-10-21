"""#2490, the folded #2748: an axis's node order is its coordinate order, and the grid owns it.

Every reader of a grid's ``coordinates`` takes index order as coordinate order: the face rule and region marking read
node 0 as the min face, the measure's trapezoid weights are the gaps between consecutive nodes, and a point is read at
its nearest node. Before, `TensorProductGrid`'s custom branch admitted an axis in any order and kept the caller's array.
A descending axis put a region on the opposite face and gave the measure weights summing to -1, a swapped pair gave
positive weights that integrate wrongly with nothing raised, and reversing the caller's array after construction undid
any check. Now the constructor refuses an axis that does not strictly ascend, and holds each axis as its own read-only
float64 copy. `CartesianGrid` declares the attribute and its order, and a reader reads it only from that type."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import numpy as np

from mfgarchon.geometry import Hyperrectangle, TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc, periodic_bc

_ASCENDING = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_ORDER_REFUSED = (
    r"custom_coordinates\[{axis}\] does not strictly ascend: node {k} is at {a} and node {k1} at {b}.*#2490"
)


def _grid(axes, bc=None):
    axes = [np.asarray(a, dtype=float) for a in axes]
    return TensorProductGrid(
        bounds=[(0.0, 1.0)] * len(axes),
        Nx_points=[len(a) for a in axes],
        spacing_type="custom",
        custom_coordinates=axes,
        boundary_conditions=bc if bc is not None else no_flux_bc(dimension=len(axes)),
    )


def _reversed_after_construction():
    """The caller reverses its own array in place once the grid holds it: the grid's axis and measure do not move."""
    x = _ASCENDING.copy()
    grid = _grid([x, _ASCENDING])
    before = np.array(grid.measure_weights())
    x[:] = x[::-1].copy()
    assert grid.coordinates[0].tolist() == _ASCENDING.tolist()
    np.testing.assert_array_equal(grid.measure_weights(), before)


def _written_by_the_library():
    """An axis is read-only on both branches, so no code can reorder one after construction."""
    for grid in (
        _grid([_ASCENDING]),
        TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[5], boundary_conditions=no_flux_bc(dimension=1)),
    ):
        with pytest.raises(ValueError, match="read-only"):
            grid.coordinates[0][0] = 2.0


# (row, axes, the refusal it gives, or None where it builds)
_ROWS = [
    (
        "a descending axis, axis 0",
        [_ASCENDING[::-1], _ASCENDING],
        _ORDER_REFUSED.format(axis=0, k=0, a=1.0, k1=1, b=0.75),
    ),
    (
        "a descending axis, axis 1",
        [_ASCENDING, _ASCENDING[::-1]],
        _ORDER_REFUSED.format(axis=1, k=0, a=1.0, k1=1, b=0.75),
    ),
    (
        "a swapped pair, axis 0",
        [[0.0, 0.5, 0.25, 0.75, 1.0], _ASCENDING],
        _ORDER_REFUSED.format(axis=0, k=1, a=0.5, k1=2, b=0.25),
    ),
    (
        "a swapped pair, axis 1",
        [_ASCENDING, [0.0, 0.5, 0.25, 0.75, 1.0]],
        _ORDER_REFUSED.format(axis=1, k=1, a=0.5, k1=2, b=0.25),
    ),
    (
        "a repeated node, axis 0",
        [[0.0, 0.25, 0.25, 0.75, 1.0], _ASCENDING],
        _ORDER_REFUSED.format(axis=0, k=1, a=0.25, k1=2, b=0.25),
    ),
    (
        "a repeated node, axis 1",
        [_ASCENDING, [0.0, 0.25, 0.25, 0.75, 1.0]],
        _ORDER_REFUSED.format(axis=1, k=1, a=0.25, k1=2, b=0.25),
    ),
    (
        "#2748's grid, x nodes 1.0 ... 0.0",
        [[1.0, 0.75, 0.5, 0.25, 0.0], [0.0, 0.5, 1.0]],
        _ORDER_REFUSED.format(axis=0, k=0, a=1.0, k1=1, b=0.75),
    ),
    ("an ascending axis", [_ASCENDING, [0.0, 0.1, 0.4, 1.0]], None),
    ("#2769's cell-centred axis, short of its bounds", [np.arange(20) / 20 + 1 / 40], None),
    ("a single node", [[0.5]], None),
]


@pytest.mark.parametrize(("row", "axes", "refusal"), _ROWS, ids=[r[0] for r in _ROWS])
def test_a_custom_axis_strictly_ascends_or_is_refused_when_built(row, axes, refusal):
    """An axis that descends, swaps two nodes or repeats one is refused by name, naming the axis, the first node where
    the order fails and the two coordinates there. #2748's grid is refused before any region is marked on it. An
    ascending axis builds, as do the axes that stop short of their bounds on purpose (#2769), and a single node. A
    built axis is the grid's own float64 copy."""
    if refusal is not None:
        with pytest.raises(ValueError, match=refusal):
            _grid(axes)
        return
    given = [np.asarray(a, dtype=float) for a in axes]
    grid = _grid(given)
    for held, axis in zip(grid.coordinates, given, strict=True):
        assert held.dtype == np.float64
        assert not np.shares_memory(held, axis)
        np.testing.assert_array_equal(held, axis)


def test_a_periodic_axis_short_of_its_bounds_builds():
    """#2769's periodic axis, np.linspace(0, 1, 20, endpoint=False), ascends and builds."""
    _grid([np.linspace(0.0, 1.0, 20, endpoint=False)], bc=periodic_bc(dimension=1))


@pytest.mark.parametrize("check", [_reversed_after_construction, _written_by_the_library], ids=lambda f: f.__name__)
def test_the_grid_owns_its_axes(check):
    """From the other side: the caller's array changed after construction, and a write into an axis."""
    check()


class _NotAGrid:
    """A geometry that is not a CartesianGrid and carries a ``coordinates`` attribute, which no reader may read: no
    order is declared for it. It marks the region "exit"."""

    dimension = 2

    @property
    def coordinates(self):
        raise AssertionError("a reader read the coordinates of a geometry that is not a CartesianGrid")

    def get_bounds(self):
        return np.zeros(2), np.ones(2)

    def mark_region(self, *args, **kwargs):
        raise NotImplementedError

    def get_region_mask(self, name):
        return np.ones(25, dtype=bool)

    def intersect_regions(self, *names):
        raise NotImplementedError

    def union_regions(self, *names):
        raise NotImplementedError

    def get_region_names(self):
        return ["exit"]


class _BoxWithCoordinates(Hyperrectangle):
    """An implicit box, which is no CartesianGrid, carrying a ``coordinates`` attribute no reader may read."""

    @property
    def coordinates(self):
        raise AssertionError("a reader read the coordinates of a geometry that is not a CartesianGrid")


def _bc():
    return BoundaryConditions(
        segments=[
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, region_name="exit"),
            BCSegment(name="rest", bc_type=BCType.NO_FLUX, value=0.0),
        ],
        dimension=2,
        domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
    )


def _region_node():
    segment = _bc().segments[0]
    with pytest.raises(ValueError, match=r"_NotAGrid is not a CartesianGrid, the type that declares node coordinates"):
        segment.matches_point(np.array([1.0, 0.5]), "x_max", np.array([[0.0, 1.0], [0.0, 1.0]]), geometry=_NotAGrid())


def _face_cell_centres():
    centres, _ = _bc().face_cell_centres(np.array([[0.0, 1.0], [0.0, 1.0]]), geometry=_NotAGrid())
    assert len(centres) > 0


def _grid_point_measure():
    from mfgarchon.operators.interaction.point_measure import GridPointMeasure

    with pytest.raises(TypeError, match=r"_NotAGrid, which is not a CartesianGrid"):
        GridPointMeasure(SimpleNamespace(geometry=_NotAGrid()))


def _get_spatial_grid():
    from mfgarchon.utils.pde_coefficients import get_spatial_grid

    with pytest.raises(AttributeError, match=r"_NotAGrid, which is not a CartesianGrid"):
        get_spatial_grid(SimpleNamespace(geometry=_NotAGrid()))


def _fp_fdm_node():
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import _node

    with pytest.raises(TypeError, match=r"FP-FDM reads node coordinates only from a CartesianGrid.*_NotAGrid"):
        _node(_NotAGrid(), (0, 0))


def _implicit_heat():
    """The solver evaluates its forcing on the grid's axes in ``solve_step``; it refuses the geometry when built."""
    import scipy.sparse as sp

    from mfgarchon.alg.numerical.pde_solvers import ImplicitHeatSolver

    class _HeatStub(_NotAGrid):
        num_spatial_points = 4

        def get_laplacian_operator(self, order=2, bc=None):
            return SimpleNamespace(get_matrix=lambda: -sp.eye(4, format="csr"))

    def build_and_step():
        solver = ImplicitHeatSolver(_HeatStub(), alpha=0.01, bc=no_flux_bc(dimension=2), forcing=lambda x, t: 0.0)
        solver.solve_step(np.zeros(4), 0.01)

    with pytest.raises(TypeError, match=r"ImplicitHeatSolver solves on a CartesianGrid.*_HeatStub"):
        build_and_step()


def _particle_bounds():
    from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver

    class _NoBounds(_NotAGrid):
        def get_bounds(self):
            raise AttributeError("no bounds")

        def get_grid_shape(self):
            return (5, 5)

    solver = object.__new__(FPParticleSolver)
    solver.problem = SimpleNamespace(geometry=_NoBounds())
    with pytest.raises(TypeError, match=r"cannot derive simulation bounds from _NoBounds"):
        solver._get_grid_params()


def _particle_axes():
    from mfgarchon import Conditions, MFGProblem, Model
    from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian

    box = _BoxWithCoordinates(np.array([[0.0, 1.0], [0.0, 1.0]]))
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.1
        ),
        domain=box,
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.1),
        Nt=2,
    )
    solver = FPParticleSolver(
        problem, num_particles=10, seed=0, density_mode="query_only", boundary_conditions=no_flux_bc(dimension=2)
    )
    params = solver._get_grid_params()
    for axis, (lo, hi), n in zip(params["coordinates"], params["bounds"], params["grid_shape"], strict=True):
        np.testing.assert_array_equal(axis, np.linspace(lo, hi, n))


_READERS = [
    _region_node,
    _face_cell_centres,
    _grid_point_measure,
    _get_spatial_grid,
    _fp_fdm_node,
    _implicit_heat,
    _particle_bounds,
    _particle_axes,
]


@pytest.mark.parametrize("reader", _READERS, ids=lambda f: f.__name__.lstrip("_"))
def test_a_reader_reads_node_coordinates_only_from_a_cartesian_grid(reader):
    """A geometry that is not a `CartesianGrid` and carries ``coordinates``, whose read fails the test, at each reader
    that read them off any geometry. Where the reader has no other way to proceed it refuses the geometry by name:
    the region reader, the grid point measure, `get_spatial_grid`, FP-FDM's node lookup, the implicit heat solver and
    the particle solver's bounds. Where it has one it takes it: the face cells are cut without node midpoints, and the
    particle solver estimates its density on axes from the bounds (#2757). Before, each of them read the attribute."""
    reader()

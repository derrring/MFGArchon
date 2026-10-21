"""#2490, the folded #2748: an axis's node order is its coordinate order, and the grid owns it.

Every reader of a grid's ``coordinates`` takes index order as coordinate order: the face rule and region marking read
node 0 as the min face, the measure's trapezoid weights are the gaps between consecutive nodes, and a point is read at
its nearest node. Before, `TensorProductGrid`'s custom branch admitted an axis in any order and of any shape, kept the
caller's array, and held the axes in a list. A descending axis put a region on the opposite face and gave the measure
weights summing to -1, a swapped pair gave positive weights that integrate wrongly with nothing raised, and an axis
reordered after construction undid any check: through the caller's array, an element of the list, or a pickled or
deep-copied grid's writeable axes. Now every axis enters through one owner, `TensorProductGrid._own_axes`, at
construction and in `__setstate__`: it refuses an axis that is not 1-D, of its length and strictly ascending, and holds
each as its own read-only float64 copy, in a tuple. `CartesianGrid` declares the attribute and its order, and a reader
reads it only from that type."""

from __future__ import annotations

import ast
import copy
import pathlib
import pickle
from types import SimpleNamespace

import pytest

import numpy as np

from mfgarchon.geometry import Hyperrectangle, TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc, periodic_bc

_ASCENDING = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_DESCENDING = _ASCENDING[::-1].copy()
_ORDER_REFUSED = r"{name} does not strictly ascend: node {k} is at {a} and node {k1} at {b}.*#2490"


def _grid(axes, bc=None):
    axes = [np.asarray(a) for a in axes]
    return TensorProductGrid(
        bounds=[(0.0, 1.0)] * len(axes),
        Nx_points=[len(a) for a in axes],
        spacing_type="custom",
        custom_coordinates=axes,
        boundary_conditions=bc if bc is not None else no_flux_bc(dimension=len(axes)),
    )


def _refused(axes, name, k, a, b):
    with pytest.raises(ValueError, match=_ORDER_REFUSED.format(name=name, k=k, a=a, k1=k + 1, b=b)):
        _grid(axes)


def _built(axes):
    """It builds, and each axis is the grid's own read-only float64 copy, equal to what was given, in a tuple."""
    given = [np.asarray(a) for a in axes]
    grid = _grid(given)
    assert isinstance(grid.coordinates, tuple)
    for held, axis in zip(grid.coordinates, given, strict=True):
        assert held.dtype == np.float64
        assert not held.flags.writeable
        assert not np.shares_memory(held, axis)
        np.testing.assert_array_equal(held, axis)


def _a_column():
    with pytest.raises(ValueError, match=r"custom_coordinates\[0\] has shape \(5, 1\): .* 1-D array of its nodes"):
        _grid([_DESCENDING.reshape(5, 1), _ASCENDING])


def _unresolved_uniform_bounds():
    """101 uniform nodes on (1e16, 1e16 + 4) collapse in float64: refused, naming the uniform axis, not an argument the
    caller never passed."""
    with pytest.raises(
        ValueError, match=r"the uniform axis 0, 101 nodes on bounds .* does not strictly ascend.*resolve"
    ):
        TensorProductGrid(bounds=[(1e16, 1e16 + 4.0)], Nx_points=[101], boundary_conditions=no_flux_bc(dimension=1))


def _caller_reverses_its_array():
    """The caller reverses its own array in place once the grid holds it: the grid's axis and measure do not move."""
    x = _ASCENDING.copy()
    grid = _grid([x, _ASCENDING])
    before = np.array(grid.measure_weights())
    x[:] = x[::-1].copy()
    assert grid.coordinates[0].tolist() == _ASCENDING.tolist()
    np.testing.assert_array_equal(grid.measure_weights(), before)


def _element_replaced():
    """An axis cannot be replaced in place: ``coordinates`` is a tuple."""
    grid = _grid([_ASCENDING, _ASCENDING])
    with pytest.raises(TypeError, match="does not support item assignment"):
        grid.coordinates[0] = _DESCENDING


def _written_into():
    """A write into an axis raises, on the custom branch and on the uniform one."""
    uniform = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[5], boundary_conditions=no_flux_bc(dimension=1))
    for grid in (_grid([_ASCENDING]), uniform):
        with pytest.raises(ValueError, match="read-only"):
            grid.coordinates[0][0] = 2.0


def _deep_copied_then_written():
    """A deep copy's axes are owned again, so a write into one raises, as on the original."""
    grid = copy.deepcopy(_grid([_ASCENDING, _ASCENDING]))
    assert isinstance(grid.coordinates, tuple)
    with pytest.raises(ValueError, match="read-only"):
        grid.coordinates[0][:] = _DESCENDING


def _pickled_then_written():
    """numpy arrays unpickle writeable whatever their flag was; the round-tripped grid's axes are owned again."""
    grid = pickle.loads(pickle.dumps(_grid([_ASCENDING, _ASCENDING])))
    np.testing.assert_array_equal(grid.coordinates[0], _ASCENDING)
    with pytest.raises(ValueError, match="read-only"):
        grid.coordinates[0][:] = _DESCENDING


def _restored_with_a_descending_axis():
    """A state holding a descending axis, as a pickle from before this change could, is refused when it is loaded."""
    state = dict(_grid([_ASCENDING, _ASCENDING]).__dict__)
    state["coordinates"] = [_DESCENDING, _ASCENDING]
    with pytest.raises(
        ValueError, match=_ORDER_REFUSED.format(name="the restored grid's axis 0", k=0, a=1.0, k1=1, b=0.75)
    ):
        TensorProductGrid.__new__(TensorProductGrid).__setstate__(state)


# (row, the check). Refused by name, built and owned, or, from the other side, a route that would reorder an axis
# after it was built.
_ROWS = [
    (
        "a descending axis, axis 0",
        lambda: _refused([_DESCENDING, _ASCENDING], r"custom_coordinates\[0\]", 0, 1.0, 0.75),
    ),
    (
        "a descending axis, axis 1",
        lambda: _refused([_ASCENDING, _DESCENDING], r"custom_coordinates\[1\]", 0, 1.0, 0.75),
    ),
    (
        "a swapped pair, axis 0",
        lambda: _refused([[0.0, 0.5, 0.25, 0.75, 1.0], _ASCENDING], r"custom_coordinates\[0\]", 1, 0.5, 0.25),
    ),
    (
        "a swapped pair, axis 1",
        lambda: _refused([_ASCENDING, [0.0, 0.5, 0.25, 0.75, 1.0]], r"custom_coordinates\[1\]", 1, 0.5, 0.25),
    ),
    (
        "a repeated node, axis 0",
        lambda: _refused([[0.0, 0.25, 0.25, 0.75, 1.0], _ASCENDING], r"custom_coordinates\[0\]", 1, 0.25, 0.25),
    ),
    (
        "a repeated node, axis 1",
        lambda: _refused([_ASCENDING, [0.0, 0.25, 0.25, 0.75, 1.0]], r"custom_coordinates\[1\]", 1, 0.25, 0.25),
    ),
    (
        "a NaN node",
        lambda: _refused([[0.0, 0.25, np.nan, 0.75, 1.0], _ASCENDING], r"custom_coordinates\[0\]", 1, 0.25, "nan"),
    ),
    (
        "#2748's grid, x nodes 1.0 ... 0.0",
        lambda: _refused([_DESCENDING, [0.0, 0.5, 1.0]], r"custom_coordinates\[0\]", 0, 1.0, 0.75),
    ),
    ("#2748's x nodes as a (5, 1) column", _a_column),
    ("uniform bounds that cannot resolve their nodes", _unresolved_uniform_bounds),
    ("an ascending axis", lambda: _built([_ASCENDING, [0.0, 0.1, 0.4, 1.0]])),
    ("an integer axis, held as float64", lambda: _built([np.arange(5), _ASCENDING])),
    ("#2769's cell-centred axis, short of its bounds", lambda: _built([np.arange(20) / 20 + 1 / 40])),
    ("a single node", lambda: _built([[0.5]])),
    ("the caller's array reversed after construction", _caller_reverses_its_array),
    ("an element of coordinates replaced", _element_replaced),
    ("a write into an axis", _written_into),
    ("a deep copy, then a write into its axis", _deep_copied_then_written),
    ("a pickle round trip, then a write into its axis", _pickled_then_written),
    ("a restored state holding a descending axis", _restored_with_a_descending_axis),
]


@pytest.mark.parametrize(("row", "check"), _ROWS, ids=[r[0] for r in _ROWS])
def test_an_axis_strictly_ascends_and_the_grid_owns_it(row, check):
    """An axis that is not 1-D, descends, swaps two nodes, repeats one or holds a NaN is refused by name, naming the
    axis, the first node where the order fails and the two coordinates there; a uniform axis whose bounds cannot
    resolve its nodes is refused naming its bounds. #2748's grid is refused before any region is marked on it, as an
    axis and as a column. An ascending axis builds, as do an integer axis, the axes that stop short of their bounds on
    purpose (#2769) and a single node, each held as the grid's own read-only float64 copy in a tuple. From the other
    side, every route that reordered an axis after construction at 932b82ae is closed: the caller's array, an element
    of the tuple, a write into an axis, a deep copy or a pickle round trip followed by one, and a restored state."""
    check()


def test_a_periodic_axis_short_of_its_bounds_builds():
    """#2769's periodic axis, np.linspace(0, 1, 20, endpoint=False), ascends and builds."""
    _grid([np.linspace(0.0, 1.0, 20, endpoint=False)], bc=periodic_bc(dimension=1))


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
    """No node midpoints are cut for a geometry that is no grid: its cells are those of no geometry at all."""
    box = np.array([[0.0, 1.0], [0.0, 1.0]])
    centres, lattice = _bc().face_cell_centres(box, geometry=_NotAGrid())
    expected, expected_lattice = _bc().face_cell_centres(box, geometry=None)
    assert lattice == expected_lattice
    assert len(centres) == len(expected)
    for got, want in zip(centres, expected, strict=True):
        np.testing.assert_array_equal(got, want)


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


def _fp_fdm_every_load_through_the_helper():
    """Every load of ``.coordinates`` in FP-FDM time-stepping is the helper's own: a call site that read the attribute
    around `_grid_axes` would read it off any geometry again. The helper's load is the control that the walk sees
    loads at all."""
    import mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping as module

    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    loads = []
    for function in ast.walk(tree):
        if isinstance(function, ast.FunctionDef):
            loads += [
                (function.name, node.lineno)
                for node in ast.walk(function)
                if isinstance(node, ast.Attribute) and node.attr == "coordinates" and isinstance(node.ctx, ast.Load)
            ]
    assert loads, "the walk found no load of .coordinates at all; it cannot see the call sites it checks"
    assert {name for name, _ in loads} == {"_grid_axes"}, f"loads of .coordinates outside _grid_axes: {loads}"


def _implicit_heat():
    """The solver evaluates its forcing on the grid's axes, and refuses a geometry that is no grid when it is built."""
    import scipy.sparse as sp

    from mfgarchon.alg.numerical.pde_solvers import ImplicitHeatSolver

    class _HeatStub(_NotAGrid):
        num_spatial_points = 4

        def get_laplacian_operator(self, order=2, bc=None):
            return SimpleNamespace(get_matrix=lambda: -sp.eye(4, format="csr"))

    with pytest.raises(TypeError, match=r"ImplicitHeatSolver solves on a CartesianGrid.*_HeatStub"):
        ImplicitHeatSolver(_HeatStub(), alpha=0.01, bc=no_flux_bc(dimension=2), forcing=lambda x, t: 0.0)


def _particle_bounds():
    """The bounds no longer fall back to a geometry's axes: one with neither ``get_bounds()`` nor ``xmin``/``xmax`` is
    refused by name, without its ``coordinates`` being read."""
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
    """The density axes for a geometry that is no grid come from its bounds, the unit square, not from its axes."""
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
    assert params["bounds"] == [(0.0, 1.0), (0.0, 1.0)]
    for axis, n in zip(params["coordinates"], params["grid_shape"], strict=True):
        np.testing.assert_array_equal(axis, np.linspace(0.0, 1.0, n))


_READERS = [
    _region_node,
    _face_cell_centres,
    _grid_point_measure,
    _get_spatial_grid,
    _fp_fdm_node,
    _fp_fdm_every_load_through_the_helper,
    _implicit_heat,
    _particle_bounds,
    _particle_axes,
]


@pytest.mark.parametrize("reader", _READERS, ids=lambda f: f.__name__.lstrip("_"))
def test_a_reader_reads_node_coordinates_only_from_a_cartesian_grid(reader):
    """A geometry that is not a `CartesianGrid` and carries ``coordinates``, whose read fails the test, at each reader
    that read them off any geometry. Where the reader cannot proceed without node coordinates it refuses the geometry
    by name: the region reader, the grid point measure, `get_spatial_grid`, FP-FDM's node lookups, all through one
    helper, and the implicit heat solver when it is built. Where it has another path it takes that one: the face cells
    are cut as for no geometry, and the particle solver takes its bounds from ``get_bounds()`` and its density axes,
    for a geometry that is no grid, from those bounds (#2757). Before, each of them read the attribute."""
    reader()

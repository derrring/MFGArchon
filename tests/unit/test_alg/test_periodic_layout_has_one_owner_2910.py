"""#2910: how many distinct nodes each wrapping axis spans on a Cartesian grid has one owner, `periodic_axis_spans`, and
its readers report what it says.

The oracles of the approved plan (#2910, issuecomment-6724735013), stage 1:
- cross-reader identity: each reader gives the span the rule gives. The rule, written here independently of the code:
  an axis that does not wrap has no span; one that wraps spans n - 1 distinct nodes where its last node repeats node 0
  (endpoint-inclusive), and n where it is distinct (endpoint-exclusive);
- twin nodes equal: on an endpoint-inclusive grid every FP solver returns node 0 and its repeat equal, from a value
  field whose repeat differs from node 0 (#2909 review 3, SF5).
A cell red at stage 1's head is ``xfail(strict=True)`` naming the stage that moves its reader. When it turns green its
strict XPASS fails the run and prints the reason, which says what to do.
"""

import re
import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling.fixed_point_utils import compute_fp_velocity_field
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver, FPFVMSolver, FPParticleSolver, FPSLSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.base import CartesianGrid
from mfgarchon.geometry.boundary import BCSegment, BoundaryConditions, no_flux_bc, periodic_bc
from mfgarchon.geometry.boundary.bc_utils import with_geometry_periodic_convention
from mfgarchon.geometry.boundary.conditions import periodic_axis_span
from mfgarchon.geometry.boundary.types import BCType, PeriodicGridConvention
from mfgarchon.geometry.grids.tensor_grid import periodic_axis_spans

N = 9
STAGE_2 = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="remove this xfail: stage 2 of #2910/#2911 moved this reader to the layout owner, so it now agrees",
)
P, W = BCType.PERIODIC, BCType.NO_FLUX
INC, EXC = PeriodicGridConvention.ENDPOINT_INCLUSIVE, PeriodicGridConvention.ENDPOINT_EXCLUSIVE


def _faces(kinds):
    segments = []
    for axis, name in enumerate("xy"):
        for side, kind in zip(("min", "max"), kinds[axis], strict=True):
            segments.append(BCSegment(name=f"{name}{side}", bc_type=kind, boundary=f"{name}_{side}"))
    return BoundaryConditions(segments=segments, dimension=2)


#: spelling -> (the boundary condition, the span the rule gives on each axis of an endpoint-inclusive 9 x 9 grid)
SPELLINGS = {
    "uniform": (lambda: periodic_bc(dimension=2), (N - 1, N - 1)),
    "per face": (lambda: _faces([(P, P), (P, P)]), (N - 1, N - 1)),
    "default_bc": (lambda: BoundaryConditions(segments=[], default_bc=P, dimension=2), (N - 1, N - 1)),
    "cylinder, x wraps": (lambda: _faces([(P, P), (W, W)]), (N - 1, None)),
    "cylinder, y wraps": (lambda: _faces([(W, W), (P, P)]), (None, N - 1)),
}


def _grid(bc):
    return TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 2.0)], Nx_points=[N, N], boundary_conditions=bc)


@pytest.mark.parametrize("spelling", list(SPELLINGS))
def test_the_layout_owner_gives_the_span_the_rule_gives(spelling):
    make, expected = SPELLINGS[spelling]
    bc = make()
    assert periodic_axis_spans(_grid(bc), bc) == expected


@pytest.mark.parametrize(
    "spelling",
    [pytest.param(s, marks=STAGE_2) if "cylinder" in s else s for s in SPELLINGS],
)
def test_the_bc_field_reader_gives_the_same_span(spelling):
    """`periodic_axis_span` reads the BC's convention field and wraps every axis or none; stage 2 moves it."""
    make, expected = SPELLINGS[spelling]
    grid = _grid(make())
    bc = grid.get_boundary_conditions()
    assert tuple(periodic_axis_span(bc, N, 2, grid) for _ in range(2)) == expected


def _mixed(bc):
    """Axis 0 endpoint-inclusive (11 nodes, x[-1] = 1), axis 1 endpoint-exclusive (10 nodes, y[-1] = 0.9)."""
    return TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[11, 10],
        spacing_type="custom",
        custom_coordinates=[np.linspace(0, 1, 11), np.linspace(0, 1, 10, endpoint=False)],
        boundary_conditions=bc,
    )


def test_a_grid_whose_axes_differ_in_layout_has_no_one_convention():
    with pytest.raises(ValueError, match=r"axes differ in where the last node sits.*periodic_axis_spans"):
        _ = _mixed(_faces([(W, W), (W, W)])).periodic_convention


#: case -> (the BC the grid holds, the convention the rule binds on it, the spans the rule gives). On a grid whose axes
#: differ in layout only the wrapped axes' layouts are read: none wrapped binds nothing (#2916 review 1, X2), one wrapped
#: binds that axis's layout, and the owner reads every wrapped axis on its own (X1).
MIXED = {
    "a periodic default every face overrides": (
        lambda: BoundaryConditions(segments=_faces([(W, W), (W, W)]).segments, default_bc=P, dimension=2),
        None,
        (None, None),
    ),
    "a cylinder over the inclusive axis": (lambda: _faces([(P, P), (W, W)]), INC, (10, None)),
    "a cylinder over the exclusive axis": (lambda: _faces([(W, W), (P, P)]), EXC, (None, 10)),
}


@pytest.mark.parametrize("case", list(MIXED))
def test_a_grid_whose_axes_differ_binds_the_layout_of_the_axes_its_bc_wraps(case):
    make, convention, spans = MIXED[case]
    grid = _mixed(make())
    bc = grid.get_boundary_conditions()
    assert bc.periodic_convention is convention
    assert periodic_axis_spans(grid, bc) == spans


def test_the_layout_owner_reads_each_wrapped_axis_of_a_grid_whose_axes_differ():
    assert periodic_axis_spans(_mixed(no_flux_bc(dimension=2)), periodic_bc(dimension=2)) == (10, 10)


def test_a_grid_whose_axes_differ_builds_with_a_segment_placed_by_region():
    """The binding reads a placed segment's faces through the grid, so the grid's caches exist before it (#2916
    review 2, N1): a cylinder over the exclusive axis, with a no-flux strip placed by `region` on a wall face."""
    strip = BCSegment(name="strip", bc_type=W, boundary="x_min", region={"y": (0.35, 0.65)}, priority=1)
    bc = BoundaryConditions(segments=[*_faces([(W, W), (P, P)]).segments, strip], dimension=2)
    assert _mixed(bc).get_boundary_conditions().periodic_convention is EXC


def test_a_one_face_bc_on_a_grid_whose_axes_differ_is_refused_for_its_face():
    """Which axes a BC wraps is the topology owner's (#2916 review 2, N2): periodic on the min face of axis 0 only,
    beside a wrapped axis 1, is refused for that face, not for the layouts of axes it does not wrap."""
    with pytest.raises(ValueError, match=r"periodic on the min face of axis 0 and not on its max face"):
        _mixed(_faces([(P, W), (P, P)]))


def _set(grid, bc):
    grid.boundary_conditions = bc


#: The public entries that bind a BC to a TensorProductGrid, each refusing a torus whose axes differ in layout with the
#: binder's message (a BC carries one convention; at ad1e800d it was bound ENDPOINT_EXCLUSIVE, so the inclusive axis
#: spanned 11 nodes where it has 10).
TORUS_READERS = {
    "TensorProductGrid": lambda: _mixed(periodic_bc(dimension=2)),
    "TensorProductGrid.set_boundary_conditions": lambda: _mixed(no_flux_bc(dimension=2)).set_boundary_conditions(
        periodic_bc(dimension=2)
    ),
    "TensorProductGrid.boundary_conditions = bc": lambda: _set(
        _mixed(no_flux_bc(dimension=2)), periodic_bc(dimension=2)
    ),
    "TensorProductGrid.get_laplacian_operator(bc=)": lambda: _mixed(no_flux_bc(dimension=2)).get_laplacian_operator(
        bc=periodic_bc(dimension=2)
    ),
    "TensorProductGrid.get_gradient_operator(bc=)": lambda: _mixed(no_flux_bc(dimension=2)).get_gradient_operator(
        bc=periodic_bc(dimension=2)
    ),
    "with_geometry_periodic_convention": lambda: with_geometry_periodic_convention(
        periodic_bc(dimension=2), _mixed(no_flux_bc(dimension=2))
    ),
}


@pytest.mark.parametrize("reader", list(TORUS_READERS))
def test_every_reader_refuses_one_convention_for_a_torus_whose_axes_differ(reader):
    with pytest.raises(ValueError, match=r"wraps axes \[0, 1\], which differ in where the last node sits"):
        TORUS_READERS[reader]()


class _Recording(SeparableHamiltonian):
    """The library quadratic H, recording where the velocity builder asks for its control."""

    def __init__(self):
        super().__init__(QuadraticControlCost(control_cost=1.0))
        self.points = []

    def evaluate_optimal_control(self, state):
        self.points.append(np.asarray(state.x, dtype=float).ravel())
        return super().evaluate_optimal_control(state)


def _problem_1d(n=21):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=periodic_bc(dimension=1))
    return MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(QuadraticControlCost(control_cost=1.0)), volatility=0.3),
        domain=grid,
        conditions=Conditions(
            m_initial=lambda x: 1.0 + 0.5 * np.cos(2 * np.pi * np.asarray(x[0])), u_terminal=lambda x: 0.0, T=0.3
        ),
        Nt=6,
    )


def test_the_velocity_builder_asks_at_the_faces():
    """The faces sit midway between neighbours, inside the domain (#2909 review 3, SF10). The face across the wrap is
    #2904's census's to pin."""
    problem, n = _problem_1d(), 21
    x = np.linspace(0.0, 1.0, n)
    U = np.stack([0.3 * np.sin(2 * np.pi * x) + 0.2 * x] * 2)
    recording = _Recording()
    compute_fp_velocity_field(problem, U, np.ones_like(U), recording)
    np.testing.assert_allclose(np.concatenate(recording.points[:1]), 0.5 * (x[:-1] + x[1:]), atol=1e-14)


class _StatesItsLayout:
    """A geometry that is not a `TensorProductGrid`, stating ``layout`` (None: none), delegating the rest to a grid."""

    def __init__(self, grid, layout):
        self._grid, self.periodic_convention = grid, layout

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._grid, name)


class _Problem:
    def __init__(self, geometry, bc):
        self.geometry, self.dt, self._bc = geometry, 0.05, bc

    def get_boundary_conditions(self):
        return self._bc


@pytest.mark.parametrize(("geometry_states", "bc_states"), [(INC, None), (None, INC)], ids=["the grid", "the BC"])
def test_the_velocity_builder_reads_the_same_span_on_a_grid_that_states_its_layout(geometry_states, bc_states):
    """Cross-reader identity across geometries: on a geometry that is not a `TensorProductGrid`, a layout stated by the
    geometry (#2909 review 3, F1(a)) or, where it states none, by the BC (#2916 review 1, X4) builds what the builder
    builds on the grid itself."""
    n = 21
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=periodic_bc(dimension=1))
    x = np.linspace(0.0, 1.0, n)
    U = np.stack([0.3 * np.sin(2 * np.pi * x) + 0.2 * x] * 2)
    H = SeparableHamiltonian(QuadraticControlCost(control_cost=1.0))

    def build(geometry):
        bc = periodic_bc(dimension=1, convention=bc_states)
        return compute_fp_velocity_field(_Problem(geometry, bc), U, np.ones_like(U), H, boundary_conditions=bc)

    np.testing.assert_array_equal(build(_StatesItsLayout(grid, geometry_states)), build(grid))


@pytest.mark.parametrize(
    "solver",
    ["FP-FDM", "FP-FVM", "FP-SL", pytest.param("FP-Particle", marks=STAGE_2)],
)
def test_every_fp_solver_returns_twin_nodes_equal(solver):
    problem, n = _problem_1d(), 21
    x = np.linspace(0.0, 1.0, n)
    m0 = 1.0 + 0.5 * np.cos(2 * np.pi * x)
    U = np.stack([0.3 * np.sin(2 * np.pi * x) + 0.2 * x] * (problem.Nt + 1))
    make = {
        "FP-FDM": lambda: FPFDMSolver(problem),
        "FP-FVM": lambda: FPFVMSolver(problem),
        "FP-SL": lambda: FPSLSolver(problem),
        "FP-Particle": lambda: FPParticleSolver(problem, num_particles=4000, seed=0),
    }[solver]
    M = np.asarray(make().solve_fp_system(m0, potential_field=U, show_progress=False))
    np.testing.assert_allclose(M[:, 0], M[:, -1], atol=1e-12 * np.abs(M).max())


def _user_grid(d, n, nodes, states):
    """A Cartesian grid that is not a `TensorProductGrid`: on [0, 1) per axis, the n nodes x = k / n, plus x = 1 when
    ``nodes`` is "inclusive"; ``states`` is the layout it states as ``periodic_convention`` (None: none). Its BC is a
    periodic one stating the same layout.

    It answers its own index maps and point count. Members it does not define fall through to an inclusive
    `TensorProductGrid` of (n + 1)^d nodes; on an exclusive grid an index map read from there would give that grid's
    answer, so the grid checks its own before it is used (#2916 review 2, B1)."""
    m = n if nodes == "exclusive" else n + 1
    h = 1.0 / n
    axis = np.arange(m) * h
    inner = TensorProductGrid(
        bounds=[(0.0, 1.0)] * d, Nx_points=[n + 1] * d, boundary_conditions=periodic_bc(dimension=d)
    )
    mesh = np.meshgrid(*[axis] * d, indexing="ij")

    class UserGrid(CartesianGrid):
        dimension = property(lambda self: d)
        geometry_type = property(lambda self: inner.geometry_type)
        num_spatial_points = property(lambda self: m**d)
        periodic_convention = property(lambda self: states)
        coordinates = property(lambda self: tuple([axis] * d))
        bounds = property(lambda self: [(0.0, 1.0)] * d)
        Nx_points = property(lambda self: [m] * d)
        spacing = property(lambda self: [h] * d)

        def get_spatial_grid(self):
            return np.stack([g.ravel() for g in mesh], axis=1)

        def get_bounds(self):
            return np.zeros(d), np.ones(d)

        def get_problem_config(self):
            shape = (m,) * d
            bounds = [(0.0, 1.0)] * d
            return {"num_spatial_points": m**d, "spatial_shape": shape, "spatial_bounds": bounds,
                    "spatial_discretization": [m] * d, "legacy_1d_attrs": None}  # fmt: skip

        def get_grid_shape(self):
            return (m,) * d

        def get_grid_spacing(self):
            return [h] * d

        def get_boundary_conditions(self):
            return periodic_bc(dimension=d, convention=states)

        def measure_weights(self, *args, **kwargs):
            return np.full((m,) * d, h**d)

        def total_points(self):
            return m**d

        def get_index(self, multi_index):
            return int(np.ravel_multi_index(tuple(int(i) for i in multi_index), (m,) * d))

        def get_multi_index(self, flat_index):
            return tuple(int(i) for i in np.unravel_index(int(flat_index), (m,) * d))

        def get_collocation_points(self):
            return self.get_spatial_grid()

        def get_boundary_handler(self, *args, **kwargs):
            raise NotImplementedError

        get_gradient_operator = get_interpolator = get_laplacian_operator = get_boundary_handler

        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return getattr(inner, name)

    traits = set().union(*(getattr(base, "__protocol_attrs__", set()) for base in TensorProductGrid.__mro__))
    for name in sorted(traits - set(UserGrid.__dict__)):
        member = getattr(TensorProductGrid, name, None)
        if isinstance(member, property):
            setattr(UserGrid, name, property(lambda self, name=name: getattr(inner, name)))
        else:
            setattr(UserGrid, name, lambda self, *a, name=name, **k: getattr(inner, name)(*a, **k))
    grid = UserGrid()
    assert grid.total_points() == m**d == grid.get_spatial_grid().shape[0]
    points = grid.get_spatial_grid()
    for flat in range(m**d):
        index = grid.get_multi_index(flat)
        assert grid.get_index(index) == flat
        np.testing.assert_array_equal(points[flat], [axis[i] for i in index])
    assert grid.get_multi_index(m**d - 1) == (m - 1,) * d
    return grid, mesh


_QUADRATIC = SeparableHamiltonian(QuadraticControlCost(control_cost=1.0))


def _on(domain):
    """The problem the FP solvers read; its m_initial is never solved from (each solve is handed m0), so the mass report
    MFGProblem makes for it is silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MFGProblem(
            model=Model(hamiltonian=_QUADRATIC, volatility=0.3),
            domain=domain,
            conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.5),
            Nt=6,
        )


def _density(problem, mesh, route, bc=None):
    """FP-FDM's density from a periodic U and m0 on ``mesh``, by the value route or the velocity route."""
    U = np.stack([sum(0.3 * np.sin(2 * np.pi * x + k) for k, x in enumerate(mesh))] * (problem.Nt + 1))
    m0 = 1.0 + 0.5 * np.prod([np.cos(2 * np.pi * x) for x in mesh], axis=0)
    solver = FPFDMSolver(problem) if bc is None else FPFDMSolver(problem, boundary_conditions=bc)
    if route == "value":
        return np.asarray(solver.solve_fp_system(m0, potential_field=U, show_progress=False))
    velocity = compute_fp_velocity_field(problem, U, np.stack([m0] * (problem.Nt + 1)), _QUADRATIC)
    return np.asarray(solver.solve_fp_system(m0, drift_field=velocity, show_progress=False))


#: (dimension, route) -> the limit its refusal names, where the exclusive layout is refused: the velocity builder's
#: 1-D faces number n - 1, and the wrap has n. At ad1e800d this cell crashed with an IndexError.
EXCLUSIVE_LIMITS = {
    (1, "velocity"): r"this 1-D grid wraps with its last node distinct from node 0 .* the 1-D faces this builds number",
}
STAGE_2_STENCIL = pytest.mark.xfail(
    strict=True, raises=NotImplementedError, reason="remove this xfail: stage 2 moved this stencil to the span"
)


@pytest.mark.parametrize(
    ("dimension", "route"),
    [
        pytest.param(d, route, marks=STAGE_2_STENCIL) if (d, route) in EXCLUSIVE_LIMITS else (d, route)
        for d in (1, 2, 3)
        for route in ("value", "velocity")
    ],
)
def test_one_torus_in_two_layouts_gives_one_density(dimension, route):
    """The oracle of the exclusive layout, one torus in two layouts (#2916 review 1, approver's ruling on item 3): the
    n nodes x = k / n of an exclusive user grid are the first n of an inclusive TensorProductGrid of n + 1 nodes, so on
    them FP-FDM's density must agree. The user grid answers its own index maps (`_user_grid`). A cell refused by name
    is an expected failure for stage 2, which takes this as its acceptance check; one that fails any other way, as an
    IndexError does, is a failure."""
    n = {1: 20, 2: 8, 3: 4}[dimension]
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * dimension,
        Nx_points=[n + 1] * dimension,
        boundary_conditions=periodic_bc(dimension=dimension),
    )
    oracle = _density(_on(grid), np.meshgrid(*grid.coordinates, indexing="ij"), route)
    exclusive, mesh = _user_grid(dimension, n, "exclusive", EXC)
    try:
        density = _density(_on(exclusive), mesh, route)
    except NotImplementedError as refusal:
        if not re.search(EXCLUSIVE_LIMITS.get((dimension, route), r"(?!)"), str(refusal)):
            raise AssertionError(f"refused, but not naming this cell's limit: {refusal}") from refusal
        raise
    shared = (slice(None), *[slice(0, n)] * dimension)
    np.testing.assert_allclose(density[shared], oracle[shared], rtol=0, atol=1e-10)


#: Inputs the layout owner refuses, on an inclusive grid: (the layout the grid states, "measured" for a
#: TensorProductGrid, the layout the BC states, the owner's message). At ad1e800d the builder read the layout the BC
#: states and not the grid's, and a TensorProductGrid refused the contradiction when binding.
LAYOUT_REFUSALS = {
    "neither states a layout": (None, None, r"neither it nor the periodic boundary condition states a node layout"),
    "the two contradict": (
        INC,
        EXC,
        r"states the layout ENDPOINT_INCLUSIVE, but the periodic boundary condition declares",
    ),
    "a TensorProductGrid and its BC contradict": (
        "measured",
        EXC,
        r"coordinates are ENDPOINT_INCLUSIVE on axis 0 .* but "
        r"the periodic boundary condition declares ENDPOINT_EXCLUSIVE",
    ),
}
#: Readers that reach such an input. FP-FDM reads the BC's convention field and does not ask the owner, in any
#: dimension; stage 2 moves it. FP-FVM and HJB-FDM reach these inputs too and are #2910's rows for stage 2.
STAGE_2_GUESS = pytest.mark.xfail(
    strict=True, raises=pytest.fail.Exception, reason="remove this xfail: stage 2 moved this reader to the layout owner"
)


def _layout_reader(reader, geometry_states, bc_states):
    d = 1 if reader.startswith("FP-FDM, 1-D") else 2
    if geometry_states == "measured":
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0)] * d, Nx_points=[9] * d, boundary_conditions=no_flux_bc(dimension=d)
        )
        mesh = np.meshgrid(*grid.coordinates, indexing="ij")
    else:
        grid, mesh = _user_grid(d, 8, "inclusive", geometry_states)
    bc = periodic_bc(dimension=d, convention=bc_states)
    if reader == "periodic_axis_spans":
        return periodic_axis_spans(grid, bc)
    if reader == "compute_fp_velocity_field":
        U = np.ones((2, *grid.get_grid_shape()))
        return compute_fp_velocity_field(_Problem(grid, bc), U, U, _QUADRATIC, boundary_conditions=bc)
    return _density(_on(grid), mesh, "value", bc=bc)


@pytest.mark.parametrize(
    "reader",
    [
        "periodic_axis_spans",
        "compute_fp_velocity_field",
        pytest.param("FP-FDM, 1-D value route", marks=STAGE_2_GUESS),
        pytest.param("FP-FDM, 2-D value route", marks=STAGE_2_GUESS),
    ],
)
@pytest.mark.parametrize("case", list(LAYOUT_REFUSALS))
def test_every_reader_refuses_a_layout_the_owner_refuses(case, reader):
    geometry_states, bc_states, message = LAYOUT_REFUSALS[case]
    with pytest.raises(ValueError, match=message):
        _layout_reader(reader, geometry_states, bc_states)


def test_the_builder_checks_u_against_the_grid_before_reading_the_layout():
    """#2756's shape check runs before the layout's 1-D check (#2916 review 2, B2): a U one node short on a periodic
    TensorProductGrid is refused for its shape, not as the exclusive layout its node count would suggest."""
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=periodic_bc(dimension=1))
    U = np.ones((2, 20))
    with pytest.raises(ValueError, match=r"U has spatial shape \(20,\).* no tensor grid of that shape .*\(#2756\)"):
        compute_fp_velocity_field(_Problem(grid, grid.get_boundary_conditions()), U, U, _QUADRATIC)

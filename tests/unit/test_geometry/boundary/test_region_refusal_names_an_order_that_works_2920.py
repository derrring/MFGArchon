"""A TensorProductGrid refusing a ``region_name`` segment it has not marked names the order the user ruled supported
(#2920): mark the region, attach the BC with ``set_boundary_conditions``, read it back with ``get_boundary_conditions``.

The message is compared whole. The route is run over the axes its path branches on -- the grid's layout (one, or
mixed), whether the BC is periodic, and the face and cover of the region (`bind_periodic_convention`, `periodic_faces`,
`_region_faces`) -- and after it, the call that had refused does not refuse this way again: it reads the segment, or
refuses for a reason of its own, named per cell. The point rule is read on the face the point lies on; read at a
corner without a face, a region on the corner node alone gives the face's own segment (#2573's ledger). The segment and
the region have different names, and the message names the region. Every other geometry keeps the text it had.
"""

import pytest

import numpy as np

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.base import CartesianGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc
from mfgarchon.geometry.graph.network_geometry import GridNetwork
from mfgarchon.geometry.implicit.hyperrectangle import Hyperrectangle

REFUSAL = "defines no region of that name"
HEAD = (
    "BC segment 'exit' is restricted to the marked region 'door', and which locations a region covers is known only "
    "to the geometry that marked it. {grid} defines no region of that name, so it cannot tell whether the "
    "segment {what}. "
)
ADVICE = (
    "Mark the region on the grid first, mark_region('door', ...), then attach this BC with set_boundary_conditions and "
    "read it back with get_boundary_conditions(). A TensorProductGrid whose axes differ in node layout reads a periodic "
    "BC's faces when it is built, before any region can be marked, so build it with a BC that names no unmarked region. "
    "Attaching checks the BC against the grid, and can refuse it for a reason of its own (#2920)."
)
BASE_ADVICE = (
    "Build the BC with mixed_bc_from_regions(geometry, ...), which resolves a whole-face region to its faces, or name "
    "the face with BCSegment(boundary=...) (Issue #2472)."
)


def _grid(layout, bc):
    if layout == "mixed":
        # The issue's grid: axis 0 endpoint-inclusive (11 nodes), axis 1 endpoint-exclusive (10 nodes).
        return TensorProductGrid(
            bounds=[(0.0, 1.0)] * 2,
            Nx_points=[11, 10],
            spacing_type="custom",
            custom_coordinates=[np.linspace(0, 1, 11), np.linspace(0, 1, 10, endpoint=False)],
            boundary_conditions=bc,
        )
    return TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[11, 11], boundary_conditions=bc)


def _bc(periodic):
    """A segment per face -- y periodic when ``periodic`` -- and a Dirichlet segment placed by the region."""
    faces = [
        BCSegment(
            name=f"{axis}{side}",
            bc_type=BCType.PERIODIC if (periodic and axis == "y") else BCType.NO_FLUX,
            boundary=f"{axis}_{side}",
        )
        for axis in "xy"
        for side in ("min", "max")
    ]
    exit_ = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=1.0, region_name="door", priority=1)
    return BoundaryConditions(segments=[*faces, exit_], dimension=2)


def _set_before_marking():
    _grid("mixed", no_flux_bc(dimension=2)).set_boundary_conditions(_bc(periodic=True))


def _face_rule():
    _bc(periodic=False).segment_for_face("x_min", geometry=_grid("one", no_flux_bc(dimension=2)))


def _point_rule():
    _bc(periodic=False).segment_for_point(
        np.array([0.0, 0.2]), face="x_min", geometry=_grid("one", no_flux_bc(dimension=2))
    )


#: Each call that refuses on a TensorProductGrid without the region, with what its message says it cannot tell.
REFUSING_CALLS = {
    "built with the BC, axes differing in layout": (lambda: _grid("mixed", _bc(periodic=True)), "governs x_min"),
    "BC attached before marking, axes differing in layout": (_set_before_marking, "governs x_min"),
    "face rule": (_face_rule, "governs x_min"),
    "point rule": (_point_rule, "holds the boundary point [0.0, 0.2]"),
}


@pytest.mark.parametrize("call", list(REFUSING_CALLS))
def test_a_grid_without_the_region_names_the_order_that_works(call):
    refuse, what = REFUSING_CALLS[call]
    with pytest.raises(ValueError) as refused:
        refuse()
    assert str(refused.value) == HEAD.format(grid="TensorProductGrid", what=what) + ADVICE


#: The region, a boundary point in it, and the face it is on.
REGIONS = {
    "x_min whole": (lambda x: x[0] < 0.05, [0.0, 0.3], "x_min"),
    "x_min part": (lambda x: (x[0] < 0.05) & (x[1] < 0.5), [0.0, 0.2], "x_min"),
    "y_min whole": (lambda x: x[1] < 0.05, [0.3, 0.0], "y_min"),
    "y_min part": (lambda x: (x[1] < 0.05) & (x[0] < 0.5), [0.2, 0.0], "y_min"),
    "corner only": (lambda x: (x[0] < 0.05) & (x[1] < 0.05), [0.0, 0.0], "x_min"),
}

#: The other refusals a cell can meet, by a phrase of their messages.
OTHER_REFUSALS = {
    "governs no face": "govern no face",
    "covers part of": "covers part of",
    "covers no node": "cover no node of any face's interior",
}

#: After the route, per (layout, periodic, region): (attaching, then the face rule and the point rule on the BC read
#: back). On a grid whose axes differ in layout, a periodic BC is checked face by face when attached (`periodic_faces`).
ROUTE = {
    **{
        (layout, periodic, region): ("attached", face, "reads exit")
        for layout, periodic in (("one", False), ("one", True), ("mixed", False))
        for region, face in (
            ("x_min whole", "governs no face"),
            ("x_min part", "covers part of"),
            ("y_min whole", "governs no face"),
            ("y_min part", "covers part of"),
            ("corner only", "covers no node"),
        )
    },
    ("mixed", True, "x_min whole"): ("attached", "governs no face", "reads exit"),
    ("mixed", True, "x_min part"): ("attached", "covers part of", "reads exit"),
    ("mixed", True, "y_min whole"): ("governs no face", None, None),
    ("mixed", True, "y_min part"): ("covers part of", None, None),
    ("mixed", True, "corner only"): ("covers no node", None, None),
}


def _outcome(call):
    try:
        return f"reads {call().name}"
    except ValueError as error:
        message = str(error)
        assert REFUSAL not in message, message
        (named,) = [name for name, phrase in OTHER_REFUSALS.items() if phrase in message]
        return named


@pytest.mark.parametrize(("layout", "periodic", "region"), list(ROUTE))
def test_the_route_run_as_written_ends_this_refusal(layout, periodic, region):
    predicate, point, face = REGIONS[region]
    grid = _grid(layout, no_flux_bc(dimension=2))
    grid.mark_region("door", predicate=predicate)
    attach, face_rule, point_rule = ROUTE[(layout, periodic, region)]
    if attach != "attached":
        assert _outcome(lambda: grid.set_boundary_conditions(_bc(periodic))) == attach
        return
    grid.set_boundary_conditions(_bc(periodic))
    bc = grid.get_boundary_conditions()
    assert _outcome(lambda: bc.segment_for_face(face)) == face_rule
    assert _outcome(lambda: bc.segment_for_point(np.array(point), face=face)) == point_rule


@pytest.mark.parametrize(
    "geometry",
    [None, Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]])), GridNetwork(4, 4)],
    ids=["no geometry", "Hyperrectangle", "GridNetwork"],
)
def test_every_other_geometry_keeps_the_advice_it_had(geometry):
    bc = BoundaryConditions(
        segments=[BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=1.0, region_name="door", priority=1)],
        default_bc=BCType.NO_FLUX,
        dimension=2,
        domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
    )
    with pytest.raises(ValueError) as refused:
        bc.segment_for_face("x_min", geometry=geometry)
    assert str(refused.value).endswith(BASE_ADVICE)


def _region_bc(dimension):
    return BoundaryConditions(
        segments=[BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=1.0, region_name="door", priority=1)],
        default_bc=BCType.NO_FLUX,
        dimension=dimension,
    )


class _SubGrid(TensorProductGrid):
    """A user's TensorProductGrid subclass."""


def _marks_another_region():
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[11, 11], boundary_conditions=no_flux_bc(dimension=2))
    grid.mark_region("outlet", predicate=lambda x: x[0] > 0.95)
    return grid


#: Grids of other dimensions and kinds, without the region: (the grid, its type's name in the message).
OTHER_GRIDS = {
    "1-D": (
        lambda: TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1)),
        1,
    ),
    "3-D": (
        lambda: TensorProductGrid(
            bounds=[(0.0, 1.0)] * 3, Nx_points=[5] * 3, boundary_conditions=no_flux_bc(dimension=3)
        ),
        3,
    ),
    "marks another region": (_marks_another_region, 2),
    "a TensorProductGrid subclass": (
        lambda: _SubGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[11, 11], boundary_conditions=no_flux_bc(dimension=2)),
        2,
    ),
}


@pytest.mark.parametrize("case", list(OTHER_GRIDS))
def test_every_tensor_product_grid_without_the_region_names_the_order(case):
    make, dimension = OTHER_GRIDS[case]
    grid = make()
    with pytest.raises(ValueError) as refused:
        _region_bc(dimension).segment_for_face("x_min", geometry=grid)
    assert str(refused.value) == HEAD.format(grid=type(grid).__name__, what="governs x_min") + ADVICE


def _plain_cartesian_grid():
    """A CartesianGrid that is not a TensorProductGrid, as a user's own subclass may be: every abstract member answered
    by an 11 x 11 TensorProductGrid."""
    inner = TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[11, 11], boundary_conditions=no_flux_bc(dimension=2))
    members = {}
    for name in set(CartesianGrid.__abstractmethods__) | {"coordinates", "get_grid_shape", "bounds", "Nx_points"}:
        if isinstance(getattr(type(inner), name, None), property):
            members[name] = property(lambda self, name=name: getattr(inner, name))
        else:
            members[name] = lambda self, *a, name=name, **k: getattr(inner, name)(*a, **k)
    return type("PlainGrid", (CartesianGrid,), members)()


def test_a_cartesian_grid_that_is_no_tensor_product_grid_keeps_the_advice_it_had():
    with pytest.raises(ValueError) as refused:
        _region_bc(2).segment_for_face("x_min", geometry=_plain_cartesian_grid())
    assert "PlainGrid defines no region of that name" in str(refused.value)
    assert str(refused.value).endswith(BASE_ADVICE)

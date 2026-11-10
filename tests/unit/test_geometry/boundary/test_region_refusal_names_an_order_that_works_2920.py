"""A refusal of a ``region_name`` segment names placement routes that are read as it says (#2920).

`_placement_advice` owns the routes, one text for each kind of geometry a refusal names: a grid that marks regions, a
grid that marks none, a geometry that is no grid, and no geometry. `_region_name_refusal` (the face rule and the point rule) and
`_region_node`'s refusal read it. For each text, one test reads the refusal, and one runs the routes it names, through
the readers it says read them: a point reader (`segment_for_point`) and a face-level reader (`segment_for_face`).
"""

import pytest

import numpy as np

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.base import CartesianGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, mixed_bc_from_regions, no_flux_bc
from mfgarchon.geometry.boundary.types import PeriodicGridConvention
from mfgarchon.geometry.graph.network_geometry import GridNetwork
from mfgarchon.geometry.implicit.hyperrectangle import Hyperrectangle
from mfgarchon.geometry.implicit.hypersphere import Hypersphere

BOX = np.array([[0.0, 1.0], [0.0, 1.0]])
WHOLE = {"predicate": lambda x: x[0] < 0.05, "region": {0: (0.0, 0.0), 1: (0.0, 1.0)}}
PART = {"predicate": lambda x: (x[0] < 0.05) & (x[1] < 0.5), "region": {0: (0.0, 0.0), 1: (0.0, 0.5)}}
IN, OUT = np.array([0.0, 0.2]), np.array([0.0, 0.8])

GRID_ADVICE = (
    "Mark the region on the grid first, mark_region('inlet', ...), then attach this BC with set_boundary_conditions"
)
OFF_GRID_ADVICE = "Off a grid, place the segment by sdf_region=..., a signed distance read at points"
NO_GEOMETRY_ADVICE = "Build the BC with mixed_bc_from_regions(grid, ...) on the grid that marks the region"
PLAIN_GRID_ADVICE = (
    "On a grid that marks no regions, place the segment by sdf_region=... (a signed distance) or region=..."
)


def _inlet(**placement):
    return BCSegment(name="inlet", bc_type=BCType.DIRICHLET, value=1.0, priority=1, **placement)


def _bc(segment, **kw):
    return BoundaryConditions(segments=[segment], default_bc=BCType.NO_FLUX, dimension=2, domain_bounds=BOX, **kw)


def _cylinder_with_inlet():
    """The issue's BC: y periodic, x no-flux, and a no-flux segment placed by an unmarked region."""
    faces = [
        BCSegment(name=f"{axis}{side}", bc_type=kind, boundary=f"{axis}_{side}")
        for axis, kind in (("x", BCType.NO_FLUX), ("y", BCType.PERIODIC))
        for side in ("min", "max")
    ]
    return BoundaryConditions(
        segments=[*faces, BCSegment(name="inlet", bc_type=BCType.NO_FLUX, region_name="inlet", priority=1)],
        dimension=2,
    )


def _mixed(bc):
    """The issue's grid: axis 0 endpoint-inclusive (11 nodes), axis 1 endpoint-exclusive (10 nodes)."""
    return TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[11, 10],
        spacing_type="custom",
        custom_coordinates=[np.linspace(0, 1, 11), np.linspace(0, 1, 10, endpoint=False)],
        boundary_conditions=bc,
    )


def _grid():
    return TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[11, 11], boundary_conditions=no_flux_bc(dimension=2))


def _set_before_marking():
    _mixed(no_flux_bc(dimension=2)).set_boundary_conditions(_cylinder_with_inlet())


#: Each route by which a TensorProductGrid without the region is refused.
GRID_REFUSALS = {
    "built with the BC, axes differing in layout": lambda: _mixed(_cylinder_with_inlet()),
    "BC attached before marking": _set_before_marking,
    "face rule": lambda: _bc(_inlet(region_name="inlet")).segment_for_face("x_min", geometry=_grid()),
    "point rule": lambda: _bc(_inlet(region_name="inlet")).segment_for_point(IN, face="x_min", geometry=_grid()),
}


@pytest.mark.parametrize("route", list(GRID_REFUSALS))
def test_a_grid_without_the_region_advises_marking_it_then_attaching_the_bc(route):
    with pytest.raises(ValueError, match="TensorProductGrid defines no region of that name") as refused:
        GRID_REFUSALS[route]()
    message = str(refused.value)
    assert GRID_ADVICE in message
    assert "a region that covers part of a face is read at points" in message
    assert "mixed_bc_from_regions(geometry" not in message


def test_the_grid_advice_run_as_written_is_read():
    """The issue's grid, built with a plain BC, the region marked, then the BC attached: it binds the layout the
    wrapped axis has, and the segment is read on its face. A part-face region is read at points, and the face-level
    reader refuses it, as the advice says."""
    grid = _mixed(no_flux_bc(dimension=2))
    grid.mark_region("inlet", predicate=WHOLE["predicate"])
    grid.set_boundary_conditions(_cylinder_with_inlet())
    bc = grid.get_boundary_conditions()
    assert bc.periodic_convention is PeriodicGridConvention.ENDPOINT_EXCLUSIVE
    assert bc.segment_for_face("x_min").name == "inlet"

    part = _grid()
    part.mark_region("inlet", predicate=PART["predicate"])
    part.set_boundary_conditions(_bc(_inlet(region_name="inlet")))
    bc = part.get_boundary_conditions()
    assert (bc.segment_for_point(IN, face="x_min").name, bc.segment_for_point(OUT, face="x_min").name) == (
        "inlet",
        "default",
    )
    with pytest.raises(ValueError, match="covers part of the x_min face"):
        bc.segment_for_face("x_min")
    assert _bc(_inlet(boundary="x_min")).segment_for_face("x_min", geometry=_grid()).name == "inlet"


#: Each refusal of a geometry that is no grid: unmarked, marked on a geometry the owner does not read, and
#: `_region_node`'s refusal on a geometry that marks regions and is no CartesianGrid.
def _marked(geometry):
    geometry.mark_region("inlet", predicate=WHOLE["predicate"])
    return geometry


OFF_GRID_REFUSALS = {
    "Hyperrectangle, unmarked": (lambda: Hyperrectangle(BOX), "Hyperrectangle defines no region of that name"),
    "Hyperrectangle, marked": (lambda: _marked(Hyperrectangle(BOX)), "Hyperrectangle marks a region of that name"),
    "GridNetwork, marked (_region_node)": (lambda: _marked(GridNetwork(4, 4)), "GridNetwork is not a CartesianGrid"),
}


@pytest.mark.parametrize("case", list(OFF_GRID_REFUSALS))
def test_a_geometry_that_is_no_grid_advises_routes_read_off_a_grid(case):
    make, opening = OFF_GRID_REFUSALS[case]
    with pytest.raises(ValueError, match=opening) as refused:
        _bc(_inlet(region_name="inlet")).segment_for_point(IN, face="x_min", geometry=make())
    message = str(refused.value)
    assert OFF_GRID_ADVICE in message
    assert "a region that is a whole face can be named with BCSegment(boundary=...)" in message
    assert "mixed_bc_from_regions" not in message
    assert "Read the region on a TensorProductGrid" not in message


def test_the_off_grid_advice_run_as_written_is_read():
    """On a box domain, a GridNetwork and a domain without faces, each route the advice names is read by the readers
    it names, and the face-level reader refuses sdf_region, as it says."""
    for geometry in (Hyperrectangle(BOX), GridNetwork(4, 4)):
        part = _bc(_inlet(region=PART["region"]))
        assert (
            part.segment_for_point(IN, face="x_min", geometry=geometry).name,
            part.segment_for_point(OUT, face="x_min", geometry=geometry).name,
        ) == ("inlet", "default")
        assert _bc(_inlet(region=WHOLE["region"])).segment_for_face("x_min", geometry=geometry).name == "inlet"
        assert _bc(_inlet(boundary="x_min")).segment_for_face("x_min", geometry=geometry).name == "inlet"
        sdf = _bc(_inlet(sdf_region=lambda x: np.maximum(np.asarray(x)[..., 0] - 0.05, np.asarray(x)[..., 1] - 0.5)))
        assert (
            sdf.segment_for_point(IN, face="x_min", geometry=geometry).name,
            sdf.segment_for_point(OUT, face="x_min", geometry=geometry).name,
        ) == ("inlet", "default")
        with pytest.raises(ValueError, match="placed by sdf_region"):
            sdf.segment_for_face("x_min", geometry=geometry)

    # Without faces, boundary="x_min" holds the boundary points the SDF assigns to x_min, by its gradient's
    # dominant axis: on a disk, a point whose outward normal points mostly along -x, and not one along +y.
    disk = Hypersphere(center=[0.5, 0.5], radius=0.5)
    bc = BoundaryConditions(
        segments=[_inlet(boundary="x_min")], default_bc=BCType.NO_FLUX, dimension=2, domain_sdf=disk.signed_distance
    )
    west = np.array([0.5 - 0.5 * np.cos(np.pi / 6), 0.5 + 0.5 * np.sin(np.pi / 6)])
    assert (
        bc.segment_for_point(west, geometry=disk).name,
        bc.segment_for_point(np.array([0.5, 1.0]), geometry=disk).name,
    ) == (
        "inlet",
        "default",
    )


@pytest.mark.parametrize("rule", ["face", "point"])
def test_a_reader_with_no_geometry_advises_routes_read_without_one(rule):
    bc = _bc(_inlet(region_name="inlet"))
    with pytest.raises(ValueError, match="This reader has no geometry") as refused:
        bc.segment_for_face("x_min") if rule == "face" else bc.segment_for_point(IN, face="x_min")
    message = str(refused.value)
    assert NO_GEOMETRY_ADVICE in message
    assert "which are read at points without one" in message
    assert "mixed_bc_from_regions(geometry" not in message


def test_the_no_geometry_advice_run_as_written_is_read():
    """A whole-face region becomes its face, which a reader without a geometry reads; a part-face region stays a
    region_name, read by a reader given the grid; region= and sdf_region= are read at points without one."""
    wall = BCSegment(name="wall", bc_type=BCType.NO_FLUX)
    grid = _grid()
    grid.mark_region("inlet", predicate=WHOLE["predicate"])
    whole = mixed_bc_from_regions(grid, {"inlet": _inlet(), "default": wall}, dimension=2)
    assert whole.segment_for_face("x_min").name == "inlet"

    grid = _grid()
    grid.mark_region("inlet", predicate=PART["predicate"])
    part = mixed_bc_from_regions(grid, {"inlet": _inlet(), "default": wall}, dimension=2)
    part = BoundaryConditions(segments=part.segments, default_bc=BCType.NO_FLUX, dimension=2, domain_bounds=BOX)
    assert [s.region_name for s in part.segments if s.name == "inlet"] == ["inlet"]
    assert part.segment_for_point(IN, face="x_min", geometry=grid).name == "inlet"

    for placed in (_inlet(region=PART["region"]), _inlet(sdf_region=lambda x: np.asarray(x)[..., 1] - 0.5)):
        bc = _bc(placed)
        assert (bc.segment_for_point(IN, face="x_min").name, bc.segment_for_point(OUT, face="x_min").name) == (
            "inlet",
            "default",
        )
    assert _bc(_inlet(boundary="x_min")).segment_for_face("x_min").name == "inlet"


def _plain_grid():
    """A CartesianGrid that marks no regions, as a user's own subclass may be: every abstract member answered by an
    11 x 11 TensorProductGrid, and no `mark_region`."""
    inner = _grid()
    members = {}
    for name in set(CartesianGrid.__abstractmethods__) | {"coordinates", "get_grid_shape", "bounds", "Nx_points"}:
        if isinstance(getattr(type(inner), name, None), property):
            members[name] = property(lambda self, name=name: getattr(inner, name))
        else:
            members[name] = lambda self, *a, name=name, **k: getattr(inner, name)(*a, **k)
    return type("PlainGrid", (CartesianGrid,), members)()


@pytest.mark.parametrize("rule", ["face", "point"])
def test_a_grid_that_marks_no_regions_advises_routes_read_on_a_grid(rule):
    grid = _plain_grid()
    assert not hasattr(grid, "mark_region")
    bc = _bc(_inlet(region_name="inlet"))
    with pytest.raises(ValueError, match="PlainGrid defines no region of that name") as refused:
        bc.segment_for_face("x_min", geometry=grid) if rule == "face" else bc.segment_for_point(
            IN, face="x_min", geometry=grid
        )
    message = str(refused.value)
    assert PLAIN_GRID_ADVICE in message
    assert "mark_region" not in message
    assert "mixed_bc_from_regions" not in message


def test_the_plain_grid_advice_run_as_written_is_read():
    """On a grid that marks no regions, sdf_region= and region= are read at points, and by the face-level reader where
    they cover a whole face; boundary= names a whole face."""
    grid = _plain_grid()
    whole_sdf = _inlet(sdf_region=lambda x: np.asarray(x)[..., 0] - 0.05)
    part_sdf = _inlet(sdf_region=lambda x: np.maximum(np.asarray(x)[..., 0] - 0.05, np.asarray(x)[..., 1] - 0.5))
    for whole, part in ((whole_sdf, part_sdf), (_inlet(region=WHOLE["region"]), _inlet(region=PART["region"]))):
        assert _bc(whole).segment_for_face("x_min", geometry=grid).name == "inlet"
        bc = _bc(part)
        assert (
            bc.segment_for_point(IN, face="x_min", geometry=grid).name,
            bc.segment_for_point(OUT, face="x_min", geometry=grid).name,
        ) == ("inlet", "default")
    assert _bc(_inlet(boundary="x_min")).segment_for_face("x_min", geometry=grid).name == "inlet"

"""A region-named BC segment governs the faces its region covers, and no others (#2472).

Measured on main at `1c84a29c`, with the documented idiom (`mark_region(name, boundary=...)` +
`mixed_bc_from_regions`) on a 9 x 9 grid and u = 1:

- one region, an outlet on `x_max`: `is_uniform` ignored `region_name`, so the outlet's Dirichlet ghost
  was written on all four faces;
- two regions, outlet on `x_max` and inlet on `x_min`: `get_bc_type_at_boundary` answered DIRICHLET on
  every face (a segment with no `boundary` covered them all), while the ghosts were the mirror on every
  face. The ghost path's region lookup indexed the grid's flat region mask with a 2-D face index, raised
  IndexError every time, and swallowed it at debug level, so the outlet reached no face.

Also pinned: a region covering part of a face is refused, since one condition per face cannot represent
it; a reader with no geometry refuses a region name it cannot resolve; and a uniform BC's unused
`default_bc` no longer makes `geometric_operations` report it as mixed (#2472 item 2).
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc, pad_array_with_ghosts
from mfgarchon.geometry.boundary.bc_utils import geometric_operations
from mfgarchon.geometry.boundary.conditions import mixed_bc_from_regions

N = 9
_SEGMENTS = {
    "outlet": BCSegment(name="outlet", bc_type=BCType.DIRICHLET, value=0.0),
    "inlet": BCSegment(name="inlet", bc_type=BCType.NO_FLUX),
    "default": BCSegment(name="wall", bc_type=BCType.NO_FLUX),
}


def _grid():
    return TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[N, N], boundary_conditions=no_flux_bc(dimension=2))


def _ghosts(regions):
    grid = _grid()
    for name, spec in regions.items():
        grid.mark_region(name, **spec)
    bc = mixed_bc_from_regions(grid, {name: _SEGMENTS[name] for name in [*regions, "default"]})
    padded = pad_array_with_ghosts(np.ones((N, N)), bc, geometry=grid)
    return bc, {
        "x_min": padded[0, 1:-1],
        "x_max": padded[-1, 1:-1],
        "y_min": padded[1:-1, 0],
        "y_max": padded[1:-1, -1],
    }


@pytest.mark.parametrize(
    "regions",
    [{"outlet": {"boundary": "x_max"}}, {"outlet": {"boundary": "x_max"}, "inlet": {"boundary": "x_min"}}],
    ids=["one_region", "two_regions"],
)
def test_a_region_named_outlet_reaches_its_face_and_only_it(regions):
    _, ghosts = _ghosts(regions)
    # Dirichlet g = 0 on u = 1 gives the ghost -1; zero flux mirrors it to +1.
    np.testing.assert_array_equal(ghosts["x_max"], -1.0)
    for face in ("x_min", "y_min", "y_max"):
        np.testing.assert_array_equal(ghosts[face], 1.0, err_msg=f"{face} took the outlet's condition")


def test_a_region_covering_part_of_a_face_is_refused():
    with pytest.raises(ValueError, match="covers part of the"):
        _ghosts({"outlet": {"predicate": lambda x: x[:, 0] > 0.6}})


def test_a_reader_without_the_geometry_refuses_a_region_it_cannot_resolve():
    grid = _grid()
    grid.mark_region("outlet", boundary="x_max")
    bc = mixed_bc_from_regions(grid, {"outlet": _SEGMENTS["outlet"], "default": _SEGMENTS["default"]})
    assert not bc.is_uniform
    with pytest.raises(ValueError, match="known only to the geometry"):
        bc.get_bc_type_at_boundary("y_min")

    # A region name that is a face label names that face, with or without a geometry.
    labelled = BoundaryConditions(
        segments=[BCSegment(name="outlet", bc_type=BCType.DIRICHLET, region_name="x_max")],
        dimension=2,
        default_bc=BCType.NO_FLUX,
        domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
    )
    assert labelled.get_bc_type_at_boundary("x_max") is BCType.DIRICHLET
    assert labelled.get_bc_type_at_boundary("y_min") is BCType.NO_FLUX


@pytest.mark.parametrize(
    ("boundary", "expected"),
    [(None, {"periodic"}), ("x_min", {"periodic", "reflect"})],
    ids=["uniform_default_unread", "mixed_default_read"],
)
def test_only_a_default_some_face_reads_counts_as_an_operation(boundary, expected):
    bc = BoundaryConditions(
        segments=[BCSegment(name="p", bc_type=BCType.PERIODIC, boundary=boundary)],
        dimension=1,
        default_bc=BCType.NO_FLUX,
        domain_bounds=np.array([[0.0, 1.0]]),
    )
    assert geometric_operations(bc) == expected

"""#2490 B1e-1: a ``region_name`` segment is read at a point in the order the face rule reads it -- the marking
geometry's mask, then a face label, then a refusal -- and a point the mask cannot be read at is refused, not read as
outside the region."""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc


@pytest.mark.parametrize(
    "row", ["a face label, read by both readers", "a point off the marking grid", "a point between the nodes"]
)
def test_a_region_named_segment_is_read_at_a_point_as_on_its_face(row):
    """- ``region_name="x_max"`` with no geometry names the face x_max: the face reader gives it x_max, and the point
      reader required a geometry and refused it at a point of that face.
    - A grid that marks "strip" (y > 0.4), read through a BC whose box reaches x = 2: the point (2, 0.5) lies on its
      x_max, a whole grid width off the grid. The point rule read the mask at the node it clamped the point to,
      (1, 0.5), so the strip governed a point the grid never marked. It is refused, and the message's remedy -- the
      BC's box inside the grid's -- reads the strip at (1, 0.5).
    - Between the nodes the mask is read at the nearest node: (1, 0.45) lies 0.05 from the marked node y = 0.5 and 0.2
      from the unmarked y = 0.25, and reads the strip. The point rule truncated the index to the node below, 0.25."""
    if row == "a face label, read by both readers":
        bc = BoundaryConditions(
            segments=[BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, region_name="x_max")],
            dimension=2,
            default_bc=BCType.NO_FLUX,
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )
        assert bc.segment_for_face("x_max").name == "exit"
        assert bc.segment_for_point(np.array([1.0, 0.5])).name == "exit"
        assert bc.segment_for_point(np.array([0.0, 0.5])).name == "default"
        return
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[5, 5], boundary_conditions=no_flux_bc(dimension=2)
    )
    grid.mark_region("strip", predicate=lambda x: x[:, 1] > 0.4)
    segments = [
        BCSegment(name="strip", bc_type=BCType.DIRICHLET, value=0.0, region_name="strip"),
        BCSegment(name="rest", bc_type=BCType.NO_FLUX, value=0.0),
    ]
    inside = BoundaryConditions(segments=segments, dimension=2, domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]))
    if row == "a point between the nodes":
        assert inside.segment_for_point(np.array([1.0, 0.45]), geometry=grid).name == "strip"
        assert inside.segment_for_point(np.array([1.0, 0.3]), geometry=grid).name == "rest"
        return
    wide = BoundaryConditions(segments=segments, dimension=2, domain_bounds=np.array([[0.0, 2.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match=r"lies off the grid that marks it.*domain_bounds inside that grid's box"):
        wide.segment_for_point(np.array([2.0, 0.5]), geometry=grid)
    assert inside.segment_for_point(np.array([1.0, 0.5]), geometry=grid).name == "strip"

"""The point owner reads its faces from the geometry, and resolves a corner at the point (#2512 row B1, ahead of B1c-1).

Two stage jumps from B1d, made for GFDM, whose boundary points the point owner is about to classify (B1c-1):

1. **The faces come from the geometry when the BC carries no bounds.** None of the 53 mixed BCs GFDM classifies in
   the suite carries `domain_bounds`, so `segment_for_point` refused every point. GFDM had passed its geometry's
   bounds to the point readers itself.
2. **A point at a corner is resolved at the point** (user ruling, 2026-10-14). Each face gives its own segment where
   it answers; where it refuses because a segment covers only part of it, the segment holding the point on that
   face stands in; any other refusal stands; the corner order then chooses. The node's answer, which the point took
   before, refused GFDM's corners on a face split by ``region`` into strips with no ``default_bc``.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
from mfgarchon.geometry.boundary.tolerances import BOUNDARY_TOL
from mfgarchon.geometry.boundary.types import BoundaryFace
from mfgarchon.geometry.implicit import Hyperrectangle

UNIT_SQUARE = np.array([[0.0, 1.0], [0.0, 1.0]])


def _split_x_max(**kwargs) -> BoundaryConditions:
    """`test_hjb_gfdm_bc_newton_residual`'s BC: no-flux walls, x_max split by ``region`` into two no-flux strips and a
    Dirichlet exit between them, no ``default_bc`` -- so no segment governs x_max whole -- and no ``domain_bounds``."""
    return BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="left", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="bottom", bc_type=BCType.NO_FLUX, boundary="y_min"),
            BCSegment(name="top", bc_type=BCType.NO_FLUX, boundary="y_max"),
            BCSegment(name="right_below", bc_type=BCType.NO_FLUX, boundary="x_max", region={"y": (0.0, 0.4)}),
            BCSegment(name="right_above", bc_type=BCType.NO_FLUX, boundary="x_max", region={"y": (0.6, 1.0)}),
            BCSegment(
                name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max", region={"y": (0.4, 0.6)}, priority=1
            ),
        ],
        **kwargs,
    )


@pytest.mark.parametrize("geometry", [Hyperrectangle(UNIT_SQUARE), "grid"], ids=["Hyperrectangle", "TensorProductGrid"])
def test_a_bc_without_bounds_reads_its_faces_from_the_geometry(geometry):
    """Gap 1. The answer equals the one a BC carrying the same bounds gives (the control)."""
    if geometry == "grid":
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[11, 11],
            boundary_conditions=_split_x_max(domain_bounds=UNIT_SQUARE),
        )
    unbounded, bounded = _split_x_max(), _split_x_max(domain_bounds=UNIT_SQUARE)
    for point, name in (
        ([0.0, 0.5], "left"),
        ([0.5, 0.0], "bottom"),
        ([1.0, 0.5], "exit"),
        ([1.0, 0.2], "right_below"),
    ):
        assert unbounded.segment_for_point(np.array(point), geometry=geometry).name == name
        assert bounded.segment_for_point(np.array(point), geometry=geometry).name == name


def test_a_bc_without_bounds_and_a_query_without_geometry_is_refused_by_name():
    with pytest.raises(ValueError, match=r"no domain_bounds or domain_sdf, and the query has no geometry"):
        _split_x_max().segment_for_point(np.array([0.0, 0.5]))


def test_a_corner_of_a_face_split_into_strips_is_answered_at_the_point():
    """Gap 3. No segment governs x_max whole and there is no default, so the node refuses both x_max corners; the
    point reads x_max at the point (the strip holding it) and y's face whole, and the corner order chooses: equal
    priority, both natural, both placed, so declaration order -- bottom and top, of the same condition as the strips
    GFDM chose before (no-flux 0)."""
    bc = _split_x_max()
    geometry = Hyperrectangle(UNIT_SQUARE)
    with pytest.raises(ValueError, match="no segment governs the x_max face"):
        _split_x_max(domain_bounds=UNIT_SQUARE).segment_for_corner([BoundaryFace(0, "max"), BoundaryFace(1, "min")])
    for corner, name in (([1.0, 0.0], "bottom"), ([1.0, 1.0], "top")):
        assert bc.segment_for_point(np.array(corner), geometry=geometry, tolerance=BOUNDARY_TOL).name == name
    # GFDM's collocation puts boundary points 1e-7 off the wall: its tolerance reads them on both faces.
    near = np.array([1.0 - 1e-7, 1e-7])
    assert bc.segment_for_point(near, geometry=geometry, tolerance=BOUNDARY_TOL).name == "bottom"

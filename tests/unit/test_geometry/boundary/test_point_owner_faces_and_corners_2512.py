"""The point owner reads its faces from the geometry, and resolves a corner at the point (#2512 row B1, ahead of B1c-1).

Two stage jumps from B1d, made for GFDM, whose boundary points the point owner is about to classify (B1c-1):

1. **The faces come from the geometry when the BC carries neither bounds nor an SDF.** None of the BCs GFDM
   classifies in the suite's CI tier carries `domain_bounds`, so `segment_for_point` refused every one of its points
   (#2663's census at 5e1c47d4). GFDM had passed its geometry's bounds to the point readers itself.
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
    with pytest.raises(
        ValueError, match=r"no domain_bounds or domain_sdf, and the query's geometry \(NoneType\) gives no box"
    ):
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


def test_a_whole_bc_refusal_still_refuses_a_corner_where_every_face_is_partly_covered():
    """Review 1 of #2663, B1: only a partial-cover refusal is read at the point. ``patch`` covers the corner box of
    both x_max and y_max and governs no face, so the BC is refused as a whole -- at the node, and at the point,
    although each face there is partly covered."""
    bc = BoundaryConditions(
        dimension=2,
        domain_bounds=UNIT_SQUARE,
        default_bc=BCType.NO_FLUX,
        segments=[
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_max", priority=1),
            BCSegment(name="top", bc_type=BCType.DIRICHLET, value=2.0, boundary="y_max", priority=1),
            BCSegment(name="patch", bc_type=BCType.NEUMANN, value=0.5, region={"x": (0.5, 1.0), "y": (0.5, 1.0)}),
            BCSegment(name="walls", bc_type=BCType.NO_FLUX),
        ],
    )
    grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[7, 7], boundary_conditions=bc)
    bc = grid.get_boundary_conditions()
    with pytest.raises(ValueError, match="govern no face of the domain"):
        bc.segment_for_corner([BoundaryFace(0, "max"), BoundaryFace(1, "max")], grid)
    with pytest.raises(ValueError, match="govern no face of the domain"):
        bc.segment_for_point(np.array([1.0, 1.0]), geometry=grid)


def test_a_bc_with_its_own_sdf_is_not_overridden_by_the_geometrys_box():
    """Review 1 of #2663, B2: gap 1 takes the box from the geometry only for a BC with neither ``domain_bounds`` nor
    ``domain_sdf``. On a circle's boundary near its rightmost point, the circle's own SDF reads x_max; the bounding
    box of the geometry would read no box face there."""
    from mfgarchon.geometry.implicit import Hypersphere

    bc = BoundaryConditions(
        dimension=2,
        domain_sdf=lambda x: np.linalg.norm(np.asarray(x, dtype=float) - 0.5, axis=-1) - 0.4,
        segments=[
            BCSegment(name="east", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_max", priority=1),
            BCSegment(name="rest", bc_type=BCType.NO_FLUX),
        ],
    )
    point = np.array([0.5 + 0.4 * np.cos(0.2), 0.5 + 0.4 * np.sin(0.2)])
    assert bc.segment_for_point(point).name == "east"
    assert bc.segment_for_point(point, geometry=Hypersphere(np.array([0.5, 0.5]), 0.4)).name == "east"


def _strip_beside_a_dirichlet():
    """Review 1 of #2663's C4: B-c with the wall spelled NO_FLUX. ``strip`` NEUMANN(0) covers x_max for y >= 0.75 at
    priority 2; ``D`` DIRICHLET(5) holds y_max at priority 1. Whether the strip imposes the wall's condition is the
    consumer's question: the interpolation applicator enforces NEUMANN(0) and NO_FLUX alike, the default does not."""
    bc = BoundaryConditions(
        dimension=2,
        domain_bounds=UNIT_SQUARE,
        segments=[
            BCSegment(name="wall", bc_type=BCType.NO_FLUX),
            BCSegment(
                name="strip", bc_type=BCType.NEUMANN, value=0.0, boundary="x_max", region={"y": (0.75, 1.0)}, priority=2
            ),
            BCSegment(name="D", bc_type=BCType.DIRICHLET, value=5.0, boundary="y_max", priority=1),
        ],
    )
    grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[7, 7], boundary_conditions=bc)
    return grid.get_boundary_conditions(), grid


def test_the_point_takes_the_nodes_answer_under_the_consumers_equivalence():
    """Review 1 of #2663, B3 (the audit session's pick): `segment_for_point` takes the consumer's ``equivalent``, as the
    face and corner owners do. Under the applicator's, the strip imposes the wall's condition, x_max answers the wall,
    and the corner is D's -- at the node and at the point. Under the default, x_max is split, the node refuses, and the
    point reads the strip that holds it."""
    from mfgarchon.geometry.boundary.applicator_interpolation import _same_enforcement

    bc, grid = _strip_beside_a_dirichlet()
    corner, faces = np.array([1.0, 1.0]), [BoundaryFace(0, "max"), BoundaryFace(1, "max")]
    assert bc.segment_for_corner(faces, grid, equivalent=_same_enforcement).name == "D"
    assert bc.segment_for_point(corner, geometry=grid, equivalent=_same_enforcement).name == "D"
    with pytest.raises(ValueError, match="covers part of the x_max face"):
        bc.segment_for_corner(faces, grid)
    assert bc.segment_for_point(corner, geometry=grid).name == "strip"


def test_a_blending_corner_is_judged_by_the_consumers_equivalence():
    """Under ``corner_strategy="average"`` a corner is refused where its faces' segments impose different conditions,
    judged by the equivalence the point was asked with. NO_FLUX on x_max and NEUMANN(0) on y_max differ under the
    default and agree under the interpolation applicator's."""
    from mfgarchon.geometry.boundary.applicator_interpolation import _same_enforcement

    bc = BoundaryConditions(
        dimension=2,
        domain_bounds=UNIT_SQUARE,
        corner_strategy="average",
        default_bc=BCType.NO_FLUX,
        segments=[BCSegment(name="n0", bc_type=BCType.NEUMANN, value=0.0, boundary="y_max")],
    )
    corner = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="impose different conditions"):
        bc.segment_for_point(corner)
    assert bc.segment_for_point(corner, equivalent=_same_enforcement).name == "n0"  # priority 0 over the default's -1

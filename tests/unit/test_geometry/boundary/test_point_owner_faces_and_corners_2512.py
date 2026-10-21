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


def test_a_whole_bc_refusal_refuses_the_corner_node_and_the_point_reader_at_the_point():
    """Review 1 of #2663, B1: ``patch`` covers the corner box of both x_max and y_max and governs no face, so the BC is
    refused as a whole at the node. At the point, the face owner's whole-BC check no longer runs (#2490 B1e-2): it read
    the grid's nodes in place of the reader's points. The point answers x_max's segment, and the point family's check,
    over the points a reader reads, refuses ``patch`` by name: outranked at the corner it holds."""
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
    corner = np.array([1.0, 1.0])
    read = bc.segment_for_point(corner, geometry=grid)
    assert read.name == "exit"
    with pytest.raises(
        ValueError,
        match=r"'patch' \(priority 0\) -- outranked at \[1\.0, 1\.0\], a corner, on x_max: 'exit' \(priority 1\)",
    ):
        bc.refuse_segments_governing_no_point([(corner, None, read)], "a reader", "its one point", geometry=grid)


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


def test_a_corner_partly_covered_under_a_robin_default_is_read_at_the_point():
    """Review 2 of #2663, S1: a ROBIN default does not answer (it carries no alpha or beta), so on a face a strip covers
    only in part, the strip stands in at the point. The node refuses with the ROBIN refusal."""
    bc = BoundaryConditions(
        dimension=2,
        domain_bounds=UNIT_SQUARE,
        default_bc=BCType.ROBIN,
        segments=[BCSegment(name="strip", bc_type=BCType.NEUMANN, value=0.0, region={"y": (0.5, 1.0)})],
    )
    with pytest.raises(ValueError, match="default_bc is ROBIN"):
        bc.segment_for_corner([BoundaryFace(0, "max"), BoundaryFace(1, "max")])
    assert bc.segment_for_point(np.array([1.0, 1.0])).name == "strip"


def test_a_cover_the_owner_cannot_decide_is_refused_at_a_corner_too():
    """Review 2 of #2663, S2: an ``sdf_region`` cover is decided on a grid's nodes, so on a Hyperrectangle the face
    cannot decide it. That refusal is not a partial cover's, so it stands at the corner, although a region strip
    also covers part of the face."""
    from mfgarchon.geometry.implicit import Hyperrectangle

    bc = BoundaryConditions(
        dimension=2,
        domain_bounds=UNIT_SQUARE,
        default_bc=BCType.NO_FLUX,
        segments=[
            BCSegment(name="strip", bc_type=BCType.NEUMANN, value=0.5, boundary="x_max", region={"y": (0.5, 1.0)}),
            BCSegment(
                name="blob", bc_type=BCType.NO_FLUX, sdf_region=lambda x: np.asarray(x, dtype=float)[..., 0] - 2.0
            ),
        ],
    )
    geometry = Hyperrectangle(UNIT_SQUARE)
    with pytest.raises(ValueError, match="decided on a grid's boundary nodes"):
        bc.segment_for_corner([BoundaryFace(0, "max"), BoundaryFace(1, "max")], geometry)
    with pytest.raises(ValueError, match="decided on a grid's boundary nodes"):
        bc.segment_for_point(np.array([1.0, 1.0]), geometry=geometry)


def test_domain_bounds_decide_the_faces_before_the_domain_sdf():
    """Review 2 of #2663, S3: the precedence is ``domain_bounds``, then ``domain_sdf``, then the geometry's box. With
    both set, the corner (1, 1) lies on x_max and y_max of the box, so the corner rule gives it to ``north``, of the
    higher priority; read from the SDF alone, the point would find one face and take that face's segment."""
    bc = BoundaryConditions(
        dimension=2,
        domain_bounds=UNIT_SQUARE,
        domain_sdf=lambda x: np.max(np.abs(np.asarray(x, dtype=float) - 0.5), axis=-1) - 0.5,
        segments=[
            BCSegment(name="east", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_max", priority=1),
            BCSegment(name="north", bc_type=BCType.DIRICHLET, value=2.0, boundary="y_max", priority=2),
            BCSegment(name="rest", bc_type=BCType.NO_FLUX),
        ],
    )
    assert bc.segment_for_point(np.array([1.0, 0.5])).name == "east"
    assert bc.segment_for_point(np.array([1.0, 1.0])).name == "north"


def test_the_readings_at_a_point_are_what_the_point_chooses_among():
    """B1c-1: `face_segments_at_point` gives each face the point lies on, in face order, with its segment there -- the
    readings `segment_for_point` chooses among under the same equivalence, and GFDM's corner normal reads. At the strip
    corner, x_max is partly covered under the default and reads the strip at the point; under the applicator's
    equivalence x_max answers the wall. One face gives one reading; an interior point none."""
    from mfgarchon.geometry.boundary.applicator_interpolation import _same_enforcement

    bc, grid = _strip_beside_a_dirichlet()

    def named(point, equivalent=None):
        return [(f.to_string(), s.name) for f, s, _ in bc.face_segments_at_point(point, grid, equivalent=equivalent)]

    corner = np.array([1.0, 1.0])
    assert named(corner) == [("x_max", "strip"), ("y_max", "D")]
    assert named(corner, _same_enforcement) == [("x_max", "wall"), ("y_max", "D")]
    assert named(np.array([1.0, 0.9])) == [("x_max", "strip")]
    assert named(np.array([0.5, 0.5])) == []
    nodes = grid.get_spatial_grid()
    on_boundary = nodes[np.any((nodes == 0.0) | (nodes == 1.0), axis=1)]
    for equivalent in (None, _same_enforcement):
        for point in on_boundary:
            chosen = bc.segment_for_point(point, geometry=grid, equivalent=equivalent)
            assert any(seg is chosen for _, seg, _ in bc.face_segments_at_point(point, grid, equivalent=equivalent)), (
                point
            )


def test_a_bc_with_only_an_sdf_reads_each_points_face_from_it():
    """Review 1 of #2674, blocker 1: both point readers take their faces from one function. A BC with ``domain_sdf``
    and no ``domain_bounds`` gives each point the one face its SDF's gradient names, and the readings carry it; GFDM
    found none there and refused every point. One face per point, so no corner rule reaches an SDF BC (#2675)."""

    def box_sdf(x):
        return float(np.max(np.abs(np.asarray(x, dtype=float) - 0.5)) - 0.5)

    bc = BoundaryConditions(
        dimension=2,
        domain_sdf=box_sdf,
        segments=[
            BCSegment(name="in", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_min"),
            BCSegment(name="wall", bc_type=BCType.NO_FLUX),
        ],
    )
    left, right = np.array([0.0, 0.5]), np.array([1.0, 0.5])
    assert bc.segment_for_point(left, tolerance=BOUNDARY_TOL).name == "in"
    assert bc.segment_for_point(right, tolerance=BOUNDARY_TOL).name == "wall"
    assert [(f.to_string(), s.name) for f, s, _ in bc.face_segments_at_point(left, tolerance=BOUNDARY_TOL)] == [
        ("x_min", "in")
    ]


def test_two_robin_segments_are_compared_as_stored():
    """Review 1 of #2674, blocker 2's family: an equivalence compares two segments and writes no row, so it reads a
    Robin coefficient as stored -- a provider is equal only to itself -- and does not resolve it. The owner's
    ``_same_condition`` resolved them and raised on a provider, so ``uniform_segment`` did; so did the interpolation
    applicator's ``_same_enforcement``. With ``beta = 1.0`` on both, the control, the BC is uniform."""
    from mfgarchon.geometry.boundary.applicator_interpolation import _same_enforcement
    from mfgarchon.geometry.boundary.providers import ConstantProvider

    def robins(beta):
        return BoundaryConditions(
            dimension=2,
            segments=[
                BCSegment(name="ra", bc_type=BCType.ROBIN, alpha=0.0, beta=beta, value=0.25, boundary="x_min"),
                BCSegment(name="rb", bc_type=BCType.ROBIN, alpha=0.0, beta=1.0, value=0.25),
            ],
        )

    provided = robins(ConstantProvider(1.0))
    assert provided.uniform_segment is None
    assert _same_enforcement(provided.segments[0], provided.segments[0])
    assert not _same_enforcement(provided.segments[0], provided.segments[1])
    assert robins(1.0).uniform_segment.name == "ra"

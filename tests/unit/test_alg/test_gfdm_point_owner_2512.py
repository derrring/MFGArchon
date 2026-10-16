"""GFDM classifies its boundary points through the point owner (#2512 row B1, stage B1c-1).

`HJBGFDMSolver._preclassify_boundary_points` asks `BoundaryConditions.segment_for_point` for every boundary collocation
point, with the problem's geometry, its ``BOUNDARY_TOL`` and its own equivalence, `_same_bc_row`: two segments are the
same when GFDM writes the same row for them. Before, it walked the segments itself, in its own priority order, on the
first face `identify_boundary_face` returned.

The row at a corner is written along the normal of R1 (#2512 plan, gap 2): the first face, in face order, whose
segment at the point writes the row the point's segment writes. The pins:

1. On `test_hjb_gfdm_bc_newton_residual`'s split x_max, the owner gives the x_max corners to ``bottom`` and ``top``,
   whose condition the x_max strips share, so R1 keeps the x_max normal GFDM wrote before. The face of the chosen
   segment itself (R2) would turn those rows to y.
2. A Neumann corner whose two faces carry different data: u = a x + b y meets every face's datum, so the corner row
   is exact only along the normal of the face whose datum it carries. The first face at the point would give |a - b|.
3. GFDM's equivalence decides the corner: NEUMANN(0) on a strip and NO_FLUX on the wall write the same row, so x_max
   answers the wall and the corner is the Dirichlet's; under the owner's default they differ and the corner is the
   strip's, a Neumann row where GFDM writes a Dirichlet one.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

NX = 9


def _solver(bc: BoundaryConditions) -> HJBGFDMSolver:
    grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx=[NX, NX], boundary_conditions=bc)
    model = Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)))
    conditions = Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=1.0)
    problem = MFGProblem(model=model, domain=grid, conditions=conditions, Nt=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HJBGFDMSolver(
            problem,
            collocation_points=grid.get_spatial_grid(),
            delta=2.0 / (NX - 1),
            k_neighbors=12,
            derivative_method="taylor",
            taylor_order=2,
            weight_function="wendland",
            monotonicity_scheme="none",
        )


def _index(solver: HJBGFDMSolver, point: list[float]) -> int:
    distance = np.linalg.norm(solver.collocation_points - np.asarray(point), axis=1)
    return int(np.argmin(distance))


def test_the_split_x_max_corners_keep_their_x_normal():
    bc = BoundaryConditions(
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
    )
    solver = _solver(bc)
    for corner, name in (([1.0, 0.0], "bottom"), ([1.0, 1.0], "top")):
        i = _index(solver, corner)
        assert solver._bc_segment_per_point[i].name == name, corner
        assert solver._bc_normal_per_point[i] == pytest.approx([1.0, 0.0]), corner


def test_a_neumann_corner_row_is_exact_along_the_face_whose_datum_it_carries():
    a, b = 1.0, 3.0
    bc = BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="x_min", bc_type=BCType.NEUMANN, value=-a, boundary="x_min"),
            BCSegment(name="x_max", bc_type=BCType.NEUMANN, value=a, boundary="x_max"),
            BCSegment(name="y_min", bc_type=BCType.NEUMANN, value=-b, boundary="y_min", priority=1),
            BCSegment(name="y_max", bc_type=BCType.NEUMANN, value=b, boundary="y_max"),
        ],
    )
    solver = _solver(bc)
    u = a * solver.collocation_points[:, 0] + b * solver.collocation_points[:, 1]
    i = _index(solver, [0.0, 0.0])
    assert solver._bc_segment_per_point[i].name == "y_min"  # priority 1 takes the corner
    bc_enum, segment, normal = solver._classify_boundary_point(i, 0, True, None, None)
    row, target = solver._bc_row_for_point(i, bc_enum, segment, normal, 2, solver.n_points, 0.0)
    assert abs(row @ u - target) < 1e-8 * abs(a - b)


def test_gfdm_judges_a_corner_by_the_row_it_writes():
    from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import _same_bc_row

    bc = BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="wall", bc_type=BCType.NO_FLUX),
            BCSegment(
                name="strip", bc_type=BCType.NEUMANN, value=0.0, boundary="x_max", region={"y": (0.75, 1.0)}, priority=2
            ),
            BCSegment(name="D", bc_type=BCType.DIRICHLET, value=5.0, boundary="y_max", priority=1),
        ],
    )
    solver = _solver(bc)
    i = _index(solver, [1.0, 1.0])
    assert solver._bc_segment_per_point[i].name == "D"
    assert solver._classify_boundary_point(i, 0, True, None, None)[0] is BCType.DIRICHLET
    grid = solver.problem.geometry
    corner = solver.collocation_points[i]
    assert bc.segment_for_point(corner, geometry=grid, equivalent=_same_bc_row).name == "D"
    assert bc.segment_for_point(corner, geometry=grid).name == "strip"  # the owner's default, for contrast


def _box_sdf(x):
    return float(np.max(np.abs(np.asarray(x, dtype=float) - 0.5)) - 0.5)


def _obstacle_sdf(x):
    return float(0.2 - np.linalg.norm(np.asarray(x, dtype=float) - 0.5))


def _inlet_and_wall(**kwargs) -> BoundaryConditions:
    segments = [
        BCSegment(name="in", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_min"),
        BCSegment(name="wall", bc_type=BCType.NO_FLUX),
    ]
    return BoundaryConditions(dimension=2, segments=segments, **kwargs)


def test_a_bc_with_its_own_sdf_is_classified_by_the_faces_its_sdf_gives():
    """Review 1 of #2674, blocker 1. A BC with ``domain_sdf`` and no ``domain_bounds`` has no box for the owner to read
    faces from (#2663: the geometry's box does not override the BC's SDF), so each point takes the face its SDF gives,
    as `segment_for_point` reads it. At the base GFDM read its box from the geometry whatever the BC carried, so a
    box SDF was classified by the geometry's box. An SDF that is an obstacle's puts the outer walls on no face of the
    BC; the refusal says so and names ``domain_bounds``, and with them the same BC classifies."""
    solver = _solver(_inlet_and_wall(domain_sdf=_box_sdf))
    assert solver._bc_segment_per_point[_index(solver, [0.0, 0.5])].name == "in"
    assert solver._bc_segment_per_point[_index(solver, [1.0, 0.5])].name == "wall"
    with pytest.raises(ValueError, match=r"pt \d+ at \[[^\]]*\]: on no face the boundary owner reads for this BC"):
        _solver(_inlet_and_wall(domain_sdf=_obstacle_sdf))
    bounded = _solver(_inlet_and_wall(domain_sdf=_obstacle_sdf, domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]])))
    assert bounded._bc_segment_per_point[_index(bounded, [0.0, 0.5])].name == "in"


def _circle_sdf(x):
    return float(np.linalg.norm(np.asarray(x, dtype=float) - 0.5) - 0.4)


def _on_a_circle(segments: list[BCSegment]) -> HJBGFDMSolver:
    """GFDM inside the circle of radius 0.4 about (0.5, 0.5), which is the BC's ``domain_sdf``: the grid's nodes inside
    it, and 48 points on it as the boundary."""
    bc = BoundaryConditions(dimension=2, segments=segments, domain_sdf=_circle_sdf)
    grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx=[21, 21], boundary_conditions=bc)
    nodes = grid.get_spatial_grid()
    inner = nodes[np.linalg.norm(nodes - 0.5, axis=1) < 0.38]
    angles = np.arange(48) * (2 * np.pi / 48)
    ring = 0.5 + 0.4 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    model = Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)))
    conditions = Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=1.0)
    problem = MFGProblem(model=model, domain=grid, conditions=conditions, Nt=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HJBGFDMSolver(
            problem,
            collocation_points=np.vstack([inner, ring]),
            boundary_indices=np.arange(len(inner), len(inner) + len(ring)),
            delta=0.16,
            k_neighbors=14,
            derivative_method="taylor",
            taylor_order=2,
            weight_function="wendland",
            monotonicity_scheme="none",
        )


def test_a_curved_sdf_boundary_takes_the_normal_the_uniform_path_writes():
    """Review 2 of #2674. On a curved ``domain_sdf`` a point's face is the SDF gradient's dominant axis, and the mixed
    path wrote that axis as the row's normal: up to 45 degrees off the circle's, where the uniform path writes the SDF's
    normal at the point. The owner now gives the SDF's normal with the reading, so at every point the mixed BC makes
    NO_FLUX, its row normal is the one the uniform NO_FLUX BC writes there."""
    uniform = _on_a_circle([BCSegment(name="wall", bc_type=BCType.NO_FLUX)])
    mixed = _on_a_circle(_inlet_and_wall().segments)
    assert not uniform.boundary_conditions.is_mixed
    assert mixed.boundary_conditions.is_mixed
    normals = uniform._bc_config["normals"]
    no_flux = [
        (local, int(i))
        for local, i in enumerate(mixed.boundary_indices)
        if mixed._bc_segment_per_point[int(i)].bc_type is BCType.NO_FLUX
    ]
    assert no_flux
    for local, i in no_flux:
        np.testing.assert_array_equal(mixed._bc_normal_per_point[i], normals[local])


def test_a_robin_coefficient_provider_does_not_stop_construction():
    """Review 1 of #2674, blocker 2. GFDM classifies at construction, before the coupling layer resolves any provider,
    so its equivalence compares Robin coefficients as stored. Reading them through the resolving owner raised on a
    ``beta`` given as a provider."""
    from mfgarchon.geometry.boundary.providers import ConstantProvider

    bc = BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(
                name="robin", bc_type=BCType.ROBIN, alpha=0.0, beta=ConstantProvider(1.0), value=0.25, boundary="x_min"
            ),
            BCSegment(name="wall", bc_type=BCType.NO_FLUX),
        ],
    )
    solver = _solver(bc)
    assert solver._bc_segment_per_point[_index(solver, [0.0, 0.5])].name == "robin"


def test_every_refusal_the_owner_makes_is_reported_per_point():
    """Review 1 of #2674, must-fix 1: a BC with no default and no segment on the y faces. The owner refuses those
    points; GFDM collects each refusal, with the owner's reason, into one message rather than raising the first.
    Review 2, should-fix 3: a ``region_name`` segment on a geometry that does not implement ``SupportsRegionMarking``
    is refused with a TypeError, which is collected too."""
    bc = BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="in", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_min"),
            BCSegment(name="out", bc_type=BCType.NO_FLUX, boundary="x_max"),
        ],
    )
    with pytest.raises(
        ValueError, match=r"(?s)pre-classification failed for \d+/\d+ boundary points.*no segment governs"
    ):
        _solver(bc)

    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.geometry.implicit.hyperrectangle import Hyperrectangle

    marked = BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="in", bc_type=BCType.DIRICHLET, value=1.0, region_name="inlet"),
            BCSegment(name="wall", bc_type=BCType.NO_FLUX),
        ],
    )
    components = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)),
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
        boundary_conditions=marked,
    )
    points = np.stack(np.meshgrid(*[np.linspace(0.0, 1.0, NX)] * 2, indexing="ij"), axis=-1).reshape(-1, 2)
    on_box = np.flatnonzero(np.any((points == 0.0) | (points == 1.0), axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)  # an implicit domain's BC needs components=
        problem = MFGProblem(
            geometry=Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]])),
            T=1.0,
            Nt=4,
            volatility=0.1,
            components=components,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(
            ValueError, match=r"refused by the boundary owner: .*Hyperrectangle defines no region of that name"
        ):
            HJBGFDMSolver(problem, collocation_points=points, boundary_indices=on_box, monotonicity_scheme="none")


def test_gfdms_equivalence_is_the_row_it_writes():
    """Review 1 of #2674, should-fix 1: two segments are one row when GFDM writes the same row for them -- the type,
    NEUMANN and NO_FLUX together, and the datum as stored -- and ``flux_capacity`` is not read (#2575)."""
    from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import _same_bc_row
    from mfgarchon.geometry.boundary.providers import ConstantProvider

    def seg(bc_type, **kwargs):
        return BCSegment(name="s", bc_type=bc_type, **kwargs)

    provider = ConstantProvider(1.0)
    assert _same_bc_row(seg(BCType.NEUMANN, value=0.0), seg(BCType.NO_FLUX))
    assert not _same_bc_row(seg(BCType.NEUMANN, value=1.0), seg(BCType.NO_FLUX))
    assert not _same_bc_row(seg(BCType.DIRICHLET, value=1.0), seg(BCType.DIRICHLET, value=2.0))
    assert not _same_bc_row(seg(BCType.DIRICHLET, value=0.0), seg(BCType.NEUMANN, value=0.0))
    assert _same_bc_row(seg(BCType.DIRICHLET, value=0.0, flux_capacity=0.1), seg(BCType.DIRICHLET, value=0.0))
    robin = {"alpha": 0.0, "beta": provider}
    assert _same_bc_row(seg(BCType.ROBIN, value=0.25, **robin), seg(BCType.ROBIN, value=0.25, **robin))
    assert not _same_bc_row(seg(BCType.ROBIN, value=0.25, **robin), seg(BCType.ROBIN, value=0.5, **robin))
    assert not _same_bc_row(
        seg(BCType.ROBIN, value=0.25, **robin), seg(BCType.ROBIN, value=0.25, alpha=0.0, beta=ConstantProvider(1.0))
    )


def test_r1_reads_the_corner_face_by_the_row_gfdm_writes():
    """R1 at a corner reads each face's segment by GFDM's own equivalence, the row it writes, not the owner's default
    sameness. NEUMANN(0) on x_min and NO_FLUX on y_min at priority 1 write one row, so the corner the y_min segment
    takes is written along x_min's normal, the first face in face order; under the owner's default the two differ
    and the normal would be y_min's."""
    bc = BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="x_min", bc_type=BCType.NEUMANN, value=0.0, boundary="x_min"),
            BCSegment(name="y_min", bc_type=BCType.NO_FLUX, boundary="y_min", priority=1),
            BCSegment(name="rest", bc_type=BCType.NO_FLUX),
        ],
    )
    solver = _solver(bc)
    i = _index(solver, [0.0, 0.0])
    assert solver._bc_segment_per_point[i].name == "y_min"
    assert solver._bc_normal_per_point[i] == pytest.approx([-1.0, 0.0])

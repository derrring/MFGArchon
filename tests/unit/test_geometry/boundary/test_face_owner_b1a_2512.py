"""Row B1a of #2512: one owner decides which segment governs a boundary location, and the FDM family reads it.

The owner is `BoundaryConditions.segment_for_face`, `segment_for_corner` and `segment_for_point`. Their rules
are stated once, in `segment_for_face`'s docstring:
- on a face: priority, then placed before global, then declaration;
- at a corner: priority, then DIRICHLET before any other condition, then placed before global, then declaration.

The plan (#2512 comment 6124387645) names these oracles:
- Fixture A: two segments competing for one face, with a swap control;
- Fixture B: a partial ``region``, which a face family refuses by name;
- Fixture C: #2624's 1-D HJB-FDM wall;
- a tie, a 2-D corner, ``normal_direction``, and point-face agreement.

Fixture A as the plan wrote it put a second whole-face segment Q on P's face only. Under the later shadow
rule, a segment that governs no face is refused, so Q now governs a face of its own (``boundary="all"``,
"the rest"). The fixture then also pins priority beating "all", and placed beating global.

Each FDM-family reader is read off its own output: HJB-FDM's Dirichlet rows, FP-FDM's node classification,
the semi-Lagrangian post-step enforcement in 2-D, and an HJB-FDM solve.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import (
    _get_dirichlet_value_at_point,
    _is_dirichlet_at_point,
)
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    dirichlet_bc,
    neumann_bc,
    no_flux_bc,
    periodic_bc,
)
from mfgarchon.geometry.boundary.applicator_fdm import dirichlet_wall_rows
from mfgarchon.geometry.boundary.applicator_interpolation import InterpolationApplicator
from mfgarchon.geometry.boundary.conditions import _same_condition
from mfgarchon.geometry.boundary.types import BoundaryFace

N = 7  # nodes per axis
UNIT_SQUARE = np.array([[0.0, 1.0], [0.0, 1.0]])
FACES_2D = [BoundaryFace(axis, side) for axis in range(2) for side in ("min", "max")]


def _seg(name, bc_type, value=0.0, **placement):
    return BCSegment(name=name, bc_type=bc_type, value=value, **placement)


def _fixture_a(p_face="x_max", n_face="x_min"):
    """P (DIRICHLET 1, priority 1) on ``p_face``; Q (DIRICHLET 2, "all", priority 0); N (NEUMANN 0) on ``n_face``;
    W (NO_FLUX) on y_min. The owner gives P its face (priority beats "all"), N and W theirs (placed beats
    global at equal priority), and Q the one face left, y_max."""
    return BoundaryConditions(
        dimension=2,
        domain_bounds=UNIT_SQUARE,
        default_bc=BCType.NO_FLUX,
        segments=[
            _seg("P", BCType.DIRICHLET, 1.0, boundary=p_face, priority=1),
            _seg("Q", BCType.DIRICHLET, 2.0, boundary="all"),
            _seg("N", BCType.NEUMANN, 0.0, boundary=n_face),
            _seg("W", BCType.NO_FLUX, boundary="y_min"),
        ],
    )


def _grid(bc):
    return TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[N, N], boundary_conditions=bc)


def _expected_dirichlet(p_face):
    """The Dirichlet nodes and values Fixture A asks for: P's face at 1, y_max at 2. The corners are read
    by the corner order: P's face's two corners are P's (priority); the corner N's face shares with y_max is
    Q's (essential before natural at equal priority); the corner N shares with y_min is N's (Neumann)."""
    mask = np.zeros((N, N), dtype=bool)
    values = np.zeros((N, N))
    mask[:, -1], values[:, -1] = True, 2.0  # y_max: Q
    p_index = -1 if p_face == "x_max" else 0
    mask[p_index, :], values[p_index, :] = True, 1.0  # P's face, its corners included
    return mask, values


# ---------------------------------------------------------------- the owner's rules


@pytest.mark.parametrize("p_face", ["x_max", "x_min"])
def test_fixture_a_each_face_has_the_owners_segment(p_face):
    n_face = "x_min" if p_face == "x_max" else "x_max"
    bc = _fixture_a(p_face, n_face)
    names = {face.to_string(): bc.segment_for_face(face).name for face in FACES_2D}
    assert names == {p_face: "P", n_face: "N", "y_min": "W", "y_max": "Q"}


def test_fixture_a_as_first_planned_is_refused_by_the_shadow_rule():
    """The plan put Q (DIRICHLET 2, priority 0) on P's face alone. P outranks it there, so Q governs no face and is
    refused by name, rather than dropped: the fixture now gives Q the face no other segment takes."""
    bc = BoundaryConditions(
        dimension=2,
        domain_bounds=UNIT_SQUARE,
        default_bc=BCType.NO_FLUX,
        segments=[
            _seg("P", BCType.DIRICHLET, 1.0, boundary="x_max", priority=1),
            _seg("Q", BCType.DIRICHLET, 2.0, boundary="x_max"),
            _seg("N", BCType.NEUMANN, 0.0, boundary="x_min"),
            _seg("W", BCType.NO_FLUX, boundary="y_min"),
        ],
    )
    with pytest.raises(ValueError, match="'Q' \\(priority 0\\) -- x_max: 'P' \\(priority 1\\) outranks it"):
        bc.segment_for_face("x_max")


def test_a_tie_on_a_face_goes_to_the_first_declared_and_the_loser_is_refused_by_name():
    """Declaration order breaks a tie, and a segment that then governs no face is refused, not dropped."""
    bc = BoundaryConditions(
        dimension=1,
        default_bc=BCType.NO_FLUX,
        segments=[
            _seg("first", BCType.DIRICHLET, 1.0, boundary="x_max"),
            _seg("second", BCType.DIRICHLET, 2.0, boundary="x_max"),
        ],
    )
    with pytest.raises(ValueError, match="'second' \\(priority 0\\) -- x_max: 'first' \\(priority 0\\) outranks it"):
        bc.segment_for_face("x_max")


@pytest.mark.parametrize("first", ["dirichlet", "neumann"])
def test_a_corner_tie_goes_to_the_essential_condition_whatever_the_declaration_order(first):
    """At equal priority a corner is DIRICHLET's, read the same way by a node and by a point."""
    d = _seg("D", BCType.DIRICHLET, 1.0, boundary="x_min")
    n = _seg("Nm", BCType.NEUMANN, 0.0, boundary="y_min")
    bc = BoundaryConditions(
        dimension=2,
        domain_bounds=UNIT_SQUARE,
        default_bc=BCType.NO_FLUX,
        segments=[d, n] if first == "dirichlet" else [n, d],
    )
    corner = [BoundaryFace(0, "min"), BoundaryFace(1, "min")]
    assert bc.segment_for_corner(corner).name == "D"
    assert bc.segment_for_point(np.array([0.0, 0.0])).name == "D"


def test_a_point_at_a_corner_takes_the_corner_order_when_dirichlet_is_declared_later():
    """`segment_for_point` used to take the first segment in face order -- priority, placed, declaration -- so with
    the NEUMANN face declared first at equal priority, a point at their corner read NEUMANN while the node
    (`segment_for_corner`) read DIRICHLET. Every corner of this square, point against node."""
    bc = BoundaryConditions(
        dimension=2,
        domain_bounds=UNIT_SQUARE,
        default_bc=BCType.NO_FLUX,
        segments=[
            _seg("Nm", BCType.NEUMANN, 0.0, boundary="y_min"),
            _seg("Nx", BCType.NEUMANN, 0.0, boundary="x_max"),
            _seg("D", BCType.DIRICHLET, 1.0, boundary="x_min"),
            _seg("Dy", BCType.DIRICHLET, 2.0, boundary="y_max"),
        ],
    )
    for x in (0.0, 1.0):
        for y in (0.0, 1.0):
            faces = [BoundaryFace(0, "min" if x == 0.0 else "max"), BoundaryFace(1, "min" if y == 0.0 else "max")]
            assert bc.segment_for_point(np.array([x, y])).name == bc.segment_for_corner(faces).name, (x, y)
    assert bc.segment_for_point(np.array([0.0, 0.0])).name == "D"  # NEUMANN declared first, DIRICHLET wins
    assert bc.segment_for_point(np.array([1.0, 1.0])).name == "Dy"


def test_at_a_corner_priority_comes_before_the_essential_condition():
    bc = BoundaryConditions(
        dimension=2,
        domain_bounds=UNIT_SQUARE,
        default_bc=BCType.NO_FLUX,
        segments=[
            _seg("D", BCType.DIRICHLET, 1.0, boundary="x_min"),
            _seg("Nm", BCType.NEUMANN, 0.0, boundary="y_min", priority=1),
        ],
    )
    assert bc.segment_for_corner([BoundaryFace(0, "min"), BoundaryFace(1, "min")]).name == "Nm"
    assert bc.segment_for_point(np.array([0.0, 0.0])).name == "Nm"


def test_a_partial_region_is_refused_by_name_unless_the_consumer_calls_it_equivalent():
    """Fixture B at the owner: a strip of x_max at a higher priority than the walls."""
    strip = _seg("strip", BCType.DIRICHLET, 1.0, region={"x": (0.9, 1.0), "y": (0.4, 0.6)}, priority=1)
    walls = _seg("walls", BCType.NEUMANN, 0.0)
    bc = BoundaryConditions(dimension=2, domain_bounds=UNIT_SQUARE, segments=[strip, walls])
    with pytest.raises(ValueError, match="'strip' covers part of the x_max face"):
        bc.segment_for_face("x_max")
    assert bc.segment_for_face("x_max", equivalent=lambda a, b: True).name == "walls"


def test_normal_direction_selects_the_face_it_points_out_of():
    bc = BoundaryConditions(
        dimension=2,
        domain_bounds=UNIT_SQUARE,
        default_bc=BCType.NO_FLUX,
        segments=[_seg("east", BCType.DIRICHLET, 1.0, normal_direction=np.array([1.0, 0.0]))],
    )
    governing = {face.to_string(): bc.segment_for_face(face) for face in FACES_2D}
    assert governing["x_max"].name == "east"
    assert all(seg.bc_type is BCType.NO_FLUX for face, seg in governing.items() if face != "x_max")


@pytest.mark.parametrize(
    "bc",
    [
        pytest.param(_fixture_a(), id="fixture_a"),
        pytest.param(_fixture_a("x_min", "x_max"), id="fixture_a_swapped"),
        pytest.param(
            BoundaryConditions(
                dimension=2,
                domain_bounds=UNIT_SQUARE,
                default_bc=BCType.NO_FLUX,
                segments=[_seg("east", BCType.DIRICHLET, 1.0, normal_direction=np.array([1.0, 0.0]))],
            ),
            id="normal_direction",
        ),
    ],
)
def test_a_point_and_its_face_agree(bc):
    """Wherever the face rule answers, the point rule gives the same segment: a face's interior points, and
    a corner's point against the corner's node."""
    for face in FACES_2D:
        point = np.full(2, 0.5)
        point[face.axis] = 0.0 if face.side == "min" else 1.0
        expected = bc.segment_for_face(face)
        got = bc.segment_for_point(point, face)
        assert got.name == expected.name, face.to_string()
    for x in (0.0, 1.0):
        for y in (0.0, 1.0):
            faces = [BoundaryFace(0, "min" if x == 0.0 else "max"), BoundaryFace(1, "min" if y == 0.0 else "max")]
            assert bc.segment_for_point(np.array([x, y])).name == bc.segment_for_corner(faces).name, (x, y)


def test_an_unset_default_and_a_robin_default_are_refused_where_they_would_govern():
    """Only where they would govern: a face a segment governs keeps it under either default. Refusing every face once
    any face falls to an unset or a ROBIN default reddens the two `x_max` lines."""
    lone = [_seg("P", BCType.DIRICHLET, 1.0, boundary="x_max")]
    with pytest.raises(ValueError, match="no segment governs the x_min face, and default_bc was not specified"):
        BoundaryConditions(dimension=1, segments=lone).segment_for_face("x_min")
    with pytest.raises(ValueError, match="ROBIN"):
        BoundaryConditions(dimension=1, segments=lone, default_bc=BCType.ROBIN).segment_for_face("x_min")
    assert BoundaryConditions(dimension=1, segments=lone).segment_for_face("x_max").name == "P"
    assert BoundaryConditions(dimension=1, segments=lone, default_bc=BCType.ROBIN).segment_for_face("x_max").name == "P"


def test_a_face_of_an_axis_the_domain_lacks_is_refused_whichever_reader_asks():
    """Ruling (c): a segment on a face of an axis the domain lacks governs nothing and is refused, even where it would
    change nothing -- here beside a global NO_FLUX that the uniform path would otherwise answer with. The check needs
    only the dimension, so it runs wherever the dimension becomes known, and no reader can route around it."""
    segments = [_seg("walls", BCType.NO_FLUX), _seg("lid", BCType.NO_FLUX, boundary="z_max")]
    with pytest.raises(ValueError, match="a face of axis 2"):  # construction
        BoundaryConditions(dimension=2, default_bc=BCType.NO_FLUX, segments=segments)
    unbound = BoundaryConditions(default_bc=BCType.NO_FLUX, segments=segments)
    assert unbound.uniform_segment is not None  # no dimension yet: nothing to check against
    with pytest.raises(ValueError, match="a face of axis 2"):  # binding
        unbound.bind_dimension(2)
    with pytest.raises(ValueError, match="a face of axis 2"):  # a geometry handing an unbound BC its dimension
        unbound.segment_for_face("x_min", _grid(no_flux_bc(dimension=2)))
    BoundaryConditions(dimension=3, default_bc=BCType.NO_FLUX, segments=segments)  # control: z_max exists in 3-D


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("factory", [dirichlet_bc, neumann_bc, no_flux_bc, periodic_bc])
def test_the_uniform_shortcut_is_the_owners_answer_on_every_face(factory, dimension):
    """`uniform_segment` is the read a uniform-only fast path takes instead of `segments[0]` (#2512 row B1)."""
    bc = factory(dimension=dimension)
    faces = [BoundaryFace(axis, side) for axis in range(dimension) for side in ("min", "max")]
    assert bc.uniform_segment is not None
    assert all(bc.segment_for_face(face) is bc.uniform_segment for face in faces)


def test_the_uniform_shortcut_is_none_when_two_faces_differ():
    assert _fixture_a().uniform_segment is None


def test_the_uniform_shortcut_is_decided_by_the_owners_sameness_rule():
    """`uniform_segment` is a segment whose condition every location carries, by the owner's one sameness rule
    (`_same_condition`: type, value, flux capacity, Robin coefficients) over every condition the BC can impose --
    each segment's and the default's. When they are all the same, which segment governs cannot matter, so no
    placement is read (a BC on a domain with no box faces is answered too, and no geometry is needed).

    - Four alike NO_FLUX segments, one per face, over a NO_FLUX default (#2626's spelling): uniform. One with a
      flux capacity: not uniform.
    - A BC with no segments and a ``default_bc``: uniform.
    - A global segment (no placement field, or ``boundary="all"``) covers every location, so the default governs none
      and is not compared: one ``boundary="all"`` segment over a different default is uniform, and so is a uniform
      factory BC whose callable value left ``default_value`` at 0.0.
    - Four alike segments over a different default are not uniform here, though they cover every face: no placement
      is read, so the default is compared. ``None`` only sends a consumer down its per-face path.
    - No ``default_bc`` and no global segment: ``None``, since a location may fall through to the unset default.
    """

    def per_face(capacities, default=BCType.NO_FLUX):
        return BoundaryConditions(
            dimension=2,
            default_bc=default,
            segments=[
                BCSegment(name=face.to_string(), bc_type=BCType.NO_FLUX, boundary=face.to_string(), flux_capacity=cap)
                for face, cap in zip(FACES_2D, capacities, strict=True)
            ],
        )

    alike = per_face([None] * 4)
    assert alike.uniform_segment in alike.segments
    assert per_face([None, None, None, 0.5]).uniform_segment is None
    assert per_face([None] * 4, default=None).uniform_segment is None
    segment_free = BoundaryConditions(dimension=2, segments=[], default_bc=BCType.NO_FLUX)
    assert segment_free.uniform_segment.bc_type is BCType.NO_FLUX
    unbound = no_flux_bc()
    assert unbound.dimension is None
    assert unbound.uniform_segment is unbound.segments[0]
    everywhere = BoundaryConditions(
        dimension=2, default_bc=BCType.NO_FLUX, segments=[_seg("a", BCType.DIRICHLET, 1.0, boundary="all")]
    )
    assert everywhere.uniform_segment is everywhere.segments[0]
    callable_value = dirichlet_bc(dimension=2, value=lambda x: 42.0)
    assert callable_value.uniform_segment is callable_value.segments[0]
    assert per_face([None] * 4, default=BCType.NEUMANN).uniform_segment is None
    alone = BoundaryConditions(dimension=2, segments=[_seg("a", BCType.DIRICHLET, 1.0, boundary="all")])
    assert alone.uniform_segment is alone.segments[0]


@pytest.mark.parametrize(
    "bc",
    [
        pytest.param(no_flux_bc(dimension=2), id="uniform"),
        pytest.param(dirichlet_bc(dimension=3, value=2.0), id="uniform_3d"),
        pytest.param(BoundaryConditions(dimension=2, segments=[], default_bc=BCType.NEUMANN), id="segment_free"),
        pytest.param(
            BoundaryConditions(
                dimension=2,
                default_bc=BCType.NO_FLUX,
                segments=[_seg(f.to_string(), BCType.NO_FLUX, boundary=f.to_string()) for f in FACES_2D],
            ),
            id="per_face_alike",
        ),
        pytest.param(
            BoundaryConditions(dimension=2, segments=[_seg("a", BCType.DIRICHLET, 1.0, boundary="all")]), id="all_alone"
        ),
    ],
)
def test_a_uniform_segment_is_the_condition_the_owner_gives_every_face(bc):
    """The property `uniform_segment` stands for, against the face owner itself."""
    uniform = bc.uniform_segment
    assert uniform is not None
    faces = [BoundaryFace(axis, side) for axis in range(bc.dimension) for side in ("min", "max")]
    assert all(_same_condition(bc.segment_for_face(face), uniform) for face in faces)


def test_a_shadowed_exit_with_another_flux_capacity_is_not_effect_neutral():
    """The shadow rule exempts a segment that imposes the same condition as the one outranking it. Two Dirichlet
    exits that differ only in flux capacity let out different mass, so the lower one is refused, not exempted."""
    bc = BoundaryConditions(
        dimension=1,
        default_bc=BCType.NO_FLUX,
        segments=[
            BCSegment(
                name="fast", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max", priority=1, flux_capacity=0.2
            ),
            BCSegment(name="slow", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max", flux_capacity=0.1),
        ],
    )
    with pytest.raises(ValueError, match="'slow' \\(priority 0\\) -- x_max: 'fast' \\(priority 1\\) outranks it"):
        bc.segment_for_face("x_max")


def test_the_owners_memo_does_not_answer_for_a_new_grid_at_a_freed_grids_address():
    """The owner memoises covers and its shadow check per geometry, keyed by ``id()``. CPython gives a freed object's
    address to a new one, and a cover read on a 3 x 3 grid then answered for an 11 x 11 grid: segment Q (x_min,
    ``region={1: (0.2, 0.8)}``) holds x_min's one interior node on the first and only part of x_min on the second.
    Whether the allocator reuses the address depends on allocation order (2 of 20 in one loop, 44 of 50 in the audit
    session's), so the reuse is simulated here: the 3 x 3 grid's entries are re-keyed under the 11 x 11 grid's id."""

    def bc():
        return BoundaryConditions(
            dimension=2,
            default_bc=BCType.NO_FLUX,
            segments=[_seg("Q", BCType.DIRICHLET, 1.0, boundary="x_min", region={1: (0.2, 0.8)}, priority=1)],
        )

    def grid(n):
        return TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[n, n], boundary_conditions=no_flux_bc(dimension=2))

    small, large = grid(3), grid(11)
    assert bc().segment_for_face("x_min", small).name == "Q"  # control: a fresh BC on each grid
    with pytest.raises(ValueError, match="'Q' covers part of the x_min face"):
        bc().segment_for_face("x_min", large)
    shared = bc()
    shared.segment_for_face("x_min", small)
    memo = shared.__dict__["_b1_owner_cache"]
    for slot in [slot for slot in memo if slot[-1] == id(small)]:
        memo[(*slot[:-1], id(large))] = memo[slot]
    with pytest.raises(ValueError, match="'Q' covers part of the x_min face"):
        shared.segment_for_face("x_min", large)


# ---------------------------------------------------------------- the FDM family, read off its own output


@pytest.mark.parametrize("p_face", ["x_max", "x_min"])
def test_fixture_a_hjb_rows_and_fp_nodes_impose_the_same_dirichlet_walls(p_face):
    """HJB-FDM's rows and FP-FDM's node classification are one answer, and it is Fixture A's. The swap moves it."""
    n_face = "x_min" if p_face == "x_max" else "x_max"
    bc = _fixture_a(p_face, n_face)
    grid = _grid(bc)
    mask, values = dirichlet_wall_rows(bc, (N, N), 0.0, grid)
    expected_mask, expected_values = _expected_dirichlet(p_face)
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(values[mask], expected_values[mask])
    fp_mask = np.zeros((N, N), dtype=bool)
    fp_values = np.zeros((N, N))
    for index in np.ndindex(N, N):
        if _is_dirichlet_at_point(bc, index, (N, N), grid):
            fp_mask[index] = True
            fp_values[index] = _get_dirichlet_value_at_point(bc, index, (N, N), time=0.0, geometry=grid)
    np.testing.assert_array_equal(fp_mask, expected_mask)
    np.testing.assert_array_equal(fp_values, expected_values)


@pytest.mark.parametrize("p_face", ["x_max", "x_min"])
def test_fixture_a_semi_lagrangian_enforcement_writes_the_same_walls(p_face):
    """The 2-D post-step enforcement writes each Dirichlet face's value, corners by the corner order."""
    n_face = "x_min" if p_face == "x_max" else "x_max"
    bc = _fixture_a(p_face, n_face)
    field = np.random.default_rng(2512).standard_normal((N, N))
    out = InterpolationApplicator(dimension=2).enforce_values(
        field, bc, spacing=(1 / (N - 1),) * 2, time=0.0, geometry=_grid(bc)
    )
    expected_mask, expected_values = _expected_dirichlet(p_face)
    np.testing.assert_array_equal(out[expected_mask], expected_values[expected_mask])


def test_fixture_a_hjb_solve_holds_the_walls_and_moves_with_the_swap():
    """The solver's own output: P's face holds 1 and y_max holds 2 at t = 0; 2 appears nowhere else on a wall."""

    def solve(p_face, n_face):
        bc = _fixture_a(p_face, n_face)
        problem = MFGProblem(
            model=Model(
                hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.3
            ),
            domain=_grid(bc),
            conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.2),
            Nt=4,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(
                HJBFDMSolver(problem).solve_hjb_system(np.ones((5, N, N)), np.zeros((N, N)), np.zeros((5, N, N)))
            )[0]

    for p_face, n_face, p_index, n_index in (("x_max", "x_min", -1, 0), ("x_min", "x_max", 0, -1)):
        u = solve(p_face, n_face)
        np.testing.assert_array_equal(u[p_index, :], 1.0)
        np.testing.assert_array_equal(u[1:-1, -1], 2.0)
        assert not np.any(u[n_index, :-1] == 2.0), (p_face, u[n_index])


_FACE_READERS = {
    "hjb_rows": lambda bc, grid: dirichlet_wall_rows(bc, (N, N), 0.0, grid),
    "fp_nodes": lambda bc, grid: [_is_dirichlet_at_point(bc, index, (N, N), grid) for index in np.ndindex(N, N)],
    "sl_enforcement": lambda bc, grid: InterpolationApplicator(dimension=2).enforce_values(
        np.ones((N, N)), bc, spacing=(1 / (N - 1),) * 2, geometry=grid
    ),
}


@pytest.mark.parametrize("family", list(_FACE_READERS))
def test_fixture_b_a_partial_region_is_refused_by_every_face_reader(family):
    strip = _seg("strip", BCType.DIRICHLET, 1.0, region={"x": (0.9, 1.0), "y": (0.4, 0.6)}, priority=1)
    walls = _seg("walls", BCType.NEUMANN, 0.0)
    bc = BoundaryConditions(dimension=2, domain_bounds=UNIT_SQUARE, segments=[strip, walls])
    grid = _grid(bc)
    with pytest.raises(ValueError, match="'strip' covers part of the x_max face"):
        _FACE_READERS[family](bc, grid)


def _solve_1d(segments):
    bc = BoundaryConditions(segments=segments, dimension=1, default_bc=BCType.NEUMANN)
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=bc)
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.3
        ),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.5),
        Nt=10,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(HJBFDMSolver(problem).solve_hjb_system(np.ones((11, 21)), np.zeros(21), np.zeros((11, 21))))


def test_fixture_c_one_wall_is_not_read_by_two_rules_in_one_solve_2624():
    """#2624: a Dirichlet placed by a face label at priority 1 over a Neumann placed by ``boundary=`` at 0, one
    wall. The ghosts read the first and the wall rows the second, and the solve gave U(0, 0) = 1.025536, which
    is neither (measured at 1a036677 and at 1052ec62). Both read the owner now, and the Neumann, which then
    governs no face, is refused by name.

    The controls are unchanged: the Dirichlet alone, by either placement, 3.000000; Neumann alone, 0.000000.
    """
    a_label = _seg("A", BCType.DIRICHLET, 3.0, region_name="x_min", priority=1)
    b = _seg("B", BCType.NEUMANN, 0.0, boundary="x_min")
    c = _seg("C", BCType.NEUMANN, 0.0, boundary="x_max")
    with pytest.raises(ValueError, match="'B' \\(priority 0\\) -- x_min: 'A' \\(priority 1\\) outranks it"):
        _solve_1d([a_label, b, c])
    assert _solve_1d([a_label, c])[0, 0] == pytest.approx(3.0, abs=1e-12)
    assert _solve_1d([_seg("A", BCType.DIRICHLET, 3.0, boundary="x_min", priority=1), c])[0, 0] == pytest.approx(
        3.0, abs=1e-12
    )
    assert _solve_1d([b, c])[0, 0] == pytest.approx(0.0, abs=1e-12)

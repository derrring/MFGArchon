"""#2490 B1e-2: a point reader refuses, by name, a segment that governs none of its own points, through the boundary
owner's point-family check (`BoundaryConditions.refuse_segments_governing_no_point`): the particle applicator over the
faces it reads a particle on, HJB-GFDM's classification over its boundary collocation points. Before, the particles
dropped such a segment with no word, and HJB-GFDM refused most of them only through its corner points, which ran the
face owner's whole-BC check on a grid's nodes in place of the reader's own points. A corner no longer runs that check."""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc
from mfgarchon.geometry.boundary.applicator_particle import ParticleApplicator, same_particle_action

D, NF = BCType.DIRICHLET, BCType.NO_FLUX


def _seg(name, bc_type, value=0.0, **placement):
    return BCSegment(name=name, bc_type=bc_type, value=value, **placement)


def _problem(grid, nt=5):
    model = Model(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.05
    )
    return MFGProblem(
        model=model, domain=grid, conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.2), Nt=nt
    )


def _particles_remaining(segments, route="own") -> int:
    """Of 200 particles drifting right through x_max near y = 0.45 (#2780's fixture), how many stay in the domain under
    a NO_FLUX default beside ``segments``: the BC passed to the solver (``route="own"``), or the problem's, set on its
    grid and read by the solver's FP view of it (``"shared"``)."""
    bc = BoundaryConditions(segments=segments, dimension=2, default_bc=NF)
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[11, 11],
        boundary_conditions=bc if route == "shared" else no_flux_bc(dimension=2),
    )
    own = bc if route == "own" else None
    solver = FPParticleSolver(_problem(grid), num_particles=200, seed=0, boundary_conditions=own)
    solver.solve_fp_system(
        initial_particles=np.random.default_rng(0).uniform([0.8, 0.43], [0.95, 0.47], (200, 2)),
        drift_field=lambda t, x, m: np.broadcast_to(np.array([3.0, 0.0]), np.shape(x)),
        drift_needs_density=False,
        show_progress=False,
    )
    final = np.asarray(solver.M_particles_trajectory[-1], dtype=float).reshape(-1, 2)
    return int(np.count_nonzero(np.all(np.isfinite(final), axis=1)))


def _gfdm(segments, points=None, delta=0.25) -> HJBGFDMSolver:
    """HJB-GFDM on the 11 x 11 grid's nodes, or on ``points``, with the BC set on the grid: a NO_FLUX default beside
    ``segments``."""
    bc = BoundaryConditions(segments=segments, dimension=2, default_bc=NF)
    grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=bc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HJBGFDMSolver(
            _problem(grid, nt=4),
            collocation_points=grid.get_spatial_grid() if points is None else points,
            delta=delta,
            monotonicity_scheme="none",
        )


def _reading(solver: HJBGFDMSolver, name: str) -> int:
    return sum(segment.name == name for segment in solver._bc_segment_per_point.values())


_SHADOWED = r"these segments govern none of {}.*'exit' \(priority 0\) -- outranked at \[1\.0, [0-9.]+\]{}[^:]*: 'wall' \(priority 1\)"
_NOWHERE = r"segment\(s\) 'exit' hold none of {}"
_PARTICLE_POINTS = r"the \d+ points it reads the faces of the box"
_GFDM_POINTS = r"its 40 boundary collocation points"

# Each case: segments beside the NO_FLUX default; then, per reader, a refusal pattern or the admitted outcome --
# particles remaining of 200 for the particles, boundary points reading 'exit' for HJB-GFDM.
CASES = {
    "an exit under a wall on its face (S3)": (
        [_seg("exit", D, boundary="x_max"), _seg("wall", NF, boundary="x_max", priority=1)],
        _SHADOWED.format(_PARTICLE_POINTS, " on x_max"),
        _SHADOWED.format(_GFDM_POINTS, ""),
    ),
    "S3's control: a wall imposing the exit's condition": (
        [_seg("exit", D, boundary="x_max"), _seg("wall", D, boundary="x_max", priority=1)],
        0,
        0,
    ),
    "an exit strip between the grid's nodes (S2)": (
        [_seg("exit", D, region={"x": (1.0, 1.0), "y": (0.42, 0.48)})],
        12,
        _NOWHERE.format(_GFDM_POINTS),
    ),
    "S2's control: a strip holding nodes": ([_seg("exit", D, region={"x": (1.0, 1.0), "y": (0.35, 0.65)})], 0, 3),
    "a region off the box (S4)": (
        [_seg("exit", D, region={"x": (1.0, 1.0), "y": (1.5, 2.0)})],
        _NOWHERE.format(_PARTICLE_POINTS),
        _NOWHERE.format(_GFDM_POINTS),
    ),
    "an sdf_region off the box (S5)": (
        [_seg("exit", D, sdf_region=lambda p: float(np.hypot(p[0] - 2.0, p[1] - 2.0) - 0.1))],
        _NOWHERE.format(_PARTICLE_POINTS),
        _NOWHERE.format(_GFDM_POINTS),
    ),
    "a normal_direction no face points along (S6)": (
        [_seg("exit", D, normal_direction=np.array([1.0, 1.0]) / np.sqrt(2.0), normal_tolerance=0.9)],
        _NOWHERE.format(_PARTICLE_POINTS),
        _NOWHERE.format(_GFDM_POINTS),
    ),
    "an exit strip under a wider wall strip (S7)": (
        [
            _seg("exit", D, region={"x": (1.0, 1.0), "y": (0.3, 0.7)}),
            _seg("wall", NF, region={"x": (1.0, 1.0), "y": (0.2, 0.8)}, priority=1),
        ],
        _SHADOWED.format(_PARTICLE_POINTS, " on x_max"),
        _SHADOWED.format(_GFDM_POINTS, ""),
    ),
    "an exit strip a wall covers in part (S8)": (
        [
            _seg("exit", D, region={"x": (1.0, 1.0), "y": (0.3, 0.7)}),
            _seg("wall", NF, region={"x": (1.0, 1.0), "y": (0.2, 0.5)}, priority=1),
        ],
        190,
        2,
    ),
}


@pytest.mark.parametrize("route", ["own", "shared"])
@pytest.mark.parametrize("case", list(CASES))
def test_the_particle_reader_refuses_a_segment_governing_none_of_its_points(case, route):
    """At f2dcb164 the particles kept all 200 under S3 to S7: the exit was dropped with no word. S2's exit governs a
    strip of x_max, which particles reach, so it is admitted; its 12 are #2780's, read past the wall outside it. Both
    routes the solver takes its BC by end in the one assignment the check follows."""
    segments, expected, _ = CASES[case]
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=f"FPParticleSolver: {expected}"):
            _particles_remaining(segments, route)
    else:
        assert _particles_remaining(segments, route) == expected


@pytest.mark.parametrize("case", list(CASES))
def test_hjb_gfdm_refuses_a_segment_governing_none_of_its_points_by_name(case):
    """At f2dcb164 HJB-GFDM refused S2 to S6 only through its four corner points, with "pre-classification failed for
    4/40 boundary points", and admitted S7 with the exit read at no point."""
    segments, _, expected = CASES[case]
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=f"HJBGFDMSolver: {expected}"):
            _gfdm(segments)
    else:
        assert _reading(_gfdm(segments), "exit") == expected


def test_hjb_gfdms_admission_no_longer_depends_on_whether_its_cloud_samples_a_corner():
    """The grid's nodes plus (1, 0.45), inside an exit strip that holds no node. With the corners in the cloud the face
    owner's check, run at the corners on the grid's nodes, refused the BC (4/41); without them it was admitted. Both
    clouds are now read on their own points: admitted, (1, 0.45) the exit's, every other point read alike."""
    exit_strip = [_seg("exit", D, region={"x": (1.0, 1.0), "y": (0.42, 0.48)})]
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=no_flux_bc(dimension=2)
    )
    nodes = np.asarray(grid.get_spatial_grid())
    cloud = np.vstack([nodes, [[1.0, 0.45]]])
    corner = np.isin(cloud[:, 0], [0.0, 1.0]) & np.isin(cloud[:, 1], [0.0, 1.0])
    readings = []
    for points in (cloud, cloud[~corner]):
        solver = _gfdm(exit_strip, points)
        readings.append(
            {
                tuple(np.round(solver.collocation_points[i], 6)): seg.name
                for i, seg in solver._bc_segment_per_point.items()
            }
        )
    with_corners, without = readings
    assert with_corners[(1.0, 0.45)] == without[(1.0, 0.45)] == "exit"
    assert len(with_corners) == len(without) + 4
    assert all(with_corners[point] == name for point, name in without.items())


def _sdf_disk_exit(radius):
    return [_seg("exit", D, sdf_region=lambda p: float(np.hypot(p[0] - 1.0, p[1] - 0.5) - radius))]


def test_an_sdf_exit_held_on_less_than_a_lattice_cell_is_refused_naming_the_lattice():
    """An sdf_region has no exact cut on a face, so the particle reader samples each face on a lattice of 1024 cells
    (FACE_LATTICE_CELLS) and reads it at their centres, which straddle y = 0.5 at 0.5/1024 on x_max. A disk at (1, 0.5)
    of radius 0.4/1024 holds none of them: refused, the lattice named, as the one error the sampling can make. At
    radius 0.6/1024 it holds two, and governs them."""
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    bc = BoundaryConditions(segments=_sdf_disk_exit(0.4 / 1024), dimension=2, default_bc=NF)
    with pytest.raises(ValueError, match=r"hold none of .*a lattice of 1024 equal parts along each axis of a face"):
        ParticleApplicator().refuse_segments_it_would_drop(bc, bounds)
    ParticleApplicator().refuse_segments_it_would_drop(
        BoundaryConditions(segments=_sdf_disk_exit(0.6 / 1024), dimension=2, default_bc=NF), bounds
    )


def test_a_set_of_no_width_on_a_face_is_refused_for_particles_and_read_by_hjb_gfdm():
    """A region whose range on x_max is the single coordinate y = 0.5: a particle reaches it with probability zero, so
    the particle reader refuses it by name. HJB-GFDM reads its own points, and (1, 0.5) is one of them."""
    point_exit = [_seg("exit", D, region={"x": (1.0, 1.0), "y": (0.5, 0.5)})]
    with pytest.raises(ValueError, match=r"'exit' hold none of .*probability zero"):
        _particles_remaining(point_exit)
    assert _reading(_gfdm(point_exit), "exit") == 1


def test_each_reader_decides_an_effect_neutral_shadow_by_its_own_sameness():
    """A REFLECTING segment under a NO_FLUX wall on x_max, beside an exit on x_min that sends the particles down the
    per-point path: a particle reflects off either and reads nothing else (`same_particle_action`), so the particles
    admit it and keep all 200, where the owner's default sameness, which compares the types, would refuse it. A
    Dirichlet exit with the value 1 under a Dirichlet wall with the value 2: HJB-GFDM writes the value into its row
    (`_same_bc_row`) and refuses the exit by name. A Neumann exit with datum 0 under the NO_FLUX wall: HJB-GFDM writes
    one row for both (review 1 of #2782, B3), so it admits it, where the default sameness would refuse it."""
    reflecting = [
        _seg("exit", BCType.REFLECTING, boundary="x_max"),
        _seg("wall", NF, boundary="x_max", priority=1),
        _seg("left", D, boundary="x_min"),
    ]
    assert _particles_remaining(reflecting) == 200
    shadowed = [_seg("exit", D, 1.0, boundary="x_max"), _seg("wall", D, 2.0, boundary="x_max", priority=1)]
    with pytest.raises(ValueError, match=_SHADOWED.format(_GFDM_POINTS, "")):
        _gfdm(shadowed)
    neumann = [_seg("exit", BCType.NEUMANN, 0.0, boundary="x_max"), _seg("wall", NF, boundary="x_max", priority=1)]
    assert _reading(_gfdm(neumann), "wall") == 11


def test_a_corner_compares_a_segment_with_its_own_faces_segment():
    """Review 1 of #2782, B1 and B2. At a corner each face gives its own segment and the corner rule chooses among
    them, so a segment held on a face at a corner is outranked there by that face's segment. An exit under a wall that
    imposes its condition on x_max is effect-neutral there, though the corner (1, 0) is a higher-priority floor's: the
    BC is admitted and read as at f2dcb164 (default 19, floor 11, wall 10). An exit held only at the corner (1, 1) is
    read at none of them: refused, naming each face's own segment there and the corner rule, with no advice to raise
    its priority, which could not help. Review 2: each segment gets its own remedy. Raising helps where the face is read
    at the point -- a door covers part of x_max -- and where the corner rule chose another face's segment; there the
    refusal advises it, and the raised BC is admitted. Under the wall alone it does not help, and is not advised.
    Review 3: on points coarser than the grid, a segment covering part of y_max with the wall's condition is held only at
    the corner (0, 1); raised, it sits ahead of the wall, which still answers for the face, so no priority reads it and
    raising is not advised. Covering it with another condition, it is advised and admitted raised. A cover that cannot
    be read gives the cover's own refusal, not a claim about it."""
    floor = [
        _seg("exit", D, boundary="x_max"),
        _seg("wall", D, boundary="x_max", priority=1),
        _seg("floor", NF, boundary="y_min", priority=2),
    ]
    solver = _gfdm(floor)
    assert {n: _reading(solver, n) for n in ("default", "floor", "wall", "exit")} == {
        "default": 19,
        "floor": 11,
        "wall": 10,
        "exit": 0,
    }
    corner_only = [_seg("exit", D, 0.3, region={"x": (0.95, 1.05), "y": (0.95, 1.05)}, priority=100)]
    with pytest.raises(
        ValueError, match=r"'exit' \(priority 100\) -- outranked at \[1\.0, 1\.0\], a corner, on x_max"
    ) as refused:
        _gfdm(corner_only)
    assert "the corner rule" in str(refused.value)
    assert "raise its priority" not in str(refused.value)
    assert "it is held only at corners, where each face answers with its own segment" in str(refused.value)

    def corner(priority, door=True, top=False):
        segments = [_seg("C", D, 0.3, region={"x": (0.95, 1.05), "y": (0.95, 1.05)}, priority=priority)]
        if door:
            segments.append(_seg("door", D, region={"x": (1.0, 1.0), "y": (0.3, 0.7)}, priority=1))
        if top:
            segments.append(_seg("top", NF, boundary="y_max", priority=2))
        else:
            segments.append(_seg("wall", NF, priority=1))
        return segments

    for raisable, raised in ((corner(0), corner(5)), (corner(1, top=True), corner(3, top=True))):
        with pytest.raises(
            ValueError, match=r"'C' \(priority \d\) -- outranked at .*: raise its priority above theirs"
        ):
            _gfdm(raisable)
        assert _reading(_gfdm(raised), "C") == 1
    for priority in (0, 5):
        with pytest.raises(ValueError, match=r"'C' .*: it is held only at corners, where each face answers"):
            _gfdm(corner(priority, door=False))

    lattice = np.array([[x, y] for x in np.linspace(0.0, 1.0, 6) for y in np.linspace(0.0, 1.0, 6)])

    def top_strip(bc_type, value, priority):
        strip = _seg("S", bc_type, value, region={"x": (-0.05, 0.15), "y": (1.0, 1.05)}, priority=priority)
        return [_seg("inlet", D, 1.0, boundary="x_min"), _seg("wall", NF), strip]

    for priority in (0, 10):
        with pytest.raises(
            ValueError, match=r"'S' \(priority \d+\) -- outranked .*: it is held only at corners, where"
        ):
            _gfdm(top_strip(NF, 0.0, priority), lattice, delta=0.45)
    with pytest.raises(ValueError, match=r"'S' \(priority 0\) -- outranked .*: raise its priority above theirs"):
        _gfdm(top_strip(BCType.NEUMANN, 1.0, 0), lattice, delta=0.45)
    assert _reading(_gfdm(top_strip(BCType.NEUMANN, 1.0, 10), lattice, delta=0.45), "S") == 1
    # Review 4: the sameness is the reader's. NEUMANN 0 writes the wall's row in HJB-GFDM, so raised it changes no
    # face's answer, though `_same_condition` calls it another condition.
    for priority in (0, 10):
        with pytest.raises(
            ValueError, match=r"'S' \(priority \d+\) -- outranked .*: it is held only at corners, where"
        ):
            _gfdm(top_strip(BCType.NEUMANN, 0.0, priority), lattice, delta=0.45)
    # A segment covering a face whole, held only at its corners because no point lies inside it: raised, it becomes
    # the face's segment.
    no_x_max = lattice[(lattice[:, 0] < 1.0) | np.isin(lattice[:, 1], (0.0, 1.0))]
    behind = [_seg("wall", NF, priority=1), _seg("S", D, 1.0, boundary="x_max")]
    with pytest.raises(ValueError, match=r"'S' \(priority 0\) -- outranked .*: raise its priority above theirs"):
        _gfdm(behind, no_x_max, delta=0.45)
    assert _reading(_gfdm([behind[0], _seg("S", D, 1.0, boundary="x_max", priority=2)], no_x_max, delta=0.45), "S") == 2

    disk = lambda x: float(np.linalg.norm(np.asarray(x, dtype=float) - [0.0, 1.0]) - 0.08)  # noqa: E731
    unread = BoundaryConditions(
        segments=[_seg("wall", NF, priority=1), _seg("S", D, 1.0, sdf_region=disk)],
        dimension=2,
        default_bc=NF,
        domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
    )
    ring = [np.array([x, y]) for x in np.linspace(0, 1, 11) for y in np.linspace(0, 1, 11) if {x, y} & {0.0, 1.0}]
    with pytest.raises(
        ValueError, match=r"'S' .*cannot be read here: BC segment 'S' is placed by sdf_region"
    ) as unread_:
        unread.refuse_segments_governing_no_point([(p, None, unread.segment_for_point(p)) for p in ring], "r", "them")
    assert "covers no part of" not in str(unread_.value)


def _particle_solver(segments, **bc_kw):
    bc = BoundaryConditions(segments=segments, dimension=2, default_bc=NF, **bc_kw)
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=no_flux_bc(dimension=2)
    )
    return FPParticleSolver(_problem(grid), num_particles=10, seed=0, boundary_conditions=bc)


_UNIT_BOX_SDF = lambda x: float(  # noqa: E731
    np.linalg.norm(np.maximum(np.abs(np.asarray(x, float) - 0.5) - 0.5, 0.0))
    + min(float(np.max(np.abs(np.asarray(x, float) - 0.5) - 0.5)), 0.0)
)
_DIAGONAL = {"normal_direction": np.array([1.0, 1.0]) / np.sqrt(2.0), "normal_tolerance": 0.9}


@pytest.mark.parametrize(
    ("refused", "bc_kw", "reason", "advised"),
    [
        (
            {"region": {"x": (1.0, 1.0), "y": (1.0, 1.0)}},
            {},
            r"meets each face it reaches in a set of no width .* or place it by boundary=",
            [{"region": {"x": (1.0, 1.0), "y": (0.9, 1.0)}}, {"boundary": "x_max"}],
        ),
        (
            {"boundary": "x_max", "region": {"y": (1.0, 1.5)}},
            {},
            r"meets each face it reaches in a set of no width",
            [{"boundary": "x_max", "region": {"y": (0.9, 1.5)}}],
        ),
        (
            {"region": {"x": (1.05, 1.2), "y": (0.3, 0.7)}},
            {"domain_bounds": np.array([[0.0, 1.2], [0.0, 1.0]])},
            r"its region lies where the BC's domain_bounds \[\[0\.0, 1\.2\], \[0\.0, 1\.0\]\] reach past the solver's box"
            r".* or place it on the solver's face by boundary=",
            [{"region": {"x": (1.0, 1.0), "y": (0.3, 0.7)}}, {"boundary": "x_max", "region": {"y": (0.3, 0.7)}}],
        ),
        (
            _DIAGONAL,
            {"domain_sdf": _UNIT_BOX_SDF},
            r"no face's outward normal lies within its normal_tolerance 0.9",
            [
                {**_DIAGONAL, "normal_tolerance": 0.7},
                {"normal_direction": np.array([1.0, 0.0]), "normal_tolerance": 0.9},
            ],
        ),
    ],
    ids=["an exit at a corner", "an exit past the face's end", "a box wider than the solver's", "a diagonal normal"],
)
def test_a_refusal_names_why_and_its_advice_is_admitted(refused, bc_kw, reason, advised):
    """Review 1 of #2782, M1, ruled option (i). Each of these absorbs particles at f2dcb164 only because `apply` reads
    a particle where its step ends, not where it crossed (#2733): at a corner, in a box wider than the solver's, at a
    normal read past a corner of the box's SDF. A particle crossing a face never meets them, so they are refused, each
    with its own reason; the advice it gives is admitted. The wider box's advice has two halves, run each alone (review
    2): the solver's box as the BC's domain_bounds with the exit on its face, or the exit on the solver's face by
    boundary= with the wider box kept; each absorbs every particle driven into it. Review 3: a segment its boundary=
    keeps off every face, with a region the wider box reaches past, is not given that reason; and a segment already
    placed on a face by boundary= is not advised to place itself by boundary=."""
    with pytest.raises(ValueError, match=reason) as refusal:
        _particle_solver([_seg("exit", D, **refused)], **bc_kw)
    if refused.get("boundary") not in (None, "all"):
        assert "or place it by boundary=" not in str(refusal.value)
    if "domain_bounds" in bc_kw:
        solver_box = {"domain_bounds": np.array([[0.0, 1.0], [0.0, 1.0]])}
        first, by_boundary = advised
        assert _particles_remaining_with([_seg("exit", D, **first)], **solver_box) == 0
        assert _particles_remaining_with([_seg("exit", D, **by_boundary)], **bc_kw) == 0
        # A segment held nowhere for another cause, under the same wider box, is not given this reason.
        for elsewhere in (
            {"boundary": "inlet", "region": {"x": (0.9, 1.1), "y": (0.3, 0.7)}},
            {"boundary": "y_min", "region": {"x": (0.9, 1.1), "y": (0.3, 0.7)}},
            {"boundary": "inlet", "region": {"x": (1.05, 1.2), "y": (0.3, 0.7)}},
            {"region": {"x": (1.3, 1.4), "y": (0.3, 0.7)}},
        ):
            with pytest.raises(ValueError, match=r"'exit': the placement holds none of these points") as other:
                _particle_solver([_seg("exit", D, **elsewhere)], **bc_kw)
            assert "reach past the solver's box" not in str(other.value)
        # On its named face's plane, past the solver's box along another axis, it is.
        with pytest.raises(ValueError, match=r"'exit': its region lies where the BC's domain_bounds"):
            _particle_solver(
                [_seg("exit", D, boundary="x_max", region={"y": (1.05, 1.2)})],
                domain_bounds=np.array([[0.0, 1.0], [0.0, 1.2]]),
            )
        return
    for placement in advised:
        _particle_solver([_seg("exit", D, **placement)], **bc_kw)


@pytest.mark.parametrize(
    ("own", "placement", "said"),
    [
        ([[0, 1.2], [0, 1]], {"boundary": "x_max", "region": {"x": (1.05, 1.2), "y": (0.3, 0.7)}}, "reach past"),
        ([[0, 0.8], [0, 1.2]], {"boundary": "x_max", "region": {"y": (1.05, 1.2)}}, "reach past"),
        ([[0, 1], [0, 1.2]], {"boundary": "x_max", "region": {"x": (0.9, 1.0), "y": (1.05, 1.2)}}, "reach past"),
        ([[0, 1.2], [0, 1]], {"boundary": "all", "region": {"x": (1.05, 1.2), "y": (0.3, 0.7)}}, "reach past"),
        (None, {"boundary": "all", "region": {"x": (1.0, 1.0), "y": (1.0, 1.0)}}, "or place it by boundary="),
    ],
    ids=["past the box across x_max", "on x_max's plane, read within a narrower box", "on x_max's plane", "all", "(a)"],
)
def test_reason_b_reads_the_face_boundary_names(own, placement, said):
    """Review 4 of #2782: (b) for a region past the solver's box across the face `boundary=` names, or on that face's
    plane (read within the BC's own box, here narrower on x) past the box along another axis; and for `boundary="all"`,
    as for no `boundary=`, which is offered (a)'s `boundary=` advice too. Read by the particle check itself."""
    bounds = None if own is None else np.array(own, dtype=float)
    bc = BoundaryConditions(segments=[_seg("exit", D, **placement)], dimension=2, default_bc=NF, domain_bounds=bounds)
    with pytest.raises(ValueError, match=r"'exit': ") as refusal:
        ParticleApplicator().refuse_segments_it_would_drop(bc, [(0.0, 1.0), (0.0, 1.0)])
    assert said in str(refusal.value)


def test_a_boundary_naming_an_axis_the_bc_box_lacks_is_refused_not_crashed():
    """Review 4 of #2782, S3: a BC with no dimension, whose box has two axes, and a segment named on a third axis' face
    is refused by the particle check with the generic cause, as at ea27ab8c, not an IndexError."""
    segment = _seg("exit", D, boundary="z_min", region={"y": (0.3, 0.7)})
    bc = BoundaryConditions(segments=[segment], default_bc=NF, domain_bounds=np.array([[0.0, 1.2], [0.0, 1.0]]))
    with pytest.raises(ValueError, match=r"'exit': the placement holds none of these points"):
        ParticleApplicator().refuse_segments_it_would_drop(bc, [(0.0, 1.0), (0.0, 1.0)])


def _particles_remaining_with(segments, **bc_kw):
    """`_particles_remaining`'s fixture, driving the particles across x_max at y in (0.43, 0.47), with ``bc_kw`` on the
    BC passed to the solver."""
    bc = BoundaryConditions(segments=segments, dimension=2, default_bc=NF, **bc_kw)
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=no_flux_bc(dimension=2)
    )
    solver = FPParticleSolver(_problem(grid), num_particles=200, seed=0, boundary_conditions=bc)
    solver.solve_fp_system(
        initial_particles=np.random.default_rng(0).uniform([0.8, 0.43], [0.95, 0.47], (200, 2)),
        drift_field=lambda t, x, m: np.broadcast_to(np.array([3.0, 0.0]), np.shape(x)),
        drift_needs_density=False,
        show_progress=False,
    )
    final = np.asarray(solver.M_particles_trajectory[-1], dtype=float).reshape(-1, 2)
    return int(np.count_nonzero(np.all(np.isfinite(final), axis=1)))


@pytest.mark.parametrize("gap", [1e-7, 1e-6, 1e-4])
def test_a_bc_box_within_the_solvers_by_a_hair_is_read_exactly(gap):
    """Review 2 of #2782, B-1: a BC whose domain_bounds sit inside the solver's box by ``gap`` keeps every segment's cuts,
    since a point is read within the BC's own box. An exit strip at its x_max, x = 1 - gap, over y = (0.42, 0.48), is
    admitted and absorbs as many particles driven across as at gap 0, as at f2dcb164. It was refused for gaps of about
    1e-8 to 1e-5, which `np.allclose` called equal: the face then had no cut at the strip's ends."""

    def remaining(g):
        exit_ = _seg("exit", D, region={"x": (1.0 - g, 1.0 - g), "y": (0.42, 0.48)})
        return _particles_remaining_with([exit_], domain_bounds=np.array([[0.0, 1.0 - g], [0.0, 1.0]]))

    at_zero = remaining(0.0)
    assert at_zero < 200, "no particle reached the exit, so the comparison is vacuous"
    assert remaining(gap) == at_zero


def test_a_region_key_is_read_by_one_owner():
    """Review 1 of #2782, B1 of the particle reader, and the audit session's ruling: every reader of a region reads its
    keys through `region_axis`. A numpy integer names its axis for all of them, as f2dcb164's particles read it; a float
    is refused by all of them, where f2dcb164's point reader truncated it with int()."""
    exit_np = [_seg("exit", D, region={np.int64(0): (1.0, 1.0), np.int64(1): (0.35, 0.65)})]
    assert _particles_remaining(exit_np) == 0
    floating = _seg("exit", D, region={"x": (1.0, 1.0), 1.0: (0.35, 0.65)})
    refusal = r"region axis 1\.0 names no axis of a 2-D domain"
    bc = BoundaryConditions(
        segments=[floating], dimension=2, default_bc=NF, domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]])
    )
    with pytest.raises(ValueError, match=refusal):
        _particle_solver([floating])
    with pytest.raises(ValueError, match=refusal):
        bc.segment_for_point(np.array([1.0, 0.5]), "x_max")
    with pytest.raises(ValueError, match=refusal):
        bc.segment_for_face("x_max")
    # Review 2: the validator reads the keys through the owner too, and a bool is not an index.
    assert any("region axis 1.0 names no axis" in warning for warning in bc.validate()[1])
    numpy_bc = BoundaryConditions(
        segments=exit_np, dimension=2, default_bc=NF, domain_bounds=np.array([[0.0, 1.0]] * 2)
    )
    assert not any("region axis" in warning for warning in numpy_bc.validate()[1])
    for flag in (True, np.bool_(True)):
        with pytest.raises(ValueError, match=r"names no axis of a 2-D domain"):
            _particle_solver([_seg("exit", D, region={"x": (1.0, 1.0), flag: (0.35, 0.65)})])


@pytest.mark.parametrize("dimension", range(2, 9))
def test_the_lattice_keeps_its_cell_budget_in_every_dimension(dimension):
    """Review 1 of #2782, M2: the lattice cuts each face into at most FACE_LATTICE_CELLS cells, n equal parts along each
    of its d - 1 axes. Rounding n gave 1296 in 5-D and 2187 in 8-D."""
    from mfgarchon.geometry.boundary.conditions import FACE_LATTICE_CELLS

    bc = BoundaryConditions(
        segments=[_seg("exit", D, sdf_region=lambda p: float(np.linalg.norm(np.asarray(p) - 1.0) - 0.1))],
        dimension=dimension,
        default_bc=NF,
    )
    centres, n = bc.face_cell_centres(np.array([[0.0, 1.0]] * dimension))
    assert n ** (dimension - 1) <= FACE_LATTICE_CELLS < (n + 1) ** (dimension - 1)
    assert len(centres) == 2 * dimension * n ** (dimension - 1)


def test_the_particle_sameness_reads_the_action_and_a_flux_capacity():
    """Two segments are the same to a particle when they take one action and neither carries a flux_capacity: the
    flux-limited path counts a capped exit's absorptions under its own name."""
    absorbing, reflecting = _seg("a", D, 1.0), _seg("b", NF)
    assert same_particle_action(absorbing, _seg("c", D, 2.0))
    assert same_particle_action(reflecting, _seg("d", BCType.NEUMANN))
    assert not same_particle_action(absorbing, reflecting)
    assert not same_particle_action(absorbing, _seg("e", D, 1.0, flux_capacity=5.0))

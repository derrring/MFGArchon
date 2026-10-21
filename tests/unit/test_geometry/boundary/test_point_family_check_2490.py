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


def _gfdm(segments, points=None) -> HJBGFDMSolver:
    """HJB-GFDM on the 11 x 11 grid's nodes, or on ``points``, with the BC set on the grid: a NO_FLUX default beside
    ``segments``."""
    bc = BoundaryConditions(segments=segments, dimension=2, default_bc=NF)
    grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=bc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HJBGFDMSolver(
            _problem(grid, nt=4),
            collocation_points=grid.get_spatial_grid() if points is None else points,
            delta=0.25,
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
    its priority, which could not help."""
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
    assert "Raise the segment's priority" not in str(refused.value)


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
            r"the BC's domain_bounds \[\[0\.0, 1\.2\], \[0\.0, 1\.0\]\] are not the solver's box",
            [{"region": {"x": (1.0, 1.0), "y": (0.3, 0.7)}}],
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
    with its own reason; the advice it gives is admitted. The advice for the wider box is run with the solver's box as
    the BC's domain_bounds."""
    with pytest.raises(ValueError, match=reason):
        _particle_solver([_seg("exit", D, **refused)], **bc_kw)
    if "domain_bounds" in bc_kw:
        bc_kw = {"domain_bounds": np.array([[0.0, 1.0], [0.0, 1.0]])}
    for placement in advised:
        _particle_solver([_seg("exit", D, **placement)], **bc_kw)


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

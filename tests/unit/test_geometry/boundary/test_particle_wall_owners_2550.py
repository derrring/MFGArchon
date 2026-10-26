"""A reader that has only where a step ended acts per axis at a corner one of whose faces is periodic, and absorbs
through the absorbing wall's owner (#2550, C6-1; #2803's shape).

`ParticleApplicator.apply` and ``apply_with_flux_limits`` gave a particle at such a corner one action for every axis:
the corner's segment, chosen by the owner's corner order, wrapped every axis or folded every axis, so a channel corner
wrapped across its wall or folded across its seam. They now read the end-position rule (CONVENTIONS.md § *Particles at
a wall*): the segment the end's nearest point of the box takes decides only whether the particle is absorbed, and
otherwise each face the end is at or past acts on its own axis by its own segment, a periodic face wrapping its axis
and any other folding it. An absorbed particle is removed and never moved, and its exit position is its end
(`corner.absorb_at_walls`, which `MeshfreeApplicator`'s absorbing mode reads too).

**The oracle.** On a box the fold and the wrap act on each axis alone: past the max face of the unit square an axis
folds to ``2 - x`` and wraps to ``x - 1``, and ``k`` spans past it wraps to ``x - k``. Which segment decides absorption
is the owner's corner order: priority, then DIRICHLET before a natural condition.

**Rows.** A channel corner with the periodic face winning the corner and with the wall winning; a wrap of two spans
beside a fold; an exit outranked by a periodic face, and one that wins on DIRICHLET-first; a 3-D vertex of a wall, an
outranked exit and a periodic face; a capacity-limited exit at a periodic corner, open and then full. The absorbing
owner at the edges of its band, `MeshfreeApplicator`'s absorbing mode on a box and on a disc, and both readers' refusal
of a non-finite position.

Held fixed: the unit box; constant values; every face placed by its label. The fuzz over BC kinds, priorities, global
and default segments, bounds up to 1e5 and ends at each band's edges is the PR's, in scratch.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, MeshfreeApplicator
from mfgarchon.geometry.boundary.applicator_particle import ParticleApplicator
from mfgarchon.geometry.boundary.corner import absorb_at_walls, absorb_positions, walls_reached
from mfgarchon.geometry.implicit import Hyperrectangle, Hypersphere

P, D, NF = BCType.PERIODIC, BCType.DIRICHLET, BCType.NO_FLUX
FLUX_WARNING = "apply_with_flux_limits is deprecated"


def _bc(d: int, faces: dict[str, tuple[BCType, int]]) -> BoundaryConditions:
    """One segment per face, named by its face, with its type and priority."""
    return BoundaryConditions(
        dimension=d,
        segments=[
            BCSegment(
                name=face, bc_type=bc_type, boundary=face, priority=priority, **({"value": 0.0} if bc_type == D else {})
            )
            for face, (bc_type, priority) in faces.items()
        ],
    )


def _step(method: str, bc: BoundaryConditions, ends, flux_limits=None):
    applicator = ParticleApplicator()
    ends = np.atleast_2d(np.asarray(ends, dtype=float))
    bounds = [(0.0, 1.0)] * ends.shape[1]
    if method == "apply":
        return (*applicator.apply(ends, bc, bounds), None)
    with pytest.warns(DeprecationWarning, match=FLUX_WARNING):
        return applicator.apply_with_flux_limits(ends, bc, bounds, flux_limits or {})


CHANNEL = {"x_min": (P, 1), "x_max": (P, 1), "y_min": (NF, 0), "y_max": (NF, 0)}
CHANNEL_WALLS_WIN = {"x_min": (P, 0), "x_max": (P, 0), "y_min": (NF, 1), "y_max": (NF, 1)}
EXIT_BESIDE_PERIODIC_Y = {"x_min": (NF, 0), "x_max": (D, 0), "y_min": (P, 1), "y_max": (P, 1)}
# (name, d, faces, end, expected end or None where absorbed)
ROWS = [
    ("channel, the periodic face wins the corner", 2, CHANNEL, (1.05, 1.03), (0.05, 0.97)),
    ("channel, the wall wins the corner", 2, CHANNEL_WALLS_WIN, (1.05, 1.03), (0.05, 0.97)),
    ("a wrap of two spans beside a fold", 2, CHANNEL, (2.35, -0.03), (0.35, 0.03)),
    ("an exit outranked by the periodic face", 2, EXIT_BESIDE_PERIODIC_Y, (1.04, 1.02), (0.96, 0.02)),
    (
        "an exit at the periodic face's priority: DIRICHLET first",
        2,
        {**EXIT_BESIDE_PERIODIC_Y, "y_min": (P, 0), "y_max": (P, 0)},
        (1.04, 1.02),
        None,
    ),
    (
        "3-D vertex: a wall, an outranked exit, a periodic face",
        3,
        {"x_min": (NF, 0), "x_max": (NF, 0), "y_min": (NF, 0), "y_max": (D, 0), "z_min": (P, 1), "z_max": (P, 1)},
        (1.02, 1.03, 1.04),
        (0.98, 0.97, 0.04),
    ),
]


@pytest.mark.parametrize("method", ["apply", "apply_with_flux_limits"])
@pytest.mark.parametrize(("name", "d", "faces", "end", "expected"), ROWS, ids=[row[0] for row in ROWS])
def test_a_periodic_corner_acts_per_axis(method, name, d, faces, end, expected):
    """Each face the end is past acts on its own axis; the corner's segment decides only absorption."""
    remaining, absorbed, exits, _ = _step(method, _bc(d, faces), end)
    if expected is None:
        assert absorbed.tolist() == [True], f"{name}: kept at {remaining}"
        np.testing.assert_array_equal(exits, [end])
    else:
        assert absorbed.tolist() == [False], f"{name}: absorbed"
        np.testing.assert_allclose(remaining[0], expected, rtol=0, atol=1e-12, err_msg=name)


def test_a_full_exit_turns_a_particle_back_per_axis():
    """A capped exit at a periodic corner, at the periodic faces' priority: the first particle is absorbed
    (DIRICHLET first), and the second, with the capacity spent, is not, so each face it is past acts on its own axis --
    the exit's axis folds and the periodic axis wraps."""
    bc = _bc(2, {**EXIT_BESIDE_PERIODIC_Y, "y_min": (P, 0), "y_max": (P, 0)})
    remaining, absorbed, exits, counts = _step("flux", bc, [(1.04, 1.02), (1.03, 1.01)], {"x_max": 1})
    assert absorbed.tolist() == [True, False]
    np.testing.assert_array_equal(exits, [(1.04, 1.02)])
    np.testing.assert_allclose(remaining, [(0.97, 0.01)], rtol=0, atol=1e-12)
    assert counts == {"x_max": 1}


def test_the_absorbing_owner_removes_never_moves_and_reads_its_band():
    """`absorb_at_walls`: a row at or past a face, within the band, leaves, and only where ``leaves`` says its wall
    absorbs; its exit position is its end; nothing is written. `walls_reached` names the face per axis."""
    band = 1e-10
    rows = np.array(
        [
            (0.5, 0.5),  # deep
            (1.0, 0.5),  # on the x_max face
            (1.0 - 0.5 * band, 0.5),  # inside the band
            (1.0 - 2.0 * band, 0.5),  # just outside it
            (1.2, -0.3),  # past two faces: its exit position is the end, not a point of the wall
            (0.0, 0.5),  # on the x_min face, where the reader's wall does not absorb
        ]
    )
    given = rows.copy()
    leaves = np.array([True, True, True, True, True, False])
    np.testing.assert_array_equal(
        walls_reached(rows, [(0.0, 1.0), (0.0, 1.0)], band), [[-1, -1], [1, -1], [1, -1], [-1, -1], [1, 0], [0, -1]]
    )
    leave, exits = absorb_at_walls(rows, [(0.0, 1.0), (0.0, 1.0)], band, leaves)
    assert leave.tolist() == [False, True, True, False, True, False]
    np.testing.assert_array_equal(exits, given[[1, 2, 4]])
    np.testing.assert_array_equal(rows, given)
    exits[:] = -1.0
    np.testing.assert_array_equal(rows, given)


@pytest.mark.parametrize(
    ("geometry", "rows", "kept"),
    [
        (
            Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]])),
            [(0.5, 0.5), (1.0, 0.5), (1.0 - 5e-9, 0.5), (1.0 - 2e-8, 0.5), (1.3, 0.5)],
            [(0.5, 0.5), (1.0 - 2e-8, 0.5)],
        ),
        (
            # (1.0, 0.8) is on the disc's box face and off the disc: on or past the box, it leaves. Before C6-1 it was
            # kept, inside the box's closed bounds and off the boundary.
            Hypersphere(center=np.array([0.5, 0.5]), radius=0.5),
            [(0.5, 0.5), (1.0, 0.5), (1.0, 0.8), (0.95, 0.95)],
            [(0.5, 0.5), (0.95, 0.95)],
        ),
    ],
    ids=["box", "disc"],
)
def test_meshfree_absorbing_reads_the_owner(geometry, rows, kept):
    """`MeshfreeApplicator`'s absorbing mode removes a particle on or past its geometry's box, through the owner with no
    band, or on the geometry's boundary at the geometry's band (1e-8 for these two). On the disc a particle outside it
    and inside its box is kept, (0.95, 0.95): #2693's shape in this mode, not this PR's (#2573)."""
    np.testing.assert_array_equal(MeshfreeApplicator(geometry).apply_particle_bc(np.array(rows), "absorbing"), kept)


def test_the_two_deprecations_warn_and_keep_their_behaviour():
    """`absorb_positions` is a clamp and stays one while it warns; `apply_with_flux_limits` with no limit is `apply`
    (the deprecation policy's equivalence test)."""
    with pytest.warns(DeprecationWarning, match="absorb_positions is deprecated"):
        clamped = absorb_positions(np.array([[-0.1, 0.5], [1.2, 0.3]]), [(0.0, 1.0), (0.0, 1.0)])
    np.testing.assert_array_equal(clamped, [[0.0, 0.5], [1.0, 0.3]])
    bc = _bc(2, EXIT_BESIDE_PERIODIC_Y)
    ends = [(1.04, 1.02), (0.5, -0.2), (1.01, 0.5), (0.3, 0.4)]
    for old, new in zip(_step("flux", bc, ends)[:3], _step("apply", bc, ends)[:3], strict=True):
        np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
def test_both_end_readers_refuse_a_non_finite_position(value):
    """A non-finite position is at no wall and past none: `ParticleApplicator` kept it, a NaN unmoved and an infinite
    coordinate folded or wrapped into NaN, and `MeshfreeApplicator`'s absorbing mode dropped it as though absorbed. The
    owner's reach now refuses it by name for both, naming the row, as the particle solver's path refuses it."""
    rows = np.array([[0.5, 0.5], [value, 0.5], [0.4, value]])
    match = r"row 1 of the particle positions is not finite"
    with pytest.raises(ValueError, match=match):
        ParticleApplicator().apply(rows, _bc(2, CHANNEL), [(0.0, 1.0), (0.0, 1.0)])
    with pytest.raises(ValueError, match=match):
        MeshfreeApplicator(Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]]))).apply_particle_bc(rows, "absorbing")

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

**C6-2, the 1-D helpers** (`utils/particle_utils`). The NumPy fallback folds and wraps through the owners, and the GPU
one, the particle solver's 1-D device path, is pinned against them here, at a nonzero lower bound whose span rounds,
several spans past and on a degenerate span. Both call the owners' per-axis fold and wrap, which keep a result on the
box where rounding took it off, at either wall and in float32 (#2830). Both refuse "dirichlet" (a function that returns
positions cannot remove a particle; it clamped) and a non-finite position. The deprecated pair in `utils/numerical/particle/
boundary.py` is that pair, and the export `geometry.boundary.wrap_positions` is the periodic owner, with the removed
corner shim's values on its signature.

**C6-3, the owners' other consumers.** `MeshfreeApplicator`'s reflecting and periodic modes, and `Hyperrectangle`'s
and `TensorProductGrid`'s ``wrap_coordinates`` (the grid's reads the periodic owner now, where it had its own
``np.mod``), against the closed form on a [0, 1] x [0, 2] box, one set of rows for all of them: inside, on the max
corner and on the min corner, past one face, past a corner, and several spans past the max face on one axis and the
min face on the other. Each ``wrap_coordinates`` leaves the caller's array as it was. `sample_from_scattered_density`'s
periodic wrap is read against the closed-form wrap of the samples it hands the owner, on the axes it names. C6's
structural guard is `test_particle_wall_has_one_owner_2550.py`.

Held fixed: the unit box (the C6-3 rows aside); constant values; every face placed by its label. The fuzzes over BC
kinds, priorities, global and default segments, bounds up to 1e5 and ends at each band's edges, over the 1-D helpers,
and over ``TensorProductGrid.wrap_coordinates``, are the PRs', in scratch.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    MeshfreeApplicator,
    corner,
    no_flux_bc,
    periodic,
    periodic_bc,
)
from mfgarchon.geometry.boundary.applicator_particle import ParticleApplicator
from mfgarchon.geometry.boundary.corner import absorb_at_walls, absorb_positions, reflect_positions, walls_reached
from mfgarchon.geometry.boundary.periodic import wrap_positions
from mfgarchon.geometry.implicit import Hyperrectangle, Hypersphere
from mfgarchon.utils.numerical.particle import boundary as deprecated_pair
from mfgarchon.utils.numerical.particle.sampling import sample_from_scattered_density
from mfgarchon.utils.particle_utils import apply_boundary_conditions_gpu, apply_boundary_conditions_numpy

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
    """No wall rule can place a non-finite position, since its fold, wrap and exit are undefined. `ParticleApplicator`
    kept it -- a NaN unmoved, an infinite coordinate folded or wrapped into NaN -- or counted it as an exit past an
    absorbing face, and `MeshfreeApplicator`'s absorbing mode dropped it as though absorbed. The owner's reach now
    refuses it by name for both, naming the row, as the particle solver's path refuses it."""
    rows = np.array([[0.5, 0.5], [value, 0.5], [0.4, value]])
    match = r"row 1 of the particle positions is not finite"
    with pytest.raises(ValueError, match=match):
        ParticleApplicator().apply(rows, _bc(2, CHANNEL), [(0.0, 1.0), (0.0, 1.0)])
    with pytest.raises(ValueError, match=match):
        MeshfreeApplicator(Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]]))).apply_particle_bc(rows, "absorbing")


LO, HI = 2.0, 2.3  # a nonzero lower bound whose span rounds: HI - LO is 0.2999999999999998
ENDS_1D = np.array([2.1, LO, HI, HI + 1e-12, LO - 1e-12, 2.45, 1.85, LO + 2.6 * 0.3, LO - 3.4 * 0.3])


def _helper(reader: str):
    """A 1-D helper as a NumPy function: the NumPy fallback, or the GPU one on the torch backend (CPU)."""
    if reader == "numpy":
        return apply_boundary_conditions_numpy
    pytest.importorskip("torch")
    from mfgarchon.backends.torch_backend import TorchBackend

    backend = TorchBackend(device="cpu")
    return lambda x, lo, hi, kind: backend.to_numpy(
        apply_boundary_conditions_gpu(backend.from_numpy(x), lo, hi, kind, backend)
    )


@pytest.mark.parametrize("kind", ["periodic", "no_flux"])
@pytest.mark.parametrize("reader", ["numpy", "torch"])
def test_the_1d_helpers_fold_and_wrap_as_the_owners_do(reader, kind):
    """The fallback reads the owners; the GPU one's device copy gives their values, to rounding. An axis with no span is
    left as it is, as the owners leave it (the GPU wrap returned NaN there)."""
    help_, owner = _helper(reader), wrap_positions if kind == "periodic" else reflect_positions
    given = ENDS_1D.copy()
    np.testing.assert_allclose(
        help_(given, LO, HI, kind), owner(ENDS_1D[:, None], [(LO, HI)])[:, 0], rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(given, ENDS_1D)
    np.testing.assert_array_equal(help_(np.array([2.0, 2.5]), 2.0, 2.0, kind), [2.0, 2.5])


def _just_below(value, dtype, count):
    """The ``count`` values just below ``value`` as ``dtype`` holds it."""
    out = [dtype(value)]
    for _ in range(count):
        out.append(np.nextafter(out[-1], dtype(-np.inf)))
    return np.array(out[1:], dtype=dtype)


@pytest.mark.parametrize(
    ("reader", "kind", "lo", "hi", "ends"),
    [
        ("the GPU helper, float32", "no_flux", 0.1, 1.1, _just_below(0.1, np.float32, 40)),
        ("the owner", "no_flux", -0.3, 0.1, np.array([0.1])),
        ("the owner", "periodic", -0.9, 0.2, _just_below(-0.9, np.float64, 1)),
        ("the GPU helper, float32", "periodic", 0.1, 1.1, _just_below(0.1, np.float32, 40)),
    ],
    ids=["fold, GPU float32, lower wall", "fold, owner, upper wall", "wrap, owner, upper wall", "wrap, GPU float32"],
)
def test_the_fold_and_the_wrap_stay_on_the_box(reader, kind, lo, hi, ends):
    """The fold lands in [lo, hi] and the wrap in [lo, hi), as the array's dtype holds them, on inputs where rounding
    took main's arithmetic off the box (#2830): the GPU fold of the float32 positions just below 0.1 on [0.1, 1.1]
    landed 9.5e-8 below 0.1, past the one float32 spacing within which a reader takes a position to be on the wall; the
    owner's fold of 0.1 on [-0.3, 0.1] gave 0.10000000000000003, since fl(0.1 + 0.3) rounds above the span; and the
    owner's wrap of the float64 just below -0.9 on [-0.9, 0.2] gave 0.20000000000000007; and the GPU wrap of the float32
    positions just below 0.1 on [0.1, 1.1] gave float32(1.1) for 4 of them."""
    if reader == "the owner":
        owner = reflect_positions if kind == "no_flux" else wrap_positions
        got = owner(ends[:, None], [(lo, hi)])[:, 0]
    else:
        pytest.importorskip("torch")
        from mfgarchon.backends.torch_backend import TorchBackend

        backend = TorchBackend(device="cpu", precision="float32")
        got = np.asarray(
            backend.to_numpy(apply_boundary_conditions_gpu(backend.from_numpy(ends), lo, hi, kind, backend))
        )
    dtype = got.dtype.type
    assert dtype is ends.dtype.type
    assert np.all(got >= dtype(lo)), got
    assert np.all(got <= dtype(hi) if kind == "no_flux" else got < dtype(hi)), got


@pytest.mark.parametrize(
    ("ends", "kind", "match"),
    [
        (np.array([0.2, 1.4]), "dirichlet", "bc_type 'dirichlet' is refused"),
        (np.array([0.2, np.nan, 0.4]), "no_flux", "row 1 of the particle positions is not finite"),
        (np.array([0.2, -np.inf]), "periodic", "row 1 of the particle positions is not finite"),
    ],
    ids=["dirichlet", "nan", "-inf"],
)
@pytest.mark.parametrize("reader", ["numpy", "torch"])
def test_the_1d_helpers_refuse_dirichlet_and_a_non_finite_position(reader, ends, kind, match):
    """A function that returns positions cannot remove a particle, and the clamp it applied for "dirichlet" kept the mass
    that should leave (#2485); a non-finite position has no fold or wrap, and the owner's reach refuses it (#2550)."""
    with pytest.raises(ValueError, match=match):
        _helper(reader)(ends, 0.0, 1.0, kind)


def test_the_deprecated_pair_is_particle_utils():
    """`utils/numerical/particle/boundary.py`'s pair warns and gives `particle_utils`' values, its refusal of "dirichlet"
    included (the deprecation policy's equivalence test)."""
    for kind in ("periodic", "no_flux"):
        with pytest.warns(DeprecationWarning, match="apply_boundary_conditions_numpy is deprecated"):
            old = deprecated_pair.apply_boundary_conditions_numpy(ENDS_1D, LO, HI, kind)
        np.testing.assert_array_equal(old, apply_boundary_conditions_numpy(ENDS_1D, LO, HI, kind))
    with (
        pytest.warns(DeprecationWarning, match="apply_boundary_conditions_numpy is deprecated"),
        pytest.raises(ValueError, match="bc_type 'dirichlet' is refused"),
    ):
        deprecated_pair.apply_boundary_conditions_numpy(ENDS_1D, LO, HI, "dirichlet")


def test_the_exported_wrap_positions_is_the_periodic_owner():
    """`geometry.boundary.wrap_positions` resolved to the deprecated `corner.wrap_positions`, which is removed; it is
    the periodic owner now, and on the shim's own signature (positions, bounds) it gives the shim's values."""
    from mfgarchon.geometry.boundary import wrap_positions as exported

    assert exported is wrap_positions
    assert not hasattr(corner, "wrap_positions")
    np.testing.assert_allclose(
        exported(np.array([[2.35, -0.2], [1.95, 1.3]]), [(LO, HI), (0.0, 1.0)]),
        [[2.05, 0.8], [2.25, 0.3]],
        rtol=0,
        atol=1e-12,
    )


def test_the_gpu_helper_refuses_with_the_owners_message():
    """The GPU helper's device check is a second statement of the owner's refusal, so it builds the owner's message
    (`corner.non_finite_refusal`): the two readers raise one refusal for one row, and cannot drift apart."""
    messages = []
    for reader in ("numpy", "torch"):
        with pytest.raises(ValueError) as caught:
            _helper(reader)(np.array([0.2, np.nan, 0.4]), 0.0, 1.0, "no_flux")
        messages.append(str(caught.value))
    assert messages[0] == messages[1], messages


def _torch_backend():
    pytest.importorskip("torch")
    from mfgarchon.backends.torch_backend import TorchBackend

    return TorchBackend(device="cpu")


def test_the_deprecated_gpu_helper_is_particle_utils():
    """The deprecated pair's GPU half warns and gives `particle_utils`' values on the torch backend, its refusal of
    "dirichlet" included (the deprecation policy's equivalence test, the half that runs on a device)."""
    backend = _torch_backend()
    for kind in ("periodic", "no_flux"):
        with pytest.warns(DeprecationWarning, match="apply_boundary_conditions_gpu is deprecated"):
            old = deprecated_pair.apply_boundary_conditions_gpu(backend.from_numpy(ENDS_1D), LO, HI, kind, backend)
        new = apply_boundary_conditions_gpu(backend.from_numpy(ENDS_1D), LO, HI, kind, backend)
        np.testing.assert_array_equal(backend.to_numpy(old), backend.to_numpy(new))
    with (
        pytest.warns(DeprecationWarning, match="apply_boundary_conditions_gpu is deprecated"),
        pytest.raises(ValueError, match="bc_type 'dirichlet' is refused"),
    ):
        deprecated_pair.apply_boundary_conditions_gpu(backend.from_numpy(ENDS_1D), LO, HI, "dirichlet", backend)


def test_the_gpu_helper_does_not_write_its_input():
    """The device copy returns new positions and leaves the tensor it was handed as it was."""
    torch = pytest.importorskip("torch")
    backend = _torch_backend()
    for kind in ("periodic", "no_flux"):
        given = backend.from_numpy(ENDS_1D)
        before = given.clone()
        apply_boundary_conditions_gpu(given, LO, HI, kind, backend)
        assert torch.equal(given, before), kind


def test_the_numpy_helper_reads_an_integer_input_as_float():
    """A non-float input is read as float64. The owners write into a copy of what they are handed, so an integer array
    would have its wrapped positions truncated: (-1, 0, 3, 7) on (0, 2.5) would give (1, 0, 0, 2)."""
    np.testing.assert_array_equal(
        apply_boundary_conditions_numpy(np.array([-1, 0, 3, 7]), 0.0, 2.5, "periodic"), [1.5, 0.0, 0.5, 2.0]
    )


# C6-3. A [0, 1] x [0, 2] box, so an axis read with the other's bounds shows. Each value is a binary fraction, so the
# closed forms are exact: past a max face by d an axis folds to hi - d and wraps to lo + d; k spans past, the fold is
# the triangle wave of period 2L and the wrap is x - kL.
BOX = Hyperrectangle(np.array([[0.0, 1.0], [0.0, 2.0]]))
OFF_THE_BOX = np.array([[0.25, 0.5], [1.0, 2.0], [0.0, 0.0], [1.25, 0.5], [-0.25, 2.5], [3.25, -4.5]])
FOLDED = [[0.25, 0.5], [1.0, 2.0], [0.0, 0.0], [0.75, 0.5], [0.25, 1.5], [0.75, 0.5]]
WRAPPED = [[0.25, 0.5], [0.0, 0.0], [0.0, 0.0], [0.25, 0.5], [0.75, 0.5], [0.25, 1.5]]


@pytest.mark.parametrize(("mode", "expected"), [("reflecting", FOLDED), ("periodic", WRAPPED)])
def test_meshfree_reflecting_and_periodic_read_the_owners(mode, expected):
    """`MeshfreeApplicator` folds through `corner.reflect_positions` and wraps through `periodic.wrap_positions` on a
    box: inside, on the max and the min corner, past a face, past a corner, and several spans past both faces."""
    np.testing.assert_array_equal(MeshfreeApplicator(BOX).apply_particle_bc(OFF_THE_BOX, mode), expected)


def test_wrap_coordinates_wraps_only_the_periodic_axes():
    """`Hyperrectangle.wrap_coordinates` wraps its periodic axis onto [lo, hi), so a point on the max face comes back on
    the min face, and leaves the other axis as it is, inside the box or not. A (d,) point comes back as (d,), and the
    caller's array is not written."""
    torus_x = Hyperrectangle(np.array([[0.0, 1.0], [0.0, 2.0]]), periodic_dims=(0,))
    given = OFF_THE_BOX.copy()
    np.testing.assert_array_equal(
        torus_x.wrap_coordinates(given), np.column_stack([np.array(WRAPPED)[:, 0], given[:, 1]])
    )
    np.testing.assert_array_equal(torus_x.wrap_coordinates(given[3]), [0.25, 0.5])
    np.testing.assert_array_equal(given, OFF_THE_BOX)


def test_the_grids_wrap_coordinates_is_the_periodic_owner():
    """`TensorProductGrid.wrap_coordinates` reads `periodic.wrap_positions` over its bounds and periodic axes. On a
    periodic grid every axis wraps onto [lo, hi) and a (d,) point stays (d,); on a no-flux grid nothing moves; the
    caller's array is not written. It had its own
    ``np.mod`` in the points' dtype, and the two rows after the closed forms are where that differed from the owner:
    - a float32 point against bounds given as Python floats is wrapped in float64 and rounded once, to the exact wrap
      of 2.2 on (0, 0.3) rounded to float32. The grid's own float32 arithmetic gave 0.099999964, 11 ulps below;
    - an axis whose span is at most 1e-14, which a grid accepts, is left as it is, the owner's rule for an axis with
      no span. The grid's own body wrapped it (3e-15 on (0, 1e-15) to 1e-15)."""
    bounds = [(0.0, 1.0), (0.0, 2.0)]
    given = OFF_THE_BOX.copy()
    grid = TensorProductGrid(bounds=bounds, Nx=[4, 4], boundary_conditions=periodic_bc(dimension=2))
    np.testing.assert_array_equal(grid.wrap_coordinates(given), WRAPPED)
    np.testing.assert_array_equal(grid.wrap_coordinates(given[4]), [0.75, 0.5])
    walls = TensorProductGrid(bounds=bounds, Nx=[4, 4], boundary_conditions=no_flux_bc(dimension=2))
    np.testing.assert_array_equal(walls.wrap_coordinates(given), OFF_THE_BOX)
    np.testing.assert_array_equal(given, OFF_THE_BOX)

    narrow = TensorProductGrid(bounds=[(0.0, 0.3), (0.0, 2.0)], Nx=[4, 4], boundary_conditions=periodic_bc(dimension=2))
    wrapped = narrow.wrap_coordinates(np.array([[2.2, 0.5]], dtype=np.float32))
    assert wrapped.dtype == np.float32
    assert wrapped[0, 0] == np.float32(0.1000000476837159), wrapped
    no_span = TensorProductGrid(bounds=[(0.0, 1e-15)], Nx=[4], boundary_conditions=periodic_bc(dimension=1))
    np.testing.assert_array_equal(no_span.wrap_coordinates(np.array([3e-15])), [3e-15])


def test_scattered_density_sampling_wraps_through_the_periodic_owner(monkeypatch):
    """`sample_from_scattered_density` with ``periodic_bounds`` samples triangles across the seam, so some samples land
    outside the box; it wraps them once, through the periodic owner. The result is the closed-form wrap of the samples
    the owner was handed, on the periodic axis only, which the owner is told; a density near both x faces puts samples
    past each."""
    handed = []
    owner = periodic.wrap_positions

    def recording(positions, bounds, periodic_dims=None):
        handed.append((np.array(positions, copy=True), periodic_dims))
        return owner(positions, bounds, periodic_dims)

    monkeypatch.setattr(periodic, "wrap_positions", recording)
    points = np.random.default_rng(0).random((80, 2)) * [1.0, 2.0]
    density = np.exp(-20 * np.minimum(points[:, 0], 1 - points[:, 0]) ** 2)
    sampled = sample_from_scattered_density(
        points, density, 400, seed=1, periodic_bounds=[(0.0, 1.0), (0.0, 2.0)], periodic_dims=(0,)
    )
    assert len(handed) == 1, len(handed)
    raw, dims = handed[0]
    assert dims is not None, "the owner was told no periodic axes, so it wrapped every axis"
    assert tuple(dims) == (0,), dims
    assert (raw[:, 0] < 0).any(), "no sample past the min face: the wrap had nothing to do there"
    assert (raw[:, 0] >= 1).any(), "no sample past the max face: the wrap had nothing to do there"
    np.testing.assert_array_equal(sampled, np.column_stack([np.mod(raw[:, 0], 1.0), raw[:, 1]]))

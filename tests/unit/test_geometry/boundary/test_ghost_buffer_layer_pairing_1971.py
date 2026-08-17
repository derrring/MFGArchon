"""`GhostBuffer` paired its ghost and interior slices in opposite directions. #1971

The copy of the #1967 pairing defect that lives in `GhostBuffer._update_bounded`, and the only one
of the three *mirror* copies that reverses **both** walls. It differs in kind from the two fixed in
#1969: those were per-layer loops with a wrong index expression, this one is a vectorised slice
assignment where the two slices run opposite ways.

    lo_ghost    = [0:g]        index g-1 is adjacent to the wall, index 0 outermost
    lo_interior = [g:2g]       index g   is adjacent to the wall
    buf[lo_ghost] = calc(buf[lo_interior])      pairs outermost ghost with nearest interior

The high side has the mirror problem: its ghost slice runs nearest-first and its interior slice
`[-2g:-g]` runs farthest-first.

**At `ghost_depth = 1` a one-element slice is its own reverse**, which is why the route looks
correct at the only depth anything uses.

**This does not close the family.** The same slice-pair construct is written out twice more in
`PreallocatedGhostBuffer` -- `_apply_linear_reflection`'s DIRICHLET branch and the per-face
`_apply_ghost_for_face` -- and both are reversed at both walls (`reversed(got) == want`, magnitudes
1.1 to 3.4 at g = 2..4). Those are #1966 Defect 2 and are not fixed here. `_update_ghosts_legacy`
carries the construct as well. The claim this file used to make -- "third and last copy" -- was
wrong, and it was wrong in the same way #1969's "the loop has two copies" was.

The fix steps the *interior* slice backwards rather than reversing the calculator's output. The
output would have to be reversed along `axis`; the obvious `[::-1]` reverses axis 0, which is
correct in 1-D and wrong for any other axis in n-D. (`np.flip(out, axis=axis)` would also be
correct -- the interior-slice form is a preference, not the only option.) The `[::-1]` version was
written first and passes **every** 1-D assertion below.

## The oracle, and why the fixture is random

A ghost rule for a mirror wall is a permutation fixed by the grid geometry alone: on a
cell-centred grid, ghost layer `k` sits at `-(k-1/2)h` and its mirror is interior layer `k` at
`+(k-1/2)h`. That is derived from the geometry, not from another code path, so it survives any
later consolidation of these copies into one owner.

The interior values are **unconstrained** by that -- evenness about a wall relates ghosts to
interior samples and says nothing about the interior itself -- so the fixture may be anything.
It must not be `cos(2*pi*x)` on a symmetric grid, which is what this file used to use: that field
is palindromic on the interior, so a high-wall/low-end confusion is value-identical on it and
passed 20 of 21 assertions here and 1180 of 1181 in `tests/unit/test_geometry/`. Seeded random
data has no symmetry to hide behind.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary import uniform_bc
from mfgarchon.geometry.boundary.applicator_fdm import create_ghost_buffer_from_bc

_SEED = 19710816


def _interior(shape) -> np.ndarray:
    """Asymmetric, non-palindromic, no repeated values -- every slot is its own witness."""
    return np.random.default_rng(_SEED).standard_normal(shape)


def _bc(bc_type, dimension, bounds, value=0.0):
    bc = uniform_bc(bc_type=bc_type, value=value, dimension=dimension)
    bc.domain_bounds = np.asarray(bounds, dtype=float)
    return bc


def _filled(bc, shape, dx, ghost_depth, interior):
    buffer = create_ghost_buffer_from_bc(bc, shape=shape, dx=dx, ghost_depth=ghost_depth)
    buffer.copy_to_interior(interior)
    buffer.update()
    return np.asarray(buffer.padded)


def _mirror_expectation(interior: np.ndarray, axis: int, g: int) -> tuple[np.ndarray, np.ndarray]:
    """Ghost blocks a zero-gradient mirror must produce, from the geometry alone.

    Low: ghost element j is layer `g-j`, whose mirror is interior layer `g-j`, i.e. interior index
    `g-j-1`. High: ghost element j is layer `j+1`, mirror interior index `-(j+1)`.
    """
    take = lambda idx: np.take(interior, idx, axis=axis)  # noqa: E731
    lo = take([g - j - 1 for j in range(g)])
    hi = take([-(j + 1) for j in range(g)])
    return lo, hi


def _mirror_index_map(n: int, g: int) -> np.ndarray:
    """Padded index -> the interior index a zero-gradient mirror puts there, one axis.

    Ghost layer `k` (1-based, counting outward from the wall) mirrors interior layer `k`. The map
    is SEPARABLE, so the expected padded array in n-D is `interior[np.ix_(*maps)]` -- which covers
    the corner blocks by construction rather than by a separate argument. The previous version
    built an expected array and never asserted against it; the corners were guarded only by
    `np.all(np.isfinite(padded))`, and since `GhostBuffer._buffer` is `np.zeros`, an unwritten
    corner is `0.0` and finite. That assertion could not fail for the reason its message gave:
    filling the corners with 7.0, or never writing them, left the whole file green.
    """
    idx = np.empty(n + 2 * g, dtype=int)
    idx[g : g + n] = np.arange(n)
    for k in range(1, g + 1):
        idx[g - k] = k - 1  # low ghost layer k mirrors interior layer k
        idx[g + n + k - 1] = n - k  # high ghost layer k likewise
    return idx


def _ghost_blocks(padded: np.ndarray, axis: int, g: int) -> tuple[np.ndarray, np.ndarray]:
    lo = [slice(None)] * padded.ndim
    hi = [slice(None)] * padded.ndim
    lo[axis], hi[axis] = slice(0, g), slice(-g, None)
    return padded[tuple(lo)], padded[tuple(hi)]


# =============================================================================
# 1-D, both walls, against the geometric mirror
# =============================================================================


@pytest.mark.parametrize("ghost_depth", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("bc_type", ["neumann", "no_flux", "reflecting"])
def test_the_ghost_layers_mirror_the_interior_at_both_walls(bc_type, ghost_depth):
    """Exact, not approximate: a mirror is a permutation of existing values, so the correct result
    is bit-identical to the interior samples.

    Measured before the fix on the old smooth fixture: 5.412e-01 at g=2 and 1.307e+00 at g=3, **at
    both walls** -- unlike the two copies fixed in #1969, where only the high wall moved. Depths
    5 and 6 are here because the slice algebra carries no depth constant and the claim that it
    cannot break at larger g should be asserted, not argued.
    """
    n = 9
    interior = _interior(n)
    padded = _filled(_bc(bc_type, 1, [[0.0, 1.0]]), (n,), 1.0 / n, ghost_depth, interior)

    got_lo, got_hi = _ghost_blocks(padded, 0, ghost_depth)
    want_lo, want_hi = _mirror_expectation(interior, 0, ghost_depth)
    np.testing.assert_array_equal(got_lo, want_lo)
    np.testing.assert_array_equal(got_hi, want_hi)


@pytest.mark.parametrize("ghost_depth", [2, 3, 4])
@pytest.mark.parametrize("wall", ["low", "high"])
def test_neither_wall_is_the_reverse_of_itself(wall, ghost_depth):
    """The signature of this defect was `reversed(got) == want` holding exactly -- every value
    right, every slot wrong. Asserting the reverse does NOT match is a different statement from
    asserting the values are right, and it is the one that fails if the pairing is swapped back
    while the arithmetic stays correct."""
    n = 9
    interior = _interior(n)
    padded = _filled(_bc("neumann", 1, [[0.0, 1.0]]), (n,), 1.0 / n, ghost_depth, interior)

    got_lo, got_hi = _ghost_blocks(padded, 0, ghost_depth)
    want_lo, want_hi = _mirror_expectation(interior, 0, ghost_depth)
    got, want = (got_lo, want_lo) if wall == "low" else (got_hi, want_hi)

    assert not np.array_equal(got[::-1], want)


def test_the_two_walls_do_not_receive_the_same_block():
    """A high-wall/low-end confusion is value-identical on a palindromic fixture, which is what
    this file used to use. On random data the two ghost blocks share no value."""
    n = 9
    interior = _interior(n)
    padded = _filled(_bc("neumann", 1, [[0.0, 1.0]]), (n,), 1.0 / n, 3, interior)
    got_lo, got_hi = _ghost_blocks(padded, 0, 3)

    assert not np.array_equal(got_lo, got_hi)
    assert not np.array_equal(got_lo, got_hi[::-1])


def test_the_interior_is_never_touched():
    """Control. A pairing fix that wrote into the interior would satisfy every ghost assertion
    above while corrupting the field."""
    n = 9
    interior = _interior(n)
    padded = _filled(_bc("neumann", 1, [[0.0, 1.0]]), (n,), 1.0 / n, 3, interior)

    np.testing.assert_array_equal(padded[3:-3], interior)


# =============================================================================
# n-D, which is where the obvious fix is wrong
# =============================================================================


@pytest.mark.parametrize("ghost_depth", [1, 2, 3, 4])
def test_every_axis_and_every_corner_mirrors_correctly_in_2d(ghost_depth):
    """The whole padded array, edges and the corner blocks both axis sweeps write.

    **This is the test that matters for the shape of the fix.** Reversing the calculator's output
    with `[::-1]` reverses axis 0 -- correct in 1-D, wrong for axis 1 in 2-D; every 1-D assertion
    above passes under that version. Deliberately non-square with unequal spacings: a square shape
    lets a swapped axis index coincide with the right answer.
    """
    nx, ny, hx, hy = 5, 8, 0.2, 0.05
    g = ghost_depth
    interior = _interior((nx, ny))
    padded = _filled(_bc("no_flux", 2, [[0, nx * hx], [0, ny * hy]]), (nx, ny), (hx, hy), g, interior)

    # The whole padded array at once, corners included. What this catches is a corner value that
    # differs from the separable mirror -- measured against six corner mutations (filled with a
    # constant, never written, swapped with the opposite corner, left as the raw interior block,
    # reversed along axis 0, transposed): 6 of 6 caught here, 0 of 6 on the file this replaced.
    #
    # It does NOT catch every way a sweep could mishandle a corner, and two such ways are not
    # defects: the axis-0 sweep skipping the corner is byte-identical to the correct output
    # because the axis-1 sweep's full-span write repairs it, and the correct code double-writes
    # the corner by design for the same reason.
    maps = [_mirror_index_map(n, g) for n, g in ((nx, g), (ny, g))]
    np.testing.assert_array_equal(padded, interior[np.ix_(*maps)])


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("ghost_depth", [2, 3])
def test_each_axis_is_mirrored_independently_in_3d(axis, ghost_depth):
    """Three distinct extents, point counts and spacings, so no axis can stand in for another.
    Parametrised on `axis` because the single unparametrised 2-D check was carrying the entire
    axis-discrimination load of this file.
    """
    shape, spacing = (4, 5, 6), (0.25, 0.1, 1.0 / 30.0)
    bounds = [[0.0, n * h] for n, h in zip(shape, spacing, strict=True)]
    g = ghost_depth
    interior = _interior(shape)
    padded = _filled(_bc("no_flux", 3, bounds), shape, spacing, g, interior)

    core = [slice(g, -g)] * 3
    core[axis] = slice(None)
    got_lo, got_hi = _ghost_blocks(padded[tuple(core)], axis, g)
    want_lo, want_hi = _mirror_expectation(interior, axis, g)

    np.testing.assert_array_equal(got_lo, want_lo)
    np.testing.assert_array_equal(got_hi, want_hi)


# =============================================================================
# Depth 1 is the blind spot and must not move
# =============================================================================


@pytest.mark.parametrize(
    ("bc_type", "value"),
    [("neumann", 0.0), ("no_flux", 0.0), ("reflecting", 0.0), ("dirichlet", 1.0)],
)
def test_depth_one_is_unchanged(bc_type, value):
    """A one-element slice is its own reverse, so every depth-1 result predates this change and
    must survive it. Every caller of this route -- of which there are none in the library today --
    would be using depth 1.

    The flux types are pinned at `value = 0`, where the ghost is the interior whatever the
    normal-derivative convention is. At nonzero flux this route disagrees with the live one in
    **three** independent ways and only two of them are #1972's: a factor of 2, a sign at the low
    wall, and -- because `Calculator.compute(interior_value, dx, side)` carries no layer index --
    an offset that stays constant across layers where it must scale as `2k-1`. Measured at
    `neumann(2.0)`, h = 0.125, offset of ghost layer k from its mirror:

        GhostBuffer              low  [-0.5, -0.5, -0.5, -0.5]   flat
        PreallocatedGhostBuffer  low  [0.25, 0.75, 1.25, 1.75]   = 0.25 * [1, 3, 5, 7]
        required ratio                [   1,    3,    5,    7]

    No constant factor and no sign flip turns a flat sequence into 1:3:5:7. None of that is what
    this file pins.
    """
    n = 9
    interior = _interior(n)
    padded = _filled(_bc(bc_type, 1, [[0.0, 1.0]], value=value), (n,), 1.0 / n, 1, interior)

    if bc_type == "dirichlet":
        assert padded[0] == pytest.approx(2 * value - interior[0])
        assert padded[-1] == pytest.approx(2 * value - interior[-1])
    else:
        assert padded[0] == pytest.approx(interior[0])
        assert padded[-1] == pytest.approx(interior[-1])

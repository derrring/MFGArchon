"""`GhostBuffer` paired its ghost and interior slices in opposite directions. #1971

The third and last copy of the pairing corrected in #1967, and the only one that reverses **both**
walls. It differs in kind from the other two: they were per-layer loops with a wrong index
expression, this one is a vectorised slice assignment where the two slices run opposite ways.

    lo_ghost    = [0:g]        index g-1 is adjacent to the wall, index 0 outermost
    lo_interior = [g:2g]       index g   is adjacent to the wall
    buf[lo_ghost] = calc(buf[lo_interior])      pairs outermost ghost with nearest interior

The high side has the mirror problem: its ghost slice runs nearest-first and its interior slice
`[-2g:-g]` runs farthest-first.

**At `ghost_depth = 1` a one-element slice is its own reverse**, which is why the route looks
correct at the only depth anything uses.

The fix steps the *interior* slice backwards rather than reversing the calculator's output,
because the output would have to be reversed along `axis` and the obvious `[::-1]` reverses axis
0 — correct in 1-D and wrong for any other axis in n-D. That mistake was made and caught by the
2-D check below before it shipped.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary import uniform_bc
from mfgarchon.geometry.boundary.applicator_fdm import create_ghost_buffer_from_bc

_N = 8
_H = 1.0 / _N
_XC = (np.arange(_N) + 0.5) * _H


def _low_centres(g: int) -> np.ndarray:
    return np.array([-(k - 0.5) * _H for k in range(1, g + 1)])[::-1]


def _high_centres(g: int) -> np.ndarray:
    return np.array([1.0 + (k - 0.5) * _H for k in range(1, g + 1)])


def _neumann_2d_bc():
    bc = uniform_bc(bc_type="neumann", value=0.0, dimension=2)
    bc.domain_bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    return bc


def _filled(bc, shape, dx, ghost_depth, interior):
    buffer = create_ghost_buffer_from_bc(bc, shape=shape, dx=dx, ghost_depth=ghost_depth)
    buffer.copy_to_interior(interior)
    buffer.update()
    return np.asarray(buffer.padded)


# =============================================================================
# 1-D, both walls, against an exact continuation
# =============================================================================


@pytest.mark.parametrize("ghost_depth", [1, 2, 3, 4])
@pytest.mark.parametrize("wall", ["low", "high"])
def test_the_ghost_layers_are_exact_at_both_walls(wall, ghost_depth):
    """`cos(2πx)` is even about both `x = 0` and `x = 1`, so `NEUMANN(0)` is exact at both and the
    correct ghost is the field itself at the ghost centres.

    Measured before the fix: `5.412e-01` at `g=2` and `1.307e+00` at `g=3`, **at both walls** —
    unlike the two copies fixed in #1967, where only the high wall was affected. `g=4` is included
    because the two other copies were each verified to depth 3 only.
    """
    bc = uniform_bc(bc_type="neumann", value=0.0, dimension=1)
    bc.domain_bounds = np.array([[0.0, 1.0]])
    padded = _filled(bc, (_N,), _H, ghost_depth, np.cos(2 * np.pi * _XC))

    if wall == "low":
        got, centres = padded[:ghost_depth], _low_centres(ghost_depth)
    else:
        got, centres = padded[-ghost_depth:], _high_centres(ghost_depth)

    np.testing.assert_allclose(got, np.cos(2 * np.pi * centres), atol=1e-12)


@pytest.mark.parametrize("ghost_depth", [2, 3])
@pytest.mark.parametrize("wall", ["low", "high"])
def test_neither_wall_is_the_reverse_of_itself(wall, ghost_depth):
    """The signature of this defect was `reversed(got) == want` holding exactly — every value
    right, every slot wrong. Asserting the reverse does *not* match is a different statement from
    asserting the values are right, and it is the one that fails if the pairing is swapped back
    while the arithmetic stays correct."""
    bc = uniform_bc(bc_type="neumann", value=0.0, dimension=1)
    bc.domain_bounds = np.array([[0.0, 1.0]])
    padded = _filled(bc, (_N,), _H, ghost_depth, np.cos(2 * np.pi * _XC))

    if wall == "low":
        got, centres = padded[:ghost_depth], _low_centres(ghost_depth)
    else:
        got, centres = padded[-ghost_depth:], _high_centres(ghost_depth)

    assert not np.allclose(got[::-1], np.cos(2 * np.pi * centres), atol=1e-12)


def test_the_interior_is_never_touched():
    """Control. A pairing fix that wrote into the interior would satisfy every ghost assertion
    above while corrupting the field."""
    bc = uniform_bc(bc_type="neumann", value=0.0, dimension=1)
    bc.domain_bounds = np.array([[0.0, 1.0]])
    interior = np.cos(2 * np.pi * _XC)
    padded = _filled(bc, (_N,), _H, 3, interior)

    np.testing.assert_array_equal(padded[3:-3], interior)


# =============================================================================
# n-D, which is where the obvious fix is wrong
# =============================================================================


@pytest.mark.parametrize("ghost_depth", [1, 2, 3])
def test_the_pairing_is_correct_on_every_axis_and_in_the_corners(ghost_depth):
    """`cos(2πx)·cos(2πy)` is even about all four walls, so the whole padded array has a known
    exact value — edges, and the corner blocks both axis sweeps write.

    **This is the test that matters for the shape of the fix.** Reversing the calculator's output
    with `[::-1]` reverses axis 0, which is correct in 1-D and wrong for axis 1 in 2-D; every 1-D
    assertion above passes under that version. Stepping the interior *slice* backwards is indexed
    by `axis` and cannot have that failure.
    """
    n = 6
    h = 1.0 / n
    c = (np.arange(n) + 0.5) * h
    xx, yy = np.meshgrid(c, c, indexing="ij")
    padded = _filled(_neumann_2d_bc(), (n, n), (h, h), ghost_depth, np.cos(2 * np.pi * xx) * np.cos(2 * np.pi * yy))

    g = ghost_depth
    gc = np.concatenate(
        [
            np.array([-(k - 0.5) * h for k in range(1, g + 1)])[::-1],
            c,
            np.array([1.0 + (k - 0.5) * h for k in range(1, g + 1)]),
        ]
    )
    gx, gy = np.meshgrid(gc, gc, indexing="ij")

    np.testing.assert_allclose(padded, np.cos(2 * np.pi * gx) * np.cos(2 * np.pi * gy), atol=1e-12)


def test_a_non_square_grid_does_not_let_an_axis_swap_hide():
    """Square shapes and equal spacings let a swapped axis index coincide with the right answer.
    Different extents, different spacings, different point counts on the two axes."""
    nx, ny = 5, 8
    hx, hy = 0.2, 0.05
    cx = (np.arange(nx) + 0.5) * hx
    cy = (np.arange(ny) + 0.5) * hy
    xx, yy = np.meshgrid(cx, cy, indexing="ij")
    interior = np.cos(np.pi * xx / (nx * hx)) * np.cos(np.pi * yy / (ny * hy))

    bc = uniform_bc(bc_type="no_flux", dimension=2)
    bc.domain_bounds = np.array([[0.0, nx * hx], [0.0, ny * hy]])
    padded = _filled(bc, (nx, ny), (hx, hy), 2, interior)

    # A no-flux wall mirrors, so the two ghost layers on each side must be the first two interior
    # planes reversed -- checked per axis, which is what an axis swap breaks.
    np.testing.assert_allclose(padded[:2, 2:-2], interior[1::-1, :], atol=1e-12)
    np.testing.assert_allclose(padded[-2:, 2:-2], interior[:-3:-1, :], atol=1e-12)
    np.testing.assert_allclose(padded[2:-2, :2], interior[:, 1::-1], atol=1e-12)
    np.testing.assert_allclose(padded[2:-2, -2:], interior[:, :-3:-1], atol=1e-12)


# =============================================================================
# Depth 1 is the blind spot and must not move
# =============================================================================


@pytest.mark.parametrize(
    ("bc_type", "value"),
    [("neumann", 0.0), ("no_flux", 0.0), ("reflecting", 0.0), ("dirichlet", 1.0)],
)
def test_depth_one_is_unchanged(bc_type, value):
    """A one-element slice is its own reverse, so every depth-1 result predates this change and
    must survive it. Every caller of this route — of which there are none in the library today —
    would be using depth 1.

    The flux types are pinned at `value = 0`, where the ghost is the interior whatever the
    normal-derivative spacing convention is. At nonzero flux this route's coefficient disagrees
    with the live one by a factor of 2 and, at the low wall, by a sign (#1972); that is a second
    defect on the same dead route and it is not what this file pins.
    """
    bc = uniform_bc(bc_type=bc_type, value=value, dimension=1)
    bc.domain_bounds = np.array([[0.0, 1.0]])
    interior = np.cos(2 * np.pi * _XC)
    padded = _filled(bc, (_N,), _H, 1, interior)

    if bc_type == "dirichlet":
        assert padded[0] == pytest.approx(2 * value - interior[0])
        assert padded[-1] == pytest.approx(2 * value - interior[-1])
    else:
        assert padded[0] == pytest.approx(interior[0])
        assert padded[-1] == pytest.approx(interior[-1])

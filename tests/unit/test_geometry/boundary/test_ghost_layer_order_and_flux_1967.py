"""Ghost layers must be paired nearest-first at both walls, and an inhomogeneous flux grows with
the layer. #1967

Two defects in the same loop, in two copies of it:

1. **Layer order.** The high ghosts occupy padded indices `-g .. -1` and `-g` is the one adjacent
   to the wall, so both the ghost walk and the interior walk must start there. The high-wall loop
   started its ghost walk at `-1`, the far end, while its interior walk started near — pairing the
   layers backwards for `g >= 2`. The low wall was already correct.
2. **Flux offset.** Ghost layer `k` sits `(2k-1)*dx` from its mirror on a cell-centred grid, so an
   inhomogeneous Neumann offset must scale with the layer. `dx * v` was applied to every layer —
   the `k = 1` value used throughout.

**Both are invisible at `ghost_depth = 1`**, where a single layer has no order to reverse and
`(2·1-1)·dx = dx`. Every caller in the library passes 1 or omits it, which is why a suite of 6147
was green with both present. The one production consumer at depth 3 is `hjb_weno.py:330`, whose
`_SUPPORTED_BC_TYPES` is `{NEUMANN, NO_FLUX, PERIODIC}` — exactly the family defect 1 hits.

**The oracle is an exact continuation, not another code path.** `cos(2πx)` is even about both
`x = 0` and `x = 1`, so `NEUMANN(0)` is satisfied exactly at both walls and the correct ghost is
the field itself at the ghost centres. For the inhomogeneous case a linear field makes the
prescribed flux exact.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    neumann_bc,
    pad_array_with_ghosts,
    uniform_bc,
)

_N = 8
_H = 1.0 / _N
_XC = (np.arange(_N) + 0.5) * _H  # cell centres, the geometry this branch uses (#1968)

_MIRROR_FAMILY = ["neumann", "no_flux", "reflecting"]


def _low_centres(g: int) -> np.ndarray:
    """Ghost layer k has centre at -(k - 1/2)h; returned outermost-first to match the buffer."""
    return np.array([-(k - 0.5) * _H for k in range(1, g + 1)])[::-1]


def _high_centres(g: int) -> np.ndarray:
    return np.array([1.0 + (k - 0.5) * _H for k in range(1, g + 1)])


def _uniform(bc_type: str, value: float = 0.0) -> BoundaryConditions:
    bc = uniform_bc(bc_type=bc_type, value=value, dimension=1)
    bc.domain_bounds = np.array([[0.0, 1.0]])
    return bc


def _per_face(bc_type: BCType, value: float = 0.0) -> BoundaryConditions:
    """The other copy of the same loop. Two segments make `is_uniform` False, which is what
    routes to `_apply_ghost_for_face` instead of `_apply_linear_reflection`."""
    return BoundaryConditions(
        segments=[
            BCSegment(name="lo", bc_type=bc_type, value=value, boundary="x_min"),
            BCSegment(name="hi", bc_type=bc_type, value=value, boundary="x_max"),
        ],
        dimension=1,
        default_bc=bc_type,
        default_value=value,
        domain_bounds=np.array([[0.0, 1.0]]),
    )


# =============================================================================
# Defect 1: layer order, against an exact continuation
# =============================================================================


@pytest.mark.parametrize("ghost_depth", [1, 2, 3])
@pytest.mark.parametrize("wall", ["low", "high"])
@pytest.mark.parametrize("bc_type", _MIRROR_FAMILY)
def test_the_mirror_family_is_exact_at_both_walls_and_every_depth(bc_type, wall, ghost_depth):
    """`cos(2πx)` is even about both walls, so a homogeneous Neumann condition is exact there and
    the ghost is the field itself.

    **Both walls are parametrised on purpose.** The low wall was machine-zero throughout while the
    high wall was `5.4e-01` at `g=2` and `1.3e+00` at `g=3`. A one-wall test measured the correct
    half of the loop and certified the whole thing — that is how #1966 came to clear this family.
    """
    field = np.cos(2 * np.pi * _XC)
    padded = pad_array_with_ghosts(field, _uniform(bc_type), ghost_depth=ghost_depth, spacing=_H)

    if wall == "low":
        got, want = padded[:ghost_depth], np.cos(2 * np.pi * _low_centres(ghost_depth))
    else:
        got, want = padded[-ghost_depth:], np.cos(2 * np.pi * _high_centres(ghost_depth))

    np.testing.assert_allclose(got, want, atol=1e-12)


@pytest.mark.parametrize("ghost_depth", [2, 3])
def test_the_high_wall_is_no_longer_the_reverse_of_itself(ghost_depth):
    """The signature of the defect was that `reversed(got) == want` held exactly — every value
    correct, every slot wrong. Asserting the reverse does *not* match is a different statement
    from asserting the values are right, and it is the one that fails if a future change
    reintroduces the swap while keeping the arithmetic."""
    field = np.cos(2 * np.pi * _XC)
    padded = pad_array_with_ghosts(field, _uniform("neumann"), ghost_depth=ghost_depth, spacing=_H)
    got = padded[-ghost_depth:]

    assert not np.allclose(got[::-1], np.cos(2 * np.pi * _high_centres(ghost_depth)), atol=1e-12)


# =============================================================================
# Defect 2: the flux offset grows with the layer
# =============================================================================


@pytest.mark.parametrize("ghost_depth", [1, 2, 3])
def test_an_inhomogeneous_flux_is_exact_at_every_layer(ghost_depth):
    """`f = -2x + 1` has `du/dx = -2`, so `du/dn = +2` at the low wall, and the linear
    continuation is the exact ghost.

    Measured before the fix: 0 at `g=1`, `5.0e-01` at `g=2`, `1.0e+00` at `g=3` — the drift is
    `(2k-2)·dx·v`, zero for the first layer, which is the whole reason a depth-1 suite saw nothing.
    """
    field = -2.0 * _XC + 1.0
    padded = pad_array_with_ghosts(field, neumann_bc(dimension=1, value=2.0), ghost_depth=ghost_depth, spacing=_H)

    np.testing.assert_allclose(padded[:ghost_depth], -2.0 * _low_centres(ghost_depth) + 1.0, atol=1e-12)


def test_the_flux_offset_actually_scales_with_the_layer():
    """Directly, without an exact solution: the gap between a `v = 0` wall and a `v != 0` one must
    be `(2k-1)·dx·v` per layer. A fix that applied a constant offset would satisfy the exactness
    test above only for the field that happens to make it exact; this holds for any field."""
    field = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    v, g = 2.0, 3

    hot = pad_array_with_ghosts(field, neumann_bc(dimension=1, value=v), ghost_depth=g, spacing=_H)
    cold = pad_array_with_ghosts(field, neumann_bc(dimension=1, value=0.0), ghost_depth=g, spacing=_H)

    # outermost-first in the array, so layer k = g, g-1, ... 1
    expected = np.array([(2 * k - 1) * _H * v for k in range(g, 0, -1)])
    np.testing.assert_allclose(hot[:g] - cold[:g], expected, atol=1e-12)


# =============================================================================
# The second copy of the same loop
# =============================================================================


@pytest.mark.parametrize("ghost_depth", [1, 2, 3])
@pytest.mark.parametrize("bc_type", [BCType.NEUMANN, BCType.NO_FLUX, BCType.REFLECTING])
def test_the_per_face_path_carries_the_same_repair(bc_type, ghost_depth):
    """`_apply_ghost_for_face` is a second copy of this arithmetic and carried both defects. It is
    reached whenever the BC names its faces rather than being uniform — measured, fixing only the
    uniform path made the two paths disagree and reddened
    `test_per_face_bc_path_1937.py::test_the_flux_is_applied_at_every_ghost_layer`."""
    field = np.cos(2 * np.pi * _XC)
    padded = pad_array_with_ghosts(field, _per_face(bc_type), ghost_depth=ghost_depth, spacing=_H)

    np.testing.assert_allclose(padded[:ghost_depth], np.cos(2 * np.pi * _low_centres(ghost_depth)), atol=1e-12)
    np.testing.assert_allclose(padded[-ghost_depth:], np.cos(2 * np.pi * _high_centres(ghost_depth)), atol=1e-12)


def test_the_two_paths_agree_on_an_inhomogeneous_flux_at_depth():
    """The cross-path check that caught the incomplete fix. It is not the oracle — after both
    copies are corrected, agreement is compatible with both being wrong — so it sits beside the
    exactness tests above rather than replacing them."""
    field = -2.0 * _XC + 1.0
    uniform = pad_array_with_ghosts(field, neumann_bc(dimension=1, value=2.0), ghost_depth=3, spacing=_H)
    per_face = pad_array_with_ghosts(field, _per_face(BCType.NEUMANN, 2.0), ghost_depth=3, spacing=_H)

    np.testing.assert_allclose(per_face, uniform, atol=1e-12)


# =============================================================================
# Depth 1 is the blind spot, and must not move
# =============================================================================


@pytest.mark.parametrize(
    ("bc_type", "value"),
    [("neumann", 0.0), ("neumann", 2.0), ("no_flux", 0.0), ("reflecting", 0.0), ("dirichlet", 1.0), ("periodic", 0.0)],
)
def test_depth_one_is_byte_identical(bc_type, value):
    """Eighteen of the twenty-four probed cells were unchanged by this fix; every `ghost_depth = 1`
    cell is among them, because that is where the two expressions coincide. Since every caller in
    the library uses depth 1, a change here would be a regression in everything that currently
    works, and nothing else in this file would see it."""
    field = np.cos(2 * np.pi * _XC)
    padded = pad_array_with_ghosts(field, _uniform(bc_type, value), ghost_depth=1, spacing=_H)

    if bc_type == "periodic":
        assert padded[0] == pytest.approx(field[-1])
        assert padded[-1] == pytest.approx(field[0])
    elif bc_type == "dirichlet":
        assert padded[0] == pytest.approx(2 * value - field[0])
    else:
        assert padded[0] == pytest.approx(field[0] + _H * value)

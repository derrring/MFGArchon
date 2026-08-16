"""`EXTRAPOLATION_*` must reach its ghost formula through the buffer, on both of its chains. #1958

`PreallocatedGhostBuffer` had no branch for either member on either dispatch chain, and the two
failed differently:

- **mixed** (`_apply_ghost_for_face`) fell to `else: # Fallback for unknown BC types: use
  reflection`;
- **uniform** (`_apply_linear_reflection`) had no terminal `else` at all, so the ghost cells kept
  the buffer's zero-initialised contents -- not a wrong condition, no condition.

`fp_semi_lagrangian` builds an `EXTRAPOLATION_QUADRATIC` BC every timestep and reached the first
one, computing a boundary Laplacian **2000% wrong**.

The formulas were already written and directly tested; nothing reached them from here. That is the
shape this file pins: not the arithmetic, which was never in doubt, but the routing.

**The fixture is deliberately asymmetric.** On `U = 0.5x^2` only the high wall shows the defect --
the parabola is symmetric about `x = 0`, so the reflection ghost and the quadratic ghost coincide
at the low wall. A symmetric field cannot see this, which is why it survived.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    pad_array_with_ghosts,
    uniform_bc,
)
from mfgarchon.operators.stencils.finite_difference import laplacian_with_bc

# Asymmetric on purpose: no two of these are related by a reflection about either wall.
_U = np.array([2.5, 3.1, 4.0, 5.2, 6.6])
_DX = 0.25

_EXACT = {
    BCType.EXTRAPOLATION_LINEAR: (2 * _U[0] - _U[1], 2 * _U[-1] - _U[-2]),
    BCType.EXTRAPOLATION_QUADRATIC: (3 * _U[0] - 3 * _U[1] + _U[2], 3 * _U[-1] - 3 * _U[-2] + _U[-3]),
}


def _mixed(bc_type: BCType) -> BoundaryConditions:
    """Two named faces -- what `fp_semi_lagrangian` builds, and what makes `is_uniform` False."""
    return BoundaryConditions(
        segments=[
            BCSegment(name="l", bc_type=bc_type, value=0.0, boundary="x_min"),
            BCSegment(name="r", bc_type=bc_type, value=0.0, boundary="x_max"),
        ],
        dimension=1,
        default_bc=bc_type,
        default_value=0.0,
        domain_bounds=np.array([[0.0, 1.0]]),
    )


def _uniform(bc_type: BCType) -> BoundaryConditions:
    bc = uniform_bc(bc_type=bc_type.value, dimension=1)
    bc.domain_bounds = np.array([[0.0, 1.0]])
    return bc


@pytest.mark.parametrize("build", [_mixed, _uniform], ids=["mixed", "uniform"])
@pytest.mark.parametrize("bc_type", list(_EXACT), ids=lambda t: t.name)
def test_the_ghost_is_the_extrapolation_formula_on_both_chains(build, bc_type):
    """Both chains are parametrised because they failed differently -- one silently reflected,
    one returned zero -- and a fix to either alone would pass a single-chain test.

    Measured before the fix on this fixture: mixed gave 3.1 / 5.2 for both members (the
    reflection ghost, which is the *second* interior cell), uniform gave 0.0 / 0.0.
    """
    lo_exact, hi_exact = _EXACT[bc_type]
    padded = pad_array_with_ghosts(_U, build(bc_type), spacing=_DX)

    assert padded[0] == pytest.approx(lo_exact), "low wall"
    assert padded[-1] == pytest.approx(hi_exact), "high wall"


def test_the_two_members_do_not_agree_on_this_fixture():
    """Negative control for the parametrisation above. If linear and quadratic gave the same
    ghost here, every row would pass over an implementation that confused them."""
    lin = pad_array_with_ghosts(_U, _mixed(BCType.EXTRAPOLATION_LINEAR), spacing=_DX)
    quad = pad_array_with_ghosts(_U, _mixed(BCType.EXTRAPOLATION_QUADRATIC), spacing=_DX)

    assert lin[0] != pytest.approx(quad[0])
    assert lin[-1] != pytest.approx(quad[-1])


def test_the_reflection_fallback_is_no_longer_what_extrapolation_gets():
    """The mixed chain's `else` branch still exists for genuinely unknown types. This asserts
    extrapolation no longer lands in it -- without this, adding a branch that happens to
    reproduce the reflection value would pass every test above."""
    extrap = pad_array_with_ghosts(_U, _mixed(BCType.EXTRAPOLATION_QUADRATIC), spacing=_DX)
    reflected = pad_array_with_ghosts(_U, _mixed(BCType.REFLECTING), spacing=_DX)

    assert extrap[0] != pytest.approx(reflected[0])
    assert extrap[-1] != pytest.approx(reflected[-1])


# =============================================================================
# The oracle from the issue: a field the extrapolation is exact for
# =============================================================================


@pytest.mark.parametrize(
    ("field_fn", "bc_type", "expected_laplacian"),
    [
        (lambda x: 0.5 * x**2, BCType.EXTRAPOLATION_QUADRATIC, 1.0),
        (lambda x: 2.0 * x + 1.0, BCType.EXTRAPOLATION_LINEAR, 0.0),
    ],
    ids=["quadratic-on-a-parabola", "linear-on-a-line"],
)
def test_the_laplacian_is_exact_where_the_extrapolation_is_exact(field_fn, bc_type, expected_laplacian):
    """Quadratic extrapolation reproduces a parabola exactly, so the discrete Laplacian must be
    exact *including the wall rows*. This is the measurement in #1958: it read -19 against a true
    1, a 2000% error, on the path `fp_semi_lagrangian` takes every timestep.

    The linear row is the sibling property and is not redundant -- it fails if the two members
    are wired to the same formula.
    """
    x = np.linspace(0.0, 1.0, 11)
    dx = x[1] - x[0]

    lap = laplacian_with_bc(field_fn(x), spacings=[dx], bc=_mixed(bc_type))

    np.testing.assert_allclose(lap, expected_laplacian, atol=1e-12)


def test_a_grid_too_small_for_the_stencil_refuses():
    """`EXTRAPOLATION_QUADRATIC` reads three interior cells. On a two-cell grid the old code
    would have read across the array; silently dropping to a lower order is the class of
    substitution this whole area has been removing."""
    with pytest.raises(ValueError, match="interior cells"):
        pad_array_with_ghosts(np.array([1.0, 2.0]), _mixed(BCType.EXTRAPOLATION_QUADRATIC), spacing=_DX)


# =============================================================================
# Scope
# =============================================================================


@pytest.mark.parametrize(
    "bc_type",
    [t for t in BCType if t not in _EXACT],
    ids=lambda t: t.name,
)
@pytest.mark.parametrize("build", [_mixed, _uniform], ids=["mixed", "uniform"])
def test_every_other_member_is_untouched(build, bc_type):
    """Twelve of the sixteen probed cells are byte-identical across this change. Only the two
    extrapolation members on the two chains moved; a fix that disturbed anything else would be a
    regression wearing this commit's message."""
    _BEFORE = {
        ("mixed", "DIRICHLET"): (-2.5, -6.6),
        ("mixed", "NEUMANN"): (2.5, 6.6),
        ("mixed", "NO_FLUX"): (2.5, 6.6),
        ("mixed", "PERIODIC"): (6.6, 2.5),
        ("mixed", "REFLECTING"): (2.5, 6.6),
        ("mixed", "ROBIN"): (-2.5, -6.6),
        ("uniform", "DIRICHLET"): (-2.5, -6.6),
        ("uniform", "NEUMANN"): (2.5, 6.6),
        ("uniform", "NO_FLUX"): (2.5, 6.6),
        ("uniform", "PERIODIC"): (6.6, 2.5),
        ("uniform", "REFLECTING"): (2.5, 6.6),
        ("uniform", "ROBIN"): (-2.5, -6.6),
    }
    kind = "mixed" if build is _mixed else "uniform"
    lo, hi = _BEFORE[(kind, bc_type.name)]
    padded = pad_array_with_ghosts(_U, build(bc_type), spacing=_DX)

    assert padded[0] == pytest.approx(lo)
    assert padded[-1] == pytest.approx(hi)

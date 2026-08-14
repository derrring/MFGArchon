"""The per-face ghost path must agree with the uniform one, and must not invent a fallback.

`PreallocatedGhostBuffer.update_ghosts` dispatches on `bc.is_uniform`: one unrestricted segment
routes to `_update_ghosts_uniform` -> `_apply_linear_reflection`, anything else to
`_update_ghosts_mixed` -> `_apply_ghost_for_face`. Both are live, and the choice between them is the
caller's phrasing of the same physical condition. #1937
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    dirichlet_bc,
    neumann_bc,
    no_flux_bc,
    pad_array_with_ghosts,
)
from mfgarchon.geometry.boundary.applicator_fdm import FDMApplicator

# u[0] != u[1] and the step is 0.75, deliberately not equal to any dx*g used below. A ramp whose
# per-axis step happens to equal dx*g makes `u[0] + dx*g` and `u[1]` identical, which collapses the
# two ghost conventions onto each other and makes the comparison below pass over the defect.
_RAMP = np.array([10.0, 10.75, 11.5, 12.25, 13.0])
_DX = 0.25


def _per_face_neumann(value: float) -> BoundaryConditions:
    return BoundaryConditions(
        segments=[
            BCSegment(name="lo", bc_type=BCType.NEUMANN, value=value, boundary="x_min"),
            BCSegment(name="hi", bc_type=BCType.NEUMANN, value=value, boundary="x_max"),
        ],
        dimension=1,
    )


@pytest.mark.parametrize("flux", [2.0, -3.0, 0.0])
def test_the_two_ghost_paths_agree_on_an_inhomogeneous_neumann_flux(flux):
    """`_apply_ghost_for_face` mirrored the interior and never read the segment's value, so a
    per-face Neumann BC was applied as zero-flux while the identical uniform BC was not.

    `flux=0.0` is the declared positive control: the two paths agreed there before the fix, so it
    passes either way and is here to show the comparison can produce agreement.
    """
    uniform = pad_array_with_ghosts(_RAMP, neumann_bc(dimension=1, value=flux), spacing=_DX)
    per_face = pad_array_with_ghosts(_RAMP, _per_face_neumann(flux), spacing=_DX)

    np.testing.assert_allclose(per_face, uniform, atol=1e-12)


@pytest.mark.parametrize("flux", [2.0, -3.0])
def test_the_per_face_path_imposes_the_flux_that_was_asked_for(flux):
    """An oracle independent of both paths: recover `du/dn` from the ghost and compare it to the
    requested value.

    A path-A-vs-path-B comparison alone goes tautological the moment the two share an
    implementation. This does not: it is the boundary condition's own definition, with the outward
    normal pointing `-x` at the low wall and `+x` at the high wall.
    """
    padded = pad_array_with_ghosts(_RAMP, _per_face_neumann(flux), spacing=_DX)

    du_dn_low = -(_RAMP[0] - padded[0]) / _DX
    du_dn_high = (padded[-1] - _RAMP[-1]) / _DX

    assert du_dn_low == pytest.approx(flux, abs=1e-12)
    assert du_dn_high == pytest.approx(flux, abs=1e-12)


def _mixed_exit_and_walls(exit_priority: int) -> BoundaryConditions:
    """The idiom from `BCSegment`'s own docstring: one Dirichlet exit, the rest walls."""
    return BoundaryConditions(
        segments=[
            BCSegment(
                name="exit",
                bc_type=BCType.DIRICHLET,
                value=7.0,
                boundary="x_min",
                priority=exit_priority,
            ),
            BCSegment(name="walls", bc_type=BCType.NO_FLUX, value=0.0),
        ],
        dimension=2,
        default_bc=BCType.NO_FLUX,
        domain_bounds=np.array([[0.0, 10.0], [0.0, 3.0]]),
    )


def _walls_carrying_the_dirichlet_ghost(bc: BoundaryConditions) -> int:
    """How many of the four walls came back with `2*7 - 1 = 13`, the Dirichlet ghost."""
    field = np.ones((4, 11))
    out = np.asarray(FDMApplicator(dimension=2).apply(field, bc))
    walls = [out[0, 1:-1], out[-1, 1:-1], out[1:-1, 0], out[1:-1, -1]]
    return sum(1 for w in walls if np.allclose(w, 13.0))


@pytest.mark.parametrize("exit_priority", [1, -5])
def test_an_unclaimed_wall_takes_default_bc_and_not_the_first_segment(exit_priority):
    """`_update_ghosts_mixed` fell back to `bc.segments[0]`, and segments sort priority-descending,
    so the highest-priority segment became the fallback for every wall it had not claimed.

    Both priorities are asserted to give the same answer, and only one of them detects anything.
    Measured against the reverted code: `exit_priority=1` gives 4 of 4 and fails; `exit_priority=-5`
    gives 1 of 4 and **passes**, because at that priority the exit sorts last and `segments[0]` is
    the walls segment, so the old fallback happens to coincide with the correct answer.

    The `-5` row is therefore a control, not a detector, and it is kept for what it shows: a result
    that moves with segment sort order is a fingerprint of the fallback rather than of any boundary
    condition. Anyone tempted to drop the `1` row as redundant should note that it is the only row
    that fails when the fallback returns.
    """
    assert _walls_carrying_the_dirichlet_ghost(_mixed_exit_and_walls(exit_priority)) == 1


def test_the_wall_count_probe_can_produce_both_answers():
    """Positive controls for the probe above, in both directions.

    Without these, `== 1` could be satisfied by a probe that cannot count, or by a Dirichlet ghost
    that never appears at all.
    """
    assert _walls_carrying_the_dirichlet_ghost(no_flux_bc(dimension=2)) == 0
    assert _walls_carrying_the_dirichlet_ghost(dirichlet_bc(dimension=2, value=7.0)) == 4


def test_an_uncovered_face_with_no_default_bc_raises_rather_than_guessing():
    """#1100: resolution must not substitute a default. The old `segments[0]` fallback supplied one
    by accident of sort order, which is exactly the silent guess that rule forbids."""
    bc = BoundaryConditions(
        segments=[BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=7.0, boundary="x_min")],
        dimension=2,
        domain_bounds=np.array([[0.0, 10.0], [0.0, 3.0]]),
    )

    with pytest.raises(ValueError, match="default_bc was not"):
        FDMApplicator(dimension=2).apply(np.ones((4, 11)), bc)


@pytest.mark.parametrize(
    ("default_bc", "default_value", "expected_hi_ghost"),
    [
        (BCType.NO_FLUX, 0.0, 13.0),  # mirror: u[-1]
        (BCType.NEUMANN, 3.0, 13.75),  # u[-1] + dx*v = 13.0 + 0.25*3
        (BCType.DIRICHLET, 5.0, -3.0),  # 2*g - u[-1] = 10 - 13
        (BCType.PERIODIC, 0.0, 10.0),  # wraps to u[0]
    ],
)
def test_the_default_branch_uses_both_default_bc_and_default_value(default_bc, default_value, expected_hi_ghost):
    """The synthetic fallback segment must carry the declared type AND the declared value.

    Independent review measured that the branch this file exercises is entered seven times across
    the suite and that `default_bc` is *always* `NO_FLUX` with `default_value=0.0` -- a fixture
    monoculture in which hardcoding `NO_FLUX`, dropping `default_value`, and corrupting the `dx`
    fallback all pass. One value of one parameter cannot discriminate a two-parameter construction.

    Each row here produces a different number by a different formula, so no single wrong constant
    can satisfy more than one. The `x_min` wall is claimed by an explicit Dirichlet segment in every
    row and its ghost is asserted unchanged at `2*7 - 10 = 4.0`, which is the control: it shows the
    rows differ because the DEFAULT branch differs, not because the whole BC changed.
    """
    bc = BoundaryConditions(
        segments=[BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=7.0, boundary="x_min")],
        dimension=1,
        default_bc=default_bc,
        default_value=default_value,
    )

    padded = pad_array_with_ghosts(_RAMP, bc, spacing=_DX)

    assert padded[-1] == pytest.approx(expected_hi_ghost, abs=1e-12)
    assert padded[0] == pytest.approx(4.0, abs=1e-12), (
        "the claimed wall moved, so the row is not isolating the default branch"
    )


@pytest.mark.parametrize("ghost_depth", [1, 2, 3])
def test_the_flux_is_applied_at_every_ghost_layer(ghost_depth):
    """The flux loop runs `for k in range(g)`. With `g=1` a fix that applied the flux only to the
    outermost layer would be indistinguishable from a correct one, and every other test in this file
    uses the default depth."""
    uniform = pad_array_with_ghosts(_RAMP, neumann_bc(dimension=1, value=2.0), ghost_depth=ghost_depth, spacing=_DX)
    per_face = pad_array_with_ghosts(_RAMP, _per_face_neumann(2.0), ghost_depth=ghost_depth, spacing=_DX)

    np.testing.assert_allclose(per_face, uniform, atol=1e-12)


def test_each_axis_uses_its_own_spacing():
    """`dx` is indexed by `axis`. Every other test here is 1-D, where axis 0 *is* the axis, so a fix
    that read `self._grid_spacing[0]` would be indistinguishable from a correct one -- measured: that
    mutation passes the whole rest of this file.

    Non-square on purpose, with three different spacings and a non-square shape, so a swapped axis
    index cannot coincide with the right answer.
    """
    spacing = (0.25, 0.1)
    flux = 2.0
    field = np.zeros((4, 6))

    bc = BoundaryConditions(
        segments=[
            BCSegment(name=f"{ax}_{side}", bc_type=BCType.NEUMANN, value=flux, boundary=f"{ax}_{side}")
            for ax in ("x", "y")
            for side in ("min", "max")
        ],
        dimension=2,
    )
    padded = pad_array_with_ghosts(field, bc, spacing=spacing)

    # Interior is all zeros, so each ghost is exactly its own axis's dx * flux.
    for axis, dx in enumerate(spacing):
        lo = padded[(0, slice(1, -1)) if axis == 0 else (slice(1, -1), 0)]
        hi = padded[(-1, slice(1, -1)) if axis == 0 else (slice(1, -1), -1)]
        np.testing.assert_allclose(lo, dx * flux, atol=1e-12, err_msg=f"axis {axis} low wall used the wrong dx")
        np.testing.assert_allclose(hi, dx * flux, atol=1e-12, err_msg=f"axis {axis} high wall used the wrong dx")

    assert spacing[0] != spacing[1], "the spacings must differ or this test cannot detect a swap"

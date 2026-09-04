"""The two diffusion-matrix builders must mean the same thing by the same `bc_type`. Issue #2242.

`build_diffusion_matrix_2d` handled `neumann` and `periodic` and had no `dirichlet` branch, so
asking for Dirichlet fell through to the interior stencil everywhere. Nothing raised, and the
matrix returned was **byte-identical to the one you got by passing a nonsense string** -- which is
what made it invisible: the only way to notice was to compare 1D against 2D, and no test did.

The functions had zero callers and zero tests when this was written. That is why the checks below
are shaped as agreement between the two builders rather than golden values: a golden matrix would
have been generated from the code and would have frozen the defect.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.adjoint.operators import (
    BC_TYPES,
    build_advection_matrix_1d,
    build_diffusion_matrix_1d,
    build_diffusion_matrix_2d,
)

SIGMA = 0.4
DT = 0.01
DX = 0.25


def _boundary_rows(nx: int, ny: int) -> list[int]:
    """Row-major indices of the perimeter, the same `idx(i, j) = i * Ny + j` the builder uses."""
    return [i * ny + j for i in range(nx) for j in range(ny) if i in (0, nx - 1) or j in (0, ny - 1)]


@pytest.mark.parametrize("bc_type", BC_TYPES)
def test_every_declared_bc_type_produces_a_distinct_2d_operator(bc_type):
    """Each declared type must change something, or the argument is decorative for that value.

    **This one would NOT have caught #2242, and saying so is the point.** Measured: with the
    Dirichlet branch removed again it still passes. The fallthrough produced the interior stencil
    everywhere, which differs from Neumann (wall rows) and from periodic (wrap terms) -- so the
    broken Dirichlet was distinct from both while being wrong. What it collided with was the
    UNVALIDATED path, and that no longer exists to compare against.

    Kept as a guard against the next version of the defect rather than this one: a type added to
    `BC_TYPES` without a branch collapses onto whichever path it falls through to, and this fires.
    The pin for #2242 itself is the next test.
    """
    a = build_diffusion_matrix_2d((4, 4), DX, SIGMA, DT, 0.5, bc_type).toarray()
    others = [
        build_diffusion_matrix_2d((4, 4), DX, SIGMA, DT, 0.5, other).toarray() for other in BC_TYPES if other != bc_type
    ]
    for other, b in zip([o for o in BC_TYPES if o != bc_type], others, strict=True):
        assert np.abs(a - b).max() > 1e-6, f"{bc_type!r} and {other!r} build the same 2D operator"


def test_dirichlet_means_identity_boundary_rows_in_both_builders():
    """The pin for #2242. Verified by removing the branch again: this is the test that goes red.

    1D already zeroed its boundary rows and set the diagonal to 1. 2D now does the same on the
    whole perimeter. Asserting the row SUM as well as the diagonal is what distinguishes an
    identity row from a row that merely happens to carry 1 on the diagonal.
    """
    a1 = build_diffusion_matrix_1d(5, DX, SIGMA, DT, 0.5, "dirichlet").toarray()
    for k in (0, 4):
        assert a1[k, k] == pytest.approx(1.0), f"1D row {k} diagonal: {a1[k, k]}"
        assert a1[k].sum() == pytest.approx(1.0), f"1D row {k} is not the identity: {a1[k]}"

    a2 = build_diffusion_matrix_2d((4, 5), DX, SIGMA, DT, 0.5, "dirichlet").toarray()
    rows = _boundary_rows(4, 5)
    assert len(rows) == 14, f"expected the perimeter of a 4x5 grid, got {len(rows)} rows"
    for k in rows:
        assert a2[k, k] == pytest.approx(1.0), f"2D row {k} diagonal: {a2[k, k]}"
        assert a2[k].sum() == pytest.approx(1.0), f"2D row {k} is not the identity"


def test_the_2d_interior_is_untouched_by_the_dirichlet_branch():
    """A boundary branch that also moved interior rows would trade one defect for a worse one."""
    neumann = build_diffusion_matrix_2d((5, 5), DX, SIGMA, DT, 0.5, "neumann").toarray()
    dirichlet = build_diffusion_matrix_2d((5, 5), DX, SIGMA, DT, 0.5, "dirichlet").toarray()
    interior = [i * 5 + j for i in range(1, 4) for j in range(1, 4)]
    assert interior, "the 5x5 grid must have interior rows for this to check anything"
    for k in interior:
        assert np.abs(neumann[k] - dirichlet[k]).max() == 0.0, f"interior row {k} moved"


@pytest.mark.parametrize(
    ("builder", "args"),
    [
        (build_diffusion_matrix_1d, (5, DX, SIGMA, DT, 0.5)),
        (build_diffusion_matrix_2d, ((4, 4), DX, SIGMA, DT, 0.5)),
        (build_advection_matrix_1d, (5, DX, np.zeros(5), DT)),
    ],
)
def test_an_unrecognised_bc_type_is_refused_rather_than_defaulted(builder, args):
    """The fallthrough returned a valid matrix for a boundary condition the caller did not ask for.

    `build_advection_matrix_1d` is included even though it has no per-type branches to add: its
    `else` deliberately treats Neumann and Dirichlet alike, and an unchecked string took that path
    just as silently.
    """
    with pytest.raises(ValueError, match="bc_type must be one of"):
        builder(*args, "definitely_not_a_bc")


def test_every_declared_type_is_accepted_by_every_builder():
    """The positive control for the refusal above: a validator that rejected everything would pass it."""
    for bc_type in BC_TYPES:
        build_diffusion_matrix_1d(5, DX, SIGMA, DT, 0.5, bc_type)
        build_diffusion_matrix_2d((4, 4), DX, SIGMA, DT, 0.5, bc_type)
        build_advection_matrix_1d(5, DX, np.zeros(5), DT, bc_type)

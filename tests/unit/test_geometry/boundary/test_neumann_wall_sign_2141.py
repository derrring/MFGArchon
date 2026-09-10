"""The two applicators the SL solver holds disagree about an inhomogeneous Neumann value. #2141

RECORDED DEFECT, not a contract. `HJBSemiLagrangianSolver.__init__` builds two of them and uses each
at a different point: `bc_applicator = FDMApplicator(...)` for the ghost-cell work, and
`interp_bc_applicator = InterpolationApplicator(...)` for post-interpolation enforcement (#636).
Handed the same `neumann_bc(value=g)`, they do two different wrong things:

    FDMApplicator             imposes du/dn = -g at the LEFT wall and +g at the right
    InterpolationApplicator   ignores g entirely: g=0 and g=0.7 are BIT-IDENTICAL

The first is a sign error on one wall, the second is the value being dropped -- the same defect
#2294 records for the FEM natural-BC arm, in a second place.

Do not read the second row as "it imposes du/dn = 0". It does not, and the file's own xfail cell
prints the contradicting number. `InterpolationApplicator` defaults to `extrapolation_order=2`, so
the Neumann path takes `enforce_neumann_value_nd`'s zero-flux branch, `u[0] = (4*u[1] - u[2])/3` --
a vanishing SECOND derivative, not a vanishing normal one. What it actually imposes therefore depends
on the field: measured, -0.100000 / +0.566667 on this file's `quadratic`, -1.0 / +1.0 on a linear
ramp, and 0 only on a constant. The invariant that IS true of every field, and the one the test
asserts, is that the result does not depend on `g` at all. Getting this wrong points whoever retires
#2141 at the wrong target -- "make it impose du/dn = 0" is not the fix.

MEASURED AT THE APPLICATOR, NOT THROUGH A SOLVE, and that is the point of this file. Driving it
through `HJBSemiLagrangianSolver` reproduces `du/dn = -0.7` at N = 11 and then drifts with
resolution (-0.40, -0.95, -1.53, -2.10 at N = 21, 41, 61, 81 with dt scaled to h), because the
converged boundary gradient is whatever the interior dynamics leave once a homogeneous wall has been
imposed. Those numbers are a property of the fixture's dynamics; the ones below are a property of
the applicators, hold for every field tried, and involve no dt, no CFL and no mesh.

Retirement: each `xfail(strict=True)` cell reports XPASS the moment its applicator imposes the
requested derivative, and the recorded-defect tests fail carrying the instruction to delete them.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary import neumann_bc
from mfgarchon.geometry.boundary.applicator_fdm import FDMApplicator
from mfgarchon.geometry.boundary.applicator_interpolation import InterpolationApplicator

_G = 0.7
_N = 11
_X = np.linspace(0.0, 1.0, _N)
_DX = float(_X[1] - _X[0])
_SPACING = np.array([_DX])

#: Two fields with different boundary slopes. A single field cannot separate "the applicator imposed
#: the requested derivative" from "the applicator left a field that happened to have it".
_FIELDS = {"quadratic": _X**2, "flat": np.zeros(_N)}

_APPLICATORS = {
    "FDMApplicator": lambda: FDMApplicator(dimension=1),
    "InterpolationApplicator": lambda: InterpolationApplicator(dimension=1),
}


def _enforced(applicator_name: str, field_name: str, g: float) -> np.ndarray:
    applicator = _APPLICATORS[applicator_name]()
    field = _FIELDS[field_name].copy()
    return np.asarray(applicator.enforce_values(field, neumann_bc(value=g, dimension=1), spacing=_SPACING))


def _normal_derivative(field: np.ndarray, wall: str) -> float:
    """`du/dn`, with the OUTWARD normal: -x at the left wall, +x at the right.

    Getting this wrong is the whole subject of the file, so it is one function used by every
    assertion rather than a sign written out at each call site.
    """
    if wall == "left":
        return float(-(field[1] - field[0]) / _DX)
    return float((field[-1] - field[-2]) / _DX)


@pytest.mark.parametrize("field_name", sorted(_FIELDS))
def test_the_right_wall_of_the_fdm_applicator_is_correct(field_name):
    """POSITIVE CONTROL. One wall of one applicator does impose the requested derivative.

    Without it, every failure below is consistent with "this file measures `du/dn` wrongly" or "no
    applicator can express a boundary datum". This is the cell that says the harness can see a
    correct imposition when there is one, and it is why the left-wall result reads as a sign error
    rather than as a broken measurement.
    """
    enforced = _enforced("FDMApplicator", field_name, _G)

    assert _normal_derivative(enforced, "right") == pytest.approx(_G, abs=1e-12), (
        f"the FDM applicator's right wall no longer imposes du/dn = {_G}; the control this file "
        f"rests on is gone and the left-wall verdict below cannot be read as a sign error."
    )


@pytest.mark.parametrize("field_name", sorted(_FIELDS))
@pytest.mark.parametrize(
    "applicator_name",
    [
        pytest.param(
            name, marks=pytest.mark.xfail(strict=True, reason=f"#2141: {name} does not impose du/dn = g at the left")
        )
        for name in sorted(_APPLICATORS)
    ],
)
def test_every_applicator_imposes_the_requested_derivative_at_the_left_wall(applicator_name, field_name):
    """THE CONTRACT. `du/dn = g` is a statement about the OUTWARD normal, so it does not change sign
    with the wall. Retires by XPASS(strict) per applicator."""
    enforced = _enforced(applicator_name, field_name, _G)

    assert _normal_derivative(enforced, "left") == pytest.approx(_G, abs=1e-9), (
        f"{applicator_name} imposed du/dn = {_normal_derivative(enforced, 'left'):+.4f} at the left "
        f"wall for a requested {_G:+.4f}."
    )


@pytest.mark.parametrize("field_name", sorted(_FIELDS))
def test_the_fdm_applicator_still_inverts_the_sign_at_the_left_wall(field_name):
    """RECORDED DEFECT (#2141). Asserts the WRONG behaviour on purpose.

    `-g` exactly, not merely "not `g`": the magnitude is imposed correctly and only the sign is
    wrong, which is what makes this a one-character defect rather than a missing branch. Asserting
    only `!= g` would also pass while the value was dropped, and that is the OTHER applicator's
    defect, recorded separately below.
    """
    imposed = _normal_derivative(_enforced("FDMApplicator", field_name, _G), "left")

    if imposed != pytest.approx(-_G, abs=1e-12):
        pytest.fail(
            f"FDMApplicator's left wall now imposes du/dn = {imposed:+.6f} rather than the recorded "
            f"{-_G:+.6f}. If it is {_G:+.6f}, #2141 is fixed: delete this test and remove "
            f"FDMApplicator from the xfail list above."
        )


@pytest.mark.parametrize("field_name", sorted(_FIELDS))
def test_the_interpolation_applicator_still_drops_the_value_entirely(field_name):
    """RECORDED DEFECT (#2141, second half). Bit-identity is the discriminator.

    The value is not applied with the wrong sign here and not applied approximately; it never
    reaches an arithmetic operation, so enforcing `g = 0` and `g = 0.7` are the same computation.
    `enforce_neumann_value_nd(..., grad_value=0.0, ...)` is called with the literal, and the
    segment's own value is never read.
    """
    zero = _enforced("InterpolationApplicator", field_name, 0.0)
    requested = _enforced("InterpolationApplicator", field_name, _G)

    if not np.array_equal(zero, requested):
        pytest.fail(
            f"InterpolationApplicator now responds to the Neumann value: max|diff| = "
            f"{np.max(np.abs(zero - requested)):.6e}. Delete this test, and remove "
            f"InterpolationApplicator from the xfail list above."
        )


def test_the_two_applicators_disagree_and_that_is_the_defect():
    """The pair, asserted together, because the SL solver holds BOTH and uses each in turn.

    A reader who sees only the FDM result concludes the value is honoured with a sign bug; a reader
    who sees only the interpolation result concludes it is unimplemented. `HJBSemiLagrangianSolver`
    builds both in `__init__` and calls them at different points of one step, so the field it
    returns has been through the two conventions in sequence.
    """
    fdm = _normal_derivative(_enforced("FDMApplicator", "quadratic", _G), "left")
    interp_zero = _enforced("InterpolationApplicator", "quadratic", 0.0)
    interp_g = _enforced("InterpolationApplicator", "quadratic", _G)

    assert fdm == pytest.approx(-_G, abs=1e-12), (
        f"FDMApplicator's left wall imposes du/dn = {fdm:+.6f}, not the recorded {-_G:+.6f}. If it "
        f"is {_G:+.6f}, that half of #2141 is fixed and this test should be deleted with the other "
        f"recorded-defect tests."
    )
    assert np.array_equal(interp_zero, interp_g), (
        "InterpolationApplicator now responds to the Neumann value, so that half of #2141 is fixed "
        "and this test should be deleted with the other recorded-defect tests."
    )
    assert fdm != pytest.approx(_normal_derivative(interp_g, "left"), abs=1e-9), (
        "the two applicators now agree at the left wall; #2141 has been fixed or has changed shape, "
        "and this file's premise -- that one solver holds two conventions -- needs re-reading."
    )

"""The two applicators the SL solver holds impose the same inhomogeneous Neumann value. #2141

FIXED. `HJBSemiLagrangianSolver.__init__` builds two of them and uses each at a different
point: `bc_applicator = FDMApplicator(...)` for the ghost-cell work, and
`interp_bc_applicator = InterpolationApplicator(...)` for post-interpolation enforcement (#636).
Handed the same `neumann_bc(value=g)`, before #2141:

    FDMApplicator             imposed du/dn = -g at the low wall        <- the sign arm, fixed first
    InterpolationApplicator   ignored g entirely: g=0 and g=0.7 BIT-IDENTICAL   <- the value arm, fixed second

The sign arm was a one-character defect in `enforcement.py`'s non-zero-gradient branch, which wrote
the min wall as `u[1] - g*h` (du/dx = +g, hence du/dn = -g against the outward normal -x) where
both walls want `neighbour + g*h`. It is the same decision #1265 had already fixed on the sibling
ghost path in `a0f40fe1`; this file's two implementations had drifted apart.

The value-drop arm read the segment's value in `enforce_values` and dropped it at the
`_enforce_boundary_1d` dispatch, which called `enforce_neumann_value_nd` with a literal 0.0; it now
passes g and the spacing, and the SL call site passes its grid spacing (which this file cannot see:
`tests/unit/test_alg/test_hjb_sl_neumann_substep_2141.py` holds that half).

The sign arm was reached through the public API before the fix -- not only in this file. Measured
at `aad4aecd` on a 9-point 2-D grid, `HJBFDMSolver.solve_hjb_system` with `neumann_bc(value=0.7)`
returned du/dn = -0.700000 at the low wall and +0.700000 at the high. `problem.solve()` does NOT
reach it, because `FPFDMSolver` refuses an inhomogeneous Neumann value first (#1686), and a 1-D
population therefore measures zero calls and misses the defect entirely.

Do not read the second row as "it imposes du/dn = 0". It does not, and the file's own xfail cell
prints the contradicting number. `InterpolationApplicator` defaults to `extrapolation_order=2`, so
the Neumann path takes `enforce_neumann_value_nd`'s zero-flux branch, `u[0] = (4*u[1] - u[2])/3` --
a vanishing SECOND derivative, not a vanishing normal one. What it actually imposes therefore depends
on the field: measured, -0.100000 / +0.566667 on this file's `quadratic`, -1.0 / +1.0 on a linear
ramp, and 0 only on a constant. The invariant that IS true of every field, and the one the test
asserts, is that the result does not depend on `g` at all. Getting this wrong points whoever retires
#2141 at the wrong target -- "make it impose du/dn = 0" is not the fix.

MEASURED AT THE APPLICATOR, NOT THROUGH A SOLVE, and that is the point of this file. Driving the
pre-fix tree through `HJBSemiLagrangianSolver` reproduced `du/dn = -0.7` at N = 11 and then drifted
with resolution (-0.40, -0.95, -1.53, -2.10 at N = 21, 41, 61, 81 with dt scaled to h), because the
converged boundary gradient is whatever the interior dynamics leave once a homogeneous wall has been
imposed. Those numbers were a property of the fixture's dynamics -- they are recorded here as the
reason this file does not measure that way, and they are NOT a current description of the tree. The
assertions below are a property of the applicators, hold for every field tried, and involve no dt,
no CFL and no mesh.

Both halves retired through this file's own mechanism: the `xfail(strict=True)` cells reported
XPASS once their applicator imposed the requested derivative, and each recorded-defect test failed
carrying the instruction to delete itself.
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
@pytest.mark.parametrize("applicator_name", sorted(_APPLICATORS))
def test_every_applicator_imposes_the_requested_derivative_at_the_left_wall(applicator_name, field_name):
    """THE CONTRACT. `du/dn = g` is a statement about the OUTWARD normal, so it does not change sign
    with the wall. Both applicators were xfails here until their half of #2141 was fixed; each cell is
    now what catches its defect coming back -- the sign for `FDMApplicator`, the dropped value for
    `InterpolationApplicator`.
    """
    enforced = _enforced(applicator_name, field_name, _G)

    assert _normal_derivative(enforced, "left") == pytest.approx(_G, abs=1e-9), (
        f"{applicator_name} imposed du/dn = {_normal_derivative(enforced, 'left'):+.4f} at the left "
        f"wall for a requested {_G:+.4f}."
    )

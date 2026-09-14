"""Transport upwinds by the sign of the velocity: every explicit update coefficient is nonnegative (#2309).

`AdvectionOperator(scheme="upwind")`'s non-conservative path selected each one-sided difference by the
sign of the field it differentiated -- the HJB momentum rule, `gradient_upwind`. For transport that is
downwind wherever the field falls along the flow, so the explicit update ``m - dt * A m`` had a
coefficient of ``-CFL`` in every case #2309 measured, and blew up at constant velocity.

The oracle is the monotonicity property itself, independent of any implementation: for ``|v| dt / h
<= 1`` an upwind update is a nonnegative combination of the node and its upwind neighbour. Checked on
the assembled update matrix, column by column, so it does not depend on the fixture's profile.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.operators.differential.advection import AdvectionOperator

_N = 60
_H = 1.0 / _N
_X = np.arange(_N) * _H

_VELOCITIES = {
    "constant_positive": np.ones(_N),
    "constant_negative": -np.ones(_N),
    "sign_changing": np.sin(2 * np.pi * _X),
    "sign_changing_mirrored": -np.sin(2 * np.pi * _X),
}


@pytest.mark.parametrize("form", ["gradient", "divergence"])
@pytest.mark.parametrize("velocity", list(_VELOCITIES))
def test_every_explicit_update_coefficient_is_nonnegative(form: str, velocity: str):
    v = _VELOCITIES[velocity]
    operator = AdvectionOperator(v[None, :], [_H], (_N,), scheme="upwind", form=form)
    dt = 0.4 * _H / np.abs(v).max()
    update = np.eye(_N) - dt * np.column_stack([operator(column) for column in np.eye(_N).T])
    assert update.min() > -1e-12, (
        f"form={form}, v={velocity}: an update coefficient is {update.min():.3e} at CFL 0.4 -- the "
        f"operator is not upwinding by the velocity (#2309)"
    )

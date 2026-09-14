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


@pytest.mark.parametrize("form", ["gradient", "divergence"])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_a_constant_velocity_update_stays_between_each_node_and_its_upwind_neighbour(form: str, sign: float):
    """The same property on a smooth profile, where the selection is the one the operator makes on real data.

    The coefficient test above assembles the operator column by column from unit spikes, and a spike's one-sided
    differences fall on the upwind side under either rule, so for the gradient form it cannot tell a field-sign
    selection from a velocity-sign one. On a profile it can: at constant ``v`` the upwind update is a convex
    combination of ``m_i`` and its upwind neighbour, so the result stays between the two; a downwind selection
    on the falling side does not.
    """
    m = 1.0 + 0.5 * np.cos(2 * np.pi * _X) + 0.2 * np.sin(6 * np.pi * _X)
    operator = AdvectionOperator(np.full((1, _N), sign), [_H], (_N,), scheme="upwind", form=form)
    updated = m - 0.4 * _H * operator(m)
    upwind_neighbour = np.roll(m, 1) if sign > 0 else np.roll(m, -1)
    low, high = np.minimum(m, upwind_neighbour), np.maximum(m, upwind_neighbour)
    outside = float(np.maximum(low - updated, updated - high).max())
    assert outside < 1e-12, (
        f"form={form}, v={sign:+.0f}: the update leaves [m_i, m_upwind] by {outside:.3e} -- a downwind selection (#2309)"
    )

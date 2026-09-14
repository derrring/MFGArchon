"""Transport upwinds by the sign of the velocity: every explicit update coefficient is nonnegative (#2309).

`AdvectionOperator(scheme="upwind")`'s non-conservative path selected each one-sided difference by the
sign of the field it differentiated -- the HJB momentum rule, `gradient_upwind`. For transport that is
downwind wherever the field falls along the flow, so the explicit update ``m - dt * A m`` had a
coefficient of ``-CFL`` in every case #2309 measured, and blew up at constant velocity.

The oracle is the monotonicity property itself, independent of any implementation: for ``max|v| dt / h
<= 1`` an upwind update is a nonnegative combination of neighbouring values. Checked on the assembled
update matrix, so it does not depend on a profile.

The velocities must be able to express a divergence-form defect, which lives at faces where the velocity
changes sign between two nodes. Selecting a whole node flux by the sign of the face-averaged velocity
sends ``v_i m_i`` downwind where ``v_i < 0 <= (v_i + v_{i+1}) / 2``: ``-CFL`` at a diverging step, and
-0.0057 on ``sin(2 pi x + 0.3)``. On ``sin(2 pi x)`` at 60 nodes the zeros fall on nodes, no face averages
across a sign change, and that rule reads 0.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry import TensorProductGrid, dirichlet_bc
from mfgarchon.operators.differential.advection import AdvectionOperator
from mfgarchon.utils.numerical.tensor_calculus import advection

_N = 60
_H = 1.0 / _N
_X = np.arange(_N) * _H

_VELOCITIES = {
    "constant_positive": np.ones(_N),
    "constant_negative": -np.ones(_N),
    "diverging_step": np.where(_X < 0.5, -1.0, 1.0),
    "converging_step": np.where(_X < 0.5, 1.0, -1.0),
    "sign_changing_between_nodes": np.sin(2 * np.pi * _X + 0.3),
}


def _operator_matrix(v: np.ndarray, form: str) -> np.ndarray:
    return AdvectionOperator(v[None, :], [_H], (_N,), scheme="upwind", form=form).as_scipy_sparse().toarray()


def _tensor_calculus_matrix(v: np.ndarray, form: str) -> np.ndarray:
    return np.column_stack([advection(column, [v], [_H], form=form, method="upwind") for column in np.eye(_N).T])


@pytest.mark.parametrize("assemble", [_operator_matrix, _tensor_calculus_matrix], ids=["operator", "tensor_calculus"])
@pytest.mark.parametrize("form", ["gradient", "divergence"])
@pytest.mark.parametrize("velocity", list(_VELOCITIES))
def test_every_explicit_update_coefficient_is_nonnegative(assemble, form: str, velocity: str):
    v = _VELOCITIES[velocity]
    dt = 0.4 * _H / np.abs(v).max()
    update = np.eye(_N) - dt * assemble(v, form)
    assert update.min() > -1e-12, (
        f"{assemble.__name__}, form={form}, v={velocity}: an update coefficient is {update.min():.3e} at CFL 0.4 "
        f"-- the transport is not upwinding by the velocity (#2309)"
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


@pytest.mark.parametrize("form", ["gradient", "divergence"])
def test_an_outflow_wall_takes_no_boundary_value(form: str):
    """Where every characteristic leaves the domain, transport takes no data from the wall.

    ``v = x - 0.5`` flows out through both walls, so a Dirichlet value on the density must not change the
    operator anywhere. Padding the velocity with the density's condition breaks that: the ghost velocity
    ``2g - v_0`` points inward for ``g = 1``, and the wall node draws mass from the ghost.
    """
    x = np.linspace(0.0, 1.0, 11)
    m = np.exp(-10.0 * (x - 0.3) ** 2) + 0.1
    results = []
    for value in (0.0, 1.0):
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=dirichlet_bc(value, dimension=1)
        )
        operator = AdvectionOperator(
            (x - 0.5)[None, :], [x[1] - x[0]], (11,), scheme="upwind", form=form, bc=grid.get_boundary_conditions()
        )
        results.append(operator(m))
    gap = float(np.abs(results[1] - results[0]).max())
    assert gap == 0.0, f"form={form}: the outflow walls' Dirichlet value moved the operator by {gap:.3e} (#2309)"

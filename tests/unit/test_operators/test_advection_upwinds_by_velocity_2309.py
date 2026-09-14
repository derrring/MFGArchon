"""Transport upwinds by the sign of the velocity: monotone, consistent, conservative on a torus (#2309).

`AdvectionOperator(scheme="upwind")`'s non-conservative path selected each one-sided difference by the
sign of the field it differentiated -- the HJB momentum rule, `gradient_upwind`. For transport that is
downwind wherever the field falls along the flow, so the explicit update ``m - dt * A m`` had a
coefficient of ``-CFL`` in every case #2309 measured, and blew up at constant velocity.

Every oracle here is a property of transport, not of an implementation, and both consumers of the rule --
`AdvectionOperator` and `tensor_calculus.advection` -- answer to each:

- monotone: for ``2 max|v| dt / h <= 1`` the explicit update is a nonnegative combination of neighbours;
- consistent: the truncation error against the exact derivative falls at first order;
- conservative: on a torus the divergence form's columns sum to zero;
- characteristic: a wall the flow leaves through takes no boundary value.

The velocities must be able to express a divergence-form defect, which lives where the velocity changes
sign. Monotonicity alone does not separate the candidates there. Selecting a whole node flux by the sign of
the face velocity is non-monotone: ``-CFL`` at a velocity step. Splitting each node flux by that node's
velocity sign is monotone and inconsistent: its truncation error does not fall with ``N``. On
``sin(2 pi x)`` at 60 nodes the zeros fall on nodes and neither defect shows, so the sign change here sits
between nodes.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry import TensorProductGrid, dirichlet_bc, periodic_bc
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
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


def _apply_operator(v: np.ndarray, h: float, form: str, m: np.ndarray, bc=None) -> np.ndarray:
    return AdvectionOperator(v[None, :], [h], (v.size,), scheme="upwind", form=form, bc=bc)(m)


def _apply_tensor_calculus(v: np.ndarray, h: float, form: str, m: np.ndarray, bc=None) -> np.ndarray:
    return advection(m, [v], [h], form=form, method="upwind", bc=bc)


_CONSUMERS = pytest.mark.parametrize(
    "apply", [_apply_operator, _apply_tensor_calculus], ids=["operator", "tensor_calculus"]
)


def _matrix(apply, v: np.ndarray, form: str, bc=None) -> np.ndarray:
    """The operator's own ``as_scipy_sparse``, which refused upwind while the operator was nonlinear (#1981)."""
    if apply is _apply_operator:
        return (
            AdvectionOperator(v[None, :], [_H], (v.size,), scheme="upwind", form=form, bc=bc)
            .as_scipy_sparse()
            .toarray()
        )
    return np.column_stack([apply(v, _H, form, column, bc) for column in np.eye(v.size).T])


@_CONSUMERS
@pytest.mark.parametrize("form", ["gradient", "divergence"])
@pytest.mark.parametrize("velocity", list(_VELOCITIES))
def test_every_explicit_update_coefficient_is_nonnegative(apply, form: str, velocity: str):
    v = _VELOCITIES[velocity]
    dt = 0.4 * _H / np.abs(v).max()
    update = np.eye(_N) - dt * _matrix(apply, v, form)
    assert update.min() > -1e-12, (
        f"{apply.__name__}, form={form}, v={velocity}: an update coefficient is {update.min():.3e} at CFL 0.4 "
        f"-- the transport is not upwinding by the velocity (#2309)"
    )


@_CONSUMERS
@pytest.mark.parametrize("form", ["gradient", "divergence"])
def test_the_truncation_error_falls_at_first_order_across_a_sign_change(apply, form: str):
    """Against the exact derivative on a torus, with ``v = sin(2 pi x + 0.3)`` changing sign between nodes.

    First order quarters the error from 100 to 400 nodes. Node-velocity flux splitting does not reduce it at
    all: its error is ``O(1)`` beside each zero of ``v`` at every resolution.
    """
    errors = []
    for n in (100, 400):
        h = 1.0 / n
        x = np.arange(n) * h
        v, dv = np.sin(2 * np.pi * x + 0.3), 2 * np.pi * np.cos(2 * np.pi * x + 0.3)
        m, dm = 1.0 + 0.5 * np.cos(2 * np.pi * x), -np.pi * np.sin(2 * np.pi * x)
        exact = v * dm if form == "gradient" else dv * m + v * dm
        errors.append(float(np.abs(apply(v, h, form, m) - exact).max()))
    assert errors[1] < errors[0] / 3.0, (
        f"{apply.__name__}, form={form}: max truncation error {errors[0]:.3e} at N=100 and {errors[1]:.3e} at "
        f"N=400 -- not first order (#2309)"
    )


@_CONSUMERS
@pytest.mark.parametrize("bc", [None, periodic_bc(dimension=1)], ids=["bc_none", "periodic_bc"])
def test_the_divergence_form_conserves_mass_on_a_torus(apply, bc):
    """``sum(div(v m)) = 0`` on a torus, for an explicit periodic condition as for none."""
    columns = _matrix(apply, _VELOCITIES["sign_changing_between_nodes"], "divergence", bc).sum(axis=0)
    assert float(np.abs(columns).max()) < 1e-12, (
        f"{apply.__name__}, bc={bc!r}: a column of the divergence sums to {float(np.abs(columns).max()):.3e} -- "
        f"mass is created on a torus"
    )


@pytest.mark.parametrize("form", ["gradient", "divergence"])
def test_a_constant_velocity_update_stays_between_each_node_and_its_upwind_neighbour(form: str):
    """The monotonicity property on a smooth profile, where the selection is the one made on real data.

    Assembling from unit spikes cannot tell a field-sign selection from a velocity-sign one in the gradient
    form: a spike's one-sided differences fall on the upwind side under either rule. On a profile it can: at
    constant ``v`` the upwind update is a convex combination of ``m_i`` and its upwind neighbour, so the
    result stays between the two; a downwind selection on the falling side does not.
    """
    m = 1.0 + 0.5 * np.cos(2 * np.pi * _X) + 0.2 * np.sin(6 * np.pi * _X)
    for apply in (_apply_operator, _apply_tensor_calculus):
        for sign in (1.0, -1.0):
            updated = m - 0.4 * _H * apply(np.full(_N, sign), _H, form, m)
            upwind_neighbour = np.roll(m, 1) if sign > 0 else np.roll(m, -1)
            low, high = np.minimum(m, upwind_neighbour), np.maximum(m, upwind_neighbour)
            outside = float(np.maximum(low - updated, updated - high).max())
            assert outside < 1e-12, (
                f"{apply.__name__}, form={form}, v={sign:+.0f}: the update leaves [m_i, m_upwind] by "
                f"{outside:.3e} -- a downwind selection (#2309)"
            )


def _uniform_dirichlet(value: float) -> BoundaryConditions:
    return dirichlet_bc(value, dimension=1)


def _dirichlet_by_segment_and_default(value: float) -> BoundaryConditions:
    return BoundaryConditions(
        dimension=1,
        segments=[BCSegment(name="low", bc_type=BCType.DIRICHLET, value=value, boundary="x_min")],
        default_bc=BCType.DIRICHLET,
        default_value=value,
    )


@_CONSUMERS
@pytest.mark.parametrize(
    "walls", [_uniform_dirichlet, _dirichlet_by_segment_and_default], ids=["uniform", "segment_and_default"]
)
def test_an_outflow_wall_takes_no_boundary_value(apply, walls):
    """Where every characteristic leaves the domain, transport takes no data from the wall.

    ``v = x - 0.5`` flows out through both walls, so a Dirichlet value on the density must not change the
    operator anywhere. Padding the velocity with the density's condition breaks that: the ghost velocity
    ``2g - v_0`` points inward for ``g = 1``, and the wall node draws mass from the ghost. The second
    construction states the high wall through ``default_bc``, which the velocity's condition must rewrite too.
    """
    x = np.linspace(0.0, 1.0, 11)
    m = np.exp(-10.0 * (x - 0.3) ** 2) + 0.1
    for form in ("gradient", "divergence"):
        results = []
        for value in (0.0, 1.0):
            grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=walls(value))
            results.append(apply(x - 0.5, x[1] - x[0], form, m, grid.get_boundary_conditions()))
        gap = float(np.abs(results[1] - results[0]).max())
        assert gap == 0.0, (
            f"{apply.__name__}, form={form}: the outflow walls' Dirichlet value moved the operator by {gap:.3e} (#2309)"
        )

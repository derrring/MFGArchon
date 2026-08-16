"""A provider may sit on `alpha` or `beta`, not only on `value`.

The impermeable wall of a Fokker-Planck equation is Robin with `alpha` the outward normal drift,
`D_pH(x, grad u) . n` -- knowable only from the current iterate, which is exactly what a provider
is for, and living on `alpha` rather than on `value`. `with_resolved_providers` resolved `value`
alone and coerced it with `float()`, so the one coefficient that needs the iterate could not be
supplied and could not have been a field if it were.

This is the same defect one layer up from `ResolvedBC`'s scalar coefficients (#1957): the protocol
already declares `compute() -> float | NDArray[np.floating]`, and each intermediate layer in turn
narrowed it back to a scalar on one field.

**The gate is the part that bites.** `with_resolved_providers` returns `self` unchanged when
`has_providers()` is False, so widening the resolver without widening the gate leaves a provider on
`alpha` silently unresolved -- the caller gets the provider *object* where a number belongs, and
the failure surfaces far downstream as a type error inside a ghost formula.
"""

from __future__ import annotations

from typing import Any

import pytest

import numpy as np

from mfgarchon.geometry.boundary import (
    AdjointConsistentProvider,
    BCSegment,
    BCType,
    BoundaryConditions,
    no_flux_bc,
)
from mfgarchon.geometry.grids import TensorProductGrid


class _ScalarProvider:
    """Minimal provider: returns a number that depends on the state, so a test can tell a
    resolved value from an unresolved one and from a hardcoded constant."""

    def __init__(self, factor: float) -> None:
        self._factor = factor

    def compute(self, state: dict[str, Any]) -> float:
        return self._factor * float(state["sigma"])


class _FieldProvider:
    """Returns one value per wall. The protocol has always declared
    `compute() -> float | NDArray[np.floating]`; this is the half that never reached a caller."""

    def compute(self, state: dict[str, Any]) -> np.ndarray:
        return -np.gradient(np.asarray(state["U_current"]))[[0, -1]]


@pytest.fixture
def state() -> dict[str, Any]:
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx=[10], boundary_conditions=no_flux_bc(dimension=1))
    x = np.linspace(0.0, 1.0, 11)
    return {
        "m_current": 0.5 + 0.3 * np.cos(np.pi * x),
        "U_current": 0.5 * x**2,
        "sigma": 0.4,
        "geometry": grid,
    }


def _one_segment(**kwargs) -> BoundaryConditions:
    defaults = {"name": "wall", "bc_type": BCType.ROBIN, "value": 0.0, "alpha": 1.0, "beta": 0.0}
    return BoundaryConditions(
        segments=[BCSegment(**(defaults | kwargs))],
        dimension=1,
        default_bc=BCType.NO_FLUX,
        domain_bounds=np.array([[0.0, 1.0]]),
    )


# =============================================================================
# The gate and the resolver must cover the same fields
# =============================================================================


@pytest.mark.parametrize("field", ["value", "alpha", "beta"])
def test_the_gate_sees_a_provider_on_any_resolved_field(field):
    """`has_providers()` guards the fast path. If it covered fewer fields than
    `with_resolved_providers` resolves, a provider on the uncovered field would be returned
    unresolved -- and `False` here is indistinguishable from "there is nothing to do"."""
    assert _one_segment(**{field: _ScalarProvider(2.0)}).has_providers()


@pytest.mark.parametrize("field", ["value", "alpha", "beta"])
def test_a_provider_on_any_field_is_resolved_to_a_number(field, state):
    """`2.0 * sigma` with `sigma = 0.4` is 0.8 -- not a value any default supplies, so this
    separates "resolved" from "left alone" from "replaced by a constant"."""
    resolved = _one_segment(**{field: _ScalarProvider(2.0)}).with_resolved_providers(state)

    assert getattr(resolved.segments[0], field) == pytest.approx(0.8)


def test_all_three_fields_resolve_in_one_pass(state):
    """A segment may carry three providers at once. Resolving one and dropping the others would
    pass every single-field test above."""
    bc = _one_segment(value=_ScalarProvider(1.0), alpha=_ScalarProvider(2.0), beta=_ScalarProvider(3.0))

    seg = bc.with_resolved_providers(state).segments[0]

    assert (seg.value, seg.alpha, seg.beta) == pytest.approx((0.4, 0.8, 1.2))


# =============================================================================
# The field-valued half of the protocol
# =============================================================================


def test_a_coefficient_provider_may_return_a_field(state):
    """The reason the `float()` coercion had to go. The Robin coefficient of a reflecting FP wall
    varies along the boundary; a scalar cannot hold it."""
    seg = _one_segment(alpha=_FieldProvider()).with_resolved_providers(state).segments[0]

    assert isinstance(seg.alpha, np.ndarray), "a field coefficient must survive resolution"
    np.testing.assert_allclose(seg.alpha, -np.gradient(state["U_current"])[[0, -1]])


def test_a_field_on_value_is_no_longer_flattened(state):
    """`value` was coerced with `float()`, which raises on a 2-element array rather than
    truncating -- so this asserts the coercion is gone rather than that it now rounds."""
    seg = _one_segment(value=_FieldProvider()).with_resolved_providers(state).segments[0]

    assert isinstance(seg.value, np.ndarray)


# =============================================================================
# Nothing that worked before may have moved
# =============================================================================


def test_the_shipped_adjoint_consistent_config_is_unchanged(state):
    """The one provider configuration this library documents (#574, #625). Captured before the
    change and asserted to the last digit: widening the resolver must not perturb the field it
    already resolved."""
    bc = BoundaryConditions(
        segments=[
            BCSegment(
                name="l",
                bc_type=BCType.ROBIN,
                alpha=0.0,
                beta=1.0,
                value=AdjointConsistentProvider(side="left", sigma=0.4),
                boundary="x_min",
            ),
            BCSegment(
                name="r",
                bc_type=BCType.ROBIN,
                alpha=0.0,
                beta=1.0,
                value=AdjointConsistentProvider(side="right", sigma=0.4),
                boundary="x_max",
            ),
        ],
        dimension=1,
        default_bc=BCType.NO_FLUX,
        domain_bounds=np.array([[0.0, 1.0]]),
    )

    lo, hi = bc.with_resolved_providers(state).segments

    assert lo.value == pytest.approx(-0.0148194617, abs=1e-10)
    assert hi.value == pytest.approx(0.0566762920, abs=1e-10)
    assert (lo.alpha, lo.beta) == (0.0, 1.0), "coefficients without providers are untouched"


def test_a_bc_with_no_providers_is_returned_unchanged(state):
    """The fast path is an identity, not a copy. Losing it would be a silent per-Picard-step
    allocation on every BC in the library."""
    bc = _one_segment(value=0.7, alpha=1.0, beta=0.3)

    assert bc.with_resolved_providers(state) is bc


def test_a_segment_without_providers_survives_beside_one_that_has_them(state):
    """Mixed BCs resolve segment by segment. A rebuild that dropped or reordered the untouched
    segment would not show up in any single-segment test."""
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="dynamic", bc_type=BCType.ROBIN, value=_ScalarProvider(2.0), boundary="x_min"),
            BCSegment(name="static", bc_type=BCType.NO_FLUX, value=0.0, boundary="x_max"),
        ],
        dimension=1,
        default_bc=BCType.NO_FLUX,
        domain_bounds=np.array([[0.0, 1.0]]),
    )

    resolved = bc.with_resolved_providers(state)

    assert [s.name for s in resolved.segments] == [s.name for s in bc.segments]
    assert resolved.segments[0].value == pytest.approx(0.8)
    assert resolved.segments[1].value == 0.0

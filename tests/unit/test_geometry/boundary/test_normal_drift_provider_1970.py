"""`NormalDriftProvider` supplies the `alpha` of a Fokker-Planck no-flux wall.

An impermeable wall is `J . n = 0` with `J = m*v - D*grad(m)`, i.e. `v_n*m - D*d_n m = 0` —
Robin in `m` with `alpha = v_n`, `beta = -D`, `g = 0`. Imposing `d_n m = 0` instead is the same
condition only when the drift is tangential at the wall. Measured on this package at a wall-normal
drift of 3.2, the non-conservative assembly loses **5.4% of the mass**.

`v_n` is a functional of the coupled solution, `v = -c*grad(U)`, so it is known only per Picard
iterate — which is what a provider is for, and why this one sits on `alpha` rather than on
`value`: `value` is the homogeneous right-hand side, zero for an impermeable wall.

**The oracle is the flux, not another code path.** Every assertion below evaluates
`v_n*m_b - D*d_n m` on the ghost the shipped API returned.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    NormalDriftProvider,
    no_flux_bc,
    pad_array_with_ghosts,
)
from mfgarchon.geometry.grids import TensorProductGrid

_N = 11
_X = np.linspace(0.0, 1.0, _N)
_DX = _X[1] - _X[0]
_M = 0.5 + 0.3 * np.cos(np.pi * _X)  # asymmetric about the domain centre on purpose
_SIGMA = 0.3
_D = _SIGMA**2 / 2
_C = 1.0


@pytest.fixture
def grid() -> TensorProductGrid:
    return TensorProductGrid(bounds=[(0.0, 1.0)], Nx=[_N - 1], boundary_conditions=no_flux_bc(dimension=1))


def _state(u: np.ndarray, grid: TensorProductGrid) -> dict:
    return {"U_current": u, "geometry": grid, "drift_coefficient": _C}


# =============================================================================
# The sign convention, against a field whose drift is known in closed form
# =============================================================================


@pytest.mark.parametrize(
    ("label", "slope"),
    [("rising", 2.0), ("falling", -3.0), ("flat", 0.0)],
)
def test_the_drift_is_projected_onto_the_outward_normal(label, slope, grid):
    """`v = -c*dU/dx`, and the outward normal is `-x` at the low wall, `+x` at the high one, so
    the two walls must report **opposite** signs for a linear `U`.

    Both directions are parametrised because a sign error that flipped the whole convention would
    pass a one-signed test, and the flat row rules out an implementation that returns a constant.
    """
    u = slope * _X
    v_x = -_C * slope

    low = NormalDriftProvider("left").compute(_state(u, grid))
    high = NormalDriftProvider("right").compute(_state(u, grid))

    assert low == pytest.approx(-v_x), "outward normal at the low wall is -x"
    assert high == pytest.approx(v_x), "outward normal at the high wall is +x"


def test_the_two_walls_see_different_drifts_when_the_field_is_not_linear(grid):
    """Control for the parametrisation above, which uses linear fields where the two walls differ
    only in sign. On `U = 0.5x^2` the magnitudes differ too, so an implementation that computed
    one derivative and negated it for the other wall would fail here."""
    u = 0.5 * _X**2

    low = NormalDriftProvider("left").compute(_state(u, grid))
    high = NormalDriftProvider("right").compute(_state(u, grid))

    assert abs(low) != pytest.approx(abs(high))


# =============================================================================
# End to end: the resolved coefficient must zero the flux
# =============================================================================


def _no_flux_bc_with_provider() -> BoundaryConditions:
    return BoundaryConditions(
        segments=[
            BCSegment(
                name="lo",
                bc_type=BCType.ROBIN,
                value=0.0,
                alpha=NormalDriftProvider("left", _C),
                beta=-_D,
                boundary="x_min",
            ),
            BCSegment(
                name="hi",
                bc_type=BCType.ROBIN,
                value=0.0,
                alpha=NormalDriftProvider("right", _C),
                beta=-_D,
                boundary="x_max",
            ),
        ],
        dimension=1,
        default_bc=BCType.NO_FLUX,
        domain_bounds=np.array([[0.0, 1.0]]),
    )


@pytest.mark.parametrize(
    ("label", "u_fn"),
    [
        ("linear-rising", lambda x: 2.0 * x),
        ("linear-falling", lambda x: -3.0 * x),
        ("parabolic", lambda x: 0.5 * x**2),
        ("constant", lambda x: np.full_like(x, 7.0)),
    ],
)
@pytest.mark.parametrize("wall", ["low", "high"])
def test_the_wall_flux_is_zero(label, u_fn, wall, grid):
    """The whole point, measured on the shipped padding path rather than on the provider alone.

    `constant` is the degenerate row where the drift vanishes and the condition collapses to
    `d_n m = 0` — it passes under either convention, and is here so that the other three rows are
    visibly the discriminating ones rather than the whole test.
    """
    resolved = _no_flux_bc_with_provider().with_resolved_providers(_state(u_fn(_X), grid))
    padded = pad_array_with_ghosts(_M, resolved, spacing=_DX)

    if wall == "low":
        v_n, ghost, interior = resolved.segments[0].alpha, padded[0], _M[0]
    else:
        v_n, ghost, interior = resolved.segments[1].alpha, padded[-1], _M[-1]

    flux = v_n * (ghost + interior) / 2 - _D * (ghost - interior) / _DX

    assert flux == pytest.approx(0.0, abs=1e-12)


def test_a_pointwise_neumann_wall_leaks_where_this_one_does_not(grid):
    """The control that makes the test above mean something. `d_n m = 0` is the condition this
    replaces; on the same field it leaves a flux proportional to `m_wall * v_n`, so a provider
    that silently returned zero would pass every assertion above and fail here."""
    u = 2.0 * _X
    resolved = _no_flux_bc_with_provider().with_resolved_providers(_state(u, grid))
    v_n = resolved.segments[0].alpha

    mirrored = pad_array_with_ghosts(_M, no_flux_bc(dimension=1), spacing=_DX)
    leaked = v_n * (mirrored[0] + _M[0]) / 2 - _D * (mirrored[0] - _M[0]) / _DX

    assert abs(leaked) > 1e-3, "the pointwise wall must leak here, or the comparison is vacuous"


# =============================================================================
# No default for a mathematical parameter, and no silent nD
# =============================================================================


def test_the_drift_coefficient_has_no_default(grid):
    """`c` is the Hamiltonian's control law (#1420), and a defaulted one silently rescales the
    wall's drift — the wrong answer still converges."""
    with pytest.raises(KeyError, match="drift_coefficient"):
        NormalDriftProvider("left").compute({"U_current": _X, "geometry": grid})


@pytest.mark.parametrize("key", ["U_current", "geometry"])
def test_a_missing_state_key_is_named(key, grid):
    state = _state(_X, grid)
    del state[key]

    with pytest.raises(KeyError, match=key):
        NormalDriftProvider("left", _C).compute(state)


def test_an_nd_wall_is_refused_rather_than_guessed(grid):
    """An nD outward normal needs the geometry's gradient operator — the same limit
    `AdjointConsistentProvider` states (#624). Guessing an axis here would reintroduce the
    axis-versus-outward-normal confusion #1907 removed."""
    with pytest.raises(NotImplementedError, match="624"):
        NormalDriftProvider("y_min", _C).compute(_state(_X, grid))


def test_an_unknown_side_is_refused_at_construction(grid):
    with pytest.raises(ValueError, match="side must be one of"):
        NormalDriftProvider("northwest", _C)


def test_the_constructor_coefficient_wins_over_the_state(grid):
    """Both channels exist; a caller that passes `c` explicitly must not have it silently
    overridden by whatever the iterator happens to have put in the state."""
    value = NormalDriftProvider("left", 2.0).compute({**_state(2.0 * _X, grid), "drift_coefficient": 99.0})

    assert value == pytest.approx(4.0), "c=2 with dU/dx=2 gives v_n = +c*dU/dx = 4 at the low wall"

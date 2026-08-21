"""There is no default Robin, and a uniform BC on the per-face path forwards rather than rebuilds.

`_update_ghosts_mixed` is reached for a face no segment claims. It used to synthesize
`BCSegment(name="__default__", bc_type=default_type, value=bc.default_value)` — carrying `bc_type`
and `value` and **nothing else**, so `alpha`, `beta` and any callable value were dropped and
`BCSegment`'s own defaults (`alpha=1.0, beta=0.0`) took their place.

**`beta = 0` IS Dirichlet.** So a defaulted Robin did not become an approximate Robin; it became a
different boundary condition, bit for bit identical to `dirichlet_bc(value=g)`, with nothing in the
output to say so. Measured before the fix: `robin_bc(alpha=1, beta=1, value=3)` gave `[5. 1. 2. 3.]`
against the correct `[1.097561 …]`, and independent of the alpha and beta actually passed.

TWO SITUATIONS ARRIVE HERE AND ONLY ONE HAS AN ANSWER
-----------------------------------------------------
(a) A **uniform** BC took this path — one segment, no boundary restriction. Forward it: the
    information is present. This is not the `bc.segments[0]` fallback the surrounding comment
    rejects; that one fired on MIXED BCs, where `[0]` means *highest priority* and handed every
    unclaimed wall the exit (measured: 4 of 4 walls got the Dirichlet ghost). `is_uniform` is
    exactly the case where that cannot happen.

(b) A **mixed** BC with an unclaimed face. Nothing to forward — `BoundaryConditions` carries
    `default_bc` and `default_value` and **no** `default_alpha` / `default_beta`. So a Robin
    default is an incomplete declaration, and this refuses rather than guessing, continuing
    #1100's ruling (`_resolve_default_bc` already raises when `default_bc` is unset).
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary import dirichlet_bc, neumann_bc, no_flux_bc, robin_bc
from mfgarchon.geometry.boundary.applicator_fdm import pad_array_with_ghosts
from mfgarchon.geometry.boundary.conditions import BoundaryConditions
from mfgarchon.geometry.boundary.types import BCSegment, BCType, BoundaryFace

_FIELD = np.array([1.0, 2.0, 3.0])
_DX = 0.05


def _pad(bc):
    return np.asarray(pad_array_with_ghosts(_FIELD, bc, ghost_depth=1, spacing=_DX)).ravel()


def _mixed_with_default(default_bc, default_value=0.0):
    """One claimed face, one unclaimed — the only way to reach the `__default__` branch."""
    claimed = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=5.0, boundary=BoundaryFace(0, "min"))
    return BoundaryConditions(dimension=1, segments=[claimed], default_bc=default_bc, default_value=default_value)


def test_a_defaulted_robin_is_refused():
    """The load-bearing test. `beta = 0` is Dirichlet, so defaulting a Robin changes the boundary
    condition's TYPE — which is not an approximation a caller can be expected to notice."""
    with pytest.raises(NotImplementedError) as excinfo:
        _pad(_mixed_with_default(BCType.ROBIN))

    message = str(excinfo.value)
    assert "alpha" in message, "must name what is missing"
    assert "beta" in message
    assert "DIRICHLET" in message, "and what the silent default would have turned it into"


@pytest.mark.parametrize("default_bc", [BCType.NO_FLUX, BCType.NEUMANN, BCType.DIRICHLET])
def test_defaults_that_need_no_coefficients_still_work(default_bc):
    """The refusal must be specific to ROBIN. A blanket refusal would break every mixed BC, and
    would pass the test above just as well — which is why this one exists."""
    padded = _pad(_mixed_with_default(default_bc))
    assert padded.shape == (5,)
    assert np.all(np.isfinite(padded))


@pytest.mark.parametrize(
    "bc",
    [
        pytest.param(robin_bc(dimension=1, alpha=1.0, beta=1.0, value=3.0), id="robin"),
        pytest.param(neumann_bc(dimension=1, value=lambda t: 2.0), id="neumann-callable"),
        pytest.param(neumann_bc(dimension=1, value=2.0), id="neumann-scalar"),
        pytest.param(dirichlet_bc(dimension=1, value=5.0), id="dirichlet"),
        pytest.param(no_flux_bc(dimension=1), id="no-flux"),
    ],
)
def test_a_uniform_bc_gives_the_same_answer_on_either_path(bc, monkeypatch):
    """Forwarding, asserted as the property that matters: the two paths must AGREE.

    Robin and a callable Neumann were the two that diverged — the dropped `alpha`/`beta` and the
    dropped callable respectively. Scalar Neumann, Dirichlet and no-flux agreed even before the
    fix, because their ghost formulas read none of the dropped fields; they are here so that a
    regression narrowing the forward to one BC type does not pass.
    """
    from mfgarchon.geometry.boundary import applicator_fdm

    expected = _pad(bc)

    # Force the per-face path for a BC the dispatch would send down the uniform one.
    original = applicator_fdm.PreallocatedGhostBuffer.update_ghosts

    def forced(self, time=0.0):
        self._update_ghosts_mixed(self._boundary_conditions, time)

    monkeypatch.setattr(applicator_fdm.PreallocatedGhostBuffer, "update_ghosts", forced)
    via_per_face = _pad(bc)
    monkeypatch.setattr(applicator_fdm.PreallocatedGhostBuffer, "update_ghosts", original)

    np.testing.assert_allclose(
        via_per_face,
        expected,
        rtol=0,
        atol=0,
        err_msg="the per-face path disagrees with the uniform path for a BC that has one segment",
    )

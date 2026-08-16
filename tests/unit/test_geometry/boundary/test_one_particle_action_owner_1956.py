"""One owner for `BCType -> particle action`, and the four cells where the two copies disagreed.

`MeshfreeApplicator.apply_particles` and `ParticleApplicator.apply` each wrote the mapping out.
They agreed on DIRICHLET, NEUMANN, NO_FLUX and PERIODIC, and diverged on the other four members:

    REFLECTING           meshfree raised          particle reflected
    ROBIN(alpha=1,beta=0) meshfree absorbed       particle reflected
    EXTRAPOLATION_LINEAR  meshfree raised         particle reflected
    EXTRAPOLATION_QUAD    meshfree raised         particle reflected

The ROBIN row is the one that mattered: the same `BoundaryConditions` object made one path build
an absorbing wall and the other an impermeable one.

**These tests do not compare the two paths against each other.** Once both route through
`particle_action_for_bc_type`, agreement is tautological and would pass over a broken owner. They
pin the owner's table against the physics, and each repaired cell against the behaviour recorded
before the consolidation. #1956
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry import Hyperrectangle
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
from mfgarchon.geometry.boundary.applicator_meshfree import MeshfreeApplicator
from mfgarchon.geometry.boundary.applicator_particle import (
    ParticleApplicator,
    particle_action_for_bc_type,
)

_BOUNDS = [(0.0, 1.0), (0.0, 1.0)]

# One interior particle and four that have left, one through each wall. The interior one is what
# makes "absorbing" distinguishable from "everything vanished"; the four are what make reflecting
# distinguishable from wrapping, since a reflected 1.2 comes back to 0.8 and a wrapped one to 0.2.
_PARTICLES = np.array([[0.5, 0.5], [1.2, 0.3], [-0.1, 0.4], [0.0, 0.7], [0.6, 1.05]])


def _uniform(bc_type: BCType, alpha: float = 1.0, beta: float = 0.0) -> BoundaryConditions:
    return BoundaryConditions(
        segments=[BCSegment(name="wall", bc_type=bc_type, value=0.0, alpha=alpha, beta=beta)],
        dimension=2,
        default_bc=bc_type,
        domain_bounds=np.array(_BOUNDS),
    )


def _meshfree(bc: BoundaryConditions) -> np.ndarray:
    return MeshfreeApplicator(Hyperrectangle([[0.0, 1.0], [0.0, 1.0]])).apply_particles(_PARTICLES.copy(), bc)


def _particle(bc: BoundaryConditions) -> np.ndarray:
    return ParticleApplicator().apply(_PARTICLES.copy(), bc, _BOUNDS)[0]


# =============================================================================
# The owner's table, pinned against the physics rather than against a caller
# =============================================================================

_EXPECTED = {
    BCType.DIRICHLET: "absorbing",
    BCType.NEUMANN: "reflecting",
    BCType.NO_FLUX: "reflecting",
    BCType.REFLECTING: "reflecting",
    BCType.PERIODIC: "periodic",
    BCType.ROBIN: "absorbing",  # at the default coefficients alpha=1, beta=0
}
_NO_PARTICLE_MEANING = {BCType.EXTRAPOLATION_LINEAR, BCType.EXTRAPOLATION_QUADRATIC}


def test_the_table_covers_every_bctype_member():
    """Derived from the enum, not listed. Adding a `BCType` member fails this until someone
    decides what a particle does when it reaches that kind of wall -- which is the property a
    hand-maintained list cannot have, and the reason the two copies could drift unnoticed."""
    assert set(_EXPECTED) | _NO_PARTICLE_MEANING == set(BCType), (
        "a BCType member is neither given a particle action nor declared to have none"
    )


@pytest.mark.parametrize(("bc_type", "expected"), sorted(_EXPECTED.items(), key=lambda kv: kv[0].name))
def test_each_bc_type_maps_to_its_particle_action(bc_type, expected):
    assert particle_action_for_bc_type(bc_type, alpha=1.0, beta=0.0) == expected


@pytest.mark.parametrize("bc_type", sorted(_NO_PARTICLE_MEANING, key=lambda t: t.name))
def test_a_field_truncation_rule_is_refused_and_named(bc_type):
    """`EXTRAPOLATION_*` says how to continue a field past a cut domain and carries no boundary
    datum. The segment-aware path used to send it to a catch-all `else` and reflect, which answers
    a question nobody asked. The message must name the type -- a bare raise leaves the caller
    unable to tell which of its segments is the problem."""
    with pytest.raises(ValueError, match=bc_type.name):
        particle_action_for_bc_type(bc_type)


# =============================================================================
# Robin: the coefficients decide, and they are not optional
# =============================================================================


@pytest.mark.parametrize(
    ("alpha", "beta", "expected"),
    [
        (1.0, 0.0, "absorbing"),  # alpha*u = g is Dirichlet
        (0.0, 1.0, "reflecting"),  # beta*du/dn = g is a flux condition
        (1.0, 1.0, "reflecting"),  # genuinely mixed: conserve mass
        (3.0, 0.0, "absorbing"),  # the scale of alpha does not enter; only beta == 0 does
    ],
)
def test_robin_dispatches_on_its_coefficients(alpha, beta, expected):
    assert particle_action_for_bc_type(BCType.ROBIN, alpha, beta) == expected


def test_robin_without_coefficients_refuses_rather_than_assuming():
    """A default here would be the #1558 failure a third time. `alpha=1, beta=0` is Dirichlet, so
    any implicit default silently turns an unspecified Robin wall into an absorbing one."""
    with pytest.raises(ValueError, match="needs alpha and beta"):
        particle_action_for_bc_type(BCType.ROBIN)


# =============================================================================
# The four repaired cells, each against the behaviour recorded before the merge
# =============================================================================


def test_meshfree_reflecting_no_longer_raises():
    """Measured before this change: `ValueError: Unsupported BC type for particles:
    BCType.REFLECTING` -- for the member `BCType`'s own docstring calls the particle spelling of
    an impermeable wall."""
    result = _meshfree(_uniform(BCType.REFLECTING))

    assert len(result) == len(_PARTICLES), "an impermeable wall absorbs nothing"
    assert np.all(Hyperrectangle([[0.0, 1.0], [0.0, 1.0]]).contains(result)), "every particle is back inside"


def test_particle_robin_at_default_coefficients_absorbs():
    """Measured before this change: reflected, because `ROBIN` was not in the dispatch and fell to
    the catch-all. With `beta = 0` the condition is `alpha*u = g` -- Dirichlet -- so the wall is an
    exit, and treating it as impermeable is the opposite physics."""
    result = _particle(_uniform(BCType.ROBIN, alpha=1.0, beta=0.0))

    assert len(result) == 1, "only the interior particle survives an absorbing wall"
    np.testing.assert_allclose(result[0], [0.5, 0.5])


def test_particle_robin_with_zero_alpha_still_reflects():
    """Control for the test above. Without it, an owner that absorbed every Robin would pass."""
    result = _particle(_uniform(BCType.ROBIN, alpha=0.0, beta=1.0))

    assert len(result) == len(_PARTICLES), "a pure flux condition absorbs nothing"


@pytest.mark.parametrize("bc_type", sorted(_NO_PARTICLE_MEANING, key=lambda t: t.name))
def test_the_segment_aware_path_no_longer_silently_reflects_a_truncation_rule(bc_type):
    """Measured before this change: reflected, silently, via the `else` branch. A wrong answer
    that looks like a right one is the failure this whole consolidation is about."""
    with pytest.raises(ValueError, match=bc_type.name):
        _particle(_uniform(bc_type))


# =============================================================================
# The cells that must NOT have moved
# =============================================================================


@pytest.mark.parametrize(
    ("bc_type", "surviving"),
    [(BCType.DIRICHLET, 1), (BCType.NEUMANN, 5), (BCType.NO_FLUX, 5), (BCType.PERIODIC, 5)],
)
def test_the_four_cells_that_already_agreed_are_unchanged(bc_type, surviving):
    """Fifteen of the twenty-two probed cells were byte-identical across the consolidation. These
    are the four where both paths already agreed; a merge that "fixed" them would be a regression
    wearing the same commit message."""
    assert len(_meshfree(_uniform(bc_type))) == surviving
    assert len(_particle(_uniform(bc_type))) == surviving


def test_the_three_actions_are_distinguishable_on_this_probe():
    """Negative control for every assertion above. If reflecting and wrapping produced the same
    positions on `_PARTICLES`, the whole file would pass over an owner that confused them --
    which is exactly the state the two copies were in for `REFLECTING`."""
    reflected = _meshfree(_uniform(BCType.NO_FLUX))
    wrapped = _meshfree(_uniform(BCType.PERIODIC))
    absorbed = _meshfree(_uniform(BCType.DIRICHLET))

    assert not np.allclose(reflected, wrapped), "reflect and wrap must differ on this fixture"
    assert len(absorbed) != len(reflected), "absorb must differ from reflect on this fixture"


# =============================================================================
# The fall-through cannot fabricate Robin coefficients
# =============================================================================


def test_a_robin_default_refuses_rather_than_fabricating_its_coefficients():
    """`BoundaryConditions` has `default_bc` and `default_value` and no default alpha/beta, so a
    fall-through segment took `BCSegment`'s dataclass defaults `alpha=1.0, beta=0.0` -- the
    Dirichlet corner of the Robin family.

    Found by independent review of this PR. The hole predates it and was inert: nothing read
    those coefficients off the returned segment. Routing the particle path through
    `particle_action_for_bc_type` makes them load-bearing, and a user's pure-flux wall
    (`alpha=0, beta=1`) then became **absorbing** on any uncovered face -- measured, 3 particles
    in and 1 out, mass destroyed with nothing raised.

    `test_robin_without_coefficients_refuses_rather_than_assuming` above asserts the owner must
    refuse rather than assume, and could not fire here: its guard tests `alpha is None`, while
    `BCSegment` hands over `1.0` and `0.0`. The refusal therefore belongs at the point of
    fabrication, not at the point of use.
    """
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="flux", bc_type=BCType.ROBIN, value=0.0, alpha=0.0, beta=1.0, boundary="x_min"),
            BCSegment(name="flux2", bc_type=BCType.ROBIN, value=0.0, alpha=0.0, beta=1.0, boundary="x_max"),
        ],
        dimension=2,
        default_bc=BCType.ROBIN,
        default_value=0.0,
        domain_bounds=np.array(_BOUNDS),
    )
    particles = np.array([[0.5, 0.5], [0.5, 1.05], [0.5, -0.05]])  # two leave through uncovered y faces

    with pytest.raises(ValueError, match="cannot carry"):
        ParticleApplicator().apply(particles, bc, _BOUNDS)


def test_a_fully_covered_robin_wall_still_reflects():
    """Control. The refusal must be narrow: a Robin condition whose segments cover every face
    carries its own coefficients and is unaffected. Without this, raising on every Robin BC would
    pass the test above."""
    bc = BoundaryConditions(
        segments=[BCSegment(name="flux", bc_type=BCType.ROBIN, value=0.0, alpha=0.0, beta=1.0)],
        dimension=2,
        default_bc=BCType.NO_FLUX,
        default_value=0.0,
        domain_bounds=np.array(_BOUNDS),
    )
    particles = np.array([[0.5, 0.5], [0.5, 1.05], [0.5, -0.05]])

    remaining, absorbed, _ = ParticleApplicator().apply(particles, bc, _BOUNDS)

    assert len(remaining) == 3, "a pure-flux wall absorbs nothing"
    assert not absorbed.any()

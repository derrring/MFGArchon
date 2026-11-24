"""The damping policy's ruled properties (#2997), pinned where they live: `DampingPolicy` and `DampingRun`."""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.damping_run import DampingRun
from mfgarchon.config.damping import DampingPolicy, FieldDamping

_RNG = np.random.default_rng(2997)
_U_OLD, _U_NEW, _M_OLD, _M_NEW = (_RNG.random((3, 5)) for _ in range(4))
_POLICIES = [
    FieldDamping(schedule="none"),
    FieldDamping(theta=0.3),
    FieldDamping(theta=0.8, schedule="harmonic", adaptive=True),
    FieldDamping(theta=0.6, first_sweep="whole"),
]


@pytest.mark.parametrize("k", [0, 3])
def test_one_fields_policy_does_not_move_the_others_iterate(k):
    """The user's ruling 5: with the same inputs, varying U's policy over four leaves M's next iterate bit-identical,
    and the mirror. The control: the varied field's own iterate does move. A mutation-verified convention pin: M blended
    with U's theta fails it."""
    M_next = {
        DampingRun(DampingPolicy(U=u, M=FieldDamping(theta=0.4))).update(k, _U_OLD, _U_NEW, _M_OLD, _M_NEW)[1].tobytes()
        for u in _POLICIES
    }
    U_next = {
        DampingRun(DampingPolicy(U=FieldDamping(theta=0.4), M=m)).update(k, _U_OLD, _U_NEW, _M_OLD, _M_NEW)[0].tobytes()
        for m in _POLICIES
    }
    U_own = {DampingRun(DampingPolicy(U=u)).update(k, _U_OLD, _U_NEW, _M_OLD, _M_NEW)[0].tobytes() for u in _POLICIES}
    assert len(M_next) == 1, "M's iterate moved with U's policy"
    assert len(U_next) == 1, "U's iterate moved with M's policy"
    assert len(U_own) > 1, "the control: U's own iterate must move with U's policy"


def test_an_adaptive_field_reads_only_its_own_history():
    """The user's ruling (i): U's change grows 1.6x at the third sweep while M's falls, so U's base decays to 0.4 and
    M's stays 0.8. A mutation-verified convention pin: M adapting on U's history fails it."""
    run = DampingRun(DampingPolicy(U=FieldDamping(theta=0.8, adaptive=True), M=FieldDamping(theta=0.8, adaptive=True)))
    for k, (change_U, change_M) in enumerate([(1.0, 1.0), (0.5, 0.9), (0.8, 0.8)]):
        run.observe(k, change_U, change_M)
    assert run.base("U") == pytest.approx(0.4)
    assert run.base("M") == 0.8


_FACTORS = {
    "constant": lambda theta, k: theta,
    "harmonic": lambda theta, k: theta * (1.0 / (k + 1)),
    "sqrt": lambda theta, k: theta * (1.0 / np.sqrt(k + 1)),
    "polynomial": lambda theta, k: theta * (1.0 / (k + 1) ** 0.6),
    "exponential": lambda theta, k: theta ** (k + 1),
}


@pytest.mark.parametrize("schedule", list(_FACTORS))
@pytest.mark.parametrize(("preset", "floor"), [("field", 0.05), ("fictitious play", 0.0)])
@pytest.mark.parametrize("theta", [1.0, 0.5, 0.01])
def test_each_schedule_is_its_closed_form(schedule, preset, floor, theta):
    """An external oracle, the schedules' closed forms (the user's ruling (ii): one registry, one value per name):
    theta_k is the form clamped below at the floor, except "constant", which no floor clamps -- FieldDamping's default
    0.05, and fictitious play's 0. Mutations: a floor on "constant", or "constant" as 0.5 theta, fail it."""
    policy = (
        DampingPolicy(U=FieldDamping(theta=theta, schedule=schedule))
        if preset == "field"
        else DampingPolicy.fictitious_play(theta, schedule)
    )
    run = DampingRun(policy)
    name = "U" if preset == "field" else "M"
    for k in (0, 1, 4, 30):
        form = _FACTORS[schedule](theta, k)
        assert run.theta(name, k) == (form if schedule == "constant" else max(form, floor))


def test_the_first_sweep_whole_then_the_schedule():
    """The first-sweep rule: "whole" takes the map's output at k = 0, then the schedule; the other field is untouched.
    A mutation-verified convention pin: ignoring the rule fails it."""
    run = DampingRun(DampingPolicy(M=FieldDamping(theta=0.6, schedule="harmonic", first_sweep="whole")))
    assert [run.theta("M", k) for k in (0, 1, 2)] == [1.0, 0.6 * (1.0 / 2), 0.6 * (1.0 / 3)]
    assert run.theta("U", 0) == 0.5

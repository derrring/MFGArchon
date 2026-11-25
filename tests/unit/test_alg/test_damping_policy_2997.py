"""The damping policy's ruled properties (#2997), pinned where they live: `DampingPolicy` and `DampingRun`, and the
routes that apply them where the property is the routes' agreement."""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling import (
    BlockGaussSeidelIterator,
    FictitiousPlayIterator,
    FixedPointIterator,
)
from mfgarchon.alg.numerical.coupling.damping_run import DampingRun
from mfgarchon.alg.numerical.coupling.newton_mfg_solver import NewtonMFGSolver
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.config import MFGSolverConfig, PicardConfig
from mfgarchon.config.damping import AndersonOptions, DampingPolicy, FieldDamping
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

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
        run.observe(k, {"l2distu_rel": change_U, "l2distm_rel": change_M})
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
    """A mutation-verified convention pin on the schedules' closed forms (the user's ruling (ii): one registry, one value
    per name):
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


@pytest.mark.parametrize(
    "field",
    [
        FieldDamping(schedule="none"),
        FieldDamping(theta=0.6, first_sweep="whole"),
        FieldDamping(theta=1.0),
        FieldDamping(theta=1.0, schedule="harmonic"),
    ],
    ids=["none", "first sweep whole", "constant 1", "harmonic from 1"],
)
def test_the_next_iterate_is_never_an_input_array(field):
    """Each way theta_k reaches 1 at k = 0, and Anderson on U: the next iterate shares no memory with an input, since the
    write-back edits it in place while the caller still measures the map's output (#2997 review 1). A
    mutation-verified convention pin: returning ``U_new`` or ``M_new`` itself at theta 1 fails the first four cases."""
    for policy in (DampingPolicy(U=field, M=field), DampingPolicy(M=field, anderson=AndersonOptions())):
        U, M = DampingRun(policy).update(0, _U_OLD, _U_NEW, _M_OLD, _M_NEW)
        for out in (U, M):
            assert not any(np.shares_memory(out, given) for given in (_U_OLD, _U_NEW, _M_OLD, _M_NEW))


def _strongly_coupled():
    """Coupling 5 at theta 1 on seven nodes: the undamped iteration's relative change grows past 1.2x, M's at the second
    sweep (1.57x) and U's at the third (1.34x), so both adaptive fields decay within five sweeps. The density's mass on
    the grid is 1, and the outer tolerance is the HJB solver's Newton tolerance, so neither solve warns."""
    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(QuadraticControlCost(control_cost=1.0), coupling=lambda m: 5.0 * m),
            volatility=0.2,
        ),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[7], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(
            m_initial=lambda x: 1.0 + 0.5 * np.cos(np.pi * np.asarray(x[0], dtype=float)),
            u_terminal=lambda x: 2.0 * (np.asarray(x[0], dtype=float) - 0.8) ** 2,
            T=1.0,
        ),
        Nt=4,
    )


def _ten_sweeps(route, damping):
    problem = _strongly_coupled()
    pair = (problem, HJBFDMSolver(problem), FPFDMSolver(problem))
    if route is NewtonMFGSolver:
        solver = NewtonMFGSolver(*pair, picard_warmup=10, newton_max_iterations=1, damping=damping)
        result = solver.solve(max_iterations=11, tolerance=1e-6, verbose=False)
    else:
        result = route(*pair, damping=damping).solve(max_iterations=10, tolerance=1e-6, verbose=False)
    return np.concatenate([np.asarray(result.error_history_U)[:10], np.asarray(result.error_history_M)[:10]])


def test_each_route_adapts_as_fixed_point_does():
    """Ruling 9 on every route: at theta 1 on a strongly coupled problem the four routes take one path, so with both
    fields adaptive each route's first ten changes are FixedPoint's, bit for bit. The control: FixedPoint's own changes
    move with adaptation, from the fourth sweep, after M's base decays at the second and U's at the third. A
    mutation-verified convention pin: a route that never observes, adapts on another measure, or applies a decay at the
    sweep it observed fails it; ten sweeps, since FictitiousPlay adapting on its averaged belief parts from FixedPoint
    only at the eighth (#2997 review 2)."""
    adaptive = FieldDamping(theta=1.0, adaptive=True)
    policy = DampingPolicy(U=adaptive, M=adaptive)
    reference = _ten_sweeps(FixedPointIterator, policy)
    assert not np.array_equal(reference, _ten_sweeps(FixedPointIterator, DampingPolicy.constant(1.0)))
    for route in (BlockGaussSeidelIterator, FictitiousPlayIterator, NewtonMFGSolver):
        assert np.array_equal(_ten_sweeps(route, policy), reference), f"{route.__name__} adapts unlike FixedPoint"


@pytest.mark.parametrize("route", [FixedPointIterator, FictitiousPlayIterator])
def test_a_config_damping_decides_and_one_given_twice_is_refused(route):
    """The user's ruling of 2026-11-24 on #2997, on both routes that take a config: a config that sets
    ``PicardConfig.damping`` decides; one that leaves it unset keeps the constructor's; both set is refused by name.
    The control: the route's default solves differently. A mutation-verified convention pin: a route that drops the
    constructor's damping beside a config, or ignores the config's, fails it."""

    def solve(damping, config_damping):
        problem = _strongly_coupled()
        config = MFGSolverConfig(
            picard=PicardConfig(max_iterations=2, tolerance=1e-6, verbose=False, damping=config_damping)
        )
        result = route(problem, HJBFDMSolver(problem), FPFDMSolver(problem), damping=damping).solve(config=config)
        return np.concatenate([result.U.ravel(), result.M.ravel()])

    policy = DampingPolicy.constant(0.3, 0.2)
    from_constructor = solve(policy, None)
    assert np.array_equal(solve(None, policy), from_constructor)
    assert not np.array_equal(solve(None, None), from_constructor)
    with pytest.raises(ValueError, match="damping is given twice"):
        solve(policy, DampingPolicy.constant(0.4))

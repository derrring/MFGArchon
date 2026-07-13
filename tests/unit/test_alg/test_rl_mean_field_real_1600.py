"""Issue #1600: the continuous-MFG RL environments must deliver a REAL mean field.

Three independently-verified defects, all fixed here:
  1. compute_mean_field_coupling ignored the `population` argument (LQ used x**2; crowd used a fixed
     distance-to-center proxy) -> the "MFG" reward did not depend on the population at all.
  2. step() advanced only agent_states[0], freezing agents 1..N-1 at their reset sample, so the
     empirical density get_population_state() bins was ~= m_0 forever.
  3. Brownian noise was applied to EVERY state dim, including declared-constant dims (crowd goal
     coords, price market depth, traffic time-remaining), silently random-walking them.

Each test is discriminating: it FAILS on the pre-fix code. The fix keeps agent 0 the controlled ego
agent and does NOT build a fictitious-play equilibrium (that is #1570/#887) -- it makes the empirical
density genuinely move and the coupling genuinely read it.
"""

from __future__ import annotations

import pytest

import numpy as np

pytest.importorskip("gymnasium", reason="continuous MFG environments require gymnasium")

from mfgarchon.alg.reinforcement.environments import (
    CrowdNavigationEnv,
    LQMFGEnv,
    PriceFormationEnv,
    ResourceAllocationEnv,
    TrafficFlowEnv,
)

ALL_ENVS = [LQMFGEnv, CrowdNavigationEnv, PriceFormationEnv, TrafficFlowEnv, ResourceAllocationEnv]


def _rollout(env, steps: int = 15, seed: int = 0):
    env.reset(seed=seed)
    initial_states = env.agent_states.copy()
    pop0 = env.get_population_state().copy()
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        env.step(rng.uniform(env.action_bounds[0], env.action_bounds[1], size=env.action_dim).astype(np.float32))
    return initial_states, pop0


@pytest.mark.parametrize("EnvCls", ALL_ENVS)
def test_population_evolves_via_all_agents(EnvCls):
    """Defect 2: the empirical mean field must MOVE, because the WHOLE population advances, not just
    agent 0. Discriminating: pre-fix step() writes only agent_states[0], so agents 1..N-1 stay at
    their reset values -> the density is frozen and ``agent_states[1:]`` is byte-identical to reset."""
    env = EnvCls()
    initial_states, pop0 = _rollout(env)
    assert np.abs(env.get_population_state() - pop0).sum() > 1e-3, "empirical mean field did not move"
    assert not np.allclose(env.agent_states[1:], initial_states[1:]), "non-ego agents stayed frozen (defect 2)"


@pytest.mark.parametrize("EnvCls", [LQMFGEnv, CrowdNavigationEnv])
def test_coupling_reads_population(EnvCls):
    """Defect 1 (the two envs that ignored it): compute_mean_field_coupling must differ for two
    DISTINCT populations at the same agent state. Discriminating: pre-fix LQ returns -c_m*x**2 and
    crowd a distance-to-center proxy -- both independent of `population` -> the two values are equal."""
    env = EnvCls()
    env.reset(seed=0)
    bins = env.population_bins
    state = env.agent_states[0]
    pop_low = np.zeros(bins, dtype=np.float32)
    pop_low[0] = 1.0
    pop_high = np.zeros(bins, dtype=np.float32)
    pop_high[-1] = 1.0
    c_low = env.compute_mean_field_coupling(state, pop_low)
    c_high = env.compute_mean_field_coupling(state, pop_high)
    assert abs(c_low - c_high) > 1e-6, f"coupling ignores population ({c_low} == {c_high})"


def test_lq_coupling_reduces_to_x_squared_at_delta_origin():
    """The LQ fix GENERALISES rather than replaces: the discrete -c_m * sum_b (x - y_b)^2 m_b equals
    the old -c_m * x**2 when the population is a delta at the origin (the bin containing 0)."""
    env = LQMFGEnv()
    env.reset(seed=0)
    bins = env.population_bins
    edges = np.linspace(-env.x_max, env.x_max, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    origin_bin = int(np.argmin(np.abs(centers)))
    delta0 = np.zeros(bins, dtype=np.float32)
    delta0[origin_bin] = 1.0
    state = np.array([0.7, 0.0], dtype=np.float32)
    coupling = env.compute_mean_field_coupling(state, delta0)
    y0 = centers[origin_bin]
    expected = -env.cost_mean_field * (state[0] - y0) ** 2
    assert coupling == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize(("EnvCls", "const_dims"), [(CrowdNavigationEnv, [4, 5]), (PriceFormationEnv, [3])])
def test_zero_drift_dims_receive_no_noise(EnvCls, const_dims):
    """Defect 3: declared-constant (zero-drift) dims must NOT random-walk. Checked on the non-ego
    agents, whose zero-drift dims stay byte-identical to reset. Discriminating: pre-fix the base adds
    N(0, noise_std*sqrt(dt)) to every dim, so these dims drift away from their reset values."""
    env = EnvCls()
    initial_states, _ = _rollout(env)
    for cd in const_dims:
        assert np.allclose(env.agent_states[1:, cd], initial_states[1:, cd]), f"dim {cd} got Brownian noise (defect 3)"


def test_traffic_time_remaining_is_deterministic_countdown():
    """Defect 3 for traffic: time-remaining (dim 2) has drift -1, so with the noise mask every agent's
    countdown advances by the SAME deterministic amount. Discriminating: pre-fix noise makes the
    per-agent deltas differ (a random deadline)."""
    env = TrafficFlowEnv()
    initial_states, _ = _rollout(env, steps=10)
    deltas = env.agent_states[1:, 2] - initial_states[1:, 2]
    assert deltas[0] < 0, "time-remaining should decrease"
    assert np.allclose(deltas, deltas[0]), "time-remaining picked up per-agent noise (defect 3)"


def test_resource_simplex_preserved_after_consolidation():
    """The resource env dropped its duplicated step() override for a _postprocess_next_states hook on
    the base _advance_population driver (single-source, Issue #1600). The portfolio constraints must
    still hold for EVERY advanced agent: allocations on the simplex, asset values non-negative.
    Discriminating: if the hook did not run (base identity), allocations would not sum to 1."""
    env = ResourceAllocationEnv()
    _rollout(env, steps=10)
    alloc = env.agent_states[:, : env.num_assets]
    assert np.allclose(alloc.sum(axis=1), 1.0, atol=1e-5), "allocations left the simplex"
    assert (alloc >= -1e-8).all(), "negative allocation weight"
    assert (env.agent_states[:, env.num_assets :] >= -1e-8).all(), "negative asset value"


@pytest.mark.parametrize("EnvCls", ALL_ENVS)
def test_get_population_state_flat_ndarray_contract(EnvCls):
    """#1615 contract preserved: get_population_state() returns a flat rank-1 NDArray for every env
    (the algorithms consume it identically)."""
    env = EnvCls()
    env.reset(seed=0)
    pop = env.get_population_state()
    assert isinstance(pop, np.ndarray)
    assert pop.ndim == 1
    assert pop.shape == (env.population_bins,)

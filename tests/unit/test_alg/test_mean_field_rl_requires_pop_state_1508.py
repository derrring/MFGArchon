"""Issue #1508: mean-field RL (DDPG/TD3/SAC) must FAIL LOUD when the env lacks get_population_state()
instead of silently zero-filling the population state (which trains on an identically-zero mean field
-> a non-MFG policy the user trusts). get_population_state is a required MFG-coupling capability."""

from __future__ import annotations

import pytest

import numpy as np

torch = pytest.importorskip("torch", reason="mean-field RL algorithms require PyTorch")


class _EnvWithoutPopState:
    """A minimal env that reset()s but does NOT expose get_population_state (the MFG coupling channel)."""

    def reset(self):
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):  # pragma: no cover - the guard raises before we get here
        return np.zeros(2, dtype=np.float32), 0.0, True, False, {}


class _EnvWithPopState:
    """Minimal env exposing the #1570 canonical contract: get_population_state() -> flat NDArray.
    step() terminates after one transition so train() stays cheap."""

    def __init__(self, state_dim: int = 2, pop_dim: int = 4):
        self._sd, self._pd = state_dim, pop_dim

    def reset(self):
        return np.zeros(self._sd, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self._sd, dtype=np.float32), 0.0, True, False, {}

    def get_population_state(self):
        return np.zeros(self._pd, dtype=np.float32)


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("mfgarchon.alg.reinforcement.algorithms.mean_field_ddpg", "MeanFieldDDPG"),
        ("mfgarchon.alg.reinforcement.algorithms.mean_field_td3", "MeanFieldTD3"),
        ("mfgarchon.alg.reinforcement.algorithms.mean_field_sac", "MeanFieldSAC"),
    ],
)
def test_algo_trains_with_ndarray_population(module_name, class_name):
    """#1601 / #1570: with the canonical flat-NDArray population contract, every algo runs against an
    env whose get_population_state() returns an ndarray. Pre-fix the algos did
    ``get_population_state().density_histogram.flatten()``, so an ndarray -- the declared contract for
    the whole ContinuousMFGEnvBase family -- raised AttributeError on the first step (15 algo x env
    pairings dead). Discriminating: reverting to ``.density_histogram.flatten()`` makes this raise
    AttributeError('numpy.ndarray' object has no attribute 'density_histogram')."""
    import importlib

    algo_cls = getattr(importlib.import_module(module_name), class_name)
    algo = algo_cls(
        env=_EnvWithPopState(state_dim=2, pop_dim=4),
        state_dim=2,
        action_dim=1,
        population_dim=4,
        action_bounds=(-1.0, 1.0),
    )
    algo.train(num_episodes=1)  # must NOT raise (pre-fix: AttributeError on .density_histogram)

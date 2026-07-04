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


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("mfgarchon.alg.reinforcement.algorithms.mean_field_ddpg", "MeanFieldDDPG"),
        ("mfgarchon.alg.reinforcement.algorithms.mean_field_td3", "MeanFieldTD3"),
        ("mfgarchon.alg.reinforcement.algorithms.mean_field_sac", "MeanFieldSAC"),
    ],
)
def test_missing_get_population_state_fails_loud(module_name, class_name):
    import importlib

    algo_cls = getattr(importlib.import_module(module_name), class_name)
    algo = algo_cls(
        env=_EnvWithoutPopState(),
        state_dim=2,
        action_dim=1,
        population_dim=4,
        action_bounds=(-1.0, 1.0),
    )
    with pytest.raises(AttributeError, match="1508"):
        algo.train(num_episodes=1)

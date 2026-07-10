"""Issue #1572: AdaptiveTrainingStrategy must consume the canonical ``training_mode`` (via the
``uses_*`` properties), not the raw deprecated ``enable_*`` booleans.

The deprecated booleans default to ``None``; ``__post_init__`` maps them to ``training_mode`` but
never back-fills them. The strategy used to read ``self.config.enable_curriculum`` / ``enable_refinement``
directly, so a canonical ``training_mode=CURRICULUM`` (with the booleans left at ``None``) made
``not None == True`` -> the feature was silently skipped, and the default ``FULL_ADAPTIVE`` skipped
both curriculum and refinement. The consumer now reads ``uses_curriculum`` / ``uses_refinement``.

Discriminator: at epoch 0, an active curriculum returns ``initial_complexity`` (0.1 by default), while
a disabled one early-returns ``1.0``. A revert to the raw boolean makes the canonical modes return
``1.0`` (silently disabled) and these tests fail.
"""

from __future__ import annotations

import pytest

from mfgarchon.alg.neural.pinn_solvers.adaptive_training import (
    TORCH_AVAILABLE,
    AdaptiveTrainingConfig,
    AdaptiveTrainingMode,
    AdaptiveTrainingStrategy,
)

# AdaptiveTrainingStrategy is only the real (stateful) class under `if TORCH_AVAILABLE`; without
# torch it degrades to a no-op stub, so these behavioral tests require torch (matching the neural
# test convention). AdaptiveTrainingConfig + uses_* are torch-free but are exercised through the
# strategy here, so the whole file is torch-gated.
pytestmark = [pytest.mark.optional_torch, pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")]


def _curriculum_complexity_at_epoch0(config: AdaptiveTrainingConfig) -> float:
    strategy = AdaptiveTrainingStrategy(config)
    strategy.state.epoch = 0
    return strategy.update_curriculum()


def test_curriculum_mode_activates_curriculum():
    """training_mode=CURRICULUM (enable_* left at None) must run curriculum, not silently skip it."""
    config = AdaptiveTrainingConfig(training_mode=AdaptiveTrainingMode.CURRICULUM)
    assert config.enable_curriculum is None  # the trap the old consumer read directly
    assert config.uses_curriculum is True
    # Active curriculum at epoch 0 returns initial_complexity (0.1), NOT the disabled 1.0.
    assert _curriculum_complexity_at_epoch0(config) == config.initial_complexity
    assert _curriculum_complexity_at_epoch0(config) < 1.0


def test_default_full_adaptive_activates_curriculum_and_refinement():
    """The default FULL_ADAPTIVE mode must enable both curriculum and refinement (it used to skip
    both because enable_* defaulted to None)."""
    config = AdaptiveTrainingConfig()  # default = FULL_ADAPTIVE
    assert config.uses_curriculum is True
    assert config.uses_refinement is True
    assert _curriculum_complexity_at_epoch0(config) == config.initial_complexity


def test_basic_mode_disables_curriculum():
    """training_mode=BASIC must disable curriculum (the property correctly returns False)."""
    config = AdaptiveTrainingConfig(training_mode=AdaptiveTrainingMode.BASIC)
    assert config.uses_curriculum is False
    assert _curriculum_complexity_at_epoch0(config) == 1.0


def test_deprecated_boolean_still_works():
    """The deprecated enable_curriculum=True path must still activate curriculum (redirect intact)."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = AdaptiveTrainingConfig(enable_curriculum=True, enable_multiscale=False, enable_refinement=False)
    assert config.uses_curriculum is True
    assert _curriculum_complexity_at_epoch0(config) == config.initial_complexity


def test_consumers_read_canonical_properties_not_raw_booleans():
    """Source-level pin covering all three consumer reads, incl. the refinement path (whose behavioral
    test would require torch): the strategy must gate on uses_curriculum/uses_refinement, never on the
    raw deprecated self.config.enable_* booleans."""
    import inspect

    for method in (
        AdaptiveTrainingStrategy.update_curriculum,
        AdaptiveTrainingStrategy.update_sampling_points,
    ):
        src = inspect.getsource(method)
        assert "self.config.enable_curriculum" not in src, f"{method.__name__} reads the raw deprecated boolean"
        assert "self.config.enable_refinement" not in src, f"{method.__name__} reads the raw deprecated boolean"
    ref_src = inspect.getsource(AdaptiveTrainingStrategy.update_sampling_points)
    assert "self.config.uses_refinement" in ref_src, "refinement gate must read the canonical uses_refinement"

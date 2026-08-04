"""
Unit tests for enum conversions with backward compatibility.

Tests the deprecation warnings and correct mapping of boolean parameters
to enum values for PINN training modes, normalization types, and DGM
variance reduction methods (Issue #277 Phase 2).
"""

import pytest

# ============================================================================
# Test: AdaptiveTrainingMode Enum (PINN adaptive_training.py)
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_adaptive_training_mode_new_api():
    """Test new AdaptiveTrainingMode API works correctly."""
    from mfgarchon.alg.neural.pinn_solvers.adaptive_training import AdaptiveTrainingConfig, AdaptiveTrainingMode

    # Test creating config with new enum API
    config = AdaptiveTrainingConfig(training_mode=AdaptiveTrainingMode.CURRICULUM)

    assert config.training_mode == AdaptiveTrainingMode.CURRICULUM
    assert config.uses_curriculum is True
    assert config.uses_multiscale is False
    assert config.uses_refinement is False


@pytest.mark.unit
@pytest.mark.fast
def test_adaptive_training_mode_deprecated_booleans():
    """Test deprecated boolean parameters emit warnings."""
    from mfgarchon.alg.neural.pinn_solvers.adaptive_training import AdaptiveTrainingConfig

    # Test deprecated API triggers warning
    with pytest.warns(DeprecationWarning, match="'enable_curriculum'.*deprecated"):
        config = AdaptiveTrainingConfig(
            enable_curriculum=True,
            enable_multiscale=False,
            enable_refinement=False,
        )

    # Verify correct mapping to enum
    from mfgarchon.alg.neural.pinn_solvers.adaptive_training import AdaptiveTrainingMode

    assert config.training_mode == AdaptiveTrainingMode.CURRICULUM


@pytest.mark.unit
@pytest.mark.fast
def test_adaptive_training_mode_boolean_mapping():
    """Test deprecated booleans map correctly to enum values."""
    from mfgarchon.alg.neural.pinn_solvers.adaptive_training import AdaptiveTrainingConfig, AdaptiveTrainingMode

    # Test all False → BASIC
    with pytest.warns(DeprecationWarning, match="deprecated"):
        config = AdaptiveTrainingConfig(
            enable_curriculum=False,
            enable_multiscale=False,
            enable_refinement=False,
        )
    assert config.training_mode == AdaptiveTrainingMode.BASIC

    # Test curriculum only → CURRICULUM
    with pytest.warns(DeprecationWarning, match="deprecated"):
        config = AdaptiveTrainingConfig(
            enable_curriculum=True,
            enable_multiscale=False,
            enable_refinement=False,
        )
    assert config.training_mode == AdaptiveTrainingMode.CURRICULUM

    # Test multiscale only → MULTISCALE
    with pytest.warns(DeprecationWarning, match="deprecated"):
        config = AdaptiveTrainingConfig(
            enable_curriculum=False,
            enable_multiscale=True,
            enable_refinement=False,
        )
    assert config.training_mode == AdaptiveTrainingMode.MULTISCALE

    # Test multiple flags → FULL_ADAPTIVE
    with pytest.warns(DeprecationWarning, match="deprecated"):
        config = AdaptiveTrainingConfig(
            enable_curriculum=True,
            enable_multiscale=True,
            enable_refinement=True,
        )
    assert config.training_mode == AdaptiveTrainingMode.FULL_ADAPTIVE


# ============================================================================
# Test: NormalizationType Enum (PINN base_pinn.py)
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif("not torch_available()")
def test_normalization_type_new_api():
    """Test new NormalizationType API works correctly."""
    from mfgarchon.alg.neural.pinn_solvers.base_pinn import NormalizationType, PINNConfig

    # Test creating config with new enum API
    config = PINNConfig(normalization=NormalizationType.BATCH)

    assert config.normalization == NormalizationType.BATCH


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif("not torch_available()")
def test_normalization_type_deprecated_booleans():
    """Test deprecated boolean parameters emit warnings."""
    from mfgarchon.alg.neural.pinn_solvers.base_pinn import PINNConfig

    # Test deprecated API triggers warning
    with pytest.warns(DeprecationWarning, match="'use_batch_norm'.*deprecated"):
        config = PINNConfig(use_batch_norm=True)

    # Verify correct mapping to enum
    from mfgarchon.alg.neural.pinn_solvers.base_pinn import NormalizationType

    assert config.normalization == NormalizationType.BATCH


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif("not torch_available()")
def test_normalization_type_boolean_mapping():
    """Test deprecated booleans map correctly to enum values."""
    from mfgarchon.alg.neural.pinn_solvers.base_pinn import NormalizationType, PINNConfig

    # Test batch norm → BATCH
    with pytest.warns(DeprecationWarning, match="deprecated"):
        config = PINNConfig(use_batch_norm=True, use_layer_norm=False)
    assert config.normalization == NormalizationType.BATCH

    # Test layer norm → LAYER
    with pytest.warns(DeprecationWarning, match="deprecated"):
        config = PINNConfig(use_batch_norm=False, use_layer_norm=True)
    assert config.normalization == NormalizationType.LAYER

    # Test both False → NONE
    with pytest.warns(DeprecationWarning, match="deprecated"):
        config = PINNConfig(use_batch_norm=False, use_layer_norm=False)
    assert config.normalization == NormalizationType.NONE


# ============================================================================
# Test: VarianceReductionMethod Enum (DGM base_dgm.py)
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_variance_reduction_method_new_api():
    """Test new VarianceReductionMethod API works correctly."""
    from mfgarchon.alg.neural.dgm.base_dgm import DGMConfig, VarianceReductionMethod

    # Test creating config with new enum API
    config = DGMConfig(variance_reduction=VarianceReductionMethod.IMPORTANCE_SAMPLING)

    assert config.variance_reduction == VarianceReductionMethod.IMPORTANCE_SAMPLING


@pytest.mark.unit
@pytest.mark.fast
def test_variance_reduction_method_deprecated_booleans():
    """Test deprecated boolean parameters emit warnings."""
    from mfgarchon.alg.neural.dgm.base_dgm import DGMConfig

    # Test deprecated API triggers warning
    with pytest.warns(DeprecationWarning, match="'use_control_variates'.*deprecated"):
        config = DGMConfig(use_control_variates=True, use_importance_sampling=False)

    # Verify correct mapping to enum
    from mfgarchon.alg.neural.dgm.base_dgm import VarianceReductionMethod

    assert config.variance_reduction == VarianceReductionMethod.CONTROL_VARIATES


@pytest.mark.unit
@pytest.mark.fast
def test_variance_reduction_method_boolean_mapping():
    """Test deprecated booleans map correctly to enum values."""
    from mfgarchon.alg.neural.dgm.base_dgm import DGMConfig, VarianceReductionMethod

    # Test both False → NONE
    with pytest.warns(DeprecationWarning, match="deprecated"):
        config = DGMConfig(use_control_variates=False, use_importance_sampling=False)
    assert config.variance_reduction == VarianceReductionMethod.NONE

    # Test control variates only → CONTROL_VARIATES
    with pytest.warns(DeprecationWarning, match="deprecated"):
        config = DGMConfig(use_control_variates=True, use_importance_sampling=False)
    assert config.variance_reduction == VarianceReductionMethod.CONTROL_VARIATES

    # Test importance sampling only → IMPORTANCE_SAMPLING
    with pytest.warns(DeprecationWarning, match="deprecated"):
        config = DGMConfig(use_control_variates=False, use_importance_sampling=True)
    assert config.variance_reduction == VarianceReductionMethod.IMPORTANCE_SAMPLING

    # Test both True → BOTH
    with pytest.warns(DeprecationWarning, match="deprecated"):
        config = DGMConfig(use_control_variates=True, use_importance_sampling=True)
    assert config.variance_reduction == VarianceReductionMethod.BOTH


# ============================================================================
# Helper Functions
# ============================================================================


def torch_available() -> bool:
    """Check if torch is available."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False

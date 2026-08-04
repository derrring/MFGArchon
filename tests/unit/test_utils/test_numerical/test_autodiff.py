#!/usr/bin/env python3
"""
Unit tests for mfgarchon/utils/numerical/autodiff.py

Tests AutoDiffBackend enum including:
- Enum values and string representation
- Backend property checks (is_numpy, is_jax, is_pytorch)
- Dependency checking (requires_dependency, get_dependency_name)
- Enum behavior (comparison, iteration, membership)
"""

import pytest

from mfgarchon.utils.numerical.autodiff import AutoDiffBackend

# ===================================================================
# Test AutoDiffBackend Enum Values
# ===================================================================


@pytest.mark.unit
def test_autodiff_backend_values():
    """Test AutoDiffBackend enum has expected values."""
    assert AutoDiffBackend.NUMPY.value == "numpy"
    assert AutoDiffBackend.JAX.value == "jax"
    assert AutoDiffBackend.PYTORCH.value == "pytorch"


@pytest.mark.unit
def test_autodiff_backend_string_representation():
    """Test AutoDiffBackend string representation."""
    # str() returns full name like "AutoDiffBackend.NUMPY"
    assert "NUMPY" in str(AutoDiffBackend.NUMPY)
    assert "JAX" in str(AutoDiffBackend.JAX)
    assert "PYTORCH" in str(AutoDiffBackend.PYTORCH)


@pytest.mark.unit
def test_autodiff_backend_from_string():
    """Test creating AutoDiffBackend from string."""
    assert AutoDiffBackend("numpy") == AutoDiffBackend.NUMPY
    assert AutoDiffBackend("jax") == AutoDiffBackend.JAX
    assert AutoDiffBackend("pytorch") == AutoDiffBackend.PYTORCH


@pytest.mark.unit
def test_autodiff_backend_invalid_string():
    """Test creating AutoDiffBackend from invalid string raises error."""
    with pytest.raises(ValueError):
        AutoDiffBackend("tensorflow")
    with pytest.raises(ValueError):
        AutoDiffBackend("invalid")


# ===================================================================
# Test is_numpy Property
# ===================================================================


@pytest.mark.unit
def test_is_numpy_true():
    """Test is_numpy property returns True for NUMPY backend."""
    assert AutoDiffBackend.NUMPY.is_numpy is True


@pytest.mark.unit
def test_is_numpy_false():
    """Test is_numpy property returns False for non-NUMPY backends."""
    assert AutoDiffBackend.JAX.is_numpy is False
    assert AutoDiffBackend.PYTORCH.is_numpy is False


# ===================================================================
# Test is_jax Property
# ===================================================================


@pytest.mark.unit
def test_is_jax_true():
    """Test is_jax property returns True for JAX backend."""
    assert AutoDiffBackend.JAX.is_jax is True


@pytest.mark.unit
def test_is_jax_false():
    """Test is_jax property returns False for non-JAX backends."""
    assert AutoDiffBackend.NUMPY.is_jax is False
    assert AutoDiffBackend.PYTORCH.is_jax is False


# ===================================================================
# Test is_pytorch Property
# ===================================================================


@pytest.mark.unit
def test_is_pytorch_true():
    """Test is_pytorch property returns True for PYTORCH backend."""
    assert AutoDiffBackend.PYTORCH.is_pytorch is True


@pytest.mark.unit
def test_is_pytorch_false():
    """Test is_pytorch property returns False for non-PYTORCH backends."""
    assert AutoDiffBackend.NUMPY.is_pytorch is False
    assert AutoDiffBackend.JAX.is_pytorch is False


# ===================================================================
# Test requires_dependency Property
# ===================================================================


@pytest.mark.unit
def test_requires_dependency_numpy():
    """Test NUMPY backend does not require dependency."""
    assert AutoDiffBackend.NUMPY.requires_dependency is False


@pytest.mark.unit
def test_requires_dependency_jax():
    """Test JAX backend requires dependency."""
    assert AutoDiffBackend.JAX.requires_dependency is True


@pytest.mark.unit
def test_requires_dependency_pytorch():
    """Test PYTORCH backend requires dependency."""
    assert AutoDiffBackend.PYTORCH.requires_dependency is True


# ===================================================================
# Test get_dependency_name Method
# ===================================================================


@pytest.mark.unit
def test_get_dependency_name_numpy():
    """Test get_dependency_name for NUMPY returns None."""
    assert AutoDiffBackend.NUMPY.get_dependency_name() is None


@pytest.mark.unit
def test_get_dependency_name_jax():
    """Test get_dependency_name for JAX returns 'jax'."""
    assert AutoDiffBackend.JAX.get_dependency_name() == "jax"


@pytest.mark.unit
def test_get_dependency_name_pytorch():
    """Test get_dependency_name for PYTORCH returns 'torch'."""
    assert AutoDiffBackend.PYTORCH.get_dependency_name() == "torch"


# ===================================================================
# Test Enum Behavior
# ===================================================================


@pytest.mark.unit
def test_autodiff_backend_equality():
    """Test AutoDiffBackend equality comparison."""
    assert AutoDiffBackend.NUMPY == AutoDiffBackend.NUMPY
    assert AutoDiffBackend.JAX == AutoDiffBackend.JAX
    assert AutoDiffBackend.PYTORCH == AutoDiffBackend.PYTORCH

    assert AutoDiffBackend.NUMPY != AutoDiffBackend.JAX
    assert AutoDiffBackend.JAX != AutoDiffBackend.PYTORCH


@pytest.mark.unit
def test_autodiff_backend_identity():
    """Test AutoDiffBackend identity (is operator)."""
    numpy1 = AutoDiffBackend.NUMPY
    numpy2 = AutoDiffBackend.NUMPY
    assert numpy1 is numpy2  # Enum members are singletons


@pytest.mark.unit
def test_autodiff_backend_membership():
    """Test AutoDiffBackend membership checks."""
    backends = [AutoDiffBackend.NUMPY, AutoDiffBackend.JAX]
    assert AutoDiffBackend.NUMPY in backends
    assert AutoDiffBackend.PYTORCH not in backends


@pytest.mark.unit
def test_autodiff_backend_iteration():
    """Test AutoDiffBackend iteration."""
    backends = list(AutoDiffBackend)
    assert len(backends) == 3
    assert AutoDiffBackend.NUMPY in backends
    assert AutoDiffBackend.JAX in backends
    assert AutoDiffBackend.PYTORCH in backends


@pytest.mark.unit
def test_autodiff_backend_count():
    """Test AutoDiffBackend has exactly 3 members."""
    assert len(list(AutoDiffBackend)) == 3


# ===================================================================
# Test String Enum Behavior
# ===================================================================


@pytest.mark.unit
def test_autodiff_backend_str_enum_comparison():
    """Test AutoDiffBackend can be compared with strings."""
    # Since AutoDiffBackend inherits from str
    assert AutoDiffBackend.NUMPY == "numpy"
    assert AutoDiffBackend.JAX == "jax"
    assert AutoDiffBackend.PYTORCH == "pytorch"


@pytest.mark.unit
def test_autodiff_backend_string_operations():
    """Test AutoDiffBackend supports string operations."""
    # Can use in string contexts
    assert AutoDiffBackend.NUMPY.upper() == "NUMPY"
    assert AutoDiffBackend.JAX.startswith("j")
    assert AutoDiffBackend.PYTORCH.endswith("torch")


# ===================================================================
# Test Property Mutual Exclusivity
# ===================================================================


@pytest.mark.unit
def test_backend_properties_mutually_exclusive():
    """Test backend properties are mutually exclusive."""
    for backend in AutoDiffBackend:
        # Each backend has exactly one True property
        properties = [backend.is_numpy, backend.is_jax, backend.is_pytorch]
        assert sum(properties) == 1, f"{backend} has multiple True properties"


@pytest.mark.unit
def test_numpy_properties():
    """Test NUMPY backend has correct property values."""
    backend = AutoDiffBackend.NUMPY
    assert backend.is_numpy is True
    assert backend.is_jax is False
    assert backend.is_pytorch is False
    assert backend.requires_dependency is False
    assert backend.get_dependency_name() is None


@pytest.mark.unit
def test_jax_properties():
    """Test JAX backend has correct property values."""
    backend = AutoDiffBackend.JAX
    assert backend.is_numpy is False
    assert backend.is_jax is True
    assert backend.is_pytorch is False
    assert backend.requires_dependency is True
    assert backend.get_dependency_name() == "jax"


@pytest.mark.unit
def test_pytorch_properties():
    """Test PYTORCH backend has correct property values."""
    backend = AutoDiffBackend.PYTORCH
    assert backend.is_numpy is False
    assert backend.is_jax is False
    assert backend.is_pytorch is True
    assert backend.requires_dependency is True
    assert backend.get_dependency_name() == "torch"


# ===================================================================
# Test Usage Patterns
# ===================================================================


@pytest.mark.unit
def test_backend_selection_pattern():
    """Test typical backend selection pattern."""

    def select_backend(backend: AutoDiffBackend) -> str:
        if backend.is_numpy:
            return "Using finite differences"
        elif backend.is_jax:
            return "Using JAX autodiff"
        elif backend.is_pytorch:
            return "Using PyTorch autograd"
        return "Unknown"

    assert select_backend(AutoDiffBackend.NUMPY) == "Using finite differences"
    assert select_backend(AutoDiffBackend.JAX) == "Using JAX autodiff"
    assert select_backend(AutoDiffBackend.PYTORCH) == "Using PyTorch autograd"


@pytest.mark.unit
def test_dependency_checking_pattern():
    """Test typical dependency checking pattern."""

    def check_dependencies(backend: AutoDiffBackend) -> str | None:
        if backend.requires_dependency:
            return backend.get_dependency_name()
        return None

    assert check_dependencies(AutoDiffBackend.NUMPY) is None
    assert check_dependencies(AutoDiffBackend.JAX) == "jax"
    assert check_dependencies(AutoDiffBackend.PYTORCH) == "torch"


@pytest.mark.unit
def test_config_usage_pattern():
    """Test typical configuration usage pattern."""

    class MockConfig:
        def __init__(self, backend: AutoDiffBackend = AutoDiffBackend.NUMPY):
            self.backend = backend

    config1 = MockConfig(backend=AutoDiffBackend.NUMPY)
    config2 = MockConfig(backend=AutoDiffBackend.JAX)
    config3 = MockConfig(backend=AutoDiffBackend.PYTORCH)

    assert config1.backend.is_numpy
    assert config2.backend.is_jax
    assert config3.backend.is_pytorch


# ===================================================================
# Test Type Annotations
# ===================================================================


@pytest.mark.unit
def test_backend_type_annotation():
    """Test AutoDiffBackend works in type annotations."""

    def process_with_backend(backend: AutoDiffBackend) -> str:
        return backend.value  # Use .value to get string

    assert process_with_backend(AutoDiffBackend.NUMPY) == "numpy"
    assert process_with_backend(AutoDiffBackend.JAX) == "jax"
    assert process_with_backend(AutoDiffBackend.PYTORCH) == "pytorch"


# ===================================================================
# Test Module Exports
# ===================================================================


@pytest.mark.unit
def test_module_docstring():
    """Test module has comprehensive docstring."""
    from mfgarchon.utils.numerical import autodiff

    assert autodiff.__doc__ is not None
    assert "Automatic Differentiation" in autodiff.__doc__

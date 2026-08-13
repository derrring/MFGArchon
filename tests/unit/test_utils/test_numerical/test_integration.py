#!/usr/bin/env python3
"""
Unit tests for mfgarchon/utils/numerical/integration.py

Tests integration utilities wrapper including:
- get_integration_info() function
- trapezoid() re-export
- Module exports and backward compatibility
"""

import pytest

import numpy as np

from mfgarchon.utils.numerical.integration import get_integration_info, trapezoid

# ===================================================================
# Test get_integration_info() Function
# ===================================================================


@pytest.mark.unit
def test_get_integration_info_returns_dict():
    """Test get_integration_info() returns dictionary."""
    info = get_integration_info()
    assert isinstance(info, dict)


@pytest.mark.unit
def test_get_integration_info_has_required_keys():
    """Test get_integration_info() contains required keys."""
    info = get_integration_info()
    required_keys = [
        "numpy_version",
        "numpy_version_tuple",
        "has_trapezoid",
        "has_trapz",
        "recommended_method",
    ]
    for key in required_keys:
        assert key in info, f"Missing key: {key}"


@pytest.mark.unit
def test_get_integration_info_numpy_version():
    """Test get_integration_info() numpy_version is string."""
    info = get_integration_info()
    assert isinstance(info["numpy_version"], str)
    assert info["numpy_version"] == np.__version__


@pytest.mark.unit
def test_get_integration_info_recommended_method():
    """Test get_integration_info() recommended_method is string."""
    info = get_integration_info()
    assert isinstance(info["recommended_method"], str)
    assert len(info["recommended_method"]) > 0


# ===================================================================
# Test trapezoid() Re-export
# ===================================================================


@pytest.mark.unit
def test_trapezoid_basic_integration():
    """Test trapezoid() basic numerical integration."""
    # Integrate y = x from 0 to 1 -> expected result = 0.5
    x = np.linspace(0, 1, 100)
    y = x
    result = trapezoid(y, x=x)
    assert abs(result - 0.5) < 1e-3


@pytest.mark.unit
def test_trapezoid_with_dx():
    """Test trapezoid() with dx parameter."""
    # Integrate y = x^2 from 0 to 1
    y = np.linspace(0, 1, 100) ** 2
    result = trapezoid(y, dx=1 / 99)
    assert abs(result - 1 / 3) < 1e-2


@pytest.mark.unit
def test_trapezoid_array_integration():
    """Test trapezoid() with array input."""
    y = np.ones(100)
    x = np.linspace(0, 1, 100)
    result = trapezoid(y, x=x)
    assert abs(result - 1.0) < 1e-3


@pytest.mark.unit
def test_trapezoid_2d_integration():
    """Test trapezoid() with 2D array."""
    y = np.ones((10, 20))
    x = np.linspace(0, 1, 20)
    result = trapezoid(y, x=x, axis=1)
    assert result.shape == (10,)
    assert np.allclose(result, 1.0)


# ===================================================================
# Test Module Exports
# ===================================================================


@pytest.mark.unit
def test_module_has_all_attribute():
    """Test module has __all__ attribute."""
    from mfgarchon.utils.numerical import integration

    assert hasattr(integration, "__all__")
    assert "get_integration_info" in integration.__all__
    assert "trapezoid" in integration.__all__


@pytest.mark.unit
def test_module_docstring():
    """Test module has docstring."""
    from mfgarchon.utils.numerical import integration

    assert integration.__doc__ is not None
    assert "Integration utilities" in integration.__doc__


# ===================================================================
# Test Backward Compatibility
# ===================================================================


# ===================================================================
# Test Integration with numpy_compat
# ===================================================================


@pytest.mark.unit
def test_integration_uses_numpy_compat():
    """Test that integration module uses numpy_compat."""
    # get_integration_info should return same info as get_numpy_info
    from mfgarchon.utils.numpy_compat import get_numpy_info

    integration_info = get_integration_info()
    numpy_info = get_numpy_info()

    # Should be identical since it's a wrapper
    assert integration_info == numpy_info


@pytest.mark.unit
def test_trapezoid_matches_numpy_compat():
    """Test that trapezoid matches numpy_compat version."""
    from mfgarchon.utils.numpy_compat import trapezoid as numpy_trapezoid

    x = np.linspace(0, 1, 100)
    y = x**2

    result1 = trapezoid(y, x=x)
    result2 = numpy_trapezoid(y, x=x)

    assert abs(result1 - result2) < 1e-15


# ===================================================================
# Test Function Equivalence
# ===================================================================


@pytest.mark.unit
def test_get_integration_info_consistency():
    """Test get_integration_info() returns consistent results."""
    info1 = get_integration_info()
    info2 = get_integration_info()

    # Multiple calls should return same info
    assert info1 == info2
    assert info1["numpy_version"] == info2["numpy_version"]


@pytest.mark.unit
def test_trapezoid_deterministic():
    """Test trapezoid() is deterministic."""
    y = np.random.rand(100)
    x = np.linspace(0, 1, 100)

    result1 = trapezoid(y, x=x)
    result2 = trapezoid(y, x=x)

    assert result1 == result2


# ===================================================================
# Test Edge Cases
# ===================================================================


@pytest.mark.unit
def test_trapezoid_empty_array():
    """Test trapezoid() with empty array."""
    y = np.array([])
    result = trapezoid(y)
    assert isinstance(result, (int, float, np.number))


@pytest.mark.unit
def test_trapezoid_single_point():
    """Test trapezoid() with single point."""
    y = np.array([1.0])
    result = trapezoid(y)
    assert result == 0.0


@pytest.mark.unit
def test_trapezoid_negative_values():
    """Test trapezoid() with negative values."""
    x = np.linspace(-1, 1, 100)
    y = x
    result = trapezoid(y, x=x)
    assert abs(result) < 1e-10


# ===================================================================
# Test Type Annotations
# ===================================================================

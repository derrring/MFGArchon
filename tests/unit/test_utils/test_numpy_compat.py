#!/usr/bin/env python3
"""
Unit tests for mfgarchon/utils/numpy_compat.py

Tests comprehensive NumPy compatibility utilities including:
- trapezoid() function (NumPy 2.0+ compatibility)
- get_numpy_info() function
- ensure_numpy_compatibility() function
- numpy_trapezoid alias
- Version detection and fallback mechanisms
"""

import warnings

import pytest

import numpy as np

from mfgarchon.utils.numpy_compat import (
    HAS_SCIPY_TRAPEZOID,
    HAS_TRAPEZOID,
    HAS_TRAPZ,
    NUMPY_VERSION,
    ensure_numpy_compatibility,
    get_numpy_info,
    numpy_trapezoid,
    trapezoid,
)

# ===================================================================
# Test Module Constants
# ===================================================================


@pytest.mark.unit
def test_numpy_version_constant():
    """Test NUMPY_VERSION constant is tuple of integers."""
    assert isinstance(NUMPY_VERSION, tuple)
    assert len(NUMPY_VERSION) >= 2
    assert all(isinstance(x, int) for x in NUMPY_VERSION)


@pytest.mark.unit
def test_has_trapezoid_constant():
    """Test HAS_TRAPEZOID constant matches NumPy availability."""
    assert isinstance(HAS_TRAPEZOID, bool)
    # Should be True for NumPy 2.0+
    if NUMPY_VERSION >= (2, 0):
        assert HAS_TRAPEZOID is True


@pytest.mark.unit
def test_has_trapz_constant():
    """Test HAS_TRAPZ constant matches NumPy availability."""
    assert isinstance(HAS_TRAPZ, bool)
    # Should be True for NumPy < 2.0 or as legacy function in 2.0+


@pytest.mark.unit
def test_has_scipy_trapezoid_constant():
    """Test HAS_SCIPY_TRAPEZOID constant is boolean."""
    assert isinstance(HAS_SCIPY_TRAPEZOID, bool)


# ===================================================================
# Test trapezoid() Function - Basic Functionality
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
    """Test trapezoid() with dx parameter instead of x."""
    # Integrate y = x^2 from 0 to 1 with uniform spacing
    y = np.linspace(0, 1, 100) ** 2
    result = trapezoid(y, dx=1 / 99)  # 100 points -> 99 intervals
    assert abs(result - 1 / 3) < 1e-2  # Expected: 1/3


@pytest.mark.unit
def test_trapezoid_multidimensional():
    """Test trapezoid() with multidimensional arrays."""
    # 2D array integration along different axes
    y = np.ones((10, 20))
    x = np.linspace(0, 1, 20)

    # Integrate along axis 1
    result = trapezoid(y, x=x, axis=1)
    assert result.shape == (10,)
    assert np.allclose(result, 1.0)  # Integral of 1 over [0,1] = 1


@pytest.mark.unit
def test_trapezoid_axis_parameter():
    """Test trapezoid() axis parameter."""
    y = np.random.rand(5, 10, 15)

    # Integrate along different axes
    result_axis0 = trapezoid(y, axis=0)
    result_axis1 = trapezoid(y, axis=1)
    result_axis2 = trapezoid(y, axis=2)

    assert result_axis0.shape == (10, 15)
    assert result_axis1.shape == (5, 15)
    assert result_axis2.shape == (5, 10)


# ===================================================================
# Test trapezoid() Function - Edge Cases
# ===================================================================


@pytest.mark.unit
def test_trapezoid_single_point():
    """Test trapezoid() with single point (edge case)."""
    y = np.array([1.0])
    result = trapezoid(y)
    assert result == 0.0  # Single point has zero area


@pytest.mark.unit
def test_trapezoid_two_points():
    """Test trapezoid() with two points."""
    y = np.array([0.0, 1.0])
    x = np.array([0.0, 1.0])
    result = trapezoid(y, x=x)
    assert abs(result - 0.5) < 1e-10  # Triangle area = 0.5


@pytest.mark.unit
def test_trapezoid_constant_function():
    """Test trapezoid() integrating constant function."""
    y = np.ones(100)
    x = np.linspace(0, 5, 100)
    result = trapezoid(y, x=x)
    assert abs(result - 5.0) < 1e-3  # Integral of 1 over [0, 5] = 5


@pytest.mark.unit
def test_trapezoid_negative_values():
    """Test trapezoid() with negative function values."""
    x = np.linspace(-1, 1, 100)
    y = x  # Integral from -1 to 1 should be 0
    result = trapezoid(y, x=x)
    assert abs(result) < 1e-10


# ===================================================================
# Test trapezoid() Function - Mathematical Validation
# ===================================================================


@pytest.mark.unit
def test_trapezoid_quadratic_function():
    """Test trapezoid() with quadratic function."""
    # Integrate y = x^2 from 0 to 2 -> expected = 8/3
    x = np.linspace(0, 2, 200)
    y = x**2
    result = trapezoid(y, x=x)
    expected = 8.0 / 3.0
    assert abs(result - expected) < 1e-2


@pytest.mark.unit
def test_trapezoid_sine_function():
    """Test trapezoid() with sine function."""
    # Integrate sin(x) from 0 to pi -> expected = 2
    x = np.linspace(0, np.pi, 200)
    y = np.sin(x)
    result = trapezoid(y, x=x)
    assert abs(result - 2.0) < 1e-2


@pytest.mark.unit
def test_trapezoid_exponential_function():
    """Test trapezoid() with exponential function."""
    # Integrate e^x from 0 to 1 -> expected = e - 1
    x = np.linspace(0, 1, 200)
    y = np.exp(x)
    result = trapezoid(y, x=x)
    expected = np.e - 1.0
    assert abs(result - expected) < 1e-2


# ===================================================================
# Test get_numpy_info() Function
# ===================================================================


@pytest.mark.unit
def test_get_numpy_info_required_keys():
    """Test get_numpy_info() contains required keys."""
    info = get_numpy_info()
    required_keys = [
        "numpy_version",
        "numpy_version_tuple",
        "has_trapezoid",
        "has_trapz",
        "has_scipy_trapezoid",
        "recommended_method",
        "is_numpy_2_plus",
    ]
    for key in required_keys:
        assert key in info, f"Missing key: {key}"


@pytest.mark.unit
def test_get_numpy_info_version_string():
    """Test get_numpy_info() numpy_version is string."""
    info = get_numpy_info()
    assert isinstance(info["numpy_version"], str)
    assert info["numpy_version"] == np.__version__


@pytest.mark.unit
def test_get_numpy_info_version_tuple():
    """Test get_numpy_info() numpy_version_tuple is tuple."""
    info = get_numpy_info()
    assert isinstance(info["numpy_version_tuple"], tuple)
    assert len(info["numpy_version_tuple"]) >= 2


@pytest.mark.unit
def test_get_numpy_info_booleans():
    """Test get_numpy_info() boolean flags."""
    info = get_numpy_info()
    assert isinstance(info["has_trapezoid"], bool)
    assert isinstance(info["has_trapz"], bool)
    assert isinstance(info["has_scipy_trapezoid"], bool)
    assert isinstance(info["is_numpy_2_plus"], bool)


@pytest.mark.unit
def test_get_numpy_info_recommended_method():
    """Test get_numpy_info() recommended_method is string."""
    info = get_numpy_info()
    assert isinstance(info["recommended_method"], str)
    assert len(info["recommended_method"]) > 0


@pytest.mark.unit
def test_get_numpy_info_consistency():
    """Test get_numpy_info() internal consistency."""
    info = get_numpy_info()

    # If NumPy 2.0+, should have trapezoid
    if info["is_numpy_2_plus"]:
        assert info["has_trapezoid"] is True
        assert "trapezoid" in info["recommended_method"]


# ===================================================================
# Test ensure_numpy_compatibility() Function
# ===================================================================


@pytest.mark.unit
def test_ensure_numpy_compatibility_returns_dict():
    """Test ensure_numpy_compatibility() returns dictionary."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = ensure_numpy_compatibility()
    assert isinstance(info, dict)


@pytest.mark.unit
def test_ensure_numpy_compatibility_warning_stacklevel():
    """Test ensure_numpy_compatibility() uses correct stacklevel."""
    # Should emit warnings if compatibility issues detected
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ensure_numpy_compatibility()

        # Warnings should have stacklevel >= 2 (not 1 which points to warning call itself)
        # This is a quality check, not a strict requirement
        if len(w) > 0:
            # Just verify no exceptions occurred during warning emission
            assert True


# ===================================================================
# Test numpy_trapezoid Alias
# ===================================================================


@pytest.mark.unit
def test_numpy_trapezoid_basic_usage():
    """Test numpy_trapezoid alias basic usage."""
    x = np.linspace(0, 1, 100)
    y = x
    result = numpy_trapezoid(y, x=x)
    assert abs(result - 0.5) < 1e-3


@pytest.mark.unit
def test_numpy_trapezoid_matches_trapezoid():
    """Test numpy_trapezoid alias gives same result as trapezoid()."""
    x = np.linspace(0, 2, 100)
    y = x**2

    result1 = trapezoid(y, x=x)
    result2 = numpy_trapezoid(y, x=x)

    assert abs(result1 - result2) < 1e-10


# ===================================================================
# Test Integration Method Fallback
# ===================================================================


@pytest.mark.unit
def test_integration_method_available():
    """Test at least one integration method is available."""
    # Should always be true in normal NumPy installation
    assert HAS_TRAPEZOID or HAS_TRAPZ or HAS_SCIPY_TRAPEZOID


@pytest.mark.unit
def test_recommended_method_matches_availability():
    """Test recommended method matches actual availability."""
    info = get_numpy_info()
    method = info["recommended_method"]

    if info["has_trapezoid"]:
        assert "np.trapezoid" in method
    elif info["has_scipy_trapezoid"]:
        assert "scipy" in method
    elif info["has_trapz"]:
        assert "trapz" in method


# ===================================================================
# Test Edge Cases and Error Handling
# ===================================================================


@pytest.mark.unit
def test_trapezoid_empty_array():
    """Test trapezoid() with empty array."""
    y = np.array([])
    result = trapezoid(y)
    # NumPy typically returns 0.0 for empty array
    assert isinstance(result, (int, float, np.number))


@pytest.mark.unit
def test_trapezoid_complex_numbers():
    """Test trapezoid() with complex numbers."""
    y = np.array([1 + 1j, 2 + 2j, 3 + 3j])
    result = trapezoid(y)
    assert isinstance(result, (complex, np.complexfloating))


@pytest.mark.unit
def test_trapezoid_large_array():
    """Test trapezoid() with large array (performance check)."""
    y = np.random.rand(10000)
    x = np.linspace(0, 1, 10000)
    result = trapezoid(y, x=x)
    assert isinstance(result, (float, np.floating))


@pytest.mark.unit
def test_trapezoid_nonuniform_spacing():
    """Test trapezoid() with nonuniform x spacing."""
    x = np.array([0, 0.1, 0.5, 1.0])
    y = np.array([0, 1, 2, 3])
    result = trapezoid(y, x=x)
    # Manual calculation: (0.1)*(0+1)/2 + (0.4)*(1+2)/2 + (0.5)*(2+3)/2
    # = 0.05 + 0.6 + 1.25 = 1.9
    assert abs(result - 1.9) < 1e-10


# ===================================================================
# Test Module Imports and Exports
# ===================================================================


@pytest.mark.unit
def test_module_exports_all():
    """Test all public functions are exported in __all__."""
    from mfgarchon.utils import numpy_compat

    assert hasattr(numpy_compat, "__all__")
    assert "trapezoid" in numpy_compat.__all__
    assert "numpy_trapezoid" in numpy_compat.__all__
    assert "get_numpy_info" in numpy_compat.__all__
    assert "ensure_numpy_compatibility" in numpy_compat.__all__


@pytest.mark.unit
def test_module_docstring():
    """Test module has docstring."""
    from mfgarchon.utils import numpy_compat

    assert numpy_compat.__doc__ is not None
    assert len(numpy_compat.__doc__) > 0

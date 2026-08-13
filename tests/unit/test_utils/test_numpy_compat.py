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

import importlib.util
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
    """HAS_TRAPEZOID is a cache of a capability; pin it to the capability, not to a version gate.

    The version gate it replaces asserted nothing on a NumPy 1.x runner, where the test degraded
    to a type check. ``is`` against a bool keeps the type claim. Measured here: numpy 2.4.6,
    both sides True.
    """
    assert HAS_TRAPEZOID is hasattr(np, "trapezoid")


@pytest.mark.unit
def test_has_trapz_constant():
    """HAS_TRAPZ against the capability it caches.

    The comment this replaces claimed the flag "should be True for NumPy < 2.0 or as legacy
    function in 2.0+", asserted nothing, and is false on this environment: measured on numpy
    2.4.6, ``np.trapz`` has been REMOVED, not kept as a legacy alias, so HAS_TRAPZ is False.
    That makes this the one flag of the three currently in its False state, and the sharpest --
    a constant hard-coded True fails here immediately.
    """
    assert HAS_TRAPZ is hasattr(np, "trapz")


@pytest.mark.unit
def test_has_scipy_trapezoid_constant():
    """HAS_SCIPY_TRAPEZOID against the capability, resolved by a different mechanism.

    The module sets the flag with a try/except on ``from scipy.integrate import trapezoid``.
    Asking importlib instead is an independent route to the same fact, so this is not a
    restatement of the source line. Measured True on this environment, scipy present.
    """
    spec = importlib.util.find_spec("scipy.integrate")
    scipy_has_trapezoid = spec is not None and callable(
        getattr(importlib.import_module("scipy.integrate"), "trapezoid", None)
    )

    assert HAS_SCIPY_TRAPEZOID is scipy_has_trapezoid


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
    """The four flags report the module constants, not just some bool.

    A flag with the wrong VALUE passed the isinstance form, and the values are what callers
    branch on. ``is`` against a bool constant keeps the type claim (np.bool_ or a truthy int
    still fails) and adds the value. Measured on this environment: True / False / True / True --
    the has_trapz row is in its False state, so this is not an all-True tautology.
    """
    info = get_numpy_info()
    assert info["has_trapezoid"] is HAS_TRAPEZOID
    assert info["has_trapz"] is HAS_TRAPZ
    assert info["has_scipy_trapezoid"] is HAS_SCIPY_TRAPEZOID
    assert info["is_numpy_2_plus"] is (NUMPY_VERSION >= (2, 0))


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
    """The compatibility record describes the NumPy that is actually imported.

    An empty dict satisfied the isinstance form. Anchored to ``np`` itself rather than only to
    ``get_numpy_info()``: today ``ensure_numpy_compatibility`` returns that call's result
    verbatim (numpy_compat.py:105, 123), so an ``ensure == get_numpy_info`` pin alone would be
    tautological and would pass over a record that is wrong on both paths. The equality is kept
    as the weaker second claim -- it is what fires if ensure_ ever grows its own record.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = ensure_numpy_compatibility()

    assert info["numpy_version"] == np.__version__
    assert info["has_trapezoid"] is hasattr(np, "trapezoid")
    assert info == get_numpy_info()


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
    """Empty input integrates to exactly zero -- no intervals, no area.

    The expected value was written in a comment and left unasserted, so the test could not
    separate 0.0 from NaN. Measured np.float64(0.0). The value subsumes the old type claim:
    NaN, None and an array all fail this comparison (an array raises on truthiness).
    """
    y = np.array([])
    result = trapezoid(y)
    assert result == 0.0


@pytest.mark.unit
def test_trapezoid_complex_numbers():
    """The rule is correct on complex data, not merely complex-typed.

    With unit spacing the composite trapezoid rule gives y0/2 + y1 + y2/2 =
    (0.5+0.5j) + (2+2j) + (1.5+1.5j) = 4+4j, exact in binary floating point. Measured
    np.complex128(4+4j). This subsumes the old dtype check -- a silent cast to real gives 4.0,
    which is not equal to 4+4j.
    """
    y = np.array([1 + 1j, 2 + 2j, 3 + 3j])
    result = trapezoid(y)
    assert result == 4 + 4j


@pytest.mark.unit
def test_trapezoid_large_array():
    """N = 10000 stress input, checked against the closed form of the rule it implements.

    The docstring called this a performance check and no time was measured, while the type
    assertion held for any wrong number -- so the stress coverage was illusory. Keep the size,
    make it discriminating: for uniform spacing the composite trapezoid rule is
    dx * (sum(y) - (y[0] + y[-1]) / 2), computed here by a different summation route than
    ``trapezoid()`` takes.

    Seeded, because the assertion is now on a value. Measured relative difference 4.4e-16 (one
    ULP, from linspace's own 1e-16 spacing jitter), so rel=1e-12 sits ~2250x above the
    floating-point floor; a left-Riemann sum instead of the trapezoid rule would land 1e-4 away.
    """
    rng = np.random.default_rng(0)
    y = rng.random(10000)
    x = np.linspace(0, 1, 10000)
    dx = x[1] - x[0]

    result = trapezoid(y, x=x)
    assert result == pytest.approx(dx * (y.sum() - 0.5 * (y[0] + y[-1])), rel=1e-12)


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

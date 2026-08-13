#!/usr/bin/env python3
"""
Unit tests for mfgarchon/utils/aux_func.py

Tests auxiliary mathematical functions including:
- ppart() - Positive part function
- npart() - Negative part function
- Scalar inputs (float, int)
- Array inputs (NumPy arrays)
- Edge cases (zero, boundary values)
- Mathematical properties and equivalences
"""

import pytest

import numpy as np

from mfgarchon.utils.aux_func import npart, ppart

# ===================================================================
# Test ppart() - Positive Part Function
# ===================================================================


@pytest.mark.unit
def test_ppart_positive_scalar():
    """Test ppart() with positive scalar."""
    assert ppart(5.0) == 5.0
    assert ppart(1.0) == 1.0
    assert ppart(0.1) == 0.1


@pytest.mark.unit
def test_ppart_negative_scalar():
    """Test ppart() with negative scalar."""
    assert ppart(-5.0) == 0.0
    assert ppart(-1.0) == 0.0
    assert ppart(-0.1) == 0.0


@pytest.mark.unit
def test_ppart_zero():
    """Test ppart() with zero."""
    assert ppart(0.0) == 0.0
    assert ppart(-0.0) == 0.0


@pytest.mark.unit
def test_ppart_integer_input():
    """Test ppart() with integer input."""
    # Should work with integers via NumPy
    result = ppart(5)
    assert result == 5
    result = ppart(-3)
    assert result == 0


@pytest.mark.unit
def test_ppart_array_positive():
    """Test ppart() with array of positive values."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ppart(arr)
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.array_equal(result, expected)


@pytest.mark.unit
def test_ppart_array_negative():
    """Test ppart() with array of negative values."""
    arr = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
    result = ppart(arr)
    expected = np.zeros(5)
    assert np.array_equal(result, expected)


@pytest.mark.unit
def test_ppart_array_mixed():
    """Test ppart() with array of mixed values."""
    arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = ppart(arr)
    expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    assert np.array_equal(result, expected)


@pytest.mark.unit
def test_ppart_array_2d():
    """Test ppart() with 2D array."""
    arr = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]])
    result = ppart(arr)
    expected = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 3.0]])
    assert np.array_equal(result, expected)


@pytest.mark.unit
def test_ppart_large_values():
    """Test ppart() with large values."""
    assert ppart(1e10) == 1e10
    assert ppart(-1e10) == 0.0


@pytest.mark.unit
def test_ppart_small_values():
    """Test ppart() with small values."""
    assert ppart(1e-10) == pytest.approx(1e-10)
    assert ppart(-1e-10) == 0.0


@pytest.mark.unit
def test_ppart_mathematical_property():
    """Test ppart(x) = max(x, 0) mathematical property."""
    x_values = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
    result = ppart(x_values)
    expected = np.maximum(x_values, 0.0)
    assert np.array_equal(result, expected)


@pytest.mark.unit
def test_ppart_squared_equivalence():
    """Test ppart(x)^2 = max(x, 0)^2 equivalence."""
    arr = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    ppart_sq = ppart(arr) ** 2
    max_sq = np.maximum(arr, 0.0) ** 2
    assert np.allclose(ppart_sq, max_sq)


# ===================================================================
# Test npart() - Negative Part Function
# ===================================================================


@pytest.mark.unit
def test_npart_positive_scalar():
    """Test npart() with positive scalar."""
    assert npart(5.0) == 0.0
    assert npart(1.0) == 0.0
    assert npart(0.1) == 0.0


@pytest.mark.unit
def test_npart_negative_scalar():
    """Test npart() with negative scalar."""
    assert npart(-5.0) == 5.0
    assert npart(-1.0) == 1.0
    assert npart(-0.1) == pytest.approx(0.1)


@pytest.mark.unit
def test_npart_zero():
    """Test npart() with zero."""
    assert npart(0.0) == 0.0
    assert npart(-0.0) == 0.0


@pytest.mark.unit
def test_npart_integer_input():
    """Test npart() with integer input."""
    result = npart(5)
    assert result == 0
    result = npart(-3)
    assert result == 3


@pytest.mark.unit
def test_npart_array_positive():
    """Test npart() with array of positive values."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = npart(arr)
    expected = np.zeros(5)
    assert np.array_equal(result, expected)


@pytest.mark.unit
def test_npart_array_negative():
    """Test npart() with array of negative values."""
    arr = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
    result = npart(arr)
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.array_equal(result, expected)


@pytest.mark.unit
def test_npart_array_mixed():
    """Test npart() with array of mixed values."""
    arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = npart(arr)
    expected = np.array([2.0, 1.0, 0.0, 0.0, 0.0])
    assert np.array_equal(result, expected)


@pytest.mark.unit
def test_npart_array_2d():
    """Test npart() with 2D array."""
    arr = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]])
    result = npart(arr)
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert np.array_equal(result, expected)


@pytest.mark.unit
def test_npart_large_values():
    """Test npart() with large values."""
    assert npart(1e10) == 0.0
    assert npart(-1e10) == 1e10


@pytest.mark.unit
def test_npart_small_values():
    """Test npart() with small values."""
    assert npart(1e-10) == 0.0
    assert npart(-1e-10) == pytest.approx(1e-10)


@pytest.mark.unit
def test_npart_mathematical_property():
    """Test npart(x) = max(-x, 0) mathematical property."""
    x_values = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
    result = npart(x_values)
    expected = np.maximum(-x_values, 0.0)
    assert np.array_equal(result, expected)


@pytest.mark.unit
def test_npart_squared_equivalence():
    """Test npart(x)^2 = min(x, 0)^2 equivalence."""
    arr = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    npart_sq = npart(arr) ** 2
    min_sq = np.minimum(arr, 0.0) ** 2
    assert np.allclose(npart_sq, min_sq)


# ===================================================================
# Test ppart() and npart() Relationships
# ===================================================================


@pytest.mark.unit
def test_ppart_npart_complementary():
    """Test ppart(x) + npart(x) = |x| (complementary property)."""
    x_values = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
    result = ppart(x_values) + npart(x_values)
    expected = np.abs(x_values)
    assert np.allclose(result, expected)


@pytest.mark.unit
def test_ppart_npart_sum_identity():
    """Test ppart(x) - npart(x) = x (sum identity)."""
    x_values = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
    result = ppart(x_values) - npart(x_values)
    assert np.allclose(result, x_values)


@pytest.mark.unit
def test_ppart_npart_orthogonal():
    """Test ppart(x) * npart(x) = 0 (orthogonality)."""
    x_values = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
    result = ppart(x_values) * npart(x_values)
    expected = np.zeros_like(x_values)
    assert np.array_equal(result, expected)


@pytest.mark.unit
def test_ppart_npart_squared_sum():
    """Test ppart(x)^2 + npart(x)^2 = x^2."""
    x_values = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
    result = ppart(x_values) ** 2 + npart(x_values) ** 2
    expected = x_values**2
    assert np.allclose(result, expected)


# ===================================================================
# Test Edge Cases and Special Values
# ===================================================================


@pytest.mark.unit
def test_ppart_npart_empty_array():
    """Test ppart() and npart() with empty array."""
    arr = np.array([])
    assert ppart(arr).size == 0
    assert npart(arr).size == 0


@pytest.mark.unit
def test_ppart_npart_single_element():
    """Test ppart() and npart() with single element array."""
    arr = np.array([3.0])
    assert np.array_equal(ppart(arr), np.array([3.0]))
    assert np.array_equal(npart(arr), np.array([0.0]))

    arr = np.array([-3.0])
    assert np.array_equal(ppart(arr), np.array([0.0]))
    assert np.array_equal(npart(arr), np.array([3.0]))


@pytest.mark.unit
def test_ppart_npart_inf_values():
    """Test ppart() and npart() with infinity."""
    assert ppart(np.inf) == np.inf
    assert ppart(-np.inf) == 0.0
    assert npart(np.inf) == 0.0
    assert npart(-np.inf) == np.inf


@pytest.mark.unit
def test_ppart_npart_nan_values():
    """Test ppart() and npart() with NaN."""
    # NaN behavior: max(NaN, 0) = NaN
    result_ppart = ppart(np.nan)
    result_npart = npart(np.nan)
    assert np.isnan(result_ppart)
    assert np.isnan(result_npart)


@pytest.mark.unit
def test_ppart_npart_very_close_to_zero():
    """Test ppart() and npart() with values very close to zero."""
    eps = 1e-15
    assert ppart(eps) == pytest.approx(eps, abs=1e-16)
    assert ppart(-eps) == 0.0
    assert npart(eps) == 0.0
    assert npart(-eps) == pytest.approx(eps, abs=1e-16)


# ===================================================================
# Test Return Types
# ===================================================================


@pytest.mark.unit
def test_ppart_return_type_scalar():
    """Test ppart() returns correct type for scalar input."""
    result = ppart(5.0)
    assert isinstance(result, (float, np.floating))


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ppart_return_type_array(dtype):
    """Test ppart() returns ndarray for array input, preserving the input float dtype."""
    # Parametrised so the dtype claim is falsifiable: with float64 only, it holds for any
    # implementation. Integer input is deliberately excluded -- np.maximum(int32, 0.0) promotes
    # to float64 under NEP 50, so the preservation claim is true for float dtypes only.
    arr = np.array([1.0, 2.0, 3.0], dtype=dtype)
    result = ppart(arr)
    assert isinstance(result, np.ndarray)
    assert result.dtype == arr.dtype


@pytest.mark.unit
def test_npart_return_type_scalar():
    """Test npart() returns correct type for scalar input."""
    result = npart(-5.0)
    assert isinstance(result, (float, np.floating))


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_npart_return_type_array(dtype):
    """Test npart() returns ndarray for array input, preserving the input float dtype."""
    # npart negates before the maximum, so it is a separate promotion path from ppart.
    arr = np.array([-1.0, -2.0, -3.0], dtype=dtype)
    result = npart(arr)
    assert isinstance(result, np.ndarray)
    assert result.dtype == arr.dtype


# ===================================================================
# Test Module Exports
# ===================================================================


@pytest.mark.unit
def test_module_docstrings():
    """Test functions have docstrings."""
    assert ppart.__doc__ is not None
    assert "Positive part" in ppart.__doc__
    assert npart.__doc__ is not None
    assert "Negative part" in npart.__doc__

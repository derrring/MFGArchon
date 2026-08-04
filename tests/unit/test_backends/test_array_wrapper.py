"""
Unit tests for Backend Array Wrapper.

Tests BackendArray and BackendArrayWrapper classes that provide
NumPy-compatible interface for backend arrays.
"""

import pytest

import numpy as np

from mfgarchon.backends.array_wrapper import (
    BackendArray,
    BackendArrayWrapper,
    create_array_wrapper,
    patch_numpy_for_backend,
)
from mfgarchon.backends.numpy_backend import NumPyBackend


@pytest.fixture
def numpy_backend():
    """Create NumPy backend instance."""
    return NumPyBackend()


@pytest.fixture
def backend_array(numpy_backend):
    """Create a simple BackendArray for testing."""
    data = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    return BackendArray(data, numpy_backend)


@pytest.fixture
def array_wrapper(numpy_backend):
    """Create BackendArrayWrapper instance."""
    return BackendArrayWrapper(numpy_backend)


class TestBackendArrayInitialization:
    """Test BackendArray initialization and properties."""

    def test_initialization(self, numpy_backend):
        """Test BackendArray initialization."""
        data = np.array([1, 2, 3])
        arr = BackendArray(data, numpy_backend)
        assert arr._backend is numpy_backend
        assert arr._module is np
        np.testing.assert_array_equal(arr._data, data)

    def test_data_property(self, backend_array):
        """Test data property access."""
        data = backend_array.data
        np.testing.assert_array_equal(data, [1, 2, 3, 4, 5])

    def test_shape_property(self, backend_array):
        """Test shape property."""
        assert backend_array.shape == (5,)

    def test_dtype_property(self, backend_array):
        """Test dtype property."""
        assert backend_array.dtype == np.float64

    def test_ndim_property(self, backend_array):
        """Test ndim property."""
        assert backend_array.ndim == 1

    def test_ndim_2d_array(self, numpy_backend):
        """Test ndim for 2D array."""
        data = np.array([[1, 2], [3, 4]])
        arr = BackendArray(data, numpy_backend)
        assert arr.ndim == 2

    def test_size_property(self, backend_array):
        """Test size property."""
        assert backend_array.size == 5

    def test_size_2d_array(self, numpy_backend):
        """Test size for 2D array."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        arr = BackendArray(data, numpy_backend)
        assert arr.size == 6


class TestBackendArrayArithmetic:
    """Test arithmetic operations on BackendArray."""

    def test_add_scalar(self, backend_array):
        """Test addition with scalar."""
        result = backend_array + 10
        assert isinstance(result, BackendArray)
        np.testing.assert_array_equal(result.data, [11, 12, 13, 14, 15])

    def test_add_array(self, backend_array, numpy_backend):
        """Test addition with another BackendArray."""
        other = BackendArray(np.array([5, 4, 3, 2, 1]), numpy_backend)
        result = backend_array + other
        assert isinstance(result, BackendArray)
        np.testing.assert_array_equal(result.data, [6, 6, 6, 6, 6])

    def test_add_numpy_array(self, backend_array):
        """Test addition with numpy array."""
        other = np.array([1, 1, 1, 1, 1])
        result = backend_array + other
        assert isinstance(result, BackendArray)
        np.testing.assert_array_equal(result.data, [2, 3, 4, 5, 6])

    def test_radd_scalar(self, backend_array):
        """Test reverse addition with scalar."""
        result = 10 + backend_array
        assert isinstance(result, BackendArray)
        np.testing.assert_array_equal(result.data, [11, 12, 13, 14, 15])

    def test_sub_scalar(self, backend_array):
        """Test subtraction with scalar."""
        result = backend_array - 1
        assert isinstance(result, BackendArray)
        np.testing.assert_array_equal(result.data, [0, 1, 2, 3, 4])

    def test_rsub_scalar(self, backend_array):
        """Test reverse subtraction."""
        result = 10 - backend_array
        assert isinstance(result, BackendArray)
        np.testing.assert_array_equal(result.data, [9, 8, 7, 6, 5])

    def test_mul_scalar(self, backend_array):
        """Test multiplication with scalar."""
        result = backend_array * 2
        assert isinstance(result, BackendArray)
        np.testing.assert_array_equal(result.data, [2, 4, 6, 8, 10])

    def test_rmul_scalar(self, backend_array):
        """Test reverse multiplication."""
        result = 3 * backend_array
        assert isinstance(result, BackendArray)
        np.testing.assert_array_equal(result.data, [3, 6, 9, 12, 15])

    def test_truediv_scalar(self, backend_array):
        """Test division with scalar."""
        result = backend_array / 2
        assert isinstance(result, BackendArray)
        np.testing.assert_array_equal(result.data, [0.5, 1, 1.5, 2, 2.5])

    def test_rtruediv_scalar(self, backend_array):
        """Test reverse division."""
        result = 12 / backend_array
        assert isinstance(result, BackendArray)
        np.testing.assert_array_equal(result.data, [12, 6, 4, 3, 2.4])

    def test_pow_scalar(self, backend_array):
        """Test power operation with scalar."""
        result = backend_array**2
        assert isinstance(result, BackendArray)
        np.testing.assert_array_equal(result.data, [1, 4, 9, 16, 25])


class TestBackendArrayIndexing:
    """Test indexing operations on BackendArray."""

    def test_getitem_single_index(self, backend_array):
        """Test single element access."""
        result = backend_array[2]
        assert isinstance(result, BackendArray)
        assert result.data == 3

    def test_getitem_slice(self, backend_array):
        """Test slice access."""
        result = backend_array[1:4]
        assert isinstance(result, BackendArray)
        np.testing.assert_array_equal(result.data, [2, 3, 4])

    def test_getitem_fancy_indexing(self, numpy_backend):
        """Test fancy indexing."""
        data = np.array([10, 20, 30, 40, 50])
        arr = BackendArray(data, numpy_backend)
        result = arr[[0, 2, 4]]
        assert isinstance(result, BackendArray)
        np.testing.assert_array_equal(result.data, [10, 30, 50])

    def test_setitem_single_index(self, backend_array):
        """Test single element assignment."""
        backend_array[2] = 99
        assert backend_array.data[2] == 99

    def test_setitem_slice(self, backend_array):
        """Test slice assignment."""
        backend_array[1:3] = [88, 99]
        np.testing.assert_array_equal(backend_array.data[1:3], [88, 99])

    def test_setitem_with_backend_array(self, backend_array, numpy_backend):
        """Test assignment with BackendArray value."""
        value = BackendArray(np.array([100, 200]), numpy_backend)
        backend_array[1:3] = value
        np.testing.assert_array_equal(backend_array.data[1:3], [100, 200])


class TestBackendArrayMethods:
    """Test BackendArray methods."""

    def test_copy(self, backend_array):
        """Test array copy."""
        copied = backend_array.copy()
        assert isinstance(copied, BackendArray)
        assert copied is not backend_array
        np.testing.assert_array_equal(copied.data, backend_array.data)
        # Verify it's a true copy
        copied.data[0] = 999
        assert backend_array.data[0] != 999

    def test_reshape(self, numpy_backend):
        """Test reshape operation."""
        data = np.arange(12)
        arr = BackendArray(data, numpy_backend)
        reshaped = arr.reshape(3, 4)
        assert isinstance(reshaped, BackendArray)
        assert reshaped.shape == (3, 4)

    def test_flatten(self, numpy_backend):
        """Test flatten operation."""
        data = np.array([[1, 2], [3, 4]])
        arr = BackendArray(data, numpy_backend)
        flat = arr.flatten()
        assert isinstance(flat, BackendArray)
        assert flat.shape == (4,)
        np.testing.assert_array_equal(flat.data, [1, 2, 3, 4])

    def test_sum_no_axis(self, backend_array):
        """Test sum without axis (returns scalar)."""
        result = backend_array.sum()
        # Should be scalar, not BackendArray
        assert not isinstance(result, BackendArray)
        assert result == 15

    def test_sum_with_axis(self, numpy_backend):
        """Test sum along axis (returns BackendArray)."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        arr = BackendArray(data, numpy_backend)
        result = arr.sum(axis=0)
        assert isinstance(result, BackendArray)
        np.testing.assert_array_equal(result.data, [5, 7, 9])

    def test_mean_no_axis(self, backend_array):
        """Test mean without axis (returns scalar)."""
        result = backend_array.mean()
        assert not isinstance(result, BackendArray)
        assert result == 3.0

    def test_mean_with_axis(self, numpy_backend):
        """Test mean along axis."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        arr = BackendArray(data, numpy_backend)
        result = arr.mean(axis=1)
        assert isinstance(result, BackendArray)
        np.testing.assert_array_equal(result.data, [2.0, 5.0])

    def test_max_no_axis(self, backend_array):
        """Test max without axis."""
        result = backend_array.max()
        assert result == 5

    def test_min_no_axis(self, backend_array):
        """Test min without axis."""
        result = backend_array.min()
        assert result == 1


class TestBackendArrayNumPyCompatibility:
    """Test NumPy compatibility features."""

    def test_array_interface(self, backend_array):
        """Test __array__ method for NumPy compatibility."""
        numpy_arr = np.array(backend_array)
        assert isinstance(numpy_arr, np.ndarray)
        np.testing.assert_array_equal(numpy_arr, [1, 2, 3, 4, 5])

    def test_repr_format(self, backend_array):
        """Test __repr__ method."""
        repr_str = repr(backend_array)
        assert "BackendArray" in repr_str
        assert "numpy" in repr_str
        assert "shape=(5,)" in repr_str


class TestBackendArrayWrapperCreation:
    """Test BackendArrayWrapper array creation methods."""

    def test_array_from_list(self, array_wrapper):
        """Test creating array from list."""
        arr = array_wrapper.array([1, 2, 3])
        assert isinstance(arr, BackendArray)
        np.testing.assert_array_equal(arr.data, [1, 2, 3])

    def test_array_with_dtype(self, array_wrapper):
        """Test array creation with custom dtype."""
        arr = array_wrapper.array([1, 2, 3], dtype=np.int32)
        assert arr.dtype == np.int32

    def test_zeros_creation(self, array_wrapper):
        """Test zeros creation."""
        arr = array_wrapper.zeros((3, 4))
        assert isinstance(arr, BackendArray)
        assert arr.shape == (3, 4)
        np.testing.assert_array_equal(arr.data, np.zeros((3, 4)))

    def test_ones_creation(self, array_wrapper):
        """Test ones creation."""
        arr = array_wrapper.ones((2, 3))
        assert isinstance(arr, BackendArray)
        np.testing.assert_array_equal(arr.data, np.ones((2, 3)))

    def test_linspace_creation(self, array_wrapper):
        """Test linspace creation."""
        arr = array_wrapper.linspace(0, 1, 5)
        assert isinstance(arr, BackendArray)
        np.testing.assert_allclose(arr.data, [0, 0.25, 0.5, 0.75, 1.0])


class TestBackendArrayWrapperConversion:
    """Test conversion methods."""

    def test_from_numpy(self, array_wrapper):
        """Test converting from numpy array."""
        numpy_arr = np.array([1, 2, 3, 4])
        backend_arr = array_wrapper.from_numpy(numpy_arr)
        assert isinstance(backend_arr, BackendArray)
        np.testing.assert_array_equal(backend_arr.data, numpy_arr)

    def test_to_numpy_from_backend_array(self, array_wrapper, backend_array):
        """Test converting BackendArray to numpy."""
        numpy_arr = array_wrapper.to_numpy(backend_array)
        assert isinstance(numpy_arr, np.ndarray)
        np.testing.assert_array_equal(numpy_arr, [1, 2, 3, 4, 5])

    def test_to_numpy_from_regular_array(self, array_wrapper):
        """Test converting regular array to numpy."""
        regular_arr = [1, 2, 3]
        numpy_arr = array_wrapper.to_numpy(regular_arr)
        assert isinstance(numpy_arr, np.ndarray)
        np.testing.assert_array_equal(numpy_arr, [1, 2, 3])


class TestBackendArrayWrapperFunctionInterception:
    """Test NumPy function interception via __getattr__."""

    def test_sin_function(self, array_wrapper):
        """Test intercepting np.sin."""
        arr = array_wrapper.array([0, np.pi / 2, np.pi])
        result = array_wrapper.sin(arr)
        assert isinstance(result, BackendArray)
        expected = np.array([0, 1, 0])
        np.testing.assert_allclose(result.data, expected, atol=1e-10)

    def test_cos_function(self, array_wrapper):
        """Test intercepting np.cos."""
        arr = array_wrapper.array([0, np.pi / 2, np.pi])
        result = array_wrapper.cos(arr)
        assert isinstance(result, BackendArray)
        expected = np.array([1, 0, -1])
        np.testing.assert_allclose(result.data, expected, atol=1e-10)

    def test_exp_function(self, array_wrapper):
        """Test intercepting np.exp."""
        arr = array_wrapper.array([0, 1, 2])
        result = array_wrapper.exp(arr)
        assert isinstance(result, BackendArray)
        expected = np.exp([0, 1, 2])
        np.testing.assert_allclose(result.data, expected)

    def test_sqrt_function(self, array_wrapper):
        """Test intercepting np.sqrt."""
        arr = array_wrapper.array([1, 4, 9, 16])
        result = array_wrapper.sqrt(arr)
        assert isinstance(result, BackendArray)
        expected = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(result.data, expected)

    def test_function_with_kwargs(self, array_wrapper):
        """Test function with keyword arguments."""
        arr = array_wrapper.array([[1, 2], [3, 4]])
        # Test sum with axis keyword
        result = array_wrapper.sum(arr, axis=0)
        assert isinstance(result, BackendArray)
        np.testing.assert_array_equal(result.data, [4, 6])


class TestCreateArrayWrapper:
    """Test create_array_wrapper factory function."""

    def test_create_from_string(self):
        """Test creating wrapper from backend string."""
        wrapper = create_array_wrapper("numpy")
        assert isinstance(wrapper, BackendArrayWrapper)
        assert wrapper.backend.name == "numpy"

    def test_create_from_backend_instance(self):
        """Test creating wrapper from backend instance."""
        backend = NumPyBackend(precision="float32")
        wrapper = create_array_wrapper(backend)
        assert isinstance(wrapper, BackendArrayWrapper)
        assert wrapper.backend is backend

    def test_wrapper_functionality(self):
        """Test that created wrapper works correctly."""
        wrapper = create_array_wrapper("numpy")
        arr = wrapper.zeros((3, 3))
        assert isinstance(arr, BackendArray)
        assert arr.shape == (3, 3)


class TestPatchNumpyForBackend:
    """Test monkey-patching NumPy functions."""

    def test_patch_zeros_function(self, numpy_backend):
        """Test patching zeros function."""
        namespace = {}
        patch_numpy_for_backend(numpy_backend, namespace)

        # zeros should now be patched
        assert "zeros" in namespace
        result = namespace["zeros"]((2, 2))
        assert isinstance(result, BackendArray)

    def test_original_functions_preserved(self, numpy_backend):
        """Test that original functions are preserved."""
        namespace = {"zeros": np.zeros}
        patch_numpy_for_backend(numpy_backend, namespace)

        # Original should be saved
        assert "_original_zeros" in namespace
        assert namespace["_original_zeros"] is np.zeros

    def test_patched_sin_function(self, numpy_backend):
        """Test patched trigonometric function."""
        namespace = {}
        patch_numpy_for_backend(numpy_backend, namespace)

        arr = namespace["zeros"](5)  # BackendArray of zeros
        result = namespace["sin"](arr)
        assert isinstance(result, BackendArray)
        np.testing.assert_allclose(result.data, np.zeros(5))

    def test_multiple_patched_functions(self, numpy_backend):
        """Test multiple functions patched correctly."""
        namespace = {}
        patch_numpy_for_backend(numpy_backend, namespace)

        # All these should be patched
        for func_name in ["zeros", "ones", "array", "linspace", "sin", "cos", "exp"]:
            assert func_name in namespace
            assert callable(namespace[func_name])

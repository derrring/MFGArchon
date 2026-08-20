"""
Unit tests for NumPy Backend.

Tests the NumPy backend implementation including array operations,
mathematical functions, linear algebra, and MFG-specific operations.
"""

import warnings

import pytest

import numpy as np

from mfgarchon.backends.numpy_backend import NumPyBackend


@pytest.fixture
def numpy_backend():
    """Create NumPy backend instance with default settings."""
    return NumPyBackend()


@pytest.fixture
def numpy_backend_float32():
    """Create NumPy backend with float32 precision."""
    return NumPyBackend(precision="float32")


class TestNumPyBackendInitialization:
    """Test NumPy backend initialization and configuration."""

    def test_default_initialization(self):
        """Test backend with default parameters."""
        backend = NumPyBackend()
        assert backend.name == "numpy"
        assert backend.device == "cpu"
        assert backend.precision == "float64"
        assert backend.dtype == np.float64

    def test_float32_precision(self):
        """Test backend with float32 precision."""
        backend = NumPyBackend(precision="float32")
        assert backend.dtype == np.float32

    def test_float64_precision(self):
        """Test backend with float64 precision."""
        backend = NumPyBackend(precision="float64")
        assert backend.dtype == np.float64

    def test_device_warning_for_non_cpu(self):
        """Test that non-CPU device triggers warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            backend = NumPyBackend(device="gpu")
            assert len(w) == 1
            assert "only supports CPU" in str(w[0].message)
            assert backend.device == "cpu"

    def test_auto_device_becomes_cpu(self):
        """Test that 'auto' device becomes 'cpu'."""
        backend = NumPyBackend(device="auto")
        assert backend.device == "cpu"


class TestNumPyBackendProperties:
    """Test NumPy backend properties."""

    def test_name_property(self, numpy_backend):
        """Test backend name."""
        assert numpy_backend.name == "numpy"

    def test_array_module_property(self, numpy_backend):
        """Test array module is numpy."""
        assert numpy_backend.array_module is np


class TestNumPyBackendArrayCreation:
    """Test array creation operations."""

    def test_array_from_list(self, numpy_backend):
        """Test creating array from list."""
        arr = numpy_backend.array([1, 2, 3, 4, 5])
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        np.testing.assert_array_equal(arr, [1, 2, 3, 4, 5])

    def test_array_with_custom_dtype(self, numpy_backend):
        """Test array creation with custom dtype."""
        arr = numpy_backend.array([1, 2, 3], dtype=np.int32)
        assert arr.dtype == np.int32

    def test_array_respects_backend_precision(self, numpy_backend_float32):
        """Test array uses backend precision by default."""
        arr = numpy_backend_float32.array([1, 2, 3])
        assert arr.dtype == np.float32

    def test_zeros_creation(self, numpy_backend):
        """Test zeros array creation."""
        arr = numpy_backend.zeros((3, 4))
        assert arr.shape == (3, 4)
        assert arr.dtype == np.float64
        np.testing.assert_array_equal(arr, np.zeros((3, 4)))

    def test_zeros_with_float32(self, numpy_backend_float32):
        """Test zeros with float32 precision."""
        arr = numpy_backend_float32.zeros((2, 3))
        assert arr.dtype == np.float32

    def test_ones_creation(self, numpy_backend):
        """Test ones array creation."""
        arr = numpy_backend.ones((2, 3))
        assert arr.shape == (2, 3)
        assert arr.dtype == np.float64
        np.testing.assert_array_equal(arr, np.ones((2, 3)))

    def test_linspace_creation(self, numpy_backend):
        """Test linspace array creation."""
        arr = numpy_backend.linspace(0, 1, 11)
        assert len(arr) == 11
        assert arr.dtype == np.float64
        np.testing.assert_allclose(arr, np.linspace(0, 1, 11))

    def test_linspace_float32(self, numpy_backend_float32):
        """Test linspace with float32 precision."""
        arr = numpy_backend_float32.linspace(0, 1, 5)
        assert arr.dtype == np.float32

    def test_meshgrid_xy_indexing(self, numpy_backend):
        """Test 2D meshgrid with xy indexing."""
        x = np.array([1, 2, 3])
        y = np.array([4, 5])
        X, Y = numpy_backend.meshgrid(x, y, indexing="xy")
        assert X.shape == (2, 3)
        assert Y.shape == (2, 3)
        np.testing.assert_array_equal(X[0], [1, 2, 3])
        np.testing.assert_array_equal(Y[:, 0], [4, 5])

    def test_meshgrid_ij_indexing(self, numpy_backend):
        """Test 2D meshgrid with ij indexing."""
        x = np.array([1, 2, 3])
        y = np.array([4, 5])
        X, Y = numpy_backend.meshgrid(x, y, indexing="ij")
        assert X.shape == (3, 2)
        assert Y.shape == (3, 2)


class TestNumPyBackendMathOperations:
    """Test mathematical operations."""

    def test_trapezoid_with_dx(self, numpy_backend):
        """Test trapezoidal integration with dx."""
        y = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        result = numpy_backend.trapezoid(y, dx=0.5)
        expected = np.trapezoid(y, dx=0.5)
        np.testing.assert_allclose(result, expected)

    def test_trapezoid_with_x_array(self, numpy_backend):
        """Test trapezoidal integration with x array."""
        x = np.array([0, 1, 2, 3, 4], dtype=np.float64)
        y = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        result = numpy_backend.trapezoid(y, x=x)
        expected = np.trapezoid(y, x)
        np.testing.assert_allclose(result, expected)

    def test_trapezoid_with_axis(self, numpy_backend):
        """Test trapezoidal integration along specific axis."""
        y = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        result = numpy_backend.trapezoid(y, dx=1.0, axis=1)
        expected = np.trapezoid(y, dx=1.0, axis=1)
        np.testing.assert_allclose(result, expected)

    def test_diff_first_order(self, numpy_backend):
        """Test first-order discrete difference."""
        a = np.array([1, 4, 9, 16, 25])
        result = numpy_backend.diff(a)
        expected = np.array([3, 5, 7, 9])
        np.testing.assert_array_equal(result, expected)

    def test_diff_second_order(self, numpy_backend):
        """Test second-order discrete difference."""
        a = np.array([1, 4, 9, 16, 25])
        result = numpy_backend.diff(a, n=2)
        expected = np.array([2, 2, 2])
        np.testing.assert_array_equal(result, expected)

    def test_interp_linear(self, numpy_backend):
        """Test 1D linear interpolation."""
        xp = np.array([0, 1, 2])
        fp = np.array([0, 10, 20])
        x = np.array([0.5, 1.5])
        result = numpy_backend.interp(x, xp, fp)
        expected = np.array([5, 15])
        np.testing.assert_allclose(result, expected)

    def test_interp_extrapolation(self, numpy_backend):
        """Test interpolation with extrapolation."""
        xp = np.array([0, 1, 2])
        fp = np.array([0, 10, 20])
        x = np.array([-1, 3])  # Outside range
        result = numpy_backend.interp(x, xp, fp)
        # NumPy interp extrapolates with boundary values
        expected = np.array([0, 20])
        np.testing.assert_array_equal(result, expected)


class TestNumPyBackendLinearAlgebra:
    """Test linear algebra operations."""

    def test_solve_2x2_system(self, numpy_backend):
        """Test solving 2x2 linear system."""
        A = np.array([[3, 1], [1, 2]], dtype=np.float64)
        b = np.array([9, 8], dtype=np.float64)
        x = numpy_backend.solve(A, b)
        expected = np.array([2, 3])
        np.testing.assert_allclose(x, expected)

    def test_solve_3x3_system(self, numpy_backend):
        """Test solving 3x3 linear system."""
        A = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]], dtype=np.float64)
        b = np.array([14, 8, 12], dtype=np.float64)
        x = numpy_backend.solve(A, b)
        # Verify Ax = b
        np.testing.assert_allclose(A @ x, b)

    def test_eig_symmetric_matrix(self, numpy_backend):
        """Test eigenvalue decomposition of symmetric matrix."""
        A = np.array([[4, -2], [-2, 1]], dtype=np.float64)
        eigenvalues, eigenvectors = numpy_backend.eig(A)
        # Verify we get 2 eigenvalues and 2x2 eigenvector matrix
        assert len(eigenvalues) == 2
        assert eigenvectors.shape == (2, 2)

    def test_eig_reconstruction(self, numpy_backend):
        """Test that eigendecomposition can reconstruct matrix."""
        A = np.array([[4, -2], [-2, 1]], dtype=np.float64)
        eigenvalues, eigenvectors = numpy_backend.eig(A)
        # Reconstruct: A = V @ diag(λ) @ V^-1
        Lambda = np.diag(eigenvalues)
        A_reconstructed = eigenvectors @ Lambda @ np.linalg.inv(eigenvectors)
        np.testing.assert_allclose(A_reconstructed, A, rtol=1e-10)


class TestNumPyBackendStatistics:
    """Test statistical operations."""

    def test_mean_1d_array(self, numpy_backend):
        """Test mean of 1D array."""
        a = np.array([1, 2, 3, 4, 5])
        result = numpy_backend.mean(a)
        assert result == 3.0

    def test_mean_2d_array_no_axis(self, numpy_backend):
        """Test mean of 2D array without axis."""
        a = np.array([[1, 2, 3], [4, 5, 6]])
        result = numpy_backend.mean(a)
        assert result == 3.5

    def test_mean_2d_array_axis0(self, numpy_backend):
        """Test mean along axis 0."""
        a = np.array([[1, 2, 3], [4, 5, 6]])
        result = numpy_backend.mean(a, axis=0)
        expected = np.array([2.5, 3.5, 4.5])
        np.testing.assert_array_equal(result, expected)

    def test_mean_2d_array_axis1(self, numpy_backend):
        """Test mean along axis 1."""
        a = np.array([[1, 2, 3], [4, 5, 6]])
        result = numpy_backend.mean(a, axis=1)
        expected = np.array([2.0, 5.0])
        np.testing.assert_array_equal(result, expected)

    def test_std_operation(self, numpy_backend):
        """Test standard deviation."""
        a = np.array([1, 2, 3, 4, 5])
        result = numpy_backend.std(a)
        expected = np.std(a)
        np.testing.assert_allclose(result, expected)

    def test_std_with_axis(self, numpy_backend):
        """Test standard deviation along axis."""
        a = np.array([[1, 2, 3], [4, 5, 6]])
        result = numpy_backend.std(a, axis=0)
        expected = np.std(a, axis=0)
        np.testing.assert_allclose(result, expected)

    def test_max_operation(self, numpy_backend):
        """Test maximum value."""
        a = np.array([1, 5, 3, 2, 4])
        result = numpy_backend.max(a)
        assert result == 5

    def test_max_with_axis(self, numpy_backend):
        """Test maximum along axis."""
        a = np.array([[1, 5, 3], [6, 2, 4]])
        result = numpy_backend.max(a, axis=0)
        expected = np.array([6, 5, 4])
        np.testing.assert_array_equal(result, expected)

    def test_min_operation(self, numpy_backend):
        """Test minimum value."""
        a = np.array([1, 5, 3, 2, 4])
        result = numpy_backend.min(a)
        assert result == 1

    def test_min_with_axis(self, numpy_backend):
        """Test minimum along axis."""
        a = np.array([[1, 5, 3], [6, 2, 4]])
        result = numpy_backend.min(a, axis=1)
        expected = np.array([1, 2])
        np.testing.assert_array_equal(result, expected)


class TestNumPyBackendDeviceManagement:
    """Test device management operations."""

    def test_to_device_is_noop(self, numpy_backend):
        """Test to_device returns array unchanged."""
        arr = np.array([1, 2, 3])
        result = numpy_backend.to_device(arr)
        assert result is arr

    def test_from_device_is_noop(self, numpy_backend):
        """Test from_device returns array unchanged."""
        arr = np.array([1, 2, 3])
        result = numpy_backend.from_device(arr)
        assert result is arr

    def test_to_numpy_conversion(self, numpy_backend):
        """Test to_numpy conversion."""
        arr = np.array([1, 2, 3])
        result = numpy_backend.to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_from_numpy_conversion(self, numpy_backend):
        """Test from_numpy conversion."""
        arr = np.array([1, 2, 3])
        result = numpy_backend.from_numpy(arr)
        assert result is arr  # NumPy backend: no conversion needed


class TestNumPyBackendInformation:
    """Test backend information methods."""

    def test_get_device_info(self, numpy_backend):
        """Test device information includes numpy version."""
        info = numpy_backend.get_device_info()
        assert info["backend"] == "numpy"
        assert info["device"] == "cpu"
        assert info["precision"] == "float64"
        assert "numpy_version" in info
        assert isinstance(info["numpy_version"], str)

    def test_memory_usage_without_psutil(self, numpy_backend, monkeypatch):
        """Test memory_usage returns None when psutil unavailable."""
        # Mock ImportError for psutil
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("psutil not available")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        result = numpy_backend.memory_usage()
        assert result is None

    def test_memory_usage_with_psutil(self, numpy_backend):
        """Test memory_usage with psutil available."""
        try:
            import psutil  # noqa: F401

            result = numpy_backend.memory_usage()
            if result is not None:
                assert "rss" in result
                assert "vms" in result
                assert "percent" in result
                assert isinstance(result["rss"], int)
                assert isinstance(result["vms"], int)
                assert isinstance(result["percent"], float)
        except ImportError:
            pytest.skip("psutil not available")


class TestNumPyBackendContextManager:
    """Test context manager usage."""

    def test_context_manager(self):
        """Test backend can be used as context manager."""
        with NumPyBackend(precision="float32") as backend:
            assert backend.name == "numpy"
            assert backend.dtype == np.float32
            arr = backend.zeros((3, 3))
            assert arr.dtype == np.float32

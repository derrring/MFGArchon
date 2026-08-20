"""
Unit tests for BaseBackend abstract interface.

Tests the abstract base class and common backend functionality.
"""

import pytest

import numpy as np

from mfgarchon.backends.base_backend import BaseBackend


class MinimalBackend(BaseBackend):
    """Minimal concrete implementation for testing BaseBackend."""

    def _setup_backend(self):
        """Minimal setup."""
        self.dtype = np.float64 if self.precision == "float64" else np.float32

    @property
    def name(self) -> str:
        return "minimal"

    @property
    def array_module(self):
        return np

    def array(self, data, dtype=None):
        return np.array(data, dtype=dtype or self.dtype)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype or self.dtype)

    def ones(self, shape, dtype=None):
        return np.ones(shape, dtype=dtype or self.dtype)

    def linspace(self, start, stop, num):
        return np.linspace(start, stop, num, dtype=self.dtype)

    def meshgrid(self, *arrays, indexing="xy"):
        return np.meshgrid(*arrays, indexing=indexing)

    def grad(self, func, argnum=0):
        """Simple finite difference gradient."""

        def gradient_func(*args):
            eps = 1e-7
            args_list = list(args)
            x = args_list[argnum]
            args_list[argnum] = x + eps
            f_plus = func(*args_list)
            args_list[argnum] = x - eps
            f_minus = func(*args_list)
            return (f_plus - f_minus) / (2 * eps)

        return gradient_func

    def trapezoid(self, y, x=None, dx=1.0, axis=-1):
        if x is not None:
            return np.trapezoid(y, x, axis=axis)
        return np.trapezoid(y, dx=dx, axis=axis)

    def diff(self, a, n=1, axis=-1):
        return np.diff(a, n=n, axis=axis)

    def interp(self, x, xp, fp):
        return np.interp(x, xp, fp)

    def solve(self, A, b):
        return np.linalg.solve(A, b)

    def eig(self, a):
        return np.linalg.eig(a)

    def mean(self, a, axis=None):
        return np.mean(a, axis=axis)

    def std(self, a, axis=None):
        return np.std(a, axis=axis)

    def max(self, a, axis=None):
        return np.max(a, axis=axis)

    def min(self, a, axis=None):
        return np.min(a, axis=axis)

    def compute_hamiltonian(self, x, p, m, problem_params):
        return 0.5 * p**2

    def compute_optimal_control(self, x, p, m, problem_params):
        return -p

    def hjb_step(self, U, M, dt, dx, problem_params):
        return U  # Dummy implementation

    def fpk_step(self, M, U, dt, dx, problem_params):
        return M  # Dummy implementation


@pytest.fixture
def minimal_backend():
    """Create minimal backend instance."""
    return MinimalBackend(device="cpu", precision="float64")


class TestBaseBackendInitialization:
    """Test backend initialization and configuration."""

    def test_default_initialization(self):
        """Test backend with default parameters."""
        backend = MinimalBackend()
        assert backend.device == "auto"
        assert backend.precision == "float64"
        assert backend.config == {}

    def test_custom_device_and_precision(self):
        """Test backend with custom device and precision."""
        backend = MinimalBackend(device="cpu", precision="float32")
        assert backend.device == "cpu"
        assert backend.precision == "float32"

    def test_kwargs_stored_in_config(self):
        """Test that additional kwargs are stored in config."""
        backend = MinimalBackend(device="cpu", custom_param=42, debug=True)
        assert backend.config["custom_param"] == 42
        assert backend.config["debug"] is True

    def test_setup_backend_called(self):
        """Test that _setup_backend is called during initialization."""
        backend = MinimalBackend(precision="float32")
        # Verify dtype was set in _setup_backend
        assert backend.dtype == np.float32


class TestBaseBackendProperties:
    """Test backend property methods."""

    def test_name_property(self, minimal_backend):
        """Test backend name property."""
        assert minimal_backend.name == "minimal"

    def test_array_module_property(self, minimal_backend):
        """Test array module property returns numpy."""
        assert minimal_backend.array_module is np


class TestBaseBackendArrayOperations:
    """Test array creation and manipulation operations."""

    def test_array_creation(self, minimal_backend):
        """Test creating arrays with backend."""
        arr = minimal_backend.array([1, 2, 3, 4, 5])
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        np.testing.assert_array_equal(arr, [1, 2, 3, 4, 5])

    def test_array_custom_dtype(self, minimal_backend):
        """Test array creation with custom dtype."""
        arr = minimal_backend.array([1, 2, 3], dtype=np.int32)
        assert arr.dtype == np.int32

    def test_zeros_creation(self, minimal_backend):
        """Test zeros array creation."""
        arr = minimal_backend.zeros((3, 4))
        assert arr.shape == (3, 4)
        assert arr.dtype == np.float64
        np.testing.assert_array_equal(arr, np.zeros((3, 4)))

    def test_ones_creation(self, minimal_backend):
        """Test ones array creation."""
        arr = minimal_backend.ones((2, 3))
        assert arr.shape == (2, 3)
        np.testing.assert_array_equal(arr, np.ones((2, 3)))

    def test_linspace_creation(self, minimal_backend):
        """Test linspace array creation."""
        arr = minimal_backend.linspace(0, 1, 11)
        assert len(arr) == 11
        np.testing.assert_allclose(arr, np.linspace(0, 1, 11))

    def test_meshgrid_2d(self, minimal_backend):
        """Test 2D meshgrid creation."""
        x = np.array([1, 2, 3])
        y = np.array([4, 5])
        X, Y = minimal_backend.meshgrid(x, y, indexing="xy")
        assert X.shape == (2, 3)
        assert Y.shape == (2, 3)


class TestBaseBackendMathOperations:
    """Test mathematical operations."""

    def test_trapezoid_integration_with_dx(self, minimal_backend):
        """Test trapezoidal integration with dx."""
        y = np.array([1, 2, 3, 4, 5])
        result = minimal_backend.trapezoid(y, dx=0.5)
        expected = np.trapezoid(y, dx=0.5)
        np.testing.assert_allclose(result, expected)

    def test_trapezoid_integration_with_x(self, minimal_backend):
        """Test trapezoidal integration with x array."""
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([1, 2, 3, 4, 5])
        result = minimal_backend.trapezoid(y, x=x)
        expected = np.trapezoid(y, x)
        np.testing.assert_allclose(result, expected)

    def test_diff_operation(self, minimal_backend):
        """Test discrete difference operation."""
        a = np.array([1, 4, 9, 16, 25])
        result = minimal_backend.diff(a)
        expected = np.array([3, 5, 7, 9])
        np.testing.assert_array_equal(result, expected)

    def test_interp_operation(self, minimal_backend):
        """Test 1D linear interpolation."""
        xp = np.array([0, 1, 2])
        fp = np.array([0, 10, 20])
        x = np.array([0.5, 1.5])
        result = minimal_backend.interp(x, xp, fp)
        expected = np.array([5, 15])
        np.testing.assert_allclose(result, expected)


class TestBaseBackendLinearAlgebra:
    """Test linear algebra operations."""

    def test_solve_linear_system(self, minimal_backend):
        """Test solving Ax = b."""
        A = np.array([[3, 1], [1, 2]], dtype=np.float64)
        b = np.array([9, 8], dtype=np.float64)
        x = minimal_backend.solve(A, b)
        expected = np.array([2, 3])
        np.testing.assert_allclose(x, expected)

    def test_eigenvalue_decomposition(self, minimal_backend):
        """Test eigenvalue/eigenvector computation."""
        A = np.array([[4, -2], [1, 1]], dtype=np.float64)
        eigenvalues, eigenvectors = minimal_backend.eig(A)
        # Verify eigenvalues and eigenvectors
        assert len(eigenvalues) == 2
        assert eigenvectors.shape == (2, 2)


class TestBaseBackendStatistics:
    """Test statistical operations."""

    def test_mean_operation(self, minimal_backend):
        """Test mean calculation."""
        a = np.array([1, 2, 3, 4, 5])
        result = minimal_backend.mean(a)
        assert result == 3.0

    def test_mean_with_axis(self, minimal_backend):
        """Test mean along specific axis."""
        a = np.array([[1, 2, 3], [4, 5, 6]])
        result = minimal_backend.mean(a, axis=0)
        np.testing.assert_array_equal(result, [2.5, 3.5, 4.5])

    def test_std_operation(self, minimal_backend):
        """Test standard deviation calculation."""
        a = np.array([1, 2, 3, 4, 5])
        result = minimal_backend.std(a)
        expected = np.std(a)
        np.testing.assert_allclose(result, expected)

    def test_max_operation(self, minimal_backend):
        """Test maximum value."""
        a = np.array([1, 5, 3, 2, 4])
        result = minimal_backend.max(a)
        assert result == 5

    def test_min_operation(self, minimal_backend):
        """Test minimum value."""
        a = np.array([1, 5, 3, 2, 4])
        result = minimal_backend.min(a)
        assert result == 1


class TestBaseBackendMFGOperations:
    """Test MFG-specific operations."""

    def test_compute_hamiltonian_default(self, minimal_backend):
        """Test default Hamiltonian computation."""
        x = np.array([0.5])
        p = np.array([2.0])
        m = np.array([0.1])
        H = minimal_backend.compute_hamiltonian(x, p, m, {})
        # Default: H = 0.5 * p^2
        expected = 0.5 * 2.0**2
        assert expected == H

    def test_compute_optimal_control_default(self, minimal_backend):
        """Test default optimal control computation."""
        x = np.array([0.5])
        p = np.array([2.0])
        m = np.array([0.1])
        a = minimal_backend.compute_optimal_control(x, p, m, {})
        # Default: a = -p
        expected = -2.0
        assert a == expected


class TestBaseBackendDeviceManagement:
    """Test device management operations."""

    def test_to_device_default(self, minimal_backend):
        """Test default to_device is no-op."""
        arr = np.array([1, 2, 3])
        result = minimal_backend.to_device(arr)
        assert result is arr  # Should return same array

    def test_from_device_default(self, minimal_backend):
        """Test default from_device converts to numpy."""
        arr = np.array([1, 2, 3])
        result = minimal_backend.from_device(arr)
        np.testing.assert_array_equal(result, arr)

    def test_to_numpy_conversion(self, minimal_backend):
        """Test conversion to numpy array."""
        arr = np.array([1, 2, 3])
        result = minimal_backend.to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_from_numpy_conversion(self, minimal_backend):
        """Test conversion from numpy array."""
        arr = np.array([1, 2, 3])
        result = minimal_backend.from_numpy(arr)
        np.testing.assert_array_equal(result, arr)


class TestBaseBackendInformation:
    """Test backend information methods."""

    def test_get_device_info(self, minimal_backend):
        """Test device information retrieval."""
        info = minimal_backend.get_device_info()
        assert info["backend"] == "minimal"
        assert info["device"] == "cpu"
        assert info["precision"] == "float64"

    def test_memory_usage_default(self, minimal_backend):
        """Test default memory_usage returns None."""
        result = minimal_backend.memory_usage()
        assert result is None


class TestBaseBackendCapabilities:
    """Test capability checking system."""

    def test_has_capability_default_false(self, minimal_backend):
        """Test default has_capability returns False."""
        assert minimal_backend.has_capability("parallel_kde") is False
        assert minimal_backend.has_capability("jit_compilation") is False
        assert minimal_backend.has_capability("low_latency") is False

    def test_get_performance_hints_default(self, minimal_backend):
        """Test default performance hints."""
        hints = minimal_backend.get_performance_hints()
        assert hints["kernel_overhead_us"] == 0
        assert hints["memory_bandwidth_gb"] == 50
        assert hints["device_type"] == "cpu"
        assert hints["optimal_problem_size"] == (5000, 50, 20)


class TestBaseBackendContextManager:
    """Test context manager protocol."""

    def test_context_manager_enter(self, minimal_backend):
        """Test __enter__ returns self."""
        result = minimal_backend.__enter__()
        assert result is minimal_backend

    def test_context_manager_exit(self, minimal_backend):
        """Test __exit__ propagates exceptions."""
        result = minimal_backend.__exit__(None, None, None)
        assert result is None

    def test_context_manager_usage(self):
        """Test backend can be used as context manager."""
        with MinimalBackend() as backend:
            assert backend.name == "minimal"
            arr = backend.zeros((2, 2))
            assert arr.shape == (2, 2)

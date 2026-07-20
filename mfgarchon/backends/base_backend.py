"""
Base Backend Interface for MFGarchon

Defines the abstract interface that all computational backends must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseBackend(ABC):
    """
    Abstract base class for computational backends.

    All backends must implement these methods to provide consistent
    interfaces for MFG computations across different numerical libraries.
    """

    def __init__(self, device: str = "auto", precision: str = "float64", **kwargs):
        """
        Initialize backend with configuration.

        Args:
            device: Device to use ("cpu", "gpu", or "auto")
            precision: Numerical precision ("float32" or "float64")
            **kwargs: Backend-specific options
        """
        self.device = device
        self.precision = precision
        self.config = kwargs
        self._setup_backend()

    @abstractmethod
    def _setup_backend(self):
        """Backend-specific initialization."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name identifier."""

    @property
    @abstractmethod
    def array_module(self):
        """The array module (numpy, jax.numpy, etc.)."""

    # Array Operations
    @abstractmethod
    def array(self, data, dtype=None):
        """Create backend-specific array."""

    @abstractmethod
    def zeros(self, shape, dtype=None):
        """Create array of zeros."""

    @abstractmethod
    def ones(self, shape, dtype=None):
        """Create array of ones."""

    @abstractmethod
    def linspace(self, start, stop, num):
        """Create linearly spaced array."""

    @abstractmethod
    def meshgrid(self, *arrays, indexing="xy"):
        """Create coordinate arrays from arrays."""

    # Mathematical Operations
    @abstractmethod
    def grad(self, func, argnum=0):
        """Compute gradient of function."""

    @abstractmethod
    def trapezoid(self, y, x=None, dx=1.0, axis=-1):
        """Trapezoidal integration."""

    @abstractmethod
    def diff(self, a, n=1, axis=-1):
        """Discrete difference."""

    @abstractmethod
    def interp(self, x, xp, fp):
        """1-D linear interpolation."""

    # Linear Algebra
    @abstractmethod
    def solve(self, A, b):
        """Solve linear system Ax = b."""

    @abstractmethod
    def eig(self, a):
        """Compute eigenvalues and eigenvectors."""

    # Statistics
    @abstractmethod
    def mean(self, a, axis=None):
        """Compute mean along axis."""

    @abstractmethod
    def std(self, a, axis=None):
        """Compute the POPULATION standard deviation (denominator ``N``, i.e. ``ddof=0``).

        The convention is part of the contract, not left to the backing library. NumPy and
        JAX default to ``ddof=0``; PyTorch defaults to Bessel-corrected ``unbiased=True``
        (``N-1``). Leaving it unstated let four implementers each inherit their own
        library's default, and TorchBackend silently returned a different quantity --
        measured 0.314315110 against 0.312755227 for the same input, a ratio of exactly
        ``sqrt(N/(N-1))``.

        Population is chosen because two of the three already did it, and because these
        arrays are discretised fields rather than samples from a population -- there is no
        sampling correction to make.
        """

    @abstractmethod
    def max(self, a, axis=None):
        """Maximum values along axis."""

    @abstractmethod
    def min(self, a, axis=None):
        """Minimum values along axis."""

    # MFG-Specific Operations
    @abstractmethod
    def compute_hamiltonian(self, x, p, m, problem_params):
        """Compute Hamiltonian H(x, p, m)."""

    @abstractmethod
    def compute_optimal_control(self, x, p, m, problem_params):
        """Compute optimal control a*(x, p, m)."""

    @abstractmethod
    def hjb_step(self, U, M, dt, dx, problem_params):
        """Single explicit HJB time step — LQ-only toy stepper, NOT the production solve path.

        ⚠️ These backend ``*_step`` methods hardcode a fixed (Linear-Quadratic-flavoured)
        Hamiltonian and do NOT honor ``problem.hamiltonian_class`` — and each backend hardcodes a
        *different* default (numpy ``0.5 p²``; torch ``0.5|p|² + V + interaction·log m``; numba
        ``0.5 p² + log m``; jax ``0.5 p²``). They are an experimental/benchmark surface with **no
        caller in the HJB/FP solver fleet** (only ``tests/`` and one benchmark demo reach them).

        The production solvers (``HJBFDMSolver``/``HJBGFDMSolver``/…) do NOT use this; they evaluate
        the problem's actual Hamiltonian through the #1071 single source
        (``base_hjb``/``h_eval.evaluate_H`` ← ``problem.hamiltonian_class``). So a custom-Hamiltonian
        solve via ``problem.solve()`` is correct regardless of this hardcode.

        Teaching these steppers ``hamiltonian_class`` (and XLA-lowering the operator tree) is the
        deferred RFC #1072 ("Functional Operator Lowering", post-v1.0). Do NOT treat this as a quick
        patch — see #1072 for the World-A-vs-World-B fork and the tracing/operator walls.
        """

    @abstractmethod
    def fpk_step(self, M, U, dt, dx, problem_params):
        """Single explicit FPK time step — LQ-only toy stepper, NOT the production solve path.

        See :meth:`hjb_step`: this is the experimental backend stepper surface (hardcoded default
        drift, not ``problem.hamiltonian_class``), with no caller in the solver fleet. The live FP
        solvers single-source the drift from the problem (#1071/#1043), not from here. Honoring
        ``hamiltonian_class`` here is the deferred RFC #1072.
        """

    # Performance and Compilation
    def compile_function(self, func, *args, **kwargs):
        """
        Compile function for performance (JIT compilation for JAX).
        Default implementation returns the function unchanged.
        """
        return func

    def vectorize(self, func, signature=None):
        """
        Vectorize function for element-wise operations.
        Default implementation uses numpy's vectorize.
        """
        return np.vectorize(func, signature=signature)

    # Device Management
    def to_device(self, array):
        """Move array to backend's target device."""
        return array  # Default: no-op

    def from_device(self, array):
        """Move array from backend's device to CPU/numpy."""
        return np.asarray(array)

    # Type Conversion
    def to_numpy(self, array) -> np.ndarray:
        """Convert backend array to numpy array."""
        return np.asarray(array)

    def from_numpy(self, array: np.ndarray):
        """Convert numpy array to backend array."""
        return self.array(array)

    # Backend Information
    def get_device_info(self) -> dict:
        """Get information about current device."""
        return {
            "backend": self.name,
            "device": self.device,
            "precision": self.precision,
        }

    def memory_usage(self) -> dict | None:
        """Get memory usage information if available."""
        return None  # Override in specific backends

    # Backend Capabilities (for auto-switching)
    def has_capability(self, capability: str) -> bool:
        """
        Check if backend supports a specific capability.

        This method enables intelligent strategy selection by querying
        backend capabilities rather than checking backend type.

        Parameters
        ----------
        capability : str
            Capability to check. Standard capabilities:
            - "parallel_kde": Efficient GPU kernel density estimation
            - "parallel_interpolation": Fast parallel interpolation
            - "low_latency": Low kernel launch overhead (<10μs)
            - "high_bandwidth": High memory bandwidth (>100 GB/s)
            - "unified_memory": CPU/GPU share memory
            - "jit_compilation": JIT compilation support (JAX, Numba)

        Returns
        -------
        bool
            True if capability is supported, False otherwise

        Examples
        --------
        >>> backend = TorchBackend(device="mps")
        >>> backend.has_capability("parallel_kde")
        True
        >>> backend.has_capability("low_latency")  # MPS has higher latency
        False
        """
        # Default implementation: no special capabilities
        return False

    def get_performance_hints(self) -> dict:
        """
        Return performance characteristics for intelligent strategy selection.

        This method provides runtime performance data that helps select
        optimal computational strategies based on hardware characteristics.

        Returns
        -------
        dict
            Performance hints with keys:
            - "kernel_overhead_us": Kernel launch overhead (microseconds)
            - "memory_bandwidth_gb": Memory bandwidth (GB/s)
            - "device_type": Device type ("cpu", "cuda", "mps", "tpu")
            - "optimal_problem_size": Recommended (N, Nx, Nt) for best performance

        Examples
        --------
        >>> backend = TorchBackend(device="mps")
        >>> hints = backend.get_performance_hints()
        >>> hints["kernel_overhead_us"]
        50
        >>> hints["device_type"]
        'mps'
        """
        # Default implementation: CPU-like performance
        return {
            "kernel_overhead_us": 0,  # No GPU kernel overhead
            "memory_bandwidth_gb": 50,  # Typical DDR4 bandwidth
            "device_type": "cpu",
            "optimal_problem_size": (5000, 50, 20),  # Small problems
        }

    # Context Management
    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        return None  # Propagate exceptions

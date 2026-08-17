"""Array dispatch. Arrays in, arrays out.

**What a backend is.** It decides what array type flows through a computation -- nothing else. It
never sees a `BoundaryConditions`, never resolves a `BCType`, and holds no solver logic. A caller
that needs a boundary condition applied is one layer too low.

**What a backend is not.** Until 2026-08-17 this class also declared `hjb_step` and `fpk_step`,
implemented in all five backends and called by **no solver in the package** -- a second, parallel
solver family, alive only through its own tests and one example. They were what forced `dx: float`
into the signature (3 of 29 methods), and `dx` is why the module read as unable to accept geometry
"even in principle" (#1920). Removed; the question dissolves with them.

**Compiled kernels are a different axis.** Replacing a hot function with a compiled one -- numba
today, Rust later -- does not change the array type and needs no dispatch layer or user choice: the
function is simply faster. That belongs beside the function it replaces, not here. `NumbaBackend`
is the counterexample: registered as a backend, it returns `np.array`/`np.zeros` like the numpy one
and, after the step kernels went, contains zero `@njit`. It is a numpy backend under another name,
and it is where a Rust "backend" would land if the distinction were not written down.

**Choosing a backend is not the user's problem.** scikit-learn and SciPy infer the namespace from
the input array (Array API standard) rather than taking a name. This library still takes a name,
which is a smaller interface than it looks: measured on this machine, torch is 9.2-361x slower than
numpy and silently narrows float64 to float32 while the library asserts to 1e-10; jax's CPU sparse
solve calls back into scipy; and there is no GPU target anywhere in `pyproject.toml` or `docs/`.
Solvers pass `backend or "numpy"` for that reason.
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

        Population is chosen because three of the four already did it, and because these
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

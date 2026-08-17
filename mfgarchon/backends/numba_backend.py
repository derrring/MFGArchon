"""
Numba Backend for MFGarchon

CPU-optimized backend using Numba JIT compilation for imperative algorithms,
tight loops, and conditional operations. Ideal for AMR, mesh operations,
and CPU-bound computations.
"""

from __future__ import annotations

import warnings

import numpy as np

from mfgarchon.utils.numerical.integration import trapezoid
from mfgarchon.utils.pde_coefficients import resolve_volatility

from .base_backend import BaseBackend

# Numba is optional, and this module must stay importable without it: a tree-wide scan
# (the deprecation census, #1787) imports every module and cannot tell "declines to load
# without its optional dependency" from "broken", so an ImportError here fails the scan
# for the whole package. Raising moved to __init__ -- see the comment there for why the
# failure must still be loud rather than a fallback.
try:
    import numba
    from numba import jit, njit
    from numba import vectorize as numba_vectorize

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    numba = None  # type: ignore[assignment]
    jit = None  # type: ignore[assignment]
    njit = None  # type: ignore[assignment]
    numba_vectorize = None  # type: ignore[assignment]


class NumbaBackend(BaseBackend):
    """
    Numba-accelerated computational backend for CPU optimization.

    Specializes in:
    - Imperative algorithms with loops and conditionals
    - Adaptive mesh refinement operations
    - CPU parallelization of numerical kernels
    - JIT compilation of performance-critical functions
    """

    def __init__(self, device: str = "auto", precision: str = "float64", **kwargs):
        # Loud at construction, never a fallback. #1638 removed a NumPy fallback from
        # `fpk_step` that dropped advection outright (`flux_div = 0.0`), which would have
        # silently solved pure diffusion instead of the FP equation; the surviving
        # `if NUMBA_AVAILABLE` branches in this file are unreachable for the same reason
        # and carry the same risk if that ever stops being true. Matches JAXBackend.
        if not NUMBA_AVAILABLE:
            raise ImportError(
                "Numba backend requested but Numba is not installed. "
                "Install with: pip install 'mfgarchon[all]' or 'pip install numba'"
            )
        super().__init__(device=device, precision=precision, **kwargs)

    def _setup_backend(self):
        """Initialize Numba backend."""
        if not NUMBA_AVAILABLE:
            warnings.warn(
                "Numba not available. Install with: pip install numba\n"
                "NumbaBackend will fall back to NumPy operations.",
                ImportWarning,
            )

        # Set dtype
        if self.precision == "float32":
            self.dtype = np.float32
            self.numba_dtype = numba.float32 if NUMBA_AVAILABLE else np.float32
        else:
            self.dtype = np.float64
            self.numba_dtype = numba.float64 if NUMBA_AVAILABLE else np.float64

        # Numba primarily targets CPU
        if self.device not in ["cpu", "auto"]:
            warnings.warn(f"NumbaBackend optimized for CPU, got device='{self.device}'")
        self.device = "cpu"

        # Configure Numba settings if available
        if NUMBA_AVAILABLE:
            self._configure_numba()

        # Compile core functions
        self._compile_core_functions()

    def _configure_numba(self):
        """Configure Numba for optimal performance."""
        # Set threading layer for parallel operations
        numba.config.THREADING_LAYER = "workqueue"  # or 'tbb' if available

        # Enable parallel operations
        self.parallel = self.config.get("parallel", True)
        self.cache = self.config.get("cache", True)
        self.fastmath = self.config.get("fastmath", True)

    def _compile_core_functions(self):
        """Pre-compile core numerical functions for performance."""
        if not NUMBA_AVAILABLE:
            return

        # Core JIT compilation options
        jit_options = {
            "nopython": True,
            "cache": self.cache,
            "fastmath": self.fastmath,
        }

        parallel_jit_options = {
            **jit_options,
            "parallel": self.parallel,
        }

        # Compile finite difference kernels
        self._finite_diff_1d = self._create_finite_diff_1d(parallel_jit_options)
        self._finite_diff_2d = self._create_finite_diff_2d(parallel_jit_options)

        # Compile integration kernels
        self._trapezoid_kernel = self._create_trapezoid_kernel(jit_options)

        # Compile MFG-specific kernels
        self._hamiltonian_kernel = self._create_hamiltonian_kernel(jit_options)

    def _create_finite_diff_1d(self, jit_options):
        """Create JIT-compiled first derivative kernel."""

        @jit(**jit_options)
        def finite_diff_1d(u, dx):
            n = len(u)
            result = np.zeros_like(u)

            # Forward difference at boundary
            result[0] = (u[1] - u[0]) / dx

            # Central difference in interior
            for i in range(1, n - 1):
                result[i] = (u[i + 1] - u[i - 1]) / (2.0 * dx)

            # Backward difference at boundary
            result[n - 1] = (u[n - 1] - u[n - 2]) / dx

            return result

        return finite_diff_1d

    def _create_finite_diff_2d(self, jit_options):
        """Create JIT-compiled second derivative kernel."""

        @jit(**jit_options)
        def finite_diff_2d(u, dx):
            n = len(u)
            result = np.zeros_like(u)

            # Second derivative using central differences
            result[0] = 0.0  # Boundary handling
            for i in range(1, n - 1):
                result[i] = (u[i + 1] - 2.0 * u[i] + u[i - 1]) / (dx * dx)
            result[n - 1] = 0.0  # Boundary handling

            return result

        return finite_diff_2d

    def _create_trapezoid_kernel(self, jit_options):
        """Create JIT-compiled trapezoidal integration kernel."""

        @jit(**jit_options)
        def trapezoid_kernel(y, dx):
            n = len(y)
            if n < 2:
                return 0.0

            result = 0.5 * (y[0] + y[n - 1])
            for i in range(1, n - 1):
                result += y[i]

            return result * dx

        return trapezoid_kernel

    def _create_hamiltonian_kernel(self, jit_options):
        """Create JIT-compiled Hamiltonian computation."""

        @jit(**jit_options)
        def hamiltonian_kernel(x, p, m, sigma):
            # H(x, p, m) = 0.5 * p^2 + potential(x, m)
            kinetic = 0.5 * p * p
            potential = np.log(m + 1e-8)  # Interaction term
            return kinetic + potential

        return hamiltonian_kernel

    @property
    def name(self) -> str:
        return f"numba{'_cpu' if NUMBA_AVAILABLE else '_fallback'}"

    @property
    def array_module(self):
        return np

    # Array Operations
    def array(self, data, dtype=None):
        if dtype is None:
            dtype = self.dtype
        return np.array(data, dtype=dtype)

    def zeros(self, shape, dtype=None):
        if dtype is None:
            dtype = self.dtype
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        if dtype is None:
            dtype = self.dtype
        return np.ones(shape, dtype=dtype)

    def linspace(self, start, stop, num):
        return np.linspace(start, stop, num, dtype=self.dtype)

    def meshgrid(self, *arrays, indexing="xy"):
        return np.meshgrid(*arrays, indexing=indexing)

    # Mathematical Operations
    def grad(self, func, argnum=0):
        """
        Numba doesn't have automatic differentiation.
        Use finite differences for gradient computation.
        """

        def grad_func(x, *args, **kwargs):
            eps = 1e-8 if self.dtype == np.float64 else 1e-4
            x_plus = np.copy(x)
            x_plus[argnum] += eps
            x_minus = np.copy(x)
            x_minus[argnum] -= eps

            return (func(x_plus, *args, **kwargs) - func(x_minus, *args, **kwargs)) / (2.0 * eps)

        return grad_func

    def trapezoid(self, y, x=None, dx=1.0, axis=-1):
        if NUMBA_AVAILABLE and x is None and axis == -1:
            # Use optimized kernel for simple case
            return self._trapezoid_kernel(y, dx)
        else:
            # Fall back to numpy/utils implementation
            return trapezoid(y, x=x, dx=dx, axis=axis)

    def diff(self, a, n=1, axis=-1):
        # Issue #1283: _finite_diff_1d is a *derivative* kernel (size-n), not a
        # discrete difference (size-(n-1)).  Using it here violated the base-class
        # contract (matching np.diff) and all other backends.  Delegate to np.diff
        # unconditionally; _finite_diff_1d remains available for gradient use only.
        return np.diff(a, n=n, axis=axis)

    def interp(self, x, xp, fp):
        return np.interp(x, xp, fp)

    # Linear Algebra
    def solve(self, A, b):
        return np.linalg.solve(A, b)

    def eig(self, a):
        return np.linalg.eig(a)

    # Statistics
    def mean(self, a, axis=None):
        return np.mean(a, axis=axis)

    def std(self, a, axis=None):
        return np.std(a, axis=axis)

    def max(self, a, axis=None):
        return np.max(a, axis=axis)

    def min(self, a, axis=None):
        return np.min(a, axis=axis)

    # MFG-Specific Operations
    def compute_hamiltonian(self, x, p, m, problem_params):
        """Compute Hamiltonian using optimized kernel."""
        # Issue #1282: numba already uses the canonical "sigma" key; route through the
        # single-source resolver (no legacy key) so all four backends share one lookup.
        sigma = resolve_volatility(problem_params, default=1.0)

        if NUMBA_AVAILABLE:
            # Use JIT-compiled kernel for performance
            return self._hamiltonian_kernel(x, p, m, sigma)
        else:
            # Fallback implementation
            kinetic = 0.5 * p * p
            potential = np.log(m + 1e-8)
            return kinetic + potential

    def compute_optimal_control(self, x, p, m, problem_params):
        """Compute optimal control a*(x, p, m)."""
        # For quadratic cost: a* = -p
        return -p

    def compile_function(self, func, *args, **kwargs):
        """Compile function using Numba JIT."""
        if NUMBA_AVAILABLE:
            jit_options = {
                "nopython": True,
                "cache": False,  # Disable cache for dynamically created functions
                "fastmath": self.fastmath,
            }

            # Only enable cache for functions with proper source files
            if hasattr(func, "__code__") and func.__code__.co_filename != "<string>":
                jit_options["cache"] = self.cache

            jit_options.update(kwargs)
            return jit(func, **jit_options)
        else:
            return func

    def vectorize(self, func, signature=None):
        """Vectorize function using Numba or NumPy."""
        if NUMBA_AVAILABLE:
            return numba_vectorize(func)
        else:
            return np.vectorize(func, signature=signature)

    # Numba-Specific Methods
    def parallel_range(self, n):
        """Create parallel range for Numba prange."""
        if NUMBA_AVAILABLE:
            return numba.prange(n)
        else:
            return range(n)

    def optimize_loop(self, loop_func, *args, **kwargs):
        """
        Optimize a loop-heavy function with Numba.

        Args:
            loop_func: Function containing loops to optimize
            *args, **kwargs: Arguments to pass to the function

        Returns:
            Optimized function result
        """
        if NUMBA_AVAILABLE:
            optimized_func = self.compile_function(loop_func)
            return optimized_func(*args, **kwargs)
        else:
            return loop_func(*args, **kwargs)

    def get_device_info(self) -> dict:
        """Get Numba backend information."""
        info = super().get_device_info()
        info.update(
            {
                "numba_available": NUMBA_AVAILABLE,
                "parallel_enabled": getattr(self, "parallel", False),
                "cache_enabled": getattr(self, "cache", False),
                "fastmath_enabled": getattr(self, "fastmath", False),
            }
        )

        if NUMBA_AVAILABLE:
            info.update(
                {
                    "numba_version": numba.__version__,
                    "threading_layer": numba.config.THREADING_LAYER,
                    "cpu_count": numba.config.NUMBA_NUM_THREADS,
                }
            )

        return info

    def memory_usage(self) -> dict | None:
        """Get memory usage information."""
        try:
            import psutil

            process = psutil.Process()
            return {
                "rss_mb": process.memory_info().rss / 1024 / 1024,
                "vms_mb": process.memory_info().vms / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
            }
        except ImportError:
            return None

    # AMR-Specific Methods (Numba speciality)
    def refine_mesh_adaptive(self, mesh_data, error_indicators, refinement_threshold=0.1):
        """
        Adaptive mesh refinement using optimized Numba kernels.

        This is where Numba excels - imperative mesh operations with
        complex conditionals and tree traversal.
        """
        if not NUMBA_AVAILABLE:
            warnings.warn("AMR operations not optimized without Numba")

        # This would contain actual AMR implementation
        # Placeholder for compatibility
        refined_indices = []
        for i, error in enumerate(error_indicators):
            if error > refinement_threshold:
                refined_indices.append(i)

        return np.array(refined_indices)

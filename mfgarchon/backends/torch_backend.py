"""
PyTorch Backend for MFGarchon

This module provides a unified PyTorch backend with comprehensive support for:
- CUDA (NVIDIA GPU acceleration)
- MPS (Apple Silicon Metal Performance Shaders)
- CPU (fallback and development)

The backend integrates seamlessly with the MFGarchon ecosystem while providing
first-class hardware acceleration across all major platforms.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.pde_coefficients import diffusion_from_volatility_torch, resolve_volatility

from .base_backend import BaseBackend

# Module logger
logger = get_logger(__name__)

# PyTorch imports with graceful fallback
try:
    import torch
    from torch import device as torch_device

    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    MPS_AVAILABLE = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    # Device capabilities
    CUDA_DEVICE_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0

    # MPS system info
    if MPS_AVAILABLE:
        try:
            # Test MPS functionality
            test_tensor = torch.tensor([1.0], device="mps")
            MPS_FUNCTIONAL = True
            del test_tensor
        except (RuntimeError, TypeError) as e:
            # Issue #547: MPS device creation can fail even when detected
            MPS_FUNCTIONAL = False
            warnings.warn(
                f"MPS device detected but not functional ({type(e).__name__}): {e}. Falling back to CPU.", stacklevel=2
            )
    else:
        MPS_FUNCTIONAL = False

except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    MPS_AVAILABLE = False
    MPS_FUNCTIONAL = False
    CUDA_DEVICE_COUNT = 0

    # Mock torch for type hints
    class MockTorch:
        @staticmethod
        def tensor(*args, **kwargs):
            return np.array(*args)

        @staticmethod
        def device(device_name):
            return device_name

        class Tensor:
            pass

    torch = MockTorch()


class TorchBackend(BaseBackend):
    """
    Unified PyTorch backend with CUDA, MPS, and CPU support.

    This backend provides comprehensive hardware acceleration across platforms:
    - NVIDIA GPUs via CUDA
    - Apple Silicon via Metal Performance Shaders (MPS)
    - CPU for development and compatibility

    Features:
    - Automatic device selection with priority: CUDA > MPS > CPU
    - Memory management and optimization
    - Mixed precision support (float32/float64)
    - Seamless fallback for unsupported operations
    - MFG-specific optimized kernels
    """

    def __init__(self, device: str = "auto", precision: str = "float64", **kwargs):
        """
        Initialize PyTorch backend.

        Args:
            device: Device specification ("auto", "cuda", "mps", "cpu", "cuda:0", etc.)
            precision: Numerical precision ("float32" or "float64")
            **kwargs: Additional configuration
                - mixed_precision: Enable automatic mixed precision (default: False)
                - memory_efficient: Enable memory optimizations (default: True)
                - compile_mode: Enable torch.compile if available (default: False)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TorchBackend. Install with: pip install torch")

        # Configuration
        self.mixed_precision = kwargs.get("mixed_precision", False)
        self.memory_efficient = kwargs.get("memory_efficient", True)
        self.compile_mode = kwargs.get("compile_mode", False)

        # Initialize base class
        super().__init__(device, precision, **kwargs)

    def _setup_backend(self):
        """Set up PyTorch backend with device selection and configuration."""
        # Select optimal device
        self.torch_device = self._select_device(self.device)
        self.device_type = self.torch_device.type

        # Set precision.
        # Issue #1283: torch.set_default_dtype() is a process-global side effect —
        # constructing any TorchBackend would mutate dtype for all subsequent
        # torch.tensor() calls in the process.  Use self.torch_dtype explicitly
        # on all tensor construction instead; never touch the global default.
        if self.precision == "float32":
            self.torch_dtype = torch.float32
        elif self.precision == "float64":
            self.torch_dtype = torch.float64
        else:
            raise ValueError(f"Unsupported precision: {self.precision}")

        # Handle MPS float64 limitation
        if self.device_type == "mps" and self.precision == "float64":
            warnings.warn(
                "MPS does not support float64, using float32 instead. Set precision='float32' to suppress this warning."
            )
            self.torch_dtype = torch.float32
            self.precision = "float32"

        # Memory management
        if self.memory_efficient:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Mixed precision setup
        if self.mixed_precision and self.device_type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self._print_backend_info()

    def _select_device(self, device_spec: str) -> torch_device:
        """
        Select optimal computational device based on specification and availability.

        Args:
            device_spec: Device specification string

        Returns:
            Selected torch device

        Raises:
            ValueError: If specified device is unavailable
        """
        if device_spec == "auto":
            # Automatic selection with priority: CUDA > MPS > CPU
            if CUDA_AVAILABLE:
                device = torch_device("cuda")
            elif MPS_AVAILABLE and MPS_FUNCTIONAL:
                device = torch_device("mps")
            else:
                device = torch_device("cpu")
        else:
            # Manual device specification
            device = torch_device(device_spec)

            # Validate device availability
            if device.type == "cuda" and not CUDA_AVAILABLE:
                raise ValueError("CUDA device requested but CUDA is not available")
            elif device.type == "mps" and not (MPS_AVAILABLE and MPS_FUNCTIONAL):
                raise ValueError("MPS device requested but MPS is not available or functional")

        return device

    def _print_backend_info(self):
        """Print backend configuration and hardware information."""
        print("=== PyTorch Backend Initialized ===")
        print(f"Device: {self.torch_device}")
        print(f"Precision: {self.precision}")

        if self.device_type == "cuda":
            gpu_name = torch.cuda.get_device_name(self.torch_device)
            memory_gb = torch.cuda.get_device_properties(self.torch_device).total_memory / 1024**3
            print(f"GPU: {gpu_name}")
            print(f"Memory: {memory_gb:.1f} GB")
            print(f"Mixed Precision: {self.mixed_precision}")
        elif self.device_type == "mps":
            print("Apple Silicon: Metal Performance Shaders enabled")
            print(f"Memory Efficient: {self.memory_efficient}")
        else:
            print(f"CPU: {torch.get_num_threads()} threads")

        print("=" * 36)

    @property
    def name(self) -> str:
        """Backend name identifier."""
        return f"torch_{self.device_type}"

    @property
    def array_module(self):
        """Return torch as the array module."""
        return torch

    def _to_torch(self, data, dtype=None):
        """Convert data to torch tensor on correct device."""
        if dtype is None:
            dtype = self.torch_dtype

        if isinstance(data, torch.Tensor):
            return data.to(device=self.torch_device, dtype=dtype)
        else:
            return torch.tensor(data, device=self.torch_device, dtype=dtype)

    def _to_numpy(self, tensor):
        """Convert torch tensor back to numpy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    # Array Operations
    def array(self, data, dtype=None):
        """Create backend-specific array."""
        return self._to_torch(data, dtype)

    def zeros(self, shape, dtype=None):
        """Create array of zeros."""
        if dtype is None:
            dtype = self.torch_dtype
        return torch.zeros(shape, dtype=dtype, device=self.torch_device)

    def ones(self, shape, dtype=None):
        """Create array of ones."""
        if dtype is None:
            dtype = self.torch_dtype
        return torch.ones(shape, dtype=dtype, device=self.torch_device)

    def linspace(self, start, stop, num):
        """Create linearly spaced array."""
        return torch.linspace(start, stop, num, dtype=self.torch_dtype, device=self.torch_device)

    def meshgrid(self, *arrays, indexing="xy"):
        """Create coordinate arrays from arrays."""
        torch_arrays = [self._to_torch(arr) for arr in arrays]
        return torch.meshgrid(*torch_arrays, indexing=indexing)

    # Mathematical Operations
    def grad(self, func, argnum=0):
        """Compute gradient of function using torch.autograd."""

        def grad_func(*args):
            # Convert inputs to tensors with gradients
            tensor_args = []
            for i, arg in enumerate(args):
                tensor = self._to_torch(arg)
                tensor.requires_grad_(i == argnum)
                tensor_args.append(tensor)

            # Compute function output
            output = func(*tensor_args)

            # Compute gradient
            if tensor_args[argnum].grad is not None:
                tensor_args[argnum].grad.zero_()

            grad_outputs = torch.ones_like(output)
            grads = torch.autograd.grad(
                output, tensor_args[argnum], grad_outputs=grad_outputs, create_graph=True, retain_graph=True
            )[0]

            return grads

        return grad_func

    def trapezoid(self, y, x=None, dx=1.0, axis=-1):
        """Trapezoidal integration."""
        y_tensor = self._to_torch(y)
        if x is not None:
            x_tensor = self._to_torch(x)
            return torch.trapezoid(y_tensor, x_tensor, dim=axis)
        else:
            return torch.trapezoid(y_tensor, dx=dx, dim=axis)

    def diff(self, a, n=1, axis=-1):
        """Discrete difference."""
        a_tensor = self._to_torch(a)
        return torch.diff(a_tensor, n=n, dim=axis)

    def interp(self, x, xp, fp):
        """1-D linear interpolation using torch's interpolate."""
        x_tensor = self._to_torch(x)
        xp_tensor = self._to_torch(xp)
        fp_tensor = self._to_torch(fp)

        # Use torch's searchsorted and linear interpolation
        indices = torch.searchsorted(xp_tensor, x_tensor, right=False)
        indices = torch.clamp(indices, 1, len(xp_tensor) - 1)

        x_left = xp_tensor[indices - 1]
        x_right = xp_tensor[indices]
        y_left = fp_tensor[indices - 1]
        y_right = fp_tensor[indices]

        # Linear interpolation
        weight = (x_tensor - x_left) / (x_right - x_left)
        return y_left + weight * (y_right - y_left)

    # Linear Algebra
    def solve(self, A, b):
        """Solve linear system Ax = b."""
        A_tensor = self._to_torch(A)
        b_tensor = self._to_torch(b)
        return torch.linalg.solve(A_tensor, b_tensor)

    def eig(self, a):
        """Compute eigenvalues and eigenvectors."""
        a_tensor = self._to_torch(a)
        eigenvals, eigenvecs = torch.linalg.eig(a_tensor)
        return eigenvals, eigenvecs

    # Statistics
    def mean(self, a, axis=None):
        """Compute mean along axis."""
        a_tensor = self._to_torch(a)
        return torch.mean(a_tensor, dim=axis)

    def std(self, a, axis=None):
        """Compute standard deviation."""
        a_tensor = self._to_torch(a)
        return torch.std(a_tensor, dim=axis)

    def max(self, a, axis=None):
        """Maximum values along axis."""
        a_tensor = self._to_torch(a)
        if axis is None:
            return torch.max(a_tensor)
        else:
            return torch.max(a_tensor, dim=axis)[0]

    def min(self, a, axis=None):
        """Minimum values along axis."""
        a_tensor = self._to_torch(a)
        if axis is None:
            return torch.min(a_tensor)
        else:
            return torch.min(a_tensor, dim=axis)[0]

    # MFG-Specific Operations
    def compute_hamiltonian(self, x, p, m, problem_params):
        """
        Compute Hamiltonian H(x, p, m) optimized for PyTorch.

        Default implementation for quadratic Hamiltonian:
        H(x, p, m) = (1/2)|p|² + V(x) + interaction(x, m)
        """
        x_tensor = self._to_torch(x)
        p_tensor = self._to_torch(p)
        m_tensor = self._to_torch(m)

        # Kinetic energy term: (1/2)|p|²
        kinetic_term = 0.5 * torch.sum(p_tensor**2, dim=-1)

        # Potential energy (can be customized based on problem_params)
        potential_term = problem_params.get("potential_strength", 0.0) * torch.sum(x_tensor**2, dim=-1)

        # Interaction term: logarithmic interaction
        epsilon = problem_params.get("interaction_epsilon", 1e-8)
        interaction_term = problem_params.get("interaction_strength", 1.0) * torch.log(m_tensor + epsilon)

        return kinetic_term + potential_term + interaction_term

    def compute_optimal_control(self, x, p, m, problem_params):
        """
        Compute optimal control a*(x, p, m) = -Hp(x, p, m).

        For quadratic Hamiltonian: a* = -p
        """
        p_tensor = self._to_torch(p)
        return -p_tensor

    def hjb_step(self, U, M, dt, dx, problem_params):
        """
        Single Hamilton-Jacobi-Bellman time step using PyTorch operations.

        Implements: dU/dt + H(x, ∇U, M) = 0
        """
        U_tensor = self._to_torch(U)
        M_tensor = self._to_torch(M)

        # Compute spatial gradient using finite differences
        U_grad = torch.gradient(U_tensor, spacing=dx, dim=-1)[0]

        # Compute Hamiltonian
        x_grid = torch.linspace(
            problem_params.get("x_min", -1),
            problem_params.get("x_max", 1),
            U_tensor.shape[-1],
            device=self.torch_device,
            dtype=self.torch_dtype,
        )

        H = self.compute_hamiltonian(x_grid, U_grad, M_tensor, problem_params)

        # Backward Euler step: U^{n+1} = U^n - dt * H
        U_new = U_tensor - dt * H

        return U_new

    def fpk_step(self, M, U, dt, dx, problem_params):
        """
        Single Fokker-Planck-Kolmogorov time step using PyTorch operations.

        Implements: dM/dt - div(M ∇Hp(x, ∇U, M)) - (σ²/2) ΔM = 0
        """
        M_tensor = self._to_torch(M)
        U_tensor = self._to_torch(U)

        # Compute spatial gradient of U
        U_grad = torch.gradient(U_tensor, spacing=dx, dim=-1)[0]

        # Compute optimal control (drift)
        x_grid = torch.linspace(
            problem_params.get("x_min", -1),
            problem_params.get("x_max", 1),
            M_tensor.shape[-1],
            device=self.torch_device,
            dtype=self.torch_dtype,
        )

        drift = self.compute_optimal_control(x_grid, U_grad, M_tensor, problem_params)

        # Compute divergence term: div(M * drift)
        flux = M_tensor * drift
        div_term = torch.gradient(flux, spacing=dx, dim=-1)[0]

        # Compute diffusion term: D * d²M/dx²  where D = σ²/2 (Issue #1189: route through D).
        # Issue #1282: read the volatility through the single-source resolver (canonical
        # "sigma" key; legacy "diffusion" key holds the volatility despite the name, default
        # preserves the prior 0.1 no-key behavior).
        sigma = resolve_volatility(problem_params, legacy_key="diffusion", legacy_is_squared=False, default=0.1)
        M_grad = torch.gradient(M_tensor, spacing=dx, dim=-1)[0]
        M_grad2 = torch.gradient(M_grad, spacing=dx, dim=-1)[0]
        diffusion_term = diffusion_from_volatility_torch(sigma) * M_grad2

        # Forward Euler step: M^{n+1} = M^n + dt * (-div(flux) + diffusion)
        # Issue #1282: sign was inverted (+ instead of -) on div_term.
        M_new = M_tensor + dt * (-div_term + diffusion_term)

        # Ensure non-negativity and mass conservation
        M_new = torch.clamp(M_new, min=1e-12)
        mass = torch.trapezoid(M_new, dx=dx)
        M_new = M_new / mass

        return M_new

    # Performance and Compilation
    def compile_function(self, func, *args, **kwargs):
        """
        Compile function using torch.compile for performance.

        Only available in PyTorch 2.0+ with compatible hardware.
        """
        if self.compile_mode and hasattr(torch, "compile"):
            try:
                return torch.compile(func, **kwargs)
            except Exception as e:
                warnings.warn(f"Failed to compile function: {e}")
                return func
        return func

    def vectorize(self, func, signature=None):
        """
        Vectorize function using torch.vmap when available.

        Falls back to standard torch operations.
        """
        if hasattr(torch, "vmap"):
            try:
                return torch.vmap(func)
            except (AttributeError, TypeError, RuntimeError) as e:
                # Issue #547: vmap can fail for incompatible function signatures
                logger.debug(
                    "torch.vmap failed for function %s: %s. "
                    "Using non-vectorized version (performance may be degraded).",
                    getattr(func, "__name__", repr(func)),
                    e,
                )
                return func
        return func

    # Backend Capabilities (for auto-switching)
    def has_capability(self, capability: str) -> bool:
        """Check if PyTorch backend supports a specific capability."""
        capabilities = {
            "parallel_kde": True,  # PyTorch supports parallel KDE on GPU
            "parallel_interpolation": True,  # GPU-accelerated interpolation
            "low_latency": self.device_type == "cuda",  # CUDA has low kernel overhead
            "high_bandwidth": self.device_type == "cuda",  # CUDA has high bandwidth
            "unified_memory": self.device_type == "mps",  # MPS uses unified memory
            "jit_compilation": hasattr(torch, "compile"),  # PyTorch 2.0+
        }
        return capabilities.get(capability, False)

    def get_performance_hints(self) -> dict:
        """Return performance characteristics for strategy selection."""
        if self.device_type == "mps":
            # Apple Silicon M-series (Metal Performance Shaders)
            return {
                "kernel_overhead_us": 50,  # Higher than CUDA
                "memory_bandwidth_gb": 200,  # M1/M2 Pro unified memory
                "device_type": "mps",
                "optimal_problem_size": (50000, 100, 50),  # Sweet spot for MPS
            }
        elif self.device_type == "cuda":
            # NVIDIA GPUs
            return {
                "kernel_overhead_us": 5,  # Very low latency
                "memory_bandwidth_gb": 900,  # High-end GPU bandwidth
                "device_type": "cuda",
                "optimal_problem_size": (10000, 100, 50),  # Efficient for most sizes
            }
        else:
            # CPU fallback
            return {
                "kernel_overhead_us": 0,  # No GPU overhead
                "memory_bandwidth_gb": 50,  # DDR4/DDR5
                "device_type": "cpu",
                "optimal_problem_size": (5000, 50, 20),  # Small problems
            }

    # Device Management
    def to_device(self, array):
        """Move array to backend's target device."""
        return self._to_torch(array)

    def to_numpy(self, array):
        """Convert backend array to numpy array."""
        return self._to_numpy(array)

    def get_memory_info(self) -> dict[str, Any]:
        """Get device memory information."""
        info: dict[str, Any] = {"device": str(self.torch_device)}

        if self.device_type == "cuda":
            info.update(
                {
                    "allocated": torch.cuda.memory_allocated(self.torch_device),
                    "reserved": torch.cuda.memory_reserved(self.torch_device),
                    "max_allocated": torch.cuda.max_memory_allocated(self.torch_device),
                    "max_reserved": torch.cuda.max_memory_reserved(self.torch_device),
                }
            )
        elif self.device_type == "mps":
            info.update(
                {
                    "allocated": torch.mps.current_allocated_memory(),
                    "driver_allocated": torch.mps.driver_allocated_memory(),
                }
            )

        return info

    def clear_cache(self):
        """Clear device memory cache."""
        if self.device_type == "cuda":
            torch.cuda.empty_cache()
        elif self.device_type == "mps":
            torch.mps.empty_cache()


# Export device capability information
def get_torch_device_info() -> dict[str, Any]:
    """Get comprehensive PyTorch device information."""
    info = {
        "torch_available": TORCH_AVAILABLE,
        "cuda_available": CUDA_AVAILABLE,
        "mps_available": MPS_AVAILABLE,
        "mps_functional": MPS_FUNCTIONAL,
        "cuda_device_count": CUDA_DEVICE_COUNT,
    }

    if TORCH_AVAILABLE:
        info["torch_version"] = torch.__version__

        if CUDA_AVAILABLE:
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["cuda_devices"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "memory_gb": torch.cuda.get_device_properties(i).total_memory / 1024**3,
                    "compute_capability": torch.cuda.get_device_properties(i).major,
                }
                for i in range(CUDA_DEVICE_COUNT)
            ]

    return info


def print_torch_device_info():
    """Print comprehensive PyTorch device information."""
    info = get_torch_device_info()

    print("=== PyTorch Device Information ===")
    print(f"PyTorch Available: {info['torch_available']}")

    if info["torch_available"]:
        print(f"PyTorch Version: {info['torch_version']}")
        print(f"CUDA Available: {info['cuda_available']}")
        print(f"MPS Available: {info['mps_available']}")
        print(f"MPS Functional: {info['mps_functional']}")

        if info["cuda_available"]:
            print(f"CUDA Version: {info.get('cuda_version', 'Unknown')}")
            print(f"cuDNN Version: {info.get('cudnn_version', 'Unknown')}")
            print(f"CUDA Devices: {info['cuda_device_count']}")

            for i, device in enumerate(info.get("cuda_devices", [])):
                print(f"  GPU {i}: {device['name']} ({device['memory_gb']:.1f} GB)")

        if info["mps_available"]:
            status = "Functional" if info["mps_functional"] else "Detected but not functional"
            print(f"Apple Silicon MPS: {status}")
    else:
        print("Install PyTorch: pip install torch")

    print("=" * 34)


# Convenience function for backend creation
def create_torch_backend(**kwargs) -> TorchBackend:
    """Create PyTorch backend with optimal settings."""
    return TorchBackend(**kwargs)

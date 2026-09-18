"""
Backend Compatibility Layer (Tier 2)

Provides helper functions for backend-agnostic operations during migration period.
This is a bridge between current NumPy-centric code and full backend protocol.

Usage:
    from mfgarchon.backends.compat import backend_aware_copy, has_nan_or_inf

    U_copy = backend_aware_copy(U, backend)
    if has_nan_or_inf(U, backend):
        raise ValueError("Invalid values")

See docs/development/BACKEND_SWITCHING_DESIGN.md for comprehensive design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mfgarchon.backends.base_backend import BaseBackend


def backend_aware_copy(array, backend: BaseBackend | None = None):
    """
    Copy array using appropriate backend method.

    Handles differences between .copy() (NumPy) and .clone() (PyTorch).

    Args:
        array: Array to copy
        backend: Backend instance (None for auto-detection)

    Returns:
        Copy of array

    Examples:
        >>> U_copy = backend_aware_copy(U, backend)
        >>> # Works for NumPy, PyTorch, JAX
    """
    if backend is not None and hasattr(backend, "copy"):
        return backend.copy(array)

    # Fallback: detect from array type
    if hasattr(array, "clone"):  # PyTorch
        return array.clone()
    elif hasattr(array, "copy"):  # NumPy
        return array.copy()
    else:  # JAX or other (immutable)
        return np.array(array)


def backend_aware_assign(target, indices, value, backend: BaseBackend | None = None):
    """
    Assign value to target[indices] using backend-appropriate method.

    Handles PyTorch .copy_() requirement for MPS tensors.

    Args:
        target: Target array
        indices: Index specification (int, slice, tuple)
        value: Value to assign
        backend: Backend instance

    Returns:
        Modified target array

    Examples:
        >>> backend_aware_assign(U, (0, slice(None)), boundary_values, backend)
        >>> # Handles NumPy direct assignment and PyTorch .copy_()
    """
    if backend is not None and hasattr(backend, "assign"):
        return backend.assign(target, indices, value)

    # PyTorch MPS tensors need special handling
    if hasattr(target, "device") and str(target.device).startswith("mps"):
        # Convert value to same backend if needed
        if not hasattr(value, "device"):
            import torch

            value = torch.tensor(value, device=target.device, dtype=target.dtype)
        # Use .copy_() for in-place assignment
        if hasattr(target[indices], "copy_"):
            target[indices].copy_(value)
            return target

    # Standard assignment for NumPy and CPU tensors
    try:
        target[indices] = value
    except (TypeError, RuntimeError) as e:
        # Last resort: try .copy_() if available
        if hasattr(target[indices], "copy_"):
            target[indices].copy_(value)
        else:
            raise TypeError(
                f"Cannot assign {type(value)} to {type(target)}. Backend-specific assignment failed."
            ) from e

    return target


def has_nan_or_inf(array, backend: BaseBackend | None = None) -> bool:
    """
    Backend-agnostic check for NaN or Inf values.

    Handles different .any() signatures and .isnan()/.isinf() methods.

    Args:
        array: Array to check
        backend: Backend instance

    Returns:
        True if array contains any NaN or Inf values

    Examples:
        >>> if has_nan_or_inf(U_final, backend):
        ...     raise ValueError("Invalid boundary condition")
    """
    if backend is not None and hasattr(backend, "has_nan_or_inf"):
        return backend.has_nan_or_inf(array)

    # Fallback implementation
    if hasattr(array, "isnan"):  # PyTorch tensors
        # Use tensor methods directly (different signature than numpy)
        return bool((array.isnan() | array.isinf()).any())
    else:  # NumPy or JAX
        return bool(np.any(np.isnan(array)) or np.any(np.isinf(array)))


def backend_aware_any(array, backend: BaseBackend | None = None) -> bool:
    """
    Backend-agnostic any() operation.

    Handles different signatures (NumPy accepts axis/out, PyTorch doesn't).

    Args:
        array: Boolean array
        backend: Backend instance

    Returns:
        True if any element is True

    Examples:
        >>> if backend_aware_any(convergence_flags, backend):
        ...     print("At least one converged")
    """
    if backend is not None and hasattr(backend, "any"):
        return bool(backend.any(array))

    # Fallback
    if hasattr(array, "any"):
        result = array.any()
        # PyTorch returns tensor, need .item()
        if hasattr(result, "item"):
            return bool(result.item())
        return bool(result)
    else:
        return bool(np.any(array))


def backend_aware_zeros_like(array, backend: BaseBackend | None = None):
    """
    Create zeros array matching input array's shape and device.

    Args:
        array: Template array
        backend: Backend instance

    Returns:
        Zeros array on same device as input

    Examples:
        >>> U_next = backend_aware_zeros_like(U, backend)
    """
    if backend is not None and hasattr(backend, "zeros_like"):
        return backend.zeros_like(array)

    # Fallback
    if hasattr(array, "device"):  # PyTorch
        import torch

        return torch.zeros_like(array)
    else:  # NumPy
        return np.zeros_like(array)


def backend_aware_ones_like(array, backend: BaseBackend | None = None):
    """
    Create ones array matching input array's shape and device.

    Args:
        array: Template array
        backend: Backend instance

    Returns:
        Ones array on same device as input

    Examples:
        >>> convergence_flags = backend_aware_ones_like(error_history, backend)
    """
    if backend is not None and hasattr(backend, "ones_like"):
        return backend.ones_like(array)

    # Fallback
    if hasattr(array, "device"):  # PyTorch
        import torch

        return torch.ones_like(array)
    else:  # NumPy
        return np.ones_like(array)


def to_numpy(array, backend: BaseBackend | None = None) -> np.ndarray:
    """
    Convert backend array to NumPy array.

    Handles device transfer (MPS/CUDA → CPU) for PyTorch.

    Args:
        array: Backend-specific array
        backend: Backend instance

    Returns:
        NumPy array

    Examples:
        >>> U_np = to_numpy(U_mps, backend)
        >>> # Safe for visualization, file I/O
    """
    if backend is not None:
        return backend.to_numpy(array)

    # Fallback
    if hasattr(array, "cpu"):  # PyTorch
        return array.detach().cpu().numpy()
    else:
        return np.asarray(array)


def from_numpy(array: np.ndarray, backend: BaseBackend | None = None):
    """
    Convert NumPy array to backend-specific array.

    Args:
        array: NumPy array
        backend: Backend instance

    Returns:
        Backend array on appropriate device

    Examples:
        >>> boundary_tensor = from_numpy(boundary_np, backend)
    """
    if backend is not None:
        return backend.from_numpy(array)

    # Fallback: return as-is (NumPy)
    return array


def get_array_device(array) -> str:
    """
    Get device string for array.

    Args:
        array: Backend array

    Returns:
        Device string ("cpu", "mps:0", "cuda:0", etc.)

    Examples:
        >>> device = get_array_device(U)
        >>> print(f"Array on {device}")
    """
    if hasattr(array, "device"):  # PyTorch
        return str(array.device)
    else:
        return "cpu"


def ensure_same_device(target, source, backend: BaseBackend | None = None):
    """
    Ensure target and source arrays are on same device.

    Args:
        target: Array to move
        source: Reference array (defines target device)
        backend: Backend instance

    Returns:
        Target array on same device as source

    Examples:
        >>> boundary_mps = ensure_same_device(boundary_cpu, U_mps, backend)
    """
    target_device = get_array_device(target)
    source_device = get_array_device(source)

    if target_device == source_device:
        return target

    # PyTorch device transfer
    if hasattr(source, "device") and hasattr(target, "to"):
        return target.to(source.device)

    # Otherwise convert through backend
    if backend is not None:
        target_np = to_numpy(target, backend)
        return from_numpy(target_np, backend)

    return target


# ==================================================================
# Utility Functions
# ==================================================================


def validate_backend_compatibility(backend1: Any, backend2: Any) -> bool:
    """
    Check if two backends are compatible.

    Args:
        backend1: First backend or None
        backend2: Second backend or None

    Returns:
        True if compatible (both None or same backend type)
    """
    if backend1 is None and backend2 is None:
        return True

    if backend1 is None or backend2 is None:
        return False

    return type(backend1) is type(backend2)


def get_backend_name(backend: BaseBackend | None) -> str:
    """
    Get backend name string.

    Args:
        backend: Backend instance

    Returns:
        Backend name ("numpy", "torch", "jax", etc.)
    """
    if backend is None:
        return "numpy"

    if hasattr(backend, "name"):
        return backend.name

    return type(backend).__name__.replace("Backend", "").lower()


__all__ = [
    "backend_aware_any",
    "backend_aware_assign",
    "backend_aware_copy",
    "backend_aware_ones_like",
    "backend_aware_zeros_like",
    "ensure_same_device",
    "from_numpy",
    "get_array_device",
    "get_backend_name",
    "has_nan_or_inf",
    "to_numpy",
    "validate_backend_compatibility",
]

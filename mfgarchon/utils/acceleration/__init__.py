"""
Acceleration Utilities for MFGarchon

This module provides acceleration utilities for high-performance computing
across different computational backends (JAX, PyTorch, etc.).

The module is organized to separate framework-specific utilities while
providing common interfaces for acceleration operations.

Components:
- JAX utilities: jax_utils.py - JAX-specific acceleration functions
- PyTorch utilities: torch_utils.py - PyTorch KDE, tridiagonal solver, device management

This replaces the old mfgarchon/accelerated/ directory with better organization.
"""

from __future__ import annotations

import importlib.util

from mfgarchon.utils.mfg_logging import get_logger

logger = get_logger(__name__)
# Check JAX availability
try:
    import jax

    HAS_JAX = True
    HAS_GPU = any("gpu" in str(d).lower() for d in jax.devices())
    DEFAULT_DEVICE = jax.devices()[0]
except ImportError:
    HAS_JAX = False
    HAS_GPU = False
    DEFAULT_DEVICE = None

# Re-export JAX utilities (explicit imports, Issue #756)
try:
    from .jax_utils import (
        HAS_JAX,
        adaptive_time_step,
        apply_boundary_conditions,
        compute_convergence_error,
        compute_drift,
        compute_hamiltonian,
        compute_jacobian,
        compute_jacobian_jit,
        compute_optimal_control,
        create_optimization_schedule,
        ensure_jax_available,
        finite_difference_1d,
        finite_difference_2d,
        from_device,
        mass_conservation_constraint,
        memory_usage_tracker,
        profile_jax_function,
        to_device,
        vectorized_solve,
    )
    from .jax_utils import (
        tridiagonal_solve as jax_tridiagonal_solve,
    )

    # Only mark as available if JAX is actually installed
    JAX_UTILS_AVAILABLE = HAS_JAX
except ImportError:
    JAX_UTILS_AVAILABLE = False

# PyTorch utilities, re-exported LAZILY (PEP 562).
#
# This block used to `from .torch_utils import (...)` eagerly, and `torch_utils.py:16` is a bare
# `import torch`. Three independent routes reach this file during `import mfgarchon` -- through
# `adjoint_validation`, through `utils.geometry`, and through `backends` -- so torch was imported
# unconditionally by anyone who touched the package, whichever route they arrived by. Measured on
# 1aa71b98: deferring this and the eager registrations in `backends/__init__.py` together takes
# `import mfgarchon` from 4.12s to 3.30s and removes torch from `sys.modules` entirely. Deferring
# either one alone changes nothing, because the other route still arrives. #1930.
#
# `TORCH_UTILS_AVAILABLE` stays eager and cheap: it answers "is torch installed", which
# `importlib.util.find_spec` settles without importing anything.
TORCH_UTILS_AVAILABLE = importlib.util.find_spec("torch") is not None

_LAZY_TORCH = {
    "HAS_CUDA": "torch_utils",
    "HAS_MPS": "torch_utils",
    "HAS_TORCH": "torch_utils",
    "GaussianKDE": "torch_utils",
    "ensure_torch_available": "torch_utils",
    "get_default_device": "torch_utils",
    "to_numpy": "torch_utils",
    "to_tensor": "torch_utils",
    "torch_tridiagonal_solve": "torch_utils",
}


def __getattr__(name: str):
    """Import `torch_utils` on first use of a name it owns, not at package import."""
    if name in _LAZY_TORCH:
        from importlib import import_module

        module = import_module(f".{_LAZY_TORCH[name]}", __name__)
        attr = "tridiagonal_solve" if name == "torch_tridiagonal_solve" else name
        value = getattr(module, attr)
        globals()[name] = value  # bind, so the next access does not re-enter here
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_LAZY_TORCH])


def get_acceleration_info():
    """Get information about available acceleration utilities."""
    info = {
        "jax_utils_available": JAX_UTILS_AVAILABLE,
        "torch_utils_available": TORCH_UTILS_AVAILABLE,
    }

    if JAX_UTILS_AVAILABLE:
        try:
            from .jax_utils import HAS_GPU, HAS_JAX

            info.update(
                {
                    "jax_available": HAS_JAX,
                    "jax_gpu_available": HAS_GPU,
                }
            )
        except ImportError as e:
            logger.debug(f"Could not retrieve JAX detailed info: {e}")

    if TORCH_UTILS_AVAILABLE:
        try:
            from .torch_utils import HAS_CUDA, HAS_MPS, HAS_TORCH

            info.update(
                {
                    "torch_available": HAS_TORCH,
                    "torch_cuda_available": HAS_CUDA,
                    "torch_mps_available": HAS_MPS,
                }
            )
        except ImportError as e:
            logger.debug(f"Could not retrieve PyTorch detailed info: {e}")

    return info


__all__ = [
    # Module-level availability flags
    "JAX_UTILS_AVAILABLE",
    "TORCH_UTILS_AVAILABLE",
    "get_acceleration_info",
    # JAX utilities (available when JAX installed)
    "HAS_JAX",
    "adaptive_time_step",
    "apply_boundary_conditions",
    "compute_convergence_error",
    "compute_drift",
    "compute_hamiltonian",
    "compute_jacobian",
    "compute_jacobian_jit",
    "compute_optimal_control",
    "create_optimization_schedule",
    "ensure_jax_available",
    "finite_difference_1d",
    "finite_difference_2d",
    "from_device",
    "mass_conservation_constraint",
    "memory_usage_tracker",
    "profile_jax_function",
    "to_device",
    "jax_tridiagonal_solve",
    "vectorized_solve",
    # PyTorch utilities (available when torch installed)
    "HAS_CUDA",
    "HAS_MPS",
    "HAS_TORCH",
    "GaussianKDE",
    "ensure_torch_available",
    "get_default_device",
    "to_numpy",
    "to_tensor",
    "torch_tridiagonal_solve",
]

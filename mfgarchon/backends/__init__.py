"""
MFGarchon Computation Backends

This module provides different computational backends for MFG solving:
- PyTorch: CUDA/MPS acceleration with neural network support
- JAX: XLA compilation with GPU/TPU support
- Numba: CPU JIT compilation for imperative algorithms
- NumPy: CPU baseline for compatibility

Tiered auto-selection priority: torch > jax > numpy
"""

from __future__ import annotations

import warnings
from typing import Any

from mfgarchon.utils.mfg_logging import get_logger

logger = get_logger(__name__)
# Backend registry
_BACKENDS = {}
_DEFAULT_BACKEND = "numpy"


def register_backend(name: str, backend_class):
    """Register a computational backend."""
    _BACKENDS[name] = backend_class


def get_available_backends() -> dict[str, bool]:
    """Get list of available backends with their availability status."""
    backends = {"numpy": True}  # NumPy is always available

    # Check PyTorch availability
    try:
        import torch

        backends["torch"] = True
        backends["torch_cuda"] = torch.cuda.is_available()
        backends["torch_mps"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        backends["torch"] = False
        backends["torch_cuda"] = False
        backends["torch_mps"] = False

    # Check JAX availability
    try:
        import jax

        backends["jax"] = True
        backends["jax_gpu"] = any("gpu" in str(d).lower() for d in jax.devices())
    except ImportError:
        backends["jax"] = False
        backends["jax_gpu"] = False

    return backends


def create_backend(backend_name: str | None = None, **kwargs):
    """
    Create a computational backend instance.

    Tiered auto-selection priority: torch > jax > numpy

    Args:
        backend_name: Backend to use ("torch", "jax", "numpy", or None for auto)
                     None/auto will select best available in order: torch > jax > numpy
        **kwargs: Backend-specific configuration

    Returns:
        Backend instance

    Example:
        >>> # Auto-select (torch > jax > numpy)
        >>> backend = create_backend()

        >>> # Explicit choice
        >>> backend = create_backend("jax")
    """
    if backend_name is None or backend_name == "auto":
        # numpy, unconditionally. ~~torch > jax > numpy~~ [CORRECTED 2026-08-17] -- the tiered
        # priority was written for "RL infrastructure" that `alg/` no longer contains, and it
        # auto-selected a backend that is measurably wrong for this library: on this machine torch
        # is 9.2-361x slower than numpy across mean/max/trapezoid, and on MPS it narrows float64 to
        # float32 while the repository asserts to 1e-10 and 1e-12 (#1921). Solvers already pass
        # `backend or "numpy"`, so this only ever caught callers who asked for "auto" and got a
        # silently lower precision than they had.
        #
        # An accelerator earns the default by a measurement on this workload, not by being present.
        backend_name = "numpy"

    if backend_name not in _BACKENDS:
        if backend_name == "torch":
            # Try to register PyTorch backend
            try:
                from .torch_backend import TorchBackend

                register_backend("torch", TorchBackend)
            except ImportError:
                raise ImportError(
                    "PyTorch backend requested but not available. Install with: pip install torch"
                ) from None
        elif backend_name == "jax":
            # Try to register JAX backend
            try:
                from .jax_backend import JAXBackend

                register_backend("jax", JAXBackend)
            except ImportError:
                raise ImportError(
                    "JAX backend requested but not available. Install with: pip install 'mfgarchon[jax]'"
                ) from None
        elif backend_name == "numpy":
            from .numpy_backend import NumPyBackend

            register_backend("numpy", NumPyBackend)
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

    return _BACKENDS[backend_name](**kwargs)


def get_backend_info() -> dict[str, Any]:
    """Get information about available backends."""
    available = get_available_backends()
    info = {
        "available_backends": available,
        "default_backend": _DEFAULT_BACKEND,
        "registered_backends": list(_BACKENDS.keys()),
    }

    # Add PyTorch-specific info if available
    if available.get("torch", False):
        try:
            import torch

            info["torch_info"] = {
                "version": torch.__version__,
                "cuda_available": available.get("torch_cuda", False),
                "mps_available": available.get("torch_mps", False),
            }

            if available.get("torch_cuda", False):
                info["torch_info"].update(
                    {
                        "cuda_version": torch.version.cuda,
                        "cuda_device_count": torch.cuda.device_count(),
                        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    }
                )

        except (ImportError, AttributeError, RuntimeError) as e:
            # Issue #547: Backend info retrieval can fail for various reasons
            error_msg = f"PyTorch available but info retrieval failed: {type(e).__name__}"
            info["torch_info"] = {"error": error_msg}
            logger.debug("Failed to retrieve PyTorch backend info: %s", e)

    # Add JAX-specific info if available
    if available.get("jax", False):
        try:
            import jax

            info["jax_info"] = {
                "version": jax.__version__,
                "devices": [str(d) for d in jax.devices()],
                "default_device": str(jax.devices()[0]),
                "has_gpu": any("gpu" in str(d).lower() for d in jax.devices()),
            }
        except (ImportError, AttributeError, RuntimeError) as e:
            # Issue #547: Backend info retrieval can fail for various reasons
            error_msg = f"JAX available but info retrieval failed: {type(e).__name__}"
            info["jax_info"] = {"error": error_msg}
            logger.debug("Failed to retrieve JAX backend info: %s", e)

    return info


# Initialize default backends
try:
    from .numpy_backend import NumPyBackend

    register_backend("numpy", NumPyBackend)
except ImportError:
    warnings.warn("NumPy backend not available")

# torch and jax are NOT registered here. `create_backend` already carries the on-demand path --
# `if backend_name not in _BACKENDS:` imports and registers whichever backend was asked for -- and
# eager registration is what made that branch unreachable. `test_backend_factory.py` records the
# consequence: "backends/__init__.py registers 'torch' into _BACKENDS whether or not torch exists.
# The `if backend_name not in _BACKENDS` branch is therefore unreachable."
#
# `torch_backend.py` imports torch (inside a `try/except ImportError`, so the module itself
# loads fine without it -- but the import still runs when torch IS installed), so registering it
# eagerly imported torch for
# anyone who touched this package. Measured on 1aa71b98: deferring this together with the eager
# `torch_utils` re-export in `utils/acceleration/__init__.py` cuts ~0.8s off `import mfgarchon`
# and removes torch from `sys.modules`. No absolute pair is quoted here: five measurement rounds
# put the absolutes between 4.01 and 4.30 (base) and 3.24 and 3.40 (head) on one machine, while
# the DELTA held at 0.73-0.85. Restating a pair invites the reader to check it against a run that
# will not reproduce it -- and two of the four restatements of it in this branch had already
# drifted apart (3.27 vs 3.30). The changelog carries one measurement with its conditions.
# Deferring either alone changes nothing -- three
# independent routes reach torch and cutting one leaves the others. #1930.
#
# numpy stays eager: it is a hard dependency, costs 0.07s, and several callers assume it is
# registered the moment the package is imported.


# Ensure essential backends are always available for compatibility
def ensure_numpy_backend():
    """Ensure NumPy backend is always available for compatibility."""
    if "numpy" not in _BACKENDS:
        try:
            from .numpy_backend import NumPyBackend

            register_backend("numpy", NumPyBackend)
        except ImportError as e:
            raise ImportError("NumPy backend is required for MFGarchon compatibility") from e


# Auto-initialize on import
ensure_numpy_backend()

# Export strategy selection utilities
__all__ = [
    "create_backend",
    "get_available_backends",
    "get_backend_info",
    "register_backend",
]

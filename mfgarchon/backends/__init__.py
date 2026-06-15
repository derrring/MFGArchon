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

# Issue #1072 (interim): the JAX backend ghost-implements its HJB/FP time steps
# (jax_backend.py `_hjb_step_impl` / `_fpk_step_impl`) using 2nd-order central
# differences only, instead of calling the high-order operators the NumPy path
# uses (e.g. WENO5 / upwind reconstruction). Selecting JAX for any scheme other
# than 2nd-order central therefore silently solves *different* math, so a
# cross-backend "speedup" benchmark compares apples to oranges. Full backend
# uniformity is the deferred design in #1072; this is the documented stop-gap.
#
# `fdm_centered` is the one scheme whose stencil the JAX 2nd-order central
# implementation faithfully reproduces — every other scheme (upwind, WENO5, SL,
# GFDM, FEM, FVM, meshless) uses a stencil JAX does not implement.
_JAX_NATIVE_SCHEME_VALUES = frozenset({"fdm_centered"})
# One-time guard, keyed by scheme value (matches the repo's set-based idiom, e.g.
# utils/pde_coefficients._VOLATILITY_LEGACY_KEY_WARNED).
_JAX_SCHEME_DOWNGRADE_WARNED: set[str] = set()


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

    # Check Numba availability
    try:
        import importlib.util

        numba_spec = importlib.util.find_spec("numba")
        backends["numba"] = numba_spec is not None
    except ImportError:
        backends["numba"] = False

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
        available = get_available_backends()

        # Tiered Priority: torch > jax > numpy (Phase 3 strategy)
        # PyTorch has priority (leverages RL infrastructure)
        if available.get("torch", False):
            backend_name = "torch"
            # Auto-detect best device: CUDA > MPS > CPU
            if available.get("torch_cuda", False):
                kwargs.setdefault("device", "cuda")
                logger.info("Auto-selected PyTorch backend with CUDA (RL infrastructure available)")
            elif available.get("torch_mps", False):
                kwargs.setdefault("device", "mps")
                logger.info("Auto-selected PyTorch backend with MPS (Apple Silicon)")
            else:
                kwargs.setdefault("device", "cpu")
                logger.info("Auto-selected PyTorch backend with CPU (no GPU available)")

        # JAX fallback (scientific computing alternative)
        elif available.get("jax", False):
            backend_name = "jax"
            # Auto-detect: GPU > CPU
            if available.get("jax_gpu", False):
                kwargs.setdefault("device", "gpu")
                logger.info("Auto-selected JAX backend with GPU (PyTorch not available)")
            else:
                kwargs.setdefault("device", "cpu")
                logger.info("Auto-selected JAX backend with CPU (PyTorch not available)")

        # NumPy baseline (universal compatibility)
        else:
            backend_name = "numpy"
            logger.info("Using NumPy backend (no acceleration available)")

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
        elif backend_name == "numba":
            # Try to register Numba backend
            try:
                from .numba_backend import NumbaBackend

                register_backend("numba", NumbaBackend)
            except ImportError:
                raise ImportError(
                    "Numba backend requested but not available. Install with: pip install numba"
                ) from None
        elif backend_name == "numpy":
            from .numpy_backend import NumPyBackend

            register_backend("numpy", NumPyBackend)
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

    return _BACKENDS[backend_name](**kwargs)


def warn_if_jax_scheme_downgraded(backend_name: str | None, scheme: Any) -> bool:
    """Warn once when the JAX backend is paired with a non-2nd-order-central scheme.

    Issue #1072 (interim mitigation). The JAX backend's ``hjb_step`` / ``fpk_step``
    ghost-implement 2nd-order central differences rather than calling the
    high-order operators (WENO5 / upwind) the NumPy path uses. When JAX is
    selected for a scheme whose stencil it does not implement, the requested
    discretization is silently replaced by 2nd-order central, so results differ
    from the NumPy high-order path and cross-backend comparison is not
    quantitatively meaningful.

    The warning is emitted at most once per distinct scheme value. It fires only
    on the JAX backend with a non-native scheme; it is a no-op for the NumPy /
    other backends, for ``fdm_centered`` (which JAX reproduces exactly), and when
    no scheme is known (e.g. Expert Mode with injected solvers).

    Parameters
    ----------
    backend_name : str | None
        Resolved backend name (e.g. ``"jax"``, ``"numpy"``). ``None`` means the
        default backend, which is never JAX.
    scheme : Any
        A ``NumericalScheme`` enum member or its string value. ``None`` skips the
        check (no scheme to classify).

    Returns
    -------
    bool
        ``True`` if a warning was emitted by this call, ``False`` otherwise.
    """
    if backend_name != "jax" or scheme is None:
        return False

    # Accept either a NumericalScheme enum (has `.value`) or a bare string.
    scheme_value = str(getattr(scheme, "value", scheme))

    if scheme_value in _JAX_NATIVE_SCHEME_VALUES:
        return False
    if scheme_value in _JAX_SCHEME_DOWNGRADE_WARNED:
        return False

    _JAX_SCHEME_DOWNGRADE_WARNED.add(scheme_value)
    warnings.warn(
        f"JAX backend selected with scheme '{scheme_value}', but the JAX backend "
        f"implements only 2nd-order central differences (it ghost-implements "
        f"hjb_step/fpk_step instead of the high-order operators, e.g. WENO5/upwind, "
        f"used by the NumPy path). The requested stencil is silently replaced by "
        f"2nd-order central, so results will differ from the NumPy high-order path "
        f"and cross-backend comparison is not quantitatively meaningful. Use the "
        f"NumPy backend for this scheme, or scheme 'fdm_centered' with JAX. "
        f"(Issue #1072, interim; full backend uniformity is deferred.)",
        stacklevel=2,
    )
    return True


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

try:
    from .torch_backend import TorchBackend

    register_backend("torch", TorchBackend)
except ImportError:
    logger.debug("PyTorch backend not available (optional)")
    # PyTorch is optional

try:
    from .jax_backend import JAXBackend

    register_backend("jax", JAXBackend)
except ImportError:
    logger.debug("JAX backend not available (optional)")
    # JAX is optional


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
    "warn_if_jax_scheme_downgraded",
]

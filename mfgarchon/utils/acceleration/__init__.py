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

import sys
from typing import TYPE_CHECKING

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
# ~0.8s off `import mfgarchon` and removes torch from `sys.modules` entirely (the changelog holds
# the one measurement with its conditions; absolutes drift ~0.4s between rounds). Deferring
# either one alone changes nothing, because the other route still arrives. #1930.
#
# `TORCH_UTILS_AVAILABLE` is deferred too, for the same reason the names are.
#
# It must answer "did `import torch` succeed", which is what the old eager
# `try: from .torch_utils import ...` measured. ~~`find_spec("torch") is not None`~~ [CORRECTED
# 2026-08-14] answers "is torch on disk" instead: for an installed-but-broken torch the old form
# gave False and that gives True, and `get_acceleration_info()` then reports the
# self-contradictory `{"torch_utils_available": True, "torch_available": False}`. Found by review.
#
# Computing it eagerly and correctly is not available: the only authority is `torch_utils.HAS_TORCH`,
# and importing that module imports torch whenever torch works -- which is the whole thing this
# change removes. (An earlier attempt at this correction did exactly that and put torch back in
# `sys.modules`; caught by the test below rather than by inspection.) So the flag joins the lazy
# map: correct on first read, free until then.

_LAZY_TORCH = {
    "TORCH_UTILS_AVAILABLE": "torch_utils",
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

# A KNOWN COST OF THIS DEFERRAL, stated here because it is not recoverable and should not be
# re-attempted. Defining a module-level `__getattr__` tells a type checker that ANY attribute of
# this module may exist, so mypy stops reporting `attr-defined` for the whole module -- not only for
# the ten deferred names. Measured with a probe asserting three nonexistent names, two through this
# module and one through `mfgarchon.backends` as a control that must stay flagged in every state:
#
#     base 7ac9df18                        3 errors
#     head, before the block below         1 error   (the control only)
#     head, with the block below           1 error   <- the block does NOT restore it
#     head, `__getattr__ -> object`        1 error
#     head, `__getattr__ -> bool`          1 error
#
# ~~Re-declaring the names under TYPE_CHECKING restores the check~~ [CORRECTED 2026-08-15] -- that
# was the fifth review's prescription and it is wrong on the mechanism, measured above: the
# catch-all is the *presence* of `__getattr__`, and declaring ten names adds types for those ten
# without removing it. Narrowing the return type does not help either. So a downstream user's
# `acc.typo` type-checks clean, and that is the price of not importing torch.
#
# The block below is kept for what it does deliver: real types for the ten names under IDE hover
# and `reveal_type`, at zero import cost, since it never runs. `ci.yml` scopes mypy to
# `mfgarchon/config`, so no gate here would have surfaced any of this.
if TYPE_CHECKING:
    from .torch_utils import (
        HAS_CUDA,
        HAS_MPS,
        HAS_TORCH,
        GaussianKDE,
        ensure_torch_available,
        get_default_device,
        to_numpy,
        to_tensor,
    )
    from .torch_utils import (
        tridiagonal_solve as torch_tridiagonal_solve,
    )

    TORCH_UTILS_AVAILABLE: bool


def __getattr__(name: str):
    """Import `torch_utils` on first use of a name it owns, not at package import."""
    if name in _LAZY_TORCH:
        from importlib import import_module

        try:
            module = import_module(f".{_LAZY_TORCH[name]}", __name__)
        except ImportError:
            if name == "TORCH_UTILS_AVAILABLE":
                globals()[name] = False
                return False
            raise
        if name == "TORCH_UTILS_AVAILABLE":
            value = bool(getattr(module, "HAS_TORCH", False))
            globals()[name] = value
            return value
        attr = "tridiagonal_solve" if name == "torch_tridiagonal_solve" else name
        value = getattr(module, attr)
        globals()[name] = value  # bind, so the next access does not re-enter here
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *_LAZY_TORCH})  # a set: a name bound by __getattr__ is in both


def get_acceleration_info():
    """Get information about available acceleration utilities."""
    info = {
        "jax_utils_available": JAX_UTILS_AVAILABLE,
        "torch_utils_available": sys.modules[__name__].TORCH_UTILS_AVAILABLE,
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

    # `sys.modules[__name__].NAME`, not the bare name and not `__getattr__(...)` directly.
    # A bare global lookup inside a function does not consult the module `__getattr__` -- it goes to
    # `module.__dict__` and raises NameError (ruff F821 caught that; the tests did not). Calling the
    # dunder by hand fixes the NameError but skips `__dict__`, so it reads THROUGH an override:
    # `m.TORCH_UTILS_AVAILABLE = False` then reported True here, and `monkeypatch.setattr` on this
    # flag would have been silently ignored. A real attribute lookup checks `__dict__` first and
    # falls through to `__getattr__` only when unbound. Found by review.
    if sys.modules[__name__].TORCH_UTILS_AVAILABLE:
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

"""One owner for `jax_enable_x64` (#1923).

`jax.config.update("jax_enable_x64", ...)` is a **process-global** switch. Three sites wrote it and
they did not agree:

    backends/jax_backend.py             False when precision='float32', True otherwise  (a policy)
    utils/acceleration/jax_utils.py     unconditional True, at MODULE IMPORT
    alg/.../meshless_galerkin/mls_basis.py  unconditional True, PER CALL

Last writer wins, and which is last depends on import and call order. Nothing failed when they
disagreed: the result is a precision, not an exception. Measured on
`jax.experimental.sparse.linalg.spsolve` against scipy on a tridiagonal system with a known answer,
the error was 2.4e-07 -- float32 -- where scipy gave 4.4e-16; setting the flag explicitly took it to
bit-identical agreement at every size.

THE ASYMMETRY THAT DECIDES THE DESIGN. Two of those writes are requirements ("this computation needs
float64") and one is a policy ("the user asked for float32"). A requirement that silently overwrites
a policy is the defect; so is a policy that silently starves a requirement. Neither can be resolved
by ordering, because both are legitimate -- so the owner makes the conflict *visible* instead of
picking a winner by import order.
"""

from __future__ import annotations

from typing import Any

__all__ = ["require_x64", "set_x64_policy", "x64_state"]

_POLICY: bool | None = None  # what a backend last declared, None if none has
_POLICY_SOURCE: str | None = None


def _jax() -> Any:
    import jax

    return jax


def x64_state() -> tuple[bool, bool | None, str | None]:
    """`(effective, declared_policy, policy_source)` — for diagnostics and tests."""
    return bool(_jax().config.jax_enable_x64), _POLICY, _POLICY_SOURCE


def set_x64_policy(enabled: bool, source: str) -> None:
    """Declare the process precision policy. Called by a backend that a user configured.

    This is the only write that may turn x64 OFF, because it is the only one that represents
    something the user asked for.
    """
    global _POLICY, _POLICY_SOURCE
    _POLICY, _POLICY_SOURCE = bool(enabled), source
    _jax().config.update("jax_enable_x64", bool(enabled))


def require_x64(reason: str) -> None:
    """Declare that the caller needs float64, and fail loud if a policy forbids it.

    Silently flipping the switch is what this replaces: a per-call `update(..., True)` inside a
    library routine turned a user's float32 backend into a float64 one for the rest of the process,
    with no diagnostic and no way to notice except by measuring a precision.
    """
    if _POLICY is False:
        raise RuntimeError(
            f"{reason} requires float64, but jax_enable_x64 was explicitly disabled by "
            f"{_POLICY_SOURCE or 'an unknown caller'}. `jax_enable_x64` is PROCESS-GLOBAL, so this "
            f"cannot be resolved locally: either construct that backend with precision='float64', "
            f"or run this computation in a separate process. Silently enabling it here would change "
            f"the precision of everything else already running (Issue #1923)."
        )
    _jax().config.update("jax_enable_x64", True)

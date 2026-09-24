#!/usr/bin/env python3
"""
Hamiltonian signature adapter -- deprecated since #2378 phase 5.

It used to guess which of several argument orders a Hamiltonian callable took and convert
to a "standard" ``(x, m, p, t)``. #2375 ruling 8 fixes one order, ``(t, x, p, m)``, and the library
refuses a callable written in another rather than guessing. These wrappers now bind through
``bind_user_callable`` with ``HAMILTONIAN_SLOTS``, the one owner of that rule, and keep their call
interface until they are removed.

Use instead:
    >>> from mfgarchon.types.callable_protocols import HAMILTONIAN_SLOTS, bind_user_callable
    >>> H = bind_user_callable(lambda t, x, p, m: 0.5 * p**2 + m, HAMILTONIAN_SLOTS, role="hamiltonian")
    >>> H(t=0.0, x=1.0, p=2.0, m=0.5)
    2.5
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mfgarchon.types.callable_protocols import HAMILTONIAN_SLOTS, bind_user_callable
from mfgarchon.utils.deprecation import deprecated

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from numpy.typing import NDArray

_REPLACEMENT = (
    "bind_user_callable(func, HAMILTONIAN_SLOTS, role=...) from mfgarchon.types.callable_protocols, "
    "or a HamiltonianBase called by keyword"
)
_REASON = (
    "#2375 ruling 8: a Hamiltonian takes (t, x, p, m); guessing among other orders is what the "
    "library now refuses (#2378 phase 5)"
)


def _refuse_hint(signature_hint: str | None) -> None:
    if signature_hint is not None:
        raise TypeError(
            f"signature_hint={signature_hint!r} is no longer accepted: since #2375 ruling 8 the order is "
            f"(t, x, p, m), read from the callable's parameter names. Reorder the callable instead."
        )


@deprecated(since="v0.22.0", replacement=_REPLACEMENT, reason=_REASON)
class HamiltonianAdapter:
    """A Hamiltonian callable bound to ``(t, x, p, m)``; see the module docstring."""

    def __init__(self, hamiltonian_func: Callable, signature_hint: str | None = None):
        _refuse_hint(signature_hint)
        self.func = hamiltonian_func
        self._bound = bind_user_callable(hamiltonian_func, HAMILTONIAN_SLOTS, role="hamiltonian")

    def __call__(
        self,
        x: float | NDArray[np.float64],
        m: float | NDArray[np.float64],
        p: float | NDArray[np.float64],
        t: float = 0.0,
    ) -> float | NDArray[np.float64]:
        """Evaluate the Hamiltonian. The adapter's own call interface is kept until removal."""
        return self._bound(t=t, x=x, p=p, m=m)

    def get_info(self) -> dict[str, Any]:
        """How the callable is bound: ``binding`` is the rule that matched."""
        return {
            "binding": self._bound.binding,
            "passes": self._bound.passes,
            "function_name": getattr(self.func, "__name__", "<unknown>"),
        }

    def __repr__(self) -> str:
        return f"HamiltonianAdapter(binding={self._bound.binding}, func={getattr(self.func, '__name__', '<lambda>')})"


@deprecated(since="v0.22.0", replacement=_REPLACEMENT, reason=_REASON)
def create_hamiltonian_adapter(
    hamiltonian_func: Callable | None = None, signature_hint: str | None = None
) -> HamiltonianAdapter | None:
    """A ``HamiltonianAdapter`` for ``hamiltonian_func``, or None; see the module docstring."""
    _refuse_hint(signature_hint)
    if hamiltonian_func is None:
        return None
    return HamiltonianAdapter(hamiltonian_func)


@deprecated(since="v0.22.0", replacement=_REPLACEMENT, reason=_REASON)
def adapt_hamiltonian(
    hamiltonian_func: Callable,
    x: float | NDArray[np.float64],
    m: float | NDArray[np.float64],
    p: float | NDArray[np.float64],
    t: float = 0.0,
    signature_hint: str | None = None,
) -> float | NDArray[np.float64]:
    """Bind and evaluate once; see the module docstring."""
    _refuse_hint(signature_hint)
    return bind_user_callable(hamiltonian_func, HAMILTONIAN_SLOTS, role="hamiltonian")(t=t, x=x, p=p, m=m)

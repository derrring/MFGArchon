"""
Callable Signature Detection and Adaptation for IC/BC functions.

Provides automatic detection and wrapping of user-provided callables (m_initial, u_terminal)
so that different signature conventions work transparently.

Supported signatures:
    - f(x) where x is scalar float (1D convention)
    - f(x) where x is ndarray of shape (d,)
    - f(t, x) spatiotemporal, time first (#2375 ruling 8), read at T for u_terminal and 0 for m_initial
    - f(x, y) expanded 2D coordinates (deprecated)
    - f(x, y, z) expanded 3D coordinates (deprecated)

Whether a callable takes x alone, and as a float or an array, is found by probing: that handles
lambdas, functools.partial, C extensions and decorated functions alike. Which argument of a
two-argument callable is time cannot be found by probing, because both orders return a number. So
the parameter names decide it, through ``bind_user_callable`` as for every other user callable, and
the order before #2378 phase 5, ``f(x, t)``, is refused rather than read with x as time (#2431).

Issue #684: Callable signature detection and adaptation.

Example:
    >>> from mfgarchon.utils.callable_adapter import adapt_ic_callable
    >>>
    >>> # User provides array-expecting callable for 1D problem
    >>> m_initial = lambda x: np.exp(-x[0]**2)
    >>>
    >>> sig, adapted = adapt_ic_callable(m_initial, dimension=1, sample_point=0.5)
    >>> # adapted(0.5) now works even though original expected x[0]
"""

from __future__ import annotations

import functools
import inspect
import warnings
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.types.callable_protocols import CONDITION_SLOTS, BoundCallable, bind_user_callable, out_of_order

if TYPE_CHECKING:
    from collections.abc import Callable


class CallableSignature(Enum):
    """Detected signature type for an IC/BC callable."""

    SPATIAL_SCALAR = auto()  # f(x) where x is scalar float (1D)
    SPATIAL_ARRAY = auto()  # f(x) where x is ndarray (d,)
    SPATIOTEMPORAL_TX = auto()  # f(t, x)
    EXPANDED_2D = auto()  # f(x, y) -- deprecated
    EXPANDED_3D = auto()  # f(x, y, z) -- deprecated
    UNKNOWN = auto()


def adapt_ic_callable(
    func: Callable,
    dimension: int,
    sample_point: float | np.ndarray,
    *,
    time_value: float = 0.0,
    role: str = "initial/terminal condition",
) -> tuple[CallableSignature, Callable]:
    """
    Detect a callable's expected signature and return a normalized wrapper.

    The returned wrapper accepts the calling convention used by
    ``_setup_custom_initial_density()`` / ``_setup_custom_final_value()``:
    - 1D: ``wrapper(x_scalar)`` where x_scalar is a Python float
    - nD: ``wrapper(x_array)`` where x_array is ndarray of shape (d,)

    The parameter names decide, through ``bind_user_callable`` (#2375 ruling 8, #2431), and nothing
    they leave open is guessed:

    - names that bind time and space, time first, are read at ``time_value``;
    - names that bind ``x`` alone, or no signature at all, are probed for a float or an array;
    - in 2-D and 3-D, names that start ``(x, y)`` / ``(x, y, z)`` with every later parameter
      defaulted, or a bare ``*args``, are the deprecated expanded coordinates;
    - ``*args, **kwargs`` names nothing and cannot be told from a wrapper around another function, so
      it is read only if x alone works (a wrapper decorated with ``functools.wraps`` is read by the
      wrapped function's names);
    - anything else that needs more than ``x`` is refused: names out of order (``f(x, t)``), time
      without space, a ``functools.partial`` that fixes time, and names that say neither order.

    Args:
        func: User-provided IC/BC callable.
        dimension: Spatial dimension of the problem (1, 2, 3, ...).
        sample_point: A representative point to probe the callable.
            In 1D this is a float; in nD an ndarray of shape (d,).
        time_value: Time value for spatiotemporal wrappers (0.0 for m_initial,
            T for u_terminal).
        role: What the callable is, for the error message (``"u_terminal"``, ``"m_initial"``).

    Returns:
        Tuple of (detected_signature, wrapped_callable).

    Raises:
        TypeError: If the callable cannot be read by one of the rules above.
    """
    attempts: list[tuple[str, str]] = []
    if dimension == 1:
        sample: float | np.ndarray = float(sample_point) if not isinstance(sample_point, float) else sample_point
    else:
        sample = sample_point if isinstance(sample_point, np.ndarray) else np.atleast_1d(sample_point)

    def refuse() -> TypeError:
        return TypeError(_format_signature_error(func, dimension, attempts))

    time_names = [name for name, slot in CONDITION_SLOTS.slot_of().items() if slot == "t"]
    frozen = sorted(set(getattr(func, "keywords", None) or {}) & set(time_names))
    if isinstance(func, functools.partial) and frozen:
        raise TypeError(
            f"{role} is a functools.partial that fixes {frozen[0]}={func.keywords[frozen[0]]!r}, but the library "
            f"reads {role} at its own time. It takes (t, x) or (x): pass lambda x: f({func.keywords[frozen[0]]!r}, x) "
            f"to fix the time yourself."
        )

    names = _positional_names(func)
    if names is None:
        # No readable signature (a builtin, a C extension): x alone is all that can be read.
        found = _probe_space_only(func, dimension, sample, attempts)
        if found is not None:
            return found
        attempts.append(("f(t, x)", "no readable signature, so which argument is time cannot be read"))
        raise refuse()

    try:
        bound = bind_user_callable(func, CONDITION_SLOTS, role=role, spatial_only=True)
    except TypeError as refusal:
        if out_of_order(names, CONDITION_SLOTS):
            raise
        # Names that say neither order. Only the deprecated expanded coordinates are named that way on
        # purpose: (x, y) in 2-D, (x, y, z) in 3-D, with any later parameter defaulted.
        if _names_coordinates(func, dimension):
            found = _probe_expanded(func, dimension, sample, attempts)
            if found is not None:
                return found
        attempts.append(("f(t, x) by parameter names", str(refusal)))
        raise refuse() from None

    if "t" in bound.passes:
        if "x" not in bound.passes:
            raise TypeError(
                f"{role} {getattr(func, '__qualname__', func)!r} takes time and no space. It is a function of "
                f"space: it takes (t, x), time first, or (x). Name its space parameter x."
            )
        result, err = _try_call(functools.partial(bound, t=time_value), x=sample)
        if err is None and _is_valid_output(result):

            def _tx_wrapper(x: float | np.ndarray, _bound: BoundCallable = bound, _t: float = time_value) -> float:
                return float(_bound(t=_t, x=x))

            return CallableSignature.SPATIOTEMPORAL_TX, _tx_wrapper
        attempts.append((f"f(t, x) with t={time_value}, bound by its parameter names", _err_str(err, result)))
        # A functools.wraps wrapper shows the wrapped function's signature, not its own: call it as it
        # really is, with x alone. Any other callable is what its names say, and is refused.
        if isinstance(err, TypeError) and inspect.unwrap(func) is not func:
            found = _probe_space_only(func, dimension, sample, attempts)
            if found is not None:
                return found
        raise refuse()

    # Its names bind x alone.
    found = _probe_space_only(func, dimension, sample, attempts)
    if found is not None:
        return found
    bare_varargs = not names and _takes_varargs(func)
    if bare_varargs and _takes_varkwargs(func):
        # `*args, **kwargs` names nothing, and cannot be told from a wrapper around another function.
        attempts.append(
            (
                "f(*args, **kwargs)",
                f"names none of its parameters, so which argument is time cannot be read. If {role} wraps "
                f"another function, decorate the wrapper with functools.wraps; otherwise take (t, x) or (x)",
            )
        )
        raise refuse()
    # In 2-D and 3-D, a bare *args (`lambda *c`, one argument per axis) and names that start (x, y[, z])
    # with the rest defaulted -- `(x, y=0.0)` binds as x alone -- are the deprecated expanded
    # coordinates, as they always were.
    if dimension in (2, 3) and (bare_varargs or _names_coordinates(func, dimension)):
        found = _probe_expanded(func, dimension, sample, attempts)
        if found is not None:
            return found
    raise refuse()


def _probe_space_only(
    func: Callable, dimension: int, sample: float | np.ndarray, attempts: list[tuple[str, str]]
) -> tuple[CallableSignature, Callable] | None:
    """Call ``func`` with x alone: a float or a ``(1,)`` array in 1-D, the point in n-D."""
    if dimension == 1:
        result, err = _try_call(func, sample)
        if err is None and _is_valid_output(result):
            return CallableSignature.SPATIAL_SCALAR, func
        attempts.append(("f(x) with x=float", _err_str(err, result)))

        result, err = _try_call(func, np.array([sample]))
        if err is None and _is_valid_output(result):

            def _array_wrapper_1d(x: float, _fn: Callable = func) -> float:
                return float(_fn(np.array([x])))

            return CallableSignature.SPATIAL_ARRAY, _array_wrapper_1d
        attempts.append(("f(x) with x=ndarray([x])", _err_str(err, result)))
        return None

    result, err = _try_call(func, sample)
    if err is None and _is_valid_output(result):
        return CallableSignature.SPATIAL_ARRAY, func
    attempts.append(("f(x) with x=ndarray", _err_str(err, result)))
    return None


def _probe_expanded(
    func: Callable, dimension: int, sample: np.ndarray, attempts: list[tuple[str, str]]
) -> tuple[CallableSignature, Callable] | None:
    """Call ``func`` with the point's coordinates as separate arguments (deprecated)."""
    coords = [float(c) for c in sample]
    result, err = _try_call(func, *coords)
    label = "f(x, y)" if dimension == 2 else "f(x, y, z)"
    if err is None and _is_valid_output(result):
        warnings.warn(
            f"IC/BC callable uses expanded coordinate signature {label}. "
            f"This is deprecated. Use f(x) where x is ndarray of shape ({dimension},) instead.",
            DeprecationWarning,
            stacklevel=4,
        )

        def _expanded(x: np.ndarray, _fn: Callable = func) -> float:
            return float(_fn(*(float(c) for c in x)))

        return (CallableSignature.EXPANDED_2D if dimension == 2 else CallableSignature.EXPANDED_3D), _expanded
    attempts.append((f"{label} with expanded coordinates", _err_str(err, result)))
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_call(func: Callable, *args: object, **kwargs: object) -> tuple[object, Exception | None]:
    """Try calling func with given args. Returns (result, None) or (None, exception)."""
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        return None, e


def _takes_varargs(func: Callable) -> bool:
    """Whether ``func`` declares ``*args``."""
    return _declares(func, inspect.Parameter.VAR_POSITIONAL)


def _takes_varkwargs(func: Callable) -> bool:
    """Whether ``func`` declares ``**kwargs``."""
    return _declares(func, inspect.Parameter.VAR_KEYWORD)


def _declares(func: Callable, kind: inspect._ParameterKind) -> bool:
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.kind is kind for p in params)


def _names_coordinates(func: Callable, dimension: int) -> bool:
    """Whether ``func``'s positional parameters are the expanded coordinates of a ``dimension``-D point:
    they start ``x, y`` (2-D) or ``x, y, z`` (3-D), and every later one has a default."""
    if dimension not in (2, 3):
        return False
    try:
        params = list(inspect.signature(func).parameters.values())
    except (TypeError, ValueError):
        return False
    kinds = inspect.Parameter
    positional = [p for p in params if p.kind in (kinds.POSITIONAL_ONLY, kinds.POSITIONAL_OR_KEYWORD)]
    head, rest = positional[:dimension], positional[dimension:]
    return [p.name for p in head] == ["x", "y", "z"][:dimension] and all(p.default is not kinds.empty for p in rest)


def _positional_names(func: Callable) -> list[str] | None:
    """The names of ``func``'s positional parameters, or None if its signature cannot be read."""
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return None
    kinds = inspect.Parameter
    return [p.name for p in params if p.kind in (kinds.POSITIONAL_ONLY, kinds.POSITIONAL_OR_KEYWORD)]


def _is_valid_output(value: object) -> bool:
    """Check that a probe result is a numeric type (signature matched).

    NaN/Inf are accepted here -- they indicate the signature worked but the
    math produced a bad result. The validation layer checks finiteness
    separately (in _validate_callable_ic).
    """
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return True
    if isinstance(value, np.ndarray):
        # Accept scalar-like or 1-element arrays
        return value.ndim == 0 or value.size == 1
    return False


def _err_str(err: Exception | None, result: object) -> str:
    """Format a probe failure for the error message."""
    if err is not None:
        return f"{type(err).__name__}: {err}"
    return f"returned invalid output: {result!r}"


def _format_signature_error(
    func: Callable,
    dimension: int,
    attempts: list[tuple[str, str]],
) -> str:
    """Format a helpful error message listing all attempted calling conventions."""
    func_name = getattr(func, "__name__", repr(func))
    lines = [
        f"Cannot determine signature of IC/BC callable '{func_name}' (dimension={dimension}).",
        "",
        "Attempted calling conventions:",
    ]
    for convention, error in attempts:
        lines.append(f"  - {convention}")
        lines.append(f"    -> {error}")

    lines.append("")
    if dimension == 1:
        lines.append(
            "Accepted signatures for 1D:\n"
            "  f(x)      -- x is a Python float\n"
            "  f(x)      -- x is ndarray of shape (1,)\n"
            "  f(t, x)   -- spatiotemporal, time first; name time t (or time), or name the space parameter x"
        )
    else:
        lines.append(
            f"Accepted signatures for {dimension}D:\n"
            f"  f(x)      -- x is ndarray of shape ({dimension},)\n"
            f"  f(t, x)   -- spatiotemporal, time first; name time t (or time), or name the space parameter x"
        )
    return "\n".join(lines)

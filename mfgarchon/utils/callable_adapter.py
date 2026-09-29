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

    The parameter names decide first, through ``bind_user_callable``: a callable they bind with a
    time slot is time-first and is read at ``time_value``, and one whose names are out of order is
    refused. Only what names cannot say is probed: whether a space-only callable takes a float or an
    array, and the deprecated expanded coordinates.

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
        TypeError: If no supported signature convention works, or if the callable's parameter names
            put time after space -- the order before #2378 phase 5 (``f(x, t)``, ``f(x, t=0.0)``).
    """
    attempts: list[tuple[str, str]] = []
    if dimension == 1:
        scalar_sample = float(sample_point) if not isinstance(sample_point, float) else sample_point
        sample: float | np.ndarray = scalar_sample
    else:
        sample = sample_point if isinstance(sample_point, np.ndarray) else np.atleast_1d(sample_point)

    # --- The names decide first (#2375 ruling 8, #2431) ---
    time_names = [name for name, slot in CONDITION_SLOTS.slot_of().items() if slot == "t"]
    frozen = sorted(set(getattr(func, "keywords", None) or {}) & set(time_names))
    if isinstance(func, functools.partial) and frozen:
        raise TypeError(
            f"{role} is a functools.partial that fixes {frozen[0]}={func.keywords[frozen[0]]!r}, but the library "
            f"reads {role} at its own time. It takes (t, x) or (x): pass lambda x: f({func.keywords[frozen[0]]!r}, x) "
            f"to fix the time yourself."
        )
    names = _positional_names(func)
    bound: BoundCallable | None = None
    refusal: TypeError | None = None
    if names is not None:
        try:
            bound = bind_user_callable(func, CONDITION_SLOTS, role=role, spatial_only=True)
        except TypeError as e:
            if out_of_order(names, CONDITION_SLOTS):
                raise
            refusal = e
    if bound is not None and "t" in bound.passes:
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
        # A TypeError can mean the signature misreports the call, as a functools.wraps wrapper's
        # does: it shows the wrapped (t, x). The probes below call it as it really is.
        if not isinstance(err, TypeError):
            raise TypeError(_format_signature_error(func, dimension, attempts))

    # --- 1D probes ---
    if dimension == 1:
        array_sample = np.array([scalar_sample])

        # Probe 1: f(scalar) -- most common 1D convention
        result, err = _try_call(func, scalar_sample)
        if err is None and _is_valid_output(result):
            return CallableSignature.SPATIAL_SCALAR, func
        attempts.append(("f(x) with x=float", _err_str(err, result)))

        # Probe 2: f(ndarray([x])) -- array-expecting 1D
        result, err = _try_call(func, array_sample)
        if err is None and _is_valid_output(result):
            # Wrap: convert scalar -> array for the user's function
            def _array_wrapper_1d(x: float, _fn: Callable = func) -> float:
                return float(_fn(np.array([x])))

            return CallableSignature.SPATIAL_ARRAY, _array_wrapper_1d
        attempts.append(("f(x) with x=ndarray([x])", _err_str(err, result)))

    # --- nD probes ---
    else:
        array_sample = sample

        # Probe 1: f(ndarray) -- standard nD convention
        result, err = _try_call(func, array_sample)
        if err is None and _is_valid_output(result):
            return CallableSignature.SPATIAL_ARRAY, func
        attempts.append(("f(x) with x=ndarray", _err_str(err, result)))

        # A callable that takes the whole point as its first argument and returns a number with a
        # second one cannot be taking coordinates: the second argument is time, in the order before
        # #2378 phase 5 (`lambda x, tau`). Refuse it rather than read it as expanded coordinates.
        result, err = _try_call(func, array_sample, time_value)
        if err is None and _is_valid_output(result):
            raise TypeError(
                f"{role} {getattr(func, '__qualname__', func)!r} takes the point x first and a second argument "
                f"after it: that is the order before #2378 phase 5, where the second argument is time. Since "
                f"#2375 ruling 8 a {role} takes (t, x), time first: reorder its parameters and name them t, x."
            )

        # Probe 2: f(*components) -- expanded coordinates (deprecated). A callable its names bind
        # time-first never reaches here.
        if dimension == 2 and array_sample.shape == (2,):
            result, err = _try_call(func, float(array_sample[0]), float(array_sample[1]))
            if err is None and _is_valid_output(result):
                warnings.warn(
                    "IC/BC callable uses expanded coordinate signature f(x, y). "
                    "This is deprecated. Use f(x) where x is ndarray of shape (2,) instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )

                def _expanded_2d(x: np.ndarray, _fn: Callable = func) -> float:
                    return float(_fn(float(x[0]), float(x[1])))

                return CallableSignature.EXPANDED_2D, _expanded_2d
            attempts.append(("f(x, y) with expanded coordinates", _err_str(err, result)))

        elif dimension == 3 and array_sample.shape == (3,):
            result, err = _try_call(func, float(array_sample[0]), float(array_sample[1]), float(array_sample[2]))
            if err is None and _is_valid_output(result):
                warnings.warn(
                    "IC/BC callable uses expanded coordinate signature f(x, y, z). "
                    "This is deprecated. Use f(x) where x is ndarray of shape (3,) instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )

                def _expanded_3d(x: np.ndarray, _fn: Callable = func) -> float:
                    return float(_fn(float(x[0]), float(x[1]), float(x[2])))

                return CallableSignature.EXPANDED_3D, _expanded_3d
            attempts.append(("f(x, y, z) with expanded coordinates", _err_str(err, result)))

    # With no readable signature (a builtin, a C extension) nothing says which argument is time, so a
    # callable that needs more than x is refused rather than guessed.
    if names is None:
        attempts.append(("f(t, x)", "no readable signature, so which argument is time cannot be read"))
    elif refusal is not None:
        attempts.append(("f(t, x) by parameter names", str(refusal)))

    # All probes failed
    msg = _format_signature_error(func, dimension, attempts)
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_call(func: Callable, *args: object, **kwargs: object) -> tuple[object, Exception | None]:
    """Try calling func with given args. Returns (result, None) or (None, exception)."""
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        return None, e


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

"""
Deprecation utilities for MFGarchon.

Provides decorators and helpers for managing deprecated APIs with strict
enforcement of the deprecation lifecycle policy.

Created: 2026-01-20 (Issue #616)
Reference: docs/development/DEPRECATION_LIFECYCLE_POLICY.md
"""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

from mfgarchon.utils.mfg_logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def deprecated(
    since: str,
    replacement: str,
    reason: str = "",
    removal: str = "v0.25.0",
    removal_blockers: list[str] | None = None,
    deprecated_on: str | None = None,
) -> Callable[[F], F]:
    """
    Mark a function, class, or method as deprecated.

    This decorator:
    1. Stores deprecation metadata on the object
    2. Issues DeprecationWarning when called
    3. Can be discovered by AST-based enforcement tools

    Removal happens when ALL conditions are met:
    - Deprecated for ≥3 versions OR ≥6 months (whichever is longer)
    - All removal blockers resolved
    - No internal production usage
    - Equivalence test exists

    Args:
        since: Version when deprecation started (e.g., "v0.17.0")
        replacement: New API to use instead (e.g., "use advection_scheme parameter")
        reason: Optional explanation of why deprecated
        removal_blockers: Conditions that prevent removal (default: standard checklist)
        deprecated_on: ISO date (``YYYY-MM-DD``) this deprecation was added. Optional; when set,
            it enables the 6-month removal criterion (the earlier of the version/time policy).
            - "internal_usage": Production code still calls this
            - "equivalence_test": No test verifying old = new
            - "migration_docs": No migration guide in DEPRECATION_MODERNIZATION_GUIDE.md

    Example:
        >>> @deprecated(
        ...     since="v0.17.0",
        ...     replacement="use create_distribution_monitor()",
        ...     reason="Renamed for clarity",
        ...     removal_blockers=["internal_usage", "equivalence_test"],
        ... )
        ... def create_default_monitor():
        ...     return create_distribution_monitor()

    Note:
        The decorated function MUST redirect to the new API internally.
        See DEPRECATION_LIFECYCLE_POLICY.md for full requirements.
    """
    # Default removal blockers if not specified
    if removal_blockers is None:
        removal_blockers = ["internal_usage", "equivalence_test", "migration_docs"]

    def decorator(func: F) -> F:
        # Store metadata on the function object (discoverable by AST tools)
        func._deprecation_meta = {  # type: ignore[attr-defined]
            "since": since,
            "replacement": replacement,
            "reason": reason,
            "removal": removal,
            "removal_blockers": removal_blockers,
            "deprecated_on": deprecated_on,  # ISO date (YYYY-MM-DD); enables the 6-month criterion
            "symbol": func.__name__,
        }

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Issue deprecation warning
            msg = f"{func.__name__} is deprecated since {since}. {replacement}"
            if reason:
                msg += f" Reason: {reason}"

            warnings.warn(msg, DeprecationWarning, stacklevel=2)

            # Call the actual function (which MUST redirect to new API)
            return func(*args, **kwargs)

        # Preserve metadata on wrapper
        wrapper._deprecation_meta = func._deprecation_meta  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return decorator


def deprecated_parameter(
    param_name: str,
    since: str,
    replacement: str,
    removal: str = "v0.25.0",
    removal_blockers: list[str] | None = None,
    deprecated_on: str | None = None,
) -> Callable[[F], F]:
    """
    Mark a function parameter as deprecated.

    Use this when deprecating a parameter while keeping the function.
    The decorator ACTIVELY checks for parameter usage and issues warnings.

    IMPORTANT: Unlike the passive approach, this decorator wraps the function
    and uses inspect.signature to catch deprecated parameter usage automatically.
    No need to manually add warnings.warn() in function body.

    Args:
        param_name: Name of deprecated parameter
        since: Version when deprecation started
        replacement: New parameter to use
        removal_blockers: Conditions preventing removal (default: standard checklist)
        deprecated_on: ISO date (``YYYY-MM-DD``) this deprecation was added. Optional; enables
            the 6-month removal criterion when set.

    Example:
        >>> @deprecated_parameter(
        ...     param_name="conservative",
        ...     since="v0.17.0",
        ...     replacement="advection_scheme",
        ...     removal_blockers=["internal_usage", "equivalence_test"],
        ... )
        ... def FPFDMSolver(advection_scheme="divergence_upwind", conservative=None):
        ...     # Decorator automatically warns if conservative is passed
        ...     if conservative is not None:
        ...         advection_scheme = "divergence_upwind" if conservative else "gradient_upwind"
        ...     # ... continue with advection_scheme

    Note:
        The function body MUST still handle the deprecated parameter and redirect
        to the new parameter. The decorator only issues the warning automatically.
    """
    if removal_blockers is None:
        removal_blockers = ["internal_usage", "equivalence_test", "migration_docs"]

    def decorator(func: F) -> F:
        # Store parameter deprecation metadata
        if getattr(func, "_deprecated_parameters", None) is None:
            func._deprecated_parameters = []  # type: ignore[attr-defined]

        func._deprecated_parameters.append(  # type: ignore[attr-defined]
            {
                "param": param_name,
                "since": since,
                "replacement": replacement,
                "removal": removal,
                "removal_blockers": removal_blockers,
                "deprecated_on": deprecated_on,
            }
        )

        # Get function signature for parameter inspection
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Warn iff the user ACTUALLY passed the deprecated parameter. Bind WITHOUT
            # apply_defaults() so bound.arguments holds only user-supplied args -- the
            # previous apply_defaults() + `value is not None` check warned on every call
            # whenever the parameter's default was non-None (false positive, regardless of
            # whether the caller passed it).
            try:
                bound = sig.bind(*args, **kwargs)
                passed = param_name in bound.arguments
            except TypeError:
                # If binding fails, fall through to original function
                # (Let the original function handle signature errors).
                passed = False

            if passed:
                warnings.warn(
                    f"Parameter '{param_name}' in '{func.__name__}' is deprecated "
                    f"since {since}. Use '{replacement}' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            # Call original function
            return func(*args, **kwargs)

        # Preserve metadata on wrapper
        wrapper._deprecated_parameters = func._deprecated_parameters  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return decorator


def deprecated_value(
    param_name: str,
    deprecated_values: dict[Any, Any],
    since: str,
    removal: str = "v0.25.0",
) -> Callable[[F], F]:
    """
    Mark specific parameter values as deprecated, with automatic remapping.

    Use this when a parameter still exists but certain values have been renamed
    or replaced. The decorator automatically remaps deprecated values to their
    replacements and issues a warning.

    Args:
        param_name: Name of the parameter whose values are deprecated
        deprecated_values: Mapping of {old_value: new_value}. When old_value is
            passed, it is silently replaced with new_value after warning.
        since: Version when deprecation started (e.g., "v0.17.0")
        removal: Version when deprecated values will stop working

    Example:
        >>> @deprecated_value(
        ...     param_name="qp_optimization_level",
        ...     deprecated_values={"smart": "auto", "tuned": "auto", "basic": "auto"},
        ...     since="v0.17.0",
        ... )
        ... def __init__(self, qp_optimization_level="auto"):
        ...     # qp_optimization_level is already remapped when we get here
        ...     self.level = qp_optimization_level

    Note:
        Unlike @deprecated_parameter (which detects deprecated parameter *names*),
        this detects deprecated parameter *values*. The function body receives the
        remapped value directly -- no manual redirect logic needed.
    """

    def decorator(func: F) -> F:
        # Store value deprecation metadata
        if getattr(func, "_deprecated_values", None) is None:
            func._deprecated_values = []  # type: ignore[attr-defined]

        func._deprecated_values.append(  # type: ignore[attr-defined]
            {
                "param": param_name,
                "values": deprecated_values,
                "since": since,
                "removal": removal,
            }
        )

        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if value in deprecated_values:
                        new_value = deprecated_values[value]
                        warnings.warn(
                            f"Value '{value}' for parameter '{param_name}' in "
                            f"'{func.__qualname__}' is deprecated since {since}. "
                            f"Use '{new_value}' instead. "
                            f"Will be removed in {removal}.",
                            DeprecationWarning,
                            stacklevel=2,
                        )
                        bound.arguments[param_name] = new_value
                        return func(*bound.args, **bound.kwargs)
            except TypeError:
                pass

            return func(*args, **kwargs)

        wrapper._deprecated_values = func._deprecated_values  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return decorator


def get_deprecation_metadata(obj: Any) -> dict[str, Any] | None:
    """
    Get deprecation metadata from a decorated object.

    Args:
        obj: Function, class, or method to inspect

    Returns:
        Deprecation metadata dict, or None if not deprecated

    Example:
        >>> meta = get_deprecation_metadata(create_default_monitor)
        >>> print(meta["since"])
        v0.17.0
    """
    return getattr(obj, "_deprecation_meta", None)


def get_deprecated_parameters(func: Callable[..., Any]) -> list[dict[str, str]]:
    """
    Get list of deprecated parameters for a function.

    Args:
        func: Function to inspect

    Returns:
        List of parameter metadata dicts

    Example:
        >>> params = get_deprecated_parameters(FPFDMSolver.__init__)
        >>> print(params[0]["param"])
        conservative
    """
    return getattr(func, "_deprecated_parameters", [])


def _removable_by_policy(since: str, deprecated_on: str | None, current_version: str) -> tuple[bool, str]:
    """Single source of truth for the time/version removal policy (CLAUDE.md).

    A deprecation is age-eligible once **3 minor versions OR 6 months** have elapsed (either
    suffices). The version criterion is always available (from ``since``); the 6-month criterion
    is the earlier path, available only when the deprecation recorded a ``deprecated_on``
    (``YYYY-MM-DD``) date. Returns ``(eligible, reason)``.
    """
    reasons: list[str] = []
    try:
        from packaging import version

        cur = version.parse(current_version.lstrip("v"))
        old = version.parse(since.lstrip("v"))
        minor_diff = (cur.major - old.major) * 100 + (cur.minor - old.minor)
        if minor_diff >= 3:
            return True, f"{minor_diff} minor version(s) since {since} (>= 3)"
        reasons.append(f"{minor_diff} minor version(s) since {since} (< 3)")
    except (ImportError, ValueError):
        reasons.append(f"unparseable version (since={since}, current={current_version})")

    if deprecated_on:
        try:
            from datetime import date

            y, m, d = (int(x) for x in str(deprecated_on).split("-"))
            days = (date.today() - date(y, m, d)).days
            if days >= 183:
                return True, f"{days} days since {deprecated_on} (>= 6 months)"
            reasons.append(f"{days} days since {deprecated_on} (< 6 months)")
        except (ValueError, TypeError):
            reasons.append(f"unparseable deprecated_on={deprecated_on!r}")
    else:
        reasons.append("no deprecated_on date (6-month criterion unavailable)")

    return False, "; ".join(reasons)


def check_removal_readiness(
    obj: Any,
    current_version: str,
    completed_blockers: list[str] | None = None,
) -> dict[str, Any]:
    """
    Check if a deprecated object is ready for removal.

    Removal requires:
    1. Age-eligible: ≥3 minor versions OR ≥6 months elapsed (either suffices, per the
       CLAUDE.md policy). The 6-month path is available only when ``deprecated_on`` was set.
    2. All removal blockers cleared.

    Args:
        obj: Deprecated function, class, or method
        current_version: Current version string (e.g., "v0.20.0")
        completed_blockers: List of blockers that have been resolved

    Returns:
        Dict with keys:
            - "ready": bool - Whether removal is allowed
            - "blocking_reasons": list[str] - Why removal is blocked
            - "minimum_age_met": bool - Whether deprecation is old enough
            - "blockers_cleared": bool - Whether all conditions met

    Example:
        >>> meta = check_removal_readiness(
        ...     old_function,
        ...     current_version="v0.20.0",
        ...     completed_blockers=["internal_usage", "equivalence_test"],
        ... )
        >>> if meta["ready"]:
        ...     print("Can safely remove old_function")
    """
    meta = get_deprecation_metadata(obj)
    if meta is None:
        return {
            "ready": False,
            "blocking_reasons": ["Not a deprecated object"],
            "minimum_age_met": False,
            "blockers_cleared": False,
        }

    completed_blockers = completed_blockers or []
    blocking_reasons = []

    # Check removal blockers
    required_blockers = set(meta.get("removal_blockers", []))
    completed_set = set(completed_blockers)
    remaining_blockers = required_blockers - completed_set

    blockers_cleared = len(remaining_blockers) == 0
    if not blockers_cleared:
        blocking_reasons.extend([f"Blocker not cleared: {b}" for b in remaining_blockers])

    # Check minimum age via the shared policy (3 minor versions OR 6 months).
    minimum_age_met, age_reason = _removable_by_policy(meta["since"], meta.get("deprecated_on"), current_version)
    if not minimum_age_met:
        blocking_reasons.append(f"Not old enough for removal: {age_reason}")

    ready = blockers_cleared and minimum_age_met

    return {
        "ready": ready,
        "blocking_reasons": blocking_reasons,
        "minimum_age_met": minimum_age_met,
        "blockers_cleared": blockers_cleared,
        "remaining_blockers": list(remaining_blockers),
    }


def deprecated_alias(
    old_name: str,
    new_class: type,
    since: str,
) -> Callable[..., Any]:
    """
    Create a deprecated alias for a renamed class or function.

    Use this when renaming a class/function but keeping the old name for
    backward compatibility. The alias will emit a deprecation warning
    when used.

    Args:
        old_name: The deprecated name (for warning message)
        new_class: The new class/function to redirect to
        since: Version when deprecation started (e.g., "v0.17.1")

    Returns:
        A wrapper function that creates instances of new_class with warning

    Example:
        >>> # In module where class was renamed:
        >>> class DriftField:  # New name
        ...     def __init__(self, U, cost, geometry):
        ...         ...
        ...
        >>> # Create deprecated alias
        >>> MFGDriftField = deprecated_alias("MFGDriftField", DriftField, "v0.17.1")
        >>>
        >>> # Usage (will warn):
        >>> drift = MFGDriftField(U, cost, grid)  # DeprecationWarning emitted

    Note:
        For simple renames, this creates a factory function that:
        1. Emits DeprecationWarning
        2. Returns new_class(*args, **kwargs)

        For classes, isinstance checks will work with the new class name only.
    """

    def alias_wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"{old_name} is deprecated since {since}. Use {new_class.__name__} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return new_class(*args, **kwargs)

    # Store metadata for discovery
    alias_wrapper._deprecation_meta = {  # type: ignore[attr-defined]
        "since": since,
        "replacement": new_class.__name__,
        "reason": f"Renamed to {new_class.__name__}",
        "removal_blockers": ["internal_usage", "equivalence_test"],
        "symbol": old_name,
        "alias_for": new_class.__name__,
    }

    # Preserve docstring
    alias_wrapper.__doc__ = f"""
    Deprecated alias for {new_class.__name__}.

    .. deprecated:: {since.lstrip("v")}
        Use `{new_class.__name__}` instead. Will be removed in v1.0.0.
    """
    alias_wrapper.__name__ = old_name

    return alias_wrapper


def validate_kwargs(
    kwargs: dict[str, Any],
    deprecated_kwargs: dict[str, str],
    recognized_kwargs: set[str],
    context: str = "function",
    error_on_deprecated: bool = True,
    warn_on_unrecognized: bool = True,
) -> None:
    """
    Validate kwargs against deprecated and recognized sets.

    Use this for functions/methods that accept **kwargs to prevent silent fail
    when users pass deprecated or unrecognized parameters.

    Args:
        kwargs: The kwargs dict to validate
        deprecated_kwargs: Mapping of deprecated kwarg names to their replacements
            Example: {"hamiltonian": "MFGComponents.hamiltonian_func"}
        recognized_kwargs: Set of valid kwargs that are actually consumed
            Example: {"m_initial", "u_terminal", "boundary_conditions"}
        context: Description for error messages (e.g., "MFGProblem", "create_solver")
        error_on_deprecated: If True, raise ValueError for deprecated kwargs.
            If False, emit DeprecationWarning.
        warn_on_unrecognized: If True, emit UserWarning for unrecognized kwargs.

    Raises:
        ValueError: If deprecated kwargs are passed and error_on_deprecated=True

    Example:
        >>> DEPRECATED = {"hamiltonian": "Use MFGComponents.hamiltonian_func"}
        >>> RECOGNIZED = {"m_initial", "u_terminal"}
        >>> validate_kwargs(
        ...     kwargs={"hamiltonian": my_func, "typo_param": 123},
        ...     deprecated_kwargs=DEPRECATED,
        ...     recognized_kwargs=RECOGNIZED,
        ...     context="MFGProblem",
        ... )
        ValueError: Deprecated kwargs in MFGProblem: 'hamiltonian' -> Use MFGComponents...

    Issue #666: Prevents silent fail where user-provided kwargs are ignored.
    """
    # Check for deprecated kwargs
    deprecated_found = []
    for kwarg_name, replacement in deprecated_kwargs.items():
        if kwarg_name in kwargs:
            deprecated_found.append((kwarg_name, replacement))

    if deprecated_found:
        msg_lines = [f"Deprecated kwargs detected in {context}:", ""]
        for name, replacement in deprecated_found:
            msg_lines.append(f"  - '{name}' -> {replacement}")

        msg = "\n".join(msg_lines)

        if error_on_deprecated:
            raise ValueError(msg)
        else:
            warnings.warn(msg, DeprecationWarning, stacklevel=3)

    # Check for unrecognized kwargs (possibly typos)
    if warn_on_unrecognized:
        all_known = set(deprecated_kwargs.keys()) | recognized_kwargs
        unrecognized = set(kwargs.keys()) - all_known
        if unrecognized:
            warnings.warn(
                f"Unrecognized kwargs in {context}: {sorted(unrecognized)}. "
                f"These may be silently ignored. Known kwargs: {sorted(recognized_kwargs)}",
                UserWarning,
                stacklevel=3,
            )


def _scan_object(obj: Any, name: str, module_name: str, results: list[dict[str, Any]]) -> None:
    """Scan a single object for deprecation metadata."""
    # For property descriptors, check the underlying fget function
    is_property = isinstance(obj, property)
    if is_property and obj.fget is not None:
        obj = obj.fget

    meta = getattr(obj, "_deprecation_meta", None)
    if meta is not None:
        results.append(
            {
                "type": "property" if is_property else ("function" if callable(obj) else "alias"),
                "name": meta.get("symbol", name),
                "module": module_name,
                "since": meta.get("since", ""),
                "replacement": meta.get("replacement", ""),
                "removal": meta.get("removal", "v1.0.0"),
                "removal_blockers": meta.get("removal_blockers") or [],
                "deprecated_on": meta.get("deprecated_on"),
            }
        )

    dep_params = getattr(obj, "_deprecated_parameters", None)
    if dep_params:
        for p in dep_params:
            results.append(
                {
                    "type": "parameter",
                    "name": f"{name}.{p['param']}",
                    "module": module_name,
                    "since": p.get("since", ""),
                    "replacement": p.get("replacement", ""),
                    "removal": p.get("removal", "v1.0.0"),
                    "removal_blockers": p.get("removal_blockers") or [],
                    "deprecated_on": p.get("deprecated_on"),
                }
            )

    dep_values = getattr(obj, "_deprecated_values", None)
    if dep_values:
        for v in dep_values:
            old_vals = ", ".join(str(k) for k in v["values"])
            new_vals = ", ".join(str(val) for val in v["values"].values())
            results.append(
                {
                    "type": "value",
                    "name": f"{name}.{v['param']}",
                    "module": module_name,
                    "since": v.get("since", ""),
                    "replacement": f"values [{old_vals}] -> [{new_vals}]",
                    "removal": v.get("removal", "v1.0.0"),
                    "removal_blockers": v.get("removal_blockers") or [],
                    "deprecated_on": v.get("deprecated_on"),
                }
            )


def scan_deprecated(module: Any, *, recursive: bool = True) -> list[dict[str, Any]]:
    """
    Scan a module tree for all decorated deprecated items.

    Returns a list of metadata dicts for every `@deprecated`, `@deprecated_parameter`,
    `@deprecated_value`, and `deprecated_alias` found in the module and its submodules.

    Args:
        module: Top-level module to scan (e.g., `import mfgarchon; scan_deprecated(mfgarchon)`)
        recursive: If True, scan submodules recursively

    Returns:
        List of dicts with keys: type, name, module, since, replacement, removal

    Example:
        >>> import mfgarchon
        >>> items = scan_deprecated(mfgarchon)
        >>> for item in items:
        ...     print(f"{item['type']:10s} {item['name']:30s} since={item['since']}")
    """
    import importlib
    import pkgutil

    results: list[dict[str, Any]] = []
    visited: set[str] = set()

    def _scan_module(mod: Any) -> None:
        mod_name = getattr(mod, "__name__", str(mod))
        if mod_name in visited:
            return
        visited.add(mod_name)

        # Scan attributes
        for attr_name in dir(mod):
            try:
                obj = getattr(mod, attr_name)
            except Exception:
                logger.debug("Cannot access %s.%s", mod_name, attr_name, exc_info=True)
                continue

            _scan_object(obj, attr_name, mod_name, results)

            # Scan class methods
            if isinstance(obj, type):
                for method_name in dir(obj):
                    if method_name.startswith("__") and method_name != "__init__":
                        continue
                    try:
                        method = getattr(obj, method_name)
                    except Exception:
                        logger.debug("Cannot access %s.%s.%s", mod_name, attr_name, method_name, exc_info=True)
                        continue
                    _scan_object(method, f"{attr_name}.{method_name}", mod_name, results)

        # Recurse into submodules
        if recursive and hasattr(mod, "__path__"):
            try:
                for _importer, submod_name, _ispkg in pkgutil.walk_packages(mod.__path__, prefix=mod.__name__ + "."):
                    try:
                        submod = importlib.import_module(submod_name)
                        _scan_module(submod)
                    except Exception:
                        logger.debug("Cannot import submodule %s", submod_name, exc_info=True)
                        continue
            except Exception:
                logger.debug("Cannot walk submodules of %s", mod_name, exc_info=True)

    _scan_module(module)

    # Deduplicate by (type, name, module)
    seen = set()
    unique = []
    for r in results:
        key = (r["type"], r["name"], r["module"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


def audit_all_deprecations(
    module: Any,
    current_version: str | None = None,
    completed_blockers: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Audit every decorated deprecation in a module against the removal policy.

    The age judgment delegates to the same policy as ``check_removal_readiness``
    (3 minor versions OR 6 months; CLAUDE.md), so the two never disagree. Each *unique*
    deprecation (deduped across the inherited subclasses that surface the same one) is sorted:
        "ready"     -- age-eligible AND all removal blockers cleared
        "not_ready" -- age-eligible but blockers remain
        "active"    -- not yet old enough to remove

    Args:
        module: Module to scan (e.g. ``import mfgarchon``).
        current_version: Version to judge age against (e.g. "v0.19.8"). Defaults to the
            installed ``mfgarchon`` version.
        completed_blockers: Removal blockers cleared globally (e.g. ``["internal_usage"]``).

    Example:
        >>> import mfgarchon
        >>> report = audit_all_deprecations(mfgarchon)
        >>> print(f"Ready for removal: {len(report['ready'])}")
    """
    if current_version is None:
        try:
            from importlib.metadata import version as _pkg_version

            current_version = "v" + _pkg_version("mfgarchon")
        except Exception:
            current_version = "v0.0.0"
    completed = set(completed_blockers or [])

    ready: list[dict[str, Any]] = []
    not_ready: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for raw in scan_deprecated(module):
        # Dedup the identical deprecation surfaced on many inherited subclasses.
        key = (raw.get("name"), raw.get("type"), raw.get("since"))
        if key in seen:
            continue
        seen.add(key)

        eligible, reason = _removable_by_policy(raw.get("since", ""), raw.get("deprecated_on"), current_version)
        item = {**raw, "age_reason": reason, "current_version": current_version}
        if not eligible:
            active.append(item)
            continue
        remaining = sorted(set(item.get("removal_blockers") or []) - completed)
        item["remaining_blockers"] = remaining
        (ready if not remaining else not_ready).append(item)

    return {"ready": ready, "not_ready": not_ready, "active": active}


__all__ = [
    "deprecated",
    "deprecated_parameter",
    "deprecated_value",
    "deprecated_alias",
    "get_deprecation_metadata",
    "get_deprecated_parameters",
    "check_removal_readiness",
    "validate_kwargs",
    "scan_deprecated",
    "audit_all_deprecations",
]

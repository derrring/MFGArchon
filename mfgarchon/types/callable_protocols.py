"""
Type protocols for callable signatures in MFGarchon.

This module defines Protocol classes for callable objects used throughout the framework,
enabling precise type checking and better IDE support.

Usage:
    from mfgarchon.types.callable_protocols import DriftFieldCallable, DiffusionFieldCallable

    def solve_fp_system(
        self,
        drift_field: np.ndarray | DriftFieldCallable | None = None,
        volatility: float | np.ndarray | DiffusionFieldCallable | None = None,
    ) -> np.ndarray:
        ...
"""

from __future__ import annotations

import functools
import inspect
import types
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterator, Mapping, Sequence


@dataclass(frozen=True)
class Slots:
    """What one kind of user callable may read. A callable is read by the names it declares (#2875), so ``order`` says
    which names there are, and the order the operator family's methods keep them in (`refuse_methods_out_of_order`).
    ``required`` are slots it must declare to be read at all: a raw Hamiltonian reads ``p``, and may read nothing else.

    What the slots beside ``x`` hold is stated, and `BoundCallable.check_on` reads the statement. ``fields`` hold a field
    over the points ``x`` holds, such as the density: a body may read one whole, ``np.max(m)``, so a point read gets it
    whole too. ``per_point`` hold one value per point, in step with ``x``: a point read gets that point's. Any other slot
    is one value for the whole call, such as ``t`` or a measure.
    """

    order: tuple[str, ...]
    required: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()
    per_point: tuple[str, ...] = ()


# What each user callable the library invokes may read: time, then space, then the u-derivative family, then the
# measure, then further parameters (#2375 ruling 8). These are the one statement of it. `DualHamiltonian`,
# `DualLagrangian` and the validators' reading of a family's method call a callable by name with no binding, and read
# `refuse_overridden` (#2869).
POTENTIAL_SLOTS = Slots(("t", "x"))
# `u_terminal` / `m_initial`: read at T and at 0 when they declare time (#2431).
CONDITION_SLOTS = Slots(("t", "x"))
BOUNDARY_VALUE_SLOTS = Slots(("t", "x"))
SOURCE_TERM_SLOTS = Slots(("t", "x", "v", "m"), fields=("v", "m"))
MEASURE_FIELD_SLOTS = Slots(("t", "x", "mu"))
COEFFICIENT_SLOTS = Slots(("t", "x", "m"), fields=("m",))
HAMILTONIAN_SLOTS = Slots(("t", "x", "p", "m"), required=("p",), per_point=("p", "m"))
CONDITIONAL_HAMILTONIAN_SLOTS = Slots(("t", "x", "p", "m", "theta"), per_point=("p", "m"))
CONDITIONAL_TERMINAL_SLOTS = Slots(("x", "theta"))
# Howard's policy update reads the time index, not a time: its own slot (user ruling, 2026-11-04).
ALPHA_STAR_SLOTS = Slots(("t_idx", "x", "p", "m"), per_point=("p", "m"))
# A graph's node plays x; its adjacency stays with it (user ruling 2026-09-25).
NETWORK_HAMILTONIAN_SLOTS = Slots(("t", "node", "neighbors", "p", "m"))
NODE_POTENTIAL_SLOTS = Slots(("t", "node"))
NODE_INTERACTION_SLOTS = Slots(("t", "node", "m"))
NODE_LAGRANGIAN_SLOTS = Slots(("t", "node", "velocity", "m"))
# A graph node's boundary value, and a CUSTOM node's rule, which also reads the field it is applied to (#2296).
NODE_VALUE_SLOTS = Slots(("t", "node"))
NODE_RULE_SLOTS = Slots(("t", "node", "field"))
SOLVER_SOURCE_SLOTS = Slots(("t", "x"))
VARIATIONAL_LAGRANGIAN_SLOTS = Slots(("t", "x", "v", "m"), per_point=("v", "m"))
# A region's signed distance or predicate, and an exact solution's fields, read what they name.
REGION_SLOTS = Slots(("x",))
EXACT_SOLUTION_SLOTS = Slots(("t", "x"))


def coordinate_tuple(points: Any, dimension: int) -> tuple[Any, ...]:
    """``points`` in the form every user callable reads ``x``: one entry per coordinate, each a scalar or an array, with
    the coordinate axis first (user ruling, 2026-11-04). ``dimension`` is what tells ``(N,)`` apart, N points on a line or
    one point in N dimensions, so the door that holds the points states it.
    """
    if isinstance(points, tuple):
        return points
    array = np.asarray(points, dtype=float)
    if dimension == 1:
        return (array[..., 0] if array.ndim >= 2 and array.shape[-1] == 1 else array,)
    if array.ndim == 0 or array.shape[-1] != dimension:
        raise ValueError(f"points of shape {array.shape} hold no coordinate axis of length {dimension}")
    return tuple(np.moveaxis(array, -1, 0))


def point_at(x: tuple[Any, ...], index: int) -> tuple[Any, ...]:
    """Point ``index`` of a batch in coordinate-tuple form, as a tuple of scalars. An entry that is already a scalar holds
    for every point."""
    return tuple(np.ravel(entry)[index] if np.ndim(entry) else entry for entry in x)


# What a refusal advises for a callable that cannot be bound by name. In words, never a lambda: one written from the slots
# passes a slot into a defaulted parameter (#2861).
_WRITE_A_FUNCTION = "a function named for the slots it reads that passes each argument to the parameter that reads it"


def _partial_levels(fn: Any) -> Iterator[functools.partial]:
    """Each ``functools.partial`` in ``fn``, outermost first. Python flattens a partial of a partial only when the inner one
    has no ``__dict__``, so one given an attribute stays a level of its own."""
    while isinstance(fn, functools.partial):
        yield fn
        fn = fn.func


def keeping_what_it_froze(fn: Any, advice: str) -> str:
    """``advice``, unless ``fn`` is a partial that froze by position. Told to name its parameters, ``partial(lambda a, b:
    100*b + a, 9.0)`` becomes ``partial(lambda t, x: 100*t + x, 9.0)``, accepted with the 9.0 moved from x to t (#2869)."""
    if any(level.args for level in _partial_levels(fn)):
        return f"Write {_WRITE_A_FUNCTION}, passing the frozen value by keyword"
    return advice


def frozen_by_keyword(fn: Any, names: Collection[str]) -> dict[str, Any]:
    """Those of ``names`` that a ``functools.partial`` in ``fn`` froze by keyword, with the value each call would see."""
    frozen: dict[str, Any] = {}
    for level in _partial_levels(fn):
        for keyword, value in level.keywords.items():
            if keyword in names:
                frozen.setdefault(keyword, value)
    return frozen


class FrozenSlotError(TypeError):
    """The refusal of a partial the library would override, naming what it froze in ``frozen``."""

    def __init__(self, message: str, frozen: list[str]) -> None:
        super().__init__(message)
        self.frozen = frozen


def refuse_overridden(fn: Any, passes: Collection[str], *, role: str) -> None:
    """Refuse ``fn``, about to be called by name with ``passes``, if that would override what a partial froze (#2869)."""
    frozen = list(frozen_by_keyword(fn, passes))
    if frozen:
        raise FrozenSlotError(
            f"{role} {getattr(fn, '__qualname__', fn)!r} is a functools.partial that freezes {', '.join(frozen)} by keyword: "
            f"called by name, it is passed that slot, so the frozen value would be overridden. Write "
            f"{_WRITE_A_FUNCTION}, passing the frozen value by keyword (#2869).",
            frozen,
        )


def _as_written(fn: Any) -> Any:
    """``fn`` rebuilt from its own code and the partials over it. ``inspect.signature`` follows ``__wrapped__`` and reads
    ``__signature__`` at face value, so a wrapper can show names it does not take (#2871)."""
    if isinstance(fn, functools.partial):
        return functools.partial(_as_written(fn.func), *fn.args, **fn.keywords)
    rebuilt = types.FunctionType(fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)
    rebuilt.__kwdefaults__ = fn.__kwdefaults__
    return rebuilt


def bind_user_callable(fn: Callable[..., Any], slots: Slots, *, role: str) -> BoundCallable:
    """Bind a user-supplied callable to its slots, once, when the library accepts it (#2875).

    The result is invoked with every slot by keyword, ``bound(x=..., t=...)``, and passes the user's function the slots
    it names. It is a function the user writes, a ``def`` or a ``lambda``, or a ``functools.partial`` of one (user ruling,
    2026-10-11): another callable's parameter names were not written against the slots.

    Refused here, rather than failing or computing at the first call: any other callable, a positional-only parameter, a
    ``*args`` or ``**kwargs`` parameter, which nothing reaches, a parameter with no default named for no slot, a function
    that names no slot or leaves out a required one, and a partial that froze a slot by keyword, which the slot passed by
    name would override. A parameter named for no slot with a default keeps its default.
    """
    name = getattr(fn, "__qualname__", fn)
    target = fn
    while isinstance(target, functools.partial):
        target = target.func
    if not inspect.isfunction(target):
        raise TypeError(
            f"{role} {name!r} is of type {type(fn).__name__}, not a plain def or lambda: a {role} is read by the parameter "
            f"names it declares, and a def that a decorator turned into another object (functools.cache, lru_cache, "
            f"numba.njit) counts as another callable. Wrap it in {_WRITE_A_FUNCTION} (#2875)."
        )
    try:
        declared = list(inspect.signature(_as_written(fn)).parameters.values())
    except (TypeError, ValueError) as unreadable:
        raise TypeError(f"{role} {name!r} has no signature that can be read: {unreadable}") from None
    takes = ", ".join(str(p) for p in declared)
    kinds = inspect.Parameter
    if any(p.kind is kinds.POSITIONAL_ONLY for p in declared):
        raise TypeError(
            f"{role} {name!r} takes ({takes}), with positional-only parameters, so it cannot be called by name. Wrap it in "
            f"{_WRITE_A_FUNCTION} (#2875)."
        )
    if any(p.kind in (kinds.VAR_POSITIONAL, kinds.VAR_KEYWORD) for p in declared):
        advice = keeping_what_it_froze(fn, f"Name each one it reads from ({', '.join(slots.order)})")
        raise TypeError(
            f"{role} {name!r} takes ({takes}): called by name, it is passed only the slots it names, so nothing reaches a "
            f"starred parameter. {advice} (#2875)."
        )
    unnamed = [p.name for p in declared if p.default is kinds.empty and p.name not in slots.order]
    if unnamed:
        raise TypeError(
            f"{role} {name!r} takes ({takes}): {', '.join(unnamed)} has no default and names no slot of "
            f"({', '.join(slots.order)}). {keeping_what_it_froze(fn, 'Name each parameter after what it reads')} (#2875)."
        )
    named = tuple(p.name for p in declared if p.name in slots.order)
    missing = [slot for slot in slots.required if slot not in named]
    if missing or not named:
        needs = ", ".join(missing) if missing else f"one of {', '.join(slots.order)}"
        raise TypeError(
            f"{role} {name!r} takes ({takes}), which does not name {needs}. {keeping_what_it_froze(fn, 'Name it')} (#2875)."
        )
    refuse_overridden(fn, named, role=role)
    return BoundCallable(fn, named, role, _defined_by_the_library(fn), slots)


def _out_of_order(names: Sequence[str], slots: Slots) -> bool:
    """Whether a method's positional parameter ``names`` break ``slots``' order: a time parameter anywhere but at the
    ``t`` slot's position, or slot-named parameters out of slot order."""
    ranked = [slots.order.index(name) for name in names if name in slots.order]
    t_at = slots.order.index("t") if "t" in slots.order else None
    return any(name == "t" and i != t_at for i, name in enumerate(names)) or ranked != sorted(ranked)


def refuse_methods_out_of_order(cls: type, table: Mapping[str, Slots | None]) -> None:
    """Refuse, at class creation, a method ``cls`` defines under a name in ``table`` whose
    parameters break that name's slot order (#2375 ruling 8).

    The library calls these methods by keyword, so an out-of-order definition would compute
    correctly from the library and wrongly for anyone calling it positionally. A ``None`` entry
    leaves that name unchecked, for a subclass whose method of the same name is another API.
    """
    for name, slots in table.items():
        if slots is None:  # a subclass whose method of that name has another signature
            continue
        member = cls.__dict__.get(name)
        fn = member.__func__ if isinstance(member, (staticmethod, classmethod)) else member
        if not callable(fn):
            continue
        try:
            params = list(inspect.signature(fn).parameters.values())
        except (TypeError, ValueError):
            continue
        kinds = inspect.Parameter
        names = [p.name for p in params if p.kind in (kinds.POSITIONAL_ONLY, kinds.POSITIONAL_OR_KEYWORD)]
        if not isinstance(member, staticmethod) and names:
            names = names[1:]  # self, or cls
        if _out_of_order(names, slots):
            raise TypeError(
                f"{cls.__qualname__}.{name} takes ({', '.join(names)}), which is out of order. Since #2378 phase 5 "
                f"(#2375 ruling 8) it takes ({', '.join(slots.order)}){', time first' if 't' in slots.order else ''}: "
                f"reorder the parameters, and call the method by keyword."
            )


@dataclass
class SourcePlan:
    """What `evaluate_solver_source` remembers of a callable: never the callable itself."""

    passes: tuple[str, ...]
    checked: bool


# How each source callable is bound -- the rule and the slots, never the callable itself -- keyed by
# the callable's identity, with a weak reference that drops the entry when the callable dies. Not by
# the callable: the couplers build a fresh closure per Picard iteration that captures that
# iteration's arrays, and holding it kept every one alive (#2406 review, round 2). Not by equality
# either: two different callables that compare equal would share one plan (round 3).
_SOURCE_PLANS: dict[tuple[int, Slots], tuple[weakref.ref, SourcePlan]] = {}


def _remembered(fn: Callable[..., Any], slots: Slots, role: str) -> tuple[BoundCallable, SourcePlan]:
    """``fn`` bound to ``slots``, and what is remembered of it: how it binds, and whether its reads have been compared. Per
    callable and not on a bound object, which held here would keep the callable alive, so a loop inspects the signature
    once (#2406 review: +7.5% on a small FP solve) and compares once. Whatever `bind_user_callable` accepts, a function or a
    partial of one, can be weakly referenced."""
    key = (id(fn), slots)
    entry = _SOURCE_PLANS.get(key)
    if entry is None or entry[0]() is not fn:
        bound = bind_user_callable(fn, slots, role=role)
        entry = (weakref.ref(fn, functools.partial(_forget_source, key)), SourcePlan(bound.passes, False))
        _SOURCE_PLANS[key] = entry
    plan = entry[1]
    return BoundCallable(fn, plan.passes, role, plan.checked, slots), plan


def evaluate_solver_source(source_term: Callable[..., Any], *, t: Any, x: Any, dimension: int) -> Any:
    """A solver's ``source_term`` at ``(t, x)``: the one place a solver evaluates the source it was handed (#2875).

    A closure the library composed is called with the solver's own ``x``, since it is not a user input: what it wraps reads
    its own door, and it is not checked, so the fresh one a coupler builds per Picard iteration adds no calls. Anything else
    reads ``x`` as a coordinate tuple.
    """
    if _defined_by_the_library(source_term):
        return source_term(t, x)
    bound, plan = _remembered(source_term, SOLVER_SOURCE_SLOTS, "source_term")
    result = bound(t=t, x=coordinate_tuple(x, dimension))
    plan.checked = bound.checked
    return result


def read_region(region: Callable[..., Any], points: Any, *, role: str = "region") -> Any:
    """A region's signed distance or predicate at ``points`` of shape ``(d,)`` or ``(..., d)``: the one place a door reads
    one (#2875). A function the user wrote reads ``x`` as a coordinate tuple. The library's own region classes are not user
    callables, and take the points as they are."""
    if _defined_by_the_library(region):
        return region(points)
    points = np.asarray(points, dtype=float)
    bound, plan = _remembered(region, REGION_SLOTS, role)
    result = bound(x=coordinate_tuple(points, points.shape[-1]))
    plan.checked = bound.checked
    return result


def _forget_source(key: tuple[int, Slots], dead: weakref.ref) -> None:
    entry = _SOURCE_PLANS.get(key)
    if entry is not None and entry[0] is dead:
        del _SOURCE_PLANS[key]


def _defined_by_the_library(fn: Any) -> bool:
    """Whether ``fn`` is a closure the library composed. It is not a user input: what it wraps has its own door, and the
    couplers build one per Picard iteration (#2875)."""
    while isinstance(fn, functools.partial):
        fn = fn.func
    return (getattr(fn, "__module__", None) or "").partition(".")[0] == "mfgarchon"


# What a body raises when it cannot read what it is handed: float() of an array (TypeError), the truth of an array or a
# refused shape (ValueError), indexing past a point's components (IndexError). Anything else propagates as it is.
_UNREADABLE = (TypeError, ValueError, IndexError, ArithmeticError)


def _refuse_unless_equal(together: NDArray, alone: NDArray, described: str, guidance: str) -> None:
    """NaN only matches NaN, and the tolerance is scaled by the finite values alone, so one infinite value cannot make the
    comparison vacuous (#2479)."""
    finite = np.abs(alone[np.isfinite(alone)])
    scale = max(1.0, float(finite.max())) if finite.size else 1.0
    if together.size != alone.size or not np.allclose(
        together.reshape(alone.shape), alone, rtol=1e-10, atol=1e-12 * scale, equal_nan=True
    ):
        raise ValueError(
            f"{described}, reads differently together ({together.ravel()}) and one at a time ({alone.ravel()}). {guidance}"
        )


def refuse_batch_disagreement(
    read: Callable[..., Any], batch: Mapping[str, Any], points: Sequence[Mapping[str, Any]], *, name: str, guidance: str
) -> None:
    """Refuse a body whose read of a batch differs from its read of each point alone, or that raises on either (#2875).

    The library chooses when to loop and when to vectorize, so a body must read the same both ways: one that does not
    comes back as a result and not as an error.
    """
    described = f"{name}, at {len(points)} samples"
    try:
        together = np.asarray(read(**batch), dtype=float)
    except _UNREADABLE as e:
        raise ValueError(f"{described} together, raised {type(e).__name__}: {e}. {guidance}") from e
    try:
        alone = np.vstack([np.asarray(read(**point), dtype=float).reshape(1, -1) for point in points])
    except _UNREADABLE as e:
        raise ValueError(f"{described} one at a time, raised {type(e).__name__}: {e}. {guidance}") from e
    if together.size == 1:  # a constant reads the same at every sample
        together = np.full((len(points), 1), float(together.reshape(-1)[0]))
    _refuse_unless_equal(together, alone, described, guidance)


def _sample_count(dimension: int) -> int:
    """How many points to compare: never one, which reads the same in either form, and never ``d``, where a coordinate
    axis passes as a batch."""
    return 4 if dimension == 3 else 3


def spread_samples(count: int, samples: int) -> NDArray:
    """Which of ``count`` points to compare. Not the ends and the middle: on a uniform grid a term such as sin(2 pi x) is
    the same at all three, so a body that misread x would read the same both ways. The golden ratio's multiples fall on
    no such symmetry."""
    return np.floor(np.mod((np.arange(samples) + 1) * 0.6180339887498949, 1.0) * count).astype(int)


def _entry_at(result: NDArray, batch_shape: tuple[int, ...], own: tuple[int, ...], index: tuple[int, ...]) -> NDArray | None:
    """A read's value at ``index``, by the form it returned: a scalar holds at every point, a result with the batch's shape
    leading holds its entry there, and a result of one point's own shape ``own`` holds as it is. Anything else has no entry
    to compare, which is None."""
    if result.ndim == 0 or result.size == 1:
        return result.reshape(-1)
    if result.shape[: len(batch_shape)] == batch_shape:
        return np.ravel(result[index])
    if result.shape == own:
        return np.ravel(result)
    return None


def _compare_reads(read: Callable[..., Any], values: Mapping[str, Any], slots: Slots, name: str, guidance: str) -> None:
    """Compare a body's read of the batch ``values`` holds with its read of a few of those points, spread across it (#2875).

    A point read gets one point's ``x`` and each ``per_point`` slot's value there, and every other slot as the batch read
    gets it: so a body may read a field whole. A batch too small to compare, of fewer than two points or of exactly ``d``,
    is cycled to a count that is neither, ``x`` and the slots stated to run with it together.

    A return that does not lead with the batch's shape is one fixed array only if every point's read returns that array
    too: a constant noise matrix does, and a body that read a coordinate axis as points does not. The door that reads it
    judges its shape, since only that door knows what a constant means there.
    """
    entries = [np.array(entry, dtype=float) for entry in np.broadcast_arrays(*values["x"])]  # a body may write into x
    shape = entries[0].shape
    d = len(entries)
    samples = _sample_count(d)
    count = int(np.prod(shape))
    batch = dict(values)
    if count < 2 or count == d:
        cycled = np.resize(np.arange(count), samples)
        entries = [np.ravel(entry)[cycled] for entry in entries]
        for slot in (*slots.fields, *slots.per_point):
            if np.ndim(batch.get(slot)) > 0:
                batch[slot] = np.asarray(batch[slot])[cycled] if slot in slots.per_point else np.ravel(batch[slot])[cycled]
        shape = entries[0].shape
        count = samples
    described = f"{name}, on a batch of {count} points"
    try:
        together = np.asarray(read(**{**batch, "x": tuple(entry.copy() for entry in entries)}), dtype=float)
    except _UNREADABLE as e:
        raise ValueError(f"{described}, raised {type(e).__name__}: {e}. {guidance}") from e
    if together.ndim == len(shape) + 1 and together.shape == (d, *shape):
        together = np.moveaxis(together, 0, -1)  # per coordinate, the coordinate axis first, as FP-FDM's drift returns it
    fixed = together.ndim > 0 and together.size != 1 and together.shape[: len(shape)] != shape
    rows_together, rows_alone = [], []
    for flat in spread_samples(count, samples):
        index = tuple(int(i) for i in np.unravel_index(int(flat), shape))
        point = dict(batch)
        point["x"] = tuple(entry[index] for entry in entries)
        for slot in slots.per_point:
            if np.ndim(batch.get(slot)) > 0:
                point[slot] = np.asarray(batch[slot])[index]
        try:
            alone = np.asarray(read(**point), dtype=float)
        except _UNREADABLE as e:
            raise ValueError(f"{described}, one point at a time, raised {type(e).__name__}: {e}. {guidance}") from e
        if alone.ndim == len(shape) + 1 and alone.shape == (d, *shape):
            alone = np.moveaxis(alone, 0, -1)  # as a batch read may return it
        if fixed and alone.shape != together.shape:
            raise ValueError(f"{described}: a return of shape {together.shape} has no entry per point. {guidance}")
        own = together.shape if fixed else together.shape[len(shape) :]
        entry = np.ravel(alone) if fixed else _entry_at(alone, shape, own, index)
        if entry is None:
            raise ValueError(f"{described}: one point's read returned shape {alone.shape}, which has no entry to compare with a batch's {together.shape}. {guidance}")
        rows_alone.append(entry)
        rows_together.append(np.ravel(together) if fixed or together.size == 1 else np.ravel(together[index]))
    width = max(row.size for row in (*rows_together, *rows_alone))
    _refuse_unless_equal(
        np.vstack([np.broadcast_to(row, (width,)) for row in rows_together]),
        np.vstack([np.broadcast_to(row, (width,)) for row in rows_alone]),
        described,
        guidance,
    )


class BoundCallable:
    """A user callable and the slots it reads, by name, and whether the library has yet compared its batch and point reads.
    A class, not a closure, so an object holding one still pickles."""

    __slots__ = ("checked", "fn", "passes", "role", "slots")

    def __init__(
        self,
        fn: Callable[..., Any],
        passes: tuple[str, ...],
        role: str = "callable",
        checked: bool = False,
        slots: Slots | None = None,
    ):
        self.fn = fn
        self.passes = passes
        self.role = role
        self.checked = checked
        self.slots = Slots(passes) if slots is None else slots

    def _read(self, **values: Any) -> Any:
        return self.fn(**{slot: values[slot] for slot in self.passes})

    def check_on(self, values: Mapping[str, Any]) -> None:
        """Compare this body's batch and point reads on ``values``, once. A door with samples when it accepts the callable
        calls this there; any other is checked at its first batch, in `__call__`, on what that batch holds. A door that
        hands one point has no batch read to compare, and this waits for one that does."""
        if self.checked or not np.broadcast_shapes(*(np.shape(entry) for entry in values["x"])):
            return
        self.checked = True
        guidance = (
            "x is a tuple with one entry per coordinate, each a scalar or an array, and the library may hand either: "
            "write the body to read both the same way, such as x[0] for a coordinate and sum(xi**2 for xi in x) for a "
            "squared norm (#2875)."
        )
        name = f"{self.role} {getattr(self.fn, '__qualname__', self.fn)!r}"
        _compare_reads(self._read, values, self.slots, name, guidance)

    def __call__(self, **values: Any) -> Any:
        if "x" in values:
            if not isinstance(values["x"], tuple):
                raise TypeError(
                    f"the library handed {getattr(self.fn, '__qualname__', self.fn)!r} x as "
                    f"{type(values['x']).__name__}, and x is a coordinate tuple: this is a defect in the door that "
                    f"called it (#2875)."
                )
            if not self.checked:
                self.check_on(values)
        return self._read(**values)


def bound_attribute(owner: Any, attr: str, slots: Slots, *, role: str) -> BoundCallable:
    """``getattr(owner, attr)`` bound to ``slots``, cached on ``owner`` and keyed on the callable.

    Keyed on identity rather than bound once at construction, because the attribute can be
    replaced afterwards (``MFGProblem`` composes a soft wall into a copy's ``_potential``), and a
    binding cached earlier would silently keep evaluating the old callable.
    """
    fn = getattr(owner, attr)
    key = f"_{attr.lstrip('_')}_binding"
    cached = owner.__dict__.get(key)
    if cached is None or cached.fn is not fn:
        cached = bind_user_callable(fn, slots, role=role)
        setattr(owner, key, cached)
    return cached


@runtime_checkable
class DriftFieldCallable(Protocol):
    """
    Protocol for callable drift field α(t, x, m).

    A drift field specifies the velocity field for Fokker-Planck evolution:
        ∂m/∂t + ∇·(α m) = (σ²/2) Δm

    Signature:
        α(t, x, m) -> drift vector

    Args:
        t: Current time (float)
        x: Spatial position (ndarray of shape (d,) or (N, d))
        m: Density field (ndarray)

    Returns:
        Drift vector (ndarray of shape (d,) or (N, d))

    Examples:
        >>> # Constant wind
        >>> def wind_drift(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        ...     return np.array([1.0, 0.5])

        >>> # State-dependent drift
        >>> def density_drift(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        ...     return -grad(m) / (m + 1e-10)  # Diffusion approximation

        >>> # Optimal control drift (MFG)
        >>> def control_drift(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        ...     return -grad_U / sigma_sq
    """

    def __call__(
        self,
        t: float,
        x: NDArray[np.floating],
        m: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Evaluate drift field at (t, x, m)."""
        ...


@runtime_checkable
class DiffusionFieldCallable(Protocol):
    """
    Protocol for callable diffusion field σ(t, x, m).

    A diffusion field specifies spatially/temporally/state-varying diffusion coefficient, in Itô form (#2841):
        ∂m/∂t + ∇·(α m) = ∂ᵢ∂ⱼ(Dᵢⱼ m),   Δ((1/2) σ(t,x,m)² m) for an isotropic σ

    Signature:
        σ(t, x, m) -> diffusion coefficient(s)

    Args:
        t: Current time (float)
        x: Spatial position (ndarray of shape (d,) or (N, d))
        m: Density field (ndarray)

    Returns:
        - Scalar: an isotropic volatility σ (the diffusion is σ²/2)
        - Array of the density's shape: a spatially varying isotropic volatility σ(x); any other shape is refused
        - With volatility_kind="tensor", the callable is read per point and returns the (d, d) volatility matrix
          Σ(t, x, m), whose diffusion is ΣΣᵀ/2

    Examples:
        >>> # Spatially varying diffusion
        >>> def varying_diffusion(t: float, x: np.ndarray, m: np.ndarray) -> float:
        ...     return 0.1 + 0.05 * np.linalg.norm(x)

        >>> # Density-dependent diffusion
        >>> def density_diffusion(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        ...     return 0.1 * (1 + m)  # Higher diffusion in dense regions

        >>> # Anisotropic diffusion (diagonal)
        >>> def anisotropic_diffusion(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        ...     return np.array([0.1, 0.05])  # Different diffusion in x and y
    """

    def __call__(
        self,
        t: float,
        x: NDArray[np.floating],
        m: NDArray[np.floating],
    ) -> float | NDArray[np.floating]:
        """Evaluate diffusion field at (t, x, m)."""
        ...


@runtime_checkable
class HamiltonianCallable(Protocol):
    """
    Protocol for the per-point Hamiltonian API, H(t, x, p, m) at one grid point.

    The Hamiltonian appears in the HJB equation:
        -∂u/∂t + H(x, m, ∇u, t) - (σ²/2) Δu = 0

    Signature:
        H(x_idx, m_at_x, derivs, **kwargs) -> float

    Args:
        x_idx: Grid index (int)
        m_at_x: Density at x (float)
        derivs: Derivatives in tuple notation:
                - 1D: {(0,): u, (1,): ∂u/∂x}
                - 2D: {(0,0): u, (1,0): ∂u/∂x, (0,1): ∂u/∂y}
        t_idx: Time index (optional, int)
        x_position: Actual spatial coordinate (optional, ndarray)
        current_time: Actual time value (optional, float)
        problem: Problem instance (optional)

    Returns:
        Hamiltonian value H(t, x, p, m)

    Examples:
        >>> # Quadratic Hamiltonian (LQ control)
        >>> def lq_hamiltonian(x_idx, m_at_x, derivs, **kwargs):
        ...     p = derivs[(1,)]
        ...     return 0.5 * p**2 - V[x_idx] - m_at_x**2

        >>> # Non-quadratic Hamiltonian
        >>> def custom_hamiltonian(x_idx, m_at_x, derivs, **kwargs):
        ...     p = derivs[(1,)]
        ...     return np.sqrt(1 + p**2) - np.log(1 + m_at_x)
    """

    def __call__(
        self,
        x_idx: int,
        m_at_x: float,
        derivs: dict[tuple[int, ...], float],
        t_idx: int | None = None,
        x_position: NDArray[np.floating] | None = None,
        current_time: float | None = None,
        problem: object | None = None,
    ) -> float:
        """Evaluate the Hamiltonian at one grid point."""
        ...


@runtime_checkable
class HamiltonianDerivativeCallable(Protocol):
    """
    Protocol for the per-point Hamiltonian derivative dH/dm, at one grid point.

    The derivative dH/dm appears in the Fokker-Planck coupling term:
        ∂m/∂t + ∇·(α m) = (σ²/2) Δm - ∇·(m ∇(dH/dm))

    Signature:
        dH/dm(x_idx, m_at_x, derivs, **kwargs) -> float

    Args:
        (Same as HamiltonianCallable)

    Returns:
        Derivative dH/dm at (t, x, p, m)

    Examples:
        >>> # For H = 0.5 p² - V - m²
        >>> def hamiltonian_dm(x_idx, m_at_x, derivs, **kwargs):
        ...     return -2 * m_at_x

        >>> # For H = 0.5 p² - V - log(m)
        >>> def hamiltonian_dm(x_idx, m_at_x, derivs, **kwargs):
        ...     return -1 / (m_at_x + 1e-10)
    """

    def __call__(
        self,
        x_idx: int,
        m_at_x: float,
        derivs: dict[tuple[int, ...], float],
        t_idx: int | None = None,
        x_position: NDArray[np.floating] | None = None,
        current_time: float | None = None,
        problem: object | None = None,
    ) -> float:
        """Evaluate dH/dm at one grid point."""
        ...


@runtime_checkable
class PotentialCallable(Protocol):
    """
    Protocol for potential function V(t, x) (#2375 ruling 8: time first).

    The potential appears in the running cost:
        Running cost = ∫ V(t, x) m(t, x) dx

    Signature:
        V(t, x) -> float

    Args:
        t: Time (float)
        x: Spatial position (float for 1D, ndarray for nD)

    Returns:
        Potential value V(t, x)

    Examples:
        >>> # Time-independent potential
        >>> def quadratic_potential(x: float) -> float:
        ...     return 0.5 * x**2

        >>> # Time-dependent potential
        >>> def moving_well(t: float, x: float) -> float:
        ...     center = np.sin(t)
        ...     return 0.5 * (x - center)**2
    """

    def __call__(self, t: float, x: float | NDArray[np.floating]) -> float:
        """Evaluate potential at (t, x)."""
        ...


@runtime_checkable
class InitialDensityCallable(Protocol):
    """
    Protocol for initial density function m₀(x).

    Signature:
        m₀(x) -> density

    Args:
        x: Spatial position (float for 1D, ndarray for nD)

    Returns:
        Initial density m₀(x) ≥ 0

    Examples:
        >>> # Gaussian density
        >>> def gaussian_initial(x: float) -> float:
        ...     return np.exp(-100 * (x - 0.5)**2)

        >>> # Multi-modal density
        >>> def bimodal_initial(x: float) -> float:
        ...     return np.exp(-200 * (x - 0.2)**2) + np.exp(-200 * (x - 0.8)**2)
    """

    def __call__(self, x: float | NDArray[np.floating]) -> float:
        """Evaluate initial density at x."""
        ...


@runtime_checkable
class FinalValueCallable(Protocol):
    """
    Protocol for terminal value function u_T(x).

    The terminal condition for the HJB equation:
        u(T, x) = u_T(x)

    Signature:
        u_T(x) -> value

    Args:
        x: Spatial position (float for 1D, ndarray for nD)

    Returns:
        Terminal value u_T(x)

    Examples:
        >>> # Quadratic terminal cost
        >>> def quadratic_terminal(x: float) -> float:
        ...     return 0.5 * (x - 0.5)**2

        >>> # Indicator terminal cost
        >>> def indicator_terminal(x: float) -> float:
        ...     return 0 if 0.4 <= x <= 0.6 else 1000
    """

    def __call__(self, x: float | NDArray[np.floating]) -> float:
        """Evaluate terminal value at x."""
        ...


# Type aliases for convenience
CoefficientFieldType = float | NDArray[np.floating] | DriftFieldCallable | DiffusionFieldCallable | None
"""Type alias for coefficient fields (scalar | array | callable | None)."""

"""
Type protocols for callable signatures in MFGarchon.

This module defines Protocol classes for callable objects used throughout the framework,
enabling precise type checking and better IDE support.

Usage:
    from mfgarchon.types.callable_protocols import DriftFieldCallable, DiffusionFieldCallable

    def solve_fp_system(
        self,
        drift_field: np.ndarray | DriftFieldCallable | None = None,
        volatility_field: float | np.ndarray | DiffusionFieldCallable | None = None,
    ) -> np.ndarray:
        ...
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


@dataclass(frozen=True)
class Slots:
    """The argument order of one kind of user callable, and what binding it may assume.

    ``aliases`` are other names a parameter may carry for a slot: ``time`` for ``t`` always, plus
    names the library documented before #2378 phase 5. ``unmoved`` are slots at the same position
    in the order before #2375 ruling 8 as after it (``v`` for a source term, ``p`` for a
    Hamiltonian): a parameter named after one cannot tell the two orders apart, so it does not
    identify the order for positional binding. ``swapped`` is the pair of slots whose relative order
    ruling 8 reversed (``v``/``m``, ``p``/``m``): a name for ``t`` or ``x`` separates the new order
    from the full old one but not from the half-migration that moved only ``t`` to the front, so a
    positional callable of such a role must also name one of the pair. ``required`` are slots a callable must declare to
    be bound by name (a raw Hamiltonian takes all four). ``optional`` are slots a positional
    callable may leave out (a conditional Hamiltonian's ``t``, as it always could); what remains
    must be in the same order before ruling 8 as after it, so a positional callable without them
    needs no name to say which order it is in.
    """

    order: tuple[str, ...]
    aliases: Mapping[str, str] = field(default_factory=dict)
    unmoved: tuple[str, ...] = ()
    swapped: tuple[str, str] | None = None
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()

    def slot_of(self) -> dict[str, str]:
        """Parameter name -> slot, for every name that identifies a slot."""
        return {name: name for name in self.order} | ({"time": "t"} if "t" in self.order else {}) | dict(self.aliases)


# The argument order of each user-supplied callable the library invokes: time, then space, then the
# u-derivative family, then the measure, then further parameters (#2375 ruling 8, #2378 phase 5).
# These are the one statement of it: every invocation goes through `bind_user_callable`.
POTENTIAL_SLOTS = Slots(("t", "x"))
SOURCE_TERM_SLOTS = Slots(("t", "x", "v", "m"), aliases={"m_t": "m", "v_t": "v"}, unmoved=("v",), swapped=("v", "m"))
MEASURE_FIELD_SLOTS = Slots(("t", "x", "mu"))
HAMILTONIAN_SLOTS = Slots(("t", "x", "p", "m"), unmoved=("p",), swapped=("p", "m"), required=("t", "x", "p", "m"))
CONDITIONAL_HAMILTONIAN_SLOTS = Slots(("t", "x", "p", "m", "theta"), optional=("t",))
ALPHA_STAR_SLOTS = Slots(("t", "x", "p", "m"), aliases={"t_idx": "t"})
# A graph's node plays x; its adjacency stays with it (user ruling 2026-09-25).
NETWORK_HAMILTONIAN_SLOTS = Slots(("t", "node", "neighbors", "p", "m"), unmoved=("p",), swapped=("p", "m"))
NODE_POTENTIAL_SLOTS = Slots(("t", "node"))
NODE_INTERACTION_SLOTS = Slots(("t", "node", "m"))
NODE_LAGRANGIAN_SLOTS = Slots(("t", "node", "velocity", "m"))

_TIME_NAMES = ("t", "time")


def bind_user_callable(
    fn: Callable[..., Any],
    slots: Slots,
    *,
    role: str,
    spatial_only: bool = False,
) -> BoundCallable:
    """Bind a user-supplied callable to its slots, once, when the library accepts it.

    The result is invoked with every slot by keyword, ``bound(x=..., t=...)``, and passes the
    user's function what it declares:

    - **By name**, when every required positional parameter is named after a slot, or ``time``
      for ``t``, or an alias the role documents (``m_t`` and ``v_t`` for a source term). Slots it
      does not declare are omitted, so a time-independent ``V(x)`` is accepted where ``t`` is. Its
      declared order then does not matter to the numbers.
    - **By position, in slot order**, when it requires exactly one parameter per slot, none of them
      is named after a different slot, at least one is named after a slot that moved in ruling 8's
      reorder, and, where the reorder swapped a pair, one of the pair is named: otherwise the list
      reads the same in the new order as in the old one, or as in the half-migration that moved
      only ``t``. Not ``*args`` either, for the same reason.
    - **``x`` alone**, where ``spatial_only`` allows a spatial callable: at most one required
      parameter, or only ``*args``, as ``MFGComponents`` has always called a ``potential_func``
      that declares no time.

    Refused here, rather than failing or computing at the first call:

    - **The order before #2378 phase 5**: a time parameter (``t`` or ``time``) anywhere but at the
      ``t`` slot's position, first in every order ruling 8 sets, or slot-named parameters declared
      out of slot order. Such a callable would compute correctly
      when bound by name, but not when its author calls it positionally, and a positional one with
      a time parameter would receive ``x`` as time.
    - A parameter named after one slot at another slot's position: bound positionally, it would
      receive the wrong one (``V(x, tau)`` would get ``t`` as ``x``).
    - Anything else that cannot be matched to the slots.
    """
    order_ = slots.order
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return BoundCallable(fn, "positional", order_)
    kinds = inspect.Parameter
    slot_of = slots.slot_of()
    positional = [p for p in params if p.kind in (kinds.POSITIONAL_ONLY, kinds.POSITIONAL_OR_KEYWORD)]
    order = [p.name for p in positional]
    if out_of_order(order, slots):
        raise TypeError(
            f"{role} {getattr(fn, '__qualname__', fn)!r} takes ({', '.join(order)}), which is out of order. Since "
            f"#2378 phase 5 (#2375 ruling 8) a {role} takes ({', '.join(order_)}), time first: reorder its "
            f"parameters, and call it by keyword wherever you call it yourself."
        )
    named = [
        p.name for p in params if p.name in slot_of and p.kind in (kinds.POSITIONAL_OR_KEYWORD, kinds.KEYWORD_ONLY)
    ]
    required = [p for p in positional if p.default is kinds.empty]
    declares_required = {slot_of[name] for name in named} >= set(slots.required)
    if named and all(p.name in named for p in required) and declares_required:
        return BoundCallable(fn, "keyword", tuple(slot_of[name] for name in named), tuple(named))
    var_positional = any(p.kind is kinds.VAR_POSITIONAL for p in params)
    # Positionally it receives every slot, or every slot but the optional ones.
    shortened = tuple(slot for slot in order_ if slot not in slots.optional)
    passes = order_ if len(required) == len(order_) else shortened if len(required) == len(shortened) else None
    misplaced = []
    if passes is not None:
        misplaced = [
            name for i, name in enumerate(order[: len(passes)]) if name in slot_of and passes.index(slot_of[name]) != i
        ]
    # Without its optional slots a callable is in the same order either way; with all of them, a
    # name must say which order it is in.
    names_moved = any(name in slot_of and slot_of[name] not in slots.unmoved for name in order)
    names_pair = slots.swapped is None or any(slot_of.get(name) in slots.swapped for name in order)
    identified = passes != order_ or (names_moved and names_pair)
    if passes is not None and not var_positional and not misplaced and identified:
        return BoundCallable(fn, "positional", passes)
    if spatial_only and len(required) <= 1 and (positional or var_positional):
        return BoundCallable(fn, "spatial", ("x",))
    if misplaced:
        why = f": {', '.join(misplaced)} would receive another slot's value. Name its parameters {', '.join(order_)}."
    elif passes is not None and not var_positional:
        why = (
            f": none of its parameters is named after a slot whose position #2375 ruling 8 changed, so they "
            f"read the same in the old order and the new. Name them {', '.join(order_)}."
        )
    else:
        why = f". Name its parameters {', '.join(order_)}, or take exactly {len(order_)} positionally in that order."
    raise TypeError(
        f"{role} {getattr(fn, '__qualname__', fn)!r} takes ({', '.join(p.name for p in params)}), which cannot be "
        f"matched to ({', '.join(order_)}){why}"
    )


def out_of_order(names: Sequence[str], slots: Slots) -> bool:
    """Whether positional parameter ``names`` break ``slots``' order: a time parameter anywhere but
    at the ``t`` slot's position, or slot-named parameters out of slot order. The one statement of
    the rule, for a callable at acceptance and for a method at class creation."""
    slot_of = slots.slot_of()
    ranked = [slots.order.index(slot_of[name]) for name in names if name in slot_of]
    t_at = slots.order.index("t") if "t" in slots.order else None
    return any(name in _TIME_NAMES and i != t_at for i, name in enumerate(names)) or ranked != sorted(ranked)


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
        if out_of_order(names, slots):
            raise TypeError(
                f"{cls.__qualname__}.{name} takes ({', '.join(names)}), which is out of order. Since #2378 phase 5 "
                f"(#2375 ruling 8) it takes ({', '.join(slots.order)}), time first: reorder the parameters, and "
                f"call the method by keyword."
            )


class BoundCallable:
    """A user callable and how it receives its slots: ``binding`` is the rule that matched,
    ``passes`` the slots it receives in the order it receives them, and ``params`` the parameter
    names they go to when bound by name. A class, not a closure, so an object holding one still
    pickles."""

    __slots__ = ("binding", "fn", "params", "passes")

    def __init__(self, fn: Callable[..., Any], binding: str, passes: tuple[str, ...], params: tuple[str, ...] = ()):
        self.fn = fn
        self.binding = binding
        self.passes = passes
        self.params = params

    def __call__(self, **values: Any) -> Any:
        if self.binding == "keyword":
            return self.fn(**{param: values[slot] for param, slot in zip(self.params, self.passes, strict=True)})
        return self.fn(*(values[slot] for slot in self.passes))


def bound_attribute(owner: Any, attr: str, slots: Slots, *, role: str, spatial_only: bool = False) -> BoundCallable:
    """``getattr(owner, attr)`` bound to ``slots``, cached on ``owner`` and keyed on the callable.

    Keyed on identity rather than bound once at construction, because the attribute can be
    replaced afterwards (``MFGProblem`` composes a soft wall into a copy's ``_potential``), and a
    binding cached earlier would silently keep evaluating the old callable.
    """
    fn = getattr(owner, attr)
    key = f"_{attr.lstrip('_')}_binding"
    cached = owner.__dict__.get(key)
    if cached is None or cached.fn is not fn:
        cached = bind_user_callable(fn, slots, role=role, spatial_only=spatial_only)
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

    A diffusion field specifies spatially/temporally/state-varying diffusion coefficient:
        ∂m/∂t + ∇·(α m) = (1/2) ∇·(σ(t,x,m)² ∇m)

    Signature:
        σ(t, x, m) -> diffusion coefficient(s)

    Args:
        t: Current time (float)
        x: Spatial position (ndarray of shape (d,) or (N, d))
        m: Density field (ndarray)

    Returns:
        - Scalar: Isotropic diffusion σ²
        - Array (N,): Spatially varying isotropic diffusion σ²(x)
        - Array (N, d): Diagonal tensor diffusion [σ_x², σ_y², ...]
        - Array (N, d, d): Full tensor diffusion Σ(x)

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

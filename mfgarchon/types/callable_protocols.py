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
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Callable

# The argument order of each user-supplied callable the library invokes (#2375 ruling 8). These
# tuples are the one statement of it: every invocation goes through `bind_user_callable`.
POTENTIAL_SLOTS: tuple[str, ...] = ("x", "t")
SOURCE_TERM_SLOTS: tuple[str, ...] = ("x", "m", "v", "t")
MEASURE_FIELD_SLOTS: tuple[str, ...] = ("x", "mu", "t")


def bind_user_callable(
    fn: Callable[..., Any],
    slots: tuple[str, ...],
    *,
    role: str,
    spatial_only: bool = False,
) -> BoundCallable:
    """Bind a user-supplied callable to its slots, once, when the library accepts it.

    The result is invoked with every slot by keyword, ``bound(x=..., t=...)``, and passes the
    user's function what it declares:

    - **By name**, when every required positional parameter is named after a slot. Slots it
      does not declare are omitted, so a time-independent ``V(x)`` is accepted where ``t`` is.
      Its declared order then does not matter to the numbers.
    - **By position, in slot order**, otherwise, when it can take every slot positionally.
    - **``x`` alone**, where ``spatial_only`` allows a one-argument spatial callable.

    Anything else cannot be told apart and is refused here rather than failing, or computing,
    at the first call.
    """
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return BoundCallable(fn, "positional", slots)
    kinds = inspect.Parameter
    positional = [p for p in params if p.kind in (kinds.POSITIONAL_ONLY, kinds.POSITIONAL_OR_KEYWORD)]
    named = [p.name for p in params if p.name in slots and p.kind in (kinds.POSITIONAL_OR_KEYWORD, kinds.KEYWORD_ONLY)]
    required = [p for p in positional if p.default is kinds.empty]
    if named and all(p.name in named for p in required):
        return BoundCallable(fn, "keyword", tuple(named))
    capacity = len(slots) if any(p.kind is kinds.VAR_POSITIONAL for p in params) else len(positional)
    if len(required) <= len(slots) <= capacity:
        return BoundCallable(fn, "positional", slots)
    if spatial_only and len(required) <= 1 <= capacity:
        return BoundCallable(fn, "spatial", ("x",))
    raise TypeError(
        f"{role} {getattr(fn, '__qualname__', fn)!r} takes ({', '.join(p.name for p in params)}), which cannot be "
        f"matched to ({', '.join(slots)}): name its parameters {', '.join(slots)}, or take all {len(slots)} "
        f"positionally in that order."
    )


class BoundCallable:
    """A user callable and how it receives its slots: ``binding`` is the rule that matched and
    ``passes`` the slots it receives, in the order it receives them. A class, not a closure, so an
    object holding one still pickles."""

    __slots__ = ("binding", "fn", "passes")

    def __init__(self, fn: Callable[..., Any], binding: str, passes: tuple[str, ...]):
        self.fn = fn
        self.binding = binding
        self.passes = passes

    def __call__(self, **values: Any) -> Any:
        if self.binding == "keyword":
            return self.fn(**{name: values[name] for name in self.passes})
        return self.fn(*(values[name] for name in self.passes))


def bound_attribute(
    owner: Any, attr: str, slots: tuple[str, ...], *, role: str, spatial_only: bool = False
) -> BoundCallable:
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
    Protocol for custom Hamiltonian function H(x, m, p, t).

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
        Hamiltonian value H(x, m, p, t)

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
        """Evaluate Hamiltonian at (x, m, p, t)."""
        ...


@runtime_checkable
class HamiltonianDerivativeCallable(Protocol):
    """
    Protocol for Hamiltonian derivative dH/dm(x, m, p, t).

    The derivative dH/dm appears in the Fokker-Planck coupling term:
        ∂m/∂t + ∇·(α m) = (σ²/2) Δm - ∇·(m ∇(dH/dm))

    Signature:
        dH/dm(x_idx, m_at_x, derivs, **kwargs) -> float

    Args:
        (Same as HamiltonianCallable)

    Returns:
        Derivative dH/dm at (x, m, p, t)

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
        """Evaluate dH/dm at (x, m, p, t)."""
        ...


@runtime_checkable
class PotentialCallable(Protocol):
    """
    Protocol for potential function V(x, t).

    The potential appears in the running cost:
        Running cost = ∫ V(x, t) m(t, x) dx

    Signature:
        V(x, t=None) -> float

    Args:
        x: Spatial position (float for 1D, ndarray for nD)
        t: Time (optional, float)

    Returns:
        Potential value V(x, t)

    Examples:
        >>> # Time-independent potential
        >>> def quadratic_potential(x: float) -> float:
        ...     return 0.5 * x**2

        >>> # Time-dependent potential
        >>> def moving_well(x: float, t: float) -> float:
        ...     center = np.sin(t)
        ...     return 0.5 * (x - center)**2
    """

    def __call__(self, x: float | NDArray[np.floating], t: float | None = None) -> float:
        """Evaluate potential at (x, t)."""
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

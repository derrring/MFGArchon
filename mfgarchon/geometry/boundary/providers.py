"""
Dynamic boundary condition value providers (Issue #625).

This module implements the callback-provider pattern for state-dependent
boundary conditions. Instead of baking coupling logic into solvers, the
BC *intent* is stored in the BCSegment and resolved to concrete values
by the coupling iterator.

Architecture:
------------
1. BCValueProvider protocol defines the compute(state) -> value contract
2. Concrete providers implement the formula (``ConstantProvider`` is the reference one)
3. BCSegment.value can hold a provider (intent) or static value
4. FixedPointIterator resolves providers before passing BC to solvers

Benefits:
---------
- Solvers remain generic (no MFG coupling knowledge)
- New dynamic BCs = new provider class (no solver modification)
- Intent is explicit in BC object, not hidden in string flags
- Providers are unit-testable in isolation

Example:
--------
    >>> from mfgarchon.geometry.boundary.providers import BaseBCValueProvider
    >>> from mfgarchon.geometry.boundary import BCSegment, BCType
    >>>
    >>> class LeftDensity(BaseBCValueProvider):
    ...     def compute(self, state):
    ...         # m_current is space-time, time first: (Nt+1, *grid). Final slice, left wall.
    ...         return float(state["m_current"][-1, 0])
    >>>
    >>> # Store intent in BCSegment
    >>> segment = BCSegment(
    ...     name="left",
    ...     bc_type=BCType.NEUMANN,
    ...     value=LeftDensity(),
    ...     boundary="x_min",
    ... )
    >>>
    >>> # Later, iterator resolves provider with current state
    >>> state = {'m_current': m, 'geometry': geometry}
    >>> concrete_value = segment.value.compute(state)

References:
-----------
- Issue #625: Dynamic BC value provider architecture
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.geometry.protocol import GeometryProtocol


# =============================================================================
# Type Definitions
# =============================================================================


class BCProviderState(TypedDict, total=False):
    """
    Typed dictionary for BC provider state.

    This defines the standard keys passed to BCValueProvider.compute().
    All keys are optional (total=False) as different providers need different subsets.

    Keys:
        m_current: The previous Picard iterate of the FP density, space-time and time first:
            shape ``(Nt+1, *grid)``.
        U_current: The previous iterate of the value function, same shape.
        geometry: Problem geometry object
        volatility: The SDE volatility σ as supplied, never D = σ²/2 (#1512, #2378). A provider that
            needs a scalar calls ``scalar_volatility`` on it, which refuses a field (#2376).
        t: Current time (for time-dependent problems)
        iteration: Current Picard iteration number
    """

    m_current: NDArray[np.floating]
    U_current: NDArray[np.floating]
    geometry: GeometryProtocol
    volatility: float
    t: float
    iteration: int


# =============================================================================
# Protocol Definition
# =============================================================================


@runtime_checkable
class BCValueProvider(Protocol):
    """
    Protocol for dynamic boundary condition value generation.

    Providers compute BC values from the current system state during
    iteration. This enables state-dependent BCs (like a wall coefficient that
    moves with the Picard iterate) without coupling logic in solvers.

    The protocol is runtime-checkable, allowing isinstance() checks:
        >>> if isinstance(segment.value, BCValueProvider):
        ...     resolved = segment.value.compute(state)

    Methods:
        compute: Generate BC value(s) from current iteration state.

    Note:
        Providers should be lightweight and stateless where possible.
        Heavy computation should be cached or precomputed.
    """

    def compute(self, state: dict[str, Any]) -> float | NDArray[np.floating]:
        """
        Compute BC value(s) from current system state.

        Args:
            state: Dictionary containing iteration state. Standard keys:
                - 'm_current': previous Picard iterate of the FP density, shape (Nt+1, *grid)
                - 'U_current': previous iterate of the value function, same shape
                - 'geometry': Problem geometry object
                - 'volatility': the SDE volatility σ as supplied (not D = σ²/2; #1512, #2378)
                - 't': Current time (for time-dependent problems)
                - 'iteration': Current Picard iteration number

        Returns:
            BC value as scalar (single boundary point) or array
            (multiple boundary points).

        Raises:
            KeyError: If required state keys are missing.
            ValueError: If state values have unexpected shapes/types.
        """
        ...


# =============================================================================
# Abstract Base Class (for inheritance-based implementations)
# =============================================================================


class BaseBCValueProvider(ABC):
    """
    Abstract base class for BC value providers.

    Use this when you need shared infrastructure (e.g., caching, logging)
    across provider implementations. For simple providers, implementing
    the BCValueProvider protocol directly is sufficient.
    """

    @abstractmethod
    def compute(self, state: dict[str, Any]) -> float | NDArray[np.floating]:
        """Compute BC value(s) from state. See BCValueProvider.compute()."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# =============================================================================
# Concrete Providers
# =============================================================================


class ConstantProvider(BaseBCValueProvider):
    """
    Trivial provider that returns a constant value.

    Useful for testing and as a reference implementation.
    In practice, use a float directly in BCSegment.value instead.
    """

    def __init__(self, value: float) -> None:
        self.value = value

    def compute(self, state: dict[str, Any]) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"ConstantProvider({self.value})"


# =============================================================================
# Utility Functions
# =============================================================================


def is_provider(value: Any) -> bool:
    """
    Check if a value is a BC value provider.

    Args:
        value: The value to check (from BCSegment.value)

    Returns:
        True if value implements BCValueProvider protocol

    Example:
        >>> from mfgarchon.geometry.boundary.providers import is_provider
        >>> is_provider(0.0)  # False
        >>> is_provider(ConstantProvider(1.0))  # True
    """
    return isinstance(value, BCValueProvider)


def resolve_provider(
    value: float | BCValueProvider,
    state: dict[str, Any],
) -> float | NDArray[np.floating]:
    """
    Resolve a BC value, computing if it's a provider.

    Args:
        value: Static value or provider
        state: Current iteration state (passed to provider.compute())

    Returns:
        Resolved value - float for scalar BCs, NDArray for spatially-varying BCs

    Example:
        >>> value = ConstantProvider(1.0)
        >>> resolved = resolve_provider(value, state)  # Calls compute()
        >>> resolve_provider(1.5, state)  # Returns 1.5 unchanged
    """
    if is_provider(value):
        result = value.compute(state)
        return float(result) if not isinstance(result, np.ndarray) else result
    return float(value)

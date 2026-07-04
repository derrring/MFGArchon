"""
Dynamic boundary condition value providers (Issue #625).

This module implements the callback-provider pattern for state-dependent
boundary conditions. Instead of baking coupling logic into solvers, the
BC *intent* is stored in the BCSegment and resolved to concrete values
by the coupling iterator.

Architecture:
------------
1. BCValueProvider protocol defines the compute(state) -> value contract
2. Concrete providers (e.g., AdjointConsistentProvider) implement the formula
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
    >>> from mfgarchon.geometry.boundary.providers import AdjointConsistentProvider
    >>> from mfgarchon.geometry.boundary import BCSegment, BCType
    >>>
    >>> # Store intent in BCSegment
    >>> segment = BCSegment(
    ...     name="left_ac",
    ...     bc_type=BCType.ROBIN,
    ...     alpha=0.0, beta=1.0,
    ...     value=AdjointConsistentProvider(side="left", sigma=0.2),
    ...     boundary="x_min",
    ... )
    >>>
    >>> # Later, iterator resolves provider with current state
    >>> state = {'m_current': m, 'geometry': geometry, 'sigma': 0.2}
    >>> concrete_value = segment.value.compute(state)

References:
-----------
- Issue #625: Dynamic BC value provider architecture
- Issue #574: Original adjoint-consistent BC implementation
- docs/development/boundary_condition_handling_summary.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypedDict, runtime_checkable

import numpy as np

from mfgarchon.geometry.boundary.bc_coupling import compute_boundary_log_density_gradient_1d
from mfgarchon.utils.deprecation import deprecated_parameter

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

    Required by AdjointConsistentProvider:
        m_current: Current FP density array
        geometry: Problem geometry object
        sigma: SDE volatility σ (canonical; or the deprecated 'diffusion' key -- also σ, Issue #1512)

    Optional (context-dependent):
        U_current: Current value function array
        t: Current time (for time-dependent problems)
        iteration: Current Picard iteration number
        diffusion: Deprecated alias for 'sigma' (Issue #1512): the key holds σ, NOT D = σ²/2
    """

    m_current: NDArray[np.floating]
    U_current: NDArray[np.floating]
    geometry: GeometryProtocol
    sigma: float
    diffusion: float  # Deprecated (Issue #1512), use 'sigma'
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
    iteration. This enables state-dependent BCs (like adjoint-consistent
    reflecting boundaries) without coupling logic in solvers.

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
                - 'm_current': Current FP density array
                - 'U_current': Current value function array
                - 'geometry': Problem geometry object
                - 'sigma': SDE volatility σ (canonical)
                - 'diffusion': deprecated alias for 'sigma' (σ, not D = σ²/2; Issue #1512)
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


class AdjointConsistentProvider(BaseBCValueProvider):
    """
    Provider for adjoint-consistent Robin BC values (Issue #574).

    Computes the Robin BC value for reflecting boundaries that maintains
    adjoint consistency between HJB and FP equations:

        g = -σ²/2 · d(ln m)/dn

    where:
        - sigma: SDE volatility σ (NOT the PDE diffusion D = σ²/2; Issue #1512)
        - m: current FP density
        - d/dn: outward normal derivative at boundary

    SCOPE LIMITATION (Issue #625 Validation):
    -----------------------------------------
    This provider is designed for **boundary stall configurations only**.

    The formula is derived from the zero-flux equilibrium condition, which
    assumes the boundary IS the equilibrium point (stall). For interior stall:
    - The derivation doesn't apply (equilibrium flux is not zero at boundary)
    - Results are worse than standard Neumann BC (validated experimentally)
    - Use Strict Adjoint Mode (Issue #622) instead for universal applicability

    Validation results:
    - Boundary stall (x=0): AC BC error 1.36 vs Neumann 2.09 (1.54x better)
    - Interior stall (x=0.5): AC BC error 5.88 vs Neumann 1.55 (3.8x WORSE)

    When to use:
    - Stall point at domain boundary (x_stall = 0 or x_stall = L)
    - Reflecting (zero-flux) boundary conditions
    - MFG problems with crowd-aversion coupling

    When NOT to use:
    - Interior stall points (use Strict Adjoint Mode instead)
    - Absorbing boundaries
    - Periodic boundaries

    Attributes:
        side: Boundary side ("left" or "right" for 1D)
        diffusion: Diffusion coefficient (can be None to read from state)
        regularization: Small constant to prevent log(0)

    Example:
        >>> # Appropriate: boundary stall configuration
        >>> provider = AdjointConsistentProvider(side="left", sigma=0.04)
        >>> state = {'m_current': m_array, 'geometry': geom}
        >>> g_left = provider.compute(state)  # Robin BC value at left boundary

    See Also:
        - Strict Adjoint Mode (Issue #622): Universal L_FP = L_HJB^T enforcement
        - docs/development/boundary_condition_handling_summary.md for full analysis
    """

    # Side name aliases: map various conventions to canonical form
    # Canonical: "left"/"right" for 1D, "{axis}_min"/"{axis}_max" for nD
    _SIDE_ALIASES: ClassVar[dict[str, str]] = {
        # 1D aliases (all map to left/right)
        "left": "left",
        "right": "right",
        "x_min": "left",
        "x_max": "right",
        "min": "left",
        "max": "right",
        # 2D/3D aliases (for future nD support)
        "y_min": "y_min",
        "y_max": "y_max",
        "z_min": "z_min",
        "z_max": "z_max",
        "bottom": "y_min",
        "top": "y_max",
        "front": "z_min",
        "back": "z_max",
    }

    @deprecated_parameter(
        param_name="diffusion",
        since="v0.20.5",
        replacement="sigma",
    )
    def __init__(
        self,
        side: str,
        sigma: float | None = None,
        regularization: float = 1e-10,
        *,
        diffusion: float | None = None,  # Deprecated alias for sigma (misnomer; Issue #1512)
    ) -> None:
        """
        Initialize adjoint-consistent BC provider.

        Args:
            side: Boundary side identifier. Accepts multiple conventions:
                - 1D: "left", "right", "x_min", "x_max", "min", "max"
                - 2D: "y_min", "y_max", "bottom", "top"
                - 3D: "z_min", "z_max", "front", "back"
            sigma: SDE volatility σ (NOT the PDE diffusion D = σ²/2 -- the value is squared internally).
                If None, reads from state. Named to match the codebase convention (``problem.sigma`` = σ,
                ``problem.diffusion`` = σ²/2); passing the D-valued ``problem.diffusion`` here is the
                Issue #1512 trap this canonical name removes.
            regularization: Small positive constant added to density
                           to prevent log(0). Default 1e-10.
            diffusion: DEPRECATED (Issue #1512): a misnomer -- this argument is σ, not D, and the name
                collided with the ``D = σ²/2`` convention. Use ``sigma``.
        """
        if side not in self._SIDE_ALIASES:
            valid = sorted(self._SIDE_ALIASES.keys())
            raise ValueError(f"side must be one of {valid}, got '{side}'")

        # Issue #1512: `diffusion` was a misnamed alias for σ (it is squared in compute) that collided
        # with the codebase D = σ²/2 convention. `sigma` is now canonical; `diffusion` redirects to it.
        if diffusion is not None and sigma is None:
            sigma = diffusion

        # Store both original and normalized side names
        self._original_side = side
        self.side = self._SIDE_ALIASES[side]  # Normalize to canonical form
        self.sigma = sigma
        self.regularization = regularization

    def compute(self, state: dict[str, Any]) -> float:
        """
        Compute adjoint-consistent Robin BC value.

        Args:
            state: Must contain:
                - 'm_current': FP density array (interior points)
                - 'geometry': Geometry object with get_grid_spacing()
                - 'diffusion' or 'sigma': Diffusion coefficient σ
                  (if not set in __init__). 'diffusion' takes priority.

        Returns:
            Robin BC value: g = -σ²/2 * d(ln m)/dn

        Raises:
            KeyError: If required state keys missing
            ValueError: If density has invalid shape
        """
        # Get density
        m = state.get("m_current")
        if m is None:
            raise KeyError("AdjointConsistentProvider requires 'm_current' in state")

        # Handle time-dependent density (take final time slice)
        if m.ndim > 1:
            m = m[-1, :]  # Final time slice for backward HJB

        # Get geometry for grid spacing
        geometry = state.get("geometry")
        if geometry is None:
            raise KeyError("AdjointConsistentProvider requires 'geometry' in state")

        # Get diffusion coefficient (try constructor, then state)
        sigma = self.sigma
        if sigma is None:
            # Issue #1512: 'sigma' is canonical (σ); 'diffusion' is the deprecated key -- also σ here,
            # NOT the D = σ²/2 the same word means elsewhere.
            sigma = state.get("sigma")
            if sigma is None:
                sigma = state.get("diffusion")
            if sigma is None:
                raise KeyError(
                    "AdjointConsistentProvider: volatility not set in constructor "
                    "and neither 'sigma' nor 'diffusion' found in state"
                )

        # Compute log-density gradient at boundary
        # Delegate to dimension-aware function in bc_coupling module
        grad_ln_m = self._compute_boundary_log_gradient(m, geometry, self.side, self.regularization)

        # Robin BC value: g = -σ²/2 * d(ln m)/dn
        return -(sigma**2) / 2 * grad_ln_m

    def _compute_boundary_log_gradient(
        self,
        m: NDArray[np.floating],
        geometry: Any,
        side: str,
        regularization: float,
    ) -> float:
        """
        Compute ∂ln(m)/∂n at boundary using geometry-appropriate method.

        This method dispatches to dimension-specific implementations.
        For nD support, geometry should provide gradient operators.

        Args:
            m: Density array (interior points)
            geometry: Geometry object with get_grid_spacing()
            side: Normalized boundary side identifier
            regularization: Small constant to prevent log(0)

        Returns:
            Outward normal derivative of ln(m) at boundary

        Note:
            Current implementation: 1D only (uses finite differences)
            Future (Issue #624): Use geometry.get_gradient_operator() for nD
        """
        dimension = getattr(geometry, "dimension", 1)

        if dimension == 1 or side in ("left", "right"):
            # 1D case: use finite difference implementation
            dx = geometry.get_grid_spacing()[0]
            return compute_boundary_log_density_gradient_1d(m, dx, side, regularization)
        else:
            # nD case: requires geometry gradient operators (Issue #624)
            raise NotImplementedError(
                f"AdjointConsistentProvider: {dimension}D not yet implemented. "
                f"See Issue #624 for nD adjoint-consistent BC support."
            )

    def __repr__(self) -> str:
        sig_str = f"{self.sigma}" if self.sigma is not None else "from_state"
        # Show original side name for user clarity
        return f"AdjointConsistentProvider(side='{self._original_side}', sigma={sig_str})"


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
        >>> is_provider(AdjointConsistentProvider("left", 0.2))  # True
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
        >>> value = AdjointConsistentProvider("left", 0.2)
        >>> resolved = resolve_provider(value, state)  # Calls compute()
        >>> resolve_provider(1.5, state)  # Returns 1.5 unchanged
    """
    if is_provider(value):
        result = value.compute(state)
        return float(result) if not isinstance(result, np.ndarray) else result
    return float(value)


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    """Quick validation of provider architecture."""
    import warnings

    print("Testing BC value providers...")
    print()

    # Test 1: Protocol check
    print("Test 1: Protocol compliance")
    provider = AdjointConsistentProvider(side="left", sigma=0.2)
    assert isinstance(provider, BCValueProvider), "Should implement protocol"
    assert is_provider(provider), "is_provider() should return True"
    assert not is_provider(1.5), "Float should not be a provider"
    print("  Protocol checks passed")
    print()

    # Test 2: Compute with mock state
    print("Test 2: Compute with mock state")

    # Create mock geometry
    class MockGeometry:
        def get_grid_spacing(self):
            return [0.1]

    # Exponential density: m(x) = exp(-x), so d(ln m)/dx = -1
    x = np.linspace(0, 1, 11)
    m = np.exp(-x)

    state = {
        "m_current": m,
        "geometry": MockGeometry(),
        "diffusion": 0.2,  # Canonical parameter name
    }

    # Left boundary: outward normal is -x, so d(ln m)/dn = -(-1) = 1
    # g = -0.2^2/2 * 1 = -0.02
    left_provider = AdjointConsistentProvider(side="left", sigma=0.2)
    g_left = left_provider.compute(state)
    expected_left = -(0.2**2) / 2 * 1.0  # -0.02
    print(f"  Left BC value: {g_left:.6f} (expected ~ {expected_left:.6f})")

    # Right boundary: outward normal is +x, so d(ln m)/dn = -1
    # g = -0.2^2/2 * (-1) = 0.02
    right_provider = AdjointConsistentProvider(side="right", sigma=0.2)
    g_right = right_provider.compute(state)
    expected_right = -(0.2**2) / 2 * (-1.0)  # 0.02
    print(f"  Right BC value: {g_right:.6f} (expected ~ {expected_right:.6f})")
    print()

    # Test 3: resolve_provider utility
    print("Test 3: resolve_provider utility")
    resolved = resolve_provider(left_provider, state)
    assert abs(resolved - g_left) < 1e-10, "Should match direct compute"
    resolved_static = resolve_provider(42.0, state)
    assert resolved_static == 42.0, "Static value should pass through"
    print("  resolve_provider works correctly")
    print()

    # Test 4: Diffusion from state (not constructor)
    print("Test 4: Diffusion from state (not constructor)")
    provider_no_diff = AdjointConsistentProvider(side="left", sigma=None)
    g_from_state = provider_no_diff.compute(state)
    assert abs(g_from_state - g_left) < 1e-10, "Should use diffusion from state"
    print("  Diffusion correctly read from state when not in constructor")
    print()

    # Test 5: Legacy 'sigma' in state (backward compatibility)
    print("Test 5: Legacy 'sigma' in state (backward compatibility)")
    state_legacy = {
        "m_current": m,
        "geometry": MockGeometry(),
        "sigma": 0.2,  # Legacy parameter name
    }
    provider_legacy = AdjointConsistentProvider(side="left", sigma=None)
    g_legacy = provider_legacy.compute(state_legacy)
    assert abs(g_legacy - g_left) < 1e-10, "Should accept 'sigma' in state for backward compat"
    print("  Legacy 'sigma' key in state works correctly")
    print()

    # Test 6: Deprecated 'diffusion' parameter (Issue #1512: 'diffusion' is a misnomer for sigma)
    print("Test 6: Deprecated 'diffusion' parameter (shows warning)")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        provider_deprecated = AdjointConsistentProvider(side="left", diffusion=0.2)
        assert len(w) == 1, "Should emit deprecation warning"
        assert "deprecated" in str(w[0].message).lower()
        assert provider_deprecated.sigma == 0.2, "diffusion should map to sigma"
    print("  Deprecated 'diffusion' parameter works with warning")
    print()

    print("All provider tests passed!")

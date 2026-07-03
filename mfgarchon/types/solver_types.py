"""
Solver Type Definitions

This module consolidates all solver-related type definitions including
return types, state representations, and solver-specific type aliases.

Usage:
    from mfgarchon.types.solver_types import SolverReturnTuple, JAXSolverReturn
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from numpy.typing import NDArray

# === Standard Solver Return Types ===

type SolverReturnTuple = tuple[NDArray, NDArray, dict[str, Any]]
"""
Standard solver return type: (U, M, convergence_info)

This is the legacy return format used by most numerical solvers.

Args:
    U: Value function array, shape (Nt+1, Nx+1)
    M: Density function array, shape (Nt+1, Nx+1)
    convergence_info: Dict with keys like 'converged', 'iterations', 'residual'
"""

type JAXSolverReturn = tuple[Any, Any, bool, int, float]
"""
JAX-accelerated solver return type.

Returns:
    U_jax: JAX array for value function
    M_jax: JAX array for density function
    converged: Boolean convergence flag
    iterations: Number of iterations performed
    residual: Final residual value
"""

# === Solver State Types ===

type SolverState = tuple[NDArray, NDArray] | dict[str, Any]
"""
Simple solver state representation.

Either:
- (u, m): Simple tuple of current solution
- dict: Complex state with metadata
"""

type ComplexSolverState = (
    tuple[NDArray, NDArray]  # Simple (u, m) state
    | dict[str, NDArray | float | int]  # Complex state with metadata
    | object  # Completely custom state objects
)
"""
Internal solver state - can be simple arrays or complex objects.

Used by iterative solvers to maintain state between iterations.
"""

type IntermediateResult = (
    NDArray  # Simple array result
    | tuple[NDArray, ...]  # Multiple arrays
    | dict[str, NDArray]  # Named array collection
    | Any  # Completely flexible
)
"""
Intermediate computation results during solving.

Solvers may return various formats during computation.
"""

# === Solver Configuration Types ===

type ParameterDict = dict[str, float | int | str | bool]
"""Dictionary of solver parameters."""

type SolverOptions = dict[str, float | int | str | bool | None]
"""Dictionary of optional solver settings."""

type ConfigValue = float | int | str | bool | None | list[float | int | str] | dict[str, Any] | Callable | Any
"""Flexible configuration value type."""

# === Callback Types ===

type ErrorCallback = Callable[[Exception], None] | None
"""Optional error handling callback function."""

type ProgressCallback = Callable[[int, float], None] | None
"""
Progress callback: (iteration, residual) -> None

Called after each solver iteration.
"""

type ConvergenceCallback = Callable[[bool, int, float], None] | None
"""
Convergence callback: (converged, iterations, final_residual) -> None

Called when solver terminates.
"""

# === GFDM-Specific Types ===

type MultiIndexTuple = tuple[int, ...]
"""Multi-index for GFDM operations, e.g., (2, 1) for ∂²/∂x²∂y."""

type DerivativeDict = dict[tuple[int, ...], float]
"""Dictionary mapping multi-indices to derivative values."""

type GradientDict = dict[str, float]
"""Dictionary for gradient components: {'dx': value, 'dy': value, ...}"""

type StencilResult = list[tuple[NDArray, bool]]
"""Stencil computation result: list of (stencil_array, success_flag)."""

# === Metadata Types ===

type MetadataDict = dict[str, float | int | str | bool | NDArray | None]
"""Flexible metadata dictionary for solver state and results."""

type ConvergenceMetadata = dict[str, Any]
"""
Convergence information metadata.

Typical keys: 'converged', 'iterations', 'residual', 'reason'
"""

# === JAX-Specific Types ===

try:
    if TYPE_CHECKING:
        from jax import Array as JAXArray
    else:
        JAXArray = Any  # Runtime fallback
except ImportError:
    JAXArray = Any  # JAX not available

type JAXStateArray = JAXArray | NDArray
"""Array type that can be either JAX or NumPy."""

# === Mathematical Function Types ===

type HamiltonianFunction = Callable[[float, float, float, float], float]
"""Hamiltonian function H(x, p, m, t) -> float"""

type LagrangianFunction = Callable[[float, float, float, float], float]
"""Lagrangian function L(x, v, m, t) -> float"""

type DensityFunction = Callable[[float], float]
"""Density function ρ(x) -> float"""

type ValueFunction = Callable[[float], float]
"""Value function g(x) -> float"""


# === Hamiltonian Jacobian Types ===


@dataclass
class HamiltonianJacobians:
    """
    Jacobian components of Hamiltonian H(x,p,m) with respect to momentum p = ∇U.

    In finite difference discretizations of the HJB equation, the Hamiltonian's
    Jacobian forms a tridiagonal coefficient matrix for Newton/policy iteration.

    Mathematical Context:
        For HJB equation: ∂U/∂t + H(x, ∇U, m) = 0

        The Jacobian ∂H/∂p (where p = ∇U) gives linearization:
        H(x, p + δp, m) ≈ H(x, p, m) + ∂H/∂p · δp

        In 1D finite differences with central/upwind schemes:
        - diagonal: coefficient for p_i (current point)
        - lower: coefficient for p_{i-1} (left neighbor)
        - upper: coefficient for p_{i+1} (right neighbor)

    Usage:
        >>> jacobians = problem.compute_hjb_jacobian_terms(U, M, t)
        >>> # Access components with clear names
        >>> A_diag = jacobians.diagonal
        >>> A_lower = jacobians.lower
        >>> A_upper = jacobians.upper

    Attributes:
        diagonal: Diagonal Jacobian coefficients J_D_H, shape (Nx,)
        lower: Lower-diagonal Jacobian coefficients J_L_H, shape (Nx,)
        upper: Upper-diagonal Jacobian coefficients J_U_H, shape (Nx,)

    See Also:
        mfgarchon.core.mfg_problem.MFGProblem.compute_hjb_jacobian_terms
    """

    diagonal: NDArray  # J_D_H
    lower: NDArray  # J_L_H
    upper: NDArray  # J_U_H


# === Solver Component Protocols ===


class NewtonSolver(Protocol):
    """Protocol for Newton-type solvers."""

    def solve_step(self, u_current: NDArray, rhs: NDArray) -> NDArray: ...


class LinearSolver(Protocol):
    """Protocol for linear system solvers."""

    def solve(self, A: NDArray, b: NDArray) -> NDArray: ...


# === Flexible Input Types ===

type FlexibleInput = (
    HamiltonianFunction
    | str  # String identifier for preset Hamiltonians
    | dict[str, float | Callable]  # Configuration dictionary
    | object  # Custom problem objects
)
"""Flexible input type that accepts multiple input formats"""

# === Structured Result Types ===


@dataclass
class NetworkSolveResult:
    """
    Result from network MFG equilibrium computation.

    This lightweight dataclass is used for network-based MFG problems where
    solution is computed directly from trajectory measures or other network-specific
    methods (not iterative solvers with convergence info).

    Attributes:
        U: Value function on network nodes, shape (Nt+1, num_nodes)
        M: Density function on network nodes, shape (Nt+1, num_nodes)
        metadata: Optional metadata about the computation

    Usage:
        >>> result = NetworkSolveResult(U=u_array, M=m_array)
        >>> # Backward compatible unpacking
        >>> u, m = result

    Mathematical Context:
        For network MFG problems, the state space is discrete (network nodes)
        rather than continuous. The value function U[t, i] represents the value
        at node i and time t, while M[t, i] represents the population density.

    See Also:
        - SolverResult: Full solver result with convergence information
    """

    U: NDArray  # Value function u(t, node)
    M: NDArray  # Density function m(t, node)
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate result data after initialization."""
        if self.U.shape != self.M.shape:
            raise ValueError(f"U and M shapes must match: U{self.U.shape} vs M{self.M.shape}")

        if self.metadata is None:
            self.metadata = {}

    # Backward compatibility: allow tuple-like unpacking
    def __iter__(self):
        """Enable tuple-like unpacking: u, m = result"""
        yield self.U
        yield self.M

    def __len__(self):
        """Return tuple length for compatibility."""
        return 2

    def __getitem__(self, index):
        """Enable indexing like a tuple for backward compatibility."""
        tuple_representation = (self.U, self.M)
        return tuple_representation[index]


# === Error Classes ===


class SolverError(Exception):
    """Base exception for solver errors."""


class ConvergenceError(SolverError):
    """Raised when solver fails to converge."""


class ConfigurationError(SolverError):
    """Raised when solver configuration is invalid."""


# === Legacy Compatibility ===

# Map old names to new standard names for backward compatibility
# These are accessed via __getattr__ to emit deprecation warnings
_DEPRECATED_ALIASES = {
    "LegacySolverReturn": ("SolverReturnTuple", SolverReturnTuple),
}


def __getattr__(name: str):
    """Emit deprecation warnings for legacy type aliases."""
    if name in _DEPRECATED_ALIASES:
        new_name, value = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"'{name}' is deprecated, use '{new_name}' instead. This alias will be removed in v1.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

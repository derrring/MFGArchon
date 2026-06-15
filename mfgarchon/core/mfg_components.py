"""MFG Components and evaluation mixins.

This module contains:
- MFGComponents: Dataclass for custom MFG problem definition
- HamiltonianMixin: Mixin providing Hamiltonian evaluation methods (H, dH_dm, potential)
- ConditionsMixin: Mixin providing initial/final/boundary condition methods

The mixin pattern allows MFGProblem to inherit these methods while keeping
the logic in separate, focused modules.
"""

from __future__ import annotations

import contextlib
import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.core.derivatives import DerivativeTensors, to_multi_index_dict

# Issue #670: npart, ppart imports removed - no default Hamiltonian
from mfgarchon.utils.mfg_logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.geometry.boundary.conditions import BoundaryConditions
    from mfgarchon.types import HamiltonianJacobians

# Define a limit for values before squaring to prevent overflow within H
VALUE_BEFORE_SQUARE_LIMIT = 1e150


# ============================================================================
# MFG Components for Custom Problem Definition
# ============================================================================


@dataclass
class MFGComponents:
    """
    Container for all components that define a custom MFG problem.

    Issue #673: Class-based Hamiltonian API only (legacy function-based removed).

    Hamiltonian Specification
    -------------------------
    Use class-based Hamiltonian (HamiltonianBase subclass):

    ```python
    from mfgarchon.core.hamiltonian import SeparableHamiltonian, QuadraticControlCost

    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: -m**2,
        coupling_dm=lambda m: -2*m,
    )
    components = MFGComponents(hamiltonian=H, m_initial=..., u_terminal=...)
    ```

    Or use Lagrangian (auto-converted via Legendre transform):

    ```python
    from mfgarchon.core.hamiltonian import LagrangianBase

    class MyLagrangian(LagrangianBase):
        def __call__(self, x, alpha, m, t=0.0):
            return 0.5 * np.sum(alpha**2)

    components = MFGComponents(lagrangian=MyLagrangian(), m_initial=..., u_terminal=...)
    ```

    Attributes
    ----------
    hamiltonian : HamiltonianBase
        Class-based Hamiltonian H(x, m, p, t). Required.
    lagrangian : LagrangianBase, optional
        Alternative to hamiltonian - auto-converted via Legendre transform.
    m_initial : Callable | NDArray
        Initial density distribution m_0(x).
    u_terminal : Callable | NDArray
        Terminal value function u_T(x).
    potential_func : Callable, optional
        Additional potential V(x, t) (if not in Hamiltonian).
    boundary_conditions : BoundaryConditions, optional
        Boundary conditions for the domain.
    """

    # Class-based Hamiltonian or Lagrangian (Issue #673)
    hamiltonian: Any = None  # HamiltonianBase instance
    lagrangian: Any = None  # LagrangianBase instance (auto-converted)

    # Initial and terminal conditions
    m_initial: Callable | NDArray | None = None  # m_0(x): initial density
    u_terminal: Callable | NDArray | None = None  # u_T(x): terminal value function
    u_final: Callable | NDArray | None = None  # DEPRECATED: use u_terminal instead

    # Optional potential (if not included in Hamiltonian)
    potential_func: Callable | None = None  # V(x, t) -> float

    # Boundary conditions
    boundary_conditions: BoundaryConditions | None = None

    # Problem parameters and metadata
    parameters: dict[str, Any] = field(default_factory=dict)
    description: str = "MFG Problem"
    problem_type: str = "mfg"

    # Internal: stores validated Hamiltonian and Lagrangian classes
    _hamiltonian_class: Any = field(default=None, init=False, repr=False)
    _lagrangian_class: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Validate and setup Hamiltonian/Lagrangian from class-based specification."""
        import warnings

        from mfgarchon.core.hamiltonian import (
            HamiltonianBase,
            LagrangianBase,
            SeparableHamiltonian,
            SeparableLagrangian,
        )

        # Handle u_final -> u_terminal deprecation
        if self.u_final is not None:
            warnings.warn(
                "u_final is deprecated, use u_terminal instead. u_final will be removed in v1.0.0.",
                DeprecationWarning,
                stacklevel=3,
            )
            if self.u_terminal is None:
                self.u_terminal = self.u_final
            self.u_final = None  # Clear deprecated field

        # Validate Lagrangian if provided
        if self.lagrangian is not None and isinstance(self.lagrangian, LagrangianBase):
            self._lagrangian_class = self.lagrangian

        # Issue #899: Accept either H or L (or both).
        # If only L is provided, derive H for PDE solvers.
        # If only H is provided, derive L if it's separable.
        if self.hamiltonian is not None and self.lagrangian is not None:
            # Both provided: use as-is, no conversion
            pass
        elif self.lagrangian is not None and self.hamiltonian is None:
            # L only: derive H for PDE solvers
            if isinstance(self.lagrangian, LagrangianBase):
                # Issue #1068: isinstance(SeparableLagrangian) replaces hasattr(, "as_hamiltonian").
                # SeparableLagrangian is the only LagrangianBase subclass with as_hamiltonian().
                if isinstance(self.lagrangian, SeparableLagrangian):
                    # SeparableLagrangian: analytic conversion, no DualHamiltonian
                    self.hamiltonian = self.lagrangian.as_hamiltonian()
                else:
                    # General: numerical Legendre transform (legacy path)
                    self.hamiltonian = self.lagrangian.legendre_transform()
        elif self.hamiltonian is not None and self.lagrangian is None:
            # H only: derive L if separable (for ADMM/variational)
            # Issue #1068: isinstance(SeparableHamiltonian) replaces hasattr(, "control_cost")
            # and hasattr(, "_potential"). Only SeparableHamiltonian carries both attributes.
            if isinstance(self.hamiltonian, SeparableHamiltonian):
                self._lagrangian_class = SeparableLagrangian(
                    control_cost=self.hamiltonian.control_cost,
                    potential=getattr(self.hamiltonian, "_potential", None),
                    coupling=getattr(self.hamiltonian, "_coupling", None),
                    sense=self.hamiltonian.sense,
                )

        # Validate: at least one of H or L must be provided
        if self.hamiltonian is None and self.lagrangian is None:
            raise ValueError(
                "MFGComponents requires a Hamiltonian or Lagrangian.\n\n"
                "Example:\n"
                "  from mfgarchon.core.hamiltonian import SeparableHamiltonian, QuadraticControlCost\n\n"
                "  H = SeparableHamiltonian(\n"
                "      control_cost=QuadraticControlCost(control_cost=1.0),\n"
                "      coupling=lambda m: -m**2,\n"
                "  )\n"
                "  components = MFGComponents(hamiltonian=H, m_initial=..., u_terminal=...)"
            )

        if self.hamiltonian is not None:
            if not isinstance(self.hamiltonian, HamiltonianBase):
                raise TypeError(
                    f"hamiltonian must be a HamiltonianBase instance, got {type(self.hamiltonian).__name__}."
                )
            self._hamiltonian_class = self.hamiltonian


# ============================================================================
# Hamiltonian Mixin - H, dH_dm, Jacobian, Coupling, Potential
# ============================================================================


class HamiltonianMixin:
    """
    Mixin class providing Hamiltonian evaluation methods.

    Provides:
    - H(): Hamiltonian evaluation
    - dH_dm(): Hamiltonian derivative w.r.t. density
    - get_hjb_hamiltonian_jacobian_contrib(): Jacobian for Newton methods
    - get_hjb_residual_m_coupling_term(): Coupling terms
    - _setup_custom_potential(): Potential initialization
    - get_potential_at_time(): Time-dependent potential access

    Required attributes from the inheriting class:
    - components: MFGComponents | None
    - is_custom: bool
    - coupling_coefficient: float
    - tSpace: np.ndarray
    - f_potential: np.ndarray
    - spatial_shape: tuple
    - dimension: int
    - _get_spatial_grid_internal(): method
    - _get_num_intervals(): method
    - _get_spacing(): method
    """

    # Type hints for attributes expected from MFGProblem
    components: MFGComponents | None
    is_custom: bool
    coupling_coefficient: float
    tSpace: np.ndarray
    f_potential: np.ndarray
    spatial_shape: tuple
    dimension: int

    # Cached signature parameters (set during validation)
    # Issue #673: Legacy function signature caching removed - class-based API only
    _potential_has_time: bool | None = None

    def _validate_hamiltonian_components(self) -> None:
        """Validate Hamiltonian-related components.

        Issue #670, #673: Class-based Hamiltonian required - no function-based API.
        """
        # Issue #670: components must be provided
        if self.components is None:
            raise ValueError(
                "MFGComponents must be provided. No default problem is supported.\n\n"
                "Example:\n"
                "  from mfgarchon.core.hamiltonian import SeparableHamiltonian, QuadraticControlCost\n\n"
                "  H = SeparableHamiltonian(\n"
                "      control_cost=QuadraticControlCost(control_cost=1.0),\n"
                "      coupling=lambda m: -m**2,\n"
                "      coupling_dm=lambda m: -2*m,\n"
                "  )\n"
                "  components = MFGComponents(hamiltonian=H, m_initial=..., u_terminal=...)\n"
                "  problem = MFGProblem(geometry=grid, components=components, ...)"
            )

        # Issue #673: Class-based Hamiltonian required
        has_hamiltonian_class = getattr(self.components, "_hamiltonian_class", None) is not None

        if not has_hamiltonian_class:
            raise ValueError(
                "Class-based Hamiltonian must be provided in MFGComponents.\n\n"
                "Example:\n"
                "  from mfgarchon.core.hamiltonian import SeparableHamiltonian, QuadraticControlCost\n\n"
                "  H = SeparableHamiltonian(\n"
                "      control_cost=QuadraticControlCost(control_cost=1.0),\n"
                "      coupling=lambda m: -m**2,\n"
                "      coupling_dm=lambda m: -2*m,\n"
                "  )\n"
                "  components = MFGComponents(hamiltonian=H, m_initial=..., u_terminal=...)"
            )

        # Cache potential signature info (optional component)
        if self.components.potential_func is not None:
            sig = inspect.signature(self.components.potential_func)
            self._potential_has_time = "t" in sig.parameters or "time" in sig.parameters

    def _setup_custom_potential(self) -> None:
        """Setup custom potential function (part of Hamiltonian)."""
        if self.components is None or self.components.potential_func is None:
            return

        potential_func = self.components.potential_func
        spatial_grid = self._get_spatial_grid_internal()
        num_intervals = self._get_num_intervals() or 0

        # Use cached signature info (set during validation)
        has_time = self._potential_has_time
        if has_time is None:
            sig = inspect.signature(potential_func)
            has_time = "t" in sig.parameters or "time" in sig.parameters

        for i in range(num_intervals + 1):
            # Extract scalar from grid point (grid has shape (Nx, 1) for 1D)
            x_i = float(spatial_grid[i, 0])
            if has_time:
                self.f_potential[i] = potential_func(x_i, 0.0)
            else:
                self.f_potential[i] = potential_func(x_i)

    def get_potential_at_time(self, t_idx: int) -> np.ndarray:
        """Get potential function at specific time (for time-dependent potentials)."""
        if self.is_custom and self.components is not None and self.components.potential_func is not None:
            # Use cached signature info (set during validation) for performance
            has_time = self._potential_has_time
            if has_time is None:
                sig = inspect.signature(self.components.potential_func)
                has_time = "t" in sig.parameters or "time" in sig.parameters

            if has_time:
                current_time = self.tSpace[t_idx] if t_idx < len(self.tSpace) else 0.0
                potential_at_t = np.zeros_like(self.f_potential)

                num_intervals = self._get_num_intervals() or 0
                spatial_grid = self._get_spatial_grid_internal()
                for i in range(num_intervals + 1):
                    # Extract scalar from grid point (grid has shape (Nx, 1) for 1D)
                    x_i = float(spatial_grid[i, 0])
                    potential_at_t[i] = self.components.potential_func(x_i, current_time)

                return potential_at_t

        return self.f_potential.copy()

    def _get_potential_at_index(self, x_idx: int | tuple[int, ...]) -> float:
        """
        Safely retrieve potential cost V(x) at given index.

        Handles both grid-based (FDM) and meshfree (GFDM) methods.
        Returns 0.0 if index is out of bounds (meshfree fallback).

        Args:
            x_idx: Grid index - either flat int or tuple for nD

        Returns:
            Potential value at index, or 0.0 if out of bounds
        """
        if self.f_potential is None:
            return 0.0

        # Normalize to flat index
        flat_idx: int | None = None

        if isinstance(x_idx, tuple):
            if self.spatial_shape is not None and len(self.spatial_shape) > 1:
                # Multi-dimensional tuple index - convert to flat
                # Use mode='clip' to avoid ValueError, then check bounds
                try:
                    flat_idx = int(np.ravel_multi_index(x_idx, self.spatial_shape, mode="raise"))
                except ValueError:
                    # Coordinate outside defined shape
                    return 0.0
            else:
                # 1D tuple like (5,)
                flat_idx = x_idx[0] if len(x_idx) > 0 else 0
        else:
            # Scalar index
            flat_idx = int(x_idx)

        # Single bounds check
        if 0 <= flat_idx < self.f_potential.size:
            return float(self.f_potential.flat[flat_idx])

        # Out of bounds - meshfree method with index beyond grid
        return 0.0

    def H(
        self,
        x_idx: int,
        m_at_x: float,
        derivs: dict[tuple, float] | DerivativeTensors | None = None,
        p_values: dict[str, float] | None = None,
        t_idx: int | None = None,
        x_position: float | np.ndarray | None = None,
        current_time: float | None = None,
    ) -> float:
        """
        Hamiltonian function H(x, m, p, t).

        Issue #673: Class-based Hamiltonian API only - no legacy function support.

        Args:
            x_idx: Grid index (0 to Nx)
            m_at_x: Density at grid point x_idx
            derivs: Derivatives - DerivativeTensors or tuple-key dict {(1,): du/dx}
            p_values: DEPRECATED - raises error. Use derivs instead.
            t_idx: Time index (optional)
            x_position: Actual position coordinate (computed from x_idx if not provided)
            current_time: Actual time value (computed from t_idx if not provided)

        Returns:
            Hamiltonian value H(x, m, p, t)
        """
        # Issue #673: Error on legacy p_values parameter
        if p_values is not None:
            raise ValueError(
                "p_values parameter is no longer supported.\n\n"
                "Use derivs with tuple notation instead:\n"
                "  # Old: p_values={'forward': 0.1, 'backward': 0.1}\n"
                "  # New: derivs={(1,): 0.1}  # 1D gradient\n\n"
                "Or use DerivativeTensors:\n"
                "  derivs = DerivativeTensors.from_gradient(np.array([0.1]))"
            )

        if derivs is None:
            raise ValueError("Must provide 'derivs' to H()")

        # Get class-based Hamiltonian
        H_class = getattr(self.components, "_hamiltonian_class", None) if self.components else None
        if H_class is None:
            raise ValueError(
                "Class-based Hamiltonian must be provided in MFGComponents.\n\n"
                "Example:\n"
                "  from mfgarchon.core.hamiltonian import SeparableHamiltonian, QuadraticControlCost\n\n"
                "  H = SeparableHamiltonian(\n"
                "      control_cost=QuadraticControlCost(control_cost=1.0),\n"
                "      coupling=lambda m: -m**2,\n"
                "      coupling_dm=lambda m: -2*m,\n"
                "  )\n"
                "  components = MFGComponents(hamiltonian=H, m_initial=..., u_terminal=...)"
            )

        # Convert derivs to numpy p array for class-based H(x, m, p, t)
        if isinstance(derivs, DerivativeTensors):
            derivs_dict = to_multi_index_dict(derivs)
        else:
            derivs_dict = derivs

        # Extract momentum p from derivs dict
        if derivs_dict:
            dim = max(len(k) for k in derivs_dict) if derivs_dict else 1
            p = np.zeros(dim)
            for i in range(dim):
                key = tuple(1 if j == i else 0 for j in range(dim))
                p[i] = derivs_dict.get(key, 0.0)
        else:
            p = np.zeros(1)

        # Compute x_position if not provided
        if x_position is None:
            if isinstance(x_idx, tuple):
                if self.geometry is not None and self.spatial_shape is not None and len(self.spatial_shape) > 1:
                    flat_idx = np.ravel_multi_index(x_idx, self.spatial_shape)
                    spatial_grid = self.geometry.get_spatial_grid()
                    x_position = spatial_grid[flat_idx]
            else:
                spatial_grid = self._get_spatial_grid_internal()
                if spatial_grid is not None:
                    x_position = spatial_grid[x_idx]

        # Compute current_time if not provided
        if current_time is None and t_idx is not None:
            current_time = self.tSpace[t_idx] if t_idx < len(self.tSpace) else 0.0
        if current_time is None:
            current_time = 0.0

        # Convert x_position to numpy array
        x = np.atleast_1d(x_position if x_position is not None else 0.0)

        # Call class-based Hamiltonian directly: H(x, m, p, t)
        return float(H_class(x, m_at_x, p, current_time))

    def dH_dm(
        self,
        x_idx: int,
        m_at_x: float,
        derivs: dict[tuple, float] | DerivativeTensors | None = None,
        p_values: dict[str, float] | None = None,
        t_idx: int | None = None,
        x_position: float | np.ndarray | None = None,
        current_time: float | None = None,
    ) -> float:
        """
        Hamiltonian derivative with respect to density dH/dm.

        Issue #673: Class-based Hamiltonian API only - no legacy function support.

        Args:
            x_idx: Grid index (0 to Nx)
            m_at_x: Density at grid point x_idx
            derivs: Derivatives - DerivativeTensors or tuple-key dict {(1,): du/dx}
            p_values: DEPRECATED - raises error. Use derivs instead.
            t_idx: Time index (optional)
            x_position: Actual position coordinate (computed from x_idx if not provided)
            current_time: Actual time value (computed from t_idx if not provided)

        Returns:
            Derivative dH/dm at (x, m, p, t)
        """
        # Issue #673: Error on legacy p_values parameter
        if p_values is not None:
            raise ValueError(
                "p_values parameter is no longer supported.\n\n"
                "Use derivs with tuple notation instead:\n"
                "  # Old: p_values={'forward': 0.1, 'backward': 0.1}\n"
                "  # New: derivs={(1,): 0.1}  # 1D gradient"
            )

        if derivs is None:
            raise ValueError("Must provide 'derivs' to dH_dm()")

        # Get class-based Hamiltonian
        H_class = getattr(self.components, "_hamiltonian_class", None) if self.components else None
        if H_class is None:
            raise ValueError(
                "Class-based Hamiltonian must be provided in MFGComponents.\n\n"
                "Example:\n"
                "  from mfgarchon.core.hamiltonian import SeparableHamiltonian, QuadraticControlCost\n\n"
                "  H = SeparableHamiltonian(\n"
                "      control_cost=QuadraticControlCost(control_cost=1.0),\n"
                "      coupling=lambda m: -m**2,\n"
                "      coupling_dm=lambda m: -2*m,\n"
                "  )\n"
                "  components = MFGComponents(hamiltonian=H, m_initial=..., u_terminal=...)"
            )

        # Convert derivs to numpy p array
        if isinstance(derivs, DerivativeTensors):
            derivs_dict = to_multi_index_dict(derivs)
        else:
            derivs_dict = derivs

        # Extract momentum p from derivs dict
        if derivs_dict:
            dim = max(len(k) for k in derivs_dict) if derivs_dict else 1
            p = np.zeros(dim)
            for i in range(dim):
                key = tuple(1 if j == i else 0 for j in range(dim))
                p[i] = derivs_dict.get(key, 0.0)
        else:
            p = np.zeros(1)

        # Compute x_position if not provided
        if x_position is None:
            if isinstance(x_idx, tuple):
                if self.geometry is not None and self.spatial_shape is not None and len(self.spatial_shape) > 1:
                    flat_idx = np.ravel_multi_index(x_idx, self.spatial_shape)
                    spatial_grid = self.geometry.get_spatial_grid()
                    x_position = spatial_grid[flat_idx]
            else:
                spatial_grid = self._get_spatial_grid_internal()
                if spatial_grid is not None:
                    x_position = spatial_grid[x_idx]

        # Compute current_time if not provided
        if current_time is None and t_idx is not None:
            current_time = self.tSpace[t_idx] if t_idx < len(self.tSpace) else 0.0
        if current_time is None:
            current_time = 0.0

        # Convert x_position to numpy array
        x = np.atleast_1d(x_position if x_position is not None else 0.0)

        # Call class-based Hamiltonian.dm() directly: dm(x, m, p, t)
        return float(H_class.dm(x, m_at_x, p, current_time))

    def dH_dp(
        self,
        x_idx: int,
        m_at_x: float,
        derivs: dict[tuple, float] | DerivativeTensors,
        t_idx: int | None = None,
        x_position: np.ndarray | None = None,
        current_time: float | None = None,
    ) -> np.ndarray:
        """
        Hamiltonian derivative with respect to momentum dH/dp.

        Issue #673: Class-based Hamiltonian API - calls H.dp() directly.

        Args:
            x_idx: Grid/point index
            m_at_x: Density at the point
            derivs: Derivatives - DerivativeTensors or tuple-key dict
            t_idx: Time index (optional)
            x_position: Actual position coordinate (optional)
            current_time: Actual time value (optional)

        Returns:
            Array of shape (dimension,) containing dH/dp_i for each dimension.
        """
        # Get class-based Hamiltonian
        H_class = getattr(self.components, "_hamiltonian_class", None) if self.components else None
        if H_class is None:
            raise ValueError("Class-based Hamiltonian must be provided in MFGComponents.")

        # Convert derivs to numpy p array
        if isinstance(derivs, DerivativeTensors):
            derivs_dict = to_multi_index_dict(derivs)
        else:
            derivs_dict = derivs

        # Extract momentum p from derivs dict
        if derivs_dict:
            dim = max(len(k) for k in derivs_dict) if derivs_dict else 1
            p = np.zeros(dim)
            for i in range(dim):
                key = tuple(1 if j == i else 0 for j in range(dim))
                p[i] = derivs_dict.get(key, 0.0)
        else:
            p = np.zeros(1)

        # Compute x_position if not provided
        if x_position is None:
            spatial_grid = self._get_spatial_grid_internal()
            if spatial_grid is not None:
                x_position = spatial_grid[x_idx]

        # Compute current_time if not provided
        if current_time is None and t_idx is not None:
            current_time = self.tSpace[t_idx] if t_idx < len(self.tSpace) else 0.0
        if current_time is None:
            current_time = 0.0

        # Convert x_position to numpy array
        x = np.atleast_1d(x_position if x_position is not None else 0.0)

        # Call class-based Hamiltonian.dp() directly: dp(x, m, p, t)
        return H_class.dp(x, m_at_x, p, current_time)

    def get_hjb_hamiltonian_jacobian_contrib(
        self,
        U_for_jacobian_terms: np.ndarray,
        t_idx_n: int,
    ) -> HamiltonianJacobians | None:
        """
        Compute Hamiltonian Jacobian components for advanced HJB solvers.

        Returns structured Jacobian coefficients (diagonal, lower, upper) that form
        a tridiagonal matrix for Newton/policy iteration schemes.

        Issue #673: Class-based Hamiltonian API - uses H.jacobian_fd() method.

        Args:
            U_for_jacobian_terms: Current value function estimate, shape (Nx,)
            t_idx_n: Time index for evaluation

        Returns:
            HamiltonianJacobians dataclass with diagonal, lower, upper components,
            or None if not applicable.
        """
        # Issue #673: Get Jacobian from class-based Hamiltonian
        H_class = getattr(self.components, "_hamiltonian_class", None) if self.components else None
        # Issue #1068: getattr with sentinel replaces hasattr() for optional-method check.
        # Body remains pass: solver reads H.jacobian_fd() directly when it needs it.
        if H_class is not None and getattr(H_class, "jacobian_fd", None) is not None:
            # Solver will use H.jacobian_fd() directly when needed
            # Return None here - let solver handle it with full context
            pass

        # No custom jacobian - return None (solver will use FD approximation)
        return None

    def get_hjb_residual_m_coupling_term(
        self,
        M_density_at_n_plus_1: np.ndarray,
        U_n_current_guess_derivatives: dict[str, np.ndarray],
        x_idx: int,
        t_idx_n: int,
    ) -> float | None:
        """Optional coupling term for residual computation.

        Issue #673: Coupling is now part of the HamiltonianBase class.
        This method returns None - coupling handled via H.dm().
        """
        # Issue #673: Coupling is inside Hamiltonian - no separate coupling_func
        return None


# ============================================================================
# Conditions Mixin - Initial, Final, Boundary Conditions
# ============================================================================


class ConditionsMixin:
    """
    Mixin class providing initial/final/boundary condition methods.

    Provides:
    - _setup_custom_initial_density(): Initial density setup
    - _setup_custom_final_value(): Final value setup
    - get_boundary_conditions(): Boundary condition access

    Required attributes from the inheriting class:
    - components: MFGComponents | None
    - is_custom: bool
    - m_init: np.ndarray
    - u_fin: np.ndarray
    - dimension: int
    - _get_spatial_grid_internal(): method
    - _get_num_intervals(): method
    """

    # Type hints for attributes expected from MFGProblem
    components: MFGComponents | None
    is_custom: bool
    m_init: np.ndarray
    u_fin: np.ndarray
    dimension: int
    # Issue #1068: explicit None-init avoids hasattr() in using_resolved_bc().
    # Must be initialized to None in MFGProblem.__init__; see mfg_problem.py.
    _temp_resolved_bc: Any | None  # BoundaryConditions | None at runtime

    def _setup_custom_initial_density(self) -> None:
        """Setup custom initial density function m_0(x).

        Issue #684: Uses adapt_ic_callable() to handle different user signatures
        (scalar, array, spatiotemporal, expanded coordinates) transparently.
        """
        if self.components is None or self.components.m_initial is None:
            return

        m_initial = self.components.m_initial

        # Issue #681: Handle NDArray m_initial (shape validated in _initialize_functions)
        if isinstance(m_initial, np.ndarray):
            self.m_initial.flat[:] = m_initial.flat[: len(self.m_initial.flat)]
            return

        # Callable path -- adapt to user's actual signature
        from mfgarchon.utils.callable_adapter import adapt_ic_callable

        num_intervals = self._get_num_intervals()
        if self.dimension == 1 and num_intervals is not None:
            spatial_grid = self._get_spatial_grid_internal()
            sample_point: float | np.ndarray = float(spatial_grid[0, 0])

            _sig, adapted_func = adapt_ic_callable(
                m_initial,
                dimension=self.dimension,
                sample_point=sample_point,
                time_value=0.0,
            )

            for i in range(num_intervals + 1):
                x_i: float | np.ndarray = float(spatial_grid[i, 0])
                # Issue #672: Remove silent clamping - validation happens in _initialize_functions()
                self.m_initial[i] = adapted_func(x_i)
        elif self.geometry is not None:
            # Issue #777: nD path using geometry (mirrors _setup_custom_final_value)
            spatial_grid = self.geometry.get_spatial_grid()
            num_points = spatial_grid.shape[0]
            ndim = spatial_grid.shape[1] if spatial_grid.ndim > 1 else 1

            if ndim == 1:
                sample_point = float(spatial_grid[0, 0])
            else:
                sample_point = spatial_grid[0]

            _sig, adapted_func = adapt_ic_callable(
                m_initial,
                dimension=ndim,
                sample_point=sample_point,
                time_value=0.0,
            )

            for i in range(num_points):
                if ndim == 1:
                    x_i = float(spatial_grid[i, 0])
                else:
                    x_i = spatial_grid[i]
                self.m_initial.flat[i] = adapted_func(x_i)
        else:
            import warnings

            warnings.warn(
                "Cannot setup custom initial density: dimension not 1D and no geometry available. "
                "Using default m_initial initialization.",
                UserWarning,
                stacklevel=2,
            )

    def _setup_custom_final_value(self) -> None:
        """Setup custom final value function u_T(x).

        Issue #684: Uses adapt_ic_callable() to handle different user signatures
        (scalar, array, spatiotemporal, expanded coordinates) transparently.
        For spatiotemporal callables, time_value defaults to T (last time step).
        """
        if self.components is None or self.components.u_terminal is None:
            return

        u_terminal = self.components.u_terminal

        # Issue #681: Handle NDArray u_terminal (shape validated in _initialize_functions)
        if isinstance(u_terminal, np.ndarray):
            self.u_terminal.flat[:] = u_terminal.flat[: len(self.u_terminal.flat)]
            return

        # Callable path -- adapt to user's actual signature
        from mfgarchon.utils.callable_adapter import adapt_ic_callable

        # Terminal time for spatiotemporal wrappers
        terminal_time = float(self.tSpace[-1]) if len(self.tSpace) > 0 else 0.0

        num_intervals = self._get_num_intervals()
        if self.dimension == 1 and num_intervals is not None:
            spatial_grid = self._get_spatial_grid_internal()
            sample_point: float | np.ndarray = float(spatial_grid[0, 0])
            _sig, adapted_func = adapt_ic_callable(
                u_terminal,
                dimension=self.dimension,
                sample_point=sample_point,
                time_value=terminal_time,
            )
            for i in range(num_intervals + 1):
                x_i: float | np.ndarray = float(spatial_grid[i, 0])
                self.u_terminal[i] = adapted_func(x_i)
        elif self.geometry is not None:
            spatial_grid = self.geometry.get_spatial_grid()
            num_points = spatial_grid.shape[0]
            ndim = spatial_grid.shape[1] if spatial_grid.ndim > 1 else 1

            if ndim == 1:
                sample_point = float(spatial_grid[0, 0])
            else:
                sample_point = spatial_grid[0]

            _sig, adapted_func = adapt_ic_callable(
                u_terminal,
                dimension=ndim,
                sample_point=sample_point,
                time_value=terminal_time,
            )

            for i in range(num_points):
                if ndim == 1:
                    x_i = float(spatial_grid[i, 0])
                else:
                    x_i = spatial_grid[i]
                self.u_terminal.flat[i] = adapted_func(x_i)
        else:
            import warnings

            warnings.warn(
                "Cannot setup custom final value: dimension not 1D and no geometry available. "
                "Using default u_terminal initialization.",
                UserWarning,
                stacklevel=2,
            )

    def get_boundary_conditions(self) -> BoundaryConditions:
        """
        Get spatial boundary conditions for the problem.

        Resolution order (SSOT principle with legacy backward compatibility):
        1. Geometry with explicit BC - if geometry has explicitly set BC, use it (SSOT)
        2. Components (legacy) - if MFGComponents has boundary_conditions, use it
        3. Geometry default - if geometry exists, use its default BC
        4. Default - periodic BC

        This ensures both HJB and FP solvers see the same spatial BC
        when querying via problem.get_boundary_conditions().
        """
        from mfgarchon.geometry.boundary.conditions import periodic_bc

        # Check geometry availability and capabilities
        has_geometry = self.geometry is not None

        # Use contextlib.suppress for protocol duck typing (Issue #543 - geometry methods)
        has_geometry_bc_method = False
        has_explicit_bc = False

        if has_geometry:
            # Intentional: Protocol duck typing - not all geometries have BC methods
            with contextlib.suppress(AttributeError):
                has_geometry_bc_method = callable(getattr(self.geometry, "get_boundary_conditions", None))

            # Intentional: Protocol duck typing - not all geometries have explicit BC support
            with contextlib.suppress(AttributeError):
                has_explicit_bc = self.geometry.has_explicit_boundary_conditions()

        # Priority 1: Geometry with explicit BC (SSOT)
        if has_explicit_bc:
            return self.geometry.get_boundary_conditions()

        # Priority 2: Components (legacy support) - DEPRECATED
        if self.is_custom and self.components is not None and self.components.boundary_conditions is not None:
            import warnings

            warnings.warn(
                "Specifying boundary conditions via MFGComponents is deprecated. "
                "Use the geometry-first API instead:\n\n"
                "  from mfgarchon.geometry import TensorProductGrid\n"
                "  from mfgarchon.geometry.boundary import BoundaryConditions, BCSegment\n\n"
                "  bc = BoundaryConditions(segments=[...])\n"
                "  grid = TensorProductGrid(..., boundary_conditions=bc)\n"
                "  problem = MFGProblem(geometry=grid, ...)\n\n"
                "Legacy BC support via components will be removed in v1.0.0. "
                "See docs/user/GEOMETRY_FIRST_API_GUIDE.md",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.components.boundary_conditions

        # Priority 3: Geometry default (no-flux)
        if has_geometry_bc_method:
            return self.geometry.get_boundary_conditions()

        # Priority 4: Default periodic BC - IMPLICIT FALLBACK
        import warnings

        warnings.warn(
            "No boundary conditions specified. Defaulting to periodic BC. "
            "This implicit fallback is deprecated. "
            "Explicitly specify boundary conditions via geometry:\n\n"
            "  from mfgarchon.geometry.boundary import periodic_bc\n"
            "  bc = periodic_bc(dimension=...)\n"
            "  grid = TensorProductGrid(..., boundary_conditions=bc)\n\n"
            "Implicit periodic BC default will be removed in v1.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return periodic_bc(dimension=self.dimension)

    @contextlib.contextmanager
    def using_resolved_bc(self, state: dict[str, Any]):
        """
        Context manager for temporarily using resolved boundary conditions (Issue #625).

        This is the unified API for dynamic BC resolution. When BoundaryConditions
        contain BCValueProvider objects (e.g., AdjointConsistentProvider), this
        context manager resolves them to concrete values for the duration of
        the context.

        Args:
            state: Iteration state dict passed to provider.compute().
                   Standard keys: 'm_current', 'U_current', 'geometry', 'sigma'.

        Yields:
            self (the problem instance with resolved BC)

        Example:
            >>> # In FixedPointIterator
            >>> state = {'m_current': M_old, 'geometry': self.problem.geometry, ...}
            >>> with self.problem.using_resolved_bc(state):
            ...     U_new = self.hjb_solver.solve_hjb_system(...)
            >>> # BC is automatically restored after context

        Note:
            - If BC has no providers, this is a no-op (fast path)
            - Thread-safety: This modifies geometry.boundary_conditions temporarily
            - The geometry must support set_boundary_conditions() method
        """
        # Get current BC
        current_bc = self.get_boundary_conditions()

        # Fast path: no providers to resolve
        if not current_bc.has_providers():
            yield self
            return

        # Resolve providers to concrete values
        resolved_bc = current_bc.with_resolved_providers(state)

        # Store original BC and set resolved
        # Use geometry's set_boundary_conditions if available
        original_bc = None
        geometry_has_setter = False

        try:
            if self.geometry is not None:
                set_bc = getattr(self.geometry, "set_boundary_conditions", None)
                if callable(set_bc):
                    geometry_has_setter = True
                    original_bc = self.geometry.get_boundary_conditions()
                    self.geometry.set_boundary_conditions(resolved_bc)

            if not geometry_has_setter:
                # Fallback: store resolved BC in a temporary attribute.
                # Issue #1068: _temp_resolved_bc is explicitly initialized to None in
                # MFGProblem.__init__; no hasattr() needed.
                original_bc = self._temp_resolved_bc
                self._temp_resolved_bc = resolved_bc

            yield self

        finally:
            # Restore original BC
            if geometry_has_setter and self.geometry is not None:
                self.geometry.set_boundary_conditions(original_bc)
            else:
                # Issue #1068: _temp_resolved_bc is always present (None-init).
                # Restore to prior value (None if it was never set).
                self._temp_resolved_bc = original_bc

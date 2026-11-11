"""MFG Components and evaluation mixins.

This module contains:
- MFGComponents: Dataclass for custom MFG problem definition
- HamiltonianMixin: Mixin providing Hamiltonian evaluation methods (H, dH_dm, dH_dp)
- ConditionsMixin: Mixin providing initial/final/boundary condition methods

The mixin pattern allows MFGProblem to inherit these methods while keeping
the logic in separate, focused modules.
"""

from __future__ import annotations

import contextlib
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.core.derivatives import DerivativeTensors, to_multi_index_dict
from mfgarchon.types.callable_protocols import (
    CONDITION_SLOTS,
    _remembered,
    coordinate_tuple,
    keeping_what_it_froze,
    point_tuple,
)

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
        coupling=lambda m: m**2,
        coupling_dm=lambda m: 2*m,
    )
    components = MFGComponents(hamiltonian=H, m_initial=..., u_terminal=...)
    ```

    Or use Lagrangian (auto-converted via Legendre transform):

    ```python
    from mfgarchon.core.hamiltonian import LagrangianBase

    class MyLagrangian(LagrangianBase):
        def __call__(self, t, x, alpha, m):
            return 0.5 * np.sum(alpha**2)

    components = MFGComponents(lagrangian=MyLagrangian(), m_initial=..., u_terminal=...)
    ```

    Attributes
    ----------
    hamiltonian : HamiltonianBase, optional
        Class-based Hamiltonian H(t, x, p, m). Give it or ``lagrangian``, not both (#2901).
    lagrangian : LagrangianBase, optional
        The alternative to ``hamiltonian``. Its Hamiltonian is derived (``as_hamiltonian()`` for a
        ``SeparableLagrangian``, the Legendre transform otherwise) and held privately: ``hamiltonian`` stays
        ``None``, as given, and the derived Hamiltonian is read through ``MFGProblem.hamiltonian_class``
        (user ruling, 2026-11-11).
    m_initial : Callable | NDArray
        Initial density distribution m_0(x).
    u_terminal : Callable | NDArray
        Terminal value function u_T(x).
    boundary_conditions : BoundaryConditions, optional
        Boundary conditions for the domain.
    """

    # Class-based Hamiltonian or Lagrangian (Issue #673)
    hamiltonian: Any = None  # HamiltonianBase instance
    lagrangian: Any = None  # LagrangianBase instance (auto-converted)

    # Initial and terminal conditions
    m_initial: Callable | NDArray | None = None  # m_0(x): initial density
    u_terminal: Callable | NDArray | None = None  # u_T(x): terminal value function

    # The fields below are keyword-only: u_final, once the fifth field, is deleted, and a fifth positional argument
    # must not land in boundary_conditions.
    _: KW_ONLY

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

        from mfgarchon.core.hamiltonian import (
            HamiltonianBase,
            LagrangianBase,
            SeparableHamiltonian,
            SeparableLagrangian,
        )

        # Validate Lagrangian if provided
        if self.lagrangian is not None and isinstance(self.lagrangian, LagrangianBase):
            self._lagrangian_class = self.lagrangian

        # Issue #899: accept H or L. If only L is provided, derive H for the PDE solvers; if only H is provided, derive L
        # where the cost stands for H. Both together are refused, as Model refuses them (#2901): every reader but
        # HJB-SL's DPP path read H, the DPP path read L, and nothing compared the two.
        if self.hamiltonian is not None and self.lagrangian is not None:
            raise ValueError(
                f"MFGComponents was given hamiltonian={type(self.hamiltonian).__name__} and "
                f"lagrangian={type(self.lagrangian).__name__}. Provide hamiltonian OR lagrangian, not both. They are dual "
                f"descriptions of the same game; supply one, as Model requires (#2901)."
            )
        elif self.lagrangian is not None and self.hamiltonian is None:
            # L only: derive H for PDE solvers
            # The derived Hamiltonian is held in `_hamiltonian_class` (`MFGProblem.hamiltonian_class`), and the public
            # field stays as the user gave it, so a copy (`dataclasses.replace`) carries one description (#2901).
            if isinstance(self.lagrangian, LagrangianBase):
                # Issue #1068: isinstance(SeparableLagrangian) replaces hasattr(, "as_hamiltonian").
                # SeparableLagrangian is the only LagrangianBase subclass with as_hamiltonian().
                if isinstance(self.lagrangian, SeparableLagrangian):
                    # SeparableLagrangian: analytic conversion, no DualHamiltonian
                    self._hamiltonian_class = self.lagrangian.as_hamiltonian()
                else:
                    # General: numerical Legendre transform (legacy path)
                    self._hamiltonian_class = self.lagrangian.legendre_transform()
        elif self.hamiltonian is not None and self.lagrangian is None:
            # H only: derive L if separable (for ADMM/variational)
            # Issue #1068: isinstance(SeparableHamiltonian) replaces hasattr(, "control_cost")
            # and hasattr(, "_potential"). Only SeparableHamiltonian carries both attributes.
            # #2901 (user ruling, 2026-11-11): the cost stands for H only as the library ships both, so a Lagrangian
            # is derived from it only there; a subclass, or a user's cost, would have HJB-SL's DPP path solve another H.
            from mfgarchon.core.hamiltonian import why_the_cost_is_not_the_hamiltonians

            if (
                isinstance(self.hamiltonian, SeparableHamiltonian)
                and why_the_cost_is_not_the_hamiltonians(self.hamiltonian) is None
            ):
                self._lagrangian_class = SeparableLagrangian(
                    control_cost=self.hamiltonian.control_cost,
                    potential=getattr(self.hamiltonian, "_potential", None),
                    coupling=getattr(self.hamiltonian, "_coupling", None),
                )

        # Validate: at least one of H or L must be provided
        if self.hamiltonian is None and self.lagrangian is None:
            raise ValueError(
                "MFGComponents requires a Hamiltonian or Lagrangian.\n\n"
                "Example:\n"
                "  from mfgarchon.core.hamiltonian import SeparableHamiltonian, QuadraticControlCost\n\n"
                "  H = SeparableHamiltonian(\n"
                "      control_cost=QuadraticControlCost(control_cost=1.0),\n"
                "      coupling=lambda m: m**2,\n"
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
# Hamiltonian Mixin - H, dH_dm, Jacobian, Coupling
# ============================================================================


class HamiltonianMixin:
    """
    Mixin class providing Hamiltonian evaluation methods.

    Provides:
    - H(): Hamiltonian evaluation
    - dH_dm(): Hamiltonian derivative w.r.t. density
    - get_hjb_hamiltonian_jacobian_contrib(): Jacobian for Newton methods
    - get_hjb_residual_m_coupling_term(): Coupling terms

    Required attributes from the inheriting class:
    - components: MFGComponents | None
    - is_custom: bool
    - coupling_coefficient: float
    - tSpace: np.ndarray
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
    spatial_shape: tuple
    dimension: int

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
                "      coupling=lambda m: m**2,\n"
                "      coupling_dm=lambda m: 2*m,\n"
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
                "      coupling=lambda m: m**2,\n"
                "      coupling_dm=lambda m: 2*m,\n"
                "  )\n"
                "  components = MFGComponents(hamiltonian=H, m_initial=..., u_terminal=...)"
            )

    def H(
        self,
        x_idx: int,
        m_at_x: float,
        derivs: dict[tuple, float] | DerivativeTensors | None = None,
        *,
        t_idx: int | None = None,
        x_position: float | np.ndarray | None = None,
        current_time: float | None = None,
    ) -> float:
        """
        Hamiltonian function H(t, x, p, m).

        Issue #673: Class-based Hamiltonian API only - no legacy function support.

        Args:
            x_idx: Grid index (0 to Nx)
            m_at_x: Density at grid point x_idx
            derivs: Derivatives - DerivativeTensors or tuple-key dict {(1,): du/dx}
            t_idx: Time index (optional)
            x_position: Actual position coordinate (computed from x_idx if not provided)
            current_time: Actual time value (computed from t_idx if not provided)

        Returns:
            Hamiltonian value H(t, x, p, m)
        """
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
                "      coupling=lambda m: m**2,\n"
                "      coupling_dm=lambda m: 2*m,\n"
                "  )\n"
                "  components = MFGComponents(hamiltonian=H, m_initial=..., u_terminal=...)"
            )

        # Convert derivs to numpy p array for class-based H(t, x, p, m)
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

        # One point, as a tuple of scalars
        x = point_tuple(x_position if x_position is not None else np.zeros(self.dimension))

        # Call class-based Hamiltonian directly: H(t, x, p, m)
        return float(H_class(x=x, m=m_at_x, p=p, t=current_time))

    def dH_dm(
        self,
        x_idx: int,
        m_at_x: float,
        derivs: dict[tuple, float] | DerivativeTensors | None = None,
        *,
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
            t_idx: Time index (optional)
            x_position: Actual position coordinate (computed from x_idx if not provided)
            current_time: Actual time value (computed from t_idx if not provided)

        Returns:
            Derivative dH/dm at (t, x, p, m)
        """
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
                "      coupling=lambda m: m**2,\n"
                "      coupling_dm=lambda m: 2*m,\n"
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

        # One point, as a tuple of scalars
        x = point_tuple(x_position if x_position is not None else np.zeros(self.dimension))

        # Call class-based Hamiltonian.dm() directly: dm(t, x, p, m)
        return float(H_class.dm(x=x, m=m_at_x, p=p, t=current_time))

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

        # One point, as a tuple of scalars
        x = point_tuple(x_position if x_position is not None else np.zeros(self.dimension))

        # Call class-based Hamiltonian.dp() directly: dp(t, x, p, m)
        return H_class.dp(x=x, m=m_at_x, p=p, t=current_time)

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

    def _condition_on_grid(self, condition: Any, time_value: float, role: str) -> NDArray | None:
        """``condition`` on the problem's grid, read by the names it declares at ``time_value`` if it declares time
        (#2431, #2875), or None with no grid to read it on. The grid is its first batch, so its reads are compared here."""
        bound = _remembered(condition, CONDITION_SLOTS, role)
        if "x" not in bound.passes:
            advice = keeping_what_it_froze(condition, "Name its space parameter x")
            raise TypeError(
                f"{role} {getattr(condition, '__qualname__', condition)!r} takes time and no space. It is a function of "
                f"space: {advice}."
            )
        num_intervals = self._get_num_intervals()
        if self.dimension == 1 and num_intervals is not None:
            grid = np.asarray(self._get_spatial_grid_internal())[: num_intervals + 1]
        elif self.geometry is not None:
            grid = np.asarray(self.geometry.get_spatial_grid())
        else:
            return None
        ndim = grid.shape[1] if grid.ndim > 1 else 1
        return np.asarray(bound(t=time_value, x=coordinate_tuple(grid, ndim)), dtype=float)

    def _setup_custom_initial_density(self) -> None:
        """Setup custom initial density function m_0(x)."""
        if self.components is None or self.components.m_initial is None:
            return

        m_initial = self.components.m_initial

        # Issue #681: Handle NDArray m_initial (shape validated in _initialize_functions)
        if isinstance(m_initial, np.ndarray):
            self.m_initial.flat[:] = m_initial.flat[: len(self.m_initial.flat)]
            return

        # Issue #672: Remove silent clamping - validation happens in _initialize_functions()
        values = self._condition_on_grid(m_initial, 0.0, "m_initial")
        if values is None:
            import warnings

            warnings.warn(
                "Cannot setup custom initial density: dimension not 1D and no geometry available. "
                "Using default m_initial initialization.",
                UserWarning,
                stacklevel=2,
            )
            return
        self.m_initial.flat[:] = np.broadcast_to(values.ravel() if values.size > 1 else values, (self.m_initial.size,))

    def _setup_custom_final_value(self) -> None:
        """Setup custom final value function u_T(x), read at the terminal time if it declares time."""
        if self.components is None or self.components.u_terminal is None:
            return

        u_terminal = self.components.u_terminal

        # Issue #681: Handle NDArray u_terminal (shape validated in _initialize_functions)
        if isinstance(u_terminal, np.ndarray):
            self.u_terminal.flat[:] = u_terminal.flat[: len(self.u_terminal.flat)]
            return

        terminal_time = float(self.tSpace[-1]) if len(self.tSpace) > 0 else 0.0
        values = self._condition_on_grid(u_terminal, terminal_time, "u_terminal")
        if values is None:
            import warnings

            warnings.warn(
                "Cannot setup custom final value: dimension not 1D and no geometry available. "
                "Using default u_terminal initialization.",
                UserWarning,
                stacklevel=2,
            )
            return
        self.u_terminal.flat[:] = np.broadcast_to(
            values.ravel() if values.size > 1 else values, (self.u_terminal.size,)
        )

    def get_boundary_conditions(self) -> BoundaryConditions | Any:
        """The BC this problem's solvers share: the one owner of which channel supplies it (#2530, row B9).

        A solver handed its own BC uses that one (`BaseMFGSolver._lookup_boundary_conditions`); every other reader
        asks here, so the HJB and the FP of one problem read one BC. A graph geometry's node BCs are its own (#1471):
        the owner returns them, None included, and a components BC beside them is refused.

        - The geometry's BC, when the geometry declares one: ``has_explicit_boundary_conditions`` on a `Geometry`,
          and a BC from ``get_boundary_conditions`` on any other `GeometryProtocol`, whose contract answers None
          when none is declared (a point cloud: always None). A components BC
          given as well is a second declaration of the same BC, and the two must declare one condition
          (`BoundaryConditions.declares_the_same_as`); two that differ are refused by name. On the
          ``spatial_bounds=`` route the grid is built with the components BC (`MFGProblem._init_grid`), so there
          the two are one.
        - The components BC, on a geometry that declares none: an implicit domain, a point cloud, a mesh with no
          BC attached.
        - The geometry's own default, where it has one.
        - A geometry periodic on every axis, a torus, answers `periodic_bc`.
        - Otherwise no channel declares a BC, and it is refused by name (#2589).

        A FLUX in the BC it returns is refused (`refuse_shared_flux`): it is the FP's alone, given to the FP solver as
        its own BC (#2581, user ruling 2026-10-09, B-2).
        """
        from mfgarchon.geometry.base import Geometry, GraphGeometry
        from mfgarchon.geometry.boundary.bc_utils import refuse_shared_flux
        from mfgarchon.geometry.boundary.conditions import periodic_bc
        from mfgarchon.geometry.protocols import SupportsPeriodic

        geometry = self.geometry
        declared = self.components.boundary_conditions if self.components is not None else None
        if isinstance(geometry, GraphGeometry) and declared is not None:
            raise ValueError(
                "MFGProblem.get_boundary_conditions: a network's node boundary conditions are declared on its "
                "graph geometry (#1471), and components.boundary_conditions is not read on a network. Remove it, "
                "or set the node BCs on the geometry (set_boundary_conditions)."
            )
        if isinstance(geometry, Geometry):
            geometry_declares = geometry.has_explicit_boundary_conditions()
        else:
            geometry_declares = geometry is not None and geometry.get_boundary_conditions() is not None
        # One exit: every channel's BC leaves through the FLUX refusal (B-2), which passes a network's node BCs through.
        if isinstance(geometry, GraphGeometry):
            bc = geometry.get_boundary_conditions()
        elif geometry_declares:
            bc = geometry.get_boundary_conditions()
            if declared is not None and not declared.declares_the_same_as(bc):
                raise ValueError(
                    "MFGProblem.get_boundary_conditions: the problem declares its boundary conditions twice, on "
                    f"the geometry ({type(geometry).__name__}) and in components.boundary_conditions, and the two "
                    "differ. Both solvers read one BC, so one of the two would be dropped. Declare it once, on the "
                    "geometry, or give components the same BC (#2530)."
                )
        elif declared is not None:
            bc = declared
        elif geometry is not None and (own := geometry.get_boundary_conditions()) is not None:
            bc = own
        else:
            dimension = geometry.dimension if geometry is not None else None
            if not (isinstance(geometry, SupportsPeriodic) and len(geometry.periodic_dimensions) == dimension):
                raise ValueError(
                    "MFGProblem.get_boundary_conditions: no channel declares boundary conditions. The geometry "
                    f"({type(geometry).__name__}) has none, and components.boundary_conditions is unset. Declare "
                    "them on the geometry -- TensorProductGrid(..., boundary_conditions=bc), or "
                    "mesh.boundary_conditions = bc -- or, on a geometry that holds none (an implicit domain, a point "
                    "cloud), in components.boundary_conditions (#2589)."
                )
            bc = periodic_bc(dimension=dimension)
        return refuse_shared_flux(bc, consumer="MFGProblem.get_boundary_conditions")

    @contextlib.contextmanager
    def using_resolved_bc(self, state: dict[str, Any]):
        """
        Context manager for temporarily using resolved boundary conditions (Issue #625).

        This is the unified API for dynamic BC resolution. When BoundaryConditions
        contain BCValueProvider objects, this
        context manager resolves them to concrete values for the duration of
        the context.

        Args:
            state: Iteration state dict passed to provider.compute().
                   Standard keys: 'm_current', 'U_current', 'geometry', 'volatility'.

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
        # A components BC beside the geometry's is the same declaration (#2530, row B9): it is resolved with it,
        # or the owner would compare the resolved geometry BC against the unresolved one and refuse the pair.
        original_components_bc = None

        try:
            if self.geometry is not None:
                set_bc = getattr(self.geometry, "set_boundary_conditions", None)
                if callable(set_bc):
                    geometry_has_setter = True
                    original_bc = self.geometry.get_boundary_conditions()
                    self.geometry.set_boundary_conditions(resolved_bc)
                    if self.components is not None and self.components.boundary_conditions is not None:
                        original_components_bc = self.components.boundary_conditions
                        self.components.boundary_conditions = resolved_bc

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
                if original_components_bc is not None:
                    self.components.boundary_conditions = original_components_bc
            else:
                # Issue #1068: _temp_resolved_bc is always present (None-init).
                # Restore to prior value (None if it was never set).
                self._temp_resolved_bc = original_bc

"""
BC protocols, enums, and abstract base classes.

This module contains the structural foundation for the boundary condition
architecture: enums, protocols, and abstract base classes that define
the interfaces for BC application across discretization methods.

Extracted from applicator_base.py (mechanical refactor, no logic changes).
Handler protocols, merged here from a former handler_protocol module (#1711).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Callable

    from .conditions import BoundaryConditions

# =============================================================================
# Generic Type Alias for Vectorized Operations (PEP 695, Python 3.12+)
# =============================================================================
# FieldData represents any valid field value type for boundary calculations.
# Used as a constraint for generic type parameters in Calculator methods.
#
# The template pattern [T: FieldData] ensures input/output type consistency:
# - If input is float, output is float
# - If input is NDArray, output is NDArray
#
# This is equivalent to C++ template<typename T> T compute(T val)
# =============================================================================
type FieldData = float | NDArray[np.floating]


class DiscretizationType(Enum):
    """Discretization method for PDE solvers."""

    FDM = auto()  # Finite Difference Method (structured grids)
    FEM = auto()  # Finite Element Method (unstructured meshes)
    FVM = auto()  # Finite Volume Method
    GFDM = auto()  # Generalized Finite Difference Method
    DGM = auto()  # Discontinuous Galerkin Method
    MESHFREE = auto()  # Meshfree/particle methods
    GRAPH = auto()  # Graph-based discretization


class GridType(Enum):
    """Grid/mesh type affecting BC application."""

    CELL_CENTERED = auto()  # Values at cell centers (boundary at faces)
    VERTEX_CENTERED = auto()  # Values at vertices (boundary at vertices)
    STAGGERED = auto()  # Staggered grid (MAC scheme)


@runtime_checkable
class BCApplicatorProtocol(Protocol):
    """Protocol for boundary condition applicators."""

    @property
    def dimension(self) -> int:
        """Spatial dimension this applicator handles."""
        ...

    @property
    def discretization_type(self) -> DiscretizationType:
        """Discretization method this applicator is designed for."""
        ...

    def supports_bc_type(self, bc: BoundaryConditions) -> bool:
        """Check if this applicator supports the given BC specification."""
        ...


@runtime_checkable
class BoundaryCapable(Protocol):
    """
    Protocol for solvers that support boundary condition handling.

    This protocol standardizes how solvers declare and use boundary conditions,
    enabling:
    1. Capability discovery - what BC types a solver supports
    2. Configuration - setting BCs via a uniform interface
    3. Infrastructure integration - automatic applicator selection

    **Integration Levels** (from BC_CAPABILITY_MATRIX.md):
    - **Low gap**: Solver fully uses geometry/boundary/ infrastructure
    - **Medium gap**: Solver uses some infrastructure (e.g., factory functions)
    - **High gap**: Solver has custom BC handling (integration candidate)

    Example implementation for a solver:
        >>> class MyFDMSolver:
        ...     _SUPPORTED_BC_TYPES = frozenset({
        ...         BCType.DIRICHLET, BCType.NEUMANN, BCType.PERIODIC, BCType.NO_FLUX
        ...     })
        ...
        ...     @property
        ...     def supported_bc_types(self) -> frozenset[BCType]:
        ...         return self._SUPPORTED_BC_TYPES
        ...
        ...     @property
        ...     def boundary_conditions(self) -> BoundaryConditions | None:
        ...         return self._bc
        ...
        ...     @boundary_conditions.setter
        ...     def boundary_conditions(self, bc: BoundaryConditions | None) -> None:
        ...         if bc is not None:
        ...             self._validate_bc_support(bc)
        ...         self._bc = bc
        ...
        ...     @property
        ...     def discretization_type(self) -> DiscretizationType:
        ...         return DiscretizationType.FDM

    Relationship to other protocols:
        - BCApplicatorProtocol: For applicator classes that apply BCs
        - BoundaryCapable: For solver classes that USE applicators (this protocol)
        - BoundaryCalculator: For computing ghost cell values (Layer 2)
    """

    @property
    def supported_bc_types(self) -> frozenset:
        """
        BC types this solver can handle.

        Returns:
            frozenset of BCType values the solver supports.

        Note:
            Use frozenset for immutability. Implementations should define
            this as a class attribute for efficiency:
                _SUPPORTED_BC_TYPES = frozenset({BCType.DIRICHLET, BCType.NEUMANN})
        """
        ...

    @property
    def boundary_conditions(self) -> BoundaryConditions | None:
        """
        Current boundary condition configuration.

        Returns:
            BoundaryConditions object or None if not configured.
        """
        ...

    @property
    def discretization_type(self) -> DiscretizationType:
        """
        Discretization method this solver uses.

        Used to select the appropriate applicator class:
        - FDM -> FDMApplicator (ghost cells)
        - FEM -> bc_adapter.py (scikit-fem condense pattern)
        - MESHFREE/GFDM -> MeshfreeApplicator (collocation/particles)
        - GRAPH -> GraphApplicator (network BCs)
        """
        ...


# =============================================================================
# Topology/Calculator Composition (Semantic Dispatch Pattern)
# =============================================================================
# This is the structural foundation for the 2-layer BC architecture:
#   Layer 1: Topology (Memory/Indexing) - how boundaries connect
#   Layer 2: Calculator (Physics Strategy) - what values to use
#
# See Issue #516 and docs/development/bc_architecture_analysis.md
# =============================================================================


@runtime_checkable
class Topology(Protocol):
    """
    Protocol for boundary topology (Layer 1 of 2-layer BC architecture).

    Topology handles the structural/memory aspects of boundary conditions:
    - Periodic: boundaries wrap around (u[-1] = u[n-1])
    - Bounded: boundaries are physical edges requiring ghost values

    This separation allows the same Calculator (physics) to work with
    different grid connectivities.

    Example:
        >>> # Periodic domain - topology handles wrap-around
        >>> topology = PeriodicTopology(dimension=2, shape=(100, 100))
        >>> if topology.is_periodic:
        ...     # Use np.pad(..., mode='wrap')
        ...     pass

        >>> # Bounded domain - need Calculator for ghost values
        >>> topology = BoundedTopology(dimension=2, shape=(100, 100))
        >>> if not topology.is_periodic:
        ...     # Apply calculator.compute() for each boundary
        ...     pass
    """

    @property
    def is_periodic(self) -> bool:
        """Whether this topology has periodic boundaries."""
        ...

    @property
    def dimension(self) -> int:
        """Spatial dimension."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Grid shape (interior points only)."""
        ...


@runtime_checkable
class BoundaryCalculator(Protocol):
    """
    Protocol for ghost value computation (Layer 2 of 2-layer BC architecture).

    Calculator handles the physics/value aspects of boundary conditions:
    - Dirichlet: u_ghost = 2*g - u_interior
    - Neumann: u_ghost = u_interior + 2*dx*g
    - Robin: combination of above
    - Extrapolation: polynomial continuation

    The Calculator is ONLY used for bounded topologies. Periodic topologies
    compute ghost values directly from wrap-around indices.

    **VECTORIZATION**: All calculators support both scalar and array inputs.
    When interior_value is an NDArray, the output is also an NDArray of the
    same shape. This enables efficient vectorized operations without Python loops.

    **TYPE SAFETY**: Uses TypeVar (template pattern) to ensure input/output type
    consistency. If input is float, output is float. If input is NDArray, output
    is NDArray. This is equivalent to C++ template<typename T> T compute(T val).

    Example:
        >>> # Scalar usage
        >>> calculator = DirichletCalculator(boundary_value=0.0)
        >>> u_ghost = calculator.compute(interior_value=1.0, dx=0.1, side='min')
        # Returns -1.0 (float)

        >>> # Vectorized usage (100x-1000x faster for large arrays)
        >>> interior = np.array([1.0, 2.0, 3.0])
        >>> u_ghost = calculator.compute(interior_value=interior, dx=0.1, side='min')
        # Returns array([-1.0, -2.0, -3.0]) (NDArray)
    """

    def compute[T: FieldData](
        self,
        interior_value: T,
        dx: float,
        side: str,
        **kwargs,
    ) -> T:
        """
        Compute ghost cell value from interior value (vectorized).

        Args:
            interior_value: Value(s) at interior point(s) adjacent to boundary.
                           Can be scalar float or NDArray for vectorized operation.
            dx: Grid spacing
            side: Boundary side ('min' or 'max')
            **kwargs: Additional parameters (e.g., time for time-varying BCs)

        Returns:
            Ghost cell value(s). Same type/shape as interior_value (template pattern).
        """
        ...


# =============================================================================
# Base Applicator Classes
# =============================================================================


class BaseBCApplicator(ABC):
    """
    Abstract base class for all BC applicators.

    Provides common infrastructure:
    - Dimension and discretization type properties
    - BC validation helpers
    - Dimension dispatch logic (optional)

    Subclasses implement discretization-specific application methods.
    """

    def __init__(self, dimension: int | None = None):
        """
        Initialize applicator.

        Args:
            dimension: Spatial dimension (1, 2, 3, or higher). None for dimension-agnostic.
        """
        self._dimension = dimension

    @property
    def dimension(self) -> int | None:
        """Spatial dimension (None if dimension-agnostic)."""
        return self._dimension

    @property
    @abstractmethod
    def discretization_type(self) -> DiscretizationType:
        """Discretization method this applicator handles."""
        ...

    def supports_bc_type(self, bc: BoundaryConditions) -> bool:
        """
        Check if this applicator supports the given BC.

        Default implementation checks dimension match.
        Subclasses can add additional checks.
        """
        if self._dimension is None:
            return True  # Dimension-agnostic
        return bc.dimension == self._dimension

    def _validate_bc(self, bc: BoundaryConditions) -> None:
        """Validate BC specification against this applicator."""
        if self._dimension is not None and bc.dimension != self._dimension:
            raise ValueError(f"BC dimension ({bc.dimension}) does not match applicator dimension ({self._dimension})")

    def __repr__(self) -> str:
        dim_str = f"dimension={self._dimension}" if self._dimension else "dimension-agnostic"
        return f"{self.__class__.__name__}({dim_str}, {self.discretization_type.name})"


# =============================================================================
# Structured Grid (FDM) Base Class
# =============================================================================


class BaseStructuredApplicator(BaseBCApplicator):
    """
    Base class for structured grid (FDM) applicators.

    Uses ghost cell method for BC application on regular grids.
    Supports dimension dispatch for optimized 1D/2D/3D implementations.

    This class provides shared infrastructure for ghost cell computation
    following the Template Method Pattern (Issue #598):
    - Shared validation logic
    - Shared ghost cell formulas (Dirichlet, Neumann, Robin)
    - Shared buffer creation utilities
    - Dimension-specific slicing in subclasses

    Usage:
        Subclasses override _fill_ghost_cells() for dimension-specific logic
        but use shared formula methods for ghost cell values.
    """

    def __init__(self, dimension: int, grid_type: GridType = GridType.CELL_CENTERED):
        """
        Initialize structured grid applicator.

        Args:
            dimension: Spatial dimension (1, 2, 3, or higher)
            grid_type: Grid type (cell-centered, vertex-centered, staggered)
        """
        super().__init__(dimension)
        self._grid_type = grid_type

    @property
    def discretization_type(self) -> DiscretizationType:
        return DiscretizationType.FDM

    @property
    def grid_type(self) -> GridType:
        """Grid type."""
        return self._grid_type

    # =========================================================================
    # Shared Ghost Cell Formula Methods (Issue #598 - DRY Principle)
    # =========================================================================

    def _compute_ghost_dirichlet(
        self,
        u_interior: NDArray | float,
        g: float | Callable[[float], float],
        time: float = 0.0,
    ) -> NDArray | float:
        """
        Compute Dirichlet ghost cell value (shared formula).

        For cell-centered grid with boundary at cell face:
            u_boundary = (u_ghost + u_interior) / 2 = g
            => u_ghost = 2*g - u_interior

        For vertex-centered grid with boundary at grid point:
            u_ghost = g

        Args:
            u_interior: Interior values adjacent to boundary
            g: Boundary value (constant or callable)
            time: Current time (for callable g)

        Returns:
            Ghost cell values with same type as u_interior

        Note:
            This formula is shared across 1D/2D/3D/nD to eliminate duplication.
            Previously duplicated at applicator_fdm.py:232, 1255, 1725.
        """
        # Evaluate callable BC values
        if callable(g):
            g_val = g(time)
        else:
            g_val = g

        # Apply formula based on grid type
        if self._grid_type == GridType.CELL_CENTERED:
            return 2.0 * g_val - u_interior
        else:  # VERTEX_CENTERED
            if isinstance(u_interior, np.ndarray):
                return np.full_like(u_interior, g_val)
            else:
                return g_val

    def _compute_ghost_neumann(
        self,
        u_interior: NDArray | float,
        u_next_interior: NDArray | float,
        g: float | Callable[[float], float],
        dx: float,
        side: str,
        time: float = 0.0,
    ) -> NDArray | float:
        """
        Compute Neumann ghost cell value (shared formula).

        For central difference gradient stencil with cell-centered grid:
        - Left boundary (normal points left/inward):
          du/dn = (u_interior - u_ghost) / (2*dx) = -g  (inward normal)
          => u_ghost = u_interior + 2*dx*g
          For zero-flux (g=0): u_ghost = u_next_interior (reflection)

        - Right boundary (normal points right/outward):
          du/dn = (u_ghost - u_interior) / (2*dx) = g  (outward normal)
          => u_ghost = u_interior + 2*dx*g
          For zero-flux (g=0): u_ghost = u_prev_interior (reflection)

        Args:
            u_interior: Interior values adjacent to boundary
            u_next_interior: Interior values one step inward from boundary
            g: Flux value (constant or callable)
            dx: Grid spacing
            side: Boundary side ("left" or "right")
            time: Current time (for callable g)

        Returns:
            Ghost cell values with same type as u_interior

        Note:
            Issue #542 fix: Uses reflection formula (ghost = u_next_interior)
            for central difference stencil, not edge extension (ghost = u_interior).

            This formula is shared across 1D/2D/3D/nD to eliminate duplication.
            Previously duplicated at applicator_fdm.py:1099, 1274, 1748.
        """
        # Evaluate callable BC values
        if callable(g):
            g_val = g(time)
        else:
            g_val = g

        # Zero-flux case: reflection (Issue #542 fix)
        if np.isclose(g_val, 0.0):
            return u_next_interior.copy() if isinstance(u_next_interior, np.ndarray) else u_next_interior

        # General Neumann case
        if side == "left":
            # Left boundary: u_ghost = u_next_interior - 2*dx*g
            return u_next_interior - 2.0 * dx * g_val
        else:  # "right"
            # Right boundary: u_ghost = u_next_interior + 2*dx*g
            return u_next_interior + 2.0 * dx * g_val

    def _compute_ghost_robin(
        self,
        u_interior: NDArray | float,
        alpha: float,
        beta: float,
        g: float | Callable[[float], float],
        dx: float,
        side: str,
        time: float = 0.0,
    ) -> NDArray | float:
        """
        Compute Robin ghost cell value (shared formula).

        Robin BC: alpha*u + beta*du/dn = g at boundary

        For cell-centered grid with boundary at cell face (ghost at -dx/2, interior at +dx/2):
            u_boundary = (u_ghost + u_interior)/2
            du/dn = (u_ghost - u_interior)/dx  (cell centers are dx apart)

            alpha * (u_ghost + u_interior)/2 + beta * (u_ghost - u_interior)/dx = g
            => u_ghost * (alpha/2 + beta/dx) = g - u_interior * (alpha/2 - beta/dx)

        Args:
            u_interior: Interior values adjacent to boundary
            alpha: Robin coefficient for value term
            beta: Robin coefficient for flux term
            g: Robin boundary value (constant or callable)
            dx: Grid spacing
            side: Boundary side ("left" or "right")
            time: Current time (for callable g)

        Returns:
            Ghost cell values with same type as u_interior

        Note:
            This formula is shared across 1D/2D/3D/nD to eliminate duplication.
            Fix (commit 0ae5515a): Changed from /(2*dx) to /dx.
        """
        # Evaluate callable BC values
        if callable(g):
            g_val = g(time)
        else:
            g_val = g

        # Robin formula (valid for both left and right boundaries)
        # Cell centers are dx apart, not 2*dx
        coeff_ghost = alpha / 2.0 + beta / dx
        coeff_interior = alpha / 2.0 - beta / dx

        return (g_val - u_interior * coeff_interior) / coeff_ghost

    # =========================================================================
    # Shared Validation and Utility Methods (Issue #598)
    # =========================================================================

    def _validate_field(self, field: NDArray) -> None:
        """
        Validate field before BC application (shared logic).

        Checks:
        - Field contains finite values (no NaN/Inf)

        Args:
            field: Interior field to validate

        Raises:
            ValueError: If validation fails

        Note:
            This validation is shared across all dimension-specific functions
            to eliminate duplication. Previously duplicated at applicator_fdm.py:1247.
        """
        if not np.isfinite(field).all():
            raise ValueError("Field contains NaN or Inf values. Check solver convergence and boundary conditions.")

    def _create_padded_buffer(
        self,
        field: NDArray,
        ghost_depth: int = 1,
    ) -> NDArray:
        """
        Create zero-initialized padded buffer (shared logic).

        Args:
            field: Interior field
            ghost_depth: Number of ghost cells per side (default: 1)

        Returns:
            Zero-initialized padded array with shape (N1+2*depth, N2+2*depth, ...)

        Note:
            Interior values are copied to padded buffer.
            Ghost cells initialized to zero (will be filled by BC application).
        """
        # Compute padded shape
        padded_shape = tuple(n + 2 * ghost_depth for n in field.shape)

        # Create zero buffer
        padded = np.zeros(padded_shape, dtype=field.dtype)

        # Copy interior values
        interior_slices = tuple(slice(ghost_depth, -ghost_depth) for _ in range(field.ndim))
        padded[interior_slices] = field

        return padded

    def _compute_grid_spacing(
        self,
        field: NDArray,
        domain_bounds: NDArray,
    ) -> tuple[float, ...]:
        """
        Compute grid spacing from domain bounds (shared logic).

        Args:
            field: Interior field (to get shape)
            domain_bounds: Domain bounds shape (ndim, 2)

        Returns:
            Grid spacing for each dimension

        Note:
            This computation is shared across dimension-specific functions
            to eliminate duplication. Previously duplicated at applicator_fdm.py:2664.
        """
        domain_bounds = np.atleast_2d(domain_bounds)
        spacing = []

        for d in range(field.ndim):
            extent = domain_bounds[d, 1] - domain_bounds[d, 0]
            n_points = field.shape[d]
            dx = extent / (n_points - 1) if n_points > 1 else extent
            spacing.append(dx)

        return tuple(spacing)


# =============================================================================
# Unstructured Mesh (FEM) Base Class
# =============================================================================


class BaseUnstructuredApplicator(BaseBCApplicator):
    """
    Base class for unstructured mesh (FEM) applicators.

    Applies BCs by modifying system matrices and RHS vectors.
    """

    def __init__(self, dimension: int):
        """
        Initialize FEM applicator.

        Args:
            dimension: Spatial dimension (1, 2, or 3)
        """
        if dimension not in (1, 2, 3):
            raise ValueError(f"FEM applicator supports 1D, 2D, and 3D, got {dimension}D")
        super().__init__(dimension)

    @property
    def discretization_type(self) -> DiscretizationType:
        return DiscretizationType.FEM


# =============================================================================
# Meshfree Base Class
# =============================================================================


class BaseMeshfreeApplicator(BaseBCApplicator):
    """
    Base class for meshfree (particle/collocation) applicators.

    Handles boundary conditions for:
    - Particle methods (Lagrangian FP solver)
    - Collocation methods (RBF, GFDM)
    - High-dimensional implicit domains
    """

    def __init__(self, dimension: int | None = None):
        """
        Initialize meshfree applicator.

        Args:
            dimension: Spatial dimension (None for dimension from geometry)
        """
        super().__init__(dimension)

    @property
    def discretization_type(self) -> DiscretizationType:
        return DiscretizationType.MESHFREE


# =============================================================================
# Graph Base Class
# =============================================================================


class BaseGraphApplicator(BaseBCApplicator):
    """
    Base class for graph-based (network/maze) applicators.

    Handles boundary conditions on discrete graph structures:
    - Network MFG (GridNetwork, RandomNetwork, ScaleFreeNetwork)
    - Maze MFG (MazeGeometry, VoronoiMaze)
    - General graph domains
    """

    def __init__(self, num_nodes: int):
        """
        Initialize graph applicator.

        Args:
            num_nodes: Number of nodes in the graph
        """
        super().__init__(dimension=None)  # Graphs are dimension-agnostic
        self._num_nodes = num_nodes

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self._num_nodes

    @property
    def discretization_type(self) -> DiscretizationType:
        return DiscretizationType.GRAPH

    def supports_bc_type(self, bc: BoundaryConditions) -> bool:
        """Graph applicators are dimension-agnostic."""
        return True


# =============================================================================
# Boundary Handler Protocols (merged here; the separate module is gone, #1711)
# =============================================================================


@runtime_checkable
class BoundaryHandler(Protocol):
    """
    Protocol for solver boundary condition handling.

    This protocol defines the minimal interface that solvers must implement
    to handle boundary conditions in a consistent way.

    Responsibilities:
    - Identify boundary points in solver's discretization
    - Apply BCs using solver-specific enforcement method
    - Provide BC type information for algorithmic decisions

    NOT Responsible for:
    - BC segment matching (handled by BCSegment.matches_point())
    - Boundary normal computation (handled by geometry)
    - BC value interpolation (handled by BoundaryConditions)
    """

    def get_boundary_indices(self) -> NDArray[np.integer]:
        """
        Identify boundary points in solver's discretization.

        Returns:
            Array of integer indices identifying boundary points in the
            solver's point set (grid points, collocation points, particles, etc.).

        Notes:
            - For grid-based methods: Indices of grid boundary
            - For meshfree methods: Indices of collocation points near domain boundary
            - For particle methods: Indices of particles on/near boundary

        Example:
            ```python
            boundary_idx = solver.get_boundary_indices()
            # array([0, 1, 2, ..., N-3, N-2, N-1])  # Boundary points
            ```
        """
        ...

    def apply_boundary_conditions(
        self,
        values: NDArray,
        bc: BoundaryConditions,
        time: float = 0.0,
    ) -> NDArray:
        """
        Apply boundary conditions to solution values.

        This is the main method that solvers implement to enforce BCs
        using their specific discretization method.

        Args:
            values: Solution values at all discretization points (N,)
            bc: Boundary conditions object from geometry
            time: Current time for time-dependent BCs

        Returns:
            Modified solution values with BCs enforced (N,)

        Enforcement Methods by Solver Type:
            - **FDM**: Ghost cells, one-sided stencils
            - **GFDM**: Ghost nodes, stencil rotation, penalty weights
            - **Particle**: Reflection, resampling
            - **FEM**: Essential BCs (modify system), Natural BCs (weak form)
            - **Semi-Lagrangian**: Interpolation with BC constraints

        Example:
            ```python
            # FDM solver
            u = np.zeros(N)
            u_with_bc = solver.apply_boundary_conditions(u, bc, time=t)
            # u_with_bc[boundary_idx] now satisfies Dirichlet/Neumann conditions
            ```
        """
        ...

    def get_bc_type_for_point(self, point_idx: int) -> str:
        """
        Determine BC type (dirichlet/neumann/periodic) for a specific point.

        Args:
            point_idx: Index of point in solver's discretization

        Returns:
            BC type string: "dirichlet", "neumann", "periodic", or "none"

        Notes:
            - Used for algorithmic decisions (stencil selection, interpolation)
            - Queries geometry BC segments to determine type
            - Returns "none" if point is not on boundary

        Example:
            ```python
            bc_type = solver.get_bc_type_for_point(idx=0)
            if bc_type == "neumann":
                # Use gradient-based stencil
                deriv_stencil = build_neumann_stencil(idx)
            ```
        """
        ...


# Optional extension for advanced BC handling
@runtime_checkable
class AdvancedBoundaryHandler(BoundaryHandler, Protocol):
    """
    Extended protocol for solvers with advanced BC features.

    Adds optional methods for:
    - Normal vector computation
    - Mixed/Robin BC support
    - Time-dependent BC caching
    """

    def get_boundary_normals(self) -> NDArray:
        """
        Compute outward-pointing normal vectors at boundary points.

        Returns:
            Array of normal vectors (n_boundary, dimension)

        Notes:
            - For rectangular domains: +/-e_i directions
            - For general domains: Computed from SDF or mesh geometry
            - Used for Neumann BC and coordinate rotation (GFDM)

        Example:
            ```python
            normals = solver.get_boundary_normals()
            # array([[-1, 0], [-1, 0], ..., [1, 0], [1, 0]])  # 2D box
            ```
        """
        ...

    def apply_robin_bc(
        self,
        values: NDArray,
        alpha: float,
        beta: float,
        gamma: float,
        time: float = 0.0,
    ) -> NDArray:
        """
        Apply Robin boundary conditions: alpha u + beta du/dn = gamma.

        Args:
            values: Solution values (N,)
            alpha: Coefficient for u term
            beta: Coefficient for du/dn term
            gamma: Right-hand side value
            time: Current time

        Returns:
            Modified solution values with Robin BC enforced (N,)

        Notes:
            - Generalization of Dirichlet (beta=0) and Neumann (alpha=0)
            - Requires both value and gradient enforcement
            - Not all solvers support Robin BCs efficiently
        """
        ...


def validate_boundary_handler(solver) -> bool:
    """
    Runtime check if solver implements BoundaryHandler protocol.

    Args:
        solver: Solver instance to validate

    Returns:
        True if solver implements the protocol, False otherwise

    Example:
        ```python
        from mfgarchon.geometry.boundary import validate_boundary_handler

        if validate_boundary_handler(my_solver):
            # Safe to use unified BC workflow
            boundary_idx = my_solver.get_boundary_indices()
        else:
            # Fallback to solver-specific BC handling
            pass
        ```
    """
    return isinstance(solver, BoundaryHandler)


__all__ = [
    # Type alias
    "FieldData",
    # Enums
    "DiscretizationType",
    "GridType",
    # Protocols
    "BCApplicatorProtocol",
    "BoundaryCapable",
    "Topology",
    "BoundaryCalculator",
    # Base classes
    "BaseBCApplicator",
    "BaseStructuredApplicator",
    "BaseUnstructuredApplicator",
    "BaseMeshfreeApplicator",
    "BaseGraphApplicator",
    # Handler protocols (merged here; the separate module is gone, #1711)
    "BoundaryHandler",
    "AdvancedBoundaryHandler",
    "validate_boundary_handler",
]

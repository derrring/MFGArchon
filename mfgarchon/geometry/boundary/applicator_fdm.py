"""
Boundary Condition Applicator for PDE Solvers.

This module provides utilities for applying mixed boundary conditions to grid-based
fields. Supports:

1. Uniform BCs (same type on all boundaries)
2. Mixed BCs (different types on different boundary segments)
3. Multiple BC types: Dirichlet, Neumann, Robin, Periodic
4. Time-dependent boundary values
5. Cell-centered and vertex-centered grids

The ghost cell formulas are derived for standard 3-point FD stencils.

Ghost Cell Formulas (cell-centered grid, boundary at cell face):
--------------------------------------------------------------
Let u_i = interior point, u_g = ghost point, u_b = boundary value
Boundary is located at midpoint: x_b = (x_g + x_i) / 2

Dirichlet (u = g at boundary):
    u_b = (u_g + u_i) / 2 = g
    => u_g = 2*g - u_i

Neumann (du/dn = g at boundary, outward normal):
    (u_i - u_g) / (2*dx) = g  (for left/bottom boundary, normal points inward)
    => u_g = u_i - 2*dx*g

    (u_g - u_i) / (2*dx) = g  (for right/top boundary, normal points outward)
    => u_g = u_i + 2*dx*g

Robin (alpha*u + beta*du/dn = g at boundary):
    alpha * (u_g + u_i)/2 + beta * (u_g - u_i)/(2*dx) = g
    => u_g * (alpha/2 + beta/(2*dx)) = g - u_i * (alpha/2 - beta/(2*dx))
    => u_g = (g - u_i * (alpha/2 - beta/(2*dx))) / (alpha/2 + beta/(2*dx))

Corner Handling (Issue #521):
-----------------------------
FDM uses **sequential dimension updates** which implicitly handle corners without
needing the geometry's `corner_strategy` configuration. The ghost cell values at
corners are filled by applying face BCs for each dimension in sequence:

    1. Apply BC for dimension 0 (fills x-face ghosts including corner columns)
    2. Apply BC for dimension 1 (fills y-face ghosts including corner rows)
    3. ...and so on for higher dimensions

At corner ghost cells, the last dimension's BC "wins" and writes the final value.
This is equivalent to a 'priority' corner strategy via dimension ordering.

This approach works correctly for FDM because:
- Finite difference stencils at corners use face ghost values, not corner ghosts
- Corner ghost values only need to be finite (not NaN) for numerical stability
- The sequential update is deterministic and consistent

Other methods (particles, meshfree) use `geometry.get_boundary_normal(corner_strategy=)`
to get proper diagonal normals at corners. FDM opts out of this mechanism entirely.

Reference: See Issue #521 for corner strategy design and implementation details.

Usage:
    from mfgarchon.geometry.boundary import apply_boundary_conditions_2d

    # With uniform BCs
    m_padded = apply_boundary_conditions_2d(m, uniform_bc, domain_bounds)

    # With mixed BCs and time
    m_padded = apply_boundary_conditions_2d(m, mixed_bc, domain_bounds, time=0.5)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.utils.mfg_logging import get_logger

from .applicator_base import (
    BaseStructuredApplicator,
    BoundaryCalculator,
    BoundedTopology,
    DirichletCalculator,
    GridType,
    LinearExtrapolationCalculator,
    NeumannCalculator,
    PeriodicTopology,
    QuadraticExtrapolationCalculator,
    RobinCalculator,
    Topology,
    ZeroFluxCalculator,
    ZeroGradientCalculator,
)
from .conditions import BoundaryConditions
from .enforcement import enforce_dirichlet_value_nd, enforce_neumann_value_nd
from .fdm_bc_1d import BoundaryConditions as BoundaryConditions1DFDM
from .ghost_cells import GhostCellConfig
from .types import BCSegment, BCType, BoundaryFace, parse_boundary_face

logger = get_logger(__name__)

# Backward compatibility alias
LegacyBoundaryConditions1D = BoundaryConditions1DFDM

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Class-Based FDM Applicator
# =============================================================================


class FDMApplicator(BaseStructuredApplicator):
    """
    Finite Difference Method boundary condition applicator.

    Provides a class-based interface for applying ghost cell boundary conditions
    to structured grids. Supports 1D, 2D, 3D, and higher dimensions with
    dimension-specific optimizations.

    Inherits from BaseStructuredApplicator for consistent interface across
    all BC applicators.

    Usage:
        # Create applicator for 2D problems
        applicator = FDMApplicator(dimension=2)

        # Apply BCs to a field
        padded = applicator.apply(field, bc, dx)

        # Or use static methods directly
        padded = FDMApplicator.apply_1d(field, bc)
        padded = FDMApplicator.apply_2d(field, bc)
    """

    def __init__(
        self,
        dimension: int,
        grid_type: GridType | str = GridType.CELL_CENTERED,
    ):
        """
        Initialize FDM applicator.

        Args:
            dimension: Spatial dimension (1, 2, 3, or higher)
            grid_type: Grid type (GridType enum or string for backward compat)
        """
        # Handle both enum and string for backward compatibility
        if isinstance(grid_type, str):
            grid_type_enum = GridType.VERTEX_CENTERED if grid_type == "vertex_centered" else GridType.CELL_CENTERED
        else:
            grid_type_enum = grid_type

        super().__init__(dimension, grid_type_enum)
        self._config = GhostCellConfig(grid_type=grid_type_enum)

    @property
    def grid_type(self) -> GridType:
        """Grid type enum."""
        return self._config.grid_type

    @property
    def grid_type_str(self) -> str:
        """Grid type as string (for backward compatibility)."""
        return "vertex_centered" if self._config.is_vertex_centered else "cell_centered"

    def apply(
        self,
        field: NDArray[np.floating],
        boundary_conditions: BoundaryConditions | LegacyBoundaryConditions1D,
        grid_spacing: float | tuple[float, ...] | None = None,
        domain_bounds: NDArray[np.floating] | None = None,
        time: float = 0.0,
        geometry=None,  # Type: SupportsRegionMarking | None (Issue #596 Phase 2.5)
    ) -> NDArray[np.floating]:
        """
        Apply boundary conditions to a field.

        Automatically dispatches to dimension-specific implementation.

        Args:
            field: Interior field values
            boundary_conditions: BC specification
            grid_spacing: Grid spacing (not used for ghost cells, but kept for API consistency)
            domain_bounds: Domain bounds (required for mixed BCs)
            time: Current time for time-dependent BCs
            geometry: Geometry object with marked regions (Issue #596 Phase 2.5).
                     Required if boundary_conditions uses region_name.

        Returns:
            Padded field with ghost cells
        """
        # Issue #577 Phase 3: Use pad_array_with_ghosts() for all BCs
        # Geometry parameter enables region_name resolution for mixed BCs
        return pad_array_with_ghosts(field, boundary_conditions, ghost_depth=1, time=time, geometry=geometry)

    @staticmethod
    def apply_1d(
        field: NDArray[np.floating],
        boundary_conditions: BoundaryConditions | LegacyBoundaryConditions1D,
        domain_bounds: NDArray[np.floating] | None = None,
        time: float = 0.0,
        config: GhostCellConfig | None = None,
    ) -> NDArray[np.floating]:
        """Static method for 1D BC application."""
        # Issue #645: Use dimension-agnostic pad_array_with_ghosts()
        return pad_array_with_ghosts(field, boundary_conditions, ghost_depth=1, time=time)

    @staticmethod
    def apply_2d(
        field: NDArray[np.floating],
        boundary_conditions: BoundaryConditions | LegacyBoundaryConditions1D,
        domain_bounds: NDArray[np.floating] | None = None,
        time: float = 0.0,
        config: GhostCellConfig | None = None,
    ) -> NDArray[np.floating]:
        """Static method for 2D BC application."""
        # Issue #645: Use dimension-agnostic pad_array_with_ghosts()
        return pad_array_with_ghosts(field, boundary_conditions, ghost_depth=1, time=time)

    @staticmethod
    def apply_3d(
        field: NDArray[np.floating],
        boundary_conditions: BoundaryConditions | LegacyBoundaryConditions1D,
        domain_bounds: NDArray[np.floating] | None = None,
        time: float = 0.0,
        config: GhostCellConfig | None = None,
    ) -> NDArray[np.floating]:
        """Static method for 3D BC application."""
        # Issue #645: Use dimension-agnostic pad_array_with_ghosts()
        return pad_array_with_ghosts(field, boundary_conditions, ghost_depth=1, time=time)

    @staticmethod
    def apply_nd(
        field: NDArray[np.floating],
        boundary_conditions: BoundaryConditions | LegacyBoundaryConditions1D,
        domain_bounds: NDArray[np.floating] | None = None,
        time: float = 0.0,
        config: GhostCellConfig | None = None,
    ) -> NDArray[np.floating]:
        """Static method for nD BC application."""
        # Issue #645: Use dimension-agnostic pad_array_with_ghosts()
        return pad_array_with_ghosts(field, boundary_conditions, ghost_depth=1, time=time)

    def enforce_values(
        self,
        field: NDArray[np.floating],
        boundary_conditions: BoundaryConditions,
        spacing: tuple[float, ...] | NDArray[np.floating],
        time: float = 0.0,
    ) -> NDArray[np.floating]:
        """
        Enforce boundary condition values on solution array (Issue #542).

        This method sets boundary values to satisfy Dirichlet/Neumann BC specifications.
        Complements ghost cell application (which enables derivative computation).

        **Distinction**:
        - `apply()`: Returns padded array with ghost cells for computing ∇u
        - `enforce_values()`: Sets boundary values to satisfy BC (u(boundary) = g)

        Args:
            field: Solution array to enforce BC on (shape: (N1, N2, ..., Nd))
            boundary_conditions: BC specification (must be BoundaryConditions, not legacy)
            spacing: Grid spacing for each dimension (needed for Neumann BC)
            time: Current time for time-dependent BC values

        Returns:
            Solution array with BC enforced at boundaries (modified in-place, also returned)

        Example:
            >>> applicator = FDMApplicator(dimension=2)
            >>> U = solver.solve_timestep(...)
            >>> U = applicator.enforce_values(U, bc, grid.spacing, time=0.5)
        """
        if not isinstance(boundary_conditions, BoundaryConditions):
            raise TypeError(
                f"enforce_values() requires BoundaryConditions, got {type(boundary_conditions).__name__}. "
                f"Legacy BC types should be converted first."
            )

        # Convert spacing to tuple if needed
        if isinstance(spacing, np.ndarray):
            spacing_tuple = tuple(spacing)
        else:
            spacing_tuple = spacing

        # Import BC types
        from mfgarchon.geometry.boundary.types import BCType

        # Enforce BC for each segment
        for segment in boundary_conditions.segments:
            # Parse boundary identifier to get dimension and side
            boundary_id = segment.boundary

            # If boundary_id is None, this is a uniform BC - apply to all boundaries
            if boundary_id is None:
                # Determine field dimension
                field_ndim = field.ndim
                # Apply to all standard boundaries (x_min, x_max, y_min, y_max, etc.)
                boundaries_to_apply = []
                for d in range(field_ndim):
                    boundaries_to_apply.append((d, "min"))
                    boundaries_to_apply.append((d, "max"))
            else:
                # Map boundary string to (dimension, side)
                dim, side = self._parse_boundary_identifier(boundary_id)
                if dim is None:
                    continue  # Unrecognized boundary format
                boundaries_to_apply = [(dim, side)]

            # Apply BC to each boundary
            for dim, side in boundaries_to_apply:
                # Get BC value (time-dependent or constant)
                if callable(segment.value):
                    bc_value = segment.value(time)
                else:
                    bc_value = segment.value

                # Get grid spacing for this dimension
                h = spacing_tuple[dim] if dim < len(spacing_tuple) else 1.0

                # Enforce BC based on type
                if segment.bc_type == BCType.DIRICHLET:
                    # Dirichlet: Set boundary values directly u(boundary) = g
                    self._apply_dirichlet_enforcement(field, dim, side, bc_value)

                elif segment.bc_type == BCType.NEUMANN:
                    # Neumann: Set boundary value to satisfy gradient constraint ∂u/∂n = g
                    self._apply_neumann_enforcement(field, dim, side, bc_value, h)

        return field

    def _parse_boundary_identifier(self, boundary_id: str) -> tuple[int | None, str | None]:
        """
        Parse boundary identifier string to (dimension, side).

        Uses BoundaryFace for dimension-agnostic parsing (Issue #946).

        Args:
            boundary_id: Boundary identifier string

        Returns:
            (dimension_index, 'min' or 'max') or (None, None) if unrecognized
        """
        face = parse_boundary_face(boundary_id)
        if face is None:
            return None, None
        return face.axis, face.side

    def _apply_dirichlet_enforcement(self, field: NDArray[np.floating], dim: int, side: str, value: float) -> None:
        """
        Apply Dirichlet BC enforcement to nD array along specified dimension and side.

        Delegates to shared enforcement utilities (Issue #636).

        Args:
            field: Solution array (modified in-place)
            dim: Dimension index (0=x, 1=y, 2=z)
            side: 'min' or 'max'
            value: Dirichlet boundary value
        """
        enforce_dirichlet_value_nd(field, dim, side, value)

    def _apply_neumann_enforcement(
        self, field: NDArray[np.floating], dim: int, side: str, grad_value: float, h: float
    ) -> None:
        """
        Apply Neumann BC enforcement to nD array along specified dimension and side.

        Delegates to shared enforcement utilities (Issue #636).
        FDM uses order=1 (first-order) since ghost cell scheme is already O(h²) accurate.

        Args:
            field: Solution array (modified in-place)
            dim: Dimension index (0=x, 1=y, 2=z)
            side: 'min' or 'max'
            grad_value: Neumann BC gradient value
            h: Grid spacing in this dimension
        """
        # FDM uses order=1 for Neumann enforcement (sufficient accuracy)
        enforce_neumann_value_nd(field, dim, side, grad_value=grad_value, spacing=h, order=1)


# =============================================================================
# Ghost Buffer with Topology/Calculator Composition (Issue #516)
# =============================================================================


def bc_to_topology_calculator(
    bc: BoundaryConditions | LegacyBoundaryConditions1D,
    shape: tuple[int, ...],
    grid_type: GridType = GridType.CELL_CENTERED,
    use_zero_flux: bool = False,
    drift_velocity: float = 0.0,
    diffusion_coeff: float = 1.0,
) -> tuple[Topology, BoundaryCalculator | None]:
    """
    Convert BoundaryConditions to Topology + Calculator pair.

    This is the bridge between the legacy BoundaryConditions API and the new
    Topology/Calculator composition architecture (Issue #516).

    **For NO_FLUX BC type, two physics-based options:**
    - `use_zero_flux=False` (default): ZeroGradientCalculator (du/dn = 0)
    - `use_zero_flux=True`: ZeroFluxCalculator (J·n = 0, mass conservation)

    Args:
        bc: Boundary condition specification (unified or legacy)
        shape: Grid shape (interior points)
        grid_type: Grid type for ghost cell formulas
        use_zero_flux: If True, NO_FLUX BC uses ZeroFluxCalculator (J·n=0)
                      for mass conservation. Default False uses ZeroGradientCalculator.
        drift_velocity: For ZeroFluxCalculator, the normal drift component
        diffusion_coeff: For ZeroFluxCalculator, the diffusion coefficient D = σ²/2

    Returns:
        Tuple of (Topology, Calculator | None). Calculator is None for periodic.

    Examples:
        >>> from mfgarchon.geometry.boundary import dirichlet_bc, no_flux_bc
        >>> bc = dirichlet_bc(0.0, dimension=2)
        >>> topology, calc = bc_to_topology_calculator(bc, shape=(100, 100))
        >>> type(calc).__name__
        'DirichletCalculator'

        >>> # Zero gradient (default for HJB)
        >>> bc = no_flux_bc(dimension=2)
        >>> topo, calc = bc_to_topology_calculator(bc, (100, 100))
        >>> type(calc).__name__
        'ZeroGradientCalculator'

        >>> # Zero flux (for FP mass conservation)
        >>> topo, calc = bc_to_topology_calculator(bc, (100, 100), use_zero_flux=True)
        >>> type(calc).__name__
        'ZeroFluxCalculator'
    """
    dimension = len(shape)

    def _get_no_flux_calculator() -> BoundaryCalculator:
        """Select calculator based on physics requirement."""
        if use_zero_flux:
            return ZeroFluxCalculator(drift_velocity, diffusion_coeff, grid_type)
        else:
            return ZeroGradientCalculator(grid_type)

    # Handle legacy BoundaryConditions from bc_1d module
    if isinstance(bc, LegacyBoundaryConditions1D):
        bc_type_str = bc.type.lower()
        if bc_type_str == "periodic":
            return PeriodicTopology(dimension, shape), None
        elif bc_type_str == "dirichlet":
            value = bc.left_value if bc.left_value is not None else 0.0
            return BoundedTopology(dimension, shape), DirichletCalculator(value, grid_type)
        elif bc_type_str in ("neumann", "no_flux"):
            return BoundedTopology(dimension, shape), _get_no_flux_calculator()
        else:
            # Default to zero gradient
            return BoundedTopology(dimension, shape), _get_no_flux_calculator()

    # Handle unified BoundaryConditions
    if not isinstance(bc, BoundaryConditions):
        raise TypeError(f"Unsupported BC type: {type(bc)}")

    # Only uniform BCs supported for now (mixed BC requires per-point dispatch)
    if not bc.is_uniform:
        raise NotImplementedError(
            "Mixed BCs not yet supported in Topology/Calculator architecture. "
            "Use PreallocatedGhostBuffer for mixed BCs."
        )

    seg = bc.segments[0]
    bc_type = seg.bc_type

    if bc_type == BCType.PERIODIC:
        return PeriodicTopology(dimension, shape), None
    elif bc_type == BCType.DIRICHLET:
        value = seg.value if not callable(seg.value) else 0.0
        return BoundedTopology(dimension, shape), DirichletCalculator(float(value), grid_type)
    elif bc_type == BCType.NEUMANN:
        flux = seg.value if not callable(seg.value) else 0.0
        return BoundedTopology(dimension, shape), NeumannCalculator(float(flux), grid_type)
    elif bc_type == BCType.NO_FLUX:
        return BoundedTopology(dimension, shape), _get_no_flux_calculator()
    elif bc_type == BCType.ROBIN:
        # Issue #543: Use getattr() instead of hasattr for optional Robin BC coefficients
        alpha = getattr(seg, "alpha", 1.0)
        beta = getattr(seg, "beta", 0.0)
        rhs = seg.value if not callable(seg.value) else 0.0
        return BoundedTopology(dimension, shape), RobinCalculator(alpha, beta, float(rhs), grid_type)
    elif bc_type == BCType.EXTRAPOLATION_LINEAR:
        return BoundedTopology(dimension, shape), LinearExtrapolationCalculator()
    elif bc_type == BCType.EXTRAPOLATION_QUADRATIC:
        return BoundedTopology(dimension, shape), QuadraticExtrapolationCalculator()
    elif bc_type == BCType.REFLECTING:
        # Reflecting BC for particles - use zero gradient for field solvers
        return BoundedTopology(dimension, shape), _get_no_flux_calculator()
    else:
        # Unknown type - default to zero gradient
        return BoundedTopology(dimension, shape), _get_no_flux_calculator()


def create_ghost_buffer_from_bc(
    bc: BoundaryConditions | LegacyBoundaryConditions1D,
    shape: tuple[int, ...],
    dx: float | tuple[float, ...],
    ghost_depth: int = 1,
    grid_type: GridType = GridType.CELL_CENTERED,
    dtype: type = np.float64,
    use_zero_flux: bool = False,
    drift_velocity: float = 0.0,
    diffusion_coeff: float = 1.0,
) -> GhostBuffer:
    """
    Factory function to create GhostBuffer from BoundaryConditions.

    This is the primary entry point for using the new Topology/Calculator
    architecture with existing BoundaryConditions specifications.

    Args:
        bc: Boundary condition specification
        shape: Interior grid shape
        dx: Grid spacing (scalar or per-dimension tuple)
        ghost_depth: Number of ghost cell layers
        grid_type: Grid type for ghost cell formulas
        dtype: Data type for buffer
        use_zero_flux: If True, NO_FLUX BC uses ZeroFluxCalculator (J·n=0)
                      for mass conservation. Default False uses ZeroGradientCalculator.
        drift_velocity: For ZeroFluxCalculator, the normal drift component
        diffusion_coeff: For ZeroFluxCalculator, the diffusion coefficient D = σ²/2

    Returns:
        GhostBuffer configured with appropriate Topology and Calculator

    Examples:
        >>> from mfgarchon.geometry.boundary import dirichlet_bc, no_flux_bc
        >>> # Default (zero gradient)
        >>> bc = dirichlet_bc(0.0, dimension=2)
        >>> buffer = create_ghost_buffer_from_bc(bc, shape=(100, 100), dx=0.01)

        >>> # Mass-conserving (zero total flux)
        >>> bc = no_flux_bc(dimension=2)
        >>> buffer = create_ghost_buffer_from_bc(
        ...     bc, shape=(100, 100), dx=0.01,
        ...     use_zero_flux=True, diffusion_coeff=0.5
        ... )
    """
    topology, calculator = bc_to_topology_calculator(
        bc, shape, grid_type, use_zero_flux, drift_velocity, diffusion_coeff
    )
    return GhostBuffer(topology, calculator, dx, ghost_depth, dtype)


class GhostBuffer:
    """
    Ghost cell buffer using Topology/Calculator composition.

    This is the structural foundation for the 2-layer BC architecture:
    - Layer 1: Topology (Memory/Indexing) - periodic vs bounded connectivity
    - Layer 2: Calculator (Physics Strategy) - ghost value computation

    The separation enables:
    - Same Calculator works with different grid topologies
    - Same Topology works with different physics (Dirichlet, Neumann, etc.)
    - Clean interface for FVM/FEM/Graph extensions

    Usage:
        >>> from mfgarchon.geometry.boundary.applicator_base import (
        ...     BoundedTopology, DirichletCalculator, PeriodicTopology
        ... )

        >>> # Bounded domain with Dirichlet BC
        >>> topology = BoundedTopology(dimension=2, shape=(100, 100))
        >>> calculator = DirichletCalculator(boundary_value=0.0)
        >>> buffer = GhostBuffer(topology, calculator, dx=0.1)

        >>> # Set interior values
        >>> buffer.interior[:] = initial_condition

        >>> # Update ghosts (in-place, zero allocation)
        >>> buffer.update()

        >>> # Periodic domain (no calculator needed)
        >>> topology = PeriodicTopology(dimension=2, shape=(100, 100))
        >>> buffer = GhostBuffer(topology)  # Calculator ignored for periodic

    See Also:
        - PreallocatedGhostBuffer: Legacy interface with BoundaryConditions
        - Issue #516: Topology/Calculator composition design
    """

    def __init__(
        self,
        topology: Topology,
        calculator: BoundaryCalculator | None = None,
        dx: float | tuple[float, ...] = 1.0,
        ghost_depth: int = 1,
        dtype: type = np.float64,
    ):
        """
        Initialize ghost buffer with topology and calculator.

        Args:
            topology: Grid topology (PeriodicTopology or BoundedTopology)
            calculator: Ghost value calculator (required for bounded, ignored for periodic)
            dx: Grid spacing (scalar or tuple for each dimension)
            ghost_depth: Number of ghost cells per boundary
            dtype: Data type for the buffer

        Raises:
            ValueError: If bounded topology provided without calculator
        """

        self._topology = topology
        self._calculator = calculator
        self._ghost_depth = ghost_depth
        self._dtype = dtype

        # Validate: bounded topology requires calculator
        if not topology.is_periodic and calculator is None:
            raise ValueError(
                "Bounded topology requires a BoundaryCalculator. "
                "Use PeriodicTopology for periodic boundaries, or provide a calculator."
            )

        # Parse grid spacing
        if isinstance(dx, (int, float)):
            self._dx = tuple([float(dx)] * topology.dimension)
        else:
            self._dx = tuple(float(d) for d in dx)

        if len(self._dx) != topology.dimension:
            raise ValueError(
                f"Grid spacing dimension {len(self._dx)} must match topology dimension {topology.dimension}"
            )

        # Compute shapes
        self._interior_shape = topology.shape
        self._dimension = topology.dimension
        self._padded_shape = tuple(s + 2 * ghost_depth for s in self._interior_shape)

        # Allocate buffer
        self._buffer = np.zeros(self._padded_shape, dtype=dtype)

        # Create interior slice (view, not copy)
        self._interior_slices = tuple(slice(ghost_depth, -ghost_depth) for _ in range(self._dimension))

    @property
    def topology(self) -> Topology:
        """Grid topology."""
        return self._topology

    @property
    def calculator(self) -> BoundaryCalculator | None:
        """Ghost value calculator (None for periodic topology)."""
        return self._calculator

    @property
    def padded(self) -> NDArray[np.floating]:
        """Full padded buffer including ghost cells."""
        return self._buffer

    @property
    def interior(self) -> NDArray[np.floating]:
        """View of interior region (no copy)."""
        return self._buffer[self._interior_slices]

    @property
    def shape(self) -> tuple[int, ...]:
        """Interior shape."""
        return self._interior_shape

    @property
    def padded_shape(self) -> tuple[int, ...]:
        """Padded shape including ghosts."""
        return self._padded_shape

    @property
    def ghost_depth(self) -> int:
        """Number of ghost cells per side."""
        return self._ghost_depth

    @property
    def dx(self) -> tuple[float, ...]:
        """Grid spacing for each dimension."""
        return self._dx

    def update(self, **kwargs) -> None:
        """
        Update ghost cells in-place based on current interior values.

        This is the core zero-allocation operation. The update strategy
        depends on topology type:
        - Periodic: ghost = wrap-around (opposite boundary interior)
        - Bounded: ghost = calculator.compute(interior, dx, side)

        Args:
            **kwargs: Additional arguments passed to calculator.compute()
                     (e.g., time for time-dependent BCs, drift for FP no-flux)
        """
        if self._topology.is_periodic:
            self._update_periodic()
        else:
            self._update_bounded(**kwargs)

    def _update_periodic(self) -> None:
        """Update ghost cells for periodic topology (wrap-around)."""
        d = self._dimension
        g = self._ghost_depth
        buf = self._buffer

        for axis in range(d):
            # Low ghost = high interior
            lo_ghost = [slice(None)] * d
            lo_ghost[axis] = slice(0, g)
            hi_interior = [slice(None)] * d
            hi_interior[axis] = slice(-2 * g, -g)
            buf[tuple(lo_ghost)] = buf[tuple(hi_interior)]

            # High ghost = low interior
            hi_ghost = [slice(None)] * d
            hi_ghost[axis] = slice(-g, None)
            lo_interior = [slice(None)] * d
            lo_interior[axis] = slice(g, 2 * g)
            buf[tuple(hi_ghost)] = buf[tuple(lo_interior)]

    def _update_bounded(self, **kwargs) -> None:
        """Update ghost cells for bounded topology using calculator (vectorized)."""
        if self._calculator is None:
            raise RuntimeError("Calculator is None for bounded topology")

        d = self._dimension
        g = self._ghost_depth
        buf = self._buffer

        for axis in range(d):
            dx = self._dx[axis]

            # Get interior and ghost slices
            lo_ghost = [slice(None)] * d
            lo_ghost[axis] = slice(0, g)
            lo_interior = [slice(None)] * d
            lo_interior[axis] = slice(g, 2 * g)

            hi_ghost = [slice(None)] * d
            hi_ghost[axis] = slice(-g, None)
            hi_interior = [slice(None)] * d
            hi_interior[axis] = slice(-2 * g, -g)

            # Get interior arrays (views, not copies)
            interior_lo = buf[tuple(lo_interior)]
            interior_hi = buf[tuple(hi_interior)]

            # VECTORIZED: Apply calculator to entire boundary array at once
            # All Calculator implementations support NDArray via NumPy broadcasting
            ghost_lo = self._calculator.compute(
                interior_value=interior_lo,
                dx=dx,
                side="min",
                **kwargs,
            )
            ghost_hi = self._calculator.compute(
                interior_value=interior_hi,
                dx=dx,
                side="max",
                **kwargs,
            )

            buf[tuple(lo_ghost)] = ghost_lo
            buf[tuple(hi_ghost)] = ghost_hi

    def reset(self, fill_value: float = 0.0) -> None:
        """Reset buffer to a constant value."""
        self._buffer.fill(fill_value)

    def copy_to_interior(self, data: NDArray[np.floating]) -> None:
        """Copy data to interior region."""
        self.interior[:] = data

    def __repr__(self) -> str:
        calc_str = repr(self._calculator) if self._calculator else "None"
        return (
            f"GhostBuffer(topology={self._topology!r}, calculator={calc_str}, "
            f"shape={self._interior_shape}, ghost_depth={self._ghost_depth})"
        )


# Type hint imports for forward references
if TYPE_CHECKING:
    from .applicator_base import BoundaryCalculator, Topology


# =============================================================================
# Pre-allocated Ghost Cell Buffer (Legacy Interface - Zero-Copy Design)
# =============================================================================


class PreallocatedGhostBuffer:
    """
    Pre-allocated buffer for zero-copy ghost cell boundary conditions.

    This class avoids repeated memory allocation when applying BCs in tight loops
    (e.g., time-stepping). It pre-allocates a padded buffer and provides a view
    into the interior region for solvers to work with directly.

    Memory Layout:
        padded[ghost, interior..., ghost] where interior is a view (no copy)

    Usage:
        # Create buffer once
        buffer = PreallocatedGhostBuffer(
            interior_shape=(100, 100),
            boundary_conditions=bc,
            domain_bounds=bounds,
        )

        # In time loop - zero allocation:
        for t in range(num_steps):
            # Work directly on interior view
            buffer.interior[:] = solve_step(buffer.interior)

            # Update ghost cells in-place
            buffer.update_ghosts(time=t * dt)

            # Access full padded array if needed
            laplacian = compute_laplacian(buffer.padded)

    Performance:
        - Initial allocation: O(N) once
        - update_ghosts(): O(boundary) per call, no allocation
        - Interior access: O(1), view only
    """

    def __init__(
        self,
        interior_shape: tuple[int, ...],
        boundary_conditions: BoundaryConditions | LegacyBoundaryConditions1D,
        domain_bounds: NDArray[np.floating] | None = None,
        dtype: np.dtype = np.float64,
        ghost_depth: int = 1,
        order: int = 2,
        config: GhostCellConfig | None = None,
        geometry: object | None = None,
    ):
        """
        Initialize pre-allocated ghost buffer.

        Args:
            interior_shape: Shape of the interior domain (Ny, Nx) for 2D, etc.
            boundary_conditions: BC specification
            domain_bounds: Domain bounds for mixed BCs
            dtype: Data type for the buffer
            ghost_depth: Number of ghost cells per side (default: 1)
            order: Accuracy order for ghost cell reconstruction (default: 2)
                - order = 2: Linear reflection (simple mirror for Neumann)
                - order > 2: Polynomial extrapolation (high-order schemes like WENO)
            config: Ghost cell configuration
            geometry: Geometry object for region_name resolution (Issue #577 Phase 3).
                     Required if BC segments use region_name.

        Raises:
            ValueError: If order < 1
        """
        # Validate order parameter
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")

        self._interior_shape = interior_shape
        self._dimension = len(interior_shape)
        self._ghost_depth = ghost_depth
        self._order = order
        self._boundary_conditions = boundary_conditions
        self._domain_bounds = domain_bounds
        self._config = config if config is not None else GhostCellConfig()
        self._geometry = geometry  # For region_name resolution (Issue #577 Phase 3)

        # Compute padded shape
        self._padded_shape = tuple(s + 2 * ghost_depth for s in interior_shape)

        # Allocate the padded buffer
        self._buffer = np.zeros(self._padded_shape, dtype=dtype)

        # Create interior slice (view, not copy)
        self._interior_slices = tuple(slice(ghost_depth, -ghost_depth) for _ in range(self._dimension))

        # Pre-compute grid spacing if domain_bounds provided
        self._grid_spacing: tuple[float, ...] | None = None
        if domain_bounds is not None:
            domain_bounds = np.atleast_2d(domain_bounds)
            spacing = []
            for d in range(self._dimension):
                extent = domain_bounds[d, 1] - domain_bounds[d, 0]
                n_points = interior_shape[d]
                spacing.append(extent / (n_points - 1) if n_points > 1 else extent)
            self._grid_spacing = tuple(spacing)

    @property
    def padded(self) -> NDArray[np.floating]:
        """Full padded buffer including ghost cells."""
        return self._buffer

    @property
    def interior(self) -> NDArray[np.floating]:
        """View of interior region (no copy)."""
        return self._buffer[self._interior_slices]

    @property
    def shape(self) -> tuple[int, ...]:
        """Interior shape."""
        return self._interior_shape

    @property
    def padded_shape(self) -> tuple[int, ...]:
        """Padded shape including ghosts."""
        return self._padded_shape

    @property
    def ghost_depth(self) -> int:
        """Number of ghost cells per side."""
        return self._ghost_depth

    def update_ghosts(self, time: float = 0.0) -> None:
        """
        Update ghost cells in-place based on current interior values.

        This is the core zero-allocation operation. Ghost cells are computed
        from the interior values according to the boundary conditions.

        Args:
            time: Current time for time-dependent BCs
        """
        bc = self._boundary_conditions

        # Get BC type for dispatch
        if isinstance(bc, LegacyBoundaryConditions1D):
            bc_type_str = bc.type.lower()
            self._update_ghosts_legacy(bc_type_str)
            return

        if not isinstance(bc, BoundaryConditions):
            raise TypeError(f"Unsupported BC type: {type(bc)}")

        if bc.is_uniform:
            seg = bc.segments[0]
            self._update_ghosts_uniform(seg.bc_type, seg.value, time)
        else:
            # Mixed BC requires more complex per-point updates
            self._update_ghosts_mixed(bc, time)

    def _update_ghosts_uniform(
        self,
        bc_type: BCType,
        value: float | None,
        time: float,
    ) -> None:
        """
        Update ghost cells for uniform BC (same type on all boundaries).

        Dispatches to appropriate reconstruction method based on self._order:
        - order <= 2: Linear reflection (simple mirror)
        - order > 2: Polynomial extrapolation (high-order schemes)
        """
        # Dispatch based on order
        if self._order <= 2:
            self._apply_linear_reflection(bc_type, value, time)
        else:
            self._apply_poly_extrapolation(bc_type, value, time)

    def _apply_linear_reflection(
        self,
        bc_type: BCType,
        value: float | None,
        time: float,
    ) -> None:
        """
        Apply linear reflection for order <= 2 ghost cell reconstruction.

        This is the original implementation that works for FDM and simple
        Semi-Lagrangian schemes. Provides O(h^2) accuracy.
        """
        d = self._dimension
        g = self._ghost_depth
        buf = self._buffer

        # Evaluate value if callable
        if callable(value):
            v = value(time)
        else:
            v = value if value is not None else 0.0

        if bc_type == BCType.PERIODIC:
            # Periodic: ghost = opposite interior
            for axis in range(d):
                # Low ghost = high interior
                lo_ghost = [slice(None)] * d
                lo_ghost[axis] = slice(0, g)
                hi_interior = [slice(None)] * d
                hi_interior[axis] = slice(-2 * g, -g)
                buf[tuple(lo_ghost)] = buf[tuple(hi_interior)]

                # High ghost = low interior
                hi_ghost = [slice(None)] * d
                hi_ghost[axis] = slice(-g, None)
                lo_interior = [slice(None)] * d
                lo_interior[axis] = slice(g, 2 * g)
                buf[tuple(hi_ghost)] = buf[tuple(lo_interior)]

        elif bc_type == BCType.DIRICHLET:
            # Dirichlet: u_ghost = 2*g - u_interior
            for axis in range(d):
                # Low boundary
                lo_ghost = [slice(None)] * d
                lo_ghost[axis] = slice(0, g)
                lo_interior = [slice(None)] * d
                lo_interior[axis] = slice(g, 2 * g)
                buf[tuple(lo_ghost)] = 2 * v - buf[tuple(lo_interior)]

                # High boundary
                hi_ghost = [slice(None)] * d
                hi_ghost[axis] = slice(-g, None)
                hi_interior = [slice(None)] * d
                hi_interior[axis] = slice(-2 * g, -g)
                buf[tuple(hi_ghost)] = 2 * v - buf[tuple(hi_interior)]

        elif bc_type in [BCType.NO_FLUX, BCType.NEUMANN, BCType.REFLECTING]:
            # Zero-gradient Neumann: ghost = adjacent interior (simple reflection).
            # For cell-centered grids with boundary at cell face:
            #   du/dn|_{boundary} = (u_interior - u_ghost)/dx = 0
            #   => u_ghost = u_interior (adjacent interior cell)
            #
            # Padded array structure: [ghost_0, ..., ghost_{g-1}, interior_0, interior_1, ...]
            # For g=1: ghost at idx 0 should equal interior at idx 1 (adjacent).
            for axis in range(d):
                # Low boundary: ghost mirrors adjacent interior
                for k in range(g):
                    lo_ghost = [slice(None)] * d
                    lo_ghost[axis] = g - 1 - k  # Ghost cells from g-1 down to 0
                    lo_interior = [slice(None)] * d
                    lo_interior[axis] = g + k  # Adjacent interior cells from g up
                    buf[tuple(lo_ghost)] = buf[tuple(lo_interior)]

                # High boundary: ghost mirrors adjacent interior
                for k in range(g):
                    hi_ghost = [slice(None)] * d
                    hi_ghost[axis] = -(k + 1)  # Ghost cells from -1 down to -g
                    hi_interior = [slice(None)] * d
                    hi_interior[axis] = -(g + k + 1)  # Adjacent interior cells
                    buf[tuple(hi_ghost)] = buf[tuple(hi_interior)]

        elif bc_type == BCType.ROBIN:
            # Robin: alpha*u + beta*du/dn = g
            # For uniform BC, we assume alpha=0, beta=1 (Neumann-like with non-zero flux)
            # Cell-centered grids: ghost and interior are dx apart, not 2*dx.
            # du/dn = (u_ghost - u_interior)/dx * sign, so:
            # u_ghost = u_interior + dx*g*sign
            # Note: For full Robin BC with arbitrary alpha/beta, use mixed BC API.
            for axis in range(d):
                # Get grid spacing for this axis
                if self._grid_spacing is not None:
                    dx = self._grid_spacing[axis]
                else:
                    dx = 1.0  # Fallback

                # Get adjacent interior values (at index g for low, -g-1 for high)
                lo_interior = [slice(None)] * d
                lo_interior[axis] = g  # Adjacent interior cell
                u_lo_interior = buf[tuple(lo_interior)]

                hi_interior = [slice(None)] * d
                hi_interior[axis] = -g - 1  # Adjacent interior cell
                u_hi_interior = buf[tuple(hi_interior)]

                for k in range(g):
                    # Low boundary (outward normal = -1)
                    lo_ghost = [slice(None)] * d
                    lo_ghost[axis] = g - 1 - k  # Ghost cells from g-1 down to 0
                    # u_ghost = u_interior + dx*g*(-1) = u_interior - dx*v
                    buf[tuple(lo_ghost)] = u_lo_interior - dx * v

                for k in range(g):
                    # High boundary (outward normal = +1)
                    hi_ghost = [slice(None)] * d
                    hi_ghost[axis] = -(g - k)  # Ghost cells from -g up to -1
                    # u_ghost = u_interior + dx*g*(+1) = u_interior + dx*v
                    buf[tuple(hi_ghost)] = u_hi_interior + dx * v

    def _apply_poly_extrapolation(
        self,
        bc_type: BCType,
        value: float | None,
        time: float,
    ) -> None:
        """
        Apply polynomial extrapolation for order > 2 ghost cell reconstruction.

        Uses Vandermonde systems to compute high-order accurate ghost values
        that satisfy boundary conditions. Provides O(h^order) accuracy.

        For Neumann BC (∂u/∂n = 0):
            Constructs polynomial p(x) that:
            1. Passes through n interior points: p(x_j) = u_j
            2. Satisfies BC: p'(0) = 0
            Then evaluates p(x_{-k}) for ghost points.

        For Dirichlet BC (u = g):
            Similar but with constraint p(0) = g.

        Args:
            bc_type: Boundary condition type
            value: BC value (or None for Neumann)
            time: Current time for time-dependent BCs

        Raises:
            NotImplementedError: Periodic/Robin BCs not yet supported for high order.
        """
        d = self._dimension
        buf = self._buffer

        # Evaluate value if callable
        if callable(value):
            v = value(time)
        else:
            v = value if value is not None else 0.0

        # Only support Neumann and Dirichlet for now
        if bc_type == BCType.PERIODIC:
            # Periodic should use linear reflection (shouldn't reach here)
            self._apply_linear_reflection(bc_type, value, time)
            return

        if bc_type not in [BCType.NEUMANN, BCType.NO_FLUX, BCType.REFLECTING, BCType.DIRICHLET]:
            raise NotImplementedError(
                f"Polynomial extrapolation for {bc_type.value} BC not yet implemented. "
                f"Currently supports: NEUMANN, NO_FLUX, REFLECTING, DIRICHLET."
            )

        # Number of stencil points needed for order n polynomial
        # For order n: need n points + 1 BC constraint = n+1 equations
        n_stencil = self._order

        # Apply extrapolation to each axis
        for axis in range(d):
            # Low boundary extrapolation
            self._extrapolate_boundary_1d(buf, axis, is_low=True, bc_type=bc_type, bc_value=v, n_stencil=n_stencil)

            # High boundary extrapolation
            self._extrapolate_boundary_1d(buf, axis, is_low=False, bc_type=bc_type, bc_value=v, n_stencil=n_stencil)

    def _extrapolate_boundary_1d(
        self,
        buf: np.ndarray,
        axis: int,
        is_low: bool,
        bc_type: BCType,
        bc_value: float,
        n_stencil: int,
    ) -> None:
        """
        Extrapolate ghost cells for one boundary using polynomial fitting.

        Solves Vandermonde system:
            V @ coeffs = rhs
        where V includes polynomial basis and BC constraint.

        Args:
            buf: Padded buffer to update in-place
            axis: Axis index (0 for x, 1 for y, etc.)
            is_low: True for low boundary, False for high
            bc_type: Boundary condition type
            bc_value: Boundary condition value
            n_stencil: Number of interior stencil points to use
        """
        g = self._ghost_depth
        d = self._dimension

        # Get grid spacing for this axis
        if self._grid_spacing is not None:
            dx = self._grid_spacing[axis]
        else:
            # Assume uniform spacing of 1.0 if not provided
            dx = 1.0

        # Extract all points along this axis (iterating over other axes)
        # We'll process each 1D slice independently
        shape = list(buf.shape)
        n_total = shape[axis]

        # Build index slicing for this axis
        # For low boundary: interior starts at index g (boundary), but we skip it for stencil
        # For high boundary: interior ends at index -g-1 (boundary), but we skip it for stencil
        if is_low:
            # Ghost points: indices 0 to g-1
            # Interior stencil: indices g+1 to g+n_stencil (skip boundary at g)
            ghost_indices = list(range(g))
            interior_indices = list(range(g + 1, g + 1 + n_stencil))
        else:
            # Ghost points: indices -g to -1 (i.e., -g, -g+1, ..., -1)
            # Interior stencil: indices -g-n_stencil-1 to -g-1 (skip boundary at -g-1).
            # Order nearest-boundary first so the pairing matches ``x_interior`` below
            # ([-dx, -2dx, ...], also nearest-first). Without the reversal the Vandermonde
            # rows pair x=-dx with the farthest interior value, fitting a garbage polynomial
            # -> badly wrong high-boundary ghosts (Issue #1200: latent until HJ-WENO5 became
            # the first high-order scheme to consume the high-boundary ghost derivative).
            ghost_indices = list(range(n_total - g, n_total))
            interior_indices = list(range(n_total - g - n_stencil - 1, n_total - g - 1))[::-1]

        # Compute polynomial coefficients once, then apply to all slices
        # Build Vandermonde matrix and solve for extrapolation weights

        # Grid points relative to boundary (x=0 at boundary)
        if is_low:
            # Low boundary at x=0, interior at x = dx, 2*dx, ...
            x_interior = np.arange(1, n_stencil + 1, dtype=np.float64) * dx
            x_ghost = np.arange(-g, 0, dtype=np.float64) * dx  # Negative x for ghosts
        else:
            # High boundary at x=0, interior at x = -dx, -2*dx, ... (going inward)
            x_interior = -np.arange(1, n_stencil + 1, dtype=np.float64) * dx
            x_ghost = np.arange(1, g + 1, dtype=np.float64) * dx  # Positive x for ghosts

        # Build Vandermonde matrix: [n_stencil+1] x [n_stencil+1]
        # Rows: n_stencil interior constraints + 1 BC constraint
        # Columns: polynomial coefficients [a_0, a_1, ..., a_{n_stencil}]
        n_poly = n_stencil  # Polynomial degree (n_stencil points determine degree n_stencil-1)
        V = np.zeros((n_poly + 1, n_poly + 1))
        rhs = np.zeros(n_poly + 1)

        # First n_stencil rows: polynomial passes through interior points
        # p(x_i) = u_i: sum_j a_j * x_i^j = u_i
        for i in range(n_stencil):
            for j in range(n_poly + 1):
                V[i, j] = x_interior[i] ** j
            # RHS will be filled with actual u values during application

        # Last row: BC constraint at x=0
        if bc_type in [BCType.NEUMANN, BCType.NO_FLUX, BCType.REFLECTING]:
            # Neumann: p'(0) = 0
            # p'(x) = sum_j j * a_j * x^{j-1}
            # p'(0) = a_1 (only linear term survives)
            V[n_stencil, 0] = 0.0  # a_0 doesn't contribute to derivative
            V[n_stencil, 1] = 1.0  # a_1 coefficient
            for j in range(2, n_poly + 1):
                V[n_stencil, j] = 0.0  # Higher terms vanish at x=0
            rhs[n_stencil] = 0.0  # Zero derivative

        elif bc_type == BCType.DIRICHLET:
            # Dirichlet: p(0) = bc_value
            # p(0) = a_0
            V[n_stencil, 0] = 1.0  # a_0 coefficient
            for j in range(1, n_poly + 1):
                V[n_stencil, j] = 0.0  # Higher terms vanish at x=0
            rhs[n_stencil] = bc_value

        # Iterate over all slices perpendicular to this axis
        # Build multi-dimensional slice iterator
        other_axes = [i for i in range(d) if i != axis]
        if d == 1:
            # 1D case: single slice
            slices_to_process = [()]
        else:
            # Multi-D: iterate over all combinations of other axes
            from itertools import product

            other_shapes = [buf.shape[i] for i in other_axes]
            slices_to_process = list(product(*[range(s) for s in other_shapes]))

        for slice_indices in slices_to_process:
            # Build full index with slice_indices inserted at other_axes positions
            full_index = [slice(None)] * d

            # Insert slice indices for other axes
            for other_idx, other_axis in enumerate(other_axes):
                full_index[other_axis] = slice_indices[other_idx]

            # Extract interior stencil values
            u_interior = np.zeros(n_stencil)
            for i, int_idx in enumerate(interior_indices):
                full_index[axis] = int_idx
                u_interior[i] = buf[tuple(full_index)]

            # Set RHS with interior values
            rhs[:n_stencil] = u_interior

            # Solve Vandermonde system for polynomial coefficients
            try:
                coeffs = np.linalg.solve(V, rhs)
            except np.linalg.LinAlgError:
                # Fallback to least squares if singular
                coeffs, _, _, _ = np.linalg.lstsq(V, rhs, rcond=None)

            # Evaluate polynomial at ghost points
            for i, ghost_idx in enumerate(ghost_indices):
                x = x_ghost[i]
                # p(x) = sum_j a_j * x^j
                ghost_value = np.polyval(coeffs[::-1], x)  # polyval expects highest degree first

                full_index[axis] = ghost_idx
                buf[tuple(full_index)] = ghost_value

    def _update_ghosts_legacy(self, bc_type_str: str) -> None:
        """Update ghost cells for legacy BC type."""
        d = self._dimension
        g = self._ghost_depth
        buf = self._buffer

        if bc_type_str == "periodic":
            for axis in range(d):
                lo_ghost = [slice(None)] * d
                lo_ghost[axis] = slice(0, g)
                hi_interior = [slice(None)] * d
                hi_interior[axis] = slice(-2 * g, -g)
                buf[tuple(lo_ghost)] = buf[tuple(hi_interior)]

                hi_ghost = [slice(None)] * d
                hi_ghost[axis] = slice(-g, None)
                lo_interior = [slice(None)] * d
                lo_interior[axis] = slice(g, 2 * g)
                buf[tuple(hi_ghost)] = buf[tuple(lo_interior)]

        elif bc_type_str in ["no_flux", "neumann"]:
            # Reflect interior values about boundary (same as BCType.NEUMANN above)
            for axis in range(d):
                for k in range(g):
                    lo_ghost = [slice(None)] * d
                    lo_ghost[axis] = k
                    lo_interior = [slice(None)] * d
                    lo_interior[axis] = 2 * g - k
                    buf[tuple(lo_ghost)] = buf[tuple(lo_interior)]

                for k in range(g):
                    hi_ghost = [slice(None)] * d
                    hi_ghost[axis] = -(k + 1)
                    hi_interior = [slice(None)] * d
                    hi_interior[axis] = -(2 * g + k + 1)  # Fixed to match BCType.NEUMANN
                    buf[tuple(hi_ghost)] = buf[tuple(hi_interior)]

        elif bc_type_str == "dirichlet":
            bc = self._boundary_conditions
            # Issue #543: Use getattr() instead of hasattr for optional legacy attribute
            v = getattr(bc, "left_value", None)
            v = v if v is not None else 0.0
            for axis in range(d):
                lo_ghost = [slice(None)] * d
                lo_ghost[axis] = slice(0, g)
                lo_interior = [slice(None)] * d
                lo_interior[axis] = slice(g, 2 * g)
                buf[tuple(lo_ghost)] = 2 * v - buf[tuple(lo_interior)]

                hi_ghost = [slice(None)] * d
                hi_ghost[axis] = slice(-g, None)
                hi_interior = [slice(None)] * d
                hi_interior[axis] = slice(-2 * g, -g)
                buf[tuple(hi_ghost)] = 2 * v - buf[tuple(hi_interior)]

    def _update_ghosts_mixed(self, bc: BoundaryConditions, time: float) -> None:
        """
        Update ghost cells for mixed BCs (different types on different boundaries).

        Issue #577 Phase 3: Rewritten to not use deprecated apply_boundary_conditions_nd.
        Iterates over (axis, side) pairs and applies the matching BC segment's ghost formula.
        """
        d = self._dimension
        g = self._ghost_depth
        buf = self._buffer

        for axis in range(d):
            for side in ["min", "max"]:
                target_face = BoundaryFace(axis, side)

                # Find matching BC segment for this face (Issue #946)
                segment = self._find_segment_for_face(bc, target_face)

                if segment is None:
                    # No explicit segment - use default BC (first segment or Neumann)
                    segment = bc.segments[0] if bc.segments else None
                    if segment is None:
                        continue  # No BC defined, skip

                # Apply ghost cell formula for this face
                self._apply_ghost_for_face(buf, axis, side, segment, time, g)

    def _find_segment_for_face(
        self,
        bc: BoundaryConditions,
        target_face: BoundaryFace,
    ) -> BCSegment | None:
        """
        Find the BC segment that applies to a specific face.

        Uses BoundaryFace for dimension-agnostic matching (Issue #946).

        Matching priority:
        1. seg.face matches target_face (handles all string aliases: "left", "x_min", etc.)
        2. region_name matches face string label
        3. region_name matches via geometry.get_region_mask()

        Args:
            bc: Boundary conditions
            target_face: Target boundary face

        Returns:
            Matching BCSegment or None if no explicit match
        """
        for segment in bc.segments:
            # Method 1: Structured face match via seg.face (Issue #946)
            if segment.boundary is not None:
                seg_face = segment.face
                if seg_face is not None and seg_face == target_face:
                    return segment

            # Method 2: region_name matches standard face label
            if segment.region_name is not None:
                face_label = target_face.to_string()
                if segment.region_name.lower() == face_label:
                    return segment

                # Method 3: Use geometry to check if region covers this boundary face
                if self._geometry is not None:
                    try:
                        from mfgarchon.geometry.protocols import SupportsRegionMarking

                        if isinstance(self._geometry, SupportsRegionMarking):
                            region_names = self._geometry.get_region_names()
                            if segment.region_name in region_names:
                                mask = self._geometry.get_region_mask(segment.region_name)
                                d = self._dimension
                                boundary_idx = [slice(None)] * d
                                boundary_idx[target_face.axis] = 0 if target_face.side == "min" else -1
                                face_mask = mask[tuple(boundary_idx)]
                                if np.any(face_mask):
                                    return segment
                    except Exception:
                        logger.debug("Geometry query failed during BC segment matching", exc_info=True)

        return None

    def _apply_ghost_for_face(
        self,
        buf: NDArray,
        axis: int,
        side: str,
        segment: BCSegment,
        time: float,
        g: int,
    ) -> None:
        """
        Apply ghost cell formula for a single face based on BC segment.

        Args:
            buf: Padded buffer array
            axis: Axis index (0=x, 1=y, ...)
            side: "min" or "max"
            segment: BC segment defining the condition
            time: Current time for time-dependent BCs
            g: Ghost depth
        """
        d = self._dimension
        bc_type = segment.bc_type

        # Evaluate value if callable
        value = segment.value
        if callable(value):
            v = value(time)
        else:
            v = value if value is not None else 0.0

        # Get slices for this face
        if side == "min":
            ghost_slices = [slice(None)] * d
            ghost_slices[axis] = slice(0, g)
            interior_slices = [slice(None)] * d
            interior_slices[axis] = slice(g, 2 * g)
            interior_next = [slice(None)] * d
            interior_next[axis] = slice(2 * g, 3 * g) if 3 * g <= buf.shape[axis] else slice(g, 2 * g)
        else:  # side == "max"
            ghost_slices = [slice(None)] * d
            ghost_slices[axis] = slice(-g, None)
            interior_slices = [slice(None)] * d
            interior_slices[axis] = slice(-2 * g, -g)
            interior_next = [slice(None)] * d
            interior_next[axis] = slice(-3 * g, -2 * g) if 3 * g <= buf.shape[axis] else slice(-2 * g, -g)

        # Apply ghost cell formula based on BC type
        if bc_type == BCType.PERIODIC:
            # Periodic: ghost = opposite interior
            if side == "min":
                opposite_interior = [slice(None)] * d
                opposite_interior[axis] = slice(-2 * g, -g)
            else:
                opposite_interior = [slice(None)] * d
                opposite_interior[axis] = slice(g, 2 * g)
            buf[tuple(ghost_slices)] = buf[tuple(opposite_interior)]

        elif bc_type == BCType.DIRICHLET:
            # Dirichlet: u_ghost = 2*g - u_interior
            buf[tuple(ghost_slices)] = 2 * v - buf[tuple(interior_slices)]

        elif bc_type in [BCType.NO_FLUX, BCType.NEUMANN, BCType.REFLECTING]:
            # Zero-gradient Neumann: ghost = adjacent interior (simple reflection).
            # For cell-centered grids: du/dn = (u_interior - u_ghost)/dx = 0
            # => u_ghost = u_interior (adjacent)
            for k in range(g):
                single_ghost = [slice(None)] * d
                single_interior = [slice(None)] * d
                if side == "min":
                    single_ghost[axis] = g - 1 - k  # Ghost cells from g-1 down to 0
                    single_interior[axis] = g + k  # Adjacent interior cells from g up
                else:
                    single_ghost[axis] = -(k + 1)  # Ghost cells from -1 down to -g
                    single_interior[axis] = -(g + k + 1)  # Adjacent interior cells
                buf[tuple(single_ghost)] = buf[tuple(single_interior)]

        elif bc_type == BCType.ROBIN:
            # Robin: alpha*u + beta*du/dn = g
            # Issue #625 fix: Use correct ghost cell formula instead of reflection
            alpha = getattr(segment, "alpha", 0.0)
            beta = getattr(segment, "beta", 1.0)

            # Get grid spacing for this axis
            if self._grid_spacing is not None:
                dx = self._grid_spacing[axis]
            else:
                dx = 1.0  # Fallback (may give incorrect results without proper dx)

            # Outward normal sign: +1 for max, -1 for min
            outward_sign = 1.0 if side == "max" else -1.0

            # Ghost cell formula (cell-centered):
            # For cell-centered grids, ghost and interior cell centers are dx apart.
            # Boundary value: u_b = (u_g + u_i)/2 (midpoint between ghost and interior)
            # Normal derivative: du/dn = (u_g - u_i)/dx * sign (not 2*dx!)
            # Robin BC: alpha * u_b + beta * du/dn = g
            # => alpha * (u_g + u_i)/2 + beta * (u_g - u_i)/dx * sign = g
            # Solving for u_g:
            # u_g * (alpha/2 + beta*sign/dx) = g - u_i * (alpha/2 - beta*sign/dx)
            coeff_ghost = alpha / 2.0 + beta * outward_sign / dx
            coeff_interior = alpha / 2.0 - beta * outward_sign / dx

            if abs(coeff_ghost) < 1e-12:
                # Degenerate case: fall back to reflection
                for k in range(g):
                    single_ghost = [slice(None)] * d
                    single_interior = [slice(None)] * d
                    if side == "min":
                        single_ghost[axis] = k
                        single_interior[axis] = 2 * g - k
                    else:
                        single_ghost[axis] = -(k + 1)
                        single_interior[axis] = -(2 * g + k + 1)
                    buf[tuple(single_ghost)] = buf[tuple(single_interior)]
            else:
                # Apply proper Robin formula using adjacent interior cell
                # For min boundary: ghost[k] uses interior[g] (first interior cell)
                # For max boundary: ghost[-k-1] uses interior[-g-1] (last interior cell)
                for k in range(g):
                    single_ghost = [slice(None)] * d
                    single_interior = [slice(None)] * d
                    if side == "min":
                        single_ghost[axis] = g - 1 - k  # Ghost cells from g-1 down to 0
                        single_interior[axis] = g  # Adjacent interior at index g
                    else:
                        single_ghost[axis] = -(g - k)  # Ghost cells from -g up to -1
                        single_interior[axis] = -g - 1  # Adjacent interior at index -g-1
                    u_interior = buf[tuple(single_interior)]
                    buf[tuple(single_ghost)] = (v - u_interior * coeff_interior) / coeff_ghost

        else:
            # Fallback for unknown BC types: use reflection
            for k in range(g):
                single_ghost = [slice(None)] * d
                single_interior = [slice(None)] * d
                if side == "min":
                    single_ghost[axis] = k
                    single_interior[axis] = 2 * g - k
                else:
                    single_ghost[axis] = -(k + 1)
                    single_interior[axis] = -(2 * g + k + 1)
                buf[tuple(single_ghost)] = buf[tuple(single_interior)]

    def reset(self, fill_value: float = 0.0) -> None:
        """Reset buffer to a constant value."""
        self._buffer.fill(fill_value)

    def copy_to_interior(self, data: NDArray[np.floating]) -> None:
        """Copy data to interior region."""
        self.interior[:] = data

    def copy_from_padded(self, data: NDArray[np.floating]) -> None:
        """Copy full padded data to buffer."""
        self._buffer[:] = data

    def get_padded_coordinates(
        self,
        coordinates: tuple[NDArray[np.floating], ...],
    ) -> tuple[NDArray[np.floating], ...]:
        """
        Extend grid coordinates to include ghost cell regions.

        For Semi-Lagrangian interpolation, we need not just padded values but also
        extended coordinate arrays so that interpolation works properly at boundaries.

        Args:
            coordinates: Tuple of 1D coordinate arrays for each dimension.
                Example: (x_array, y_array) for 2D where x_array has shape (Nx,)
                and y_array has shape (Ny,).

        Returns:
            Tuple of extended coordinate arrays with ghost cell positions.
            Each array is extended by ghost_depth points on each side.

        Example:
            For ghost_depth=1, x=[0, 0.1, 0.2, ..., 1.0] with dx=0.1 becomes:
            x_ext=[-0.1, 0, 0.1, 0.2, ..., 1.0, 1.1]

        Note:
            This assumes uniform grid spacing per dimension. The spacing is
            computed from consecutive coordinate differences.
        """
        if len(coordinates) != self._dimension:
            raise ValueError(f"Expected {self._dimension} coordinate arrays, got {len(coordinates)}")

        g = self._ghost_depth
        extended_coords = []

        for dim, coord in enumerate(coordinates):
            # Compute grid spacing from coordinates (assumes uniform grid)
            if len(coord) > 1:
                h = coord[1] - coord[0]
            elif self._grid_spacing is not None:
                h = self._grid_spacing[dim]
            else:
                h = 1.0  # Fallback for single-point dimension

            # Extend coordinates: add g points on each side
            lo_ghost_coords = coord[0] - h * np.arange(g, 0, -1)  # [x_min - g*h, ..., x_min - h]
            hi_ghost_coords = coord[-1] + h * np.arange(1, g + 1)  # [x_max + h, ..., x_max + g*h]

            extended = np.concatenate([lo_ghost_coords, coord, hi_ghost_coords])
            extended_coords.append(extended)

        return tuple(extended_coords)


def pad_array_with_ghosts(
    array: NDArray[np.floating],
    bc: BoundaryConditions,
    ghost_depth: int = 1,
    time: float = 0.0,
    geometry: object | None = None,
) -> NDArray[np.floating]:
    """
    Pad array with ghost cells based on boundary conditions.

    This is a simple convenience function for one-time use. For time-stepping
    loops where performance matters, use PreallocatedGhostBuffer instead.

    Args:
        array: Interior field array (any dimension)
        bc: Boundary conditions specification
        ghost_depth: Number of ghost layers (default 1)
        time: Current time for time-dependent BC values
        geometry: Geometry object for region_name resolution (Issue #577 Phase 3).
                 Required if BC segments use region_name.

    Returns:
        New array with ghost cells added on all boundaries

    Example:
        >>> from mfgarchon.geometry.boundary import neumann_bc, pad_array_with_ghosts
        >>> u = np.array([1.0, 2.0, 3.0])
        >>> bc = neumann_bc(dimension=1)
        >>> u_padded = pad_array_with_ghosts(u, bc)
        >>> # u_padded is [2.0, 1.0, 2.0, 3.0, 2.0] for Neumann BC
    """
    # Extract domain_bounds from geometry if available (needed for Robin BC)
    domain_bounds = None
    if geometry is not None:
        domain_bounds = getattr(geometry, "domain_bounds", None)

    buffer = PreallocatedGhostBuffer(
        interior_shape=array.shape,
        boundary_conditions=bc,
        domain_bounds=domain_bounds,
        ghost_depth=ghost_depth,
        geometry=geometry,
    )
    # Copy array into interior view and update ghosts
    buffer.interior[:] = array
    buffer.update_ghosts(time=time)
    return buffer.padded.copy()


__all__ = [
    # Configuration
    "GhostCellConfig",
    # Concrete function API (Issue #577 - preferred)
    "pad_array_with_ghosts",
    # Class-based API (Topology/Calculator composition - Issue #516)
    "GhostBuffer",
    "FDMApplicator",
    "PreallocatedGhostBuffer",
]


if __name__ == "__main__":
    """Smoke tests for applicator_fdm module."""

    print("Testing FDM boundary condition applicators...")

    # Test 1: Basic 2D Neumann BC (using pad_array_with_ghosts)
    print("\n1. Testing 2D Neumann BC (no-flux)...")
    from mfgarchon.geometry.boundary import neumann_bc

    field_2d = np.ones((5, 5))
    bc = neumann_bc(dimension=2)
    padded = pad_array_with_ghosts(field_2d, bc)
    assert padded.shape == (7, 7), f"Expected (7,7), got {padded.shape}"
    # Neumann with zero flux: ghost = interior
    assert np.allclose(padded[0, 1:-1], padded[1, 1:-1]), "Neumann BC failed"
    print("   PASS: 2D Neumann BC")

    # Test 2: PreallocatedGhostBuffer (2D Dirichlet)
    print("\n2. Testing PreallocatedGhostBuffer (2D Dirichlet)...")
    from mfgarchon.geometry.boundary import dirichlet_bc

    bounds_2d = np.array([[0.0, 1.0], [0.0, 1.0]])
    bc_dirichlet = dirichlet_bc(dimension=2, value=0.0)
    buffer = PreallocatedGhostBuffer(
        interior_shape=(5, 5),
        boundary_conditions=bc_dirichlet,
        domain_bounds=bounds_2d,
    )
    # Set interior to constant
    buffer.interior[:] = 1.0
    # Update ghosts
    buffer.update_ghosts()
    # Dirichlet with g=0: ghost = 2*0 - interior = -interior
    padded_buf = buffer.padded
    assert padded_buf.shape == (7, 7), f"Expected (7,7), got {padded_buf.shape}"
    assert np.allclose(padded_buf[0, 1:-1], -1.0), "Dirichlet ghost failed (low)"
    assert np.allclose(buffer.interior, 1.0), "Interior modified unexpectedly"
    print("   PASS: PreallocatedGhostBuffer (2D Dirichlet)")

    # Test 3: PreallocatedGhostBuffer (3D Periodic)
    print("\n3. Testing PreallocatedGhostBuffer (3D Periodic)...")
    from mfgarchon.geometry.boundary import periodic_bc

    bc_periodic = periodic_bc(dimension=3)
    bounds_3d = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    buffer_3d = PreallocatedGhostBuffer(
        interior_shape=(4, 4, 4),
        boundary_conditions=bc_periodic,
        domain_bounds=bounds_3d,
    )
    # Set a gradient in x
    for i in range(4):
        buffer_3d.interior[i, :, :] = float(i)
    buffer_3d.update_ghosts()
    padded_3d = buffer_3d.padded
    # Periodic: low ghost = high interior
    assert np.allclose(padded_3d[0, 1:-1, 1:-1], padded_3d[-2, 1:-1, 1:-1]), "Periodic low ghost failed"
    # Periodic: high ghost = low interior
    assert np.allclose(padded_3d[-1, 1:-1, 1:-1], padded_3d[1, 1:-1, 1:-1]), "Periodic high ghost failed"
    print("   PASS: PreallocatedGhostBuffer (3D Periodic)")

    # Test 4: Zero-copy verification
    print("\n4. Testing zero-copy interior view...")
    bc_neumann = neumann_bc(dimension=2)
    buffer2 = PreallocatedGhostBuffer(
        interior_shape=(3, 3),
        boundary_conditions=bc_neumann,
        domain_bounds=bounds_2d,
    )
    interior_view = buffer2.interior
    interior_view[1, 1] = 42.0
    # Modification should reflect in padded
    assert buffer2.padded[2, 2] == 42.0, "Zero-copy failed"
    print("   PASS: Zero-copy interior view")

    # Test 5: 3D face ghost handling (Dirichlet)
    print("\n5. Testing 3D face ghost handling (Dirichlet)...")
    field_3d = np.ones((3, 3, 3))
    bc_3d_dir = dirichlet_bc(dimension=3, value=0.0)
    padded_3d_dir = pad_array_with_ghosts(field_3d, bc_3d_dir)
    # For Dirichlet g=0 with interior=1: ghost = 2*0 - 1 = -1
    # Face ghosts should be -1 (these are the important ones for derivatives)
    assert np.allclose(padded_3d_dir[0, 2, 2], -1.0), "3D face ghost failed"
    assert np.allclose(padded_3d_dir[-1, 2, 2], -1.0), "3D face ghost (high) failed"
    # Edge/corner ghosts: filled by sequential face application
    # (less critical for accuracy, just need to exist and be finite)
    assert np.isfinite(padded_3d_dir[0, 0, 2]), "3D edge ghost not filled"
    assert np.isfinite(padded_3d_dir[0, 0, 0]), "3D corner ghost not filled"
    print("   PASS: 3D face ghost handling (Dirichlet)")

    # Test 6: 4D face ghost handling (Dirichlet)
    print("\n6. Testing 4D face ghost handling (Dirichlet)...")
    field_4d = np.ones((2, 2, 2, 2))
    bc_4d = dirichlet_bc(dimension=4, value=0.0)
    padded_4d = pad_array_with_ghosts(field_4d, bc_4d)
    # For Dirichlet g=0 with interior=1: ghost = 2*0 - 1 = -1
    # Face ghosts should be -1
    assert np.allclose(padded_4d[0, 1, 1, 1], -1.0), "4D face ghost failed"
    # Edge/corner ghosts: just verify they're filled
    assert np.isfinite(padded_4d[0, 0, 1, 1]), "4D edge ghost not filled"
    assert np.isfinite(padded_4d[0, 0, 0, 0]), "4D corner ghost not filled"
    print("   PASS: 4D face ghost handling (Dirichlet)")

    print("\n" + "=" * 50)
    print("All smoke tests passed!")
    print("=" * 50)

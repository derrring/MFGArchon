"""
Deprecated ghost-value entry point, retained until its declared removal.

`get_ghost_values_nd` is the only public name left here. It is not in the package
``__all__``; it stays reachable as an attribute of ``mfgarchon.geometry.boundary``
through that module's ``__getattr__``, which is the surface its v0.25.0 removal
notice applies to. Everything else it needs is private to this module.

The canonical way to apply boundary conditions is `FDMApplicator` or
`pad_array_with_ghosts()`.

The array-padding entry points that used to live here -- ``apply_boundary_conditions``
in its 1D/2D/3D/nD forms, ``create_boundary_mask_2d``, and their twelve helpers --
were deleted 2026-08-15. Their public export had already been withdrawn (#577
Phase 3), no library module imported them, and the only remaining importer was one
test file reaching into this private module by name. They carried six of the
package's nineteen ``BCType`` dispatch chains -- three of which raised on an
unrecognised type and three of which fell through silently.

.. deprecated:: v0.17.0
    `get_ghost_values_nd` will be removed in v0.25.0.
    See issue #577 for the migration guide.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.utils.deprecation import deprecated
from mfgarchon.utils.mfg_logging import get_logger

from .conditions import BoundaryConditions
from .fdm_bc_1d import BoundaryConditions as BoundaryConditions1DFDM

# GhostCellConfig moved to ghost_cells.py (canonical location).
# Re-exported here for backward compatibility.
from .ghost_cells import GhostCellConfig, ghost_cell_neumann, ghost_cell_robin
from .types import BCType

logger = get_logger(__name__)

# Backward compatibility alias
LegacyBoundaryConditions1D = BoundaryConditions1DFDM

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Ghost Value Computation for HJB Upwind Schemes
#
# All that survives of this module. The 2D/nD/1D/3D array-padding entry points
# and their twelve helpers were deleted 2026-08-15: no library module imported
# any of them, and their public export had already been withdrawn (#577 Phase 3).
# `get_ghost_values_nd` stays because it is still reachable as an attribute of
# `mfgarchon.geometry.boundary`, and its removal is declared for v0.25.0.
# =============================================================================


def _get_boundary_name_nd(axis: int, side: str, dimension: int) -> str:
    """
    Get boundary name for axis and side in arbitrary dimension.

    Args:
        axis: Axis index (0, 1, 2, ...)
        side: "min" or "max"
        dimension: Total dimension

    Returns:
        Boundary name string (e.g., "x_min", "y_max", "dim3_min")
    """
    if dimension <= 3:
        axis_names = ["x", "y", "z"]
        return f"{axis_names[axis]}_{side}"
    else:
        # For d > 3, use generic naming
        if axis < 3:
            axis_names = ["x", "y", "z"]
            return f"{axis_names[axis]}_{side}"
        else:
            return f"dim{axis}_{side}"


@deprecated(
    since="v0.17.0",
    replacement="Use pad_array_with_ghosts() or PreallocatedGhostBuffer instead. See issue #577.",
)
def get_ghost_values_nd(
    field: NDArray[np.floating],
    boundary_conditions: BoundaryConditions | LegacyBoundaryConditions1D,
    spacing: tuple[float, ...] | NDArray[np.floating],
    config: GhostCellConfig | None = None,
    time: float = 0.0,
) -> dict[tuple[int, int], NDArray[np.floating]]:
    """
    Compute ghost values for each boundary without padding the array.

    This function is designed for HJB upwind schemes that need ghost values
    BEFORE computing the Hamiltonian. Unlike apply_boundary_conditions_nd
    which returns a padded array, this returns ghost values separately.

    For upwind schemes at boundary i=0:
    - If drift v > 0 (flow from left): need ghost value u[-1] for backward diff
    - If drift v < 0 (flow from right): use interior u[1] for forward diff

    The ghost values are derived from BC type:
    - Dirichlet: u_ghost = 2*g - u_interior (cell-centered)
    - Neumann/No-flux: u_ghost = u_interior + dx*g, g = du/dn (`ghost_cell_neumann` owns it)
    - Periodic: u_ghost = u_opposite_boundary

    Supports mixed BCs where different boundaries have different types.
    Issue #542 fix: Properly handles per-boundary BC types.

    Args:
        field: Interior field of shape (N_1, N_2, ..., N_d)
        boundary_conditions: BC specification (unified or legacy)
        spacing: Grid spacing for each dimension, tuple or array of length d
        config: Ghost cell configuration (grid type)
        time: Current time for time-dependent BC values (default: 0.0)

    Returns:
        Dictionary mapping (dimension, side) to ghost value arrays:
        - Key (d, 0): ghost values for left boundary of dimension d
        - Key (d, 1): ghost values for right boundary of dimension d
        Each ghost array has shape matching the boundary slice.

    Example:
        >>> u = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 field
        >>> bc = dirichlet_bc(dimension=2, value=0.0)
        >>> ghosts = get_ghost_values_nd(u, bc, spacing=(0.1, 0.1))
        >>> ghosts[(0, 0)]  # Ghost for left boundary of dim 0 (shape: (3,))
        >>> ghosts[(1, 1)]  # Ghost for right boundary of dim 1 (shape: (2,))

    .. deprecated::
        This function-based API will be removed in v0.25.0.
        Use pad_array_with_ghosts() or PreallocatedGhostBuffer instead.
        See issue #577 for migration guide.
    """
    if config is None:
        config = GhostCellConfig()

    d = field.ndim
    spacing = np.asarray(spacing)
    if len(spacing) != d:
        raise ValueError(f"Spacing length {len(spacing)} != field dimension {d}")

    ghosts: dict[tuple[int, int], NDArray[np.floating]] = {}

    # Check if we have mixed BCs
    is_mixed = isinstance(boundary_conditions, BoundaryConditions) and not boundary_conditions.is_uniform

    if not is_mixed:
        # Uniform BC - same type on all boundaries
        bc_type, bc_value = _get_bc_type_and_value(boundary_conditions)

        for axis in range(d):
            dx = spacing[axis]
            shape_axis = field.shape[axis]

            # Get interior values adjacent to boundaries
            slices_left = [slice(None)] * d
            slices_left[axis] = 0
            u_int_left = field[tuple(slices_left)]

            slices_next_left = [slice(None)] * d
            slices_next_left[axis] = 1 if shape_axis > 1 else 0
            u_next_left = field[tuple(slices_next_left)]

            slices_right = [slice(None)] * d
            slices_right[axis] = -1
            u_int_right = field[tuple(slices_right)]

            slices_prev_right = [slice(None)] * d
            slices_prev_right[axis] = -2 if shape_axis > 1 else -1
            u_prev_right = field[tuple(slices_prev_right)]

            alpha, beta = _robin_coefficients(boundary_conditions) if bc_type == BCType.ROBIN else (1.0, 0.0)
            ghosts[(axis, 0)], ghosts[(axis, 1)] = _compute_ghost_pair(
                bc_type,
                bc_value,
                u_int_left,
                u_int_right,
                u_next_left,
                u_prev_right,
                dx,
                time,
                config,
                alpha,
                beta,
            )

    else:
        # Mixed BC - need to query per-boundary type
        # Issue #542 fix: Get BC type for each boundary separately
        for axis in range(d):
            dx = spacing[axis]
            shape_axis = field.shape[axis]

            # Get interior values
            slices_left = [slice(None)] * d
            slices_left[axis] = 0
            u_int_left = field[tuple(slices_left)]

            slices_next_left = [slice(None)] * d
            slices_next_left[axis] = 1 if shape_axis > 1 else 0
            u_next_left = field[tuple(slices_next_left)]

            slices_right = [slice(None)] * d
            slices_right[axis] = -1
            u_int_right = field[tuple(slices_right)]

            slices_prev_right = [slice(None)] * d
            slices_prev_right[axis] = -2 if shape_axis > 1 else -1
            u_prev_right = field[tuple(slices_prev_right)]

            # Get boundary names for this axis
            boundary_min = _get_boundary_name_nd(axis, "min", d)
            boundary_max = _get_boundary_name_nd(axis, "max", d)

            # Get BC type at each boundary
            bc_type_left = boundary_conditions.get_bc_type_at_boundary(boundary_min)
            bc_type_right = boundary_conditions.get_bc_type_at_boundary(boundary_max)

            # Get BC value at each boundary (need a representative point)
            # For 1D: just use the boundary coordinate
            # For nD: use center of the boundary face
            bc_value_left = _get_bc_value_at_boundary(boundary_conditions, boundary_min, time)
            bc_value_right = _get_bc_value_at_boundary(boundary_conditions, boundary_max, time)

            # Compute ghost for left boundary
            alpha_l, beta_l = (
                _robin_coefficients(boundary_conditions, boundary_min) if bc_type_left == BCType.ROBIN else (1.0, 0.0)
            )
            ghosts[(axis, 0)] = _compute_single_ghost(
                bc_type_left,
                bc_value_left,
                u_int_left,
                u_next_left,
                dx,
                time,
                config,
                alpha_l,
                beta_l,
            )

            # Compute ghost for right boundary
            alpha_r, beta_r = (
                _robin_coefficients(boundary_conditions, boundary_max) if bc_type_right == BCType.ROBIN else (1.0, 0.0)
            )
            ghosts[(axis, 1)] = _compute_single_ghost(
                bc_type_right,
                bc_value_right,
                u_int_right,
                u_prev_right,
                dx,
                time,
                config,
                alpha_r,
                beta_r,
            )

    return ghosts


def _robin_coefficients(boundary_conditions, boundary: str | None = None) -> tuple[float, float]:
    """Read Robin (alpha, beta) for a boundary, or the defaults if no Robin segment governs it.

    #1961: this module's two ghost helpers took only (bc_type, bc_value), so a Robin condition
    arrived with its coefficients already discarded and both branches returned the adjacent
    interior cell -- the impermeable-wall mirror. Measured on three coefficient sets, the ghost
    was 3.10000 every time, and the residual of the condition the caller wrote ranged over
    1.7 to 5.2.
    """
    segments = getattr(boundary_conditions, "segments", None) or []
    for seg in segments:
        if seg.bc_type != BCType.ROBIN:
            continue
        if boundary is None or seg.boundary is None or boundary_conditions._segment_covers(seg, boundary):
            return float(seg.alpha), float(seg.beta)
    raise ValueError(
        f"get_ghost_values_nd: a ROBIN condition governs {boundary or 'this boundary'} but no "
        "BCSegment carries its alpha/beta. They live on the segment, not on BoundaryConditions."
    )


def _compute_ghost_pair(
    bc_type: BCType,
    bc_value: float | None,
    u_int_left: NDArray,
    u_int_right: NDArray,
    u_next_left: NDArray,
    u_prev_right: NDArray,
    dx: float,
    time: float,
    config: GhostCellConfig,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> tuple[NDArray, NDArray]:
    """Compute ghost values for both boundaries with same BC type."""
    g = bc_value if bc_value is not None else 0.0
    if callable(g):
        g = g(time)

    if bc_type == BCType.PERIODIC:
        return u_int_right.copy(), u_int_left.copy()

    elif bc_type == BCType.DIRICHLET:
        if config.is_vertex_centered:
            return np.full_like(u_int_left, g), np.full_like(u_int_right, g)
        else:
            return 2 * g - u_int_left, 2 * g - u_int_right

    elif bc_type in [BCType.NO_FLUX, BCType.NEUMANN, BCType.REFLECTING]:
        # #2067: `ghost_cell_neumann` owns this, the same way `ghost_cell_robin` owns the branch
        # below since #1961. What was here read `g` as du/dx -- one signed value applied with
        # OPPOSITE signs at the two walls -- while passing the same `g` to `ghost_cell_robin`,
        # which reads du/dn. Whichever convention a caller held, one of the two branches was
        # wrong at the low wall.
        #
        # It also used the retired `2*dx` separation off the SECOND interior cell, so it diverged
        # from the owner at g = 0 too: measured on a no-flux wall, 0.3 on `u = 3x` and 0.588 on
        # `u = sin(2*pi*x)`. Only a constant field agreed, which is why nothing noticed.
        return (
            ghost_cell_neumann(u_int_left, g, dx),
            ghost_cell_neumann(u_int_right, g, dx),
        )

    elif bc_type == BCType.ROBIN:
        # #1961: this returned the adjacent interior cell -- the impermeable-wall mirror -- for
        # every coefficient pair, so a Robin wall was not a Robin wall. `ghost_cell_robin` owns
        # the formula and is side-free on a cell-centred grid (#1907).
        return (
            ghost_cell_robin(u_int_left, g, alpha, beta, dx, grid_type=config.grid_type),
            ghost_cell_robin(u_int_right, g, alpha, beta, dx, grid_type=config.grid_type),
        )

    else:
        return u_next_left.copy(), u_prev_right.copy()


def _compute_single_ghost(
    bc_type: BCType,
    bc_value: float | None,
    u_int: NDArray,
    u_neighbor: NDArray,
    dx: float,
    time: float,
    config: GhostCellConfig,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> NDArray:
    """Compute ghost value for a single boundary.

    Args:
        bc_type: BC type at this boundary
        bc_value: BC value (constant or None)
        u_int: Interior value at boundary (u[0] for left, u[-1] for right)
        u_neighbor: Next interior value (u[1] for left, u[-2] for right)
        dx: Grid spacing
        time: Current time
        config: Ghost cell configuration
    """
    g = bc_value if bc_value is not None else 0.0
    if callable(g):
        g = g(time)

    if bc_type == BCType.PERIODIC:
        # For mixed BC with periodic, this shouldn't normally happen
        # Just return neighbor as fallback
        return u_neighbor.copy()

    elif bc_type == BCType.DIRICHLET:
        if config.is_vertex_centered:
            return np.full_like(u_int, g)
        else:
            # Cell-centered: ghost = 2*g - u_interior
            return 2 * g - u_int

    elif bc_type in [BCType.NO_FLUX, BCType.NEUMANN, BCType.REFLECTING]:
        # #2067, same as the sibling branch in `_compute_ghost_pair`. The `side` parameter this
        # function took existed only to undo the du/dx sign; du/dn already carries the wall's
        # direction, so the owner needs no side argument and neither does this. It is gone with
        # the arithmetic that needed it -- nothing else in this function read it.
        return ghost_cell_neumann(u_int, g, dx)

    elif bc_type == BCType.ROBIN:
        # #1961: this returned the adjacent interior cell for every coefficient pair. See the
        # sibling branch in `_compute_ghost_pair`.
        return ghost_cell_robin(u_int, g, alpha, beta, dx, grid_type=config.grid_type)

    else:
        return u_neighbor.copy()


def _get_bc_value_at_boundary(
    bc: BoundaryConditions,
    boundary: str,
    time: float,
) -> float | None:
    """Get BC value at a specific boundary for mixed BCs."""
    # Find segment that matches this boundary
    for segment in bc.segments:
        if segment.boundary == boundary:
            val = segment.value
            if callable(val):
                return val(time)
            return val

    # Fall back to default value
    return bc.default_value


def _get_bc_type_and_value(
    boundary_conditions: BoundaryConditions | LegacyBoundaryConditions1D,
) -> tuple[BCType, float | None]:
    """Extract BC type and value from unified or legacy BC specification."""
    if isinstance(boundary_conditions, LegacyBoundaryConditions1D):
        bc_type_str = boundary_conditions.type.lower()
        if bc_type_str == "periodic":
            return BCType.PERIODIC, None
        elif bc_type_str == "dirichlet":
            return BCType.DIRICHLET, boundary_conditions.left_value
        elif bc_type_str in ["no_flux", "neumann"]:
            return BCType.NO_FLUX, 0.0
        else:
            return BCType.NO_FLUX, 0.0

    if not isinstance(boundary_conditions, BoundaryConditions):
        raise TypeError(f"Unsupported BC type: {type(boundary_conditions)}")

    if boundary_conditions.is_uniform:
        seg = boundary_conditions.segments[0]
        return seg.bc_type, seg.value

    # Mixed BC - return default type (first segment)
    seg = boundary_conditions.segments[0]
    return seg.bc_type, seg.value

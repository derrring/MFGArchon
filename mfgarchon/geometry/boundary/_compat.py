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

from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.utils.deprecation import deprecated
from mfgarchon.utils.mfg_logging import get_logger

from .conditions import BoundaryConditions

# GhostCellConfig moved to ghost_cells.py (canonical location).
# Re-exported here for backward compatibility.
from .ghost_cells import GhostCellConfig, ghost_cell_neumann, ghost_cell_robin
from .types import BCType, BoundaryFace

logger = get_logger(__name__)

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


@deprecated(
    since="v0.17.0",
    replacement="Use pad_array_with_ghosts() or PreallocatedGhostBuffer instead. See issue #577.",
)
def get_ghost_values_nd(
    field: NDArray[np.floating],
    boundary_conditions: BoundaryConditions,
    spacing: tuple[float, ...] | NDArray[np.floating],
    config: GhostCellConfig | None = None,
    time: float | None = None,
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
        time: Current time for time-dependent BC values (None, the default, reads a constant or g(x) value and refuses one that declares t (#2512))

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
        if not isinstance(boundary_conditions, BoundaryConditions):
            raise TypeError(f"Unsupported BC type: {type(boundary_conditions)}")
        # A uniform BC is one segment on every face; asked on the first face, as a face
        bc_type, bc_value, alpha, beta = _face_condition(boundary_conditions, BoundaryFace(0, "min"), time)

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

            # The axis's two faces, asked of the owner as faces: no name is built (#2628)
            boundary_min = BoundaryFace(axis, "min")
            boundary_max = BoundaryFace(axis, "max")

            # One segment per face, from the owner (#2512 row B1): its type, value and Robin
            # coefficients come from that one segment, not from three resolvers that could each pick a
            # different one.
            bc_type_left, bc_value_left, alpha_l, beta_l = _face_condition(boundary_conditions, boundary_min, time)
            bc_type_right, bc_value_right, alpha_r, beta_r = _face_condition(boundary_conditions, boundary_max, time)

            # Compute ghost for left boundary
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


def _face_condition(
    boundary_conditions, boundary: BoundaryFace, time: float | None
) -> tuple[BCType, Any, float, float]:
    """The condition on one face: the owner's segment, its value, and its Robin coefficients if ROBIN.

    #1961: a Robin condition must arrive with its own coefficients, not the defaults (1, 0), which make
    it a Dirichlet wall.
    """
    from .types import segment_robin_coefficients, segment_value

    where = f"get_ghost_values_nd on face {boundary.to_string()}"
    segment = boundary_conditions.segment_for_face(boundary)
    value = segment_value(segment, t=time, x=None, where=where, time_keyword="time")
    if segment.bc_type == BCType.ROBIN:
        alpha, beta = segment_robin_coefficients(segment, where=f"{where} (Robin)")
        return segment.bc_type, value, alpha, beta
    return segment.bc_type, value, 1.0, 0.0


def _compute_ghost_pair(
    bc_type: BCType,
    bc_value: float | None,
    u_int_left: NDArray,
    u_int_right: NDArray,
    u_next_left: NDArray,
    u_prev_right: NDArray,
    dx: float,
    time: float | None,
    config: GhostCellConfig,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> tuple[NDArray, NDArray]:
    """Compute ghost values for both boundaries with same BC type."""
    g = bc_value  # read by the owner before it gets here, `segment_value` or `boundary_value` (#2512 row B2)

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
    time: float | None,
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
    g = bc_value  # read by the owner before it gets here, `segment_value` or `boundary_value` (#2512 row B2)

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

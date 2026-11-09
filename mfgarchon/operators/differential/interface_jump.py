"""
Interface jump operators for multiphase/free boundary problems.

Computes jumps [f] = f_right - f_left across an interface defined by
a level set function φ(x) = 0. Essential for Stefan problems, free boundary
problems, and multi-phase flow simulations.

**Note on Naming**:
"Interface jump" refers to discontinuities across material boundaries (classical PDE),
not Lindblad jump operators from quantum mechanics. The term "jump" is standard
in level set literature (Osher & Fedkiw 2003, Gibou et al. 2018).

**Mathematical Background**:
For a field f(x) with discontinuity across interface Γ = {x : φ(x) = 0}:
    [f] = lim(f(x + ε·n)) - lim(f(x - ε·n))
         ε→0+              ε→0+

where n = ∇φ/|∇φ| is the interface normal.

**Stefan Problem Application**:
Stefan condition relates interface velocity to heat flux jump:
    V = -κ·[∂T/∂n]
where [∂T/∂n] is the normal derivative jump across the interface.

References:
- Gibou et al. (2018): A review of level set methods and some recent applications
- Osher & Fedkiw (2003): Level Set Methods, Chapter 8

Created: 2026-01-18 (Issue #605 Phase 2.2)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid

logger = get_logger(__name__)


class InterfaceJumpOperator:
    """
    Compute jumps [f] = f_right - f_left across level set interface.

    For a level set φ(x) defining interface φ = 0, computes the jump
    in field values or derivatives across the interface.

    Examples
    --------
    >>> # 1D Stefan problem: compute heat flux jump
    >>> from mfgarchon.geometry.boundary import no_flux_bc
    >>> grid = TensorProductGrid(bounds=[(0, 1)], Nx=[100],
    ...                          boundary_conditions=no_flux_bc(dimension=1))
    >>> phi = x - 0.5  # Interface at x = 0.5
    >>> jump_op = InterfaceJumpOperator(grid, phi)
    >>>
    >>> # Temperature field
    >>> T = some_temperature_distribution(x)
    >>>
    >>> # Compute gradient jump: [∂T/∂x] = ∂T/∂x|_right - ∂T/∂x|_left
    >>> grad_jump = jump_op.compute_jump(T, quantity="gradient")
    >>>
    >>> # Stefan condition: V = -κ·[∂T/∂n]
    >>> V_interface = -thermal_conductivity * grad_jump

    Notes
    -----
    Current implementation uses simple finite difference stencils with
    offset points on each side of the interface. Future improvements:
    - Ghost fluid method for sharper jumps
    - Higher-order interpolation
    - Subcell interface location
    """

    def __init__(
        self,
        grid: TensorProductGrid,
        interface_phi: NDArray[np.float64],
        offset_distance: float | None = None,
    ):
        """
        Initialize jump operator for given interface.

        Parameters
        ----------
        grid : TensorProductGrid
            Spatial grid providing geometry information.
        interface_phi : NDArray
            Level set function defining interface (φ = 0 is interface).
        offset_distance : float, optional
            Distance from interface to sample values (default: 2×min(spacing)).
            Controls how far from interface to compute one-sided derivatives.

        Notes
        -----
        The offset_distance should be:
        - Large enough to avoid numerical noise at the interface
        - Small enough to capture sharp gradients
        - Typically 2-3 grid spacings works well
        """
        self.grid = grid
        self.interface_phi = interface_phi
        self.spacing = grid.spacing

        # Auto-select offset distance
        if offset_distance is None:
            self.offset_distance = 2.0 * min(self.spacing)
        else:
            self.offset_distance = offset_distance

        # Find interface location (points where φ ≈ 0)
        self._identify_interface()

        logger.debug(
            f"InterfaceJumpOperator initialized: dimension={grid.dimension}, "
            f"interface_points={self.n_interface_points}, offset={self.offset_distance:.4f}"
        )

    def _identify_interface(self):
        """Identify grid points near the interface."""
        # Interface region: |φ| < offset_distance
        # These are the points where we'll compute jumps
        self.interface_mask = np.abs(self.interface_phi) < self.offset_distance

        self.n_interface_points = np.sum(self.interface_mask)

        if self.n_interface_points == 0:
            logger.warning("No interface points found! Check level set function.")

    def compute_jump(
        self,
        field: NDArray[np.float64],
        quantity: Literal["value", "gradient"] = "gradient",
    ) -> NDArray[np.float64]:
        """
        Compute jump [f] across interface.

        Parameters
        ----------
        field : NDArray
            Field to compute jump for (e.g., temperature, density).
        quantity : {"value", "gradient"}, default="gradient"
            What to compute jump of:
            - "value": Jump in field values [f]
            - "gradient": Jump in normal derivative [∂f/∂n]

        Returns
        -------
        jump : NDArray
            Jump values at interface points. Same shape as field.
            Non-interface points have jump = 0.

        Notes
        -----
        For 1D:
        - Interface normal is simply ±1
        - Gradient jump is difference of one-sided derivatives

        For 2D/3D:
        - Interface normal: n = ∇φ/|∇φ|
        - Normal derivative: ∂f/∂n = ∇f · n
        """
        if quantity == "value":
            return self._compute_value_jump(field)
        elif quantity == "gradient":
            return self._compute_gradient_jump(field)
        else:
            raise ValueError(f"Unknown quantity: {quantity}. Use 'value' or 'gradient'.")

    def _compute_value_jump(self, field: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute jump in field values: [f] = f_right - f_left.

        Uses simple offset sampling on each side of interface.
        """
        if self.grid.dimension == 1:
            return self._compute_value_jump_1d(field)
        return self._compute_value_jump_nd(field)

    def _compute_gradient_jump(self, field: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute jump in normal derivative: [∂f/∂n] = ∂f/∂n|_right - ∂f/∂n|_left.

        Uses one-sided finite differences on each side of interface.
        """
        if self.grid.dimension == 1:
            return self._compute_gradient_jump_1d(field)
        return self._compute_gradient_jump_nd(field)

    def _compute_value_jump_1d(self, field: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute value jump in 1D."""
        jump = np.zeros_like(field)

        # Find interface index (zero crossing)
        idx_interface = np.argmin(np.abs(self.interface_phi))

        # Offset in grid points
        dx = self.spacing[0]
        offset_points = max(1, int(np.round(self.offset_distance / dx)))

        # Sample on each side
        if idx_interface > offset_points and idx_interface < len(field) - offset_points:
            f_left = field[idx_interface - offset_points]
            f_right = field[idx_interface + offset_points]

            # Jump: right - left
            jump[idx_interface] = f_right - f_left

        return jump

    def _compute_gradient_jump_1d(self, field: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute gradient jump in 1D using one-sided finite differences.

        For 1D: ∂f/∂n = ∂f/∂x (since normal is ±1).
        """
        jump = np.zeros_like(field)

        # Find interface index
        idx_interface = np.argmin(np.abs(self.interface_phi))

        # Offset in grid points
        dx = self.spacing[0]
        offset_points = max(1, int(np.round(self.offset_distance / dx)))

        # Compute one-sided gradients
        if idx_interface > offset_points and idx_interface < len(field) - offset_points:
            # Right side gradient (using points to the right)
            # Forward difference: ∂f/∂x ≈ (f[i+offset] - f[i]) / (offset·dx)
            grad_right = (field[idx_interface + offset_points] - field[idx_interface]) / (offset_points * dx)

            # Left side gradient (using points to the left)
            # Backward difference: ∂f/∂x ≈ (f[i] - f[i-offset]) / (offset·dx)
            grad_left = (field[idx_interface] - field[idx_interface - offset_points]) / (offset_points * dx)

            # Jump: grad_right - grad_left
            jump[idx_interface] = grad_right - grad_left

        return jump

    def _get_interface_normal(self) -> NDArray[np.float64]:
        """Compute unit normal field n = nabla(phi) / |nabla(phi)| on the grid."""
        from mfgarchon.geometry.level_set.core import LevelSetFunction

        ls = LevelSetFunction(self.interface_phi, self.grid)
        return ls.get_normal()

    def _physical_to_index(self, coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert physical coordinates to fractional grid indices.

        Args:
            coords: Physical coordinates, shape (ndim, n_points).

        Returns:
            Fractional grid indices, shape (ndim, n_points).
        """
        Nx_points = self.grid.Nx_points
        indices = np.empty_like(coords)
        for d in range(self.grid.dimension):
            x_min, x_max = self.grid.bounds[d]
            indices[d] = (coords[d] - x_min) / (x_max - x_min) * (Nx_points[d] - 1)
        return indices

    def _compute_value_jump_nd(self, field: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute value jump [f] = f_right - f_left in nD.

        Uses scipy.ndimage.map_coordinates to interpolate field values
        at offset positions on each side of the interface.
        """
        from scipy.ndimage import map_coordinates

        jump = np.zeros_like(field)
        normal = self._get_interface_normal()  # shape (ndim, *grid_shape)

        # Get coordinates of interface points
        interface_indices = np.argwhere(self.interface_mask)  # (n_pts, ndim)
        if len(interface_indices) == 0:
            return jump

        # Build physical coordinates of interface points
        Nx_points = self.grid.Nx_points
        coords = np.empty((self.grid.dimension, len(interface_indices)))
        for d in range(self.grid.dimension):
            x_min, x_max = self.grid.bounds[d]
            coords[d] = x_min + interface_indices[:, d] * (x_max - x_min) / (Nx_points[d] - 1)

        # Normal at interface points
        n_at_pts = np.empty_like(coords)
        idx_tuple = tuple(interface_indices[:, k] for k in range(self.grid.dimension))
        for d in range(self.grid.dimension):
            n_at_pts[d] = normal[d][idx_tuple]

        # Offset positions
        offset = self.offset_distance
        coords_right = coords + offset * n_at_pts
        coords_left = coords - offset * n_at_pts

        # Convert to fractional grid indices
        idx_right = self._physical_to_index(coords_right)
        idx_left = self._physical_to_index(coords_left)

        # Interpolate
        f_right = map_coordinates(field, idx_right, order=1, mode="nearest")
        f_left = map_coordinates(field, idx_left, order=1, mode="nearest")

        # Store jump at interface points
        jump[idx_tuple] = f_right - f_left

        return jump

    def _compute_gradient_jump_nd(self, field: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute gradient jump [partial f / partial n] in nD.

        Computes the normal derivative partial f / partial n = nabla f . n
        on each side of the interface and returns the jump.
        """
        from scipy.ndimage import map_coordinates

        jump = np.zeros_like(field)
        normal = self._get_interface_normal()

        interface_indices = np.argwhere(self.interface_mask)
        if len(interface_indices) == 0:
            return jump

        # Compute field gradient via grid operators
        grad_ops = self.grid.get_gradient_operator()
        grad_field = np.array([grad_op(field) for grad_op in grad_ops])  # (ndim, *shape)

        # Build physical coordinates of interface points
        Nx_points = self.grid.Nx_points
        coords = np.empty((self.grid.dimension, len(interface_indices)))
        for d in range(self.grid.dimension):
            x_min, x_max = self.grid.bounds[d]
            coords[d] = x_min + interface_indices[:, d] * (x_max - x_min) / (Nx_points[d] - 1)

        # Normal at interface points
        n_at_pts = np.empty_like(coords)
        idx_tuple = tuple(interface_indices[:, k] for k in range(self.grid.dimension))
        for d in range(self.grid.dimension):
            n_at_pts[d] = normal[d][idx_tuple]

        # Offset positions
        offset = self.offset_distance
        idx_right = self._physical_to_index(coords + offset * n_at_pts)
        idx_left = self._physical_to_index(coords - offset * n_at_pts)

        # Compute normal derivative on each side: partial f / partial n = nabla f . n
        dn_right = np.zeros(len(interface_indices))
        dn_left = np.zeros(len(interface_indices))

        for d in range(self.grid.dimension):
            grad_d_right = map_coordinates(grad_field[d], idx_right, order=1, mode="nearest")
            grad_d_left = map_coordinates(grad_field[d], idx_left, order=1, mode="nearest")
            dn_right += grad_d_right * n_at_pts[d]
            dn_left += grad_d_left * n_at_pts[d]

        jump[idx_tuple] = dn_right - dn_left

        return jump

    def get_interface_velocity_field(self, jump_value: float) -> NDArray[np.float64]:
        """
        Create velocity field for level set evolution from interface velocity.

        Parameters
        ----------
        jump_value : float
            Interface velocity (e.g., from Stefan condition V = -κ·[∂T/∂n]).

        Returns
        -------
        velocity_field : NDArray
            Velocity field for level set evolution, same shape as grid.
            Constant near interface, zero elsewhere.

        Notes
        -----
        For level set evolution: ∂φ/∂t + V|∇φ| = 0
        The velocity V is typically computed from physics (Stefan condition,
        curvature flow, etc.) and needs to be extended to a field.

        Current implementation uses constant extension (simple but diffusive).
        Future: Fast marching or PDE-based extension.
        """
        velocity_field = np.zeros_like(self.interface_phi)

        # Apply velocity near interface
        velocity_field[self.interface_mask] = jump_value

        return velocity_field

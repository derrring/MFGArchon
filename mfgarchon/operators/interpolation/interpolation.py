"""
Interpolation operator for tensor product grids.

This module provides LinearOperator implementation of grid interpolation
for structured grids, wrapping scipy's RegularGridInterpolator.

Mathematical Background:
    Interpolation evaluates a grid function at arbitrary query points:
        u_interp(x_query) = ∑ᵢ wᵢ(x_query) · u_grid[i]

    where wᵢ are interpolation weights computed from:
        - **Linear** (1st-order): Trilinear in 3D, bilinear in 2D
        - **Cubic** (3rd-order): Tricubic in 3D, bicubic in 2D

    Properties:
        - **Linearity**: Interpolation is a linear operation
        - **Locality**: Each query point depends only on nearby grid points
        - **Conservation** (for piecewise constant): ∫u_interp dx = ∫u_grid dx

    Applications in MFG:
        - **Semi-Lagrangian**: Interpolate at foot-of-characteristic points
        - **Particle methods**: Transfer from grid to particles
        - **Post-processing**: Evaluate solution at specific locations

References:
    - Press et al (2007): Numerical Recipes, Ch. 3 (Interpolation)
    - Burden & Faires (2010): Numerical Analysis, Ch. 3
    - Falcone & Ferretti (1998): Semi-Lagrangian schemes for HJB

Created: 2026-01-17 (Issue #595 Phase 2 - Operator Refactoring)
Part of: Geometry Operator LinearOperator Migration
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import LinearOperator

if TYPE_CHECKING:
    from numpy.typing import NDArray


class InterpolationOperator(LinearOperator):
    """
    Discrete interpolation operator for tensor product grids.

    Implements scipy.sparse.linalg.LinearOperator interface for grid-to-point
    interpolation. Given a field defined on a regular grid, evaluates it at
    arbitrary query points via linear combination of grid values.

    **Mathematical Context**:
        Interpolation is a LINEAR operation:
            I(αu + βv) = αI(u) + βI(v)

        For each query point x_q, interpolated value is:
            u(x_q) = ∑ᵢ wᵢ(x_q) · u[i]

        where weights wᵢ depend only on grid geometry and query point location,
        NOT on field values.

    **Operator Shape**:
        Input:  Grid field u flattened to shape (N,) where N = ∏(Nx, Ny, ...)
        Output: Values at query points, shape (num_query,)
        Operator shape: (num_query, N)

        **Note**: This is a NON-SQUARE operator (unlike Laplacian, Divergence)

    **Performance**:
        Each _matvec call creates a new RegularGridInterpolator (scipy limitation).
        For repeated use with same field, caching at higher level is recommended.

    Attributes:
        grid_points: Tuple of 1D arrays defining grid coordinates (x, y, z, ...)
        query_points: Points at which to interpolate, shape (num_query, dimension)
        order: Interpolation order (1=linear, 3=cubic)
        extrapolation_mode: Handling of out-of-bounds points
        field_shape: Shape of grid fields (Nx, Ny, ...) inferred from grid_points
        shape: Operator shape (num_query, N)
        dtype: Data type (float64)

    Usage:
        >>> # Create operator for specific query points
        >>> x = np.linspace(0, 1, 50)
        >>> y = np.linspace(0, 1, 50)
        >>> grid_points = (x, y)
        >>> query_pts = np.array([[0.5, 0.5], [0.25, 0.75]])
        >>> interp_op = InterpolationOperator(
        ...     grid_points=grid_points,
        ...     query_points=query_pts,
        ...     order=1
        ... )
        >>>
        >>> # Apply via matrix-vector product
        >>> u_flat = u.ravel()
        >>> u_query_flat = interp_op @ u_flat  # Shape: (2,)
        >>>
        >>> # Apply via callable (preserves grid shape)
        >>> u_query = interp_op(u)  # Input: (50, 50), Output: (2,)

    Example (Semi-Lagrangian):
        >>> import numpy as np
        >>> from mfgarchon.operators import InterpolationOperator
        >>>
        >>> # Grid setup
        >>> Nx, Ny = 100, 100
        >>> x = np.linspace(0, 1, Nx)
        >>> y = np.linspace(0, 1, Ny)
        >>> grid_points = (x, y)
        >>>
        >>> # Semi-Lagrangian: backtrack characteristics by velocity × dt
        >>> dt = 0.01
        >>> v = np.random.rand(2, Nx, Ny)  # Velocity field
        >>> X, Y = np.meshgrid(x, y, indexing='ij')
        >>> foot_points_x = (X - dt * v[0]).ravel()
        >>> foot_points_y = (Y - dt * v[1]).ravel()
        >>> foot_points = np.column_stack([foot_points_x, foot_points_y])
        >>>
        >>> # Create interpolation operator for foot points
        >>> interp_op = InterpolationOperator(
        ...     grid_points=grid_points,
        ...     query_points=foot_points,
        ...     order=1
        ... )
        >>>
        >>> # Interpolate u at foot points
        >>> u = np.random.rand(Nx, Ny)
        >>> u_foot = interp_op(u)  # Shape: (Nx*Ny,)
        >>> u_next = u_foot.reshape(Nx, Ny)
    """

    def __init__(
        self,
        grid_points: tuple[NDArray, ...],
        query_points: NDArray,
        order: int = 1,
        extrapolation_mode: Literal["constant", "nearest", "boundary"] = "boundary",
    ):
        """
        Initialize interpolation operator.

        Args:
            grid_points: Tuple of 1D arrays defining grid coordinates
                Example: (x, y, z) where x, y, z are 1D arrays
            query_points: Points at which to interpolate, shape (num_query, dimension)
            order: Interpolation order
                - 1: Linear (bilinear in 2D, trilinear in 3D)
                - 3: Cubic (bicubic in 2D, tricubic in 3D)
            extrapolation_mode: Handling for out-of-bounds points
                - "constant": Use NaN for points outside grid
                - "nearest": Use nearest boundary value
                - "boundary": Project to boundary and interpolate

        Raises:
            ValueError: If order not in {1, 3}
            ValueError: If query_points.shape[1] != len(grid_points)
        """
        self.grid_points = grid_points
        self.query_points = query_points
        self.order = order
        self.extrapolation_mode = extrapolation_mode

        # Infer field shape and dimension
        self.field_shape = tuple(len(g) for g in grid_points)
        self.dimension = len(grid_points)

        # Validate query_points shape
        if query_points.ndim != 2:
            raise ValueError(f"query_points must be 2D array, got shape {query_points.shape}")

        num_query = query_points.shape[0]
        if query_points.shape[1] != self.dimension:
            raise ValueError(
                f"query_points dimension {query_points.shape[1]} doesn't match grid dimension {self.dimension}"
            )

        # Validate order
        if order not in (1, 3):
            raise ValueError(f"order must be 1 (linear) or 3 (cubic), got {order}")

        # Map order to scipy method
        self._method = {1: "linear", 3: "cubic"}[order]

        # Configure extrapolation
        if extrapolation_mode == "constant":
            self._fill_value = None  # scipy uses NaN by default
            self._bounds_error = False
        elif extrapolation_mode in ("nearest", "boundary"):
            # scipy's fill_value=None with bounds_error=False gives nearest extrapolation
            self._fill_value = None
            self._bounds_error = False
        else:
            raise ValueError(
                f"Unknown extrapolation_mode: {extrapolation_mode}. Use 'constant', 'nearest', or 'boundary'."
            )

        # Compute operator shape
        # Input: grid field (N,) where N = ∏field_shape
        # Output: query values (num_query,)
        N = int(np.prod(self.field_shape))
        super().__init__(shape=(num_query, N), dtype=np.float64)

    def _matvec(self, u_flat: NDArray) -> NDArray:
        """
        Interpolate grid field at query points.

        This is the core LinearOperator method required by scipy.

        Args:
            u_flat: Flattened grid field, shape (N,)

        Returns:
            Interpolated values at query points, shape (num_query,)

        Note:
            This creates a new RegularGridInterpolator on each call (scipy limitation).
            For performance-critical code with repeated interpolation, consider
            caching the operator or using sparse matrix representation.
        """
        # Reshape to grid
        u = u_flat.reshape(self.field_shape)

        # Create interpolator
        # Note: Cannot cache because field values change
        interpolator = RegularGridInterpolator(
            self.grid_points,
            u,
            method=self._method,
            bounds_error=self._bounds_error,
            fill_value=self._fill_value,
        )

        # Interpolate at query points
        return interpolator(self.query_points)

    def __call__(self, u: NDArray) -> NDArray:
        """
        Interpolate grid field at query points (convenience method).

        Args:
            u: Grid field, shape (Nx, Ny, ...)

        Returns:
            Interpolated values at query points, shape (num_query,)

        Example:
            >>> interp_op = InterpolationOperator(grid_points, query_pts, order=1)
            >>> u = np.random.rand(50, 50)
            >>> u_query = interp_op(u)  # Shape: (num_query,)
        """
        # Validate input shape
        if u.shape != self.field_shape:
            raise ValueError(f"Field shape {u.shape} doesn't match grid shape {self.field_shape}")

        # Apply via _matvec
        return self._matvec(u.ravel())

    def __repr__(self) -> str:
        """String representation of operator."""
        num_query = self.query_points.shape[0]
        return (
            f"InterpolationOperator(\n"
            f"  dimension={self.dimension},\n"
            f"  field_shape={self.field_shape},\n"
            f"  num_query_points={num_query},\n"
            f"  order={self.order} ({'linear' if self.order == 1 else 'cubic'}),\n"
            f"  extrapolation='{self.extrapolation_mode}',\n"
            f"  operator_shape={self.shape}\n"
            f")"
        )

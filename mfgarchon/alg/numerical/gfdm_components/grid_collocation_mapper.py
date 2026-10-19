"""
Grid-Collocation Mapping Component for GFDM Solvers.

This component provides bidirectional mapping between:
- Regular grid representation (for visualization, coupling with FDM solvers)
- Collocation point representation (for GFDM computation)

Extracted from GFDMInterpolationMixin as part of Issue #545 (mixin refactoring).
Uses composition pattern instead of inheritance for better testability and reusability.
"""

from __future__ import annotations

import numpy as np

from mfgarchon.utils.numerical.particle.interpolation import interpolate_grid_to_particles


class GridCollocationMapper:
    """
    Handles grid-to-collocation and collocation-to-grid interpolation for GFDM.

    This component enables seamless conversion between:
    - Regular grid representation (structured, for output/visualization)
    - Collocation point representation (unstructured, for GFDM computation)

    Grid → Collocation: Uses scipy RegularGridInterpolator for smooth mapping
    Collocation → Grid: Uses Delaunay triangulation with barycentric interpolation

    Both directions work on the grid's axes, its node coordinates per dimension. Given only bounds, the axes are
    uniform, ``np.linspace(lo, hi, n)``, which is a uniform grid's own; a graded grid passes its axes (#2756).

    The collocation-to-grid direction uses a pre-computed sparse interpolation
    matrix for efficiency (O(n*m) vs O(n^3) for repeated mappings).

    Composition Pattern (Issue #545):
        This component is injected into HJBGFDMSolver instead of using mixin inheritance.
        Benefits: Testable independently, reusable for FDM/FEM solvers, clear dependencies.

    Parameters
    ----------
    collocation_points : np.ndarray
        Collocation points, shape (n_points, dimension)
    grid_shape : tuple[int, ...]
        Shape of output grid (Nx, Ny, Nz, ...)
    domain_bounds : list[tuple[float, float]], optional
        Domain bounds per dimension: [(xmin, xmax), (ymin, ymax), ...], for a uniform grid. Needed only when
        ``grid_axes`` is not given.
    grid_axes : list[np.ndarray], optional
        The grid's node coordinates per dimension, of lengths ``grid_shape``.

    Attributes
    ----------
    collocation_points : np.ndarray
        Collocation points (N, dimension)
    n_points : int
        Number of collocation points
    grid_shape : tuple[int, ...]
        Output grid shape
    grid_axes : list[np.ndarray]
        The grid's node coordinates per dimension
    domain_bounds : list[tuple[float, float]]
        Domain bounds per dimension, the axes' end points
    _interp_matrix : scipy.sparse matrix | None
        Cached sparse interpolation matrix (collocation → grid)
    _coll_tree : scipy.spatial.cKDTree | None
        KDTree for nearest neighbor fallback

    Examples
    --------
    >>> # Create mapper for 2D GFDM solver
    >>> collocation = np.random.uniform(0, 1, (500, 2))
    >>> mapper = GridCollocationMapper(
    ...     collocation_points=collocation,
    ...     grid_shape=(21, 21),
    ...     domain_bounds=[(0, 1), (0, 1)]
    ... )
    >>>
    >>> # Map grid values to collocation points
    >>> u_grid = np.random.randn(21, 21).flatten()
    >>> u_coll = mapper.map_grid_to_collocation(u_grid)
    >>> assert u_coll.shape == (500,)
    >>>
    >>> # Map collocation values back to grid
    >>> u_grid_reconstructed = mapper.map_collocation_to_grid(u_coll)
    >>> assert u_grid_reconstructed.shape == (441,)  # 21*21
    >>>
    >>> # A graded 1D grid passes its own nodes: a linear field maps exactly
    >>> nodes = np.linspace(0, 1, 11) ** 2
    >>> graded = GridCollocationMapper(np.linspace(0, 1, 7).reshape(-1, 1), grid_shape=(11,), grid_axes=[nodes])
    >>> bool(np.allclose(graded.map_grid_to_collocation(nodes), np.linspace(0, 1, 7)))
    True
    """

    def __init__(
        self,
        collocation_points: np.ndarray,
        grid_shape: tuple[int, ...],
        domain_bounds: list[tuple[float, float]] | None = None,
        *,
        grid_axes: list[np.ndarray] | None = None,
    ):
        """Initialize grid-collocation mapper."""
        self.collocation_points = np.asarray(collocation_points)
        self.n_points = len(collocation_points)
        self.grid_shape = grid_shape
        if grid_axes is None:
            if domain_bounds is None:
                raise ValueError("GridCollocationMapper needs the grid's axes, or the bounds of a uniform grid.")
            grid_axes = [np.linspace(lo, hi, n) for (lo, hi), n in zip(domain_bounds, grid_shape, strict=True)]
        self.grid_axes = [np.asarray(axis, dtype=float) for axis in grid_axes]
        if tuple(axis.size for axis in self.grid_axes) != tuple(grid_shape):
            raise ValueError(
                f"grid_axes have lengths {tuple(axis.size for axis in self.grid_axes)}, not grid_shape {tuple(grid_shape)}."
            )
        self.domain_bounds = [(float(axis[0]), float(axis[-1])) for axis in self.grid_axes]
        # A singleton axis carries no interpolation: e.g. (21, 1) is read as a 1D grid of 21 nodes.
        self._active = [k for k, axis in enumerate(self.grid_axes) if axis.size > 1] or [0]

        # Cached interpolation matrix (built lazily on first use)
        self._interp_matrix = None
        self._coll_tree = None  # KDTree for nearest neighbor fallback

    def map_grid_to_collocation(self, u_grid: np.ndarray) -> np.ndarray:
        """
        Map values from regular grid to collocation points using interpolation.

        Uses scipy RegularGridInterpolator via particle interpolation utilities, on the grid's axes.

        Parameters
        ----------
        u_grid : np.ndarray
            Values on regular grid, flattened to 1D

        Returns
        -------
        np.ndarray
            Values at collocation points, shape (n_points,)
        """
        axes = [self.grid_axes[k] for k in self._active]
        values = np.asarray(u_grid).reshape(tuple(axis.size for axis in axes))
        return interpolate_grid_to_particles(
            values,
            grid_bounds=tuple((float(axis[0]), float(axis[-1])) for axis in axes),
            particle_positions=self.collocation_points[:, self._active],
            method="linear",
            grid_axes=axes,
        )

    def map_collocation_to_grid(self, u_collocation: np.ndarray) -> np.ndarray:
        """
        Map values from collocation points to regular grid using cached interpolation.

        Uses pre-computed interpolation matrix for efficiency (O(n*m) vs O(n^3)).

        Parameters
        ----------
        u_collocation : np.ndarray
            Values at collocation points, shape (n_points,)

        Returns
        -------
        np.ndarray
            Values on regular grid, flattened to 1D
        """
        # Build interpolation matrix on first use (lazy initialization)
        if self._interp_matrix is None:
            self._build_interpolation_matrix()

        return self._interp_matrix @ u_collocation

    def map_grid_to_collocation_batch(self, U_grid: np.ndarray) -> np.ndarray:
        """
        Batch version of map_grid_to_collocation for time-series data.

        Parameters
        ----------
        U_grid : np.ndarray
            Shape (Nt, ...) where ... represents spatial dimensions

        Returns
        -------
        np.ndarray
            U_collocation with shape (Nt, n_points)
        """
        Nt = U_grid.shape[0]
        U_collocation = np.zeros((Nt, self.n_points))

        for n in range(Nt):
            u_grid_flat = U_grid[n].flatten()
            U_collocation[n, :] = self.map_grid_to_collocation(u_grid_flat)

        return U_collocation

    def map_collocation_to_grid_batch(self, U_collocation: np.ndarray) -> np.ndarray:
        """
        Batch version of map_collocation_to_grid for time-series data.

        Parameters
        ----------
        U_collocation : np.ndarray
            Shape (Nt, n_points)

        Returns
        -------
        np.ndarray
            U_grid with shape (Nt, ...) where ... matches original spatial shape
        """
        Nt = U_collocation.shape[0]
        n_spatial_points = int(np.prod(self.grid_shape))

        # Map each timestep
        U_grid_flat = np.zeros((Nt, n_spatial_points))
        for n in range(Nt):
            U_grid_flat[n, :] = self.map_collocation_to_grid(U_collocation[n, :])

        # Reshape to original spatial dimensions
        output_shape = (Nt, *self.grid_shape)
        return U_grid_flat.reshape(output_shape)

    def _build_interpolation_matrix(self) -> None:
        """
        Pre-compute sparse interpolation matrix for collocation → grid mapping.

        Uses scipy LinearNDInterpolator with barycentric weights for efficiency.
        The matrix is sparse due to triangulation locality.

        For 1D: Uses linear interpolation between sorted collocation points
        For 2D+: Uses Delaunay triangulation with barycentric coordinates
        """
        from scipy.spatial import Delaunay

        axes = [self.grid_axes[k] for k in self._active]
        ndim = len(axes)
        coll = self.collocation_points[:, self._active]

        # The grid's nodes, in its C order
        grid_points = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], axis=1)

        n_grid = grid_points.shape[0]
        n_coll = self.n_points

        # Build interpolation matrix using Delaunay triangulation
        # Each grid point is expressed as weighted sum of collocation points
        try:
            # Special handling for 1D: Delaunay doesn't work for 1D points
            if ndim == 1:
                # For 1D, use linear interpolation weights
                from scipy.sparse import lil_matrix

                interp_matrix = lil_matrix((n_grid, n_coll))

                # Sort collocation points for 1D interpolation
                coll_x = coll[:, 0]
                sort_idx = np.argsort(coll_x)
                coll_x_sorted = coll_x[sort_idx]
                grid_x = grid_points[:, 0]

                # For each grid point, find surrounding collocation points and compute weights
                for i, x in enumerate(grid_x):
                    # Find where x would be inserted to maintain sorted order
                    idx_right = np.searchsorted(coll_x_sorted, x)

                    if idx_right == 0:
                        # x is before all collocation points - use nearest
                        interp_matrix[i, sort_idx[0]] = 1.0
                    elif idx_right == n_coll:
                        # x is after all collocation points - use nearest
                        interp_matrix[i, sort_idx[-1]] = 1.0
                    else:
                        # x is between two collocation points - use linear interpolation
                        idx_left = idx_right - 1
                        x_left = coll_x_sorted[idx_left]
                        x_right = coll_x_sorted[idx_right]

                        # Linear interpolation weights
                        weight_right = (x - x_left) / (x_right - x_left)
                        weight_left = 1.0 - weight_right

                        interp_matrix[i, sort_idx[idx_left]] = weight_left
                        interp_matrix[i, sort_idx[idx_right]] = weight_right

                self._interp_matrix = interp_matrix.tocsr()
            else:
                # For 2D+, use Delaunay triangulation
                tri = Delaunay(coll)
                simplex_indices = tri.find_simplex(grid_points)

                # Build sparse matrix
                from scipy.sparse import lil_matrix

                interp_matrix = lil_matrix((n_grid, n_coll))

                for i, (pt, simplex_idx) in enumerate(zip(grid_points, simplex_indices, strict=True)):
                    if simplex_idx >= 0:
                        # Point inside triangulation - use barycentric coords
                        simplex = tri.simplices[simplex_idx]
                        b = tri.transform[simplex_idx, :ndim].dot(pt - tri.transform[simplex_idx, ndim])
                        bary = np.append(b, 1 - b.sum())
                        for j, weight in zip(simplex, bary, strict=True):
                            interp_matrix[i, j] = weight
                    else:
                        # Point outside - use nearest neighbor
                        from scipy.spatial import cKDTree

                        if self._coll_tree is None:
                            self._coll_tree = cKDTree(coll)
                        _, nearest_idx = self._coll_tree.query(pt)
                        interp_matrix[i, nearest_idx] = 1.0

                self._interp_matrix = interp_matrix.tocsr()
        except Exception as e:
            raise RuntimeError(f"Failed to build interpolation matrix: {e}") from e

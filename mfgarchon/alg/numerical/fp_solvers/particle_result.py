"""
Particle Solver Result with Direct Density Query Support.

Provides result container for particle-based FP solver that supports both
traditional grid-based density access and efficient direct queries at arbitrary points.

This enables:
- Grid-based density: M[t, i, j] for visualization
- Direct queries: query_density(x, t) for HJB coupling
- Particle access: get_particles(t) for analysis

Created: 2026-01-18 (Issue #489 Phase 2 - FP Solver Integration)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.particle_density_query import ParticleDensityQuery
from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)


class FPParticleResult:
    """
    Result container for particle-based FP solver with query support.

    Stores both grid-based density (for compatibility) and particle positions
    (for direct queries), enabling efficient density estimation at arbitrary points.

    **The two representations do not share a mass convention, and are not interchangeable.**
    `M_grid` carries the mass the caller handed to the solver: since #2181 `solve_fp_system` scales
    it so that `geometry.integrate(M_grid[0])` equals the caller's initial mass, because a particle
    method carries no mass of its own -- positions have no scale, and a KDE reconstruction returns
    about 1 whatever went in.

    `query_density` carries NO mass convention. It rebuilds a density from `particle_history` and
    never consults `M_grid`, so its output is unaffected by that scaling, and its three methods do
    not even agree with each other. Measured on one fixture whose `M_grid` mass is 0.300000:

        kernel  2.506588      knn  0.989115      hybrid  0.290921

    -- a spread of 8.6x between methods. This predates #2181, which changed only which number they
    disagree with (`M_grid` was 1.0 before). Do not integrate a `query_density` result and read it as
    a mass, and do not compare one against `M_grid`. Tracked separately.

    Attributes
    ----------
    M_grid : NDArray
        Density on grid, shape (Nt+1, *grid_shape).
    particle_history : list[NDArray] | None
        Particle positions at each timestep, each shape (N_particles, dimension).
        None if density_mode="grid_only".
    time_grid : NDArray
        Time points, shape (Nt+1,).
    bandwidth : float or {"scott", "silverman"} or None
        KDE bandwidth: a factor of each timestep's sample spread, or a rule (CONVENTIONS.md § 9).
    deposited_levels : tuple[int, ...]
        The time levels whose `M_grid` slice is the sample deposited on the grid's nodes, because the grid could not
        read its kernel (`FPParticleSolver.deposited_levels`, #2556).

    Examples
    --------
    >>> # Traditional grid access (unchanged)
    >>> result = solver.solve(...)
    >>> m_at_t5 = result.M_grid[5]  # Density at t=5 on full grid
    >>>
    >>> # Direct density query (new)
    >>> x_query = np.array([0.5, 0.5])
    >>> density = result.query_density(x_query, timestep=5, method="hybrid")
    >>>
    >>> # Batch query along characteristic
    >>> characteristic_points = np.linspace([0, 0], [1, 1], 50).reshape(-1, 2)
    >>> densities = result.query_density(characteristic_points, timestep=5)
    """

    def __init__(
        self,
        M_grid: NDArray[np.float64],
        time_grid: NDArray[np.float64],
        particle_history: list[NDArray[np.float64]] | None = None,
        bandwidth: float | str | None = None,
        deposited_levels: tuple[int, ...] = (),
    ):
        """
        Initialize particle result container.

        Parameters
        ----------
        M_grid : NDArray
            Density on grid, shape (Nt+1, *grid_shape).
        time_grid : NDArray
            Time points, shape (Nt+1,).
        particle_history : list[NDArray], optional
            Particle positions at each timestep. If None, direct queries unavailable.
        bandwidth : float or {"scott", "silverman"}, optional
            KDE bandwidth: a factor of each timestep's sample spread, or a rule, as the solver's ``kde_bandwidth`` is
            (CONVENTIONS.md § 9). If None, the query's own Scott rule.
        deposited_levels : tuple[int, ...], optional
            The time levels whose slice is a deposit rather than a kernel estimate (#2556). Empty if none.
        """
        self.M_grid = M_grid
        self.time_grid = time_grid
        self.particle_history = particle_history
        self.bandwidth = bandwidth
        self.deposited_levels = deposited_levels

        # Cache for density query objects (one per timestep)
        self._query_cache: dict[int, ParticleDensityQuery] = {}

        logger.debug(
            f"FPParticleResult created: Nt={len(time_grid) - 1}, "
            f"grid_shape={M_grid.shape[1:]}, "
            f"particles_stored={particle_history is not None}"
        )

    def query_density(
        self,
        query_points: NDArray[np.float64],
        timestep: int,
        method: Literal["kernel", "knn", "hybrid"] = "hybrid",
        k: int = 50,
    ) -> NDArray[np.float64]:
        """
        Query density at arbitrary points using particle positions.

        Parameters
        ----------
        query_points : NDArray
            Query point locations, shape (N_query, dimension) or (dimension,) for single point.
        timestep : int
            Timestep index (0 to Nt).
        method : str, default="hybrid"
            Density estimation method:
            - "kernel": Exact Gaussian KDE (slow, matches grid exactly)
            - "knn": k-nearest neighbors (fast, discontinuous)
            - "hybrid": Kernel on k-NN subset (recommended)
        k : int, default=50
            Number of nearest neighbors for "knn" and "hybrid" methods.

        Returns
        -------
        densities : NDArray
            Estimated density at query points, shape (N_query,).

        Raises
        ------
        ValueError
            If particle_history is None (solver was run with density_mode="grid_only").
        IndexError
            If timestep is out of range.

        Examples
        --------
        >>> # Query density along a path
        >>> path = np.linspace([0, 0], [1, 1], 100).reshape(-1, 2)
        >>> densities = result.query_density(path, timestep=10, method="hybrid")
        >>>
        >>> # Query at specific point
        >>> x = np.array([0.5, 0.5])
        >>> density = result.query_density(x, timestep=10)
        """
        if self.particle_history is None:
            raise ValueError(
                "Direct density queries require particle history. "
                "Solver was run with density_mode='grid_only'. "
                "Re-run with density_mode='hybrid' or 'query_only' to enable queries."
            )

        if timestep < 0 or timestep >= len(self.time_grid):
            raise IndexError(f"Timestep {timestep} out of range [0, {len(self.time_grid) - 1}]")

        # Get or create ParticleDensityQuery for this timestep
        if timestep not in self._query_cache:
            particles_t = self.particle_history[timestep]

            # Create query object. Its numeric bandwidth is an absolute isotropic width, so a factor or a rule is
            # converted here, at this timestep, to the mean of the kernel's per-axis widths (#2556).
            if self.bandwidth is None:
                self._query_cache[timestep] = ParticleDensityQuery(particles_t, bandwidth_rule="scott")
            else:
                from mfgarchon.utils.numerical.particle.interpolation import kde_kernel_widths

                self._query_cache[timestep] = ParticleDensityQuery(
                    particles_t,
                    bandwidth=float(np.mean(kde_kernel_widths(particles_t, self.bandwidth))),
                    bandwidth_rule="fixed",
                )

        # Query density
        return self._query_cache[timestep].query_density(query_points, method=method, k=k)

    def get_particles(self, timestep: int) -> NDArray[np.float64]:
        """
        Get particle positions at specific timestep.

        Parameters
        ----------
        timestep : int
            Timestep index (0 to Nt).

        Returns
        -------
        particles : NDArray
            Particle positions, shape (N_particles, dimension).

        Raises
        ------
        ValueError
            If particle_history is None.
        IndexError
            If timestep is out of range.
        """
        if self.particle_history is None:
            raise ValueError("Particle history not stored (density_mode='grid_only')")

        if timestep < 0 or timestep >= len(self.time_grid):
            raise IndexError(f"Timestep {timestep} out of range [0, {len(self.time_grid) - 1}]")

        return self.particle_history[timestep]

    def get_grid_density(self, timestep: int) -> NDArray[np.float64]:
        """
        Get grid-based density at specific timestep.

        Parameters
        ----------
        timestep : int
            Timestep index (0 to Nt).

        Returns
        -------
        density : NDArray
            Density on grid, shape (*grid_shape,).
        """
        if timestep < 0 or timestep >= len(self.time_grid):
            raise IndexError(f"Timestep {timestep} out of range [0, {len(self.time_grid) - 1}]")

        return self.M_grid[timestep]

    @property
    def has_particle_queries(self) -> bool:
        """Check if direct particle queries are available."""
        return self.particle_history is not None

    def __repr__(self) -> str:
        """String representation for debugging."""
        Nt = len(self.time_grid) - 1
        grid_shape = self.M_grid.shape[1:]
        has_particles = "Yes" if self.particle_history is not None else "No"
        n_particles = len(self.particle_history[0]) if self.particle_history else 0

        return (
            f"FPParticleResult(\n"
            f"  Nt={Nt},\n"
            f"  grid_shape={grid_shape},\n"
            f"  has_particles={has_particles},\n"
            f"  n_particles={n_particles},\n"
            f"  bandwidth={self.bandwidth}\n"
            f")"
        )

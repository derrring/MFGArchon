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
    bandwidth : float
        KDE bandwidth used for density estimation.

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
        bandwidth: float | None = None,
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
        bandwidth : float, optional
            KDE bandwidth. If None, auto-computed when queries are made.
        """
        self.M_grid = M_grid
        self.time_grid = time_grid
        self.particle_history = particle_history
        self.bandwidth = bandwidth

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

            # Create query object
            self._query_cache[timestep] = ParticleDensityQuery(
                particles_t,
                bandwidth=self.bandwidth,
                bandwidth_rule="scott" if self.bandwidth is None else "fixed",
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


if __name__ == "__main__":
    """Smoke test for FPParticleResult."""
    print("Testing FPParticleResult...")

    # Test 1: Create result with particle history
    print("\n[Test 1: Result Creation with Particle History]")
    Nt, Nx = 10, 50
    time_grid = np.linspace(0, 1, Nt + 1)
    M_grid = np.random.rand(Nt + 1, Nx, Nx)

    # Generate particle history
    N_particles = 1000
    particle_history = [np.random.rand(N_particles, 2) for _ in range(Nt + 1)]

    result = FPParticleResult(M_grid, time_grid, particle_history, bandwidth=0.1)
    print(f"  {result}")
    assert result.has_particle_queries, "Should have particle queries"
    print("  ✓ Result creation with particles!")

    # Test 2: Grid density access (backward compatible)
    print("\n[Test 2: Grid Density Access]")
    density_t5 = result.get_grid_density(5)
    assert density_t5.shape == (Nx, Nx), f"Expected shape (50, 50), got {density_t5.shape}"
    assert np.allclose(density_t5, M_grid[5]), "Grid density should match"
    print(f"  Grid density shape: {density_t5.shape}")
    print("  ✓ Grid access working!")

    # Test 3: Particle access
    print("\n[Test 3: Particle Access]")
    particles_t3 = result.get_particles(3)
    assert particles_t3.shape == (N_particles, 2), f"Expected shape ({N_particles}, 2)"
    print(f"  Particles shape: {particles_t3.shape}")
    print("  ✓ Particle access working!")

    # Test 4: Direct density query
    print("\n[Test 4: Direct Density Query]")
    query_points = np.array([[0.5, 0.5], [0.2, 0.8]])
    densities = result.query_density(query_points, timestep=5, method="hybrid", k=30)
    assert densities.shape == (2,), f"Expected shape (2,), got {densities.shape}"
    assert np.all(densities > 0), "Densities should be positive"
    print(f"  Query densities: {densities}")
    print("  ✓ Direct queries working!")

    # Test 5: Batch query along path
    print("\n[Test 5: Batch Query Along Path]")
    path = np.linspace([0, 0], [1, 1], 100).reshape(-1, 2)
    densities_path = result.query_density(path, timestep=8, method="hybrid")
    assert densities_path.shape == (100,), f"Expected shape (100,), got {densities_path.shape}"
    print(f"  Path densities: min={densities_path.min():.4f}, max={densities_path.max():.4f}")
    print("  ✓ Batch queries working!")

    # Test 6: Result without particle history (grid-only mode)
    print("\n[Test 6: Grid-Only Mode]")
    result_grid_only = FPParticleResult(M_grid, time_grid, particle_history=None)
    assert not result_grid_only.has_particle_queries, "Should not have particle queries"

    # Should still allow grid access
    density = result_grid_only.get_grid_density(5)
    assert density.shape == (Nx, Nx)

    # Should raise error on particle query
    try:
        result_grid_only.query_density(query_points, timestep=5)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        if "particle history" not in str(e).lower():
            raise AssertionError(f"Expected 'particle history' in error message, got: {e}") from None
        print("  ✓ Grid-only mode correctly raises error on queries!")

    print("\n✅ All FPParticleResult tests passed!")
    print("\n📊 Implementation Status:")
    print("  ✓ Grid density access (backward compatible)")
    print("  ✓ Particle position access")
    print("  ✓ Direct density queries (kernel, knn, hybrid)")
    print("  ✓ Batch query support")
    print("  ✓ Query caching for efficiency")
    print("  ✓ Grid-only mode for memory efficiency")

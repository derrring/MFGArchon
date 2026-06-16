"""
Measure representations for MFG on measure space (Layer 2).

Provides the ``MeasureRepresentation`` Protocol and concrete implementations
for representing probability measures mu in P_2(R^d). These are the building
blocks for:
- Master Equation: v(x, mu, t) where mu is a measure argument
- Common Noise PDE: conditional measures mu^theta evolving in measure space
- Lions derivative: delta U / delta m computed via measure perturbation

Issue #956: Layer 2 alignment.

Current implementations:
    - ParticleMeasure: empirical measure m_N = (1/N) sum_i w_i delta_{y_i}

Planned (not yet implemented):
    - BasisExpansionMeasure: density expanded in basis (d <= 2)
    - NeuralMeasure: neural network parametrization (high-dimensional)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


logger = get_logger(__name__)


@runtime_checkable
class MeasureRepresentation(Protocol):
    """Protocol for probability measure representations.

    Any object satisfying this protocol can be used as a measure argument
    in MFG solvers operating on measure space (Layer 2).

    Methods:
        to_density: Project measure onto a spatial grid as density array.
        total_mass: Return total mass (should be 1.0 for probability measures).
        dimension: Spatial dimension of the measure support.
    """

    def to_density(self, grid_points: NDArray) -> NDArray:
        """Project measure onto spatial grid as density array.

        Args:
            grid_points: Spatial grid, shape (N,) for 1D or (N, d) for dD.

        Returns:
            Density values at grid points, shape (N,).
        """
        ...

    def total_mass(self) -> float:
        """Return total mass of the measure."""
        ...

    @property
    def dimension(self) -> int:
        """Spatial dimension of the measure support."""
        ...


class ParticleMeasure:
    """Empirical measure: m_N = sum_i w_i delta_{y_i}.

    Represents a probability measure as a weighted sum of Dirac masses
    at particle positions. This is the natural representation for:
    - Monte Carlo methods (common noise sampling)
    - N-player game convergence
    - Particle-based FP solvers
    - Functional derivative computation via position perturbation

    Parameters
    ----------
    positions : NDArray
        Particle positions, shape (N,) for 1D or (N, d) for dD.
    weights : NDArray | None
        Particle weights, shape (N,). Default: uniform 1/N.

    Example
    -------
    >>> mu = ParticleMeasure(np.array([0.2, 0.5, 0.8]))
    >>> density = mu.to_density(np.linspace(0, 1, 101))
    >>> mu.total_mass()  # 1.0
    """

    def __init__(self, positions: NDArray, weights: NDArray | None = None):
        self._positions = np.asarray(positions, dtype=float)
        if self._positions.ndim == 1:
            self._positions = self._positions.reshape(-1, 1)
        self._n_particles = self._positions.shape[0]
        self._dim = self._positions.shape[1]

        if weights is None:
            self._weights = np.ones(self._n_particles) / self._n_particles
        else:
            self._weights = np.asarray(weights, dtype=float)
            self._weights = self._weights / self._weights.sum()

    @property
    def positions(self) -> NDArray:
        """Particle positions, shape (N, d)."""
        return self._positions

    @property
    def weights(self) -> NDArray:
        """Particle weights, shape (N,), sum to 1."""
        return self._weights

    @property
    def n_particles(self) -> int:
        """Number of particles."""
        return self._n_particles

    @property
    def dimension(self) -> int:
        """Spatial dimension."""
        return self._dim

    def total_mass(self) -> float:
        """Total mass (should be 1.0 for probability measure)."""
        return float(self._weights.sum())

    def to_density(self, grid_points: NDArray, bandwidth: float | None = None) -> NDArray:
        """Project onto spatial grid via kernel density estimation.

        Uses Gaussian KDE with bandwidth selected by Scott's rule if not specified.

        Args:
            grid_points: Grid points, shape (M,) for 1D or (M, d) for dD.
            bandwidth: KDE bandwidth. None = Scott's rule: h = N^{-1/(d+4)} * std.

        Returns:
            Density values at grid points, shape (M,).
        """
        grid = np.asarray(grid_points, dtype=float)
        if grid.ndim == 1:
            grid = grid.reshape(-1, 1)

        if self._dim == 1:
            return self._kde_1d(grid[:, 0], bandwidth)

        # General dD KDE
        return self._kde_nd(grid, bandwidth)

    def _kde_1d(self, x: NDArray, bandwidth: float | None) -> NDArray:
        """1D kernel density estimation."""
        positions = self._positions[:, 0]
        if bandwidth is None:
            std = np.std(positions)
            if std > 0:
                bandwidth = std * self._n_particles ** (-1 / 5)
            else:
                # Issue #1071: degenerate cloud (zero spread) — fall back to a fixed bandwidth
                # to avoid div-by-zero, but surface it (the density estimate is unreliable).
                logger.warning(
                    "MeasureField 1D KDE: particle cloud has zero spread (std=0, degenerate); "
                    "using fallback bandwidth 0.1 — the density estimate is unreliable."
                )
                bandwidth = 0.1

        # Gaussian kernel: K(u) = (1/sqrt(2*pi)) * exp(-u^2/2)
        # density(x) = sum_i w_i * K((x - y_i) / h) / h
        diff = x[:, None] - positions[None, :]  # (M, N)
        kernel_vals = np.exp(-0.5 * (diff / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))
        return kernel_vals @ self._weights  # (M,)

    def _kde_nd(self, x: NDArray, bandwidth: float | None) -> NDArray:
        """General dD kernel density estimation (product kernel)."""
        d = self._dim
        if bandwidth is None:
            std = np.std(self._positions, axis=0).mean()
            if std > 0:
                bandwidth = std * self._n_particles ** (-1 / (d + 4))
            else:
                logger.warning(
                    "MeasureField %dD KDE: particle cloud has zero spread (std=0, degenerate); "
                    "using fallback bandwidth 0.1 — the density estimate is unreliable.",
                    d,
                )
                bandwidth = 0.1

        M = x.shape[0]
        density = np.zeros(M)
        for i in range(self._n_particles):
            diff = x - self._positions[i]  # (M, d)
            sq_dist = np.sum(diff**2, axis=1)  # (M,)
            kernel_vals = np.exp(-0.5 * sq_dist / bandwidth**2) / (bandwidth * np.sqrt(2 * np.pi)) ** d
            density += self._weights[i] * kernel_vals
        return density

    def wasserstein_distance(self, other: ParticleMeasure, p: int = 2) -> float:
        """Compute p-Wasserstein distance to another ParticleMeasure.

        For 1D with equal weights, uses the exact formula via sorted quantiles:
            W_p(mu, nu) = (int_0^1 |F_mu^{-1}(t) - F_nu^{-1}(t)|^p dt)^{1/p}

        For general case (nD or unequal weights), uses the discrete optimal
        transport formulation via scipy.optimize.linear_sum_assignment on
        the pairwise cost matrix.

        Args:
            other: Another ParticleMeasure.
            p: Order of Wasserstein distance (1 or 2). Default 2.

        Returns:
            W_p(self, other) >= 0.

        Raises:
            ValueError: If dimensions don't match.
        """
        if self._dim != other._dim:
            raise ValueError(f"Dimension mismatch: {self._dim} vs {other._dim}")

        # 1D with equal uniform weights: exact sorted quantile formula
        if self._dim == 1 and self._n_particles == other._n_particles and np.allclose(self._weights, other._weights):
            x_sorted = np.sort(self._positions[:, 0])
            y_sorted = np.sort(other._positions[:, 0])
            if p == 1:
                return float(np.mean(np.abs(x_sorted - y_sorted)))
            return float(np.mean(np.abs(x_sorted - y_sorted) ** p) ** (1.0 / p))

        # General case: discrete OT via assignment problem
        # Cost matrix: C_{ij} = |x_i - y_j|^p
        diff = self._positions[:, None, :] - other._positions[None, :, :]  # (N, M, d)
        cost = np.sum(diff**2, axis=2)  # (N, M) squared distances
        if p == 1:
            cost = np.sqrt(cost)
        # For p=2: cost is already |x-y|^2, take p-th root at the end

        # Equal weights: use linear_sum_assignment (Hungarian algorithm)
        if self._n_particles == other._n_particles and np.allclose(self._weights, other._weights):
            from scipy.optimize import linear_sum_assignment

            row_ind, col_ind = linear_sum_assignment(cost)
            total_cost = cost[row_ind, col_ind].mean()
            if p == 2:
                return float(np.sqrt(total_cost))
            return float(total_cost ** (1.0 / p))

        # Unequal weights or different N: use scipy 1D or POT if available
        if self._dim == 1:
            from scipy.stats import wasserstein_distance as _w1d

            if p == 1:
                return float(
                    _w1d(
                        self._positions[:, 0],
                        other._positions[:, 0],
                        u_weights=self._weights,
                        v_weights=other._weights,
                    )
                )
            # For p=2, 1D: use quantile integration
            # Approximate via sorted weighted quantiles
            return self._wasserstein_1d_weighted(other, p)

        # nD unequal weights: Sinkhorn-regularized OT (pure numpy)
        if p == 1:
            return self._sinkhorn_distance(cost, self._weights, other._weights, reg=0.1)
        # p=2: cost matrix is squared distances, Sinkhorn on it gives W_2^2
        return float(np.sqrt(self._sinkhorn_distance(cost, self._weights, other._weights, reg=0.1)))

    @staticmethod
    def _sinkhorn_distance(
        cost: NDArray, a: NDArray, b: NDArray, reg: float = 0.1, max_iter: int = 500, tol: float = 1e-9
    ) -> float:
        """Sinkhorn-regularized OT distance (pure numpy, no external deps).

        For large N where Hungarian O(N^3) is too slow. Complexity O(N^2 / reg).
        Approximation improves as reg -> 0 (but numerical stability degrades).

        Args:
            cost: Cost matrix, shape (N, M).
            a: Source weights, shape (N,), sum to 1.
            b: Target weights, shape (M,), sum to 1.
            reg: Entropic regularization. Smaller = more accurate.
            max_iter: Maximum Sinkhorn iterations.
            tol: Convergence tolerance on marginal error.
        """
        K = np.exp(-cost / reg)
        u = np.ones_like(a)
        for _ in range(max_iter):
            u_prev = u
            u = a / (K @ (b / (K.T @ u)))
            if np.max(np.abs(u - u_prev)) < tol:
                break
        v = b / (K.T @ u)
        P = u[:, None] * K * v[None, :]
        return float(np.sum(P * cost))

    def _wasserstein_1d_weighted(self, other: ParticleMeasure, p: int) -> float:
        """1D Wasserstein-p for weighted empirical measures via quantile matching."""
        # Sort both by position
        idx_self = np.argsort(self._positions[:, 0])
        idx_other = np.argsort(other._positions[:, 0])

        x = self._positions[idx_self, 0]
        w_x = self._weights[idx_self]
        y = other._positions[idx_other, 0]
        w_y = other._weights[idx_other]

        # Merge CDFs and compute integral
        cdf_x = np.cumsum(w_x)
        cdf_y = np.cumsum(w_y)

        # Piecewise constant quantile functions
        all_levels = np.unique(np.concatenate([np.array([0.0]), cdf_x, cdf_y, np.array([1.0])]))
        cost = 0.0
        for i in range(len(all_levels) - 1):
            t = 0.5 * (all_levels[i] + all_levels[i + 1])
            dt = all_levels[i + 1] - all_levels[i]
            qx = x[np.searchsorted(cdf_x, t, side="left").clip(0, len(x) - 1)]
            qy = y[np.searchsorted(cdf_y, t, side="left").clip(0, len(y) - 1)]
            cost += abs(qx - qy) ** p * dt
        return float(cost ** (1.0 / p))

    @classmethod
    def from_density(
        cls,
        density: NDArray,
        grid_points: NDArray,
        n_particles: int | None = None,
    ) -> ParticleMeasure:
        """Construct ParticleMeasure from grid density via weighted sampling.

        Converts a density array m(x_i) on a grid to a particle representation
        by placing particles at grid points with weights proportional to m(x_i).

        Args:
            density: Density values, shape (N,). Must be non-negative.
            grid_points: Grid point positions, shape (N,) for 1D or (N, d) for dD.
            n_particles: If None, use all grid points with non-zero density.
                If specified, subsample to n_particles by importance sampling.

        Returns:
            ParticleMeasure with particles at grid points.
        """
        density = np.asarray(density, dtype=float).ravel()
        grid = np.asarray(grid_points, dtype=float)
        if grid.ndim == 1:
            grid = grid.reshape(-1, 1)

        # Filter zero-density points
        mask = density > 0
        positions = grid[mask]
        weights = density[mask]

        if n_particles is not None and n_particles < len(positions):
            # Importance sampling: keep top-n by weight
            top_idx = np.argsort(weights)[-n_particles:]
            positions = positions[top_idx]
            weights = weights[top_idx]

        return cls(positions, weights)

    def __repr__(self) -> str:
        return f"ParticleMeasure(n={self._n_particles}, d={self._dim})"

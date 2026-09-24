"""
Measure-dependent fields for Layer 2 MFG.

A MeasureField represents v(t, x, mu) — a function on the product space
Omega x P_2(R^d) x [0,T]. This is the central object in:

- Master equation: dU/dt + H(x, nabla_x U, mu) + integral of Lions derivative = 0
- Common noise MFG: conditional value function U^theta(t, x)
- Sensitivity analysis: how the equilibrium responds to measure perturbation

The key distinction from classical MFG is that v depends on the full
measure mu, not just the local density m(x). This enables:
- Lions derivative computation: delta v / delta mu
- Wasserstein continuity: v(x, mu_1, t) close to v(x, mu_2, t) when W_2(mu_1, mu_2) small
- Master equation discretization (future)

Issue #956: Part of Layer 2 (Measure-Dependent MFG).

Mathematical reference:
    Carmona & Delarue (2018), Vol II, Ch. 5: The Master Equation
    Lions (2007-2011), College de France lectures on Mean Field Games
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from mfgarchon.types.callable_protocols import MEASURE_FIELD_SLOTS, BoundCallable, bound_attribute

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.core.measure import MeasureRepresentation


@runtime_checkable
class MeasureFieldProtocol(Protocol):
    """Protocol for measure-dependent fields v(t, x, mu).

    Any object satisfying this protocol can be used as a value function
    in measure-dependent MFG solvers (master equation, common noise).

    The field maps (spatial point, probability measure, time) -> scalar:
        v: Omega x P_2(R^d) x [0,T] -> R
    """

    def evaluate(self, t: float, x: NDArray, mu: MeasureRepresentation) -> NDArray:
        """Evaluate v(t, x, mu) at spatial points x.

        Args:
            x: Spatial points, shape (N,) for 1D or (N, d) for dD.
            mu: Probability measure (any MeasureRepresentation).
            t: Time.

        Returns:
            Field values, shape (N,).
        """
        ...

    def spatial_gradient(self, t: float, x: NDArray, mu: MeasureRepresentation) -> NDArray:
        """Compute nabla_x v(t, x, mu).

        Args:
            x: Spatial points, shape (N,) for 1D or (N, d) for dD.
            mu: Probability measure.
            t: Time.

        Returns:
            Gradient, shape (N,) for 1D or (N, d) for dD.
        """
        ...


class GridMeasureField:
    """MeasureField stored on a spatial grid with measure snapshots.

    Stores v(x_i, mu_k, t_n) as a collection of grid-based value functions
    indexed by (measure_index, time_index, spatial_index). This is the natural
    representation when the value function is computed at a discrete set of
    measure snapshots (e.g., Picard iterates, parameter sweeps, MC samples).

    The field supports:
    - Evaluation at stored measure snapshots (exact lookup)
    - Evaluation at new measures (interpolation in Wasserstein space)
    - Lions derivative via finite differences between measure snapshots
    - Restriction to a fixed density (classical value function)

    Parameters
    ----------
    grid_points : NDArray
        Spatial grid, shape (Nx,) for 1D or (Nx, d) for dD.
    times : NDArray
        Time grid, shape (Nt+1,).

    Example
    -------
    >>> field = GridMeasureField(x_grid, t_grid)
    >>> field.add_snapshot(mu_0, U_0)  # U_0 shape (Nt+1, Nx)
    >>> field.add_snapshot(mu_1, U_1)
    >>> v = field.evaluate(t=0.5, x=x_query, mu=mu_query)
    """

    def __init__(self, grid_points: NDArray, times: NDArray):
        self._grid = np.asarray(grid_points, dtype=float)
        if self._grid.ndim == 1:
            self._grid = self._grid.reshape(-1, 1)
        self._times = np.asarray(times, dtype=float)
        self._nx = self._grid.shape[0]
        self._dim = self._grid.shape[1]
        self._nt = len(self._times)

        # Storage: list of (measure, value_function) pairs
        self._snapshots: list[tuple[MeasureRepresentation, NDArray]] = []

    @property
    def n_snapshots(self) -> int:
        """Number of stored measure snapshots."""
        return len(self._snapshots)

    @property
    def grid_points(self) -> NDArray:
        """Spatial grid, shape (Nx, d)."""
        return self._grid

    @property
    def times(self) -> NDArray:
        """Time grid, shape (Nt+1,)."""
        return self._times

    def add_snapshot(self, mu: MeasureRepresentation, values: NDArray) -> None:
        """Add a value function snapshot at a given measure.

        Args:
            mu: The probability measure at which v was computed.
            values: Value function v(x_i, mu, t_n), shape (Nt+1, Nx).

        Raises:
            ValueError: If values shape doesn't match grid.
        """
        values = np.asarray(values, dtype=float)
        if values.shape != (self._nt, self._nx):
            raise ValueError(f"Values shape {values.shape} doesn't match (Nt+1={self._nt}, Nx={self._nx})")
        self._snapshots.append((mu, values.copy()))

    def get_snapshot(self, index: int) -> tuple[MeasureRepresentation, NDArray]:
        """Get the (measure, values) pair at given index."""
        return self._snapshots[index]

    def evaluate(self, t: float, x: NDArray, mu: MeasureRepresentation) -> NDArray:
        """Evaluate v(t, x, mu) by nearest-snapshot lookup.

        Finds the stored measure closest to mu in Wasserstein distance
        and returns the corresponding value function at time t.

        Args:
            x: Query points, shape (N,) for 1D or (N, d) for dD.
                If x matches the stored grid, exact lookup is used.
            mu: Query measure.
            t: Query time.

        Returns:
            Field values at query points, shape (N,).
        """
        if len(self._snapshots) == 0:
            raise RuntimeError("No snapshots stored. Call add_snapshot() first.")

        # Find time index
        t_idx = int(np.argmin(np.abs(self._times - t)))

        if len(self._snapshots) == 1:
            return self._snapshots[0][1][t_idx]

        # Find nearest snapshot by Wasserstein distance
        best_idx = self._find_nearest_snapshot(mu)
        return self._snapshots[best_idx][1][t_idx]

    def spatial_gradient(self, t: float, x: NDArray, mu: MeasureRepresentation) -> NDArray:
        """Compute nabla_x v at nearest snapshot via finite differences.

        Args:
            x: Query points (must match stored grid).
            mu: Query measure.
            t: Query time.

        Returns:
            Gradient, shape (Nx,) for 1D or (Nx, d) for dD.
        """
        t_idx = int(np.argmin(np.abs(self._times - t)))
        best_idx = self._find_nearest_snapshot(mu)
        u = self._snapshots[best_idx][1][t_idx]

        if self._dim == 1:
            dx = self._grid[1, 0] - self._grid[0, 0] if self._nx > 1 else 1.0
            return np.gradient(u, dx, edge_order=2)
        # nD: gradient per axis
        grads = []
        for axis in range(self._dim):
            spacing = self._grid[1, axis] - self._grid[0, axis] if self._nx > 1 else 1.0
            grads.append(np.gradient(u.reshape(-1), spacing, edge_order=2))
        return np.column_stack(grads)

    def restrict_to_density(self, mu: MeasureRepresentation) -> NDArray:
        """Get classical value function u(t, x) = v(t, x, mu) for fixed mu.

        Args:
            mu: The measure to fix.

        Returns:
            Value function array, shape (Nt+1, Nx).
        """
        if len(self._snapshots) == 0:
            raise RuntimeError("No snapshots stored.")
        best_idx = self._find_nearest_snapshot(mu)
        return self._snapshots[best_idx][1].copy()

    def lions_derivative_fd(
        self, mu: MeasureRepresentation, t: float, snapshot_idx_1: int, snapshot_idx_2: int
    ) -> NDArray:
        """Estimate Lions derivative via finite differences between snapshots.

        Approximates delta v / delta mu by comparing v at two nearby measures:

            delta v / delta mu ≈ (v(·, mu_2, t) - v(·, mu_1, t)) / W_2(mu_1, mu_2)

        This is a directional derivative in measure space, not the full
        functional derivative. For the full Lions derivative, use
        FunctionalDerivative from functional_calculus.py.

        Args:
            mu: Not used directly (for API consistency).
            t: Time at which to evaluate.
            snapshot_idx_1: Index of first snapshot.
            snapshot_idx_2: Index of second snapshot.

        Returns:
            Approximate derivative, shape (Nx,).
        """
        mu1, U1 = self._snapshots[snapshot_idx_1]
        mu2, U2 = self._snapshots[snapshot_idx_2]
        t_idx = int(np.argmin(np.abs(self._times - t)))

        # Wasserstein distance between snapshots
        from mfgarchon.core.measure import ParticleMeasure

        if isinstance(mu1, ParticleMeasure) and isinstance(mu2, ParticleMeasure):
            w_dist = mu1.wasserstein_distance(mu2, p=2)
        else:
            # Fallback: L2 distance between densities on grid
            d1 = mu1.to_density(self._grid[:, 0] if self._dim == 1 else self._grid)
            d2 = mu2.to_density(self._grid[:, 0] if self._dim == 1 else self._grid)
            w_dist = float(np.sqrt(np.mean((d1 - d2) ** 2)))

        if w_dist < 1e-15:
            return np.zeros(self._nx)

        return (U2[t_idx] - U1[t_idx]) / w_dist

    def wasserstein_continuity_estimate(self, t: float) -> float | None:
        """Estimate Lipschitz constant of v(·, ·, t) in Wasserstein distance.

        Computes max_{i,j} ||v(·, mu_i, t) - v(·, mu_j, t)|| / W_2(mu_i, mu_j)
        over all pairs of stored snapshots.

        Returns None if fewer than 2 snapshots are stored.

        Args:
            t: Time at which to evaluate.

        Returns:
            Estimated Lipschitz constant, or None.
        """
        n = len(self._snapshots)
        if n < 2:
            return None

        from mfgarchon.core.measure import ParticleMeasure

        t_idx = int(np.argmin(np.abs(self._times - t)))
        max_lip = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                mu_i, U_i = self._snapshots[i]
                mu_j, U_j = self._snapshots[j]

                # Value difference
                du = float(np.max(np.abs(U_i[t_idx] - U_j[t_idx])))

                # Measure distance
                if isinstance(mu_i, ParticleMeasure) and isinstance(mu_j, ParticleMeasure):
                    w_dist = mu_i.wasserstein_distance(mu_j, p=2)
                else:
                    d_i = mu_i.to_density(self._grid[:, 0] if self._dim == 1 else self._grid)
                    d_j = mu_j.to_density(self._grid[:, 0] if self._dim == 1 else self._grid)
                    w_dist = float(np.sqrt(np.mean((d_i - d_j) ** 2)))

                if w_dist > 1e-15:
                    max_lip = max(max_lip, du / w_dist)

        return max_lip

    def _find_nearest_snapshot(self, mu: MeasureRepresentation) -> int:
        """Find index of stored snapshot nearest to mu."""
        if len(self._snapshots) == 1:
            return 0

        from mfgarchon.core.measure import ParticleMeasure

        best_idx = 0
        best_dist = float("inf")

        for i, (mu_i, _) in enumerate(self._snapshots):
            if isinstance(mu, ParticleMeasure) and isinstance(mu_i, ParticleMeasure):
                dist = mu.wasserstein_distance(mu_i, p=2)
            else:
                # Fallback: L2 distance on densities
                grid_1d = self._grid[:, 0] if self._dim == 1 else self._grid
                d_query = mu.to_density(grid_1d)
                d_stored = mu_i.to_density(grid_1d)
                dist = float(np.sqrt(np.mean((d_query - d_stored) ** 2)))

            if dist < best_dist:
                best_dist = dist
                best_idx = i

        return best_idx

    def __repr__(self) -> str:
        return f"GridMeasureField(Nx={self._nx}, d={self._dim}, Nt={self._nt}, snapshots={self.n_snapshots})"


class FunctionalMeasureField:
    """MeasureField defined by a callable v(t, x, mu).

    Wraps a user-provided function into the MeasureFieldProtocol interface.
    Useful for analytical solutions or parameterized models.

    Parameters
    ----------
    field_fn : Callable[[float, NDArray, MeasureRepresentation], NDArray]
        The field function v(t, x, mu) -> values at x (#2375 ruling 8: time first).
    gradient_fn : Callable or None
        Optional nabla_x v(t, x, mu). If None, computed via finite differences.
    grid_spacing : float
        Grid spacing for FD gradient (only used if gradient_fn is None).

    Example
    -------
    >>> # Analytical: v(t, x, mu) = -log(mu.to_density(x))
    >>> def v_fn(t, x, mu):
    ...     m = mu.to_density(x)
    ...     return -np.log(np.maximum(m, 1e-15))
    >>> field = FunctionalMeasureField(v_fn)
    """

    def __init__(
        self,
        field_fn: Callable,
        gradient_fn: Callable | None = None,
        grid_spacing: float = 0.01,
    ):
        self._fn = field_fn
        self._grad_fn = gradient_fn
        self._dx = grid_spacing
        # Bound at acceptance, so a signature that cannot be matched is refused here (#2375 ruling 8)
        self._field()
        if gradient_fn is not None:
            self._gradient()

    def _field(self) -> BoundCallable:
        return bound_attribute(self, "_fn", MEASURE_FIELD_SLOTS, role="field_fn")

    def _gradient(self) -> BoundCallable:
        return bound_attribute(self, "_grad_fn", MEASURE_FIELD_SLOTS, role="gradient_fn")

    def evaluate(self, t: float, x: NDArray, mu: MeasureRepresentation) -> NDArray:
        """Evaluate v(t, x, mu)."""
        return np.asarray(self._field()(x=x, mu=mu, t=t))

    def spatial_gradient(self, t: float, x: NDArray, mu: MeasureRepresentation) -> NDArray:
        """Compute nabla_x v via provided function or finite differences."""
        if self._grad_fn is not None:
            return np.asarray(self._gradient()(x=x, mu=mu, t=t))

        # Central finite differences
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            v_plus = self._field()(x=x + self._dx, mu=mu, t=t)
            v_minus = self._field()(x=x - self._dx, mu=mu, t=t)
            return (np.asarray(v_plus) - np.asarray(v_minus)) / (2 * self._dx)

        # nD: per-axis
        grads = []
        for axis in range(x.shape[1]):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[:, axis] += self._dx
            x_minus[:, axis] -= self._dx
            v_plus = self._field()(x=x_plus, mu=mu, t=t)
            v_minus = self._field()(x=x_minus, mu=mu, t=t)
            grads.append((np.asarray(v_plus) - np.asarray(v_minus)) / (2 * self._dx))
        return np.column_stack(grads)

    def __repr__(self) -> str:
        return f"FunctionalMeasureField(fn={self._fn.__name__ if hasattr(self._fn, '__name__') else '...'})"

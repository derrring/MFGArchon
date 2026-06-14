#!/usr/bin/env python3
"""
General-purpose convergence metrics and utilities.

This module provides convergence utilities that work for any PDE solver,
not just MFG problems. It includes:

- DistributionComparator: Wasserstein, KL divergence, moments
- RollingConvergenceMonitor: Window-based statistical convergence (renamed from StochasticConvergenceMonitor)
- calculate_error: Unified error computation (L1, L2, Linf)
- ConvergenceConfig: Configuration dataclass for future solver integration

These utilities can be used for standalone HJB, FP, heat equation,
or any iterative PDE solver.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

# =============================================================================
# DISTRIBUTION COMPARISON UTILITIES
# =============================================================================


class DistributionComparator:
    """
    Utilities for comparing probability distributions with robust metrics.

    All methods are static and can be called without instantiation.
    """

    @staticmethod
    def wasserstein_1d(p: np.ndarray, q: np.ndarray, x: np.ndarray) -> float:
        """
        Compute Wasserstein-1 distance (Earth Mover's Distance) for 1D distributions.

        Args:
            p, q: Probability distributions (must sum to 1)
            x: Support points (spatial coordinates)

        Returns:
            Wasserstein-1 distance
        """
        # Ensure proper normalization
        p = p / np.sum(p) if np.sum(p) > 0 else p
        q = q / np.sum(q) if np.sum(q) > 0 else q

        # Compute cumulative distributions
        P_cdf = np.cumsum(p)
        Q_cdf = np.cumsum(q)

        # Wasserstein distance is the L1 distance between CDFs
        # weighted by the grid spacing
        dx = np.diff(x) if len(x) > 1 else np.array([1.0])
        dx = np.append(dx, dx[-1])  # Handle last point

        return float(np.sum(np.abs(P_cdf - Q_cdf) * dx))

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-12) -> float:
        """
        Compute Kullback-Leibler divergence with numerical stability.

        Args:
            p, q: Probability distributions
            epsilon: Small value to avoid log(0) errors

        Returns:
            KL divergence D_KL(p || q)
        """
        # Add epsilon for numerical stability
        p_safe = p + epsilon
        q_safe = q + epsilon

        # Normalize to ensure they're proper distributions
        p_safe = p_safe / np.sum(p_safe)
        q_safe = q_safe / np.sum(q_safe)

        # Compute KL divergence
        return float(np.sum(p_safe * np.log(p_safe / q_safe)))

    @staticmethod
    def statistical_moments(distribution: np.ndarray, x: np.ndarray) -> dict[str, float]:
        """
        Compute statistical moments of a distribution.

        Args:
            distribution: Probability distribution
            x: Support points

        Returns:
            Dictionary with mean, variance, skewness, kurtosis
        """
        # Normalize distribution
        dist = distribution / np.sum(distribution) if np.sum(distribution) > 0 else distribution

        # Compute moments
        mean = float(np.sum(x * dist))
        variance = float(np.sum((x - mean) ** 2 * dist))

        # Higher moments (if variance > 0)
        if variance > 1e-12:
            skewness = float(np.sum(((x - mean) / np.sqrt(variance)) ** 3 * dist))
            kurtosis = float(np.sum(((x - mean) / np.sqrt(variance)) ** 4 * dist) - 3)
        else:
            skewness = 0.0
            kurtosis = 0.0

        return {
            "mean": mean,
            "variance": variance,
            "std": float(np.sqrt(variance)),
            "skewness": skewness,
            "kurtosis": kurtosis,
        }


# =============================================================================
# ROLLING CONVERGENCE MONITOR (renamed from StochasticConvergenceMonitor)
# =============================================================================


class RollingConvergenceMonitor:
    """
    Convergence monitoring using rolling window statistics.

    Suitable for stochastic methods (particle-based solvers) where
    error spikes from noise are normal. Uses statistical stopping criteria:
    - Median error over rolling window (robust to outliers)
    - Quantile-based thresholds
    - Running average with confidence bounds

    This monitor tracks scalar errors over time and assesses convergence
    using robust statistics rather than instantaneous values.
    """

    def __init__(
        self,
        window_size: int = 10,
        median_tolerance: float = 1e-4,
        quantile: float = 0.9,
        quantile_tolerance: float | None = None,
        min_iterations: int = 5,
    ):
        """
        Initialize rolling convergence monitor.

        Args:
            window_size: Number of recent iterations to analyze
            median_tolerance: Tolerance for median error over window
            quantile: Quantile to check (e.g., 0.9 for 90th percentile)
            quantile_tolerance: Tolerance for quantile (defaults to 2x median_tolerance)
            min_iterations: Minimum iterations before checking convergence
        """
        self.window_size = window_size
        self.median_tolerance = median_tolerance
        self.quantile = quantile
        self.quantile_tolerance = quantile_tolerance or (2.0 * median_tolerance)
        self.min_iterations = min_iterations

        # Error history
        self.errors_u: deque[float] = deque(maxlen=window_size)
        self.errors_m: deque[float] = deque(maxlen=window_size)
        self.iteration_count = 0

    def add_iteration(self, error_u: float, error_m: float):
        """Add iteration errors to history."""
        self.errors_u.append(error_u)
        self.errors_m.append(error_m)
        self.iteration_count += 1

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistical summary of recent errors.

        Returns:
            Dictionary with median, mean, std, quantiles, min, max
        """
        if len(self.errors_u) == 0:
            return {"status": "no_data"}

        errors_u_arr = np.array(self.errors_u)
        errors_m_arr = np.array(self.errors_m)

        return {
            "iterations": self.iteration_count,
            "window_size": len(self.errors_u),
            "u_stats": {
                "median": float(np.median(errors_u_arr)),
                "mean": float(np.mean(errors_u_arr)),
                "std": float(np.std(errors_u_arr)),
                "quantile": float(np.quantile(errors_u_arr, self.quantile)),
                "min": float(np.min(errors_u_arr)),
                "max": float(np.max(errors_u_arr)),
            },
            "m_stats": {
                "median": float(np.median(errors_m_arr)),
                "mean": float(np.mean(errors_m_arr)),
                "std": float(np.std(errors_m_arr)),
                "quantile": float(np.quantile(errors_m_arr, self.quantile)),
                "min": float(np.min(errors_m_arr)),
                "max": float(np.max(errors_m_arr)),
            },
        }

    def check_convergence(self) -> tuple[bool, dict[str, Any]]:
        """
        Check convergence using robust statistical criteria.

        Returns:
            (converged, diagnostics)
        """
        # Need sufficient history
        if self.iteration_count < self.min_iterations or len(self.errors_u) < self.window_size:
            return False, {
                "status": "insufficient_history",
                "iterations": self.iteration_count,
                "window_filled": len(self.errors_u),
                "required": self.window_size,
            }

        stats = self.get_statistics()

        # Check median convergence (robust to spikes)
        u_median = stats["u_stats"]["median"]
        m_median = stats["m_stats"]["median"]

        median_converged_u = u_median < self.median_tolerance
        median_converged_m = m_median < self.median_tolerance

        # Check quantile convergence (ensure most errors are small)
        u_quantile = stats["u_stats"]["quantile"]
        m_quantile = stats["m_stats"]["quantile"]

        quantile_converged_u = u_quantile < self.quantile_tolerance
        quantile_converged_m = m_quantile < self.quantile_tolerance

        # Overall convergence: both median and quantile must satisfy criteria
        converged = median_converged_u and median_converged_m and quantile_converged_u and quantile_converged_m

        diagnostics = {
            "status": "converged" if converged else "not_converged",
            "iterations": self.iteration_count,
            "statistics": stats,
            "criteria": {
                "median_u": {"value": u_median, "threshold": self.median_tolerance, "passed": median_converged_u},
                "median_m": {"value": m_median, "threshold": self.median_tolerance, "passed": median_converged_m},
                f"quantile_{int(self.quantile * 100)}%_u": {
                    "value": u_quantile,
                    "threshold": self.quantile_tolerance,
                    "passed": quantile_converged_u,
                },
                f"quantile_{int(self.quantile * 100)}%_m": {
                    "value": m_quantile,
                    "threshold": self.quantile_tolerance,
                    "passed": quantile_converged_m,
                },
            },
        }

        return converged, diagnostics

    def is_stagnating(self, stagnation_threshold: float = 1e-6) -> bool:
        """
        Check if error has stagnated (no improvement over window).

        Args:
            stagnation_threshold: Minimum improvement required

        Returns:
            True if stagnating
        """
        if len(self.errors_u) < self.window_size:
            return False

        errors_arr = np.array(self.errors_u)

        # Check if range of errors in window is very small
        error_range = np.max(errors_arr) - np.min(errors_arr)

        return bool(error_range < stagnation_threshold)


# =============================================================================
# UNIFIED ERROR CALCULATION
# =============================================================================


def calculate_error(
    new: np.ndarray,
    old: np.ndarray,
    dx: float = 1.0,
    dt: float = 1.0,
    norm: Literal["l1", "l2", "linf"] = "l2",
) -> dict[str, float]:
    """
    Calculate error between arrays with multiple norm options.

    Args:
        new, old: Arrays to compare
        dx, dt: Grid spacing (for proper scaling in L1/L2 norms)
        norm: 'l1', 'l2', or 'linf'

    Returns:
        dict with 'absolute' and 'relative' errors

    Notes:
        - L1: ||f||_1 = integral(|f|)dx  (mass/total variation)
        - L2: ||f||_2 = sqrt(integral(f^2)dx)  (energy norm)
        - Linf: ||f||_inf = max|f|  (pointwise, no grid scaling)
    """
    diff = new - old

    if norm == "l1":
        # L1 norm with grid scaling
        scale = np.sqrt(dx * dt)
        abs_error = float(np.sum(np.abs(diff)) * scale)
        norm_new = float(np.sum(np.abs(new)) * scale)
    elif norm == "l2":
        # L2 norm with grid scaling
        scale = np.sqrt(dx * dt)
        abs_error = float(np.linalg.norm(diff) * scale)
        norm_new = float(np.linalg.norm(new) * scale)
    elif norm == "linf":
        # Linf norm - no grid scaling
        abs_error = float(np.max(np.abs(diff)))
        norm_new = float(np.max(np.abs(new)))
    else:
        raise ValueError(f"Unknown norm: {norm}. Use 'l1', 'l2', or 'linf'.")

    rel_error = abs_error / norm_new if norm_new > 1e-12 else abs_error

    return {
        "absolute": abs_error,
        "relative": rel_error,
        "norm": norm,
    }


def calculate_l2_convergence_metrics(
    U_new: np.ndarray,
    U_old: np.ndarray,
    M_new: np.ndarray,
    M_old: np.ndarray,
    Dx: float,
    Dt: float,
) -> dict[str, float]:
    """
    Calculate L2 convergence metrics for MFG fixed-point iterations.

    This helper eliminates code duplication between fixed-point iterator
    implementations by providing a single source for computing both
    absolute and relative L2 errors.

    Args:
        U_new: Current value function array
        U_old: Previous value function array
        M_new: Current density array
        M_old: Previous density array
        Dx: Spatial grid spacing
        Dt: Temporal grid spacing

    Returns:
        Dictionary with keys:
            - 'l2distu_abs': Absolute L2 error for U
            - 'l2distu_rel': Relative L2 error for U
            - 'l2distm_abs': Absolute L2 error for M
            - 'l2distm_rel': Relative L2 error for M

    Example:
        >>> metrics = calculate_l2_convergence_metrics(U_new, U_old, M_new, M_old, Dx, Dt)
        >>> print(f"U relative error: {metrics['l2distu_rel']:.2e}")

    Note:
        The normalization factor sqrt(Dx * Dt) accounts for grid discretization,
        making errors comparable across different grid resolutions.
    """
    u_error = calculate_error(U_new, U_old, dx=Dx, dt=Dt, norm="l2")
    m_error = calculate_error(M_new, M_old, dx=Dx, dt=Dt, norm="l2")

    return {
        "l2distu_abs": u_error["absolute"],
        "l2distu_rel": u_error["relative"],
        "l2distm_abs": m_error["absolute"],
        "l2distm_rel": m_error["relative"],
    }


# =============================================================================
# CONVERGENCE CONFIG (for future solver integration)
# =============================================================================


@dataclass
class ConvergenceConfig:
    """
    Configuration for solver convergence criteria.

    This dataclass can be passed to solvers to configure convergence
    behavior. Designed for future integration (see issue #456).

    Usage:
        config = ConvergenceConfig(tolerance=1e-6, max_iterations=100)
        result = solver.solve(convergence=config)

        # Or use presets
        result = solver.solve(convergence=ConvergenceConfig.strict())
    """

    # Basic settings
    tolerance: float = 1e-6
    max_iterations: int = 100
    min_iterations: int = 0  # Avoid pseudo-convergence in early steps
    norm: Literal["l1", "l2", "linf"] = "l2"
    relative: bool = True
    check_interval: int = 1
    track_history: bool = True

    # For coupled systems (MFG)
    tolerance_U: float | None = None  # HJB tolerance (defaults to tolerance)
    tolerance_M: float | None = None  # FP tolerance (defaults to tolerance)
    require_both: bool = True  # Both U and M must converge

    # For stochastic/particle methods
    window_size: int | None = None
    median_tolerance: float | None = None

    # Divergence detection
    divergence_threshold: float = 1e10  # Safety threshold for error explosion

    # Internal tracking
    _history: list[dict[str, float]] = field(default_factory=list, repr=False)

    def get_tolerance_U(self) -> float:
        """Get tolerance for U (HJB)."""
        return self.tolerance_U if self.tolerance_U is not None else self.tolerance

    def get_tolerance_M(self) -> float:
        """Get tolerance for M (FP)."""
        return self.tolerance_M if self.tolerance_M is not None else self.tolerance

    @classmethod
    def strict(cls) -> ConvergenceConfig:
        """Create strict convergence config."""
        return cls(tolerance=1e-8, max_iterations=500)

    @classmethod
    def relaxed(cls) -> ConvergenceConfig:
        """Create relaxed convergence config."""
        return cls(tolerance=1e-4, max_iterations=50)

    @classmethod
    def stochastic(cls, window_size: int = 10, median_tolerance: float = 1e-4) -> ConvergenceConfig:
        """Create config for stochastic/particle methods."""
        return cls(
            window_size=window_size,
            median_tolerance=median_tolerance,
            tolerance=median_tolerance,
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_rolling_monitor(
    window_size: int = 10,
    median_tolerance: float = 1e-4,
    quantile: float = 0.9,
    **kwargs,
) -> RollingConvergenceMonitor:
    """
    Create rolling convergence monitor with sensible defaults.

    Args:
        window_size: Rolling window size for statistics
        median_tolerance: Threshold for median error
        quantile: Quantile to monitor (default: 0.9 for 90th percentile)
        **kwargs: Additional parameters

    Returns:
        Configured RollingConvergenceMonitor

    Usage:
        monitor = create_rolling_monitor()
        for iteration in range(max_iterations):
            # ... compute errors ...
            monitor.add_iteration(error_u, error_m)
            converged, diagnostics = monitor.check_convergence()
            if converged:
                break
    """
    return RollingConvergenceMonitor(
        window_size=window_size,
        median_tolerance=median_tolerance,
        quantile=quantile,
        **kwargs,
    )


# =============================================================================
# MOMENT-BASED CONVERGENCE MONITOR
# =============================================================================


class MomentConvergenceMonitor:
    """
    Convergence monitoring based on distribution moments.

    For particle-based methods, comparing raw density fields is unreliable
    due to sampling noise O(1/sqrt(N)). Instead, this monitor tracks
    statistical moments (mean, variance) which are more robust to noise.

    The distribution is considered converged when moments stabilize
    within tolerance over a rolling window.
    """

    def __init__(
        self,
        mean_tolerance: float = 1e-3,
        variance_tolerance: float = 1e-2,
        window_size: int = 5,
        min_iterations: int = 5,
    ):
        """
        Initialize moment-based convergence monitor.

        Args:
            mean_tolerance: Relative tolerance for mean convergence
            variance_tolerance: Relative tolerance for variance convergence
            window_size: Number of iterations to check for stability
            min_iterations: Minimum iterations before checking convergence
        """
        self.mean_tolerance = mean_tolerance
        self.variance_tolerance = variance_tolerance
        self.window_size = window_size
        self.min_iterations = min_iterations

        # History of moments
        self.mean_history: list[np.ndarray] = []  # Per-timestep means
        self.variance_history: list[np.ndarray] = []  # Per-timestep variances
        self.iteration_count = 0

    def add_iteration(
        self,
        M: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray | None = None,
    ) -> None:
        """
        Add density field from an iteration and compute moments.

        Args:
            M: Density array, shape (Nt+1, Nx, Ny) or (Nt+1, Nx)
            grid_x: Spatial grid in x direction
            grid_y: Spatial grid in y direction (for 2D)
        """
        # Compute moments for each timestep
        n_timesteps = M.shape[0]
        means = np.zeros(n_timesteps)
        variances = np.zeros(n_timesteps)

        if grid_y is not None:
            # 2D case
            X, Y = np.meshgrid(grid_x, grid_y, indexing="ij")
            dx = grid_x[1] - grid_x[0] if len(grid_x) > 1 else 1.0
            dy = grid_y[1] - grid_y[0] if len(grid_y) > 1 else 1.0

            for t in range(n_timesteps):
                m_t = M[t]
                # Normalize
                total_mass = np.sum(m_t) * dx * dy
                if total_mass > 1e-12:
                    m_norm = m_t / total_mass
                    # Mean position (center of mass)
                    mean_x = np.sum(X * m_norm) * dx * dy
                    mean_y = np.sum(Y * m_norm) * dx * dy
                    means[t] = np.sqrt(mean_x**2 + mean_y**2)  # Distance from origin
                    # Variance (spread)
                    var_x = np.sum((X - mean_x) ** 2 * m_norm) * dx * dy
                    var_y = np.sum((Y - mean_y) ** 2 * m_norm) * dx * dy
                    variances[t] = var_x + var_y
                else:
                    means[t] = 0.0
                    variances[t] = 0.0
        else:
            # 1D case
            dx = grid_x[1] - grid_x[0] if len(grid_x) > 1 else 1.0
            for t in range(n_timesteps):
                m_t = M[t]
                total_mass = np.sum(m_t) * dx
                if total_mass > 1e-12:
                    m_norm = m_t / total_mass
                    means[t] = np.sum(grid_x * m_norm) * dx
                    variances[t] = np.sum((grid_x - means[t]) ** 2 * m_norm) * dx
                else:
                    means[t] = 0.0
                    variances[t] = 0.0

        self.mean_history.append(means)
        self.variance_history.append(variances)
        self.iteration_count += 1

    def get_moment_changes(self) -> dict[str, Any]:
        """
        Get relative changes in moments between last two iterations.

        Returns:
            Dictionary with mean and variance changes
        """
        if len(self.mean_history) < 2:
            return {"status": "insufficient_history"}

        mean_curr = self.mean_history[-1]
        mean_prev = self.mean_history[-2]
        var_curr = self.variance_history[-1]
        var_prev = self.variance_history[-2]

        # Relative changes (averaged over timesteps)
        mean_denom = np.maximum(np.abs(mean_prev), 1e-12)
        var_denom = np.maximum(np.abs(var_prev), 1e-12)

        mean_change = np.mean(np.abs(mean_curr - mean_prev) / mean_denom)
        var_change = np.mean(np.abs(var_curr - var_prev) / var_denom)

        return {
            "mean_change": float(mean_change),
            "variance_change": float(var_change),
            "mean_current": float(np.mean(mean_curr)),
            "variance_current": float(np.mean(var_curr)),
        }

    def check_convergence(self) -> tuple[bool, dict[str, Any]]:
        """
        Check convergence based on moment stability.

        Returns:
            (converged, diagnostics)
        """
        if self.iteration_count < self.min_iterations:
            return False, {
                "status": "insufficient_iterations",
                "iterations": self.iteration_count,
                "required": self.min_iterations,
            }

        if len(self.mean_history) < self.window_size:
            return False, {
                "status": "insufficient_history",
                "history_size": len(self.mean_history),
                "required": self.window_size,
            }

        # Check stability over window
        recent_means = self.mean_history[-self.window_size :]
        recent_vars = self.variance_history[-self.window_size :]

        # Compute max relative change in window
        mean_changes = []
        var_changes = []
        for i in range(1, self.window_size):
            mean_denom = np.maximum(np.abs(recent_means[i - 1]), 1e-12)
            var_denom = np.maximum(np.abs(recent_vars[i - 1]), 1e-12)
            mean_changes.append(np.mean(np.abs(recent_means[i] - recent_means[i - 1]) / mean_denom))
            var_changes.append(np.mean(np.abs(recent_vars[i] - recent_vars[i - 1]) / var_denom))

        max_mean_change = max(mean_changes)
        max_var_change = max(var_changes)

        mean_converged = max_mean_change < self.mean_tolerance
        var_converged = max_var_change < self.variance_tolerance
        converged = mean_converged and var_converged

        diagnostics = {
            "status": "converged" if converged else "not_converged",
            "iterations": self.iteration_count,
            "max_mean_change": float(max_mean_change),
            "max_variance_change": float(max_var_change),
            "mean_tolerance": self.mean_tolerance,
            "variance_tolerance": self.variance_tolerance,
            "mean_converged": mean_converged,
            "variance_converged": var_converged,
            "final_mean": float(np.mean(self.mean_history[-1])),
            "final_variance": float(np.mean(self.variance_history[-1])),
        }

        return converged, diagnostics


def create_moment_monitor(
    mean_tolerance: float = 1e-3,
    variance_tolerance: float = 1e-2,
    window_size: int = 5,
    **kwargs,
) -> MomentConvergenceMonitor:
    """
    Create moment-based convergence monitor.

    Args:
        mean_tolerance: Relative tolerance for mean convergence
        variance_tolerance: Relative tolerance for variance convergence
        window_size: Window size for stability check
        **kwargs: Additional parameters

    Returns:
        Configured MomentConvergenceMonitor

    Usage:
        monitor = create_moment_monitor()
        for iteration in range(max_iterations):
            # ... solve and get M ...
            monitor.add_iteration(M, grid_x, grid_y)
            converged, diagnostics = monitor.check_convergence()
            if converged:
                print(f"Moments converged: {diagnostics}")
                break
    """
    return MomentConvergenceMonitor(
        mean_tolerance=mean_tolerance,
        variance_tolerance=variance_tolerance,
        window_size=window_size,
        **kwargs,
    )

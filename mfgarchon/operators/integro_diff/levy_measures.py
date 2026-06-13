"""
Lévy measure specifications for jump-diffusion processes.

A Lévy measure nu(dz) characterizes the jump intensity and distribution
in the Lévy-Itô decomposition: dX_t = b dt + sigma dW_t + integral z N(dt,dz).

For finite-activity processes (integral nu(dz) < infinity), jumps arrive
as a compound Poisson process with rate lambda = integral nu(dz).

Issue #923: Part of Layer 1 (Generalized PDE & Institutional MFG Plan).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@runtime_checkable
class LevyMeasure(Protocol):
    """Protocol for Lévy jump size distributions.

    Implementations define the density, support, and total mass of the
    jump measure nu. Used by LevyIntegroDiffOperator for quadrature.

    Mathematical background:
        The non-local operator J[v](x) involves integrals against nu:
        J[v](x) = lambda * integral [v(x+z) - v(x) - z*Dv(x)] nu(dz)

        For finite-activity processes: total_mass() < infinity.
        For infinite-activity (e.g., alpha-stable): total_mass() = infinity,
        and the compensator z*Dv(x) is essential for integrability.
    """

    def density(self, z: NDArray) -> NDArray:
        """Evaluate nu(z) at quadrature points.

        Args:
            z: Jump sizes, shape (Q,) for Q quadrature points.

        Returns:
            Density values, shape (Q,).
        """
        ...

    def support_bounds(self) -> tuple[float, float]:
        """Return (z_min, z_max) for compact support or truncation.

        For distributions with infinite support (e.g., Gaussian),
        return truncation bounds (e.g., +/- 4*sigma).
        """
        ...

    def total_mass(self) -> float:
        """Return integral nu(dz). Must be finite for finite-activity."""
        ...


class GaussianJumps:
    """Gaussian jump size distribution: z ~ N(mu, sigma^2).

    Truncated at +/- truncate_at * sigma for finite support.

    This models symmetric or asymmetric jumps with moderate tails.
    Suitable for trade tariff shocks (Institutional Proposal Project A)
    where jump sizes are approximately normally distributed.

    Parameters
    ----------
    mu : float
        Mean jump size (default 0.0 for symmetric jumps).
    sigma : float
        Standard deviation of jump sizes.
    truncate_at : float
        Truncation in units of sigma (default 4.0 = 99.99% of mass).
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0, truncate_at: float = 4.0):
        self.mu = mu
        self.sigma = sigma
        self.truncate_at = truncate_at

    def density(self, z: NDArray) -> NDArray:
        """Gaussian density: (1/sqrt(2*pi*sigma^2)) * exp(-(z-mu)^2 / (2*sigma^2))."""
        return np.exp(-0.5 * ((z - self.mu) / self.sigma) ** 2) / (self.sigma * np.sqrt(2 * np.pi))

    def support_bounds(self) -> tuple[float, float]:
        """Truncated support: [mu - truncate_at*sigma, mu + truncate_at*sigma]."""
        half = self.truncate_at * self.sigma
        return (self.mu - half, self.mu + half)

    def total_mass(self) -> float:
        """Approximate total mass (truncated Gaussian integrates to ~1)."""
        from scipy.stats import norm

        z_min, z_max = self.support_bounds()
        return float(norm.cdf(z_max, self.mu, self.sigma) - norm.cdf(z_min, self.mu, self.sigma))


class CompoundPoissonJumps:
    """Compound Poisson jump distribution with arbitrary jump density.

    Models finite-activity jumps: events arrive at rate `intensity`,
    each with size drawn from `jump_density`.

    The Lévy measure is: nu(dz) = intensity * f(z) dz
    where f is the jump size density.

    Parameters
    ----------
    intensity : float
        Jump arrival rate (Poisson parameter lambda).
    jump_density : LevyMeasure
        Distribution of jump sizes (e.g., GaussianJumps).
    """

    def __init__(self, intensity: float, jump_density: LevyMeasure):
        self.intensity = intensity
        self.jump_density = jump_density

    def density(self, z: NDArray) -> NDArray:
        """nu(z) = intensity * f(z)."""
        return self.intensity * self.jump_density.density(z)

    def support_bounds(self) -> tuple[float, float]:
        """Inherit support from jump density."""
        return self.jump_density.support_bounds()

    def total_mass(self) -> float:
        """Total mass = intensity * integral f(z) dz ≈ intensity."""
        return self.intensity * self.jump_density.total_mass()

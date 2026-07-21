"""
Functional calculus utilities for Master Equation formulations.

This module provides tools for computing functional derivatives in the Master Equation
formulation of Mean Field Games, where the value function lives on measure space:

    U: [0,T] × ℝ^d × P(ℝ^d) → ℝ

Mathematical Background:
    The Master Equation is a PDE on the space of probability measures:

    ∂U/∂t + H(x, ∇_x U, δU/δm, m) + (σ²/2) Δ_x U + ∫ L[δU/δm](x,y) m(dy) = 0

    where:
    - U(t,x,m): Value function at time t, position x, given measure m
    - δU/δm: First-order functional derivative (linear derivative)
    - L: Lift operator acting on functional derivative
    - ∇_x U: Spatial gradient
    - Δ_x U: Spatial Laplacian

Functional Derivatives:
    First-Order (Linear Derivative):
        δU/δm[m](x,y) = lim_{ε→0} (U[m + εδ_y] - U[m]) / ε

    Second-Order:
        δ²U/δm²[m](x,y,z) = lim_{ε→0} (δU/δm[m + εδ_z](x,y) - δU/δm[m](x,y)) / ε

Approximation Methods:
    1. Finite Differences: δU/δm ≈ (U[m + εδ_y] - U[m]) / ε
    2. Particle Approximation: m ≈ (1/N)Σᵢ δ_{yᵢ}, represent m by particles

    An automatic-differentiation route (JAX/PyTorch) is not implemented here; see
    Issue #1642 (capability E2) if one becomes necessary.

References:
    - Cardaliaguet et al. (2019): The Master Equation and Convergence Problem
    - Lions Lectures (2007-2011): Mean Field Games and Applications
    - Gangbo & Święch (2015): Existence of Solutions to Master Equations
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Type definitions
MeasureDensity = NDArray  # Probability density on spatial domain
FunctionalOnMeasures = Callable[[MeasureDensity], float | NDArray]  # U[m]
LinearDerivative = Callable[[NDArray, NDArray], float | NDArray]  # δU/δm(x,y)


@dataclass
class FunctionalDerivativeConfig:
    """Configuration for functional derivative computation."""

    # Finite difference parameters
    epsilon: float = 1e-4  # Perturbation size for finite differences
    method: str = "forward"  # "forward" | "central" | "automatic"

    # Particle approximation
    num_particles: int = 50  # Number of particles to represent measure
    particle_method: str = "uniform"  # "uniform" | "adaptive" | "quasi_random"

    # Numerical stability
    regularization: float = 1e-10  # Small regularization to avoid division by zero
    clip_gradients: bool = True  # Clip large gradients
    max_gradient: float = 1e6  # Maximum gradient value


class FunctionalDerivative(ABC):
    """
    Abstract base class for functional derivative operators.

    Provides interface for computing δU/δm for functionals U on measure space.
    """

    @abstractmethod
    def compute(
        self,
        functional: FunctionalOnMeasures,
        measure: MeasureDensity,
        x_points: NDArray,
        y_points: NDArray,
    ) -> NDArray:
        """
        Compute functional derivative δU/δm[m](x,y).

        Args:
            functional: Functional U: P(ℝ^d) → ℝ or ℝ^n
            measure: Current measure m (density representation)
            x_points: Points x where to evaluate δU/δm
            y_points: Points y for perturbation direction

        Returns:
            Array of functional derivative values δU/δm(x,y)
        """
        ...

    @abstractmethod
    def compute_second_order(
        self,
        functional: FunctionalOnMeasures,
        measure: MeasureDensity,
        x_points: NDArray,
        y_points: NDArray,
        z_points: NDArray,
    ) -> NDArray:
        """
        Compute second-order functional derivative δ²U/δm²[m](x,y,z).

        Args:
            functional: Functional U on measure space
            measure: Current measure m
            x_points: First evaluation points
            y_points: First perturbation direction
            z_points: Second perturbation direction

        Returns:
            Second-order functional derivative values
        """
        ...


class FiniteDifferenceFunctionalDerivative(FunctionalDerivative):
    """
    Finite difference approximation of functional derivatives.

    Uses finite differences to approximate:
        δU/δm[m](x,y) ≈ (U[m + εδ_y] - U[m]) / ε

    where δ_y is the Dirac delta at y.

    Example:
        >>> def my_functional(m):
        ...     # Simple functional: U[m] = ∫ m(y) V(y) dy
        ...     return np.sum(m * V)
        >>>
        >>> derivative = FiniteDifferenceFunctionalDerivative(epsilon=1e-4)
        >>> dU_dm = derivative.compute(my_functional, m, x_points, y_points)
    """

    def __init__(self, epsilon: float = 1e-4, method: str = "forward"):
        """
        Initialize finite difference functional derivative.

        Args:
            epsilon: Perturbation size
            method: Finite difference method ("forward" | "central")
        """
        self.epsilon = epsilon
        self.method = method

    def compute(
        self,
        functional: FunctionalOnMeasures,
        measure: MeasureDensity,
        x_points: NDArray,
        y_points: NDArray,
    ) -> NDArray:
        """
        Compute δU/δm via finite differences.

        Approximates:
            δU/δm[m](x,y) ≈ (U[m + εδ_y] - U[m]) / ε

        Args:
            functional: Functional U[m]
            measure: Current measure m (density on grid)
            x_points: Evaluation points (not used for simple functionals)
            y_points: Perturbation points (indices into measure grid)

        Returns:
            Functional derivative values
        """
        # Baseline functional value
        U_baseline = functional(measure)

        # Perturbed values
        derivatives = []

        for y_idx in y_points:
            # Create perturbation: m + ε δ_y
            # In discrete setting: add ε at position y_idx
            perturbed_measure = measure.copy()
            perturbed_measure[y_idx] += self.epsilon

            # NOTE: Do NOT normalize! Functional derivative is defined as
            # δU/δm = lim_ε (U[m + εδ_y] - U[m])/ε
            # The perturbation m + εδ_y is not a probability measure,
            # but that's correct for the derivative definition.

            # Evaluate functional on perturbed measure
            U_perturbed = functional(perturbed_measure)

            # Finite difference
            if self.method == "forward":
                derivative = (U_perturbed - U_baseline) / self.epsilon
            elif self.method == "central":
                # Central difference: (U[m + ε/2 δ_y] - U[m - ε/2 δ_y]) / ε
                # Backward perturbation. Issue #1514: do NOT clip to non-negativity here. The forward
                # perturbation is uncapped and the quotient divides by the full ε, so clipping only the
                # backward step (which fires when measure[y_idx] < ε/2, i.e. in density tails) makes the
                # stencil asymmetric -> a silently biased derivative exactly at low-density points. The
                # module's own contract (see the forward branch above) is that a perturbed measure need
                # NOT be a probability measure, so the clip is both wrong and unnecessary.
                backward_measure = measure.copy()
                backward_measure[y_idx] -= self.epsilon / 2

                U_backward = functional(backward_measure)

                # Forward perturbation
                forward_measure = measure.copy()
                forward_measure[y_idx] += self.epsilon / 2

                U_forward = functional(forward_measure)

                derivative = (U_forward - U_backward) / self.epsilon
            else:
                raise ValueError(f"Unknown finite difference method: {self.method}")

            derivatives.append(derivative)

        return np.array(derivatives)

    def compute_second_order(
        self,
        functional: FunctionalOnMeasures,
        measure: MeasureDensity,
        x_points: NDArray,
        y_points: NDArray,
        z_points: NDArray,
    ) -> NDArray:
        """
        Compute second-order functional derivative via nested finite differences.

        δ²U/δm²[m](x,y,z) ≈ (δU/δm[m + εδ_z](x,y) - δU/δm[m](x,y)) / ε

        Args:
            functional: Functional U[m]
            measure: Current measure
            x_points: First evaluation points
            y_points: First perturbation indices
            z_points: Second perturbation indices

        Returns:
            Second-order functional derivative
        """
        # First derivative at baseline measure
        dU_dm_baseline = self.compute(functional, measure, x_points, y_points)

        # Perturb in z direction
        second_derivatives = []

        for z_idx in z_points:
            perturbed_measure = measure.copy()
            perturbed_measure[z_idx] += self.epsilon

            # First derivative at perturbed measure
            dU_dm_perturbed = self.compute(functional, perturbed_measure, x_points, y_points)

            # Second derivative
            d2U_dm2 = (dU_dm_perturbed - dU_dm_baseline) / self.epsilon
            second_derivatives.append(d2U_dm2)

        return np.array(second_derivatives)


class ParticleApproximationFunctionalDerivative(FunctionalDerivative):
    """
    Particle approximation for functional derivatives.

    Represents measures as empirical measures:
        m_N = (1/N) Σᵢ₌₁ᴺ δ_{yᵢ}

    Then functional derivatives become:
        δU/δm[m_N](x,y) ≈ (1/N) Σᵢ ∂U/∂yᵢ evaluated at yᵢ

    This reduces infinite-dimensional functional calculus to finite-dimensional
    calculus of variations.

    Applications:
        - Master Equation approximation with N particles
        - High-dimensional problems where particle methods excel
        - Connection to N-player game convergence

    Example:
        >>> # Represent measure by N particles
        >>> particles = np.random.uniform(0, 1, size=(50, 1))  # 50 particles in 1D
        >>> weights = np.ones(50) / 50  # Equal weights
        >>>
        >>> derivative = ParticleApproximationFunctionalDerivative(particles, weights)
        >>> dU_dm = derivative.compute(functional, particles, x_points, y_points)
    """

    def __init__(self, particles: NDArray, weights: NDArray | None = None):
        """
        Initialize particle-based functional derivative.

        Args:
            particles: Particle positions, shape (N, d) for N particles in d dimensions
            weights: Particle weights (default: uniform 1/N)
        """
        self.particles = particles
        self.N = len(particles)

        if weights is None:
            self.weights = np.ones(self.N) / self.N
        else:
            self.weights = weights / weights.sum()  # Normalize

    def compute(
        self,
        functional: FunctionalOnMeasures,
        measure: MeasureDensity,
        x_points: NDArray,
        y_points: NDArray,
    ) -> NDArray:
        """
        Compute functional derivative via particle approximation.

        For empirical measure m_N = (1/N) Σᵢ δ_{yᵢ}, the functional derivative
        is approximated by partial derivatives with respect to particle positions.

        Args:
            functional: Functional U[m] evaluated on particle measures
            measure: Particle positions (N, d)
            x_points: Evaluation points
            y_points: Perturbation points (particle indices)

        Returns:
            Functional derivative values
        """
        # For particle approximation, functional takes particle positions
        # δU/δm(x,y) ≈ ∂U/∂yᵢ where yᵢ is closest particle to y

        derivatives = []
        epsilon = 1e-6

        U_baseline = functional(self.particles)

        for particle_idx in y_points:
            # Perturb particle
            perturbed_particles = self.particles.copy()
            perturbed_particles[particle_idx] += epsilon

            U_perturbed = functional(perturbed_particles)

            # Finite difference
            derivative = (U_perturbed - U_baseline) / epsilon
            derivatives.append(derivative)

        return np.array(derivatives)

    def compute_second_order(
        self,
        functional: FunctionalOnMeasures,
        measure: MeasureDensity,
        x_points: NDArray,
        y_points: NDArray,
        z_points: NDArray,
    ) -> NDArray:
        """
        Compute second-order derivative via particle approximation.

        δ²U/δm² ≈ ∂²U/∂yᵢ∂yⱼ

        Args:
            functional: Functional on particle space
            measure: Particle positions
            x_points: Evaluation points
            y_points: First particle indices
            z_points: Second particle indices

        Returns:
            Second-order functional derivative (mixed partials)
        """
        epsilon = 1e-6

        # Baseline gradient
        dU_baseline = self.compute(functional, measure, x_points, y_points)

        second_derivatives = []

        for z_idx in z_points:
            # Perturb in z direction
            perturbed_particles = self.particles.copy()
            perturbed_particles[z_idx] += epsilon

            # Update particle approximation temporarily
            old_particles = self.particles
            self.particles = perturbed_particles

            # Compute gradient at perturbed state
            dU_perturbed = self.compute(functional, perturbed_particles, x_points, y_points)

            # Restore particles
            self.particles = old_particles

            # Second derivative
            d2U = (dU_perturbed - dU_baseline) / epsilon
            second_derivatives.append(d2U)

        return np.array(second_derivatives)


def create_particle_measure(
    domain_bounds: tuple[float, float],
    num_particles: int,
    method: str = "uniform",
    seed: int | None = None,
) -> tuple[NDArray, NDArray]:
    """
    Create particle approximation of a measure on bounded domain.

    Args:
        domain_bounds: (xmin, xmax) spatial domain bounds
        num_particles: Number of particles N
        method: Particle placement method:
            - "uniform": Uniformly spaced particles
            - "random": Random uniform particles
            - "sobol": Quasi-random Sobol sequence
        seed: Random seed (for random/sobol methods)

    Returns:
        Tuple of (particles, weights) where:
            - particles: Array of shape (N,) with particle positions
            - weights: Array of shape (N,) with particle weights (sum to 1)

    Example:
        >>> particles, weights = create_particle_measure((0, 1), num_particles=100)
        >>> # Empirical measure: m_N = Σᵢ wᵢ δ_{particles[i]}
    """
    xmin, xmax = domain_bounds

    if method == "uniform":
        # Uniformly spaced particles
        particles = np.linspace(xmin, xmax, num_particles)
    elif method == "random":
        # Random uniform particles
        if seed is not None:
            np.random.seed(seed)
        particles = np.random.uniform(xmin, xmax, size=num_particles)
    elif method == "sobol":
        # Quasi-random Sobol sequence using scipy directly
        try:
            from scipy.stats import qmc

            sobol = qmc.Sobol(d=1, scramble=True, seed=seed)
            quasi_samples = sobol.random(num_particles)
            # Scale to domain
            particles = xmin + (xmax - xmin) * quasi_samples[:, 0]
        except ImportError:
            # Fallback to random if scipy not available
            if seed is not None:
                np.random.seed(seed)
            particles = np.random.uniform(xmin, xmax, size=num_particles)
    else:
        raise ValueError(f"Unknown particle method: {method}")

    # Equal weights for empirical measure
    weights = np.ones(num_particles) / num_particles

    return particles, weights


def verify_functional_derivative_accuracy(
    functional: FunctionalOnMeasures,
    analytical_derivative: Callable[[NDArray], NDArray] | None = None,
    domain_bounds: tuple[float, float] = (0.0, 1.0),
    num_particles: int = 50,
) -> dict[str, float]:
    """
    Verify accuracy of functional derivative computation.

    Compares finite difference approximation with analytical derivative (if available)
    or checks convergence properties.

    Args:
        functional: Functional U[m] to test
        analytical_derivative: Known analytical δU/δm (if available)
        domain_bounds: Spatial domain
        num_particles: Number of particles for approximation

    Returns:
        Dictionary with error metrics

    Example:
        >>> # Test with simple functional: U[m] = ∫ m(y)² dy
        >>> def functional(m):
        ...     return np.sum(m**2)
        >>>
        >>> def analytical_deriv(m):
        ...     return 2 * m  # δU/δm = 2m
        >>>
        >>> errors = test_functional_derivative_accuracy(
        ...     functional, analytical_deriv
        ... )
        >>> print(f"Max error: {errors['max_error']:.6e}")
    """
    # Create test measure
    _particles, weights = create_particle_measure(domain_bounds, num_particles, method="uniform")

    # Create measure density (histogram approximation)
    measure = weights  # For 1D uniform particles

    # Compute numerical derivative
    derivative_op = FiniteDifferenceFunctionalDerivative(epsilon=1e-4)
    y_points = np.arange(num_particles)  # All particle indices
    numerical_deriv = derivative_op.compute(functional, measure, None, y_points)

    if analytical_derivative is not None:
        # Compare with analytical
        analytical_deriv = analytical_derivative(measure)

        max_error = np.max(np.abs(numerical_deriv - analytical_deriv))
        mean_error = np.mean(np.abs(numerical_deriv - analytical_deriv))
        relative_error = max_error / (np.max(np.abs(analytical_deriv)) + 1e-10)

        return {
            "max_error": max_error,
            "mean_error": mean_error,
            "relative_error": relative_error,
            "converged": relative_error < 1e-3,
        }
    else:
        # Check self-consistency: derivative should satisfy properties
        return {
            "derivative_norm": np.linalg.norm(numerical_deriv),
            "note": "No analytical derivative provided for comparison",
        }

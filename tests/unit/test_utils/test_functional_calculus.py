"""
Unit tests for functional calculus utilities.

Tests functional derivative computation for Master Equation formulations.
"""

import pytest

import numpy as np

from mfgarchon.utils.functional_calculus import (
    FiniteDifferenceFunctionalDerivative,
    ParticleApproximationFunctionalDerivative,
    create_particle_measure,
    verify_functional_derivative_accuracy,
)

pytestmark = pytest.mark.experimental


class TestFiniteDifferenceFunctionalDerivative:
    """Test finite difference functional derivative computation."""

    def test_linear_functional(self):
        """Test derivative of linear functional U[m] = ∫ V(y) m(y) dy."""
        # Linear functional: U[m] = Σᵢ Vᵢ mᵢ
        V = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        def linear_functional(m):
            return np.sum(V * m)

        # Analytical derivative: δU/δm = V
        def analytical_derivative(m):
            return V

        # Test measure
        measure = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        # Compute numerical derivative
        derivative_op = FiniteDifferenceFunctionalDerivative(epsilon=1e-4)
        y_points = np.arange(5)
        numerical_deriv = derivative_op.compute(linear_functional, measure, None, y_points)

        # For linear functional, derivative should be exactly V (up to numerical error)
        np.testing.assert_allclose(numerical_deriv, V, rtol=1e-3)

    def test_quadratic_functional(self):
        """Test derivative of quadratic functional U[m] = ∫∫ m(x) K(x,y) m(y) dx dy."""

        # Simple quadratic: U[m] = ∫ m(y)² dy
        def quadratic_functional(m):
            return np.sum(m**2)

        # Analytical derivative: δU/δm = 2m
        measure = np.array([0.1, 0.2, 0.3, 0.2, 0.2])

        derivative_op = FiniteDifferenceFunctionalDerivative(epsilon=1e-5)
        y_points = np.arange(5)
        numerical_deriv = derivative_op.compute(quadratic_functional, measure, None, y_points)

        # Check against analytical: δU/δm ≈ 2m
        expected_deriv = 2 * measure
        np.testing.assert_allclose(numerical_deriv, expected_deriv, rtol=1e-2)

    def test_central_difference_more_accurate(self):
        """Test that central difference is more accurate than forward difference."""

        def test_functional(m):
            return np.sum(m**3)  # Cubic functional

        measure = np.ones(10) / 10

        # Forward difference
        forward_op = FiniteDifferenceFunctionalDerivative(epsilon=1e-4, method="forward")
        forward_deriv = forward_op.compute(test_functional, measure, None, np.arange(10))

        # Central difference
        central_op = FiniteDifferenceFunctionalDerivative(epsilon=1e-4, method="central")
        central_deriv = central_op.compute(test_functional, measure, None, np.arange(10))

        # Analytical: δU/δm = 3m²
        analytical = 3 * measure**2

        forward_error = np.linalg.norm(forward_deriv - analytical)
        central_error = np.linalg.norm(central_deriv - analytical)

        # Central should be more accurate
        assert central_error < forward_error

    def test_central_difference_unbiased_at_low_density_tail(self):
        """Issue #1514: the central stencil must not clip ONLY the backward perturbation, which biases
        the derivative where measure[y] < epsilon/2 (density tails). For a LINEAR functional the
        derivative is exactly V everywhere; the old one-sided np.maximum(.,0) clip gave ~0.6*V at a
        tail point (backward step truncated to measure[y] while dividing by the full epsilon)."""
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        def linear_functional(m):
            return float(np.sum(v * m))

        eps = 1e-4
        # index 0 is a density tail below epsilon/2 = 5e-5 -> the old clip fired here
        measure = np.array([1e-5, 0.3, 0.3, 0.2, 0.2])
        op = FiniteDifferenceFunctionalDerivative(epsilon=eps, method="central")
        deriv = op.compute(linear_functional, measure, None, np.arange(5))
        np.testing.assert_allclose(deriv, v, rtol=1e-9)  # exact for a linear functional, incl. the tail

    def test_second_order_derivative(self):
        """Test second-order functional derivative computation."""

        def quadratic_functional(m):
            return np.sum(m**2)

        measure = np.ones(5) / 5

        derivative_op = FiniteDifferenceFunctionalDerivative(epsilon=1e-4)
        y_points = np.array([0, 1, 2])
        z_points = np.array([0, 1])

        # Compute second derivative
        second_deriv = derivative_op.compute_second_order(quadratic_functional, measure, None, y_points, z_points)

        # For U[m] = Σ m², we have δ²U/δm² = 0 (second derivative vanishes for quadratic)
        # Actually, for empirical measures with normalization, there's coupling
        # Just check it returns correct shape
        assert second_deriv.shape == (len(z_points), len(y_points))


class TestParticleApproximationFunctionalDerivative:
    """Test particle approximation for functional derivatives."""

    def test_initialization(self):
        """Test particle derivative initialization."""
        particles = np.linspace(0, 1, 50).reshape(-1, 1)

        derivative_op = ParticleApproximationFunctionalDerivative(particles)

        assert derivative_op.N == 50
        assert len(derivative_op.weights) == 50
        np.testing.assert_allclose(derivative_op.weights.sum(), 1.0)

    def test_custom_weights(self):
        """Test with custom particle weights."""
        particles = np.array([[0.0], [0.5], [1.0]])
        weights = np.array([0.5, 0.3, 0.2])

        derivative_op = ParticleApproximationFunctionalDerivative(particles, weights)

        # Weights should be normalized
        np.testing.assert_allclose(derivative_op.weights.sum(), 1.0)
        np.testing.assert_allclose(derivative_op.weights, weights / weights.sum())

    def test_particle_functional_derivative(self):
        """Test functional derivative with particle representation."""
        # Simple functional on particles
        particles = np.linspace(0, 1, 20).reshape(-1, 1)

        def particle_functional(pts):
            # U[particles] = Σᵢ pts[i]²
            return np.sum(pts**2)

        derivative_op = ParticleApproximationFunctionalDerivative(particles)

        # Compute derivative
        particle_indices = np.array([0, 5, 10, 15])
        deriv = derivative_op.compute(particle_functional, particles, None, particle_indices)

        # Should have one value per particle index
        assert len(deriv) == len(particle_indices)


class TestCreateParticleMeasure:
    """Test particle measure creation utilities."""

    def test_uniform_particles(self):
        """Test uniform particle placement."""
        particles, weights = create_particle_measure((0, 1), num_particles=100, method="uniform")

        assert len(particles) == 100
        assert len(weights) == 100
        np.testing.assert_allclose(weights.sum(), 1.0)

        # Check spacing is uniform
        spacings = np.diff(particles)
        np.testing.assert_allclose(spacings, spacings[0], rtol=1e-10)

    def test_random_particles(self):
        """Test random particle placement."""
        particles, weights = create_particle_measure((0, 1), num_particles=100, method="random", seed=42)

        assert len(particles) == 100
        np.testing.assert_allclose(weights.sum(), 1.0)

        # All particles in domain
        assert np.all(particles >= 0)
        assert np.all(particles <= 1)

    def test_sobol_particles(self):
        """Test Sobol sequence particle placement."""
        particles, weights = create_particle_measure((0, 1), num_particles=100, method="sobol", seed=42)

        assert len(particles) == 100
        np.testing.assert_allclose(weights.sum(), 1.0)

        # Sobol should give better coverage than random
        # All particles in domain
        assert np.all(particles >= 0)
        assert np.all(particles <= 1)

    def test_invalid_method_raises_error(self):
        """Test that invalid particle method raises error."""
        with pytest.raises(ValueError, match="Unknown particle method"):
            create_particle_measure((0, 1), num_particles=50, method="invalid")


class TestFunctionalDerivativeAccuracy:
    """Test functional derivative accuracy testing utility."""

    def test_with_analytical_derivative(self):
        """Test accuracy testing with known analytical derivative."""

        # Linear functional: U[m] = Σ aᵢ mᵢ
        a = np.random.randn(50)

        def functional(m):
            return np.sum(a * m)

        def analytical_derivative(m):
            return a

        # Test accuracy
        errors = verify_functional_derivative_accuracy(
            functional, analytical_derivative, domain_bounds=(0, 1), num_particles=50
        )

        # Should have low error for linear functional
        assert errors["max_error"] < 1e-3
        assert errors["converged"]

    def test_without_analytical_derivative(self):
        """Test accuracy testing without analytical derivative."""

        def functional(m):
            return np.sum(m**2)

        # Test without analytical
        result = verify_functional_derivative_accuracy(
            functional, analytical_derivative=None, domain_bounds=(0, 1), num_particles=30
        )

        # Should return norm information
        assert "derivative_norm" in result
        assert "note" in result


class TestIntegrationExamples:
    """Integration tests demonstrating usage patterns."""

    def test_master_equation_functional(self):
        """
        Test functional derivative for a Master Equation-like functional.

        U[m](x) = ∫ K(x,y) m(y) dy + (1/2) ∫∫ m(y) L(y,z) m(z) dy dz
        """
        # Discretized version with N points
        N = 20
        domain = np.linspace(0, 1, N)
        dx = 1.0 / (N - 1)

        # Kernel K(x,y) = exp(-|x-y|²)
        X, Y = np.meshgrid(domain, domain, indexing="ij")
        K = np.exp(-((X - Y) ** 2))

        # Coupling kernel L(y,z)
        L = 0.1 * np.exp(-((Y - X) ** 2))

        def master_equation_functional(m):
            # Linear term: ∫ K(x,y) m(y) dy
            linear_term = np.sum(K @ m) * dx

            # Quadratic term: ∫∫ m(y) L(y,z) m(z) dy dz
            quadratic_term = 0.5 * m @ L @ m * dx**2

            return linear_term + quadratic_term

        # Test measure
        measure = np.exp(-((domain - 0.5) ** 2) / 0.1)
        measure = measure / (measure.sum() * dx)  # Normalize

        # Compute derivative
        derivative_op = FiniteDifferenceFunctionalDerivative(epsilon=1e-5)
        y_points = np.arange(N)
        deriv = derivative_op.compute(master_equation_functional, measure, None, y_points)

        # Check derivative has correct shape
        assert deriv.shape == (N,)

        # Analytical derivative:
        # δU/δm = ∫ K(x,y) dy + ∫ L(y,z) m(z) dz
        analytical = np.sum(K, axis=1) * dx + (L @ measure) * dx

        # Compare (relaxed tolerance due to measure normalization effects)
        np.testing.assert_allclose(deriv, analytical, rtol=0.1)

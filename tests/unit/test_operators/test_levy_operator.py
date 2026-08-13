"""Tests for Lévy integro-differential operator.

Tests LevyIntegroDiffOperator, GaussianJumps, CompoundPoissonJumps
from mfgarchon.operators.integro_diff.

Covers:
- Basic evaluation (J[v] produces finite, correct-shape output)
- Adjoint consistency on uniform and non-uniform grids (Binding Constraint #2)
- Sparse matrix equivalence with matvec
- Lévy measure implementations (GaussianJumps, CompoundPoissonJumps)
- Mass conservation of adjoint operator
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.operators.integro_diff.levy_integro_diff import LevyIntegroDiffOperator
from mfgarchon.operators.integro_diff.levy_measures import (
    CompoundPoissonJumps,
    GaussianJumps,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def uniform_grid():
    return np.linspace(0, 2 * np.pi, 101)


@pytest.fixture
def nonuniform_grid():
    """Non-uniform grid: denser near x=0, coarser near x=2*pi."""
    x = np.sort(
        np.unique(
            np.concatenate(
                [
                    np.linspace(0, 1, 30),
                    np.linspace(1, 2 * np.pi, 71),
                ]
            )
        )
    )
    return x


@pytest.fixture
def gaussian_jumps():
    return GaussianJumps(mu=0.0, sigma=0.3, truncate_at=3.0)


@pytest.fixture
def compound_poisson(gaussian_jumps):
    return CompoundPoissonJumps(intensity=2.0, jump_density=gaussian_jumps)


# ---------------------------------------------------------------------------
# Lévy measure tests
# ---------------------------------------------------------------------------


class TestGaussianJumps:
    def test_density_shape(self, gaussian_jumps):
        z = np.linspace(-1, 1, 50)
        nu = gaussian_jumps.density(z)
        assert nu.shape == (50,)
        assert np.all(nu >= 0)

        # Closed form for N(0, 0.3^2). Measured max relative deviation 1.0e-15, so rtol=1e-12
        # leaves ~1000x margin without pinning float noise.
        sigma = 0.3
        np.testing.assert_allclose(nu, np.exp(-(z**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi)), rtol=1e-12)

        # This grid extends outside support_bounds() = +/-0.9, and `density` does NOT zero there:
        # truncation lives in support_bounds() alone. Measured nu(1.0) = 0.00514. Asserted so that
        # a future implementation which starts truncating inside `density` is caught rather than
        # silently changing the operator's quadrature.
        assert np.all(nu[np.abs(z) > 0.9] > 0)

    def test_density_peak_at_mean(self, gaussian_jumps):
        z = np.array([0.0, 0.3, -0.3, 1.0])
        nu = gaussian_jumps.density(z)
        assert nu[0] == np.max(nu), "Peak should be at mu=0"

    def test_support_bounds(self, gaussian_jumps):
        z_min, z_max = gaussian_jumps.support_bounds()
        assert z_min == pytest.approx(-0.9, abs=1e-10)
        assert z_max == pytest.approx(0.9, abs=1e-10)

    def test_total_mass_near_one(self, gaussian_jumps):
        mass = gaussian_jumps.total_mass()
        # Truncation at 3 sigma captures ~99.7%
        assert 0.99 < mass < 1.001


class TestCompoundPoissonJumps:
    def test_density_scales_with_intensity(self, gaussian_jumps, compound_poisson):
        z = np.linspace(-0.5, 0.5, 20)
        nu_gauss = gaussian_jumps.density(z)
        nu_cp = compound_poisson.density(z)
        np.testing.assert_allclose(nu_cp, 2.0 * nu_gauss)

    def test_total_mass(self, gaussian_jumps, compound_poisson):
        assert compound_poisson.total_mass() == pytest.approx(2.0 * gaussian_jumps.total_mass(), rel=1e-10)

    def test_support_inherited(self, gaussian_jumps, compound_poisson):
        assert compound_poisson.support_bounds() == gaussian_jumps.support_bounds()


# ---------------------------------------------------------------------------
# Operator tests
# ---------------------------------------------------------------------------


class TestLevyOperatorBasic:
    def test_zero_input_gives_zero(self, uniform_grid, gaussian_jumps):
        J = LevyIntegroDiffOperator(uniform_grid, gaussian_jumps)
        v = np.zeros_like(uniform_grid)
        Jv = J @ v
        np.testing.assert_allclose(Jv, 0.0, atol=1e-14)

    def test_constant_input_gives_zero(self, uniform_grid, gaussian_jumps):
        """J[constant] = 0 because v(x+z) - v(x) = 0 and z*Dv = 0."""
        J = LevyIntegroDiffOperator(uniform_grid, gaussian_jumps)
        v = np.ones_like(uniform_grid) * 3.7
        Jv = J @ v
        np.testing.assert_allclose(Jv, 0.0, atol=1e-10)

    def test_linearity(self, uniform_grid, gaussian_jumps):
        """J[alpha*v + beta*w] = alpha*J[v] + beta*J[w]."""
        J = LevyIntegroDiffOperator(uniform_grid, gaussian_jumps)
        v = np.sin(uniform_grid)
        w = np.cos(uniform_grid)
        alpha, beta = 2.3, -0.7

        lhs = J @ (alpha * v + beta * w)
        rhs = alpha * (J @ v) + beta * (J @ w)
        np.testing.assert_allclose(lhs, rhs, atol=1e-10)

    def test_without_compensator(self, uniform_grid, gaussian_jumps):
        """The compensator subtracts z*Dv*nu(dz); it is a no-op iff the measure has zero first moment."""
        J_comp = LevyIntegroDiffOperator(uniform_grid, gaussian_jumps, compensate=True)
        J_nocomp = LevyIntegroDiffOperator(uniform_grid, gaussian_jumps, compensate=False)
        v = np.sin(uniform_grid)

        # The fixture is symmetric (mu=0), so integral z*nu(dz) = 0 and the two operators must
        # AGREE, not differ. Measured max|difference| = 2.78e-17; 1e-14 is ~360x above that.
        np.testing.assert_allclose(J_comp @ v, J_nocomp @ v, atol=1e-14)

        # An asymmetric measure is required to separate the two branches at all.
        nu_asym = GaussianJumps(mu=0.3, sigma=0.1, truncate_at=3.0)
        J_comp_a = LevyIntegroDiffOperator(uniform_grid, nu_asym, compensate=True)
        J_nocomp_a = LevyIntegroDiffOperator(uniform_grid, nu_asym, compensate=False)

        # Closed form on v(x) = x. Compensated: the integrand v(x+z) - v(x) - z*Dv vanishes
        # identically for linear v, so J_comp[x] = 0 exactly. Measured 8.83e-15 over the whole
        # grid; 1e-12 leaves ~113x margin.
        assert np.max(np.abs(J_comp_a @ uniform_grid)) < 1e-12

        # Uncompensated: J[x] = Dv * integral z*nu(dz) = mu * total_mass, on the interior points
        # whose full jump support [0, 0.6] stays inside the grid. Measured relative deviation
        # 2.4e-15 against 0.2991900611810219; rtol=1e-12 leaves ~400x margin.
        np.testing.assert_allclose((J_nocomp_a @ uniform_grid)[40:60], 0.3 * nu_asym.total_mass(), rtol=1e-12)

        # With the first moment now non-zero, the docstring's original claim is true. Measured
        # max|difference| = 0.299.
        assert not np.allclose(J_comp_a @ v, J_nocomp_a @ v)


class TestLevyAdjoint:
    """Adjoint consistency: <J[v], m>_W = <v, J*[m]>_W."""

    def _check_adjoint(self, J, v, m):
        W = J.integration_weights
        lhs = np.dot(W * m, J @ v)
        rhs = np.dot(W * v, J.apply_adjoint(m))
        rel_error = abs(lhs - rhs) / (abs(lhs) + 1e-15)
        return rel_error

    def test_adjoint_uniform_grid(self, uniform_grid, gaussian_jumps):
        J = LevyIntegroDiffOperator(uniform_grid, gaussian_jumps)
        rng = np.random.RandomState(42)
        v = np.sin(uniform_grid)
        m = np.abs(rng.randn(len(uniform_grid)))
        error = self._check_adjoint(J, v, m)
        assert error < 1e-10, f"Adjoint consistency failed on uniform grid: {error:.2e}"

    def test_adjoint_nonuniform_grid(self, nonuniform_grid, gaussian_jumps):
        """Binding Constraint #2: adjoint MUST work on non-uniform grids."""
        J = LevyIntegroDiffOperator(nonuniform_grid, gaussian_jumps)
        rng = np.random.RandomState(7)
        v = np.sin(nonuniform_grid)
        m = np.abs(rng.randn(len(nonuniform_grid)))
        error = self._check_adjoint(J, v, m)
        assert error < 1e-10, f"Adjoint consistency failed on non-uniform grid: {error:.2e}"

    def test_adjoint_compound_poisson(self, uniform_grid, compound_poisson):
        J = LevyIntegroDiffOperator(uniform_grid, compound_poisson)
        rng = np.random.RandomState(99)
        v = np.sin(uniform_grid)
        m = np.abs(rng.randn(len(uniform_grid)))
        error = self._check_adjoint(J, v, m)
        assert error < 1e-10, f"CP adjoint consistency failed: {error:.2e}"

    def test_adjoint_random_vectors(self, uniform_grid, gaussian_jumps):
        """Multiple random test vectors for robustness."""
        J = LevyIntegroDiffOperator(uniform_grid, gaussian_jumps)
        rng = np.random.RandomState(123)
        for _ in range(5):
            v = rng.randn(len(uniform_grid))
            m = np.abs(rng.randn(len(uniform_grid)))
            error = self._check_adjoint(J, v, m)
            assert error < 1e-10


class TestLevySparseMatrix:
    def test_sparse_matches_matvec(self, uniform_grid, gaussian_jumps):
        J = LevyIntegroDiffOperator(uniform_grid, gaussian_jumps)
        v = np.sin(uniform_grid)
        Jv_matvec = J @ v
        J_sparse = J.as_sparse()
        Jv_sparse = J_sparse @ v
        np.testing.assert_allclose(Jv_matvec, Jv_sparse, atol=1e-12)

    def test_sparse_caching(self, uniform_grid, gaussian_jumps):
        """Second call to as_sparse() should return cached result."""
        J = LevyIntegroDiffOperator(uniform_grid, gaussian_jumps)
        s1 = J.as_sparse()
        s2 = J.as_sparse()
        assert s1 is s2


class TestIntegrationWeights:
    def test_weights_sum_to_domain_length(self, uniform_grid, gaussian_jumps):
        J = LevyIntegroDiffOperator(uniform_grid, gaussian_jumps)
        W = J.integration_weights
        domain_length = uniform_grid[-1] - uniform_grid[0]
        assert abs(W.sum() - domain_length) < 1e-10

    def test_nonuniform_weights(self, nonuniform_grid, gaussian_jumps):
        J = LevyIntegroDiffOperator(nonuniform_grid, gaussian_jumps)
        W = J.integration_weights
        domain_length = nonuniform_grid[-1] - nonuniform_grid[0]
        assert abs(W.sum() - domain_length) < 1e-10
        assert np.all(W > 0)

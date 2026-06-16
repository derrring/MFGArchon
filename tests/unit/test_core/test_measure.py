"""Tests for measure representations (Layer 2 foundation).

Tests MeasureRepresentation Protocol and ParticleMeasure implementation
from mfgarchon.core.measure.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.core.measure import MeasureRepresentation, ParticleMeasure


class TestMeasureRepresentationProtocol:
    def test_particle_measure_satisfies_protocol(self):
        mu = ParticleMeasure(np.array([0.5]))
        assert isinstance(mu, MeasureRepresentation)

    def test_protocol_is_runtime_checkable(self):
        assert hasattr(MeasureRepresentation, "__protocol_attrs__") or isinstance(MeasureRepresentation, type)


class TestParticleMeasureInit:
    def test_1d_from_flat_array(self):
        mu = ParticleMeasure(np.array([0.1, 0.5, 0.9]))
        assert mu.n_particles == 3
        assert mu.dimension == 1
        assert mu.positions.shape == (3, 1)

    def test_2d_particles(self):
        pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        mu = ParticleMeasure(pos)
        assert mu.n_particles == 3
        assert mu.dimension == 2

    def test_uniform_weights_default(self):
        mu = ParticleMeasure(np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(mu.weights, [1 / 3, 1 / 3, 1 / 3])

    def test_custom_weights_normalized(self):
        mu = ParticleMeasure(np.array([0.0, 1.0]), weights=np.array([3.0, 1.0]))
        np.testing.assert_allclose(mu.weights, [0.75, 0.25])

    def test_total_mass_is_one(self):
        mu = ParticleMeasure(np.random.RandomState(42).randn(50))
        assert mu.total_mass() == pytest.approx(1.0)


class TestParticleMeasureToDensity:
    def test_1d_density_shape(self):
        mu = ParticleMeasure(np.array([0.2, 0.5, 0.8]))
        grid = np.linspace(0, 1, 101)
        density = mu.to_density(grid)
        assert density.shape == (101,)

    def test_1d_density_nonnegative(self):
        mu = ParticleMeasure(np.array([0.2, 0.5, 0.8]))
        density = mu.to_density(np.linspace(0, 1, 101))
        assert np.all(density >= 0)

    def test_1d_density_peaks_near_particles(self):
        mu = ParticleMeasure(np.array([0.3, 0.7]))
        grid = np.linspace(0, 1, 201)
        density = mu.to_density(grid)
        # Peaks should be near 0.3 and 0.7
        peak_idx = np.argmax(density)
        assert abs(grid[peak_idx] - 0.3) < 0.15 or abs(grid[peak_idx] - 0.7) < 0.15

    def test_1d_weighted_density(self):
        """Heavier weight at 0.2 should produce higher peak there."""
        mu = ParticleMeasure(np.array([0.2, 0.8]), weights=np.array([9.0, 1.0]))
        grid = np.linspace(0, 1, 201)
        density = mu.to_density(grid)
        idx_02 = 40  # ~0.2
        idx_08 = 160  # ~0.8
        assert density[idx_02] > density[idx_08]

    def test_2d_density(self):
        pos = np.random.RandomState(7).randn(30, 2)
        mu = ParticleMeasure(pos)
        grid_2d = np.random.RandomState(8).randn(50, 2)
        density = mu.to_density(grid_2d)
        assert density.shape == (50,)
        assert np.all(np.isfinite(density))

    def test_custom_bandwidth(self):
        mu = ParticleMeasure(np.array([0.5]))
        grid = np.linspace(0, 1, 101)
        # Narrow bandwidth -> sharper peak
        d_narrow = mu.to_density(grid, bandwidth=0.01)
        d_wide = mu.to_density(grid, bandwidth=0.5)
        assert d_narrow.max() > d_wide.max()


class TestWassersteinDistance:
    def test_identical_measures_zero(self):
        mu = ParticleMeasure(np.array([0.2, 0.5, 0.8]))
        assert mu.wasserstein_distance(mu) == pytest.approx(0.0, abs=1e-12)

    def test_1d_dirac_distance(self):
        """W_2(delta_0, delta_1) = 1."""
        mu = ParticleMeasure(np.array([0.0]))
        nu = ParticleMeasure(np.array([1.0]))
        assert mu.wasserstein_distance(nu, p=2) == pytest.approx(1.0, abs=1e-10)

    def test_1d_translation(self):
        """W_2(mu, mu + shift) = shift for any mu."""
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        mu = ParticleMeasure(x)
        nu = ParticleMeasure(x + 0.5)
        assert mu.wasserstein_distance(nu, p=2) == pytest.approx(0.5, abs=1e-10)

    def test_w1_triangle_inequality(self):
        """W_1(mu, rho) <= W_1(mu, nu) + W_1(nu, rho)."""
        mu = ParticleMeasure(np.array([0.0, 0.1, 0.2]))
        nu = ParticleMeasure(np.array([0.3, 0.4, 0.5]))
        rho = ParticleMeasure(np.array([0.7, 0.8, 0.9]))
        d_mr = mu.wasserstein_distance(rho, p=1)
        d_mn = mu.wasserstein_distance(nu, p=1)
        d_nr = nu.wasserstein_distance(rho, p=1)
        assert d_mr <= d_mn + d_nr + 1e-10

    def test_symmetric(self):
        mu = ParticleMeasure(np.array([0.1, 0.5]))
        nu = ParticleMeasure(np.array([0.3, 0.8]))
        assert mu.wasserstein_distance(nu) == pytest.approx(nu.wasserstein_distance(mu))

    def test_2d_distance(self):
        """2D distance should be non-negative and symmetric."""
        mu = ParticleMeasure(np.array([[0.0, 0.0], [1.0, 0.0]]))
        nu = ParticleMeasure(np.array([[0.0, 1.0], [1.0, 1.0]]))
        d = mu.wasserstein_distance(nu, p=2)
        assert d > 0
        assert d == pytest.approx(nu.wasserstein_distance(mu, p=2))

    def test_dimension_mismatch_raises(self):
        mu = ParticleMeasure(np.array([0.5]))
        nu = ParticleMeasure(np.array([[0.5, 0.5]]))
        with pytest.raises(ValueError, match="Dimension mismatch"):
            mu.wasserstein_distance(nu)

    def test_weighted_1d(self):
        """Weighted measures: heavier weight at 0 vs 1 should give distance < 1."""
        mu = ParticleMeasure(np.array([0.0, 1.0]), weights=np.array([0.9, 0.1]))
        nu = ParticleMeasure(np.array([0.0, 1.0]), weights=np.array([0.1, 0.9]))
        d = mu.wasserstein_distance(nu, p=1)
        assert 0 < d < 1.0

    def test_2d_unequal_weights_sinkhorn(self):
        """nD unequal weights should use Sinkhorn path and give valid distance."""
        mu = ParticleMeasure(
            np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]]),
            weights=np.array([0.5, 0.3, 0.2]),
        )
        nu = ParticleMeasure(
            np.array([[0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]),
            weights=np.array([0.2, 0.5, 0.3]),
        )
        d = mu.wasserstein_distance(nu, p=2)
        assert d > 0
        assert np.isfinite(d)


class TestFromDensity:
    def test_basic_1d(self):
        x = np.linspace(0, 1, 51)
        m = np.exp(-((x - 0.5) ** 2) / 0.02)
        mu = ParticleMeasure.from_density(m, x)
        assert mu.n_particles > 0
        assert mu.dimension == 1
        assert mu.total_mass() == pytest.approx(1.0)

    def test_2d_grid(self):
        x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        m = np.array([0.1, 0.3, 0.2, 0.4])
        mu = ParticleMeasure.from_density(m, x)
        assert mu.n_particles == 4
        assert mu.dimension == 2

    def test_subsampling(self):
        x = np.linspace(0, 1, 100)
        m = np.ones(100)
        mu = ParticleMeasure.from_density(m, x, n_particles=20)
        assert mu.n_particles == 20

    def test_zero_density_filtered(self):
        x = np.linspace(0, 1, 10)
        m = np.array([0, 0, 0, 1, 2, 3, 0, 0, 0, 0], dtype=float)
        mu = ParticleMeasure.from_density(m, x)
        assert mu.n_particles == 3


class TestParticleMeasureRepr:
    def test_repr(self):
        mu = ParticleMeasure(np.zeros((10, 3)))
        assert "n=10" in repr(mu)
        assert "d=3" in repr(mu)


class TestDegenerateCloudWarns1071:
    """Issue #1071: a degenerate particle cloud (zero spread) must not silently fall back to a
    fixed KDE bandwidth — it warns (the fallback is kept, but it is no longer silent).

    Spies on the module logger directly (mfgarchon's get_logger does not propagate to the
    pytest caplog root handler).
    """

    @staticmethod
    def _spy_warnings(monkeypatch):
        import mfgarchon.core.measure as measure_mod

        calls: list[str] = []
        monkeypatch.setattr(measure_mod.logger, "warning", lambda msg, *a, **k: calls.append(str(msg)))
        return calls

    def test_1d_zero_spread_warns(self, monkeypatch):
        calls = self._spy_warnings(monkeypatch)
        mu = ParticleMeasure(np.array([0.5, 0.5, 0.5]))  # std == 0 (degenerate)
        density = mu.to_density(np.linspace(0.0, 1.0, 51))
        assert density.shape == (51,)
        assert any("zero spread" in c for c in calls)

    def test_nd_zero_spread_warns(self, monkeypatch):
        calls = self._spy_warnings(monkeypatch)
        mu = ParticleMeasure(np.full((5, 2), 0.5))  # all at (0.5, 0.5): std == 0
        grid = np.array([[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]])
        mu.to_density(grid)
        assert any("zero spread" in c for c in calls)

    def test_nondegenerate_does_not_warn(self, monkeypatch):
        calls = self._spy_warnings(monkeypatch)
        mu = ParticleMeasure(np.array([0.2, 0.5, 0.8]))  # std > 0
        mu.to_density(np.linspace(0.0, 1.0, 51))
        assert not any("zero spread" in c for c in calls)

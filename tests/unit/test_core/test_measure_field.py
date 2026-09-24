"""Tests for MeasureField (Layer 2).

Tests GridMeasureField and FunctionalMeasureField from mfgarchon.core.measure_field.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.core.measure import ParticleMeasure
from mfgarchon.core.measure_field import (
    FunctionalMeasureField,
    GridMeasureField,
    MeasureFieldProtocol,
)


def _make_grid_field():
    """Create a simple GridMeasureField for testing."""
    x = np.linspace(0, 1, 21)
    t = np.linspace(0, 1, 6)
    return GridMeasureField(x, t)


class TestMeasureFieldProtocol:
    def test_grid_field_satisfies_protocol(self):
        field = _make_grid_field()
        assert isinstance(field, MeasureFieldProtocol)

    def test_functional_field_satisfies_protocol(self):
        field = FunctionalMeasureField(lambda t, x, mu: np.zeros_like(x))
        assert isinstance(field, MeasureFieldProtocol)


class TestGridMeasureFieldInit:
    def test_basic_creation(self):
        field = _make_grid_field()
        assert field.n_snapshots == 0
        assert field.grid_points.shape == (21, 1)
        assert len(field.times) == 6

    def test_repr(self):
        field = _make_grid_field()
        r = repr(field)
        assert "Nx=21" in r
        assert "snapshots=0" in r


class TestGridMeasureFieldSnapshots:
    def test_add_snapshot(self):
        field = _make_grid_field()
        mu = ParticleMeasure(np.array([0.3, 0.5, 0.7]))
        U = np.random.RandomState(42).randn(6, 21)
        field.add_snapshot(mu, U)
        assert field.n_snapshots == 1

    def test_add_wrong_shape_raises(self):
        field = _make_grid_field()
        mu = ParticleMeasure(np.array([0.5]))
        with pytest.raises(ValueError, match="shape"):
            field.add_snapshot(mu, np.zeros((3, 10)))

    def test_get_snapshot(self):
        field = _make_grid_field()
        mu = ParticleMeasure(np.array([0.5]))
        U = np.ones((6, 21))
        field.add_snapshot(mu, U)
        mu_out, U_out = field.get_snapshot(0)
        assert mu_out is mu
        np.testing.assert_array_equal(U_out, U)

    def test_snapshot_is_copied(self):
        """Modifying original array shouldn't affect stored snapshot."""
        field = _make_grid_field()
        mu = ParticleMeasure(np.array([0.5]))
        U = np.ones((6, 21))
        field.add_snapshot(mu, U)
        U[0, 0] = 999.0
        _, U_stored = field.get_snapshot(0)
        assert U_stored[0, 0] == 1.0


class TestGridMeasureFieldEvaluate:
    def test_single_snapshot(self):
        field = _make_grid_field()
        mu = ParticleMeasure(np.array([0.5]))
        U = np.arange(6 * 21, dtype=float).reshape(6, 21)
        field.add_snapshot(mu, U)

        x = field.grid_points[:, 0]
        result = field.evaluate(t=0.0, x=x, mu=mu)
        np.testing.assert_array_equal(result, U[0])

    def test_time_interpolation_nearest(self):
        field = _make_grid_field()
        mu = ParticleMeasure(np.array([0.5]))
        U = np.zeros((6, 21))
        U[3] = 1.0  # t=0.6
        field.add_snapshot(mu, U)

        result = field.evaluate(t=0.55, x=field.grid_points[:, 0], mu=mu)
        # Nearest to 0.55 in [0, 0.2, 0.4, 0.6, 0.8, 1.0] is 0.6 (index 3)
        np.testing.assert_array_equal(result, U[3])

    def test_nearest_snapshot_selection(self):
        """With multiple snapshots, should select nearest in Wasserstein."""
        field = _make_grid_field()
        mu1 = ParticleMeasure(np.array([0.2]))
        mu2 = ParticleMeasure(np.array([0.8]))
        U1 = np.ones((6, 21))
        U2 = np.ones((6, 21)) * 2.0
        field.add_snapshot(mu1, U1)
        field.add_snapshot(mu2, U2)

        # Query near mu1
        mu_query = ParticleMeasure(np.array([0.25]))
        result = field.evaluate(t=0.0, x=field.grid_points[:, 0], mu=mu_query)
        np.testing.assert_array_equal(result, U1[0])

        # Query near mu2
        mu_query2 = ParticleMeasure(np.array([0.75]))
        result2 = field.evaluate(t=0.0, x=field.grid_points[:, 0], mu=mu_query2)
        np.testing.assert_array_equal(result2, U2[0])

    def test_no_snapshots_raises(self):
        field = _make_grid_field()
        mu = ParticleMeasure(np.array([0.5]))
        with pytest.raises(RuntimeError, match="No snapshots"):
            field.evaluate(t=0.0, x=field.grid_points[:, 0], mu=mu)


class TestGridMeasureFieldGradient:
    def test_gradient_linear(self):
        """Gradient of v(x) = x should be 1."""
        field = _make_grid_field()
        mu = ParticleMeasure(np.array([0.5]))
        x = field.grid_points[:, 0]
        U = np.tile(x, (6, 1))  # v(x,t) = x for all t
        field.add_snapshot(mu, U)

        grad = field.spatial_gradient(t=0.0, x=x, mu=mu)
        np.testing.assert_allclose(grad, 1.0, atol=1e-10)

    def test_gradient_quadratic(self):
        """Gradient of v(x) = x^2 should be 2x (edge_order=2 gives exact for degree 2)."""
        field = _make_grid_field()
        mu = ParticleMeasure(np.array([0.5]))
        x = field.grid_points[:, 0]
        U = np.tile(x**2, (6, 1))
        field.add_snapshot(mu, U)

        grad = field.spatial_gradient(t=0.0, x=x, mu=mu)
        np.testing.assert_allclose(grad, 2 * x, atol=1e-10)


class TestGridMeasureFieldRestrict:
    def test_restrict_returns_full_array(self):
        field = _make_grid_field()
        mu = ParticleMeasure(np.array([0.5]))
        U = np.random.RandomState(7).randn(6, 21)
        field.add_snapshot(mu, U)

        restricted = field.restrict_to_density(mu)
        np.testing.assert_array_equal(restricted, U)
        assert restricted is not U  # should be a copy


class TestGridMeasureFieldLionsDerivative:
    def test_fd_derivative_between_snapshots(self):
        field = _make_grid_field()
        mu1 = ParticleMeasure(np.array([0.3]))
        mu2 = ParticleMeasure(np.array([0.7]))
        U1 = np.ones((6, 21))
        U2 = np.ones((6, 21)) * 3.0
        field.add_snapshot(mu1, U1)
        field.add_snapshot(mu2, U2)

        deriv = field.lions_derivative_fd(mu1, t=0.0, snapshot_idx_1=0, snapshot_idx_2=1)
        # (3.0 - 1.0) / W_2(delta_0.3, delta_0.7) = 2.0 / 0.4 = 5.0
        assert deriv.shape == (21,)
        np.testing.assert_allclose(deriv, 5.0, atol=0.1)

    def test_identical_snapshots_gives_zero(self):
        field = _make_grid_field()
        mu = ParticleMeasure(np.array([0.5]))
        U = np.ones((6, 21))
        field.add_snapshot(mu, U)
        field.add_snapshot(mu, U)

        deriv = field.lions_derivative_fd(mu, t=0.0, snapshot_idx_1=0, snapshot_idx_2=1)
        np.testing.assert_allclose(deriv, 0.0, atol=1e-10)


class TestGridMeasureFieldContinuity:
    def test_lipschitz_estimate(self):
        field = _make_grid_field()
        mu1 = ParticleMeasure(np.array([0.2]))
        mu2 = ParticleMeasure(np.array([0.8]))
        U1 = np.zeros((6, 21))
        U2 = np.ones((6, 21))
        field.add_snapshot(mu1, U1)
        field.add_snapshot(mu2, U2)

        lip = field.wasserstein_continuity_estimate(t=0.0)
        assert lip is not None
        assert lip > 0

    def test_single_snapshot_returns_none(self):
        field = _make_grid_field()
        mu = ParticleMeasure(np.array([0.5]))
        field.add_snapshot(mu, np.zeros((6, 21)))
        assert field.wasserstein_continuity_estimate(t=0.0) is None


class TestFunctionalMeasureField:
    def test_evaluate(self):
        def v(t, x, mu):
            return np.sin(np.pi * x) * t

        field = FunctionalMeasureField(v)
        x = np.linspace(0, 1, 11)
        mu = ParticleMeasure(np.array([0.5]))
        result = field.evaluate(t=1.0, x=x, mu=mu)
        np.testing.assert_allclose(result, np.sin(np.pi * x))

    def test_gradient_fd(self):
        """FD gradient of sin(pi*x) should be pi*cos(pi*x)."""

        def v(t, x, mu):
            return np.sin(np.pi * x)

        field = FunctionalMeasureField(v, grid_spacing=0.001)
        x = np.linspace(0.1, 0.9, 9)
        mu = ParticleMeasure(np.array([0.5]))
        grad = field.spatial_gradient(t=0.0, x=x, mu=mu)
        np.testing.assert_allclose(grad, np.pi * np.cos(np.pi * x), atol=1e-4)

    def test_gradient_analytical(self):
        """Provided gradient function should be used."""

        def v(t, x, mu):
            return x**2

        def grad_v(t, x, mu):
            return 2 * x

        field = FunctionalMeasureField(v, gradient_fn=grad_v)
        x = np.linspace(0, 1, 11)
        mu = ParticleMeasure(np.array([0.5]))
        grad = field.spatial_gradient(t=0.0, x=x, mu=mu)
        np.testing.assert_allclose(grad, 2 * x)

    def test_repr(self):
        field = FunctionalMeasureField(lambda t, x, mu: x)
        assert "FunctionalMeasureField" in repr(field)

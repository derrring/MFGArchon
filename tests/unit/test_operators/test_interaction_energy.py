"""Tests for interaction energy functionals (Issue #1023).

Gate 2 (analytic vs FD derivative): QuadraticInteractionEnergy.lions_derivative
equals the finite-difference gradient of its .energy (within FD tolerance),
proving delta F / delta m = K * m.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.operators.interaction.convolution import ConvolutionCouplingOperator
from mfgarchon.operators.interaction.energy_functionals import (
    CombinedEnergy,
    EnergyFunctional,
    PotentialEnergy,
    QuadraticInteractionEnergy,
)
from mfgarchon.operators.interaction.kernels import GaussianKernel, WendlandKernel
from mfgarchon.utils.functional_calculus import FiniteDifferenceFunctionalDerivative


def _grid(N):
    x = np.linspace(0.0, 1.0, N)
    return x, x[1] - x[0]


class TestProtocolConformance:
    def test_quadratic_is_energy_functional(self):
        _, dx = _grid(20)
        conv = ConvolutionCouplingOperator(GaussianKernel(1.0, 0.1), grid_shape=(20,), spacings=[dx])
        assert isinstance(QuadraticInteractionEnergy(conv), EnergyFunctional)

    def test_potential_is_energy_functional(self):
        assert isinstance(PotentialEnergy(np.ones(20)), EnergyFunctional)

    def test_combined_is_energy_functional(self):
        assert isinstance(CombinedEnergy([PotentialEnergy(np.ones(20))]), EnergyFunctional)


class TestGate2AnalyticVsFD:
    """delta F / delta m analytic == finite-difference gradient of the energy."""

    @pytest.mark.parametrize("kernel", [GaussianKernel(1.3, 0.1), WendlandKernel(2.0, 0.25)])
    def test_quadratic_interaction_derivative(self, kernel):
        N = 60
        x, dx = _grid(N)
        conv = ConvolutionCouplingOperator(kernel, grid_shape=(N,), spacings=[dx], use_fft=False)
        energy = QuadraticInteractionEnergy(conv)
        m = np.sin(np.pi * x) + 1.2

        analytic = energy.lions_derivative(m)
        fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-5, method="central")
        numeric = fd.compute(energy.energy, m, x_points=None, y_points=np.arange(N))

        rel = np.max(np.abs(analytic - numeric)) / np.max(np.abs(analytic))
        assert rel < 1e-6

    def test_potential_derivative(self):
        N = 40
        x, _ = _grid(N)
        V = np.cos(2 * np.pi * x)
        energy = PotentialEnergy(V)
        m = np.abs(np.sin(np.pi * x)) + 0.5

        analytic = energy.lions_derivative(m)
        np.testing.assert_allclose(analytic, V, atol=1e-14)

        fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-5, method="central")
        numeric = fd.compute(energy.energy, m, x_points=None, y_points=np.arange(N))
        np.testing.assert_allclose(analytic, numeric, atol=1e-7)

    def test_combined_derivative_is_additive(self):
        N = 50
        x, dx = _grid(N)
        conv = ConvolutionCouplingOperator(GaussianKernel(1.0, 0.12), grid_shape=(N,), spacings=[dx])
        inter = QuadraticInteractionEnergy(conv)
        pot = PotentialEnergy(3.0 * (x - 0.5) ** 2)
        combined = CombinedEnergy([inter, pot])
        m = np.exp(-((x - 0.5) ** 2) / 0.05)

        np.testing.assert_allclose(
            combined.lions_derivative(m),
            inter.lions_derivative(m) + pot.lions_derivative(m),
            atol=1e-12,
        )
        assert combined.energy(m) == pytest.approx(inter.energy(m) + pot.energy(m))

    def test_combined_derivative_vs_fd(self):
        N = 50
        x, dx = _grid(N)
        conv = ConvolutionCouplingOperator(GaussianKernel(1.5, 0.12), grid_shape=(N,), spacings=[dx], use_fft=False)
        combined = CombinedEnergy([QuadraticInteractionEnergy(conv), PotentialEnergy(3.0 * (x - 0.5) ** 2)])
        m = np.exp(-((x - 0.5) ** 2) / 0.05) + 0.3

        analytic = combined.lions_derivative(m)
        fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-5, method="central")
        numeric = fd.compute(combined.energy, m, x_points=None, y_points=np.arange(N))
        rel = np.max(np.abs(analytic - numeric)) / np.max(np.abs(analytic))
        assert rel < 1e-6


class TestEnergyValues:
    def test_quadratic_energy_nonnegative_for_repulsive(self):
        """For a positive-definite kernel, F[m] = (1/2)<m, K*m> >= 0."""
        N = 64
        x, dx = _grid(N)
        conv = ConvolutionCouplingOperator(WendlandKernel(1.0, 0.2), grid_shape=(N,), spacings=[dx])
        energy = QuadraticInteractionEnergy(conv)
        m = np.abs(np.sin(2 * np.pi * x)) + 0.1
        assert energy.energy(m) > 0

    def test_combined_requires_components(self):
        with pytest.raises(ValueError):
            CombinedEnergy([])

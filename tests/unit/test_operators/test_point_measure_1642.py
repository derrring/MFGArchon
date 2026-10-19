"""A coupling cost binds on a point measure, points with one quadrature weight each (#1642 stage 1b-2).

The grid is one point measure (`GridPointMeasure`), and its interaction takes the convolution's FFT path. That fast
path must keep the protocol's semantics: on a uniform grid it equals the direct quadrature on the same points and
weights handed in as a plain point measure. The contract, weights non-negative and summing to the domain's measure, is
checked where a measure is put to use (`bind_on_measure`), and a violation refuses the binding.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.operators.interaction import (
    GridPointMeasure,
    InteractionCost,
    PotentialCost,
    bind_on_measure,
)
from mfgarchon.operators.interaction.kernels import GaussianKernel


@dataclass
class _Points:
    """A point measure that is not a grid's: the direct quadrature, no FFT."""

    points: np.ndarray
    weights: np.ndarray
    domain_measure: float


def _problem():
    grid = TensorProductGrid(
        bounds=[(-1.0, 1.0), (0.0, 2.0)], Nx_points=[17, 13], boundary_conditions=no_flux_bc(dimension=2)
    )
    return MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))),
        domain=grid,
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 0.25, T=0.1),
        Nt=2,
    )


def test_the_grid_s_fft_path_is_the_direct_quadrature_on_the_same_measure():
    """``InteractionCost(K).bind(problem)`` convolves by FFT on a uniform grid; ``bind_on`` a plain point measure with
    the grid's points, weights and box sums ``K(|x - y_j|) w_j m_j`` directly. With an m that is not symmetric on a 17 x
    13 grid, the two agree at the nodes and at points off them to 1e-13 of the field's size."""
    problem = _problem()
    grid_measure = GridPointMeasure(problem)
    cost = InteractionCost(GaussianKernel(amplitude=1.0, length_scale=0.4))
    fft = cost.bind(problem)
    direct = bind_on_measure(
        cost, _Points(grid_measure.points.copy(), grid_measure.weights.copy(), grid_measure.domain_measure)
    )
    nodes = grid_measure.points
    m = 1.0 + 0.5 * np.cos(np.pi * nodes[:, 0]) + 0.3 * nodes[:, 1]
    x = np.random.default_rng(2).uniform([-1.0, 0.0], [1.0, 2.0], (23, 2))
    scale = np.max(np.abs(direct.flat_derivative(m)))
    np.testing.assert_allclose(fft.flat_derivative(m), direct.flat_derivative(m), rtol=0, atol=1e-13 * scale)
    np.testing.assert_allclose(
        fft.flat_derivative_at(m, x), direct.flat_derivative_at(m, x), rtol=0, atol=1e-13 * scale
    )
    np.testing.assert_allclose(fft.energy(m), direct.energy(m), rtol=1e-13)


def test_a_measure_that_is_not_a_quadrature_on_its_domain_is_refused():
    """Weights that are negative, that do not sum to the domain's measure, or that are not one per point refuse the
    binding by name, and so does a cost that has no ``bind_on``."""
    problem = _problem()
    measure = GridPointMeasure(problem)
    points, weights, total = measure.points, measure.weights, measure.domain_measure
    for bad, reason in (
        (_Points(points, np.where(np.arange(weights.size) == 3, -weights, weights), total), "a negative weight"),
        (_Points(points, 1.01 * weights, total), "not to the domain's measure"),
        (_Points(points, weights[:-1], total), "weights for points of shape"),
    ):
        with pytest.raises(ValueError, match=rf"not a quadrature on its domain: .*{reason}"):
            bind_on_measure(PotentialCost(lambda x: x[0]), bad)

    class GridOnly:
        def bind(self, problem):
            return PotentialCost(lambda x: x[0]).bind(problem)

    for cost in (GridOnly(), PotentialCost(lambda x: x[0]) + GridOnly()):
        with pytest.raises(
            NotImplementedError, match=r"GridOnly has no bind_on\(measure\), so it binds only on a grid"
        ):
            bind_on_measure(cost, measure)

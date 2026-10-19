"""A coupling cost binds on a point measure, points with one quadrature weight each (#1642 stage 1b-2).

The grid is one point measure (`GridPointMeasure`), and its interaction takes the convolution's FFT path. That fast
path must keep the protocol's semantics: on a uniform grid it equals the direct quadrature on the same points and
weights handed in as a plain point measure. The contract, weights positive and summing to the domain's measure, is
checked where a measure is put to use (`bind_on_measure`), and a violation refuses the binding.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc, periodic_bc
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


def _problem(grid=None, coupling_cost=None, m0=0.25):
    if grid is None:
        grid = TensorProductGrid(
            bounds=[(-1.0, 1.0), (0.0, 2.0)], Nx_points=[17, 13], boundary_conditions=no_flux_bc(dimension=2)
        )
    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
            coupling_cost=coupling_cost,
        ),
        domain=grid,
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: m0, T=0.1),
        Nt=2,
    )


def test_the_grid_s_fft_path_is_the_direct_quadrature_on_the_same_measure():
    """``InteractionCost(K).bind(problem)`` convolves by FFT on a uniform grid; ``bind_on`` a plain point measure with
    the grid's points, weights and box sums ``K(|x - y_j|) w_j m_j`` directly. With an m that is not symmetric on a 17 x
    13 grid, the two agree at the nodes to 1e-13 of the field's size, and in energy to 1e-13 of the energy. Off the
    nodes both evaluate the dense sum, so that half pins that the grid's binding carries its nodes."""
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
    """A measure whose points or weights have another shape, whose weights are not positive or do not sum to the
    domain's measure, or whose domain measure is not a number refuses the binding by name; so do a cost with no
    ``bind_on``, and one whose ``bind_on`` returns no energy on the measure's points. A float32 measure is checked in
    float64, so one whose weights sum to its domain's measure binds. The grid's measure refuses being subclassed, and
    its points and weights refuse writes, so that the grid path, which reads the problem, cannot ignore a change; a
    periodic grid's interaction is refused there (#2725); and ``use_fft=True`` off a grid is refused."""
    problem = _problem()
    measure = GridPointMeasure(problem)
    points, weights, total = measure.points, measure.weights, measure.domain_measure
    for bad, reason in (
        (_Points(points, np.where(np.arange(weights.size) == 3, 0.0, weights), total), "a weight that is not positive"),
        (_Points(points, (1.0 + 1e-6) * weights, total), "not to the domain's measure"),
        (_Points(points, weights[:-1], total), r"weights of shape \(220,\)"),
        (_Points(points, weights[:, None], total), r"weights of shape \(221, 1\)"),
        (_Points(points, weights, None), "a domain measure of None"),
    ):
        with pytest.raises(ValueError, match=rf"not a quadrature on its domain: .*{reason}"):
            bind_on_measure(PotentialCost(lambda x: x[0]), bad)
    single = weights.astype(np.float32)
    bind_on_measure(PotentialCost(lambda x: x[0]), _Points(points, single, float(single.astype(float).sum())))

    class GridOnly:
        def bind(self, problem):
            return PotentialCost(lambda x: x[0]).bind(problem)

    for cost in (GridOnly(), PotentialCost(lambda x: x[0]) + GridOnly()):
        with pytest.raises(
            NotImplementedError, match=r"GridOnly has no bind_on\(measure\), so it binds only on a grid"
        ):
            bind_on_measure(cost, measure)

    class Returns:
        def __init__(self, energy):
            self.energy = energy

        def bind_on(self, measure):
            return self.energy

    with pytest.raises(
        TypeError, match=r"Returns.bind_on\(measure\) returned object, which is not an EnergyFunctional"
    ):
        bind_on_measure(Returns(object()), measure)
    short = PotentialCost(lambda x: x[0]).bind_on(_Points(points[:-1], weights[:-1], float(weights[:-1].sum())))
    with pytest.raises(ValueError, match=r"returned an energy on \(220,\) weights for a measure of 221 points"):
        bind_on_measure(Returns(short), measure)

    with pytest.raises(TypeError, match=r"Simpson subclasses GridPointMeasure"):

        class Simpson(GridPointMeasure):
            pass

    with pytest.raises(ValueError, match="read-only"):
        measure.weights[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        measure.points[0, 0] = 1.0

    ring = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[20],
        spacing_type="custom",
        custom_coordinates=[np.linspace(0.0, 1.0, 20, endpoint=False)],
        boundary_conditions=periodic_bc(dimension=1),
    )
    with pytest.raises(ValueError, match=r"axis \[0\] of the problem's grid is periodic"):
        _problem(ring, InteractionCost(GaussianKernel(amplitude=1.0, length_scale=0.2)), m0=1 / 0.95)

    with pytest.raises(ValueError, match=r"InteractionCost\(use_fft=True\) binds on _Points"):
        bind_on_measure(
            InteractionCost(GaussianKernel(amplitude=1.0, length_scale=0.4), use_fft=True),
            _Points(points.copy(), weights.copy(), total),
        )


def test_the_grid_s_domain_is_what_its_measure_integrates_over():
    """On cell-centred nodes the problem's measure integrates over the nodes' hull, 0.95 of the box (#2769), and the
    grid's point measure says so: its domain measure is that hull, read from the grid's coordinates, so the costs bind
    as they did before the protocol. #2769's fix returns it to the box."""
    cells = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[20],
        spacing_type="custom",
        custom_coordinates=[np.arange(20) / 20 + 1 / 40],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    cost = PotentialCost(lambda x: x) + InteractionCost(GaussianKernel(amplitude=1.0, length_scale=0.2))
    problem = _problem(cells, cost, m0=1 / 0.95)
    assert GridPointMeasure(problem).domain_measure == pytest.approx(0.95, rel=1e-15)
    np.testing.assert_allclose(np.sum(problem.coupling_energy.weights), 0.95, rtol=1e-15)

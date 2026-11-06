"""In a multi-population solve `m` is the population's own density, and every population's is `cross_density` (#2335).

Before, each population's Hamiltonian was handed the stack of every population's density as `m`, and its terms paired
the stack with the points by position: every population read population 0's density, with no error, and the FP
velocity read node values at the faces. The oracle is the one-population run: with terms that read only the own
density, populations do not interact, so each must equal its own K = 1 run of the same iterator.
"""

from __future__ import annotations

import logging
import warnings

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.fixed_point_utils import compute_fp_velocity_field
from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import (
    CongestionHamiltonian,
    HamiltonianBase,
    HEvalState,
    L1ControlCost,
    QuadraticControlCost,
    SeparableHamiltonian,
    _a_point_of,
)
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.types.callable_protocols import refuse_batch_disagreement

N, NT, SWEEPS = 21, 10, 4


def _problem(hamiltonian, centre=0.4):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[N], boundary_conditions=no_flux_bc(dimension=1))
    return MFGProblem(
        geometry=grid,
        T=0.4,
        Nt=NT,
        volatility=0.3,
        components=MFGComponents(
            hamiltonian=hamiltonian,
            m_initial=lambda x: 0.2 + np.exp(-40 * (x[0] - centre) ** 2),
            u_terminal=lambda x: 0.5 * (x[0] - 0.5) ** 2,
        ),
    )


def _solve(hamiltonians, centres):
    """Every population's (U, M) after a fixed number of sweeps: the tolerance is never met, so runs of any K stop together."""
    problems = [_problem(h, c) for h, c in zip(hamiltonians, centres, strict=True)]
    multi = MultiPopulationProblem(populations=problems, population_names=[f"p{k}" for k in range(len(problems))])
    iterator = MultiPopulationIterator(
        multi, hjb_solvers=[HJBFDMSolver(p) for p in problems], fp_solvers=[FPFDMSolver(p) for p in problems]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.CRITICAL)
        try:
            result = iterator.solve(max_iterations=SWEEPS, tolerance=1e-14)
        finally:
            logging.disable(logging.NOTSET)
    return [(np.asarray(result.U[k]), np.asarray(result.M[k])) for k in range(len(problems))]


def _separable(coupling):
    return SeparableHamiltonian(QuadraticControlCost(control_cost=1.0), coupling=coupling)


KINDS = {
    "separable": lambda: SeparableHamiltonian(QuadraticControlCost(control_cost=1.0), coupling=lambda m: m),
    "separable_l1_velocity_door": lambda: SeparableHamiltonian(L1ControlCost(lambda_=1.0), coupling=lambda m: m),
    "congestion": lambda: CongestionHamiltonian(
        QuadraticControlCost(control_cost=1.0),
        congestion_factor=lambda m: 1.0 + 0.5 * m,
        potential=lambda x: 0.5 * x[0],
    ),
}
CENTRES = (0.25, 0.75, 0.5)


@pytest.mark.parametrize("populations", [2, 3])
@pytest.mark.parametrize("kind", sorted(KINDS))
def test_each_population_of_a_multi_population_solve_is_its_own_one_population_run(kind, populations):
    """A term written for one population, `lambda m: m` or `1 + 0.5 m`, reads the own density at every door, the FP
    velocity's faces included. Measured before: population 1 was 0.2228 off its own run, and the congestion velocity
    read population 0's node densities."""
    make = KINDS[kind]
    together = _solve([make() for _ in range(populations)], CENTRES[:populations])
    for k in range(populations):
        alone = _solve([make()], [CENTRES[k]])[0]
        np.testing.assert_array_equal(together[k][0], alone[0], err_msg=f"U of population {k}")
        np.testing.assert_array_equal(together[k][1], alone[1], err_msg=f"M of population {k}")


def _congested(factor):
    return CongestionHamiltonian(
        QuadraticControlCost(control_cost=1.0), congestion_factor=factor, potential=lambda x: 0.5 * x[0]
    )


@pytest.mark.parametrize("c", [0.5, 0.0])
@pytest.mark.parametrize("door", ["coupling", "congestion_at_the_velocity_faces"])
def test_a_term_reading_every_population_sees_the_sum_it_declares(door, c):
    """Two identical populations, each reading the sum of both, `m + c * (cross_density.sum(axis=0) - m)`, are each the
    one-population run with `(1 + c) * m`; at c = 0 that is the run with `m`. As a congestion factor it is read by the FP
    velocity too, at the faces, where each population's row is averaged as `m` is."""

    def summed(m, cross_density):
        return m + c * (cross_density.sum(axis=0) - m)

    if door == "coupling":
        pair = _solve([_separable(summed), _separable(summed)], [0.4, 0.4])
        alone = _solve([_separable(lambda m: (1.0 + c) * m)], [0.4])[0]
    else:
        pair = _solve([_congested(lambda m, cross_density: 1.0 + 0.5 * summed(m, cross_density))] * 2, [0.4, 0.4])
        alone = _solve([_congested(lambda m: 1.0 + 0.5 * (1.0 + c) * m)], [0.4])[0]
    for k in range(2):
        np.testing.assert_array_equal(pair[k][0], alone[0])
        np.testing.assert_array_equal(pair[k][1], alone[1])


def test_a_term_reading_the_other_population_reads_its_row():
    """A symmetric pair cannot tell two rows swapped, so these differ: each population reads the other's row. Read
    as its own row instead, which is what a swap would hand it, each is the plain one-population coupling `m`."""

    def other(k):
        return lambda m, cross_density: cross_density[1 - k] if len(cross_density) > 1 else 0.0 * m

    def own(k):
        return lambda m, cross_density: cross_density[k] if len(cross_density) > 1 else m

    crossed = _solve([_separable(other(0)), _separable(other(1))], [0.25, 0.75])
    owned = _solve([_separable(own(0)), _separable(own(1))], [0.25, 0.75])
    plain = _solve([_separable(lambda m: m), _separable(lambda m: m)], [0.25, 0.75])
    for k in range(2):
        np.testing.assert_array_equal(owned[k][0], plain[k][0])
        assert np.abs(crossed[k][0] - owned[k][0]).max() > 0.1, f"population {k} read its own row as the other's"


def test_the_velocity_door_at_one_population_hands_the_faces_a_single_population_solve_does():
    """The iterator hands the FP velocity the stack at K = 1 too, where it was read as m: node values at the faces."""
    hamiltonian = CongestionHamiltonian(QuadraticControlCost(control_cost=1.0), congestion_factor=lambda m: 1.0 + m)
    problem = _problem(hamiltonian)
    nodes = np.linspace(0.0, 1.0, N)
    U = np.outer(np.linspace(1.0, 0.5, NT + 1), 0.7 * nodes + 0.4 * nodes**2)
    M = np.outer(np.linspace(1.0, 1.5, NT + 1), 0.2 + nodes**2)
    single = compute_fp_velocity_field(problem, U, M, hamiltonian)
    as_the_iterator_hands_it = compute_fp_velocity_field(problem, U, M, hamiltonian, cross_density=M.copy())
    np.testing.assert_array_equal(as_the_iterator_hands_it, single)


def test_a_density_term_returning_another_count_is_refused_saying_what_it_returned():
    hamiltonian = _separable(lambda m: np.concatenate([m, m]))
    state = HEvalState(x=np.linspace(0.0, 1.0, 4).reshape(-1, 1), p=np.zeros((4, 1)), m=np.ones(4), t=0.0)
    with pytest.raises(ValueError, match=r"coupling was called once .* 4 values for 4 points, and returned 8 values"):
        hamiltonian.evaluate_H(state)


def test_a_per_point_loop_handed_another_count_of_densities_is_refused():
    class PointOnly(HamiltonianBase):
        def __call__(self, t, x, p, m):
            return 0.5 * float(p[0]) ** 2 - float(m)  # one point only, so dp's batch call raises

    x = (np.linspace(0.0, 1.0, 4),)
    with pytest.raises(
        ValueError, match=r"per-point loop pairs one density value with each of 4 points and was handed 8"
    ):
        PointOnly().dp(t=0.0, x=x, p=np.zeros((4, 1)), m=np.ones(8))


def test_the_checks_point_read_of_cross_density_is_that_points_column():
    """`cross_density` holds one row per population, so a point is a column of it, where every other per-point slot is
    indexed on its first axis. A body reading the second population reads the same batch and points."""

    class SecondPopulation(HamiltonianBase):
        def __call__(self, t, x, p, m, cross_density):
            return 0.5 * np.sum(np.asarray(p) ** 2, axis=-1) - np.asarray(cross_density)[1]

    hamiltonian = SecondPopulation()
    batch = {
        "x": np.linspace(0.0, 1.0, 3).reshape(-1, 1),
        "p": np.linspace(-1.0, 1.0, 3).reshape(-1, 1),
        "m": np.array([0.5, 0.9, 1.3]),
        "t": 0.0,
        "cross_density": np.array([[0.5, 0.9, 1.3], [2.0, 3.0, 5.0]]),
    }
    refuse_batch_disagreement(
        lambda **state: hamiltonian.evaluate_H(HEvalState(**state)),
        batch,
        [_a_point_of(batch, i) for i in range(3)],
        name="SecondPopulation.evaluate_H",
        guidance="",
    )

    class OwnRow(HamiltonianBase):
        def __call__(self, t, x, p, m, cross_density):
            return 0.5 * np.sum(np.asarray(p) ** 2, axis=-1) - np.asarray(cross_density)[0]

    # A problem's check reads it at its own density alone, one row, and takes it.
    _problem(OwnRow())

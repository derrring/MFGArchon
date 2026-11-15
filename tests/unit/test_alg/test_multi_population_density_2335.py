"""In a multi-population solve `m` is the population's own density, and every population's reaches a density term that
declares `cross_density` (#2335).

Before, each population's Hamiltonian was handed the stack of every population's density as `m`, and its terms paired
the stack with the points by position: every population read population 0's density, with no error, and the FP
velocity read node values at the faces. The oracle is the one-population run: with terms that read only the own
density, populations do not interact, so each must equal its own K = 1 run of the same iterator.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import types
import warnings
from abc import ABCMeta

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling.fixed_point_utils import compute_fp_velocity_field
from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver, FPFVMSolver, FPParticleSolver, FPSLSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver
from mfgarchon.core import hamiltonian as hamiltonian_module
from mfgarchon.core.hamiltonian import (
    CongestionHamiltonian,
    HamiltonianBase,
    HEvalState,
    L1ControlCost,
    QuadraticControlCost,
    SeparableHamiltonian,
    SeparableLagrangian,
)
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.types.callable_protocols import FrozenSlotError

N, NT, SWEEPS = 21, 10, 4


def _problem(hamiltonian, centre=0.4):
    nodes = np.linspace(0.0, 1.0, N)
    mass = np.trapezoid(0.2 + np.exp(-40 * (nodes - centre) ** 2), nodes)  # the grid's measure, so the mass is 1
    return MFGProblem(
        model=Model(hamiltonian=hamiltonian, volatility=0.3),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[N], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(
            m_initial=lambda x: (0.2 + np.exp(-40 * (x[0] - centre) ** 2)) / mass,
            u_terminal=lambda x: 0.5 * (x[0] - 0.5) ** 2,
            T=0.4,
        ),
        Nt=NT,
    )


def _solve(hamiltonians, centres, fp=FPFDMSolver, problems=None):
    """Every population's (U, M) after a fixed number of sweeps: the tolerance is never met, so runs of any K stop together."""
    problems = problems or [_problem(h, c) for h, c in zip(hamiltonians, centres, strict=True)]
    multi = MultiPopulationProblem(populations=problems, population_names=[f"p{k}" for k in range(len(problems))])
    iterator = MultiPopulationIterator(
        multi, hjb_solvers=[HJBFDMSolver(p) for p in problems], fp_solvers=[fp(p) for p in problems]
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


def _congested(factor):
    return CongestionHamiltonian(
        QuadraticControlCost(control_cost=1.0), congestion_factor=factor, potential=lambda x: 0.5 * x[0]
    )


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
    velocity's faces included. Before, every population read population 0's density, and the congestion velocity read
    node densities at the faces."""
    make = KINDS[kind]
    together = _solve([make() for _ in range(populations)], CENTRES[:populations])
    for k in range(populations):
        alone = _solve([make()], [CENTRES[k]])[0]
        np.testing.assert_array_equal(together[k][0], alone[0], err_msg=f"U of population {k}")
        np.testing.assert_array_equal(together[k][1], alone[1], err_msg=f"M of population {k}")


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
    """A direct call handing the FP velocity a one-row stack at one population reads the faces a single-population solve
    does. At f59bb154 that stack was read as m, node values at the faces. The iterator hands no stack at K = 1."""
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


class _PointOnly(HamiltonianBase):
    def __call__(self, t, x, p, m):
        return 0.5 * float(p[0]) ** 2 - float(m)  # one point only, so a batch call raises


@pytest.mark.parametrize(
    ("hamiltonian", "method"),
    [
        (_PointOnly, "dp"),
        (_PointOnly, "dm"),
        (_PointOnly, "dx"),
        (lambda: SeparableHamiltonian(QuadraticControlCost(control_cost=1.0), coupling=lambda m: m), "dm"),
        (
            lambda: CongestionHamiltonian(QuadraticControlCost(control_cost=1.0), congestion_factor=lambda m: 1.0 + m),
            "dm",
        ),
    ],
    ids=["dp", "dm", "dx", "separable_dm", "congestion_dm"],
)
def test_a_per_point_loop_handed_another_count_of_densities_is_refused(hamiltonian, method):
    x = (np.linspace(0.0, 1.0, 4),)
    with pytest.raises(
        ValueError, match=rf"\.{method}'s per-point loop pairs one density value with each of 4 points and was handed 8"
    ):
        getattr(hamiltonian(), method)(t=0.0, x=x, p=np.zeros((4, 1)), m=np.ones(8))


def test_a_congestion_factor_reading_the_other_row_reads_it_at_every_door():
    """Two populations, each congested by the other, written two ways: by row, `cross_density[1 - k]`, and as
    `cross_density.sum(axis=0) - m`, which no order of the rows can change. At K = 2 the two are one density, so the
    solves agree, and differ from the run with no cross term. The factor is read at the HJB residual, the Jacobian and
    the FP velocity, so a row handed in another order at any one of them is a difference."""

    def by_row(k):
        return lambda m, cross_density: (
            1.0 + 0.5 * m + 0.8 * (cross_density[1 - k] if len(cross_density) > 1 else 0.0 * m)
        )

    def by_sum(m, cross_density):
        return 1.0 + 0.5 * m + 0.8 * (cross_density.sum(axis=0) - m)

    centres = [0.25, 0.75]
    rows = _solve([_congested(by_row(k)) for k in range(2)], centres)
    summed = _solve([_congested(by_sum) for _ in range(2)], centres)
    uncrossed = _solve([_congested(lambda m: 1.0 + 0.5 * m) for _ in range(2)], centres)
    for k in range(2):
        np.testing.assert_allclose(rows[k][0], summed[k][0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(rows[k][1], summed[k][1], rtol=0, atol=1e-12)
        assert np.abs(rows[k][1] - uncrossed[k][1]).max() > 1e-3, f"population {k} read no other population"


def test_the_velocity_door_reads_each_density_at_the_faces_as_the_mean_of_its_nodes():
    """At the FP velocity a density is read at the faces: a linear one's value there is `a + b * x_face`, which no
    node holds. Both `m` and each row of `cross_density` are read so, by a congestion factor that declares the slot."""
    seen = []

    def factor(m, cross_density):
        if np.size(m) == N - 1:  # the faces; the problem's own checks read fewer points
            seen.append((np.array(m, dtype=float), np.array(cross_density, dtype=float)))
        return 1.0 + 0.0 * np.asarray(m)

    hamiltonian = _congested(factor)
    problem = _problem(hamiltonian)
    nodes = np.linspace(0.0, 1.0, N)
    faces = 0.5 * (nodes[:-1] + nodes[1:])
    own, other = 0.3 + 2.0 * nodes, 1.2 - 0.5 * nodes
    U = np.outer(np.linspace(1.0, 0.5, NT + 1), 0.7 * nodes + 0.4 * nodes**2)
    M = np.tile(own, (NT + 1, 1))
    seen.clear()
    compute_fp_velocity_field(
        problem, U, M, hamiltonian, cross_density=np.concatenate([M, np.tile(other, (NT + 1, 1))], axis=-1)
    )
    assert seen
    for m, rows in seen:
        np.testing.assert_allclose(m, 0.3 + 2.0 * faces, rtol=0, atol=1e-13)
        np.testing.assert_allclose(rows, np.stack([0.3 + 2.0 * faces, 1.2 - 0.5 * faces]), rtol=0, atol=1e-13)


def _crowd(k):
    """`1 + m + 0.8 q`, q the other population's density, read as a density term."""
    return lambda m, cross_density: 1.0 + m + 0.8 * (cross_density[1 - k] if len(cross_density) > 1 else 0.0 * m)


@pytest.mark.parametrize("method", ["__call__", "dp", "optimal_control", "dm"])
def test_a_family_method_that_declares_cross_density_is_refused_and_its_advice_runs(method):
    """A family method reads the own density as m, so a class whose family method declares the slot is refused when
    it is created. Its advice runs: `|p|^2 / (2 (1 + m + 0.8 q)) - m` written with the cross term in a
    congestion factor reads the other population, and solves apart from the run with no cross term."""

    def body(self, t, x, p, m, cross_density):
        return 0.5 * np.sum(np.asarray(p) ** 2, axis=-1)

    members = {"__call__": lambda self, t, x, p, m: 0.0, method: body}
    with pytest.raises(
        TypeError,
        match=rf"\.{method} declares cross_density\. .* coupling= or congestion_factor= of a SeparableHamiltonian or "
        r"CongestionHamiltonian as the library ships it",
    ):
        type("Declares", (HamiltonianBase,), members)
    cost = QuadraticControlCost(control_cost=1.0)
    centres = [0.25, 0.75]
    crowded = _solve(
        [CongestionHamiltonian(cost, congestion_factor=_crowd(k), coupling=lambda m: m) for k in range(2)], centres
    )
    alone = _solve(
        [CongestionHamiltonian(cost, congestion_factor=lambda m: 1.0 + m, coupling=lambda m: m)] * 2, centres
    )
    for k in range(2):
        assert np.abs(crowded[k][1] - alone[k][1]).max() > 1e-3, f"population {k} read no other population"


@pytest.mark.parametrize(
    "member",
    [
        staticmethod(lambda t, x, p, m, cross_density: p),
        classmethod(lambda cls, t, x, p, m, cross_density: p),
    ],
    ids=["staticmethod", "classmethod"],
)
def test_a_static_or_class_method_declaring_cross_density_is_refused_too(member):
    """The class-creation check reads the function a static or class method wraps."""
    with pytest.raises(TypeError, match=r"\.dp declares cross_density\."):
        type("Declares", (HamiltonianBase,), {"__call__": lambda self, t, x, p, m: 0.0, "dp": member})


def _second_population(nodes, upper, *, t_end=0.4, nt=NT):
    return MFGProblem(
        model=Model(hamiltonian=_separable(lambda m: m), volatility=0.3),
        domain=TensorProductGrid(bounds=[(0.0, upper)], Nx_points=[nodes], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(m_initial=lambda x: 1.0 / upper + 0.0 * x[0], u_terminal=lambda x: 0.0 * x[0], T=t_end),
        Nt=nt,
    )


@pytest.mark.parametrize(
    ("second", "said"),
    [
        (_second_population(2 * N, 1.0), "42 nodes against 21"),
        (_second_population(N, 2.0), r"coordinates up to 1\.000e\+00 apart"),
        (_second_population(N, 0.1 * 3 / 0.3), r"coordinates up to 2\.220e-16 apart"),
    ],
    ids=["more_nodes", "same_count_on_another_domain", "apart_by_round_off"],
)
def test_populations_not_on_one_grid_are_refused(second, said):
    """`cross_density` reads every population at one population's points, row for row, so the populations of one solve
    share one grid, compared exactly: one node count on [0, 1] and on [0, 2] is two grids, and so is an upper bound of
    `0.1 * 3 / 0.3`. The refusal says how far apart they are."""
    with pytest.raises(
        ValueError, match=rf"population\(s\) \[1\] sit on a grid other than population 0's \(population 1: {said}"
    ):
        _solve(None, None, problems=[_problem(_separable(lambda m: m), 0.25), second])


@pytest.mark.parametrize(
    ("second", "said"),
    [
        (_second_population(N, 1.0, nt=12), r"\(0\.4, 12\)\] against \(0\.4, 10\)"),
        (_second_population(N, 1.0, t_end=1.2), r"\(1\.2, 10\)\] against \(0\.4, 10\)"),
    ],
    ids=["other_nt", "other_t"],
)
def test_populations_not_on_one_time_grid_are_refused(second, said):
    """Every population's density is handed at the same step, so the populations of one solve share one time grid. A
    different `Nt` failed in numpy's concatenate; a different `T` with the same `Nt` solved, each population reading the
    other's density at another time."""
    with pytest.raises(
        ValueError, match=rf"population\(s\) \[1\] step through time other than population 0's: \(T, Nt\) \[{said}"
    ):
        _solve(None, None, problems=[_problem(_separable(lambda m: m), 0.25), second])


def test_a_grid_of_two_dimensions_is_refused():
    """The iterator stacks one row of nodes per population and reads the rows on a 1-D grid; a 2-D grid failed in numpy's
    broadcast when the iterator set the densities up."""
    problems = [
        MFGProblem(
            model=Model(hamiltonian=_separable(lambda m: m), volatility=0.3),
            domain=TensorProductGrid(
                bounds=[(0.0, 1.0)] * 2, Nx_points=[7, 7], boundary_conditions=no_flux_bc(dimension=2)
            ),
            conditions=Conditions(m_initial=lambda x: 1.0 + 0.0 * x[0], u_terminal=lambda x: 0.0 * x[0], T=0.4),
            Nt=4,
        )
        for _ in range(2)
    ]
    with pytest.raises(NotImplementedError, match=r"population\(s\) \[0, 1\] sit on a grid of dimension \[2, 2\]"):
        _solve(None, None, problems=problems)


def test_a_term_that_froze_cross_density_by_keyword_is_refused():
    """The library passes a term `cross_density` by name, so a partial that froze it by keyword would have its value
    overridden: refused where the term is handed over (#2869)."""

    def summed(m, cross_density):
        return np.asarray(cross_density).sum(axis=0)

    with pytest.raises(FrozenSlotError, match="freezes cross_density"):
        _separable(functools.partial(summed, cross_density=np.zeros((1, 4))))


def _sum_coupling(m, cross_density):
    return m + 0.5 * (np.asarray(cross_density).sum(axis=0) - m)


class _OwnCall(SeparableHamiltonian):
    def __call__(self, t, x, p, m):
        return super().__call__(t=t, x=x, p=p, m=m)


def test_a_subclass_over_a_term_that_does_not_declare_the_slot_solves_as_the_library_class():
    """A subclass is read through its public methods with the own density, which is all a term that does not declare
    the slot reads: at two populations it solves as the library class."""
    cost = QuadraticControlCost(control_cost=1.0)
    overridden = _solve([_OwnCall(cost, coupling=lambda m: m) for _ in range(2)], [0.25, 0.75])
    library = _solve([_separable(lambda m: m) for _ in range(2)], [0.25, 0.75])
    for k in range(2):
        np.testing.assert_array_equal(overridden[k][0], library[k][0])
        np.testing.assert_array_equal(overridden[k][1], library[k][1])


class _CouplingOfItsOwn(SeparableHamiltonian):
    """`__call__` of its own, `|p|^2 / 2 - m`, over a base whose coupling is `0.5 m`."""

    def __call__(self, t, x, p, m):
        return 0.5 * np.sum(np.asarray(p, dtype=float) ** 2, axis=-1) - np.asarray(m, dtype=float)


def _call_for_factor_1_plus_m(t, x, p, m):
    return 0.5 * np.sum(np.asarray(p, dtype=float) ** 2, axis=-1) / (1.0 + np.asarray(m, dtype=float))


def _dp_for_factor_1_plus_m(t, x, p, m):
    p = np.asarray(p, dtype=float)
    crowd = 1.0 + np.asarray(m, dtype=float)
    return p / crowd[:, np.newaxis] if p.ndim == 2 else p / crowd


def _control_for_factor_1_plus_m(t, x, p, m):
    return -_dp_for_factor_1_plus_m(t=t, x=x, p=p, m=m)


class _CallAndDp(CongestionHamiltonian):
    """`__call__` and `dp` of its own, for a congestion factor `1 + m`, over a base whose factor is `1 + 0.5 m`; the
    library's `optimal_control`, which is `-dp`."""

    def __call__(self, t, x, p, m):
        return _call_for_factor_1_plus_m(t=t, x=x, p=p, m=m)

    def dp(self, t, x, p, m):
        return _dp_for_factor_1_plus_m(t=t, x=x, p=p, m=m)


class _FactorOfItsOwn(_CallAndDp):
    """`_CallAndDp` with its `optimal_control` written out too: `__call__`, `dp` and `optimal_control` of its own."""

    def optimal_control(self, t, x, p, m):
        return _control_for_factor_1_plus_m(t=t, x=x, p=p, m=m)


@pytest.mark.parametrize(
    ("own", "library"),
    [
        (
            lambda: _CouplingOfItsOwn(QuadraticControlCost(control_cost=1.0), coupling=lambda m: 0.5 * m),
            lambda: _separable(lambda m: m),
        ),
        (
            lambda: _FactorOfItsOwn(QuadraticControlCost(control_cost=1.0), congestion_factor=lambda m: 1.0 + 0.5 * m),
            lambda: CongestionHamiltonian(QuadraticControlCost(control_cost=1.0), congestion_factor=lambda m: 1.0 + m),
        ),
        (
            lambda: _CallAndDp(QuadraticControlCost(control_cost=1.0), congestion_factor=lambda m: 1.0 + 0.5 * m),
            lambda: _FactorOfItsOwn(QuadraticControlCost(control_cost=1.0), congestion_factor=lambda m: 1.0 + 0.5 * m),
        ),
    ],
    ids=["separable_call", "congestion_call_dp_and_optimal_control", "congestion_call_and_dp_against_its_twin"],
)
def test_a_body_of_yours_overriding_a_library_method_is_read_at_every_door(own, library):
    """A door reads a library class through its own primitive, which reads an override of the method it stands for: a
    subclass that redefines H solves as the library class for that H, at the residual, the Jacobian and the velocity. A
    congestion `dp` of its own is read at the velocity door also where `optimal_control` is left to the library, whose
    body is `-dp`: the class solves as its twin with `optimal_control` written out."""
    (overridden,), (reference,) = _solve([own()], [0.4]), _solve([library()], [0.4])
    np.testing.assert_allclose(overridden[0], reference[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(overridden[1], reference[1], rtol=0, atol=1e-12)


def test_an_optimal_control_assigned_on_the_instance_is_read_at_the_velocity_door():
    """At one population every primitive calls the public method, so the velocity door calls a congestion
    Hamiltonian's optimal_control assigned on it. Calls are counted during the solve only, so a read at the problem's
    construction does not count."""
    hamiltonian = CongestionHamiltonian(
        QuadraticControlCost(control_cost=1.0), congestion_factor=lambda m: 1.0 + 0.5 * m
    )
    library_control, calls = hamiltonian.optimal_control, []

    def counted(t, x, p, m):
        calls.append(np.shape(m))
        return library_control(t=t, x=x, p=p, m=m)

    hamiltonian.optimal_control = counted
    problems = [_problem(hamiltonian, 0.4)]
    calls.clear()
    _solve([], [], problems=problems)
    assert calls


class _Served(CongestionHamiltonian):
    """`__call__` of its own, and `dp` and `optimal_control` served by `__getattribute__`, for a congestion factor
    `1 + m`, over a base whose factor is `1 + 0.5 m`."""

    def __call__(self, t, x, p, m):
        return _call_for_factor_1_plus_m(t=t, x=x, p=p, m=m)

    def __getattribute__(self, name):
        if name == "dp":
            return _dp_for_factor_1_plus_m
        if name == "optimal_control":
            return _control_for_factor_1_plus_m
        return super().__getattribute__(name)


class _TwoFaced:
    """A descriptor whose class face is a library function, as a function's is, and whose instance face another body."""

    def __init__(self, library, body):
        self.library, self.body = library, body

    def __get__(self, instance, owner=None):
        return self.library if instance is None else types.MethodType(self.body, instance)


_OWN_BODIES = {
    "__call__": lambda self, t, x, p, m: _call_for_factor_1_plus_m(t, x, p, m),
    "dp": lambda self, t, x, p, m: _dp_for_factor_1_plus_m(t, x, p, m),
    "optimal_control": lambda self, t, x, p, m: _control_for_factor_1_plus_m(t, x, p, m),
}


class _ShowsTheLibrary(ABCMeta):
    """A metaclass whose `__getattribute__` shows `CongestionHamiltonian`'s `__call__`, `dp` and `optimal_control` on
    the class, whatever its body defines."""

    def __getattribute__(cls, name):
        if name in _OWN_BODIES:
            return getattr(CongestionHamiltonian, name)
        return super().__getattribute__(name)


def _default_solve(hamiltonian):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.CRITICAL)
        try:
            result = _problem(hamiltonian).solve(max_iterations=SWEEPS, tolerance=1e-14, verbose=False)
        finally:
            logging.disable(logging.NOTSET)
    return np.asarray(result.U), np.asarray(result.M)


@pytest.mark.parametrize("through", ["problem_solve", "the_iterator_at_one_population"])
@pytest.mark.parametrize(
    "served",
    [
        "by_getattribute",
        "by_a_descriptor_showing_the_library",
        "under_a_metaclass_showing_the_library",
        "replaced_on_the_library_class",
    ],
)
def test_a_body_served_from_outside_the_class_body_is_read_at_one_population(served, through, monkeypatch):
    """At one population the iterator and `problem.solve()` hand no stack, so every primitive calls the public method:
    `__call__`, `dp` and `optimal_control` served however Python serves them are read at every door, and the default
    scheme, or the iterator at K = 1, solves as the twin that writes them in its class body. At ca9e9514 the doors
    asked a class-level identity instead and read the library's bodies: the default-scheme rows failed there on U."""
    cost = QuadraticControlCost(control_cost=1.0)
    solve = _default_solve if through == "problem_solve" else (lambda hamiltonian: _solve([hamiltonian], [0.4])[0])
    twin = solve(_FactorOfItsOwn(cost, congestion_factor=lambda m: 1.0 + 0.5 * m))
    if served == "by_getattribute":
        hamiltonian = _Served(cost, congestion_factor=lambda m: 1.0 + 0.5 * m)
    elif served == "by_a_descriptor_showing_the_library":
        members = {name: _TwoFaced(getattr(CongestionHamiltonian, name), body) for name, body in _OWN_BODIES.items()}
        hamiltonian = type("_TwoFacedFactor", (CongestionHamiltonian,), members)(
            cost, congestion_factor=lambda m: 1.0 + 0.5 * m
        )
    elif served == "under_a_metaclass_showing_the_library":
        hamiltonian = _ShowsTheLibrary("_ShownFactor", (CongestionHamiltonian,), dict(_OWN_BODIES))(
            cost, congestion_factor=lambda m: 1.0 + 0.5 * m
        )
    else:
        monkeypatch.setattr(
            CongestionHamiltonian, "__call__", lambda self, t, x, p, m: _call_for_factor_1_plus_m(t, x, p, m)
        )
        monkeypatch.setattr(CongestionHamiltonian, "dp", lambda self, t, x, p, m: _dp_for_factor_1_plus_m(t, x, p, m))
        monkeypatch.setattr(
            CongestionHamiltonian, "optimal_control", lambda self, t, x, p, m: _control_for_factor_1_plus_m(t, x, p, m)
        )
        hamiltonian = CongestionHamiltonian(cost, congestion_factor=lambda m: 1.0 + 0.5 * m)
    solved = solve(hamiltonian)
    np.testing.assert_allclose(solved[0], twin[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(solved[1], twin[1], rtol=0, atol=1e-12)


class _OwnControl(SeparableHamiltonian):
    """`optimal_control` of its own, the library's `-p` written out (#2900)."""

    def optimal_control(self, t, x, p, m):
        return -np.asarray(p, dtype=float)


class _OwnDp(SeparableHamiltonian):
    """`dp` of its own, the library's `p` for a unit quadratic cost written out."""

    def dp(self, t, x, p, m):
        return np.asarray(p, dtype=float)


class _DpOnly(CongestionHamiltonian):
    def dp(self, t, x, p, m):
        return _dp_for_factor_1_plus_m(t=t, x=x, p=p, m=m)


def _separable_form(cls):
    return lambda k: cls(QuadraticControlCost(control_cost=1.0), coupling=_sum_coupling)


def _dp_assigned_on_the_instance(k):
    hamiltonian = _separable(_sum_coupling)
    hamiltonian.dp = lambda t, x, p, m: np.asarray(p, dtype=float)
    return hamiltonian


def _library_method_assigned_on_the_instance(name):
    """A congestion Hamiltonian over a declaring factor whose instance holds its own library method `name`, bound: the
    body is the library's, and only the instance's `__dict__` differs from the class as it ships."""

    def make(k):
        hamiltonian = CongestionHamiltonian(QuadraticControlCost(control_cost=1.0), congestion_factor=_crowd(k))
        setattr(hamiltonian, name, getattr(hamiltonian, name))
        return hamiltonian

    return make


_SEPARABLE_TWO_FACED = type(
    "_SeparableTwoFaced",
    (SeparableHamiltonian,),
    {
        "__call__": _TwoFaced(
            SeparableHamiltonian.__call__, lambda self, t, x, p, m: SeparableHamiltonian.__call__(self, t, x, p, m)
        )
    },
)
_SEPARABLE_SHOWN = _ShowsTheLibrary(
    "_SeparableShown",
    (SeparableHamiltonian,),
    {"__call__": lambda self, t, x, p, m: SeparableHamiltonian.__call__(self, t, x, p, m)},
)
#: name -> (one population's Hamiltonian, its terms that declare the slot, why it is not as the library ships it).
NOT_AS_IT_SHIPS = {
    "subclass_with_its_own_call": (_separable_form(_OwnCall), "coupling", "its class is _OwnCall"),
    "subclass_with_its_own_optimal_control": (_separable_form(_OwnControl), "coupling", "its class is _OwnControl"),
    "subclass_with_its_own_dp": (_separable_form(_OwnDp), "coupling", "its class is _OwnDp"),
    "subclass_defining_nothing": (
        _separable_form(type("_Bare", (SeparableHamiltonian,), {})),
        "coupling",
        "its class is _Bare",
    ),
    "descriptor_showing_the_library": (
        _separable_form(_SEPARABLE_TWO_FACED),
        "coupling",
        "its class is _SeparableTwoFaced",
    ),
    "under_a_metaclass_showing_the_library": (
        _separable_form(_SEPARABLE_SHOWN),
        "coupling",
        "its class is _SeparableShown",
    ),
    "dp_assigned_on_the_instance": (
        _dp_assigned_on_the_instance,
        "coupling",
        r"its instance holds \['dp'\] of its own",
    ),
    "congestion_subclass_with_its_own_dp": (
        lambda k: _DpOnly(QuadraticControlCost(control_cost=1.0), congestion_factor=_crowd(k)),
        "congestion_factor",
        "its class is _DpOnly",
    ),
    **{
        f"{name}_assigned_on_the_instance": (
            _library_method_assigned_on_the_instance(name),
            "congestion_factor",
            rf"its instance holds \['{name}'\] of its own",
        )
        for name in ("__call__", "dm", "dx", "optimal_control", "evaluate_H", "evaluate_dp", "evaluate_optimal_control")
    },
}


@pytest.mark.parametrize("form", sorted(NOT_AS_IT_SHIPS))
def test_at_two_populations_a_term_declaring_the_slot_takes_a_library_class_as_it_ships(form):
    """At two populations a Hamiltonian with a term that declares the slot, which is not a `SeparableHamiltonian` or
    `CongestionHamiltonian` as the library ships it, is refused before the first sweep, naming the class, the term and
    why. The `*_assigned_on_the_instance` rows hold one name each in the instance's `__dict__`."""
    make, terms, why = NOT_AS_IT_SHIPS[form]
    population = type(make(0)).__name__
    with pytest.raises(
        NotImplementedError,
        match=rf"population 0's {population} has term\(s\) \['{terms}'\] that declare cross_density, and is not a "
        rf"SeparableHamiltonian or CongestionHamiltonian as the library ships it: {why}\.",
    ):
        _solve([make(k) for k in range(2)], [0.25, 0.75])


def _recording(control, reads):
    """The control it wraps, recording the density each read hands it."""

    def read(t, x, p, m):
        reads.append(np.shape(m))
        return control(t=t, x=x, p=p, m=m)

    return read


def test_at_two_populations_an_optimal_control_assigned_on_the_instance_is_what_the_velocity_door_reads():
    """Over a congestion factor that does not declare the slot, each of two populations holds an `optimal_control`
    assigned on its instance: the velocity door reads the assignment at two populations too, not the rows path.

    The assignment returns the library's control and records each read. Since #2901 construction reads it too, at its
    probes, so the records are cleared once the problems are built: what is counted is the solve's reads alone, each an
    array of densities, one per node the door reads."""

    def factor(m):
        return 1.0 + 0.5 * m

    def potential(x):
        return 0.5 * x[0]

    cost = QuadraticControlCost(control_cost=1.0)
    assigned = [CongestionHamiltonian(cost, congestion_factor=factor, potential=potential) for _ in range(2)]
    reads: list[list[tuple[int, ...]]] = [[], []]
    for hamiltonian, record in zip(assigned, reads, strict=True):
        hamiltonian.optimal_control = _recording(hamiltonian.optimal_control, record)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problems = [_problem(h, c) for h, c in zip(assigned, [0.25, 0.75], strict=True)]
    for record in reads:
        record.clear()
    solved = _solve(assigned, [0.25, 0.75], problems=problems)
    for k in range(2):
        assert reads[k], f"population {k}'s assigned optimal_control was not read by the solve"
        assert all(len(shape) == 1 and shape[0] > 1 for shape in reads[k]), (k, sorted(set(reads[k])))
        assert np.all(np.isfinite(solved[k][1])), f"population {k}"


def _lagrangian_problem(operator, centre):
    nodes = np.linspace(0.0, 1.0, N)
    mass = np.trapezoid(0.2 + np.exp(-40 * (nodes - centre) ** 2), nodes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MFGProblem(
            geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[N], boundary_conditions=no_flux_bc(dimension=1)),
            T=0.4,
            Nt=NT,
            volatility=0.3,
            components=MFGComponents(
                m_initial=lambda x: (0.2 + np.exp(-40 * (x[0] - centre) ** 2)) / mass,
                u_terminal=lambda x: 0.5 * (x[0] - 0.5) ** 2,
                **operator,
            ),
        )


def test_a_problem_built_from_a_separable_lagrangian_hands_its_coupling_every_row():
    """`MFGComponents(lagrangian=...)` is solved through the `SeparableHamiltonian` it converts to, as the library ships
    it and sharing the Lagrangian's coupling, so at two populations a coupling that declares the slot is handed every
    population's rows, and the solve equals the one built from that Hamiltonian."""
    handed = []

    def coupling(k):
        def read(m, cross_density):
            handed.append(len(cross_density))
            return m + 0.7 * (cross_density[1 - k] if len(cross_density) > 1 else 0.0 * m)

        return read

    cost = QuadraticControlCost(control_cost=1.0)
    centres = (0.25, 0.75)
    built = [
        _lagrangian_problem({"lagrangian": SeparableLagrangian(cost, coupling=coupling(k))}, c)
        for k, c in enumerate(centres)
    ]
    assert type(built[0].hamiltonian_class) is SeparableHamiltonian
    by_lagrangian = _solve(None, None, problems=built)
    assert 2 in handed
    direct = [
        _lagrangian_problem({"hamiltonian": SeparableHamiltonian(cost, coupling=coupling(k))}, c)
        for k, c in enumerate(centres)
    ]
    by_hamiltonian = _solve(None, None, problems=direct)
    for k in range(2):
        np.testing.assert_array_equal(by_lagrangian[k][0], by_hamiltonian[k][0])
        np.testing.assert_array_equal(by_lagrangian[k][1], by_hamiltonian[k][1])


def test_a_per_point_loop_handed_one_density_value_reads_it_at_every_point():
    """One density value holds at every point of a batch, in a per-point loop as in a batch call."""
    x, p = (np.linspace(0.0, 1.0, 4),), np.linspace(-1.0, 1.0, 4).reshape(-1, 1)
    np.testing.assert_array_equal(
        _PointOnly().dp(t=0.0, x=x, p=p, m=np.array([0.7])), _PointOnly().dp(t=0.0, x=x, p=p, m=np.full(4, 0.7))
    )


def test_a_stack_that_is_not_whole_rows_is_refused_at_the_door():
    """A door reads a solver's flat stack as one row per population at its points, and refuses a stack of another
    size."""
    hamiltonian = _congested(lambda m, cross_density: 1.0 + np.asarray(cross_density).sum(axis=0))
    nodes = np.linspace(0.0, 1.0, N)
    U = np.outer(np.linspace(1.0, 0.5, NT + 1), 0.7 * nodes)
    M = np.outer(np.ones(NT + 1), 0.2 + nodes**2)
    with pytest.raises(
        ValueError, match=rf"a cross density of {2 * N + 3} values is not a whole number of rows of the {N}"
    ):
        compute_fp_velocity_field(_problem(hamiltonian), U, M, hamiltonian, cross_density=np.ones((NT + 1, 2 * N + 3)))


def test_an_own_optimal_control_is_read_at_the_own_density_in_a_multi_population_solve():
    """#2900 routes an `optimal_control` of a class's own to the FP velocity door, which reads it at the own density in
    a multi-population solve too: each population of a K = 2 solve is its own K = 1 run."""
    cost = QuadraticControlCost(control_cost=1.0)
    together = _solve([_OwnControl(cost, coupling=lambda m: m) for _ in range(2)], [0.25, 0.75])
    for k, centre in enumerate([0.25, 0.75]):
        (alone,) = _solve([_OwnControl(cost, coupling=lambda m: m)], [centre])
        np.testing.assert_array_equal(together[k][0], alone[0])
        np.testing.assert_array_equal(together[k][1], alone[1])


def test_a_direct_call_reads_a_term_that_declares_the_slot_at_the_own_row():
    """A direct call of a Hamiltonian whose term declares the slot is one population's: the term reads the own row,
    `m[np.newaxis]`, and the Hamiltonian reads as the same one written for one population."""
    x, p, m = (np.linspace(0.0, 1.0, 4),), np.linspace(-1.0, 1.0, 4).reshape(-1, 1), np.array([0.5, 0.9, 1.3, 1.7])
    declaring = _separable(lambda m, cross_density: m + 0.5 * (cross_density.sum(axis=0) - m))
    plain = _separable(lambda m: m)
    np.testing.assert_array_equal(declaring(t=0.0, x=x, p=p, m=m), plain(t=0.0, x=x, p=p, m=m))


@pytest.mark.parametrize("kind", ["separable", "congestion"])
def test_dm_reads_a_term_that_declares_the_slot_at_the_own_density(kind):
    """`dm` is read on single-population paths only, so it hands its terms the own row: a term written over every
    population there reads as the same term written for one."""
    cost = QuadraticControlCost(control_cost=1.0)
    if kind == "separable":
        declaring = SeparableHamiltonian(
            cost,
            coupling=lambda m, cross_density: 0.5 * np.asarray(cross_density).sum(axis=0) ** 2,
            coupling_dm=lambda m, cross_density: np.asarray(cross_density).sum(axis=0),
        )
        plain = SeparableHamiltonian(cost, coupling=lambda m: 0.5 * m**2, coupling_dm=lambda m: m)
    else:
        declaring = CongestionHamiltonian(
            cost,
            congestion_factor=lambda m, cross_density: 1.0 + 0.5 * np.asarray(cross_density).sum(axis=0),
            congestion_factor_dm=lambda m, cross_density: 0.5 + 0.0 * np.asarray(cross_density).sum(axis=0),
        )
        plain = CongestionHamiltonian(
            cost, congestion_factor=lambda m: 1.0 + 0.5 * m, congestion_factor_dm=lambda m: 0.5 + 0.0 * m
        )
    x, p, m = (np.linspace(0.0, 1.0, 4),), np.linspace(-1.0, 1.0, 4).reshape(-1, 1), np.array([0.5, 0.9, 1.3, 1.7])
    np.testing.assert_array_equal(declaring.dm(t=0.0, x=x, p=p, m=m), plain.dm(t=0.0, x=x, p=p, m=m))


def test_a_one_population_solve_hands_a_term_that_declares_the_slot_its_own_row():
    """With no stack handed, a door's state carries none, and the library class reads its terms at the own row: a
    congestion factor written over every population reads, at one population, as the same factor written for one. The
    iterator hands its HJB step no stack at K = 1; the FP velocity door is called with none here."""
    cost = QuadraticControlCost(control_cost=1.0)
    declaring = CongestionHamiltonian(
        cost, congestion_factor=lambda m, cross_density: 1.0 + np.asarray(cross_density).sum(axis=0)
    )
    plain = CongestionHamiltonian(cost, congestion_factor=lambda m: 1.0 + m)
    (declared,), (alone,) = _solve([declaring], [0.4]), _solve([plain], [0.4])
    np.testing.assert_array_equal(declared[0], alone[0])
    np.testing.assert_array_equal(declared[1], alone[1])
    problem = _problem(plain)
    nodes = np.linspace(0.0, 1.0, N)
    U = np.outer(np.linspace(1.0, 0.5, NT + 1), 0.7 * nodes + 0.4 * nodes**2)
    M = np.outer(np.linspace(1.0, 1.5, NT + 1), 0.2 + nodes**2)
    np.testing.assert_array_equal(
        compute_fp_velocity_field(problem, U, M, declaring), compute_fp_velocity_field(problem, U, M, plain)
    )


FP_SOLVERS = {
    "FPFDMSolver": FPFDMSolver,
    "FPFVMSolver": FPFVMSolver,
    "FPParticleSolver": functools.partial(FPParticleSolver, num_particles=1000, seed=5),
    "FPSLSolver": FPSLSolver,
    "MeshlessGalerkinFPSolver": lambda problem: MeshlessGalerkinFPSolver(
        problem, collocation_points=np.linspace(0.0, 1.0, N)[:, None], delta=3.5 / (N - 1)
    ),
}
DECLARING = {
    "separable_coupling": lambda k: _separable(_sum_coupling),
    "congestion_factor": lambda k: _congested(_crowd(k)),
}
# A non-separable H on a solver handed the value function: the route refuses it at the first FP step (#1489).
ROUTE_REFUSES = {("congestion_factor", fp) for fp in ("FPParticleSolver", "FPSLSolver", "MeshlessGalerkinFPSolver")}


@pytest.mark.parametrize("fp", sorted(FP_SOLVERS))
@pytest.mark.parametrize("declaring", sorted(DECLARING))
def test_every_term_read_inside_a_two_population_solve_is_handed_both_rows(declaring, fp, monkeypatch):
    """The closed set: each library class whose term declares the slot, on each FP solver the iterator accepts on a 1-D
    grid. Every read of a term inside the solve is handed both populations' rows; a solve the route refuses stops, its
    reads so far handed both too. FEM needs a mesh, FP-GFDM is withdrawn, and the network solver refuses a grid."""
    rows = []
    read = hamiltonian_module._read_of_the_density
    solving = [False]

    def counted(owner, name, m, cross_density, count=None):
        if solving[0]:
            rows.append(np.shape(cross_density)[0])
        return read(owner, name, m, cross_density, count)

    monkeypatch.setattr(hamiltonian_module, "_read_of_the_density", counted)
    problems = [_problem(DECLARING[declaring](k), c) for k, c in enumerate([0.25, 0.75])]
    iterator = MultiPopulationIterator(
        MultiPopulationProblem(populations=problems, population_names=["a", "b"]),
        hjb_solvers=[HJBFDMSolver(p) for p in problems],
        fp_solvers=[FP_SOLVERS[fp](p) for p in problems],
    )
    refused = (declaring, fp) in ROUTE_REFUSES
    solving[0] = True
    try:
        with (
            warnings.catch_warnings(),
            pytest.raises(ValueError, match="Cannot route the value function") if refused else contextlib.nullcontext(),
        ):
            warnings.simplefilter("ignore")
            iterator.solve(max_iterations=2, tolerance=1e-14)
    finally:
        solving[0] = False
    assert rows, "no term was read"
    assert set(rows) == {2}

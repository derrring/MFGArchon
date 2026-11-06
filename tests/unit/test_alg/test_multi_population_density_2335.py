"""In a multi-population solve `m` is the population's own density, and every population's is `cross_density` (#2335).

Before, each population's Hamiltonian was handed the stack of every population's density as `m`, and its terms paired
the stack with the points by position: every population read population 0's density, with no error, and the FP
velocity read node values at the faces. The oracle is the one-population run: with terms that read only the own
density, populations do not interact, so each must equal its own K = 1 run of the same iterator.
"""

from __future__ import annotations

import functools
import logging
import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
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
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.types.callable_protocols import FrozenSlotError, refuse_batch_disagreement
from mfgarchon.utils.validation.protocol import ValidationError

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
    velocity's faces included. Before, every population read population 0's density, and the congestion velocity read
    node densities at the faces."""
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
    the FP velocity, so a row handed in another order, or not handed, at any one of them is a difference."""

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


class _OnCallOnly(HamiltonianBase):
    """`|p|^2 / (2 (1 + m + 0.8 q)) - m`, q the other population's density, read on `__call__` alone: `dp` and
    `optimal_control` are the base class's, which hand it on."""

    def __init__(self, k):
        super().__init__()
        self.k = k

    def _other(self, m, cross_density):
        return cross_density[1 - self.k] if len(cross_density) > 1 else 0.0 * np.asarray(m, dtype=float)

    def __call__(self, t, x, p, m, cross_density):
        crowd = 1.0 + np.asarray(m) + 0.8 * self._other(m, cross_density)
        return 0.5 * np.sum(np.asarray(p) ** 2, axis=-1) / crowd - np.asarray(m)


class _ExplicitTwin(_OnCallOnly):
    """The same Hamiltonian with its derivative written out, so nothing the library hands on stands between them."""

    def dp(self, t, x, p, m, cross_density):
        crowd = 1.0 + np.asarray(m) + 0.8 * self._other(m, cross_density)
        return np.asarray(p) / np.reshape(crowd, (-1, 1)) if np.ndim(p) == 2 else np.asarray(p) / crowd

    def optimal_control(self, t, x, p, m, cross_density):
        return -self.dp(t=t, x=x, p=p, m=m, cross_density=cross_density)


def test_a_user_hamiltonian_reading_every_population_on_call_alone_is_its_explicit_twin():
    """A subclass that declares `cross_density` on `__call__` only gets it in the base class's `dp` and
    `optimal_control` too: it solves as the twin that writes them out, to the finite difference's error."""
    centres = [0.25, 0.75]
    inherited = _solve([_OnCallOnly(k) for k in range(2)], centres)
    explicit = _solve([_ExplicitTwin(k) for k in range(2)], centres)
    for k in range(2):
        np.testing.assert_allclose(inherited[k][0], explicit[k][0], rtol=0, atol=1e-6)
        np.testing.assert_allclose(inherited[k][1], explicit[k][1], rtol=0, atol=1e-6)


def test_the_velocity_door_reads_each_density_at_the_faces_as_the_mean_of_its_nodes():
    """At the FP velocity a density is read at the faces: a linear one's value there is `a + b * x_face`, which no
    node holds. Both `m` and each row of `cross_density` are read so."""
    seen = []

    class Recorder(HamiltonianBase):
        def __call__(self, t, x, p, m, cross_density):
            return 0.5 * np.sum(np.asarray(p) ** 2, axis=-1)

        def optimal_control(self, t, x, p, m, cross_density):
            seen.append((np.array(m, dtype=float), np.array(cross_density, dtype=float)))
            return -np.asarray(p, dtype=float).ravel()

    hamiltonian = Recorder()
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


class _DropsTheSlot(HamiltonianBase):
    """`__call__` reads every population; `dp` is written as before #2335 and calls it without passing the slot on."""

    def __call__(self, t, x, p, m, cross_density):
        return 0.5 * np.sum(np.asarray(p) ** 2, axis=-1) - np.asarray(cross_density).sum(axis=0)

    def dp(self, t, x, p, m):
        return self(t=t, x=x, p=p, m=m)


def test_a_method_of_yours_that_drops_the_slot_raises_naming_it():
    """The library hands a method that declares `cross_density` the slot; a method of yours that calls it without the
    slot gets Python's TypeError naming it, rather than the own density alone."""
    state = HEvalState(x=np.linspace(0.0, 1.0, 4).reshape(-1, 1), p=np.zeros((4, 1)), m=np.ones(4), t=0.0)
    with pytest.raises(TypeError, match="cross_density"):
        _DropsTheSlot().evaluate_dp(state)


# Calls that pass x, p and m by keyword but are not a Hamiltonian's family method: a conditional Hamiltonian (theta), a
# bound raw callable, the problem-level `hamiltonian` method, Howard's alpha_star, a variational problem's converted
# functions, the dual's own argmax, and helpers that go through `evaluate_H` / `evaluate_dp`.
_OTHER_FAMILIES = {
    "HEvalState",
    "assemble_hjb_residual",
    "assemble_hjb_jacobian_diag",
    "self._alpha_star",
    "self.conjugate_argmax",
    "self.problem.hamiltonian",
    "hamiltonian_funcs['hamiltonian']",
    "hamiltonian_funcs['hamiltonian_dm']",
}


def _family_calls_outside_the_owner():
    """Every call in the library with a family method's signature, by keyword, not made through `_family_call`, and
    those in `__main__` demonstrations apart."""
    import ast
    import pathlib

    import mfgarchon

    outside, demo = [], []
    for path in sorted(pathlib.Path(mfgarchon.__file__).parent.rglob("*.py")):
        tree = ast.parse(path.read_text())
        main = {
            node.lineno
            for top in tree.body
            if isinstance(top, ast.If) and "__main__" in ast.unparse(top.test)
            for node in ast.walk(top)
            if hasattr(node, "lineno")
        }
        owner = {}
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for node in ast.walk(fn):
                    owner.setdefault(getattr(node, "lineno", -1), fn.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or isinstance(node.func, ast.Call):
                continue
            names = {keyword.arg for keyword in node.keywords}
            callee = ast.unparse(node.func)
            if not {"x", "p", "m"} <= names or "theta" in names or callee in _OTHER_FAMILIES | {"_family_call"}:
                continue
            if owner.get(node.lineno) == "_family_call":
                continue
            (demo if node.lineno in main else outside).append(f"{path.name}:{node.lineno} {callee}")
    return outside, demo


def test_the_library_calls_a_family_method_only_through_its_one_owner():
    """`_family_call` is how the library calls `__call__`, `dp`, `dm`, `dx` and `optimal_control`, so a method that
    declares `cross_density` is handed it wherever the library calls it. Control: the `__main__` demonstrations, which
    call a Hamiltonian as a user does, are found."""
    outside, demo = _family_calls_outside_the_owner()
    assert outside == []
    assert demo, "the scan found no call at all"


@pytest.mark.parametrize("second", ["more_nodes", "same_count_on_another_domain"])
def test_populations_not_on_one_grid_are_refused(second):
    """`cross_density` reads every population at one population's points, row for row, so the populations of one solve
    share one grid, compared by coordinates: one node count on [0, 1] and on [0, 2] is two grids."""
    first = _problem(_separable(lambda m: m), 0.25)
    nodes, upper = (2 * N, 1.0) if second == "more_nodes" else (N, 2.0)
    other = MFGProblem(
        model=Model(hamiltonian=_separable(lambda m: m), volatility=0.3),
        domain=TensorProductGrid(bounds=[(0.0, upper)], Nx_points=[nodes], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(m_initial=lambda x: 1.0 / upper, u_terminal=lambda x: 0.0 * x[0], T=0.4),
        Nt=NT,
    )
    problems = [first, other]
    iterator = MultiPopulationIterator(
        MultiPopulationProblem(populations=problems, population_names=["a", "b"]),
        hjb_solvers=[HJBFDMSolver(p) for p in problems],
        fp_solvers=[FPFDMSolver(p) for p in problems],
    )
    with pytest.raises(ValueError, match=r"population\(s\) \[1\] sit on a grid other than population 0's"):
        iterator.solve(max_iterations=1, tolerance=1e-14)


def test_a_partial_that_froze_cross_density_is_refused_where_the_library_passes_it():
    def raw(t, x, p, m, cross_density):
        return 0.5 * np.sum(np.asarray(p) ** 2, axis=-1) - np.asarray(cross_density).sum(axis=0)

    class Frozen(HamiltonianBase):
        __call__ = functools.partial(raw, cross_density=np.zeros((1, 4)))

    state = HEvalState(x=np.linspace(0.0, 1.0, 4).reshape(-1, 1), p=np.zeros((4, 1)), m=np.ones(4), t=0.0)
    with pytest.raises(FrozenSlotError, match="freezes cross_density"):
        Frozen().evaluate_H(state)


def _sum_coupling(rows_seen):
    def coupling(m, cross_density):
        if np.size(m) >= N - 1:  # a door's batch: the grid's nodes, or its faces; the problem's own checks read fewer
            rows_seen.append(len(cross_density))
        return m + 0.5 * (np.asarray(cross_density).sum(axis=0) - m)

    return coupling


class _DropsOnOverride(SeparableHamiltonian):
    def __call__(self, t, x, p, m):
        return super().__call__(t=t, x=x, p=p, m=m)


class _ForwardsOnOverride(SeparableHamiltonian):
    def __call__(self, t, x, p, m, cross_density):
        return super().__call__(t=t, x=x, p=p, m=m, cross_density=cross_density)


def test_an_override_that_drops_the_slot_is_refused_where_a_term_reads_it():
    """A subclass of a library Hamiltonian whose override drops `cross_density`, calling the library's method through
    `super()`, would read a term that declares the slot at the own density alone, with no error: refused, at the
    problem's first read. The same subclass forwarding the slot reads every population."""
    seen = []
    # Refused at the problem's first read of it, its validation, which reports the refusal it raised.
    with pytest.raises(ValidationError, match=r"coupling declares cross_density, and was handed none"):
        _solve(
            [_DropsOnOverride(QuadraticControlCost(control_cost=1.0), coupling=_sum_coupling([])) for _ in range(2)],
            [0.25, 0.75],
        )
    _solve(
        [_ForwardsOnOverride(QuadraticControlCost(control_cost=1.0), coupling=_sum_coupling(seen)) for _ in range(2)],
        [0.25, 0.75],
    )
    assert seen
    assert set(seen) == {2}


def test_a_direct_call_passing_the_own_row_reads_as_the_one_population_body():
    """Control: a direct single-population call of a Hamiltonian whose term declares the slot passes `m[np.newaxis]`,
    and reads as the same Hamiltonian written for one population."""
    x, p, m = (np.linspace(0.0, 1.0, 4),), np.linspace(-1.0, 1.0, 4).reshape(-1, 1), np.array([0.5, 0.9, 1.3, 1.7])
    declaring = _separable(lambda m, cross_density: m + 0.5 * (cross_density.sum(axis=0) - m))
    plain = _separable(lambda m: m)
    np.testing.assert_array_equal(
        declaring(t=0.0, x=x, p=p, m=m, cross_density=m[np.newaxis]), plain(t=0.0, x=x, p=p, m=m)
    )


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


class _LegacyOptimalControl(_OnCallOnly):
    """`optimal_control` written as before #2335: it calls `dp` without passing the slot on."""

    def optimal_control(self, t, x, p, m):
        return -self.dp(t=t, x=x, p=p, m=m)


class _ForwardingOptimalControl(_OnCallOnly):
    def optimal_control(self, t, x, p, m, cross_density):
        return -self.dp(t=t, x=x, p=p, m=m, cross_density=cross_density)


def test_an_optimal_control_that_drops_the_slot_is_refused_where_it_reaches_a_declaring_call():
    """The library's `dp` handed no slot passes `None` on; the `__call__` it reaches requires the slot, so it is refused
    rather than read at the own density alone while the residual reads every population. Forwarded, the same
    Hamiltonian solves as the one whose base-class `optimal_control` hands the slot on."""
    centres = [0.25, 0.75]
    with pytest.raises(TypeError, match=r"__call__ declares cross_density, and was handed none"):
        _solve([_LegacyOptimalControl(k) for k in range(2)], centres)
    forwarded = _solve([_ForwardingOptimalControl(k) for k in range(2)], centres)
    inherited = _solve([_OnCallOnly(k) for k in range(2)], centres)
    for k in range(2):
        np.testing.assert_array_equal(forwarded[k][0], inherited[k][0])
        np.testing.assert_array_equal(forwarded[k][1], inherited[k][1])


def test_a_raw_subclass_called_directly_with_the_own_row_reads_as_its_plain_twin():
    """Control: a subclass that requires the slot, called directly at one population with `cross_density=m[np.newaxis]`,
    reads as the same Hamiltonian written without it."""

    class Plain(HamiltonianBase):
        def __call__(self, t, x, p, m):
            return 0.5 * np.sum(np.asarray(p) ** 2, axis=-1) / (1.0 + np.asarray(m)) - np.asarray(m)

    x, p, m = (np.linspace(0.0, 1.0, 4),), np.linspace(-1.0, 1.0, 4).reshape(-1, 1), np.array([0.5, 0.9, 1.3, 1.7])
    np.testing.assert_array_equal(
        _OnCallOnly(0)(t=0.0, x=x, p=p, m=m, cross_density=m[np.newaxis]), Plain()(t=0.0, x=x, p=p, m=m)
    )


def test_a_one_population_solve_hands_a_term_that_declares_the_slot_its_own_row():
    """With no stack handed, each door builds the own row itself: a congestion factor written over every population reads,
    at one population, as the same factor written for one. The iterator hands its HJB step no stack at K = 1, so the
    residual and Jacobian doors build it there; the FP velocity door builds it when it is called with none."""
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

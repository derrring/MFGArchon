"""Constructing a problem does not change the components it was given (#2702).

A construction writes into the components it holds: the soft wall's composed Hamiltonian and Lagrangian, and a
network's own Hamiltonian. It writes into its own copy (`own_copy_for_a_problem`), so the caller's object stays as
given. The composed Hamiltonian goes to `_hamiltonian_class` alone, and the public `hamiltonian` field holds the
Hamiltonian as given (user ruling, 2026-11-12). The potential is read at x = 0.05, inside a wall of scale 5, through the
Hamiltonian itself: with p = 0 and no coupling, H = -V, so one wall reads 5.

- **One components given to two walled problems carries one wall in each** (a mutation-verified pin). Before, the
  second construction composed a second wall into the object both held, and the first problem read it too.
- **A rebuild carries one wall and leaves the problem it rebuilt** (a mutation-verified pin). `_rebuilt` replays the
  inputs, which hold the caller's object.
- **A copy of a walled problem's components carries the Hamiltonian as given** (a mutation-verified pin), into a problem
  with the same wall and into one with none. Before, an H-defined copy carried the composed Hamiltonian in its public
  field.
- **A second network problem from the same components leaves the first on its own network** (a mutation-verified pin).
  Before, the second construction wrote its Hamiltonian into the object both held, and the first solved on it.
- **The components a construction was given are left as given** (the fact, stated directly): every field holds the
  object it held before.
"""

from __future__ import annotations

import dataclasses
import logging
import warnings

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.core import hamiltonian as hm
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _a(v):
    return np.asarray(v, dtype=float)


def _wall(x):
    return (_a(x[0]) < 0.15).astype(float)


_KINDS = {
    "H-defined": lambda: {"hamiltonian": hm.SeparableHamiltonian(hm.QuadraticControlCost(control_cost=1.0))},
    "SeparableLagrangian": lambda: {
        "lagrangian": hm.SeparableLagrangian(control_cost=hm.QuadraticControlCost(control_cost=1.0))
    },
}


def _components(kind):
    return MFGComponents(
        m_initial=lambda x: 1.0 + 0.0 * _a(x[0]), u_terminal=lambda x: 0.0 * _a(x[0]), **_KINDS[kind]()
    )


def _problem(components, *, walled=True):
    extra = {"state_penalty": _wall, "state_penalty_scale": 5.0} if walled else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.CRITICAL)
        try:
            grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
            return MFGProblem(geometry=grid, T=0.5, Nt=10, volatility=0.3, components=components, **extra)
        finally:
            logging.disable(logging.NOTSET)


def _potential_in_the_wall(problem):
    value = problem.hamiltonian_class(0.0, (np.array([0.05]),), np.array([0.0]), 1.0)
    return -float(np.ravel(_a(value))[0])


def _snapshot(components):
    return {name: id(value) for name, value in vars(components).items()}


@pytest.mark.parametrize("kind", list(_KINDS))
def test_one_components_given_to_two_walled_problems_carries_one_wall_in_each(kind):
    shared = _components(kind)
    first = _problem(shared)
    second = _problem(shared)
    assert _potential_in_the_wall(second) == pytest.approx(5.0)
    assert _potential_in_the_wall(first) == pytest.approx(5.0)


@pytest.mark.parametrize("kind", list(_KINDS))
def test_a_rebuild_carries_one_wall_and_leaves_the_problem_it_rebuilt(kind):
    problem = _problem(_components(kind))
    rebuilt = problem._rebuilt()
    assert _potential_in_the_wall(rebuilt) == pytest.approx(5.0)
    assert _potential_in_the_wall(problem) == pytest.approx(5.0)


@pytest.mark.parametrize(("walled", "expected"), [(True, 5.0), (False, 0.0)], ids=["the same wall", "no wall"])
@pytest.mark.parametrize("kind", list(_KINDS))
def test_a_copy_of_a_walled_problems_components_carries_the_hamiltonian_as_given(kind, walled, expected):
    given = _components(kind)
    problem = _problem(given)
    assert problem.components.hamiltonian is given.hamiltonian
    copied = _problem(dataclasses.replace(problem.components, description="copied"), walled=walled)
    assert _potential_in_the_wall(copied) == pytest.approx(expected, abs=1e-12)


def _network(width, height):
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    network = GridNetwork(width=width, height=height)
    network.create_network()
    return network


def _network_components():
    from mfgarchon.extensions.topology import NetworkMFGComponents

    return NetworkMFGComponents(
        initial_node_density_func=lambda t, node: 1.0 + 0.5 * node, terminal_node_value_func=lambda t, node: 0.2 * node
    )


def _network_problem(network, components):
    from mfgarchon.extensions.topology import NetworkMFGProblem

    return NetworkMFGProblem(geometry=network, T=0.5, Nt=4, components=components)


def test_a_second_network_problem_from_the_same_components_leaves_the_first_on_its_own_network():
    from mfgarchon.alg.numerical.network_solvers.hjb_network import NetworkHJBSolver

    def value(problem):
        return np.asarray(
            NetworkHJBSolver(problem).solve_hjb_system(
                M_density=np.ones((5, 4)) / 4, U_terminal=0.2 * np.arange(4, dtype=float)
            ),
            dtype=float,
        )

    shared = _network_components()
    first = _network_problem(_network(2, 2), shared)
    before = value(first)
    _network_problem(_network(4, 1), shared)
    assert first.hamiltonian_class.network_data.num_nodes == 4
    np.testing.assert_array_equal(value(first), before)
    assert first.hamiltonian_class.network_data is first.network_data


@pytest.mark.parametrize("construction", ["a walled problem", "a network problem"])
def test_the_components_a_construction_was_given_are_left_as_given(construction):
    if construction == "a walled problem":
        given = _components("H-defined")
        before = _snapshot(given)
        _problem(given)
    else:
        given = _network_components()
        before = _snapshot(given)
        _network_problem(_network(2, 2), given)
    assert _snapshot(given) == before

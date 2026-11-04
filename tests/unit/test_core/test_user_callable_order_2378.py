"""User callables take ruling 8's order, time first (#2375 ruling 8, #2378 phase 5 part 2).

A user's ``lambda x, t`` still runs if the library calls it in the new order and returns wrong
numbers, since both arguments are numeric. The library therefore inspects each callable when it
accepts it (``bind_user_callable``), refuses the old order there, and passes the slots by name.
"""

from __future__ import annotations

import copy

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.measure_field import FunctionalMeasureField
from mfgarchon.types.callable_protocols import (
    SOURCE_TERM_SLOTS,
    bind_user_callable,
)

_X = (0.3,)
_ARGS = {"t": 0.5, "x": _X, "p": np.array([0.0]), "m": 0.0}


def _hamiltonian(potential):
    return SeparableHamiltonian(control_cost=QuadraticControlCost(), potential=potential)


def test_a_new_order_potential_constructs_and_computes_the_expected_value():
    # H = |p|^2/2 - V with p = 0, so H = -V(t, x) = -(0.09 + 0.5).
    H = _hamiltonian(lambda t, x: x[0] ** 2 + t)
    assert H(**_ARGS) == pytest.approx(-0.59, abs=1e-15)


def test_named_slots_are_passed_by_name_and_an_omitted_one_is_not_passed():
    seen = {}

    def source(t, x, m):  # takes no v
        seen.update(t=t, x=x, m=m)
        return 0.0

    bind_user_callable(source, SOURCE_TERM_SLOTS, role="source_term_hjb")(t=1.0, x=(2.0,), v=3.0, m=4.0)
    assert seen == {"t": 1.0, "x": (2.0,), "m": 4.0}


def test_a_potential_replaced_on_a_copy_is_the_one_evaluated():
    """The binding follows the attribute: MFGProblem composes a soft wall by replacing a copy's
    `_potential`, and a binding cached at construction would keep evaluating the original."""
    H = _hamiltonian(lambda t, x: 1.0)
    replaced = copy.copy(H)
    replaced._potential = lambda t, x: 7.0
    assert (H(**_ARGS), replaced(**_ARGS)) == (-1.0, -7.0)


def test_a_measure_field_reads_x_as_a_tuple_from_a_query_of_points():
    field = FunctionalMeasureField(lambda t, x, mu: t + x[0])
    np.testing.assert_array_equal(field.evaluate(t=0.5, x=np.array([0.3]), mu=None), 0.5 + np.array([0.3]))


# ---------------------------------------------------------------------------
# Part 3a: the Hamiltonian signature on objects other than the family
# ---------------------------------------------------------------------------


def test_a_problem_subclass_defining_an_old_order_hamiltonian_is_refused_at_class_creation():
    from mfgarchon.core.mfg_problem import MFGProblem

    with pytest.raises(TypeError, match=r"_OldProblem\.hamiltonian takes \(x, m, p, t\), which is out of order"):

        class _OldProblem(MFGProblem):
            def hamiltonian(self, x, m, p, t):
                return 0.0

    with pytest.raises(TypeError, match=r"_OldCost\.running_cost takes \(x, m, t\), which is out of order"):

        class _OldCost(MFGProblem):
            def running_cost(self, x, m, t):
                return 0.0

    class _NewProblem(MFGProblem):
        def hamiltonian(self, t, x, p, m):
            return 0.0

        def running_cost(self, t, x, m):
            return 0.0


def test_the_extensions_register_their_own_method_orders():
    """Each problem class declares the methods it owns; the tables merge along the MRO."""
    from mfgarchon.core.stochastic.stochastic_problem import StochasticMFGProblem
    from mfgarchon.extensions.multi_population import MultiPopulationMFGProblem
    from mfgarchon.extensions.topology import NetworkMFGProblem

    with pytest.raises(TypeError, match=r"hamiltonian_k takes \(k, x, m_all, p, t\)"):

        class _OldPopulations(MultiPopulationMFGProblem):
            def hamiltonian_k(self, k, x, m_all, p, t):
                return 0.0

    with pytest.raises(TypeError, match=r"terminal_cost_k takes \(k, x\)"):

        class _OldTerminal(MultiPopulationMFGProblem):
            def terminal_cost_k(self, k, x):
                return 0.0

    with pytest.raises(TypeError, match=r"H_conditional takes \(x, p, m, theta, t\)"):

        class _OldConditional(StochasticMFGProblem):
            def H_conditional(self, x, p, m, theta, t):
                return 0.0

    # The network's own order: the node for x, its adjacency beside it (part 3b, ruling 2026-09-25).
    with pytest.raises(TypeError, match=r"_OldNetwork\.hamiltonian takes \(node, neighbors, m, p, t\)"):

        class _OldNetwork(NetworkMFGProblem):
            def hamiltonian(self, node, neighbors, m, p, t):
                return 0.0

    class _Network(NetworkMFGProblem):
        def hamiltonian(self, t, node, neighbors, p, m):
            return 0.0


@pytest.mark.parametrize(
    ("callable_", "expected"),
    [
        (lambda t, x, p, m, theta: (t, theta), (0.5, 0.9)),  # the new order, by name
        (lambda x, p, m, theta: theta, 0.9),  # no time, by name: the form every caller used
        (lambda a, b, c, d: d, 0.9),  # no time, unnamed: t is the optional slot, the rest in order
    ],
)
def test_a_conditional_hamiltonian_may_leave_out_time(callable_, expected):
    from mfgarchon.types.callable_protocols import CONDITIONAL_HAMILTONIAN_SLOTS

    bound = bind_user_callable(callable_, CONDITIONAL_HAMILTONIAN_SLOTS, role="conditional_hamiltonian")
    assert bound(t=0.5, x=0.1, p=0.2, m=0.3, theta=0.9) == expected
    with pytest.raises(TypeError, match=r"out of order"):
        bind_user_callable(lambda x, p, m, theta, t: 0.0, CONDITIONAL_HAMILTONIAN_SLOTS, role="c")


# ---------------------------------------------------------------------------
# Part 3b: the network callables (node for x, its adjacency beside it; ruling 2026-09-25)
# ---------------------------------------------------------------------------


def test_a_network_hamiltonian_in_the_new_order_receives_each_slot():
    from mfgarchon.types.callable_protocols import NETWORK_HAMILTONIAN_SLOTS

    values = {"t": 0.5, "node": 2, "neighbors": [1, 3], "p": "P", "m": "M"}
    for new_order in (
        lambda t, node, neighbors, p, m: (t, node, neighbors, p, m),
        lambda t, n, nb, p, m: (t, n, nb, p, m),
    ):
        bound = bind_user_callable(new_order, NETWORK_HAMILTONIAN_SLOTS, role="hamiltonian_func")
        assert bound(**values) == (0.5, 2, [1, 3], "P", "M")


def test_node_callables_take_time_first():
    from mfgarchon.types.callable_protocols import NODE_INTERACTION_SLOTS, NODE_POTENTIAL_SLOTS

    assert bind_user_callable(lambda t, n: (t, n), NODE_POTENTIAL_SLOTS, role="node_potential_func")(t=0.5, node=3) == (
        0.5,
        3,
    )
    assert (
        bind_user_callable(lambda t, n, m: m, NODE_INTERACTION_SLOTS, role="node_interaction_func")(
            t=0.5, node=3, m="M"
        )
        == "M"
    )
    for old, slots in ((lambda n, t: 0.0, NODE_POTENTIAL_SLOTS), (lambda n, m, t: 0.0, NODE_INTERACTION_SLOTS)):
        with pytest.raises(TypeError, match=r"out of order"):
            bind_user_callable(old, slots, role="node callable")

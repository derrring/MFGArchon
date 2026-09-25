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
    MEASURE_FIELD_SLOTS,
    POTENTIAL_SLOTS,
    SOURCE_TERM_SLOTS,
    bind_user_callable,
)
from mfgarchon.utils.validation import detect_callable_signature

_X = np.array([0.3])
_ARGS = {"t": 0.5, "x": _X, "p": np.array([0.0]), "m": 0.0}


def _hamiltonian(potential):
    return SeparableHamiltonian(control_cost=QuadraticControlCost(), potential=potential)


def test_an_old_order_potential_is_refused_at_construction():
    with pytest.raises(TypeError, match=r"out of order.*#2375 ruling 8.*\(t, x\).*by keyword"):
        _hamiltonian(lambda x, t: float(np.sum(x**2)) + t)


def test_a_new_order_potential_constructs_and_computes_the_expected_value():
    # H = |p|^2/2 - V with p = 0, so H = -V(t, x) = -(0.09 + 0.5).
    H = _hamiltonian(lambda t, x: float(np.sum(x**2)) + t)
    assert H(**_ARGS) == pytest.approx(-0.59, abs=1e-15)


@pytest.mark.parametrize(
    ("slots", "old_order"),
    [
        (POTENTIAL_SLOTS, lambda x, time: 0.0),  # a time parameter named `time`
        (POTENTIAL_SLOTS, lambda xx, t: 0.0),  # x not named x: t second is still the old order
        (SOURCE_TERM_SLOTS, lambda x, m, v, t: 0.0),
        (SOURCE_TERM_SLOTS, lambda t, x, m, v: 0.0),  # the half-migration: t moved, m and v not
        (MEASURE_FIELD_SLOTS, lambda x, mu, t: 0.0),
    ],
)
def test_every_old_order_shape_is_refused(slots, old_order):
    with pytest.raises(TypeError, match=r"out of order"):
        bind_user_callable(old_order, slots, role="callable")


def test_named_slots_are_passed_by_name_and_an_omitted_one_is_not_passed():
    seen = {}

    def source(t, x, m):  # takes no v
        seen.update(t=t, x=x, m=m)
        return 0.0

    bind_user_callable(source, SOURCE_TERM_SLOTS, role="source_term_hjb")(t=1.0, x=2.0, v=3.0, m=4.0)
    assert seen == {"t": 1.0, "x": 2.0, "m": 4.0}


def test_parameters_bind_positionally_only_when_one_names_the_order():
    """#2404 review: a list naming no slot whose position ruling 8 changed reads the same in either
    order, so it is refused; one such name at its new index settles which order it is in."""
    received = []
    bound = bind_user_callable(lambda t, pos: received.append((t, pos)), POTENTIAL_SLOTS, role="potential")
    bound(t=0.25, x=_X)
    assert received == [(0.25, _X)]
    for unnamed in (lambda a, b: 0.0, lambda pos, s: 0.0):
        with pytest.raises(TypeError, match=r"read the same in the old order and the new"):
            bind_user_callable(unnamed, POTENTIAL_SLOTS, role="potential")


def test_a_one_argument_spatial_potential_func_receives_x_only():
    received = []
    bound = bind_user_callable(received.append, POTENTIAL_SLOTS, role="potential_func", spatial_only=True)
    bound(t=0.25, x=0.75)
    assert received == [0.75]


def test_a_signature_that_cannot_be_matched_is_refused():
    with pytest.raises(TypeError, match=r"cannot be matched"):
        bind_user_callable(lambda a, b, c: 0.0, POTENTIAL_SLOTS, role="potential")


def test_a_potential_replaced_on_a_copy_is_the_one_evaluated():
    """The binding follows the attribute: MFGProblem composes a soft wall by replacing a copy's
    `_potential`, and a binding cached at construction would keep evaluating the original."""
    H = _hamiltonian(lambda t, x: 1.0)
    replaced = copy.copy(H)
    replaced._potential = lambda t, x: 7.0
    assert (H(**_ARGS), replaced(**_ARGS)) == (-1.0, -7.0)


def test_a_measure_field_takes_time_first_and_refuses_the_old_order():
    field = FunctionalMeasureField(lambda t, x, mu: t + x)
    np.testing.assert_array_equal(field.evaluate(t=0.5, x=_X, mu=None), 0.5 + _X)
    with pytest.raises(TypeError, match=r"out of order"):
        FunctionalMeasureField(lambda x, mu, t: x)


def test_detect_callable_signature_tells_the_two_orders_apart():
    new = detect_callable_signature(lambda t, x: 0.0)
    old = detect_callable_signature(lambda x, t: 0.0)
    assert (new.context["signature_type"], new.is_valid) == ("spatiotemporal", True)
    assert (old.context["signature_type"], old.is_valid) == ("spatiotemporal_old_order", False)


@pytest.mark.parametrize(
    ("slots", "callable_"),
    [
        (POTENTIAL_SLOTS, lambda x, tau: 0.0),  # positionally, x would receive t
        (SOURCE_TERM_SLOTS, lambda t, x, m, v_t: 0.0),  # m would receive v
        (SOURCE_TERM_SLOTS, lambda t, x, dens, v: 0.0),  # v would receive m
        (SOURCE_TERM_SLOTS, lambda t, x, m_t, v_t: 0.0),  # the documented names, half-migrated
        (SOURCE_TERM_SLOTS, lambda x_, m_, v, t_: 0.0),  # v is at index 2 in both orders: it pins nothing
        (SOURCE_TERM_SLOTS, lambda a, b, c, d: 0.0),  # nothing says which order
        # The half-migration, t moved to the front and v, m left in the old order: t names the order
        # against the full old one, not against this, so one of v and m must be named (#2404 round 2).
        (SOURCE_TERM_SLOTS, lambda t, x, a, b: 0.0),
        (SOURCE_TERM_SLOTS, lambda t, pos, dens, val: 0.0),
        (SOURCE_TERM_SLOTS, lambda time, pos, dens, val: 0.0),
        (POTENTIAL_SLOTS, lambda a, b=1.0: 0.0),  # a default is not a slot to fill
    ],
)
def test_a_parameter_that_positional_binding_would_misfill_is_refused(slots, callable_):
    """#2402 review: a slot-named parameter at another slot's index was bound to that other slot."""
    with pytest.raises(TypeError, match=r"out of order|cannot be matched"):
        bind_user_callable(callable_, slots, role="callable")


def test_the_source_term_names_the_library_documented_are_bound_by_name():
    bound = bind_user_callable(lambda t, x, v_t, m_t: (v_t, m_t), SOURCE_TERM_SLOTS, role="source_term_hjb")
    assert bound(t=0.0, x=_X, v="v", m="m") == ("v", "m")


def test_a_ufunc_is_a_spatial_potential_func_and_its_out_parameter_is_left_alone():
    """#2402 review: `np.cos` has (x, /, out=None, ...); filling out with a slot wrote into x."""
    x = np.array([0.5])
    bound = bind_user_callable(np.cos, POTENTIAL_SLOTS, role="potential_func", spatial_only=True)
    np.testing.assert_array_equal(bound(t=np.array([9.0]), x=x), np.cos([0.5]))
    np.testing.assert_array_equal(x, [0.5])


def test_time_names_the_t_slot_for_every_role():
    """#2402 review round 2: `time` counted as a time name in the order check but was not bound as t."""
    seen = {}

    def source(time, x, v, m):
        seen.update(time=time, v=v, m=m)
        return 0.0

    bind_user_callable(source, SOURCE_TERM_SLOTS, role="source_term_hjb")(t=1.0, x=_X, v=3.0, m=4.0)
    only_time = bind_user_callable(lambda time: time, POTENTIAL_SLOTS, role="potential_func", spatial_only=True)
    assert (seen, only_time(t=1.0, x=0.75)) == ({"time": 1.0, "v": 3.0, "m": 4.0}, 1.0)


@pytest.mark.parametrize(
    "star_args", [lambda *a: float(a[0]) ** 2, np.vectorize(lambda x: float(x) ** 2)], ids=["lambda", "vectorize"]
)
def test_star_args_is_a_spatial_potential_func_and_refused_where_order_matters(star_args):
    """#2402 review round 2: `*args` said nothing about its order and was bound positionally as (t, x),
    so a spatial `potential_func` that received x on main received t."""
    bound = bind_user_callable(star_args, POTENTIAL_SLOTS, role="potential_func", spatial_only=True)
    assert bound(t=4.0, x=0.1) == pytest.approx(0.01)
    with pytest.raises(TypeError, match=r"cannot be matched"):
        bind_user_callable(star_args, POTENTIAL_SLOTS, role="potential")


def test_a_defaulted_single_parameter_is_still_a_spatial_potential_func():
    bound = bind_user_callable(lambda pos=0.0: pos, POTENTIAL_SLOTS, role="potential_func", spatial_only=True)
    assert bound(t=4.0, x=0.1) == 0.1


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


def test_alpha_star_takes_time_first_with_t_idx_for_t():
    from mfgarchon.types.callable_protocols import ALPHA_STAR_SLOTS

    bound = bind_user_callable(lambda t_idx, x, p, m: (t_idx, p), ALPHA_STAR_SLOTS, role="alpha_star")
    assert bound(t=3, x=_X, p=0.2, m=0.3) == (3, 0.2)
    # `t_idx` counts as t, so the old order is refused under that name too. Without the alias this
    # one binds positionally and receives t as x -- the shape hjb_gfdm's own alpha_star had.
    for old_order in (lambda x, p, m, t: -p, lambda x, p, m, t_idx: -p):
        with pytest.raises(TypeError, match=r"out of order"):
            bind_user_callable(old_order, ALPHA_STAR_SLOTS, role="alpha_star")


def test_a_raw_hamiltonian_must_declare_all_four_slots_and_name_one_that_moved():
    """Validation has always demanded (t, x, p, m) of a raw callable; ruling 8 moved t, x and m."""
    from mfgarchon.types.callable_protocols import HAMILTONIAN_SLOTS

    by_name = bind_user_callable(lambda t, x, p, m: p - m, HAMILTONIAN_SLOTS, role="h")
    pinned = bind_user_callable(lambda t, x, grad, m: grad - m, HAMILTONIAN_SLOTS, role="h")  # m pins the pair
    assert by_name(t=0.0, x=0.0, p=2.0, m=0.5) == pinned(t=0.0, x=0.0, p=2.0, m=0.5) == 1.5
    # Omits slots; or names only p, which is at index 2 in both orders (#2404 review).
    for incomplete in (
        lambda x: 0.0,
        lambda x_, m_, p, t_: 0.0,
        lambda t, x, a, b: 0.0,
        lambda t, pos, dens, grad: 0.0,
    ):
        with pytest.raises(TypeError, match=r"cannot be matched"):
            bind_user_callable(incomplete, HAMILTONIAN_SLOTS, role="h")


def test_a_source_term_naming_m_binds_its_other_parameter_positionally():
    """m moved in ruling 8's reorder, so naming it at its new index settles the order."""
    bound = bind_user_callable(lambda t, x, val, m: (val, m), SOURCE_TERM_SLOTS, role="source_term_hjb")
    assert bound(t=0.0, x=_X, v="v", m="m") == ("v", "m")


def test_the_deprecated_hamiltonian_adapter_redirects_to_the_binder():
    """Deprecation policy (AGENTS.md): the old API calls the new one, and an equivalence test pins it."""
    from mfgarchon.types.callable_protocols import HAMILTONIAN_SLOTS
    from mfgarchon.utils import HamiltonianAdapter, adapt_hamiltonian

    def H(t, x, p, m):
        return 0.5 * p**2 + m + t

    new = bind_user_callable(H, HAMILTONIAN_SLOTS, role="hamiltonian")(t=0.25, x=1.0, p=2.0, m=0.5)
    with pytest.warns(DeprecationWarning, match="HamiltonianAdapter|adapt_hamiltonian"):
        adapter = HamiltonianAdapter(H)
    with pytest.warns(DeprecationWarning, match="HamiltonianAdapter|adapt_hamiltonian"):
        one_shot = adapt_hamiltonian(H, x=1.0, m=0.5, p=2.0, t=0.25)
    assert adapter(x=1.0, m=0.5, p=2.0, t=0.25) == one_shot == new == 2.75
    with (
        pytest.warns(DeprecationWarning, match="HamiltonianAdapter|adapt_hamiltonian"),
        pytest.raises(TypeError, match=r"signature_hint"),
    ):
        HamiltonianAdapter(H, signature_hint="legacy")


# ---------------------------------------------------------------------------
# Part 3b: the network callables (node for x, its adjacency beside it; ruling 2026-09-25)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "refused",
    [
        lambda node, neighbors, m, p, t: 0.0,  # the old order
        lambda t, node, neighbors, m, p: 0.0,  # the half-migration, named
        lambda t, n, nb, a, b: 0.0,  # the half-migration, unnamed: t alone cannot tell it from the new order
    ],
)
def test_a_network_hamiltonian_in_an_old_order_is_refused(refused):
    from mfgarchon.types.callable_protocols import NETWORK_HAMILTONIAN_SLOTS

    with pytest.raises(TypeError, match=r"out of order|cannot be matched"):
        bind_user_callable(refused, NETWORK_HAMILTONIAN_SLOTS, role="hamiltonian_func")


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

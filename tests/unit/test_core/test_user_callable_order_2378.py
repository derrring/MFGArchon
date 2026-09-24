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


def test_unnamed_parameters_receive_the_slots_positionally_in_the_new_order():
    received = []
    bound = bind_user_callable(lambda a, b: received.append((a, b)), POTENTIAL_SLOTS, role="potential")
    bound(t=0.25, x=_X)
    assert received == [(0.25, _X)]


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
        (SOURCE_TERM_SLOTS, lambda t, x, a, b: 0.0),  # m and v swapped places: names are required
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

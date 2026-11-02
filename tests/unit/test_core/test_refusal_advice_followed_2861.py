"""A refusal's advice, followed as it says on a callable written in another order, reads right (#2861).

Each row reads the advice from the message, follows every arm it offers, and reads the result. A rename in place keeps
the order the callable is written in, so it reads a callable written another way swapped, with no error. Here
f = 100 t + x at t = 0.25, x = 3: 28.0 read right, 300.25 swapped.
"""

from __future__ import annotations

import functools
import re

import pytest

import numpy as np

from mfgarchon.geometry.boundary import applicator_graph as graph
from mfgarchon.geometry.boundary.types import bind_boundary_value
from mfgarchon.types.callable_protocols import CONDITION_SLOTS, SOLVER_SOURCE_SLOTS, bind_user_callable
from mfgarchon.utils.callable_adapter import adapt_ic_callable

_COEF = {"t": 100.0, "x": 1.0, "node": 1.0, "field": 10000.0, "extra": 0.0}
_AT = {"t": 0.25, "x": 3.0, "node": 3.0, "field": 7.0}


def _written(names, reads, star="", default=()):
    """A function of ``names``, in that order, each reading the slot at its position in ``reads``."""
    body = " + ".join(f"{_COEF[slot]}*{name}" for name, slot in zip(names, reads, strict=True))
    parameters = [f"{name}=0.0" if name in default else name for name in names]
    return eval(f"lambda {', '.join([*parameters, *([star] if star else [])])}: {body}")


def _arms(refusal, names, reads):
    """Each callable the refusal advises. By what each parameter reads and in slot order, a parameter takes the slot it is
    named for. Named in place, a parameter keeps what it read at its position, and what the names leave over is defaulted."""
    arms = []
    in_order = re.search(r"put them in the order \(([^)]*)\)", refusal)
    if in_order:
        order = in_order.group(1).split(", ")
        arms.append(_written(order, order))
    in_place = re.search(
        r"Name its parameters ([a-z, ]+?)(?:, time first|, or take)|It takes \(([^)]*)\); name its|names them ([a-z, ]+), time first",
        refusal,
    )
    if in_place:
        renamed = next(group for group in in_place.groups() if group).split(", ")
        rest = list(names[len(renamed) :])
        arms.append(_written([*renamed, *rest], reads, default=rest))
    return arms


@pytest.mark.parametrize(
    ("read", "names", "reads", "star"),
    [
        (lambda f: bind_user_callable(f, CONDITION_SLOTS, role="r"), ["a", "b"], ["x", "t"], "*rest"),
        (lambda f: bind_user_callable(f, SOLVER_SOURCE_SLOTS, role="r"), ["x", "b"], ["x", "t"], "*rest"),
        (lambda f: bind_user_callable(f, SOLVER_SOURCE_SLOTS, role="r"), ["a", "b", "c"], ["x", "t", "extra"], ""),
        (lambda f: graph._rule_reader(f, "r"), ["t", "a", "b"], ["t", "field", "node"], ""),
    ],
    ids=[
        "a reordered role, *rest",
        "never reordered, a misplaced name",
        "never reordered, a wrong count",
        "not all named",
    ],
)
def test_every_arm_of_a_binders_advice_reads_right(read, names, reads, star):
    with pytest.raises(TypeError) as refused:
        read(_written(names, reads, star))
    arms = _arms(str(refused.value), names, reads)
    assert arms, str(refused.value)
    for arm in arms:
        slots = [slot for slot in reads if slot != "extra"]
        assert read(arm)(**{slot: _AT[slot] for slot in slots}) == sum(_COEF[slot] * _AT[slot] for slot in slots)


def _positional_only(t, x, /):
    return 100.0 * t + x


@pytest.mark.parametrize(
    ("read", "refused"),
    [
        (bind_boundary_value, np.vectorize(lambda s: 100.0 * s)),
        (bind_boundary_value, _positional_only),
        (bind_boundary_value, functools.partial(lambda x, t: 100.0 * t + x, t=0.25)),
        (lambda f: graph._value_reader(f, "r"), np.vectorize(lambda node, gain=2.0: gain * node)),
        (lambda f: graph._rule_reader(f, "r"), np.vectorize(lambda t, node, field: 0.0)),
    ],
    ids=[
        "grid, another callable",
        "grid, positional-only",
        "grid, a slot frozen by keyword",
        "graph value",
        "graph rule",
    ],
)
def test_a_callable_that_cannot_be_bound_by_name_is_told_in_words_not_with_a_lambda(read, refused):
    """Each lambda these refusals used to write read some callable wrong with no error: ``lambda t: f(t)`` a function of
    space, ``lambda t, node: f(node, t)`` one with a defaulted parameter, which t landed in, and "Freeze it by position"
    froze the first parameter, whichever it was."""
    with pytest.raises(TypeError) as excinfo:
        read(refused)
    message = str(excinfo.value)
    assert "that passes each argument to the parameter that reads it" in message, message
    assert "`lambda" not in message, message
    assert "Freeze it by position" not in message, message


@pytest.mark.parametrize(
    "written",
    [
        lambda t, x: 100.0 * t + x,
        lambda x, t: 100.0 * t + x,
        lambda time, x: 100.0 * time + x,
        lambda x, time: 100.0 * time + x,
    ],
    ids=["t,x", "x,t", "time,x", "x,time"],
)
def test_the_example_for_a_partial_fixing_time_reads_right_in_either_order(written):
    """It was ``lambda x: f(0.25, x)``, whatever order f is written in. The example holds for an f that names x and the
    time, and takes the time the caller wants: frozen at 0.7 and read at T = 0.25, a value copied from the partial, or a
    slot left to override it, would not read 28.0."""
    time_name = "time" if "time" in written.__code__.co_varnames else "t"
    with pytest.raises(TypeError) as refused:
        adapt_ic_callable(functools.partial(written, **{time_name: 0.7}), 1, 3.0, role="m_initial")
    example = re.search(r"such as `(lambda x: f\(.*?\))` for an f that names them x and (\w+)", str(refused.value))
    assert example.group(2) == time_name, str(refused.value)
    _, read = adapt_ic_callable(eval(example.group(1), {"f": written, "T": 0.25}), 1, 3.0, role="m_initial")
    assert read(3.0) == 28.0


@pytest.mark.parametrize(
    "refused",
    [functools.partial(lambda t, x: 100.0 * t + x, t=0.25), lambda t: t, lambda *c, **k: c[1]],
    ids=["a partial fixing time", "time and no space", "names nothing"],
)
def test_no_refusal_says_a_signature_that_is_false_of_a_potential(refused):
    """``PotentialCost`` reads its ``V`` through this adapter, and a ``V`` that takes (t, x) is refused."""
    with pytest.raises(TypeError) as excinfo:
        adapt_ic_callable(refused, 1, 3.0, role="V")
    assert not re.search(r"\(t, x\)(, time first,)? or \(x\)", str(excinfo.value)), str(excinfo.value)

"""A refusal's advice, followed as it says on a callable written in another order, reads right (#2861).

Each row reads the advice from the message, follows every arm it offers, and reads the result. A rename in place keeps
the order the callable is written in, and a freeze by position freezes the first parameter, so either reads a callable
written another way swapped, with no error. Here f = 100 t + x at t = 0.25, x = 3: 28.0 read right, 300.25 swapped.
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


@pytest.mark.parametrize(
    ("read", "names", "frozen"),
    [
        (bind_boundary_value, ["x", "t"], "t"),
        (bind_boundary_value, ["t", "x"], "x"),
        (lambda f: graph._value_reader(f, "r"), ["t", "node"], "node"),
        (lambda f: graph._rule_reader(f, "r"), ["node", "t", "field"], "t"),
    ],
    ids=["grid, x then t", "grid, t then x", "graph value", "graph rule"],
)
def test_the_advice_for_a_slot_frozen_by_keyword_reads_right(read, names, frozen):
    """It said "Freeze it by position", which freezes the first parameter, whichever it is."""
    f = _written(names, names)
    with pytest.raises(TypeError) as refused:
        read(functools.partial(f, **{frozen: _AT[frozen]}))
    message = str(refused.value)
    want = sum(_COEF[slot] * _AT[slot] for slot in names)
    if "Freeze it by position" in message:
        kept = read(functools.partial(f, _AT[frozen]))
        assert kept(**{slot: _AT[slot] for slot in kept.passes}) == want
    wrapped = re.search(r"`(lambda [^`]*: f\([^`]*\))`", message)
    assert wrapped, message
    bound = read(eval(wrapped.group(1), {"f": f}))
    assert bound(**{slot: _AT[slot] for slot in names}) == want


@pytest.mark.parametrize(
    ("f", "wants", "reads"),
    [
        (np.vectorize(lambda s: 100.0 * s), 25.0, "time"),
        (np.vectorize(lambda s: 100.0 * s), 300.0, "space"),
        (functools.lru_cache(lambda s: 100.0 * s), 25.0, "time"),
    ],
    ids=["vectorized, of time", "vectorized, of space", "cached, of time"],
)
def test_the_wrap_for_another_callable_reads_what_it_reads(f, wants, reads):
    """It offered only ``lambda t: f(t)``, which reads time into a function of space."""
    with pytest.raises(TypeError) as refused:
        bind_boundary_value(f)
    named = re.search(rf"`(lambda \w+: f\([^`]*\))` for one that reads {reads}", str(refused.value))
    wrap = named.group(1) if named else re.search(r"`(lambda \w+: f\([^`]*\))`", str(refused.value)).group(1)
    got = bind_boundary_value(eval(wrap, {"f": f}))(t=0.25, x=np.array([3.0]))
    assert float(np.ravel(got)[0]) == wants


@pytest.mark.parametrize(
    "written",
    [
        lambda t, x: 100.0 * t + x,
        lambda x, t: 100.0 * t + x,
        lambda time, x: 100.0 * time + x,
        lambda x, time: 100.0 * time + x,
        lambda y, t: 100.0 * t + y,
        lambda t, x=0.0: 100.0 * t + x,
        lambda x=0.0, t=1.0: 100.0 * t + x,
        lambda t, x, c=0.0: 100.0 * t + x + c,
        lambda x, *, t: 100.0 * t + x,
        lambda x, **kw: 100.0 * kw["t"] + x,
    ],
    ids=["t,x", "x,t", "time,x", "x,time", "y,t", "t,x=0", "x=0,t=1", "t,x,c=0", "x,*,t", "x,**kw"],
)
def test_the_advice_for_a_partial_fixing_time_reads_right_in_either_order(written):
    """The advice was ``lambda x: f(0.25, x)``, whatever order f is written in."""
    time_name = "time" if "time" in written.__code__.co_varnames else "t"
    with pytest.raises(TypeError) as refused:
        adapt_ic_callable(functools.partial(written, **{time_name: 0.25}), 1, 3.0, role="m_initial")
    advice = re.search(r"pass (lambda x: f\(.*?\)) to fix the time yourself", str(refused.value)).group(1)
    _, read = adapt_ic_callable(eval(advice, {"f": written}), 1, 3.0, role="m_initial")
    assert read(3.0) == 28.0


def _positional_only(x, /, t):
    return 100.0 * t + x


@pytest.mark.parametrize(
    "partial",
    [
        functools.partial(max, t=0.25),
        functools.partial(np.add, t=0.25),
        functools.partial(lambda t, x, c: 100.0 * t + x + c, t=0.25),
        functools.partial(lambda t, c, x=0.0: 100.0 * t + x, t=0.25),
        functools.partial(lambda c, t, x=0.0: 100.0 * t + x, t=0.25),
        functools.partial(lambda t, time: 100.0 * time + t, time=0.25),
        functools.partial(lambda c, x, t: 100.0 * t + x + c, 1.0, t=0.25),
        functools.partial(lambda t, x, c: 100.0 * t + x + c, t=0.25, c=1.0),
        functools.partial(lambda t: t, t=0.25),
        functools.partial(_positional_only, t=0.25),
    ],
    ids=[
        "builtin",
        "ufunc",
        "two required left",
        "a required parameter beside a defaulted x",
        "the same, written first",
        "a time name left",
        "a parameter frozen by position",
        "a second keyword frozen",
        "no space",
        "positional only",
    ],
)
def test_the_advice_writes_no_lambda_where_one_would_not_read_right(partial):
    """Where f's names cannot be read, a lambda written from the slots could swap. Where something else is left or frozen,
    it would raise, or pass space to a parameter that does not read it."""
    with pytest.raises(TypeError, match="fixes the time itself, passing each argument") as refused:
        adapt_ic_callable(partial, 1, 3.0, role="m_initial")
    assert "lambda" not in str(refused.value)

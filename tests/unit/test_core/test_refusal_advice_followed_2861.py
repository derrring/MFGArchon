"""A refusal's advice, followed as it says on a callable written in another order, reads right (#2861).

Each row reads the advice from the message and does what it says. A rename in place keeps the order the callable is
written in, so advice to name the parameters ``t, x`` reads a callable written ``(x, t)`` swapped, with no error. Here
f = 100 t + x at t = 0.25, x = 3: 28.0 read right, 300.25 read swapped.
"""

from __future__ import annotations

import functools
import re

import pytest

import numpy as np

from mfgarchon.types.callable_protocols import CONDITION_SLOTS, SOLVER_SOURCE_SLOTS, bind_user_callable
from mfgarchon.utils.callable_adapter import adapt_ic_callable

_COEF = {"t": 100.0, "x": 1.0}


def _written(names, reads, star=""):
    """A function of ``names``, in that order, each reading the slot at its position in ``reads``."""
    body = " + ".join(f"{_COEF[slot]}*{name}" for name, slot in zip(names, reads, strict=True))
    return eval(f"lambda {', '.join([*names, *([star] if star else [])])}: {body}")


def _followed(refusal, reads):
    """The callable the refusal advises. Ordered by slot, each parameter takes the slot it is named for; named in
    place, a parameter keeps what it read at its position."""
    in_order = re.search(r"put them in the order \(([^)]*)\)", refusal)
    if in_order:
        names = in_order.group(1).split(", ")
        return _written(names, names)
    in_place = re.search(r"Name its parameters ([a-z, ]+), time first|It takes \(([^)]*)\); name its", refusal)
    return _written((in_place.group(1) or in_place.group(2)).split(", "), reads)


@pytest.mark.parametrize(
    ("slots", "names"),
    [(CONDITION_SLOTS, ["a", "b"]), (SOLVER_SOURCE_SLOTS, ["x", "b"])],
    ids=["a reordered role, unnamed with *rest", "a role never reordered, one name misplaced"],
)
def test_the_binders_advice_followed_reads_right(slots, names):
    reads = ["x", "t"]
    with pytest.raises(TypeError) as refused:
        bind_user_callable(_written(names, reads, "*rest"), slots, role="the role")
    assert bind_user_callable(_followed(str(refused.value), reads), slots, role="the role")(t=0.25, x=3.0) == 28.0


@pytest.mark.parametrize(
    ("names", "reads"),
    [
        (["t", "x"], ["t", "x"]),
        (["x", "t"], ["x", "t"]),
        (["time", "x"], ["t", "x"]),
        (["x", "time"], ["x", "t"]),
        (["y", "t"], ["x", "t"]),
    ],
    ids=lambda v: ",".join(v),
)
def test_the_advice_for_a_partial_fixing_time_reads_right_in_either_order(names, reads):
    """The advice was ``lambda x: f(0.25, x)``, whatever order f is written in."""
    f = _written(names, reads)
    time_name = names[reads.index("t")]
    with pytest.raises(TypeError) as refused:
        adapt_ic_callable(functools.partial(f, **{time_name: 0.25}), 1, 3.0, role="m_initial")
    advice = re.search(r"pass (lambda x: f\(.*?\)) to fix the time yourself", str(refused.value)).group(1)
    _, read = adapt_ic_callable(eval(advice, {"f": f}), 1, 3.0, role="m_initial")
    assert read(3.0) == 28.0


def _positional_only(x, /, t):
    return 100.0 * t + x


@pytest.mark.parametrize(
    "partial",
    [
        functools.partial(max, t=0.25),
        functools.partial(np.add, t=0.25),
        functools.partial(lambda t, x, c: 100.0 * t + x + c, t=0.25),
        functools.partial(lambda c, x, t: 100.0 * t + x + c, 1.0, t=0.25),
        functools.partial(lambda t, x, c: 100.0 * t + x + c, t=0.25, c=1.0),
        functools.partial(lambda t: t, t=0.25),
        functools.partial(_positional_only, t=0.25),
    ],
    ids=[
        "builtin",
        "ufunc",
        "two parameters left",
        "a parameter frozen by position",
        "two frozen",
        "no space",
        "positional only",
    ],
)
def test_the_advice_writes_no_lambda_where_it_cannot_say_which_parameter_is_space(partial):
    """A lambda needs the names f reads. Without them, a lambda written from the slots could swap what it passes."""
    with pytest.raises(TypeError, match="fixes the time itself, passing each argument") as refused:
        adapt_ic_callable(partial, 1, 3.0, role="m_initial")
    assert "lambda" not in str(refused.value)

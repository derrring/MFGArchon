"""A ``functools.partial`` handed to a binder reads what it freezes, or is refused with advice that keeps it (#2869).

The oracle is what the partial means by its own names: a frozen parameter reads the frozen value, and every other
parameter reads the slot it is named for. Called by name, the slot passed overrides a value frozen by keyword, with no
error, so that is refused. A positional freeze of a named parameter reads right, and stays accepted. Reordering a function
a partial froze by position moves the frozen value to another parameter, so no refusal of one says to reorder it.
"""

from __future__ import annotations

import functools
import itertools

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import applicator_graph as graph
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.boundary.types import bind_boundary_value
from mfgarchon.types import callable_protocols as cp
from mfgarchon.utils.callable_adapter import adapt_ic_callable

_TABLES = {name: slots for name, slots in vars(cp).items() if isinstance(slots, cp.Slots)}
_NAMES = sorted({slot for slots in _TABLES.values() for slot in slots.order})
_COEF = {slot: 10.0**i for i, slot in enumerate(_NAMES)}
_AT = {slot: 2.0 + i for i, slot in enumerate(_NAMES)}
_FROZEN = 97.0


def _written(params, reads):
    return eval(
        f"lambda {', '.join(params)}: " + " + ".join(f"{_COEF[r]!r}*{p}" for p, r in zip(params, reads, strict=True))
    )


def _meant(order, frozen):
    return sum(_COEF[slot] * frozen.get(slot, _AT[slot]) for slot in order)


def _bind(slots):
    return lambda f: cp.bind_user_callable(f, slots, role="r")


def _call(bound, order):
    return bound(**{slot: _AT[slot] for slot in order})


@pytest.mark.parametrize("name", list(_TABLES))
def test_a_slot_frozen_by_keyword_is_refused_on_every_binding_path(name):
    """Every way a slot's value can be frozen by keyword with the slot still named: by its name, by an alias, and after a
    positional freeze. Each read other at 2ace3724, with no error."""
    slots = _TABLES[name]
    order = slots.order
    partials = [functools.partial(_written(order, order), **{slot: _FROZEN}) for slot in order]
    partials += [
        functools.partial(_written([alias if s == slot else s for s in order], order), **{alias: _FROZEN})
        for alias, slot in slots.slot_of().items()
        if alias != slot
    ]
    if len(order) >= 2:
        partials.append(functools.partial(_written(order, order), _FROZEN, **{order[1]: _FROZEN}))
    for partial in partials:
        with pytest.raises(TypeError, match="by keyword: called by name"):
            cp.bind_user_callable(partial, slots, role="r")


@pytest.mark.parametrize(
    ("slots", "partial"),
    [
        (cp.SOLVER_SOURCE_SLOTS, functools.partial(lambda a, b, x=0.0: a + b + x, x=9.0)),
        (cp.SOURCE_TERM_SLOTS, functools.partial(lambda x, t, v, m=0.0: x + t + v + m, m=9.0)),
    ],
    ids=["ahead of the never-reordered positional return", "ahead of the out-of-order refusal"],
)
def test_the_frozen_slot_is_refused_before_any_other_path_reads_the_signature(slots, partial):
    """Reached first by neither, the first is accepted and the second is told to reorder a partial it should not keep."""
    with pytest.raises(TypeError, match="by keyword: called by name"):
        cp.bind_user_callable(partial, slots, role="r")


@pytest.mark.parametrize("name", list(_TABLES))
def test_what_a_partial_freezes_by_position_or_under_no_slots_name_stands(name):
    """The rows the refusal must not reach, with the frozen value away from the read point."""
    slots = _TABLES[name]
    order = slots.order
    read = _bind(slots)
    by_position = functools.partial(_written(order, order), _FROZEN)
    if slots.required:
        with pytest.raises(TypeError, match="cannot be matched"):
            read(by_position)
    else:
        assert _call(read(by_position), order) == _meant(order, {order[0]: _FROZEN})
    gain = eval(f"lambda {', '.join(order)}, gain=1.0: gain*(" + " + ".join(f"{_COEF[s]!r}*{s}" for s in order) + ")")
    assert _call(read(functools.partial(gain, gain=2.0)), order) == 2.0 * _meant(order, {})
    rest = list(order[1:])
    into_kw = eval(
        f"lambda {', '.join(rest)}, **kw: "
        + " + ".join([f"{_COEF[order[0]]!r}*kw[{order[0]!r}]", *(f"{_COEF[s]!r}*{s}" for s in rest)])
    )
    if slots.required:
        with pytest.raises(TypeError, match="cannot be matched"):
            read(functools.partial(into_kw, **{order[0]: _FROZEN}))
    else:
        assert _call(read(functools.partial(into_kw, **{order[0]: _FROZEN})), order) == _meant(
            order, {order[0]: _FROZEN}
        )


_SURFACES = {name: (_bind(slots), slots.order) for name, slots in _TABLES.items()} | {
    "by name, grid": (bind_boundary_value, cp.BOUNDARY_VALUE_SLOTS.order),
    "by name, graph value": (lambda f: graph._value_reader(f, "r"), cp.NODE_VALUE_SLOTS.order),
    "by name, graph rule": (lambda f: graph._rule_reader(f, "r"), cp.NODE_RULE_SLOTS.order),
}


@pytest.mark.parametrize("name", list(_SURFACES))
def test_a_refused_positional_partial_is_not_told_to_reorder_its_function(name):
    """``partial(lambda a, b: 100*b + a, 9.0)``, told to name and order its parameters, became ``partial(lambda t, x: 100*t +
    x, 9.0)``: accepted, with the 9.0 moved from x to t, 903.0 for 34.0. So did one named and out of order. The function the
    refusal advises reads the frozen value, for each written order, named and unnamed."""
    read, order = _SURFACES[name]
    refused = 0
    for written in itertools.permutations(order):
        for params in (list(written), [f"a{i}" for i in range(len(order))]):
            target = _written(params, written)
            try:
                read(functools.partial(target, _FROZEN))
            except TypeError as refusal:
                message = str(refusal)
            else:
                continue
            refused += 1
            assert "reorder its parameters" not in message, message
            assert "put them in the order" not in message, message
            assert "passing the frozen value by keyword" in message, message
            # A raw Hamiltonian refuses a function of fewer than its four slots, so what it is told takes them all.
            if name == "HAMILTONIAN_SLOTS":
                assert "a function of (t, x, p, m)" in message, message
            left = list(order) if name == "HAMILTONIAN_SLOTS" else [slot for slot in order if slot != written[0]]
            passed = ", ".join(
                f"{p}={s}" if s != written[0] else f"{p}={_FROZEN!r}" for p, s in zip(params, written, strict=True)
            )
            follows = eval(f"lambda {', '.join(left)}: target({passed})", {"target": target})
            assert _call(read(follows), order) == _meant(order, {written[0]: _FROZEN})
    assert refused, "no row reached a refusal"


def test_a_slot_other_than_time_frozen_on_a_condition_keeps_its_refusal():
    """``adapt_ic_callable`` catches a binder's refusal and tries other readings, which would lose this one's text. Time
    keeps the text written for it."""
    with pytest.raises(TypeError, match=r"^m_initial .* freezes x by keyword: called by name"):
        adapt_ic_callable(functools.partial(lambda t, x: 100.0 * t + x, x=9.0), 1, 3.0, role="m_initial")
    with pytest.raises(TypeError, match=r"fixes t=0\.7, but the library reads m_initial at its own time"):
        adapt_ic_callable(functools.partial(lambda t, x: 100.0 * t + x, t=0.7), 1, 3.0, role="m_initial")


@pytest.mark.parametrize("frozen", ["potential", "source_term_hjb"])
def test_a_solve_refuses_a_keyword_frozen_slot_rather_than_solve_the_unfrozen_function(frozen):
    """Both returned converged at 2ace3724, equal to the unfrozen function and 0.299 from the frozen one in U."""

    def potential(t, x):
        return 8.0 * t * (np.asarray(x, dtype=float)[..., 0] - 0.5)

    def source(t, x, v, m):
        return 8.0 * t * (np.asarray(x, dtype=float)[..., 0] - 0.5) * np.ones_like(np.asarray(v, dtype=float))

    hamiltonian = {"potential": functools.partial(potential, t=0.45)} if frozen == "potential" else {}
    problem = {"source_term_hjb": functools.partial(source, t=0.45)} if frozen == "source_term_hjb" else {}
    with pytest.raises(TypeError, match="freezes t by keyword: called by name"):
        MFGProblem(
            model=Model(
                hamiltonian=SeparableHamiltonian(
                    control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, **hamiltonian
                ),
                volatility=0.4,
            ),
            domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1)),
            conditions=Conditions(
                m_initial=lambda x: np.exp(-40 * (np.asarray(x, dtype=float)[..., 0] - 0.3) ** 2),
                u_terminal=lambda x: 2.0 * (np.asarray(x, dtype=float)[..., 0] - 0.7) ** 2,
                T=0.5,
            ),
            Nt=4,
            **problem,
        ).solve()

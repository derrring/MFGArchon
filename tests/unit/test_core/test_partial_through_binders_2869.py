"""A ``functools.partial`` handed to a binder reads what it freezes, or is refused with advice that keeps it (#2869).

The oracle is what the partial means by its own names: a frozen parameter reads the frozen value, and every other
parameter reads the slot it is named for. Called by name, the slot passed overrides a value frozen by keyword, with no
error, so that is refused, and only that: where the library does not pass the name, the frozen value stands. A positional
freeze of a named parameter reads right, and stays accepted. Reordering a function a partial froze by position moves the
frozen value to another parameter, so the refusals below do not say to reorder it.
"""

from __future__ import annotations

import functools
import itertools
from types import SimpleNamespace

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import DualHamiltonian, DualLagrangian, QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import applicator_graph as graph
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.boundary.types import bind_boundary_value
from mfgarchon.operators.interaction import PotentialCost
from mfgarchon.types import callable_protocols as cp
from mfgarchon.utils.callable_adapter import adapt_ic_callable
from mfgarchon.utils.pde_coefficients import CoefficientField
from mfgarchon.utils.validation.functions import _call_h

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
    """A slot's value frozen by keyword with the slot still named, so the library passes it: by its name, by an alias, after a
    positional freeze, and in a partial Python leaves as two levels. Each read other at 2ace3724, with no error."""
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
    # Python flattens a partial of a partial only when the inner one has no __dict__. These stay two levels.
    wrapped = functools.partial(_written(order, order), **{order[0]: _FROZEN})
    partials.append(functools.partial(functools.update_wrapper(wrapped, _written(order, order))))
    marked = functools.partial(_written(order, order), **{order[0]: _FROZEN})
    marked.note = "an attribute keeps it a level of its own"
    partials.append(functools.partial(marked))
    for partial in partials:
        # Refused, or, for a level Python's signature sees through to the function it wraps, loud at the first read. A raw
        # Hamiltonian's `required` may refuse it first.
        with pytest.raises(TypeError, match=r"by keyword: called by name|multiple values|cannot be matched"):
            _call(cp.bind_user_callable(partial, slots, role="r"), order)


@pytest.mark.parametrize("name", list(_TABLES))
def test_a_slot_frozen_on_an_unnamed_tail_stands_where_the_library_does_not_pass_it(name):
    """Bound positionally, the library passes the tail and not the frozen name, so the frozen value reads right. Each table's
    rows of this shape are accepted and read it, or are refused for the tail, as they are with no partial."""
    slots = _TABLES[name]
    order = slots.order
    accepted = 0
    for frozen in order:
        for tail in {len(order), len([slot for slot in order if slot not in slots.optional])}:
            parameters = [f"a{i}" for i in range(tail)]
            function = eval(f"lambda {', '.join([*parameters, frozen + '=0.0'])}: {frozen}")
            partial = functools.partial(function, **{frozen: _FROZEN})
            try:
                bound = cp.bind_user_callable(partial, slots, role="r")
            except cp.FrozenSlotError:
                pytest.fail(f"nothing here passes {frozen} by name: {partial}")
            except TypeError:
                continue
            accepted += 1
            assert bound.binding == "positional"
            assert _call(bound, order) == _FROZEN
    assert accepted == {
        "CONDITIONAL_HAMILTONIAN_SLOTS": 5,
        "SOLVER_SOURCE_SLOTS": 2,
        "VARIATIONAL_LAGRANGIAN_SLOTS": 4,
    }.get(name, 0)


@pytest.mark.parametrize(
    ("read", "partial", "meant"),
    [
        (
            lambda f: adapt_ic_callable(f, 1, 3.0, role="m_initial")[1](3.0),
            functools.partial(lambda a, x=0.0: 100.0 * a + x, x=9.0),
            309.0,
        ),
        (
            lambda f: adapt_ic_callable(f, 1, 3.0, role="m_initial")[1](3.0),
            functools.partial(lambda a, t=0.0: 100.0 * t + a, t=0.5),
            53.0,
        ),
        (
            lambda f: adapt_ic_callable(f, 1, 3.0, role="m_initial")[1](3.0),
            functools.partial(lambda x, **kw: 100.0 * kw["t"] + x, t=0.5),
            53.0,
        ),
        (
            lambda f: cp.bind_user_callable(f, cp.SOLVER_SOURCE_SLOTS, role="r")(t=0.25, x=3.0),
            functools.partial(lambda *a, t=0.0: a[0] + 100.0 * a[1] + 1000.0 * t, t=0.7),
            1000.25,
        ),
        (
            lambda f: cp.bind_user_callable(f, cp.VARIATIONAL_LAGRANGIAN_SLOTS, role="r")(t=1.0, x=2.0, v=3.0, m=4.0),
            functools.partial(lambda *a, m=0.0: sum(a) + 1000.0 * m, m=0.7),
            710.0,
        ),
    ],
    ids=["a condition, x", "a condition, time", "a condition, time into **kw", "a source, *a", "a Lagrangian, *a"],
)
def test_a_slot_frozen_where_the_library_does_not_pass_it_reads_what_it_froze(read, partial, meant):
    assert read(partial) == meant


def _two_levels(function, *args):
    inner = functools.partial(function, *args)
    inner.note = "an attribute keeps it a level of its own"
    return functools.partial(inner)


_ONE_ARG_SURFACES = {
    "condition, 1-D": lambda f: adapt_ic_callable(f, 1, 3.0, role="m_initial"),
    "condition, 2-D": lambda f: adapt_ic_callable(f, 2, np.array([0.1, 0.2]), role="m_initial"),
    "PotentialCost": lambda f: PotentialCost(f).bind_on(
        SimpleNamespace(points=np.array([[0.5]]), weights=np.array([1.0]))
    ),
}


@pytest.mark.parametrize(
    ("read", "partial"),
    [
        (_bind(cp.SOLVER_SOURCE_SLOTS), functools.partial(lambda a, x, b, *rest: a + x + b, _FROZEN)),
        (bind_boundary_value, functools.partial(lambda a0, a1, *rest: a0 + a1, _FROZEN)),
        (lambda f: graph._value_reader(f, "r"), functools.partial(lambda a0, a1, **kw: a0 + a1, _FROZEN)),
        (bind_boundary_value, functools.partial(lambda q, t, pos: q + t + pos, _FROZEN)),
        (lambda f: graph._rule_reader(f, "r"), functools.partial(lambda q, t, a, b: q + t + a + b, _FROZEN)),
        (_ONE_ARG_SURFACES["condition, 1-D"], functools.partial(lambda x, t: x + t, _FROZEN)),
        (_ONE_ARG_SURFACES["condition, 1-D"], functools.partial(lambda *a, **k: a[5], _FROZEN)),
        (_ONE_ARG_SURFACES["condition, 2-D"], functools.partial(lambda a, b: a + b, _FROZEN)),
        (_ONE_ARG_SURFACES["PotentialCost"], functools.partial(lambda q, a, b: q + a + b, _FROZEN)),
        (bind_boundary_value, _two_levels(lambda a, b: a + b, _FROZEN)),
        (_bind(cp.POTENTIAL_SLOTS), _two_levels(lambda q, a, b: q + a + b, _FROZEN)),
    ],
    ids=[
        "never reordered, a misplaced name",
        "by name, *rest",
        "by name, **kw",
        "by name, not all named",
        "by name, not all named, a rule",
        "condition, time and no space",
        "condition, names nothing",
        "condition, 2-D, the signature footer",
        "PotentialCost",
        "by name, two levels",
        "binder, two levels",
    ],
)
def test_every_refusal_a_positional_partial_reaches_advises_keeping_what_it_froze(read, partial):
    """None of them may advise the rewrite that moves the frozen value, in its outer message or a chained one. Each once
    said to reorder or to name the function's parameters."""
    with pytest.raises(TypeError) as refused:
        read(partial)
    messages = []
    error = refused.value
    while error is not None:
        messages.append(str(error))
        error = error.__cause__
    message = "\n".join(messages)
    for advice in (
        "reorder its parameters",
        "put them in the order",
        "Name its space parameter x",
        "name the space parameter x",
        "name its parameters after",
    ):
        assert advice not in message, message
    assert "passing the frozen value by keyword" in message, message


@pytest.mark.parametrize("name", list(_TABLES))
def test_what_a_partial_freezes_by_position_or_under_no_slots_name_stands(name):
    """The rows the refusal must not reach, with the frozen value away from the read point."""
    slots = _TABLES[name]
    order = slots.order
    read = _bind(slots)
    assert _call(read(functools.partial(_written(order, order), _FROZEN)), order) == _meant(order, {order[0]: _FROZEN})
    gain = eval(f"lambda {', '.join(order)}, gain=1.0: gain*(" + " + ".join(f"{_COEF[s]!r}*{s}" for s in order) + ")")
    assert _call(read(functools.partial(gain, gain=2.0)), order) == 2.0 * _meant(order, {})
    rest = list(order[1:])
    into_kw = eval(
        f"lambda {', '.join(rest)}, **kw: "
        + " + ".join([f"{_COEF[order[0]]!r}*kw[{order[0]!r}]", *(f"{_COEF[s]!r}*{s}" for s in rest)])
    )
    assert _call(read(functools.partial(into_kw, **{order[0]: _FROZEN})), order) == _meant(order, {order[0]: _FROZEN})


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
            if written[0] == "p" and name == "HAMILTONIAN_SLOTS":
                continue  # a function that does not read p is not a raw Hamiltonian, and is refused for it
            left = [slot for slot in order if slot != written[0]]
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


def _solve(*, hamiltonian=None, model=None, problem=None):
    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, **(hamiltonian or {})
            ),
            **{"volatility": 0.4, **(model or {})},
        ),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(
            m_initial=lambda x: np.exp(-40 * (np.asarray(x, dtype=float)[..., 0] - 0.3) ** 2),
            u_terminal=lambda x: 2.0 * (np.asarray(x, dtype=float)[..., 0] - 0.7) ** 2,
            T=0.5,
        ),
        Nt=4,
        **(problem or {}),
    ).solve()


def _in_space(x, like):
    return 8.0 * (np.asarray(x, dtype=float)[..., 0] - 0.5) * np.ones_like(np.asarray(like, dtype=float))


_POTENTIAL = functools.partial(lambda t, x: 8.0 * t * (np.asarray(x, dtype=float)[..., 0] - 0.5), t=0.45)


@pytest.mark.parametrize(
    "arguments",
    [
        {"hamiltonian": {"potential": _POTENTIAL}},
        {"problem": {"source_term_hjb": functools.partial(lambda t, x, v, m: t * _in_space(x, v), t=0.45)}},
        {"problem": {"source_term_fp": functools.partial(lambda t, x, m: t * _in_space(x, m), t=0.45)}},
    ],
    ids=["potential", "source_term_hjb", "source_term_fp"],
)
def test_a_solve_refuses_a_keyword_frozen_slot_rather_than_solve_the_unfrozen_function(arguments):
    """Each returned converged at 2ace3724, equal to the unfrozen function, with no error."""
    with pytest.raises(TypeError, match="freezes t by keyword: called by name"):
        _solve(**arguments)


def test_a_frozen_volatility_is_refused_before_the_solve_starts():
    """``CoefficientField`` calls a callable by name with no binder, so it solved the unfrozen function and converged. No
    solve starts: the volatility is never called."""
    calls = []

    def sigma(t, x, m):
        calls.append(t)
        return 0.2 + t

    with pytest.raises(TypeError, match=r"^volatility .* freezes t by keyword: called by name"):
        _solve(model={"volatility": functools.partial(sigma, t=0.45)})
    assert not calls


@pytest.mark.parametrize(
    "read",
    [
        lambda v: MFGProblem(
            geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[5], boundary_conditions=no_flux_bc(dimension=1)),
            T=0.5,
            Nt=2,
            volatility=v,
            components=MFGComponents(
                hamiltonian=SeparableHamiltonian(
                    control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m
                ),
                m_initial=lambda x: 1.0,
                u_terminal=lambda x: 0.0,
            ),
        ),
        lambda v: CoefficientField(v, 0.1, "volatility"),
    ],
    ids=["MFGProblem", "CoefficientField, a solver's argument"],
)
def test_the_other_ways_a_volatility_arrives_refuse_it_too(read):
    with pytest.raises(TypeError, match=r"^volatility .* freezes t by keyword: called by name"):
        read(functools.partial(lambda t, x, m: 0.2 + t, t=0.45))


def test_what_a_raw_hamiltonian_is_told_about_a_frozen_measure_is_accepted_and_reads_it():
    def H(t, x, p, m):
        return 100.0 * p + x + 1000.0 * m

    with pytest.raises(TypeError, match="passing the frozen value by keyword"):
        cp.bind_user_callable(functools.partial(H, m=0.5), cp.HAMILTONIAN_SLOTS, role="r")
    bound = cp.bind_user_callable(lambda t, x, p: H(t=t, x=x, p=p, m=0.5), cp.HAMILTONIAN_SLOTS, role="r")
    assert bound(t=0.25, x=3.0, p=2.0, m=7.0) == 703.0


def test_a_frozen_slot_with_no_readable_signature_is_refused_where_it_would_be_passed_by_name():
    """``dict`` has no signature Python can read, so nothing says the frozen name is not a parameter."""
    with pytest.raises(TypeError, match="freezes t by keyword: called by name"):
        CoefficientField(functools.partial(dict, t=0.45), 0.1, "volatility")


def test_a_coefficient_frozen_under_no_name_it_is_passed_stays_accepted():
    field = CoefficientField(functools.partial(lambda t, x, m, gain=1.0: gain * (0.2 + t), gain=2.0), 0.1, "volatility")
    assert field.evaluate_at(2, np.linspace(0.0, 1.0, 3), np.ones(3), 0.25) == pytest.approx(1.4)


@pytest.mark.parametrize(
    "read",
    [
        lambda: DualHamiltonian(functools.partial(lambda x, alpha, m, t: 0.5 * float(np.sum(alpha**2)) + m, m=0.5)),
        lambda: DualLagrangian(functools.partial(lambda t, x, p, m: 0.5 * p * p + m, m=0.5)),
        lambda: _call_h(functools.partial(_family_member().dm, m=0.5), 0.0, 0.0, 1.0, 7.0),
    ],
    ids=["DualHamiltonian", "DualLagrangian", "a validator, a family's method"],
)
def test_the_doors_off_the_default_path_read_the_same_check(read):
    """Each called by name with no binder, and read the slot and not the frozen value."""
    with pytest.raises(TypeError, match=r"freezes m by keyword: called by name"):
        read()


def _family_member():
    return SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m)

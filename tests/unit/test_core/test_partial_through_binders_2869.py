"""A ``functools.partial`` handed to a binder reads what it freezes, or is refused with advice that keeps it (#2869, #2875).

The oracle is what the partial means by its own names: a frozen parameter reads the frozen value, and every other
parameter reads the slot it is named for. Called by name, the slot passed overrides a value frozen by keyword, with no
error, so that is refused. A positional freeze of a named parameter reads right, and stays accepted. Renaming or
reordering a function a partial froze by position moves the frozen value to another parameter, so the refusals below do
not say to.
"""

from __future__ import annotations

import functools
import itertools
from types import SimpleNamespace

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import DualHamiltonian, DualLagrangian, QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import applicator_graph as graph
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.boundary.types import bind_boundary_value
from mfgarchon.operators.interaction import PotentialCost
from mfgarchon.types import callable_protocols as cp
from mfgarchon.utils.pde_coefficients import CoefficientField
from mfgarchon.utils.validation.functions import _call_h, validate_hamiltonian, validate_hamiltonian_derivative
from mfgarchon.utils.validation.protocol import ValidationError

_TABLES = {name: slots for name, slots in vars(cp).items() if isinstance(slots, cp.Slots)}
_NAMES = sorted({slot for slots in _TABLES.values() for slot in slots.order})
_AT = {slot: 2.0 + i for i, slot in enumerate(_NAMES)}
_FROZEN = 97.0


def _coef(order, slot):
    """Slot k of a door's order is the k-th base-100 digit. Every value read is an integer below 100, so a misread changes
    a digit, and a door's at most five digits sum to an integer below 2**53, exact in floating point: compared by ==."""
    return 100 ** list(order).index(slot)


def _term(param, slot, order):
    return f"{_coef(order, slot)}*{param}[0]" if slot == "x" else f"{_coef(order, slot)}*{param}"


def _written(params, reads, order):
    return eval(
        f"lambda {', '.join(params)}: " + " + ".join(_term(p, r, order) for p, r in zip(params, reads, strict=True))
    )


def _frozen(slot):
    return (_FROZEN,) if slot == "x" else _FROZEN


def _meant(order, frozen):
    return sum(_coef(order, slot) * frozen.get(slot, _AT[slot]) for slot in order)


def _timed(function):
    """A decorator as a user writes one: ``inspect.signature`` reads through it to what it wraps."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    return wrapper


def _bind(slots):
    return lambda f: cp.bind_user_callable(f, slots, role="r")


def _call(bound, order):
    return bound(**{slot: (_AT[slot],) if slot == "x" else _AT[slot] for slot in order})


@pytest.mark.parametrize("name", list(_TABLES))
def test_a_slot_frozen_by_keyword_is_refused_on_every_binding_path(name):
    """A slot's value frozen by keyword with the slot still named, so the library passes it: by its name, after a positional
    freeze, and in a partial Python leaves as two levels. Each read other at 2ace3724, with no error."""
    slots = _TABLES[name]
    order = slots.order
    partials = [functools.partial(_written(order, order, order), **{slot: _frozen(slot)}) for slot in order]
    if len(order) >= 2:
        partials.append(
            functools.partial(_written(order, order, order), _frozen(order[0]), **{order[1]: _frozen(order[1])})
        )
    # Python flattens a partial of a partial only when the inner one has no __dict__. These stay two levels.
    wrapped = functools.partial(_written(order, order, order), **{order[0]: _frozen(order[0])})
    partials.append(functools.partial(functools.update_wrapper(wrapped, _written(order, order, order))))
    marked = functools.partial(_written(order, order, order), **{order[0]: _frozen(order[0])})
    marked.note = "an attribute keeps it a level of its own"
    partials.append(functools.partial(marked))
    partials.append(_timed(functools.partial(_written(order, order, order), **{order[0]: _frozen(order[0])})))
    for partial in partials:
        # A wrapper's own names are read and not what it wraps, so it is refused for its stars.
        with pytest.raises(TypeError, match=r"by keyword: called by name|starred parameter"):
            cp.bind_user_callable(partial, slots, role="r")


def _two_levels(function, *args):
    inner = functools.partial(function, *args)
    inner.note = "an attribute keeps it a level of its own"
    return functools.partial(inner)


def _problem_with_a_condition(condition, dimension):
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * dimension, Nx_points=[5] * dimension, boundary_conditions=no_flux_bc(dimension=dimension)
    )
    hamiltonian = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m)
    return MFGProblem(
        model=Model(hamiltonian=hamiltonian, volatility=0.4),
        domain=grid,
        conditions=Conditions(m_initial=condition, u_terminal=lambda x: 0.0, T=0.5),
        Nt=2,
    )


_ONE_ARG_SURFACES = {
    "condition, 1-D": lambda f: _problem_with_a_condition(f, 1),
    "condition, 2-D": lambda f: _problem_with_a_condition(f, 2),
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
    with pytest.raises((TypeError, ValidationError)) as refused:
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
    if len(order) > 1 and order[0] not in slots.required:
        by_position = functools.partial(_written(order, order, order), _frozen(order[0]))
        assert _call(read(by_position), order) == _meant(order, {order[0]: _FROZEN})
    gain = eval(f"lambda {', '.join(order)}, gain=1.0: gain*(" + " + ".join(_term(s, s, order) for s in order) + ")")
    assert _call(read(functools.partial(gain, gain=2.0)), order) == 2.0 * _meant(order, {})


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
            target = _written(params, written, order)
            try:
                read(functools.partial(target, _frozen(written[0])))
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
            if not left:
                continue  # a function of the frozen value alone reads no slot, and is refused for it
            passed = ", ".join(
                f"{p}={s}" if s != written[0] else f"{p}={_frozen(s)!r}" for p, s in zip(params, written, strict=True)
            )
            follows = eval(f"lambda {', '.join(left)}: target({passed})", {"target": target})
            assert _call(read(follows), order) == _meant(order, {written[0]: _FROZEN})
    assert refused, "no row reached a refusal"


def _problem(*, hamiltonian=None, model=None, problem=None):
    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, **(hamiltonian or {})
            ),
            **{"volatility": 0.4, **(model or {})},
        ),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(
            m_initial=lambda x: np.exp(-40 * (x[0] - 0.3) ** 2),
            u_terminal=lambda x: 2.0 * (x[0] - 0.7) ** 2,
            T=0.5,
        ),
        Nt=4,
        **(problem or {}),
    )


def _solve(**arguments):
    return _problem(**arguments).solve()


def _in_space(x, like):
    return 8.0 * (np.asarray(x[0], dtype=float) - 0.5) * np.ones_like(np.asarray(like, dtype=float))


_POTENTIAL = functools.partial(lambda t, x: 8.0 * t * (np.asarray(x[0], dtype=float) - 0.5), t=0.45)


@pytest.mark.parametrize(
    "arguments",
    [
        {"hamiltonian": {"potential": _POTENTIAL}},
        {"problem": {"source_term_hjb": functools.partial(lambda t, x, v, m: t * _in_space(x, v), t=0.45)}},
        {"problem": {"source_term_fp": functools.partial(lambda t, x, m: t * _in_space(x, m), t=0.45)}},
        {"hamiltonian": {"potential": _timed(_POTENTIAL)}},
        {"problem": {"source_term_hjb": _timed(functools.partial(lambda t, x, v, m: t * _in_space(x, v), t=0.45))}},
        {"model": {"volatility": _timed(functools.partial(lambda t, x, m: 0.2 + t, t=0.45))}},
    ],
    ids=[
        "potential",
        "source_term_hjb",
        "source_term_fp",
        "potential, a wrapper",
        "source_term_hjb, a wrapper",
        "volatility, a wrapper",
    ],
)
def test_a_solve_refuses_a_keyword_frozen_slot_rather_than_solve_the_unfrozen_function(arguments):
    """Each returned converged at 2ace3724, equal to the unfrozen function, with no error."""
    with pytest.raises(TypeError, match=r"freezes t by keyword: called by name|starred parameter"):
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


def test_a_frozen_volatility_is_refused_when_the_problem_is_built():
    """`MFGProblem` reads the check, so no ``solve()`` is needed to see it. `Model` holds what it is given."""
    with pytest.raises(TypeError, match=r"^volatility .* freezes t by keyword: called by name"):
        _problem(model={"volatility": functools.partial(lambda t, x, m: 0.2 + t, t=0.45)})


def test_a_solvers_own_volatility_argument_is_refused_too():
    """A solver takes a volatility as its own argument, into `CoefficientField`, with no problem to read it first."""
    with pytest.raises(TypeError, match=r"^volatility .* freezes t by keyword: called by name"):
        CoefficientField(functools.partial(lambda t, x, m: 0.2 + t, t=0.45), 0.1, "volatility")


def test_what_a_raw_hamiltonian_is_told_about_a_frozen_measure_is_accepted_and_reads_it():
    def H(t, x, p, m):
        return 100.0 * p + x[0] + 1000.0 * m

    with pytest.raises(TypeError, match="passing the frozen value by keyword"):
        cp.bind_user_callable(functools.partial(H, m=0.5), cp.HAMILTONIAN_SLOTS, role="r")
    bound = cp.bind_user_callable(lambda t, x, p: H(t=t, x=x, p=p, m=0.5), cp.HAMILTONIAN_SLOTS, role="r")
    assert bound(t=0.25, x=(3.0,), p=2.0, m=7.0) == 703.0


def test_a_frozen_slot_with_no_readable_signature_is_refused_where_it_would_be_passed_by_name():
    """A door passes every slot whatever a signature says, so ``dict``, with none Python can read, is refused the same."""
    with pytest.raises(TypeError, match="not a plain def or lambda"):
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


class _FamilyWithStars(SeparableHamiltonian):
    def custom(self, t, x, p, **kw):
        return kw["m"]


@pytest.mark.parametrize(
    "read",
    [
        lambda: CoefficientField(functools.partial(lambda x, m, **kw: 0.2 + kw["t"], t=0.45), 0.1, "volatility"),
        lambda: CoefficientField(functools.partial(lambda **kw: 0.2 + kw["t"], t=0.45), 0.1, "volatility"),
        lambda: CoefficientField(
            functools.update_wrapper(functools.partial(lambda **kw: 0.2 + kw["t"], t=0.45), lambda x, m: 0.0),
            0.1,
            "volatility",
        ),
        lambda: _problem(model={"volatility": functools.partial(lambda x, m, **kw: 0.2 + kw["t"], t=0.45)}),
        lambda: DualHamiltonian(functools.partial(lambda x, alpha, t, **kw: kw["m"], m=0.5)),
        lambda: DualLagrangian(functools.partial(lambda t, x, p, **kw: kw["m"], m=0.5)),
        lambda: _call_h(
            functools.partial(
                _FamilyWithStars(control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m).custom,
                m=0.5,
            ),
            0.0,
            7.0,
            1.0,
        ),
    ],
    ids=[
        "CoefficientField, **kw beside names",
        "CoefficientField, **kw alone",
        "CoefficientField, a signature that hides it",
        "MFGProblem",
        "DualHamiltonian",
        "DualLagrangian",
        "a validator, a family's method",
    ],
)
def test_a_door_that_passes_every_slot_overrides_what_is_frozen_into_kw(read):
    """A binding passes the names it read, so a slot frozen into ``**kw`` stands there. These pass every slot, so it did not:
    through ``problem.solve()``, 6.78e-2 from the frozen function in ``U`` on a 21-node fixture, converged, at 2ace3724."""
    with pytest.raises(TypeError, match=r"freezes [tm] by keyword: called by name|starred parameter"):
        read()


@pytest.mark.parametrize(
    "partial", [lambda x, m, time=0.0: 0.2 + time, lambda time, x, m: 0.2 + time], ids=["last", "first"]
)
def test_an_alias_a_door_never_passes_stands(partial):
    field = CoefficientField(functools.partial(partial, time=0.45), 0.1, "volatility")
    assert field.evaluate_at(2, np.linspace(0.0, 1.0, 3), np.ones(3), 0.25) == pytest.approx(0.65)


@pytest.mark.parametrize(
    ("read", "partial"),
    [
        (
            lambda f: cp.bind_user_callable(f, cp.POTENTIAL_SLOTS, role="r"),
            lambda: _timed(functools.partial(lambda t, x: 100.0 * t + x, t=0.25)),
        ),
        (bind_boundary_value, lambda: _timed(functools.partial(lambda t, x: 100.0 * t + x, t=0.25))),
        (
            lambda f: cp.bind_user_callable(f, cp.CONDITION_SLOTS, role="m_initial"),
            lambda: _timed(functools.partial(lambda t, x: 100.0 * t + x, t=0.25)),
        ),
        (
            lambda f: CoefficientField(f, 0.1, "volatility"),
            lambda: _timed(functools.partial(lambda t, x, m: 0.2 + t, t=0.25)),
        ),
        (
            lambda f: cp.bind_user_callable(f, cp.POTENTIAL_SLOTS, role="r"),
            lambda: functools.lru_cache(maxsize=None)(functools.partial(lambda t, x: 100.0 * t + x, t=0.25)),
        ),
        (
            lambda f: cp.bind_user_callable(f, cp.POTENTIAL_SLOTS, role="r"),
            lambda: functools.partial(_timed(functools.partial(lambda t, x: 100.0 * t + x, t=0.25))),
        ),
    ],
    ids=["binder", "by name", "condition", "CoefficientField", "binder, lru_cache", "binder, a partial of a wrapper"],
)
def test_a_wrapper_does_not_hide_a_partial_from_the_check(read, partial):
    """``inspect.signature`` reads through ``__wrapped__``, so the wrapper was called with the partial's names, and read 53.0
    for 28.0 at 2ace3724 with no error."""
    with pytest.raises(TypeError, match=r"freezes t by keyword: called by name|starred parameter|not a plain def"):
        read(partial())


@pytest.mark.parametrize(
    "validate",
    [
        lambda f, grid: validate_hamiltonian(f, grid),
        lambda f, grid: validate_hamiltonian_derivative(f, grid, "dH_dm"),
    ],
    ids=["validate_hamiltonian", "validate_hamiltonian_derivative"],
)
def test_what_the_validators_tell_a_positional_partial_keeps_what_it_froze(validate):
    """They said to name a raw callable's parameters t, x, p, m. Followed, with the partial kept, it validates and reads t = 9."""
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
    result = validate(functools.partial(lambda a, b, c, d: a + b + c + d, 9.0), grid)
    assert not result.is_valid
    assert any("passing the frozen value by keyword" in (issue.suggestion or "") for issue in result.issues), (
        result.issues
    )


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
    froze the first parameter, whichever it was (#2861)."""
    with pytest.raises(TypeError) as excinfo:
        read(refused)
    message = str(excinfo.value)
    assert "that passes each argument to the parameter that reads it" in message, message
    assert "`lambda" not in message, message
    assert "Freeze it by position" not in message, message
    assert ("passing the frozen value by keyword" in message) == isinstance(refused, functools.partial), message

"""-c*grad(U) drives the FP only for a Hamiltonian and its cost exactly as the library ships them (#2907, #2914).

At 8f7c0db8 `drift_is_the_control_costs` compared the control cost's type attribute with the library's, so a cost
serving another control on its instance (an assignment, a partial, a callable object, another cost's library method)
was taken for the library's: the default scheme and FP-SL drove the FP by -c*grad(U) with c = 1, 1.2074e+00 and
1.4437e+00 off in M (1-D no-flux, 21 nodes, at 8f7c0db8), with no error. #2915's first head read the served method's
`__func__` and `__self__`, which a forwarding wrapper reports falsely (#2915 review 1, B1 and F3). Under the user's
coarse rule of 2026-11-09 it now checks types: the Hamiltonian's class one of the library's shipped classes whose drift
is the cost's, its cost a `QuadraticControlCost`, and no family method in either instance's `__dict__`, through the one
owner of "as the library ships it", `_why_not_as_it_ships`. A library class patched at run time is outside the rule
(a stated limit, as for #2899's rows path; #2917).

Each form describes H = 1.5|p|^2 - 0.5 m: the library `SeparableHamiltonian(QuadraticControlCost(1/3))`. Uncertified,
the default scheme takes the control as a velocity and solves as that library H, and FP-SL, which takes the value
function only, refuses with the owner's reason. These are pins comparing two routes of one scheme (AGENTS.md § Closing
out a fix, item 2), not an external oracle. 1-D no-flux, 21 nodes.
"""

from __future__ import annotations

import functools
import logging
import types
import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers import FPSLSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.validation import ValidationError

COST_METHODS = ("evaluate", "dp", "optimal_control")
H_METHODS = ("__call__", "dp", "optimal_control")
COST_LIBRARY = {name: QuadraticControlCost.__dict__[name] for name in COST_METHODS}
H_LIBRARY = {name: SeparableHamiltonian.__dict__[name] for name in H_METHODS}
STEEP = QuadraticControlCost(control_cost=1 / 3)


def _coupling(m):
    return 0.5 * m


STEEP_H = SeparableHamiltonian(QuadraticControlCost(control_cost=1 / 3), coupling=_coupling)


def _problem(hamiltonian):
    nodes = np.linspace(0.0, 1.0, 21)
    mass = np.trapezoid(0.2 + np.exp(-40 * (nodes - 0.3) ** 2), nodes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MFGProblem(
            model=Model(hamiltonian=hamiltonian, volatility=0.3),
            domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1)),
            conditions=Conditions(
                m_initial=lambda x: (0.2 + np.exp(-40 * (x[0] - 0.3) ** 2)) / mass,
                u_terminal=lambda x: 2.0 * (x[0] - 0.8) ** 2,
                T=0.5,
            ),
            Nt=10,
        )


def _quiet(run):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.CRITICAL)
        try:
            return np.asarray(run().M)
        finally:
            logging.disable(logging.NOTSET)


def _default(hamiltonian):
    return _quiet(lambda: _problem(hamiltonian).solve(max_iterations=6, tolerance=1e-12, verbose=False))


def _fp_sl(hamiltonian):
    problem = _problem(hamiltonian)
    return _quiet(
        lambda: FixedPointIterator(problem, HJBFDMSolver(problem), FPSLSolver(problem)).solve(
            max_iterations=6, tolerance=1e-12, verbose=False
        )
    )


def _steep_cost_body(name):
    return lambda self, p: COST_LIBRARY[name](STEEP, p)


def _steep_h_body(name):
    return lambda self, t, x, p, m: H_LIBRARY[name](STEEP_H, t=t, x=x, p=p, m=m)


def _another_costs_library_methods(monkeypatch):
    cost = QuadraticControlCost(control_cost=1.0)
    for name in COST_METHODS:
        setattr(cost, name, getattr(QuadraticControlCost(control_cost=1 / 3), name))
    return SeparableHamiltonian(cost, coupling=_coupling)


def _partials(monkeypatch):
    cost = QuadraticControlCost(control_cost=1.0)
    for name in COST_METHODS:
        setattr(cost, name, functools.partial(COST_LIBRARY[name], STEEP))
    return SeparableHamiltonian(cost, coupling=_coupling)


class _Forwarding:
    """A decorator serving its own body that forwards every attribute read, ``__func__`` and ``__self__`` included, to
    the library method it wraps (#2915 review 1, B1 and F3): it reports the library's method and calls another."""

    def __init__(self, library, body):
        self.library, self.body = library, body

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        reported, body = self.library.__get__(instance, owner), self.body

        class Bound:
            def __call__(self, *args, **kwargs):
                return body(instance, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(reported, name)

        return Bound()


def _plain_subclass(monkeypatch):
    cls = type("PlainH", (SeparableHamiltonian,), {})
    return cls(QuadraticControlCost(control_cost=1 / 3), coupling=_coupling)


def _forwarding_cost(monkeypatch):
    members = {name: _Forwarding(COST_LIBRARY[name], _steep_cost_body(name)) for name in COST_METHODS}
    return SeparableHamiltonian(type("FwdCost", (QuadraticControlCost,), members)(control_cost=1.0), coupling=_coupling)


def _forwarding_h(monkeypatch):
    members = {name: _Forwarding(H_LIBRARY[name], _steep_h_body(name)) for name in H_METHODS}
    return type("FwdH", (SeparableHamiltonian,), members)(QuadraticControlCost(control_cost=1.0), coupling=_coupling)


def _lambda_disagreeing(monkeypatch):
    """The library body divides by ``_lambda`` = 1/3, while a class attribute ``lambda_`` reads 1 (#2915 review 1, F4)."""
    cls = type("LambdaCost", (QuadraticControlCost,), {"lambda_": 1.0})
    return SeparableHamiltonian(cls(control_cost=1 / 3), coupling=_coupling)


def _property_serving_the_library(monkeypatch):
    cls = type(
        "SelfBound",
        (QuadraticControlCost,),
        {name: property(lambda self, name=name: types.MethodType(COST_LIBRARY[name], self)) for name in COST_METHODS},
    )
    return SeparableHamiltonian(cls(control_cost=1 / 3), coupling=_coupling)


#: Forms the type rule does not certify: instance entries on the cost, a plain subclass, forwarding wrappers on each
#: side, a cost whose lambda_ disagrees with the _lambda its body reads, and a property serving the library method.
NOT_AS_SHIPPED = {
    "another_costs_library_methods_on_the_instance": _another_costs_library_methods,
    "partials_on_the_cost_instance": _partials,
    "plain_subclass_of_SeparableHamiltonian": _plain_subclass,
    "forwarding_wrappers_on_the_cost": _forwarding_cost,
    "forwarding_wrappers_on_the_hamiltonian": _forwarding_h,
    "lambda_disagreeing_with_its_body": _lambda_disagreeing,
    "property_serving_the_library_method": _property_serving_the_library,
}


@pytest.mark.parametrize("form", sorted(NOT_AS_SHIPPED))
def test_a_hamiltonian_not_as_shipped_is_read_as_it_serves(form, monkeypatch):
    """The default scheme solves as the library H the form describes, and FP-SL, which would form -c*grad(U), refuses,
    naming why the Hamiltonian is not as shipped."""
    hamiltonian = NOT_AS_SHIPPED[form](monkeypatch)
    np.testing.assert_allclose(_default(hamiltonian), _default(STEEP_H), rtol=0, atol=1e-12)
    with pytest.raises(
        NotImplementedError,
        match=r"^FPSLSolver\.solve_fp_system\(potential_field=U\), run on .*: \w+ is not as the library ships it \(",
    ):
        _fp_sl(hamiltonian)


@pytest.mark.parametrize("held_by", ["instance", "class"])
def test_a_cost_serving_another_costs_control_alone_is_refused_at_construction(held_by):
    """A QuadraticControlCost whose instance, or whose class, holds only an optimal_control, another cost's, is not as the
    library ships it (#2914), and its control is not its own value's -dp: the problem refuses it when it is built,
    naming the pair (#2901). Before #2901 it solved, with the control from one cost and the value from another."""
    if held_by == "instance":
        cost = QuadraticControlCost(control_cost=1.0)
        cost.optimal_control = STEEP.optimal_control
    else:
        own = type("OwnControl", (QuadraticControlCost,), {"optimal_control": lambda self, p: STEEP.optimal_control(p)})
        cost = own(control_cost=1.0)
    with pytest.raises(ValidationError, match=r"optimal_control disagrees with dp and with H at x="):
        _problem(SeparableHamiltonian(cost, coupling=_coupling))


def test_the_shipped_hamiltonian_is_certified_and_fp_sl_takes_it():
    """The library H itself, and a deep copy of it, are certified, and FP-SL, which forms -c*grad(U), solves the copy
    as it solves the library H."""
    import copy

    from mfgarchon.utils.pde_coefficients import drift_is_the_control_costs

    twin = copy.deepcopy(STEEP_H)
    assert drift_is_the_control_costs(STEEP_H)
    assert drift_is_the_control_costs(twin)
    np.testing.assert_allclose(_fp_sl(twin), _fp_sl(STEEP_H), rtol=0, atol=1e-12)

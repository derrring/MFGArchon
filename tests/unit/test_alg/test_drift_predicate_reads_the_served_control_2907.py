"""-c*grad(U) drives the FP only for the library's own control, as each object serves it (#2907).

`drift_is_the_control_costs` compared the control cost's type attribute with `QuadraticControlCost.optimal_control`,
and both sides with the class attribute as it stood. A cost serving another control on its instance, through
`__getattribute__`, a descriptor, a partial or a callable object, or another cost's library method, and either class
with its control replaced, were taken for the library's: `problem.solve()`'s default scheme and FP-SL drove the FP by
-c*grad(U) with c = 1, 1.2074e+00 and 1.4437e+00 off in M at 8f7c0db8, with no error. The predicate now asks both
sides one question: is the served `optimal_control` the library body, held when `mfgarchon.core.hamiltonian` was
imported, bound to its own instance?

The oracle is the solve, not the predicate's value. Each form describes H = 1.5|p|^2 - 0.5 m on a
`QuadraticControlCost(1)`, which is the library `SeparableHamiltonian(QuadraticControlCost(1/3))`. Where the predicate
fails, the default scheme takes the control as a velocity and solves as that library H, and FP-SL, which forms
-c*grad(U) itself, refuses. Where it holds, FP-SL solves as that library H. 1-D no-flux, 21 nodes.
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


def _cost_class_replaced(monkeypatch):
    for name in COST_METHODS:
        monkeypatch.setattr(QuadraticControlCost, name, _steep_cost_body(name))
    return SeparableHamiltonian(QuadraticControlCost(control_cost=1.0), coupling=_coupling)


def _h_class_replaced(monkeypatch):
    for name in H_METHODS:
        monkeypatch.setattr(SeparableHamiltonian, name, _steep_h_body(name))
    return SeparableHamiltonian(QuadraticControlCost(control_cost=1.0), coupling=_coupling)


def _cost_subclass(monkeypatch):
    cls = type("SteepCost", (QuadraticControlCost,), {name: _steep_cost_body(name) for name in COST_METHODS})
    return SeparableHamiltonian(cls(control_cost=1.0), coupling=_coupling)


def _h_subclass(monkeypatch):
    cls = type("SteepH", (SeparableHamiltonian,), {name: _steep_h_body(name) for name in H_METHODS})
    return cls(QuadraticControlCost(control_cost=1.0), coupling=_coupling)


#: Forms whose served control is not the library's: the owner check, the served read on the cost, the library body held
#: at import on each side, and the two subclass controls (#2900), which read False at the base too.
NOT_THE_LIBRARYS = {
    "another_costs_library_methods_on_the_instance": _another_costs_library_methods,
    "partials_on_the_cost_instance": _partials,
    "QuadraticControlCost_replaced_on_the_class": _cost_class_replaced,
    "SeparableHamiltonian_replaced_on_the_class": _h_class_replaced,
    "cost_subclass_control": _cost_subclass,
    "hamiltonian_subclass_control": _h_subclass,
}


@pytest.mark.parametrize("form", sorted(NOT_THE_LIBRARYS))
def test_a_control_that_is_not_the_librarys_drives_the_fp_itself(form, monkeypatch):
    """The default scheme solves as the library H the form describes, and FP-SL, which would form -c*grad(U) with
    c = 1, refuses."""
    hamiltonian = NOT_THE_LIBRARYS[form](monkeypatch)
    np.testing.assert_allclose(_default(hamiltonian), _default(STEEP_H), rtol=0, atol=1e-12)
    with pytest.raises(NotImplementedError, match=r"^FPSLSolver\.solve_fp_system\(potential_field=U\), run on "):
        _fp_sl(hamiltonian)


def test_a_property_serving_the_library_control_bound_to_its_cost_is_the_librarys():
    """A cost subclass whose property returns the library method bound to the cost serves the library's control: FP-SL
    takes it and solves as the library H. At 8f7c0db8 the type attribute, a property, read False and FP-SL refused."""
    cls = type(
        "SelfBound",
        (QuadraticControlCost,),
        {name: property(lambda self, name=name: types.MethodType(COST_LIBRARY[name], self)) for name in COST_METHODS},
    )
    hamiltonian = SeparableHamiltonian(cls(control_cost=1 / 3), coupling=_coupling)
    np.testing.assert_allclose(_fp_sl(hamiltonian), _fp_sl(STEEP_H), rtol=0, atol=1e-12)

"""A SeparableHamiltonian whose optimal_control is its own is driven by it, not by -c*grad(U) (#2900).

The FP drift of a smooth separable H was formed as -c*grad(U), its control cost's, by any solver handed the value
function, so a subclass that redefined H solved with the base class's drift on problem.solve()'s default scheme, with
no error. The oracle is the library SeparableHamiltonian for the same H: the subclass must solve as it does.
"""

from __future__ import annotations

import functools
import logging
import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.types import NumericalScheme


def _coupling(m):
    return 0.5 * m


def _steep_control(self, t, x, p, m):
    return -3.0 * np.asarray(p, dtype=float)


class _Steep(SeparableHamiltonian):
    """H = 1.5|p|^2 - f(m) on a control_cost=1 base, whose control cost's drift is -grad(U), not -3 grad(U)."""

    def __call__(self, t, x, p, m):
        return 1.5 * np.sum(np.asarray(p, dtype=float) ** 2, axis=-1) - _coupling(np.asarray(m, dtype=float))

    def dp(self, t, x, p, m):
        return 3.0 * np.asarray(p, dtype=float)


class _OwnControl(_Steep):
    optimal_control = _steep_control


class _InheritsOwnControl(_OwnControl):
    pass


class _WrappedOwnControl(_Steep):
    optimal_control = functools.wraps(_steep_control)(
        lambda self, *args, **kwargs: _steep_control(self, *args, **kwargs)
    )


def _assigned_on_the_instance():
    hamiltonian = _Steep(QuadraticControlCost(control_cost=1.0), coupling=_coupling)
    hamiltonian.optimal_control = functools.partial(_steep_control, hamiltonian)
    return hamiltonian


OWN_CONTROL = {
    "on_the_class": lambda: _OwnControl(QuadraticControlCost(control_cost=1.0), coupling=_coupling),
    "inherited_from_a_subclass": lambda: _InheritsOwnControl(
        QuadraticControlCost(control_cost=1.0), coupling=_coupling
    ),
    "wrapped_with_functools_wraps": lambda: _WrappedOwnControl(
        QuadraticControlCost(control_cost=1.0), coupling=_coupling
    ),
    "assigned_on_the_instance": _assigned_on_the_instance,
}


def _library():
    return SeparableHamiltonian(QuadraticControlCost(control_cost=1 / 3), coupling=_coupling)


NODES = np.linspace(0.0, 1.0, 21)
MASS = np.trapezoid(0.2 + np.exp(-40 * (NODES - 0.3) ** 2), NODES)  # the grid's measure, so the mass is 1


def _solve(hamiltonian, **solver):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(
        model=Model(hamiltonian=hamiltonian, volatility=0.3),
        domain=grid,
        conditions=Conditions(
            m_initial=lambda x: (0.2 + np.exp(-40 * (x[0] - 0.3) ** 2)) / MASS,
            u_terminal=lambda x: 2.0 * (x[0] - 0.8) ** 2,
            T=0.5,
        ),
        Nt=10,
    )
    if "fp_solver" in solver:
        solver = {"hjb_solver": HJBFDMSolver(problem), "fp_solver": solver["fp_solver"](problem)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.CRITICAL)
        try:
            result = problem.solve(max_iterations=6, tolerance=1e-12, verbose=False, **solver)
        finally:
            logging.disable(logging.NOTSET)
    return np.asarray(result.U), np.asarray(result.M)


@pytest.mark.parametrize("own", sorted(OWN_CONTROL))
def test_an_own_optimal_control_solves_as_the_library_hamiltonian_it_describes(own):
    """On the default scheme the subclass solves as the library H for the same function: 7.411e-01 off in M at
    6f02506c, where its optimal_control was never called, wherever it was given."""
    U_ref, M_ref = _solve(_library())
    U, M = _solve(OWN_CONTROL[own]())
    np.testing.assert_allclose(U, U_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(M, M_ref, rtol=0, atol=1e-12)


def test_a_solver_that_forms_minus_c_grad_u_refuses_it_and_its_advice_runs():
    """FP-SL forms the drift as -c*grad(U), so it refuses rather than solve with the control cost's drift (9.444e-01 off
    at 6f02506c). The refusal names the schemes that read optimal_control; each solves as the library H on that scheme."""
    with pytest.raises(NotImplementedError, match=r"FPSLSolver: _OwnControl\.optimal_control is its own") as refused:
        _solve(OWN_CONTROL["on_the_class"](), scheme=NumericalScheme.SL_LINEAR)
    for scheme in ("FDM_UPWIND", "FVM_UPWIND"):
        assert scheme in str(refused.value)
        _, M_ref = _solve(_library(), scheme=NumericalScheme[scheme])
        _, M = _solve(OWN_CONTROL["on_the_class"](), scheme=NumericalScheme[scheme])
        np.testing.assert_allclose(M, M_ref, rtol=0, atol=1e-12, err_msg=scheme)


def test_particle_still_reads_an_own_optimal_control():
    """The particle solver reads optimal_control itself, so it keeps solving such a subclass: the scalar it builds is for a
    non-separable H only."""
    particles = functools.partial(FPParticleSolver, num_particles=4000, seed=7)
    _, M_ref = _solve(_library(), fp_solver=particles)
    _, M = _solve(OWN_CONTROL["on_the_class"](), fp_solver=particles)
    np.testing.assert_allclose(M, M_ref, rtol=0, atol=1e-12)

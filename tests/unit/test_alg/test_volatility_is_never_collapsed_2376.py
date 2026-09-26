"""#2376: a non-scalar volatility is never collapsed to a representative scalar.

``MFGProblem`` once reduced an array volatility to its mean and a callable one to the literal
``1.0`` (a factor of 400 in ``D`` for ``sigma = 0.05``), and every reader of that scalar solved a
different problem without a word. There is no such scalar any more: ``problem.volatility`` is held
as supplied, and each consumer either reads it or refuses it by name.

ADMISSION (#2227). Class 3, a pin of the fix: each test below reddens when the refusal or the read
it names is taken out, which is the measurement its assertion is. The fixture is 1-D where the
property is a scalar contract, and 2-D only for the tensor rows, which have no 1-D reader.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

N = 11
RAMP = np.linspace(0.2, 0.6, N)  # mean 0.4: a consumer averaging it would look plausible


def _components(dim=1):
    return MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        m_initial=lambda x: np.exp(-10 * np.sum((np.atleast_1d(np.asarray(x)) - 0.5) ** 2, axis=-1)),
        u_terminal=lambda x: np.zeros(np.asarray(x).shape[:-1]) if dim > 1 else 0.0 * np.asarray(x),
    )


def _problem(**volatility):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[N], boundary_conditions=no_flux_bc(dimension=1))
    return MFGProblem(geometry=grid, T=0.2, Nt=4, components=_components(), **volatility)


def _problem_2d(**volatility):
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[6, 5], boundary_conditions=no_flux_bc(dimension=2)
    )
    return MFGProblem(geometry=grid, T=0.2, Nt=4, components=_components(dim=2), **volatility)


FIELD = {"volatility": RAMP, "volatility_kind": "field"}
CALLABLE = {"volatility": lambda t, x, m: 0.05}


@pytest.mark.parametrize("volatility", [FIELD, CALLABLE], ids=["field", "callable"])
def test_scalar_only_hjb_solvers_refuse_by_name(volatility):
    """WENO and SL each read one constant volatility; a field or callable is refused, naming the solver."""
    from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver, HJBWENOSolver

    problem = _problem(**volatility)
    with pytest.raises(NotImplementedError, match="HJBWENOSolver uses one scalar volatility"):
        HJBWENOSolver(problem)
    with pytest.raises(NotImplementedError, match="HJBSemiLagrangianSolver reads one constant volatility"):
        HJBSemiLagrangianSolver(problem)


def test_the_adjoint_consistent_provider_refuses_a_field():
    """The provider squares one scalar; the BC state now carries the problem's volatility as supplied."""
    from mfgarchon.geometry.boundary import AdjointConsistentProvider

    problem = _problem(**FIELD)
    state = {"m_current": np.ones(N), "geometry": problem.geometry, "sigma": problem.volatility}
    with pytest.raises(NotImplementedError, match="AdjointConsistentProvider uses one scalar volatility"):
        AdjointConsistentProvider(side="left").compute(state)
    # Control: the same state with a scalar computes.
    state["sigma"] = 0.4
    assert np.isfinite(AdjointConsistentProvider(side="left").compute(state))


def test_the_hjb_residual_refuses_a_callable_it_cannot_evaluate():
    """The 1-D residual has no (t, m) to evaluate a callable at; the per-timestep driver does."""
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import _volatility_at_n

    with pytest.raises(NotImplementedError, match="evaluated at t_n"):
        _volatility_at_n(_problem(**CALLABLE), None)
    np.testing.assert_array_equal(_volatility_at_n(_problem(**FIELD), None), RAMP)


def test_the_strict_adjoint_fp_step_refuses_a_callable():
    """Before #2376 the step took problem.sigma, the literal 1.0 for any callable."""
    from scipy import sparse

    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    problem = _problem(**CALLABLE)
    solver = FPFDMSolver(problem)
    with pytest.raises(NotImplementedError, match="adjoint_mode='off'"):
        solver.solve_fp_step_adjoint_mode(np.ones(N) / N, sparse.csr_matrix((N, N)))


def test_gfdm_falls_back_to_the_field_not_to_one():
    """HJBGFDMSolver's pre-solve volatility is the field at the collocation points, never 1.0.

    Before #2376 ``_get_sigma_value(None)`` returned ``float(getattr(problem, "sigma", 1.0))``: the
    mean for an array, and 1.0 for a callable problem -- reached by a standalone Howard solve and by
    the construction-time LLF base.
    """
    from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver

    points = np.linspace(0.0, 1.0, N).reshape(-1, 1)
    solver = HJBGFDMSolver(_problem(**FIELD), collocation_points=points)
    np.testing.assert_allclose(solver._get_sigma_value(None), RAMP, rtol=0, atol=1e-15)
    assert solver._get_sigma_value(3) == pytest.approx(RAMP[3], abs=1e-15)
    with pytest.raises(NotImplementedError, match="space-only volatility callable"):
        HJBGFDMSolver(_problem(**CALLABLE), collocation_points=points)._get_sigma_value(None)


def test_fvm_refuses_the_problems_tensor_even_when_its_entries_are_equal():
    """An all-equal matrix passes FVM's constancy check and would be read as the scalar s."""
    from mfgarchon.alg.numerical.fp_solvers import FPFVMSolver

    problem = _problem_2d(volatility=np.full((2, 2), 0.3), volatility_kind="tensor")
    with pytest.raises(NotImplementedError, match="the problem's is a tensor"):
        FPFVMSolver(problem)._scalar_diffusion(None)


def test_the_pairing_guard_compares_arrays_by_content():
    """Two problems whose array volatilities share a mean are different problems.

    Before #2376 the guard compared each problem's collapsed mean, so this pair passed.
    """
    from mfgarchon.alg.numerical.coupling.base_mfg import assert_paired_solver_sigma

    class _Solver:
        def __init__(self, problem):
            self.problem = problem

    a = _problem(**FIELD)
    b = _problem(volatility=RAMP[::-1].copy(), volatility_kind="field")  # same mean, other field
    assert float(np.mean(a.volatility)) == pytest.approx(float(np.mean(b.volatility)))
    with pytest.raises(ValueError, match="different volatility"):
        assert_paired_solver_sigma(_Solver(a), _Solver(b), "test")
    assert_paired_solver_sigma(_Solver(a), _Solver(a), "test")  # control: the same problem passes


def test_distinct_per_population_volatilities_are_refused():
    """No multi-population solver reads a population's own volatility; all were solved at population 0's."""
    from mfgarchon.extensions.multi_population import MultiPopulationMFGProblem

    kwargs = {
        "num_populations": 2,
        "spatial_bounds": [(0.0, 1.0)],
        "spatial_discretization": [N - 1],
        "T": 0.2,
        "Nt": 4,
        "components": _components(),
    }
    with pytest.raises(NotImplementedError, match="population 0's"):
        MultiPopulationMFGProblem(volatility=[0.1, 0.3], **kwargs)
    assert MultiPopulationMFGProblem(volatility=[0.2, 0.2], **kwargs).volatility == 0.2

"""The outer-tolerance verdict refuses a non-finite change instead of judging around it (#2578).

`check_convergence_criteria` takes ``max`` over the two fields, and ``max(1e-7, nan)`` is ``1e-7``: a NaN
in the second field was dropped, so ``(1e-7, nan, 1e-7, nan, 1e-5)`` reported converged while the same NaN
in the first field did not. The verdict's owner refuses, naming the value.

The multi-field iterators had the same defect one level up: each took builtin ``max`` over its
populations, nodes or regimes before calling the owner, so a NaN in any field after the first never
reached it. They aggregate through `worst_sweep_change`, which refuses and names the field's index.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, Model
from mfgarchon.alg.numerical.coupling.fixed_point_utils import check_convergence_criteria
from mfgarchon.alg.numerical.coupling.graph_coupling import AdjacencyCoupling
from mfgarchon.alg.numerical.coupling.graph_mfg_solver import GraphMFGSolver
from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
from mfgarchon.alg.numerical.coupling.regime_switching_iterator import RegimeSwitchingIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.core.regime_switching import RegimeSwitchingConfig
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.convergence import worst_sweep_change

FIELDS = ("l2distu_rel", "l2distm_rel", "l2distu_abs", "l2distm_abs")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("field", FIELDS)
def test_a_non_finite_change_in_any_field_is_refused_by_name(field, bad):
    changes = dict.fromkeys(FIELDS, 1e-7)
    changes[field] = bad
    with pytest.raises(ValueError, match=rf"{field}={bad}"):
        check_convergence_criteria(*changes.values(), 1e-5, 1e-6)


def test_a_nan_density_change_no_longer_reads_as_converged():
    """The defect in both argument orders: the second field used to be dropped, the first did not."""
    nan = float("nan")
    for args in ((1e-7, nan, 1e-7, nan, 1e-5), (nan, 1e-7, nan, 1e-7, 1e-5)):
        with pytest.raises(ValueError, match="must be finite"):
            check_convergence_criteria(*args)
    assert check_convergence_criteria(1e-7, 1e-8, 3.0, 2.0, 1e-6) == (True, "Converged: Rel err 1.0e-07 < tol 1.0e-06")


def test_the_worst_change_is_each_values_max_over_fields_and_refuses_naming_the_field():
    per_field = [
        {"l2distu_abs": 1.0, "l2distu_rel": 4e-7, "l2distm_abs": 2.0, "l2distm_rel": 1e-7},
        {"l2distu_abs": 3.0, "l2distu_rel": 2e-7, "l2distm_abs": 0.5, "l2distm_rel": 5e-7},
    ]
    assert worst_sweep_change(per_field, "node") == {
        "l2distu_abs": 3.0,
        "l2distu_rel": 4e-7,
        "l2distm_abs": 2.0,
        "l2distm_rel": 5e-7,
    }
    per_field[1]["l2distm_rel"] = float("nan")
    with pytest.raises(ValueError, match=r"got l2distm_rel=nan in node k=1 "):
        worst_sweep_change(per_field, "node")


def _problem() -> MFGProblem:
    """A 1-D problem with a uniform density and no coupling or terminal cost, so neither field moves."""
    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: 0.0 * np.asarray(m),
                coupling_dm=lambda m: 0.0 * np.asarray(m),
            ),
            volatility=0.3,
        ),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[8], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(
            m_initial=lambda x: 1.0 + 0.0 * np.asarray(x), u_terminal=lambda x: 0.0 * np.asarray(x), T=0.2
        ),
        Nt=4,
    )


def _fp(problem: MFGProblem, nan_at_sweep: int | None) -> FPFDMSolver:
    """An FDM FP solver whose density is NaN at sweep ``nan_at_sweep`` (1-based), or never."""
    fp = FPFDMSolver(problem)
    solve, calls = fp.solve_fp_system, [0]

    def patched(*args, **kwargs):
        calls[0] += 1
        m = solve(*args, **kwargs)
        return np.full_like(m, np.nan) if calls[0] == nan_at_sweep else m

    fp.solve_fp_system = patched
    return fp


#: The first sweep each iterator judges. A regime's first sweep compares the 1-D initial density with a
#: trajectory and is not judged; a NaN density earlier than the judged sweep would reach the next HJB solve
#: and stop the loop as a diverged value function instead.
_JUDGED_SWEEP = {"population": 1, "node": 1, "regime": 2}


def _converged(kind: str, bad: int | None) -> bool:
    """Solve a three-field ``kind`` problem up to its first judged sweep, with field ``bad``'s density NaN there."""
    sweep = _JUDGED_SWEEP[kind]
    ps = [_problem() for _ in range(3)]
    hjbs = [HJBFDMSolver(p) for p in ps]
    fps = [_fp(p, sweep if k == bad else None) for k, p in enumerate(ps)]
    if kind == "population":
        multi = MultiPopulationProblem(populations=ps, population_names=["a", "b", "c"])
        return MultiPopulationIterator(multi, hjbs, fps).solve(max_iterations=sweep, tolerance=1e3).converged
    if kind == "node":
        coupling = AdjacencyCoupling(np.ones((3, 3)) - np.eye(3), alpha=0.0, beta=0.0)
        return GraphMFGSolver(ps, coupling, hjbs, fps, max_iterations=sweep, tolerance=1e3).solve().converged
    config = RegimeSwitchingConfig(transition_matrix=np.zeros((3, 3)))
    return RegimeSwitchingIterator(ps, config, hjbs, fps, max_iterations=sweep, tolerance=1e3).solve().converged


@pytest.mark.parametrize("bad", [0, 1])
@pytest.mark.parametrize("kind", sorted(_JUDGED_SWEEP))
def test_a_nan_density_in_any_field_of_a_multi_field_solve_is_refused_by_index(kind, bad):
    """Index 1 is the defect: builtin ``max`` over fields dropped it, and the solve reported converged."""
    with pytest.raises(ValueError, match=rf"got l2distm_abs=nan, l2distm_rel=nan in {kind} k={bad} "):
        _converged(kind, bad)


@pytest.mark.parametrize("kind", sorted(_JUDGED_SWEEP))
def test_the_same_solve_without_a_nan_converges_at_that_sweep(kind):
    """The control: the sweep the pins break is one the solve judges and passes."""
    assert _converged(kind, None) is True

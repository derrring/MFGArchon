"""NewtonMFGSolver decides through the outer-tolerance owners, at both of its stops (#2565).

Newton stops on the Picard residual Phi(x) - x: ``F_HJB = U_new - U``, ``F_FP = M_new - M``. That is the
map's output against its input, the quantity C1 governs (docs/user/CONVENTIONS.md § 9). Before #2565 the
check after the warm-up compared an unscaled 2-norm of it with ``solve(tolerance=)``, and phase 2 stopped
on the same unscaled norm against the constructor's ``newton_tolerance``, so ``tolerance`` was never read
there. Both stops now go through `sweep_change` and `check_convergence_criteria`. The deprecated
``newton_tolerance`` still acts during its window, as ``absolute_tolerance``.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling import NewtonMFGSolver
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.convergence import sweep_change
from mfgarchon.utils.numerical.nonlinear_solvers import NewtonSolver

N, NT = 9, 4


def _problem() -> MFGProblem:
    x = np.linspace(0.0, 1.0, N)
    bump = lambda z: np.exp(-20 * (np.asarray(z, dtype=float) - 0.4) ** 2)  # noqa: E731
    scale = 1.0 / np.trapezoid(bump(x), x)
    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: 0.5 * np.asarray(m),
                coupling_dm=lambda m: 0.5 + 0.0 * np.asarray(m),
            ),
            volatility=0.3,
        ),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[N], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(
            m_initial=lambda z: scale * bump(z), u_terminal=lambda z: 0.0 * np.asarray(z, dtype=float), T=0.3
        ),
        Nt=NT,
    )


def _solve(solve_kw: dict, **ctor_kw):
    problem = _problem()
    solver = NewtonMFGSolver(problem, HJBFDMSolver(problem), FPFDMSolver(problem), **ctor_kw)
    U, M, info = solver.solve(max_iterations=30, verbose=False, **solve_kw)
    _, F_HJB, F_FP = solver.mfg_residual.compute_residual(U, M, return_components=True)
    change = sweep_change(U + F_HJB, U, M + F_FP, M, problem.spatial_measure().integrate, problem.dt)
    return U, M, info, max(change["l2distu_rel"], change["l2distm_rel"])


def test_the_outer_tolerance_decides_newtons_own_iterations():
    """A looser tolerance stops Newton earlier, and a reported convergence holds below its bound.

    Before #2565 phase 2 never read `tolerance`: at 1e-13 it reported converged at a relative change of
    3.4e-13, above the bound, and took 4 Newton iterations at 1e-5 as at 1e-13 (measured at `455b39bf`).
    """
    _, _, loose, loose_change = _solve({"tolerance": 1e-5})
    _, _, tight, tight_change = _solve({"tolerance": 1e-13})
    assert loose["converged"]
    assert tight["converged"]
    assert loose_change < 1e-5
    assert tight_change < 1e-13
    assert loose["newton_iterations"] < tight["newton_iterations"]


def test_the_warm_up_stops_on_the_relative_change_in_the_problems_measure():
    """After three warm-up sweeps the relative change is 0.374 and the unscaled 2-norm 0.880: at
    tolerance 0.5 the owners' verdict stops there, where the unscaled norm did not."""
    _, _, info, change = _solve({"tolerance": 0.5}, picard_warmup=3)
    assert info["convergence_reason"] == "Converged during Picard warm-up"
    assert change < 0.5


def test_absolute_tolerance_reaches_the_verdict():
    """The absolute bound is an extra requirement on the same change, at both stops.

    After three warm-up sweeps the relative change, 0.374, meets tolerance 0.5, and the absolute change,
    0.082 (dt-weighted, in the problem's measure), then decides the warm-up stop: 0.1 stops it, 0.05 does
    not. In phase 2 an absolute 1e-10 beside tolerance 1e-3 takes more Newton iterations than none.
    """
    _, _, met, _ = _solve({"tolerance": 0.5, "absolute_tolerance": 0.1}, picard_warmup=3)
    assert met["convergence_reason"] == "Converged during Picard warm-up"
    _, _, unmet, _ = _solve({"tolerance": 0.5, "absolute_tolerance": 0.05}, picard_warmup=3, newton_max_iterations=1)
    assert unmet["convergence_reason"] != "Converged during Picard warm-up"
    _, _, relative_only, _ = _solve({"tolerance": 1e-3})
    _, _, with_absolute, _ = _solve({"tolerance": 1e-3, "absolute_tolerance": 1e-10})
    assert relative_only["converged"]
    assert with_absolute["converged"]
    assert with_absolute["newton_iterations"] > relative_only["newton_iterations"]


@pytest.mark.parametrize(
    "bound", [0.5, 0.1, 0.05], ids=["met_at_warm_up", "met_by_absolute_only", "not_met_at_warm_up"]
)
def test_the_deprecated_newton_tolerance_acts_as_absolute_tolerance(bound):
    """The equivalence test of the deprecation policy: newton_tolerance=x solves as absolute_tolerance=x."""
    with pytest.warns(DeprecationWarning, match=r"dt-weighted space-time L2 change, not an unscaled 2-norm"):
        U_old, M_old, info_old, _ = _solve(
            {"tolerance": 0.5}, newton_tolerance=bound, picard_warmup=3, newton_max_iterations=1
        )
    U_new, M_new, info_new, _ = _solve(
        {"tolerance": 0.5, "absolute_tolerance": bound}, picard_warmup=3, newton_max_iterations=1
    )
    np.testing.assert_array_equal(U_old, U_new)
    np.testing.assert_array_equal(M_old, M_new)
    assert info_old["convergence_reason"] == info_new["convergence_reason"]
    assert info_old["total_iterations"] == info_new["total_iterations"]


def test_an_assigned_newton_tolerance_acts_as_well():
    """The public attribute is read at solve(), so assigning it is not a silent no-op."""
    problem = _problem()
    solver = NewtonMFGSolver(
        problem, HJBFDMSolver(problem), FPFDMSolver(problem), picard_warmup=3, newton_max_iterations=1
    )
    solver.newton_tolerance = 0.05
    _, _, info = solver.solve(max_iterations=30, tolerance=0.5, verbose=False)
    assert info["convergence_reason"] != "Converged during Picard warm-up"


def test_newton_tolerance_and_absolute_tolerance_together_are_refused():
    with pytest.warns(DeprecationWarning, match="newton_tolerance"), pytest.raises(ValueError, match="both given"):
        _solve({"absolute_tolerance": 1e-6}, newton_tolerance=1e-6)


def test_a_newton_solver_given_no_verdict_keeps_its_norm_test():
    """`converged` is opt-in. Without it NewtonSolver stops at the first iterate whose residual norm is under
    its tolerance; with it, the verdict decides, and it sees F(x) itself."""
    F = lambda x: x**2 - 2.0  # noqa: E731
    # The residuals from 1.0 are 0.25, 6.9e-3, 6.0e-6, 4.5e-12: at 1e-6 the stop falls between the last two,
    # so a threshold moved by a factor of 10 stops one iterate early.
    x, info = NewtonSolver(tolerance=1e-6, line_search=False).solve(F, 1.0)
    assert info.converged
    assert info.residual_history[-1] < 1e-6 <= info.residual_history[-2]
    assert abs(x - np.sqrt(2.0)) < 1e-9

    seen = []

    def verdict(x_k, F_k):
        seen.append((float(x_k), float(F_k)))
        return len(seen) == 2, "second iterate"

    _, info = NewtonSolver(tolerance=1e-10, line_search=False).solve(F, 1.0, converged=verdict)
    assert info.iterations == 2
    assert all(f == x_k**2 - 2.0 for x_k, f in seen)

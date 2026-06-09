"""Fast Newton↔Picard agreement guards for the MFG coupling layer (Issue #1233).

These are the lightweight CI counterparts to ``test_newton_mfg_solver.py`` (which is
marked ``slow`` because its realistic-grid Newton runs are expensive). The slow suite
is skipped on PRs, so the drift-convention regression that #1233 fixed went unnoticed
on ``main``. The tests here run on a tiny grid and stay un-marked so every PR exercises:

1. ``MFGResidual`` evaluates to ~0 at the Picard fixed point — i.e. the Newton residual
   uses the *same* FP drift/potential convention as the Picard ``FixedPointIterator``.
   This is the precise guard for the single-sourced ``resolve_fp_drift_kwargs``: a future
   private re-fork of the drift convention makes ``||F(Picard soln)||`` blow up here.
2. The full Newton solve converges to the same solution as Picard (with an adequate
   warmup that lands in the physical basin).
"""

from __future__ import annotations

import numpy as np

from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.alg.numerical.coupling.mfg_residual import MFGResidual
from mfgarchon.alg.numerical.coupling.newton_mfg_solver import NewtonMFGSolver
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _small_problem():
    """Tiny 1D LQ-ish MFG: small enough that the O(N) FD-Jacobian Newton runs in seconds."""
    geometry = TensorProductGrid(bounds=[(0.0, 1.0)], boundary_conditions=no_flux_bc(dimension=1), Nx_points=[6])
    components = MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: m,
            coupling_dm=lambda m: 1.0,
        ),
    )
    return MFGProblem(geometry=geometry, T=0.15, Nt=3, sigma=0.3, components=components)


def _picard_solution(problem):
    picard = FixedPointIterator(
        problem,
        hjb_solver=HJBFDMSolver(problem),
        fp_solver=FPFDMSolver(problem),
        relaxation=0.5,
    )
    result = picard.solve(max_iterations=80, tolerance=1e-6, verbose=False)
    return result[0], result[1]


def test_residual_consistent_with_picard_fixed_point():
    """The Newton residual must vanish at the Picard fixed point (Issue #1233).

    Before the fix, ``MFGResidual`` passed the value function as ``drift_field`` — which
    the v0.18.6 rename redefined as the *velocity* — so ``||F_FP(Picard soln)||`` was
    O(10), the Newton and Picard solvers chased inconsistent roots, and the comparison
    test failed by ~99.5%. With the drift convention single-sourced, the residual is ~0.
    """
    problem = _small_problem()
    U_picard, M_picard = _picard_solution(problem)

    residual = MFGResidual(problem, HJBFDMSolver(problem), FPFDMSolver(problem))
    _, F_hjb, F_fp = residual.compute_residual(U_picard, M_picard, return_components=True)

    # HJB residual floor is set by the inner HJB solver tolerance (~1e-6); the FP residual
    # is the one that regresses on a drift-convention divergence — pin it tightly.
    assert np.linalg.norm(F_fp) < 1e-4, (
        f"FP residual does not vanish at the Picard fixed point: ||F_FP||={np.linalg.norm(F_fp):.3e}. "
        "The Newton MFGResidual FP drift/potential convention has diverged from the Picard "
        "FixedPointIterator (resolve_fp_drift_kwargs no longer single-sourced)."
    )
    assert np.linalg.norm(F_hjb) < 1e-3


def test_newton_matches_picard_small():
    """End-to-end: Newton converges to the Picard solution on a tiny grid (Issue #1233).

    Uses an adequate Picard warmup so the iterate enters the basin of the physical
    equilibrium before Newton (a local root-finder) takes over.
    """
    problem = _small_problem()
    U_picard, M_picard = _picard_solution(problem)

    newton = NewtonMFGSolver(
        problem,
        HJBFDMSolver(problem),
        FPFDMSolver(problem),
        picard_warmup=5,
        newton_max_iterations=3,
        line_search=True,
    )
    U_newton, M_newton, _info = newton.solve(max_iterations=8, tolerance=1e-5, verbose=False)

    u_diff = np.linalg.norm(U_newton - U_picard) / (np.linalg.norm(U_picard) + 1e-10)
    m_diff = np.linalg.norm(M_newton - M_picard) / (np.linalg.norm(M_picard) + 1e-10)
    assert u_diff < 0.05, f"Newton U differs from Picard by {u_diff * 100:.1f}%"
    assert m_diff < 0.05, f"Newton M differs from Picard by {m_diff * 100:.1f}%"


def test_default_uses_finite_difference_jacobian():
    """NewtonMFGSolver defaults to the FD-Jacobian path (Issue #1233).

    The MFG residual wraps black-box numpy/scipy solvers that JAX cannot trace, so the
    autodiff path is off by default to avoid a guaranteed TracerArrayConversionError +
    fallback warning on every run.
    """
    problem = _small_problem()
    newton = NewtonMFGSolver(problem, HJBFDMSolver(problem), FPFDMSolver(problem))
    assert newton.use_jax_autodiff is False

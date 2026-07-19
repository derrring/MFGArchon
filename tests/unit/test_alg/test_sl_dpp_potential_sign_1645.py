"""The DPP path's (V, f) sign, pinned against HJB-FDM (Issue #1645).

`TestSLHJBConsistency::test_matches_fdm_with_potential` pins the same sign on the
H-BASED CHARACTERISTIC path (Issue #575/#1413), but it builds a `QuadraticControlCost`,
whose `is_smooth()` is True, so `_use_dpp` is False and the Lagrangian is never called.
The L-based DPP path had no equivalent, and that asymmetry is why the fork survived:
`SeparableLagrangian.__call__` returned ``L_ctrl + V + f`` while its own
`evaluate_hamiltonian` returned ``H_ctrl + V + f``, so the class was not self-conjugate.

Reaching the DPP path requires a NON-SMOOTH control cost, hence L1 here. Measured on the
pristine tree the error was ``V + O(h)`` -- mesh-independent and exactly linear in the
potential amplitude (V=0/1/2/5 gave 0.0054 / 1.0054 / 2.0054 / 5.0054), i.e. a wrong limit
sitting on top of an otherwise converging scheme, not a discretization error.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver, HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import L1ControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _problem(v_amp: float, nx: int, nt: int) -> MFGProblem:
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=no_flux_bc(dimension=1))
    hamiltonian = SeparableHamiltonian(
        control_cost=L1ControlCost(lambda_=1.0),
        potential=lambda x, t=0.0: v_amp * np.ones_like(np.atleast_1d(x)).squeeze(),
    )
    return MFGProblem(
        model=Model(hamiltonian=hamiltonian, sigma=0.15),
        domain=grid,
        conditions=Conditions(u_terminal=lambda x: (x - 0.5) ** 2, m_initial=lambda x: 1.0, T=0.5),
        Nt=nt,
    )


def _solve(solver_cls, v_amp: float, nx: int, nt: int) -> np.ndarray:
    problem = _problem(v_amp, nx, nt)
    xs = problem.geometry.coordinates[0]
    density = np.tile(1.0 + 0.5 * np.exp(-30.0 * (xs - 0.35) ** 2), (nt + 1, 1))
    return solver_cls(problem).solve_hjb_system(density, (xs - 0.5) ** 2, U_coupling_prev=np.zeros((nt + 1, nx)))[0]


def test_the_dpp_path_is_actually_exercised():
    """Guard the guard: if this ever goes False the tests below stop testing the L path."""
    solver = HJBSemiLagrangianSolver(_problem(2.0, 51, 40))
    assert solver._use_dpp, "L1 must route through the DPP path, else the (V,f) sign below is untested"
    assert solver.problem.lagrangian_class is not None


@pytest.mark.parametrize("v_amp", [0.0, 1.0, 2.0, 5.0])
def test_dpp_error_does_not_grow_with_the_potential(v_amp):
    """The load-bearing assertion: the error must be INDEPENDENT of the potential amplitude.

    Pre-fix the error was ``V + O(h)`` -- so this fails at V=1, 2 and 5 and passes at V=0, which
    is exactly the discrimination a single-amplitude test would miss.

    The grid is deliberately COARSE. Because the defect is mesh-INDEPENDENT, refining buys no
    detection power: measured on the pristine tree the V=2 minus V=0 spread is 2.0000 at
    nx=41/nt=20, at 61/30 and at 81/40 alike, while the cost goes 2.6s -> 6.1s -> 11.1s. Paying
    for a finer grid here would slow the PR gate without making the test any sharper.
    """
    sl = _solve(HJBSemiLagrangianSolver, v_amp, nx=41, nt=20)
    fdm = _solve(HJBFDMSolver, v_amp, nx=41, nt=20)
    err = float(np.abs(sl - fdm).max())

    assert err < 0.05, (
        f"V={v_amp}: max|SL-FDM| = {err:.6f}. Pre-fix this was V + 0.0054, i.e. the DPP running "
        f"cost carried +V instead of -V (Issue #1645)."
    )


@pytest.mark.slow
def test_dpp_error_converges_under_refinement():
    """Convergence, not just smallness: the pre-fix error PLATEAUED (ratios 1.0018 / 1.0007).

    A tolerance at one resolution cannot distinguish a wrong limit from a merely-small error.

    Marked ``slow`` -- it needs genuinely fine grids to make a convergence statement, which the
    detection tests above do not. The PR gate keeps the discrimination; nightly keeps the rate.
    """
    errors = []
    for nx, nt in ((81, 40), (161, 80)):
        sl = _solve(HJBSemiLagrangianSolver, 2.0, nx, nt)
        fdm = _solve(HJBFDMSolver, 2.0, nx, nt)
        errors.append(float(np.abs(sl - fdm).max()))

    assert errors[1] < errors[0] / 2.0, f"error did not at least halve under refinement: {errors}"

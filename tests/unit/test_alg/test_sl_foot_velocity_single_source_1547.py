"""The SL characteristic foot must read dH/dp from the Hamiltonian (Issue #1547 / RFC #1574).

Every foot site in `HJBSemiLagrangianSolver` used to hardcode the QUADRATIC control-cost
velocity ``dH/dp = p/lambda``, while the Lax-Oleinik value term routed through
``eval_H_batch`` and evaluated the real H. For a Hamiltonian whose true dH/dp is not
p/lambda that hybrid solves without complaint and converges to the WRONG LIMIT.

`eval_dH_dp_batch` already owned dH/dp for `HJBFDMSolver` and `HJBGFDMSolver`; SL was the
only HJB solver in the family holding a private quadratic copy. The fix routes the feet
through the same owner, so this is a single-source consolidation and owes a pinning test
that fails if the fork reopens.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver, HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import CongestionHamiltonian, QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

GAMMA = 3.0


def _grid(nx: int) -> TensorProductGrid:
    return TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=no_flux_bc(dimension=1))


def _problem(hamiltonian, nx: int, nt: int) -> MFGProblem:
    return MFGProblem(
        model=Model(hamiltonian=hamiltonian, sigma=0.15),
        domain=_grid(nx),
        conditions=Conditions(u_terminal=lambda x: (x - 0.5) ** 2, m_initial=lambda x: 1.0, T=0.5),
        Nt=nt,
    )


def _frozen_density(problem, nt: int) -> np.ndarray:
    """A non-uniform, strictly positive frozen density, so c(m) genuinely varies in space."""
    xs = problem.geometry.coordinates[0]
    return np.tile(1.0 + 2.0 * np.exp(-30.0 * (xs - 0.35) ** 2), (nt + 1, 1))


def _solve(solver_cls, hamiltonian, nx: int, nt: int, **kwargs) -> np.ndarray:
    problem = _problem(hamiltonian, nx, nt)
    xs = problem.geometry.coordinates[0]
    u_terminal = (xs - 0.5) ** 2
    density = _frozen_density(problem, nt)
    solver = solver_cls(problem, **kwargs)
    return solver.solve_hjb_system(density, u_terminal, U_coupling_prev=np.zeros((nt + 1, nx)))


def _quadratic():
    return SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))


def _congestion(gamma: float = GAMMA):
    return CongestionHamiltonian(QuadraticControlCost(lambda_=1.0), congestion_factor=lambda m: 1.0 + gamma * m)


# --- the quadratic case must not move -------------------------------------------------


@pytest.mark.parametrize("control_cost", [1.0, 2.5])
@pytest.mark.parametrize("characteristic_solver", [None, "rk4"])
def test_quadratic_case_is_unchanged(control_cost, characteristic_solver):
    """For a quadratic cost the Hamiltonian's dH/dp IS p/lambda, so routing changes nothing.

    This is the byte-identity half of the consolidation: the fix may only repair the
    Hamiltonians the hardcode got wrong, never perturb the one it got right.
    """
    hamiltonian = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=control_cost))
    kwargs = {} if characteristic_solver is None else {"characteristic_solver": characteristic_solver}

    result = _solve(HJBSemiLagrangianSolver, hamiltonian, nx=41, nt=10, **kwargs)

    assert np.isfinite(result).all()
    # alpha* = -grad(u)/lambda exactly; the analytic dp returns p/lambda from the same float.
    solver = HJBSemiLagrangianSolver(_problem(hamiltonian, 41, 10), **kwargs)
    p_probe = np.linspace(-2.0, 2.0, 41).reshape(-1, 1)
    dp = solver._characteristic_foot_velocity(solver.x_grid.reshape(-1, 1), np.ones(41), p_probe, 0.0).reshape(-1)
    assert np.allclose(dp, p_probe.reshape(-1) / control_cost, rtol=0, atol=0), (
        "for a quadratic cost the routed foot velocity must equal p/lambda exactly"
    )


# --- the congestion case must now be right --------------------------------------------


def test_foot_velocity_carries_the_congestion_factor():
    """dH/dp = p / (lambda*c(m)); the hardcode dropped c(m) entirely."""
    hamiltonian = _congestion()
    solver = HJBSemiLagrangianSolver(_problem(hamiltonian, 41, 10))

    x = solver.x_grid.reshape(-1, 1)
    p = np.full((41, 1), 2.0)
    m = np.ones(41)

    vel = solver._characteristic_foot_velocity(x, m, p, 0.0).reshape(-1)
    expected = 2.0 / (1.0 + GAMMA * 1.0)  # lambda = 1

    assert np.allclose(vel, expected), f"foot velocity {vel[:3]} should be p/(lambda*c(m)) = {expected}"
    assert not np.allclose(vel, 2.0), "the hardcoded p/lambda value must no longer be used"


def test_congestion_solve_tracks_an_independent_fdm_reference():
    """The load-bearing assertion: SL must now converge to the same answer as HJB-FDM.

    HJB-FDM routes its drift through the same `eval_dH_dp_batch` owner and is therefore an
    independent check on the foot, not a restatement of it. Before the fix the SL error
    against this reference PLATEAUED under refinement (0.0766 -> 0.0788 -> 0.0798, ratio
    1.0) -- a wrong limit, not discretization error. Reverting any foot site to p/lambda
    reopens that plateau and fails this test.
    """
    nx, nt = 201, 160
    hamiltonian = _congestion()

    sl = _solve(HJBSemiLagrangianSolver, hamiltonian, nx, nt)[0]
    fdm = _solve(HJBFDMSolver, hamiltonian, nx, nt)[0]

    assert np.isfinite(sl).all()
    assert np.isfinite(fdm).all()
    assert np.abs(sl - fdm).max() < 5e-3, (
        f"SL deviates from the FDM reference by {np.abs(sl - fdm).max():.6f}; the "
        f"characteristic foot is not carrying dH/dp (pre-fix this was ~0.08 and did not "
        f"shrink under refinement)"
    )


def test_congestion_error_actually_decreases_under_refinement():
    """Convergence, not just closeness: the pre-fix error was flat in h, which a single
    tolerance at one resolution cannot distinguish from a merely-small wrong limit."""
    hamiltonian = _congestion()
    errors = []
    for nx, nt in ((51, 40), (101, 80), (201, 160)):
        sl = _solve(HJBSemiLagrangianSolver, hamiltonian, nx, nt)[0]
        fdm = _solve(HJBFDMSolver, hamiltonian, nx, nt)[0]
        errors.append(float(np.abs(sl - fdm).max()))

    assert errors[1] < errors[0] / 1.5, f"error did not fall on the first refinement: {errors}"
    assert errors[2] < errors[1] / 1.5, f"error did not fall on the second refinement: {errors}"


# --- the owner is shared, not re-forked ------------------------------------------------


def test_sl_uses_the_same_dh_dp_owner_as_the_other_hjb_solvers():
    """Consolidation pin: SL must consume `eval_dH_dp_batch`, the primitive FDM and GFDM use.

    Scope stated precisely: this asserts the module imports and calls the shared owner. It
    does not prevent someone reintroducing an inline quadratic expression at a call site --
    that case is caught numerically by the refinement test above, which is the check that
    matters.
    """
    import inspect

    from mfgarchon.alg.numerical.hjb_solvers import hjb_semi_lagrangian as sl_module

    source = inspect.getsource(sl_module)
    assert "eval_dH_dp_batch" in source, "SL no longer routes through the shared dH/dp owner"

    velocity_source = inspect.getsource(HJBSemiLagrangianSolver._characteristic_foot_velocity)
    assert "eval_dH_dp_batch" in velocity_source, (
        "the foot-velocity owner must call the shared primitive, not recompute dH/dp"
    )

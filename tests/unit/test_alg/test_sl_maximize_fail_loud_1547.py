"""Issue #1547 / RFC #1574 Phase 0: HJBSemiLagrangianSolver must fail loud on a MAXIMIZE control cost.

The SL characteristic-foot velocity dH/dp = p/lambda is hardcoded MINIMIZE-signed (departures
x - (grad_u/lambda)*dt, i.e. alpha* = -grad_u/lambda). A MAXIMIZE control cost has alpha* =
+grad_u/lambda, so the feet would be traced in the wrong direction and the solve would converge to a
different equilibrium — silently, because the MAXIMIZE-quadratic Hamiltonian is smooth so the
non-smooth DPP reroute never fires. The solver now raises NotImplementedError at construction rather
than running the wrong scheme (mirroring the HJBGFDMSolver Howard gate).
"""

from __future__ import annotations

import pytest

from mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian import HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import OptimizationSense, QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid


def _problem(sense: OptimizationSense) -> MFGProblem:
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], num_points=[11], boundary_conditions=no_flux_bc(dimension=1))
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0, sense=sense))
    return MFGProblem(
        geometry=grid,
        T=0.2,
        Nt=2,
        sigma=0.1,
        components=MFGComponents(hamiltonian=H, u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0),
    )


def test_sl_maximize_control_cost_fails_loud():
    """A MAXIMIZE control cost must raise NotImplementedError at construction (wrong foot direction)."""
    with pytest.raises(NotImplementedError, match="MAXIMIZE"):
        HJBSemiLagrangianSolver(problem=_problem(OptimizationSense.MAXIMIZE))


def test_sl_minimize_control_cost_constructs():
    """The MINIMIZE (default paper) case must be unaffected by the guard."""
    solver = HJBSemiLagrangianSolver(problem=_problem(OptimizationSense.MINIMIZE))
    assert solver is not None


def test_the_two_senses_trace_feet_in_opposite_directions():
    """Measure the wrong physics the guard refuses, instead of only the refusal.

    Issue #1714: 47 of the 85 tests the fail-loud campaign added can only fail if the guard is
    deleted -- reverse-applying the guard hunk produced, in every case, exactly one failure mode,
    `DID NOT RAISE`. This file was one of the 20 with no numeric assertion at all.

    The guard's message makes a checkable claim: the semi-Lagrangian path traces feet with the
    MINIMIZE-signed velocity alpha* = -grad(u)/lambda, and "a MAXIMIZE control cost has
    alpha* = +grad(u)/lambda, so the feet would move in the wrong direction". That is measurable
    without going near the solver, and it is what makes the refusal correct rather than merely
    conservative.

    This asserts the physics, so it holds with or without the guard, and it fails if the two
    senses ever produce the same control -- the state in which the SL path could silently accept
    MAXIMIZE and the guard would be refusing nothing.
    """
    import numpy as np

    from mfgarchon.core.hamiltonian import OptimizationSense, QuadraticControlCost, SeparableHamiltonian

    lam = 2.0
    x = np.array([0.3])
    m = np.array([1.0])
    grad_u = np.array([0.5])

    def alpha_star(sense):
        cost = QuadraticControlCost(sense=sense, control_cost=lam)
        hamiltonian = SeparableHamiltonian(control_cost=cost, coupling=lambda mm: mm, coupling_dm=lambda mm: 1.0)
        return float(np.ravel(hamiltonian.optimal_control(x, m, grad_u))[0])

    minimize = alpha_star(OptimizationSense.MINIMIZE)
    maximize = alpha_star(OptimizationSense.MAXIMIZE)

    assert minimize == pytest.approx(-grad_u[0] / lam), (
        f"MINIMIZE must give alpha* = -grad(u)/lambda = {-grad_u[0] / lam}, got {minimize}"
    )
    assert maximize == pytest.approx(+grad_u[0] / lam), (
        f"MAXIMIZE must give alpha* = +grad(u)/lambda = {grad_u[0] / lam}, got {maximize}"
    )
    assert minimize == pytest.approx(-maximize), (
        f"the two senses must be exact opposites -- that is why the feet would be traced backwards "
        f"and why the refusal is correct. Got MINIMIZE={minimize}, MAXIMIZE={maximize}. If these "
        f"ever agree, this guard is refusing a configuration that would have been fine."
    )
    assert minimize != pytest.approx(0.0), "a zero control would make the comparison vacuous"

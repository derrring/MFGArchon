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

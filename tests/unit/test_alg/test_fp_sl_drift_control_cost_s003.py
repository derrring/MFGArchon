#!/usr/bin/env python3
"""Issue #1420 / G-017 / audit finding S0-03: the semi-Lagrangian FP solvers computed the drift as
``alpha = -grad(U)`` — dropping the ``1/lambda`` (``1/control_cost``) factor — so for
``control_cost != 1`` the transported drift had the wrong magnitude (the HJB used control cost
``lambda`` while the FP advected ``-grad(U)``, i.e. ``c_eff = 1``).

The fix single-sources the drift coefficient from the Hamiltonian's ``control_cost`` via
``pde_coefficients.fp_drift_coefficient`` (= ``1/control_cost`` for a quadratic-MINIMIZE
``SeparableHamiltonian``), so ``alpha* = -grad(U)/control_cost``. This is **byte-identical when
``control_cost == 1``** and corrects the magnitude otherwise.

This is a behaviour change for ``control_cost != 1`` (the S0-03 bug). These pins assert the corrected
drift and that it is distinct from the old ``-grad(U)`` (the relevance guard). The Jacobian SL solver
these pins also covered, and its divergence pin, were removed with the solver in #1756.

Refs #1420, #1430. Audit finding S0-03.
"""

import warnings

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

N = 21


def _problem(control_cost: float) -> MFGProblem:
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[N], boundary_conditions=no_flux_bc(dimension=1))
    comp = MFGComponents(
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=control_cost)),
    )
    return MFGProblem(geometry=grid, components=comp, T=0.5, Nt=5, volatility=0.3)


def _U() -> np.ndarray:
    x = np.linspace(0.0, 1.0, N)
    return 0.5 * (x - 0.3) ** 2  # non-trivial, nonzero gradient


@pytest.mark.parametrize("control_cost", [0.5, 1.0, 2.0])
@pytest.mark.parametrize(("solver_cls", "vel_method"), [(FPSLSolver, "_compute_velocity_1d")])
def test_sl_velocity_uses_control_cost(solver_cls, vel_method, control_cost):
    """SL drift must be α* = -∇U/control_cost (S0-03), not -∇U."""
    problem = _problem(control_cost)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = solver_cls(problem)
    u = _U()
    # This pins the coefficient. FPSLSolver is compared on interior nodes only: at the walls it takes the
    # HJB half's gradient (#2439), a second-order one-sided difference (#2467), where np.gradient's is first
    # order.
    nodes = slice(1, -1)
    alpha = np.asarray(getattr(solver, vel_method)(u, time=0.0)).ravel()[nodes]
    unit = -np.gradient(u, solver.dx).ravel()[nodes]
    np.testing.assert_allclose(
        alpha, unit / control_cost, rtol=0, atol=1e-12, err_msg="SL drift must be -∇U/control_cost (S0-03)"
    )
    if control_cost != 1.0:
        # Relevance guard: distinct from the pre-fix -∇U (the dropped-1/λ bug)
        assert not np.allclose(alpha, unit, atol=1e-9), (
            "SL drift is still -∇U (missing 1/control_cost) for control_cost != 1 (S0-03 not fixed)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

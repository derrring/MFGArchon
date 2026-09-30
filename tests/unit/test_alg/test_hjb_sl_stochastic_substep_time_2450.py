"""The stochastic SL path evaluates a sub-step at the step's time, not at time_idx * dt_sub (#2450).

When it sub-stepped, the stochastic path evaluated dH/dp, the value update and the boundary data at
time_idx times the SUB-step's dt: neither the step's time nor the sub-step's. For a Hamiltonian that
depends on t that is a different equation.

A mutation-verified pin, not an external oracle. The reference is the ADI path refined in CFL only
(cfl_target = 0.05); it shares _characteristic_foot_velocity, _sl_value_update and the step-time
convention with the path under test. Measured, the check fails when the fix is reverted (2.6e-2),
when only the foot's time is reverted (5.7e-2), when only the value update's is (1.6e-2), and when
the step's time is replaced by 0 (3.0e-2). It cannot see the boundary data's time (no-flux here;
#2453), a time defect inside the shared helpers, or the n-D path.
"""

from __future__ import annotations

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import HamiltonianBase
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


class _TimeDependentH(HamiltonianBase):
    """H = (1 + 4t)|p|^2/2 - m, so dH/dp = (1 + 4t) p."""

    def __call__(self, t, x, p, m):
        return (1 + 4 * t) * 0.5 * np.sum(np.asarray(p, dtype=float) ** 2, axis=-1) - np.asarray(m, dtype=float)

    def dp(self, t, x, p, m):
        return (1 + 4 * t) * np.asarray(p, dtype=float)


def test_the_stochastic_path_agrees_with_refined_adi_under_a_time_dependent_hamiltonian():
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(
        model=Model(hamiltonian=_TimeDependentH(), volatility=0.2),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=1.0),
        Nt=10,
    )
    ut = 0.4 * (grid.coordinates[0] - 0.4) ** 2
    m, coupling = np.ones((11, 41)), np.tile(ut, (11, 1))
    stochastic = HJBSemiLagrangianSolver(problem, diffusion_method="stochastic").solve_hjb_system(m, ut, coupling)
    reference = HJBSemiLagrangianSolver(problem, cfl_target=0.05, max_substeps=10000).solve_hjb_system(m, ut, coupling)
    # Measured 2.6e-3, and 2.6e-2 before the fix; the default ADI solve reads 8.8e-4 against the same reference.
    assert np.max(np.abs(stochastic - reference)) < 8e-3

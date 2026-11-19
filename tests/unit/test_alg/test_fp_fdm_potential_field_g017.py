#!/usr/bin/env python3
"""Issue #1420 / G-017: the FP-FDM ``potential_field=U`` path derives its drift from the Hamiltonian's
control law, α* = optimal_control = -∇u/control_cost, the single source.

G-017: an independent problem-level ``coupling_coefficient`` (default 0.5) once stood beside the
Hamiltonian's control cost and silently diverged from 1/control_cost, giving a 4–5× too-wide
equilibrium in exp16 Tier-2. ``MFGProblem`` no longer takes it (#2977).

These pins compare the two FP drift channels on the SAME problem, at two control costs:
- ``potential_field=U`` → ``FPFDMSolver`` forms ``-c·∇U`` with c read from the Hamiltonian;
- ``drift_field=compute_fp_velocity_field(...)`` → α* through ``H.optimal_control``.

Refs #1420, #1430. Pattern: single source of truth — sharp form (one owner + pinning test).
"""

import warnings

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.fixed_point_utils import compute_fp_velocity_field
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

NX = 41
NT = 10
T = 0.5
SIGMA = 0.2


def _problem(control_cost: float) -> MFGProblem:
    comp = MFGComponents(
        m_initial=lambda x: np.exp(-20 * (x[0] - 0.5) ** 2) + 0.1,
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=control_cost),
            coupling=lambda m: m,
            coupling_dm=lambda m: 1.0,
        ),
    )
    geom = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[NX],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    return MFGProblem(geometry=geom, components=comp, T=T, Nt=NT, volatility=SIGMA)


def _u_field() -> np.ndarray:
    """A smooth, time-varying value function so the drift is non-trivial."""
    x = np.linspace(0.0, 1.0, NX)
    return np.stack([0.5 * (x - 0.7) ** 2 * (1.0 + 0.3 * k / (NT)) for k in range(NT + 1)])


def _solve_both_paths(problem: MFGProblem) -> tuple[np.ndarray, np.ndarray]:
    """Return (M via potential_field=U, M via drift_field=optimal_control) for the same problem."""
    x = np.linspace(0.0, 1.0, NX)
    u = _u_field()
    m0 = np.exp(-20 * (x - 0.5) ** 2) + 0.1
    m0 = m0 / np.trapezoid(m0, x)
    m_traj = np.tile(m0, (NT + 1, 1))
    solver = FPFDMSolver(problem)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_potential = solver.solve_fp_system(m0, potential_field=u)
        alpha = compute_fp_velocity_field(problem, u, m_traj, problem.hamiltonian_class)
        m_velocity = solver.solve_fp_system(m0, drift_field=alpha)
    return m_potential, m_velocity


class TestFPFDMPotentialFieldDriftSource:
    def test_potential_path_byte_identical_when_consistent(self):
        """The potential-field drift (-c·∇U, c from the Hamiltonian) and the optimal_control velocity
        (-∇U/control_cost) are the SAME face velocity → byte-identical density."""
        control_cost = 2.0
        problem = _problem(control_cost=control_cost)
        m_potential, m_velocity = _solve_both_paths(problem)
        assert np.array_equal(m_potential, m_velocity), (
            "potential_field=U and drift_field=optimal_control must be byte-identical; max|diff| = "
            f"{float(np.max(np.abs(m_potential - m_velocity))):.3e}."
        )

    def test_the_potential_path_drift_is_minus_grad_u_over_the_control_cost(self):
        """The potential-field drift equals α* = -∇U/control_cost at a second control cost. Before the
        G-017 fix (#1420) the two channels diverged ~20% here, when the problem-level scalar (0.5)
        was not 1/control_cost."""
        control_cost = 1.0
        problem = _problem(control_cost=control_cost)
        m_potential, m_velocity = _solve_both_paths(problem)
        rel = float(np.max(np.abs(m_potential - m_velocity))) / (float(np.max(np.abs(m_velocity))) + 1e-12)
        assert rel < 1e-10, (
            f"potential_field drift diverges from optimal_control by {rel:.3e} at control_cost "
            f"{control_cost}; the drift must come from the Hamiltonian's control cost (G-017)."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

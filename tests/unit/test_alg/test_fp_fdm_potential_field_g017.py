#!/usr/bin/env python3
"""Issue #1420 / G-017: the FP-FDM ``potential_field=U`` path must derive its drift from the
Hamiltonian's control law (α* = optimal_control = -∇u/control_cost), NOT from an independent
``coupling_coefficient`` scalar.

Background. For a smooth-separable (quadratic) Hamiltonian the coupled solver routes the value
function as ``potential_field=U`` (``resolve_fp_drift_kwargs``), and ``FPFDMSolver`` forms the drift
internally as ``-coupling_coefficient·∇U`` (``fp_fdm.py``). But ``coupling_coefficient`` (default 0.5)
and the Hamiltonian's ``control_cost`` λ are *independent fields* that must satisfy
``coupling_coefficient = 1/control_cost`` — and silently diverge when they don't. The correct drift is
α* = ``H.optimal_control(x, m, ∇u, t)`` = -∇u/control_cost (the single source), independent of
``coupling_coefficient``. This is gotcha G-017; exp16 Tier-2 hit it (~4–5× too-wide equilibrium until
``coupling_coefficient`` was set to 1/control_cost).

These pins compare the two FP drift channels on the SAME problem:
- ``potential_field=U`` → ``FPFDMSolver`` forms ``-coupling_coefficient·∇U`` internally.
- ``drift_field=compute_fp_velocity_field(...)`` → α* via ``H.optimal_control`` (control_cost-based).

Measured: when ``coupling_coefficient == 1/control_cost`` the two are **byte-identical** (rel diff
0.0); when they disagree the densities diverge ~20%. So:
- ``test_potential_path_byte_identical_when_consistent`` locks the consistent-case agreement (must
  stay byte-identical through the single-source refactor).
- ``test_potential_path_uses_control_cost_not_coupling`` asserts the CORRECT behavior (the two paths
  agree even when ``coupling_coefficient ≠ 1/control_cost``, i.e. the potential path ignores the
  redundant scalar). Before the fix this diverged ~20% (G-017); the FP-FDM drift is now single-sourced
  from the Hamiltonian's control_cost (``fp_fdm_time_stepping._fp_drift_coefficient``).

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


def _problem(control_cost: float, coupling_coefficient: float) -> MFGProblem:
    comp = MFGComponents(
        m_initial=lambda x: np.exp(-20 * (x - 0.5) ** 2) + 0.1,
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
    return MFGProblem(
        geometry=geom, components=comp, T=T, Nt=NT, sigma=SIGMA, coupling_coefficient=coupling_coefficient
    )


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
        """When coupling_coefficient == 1/control_cost, the potential-field drift (-coupling·∇U) and
        the optimal_control velocity (-∇U/control_cost) are the SAME face velocity → byte-identical
        density. This must remain true through the single-source refactor."""
        control_cost = 2.0
        problem = _problem(control_cost=control_cost, coupling_coefficient=1.0 / control_cost)
        m_potential, m_velocity = _solve_both_paths(problem)
        assert np.array_equal(m_potential, m_velocity), (
            "potential_field=U and drift_field=optimal_control must be byte-identical when "
            f"coupling_coefficient == 1/control_cost; max|diff| = "
            f"{float(np.max(np.abs(m_potential - m_velocity))):.3e}."
        )

    def test_potential_path_uses_control_cost_not_coupling(self):
        """The potential-field drift equals α* = -∇U/control_cost regardless of the (now-redundant)
        coupling_coefficient. Before the G-017 fix (#1420) this diverged ~20% when
        coupling_coefficient ≠ 1/control_cost; the FP-FDM drift is now single-sourced from the
        Hamiltonian's control_cost (``_fp_drift_coefficient``), so the two channels agree."""
        control_cost = 1.0
        problem = _problem(control_cost=control_cost, coupling_coefficient=0.5)  # 0.5 != 1/1.0
        m_potential, m_velocity = _solve_both_paths(problem)
        rel = float(np.max(np.abs(m_potential - m_velocity))) / (float(np.max(np.abs(m_velocity))) + 1e-12)
        assert rel < 1e-10, (
            f"potential_field drift diverges from optimal_control by {rel:.3e} when "
            f"coupling_coefficient ({0.5}) != 1/control_cost ({1.0 / control_cost}); the drift must "
            f"come from control_cost (the Hamiltonian), not coupling_coefficient (G-017)."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

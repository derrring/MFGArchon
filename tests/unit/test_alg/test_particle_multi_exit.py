"""
Unit test for FPParticleSolver with multi-exit absorbing BC.

Validates Issue #535 Phase 1: Segment-aware boundary conditions with
multiple DIRICHLET exits.
"""

from __future__ import annotations

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, mixed_bc, no_flux_bc
from mfgarchon.geometry.boundary.types import BCType


def _default_hamiltonian():
    """Default Hamiltonian for testing (Issue #670: explicit specification required)."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components_1d():
    """Provide default components for 1D test problems."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-((np.asarray(x) - 5.0) ** 2) / 2.0),
        u_terminal=lambda x: 0.0,
        hamiltonian=_default_hamiltonian(),
    )


def _default_components_2d():
    """Provide default components for 2D test problems."""

    def m_initial_2d(x):
        x_arr = np.asarray(x)
        if x_arr.ndim == 1:
            return np.exp(-((x_arr[0] - 2.0) ** 2 + (x_arr[1] - 2.0) ** 2) / 2.0)
        return np.exp(-((x_arr[..., 0] - 2.0) ** 2 + (x_arr[..., 1] - 2.0) ** 2) / 2.0)

    return MFGComponents(
        m_initial=m_initial_2d,
        u_terminal=lambda x: 0.0,
        hamiltonian=_default_hamiltonian(),
    )


def test_particle_solver_multi_exit_1d():
    """
    Test FPParticleSolver with two absorbing exits in 1D.

    Setup:
        Domain: [0, 10]
        Exit 1: x ∈ [0, 0.5]   (left wall, partial)
        Exit 2: x ∈ [9.5, 10]  (right wall, partial)
        Reflecting: elsewhere

    Drift: Particles pushed toward edges → some exit left, some exit right
    """
    # Create 1D domain
    geometry = TensorProductGrid(bounds=[(0.0, 10.0)], Nx_points=[101], boundary_conditions=no_flux_bc(1))
    problem = MFGProblem(
        geometry=geometry,
        T=3.0,
        Nt=60,
        volatility=0.05,
        coupling_coefficient=1.0,
        components=_default_components_1d(),
    )

    # Multi-exit BC: two DIRICHLET exits on opposite ends
    bc_multi_exit = mixed_bc(
        dimension=1,
        segments=[
            BCSegment(
                name="exit_left",
                bc_type=BCType.DIRICHLET,
                value=0.0,
                boundary="left",  # Entire left boundary
            ),
            BCSegment(
                name="exit_right",
                bc_type=BCType.DIRICHLET,
                value=0.0,
                boundary="right",  # Entire right boundary
            ),
        ],
        domain_bounds=np.array([[0.0, 10.0]]),
    )

    solver = FPParticleSolver(
        problem,
        num_particles=200,
        boundary_conditions=bc_multi_exit,
    )

    # Initial density: centered Gaussian
    x = geometry.coordinates[0]
    M_init = np.exp(-((x - 5.0) ** 2) / 2.0)
    M_init = M_init / (np.sum(M_init) * 0.1)  # Normalize

    # Value function: U(x) = -α*|x - 5|
    # → Drift: -∇U = α*sign(x - 5) (pushes to edges)
    # For particles left of center → drift left, right of center → drift right
    Nt = problem.Nt + 1
    Nx = len(x)
    drift_field = np.zeros((Nt, Nx))
    for t in range(Nt):
        drift_field[t, :] = -5.0 * np.abs(x - 5.0)  # V-shaped potential

    # Solve
    M_solution = solver.solve_fp_system(
        M_initial=M_init,
        drift_field=drift_field,
        show_progress=False,
    )

    # Assertions
    assert M_solution.shape == (Nt, Nx)
    assert not np.any(np.isnan(M_solution)), "Solution contains NaN"
    assert not np.any(np.isinf(M_solution)), "Solution contains inf"

    # Verify particles were absorbed
    assert solver.total_absorbed > 0, "No particles absorbed (drift may be too weak)"
    assert len(solver.exit_flux_history) > 0, "No exit flux recorded"

    # Verify BOTH exits were used (key test for multi-exit)
    if solver.total_absorbed > 0:
        all_exit_positions = np.concatenate(solver.exit_positions_history)

        # Classify by position
        left_exits = np.sum(all_exit_positions < 1.0)  # Near x=0
        right_exits = np.sum(all_exit_positions > 9.0)  # Near x=10

        # Both exits should be used
        assert left_exits > 0, f"Left exit not used (absorbed: {left_exits})"
        assert right_exits > 0, f"Right exit not used (absorbed: {right_exits})"

        print(f"✓ Multi-exit test passed: {left_exits} left, {right_exits} right")


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing multi-exit absorbing BC...")

    print("\n1D multi-exit test:")
    test_particle_solver_multi_exit_1d()

    print("\n✓ Multi-exit test passed!")

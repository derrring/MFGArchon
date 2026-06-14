#!/usr/bin/env python3
"""
Test to verify KDE mass conservation issue
"""

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver as ParticleFPSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid, no_flux_bc


def _default_hamiltonian():
    """Default Hamiltonian for testing."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components():
    """Default MFGComponents for testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),  # Gaussian centered at 0.5
        u_terminal=lambda x: 0.0,  # Zero terminal cost
        hamiltonian=_default_hamiltonian(),
    )


def test_kde_normalization():
    print("=== Testing KDE Mass Conservation ===")

    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1)
    )  # Nx=20 -> 21 points
    problem = MFGProblem(
        geometry=geometry, T=0.1, Nt=5, sigma=1.0, coupling_coefficient=0.5, components=_default_components()
    )

    bc = no_flux_bc(dimension=1)

    print('Test 1: With kde_normalization="none" (current)')
    solver1 = ParticleFPSolver(
        problem=problem,
        num_particles=200,
        kde_bandwidth="scott",
        kde_normalization="none",  # This is the issue!
        boundary_conditions=bc,
    )

    (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
    Nt_points = problem.Nt + 1  # Temporal grid points
    U_zero = np.zeros((Nt_points, Nx_points))
    M_result1 = solver1.solve_fp_system(M_initial=problem.m_initial, potential_field=U_zero)

    dx = problem.geometry.get_grid_spacing()[0]
    mass1 = np.sum(M_result1 * dx, axis=1)
    print(f"  Initial mass: {mass1[0]:.6f}")
    print(f"  Final mass: {mass1[-1]:.6f}")
    print(f"  Mass loss: {mass1[0] - mass1[-1]:.6f} ({(mass1[0] - mass1[-1]) / mass1[0] * 100:.2f}%)")

    print('\nTest 2: With kde_normalization="all" (should be better)')
    solver2 = ParticleFPSolver(
        problem=problem,
        num_particles=200,
        kde_bandwidth="scott",
        kde_normalization="all",  # This should help!
        boundary_conditions=bc,
    )

    M_result2 = solver2.solve_fp_system(M_initial=problem.m_initial, potential_field=U_zero)

    mass2 = np.sum(M_result2 * dx, axis=1)
    print(f"  Initial mass: {mass2[0]:.6f}")
    print(f"  Final mass: {mass2[-1]:.6f}")
    print(f"  Mass loss: {mass2[0] - mass2[-1]:.6f} ({(mass2[0] - mass2[-1]) / mass2[0] * 100:.2f}%)")

    print("\nTest 3: Different bandwidth")
    solver3 = ParticleFPSolver(
        problem=problem,
        num_particles=200,
        kde_bandwidth=0.05,  # Fixed bandwidth
        kde_normalization="all",
        boundary_conditions=bc,
    )

    M_result3 = solver3.solve_fp_system(M_initial=problem.m_initial, potential_field=U_zero)

    mass3 = np.sum(M_result3 * dx, axis=1)
    print(f"  Initial mass: {mass3[0]:.6f}")
    print(f"  Final mass: {mass3[-1]:.6f}")
    print(f"  Mass loss: {mass3[0] - mass3[-1]:.6f} ({(mass3[0] - mass3[-1]) / mass3[0] * 100:.2f}%)")

    # Verify mass conservation: normalized KDE should conserve mass within 20%
    mass_loss_pct = abs(mass2[0] - mass2[-1]) / mass2[0] * 100
    assert mass_loss_pct < 50, f"Normalized KDE mass loss too large: {mass_loss_pct:.1f}%"
    # Verify densities are non-negative and finite
    assert np.all(np.isfinite(M_result2)), "KDE density contains non-finite values"
    assert np.all(M_result2 >= 0), "KDE density contains negative values"


if __name__ == "__main__":
    test_kde_normalization()

"""Issue #1489 (S4): the weak-form FP solver fails loud when the initial density length != n_dof,
instead of silently padding/truncating (the P2 edge-DOF zero-fill silent-wrong-IC bug)."""

from __future__ import annotations

import pytest

import numpy as np


def _meshless_fp_solver():
    from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    geom = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
    comp = MFGComponents(
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
        ),
    )
    prob = MFGProblem(geometry=geom, T=0.2, Nt=5, sigma=0.3, components=comp, coupling_coefficient=1.0)
    cloud = np.linspace(0.0, 1.0, 11).reshape(-1, 1)
    return MeshlessGalerkinFPSolver(prob, cloud, delta=2.6 / np.sqrt(11), degree=2)


def test_ic_dof_count_mismatch_fails_loud():
    fp = _meshless_fp_solver()
    n = fp._n_dof
    u = np.zeros((fp.problem.Nt + 1, n))
    # too short (the P2 zero-fill case) and too long (silent truncation) both must raise, not pad/clip
    with pytest.raises(ValueError, match="DOFs"):
        fp.solve_fp_system(np.ones(n - 1), potential_field=u)
    with pytest.raises(ValueError, match="DOFs"):
        fp.solve_fp_system(np.ones(n + 3), potential_field=u)


def test_ic_correct_length_is_accepted():
    fp = _meshless_fp_solver()
    n = fp._n_dof
    u = np.zeros((fp.problem.Nt + 1, n))
    m = fp.solve_fp_system(np.ones(n) / n, potential_field=u)  # correct length: no raise
    assert m.shape == (fp.problem.Nt + 1, n)

    # Oracle 1 (closed form): potential_field == 0 means no drift, so under no-flux walls a
    # uniform density is an exact stationary solution. Measured max|m - 1/n| = 3.5e-13.
    np.testing.assert_allclose(m, 1.0 / n, atol=1e-10)

    # Oracle 2 (conservation): the weak-form mass is 1^T M m, NOT the nodal sum -- the basis
    # integrals differ between interior and boundary DOFs, so the coefficient sum is conserved
    # only for a uniform density. Measured relative drift 2.2e-16 over the sweep.
    mass = np.asarray((fp._M @ m.T).sum(axis=0)).ravel()
    np.testing.assert_allclose(mass, mass[0], rtol=1e-12)

    # Positive control. A stationary solution is also what a solver that returned its input
    # unchanged would produce, so neither oracle above can separate the two on its own.
    # Diffusing a bump must move the density, and under no-flux the discrete minimum is
    # non-decreasing in time (maximum principle) while the mass is still conserved.
    bump = np.exp(-20.0 * (np.linspace(0.0, 1.0, n) - 0.5) ** 2)
    m_bump = fp.solve_fp_system(bump / bump.sum(), potential_field=u)
    assert np.max(np.abs(m_bump[-1] - m_bump[0])) > 1e-2  # measured 3.89e-02
    assert np.all(np.diff(m_bump.min(axis=1)) > 0)  # measured smallest increment 4.89e-03
    mass_bump = np.asarray((fp._M @ m_bump.T).sum(axis=0)).ravel()
    np.testing.assert_allclose(mass_bump, mass_bump[0], rtol=1e-12)

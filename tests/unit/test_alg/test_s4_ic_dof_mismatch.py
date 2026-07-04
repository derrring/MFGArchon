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

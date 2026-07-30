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


def test_zero_padding_a_p2_initial_density_destroys_it_entirely():
    """The guard says padding "mis-places" the density. Measured, it deletes it.

    Issue #1714: this file was one of the 20 fail-loud test files with no numeric assertion --
    `pytest.raises` on the guard's message records that the guard fires, not what it prevents.

    The former behaviour reconciled lengths by truncating or zero-padding. On P2 the caller
    resolves `m_initial` on the vertices, so every edge-midpoint DOF gets zero. On a refined
    symmetric unit square that is 208 of 289 DOFs (72%), and the consequence is not a distortion:

        integral of the density, correct IC  : 0.104696
        integral of the density, zero-padded : 0.000000

    The integral collapses to exactly zero because the P2 vertex shape function integrates to zero
    over a triangle -- the same identity that makes row-sum mass lumping invalid at P2
    (`test_weak_form_hjb_p2_gradient_1252.py`). The mass carried by a P2 field lives entirely in
    its edge DOFs, which is precisely the set that padding zeroes.

    So the guard is not being cautious about an approximation; it is refusing an IC with no mass.
    """
    import numpy as np

    skfem = pytest.importorskip("skfem", reason="scikit-fem required")
    from skfem.models.poisson import mass

    mesh = skfem.MeshTri.init_sqsymmetric().refined(2)
    basis_p2 = skfem.Basis(mesh, skfem.ElementTriP2())
    n_vertices = mesh.p.shape[1]
    n_dof = basis_p2.N

    assert n_dof > n_vertices, "P2 must add edge DOFs beyond the vertices, or this proves nothing"

    def density(xy):
        return np.exp(-30 * ((xy[0] - 0.5) ** 2 + (xy[1] - 0.5) ** 2))

    on_all_dofs = density(basis_p2.doflocs)
    zero_padded = np.pad(density(mesh.p), (0, n_dof - n_vertices))

    mass_matrix = skfem.asm(mass, basis_p2)
    ones = np.ones(n_dof)
    integral_correct = float(on_all_dofs @ (mass_matrix @ ones))
    integral_padded = float(zero_padded @ (mass_matrix @ ones))

    assert integral_correct > 0.01, f"the correctly-resolved IC must carry real mass, got {integral_correct}"
    assert integral_padded == pytest.approx(0.0, abs=1e-12), (
        f"zero-padding must destroy the density outright, not merely distort it -- the P2 vertex "
        f"shape function integrates to zero, so all the mass lives in the edge DOFs that padding "
        f"zeroes. Got {integral_padded}. If this ever becomes non-zero the guard's premise has "
        f"changed and its message should be re-read."
    )

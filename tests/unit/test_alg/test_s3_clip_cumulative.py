"""Issue #1489 (S3): the weak-form FP positivity-clip monitor reports the CUMULATIVE mass injection
over the whole solve, not just the first-exceedance step (the former one-shot latch under-reported)."""

from __future__ import annotations

import logging

import numpy as np


def _meshless_fp_steep_drift():
    from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    geom = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    comp = MFGComponents(
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
        ),
    )
    prob = MFGProblem(geometry=geom, T=1.0, Nt=20, sigma=0.05, components=comp, coupling_coefficient=1.0)
    cloud = np.linspace(0.0, 1.0, 21).reshape(-1, 1)
    fp = MeshlessGalerkinFPSolver(prob, cloud, delta=2.6 / np.sqrt(21), degree=2)  # unstabilized (no SD)
    n = fp._n_dof
    x = fp._disc.dof_coordinates.ravel()
    u = np.tile(40.0 * (x - 0.5) ** 2, (prob.Nt + 1, 1))  # sharp confining potential -> steep drift
    m0 = np.ones(n) / n
    return fp, m0, u


def test_clip_reports_cumulative_injection(mfg_caplog):
    fp, m0, u = _meshless_fp_steep_drift()
    with mfg_caplog.at_level(logging.WARNING, logger="mfgarchon.alg.numerical.weak_form_fp_solver"):
        fp.solve_fp_system(m0, potential_field=u)
    msgs = mfg_caplog.messages
    assert any("positivity clip" in m for m in msgs), f"expected a clip warning; got {msgs}"
    assert any("CUMULATIVE" in m for m in msgs), (
        f"S3: the solve-end warning must report the cumulative clip injection, not only the first step; got {msgs}"
    )

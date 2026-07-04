"""Issue #1507: the strict-adjoint FP-FDM step must conserve mass and surface the clip. The adjoint
advection operator (transposed HJB) is not an M-matrix, so at high Péclet the solve undershoots
negative; the old code clipped to 0 (ADDING mass) with no renormalization and no diagnostic, so the
coupled fixed point converged self-consistently wrong. Now it renormalizes to the pre-step mass and
warns."""

from __future__ import annotations

import logging

import numpy as np
from scipy import sparse

from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _fp_solver_and_advection(n=21, drift=40.0):
    geom = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    comp = MFGComponents(
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
        ),
    )
    prob = MFGProblem(geometry=geom, T=0.2, Nt=5, sigma=0.05, components=comp, coupling_coefficient=1.0)
    fp = FPFDMSolver(prob)
    h = 1.0 / (n - 1)
    off = drift / (2 * h)
    a = sparse.diags([-off * np.ones(n - 1), np.zeros(n), off * np.ones(n - 1)], [-1, 0, 1]).tocsr()
    return fp, a, h, n


def test_strict_adjoint_step_conserves_mass_and_is_nonnegative():
    fp, a, h, n = _fp_solver_and_advection()
    m0 = np.zeros(n)
    m0[n // 2] = 1.0 / h  # peaked density, high Péclet -> the solve undershoots negative
    m_next = fp.solve_fp_step_adjoint_mode(m0, a, sigma=0.05)
    assert (m_next >= 0.0).all()  # clip enforced
    assert np.isclose(m_next.sum(), m0.sum(), rtol=1e-12)  # renormalized -> no silent mass injection


def test_strict_adjoint_clip_warns():
    fp, a, h, n = _fp_solver_and_advection()
    m0 = np.zeros(n)
    m0[n // 2] = 1.0 / h
    records: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r.getMessage())  # type: ignore[method-assign]
    log = logging.getLogger("mfgarchon.alg.numerical.fp_solvers.fp_fdm")
    log.addHandler(handler)
    log.setLevel(logging.WARNING)
    try:
        fp.solve_fp_step_adjoint_mode(m0, a, sigma=0.05)
    finally:
        log.removeHandler(handler)
    assert any("clipped" in r and "Issue #1507" in r for r in records)

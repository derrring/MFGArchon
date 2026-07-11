"""Issue #1556: FPGFDMSolver must thread obstacle_sdf into its TaylorOperator so the FP density
derivatives (D_lap / D_grad) respect obstacle connectivity — mirroring the HJB-GFDM #1124 fix.

Without it, a coupled obstacle-cloud solve has the FP stencils coupling through walls while the HJB
stencils are visibility-filtered (asymmetric physics).
"""

from __future__ import annotations

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_gfdm import FPGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid
from mfgarchon.geometry.implicit import Hypersphere


def _pillar_cloud(seed: int = 42, n: int = 100):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 10, size=(n, 2))
    pillar = Hypersphere(center=[5.0, 5.0], radius=1.5)
    return pts[pillar.signed_distance(pts) > 0.1], pillar


def _problem() -> MFGProblem:
    grid = TensorProductGrid(
        bounds=[(0.0, 10.0), (0.0, 10.0)], num_points=[5, 5], boundary_conditions=no_flux_bc(dimension=2)
    )
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0))
    return MFGProblem(
        geometry=grid,
        T=0.2,
        Nt=2,
        sigma=0.1,
        components=MFGComponents(hamiltonian=H, u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0),
    )


def _count_cross_wall_edges(op, pts, pillar) -> int:
    n_cross = 0
    for i in range(len(pts)):
        for j in op.neighborhoods[i]["indices"]:
            if j == i:
                continue
            if pillar.signed_distance(np.array([0.5 * (pts[i] + pts[j])]))[0] < 0:
                n_cross += 1
    return n_cross


def test_fp_gfdm_without_obstacle_has_cross_wall_edges():
    """Baseline: without obstacle_sdf, some FP stencil edges cross the pillar."""
    pts, pillar = _pillar_cloud()
    solver = FPGFDMSolver(_problem(), pts, delta=2.5)
    assert _count_cross_wall_edges(solver.gfdm_operator, pts, pillar) > 0


def test_fp_gfdm_with_obstacle_has_no_cross_wall_edges_1556():
    """With obstacle_sdf threaded through to TaylorOperator, no FP stencil edge crosses the wall."""
    pts, pillar = _pillar_cloud()
    solver = FPGFDMSolver(_problem(), pts, delta=2.5, obstacle_sdf=pillar.signed_distance)
    assert _count_cross_wall_edges(solver.gfdm_operator, pts, pillar) == 0

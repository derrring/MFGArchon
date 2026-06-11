"""Issue #1279 (2026-06-11 survey): FPGFDMSolver default must solve continuity, not transport.

The default `upwind_scheme="none"` path computed only `α·∇m` (transport form), silently dropping
the `m·div(α)` term of `div(mα)`. With a UNIFORM initial density and a drift whose divergence
varies in space, the continuity term `m·div(α)` drives the density into a non-uniform profile,
while the transport form is zero for uniform `m` (∇m = 0) and diffusion of a uniform field is also
zero — so the buggy path leaves the density uniform.
"""

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_gfdm import FPGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import Hyperrectangle, TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _problem_2d(sigma=0.1):
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=no_flux_bc(dimension=2)
    )
    comp = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
    )
    return MFGProblem(geometry=grid, components=comp, T=0.1, Nt=5, sigma=sigma)


def test_fp_gfdm_default_uses_continuity_not_transport():
    problem = _problem_2d()
    domain = Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]]))
    points = domain.sample_uniform(150, seed=0)
    solver = FPGFDMSolver(problem, collocation_points=points, delta=0.25)
    assert solver.upwind_scheme == "none"  # exercise the default (buggy) path

    n = points.shape[0]
    m0 = np.ones(n) / n  # uniform: transport term and diffusion term both vanish on it
    # drift alpha(x) = (x^2, 0) -> div(alpha) = 2x varies in space, so m*div(alpha) != 0
    alpha = np.zeros((problem.Nt + 1, n, 2))
    alpha[:, :, 0] = (points[:, 0] ** 2)[None, :]

    M = solver.solve_fp_system(m0, drift_field=alpha, show_progress=False)
    final = M[-1]
    rel_spread = float(np.std(final) / np.mean(final))
    assert rel_spread > 1e-2, (
        f"density stayed ~uniform (rel_spread={rel_spread:.2e}); the m·div(α) continuity term was "
        f"dropped (transport form α·∇m)."
    )

"""GFDM pure-diffusion magnitude probe (joint_socp + precompute, per-point Newton path).

Goal: verify HJBGFDMSolver applies D = sigma^2/2 (Issue #811), the production
GFDM path. The Hamiltonian H = |p|^2/(2 lambda) adds advection, so pure diffusion
is isolated by MMS source-cancellation:

  per-point residual (hjb_gfdm.py:2243):
      r[i] = -(u^{n+1}-u^n)/dt + H(grad u^n) - (sigma^2/2) lap(u^n) + L^n[i]

  Set running_cost L^n[i] = -H(grad u*^n) evaluated on the analytic field u*.
  If the diffusion magnitude is correct the solver field u^n tracks the analytic
  backward-decaying eigenmode and the H term cancels; the recovered decay factor
  then matches exp(-D k^2 (T-t)).  A wrong magnitude (e.g. sigma->D, half-D, or a
  dt/dimension bug) detunes the decay and the relerr blows up.

  Amplitude A is kept small so the (quadratic) Hamiltonian term is O(A^2 k^2/lambda)
  while diffusion is O(A sigma^2 k^2): diffusion dominates linearly, and the residual
  cancellation removes the small remnant.  This makes the measured decay a clean
  read on the diffusion coefficient while still driving the full per-point H path.
"""

from __future__ import annotations

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def run(sigma=1.0, n_x=41, T=0.05, Nt=50, A=1e-3, lam=1.0, delta=0.3, k=np.pi):
    """1D pure-diffusion magnitude test on the GFDM joint_socp+precompute path.

    cos(k x) with k=pi is a Laplacian eigenmode obeying no-flux at x=0,1.
    Backward HJB pure diffusion: u(t,x) = A exp(-D k^2 (T-t)) cos(k x), D=sigma^2/2.
    """
    D = 0.5 * sigma**2

    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[n_x],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    # coupling=const so dH/dm has no effect; H = |p|^2/(2 lam) - 0 (const dropped by grad)
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=lam),
        coupling=lambda m: 0.0 * np.asarray(m),
        coupling_dm=lambda m: 0.0 * np.asarray(m),
    )
    components = MFGComponents(
        m_initial=lambda x: np.ones_like(np.asarray(x, dtype=float)),
        u_terminal=lambda x: A * np.cos(k * np.asarray(x, dtype=float)),
        hamiltonian=H,
    )
    problem = MFGProblem(geometry=geometry, T=T, Nt=Nt, sigma=sigma, components=components)

    x_coords = np.linspace(0.0, 1.0, n_x)
    collocation_points = x_coords.reshape(-1, 1)

    solver = HJBGFDMSolver(
        problem,
        collocation_points,
        delta=delta,
        monotonicity_scheme="joint_socp",
        monotonicity_application="precompute",
    )

    # confirm the per-point (precompute) Newton path is the one selected
    assert solver.qp_optimization_level == "precompute", solver.qp_optimization_level

    dt = T / Nt
    n_time_points = Nt + 1
    tspace = np.linspace(0.0, T, n_time_points)

    # analytic backward-decaying eigenmode at every (t, x)
    def u_star(t):
        return A * np.exp(-D * k**2 * (T - t)) * np.cos(k * x_coords)

    # MMS source: L^n[i] = -H(grad u*^n).  Analytic gradient of A exp(...) cos(k x)
    # is -A exp(...) k sin(k x); H = |p|^2 / (2 lam).
    def grad_u_star(t):
        return -A * np.exp(-D * k**2 * (T - t)) * k * np.sin(k * x_coords)

    def running_cost_fn(n):
        p = grad_u_star(tspace[n])
        H_val = (p**2) / (2.0 * lam)  # H evaluated on analytic field
        return -H_val  # cancels the H contribution in the residual

    U_terminal = A * np.cos(k * x_coords)  # at t = T
    M_density = np.ones((n_time_points, n_x))

    U = solver.solve_hjb_system(
        M_density=M_density,
        U_terminal=U_terminal,
        running_cost=running_cost_fn,
        show_progress=False,
    )

    # recover decay factor at an interior point, t=0 vs t=T
    i = n_x // 3
    measured_fac = U[0, i] / U[Nt, i]
    analytic_fac = np.exp(-D * k**2 * T)  # decay over full backward horizon
    relerr_decay = abs(measured_fac - analytic_fac) / abs(1.0 - analytic_fac)

    # also a field-wide L2 check on the recovered profile shape vs analytic
    u0 = U[0, :]
    u0_star = u_star(0.0)
    relerr_field = np.linalg.norm(u0 - u0_star) / np.linalg.norm(u0_star)

    return dict(
        sigma=sigma, D=D, n_x=n_x, T=T, Nt=Nt, A=A, k=k, delta=delta,
        measured_fac=measured_fac, analytic_fac=analytic_fac,
        relerr_decay=relerr_decay, relerr_field=relerr_field,
    )


if __name__ == "__main__":
    for sigma in (0.5, 1.0, 1.5):
        r = run(sigma=sigma)
        print(
            f"sigma={sigma:.3f} D={r['D']:.4f}  "
            f"measured_fac={r['measured_fac']:.6f} analytic_fac={r['analytic_fac']:.6f}  "
            f"relerr_decay={r['relerr_decay']:.3e}  relerr_field={r['relerr_field']:.3e}"
        )

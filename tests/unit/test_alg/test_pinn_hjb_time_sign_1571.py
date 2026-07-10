"""Issue #1571: the PINN HJB residual must use the backward-HJB time sign -du/dt, not +du/dt.

The HJB is backward (terminal condition u(T,x)=g at t=T, t fed un-reversed), so the canonical
residual is ``-du/dt + H - (sigma^2/2) u_xx = 0``. An earlier version assembled ``+du/dt + H - ...``
(mirroring the FORWARD FP diffusion sign onto the time derivative), fitting the wrong PDE.

Discriminator (robust to any H / potential / coupling in the fixture): feed two controlled value
functions ``u = +c*t + 0.5 x**2`` and ``u = -c*t + 0.5 x**2`` on the SAME collocation points. Their
x-parts are identical, so H(u_x) and the viscous term D*u_xx are identical and cancel in the
residual difference, which then isolates the time-derivative term (call the shared part S):

    res(+c*t) - res(-c*t) = (-c + S) - (+c + S) = -2c   under the correct -du/dt
                          = (+c + S) - (-c + S) = +2c   under the buggy +du/dt

so the difference is negative iff the sign is correct. A revert to ``+u_t`` flips it positive.
"""

from __future__ import annotations

import pytest

import numpy as np

torch = pytest.importorskip("torch")
nn = torch.nn

pytestmark = pytest.mark.optional_torch


def _make_problem():
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
    from mfgarchon.geometry import Hyperrectangle

    geo = Hyperrectangle(bounds=[(0.0, 1.0)])
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: np.zeros_like(m),
    )
    components = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: np.zeros_like(x) if hasattr(x, "__len__") else 0.0,
        m_initial=lambda x: np.ones_like(x) if hasattr(x, "__len__") else 1.0,
    )
    return MFGProblem(T=1.0, geometry=geo, sigma=0.1, components=components)


def _tiny_config():
    from mfgarchon.alg.neural.pinn_solvers.base_pinn import PINNConfig

    return PINNConfig(hidden_layers=[8, 8], device="cpu")


class _LinearTNet(nn.Module):
    """u(t, x) = c * t + 0.5 * x**2. u_t = c, u_x = x, u_xx = 1.

    The x-part is IDENTICAL for the +c and -c nets, so the Hamiltonian H(u_x) and the viscous
    term D*u_xx are the same for both and cancel in the residual difference — which then isolates
    the time-derivative term, 2*u_t. (A pure ``c*t`` would make u_x=u_xx=0, but then u_xx has no
    grad path and autograd raises; the x**2 term gives a real, computable second derivative.)"""

    def __init__(self, c: float):
        super().__init__()
        self.c = float(c)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        t = inp[:, 0:1]
        x = inp[:, 1:2]
        return self.c * t + 0.5 * x**2


def _hjb_residual_mean(solver, c: float, t: torch.Tensor, x: torch.Tensor) -> float:
    """Mean HJB residual for u = c*t + 0.5 x**2 at the GIVEN collocation points."""
    solver.u_net = _LinearTNet(c)
    solver.networks["u_net"] = solver.u_net
    res = solver.compute_pde_residual(t, x)["hjb"]
    return res.mean().item()


def _time_sign_diff(solver) -> tuple[float, float]:
    """Return (diff, c) where diff = res(+c*t) - res(-c*t) on SHARED collocation points, so the
    H(u_x) and viscous D*u_xx terms (identical for both sign choices) cancel exactly and the
    difference is purely the time-derivative contribution."""
    c = 2.0
    n = 16
    t = torch.rand(n, 1)
    x = torch.rand(n, 1) * 0.8 + 0.1  # interior, away from the boundary
    return _hjb_residual_mean(solver, c, t, x) - _hjb_residual_mean(solver, -c, t, x), c


def test_hjb_pinn_uses_backward_time_sign():
    from mfgarchon.alg.neural.pinn_solvers.hjb_pinn_solver import HJBPINNSolver

    solver = HJBPINNSolver(_make_problem(), config=_tiny_config())
    diff, c = _time_sign_diff(solver)
    # Correct -du/dt: diff = -2c = -4.0. Buggy +du/dt would give +4.0.
    assert diff == pytest.approx(-2.0 * c, abs=1e-5), (
        f"HJB residual time-derivative sign is wrong: res(+c*t)-res(-c*t)={diff:.6f}, "
        f"expected {-2.0 * c:.4f} (backward HJB -du/dt). A positive value means the buggy +du/dt "
        f"(Issue #1571)."
    )


def test_mfg_pinn_uses_backward_time_sign():
    from mfgarchon.alg.neural.pinn_solvers.mfg_pinn_solver import MFGPINNSolver

    solver = MFGPINNSolver(_make_problem(), config=_tiny_config())
    diff, c = _time_sign_diff(solver)
    assert diff == pytest.approx(-2.0 * c, abs=1e-5), (
        f"MFG-PINN HJB residual time-derivative sign is wrong: res(+c*t)-res(-c*t)={diff:.6f}, "
        f"expected {-2.0 * c:.4f} (backward HJB -du/dt) (Issue #1571)."
    )

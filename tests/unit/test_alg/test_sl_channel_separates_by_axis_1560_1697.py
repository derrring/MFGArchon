"""The semi-Lagrangian pair applies each axis's own boundary condition (#1560, #1697).

ADMISSION (#2257). Class 2, an external oracle: separability. On a channel over [0, 1] x [0, 2] --
no-flux walls in x, periodic in y -- a separable problem splits into its two axes:

- HJB-SL, H = |p|^2 / 2 and u_T = f(x) + g(y):  U_channel(x, y) = U_x(x) + U_y(y).
- FP-SL, a potential U = f(x) + g(y) and m_0 = a(x) b(y):  M_channel(x, y) = M_x(x) M_y(y).

U_x and M_x solve the problem on [0, 1] with no-flux walls, U_y and M_y on [0, 2] with a periodic BC.
The law is the PDE's. The scheme reproduces it to rounding because each axis's foot depends on that
axis alone, and every later stage is a tensor product: interpolation and splatting weights multiply
across axes, and a sweep along an axis leaves a field that is constant along it unchanged.

For HJB-SL with ADI and for FP-SL the right-hand side comes from the 1-D solver on each axis's own
interval. That is another code path, which agrees with the 2-D solve of one-axis data to 1.5e-15
(HJB) and 6.9e-15 (FP, relative). The unequal intervals make an axis folded with another axis's
bounds visible.

The stochastic step's quadrature takes 2d departures per node, so its 1-D step is a different
scheme, 2.5e-02 away. Its reference is therefore the same 2-D solver on uniform BCs, with data
constant along the other axis. That compares the solver with itself, so a defect shared with the
uniform solves passes it.

What this pins is per-axis handling, not each operation. A no-flux wall read as a clamp moves the 1-D
reference with it; that is pinned in `tests/unit/test_geometry/test_mixed_bc_refused_1697.py`.

Each mutation below was patched in-process and run through this file. The table gives the residual
at which it failed; the shipped code passes at 2.6e-15 or less.

| mutation                                                  | HJB adi | HJB stochastic | FP-SL   |
|-----------------------------------------------------------|---------|----------------|---------|
| every axis given axis 0's operation (the old collapse)    | 2.5e-01 | 3.3e-02        | 1.7e-01 |
| every axis given the last axis's operation                | 1.8e-01 | 2.2e-01        | 5.1e-01 |
| the operations transposed                                 | 3.1e-01 | 2.2e-01        | 5.3e-01 |
| the fold collapsed, the diffusion still per axis          | 6.6e-02 | 3.3e-02        | 7.0e-02 |
| the ADI sweep handed axis 0's BC on every axis            | 5.9e-02 | passes         | 2.7e-03 |
| the diffusion types collapsed (ADI BC and identification) | 5.9e-02 | passes         | 1.5e-01 |
| every axis folded with axis 0's bounds                    | 4.0e-01 | passes         | 1.0e+00 |

The stochastic step has no ADI sweep, and its reference shares the fold's bounds with the channel
solve.

Sub-stepping is off. Its count is set by the largest velocity on the grid, so the channel and its
one-axis solves would cut their steps differently, and the law would hold only to O(dt): at Nt=10
the ADI residual is 5.6e-03 with sub-stepping on and 7.2e-16 with it off. At Nt=40 the CFL number,
computed from the solved U, is 0.54 in x and 0.10 in y, where the explicit-alpha* steps stay stable
without sub-stepping.

Not covered here:

- The DPP path (`L1ControlCost`) and `diffusion_method="canonical_cs"`. Both minimise per node, and
  that minimisation does not split by axis even on a uniform BC. On no-flux over [0, 1]^2,
  U[f + h] against U[f] + U[h] misses by 1.6e-02 on DPP and 3.4e-03 on canonical_cs, with
  h(y) = 0.3 cos(pi y) + 0.2 y (#2501).
- `diffusion_method="explicit"`. Its Laplacian reads no BC at all, uniform or not, so it is equally
  wrong in every solve and passes the law (#2504).
- The sub-stepped step, and 1-D, where there is only one axis.
"""

from __future__ import annotations

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc, periodic_bc

LX, LY = 1.0, 2.0  # unequal, so an axis folded with another axis's bounds is visible
NX, NY, NT, T, SIGMA = 21, 17, 40, 0.5, 0.3
# (bounds, nodes) of the channel and of each of its axes on its own
PLANE = ([(0.0, LX), (0.0, LY)], [NX, NY])
X_AXIS = ([(0.0, LX)], [NX])
Y_AXIS = ([(0.0, LY)], [NY])


def f(x):
    return 0.4 * np.sin(1.5 * np.pi * x) + 0.3 * x


def g(y):
    return 0.2 * np.sin(2 * np.pi * y / LY) + 0.1 * np.cos(4 * np.pi * y / LY)


def a(x):
    return 1.0 + 0.5 * np.cos(np.pi * x) + 0.3 * x


def b(y):
    return 1.0 + 0.4 * np.sin(2 * np.pi * y / LY)


def _channel() -> BoundaryConditions:
    walls = [BCSegment(name=f"x_{s}", bc_type=BCType.NO_FLUX, boundary=f"x_{s}") for s in ("min", "max")]
    seam = [BCSegment(name=f"y_{s}", bc_type=BCType.PERIODIC, boundary=f"y_{s}") for s in ("min", "max")]
    return BoundaryConditions(dimension=2, segments=walls + seam)


def _problem(domain, bc):
    bounds, npts = domain
    volume = float(np.prod([hi - lo for lo, hi in bounds]))
    grid = TensorProductGrid(bounds=bounds, Nx_points=npts, boundary_conditions=bc)
    problem = MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=SIGMA),
        domain=grid,
        conditions=Conditions(m_initial=lambda z: 1.0 / volume, u_terminal=lambda z: 0.0, T=T),
        Nt=NT,
    )
    return problem, tuple(npts), np.meshgrid(*grid.coordinates, indexing="ij")


def _hjb(domain, bc, terminal, diffusion_method):
    problem, npts, coords = _problem(domain, bc)
    solver = HJBSemiLagrangianSolver(problem, diffusion_method=diffusion_method, enable_adaptive_substepping=False)
    shape = (NT + 1, *npts)
    return solver.solve_hjb_system(np.ones(shape), terminal(*coords), np.zeros(shape))


def _fp(domain, bc, initial, potential):
    problem, npts, coords = _problem(domain, bc)
    U = np.broadcast_to(potential(*coords), (NT + 1, *npts)).copy()
    solver = FPSLSolver(problem, enable_adaptive_substepping=False)
    return solver.solve_fp_system(M_initial=initial(*coords), potential_field=U, show_progress=False)


def test_hjb_sl_channel_is_the_sum_of_its_axes_1d_solves():
    """ADI: the reference is the 1-D solver on each axis's own interval, another code path."""
    channel = _hjb(PLANE, _channel(), lambda X, Y: f(X) + g(Y), "adi")
    along_x = _hjb(X_AXIS, no_flux_bc(dimension=1), f, "adi")
    along_y = _hjb(Y_AXIS, periodic_bc(dimension=1), g, "adi")
    residual = float(np.abs(channel - along_x[:, :, None] - along_y[:, None, :]).max())
    assert residual < 1e-12, (
        f"max|U_channel - (U_1D,no-flux + U_1D,periodic)| = {residual:.3e}: an axis of the channel is not "
        "treated as the 1-D solver treats that axis on its own interval -- its fold, its bounds, its ADI BC "
        "or its periodic identification; or the 1-D and n-D schemes have diverged. A collapse to one "
        "operation for every axis measured 3.3e-01 (#1560)."
    )


def test_hjb_sl_stochastic_channel_is_the_sum_of_its_axes():
    """The stochastic step's quadrature takes 2d departures per node, so its 1-D step is a different scheme:
    the reference here is the same 2-D solver on uniform BCs, with data constant along the other axis."""
    channel = _hjb(PLANE, _channel(), lambda X, Y: f(X) + g(Y), "stochastic")
    axes = _hjb(PLANE, no_flux_bc(dimension=2), lambda X, Y: f(X) + 0 * Y, "stochastic") + _hjb(
        PLANE, periodic_bc(dimension=2), lambda X, Y: g(Y) + 0 * X, "stochastic"
    )
    residual = float(np.abs(channel - axes).max())
    assert residual < 1e-12, (
        f"max|U_channel - (U_no-flux + U_periodic)| = {residual:.3e}: an axis of the channel is not treated "
        "as the uniform solve of that axis treats it. A collapse to one operation for every axis measured "
        "7.2e-02 (#1560)."
    )


def test_fp_sl_channel_is_the_product_of_its_axes_1d_solves():
    channel = _fp(PLANE, _channel(), lambda X, Y: a(X) * b(Y), lambda X, Y: f(X) + g(Y))
    along_x = _fp(X_AXIS, no_flux_bc(dimension=1), a, f)
    along_y = _fp(Y_AXIS, periodic_bc(dimension=1), b, g)
    axes = along_x[:, :, None] * along_y[:, None, :]
    residual = float(np.abs(channel - axes).max() / np.abs(axes).max())
    assert residual < 1e-12, (
        f"max|M_channel - M_1D,no-flux * M_1D,periodic| / max = {residual:.3e}: an axis of the channel is not "
        "treated as the 1-D solver treats that axis on its own interval -- its fold, its bounds, its ADI BC "
        "or its periodic identification; or the 1-D and n-D schemes have diverged. A collapse to one "
        "operation for every axis measured 2.2e-01 (#1697)."
    )

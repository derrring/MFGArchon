"""The semi-Lagrangian pair applies each axis's own boundary condition (#1560, #1697).

ADMISSION (#2257). Class 2, an external oracle: separability. On a channel -- no-flux walls on one axis
over [0, 1], periodic on the other over [0, 2] -- a separable problem splits into its two axes. With w the
wall coordinate and s the seam coordinate:

- HJB-SL, H = |p|^2 / 2 and u_T = f(w) + g(s):  U_channel = U_wall(w) + U_seam(s).
- FP-SL, a potential U = f(w) + g(s) and m_0 = a(w) b(s):  M_channel = M_wall(w) M_seam(s).

U_wall and M_wall solve the problem on [0, 1] with no-flux walls, U_seam and M_seam on [0, 2] with a
periodic BC. The ADI and FP tests run with the seam in y and with it in x, so that a defect tied to an
axis's position rather than its BC shows.
The law is the PDE's. The scheme reproduces it to rounding because each axis's foot depends on that
axis alone, and every later stage is a tensor product: interpolation and splatting weights multiply
across axes, and a sweep along an axis leaves a field that is constant along it unchanged.

For HJB-SL with ADI and for FP-SL the right-hand side comes from the 1-D solver on each axis's own
interval. That is another code path, which agrees with the 2-D solve of one-axis data to 1.5e-15
(HJB) and 6.9e-15 (FP, relative), measured with the seam in y. The unequal intervals make an axis folded
with another axis's bounds visible.

The stochastic step's quadrature takes 2d departures per node, so its 1-D step is a different
scheme, 2.5e-02 away. Its reference is therefore the same 2-D solver on uniform BCs, with data
constant along the other axis. That compares the solver with itself, so a defect shared with the
uniform solves passes it.

What this pins is per-axis handling, not each operation. A no-flux wall read as a clamp moves the 1-D
reference with it; that is pinned in `tests/unit/test_geometry/test_mixed_bc_refused_1697.py`.

Each mutation below was patched in-process and run through this file. The table gives the residual
at which each case failed, or the exception it raised; the shipped code passes every case at 2.6e-15 or
less.

| mutation                                                  | ADI, seam y | ADI, seam x | stochastic | FP, seam y | FP, seam x |
|-----------------------------------------------------------|-------------|-------------|------------|------------|------------|
| every axis given axis 0's operation (the old collapse)    | 2.5e-01     | 1.8e-01     | 3.3e-02    | 1.7e-01    | 5.1e-01    |
| every axis given the last axis's operation                | 1.8e-01     | 2.5e-01     | 2.2e-01    | 5.1e-01    | 1.7e-01    |
| the operations transposed                                 | 3.1e-01     | 3.1e-01     | 2.2e-01    | 5.3e-01    | 5.3e-01    |
| the fold collapsed, the diffusion still per axis          | 6.6e-02     | 2.3e-01     | 3.3e-02    | 7.0e-02    | 9.9e-01    |
| the ADI sweep handed axis 0's BC on every axis            | 5.9e-02     | ValueError  | passes     | 2.7e-03    | ValueError |
| the diffusion types collapsed (ADI BC and identification) | 5.9e-02     | 1.4e-01     | passes     | 1.5e-01    | 2.6e-01    |
| every axis folded with axis 0's bounds                    | 4.0e-01     | 2.9e-05     | passes     | 1.0e+00    | passes     |
| periodic identification always along the last axis       | passes      | 1.3e-01     | passes     | passes     | ValueError |

The ValueError is the periodic sweep refusing a field whose end nodes were never identified. The
stochastic step has no ADI sweep, and its reference shares the fold's bounds with the channel solve.
Where the bounds row passes with the seam in x, the borrowed interval [0, 2] shares the wall axis's
lower end, and no foot crosses its upper wall.

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

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc, periodic_bc

# (interval, nodes) of the no-flux axis and of the periodic one. Unequal intervals, so an axis folded with
# the other axis's bounds is visible.
WALL = ((0.0, 1.0), 21)
SEAM = ((0.0, 2.0), 17)
NT, T, SIGMA = 40, 0.5, 0.3
SEAM_LENGTH = SEAM[0][1] - SEAM[0][0]

# Which axis is periodic. Both, so that a defect tied to an axis's position rather than its BC shows.
ORIENTATION = pytest.mark.parametrize("seam_axis", [1, 0], ids=["seam-in-y", "seam-in-x"])


def f(w):
    return 0.4 * np.sin(1.5 * np.pi * w) + 0.3 * w


def g(s):
    return 0.2 * np.sin(2 * np.pi * s / SEAM_LENGTH) + 0.1 * np.cos(4 * np.pi * s / SEAM_LENGTH)


def a(w):
    return 1.0 + 0.5 * np.cos(np.pi * w) + 0.3 * w


def b(s):
    return 1.0 + 0.4 * np.sin(2 * np.pi * s / SEAM_LENGTH)


def _channel(seam_axis) -> BoundaryConditions:
    wall, seam = "xy"[1 - seam_axis], "xy"[seam_axis]
    walls = [BCSegment(name=f"{wall}_{s}", bc_type=BCType.NO_FLUX, boundary=f"{wall}_{s}") for s in ("min", "max")]
    seams = [BCSegment(name=f"{seam}_{s}", bc_type=BCType.PERIODIC, boundary=f"{seam}_{s}") for s in ("min", "max")]
    return BoundaryConditions(dimension=2, segments=walls + seams)


def _plane(seam_axis):
    axes = (WALL, SEAM) if seam_axis == 1 else (SEAM, WALL)
    return [axis[0] for axis in axes], [axis[1] for axis in axes]


def _line(axis):
    return [axis[0]], [axis[1]]


def _on_plane(seam_axis, field):
    """``field(wall coordinate, seam coordinate)`` as a function of the plane's two coordinates."""
    return (lambda c0, c1: field(c0, c1)) if seam_axis == 1 else (lambda c0, c1: field(c1, c0))


def _outer(wall_part, seam_part, seam_axis, op):
    """Combine a 1-D solution along the wall axis with one along the seam axis onto the plane."""
    if seam_axis == 1:
        return op(wall_part[:, :, None], seam_part[:, None, :])
    return op(seam_part[:, :, None], wall_part[:, None, :])


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


@ORIENTATION
def test_hjb_sl_channel_is_the_sum_of_its_axes_1d_solves(seam_axis):
    """ADI: the reference is the 1-D solver on each axis's own interval, another code path."""
    channel = _hjb(_plane(seam_axis), _channel(seam_axis), _on_plane(seam_axis, lambda w, s: f(w) + g(s)), "adi")
    along_wall = _hjb(_line(WALL), no_flux_bc(dimension=1), f, "adi")
    along_seam = _hjb(_line(SEAM), periodic_bc(dimension=1), g, "adi")
    residual = float(np.abs(channel - _outer(along_wall, along_seam, seam_axis, np.add)).max())
    assert residual < 1e-12, (
        f"max|U_channel - (U_1D,no-flux + U_1D,periodic)| = {residual:.3e}: an axis of the channel is not "
        "treated as the 1-D solver treats that axis on its own interval -- its fold, its bounds, its ADI BC "
        "or its periodic identification; or the 1-D and n-D schemes have diverged. A collapse to one "
        "operation for every axis measured 2.5e-01 with the seam in y (#1560)."
    )


def test_hjb_sl_stochastic_channel_is_the_sum_of_its_axes():
    """The stochastic step's quadrature takes 2d departures per node, so its 1-D step is a different scheme:
    the reference here is the same 2-D solver on uniform BCs, with data constant along the other axis."""
    plane = _plane(1)
    channel = _hjb(plane, _channel(1), _on_plane(1, lambda w, s: f(w) + g(s)), "stochastic")
    axes = _hjb(plane, no_flux_bc(dimension=2), _on_plane(1, lambda w, s: f(w) + 0 * s), "stochastic") + _hjb(
        plane, periodic_bc(dimension=2), _on_plane(1, lambda w, s: g(s) + 0 * w), "stochastic"
    )
    residual = float(np.abs(channel - axes).max())
    assert residual < 1e-12, (
        f"max|U_channel - (U_no-flux + U_periodic)| = {residual:.3e}: an axis of the channel is not treated "
        "as the uniform solve of that axis treats it. A collapse to one operation for every axis measured "
        "3.3e-02 (#1560)."
    )


@ORIENTATION
def test_fp_sl_channel_is_the_product_of_its_axes_1d_solves(seam_axis):
    channel = _fp(
        _plane(seam_axis),
        _channel(seam_axis),
        _on_plane(seam_axis, lambda w, s: a(w) * b(s)),
        _on_plane(seam_axis, lambda w, s: f(w) + g(s)),
    )
    along_wall = _fp(_line(WALL), no_flux_bc(dimension=1), a, f)
    along_seam = _fp(_line(SEAM), periodic_bc(dimension=1), b, g)
    axes = _outer(along_wall, along_seam, seam_axis, np.multiply)
    residual = float(np.abs(channel - axes).max() / np.abs(axes).max())
    assert residual < 1e-12, (
        f"max|M_channel - M_1D,no-flux * M_1D,periodic| / max = {residual:.3e}: an axis of the channel is not "
        "treated as the 1-D solver treats that axis on its own interval -- its fold, its bounds, its ADI BC "
        "or its periodic identification; or the 1-D and n-D schemes have diverged. A collapse to one "
        "operation for every axis measured 1.7e-01 with the seam in y (#1697)."
    )

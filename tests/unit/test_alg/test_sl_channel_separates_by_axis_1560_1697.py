"""The semi-Lagrangian pair applies each axis's own boundary condition (#1560, #1697).

ADMISSION (#2257). Class 2, an external oracle: separability. On a channel -- no-flux walls in x,
periodic in y -- a separable problem splits into its two axes, and each axis is the same problem on
a uniform BC with data constant along the other axis:

- HJB-SL, H = |p|^2 / 2 and u_T = f(x) + g(y):  U_channel = U_no-flux[f] + U_periodic[g].
- FP-SL, a potential U = f(x) + g(y) and m_0 = a(x) b(y):  M_channel = M_no-flux[a] * M_periodic[b].

The law is the PDE's. The scheme reproduces it to rounding because each axis's foot depends on that
axis alone and every later stage is a tensor product: interpolation and splatting weights multiply
across axes, and a sweep along an axis leaves a field that is constant along it unchanged. So the
channel solve passes exactly when it applies, on each axis, what the uniform solve of that axis
applies.

Each mutation below was patched in-process and run through this file; each failed both segment
orders with the residual shown, where the shipped code passes at 1e-15 or less.

| mutation                                                  | HJB adi | HJB stochastic | FP-SL   |
|-----------------------------------------------------------|---------|----------------|---------|
| every axis given axis 0's operation (the old collapse)    | 3.3e-01 | 7.2e-02        | 2.2e-01 |
| every axis given the last axis's operation                | 1.8e-01 | 2.2e-01        | 5.1e-01 |
| the operations transposed                                 | 4.0e-01 | 2.2e-01        | 5.6e-01 |
| the fold collapsed, the diffusion still per axis          | 6.9e-02 | 7.2e-02        | 6.1e-02 |
| the ADI sweep handed axis 0's BC on every axis            | 6.1e-02 | passes         | 4.4e-03 |
| the diffusion types collapsed (ADI BC and identification) | 6.1e-02 | passes         | 3.4e-02 |

The stochastic step has no ADI sweep, so the last two rows cannot reach it.

Sub-stepping is off. Its count is set by the largest velocity on the grid, so the channel and its
one-axis solves would cut their steps differently, and the law would then hold only to O(dt): at
Nt=10 the adi residual is 8.4e-03 with sub-stepping on and 2.5e-16 with it off. Nt=40 keeps the CFL
number below 1 on both axes, where the explicit-alpha* steps stay stable without it.

Not covered, because there the law cannot tell a fold from an optimiser: the DPP path
(`L1ControlCost`) and `diffusion_method="canonical_cs"` minimise per node, and that minimisation does
not split by axis even on a uniform BC. On no-flux, U[f + h] against U[f] + U[h] misses by 1.6e-02
on DPP and 3.4e-03 on canonical_cs, with h(y) = 0.3 cos(pi y) + 0.2 y.
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

NX, NY, NT, T, SIGMA = 21, 17, 40, 0.5, 0.3


def f(x):
    return 0.4 * np.sin(1.5 * np.pi * x) + 0.3 * x


def g(y):
    return 0.2 * np.sin(2 * np.pi * y) + 0.1 * np.cos(4 * np.pi * y)


def a(x):
    return 1.0 + 0.5 * np.cos(np.pi * x) + 0.3 * x


def b(y):
    return 1.0 + 0.4 * np.sin(2 * np.pi * y)


def _channel(walls_first: bool) -> BoundaryConditions:
    walls = [BCSegment(name=f"x_{s}", bc_type=BCType.NO_FLUX, boundary=f"x_{s}") for s in ("min", "max")]
    seam = [BCSegment(name=f"y_{s}", bc_type=BCType.PERIODIC, boundary=f"y_{s}") for s in ("min", "max")]
    return BoundaryConditions(dimension=2, segments=walls + seam if walls_first else seam + walls)


def _problem(bc):
    grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[NX, NY], boundary_conditions=bc)
    problem = MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=SIGMA),
        domain=grid,
        conditions=Conditions(m_initial=lambda z: 1.0, u_terminal=lambda z: 0.0, T=T),
        Nt=NT,
    )
    return problem, np.meshgrid(*grid.coordinates, indexing="ij")


def _hjb(bc, terminal, diffusion_method):
    problem, (X, Y) = _problem(bc)
    solver = HJBSemiLagrangianSolver(problem, diffusion_method=diffusion_method, enable_adaptive_substepping=False)
    shape = (NT + 1, NX, NY)
    return solver.solve_hjb_system(np.ones(shape), terminal(X, Y), np.zeros(shape))


def _fp(bc, initial, potential):
    problem, (X, Y) = _problem(bc)
    U = np.broadcast_to(potential(X, Y), (NT + 1, NX, NY)).copy()
    solver = FPSLSolver(problem, enable_adaptive_substepping=False)
    return solver.solve_fp_system(M_initial=initial(X, Y), potential_field=U, show_progress=False)


ORDERS = pytest.mark.parametrize("walls_first", [True, False], ids=["walls-first", "seam-first"])


@ORDERS
@pytest.mark.parametrize("diffusion_method", ["adi", "stochastic"])
def test_hjb_sl_channel_is_the_sum_of_its_axes(diffusion_method, walls_first):
    channel = _hjb(_channel(walls_first), lambda X, Y: f(X) + g(Y), diffusion_method)
    axes = _hjb(no_flux_bc(dimension=2), lambda X, Y: f(X) + 0 * Y, diffusion_method) + _hjb(
        periodic_bc(dimension=2), lambda X, Y: g(Y) + 0 * X, diffusion_method
    )
    residual = float(np.abs(channel - axes).max())
    assert residual < 1e-12, (
        f"max|U_channel - (U_no-flux + U_periodic)| = {residual:.3e}: an axis of the channel is not "
        "treated as the uniform solve of that axis treats it -- its fold, its ADI BC or its periodic "
        "identification. A collapse to one operation for every axis measured 3.4e-01 on adi (#1560)."
    )


@ORDERS
def test_fp_sl_channel_is_the_product_of_its_axes(walls_first):
    channel = _fp(_channel(walls_first), lambda X, Y: a(X) * b(Y), lambda X, Y: f(X) + g(Y))
    axes = _fp(no_flux_bc(dimension=2), lambda X, Y: a(X) + 0 * Y, lambda X, Y: f(X) + 0 * Y) * _fp(
        periodic_bc(dimension=2), lambda X, Y: b(Y) + 0 * X, lambda X, Y: g(Y) + 0 * X
    )
    residual = float(np.abs(channel - axes).max() / np.abs(axes).max())
    assert residual < 1e-12, (
        f"max|M_channel - M_no-flux * M_periodic| / max = {residual:.3e}: an axis of the channel is not "
        "treated as the uniform solve of that axis treats it -- its fold, its ADI BC or its periodic "
        "identification. A collapse to one operation for every axis measured 2.2e-01 (#1697)."
    )

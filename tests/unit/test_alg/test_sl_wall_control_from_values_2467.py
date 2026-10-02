"""The SL pair takes U's gradient at a wall node from U, not from the FDM ghost (#2467).

Both halves differentiate U with the geometry's central operator, whose wall value is read from the ghost.
The cell-centred no-flux ghost made it half the one-sided slope; the node-centred mirror #1935 needs makes
it the boundary datum whatever U is, which freezes the wall characteristic. `value_gradient` now closes
every non-periodic wall with the one-sided difference of U, for the HJB half's control and the FP half's
drift alike.

Oracles, both exact and independent of the scheme:

- HJB: sigma = 0, quadratic cost, f(m) = m at m = 1, terminal |x - c|^2 / 2. The Hopf-Lax minimiser
  (x + tau c) / (1 + tau) lies in the domain, so no optimal path meets a wall and
  u = |x - c|^2 / (2 (1 + tau)) + tau, whose wall slope is not 0. A wall control taken from the ghost
  makes the wall rows the worst: wall / interior error at 101 points in 1-D is 1.61 from the half slope
  and 4.5 from the datum; edge 1.38 and corner 1.71 in 2-D at 21. From U it is 1.01, and 1.03 / 1.06.
- FP: translation by alpha = +0.4 from a density sitting on the low wall, sigma = 0, so the mean position
  moves by alpha T exactly. A wall node drifting at half speed lags: 3.7e-03 off at 51 points, 5.3e-04
  from U.
"""

from __future__ import annotations

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _hopf_lax_errors(n: int, centre: tuple[float, ...]) -> np.ndarray:
    """|U - u| over (t, x...) for the sigma = 0 closed form described above."""
    dims = len(centre)
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * dims, Nx_points=[n] * dims, boundary_conditions=no_flux_bc(dims))
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
    )
    problem = MFGProblem(
        model=Model(hamiltonian=hamiltonian, volatility=0.0),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=1.0),
        Nt=n - 1,
    )
    mesh = np.meshgrid(*grid.coordinates, indexing="ij")
    r2 = sum((axis - c) ** 2 for axis, c in zip(mesh, centre, strict=True))
    tau = (1.0 - np.linspace(0.0, 1.0, n)).reshape((n,) + (1,) * dims)
    exact = r2 / (2.0 * (1.0 + tau)) + tau
    shape = (n,) * (dims + 1)
    U = HJBSemiLagrangianSolver(problem).solve_hjb_system(np.ones(shape), 0.5 * r2, np.zeros(shape))
    return np.abs(U - exact)


def test_the_hjb_wall_row_is_no_worse_than_the_interior_in_1d():
    e = _hopf_lax_errors(101, (0.5,))
    wall, interior = max(e[:, 0].max(), e[:, -1].max()), e[:, 1:-1].max()
    assert wall < 1.1 * interior, (
        f"wall error {wall:.3e} against interior {interior:.3e}: the control at the wall is not U's slope "
        f"(1.61x from the ghost's half slope, 4.5x from the boundary datum; 1.01x from U) (#2467)"
    )


def test_the_hjb_edges_and_corners_are_no_worse_than_the_interior_in_2d():
    """c = (0.5, 0.4), so the two axes' walls see different slopes and a transposed closure shows."""
    e = _hopf_lax_errors(21, (0.5, 0.4))
    interior = e[:, 1:-1, 1:-1].max()
    edge = max(e[:, 0, 1:-1].max(), e[:, -1, 1:-1].max(), e[:, 1:-1, 0].max(), e[:, 1:-1, -1].max())
    corner = max(e[:, i, j].max() for i in (0, -1) for j in (0, -1))
    assert edge < 1.2 * interior, f"edge error {edge:.3e} against interior {interior:.3e} (1.38x before #2467)"
    assert corner < 1.2 * interior, f"corner error {corner:.3e} against interior {interior:.3e} (1.71x before #2467)"


def test_the_fp_drift_moves_wall_mass_at_the_velocity_of_u():
    n, T, speed = 51, 0.5, 0.4
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.0),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=T),
        Nt=n - 1,
    )
    x = grid.coordinates[0]
    m0 = np.exp(-200.0 * x**2)
    m0 /= grid.integrate(m0)
    M = FPSLSolver(problem).solve_fp_system(M_initial=m0, potential_field=np.tile(-speed * x, (n, 1)))

    fine = np.linspace(0.0, 1.0, 200_001)
    profile = np.exp(-200.0 * fine**2)
    exact = np.trapezoid(fine * profile, fine) / np.trapezoid(profile, fine) + speed * T
    moved = grid.integrate(M[-1] * x)
    assert abs(moved - exact) < 1.5e-3, (
        f"mean position {moved:.5f} at T, exact {exact:.5f}: the wall nodes are not drifting at -grad U "
        f"(3.7e-03 off from the ghost's half slope, 5.3e-04 from U) (#2467)"
    )


def test_a_periodic_seam_keeps_the_wrap():
    """On a periodic domain the seam has neighbours, so it is not a wall: the closure must not touch it.

    The central stencil on U = sin(2 pi (x - s)) returns cos(2 pi (x - s)) sin(2 pi h) / h exactly at every
    node -- the discrete operator is translation-invariant -- and only the wrap keeps that at the seam. Closing
    the seam one-sidedly is still consistent, which is why no periodic solve in the suite noticed: 207 passed
    with the closure applied on periodic domains too, and this fails.
    """
    from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_characteristics import value_gradient
    from mfgarchon.geometry.boundary import periodic_bc

    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=periodic_bc(dimension=1))
    x = grid.coordinates[0]
    h = float(grid.get_grid_spacing()[0])
    shift = 0.137
    (grad,) = value_gradient(grid, np.sin(2 * np.pi * (x - shift)), time=0.0)
    np.testing.assert_allclose(grad, np.cos(2 * np.pi * (x - shift)) * np.sin(2 * np.pi * h) / h, rtol=0, atol=1e-12)

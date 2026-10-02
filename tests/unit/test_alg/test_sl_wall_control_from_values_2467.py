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
  moves by alpha T exactly. A wall node drifting at half speed lags: 3.7e-03 off at 51 points, 5.4e-04
  from U.
"""

from __future__ import annotations

import pytest

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
    """c = (0.5, 0.4), so the two axes' walls see different slopes."""
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
        f"(3.7e-03 off from the ghost's half slope, 5.4e-04 from U) (#2467)"
    )


@pytest.mark.parametrize("periodic_first", [False, True])
def test_a_periodic_axis_keeps_its_wrap_beside_a_closed_one(periodic_first):
    """Periodicity is decided per axis, whatever order the segments come in.

    x is no-flux and y periodic. On x the closure is exact on a quadratic, walls included. On y the seam has
    neighbours, and the central stencil on sin(2 pi (y - s)) returns cos(2 pi (y - s)) sin(2 pi h) / h exactly
    at every node -- the discrete operator is translation-invariant -- which only the wrap keeps at the seam.
    Reading one type for the whole boundary takes the first segment's: no-flux first closes the seam,
    periodic first leaves the x walls on the ghost. Reachable through `FPSLSolver`'s own
    ``boundary_conditions``, which is m's and is what that solver validates. Closing the seam is still
    consistent, which is why no periodic solve in the suite noticed it.
    """
    from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_characteristics import value_gradient
    from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

    walls = [BCSegment(name=f"x{s}", boundary=f"x_{s}", bc_type=BCType.NO_FLUX) for s in ("min", "max")]
    seam = [BCSegment(name=f"y{s}", boundary=f"y_{s}", bc_type=BCType.PERIODIC) for s in ("min", "max")]
    bc = BoundaryConditions(dimension=2, segments=seam + walls if periodic_first else walls + seam)
    grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[9, 41], boundary_conditions=bc)
    x, y = np.meshgrid(*grid.coordinates, indexing="ij")
    h = float(grid.get_grid_spacing()[1])
    shift = 0.137
    gx, gy = value_gradient(grid, 0.7 * (x - 0.3) ** 2 + np.sin(2 * np.pi * (y - shift)), time=0.0)
    np.testing.assert_allclose(gx, 1.4 * (x - 0.3), rtol=0, atol=1e-11, err_msg="the closed axis is not closed")
    np.testing.assert_allclose(
        gy,
        np.cos(2 * np.pi * (y - shift)) * np.sin(2 * np.pi * h) / h,
        rtol=0,
        atol=1e-11,
        err_msg="the seam lost its wrap",
    )


def _half_periodic():
    from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

    return BoundaryConditions(
        dimension=2,
        default_bc=BCType.NO_FLUX,
        segments=[BCSegment(name="seam", boundary="x_min", bc_type=BCType.PERIODIC)],
    )


def _region_named_mix():
    from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

    walls = [BCSegment(name=f"x{s}", region_name=f"x_{s}", bc_type=BCType.NO_FLUX) for s in ("min", "max")]
    seam = [BCSegment(name=f"y{s}", region_name=f"y_{s}", bc_type=BCType.PERIODIC) for s in ("min", "max")]
    return BoundaryConditions(dimension=2, default_bc=BCType.NO_FLUX, segments=walls + seam)


@pytest.mark.parametrize(
    ("make_bc", "message"),
    [(_half_periodic, "periodic on one face only"), (_region_named_mix, "without naming a face")],
    ids=["half-periodic-axis", "region-named-mix"],
)
def test_a_boundary_whose_axes_cannot_be_read_per_face_is_refused(make_bc, message):
    """Two shapes the per-face reading cannot settle, refused rather than guessed.

    An axis periodic on one face has no answer: the ghosts would wrap one wall and mirror the other. A
    segment addressed by region name covers every face for `get_bc_type_at_boundary`, while the ghosts'
    resolver matches it to one face, so in a periodic mix the two disagree: this BC, read in segment order,
    closed the periodic seam or left the walls on the ghost, a drift difference of 0.36.
    """
    from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_characteristics import value_gradient

    grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[9, 9], boundary_conditions=make_bc())
    x, _ = np.meshgrid(*grid.coordinates, indexing="ij")
    with pytest.raises(NotImplementedError, match=message):
        value_gradient(grid, x**2, time=0.0)


@pytest.mark.parametrize("shape", [(7, 11), (9, 2)])
def test_the_closure_is_exact_where_its_stencil_is(shape):
    """Both second-order stencils are exact on a quadratic, and on a 2-point axis the one difference is exact
    on a linear function, so `value_gradient` must reproduce the analytic gradient to rounding.

    The grid is anisotropic and not square, so a closure that takes the wrong spacing, the wrong axis, the
    wrong wall, the wrong sign or a first-order stencil fails here, where the solver-level oracles above
    pass several of those (a first-order closure: 1.01 / 1.03 / 1.06).
    """
    from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_characteristics import value_gradient

    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (-0.5, 1.5)], Nx_points=list(shape), boundary_conditions=no_flux_bc(dimension=2)
    )
    x, y = np.meshgrid(*grid.coordinates, indexing="ij")
    a, b, c, d, e, f = np.random.default_rng(2467).normal(size=6)
    yy = f if shape[1] > 2 else 0.0  # a 2-point axis cannot carry curvature
    U = a * x + b * y + c * x**2 + d * x * y + e + yy * y**2
    gx, gy = value_gradient(grid, U, time=0.0)
    np.testing.assert_allclose(gx, a + 2 * c * x + d * y, rtol=0, atol=1e-11)
    np.testing.assert_allclose(gy, b + d * x + 2 * yy * y, rtol=0, atol=1e-11)

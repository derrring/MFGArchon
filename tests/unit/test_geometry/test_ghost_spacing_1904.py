"""The ghost buffer must use the grid's spacing, not fall back to dx = 1.0.

`PreallocatedGhostBuffer` derived its spacing only from `domain_bounds`, and almost no caller
passes those: `pad_array_with_ghosts(..., geometry=None)` left `_grid_spacing = None` and every
consumer read `dx = 1.0`. An inhomogeneous Neumann condition was therefore applied as `g/h`
instead of `g`. Reading the flux back off the ghost, `-(padded[1] - padded[0])/dx`, returned
exactly `g/h` for a requested `g = 2` -- and does so for any state, because that expression is the
algebraic inverse of the ghost write:

    Nx = 21 / 41 / 81 / 161   ->   40 / 80 / 160 / 320      before
                              ->   2.0 at every Nx          after

So what the assertions below certify is that the ghost ENCODES what the caller asked for. The ghost
mirrors about the wall node since #1935, `ghost = u_1 + 2 h g`, so the read-back is the central
difference about that node, `(ghost - u_1) / (2h)` outward; until #1935 it was the one-sided
`(ghost - u_0) / h` of a cell-centred ghost, and the centred wall gradient the HJB path builds from it
converged to `g/2 - u'(wall)/2` -- the node-centring half of #1904, which #1935 closed. The centred
wall gradient is now the requested flux exactly.

Robin read the same fallback, in the denominator `alpha + beta/dx`.

Found by review of #1902 (which is blocked on #1904, the node-centring half of the same function).
`HJBFDMSolver.honors_inhomogeneous_neumann` is True and pinned, so the declaration was live and
wrong -- the existing applicator test builds its buffer WITH `domain_bounds`, validating the
instrument on a path the solver never takes, which is why this passed.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers.base_hjb import _compute_gradient_array_1d, _compute_laplacian_1d
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, neumann_bc, no_flux_bc, robin_bc
from mfgarchon.geometry.boundary.applicator_fdm import pad_array_with_ghosts


@pytest.mark.parametrize("nx", [21, 41, 81, 161])
def test_inhomogeneous_neumann_recovers_the_requested_flux_at_every_resolution(nx: int):
    """The law: what the caller asked for is what the ghost encodes, independent of h.

    Read back about the wall node, where the ghost is written (#1935). The fixture must keep the two
    ghosts apart: the cell-centred `u_0 + h g` and the node-centred `u_1 + 2 h g` coincide exactly when
    `(u_1 - u_0)/h = -g`, and then the read-back cannot tell which one was written. Measured here,
    `(u_1 - u_0)/h` is -0.49 at 21 points and tends to `u'(0) = 0`, clear of -2.
    """
    g = 2.0
    dx = 1.0 / (nx - 1)
    u = 0.5 * np.cos(2 * np.pi * np.linspace(0.0, 1.0, nx))
    assert abs((u[1] - u[0]) / dx + g) > 1e-6, "fixture is degenerate: the two ghost conventions coincide"
    padded = pad_array_with_ghosts(u, neumann_bc(dimension=1, value=g), ghost_depth=1, time=0.0, spacing=dx)

    # Outward normal is -x at the low wall, +x at the high wall; padded = [ghost, u_0, u_1, ...].
    low = (padded[0] - padded[2]) / (2 * dx)
    high = (padded[-1] - padded[-3]) / (2 * dx)
    assert low == pytest.approx(g, abs=1e-12), f"nx={nx}: du/dn = {low} at the low wall, requested {g}"
    assert high == pytest.approx(g, abs=1e-12), f"nx={nx}: du/dn = {high} at the high wall, requested {g}"


def test_the_fallback_would_diverge_which_is_why_the_spacing_is_threaded():
    """The un-threaded call used to fall back to dx = 1.0, so the flux scaled as 1/h, and ten
    production call sites took it (#2140). It is refused now. The control keeps the test above honest:
    an explicit spacing of 1.0 is what the fallback substituted and 0.05 is not, and the two ghosts
    differ, so the threading is what the law above measures rather than something that was always
    true. Ghost = u_1 + 2 h g (#1935): 2 + 4 at h = 1, 2 + 0.2 at h = 0.05.
    """
    u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bc = neumann_bc(dimension=1, value=2.0)
    with pytest.raises(TypeError, match="#2140"):
        pad_array_with_ghosts(u, bc, ghost_depth=1, time=0.0)
    at_one = pad_array_with_ghosts(u, bc, ghost_depth=1, time=0.0, spacing=1.0)
    with_small = pad_array_with_ghosts(u, bc, ghost_depth=1, time=0.0, spacing=0.05)
    assert at_one[0] == pytest.approx(6.0), "spacing 1.0 no longer gives the fallback's ghost"
    assert with_small[0] == pytest.approx(2.2), "the threaded call does not use the spacing given"
    assert at_one[0] != with_small[0], "spacing makes no difference; the threading is inert"


@pytest.mark.parametrize("scalar_or_sequence", [0.05, [0.05], (0.05,)])
def test_spacing_accepts_a_scalar_or_one_value_per_axis(scalar_or_sequence):
    u = np.array([1.0, 2.0, 3.0])
    bc = neumann_bc(dimension=1, value=1.0)
    padded = pad_array_with_ghosts(u, bc, ghost_depth=1, time=0.0, spacing=scalar_or_sequence)
    assert padded[0] == pytest.approx(2.1)  # u_1 + 2 h g (#1935)


def test_a_wrong_length_spacing_raises_rather_than_broadcasting_silently():
    """Fail loud: a 2-entry spacing for a 1-D array is a caller bug, not something to guess at."""
    u = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="one per axis"):
        pad_array_with_ghosts(u, neumann_bc(dimension=1, value=1.0), ghost_depth=1, time=0.0, spacing=[0.1, 0.2])


def test_zero_flux_is_unaffected_because_it_multiplies_the_spacing_by_zero():
    """Scope control: no-flux must be byte-identical, or this change is wider than claimed.

    The node-centring defect in the same function (#1904) is NOT addressed here and this test
    would not see it -- it compares the two spellings against each other, not against the truth.
    """
    u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    for bc in (no_flux_bc(dimension=1), neumann_bc(dimension=1, value=0.0)):
        a = pad_array_with_ghosts(u, bc, ghost_depth=1, time=0.0, spacing=1.0)
        b = pad_array_with_ghosts(u, bc, ghost_depth=1, time=0.0, spacing=0.05)
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("nx", [21, 81])
@pytest.mark.parametrize("g", [2.0, -3.0])
def test_the_solver_gradient_path_now_carries_the_spacing(nx: int, g: float):
    """End-to-end through `_compute_gradient_array_1d`, which is what the residual calls.

    The law, not a band: the ghost differs from the `g = 0` ghost by exactly `2*dx*g` (#1935; `dx*g`
    before it, and the shifts were `-g/2` and `+g/2`), and the centred difference divides by `2*dx`,
    so requesting `g` shifts the wall rows by exactly `-g` and `+g` -- for ANY interior field and ANY
    spacing. A random field is used to
    make that independence part of the assertion rather than a remark.

    ~~`1e-9 < abs(tight[0] - loose[0]) < 5.0`~~ was the original form and pinned nothing
    [CORRECTED 2026-08-13, found by independent review of #1906]: that band admits the correct
    1.0, the halved 0.5, and the measured-but-wrong 1.2447 alike.
    """
    dx = 1.0 / (nx - 1)
    u = np.random.default_rng(7).normal(size=nx)
    tight = _compute_gradient_array_1d(u, dx, bc=neumann_bc(dimension=1, value=g), upwind=False, time=0.0)
    loose = _compute_gradient_array_1d(u, dx, bc=neumann_bc(dimension=1, value=0.0), upwind=False, time=0.0)
    assert tight[0] - loose[0] == pytest.approx(-g, abs=1e-12), f"nx={nx}, g={g}: low wall"
    assert tight[-1] - loose[-1] == pytest.approx(g, abs=1e-12), f"nx={nx}, g={g}: high wall"


@pytest.mark.parametrize("dx", [0.5, 0.05, 0.01])
def test_robin_reads_the_same_spacing(dx: float):
    """Robin used the fallback too, in the denominator `alpha + beta/dx`.

    The law: the ghost is the exact algebraic solution of the condition the applicator writes,
    `alpha*(u_g + u_i)/2 + beta*(u_g - u_i)/dx*outward_sign = g`, FOR THE SPACING GIVEN. Assert
    that residual at both walls. It is machine-zero when the spacing is threaded and O(1/dx)
    when the buffer substitutes 1.0 -- 32 and 105 at dx = 0.05 and 0.01 on this fixture.

    ~~`coarse[0] != fine[0]`~~ was the original form and pinned nothing [CORRECTED 2026-08-13,
    found by independent review of #1906]: two spacings give different ghosts under any
    denominator whatever, and it probed only index 0.

    RE-POINTED 2026-08-16, when #1907 landed. This previously asserted the residual of the
    equation the applicator then wrote, `... /dx*outward_sign = g` with `outward_sign = -1` at
    the low wall, and said so in this note: "deliberately NOT the oracle for that defect, and it
    will need re-pointing when #1907 lands". The applicator now writes the condition
    `BCSegment.beta` declares -- `du/dn` the OUTWARD normal derivative -- and on a cell-centred
    grid the ghost lies outside at both walls, so `(u_g - u_i)/dx` IS that derivative and no
    further sign is due. The two `outward_sign` factors are therefore gone from the oracle, and
    with them the last place in the suite that asserted the low wall's old convention.

    The spacing property this test exists for is untouched: the residual is machine-zero when
    the spacing is threaded and O(1/dx) when the buffer substitutes 1.0.

    RE-POINTED for #1935: the ghost is node-centred, so the condition holds at the wall node,
    `alpha*u_b + beta*(u_g - u_m)/(2 dx) = g` with `u_b` the wall node and `u_m` its mirror.
    """
    u = np.random.default_rng(7).normal(size=7)
    alpha, beta, g = 1.0, 1.0, 0.5
    bc = robin_bc(dimension=1, alpha=alpha, beta=beta, value=g)
    p = pad_array_with_ghosts(u, bc, ghost_depth=1, time=0.0, spacing=dx)
    low = alpha * p[1] + beta * (p[0] - p[2]) / (2 * dx) - g
    high = alpha * p[-2] + beta * (p[-1] - p[-3]) / (2 * dx) - g
    assert low == pytest.approx(0.0, abs=1e-12), f"dx={dx}: Robin low wall residual {low:.3e}"
    assert high == pytest.approx(0.0, abs=1e-12), f"dx={dx}: Robin high wall residual {high:.3e}"


@pytest.mark.parametrize("h", [0.5, 0.05, 0.01])
@pytest.mark.parametrize("g", [3.0, -1.5])
def test_the_laplacian_path_threads_its_own_spacings_parameter(h: float, g: float):
    """`laplacian_with_bc` already took `spacings` and dropped it at the padding call.

    The law: the wall row is `(u[1] - 2*u[0] + ghost)/h^2` with `ghost = u[1] + 2*h*g` (#1935), so
    `lap[0]*h^2 - 2*(u[1] - u[0])` is exactly `2*h*g` -- for any field and any h. Under the
    `dx = 1.0` fallback the ghost carries `2*g` instead and the identity misses by `2*(1-h)*g`.

    ~~`not np.allclose(coarse[0]*0.5**2, fine[0]*0.01**2)`~~ was the original form and pinned
    nothing [CORRECTED 2026-08-13, found by independent review of #1906]: it asserts only that
    the wall row is not spacing-independent, which almost any wrong formula also satisfies.
    """
    u = np.random.default_rng(7).normal(size=6)
    lap = _compute_laplacian_1d(u, h, bc=neumann_bc(dimension=1, value=g), time=0.0)
    assert lap[0] * h * h - 2 * (u[1] - u[0]) == pytest.approx(2 * h * g, abs=1e-12), f"h={h}, g={g}: wall row"


@pytest.mark.parametrize("nx", [11, 41])
def test_the_gradient_operator_reproduces_a_linear_slope_at_the_wall(nx):
    """External oracle through the production path (#2140): `TensorProductGrid.get_gradient_operator`.

    For u = a x the central difference through the ghost is exact once the ghost carries the outward
    derivative at the grid's spacing: ghost = u_1 + 2 h g with g = -a at the low wall gives
    (u_1 - ghost) / (2h) = a, and likewise at the high wall (#1935; the cell-centred ghost u_0 + h g was
    exact on a linear field too, which is why this test did not move). Until #2140 the operator built the ghost at
    h = 1.0; on u = -0.3 (x - 1/2)^2 with its exact data -0.3, its wall gradient was +1.635 / +3.143 /
    +6.146 at 11 / 21 / 41 points against an exact +0.300, growing as 1/h.
    """
    a = 0.7
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=neumann_bc(dimension=1, value=0.0)
    )
    x = grid.coordinates[0]
    # Outward normal is -x at the low wall and +x at the high wall: du/dn = -a there and +a here.
    bc_low_high = BoundaryConditions(
        segments=[
            BCSegment(name="low", bc_type=BCType.NEUMANN, value=-a, boundary="x_min"),
            BCSegment(name="high", bc_type=BCType.NEUMANN, value=a, boundary="x_max"),
        ],
        dimension=1,
    )
    slope = grid.get_gradient_operator(scheme="central", bc=bc_low_high)[0](a * x)
    np.testing.assert_allclose(slope, a, rtol=0, atol=1e-12)


def test_a_linear_field_on_an_anisotropic_grid_is_exact_on_every_axis():
    """External oracle in 2-D, so axis order is observable (#2140): dx = 0.1, dy = 0.4.

    For u = a x + b y with outward Neumann data -a, +a, -b, +b on the four faces, a ghost built at each
    axis's own spacing makes the central gradient exactly (a, b) and the Laplacian exactly 0 at every
    node, walls included. A spacing of 1.0, or the two spacings swapped, moves the wall values. Checked
    through the operator and through `tensor_calculus`, which threads the spacing separately.
    """
    from mfgarchon.utils.numerical.tensor_calculus import gradient, laplacian

    a, b = 0.7, -0.4
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="xl", bc_type=BCType.NEUMANN, value=-a, boundary="x_min"),
            BCSegment(name="xh", bc_type=BCType.NEUMANN, value=a, boundary="x_max"),
            BCSegment(name="yl", bc_type=BCType.NEUMANN, value=-b, boundary="y_min"),
            BCSegment(name="yh", bc_type=BCType.NEUMANN, value=b, boundary="y_max"),
        ],
        dimension=2,
    )
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 2.0)], Nx_points=[11, 6], boundary_conditions=neumann_bc(dimension=2, value=0.0)
    )
    X, Y = np.meshgrid(*grid.coordinates, indexing="ij")
    u = a * X + b * Y
    ops = grid.get_gradient_operator(scheme="central", bc=bc)
    np.testing.assert_allclose(ops[0](u), a, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ops[1](u), b, rtol=0, atol=1e-12)
    du = gradient(u, spacings=grid.get_grid_spacing(), bc=bc)
    np.testing.assert_allclose(du[0], a, rtol=0, atol=1e-12)
    np.testing.assert_allclose(du[1], b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(laplacian(u, spacings=grid.get_grid_spacing(), bc=bc), 0.0, rtol=0, atol=1e-10)

"""Robin's `du/dn` is the outward normal derivative, at both walls. #1907

The low wall applied the outward sign a second time. On a cell-centred grid the ghost lies
outside at both walls, so the quotient toward it, `(u_g - u_i)/dx`, IS the outward normal
derivative and no further factor is due. Multiplying by `outward_sign = -1` there imposes
`alpha*u + beta*du/dx = g` -- the axis condition, a physically different wall.

Measured on `main` before the fix, `alpha=1, beta=0.3, g=0.7`: the declared condition's residual
was **6.17 at the min wall** and 0 at the max.

**The oracle is the condition itself, not the other code path.** Four implementations wrote this
formula; the fix routes all of them through `ghost_cell_robin`, after which comparing any two is
tautological and would pass over a broken owner. So every assertion below evaluates
`alpha*u_b + beta*du/dn - g` on the ghost the shipped API returned.

**Since #1935 the ghost is node-centred**: the wall is a node `u_b`, ghost and mirror `u_m` lie `dx`
outside and inside it, and the condition is imposed by the central difference about the node,
`du/dn = (u_g - u_m)/(2 dx)`. The outward-normal argument above is unchanged -- the ghost is outside at
both walls -- and the residual below is taken at the node accordingly.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    neumann_bc,
    pad_array_with_ghosts,
    robin_bc,
)

_U = np.array([2.5, 3.1, 4.0, 5.2, 6.6])
_DX = 0.25


def _residual(ghost: float, wall: float, mirror: float, alpha: float, beta: float, g: float, dx: float) -> float:
    """`alpha*u_b + beta*du/dn - g` at the wall node, with `du/dn` the central quotient toward the ghost.

    Side-free by construction: at the low wall the outward normal is $-x$ and the ghost sits at
    lower $x$, so $\\partial_n u = -\\partial_x u = (u_g - u_m)/(2\\,dx)$; at the high wall the outward
    normal is $+x$ and the ghost sits at higher $x$, giving the same expression (#1935).
    """
    return alpha * wall + beta * (ghost - mirror) / (2 * dx) - g


def _uniform_robin(alpha: float, beta: float, g: float) -> BoundaryConditions:
    return BoundaryConditions(
        segments=[BCSegment(name="wall", bc_type=BCType.ROBIN, value=g, alpha=alpha, beta=beta)],
        dimension=1,
        default_bc=BCType.ROBIN,
        default_value=g,
        domain_bounds=np.array([[0.0, 1.0]]),
    )


@pytest.mark.parametrize(
    ("alpha", "beta", "g"),
    [
        (1.0, 0.3, 0.7),  # the case measured at residual 6.17 before the fix
        (1.0, 0.3, 0.0),  # homogeneous: the conventions still differ, beta is what matters
        (0.0, 1.0, 0.7),  # pure flux
        (2.0, 0.5, 0.0),
        (3.0, -0.4, 1.1),  # negative beta
        (0.5, 2.0, -0.3),  # negative g
    ],
)
@pytest.mark.parametrize("wall", ["min", "max"])
def test_the_ghost_satisfies_the_declared_condition_at_both_walls(alpha, beta, g, wall):
    """`BCSegment.beta` is documented "Weight on du/dn (Neumann term)" and `BCType.ROBIN` as
    `alpha*u + beta*du/dn = g`. That is the contract; this asserts the shipped ghost meets it.

    Both walls are parametrised on purpose. The max wall passed before the fix and passes now --
    a one-wall test would have been green throughout and pinned nothing.
    """
    padded = pad_array_with_ghosts(_U, _uniform_robin(alpha, beta, g), spacing=_DX)
    ghost, node, mirror = (padded[0], _U[0], _U[1]) if wall == "min" else (padded[-1], _U[-1], _U[-2])

    assert _residual(ghost, node, mirror, alpha, beta, g, _DX) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("g", [0.0, 2.0, -1.5])
def test_robin_with_alpha_zero_beta_one_is_the_neumann_condition(g):
    """The headline of #1907: `robin_bc(g, alpha=0, beta=1)` and `neumann_bc(g)` are the same
    condition, `du/dn = g`, and they disagreed at the low wall by exactly `2*dx*g`.

    Measured before the fix at `g = 2.0`, spacing threaded: `neumann` gave 0.600000 and
    `robin(0,1)` gave 0.400000 on a 21-point grid, difference `0.2 = 2*h*g`, shrinking with `h`
    and identically zero at the high wall. `g = 0` is included as the degenerate row where the
    two agreed even before the fix -- without it a test that simply returned the Neumann value
    would pass, and with it alone the whole property would look already satisfied.
    """
    robin = pad_array_with_ghosts(_U, robin_bc(dimension=1, alpha=0.0, beta=1.0, value=g), spacing=_DX)
    neumann = pad_array_with_ghosts(_U, neumann_bc(dimension=1, value=g), spacing=_DX)

    assert robin[0] == pytest.approx(neumann[0], abs=1e-12), "low wall"
    assert robin[-1] == pytest.approx(neumann[-1], abs=1e-12), "high wall"


def test_a_uniform_robin_wall_is_the_same_physics_at_both_ends():
    """The user-visible consequence, stated without reference to any formula.

    `uniform Robin` reads as "the same condition on every wall". Under the axis convention it was
    not: the low wall got `alpha*u + beta*du/dx = g` and the high wall `alpha*u + beta*du/dn = g`,
    which are different physical walls. A field that is symmetric about the domain centre must
    therefore produce symmetric ghosts.
    """
    symmetric = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    padded = pad_array_with_ghosts(symmetric, _uniform_robin(1.0, 0.4, 0.9), spacing=_DX)

    assert padded[0] == pytest.approx(padded[-1], abs=1e-12), (
        "a symmetric field under a wall-symmetric condition must give symmetric ghosts"
    )


def test_the_fixture_would_expose_an_asymmetry():
    """Control for the test above. The symmetric field makes the two walls comparable only
    because the interior values at the two ends are equal; this asserts the probe can still tell
    the walls apart when the condition genuinely differs between them, so the symmetry check is
    not vacuous."""
    symmetric = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    mixed = BoundaryConditions(
        segments=[
            BCSegment(name="lo", bc_type=BCType.ROBIN, value=0.9, alpha=1.0, beta=0.4, boundary="x_min"),
            BCSegment(name="hi", bc_type=BCType.DIRICHLET, value=0.9, boundary="x_max"),
        ],
        dimension=1,
        default_bc=BCType.ROBIN,
        default_value=0.9,
        domain_bounds=np.array([[0.0, 1.0]]),
    )
    padded = pad_array_with_ghosts(symmetric, mixed, spacing=_DX)

    assert padded[0] != pytest.approx(padded[-1], abs=1e-12)


def test_a_singular_coefficient_raises_rather_than_mirroring():
    """A coefficient pair that leaves the ghost undetermined raises; mirroring the interior there,
    as two of the four deleted implementations did, answers a question the caller did not ask.

    Which pair that is depends on the centring. The cell-centred form was singular at
    `alpha/2 + beta/dx = 0` -- `alpha = 2, beta = -0.25` at `dx = 0.25`, which the node-centred form
    of #1935 determines. There the only undetermined case is `alpha = beta = 0`, a condition that
    constrains nothing; `beta = 0` alone is Dirichlet and is computed.
    """
    padded = pad_array_with_ghosts(_U, _uniform_robin(2.0, -0.25, 0.5), spacing=_DX)
    assert _residual(padded[0], _U[0], _U[1], 2.0, -0.25, 0.5, _DX) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="singular"):
        pad_array_with_ghosts(_U, _uniform_robin(0.0, 0.0, 0.5), spacing=_DX)


@pytest.mark.parametrize(
    ("bc_factory", "label"),
    [
        (lambda: neumann_bc(dimension=1, value=0.0), "NEUMANN"),
        (lambda: robin_bc(dimension=1, alpha=1.0, beta=0.0, value=0.7), "ROBIN with beta=0"),
    ],
)
def test_the_cells_that_must_not_have_moved(bc_factory, label):
    """Nine of the thirteen probed cells were byte-identical across #1907; these are the two that
    could plausibly have been disturbed. `beta = 0` removes the derivative term, so the conventions
    coincide there -- a fix that changed it would be a regression, and #1935 did not.

    Homogeneous Neumann did move in #1935, on purpose: its ghost mirrors the node beside the wall,
    `u_1`, where it copied the wall node `u_0`. `beta = 0` moved in #1968 for the same reason: its ghost
    is the odd reflection `2 g/alpha - u_1` through the node beside the wall, where it was `2 g/alpha - u_0`."""
    padded = pad_array_with_ghosts(_U, bc_factory(), spacing=_DX)

    if label == "NEUMANN":
        assert padded[0] == pytest.approx(_U[1])
        assert padded[-1] == pytest.approx(_U[-2])
    else:
        assert padded[0] == pytest.approx(2 * 0.7 - _U[1])
        assert padded[-1] == pytest.approx(2 * 0.7 - _U[-2])

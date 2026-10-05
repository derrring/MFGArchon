"""The FDM ghost mirrors about the wall node, because the wall is a node (#1935).

`TensorProductGrid` puts x_0 on the wall, h = span/(N-1). The ghost used to copy the wall node,
u_g = u_0 + h g -- the cell-centred mirror -- so the 3-point Laplacian's wall row read
(u_1 - u_0)/h^2 -> u''/2 for a zero-flux field: half the true value at every h, order 0.00. The node
mirror u_g = u_1 + 2 h (g - alpha u_0)/beta makes it u'' + O(h^2), for Neumann (alpha = 0, beta = 1)
and Robin alike, in both ghost paths.

Oracles, all exact laws rather than a path compared with a path:

- u = cos(pi x) has u' = 0 at both walls, so no-flux is its exact data and u'' = -pi^2 cos(pi x) is the
  Laplacian; u(0) != u(1), so the periodic wrap cannot impersonate the mirror.
- u = cos(pi x) + 0.7 x + 2 has a wall slope, so the Robin data g = alpha u + beta du/dn exercises alpha
  and its sign. With du/dn = 0 instead, g = alpha u and the alpha term cancels out of the ghost.

The sparse assembly, node-centred already, now agrees with the matvec at the walls too; that is pinned in
`test_laplacian.py::test_sparse_matches_matvec_neumann`.
"""

from __future__ import annotations

import doctest

import pytest

import numpy as np

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc


def _wall_and_interior_errors(n: int, bc, u, lap_true) -> tuple[float, float]:
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=bc)
    x = grid.coordinates[0]
    lap = np.asarray(grid.get_laplacian_operator() @ u(x)).ravel()
    e = np.abs(lap - lap_true(x))
    return max(e[0], e[-1]), float(e[1:-1].max())


def test_the_no_flux_wall_row_converges_at_second_order():
    """Measured at the cell-centred ghost: wall error 4.937 / 4.935 at 41 / 81 points (order 0.00), against
    an interior error of 1.3e-03 at 81."""
    u = lambda x: np.cos(np.pi * x)  # noqa: E731
    lap_true = lambda x: -(np.pi**2) * np.cos(np.pi * x)  # noqa: E731
    coarse, _ = _wall_and_interior_errors(41, no_flux_bc(dimension=1), u, lap_true)
    wall, interior = _wall_and_interior_errors(81, no_flux_bc(dimension=1), u, lap_true)
    assert np.log2(coarse / wall) > 1.9, f"wall order {np.log2(coarse / wall):.2f} (0.00 with the cell mirror)"
    assert wall < 1.1 * interior, f"wall error {wall:.3e} against interior {interior:.3e}"

    # Control: the ghost path is what the wall row reads. Without a BC the operator wraps, and the wall row of
    # this non-periodic field is wrong by O(1/h^2) -- 795 at 21 points in #1935's measurement.
    from mfgarchon.operators.differential.laplacian import LaplacianOperator

    x = np.linspace(0.0, 1.0, 81)
    bare = np.asarray(LaplacianOperator(spacings=[x[1] - x[0]], field_shape=(81,), bc=None) @ u(x)).ravel()
    assert abs(bare[0] - lap_true(x)[0]) > 100 * wall, "without a BC the wall row did not move: the ghost is not read"


@pytest.mark.parametrize(("alpha", "beta"), [(0.0, 1.0), (1.0, 1.0), (2.0, 0.5), (-0.5, 2.0)])
def test_a_robin_wall_with_a_slope_converges_at_second_order(alpha, beta):
    """Measured at the cell-centred ghost: order 0.00 for every row, wall error 4.8-6.3 at 161 points."""
    u = lambda x: np.cos(np.pi * x) + 0.7 * x + 2.0  # noqa: E731
    lap_true = lambda x: -(np.pi**2) * np.cos(np.pi * x)  # noqa: E731
    bc = BoundaryConditions(
        segments=[
            BCSegment(
                name="l",
                bc_type=BCType.ROBIN,
                alpha=alpha,
                beta=beta,
                value=alpha * u(0.0) - 0.7 * beta,
                boundary="x_min",
            ),
            BCSegment(
                name="r",
                bc_type=BCType.ROBIN,
                alpha=alpha,
                beta=beta,
                value=alpha * u(1.0) + 0.7 * beta,
                boundary="x_max",
            ),
        ],
        dimension=1,
        domain_bounds=np.array([[0.0, 1.0]]),
    )
    coarse, _ = _wall_and_interior_errors(41, bc, u, lap_true)
    wall, interior = _wall_and_interior_errors(81, bc, u, lap_true)
    assert np.log2(coarse / wall) > 1.9, f"alpha={alpha}, beta={beta}: wall order {np.log2(coarse / wall):.2f}"
    assert wall < 1.1 * interior, f"alpha={alpha}, beta={beta}: wall {wall:.3e} against interior {interior:.3e}"


def test_edges_and_corners_in_two_dimensions():
    """A corner has two walls, so the cell mirror left it at half the true Laplacian (0.50) and an edge at
    three quarters (0.75), at every h. u = cos(pi x) cos(pi y) has zero normal derivative on all four walls."""
    n = 41
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[n, n], boundary_conditions=no_flux_bc(dimension=2))
    x, y = np.meshgrid(*grid.coordinates, indexing="ij")
    u = np.cos(np.pi * x) * np.cos(np.pi * y)
    lap = np.asarray(grid.get_laplacian_operator() @ u.ravel()).reshape(n, n)
    e = np.abs(lap + 2 * np.pi**2 * u)
    interior = e[1:-1, 1:-1].max()
    edge = max(e[0, 1:-1].max(), e[-1, 1:-1].max(), e[1:-1, 0].max(), e[1:-1, -1].max())
    corner = max(e[0, 0], e[0, -1], e[-1, 0], e[-1, -1])
    assert edge < 1.1 * interior, f"edge {edge:.3e} against interior {interior:.3e}"
    assert corner < 1.1 * interior, f"corner {corner:.3e} against interior {interior:.3e}"


def test_the_documented_example_is_what_the_function_returns():
    """`pad_array_with_ghosts`'s docstring claimed the node mirror while the code returned the cell one;
    nothing executed the example. It is the public statement of the convention, so it runs here."""
    from mfgarchon.geometry.boundary import applicator_fdm

    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in doctest.DocTestFinder().find(applicator_fdm.pad_array_with_ghosts, "pad_array_with_ghosts"):
        runner.run(test)
    result = runner.summarize(verbose=False)
    assert result.attempted > 0, "the example was not found"
    assert result.failed == 0, "the documented example is not what pad_array_with_ghosts returns"


def test_a_robin_wall_whose_beta_vanishes_on_part_of_the_face():
    """Field coefficients, as an FP wall carries them: per point on the face. Where beta = 0 the condition is
    Dirichlet, u = g/alpha, and takes the odd reflection 2 g/alpha - u_1 through the node beside the wall, as
    the Dirichlet branch does (#1968); elsewhere the node form. The third point has alpha/2 + beta/dx = 0,
    where the cell form was singular; the node form determines it."""
    from mfgarchon.geometry.boundary.applicator_fdm import _write_wall_ghosts

    rng = np.random.default_rng(1935)
    buf = rng.normal(size=(6, 3))  # one ghost layer each side of 4 nodes along axis 0, 3 points along the face
    alpha, beta, value, dx = np.array([2.0, 1.0, 2.0]), np.array([0.0, 1.0, -0.1]), np.array([3.0, 0.5, -1.0]), 0.1
    wall, mirror = buf[1].copy(), buf[2].copy()
    _write_wall_ghosts(buf, 0, "min", 1, dx, value, alpha, beta)
    expected = mirror + 2 * dx * (value - alpha * wall) / np.where(beta == 0, 1.0, beta)
    expected[0] = 2 * value[0] / alpha[0] - mirror[0]
    np.testing.assert_allclose(buf[0], expected, rtol=0, atol=1e-12)


def test_an_axis_too_short_for_the_mirror_raises():
    """The mirror of ghost layer k is node k+1, so g layers need g+1 nodes. With fewer, the mirror index lands
    in the opposite ghost layer and the ghost is silently stale; the extrapolation branch refuses the same way."""
    from mfgarchon.geometry.boundary import pad_array_with_ghosts

    with pytest.raises(ValueError, match="needs 2 nodes"):
        pad_array_with_ghosts(np.array([1.0]), no_flux_bc(dimension=1), ghost_depth=1, spacing=0.1)
    with pytest.raises(ValueError, match="needs 4 nodes"):
        pad_array_with_ghosts(np.array([1.0, 2.0, 3.0]), no_flux_bc(dimension=1), ghost_depth=3, spacing=0.1)


@pytest.mark.parametrize("path", ["uniform", "per-face", "robin-beta0"])
def test_the_dirichlet_ghost_converges_to_the_node_oracle_at_second_order(path):
    """#1968: on this node-centred grid the Dirichlet ghost is the odd reflection about the wall node, so it
    approximates u(-h) to O(h^2) at both walls. The cell relation it replaced, 2 g - u_wall, is O(h): it read
    0.118 / 0.061 / 0.031 at 9 / 17 / 33 points on u = exp(x).

    The uniform condition carries one value, so its field vanishes at both walls, u = sin(pi x) exp(x); the
    per-face ones take u = exp(x) with each wall's own value. Robin with beta = 0 is the same condition."""
    from mfgarchon.geometry.boundary import dirichlet_bc, pad_array_with_ghosts

    def exact(x):
        return np.sin(np.pi * x) * np.exp(x) if path == "uniform" else np.exp(x)

    if path == "uniform":
        conditions = dirichlet_bc(value=0.0, dimension=1)
    else:
        kind, coeffs = (BCType.DIRICHLET, {}) if path == "per-face" else (BCType.ROBIN, {"alpha": 1.0, "beta": 0.0})
        conditions = BoundaryConditions(
            segments=[
                BCSegment(name="lo", bc_type=kind, value=1.0, boundary="x_min", **coeffs),
                BCSegment(name="hi", bc_type=kind, value=float(np.e), boundary="x_max", **coeffs),
            ],
            dimension=1,
        )
    errors = []
    for n in (17, 33, 65):
        h = 1.0 / (n - 1)
        x = np.linspace(0.0, 1.0, n)
        padded = pad_array_with_ghosts(exact(x), conditions, ghost_depth=1, spacing=h)
        errors.append(max(abs(padded[0] - exact(-h)), abs(padded[-1] - exact(1.0 + h))))
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(orders > 1.8), f"observed orders {orders}, errors {errors}"

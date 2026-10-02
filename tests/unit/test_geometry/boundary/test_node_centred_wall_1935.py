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

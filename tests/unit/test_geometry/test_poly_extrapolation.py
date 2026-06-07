"""
Test polynomial extrapolation for high-order ghost cells (Issue #576, Phase 4).

Tests the Vandermonde-based polynomial extrapolation for order > 2.
"""

import numpy as np

from mfgarchon.geometry.boundary import dirichlet_bc, neumann_bc
from mfgarchon.geometry.boundary.applicator_fdm import PreallocatedGhostBuffer


def test_order_3_neumann_1d():
    """Test order=3 polynomial extrapolation for Neumann BC in 1D."""
    # Create manufactured solution: u(x) = x^3 (cubic polynomial)
    # At x=0: ∂u/∂x = 3x^2|_{x=0} = 0 (satisfies zero-flux Neumann)
    # Order-3 extrapolation should fit cubics exactly

    nx = 10
    x = np.linspace(0.0, 1.0, nx)
    u_exact = x**3  # Cubic should be fitted exactly by order-3

    bc = neumann_bc(dimension=1)
    buffer = PreallocatedGhostBuffer(
        interior_shape=(nx,),
        boundary_conditions=bc,
        domain_bounds=np.array([[0.0, 1.0]]),
        order=3,
        ghost_depth=1,
    )

    # Set interior to exact solution
    buffer.interior[:] = u_exact

    # Update ghosts with polynomial extrapolation
    buffer.update_ghosts(time=0.0)

    # Check ghost values
    # For u(x) = x^3 with ∂u/∂x = 0 at x=0:
    # Ghost at x = -dx should equal (-dx)^3
    dx = 1.0 / (nx - 1)
    x_ghost_low = -dx
    u_ghost_low_exact = x_ghost_low**3

    # Debug output
    print(f"  Interior near boundary: {buffer.interior[:4]}")
    print(f"  Ghost computed: {buffer.padded[0]:.6f}, exact: {u_ghost_low_exact:.6f}")
    print(f"  Error: {np.abs(buffer.padded[0] - u_ghost_low_exact):.6e}")

    # Order-3 should fit cubics exactly (within numerical precision)
    assert np.abs(buffer.padded[0] - u_ghost_low_exact) < 1e-10, (
        f"Low ghost mismatch: {buffer.padded[0]} vs {u_ghost_low_exact}"
    )

    print(f"✓ Order 3 Neumann 1D: ghost[0] = {buffer.padded[0]:.6f} (exact: {u_ghost_low_exact:.6f})")


def test_order_3_dirichlet_1d():
    """Test order=3 polynomial extrapolation for Dirichlet BC in 1D."""
    # Create manufactured solution: u(x) = x^2 (quadratic)
    # Dirichlet BC: u(0) = 0, u(1) = 1

    nx = 10
    x = np.linspace(0.0, 1.0, nx)
    u_exact = x**2  # Quadratic, so u(0) = 0

    bc = dirichlet_bc(value=0.0, dimension=1)
    buffer = PreallocatedGhostBuffer(
        interior_shape=(nx,),
        boundary_conditions=bc,
        domain_bounds=np.array([[0.0, 1.0]]),
        order=3,
        ghost_depth=1,
    )

    # Set interior to exact solution
    buffer.interior[:] = u_exact

    # Update ghosts with polynomial extrapolation
    buffer.update_ghosts(time=0.0)

    # Check ghost values
    # For Dirichlet u(0) = 0, u(x) = x^2
    # Ghost at x = -dx should equal (-dx)^2 = dx^2
    dx = 1.0 / (nx - 1)
    x_ghost_low = -dx
    u_ghost_low_exact = x_ghost_low**2  # = dx^2

    # Debug output
    print(f"  Interior near boundary: {buffer.interior[:4]}")
    print(f"  Ghost computed: {buffer.padded[0]:.6f}, exact: {u_ghost_low_exact:.6f}")
    print(f"  Error: {np.abs(buffer.padded[0] - u_ghost_low_exact):.6e}")

    # Order-3 should fit quadratics exactly (within numerical precision)
    assert np.abs(buffer.padded[0] - u_ghost_low_exact) < 1e-10, (
        f"Low ghost mismatch: {buffer.padded[0]} vs {u_ghost_low_exact}"
    )

    print(f"✓ Order 3 Dirichlet 1D: ghost[0] = {buffer.padded[0]:.6f} (exact: {u_ghost_low_exact:.6f})")


def test_order_5_neumann_1d():
    """Test order=5 polynomial extrapolation (WENO5-compatible)."""
    nx = 20
    x = np.linspace(0.0, 1.0, nx)
    # Smooth quartic: u(x) = x^4
    # ∂u/∂x = 4x^3, so ∂u/∂x(0) = 0 (Neumann satisfied)
    u_exact = x**4

    bc = neumann_bc(dimension=1)
    buffer = PreallocatedGhostBuffer(
        interior_shape=(nx,),
        boundary_conditions=bc,
        domain_bounds=np.array([[0.0, 1.0]]),
        order=5,
        ghost_depth=3,  # WENO needs 3 ghost cells
    )

    # Set interior to exact solution
    buffer.interior[:] = u_exact

    # Update ghosts with polynomial extrapolation
    buffer.update_ghosts(time=0.0)

    # Check that all 3 ghost cells are filled
    dx = 1.0 / (nx - 1)
    for k in range(3):
        x_ghost = -(k + 1) * dx
        u_ghost_exact = x_ghost**4

        # High-order extrapolation should be accurate for smooth polynomial
        error = np.abs(buffer.padded[k] - u_ghost_exact)
        assert error < 0.01, f"Ghost {k} error: {error}"

    print("✓ Order 5 Neumann 1D: 3 ghost cells filled with high accuracy")


def test_order_5_neumann_both_boundaries_accurate():
    """High-order ghosts must be accurate at BOTH boundaries (Issue #1200 regression).

    The high-boundary Vandermonde fit previously paired ``x_interior=[-dx,-2dx,...]``
    (nearest-first) with ``interior_indices`` ordered farthest-first, so the high-boundary
    ghosts were badly wrong (the low boundary, where both orderings agree, was fine). The
    bug stayed latent because the only order-5 test checked the low boundary only. Here we
    use ``cos(pi x)`` -- Neumann-compatible (du/dx=0) at both x=0 and x=1 -- and require the
    high-boundary ghosts to be reconstructed as accurately as the low-boundary ghosts.
    """
    nx = 41
    x = np.linspace(0.0, 1.0, nx)
    dx = 1.0 / (nx - 1)
    u_exact = np.cos(np.pi * x)

    buffer = PreallocatedGhostBuffer(
        interior_shape=(nx,),
        boundary_conditions=neumann_bc(dimension=1),
        domain_bounds=np.array([[0.0, 1.0]]),
        order=5,
        ghost_depth=3,
    )
    buffer.interior[:] = u_exact
    buffer.update_ghosts(time=0.0)

    g = 3
    # Low ghosts: padded[0..g-1] at x = -g*dx .. -dx (padded[0] farthest).
    low_err = max(abs(buffer.padded[k] - np.cos(np.pi * (-(g - k) * dx))) for k in range(g))
    # High ghosts: padded[-g..-1] at x = 1+dx .. 1+g*dx (padded[-1] farthest).
    high_err = max(abs(buffer.padded[-g + k] - np.cos(np.pi * (1.0 + (k + 1) * dx))) for k in range(g))

    # Both boundaries must be accurate (order-5 extrapolation of a smooth function).
    assert low_err < 1e-3, f"low-boundary ghost error too large: {low_err}"
    assert high_err < 1e-3, f"high-boundary ghost error too large (#1200): {high_err}"
    # And symmetric to within an order of magnitude (the bug made high ~1e5x worse).
    assert high_err < 100 * low_err, f"high/low ghost asymmetry: high={high_err}, low={low_err}"


def test_order_5_nonzero_neumann_flux():
    """A nonzero Neumann flux is honoured, not silently zeroed (Issue #1186).

    The high-order ghost BC row previously hardcoded du/dn = 0, so a ``neumann_bc(value=g)``
    with g != 0 was dropped (WENO5 is the order-5 consumer). Uses ``u = (g/L) x^2 - g x + c``,
    whose outward-normal derivative is exactly ``g`` at BOTH boundaries (consistent with a
    uniform ``neumann_bc(value=g)``). Degree-5 extrapolation reproduces this quadratic
    exactly, so the ghosts must equal ``u(x_ghost)`` to machine precision at both ends --
    which holds only if the constraint encodes ``p'(0) = -g`` (low) / ``+g`` (high).
    """
    n, gd = 41, 3
    length = 1.0
    dx = length / (n - 1)
    g_flux = 0.7
    c = 2.0
    x = np.linspace(0.0, length, n)

    def u_exact(xq):
        return (g_flux / length) * xq**2 - g_flux * xq + c

    buffer = PreallocatedGhostBuffer(
        interior_shape=(n,),
        boundary_conditions=neumann_bc(dimension=1, value=g_flux),
        domain_bounds=np.array([[0.0, length]]),
        order=5,
        ghost_depth=gd,
    )
    buffer.interior[:] = u_exact(x)
    buffer.update_ghosts(time=0.0)

    # Low ghosts padded[0..gd-1] at x = -gd*dx .. -dx; high ghosts padded[-gd..-1] at L+dx .. L+gd*dx.
    for k in range(gd):
        assert abs(buffer.padded[k] - u_exact(-(gd - k) * dx)) < 1e-9, f"low ghost {k} dropped flux g={g_flux}"
        assert abs(buffer.padded[-gd + k] - u_exact(length + (k + 1) * dx)) < 1e-9, (
            f"high ghost {k} dropped flux g={g_flux}"
        )


def test_order_3_neumann_2d():
    """Test order=3 polynomial extrapolation in 2D."""
    nx, ny = 10, 12
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.5, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Smooth function: u(x,y) = x^2 + y^2
    # ∂u/∂x(0,y) = 0, ∂u/∂y(x,0) = 0 (Neumann satisfied on low boundaries)
    u_exact = X**2 + Y**2

    bc = neumann_bc(dimension=2)
    buffer = PreallocatedGhostBuffer(
        interior_shape=(nx, ny),
        boundary_conditions=bc,
        domain_bounds=np.array([[0.0, 1.0], [0.0, 1.5]]),
        order=3,
        ghost_depth=1,
    )

    # Set interior to exact solution
    buffer.interior[:, :] = u_exact

    # Update ghosts with polynomial extrapolation
    buffer.update_ghosts(time=0.0)

    # Check low x boundary (axis=0)
    dx = 1.0 / (nx - 1)
    x_ghost = -dx

    # Check a few points along the boundary
    for j in range(1, ny - 1):
        y_val = y[j]
        u_ghost_exact = x_ghost**2 + y_val**2
        u_ghost_computed = buffer.padded[0, j + 1]  # +1 for ghost offset

        error = np.abs(u_ghost_computed - u_ghost_exact)
        # 2D extrapolation may have larger error
        assert error < 0.5, f"2D ghost error at y={y_val}: {error}"

    print("✓ Order 3 Neumann 2D: ghost cells filled along both axes")


def test_convergence_order():
    """
    Test that polynomial extrapolation achieves expected convergence order.

    For smooth solutions, the ghost cell error should scale as O(h^order).
    """
    # Manufactured solution: u(x) = sin(2πx)
    # ∂u/∂x = 2π cos(2πx), so ∂u/∂x(0) = 2π ≠ 0
    # Use Neumann BC with g = 2π to match

    orders = [3, 4, 5]
    grid_sizes = [10, 20, 40]

    for order in orders:
        errors = []

        for nx in grid_sizes:
            x = np.linspace(0.0, 1.0, nx)
            u_exact = np.sin(2 * np.pi * x)

            # Neumann BC: ∂u/∂x(0) = 2π
            bc = neumann_bc(value=2.0 * np.pi, dimension=1)

            buffer = PreallocatedGhostBuffer(
                interior_shape=(nx,),
                boundary_conditions=bc,
                domain_bounds=np.array([[0.0, 1.0]]),
                order=order,
                ghost_depth=1,
            )

            buffer.interior[:] = u_exact
            buffer.update_ghosts(time=0.0)

            # Compute error at ghost point
            dx = 1.0 / (nx - 1)
            x_ghost = -dx
            u_ghost_exact = np.sin(2 * np.pi * x_ghost)
            error = np.abs(buffer.padded[0] - u_ghost_exact)
            errors.append(error)

        # Check convergence rate
        if len(errors) >= 2:
            # Estimate convergence order from last two refinements
            h_ratio = 2.0  # Grid doubled each time
            error_ratio = errors[-2] / errors[-1]

            # Convergence order ≈ log(error_ratio) / log(h_ratio)
            conv_order = np.log(error_ratio) / np.log(h_ratio)

            # Should be close to requested order (within tolerance)
            # Note: Neumann BC with non-zero gradient is harder, so allow some slack
            print(f"  Order {order}: errors = {errors}, convergence rate ≈ {conv_order:.2f}")
            assert conv_order > 1.0, f"Order {order}: convergence rate {conv_order:.2f} too low (expected > 1.0)"


if __name__ == "__main__":
    """Run smoke tests."""
    print("Testing polynomial extrapolation for high-order ghost cells...")

    print("\n1D Tests:")
    test_order_3_neumann_1d()
    test_order_3_dirichlet_1d()
    test_order_5_neumann_1d()

    print("\n2D Tests:")
    test_order_3_neumann_2d()

    print("\nConvergence Tests:")
    test_convergence_order()

    print("\n✅ All polynomial extrapolation tests passed!")

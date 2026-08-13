"""
Unit tests for finite difference stencil functions.

Tests the low-level stencil building blocks that differential operators
use internally. These are pure array operations using np.roll (periodic wrapping).

Created: 2026-02-10 (Issue #768 - Test coverage for operators/)
"""

import pytest

import numpy as np

from mfgarchon.geometry.boundary import neumann_bc
from mfgarchon.operators.stencils.finite_difference import (
    fix_boundaries_one_sided,
    get_gradient_stencil_coefficients,
    get_laplacian_stencil_coefficients,
    gradient_backward,
    gradient_central,
    gradient_forward,
    gradient_nd,
    gradient_upwind,
    laplacian_stencil_1d,
    laplacian_stencil_nd,
    laplacian_with_bc,
)

# =============================================================================
# First-Order Gradient Stencils
# =============================================================================


class TestGradientCentral:
    """Tests for central difference gradient."""

    @pytest.mark.unit
    def test_linear_exact(self):
        """Central difference of linear function should be exact (periodic interior)."""
        n = 100
        x = np.linspace(0, 1, n, endpoint=False)
        h = x[1] - x[0]
        u = 3.0 * x + 1.0

        du = gradient_central(u, axis=0, h=h)
        # For periodic domain, linear on [0,1) has wrap-around artifact at boundaries
        # but interior should be exact
        np.testing.assert_allclose(du[2:-2], 3.0, atol=1e-10)

    @pytest.mark.unit
    def test_quadratic_exact(self):
        """Central difference of x^2 should give 2x (exact for polynomials up to degree 2)."""
        n = 100
        x = np.linspace(0, 1, n, endpoint=False)
        h = x[1] - x[0]
        u = x**2

        du = gradient_central(u, axis=0, h=h)
        expected = 2.0 * x
        np.testing.assert_allclose(du[2:-2], expected[2:-2], atol=1e-10)

    @pytest.mark.unit
    def test_2d_axis0(self):
        """Should differentiate along axis 0 in 2D."""
        nx, ny = 30, 20
        x = np.linspace(0, 1, nx, endpoint=False)
        y = np.linspace(0, 1, ny, endpoint=False)
        dx = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing="ij")

        u = X**2 + Y  # du/dx = 2x
        du_dx = gradient_central(u, axis=0, h=dx)

        expected = 2.0 * X
        np.testing.assert_allclose(du_dx[2:-2, :], expected[2:-2, :], atol=1e-10)


class TestGradientForward:
    """Tests for forward difference gradient."""

    @pytest.mark.unit
    def test_linear_exact(self):
        """Forward difference of linear function should be exact."""
        n = 50
        x = np.linspace(0, 1, n, endpoint=False)
        h = x[1] - x[0]
        u = 5.0 * x + 2.0

        du = gradient_forward(u, axis=0, h=h)
        np.testing.assert_allclose(du[:-2], 5.0, atol=1e-10)

    @pytest.mark.unit
    def test_first_order_accuracy(self):
        """Forward diff of sin(x) should have O(h) error."""
        n = 200
        x = np.linspace(0, 2 * np.pi, n, endpoint=False)
        h = x[1] - x[0]
        u = np.sin(x)

        du = gradient_forward(u, axis=0, h=h)
        expected = np.cos(x)

        error = np.max(np.abs(du[5:-5] - expected[5:-5]))
        # O(h) ~ 0.03
        assert error < 0.1


class TestGradientBackward:
    """Tests for backward difference gradient."""

    @pytest.mark.unit
    def test_linear_exact(self):
        """Backward difference of linear function should be exact."""
        n = 50
        x = np.linspace(0, 1, n, endpoint=False)
        h = x[1] - x[0]
        u = 5.0 * x + 2.0

        du = gradient_backward(u, axis=0, h=h)
        np.testing.assert_allclose(du[2:], 5.0, atol=1e-10)


class TestGradientUpwind:
    """Tests for Godunov upwind gradient."""

    @pytest.mark.unit
    def test_monotone_increasing(self):
        """Accuracy on monotone increasing u: gradient_upwind is within 0.02 of 2x.

        This does NOT check which branch was selected, despite what its name suggests. Forward
        and backward differences are equal in magnitude on ``u = x**2``, so inverting the
        selection leaves this green at any tolerance. The selection is pinned by identity in
        ``test_selection_rule_is_the_godunov_one_not_merely_accurate`` below.
        """
        n = 100
        x = np.linspace(0, 1, n, endpoint=False)
        h = x[1] - x[0]
        u = x**2  # Increasing on [0, 1]

        du = gradient_upwind(u, axis=0, h=h)
        expected = 2.0 * x

        # Upwind is O(h) accurate
        error = np.max(np.abs(du[5:-5] - expected[5:-5]))
        assert error < 0.1

    @pytest.mark.unit
    def test_selection_rule_is_the_godunov_one_not_merely_accurate(self):
        """Which stencil is selected, asserted as an identity rather than a tolerance.

        `test_monotone_increasing` states the rule in its docstring and cannot see it. Forward and
        backward differences are both O(h) on smooth data, and on `u = x**2` their errors are
        *equal in magnitude* -- 0.0100 each at n=100, symmetric about the exact value -- so no
        tolerance separates them at any n. Inverting the selection in `gradient_upwind` leaves
        that test green.

        What separates them is the value: at x = 0.5, backward = 0.990000 and forward = 1.010000,
        differing by exactly h*u'' = 0.02. So compare against the stencil functions directly.

        Both branches are covered. What was missing is a test that can SEE which one was taken:
        the forward branch executes on every run of `test_result_shape` (its `np.random.randn(50)`
        sends a mean of 25 nodes of 50 down it, and zero of 20000 replications sent none), but that
        test asserts only the output shape, so inverting the selection leaves it green. Line
        coverage of the branch was never the gap; a discriminating assertion was.

        Interior only: these stencils use ``np.roll``, so at node 0 the backward difference wraps
        to node n-1 and reads a spurious -98.01 on this fixture. The wraparound is the caller's
        problem (``fix_boundaries_one_sided`` exists for it) and not what the selection rule says.
        """
        n = 100
        x = np.linspace(0, 1, n, endpoint=False)
        h = x[1] - x[0]

        rising = x**2  # du/dx >= 0 -> information flows right -> read from the LEFT
        interior = slice(1, -1)
        np.testing.assert_array_equal(
            gradient_upwind(rising, axis=0, h=h)[interior],
            gradient_backward(rising, axis=0, h=h)[interior],
            err_msg="increasing u must select the backward (upwind) stencil",
        )

        falling = -(x**2)  # du/dx < 0 -> information flows left -> read from the RIGHT
        np.testing.assert_array_equal(
            gradient_upwind(falling, axis=0, h=h)[interior],
            gradient_forward(falling, axis=0, h=h)[interior],
            err_msg="decreasing u must select the forward (upwind) stencil",
        )

        # The two candidates must differ ON THE SLICE THE IDENTITIES USE, or those identities are
        # vacuous. Comparing full arrays is a different check: node 0's backward difference wraps,
        # so `not allclose(full)` is satisfied by the wraparound alone. On a linear `rising` both
        # identities pass vacuously (interior forward == backward) while a full-array guard still
        # reports "they differ" -- the guard certifying its own blind spot.
        for label, fixture in (("rising", rising), ("falling", falling)):
            assert not np.allclose(
                gradient_backward(fixture, axis=0, h=h)[interior],
                gradient_forward(fixture, axis=0, h=h)[interior],
            ), f"{label} is too smooth to distinguish the stencils -- the identities prove nothing"

    @pytest.mark.unit
    def test_the_selection_predicate_is_pinned_not_only_the_branch_bodies(self):
        """The tie-break at ``grad_central == 0``, which strictly monotone fixtures never reach.

        The test above uses monotone fixtures, so `sign(grad_forward) == sign(grad_backward) ==
        sign(grad_central)` at every interior node and `grad_central` is never zero. That pins
        which stencil each branch returns, but not the predicate choosing between them -- and the
        predicate is what makes the scheme Godunov. Three non-equivalent mutations survive without
        this: `>=` weakened to `>`, and the predicate reading `grad_forward` or `grad_backward`
        instead of `grad_central`.

        A quadratic with an interior extremum puts a node exactly on the tie, `grad_central == 0.0`
        to the bit. The documented rule sends it to the backward stencil (`>= 0`).

        **Both** signs are needed. At a maximum `grad_backward` is positive, so a mutant predicate
        reading `grad_backward` agrees with the correct one by coincidence and survives; at a
        minimum it is negative and they part. One extremum is not a probe of the tie, it is a probe
        of one side of it.
        """
        n = 100
        x = np.linspace(0, 1, n, endpoint=False)
        h = x[1] - x[0]

        for label, u in (("maximum", -((x - 0.5) ** 2)), ("minimum", (x - 0.5) ** 2)):
            tie = int(np.argmin(np.abs(x - 0.5)))
            bwd = gradient_backward(u, axis=0, h=h)
            fwd = gradient_forward(u, axis=0, h=h)
            central = (fwd + bwd) / 2.0
            assert central[tie] == 0.0, (
                f"{label}: the fixture must land a node exactly on the tie, got {central[tie]!r}"
            )
            assert bwd[tie] != fwd[tie], f"{label}: the branches must disagree at the tie"

            got = gradient_upwind(u, axis=0, h=h)[tie]
            assert got == bwd[tie], f"{label}: grad_central == 0 must take the backward branch, as the >= says"

    @pytest.mark.unit
    def test_result_shape(self):
        """Output shape should match input."""
        u = np.random.randn(50)
        du = gradient_upwind(u, axis=0, h=0.1)
        assert du.shape == (50,)


# =============================================================================
# Boundary Handling
# =============================================================================


class TestFixBoundariesOneSided:
    """Tests for boundary correction function."""

    @pytest.mark.unit
    def test_1d_boundary_correction(self):
        """Boundaries should use 2nd-order one-sided differences (Issue #1084)."""
        n = 50
        x = np.linspace(0, 1, n)
        h = x[1] - x[0]
        u = x**2

        # Start with central diff (wraps at boundaries)
        grad = gradient_central(u, axis=0, h=h)
        # Fix boundaries
        grad_fixed = fix_boundaries_one_sided(grad.copy(), u, axis=0, h=h)

        # Left boundary: 2nd-order forward (-3u0 + 4u1 - u2)/(2h)
        expected_left = (-3.0 * u[0] + 4.0 * u[1] - u[2]) / (2.0 * h)
        assert abs(grad_fixed[0] - expected_left) < 1e-14
        # x^2 is a quadratic, so the 2nd-order stencil is exact: f'(0) = 0
        assert abs(grad_fixed[0] - 0.0) < 1e-12

        # Right boundary: 2nd-order backward (3u_{-1} - 4u_{-2} + u_{-3})/(2h)
        expected_right = (3.0 * u[-1] - 4.0 * u[-2] + u[-3]) / (2.0 * h)
        assert abs(grad_fixed[-1] - expected_right) < 1e-14
        # exact derivative f'(1) = 2
        assert abs(grad_fixed[-1] - 2.0) < 1e-12

    @pytest.mark.unit
    def test_2d_boundary_correction(self):
        """Should correct boundaries along specified axis in 2D (2nd-order)."""
        nx, ny = 20, 20
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        dx = x[1] - x[0]
        X, _Y = np.meshgrid(x, y, indexing="ij")
        u = X**2

        grad = gradient_central(u, axis=0, h=dx)
        grad_fixed = fix_boundaries_one_sided(grad.copy(), u, axis=0, h=dx)

        # Left boundary (X=0): 2nd-order forward, exact for quadratic (f'(0) = 0)
        expected_left = (-3.0 * u[0, :] + 4.0 * u[1, :] - u[2, :]) / (2.0 * dx)
        np.testing.assert_allclose(grad_fixed[0, :], expected_left, atol=1e-14)
        np.testing.assert_allclose(grad_fixed[0, :], 0.0, atol=1e-12)

    @pytest.mark.unit
    def test_boundary_eoc_second_order(self):
        """One-sided boundary correction converges at O(h^2) (Issue #1084).

        Uses a non-polynomial smooth field so the truncation error is nonzero
        and the empirical order of convergence is observable.
        """

        def f(x):
            return np.exp(np.sin(3.0 * x))

        def fprime(x):
            return 3.0 * np.cos(3.0 * x) * np.exp(np.sin(3.0 * x))

        hs = []
        errs = []
        for n in (40, 80, 160, 320):
            x = np.linspace(0.0, 1.0, n)
            h = x[1] - x[0]
            u = f(x)
            grad = fix_boundaries_one_sided(gradient_central(u, axis=0, h=h), u, axis=0, h=h)
            err = max(abs(grad[0] - fprime(x[0])), abs(grad[-1] - fprime(x[-1])))
            hs.append(h)
            errs.append(err)

        slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
        assert slope > 1.8, f"boundary EOC {slope:.2f} is not second order"

    @pytest.mark.unit
    def test_boundary_two_point_falls_back_to_first_order(self):
        """With only 2 points the 3-point stencil is unavailable; use 1st-order."""
        u = np.array([1.0, 3.0])
        h = 0.5
        grad = fix_boundaries_one_sided(gradient_central(u, axis=0, h=h), u, axis=0, h=h)
        # both endpoints use (u[1]-u[0])/h = 4.0
        assert abs(grad[0] - 4.0) < 1e-14
        assert abs(grad[-1] - 4.0) < 1e-14


# =============================================================================
# Second-Order Stencils
# =============================================================================


class TestLaplacianStencil1D:
    """Tests for 1D Laplacian stencil."""

    @pytest.mark.unit
    def test_quadratic_exact(self):
        """3-point stencil of x^2 should give exactly 2 (interior)."""
        n = 50
        x = np.linspace(0, 1, n, endpoint=False)
        h = x[1] - x[0]
        u = x**2

        Lu = laplacian_stencil_1d(u, h=h)
        # Interior points (periodic wrapping affects boundaries)
        np.testing.assert_allclose(Lu[2:-2], 2.0, atol=1e-10)

    @pytest.mark.unit
    def test_constant_zero(self):
        """Laplacian of constant should be 0."""
        u = np.ones(50) * 7.0
        Lu = laplacian_stencil_1d(u, h=0.1)
        np.testing.assert_allclose(Lu, 0.0, atol=1e-12)

    @pytest.mark.unit
    def test_linear_zero(self):
        """Laplacian of linear function should be 0 (interior)."""
        n = 50
        x = np.linspace(0, 1, n, endpoint=False)
        h = x[1] - x[0]
        u = 3.0 * x + 1.0

        Lu = laplacian_stencil_1d(u, h=h)
        np.testing.assert_allclose(Lu[2:-2], 0.0, atol=1e-10)


class TestLaplacianStencilND:
    """Tests for n-dimensional Laplacian stencil."""

    @pytest.mark.unit
    def test_2d_quadratic_exact(self):
        """Laplacian of x^2 + y^2 should be 4 (interior)."""
        nx, ny = 30, 30
        x = np.linspace(0, 1, nx, endpoint=False)
        y = np.linspace(0, 1, ny, endpoint=False)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="ij")

        u = X**2 + Y**2
        Lu = laplacian_stencil_nd(u, spacings=[dx, dy])

        np.testing.assert_allclose(Lu[2:-2, 2:-2], 4.0, atol=1e-10)

    @pytest.mark.unit
    def test_1d_matches_stencil_1d(self):
        """ND stencil with 1D input should match 1D stencil."""
        n = 50
        x = np.linspace(0, 1, n, endpoint=False)
        h = x[1] - x[0]
        u = x**2

        Lu_1d = laplacian_stencil_1d(u, h=h)
        Lu_nd = laplacian_stencil_nd(u, spacings=[h])

        np.testing.assert_allclose(Lu_nd, Lu_1d, atol=1e-14)


# =============================================================================
# Stencil Coefficients
# =============================================================================


class TestStencilCoefficients:
    """Tests for stencil coefficient extraction."""

    @pytest.mark.unit
    def test_central_gradient_coefficients(self):
        """Central gradient: [-1/(2h), 1/(2h)] at offsets [-1, 1]."""
        h = 0.1
        offsets, coeffs = get_gradient_stencil_coefficients("central", h)

        assert offsets == [-1, 1]
        np.testing.assert_allclose(coeffs, [-1.0 / (2 * h), 1.0 / (2 * h)])

    @pytest.mark.unit
    def test_forward_gradient_coefficients(self):
        """Forward gradient: [-1/h, 1/h] at offsets [0, 1]."""
        h = 0.2
        offsets, coeffs = get_gradient_stencil_coefficients("forward", h)

        assert offsets == [0, 1]
        np.testing.assert_allclose(coeffs, [-1.0 / h, 1.0 / h])

    @pytest.mark.unit
    def test_backward_gradient_coefficients(self):
        """Backward gradient: [-1/h, 1/h] at offsets [-1, 0]."""
        h = 0.05
        offsets, coeffs = get_gradient_stencil_coefficients("backward", h)

        assert offsets == [-1, 0]
        np.testing.assert_allclose(coeffs, [-1.0 / h, 1.0 / h])

    @pytest.mark.unit
    def test_laplacian_coefficients(self):
        """Laplacian: [1/h^2, -2/h^2, 1/h^2] at offsets [-1, 0, 1]."""
        h = 0.1
        offsets, coeffs = get_laplacian_stencil_coefficients(h)

        assert offsets == [-1, 0, 1]
        h2 = h * h
        np.testing.assert_allclose(coeffs, [1.0 / h2, -2.0 / h2, 1.0 / h2])

    @pytest.mark.unit
    def test_unknown_scheme_raises(self):
        """Should raise ValueError for unknown scheme."""
        with pytest.raises(ValueError, match="Unknown scheme"):
            get_gradient_stencil_coefficients("weno5", 0.1)


# =============================================================================
# Composite Functions
# =============================================================================


class TestGradientND:
    """Tests for gradient_nd helper."""

    @pytest.mark.unit
    def test_2d_linear(self):
        """Gradient of u=3x+2y should be [3, 2] (interior)."""
        nx, ny = 30, 30
        x = np.linspace(0, 1, nx, endpoint=False)
        y = np.linspace(0, 1, ny, endpoint=False)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="ij")

        u = 3.0 * X + 2.0 * Y
        grad = gradient_nd(u, spacings=[dx, dy])

        assert len(grad) == 2
        np.testing.assert_allclose(grad[0][2:-2, 2:-2], 3.0, atol=1e-10)
        np.testing.assert_allclose(grad[1][2:-2, 2:-2], 2.0, atol=1e-10)

    @pytest.mark.unit
    def test_zero_spacing_returns_zero(self):
        """Near-zero spacing should return zero gradient for that axis."""
        u = np.random.randn(20, 20)
        grad = gradient_nd(u, spacings=[0.1, 1e-16])

        # Second component should be all zeros
        np.testing.assert_allclose(grad[1], 0.0, atol=1e-14)


class TestLaplacianWithBC:
    """Tests for laplacian_with_bc composite function."""

    @pytest.mark.unit
    def test_neumann_quadratic(self):
        """With Neumann BC, Laplacian of x^2 should be 2 at interior points.

        Ghost cell for Neumann uses copy (u_ghost = u_boundary), which is
        1st-order at boundary. Interior is exact for quadratic.
        """
        n = 50
        x = np.linspace(0, 1, n)
        dx = x[1] - x[0]
        u = x**2

        bc = neumann_bc(dimension=1)
        Lu = laplacian_with_bc(u, spacings=[dx], bc=bc)

        assert Lu.shape == (n,)
        # Interior: exact for quadratic
        np.testing.assert_allclose(Lu[2:-2], 2.0, atol=1e-10)

    @pytest.mark.unit
    def test_no_bc_matches_stencil(self):
        """With bc=None, should match bare laplacian_stencil_nd."""
        n = 50
        x = np.linspace(0, 1, n, endpoint=False)
        h = x[1] - x[0]
        u = x**2

        Lu_bc = laplacian_with_bc(u, spacings=[h], bc=None)
        Lu_stencil = laplacian_stencil_nd(u, spacings=[h])

        np.testing.assert_allclose(Lu_bc, Lu_stencil, atol=1e-14)


class TestBackendRollEquivalence:
    """Issue #1194: roll-based stencils must work on the torch backend (torch.roll
    takes dims=, not numpy's axis=) and match the numpy result exactly. Pre-fix the
    torch path raised TypeError; the numpy path is unchanged (byte-identical)."""

    def test_stencils_torch_backend_match_numpy(self):
        torch = pytest.importorskip("torch")  # noqa: F841
        from mfgarchon.backends.torch_backend import TorchBackend
        from mfgarchon.utils.numerical.tensor_calculus import divergence, hessian, laplacian

        rng = np.random.RandomState(0)
        a = rng.rand(6, 5)
        sp = [0.2, 0.3]
        be = TorchBackend(device="cpu")
        t = be.array_module.tensor(a, dtype=be.array_module.float64)

        # tensor_calculus operators (backend=) -- pre-fix raised TypeError on roll(axis=)
        assert np.allclose(laplacian(t, sp, backend=be).numpy(), laplacian(a, sp))
        assert np.allclose(divergence([t, t * 2], sp, backend=be).numpy(), divergence([a, a * 2], sp))
        assert np.allclose(hessian(t, sp, backend=be).numpy(), hessian(a, sp))

        # stencil module (xp = torch module directly), the path the GPU particle solver hits
        g_t = gradient_central(t, axis=1, h=0.1, xp=be.array_module)
        assert np.allclose(g_t.numpy(), gradient_central(a, axis=1, h=0.1, xp=np))
        assert all(
            np.allclose(x.numpy(), y)
            for x, y in zip(gradient_nd(t, sp, xp=be.array_module), gradient_nd(a, sp, xp=np), strict=True)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

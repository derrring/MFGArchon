"""
Unit tests for tensor diffusion operators.

Tests the tensor diffusion via tensor_calculus.diffusion():
- Diagonal tensor = scalar equivalence
- Anisotropic 2D diffusion
- Cross-diffusion terms
- Boundary condition handling
- PSD validation

Note: Tests migrated from tensor_operators.py to use the unified
tensor_calculus.diffusion() API as of v0.17.0.
"""

from __future__ import annotations

import numpy as np

from mfgarchon.geometry.boundary import (
    dirichlet_bc,
    no_flux_bc,
    periodic_bc,
)
from mfgarchon.utils.numerical.tensor_calculus import diffusion


class TestDiagonalTensorEqualsScalar:
    """Test that diagonal tensor Σ = σ²I matches scalar diffusion."""

    def test_2d_isotropic_tensor_matches_scalar(self):
        """Isotropic tensor Σ = σ²I should match scalar Laplacian."""
        # Use a smooth polynomial instead of sinusoidal for better numerical accuracy
        Nx, Ny = 16, 16
        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Smooth test function: m(x,y) = x²(1-x) + y²(1-y)
        # Chosen to satisfy Dirichlet BCs naturally
        m = X**2 * (1 - X) + Y**2 * (1 - Y)

        dx = x[1] - x[0]
        dy = y[1] - y[0]
        sigma_squared = 0.1

        # Scalar diffusion: Σ = σ²I
        sigma_tensor = sigma_squared * np.eye(2)  # (2, 2) constant tensor

        bc = dirichlet_bc(dimension=2)

        # Compute tensor diffusion
        result_tensor = diffusion(m, sigma_tensor, [dx, dy], bc=bc)

        # Compute analytical Laplacian
        # Δm = ∂²m/∂x² + ∂²m/∂y²
        # For m = x²(1-x) + y²(1-y) = x² - x³ + y² - y³:
        # ∂m/∂x = 2x - 3x²,  ∂²m/∂x² = 2 - 6x
        # ∂m/∂y = 2y - 3y²,  ∂²m/∂y² = 2 - 6y
        # Δm = (2 - 6x) + (2 - 6y) = 4 - 6x - 6y
        laplacian_analytical = 4 - 6 * X - 6 * Y
        result_expected = sigma_squared * laplacian_analytical

        # Check interior points (away from boundaries where discretization error is larger)
        np.testing.assert_allclose(result_tensor[2:-2, 2:-2], result_expected[2:-2, 2:-2], rtol=0.01, atol=0.01)

    def test_diagonal_diffusion_matches_component_wise_laplacian(self):
        """Diagonal tensor should match sum of component-wise Laplacians."""
        Nx, Ny = 8, 8
        m = np.random.rand(Nx, Ny)
        dx, dy = 0.1, 0.1

        # Different diffusion per direction
        sigma_x = 0.2
        sigma_y = 0.05

        sigma_diag = np.array([sigma_x, sigma_y])
        bc = periodic_bc(dimension=2)

        # Convert diagonal to full tensor for unified API
        sigma_tensor = np.diag(sigma_diag)
        result_diag = diffusion(m, sigma_tensor, [dx, dy], bc=bc)

        assert result_diag.shape == m.shape
        assert np.all(np.isfinite(result_diag))

        # The comparison the test is named for. Under periodic BC the divergence-form operator
        # with a constant diagonal tensor is exactly the weighted sum of component-wise Laplacians.
        #
        # Note which weight lands on which axis: sigma_x = 0.2 is the tensor's FIRST index but it
        # multiplies the Laplacian along ARRAY AXIS 1, because _tensor_diffusion_2d unpacks
        # `Ny, Nx = u.shape` (tensor_calculus.py). The nD branch, `_tensor_diffusion_nd`,
        # uses the opposite convention -- tensor axis i acts on array axis i.
        lap_axis0 = (np.roll(m, -1, axis=0) - 2 * m + np.roll(m, 1, axis=0)) / dx**2
        lap_axis1 = (np.roll(m, -1, axis=1) - 2 * m + np.roll(m, 1, axis=1)) / dy**2
        # Measured: max deviation 2.1e-14 (worst over 300 random draws) against a signal of
        # amplitude ~38; atol 1e-12 is ~47x margin. The swapped assignment
        # (sigma_x*lap_axis0 + sigma_y*lap_axis1) differs by ~22, so this separates the two.
        np.testing.assert_allclose(result_diag, sigma_y * lap_axis0 + sigma_x * lap_axis1, atol=1e-12)


class TestAnisotropic2D:
    """Test anisotropic diffusion in 2D."""

    def test_constant_anisotropic_tensor(self):
        """Test with constant anisotropic tensor."""
        Nx, Ny = 10, 10
        m = np.random.rand(Nx, Ny)
        dx, dy = 0.1, 0.1

        # Anisotropic: higher diffusion in x than y, no cross-terms
        sigma_tensor = np.array([[0.2, 0.0], [0.0, 0.05]])

        bc = periodic_bc(dimension=2)

        result = diffusion(m, sigma_tensor, [dx, dy], bc=bc)

        # Check basic properties
        assert result.shape == m.shape
        assert np.all(np.isfinite(result))

        # Axis-discriminating oracle. Every other tensor test in this file uses an isotropic
        # tensor, under which the axis assignment is invisible; a symmetric probe (X**2 + Y**2)
        # is invariant under the swap and must not be used here.
        #
        # CONVENTION, as measured: _tensor_diffusion_2d unpacks `Ny, Nx = u.shape`, so the
        # tensor's FIRST index addresses the SECOND array axis. With meshgrid(indexing="ij"),
        # u = X**2 varies along array axis 0 and yields 2*D[1,1], not 2*D[0,0].
        # This is the OPPOSITE of the nD branch (see TestNDDispatcher.test_3d_tensor_diffusion,
        # where tensor axis i acts on array axis i). tensor_calculus.py's two branches.
        x = np.linspace(0, 1, 16)
        X, Y = np.meshgrid(x, x, indexing="ij")
        hx = x[1] - x[0]
        D = np.array([[0.2, 0.0], [0.0, 0.05]])
        # Measured deviations 3.1e-15 (X) and 1.3e-14 (Y); atol 1e-10 is >7000x margin.
        rx = diffusion(X**2, D, [hx, hx], bc=periodic_bc(dimension=2))
        np.testing.assert_allclose(rx[2:-2, 2:-2], 2 * D[1, 1], atol=1e-10)
        ry = diffusion(Y**2, D, [hx, hx], bc=periodic_bc(dimension=2))
        np.testing.assert_allclose(ry[2:-2, 2:-2], 2 * D[0, 0], atol=1e-10)

    def test_spatially_varying_tensor(self):
        """Test with spatially-varying diffusion tensor."""
        Nx, Ny = 8, 8
        m = np.random.rand(Nx, Ny)
        dx, dy = 0.1, 0.1

        # Create spatially varying tensor
        sigma_tensor = np.zeros((Nx, Ny, 2, 2))
        for i in range(Nx):
            for j in range(Ny):
                # Increase diffusion with x
                sigma_local = 0.05 + 0.1 * (i / Nx)
                sigma_tensor[i, j] = sigma_local * np.eye(2)

        bc = periodic_bc(dimension=2)

        result = diffusion(m, sigma_tensor, [dx, dy], bc=bc)

        assert result.shape == m.shape
        assert np.all(np.isfinite(result))

        # (1) Single-source pin: a constant field must reproduce the constant-tensor branch
        # byte-for-byte. Measured max difference exactly 0.0 under all three BC families.
        Sigma_const = np.array([[0.15, 0.03], [0.03, 0.2]])
        Sigma_field = np.broadcast_to(Sigma_const, (Nx, Ny, 2, 2)).copy()
        for bc_variant in (periodic_bc(dimension=2), dirichlet_bc(dimension=2), no_flux_bc(dimension=2)):
            np.testing.assert_array_equal(
                diffusion(m, Sigma_const, [dx, dy], bc=bc_variant),
                diffusion(m, Sigma_field, [dx, dy], bc=bc_variant),
            )

        # (2) Closed form for a genuinely varying coefficient. For isotropic D(s) = alpha + beta*s
        # varying along one array axis and u = c*s^2 in the same coordinate, the divergence-form
        # operator gives d/ds(D du/ds) = 2c(alpha + 2 beta s) exactly.
        # u is quadratic on purpose: with u linear the face-averaged and the cell-valued face
        # coefficient agree identically (both give c*beta), so a linear probe cannot see the
        # face averaging at all. Quadratic u separates them by 0.08 here.
        alpha, beta, c, h, N = 0.05, 0.4, 2.0, 0.1, 9
        s = np.arange(N) * h
        for axis in (0, 1):
            coeff_1d = alpha + beta * s
            profile = coeff_1d[:, None] * np.ones((1, N)) if axis == 0 else np.ones((N, 1)) * coeff_1d[None, :]
            u_quad = c * (s[:, None] ** 2 * np.ones((1, N)) if axis == 0 else np.ones((N, 1)) * s[None, :] ** 2)
            exact = 2 * c * (alpha + 2 * beta * (s[:, None] if axis == 0 else s[None, :])) * np.ones((N, N))
            Sigma_var = np.zeros((N, N, 2, 2))
            Sigma_var[..., 0, 0] = profile
            Sigma_var[..., 1, 1] = profile
            r_var = diffusion(u_quad, Sigma_var, [h, h], bc=no_flux_bc(dimension=2))
            # Measured max deviation 6.4e-15 on both axes; atol 1e-10 is >15000x margin.
            np.testing.assert_allclose(r_var[1:-1, 1:-1], exact[1:-1, 1:-1], atol=1e-10)


class TestCrossDiffusion:
    """Test cross-diffusion terms (σ₁₂ ≠ 0)."""

    def test_cross_diffusion_symmetric(self):
        """Test symmetric cross-diffusion tensor."""
        Nx, Ny = 10, 10
        m = np.random.rand(Nx, Ny)
        dx, dy = 0.1, 0.1

        # Symmetric cross-diffusion
        sigma_tensor = np.array([[0.1, 0.02], [0.02, 0.1]])

        # Verify symmetry
        assert np.allclose(sigma_tensor, sigma_tensor.T)

        # Verify PSD (eigenvalues ≥ 0)
        eigenvalues = np.linalg.eigvalsh(sigma_tensor)
        assert np.all(eigenvalues >= -1e-10)

        bc = periodic_bc(dimension=2)

        result = diffusion(m, sigma_tensor, [dx, dy], bc=bc)

        assert result.shape == m.shape
        assert np.all(np.isfinite(result))

    def test_rotation_tensor(self):
        """Test rotated diffusion tensor (anisotropy in rotated coordinates)."""
        # Σ = R diag(σ₁², σ₂²) Rᵀ where R is rotation matrix
        theta = np.pi / 4  # 45 degree rotation
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])

        # Diagonal in rotated frame
        D_rotated = np.diag([0.2, 0.05])

        # Transform back to original frame
        sigma_tensor = R @ D_rotated @ R.T

        # Should still be symmetric and PSD
        assert np.allclose(sigma_tensor, sigma_tensor.T)
        assert np.all(np.linalg.eigvalsh(sigma_tensor) >= -1e-10)

        Nx, Ny = 8, 8
        m = np.random.rand(Nx, Ny)
        dx, dy = 0.1, 0.1
        bc = periodic_bc(dimension=2)

        result = diffusion(m, sigma_tensor, [dx, dy], bc=bc)

        assert result.shape == m.shape
        assert np.all(np.isfinite(result))


class TestBoundaryConditions:
    """Test different boundary conditions."""

    def test_no_flux_bc(self):
        """Test no-flux (Neumann) boundary conditions."""
        Nx, Ny = 8, 8
        m = np.random.rand(Nx, Ny)
        dx, dy = 0.1, 0.1
        sigma_tensor = 0.1 * np.eye(2)

        bc = no_flux_bc(dimension=2)

        result = diffusion(m, sigma_tensor, [dx, dy], bc=bc)

        assert result.shape == m.shape
        assert np.all(np.isfinite(result))

        # No flux through the boundary means the integral of the divergence vanishes
        # (divergence theorem) -- the property the FP side of every MFG problem here relies on.
        # Measured |sum| <= 2.5e-16 (worst over 300 random 8x8 draws); 1e-12 is a ~4000x margin.
        # This discriminates the BC rather than merely holding: the same expression under
        # dirichlet_bc measures ~3.1 on the same input, so a silent degradation of the no-flux
        # ghost treatment to the Dirichlet one fails here.
        assert abs(np.sum(result) * dx * dy) < 1e-12


class TestNDDispatcher:
    """Test nD dispatcher function."""

    def test_1d_fallback(self):
        """Test 1D case falls back to scalar diffusion."""
        Nx = 20
        x = np.linspace(0, 1, Nx, endpoint=False)
        m = np.sin(2 * np.pi * x)
        dx = [x[1] - x[0]]

        # 1D "tensor" is just a scalar
        sigma_tensor = 0.1

        bc = periodic_bc(dimension=1)

        # This should work (fallback to 1D laplacian)
        result = diffusion(m, sigma_tensor, dx, bc=bc)

        assert result.shape == m.shape
        assert np.all(np.isfinite(result))

        # Pin the scalar fast path exactly: D * (three-point stencil with periodic wrap).
        # RFC #1596 -- coeff IS the PDE coefficient D, applied with no internal squaring,
        # so a re-introduced sigma-squaring (the #1549 shape-flip) fails here immediately.
        expected = 0.1 * (np.roll(m, -1) - 2 * m + np.roll(m, 1)) / dx[0] ** 2
        # Measured max deviation 4.4e-16 against a signal of amplitude 3.9; atol 1e-14 is ~20x margin.
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_3d_tensor_diffusion(self):
        """Test that 3D tensor diffusion works (nD implementation)."""
        m = np.random.rand(5, 5, 5)
        dx = [0.1, 0.1, 0.1]
        sigma_tensor = 0.1 * np.eye(3)

        bc = periodic_bc(dimension=3)

        result = diffusion(m, sigma_tensor, dx, bc=bc)
        assert result.shape == m.shape
        assert np.all(np.isfinite(result))

        # Polynomial oracle for the nD branch (_tensor_diffusion_nd, taken for d >= 3).
        # For u = x^2+y^2+z^2 the exact answer is D*Δu = 0.1*6 = 0.6, constant.
        g = np.linspace(0, 1, 7)
        h = g[1] - g[0]
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        result_iso = diffusion(X**2 + Y**2 + Z**2, 0.1 * np.eye(3), [h, h, h], bc=periodic_bc(dimension=3))
        # Measured interior deviation 5.8e-15; atol 1e-12 is ~170x margin.
        np.testing.assert_allclose(result_iso[1:-1, 1:-1, 1:-1], 0.6, atol=1e-12)

        # Axis pin: in the nD branch tensor axis i acts on ARRAY axis i, so D = diag(0.3, 0.1, 0.05)
        # gives 2*D[i,i] on the corresponding quadratic. (The 2D branch uses the OPPOSITE
        # convention -- see TestAnisotropic2D.test_constant_anisotropic_tensor.)
        D = np.diag([0.3, 0.1, 0.05])
        for field, expected in ((X**2, 0.6), (Y**2, 0.2), (Z**2, 0.1)):
            r = diffusion(field, D, [h, h, h], bc=periodic_bc(dimension=3))
            np.testing.assert_allclose(r[1:-1, 1:-1, 1:-1], expected, atol=1e-12)


class TestMassConservation:
    """Test that diffusion conserves mass (with appropriate BCs)."""

    def test_periodic_bc_conserves_mass(self):
        """Periodic BC with zero-mean should preserve total mass."""
        Nx, Ny = 16, 16
        x = np.linspace(0, 1, Nx, endpoint=False)
        y = np.linspace(0, 1, Ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Zero-mean initial condition
        m = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        assert abs(np.sum(m)) < 1e-10

        dx = x[1] - x[0]
        dy = y[1] - y[0]
        sigma_tensor = 0.1 * np.eye(2)

        bc = periodic_bc(dimension=2)

        diffusion_term = diffusion(m, sigma_tensor, [dx, dy], bc=bc)

        # Integral of divergence should be zero (by divergence theorem)
        total_diffusion = np.sum(diffusion_term) * dx * dy
        assert abs(total_diffusion) < 1e-6


class TestNumericalAccuracy:
    """Test numerical accuracy against known solutions."""

    def test_laplacian_of_polynomial(self):
        """Test Laplacian of quadratic function."""
        # For m(x,y) = x² + y², we have:
        # Δm = ∂²m/∂x² + ∂²m/∂y² = 2 + 2 = 4 (constant)

        Nx, Ny = 16, 16
        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)
        X, Y = np.meshgrid(x, y, indexing="ij")

        m = X**2 + Y**2

        dx = x[1] - x[0]
        dy = y[1] - y[0]
        sigma_squared = 0.1
        sigma_tensor = sigma_squared * np.eye(2)

        bc = dirichlet_bc(dimension=2)

        result = diffusion(m, sigma_tensor, [dx, dy], bc=bc)

        # Analytical: σ² Δm = 0.1 * 4 = 0.4 (constant)
        expected = 0.4 * np.ones_like(m)

        # Check interior points (boundaries affected by BC)
        np.testing.assert_allclose(result[2:-2, 2:-2], expected[2:-2, 2:-2], atol=0.05)

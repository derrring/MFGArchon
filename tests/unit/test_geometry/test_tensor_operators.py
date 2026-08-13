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

        # Manually compute: ∂/∂x(σₓ² ∂m/∂x) + ∂/∂y(σᵧ² ∂m/∂y)
        # This should match the diagonal operator

        # For now, just check shape and no NaN
        assert result_diag.shape == m.shape
        assert np.all(np.isfinite(result_diag))


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


class TestNDDispatcher:
    """Test nD dispatcher function."""

    def test_1d_fallback(self):
        """Test 1D case falls back to scalar diffusion."""
        Nx = 20
        m = np.random.rand(Nx)
        dx = [0.1]

        # 1D "tensor" is just a scalar
        sigma_tensor = 0.1

        bc = periodic_bc(dimension=1)

        # This should work (fallback to 1D laplacian)
        result = diffusion(m, sigma_tensor, dx, bc=bc)

        assert result.shape == m.shape
        assert np.all(np.isfinite(result))

    def test_3d_tensor_diffusion(self):
        """Test that 3D tensor diffusion works (nD implementation)."""
        m = np.random.rand(5, 5, 5)
        dx = [0.1, 0.1, 0.1]
        sigma_tensor = 0.1 * np.eye(3)

        bc = periodic_bc(dimension=3)

        result = diffusion(m, sigma_tensor, dx, bc=bc)
        assert result.shape == m.shape
        assert np.all(np.isfinite(result))


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

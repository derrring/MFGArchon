"""
Quick smoke test for shared ghost cell formula methods (Issue #598).

Tests the new shared methods in BaseStructuredApplicator to verify:
1. Dirichlet ghost cell formula
2. Neumann ghost cell formula
3. Robin ghost cell formula
4. Validation and utility methods

Run: python mfgarchon/geometry/boundary/_test_shared_ghost_formulas.py
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary.applicator_base import BaseStructuredApplicator, GridType


class TestApplicator(BaseStructuredApplicator):
    """Test subclass for BaseStructuredApplicator."""

    def __init__(self, dimension: int = 1, grid_type: GridType = GridType.CELL_CENTERED):
        super().__init__(dimension, grid_type)


def test_dirichlet_cell_centered():
    """Test Dirichlet ghost cell formula for cell-centered grid."""
    print("Testing Dirichlet (cell-centered)...")

    applicator = TestApplicator(dimension=1, grid_type=GridType.CELL_CENTERED)

    # Test with scalar
    u_interior = 0.5
    g = 1.0
    u_ghost = applicator._compute_ghost_dirichlet(u_interior, g)
    expected = 2.0 * g - u_interior  # 2*1.0 - 0.5 = 1.5
    assert np.isclose(u_ghost, expected), f"Expected {expected}, got {u_ghost}"
    print(f"  ✓ Scalar: u_ghost = {u_ghost} (expected {expected})")

    # Test with array
    u_interior = np.array([0.5, 0.6, 0.7])
    g = 1.0
    u_ghost = applicator._compute_ghost_dirichlet(u_interior, g)
    expected = 2.0 * g - u_interior  # [1.5, 1.4, 1.3]
    assert np.allclose(u_ghost, expected), f"Expected {expected}, got {u_ghost}"
    print(f"  ✓ Array: u_ghost = {u_ghost}")

    # Test with callable BC value
    def g_func(t):
        return 1.0 + 0.5 * t

    u_ghost = applicator._compute_ghost_dirichlet(u_interior, g_func, time=2.0)
    expected = 2.0 * (1.0 + 0.5 * 2.0) - u_interior  # 2*2.0 - u_interior
    assert np.allclose(u_ghost, expected), f"Expected {expected}, got {u_ghost}"
    print(f"  ✓ Callable: u_ghost = {u_ghost} (g(t=2.0) = {g_func(2.0)})")


def test_dirichlet_vertex_centered():
    """Test Dirichlet ghost cell formula for vertex-centered grid."""
    print("\nTesting Dirichlet (vertex-centered)...")

    applicator = TestApplicator(dimension=1, grid_type=GridType.VERTEX_CENTERED)

    # Test with scalar
    u_interior = 0.5
    g = 1.0
    u_ghost = applicator._compute_ghost_dirichlet(u_interior, g)
    expected = g  # Ghost = boundary value directly
    assert np.isclose(u_ghost, expected), f"Expected {expected}, got {u_ghost}"
    print(f"  ✓ Scalar: u_ghost = {u_ghost} (expected {expected})")

    # Test with array
    u_interior = np.array([0.5, 0.6, 0.7])
    g = 1.0
    u_ghost = applicator._compute_ghost_dirichlet(u_interior, g)
    expected = np.full_like(u_interior, g)
    assert np.allclose(u_ghost, expected), f"Expected {expected}, got {u_ghost}"
    print(f"  ✓ Array: u_ghost = {u_ghost}")


def test_neumann_zero_flux():
    """Test Neumann ghost cell formula for zero-flux case."""
    print("\nTesting Neumann (zero-flux)...")

    applicator = TestApplicator(dimension=1, grid_type=GridType.CELL_CENTERED)

    # Zero-flux: ghost = u_next_interior (reflection)
    u_interior = 0.5
    u_next_interior = 0.7
    g = 0.0  # Zero flux
    dx = 0.1

    u_ghost = applicator._compute_ghost_neumann(u_interior, u_next_interior, g, dx, side="left")
    expected = u_next_interior  # Reflection
    assert np.isclose(u_ghost, expected), f"Expected {expected}, got {u_ghost}"
    print(f"  ✓ Left: u_ghost = {u_ghost} (reflection)")

    u_ghost = applicator._compute_ghost_neumann(u_interior, u_next_interior, g, dx, side="right")
    expected = u_next_interior  # Reflection
    assert np.isclose(u_ghost, expected), f"Expected {expected}, got {u_ghost}"
    print(f"  ✓ Right: u_ghost = {u_ghost} (reflection)")


def test_neumann_nonzero_flux():
    """Test Neumann ghost cell formula for non-zero flux."""
    print("\nTesting Neumann (non-zero flux)...")

    applicator = TestApplicator(dimension=1, grid_type=GridType.CELL_CENTERED)

    u_interior = 0.5
    u_next_interior = 0.7
    g = 0.1  # Non-zero flux
    dx = 0.1

    # Left boundary: u_ghost = u_next_interior - 2*dx*g
    u_ghost = applicator._compute_ghost_neumann(u_interior, u_next_interior, g, dx, side="left")
    expected = u_next_interior - 2.0 * dx * g
    assert np.isclose(u_ghost, expected), f"Expected {expected}, got {u_ghost}"
    print(f"  ✓ Left: u_ghost = {u_ghost} (expected {expected})")

    # Right boundary: u_ghost = u_next_interior + 2*dx*g
    u_ghost = applicator._compute_ghost_neumann(u_interior, u_next_interior, g, dx, side="right")
    expected = u_next_interior + 2.0 * dx * g
    assert np.isclose(u_ghost, expected), f"Expected {expected}, got {u_ghost}"
    print(f"  ✓ Right: u_ghost = {u_ghost} (expected {expected})")


def test_robin():
    """Test Robin ghost cell formula.

    Cell-centered grid: the boundary lies at the cell face, with the ghost
    cell center and the first interior cell center straddling it exactly ``dx``
    apart (each ``dx/2`` from the face). Robin BC ``alpha*u + beta*du/dn = g``
    discretizes as

        value at face   u_face   = (u_ghost + u_interior) / 2
        normal deriv    du/dn    = (u_ghost - u_interior) / dx   (cells dx apart)

    giving ``alpha*(u_ghost+u_interior)/2 + beta*(u_ghost-u_interior)/dx = g``,
    i.e. ``u_ghost*(alpha/2 + beta/dx) = g - u_interior*(alpha/2 - beta/dx)``.

    The ``beta/dx`` factor (not ``beta/(2*dx)``) is required for consistency with
    the ``(u_ghost+u_interior)/2`` value term, which commits to the face-midpoint
    geometry where the two cells are ``dx`` apart. This matches the Dirichlet
    sibling (``2*g - u_interior`` from the same face-midpoint average) and the
    production fix in commit 0ae5515a. The previous expected formula used
    ``beta/(2*dx)`` (the stale pre-fix factor), which xfailed against the
    corrected production code (Refs #1237).
    """
    print("\nTesting Robin...")

    applicator = TestApplicator(dimension=1, grid_type=GridType.CELL_CENTERED)

    u_interior = 0.5
    alpha = 1.0
    beta = 0.1
    g = 1.0
    dx = 0.1

    # Robin: u_ghost = (g - u_interior * (alpha/2 - beta/dx)) / (alpha/2 + beta/dx)
    u_ghost = applicator._compute_ghost_robin(u_interior, alpha, beta, g, dx, side="left")
    coeff_ghost = alpha / 2.0 + beta / dx
    coeff_interior = alpha / 2.0 - beta / dx
    expected = (g - u_interior * coeff_interior) / coeff_ghost
    assert np.isclose(u_ghost, expected), f"Expected {expected}, got {u_ghost}"
    print(f"  ✓ u_ghost = {u_ghost} (expected {expected})")


def test_validation():
    """Test field validation."""
    print("\nTesting validation...")

    applicator = TestApplicator(dimension=2, grid_type=GridType.CELL_CENTERED)

    # Valid field
    field = np.ones((10, 10))
    applicator._validate_field(field)  # Should not raise
    print("  ✓ Valid field accepted")

    # Field with NaN should raise ValueError
    field_nan = np.ones((10, 10))
    field_nan[5, 5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        applicator._validate_field(field_nan)
    print("  ✓ NaN detected")

    # Field with Inf should raise ValueError
    field_inf = np.ones((10, 10))
    field_inf[5, 5] = np.inf
    with pytest.raises(ValueError, match="Inf"):
        applicator._validate_field(field_inf)
    print("  ✓ Inf detected")


def test_buffer_creation():
    """Test padded buffer creation."""
    print("\nTesting buffer creation...")

    applicator = TestApplicator(dimension=2, grid_type=GridType.CELL_CENTERED)

    # Create padded buffer for 2D field
    field = np.ones((10, 10)) * 0.5
    padded = applicator._create_padded_buffer(field, ghost_depth=1)

    # Check shape
    assert padded.shape == (12, 12), f"Expected (12, 12), got {padded.shape}"
    print(f"  ✓ Shape: {padded.shape}")

    # Check interior values
    assert np.allclose(padded[1:-1, 1:-1], field), "Interior values not preserved"
    print("  ✓ Interior values preserved")

    # Check ghost cells initialized to zero
    assert np.allclose(padded[0, :], 0.0), "Top ghost cells not zero"
    assert np.allclose(padded[-1, :], 0.0), "Bottom ghost cells not zero"
    assert np.allclose(padded[:, 0], 0.0), "Left ghost cells not zero"
    assert np.allclose(padded[:, -1], 0.0), "Right ghost cells not zero"
    print("  ✓ Ghost cells initialized to zero")


def test_grid_spacing():
    """Test grid spacing computation."""
    print("\nTesting grid spacing computation...")

    applicator = TestApplicator(dimension=2, grid_type=GridType.CELL_CENTERED)

    # 2D field with known domain bounds
    field = np.ones((10, 20))
    domain_bounds = np.array([[0.0, 1.0], [0.0, 2.0]])  # [0,1] x [0,2]

    spacing = applicator._compute_grid_spacing(field, domain_bounds)

    # Expected: dx = 1.0 / (10 - 1) = 1/9, dy = 2.0 / (20 - 1) = 2/19
    expected_dx = 1.0 / (10 - 1)
    expected_dy = 2.0 / (20 - 1)

    assert np.isclose(spacing[0], expected_dx), f"Expected dx={expected_dx}, got {spacing[0]}"
    assert np.isclose(spacing[1], expected_dy), f"Expected dy={expected_dy}, got {spacing[1]}"
    print(f"  ✓ Spacing: dx={spacing[0]:.6f}, dy={spacing[1]:.6f}")


if __name__ == "__main__":
    print("=" * 70)
    print("Smoke Test: Shared Ghost Cell Formula Methods (Issue #598)")
    print("=" * 70)

    test_dirichlet_cell_centered()
    test_dirichlet_vertex_centered()
    test_neumann_zero_flux()
    test_neumann_nonzero_flux()
    test_robin()
    test_validation()
    test_buffer_creation()
    test_grid_spacing()

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    print("\nShared ghost cell formula methods are ready to use.")
    print("Next: Migrate FDMApplicator to use these shared methods (Phase 2)")

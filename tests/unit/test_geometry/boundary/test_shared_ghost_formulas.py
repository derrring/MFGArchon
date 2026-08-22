"""
Quick smoke test for shared ghost cell formula methods (Issue #598).

Tests the utility methods on BaseStructuredApplicator.

The ghost-formula tests that lived here were removed with the methods they covered (#2057):
`_compute_ghost_{dirichlet,neumann,robin}` had zero production callers and encoded `du/dx` where
the live path uses `du/dn`, so the tests pinned a convention nothing consumed.

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
    print("Smoke Test: BaseStructuredApplicator utility methods (Issue #598)")
    print("=" * 70)

    test_validation()
    test_buffer_creation()
    test_grid_spacing()

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    print("\nShared ghost cell formula methods are ready to use.")
    print("Next: Migrate FDMApplicator to use these shared methods (Phase 2)")

"""
Smoke tests for PreallocatedGhostBuffer order parameter (Issue #576).

Tests the newly added order parameter and dispatch logic.
"""

import pytest

import numpy as np

from mfgarchon.geometry.boundary import neumann_bc
from mfgarchon.geometry.boundary.applicator_fdm import PreallocatedGhostBuffer


def test_default_order():
    """Test that default order=2 works correctly."""
    bc = neumann_bc(dimension=1)
    buffer = PreallocatedGhostBuffer(
        interior_shape=(10,),
        boundary_conditions=bc,
        domain_bounds=np.array([[0.0, 1.0]]),
    )

    # Verify order was set
    assert buffer._order == 2

    # Set interior values that make ghost verification clear
    # Use [1, 2, ...] not [0, 1, ...] so ghost != 0 is meaningful
    buffer.interior[:] = np.arange(1, 11, dtype=np.float64)

    # Update ghosts (should use linear reflection)
    buffer.update_ghosts(time=0.0)

    # For Neumann BC with g=0 the ghost mirrors the node beside the wall node (#1935)
    assert buffer.padded[0] == buffer.interior[1]
    assert buffer.padded[-1] == buffer.interior[-2]


def test_explicit_order_2():
    """Test explicit order=2 parameter."""
    bc = neumann_bc(dimension=1)
    buffer = PreallocatedGhostBuffer(
        interior_shape=(10,),
        boundary_conditions=bc,
        domain_bounds=np.array([[0.0, 1.0]]),
        order=2,
    )

    assert buffer._order == 2

    # Should work identically to default
    buffer.interior[:] = np.arange(1, 11, dtype=np.float64)
    buffer.update_ghosts(time=0.0)

    # Verify Neumann BC: with g=0 the ghost mirrors the node beside the wall node (#1935)
    assert buffer.padded[0] == buffer.interior[1]


def test_order_validation():
    """Test that order < 1 raises ValueError."""
    bc = neumann_bc(dimension=1)

    with pytest.raises(ValueError, match="order must be >= 1"):
        PreallocatedGhostBuffer(
            interior_shape=(10,),
            boundary_conditions=bc,
            order=0,
        )

    with pytest.raises(ValueError, match="order must be >= 1"):
        PreallocatedGhostBuffer(
            interior_shape=(10,),
            boundary_conditions=bc,
            order=-1,
        )


def test_order_1_allowed():
    """Test that order=1 is allowed (edge case)."""
    bc = neumann_bc(dimension=1)
    buffer = PreallocatedGhostBuffer(
        interior_shape=(10,),
        boundary_conditions=bc,
        order=1,
    )

    assert buffer._order == 1

    # Should use linear reflection (order <= 2)
    buffer.interior[:] = np.arange(10, dtype=np.float64)
    buffer.update_ghosts(time=0.0)


def test_order_3_works():
    """Test that order=3 works with polynomial extrapolation (Phase 4 complete)."""
    bc = neumann_bc(dimension=1)
    buffer = PreallocatedGhostBuffer(
        interior_shape=(10,),
        boundary_conditions=bc,
        order=3,
    )

    assert buffer._order == 3

    # Should work with polynomial extrapolation
    buffer.interior[:] = np.arange(10, dtype=np.float64)
    buffer.update_ghosts(time=0.0)

    # Check that ghost cells were filled
    assert buffer.padded[0] != 0.0  # Ghost cell should be set
    assert not np.isnan(buffer.padded[0])  # Should be valid number


def test_order_5_works():
    """Test that order=5 (for WENO) works with polynomial extrapolation."""
    bc = neumann_bc(dimension=1)
    buffer = PreallocatedGhostBuffer(
        interior_shape=(20,),  # Need enough points for order-5 stencil
        boundary_conditions=bc,
        ghost_depth=3,  # WENO needs 3 ghost cells
        order=5,
    )

    assert buffer._order == 5
    assert buffer._ghost_depth == 3

    # Should work with polynomial extrapolation
    buffer.interior[:] = np.arange(20, dtype=np.float64)
    buffer.update_ghosts(time=0.0)

    # Check that all 3 ghost cells were filled
    assert buffer.padded[0] != 0.0
    assert buffer.padded[1] != 0.0
    assert buffer.padded[2] != 0.0
    assert not np.isnan(buffer.padded[0])  # Should be valid numbers


def test_backward_compatibility():
    """Test that existing code without order parameter still works."""
    bc = neumann_bc(dimension=2)

    # Old-style call without order parameter
    buffer = PreallocatedGhostBuffer(
        interior_shape=(20, 30),
        boundary_conditions=bc,
        domain_bounds=np.array([[0.0, 1.0], [0.0, 1.5]]),
    )

    # Should default to order=2
    assert buffer._order == 2

    # Should work as before
    buffer.interior[:] = np.random.rand(20, 30)
    buffer.update_ghosts(time=0.0)

    # Verify ghost cells were updated
    assert not np.allclose(buffer.padded[0, :], 0.0)
    assert not np.allclose(buffer.padded[-1, :], 0.0)


if __name__ == "__main__":
    """Run smoke tests."""
    print("Testing PreallocatedGhostBuffer order parameter...")

    print("✓ test_default_order")
    test_default_order()

    print("✓ test_explicit_order_2")
    test_explicit_order_2()

    print("✓ test_order_validation")
    test_order_validation()

    print("✓ test_order_1_allowed")
    test_order_1_allowed()

    print("✓ test_order_3_works")
    test_order_3_works()

    print("✓ test_order_5_works")
    test_order_5_works()

    print("✓ test_backward_compatibility")
    test_backward_compatibility()

    print("\n✅ All smoke tests passed!")

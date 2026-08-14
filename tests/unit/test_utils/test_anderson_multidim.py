"""
Regression Test for Anderson Accelerator Multi-Dimensional Support

Tests that AndersonAccelerator correctly handles multi-dimensional arrays.

Issue:
    The original Anderson accelerator used `np.column_stack` which expects 1D arrays.
    When used with 2D arrays (e.g., MFG density m(x,y) on a grid), it would fail.

Fix Location:
    mfgarchon/alg/numerical/coupling/anderson_acceleration.py
    (~~mfgarchon/utils/numerical/anderson_acceleration.py:112-207~~ [CORRECTED 2026-08-14] --
    that shim was 23 lines and never had those, and it is now deleted as overdue since v0.21.0)

Test Strategy:
    1. Test with 1D arrays (original functionality)
    2. Test with 2D arrays (bug case - MFG on 2D grid)
    3. Test with 3D arrays (extensibility)
    4. Verify output shape matches input shape
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.anderson_acceleration import AndersonAccelerator

pytestmark = pytest.mark.experimental


def test_1d_arrays():
    """Test that 1D arrays still work (original functionality)"""
    # Use damping for stability
    accelerator = AndersonAccelerator(depth=3, beta=0.5)

    # Fixed-point iteration x_{k+1} = 0.5 * x_k + b, whose fixed point x* = 2b differs from element
    # to element. A uniform b (the previous 0.5) makes every element carry the same value, so a
    # flatten/unflatten pair that preserved shape while permuting elements was invisible.
    b = np.linspace(0.1, 1.0, 3)
    x = np.zeros(3)  # 1D array

    for _i in range(40):
        f = 0.5 * x + b
        x_next = accelerator.update(x, f)
        assert x_next.shape == x.shape, f"Shape mismatch: {x_next.shape} vs {x.shape}"
        assert isinstance(x_next, np.ndarray), f"Expected ndarray, got {type(x_next)}"
        x = x_next

    # Main test: verify shape preservation (not convergence rate)
    assert x.shape == (3,), f"Shape changed: expected (3,), got {x.shape}"

    # Analytic fixed point of x <- 0.5x + b. Measured max|x - 2b| = 1.24e-10 after 40 iterations;
    # atol=1e-8 leaves ~80x margin.
    np.testing.assert_allclose(x, 2.0 * b, atol=1e-8)
    print("✓ 1D arrays: PASS (shape preserved)")


def test_2d_arrays():
    """Test that 2D arrays work (Bug fix case - MFG density on grid)"""
    accelerator = AndersonAccelerator(depth=3, beta=0.5)

    # 2D fixed-point iteration with a per-cell offset, so the fixed point 2b is different in every
    # cell. This is what pins the element ordering across the ravel/reshape the fix introduced --
    # the column_stack-class bug that survives a shape check is a permutation, not a reshape.
    b = np.linspace(0.1, 1.0, 25).reshape(5, 5)
    x = np.zeros((5, 5))  # 2D array (like MFG density on 5×5 grid)

    for _i in range(40):
        f = 0.5 * x + b
        x_next = accelerator.update(x, f)
        assert x_next.shape == x.shape, f"Shape mismatch: {x_next.shape} vs {x.shape}"
        assert isinstance(x_next, np.ndarray), f"Expected ndarray, got {type(x_next)}"
        x = x_next

    # Main test: verify shape preservation (not convergence)
    assert x.shape == (5, 5), f"Shape changed: expected (5, 5), got {x.shape}"

    # Measured max|x - 2b| = 2.06e-11 after 40 iterations (atol=1e-8 leaves ~480x margin), while the
    # same comparison against a transposed 2b is off by 1.200 -- so this separates orderings.
    np.testing.assert_allclose(x, 2.0 * b, atol=1e-8)
    print("✓ 2D arrays: PASS (Bug fix verified - no crash, shape preserved)")


def test_3d_arrays():
    """Test that 3D arrays work (Extensibility test)"""
    accelerator = AndersonAccelerator(depth=3, beta=0.5)

    # 3D fixed-point iteration with a per-element offset. The three axis lengths are distinct, so a
    # reshape that transposed or mis-ordered axes cannot accidentally still fit -- the strongest of
    # the three ordering pins.
    b = np.linspace(0.1, 1.0, 60).reshape(3, 4, 5)
    x = np.zeros((3, 4, 5))  # 3D array

    for _i in range(40):
        f = 0.5 * x + b
        x_next = accelerator.update(x, f)
        assert x_next.shape == x.shape, f"Shape mismatch: {x_next.shape} vs {x.shape}"
        assert isinstance(x_next, np.ndarray), f"Expected ndarray, got {type(x_next)}"
        x = x_next

    # Main test: verify shape preservation
    assert x.shape == (3, 4, 5), f"Shape changed: expected (3, 4, 5), got {x.shape}"

    # Measured max|x - 2b| = 1.07e-11 after 40 iterations; atol=1e-8 leaves ~930x margin.
    np.testing.assert_allclose(x, 2.0 * b, atol=1e-8)
    print("✓ 3D arrays: PASS (Extensibility verified)")


def test_type2_2d():
    """Test Type II Anderson acceleration with 2D arrays"""
    accelerator = AndersonAccelerator(depth=3, beta=0.5)

    # 2D fixed-point iteration with Type II method, non-uniform fixed point 2b
    b = np.linspace(0.1, 1.0, 16).reshape(4, 4)
    x = np.zeros((4, 4))

    for _i in range(40):
        f = 0.5 * x + b
        x_next = accelerator.update(x, f, method="type2")
        assert x_next.shape == x.shape, f"Shape mismatch: {x_next.shape} vs {x.shape}"
        assert isinstance(x_next, np.ndarray), f"Expected ndarray, got {type(x_next)}"
        x = x_next

    # Main test: verify shape preservation with Type II
    assert x.shape == (4, 4), f"Shape changed: expected (4, 4), got {x.shape}"

    # type2 drives this affine map to its analytic fixed point at machine precision: measured
    # max|x - 2b| = 4.44e-16, i.e. eps for values of order 2, so atol=1e-12 sits ~2000x above the
    # noise floor. The tolerance is also what discriminates the branch -- the default (type1) run of
    # the same 40 iterations lands at 2.34e-11 and would fail here, so a silent fallback is caught.
    np.testing.assert_allclose(x, 2.0 * b, atol=1e-12)
    print("✓ Type II with 2D arrays: PASS")


def test_realistic_mfg_shape():
    """Test with realistic MFG density shape (50×50 grid)"""
    accelerator = AndersonAccelerator(depth=5, beta=0.5)

    # Simulate MFG density on 50×50 spatial grid
    m = np.ones((50, 50)) / 2500  # Initial uniform density

    for _i in range(5):
        # Simulate FP evolution (simplified)
        m_next = 0.9 * m + 0.1 * np.ones((50, 50)) / 2500
        m = accelerator.update(m, m_next)

        assert m.shape == (50, 50), f"Shape mismatch: {m.shape} vs (50, 50)"

    # The uniform initial density is an exact fixed point of m <- 0.9m + 0.1*uniform, so every
    # residual in the history is identically zero and the Anderson least-squares system is
    # rank-deficient. That singular-history path is what this fixture reaches, and what it must
    # produce is the iterate untouched: measured deviation exactly 0.0 after 5 iterations (an
    # unregularised lstsq would return NaN here and pass a shape check).
    np.testing.assert_array_equal(m, np.ones((50, 50)) / 2500)
    assert np.isfinite(m).all()

    # Main test: large 2D arrays work without crashes
    print("✓ Realistic MFG shape (50×50): PASS (no crash with large 2D arrays)")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Anderson Accelerator Multi-Dimensional Regression Tests")
    print("=" * 80 + "\n")

    test_1d_arrays()
    test_2d_arrays()
    test_3d_arrays()
    test_type2_2d()
    test_realistic_mfg_shape()

    print("\n" + "=" * 80)
    print("All tests PASSED - Anderson multi-dimensional support verified!")
    print("=" * 80 + "\n")

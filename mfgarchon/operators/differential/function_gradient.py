"""
Pointwise gradient of callable functions via finite differences.

This module provides gradient computation for callable functions (like SDFs)
at arbitrary points, as opposed to grid-based gradient operators which work
on field arrays.

Use Cases:
    - SDF gradient at boundary points (for outward normals)
    - Function optimization
    - Sensitivity analysis

Consolidates Issue #662: Three duplicate SDF gradient implementations
combined into single canonical implementation.

Mathematical Background:
    Central differences: df/dxi = (f(x + e*ei) - f(x - e*ei)) / (2*e)

    For SDF phi, the outward normal is: n = grad(phi) / |grad(phi)|
    (SDF convention: phi > 0 outside, so grad points outward)

References:
    - Issue #662: SDF gradient consolidation
    - Issue #661: Universal outward normal convention

Created: 2026-01-25 (Issue #662)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from mfgarchon.types.callable_protocols import read_region

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@runtime_checkable
class HasAnalyticalGradient(Protocol):
    """Protocol for objects that can provide analytical SDF gradient."""

    def signed_distance_gradient(self, points: NDArray) -> NDArray:
        """Analytical gradient of signed distance function."""
        ...


def function_gradient(
    func: Callable[[NDArray], NDArray | float],
    points: NDArray,
    eps: float = 1e-6,
    adaptive_eps: bool = False,
) -> NDArray:
    """
    Compute gradient of scalar function at given points via central differences.

    This is the canonical implementation for pointwise function gradient,
    consolidating previous duplicates in boundary/types.py, collocation.py,
    and applicator_implicit.py.

    Args:
        func: Scalar function f: R^d -> R
            - Single point: func(x) -> float, where x has shape (d,)
            - Batch: func(points) -> array, where points has shape (n, d)
        points: Evaluation points
            - Single point: shape (d,)
            - Batch: shape (n, d)
        eps: Base finite difference step size (default 1e-6)
        adaptive_eps: If True, scale eps by coordinate magnitude for
            numerical stability across different scales

    Returns:
        Gradient vectors, same shape as points:
            - Single point: shape (d,)
            - Batch: shape (n, d)

    Example:
        >>> # Single point
        >>> def sphere_sdf(x):
        ...     return np.linalg.norm(x) - 1.0
        >>> grad = function_gradient(sphere_sdf, np.array([1.0, 0.0, 0.0]))
        >>> print(grad)  # [1, 0, 0] (outward normal)

        >>> # Batch of points
        >>> points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        >>> grads = function_gradient(sphere_sdf, points)

    Note:
        For SDF functions, the gradient points in the direction of steepest
        increase (outward from the surface). Use outward_normal_from_sdf()
        to get unit normals.
    """
    points = np.asarray(points, dtype=np.float64)
    is_single_point = points.ndim == 1

    # Ensure 2D for uniform processing
    if is_single_point:
        points = points.reshape(1, -1)

    n_points, dim = points.shape
    grad = np.zeros((n_points, dim), dtype=np.float64)

    for d in range(dim):
        # Compute epsilon for this dimension
        if adaptive_eps:
            # Scale by coordinate magnitude (from boundary/types.py)
            coord_scale = np.abs(points[:, d])
            eps_d = eps * np.maximum(coord_scale, 1.0)
        else:
            eps_d = eps

        # Perturb in dimension d
        points_plus = points.copy()
        points_minus = points.copy()

        if adaptive_eps:
            points_plus[:, d] += eps_d
            points_minus[:, d] -= eps_d
            # Central difference with varying step
            f_plus = _eval_func(func, points_plus)
            f_minus = _eval_func(func, points_minus)
            grad[:, d] = (f_plus - f_minus) / (2 * eps_d)
        else:
            points_plus[:, d] += eps
            points_minus[:, d] -= eps
            f_plus = _eval_func(func, points_plus)
            f_minus = _eval_func(func, points_minus)
            grad[:, d] = (f_plus - f_minus) / (2 * eps)

    # Validate output
    if not np.all(np.isfinite(grad)):
        raise ValueError("Function returned non-finite values during gradient computation")

    # Return in original shape
    if is_single_point:
        return grad[0]
    return grad


def _eval_func(
    func: Callable[[NDArray], NDArray | float],
    points: NDArray,
) -> NDArray:
    """
    Evaluate function, handling both single-point and batch interfaces.

    Args:
        func: Function to evaluate
        points: Points array, shape (n, d)

    Returns:
        Function values, shape (n,)
    """
    result = func(points)
    result = np.asarray(result, dtype=np.float64)

    # Handle scalar return from batch input
    if result.ndim == 0:
        result = result.reshape(1)

    return result.ravel()


def outward_normal_from_sdf(
    sdf: Callable[..., NDArray | float],
    points: NDArray,
    eps: float = 1e-6,
    adaptive_eps: bool = False,
) -> NDArray:
    """
    Compute outward unit normal from SDF gradient.

    The SDF convention is phi > 0 outside the domain, so grad(phi)
    points outward. This function normalizes the gradient to unit length.

    Args:
        sdf: Signed distance function
            Convention: phi < 0 inside, phi > 0 outside, phi = 0 on boundary
        points: Evaluation points, shape (d,) or (n, d)
        eps: Finite difference step size
        adaptive_eps: Scale eps by coordinate magnitude

    Returns:
        Unit outward normal vectors, same shape as points

    Example:
        >>> # Circle SDF: x is a coordinate tuple
        >>> def circle_sdf(x):
        ...     return np.sqrt(x[0] ** 2 + x[1] ** 2) - 1.0
        >>> normals = outward_normal_from_sdf(circle_sdf, boundary_points)
    """
    return _unit_normal(
        function_gradient(_region_at_rows(sdf, points, "sdf"), points, eps=eps, adaptive_eps=adaptive_eps)
    )


def _region_at_rows(sdf: Callable[..., NDArray | float], points: NDArray, role: str) -> Callable[[NDArray], NDArray]:
    """``sdf`` as `function_gradient` evaluates it, on the ``(n, d)`` rows it builds: at one point where ``points`` is
    one, of shape ``(d,)``, so a body is handed a point as a tuple of scalars (#2875)."""
    if np.ndim(points) == 1:
        return lambda rows: read_region(sdf, rows[0], role=role, one_point=True)
    return lambda rows: read_region(sdf, rows, role=role)


def geometry_outward_normal(signed_distance: Callable[..., NDArray], points: NDArray, eps: float = 1e-6) -> NDArray:
    """The outward normal from a geometry's own ``signed_distance`` method, which takes points as an array by the
    ``ImplicitGeometry`` contract, on a subclass a user wrote as well as on the library's."""
    return _unit_normal(function_gradient(signed_distance, points, eps=eps))


def _unit_normal(grad: NDArray) -> NDArray:
    if grad.ndim == 1:
        norm = np.linalg.norm(grad)
        if norm < 1e-12:
            raise ValueError("SDF gradient is zero (singular point)")
        return grad / norm
    else:
        norms = np.linalg.norm(grad, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # Avoid division by zero
        return grad / norms


def sdf_gradient_with_analytical_fallback(
    geometry: HasAnalyticalGradient | object,
    points: NDArray,
    sdf_func: Callable[..., NDArray] | None = None,
    eps: float = 1e-6,
) -> NDArray:
    """
    Compute SDF gradient, preferring analytical method if available.

    This function first tries to use the geometry's analytical gradient
    method (if it exists), falling back to numerical finite differences.

    Args:
        geometry: Geometry object, may have signed_distance_gradient method
        points: Evaluation points, shape (n, d)
        sdf_func: SDF function for numerical fallback, reading x as a coordinate tuple
            (if None, uses geometry.signed_distance, which takes points as an array)
        eps: Finite difference step size for fallback

    Returns:
        Gradient vectors, shape (n, d)

    Example:
        >>> grad = sdf_gradient_with_analytical_fallback(
        ...     geometry=my_implicit_domain,
        ...     points=boundary_points,
        ... )
    """
    # Try analytical gradient first (from collocation.py pattern)
    if isinstance(geometry, HasAnalyticalGradient):
        try:
            return geometry.signed_distance_gradient(points)
        except (AttributeError, NotImplementedError):
            pass

    # Fallback to numerical
    if sdf_func is not None:
        return function_gradient(_region_at_rows(sdf_func, points, "sdf_func"), points, eps=eps)
    else:
        # Get SDF function from geometry via explicit interface (Issue #794)
        # ImplicitGeometry ABC guarantees signed_distance(); getattr for legacy alias
        from mfgarchon.geometry.base import ImplicitGeometry

        if isinstance(geometry, ImplicitGeometry):
            sdf_func = geometry.signed_distance
        else:
            # Legacy fallback: optional attribute access (CLAUDE.md getattr pattern)
            sdf_candidate = getattr(geometry, "signed_distance", None)
            if not callable(sdf_candidate):
                sdf_candidate = getattr(geometry, "sdf", None)
            if callable(sdf_candidate):
                sdf_func = sdf_candidate
            else:
                raise ValueError(
                    f"No SDF function provided and {type(geometry).__name__} "
                    f"does not implement ImplicitGeometry or have signed_distance/sdf method"
                )

    return function_gradient(sdf_func, points, eps=eps)


if __name__ == "__main__":
    """Smoke tests for function_gradient module."""
    print("=" * 60)
    print("Testing function_gradient module")
    print("=" * 60)

    # Test 1: Single point gradient
    print("\n[Test 1: Single Point Gradient]")

    def sphere_sdf(x: NDArray) -> float | NDArray:
        """SDF for unit sphere: phi = |x| - 1"""
        x = np.asarray(x)
        if x.ndim == 1:
            return float(np.linalg.norm(x) - 1.0)
        return np.linalg.norm(x, axis=1) - 1.0

    point = np.array([2.0, 0.0, 0.0])
    grad = function_gradient(sphere_sdf, point)
    expected = np.array([1.0, 0.0, 0.0])  # Radial outward
    error = np.linalg.norm(grad - expected)
    print(f"  Point: {point}")
    print(f"  Gradient: {grad}")
    print(f"  Expected: {expected}")
    print(f"  Error: {error:.2e}")
    assert error < 1e-6, f"Single point error too large: {error}"
    print("  PASS")

    # Test 2: Batch gradient
    print("\n[Test 2: Batch Gradient]")

    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    grads = function_gradient(sphere_sdf, points)
    print(f"  Points shape: {points.shape}")
    print(f"  Grads shape: {grads.shape}")

    # Check each gradient points radially outward
    for i, (p, g) in enumerate(zip(points, grads, strict=True)):
        expected_dir = p / np.linalg.norm(p)
        g_normalized = g / np.linalg.norm(g)
        dot = np.dot(g_normalized, expected_dir)
        print(f"  Point {i}: dot(grad, expected) = {dot:.6f}")
        assert dot > 0.999, f"Gradient not pointing outward at point {i}"
    print("  PASS: All gradients point radially outward")

    # Test 3: Outward normal
    print("\n[Test 3: Outward Normal from SDF]")

    def radial_sdf(x: tuple[NDArray, ...]) -> NDArray:
        """The same phi = |x| - 1, as an sdf reads it: x a coordinate tuple."""
        return np.sqrt(sum(xi**2 for xi in x)) - 1.0

    normals = outward_normal_from_sdf(radial_sdf, points)
    print(f"  Normals shape: {normals.shape}")

    # Check unit length
    norms = np.linalg.norm(normals, axis=1)
    print(f"  Normal magnitudes: {norms}")
    assert np.allclose(norms, 1.0), "Normals not unit length"
    print("  PASS: All normals are unit vectors")

    # Test 4: 2D circle
    print("\n[Test 4: 2D Circle SDF]")

    # Points on the circle
    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    circle_points = np.stack([np.cos(theta), np.sin(theta)], axis=1)

    normals_2d = outward_normal_from_sdf(radial_sdf, circle_points)

    # Normals should equal the points (for unit circle)
    error_2d = np.max(np.abs(normals_2d - circle_points))
    print(f"  Circle points shape: {circle_points.shape}")
    print(f"  Max error |n - p|: {error_2d:.2e}")
    assert error_2d < 1e-5, f"2D circle normal error too large: {error_2d}"
    print("  PASS: 2D circle normals correct")

    # Test 5: Adaptive epsilon
    print("\n[Test 5: Adaptive Epsilon]")

    # Test at large coordinates
    large_point = np.array([1000.0, 0.0, 0.0])
    grad_fixed = function_gradient(sphere_sdf, large_point, eps=1e-6, adaptive_eps=False)
    grad_adaptive = function_gradient(sphere_sdf, large_point, eps=1e-6, adaptive_eps=True)

    print(f"  Large point: {large_point}")
    print(f"  Fixed eps gradient: {grad_fixed}")
    print(f"  Adaptive eps gradient: {grad_adaptive}")
    print("  Expected: [1, 0, 0]")

    # Both should work, adaptive may be more stable
    assert np.allclose(grad_fixed[:1], [1.0], atol=1e-3), "Fixed eps failed at large coord"
    assert np.allclose(grad_adaptive[:1], [1.0], atol=1e-5), "Adaptive eps failed"
    print("  PASS: Adaptive epsilon works")

    # Test 6: Quadratic function (exact gradient)
    print("\n[Test 6: Quadratic Function (Exact)]")

    def quadratic(x: NDArray) -> float | NDArray:
        """f(x,y) = x^2 + 2*y^2, grad = (2x, 4y)"""
        x = np.asarray(x)
        if x.ndim == 1:
            return float(x[0] ** 2 + 2 * x[1] ** 2)
        return x[:, 0] ** 2 + 2 * x[:, 1] ** 2

    test_points = np.array([[1.0, 1.0], [2.0, 3.0], [-1.0, 2.0]])
    grads_quad = function_gradient(quadratic, test_points)
    expected_grads = np.stack([2 * test_points[:, 0], 4 * test_points[:, 1]], axis=1)

    error_quad = np.max(np.abs(grads_quad - expected_grads))
    print(f"  Max error: {error_quad:.2e}")
    assert error_quad < 1e-8, f"Quadratic gradient error: {error_quad}"
    print("  PASS: Quadratic function gradient exact")

    # Test 7: Error handling
    print("\n[Test 7: Error Handling]")

    def bad_sdf(x):
        return np.inf

    try:
        function_gradient(bad_sdf, np.array([1.0, 0.0]))
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        print(f"  Correctly raised: {e}")
    print("  PASS: Error handling works")

    print("\n" + "=" * 60)
    print("All function_gradient tests passed!")
    print("=" * 60)

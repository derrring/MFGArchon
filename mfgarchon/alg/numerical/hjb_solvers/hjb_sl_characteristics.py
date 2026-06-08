"""
Characteristic Tracing Methods for Semi-Lagrangian HJB Solver.

This module provides methods for tracing characteristics backward in time,
which is the core operation in the semi-Lagrangian scheme.

The characteristic equation is:
    dX/dt = -∇_p H(X, P, m)

For standard MFG with quadratic Hamiltonian H = |p|²/2 + f(m):
    dX/dt = -P, so X(t-dt) = X(t) - P*dt

Supported integration methods:
- explicit_euler: First-order Euler method
- rk2: Second-order midpoint method
- rk4: Fourth-order Runge-Kutta (via scipy.solve_ivp)

Module structure per issue #392:
    hjb_sl_characteristics.py - Characteristic tracing for semi-Lagrangian solver

Functions:
    trace_characteristic_backward_1d: 1D characteristic tracing
    trace_characteristic_backward_nd: nD characteristic tracing
    apply_boundary_conditions_1d: 1D boundary condition handling
    apply_boundary_conditions_nd: nD boundary condition handling
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def trace_characteristic_backward_1d(
    x_current: float,
    p_optimal: float,
    dt: float,
    method: str = "explicit_euler",
    use_jax: bool = False,
    jax_solve_fn: object | None = None,
    ode_rtol: float = 1e-6,
    ode_atol: float = 1e-8,
) -> float:
    """
    Trace characteristic backward in time for 1D problems.

    The characteristic equation is dx/dt = -p.
    Computes X(t-dt) given X(t) and optimal control p.

    Args:
        x_current: Current spatial position (scalar)
        p_optimal: Optimal control value (scalar)
        dt: Time step size
        method: Integration method ('explicit_euler', 'rk2', 'rk4')
        use_jax: Whether to use JAX acceleration
        jax_solve_fn: JAX solve function if available
        ode_rtol: Relative tolerance for solve_ivp (rk4 method). Default 1e-6.
        ode_atol: Absolute tolerance for solve_ivp (rk4 method). Default 1e-8.

    Returns:
        Departure point X(t-dt)
    """
    x_scalar = float(x_current) if np.ndim(x_current) > 0 else x_current
    p_scalar = float(p_optimal) if np.ndim(p_optimal) > 0 else p_optimal

    # JAX acceleration path
    if use_jax and jax_solve_fn is not None:
        return float(jax_solve_fn(x_scalar, p_scalar, dt))

    if method == "explicit_euler":
        # First-order: X(t-dt) = X(t) - p*dt
        x_departure = x_scalar - p_scalar * dt

    elif method == "rk2":
        # Second-order Runge-Kutta (midpoint method)
        # For characteristic dx/dt = -p(x), this reduces to
        # X(t-dt) = X(t) - p(X(t))*dt for constant velocity field
        # k1 = -p_scalar
        # x_mid = x_scalar + 0.5 * dt * k1  # Could use for spatially varying p
        # For now, use same velocity (assumes locally constant field)
        x_departure = x_scalar - p_scalar * dt

    elif method == "rk4":
        # Fourth-order Runge-Kutta using scipy.solve_ivp
        # More robust than hand-coded RK4, uses adaptive stepping
        def velocity_field(t, x):
            # Characteristic equation: dx/dt = -p (constant velocity assumption)
            return -p_scalar

        sol = solve_ivp(
            velocity_field,
            t_span=[0, dt],
            y0=[x_scalar],
            method="RK45",
            rtol=ode_rtol,
            atol=ode_atol,
        )
        x_departure = sol.y[0, -1]

    else:
        # Default to explicit Euler
        x_departure = x_scalar - p_scalar * dt

    return x_departure


def trace_characteristic_backward_nd(
    x_current: np.ndarray,
    p_optimal: np.ndarray,
    dt: float,
    dimension: int,
    method: str = "explicit_euler",
    ode_rtol: float = 1e-6,
    ode_atol: float = 1e-8,
) -> np.ndarray:
    """
    Trace characteristic backward in time for nD problems.

    The characteristic equation is dX/dt = -P.
    Computes X(t-dt) given X(t) and optimal control P.

    Args:
        x_current: Current spatial position, shape (dimension,)
        p_optimal: Optimal control value, shape (dimension,)
        dt: Time step size
        dimension: Spatial dimension
        method: Integration method ('explicit_euler', 'rk2', 'rk4')
        ode_rtol: Relative tolerance for solve_ivp (rk4 method). Default 1e-6.
        ode_atol: Absolute tolerance for solve_ivp (rk4 method). Default 1e-8.

    Returns:
        Departure point X(t-dt), shape (dimension,)
    """
    x_vec = np.atleast_1d(x_current)
    p_vec = np.atleast_1d(p_optimal)

    if len(x_vec) != dimension or len(p_vec) != dimension:
        raise ValueError(f"x_current and p_optimal must have {dimension} components, got {len(x_vec)} and {len(p_vec)}")

    if method == "explicit_euler":
        # Vector Euler: X(t-dt) = X(t) - P*dt
        x_departure = x_vec - p_vec * dt

    elif method == "rk2":
        # Vector RK2 (midpoint method)
        # k1 = -p_vec
        # x_mid = x_vec + 0.5 * dt * k1  # Could interpolate velocity here
        # For now, use same velocity (assumes locally constant field)
        x_departure = x_vec - p_vec * dt

    elif method == "rk4":
        # Vector RK4 using scipy.solve_ivp
        def velocity_field(t, x):
            # Characteristic equation: dX/dt = -P (constant velocity assumption)
            return -p_vec

        sol = solve_ivp(
            velocity_field,
            t_span=[0, dt],
            y0=x_vec,
            method="RK45",
            rtol=ode_rtol,
            atol=ode_atol,
        )
        x_departure = sol.y[:, -1]

    else:
        # Default to explicit Euler
        x_departure = x_vec - p_vec * dt

    return x_departure


def apply_boundary_conditions_1d(
    x: float,
    xmin: float,
    xmax: float,
    bc_type: str | None = None,
    max_reflections: int = 10,
) -> float:
    """
    Apply boundary conditions to ensure x is in valid domain (1D).

    Issue #702: Support all BC types for adjoint consistency between HJB-SL and FP-SL.

    Args:
        x: Position to constrain
        xmin: Domain minimum
        xmax: Domain maximum
        bc_type: Boundary condition type:
            - 'periodic': Wrap around domain
            - 'reflect' / 'no_flux' / 'neumann': Mirror about boundary (default for MFG)
            - 'clamp' / 'dirichlet' / None: Clamp to boundary
        max_reflections: Maximum number of boundary reflections for reflect/neumann BC.
            Large displacements may require multiple reflections. Default 10.

    Returns:
        Position within domain bounds
    """
    if bc_type == "periodic":
        length = xmax - xmin
        while x < xmin:
            x += length
        while x > xmax:
            x -= length
        return x

    if bc_type in ("reflect", "no_flux", "neumann"):
        # Mirror reflection about boundaries
        # Handles multiple reflections for large displacements
        for _ in range(max_reflections):
            if x < xmin:
                x = 2 * xmin - x
            elif x > xmax:
                x = 2 * xmax - x
            else:
                break
        # Safety clamp after reflections
        return float(np.clip(x, xmin, xmax))

    # Default: clamp to domain (dirichlet / absorbing)
    return float(np.clip(x, xmin, xmax))


def reflect_into_domain(
    x: np.ndarray,
    xmin: float | np.ndarray,
    xmax: float | np.ndarray,
) -> np.ndarray:
    r"""
    Vectorized mirror-reflection of points into ``[xmin, xmax]`` (no-flux / Neumann fold).

    Closed-form triangle wave, equivalent to the iterated scalar reflection in
    :func:`apply_boundary_conditions_1d` (apply ``2*xmin - x`` / ``2*xmax - x`` until in
    bounds):

    .. math::
        x' = x_{\min} + L - \bigl|\,((x - x_{\min}) \bmod 2L) - L\,\bigr|,
        \qquad L = x_{\max} - x_{\min}.

    Identity for in-bounds ``x``; resolves arbitrarily many bounces in one pass. ``xmin`` /
    ``xmax`` may be scalars (1D) or per-axis arrays broadcastable against ``x`` (nD).

    The leading ``L -`` is load-bearing: the variant ``xmin + |((x - xmin) mod 2L) - L|``
    (without it) is a point-inversion about the domain *center* (``x -> xmin + xmax - x``),
    not a boundary reflection — it displaces even in-bounds points and is correct only on
    domains/data symmetric about the midpoint.
    """
    span = xmax - xmin
    return xmin + span - np.abs(((x - xmin) % (2.0 * span)) - span)


def apply_boundary_conditions_nd(
    x: np.ndarray,
    bounds: list[tuple[float, float]],
    bc_type: str | None = None,
    max_reflections: int = 10,
) -> np.ndarray:
    """
    Apply boundary conditions to ensure x is in valid domain (nD).

    Issue #702: Support all BC types for adjoint consistency between HJB-SL and FP-SL.

    Args:
        x: Position vector to constrain, shape (dimension,)
        bounds: List of (min, max) tuples for each dimension
        bc_type: Boundary condition type:
            - 'periodic': Wrap around domain
            - 'reflect' / 'no_flux' / 'neumann': Mirror about boundary (default for MFG)
            - 'clamp' / 'dirichlet' / None: Clamp to boundary
        max_reflections: Maximum number of boundary reflections per dimension
            for reflect/neumann BC. Default 10.

    Returns:
        Position within domain bounds, shape (dimension,)
    """
    x_bounded = x.copy()
    dimension = len(bounds)

    for d in range(dimension):
        xmin, xmax = bounds[d]
        if bc_type == "periodic":
            length = xmax - xmin
            while x_bounded[d] < xmin:
                x_bounded[d] += length
            while x_bounded[d] > xmax:
                x_bounded[d] -= length
        elif bc_type in ("reflect", "no_flux", "neumann"):
            # Mirror reflection about boundaries
            for _ in range(max_reflections):
                if x_bounded[d] < xmin:
                    x_bounded[d] = 2 * xmin - x_bounded[d]
                elif x_bounded[d] > xmax:
                    x_bounded[d] = 2 * xmax - x_bounded[d]
                else:
                    break
            x_bounded[d] = np.clip(x_bounded[d], xmin, xmax)
        else:
            # Default: clamp (dirichlet / absorbing)
            x_bounded[d] = np.clip(x_bounded[d], xmin, xmax)

    return x_bounded


# =============================================================================
# Smoke Tests
# =============================================================================

if __name__ == "__main__":
    """Smoke test for characteristic tracing methods."""
    print("Testing characteristic tracing methods...")

    # Test 1: 1D explicit Euler
    print("\n1. Testing 1D explicit Euler...")
    x0 = 0.5
    p0 = 0.1
    dt = 0.01

    x_departure = trace_characteristic_backward_1d(x0, p0, dt, method="explicit_euler")
    expected = x0 - p0 * dt
    assert abs(x_departure - expected) < 1e-10
    print(f"   x0={x0}, p={p0}, dt={dt} -> x_departure={x_departure:.6f}")
    print("   1D explicit Euler: OK")

    # Test 2: 1D RK2
    print("\n2. Testing 1D RK2...")
    x_departure_rk2 = trace_characteristic_backward_1d(x0, p0, dt, method="rk2")
    # For constant velocity field, RK2 should equal Euler
    assert abs(x_departure_rk2 - expected) < 1e-10
    print(f"   RK2 result: {x_departure_rk2:.6f}")
    print("   1D RK2: OK")

    # Test 3: 1D RK4
    print("\n3. Testing 1D RK4...")
    x_departure_rk4 = trace_characteristic_backward_1d(x0, p0, dt, method="rk4")
    # For constant velocity field, RK4 should be close to Euler
    assert abs(x_departure_rk4 - expected) < 1e-6
    print(f"   RK4 result: {x_departure_rk4:.6f}")
    print("   1D RK4: OK")

    # Test 4: 2D explicit Euler
    print("\n4. Testing 2D explicit Euler...")
    x0_2d = np.array([0.5, 0.5])
    p0_2d = np.array([0.1, 0.2])

    x_departure_2d = trace_characteristic_backward_nd(x0_2d, p0_2d, dt, dimension=2, method="explicit_euler")
    expected_2d = x0_2d - p0_2d * dt
    assert np.allclose(x_departure_2d, expected_2d)
    print(f"   x0={x0_2d}, p={p0_2d}")
    print(f"   x_departure={x_departure_2d}")
    print("   2D explicit Euler: OK")

    # Test 5: 3D RK4
    print("\n5. Testing 3D RK4...")
    x0_3d = np.array([0.5, 0.5, 0.5])
    p0_3d = np.array([0.1, 0.2, 0.15])

    x_departure_3d = trace_characteristic_backward_nd(x0_3d, p0_3d, dt, dimension=3, method="rk4")
    expected_3d = x0_3d - p0_3d * dt
    assert np.allclose(x_departure_3d, expected_3d, rtol=1e-5)
    print(f"   x_departure={x_departure_3d}")
    print("   3D RK4: OK")

    # Test 6: 1D boundary conditions (clamping)
    print("\n6. Testing 1D boundary conditions (clamping)...")
    x_out = 1.5  # Outside [0, 1]
    x_clamped = apply_boundary_conditions_1d(x_out, xmin=0.0, xmax=1.0)
    assert x_clamped == 1.0
    print(f"   x={x_out} -> clamped to {x_clamped}")
    print("   1D clamping: OK")

    # Test 7: 1D periodic boundary conditions
    print("\n7. Testing 1D periodic boundary conditions...")
    x_out_periodic = 1.3
    x_periodic = apply_boundary_conditions_1d(x_out_periodic, xmin=0.0, xmax=1.0, bc_type="periodic")
    assert abs(x_periodic - 0.3) < 1e-10
    print(f"   x={x_out_periodic} -> periodic to {x_periodic}")
    print("   1D periodic: OK")

    # Test 8: 2D boundary conditions
    print("\n8. Testing 2D boundary conditions...")
    x_out_2d = np.array([1.2, -0.1])
    bounds_2d = [(0.0, 1.0), (0.0, 1.0)]
    x_bounded_2d = apply_boundary_conditions_nd(x_out_2d, bounds_2d)
    assert np.allclose(x_bounded_2d, np.array([1.0, 0.0]))
    print(f"   x={x_out_2d} -> bounded to {x_bounded_2d}")
    print("   2D boundary conditions: OK")

    print("\nAll characteristic tracing smoke tests passed!")

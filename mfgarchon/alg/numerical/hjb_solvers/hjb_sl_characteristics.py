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

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.integrate import solve_ivp

from mfgarchon.geometry.boundary.bc_utils import get_bc_type_string

if TYPE_CHECKING:
    from collections.abc import Sequence

# How finely a semi-Lagrangian step is cut when its CFL number exceeds max(1, cfl_target). Both halves of the SL pair
# (HJBSemiLagrangianSolver and FPSLSolver) share this function; each measures the CFL number on its own
# velocity and time level (#1880).
DEFAULT_CFL_TARGET = 0.9
DEFAULT_MAX_SUBSTEPS = 100


def _refuse_meaningless_substep_settings(cfl_target: float, max_substeps: int) -> None:
    """Refuse a sub-step schedule that means nothing: no positive, finite target, or no cap of at least 1."""
    if not 0 < cfl_target < np.inf:  # also refuses NaN
        raise ValueError(f"cfl_target must be positive and finite, got {cfl_target}")
    if not max_substeps >= 1:  # also refuses NaN, which would switch the cap off
        raise ValueError(f"max_substeps must be at least 1, got {max_substeps}")


def check_substep_settings(cfl_target: float, max_substeps: int) -> None:
    """Check a half's sub-step schedule when it is built: refuse one that means nothing, warn above 1.

    ``cfl_target`` is the *planned* crossing: the cells a sub-step's foot would cross at the velocity
    measured at the step's start. The HJB half's traced feet run past it, since the gradient steepens
    within the step (#2439); the FP half holds its velocity fixed within a step and does not. A step
    is not split at all while its CFL number is at most ``max(1, cfl_target)``, so an unsplit foot can
    cross that many cells: one at the default, ``cfl_target`` above 1.

    Above 1 the target is accepted with a warning, not refused (#2458): where the pair becomes
    unstable is a measurement on one fixture, not a property of the scheme, and the growth measured
    above 1 changed with the grid (#2448: a 2-D HJB half at 1.5 read 1.152 against 1-D's 2.190). On
    #1880's matrix fixture (sigma = 0.2), with the HJB half at c and the FP half at its default, a
    seeded asymmetry changes per sweep by 0.984 at c = 1.0, 1.005 at 1.02,
    0.994 at 1.05, 1.121 at 1.1 and 2.190 at 1.5 (#2448). An FP half at up to 1.5 beside a default
    HJB half was stable; at 50 either half behaves as if it did not sub-step at all, which the pair's
    duality check does not see, since it compares the declared switch (#2443).
    """
    _refuse_meaningless_substep_settings(cfl_target, max_substeps)
    if cfl_target > 1:
        warnings.warn(
            f"cfl_target={cfl_target} is above 1: a sub-step's foot may be planned to cross up to "
            f"{cfl_target} cells, on a half that sub-steps, and a step whose CFL number is at most "
            f"{cfl_target} is not split at all. It is accepted, but measure it on your problem. On "
            f"#1880's matrix fixture (sigma = 0.2) the HJB "
            f"half grew a seeded asymmetry 1.005x per sweep at 1.02, 1.121x at 1.1 and 2.190x at 1.5, "
            f"and an FP half at up to 1.5 beside a default HJB half was stable. At 50 either half "
            f"behaved as if it did not sub-step, which check_solver_duality does not detect. The "
            f"default is 0.9 (#2448, #2458).",
            UserWarning,
            stacklevel=3,
        )


def cfl_substeps(
    cfl: float, *, cfl_target: float = DEFAULT_CFL_TARGET, max_substeps: int = DEFAULT_MAX_SUBSTEPS
) -> int:
    """Sub-steps for one semi-Lagrangian step of CFL number ``cfl``.

    One when ``cfl <= 1``; otherwise ``ceil(cfl / cfl_target)``. A step that needs more than
    ``max_substeps`` is refused rather than capped (#2438). On the #1880 fixture at 41 points, a
    capped HJB step grew U smoothly while its sub-step CFL was 1.6-1.9. From 2.07 on, U grew faster
    than in that smooth phase, 2.2 to 267 over ten sub-steps, then roughly squared per sub-step, and the solve
    ended in NaN. With room for the 195 sub-steps it needed, it stayed finite and symmetric. The HJB
    half's pointwise update takes one foot from the node's own gradient; for the quadratic control
    cost that can only overestimate the Lax-Oleinik infimum, and does once the foot is cells away.

    Both halves measure the CFL number on the velocity their foot moves at: the HJB half on dH/dp,
    the FP half on its drift. The HJB half used max|grad u| until #2439, which is that speed only at
    lambda = 1.

    The two halves of the SL pair must structure a step alike: sub-stepped in both, or in neither
    (#1880). When the HJB half sub-stepped and the FP half made one forward splat of ~10 cells, the
    coupled Picard map amplified an antisymmetric perturbation ~3.3x per sweep on a symmetric fixture,
    where upwind FD damps it at 0.89. The mirror mismatch -- FP sub-stepping while HJB does not --
    destabilises the pair as well.
    """
    _refuse_meaningless_substep_settings(cfl_target, max_substeps)
    if not np.isfinite(cfl):
        raise ValueError(f"cfl_substeps got a non-finite CFL number ({cfl}): the velocity field is not finite")
    if cfl <= 1.0:
        return 1
    needed = int(np.ceil(cfl / cfl_target))
    if needed > max_substeps:
        raise ValueError(
            f"A semi-Lagrangian step of CFL number {cfl:.2f} needs {needed} sub-steps at "
            f"cfl_target={cfl_target}, more than max_substeps={max_substeps} (capped, each sub-step "
            f"would have CFL number {cfl / max_substeps:.2f}). Neither half caps: a capped HJB step can "
            f"run away (#2438), and the FP half must sub-step as its HJB half does (#1880). Refine dt "
            f"(a larger Nt, with the solvers rebuilt for it, #2446), or raise max_substeps to at least "
            f"{needed}; a later Picard sweep can need more. In Expert Mode, create_paired_solvers(..., "
            f"hjb_config={{'max_substeps': n}}) hands the value to both halves."
        )
    return needed


def wall_gradient_from_values(
    values: np.ndarray, gradient: Sequence[np.ndarray], spacings: Sequence[float]
) -> tuple[np.ndarray, ...]:
    """``gradient`` with every wall slice replaced by the second-order one-sided difference of ``values``.

    Both halves of the SL pair differentiate U with the geometry's central operator, whose value at a wall
    node is read from the FDM ghost: the control there was whatever the ghost encoded, not a property of
    U (#2467). The cell-centred no-flux ghost gives half the one-sided slope; the node-centred mirror #1935
    needs gives the boundary datum whatever U is, which freezes the wall characteristic.

    Measured on `test_hjb_semi_lagrangian.py`'s sigma = 0 fixture, whose exact solution is the Hopf-Lax
    closed form (x - 1/2)^2 / (2 (1 + tau)) + tau with wall slope -1 / (2 (1 + tau)): at 201 points the max
    error is 5.3e-04 from the ghost's half slope, wall/interior 1.60, and 3.3e-04 from this, wall/interior
    1.007. A gradient forced to the datum gives 7.7e-03.

    Interior values are returned unchanged. The caller decides when this applies: never on a periodic axis,
    whose wrap is the right neighbour.
    """
    closed = []
    for axis, (component, h) in enumerate(zip(gradient, spacings, strict=True)):
        if values.shape[axis] < 3:
            raise ValueError(
                f"the one-sided wall gradient needs 3 points along axis {axis}; this grid has {values.shape[axis]}"
            )
        out = np.array(component, dtype=float)
        low = [slice(None)] * values.ndim
        high = [slice(None)] * values.ndim
        low[axis], high[axis] = 0, -1

        def at(i: int, ax: int = axis) -> np.ndarray:
            return np.take(values, i, axis=ax)

        out[tuple(low)] = (-3.0 * at(0) + 4.0 * at(1) - at(2)) / (2.0 * h)
        out[tuple(high)] = (3.0 * at(-1) - 4.0 * at(-2) + at(-3)) / (2.0 * h)
        closed.append(out)
    return tuple(closed)


def value_gradient(geometry: Any, values: np.ndarray, time: float) -> tuple[np.ndarray, ...]:
    """grad(U) as both halves of the SL pair take it: the HJB half for its control, the FP half for its drift.

    The geometry's central operator, whose ghosts at ``time`` carry the geometry's boundary conditions; on a
    domain those conditions do not make periodic, every wall node then takes the one-sided difference of
    ``values`` (`wall_gradient_from_values`, #2467). Periodicity is read from the same conditions the ghosts
    use, so the two cannot disagree about which neighbour a wall node has.
    """
    components = tuple(op(values) for op in geometry.get_gradient_operator(scheme="central", time=time))
    if get_bc_type_string(geometry.get_boundary_conditions()) == "periodic":
        return components
    return wall_gradient_from_values(values, components, geometry.get_grid_spacing())


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


#: The geometric operations :func:`bc_type_to_geometric_operation` can return. Named here so
#: :func:`fold_into_domain` can refuse anything else instead of folding it silently.
_FOLD_OPERATIONS = frozenset({"reflect", "periodic", "clamp"})


def fold_into_domain(
    x: np.ndarray,
    xmin: float | np.ndarray,
    xmax: float | np.ndarray,
    bc_op: str,
) -> np.ndarray:
    r"""Fold departure points into ``[xmin, xmax]`` under the geometric operation ``bc_op``.

    The one owner of the vectorized Semi-Lagrangian boundary fold. ``bc_op`` must come from
    :func:`~mfgarchon.geometry.boundary.bc_utils.bc_type_to_geometric_operation`; this function
    dispatches on that vocabulary and no other.

    ``xmin`` / ``xmax`` are scalars (1D) or per-axis arrays broadcastable against ``x`` (nD).

    Raises:
        ValueError: for any other operation. Do not add a fall-through -- an unrecognised
            spelling would then silently pick a boundary condition instead of stopping.
    """
    if bc_op == "reflect":
        return reflect_into_domain(x, xmin, xmax)
    if bc_op == "periodic":
        return xmin + (x - xmin) % (xmax - xmin)
    if bc_op == "clamp":
        return np.clip(x, xmin, xmax)
    raise ValueError(
        f"unknown geometric boundary operation {bc_op!r}; expected one of "
        f"{sorted(_FOLD_OPERATIONS)}. This fold dispatches on the vocabulary of "
        "bc_type_to_geometric_operation (Issue #1739)."
    )


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

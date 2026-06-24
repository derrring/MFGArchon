"""
Drift Field Helper Functions for Fokker-Planck Solvers.

This module provides convenient helper functions for creating drift fields
compatible with FP solver APIs. These helpers make it explicit what type of
drift is being used without requiring a full Strategy Pattern hierarchy.

The Fokker-Planck equation is:
    ∂m/∂t + ∇·(α m) = ∇·(D ∇m)

where α(t,x,m) is the drift field. This module helps construct α for various
physical scenarios.

IMPORTANT - Current Implementation Status:
    The BaseFPSolver API supports drift_field as None, np.ndarray, or Callable.
    However, as of v0.13, concrete FP solvers (FPFDMSolver, FPParticleSolver, etc.)
    only support None and np.ndarray. Callable support is planned for Phase 2.

    Current Usage:
        # Zero drift (works now)
        M = fp_solver.solve_fp_system(m0, drift_field=None)

        # Precomputed array drift (works now)
        drift_array = compute_drift_field(...)  # shape (Nt, Nx) or (Nt, Nx, d)
        M = fp_solver.solve_fp_system(m0, drift_field=drift_array)

    Future Usage (Phase 2):
        # Callable drift (coming soon)
        M = fp_solver.solve_fp_system(m0, drift_field=zero_drift())
        M = fp_solver.solve_fp_system(m0, drift_field=optimal_control_drift(U, problem))

    These helper functions are provided now to:
    1. Document intended usage patterns for future API
    2. Enable early adoption when callable support lands
    3. Make code intent explicit even when using precomputed arrays

Usage (future API, Phase 2):
    from mfgarchon.utils.drift_helpers import (
        zero_drift,
        optimal_control_drift,
        prescribed_drift,
        composite_drift,
    )

    # Pure diffusion (heat equation)
    M = fp_solver.solve_fp_system(m0, drift_field=zero_drift())

    # MFG optimal control
    M = fp_solver.solve_fp_system(m0, drift_field=optimal_control_drift(U, problem))

    # Prescribed wind
    M = fp_solver.solve_fp_system(m0, drift_field=prescribed_drift(wind_func))

    # Multiple sources
    M = fp_solver.solve_fp_system(
        m0,
        drift_field=composite_drift([
            optimal_control_drift(U, problem),
            prescribed_drift(lambda t, x, m: gravity(x)),
        ])
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mfgarchon.utils.deprecation import deprecated

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np

    from mfgarchon.core.mfg_problem import MFGProblem


def zero_drift() -> Callable:
    """
    Create zero drift field (pure diffusion, heat equation).

    Returns a callable that always returns zero drift, useful for solving
    the heat equation or pure diffusion without advection.

    Equation:
        ∂m/∂t = ∇·(D ∇m)  (no advection term)

    Returns:
        Callable with signature α(t, x, m) -> 0

    Example:
        >>> from mfgarchon.utils.drift_helpers import zero_drift
        >>> from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
        >>> solver = FPFDMSolver(problem)
        >>> M = solver.solve_fp_system(m0, drift_field=zero_drift())
    """

    def _zero_drift(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Zero drift field α = 0."""
        import numpy as np

        if m.ndim == 1:
            # 1D case
            return np.zeros_like(m)
        elif m.ndim == 2:
            # 2D case: return (Nx, Ny, 2) for vector field
            return np.zeros((*m.shape, 2))
        else:
            # 3D case: return (Nx, Ny, Nz, 3) for vector field
            return np.zeros((*m.shape, 3))

    return _zero_drift


@deprecated(
    since="v0.19.0",
    replacement="use H.optimal_control(x, m, grad_U, t) directly, or let FixedPointIterator handle it automatically",
    reason="Hardcodes quadratic Hamiltonian assumption (Issue #896)",
)
def optimal_control_drift(
    U: np.ndarray,
    problem: MFGProblem,
    control_cost: float | None = None,
) -> Callable:
    """
    Create MFG optimal control drift field α = -∇U / λ.

    .. deprecated:: 0.19.0
        This function hardcodes the quadratic Hamiltonian assumption (α = -∇U/λ).
        For general Hamiltonians, use the inline pipeline instead::

            grad_U = np.gradient(U, dx, axis=-1)
            alpha = H.optimal_control(x, m, grad_U, t)

        The FixedPointIterator handles this automatically via
        ``fixed_point_utils.resolve_fp_drift_kwargs`` /
        ``fixed_point_utils.compute_fp_velocity_field``. Will be removed in v0.25.0.

    Equation:
        α(t,x) = -∇U(t,x) / λ

    where:
        U: HJB value function
        λ: control cost parameter (default: problem.control_cost)

    Args:
        U: HJB value function array, shape (Nt, Nx) or (Nt, Nx, Ny) etc.
        problem: MFG problem instance (for gradient computation)
        control_cost: Control cost λ (default: use problem.control_cost)

    Returns:
        Callable with signature α(t, x, m) -> -∇U / λ

    Example:
        >>> from mfgarchon.utils.drift_helpers import optimal_control_drift
        >>> drift = optimal_control_drift(U_hjb, problem)
        >>> M = fp_solver.solve_fp_system(m0, drift_field=drift)

    Note:
        This helper precomputes the full drift field -∇U / λ for efficiency.
        The returned callable simply indexes into the precomputed array.
    """
    import numpy as np

    # Get control cost
    lambda_cost = control_cost if control_cost is not None else getattr(problem, "control_cost", 1.0)

    # Precompute gradient
    if hasattr(problem, "compute_gradient"):
        # Use problem's gradient method (handles boundary conditions)
        grad_U = problem.compute_gradient(U)
    else:
        # Fallback: simple central difference
        grid_spacing = problem.geometry.get_grid_spacing()
        if U.ndim == 2:
            # 1D spatial: U is (Nt, Nx_points)
            grad_U = np.gradient(U, grid_spacing[0], axis=1)
        elif U.ndim == 3:
            # 2D spatial: U is (Nt, Nx_points, Ny_points)
            grad_U_x = np.gradient(U, grid_spacing[0], axis=1)
            grad_U_y = np.gradient(U, grid_spacing[1], axis=2)
            grad_U = np.stack([grad_U_x, grad_U_y], axis=-1)
        else:
            raise NotImplementedError(f"Gradient for {U.ndim}D not implemented")

    # Compute drift: α = -∇U / λ
    alpha_field = -grad_U / lambda_cost

    def _optimal_control_drift(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        """MFG optimal control drift α = -∇U / λ."""
        # Find time index (assuming uniform time grid)
        Nt = U.shape[0]
        T = problem.T
        dt = T / (Nt - 1)
        t_idx = min(int(t / dt), Nt - 1)

        return alpha_field[t_idx]

    return _optimal_control_drift


def prescribed_drift(drift_func: Callable) -> Callable:
    """
    Create drift from prescribed external field (wind, currents, etc.).

    Wraps a user-provided drift function for use with FP solvers.
    Useful for modeling external forces like wind, ocean currents,
    or gravitational fields.

    Args:
        drift_func: User function α(t, x) or α(t, x, m)
            - For time-space dependent: α(t, x) -> np.ndarray
            - For state-dependent: α(t, x, m) -> np.ndarray

    Returns:
        Callable with signature α(t, x, m) compatible with FP solvers

    Examples:
        >>> # Constant wind
        >>> wind = lambda t, x, m: np.array([1.0, 0.5])
        >>> drift = prescribed_drift(wind)

        >>> # Time-varying wind
        >>> wind = lambda t, x, m: np.array([np.sin(2*np.pi*t), 0.0])
        >>> drift = prescribed_drift(wind)

        >>> # Spatially-varying field
        >>> def gravity(t, x, m):
        ...     x_arr, y_arr = x
        ...     return np.array([0.0, -9.8 * np.ones_like(x_arr)])
        >>> drift = prescribed_drift(gravity)
    """
    import inspect

    # Check if drift_func takes (t, x) or (t, x, m)
    sig = inspect.signature(drift_func)
    n_params = len(sig.parameters)

    if n_params == 2:
        # drift_func(t, x) -> wrap to add m parameter
        def _prescribed_drift(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
            return drift_func(t, x)

        return _prescribed_drift
    elif n_params == 3:
        # drift_func(t, x, m) -> use directly
        return drift_func
    else:
        raise ValueError(f"drift_func must take 2 or 3 parameters (t, x) or (t, x, m), got {n_params}")


def density_dependent_drift(drift_func: Callable) -> Callable:
    """
    Create state-dependent (nonlinear) drift α = α(t, x, m).

    For nonlinear PDEs where drift depends on density itself, such as:
    - Burgers equation: α = m (self-advection)
    - Traffic flow: α = V(m) (density-dependent velocity)
    - Crowd dynamics: α = -∇P(m) (pressure-based motion)

    Args:
        drift_func: Function α(t, x, m) -> np.ndarray

    Returns:
        Callable with signature α(t, x, m)

    Examples:
        >>> # Burgers equation (α = m)
        >>> burgers = lambda t, x, m: m
        >>> drift = density_dependent_drift(burgers)

        >>> # Traffic flow with fundamental diagram
        >>> def traffic_velocity(t, x, m):
        ...     rho = m  # density
        ...     v_max = 1.0
        ...     rho_max = 1.0
        ...     return v_max * (1 - rho / rho_max)
        >>> drift = density_dependent_drift(traffic_velocity)

        >>> # Pressure-driven crowd motion
        >>> def crowd_pressure(t, x, m):
        ...     # α = -∇P(m), P(m) = k*m^2 (quadratic pressure)
        ...     k = 0.5
        ...     grad_m_x = np.gradient(m, dx, axis=0)
        ...     grad_m_y = np.gradient(m, dy, axis=1)
        ...     return -k * np.stack([2*m*grad_m_x, 2*m*grad_m_y], axis=-1)
        >>> drift = density_dependent_drift(crowd_pressure)

    Note:
        Density-dependent drift makes the FP equation **nonlinear**, which
        may require smaller timesteps or implicit schemes for stability.
    """
    return drift_func  # Already has correct signature


def composite_drift(drift_list: Sequence[Callable]) -> Callable:
    """
    Combine multiple drift sources: α = α₁ + α₂ + ... + αₙ.

    For multi-physics problems with multiple drift mechanisms acting
    simultaneously, such as:
    - MFG optimal control + external wind
    - Optimal control + gravity
    - Self-advection + pressure gradient

    Args:
        drift_list: List of drift callables, each with signature α(t, x, m)

    Returns:
        Callable that sums all drift contributions

    Example:
        >>> from mfgarchon.utils.drift_helpers import (
        ...     optimal_control_drift,
        ...     prescribed_drift,
        ...     composite_drift,
        ... )
        >>>
        >>> # MFG control + constant wind
        >>> control = optimal_control_drift(U_hjb, problem)
        >>> wind = prescribed_drift(lambda t, x, m: np.array([1.0, 0.0]))
        >>> total_drift = composite_drift([control, wind])
        >>> M = fp_solver.solve_fp_system(m0, drift_field=total_drift)

    Note:
        All drift components are evaluated at the same (t, x, m) and summed.
        Ensure all components return arrays with compatible shapes.
    """

    def _composite_drift(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Sum of multiple drift contributions."""
        total = None
        for drift_fn in drift_list:
            contribution = drift_fn(t, x, m)
            if total is None:
                total = contribution
            else:
                total = total + contribution
        return total

    return _composite_drift


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing drift helper functions...")

    import numpy as np

    # Test zero_drift
    drift_zero = zero_drift()
    m_test_1d = np.array([0.1, 0.2, 0.3])
    alpha_zero = drift_zero(0.0, None, m_test_1d)
    assert np.allclose(alpha_zero, 0.0)
    print("  zero_drift() works")

    # Test prescribed_drift with 2-param function
    def wind_2param(t, x):
        return np.array([1.0, 0.5])

    drift_wind = prescribed_drift(wind_2param)
    alpha_wind = drift_wind(0.0, None, None)
    assert np.allclose(alpha_wind, [1.0, 0.5])
    print("  prescribed_drift() with (t, x) works")

    # Test prescribed_drift with 3-param function
    def wind_3param(t, x, m):
        return np.array([1.0, 0.5])

    drift_wind2 = prescribed_drift(wind_3param)
    alpha_wind2 = drift_wind2(0.0, None, None)
    assert np.allclose(alpha_wind2, [1.0, 0.5])
    print("  prescribed_drift() with (t, x, m) works")

    # Test density_dependent_drift
    def burgers(t, x, m):
        return m

    drift_burgers = density_dependent_drift(burgers)
    m_test = np.array([0.1, 0.2, 0.3])
    alpha_burgers = drift_burgers(0.0, None, m_test)
    assert np.allclose(alpha_burgers, m_test)
    print("  density_dependent_drift() works")

    # Test composite_drift
    def drift1(t, x, m):
        return np.array([1.0, 0.0])

    def drift2(t, x, m):
        return np.array([0.0, 1.0])

    drift_combined = composite_drift([drift1, drift2])
    alpha_combined = drift_combined(0.0, None, None)
    assert np.allclose(alpha_combined, [1.0, 1.0])
    print("  composite_drift() works")

    print("All smoke tests passed!")

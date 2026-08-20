"""
JAX utility functions for MFGarchon accelerated solvers.

This module provides common utilities for JAX-based implementations,
including numerical operations, device management, and optimization helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import DEFAULT_DEVICE, HAS_GPU, HAS_JAX

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfgarchon.types.solver_types import JAXArray
else:
    pass

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    from jax import device_put, grad, jacfwd, jacrev, jit, vmap
    from jax.lax import cond, scan

    # Through the single owner (#1923). This was an unconditional write AT MODULE IMPORT, so
    # whether a user's float32 backend survived depended on import order.
    from mfgarchon.utils.acceleration.jax_precision import require_x64

    require_x64("jax_utils (Newton's method needs float64 for tight tolerances)")

    # Optax is optional for optimization schedules
    try:
        import optax

        HAS_OPTAX = True
    except ImportError:
        optax = None
        HAS_OPTAX = False
else:
    # Dummy implementations for graceful fallback
    jax = None
    jnp = np  # type: ignore[misc]

    def device_put(x: Any, device: Any = None) -> Any:
        return x

    def grad(f: Any) -> Any:  # type: ignore[misc]
        return lambda x: x

    def jacfwd(f: Any) -> Any:  # type: ignore[misc]
        return lambda x: x

    def jacrev(f: Any) -> Any:
        return lambda x: x

    def jit(f: Any) -> Any:  # type: ignore[misc]
        return f  # Identity decorator when JAX not available

    def vmap(f: Any) -> Any:  # type: ignore[misc]
        return f

    def cond(pred: Any, true_fn: Any, false_fn: Any, operand: Any) -> Any:  # type: ignore[misc]
        return true_fn(operand) if pred else false_fn(operand)

    def scan(f: Any, init: Any, xs: Any) -> Any:  # type: ignore[misc]
        return (init, xs)

    optax = None
    HAS_OPTAX = False


def ensure_jax_available():
    """Ensure JAX is available, raise error if not."""
    if not HAS_JAX:
        raise ImportError("JAX is required for accelerated solvers. Install with: pip install jax jaxlib")


def to_device(array: np.ndarray | Any, device: Any | None = None) -> Any:
    """Move array to specified device."""
    ensure_jax_available()

    if device is None:
        device = DEFAULT_DEVICE

    return device_put(array, device)


def from_device(array: Any) -> np.ndarray:
    """Move array from device to numpy."""
    ensure_jax_available()

    if hasattr(array, "__array__"):
        return np.asarray(array)
    return array


def compute_jacobian(
    F: Callable,
    x: np.ndarray,
    original_shape: tuple | None = None,
    mode: str = "forward",
) -> np.ndarray:
    """
    Compute Jacobian of F at x using JAX automatic differentiation.

    This function provides O(1) Jacobian computation via reverse-mode AD,
    replacing the O(N) finite-difference approach.

    Args:
        F: Function F: R^n -> R^m (accepts array, returns array)
        x: Point at which to compute Jacobian (any shape)
        original_shape: Original shape of x (for reshaping)
        mode: 'forward' (jacfwd) or 'reverse' (jacrev)
            - forward: efficient when m >> n (outputs >> inputs)
            - reverse: efficient when n >> m (inputs >> outputs)

    Returns:
        Jacobian matrix J[i,j] = dF_i/dx_j as numpy array

    Example:
        >>> def F(x):
        ...     return x**2 - 2.0
        >>> J = compute_jacobian(F, np.array([1.5]))
        >>> print(J)  # [[3.0]]

    Note:
        For Newton's method solving F(x)=0 with N unknowns,
        this provides O(1) complexity vs O(N) for finite differences.

        The function F must be JAX-compatible (use jax.numpy operations).
        For numpy-based functions, the finite-difference fallback is used.
    """
    ensure_jax_available()

    if original_shape is None:
        original_shape = x.shape

    x_flat = x.flatten()

    # Wrap F to handle flat input/output for JAX
    # Important: Keep everything as JAX arrays inside the tracer
    def F_flat(x_in: JAXArray) -> JAXArray:
        x_reshaped = x_in.reshape(original_shape)
        # Call F with JAX array - F must be JAX-compatible
        result = F(x_reshaped)
        return jnp.asarray(result).flatten()

    # Convert input to JAX array
    x_jax = jnp.asarray(x_flat)

    # Compute Jacobian using JAX autodiff
    if mode == "forward":
        # Forward-mode AD: efficient when outputs >> inputs
        jac_fn = jacfwd(F_flat)
    else:
        # Reverse-mode AD: efficient when inputs >> outputs
        jac_fn = jacrev(F_flat)

    # Compute Jacobian matrix
    J = jac_fn(x_jax)

    # Convert back to numpy
    return np.asarray(J)


def compute_jacobian_jit(
    F: Callable,
    x: np.ndarray,
    original_shape: tuple | None = None,
    mode: str = "reverse",
    jit_cache: dict | None = None,
) -> np.ndarray:
    """
    Compute Jacobian with JIT compilation caching.

    For repeated calls with the same F, this caches the JIT-compiled
    Jacobian function for better performance.

    Args:
        F: Function F: R^n -> R^m
        x: Point at which to compute Jacobian
        original_shape: Original shape of x
        mode: 'forward' or 'reverse'
        jit_cache: Optional dict to store JIT-compiled functions

    Returns:
        Jacobian matrix as numpy array
    """
    ensure_jax_available()

    if original_shape is None:
        original_shape = x.shape

    x_flat = x.flatten()

    # Create cache key from function id and shape
    cache_key = (id(F), original_shape, mode)

    # Check if we have a cached JIT-compiled Jacobian
    if jit_cache is not None and cache_key in jit_cache:
        jac_fn_jit = jit_cache[cache_key]
    else:
        # Create wrapper function for flat arrays
        def F_flat(x_in: JAXArray) -> JAXArray:
            x_reshaped = x_in.reshape(original_shape)
            result = F(np.asarray(x_reshaped))
            return jnp.asarray(result).flatten()

        # Select AD mode
        if mode == "forward":
            jac_fn = jacfwd(F_flat)
        else:
            jac_fn = jacrev(F_flat)

        # JIT compile
        jac_fn_jit = jit(jac_fn)

        # Cache if provided
        if jit_cache is not None:
            jit_cache[cache_key] = jac_fn_jit

    # Compute Jacobian
    x_jax = jnp.asarray(x_flat)
    J = jac_fn_jit(x_jax)

    return np.asarray(J)


@jit
def finite_difference_1d(u: JAXArray, dx: float, order: int = 2) -> JAXArray:
    """
    Compute first derivative using finite differences.

    Args:
        u: 1D array to differentiate
        dx: Grid spacing
        order: Order of accuracy (2, 4, or 6)

    Returns:
        First derivative array
    """
    if order == 2:
        # Second-order central difference
        u_padded = jnp.pad(u, 1, mode="edge")
        return (u_padded[2:] - u_padded[:-2]) / (2 * dx)

    elif order == 4:
        # Fourth-order central difference
        u_padded = jnp.pad(u, 2, mode="edge")
        return (-u_padded[4:] + 8 * u_padded[3:-1] - 8 * u_padded[1:-3] + u_padded[:-4]) / (12 * dx)

    elif order == 6:
        # Sixth-order central difference
        u_padded = jnp.pad(u, 3, mode="edge")
        return (
            u_padded[6:]
            - 9 * u_padded[5:-1]
            + 45 * u_padded[4:-2]
            - 45 * u_padded[2:-4]
            + 9 * u_padded[1:-5]
            - u_padded[:-6]
        ) / (60 * dx)

    else:
        raise ValueError(f"Unsupported order: {order}")


@jit
def finite_difference_2d(u: JAXArray, dx: float, order: int = 2) -> JAXArray:
    """
    Compute second derivative using finite differences.

    Args:
        u: 1D array to differentiate
        dx: Grid spacing
        order: Order of accuracy (2, 4, or 6)

    Returns:
        Second derivative array
    """
    if order == 2:
        # Second-order central difference for second derivative
        u_padded = jnp.pad(u, 1, mode="edge")
        return (u_padded[2:] - 2 * u_padded[1:-1] + u_padded[:-2]) / (dx**2)

    elif order == 4:
        # Fourth-order central difference for second derivative
        u_padded = jnp.pad(u, 2, mode="edge")
        return (-u_padded[4:] + 16 * u_padded[3:-1] - 30 * u_padded[2:-2] + 16 * u_padded[1:-3] - u_padded[:-4]) / (
            12 * dx**2
        )

    elif order == 6:
        # Sixth-order central difference for second derivative
        u_padded = jnp.pad(u, 3, mode="edge")
        return (
            2 * u_padded[6:]
            - 27 * u_padded[5:-1]
            + 270 * u_padded[4:-2]
            - 490 * u_padded[3:-3]
            + 270 * u_padded[2:-4]
            - 27 * u_padded[1:-5]
            + 2 * u_padded[:-6]
        ) / (180 * dx**2)

    else:
        raise ValueError(f"Unsupported order: {order}")


def apply_boundary_conditions(u: JAXArray, bc_type: str = "neumann", bc_value: float = 0.0) -> JAXArray:
    """
    Apply boundary conditions to array.

    Args:
        u: Array to apply boundary conditions to
        bc_type: Type of boundary condition ('neumann', 'dirichlet', 'periodic')
        bc_value: Boundary condition value

    Returns:
        Array with boundary conditions applied
    """
    # Handle both JAX and numpy arrays
    if HAS_JAX and hasattr(u, "at"):
        # JAX array - use immutable updates with type-safe access
        try:
            if bc_type == "dirichlet":
                u = u.at[0].set(bc_value)
                u = u.at[-1].set(bc_value)
            elif bc_type == "neumann":
                u = u.at[0].set(u[1])
                u = u.at[-1].set(u[-2])
            elif bc_type == "periodic":
                u = u.at[0].set(u[-2])
                u = u.at[-1].set(u[1])
        except AttributeError:
            # Fallback to numpy path if .at access fails
            u = u.copy()
            if bc_type == "dirichlet":
                u[0] = bc_value
                u[-1] = bc_value
            elif bc_type == "neumann":
                u[0] = u[1]
                u[-1] = u[-2]
            elif bc_type == "periodic":
                u[0] = u[-2]
                u[-1] = u[1]
    else:
        # Numpy array - use mutable updates
        u = u.copy()  # Make a copy to avoid modifying input
        if bc_type == "dirichlet":
            u[0] = bc_value
            u[-1] = bc_value
        elif bc_type == "neumann":
            u[0] = u[1]
            u[-1] = u[-2]
        elif bc_type == "periodic":
            u[0] = u[-2]
            u[-1] = u[1]

    return u


def tridiagonal_solve(a: JAXArray, b: JAXArray, c: JAXArray, d: JAXArray) -> JAXArray:
    """
    Solve tridiagonal system using Thomas algorithm.

    Args:
        a: Lower diagonal (size n-1)
        b: Main diagonal (size n)
        c: Upper diagonal (size n-1)
        d: Right-hand side (size n)

    Returns:
        Solution vector
    """
    if HAS_JAX and hasattr(a, "at"):
        # JAX implementation with scan and immutable updates
        return _tridiagonal_solve_jax(a, b, c, d)
    else:
        # Numpy implementation with simple loops
        return _tridiagonal_solve_numpy(a, b, c, d)


@jit
def _tridiagonal_solve_jax(a: JAXArray, b: JAXArray, c: JAXArray, d: JAXArray) -> JAXArray:
    """JAX-specific tridiagonal solver using scan."""
    n = len(b)

    # Pad arrays to same size for vectorization
    a_pad = jnp.concatenate([jnp.array([0.0]), a])
    c_pad = jnp.concatenate([c, jnp.array([0.0])])

    # Forward elimination
    def forward_step(carry, i):
        c_mod, d_mod = carry
        w = a_pad[i] / b[i - 1] if i > 0 else 0.0
        c_new = c_mod.at[i].set(c_pad[i] / (b[i] - w * c_mod[i - 1]) if i > 0 else c_pad[i] / b[i])
        d_new = d_mod.at[i].set((d[i] - w * d_mod[i - 1]) / (b[i] - w * c_mod[i - 1]) if i > 0 else d[i] / b[i])
        return (c_new, d_new), None

    c_mod = jnp.zeros_like(c_pad)
    d_mod = jnp.zeros_like(d)
    (c_mod, d_mod), _ = scan(forward_step, (c_mod, d_mod), jnp.arange(n))

    # Back substitution
    def backward_step(carry, i):
        x = carry
        i_actual = n - 1 - i
        x_new = d_mod[i_actual] - c_mod[i_actual] * x[i_actual + 1] if i_actual < n - 1 else d_mod[i_actual]
        x = x.at[i_actual].set(x_new)
        return x, None

    x = jnp.zeros(n)
    x, _ = scan(backward_step, x, jnp.arange(n))

    return x


def _tridiagonal_solve_numpy(a: JAXArray, b: JAXArray, c: JAXArray, d: JAXArray) -> JAXArray:
    """Numpy-compatible tridiagonal solver using simple loops."""
    n = len(b)

    # Convert to numpy arrays and make copies
    a = np.asarray(a)
    b = np.asarray(b).copy()
    c = np.asarray(c).copy()
    d = np.asarray(d).copy()

    # Forward elimination
    for i in range(1, n):
        w = a[i - 1] / b[i - 1]
        b[i] = b[i] - w * c[i - 1]
        d[i] = d[i] - w * d[i - 1]

    # Back substitution
    x = np.zeros(n)
    x[n - 1] = d[n - 1] / b[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


@jit
def compute_hamiltonian(u_x: JAXArray, m: JAXArray, sigma: float) -> JAXArray:
    """
    Compute Hamiltonian for HJB equation.

    Args:
        u_x: Spatial derivative of value function
        m: Density function
        sigma: Diffusion coefficient

    Returns:
        Hamiltonian value
    """
    # H(x, p, m) = (1/2) * p^2 + interaction_term(x, m)
    kinetic_term = 0.5 * u_x**2

    # Simple interaction term (can be customized)
    interaction_term = jnp.log(m + 1e-8)  # Logarithmic interaction

    return kinetic_term + interaction_term


@jit
def compute_optimal_control(u_x: JAXArray) -> JAXArray:
    """
    Compute optimal control from value function gradient.

    Args:
        u_x: Spatial derivative of value function

    Returns:
        Optimal control
    """
    return -u_x  # For quadratic cost


@jit
def compute_drift(u_x: JAXArray, sigma: float) -> JAXArray:
    """
    Compute drift term for Fokker-Planck equation.

    Args:
        u_x: Spatial derivative of value function
        sigma: Diffusion coefficient

    Returns:
        Drift term
    """
    return compute_optimal_control(u_x)


@jit
def mass_conservation_constraint(m: JAXArray, dx: float) -> float:
    """
    Compute mass conservation constraint.

    Args:
        m: Density function
        dx: Grid spacing

    Returns:
        Mass conservation error
    """
    total_mass = jnp.sum(m) * dx
    return float(jnp.abs(total_mass - 1.0))


def create_optimization_schedule(learning_rate: float = 1e-3, decay_steps: int = 1000, decay_rate: float = 0.9) -> Any:
    """
    Create learning rate schedule for optimization.

    Args:
        learning_rate: Initial learning rate
        decay_steps: Steps between decay
        decay_rate: Decay rate

    Returns:
        Optax scheduler
    """
    ensure_jax_available()

    if not HAS_OPTAX or optax is None:
        raise ImportError("Optimization schedules require optax. Install with: pip install optax")

    schedule = optax.exponential_decay(init_value=learning_rate, transition_steps=decay_steps, decay_rate=decay_rate)

    return optax.adam(schedule)


@jit
def compute_convergence_error(u_new: JAXArray, u_old: JAXArray, m_new: JAXArray, m_old: JAXArray) -> float:
    """
    Compute convergence error for MFG system.

    Args:
        u_new: New value function
        u_old: Old value function
        m_new: New density function
        m_old: Old density function

    Returns:
        Convergence error
    """
    u_error = jnp.linalg.norm(u_new - u_old) / (jnp.linalg.norm(u_old) + 1e-8)
    m_error = jnp.linalg.norm(m_new - m_old) / (jnp.linalg.norm(m_old) + 1e-8)

    return float(u_error + m_error)


@jit
def adaptive_time_step(
    error: float,
    dt: float,
    tolerance: float = 1e-6,
    safety_factor: float = 0.9,
    max_factor: float = 2.0,
    min_factor: float = 0.1,
) -> float:
    """
    Compute adaptive time step based on error estimate.

    Args:
        error: Current error estimate
        dt: Current time step
        tolerance: Target tolerance
        safety_factor: Safety factor for step size adjustment
        max_factor: Maximum increase factor
        min_factor: Minimum decrease factor

    Returns:
        New time step
    """
    if error == 0:
        factor: Any = max_factor
    else:
        factor = safety_factor * jnp.power(tolerance / error, 0.2)

    factor = jnp.clip(factor, min_factor, max_factor)
    return float(dt * factor)


def vectorized_solve(solve_func: Callable, problems: list, batch_size: int | None = None) -> list:
    """
    Solve multiple problems in vectorized batches.

    Args:
        solve_func: JAX-compiled solving function
        problems: List of problem configurations
        batch_size: Batch size for processing (None for auto)

    Returns:
        List of solutions
    """
    ensure_jax_available()

    if batch_size is None:
        # Auto-determine batch size based on available memory
        batch_size = min(len(problems), 32)  # Conservative default

    results = []

    for i in range(0, len(problems), batch_size):
        batch = problems[i : i + batch_size]

        # Convert to JAX arrays if needed
        batch_arrays = []
        for problem in batch:
            if hasattr(problem, "to_jax_arrays"):
                batch_arrays.append(problem.to_jax_arrays())
            else:
                batch_arrays.append(problem)

        # Vectorized solve
        batch_results = vmap(solve_func)(batch_arrays)

        # Convert back to numpy if needed
        batch_results_np = [from_device(result) for result in batch_results]
        results.extend(batch_results_np)

    return results


def profile_jax_function(func: Callable, *args, num_runs: int = 10) -> dict:
    """
    Profile JAX function performance.

    Args:
        func: JAX function to profile
        *args: Arguments to function
        num_runs: Number of runs for timing

    Returns:
        Performance metrics
    """
    ensure_jax_available()

    import time

    # Compile function first
    compiled_func = jit(func)

    # Warm-up run
    _ = compiled_func(*args)

    # Time compilation
    compile_start = time.time()
    compiled_func = jit(func)
    _ = compiled_func(*args)
    compile_time = time.time() - compile_start

    # Time execution
    exec_times = []
    for _ in range(num_runs):
        start = time.time()
        result = compiled_func(*args)

        if HAS_GPU:
            # Ensure GPU computation is complete
            result.block_until_ready()

        exec_times.append(time.time() - start)

    return {
        "compile_time": compile_time,
        "mean_exec_time": np.mean(exec_times),
        "std_exec_time": np.std(exec_times),
        "min_exec_time": np.min(exec_times),
        "max_exec_time": np.max(exec_times),
        "speedup_vs_numpy": None,  # Could be computed with numpy baseline
    }


def memory_usage_tracker():
    """Create memory usage tracker for JAX computations."""
    ensure_jax_available()

    class MemoryTracker:
        def __init__(self):
            self.peak_memory = 0
            self.current_memory = 0

        def update(self):
            if HAS_GPU and jax is not None:
                try:
                    # Get GPU memory usage
                    for device in jax.devices("gpu"):
                        if hasattr(device, "memory_stats"):
                            stats = device.memory_stats()
                            current = stats.get("bytes_in_use", 0) / 1024**3  # GB
                            self.current_memory = max(self.current_memory, current)
                            self.peak_memory = max(self.peak_memory, current)
                except (AttributeError, KeyError, RuntimeError):
                    # Memory stats may not be available or accessible
                    pass

        def reset(self):
            self.peak_memory = 0
            self.current_memory = 0

    return MemoryTracker()


# Export utility functions
__all__ = [
    "adaptive_time_step",
    "apply_boundary_conditions",
    "compute_convergence_error",
    "compute_drift",
    "compute_hamiltonian",
    "compute_jacobian",
    "compute_jacobian_jit",
    "compute_optimal_control",
    "create_optimization_schedule",
    "ensure_jax_available",
    "finite_difference_1d",
    "finite_difference_2d",
    "from_device",
    "mass_conservation_constraint",
    "memory_usage_tracker",
    "profile_jax_function",
    "to_device",
    "tridiagonal_solve",
    "vectorized_solve",
]

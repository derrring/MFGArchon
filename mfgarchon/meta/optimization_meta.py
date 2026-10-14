"""
Meta-programming for optimization and JIT compilation of MFG solvers.

This module provides advanced optimization capabilities including:
- Runtime code optimization and specialization
- JIT compilation with backend selection
- Performance-driven solver generation
- Automatic memory layout optimization
"""

from __future__ import annotations

import functools
import inspect
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.utils.deprecation import retired_parameters

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class PerformanceProfile:
    """Performance characteristics for optimization decisions."""

    problem_size: int
    backend: str
    target_precision: float | None = None  # refused when passed; never read (#2648)
    memory_constraint: int | None = None  # MB
    time_constraint: float | None = None  # refused when passed; never read (#2648)
    optimization_level: str = "balanced"  # "speed", "balanced", "accuracy"


PerformanceProfile.__init__ = retired_parameters(  # type: ignore[method-assign]
    {
        "target_precision": "no optimisation decision reads a precision target",
        "time_constraint": "no optimisation decision reads a time budget",
    },
    issue="#2648",
    since="v0.22.0",
    never_read=True,
)(PerformanceProfile.__init__)


@dataclass
class OptimizationHints:
    """Hints for code optimization and specialization."""

    vectorizable: bool = True
    parallelizable: bool = True
    memory_bound: bool = False
    compute_bound: bool = True

    # Loop optimization hints
    unroll_small_loops: bool = True  # refused when passed; never read (#2648)
    vectorize_inner_loops: bool = True  # refused when passed; never read (#2648)

    # Backend-specific hints
    jax_hints: dict[str, Any] | None = None
    numba_hints: dict[str, Any] | None = None

    def __post_init__(self):
        if self.jax_hints is None:
            self.jax_hints = {"jit": True, "static_argnums": ()}
        if self.numba_hints is None:
            self.numba_hints = {"nopython": True, "parallel": False}


OptimizationHints.__init__ = retired_parameters(  # type: ignore[method-assign]
    {
        "unroll_small_loops": "no compiler step unrolls loops from this hint",
        "vectorize_inner_loops": "no compiler step vectorises from this hint",
    },
    issue="#2648",
    since="v0.22.0",
    never_read=True,
)(OptimizationHints.__init__)


class OptimizationCompiler:
    """
    Compiler for optimizing MFG solver code.

    Analyzes solver implementations and generates optimized versions
    based on problem characteristics and performance requirements.
    """

    def __init__(self, backend: str = "auto"):
        self.backend = backend
        self.optimization_cache: dict[str, Callable] = {}
        self.performance_stats: dict[str, dict] = {}

    def optimize_function(
        self,
        func: Callable,
        profile: PerformanceProfile,
        hints: OptimizationHints | None = None,
    ) -> Callable:
        """Optimize function based on performance profile."""

        if hints is None:
            hints = self._infer_optimization_hints(func, profile)

        # Create cache key
        cache_key = self._create_cache_key(func, profile, hints)

        if cache_key in self.optimization_cache:
            return self.optimization_cache[cache_key]

        # Select optimization strategy
        if self.backend == "auto":
            backend = self._select_backend(profile, hints)
        else:
            backend = self.backend

        # Apply optimizations
        optimized_func = self._apply_optimizations(func, backend, profile, hints)

        # Cache result
        self.optimization_cache[cache_key] = optimized_func

        return optimized_func

    def _infer_optimization_hints(self, func: Callable, profile: PerformanceProfile) -> OptimizationHints:
        """Infer optimization hints from function analysis."""

        # Analyze function source
        source = inspect.getsource(func)

        # Simple heuristics for optimization hints
        vectorizable = "numpy" in source or "np." in source
        parallelizable = profile.problem_size > 1000 and "for" in source
        memory_bound = "array" in source and profile.problem_size > 10000

        return OptimizationHints(
            vectorizable=vectorizable,
            parallelizable=parallelizable,
            memory_bound=memory_bound,
            compute_bound=not memory_bound,
        )

    def _select_backend(self, profile: PerformanceProfile, hints: OptimizationHints) -> str:
        """Select optimal backend based on profile and hints."""

        # Large problems with vectorization -> JAX
        if profile.problem_size > 5000 and hints.vectorizable:
            return "jax"

        # Small to medium problems with tight loops -> Numba
        elif profile.problem_size < 5000 and hints.compute_bound:
            return "numba"

        # Default to NumPy
        else:
            return "numpy"

    def _apply_optimizations(
        self,
        func: Callable,
        backend: str,
        profile: PerformanceProfile,
        hints: OptimizationHints,
    ) -> Callable:
        """Apply backend-specific optimizations."""

        if backend == "jax":
            return self._optimize_with_jax(func, profile, hints)
        elif backend == "numba":
            return self._optimize_with_numba(func, profile, hints)
        elif backend == "numpy":
            return self._optimize_with_numpy(func, profile, hints)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _optimize_with_jax(self, func: Callable, profile: PerformanceProfile, hints: OptimizationHints) -> Callable:
        """Apply JAX-specific optimizations."""
        try:
            import jax
            import jax.numpy as jnp  # noqa: F401

            # Determine JIT settings
            jit_kwargs = hints.jax_hints.copy()

            if profile.optimization_level == "speed":
                jit_kwargs["device"] = jax.devices()[0]

            # Apply JIT compilation
            optimized = jax.jit(func, **jit_kwargs)

            # Add memory optimization if needed
            if profile.memory_constraint:
                optimized = self._add_memory_optimization(optimized)

            return optimized

        except ImportError:
            print("JAX not available, falling back to NumPy")
            return self._optimize_with_numpy(func, profile, hints)

    def _optimize_with_numba(self, func: Callable, profile: PerformanceProfile, hints: OptimizationHints) -> Callable:
        """Apply Numba-specific optimizations."""
        try:
            import numba

            # Configure Numba compilation
            numba_kwargs = hints.numba_hints.copy()

            if hints.parallelizable and profile.problem_size > 1000:
                numba_kwargs["parallel"] = True

            if profile.optimization_level == "speed":
                numba_kwargs["fastmath"] = True

            # Apply JIT compilation
            optimized = numba.jit(**numba_kwargs)(func)

            return optimized

        except ImportError:
            print("Numba not available, falling back to NumPy")
            return self._optimize_with_numpy(func, profile, hints)

    def _optimize_with_numpy(self, func: Callable, profile: PerformanceProfile, hints: OptimizationHints) -> Callable:
        """Apply NumPy-specific optimizations."""

        # For NumPy, we focus on algorithmic optimizations
        @functools.wraps(func)
        def optimized_func(*args, **kwargs):
            # Pre-allocate arrays if possible
            if hints.memory_bound and profile.memory_constraint:
                # Could implement memory pooling here
                pass

            # Call original function
            result = func(*args, **kwargs)

            return result

        return optimized_func

    def _add_memory_optimization(self, func: Callable) -> Callable:
        """Add memory optimization wrapper."""

        @functools.wraps(func)
        def memory_optimized(*args, **kwargs):
            # Monitor memory usage
            # Could implement chunking for large arrays
            return func(*args, **kwargs)

        return memory_optimized

    def _create_cache_key(self, func: Callable, profile: PerformanceProfile, hints: OptimizationHints) -> str:
        """Create unique cache key for optimization."""
        return f"{func.__name__}_{profile.problem_size}_{profile.backend}_{profile.optimization_level}"

    def benchmark_function(self, func: Callable, args: tuple, kwargs: dict, num_runs: int = 10) -> dict[str, float]:
        """Benchmark function performance."""

        times = []

        for _ in range(num_runs):
            start_time = time.perf_counter()
            func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        return {
            "mean_time": float(np.mean(times)),
            "std_time": float(np.std(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
        }


class JITSolverFactory:
    """
    Factory for creating JIT-compiled MFG solvers.

    Automatically selects and compiles optimal solver implementations
    based on problem characteristics.
    """

    def __init__(self) -> None:
        self.compiler = OptimizationCompiler()
        self.solver_cache: dict[str, type] = {}

    def create_optimized_solver(
        self,
        base_solver_class: type,
        problem,
        performance_profile: PerformanceProfile | None = None,
    ) -> type:
        """Create optimized solver class for given problem."""

        if performance_profile is None:
            performance_profile = self._infer_performance_profile(problem)

        # Create cache key
        cache_key = f"{base_solver_class.__name__}_{performance_profile.problem_size}_{performance_profile.backend}"

        if cache_key in self.solver_cache:
            return self.solver_cache[cache_key]

        # Create optimized solver class
        optimized_class = self._create_optimized_class(base_solver_class, performance_profile)

        # Cache result
        self.solver_cache[cache_key] = optimized_class

        return optimized_class

    def _infer_performance_profile(self, problem) -> PerformanceProfile:
        """Infer performance profile from problem characteristics."""

        # Estimate problem size (Issue #643: getattr pattern)
        geometry = getattr(problem, "geometry", None)
        Nt = getattr(problem, "Nt", None)
        if geometry is not None and Nt is not None:
            grid_shape = geometry.get_grid_shape()
            problem_size = int(np.prod(grid_shape)) * Nt
        else:
            raise ValueError(
                "Cannot infer problem size: problem must have 'geometry' and 'Nt'. "
                "Provide an explicit PerformanceProfile if using a custom problem type."
            )

        # Select backend based on size
        if problem_size > 10000:
            backend = "jax"
        elif problem_size > 1000:
            backend = "numba"
        else:
            backend = "numpy"

        return PerformanceProfile(
            problem_size=problem_size,
            backend=backend,
            optimization_level="balanced",
        )

    def _create_optimized_class(self, base_class: type, profile: PerformanceProfile) -> type:
        """Create optimized version of solver class."""

        class OptimizedSolver(base_class):
            """Dynamically optimized solver class."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.performance_profile = profile
                self._optimize_methods()

            def _optimize_methods(self):
                """Optimize key solver methods."""

                # Get methods to optimize
                methods_to_optimize = [
                    "_solve_hjb_step",
                    "_solve_fp_step",
                    "_compute_residual",
                    "_apply_boundary_conditions",
                ]

                for method_name in methods_to_optimize:
                    # Issue #643: getattr pattern with callable check
                    original_method = getattr(self, method_name, None)
                    if original_method is not None and callable(original_method):
                        optimized_method = self.compiler.optimize_function(original_method, profile)
                        setattr(self, method_name, optimized_method)

        # Set class name
        OptimizedSolver.__name__ = f"Optimized{base_class.__name__}"
        OptimizedSolver.__qualname__ = f"Optimized{base_class.__qualname__}"

        return OptimizedSolver


def create_optimized_solver(
    solver_class: type,
    problem,
    optimization_level: str = "balanced",
    backend: str = "auto",
):
    """
    Convenience function to create optimized solver instance.

    Args:
        solver_class: Base solver class to optimize
        problem: MFG problem instance
        optimization_level: speed, "balanced", or "accuracy"
        backend: Computational backend ("auto", "jax", "numba", "numpy")

    Returns:
        Optimized solver instance
    """

    factory = JITSolverFactory()

    # Create performance profile (Issue #643: getattr pattern)
    geometry = getattr(problem, "geometry", None)
    Nt = getattr(problem, "Nt", None)
    if geometry is not None and Nt is not None:
        grid_shape = geometry.get_grid_shape()
        problem_size = int(np.prod(grid_shape)) * Nt
    else:
        raise ValueError(
            "Cannot create optimized solver: problem missing 'geometry' or 'Nt'. "
            "Optimization requires explicit problem size metrics."
        )

    profile = PerformanceProfile(
        problem_size=problem_size,
        backend=backend,
        optimization_level=optimization_level,
    )

    # Create optimized class
    optimized_class = factory.create_optimized_solver(solver_class, problem, profile)

    # Return instance
    return optimized_class(problem)


# Decorators for automatic optimization


def jit_optimize(backend: str = "auto", optimization_level: str = "balanced"):
    """Decorator for automatic JIT optimization of solver methods."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get self (solver instance) if method (Issue #643: getattr pattern)
            if args:
                solver = args[0]

                # Create performance profile if not exists
                opt_profile = getattr(solver, "_optimization_profile", None)
                if opt_profile is None:
                    # Optimization profile requires a valid problem with size metrics
                    problem = getattr(solver, "problem", None)
                    if problem is None:
                        raise AttributeError("Optimization failed: solver missing 'problem' attribute.")

                    # Try to get problem size explicitly or via modern attributes
                    geometry = getattr(problem, "geometry", None)
                    Nt = getattr(problem, "Nt", None)
                    if geometry is not None and Nt is not None:
                        grid_shape = geometry.get_grid_shape()
                        problem_size = int(np.prod(grid_shape)) * Nt
                    else:
                        # Fallback to legacy 1D attributes
                        Nx = getattr(problem, "Nx", None)
                        if Nx is not None and Nt is not None:
                            problem_size = Nx * Nt
                        else:
                            raise ValueError(
                                "Cannot determine problem size for JIT optimization. Problem must have 'geometry' and 'Nt'."
                            )

                    solver._optimization_profile = PerformanceProfile(
                        problem_size=problem_size,
                        backend=backend,
                        optimization_level=optimization_level,
                    )
                    opt_profile = solver._optimization_profile

                # Optimize function if not already done
                cache_key = f"{func.__name__}_optimized"
                optimized_func = getattr(solver, cache_key, None)
                if optimized_func is None:
                    compiler = OptimizationCompiler(backend)
                    optimized_func = compiler.optimize_function(func, opt_profile)
                    setattr(solver, cache_key, optimized_func)

                return optimized_func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def adaptive_backend(backends: list[str] | None = None):
    """Decorator for adaptive backend selection based on problem size."""

    if backends is None:
        backends = ["numpy", "numba", "jax"]

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract problem information (Issue #643: getattr pattern)
            problem_size = 1000  # Default

            if args:
                solver = args[0]
                problem = getattr(solver, "problem", None)
                if problem is not None:
                    geometry = getattr(problem, "geometry", None)
                    Nt = getattr(problem, "Nt", None)
                    if geometry is not None and Nt is not None:
                        grid_shape = geometry.get_grid_shape()
                        problem_size = int(np.prod(grid_shape)) * Nt

            # Select backend based on problem size
            if problem_size > 10000 and "jax" in backends:
                selected_backend = "jax"
            elif problem_size > 1000 and "numba" in backends:
                selected_backend = "numba"
            else:
                selected_backend = "numpy"

            # Apply optimization
            compiler = OptimizationCompiler(selected_backend)
            profile = PerformanceProfile(
                problem_size=problem_size,
                backend=selected_backend,
            )

            optimized_func = compiler.optimize_function(func, profile)
            return optimized_func(*args, **kwargs)

        return wrapper

    return decorator


# Example usage and testing


def test_optimization_framework():
    """Test the optimization meta-programming framework."""

    # Example function to optimize
    def example_mfg_step(u, m, dt, dx):
        """Example MFG iteration step."""
        grad_u = np.gradient(u, dx)
        laplacian_u = np.gradient(grad_u, dx)

        # HJB update
        u_new = u + dt * (0.5 * laplacian_u - 0.5 * grad_u**2)

        # FP update
        flux = m * grad_u
        div_flux = np.gradient(flux, dx)
        m_new = m + dt * (0.5 * np.gradient(np.gradient(m, dx), dx) - div_flux)

        return u_new, m_new

    # Create optimization profile
    profile = PerformanceProfile(
        problem_size=5000,
        backend="auto",
        optimization_level="balanced",
    )

    # Optimize function
    compiler = OptimizationCompiler()
    optimized_step = compiler.optimize_function(example_mfg_step, profile)

    # Test performance
    u = np.random.randn(100)
    m = np.abs(np.random.randn(100))
    m = m / np.sum(m)  # Normalize
    dt, dx = 0.01, 0.01

    # Benchmark
    original_stats = compiler.benchmark_function(example_mfg_step, (u, m, dt, dx), {}, num_runs=5)

    optimized_stats = compiler.benchmark_function(optimized_step, (u, m, dt, dx), {}, num_runs=5)

    print(f"Original time: {original_stats['mean_time']:.6f}s")
    print(f"Optimized time: {optimized_stats['mean_time']:.6f}s")
    print(f"Speedup: {original_stats['mean_time'] / optimized_stats['mean_time']:.2f}x")


if __name__ == "__main__":
    test_optimization_framework()

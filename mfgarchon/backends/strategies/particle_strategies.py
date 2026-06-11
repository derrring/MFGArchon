"""
Strategy pattern implementations for particle-based FP solvers.

This module defines abstract strategy interface and concrete implementations
for CPU, GPU, and hybrid particle solving approaches. Strategies enable
intelligent backend selection based on problem size and hardware capabilities.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver

if TYPE_CHECKING:
    from mfgarchon.backends.base_backend import BaseBackend
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry.boundary.fdm_bc_1d import BoundaryConditions


class ParticleStrategy(ABC):
    """
    Abstract base class for particle solver strategies.

    Strategies encapsulate different implementation approaches for particle-based
    Fokker-Planck solvers, enabling automatic selection based on backend
    capabilities and problem characteristics.
    """

    @abstractmethod
    def solve(
        self,
        m_initial: np.ndarray,
        U_drift: np.ndarray,
        problem: "MFGProblem",
        num_particles: int,
        kde_bandwidth,
        normalize_kde_output: bool,
        boundary_conditions: "BoundaryConditions",
        backend: "BaseBackend | None",
    ) -> np.ndarray:
        """
        Solve Fokker-Planck system using specific strategy.

        Parameters
        ----------
        m_initial : np.ndarray
            Initial density condition, shape (Nx+1,)
        U_drift : np.ndarray
            Value function for drift computation, shape (Nt+1, Nx+1)
        problem : MFGProblem
            MFG problem definition
        num_particles : int
            Number of particles to simulate
        kde_bandwidth : float or str
            Bandwidth for kernel density estimation
        normalize_kde_output : bool
            Whether to normalize KDE output to unit mass
        boundary_conditions : BoundaryConditions
            Boundary condition specification
        backend : BaseBackend, optional
            Computational backend (None for CPU)

        Returns
        -------
        np.ndarray
            Density evolution, shape (Nt+1, Nx+1)
        """

    @abstractmethod
    def estimate_cost(self, problem_size: tuple[int, int, int]) -> float:
        """
        Estimate computational cost (seconds) for given problem size.

        This method enables intelligent strategy selection by comparing
        estimated costs before execution.

        Parameters
        ----------
        problem_size : tuple
            (num_particles, grid_size, time_steps)

        Returns
        -------
        float
            Estimated execution time in seconds
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging and debugging."""


class CPUParticleStrategy(ParticleStrategy):
    """
    CPU-based particle solver using NumPy + scipy.

    This strategy uses the existing _solve_fp_system_cpu implementation
    with NumPy operations and scipy.stats.gaussian_kde for density estimation.

    Best for:
    - Small problems (N < 10,000 particles)
    - Systems without GPU
    - Debugging and validation
    """

    @property
    def name(self) -> str:
        return "cpu"

    def solve(
        self,
        m_initial: np.ndarray,
        U_drift: np.ndarray,
        problem: "MFGProblem",
        num_particles: int,
        kde_bandwidth,
        normalize_kde_output: bool,
        boundary_conditions: "BoundaryConditions",
        backend: "BaseBackend | None",
    ) -> np.ndarray:
        """CPU implementation using NumPy + scipy."""
        # Use existing CPU implementation
        return FPParticleSolver._solve_fp_system_cpu_static(
            m_initial,
            U_drift,
            problem,
            num_particles,
            kde_bandwidth,
            normalize_kde_output,
            boundary_conditions,
        )

    def estimate_cost(self, problem_size: tuple[int, int, int]) -> float:
        """Estimate CPU cost based on empirical model."""
        N, Nx, Nt = problem_size

        # KDE dominates cost: O(Nt * N * Nx)
        kde_cost = Nt * N * Nx * 1e-8  # ~10 ns per operation

        # Other operations: particle update, interpolation, boundary conditions
        other_cost = Nt * N * 1e-9  # ~1 ns per particle per timestep

        return kde_cost + other_cost


class GPUParticleStrategy(ParticleStrategy):
    """
    GPU-based particle solver with internal KDE.

    This strategy uses the existing _solve_fp_system_gpu implementation
    with PyTorch/JAX backend operations and internal GPU KDE to eliminate
    GPU↔CPU transfers.

    Best for:
    - Large problems (N ≥ 50,000 particles on MPS, N ≥ 10,000 on CUDA)
    - Systems with GPU available
    - Production runs requiring speed
    """

    def __init__(self, backend: "BaseBackend"):
        if backend is None:
            raise ValueError("GPUParticleStrategy requires a backend")
        self.backend = backend

    @property
    def name(self) -> str:
        hints = self.backend.get_performance_hints()
        device = hints.get("device_type", "gpu")
        return f"gpu-{device}"

    def solve(
        self,
        m_initial: np.ndarray,
        U_drift: np.ndarray,
        problem: "MFGProblem",
        num_particles: int,
        kde_bandwidth,
        normalize_kde_output: bool,
        boundary_conditions: "BoundaryConditions",
        backend: "BaseBackend | None",
    ) -> np.ndarray:
        """GPU implementation using internal GPU KDE."""
        # Use existing GPU implementation
        return FPParticleSolver._solve_fp_system_gpu_static(
            m_initial,
            U_drift,
            problem,
            num_particles,
            kde_bandwidth,
            normalize_kde_output,
            boundary_conditions,
            self.backend,
        )

    def estimate_cost(self, problem_size: tuple[int, int, int]) -> float:
        """Estimate GPU cost based on backend performance hints."""
        N, Nx, Nt = problem_size
        hints = self.backend.get_performance_hints()

        # Kernel launch overhead
        kernel_overhead_us = hints["kernel_overhead_us"]
        kernels_per_iteration = 5  # KDE, interpolation, drift, boundary, etc.
        num_kernels = Nt * kernels_per_iteration
        overhead_cost = num_kernels * kernel_overhead_us * 1e-6  # Convert to seconds

        # Compute cost (GPU is faster per operation than CPU)
        kde_speedup = 5.0  # GPU KDE is ~5x faster than CPU scipy
        kde_cost = Nt * N * Nx * 1e-8 / kde_speedup

        other_speedup = 2.0  # Other ops get modest GPU speedup
        other_cost = Nt * N * 1e-9 / other_speedup

        return overhead_cost + kde_cost + other_cost


class HybridParticleStrategy(ParticleStrategy):
    """
    Hybrid CPU/GPU strategy for intermediate problems.

    This strategy uses GPU for computationally intensive operations (KDE)
    and CPU for lightweight operations, minimizing transfer overhead while
    leveraging GPU compute power.

    Best for:
    - Medium problems (10,000 ≤ N < 50,000 particles)
    - MPS backend (high kernel overhead)
    - Experimental optimization
    """

    def __init__(self, backend: "BaseBackend | None"):
        """
        Initialize hybrid strategy.

        Parameters
        ----------
        backend : BaseBackend | None
            GPU backend for large problems. If None, will use CPU for all problems.
        """
        self.backend = backend

    @property
    def name(self) -> str:
        return "hybrid"

    def solve(
        self,
        m_initial: np.ndarray,
        U_drift: np.ndarray,
        problem: "MFGProblem",
        num_particles: int,
        kde_bandwidth,
        normalize_kde_output: bool,
        boundary_conditions: "BoundaryConditions",
        backend: "BaseBackend | None",
    ) -> np.ndarray:
        """
        Adaptive hybrid strategy: Intelligently chooses CPU or GPU based on problem size.

        Decision criteria:
        - CPU: Small problems (N < 10,000 particles, avoids GPU overhead)
        - GPU: Large problems (N ≥ 10,000 particles, benefits from parallelism)

        Note: A true mixed-execution hybrid (CPU propagation + GPU KDE) requires
        careful management of GPU↔CPU transfers. This adaptive approach provides
        better performance by avoiding transfer overhead for small problems while
        using full GPU acceleration for large problems.
        """
        # Decision threshold: Use GPU when KDE benefit outweighs transfer overhead
        # For KDE: compute cost O(N*Nx), transfer cost O(N+Nx)
        # GPU wins when: speedup * (N*Nx) > overhead + (N+Nx)
        #
        # Empirically:
        # - Apple MPS: threshold ~10,000 particles
        # - NVIDIA CUDA: threshold ~5,000 particles (better GPU utilization)
        #
        # Conservative threshold: 10,000 particles
        gpu_threshold = 10_000

        if num_particles < gpu_threshold:
            # Small problem: Use CPU (avoids GPU transfer overhead)
            # Issue #1283: CPUParticleStrategy has no __init__; passing args raises TypeError.
            cpu_strategy = CPUParticleStrategy()
            return cpu_strategy.solve(
                m_initial,
                U_drift,
                problem,
                num_particles,
                kde_bandwidth,
                normalize_kde_output,
                boundary_conditions,
                backend,
            )
        else:
            # Large problem: Use full GPU (benefits from parallelism)
            if self.backend is None:
                # No GPU available, fall back to CPU
                # Issue #1283: CPUParticleStrategy has no __init__; passing args raises TypeError.
                cpu_strategy = CPUParticleStrategy()
                return cpu_strategy.solve(
                    m_initial,
                    U_drift,
                    problem,
                    num_particles,
                    kde_bandwidth,
                    normalize_kde_output,
                    boundary_conditions,
                    backend,
                )

            gpu_strategy = GPUParticleStrategy(self.backend)
            return gpu_strategy.solve(
                m_initial,
                U_drift,
                problem,
                num_particles,
                kde_bandwidth,
                normalize_kde_output,
                boundary_conditions,
                backend,
            )

    def estimate_cost(self, problem_size: tuple[int, int, int]) -> float:
        """
        Estimate hybrid cost using adaptive strategy.

        For small problems (N < 10,000): estimates CPU cost
        For large problems (N ≥ 10,000): estimates GPU cost
        """
        N, Nx, Nt = problem_size
        gpu_threshold = 10_000

        if gpu_threshold > N:
            # Small problem: CPU cost
            kde_cost = Nt * N * Nx * 1e-8  # ~10 ns per KDE operation
            other_cost = Nt * N * 1e-9  # ~1 ns per particle per timestep
            return kde_cost + other_cost
        else:
            # Large problem: GPU cost
            if self.backend is None:
                # No GPU: fall back to CPU cost
                kde_cost = Nt * N * Nx * 1e-8
                other_cost = Nt * N * 1e-9
                return kde_cost + other_cost

            hints = self.backend.get_performance_hints()
            kernel_overhead_us = hints["kernel_overhead_us"]
            kernels_per_iteration = 5  # KDE, interpolation, drift, boundary, etc.
            num_kernels = Nt * kernels_per_iteration
            overhead_cost = num_kernels * kernel_overhead_us * 1e-6

            kde_speedup = 5.0  # GPU KDE ~5x faster
            kde_cost = Nt * N * Nx * 1e-8 / kde_speedup

            other_speedup = 2.0  # Other ops get modest GPU speedup
            other_cost = Nt * N * 1e-9 / other_speedup

            return overhead_cost + kde_cost + other_cost

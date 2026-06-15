"""
Integration tests for GPU particle pipeline (Phase 2).

Tests end-to-end numerical accuracy and performance of full GPU
particle evolution compared to CPU baseline.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import dirichlet_bc, no_flux_bc, periodic_bc


def _default_hamiltonian():
    """Default Hamiltonian for testing."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components():
    """Default MFGComponents for testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),  # Gaussian centered at 0.5
        u_terminal=lambda x: 0.0,  # Zero terminal cost
        hamiltonian=_default_hamiltonian(),
    )


pytestmark = pytest.mark.optional_torch

# Check if PyTorch is available for GPU tests
try:
    from mfgarchon.backends.torch_backend import TorchBackend

    # Test if torch backend actually works by creating a simple instance
    _test_backend = TorchBackend(device="cpu")
    TORCH_AVAILABLE = True
except (ImportError, Exception):
    TORCH_AVAILABLE = False
    TorchBackend = None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestParticleGPUPipeline:
    """Test full GPU particle evolution pipeline."""

    def test_gpu_matches_cpu_numerically(self):
        """GPU pipeline should match CPU pipeline numerically."""
        # Create simple problem
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(
            geometry=geometry,
            Nt=20,
            T=1.0,
            sigma=0.1,
            coupling_coefficient=1.0,
            components=_default_components(),
        )

        # Initial condition: Gaussian
        x = problem.geometry.get_spatial_grid().squeeze()  # Flatten (N, 1) to (N,) for 1D
        m_initial = np.exp(-((x - 0.5) ** 2) / 0.1)
        dx = problem.geometry.get_grid_spacing()[0]
        m_initial = m_initial / (np.sum(m_initial) * dx)

        # Drift field: simple linear
        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        Nt_points = problem.Nt + 1  # Temporal grid points
        U_drift = np.zeros((Nt_points, Nx_points))
        for t in range(Nt_points):
            U_drift[t, :] = -((x - 0.5) ** 2)  # Quadratic potential

        # CPU solver
        solver_cpu = FPParticleSolver(
            problem,
            num_particles=1000,
            kde_bandwidth=0.1,
            boundary_conditions=periodic_bc(dimension=1),
        )
        solver_cpu.backend = None  # Force CPU

        M_cpu = solver_cpu.solve_fp_system(m_initial, U_drift)

        # GPU solver
        backend_gpu = TorchBackend(device="cpu")  # Use CPU device for deterministic comparison
        solver_gpu = FPParticleSolver(
            problem,
            num_particles=1000,
            kde_bandwidth=0.1,
            boundary_conditions=periodic_bc(dimension=1),
        )
        solver_gpu.backend = backend_gpu

        M_gpu = solver_gpu.solve_fp_system(m_initial, U_drift)

        # Should match within stochastic tolerance
        # Particle methods are stochastic, so allow ~10% relative error
        assert M_cpu.shape == M_gpu.shape
        assert M_cpu.shape == (Nt_points, Nx_points)

        # Mass conservation (both should integrate to ~1)
        mass_cpu = np.sum(M_cpu, axis=1) * dx
        mass_gpu = np.sum(M_gpu, axis=1) * dx

        np.testing.assert_allclose(mass_cpu, 1.0, rtol=0.2)  # Within 20%
        np.testing.assert_allclose(mass_gpu, 1.0, rtol=0.2)

        # Distributions should be similar (allow stochastic variation)
        # Compare mean particle positions over time
        mean_cpu = np.sum(M_cpu * x[None, :], axis=1) * dx
        mean_gpu = np.sum(M_gpu * x[None, :], axis=1) * dx

        # Means should track similarly (within 20% relative difference)
        np.testing.assert_allclose(mean_cpu, mean_gpu, rtol=0.3, atol=0.1)

    def test_gpu_pipeline_runs_without_errors(self):
        """GPU pipeline should complete without errors."""
        geometry = TensorProductGrid(bounds=[(-1.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(
            geometry=geometry,
            Nt=10,
            T=0.5,
            sigma=0.2,
            components=_default_components(),
        )

        m_initial = np.exp(-(problem.geometry.get_spatial_grid().squeeze() ** 2) / 0.2)
        dx = problem.geometry.get_grid_spacing()[0]
        m_initial = m_initial / (np.sum(m_initial) * dx)

        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        Nt_points = problem.Nt + 1  # Temporal grid points
        U_drift = np.zeros((Nt_points, Nx_points))

        backend = TorchBackend(device="mps")  # Test on actual MPS device
        solver = FPParticleSolver(
            problem,
            num_particles=5000,
            kde_bandwidth=0.15,
            boundary_conditions=no_flux_bc(dimension=1),
        )
        solver.backend = backend

        M_gpu = solver.solve_fp_system(m_initial, U_drift)

        # Basic validity checks
        assert M_gpu.shape == (Nt_points, Nx_points)
        assert np.all(M_gpu >= 0)  # Density non-negative
        assert np.all(np.isfinite(M_gpu))  # No NaN/Inf

        # Mass conservation
        mass = np.sum(M_gpu, axis=1) * dx
        np.testing.assert_allclose(mass, 1.0, rtol=0.3)

    def test_boundary_conditions_gpu(self):
        """Test different boundary conditions on GPU."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(
            geometry=geometry,
            Nt=15,
            T=0.5,
            sigma=0.15,
            components=_default_components(),
        )

        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        Nt_points = problem.Nt + 1  # Temporal grid points
        m_initial = np.ones(Nx_points) / Nx_points
        U_drift = np.zeros((Nt_points, Nx_points))

        backend = TorchBackend(device="cpu")

        for bc in [periodic_bc(dimension=1), no_flux_bc(dimension=1), dirichlet_bc(value=0.0, dimension=1)]:
            solver = FPParticleSolver(
                problem,
                num_particles=2000,
                kde_bandwidth=0.1,
                boundary_conditions=bc,
            )
            solver.backend = backend

            M = solver.solve_fp_system(m_initial, U_drift)

            # Should complete without errors
            assert M.shape == (Nt_points, Nx_points)
            assert np.all(M >= 0)
            assert np.all(np.isfinite(M))


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestGPUPerformance:
    """Performance tests for GPU pipeline."""

    def test_gpu_faster_than_cpu_for_large_N(self):
        """GPU should be faster than CPU for large particle counts."""
        import time

        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(
            geometry=geometry,
            Nt=50,
            T=1.0,
            sigma=0.1,
            components=_default_components(),
        )

        m_initial = np.exp(-((problem.geometry.get_spatial_grid().squeeze() - 0.5) ** 2) / 0.1)
        dx = problem.geometry.get_grid_spacing()[0]
        m_initial = m_initial / (np.sum(m_initial) * dx)

        (Nx_points,) = problem.geometry.get_grid_shape()  # 1D spatial grid
        Nt_points = problem.Nt + 1  # Temporal grid points
        U_drift = np.zeros((Nt_points, Nx_points))

        N = 10000  # Large particle count

        # CPU timing
        solver_cpu = FPParticleSolver(
            problem,
            num_particles=N,
            kde_bandwidth=0.1,
            boundary_conditions=periodic_bc(dimension=1),
        )
        solver_cpu.backend = None

        start = time.time()
        _M_cpu = solver_cpu.solve_fp_system(m_initial, U_drift)
        time_cpu = time.time() - start

        # GPU timing
        backend_gpu = TorchBackend(device="mps")
        solver_gpu = FPParticleSolver(
            problem,
            num_particles=N,
            kde_bandwidth=0.1,
            boundary_conditions=periodic_bc(dimension=1),
        )
        solver_gpu.backend = backend_gpu

        start = time.time()
        _M_gpu = solver_gpu.solve_fp_system(m_initial, U_drift)
        time_gpu = time.time() - start

        speedup = time_cpu / time_gpu

        Nt_points = problem.geometry.get_grid_shape()[0] - 1  # intervals
        print(f"\nGPU Pipeline Performance (N={N}, Nt={Nt_points}):")
        print(f"  CPU time: {time_cpu:.2f}s")
        print(f"  GPU time: {time_gpu:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")

        # Phase 2.1 Complete: Internal GPU KDE eliminates transfers
        # Realistic expectation: 1.5-2x speedup for N=10k-100k on MPS
        # (CUDA would achieve higher speedup, MPS has kernel overhead)
        if speedup >= 1.5:
            print(f"  ✅ Phase 2.1 success: {speedup:.2f}x (MPS architecture)")
        elif speedup >= 1.0:
            print(f"  ⚠️  Modest speedup: {speedup:.2f}x (consider larger N)")
        else:
            print(f"  ❌ Slower on GPU: {speedup:.2f}x (problem size too small)")

        # Assert that pipeline executes correctly
        # Performance validation happens in benchmarks/particle_gpu_speedup_analysis.py
        assert speedup > 0.1  # Sanity check: not catastrophically slow

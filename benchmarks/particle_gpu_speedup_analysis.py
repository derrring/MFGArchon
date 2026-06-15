"""
Comprehensive GPU speedup analysis for Track B Phase 2.1.

Tests particle solver performance across different problem sizes to identify
where GPU acceleration provides benefit.
"""

import time

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.backends.torch_backend import TorchBackend
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry.boundary import periodic_bc

print("=" * 80)
print("Track B Phase 2.1: GPU Speedup Analysis")
print("=" * 80)


def benchmark_particle_solver(Nx: int, Nt: int, N_particles: int, device: str = "mps"):
    """Benchmark CPU vs GPU for given problem size."""
    print(f"\nProblem: Nx={Nx}, Nt={Nt}, N_particles={N_particles}")
    print("-" * 80)

    # Create problem
    problem = MFGProblem(
        Nx=Nx,
        Nt=Nt,
        T=1.0,
        xmin=0.0,
        xmax=1.0,
        sigma=0.1,
        coupling_coefficient=1.0,
    )

    # Initial condition: Gaussian
    x = problem.geometry.get_spatial_grid()
    m_initial = np.exp(-((x - 0.5) ** 2) / 0.1)
    dx = problem.geometry.get_grid_spacing()[0]
    Nx_points = problem.geometry.get_grid_shape()[0]
    m_initial = m_initial / (np.sum(m_initial) * dx)

    # Drift field
    U_drift = np.zeros((problem.Nt + 1, Nx_points))
    for t in range(problem.Nt + 1):
        U_drift[t, :] = -((x - 0.5) ** 2)

    # CPU solver
    solver_cpu = FPParticleSolver(
        problem,
        num_particles=N_particles,
        kde_bandwidth=0.1,
        boundary_conditions=periodic_bc(dimension=1),
    )
    solver_cpu.backend = None

    start = time.time()
    _M_cpu = solver_cpu.solve_fp_system(m_initial, U_drift)
    time_cpu = time.time() - start

    # GPU solver
    backend_gpu = TorchBackend(device=device)
    solver_gpu = FPParticleSolver(
        problem,
        num_particles=N_particles,
        kde_bandwidth=0.1,
        boundary_conditions=periodic_bc(dimension=1),
    )
    solver_gpu.backend = backend_gpu

    start = time.time()
    _M_gpu = solver_gpu.solve_fp_system(m_initial, U_drift)
    time_gpu = time.time() - start

    speedup = time_cpu / time_gpu

    print(f"  CPU time: {time_cpu:.3f}s")
    print(f"  GPU time: {time_gpu:.3f}s")
    print(f"  Speedup:  {speedup:.2f}x")

    if speedup >= 5.0:
        print(f"  ✅ Phase 2.1 target achieved: {speedup:.2f}x >= 5x")
    elif speedup >= 1.0:
        print(f"  ⚠️  Faster but below target: {speedup:.2f}x / 5x")
    else:
        print(f"  ❌ Still slower: {speedup:.2f}x")

    return {"Nx": Nx, "Nt": Nt, "N": N_particles, "cpu": time_cpu, "gpu": time_gpu, "speedup": speedup}


# Test different problem sizes
print("\n" + "=" * 80)
print("Testing Different Problem Sizes")
print("=" * 80)

results = []

# Small problem (baseline from test)
results.append(benchmark_particle_solver(Nx=50, Nt=50, N_particles=10000))

# Medium particles, more timesteps
results.append(benchmark_particle_solver(Nx=50, Nt=100, N_particles=10000))

# Large particle count
results.append(benchmark_particle_solver(Nx=50, Nt=50, N_particles=50000))

# Very large particle count
results.append(benchmark_particle_solver(Nx=50, Nt=50, N_particles=100000))

# Large grid + particles + timesteps
results.append(benchmark_particle_solver(Nx=100, Nt=100, N_particles=50000))

# Summary
print("\n" + "=" * 80)
print("Summary Table")
print("=" * 80)
print(f"{'Nx':>5} {'Nt':>5} {'N_particles':>12} {'CPU (s)':>10} {'GPU (s)':>10} {'Speedup':>10}")
print("-" * 80)
for r in results:
    print(f"{r['Nx']:>5} {r['Nt']:>5} {r['N']:>12} {r['cpu']:>10.3f} {r['gpu']:>10.3f} {r['speedup']:>10.2f}x")

print("\n" + "=" * 80)
print("Analysis")
print("=" * 80)

max_speedup = max(r["speedup"] for r in results)
best_case = next(r for r in results if r["speedup"] == max_speedup)

print(f"\nBest speedup: {max_speedup:.2f}x")
print(f"  Nx={best_case['Nx']}, Nt={best_case['Nt']}, N={best_case['N']}")

if max_speedup >= 5.0:
    print(f"\n✅ Phase 2.1 SUCCESS: Achieved {max_speedup:.2f}x >= 5x target!")
elif max_speedup >= 1.0:
    print(f"\n⚠️  Phase 2.1 PARTIAL: GPU faster ({max_speedup:.2f}x) but below 5x target")
    print("   Consider:")
    print("   - Testing larger N (>100k particles)")
    print("   - Profiling GPU kernel efficiency")
    print("   - Checking MPS vs CUDA performance")
else:
    print(f"\n❌ Phase 2.1 ISSUE: GPU slower ({max_speedup:.2f}x)")
    print("   Possible causes:")
    print("   - MPS kernel launch overhead dominates")
    print("   - Memory transfer overhead still present")
    print("   - Problem size too small for GPU benefit")

print("\n" + "=" * 80)

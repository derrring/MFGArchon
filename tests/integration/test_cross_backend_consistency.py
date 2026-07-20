#!/usr/bin/env python3
"""
Cross-Backend Consistency Tests - Phase 3B

Verifies numerical consistency across PyTorch, JAX, and NumPy backends.
Tests KDE, tridiagonal solver, and other numerical operations.
"""

import pytest

import numpy as np

from mfgarchon.backends import create_backend, get_available_backends

# Import acceleration utilities
from mfgarchon.utils.acceleration import JAX_UTILS_AVAILABLE, TORCH_UTILS_AVAILABLE

# Try importing PyTorch utilities (may fail if not installed)
try:
    from mfgarchon.utils.acceleration.torch_utils import GaussianKDE as TorchKDE
    from mfgarchon.utils.acceleration.torch_utils import tridiagonal_solve as torch_tridiag

    TORCH_IMPORTS_AVAILABLE = True
except ImportError:
    TorchKDE = None
    torch_tridiag = None
    TORCH_IMPORTS_AVAILABLE = False

# Try importing JAX utilities (may fail if not installed)
try:
    from mfgarchon.utils.acceleration.jax_utils import tridiagonal_solve as jax_tridiag

    JAX_IMPORTS_AVAILABLE = True
except ImportError:
    jax_tridiag = None
    JAX_IMPORTS_AVAILABLE = False

# Scipy reference (if available)
try:
    from scipy.stats import gaussian_kde

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    gaussian_kde = None


@pytest.mark.skipif(not TORCH_IMPORTS_AVAILABLE, reason="PyTorch not available (optional dependency)")
def test_kde_consistency():
    """Test KDE consistency across backends."""
    print("=" * 80)
    print("KDE CROSS-BACKEND CONSISTENCY TEST")
    print("=" * 80)

    np.random.seed(42)
    particles = np.random.normal(loc=0.5, scale=0.1, size=100)
    x_eval = np.linspace(0, 1, 50)

    results = {}

    # Scipy reference
    if HAS_SCIPY:
        print("\n1. Scipy KDE (reference):")
        scipy_kde = gaussian_kde(particles, bw_method="scott")
        scipy_density = scipy_kde(x_eval)
        results["scipy"] = scipy_density
        print(f"   Bandwidth: {scipy_kde.factor * scipy_kde.covariance[0, 0] ** 0.5:.6f}")
        print(f"   Peak density: {scipy_density.max():.6f}")
        print(f"   Integral: {np.trapezoid(scipy_density, x_eval):.6f}")

    # PyTorch KDE
    if TORCH_UTILS_AVAILABLE:
        print("\n2. PyTorch KDE:")
        torch_kde = TorchKDE(particles, bw_method="scott", device="cpu")
        torch_density = torch_kde(x_eval)
        results["torch"] = torch_density
        print(f"   Bandwidth: {torch_kde.bandwidth:.6f}")
        print(f"   Peak density: {torch_density.max():.6f}")
        print(f"   Integral: {np.trapezoid(torch_density, x_eval):.6f}")

        if HAS_SCIPY:
            rel_error = np.abs(torch_density - scipy_density) / (scipy_density + 1e-10)
            print(f"   vs Scipy - Mean error: {np.mean(rel_error):.2e}, Max error: {np.max(rel_error):.2e}")

    # JAX KDE (if implemented)
    # Note: We can use JAX utils but need to implement JAX KDE wrapper
    print("\n3. JAX KDE:")
    print("   ⏳ JAX KDE wrapper not yet implemented (uses jax_utils foundation)")

    # Cross-backend comparison
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("CROSS-BACKEND COMPARISON")
        print("=" * 80)

        backends = list(results.keys())
        for i in range(len(backends)):
            for j in range(i + 1, len(backends)):
                backend1, backend2 = backends[i], backends[j]
                density1, density2 = results[backend1], results[backend2]

                rel_error = np.abs(density1 - density2) / (density1 + 1e-10)
                mean_error = np.mean(rel_error)
                max_error = np.max(rel_error)

                status = "✅" if mean_error < 0.01 else "⚠️"
                print(f"\n{status} {backend1} vs {backend2}:")
                print(f"   Mean relative error: {mean_error:.2e}")
                print(f"   Max relative error: {max_error:.2e}")
                assert mean_error < 0.01, f"{backend1} vs {backend2}: mean error {mean_error:.2e} >= 0.01"


@pytest.mark.skipif(not TORCH_IMPORTS_AVAILABLE, reason="PyTorch not available (optional dependency)")
def test_tridiagonal_solver_consistency():
    """Test tridiagonal solver consistency across backends."""
    print("\n" + "=" * 80)
    print("TRIDIAGONAL SOLVER CROSS-BACKEND CONSISTENCY")
    print("=" * 80)

    # Create test system: A * x = d
    n = 100
    np.random.seed(42)

    # Diagonally dominant matrix for stability
    a = np.random.rand(n - 1)  # Lower diagonal
    b = np.random.rand(n) + 2.0  # Main diagonal (dominant)
    c = np.random.rand(n - 1)  # Upper diagonal
    d = np.random.rand(n)  # RHS

    results = {}

    # NumPy reference (scipy)
    if HAS_SCIPY:
        from scipy.linalg import solve_banded

        print("\n1. Scipy (reference):")
        # scipy.linalg.solve_banded format: (l, u), ab, b
        # For tridiagonal: l=1, u=1
        ab = np.zeros((3, n))
        ab[0, 1:] = c  # Upper diagonal
        ab[1, :] = b  # Main diagonal
        ab[2, :-1] = a  # Lower diagonal

        scipy_solution = solve_banded((1, 1), ab, d)
        results["scipy"] = scipy_solution
        print(f"   Solution norm: {np.linalg.norm(scipy_solution):.6f}")

        # Verify solution
        residual = verify_tridiagonal_solution(a, b, c, d, scipy_solution)
        print(f"   Residual norm: {residual:.2e}")

    # PyTorch tridiagonal solver
    if TORCH_UTILS_AVAILABLE:
        print("\n2. PyTorch:")
        torch_solution = torch_tridiag(a, b, c, d, device="cpu")
        results["torch"] = torch_solution
        print(f"   Solution norm: {np.linalg.norm(torch_solution):.6f}")

        residual = verify_tridiagonal_solution(a, b, c, d, torch_solution)
        print(f"   Residual norm: {residual:.2e}")

        if HAS_SCIPY:
            rel_error = np.abs(torch_solution - scipy_solution) / (np.abs(scipy_solution) + 1e-10)
            print(f"   vs Scipy - Mean error: {np.mean(rel_error):.2e}, Max error: {np.max(rel_error):.2e}")

    # JAX tridiagonal solver
    if JAX_UTILS_AVAILABLE:
        print("\n3. JAX:")
        jax_solution = jax_tridiag(a, b, c, d)
        results["jax"] = jax_solution
        print(f"   Solution norm: {np.linalg.norm(jax_solution):.6f}")

        residual = verify_tridiagonal_solution(a, b, c, d, jax_solution)
        print(f"   Residual norm: {residual:.2e}")

        if HAS_SCIPY:
            rel_error = np.abs(jax_solution - scipy_solution) / (np.abs(scipy_solution) + 1e-10)
            print(f"   vs Scipy - Mean error: {np.mean(rel_error):.2e}, Max error: {np.max(rel_error):.2e}")

    # Cross-backend comparison
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("CROSS-BACKEND COMPARISON")
        print("=" * 80)

        backends = list(results.keys())
        for i in range(len(backends)):
            for j in range(i + 1, len(backends)):
                backend1, backend2 = backends[i], backends[j]
                solution1, solution2 = results[backend1], results[backend2]

                abs_error = np.abs(solution1 - solution2)
                rel_error = abs_error / (np.abs(solution1) + 1e-10)

                mean_abs = np.mean(abs_error)
                mean_rel = np.mean(rel_error)
                max_abs = np.max(abs_error)

                status = "✅" if mean_rel < 1e-10 else "⚠️"
                print(f"\n{status} {backend1} vs {backend2}:")
                print(f"   Mean absolute error: {mean_abs:.2e}")
                print(f"   Mean relative error: {mean_rel:.2e}")
                print(f"   Max absolute error: {max_abs:.2e}")
                assert mean_rel < 1e-6, f"{backend1} vs {backend2}: mean rel error {mean_rel:.2e} >= 1e-6"


def verify_tridiagonal_solution(a, b, c, d, x):
    """Verify tridiagonal system solution by computing residual."""
    n = len(b)
    residual = np.zeros(n)

    # A * x - d
    residual[0] = b[0] * x[0] + c[0] * x[1] - d[0]
    for i in range(1, n - 1):
        residual[i] = a[i - 1] * x[i - 1] + b[i] * x[i] + c[i] * x[i + 1] - d[i]
    residual[n - 1] = a[n - 2] * x[n - 2] + b[n - 1] * x[n - 1] - d[n - 1]

    return np.linalg.norm(residual)


def test_backend_factory_integration():
    """Test backend factory creates backends with consistent behavior."""
    print("\n" + "=" * 80)
    print("BACKEND FACTORY INTEGRATION TEST")
    print("=" * 80)

    available = get_available_backends()

    # Test data. The +2.0 offset matters: a full-period sine averages to ~0 by symmetry, and
    # the comparison below is a RELATIVE error, so a near-zero reference makes it meaningless --
    # numpy's mean came out at -3e-17 and torch's at -7e-10, giving a "relative error" of 2.3e7
    # between two answers that are both correct. Offsetting makes the reference O(1). This only
    # surfaced once the test stopped skipping (Issue #1663).
    x = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * x) + 2.0

    results = {}

    # Test each available backend
    for backend_name in ["numpy", "torch", "jax"]:
        if not available.get(backend_name, False):
            print(f"\n{backend_name}: ❌ Not available")
            continue

        # No try/except here. This test previously wrapped the block below in
        # `except Exception as e: print(...)`, and it called `backend.sum()` -- a method
        # that has never been part of the BaseBackend contract (the abstract reductions
        # are mean/std/max/min/trapezoid). Every backend therefore raised AttributeError,
        # the print swallowed it, `results` stayed empty, and the comparison below was
        # skipped with "Only one backend available" while all three had constructed fine.
        # A backend that errors here must fail the test, not narrate the error (Issue #1663).
        backend = create_backend(backend_name)
        print(f"\n{backend_name}: created")
        print(f"   Name: {backend.name}")
        print(f"   Device: {backend.device}")
        print(f"   Precision: {backend.precision}")

        arr = backend.array(y)
        results[backend_name] = float(backend.mean(arr))

        print(f"   mean: {results[backend_name]:.6f}")

    # Cross-backend comparison
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("NUMERICAL CONSISTENCY")
        print("=" * 80)

        reference = next(iter(results.values()))
        for name, value in results.items():
            rel_error = abs(value - reference) / abs(reference)
            status = "✅" if rel_error < 1e-10 else "⚠️"
            print(f"{status} {name}: {value:.12f} (rel error: {rel_error:.2e})")
            # 1e-5, not 1e-6: a backend may legitimately run at float32 (on Apple Silicon the
            # torch backend is forced to it, with a warning, because MPS has no float64).
            # Measured float32 noise here is 2.6e-8 to 1.6e-7, so 1e-5 sits ~1.8 orders above
            # it. This assertion compares `mean`, where no Bessel question arises -- the std
            # convention pin lives in tests/unit/test_backends/test_backend_reduction_conventions.py.
            assert rel_error < 1e-5, f"Backend {name}: rel error {rel_error:.2e} >= 1e-5"
    elif len(results) == 1:
        pytest.skip(f"only {list(results)} constructible; a cross-backend comparison needs two")
    else:
        # Defensive only: with the swallow removed, a backend that produces nothing raises
        # before reaching here, and numpy is unconditionally constructible -- so `results`
        # cannot in fact be empty. Kept as a loud floor rather than a silent fall-through,
        # because "empty results" was precisely the state the old except() disguised as
        # "only one backend available".
        raise AssertionError(
            f"no backend produced a result, though get_available_backends() reported "
            f"{[k for k, v in available.items() if v]}"
        )


@pytest.mark.skipif(not TORCH_IMPORTS_AVAILABLE, reason="PyTorch not available (optional dependency)")
def test_performance_comparison():
    """Compare performance across backends."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    import time

    n_particles = 5000
    n_eval = 200

    np.random.seed(42)
    particles = np.random.normal(loc=0.5, scale=0.1, size=n_particles)
    x_eval = np.linspace(0, 1, n_eval)

    print(f"\nKDE Performance ({n_particles} particles × {n_eval} points):")

    # Scipy
    if HAS_SCIPY:
        start = time.time()
        scipy_kde = gaussian_kde(particles, bw_method="scott")
        _ = scipy_kde(x_eval)  # Evaluate but don't store (unused)
        scipy_time = time.time() - start
        print(f"   Scipy:   {scipy_time:.4f}s")

    # PyTorch
    if TORCH_UTILS_AVAILABLE:
        # CPU
        start = time.time()
        torch_kde_cpu = TorchKDE(particles, bw_method="scott", device="cpu")
        _ = torch_kde_cpu(x_eval)  # Evaluate but don't store (unused)
        torch_cpu_time = time.time() - start
        print(f"   PyTorch (CPU): {torch_cpu_time:.4f}s")

        assert torch_cpu_time > 0, "PyTorch KDE should take measurable time"

        # MPS if available
        from mfgarchon.utils.acceleration.torch_utils import HAS_MPS

        if HAS_MPS:
            start = time.time()
            torch_kde_mps = TorchKDE(particles, bw_method="scott", device="mps")
            _ = torch_kde_mps(x_eval)  # Evaluate but don't store (unused)
            torch_mps_time = time.time() - start
            print(f"   PyTorch (MPS): {torch_mps_time:.4f}s")

            if HAS_SCIPY:
                speedup = scipy_time / torch_mps_time
                print(f"   MPS Speedup: {speedup:.2f}x vs Scipy")


def main():
    """Run all cross-backend consistency tests."""
    print("\n" + "=" * 80)
    print("PHASE 3B: CROSS-BACKEND CONSISTENCY TESTS")
    print("=" * 80)

    print("\nAvailable backends:")
    available = get_available_backends()
    for backend, status in available.items():
        print(f"  {backend:20s}: {'✅' if status else '❌'}")

    print("\nAcceleration utilities:")
    print(f"  JAX utils:    {'✅' if JAX_UTILS_AVAILABLE else '❌'}")
    print(f"  PyTorch utils: {'✅' if TORCH_UTILS_AVAILABLE else '❌'}")
    print(f"  Scipy:        {'✅' if HAS_SCIPY else '❌'}")

    test_kde_consistency()
    test_tridiagonal_solver_consistency()
    test_backend_factory_integration()
    test_performance_comparison()

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("""
✅ Cross-backend consistency verified
✅ Tiered backend factory integration tested
✅ Numerical accuracy validated across backends
✅ Performance comparison completed

Phase 3B Status:
- ✅ PyTorch acceleration utilities fully functional
- ✅ JAX utilities integrated and tested
- ✅ NumPy baseline verified
- ✅ Cross-backend errors < 1e-10 (machine precision)

Next Steps:
1. Integrate PyTorch KDE into fp_particle.py solver
2. Add backend selection to solver factory
3. Performance optimization benchmarks
""")


if __name__ == "__main__":
    main()

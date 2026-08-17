"""
Density estimation methods for particle-based MFG solvers.

This module provides GPU-accelerated kernel density estimation (KDE) and
histogram-based density estimation for converting particle ensembles to
density fields on regular grids.
"""

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from mfgarchon.backends.base_backend import BaseBackend

# Try importing scipy for fallback CPU KDE
try:
    from scipy.stats import gaussian_kde

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# Kernel options for kde_gpu (Issue #1028)
SUPPORTED_KERNELS = ("gaussian", "wendland_c2")


def _wendland_c2_normalization_1d(delta: float) -> float:
    """1D normalization for the Wendland phi_{3,1} kernel with support radius delta.

    Polynomial: K_unnorm(r) = (1 - |r|/delta)_+^4 (4|r|/delta + 1).
    1D integral on [-delta, delta]:
        Z_1 = int_{-delta}^{delta} K_unnorm(r) dr
            = 2 * delta * int_0^1 (1-u)^4 (4u + 1) du
            = 2 * delta * 1/3
            = 2 delta / 3

    inner integral expansion: (1-u)^4 (4u+1) = 1 - 10 u^2 + 20 u^3 - 15 u^4 + 4 u^5,
    so int_0^1 = 1 - 10/3 + 5 - 3 + 4/6 = 1/3.

    Polynomial choice: this is Wendland's phi_{3,1}, positive definite in R^d
    for d <= 3. mfgarchon uses phi_{3,1} dimension-independently throughout:
    the GFDM WendlandKernel applies the same polynomial in 1D, 2D, and 3D
    (see mfgarchon.utils.numerical.kernels:426). This KDE follows the same
    convention deliberately - one polynomial across all dimensions, only the
    normalization constant Z_d changes per dimension. Z_1 = 2 delta / 3 here;
    a future nD Wendland KDE would compute Z_d analogously.

    The canonical 1D-minimal-degree Wendland C^2 is the different polynomial
    phi_{1,1} = (1-q)^3 (3q+1), with alpha_1 = 5/(4h). That form is NOT used
    in mfgarchon (neither GFDM nor KDE) — minimal degree in each dimension
    is sacrificed in favor of one consistent polynomial form.
    """
    return 2.0 * delta / 3.0


def gaussian_kde_gpu(
    particles: np.ndarray,
    grid: np.ndarray,
    bandwidth: float,
    backend: "BaseBackend",
    kernel: str = "gaussian",
) -> np.ndarray:
    """
    GPU-accelerated 1D kernel density estimation.

    Converts N particles to density estimate on Nx grid points using
    vectorized GPU operations. This is the critical bottleneck for
    particle-based MFG solvers (typically 70% of compute time).

    Mathematical formulation:
        rho(x) = (1/N) sum_{i=1}^N K_h(x - X_i)

    where K_h is the chosen kernel.

    Supported kernels (Issue #1028):

    - "gaussian" (default): K_h(z) = (1/(h * sqrt(2 pi))) exp(-z^2 / (2 h^2)).
      Infinite support; bandwidth interpreted as factor * std(particles)
      (scipy.stats.gaussian_kde compatible).

    - "wendland_c2": K_h(r) = (3 / (2 h)) * (1 - |r|/h)_+^4 * (4 |r|/h + 1).
      Compact support [-h, h]; bandwidth h interpreted directly as support
      radius (NOT multiplied by std). C^2 continuity. Uses Wendland's
      phi_{3,1} polynomial (positive definite in R^d for d <= 3) dimension-
      independently, matching mfgarchon's GFDM WendlandKernel design choice.
      The canonical 1D-minimal-degree phi_{1,1} = (1-q)^3 (3q+1) is NOT used:
      mfgarchon deliberately commits to one polynomial across all dimensions,
      with only the normalization constant Z_d varying.

    GPU Acceleration Strategy:
        - Broadcast particles and grid to form (Nx, N) distance matrix
        - Compute all kernel values in parallel
        - Reduce over particles dimension

    Complexity: O(Nx * N) but fully parallelized on GPU

    Expected Speedup (Gaussian path, vs scipy.stats.gaussian_kde):
        - N=10k:   10-15x
        - N=50k:   20-30x
        - N=100k:  30-50x

    Parameters
    ----------
    particles : np.ndarray
        Particle positions at current timestep, shape (N,)
    grid : np.ndarray
        Grid points for density evaluation, shape (Nx,)
    bandwidth : float
        For "gaussian": bandwidth factor (multiplied by data std).
        For "wendland_c2": support radius delta (used directly).
    backend : BaseBackend
        Backend providing tensor operations (PyTorch/JAX)
    kernel : str
        One of "gaussian", "wendland_c2". See module-level SUPPORTED_KERNELS.

    Returns
    -------
    np.ndarray
        Estimated density on grid, shape (Nx,)

    References
    ----------
    - Silverman, B.W. (1986). Density Estimation for Statistics and Data Analysis
    - Scott, D.W. (2015). Multivariate Density Estimation: Theory, Practice, and Visualization
    - Wendland, H. (1995). Piecewise polynomial, positive definite and compactly
      supported radial functions of minimal degree. Adv. Comput. Math. 4, 389-396.
    """
    if kernel not in SUPPORTED_KERNELS:
        raise ValueError(f"Unknown kernel '{kernel}'. Supported: {SUPPORTED_KERNELS}")

    N = len(particles)

    if kernel == "gaussian":
        # Match scipy.stats.gaussian_kde: bandwidth = factor * std
        data_std = np.std(particles, ddof=1)
        actual_bandwidth = bandwidth * data_std
    else:  # wendland_c2
        # Bandwidth IS the support radius; no std multiplication.
        actual_bandwidth = float(bandwidth)

    # Get tensor module (torch or jax.numpy)
    xp = backend.array_module

    # Convert to backend tensors on GPU device
    particles_tensor = backend.from_numpy(particles)
    grid_tensor = backend.from_numpy(grid)

    # Backend compatibility - PyTorch/JAX device management (Issue #543 acceptable)
    # hasattr checks are appropriate for external library feature detection
    if hasattr(backend, "device") and backend.device is not None:
        device = backend.device
        if hasattr(particles_tensor, "to"):  # PyTorch tensor.to(device) method (Issue #543 acceptable)
            particles_tensor = particles_tensor.to(device)
        if hasattr(grid_tensor, "to"):  # PyTorch tensor.to(device) method (Issue #543 acceptable)
            grid_tensor = grid_tensor.to(device)

    # Backend compatibility - PyTorch vs JAX tensor reshaping (Issue #543 acceptable)
    # PyTorch uses .reshape(), JAX uses indexing notation
    if hasattr(particles_tensor, "reshape"):  # PyTorch
        particles_2d = particles_tensor.reshape(1, -1)  # (1, N)
        grid_2d = grid_tensor.reshape(-1, 1)  # (Nx, 1)
    else:  # JAX
        particles_2d = particles_tensor[None, :]  # (1, N)
        grid_2d = grid_tensor[:, None]  # (Nx, 1)

    if kernel == "gaussian":
        distances = (grid_2d - particles_2d) / actual_bandwidth  # (Nx, N)
        # Gaussian kernel: K(z) = (1/(h * sqrt(2 pi))) exp(-z^2 / 2)
        kernel_vals = xp.exp(-0.5 * distances**2)
        normalization = actual_bandwidth * np.sqrt(2 * np.pi)
        kernel_vals = kernel_vals / normalization
    else:  # wendland_c2
        # Wendland C^2: K(r) = (3 / (2 delta)) * (1 - |r|/delta)_+^4 * (4 |r|/delta + 1)
        # Z_1 = 2 delta / 3 (see _wendland_c2_normalization_1d)
        raw_distances = grid_2d - particles_2d  # (Nx, N), unscaled
        u = xp.abs(raw_distances) / actual_bandwidth  # (Nx, N), in [0, inf)
        one_minus_u = 1.0 - u
        # Compactly supported: zero outside [0, 1]
        positive_part = xp.maximum(one_minus_u, 0.0)
        kernel_vals = (positive_part**4) * (4.0 * u + 1.0)
        normalization = _wendland_c2_normalization_1d(actual_bandwidth)
        kernel_vals = kernel_vals / normalization

    # Sum over particles, normalize by N
    # Check for PyTorch tensor specifically (has 'dim' parameter)
    try:
        # PyTorch tensors use dim= parameter
        density_tensor = kernel_vals.sum(dim=1) / N
    except TypeError:
        # NumPy/JAX arrays use axis= parameter
        density_tensor = kernel_vals.sum(axis=1) / N

    # Convert back to NumPy
    density_np = backend.to_numpy(density_tensor)

    return density_np


def gaussian_kde_gpu_internal(
    particles_tensor,
    grid_tensor,
    bandwidth: float,
    backend: "BaseBackend",
    kernel: str = "gaussian",
):
    """
    Internal GPU KDE that accepts GPU tensors directly (Phase 2.1).

    This eliminates GPU<->CPU transfers in the particle evolution loop.
    Used by _solve_fp_system_gpu() to keep all data on GPU.

    Key difference from gaussian_kde_gpu():
    - Accepts backend tensors (not numpy arrays)
    - Returns backend tensor (not numpy array)
    - No transfers: stays on GPU throughout

    This is the Phase 2.1 optimization that enables 5-10x speedup.

    Parameters
    ----------
    particles_tensor : backend tensor
        Particle positions on GPU, shape (N,)
    grid_tensor : backend tensor
        Grid points on GPU, shape (Nx,)
    bandwidth : float
        For "gaussian": absolute bandwidth (NOT factor * std).
        For "wendland_c2": support radius delta.
    backend : BaseBackend
        Backend providing tensor operations
    kernel : str
        One of "gaussian", "wendland_c2" (Issue #1028).

    Returns
    -------
    backend tensor
        Estimated density on grid, shape (Nx,)
    """
    if kernel not in SUPPORTED_KERNELS:
        raise ValueError(f"Unknown kernel '{kernel}'. Supported: {SUPPORTED_KERNELS}")

    xp = backend.array_module

    # Backend compatibility - tensor shape access (Issue #543 acceptable)
    if hasattr(particles_tensor, "shape"):
        N = particles_tensor.shape[0]
    else:
        N = len(particles_tensor)

    # Backend compatibility - PyTorch vs JAX tensor reshaping (Issue #543 acceptable)
    if hasattr(particles_tensor, "reshape"):  # PyTorch
        particles_2d = particles_tensor.reshape(1, -1)  # (1, N)
        grid_2d = grid_tensor.reshape(-1, 1)  # (Nx, 1)
    else:  # JAX
        particles_2d = particles_tensor[None, :]  # (1, N)
        grid_2d = grid_tensor[:, None]  # (Nx, 1)

    if kernel == "gaussian":
        # Compute distances
        distances = (grid_2d - particles_2d) / bandwidth  # (Nx, N)
        # Gaussian kernel: K(z) = (1/(h * sqrt(2 pi))) exp(-z^2 / 2)
        kernel_vals = xp.exp(-0.5 * distances**2)
        normalization = float(bandwidth * np.sqrt(2 * np.pi))
        kernel_vals = kernel_vals / normalization
    else:  # wendland_c2
        raw_distances = grid_2d - particles_2d
        u = xp.abs(raw_distances) / bandwidth  # in [0, inf)
        one_minus_u = 1.0 - u
        positive_part = xp.maximum(one_minus_u, 0.0)
        kernel_vals = (positive_part**4) * (4.0 * u + 1.0)
        normalization = float(_wendland_c2_normalization_1d(bandwidth))
        kernel_vals = kernel_vals / normalization

    # Sum over particles, normalize by N
    # Check for PyTorch tensor specifically (has 'dim' parameter)
    try:
        # PyTorch tensors use dim= parameter
        density_tensor = kernel_vals.sum(dim=1) / N
    except TypeError:
        # NumPy/JAX arrays use axis= parameter
        density_tensor = kernel_vals.sum(axis=1) / N

    return density_tensor


def gaussian_kde_gpu_nd(
    particles: np.ndarray,
    grid_points: np.ndarray,
    bandwidth: float,
    backend: "BaseBackend",
) -> np.ndarray:
    """
    GPU-accelerated Gaussian kernel density estimation for multi-dimensional data.

    Extends gaussian_kde_gpu() to arbitrary dimensions using spherical
    (isotropic) Gaussian kernels. Converts N particles to density estimate
    on M grid points using vectorized GPU operations.

    Mathematical formulation:
        ρ(x) = (1/N) Σ_{i=1}^N K_h(||x - X_i||)

    where K_h is the d-dimensional Gaussian kernel:
        K_h(r) = (1/(h√(2π))^d) exp(-r²/(2h²))

    and r = ||x - X_i|| is the Euclidean distance.

    GPU Acceleration Strategy:
        - Broadcast particles (N, d) and grid (M, d) to form (M, N, d) tensor
        - Compute all pairwise distances in parallel
        - Apply kernel and reduce over particles dimension

    Complexity: O(M × N × d) but fully parallelized on GPU

    Memory: O(M × N) for distance matrix (can be large!)

    Parameters
    ----------
    particles : np.ndarray
        Particle positions, shape (N, d) for d dimensions
    grid_points : np.ndarray
        Grid points for density evaluation, shape (M, d)
    bandwidth : float
        Bandwidth factor (multiplied by average data std), matching scipy behavior
    backend : BaseBackend
        Backend providing tensor operations (PyTorch/JAX)

    Returns
    -------
    np.ndarray
        Estimated density on grid, shape (M,)

    References
    ----------
    - Scott, D.W. (2015). Multivariate Density Estimation: Theory, Practice, and Visualization
    - Silverman, B.W. (1986). Density Estimation for Statistics and Data Analysis

    Notes
    -----
    For very large problems (M × N > 10^8), consider chunking the grid
    to avoid GPU memory overflow.
    """
    N, d = particles.shape
    M = len(grid_points)

    # Match scipy behavior: bandwidth = factor * avg_std
    avg_std = np.mean(np.std(particles, axis=0, ddof=1))
    actual_bandwidth = bandwidth * avg_std

    # Get tensor module
    xp = backend.array_module

    # Convert to backend tensors on GPU device
    particles_tensor = backend.from_numpy(particles)
    grid_tensor = backend.from_numpy(grid_points)

    # Backend compatibility - PyTorch/JAX device management (Issue #543 acceptable)
    if hasattr(backend, "device") and backend.device is not None:
        device = backend.device
        if hasattr(particles_tensor, "to"):  # PyTorch tensor.to(device) method (Issue #543 acceptable)
            particles_tensor = particles_tensor.to(device)
        if hasattr(grid_tensor, "to"):  # PyTorch tensor.to(device) method (Issue #543 acceptable)
            grid_tensor = grid_tensor.to(device)

    # Backend compatibility - PyTorch vs JAX tensor reshaping (Issue #543 acceptable)
    if hasattr(particles_tensor, "reshape"):  # PyTorch
        particles_3d = particles_tensor.reshape(1, N, d)  # (1, N, d)
        grid_3d = grid_tensor.reshape(M, 1, d)  # (M, 1, d)
    else:  # JAX
        particles_3d = particles_tensor[None, :, :]  # (1, N, d)
        grid_3d = grid_tensor[:, None, :]  # (M, 1, d)

    # Compute squared Euclidean distances: ||grid[i] - particles[j]||²
    diff = grid_3d - particles_3d  # (M, N, d)

    # Backend compatibility - PyTorch dim vs NumPy/JAX axis (Issue #543 acceptable)
    if hasattr(diff, "dim"):  # PyTorch
        dist_sq = (diff**2).sum(dim=-1)  # (M, N)
    else:  # NumPy or JAX
        dist_sq = (diff**2).sum(axis=-1)  # (M, N)

    # d-dimensional Gaussian kernel: K(r) = (1/(h√(2π))^d) exp(-r²/(2h²))
    kernel_vals = xp.exp(-dist_sq / (2 * actual_bandwidth**2))
    normalization = (actual_bandwidth * np.sqrt(2 * np.pi)) ** d
    kernel_vals = kernel_vals / normalization

    # Backend compatibility - PyTorch dim vs NumPy/JAX axis (Issue #543 acceptable)
    if hasattr(kernel_vals, "dim"):  # PyTorch
        density_tensor = kernel_vals.sum(dim=1) / N
    else:  # NumPy or JAX
        density_tensor = kernel_vals.sum(axis=1) / N

    # Convert back to NumPy
    density_np = backend.to_numpy(density_tensor)

    return density_np


def gaussian_kde_numpy(
    particles: np.ndarray,
    grid: np.ndarray,
    bandwidth: float | str,
) -> np.ndarray:
    """
    CPU fallback for Gaussian KDE using scipy.stats.gaussian_kde.

    This is the original implementation used when backend is None or
    scipy is not available. Much slower than GPU version for large N.

    Note: scipy.stats.gaussian_kde interprets numeric bandwidth as
    a factor to multiply by data std, not absolute bandwidth.
    Use adaptive_bandwidth_selection() for absolute bandwidth.

    Parameters
    ----------
    particles : np.ndarray
        Particle positions, shape (N,)
    grid : np.ndarray
        Grid points, shape (Nx,)
    bandwidth : float or str
        Bandwidth parameter or method ('scott', 'silverman')
        If float, used as factor * std(particles)

    Returns
    -------
    np.ndarray
        Estimated density on grid, shape (Nx,)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for CPU KDE. Install with: pip install scipy")

    kde = gaussian_kde(particles, bw_method=bandwidth)
    density = kde(grid)
    return density


def estimate_density_from_particles(
    particles: np.ndarray,
    grid: np.ndarray,
    bandwidth: float | str = 0.1,
    backend: Optional["BaseBackend"] = None,
    method: str = "kde",
) -> np.ndarray:
    """
    Unified interface for density estimation from particles.

    Automatically selects GPU-accelerated KDE if backend is available,
    otherwise falls back to scipy CPU implementation.

    Parameters
    ----------
    particles : np.ndarray
        Particle positions, shape (N,)
    grid : np.ndarray
        Grid points for density evaluation, shape (Nx,)
    bandwidth : float or str
        Kernel bandwidth (for KDE) or bin width (for histogram)
    backend : BaseBackend, optional
        If provided, use GPU acceleration. If None, use CPU scipy.
    method : str
        Density estimation method: 'kde' or 'histogram'

    Returns
    -------
    np.ndarray
        Estimated density on grid, shape (Nx,)

    Examples
    --------
    >>> from mfgarchon.backends.torch_backend import TorchBackend
    >>> backend = TorchBackend(device='mps')
    >>> particles = np.random.randn(10000)
    >>> grid = np.linspace(-3, 3, 100)
    >>> density = estimate_density_from_particles(particles, grid, 0.1, backend)
    >>> density.shape
    (100,)
    """
    if method == "kde":
        if backend is not None:
            # GPU-accelerated KDE
            return gaussian_kde_gpu(particles, grid, float(bandwidth), backend)
        else:
            # CPU fallback
            return gaussian_kde_numpy(particles, grid, bandwidth)
    elif method == "histogram":
        # Fast histogram-based estimate (future enhancement)
        raise NotImplementedError("Histogram method coming in Phase 2")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'kde' or 'histogram'")


def adaptive_bandwidth_selection(particles: np.ndarray, method: str = "silverman") -> float:
    """
    Automatic bandwidth selection for KDE.

    Implements Scott's rule and Silverman's rule of thumb for
    bandwidth selection.

    Parameters
    ----------
    particles : np.ndarray
        Particle positions, shape (N,)
    method : str
        Selection method: 'scott' or 'silverman'

    Returns
    -------
    float
        Optimal bandwidth estimate

    References
    ----------
    - Scott, D.W. (1992). Multivariate Density Estimation
    - Silverman, B.W. (1986). Density Estimation for Statistics and Data Analysis
    """
    N = len(particles)
    std = np.std(particles, ddof=1)

    if method == "scott":
        # Scott's rule: h = σ N^(-1/5)
        return std * (N ** (-1 / 5))
    elif method == "silverman":
        # Silverman's rule: h = 0.9 min(σ, IQR/1.34) N^(-1/5)
        iqr = np.percentile(particles, 75) - np.percentile(particles, 25)
        scale = min(std, iqr / 1.34)
        return 0.9 * scale * (N ** (-1 / 5))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'scott' or 'silverman'")


def adaptive_bandwidth_selection_nd(particles: np.ndarray, method: str = "silverman") -> float:
    """
    Automatic bandwidth selection for multi-dimensional KDE.

    Extends Scott's rule and Silverman's rule to multi-dimensional case
    using spherical (isotropic) Gaussian kernels.

    Mathematical formulation:
        h = σ̄ * N^(-1/(d+4))

    where σ̄ is the average standard deviation across dimensions and
    d is the number of dimensions.

    Parameters
    ----------
    particles : np.ndarray
        Particle positions, shape (N, d) for d dimensions
    method : str
        Selection method: 'scott' or 'silverman'

    Returns
    -------
    float
        Optimal bandwidth estimate (isotropic)

    References
    ----------
    - Scott, D.W. (2015). Multivariate Density Estimation: Theory, Practice, and Visualization
    - Silverman, B.W. (1986). Density Estimation for Statistics and Data Analysis
    """
    N, d = particles.shape

    # Compute per-dimension statistics
    stds = np.std(particles, axis=0, ddof=1)
    avg_std = np.mean(stds)

    if method == "scott":
        # Scott's rule for d dimensions: h = σ̄ N^(-1/(d+4))
        return avg_std * (N ** (-1 / (d + 4)))
    elif method == "silverman":
        # Silverman's rule adapted for d dimensions
        # Use average scale across dimensions
        scales = []
        for dim in range(d):
            dim_data = particles[:, dim]
            iqr = np.percentile(dim_data, 75) - np.percentile(dim_data, 25)
            scale = min(stds[dim], iqr / 1.34)
            scales.append(scale)
        avg_scale = np.mean(scales)
        return 0.9 * avg_scale * (N ** (-1 / (d + 4)))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'scott' or 'silverman'")


if __name__ == "__main__":
    """Smoke test for multi-D GPU KDE."""
    print("Testing multi-D GPU KDE implementation...")

    # Test 2D case
    print("\n1. Testing 2D Gaussian KDE...")
    np.random.seed(42)
    N = 5000
    particles_2d = np.random.randn(N, 2)  # 2D Gaussian particles

    # Create 2D grid
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    grid_points_2d = np.column_stack([xx.ravel(), yy.ravel()])

    # Adaptive bandwidth
    bw = adaptive_bandwidth_selection_nd(particles_2d, method="silverman")
    print(f"   Adaptive bandwidth: {bw:.4f}")

    # Test with numpy backend (CPU)
    from mfgarchon.backends.numpy_backend import NumPyBackend

    backend = NumPyBackend()

    density_2d = gaussian_kde_gpu_nd(particles_2d, grid_points_2d, bandwidth=bw, backend=backend)

    assert density_2d.shape == (len(grid_points_2d),), (
        f"Expected shape {(len(grid_points_2d),)}, got {density_2d.shape}"
    )
    assert np.all(density_2d >= 0), "Density must be non-negative"
    assert np.isfinite(density_2d).all(), "Density must be finite"

    # Check normalization (should integrate to ~1)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    total_mass = np.sum(density_2d.reshape(50, 50)) * dx * dy
    print(f"   Total mass (should be ~1): {total_mass:.4f}")
    assert 0.8 < total_mass < 1.2, f"Mass conservation issue: {total_mass}"

    print("   ✓ 2D KDE passed")

    # Test 3D case
    print("\n2. Testing 3D Gaussian KDE...")
    N_3d = 1000
    particles_3d = np.random.randn(N_3d, 3)  # 3D Gaussian particles

    # Create 3D grid (coarser for memory)
    x3 = np.linspace(-2, 2, 20)
    y3 = np.linspace(-2, 2, 20)
    z3 = np.linspace(-2, 2, 20)
    xx3, yy3, zz3 = np.meshgrid(x3, y3, z3, indexing="ij")
    grid_points_3d = np.column_stack([xx3.ravel(), yy3.ravel(), zz3.ravel()])

    bw_3d = adaptive_bandwidth_selection_nd(particles_3d, method="scott")
    print(f"   Adaptive bandwidth: {bw_3d:.4f}")

    density_3d = gaussian_kde_gpu_nd(particles_3d, grid_points_3d, bandwidth=bw_3d, backend=backend)

    assert density_3d.shape == (len(grid_points_3d),), (
        f"Expected shape {(len(grid_points_3d),)}, got {density_3d.shape}"
    )
    assert np.all(density_3d >= 0), "Density must be non-negative"
    assert np.isfinite(density_3d).all(), "Density must be finite"

    # Check normalization
    dx3 = x3[1] - x3[0]
    total_mass_3d = np.sum(density_3d) * dx3**3
    print(f"   Total mass (should be ~1): {total_mass_3d:.4f}")

    print("   ✓ 3D KDE passed")

    # Test bandwidth selection
    print("\n3. Testing bandwidth selection methods...")
    for method in ["scott", "silverman"]:
        bw_test = adaptive_bandwidth_selection_nd(particles_2d, method=method)
        print(f"   {method}: {bw_test:.4f}")
        assert bw_test > 0, f"Bandwidth must be positive for {method}"

    print("   ✓ Bandwidth selection passed")

    print("\n✓ All smoke tests passed!")

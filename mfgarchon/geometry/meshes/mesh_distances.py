"""
Compute mesh quality metrics for GFDM EOC analysis.

For meshfree methods, the key distances are:
1. Fill distance h: max radius of empty circle (largest gap in coverage)
2. Separation distance q: HALF of min distance between points (safe radius)
3. Mesh ratio h/q: always >= 1, optimal ~1.155 (hexagonal) or ~1.414 (square grid)

Definitions follow Wendland's convention where q = (1/2) * min_{j≠k} ||x_j - x_k||,
which guarantees h/q >= 1 for any point distribution.

Reference:
- Wendland, H. "Scattered Data Approximation" (2005), Chapter 14
- Fasshauer, G. "Meshfree Approximation Methods with MATLAB" (2007)
"""

from typing import NamedTuple

import numpy as np
from scipy.spatial import cKDTree


class MeshDistances(NamedTuple):
    """Container for mesh distance metrics."""

    fill_distance: float  # h = max empty circle radius
    separation_distance: float  # q = (1/2) * min distance between points
    mesh_ratio: float  # h/q >= 1, optimal ~1.155 (hexagonal)
    mean_nearest_neighbor: float  # average NN distance
    std_nearest_neighbor: float  # std of NN distances

    def summary(self) -> str:
        return (
            f"Mesh Distances:\n"
            f"  Fill distance h = {self.fill_distance:.6f}\n"
            f"  Separation q    = {self.separation_distance:.6f}\n"
            f"  Mesh ratio h/q  = {self.mesh_ratio:.2f}\n"
            f"  Mean NN dist    = {self.mean_nearest_neighbor:.6f}\n"
            f"  Std NN dist     = {self.std_nearest_neighbor:.6f}"
        )


def compute_mesh_distances(
    collocation_points: np.ndarray,
    domain_bounds: list[tuple[float, float]],
    n_test_points: int = 10000,
    seed: int = 42,
) -> MeshDistances:
    """
    Compute mesh quality distances for EOC analysis.

    Args:
        collocation_points: (N, d) array of collocation point coordinates
        domain_bounds: List of (min, max) for each dimension
        n_test_points: Number of random test points for fill distance estimation
        seed: Random seed for test point generation

    Returns:
        MeshDistances namedtuple with all metrics

    Example:
        >>> points = np.random.rand(100, 2)  # 100 points in [0,1]²
        >>> bounds = [(0, 1), (0, 1)]
        >>> distances = compute_mesh_distances(points, bounds)
        >>> print(distances.summary())
    """
    _N, d = collocation_points.shape

    # Build KD-tree for efficient nearest neighbor queries
    tree = cKDTree(collocation_points)

    # 1. Separation distance: HALF of min distance between any two points
    # This follows Wendland's definition: q = (1/2) * min_{j≠k} ||x_j - x_k||
    # Guarantees h/q >= 1 for any point distribution
    distances_to_nn, _ = tree.query(collocation_points, k=2)
    nn_distances = distances_to_nn[:, 1]  # Skip self (distance 0)

    min_distance = np.min(nn_distances)
    separation_distance = min_distance / 2.0  # q = min_dist / 2
    mean_nn = np.mean(nn_distances)
    std_nn = np.std(nn_distances)

    # 2. Fill distance: max distance from any point in domain to nearest collocation point
    # Estimated via Monte Carlo sampling
    rng = np.random.RandomState(seed)
    test_points = np.zeros((n_test_points, d))
    for i, (lo, hi) in enumerate(domain_bounds):
        test_points[:, i] = rng.uniform(lo, hi, n_test_points)

    distances_to_nearest, _ = tree.query(test_points, k=1)
    fill_distance = np.max(distances_to_nearest)

    # 3. Mesh ratio
    mesh_ratio = fill_distance / separation_distance

    return MeshDistances(
        fill_distance=fill_distance,
        separation_distance=separation_distance,
        mesh_ratio=mesh_ratio,
        mean_nearest_neighbor=mean_nn,
        std_nearest_neighbor=std_nn,
    )


def compute_distances_for_eoc_study(
    resolutions: list[int],
    domain_L: float = 1.0,
    sampling_method: str = "grid",
    seed: int = 42,
) -> dict[int, MeshDistances]:
    """
    Compute mesh distances for EOC convergence study.

    Args:
        resolutions: List of N_x values (grid points per dimension)
        domain_L: Domain size [0, L]^2
        sampling_method: "grid" for structured, "lloyd" for meshfree
        seed: Random seed

    Returns:
        Dict mapping resolution to MeshDistances

    Example:
        >>> results = compute_distances_for_eoc_study([10, 20, 40])
        >>> for n, dist in results.items():
        ...     print(f"N={n}x{n}: h={dist.fill_distance:.4f}, q={dist.separation_distance:.4f}")
    """
    results = {}
    domain_bounds = [(0, domain_L), (0, domain_L)]

    for Nx in resolutions:
        if sampling_method == "grid":
            # Structured grid
            x = np.linspace(0, domain_L, Nx)
            y = np.linspace(0, domain_L, Nx)
            X, Y = np.meshgrid(x, y, indexing="ij")
            points = np.column_stack([X.ravel(), Y.ravel()])
        else:
            # Meshfree (Lloyd)
            from mfgarchon.geometry.implicit import Hyperrectangle

            domain = Hyperrectangle(bounds=np.array([[0, domain_L], [0, domain_L]]))
            n_total = Nx * Nx
            n_boundary = 4 * (Nx - 1)  # Approximate
            n_interior = n_total - n_boundary
            coll = domain.get_collocation_points(
                n_interior=n_interior,
                n_boundary=n_boundary,
                method="lloyd",
                seed=seed,
            )
            points = coll.points

        distances = compute_mesh_distances(points, domain_bounds)
        results[Nx] = distances

    return results

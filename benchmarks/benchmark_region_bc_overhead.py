"""How much does a region-based boundary condition cost against a plain one?

This was `tests/integration/test_region_based_bc.py::TestRegionBasedBCPerformance`
until #1877, where it failed the nightly full suite at "overhead 12.3% exceeds 10%".
It was measuring the runner, not the code.

The original estimator timed 100 standard applications, then 100 region applications, and
divided. Those blocks run at different moments, so anything the machine does in between is
charged to the region path. Measured over 21 repeats on a loaded machine: min -59.5%, median
1.0%, max +105.3%, sd 34.7pp, exceeding its own 10% bound 24% of the time. An overhead of
-59.5% is not a number the region path can produce.

Interleaving the two variants so machine drift cancels -- same machine, same load, same work,
only the estimator changed -- gives min -4.3%, median 0.4%, max 4.2%, sd 2.0pp. That isolates
the estimator's structure as the cause rather than the environment.

Interleaving alone is still not enough to gate on. In a fresh short-lived process the FIRST
repeat is systematically the worst (14.5%, 9.5%, 11.0% in three of six runs), and individual
repeats excurse to -35% even after warmup; medians of 5 above 5% do occur under heavy
contention. The true overhead is 0.1-1%, so no threshold both admits the truth and survives the
noise. Hence a benchmark, run deliberately, rather than a test that gates a merge.

    python benchmarks/benchmark_region_bc_overhead.py
"""

from __future__ import annotations

import time

import numpy as np

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    FDMApplicator,
    mixed_bc_from_regions,
    no_flux_bc,
)

N_ITERATIONS = 100
N_REPEATS = 9
N_WARMUP = 3


def _build():
    geometry = TensorProductGrid(
        bounds=[(0, 1), (0, 1)], boundary_conditions=no_flux_bc(dimension=2), Nx_points=[101, 101]
    )
    bc_standard = BoundaryConditions(
        dimension=2,
        segments=[BCSegment(name="all", bc_type=BCType.DIRICHLET, value=0.0, boundary=None)],
    )
    geometry.mark_region("all_domain", predicate=lambda x: np.ones(x.shape[0], dtype=bool))
    bc_region = mixed_bc_from_regions(
        geometry,
        {"all_domain": BCSegment(name="all_bc", bc_type=BCType.DIRICHLET, value=0.0)},
    )
    return geometry, bc_standard, bc_region


def measure() -> tuple[float, list[float]]:
    """Median relative overhead in percent, and the per-repeat values behind it."""
    geometry, bc_standard, bc_region = _build()
    field = np.random.randn(101, 101)
    domain_bounds = np.array([[0, 1], [0, 1]])
    applicator = FDMApplicator(dimension=2)

    # The first repeat is systematically the worst in a cold process, so warm up properly
    # rather than letting the statistic absorb it.
    for _ in range(N_WARMUP):
        for _ in range(N_ITERATIONS):
            applicator.apply(field, bc_standard, domain_bounds=domain_bounds)
            applicator.apply(field, bc_region, domain_bounds=domain_bounds, geometry=geometry)

    overheads = []
    for _ in range(N_REPEATS):
        t_standard = 0.0
        t_region = 0.0
        for _ in range(N_ITERATIONS):
            start = time.perf_counter()
            applicator.apply(field, bc_standard, domain_bounds=domain_bounds)
            t_standard += time.perf_counter() - start

            start = time.perf_counter()
            applicator.apply(field, bc_region, domain_bounds=domain_bounds, geometry=geometry)
            t_region += time.perf_counter() - start
        overheads.append((t_region - t_standard) / t_standard * 100)

    return float(np.median(overheads)), overheads


def main() -> None:
    median, overheads = measure()
    print(f"Region-based BC overhead, {N_ITERATIONS} interleaved applications x {N_REPEATS} repeats")
    print(f"  per repeat: {[f'{o:.1f}%' for o in overheads]}")
    print(f"  median:     {median:.2f}%")
    print(f"  spread:     min {min(overheads):.1f}%  max {max(overheads):.1f}%")
    print("\nRead the median. A single repeat carries the machine's noise, not the code's cost.")


if __name__ == "__main__":
    main()

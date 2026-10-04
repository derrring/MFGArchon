"""How much does a region-based boundary condition cost against a plain one?

This was `tests/integration/test_region_based_bc.py::TestRegionBasedBCPerformance` until #1877,
where it failed the nightly full suite at "overhead 12.3% exceeds 10%". No wall-clock ratio of
these two paths reaches that precision, so the assertion could not have been satisfied reliably by
any threshold near its own bound.

The original estimator timed 100 standard applications, then 100 region applications, and divided.
Those blocks run at different moments, so anything the machine does in between is charged to the
region path. Over 21 repeats on a loaded machine: min -59.5%, median 1.0%, max +105.3%, sd 34.7pp,
exceeding its own 10% bound 24% of the time. An overhead of -59.5% is not a cost the region path
can have.

Interleaving the two variants fixes that particular defect -- within one process, 11 repeats give
min -4.3%, median 0.4%, max 4.2%, sd 2.0pp. **That figure does not transfer to a fresh process,
and an earlier version of this file implied it did.** Repeats inside one process share warm caches,
one allocation layout and one core assignment; the ratio itself shifts between processes. Measured
across 14 fresh invocations of this script at load average 2.0 -- an idle machine -- the headline
median-of-9 ran from -13.33% to +13.25%, sd 5.50, with 86% of invocations outside the 0.1-1% band
that repeated measurement puts the true cost in. Taking the ratio of per-variant MINIMA instead of
the median of ratios, on the theory that noise is one-sided, does not help: 16 fresh processes gave
sd 5.60 against the median's 5.58.

So: the central estimate is that the region path costs about **1% or less**, and a single
invocation of this script carries roughly +/-14 points around it. Compare builds by running it many
times, not once.

    python benchmarks/benchmark_region_bc_overhead.py
"""

from __future__ import annotations

import os
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
    print(f"  load average: {os.getloadavg()[0]:.1f}")
    print(f"  per repeat:   {[f'{o:.1f}%' for o in overheads]}")
    print(f"  median:       {median:.2f}%")
    print(f"  spread:       min {min(overheads):.1f}%  max {max(overheads):.1f}%")
    print(
        "\nOne invocation is not an answer. These repeats share a process, so they share caches, "
        "\nallocation layout and core assignment; measured across 14 fresh invocations on an idle "
        "\nmachine the median above ranged -13.3% to +13.2% (sd 5.5). The central estimate from "
        "\nrepeated running is about 1% or less. Compare builds over many invocations."
    )


if __name__ == "__main__":
    main()

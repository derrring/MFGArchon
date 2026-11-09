"""
Fast Sweeping Method (FSM) for the Eikonal Equation.

The Fast Sweeping Method solves the Eikonal equation |grad T| = 1/F using
Gauss-Seidel iterations with alternating sweep directions. It has O(N)
complexity per sweep and typically converges in a small number of iterations.

Algorithm Overview:
    1. Initialize T = inf everywhere except frozen points
    2. Repeat until convergence:
       - Sweep in all 2^d directions (++, +-, -+, -- in 2D)
       - For each point, compute Godunov update using current T values
       - Update T if new value is smaller (monotonicity)
    3. Return T array

Compared to FMM:
    - FSM: O(N) per sweep, but may need multiple iterations
    - FMM: O(N log N) total, single pass
    - FSM is often faster for simple domains, FMM for complex geometries

Sweep Directions:
    In 2D, the 4 sweeps are:
    - (1,1): i = 0..N-1, j = 0..M-1
    - (1,-1): i = 0..N-1, j = M-1..0
    - (-1,1): i = N-1..0, j = 0..M-1
    - (-1,-1): i = N-1..0, j = M-1..0

    In nD, there are 2^n sweep directions.

Convergence:
    For simple domains (convex obstacles), FSM converges in 2^d iterations.
    For complex domains, more iterations may be needed.

References:
- Zhao (2005): A fast sweeping method for Eikonal equations, Math. Comp.
- Tsai et al. (2003): Fast sweeping algorithms for a class of HJ equations

Created: 2026-02-06 (Issue #664)
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.geometry.level_set.eikonal.godunov_update import godunov_update_nd
from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid

logger = get_logger(__name__)


class FastSweepingMethod:
    """
    Fast Sweeping Method for solving the Eikonal equation.

    Solves |grad T| = 1/F using Gauss-Seidel iterations with alternating sweeps.

    Attributes:
        geometry: TensorProductGrid providing grid structure and spacing.
        max_iterations: Maximum number of full sweep cycles.
        tolerance: Convergence tolerance for max change in T.

    Example:
        >>> from mfgarchon.geometry.grids import TensorProductGrid
        >>> from mfgarchon.geometry.boundary import no_flux_bc
        >>> grid = TensorProductGrid(
        ...     bounds=[(0, 1), (0, 1)], Nx=[100, 100],
        ...     boundary_conditions=no_flux_bc(dimension=2)
        ... )
        >>> fsm = FastSweepingMethod(grid)
        >>> phi_sdf = fsm.compute_signed_distance(phi_initial)
    """

    def __init__(
        self,
        geometry: TensorProductGrid,
        max_iterations: int = 10,
        tolerance: float = 1e-10,
    ) -> None:
        """
        Initialize FSM solver.

        Args:
            geometry: Grid providing spacing and structure.
            max_iterations: Max sweep cycles (default: 10, usually 2-3 suffice).
            tolerance: Stop when max(|T_new - T_old|) < tolerance.
        """
        self.geometry = geometry
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self._spacing = tuple(geometry.spacing)
        self._shape = tuple(geometry.Nx_points)
        self._ndim = len(self._shape)

        # Precompute sweep directions: all combinations of +1 and -1
        self._sweep_directions = list(product([-1, 1], repeat=self._ndim))

    def solve(
        self,
        speed: NDArray[np.float64] | float,
        frozen_mask: NDArray[np.bool_],
        frozen_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Solve the Eikonal equation |grad T| = 1/F.

        Args:
            speed: Speed function F(x) > 0. Scalar or array matching grid shape.
            frozen_mask: Boolean mask of points with known values.
            frozen_values: Values at frozen points.

        Returns:
            Solution T with |grad T| = 1/F.
        """
        # Convert speed to array if scalar
        if np.isscalar(speed):
            speed_array = np.full(self._shape, float(speed), dtype=np.float64)
        else:
            speed_array = np.asarray(speed, dtype=np.float64)

        # Initialize T
        T = np.full(self._shape, np.inf, dtype=np.float64)
        T[frozen_mask] = frozen_values[frozen_mask]

        # Iterative sweeping
        for iteration in range(self.max_iterations):
            max_change = 0.0

            # Perform all 2^d sweeps
            for direction in self._sweep_directions:
                change = self._single_sweep(T, speed_array, frozen_mask, direction)
                max_change = max(max_change, change)

            logger.debug(f"FSM iteration {iteration}: max change = {max_change:.2e}")

            if max_change < self.tolerance:
                logger.debug(f"FSM converged in {iteration + 1} iterations")
                break

        return T

    def compute_signed_distance(
        self,
        phi_initial: NDArray[np.float64],
        subcell_accuracy: bool = True,
    ) -> NDArray[np.float64]:
        """
        Compute signed distance function from initial level set.

        Args:
            phi_initial: Initial level set. Zero level set is the interface.
            subcell_accuracy: If True, use subcell interface location.

        Returns:
            Signed distance function with |grad phi| = 1.
        """
        # Initialize frozen points at interface
        frozen_mask, frozen_values = self._initialize_interface(phi_initial, subcell_accuracy=subcell_accuracy)

        # Handle exact zeros: points with phi = 0 exactly should be frozen at 0
        exact_zeros = np.abs(phi_initial) < 1e-14
        frozen_mask = frozen_mask | exact_zeros
        frozen_values[exact_zeros] = 0.0

        # Solve for positive region (includes zeros)
        T_positive = self._solve_one_sided(frozen_mask, frozen_values, phi_initial >= 0)

        # Solve for negative region (includes zeros)
        T_negative = self._solve_one_sided(frozen_mask, frozen_values, phi_initial <= 0)

        # Combine: positive distance where phi >= 0, negative where phi < 0
        result = np.where(phi_initial >= 0, T_positive, -T_negative)

        return result

    def _solve_one_sided(
        self,
        frozen_mask: NDArray[np.bool_],
        frozen_values: NDArray[np.float64],
        region_mask: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Solve FSM for one side of the interface."""
        T = np.full(self._shape, np.inf, dtype=np.float64)
        T[frozen_mask] = np.abs(frozen_values[frozen_mask])

        # Create effective frozen mask for this region
        effective_frozen = frozen_mask.copy()

        # Speed = 1 for SDF
        speed_array = np.ones(self._shape, dtype=np.float64)

        for _iteration in range(self.max_iterations):
            max_change = 0.0

            for direction in self._sweep_directions:
                change = self._single_sweep_onesided(T, speed_array, effective_frozen, direction, region_mask)
                max_change = max(max_change, change)

            if max_change < self.tolerance:
                break

        return T

    def _single_sweep(
        self,
        T: NDArray[np.float64],
        speed: NDArray[np.float64],
        frozen_mask: NDArray[np.bool_],
        direction: tuple[int, ...],
    ) -> float:
        """Perform one sweep in a given direction."""
        max_change = 0.0

        # Build index ranges for this sweep direction
        ranges = []
        for dim in range(self._ndim):
            if direction[dim] == 1:
                ranges.append(range(self._shape[dim]))
            else:
                ranges.append(range(self._shape[dim] - 1, -1, -1))

        # Sweep through all points
        for idx in product(*ranges):
            if frozen_mask[idx]:
                continue

            T_old = T[idx]
            T_new = self._compute_update(T, idx, speed[idx])

            if T_new < T_old:
                T[idx] = T_new
                max_change = max(max_change, T_old - T_new)

        return max_change

    def _single_sweep_onesided(
        self,
        T: NDArray[np.float64],
        speed: NDArray[np.float64],
        frozen_mask: NDArray[np.bool_],
        direction: tuple[int, ...],
        region_mask: NDArray[np.bool_],
    ) -> float:
        """Perform one sweep restricted to a region."""
        max_change = 0.0

        ranges = []
        for dim in range(self._ndim):
            if direction[dim] == 1:
                ranges.append(range(self._shape[dim]))
            else:
                ranges.append(range(self._shape[dim] - 1, -1, -1))

        for idx in product(*ranges):
            if frozen_mask[idx] or not region_mask[idx]:
                continue

            T_old = T[idx]
            T_new = self._compute_update(T, idx, speed[idx])

            if T_new < T_old:
                T[idx] = T_new
                max_change = max(max_change, T_old - T_new)

        return max_change

    def _compute_update(
        self,
        T: NDArray[np.float64],
        idx: tuple[int, ...],
        speed: float,
    ) -> float:
        """Compute Godunov update at a grid point."""
        T_neighbors = []
        for dim in range(self._ndim):
            T_minus = np.inf
            T_plus = np.inf

            if idx[dim] > 0:
                neighbor_idx = list(idx)
                neighbor_idx[dim] -= 1
                T_minus = T[tuple(neighbor_idx)]

            if idx[dim] < self._shape[dim] - 1:
                neighbor_idx = list(idx)
                neighbor_idx[dim] += 1
                T_plus = T[tuple(neighbor_idx)]

            T_neighbors.append((T_minus, T_plus))

        return godunov_update_nd(T_neighbors, self._spacing, speed)

    def _initialize_interface(
        self,
        phi: NDArray[np.float64],
        subcell_accuracy: bool = True,
    ) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
        """Initialize frozen points at the interface (same as FMM)."""
        frozen_mask = np.zeros(self._shape, dtype=bool)
        frozen_values = np.zeros(self._shape, dtype=np.float64)

        it = np.nditer(phi, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            phi_i = phi[idx]

            for dim in range(self._ndim):
                for direction in [-1, 1]:
                    neighbor_idx = list(idx)
                    neighbor_idx[dim] += direction
                    neighbor = tuple(neighbor_idx)

                    if not (0 <= neighbor[dim] < self._shape[dim]):
                        continue

                    phi_j = phi[neighbor]

                    if phi_i * phi_j < 0:
                        if subcell_accuracy:
                            theta = phi_i / (phi_i - phi_j)
                            dist_to_interface = theta * self._spacing[dim]
                        else:
                            dist_to_interface = 0.5 * self._spacing[dim]

                        if not frozen_mask[idx] or dist_to_interface < frozen_values[idx]:
                            frozen_mask[idx] = True
                            frozen_values[idx] = dist_to_interface

            it.iternext()

        return frozen_mask, frozen_values

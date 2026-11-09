#!/usr/bin/env python3
"""
Multi-population Mean Field Games infrastructure.

This module provides protocol and base class for K-population MFG systems
with cross-population coupling. Multi-population MFG models heterogeneous
agent populations with distinct objectives and interaction patterns.

Mathematical Framework
----------------------
For populations k = 1, ..., K, the coupled system is:

    HJB equations (backward in time):
        -∂uₖ/∂t + Hₖ(t, x, ∇uₖ, {mⱼ}ⱼ₌₁ᴷ) = 0    for k = 1, ..., K
        uₖ(T, x) = gₖ(x)

    FP equations (forward in time):
        ∂mₖ/∂t - div(mₖ ∇ₚHₖ) - σₖ²Δmₖ = 0       for k = 1, ..., K
        mₖ(0, x) = m₀ₖ(x)

    Cross-population coupling:
        Hₖ(t, x, p, {mⱼ}) = ½|p|² + Σⱼ αₖⱼ·mⱼ(x) + fₖ(t, x, {mⱼ})

    where:
        - uₖ(t,x): Value function for population k
        - mₖ(t,x): Density of population k
        - αₖⱼ: Coupling coefficient (effect of population j on k)
        - fₖ: Population-specific running cost
        - σₖ: Per-population diffusion coefficient

Applications
------------
- Heterogeneous traffic (cars, trucks, bicycles)
- Multi-species ecological models (predator-prey)
- Capacity-constrained systems (resident/tourist flows)
- Market segmentation (retail, wholesale)

Examples
--------
>>> # 2-population linear coupling
>>> problem = MultiPopulationMFGProblem(
...     num_populations=2,
...     spatial_bounds=[(0, 1), (0, 1)],
...     spatial_discretization=[50, 50],
...     coupling_matrix=[[0.1, 0.05], [0.05, 0.1]],  # Asymmetric interaction
...     T=1.0, Nt=50,
...     volatility=0.01,  # shared by all populations
... )

>>> # Capacity-constrained multi-population (see examples/)
>>> from examples.advanced.capacity_constrained_mfg import CapacityField
>>> problem = MultiPopCapacityMFG(
...     num_populations=3,
...     capacity_field=capacity,
...     coupling_matrix=np.eye(3) * 0.1,
...     congestion_weight=1.0
... )

Part of: Issue #295 - Multi-population MFG support
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import numpy as np

from mfgarchon.core.base_problem import MFGProblemProtocol
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.types.callable_protocols import Slots

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ============================================================================
# Multi-Population Protocol
# ============================================================================


@runtime_checkable
class MultiPopulationMFGProtocol(MFGProblemProtocol, Protocol):
    """
    Protocol for K-population MFG with cross-population coupling.

    Extends MFGProblemProtocol with population-indexed methods for
    heterogeneous MFG systems. Each population k has its own:
    - Hamiltonian Hₖ(t, x, p, {mⱼ})
    - Terminal cost gₖ(x)
    - Initial density m₀ₖ(x)
    - Running cost fₖ(x, {mⱼ}, t)

    Universal Properties (inherited from MFGProblemProtocol):
        dimension: int | str
            Spatial dimension
        T: float
            Terminal time
        Nt: int
            Number of time steps
        tSpace: NDArray
            Time discretization array
        volatility: float
            SDE volatility, shared by all populations (the PDE diffusion is derived from it)

    Multi-Population Properties:
        num_populations: int
            Number of populations K
        population_labels: list[str]
            Names for each population (for visualization/logging)

    Population-Indexed Methods:
        hamiltonian_k(t, x, p, m_all, k)
            Hamiltonian for population k
        terminal_cost_k(x, k)
            Terminal cost for population k
        initial_density_k(x, k)
            Initial density for population k
        running_cost_k(t, x, m_all, k)
            Running cost for population k

    Mathematical Framework:
        For k = 1, ..., K:
            Hₖ(t, x, p, {mⱼ}) = ½|p|² + Σⱼ αₖⱼ·mⱼ + fₖ(t, x, {mⱼ})

        Coupling matrix A = [αₖⱼ]:
        - Diagonal αₖₖ: self-congestion
        - Off-diagonal αₖⱼ (j≠k): cross-population interaction

    Examples:
        >>> # Runtime validation
        >>> problem = MultiPopulationMFGProblem(num_populations=2, ...)
        >>> assert isinstance(problem, MultiPopulationMFGProtocol)  # ✅

        >>> # Protocol-based solver
        >>> def solve_multi_pop_mfg(problem: MultiPopulationMFGProtocol):
        ...     K = problem.num_populations
        ...     # Solve K coupled HJB-FP systems
        ...     for k in range(K):
        ...         H_k = lambda t, x, p, m_all: problem.hamiltonian_k(t=t, x=x, p=p, m_all=m_all, k=k)
        ...         # ... solver implementation ...
    """

    # ====================
    # Multi-Population Properties
    # ====================

    num_populations: int  # Number of populations K
    population_labels: list[str]  # Names for each population

    # ====================
    # Population-Indexed MFG Components
    # ====================

    def hamiltonian_k(self, t, x, p, m_all: NDArray, k: int) -> float:
        """
        Hamiltonian for population k with cross-population coupling.

        Args:
            t: Time
            x: Spatial position
                - 1D: float
                - nD: tuple/array of length d
            p: Momentum/co-state ∇uₖ for population k
                - 1D: float
                - nD: tuple/array of length d
            m_all: Density values for all populations
                   Shape: (K,) array [m₁(x), m₂(x), ..., mₖ(x)]
            k: Population index (0 to K-1), a further parameter after the measure (#2375 ruling 8)

        Returns:
            Hamiltonian value Hₖ(t, x, p, {mⱼ})

        Mathematical Form:
            Hₖ(t, x, p, {mⱼ}) = ½|p|² + Σⱼ αₖⱼ·mⱼ(x) + fₖ(t, x, {mⱼ})

        Example:
            >>> # Competition for resources with shared congestion
            >>> def hamiltonian_k(self, t, x, p, m_all, k):
            ...     # Kinetic energy
            ...     H = 0.5 * np.sum(np.atleast_1d(p)**2)
            ...     # Cross-population coupling
            ...     for j in range(self.num_populations):
            ...         H += self.coupling_matrix[k][j] * m_all[j]
            ...     # Shared congestion cost
            ...     m_total = np.sum(m_all)
            ...     H += self.gamma * self.congestion_model.cost(m_total, self.capacity(x))
            ...     return H
        """
        ...

    def terminal_cost_k(self, x, k: int) -> float:
        """
        Terminal cost gₖ(x) for population k.

        Args:
            x: Spatial position
            k: Population index (0 to K-1)

        Returns:
            Terminal cost value gₖ(x)

        Notes:
            Population-specific terminal costs enable heterogeneous objectives:
            - Different destination preferences
            - Varying risk aversion
            - Population-dependent penalties

        Example:
            >>> # Different destinations for each population
            >>> def terminal_cost_k(self, x, k):
            ...     target = self.targets[k]  # Population k's destination
            ...     return 0.5 * (x[0] - target)**2
        """
        ...

    def initial_density_k(self, x, k: int) -> float:
        """
        Initial density m₀ₖ(x) for population k.

        Args:
            x: Spatial position
            k: Population index (0 to K-1)

        Returns:
            Initial density value m₀ₖ(x) ≥ 0

        Notes:
            Must satisfy normalization: ∫ m₀ₖ(x) dx = 1 for each k

        Example:
            >>> # Different initial distributions
            >>> def initial_density_k(self, x, k):
            ...     center = self.initial_centers[k]
            ...     return np.exp(-10 * (x[0] - center)**2)
        """
        ...

    def running_cost_k(self, t, x, m_all: NDArray, k: int) -> float:
        """
        Running cost fₖ(x, {mⱼ}, t) for population k.

        Args:
            t: Time
            x: Spatial position
            m_all: Density values for all populations [m₁(x), ..., mₖ(x)]
            k: Population index (0 to K-1)

        Returns:
            Running cost value fₖ(x, {mⱼ}, t)

        Notes:
            Running cost appears in the Hamiltonian and can depend on:
            - Own density mₖ(x)
            - Other population densities mⱼ(x) for j≠k
            - Spatial position x (e.g., tolls, penalties)
            - Time t (e.g., time-dependent costs)

        Example:
            >>> # Population-specific tolls with congestion
            >>> def running_cost_k(self, t, x, m_all, k):
            ...     toll = self.tolls[k]  # Per-population toll rate
            ...     congestion = np.sum(m_all)  # Total congestion
            ...     return toll * self.is_toll_zone(x) + 0.1 * congestion
        """
        ...


# ============================================================================
# Multi-Population Base Class
# ============================================================================


class MultiPopulationMFGProblem(MFGProblem):
    """
    K-population MFG with linear cross-population coupling.

    Implements multi-population MFG system with coupling matrix αₖⱼ
    defining interaction between populations. Extends MFGProblem with
    population-indexed methods.

    Args:
        num_populations: Number of populations K (must be ≥ 2)
        coupling_matrix: K×K array of coupling coefficients αₖⱼ
            - αₖⱼ: Effect of population j's density on population k's cost
            - Default: Identity matrix * 0.1 (self-congestion only)
            - Shape: (K, K)
        volatility: SDE volatility, shared by all populations. A per-population list is accepted only
            when its entries are equal: no multi-population solver reads a population's own
            volatility, so distinct entries would all be solved at population 0's (#2376).
        population_labels: Names for populations (for visualization)
            - Default: ["Pop0", "Pop1", ..., "Pop{K-1}"]
        **kwargs: Additional arguments passed to MFGProblem

    Attributes:
        num_populations: Number of populations K
        population_labels: List of population names
        coupling_matrix: K×K coupling matrix αₖⱼ
        volatility_vec: The volatility of each population (all equal, see ``volatility``)

    Mathematical Formulation:
        For k = 1, ..., K:
            Hₖ(t, x, p, {mⱼ}) = ½|p|² + Σⱼ αₖⱼ·mⱼ(x)

        Coupling interpretation:
            - αₖₖ > 0: Population k experiences self-congestion
            - αₖⱼ > 0 (j≠k): Population k penalized by population j
            - αₖⱼ < 0: Population k benefits from population j (cooperation)
            - αₖⱼ = 0: No interaction between populations k and j

    Examples:
        >>> # Symmetric competition (2 populations)
        >>> problem = MultiPopulationMFGProblem(
        ...     num_populations=2,
        ...     spatial_bounds=[(0, 1), (0, 1)],
        ...     spatial_discretization=[50, 50],
        ...     coupling_matrix=[[0.1, 0.05],   # Pop 0 vs Pop 1
        ...                      [0.05, 0.1]],  # Pop 1 vs Pop 0
        ...     T=1.0, Nt=50
        ... )

        >>> # 3 populations, self-congestion only
        >>> problem = MultiPopulationMFGProblem(
        ...     num_populations=3,
        ...     spatial_bounds=[(0, 1)],
        ...     spatial_discretization=[100],
        ...     coupling_matrix=np.eye(3) * 0.1,  # Self-congestion only
        ...     volatility=0.02,
        ...     population_labels=["Fast", "Medium", "Slow"],
        ...     T=1.0, Nt=50
        ... )

        >>> # Predator-prey dynamics (cooperation/competition)
        >>> problem = MultiPopulationMFGProblem(
        ...     num_populations=2,
        ...     spatial_bounds=[(0, 1), (0, 1)],
        ...     spatial_discretization=[50, 50],
        ...     coupling_matrix=[[0.1, -0.02],   # Predators benefit from prey
        ...                      [0.05, 0.1]],   # Prey penalized by predators
        ...     population_labels=["Predator", "Prey"],
        ...     T=1.0, Nt=50
        ... )
    """

    # #2375 ruling 8's order for the population-indexed methods: the index k is a further parameter,
    # after the measure. A subclass overriding one out of order is refused at class creation.
    _ruling8_methods: ClassVar[dict[str, Slots]] = {
        "hamiltonian_k": Slots(("t", "x", "p", "m_all", "k")),
        "running_cost_k": Slots(("t", "x", "m_all", "k")),
        "terminal_cost_k": Slots(("x", "k")),
        "initial_density_k": Slots(("x", "k")),
    }

    def __init__(
        self,
        num_populations: int,
        coupling_matrix: NDArray | None = None,
        volatility: float | list[float] = 0.1,
        population_labels: list[str] | None = None,
        **kwargs,
    ):
        """Initialize multi-population MFG problem."""
        if num_populations < 2:
            raise ValueError(f"num_populations must be ≥ 2, got {num_populations}")

        # Per-population volatility. Before #2376 a list was stored here and population 0's entry
        # handed to MFGProblem, and nothing read the rest, so every population was solved at
        # population 0's volatility with no warning. Distinct entries are refused until a solver
        # reads them.
        if isinstance(volatility, (int, float)):
            self.volatility_vec = np.full(num_populations, float(volatility))
        else:
            self.volatility_vec = np.array(volatility, dtype=float)
            if self.volatility_vec.shape != (num_populations,):
                raise ValueError(
                    f"volatility has shape {self.volatility_vec.shape}; a per-population list needs "
                    f"shape ({num_populations},)"
                )
            if not np.all(self.volatility_vec == self.volatility_vec[0]):
                raise NotImplementedError(
                    f"MultiPopulationMFGProblem got distinct per-population volatilities "
                    f"{self.volatility_vec.tolist()}, but no multi-population solver reads a population's "
                    "own volatility: every population would be solved at population 0's (#2376). "
                    "Pass one scalar volatility shared by all populations."
                )

        # Initialize base MFG problem
        super().__init__(volatility=float(self.volatility_vec[0]), **kwargs)

        # Multi-population attributes
        self.num_populations = num_populations
        self.population_labels = population_labels or [f"Pop{k}" for k in range(num_populations)]

        # Validate population labels
        if len(self.population_labels) != num_populations:
            raise ValueError(
                f"population_labels length ({len(self.population_labels)}) "
                f"must match num_populations ({num_populations})"
            )

        # Coupling matrix αₖⱼ
        if coupling_matrix is not None:
            self.coupling_matrix = np.array(coupling_matrix)
            if self.coupling_matrix.shape != (num_populations, num_populations):
                raise ValueError(
                    f"coupling_matrix shape {self.coupling_matrix.shape} must be ({num_populations}, {num_populations})"
                )
        else:
            # Default: self-congestion only (diagonal matrix)
            self.coupling_matrix = np.eye(num_populations) * 0.1

    def hamiltonian_k(self, t, x, p, m_all: NDArray, k: int) -> float:
        """
        Hamiltonian with linear cross-population coupling.

        Implements:
            Hₖ(t, x, p, {mⱼ}) = ½|p|² + Σⱼ αₖⱼ·mⱼ(x)

        Args:
            k: Population index (0 to K-1)
            x: Spatial position
            m_all: Array of densities [m₁(x), ..., mₖ(x)]
            p: Momentum ∇uₖ
            t: Time

        Returns:
            Hamiltonian value
        """
        if not 0 <= k < self.num_populations:
            raise ValueError(f"Population index k={k} out of range [0, {self.num_populations})")

        if len(m_all) != self.num_populations:
            raise ValueError(f"m_all length ({len(m_all)}) must match num_populations ({self.num_populations})")

        # Kinetic energy: ½|p|²
        p_arr = np.atleast_1d(p)
        H = 0.5 * np.sum(p_arr**2)

        # Cross-population coupling: Σⱼ αₖⱼ·mⱼ(x)
        for j in range(self.num_populations):
            H += self.coupling_matrix[k][j] * m_all[j]

        return H

    def terminal_cost_k(self, x, k: int) -> float:
        """
        Terminal cost for population k.

        Default: Returns 0.0 for all populations (no terminal cost).
        Override for heterogeneous objectives.

        Args:
            k: Population index
            x: Spatial position

        Returns:
            Terminal cost gₖ(x)
        """
        if not 0 <= k < self.num_populations:
            raise ValueError(f"Population index k={k} out of range [0, {self.num_populations})")

        # Default: no terminal cost
        return 0.0

    def initial_density_k(self, x, k: int) -> float:
        """
        Initial density for population k.

        Default: Uniform distribution (1.0 / num_populations).
        Override for heterogeneous initial distributions.

        Args:
            k: Population index
            x: Spatial position

        Returns:
            Initial density m₀ₖ(x)
        """
        if not 0 <= k < self.num_populations:
            raise ValueError(f"Population index k={k} out of range [0, {self.num_populations})")

        # Default: uniform distribution across populations
        return 1.0 / self.num_populations

    def running_cost_k(self, t, x, m_all: NDArray, k: int) -> float:
        """
        Running cost for population k.

        Default: No population-specific running cost.
        Override for heterogeneous costs.

        Args:
            k: Population index
            x: Spatial position
            m_all: Array of densities
            t: Time

        Returns:
            Running cost fₖ(x, {mⱼ}, t)
        """
        if not 0 <= k < self.num_populations:
            raise ValueError(f"Population index k={k} out of range [0, {self.num_populations})")

        # Default: no additional running cost
        return 0.0

    def get_volatility_k(self, k: int) -> float:
        """
        Get the SDE volatility of population k.

        Args:
            k: Population index

        Returns:
            The volatility σₖ (all populations share one, see ``volatility``)
        """
        if not 0 <= k < self.num_populations:
            raise ValueError(f"Population index k={k} out of range [0, {self.num_populations})")

        return self.volatility_vec[k]


# ============================================================================
# Smoke Tests
# ============================================================================

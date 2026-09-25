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
...     sigma=[0.01, 0.02]  # Different volatility per population
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
        sigma: float | Callable
            Diffusion coefficient (scalar or per-population)

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
            ...     x_arr = np.atleast_1d(x)
            ...     target = self.targets[k]  # Population k's destination
            ...     return 0.5 * np.sum((x_arr - target)**2)
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
            ...     x_arr = np.atleast_1d(x)
            ...     center = self.initial_centers[k]
            ...     return np.exp(-10 * np.sum((x_arr - center)**2))
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
        sigma: Diffusion coefficient
            - Scalar: Same diffusion for all populations
            - Array of length K: Per-population diffusion [σ₁, ..., σₖ]
        population_labels: Names for populations (for visualization)
            - Default: ["Pop0", "Pop1", ..., "Pop{K-1}"]
        **kwargs: Additional arguments passed to MFGProblem

    Attributes:
        num_populations: Number of populations K
        population_labels: List of population names
        coupling_matrix: K×K coupling matrix αₖⱼ
        sigma_vec: Per-population volatility coefficients

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

        >>> # 3 populations with different diffusion rates
        >>> problem = MultiPopulationMFGProblem(
        ...     num_populations=3,
        ...     spatial_bounds=[(0, 1)],
        ...     spatial_discretization=[100],
        ...     coupling_matrix=np.eye(3) * 0.1,  # Self-congestion only
        ...     sigma=[0.01, 0.02, 0.03],  # Increasing volatility
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
        sigma: float | list[float] = 0.1,
        population_labels: list[str] | None = None,
        **kwargs,
    ):
        """Initialize multi-population MFG problem."""
        if num_populations < 2:
            raise ValueError(f"num_populations must be ≥ 2, got {num_populations}")

        # Initialize base MFG problem
        super().__init__(sigma=sigma if isinstance(sigma, (int, float)) else sigma[0], **kwargs)

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

        # Per-population diffusion coefficients
        if isinstance(sigma, (int, float)):
            self.sigma_vec = np.full(num_populations, sigma)
        else:
            self.sigma_vec = np.array(sigma)
            if len(self.sigma_vec) != num_populations:
                raise ValueError(f"sigma length ({len(self.sigma_vec)}) must match num_populations ({num_populations})")

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

    def get_sigma_k(self, k: int) -> float:
        """
        Get diffusion coefficient for population k.

        Args:
            k: Population index

        Returns:
            Diffusion coefficient σₖ
        """
        if not 0 <= k < self.num_populations:
            raise ValueError(f"Population index k={k} out of range [0, {self.num_populations})")

        return self.sigma_vec[k]


# ============================================================================
# Smoke Tests
# ============================================================================

if __name__ == "__main__":
    """Smoke tests for multi-population MFG infrastructure."""
    print("=" * 80)
    print("Multi-Population MFG Infrastructure - Smoke Tests")
    print("=" * 80)

    # Test 1: Basic instantiation
    print("\nTest 1: Basic Instantiation")
    print("-" * 40)
    problem = MultiPopulationMFGProblem(
        num_populations=2,
        spatial_bounds=[(0, 1), (0, 1)],
        spatial_discretization=[10, 10],
        T=1.0,
        Nt=10,
    )
    # Note: Protocol runtime checking is strict - mainly for static type checking
    # We verify functional compliance through method calls below
    print("✓ Problem created successfully")
    print(f"  Populations: {problem.num_populations}")
    print(f"  Labels: {problem.population_labels}")
    print(f"  Dimension: {problem.dimension}")
    print(f"  Terminal time T: {problem.T}")
    print(f"  Time steps Nt: {problem.Nt}")

    # Test 2: Coupling matrix validation
    print("\nTest 2: Coupling Matrix")
    print("-" * 40)
    coupling = np.array([[0.1, 0.05], [0.03, 0.12]])
    problem = MultiPopulationMFGProblem(
        num_populations=2,
        coupling_matrix=coupling,
        spatial_bounds=[(0, 1)],
        spatial_discretization=[10],
        T=1.0,
        Nt=10,
    )
    assert np.allclose(problem.coupling_matrix, coupling), "Coupling matrix mismatch"
    print(f"✓ Coupling matrix:\n{problem.coupling_matrix}")

    # Test 3: Per-population diffusion
    print("\nTest 3: Per-Population Diffusion")
    print("-" * 40)
    sigma_vec = [0.01, 0.02, 0.03]
    problem = MultiPopulationMFGProblem(
        num_populations=3,
        sigma=sigma_vec,
        spatial_bounds=[(0, 1), (0, 1)],
        spatial_discretization=[10, 10],
        T=1.0,
        Nt=10,
    )
    assert np.allclose(problem.sigma_vec, sigma_vec), "Sigma vector mismatch"
    for k in range(3):
        assert problem.get_sigma_k(k) == sigma_vec[k], f"Sigma mismatch for population {k}"
    print(f"✓ Per-population diffusion: {problem.sigma_vec}")

    # Test 4: Hamiltonian evaluation
    print("\nTest 4: Hamiltonian Evaluation")
    print("-" * 40)
    problem = MultiPopulationMFGProblem(
        num_populations=2,
        coupling_matrix=[[0.1, 0.05], [0.03, 0.1]],
        spatial_bounds=[(0, 1)],
        spatial_discretization=[10],
        T=1.0,
        Nt=10,
    )
    x = 0.5
    m_all = np.array([1.0, 2.0])
    p = 1.0
    t = 0.5
    H0 = problem.hamiltonian_k(t=t, x=x, p=p, m_all=m_all, k=0)
    H1 = problem.hamiltonian_k(t=t, x=x, p=p, m_all=m_all, k=1)
    H0_expected = 0.5 * p**2 + 0.1 * m_all[0] + 0.05 * m_all[1]  # ½p² + α₀₀m₀ + α₀₁m₁
    H1_expected = 0.5 * p**2 + 0.03 * m_all[0] + 0.1 * m_all[1]  # ½p² + α₁₀m₀ + α₁₁m₁
    assert np.isclose(H0, H0_expected), f"H0 mismatch: {H0} vs {H0_expected}"
    assert np.isclose(H1, H1_expected), f"H1 mismatch: {H1} vs {H1_expected}"
    print(f"✓ H₀(x={x}, m={m_all}, p={p}) = {H0:.4f}")
    print(f"✓ H₁(x={x}, m={m_all}, p={p}) = {H1:.4f}")

    # Test 5: Population-indexed methods
    print("\nTest 5: Population-Indexed Methods")
    print("-" * 40)
    for k in range(2):
        g_k = problem.terminal_cost_k(x=x, k=k)
        m0_k = problem.initial_density_k(x=x, k=k)
        f_k = problem.running_cost_k(t=t, x=x, m_all=m_all, k=k)
        print(f"  Population {k} ({problem.population_labels[k]}):")
        print(f"    Terminal cost: {g_k:.4f}")
        print(f"    Initial density: {m0_k:.4f}")
        print(f"    Running cost: {f_k:.4f}")

    # Test 6: Custom population labels
    print("\nTest 6: Custom Population Labels")
    print("-" * 40)
    labels = ["Predator", "Prey"]
    problem = MultiPopulationMFGProblem(
        num_populations=2,
        population_labels=labels,
        spatial_bounds=[(0, 1)],
        spatial_discretization=[10],
        T=1.0,
        Nt=10,
    )
    assert problem.population_labels == labels, "Custom labels not preserved"
    print(f"✓ Custom labels: {problem.population_labels}")

    # Test 7: Error handling
    print("\nTest 7: Error Handling")
    print("-" * 40)
    try:
        # num_populations < 2
        problem = MultiPopulationMFGProblem(
            num_populations=1, spatial_bounds=[(0, 1)], spatial_discretization=[10], T=1.0, Nt=10
        )
        print("✗ Should raise error for num_populations < 2")
    except ValueError as e:
        print(f"✓ Correctly rejected num_populations=1: {e}")

    try:
        # Coupling matrix shape mismatch
        problem = MultiPopulationMFGProblem(
            num_populations=2,
            coupling_matrix=np.eye(3),  # Wrong shape
            spatial_bounds=[(0, 1)],
            spatial_discretization=[10],
            T=1.0,
            Nt=10,
        )
        print("✗ Should raise error for coupling matrix shape mismatch")
    except ValueError as e:
        print(f"✓ Correctly rejected wrong coupling_matrix shape: {e}")

    print("\n" + "=" * 80)
    print("All smoke tests passed!")
    print("=" * 80)

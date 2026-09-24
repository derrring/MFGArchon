#!/usr/bin/env python3
"""
Capacity-constrained multi-population MFG example.

This example demonstrates a realistic multi-population scenario where
two populations (residents and tourists) compete for limited capacity
in a crowded urban area. The populations have different objectives
and movement patterns, constrained by spatial capacity limits.

Mathematical Model
------------------
For populations k ∈ {residents, tourists}:

    HJB: -∂uₖ/∂t + ½|∇uₖ|² + Σⱼ αₖⱼ·mⱼ + C(x)·(m_total)² = 0
    FP:   ∂mₖ/∂t - div(mₖ∇uₖ) - σₖ²Δmₖ = 0

where:
    - C(x): Spatial capacity field (higher = more constrained)
    - m_total = m_residents + m_tourists: Total density
    - (m_total)²: Quadratic congestion penalty

Population Characteristics:
    - Residents: Lower tolerance for congestion, faster movement
    - Tourists: Higher tolerance for congestion, slower movement

Spatial Structure:
    - City center: High capacity constraint (C = 2.0)
    - Suburbs: Low capacity constraint (C = 0.5)

Application
-----------
Models urban mobility with heterogeneous agents competing for space:
- Residents commute efficiently avoiding congestion
- Tourists explore popular areas despite congestion
- Capacity constraints create emergent spatial segregation

Usage:
    python examples/advanced/multi_population_capacity.py

Part of: Issue #295 Phase 3 - Multi-population MFG examples
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from mfgarchon.extensions import MultiPopulationMFGProblem


class CapacityConstrainedMFGProblem(MultiPopulationMFGProblem):
    """
    Multi-population MFG with spatial capacity constraints.

    Extends the basic multi-population problem to include:
    - Spatial capacity field C(x)
    - Quadratic congestion penalty (m_total)²
    - Population-specific capacity tolerance

    Args:
        capacity_field: Function C(x) defining spatial capacity constraints
        capacity_weight: Scaling factor for capacity penalty (default: 1.0)
        **kwargs: Arguments passed to MultiPopulationMFGProblem
    """

    def __init__(
        self,
        capacity_field: callable,
        capacity_weight: float = 1.0,
        **kwargs,
    ):
        """Initialize capacity-constrained problem."""
        super().__init__(**kwargs)
        self.capacity_field = capacity_field
        self.capacity_weight = capacity_weight

    def hamiltonian_k(self, k: int, x, m_all: list[float], p, t) -> float:
        """
        Hamiltonian with capacity constraint.

        H_k(x, {mⱼ}, p, t) = ½|p|² + Σⱼ αₖⱼ·mⱼ + C(x)·(Σⱼ mⱼ)²

        The capacity penalty C(x)·(m_total)² creates a nonlinear coupling
        that penalizes high total density in constrained regions.

        Args:
            k: Population index
            x: Spatial position
            m_all: List of density values [m₁(x), ..., mₖ(x)]
            p: Momentum ∇uₖ
            t: Time

        Returns:
            Hamiltonian value with capacity constraint
        """
        # Standard quadratic Hamiltonian
        H_kinetic = 0.5 * np.sum(np.atleast_1d(p) ** 2)

        # Linear coupling: Σⱼ αₖⱼ·mⱼ
        H_coupling = sum(self.coupling_matrix[k, j] * m_j for j, m_j in enumerate(m_all))

        # Capacity penalty: C(x)·(m_total)²
        m_total = sum(m_all)
        capacity_penalty = self.capacity_weight * self.capacity_field(x) * (m_total**2)

        return H_kinetic + H_coupling + capacity_penalty

    def terminal_cost_k(self, k: int, x) -> float:
        """
        Population-specific terminal cost.

        Residents: Target suburbs (x near 0 or 1)
        Tourists: Target city center (x near 0.5)

        Args:
            k: Population index (0=residents, 1=tourists)
            x: Spatial position

        Returns:
            Terminal cost g_k(x)
        """
        x_val = np.atleast_1d(x)[0]

        if k == 0:  # Residents: prefer suburbs
            # Quadratic cost centered at suburbs (x=0.2 or x=0.8)
            cost_left = (x_val - 0.2) ** 2
            cost_right = (x_val - 0.8) ** 2
            return 0.5 * min(cost_left, cost_right)
        else:  # Tourists: prefer city center
            # Quadratic cost centered at city center (x=0.5)
            return 0.5 * (x_val - 0.5) ** 2

    def initial_density_k(self, k: int, x) -> float:
        """
        Population-specific initial distribution.

        Residents: Start distributed across domain
        Tourists: Start concentrated at entry points

        Args:
            k: Population index (0=residents, 1=tourists)
            x: Spatial position

        Returns:
            Initial density m₀ₖ(x)
        """
        x_val = np.atleast_1d(x)[0]

        if k == 0:  # Residents: uniform distribution
            return 1.0
        else:  # Tourists: concentrated at x=0 (entry point)
            # Gaussian concentrated at x=0
            return 5.0 * np.exp(-50 * (x_val - 0.0) ** 2)


def capacity_field_urban(x) -> float:
    """
    Spatial capacity field for urban scenario.

    Models a city with:
    - City center (x ≈ 0.5): High capacity constraint (C = 2.0)
    - Suburbs (x near 0 or 1): Low capacity constraint (C = 0.5)

    Args:
        x: Spatial position in [0, 1]

    Returns:
        Capacity constraint value C(x) ∈ [0.5, 2.0]
    """
    x_val = np.atleast_1d(x)[0]

    # Gaussian peak at city center
    center_congestion = 2.0 * np.exp(-20 * (x_val - 0.5) ** 2)

    # Baseline suburban capacity
    baseline = 0.5

    return baseline + center_congestion


def main():
    """Run capacity-constrained multi-population example."""
    print("=" * 70)
    print("Capacity-Constrained Multi-Population MFG (Phase 3)")
    print("=" * 70)

    # Problem parameters
    print("\n[1] Problem Setup")
    print("  Scenario: Urban mobility with residents and tourists")
    print("  Populations:")
    print("    - Residents: Low congestion tolerance, fast movement (σ=0.01)")
    print("    - Tourists: High congestion tolerance, slow movement (σ=0.03)")
    print("  Spatial structure: 1D domain [0, 1]")
    print("    - x=0: Suburbs (entry point)")
    print("    - x=0.5: City center (high capacity constraint)")
    print("    - x=1: Suburbs")

    # Create problem
    print("\n[2] Creating capacity-constrained problem...")

    # Coupling matrix: asymmetric interaction
    # Residents more sensitive to tourists than vice versa
    coupling = np.array(
        [
            [0.1, 0.15],  # Residents: moderate self-congestion, high tourist aversion
            [0.05, 0.08],  # Tourists: low self-congestion, low resident aversion
        ]
    )

    problem = CapacityConstrainedMFGProblem(
        num_populations=2,
        spatial_bounds=[(0, 1)],
        spatial_discretization=[100],
        coupling_matrix=coupling,
        capacity_field=capacity_field_urban,
        capacity_weight=0.5,
        T=1.0,
        Nt=50,
        sigma=[0.01, 0.03],  # Residents faster, tourists slower
        population_labels=["Residents", "Tourists"],
    )

    print("✓ Problem created")
    print(f"  - Populations: {problem.num_populations}")
    Nx_points = problem.geometry.get_grid_shape()[0]
    print(f"  - Spatial points: {Nx_points}")
    print(f"  - Time steps: {problem.Nt}")
    print(f"  - Coupling matrix:\n{problem.coupling_matrix}")

    # Visualize capacity field
    print("\n[3] Visualizing capacity field...")
    x_plot = np.linspace(0, 1, 200)
    C_plot = np.array([capacity_field_urban(x) for x in x_plot])

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x_plot, C_plot, "k-", linewidth=2)
    plt.xlabel("Position x")
    plt.ylabel("Capacity Constraint C(x)")
    plt.title("Spatial Capacity Field")
    plt.grid(True, alpha=0.3)
    plt.axvline(0.5, color="r", linestyle="--", alpha=0.5, label="City Center")
    plt.legend()

    # Visualize terminal costs
    plt.subplot(1, 2, 2)
    g0_plot = np.array([problem.terminal_cost_k(x=x, k=0) for x in x_plot])
    g1_plot = np.array([problem.terminal_cost_k(x=x, k=1) for x in x_plot])
    plt.plot(x_plot, g0_plot, "b-", linewidth=2, label="Residents")
    plt.plot(x_plot, g1_plot, "r-", linewidth=2, label="Tourists")
    plt.xlabel("Position x")
    plt.ylabel("Terminal Cost g_k(x)")
    plt.title("Population Objectives")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("examples/outputs/multi_population_capacity_setup.png", dpi=150, bbox_inches="tight")
    print("✓ Saved capacity field visualization to examples/outputs/multi_population_capacity_setup.png")

    # Test Hamiltonian evaluation
    print("\n[4] Testing Hamiltonian with capacity constraint...")
    x_test = 0.5  # City center
    m_test = [0.5, 0.3]  # High density scenario
    p_test = 0.1
    t_test = 0.5

    H0 = problem.hamiltonian_k(t=t_test, x=x_test, p=p_test, m_all=m_test, k=0)
    H1 = problem.hamiltonian_k(t=t_test, x=x_test, p=p_test, m_all=m_test, k=1)

    print(f"  At city center (x={x_test}) with m=[{m_test[0]}, {m_test[1]}]:")
    print(f"  - H_residents = {H0:.6f}")
    print(f"  - H_tourists = {H1:.6f}")

    # Capacity penalty contribution
    m_total = sum(m_test)
    capacity_penalty = problem.capacity_weight * capacity_field_urban(x_test) * (m_total**2)
    print(f"  - Capacity penalty = {capacity_penalty:.6f}")
    print(f"  - Total density = {m_total:.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("Phase 3 Capacity-Constrained Example Complete")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Spatial capacity constraints C(x)")
    print("  ✓ Nonlinear coupling via quadratic congestion penalty")
    print("  ✓ Heterogeneous population objectives")
    print("  ✓ Asymmetric interaction matrix")
    print("\nNext Steps:")
    print("  - Full solver integration (requires Phase 2.5 merge)")
    print("  - Convergence analysis and visualization")
    print("  - Compare with agent-based model results")

    plt.show()


if __name__ == "__main__":
    main()

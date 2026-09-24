#!/usr/bin/env python3
"""
MFG 1D Explicit Formulation Example.

This example demonstrates how to set up a complete 1D Mean Field Game problem
with ALL components explicitly specified. This serves as a learning template
for understanding the mathematical structure of MFG problems.

Mathematical System:
    HJB (backward): -du/dt + H(x, m, Du) = (sigma^2/2) * d^2u/dx^2
    FP  (forward):   dm/dt + div(m * v) = (sigma^2/2) * d^2m/dx^2

    where v = -dH/dp (optimal control)

Components Demonstrated:
    1. Terminal condition: u(T, x) = u_final(x)
    2. Initial density: m(0, x) = m_initial(x)
    3. Potential field: V(x) in Hamiltonian
    4. Hamiltonian: H(x, m, p) = (1/2)|p|^2 + V(x) + coupling * m

Issue #670, #671: These functions were previously silent defaults.
Now they must be explicitly provided for clarity and correctness.

Author: MFGarchon Team
Created: 2026-01-26
"""

import matplotlib.pyplot as plt
import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

# =============================================================================
# Step 1: Define Problem Domain
# =============================================================================

# Spatial domain: [0, 1]
x_min, x_max = 0.0, 1.0
Nx = 51  # Number of grid points

# Time domain: [0, T]
T = 1.0
Nt = 51  # Number of time steps

# Physical parameters
sigma = 0.2  # Diffusion coefficient
coupling_coefficient = 0.5  # Coupling strength between agents


# =============================================================================
# Step 2: Define Mathematical Components (Previously Hidden Defaults)
# =============================================================================


def terminal_condition(x: float) -> float:
    """
    Terminal cost u(T, x) - what agents want to achieve at final time.

    This example uses a smooth periodic function.
    Physical interpretation: regions with low u_final are "desirable".

    Args:
        x: Spatial position in [0, 1]

    Returns:
        Terminal cost at position x
    """
    Lx = x_max - x_min
    return 5.0 * (np.cos(x * 2 * np.pi / Lx) + 0.4 * np.sin(x * 4 * np.pi / Lx))


def initial_density_unnormalized(x: float) -> float:
    """
    Unnormalized initial density shape.

    This defines the SHAPE of the distribution, not the actual density.
    Must be normalized after discretization to satisfy ∫m dx = 1.

    Args:
        x: Spatial position in [0, 1]

    Returns:
        Unnormalized density value (proportional to probability)
    """
    # Two Gaussian peaks
    peak1 = 2.0 * np.exp(-200 * (x - 0.2) ** 2)
    peak2 = 1.0 * np.exp(-200 * (x - 0.8) ** 2)
    return peak1 + peak2


def initial_density(x: float) -> float:
    """
    Normalized initial density m(0, x).

    This is a wrapper that will be used after grid-based normalization.
    For now, returns unnormalized value - normalization happens at grid level.

    Note (Issue #672): Future versions will require ∫m dx = 1.
    The normalization must happen AFTER discretization on the grid.

    Args:
        x: Spatial position in [0, 1]

    Returns:
        Initial density at position x
    """
    return initial_density_unnormalized(x)


def potential_field(x: float) -> float:
    """
    External potential V(x) in the Hamiltonian.

    This represents external forces or preferences in the environment.
    Agents prefer regions with low potential.

    Args:
        x: Spatial position in [0, 1]

    Returns:
        Potential energy at position x
    """
    Lx = x_max - x_min
    return 50.0 * (
        0.1 * np.cos(x * 2 * np.pi / Lx) + 0.25 * np.sin(x * 2 * np.pi / Lx) + 0.1 * np.sin(x * 4 * np.pi / Lx)
    )


def hamiltonian_potential(t: float, x: np.ndarray) -> float:
    """
    Potential function V(t, x) for the class-based Hamiltonian (#2375 ruling 8: time first).

    This wraps the potential_field function to match the expected signature.

    Args:
        x: Position array (1D for this example)
        t: Time (unused for time-independent potential)

    Returns:
        Potential value at position x
    """
    x_scalar = float(x[0]) if hasattr(x, "__len__") else float(x)
    return potential_field(x_scalar)


# =============================================================================
# Step 5: Create MFG Problem with Explicit Components
# =============================================================================

# =============================================================================
# Step 3: Define Boundary Conditions
# =============================================================================

# No-flux (Neumann with zero gradient) - standard for MFG on bounded domains
# Physical meaning: probability mass cannot leave the domain
boundary_conditions = no_flux_bc(dimension=1)


# =============================================================================
# Step 4: Create Geometry with Explicit BC
# =============================================================================

geometry = TensorProductGrid(
    bounds=[(x_min, x_max)],  # 1D domain; dimension inferred (Issue #676)
    Nx=Nx - 1,  # Scalar OK for 1D (Issue #675); intervals = Nx-1 → Nx points
    boundary_conditions=boundary_conditions,  # Explicit BC specification
)

# Bundle all mathematical components
#
# Note on initial_density (Issue #672):
# The density function defines the SHAPE. Normalization to ∫m dx = 1
# currently happens internally in MFGProblem. Future versions will
# require the user to provide already-normalized density.
#
# To normalize explicitly on a grid:
#   x_grid = np.linspace(x_min, x_max, Nx)
#   dx = x_grid[1] - x_grid[0]
#   m_raw = np.array([initial_density(x) for x in x_grid])
#   m_normalized = m_raw / (np.sum(m_raw) * dx)  # Now ∫m dx = 1
#
# Class-based Hamiltonian: H = (1/2)|p|² + V(x) + coupling * m
# Coupling is part of the Hamiltonian in v1.0 (no separate coupling_coefficient).
hamiltonian = SeparableHamiltonian(
    control_cost=QuadraticControlCost(control_cost=1.0),
    potential=hamiltonian_potential,  # V(t, x) = potential_field(x)
    coupling=lambda m: coupling_coefficient * m,  # Linear congestion
    coupling_dm=lambda m: coupling_coefficient,  # dH/dm = coupling_coefficient
)

# v1.0 API: Model holds game rules, Conditions holds IC/TC + time horizon
model = Model(hamiltonian=hamiltonian, sigma=sigma)
conditions = Conditions(
    u_terminal=terminal_condition,
    m_initial=initial_density,
    T=T,
)

# Create problem
problem = MFGProblem(
    model=model,
    domain=geometry,
    conditions=conditions,
    Nt=Nt,
)


# =============================================================================
# Step 6: Solve and Visualize
# =============================================================================


def main():
    """Run the MFG solver and visualize results."""
    print("=" * 60)
    print("MFG 1D Explicit Formulation Example")
    print("=" * 60)

    # Display problem setup
    print("\nProblem Configuration:")
    print(f"  Domain: [{x_min}, {x_max}]")
    print(f"  Grid points: {Nx}")
    print(f"  Time horizon: T = {T}")
    print(f"  Time steps: {Nt}")
    print(f"  Diffusion (sigma): {sigma}")
    print(f"  Coupling coefficient: {coupling_coefficient}")

    # Get spatial grid for plotting
    x_grid = np.linspace(x_min, x_max, Nx)

    # Visualize the components
    _fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Terminal condition
    ax1 = axes[0, 0]
    u_final = np.array([terminal_condition(x) for x in x_grid])
    ax1.plot(x_grid, u_final, "b-", linewidth=2)
    ax1.set_xlabel("x")
    ax1.set_ylabel("u(T, x)")
    ax1.set_title("Terminal Condition u_final(x)")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Plot 2: Initial density
    ax2 = axes[0, 1]
    m_init = np.array([initial_density(x) for x in x_grid])
    # Normalize for display
    dx = x_grid[1] - x_grid[0]
    m_init_normalized = m_init / (np.sum(m_init) * dx)
    ax2.plot(x_grid, m_init_normalized, "r-", linewidth=2)
    ax2.fill_between(x_grid, 0, m_init_normalized, alpha=0.3, color="red")
    ax2.set_xlabel("x")
    ax2.set_ylabel("m(0, x)")
    ax2.set_title("Initial Density m_initial(x) (normalized)")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Potential field
    ax3 = axes[1, 0]
    V_x = np.array([potential_field(x) for x in x_grid])
    ax3.plot(x_grid, V_x, "g-", linewidth=2)
    ax3.set_xlabel("x")
    ax3.set_ylabel("V(x)")
    ax3.set_title("Potential Field V(x)")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Plot 4: All components together (normalized for comparison)
    ax4 = axes[1, 1]
    # Normalize each for comparison
    u_norm = (u_final - u_final.min()) / (u_final.max() - u_final.min() + 1e-10)
    m_norm = m_init_normalized / (m_init_normalized.max() + 1e-10)
    V_norm = (V_x - V_x.min()) / (V_x.max() - V_x.min() + 1e-10)

    ax4.plot(x_grid, u_norm, "b-", linewidth=2, label="u_final (normalized)")
    ax4.plot(x_grid, m_norm, "r-", linewidth=2, label="m_initial (normalized)")
    ax4.plot(x_grid, V_norm, "g-", linewidth=2, label="V(x) (normalized)")
    ax4.set_xlabel("x")
    ax4.set_ylabel("Normalized value")
    ax4.set_title("All Components (normalized)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("mfg_1d_explicit_components.png", dpi=150)
    print("\nSaved component visualization to: mfg_1d_explicit_components.png")

    # Try to solve (may require additional solver setup)
    print("\n" + "-" * 60)
    print("Problem components defined. To solve:")
    print("  result = problem.solve()")
    print("-" * 60)

    # Show the plot
    plt.show()

    print("\nExample complete.")
    print("\nKey Learning Points:")
    print("  1. terminal_condition(): What agents want at time T")
    print("  2. initial_density(): Where agents start at time 0")
    print("  3. potential_field(): External environment preferences")
    print("  4. hamiltonian(): Complete cost structure H(x, m, p)")
    print("\nAll these must be EXPLICITLY provided - no hidden defaults!")


if __name__ == "__main__":
    main()

"""
MFG with Expanding Exit - Dynamic Boundary via Level Set Method.

Problem Setup:
    Domain: 1D corridor [0, L]
    Exit: Small region at x = L (expands based on congestion)

    MFG System:
        - Agents minimize: ∫[0,T] (1 + m(t,x)) dt  (cost + congestion)
        - Terminal cost: 0 at exit, ∞ elsewhere
        - Running cost: g(m) = 1 + m  (linear congestion)

    Exit Evolution:
        - Level set: φ(t,x) defines exit boundary
        - Exit location: {x : φ(t,x) ≤ 0}
        - Expansion velocity: V = k·max(m_exit - m_threshold, 0)
        - Physical interpretation: Exit expands when crowded

    Coupling:
        HJB: ∂u/∂t + (σ²/2)|∇u|² = 1 + m
        FP:  ∂m/∂t - σ²Δm - ∇·(m∇u) = 0
        LS:  ∂φ/∂t + V(m)|∇φ| = 0

    Demonstration:
        - Level set evolution coupled to MFG
        - Exit grows when density high, stabilizes when density acceptable
        - Picard iteration for HJB-FP-LS coupling

Created: 2026-01-18 (Issue #605 Phase 3.1)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.level_set import LevelSetEvolver, TimeDependentDomain
from mfgarchon.utils.mfg_logging import configure_research_logging, get_logger

# Configure logging
configure_research_logging("mfg_expanding_exit", level="INFO")
logger = get_logger(__name__)

# ========================================
# Problem Parameters
# ========================================
print("=" * 70)
print("MFG with Expanding Exit - Level Set Coupling")
print("=" * 70)

# Domain
L = 2.0  # Corridor length
Nx = 200  # Number of grid intervals
# TensorProductGrid creates Nx+1 points

# Time
T_final = 1.0  # Final time
dt = 0.01
Nt = int(T_final / dt)

# MFG parameters
sigma = 0.2  # Diffusion coefficient
congestion_weight = 1.0  # Weight of congestion term

# Exit expansion parameters
exit_initial_size = 0.1  # Initial exit width
exit_threshold = 0.5  # Density threshold for expansion
expansion_rate = 0.3  # Rate of exit growth when crowded

# Simplified iteration (demonstration)
max_iterations = 20  # Reduced for demonstration
# Note: Full MFG solve at each iteration is expensive
# This example uses simplified density evolution to demonstrate coupling

# ========================================
# Grid and Initial Conditions
# ========================================
print("\n" + "=" * 70)
print("Initialization")
print("=" * 70)

grid = TensorProductGrid(bounds=[(0, L)], Nx_points=[Nx + 1], boundary_conditions=no_flux_bc(dimension=1))
x = grid.coordinates[0]
N_points = len(x)  # Actual number of grid points (Nx+1)
dx = x[1] - x[0]

print(f"\nDomain: L = {L}, Nx = {Nx} intervals, N_points = {N_points}, dx = {dx:.4f}")
print(f"Time: T = {T_final}, dt = {dt}, Nt = {Nt}")
print(f"MFG: σ = {sigma}, congestion weight = {congestion_weight}")
print(f"Exit: initial size = {exit_initial_size}, threshold = {exit_threshold}")
print(f"      expansion rate = {expansion_rate}")

# Initial density: uniform in corridor
m0 = np.ones(N_points)
m0 = m0 / (np.sum(m0) * dx)  # Normalize to probability

# Initial exit: small region at right boundary
# Level set: φ < 0 is exit, φ > 0 is corridor
# Exit region: [L - exit_initial_size, L]
exit_left = L - exit_initial_size

# Signed distance function:
# φ(x) = exit_left - x  (negative for x > exit_left, i.e., inside exit)
phi0 = exit_left - x
exit_boundary = exit_left

# Create time-dependent domain for exit
ls_domain = TimeDependentDomain(phi0, grid, initial_time=0.0, is_signed_distance=True)

print("\nInitial conditions:")
print(f"  Total mass: {np.sum(m0) * dx:.6f} (expect 1.0)")
print(f"  Exit boundary: x = {exit_boundary:.3f}")
print(f"  Exit size: {exit_initial_size:.3f}")

# ========================================
# MFG Problem Setup
# ========================================
print("\n" + "=" * 70)
print("MFG Problem Configuration")
print("=" * 70)


def create_mfg_problem(phi_current: np.ndarray) -> MFGProblem:
    """
    Create MFG problem with current exit configuration.

    Exit defined by: phi(x) <= 0
    Terminal cost: 0 inside exit, large outside

    Uses Model + Conditions + SeparableHamiltonian (v1.0 API).
    The running cost g(m) = 1 + w*m is split:
    - constant 1: absorbed into base HJB structure
    - density-dependent w*m: the coupling term in SeparableHamiltonian
    """
    # Terminal value: 0 inside exit (phi <= 0), large penalty outside
    u_final = np.where(phi_current <= 0, 0.0, 100.0)

    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: congestion_weight * m,
        coupling_dm=lambda m: congestion_weight,
    )

    # Model: game rules (Hamiltonian + diffusion)
    model = Model(hamiltonian=hamiltonian, volatility=sigma)

    # Conditions: problem data (initial/terminal + time horizon)
    # Wrap arrays in callables for Conditions API
    m0_snapshot = m0.copy()
    u_final_snapshot = u_final.copy()
    conditions = Conditions(
        m_initial=lambda _x: m0_snapshot,
        u_terminal=lambda _x: u_final_snapshot,
        T=T_final,
    )

    problem = MFGProblem(
        model=model,
        domain=grid,
        conditions=conditions,
        Nt=Nt,
    )

    return problem


# Initial problem
problem = create_mfg_problem(phi0)
print("\nMFG problem created:")
print(f"  Geometry: 1D, {N_points} points")
print(f"  T = {problem.T}, Nt = {Nt}")
print(f"  Diffusion = {sigma}")

# ========================================
# Simplified Level Set Coupling Demonstration
# ========================================
print("\n" + "=" * 70)
print("Level Set Evolution (Simplified Coupling)")
print("=" * 70)

# Storage for tracking
exit_sizes = []
density_at_exit = []
velocities = []

# Level set evolver
ls_evolver = LevelSetEvolver(grid, scheme="upwind")

print("\nSimulating exit expansion...")
print(f"  Iterations: {max_iterations}")
print("  Note: Using simplified density model for demonstration")

# Simplified density evolution: crowd moves toward exit
m_current = m0.copy()

for iteration in range(max_iterations):
    # Get current exit configuration
    phi_current = ls_domain.current_phi

    # Simplified density model: agents drift toward exit
    # m increases near exit as agents converge
    exit_mask = phi_current <= 0.05  # Near-exit region

    # Simulate crowding: density increases near exit over time
    crowding_factor = 1.0 + 0.5 * iteration / max_iterations
    m_current[exit_mask] *= crowding_factor

    # Renormalize
    m_current = m_current / (np.sum(m_current) * dx)

    # Compute exit density (for expansion velocity)
    exit_interior = phi_current <= 0
    if np.any(exit_interior):
        m_exit = np.mean(m_current[exit_interior])
    else:
        m_exit = 0.0

    # Compute exit expansion velocity
    # V = expansion_rate * max(m_exit - threshold, 0)
    velocity_magnitude = expansion_rate * max(m_exit - exit_threshold, 0.0)

    # Velocity field (constant in space)
    V_field = velocity_magnitude * np.ones_like(phi_current)

    # Evolve level set (expand exit)
    if velocity_magnitude > 1e-6:
        ls_domain.evolve_step(V_field, dt, reinitialize=False, save_to_history=True)

    # Track metrics
    exit_size_current = np.sum(exit_interior) * dx
    exit_sizes.append(exit_size_current)
    density_at_exit.append(m_exit)
    velocities.append(velocity_magnitude)

    # Progress report
    if iteration % 5 == 0:
        print(
            f"  Iteration {iteration:2d}: m_exit = {m_exit:.4f}, "
            f"exit_size = {exit_size_current:.4f}, V = {velocity_magnitude:.6f}"
        )

print("\nFinal state:")
print(
    f"  Exit size: {exit_sizes[-1]:.4f} (initial: {exit_initial_size:.3f}, "
    f"{exit_sizes[-1] / exit_initial_size:.1f}× expansion)"
)
print(f"  Density at exit: {density_at_exit[-1]:.4f} (threshold: {exit_threshold:.2f})")
print(f"  Total iterations: {max_iterations}")

# ========================================
# Visualization
# ========================================
print("\n" + "=" * 70)
print("Creating Visualizations")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Final density
ax = axes[0, 0]
ax.plot(x, m_current, "r-", linewidth=2, label="Density m(x)")
ax.axvline(exit_boundary, color="k", linestyle="--", alpha=0.5, label="Initial exit")
exit_final = x[ls_domain.current_phi <= 0]
if len(exit_final) > 0:
    ax.axvspan(exit_final[0], exit_final[-1], alpha=0.2, color="green", label="Final exit")
ax.set_xlabel("Position x")
ax.set_ylabel("Density m")
ax.set_title("Final Agent Density")
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 2: Level set function
ax = axes[0, 1]
ax.plot(x, ls_domain.current_phi, "b-", linewidth=2, label="Level set φ(x)")
ax.axhline(0, color="k", linestyle="-", alpha=0.5, label="Exit boundary (φ=0)")
ax.fill_between(x, -5, 0, where=(ls_domain.current_phi <= 0), alpha=0.2, color="green", label="Exit region")
ax.set_xlabel("Position x")
ax.set_ylabel("Level Set φ")
ax.set_title("Final Level Set Configuration")
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 3: Exit size evolution
ax = axes[1, 0]
ax.plot(exit_sizes, "g-", linewidth=2, marker="o", markersize=4)
ax.axhline(exit_initial_size, color="k", linestyle="--", alpha=0.5, label="Initial size")
ax.set_xlabel("Picard Iteration")
ax.set_ylabel("Exit Size")
ax.set_title("Exit Expansion During Coupling")
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 4: Expansion velocity history
ax = axes[1, 1]
if len(velocities) > 0:
    ax.plot(velocities, "b-", linewidth=2, marker="s", markersize=4, label="Velocity")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Expansion Velocity V")
    ax.set_title("Exit Expansion Velocity History")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
output_dir = Path(__file__).parent.parent / "outputs" / "level_set_methods"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "mfg_expanding_exit.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nSaved figure: {output_path}")

plt.show()

# ========================================
# Summary
# ========================================
print("\n" + "=" * 70)
print("MFG Expanding Exit - Complete")
print("=" * 70)

print("\nPhysics Summary:")
print(f"  - Initial exit size: {exit_initial_size:.3f}")
print(f"  - Final exit size: {exit_sizes[-1]:.4f} ({exit_sizes[-1] / exit_initial_size:.1f}× expansion)")
print(f"  - Density threshold: {exit_threshold:.2f}")
print(f"  - Final exit density: {density_at_exit[-1]:.4f}")
print(f"  - Maximum expansion velocity: {max(velocities):.4f}")

print("\nKey Demonstration:")
print("  - Level set method tracks dynamic exit boundary")
print("  - Exit expands when density exceeds threshold (V ~ max(m - m_threshold, 0))")
print("  - Coupling mechanism: density -> velocity -> level set evolution")
print("  - Self-regulating: expansion reduces congestion -> velocity decreases")

print("\nNote:")
print("  This example uses simplified density model for demonstration.")
print("  Full MFG-LS coupling would solve HJB-FP system at each iteration.")

print("\n" + "=" * 70)

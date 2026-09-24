"""
Custom Hamiltonian Demo with Class-Based API.

Issue #673: Updated to class-based Hamiltonian API only.

This example demonstrates how to define custom Hamiltonians using the
class-based SeparableHamiltonian API - the recommended approach for MFG problems.

Mathematical Formulation:
    State: x in [0, L]

    Custom Hamiltonian:
        H(t, x, p, m) = (1/2 lambda)|p|^2 + |x - x_target| + gamma * m

    where:
    - (1/2 lambda)|p|^2 is the control cost (quadratic)
    - |x - x_target| is the running cost (distance to target)
    - gamma * m is the congestion cost

Key Features:
    - Class-based SeparableHamiltonian for composable, type-safe Hamiltonians
    - Auto-differentiation for dp() and dm() methods
    - Clean separation of control cost, potential, and coupling
    - Legendre transform duality support (H <-> L)

Usage:
    python examples/basic/core_infrastructure/custom_hamiltonian_derivs_demo.py

See Also:
    - mfgarchon/core/hamiltonian.py (class definitions)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.mfg_logging import configure_research_logging, get_logger

# Configure logging
configure_research_logging("custom_hamiltonian_derivs_demo", level="INFO")
logger = get_logger(__name__)

# Output directory
EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR.parent / "outputs" / "basic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Class-based Hamiltonian API (Recommended)
# ==============================================================================


def create_custom_problem(
    L: float = 10.0,
    target_x: float = 5.0,
    control_cost: float = 0.5,
    congestion_cost: float = 0.2,
    Nx: int = 101,
    T: float = 3.0,
    Nt: int = 51,
    sigma: float = 0.3,
):
    """
    Create custom MFG problem using the class-based Hamiltonian API.

    This demonstrates the recommended approach for custom Hamiltonians:
    - Use SeparableHamiltonian with control_cost, potential, and coupling
    - Pass to Model(hamiltonian=H) + Conditions(u_terminal=..., m_initial=...)

    The class-based approach offers:
    - Type-safe, composable Hamiltonians
    - Auto-differentiation for dp() and dm()
    - Legendre transform duality (H <-> L)
    - Clean separation of control cost, potential, and coupling

    Args:
        L: Domain length [0, L]
        target_x: Target position
        control_cost: Coefficient for control cost (lambda)
        congestion_cost: Coefficient for congestion (gamma)
        Nx: Number of spatial grid points
        T: Time horizon
        Nt: Number of time steps
        sigma: Diffusion coefficient

    Returns:
        MFGProblem with class-based Hamiltonian
    """

    # Define the potential V(x, t) = |x - target|
    def potential(x, t=0.0):
        return float(np.abs(x[0] - target_x))

    # Define the density coupling f(m) = gamma * m and its derivative df/dm = gamma
    def coupling(m):
        return congestion_cost * m

    def coupling_dm(m):
        return congestion_cost

    # Create class-based Hamiltonian: H = H_control(p) + V(x, t) + f(m)
    # H(t, x, p, m) = (1/2 lambda)|p|^2 + |x - target| + gamma * m
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=control_cost),
        potential=potential,
        coupling=coupling,
        coupling_dm=coupling_dm,
    )

    # v1.0 API: Model holds game rules, Conditions holds IC/TC + time horizon
    model = Model(hamiltonian=hamiltonian, sigma=sigma)
    conditions = Conditions(
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: np.exp(-((x - 0.5) ** 2) / 0.5),
        T=T,
    )

    # Create geometry
    domain = TensorProductGrid(
        bounds=[(0.0, L)],
        Nx_points=[Nx],
        boundary_conditions=no_flux_bc(dimension=1),
    )

    # Create problem
    problem = MFGProblem(
        model=model,
        domain=domain,
        conditions=conditions,
        Nt=Nt,
    )

    return problem


def demonstrate_hamiltonian_methods():
    """
    Demonstrate the methods available on class-based Hamiltonians.

    Shows:
    - H(t, x, p, m): Hamiltonian evaluation
    - dp(t, x, p, m): Optimal control (dH/dp)
    - dm(t, x, p, m): Coupling derivative (dH/dm)
    - legendre_transform(): Convert to Lagrangian
    """
    logger.info("\n" + "=" * 70)
    logger.info("Class-Based Hamiltonian Methods Demo")
    logger.info("=" * 70)

    # Create Hamiltonian
    control_cost = 0.5
    congestion_cost = 0.2
    target_x = 5.0

    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=control_cost),
        potential=lambda x, t=0.0: float(np.abs(x[0] - target_x)),
        coupling=lambda m: congestion_cost * m,
        coupling_dm=lambda m: congestion_cost,
    )

    # Test point
    x = np.array([3.0])  # Position
    m = 0.5  # Density
    p = np.array([1.0])  # Momentum
    t = 0.0  # Time

    logger.info("\nTest point:")
    logger.info(f"  x = {x[0]}, m = {m}, p = {p[0]}, t = {t}")

    # Evaluate Hamiltonian
    H_val = hamiltonian(x=x, m=m, p=p, t=t)
    logger.info(f"\nH(t, x, p, m) = {H_val:.4f}")
    logger.info(f"  Expected: (1/2*{control_cost})*{p[0]}^2 + |{x[0]}-{target_x}| + {congestion_cost}*{m}")
    expected = (p[0] ** 2) / (2 * control_cost) + abs(x[0] - target_x) + congestion_cost * m
    logger.info(f"  Computed: {expected:.4f}")

    # Optimal control (dH/dp)
    dp_val = hamiltonian.dp(x=x, m=m, p=p, t=t)
    logger.info(f"\ndH/dp (optimal control): {dp_val[0]:.4f}")
    logger.info(f"  Expected: p / lambda = {p[0] / control_cost:.4f}")

    # Coupling derivative (dH/dm)
    dm_val = hamiltonian.dm(x=x, m=m, p=p, t=t)
    logger.info(f"\ndH/dm (coupling derivative): {dm_val:.4f}")
    logger.info(f"  Expected: gamma = {congestion_cost}")

    # Legendre transform to Lagrangian
    logger.info("\nLegendre Transform:")
    lagrangian = hamiltonian.legendre_transform()
    logger.info(f"  L type: {type(lagrangian).__name__}")

    # Evaluate Lagrangian at optimal control
    alpha = dp_val  # Optimal control = dH/dp
    L_val = lagrangian(x=x, alpha=alpha, m=m, t=t)
    logger.info(f"  L(x, alpha*, m, t) = {L_val:.4f}")

    return hamiltonian


def run_mfg_problem():
    """
    Run the MFG problem and visualize results.
    """
    logger.info("\n" + "=" * 70)
    logger.info("Custom MFG Problem: Target Navigation with Congestion")
    logger.info("=" * 70)

    # Problem parameters
    L = 10.0
    target_x = 5.0
    Nx = 101
    T = 3.0
    Nt = 51

    logger.info("\nProblem Setup:")
    logger.info(f"  Domain: [0, {L}]")
    logger.info(f"  Target: x = {target_x}")
    logger.info(f"  Time horizon: T = {T}")
    logger.info(f"  Grid: {Nx} spatial points, {Nt} time steps")

    # Create and solve problem
    logger.info("\nCreating problem with class-based Hamiltonian...")
    problem = create_custom_problem(L=L, target_x=target_x, Nx=Nx, T=T, Nt=Nt)

    logger.info("Solving MFG system...")
    result = problem.solve(max_iterations=30, tolerance=1e-3)

    logger.info("\nResults:")
    logger.info(f"  Converged: {result.converged}")
    logger.info(f"  Iterations: {result.iterations}")
    if hasattr(result, "final_error_M"):
        logger.info(f"  Final error (M): {result.final_error_M:.2e}")
    if hasattr(result, "final_error_U"):
        logger.info(f"  Final error (U): {result.final_error_U:.2e}")

    # Visualization
    logger.info("\nGenerating visualizations...")

    x_grid = np.linspace(0, L, Nx + 1)
    t_grid = np.linspace(0, T, Nt + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Density evolution
    ax = axes[0, 0]
    for i, t_idx in enumerate([0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]):
        t_val = t_grid[t_idx]
        ax.plot(x_grid, result.M[t_idx, :], linewidth=2, label=f"t = {t_val:.1f}")
    ax.axvline(target_x, color="orange", linestyle="--", linewidth=1.5, alpha=0.7, label="Target")
    ax.set_xlabel("Position x", fontsize=11)
    ax.set_ylabel("Density m(x, t)", fontsize=11)
    ax.set_title("Density Evolution", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 2: Value function evolution
    ax = axes[0, 1]
    for i, t_idx in enumerate([0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]):
        t_val = t_grid[t_idx]
        ax.plot(x_grid, result.U[t_idx, :], linewidth=2, label=f"t = {t_val:.1f}")
    ax.set_xlabel("Position x", fontsize=11)
    ax.set_ylabel("Value u(x, t)", fontsize=11)
    ax.set_title("Value Function Evolution", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 3: Convergence history
    ax = axes[1, 0]
    ax.semilogy(
        range(1, result.iterations + 1),
        result.error_history_M[: result.iterations],
        "b-o",
        linewidth=2,
        markersize=4,
        label="Density error",
    )
    ax.semilogy(
        range(1, result.iterations + 1),
        result.error_history_U[: result.iterations],
        "r-s",
        linewidth=2,
        markersize=4,
        label="Value error",
    )
    ax.set_xlabel("Picard iteration", fontsize=11)
    ax.set_ylabel("L2 error", fontsize=11)
    ax.set_title("Convergence History", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 4: Space-time density heatmap
    ax = axes[1, 1]
    X, T_mesh = np.meshgrid(x_grid, t_grid)
    pcm = ax.pcolormesh(X, T_mesh, result.M, shading="auto", cmap="viridis")
    ax.axvline(target_x, color="red", linestyle="--", linewidth=2, label="Target")
    ax.set_xlabel("Position x", fontsize=11)
    ax.set_ylabel("Time t", fontsize=11)
    ax.set_title("Density m(x, t) Space-Time", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    plt.colorbar(pcm, ax=ax, label="Density")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "custom_hamiltonian_class_based.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"  Saved: {output_path}")

    # Show plot
    plt.show()

    return result


def main():
    """Run the demonstration."""
    logger.info("=" * 70)
    logger.info("Custom Hamiltonian Demo: Class-Based API")
    logger.info("=" * 70)
    logger.info("\nThis demo showcases the recommended class-based Hamiltonian API.")

    # Part 1: Demonstrate Hamiltonian methods
    demonstrate_hamiltonian_methods()

    # Part 2: Run MFG problem
    run_mfg_problem()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Demo Complete!")
    logger.info("=" * 70)
    logger.info("\nKey Takeaways:")
    logger.info("  1. Use SeparableHamiltonian for most MFG problems")
    logger.info("  2. Components: control_cost, potential, coupling, coupling_dm")
    logger.info("  3. Methods: H(), dp(), dm(), legendre_transform()")
    logger.info("  4. Pass to Model(hamiltonian=H) + Conditions(u_terminal=..., m_initial=...)")
    logger.info("\nSee Also:")
    logger.info("  - mfgarchon/core/hamiltonian.py (HamiltonianBase, SeparableHamiltonian)")
    logger.info("  - examples/basic/core_infrastructure/ (more examples)")


if __name__ == "__main__":
    main()

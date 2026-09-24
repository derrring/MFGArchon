"""
Towel on the Beach: Spatial Competition Mean Field Game.

This example demonstrates spatial competition where agents choose positions on a beach,
balancing proximity to an ice cream stall against crowding from other agents.

Mathematical Formulation:
    State: x in [0, L] (position on beach)
    Ice cream stall location: x_stall

    Running cost:
        L(x, u, m) = |x - x_stall| + lambda * log(m(x)) + (1/2)|u|^2
        [proximity]      [congestion]         [movement]

    HJB equation:
        -du/dt + H(x, du/dx, m) - (sigma^2/2) * d^2u/dx^2 = 0
        H(x, p, m) = (1/2)p^2 - |x - x_stall| - lambda * log(m)

    FPK equation:
        dm/dt - d/dx(m * du/dx) - (sigma^2/2) * d^2m/dx^2 = 0

Phase Transitions (controlled by lambda - crowd aversion parameter):
    - Low lambda (lambda < 1): Single peak at stall (proximity dominates)
    - Medium lambda (lambda ~ 1): Transition regime
    - High lambda (lambda > 1): Crater equilibrium (avoid crowding at stall)

Key Features:
    - Continuous spatial state
    - Proximity-congestion trade-off
    - Phase transitions via parameter lambda
    - Log-barrier congestion term

References:
    - docs/development/TOWEL_ON_BEACH_1D_PROTOCOL.md: Validation protocol
    - Lasry & Lions (2007): Mean field games with congestion

Example Usage:
    python examples/applications/economics/towel_beach_demo.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import HamiltonianBase
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import neumann_bc
from mfgarchon.utils.mfg_logging import configure_research_logging, get_logger

# Configure logging
configure_research_logging("towel_beach_demo", level="INFO")
logger = get_logger(__name__)

# Output directory
EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR.parent.parent / "outputs" / "applications"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class TowelBeachHamiltonian(HamiltonianBase):
    """
    Custom Hamiltonian for Towel-on-Beach problem.

    H(x, p, m) = (1/2)|p|^2 - |x - x_stall| - lambda * log(m)

    This implements the spatial competition model where:
    - (1/2)|p|^2: kinetic/control cost
    - -|x - x_stall|: proximity benefit to ice cream stall
    - -lambda * log(m): log-barrier congestion cost
    """

    def __init__(self, stall_position: float, crowd_aversion: float):
        super().__init__()
        self.stall_position = stall_position
        self.crowd_aversion = crowd_aversion

    def __call__(self, t: float, x: np.ndarray, p: np.ndarray, m: float) -> float:
        """Evaluate Hamiltonian H(t, x, p, m)."""
        # Extract scalar position
        x_scalar = float(x[0]) if hasattr(x, "__len__") else float(x)

        # Kinetic energy (quadratic control cost)
        p_norm_sq = float(np.sum(p**2))
        kinetic = 0.5 * p_norm_sq

        # Proximity cost to stall (negative = benefit)
        proximity = -abs(x_scalar - self.stall_position)

        # Congestion cost (log-barrier)
        m_reg = max(float(m), 1e-10)  # Regularization to avoid log(0)
        congestion = -self.crowd_aversion * np.log(m_reg)

        return kinetic + proximity + congestion

    def dm(self, t: float, x: np.ndarray, p: np.ndarray, m: float) -> float:
        """Derivative of Hamiltonian with respect to density: dH/dm = -lambda / m."""
        m_reg = max(float(m), 1e-10)
        return -self.crowd_aversion / m_reg

    def dp(self, t: float, x: np.ndarray, p: np.ndarray, m: float) -> np.ndarray:
        """Derivative of Hamiltonian with respect to momentum: dH/dp = p."""
        return np.atleast_1d(p)


def create_towel_beach_problem(
    beach_length: float = 10.0,
    stall_position: float = 5.0,
    crowd_aversion: float = 1.0,
    Nx: int = 201,
    T: float = 5.0,
    Nt: int = 101,
    sigma: float = 0.3,
):
    """
    Create Towel on Beach MFG problem using Model + Conditions API.

    Args:
        beach_length: Length of beach [0, L]
        stall_position: Ice cream stall location
        crowd_aversion: Congestion parameter lambda (controls phase transition)
        Nx: Number of spatial grid points
        T: Time horizon
        Nt: Number of time steps
        sigma: Diffusion coefficient

    Returns:
        MFGProblem for spatial competition
    """

    # Initial distribution: Uniform with small perturbation
    def initial_density_func(x):
        """Slightly perturbed uniform distribution."""
        return 1.0 + 0.1 * np.cos(2 * np.pi * x / beach_length)

    # Terminal value: zero (standard for running cost problems)
    def terminal_value_func(x):
        """Zero terminal cost."""
        return 0.0

    # Class-based Hamiltonian for Towel Beach problem
    hamiltonian = TowelBeachHamiltonian(
        stall_position=stall_position,
        crowd_aversion=crowd_aversion,
    )

    # Model: game rules (Hamiltonian + diffusion)
    model = Model(hamiltonian=hamiltonian, sigma=sigma)

    # Conditions: problem data (initial/terminal + time horizon)
    conditions = Conditions(
        m_initial=initial_density_func,
        u_terminal=terminal_value_func,
        T=T,
    )

    # Create geometry with Neumann (reflecting) boundary conditions
    domain = TensorProductGrid(
        bounds=[(0.0, beach_length)],
        Nx_points=[Nx],
        boundary_conditions=neumann_bc(dimension=1),
    )

    # Create problem with v1.0 API
    problem = MFGProblem(
        model=model,
        domain=domain,
        conditions=conditions,
        Nt=Nt,
    )

    return problem


def plot_phase_transition(beach_length, stall_position):
    """
    Demonstrate phase transition for different lambda values.

    Args:
        beach_length: Length of beach
        stall_position: Stall location
    """
    lambda_values = [0.5, 1.0, 2.0]  # Low, medium, high crowd aversion
    colors = ["blue", "green", "red"]
    labels = ["Low lambda=0.5 (peak)", "Medium lambda=1.0 (transition)", "High lambda=2.0 (crater)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (lam, color, label) in enumerate(zip(lambda_values, colors, labels, strict=False)):
        logger.info(f"\n  Solving for lambda = {lam}...")

        # Create and solve problem
        problem = create_towel_beach_problem(
            beach_length=beach_length,
            stall_position=stall_position,
            crowd_aversion=lam,
            Nx=151,
            T=5.0,
            Nt=51,
            sigma=0.3,
        )

        result = problem.solve(max_iterations=50, tolerance=1e-3, verbose=False)

        # Extract final equilibrium
        bounds = problem.geometry.get_bounds()
        Nx_points = problem.geometry.get_grid_shape()[0]
        x_grid = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        m_final = result.M[-1, :]

        # Plot
        ax = axes[idx]
        ax.plot(x_grid, m_final, color=color, linewidth=2.5)
        ax.axvline(stall_position, color="orange", linestyle="--", linewidth=2, label="Stall location")
        ax.fill_between(x_grid, 0, m_final, alpha=0.3, color=color)
        ax.set_xlabel("Beach position x", fontsize=11)
        ax.set_ylabel("Equilibrium density m(x)", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, beach_length])

        # Add annotation
        if lam < 1:
            ax.text(
                0.5,
                0.95,
                "Peak at stall",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
            )
        elif lam > 1.5:
            ax.text(
                0.5,
                0.95,
                "Crater at stall",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
            )

        logger.info(f"    Converged: {result.converged}, Iterations: {result.iterations}")

    plt.tight_layout()
    return fig


def main():
    """
    Demonstrate Towel on Beach spatial competition with phase transitions.
    """
    logger.info("=" * 70)
    logger.info("Towel on the Beach: Spatial Competition MFG")
    logger.info("=" * 70)

    beach_length = 10.0
    stall_position = 5.0

    logger.info(f"\n  Beach length: {beach_length}")
    logger.info(f"  Stall position: {stall_position}")
    logger.info("\n  Demonstrating phase transition for lambda in {0.5, 1.0, 2.0}...")

    # Generate phase transition plot
    fig = plot_phase_transition(beach_length, stall_position)

    output_path = OUTPUT_DIR / "towel_beach_phase_transition.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"\n  Saved: {output_path}")

    plt.show()

    logger.info("\n" + "=" * 70)
    logger.info("Towel Beach Demo Complete!")
    logger.info("=" * 70)
    logger.info("\nKey Insights:")
    logger.info("  - lambda < 1: Proximity dominates -> peak at stall")
    logger.info("  - lambda ~ 1: Transition regime")
    logger.info("  - lambda > 1: Crowding dominates -> crater at stall")
    logger.info("  - Phase transition controlled by crowd aversion parameter lambda")


if __name__ == "__main__":
    main()

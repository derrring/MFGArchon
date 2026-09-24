"""
Santa Fe Bar Problem: Continuous Preference Evolution in Coordination Games.

This example demonstrates the Santa Fe Bar problem (also called El Farol Bar)
using a continuous preference evolution formulation. Agents' attendance tendencies
evolve continuously based on payoff differences, creating emergent oscillations.

Mathematical Formulation:
    State: θ ∈ [-θ_max, θ_max] (attendance tendency)
    Aggregate attendance: m_attend = ∫_{θ>0} m(θ) dθ

    HJB equation:
        -∂u/∂t + H(θ, ∂u/∂θ, m_attend) - (σ²/2)∂²u/∂θ² = 0

    FPK equation:
        ∂m/∂t - ∂/∂θ(m · ∂u/∂θ) - (σ²/2)∂²m/∂θ² = 0

    Payoff function:
        Attend if θ > 0, stay home if θ < 0
        Payoff F(m_attend) = threshold-based

Key Differences from El Farol:
    - El Farol: Discrete states {home, bar}
    - Santa Fe: Continuous preference evolution θ(t)
    - Both exhibit same coordination paradox

References:
    - Arthur, W. B. (1994): "Inductive reasoning and bounded rationality"
    - Section 3 of coordination_games_mfg.md: Continuous formulation

Example Usage:
    python examples/basic/santa_fe_bar_demo.py
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
configure_research_logging("santa_fe_bar_demo", level="INFO")
logger = get_logger(__name__)

# Output directory
EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR.parent / "outputs" / "basic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class SantaFeHamiltonian(HamiltonianBase):
    """
    Hamiltonian for Santa Fe Bar continuous preference evolution.

    H(theta, p, m) = (1/2)|p|^2 + V(theta, m_attend)

    where V depends on whether the agent attends (theta > 0) and the
    aggregate attendance fraction m_attend (non-local coupling).
    """

    def __init__(self, payoff_function, attendance_cache: dict):
        super().__init__()
        self.payoff_function = payoff_function
        self.attendance_cache = attendance_cache

    def __call__(self, t: float, x: np.ndarray, p: np.ndarray, m: float) -> float:
        """Evaluate H(theta, m, p, t)."""
        x_arr = np.atleast_1d(x)
        p_arr = np.atleast_1d(p)
        theta = float(x_arr[0])
        p_val = float(p_arr[0])

        # Use cached aggregate attendance
        m_attend = self.attendance_cache["m_attend"]
        F = self.payoff_function(m_attend)

        # Potential: attend if theta > 0
        V = -F if theta > 0 else 0.0

        return 0.5 * p_val**2 + V

    def dm(self, t: float, x: np.ndarray, p: np.ndarray, m: float) -> float:
        """dH/dm: simplified (local changes have negligible effect on aggregate)."""
        return 0.0

    def dp(self, t: float, x: np.ndarray, p: np.ndarray, m: float) -> np.ndarray:
        """dH/dp = p (quadratic control cost)."""
        return np.atleast_1d(p)


def create_santa_fe_problem(
    threshold: float = 0.6,
    good_payoff: float = 10.0,
    bad_payoff: float = -5.0,
    theta_max: float = 5.0,
    Nx: int = 201,
    T: float = 10.0,
    Nt: int = 201,
    sigma: float = 0.5,
):
    """
    Create Santa Fe Bar MFG with continuous preference evolution using Model + Conditions API.

    Args:
        threshold: Attendance threshold (overcrowding if > threshold)
        good_payoff: Payoff when attendance < threshold
        bad_payoff: Payoff when attendance ≥ threshold
        theta_max: Maximum attendance tendency
        Nx: Number of grid points in preference space
        T: Time horizon
        Nt: Number of time steps
        sigma: Preference diffusion

    Returns:
        MFGProblem for continuous coordination game
    """

    def compute_attendance_fraction(m_distribution, theta_grid):
        """
        Compute fraction attending: m_attend = ∫_{θ>0} m(θ) dθ.

        Args:
            m_distribution: Density over preference space
            theta_grid: Grid of preference values

        Returns:
            Fraction of population attending (θ > 0)
        """
        # Find index where θ = 0
        zero_idx = np.argmin(np.abs(theta_grid))

        # Integrate density for θ > 0
        if zero_idx < len(m_distribution) - 1:
            m_attend = np.trapz(m_distribution[zero_idx + 1 :], theta_grid[zero_idx + 1 :])
        else:
            m_attend = 0.0

        return np.clip(m_attend, 0.0, 1.0)

    def payoff_function(m_attend):
        """
        Threshold-based payoff.

        F(m) = G if m < threshold, else B
        """
        epsilon = 0.05  # Smoothness
        transition = 1.0 / (1.0 + np.exp(-(m_attend - threshold) / epsilon))
        return good_payoff - (good_payoff - bad_payoff) * transition

    # Store theta_grid for use in Hamiltonian
    theta_grid_global = np.linspace(-theta_max, theta_max, Nx)

    # Cache for aggregate attendance (updated per iteration)
    attendance_cache = {"m_attend": 0.5}  # Start with neutral assumption

    # Initial distribution: Gaussian centered at θ=0 (no strong preference)
    def initial_density_func(theta):
        """Gaussian initial distribution."""
        return np.exp(-0.5 * (theta / 1.0) ** 2)

    # Terminal value: zero
    def terminal_value_func(theta):
        """Zero terminal cost."""
        return 0.0

    # Model: game rules (Hamiltonian + diffusion)
    hamiltonian = SantaFeHamiltonian(
        payoff_function=payoff_function,
        attendance_cache=attendance_cache,
    )
    model = Model(hamiltonian=hamiltonian, sigma=sigma)

    # Conditions: problem data (initial/terminal + time horizon)
    conditions = Conditions(
        m_initial=initial_density_func,
        u_terminal=terminal_value_func,
        T=T,
    )

    # Domain: preference space [-theta_max, theta_max]
    domain = TensorProductGrid(
        bounds=[(-theta_max, theta_max)],
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

    # Store helpers for external access
    problem._attendance_cache = attendance_cache
    problem._compute_attendance = lambda m: compute_attendance_fraction(m, theta_grid_global)

    return problem, theta_grid_global


def plot_results(t_grid, theta_grid, m_traj, attendance_traj, threshold):
    """
    Visualize preference evolution and attendance dynamics.

    Args:
        t_grid: Time grid
        theta_grid: Preference grid
        m_traj: Density trajectory m(t, θ)
        attendance_traj: Attendance fraction over time
        threshold: Overcrowding threshold
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Panel 1: Preference distribution heatmap
    ax1 = fig.add_subplot(gs[0, :])
    T_mesh, Theta_mesh = np.meshgrid(t_grid, theta_grid)
    im = ax1.pcolormesh(T_mesh, Theta_mesh, m_traj.T, shading="auto", cmap="hot")
    ax1.axhline(0, color="cyan", linestyle="--", linewidth=2, label="θ=0 (neutral)")
    ax1.set_xlabel("Time t", fontsize=12)
    ax1.set_ylabel("Preference θ", fontsize=12)
    ax1.set_title("Preference Distribution Evolution m(t, θ)", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right")
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label("Density", fontsize=10)

    # Panel 2: Attendance over time
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_grid, attendance_traj, "b-", linewidth=2.5, label="Bar attendance")
    ax2.axhline(threshold, color="r", linestyle="--", linewidth=2, label=f"Threshold ({threshold})")
    ax2.fill_between(t_grid, 0, threshold, alpha=0.2, color="green", label="Good")
    ax2.fill_between(t_grid, threshold, 1, alpha=0.2, color="red", label="Bad")
    ax2.set_xlabel("Time t", fontsize=12)
    ax2.set_ylabel("Attendance fraction", fontsize=12)
    ax2.set_title("Bar Attendance Dynamics", fontsize=13, fontweight="bold")
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # Panel 3: Preference distribution snapshots
    ax3 = fig.add_subplot(gs[1, 1])
    snapshot_times = [0, len(t_grid) // 3, 2 * len(t_grid) // 3, -1]
    colors = ["blue", "green", "orange", "red"]
    for idx, color in zip(snapshot_times, colors, strict=False):
        ax3.plot(theta_grid, m_traj[idx, :], color=color, linewidth=2, label=f"t={t_grid[idx]:.2f}")
    ax3.axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax3.set_xlabel("Preference θ", fontsize=12)
    ax3.set_ylabel("Density m(θ)", fontsize=12)
    ax3.set_title("Preference Distribution Snapshots", fontsize=13, fontweight="bold")
    ax3.legend(loc="best", fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Phase diagram
    ax4 = fig.add_subplot(gs[2, :])
    mean_preference = np.array([np.trapz(theta_grid * m_traj[i, :], theta_grid) for i in range(len(t_grid))])
    sc = ax4.scatter(attendance_traj, mean_preference, c=t_grid, cmap="viridis", s=30, alpha=0.7)
    ax4.axvline(threshold, color="r", linestyle="--", linewidth=2, alpha=0.7)
    ax4.axhline(0, color="k", linestyle="-", linewidth=0.8)
    ax4.set_xlabel("Bar attendance", fontsize=12)
    ax4.set_ylabel("Mean preference E[θ]", fontsize=12)
    ax4.set_title("Phase Portrait: Attendance vs Mean Preference", fontsize=13, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(sc, ax=ax4)
    cbar2.set_label("Time", fontsize=10)

    return fig


def main():
    """
    Solve Santa Fe Bar problem with continuous preference evolution.
    """
    logger.info("=" * 70)
    logger.info("Santa Fe Bar: Continuous Preference Evolution")
    logger.info("=" * 70)

    # Create problem
    logger.info("\n[1/3] Creating continuous coordination game...")
    problem, theta_grid = create_santa_fe_problem(
        threshold=0.6,
        good_payoff=10.0,
        bad_payoff=-5.0,
        theta_max=5.0,
        Nx=201,
        T=10.0,
        Nt=201,
        sigma=0.5,  # Still uses sigma parameter in function call
    )

    bounds = problem.geometry.get_bounds()
    logger.info(f"  Time horizon: T = {problem.T}")
    logger.info(f"  Preference space: θ ∈ [{bounds[0][0]}, {bounds[1][0]}]")
    logger.info("  Attendance threshold: 60%")
    logger.info(f"  Preference diffusion: σ = {problem.diffusion_coefficient}")

    # Solve
    logger.info("\n[2/3] Solving MFG system...")
    result = problem.solve(max_iterations=50, tolerance=1e-4)

    logger.info(f"  Converged: {result.converged}")
    logger.info(f"  Iterations: {result.iterations}")
    final_error_U = result.error_history_U[result.iterations - 1] if result.iterations > 0 else np.inf
    final_error_M = result.error_history_M[result.iterations - 1] if result.iterations > 0 else np.inf
    logger.info(f"  Final error: U={final_error_U:.2e}, M={final_error_M:.2e}")

    # Compute attendance trajectory
    t_grid = problem.tSpace
    m_traj = result.M  # Use result.M instead of result.m_traj
    zero_idx = np.argmin(np.abs(theta_grid))

    attendance_traj = np.array(
        [
            np.trapz(m_traj[i, zero_idx + 1 :], theta_grid[zero_idx + 1 :]) if zero_idx < len(theta_grid) - 1 else 0.0
            for i in range(len(t_grid))
        ]
    )
    attendance_traj = np.clip(attendance_traj, 0, 1)

    # Analysis
    logger.info("\n[3/3] Analysis:")
    mean_attendance = np.mean(attendance_traj)
    std_attendance = np.std(attendance_traj)
    time_overcrowded = np.mean(attendance_traj > 0.6)

    logger.info(f"  Mean attendance: {mean_attendance:.1%}")
    logger.info(f"  Std attendance: {std_attendance:.1%}")
    logger.info(f"  Time overcrowded: {time_overcrowded:.1%}")
    logger.info(f"  Attendance range: [{attendance_traj.min():.1%}, {attendance_traj.max():.1%}]")

    # Visualize
    logger.info("\nGenerating visualizations...")
    fig = plot_results(t_grid, theta_grid, m_traj, attendance_traj, threshold=0.6)

    output_path = OUTPUT_DIR / "santa_fe_bar_continuous.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"  Saved: {output_path}")

    plt.show()

    logger.info("\n" + "=" * 70)
    logger.info("Santa Fe Bar Demo Complete!")
    logger.info("=" * 70)
    logger.info("\nKey Insights:")
    logger.info("  - Continuous preferences evolve via diffusion + drift")
    logger.info("  - Attendance = ∫_{θ>0} m(θ) dθ (fraction with positive preference)")
    logger.info("  - Same coordination paradox as discrete El Farol")
    logger.info("  - Richer dynamics with preference distribution evolution")


if __name__ == "__main__":
    main()

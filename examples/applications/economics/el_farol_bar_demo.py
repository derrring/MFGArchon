"""
El Farol Bar Problem: Discrete State Mean Field Game.

This example demonstrates the classic coordination game in MFG theory, where agents
decide whether to attend a bar based on expected attendance. The bar is enjoyable
only if attendance is below a threshold (60%), creating a self-referential paradox.

Mathematical Formulation:
    States: {0 = home, 1 = bar}
    Population fraction at bar: m(t) ∈ [0,1]

    HJB equations (backward in time):
        -du₁/dt = F(m(t)) - ν log(1 + exp((u₀ - u₁)/ν))
        -du₀/dt = U_home - ν log(1 + exp((u₁ - u₀)/ν))

    FPK equation (forward in time):
        dm/dt = (1-m)·P₀→₁ - m·P₁→₀

    where transition probabilities use logit choice:
        P₀→₁ = exp(u₁/ν)/(exp(u₀/ν) + exp(u₁/ν))

    Payoff function:
        F(m) = G if m < m_threshold (good time)
               B if m ≥ m_threshold (bad time)

Key Features:
    - Discrete state space (binary choice)
    - Threshold-based payoff with phase transition
    - Logit dynamics (bounded rationality)
    - Emergence of stable attendance oscillations

References:
    - Arthur, W. B. (1994). "Inductive reasoning and bounded rationality."
    - Lasry & Lions (2007): Mean field games

Example Usage:
    python examples/basic/el_farol_bar_demo.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from mfgarchon.alg.numerical.coupling.network_mfg_solver import create_simple_network_solver
from mfgarchon.extensions.topology import NetworkMFGComponents, NetworkMFGProblem
from mfgarchon.geometry.graph.network_geometry import BaseNetworkGeometry, NetworkData, NetworkType
from mfgarchon.utils.mfg_logging import configure_research_logging, get_logger

# Configure logging
configure_research_logging("el_farol_bar_demo", level="INFO")
logger = get_logger(__name__)

# Output directory
EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR.parent / "outputs" / "basic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class TwoNodeNetwork(BaseNetworkGeometry):
    """Simple 2-node network for binary choice problems."""

    def __init__(self):
        super().__init__(num_nodes=2, network_type=NetworkType.CUSTOM)

    def create_network(self, **kwargs) -> NetworkData:
        """Create a 2-node bidirectional network."""
        # Adjacency matrix: home (0) <--> bar (1)
        adj = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))

        # Node positions for visualization
        positions = np.array([[0.0, 0.0], [1.0, 0.0]])

        # Create network data
        network_data = NetworkData(
            adjacency_matrix=adj,
            num_nodes=2,
            num_edges=1,  # One undirected edge
            network_type=NetworkType.CUSTOM,
            node_positions=positions,
            is_directed=False,
        )

        self.network_data = network_data
        return network_data

    def compute_distance_matrix(self) -> np.ndarray:
        """Compute shortest path distances (all paths are length 1)."""
        return np.array([[0.0, 1.0], [1.0, 0.0]])


def create_el_farol_problem(
    threshold: float = 0.6,
    good_payoff: float = 10.0,
    bad_payoff: float = -5.0,
    home_utility: float = 0.0,
    T: float = 10.0,
    Nt: int = 201,
):
    """
    Create El Farol Bar MFG problem as 2-node network.

    Args:
        threshold: Attendance threshold for overcrowding (default 0.6 = 60%)
        good_payoff: Payoff when attendance < threshold
        bad_payoff: Payoff when attendance ≥ threshold
        home_utility: Utility of staying home (baseline)
        T: Time horizon
        Nt: Number of time steps

    Returns:
        NetworkMFGProblem with 2 nodes (home, bar)
    """
    # Create 2-node network: home (0) <--> bar (1)
    network = TwoNodeNetwork()
    network.create_network()

    def terminal_value_func(node: int) -> float:
        """Terminal cost (zero for both states)."""
        return 0.0

    def initial_density_func(node: int) -> float:
        """Start with 50% home, 50% bar."""
        return 0.5

    def node_potential_func(node: int, t: float) -> float:
        """Running cost at each node (before interaction)."""
        if node == 0:  # Home
            return -home_utility  # Negative utility = positive cost
        else:  # Bar
            return 0.0  # Base cost for being at bar

    def node_interaction_func(node: int, m: np.ndarray, t: float) -> float:
        """
        Interaction cost based on bar attendance.

        Args:
            node: Current node (0=home, 1=bar)
            m: Density distribution [m_home, m_bar]
            t: Current time

        Returns:
            Interaction cost (negative of payoff)
        """
        if node == 0:  # Home - no interaction
            return 0.0
        else:  # Bar - threshold-based payoff
            m_bar = m[1]  # Attendance fraction at bar

            # Smooth approximation of threshold function
            epsilon = 0.05
            transition = 1.0 / (1.0 + np.exp(-(m_bar - threshold) / epsilon))
            payoff = good_payoff - (good_payoff - bad_payoff) * transition

            return -payoff  # Cost = -payoff

    # Create problem components
    components = NetworkMFGComponents(
        terminal_node_value_func=terminal_value_func,
        initial_node_density_func=initial_density_func,
        node_potential_func=node_potential_func,
        node_interaction_func=node_interaction_func,
        diffusion_coefficient=0.05,  # Small diffusion for discrete states
        drift_coefficient=1.0,
    )

    # Create problem
    problem = NetworkMFGProblem(
        geometry=network,
        T=T,
        Nt=Nt,
        components=components,
        problem_name="ElFarolBar",
    )

    return problem


def plot_results(t_grid, m_bar, u_home, u_bar, threshold=0.6):
    """
    Visualize El Farol Bar dynamics.

    Args:
        t_grid: Time grid
        m_bar: Bar attendance trajectory m(t)
        u_home: Value function for staying home
        u_bar: Value function for going to bar
        threshold: Overcrowding threshold
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Bar attendance over time
    ax = axes[0, 0]
    ax.plot(t_grid, m_bar, "b-", linewidth=2, label="Bar attendance m(t)")
    ax.axhline(threshold, color="r", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})")
    ax.fill_between(t_grid, 0, threshold, alpha=0.2, color="green", label="Good (uncrowded)")
    ax.fill_between(t_grid, threshold, 1, alpha=0.2, color="red", label="Bad (crowded)")
    ax.set_xlabel("Time t", fontsize=12)
    ax.set_ylabel("Attendance fraction", fontsize=12)
    ax.set_title("Bar Attendance Dynamics", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Panel 2: Value functions
    ax = axes[0, 1]
    ax.plot(t_grid, u_home, "g-", linewidth=2, label="u₀(t) [home]")
    ax.plot(t_grid, u_bar, "b-", linewidth=2, label="u₁(t) [bar]")
    ax.set_xlabel("Time t", fontsize=12)
    ax.set_ylabel("Value function", fontsize=12)
    ax.set_title("Optimal Values: Home vs Bar", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 3: Value difference (decision signal)
    ax = axes[1, 0]
    value_diff = u_bar - u_home
    ax.plot(t_grid, value_diff, "purple", linewidth=2)
    ax.axhline(0, color="k", linestyle="-", linewidth=0.8)
    ax.fill_between(t_grid, 0, value_diff, where=(value_diff > 0), alpha=0.3, color="blue", label="Bar preferred")
    ax.fill_between(t_grid, value_diff, 0, where=(value_diff < 0), alpha=0.3, color="green", label="Home preferred")
    ax.set_xlabel("Time t", fontsize=12)
    ax.set_ylabel("u₁ - u₀", fontsize=12)
    ax.set_title("Decision Signal: Bar vs Home", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 4: Phase diagram (attendance vs value difference)
    ax = axes[1, 1]
    sc = ax.scatter(m_bar, value_diff, c=t_grid, cmap="viridis", s=20, alpha=0.7)
    ax.axvline(threshold, color="r", linestyle="--", linewidth=1.5, label="Threshold")
    ax.axhline(0, color="k", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Bar attendance m(t)", fontsize=12)
    ax.set_ylabel("Value difference u₁ - u₀", fontsize=12)
    ax.set_title("Phase Portrait", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Time", fontsize=10)

    plt.tight_layout()
    return fig


def main():
    """
    Solve El Farol Bar problem and visualize results.
    """
    logger.info("=" * 70)
    logger.info("El Farol Bar Problem: Discrete State Mean Field Game")
    logger.info("=" * 70)

    # Create problem
    logger.info("\n[1/3] Creating El Farol Bar problem...")
    problem = create_el_farol_problem(
        threshold=0.6,  # 60% capacity
        good_payoff=10.0,
        bad_payoff=-5.0,
        home_utility=0.0,
        T=10.0,
        Nt=201,
    )

    logger.info(f"  Time horizon: T = {problem.T}")
    logger.info("  Attendance threshold: 60%")
    logger.info("  Payoff: 10.0 (good) vs -5.0 (bad)")
    logger.info("  Network: 2 nodes (home <--> bar)")

    # Solve using network MFG solver
    logger.info("\n[2/3] Solving MFG system...")
    solver = create_simple_network_solver(problem, scheme="implicit", damping=0.6)

    # solve() returns (U, M, iterations, error_U_history, error_M_history)
    U, M, iterations, error_U_hist, error_M_hist = solver.solve(max_iterations=100, tolerance=1e-6)

    converged = error_U_hist[iterations - 1] < 1e-6 and error_M_hist[iterations - 1] < 1e-6
    final_error = max(error_U_hist[iterations - 1], error_M_hist[iterations - 1])

    logger.info(f"  Converged: {converged}")
    logger.info(f"  Iterations: {iterations}")
    logger.info(f"  Final error: {final_error:.2e}")

    # Extract trajectories
    t_grid = np.linspace(0, problem.T, problem.Nt + 1)
    m_bar = M[:, 1]  # Fraction at bar (node 1)
    u_home = U[:, 0]  # Value of home (node 0)
    u_bar = U[:, 1]  # Value of bar (node 1)

    # Summary statistics
    logger.info("\n[3/3] Analysis:")
    mean_attendance = np.mean(m_bar)
    time_above_threshold = np.mean(m_bar > 0.6)
    logger.info(f"  Mean attendance: {mean_attendance:.1%}")
    logger.info(f"  Time overcrowded: {time_above_threshold:.1%}")
    logger.info(f"  Attendance range: [{m_bar.min():.1%}, {m_bar.max():.1%}]")

    # Visualize
    logger.info("\nGenerating visualizations...")
    fig = plot_results(t_grid, m_bar, u_home, u_bar, threshold=0.6)

    output_path = OUTPUT_DIR / "el_farol_bar_dynamics.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"  Saved: {output_path}")

    plt.show()

    logger.info("\n" + "=" * 70)
    logger.info("El Farol Bar Demo Complete!")
    logger.info("=" * 70)
    logger.info("\nKey Insight:")
    logger.info("  The equilibrium attendance oscillates around the threshold,")
    logger.info("  demonstrating the fundamental coordination paradox:")
    logger.info("  No universally correct prediction exists!")


if __name__ == "__main__":
    main()

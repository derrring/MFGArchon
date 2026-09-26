"""
Linear-Quadratic MFG with Common Noise.

This example demonstrates solving a stochastic Mean Field Game with common noise
using Monte Carlo methods. We use a Linear-Quadratic (LQ) problem where market
volatility (modeled as an Ornstein-Uhlenbeck process) affects all agents.

Mathematical Formulation:
    State dynamics: dx_t = α_t dt + σ dW_t  (individual noise)
    Common noise: dθ_t = κ(μ - θ_t) dt + σ_θ dB_t  (OU process)

    Conditional cost:
        J[α] = E[∫_0^T (|x_t - x_ref|² + λ(θ_t)|α_t|² + γm(t,x_t)) dt + g(x_T)]

    where λ(θ_t) = λ_0(1 + β·θ_t) models risk-aversion depending on market volatility.

Solution approach:
    1. Sample K noise paths θ^k_t from OU process
    2. Solve conditional MFG for each noise realization
    3. Aggregate via Monte Carlo averaging

References:
    - Carmona & Delarue (2018): Probabilistic Theory of Mean Field Games
    - Issue #68: Phase 2.2 Stochastic MFG Extensions
"""

import matplotlib.pyplot as plt
import numpy as np

from mfgarchon.alg.numerical.stochastic import CommonNoiseMFGSolver
from mfgarchon.core.stochastic import OrnsteinUhlenbeckProcess, StochasticMFGProblem
from mfgarchon.utils.mfg_logging import configure_research_logging, get_logger

# Configure logging
configure_research_logging("common_noise_lq_demo", level="INFO")
logger = get_logger(__name__)


# LQ cost parameters (module-level for pickling in parallel execution)
X_REF = 0.0  # Reference position
LAMBDA_0 = 1.0  # Base control cost
BETA = 0.3  # Sensitivity to volatility
GAMMA = 0.1  # Congestion cost


def conditional_hamiltonian(x, p, m, theta):
    """
    Hamiltonian with volatility-dependent control cost.

    H(x, p, m, θ) = p²/(2λ(θ)) + (x-x_ref)²/2 + γm

    where λ(θ) = λ_0(1 + β|θ|) models risk-aversion depending on market volatility.

    Args:
        x: State
        p: Co-state (momentum)
        m: Population density
        theta: Common noise (market volatility)

    Returns:
        Hamiltonian value
    """
    lambda_theta = LAMBDA_0 * (1.0 + BETA * abs(theta))  # Risk aversion increases with |θ|
    kinetic_term = p**2 / (2.0 * lambda_theta)
    potential_term = 0.5 * (x - X_REF) ** 2
    congestion_term = GAMMA * m
    return kinetic_term + potential_term + congestion_term


def terminal_cost(x):
    """
    Terminal cost function.

    g(x) = (x - x_ref)²/2

    Args:
        x: Final state

    Returns:
        Terminal cost
    """
    return 0.5 * (x - X_REF) ** 2


def create_market_volatility_problem():
    """
    Create LQ-MFG problem with market volatility as common noise.

    Returns:
        StochasticMFGProblem with OU process representing market volatility
    """
    # Problem parameters
    xmin, xmax = -3.0, 3.0
    Nx = 101
    T = 1.0
    Nt = 101
    sigma = 0.3  # Individual agent diffusion

    # Market volatility parameters
    kappa = 2.0  # Mean reversion speed
    mu_theta = 0.0  # Long-term mean volatility
    sigma_theta = 0.5  # Volatility of volatility

    # Create OU process for market volatility
    market_volatility = OrnsteinUhlenbeckProcess(kappa=kappa, mu=mu_theta, sigma=sigma_theta)

    # Create stochastic problem (simplified API, no MFGComponents needed)
    problem = StochasticMFGProblem(
        xmin=xmin,
        xmax=xmax,
        Nx=Nx,
        T=T,
        Nt=Nt,
        volatility=sigma,
        noise_process=market_volatility,
        conditional_hamiltonian=conditional_hamiltonian,
        theta_initial=0.0,  # Start at mean volatility
    )

    # Set initial density: Gaussian centered at x=1
    x = np.linspace(xmin, xmax, Nx)
    rho0 = np.exp(-((x - 1.0) ** 2) / 0.5)
    rho0 /= np.trapezoid(rho0, x)  # Normalize
    problem.rho0 = rho0

    # Set terminal cost (using module-level function)
    problem.terminal_cost = terminal_cost

    logger.info("Created LQ-MFG problem with market volatility")
    logger.info(f"  Domain: [{xmin}, {xmax}] with {Nx} points")
    logger.info(f"  Time: [0, {T}] with {Nt} steps")
    logger.info(f"  Market volatility: OU(κ={kappa}, μ={mu_theta}, σ={sigma_theta})")
    logger.info(f"  Control cost: λ(θ) = {LAMBDA_0}(1 + {BETA}|θ|)")

    return problem


def solve_and_visualize(num_noise_samples=20, use_variance_reduction=True, parallel=True):
    """
    Solve common noise MFG and visualize results.

    Args:
        num_noise_samples: Number of Monte Carlo samples K
        use_variance_reduction: Use quasi-Monte Carlo
        parallel: Solve noise realizations in parallel
    """
    logger.info(f"\n{'=' * 70}")
    logger.info("Common Noise MFG Demo: Market Volatility in LQ Problem")
    logger.info(f"{'=' * 70}\n")

    # Create problem
    problem = create_market_volatility_problem()

    # Create solver
    solver = CommonNoiseMFGSolver(
        problem=problem,
        num_noise_samples=num_noise_samples,
        variance_reduction=use_variance_reduction,
        parallel=parallel,
        seed=42,  # For reproducibility
    )

    logger.info("\nSolver configuration:")
    logger.info(f"  Noise samples: {num_noise_samples}")
    logger.info(f"  Variance reduction: {use_variance_reduction}")
    logger.info(f"  Parallel execution: {parallel}")

    # Solve
    logger.info("\n" + "-" * 70)
    logger.info("Starting Monte Carlo solve...")
    logger.info("-" * 70 + "\n")

    result = solver.solve(verbose=True)

    logger.info("\n" + "-" * 70)
    logger.info("Solution Summary")
    logger.info("-" * 70)
    logger.info(f"  Computation time: {result.computation_time:.2f}s")
    logger.info(f"  MC error (u): {result.mc_error_u:.6e}")
    logger.info(f"  MC error (m): {result.mc_error_m:.6e}")
    logger.info(f"  Variance reduction: {result.variance_reduction_factor:.2f}x")
    logger.info(f"  All converged: {result.converged}")

    # Visualize results
    visualize_results(problem, result)


def visualize_results(problem, result):
    """
    Create comprehensive visualization of common noise MFG solution.

    Args:
        problem: StochasticMFGProblem instance
        result: CommonNoiseMFGResult from solver
    """
    x_grid = problem.x
    t_grid = problem.t

    # Create 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Common Noise MFG: Market Volatility Effect", fontsize=16, fontweight="bold")

    # 1. Sample noise paths
    ax = axes[0, 0]
    for k, noise_path in enumerate(result.noise_paths[:10]):  # Show first 10 paths
        ax.plot(t_grid, noise_path, alpha=0.3, color="steelblue", linewidth=1)
    ax.plot(t_grid, result.noise_paths[0], color="darkblue", linewidth=2, label="Sample path")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Mean μ=0")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Volatility θ_t")
    ax.set_title(f"Market Volatility Paths (K={result.num_noise_samples})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Mean value function E[u(t,x)]
    ax = axes[0, 1]
    X, T_mesh = np.meshgrid(x_grid, t_grid)
    contour = ax.contourf(X, T_mesh, result.u_mean, levels=20, cmap="RdYlBu_r")
    ax.set_xlabel("Position x")
    ax.set_ylabel("Time t")
    ax.set_title("Mean Value Function E[u^θ(t,x)]")
    plt.colorbar(contour, ax=ax, label="u")

    # 3. Mean density E[m(t,x)]
    ax = axes[0, 2]
    contour = ax.contourf(X, T_mesh, result.m_mean, levels=20, cmap="YlOrRd")
    ax.set_xlabel("Position x")
    ax.set_ylabel("Time t")
    ax.set_title("Mean Density E[m^θ(t,x)]")
    plt.colorbar(contour, ax=ax, label="m")

    # 4. Uncertainty in value function (std)
    ax = axes[1, 0]
    contour = ax.contourf(X, T_mesh, result.u_std, levels=20, cmap="Purples")
    ax.set_xlabel("Position x")
    ax.set_ylabel("Time t")
    ax.set_title("Value Function Uncertainty Std[u^θ]")
    plt.colorbar(contour, ax=ax, label="σ_u")

    # 5. Uncertainty in density (std)
    ax = axes[1, 1]
    contour = ax.contourf(X, T_mesh, result.m_std, levels=20, cmap="Purples")
    ax.set_xlabel("Position x")
    ax.set_ylabel("Time t")
    ax.set_title("Density Uncertainty Std[m^θ]")
    plt.colorbar(contour, ax=ax, label="σ_m")

    # 6. Confidence intervals at final time
    ax = axes[1, 2]
    u_final_mean = result.u_mean[-1, :]
    m_final_mean = result.m_mean[-1, :]

    u_lower, u_upper = result.get_confidence_interval_u(confidence=0.95)
    m_lower, m_upper = result.get_confidence_interval_m(confidence=0.95)

    # Plot u with confidence band
    ax_twin = ax.twinx()
    ax.plot(x_grid, u_final_mean, "b-", linewidth=2, label="E[u(T,x)]")
    ax.fill_between(x_grid, u_lower[-1, :], u_upper[-1, :], alpha=0.3, color="blue", label="95% CI (u)")

    # Plot m on twin axis
    ax_twin.plot(x_grid, m_final_mean, "r-", linewidth=2, label="E[m(T,x)]")
    ax_twin.fill_between(x_grid, m_lower[-1, :], m_upper[-1, :], alpha=0.3, color="red", label="95% CI (m)")

    ax.set_xlabel("Position x")
    ax.set_ylabel("Value u(T,x)", color="blue")
    ax_twin.set_ylabel("Density m(T,x)", color="red")
    ax.set_title("Final Time Profiles with 95% CI")
    ax.tick_params(axis="y", labelcolor="blue")
    ax_twin.tick_params(axis="y", labelcolor="red")
    ax.legend(loc="upper left")
    ax_twin.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("common_noise_lq_demo.png", dpi=150, bbox_inches="tight")
    logger.info("\nVisualization saved to: common_noise_lq_demo.png")
    plt.show()


if __name__ == "__main__":
    # Run demo with moderate number of samples
    # Note: parallel=False due to pickling issues with lambda functions in CommonNoiseMFGSolver
    solve_and_visualize(num_noise_samples=20, use_variance_reduction=True, parallel=False)

    logger.info("\n" + "=" * 70)
    logger.info("Demo completed successfully!")
    logger.info("=" * 70)

"""
2D Epidemic Modeling with Mean Field Games.

This example demonstrates optimal disease containment strategies using:
- 2D spatial domain for population movement
- Tensor product grids for efficient epidemic modeling
- Sparse matrix operations for large-scale spatial systems
- Multi-dimensional visualization for disease spread patterns

Problem Description:
    Individuals navigate a 2D spatial region while minimizing:
    - Risk of infection from local disease prevalence
    - Cost of mobility restrictions (social distancing)
    - Economic/social costs of movement limitations

Mathematical Model:
    State: (x,y) ∈ [0,L]×[0,L] (spatial position)

    Dynamics: Individuals move to balance infection risk vs mobility benefits

    HJB Equation:
        -∂u/∂t + H(∇u, I(x,y,t)) + σ²Δu = 0
        u(T,x,y) = g(x,y)

    where H(p,I) models:
        - Mobility control cost: ½λ|p|²
        - Infection risk: β·I(x,y,t) (depends on local infected density)
        - Economic activity reward: -f(x,y)

    Disease Dynamics (Simplified):
        ∂I/∂t = α·S·I - γ·I + spatial_diffusion
        where:
        - S: Susceptible density (inferred from total - infected)
        - I: Infected density (mean field variable)
        - α: Infection rate
        - γ: Recovery rate

    Fokker-Planck (Population Movement):
        ∂m/∂t - ∇·(m v*) + σ²Δm = 0
        m(0,x,y) = m₀(x,y)

Application Context:
    - COVID-19 spatial modeling and containment
    - Optimal lockdown policies
    - Epidemic control with mobility restrictions
    - Public health resource allocation

References:
    - Acemoglu et al. (2021): Optimal Targeted Lockdowns in Epidemic Model
    - Elie et al. (2020): Contact Rate Epidemic Control of COVID-19
    - Bertuzzo et al. (2020): Spatial Spread of COVID-19
"""

from pathlib import Path

import numpy as np

# Multi-dimensional infrastructure
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils import SparseMatrixBuilder, SparseSolver


def create_epidemic_problem():
    """
    Create 2D epidemic MFG problem.

    Returns:
        Dictionary with problem parameters and grid
    """
    # Spatial domain: [0,L]×[0,L] km (city/region)
    L = 20.0  # 20 km × 20 km region
    Nx = Ny = 51  # 51×51 spatial grid

    # Time horizon
    T = 30.0  # 30 days
    Nt = 60

    # Create 2D tensor product grid
    grid = TensorProductGrid(
        bounds=[(0.0, L), (0.0, L)],
        Nx_points=[Nx, Ny],
        boundary_conditions=no_flux_bc(dimension=2),
    )

    # Epidemic parameters
    alpha = 0.3  # Infection rate (contact × transmission)
    beta_infection = 2.0  # Cost coefficient for infection risk
    gamma_recovery = 0.1  # Recovery rate (1/gamma = 10 days recovery)

    # Movement parameters
    sigma = 0.5  # Population diffusion
    lambda_mobility = 1.0  # Cost of mobility control

    # Economic parameters
    kappa_economic = 0.5  # Economic activity value at certain locations

    # Initial infection parameters
    I0_center = (L / 4, L / 4)  # Initial outbreak location
    I0_fraction = 0.05  # 5% initially infected in outbreak zone

    return {
        "grid": grid,
        "T": T,
        "Nt": Nt,
        "alpha": alpha,
        "beta_infection": beta_infection,
        "gamma_recovery": gamma_recovery,
        "sigma": sigma,
        "lambda_mobility": lambda_mobility,
        "kappa_economic": kappa_economic,
        "I0_center": I0_center,
        "I0_fraction": I0_fraction,
        "L": L,
    }


def initial_population_distribution(grid):
    """
    Initial population distribution m₀(x,y).

    Population concentrated in urban centers.
    """
    X, Y = grid.meshgrid(indexing="ij")
    L = grid.bounds[0][1]

    # Two urban centers: one at (L/4, 3L/4), one at (3L/4, L/4)
    center1 = (L / 4, 3 * L / 4)
    center2 = (3 * L / 4, L / 4)
    sigma_urban = 2.0

    # Bimodal population distribution
    m0 = 0.6 * np.exp(-((X - center1[0]) ** 2 + (Y - center1[1]) ** 2) / (2 * sigma_urban**2)) + 0.4 * np.exp(
        -((X - center2[0]) ** 2 + (Y - center2[1]) ** 2) / (2 * sigma_urban**2)
    )

    # Normalize to probability distribution
    m0 = m0 / (np.sum(m0) * grid.volume_element())

    return m0


def initial_infection_distribution(grid, I0_center, I0_fraction):
    """
    Initial infected distribution I₀(x,y).

    Localized outbreak at specified center.
    """
    X, Y = grid.meshgrid(indexing="ij")

    # Gaussian outbreak centered at I0_center
    sigma_outbreak = 1.5
    I0 = I0_fraction * np.exp(-((X - I0_center[0]) ** 2 + (Y - I0_center[1]) ** 2) / (2 * sigma_outbreak**2))

    # Normalize
    I0 = I0 / (np.sum(I0) * grid.volume_element()) * I0_fraction

    return I0


def terminal_cost(grid, I0_center):
    """
    Terminal cost g(x,y): Penalty for being in high-risk zones.

    Higher cost near initial outbreak location.
    """
    X, Y = grid.meshgrid(indexing="ij")

    # Quadratic penalty for proximity to outbreak zone
    g = 0.5 * ((X - I0_center[0]) ** 2 + (Y - I0_center[1]) ** 2)

    return g


def economic_activity_map(grid, L, kappa):
    """
    Economic activity value f(x,y).

    Higher value in city centers (work locations).
    """
    X, Y = grid.meshgrid(indexing="ij")

    # Economic centers at (L/2, L/2) - central business district
    center = (L / 2, L / 2)
    sigma_econ = 3.0

    # Gaussian concentration of economic activity
    f = kappa * np.exp(-((X - center[0]) ** 2 + (Y - center[1]) ** 2) / (2 * sigma_econ**2))

    return f


def hamiltonian(px, py, infected, f_econ, lambda_mobility, beta_infection):
    """
    Hamiltonian for epidemic control MFG.

    Args:
        px, py: Momentum components (∇u)
        infected: Infected density
        f_econ: Economic activity value
        lambda_mobility: Mobility cost parameter
        beta_infection: Infection risk coefficient

    Returns:
        Hamiltonian value
    """
    # Mobility control cost: ½λ|p|²
    mobility_cost = 0.5 * lambda_mobility * (px**2 + py**2)

    # Infection risk: β·I(x,y)
    infection_risk = beta_infection * infected

    # Economic activity reward (negative = benefit)
    economic_reward = -f_econ

    return mobility_cost + infection_risk + economic_reward


def solve_epidemic_mfg_simple(problem):
    """
    Solve 2D epidemic MFG using simplified coupled system.

    Couples:
    - HJB equation for individual optimal movement
    - FP equation for population density m
    - Disease dynamics for infected density I (simplified SIR)
    """
    grid = problem["grid"]
    T = problem["T"]
    Nt = problem["Nt"]
    sigma = problem["sigma"]
    alpha = problem["alpha"]
    gamma_recovery = problem["gamma_recovery"]

    dt = T / Nt
    Nx, Ny = grid.num_points

    # Initialize
    m0 = initial_population_distribution(grid)
    I0 = initial_infection_distribution(grid, problem["I0_center"], problem["I0_fraction"])
    u_terminal = terminal_cost(grid, problem["I0_center"])
    f_econ = economic_activity_map(grid, problem["L"], problem["kappa_economic"])

    # Build sparse Laplacian
    builder = SparseMatrixBuilder(grid, matrix_format="csr")
    L = builder.build_laplacian(boundary_conditions="neumann")

    # Initialize solution arrays
    u = np.zeros((Nt + 1, Nx, Ny))
    m = np.zeros((Nt + 1, Nx, Ny))
    infected = np.zeros((Nt + 1, Nx, Ny))  # Infected density over time

    # Boundary/initial conditions
    u[-1, :, :] = u_terminal
    m[0, :, :] = m0
    infected[0, :, :] = I0

    # Fixed-point iteration
    max_iter = 15
    tol = 1e-4

    print(f"Solving 2D Epidemic MFG on {Nx}×{Ny} grid...")
    print(f"Spatial domain: {problem['L']:.0f} km × {problem['L']:.0f} km")
    print(f"Time horizon: {T:.0f} days")
    print(f"Total unknowns: {Nx * Ny} per time step")
    print()

    for iteration in range(max_iter):
        u_old = u.copy()
        m_old = m.copy()
        infected_old = infected.copy()

        # Forward disease dynamics (simplified SIR)
        for n in range(Nt):
            # Susceptible density: S ≈ m - I (simplified)
            S_n = np.maximum(m[n, :, :] - infected[n, :, :], 0)

            # Disease evolution: dI/dt = α·S·I - γ·I + diffusion
            # Simplified forward Euler
            infected_flat = infected[n, :, :].flatten()
            new_infections = alpha * S_n * infected[n, :, :]
            recoveries = gamma_recovery * infected[n, :, :]
            diffusion = sigma * 0.1 * (L @ infected_flat).reshape(Nx, Ny)  # Small disease diffusion

            infected[n + 1, :, :] = np.maximum(infected[n, :, :] + dt * (new_infections - recoveries + diffusion), 0)

            # Clip to avoid numerical issues
            infected[n + 1, :, :] = np.minimum(infected[n + 1, :, :], m[n, :, :])

        # Backward HJB solve
        for n in range(Nt - 1, -1, -1):
            # Build gradient matrices
            Gx = builder.build_gradient(direction=0, order=2)
            Gy = builder.build_gradient(direction=1, order=2)

            # Compute gradients
            u_next = u[n + 1, :, :].flatten()
            px = (Gx @ u_next).reshape(Nx, Ny)
            py = (Gy @ u_next).reshape(Nx, Ny)

            # Hamiltonian with current infected density
            H = hamiltonian(px, py, infected[n, :, :], f_econ, problem["lambda_mobility"], problem["beta_infection"])

            # Implicit time step
            identity_matrix = np.eye(Nx * Ny)
            A = identity_matrix + dt * sigma**2 * (-L)
            b = u_next + dt * H.flatten()

            # Solve
            solver = SparseSolver(method="direct")
            u[n, :, :] = solver.solve(A, b).reshape(Nx, Ny)

        # Forward FP solve (simplified - diffusion only)
        for n in range(Nt):
            m_n = m[n, :, :].flatten()
            m[n + 1, :, :] = m[n, :, :] + dt * sigma**2 * (L @ m_n).reshape(Nx, Ny)

        # Check convergence
        u_change = np.max(np.abs(u - u_old))
        m_change = np.max(np.abs(m - m_old))
        infected_change = np.max(np.abs(infected - infected_old))

        print(f"Iteration {iteration + 1}: Δu = {u_change:.6f}, Δm = {m_change:.6f}, Δinfected = {infected_change:.6f}")

        if u_change < tol and m_change < tol and infected_change < tol:
            print(f"Converged in {iteration + 1} iterations")
            break

    return {"u": u, "m": m, "infected": infected, "grid": grid}


def visualize_results(solution, problem, output_dir="epidemic_modeling_2d"):
    """
    Create comprehensive visualizations of epidemic MFG solution.

    Solution-field plotting is done with matplotlib directly: the
    ``mfgarchon.visualization`` module provides convergence/diagnostics
    plotting and rendering-free data extraction, and points to
    matplotlib/plotly/pyvista for solution-field figures.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    u = solution["u"]
    m = solution["m"]
    infected = solution["infected"]
    grid = solution["grid"]

    # Physical coordinates (indexing="ij": X varies along axis 0, Y along axis 1)
    X, Y = grid.meshgrid(indexing="ij")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\nCreating visualizations in {output_dir}/...")

    # 1. Initial infection map (heatmap; red colormap for infections)
    print("  - Initial infection heatmap...")
    fig, ax = plt.subplots(figsize=(7, 6))
    pcm = ax.pcolormesh(X, Y, infected[0, :, :], cmap="Reds", shading="gouraud")
    ax.set_title("Initial Infection Distribution I0(x,y)")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    fig.colorbar(pcm, ax=ax)
    fig.savefig(output_path / "infection_initial.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Final infection map
    print("  - Final infection heatmap...")
    fig, ax = plt.subplots(figsize=(7, 6))
    pcm = ax.pcolormesh(X, Y, infected[-1, :, :], cmap="Reds", shading="gouraud")
    ax.set_title("Final Infection Distribution I(T,x,y)")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    fig.colorbar(pcm, ax=ax)
    fig.savefig(output_path / "infection_final.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Peak infection map (time of maximum total infections)
    total_infections = [np.sum(infected[n, :, :]) for n in range(len(infected))]
    peak_time = int(np.argmax(total_infections))
    peak_day = peak_time * problem["T"] / problem["Nt"]
    print(f"  - Peak infection map (day {peak_day:.1f})...")
    fig, ax = plt.subplots(figsize=(7, 6))
    pcm = ax.pcolormesh(X, Y, infected[peak_time, :, :], cmap="Reds", shading="gouraud")
    ax.set_title(f"Peak Infection (Day {peak_day:.1f})")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    fig.colorbar(pcm, ax=ax)
    fig.savefig(output_path / "infection_peak.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Infection spread animation
    print("  - Infection spread animation...")
    fig, ax = plt.subplots(figsize=(7, 6))
    pcm = ax.pcolormesh(X, Y, infected[0, :, :], cmap="Reds", shading="gouraud", vmin=0.0, vmax=float(infected.max()))
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Infected Density")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")

    def _update_infection(frame):
        pcm.set_array(infected[frame, :, :].ravel())
        ax.set_title(f"Infection Spread Evolution I(t,x,y) - step {frame}")
        return (pcm,)

    anim_I = FuncAnimation(fig, _update_infection, frames=infected.shape[0], blit=False)
    anim_I.save(output_path / "infection_evolution.gif", writer=PillowWriter(fps=5))
    plt.close(fig)

    # 5. Value function (cost-to-go with epidemic) as 3D surface
    print("  - Value function surface...")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, u[-1, :, :], cmap="viridis")
    ax.set_title("Value Function u(x,y) - Epidemic Risk Cost")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("Cost-to-go")
    fig.savefig(output_path / "value_function.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 6. Population distribution evolution animation
    print("  - Population movement animation...")
    fig, ax = plt.subplots(figsize=(7, 6))
    pcm = ax.pcolormesh(X, Y, m[0, :, :], cmap="Blues", shading="gouraud", vmin=0.0, vmax=float(m.max()))
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Population Density")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")

    def _update_population(frame):
        pcm.set_array(m[frame, :, :].ravel())
        ax.set_title(f"Population Distribution Evolution m(t,x,y) - step {frame}")
        return (pcm,)

    anim_m = FuncAnimation(fig, _update_population, frames=m.shape[0], blit=False)
    anim_m.save(output_path / "population_evolution.gif", writer=PillowWriter(fps=5))
    plt.close(fig)

    print(f"\nVisualizations saved to {output_dir}/")
    print("\nGenerated files:")
    print("  - infection_initial.png: Initial outbreak location")
    print("  - infection_final.png: Final infection distribution")
    print("  - infection_peak.png: Peak infection snapshot")
    print("  - infection_evolution.gif: Disease spread animation")
    print("  - value_function.png: Optimal movement cost with epidemic")
    print("  - population_evolution.gif: Population movement response")


def main():
    """
    Main execution: Solve 2D epidemic MFG and visualize results.
    """
    print("=" * 70)
    print("2D Epidemic Modeling via Mean Field Games")
    print("=" * 70)
    print("\nProblem Setup:")
    print("  - Domain: 20 km × 20 km spatial region")
    print("  - Time horizon: 30 days")
    print("  - Objective: Balance infection risk vs mobility benefits")
    print("  - Disease model: Simplified SIR dynamics")
    print()

    # Create problem
    problem = create_epidemic_problem()

    print("Grid Information:")
    print(f"  - Spatial: {problem['grid'].num_points[0]}×{problem['grid'].num_points[1]} points")
    print(f"  - Temporal: {problem['Nt']} time steps")
    print(f"  - Total DOF: {problem['grid'].total_points() * problem['Nt']}")
    print()

    print("Epidemic Parameters:")
    print(f"  - Infection rate (α): {problem['alpha']:.2f}")
    print(
        f"  - Recovery rate (γ): {problem['gamma_recovery']:.2f} (avg. recovery: {1 / problem['gamma_recovery']:.1f} days)"
    )
    print(f"  - Basic reproduction number (R₀): {problem['alpha'] / problem['gamma_recovery']:.2f}")
    print(f"  - Initial infected fraction: {problem['I0_fraction']:.1%}")
    print()

    # Solve MFG
    solution = solve_epidemic_mfg_simple(problem)

    # Visualize
    visualize_results(solution, problem)

    # Analysis
    print("\n" + "=" * 70)
    print("Epidemic Analysis:")
    print("=" * 70)

    infected_initial = solution["infected"][0, :, :]
    infected_final = solution["infected"][-1, :, :]

    # Total infected over time
    total_infected = [
        np.sum(solution["infected"][n, :, :]) * problem["grid"].volume_element() for n in range(problem["Nt"] + 1)
    ]
    peak_infections = max(total_infected)
    peak_day = total_infected.index(peak_infections) * problem["T"] / problem["Nt"]

    print("\nInfection Statistics:")
    print(f"  - Initial total infected: {total_infected[0]:.1%}")
    print(f"  - Peak total infected: {peak_infections:.1%} (day {peak_day:.1f})")
    print(f"  - Final total infected: {total_infected[-1]:.1%}")

    print("\nSpatial Distribution:")
    print(f"  - Initial max local density: {infected_initial.max():.4f}")
    print(f"  - Final max local density: {infected_final.max():.4f}")

    # Find location of peak infection
    infected_peak_idx = np.unravel_index(infected_final.argmax(), infected_final.shape)
    coords = solution["grid"].coordinates
    print("\nPeak Infection Location:")
    print(f"  - Position: ({coords[0][infected_peak_idx[0]]:.1f}, {coords[1][infected_peak_idx[1]]:.1f}) km")

    print("\n" + "=" * 70)
    print("Demo Complete")
    print("=" * 70)


if __name__ == "__main__":
    # Set matplotlib backend for non-interactive execution
    import matplotlib

    matplotlib.use("Agg")

    main()

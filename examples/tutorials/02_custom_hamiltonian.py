"""
Tutorial 02: Custom Hamiltonian

Learn how to define your own MFG problem with a custom Hamiltonian.

What you'll learn:
- How to create a custom Hamiltonian class
- How to define H(x, p, m) and its derivatives
- How to specify terminal costs and initial density
- How to use the Model/Conditions API

Mathematical Problem:
    A crowd evacuation problem with congestion:
    - Hamiltonian: H(x, p, m) = |p|^2 / (2*(1 + lambda*m)) (congestion slows movement)
    - Terminal cost: Distance to exit at x=1
    - Initial density: Uniform distribution (crowd at rest)
"""

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import HamiltonianBase
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

if __name__ == "__main__":
    # ==============================================================================
    # Step 1: Define Custom Hamiltonian
    # ==============================================================================

    print("=" * 70)
    print("TUTORIAL 02: Custom Hamiltonian")
    print("=" * 70)
    print()

    class CongestionHamiltonian(HamiltonianBase):
        """
        Custom Hamiltonian for crowd evacuation with congestion.

        H(x, m, p) = |p|^2 / (2 * (1 + lambda*m))

        The density sits in the DENOMINATOR, so a denser crowd raises the effective
        cost of moving: the optimal speed |alpha*| = |p| / (1 + lambda*m) falls as m
        grows. Putting m in the numerator instead would make crowds speed agents up.

        This is the Legendre transform of L(x, m, alpha) = (1/2)(1 + lambda*m)|alpha|^2.
        """

        def __init__(self, congestion_strength: float = 2.0):
            """
            Args:
                congestion_strength: lambda parameter controlling congestion effect
            """
            super().__init__()
            self.congestion_strength = congestion_strength

        def __call__(self, x, m, p, t=0.0):
            """
            Evaluate H(x, m, p, t).

            Sum over p's momentum-component axis so H is a scalar per grid point
            (the solver evaluates H pointwise). |p|^2 = sum_i p_i^2.

            Args:
                x: Spatial position(s)
                m: Density
                p: Momentum (gradient of value function)
                t: Time (unused in this example)

            Returns:
                Hamiltonian value
            """
            p_sq = np.sum(np.square(p), axis=-1)
            return p_sq / (2.0 * (1.0 + self.congestion_strength * m))

        def dp(self, x, m, p, t=0.0):
            """
            Derivative of H w.r.t. momentum: dH/dp = p / (1 + lambda*m).

            Used to compute optimal control: alpha* = -dH/dp. Its magnitude
            decreases with m, which is what makes a crowd slow agents down.
            """
            # Broadcast the scalar-per-point density m onto p's component axis.
            m_b = np.reshape(m, np.shape(p)) if np.size(m) == np.size(p) else m
            return p / (1.0 + self.congestion_strength * m_b)

        def dm(self, x, m, p, t=0.0):
            """
            Derivative of H w.r.t. density: dH/dm = -lambda*|p|^2 / (2*(1 + lambda*m)^2).

            Appears in the Fokker-Planck equation drift term.
            """
            p_sq = np.sum(np.square(p), axis=-1)
            return -self.congestion_strength * p_sq / (2.0 * (1.0 + self.congestion_strength * m) ** 2)

    # ==============================================================================
    # Step 2: Define Problem Components
    # ==============================================================================

    print("Setting up problem components...")
    print()

    # Create grid
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[61],
        boundary_conditions=no_flux_bc(dimension=1),
    )

    # Physical parameters
    sigma = 0.1  # SDE volatility
    congestion_strength = 2.0

    # Terminal cost: distance to exit at x=1
    def terminal_cost(x):
        """Agents want to be close to exit (x=1) at final time."""
        return (x - 1.0) ** 2

    # Initial density: uniform on [0.1, 0.9]
    def initial_density(x):
        """Uniform crowd distribution, avoiding boundaries."""
        density = np.where((x >= 0.1) & (x <= 0.9), 1.0, 0.01)
        return density

    # Create custom Hamiltonian
    hamiltonian = CongestionHamiltonian(congestion_strength=congestion_strength)

    # Bundle model and conditions
    model = Model(hamiltonian=hamiltonian, sigma=sigma)
    conditions = Conditions(u_terminal=terminal_cost, m_initial=initial_density, T=1.0)

    # ==============================================================================
    # Step 3: Create and Solve Problem
    # ==============================================================================

    print("Creating problem with custom Hamiltonian...")
    problem = MFGProblem(
        model=model,
        domain=grid,
        conditions=conditions,
        Nt=50,
    )

    print(f"  Congestion strength: lambda = {congestion_strength}")
    print()

    print("Solving MFG system...")
    result = problem.solve(verbose=True)

    print()
    print(f"Converged: {result.converged} (iterations: {result.iterations})")
    print()

    # ==============================================================================
    # Step 4: Analyze Congestion Effect
    # ==============================================================================

    print("=" * 70)
    print("CONGESTION ANALYSIS")
    print("=" * 70)
    print()

    # Compare with no-congestion case
    print("Solving baseline (no congestion)...")

    baseline_hamiltonian = CongestionHamiltonian(congestion_strength=0.0)
    baseline = problem.with_model(Model(hamiltonian=baseline_hamiltonian, sigma=sigma))
    result_baseline = baseline.solve(verbose=False)

    # Measure evacuation efficiency
    x = grid.coordinates[0]
    dx = grid.get_grid_spacing()[0]
    exit_mask = x > 0.9
    exit_mass_congested = np.sum(result.M[-1, exit_mask]) * dx
    exit_mass_baseline = np.sum(result_baseline.M[-1, exit_mask]) * dx

    print(f"Mass at exit (with congestion):    {exit_mass_congested:.3f}")
    print(f"Mass at exit (without congestion): {exit_mass_baseline:.3f}")
    if exit_mass_baseline > 0:
        slowdown = (1 - exit_mass_congested / exit_mass_baseline) * 100
        print(f"Evacuation slowdown: {slowdown:.1f}%")
    print()

    # ==============================================================================
    # Step 5: Visualize
    # ==============================================================================

    print("=" * 70)
    print("VISUALIZATION")
    print("=" * 70)
    print()

    try:
        from pathlib import Path

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        t = np.linspace(0, problem.T, problem.Nt + 1)
        X, T_grid = np.meshgrid(x, t)

        # Density evolution with congestion
        ax = axes[0, 0]
        c = ax.contourf(X, T_grid, result.M, levels=20, cmap="viridis")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(f"Density with congestion (lambda={congestion_strength})")
        plt.colorbar(c, ax=ax)

        # Density evolution without congestion
        ax = axes[0, 1]
        c = ax.contourf(X, T_grid, result_baseline.M, levels=20, cmap="viridis")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title("Density without congestion (lambda=0)")
        plt.colorbar(c, ax=ax)

        # Final density comparison
        ax = axes[1, 0]
        ax.plot(x, result.M[-1, :], "b-", linewidth=2, label=f"lambda={congestion_strength}")
        ax.plot(x, result_baseline.M[-1, :], "r--", linewidth=2, label="lambda=0")
        ax.axvline(x=0.9, color="g", linestyle=":", label="Exit zone")
        ax.set_xlabel("x")
        ax.set_ylabel("m(T, x)")
        ax.set_title("Final Density Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Value function comparison
        ax = axes[1, 1]
        # u[0] -- the value at t=0. u[-1] is the terminal condition (x-1)^2, which is
        # model-independent, so plotting it draws the two curves exactly on top of
        # each other and shows nothing about congestion.
        ax.plot(x, result.U[0, :], "b-", linewidth=2, label=f"lambda={congestion_strength}")
        ax.plot(x, result_baseline.U[0, :], "r--", linewidth=2, label="lambda=0")
        ax.set_xlabel("x")
        ax.set_ylabel("u(0, x)")
        ax.set_title("Value Function at t=0")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_dir = Path(__file__).parent.parent / "outputs" / "tutorials"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "02_custom_hamiltonian.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")
        print()

    except ImportError:
        print("Matplotlib not available - skipping visualization")
        print()

    # ==============================================================================
    # Summary
    # ==============================================================================

    print("=" * 70)
    print("TUTORIAL COMPLETE")
    print("=" * 70)
    print()
    print("What you learned:")
    print("  1. Create a custom Hamiltonian by subclassing HamiltonianBase")
    print("  2. Implement __call__(x, m, p, t) for H(x, m, p)")
    print("  3. Implement dp() and dm() for derivatives")
    print("  4. Bundle with Model/Conditions and solve")
    print()
    print("Key insight:")
    print("  Congestion (|p|^2 / (2*(1 + lambda*m))) slows evacuation by making")
    print("  movement costlier in crowded areas, causing agents to spread out.")
    print()
    print("Next: Tutorial 03 - 2D Geometry")
    print("=" * 70)

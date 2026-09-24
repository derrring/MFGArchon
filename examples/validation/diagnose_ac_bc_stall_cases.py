"""
Diagnostic: Adjoint-Consistent BC for Boundary vs Middle Stall Cases.

Tests whether adjoint-consistent BC (AdjointConsistentProvider) helps or hurts
for different stall point locations:
1. Boundary stall (x_stall = 0.0) - where AC BC should theoretically help
2. Middle stall (x_stall = 0.5) - where AC BC is known to hurt

Key diagnostics:
- Show BC values being computed at each iteration
- Compare final error for both BC types
- Validate mathematical consistency of provider computation

References:
- Issue #574: Adjoint-consistent BC implementation
- Issue #625: BCValueProvider architecture
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import (
    AdjointConsistentProvider,
    BCSegment,
    BCType,
    BoundaryConditions,
    neumann_bc,
)
from mfgarchon.geometry.boundary.bc_coupling import compute_boundary_log_density_gradient_1d

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analytic_boltzmann_gibbs(
    x: np.ndarray,
    x_stall: float,
    sigma: float,
    lambda_crowd: float,
) -> np.ndarray:
    """Analytic Boltzmann-Gibbs stationary distribution."""
    T_eff = sigma**2 / 2 + lambda_crowd
    V = np.abs(x - x_stall)
    unnormalized = np.exp(-V / T_eff)
    Z = np.trapezoid(unnormalized, x)
    return unnormalized / Z


def create_towel_problem(
    L: float,
    x_stall: float,
    sigma: float,
    lambda_crowd: float,
    Nx: int,
    T: float,
    Nt: int,
    bc_type: str,  # "neumann" or "adjoint_consistent"
) -> MFGProblem:
    """Create towel-on-beach problem with specified BC type."""

    # Class-based Hamiltonian: H = (1/2)|p|^2 + |x - x_stall| + lambda_crowd * ln(m)
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        potential=lambda t, x: abs(x[0] - x_stall),
        coupling=lambda m: lambda_crowd * np.log(max(m, 1e-10)),
        coupling_dm=lambda m: lambda_crowd / max(m, 1e-10),
    )
    model = Model(hamiltonian=hamiltonian, sigma=sigma)
    conditions = Conditions(
        m_initial=lambda x: np.ones_like(x) / L,
        u_terminal=lambda x: np.zeros_like(x),
        T=T,
    )

    # Determine boundary conditions
    if bc_type == "neumann":
        bc = neumann_bc(dimension=1)
    elif bc_type == "adjoint_consistent":
        bc = BoundaryConditions(
            segments=[
                BCSegment(
                    name="left_ac",
                    bc_type=BCType.ROBIN,
                    alpha=0.0,
                    beta=1.0,
                    value=AdjointConsistentProvider(side="left", diffusion=sigma),
                    boundary="x_min",
                ),
                BCSegment(
                    name="right_ac",
                    bc_type=BCType.ROBIN,
                    alpha=0.0,
                    beta=1.0,
                    value=AdjointConsistentProvider(side="right", diffusion=sigma),
                    boundary="x_max",
                ),
            ],
            dimension=1,
        )
    else:
        raise ValueError(f"Unknown bc_type: {bc_type}")

    # Create geometry with BC
    geometry = TensorProductGrid(
        bounds=[(0.0, L)],
        Nx_points=[Nx],
        boundary_conditions=bc,
    )

    return MFGProblem(
        model=model,
        domain=geometry,
        conditions=conditions,
        Nt=Nt,
    )


def diagnose_bc_values(x_stall: float, sigma: float, x_grid: np.ndarray, m_analytic: np.ndarray):
    """Diagnose what BC values the provider would compute for analytic solution."""
    dx = x_grid[1] - x_grid[0]

    grad_ln_m_left = compute_boundary_log_density_gradient_1d(m_analytic, dx, "left")
    grad_ln_m_right = compute_boundary_log_density_gradient_1d(m_analytic, dx, "right")

    # AC BC values: g = -sigma^2/2 * grad_ln_m
    diffusion_coeff = sigma**2 / 2
    g_left = -diffusion_coeff * grad_ln_m_left
    g_right = -diffusion_coeff * grad_ln_m_right

    print(f"\n  Diagnostic for x_stall={x_stall}:")
    print(f"    m[0] = {m_analytic[0]:.6f}, m[1] = {m_analytic[1]:.6f}")
    print(f"    m[-2] = {m_analytic[-2]:.6f}, m[-1] = {m_analytic[-1]:.6f}")
    print(f"    grad_ln_m at left = {grad_ln_m_left:.4f}")
    print(f"    grad_ln_m at right = {grad_ln_m_right:.4f}")
    print(f"    AC BC value at left (g_left) = {g_left:.6f}")
    print(f"    AC BC value at right (g_right) = {g_right:.6f}")
    print("    For comparison: standard Neumann has g = 0")

    return g_left, g_right


def run_comparison(x_stall: float, label: str):
    """Run comparison for given stall location."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {label}")
    print(f"  x_stall = {x_stall}")
    print(f"{'=' * 70}")

    # Parameters
    L = 1.0
    sigma = 0.2
    lambda_crowd = 0.5
    Nx = 51
    T = 3.0
    Nt = 61
    max_iter = 50
    tol = 1e-4

    x_grid = np.linspace(0, L, Nx)
    m_analytic = analytic_boltzmann_gibbs(x_grid, x_stall, sigma, lambda_crowd)

    # Diagnose what AC BC should compute for analytic solution
    diagnose_bc_values(x_stall, sigma, x_grid, m_analytic)

    results = {}

    for bc_type, bc_label in [("neumann", "Standard Neumann"), ("adjoint_consistent", "Adjoint-Consistent")]:
        print(f"\n  Running {bc_label}...")
        problem = create_towel_problem(
            L=L, x_stall=x_stall, sigma=sigma, lambda_crowd=lambda_crowd, Nx=Nx, T=T, Nt=Nt, bc_type=bc_type
        )

        try:
            result = problem.solve(max_iterations=max_iter, tolerance=tol, verbose=False)
            m_final = result.M[-1, :]
            if len(m_final) != Nx:
                m_final = m_final[:Nx] if len(m_final) > Nx else m_final
            error = np.max(np.abs(m_final - m_analytic))

            results[bc_type] = {
                "converged": result.converged,
                "iterations": result.iterations,
                "error": error,
                "m_final": m_final,
            }
            print(f"    Converged: {result.converged}, iterations: {result.iterations}, error: {error:.6f}")
        except Exception as e:
            print(f"    Failed: {e}")
            results[bc_type] = {"error": float("inf"), "converged": False}

    # Compare
    if (
        results.get("neumann", {}).get("error", float("inf")) > 0
        and results.get("adjoint_consistent", {}).get("error", float("inf")) > 0
    ):
        ratio = results["neumann"]["error"] / results["adjoint_consistent"]["error"]
        if ratio > 1:
            print(f"\n  Result: Adjoint-Consistent is {ratio:.2f}x BETTER")
        else:
            print(f"\n  Result: Adjoint-Consistent is {1 / ratio:.2f}x WORSE")

    return results, x_grid, m_analytic


def main():
    print("Adjoint-Consistent BC: Boundary vs Middle Stall Comparison")
    print("=" * 70)

    # Test case 1: Boundary stall (x_stall = 0)
    results_boundary, x_grid, m_analytic_boundary = run_comparison(0.0, "Boundary Stall (x_stall = 0.0)")

    # Test case 2: Middle stall (x_stall = 0.5)
    results_middle, _, m_analytic_middle = run_comparison(0.5, "Middle Stall (x_stall = 0.5)")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Row 1: Boundary stall
    ax = axes[0, 0]
    ax.plot(x_grid, m_analytic_boundary, "k--", lw=2, label="Analytic")
    if "m_final" in results_boundary.get("neumann", {}):
        ax.plot(
            x_grid,
            results_boundary["neumann"]["m_final"],
            "b-",
            alpha=0.7,
            label=f"Neumann (err={results_boundary['neumann']['error']:.4f})",
        )
    if "m_final" in results_boundary.get("adjoint_consistent", {}):
        ax.plot(
            x_grid,
            results_boundary["adjoint_consistent"]["m_final"],
            "r-",
            alpha=0.7,
            label=f"AC (err={results_boundary['adjoint_consistent']['error']:.4f})",
        )
    ax.axvline(0.0, color="orange", ls=":", label="Stall")
    ax.set_title("Boundary Stall (x=0): Density")
    ax.set_xlabel("x")
    ax.set_ylabel("m(x)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    if "m_final" in results_boundary.get("neumann", {}):
        ax.plot(x_grid, np.abs(results_boundary["neumann"]["m_final"] - m_analytic_boundary), "b-", label="Neumann")
    if "m_final" in results_boundary.get("adjoint_consistent", {}):
        ax.plot(
            x_grid, np.abs(results_boundary["adjoint_consistent"]["m_final"] - m_analytic_boundary), "r-", label="AC"
        )
    ax.set_title("Boundary Stall: Error Profile")
    ax.set_xlabel("x")
    ax.set_ylabel("|error|")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    # Row 2: Middle stall
    ax = axes[1, 0]
    ax.plot(x_grid, m_analytic_middle, "k--", lw=2, label="Analytic")
    if "m_final" in results_middle.get("neumann", {}):
        ax.plot(
            x_grid,
            results_middle["neumann"]["m_final"],
            "b-",
            alpha=0.7,
            label=f"Neumann (err={results_middle['neumann']['error']:.4f})",
        )
    if "m_final" in results_middle.get("adjoint_consistent", {}):
        ax.plot(
            x_grid,
            results_middle["adjoint_consistent"]["m_final"],
            "r-",
            alpha=0.7,
            label=f"AC (err={results_middle['adjoint_consistent']['error']:.4f})",
        )
    ax.axvline(0.5, color="orange", ls=":", label="Stall")
    ax.set_title("Middle Stall (x=0.5): Density")
    ax.set_xlabel("x")
    ax.set_ylabel("m(x)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if "m_final" in results_middle.get("neumann", {}):
        ax.plot(x_grid, np.abs(results_middle["neumann"]["m_final"] - m_analytic_middle), "b-", label="Neumann")
    if "m_final" in results_middle.get("adjoint_consistent", {}):
        ax.plot(x_grid, np.abs(results_middle["adjoint_consistent"]["m_final"] - m_analytic_middle), "r-", label="AC")
    ax.set_title("Middle Stall: Error Profile")
    ax.set_xlabel("x")
    ax.set_ylabel("|error|")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "ac_bc_stall_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {output_path}")
    plt.show()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nBoundary Stall (x=0):")
    print(f"  Neumann error: {results_boundary.get('neumann', {}).get('error', 'N/A')}")
    print(f"  AC error: {results_boundary.get('adjoint_consistent', {}).get('error', 'N/A')}")
    print("\nMiddle Stall (x=0.5):")
    print(f"  Neumann error: {results_middle.get('neumann', {}).get('error', 'N/A')}")
    print(f"  AC error: {results_middle.get('adjoint_consistent', {}).get('error', 'N/A')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MFG with Diagonal Anisotropic Tensor Diffusion.

Demonstrates full MFG coupling with diagonal tensor diffusion, where agents have
different diffusion rates in different spatial directions.

Example: Crowd moving through a corridor where horizontal movement is easier than
vertical movement (e.g., due to physical constraints or social norms).

Mathematical Setup:
-------------------
HJB equation:
    -du/dt + H(x, m, grad(u)) - (1/2) tr(Sigma * D^2 u) = 0
    H = (alpha/2)|grad(u)|^2 + F(m)   (convective Hamiltonian via H_class)

FP equation:
    dm/dt = sum_i d/dx_i(sigma_i^2 dm/dx_i) - div(alpha m grad(u))

where Sigma = diag(sigma_1^2, sigma_2^2) is the diagonal tensor diffusion.
Diffusion enters as an anisotropic Laplacian (1/2) sum_i sigma_i^2 d^2u/dx_i^2.
"""

import matplotlib.pyplot as plt
import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary.conditions import no_flux_bc

print("=" * 70)
print("MFG with Diagonal Anisotropic Tensor Diffusion")
print("=" * 70)
print()

# ============================================================================
# Problem Setup
# ============================================================================

print("Setting up problem parameters...")
print()

# Create 2D domain: corridor [0, 1] x [0, 0.6]
domain = TensorProductGrid(
    bounds=[(0.0, 1.0), (0.0, 0.6)],
    Nx_points=[31, 21],  # Nx x Ny grid points
    boundary_conditions=no_flux_bc(dimension=2),
)

# Create LQ Hamiltonian with coupling
hamiltonian = SeparableHamiltonian(
    control_cost=QuadraticControlCost(control_cost=0.5),
    coupling=lambda m: 0.5 * m,
    coupling_dm=lambda m: 0.5,
)

# Get grid coordinates for initial/terminal conditions
x_coords = domain.coordinates[0]
y_coords = domain.coordinates[1]
X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
dx, dy = domain.get_grid_spacing()

# Initial Gaussian centered at (0.3, 0.3)
x0, y0 = 0.3, 0.3
m0 = np.exp(-50 * ((X - x0) ** 2 + (Y - y0) ** 2))
m0 /= np.sum(m0) * dx * dy  # Normalize


# Terminal cost: quadratic distance to center
def terminal_cost_2d(xy):
    """Terminal cost for 2D problem."""
    if xy.ndim == 1:
        return (xy[0] - 0.5) ** 2 + (xy[1] - 0.3) ** 2
    return (xy[:, 0] - 0.5) ** 2 + (xy[:, 1] - 0.3) ** 2


# v1.0 API: Model holds game rules, Conditions holds IC/TC + time horizon
model = Model(hamiltonian=hamiltonian, volatility=0.1)
conditions = Conditions(
    u_terminal=terminal_cost_2d,
    m_initial=lambda xy: np.exp(-50 * ((xy[:, 0] - x0) ** 2 + (xy[:, 1] - y0) ** 2)),
    T=0.1,  # Short time horizon
)

# Nt=20 for CFL stability: the anisotropic Laplacian term requires
# dt * (sigma_x^2/dx^2 + sigma_y^2/dy^2) < O(1) for the explicit scheme.
problem = MFGProblem(
    model=model,
    domain=domain,
    conditions=conditions,
    Nt=20,  # 20 timesteps (dt=0.005 for CFL stability)
)

print("Domain: [0.0, 1.0] x [0.0, 0.6]")
print("Grid: 30 x 20 points")
print(f"Time: T = {problem.T}, Nt = {problem.Nt}, dt = {problem.dt:.4f}")
print()

# ============================================================================
# Diagonal Tensor Diffusion
# ============================================================================

print("Configuring anisotropic diagonal tensor diffusion...")
print()

# Diagonal tensor: fast horizontal, slow vertical
# Sigma = [[sigma_x^2, 0        ],
#          [0,          sigma_y^2]]
sigma_x = 0.15  # Fast horizontal diffusion
sigma_y = 0.05  # Slow vertical diffusion

Sigma = np.diag([sigma_x**2, sigma_y**2])

print("Tensor diffusion matrix:")
print(f"  Sigma = [[{Sigma[0, 0]:.4f}, {Sigma[0, 1]:.4f}],")
print(f"           [{Sigma[1, 0]:.4f}, {Sigma[1, 1]:.4f}]]")
print()
print("Physical interpretation:")
print(f"  - Horizontal diffusion: sigma_x = {sigma_x:.3f} (agents spread quickly left-right)")
print(f"  - Vertical diffusion:   sigma_y = {sigma_y:.3f} (agents spread slowly up-down)")
print(f"  - Ratio: sigma_x/sigma_y = {sigma_x / sigma_y:.1f}x")
print()

# ============================================================================
# Initial Density (already configured in Conditions above)
# ============================================================================

print("Initial density: Gaussian centered at (0.3, 0.3)")
dx, dy = domain.get_grid_spacing()
print(f"  Mass: {np.sum(m0) * dx * dy:.6f}")
print()

# ============================================================================
# Create Solvers
# ============================================================================

print("Creating HJB and FP solvers...")
print()

boundary_conditions = no_flux_bc(dimension=2)

# Fixed-point solver is more robust than Newton for coupled Picard iteration
# with Laplacian-based diffusion (Issue #784).
hjb_solver = HJBFDMSolver(
    problem,
    solver_type="fixed_point",
    damping_factor=0.8,
    max_newton_iterations=100,
)

fp_solver = FPFDMSolver(problem, boundary_conditions=boundary_conditions)

print("[OK] Solvers created")
print()

# ============================================================================
# Picard Iteration with Damping
# ============================================================================

print("Running Picard iteration with damping...")
print()

# Get grid shape
Nx, Ny = domain.get_grid_shape()
Nt = problem.Nt + 1

# Initialize: M uniform, U zero
M = np.ones((Nt, Nx, Ny)) / (Nx * Ny)
U = np.zeros((Nt, Nx, Ny))

# Set initial density
M[0] = m0.copy()

# Normalize all timesteps
for n in range(Nt):
    M[n] /= np.sum(M[n]) * dx * dy

# Run Picard iterations with damping for stability
max_iterations = 3
picard_damping = 0.3

for iteration in range(max_iterations):
    print(f"Picard Iteration {iteration + 1}/{max_iterations}")

    # Solve HJB with diagonal tensor
    U_new = hjb_solver.solve_hjb_system(
        M_density=M,
        U_terminal=U[-1],
        U_coupling_prev=U,
        tensor_diffusion_field=Sigma,
    )

    # Solve FP with diagonal tensor
    M_new = fp_solver.solve_fp_system(
        M_initial=m0,
        potential_field=U_new,
        tensor_diffusion_field=Sigma,
        show_progress=False,
    )

    # Picard damping for stability
    U_damped = picard_damping * U_new + (1 - picard_damping) * U
    M_damped = picard_damping * M_new + (1 - picard_damping) * M

    # Check convergence
    U_change = np.linalg.norm(U_damped - U) / (np.linalg.norm(U_damped) + 1e-10)
    M_change = np.linalg.norm(M_damped - M) / (np.linalg.norm(M_damped) + 1e-10)

    print(f"  dU / ||U||: {U_change:.2e}")
    print(f"  dM / ||M||: {M_change:.2e}")

    # Check mass conservation
    masses = np.sum(M_new, axis=(1, 2)) * dx * dy
    mass_error = np.abs(masses - 1.0).max()
    print(f"  Mass conservation error: {mass_error:.2e}")
    print()

    # Update with damped values
    U = U_damped.copy()
    M = M_damped.copy()

print("[OK] Picard iterations completed")
print()

# ============================================================================
# Visualization
# ============================================================================

print("Generating visualization...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(
    f"MFG with Diagonal Tensor Diffusion (sigma_x={sigma_x}, sigma_y={sigma_y})",
    fontsize=14,
    fontweight="bold",
)

# Time indices to plot
time_indices = [0, Nt // 2, Nt - 1]
time_labels = ["t = 0", f"t = {problem.T / 2:.2f}", f"t = {problem.T:.2f}"]

for i, (t_idx, t_label) in enumerate(zip(time_indices, time_labels, strict=True)):
    # Plot density M(t, x, y)
    ax = axes[0, i]
    im = ax.contourf(X, Y, M[t_idx], levels=20, cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Density m(x, y) at {t_label}")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)

    # Plot value function U(t, x, y)
    ax = axes[1, i]
    im = ax.contourf(X, Y, U[t_idx], levels=20, cmap="coolwarm")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Value u(x, y) at {t_label}")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)

plt.tight_layout()

# Save figure
output_path = "examples/outputs/diagonal_tensor_mfg.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"[OK] Figure saved to {output_path}")
print()

# Show plot
plt.show()

print("=" * 70)
print("MFG with Diagonal Tensor Diffusion - Complete")
print("=" * 70)
print()
print("Key Results:")
print(f"  - Anisotropic diffusion ratio: {sigma_x / sigma_y:.1f}x (horizontal vs vertical)")
print(f"  - Mass conservation error: < {mass_error:.2e}")
print(f"  - Picard convergence: {max_iterations} iterations")
print()
print("Observation:")
print("  - Density spreads primarily in horizontal direction (sigma_x > sigma_y)")
print("  - Agents prefer horizontal movement due to lower diffusion cost")
print("  - Value function reflects anisotropic cost structure")

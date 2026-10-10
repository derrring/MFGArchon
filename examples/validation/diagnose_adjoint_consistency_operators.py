#!/usr/bin/env python3
"""
Diagnostic: Adjoint Consistency Analysis using Operators Framework.

This script uses the NEW operators framework (not deprecated tensor_calculus)
to analyze boundary adjoint consistency in MFG solvers.

Key Questions:
1. Does A_HJB^T preserve no-flux BC?
2. Is mass conserved when using strict adjoint mode?
3. What is the boundary flux leakage magnitude?

Issue #625: Boundary adjoint consistency investigation
"""

import numpy as np
import scipy.sparse as sparse

from mfgarchon.geometry.boundary import neumann_bc
from mfgarchon.operators.differential.laplacian import LaplacianOperator


def build_hjb_advection_matrix_1d(
    U: np.ndarray,
    dx: float,
    coupling_coeff: float,
) -> sparse.csr_matrix:
    """
    Build HJB advection matrix using Godunov upwind (point-based velocity).

    This matches the implementation in hjb_fdm.py:_build_advection_matrix_1d
    but uses direct computation instead of deprecated tensor_calculus.

    Mathematical form:
        v_i = -coupling * dU/dx|_i
        If v_i > 0: backward difference (upwind from left)
        If v_i < 0: forward difference (upwind from right)

    Boundary handling:
        At i=0 with v_0 > 0: only diagonal term (no left neighbor)
        At i=Nx-1 with v_{Nx-1} < 0: only diagonal term (no right neighbor)
    """
    Nx = len(U)

    # Compute gradient using Godunov upwind at each point
    # Forward: (U[i+1] - U[i]) / dx
    # Backward: (U[i] - U[i-1]) / dx
    # Central (for sign): (forward + backward) / 2

    grad_U = np.zeros(Nx)

    for i in range(Nx):
        # Compute forward and backward differences (with boundary handling)
        if i < Nx - 1:
            grad_forward = (U[i + 1] - U[i]) / dx
        else:
            grad_forward = (U[i] - U[i - 1]) / dx  # One-sided at right

        if i > 0:
            grad_backward = (U[i] - U[i - 1]) / dx
        else:
            grad_backward = (U[i + 1] - U[i]) / dx  # One-sided at left

        # Godunov: select based on sign of central gradient
        grad_central = (grad_forward + grad_backward) / 2.0
        if grad_central >= 0:
            grad_U[i] = grad_backward
        else:
            grad_U[i] = grad_forward

    # Velocity: v = -coupling * grad_U
    velocity = -coupling_coeff * grad_U

    # Build sparse matrix
    row_indices = []
    col_indices = []
    data_values = []

    for i in range(Nx):
        v_i = velocity[i]

        if abs(v_i) < 1e-14:
            continue

        if v_i > 0:
            # Flow to right: backward difference
            row_indices.append(i)
            col_indices.append(i)
            data_values.append(v_i / dx)

            if i > 0:
                row_indices.append(i)
                col_indices.append(i - 1)
                data_values.append(-v_i / dx)
            # At i=0: no left neighbor, skip

        else:  # v_i < 0
            # Flow to left: forward difference
            row_indices.append(i)
            col_indices.append(i)
            data_values.append(-v_i / dx)

            if i < Nx - 1:
                row_indices.append(i)
                col_indices.append(i + 1)
                data_values.append(v_i / dx)
            # At i=Nx-1: no right neighbor, skip

    A_hjb = sparse.coo_matrix((data_values, (row_indices, col_indices)), shape=(Nx, Nx)).tocsr()
    return A_hjb


def analyze_mass_conservation(
    Nx: int = 51,
    x_stall: float = 0.3,
    sigma: float = 0.2,
    coupling_coeff: float = 0.5,
    dt: float = 0.01,
    n_steps: int = 100,
):
    """
    Test mass conservation in strict adjoint mode.

    Simulates FP evolution using A_HJB^T and measures mass drift.
    """
    print("=" * 70)
    print("Mass Conservation Test: Strict Adjoint Mode")
    print("=" * 70)

    dx = 1.0 / (Nx - 1)
    x = np.linspace(0, 1, Nx)

    # Value function with stall at x_stall
    U = (x - x_stall) ** 2

    print("\nSetup:")
    print(f"  Grid: Nx={Nx}, dx={dx:.4f}")
    print(f"  Stall point: x = {x_stall}")
    print(f"  Parameters: sigma={sigma}, coupling={coupling_coeff}")
    print(f"  Time stepping: dt={dt}, n_steps={n_steps}")
    print()

    # Build HJB advection matrix
    A_hjb = build_hjb_advection_matrix_1d(U, dx, coupling_coeff)
    A_hjb_T = A_hjb.T.tocsr()

    # Build Laplacian using operators framework (with Neumann BC)
    bc = neumann_bc(dimension=1)
    L_op = LaplacianOperator(spacings=[dx], field_shape=(Nx,), bc=bc)
    L_matrix = L_op.as_scipy_sparse()

    # Diffusion coefficient
    D = 0.5 * sigma**2

    # Build system matrix: (I/dt + A_advection_T - D*Laplacian)
    identity = sparse.eye(Nx)
    A_system = identity / dt + A_hjb_T - D * L_matrix

    print("Matrix Analysis:")
    print(f"  A_HJB nnz: {A_hjb.nnz}")
    print(f"  A_HJB^T nnz: {A_hjb_T.nnz}")
    print(f"  L_matrix nnz: {L_matrix.nnz}")

    # Check row sums (should be ~1/dt for mass conservation)
    row_sums = np.array(A_system.sum(axis=1)).ravel()
    print("\nA_system row sums:")
    print(f"  Row 0 (left boundary):  {row_sums[0]:.6f} (expected: {1 / dt:.6f})")
    print(f"  Row {Nx - 1} (right boundary): {row_sums[-1]:.6f} (expected: {1 / dt:.6f})")
    print(f"  Interior mean:          {np.mean(row_sums[1:-1]):.6f}")

    # Initial density: uniform
    M = np.ones(Nx) / Nx
    initial_mass = np.sum(M) * dx

    print(f"\nInitial mass: {initial_mass:.10f}")

    # Time evolution
    masses = [initial_mass]

    for step in range(n_steps):
        b_rhs = M / dt
        M_next = sparse.linalg.spsolve(A_system, b_rhs)
        M = np.maximum(M_next, 0.0)  # Ensure non-negativity
        masses.append(np.sum(M) * dx)

    final_mass = masses[-1]
    mass_drift = final_mass - initial_mass
    relative_drift = mass_drift / initial_mass * 100

    print("\nMass Evolution:")
    print(f"  Final mass: {final_mass:.10f}")
    print(f"  Mass drift: {mass_drift:+.2e}")
    print(f"  Relative drift: {relative_drift:+.4f}%")

    # Boundary flux analysis
    print("\n" + "=" * 70)
    print("Boundary Flux Analysis")
    print("=" * 70)

    # A_HJB^T boundary rows
    A_T_row0 = A_hjb_T.toarray()[0, :]
    A_T_rowN = A_hjb_T.toarray()[-1, :]

    print(f"\nA_HJB^T[0, :5]:   {A_T_row0[:5]}")
    print(f"A_HJB^T[-1, -5:]: {A_T_rowN[-5:]}")

    # Off-diagonal entries at boundaries = potential flux leakage
    flux_leak_left = A_T_row0[1] if Nx > 1 else 0
    flux_leak_right = A_T_rowN[-2] if Nx > 1 else 0

    print("\nPotential flux leakage:")
    print(f"  Left boundary (A^T[0,1]):   {flux_leak_left:.6f}")
    print(f"  Right boundary (A^T[-1,-2]): {flux_leak_right:.6f}")

    if abs(flux_leak_left) > 1e-10 or abs(flux_leak_right) > 1e-10:
        print("\n  WARNING: Non-zero off-diagonal at boundaries!")
        print("           This allows mass flux through boundaries.")

    return {
        "initial_mass": initial_mass,
        "final_mass": final_mass,
        "mass_drift": mass_drift,
        "relative_drift_pct": relative_drift,
        "masses": masses,
    }


def compare_with_boundary_fix(
    Nx: int = 51,
    x_stall: float = 0.3,
    sigma: float = 0.2,
    coupling_coeff: float = 0.5,
    dt: float = 0.01,
    n_steps: int = 100,
):
    """
    Compare mass conservation with and without boundary fix.
    """
    print("\n" + "=" * 70)
    print("Comparison: With vs Without Boundary Fix")
    print("=" * 70)

    dx = 1.0 / (Nx - 1)
    x = np.linspace(0, 1, Nx)
    U = (x - x_stall) ** 2

    # Build matrices
    A_hjb = build_hjb_advection_matrix_1d(U, dx, coupling_coeff)
    A_hjb_T = A_hjb.T.tocsr()

    # Apply boundary fix: zero out boundary off-diagonals
    A_hjb_T_fixed = A_hjb_T.tolil()
    A_hjb_T_fixed[0, 1:] = 0  # Zero left boundary off-diagonals
    A_hjb_T_fixed[-1, :-1] = 0  # Zero right boundary off-diagonals
    A_hjb_T_fixed = A_hjb_T_fixed.tocsr()

    # Laplacian
    bc = neumann_bc(dimension=1)
    L_op = LaplacianOperator(spacings=[dx], field_shape=(Nx,), bc=bc)
    L_matrix = L_op.as_scipy_sparse()
    D = 0.5 * sigma**2

    identity = sparse.eye(Nx)

    # Two system matrices
    A_system_original = identity / dt + A_hjb_T - D * L_matrix
    A_system_fixed = identity / dt + A_hjb_T_fixed - D * L_matrix

    # Initial density
    M_original = np.ones(Nx) / Nx
    M_fixed = M_original.copy()
    initial_mass = np.sum(M_original) * dx

    # Evolve both
    for step in range(n_steps):
        M_original = np.maximum(sparse.linalg.spsolve(A_system_original, M_original / dt), 0.0)
        M_fixed = np.maximum(sparse.linalg.spsolve(A_system_fixed, M_fixed / dt), 0.0)

    mass_original = np.sum(M_original) * dx
    mass_fixed = np.sum(M_fixed) * dx

    print(f"\nAfter {n_steps} steps:")
    print(
        f"  Original A^T:     mass = {mass_original:.8f}, drift = {(mass_original - initial_mass) / initial_mass * 100:+.4f}%"
    )
    print(
        f"  Fixed boundary:   mass = {mass_fixed:.8f}, drift = {(mass_fixed - initial_mass) / initial_mass * 100:+.4f}%"
    )

    return {
        "mass_original": mass_original,
        "mass_fixed": mass_fixed,
        "improvement": abs(mass_fixed - initial_mass) < abs(mass_original - initial_mass),
    }


def sensitivity_analysis():
    """Test different stall locations."""
    print("\n" + "=" * 70)
    print("Sensitivity Analysis: Stall Location vs Mass Drift")
    print("=" * 70)

    stall_locations = [0.0, 0.25, 0.5, 0.75, 1.0]

    print(f"\n{'Stall':>8} | {'Mass Drift (%)':>14} | {'Final Mass':>12}")
    print("-" * 45)

    for x_stall in stall_locations:
        result = analyze_mass_conservation(Nx=51, x_stall=x_stall, n_steps=100)
        print(f"{x_stall:>8.2f} | {result['relative_drift_pct']:>14.4f} | {result['final_mass']:>12.8f}")


if __name__ == "__main__":
    # Main analysis
    result = analyze_mass_conservation(Nx=51, x_stall=0.3, n_steps=100)

    # Comparison with fix
    compare_with_boundary_fix(Nx=51, x_stall=0.3, n_steps=100)

    # Sensitivity
    sensitivity_analysis()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The removed adjoint_mode="transpose" passed A_HJB^T to solve_fp_step_adjoint_mode
directly, without boundary post-processing (#2594 removed it; "jacobian_transpose"
passes the linearised operator's transpose). That caused:

1. Non-zero off-diagonal entries at boundary rows of A^T
2. Mass flux through boundaries (violates no-flux BC)
3. Cumulative mass drift over time

A FIX ONCE CONSIDERED (Issue #625) for the removed adjoint_mode="transpose" (#2594). It does not
apply to the remaining mode, which hands solve_fp_step_adjoint_mode the linearised HJB operator's
transpose. For the removed mode, post-process A_advection_T to enforce no-flux:

```python
# After receiving A_advection_T, fix boundary rows
A_advection_T_fixed = A_advection_T.tolil()
A_advection_T_fixed[0, 1:] = 0       # Left boundary
A_advection_T_fixed[-1, :-1] = 0    # Right boundary
A_advection_T = A_advection_T_fixed.tocsr()
```

This preserves interior adjoint consistency while enforcing no-flux BC.
""")

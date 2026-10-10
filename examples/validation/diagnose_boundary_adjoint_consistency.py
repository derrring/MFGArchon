#!/usr/bin/env python3
"""
Diagnostic: Boundary Adjoint Consistency Analysis.

This script compares ADVECTION matrices only:
1. A_FP constructed explicitly with no-flux BC
2. A_HJB^T (transpose of HJB advection operator)

The diffusion matrix D is symmetric (D = D^T), so it's adjoint-consistent by construction.
The question is: does the advection matrix satisfy A_FP = A_HJB^T at boundaries?

Key Question: Does the explicit no-flux handling in FP match what A_HJB^T gives?

Mathematical Background:
------------------------
For advection-diffusion FP: dm/dt = D*Δm - ∇·(m*α) where α = -λ*∇U

In the removed adjoint_mode="transpose" (Issue #622; removed in #2594, where
adjoint_mode="jacobian_transpose" passes the linearised operator's transpose instead):
- HJB builds advection matrix A_HJB using velocity-based linear upwind
- FP uses A_HJB^T directly (transpose)
- Diffusion is handled separately (symmetric, always adjoint-consistent)

At boundaries:
- HJB build_advection_matrix: skips off-diagonal if no neighbor exists
- FP explicit no-flux: zeros outward advection flux

Are these equivalent?

Issue #625: Adjoint-consistent BC investigation
"""

import numpy as np
import scipy.sparse as sp


def build_hjb_advection_matrix_1d(
    U: np.ndarray,
    dx: float,
    coupling_coeff: float,
) -> sp.csr_matrix:
    """
    Build HJB ADVECTION-ONLY matrix (matching codebase implementation).

    This matches HJBFDMSolver.build_advection_matrix() in hjb_fdm.py.

    Uses velocity-based linear upwind:
    - v_i = -coupling * ∂U/∂x at point i
    - If v_i > 0: backward difference (upwind from left)
    - If v_i < 0: forward difference (upwind from right)

    At boundaries: simply skip the off-diagonal if no neighbor exists.
    """
    Nx = len(U)

    # Compute gradient at each point (one-sided at boundaries)
    grad_U = np.zeros(Nx)
    grad_U[0] = (U[1] - U[0]) / dx  # Forward at left
    grad_U[-1] = (U[-1] - U[-2]) / dx  # Backward at right
    for i in range(1, Nx - 1):
        # Central difference for interior
        grad_U[i] = (U[i + 1] - U[i - 1]) / (2 * dx)

    # Velocity: v = -coupling * grad_U
    velocity = -coupling_coeff * grad_U

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
            # At i=0: no left neighbor, skip (this is the boundary treatment)

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

    A_hjb = sp.coo_matrix((data_values, (row_indices, col_indices)), shape=(Nx, Nx)).tocsr()
    return A_hjb


def build_fp_advection_matrix_1d(
    U: np.ndarray,
    dx: float,
    coupling_coeff: float,
) -> sp.csr_matrix:
    """
    Build FP ADVECTION-ONLY matrix with explicit no-flux BC.

    This matches the advection part of fp_fdm_bc.py.

    Uses divergence-upwind (conservative form):
    - Compute flux at cell faces
    - Upwind selection for flux direction
    - At boundaries: zero outward flux (no-flux BC)
    """
    Nx = len(U)

    # Compute velocity at interior cell faces
    # v_{i+1/2} = -coupling * (U[i+1] - U[i]) / dx
    v_faces = np.zeros(Nx + 1)
    for i in range(Nx - 1):
        grad_U = (U[i + 1] - U[i]) / dx
        v_faces[i + 1] = -coupling_coeff * grad_U

    # Boundary faces: v_0 and v_Nx are exterior, set to 0 for no-flux
    v_faces[0] = 0.0
    v_faces[-1] = 0.0

    row_indices = []
    col_indices = []
    data_values = []

    for i in range(Nx):
        diagonal = 0.0

        # Right face flux contribution to cell i
        v_right = v_faces[i + 1]
        if i < Nx - 1:  # Has right neighbor
            if v_right >= 0:
                # Outflow to right: upwind from cell i
                diagonal += v_right / dx
            else:
                # Inflow from right: upwind from cell i+1
                row_indices.append(i)
                col_indices.append(i + 1)
                data_values.append(v_right / dx)

        # Left face flux contribution to cell i (negative because flux enters)
        v_left = v_faces[i]
        if i > 0:  # Has left neighbor
            if v_left >= 0:
                # Inflow from left: upwind from cell i-1
                row_indices.append(i)
                col_indices.append(i - 1)
                data_values.append(-v_left / dx)
            else:
                # Outflow to left: upwind from cell i
                diagonal += -v_left / dx

        if abs(diagonal) > 1e-14:
            row_indices.append(i)
            col_indices.append(i)
            data_values.append(diagonal)

    A_fp = sp.coo_matrix((data_values, (row_indices, col_indices)), shape=(Nx, Nx)).tocsr()
    return A_fp


def analyze_adjoint_consistency(
    Nx: int = 21,
    coupling_coeff: float = 0.5,
):
    """
    Compare A_FP (advection only) with A_HJB^T at boundary and interior points.

    Diffusion is symmetric, so we only need to verify advection adjoint consistency.
    """
    dx = 1.0 / (Nx - 1)

    # Create a test value function (quadratic, stall at x=0.3)
    x = np.linspace(0, 1, Nx)
    x_stall = 0.3
    U = (x - x_stall) ** 2

    print("=" * 70)
    print("Boundary Adjoint Consistency Analysis (ADVECTION ONLY)")
    print("=" * 70)
    print(f"\nGrid: Nx={Nx}, dx={dx:.4f}")
    print(f"Parameters: lambda={coupling_coeff}")
    print(f"Value function: U(x) = (x - {x_stall})^2")
    print(f"Stall point: x = {x_stall} (interior)")
    print()

    # Build ADVECTION-ONLY matrices
    A_hjb = build_hjb_advection_matrix_1d(U, dx, coupling_coeff)
    A_fp = build_fp_advection_matrix_1d(U, dx, coupling_coeff)
    A_hjb_T = A_hjb.T.tocsr()

    print("Matrix shapes:")
    print(f"  A_HJB:   {A_hjb.shape}")
    print(f"  A_FP:    {A_fp.shape}")
    print(f"  A_HJB^T: {A_hjb_T.shape}")
    print()

    # Compare matrices
    diff = A_fp - A_hjb_T
    diff_dense = diff.toarray()

    print("=" * 70)
    print("Comparison: A_FP vs A_HJB^T (advection only)")
    print("=" * 70)

    # Interior analysis
    interior_diff = diff_dense[1:-1, :]
    interior_max = np.max(np.abs(interior_diff))
    interior_norm = np.linalg.norm(interior_diff, "fro")

    print(f"\nInterior points (rows 1 to {Nx - 2}):")
    print(f"  Max |A_FP - A_HJB^T|: {interior_max:.2e}")
    print(f"  Frobenius norm:       {interior_norm:.2e}")

    if interior_max < 1e-10:
        print("  => Interior: ADJOINT CONSISTENT")
    else:
        print("  => Interior: NOT ADJOINT CONSISTENT")

    # Boundary analysis
    print("\nLeft boundary (row 0):")
    print(f"  A_FP[0, :]:    {A_fp.toarray()[0, :5]}")
    print(f"  A_HJB^T[0, :]: {A_hjb_T.toarray()[0, :5]}")
    print(f"  Difference:    {diff_dense[0, :5]}")
    left_diff = np.max(np.abs(diff_dense[0, :]))
    print(f"  Max |diff|:    {left_diff:.2e}")

    print(f"\nRight boundary (row {Nx - 1}):")
    print(f"  A_FP[-1, :]:   {A_fp.toarray()[-1, -5:]}")
    print(f"  A_HJB^T[-1, :]: {A_hjb_T.toarray()[-1, -5:]}")
    print(f"  Difference:    {diff_dense[-1, -5:]}")
    right_diff = np.max(np.abs(diff_dense[-1, :]))
    print(f"  Max |diff|:    {right_diff:.2e}")

    # Analyze the nature of the discrepancy
    print("\n" + "=" * 70)
    print("Analysis of Boundary Discrepancy")
    print("=" * 70)

    # Check velocity at boundaries
    grad_U_left = (U[1] - U[0]) / dx
    grad_U_right = (U[-1] - U[-2]) / dx
    v_left = -coupling_coeff * grad_U_left
    v_right = -coupling_coeff * grad_U_right

    print("\nLeft boundary (x=0):")
    print(f"  ∂U/∂x = {grad_U_left:.4f}")
    print(f"  v = -λ·∂U/∂x = {v_left:.4f}")
    print(f"  Flow direction: {'outward (→)' if v_left > 0 else 'inward (←)' if v_left < 0 else 'zero'}")

    print("\nRight boundary (x=1):")
    print(f"  ∂U/∂x = {grad_U_right:.4f}")
    print(f"  v = -λ·∂U/∂x = {v_right:.4f}")
    print(f"  Flow direction: {'inward (→)' if v_right > 0 else 'outward (←)' if v_right < 0 else 'zero'}")

    # Explain the discrepancy
    print("\n" + "=" * 70)
    print("Root Cause Analysis")
    print("=" * 70)

    print("\nHJB build_advection_matrix approach:")
    print("  - At boundary: skips off-diagonal if no neighbor")
    print("  - Effect: advection coefficient just isn't added")
    print("  - Problem: transpose doesn't preserve no-flux semantics")

    print("\nFP explicit no-flux approach:")
    print("  - Computes face velocities (cell-centered)")
    print("  - Boundary faces have v = 0 (explicit no-flux)")
    print("  - Effect: boundary cell only sees interior flux")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    boundary_consistent = left_diff < 1e-10 and right_diff < 1e-10
    interior_consistent = interior_max < 1e-10

    print(f"\n  Interior adjoint consistency: {'YES' if interior_consistent else 'NO'}")
    print(f"  Boundary adjoint consistency: {'YES' if boundary_consistent else 'NO'}")

    if not boundary_consistent:
        print("\n  IMPLICATION: Using A_HJB^T directly violates no-flux BC!")
        print("\n  The discrepancy comes from different boundary treatments:")
        print("  - HJB: point-based velocity, skip missing neighbors")
        print("  - FP: face-based flux, explicitly zero boundary faces")
        print("\n  SOLUTIONS:")
        print("  1. Modify HJB matrix to use face-based fluxes (match FP)")
        print("  2. Post-process A_HJB^T to fix boundary rows")
        print("  3. Build A_FP explicitly (current approach in code)")

    return {
        "interior_max_diff": interior_max,
        "left_boundary_diff": left_diff,
        "right_boundary_diff": right_diff,
        "interior_consistent": interior_consistent,
        "boundary_consistent": boundary_consistent,
    }


def test_different_stall_locations():
    """Test adjoint consistency for different stall locations."""
    print("\n" + "=" * 70)
    print("Sensitivity to Stall Location (Advection Only)")
    print("=" * 70)

    Nx = 21
    dx = 1.0 / (Nx - 1)
    x = np.linspace(0, 1, Nx)
    coupling_coeff = 0.5

    stall_locations = [0.0, 0.25, 0.5, 0.75, 1.0]

    print(f"\n{'Stall':>8} | {'Left BC':>10} | {'Right BC':>10} | {'Interior':>10} | {'Flow@L':>8} | {'Flow@R':>8}")
    print("-" * 75)

    for x_stall in stall_locations:
        U = (x - x_stall) ** 2

        A_hjb = build_hjb_advection_matrix_1d(U, dx, coupling_coeff)
        A_fp = build_fp_advection_matrix_1d(U, dx, coupling_coeff)
        A_hjb_T = A_hjb.T.tocsr()

        diff = (A_fp - A_hjb_T).toarray()
        left_diff = np.max(np.abs(diff[0, :]))
        right_diff = np.max(np.abs(diff[-1, :]))
        interior_diff = np.max(np.abs(diff[1:-1, :])) if Nx > 2 else 0.0

        # Flow directions
        v_left = -coupling_coeff * (U[1] - U[0]) / dx
        v_right = -coupling_coeff * (U[-1] - U[-2]) / dx
        flow_left = "out" if v_left > 0 else "in" if v_left < 0 else "zero"
        flow_right = "in" if v_right > 0 else "out" if v_right < 0 else "zero"

        print(
            f"{x_stall:>8.2f} | {left_diff:>10.2e} | {right_diff:>10.2e} | {interior_diff:>10.2e} | {flow_left:>8} | {flow_right:>8}"
        )


if __name__ == "__main__":
    # Main analysis
    results = analyze_adjoint_consistency(Nx=21, coupling_coeff=0.5)

    # Sensitivity analysis
    test_different_stall_locations()

    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
The boundary adjoint inconsistency arises from different discretization approaches:

1. HJB uses POINT-based velocity:
   - v_i = -λ * ∂U/∂x evaluated at grid point i
   - At boundary, one-sided gradient for v_i
   - Matrix entry depends on sign of v_i at the boundary point

2. FP uses FACE-based flux (finite volume):
   - F_{i+1/2} = v_{i+1/2} * m_upwind
   - Boundary faces (i=-1/2, i=Nx-1/2) have F = 0 explicitly
   - This ensures mass conservation

The transpose of HJB's point-based matrix does NOT equal FP's face-based matrix
because the boundary treatment differs fundamentally.

RECOMMENDATION:
For strict adjoint mode to work correctly at boundaries, either:
a) HJB should also use face-based flux discretization, OR
b) Post-process A_HJB^T to zero out boundary flux contributions
""")

    print("\n" + "=" * 70)
    print("Diagnostic complete.")
    print("=" * 70)

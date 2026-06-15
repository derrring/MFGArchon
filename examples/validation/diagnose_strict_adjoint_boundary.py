#!/usr/bin/env python3
"""
Diagnostic: Strict Adjoint Mode Boundary Behavior.

This script uses the ACTUAL codebase implementations to verify:
1. What A_HJB^T looks like at boundaries
2. How solve_fp_step_adjoint_mode() treats these boundaries
3. Whether mass is conserved (no-flux violation detection)

Issue #625: Strict Adjoint Mode boundary consistency investigation
"""

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def diagnose_boundary_behavior(Nx: int = 21, x_stall: float = 0.3):
    """
    Use actual solvers to build matrices and analyze boundary behavior.
    """
    print("=" * 70)
    print("Strict Adjoint Mode: Boundary Behavior Diagnosis")
    print("=" * 70)

    # Create a simple 1D problem
    geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[Nx + 1], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(
        geometry=geometry,
        T=1.0,
        Nt=11,
        sigma=0.2,
    )

    # Create value function with stall at x_stall
    x = np.linspace(0, 1, Nx)
    U = (x - x_stall) ** 2

    print("\nProblem Setup:")
    print(f"  Grid: Nx={Nx}, dx={1.0 / (Nx - 1):.4f}")
    print(f"  Value function: U(x) = (x - {x_stall})^2")
    print(f"  Stall point: x = {x_stall}")
    print()

    # Create solvers
    hjb_solver = HJBFDMSolver(problem)
    fp_solver = FPFDMSolver(problem)

    # Build HJB advection matrix using actual codebase
    coupling = getattr(problem, "coupling_coefficient", 0.5)
    A_hjb = hjb_solver.build_advection_matrix(U, coupling_coefficient=coupling)
    A_hjb_T = A_hjb.T.tocsr()

    print("=" * 70)
    print("HJB Advection Matrix Analysis")
    print("=" * 70)

    print(f"\nMatrix A_HJB shape: {A_hjb.shape}")
    print(f"Matrix A_HJB^T shape: {A_hjb_T.shape}")
    print(f"Non-zero entries: {A_hjb.nnz}")

    # Analyze boundary rows
    print("\nLeft boundary (row 0):")
    print(f"  A_HJB[0, :5]:   {A_hjb.toarray()[0, :5]}")
    print(f"  A_HJB^T[0, :5]: {A_hjb_T.toarray()[0, :5]}")

    print(f"\nRight boundary (row {Nx - 1}):")
    print(f"  A_HJB[-1, -5:]:   {A_hjb.toarray()[-1, -5:]}")
    print(f"  A_HJB^T[-1, -5:]: {A_hjb_T.toarray()[-1, -5:]}")

    # Check for boundary flux leakage
    print("\n" + "=" * 70)
    print("No-Flux Boundary Condition Analysis")
    print("=" * 70)

    # For no-flux BC, the row sum should be zero (mass conservation)
    # If row sum != 0, mass can leak through boundary
    row_sums = np.array(A_hjb_T.sum(axis=1)).ravel()

    print("\nA_HJB^T row sums (should be ~0 for mass conservation):")
    print(f"  Row 0 (left boundary):  {row_sums[0]:.6f}")
    print(f"  Row {Nx - 1} (right boundary): {row_sums[-1]:.6f}")
    print(f"  Interior max |row sum|: {np.max(np.abs(row_sums[1:-1])):.6f}")

    # Physical interpretation
    if abs(row_sums[0]) > 1e-10 or abs(row_sums[-1]) > 1e-10:
        print("\n  WARNING: Non-zero boundary row sums indicate mass flux at boundaries!")
        print("           This violates no-flux BC when using A_HJB^T directly.")

    # Check column sums for completeness
    col_sums = np.array(A_hjb_T.sum(axis=0)).ravel()
    print("\nA_HJB^T column sums:")
    print(f"  Col 0 (left boundary):  {col_sums[0]:.6f}")
    print(f"  Col {Nx - 1} (right boundary): {col_sums[-1]:.6f}")

    # Test mass conservation with actual FP step
    print("\n" + "=" * 70)
    print("Mass Conservation Test (FP Step with A_HJB^T)")
    print("=" * 70)

    # Initial density: uniform
    M_initial = np.ones(Nx) / Nx  # Normalized
    initial_mass = np.sum(M_initial)

    print(f"\nInitial mass: {initial_mass:.10f}")

    # Run one FP step using solve_fp_step_adjoint_mode
    M_next = fp_solver.solve_fp_step_adjoint_mode(
        M_current=M_initial,
        A_advection_T=A_hjb_T,
        sigma=problem.sigma,
        time=0.0,
    )
    final_mass = np.sum(M_next)

    print(f"Mass after 1 FP step: {final_mass:.10f}")
    print(f"Mass change: {final_mass - initial_mass:+.2e}")
    print(f"Relative mass change: {(final_mass - initial_mass) / initial_mass * 100:+.4f}%")

    if abs(final_mass - initial_mass) > 1e-6:
        print("\n  WARNING: Mass not conserved! This confirms no-flux BC violation.")
    else:
        print("\n  Mass is conserved (within numerical tolerance).")

    # Run multiple steps to see accumulation
    print("\n" + "=" * 70)
    print("Multi-Step Mass Tracking")
    print("=" * 70)

    M_current = M_initial.copy()
    masses = [initial_mass]

    for step in range(10):
        M_current = fp_solver.solve_fp_step_adjoint_mode(M_current, A_hjb_T, sigma=problem.sigma)
        masses.append(np.sum(M_current))

    print("\nMass evolution over 10 FP steps:")
    print("  Step | Mass       | Change")
    print("  " + "-" * 35)
    for i, mass in enumerate(masses):
        change = mass - initial_mass
        print(f"  {i:4d} | {mass:.8f} | {change:+.2e}")

    total_drift = masses[-1] - masses[0]
    print(f"\nTotal mass drift: {total_drift:+.2e}")

    return {
        "A_hjb": A_hjb,
        "A_hjb_T": A_hjb_T,
        "row_sums": row_sums,
        "masses": masses,
        "mass_drift": total_drift,
    }


def compare_with_explicit_fp(Nx: int = 21, x_stall: float = 0.3):
    """
    Compare mass conservation between:
    1. Strict Adjoint Mode (A_HJB^T)
    2. Standard FP (explicit FP matrix)
    """
    print("\n" + "=" * 70)
    print("Comparison: Strict Adjoint vs Standard FP")
    print("=" * 70)

    geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[Nx + 1], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(
        geometry=geometry,
        T=1.0,
        Nt=11,
        sigma=0.2,
    )

    x = np.linspace(0, 1, Nx)
    U = (x - x_stall) ** 2

    hjb_solver = HJBFDMSolver(problem)
    fp_solver = FPFDMSolver(problem)

    coupling = getattr(problem, "coupling_coefficient", 0.5)
    A_hjb = hjb_solver.build_advection_matrix(U, coupling_coefficient=coupling)

    # Initial density
    M_initial = np.ones(Nx) / Nx

    # Method 1: Strict Adjoint Mode
    M_adjoint = M_initial.copy()
    for _ in range(10):
        M_adjoint = fp_solver.solve_fp_step_adjoint_mode(M_adjoint, A_hjb.T, sigma=problem.sigma)

    # Method 2: Standard FP (uses internal matrix construction)
    M_solution = fp_solver.solve(M_initial, U)
    # Take slice at t=10*dt
    step_idx = min(10, M_solution.shape[0] - 1)
    M_standard = M_solution[step_idx]

    print("\nAfter 10 timesteps:")
    print(f"  Strict Adjoint Mode mass: {np.sum(M_adjoint):.8f}")
    print(f"  Standard FP mass:         {np.sum(M_standard):.8f}")
    print(f"  Initial mass:             {np.sum(M_initial):.8f}")

    print("\nMass drift:")
    print(f"  Strict Adjoint: {np.sum(M_adjoint) - np.sum(M_initial):+.2e}")
    print(f"  Standard FP:    {np.sum(M_standard) - np.sum(M_initial):+.2e}")


if __name__ == "__main__":
    # Main diagnosis
    result = diagnose_boundary_behavior(Nx=21, x_stall=0.3)

    # Comparison with standard FP
    compare_with_explicit_fp(Nx=21, x_stall=0.3)

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The strict adjoint mode (Issue #622) currently uses A_HJB^T directly without
any boundary correction. This means:

1. The FP equation sees advection flux from A_HJB^T at boundaries
2. A_HJB^T preserves terms that allow mass to "flow through" boundaries
3. This violates the no-flux boundary condition

RECOMMENDED FIX:
Post-process A_HJB^T to zero out boundary flux contributions:

```python
def fix_boundary_no_flux(A_T, Nx):
    '''Zero out boundary flux in transposed advection matrix.'''
    A_fixed = A_T.copy()
    # Left boundary: zero the off-diagonal in row 0
    A_fixed[0, 1:] = 0  # No flux from interior to left boundary
    # Right boundary: zero the off-diagonal in row Nx-1
    A_fixed[-1, :-1] = 0  # No flux from interior to right boundary
    return A_fixed
```

This preserves the interior adjoint consistency while enforcing no-flux BC.
""")

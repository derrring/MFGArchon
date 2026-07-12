#!/usr/bin/env python3
"""
MFG Problem Solving Demonstration

Demonstrates the primary API for solving Mean Field Games problems.

Primary API: problem.solve()
----------------------------
Build a problem from its model (Hamiltonian + volatility), domain (grid), and
conditions (terminal/initial data), then call ``solve()``. The same problem can
be solved in three modes:

- Auto Mode:   ``problem.solve()`` -- the solver picks a scheme.
- Safe Mode:   ``problem.solve(scheme=...)`` -- you pick the scheme; the HJB/FP
               pair is built for you with guaranteed adjoint duality.
- Expert Mode: ``problem.solve(hjb_solver=..., fp_solver=...)`` -- you inject
               solvers; the pairing is validated.

Run:
    python examples/basic/core_infrastructure/solve_mfg_demo.py
"""

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.types import NumericalScheme


def build_problem(Nx: int = 20, Nt: int = 10) -> MFGProblem:
    """Build a small, well-posed 1D LQ-MFG problem (v1.0 API).

    - model: separable Hamiltonian with quadratic control cost and a weak
      congestion coupling, plus volatility sigma.
    - domain: a 1D tensor-product grid with no-flux boundaries.
    - conditions: terminal cost u_terminal and initial density m_initial.
    """
    model = Model(
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: 0.05 * m,
            coupling_dm=lambda m: 0.05,
        ),
        sigma=0.15,
    )
    conditions = Conditions(
        u_terminal=lambda x: (x - 0.5) ** 2,
        m_initial=lambda x: np.exp(-50.0 * (x - 0.5) ** 2),
        T=1.0,
    )
    domain = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[Nx],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    return MFGProblem(model=model, domain=domain, conditions=conditions, Nt=Nt)


def _report(result) -> None:
    """Print the standard SolverResult summary."""
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Final error (max): {result.max_error:.2e}")
    if result.execution_time is not None:
        print(f"  Execution time: {result.execution_time:.3f}s")
    print(f"  Solution shapes: U={result.U.shape}, M={result.M.shape}")


def demo_simple_usage():
    """Auto Mode: problem.solve() with defaults."""
    print("\n" + "=" * 60)
    print("Demo 1: Simplest Usage (Auto Mode)")
    print("=" * 60)

    problem = build_problem()
    result = problem.solve(tolerance=1e-4)
    print("\nSolved:")
    _report(result)


def demo_custom_parameters():
    """Auto Mode with explicit solve parameters."""
    print("\n" + "=" * 60)
    print("Demo 2: Custom Solve Parameters")
    print("=" * 60)

    problem = build_problem()
    result = problem.solve(max_iterations=100, tolerance=1e-6, verbose=False)
    print("\nSolved:")
    _report(result)


def demo_safe_mode():
    """Safe Mode: choose the numerical scheme; the HJB/FP pair is built for you."""
    print("\n" + "=" * 60)
    print("Demo 3: Safe Mode (explicit scheme)")
    print("=" * 60)

    problem = build_problem()
    result = problem.solve(scheme=NumericalScheme.FDM_UPWIND, max_iterations=100, tolerance=1e-4)
    print("\nSolved:")
    _report(result)


def demo_expert_mode():
    """Expert Mode: inject HJB and FP solvers directly (pairing is validated)."""
    print("\n" + "=" * 60)
    print("Demo 4: Expert Mode (manual solver injection)")
    print("=" * 60)

    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    problem = build_problem()
    hjb_solver = HJBFDMSolver(problem)
    fp_solver = FPFDMSolver(problem)

    result = problem.solve(hjb_solver=hjb_solver, fp_solver=fp_solver, max_iterations=100, tolerance=1e-4)
    print("\nSolved:")
    _report(result)


if __name__ == "__main__":
    print("MFG Problem Solving Demonstration")
    print("=" * 60)
    print("Primary API: problem.solve()")
    print("=" * 60)

    demo_simple_usage()
    demo_custom_parameters()
    demo_safe_mode()
    demo_expert_mode()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)

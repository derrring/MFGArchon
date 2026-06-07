"""
Test that WENO solver uses high-order ghost cells (Issue #576, Phase 6).

Verifies that HJBWENOSolver correctly uses order=5 polynomial extrapolation
for ghost cell generation, enabling true 5th-order boundary accuracy.
"""

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _default_hamiltonian():
    """Default Hamiltonian for testing (Issue #670: explicit specification required)."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components():
    """Default MFGComponents for testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2).squeeze(),
        u_terminal=lambda x: 0.0,
        hamiltonian=_default_hamiltonian(),
    )


def test_weno_uses_high_order_ghosts():
    """Test that WENO solver creates ghost buffer with order=5."""
    # Create a simple 1D MFG problem using modern geometry-first API
    domain = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[50], boundary_conditions=no_flux_bc(dimension=1))

    problem = MFGProblem(
        geometry=domain,
        T=1.0,
        Nt=10,
        sigma=0.1,
        components=_default_components(),
    )

    # Import WENO solver
    from mfgarchon.alg.numerical.hjb_solvers import HJBWenoSolver

    # Create WENO solver
    solver = HJBWenoSolver(problem)

    # Check that ghost buffer was created with correct parameters.
    # Issue #1200: the Osher-Shu HJ-WENO5 one-sided derivative stencil spans
    # u_{i-3}..u_{i+3}, so WENO5 needs 3 ghost cells per side (was 2 for the old,
    # incorrect, value-interface reconstruction).
    assert solver.ghost_buffer is not None, "WENO should create ghost buffer in 1D"
    assert solver.ghost_depth == 3, "HJ-WENO5 needs 3 ghost cells per side (#1200)"
    assert solver.ghost_order == 5, "WENO5 should use order=5 for high-order accuracy"
    assert solver.ghost_buffer._order == 5, "Ghost buffer should have order=5"
    assert solver.ghost_buffer._ghost_depth == 3, "Ghost buffer should have depth=3"

    print(f"✓ WENO ghost buffer: depth={solver.ghost_depth}, order={solver.ghost_order}")


def test_weno_ghost_cells_work():
    """Test that WENO can update ghost cells with polynomial extrapolation."""
    # Create a simple 1D MFG problem using modern geometry-first API
    domain = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[50], boundary_conditions=no_flux_bc(dimension=1))

    problem = MFGProblem(
        geometry=domain,
        T=1.0,
        Nt=10,
        sigma=0.1,
        components=_default_components(),
    )

    # Import WENO solver
    from mfgarchon.alg.numerical.hjb_solvers import HJBWenoSolver

    # Create WENO solver
    solver = HJBWenoSolver(problem)

    # Test ghost cell update with smooth function
    x = np.linspace(0.0, 1.0, solver.num_grid_points_x)
    u = x**3  # Cubic function

    # Update ghost cells
    solver.ghost_buffer.interior[:] = u
    solver.ghost_buffer.update_ghosts(time=0.0)

    # Ghost cell ordering (depth g): padded[0] is furthest from the boundary
    # (x = -g*dx), padded[g-1] is nearest (x = -dx). Extrapolation error grows with
    # distance, so the nearest ghost is the tight accuracy gate; the farther ones
    # only need to stay bounded (Issue #1200 bumped the depth 2 -> 3).
    g = solver.ghost_depth
    dx = 1.0 / (solver.num_grid_points_x - 1)

    ghosts = [solver.ghost_buffer.padded[k] for k in range(g)]
    assert np.all(np.isfinite(ghosts)), f"ghost cells must be finite, got {ghosts}"

    # The data is a cubic and the ghost scheme is a degree-5 polynomial extrapolation
    # (n_stencil=order=5 interior points + p'(0)=0), so x^3 is reproduced EXACTLY at every
    # ghost regardless of distance -- a true machine-precision guard (a distance-scaled
    # tolerance would let a real boundary-extrapolation regression pass silently).
    for k in range(g):
        x_ghost = -(g - k) * dx  # padded[k] sits at this coordinate
        u_exact = x_ghost**3
        error = np.abs(ghosts[k] - u_exact)
        print(f"  Ghost[{k}] @ x={x_ghost:.6f}: computed={ghosts[k]:.6f}, exact={u_exact:.6f}, error={error:.6e}")
        assert error < 1e-10, f"Ghost {k} (x={x_ghost:.4f}) error too large: {error}"

    print("✓ WENO high-order ghost cells work correctly")


if __name__ == "__main__":
    """Run smoke tests."""
    print("Testing WENO high-order ghost cell integration...")

    print("\n1. Check WENO ghost buffer configuration:")
    test_weno_uses_high_order_ghosts()

    print("\n2. Verify ghost cell update works:")
    test_weno_ghost_cells_work()

    print("\n✅ All WENO ghost integration tests passed!")

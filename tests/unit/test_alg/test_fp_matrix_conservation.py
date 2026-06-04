"""Conservation tests for the FP FDM solver.

Verifies the production Fokker-Planck implicit assembly conserves mass and stays
positive -- the linear-algebraic root of MFG mass conservation (Achdou &
Capuzzo-Dolcetta 2010, Issue #486).

History: an earlier ``TestFPMatrixConservation`` asserted column-sum = 1/dt on a
*test-local re-implementation* of the matrix (a shadow that could pass even if the
production ``FPFDMSolver`` assembly diverged) and a no-op placeholder test. Those are
replaced by ``TestFPProductionConservation``, which runs the real solver: the
machine-precision mass invariant is the signal a wrong boundary stencil (the
retrospect's non-conservation-at-boundary bug class) would break, and the loose
``rtol=0.1`` end-to-end checks in ``test_fp_fdm_solver.py`` would miss.
``TestLinearConstraintMatrixAssembly`` below already points at production
(``_build_diffusion_matrix_with_bc``) and is unchanged.
"""

from __future__ import annotations

import pytest

import numpy as np
import scipy.sparse as sparse

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc, periodic_bc


def _zero_drift_density_evolution(bc, n=41, nt=40, T=0.2, sigma=0.5):
    """Evolve m = 1 + 0.4 cos(2 pi x) under the production FP implicit step, zero drift.

    Zero drift isolates pure diffusion, where every conservative scheme must preserve
    total mass exactly (no advective boundary subtlety). Returns the full M history.
    """
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=bc)
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.0 * np.asarray(m),
        coupling_dm=lambda m: 0.0 * np.asarray(m),
    )
    x = np.linspace(0.0, 1.0, n)
    comps = MFGComponents(
        m_initial=lambda xx: 1.0 + 0.4 * np.cos(2 * np.pi * np.asarray(xx)),
        u_terminal=lambda xx: 0.0 * np.asarray(xx),
        hamiltonian=H,
    )
    prob = MFGProblem(geometry=grid, T=T, Nt=nt, sigma=sigma, components=comps)
    solver = FPFDMSolver(prob)
    m0 = 1.0 + 0.4 * np.cos(2 * np.pi * x)
    # drift_field as an (nt+1, n) array routes through the implicit per-point assembly
    M = solver.solve_fp_system(m0.copy(), drift_field=np.zeros((nt + 1, n)), volatility_field=sigma)
    return M, 1.0 / (n - 1)


class TestFPProductionConservation:
    """Mass + positivity of the real FPFDMSolver implicit step (not a shadow matrix)."""

    def test_mass_conserved_periodic_zero_drift(self):
        M, dx = _zero_drift_density_evolution(periodic_bc(dimension=1))
        mass = M.sum(axis=1) * dx
        relerr = abs(mass[-1] - mass[0]) / mass[0]
        assert relerr < 1e-12, f"periodic zero-drift mass not conserved: relerr {relerr:.2e}"

    def test_mass_conserved_no_flux_zero_drift(self):
        M, dx = _zero_drift_density_evolution(no_flux_bc(dimension=1))
        mass = M.sum(axis=1) * dx
        relerr = abs(mass[-1] - mass[0]) / mass[0]
        assert relerr < 1e-12, f"no-flux zero-drift mass not conserved: relerr {relerr:.2e}"

    def test_density_stays_positive(self):
        # an M-matrix implicit step keeps a positive initial density positive
        M, _ = _zero_drift_density_evolution(periodic_bc(dimension=1))
        assert np.all(M[-1] > -1e-12), "production FP step produced a negative density"


class TestLinearConstraintMatrixAssembly:
    """
    Test LinearConstraint-based matrix assembly.

    These tests verify the implementation of the matrix assembly protocol
    from docs/development/matrix_assembly_bc_protocol.md.

    The key principle is Algebraic-Geometric Equivalence:
    - Explicit solver (ghost cells) and implicit solver (matrix folding)
      must produce identical numerical results.
    """

    def test_neumann_bc_via_linear_constraint(self):
        """Test Neumann BC using LinearConstraint coefficient folding."""
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import (
            _build_diffusion_matrix_with_bc,
        )
        from mfgarchon.geometry.boundary.applicator_base import LinearConstraint

        # Setup
        Nx = 11
        dx = 0.1
        dt = 0.01
        D = 0.05  # diffusion coefficient

        shape = (Nx,)
        spacing = (dx,)
        ndim = 1

        # Neumann BC: du/dn = 0 -> u_ghost = u_inner (Tier 2)
        neumann_constraint = LinearConstraint(weights={0: 1.0}, bias=0.0)

        A, b_bc = _build_diffusion_matrix_with_bc(
            shape=shape,
            spacing=spacing,
            D=D,
            dt=dt,
            ndim=ndim,
            bc_constraint_min=neumann_constraint,
            bc_constraint_max=neumann_constraint,
        )

        # Check matrix shape
        assert A.shape == (Nx, Nx), f"Expected ({Nx}, {Nx}), got {A.shape}"

        # Check b_bc is zero for homogeneous Neumann
        np.testing.assert_allclose(b_bc, 0.0, atol=1e-14)

        # Check row sums (should equal 1/dt for implicit scheme)
        row_sums = np.array(A.sum(axis=1)).flatten()
        np.testing.assert_allclose(
            row_sums,
            1.0 / dt,
            rtol=1e-10,
            err_msg="Row sums should equal 1/dt",
        )

    def test_dirichlet_bc_via_linear_constraint(self):
        """Test Dirichlet BC using LinearConstraint coefficient folding."""
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import (
            _build_diffusion_matrix_with_bc,
        )
        from mfgarchon.geometry.boundary.applicator_base import LinearConstraint

        # Setup
        Nx = 11
        dx = 0.1
        dt = 0.01
        D = 0.05

        shape = (Nx,)
        spacing = (dx,)
        ndim = 1

        # Dirichlet BC: u = g at boundary (Tier 1)
        # For cell-centered: u_ghost = 2*g - u_inner
        # LinearConstraint: weights={0: -1.0}, bias=2*g
        g_left = 1.0
        g_right = 2.0
        dirichlet_left = LinearConstraint(weights={0: -1.0}, bias=2 * g_left)
        dirichlet_right = LinearConstraint(weights={0: -1.0}, bias=2 * g_right)

        A, b_bc = _build_diffusion_matrix_with_bc(
            shape=shape,
            spacing=spacing,
            D=D,
            dt=dt,
            ndim=ndim,
            bc_constraint_min=dirichlet_left,
            bc_constraint_max=dirichlet_right,
        )

        # Check matrix shape
        assert A.shape == (Nx, Nx)

        # Check b_bc has non-zero entries at boundaries
        # The bias terms should contribute to b_bc[0] and b_bc[-1]
        assert b_bc[0] != 0.0, "Left boundary should have BC contribution"
        assert b_bc[-1] != 0.0, "Right boundary should have BC contribution"

    def test_linear_extrapolation_bc_via_linear_constraint(self):
        """Test linear extrapolation BC using LinearConstraint (Tier 4)."""
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import (
            _build_diffusion_matrix_with_bc,
        )
        from mfgarchon.geometry.boundary.applicator_base import LinearConstraint

        # Setup
        Nx = 11
        dx = 0.1
        dt = 0.01
        D = 0.05

        shape = (Nx,)
        spacing = (dx,)
        ndim = 1

        # Linear extrapolation: u_ghost = 2*u[0] - u[1]
        # For left boundary: weights={0: 2.0, 1: -1.0}
        # For right boundary: weights={0: 2.0, 1: -1.0} (relative to boundary-adjacent)
        linear_extrap = LinearConstraint(weights={0: 2.0, 1: -1.0}, bias=0.0)

        A, b_bc = _build_diffusion_matrix_with_bc(
            shape=shape,
            spacing=spacing,
            D=D,
            dt=dt,
            ndim=ndim,
            bc_constraint_min=linear_extrap,
            bc_constraint_max=linear_extrap,
        )

        # Check matrix shape
        assert A.shape == (Nx, Nx)

        # Check b_bc is zero for extrapolation (no bias)
        np.testing.assert_allclose(b_bc, 0.0, atol=1e-14)

        # Matrix should be non-singular (solvable)
        m_test = np.ones(Nx)
        b_test = m_test / dt
        m_result = sparse.linalg.spsolve(A, b_test)
        assert not np.any(np.isnan(m_result)), "Matrix should be solvable"

    def test_2d_neumann_bc_via_linear_constraint(self):
        """Test 2D Neumann BC using LinearConstraint."""
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import (
            _build_diffusion_matrix_with_bc,
        )
        from mfgarchon.geometry.boundary.applicator_base import LinearConstraint

        # Setup
        Nx, Ny = 6, 6
        dx, dy = 0.1, 0.1
        dt = 0.01
        D = 0.05

        shape = (Nx, Ny)
        spacing = (dx, dy)
        ndim = 2

        # Neumann BC
        neumann = LinearConstraint(weights={0: 1.0}, bias=0.0)

        A, b_bc = _build_diffusion_matrix_with_bc(
            shape=shape,
            spacing=spacing,
            D=D,
            dt=dt,
            ndim=ndim,
            bc_constraint_min=neumann,
            bc_constraint_max=neumann,
        )

        N_total = Nx * Ny
        assert A.shape == (N_total, N_total)

        # Check b_bc is zero
        np.testing.assert_allclose(b_bc, 0.0, atol=1e-14)

        # Check row sums
        row_sums = np.array(A.sum(axis=1)).flatten()
        np.testing.assert_allclose(
            row_sums,
            1.0 / dt,
            rtol=1e-10,
            err_msg="Row sums should equal 1/dt in 2D",
        )

    def test_mass_conservation_with_linear_constraint_neumann(self):
        """Test mass conservation using LinearConstraint-based matrix assembly."""
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import (
            _build_diffusion_matrix_with_bc,
        )
        from mfgarchon.geometry.boundary.applicator_base import LinearConstraint

        # Setup
        Nx = 21
        dx = 1.0 / (Nx - 1)
        dt = 0.01
        sigma = 0.5
        D = sigma**2 / 2.0

        shape = (Nx,)
        spacing = (dx,)
        ndim = 1

        # Initial Gaussian density
        x = np.linspace(0, 1, Nx)
        m0 = np.exp(-((x - 0.5) ** 2) / 0.1)
        m0 = m0 / (np.sum(m0) * dx)  # Normalize

        initial_mass = np.sum(m0) * dx

        # Neumann BC for mass conservation
        neumann = LinearConstraint(weights={0: 1.0}, bias=0.0)

        A, b_bc = _build_diffusion_matrix_with_bc(
            shape=shape,
            spacing=spacing,
            D=D,
            dt=dt,
            ndim=ndim,
            bc_constraint_min=neumann,
            bc_constraint_max=neumann,
        )

        # Evolve for multiple timesteps
        m = m0.copy()
        for _ in range(10):
            b_rhs = m / dt + b_bc
            m = sparse.linalg.spsolve(A, b_rhs)

        final_mass = np.sum(m) * dx

        np.testing.assert_allclose(
            final_mass,
            initial_mass,
            rtol=1e-10,
            err_msg="Mass should be conserved with Neumann BC",
        )

    def test_calculator_to_constraint_conversion(self):
        """Test conversion from Calculator to LinearConstraint."""
        from mfgarchon.geometry.boundary.applicator_base import (
            DirichletCalculator,
            NeumannCalculator,
            ZeroFluxCalculator,
            ZeroGradientCalculator,
            calculator_to_constraint,
        )

        dx = 0.1

        # Tier 1: Dirichlet -> bias only
        dirichlet = DirichletCalculator(boundary_value=5.0)
        constraint = calculator_to_constraint(dirichlet, dx, side="min")
        assert constraint.weights == {}
        assert constraint.bias == 5.0

        # Tier 2: Neumann -> weight=1.0, bias depends on flux
        neumann = NeumannCalculator(flux_value=0.0)
        constraint = calculator_to_constraint(neumann, dx, side="min")
        assert constraint.weights == {0: 1.0}
        assert constraint.bias == 0.0

        # Tier 2: ZeroGradient -> weight=1.0, bias=0.0
        zero_grad = ZeroGradientCalculator()
        constraint = calculator_to_constraint(zero_grad, dx, side="min")
        assert constraint.weights == {0: 1.0}
        assert constraint.bias == 0.0

        # Tier 3: ZeroFlux -> Robin coefficient
        zero_flux = ZeroFluxCalculator(drift_velocity=0.1, diffusion_coeff=0.5)
        constraint = calculator_to_constraint(zero_flux, dx, side="max")
        assert 0 in constraint.weights  # Should have weight for boundary cell
        assert constraint.bias == 0.0  # No bias term

    def test_calculator_to_constraint_robin(self):
        """Test Robin case: alpha*u + beta*du/dn = g gives correct ghost cell relation."""
        from mfgarchon.geometry.boundary.applicator_base import (
            RobinCalculator,
            calculator_to_constraint,
        )

        dx = 0.1

        # General Robin: alpha=1, beta=1, g=2
        robin = RobinCalculator(alpha=1.0, beta=1.0, rhs_value=2.0)

        # Min side: outward_sign = -1, beta_eff = -1
        # denom = -1 + 0.2 = -0.8
        # weight = (-1 - 0.2) / (-0.8) = 1.5
        # bias = 2*2*0.1 / (-0.8) = -0.5
        c_min = calculator_to_constraint(robin, dx, side="min")
        assert abs(c_min.weights[0] - 1.5) < 1e-10
        assert abs(c_min.bias - (-0.5)) < 1e-10

        # Max side: outward_sign = +1, beta_eff = 1
        # denom = 1 + 0.2 = 1.2
        # weight = (1 - 0.2) / 1.2 = 2/3
        # bias = 0.4 / 1.2 = 1/3
        c_max = calculator_to_constraint(robin, dx, side="max")
        assert abs(c_max.weights[0] - 2 / 3) < 1e-10
        assert abs(c_max.bias - 1 / 3) < 1e-10

        # Pure Neumann-like (alpha=0, beta=1, g=0.5) at min
        # beta_eff = -1, denom = -1 + 0 = -1
        # weight = (-1 - 0) / (-1) = 1.0, bias = 2*0.5*0.1/(-1) = -0.1
        robin_neum = RobinCalculator(alpha=0.0, beta=1.0, rhs_value=0.5)
        c_neum = calculator_to_constraint(robin_neum, dx, side="min")
        assert abs(c_neum.weights[0] - 1.0) < 1e-10
        assert abs(c_neum.bias - (-0.1)) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

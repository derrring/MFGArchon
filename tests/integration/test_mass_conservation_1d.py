"""
Mass Conservation Tests for 1D MFG Systems with No-Flux Neumann BC.

Tests mass conservation for solver combinations:
1. FP Particle + HJB FDM (standard finite difference)
2. FP Particle + HJB GFDM (particle collocation)

Both should conserve mass with no-flux Neumann boundary conditions.

Consolidated from test_mass_conservation_1d.py and test_mass_conservation_1d_simple.py.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid, no_flux_bc


def _default_hamiltonian():
    """Default Hamiltonian for testing."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components():
    """Default MFGComponents for testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),  # Gaussian centered at 0.5
        u_terminal=lambda x: 0.0,  # Zero terminal cost
        hamiltonian=_default_hamiltonian(),
    )


def compute_total_mass(density: np.ndarray, grid) -> float:
    """Total mass on the grid's own measure (#2145).

    This was `np.sum(density) * dx`, justified in its own docstring as "consistent with
    FPParticleSolver normalization which uses np.sum(density) * dx". That consistency was the
    problem: two components agreeing on one convention is not evidence the convention is right, and
    this one is not. `TensorProductGrid` is endpoint-inclusive, so the wall lies ON x_0 and the two
    end nodes own half a cell each -- the measure is the trapezoid, and those weights ARE the
    control volumes the conservative FP schemes telescope against. The rectangle over-counts by
    `dx*(m[0]+m[-1])/2`, so it read a correct solve as drifting and a wall losing real mass as exact.
    """
    return float(grid.integrate(np.asarray(density, dtype=float)))


class TestMassConservation1D:
    """Test mass conservation for 1D MFG with no-flux Neumann BC using built-in MFGProblem."""

    @pytest.fixture
    def problem(self):
        """Create standard MFG problem with Neumann BC."""
        geometry = TensorProductGrid(bounds=[(0.0, 2.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        return MFGProblem(
            geometry=geometry,
            T=1.0,
            Nt=20,
            sigma=0.1,
            coupling_coefficient=1.0,
            components=_default_components(),
        )

    @pytest.fixture
    def boundary_conditions(self):
        """No-flux Neumann boundary conditions."""
        return no_flux_bc(dimension=1)

    @pytest.mark.slow
    def test_fp_particle_hjb_fdm_mass_conservation(self, problem, boundary_conditions):
        """
        Test mass conservation for FP Particle + HJB FDM combination.

        With no-flux Neumann BC, total mass should be preserved:
        integral m(x,t)dx approx 1 for all t in [0,T]
        """
        from mfgarchon import KDENormalization

        fp_solver = FPParticleSolver(
            problem,
            num_particles=5000,
            kde_normalization=KDENormalization.NONE,
            boundary_conditions=boundary_conditions,
        )

        hjb_solver = HJBFDMSolver(problem)

        mfg_solver = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
        )

        # Issue #1567: no try/except -> skip; a solver raise must fail this mass-conservation test.
        result = mfg_solver.solve(max_iterations=50, tolerance=1e-3)

        m_solution = result.M

        grid = problem.geometry
        Nt_points = problem.Nt + 1
        masses = np.array([compute_total_mass(m_solution[t_idx, :], grid) for t_idx in range(Nt_points)])

        initial_mass = masses[0]
        print("\n=== FP Particle + HJB FDM Mass Conservation ===")
        print(f"Initial mass: {initial_mass:.6f}")
        print(f"Final mass: {masses[-1]:.6f}")

        mass_errors = np.abs(masses - initial_mass)
        max_mass_error = np.max(mass_errors)
        print(f"Max mass change: {max_mass_error:.6e}")
        print(f"Mean mass: {np.mean(masses):.6f} +/- {np.std(masses):.6e}")

        assert max_mass_error < 0.1 * initial_mass, (
            f"Mass conservation violated: max change = {max_mass_error:.6e}\n"
            f"Initial mass: {initial_mass:.6f}\n"
            f"Final mass: {masses[-1]:.6f}"
        )

    @pytest.mark.slow
    def test_fp_particle_hjb_gfdm_mass_conservation(self, problem, boundary_conditions):
        """
        Test mass conservation for FP Particle + HJB GFDM (particle collocation).

        With no-flux Neumann BC, total mass should be preserved:
        integral m(x,t)dx approx 1 for all t in [0,T]
        """
        from mfgarchon import KDENormalization

        fp_solver = FPParticleSolver(
            problem,
            num_particles=5000,
            kde_normalization=KDENormalization.NONE,
            boundary_conditions=boundary_conditions,
        )

        collocation_points = problem.geometry.get_spatial_grid().reshape(-1, 1)
        hjb_solver = HJBGFDMSolver(problem, collocation_points=collocation_points, delta=0.3)

        mfg_solver = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
        )

        # Issue #1567: no try/except -> skip; a solver raise must fail this mass-conservation test.
        result = mfg_solver.solve(max_iterations=50, tolerance=1e-3)

        m_solution = result.M

        grid = problem.geometry
        Nt_points = problem.Nt + 1
        masses = np.array([compute_total_mass(m_solution[t_idx, :], grid) for t_idx in range(Nt_points)])

        initial_mass = masses[0]
        print("\n=== FP Particle + HJB GFDM Mass Conservation ===")
        print(f"Initial mass: {initial_mass:.6f}")
        print(f"Final mass: {masses[-1]:.6f}")

        mass_errors = np.abs(masses - initial_mass)
        max_mass_error = np.max(mass_errors)
        print(f"Max mass change: {max_mass_error:.6e}")
        print(f"Mean mass: {np.mean(masses):.6f} +/- {np.std(masses):.6e}")

        assert max_mass_error < 0.1 * initial_mass, (
            f"Mass conservation violated: max change = {max_mass_error:.6e}\n"
            f"Initial mass: {initial_mass:.6f}\n"
            f"Final mass: {masses[-1]:.6f}"
        )

    @pytest.mark.slow
    def test_compare_mass_conservation_methods(self, problem, boundary_conditions):
        """
        Compare mass conservation between FDM and GFDM methods.

        Both methods should preserve mass with similar quality.
        """
        # Solver 1: FP Particle + HJB FDM
        fp_solver_1 = FPParticleSolver(
            problem,
            num_particles=5000,
            kde_normalization="all",
            boundary_conditions=boundary_conditions,
        )
        hjb_solver_1 = HJBFDMSolver(problem)
        mfg_solver_1 = FixedPointIterator(problem, hjb_solver=hjb_solver_1, fp_solver=fp_solver_1)

        # Issue #1567: no try/except -> skip; a solver raise must fail this cross-solver test.
        result_1 = mfg_solver_1.solve(max_iterations=50, tolerance=1e-3)

        # Solver 2: FP Particle + HJB GFDM
        fp_solver_2 = FPParticleSolver(
            problem,
            num_particles=5000,
            kde_normalization="all",
            boundary_conditions=boundary_conditions,
        )
        collocation_points = problem.geometry.get_spatial_grid().reshape(-1, 1)
        hjb_solver_2 = HJBGFDMSolver(problem, collocation_points=collocation_points, delta=0.3)
        mfg_solver_2 = FixedPointIterator(problem, hjb_solver=hjb_solver_2, fp_solver=fp_solver_2)

        # Issue #1567: no try/except -> skip; a solver raise must fail this cross-solver test.
        result_2 = mfg_solver_2.solve(max_iterations=50, tolerance=1e-3)

        grid = problem.geometry
        Nt_points = problem.Nt + 1
        masses_fdm = []
        masses_gfdm = []

        for t_idx in range(Nt_points):
            mass_fdm = compute_total_mass(result_1.M[t_idx, :], grid)
            mass_gfdm = compute_total_mass(result_2.M[t_idx, :], grid)
            masses_fdm.append(mass_fdm)
            masses_gfdm.append(mass_gfdm)

        masses_fdm = np.array(masses_fdm)
        masses_gfdm = np.array(masses_gfdm)

        initial_fdm = masses_fdm[0]
        initial_gfdm = masses_gfdm[0]

        change_fdm = np.abs(masses_fdm - initial_fdm)
        change_gfdm = np.abs(masses_gfdm - initial_gfdm)

        print("\n=== Mass Conservation Comparison ===")
        print(f"FDM  - Initial: {initial_fdm:.6f}, Final: {masses_fdm[-1]:.6f}, Max change: {np.max(change_fdm):.6e}")
        print(
            f"GFDM - Initial: {initial_gfdm:.6f}, Final: {masses_gfdm[-1]:.6f}, Max change: {np.max(change_gfdm):.6e}"
        )

        assert np.max(change_fdm) < 0.2 * initial_fdm, f"FDM mass change too large: {np.max(change_fdm):.6e}"
        assert np.max(change_gfdm) < 0.2 * initial_gfdm, f"GFDM mass change too large: {np.max(change_gfdm):.6e}"

    @pytest.mark.slow
    @pytest.mark.parametrize("num_particles", [1000, 3000, 5000])
    def test_mass_conservation_particle_count(self, problem, boundary_conditions, num_particles):
        """
        Test that mass conservation quality with different particle counts.

        Args:
            num_particles: Number of particles to use
        """
        fp_solver = FPParticleSolver(
            problem,
            num_particles=num_particles,
            # NOT "all" (#2181): ALL pins every slice to the caller's mass on the grid
            # measure -- the same functional `compute_total_mass` uses below -- so the drift
            # this test asserts would be 2.220446e-16 against a bound of 0.1 and could not
            # fail. Measured. NONE carries one factor calibrated on the first slice, so the
            # reconstruction's own drift stays in the number. NOTE it is NOT a measure of particle
            # loss -- the KDE is blind to that (#2188); it is the reconstruction's drift.
            kde_normalization="none",
            boundary_conditions=boundary_conditions,
        )

        hjb_solver = HJBFDMSolver(problem)
        mfg_solver = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            use_anderson=True,
            anderson_depth=5,
            backend=None,
        )

        # Issue #1567: no try/except -> skip; a solver raise must fail this mass-conservation test.
        result = mfg_solver.solve(max_iterations=50, tolerance=1e-4, return_structured=True)

        grid = problem.geometry
        Nt_points = problem.Nt + 1
        masses = np.array([compute_total_mass(result.M[t, :], grid) for t in range(Nt_points)])
        # #2181: drift from the INITIAL mass, not deviation from 1. These two lines read
        # `max|masses - 1.0|` until the particle solver stopped returning mass 1 whatever went in --
        # a particle method carries no mass, so the KDE reconstruction came back at 1 regardless and
        # this assertion was pinning the representation's loss. This fixture's density integrates to
        # 0.546 on the grid measure, so the old form now reads about 0.454 and fails a correct
        # solver. A ratio is what "conservation" means and is invariant to the caller's choice of
        # initial mass, which #1887 made the caller's to make.
        max_error = float(np.max(np.abs(masses / masses[0] - 1.0)))

        print(f"\nParticles: {num_particles}, initial mass {masses[0]:.6f}, max drift: {max_error:.6e}")

        assert max_error < 0.1, f"Mass drift too large with {num_particles} particles: {max_error:.6e}"

    @pytest.mark.slow
    def test_mass_conservation_different_initial_conditions(self, boundary_conditions):
        """
        Test mass conservation with different initial density distributions.
        """
        test_cases = [
            ("Gaussian left", 0.25, 0.1),
            ("Gaussian center", 0.5, 0.15),
            ("Gaussian right", 0.75, 0.1),
        ]

        for name, center_frac, std_frac in test_cases:
            L = 2.0

            def make_custom_components(cfrac, sfrac, domain_L):
                center = domain_L * cfrac
                std = domain_L * sfrac

                def m_initial(x):
                    density = np.exp(-((x - center) ** 2) / (2 * std**2))
                    return density

                return MFGComponents(m_initial=m_initial, u_terminal=lambda x: 0.0, hamiltonian=_default_hamiltonian())

            geometry = TensorProductGrid(bounds=[(0.0, L)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
            problem = MFGProblem(
                geometry=geometry,
                T=1.0,
                Nt=20,
                sigma=0.1,
                coupling_coefficient=1.0,
                components=make_custom_components(center_frac, std_frac, L),
            )

            fp_solver = FPParticleSolver(
                problem,
                num_particles=5000,
                # NOT "all" (#2181): ALL pins every slice to the caller's mass on the grid
                # measure -- the same functional `compute_total_mass` uses below -- so the drift
                # this test asserts would be 2.220446e-16 against a bound of 0.1 and could not
                # fail. Measured. NONE carries one factor calibrated on the first slice, so the
                # reconstruction's own drift stays in the number. NOTE it is NOT a measure of
                # particle loss -- the KDE is blind to that (#2188); it is reconstruction drift.
                kde_normalization="none",
                boundary_conditions=boundary_conditions,
            )
            hjb_solver = HJBFDMSolver(problem)
            mfg_solver = FixedPointIterator(
                problem,
                hjb_solver=hjb_solver,
                fp_solver=fp_solver,
                use_anderson=True,
                anderson_depth=5,
                backend=None,
            )

            # Issue #1567: no try/except -> skip; a solver raise must fail this mass-conservation test.
            result = mfg_solver.solve(max_iterations=50, tolerance=1e-4, return_structured=True)

            grid = problem.geometry
            Nt_points = problem.Nt + 1
            masses = np.array([compute_total_mass(result.M[t, :], grid) for t in range(Nt_points)])
            # See the note above: drift from the initial mass, not deviation from 1 (#2181).
            max_error = float(np.max(np.abs(masses / masses[0] - 1.0)))

            print(f"\n{name}: Initial mass = {masses[0]:.6f}, Max drift = {max_error:.6e}")

            assert max_error < 0.1, f"Mass conservation failed for {name}: {max_error:.6e}"

    def test_coupled_fdm_mass_conservation_fast_tier(self):
        """Fast-tier (non-@slow) coupled MFG mass-conservation gate (Issue #1567).

        Every other coupled mass test in this class is @slow (5000-particle stochastic KDE),
        so the only mass check the PR gate ran was the analytic pure-diffusion one -- a coupled
        regression that leaked mass would surface only on the nightly tier. This runs a small,
        deterministic coupled FDM solve (HJB-FDM + FP-FDM, no-flux, n=21, Nt=8, 3 Picard steps,
        ~1s) and asserts total mass is preserved across every time step. Mass conservation here
        is structural (the column-conservative no-flux FP stencil, #1184), so it holds regardless
        of Picard convergence -- a real leak (a broken no-flux stencil, a lost normalization)
        trips it, but non-convergence does not."""
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver

        n, nt, T, sigma = 21, 8, 0.5, 0.3
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
        H = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: np.asarray(m),
            coupling_dm=lambda m: np.ones_like(np.asarray(m)),
        )
        comps = MFGComponents(
            m_initial=lambda x: np.exp(-30 * (np.asarray(x) - 0.5) ** 2),
            u_terminal=lambda x: 0.0 * np.asarray(x),
            hamiltonian=H,
        )
        prob = MFGProblem(geometry=grid, T=T, Nt=nt, sigma=sigma, coupling_coefficient=1.0, components=comps)
        mfg_solver = FixedPointIterator(prob, hjb_solver=HJBFDMSolver(prob), fp_solver=FPFDMSolver(prob))
        result = mfg_solver.solve(max_iterations=3, tolerance=1e-4)

        grid = prob.geometry
        masses = np.array([compute_total_mass(result.M[t, :], grid) for t in range(nt + 1)])
        drift = np.max(np.abs(masses - masses[0])) / masses[0]
        assert drift < 1e-10, f"coupled FDM mass drift {drift:.2e} across time (no-flux must conserve)"


class TestExplicitDriftNoFluxDiffusionConservation:
    """Issue #1184: the explicit-drift FP path's implicit DIFFUSION sub-step must conserve
    mass at no-flux walls. The diffusion matrix is `LaplacianOperator.as_scipy_sparse()`; with
    the column-conservative (mass_conservative=True) no-flux stencil, pure diffusion of a
    wall-touching density conserves mass to machine precision (was ~0.84% leak from the
    column-sum defect). The advection sub-step's non-conservation under drift is a separate
    follow-up (#1184 step 4)."""

    def test_pure_diffusion_no_flux_conserves_mass(self):
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver

        n, nt, T, sigma = 51, 50, 0.5, 0.3
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
        H = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: 0.0 * np.asarray(m),
            coupling_dm=lambda m: 0.0 * np.asarray(m),
        )
        x = np.linspace(0.0, 1.0, n)
        comps = MFGComponents(
            m_initial=lambda xx: np.exp(-30 * (np.asarray(xx) - 0.5) ** 2),
            u_terminal=lambda xx: 0.0 * np.asarray(xx),
            hamiltonian=H,
        )
        prob = MFGProblem(geometry=grid, T=T, Nt=nt, sigma=sigma, components=comps)
        solver = FPFDMSolver(prob)
        m0 = np.exp(-30 * (x - 0.5) ** 2)
        m0 /= compute_total_mass(m0, grid)
        # callable (zero) drift routes through the explicit-drift path; pure diffusion
        M = solver.solve_fp_system(m0.copy(), drift_field=lambda t, g, m: np.zeros(n), volatility_field=sigma)
        # The drift is zero, so the FV advection kernel contributes nothing and only the node-based
        # diffusion runs. That is why this stays at machine precision on the grid measure where the
        # DRIFTED case in test_fdm_centered_conservation records 2.388e-05 (#2145/#1184).
        mass = np.asarray(grid.integrate(M), dtype=float)
        rel = abs(mass[-1] - mass[0]) / mass[0]
        assert rel < 1e-12, f"explicit-drift pure-diffusion no-flux mass leak {rel:.2e} (was ~0.84%)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

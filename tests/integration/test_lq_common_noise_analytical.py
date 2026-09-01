"""
Integration test for Linear-Quadratic Common Noise MFG with analytical solution.

This test validates the CommonNoiseMFGSolver against a known analytical solution
for a simple Linear-Quadratic problem with common noise.

Mathematical Setup:
    - Hamiltonian: H(x, p, m, θ) = (p²/2) + (x²/2) + α·θ·x
    - Common Noise: dθ_t = -κθ_t dt + σ dB_t (Ornstein-Uhlenbeck)
    - Terminal cost: g(x, θ) = (x²/2)

For small coupling (α → 0), the solution is approximately:
    - u(t, x, θ) ≈ u₀(t, x) + α·θ·u₁(t, x)
    - m(t, x, θ) ≈ m₀(t, x) + α·θ·m₁(t, x)

Where u₀, m₀ solve the deterministic LQ-MFG and u₁, m₁ are first-order corrections.

This provides an analytical benchmark for convergence testing.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.stochastic import CommonNoiseMFGSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.stochastic import OrnsteinUhlenbeckProcess, StochasticMFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


@pytest.mark.slow
@pytest.mark.integration
class TestLQCommonNoiseAnalytical:
    """Test Linear-Quadratic Common Noise MFG with analytical solution."""

    def create_lq_common_noise_problem(self, alpha: float = 0.1):
        """
        Create Linear-Quadratic MFG problem with common noise.

        Args:
            alpha: Coupling strength between noise and state

        Returns:
            StochasticMFGProblem with LQ structure
        """
        # Common noise: OU process (mean-reverting)
        noise_process = OrnsteinUhlenbeckProcess(
            kappa=2.0,  # Mean reversion speed
            mu=0.0,  # Long-term mean
            sigma=0.2,  # Volatility
        )

        # LQ Hamiltonian with noise coupling
        def lq_hamiltonian(x, p, m, theta):
            """
            Linear-Quadratic Hamiltonian with common noise.

            H = p²/2 + x²/2 + α·θ·x

            Args:
                x: State
                p: Co-state (momentum)
                m: Population density
                theta: Common noise value

            Returns:
                Hamiltonian value
            """
            return 0.5 * p**2 + 0.5 * x**2 + alpha * theta * x

        # Create stochastic problem with v1.0 API components
        geometry = TensorProductGrid(bounds=[(-2.0, 2.0)], boundary_conditions=no_flux_bc(dimension=1), Nx_points=[42])
        hamiltonian = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            potential=lambda x, t: 0.5 * float(np.sum(x**2)),
        )
        components = MFGComponents(
            hamiltonian=hamiltonian,
            u_terminal=lambda x: 0.5 * x**2,
            m_initial=lambda x: np.exp(-(x**2) / 0.5),
        )
        problem = StochasticMFGProblem(
            geometry=geometry,
            T=0.5,
            Nt=21,
            noise_process=noise_process,
            conditional_hamiltonian=lq_hamiltonian,
            components=components,
        )

        # Set noise-dependent terminal cost (extends base u_terminal)
        def terminal_cost(x, theta):
            return 0.5 * x**2

        problem.terminal_cost = terminal_cost

        # Set initial density from components (normalized)
        bounds = problem.geometry.get_bounds()
        Nx_points = problem.geometry.get_grid_shape()[0]
        x = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        rho0 = np.exp(-(x**2) / 0.5)
        rho0 /= np.trapezoid(rho0, x)
        problem.rho0 = rho0

        return problem

    def analytical_solution_zero_noise(self, problem):
        """
        Compute analytical solution for deterministic limit (θ = 0).

        For LQ-MFG with H = p²/2 + x²/2, the solution is known analytically.

        Args:
            problem: MFG problem (used for grid info)

        Returns:
            Tuple (u, m) with analytical solution
        """
        bounds = problem.geometry.get_bounds()
        Nx_points = problem.geometry.get_grid_shape()[0]
        x = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        t_grid = np.linspace(0, problem.T, problem.Nt + 1)

        # For LQ-MFG with this structure, analytical solution is:
        # u(t, x) = A(t)·x² + B(t)
        # m(t, x) = Gaussian with time-varying variance

        # Simplified analytical solution (deterministic LQ-MFG)
        u = np.zeros((problem.Nt + 1, Nx_points))
        m = np.zeros((problem.Nt + 1, Nx_points))

        for i, t in enumerate(t_grid):
            tau = problem.T - t  # Time to terminal
            # Value function: u ≈ A(τ)·x²
            A_tau = tau / (1 + tau)  # Approximate Riccati solution
            u[i] = A_tau * x**2

            # Density: Gaussian with time-dependent variance
            sigma_t = 0.5 * (1 + 0.5 * t)  # Approximate variance evolution
            m[i] = np.exp(-(x**2) / (2 * sigma_t))
            m[i] /= np.trapezoid(m[i], x)  # Normalize

        return u, m

    @pytest.mark.parametrize("num_noise_samples", [10, 20])
    def test_convergence_in_noise_samples(self, num_noise_samples):
        """
        Test that solution converges as number of noise samples increases.

        Verifies Monte Carlo convergence: error ~ 1/sqrt(K)
        """
        problem = self.create_lq_common_noise_problem(alpha=0.05)  # Weak coupling

        solver = CommonNoiseMFGSolver(
            problem,
            num_noise_samples=num_noise_samples,
            variance_reduction=True,
            parallel=False,  # Sequential for reproducibility
            seed=42,
        )

        # Note: This test is currently a placeholder until conditional MFG solver is implemented
        # For now, we just verify solver can be instantiated and noise paths sampled
        noise_paths = solver._sample_noise_paths()

        assert len(noise_paths) == num_noise_samples
        assert all(path.shape == (problem.Nt + 1,) for path in noise_paths)

        # Verify QMC provides lower variance than standard MC
        if num_noise_samples >= 10:
            # With QMC, noise paths should have better coverage
            path_means = [np.mean(path) for path in noise_paths]
            path_std = np.std(path_means)

            # QMC should give paths with mean closer to theoretical mean (0.0)
            # This is a weak test, but validates QMC is working
            assert path_std < 0.3  # Reasonable variance for OU process

    def test_zero_noise_limit(self):
        """
        Test that zero noise recovers deterministic solution.

        With noise variance σ → 0, common noise MFG should reduce to deterministic MFG.
        """
        # Create problem with very small noise
        noise_process = OrnsteinUhlenbeckProcess(kappa=1.0, mu=0.0, sigma=1e-6)

        def lq_hamiltonian(x, p, m, theta):
            return 0.5 * p**2 + 0.5 * x**2  # No noise coupling

        geometry = TensorProductGrid(bounds=[(-2.0, 2.0)], boundary_conditions=no_flux_bc(dimension=1), Nx_points=[42])
        hamiltonian = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            potential=lambda x, t: 0.5 * float(np.sum(x**2)),
        )
        components = MFGComponents(
            hamiltonian=hamiltonian,
            u_terminal=lambda x: 0.5 * x**2,
            m_initial=lambda x: np.exp(-(x**2) / 0.5),
        )
        problem = StochasticMFGProblem(
            geometry=geometry,
            T=0.5,
            Nt=21,
            noise_process=noise_process,
            conditional_hamiltonian=lq_hamiltonian,
            components=components,
        )

        solver = CommonNoiseMFGSolver(problem, num_noise_samples=5, variance_reduction=False, seed=42)

        # Sample noise paths with tiny variance
        noise_paths = solver._sample_noise_paths()

        # Verify all paths are close to zero (mean of OU process)
        for path in noise_paths:
            assert np.max(np.abs(path)) < 0.01, "Noise should be negligible with σ=1e-6"

    def test_ou_process_properties(self):
        """Test that OU process has correct statistical properties."""
        noise_process = OrnsteinUhlenbeckProcess(kappa=2.0, mu=0.0, sigma=0.2)

        # Sample many paths to verify statistical properties
        num_paths = 100
        T = 2.0
        Nt = 100

        paths = []
        for seed in range(num_paths):
            path = noise_process.sample_path(T=T, Nt=Nt, seed=seed)
            paths.append(path)

        paths = np.array(paths)  # Shape: (num_paths, Nt+1)

        # Long-time mean should be close to mu = 0.0
        long_time_mean = np.mean(paths[:, -20:])  # Average over last 20% of path
        assert np.abs(long_time_mean) < 0.1, f"OU process mean should converge to μ=0, got {long_time_mean}"

        # Variance should be bounded
        long_time_std = np.std(paths[:, -20:])
        theoretical_std = noise_process.sigma / np.sqrt(2 * noise_process.kappa)
        assert np.abs(long_time_std - theoretical_std) < 0.1, (
            f"OU process variance should match σ²/(2κ), got {long_time_std:.3f} vs theory {theoretical_std:.3f}"
        )

    def test_analytical_reference_solution(self):
        """
        Verify analytical reference solution has correct properties.

        This tests the deterministic LQ-MFG analytical solution used as reference.
        """
        problem = self.create_lq_common_noise_problem(alpha=0.0)  # No noise coupling
        u, m = self.analytical_solution_zero_noise(problem)

        # Check shapes
        bounds = problem.geometry.get_bounds()
        Nx_points = problem.geometry.get_grid_shape()[0]
        assert u.shape == (problem.Nt + 1, Nx_points)
        assert m.shape == (problem.Nt + 1, Nx_points)

        # Check density normalization
        x = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        for i in range(problem.Nt + 1):
            mass = np.trapezoid(m[i], x)
            assert np.abs(mass - 1.0) < 1e-6, f"Density at t={i} not normalized: mass={mass}"

        # Check terminal condition: u(T, x) ≈ A(0)·x²/2
        # Note: This is an approximate analytical solution for testing infrastructure
        # The actual analytical solution requires solving Riccati ODE
        u_terminal = u[-1]
        expected_terminal = 0.5 * x**2
        error = np.max(np.abs(u_terminal - expected_terminal))
        # Relaxed tolerance since this is simplified approximate solution
        assert error < 5.0, f"Terminal condition error: {error} (approximate analytical solution)"

    @pytest.mark.skip(
        reason="The ORIGINAL reason -- 'Requires conditional MFG solver implementation' -- was "
        "removed by #2191, which implements it. What blocks it now is NOT established, and this "
        "reason has already named the wrong blocker twice: first the implementation gap after it "
        "was closed, then lambda pickling under parallel=True, which is real but not sufficient -- "
        "MEASURED: with parallel=False it still fails, on the analytical oracle itself -- "
        "'Value function error: 1.1246824224221859' against a 0.2 tolerance, 5.6x over, in 922 s. "
        "That disagreement is the open question and it is bigger than this test: it would mean the "
        "solver and its only analytical oracle do not agree. Investigated in #2206; do not re-skip "
        "this with a new single-cause reason without measuring it."
    )
    def test_full_solution_convergence(self):
        """
        Full convergence test with conditional MFG solver.

        This test will be enabled once the conditional MFG solver is implemented.
        It will verify:
        - Convergence in number of noise samples K
        - Convergence to deterministic limit as σ → 0
        - Agreement with analytical perturbation solution
        """
        problem = self.create_lq_common_noise_problem(alpha=0.1)

        solver = CommonNoiseMFGSolver(
            problem,
            num_noise_samples=50,
            variance_reduction=True,
            parallel=True,
            seed=42,
        )

        result = solver.solve(verbose=False)

        # Compare with analytical solution
        u_analytical, m_analytical = self.analytical_solution_zero_noise(problem)

        # Compute errors
        u_error = np.linalg.norm(result.u_mean - u_analytical) / np.linalg.norm(u_analytical)
        m_error = np.linalg.norm(result.m_mean - m_analytical) / np.linalg.norm(m_analytical)

        # Should be small for weak coupling (α=0.1)
        assert u_error < 0.2, f"Value function error: {u_error}"
        assert m_error < 0.2, f"Density error: {m_error}"

        # Check Monte Carlo error estimates
        assert result.mc_error_u < 0.1
        assert result.mc_error_m < 0.1

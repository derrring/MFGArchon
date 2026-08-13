"""
Unit tests for CommonNoiseMFGSolver.

Tests individual methods and components of the Common Noise MFG solver,
focusing on coverage of internal logic, edge cases, and error handling.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.stochastic import CommonNoiseMFGResult, CommonNoiseMFGSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.stochastic import OrnsteinUhlenbeckProcess, StochasticMFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.numerical.particle.sampling import MCConfig


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


class TestCommonNoiseMFGResultDataclass:
    """Test CommonNoiseMFGResult dataclass functionality."""

    def test_result_initialization(self):
        """Test result dataclass can be initialized."""
        u_mean = np.random.rand(11, 21)
        m_mean = np.random.rand(11, 21)
        u_std = np.random.rand(11, 21)
        m_std = np.random.rand(11, 21)

        result = CommonNoiseMFGResult(
            u_mean=u_mean,
            m_mean=m_mean,
            u_std=u_std,
            m_std=m_std,
            u_samples=[u_mean],
            m_samples=[m_mean],
            noise_paths=[np.zeros(11)],
            num_noise_samples=1,
            mc_error_u=0.01,
            mc_error_m=0.01,
        )

        assert result.num_noise_samples == 1
        assert result.mc_error_u == 0.01
        assert result.mc_error_m == 0.01
        assert result.variance_reduction_factor == 1.0  # Default
        assert result.computation_time == 0.0  # Default
        assert not result.converged  # Default

    def test_confidence_interval_u(self):
        """Test confidence interval computation for u."""
        u_mean = np.ones((11, 21))
        u_std = 0.1 * np.ones((11, 21))

        result = CommonNoiseMFGResult(
            u_mean=u_mean,
            m_mean=u_mean.copy(),
            u_std=u_std,
            m_std=u_std.copy(),
            u_samples=[u_mean],
            m_samples=[u_mean],
            noise_paths=[np.zeros(11)],
            num_noise_samples=100,
            mc_error_u=0.01,
            mc_error_m=0.01,
        )

        from scipy import stats

        lower, upper = result.get_confidence_interval_u(confidence=0.95)

        assert lower.shape == u_mean.shape
        assert upper.shape == u_mean.shape
        assert np.all(lower < u_mean)
        assert np.all(upper > u_mean)
        assert np.all(upper - lower > 0)

        # The bracket above admits any half-width whatever. Pin the closed form: it fixes the
        # z-quantile, the 1/sqrt(K) Monte Carlo scaling, and the symmetry of the interval, none of
        # which anything else in this file asserts. Measured half-width 0.019599639845400540
        # against 0.019599639845400585 expected -- relative deviation 2.3e-15, so rtol=1e-12
        # leaves ~430x margin.
        half = stats.norm.ppf(0.975) * 0.1 / np.sqrt(100)
        np.testing.assert_allclose(u_mean - lower, half, rtol=1e-12)
        np.testing.assert_allclose(upper - u_mean, half, rtol=1e-12)

    def test_confidence_interval_m(self):
        """Test confidence interval computation for m."""
        m_mean = np.ones((11, 21))
        m_std = 0.1 * np.ones((11, 21))

        result = CommonNoiseMFGResult(
            u_mean=m_mean.copy(),
            m_mean=m_mean,
            u_std=m_std.copy(),
            m_std=m_std,
            u_samples=[m_mean],
            m_samples=[m_mean],
            noise_paths=[np.zeros(11)],
            num_noise_samples=100,
            mc_error_u=0.01,
            mc_error_m=0.01,
        )

        from scipy import stats

        lower, upper = result.get_confidence_interval_m(confidence=0.95)

        assert lower.shape == m_mean.shape
        assert upper.shape == m_mean.shape
        assert np.all(lower < m_mean)
        assert np.all(upper > m_mean)

        # get_confidence_interval_m is a hand-copied duplicate of get_confidence_interval_u, so the
        # sibling test does not cover it and the two halves can drift apart independently. Same
        # closed form, measured to the same 2.3e-15 relative deviation.
        half = stats.norm.ppf(0.975) * 0.1 / np.sqrt(100)
        np.testing.assert_allclose(m_mean - lower, half, rtol=1e-12)
        np.testing.assert_allclose(upper - m_mean, half, rtol=1e-12)

    def test_confidence_interval_different_levels(self):
        """Test confidence intervals with different confidence levels."""
        u_mean = np.ones((11, 21))
        u_std = 0.1 * np.ones((11, 21))

        result = CommonNoiseMFGResult(
            u_mean=u_mean,
            m_mean=u_mean.copy(),
            u_std=u_std,
            m_std=u_std.copy(),
            u_samples=[u_mean],
            m_samples=[u_mean],
            noise_paths=[np.zeros(11)],
            num_noise_samples=100,
            mc_error_u=0.01,
            mc_error_m=0.01,
        )

        # 95% CI should be wider than 90% CI
        lower_95, upper_95 = result.get_confidence_interval_u(confidence=0.95)
        lower_90, upper_90 = result.get_confidence_interval_u(confidence=0.90)

        width_95 = np.mean(upper_95 - lower_95)
        width_90 = np.mean(upper_90 - lower_90)

        assert width_95 > width_90


class TestCommonNoiseSolverInitialization:
    """Test CommonNoiseMFGSolver initialization."""

    def _create_simple_problem(self):
        """Helper to create simple stochastic problem."""
        noise_process = OrnsteinUhlenbeckProcess(kappa=1.0, mu=0.0, sigma=0.1)

        def simple_hamiltonian(x, p, m, theta):
            return 0.5 * p**2 + 0.1 * m

        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[22], boundary_conditions=no_flux_bc(dimension=1))
        return StochasticMFGProblem(
            geometry=geometry,
            T=0.5,
            Nt=11,
            noise_process=noise_process,
            conditional_hamiltonian=simple_hamiltonian,
            components=_default_components(),
        )

    def test_basic_initialization(self):
        """Test solver initializes with default parameters."""
        problem = self._create_simple_problem()
        solver = CommonNoiseMFGSolver(problem, num_noise_samples=10)

        assert solver.K == 10
        assert solver.variance_reduction is True  # Default
        assert solver.parallel is True  # Default
        assert solver.seed is None
        assert solver.mc_config is not None

    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        problem = self._create_simple_problem()

        solver = CommonNoiseMFGSolver(
            problem,
            num_noise_samples=50,
            variance_reduction=False,
            parallel=False,
            num_workers=4,
            seed=42,
        )

        assert solver.K == 50
        assert solver.variance_reduction is False
        assert solver.parallel is False
        assert solver.num_workers == 4
        assert solver.seed == 42

    def test_initialization_with_mc_config(self):
        """Test initialization with custom MCConfig."""
        problem = self._create_simple_problem()

        mc_config = MCConfig(
            num_samples=20,
            sampling_method="halton",
            use_control_variates=True,
            seed=123,
        )

        solver = CommonNoiseMFGSolver(problem, num_noise_samples=20, mc_config=mc_config)

        assert solver.mc_config.sampling_method == "halton"
        assert solver.mc_config.use_control_variates is True
        assert solver.mc_config.seed == 123

    def test_initialization_with_custom_solver_factory(self):
        """Test initialization with custom conditional solver factory."""
        problem = self._create_simple_problem()

        def custom_factory(prob):
            # Use the new problem.solve() API
            return prob.solve(verbose=False)

        solver = CommonNoiseMFGSolver(problem, num_noise_samples=10, conditional_solver_factory=custom_factory)

        # Identity, not callability: a constructor that ignored the kwarg and installed the default
        # lambda passes `is not None` and `callable`. custom_factory is written as a copy of the
        # default body, so even a value comparison would not discriminate -- identity is the only
        # assertion that separates the non-default branch from the default one.
        assert solver.conditional_solver_factory is custom_factory

    def test_raises_error_for_non_stochastic_problem(self):
        """Test that solver raises error for problem without common noise."""
        from mfgarchon.core import MFGProblem

        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[22], boundary_conditions=no_flux_bc(dimension=1)
        )  # Nx=21 intervals -> 22 points
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=11, components=_default_components())

        with pytest.raises(ValueError, match="must have common noise"):
            CommonNoiseMFGSolver(problem, num_noise_samples=10)


class TestCommonNoiseSolverNoiseSampling:
    """Test noise path sampling methods."""

    def _create_simple_problem(self):
        """Helper to create simple stochastic problem."""
        noise_process = OrnsteinUhlenbeckProcess(kappa=1.0, mu=0.0, sigma=0.1)

        def simple_hamiltonian(x, p, m, theta):
            return 0.5 * p**2

        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[22], boundary_conditions=no_flux_bc(dimension=1))
        return StochasticMFGProblem(
            geometry=geometry,
            T=0.5,
            Nt=11,
            noise_process=noise_process,
            conditional_hamiltonian=simple_hamiltonian,
            components=_default_components(),
        )

    def test_sample_noise_paths_standard_mc(self):
        """Test standard Monte Carlo noise path sampling."""
        problem = self._create_simple_problem()
        solver = CommonNoiseMFGSolver(problem, num_noise_samples=20, variance_reduction=False, seed=42)

        paths = solver._sample_noise_paths()

        assert len(paths) == 20
        assert all(path.shape == (12,) for path in paths)  # Nt+1 = 12
        assert all(isinstance(path, np.ndarray) for path in paths)

        # This branch owns a seeding contract nothing else pins: it seeds the GLOBAL np.random
        # (common_noise_solver.py:349), so reproducibility here doubles as the pin on that side
        # effect. Measured bitwise identical across two constructions at seed=42.
        repeat = CommonNoiseMFGSolver(problem, num_noise_samples=20, variance_reduction=False, seed=42)
        for a, b in zip(paths, repeat._sample_noise_paths(), strict=True):
            np.testing.assert_array_equal(a, b)

        # The two arms of the variance_reduction dispatch are otherwise indistinguishable: the
        # assertions above hold identically for the QMC twin. Measured at the same seed: all 20
        # paths differ.
        qmc = CommonNoiseMFGSolver(problem, num_noise_samples=20, variance_reduction=True, seed=42)
        assert not all(np.array_equal(a, b) for a, b in zip(paths, qmc._sample_noise_paths(), strict=True)), (
            "variance_reduction=False produced the same paths as the QMC branch"
        )

    def test_sample_noise_paths_quasi_mc(self):
        """Test quasi-Monte Carlo noise path sampling with variance reduction."""
        problem = self._create_simple_problem()
        solver = CommonNoiseMFGSolver(problem, num_noise_samples=20, variance_reduction=True, seed=42)

        paths = solver._sample_noise_paths()

        assert len(paths) == 20
        assert all(path.shape == (12,) for path in paths)

        # As written this test was indistinguishable from its standard-MC sibling. Require that
        # variance_reduction is actually live: the MC twin at the SAME seed must not reproduce
        # these paths. Measured at seed=42: all 20 differ.
        mc = CommonNoiseMFGSolver(problem, num_noise_samples=20, variance_reduction=False, seed=42)
        assert not all(np.array_equal(a, b) for a, b in zip(paths, mc._sample_noise_paths(), strict=True)), (
            "variance_reduction=True did not change the sampler"
        )

        # OU initial condition: every path starts at theta_initial. Measured exactly 0.0 for all 20.
        assert all(path[0] == 0.0 for path in paths)

    def test_noise_paths_reproducible_with_seed(self):
        """Test that noise paths are reproducible with fixed seed."""
        problem = self._create_simple_problem()

        solver1 = CommonNoiseMFGSolver(problem, num_noise_samples=10, variance_reduction=False, seed=42)
        paths1 = solver1._sample_noise_paths()

        solver2 = CommonNoiseMFGSolver(problem, num_noise_samples=10, variance_reduction=False, seed=42)
        paths2 = solver2._sample_noise_paths()

        # Paths should be identical with same seed
        for p1, p2 in zip(paths1, paths2, strict=False):
            np.testing.assert_array_almost_equal(p1, p2)

    def test_noise_paths_different_without_seed(self):
        """Test that noise paths differ when no seed specified."""
        problem = self._create_simple_problem()

        solver1 = CommonNoiseMFGSolver(problem, num_noise_samples=10, variance_reduction=False, seed=None)
        paths1 = solver1._sample_noise_paths()

        solver2 = CommonNoiseMFGSolver(problem, num_noise_samples=10, variance_reduction=False, seed=None)
        paths2 = solver2._sample_noise_paths()

        # At least some paths should differ (very high probability)
        differences = [not np.array_equal(p1, p2) for p1, p2 in zip(paths1, paths2, strict=False)]
        assert any(differences)


class TestCommonNoiseSolverAggregation:
    """Test solution aggregation methods."""

    def test_aggregate_solutions_basic(self):
        """Test basic solution aggregation."""
        from mfgarchon.alg.numerical.stochastic import CommonNoiseMFGSolver

        # Create mock conditional solutions
        u1 = np.ones((12, 21))
        m1 = np.ones((12, 21))
        u2 = 2.0 * np.ones((12, 21))
        m2 = 2.0 * np.ones((12, 21))

        conditional_solutions = [(u1, m1, True), (u2, m2, True)]
        noise_paths = [np.zeros(12), np.zeros(12)]

        # Create a dummy solver just to access the method
        noise_process = OrnsteinUhlenbeckProcess(kappa=1.0, mu=0.0, sigma=0.1)
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[22], boundary_conditions=no_flux_bc(dimension=1))
        problem = StochasticMFGProblem(
            geometry=geometry,
            T=0.5,
            Nt=11,
            noise_process=noise_process,
            conditional_hamiltonian=lambda x, p, m, theta: 0.5 * p**2,
            components=_default_components(),
        )
        solver = CommonNoiseMFGSolver(problem, num_noise_samples=2)

        result = solver._aggregate_solutions(conditional_solutions, noise_paths)

        # Check means
        np.testing.assert_array_almost_equal(result.u_mean, 1.5 * np.ones((12, 21)))
        np.testing.assert_array_almost_equal(result.m_mean, 1.5 * np.ones((12, 21)))

        # Check standard deviations (with ddof=1, std = sqrt((1/1) * sum((x - mean)^2)))
        # For [1, 2], mean=1.5, std = sqrt((0.5^2 + 0.5^2)/1) = sqrt(0.5) ≈ 0.707
        expected_std = np.sqrt(0.5) * np.ones((12, 21))
        np.testing.assert_array_almost_equal(result.u_std, expected_std, decimal=10)
        np.testing.assert_array_almost_equal(result.m_std, expected_std, decimal=10)

        # Check convergence
        assert result.converged is True

    def test_aggregate_solutions_with_non_convergence(self):
        """Test aggregation when some solvers don't converge."""
        from mfgarchon.alg.numerical.stochastic import CommonNoiseMFGSolver

        u1 = np.ones((12, 21))
        m1 = np.ones((12, 21))

        conditional_solutions = [(u1, m1, True), (u1, m1, False), (u1, m1, True)]
        noise_paths = [np.zeros(12)] * 3

        noise_process = OrnsteinUhlenbeckProcess(kappa=1.0, mu=0.0, sigma=0.1)
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[22], boundary_conditions=no_flux_bc(dimension=1))
        problem = StochasticMFGProblem(
            geometry=geometry,
            T=0.5,
            Nt=11,
            noise_process=noise_process,
            conditional_hamiltonian=lambda x, p, m, theta: 0.5 * p**2,
            components=_default_components(),
        )
        solver = CommonNoiseMFGSolver(problem, num_noise_samples=3)

        result = solver._aggregate_solutions(conditional_solutions, noise_paths)

        # Should report non-convergence if any failed
        assert result.converged is False

    def test_aggregate_solutions_mc_error_estimate(self):
        """Test Monte Carlo error estimation in aggregation."""
        from mfgarchon.alg.numerical.stochastic import CommonNoiseMFGSolver

        # Create solutions with known variance
        K = 100
        u_samples = [np.random.rand(12, 21) for _ in range(K)]
        m_samples = [np.random.rand(12, 21) for _ in range(K)]

        conditional_solutions = [(u, m, True) for u, m in zip(u_samples, m_samples, strict=False)]
        noise_paths = [np.zeros(12)] * K

        noise_process = OrnsteinUhlenbeckProcess(kappa=1.0, mu=0.0, sigma=0.1)
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[22], boundary_conditions=no_flux_bc(dimension=1))
        problem = StochasticMFGProblem(
            geometry=geometry,
            T=0.5,
            Nt=11,
            noise_process=noise_process,
            conditional_hamiltonian=lambda x, p, m, theta: 0.5 * p**2,
            components=_default_components(),
        )
        solver = CommonNoiseMFGSolver(problem, num_noise_samples=K)

        result = solver._aggregate_solutions(conditional_solutions, noise_paths)

        # MC error should decrease with sqrt(K)
        # Error = std / sqrt(K)
        assert result.mc_error_u > 0
        assert result.mc_error_m > 0
        assert result.mc_error_u < 0.1  # Should be small with K=100

    def test_variance_reduction_factor_standard_mc(self):
        """Test variance reduction factor for standard MC."""
        from mfgarchon.alg.numerical.stochastic import CommonNoiseMFGSolver

        u = np.ones((12, 21))
        m = np.ones((12, 21))
        conditional_solutions = [(u, m, True)] * 10
        noise_paths = [np.zeros(12)] * 10

        noise_process = OrnsteinUhlenbeckProcess(kappa=1.0, mu=0.0, sigma=0.1)
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[22], boundary_conditions=no_flux_bc(dimension=1))
        problem = StochasticMFGProblem(
            geometry=geometry,
            T=0.5,
            Nt=11,
            noise_process=noise_process,
            conditional_hamiltonian=lambda x, p, m, theta: 0.5 * p**2,
            components=_default_components(),
        )
        solver = CommonNoiseMFGSolver(problem, num_noise_samples=10, variance_reduction=False)

        result = solver._aggregate_solutions(conditional_solutions, noise_paths)

        # Standard MC should have variance_reduction_factor = 1.0
        assert result.variance_reduction_factor == 1.0

    def test_variance_reduction_factor_with_control_variates(self):
        """Test variance reduction factor with control variates enabled."""
        from mfgarchon.alg.numerical.stochastic import CommonNoiseMFGSolver

        u = np.ones((12, 21))
        m = np.ones((12, 21))
        conditional_solutions = [(u, m, True)] * 10
        noise_paths = [np.zeros(12)] * 10

        noise_process = OrnsteinUhlenbeckProcess(kappa=1.0, mu=0.0, sigma=0.1)
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[22], boundary_conditions=no_flux_bc(dimension=1))
        problem = StochasticMFGProblem(
            geometry=geometry,
            T=0.5,
            Nt=11,
            noise_process=noise_process,
            conditional_hamiltonian=lambda x, p, m, theta: 0.5 * p**2,
            components=_default_components(),
        )

        mc_config = MCConfig(num_samples=10, use_control_variates=True, sampling_method="sobol")
        solver = CommonNoiseMFGSolver(problem, num_noise_samples=10, variance_reduction=True, mc_config=mc_config)

        result = solver._aggregate_solutions(conditional_solutions, noise_paths)

        # With variance reduction, factor should be > 1.0
        assert result.variance_reduction_factor > 1.0


class TestCommonNoiseSolverEdgeCases:
    """Test edge cases and error handling."""

    def test_single_noise_sample(self):
        """Test solver with K=1 noise sample."""
        noise_process = OrnsteinUhlenbeckProcess(kappa=1.0, mu=0.0, sigma=0.1)
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[22], boundary_conditions=no_flux_bc(dimension=1))
        problem = StochasticMFGProblem(
            geometry=geometry,
            T=0.5,
            Nt=11,
            noise_process=noise_process,
            conditional_hamiltonian=lambda x, p, m, theta: 0.5 * p**2,
            components=_default_components(),
        )

        solver = CommonNoiseMFGSolver(problem, num_noise_samples=1)
        paths = solver._sample_noise_paths()

        assert len(paths) == 1
        assert paths[0].shape == (12,)
        assert paths[0][0] == 0.0  # OU paths start at theta_initial

        # K=1 is the ddof=1 boundary, and the test stopped one call short of the edge it is named
        # for. RECORDED DEFECT, not a contract: the aggregation divides by K-1 = 0 and returns a
        # silently NaN spread rather than failing loud or falling back to ddof=0. Measured at HEAD:
        # mc_error_u = nan, u_std all-NaN, and the resulting confidence interval entirely NaN.
        # Asserted so that fixing it (ddof=0 at K=1, or a raise) trips this test rather than
        # passing unnoticed. The repo's fail-fast rule says the silent NaN is the defect.
        result = solver._aggregate_solutions([(np.ones((12, 21)), np.ones((12, 21)), True)], [paths[0]])
        assert np.isnan(result.mc_error_u), "K=1 aggregation no longer returns a NaN MC error -- update this pin"
        assert np.all(np.isnan(result.u_std))
        assert np.all(np.isnan(result.get_confidence_interval_u()[0]))


class TestCommonNoiseSolverConfiguration:
    """Test solver configuration and parameter validation."""

    def test_default_conditional_solver_factory(self):
        """Test that default solver factory is created when none provided."""
        noise_process = OrnsteinUhlenbeckProcess(kappa=1.0, mu=0.0, sigma=0.1)
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[22], boundary_conditions=no_flux_bc(dimension=1))
        problem = StochasticMFGProblem(
            geometry=geometry,
            T=0.5,
            Nt=11,
            noise_process=noise_process,
            conditional_hamiltonian=lambda x, p, m, theta: 0.5 * p**2,
            components=_default_components(),
        )

        solver = CommonNoiseMFGSolver(problem, num_noise_samples=10)

        assert solver.conditional_solver_factory is not None
        assert callable(solver.conditional_solver_factory)

        # `callable` is satisfied by any function at all, including the custom-factory branch this
        # test exists to distinguish itself from. Assert what the default factory DOES -- the
        # documented "uses problem.solve() API" contract and its verbose suppression -- without
        # running a solve.
        class _Recorder:
            def solve(self, **kwargs):
                self.kwargs = kwargs
                return "sentinel"

        recorder = _Recorder()
        assert solver.conditional_solver_factory(recorder) == "sentinel"
        assert recorder.kwargs == {"verbose": False}

    def test_mc_config_created_when_not_provided(self):
        """Test that MCConfig is created with appropriate defaults."""
        noise_process = OrnsteinUhlenbeckProcess(kappa=1.0, mu=0.0, sigma=0.1)
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[22], boundary_conditions=no_flux_bc(dimension=1))
        problem = StochasticMFGProblem(
            geometry=geometry,
            T=0.5,
            Nt=11,
            noise_process=noise_process,
            conditional_hamiltonian=lambda x, p, m, theta: 0.5 * p**2,
            components=_default_components(),
        )

        # With variance reduction
        solver_qmc = CommonNoiseMFGSolver(problem, num_noise_samples=20, variance_reduction=True)
        assert solver_qmc.mc_config.sampling_method == "sobol"
        assert solver_qmc.mc_config.use_control_variates is True

        # Without variance reduction
        solver_mc = CommonNoiseMFGSolver(problem, num_noise_samples=20, variance_reduction=False)
        assert solver_mc.mc_config.sampling_method == "uniform"
        assert solver_mc.mc_config.use_control_variates is False

    def test_num_workers_configuration(self):
        """Test that num_workers is properly stored."""
        noise_process = OrnsteinUhlenbeckProcess(kappa=1.0, mu=0.0, sigma=0.1)
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[22], boundary_conditions=no_flux_bc(dimension=1))
        problem = StochasticMFGProblem(
            geometry=geometry,
            T=0.5,
            Nt=11,
            noise_process=noise_process,
            conditional_hamiltonian=lambda x, p, m, theta: 0.5 * p**2,
            components=_default_components(),
        )

        # Default (None)
        solver_default = CommonNoiseMFGSolver(problem, num_noise_samples=10)
        assert solver_default.num_workers is None

        # Custom value
        solver_custom = CommonNoiseMFGSolver(problem, num_noise_samples=10, num_workers=4)
        assert solver_custom.num_workers == 4

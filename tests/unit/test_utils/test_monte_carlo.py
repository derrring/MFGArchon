"""
Unit tests for Monte Carlo utilities.

Tests sampling strategies, variance reduction, integration methods,
and convergence diagnostics for Monte Carlo computations.
"""

import pytest

import numpy as np

from mfgarchon.utils.numerical.particle.sampling import (
    ControlVariates,
    ImportanceMCSampler,
    MCConfig,
    MCResult,
    QuasiMCSampler,
    StratifiedMCSampler,
    UniformMCSampler,
    adaptive_monte_carlo,
    estimate_expectation,
    monte_carlo_integrate,
)

pytestmark = pytest.mark.experimental

# ============================================================================
# Test: MCConfig Dataclass
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_mc_config_defaults():
    """Test MCConfig default values."""
    config = MCConfig()

    assert config.num_samples == 10000
    assert config.sampling_method == "uniform"
    assert config.seed is None
    assert config.use_control_variates is False
    assert config.use_antithetic_variables is False
    assert config.adaptive is False
    assert config.parallel is False


@pytest.mark.unit
@pytest.mark.fast
def test_mc_config_custom_values():
    """Test MCConfig with custom values."""
    config = MCConfig(
        num_samples=5000,
        sampling_method="sobol",
        seed=42,
        use_control_variates=True,
        error_tolerance=1e-4,
        parallel=True,
    )

    assert config.num_samples == 5000
    assert config.sampling_method == "sobol"
    assert config.seed == 42
    assert config.use_control_variates is True
    assert config.error_tolerance == 1e-4
    assert config.parallel is True


@pytest.mark.unit
@pytest.mark.fast
def test_mc_config_stratification_params():
    """Test MCConfig stratification parameters."""
    config = MCConfig(sampling_method="stratified", num_strata_per_dim=5)

    assert config.num_strata_per_dim == 5


# ============================================================================
# Test: MCResult Dataclass
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_mc_result_creation():
    """Test MCResult dataclass creation."""
    result = MCResult(
        estimate=3.14,
        standard_error=0.01,
        confidence_interval=(3.12, 3.16),
        num_samples_used=10000,
        effective_sample_size=10000,
        converged=True,
        final_error=0.01,
    )

    assert result.estimate == 3.14
    assert result.standard_error == 0.01
    assert result.confidence_interval == (3.12, 3.16)
    assert result.num_samples_used == 10000
    assert result.converged is True


@pytest.mark.unit
@pytest.mark.fast
def test_mc_result_default_values():
    """Test MCResult default field values."""
    result = MCResult(
        estimate=1.0,
        standard_error=0.1,
        confidence_interval=(0.9, 1.1),
        num_samples_used=1000,
        effective_sample_size=1000,
    )

    assert result.variance_reduction_factor == 1.0
    assert result.converged is False
    assert result.final_error == np.inf
    assert result.iterations == 0
    assert result.computation_time == 0.0


# ============================================================================
# Test: Uniform Monte Carlo Sampler
# ============================================================================


@pytest.mark.unit
def test_uniform_sampler_reproducibility():
    """Test uniform sampler reproducibility with seed."""
    domain = [(0.0, 1.0)]
    config = MCConfig(seed=42)
    sampler1 = UniformMCSampler(domain, config)
    sampler2 = UniformMCSampler(domain, config)

    samples1 = sampler1.sample(50)
    samples2 = sampler2.sample(50)

    np.testing.assert_array_equal(samples1, samples2)


@pytest.mark.unit
def test_uniform_sampler_custom_domain():
    """Test uniform sampler with custom domain bounds."""
    domain = [(-5.0, 5.0), (10.0, 20.0), (0.0, 100.0)]
    config = MCConfig(num_samples=1000, seed=123)
    sampler = UniformMCSampler(domain, config)

    samples = sampler.sample(1000)

    assert samples.shape == (1000, 3)
    # Check bounds for each dimension
    assert np.all(samples[:, 0] >= -5.0)
    assert np.all(samples[:, 0] <= 5.0)
    assert np.all(samples[:, 1] >= 10.0)
    assert np.all(samples[:, 1] <= 20.0)
    assert np.all(samples[:, 2] >= 0.0)
    assert np.all(samples[:, 2] <= 100.0)


@pytest.mark.unit
def test_uniform_sampler_weights():
    """Test uniform sampler returns uniform weights."""
    domain = [(0.0, 1.0), (0.0, 1.0)]
    config = MCConfig(seed=42)
    sampler = UniformMCSampler(domain, config)

    samples = sampler.sample(100)
    weights = sampler.get_weights(samples)

    assert len(weights) == 100
    # All weights should be equal (1/N)
    expected_weight = 1.0 / 100
    np.testing.assert_array_almost_equal(weights, expected_weight * np.ones(100))


# ============================================================================
# Test: Stratified Monte Carlo Sampler
# ============================================================================


@pytest.mark.unit
def test_stratified_sampler_coverage():
    """Test stratified sampler covers domain uniformly."""
    domain = [(0.0, 10.0)]
    config = MCConfig(num_samples=1000, sampling_method="stratified", num_strata_per_dim=10, seed=42)
    sampler = StratifiedMCSampler(domain, config)

    samples = sampler.sample(1000)

    # Check that samples are distributed across all strata
    # Divide domain into 10 bins
    bins = np.linspace(0, 10, 11)
    counts, _ = np.histogram(samples.flatten(), bins=bins)

    # Each bin should have samples (not necessarily equal, but non-zero)
    assert np.all(counts > 0)


@pytest.mark.unit
def test_stratified_sampler_dimension_handling():
    """Test stratified sampler with different dimensions."""
    # 3D domain
    domain = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    # With 4 strata per dimension, we have 4^3 = 64 total strata
    # Use 192 samples (exactly 3 per stratum) to avoid truncation
    config = MCConfig(num_samples=192, sampling_method="stratified", num_strata_per_dim=4, seed=42)
    sampler = StratifiedMCSampler(domain, config)

    samples = sampler.sample(192)

    assert samples.shape == (192, 3)
    # 192 samples across 64 strata = exactly 3 samples per stratum


# ============================================================================
# Test: Quasi-Monte Carlo Sampler
# ============================================================================


@pytest.mark.unit
def test_quasi_mc_sampler_halton():
    """Test Halton sequence sampler."""
    pytest.importorskip("scipy")

    domain = [(0.0, 1.0)]
    config = MCConfig(num_samples=50, seed=42)
    sampler = QuasiMCSampler(domain, config, sequence_type="halton")

    samples = sampler.sample(50)

    assert samples.shape == (50, 1)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)


@pytest.mark.unit
def test_quasi_mc_sampler_latin_hypercube():
    """Test Latin hypercube sampler."""
    pytest.importorskip("scipy")

    domain = [(0.0, 1.0), (0.0, 1.0)]
    config = MCConfig(num_samples=100, seed=42)
    sampler = QuasiMCSampler(domain, config, sequence_type="latin_hypercube")

    samples = sampler.sample(100)

    assert samples.shape == (100, 2)


@pytest.mark.unit
def test_quasi_mc_sampler_custom_domain():
    """Test quasi-MC sampler with custom domain."""
    pytest.importorskip("scipy")

    domain = [(-2.0, 2.0), (5.0, 15.0)]
    config = MCConfig(num_samples=100, seed=42)
    sampler = QuasiMCSampler(domain, config, sequence_type="sobol")

    samples = sampler.sample(100)

    assert np.all(samples[:, 0] >= -2.0)
    assert np.all(samples[:, 0] <= 2.0)
    assert np.all(samples[:, 1] >= 5.0)
    assert np.all(samples[:, 1] <= 15.0)


# ============================================================================
# Test: Importance Monte Carlo Sampler
# ============================================================================


@pytest.mark.unit
def test_importance_sampler_basic():
    """Test importance sampler with custom importance function."""
    domain = [(0.0, 1.0)]
    config = MCConfig(num_samples=100, seed=42)

    # Importance function: favor center of domain
    def importance_fn(x):
        return np.exp(-10 * (x[:, 0] - 0.5) ** 2)

    sampler = ImportanceMCSampler(domain, config, importance_fn)

    samples = sampler.sample(100)

    assert samples.shape == (100, 1)
    # Samples should be more concentrated near 0.5
    mean_sample = np.mean(samples)
    assert 0.3 < mean_sample < 0.7  # Should be near 0.5


@pytest.mark.unit
def test_importance_sampler_weights():
    """Test importance sampler returns non-uniform weights."""
    domain = [(0.0, 1.0)]
    config = MCConfig(seed=42)

    def importance_fn(x):
        return np.ones(len(x))  # Uniform importance (for testing)

    sampler = ImportanceMCSampler(domain, config, importance_fn)

    samples = sampler.sample(50)
    weights = sampler.get_weights(samples)

    assert len(weights) == 50
    # Weights should sum to 1
    assert np.abs(np.sum(weights) - 1.0) < 1e-10


# ============================================================================
# Test: Monte Carlo Integration
# ============================================================================


@pytest.mark.unit
def test_monte_carlo_integrate_1d_simple():
    """Test MC integration of simple 1D function."""

    # Integrate f(x) = x over [0, 1], analytical result = 0.5
    def integrand(x):
        return x[:, 0]

    domain = [(0.0, 1.0)]
    config = MCConfig(num_samples=10000, seed=42)

    result = monte_carlo_integrate(integrand, domain, config)

    # Should be close to 0.5
    assert abs(result.estimate - 0.5) < 0.05
    assert result.standard_error > 0
    assert result.num_samples_used == 10000


@pytest.mark.unit
def test_monte_carlo_integrate_constant():
    """Test MC integration of constant function."""

    # Integrate f(x) = 2 over [0, 1], analytical result = 2
    def integrand(x):
        return 2.0 * np.ones(len(x))

    domain = [(0.0, 1.0)]
    config = MCConfig(num_samples=1000, seed=42)

    result = monte_carlo_integrate(integrand, domain, config)

    # Should be very close to 2.0 (constant function, low variance)
    assert abs(result.estimate - 2.0) < 0.01


@pytest.mark.unit
def test_monte_carlo_integrate_2d():
    """Test MC integration in 2D."""

    # Integrate f(x,y) = x * y over [0,1]×[0,1], analytical result = 0.25
    def integrand(x):
        return x[:, 0] * x[:, 1]

    domain = [(0.0, 1.0), (0.0, 1.0)]
    config = MCConfig(num_samples=20000, seed=42)

    result = monte_carlo_integrate(integrand, domain, config)

    # Should be close to 0.25
    assert abs(result.estimate - 0.25) < 0.05


@pytest.mark.unit
def test_monte_carlo_integrate_confidence_interval():
    """Test MC integration confidence interval."""

    def integrand(x):
        return x[:, 0] ** 2

    domain = [(0.0, 1.0)]
    config = MCConfig(num_samples=5000, seed=42)

    result = monte_carlo_integrate(integrand, domain, config)

    ci_lower, ci_upper = result.confidence_interval

    # CI should contain the estimate
    assert ci_lower < result.estimate < ci_upper
    # CI should contain analytical result (1/3)
    assert ci_lower < 1.0 / 3.0 < ci_upper


@pytest.mark.unit
def test_monte_carlo_integrate_stratified():
    """Test MC integration with stratified sampling."""

    def integrand(x):
        return np.sin(np.pi * x[:, 0])

    domain = [(0.0, 1.0)]
    config = MCConfig(num_samples=5000, sampling_method="stratified", num_strata_per_dim=10, seed=42)

    result = monte_carlo_integrate(integrand, domain, config)

    # Analytical result: ∫₀¹ sin(πx)dx = 2/π ≈ 0.6366
    analytical = 2.0 / np.pi
    assert abs(result.estimate - analytical) < 0.05


@pytest.mark.unit
def test_monte_carlo_integrate_quasi():
    """Test MC integration with quasi-Monte Carlo."""
    pytest.importorskip("scipy")

    def integrand(x):
        return x[:, 0] ** 2

    domain = [(0.0, 1.0)]
    config = MCConfig(num_samples=5000, sampling_method="sobol", seed=42)

    result = monte_carlo_integrate(integrand, domain, config)

    # Should converge faster than standard MC (but we just check it works)
    assert abs(result.estimate - 1.0 / 3.0) < 0.05


@pytest.mark.unit
def test_monte_carlo_integrate_custom_domain():
    """Test MC integration with custom domain bounds."""

    # Integrate f(x) = 1 over [-2, 2], analytical result = 4
    def integrand(x):
        return np.ones(len(x))

    domain = [(-2.0, 2.0)]
    config = MCConfig(num_samples=1000, seed=42)

    result = monte_carlo_integrate(integrand, domain, config)

    assert abs(result.estimate - 4.0) < 0.1


# ============================================================================
# Test: Control Variates
# ============================================================================


@pytest.mark.unit
def test_control_variates_calibration():
    """Test control variates calibration."""

    # Control function: g(x) = x (known mean = 0.5 on [0,1])
    def control_fn(x):
        return x[:, 0]

    cv = ControlVariates(control_fn)

    # Generate sample data
    np.random.seed(42)
    sample_points = np.random.uniform(0, 1, (1000, 1))
    target_values = sample_points[:, 0] ** 2  # f(x) = x^2

    cv.calibrate(sample_points, target_values, control_expectation=0.5)

    assert cv.is_calibrated
    assert cv.control_mean == 0.5
    assert cv.control_coefficient != 0.0


@pytest.mark.unit
def test_control_variates_variance_reduction():
    """Test control variates actually reduces variance."""

    def control_fn(x):
        return x[:, 0]

    cv = ControlVariates(control_fn)

    # Sample data
    np.random.seed(42)
    sample_points = np.random.uniform(0, 1, (5000, 1))
    target_values = sample_points[:, 0] ** 2

    # Calibrate
    cv.calibrate(sample_points, target_values, control_expectation=0.5)

    # Apply control variate
    reduced_values = cv.apply(sample_points, target_values)

    # Variance should be reduced
    original_var = np.var(target_values)
    reduced_var = np.var(reduced_values)

    assert reduced_var < original_var


@pytest.mark.unit
def test_control_variates_without_calibration():
    """Test control variates returns original values if not calibrated."""

    def control_fn(x):
        return x[:, 0]

    cv = ControlVariates(control_fn)

    sample_points = np.random.uniform(0, 1, (100, 1))
    target_values = sample_points[:, 0] ** 2

    # Apply without calibration
    result = cv.apply(sample_points, target_values)

    # Should return original values
    np.testing.assert_array_equal(result, target_values)


# ============================================================================
# Test: Utility Functions
# ============================================================================


@pytest.mark.unit
def test_estimate_expectation_uniform_weights():
    """Test expectation estimation with uniform weights."""
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    expectation, std_error = estimate_expectation(samples)

    assert expectation == 3.0  # Mean of [1,2,3,4,5]
    assert std_error > 0


@pytest.mark.unit
def test_estimate_expectation_custom_weights():
    """Test expectation estimation with custom weights."""
    samples = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.5, 0.3, 0.2])

    expectation, std_error = estimate_expectation(samples, weights)

    # Weighted mean: 1*0.5 + 2*0.3 + 3*0.2 = 1.7
    expected_mean = 1.7
    assert abs(expectation - expected_mean) < 1e-10
    assert std_error > 0


@pytest.mark.unit
def test_estimate_expectation_zero_variance():
    """Test expectation estimation with constant samples."""
    samples = np.array([5.0, 5.0, 5.0, 5.0])

    expectation, std_error = estimate_expectation(samples)

    assert expectation == 5.0
    assert std_error == 0.0


# ============================================================================
# Test: Adaptive Monte Carlo
# ============================================================================


@pytest.mark.unit
def test_adaptive_monte_carlo_convergence():
    """Test adaptive MC converges to target error."""

    # Simple integrand
    def integrand(x):
        return x[:, 0] ** 2

    domain = [(0.0, 1.0)]

    result = adaptive_monte_carlo(
        integrand, domain, target_error=0.01, initial_samples=100, max_samples=10000, batch_size=500
    )

    # Should converge or reach max samples
    assert result.num_samples_used <= 10000
    assert result.iterations > 0


@pytest.mark.unit
def test_adaptive_monte_carlo_initial_samples():
    """Test adaptive MC starts with initial samples."""

    def integrand(x):
        return np.ones(len(x))

    domain = [(0.0, 1.0)]

    result = adaptive_monte_carlo(integrand, domain, initial_samples=500, max_samples=5000, batch_size=500)

    assert result.num_samples_used >= 500


# ============================================================================
# Test: Edge Cases and Error Handling
# ============================================================================


@pytest.mark.unit
def test_mc_integrate_small_samples():
    """Test MC integration with very few samples."""

    def integrand(x):
        return x[:, 0]

    domain = [(0.0, 1.0)]
    config = MCConfig(num_samples=10)

    result = monte_carlo_integrate(integrand, domain, config)

    # Should still return a result (though with high error)
    assert result.num_samples_used == 10
    assert result.standard_error > 0


@pytest.mark.unit
def test_mc_integrate_high_dimensional():
    """Test MC integration in high dimensions."""

    # 5D constant function
    def integrand(x):
        return np.ones(len(x))

    domain = [(0.0, 1.0)] * 5  # 5D unit cube
    config = MCConfig(num_samples=5000, seed=42)

    result = monte_carlo_integrate(integrand, domain, config)

    # Volume of unit cube is 1
    assert abs(result.estimate - 1.0) < 0.1


@pytest.mark.unit
def test_sampler_domain_volume_computation():
    """Test sampler computes domain volume correctly."""
    domain = [(0.0, 2.0), (0.0, 3.0)]  # Volume = 2 * 3 = 6
    config = MCConfig(seed=42)
    sampler = UniformMCSampler(domain, config)

    assert sampler.domain_volume == 6.0


@pytest.mark.unit
def test_sampler_single_dimension():
    """Test sampler works with 1D domain."""
    domain = [(0.0, 1.0)]
    config = MCConfig(seed=42)
    sampler = UniformMCSampler(domain, config)

    samples = sampler.sample(100)

    assert samples.shape == (100, 1)
    assert sampler.dimension == 1

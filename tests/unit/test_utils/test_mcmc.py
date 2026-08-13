"""
Unit tests for MCMC and Hamiltonian Monte Carlo utilities.

Tests for mfgarchon.utils.numerical.particle.mcmc.py covering:
- MCMCConfig and MCMCResult dataclasses
- Metropolis-Hastings sampler
- Hamiltonian Monte Carlo (HMC) with leapfrog integration
- No-U-Turn Sampler (NUTS)
- Langevin Dynamics
- Convergence diagnostics (R-hat, ESS)
- MFG-specific convenience functions
"""

import pytest

import numpy as np

from mfgarchon.utils.numerical.particle.mcmc import (
    HamiltonianMonteCarlo,
    LangevinDynamics,
    MCMCConfig,
    MCMCResult,
    MetropolisHastings,
    NoUTurnSampler,
    compute_rhat,
    effective_sample_size,
    sample_mfg_posterior,
)

pytestmark = pytest.mark.experimental

# =============================================================================
# Test MCMCConfig and MCMCResult Dataclasses
# =============================================================================


@pytest.mark.unit
def test_mcmc_config_defaults():
    """Test MCMCConfig default values."""
    config = MCMCConfig()

    assert config.num_samples == 10000
    assert config.num_warmup == 1000
    assert config.num_chains == 1
    assert config.thinning == 1
    assert config.step_size == 0.1
    assert config.adapt_step_size is True
    assert config.target_accept_rate == 0.65
    assert config.num_leapfrog_steps == 10
    assert config.max_tree_depth == 10


@pytest.mark.unit
def test_mcmc_config_custom():
    """Test MCMCConfig with custom values."""
    config = MCMCConfig(num_samples=5000, num_warmup=500, step_size=0.05, num_leapfrog_steps=20, seed=42)

    assert config.num_samples == 5000
    assert config.num_warmup == 500
    assert config.step_size == 0.05
    assert config.num_leapfrog_steps == 20
    assert config.seed == 42


@pytest.mark.unit
def test_mcmc_result_dataclass():
    """Test MCMCResult dataclass creation."""
    samples = np.random.randn(100, 1, 2)
    log_densities = np.random.randn(100)

    result = MCMCResult(
        samples=samples,
        log_densities=log_densities,
        acceptance_rate=0.7,
        num_samples=100,
        num_warmup=50,
    )

    assert result.samples.shape == (100, 1, 2)
    assert len(result.log_densities) == 100
    assert result.acceptance_rate == 0.7
    assert result.num_samples == 100
    assert result.num_warmup == 50


# =============================================================================
# Test Metropolis-Hastings Sampler
# =============================================================================


@pytest.mark.unit
def test_metropolis_hastings_basic():
    """Test Metropolis-Hastings samples from standard Gaussian."""

    # Target: N(0, 1)
    def potential_fn(x):
        return 0.5 * np.sum(x**2)

    config = MCMCConfig(num_samples=1000, num_warmup=200, step_size=1.0, adapt_step_size=False, seed=42)
    sampler = MetropolisHastings(potential_fn, proposal_std=1.0, config=config)

    initial_state = np.array([0.0])
    result = sampler.sample(initial_state, config.num_samples)

    assert result.samples.shape == (1000, 1, 1)
    assert len(result.log_densities) == 1000
    assert 0.0 < result.acceptance_rate < 1.0

    # Check that samples are finite (basic sanity check)
    assert np.all(np.isfinite(result.samples))
    assert result.acceptance_rate > 0.1


@pytest.mark.unit
def test_metropolis_hastings_2d():
    """Test Metropolis-Hastings in 2D."""

    # Target: N([0,0], I)
    def potential_fn(x):
        return 0.5 * np.sum(x**2)

    config = MCMCConfig(num_samples=1000, num_warmup=200, seed=42)
    sampler = MetropolisHastings(potential_fn, proposal_std=1.0, config=config)

    initial_state = np.array([0.0, 0.0])
    result = sampler.sample(initial_state, config.num_samples)

    assert result.samples.shape == (1000, 1, 2)
    assert result.acceptance_rate > 0.1


@pytest.mark.unit
def test_metropolis_hastings_thinning():
    """Test Metropolis-Hastings with thinning."""

    def potential_fn(x):
        return 0.5 * np.sum(x**2)

    config = MCMCConfig(num_samples=1000, num_warmup=100, thinning=5, seed=42)
    sampler = MetropolisHastings(potential_fn, config=config)

    result = sampler.sample(np.array([0.0]), config.num_samples)

    # After thinning, should have 1000/5 = 200 samples
    assert result.samples.shape[0] == 200

    # Which samples survive, not just how many.  Thinning exists to drop the autocorrelated head
    # of a chain, and a regression that returned the FIRST 200 samples satisfies the count above
    # while doing the opposite.  Thinning affects only the final slice, so at the same seed the
    # underlying chain is identical and the stride identity is exact (verified: the truncation
    # alternative, thin1[:200], does NOT match).
    config_full = MCMCConfig(num_samples=1000, num_warmup=100, thinning=1, seed=42)
    result_full = MetropolisHastings(potential_fn, config=config_full).sample(np.array([0.0]), 1000)
    assert np.array_equal(result.samples, result_full.samples[::5])


@pytest.mark.unit
def test_metropolis_hastings_step_size_adaptation():
    """Test step size adaptation during warmup."""

    def potential_fn(x):
        return 0.5 * np.sum(x**2)

    config = MCMCConfig(num_samples=500, num_warmup=200, step_size=0.1, adapt_step_size=True, seed=42)
    sampler = MetropolisHastings(potential_fn, config=config)

    initial_step_size = sampler.step_size
    result = sampler.sample(np.array([0.0]), config.num_samples)

    # Step size should have adapted
    assert result.final_step_size != initial_step_size
    assert result.final_step_size > 0


@pytest.mark.unit
def test_metropolis_hastings_performance_metrics():
    """Test Metropolis-Hastings records performance metrics."""

    def potential_fn(x):
        return 0.5 * np.sum(x**2)

    config = MCMCConfig(num_samples=100, num_warmup=20, seed=42)
    sampler = MetropolisHastings(potential_fn, config=config)

    result = sampler.sample(np.array([0.0]), config.num_samples)

    assert result.sampling_time > 0
    assert result.samples_per_second > 0


# =============================================================================
# Test Hamiltonian Monte Carlo (HMC)
# =============================================================================


@pytest.mark.unit
def test_hmc_basic():
    """Test HMC samples from standard Gaussian."""

    # Target: N(0, 1)
    def potential_fn(x):
        return 0.5 * np.sum(x**2)

    def gradient_fn(x):
        return x

    config = MCMCConfig(num_samples=500, num_warmup=100, step_size=0.15, num_leapfrog_steps=10, seed=42)
    sampler = HamiltonianMonteCarlo(potential_fn, gradient_fn, config=config)

    initial_state = np.array([0.0])
    result = sampler.sample(initial_state, config.num_samples)

    assert result.samples.shape == (500, 1, 1)
    assert 0.0 < result.acceptance_rate <= 1.0

    # Check approximate correctness
    sample_mean = np.mean(result.samples)
    sample_std = np.std(result.samples)
    assert abs(sample_mean) < 0.5
    assert 0.5 < sample_std < 1.5


@pytest.mark.unit
def test_hmc_2d_gaussian():
    """Test HMC on 2D Gaussian."""

    # Target: N([0,0], I)
    def potential_fn(x):
        return 0.5 * np.sum(x**2)

    def gradient_fn(x):
        return x

    config = MCMCConfig(num_samples=500, num_warmup=100, step_size=0.1, num_leapfrog_steps=10, seed=42)
    sampler = HamiltonianMonteCarlo(potential_fn, gradient_fn, config=config)

    initial_state = np.array([0.0, 0.0])
    result = sampler.sample(initial_state, config.num_samples)

    assert result.samples.shape == (500, 1, 2)
    assert result.acceptance_rate > 0.3


@pytest.mark.unit
def test_hmc_leapfrog_integration():
    """Test HMC leapfrog integration preserves energy approximately."""

    def potential_fn(x):
        return 0.5 * np.sum(x**2)

    def gradient_fn(x):
        return x

    config = MCMCConfig(num_samples=100, num_warmup=20, step_size=0.05, seed=42)
    sampler = HamiltonianMonteCarlo(potential_fn, gradient_fn, config=config)

    # Initialize mass matrix
    sampler.mass_matrix = np.eye(1)

    # Test leapfrog integration manually
    q = np.array([1.0])
    p = np.array([0.5])

    q_new, p_new = sampler._leapfrog_integrate(q, p, num_steps=10, step_size=0.05)

    # Energy should be approximately conserved (within numerical error)
    initial_energy = potential_fn(q) + 0.5 * p.T @ p
    final_energy = potential_fn(q_new) + 0.5 * p_new.T @ p_new

    # Leapfrog is symplectic so energy error should be small
    assert abs(final_energy - initial_energy) < 0.5


@pytest.mark.unit
def test_hmc_mass_matrix_adaptation():
    """Test HMC adapts mass matrix during warmup.

    Two things are needed to make this test see the flag it is named for, and the previous
    fixture had neither:

    1. num_warmup must exceed 200.  The adaptation branch is `metric_adaptation and i > 100 and
       i % 100 == 0` nested inside `i < num_warmup`, so at num_warmup=200 no index qualifies and
       adaptation never runs at all.
    2. The target must be anisotropic.  The adapted matrix is a sample covariance, and on
       N(0, I) that is the identity -- which is also the un-adapted initialisation, so the two
       branches are indistinguishable.

    With the old fixture the mass matrix came out exactly eye(2) whether metric_adaptation was
    True or False.
    """
    # Target covariance diag(1, 25): potential 0.5 * x^T A x with A = diag(1, 1/25).
    A = np.diag([1.0, 1.0 / 25.0])

    def potential_fn(x):
        return 0.5 * x @ A @ x

    def gradient_fn(x):
        return A @ x

    config = MCMCConfig(
        num_samples=400,
        num_warmup=300,
        step_size=0.1,
        metric_adaptation=True,
        seed=42,
    )
    sampler = HamiltonianMonteCarlo(potential_fn, gradient_fn, config=config)

    sampler.sample(np.array([0.0, 0.0]), config.num_samples)

    # Mass matrix should have been adapted
    assert sampler.mass_matrix is not None
    assert sampler.mass_matrix.shape == (2, 2)

    # The adapted matrix must have picked up the 25:1 anisotropy of the target.  Measured
    # [[1.156, 0.230], [0.230, 24.257]], ratio 21.0 against the true 25; the same run with
    # metric_adaptation=False leaves it exactly eye(2), ratio 1.0.  A threshold of 10 sits 2.1x
    # below the adapted value and 10x above the un-adapted one.
    ratio = sampler.mass_matrix[1, 1] / sampler.mass_matrix[0, 0]
    assert ratio > 10.0, f"mass matrix did not adapt to the target anisotropy: ratio = {ratio}"


@pytest.mark.unit
def test_hmc_custom_mass_matrix():
    """Test HMC with custom mass matrix."""

    def potential_fn(x):
        return 0.5 * x.T @ np.array([[2.0, 0.0], [0.0, 0.5]]) @ x

    def gradient_fn(x):
        return np.array([[2.0, 0.0], [0.0, 0.5]]) @ x

    # Use mass matrix matching the metric
    custom_mass = np.array([[2.0, 0.0], [0.0, 0.5]])

    config = MCMCConfig(num_samples=300, num_warmup=100, step_size=0.1, mass_matrix=custom_mass, seed=42)
    sampler = HamiltonianMonteCarlo(potential_fn, gradient_fn, config=config)

    result = sampler.sample(np.array([0.0, 0.0]), config.num_samples)

    assert result.acceptance_rate > 0.3
    assert np.allclose(sampler.mass_matrix, custom_mass)


@pytest.mark.unit
def test_hmc_thinning():
    """Test HMC with thinning."""

    def potential_fn(x):
        return 0.5 * np.sum(x**2)

    def gradient_fn(x):
        return x

    config = MCMCConfig(num_samples=1000, num_warmup=100, thinning=10, step_size=0.1, seed=42)
    sampler = HamiltonianMonteCarlo(potential_fn, gradient_fn, config=config)

    result = sampler.sample(np.array([0.0]), config.num_samples)

    # After thinning, should have 1000/10 = 100 samples
    assert result.samples.shape[0] == 100

    # HMC inherits the same final slice as Metropolis-Hastings but reaches it through its own
    # sample loop, so the stride identity needs its own case.  Verified exact; the truncation
    # alternative, thin1[:100], does not match.
    config_full = MCMCConfig(num_samples=1000, num_warmup=100, thinning=1, step_size=0.1, seed=42)
    result_full = HamiltonianMonteCarlo(potential_fn, gradient_fn, config=config_full).sample(np.array([0.0]), 1000)
    assert np.array_equal(result.samples, result_full.samples[::10])


# =============================================================================
# Test No-U-Turn Sampler (NUTS)
# =============================================================================


@pytest.mark.unit
def test_nuts_basic():
    """Test NUTS sampler basic functionality."""

    def potential_fn(x):
        return 0.5 * np.sum(x**2)

    def gradient_fn(x):
        return x

    config = MCMCConfig(num_samples=300, num_warmup=100, step_size=0.1, seed=42)
    sampler = NoUTurnSampler(potential_fn, gradient_fn, config=config)

    result = sampler.sample(np.array([0.0, 0.0]), config.num_samples)

    assert result.samples.shape == (300, 1, 2)
    assert result.acceptance_rate > 0.0


@pytest.mark.unit
def test_nuts_adaptive_steps():
    """Test NUTS chooses number of leapfrog steps adaptively."""

    def potential_fn(x):
        return 0.5 * np.sum(x**2)

    def gradient_fn(x):
        return x

    config = MCMCConfig(num_samples=100, num_warmup=20, seed=42)
    sampler = NoUTurnSampler(potential_fn, gradient_fn, config=config)

    # Check that NUTS chooses steps based on dimension
    num_steps_2d = sampler._choose_num_steps(np.array([0.0, 0.0]))
    num_steps_10d = sampler._choose_num_steps(np.zeros(10))

    assert num_steps_2d > 0
    assert num_steps_10d > num_steps_2d


# =============================================================================
# Test Langevin Dynamics
# =============================================================================


@pytest.mark.unit
def test_langevin_dynamics_basic():
    """Test Langevin dynamics samples from Gaussian."""

    # Target: N(0, 1)
    def potential_fn(x):
        return 0.5 * np.sum(x**2)

    def gradient_fn(x):
        return x

    config = MCMCConfig(num_samples=1000, num_warmup=200, step_size=0.01, seed=42)
    sampler = LangevinDynamics(potential_fn, gradient_fn, config=config)

    result = sampler.sample(np.array([0.0]), config.num_samples)

    assert result.samples.shape == (1000, 1, 1)
    assert result.acceptance_rate == 1.0

    # Check approximate correctness (Langevin can have larger error)
    sample_mean = np.mean(result.samples)
    sample_std = np.std(result.samples)
    assert abs(sample_mean) < 0.5
    assert abs(sample_std - 1.0) < 0.5


@pytest.mark.unit
def test_langevin_dynamics_step_size_adaptation():
    """Test Langevin dynamics adapts step size based on gradient norm."""

    def potential_fn(x):
        return 0.5 * np.sum(x**2)

    def gradient_fn(x):
        return x

    config = MCMCConfig(num_samples=300, num_warmup=100, step_size=0.1, adapt_step_size=True, seed=42)
    sampler = LangevinDynamics(potential_fn, gradient_fn, config=config)

    initial_step_size = sampler.step_size
    result = sampler.sample(np.array([0.0]), config.num_samples)

    # Step size should have adapted during warmup
    assert result.final_step_size != initial_step_size


# =============================================================================
# Test Convergence Diagnostics
# =============================================================================


@pytest.mark.unit
def test_compute_rhat_single_chain():
    """Test R-hat returns 1.0 for single chain."""
    samples = np.random.randn(100, 1, 2)

    rhat = compute_rhat(samples)

    assert rhat.shape == (2,)
    assert np.allclose(rhat, 1.0)


@pytest.mark.unit
def test_compute_rhat_multiple_chains_converged():
    """Test R-hat for converged chains (should be close to 1)."""
    # Generate multiple chains from same distribution
    np.random.seed(42)
    num_samples = 200
    num_chains = 4
    dimension = 2

    chains = np.random.randn(num_samples, num_chains, dimension)

    rhat = compute_rhat(chains)

    assert rhat.shape == (dimension,)
    assert np.all(rhat < 1.2)


@pytest.mark.unit
def test_compute_rhat_multiple_chains_diverged():
    """Test R-hat for diverged chains (should be > 1)."""
    num_samples = 100
    num_chains = 3
    dimension = 1

    # Create diverged chains with different means
    chains = np.zeros((num_samples, num_chains, dimension))
    chains[:, 0, 0] = np.random.randn(num_samples) + 0.0
    chains[:, 1, 0] = np.random.randn(num_samples) + 5.0
    chains[:, 2, 0] = np.random.randn(num_samples) + 10.0

    rhat = compute_rhat(chains)

    # R-hat should be significantly > 1 for diverged chains
    assert rhat[0] > 1.5


@pytest.mark.unit
def test_effective_sample_size_basic():
    """Test effective sample size calculation."""
    np.random.seed(42)
    num_samples = 500
    num_chains = 2
    dimension = 2

    chains = np.random.randn(num_samples, num_chains, dimension)

    ess = effective_sample_size(chains)

    assert ess.shape == (dimension,)
    assert np.all(ess > 0)
    assert np.all(ess <= num_samples * num_chains)

    # These chains are i.i.d. draws, so the autocorrelation is zero and the effective sample
    # size must be the total sample count -- not merely somewhere in (0, N*C], which an
    # estimator returning 1 for every dimension also satisfies.  Measured exactly 1000.0 on
    # both dimensions, so rtol=0.1 is generous.
    np.testing.assert_allclose(ess, num_samples * num_chains, rtol=0.1)


@pytest.mark.unit
def test_effective_sample_size_high_correlation():
    """Test ESS for highly autocorrelated samples."""
    np.random.seed(42)
    num_samples = 200
    num_chains = 1
    dimension = 1

    # Generate highly autocorrelated samples
    chains = np.zeros((num_samples, num_chains, dimension))
    chains[0, 0, 0] = 0.0
    for i in range(1, num_samples):
        chains[i, 0, 0] = 0.95 * chains[i - 1, 0, 0] + np.random.randn() * 0.1

    ess = effective_sample_size(chains)

    # ESS should be less than num_samples due to autocorrelation
    # (relaxed assertion since ESS calculation can vary)
    assert ess[0] > 0
    assert ess[0] <= num_samples


# =============================================================================
# Test MFG-Specific Convenience Functions
# =============================================================================


@pytest.mark.unit
def test_sample_mfg_posterior_hmc():
    """Test sample_mfg_posterior with HMC method."""

    def log_posterior(x):
        return -0.5 * np.sum(x**2)

    def grad_log_posterior(x):
        return -x

    initial_params = np.array([0.0, 0.0])

    result = sample_mfg_posterior(
        log_posterior,
        grad_log_posterior,
        initial_params,
        method="hmc",
        num_samples=200,
        num_warmup=50,
        step_size=0.1,
        seed=42,
    )

    assert result.samples.shape == (200, 1, 2)
    assert result.acceptance_rate > 0.0


@pytest.mark.unit
def test_sample_mfg_posterior_nuts():
    """Test sample_mfg_posterior with NUTS method."""

    def log_posterior(x):
        return -0.5 * np.sum(x**2)

    def grad_log_posterior(x):
        return -x

    result = sample_mfg_posterior(
        log_posterior,
        grad_log_posterior,
        np.array([0.0]),
        method="nuts",
        num_samples=100,
        num_warmup=20,
        seed=42,
    )

    assert result.samples.shape[0] == 100


@pytest.mark.unit
def test_sample_mfg_posterior_mh():
    """Test sample_mfg_posterior with Metropolis-Hastings."""

    def log_posterior(x):
        return -0.5 * np.sum(x**2)

    def grad_log_posterior(x):
        return -x

    result = sample_mfg_posterior(
        log_posterior,
        grad_log_posterior,
        np.array([0.0]),
        method="mh",
        num_samples=200,
        num_warmup=50,
        seed=42,
    )

    assert result.samples.shape == (200, 1, 1)


@pytest.mark.unit
def test_sample_mfg_posterior_langevin():
    """Test sample_mfg_posterior with Langevin dynamics."""

    def log_posterior(x):
        return -0.5 * np.sum(x**2)

    def grad_log_posterior(x):
        return -x

    result = sample_mfg_posterior(
        log_posterior,
        grad_log_posterior,
        np.array([0.0, 0.0]),
        method="langevin",
        num_samples=300,
        num_warmup=50,
        step_size=0.01,
        seed=42,
    )

    assert result.samples.shape == (300, 1, 2)

    # Langevin has no accept/reject step, so every proposal is kept.  This narrows the dispatch:
    # an implementation that ignored method= and fell back to MH would report ~0.9 here.  It is
    # not a complete discriminator -- NUTS also returns exactly 1.0 on this target -- so it is
    # paired with the moment check below rather than used alone.
    assert result.acceptance_rate == 1.0

    # The target is N(0, I_2); measured sample mean [-0.021, 0.066], so 0.3 is ~4.6x margin.
    s = result.samples.reshape(300, 2)
    assert np.abs(s.mean(axis=0)).max() < 0.3


@pytest.mark.unit
def test_sample_mfg_posterior_invalid_method():
    """Test sample_mfg_posterior raises error for invalid method."""

    def log_posterior(x):
        return -0.5 * np.sum(x**2)

    def grad_log_posterior(x):
        return -x

    with pytest.raises(ValueError, match="Unknown MCMC method"):
        sample_mfg_posterior(
            log_posterior,
            grad_log_posterior,
            np.array([0.0]),
            method="invalid_method",
        )


# =============================================================================
# Test Numerical Gradient Fallback
# =============================================================================


@pytest.mark.unit
def test_numerical_gradient_accuracy():
    """Test numerical gradient approximation accuracy."""

    def potential_fn(x):
        return 0.5 * x[0] ** 2 + 2.0 * x[1] ** 2

    def analytical_gradient(x):
        return np.array([x[0], 4.0 * x[1]])

    config = MCMCConfig(seed=42)
    sampler = MetropolisHastings(potential_fn, config=config)

    x = np.array([1.0, 0.5])

    numerical_grad = sampler._numerical_gradient(x)
    analytical_grad = analytical_gradient(x)

    # Numerical gradient should be close to analytical
    assert np.allclose(numerical_grad, analytical_grad, atol=1e-4)


# =============================================================================
# Test Edge Cases
# =============================================================================


@pytest.mark.unit
def test_mcmc_high_dimensional():
    """Test MCMC in higher dimensions (10D)."""

    def potential_fn(x):
        return 0.5 * np.sum(x**2)

    def gradient_fn(x):
        return x

    config = MCMCConfig(num_samples=200, num_warmup=50, step_size=0.05, seed=42)
    sampler = HamiltonianMonteCarlo(potential_fn, gradient_fn, config=config)

    result = sampler.sample(np.zeros(10), config.num_samples)

    assert result.samples.shape == (200, 1, 10)

    # Shape cannot distinguish a working 10D sampler from one that never leaves the origin.
    # The target is N(0, I_10); these are the same moment checks test_hmc_basic already makes
    # for 1D and 2D.  Measured max|mean| = 0.259 (threshold 0.5) and per-coordinate std in
    # [0.898, 1.121] (thresholds 0.7 / 1.3).
    s = result.samples.reshape(200, 10)
    assert np.abs(s.mean(axis=0)).max() < 0.5
    assert (s.std(axis=0) > 0.7).all()
    assert (s.std(axis=0) < 1.3).all()


@pytest.mark.unit
def test_mcmc_multimodal_distribution():
    """Test MCMC on multimodal distribution (mixture of Gaussians)."""

    # Mixture of two Gaussians: 0.5*N(-3,1) + 0.5*N(3,1)
    def potential_fn(x):
        # -log(0.5*exp(-0.5*(x+3)^2) + 0.5*exp(-0.5*(x-3)^2))
        log_density = np.log(0.5 * np.exp(-0.5 * (x[0] + 3) ** 2) + 0.5 * np.exp(-0.5 * (x[0] - 3) ** 2))
        return -log_density

    config = MCMCConfig(num_samples=500, num_warmup=100, seed=42)
    sampler = MetropolisHastings(potential_fn, proposal_std=2.0, config=config)

    result = sampler.sample(np.array([0.0]), config.num_samples)

    # Samples should cover both modes
    samples_flat = result.samples.flatten()
    assert np.any(samples_flat < 0)
    assert np.any(samples_flat > 0)


@pytest.mark.unit
def test_mcmc_zero_warmup():
    """Test MCMC with zero warmup samples."""

    def potential_fn(x):
        return 0.5 * np.sum(x**2)

    config = MCMCConfig(num_samples=100, num_warmup=0, seed=42)
    sampler = MetropolisHastings(potential_fn, config=config)

    result = sampler.sample(np.array([0.0]), config.num_samples)

    assert result.samples.shape[0] == 100
    assert result.num_warmup == 0


@pytest.mark.unit
def test_mcmc_no_adaptation():
    """Test MCMC with adaptation disabled."""

    def potential_fn(x):
        return 0.5 * np.sum(x**2)

    config = MCMCConfig(num_samples=100, num_warmup=50, adapt_step_size=False, step_size=0.5, seed=42)
    sampler = MetropolisHastings(potential_fn, proposal_std=0.5, config=config)

    result = sampler.sample(np.array([0.0]), config.num_samples)

    # Step size should remain unchanged (within tolerance)
    assert abs(result.final_step_size - 0.5) < 0.01

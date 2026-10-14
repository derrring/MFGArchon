"""
Markov Chain Monte Carlo (MCMC) and Hamiltonian Monte Carlo (HMC) utilities.

This module provides advanced MCMC methods for sampling from complex distributions
in Mean Field Games, particularly useful for:
- Bayesian inference in Neural MFG solvers (uncertainty quantification)
- Sampling from equilibrium distributions in RL paradigm
- Importance sampling from intractable posterior distributions
- High-dimensional integration with complex geometry

Key Methods:
- Metropolis-Hastings: General MCMC with acceptance/rejection
- Hamiltonian Monte Carlo (HMC): Gradient-based MCMC using Hamiltonian dynamics
- No-U-Turn Sampler (NUTS): Self-tuning HMC variant
- Langevin Dynamics: Gradient-based sampling for smooth potentials
- Ensemble samplers: Parallel chain methods for multimodal distributions

Mathematical Framework:
- Target Distribution: π(x) ∝ exp(-U(x)) where U(x) is potential energy
- Hamiltonian: H(q,p) = U(q) + (1/2)p^T M^(-1) p
- Hamilton's Equations: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
- Metropolis Acceptance: α = min(1, exp(-ΔH))

Applications in MFG:
- Bayesian PINNs: Sample network weights to quantify uncertainty
- Policy Sampling: Sample policies in MFRL from posterior distributions
- Equilibrium Sampling: Sample from MFG equilibrium measures
- Parameter Inference: Bayesian estimation of MFG model parameters
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mfgarchon.utils.deprecation import NEVER_READ, retired_parameters
from mfgarchon.utils.mfg_logging import get_logger

logger = get_logger(__name__)
# Type definitions for MCMC
PotentialFunction = Callable[[NDArray], float]  # U(x) -> energy
GradientFunction = Callable[[NDArray], NDArray]  # ∇U(x) -> gradient
LogDensityFunction = Callable[[NDArray], float]  # log π(x) -> log density


@dataclass
class MCMCConfig:
    """Configuration for MCMC samplers."""

    # General MCMC parameters
    num_samples: int = 10000
    num_warmup: int = 1000  # Burn-in period
    num_chains: int = NEVER_READ  # refused when passed; never read (#2648)
    thinning: int = 1  # Take every nth sample

    # Step size and adaptation
    step_size: float = 0.1  # Initial step size
    adapt_step_size: bool = True  # Adaptive step size tuning
    target_accept_rate: float = 0.65  # Target acceptance rate

    # HMC specific parameters
    num_leapfrog_steps: int = 10  # Number of leapfrog steps
    mass_matrix: NDArray | None = None  # Mass matrix M for HMC
    max_tree_depth: int = NEVER_READ  # refused when passed; never read (#2648)

    # Convergence diagnostics
    compute_rhat: bool = NEVER_READ  # refused when passed; never read (#2648)
    effective_sample_size: bool = NEVER_READ  # refused when passed; never read (#2648)

    # Computational settings
    seed: int | None = None
    parallel: bool = NEVER_READ  # refused when passed; never read (#2648)

    # Advanced options
    metric_adaptation: bool = True  # Adapt mass matrix
    step_size_adaptation: bool = True  # Dual averaging for step size
    max_adapt_iter: int = NEVER_READ  # refused when passed; never read (#2648)


MCMCConfig.__init__ = retired_parameters(  # type: ignore[method-assign]
    {
        "num_chains": "every sampler runs one chain from its initial_state; run it again for another chain",
        "max_tree_depth": "NoUTurnSampler builds no tree; it is simplified HMC that picks its own number of leapfrog steps",
        "max_adapt_iter": "step-size adaptation runs over the first num_warmup iterations; set num_warmup",
        "parallel": "every sampler runs one chain, in this process",
        "compute_rhat": "the sampler runs one chain; call compute_rhat(chains) on samples from several runs",
        "effective_sample_size": (
            "the sampler computes none; call effective_sample_size(chains) on the samples, shaped "
            "(num_samples, num_chains, dimension)"
        ),
    },
    issue="#2648",
    since="v0.22.0",
    never_read=True,
)(MCMCConfig.__init__)


@dataclass
class MCMCResult:
    """Result container for MCMC sampling."""

    # Samples and diagnostics
    samples: NDArray  # Shape: (num_samples, num_chains, dimension)
    log_densities: NDArray  # Log density at each sample
    acceptance_rate: float  # Overall acceptance rate

    # Convergence diagnostics
    rhat: NDArray | None = None  # R-hat for each parameter
    effective_sample_size: NDArray | None = None  # ESS for each parameter
    converged: bool = False

    # Sampling statistics
    num_samples: int = 0
    num_warmup: int = 0
    num_chains: int = 1
    final_step_size: float = 0.0

    # Performance metrics
    sampling_time: float = 0.0
    samples_per_second: float = 0.0

    # Chain-specific diagnostics
    chain_acceptance_rates: NDArray | None = None
    chain_step_sizes: NDArray | None = None


class MCMCSampler(ABC):
    """Abstract base class for MCMC samplers."""

    def __init__(
        self,
        potential_fn: PotentialFunction,
        gradient_fn: GradientFunction | None = None,
        config: MCMCConfig | None = None,
    ):
        """
        Initialize MCMC sampler.

        Args:
            potential_fn: Potential energy function U(x)
            gradient_fn: Gradient ∇U(x) (computed numerically if None)
            config: MCMC configuration
        """
        self.potential_fn = potential_fn
        self.gradient_fn = gradient_fn or self._numerical_gradient
        self.config = config or MCMCConfig()

        # Initialize random number generator
        self.rng = np.random.RandomState(self.config.seed)

        # State variables
        self.step_size = self.config.step_size
        self.mass_matrix = None
        self.adaptation_info = {}

    def _numerical_gradient(self, x: NDArray, eps: float = 1e-6) -> NDArray:
        """Compute numerical gradient using finite differences."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (self.potential_fn(x_plus) - self.potential_fn(x_minus)) / (2 * eps)
        return grad

    @abstractmethod
    def sample(self, initial_state: NDArray, num_samples: int) -> MCMCResult:
        """
        Generate MCMC samples.

        Args:
            initial_state: Initial state for chain
            num_samples: Number of samples to generate

        Returns:
            MCMCResult with samples and diagnostics
        """

    def _adapt_step_size(self, acceptance_rate: float, iteration: int) -> None:
        """Adapt step size using dual averaging."""
        if not self.config.step_size_adaptation:
            return

        # Dual averaging algorithm (simplified)
        target_rate = self.config.target_accept_rate
        t0 = 10

        eta = 1.0 / (iteration + t0)
        self.step_size *= np.exp(eta * (acceptance_rate - target_rate))

        # Prevent step size from becoming too small or large
        self.step_size = np.clip(self.step_size, 1e-5, 10.0)


class MetropolisHastings(MCMCSampler):
    """Metropolis-Hastings MCMC sampler."""

    def __init__(self, potential_fn: PotentialFunction, proposal_std: float = 1.0, config: MCMCConfig | None = None):
        """
        Initialize Metropolis-Hastings sampler.

        Args:
            potential_fn: Potential energy function
            proposal_std: Standard deviation for proposal distribution
            config: MCMC configuration
        """
        super().__init__(potential_fn, config=config)
        self.proposal_std = proposal_std

    def sample(self, initial_state: NDArray, num_samples: int) -> MCMCResult:
        """Generate samples using Metropolis-Hastings algorithm."""
        dimension = len(initial_state)
        total_samples = num_samples + self.config.num_warmup

        # Storage for samples
        samples = np.zeros((total_samples, dimension))
        log_densities = np.zeros(total_samples)
        accepted = 0

        # Initialize chain
        current_state = initial_state.copy()
        current_energy = self.potential_fn(current_state)
        current_log_density = -current_energy

        import time

        start_time = time.time()

        for i in range(total_samples):
            # Propose new state
            proposal = current_state + self.rng.normal(0, self.proposal_std, dimension)
            proposal_energy = self.potential_fn(proposal)
            proposal_log_density = -proposal_energy

            # Metropolis acceptance criterion
            log_alpha = current_log_density - proposal_log_density
            accept = log_alpha > 0 or self.rng.random() < np.exp(log_alpha)

            if accept:
                current_state = proposal
                current_energy = proposal_energy
                current_log_density = proposal_log_density
                accepted += 1

            # Store sample
            samples[i] = current_state
            log_densities[i] = current_log_density

            # Adapt step size during warmup
            if i < self.config.num_warmup and self.config.adapt_step_size:
                accept_rate = accepted / (i + 1)
                self._adapt_step_size(accept_rate, i)
                # Update proposal standard deviation
                self.proposal_std = self.step_size

        # Remove warmup samples
        final_samples = samples[self.config.num_warmup :]
        final_log_densities = log_densities[self.config.num_warmup :]

        # Apply thinning
        if self.config.thinning > 1:
            indices = np.arange(0, len(final_samples), self.config.thinning)
            final_samples = final_samples[indices]
            final_log_densities = final_log_densities[indices]

        # Create result
        result = MCMCResult(
            samples=final_samples.reshape(len(final_samples), 1, dimension),
            log_densities=final_log_densities,
            acceptance_rate=accepted / total_samples,
            num_samples=len(final_samples),
            num_warmup=self.config.num_warmup,
            final_step_size=self.proposal_std,
            sampling_time=time.time() - start_time,
        )

        result.samples_per_second = result.num_samples / result.sampling_time

        return result


class HamiltonianMonteCarlo(MCMCSampler):
    """Hamiltonian Monte Carlo (HMC) sampler."""

    def __init__(
        self, potential_fn: PotentialFunction, gradient_fn: GradientFunction, config: MCMCConfig | None = None
    ):
        """
        Initialize HMC sampler.

        Args:
            potential_fn: Potential energy function U(q)
            gradient_fn: Gradient function ∇U(q)
            config: MCMC configuration
        """
        super().__init__(potential_fn, gradient_fn, config)

        # Initialize mass matrix
        if self.config.mass_matrix is not None:
            self.mass_matrix = self.config.mass_matrix
        else:
            # Default to identity matrix
            self.mass_matrix = None  # Will be set based on dimension

    def sample(self, initial_state: NDArray, num_samples: int) -> MCMCResult:
        """Generate samples using Hamiltonian Monte Carlo."""
        dimension = len(initial_state)
        total_samples = num_samples + self.config.num_warmup

        # Initialize mass matrix if not provided
        if self.mass_matrix is None:
            self.mass_matrix = np.eye(dimension)

        # Storage for samples
        samples = np.zeros((total_samples, dimension))
        log_densities = np.zeros(total_samples)
        accepted = 0

        # Initialize chain
        current_q = initial_state.copy()
        current_energy = self.potential_fn(current_q)
        current_log_density = -current_energy

        import time

        start_time = time.time()

        for i in range(total_samples):
            # Sample momentum
            current_p = self.rng.multivariate_normal(np.zeros(dimension), self.mass_matrix)

            # Compute current Hamiltonian
            current_kinetic = 0.5 * current_p.T @ np.linalg.solve(self.mass_matrix, current_p)
            current_hamiltonian = current_energy + current_kinetic

            # Leapfrog integration
            q_new, p_new = self._leapfrog_integrate(
                current_q, current_p, self.config.num_leapfrog_steps, self.step_size
            )

            # Compute proposed Hamiltonian
            proposed_energy = self.potential_fn(q_new)
            proposed_kinetic = 0.5 * p_new.T @ np.linalg.solve(self.mass_matrix, p_new)
            proposed_hamiltonian = proposed_energy + proposed_kinetic

            # Metropolis acceptance for Hamiltonian
            delta_h = proposed_hamiltonian - current_hamiltonian
            accept = delta_h < 0 or self.rng.random() < np.exp(-delta_h)

            if accept:
                current_q = q_new
                current_energy = proposed_energy
                current_log_density = -current_energy
                accepted += 1

            # Store sample
            samples[i] = current_q
            log_densities[i] = current_log_density

            # Adapt parameters during warmup
            if i < self.config.num_warmup:
                accept_rate = accepted / (i + 1)
                if self.config.adapt_step_size:
                    self._adapt_step_size(accept_rate, i)

                # Adapt mass matrix (simplified)
                if self.config.metric_adaptation and i > 100 and i % 100 == 0:
                    self._adapt_mass_matrix(samples[: i + 1])

        # Remove warmup and apply thinning
        final_samples = samples[self.config.num_warmup :: self.config.thinning]
        final_log_densities = log_densities[self.config.num_warmup :: self.config.thinning]

        # Create result
        result = MCMCResult(
            samples=final_samples.reshape(len(final_samples), 1, dimension),
            log_densities=final_log_densities,
            acceptance_rate=accepted / total_samples,
            num_samples=len(final_samples),
            num_warmup=self.config.num_warmup,
            final_step_size=self.step_size,
            sampling_time=time.time() - start_time,
        )

        result.samples_per_second = result.num_samples / result.sampling_time

        return result

    def _leapfrog_integrate(self, q: NDArray, p: NDArray, num_steps: int, step_size: float) -> tuple[NDArray, NDArray]:
        """Perform leapfrog integration for Hamiltonian dynamics."""
        q_new = q.copy()
        p_new = p.copy()

        # Half step for momentum
        p_new -= 0.5 * step_size * self.gradient_fn(q_new)

        # Full steps
        for _ in range(num_steps):
            # Full step for position
            q_new += step_size * np.linalg.solve(self.mass_matrix, p_new)

            # Full step for momentum (except last)
            if _ < num_steps - 1:
                p_new -= step_size * self.gradient_fn(q_new)

        # Final half step for momentum
        p_new -= 0.5 * step_size * self.gradient_fn(q_new)

        return q_new, p_new

    def _adapt_mass_matrix(self, samples: NDArray) -> None:
        """Adapt mass matrix based on sample covariance."""
        if len(samples) < 100:
            return

        # Use regularized sample covariance
        sample_cov = np.cov(samples, rowvar=False)
        regularization = 1e-6 * np.eye(sample_cov.shape[0])
        self.mass_matrix = sample_cov + regularization

        logger.debug("Mass matrix adapted based on sample covariance")


class NoUTurnSampler(HamiltonianMonteCarlo):
    """No-U-Turn Sampler (NUTS) - self-tuning HMC variant."""

    def sample(self, initial_state: NDArray, num_samples: int) -> MCMCResult:
        """Generate samples using NUTS algorithm."""
        # Simplified NUTS implementation
        # In practice, would implement the full tree doubling algorithm

        # For now, use adaptive HMC with automatic step number selection
        self.config.num_leapfrog_steps = self._choose_num_steps(initial_state)

        return super().sample(initial_state, num_samples)

    def _choose_num_steps(self, state: NDArray) -> int:
        """Choose number of leapfrog steps adaptively."""
        # Simplified adaptive step selection
        # Real NUTS uses sophisticated tree doubling
        dimension = len(state)
        return max(5, min(50, int(np.sqrt(dimension) * 2)))


class LangevinDynamics(MCMCSampler):
    """Langevin dynamics MCMC for gradient-based sampling."""

    def sample(self, initial_state: NDArray, num_samples: int) -> MCMCResult:
        """Generate samples using Langevin dynamics."""
        dimension = len(initial_state)
        total_samples = num_samples + self.config.num_warmup

        # Storage
        samples = np.zeros((total_samples, dimension))
        log_densities = np.zeros(total_samples)

        # Initialize
        current_state = initial_state.copy()
        current_log_density = -self.potential_fn(current_state)

        import time

        start_time = time.time()

        for i in range(total_samples):
            # Langevin update: x_{t+1} = x_t - ε∇U(x_t) + √(2ε) N(0,I)
            gradient = self.gradient_fn(current_state)
            noise = self.rng.normal(0, np.sqrt(2 * self.step_size), dimension)

            current_state = current_state - self.step_size * gradient + noise
            current_log_density = -self.potential_fn(current_state)

            # Store sample
            samples[i] = current_state
            log_densities[i] = current_log_density

            # Adapt step size
            if i < self.config.num_warmup and self.config.adapt_step_size:
                # For Langevin, we don't have acceptance rate, so use gradient norm
                grad_norm = np.linalg.norm(gradient)
                if grad_norm > 10:  # Large gradients suggest step size too large
                    self.step_size *= 0.99
                elif grad_norm < 1:  # Small gradients suggest we can increase step size
                    self.step_size *= 1.01

        # Process results
        final_samples = samples[self.config.num_warmup :: self.config.thinning]
        final_log_densities = log_densities[self.config.num_warmup :: self.config.thinning]

        result = MCMCResult(
            samples=final_samples.reshape(len(final_samples), 1, dimension),
            log_densities=final_log_densities,
            acceptance_rate=1.0,  # Langevin always accepts
            num_samples=len(final_samples),
            num_warmup=self.config.num_warmup,
            final_step_size=self.step_size,
            sampling_time=time.time() - start_time,
        )

        result.samples_per_second = result.num_samples / result.sampling_time

        return result


def compute_rhat(chains: NDArray) -> NDArray:
    """
    Compute R-hat convergence diagnostic for multiple chains.

    Args:
        chains: Shape (num_samples, num_chains, dimension)

    Returns:
        R-hat values for each dimension
    """
    num_samples, num_chains, dimension = chains.shape

    if num_chains < 2:
        return np.ones(dimension)  # Cannot compute R-hat with single chain

    # Vectorized R-hat computation over all dimensions
    # chains shape: (num_samples, num_chains, dimension)

    # Chain means: mean over samples axis -> (num_chains, dimension)
    chain_means = np.mean(chains, axis=0)

    # Between-chain variance for each dimension
    B = num_samples * np.var(chain_means, axis=0, ddof=1)  # shape: (dimension,)

    # Within-chain variance: variance over samples for each chain/dim
    chain_vars = np.var(chains, axis=0, ddof=1)  # shape: (num_chains, dimension)
    W = np.mean(chain_vars, axis=0)  # shape: (dimension,)

    # Pooled variance estimate
    var_plus = ((num_samples - 1) * W + B) / num_samples  # shape: (dimension,)

    # R-hat with safe division
    rhat = np.where(W > 0, np.sqrt(var_plus / np.maximum(W, 1e-16)), 1.0)

    return rhat


def _compute_chain_ess(chain_data: NDArray, num_samples: int) -> float:
    """Compute ESS for a single chain (helper function)."""
    # Compute autocorrelation (simplified)
    autocorr = np.correlate(chain_data, chain_data, mode="full")
    autocorr = autocorr[autocorr.size // 2 :]
    if autocorr[0] != 0:
        autocorr = autocorr / autocorr[0]
    else:
        return float(num_samples)  # No correlation if variance is zero

    # Find cutoff where autocorrelation becomes negligible
    cutoff = 1
    max_lag = min(len(autocorr), num_samples // 4)
    # Vectorized cutoff search: find first index where autocorr < 0.1
    below_threshold = autocorr[1:max_lag] < 0.1
    if np.any(below_threshold):
        cutoff = np.argmax(below_threshold) + 1
    else:
        cutoff = max_lag

    # Effective sample size for this chain
    autocorr_sum = np.sum(autocorr[1:cutoff])
    return num_samples / max(1 + 2 * autocorr_sum, 1e-16)


def effective_sample_size(chains: NDArray) -> NDArray:
    """
    Compute effective sample size for MCMC chains.

    Args:
        chains: Shape (num_samples, num_chains, dimension)

    Returns:
        Effective sample size for each dimension
    """
    num_samples, num_chains, dimension = chains.shape

    # Reshape to process all chain-dimension pairs at once
    # (num_samples, num_chains, dimension) -> (num_samples, num_chains * dimension)
    flat_chains = chains.reshape(num_samples, -1)
    n_total = flat_chains.shape[1]  # num_chains * dimension

    # Compute ESS for each flattened chain (still need loop for autocorrelation)
    flat_ess = np.array([_compute_chain_ess(flat_chains[:, i], num_samples) for i in range(n_total)])

    # Reshape back to (num_chains, dimension) and sum over chains
    ess_matrix = flat_ess.reshape(num_chains, dimension)
    ess = np.sum(ess_matrix, axis=0)  # Sum ESS over chains for each dimension

    return ess


# Convenience functions for MFG applications
def sample_mfg_posterior(
    log_posterior: LogDensityFunction,
    grad_log_posterior: GradientFunction,
    initial_params: NDArray,
    method: str = "hmc",
    **kwargs,
) -> MCMCResult:
    """
    Sample from MFG model posterior distribution.

    Args:
        log_posterior: Log posterior density function
        grad_log_posterior: Gradient of log posterior
        initial_params: Initial parameter values
        method: "hmc" | "nuts" | "mh" | "langevin"
        **kwargs: Additional arguments for sampler

    Returns:
        MCMC sampling result
    """

    # Convert log posterior to potential (negative log posterior)
    def potential_fn(x):
        return -log_posterior(x)

    def gradient_fn(x):
        return -grad_log_posterior(x)

    config = MCMCConfig(**kwargs)

    if method == "hmc":
        sampler = HamiltonianMonteCarlo(potential_fn, gradient_fn, config)
    elif method == "nuts":
        sampler = NoUTurnSampler(potential_fn, gradient_fn, config)
    elif method == "mh":
        sampler = MetropolisHastings(potential_fn, config=config)
    elif method == "langevin":
        sampler = LangevinDynamics(potential_fn, gradient_fn, config)
    else:
        raise ValueError(f"Unknown MCMC method: {method}")

    return sampler.sample(initial_params, config.num_samples)


def bayesian_neural_network_sampling(
    neural_network: Callable,
    data: tuple[NDArray, NDArray],
    prior_std: float = 1.0,
    likelihood_std: float = 0.1,
    **mcmc_kwargs,
) -> MCMCResult:
    """
    Sample neural network weights using MCMC for Bayesian neural networks.

    Args:
        neural_network: Neural network function(weights, inputs) -> outputs
        data: Tuple of (inputs, targets)
        prior_std: Standard deviation for weight prior
        likelihood_std: Standard deviation for likelihood
        **mcmc_kwargs: Arguments for MCMC sampler

    Returns:
        MCMC result with weight samples
    """
    inputs, targets = data

    def log_posterior(weights):
        # Prior: Gaussian on weights
        log_prior = -0.5 * np.sum(weights**2) / prior_std**2

        # Likelihood: Gaussian noise model
        outputs = neural_network(weights, inputs)
        residuals = targets - outputs
        log_likelihood = -0.5 * np.sum(residuals**2) / likelihood_std**2

        return log_prior + log_likelihood

    def grad_log_posterior(weights):
        # Numerical gradient (in practice, would use automatic differentiation)
        eps = 1e-6
        grad = np.zeros_like(weights)
        for i in range(len(weights)):
            weights_plus = weights.copy()
            weights_minus = weights.copy()
            weights_plus[i] += eps
            weights_minus[i] -= eps
            grad[i] = (log_posterior(weights_plus) - log_posterior(weights_minus)) / (2 * eps)
        return grad

    # Initialize weights
    num_weights = 100  # Would be determined by network architecture
    initial_weights = np.random.normal(0, prior_std, num_weights)

    return sample_mfg_posterior(log_posterior, grad_log_posterior, initial_weights, method="hmc", **mcmc_kwargs)

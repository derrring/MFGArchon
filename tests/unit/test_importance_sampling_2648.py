"""#2648, ruling 8: `monte_carlo_integrate(sampling_method="importance")` samples from the given density.

Before this, "importance" fell through to "Unknown sampling method", logged, and sampled uniformly. The
`ImportanceMCSampler` it should have reached returned 0.00000 for a density matched to the integrand: it filled
missing rejection samples with uniform points and self-normalized 1/q weights, so tail points with 1/q near e^139
took all the weight.

The oracle is f(x) = exp(-x^2 / (2 * 0.3^2)) on [-5, 5], whose integral is exact by erf:
1. With the density matched to f, the estimate is exact: f / p is constant.
2. With wider densities, the mean over seeds is within 3 standard errors of the exact value, and the spread is below
   uniform sampling's over the same seeds.
3. Uniform sampling is unchanged: its estimates at three fixed seeds equal, bit for bit, those measured on main at
   25fd097b before this change. This is the control.
4. A density that is not normalized, a rejection bound below the density, an infeasible acceptance rate, a missing
   importance_function, and an unknown sampling_method are each refused by name.
"""

from __future__ import annotations

import math

import pytest

import numpy as np

from mfgarchon.utils.numerical.particle.sampling import ImportanceMCSampler, MCConfig, monte_carlo_integrate

S_F, L, N, SEEDS = 0.3, 5.0, 2000, 100
DOMAIN = [(-L, L)]
EXACT = S_F * math.sqrt(2 * math.pi) * math.erf(L / (S_F * math.sqrt(2)))


def f(x):
    return np.exp(-np.sum(x**2, axis=1) / (2 * S_F**2))


def density(s):
    """A Gaussian of width s, normalized on the domain."""
    z = s * math.sqrt(2 * math.pi) * math.erf(L / (s * math.sqrt(2)))
    return lambda x: np.exp(-np.sum(x**2, axis=1) / (2 * s**2)) / z


def estimates(**config):
    return np.array(
        [
            monte_carlo_integrate(f, DOMAIN, MCConfig(num_samples=N, seed=seed, **config)).estimate
            for seed in range(SEEDS)
        ]
    )


def test_a_density_matched_to_the_integrand_gives_the_exact_integral():
    for seed in (0, 1, 7):
        result = monte_carlo_integrate(
            f,
            DOMAIN,
            MCConfig(num_samples=N, seed=seed, sampling_method="importance", importance_function=density(S_F)),
        )
        assert result.estimate == pytest.approx(EXACT, rel=1e-12), seed


@pytest.mark.parametrize("width", [0.6, 1.5])
def test_a_wider_density_is_unbiased_and_beats_uniform(width):
    importance = estimates(sampling_method="importance", importance_function=density(width))
    uniform = estimates()
    error = importance.std(ddof=1) / math.sqrt(SEEDS)
    assert abs(importance.mean() - EXACT) < 3 * error, (importance.mean(), EXACT, error)
    assert importance.std(ddof=1) < uniform.std(ddof=1), (importance.std(ddof=1), uniform.std(ddof=1))


def test_uniform_sampling_is_unchanged():
    """Control: measured on main at 25fd097b, before this change."""
    for seed, value in ((0, 0.6982045585893328), (1, 0.7319098980065832), (7, 0.7491218261522968)):
        assert monte_carlo_integrate(f, DOMAIN, MCConfig(num_samples=N, seed=seed)).estimate == value, seed


def test_a_density_that_is_not_normalized_is_refused():
    def unnormalized(x):
        return np.exp(-np.sum(x**2, axis=1) / (2 * S_F**2))

    with pytest.raises(ValueError, match=r"must be a probability density normalized on the domain; its integral is"):
        ImportanceMCSampler(DOMAIN, MCConfig(seed=0), unnormalized)


def test_a_rejection_bound_below_the_density_is_refused_not_biased():
    p = density(S_F)
    sampler = ImportanceMCSampler(DOMAIN, MCConfig(seed=0), p, importance_max=0.5 * float(p(np.zeros((1, 1)))[0]))
    with pytest.raises(ValueError, match=r"exceeds the rejection bound M"):
        sampler.sample(N)


def test_an_infeasible_acceptance_rate_is_refused():
    with pytest.raises(ValueError, match=r"acceptance rate 1/\(V M\)"):
        ImportanceMCSampler(DOMAIN, MCConfig(seed=0), density(1e-5))


def test_importance_without_a_density_is_refused():
    with pytest.raises(ValueError, match=r"needs MCConfig.importance_function"):
        monte_carlo_integrate(f, DOMAIN, MCConfig(sampling_method="importance"))


def test_an_unknown_sampling_method_is_refused_by_name():
    with pytest.raises(ValueError, match=r"unknown sampling_method 'bogus'; choose one of"):
        monte_carlo_integrate(f, DOMAIN, MCConfig(sampling_method="bogus"))

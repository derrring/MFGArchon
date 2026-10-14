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
4. Refused by name: a density that is not normalized (to within max(1e-6, 4 standard errors of the check)), not
   finite or negative; a rejection bound below the density; an infeasible acceptance rate; a missing
   importance_function; an unknown sampling_method; control variates under importance sampling; and an integrand
   that is non-zero where the density vanishes, at a check point.
5. Review 1 of #2664: `integrate_gaussian_quadrature_mc` draws from N(mean, cov) directly and returns the integral
   over all of space, in any dimension; stratified sampling refuses more strata than samples, which used to confine
   the first coordinate to its low strata.
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
    """At construction, against the check points; and at each draw, against the candidates, for a bound the check
    points did not catch (here lowered after construction)."""
    p = density(S_F)
    with pytest.raises(ValueError, match=r"exceeds the rejection bound M"):
        ImportanceMCSampler(DOMAIN, MCConfig(seed=0), p, importance_max=0.5 * float(p(np.zeros((1, 1)))[0]))
    sampler = ImportanceMCSampler(DOMAIN, MCConfig(seed=0), p)
    sampler.importance_max = 0.5 * float(p(np.zeros((1, 1)))[0])
    with pytest.raises(ValueError, match=r"exceeds the rejection bound M"):
        sampler.sample(N)


def test_a_density_off_by_a_few_percent_is_refused():
    """Review 1 of #2664, B2: the standard normal pdf on [-2, 2] integrates to 0.9545 there. With a floor of 5% it
    was accepted and the estimate came back 4.8% high; the window is now max(1e-6, 4 standard errors of the check)."""

    def standard_normal(x):
        return np.exp(-np.sum(x**2, axis=1) / 2) / math.sqrt(2 * math.pi)

    with pytest.raises(ValueError, match=r"must be a probability density normalized on the domain"):
        ImportanceMCSampler([(-2.0, 2.0)], MCConfig(seed=0), standard_normal)


@pytest.mark.parametrize("bad", [np.nan, -1.0])
def test_a_density_that_is_not_finite_or_is_negative_is_refused(bad):
    """Review 1 of #2664, B3 and S2: a NaN would pass both comparisons and loop the rejection; a negative value is
    no density."""
    p = density(S_F)

    def spoiled(x):
        values = p(x)
        values[0] = bad
        return values

    with pytest.raises(ValueError, match=r"must be finite and non-negative on the domain"):
        ImportanceMCSampler(DOMAIN, MCConfig(seed=0), spoiled)


def test_control_variates_with_importance_sampling_are_refused():
    """Review 1 of #2664, B1: the control variate is calibrated on equal weights, so under importance weights the
    estimate was biased (mean 4.12 against 0.752 over 100 seeds). The combination was unreachable before the
    dispatch existed."""
    config = MCConfig(
        sampling_method="importance",
        importance_function=density(0.6),
        use_control_variates=True,
        control_function=lambda x: np.exp(-np.sum(x**2, axis=1) / (2 * 0.5**2)),
    )
    with pytest.raises(ValueError, match=r"use_control_variates does not combine with sampling_method='importance'"):
        monte_carlo_integrate(f, DOMAIN, config)


def test_an_infeasible_acceptance_rate_is_refused():
    with pytest.raises(ValueError, match=r"acceptance rate 1/\(V M\)"):
        ImportanceMCSampler(DOMAIN, MCConfig(seed=0), density(1e-5))


def test_importance_without_a_density_is_refused():
    with pytest.raises(ValueError, match=r"needs MCConfig.importance_function"):
        monte_carlo_integrate(f, DOMAIN, MCConfig(sampling_method="importance"))


def test_an_unknown_sampling_method_is_refused_by_name():
    with pytest.raises(ValueError, match=r"unknown sampling_method 'bogus'; choose one of"):
        monte_carlo_integrate(f, DOMAIN, MCConfig(sampling_method="bogus"))


def test_an_integrand_outside_the_densitys_support_is_refused():
    """Review 1 of #2664, B3: p = 1/2 on [-1, 1] and 0 elsewhere is normalized on [-5, 5], but importance sampling
    never draws where p = 0, so f = 1 gave 2 against 10. The check points where p = 0 are evaluated through f; a
    non-zero value is refused. An integrand vanishing there too is the control: support matched, estimate exact."""

    def half_on_unit(x):
        return np.where(np.abs(x[:, 0]) <= 1.0, 0.5, 0.0)

    config = MCConfig(num_samples=N, seed=0, sampling_method="importance", importance_function=half_on_unit)
    with pytest.raises(ValueError, match=r"the importance density is 0 at x = .*, where the integrand is 1.0"):
        monte_carlo_integrate(lambda x: np.ones(len(x)), DOMAIN, config)
    inside = monte_carlo_integrate(lambda x: np.where(np.abs(x[:, 0]) <= 1.0, 1.0, 0.0), DOMAIN, config)
    assert inside.estimate == pytest.approx(2.0, rel=1e-12)


def test_the_gaussian_integrator_returns_the_integral_over_all_of_space():
    """Review 1 of #2664, B4 and B5: draws from N(mean, cov) directly, in any dimension. With f proportional to the
    normal density the estimate is exact, (2 pi)^(d/2) sqrt(det cov); with f = exp(-|x|^2) it is pi^(3/2) within its
    error. The old rejection from a 6-sigma box failed from d = 6 on."""
    from mfgarchon.utils.numerical.particle.sampling import integrate_gaussian_quadrature_mc

    mean = np.array([0.3, -0.2, 0.5])
    cov = np.array([[1.0, 0.4, 0.1], [0.4, 2.0, -0.3], [0.1, -0.3, 0.5]])
    inverse = np.linalg.inv(cov)

    def proportional(x):
        d = x - mean
        return np.exp(-0.5 * np.einsum("ij,jk,ik->i", d, inverse, d))

    exact = (2 * math.pi) ** 1.5 * math.sqrt(np.linalg.det(cov))
    assert integrate_gaussian_quadrature_mc(mean, cov, proportional, num_samples=500).estimate == pytest.approx(
        exact, rel=1e-10
    )
    result = integrate_gaussian_quadrature_mc(np.zeros(3), np.eye(3), lambda x: np.exp(-np.sum(x**2, axis=1)))
    assert abs(result.estimate - math.pi**1.5) < 6 * result.standard_error
    seven = integrate_gaussian_quadrature_mc(np.zeros(7), np.eye(7), lambda x: np.exp(-0.5 * np.sum(x**2, axis=1)), 200)
    assert seven.estimate == pytest.approx((2 * math.pi) ** 3.5, rel=1e-10)


def test_stratified_sampling_refuses_more_strata_than_samples_and_reports_what_it_drew():
    """Review 1 of #2664, B6: with num_samples < strata**d the first strata in C order were kept, confining x0 to its
    low strata -- at d = 5 with the defaults the mean of x0 was 0.05 against 0.5. Refused now, naming the remedy. When
    num_samples is not a multiple of the strata, the points drawn are what num_samples_used reports."""
    with pytest.raises(ValueError, match=r"10 strata per dimension in d = 5 make 100000 strata, more than the 10000"):
        monte_carlo_integrate(lambda x: x[:, 0], [(0.0, 1.0)] * 5, MCConfig(seed=0, sampling_method="stratified"))
    result = monte_carlo_integrate(
        lambda x: x[:, 0], [(0.0, 1.0)] * 2, MCConfig(num_samples=250, seed=0, sampling_method="stratified")
    )
    assert result.num_samples_used == 200
    assert result.estimate == pytest.approx(0.5, abs=6 * result.standard_error)

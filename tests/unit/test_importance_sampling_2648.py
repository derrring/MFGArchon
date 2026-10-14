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
   with more than 1e-6 of the integral of |f| where the density vanishes, estimated on the check points.
5. Review 1 of #2664: `integrate_gaussian_quadrature_mc` draws from N(mean, cov) directly and returns the integral
   over all of space; stratified sampling refuses more strata than samples, which used to confine the first
   coordinate to its low strata.
6. Review 2 of #2664: the support check weighs mass, so a density whose tail underflows to 0 where f is subnormal is
   not refused; the Gaussian integrator forms f / p in log space, so neither p nor 1 / p overflows on its own; its
   covariance is checked; and the standard error and confidence interval are pinned.
7. Review 3 of #2664: a NaN or infinite f where the density is 0 is refused by name, and an f too large to sum is
   still weighed; the refusal prints the signed value; the integrator's seed fixes its draw; its sign, an f that is 0
   everywhere, and its rescale before averaging are pinned.
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
    never draws where p = 0, so f = 1 gave 2 against 10. The check points where p = 0 are evaluated through f, and the
    fraction of the integral of |f| lying there is refused above 1e-6 -- 0.8 for f = 1 and for f = -1, whose refusal
    prints the signed value (review 3). An integrand vanishing there too is the control: support matched, estimate
    exact."""

    def half_on_unit(x):
        return np.where(np.abs(x[:, 0]) <= 1.0, 0.5, 0.0)

    config = MCConfig(num_samples=N, seed=0, sampling_method="importance", importance_function=half_on_unit)
    with pytest.raises(
        ValueError, match=r"an estimated 0\.8\d* of the integral of \|f\| lies where the importance density is 0"
    ):
        monte_carlo_integrate(lambda x: np.ones(len(x)), DOMAIN, config)
    with pytest.raises(ValueError, match=r"of the integral of \|f\| lies where .* the integrand is -1\.0\)"):
        monte_carlo_integrate(lambda x: -np.ones(len(x)), DOMAIN, config)
    inside = monte_carlo_integrate(lambda x: np.where(np.abs(x[:, 0]) <= 1.0, 1.0, 0.0), DOMAIN, config)
    assert inside.estimate == pytest.approx(2.0, rel=1e-12)


def test_the_support_check_weighs_mass_not_a_single_point():
    """Review 2 of #2664, B-1. A missing fraction of the integral of |f| above 1e-6 is refused; one below it is not,
    nor is a matched density whose tail underflows to p = 0 where f is still subnormal (5e-324 at x = 38.6), which the
    witness rule refused. p = 2 on [0, 0.5] and f = 1 there, f = delta on (0.5, 1]: the missing fraction is about
    delta."""

    def left_half(x):
        return np.where(x[:, 0] <= 0.5, 2.0, 0.0)

    def tail(delta):
        return lambda x: np.where(x[:, 0] <= 0.5, 1.0, delta)

    config = MCConfig(num_samples=N, seed=0, sampling_method="importance", importance_function=left_half)
    with pytest.raises(ValueError, match=r"of the integral of \|f\| lies where the importance density is 0") as refused:
        monte_carlo_integrate(tail(1e-5), [(0.0, 1.0)], config)
    fraction = float(str(refused.value).split("an estimated ")[1].split(" ")[0])
    assert 0.5e-5 < fraction < 2e-5, fraction
    assert monte_carlo_integrate(tail(1e-7), [(0.0, 1.0)], config).estimate == pytest.approx(0.5, rel=1e-12)
    # The threshold is NORMALIZATION_TOL, 1e-6: 1.5e-6 refused, 7e-7 not (review 3, S3-1).
    with pytest.raises(ValueError, match=r"of the integral of \|f\| lies where the importance density is 0"):
        monte_carlo_integrate(tail(1.5e-6), [(0.0, 1.0)], config)
    assert monte_carlo_integrate(tail(7e-7), [(0.0, 1.0)], config).estimate == pytest.approx(0.5, rel=1e-12)
    # The part seen is the integral of |f| too: f = -1 where p > 0 and -1e-7 beyond misses 1e-7 of it, not more.
    negative = monte_carlo_integrate(lambda x: np.where(x[:, 0] <= 0.5, -1.0, -1e-7), [(0.0, 1.0)], config)
    assert negative.estimate == pytest.approx(-0.5, rel=1e-12)
    for half_width in (40.0, 50.0):
        z = math.erf(half_width / math.sqrt(2))
        matched = MCConfig(
            num_samples=N,
            seed=0,
            sampling_method="importance",
            importance_function=lambda x, z=z: np.exp(-np.sum(x**2, axis=1) / 2) / (math.sqrt(2 * math.pi) * z),
        )
        result = monte_carlo_integrate(
            lambda x: np.exp(-np.sum(x**2, axis=1) / 2), [(-half_width, half_width)], matched
        )
        assert result.estimate == pytest.approx(math.sqrt(2 * math.pi) * z, rel=1e-12), half_width


def test_a_non_finite_or_overflowing_integrand_where_the_density_vanishes_is_refused():
    """Review 3 of #2664, B3-1: the mass criterion compared NaN and inf as False, and summed 1e304 into inf, so all
    three were accepted where the witness rule had refused them. A non-finite value is refused by name; a large one is
    weighed relative to the largest |f|, which leaves the fraction, 0.8 here, unchanged."""

    def half_on_unit(x):
        return np.where(np.abs(x[:, 0]) <= 1.0, 0.5, 0.0)

    config = MCConfig(num_samples=N, seed=0, sampling_method="importance", importance_function=half_on_unit)
    for bad, shown in ((np.nan, "nan"), (np.inf, "inf")):
        with pytest.raises(ValueError, match=rf"the integrand is {shown} at x = .*, where the importance density is 0"):
            monte_carlo_integrate(lambda x, bad=bad: np.where(np.abs(x[:, 0]) <= 1.0, 1.0, bad), DOMAIN, config)
    with pytest.raises(ValueError, match=r"an estimated 0\.8\d* of the integral of \|f\|"):
        monte_carlo_integrate(lambda x: np.full(len(x), 1e304), DOMAIN, config)


def test_the_normalization_window_is_its_floor_or_four_check_errors():
    """Review 2 of #2664, S-2. A constant density 1e-5 too heavy has a check error of 0, so only the 1e-6 floor can
    refuse it; 2.04 x on [0, 1] is 2% too heavy, 10 check errors, and only a window of 4 errors refuses it."""
    with pytest.raises(ValueError, match=r"must be a probability density normalized on the domain"):
        ImportanceMCSampler([(0.0, 1.0)], MCConfig(seed=0), lambda x: np.full(len(x), 1.0 + 1e-5))
    with pytest.raises(ValueError, match=r"must be a probability density normalized on the domain"):
        ImportanceMCSampler([(0.0, 1.0)], MCConfig(seed=0), lambda x: 2.04 * x[:, 0])


def test_the_standard_error_and_confidence_interval_are_those_of_the_estimate():
    """Review 2 of #2664, S-2: f = x on [0, 2], sampled uniformly. Each point contributes 2 x, whose standard
    deviation is 2 / sqrt(3), so the standard error is that over sqrt(N); the 95% interval is 1.96 of them either
    side."""
    n = 10000
    result = monte_carlo_integrate(lambda x: x[:, 0], [(0.0, 2.0)], MCConfig(num_samples=n, seed=0))
    assert result.standard_error == pytest.approx(2 / math.sqrt(3) / math.sqrt(n), rel=0.03)
    assert result.confidence_interval == pytest.approx(
        (result.estimate - 1.96 * result.standard_error, result.estimate + 1.96 * result.standard_error), rel=1e-14
    )


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
    result = integrate_gaussian_quadrature_mc(np.zeros(3), np.eye(3), lambda x: np.exp(-np.sum(x**2, axis=1)), seed=0)
    assert abs(result.estimate - math.pi**1.5) < 6 * result.standard_error
    negative = integrate_gaussian_quadrature_mc(
        np.zeros(3), np.eye(3), lambda x: -np.exp(-np.sum(x**2, axis=1)), seed=0
    )
    assert negative.estimate == -result.estimate  # the sign survives the log-space form (review 3)
    zero = integrate_gaussian_quadrature_mc(np.zeros(3), np.eye(3), lambda x: np.zeros(len(x)), seed=0)
    assert zero.estimate == 0.0
    assert zero.standard_error == 0.0
    seven = integrate_gaussian_quadrature_mc(np.zeros(7), np.eye(7), lambda x: np.exp(-0.5 * np.sum(x**2, axis=1)), 200)
    assert seven.estimate == pytest.approx((2 * math.pi) ** 3.5, rel=1e-10)


def test_the_gaussian_integrator_overflows_only_where_the_largest_f_over_p_does():
    """Review 2 of #2664, MF-3: with f = exp(-|x|^2 / (2 s^2)) and cov = s^2 I, f / p is (2 pi s^2)^(d/2) exactly:
    10^279.8 at d = 200, s = 10, a finite float, which mean(f * exp(-log p)) returned as inf because 1 / p overflows
    alone. Review 3: f / p = 1e306 at every point is finite too, but 2000 of them summed are not, so the contributions
    are rescaled by the largest before they are averaged."""
    from mfgarchon.utils.numerical.particle.sampling import integrate_gaussian_quadrature_mc

    d, s = 200, 10.0
    result = integrate_gaussian_quadrature_mc(
        np.zeros(d), s**2 * np.eye(d), lambda x: np.exp(-np.sum(x**2, axis=1) / (2 * s**2)), num_samples=50, seed=0
    )
    assert result.estimate == pytest.approx(math.exp(d / 2 * math.log(2 * math.pi * s**2)), rel=1e-10)
    c = 1e306 / math.sqrt(2 * math.pi)
    large = integrate_gaussian_quadrature_mc(
        np.zeros(1), np.eye(1), lambda x: c * np.exp(-np.sum(x**2, axis=1) / 2), num_samples=2000, seed=0
    )
    assert large.estimate == pytest.approx(1e306, rel=1e-12)


def test_the_gaussian_integrators_seed_fixes_its_draw():
    """Review 3 of #2664, B3-2: the seed reaches the draw -- the same seed twice gives the same estimate, another seed
    a different one."""
    from mfgarchon.utils.numerical.particle.sampling import integrate_gaussian_quadrature_mc

    def run(seed):
        return integrate_gaussian_quadrature_mc(
            np.zeros(2), np.eye(2), lambda x: np.exp(-np.sum(x**2, axis=1)), num_samples=500, seed=seed
        ).estimate

    assert run(0) == run(0)
    assert run(0) != run(1)


@pytest.mark.parametrize(
    ("cov", "message"),
    [
        (np.array([[1.0, 2.0], [2.0, 1.0]]), r"cov must be positive definite"),
        (np.array([[1.0, 0.5], [0.0, 1.0]]), r"cov must be a symmetric 2x2 matrix"),
    ],
    ids=["indefinite", "asymmetric"],
)
def test_the_gaussian_integrator_refuses_a_covariance_it_cannot_factor_by_name(cov, message):
    """Review 2 of #2664, S-4: an indefinite covariance raised numpy's bare LinAlgError, and an asymmetric one was
    read through its lower triangle without a word."""
    from mfgarchon.utils.numerical.particle.sampling import integrate_gaussian_quadrature_mc

    with pytest.raises(ValueError, match=message):
        integrate_gaussian_quadrature_mc(np.zeros(2), cov, lambda x: np.ones(len(x)), seed=0)


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

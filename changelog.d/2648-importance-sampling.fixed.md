- **`monte_carlo_integrate(sampling_method="importance")` now samples from the density it is given** (#2648, ruling
  8). Before, "importance" fell through to "Unknown sampling method", was logged, and was sampled uniformly.
  - **`ImportanceMCSampler` is fixed, and its contract is now stated.** `importance_function` is a density on the
    domain D: finite, non-negative, positive wherever the integrand is non-zero, with integral Z over D equal to 1.
    The estimate's expectation is the integral of f over D divided by Z. Because of the fallthrough, only a direct
    construction of the public `particle.ImportanceMCSampler` reached the old sampler. It filled missing rejection
    samples with uniform points and self-normalized 1/q weights. For a density matched to the integrand, the
    estimate `monte_carlo_integrate` forms from its points and weights was 0.00000, against an exact 0.751988
    (f = exp(-x^2 / (2 * 0.3^2)) on [-5, 5], N = 2000, 200 seeds).
  - **Refused by name; the old sampler accepted each:**
    - a density whose integral over the domain differs from 1 by more than max(1e-6, 4 standard errors of a
      100,000-point check). A check cannot certify Z more closely than its own error, so that is the window the
      estimate may still carry. The check draws uniformly, so for a sharply peaked density the window is wide: about
      21% for a normal of width 0.01 on [-5, 5];
    - a density that is not finite, or is negative;
    - an integrand with more than 1e-6 of the integral of |f| where the density is 0, which importance sampling never
      draws, or one that is NaN or infinite there. The fraction is estimated on the check points, so a miss between
      them is not seen.
  - **Refused by name where nothing was wrong before, because the dispatch or the parameter is new:**
    - `use_control_variates` with `sampling_method="importance"`, whose calibration assumes equal weights. Before,
      "importance" was sampled uniformly, and control variates leave a uniform estimate unbiased;
    - a rejection bound `importance_max`, a new parameter, below the density;
    - any other unknown `sampling_method`, listing the valid ones. Before, it was sampled uniformly, which is a valid
      estimate of the integral.
  - **`integrate_gaussian_quadrature_mc` now returns the integral of the integrand over all of R^d,** by importance
    sampling from N(mean, cov) drawn directly. It is not the Gaussian expectation E_N[f], and its error is finite
    only when the integral of f^2 / p is.
    - Before, "importance" fell through to uniform sampling of a 6-sigma box. Over 50 seeds, its integral of
      exp(-|x|^2) in 2-D (cov 0.5 I) had mean 3.1465 and standard deviation 0.144.
    - f / p is formed in log space, so neither p nor 1 / p overflows on its own in high dimension.
    - A covariance that is not symmetric positive definite is refused by name.
    - A new `seed` argument fixes the draw.
  - **Stratified sampling refuses more strata than samples.** It kept the first `num_samples` strata in C order, so
    at d = 5 with the defaults the mean of x0 came back 0.05 against 0.5. `MCResult.num_samples_used` is the number of
    points drawn, which is fewer than `num_samples` when that is not a multiple of the strata.

- **`monte_carlo_integrate(sampling_method="importance")` now samples from the density it is given** (#2648, ruling
  8). Before, "importance" fell through to "Unknown sampling method", was logged, and was sampled uniformly.
  - **`ImportanceMCSampler` is fixed, and its contract is now stated.** `importance_function` is a probability
    density normalized on the domain, and the estimate is the unbiased mean(f / p). The old sampler filled missing
    rejection samples with uniform points and self-normalized 1/q weights. For a density matched to the integrand it
    returned 0.00000 against an exact 0.751988 (f = exp(-x^2 / (2 * 0.3^2)) on [-5, 5], 200 seeds).
  - **A direct caller passing an unnormalized function is now refused, where before it got that garbage.** So is a
    rejection bound `importance_max` below the density, and a density too peaked for rejection from the domain.
  - **Any other unknown `sampling_method` is refused by name,** listing the valid ones: `"uniform"`,
    `"stratified"`, `"sobol"`, `"halton"`, `"latin_hypercube"`, `"importance"`.
  - **`integrate_gaussian_quadrature_mc` normalizes its Gaussian.** Over 50 seeds, its integral of exp(-|x|^2) in
    2-D (cov 0.5 I) moves from 3.1465 +- 0.020, which was uniform sampling, to 3.141593 with a spread of 4e-16. The
    remaining 3.9e-9 relative is the mass outside its 6-sigma box.

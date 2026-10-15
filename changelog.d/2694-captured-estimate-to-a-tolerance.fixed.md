- **A test compared captured Monte Carlo estimates bit for bit, and the nightly went red on another runner** (#2694).
  `test_uniform_sampling_is_unchanged` compared three uniform-sampling estimates with `==` against values captured on
  one machine; the nightly's runner gave seed 1's estimate 2 ulps away. It now compares to a relative 1e-12. A test
  change only; the library is unchanged.

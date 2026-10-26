- **`kde_factor` and `kde_kernel_covariance`** (`mfgarchon.utils.numerical.particle.interpolation`, #2556): the owner
  of a KDE bandwidth's meaning (CONVENTIONS.md § 9), beside `estimate_kde_bandwidth`, which states scipy's rules.
  `kde_factor` reads "scott" / "silverman" through it and passes a positive finite number through, refusing anything
  else by name. `kde_kernel_covariance` gives scipy's kernel, factor² × the sample covariance, and refuses a sample
  with no spread by name.

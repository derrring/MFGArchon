- **`kde_factor`, `kde_kernel_covariance`, `kde_kernel_widths` and `kde_sample_has_spread`**
  (`mfgarchon.utils.numerical.particle.interpolation`, #2556): the owner of a KDE bandwidth's meaning (CONVENTIONS.md
  § 9), beside `estimate_kde_bandwidth`, which states scipy's rules.
  `kde_factor` reads "scott" / "silverman" through it and passes a positive finite number through, refusing anything
  else by name. `kde_kernel_covariance` gives scipy's kernel, factor² × the sample covariance, and refuses by name a
  sample with no spread in some direction, which `kde_sample_has_spread` decides at the precision the numbers carry.
  `kde_kernel_widths` gives the kernel's per-axis widths, which a sample on a line or a plane has, and refuses a sample
  at one point.

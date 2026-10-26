- **`density_estimation.adaptive_bandwidth_selection` and `adaptive_bandwidth_selection_nd`** (#2556): they returned an
  absolute width, with a "silverman" that is not scipy's (0.9·min(σ, IQR/1.34)·n^(-1/5) in 1-D, and that with
  n^(-1/(d+4)) and the mean over axes in d dimensions), and their readers took it
  as a factor in the projector and in `estimate_density_kde`. Use `kde_kernel_covariance` (or `kde_factor`) from
  `mfgarchon.utils.numerical.particle.interpolation`.
- **`fp_solvers.fp_particle_density.estimate_density_kde`** and its private helpers (#2556): no caller outside its
  module, and its 1-D path scaled a rule's width twice. Use `reflection_kde` or `gaussian_kde_gpu`.

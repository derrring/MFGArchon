- **The library's own KDE estimators read a bandwidth as the convention says** (#2556, C3-1): a factor of the sample's
  spread, or scipy's rule, with scipy's kernel. Each of these behaves differently now:
  - `reflection_kde`:
    - A number is a factor of the sample's own spread. It was a factor of the spread of the sample with its ghost
      particles, and its ghost zone took the number as an absolute width: 0.153 of scipy's density at a sample's
      centre, for 0.5.
    - The estimator adds the sample's kernel at each image of the evaluation point across a nearby wall, so on a
      correlated sample its slope normal to a wall vanishes on the wall.
    - A per-axis array is refused.
  - `renormalization_kde` reads "scott" and "silverman" (it raised `TypeError`), and its renormalization uses the
    kernel's per-axis widths, where it took the number as an absolute width.
  - `gaussian_kde_gpu` reads "scott" and "silverman" (`TypeError` before).
  - `gaussian_kde_gpu_nd` uses scipy's full kernel covariance, where it used an isotropic factor × mean std.
  - The particle-to-grid projector no longer scales a rule's width twice in 2-D and up, and refuses an unknown rule
    where its 1-D branch used a factor of 1.
  - `FPParticleSolver`:
    - Its GPU path computes the kernel width from each step's sample. It used the initial sample's at every step, and
      returned NaN throughout from an initial sample with no spread, which it now refuses by name.
    - `kde_method="beta"` reads the rules (`TypeError` before).
  - `FPParticleResult(bandwidth=)` is a factor of each timestep's spread, as the solver's `kde_bandwidth` is; its query
    took the number as an absolute width.
  - A sample with no spread is refused by name by every estimator above, where some raised scipy's `LinAlgError` and
    some returned NaN.

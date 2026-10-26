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
    - `kde_method="beta"` reads the rules (`TypeError` before), and a number as a factor of the sample's spread,
      where `beta_kde` was handed it as an absolute width.
    - Its n-D density reader deposits a sample with no spread in some direction (on a line or a plane, as particles
      placed there with no diffusion stay) on its nearest nodes, each particle carrying mass 1/N in the problem's
      measure. It returned a KDE of the sample with its ghost particles, or raised scipy's `LinAlgError`, depending on
      the line and the bandwidth.
  - `FPParticleResult(bandwidth=)` is a factor of each timestep's spread, or a rule, as the solver's `kde_bandwidth` is,
    and the solver hands it either; its query took a number as an absolute width, and the solver handed it no rule, so
    the query's own Scott rule took its place. A query at a timestep whose sample is at one point refuses by name: it
    returned NaN for a rule, and read a number as the width.
  - Called directly, every estimator above refuses by name a sample with no spread in some direction
    (`kde_sample_has_spread`). Before, some raised scipy's `LinAlgError`, some returned NaN, and `reflection_kde`
    returned a density from the spread its ghost particles added: for a 1-D point with a number, and for a 2-D
    sample on a line or a plane.

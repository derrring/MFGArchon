- **`FictitiousPlayIterator` runs fictitious play by default, and stops on the map's residual
  (#2415).** Fictitious play best-responds to the unfloored $1/(n+1)$ average of past densities.
  That is Cardaliaguet–Hadikhanloo (2017): (2.2) with Theorem 2.1 in the second-order case, and
  (3.3) with Theorem 3.1 in the first-order case. Their convergence results are for that
  procedure.

  - **The default `min_learning_rate` is now `0.0`, not `0.01`.** From the sweep where
    $1/(k+1) < 0.01$, the old default was constant damping, with no such guarantee. An explicit
    floor is still honoured. The default also no longer floors the `"sqrt"`, `"polynomial"` and
    custom schedules.
  - **Convergence is measured on the map's output against its input**: `M_candidate` against
    `M_old`, and in hybrid mode (`damp_value_function=True`) `U_new` against `U_old`. It was measured
    on the averaged step, which is exactly $\alpha_k$ times the map difference, so a decaying
    $\alpha_k$ shrank it whether or not the belief was near a fixed point. `FixedPointIterator`
    already measures this way (#1684 item 7).

  What changes for a user, measured at `0f937601`:

  - **Which is faster depends on the problem.** On #1914's $\sigma = 0$ fixture, after 300 sweeps
    the map residual $\lVert \Phi(M) - M \rVert / \lVert M \rVert$ was $2.45 \times 10^{-1}$ with the
    0.01 floor and $4.67 \times 10^{-4}$ without it. On the second-order fixture of
    `test_harmonic_is_fictitious_play_2415.py`, where damped Picard contracts, after 400 sweeps it
    was $3.75 \times 10^{-6}$ with the floor and $1.67 \times 10^{-5}$ without it, so there
    fictitious play is the slower of the two. A floor can still be passed explicitly.
  - **`metadata["l2distm_rel"]` and `error_history_M`, and in hybrid mode `l2distu_*` and
    `error_history_U`, now report the map residual.** The same run reports larger errors and stops
    later, or not at all. On #1914's fixture with the floor removed, the old measure reported
    `converged=True` at sweep 359 while the map residual was $3.57 \times 10^{-4}$, 357 times the
    default tolerance. A stochastic FP solver whose residual floors at its sampling noise no longer
    reports convergence below that noise.

  **`FixedPointIterator` is unchanged, and its floor is now documented.** It clamps the
  `"harmonic"`, `"sqrt"` and `"exponential"` schedules at `adaptive_relaxation_min` (default 0.05)
  whether or not `adaptive_relaxation` is on. `PicardConfig.relaxation_schedule` now says so. With
  that floor, harmonic is damped Picard, not fictitious play, and the floor is doing work. On the
  fixture above it converges in 175 sweeps at `relaxation=1.0` and in 248 at the default 0.5. With
  `adaptive_relaxation_min=0.0` it had not converged after 400.

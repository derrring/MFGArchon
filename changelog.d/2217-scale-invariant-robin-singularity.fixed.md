`ghost_cell_robin`'s singularity guard is now scale-invariant. It compared `|alpha/2 + beta/dx|` —
a quantity carrying units — against an absolute `1e-12`, so the same physical boundary condition was
refused or answered depending on the grid spacing and on however the caller happened to scale its
coefficient pair. Scaling `(alpha, beta)` by any non-zero constant leaves the Robin condition, and
therefore the ghost, unchanged; it did not leave the verdict unchanged. Measured over 144
coefficient families built to straddle the cancellation point, the verdict flipped under rescaling
in **96**; it now flips in **0**.

The threshold is `max(|alpha|/2, |beta|/dx)`, so it fires on genuine cancellation rather than on
both terms merely being small. Two alternatives were measured and rejected: `|alpha|/2 + |beta|/dx`
is equally scale-invariant but strictly more conservative for no gain, and normalising the pair by
`max(|alpha|, |beta|)` — proposed in #2217 as the cleanest — is **not** scale-invariant, because it
does not account for `dx`; it still flipped 4 of the 144. The comparison is `<=` rather than `<`, so
the wholly degenerate pair `alpha = beta = 0`, where the scale is itself 0, still raises.

**One behaviour change, and it is the point.** Against the previous code over 720 FP-wall inputs,
exactly **12** verdicts move and all are the same case: `v_n = 0` at `D = 1e-13`, every `dx`, both
wall signs, cell-centred, where the old code **raised**. `v_n = 0` is homogeneous Neumann, the ghost
is exactly the interior value, and refusing it was the well-posed problem #2217 reports being
rejected through the public `ghost_cell_advection_diffusion_no_flux`. All 12 are refusals that
became answers; none is an answer that became a refusal. Generic coefficients are untouched — at
`alpha, beta ~ 1` and `dx ~ 0.01–0.1` neither threshold fires.

**`ghost_cell_fp_no_flux` no longer scales its pair by `2*dx`.** That scaling existed only to make
robin's absolute threshold mean this function's own pre-#2128 predicate, so #2217 removes its
reason for existing; it is now provably inert (0 verdict differences across scales `1.0 / 2*dx /
4*dx / 17.3` over 720 inputs) and removing it is this fix's own tail. The cost is rounding only:
276 of 1792 inputs differ, worst **5 ulp** and 9.3e-16 relative, 0 verdict changes. Two failure
modes documented under the old scaling go with the multiplication that caused them — the overflow
to a silent `nan` at `dx = D = 1e200`, and the denormal breakdown at `D = 1e-300, dx = 1e-20`.

**A test that pinned the defect was replaced, not deleted.**
`test_the_scales_magnitude_is_pinned_from_both_sides` asserted that `v_n = 0, D = 3.75e-13, dx = 1`
must **raise** — a well-posed homogeneous-Neumann wall. It was written in round 6 of #2216 to pin
the `2*dx` workaround, and pinning the workaround pinned the defect underneath it. It is now
`test_the_singularity_verdict_is_invariant_under_rescaling_the_pair` plus
`test_homogeneous_neumann_is_answered_not_refused`. Both are mutation-verified: restoring the
absolute threshold fails them. The invariance test's vertex arm is **inert** against that mutant —
that branch never computes `coeff_ghost` — and its docstring says so rather than letting it read as
coverage.

Note for anyone reading #2217's acceptance test literally: its "same ghost" half holds only away
from cancellation. In the cancellation band the ghost is `something / tiny` and rescaling perturbs
it by ~7e-6 relative — conditioning, not a defect. The "same verdict" half is the one that must
hold everywhere, and does. (#2217)

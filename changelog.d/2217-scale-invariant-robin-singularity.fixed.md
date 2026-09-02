`ghost_cell_robin`'s singularity guard is now scale-invariant. It compared `|alpha/2 + beta/dx|` —
a quantity carrying units — against an absolute `1e-12`, so the same physical boundary condition was
refused or answered depending on the grid spacing and on however the caller happened to scale its
coefficient pair. The scale is not a property of the wall — `coeff_ghost` depends on it and the
physical condition does not — so it must not move the verdict. It did. Measured over 144
coefficient families built to straddle the cancellation point, the verdict flipped under rescaling
in **96**; it now flips in **0**.

The threshold is `max(|alpha|/2, |beta|/dx)`, so it fires on genuine cancellation rather than on
both terms merely being small. Two alternatives were measured and rejected. `|alpha|/2 + |beta|/dx`
is equally invariant and strictly more conservative for no gain. Normalising the pair by
`max(|alpha|, |beta|)` — #2217's own preferred option — is **equally scale-invariant**, both being
positively homogeneous of degree 1; its defect is **dimensional**, in that its effective tolerance
drifts with `dx`: measured 2.0e-12 at `dx = 0.01` and `0.1`, then 1.0e-11 at `dx = 10` and 1.0e-10
at `dx = 100`, where the chosen scale holds 1.0e-12 throughout. The comparison is `<=` rather than
`<`, so the wholly degenerate pair `alpha = beta = 0`, where the scale is itself 0, still raises.

**The refusal set moves in both directions, and the counts are grid-dependent.** The dominant
direction is refusal → answer, headed by `v_n = 0`: homogeneous Neumann, ghost exactly the interior
value, refused whenever `2D < 1e-12`, which is the well-posed problem #2217 reports being rejected
through the public `ghost_cell_advection_diffusion_no_flux`. A 720-input grid shows 12 such changes;
a 1944-input grid shows 106, spanning `v_n = 0` and `v_n = ±1e-13`.

The **other** direction exists and has a sharp entry condition. The old effective threshold on
`coeff_ghost` was `1e-12/(2·dx)`; the new one at cancellation is `1e-12·(D/dx)`, so inputs the old
code answered and this one refuses occupy a band that is non-empty exactly when **`D > 1/2`**.
Measured, sweeping `v_n` through cancellation at three spacings: 0 hits at `D = 0.25` and `0.4`,
then 28 at `D = 0.6`, 134 at `D = 1.0`, 266 at `D = 10`. Not regressions — `2D` and `v_n·dx` agree
to ~13 significant figures there and the old answers were cancellation garbage (`-2.0e12` at one
such point) — but a widening, recorded rather than left to be found later.

**The two directions are mirrors of the same number, so neither is "dominant".** Near cancellation
the old code refuses below `D = 1/2` and the new code refuses above it, at every `dx`. Which
direction a sweep reports is decided by which side of `1/2` its `D` values sample: varying only `D`
on one construction, `D ∈ {0.6, 1, 2, 10}` gives 232 answer→refusal and 0 the other way, while
`D ∈ {0.05, 0.1, 0.25, 0.4}` gives 0 and 1230. An earlier version of this fragment gave the
answer→refusal side an exact criterion and left the other side an anecdote about `v_n = 0`.

**`ghost_cell_robin`'s own refusal set moves both ways too**, for the three modules that call it
directly rather than through the FP wall, two of which pass a non-zero `g`. The new/old threshold
ratio is `S = max(|alpha|/2, |beta|/dx)` — wider when `S > 1`, narrower when `S < 1`. Measured at
`g = 0.7`: `alpha=-200, beta=1, dx=0.01` went ok → raise; `alpha=-0.02, beta=0.01, dx=1` went
raise → ok. Exact cancellation still refuses under both. Generic coefficients are
untouched: at `alpha, beta ~ 1` and `dx ~ 0.01–0.1` neither threshold fires.

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

Note for anyone reading #2217's acceptance test literally: its **"same ghost" half is not a
property of the scheme at all**, for two separate reasons, and only the second is about
cancellation. First, scaling `alpha` and `beta` by `s` turns `alpha*u + beta*du/dn = g` into
`... = g/s`, so the wall is unchanged only at `g = 0`; with `g = 0.7, alpha = 2, beta = 0.5,
dx = 0.1` the ghost moves 99% across three decades of scale, nowhere near cancellation. The FP wall
always passes `g = 0`, which is why the invariance test can assert a value at all. Second, inside
the cancellation band the ghost is `something / tiny` and rescaling perturbs it by 1.6e-5 relative at
`r = 1e-11` — conditioning, not a defect. The **"same verdict"** half is the one that holds
everywhere, and is what this fix delivers. (#2217)

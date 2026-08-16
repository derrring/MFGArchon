`particle_action_for_bc_type` no longer takes `alpha`. It required the coefficient and never
read it: the Robin branch dispatches on `beta` alone, so `ROBIN(alpha=0, beta=1)` and
`ROBIN(alpha=9, beta=1)` went through the same `return`.

The docstring described three Robin cases as though they were three branches. There are two,
and the vocabulary is why. A genuinely mixed Robin — both coefficients nonzero — is a
**partially absorbing** wall: the particle counterpart of `alpha*m + beta*d_n m = g` is a
diffusion that reflects with a probability set by `alpha/beta`, Feller's elastic boundary.
`ParticleAction` has three members and none of them is that, so the function returns
`"reflecting"`, which conserves mass and is the conservative reading of a wall it cannot
express. That is now stated where it can be read.

`test_robin_dispatches_on_its_coefficients` had four rows, two of which passed through the same
`return` — it looked like it discriminated the pure-flux case from the mixed one and did not.
Re-pointed to the two cases that exist.

`alpha` comes back the day `ParticleAction` grows a partially-absorbing member, because that
member's probability is exactly what `alpha/beta` sets.

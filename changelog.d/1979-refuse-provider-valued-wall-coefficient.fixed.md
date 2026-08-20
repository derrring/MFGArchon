A provider-valued wall coefficient no longer disappears (#1979). `BCValueProvider` exists so a wall
coefficient can be recomputed each Picard iterate — that is what `AdjointConsistentProvider` and
#1970's `NormalDriftProvider` are for, and both live on `alpha`, not on `value`. Two consumers took
one and did the wrong thing with it.

**FDM**: `_BOUNDARY_HANDLERS` is keyed on the advection scheme and its handlers take no
`boundary_conditions` argument at all, so nothing read `alpha` and the segment assembled
byte-identically to a plain no-flux wall, with no diagnostic — the wall a user wired a coefficient
to *avoid*, returned as though it were their request. `solve_fp_nd_full_system` now refuses at entry,
naming the offending `segment.field`, the segment types involved, and `with_resolved_providers` as
the remedy.

**FEM**: `assemble_robin_terms` coerced with `float()`, giving a bare builtin `TypeError` — unnamed,
ungreppable, silent about the remedy — three lines above a `NotImplementedError` guard that already
did this correctly for `g`. `alpha` and `beta` now get the same guard, with the same wording.

**Correction to the issue's premise, which changes what the fix is for.** #1979 was filed against a
ROBIN segment, and that route is not reachable: every grid FP solver refuses `ROBIN` at construction
(`_validate_bc_support`, #1456, stated in `conditions.py:1149`), measured — `FPFDMSolver` raises
before any assembly. The reachable case is a provider on the `alpha` of a **NO_FLUX** or **NEUMANN**
segment, which passes the capability gate. That is precisely what `NormalDriftProvider` produces,
since an impermeable wall *is* Robin in `m` (`alpha*m - D*d_n m = 0`) and its coefficient lives
there. So the defect is not narrower than filed; it sits on the path the provider layer was built
for.

Refusing is the fix, not reading. The conservative grid schemes already impose `J·n = 0`
structurally, so teaching those handlers to add `(alpha, beta, g)` would count the drift twice —
measured there at −79.5% mass against −7.4e-15.

Three corrections from review, each of which changed the fix rather than its wording:

- **The guard was not where the hazard is.** It sat in `solve_fp_nd_full_system`, while
  `_BOUNDARY_HANDLERS` is dispatched from `solve_timestep_full_nd` — so calling that directly walked
  straight past it. That is the same shape the guard exists to fix, one layer up. It is now a single
  owner called from both entry points, pinned by a test that fails when either call is removed.
- **`value` was uncovered.** #1686 records that every FP solver silently drops the value in
  `neumann_bc(value=g)`. A guard on `alpha`/`beta` alone would refuse the coefficient and keep
  dropping the datum, so `value` is checked with them.
- **The message prescribed its own defeat.** It offered `bc.with_resolved_providers(state)` as *the*
  remedy. A provider exists to be recomputed each Picard iterate; resolving it freezes it at one
  state, so that advice converts the feature into its absence and calls it a fix. The message now
  names `FPFEMSolver` — whose weak form implements a general Robin wall and reads `alpha`/`beta`/`g`,
  established when #1975 was closed — as the path that can actually do this, and labels resolving as
  a downgrade. The pointer to #1975 as future work is dropped: it is closed COMPLETED, and its
  finding is that the FEM path already does this.

Pinned in `tests/unit/test_geometry/test_provider_robin_coefficient_refused_1979.py`, both paths,
each `raises` paired with a float-coefficient control so that a blanket refusal cannot pass. Both
guards verified mutation-red: removing the FDM one fails 2 of 7, removing the FEM one fails 3 of 7.

Unblocks #1970, which is in draft pending this.
